//! Traffic path network and AI road vehicles (original units `mc_path`, `mc_pathrule`,
//! the AI part of `mc_roadvehicle`).
//!
//! Lanes come from `[path]` entries of splines (running along the spline at a lateral
//! offset) and of scenery objects (arcs in the object's frame). Lane ends are linked by
//! proximity, which reproduces the original's spline `prev`/`next` chains and the object
//! `[splinehelper]` connectors without needing their ids.

use glam::{DVec2, DVec3};
use hashbrown::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LaneKind {
    Street,
    Sidewalk,
    Rail,
    /// `[path]` type 3: flight paths of AI aircraft.
    Air,
}

impl LaneKind {
    pub fn from_code(c: i32) -> LaneKind {
        match c {
            1 => LaneKind::Sidewalk,
            2 => LaneKind::Rail,
            3 => LaneKind::Air,
            _ => LaneKind::Street,
        }
    }
}

/// Identity of a lane in map terms: the tile, the spline/object id and the `[path]` index.
/// Timetable tracks reference lanes this way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct LaneKey {
    pub tile: (i32, i32),
    pub id: i64,
    pub path: u16,
}

/// A sampled lane: points with headings, in world coordinates.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Lane {
    pub key: Option<LaneKey>,
    /// True for the backwards lane of a two-way path.
    pub reversed: bool,
    /// [ROLLBACK pedoneway-1] The path is one-way (direction 0 or 1): a lit pavement stub
    /// or a crossing piece is walked its way only, so the stub is the only way onto its
    /// crossing (a street lane is driven its way in any case).
    pub one_way: bool,
    pub kind: LaneKind,
    pub width: f32,
    /// World positions along the lane (in travel direction).
    pub points: Vec<DVec3>,
    /// Heading (degrees, clockwise from north) at each point: the tangent of the spline or
    /// arc the lane was sampled from, not the direction of the chord to the next point.
    pub headings: Vec<f32>,
    /// Signed curvature at each point (1/m, positive = turning right).
    pub curvature: Vec<f32>,
    /// Cumulative distance at each point.
    pub dist: Vec<f32>,
    pub speed_limit_kmh: f32,
    /// Lanes reachable from the end of this one.
    pub next: Vec<usize>,
    /// Traffic light controlling entry into this lane: (object instance, light index).
    pub traffic_light: Option<(usize, usize)>,
    /// Turn indicator for AI: 0 none, 1 left, 2 right.
    pub turn: i32,
    /// Source for debugging.
    pub source: u32,
    /// Lateral offset of the spline path (m, positive = right of the spline direction).
    pub offset: f32,
    /// Spline/object file the lane came from (debugging).
    pub name: String,
    /// `[rule] trafficdensity`: how much of the road traffic uses this lane (0 = none), of
    /// any random traffic group.
    pub density: f32,
    /// `[rule] trafficdensity <value> <group>` of the path, the last per group: the group
    /// is the random traffic group's place in the map's `unsched_vehgroups.txt`. A group
    /// without one takes its default there (Berlin-Spandau's GDR cars only drive where
    /// the Falkensee paths ask for them).
    pub group_density: Vec<(u16, f32)>,
    /// `[rule] no_cars`: cars keep off this lane.
    pub no_cars: bool,
    /// `[rule] bus` / `[rule] trucks` on the path: it is open to the AI vehicles of
    /// `[ai_veh_type]` 2 / 3 (see [`Lane::allows`]). The rules are switches: the value
    /// after them is not read (Grundorf's 110 `trucks` rules all say 0).
    pub rule_bus: bool,
    pub rule_trucks: bool,
    /// [ROLLBACK unsched-93] the path is marked "no unscheduled traffic": no random AI car
    /// is put on it or drives over it, and no pedestrian is spawned on it.
    pub no_unsched: bool,
    /// The spline this lane belongs to is editor-only: OMSI's invisible service roads at
    /// the edge of a map, where AI traffic drives with no road drawn under it.
    pub invisible: bool,
    /// Parallel lanes of the same road in the same direction (for lane changes).
    pub left: Option<usize>,
    pub right: Option<usize>,
    /// `[rule] priority` (`TPRIPriority`): who goes first where two paths of a junction
    /// meet. The stock maps put 192 on the straight paths of the main road and 64 on the
    /// paths coming out of a side road and leave the rest alone, so an unmarked path sits
    /// in between at `DEFAULT_PRIORITY`.
    pub priority: f32,
    /// Other `[path]`s of its object this lane blocks while taken (`[blockpath]`), beyond
    /// the places where they cross.
    pub blocks: Vec<u16>,
    /// `[rule] overtaking_prohib`: the solid line. No lane change of the driver's own out of
    /// or into this lane, and no passing on the oncoming lane beside it (a timetable route
    /// still moves over where it must).
    pub no_overtaking: bool,
    /// How far the drawn road goes on left and right of the lane's axis (m) before a kerb,
    /// a pavement, the asphalt's end or the next street lane's painted edge, from the
    /// spline's `[profile]` cross-section (`omsi_scenery::profile::lane_drivable`): what a
    /// car may use to squeeze past something or to ease off a walker. Zero (a recording
    /// made before it, a junction object's piece with no edge data): the lane's half width
    /// either side, see [`Lane::drivable`].
    #[serde(default)]
    pub drivable: (f32, f32),
    /// [round 8 H8-7, fix pass 1] The lane ends at the edge of the loaded tiles: the road goes on
    /// into a tile of the map that is not loaded (set by the game, `Traffic::mark_loaded_edges`).
    /// A vehicle stops before its end (`plan`), as at the loaded end of a bus route, instead of
    /// driving off the network: it is not the map's end.
    #[serde(default)]
    pub loaded_edge: bool,
}

/// [round 9 A4 fix pass 2] How far along the road (m) `Network::ends_at_loaded_edge` looks for the
/// edge of the loaded tiles: SUMO's `lcStrategicLookahead` default, how far a vehicle's best lanes
/// look along its way (sumo.md; omsi-traffic's `LC_STRATEGIC_LOOKAHEAD`).
pub const EDGE_LOOKAHEAD: f64 = 3000.0;

/// Priority of a path without a `[rule] priority`.
pub const DEFAULT_PRIORITY: f32 = 128.0;

/// At most this much between the painted edges of two lanes that are next to each other (m).
pub const NEIGHBOUR_EDGE_GAP: f32 = 2.0;

/// How much of `unsched_vehgroups.txt` group `pool`'s traffic a path carries: its `[rule]
/// trafficdensity` for the group (`rules`, see `Lane::group_density`: the last rule of a
/// group counts), else the group's default there (`defaults`): 0 none, for the first group 1
/// its medium density, for any other k that of the k-th group on the same path.
pub fn pool_density(rules: &[(u16, f32)], defaults: &[i32], pool: usize) -> f32 {
    let mut u = pool;
    // (a default naming another group that names this one again would go round for ever)
    for _ in 0..=defaults.len() {
        if let Some(&(_, v)) = rules.iter().rev().find(|(k, _)| *k as usize == u) {
            return v;
        }
        match defaults.get(u).copied().unwrap_or(0) {
            d if d <= 0 => return 0.0,
            _ if u == 0 => return 1.0,
            d => u = d as usize - 1,
        }
    }
    0.0
}

impl Lane {
    /// The drivable extent (left, right) of the lane's axis in metres ([`Lane::drivable`]
    /// as stored, else the lane's half width either side).
    pub fn drivable(&self) -> (f32, f32) {
        if self.drivable.0 > 0.0 && self.drivable.1 > 0.0 {
            self.drivable
        } else {
            (self.width * 0.5, self.width * 0.5)
        }
    }

    /// Are two same-direction paths of one spline next to each other, so that a vehicle
    /// may change between them? By their edges, not by their middles: ordinary lanes are
    /// 3 to 4 m apart middle to middle with their painted widths 0.6 to 1.8 m apart at the
    /// edges, while a cycle lane beyond a green strip lies 5 to 8 m out with 4 m or more of
    /// strip between the edges (BeijingCity). The strip itself is only drawn, by a spline
    /// without paths, so the edge gap is the one cue the path data gives.
    pub fn beside_each_other(a: &Lane, b: &Lane) -> bool {
        let gap = (b.offset - a.offset).abs();
        let edge = gap - (a.width.max(2.0) + b.width.max(2.0)) * 0.5;
        gap > 1.0 && edge <= NEIGHBOUR_EDGE_GAP
    }

    /// May an AI vehicle of `[ai_veh_type]` `veh_type` drive here (Omsi.exe 0x71d714, by
    /// the path's rules): a car (0) where there is no `no_cars`, a taxi (1) also where
    /// `bus` or `trucks` opens a no_cars path, a bus (2) only where `bus` and a truck (3)
    /// only where `trucks` is set. Other values (and the timetable's buses, -1) anywhere.
    pub fn allows(&self, veh_type: i32) -> bool {
        match veh_type {
            0 => !self.no_cars,
            1 => !self.no_cars || self.rule_bus || self.rule_trucks,
            2 => self.rule_bus,
            3 => self.rule_trucks,
            _ => true,
        }
    }

    /// How much of `unsched_vehgroups.txt` group `pool`'s traffic the lane carries (see
    /// [`pool_density`]).
    pub fn pool_density(&self, defaults: &[i32], pool: usize) -> f32 {
        pool_density(&self.group_density, defaults, pool)
    }

    /// Closest point of the lane's polyline to `p`: (distance along the lane, distance to it).
    pub fn nearest_point(&self, p: DVec3) -> Option<(f32, f64)> {
        let mut best: Option<(f32, f64)> = None;
        for k in 0..self.points.len().saturating_sub(1) {
            let a = self.points[k];
            let b = self.points[k + 1];
            let ab = b - a;
            let t = ((p - a).dot(ab) / ab.length_squared().max(1e-9)).clamp(0.0, 1.0);
            let d = (a + ab * t - p).length();
            if best.map(|b| d < b.1).unwrap_or(true) {
                best = Some((self.dist[k] + (self.dist[k + 1] - self.dist[k]) * t as f32, d));
            }
        }
        best
    }

    pub fn length(&self) -> f32 {
        *self.dist.last().unwrap_or(&0.0)
    }

    /// Measure the lane again after its points were moved (an object's tilt).
    pub fn refresh(&mut self) {
        self.dist = cumulative(&self.points);
    }

    /// Segment and fraction along it at distance `s` (clamped to the lane).
    fn locate(&self, s: f32) -> (usize, f32) {
        // (a position gone NaN takes the lane's start rather than stopping the game)
        let s = if s.is_nan() { 0.0 } else { s.clamp(0.0, self.length()) };
        let i = match self.dist.binary_search_by(|d| d.total_cmp(&s)) {
            Ok(i) => i.min(self.points.len() - 2),
            Err(i) => i.saturating_sub(1).min(self.points.len() - 2),
        };
        let (d0, d1) = (self.dist[i], self.dist[i + 1]);
        (i, if d1 > d0 { (s - d0) / (d1 - d0) } else { 0.0 })
    }

    /// Position and heading at distance `s`. Between two samples the lane is a cubic
    /// Hermite curve whose tangents are the sampled headings, so a bend stays round
    /// instead of becoming a chain of chords: a car following the chords turned in small
    /// jerks at every sample and at every lane joint.
    pub fn at(&self, s: f32) -> (DVec3, f32) {
        self.at_with(s, &mut Tangents::default())
    }

    /// `at`, the tangents of the segment kept in `cache` for the next point.
    fn at_with(&self, s: f32, cache: &mut Tangents) -> (DVec3, f32) {
        if self.points.len() < 2 {
            return (self.points.first().copied().unwrap_or(DVec3::ZERO), self.headings.first().copied().unwrap_or(0.0));
        }
        let (i, t) = self.locate(s);
        let (h0, h1) = (self.headings[i], self.headings[i + 1]);
        (self.hermite(i, t as f64, cache), h0 + wrap_deg(h1 - h0) * t)
    }

    /// The heading of `at`, without the point.
    fn heading_at(&self, s: f32) -> f32 {
        if self.points.len() < 2 {
            return self.headings.first().copied().unwrap_or(0.0);
        }
        let (i, t) = self.locate(s);
        let (h0, h1) = (self.headings[i], self.headings[i + 1]);
        h0 + wrap_deg(h1 - h0) * t
    }

    fn hermite(&self, i: usize, t: f64, cache: &mut Tangents) -> DVec3 {
        let (a, b) = (self.points[i], self.points[i + 1]);
        let lin = a.lerp(b, t);
        let chord = (b - a).truncate();
        let len = chord.length();
        if len < 1e-6 {
            return lin;
        }
        let key = (self as *const Lane as usize, i);
        let (d0, d1) = match cache.0 {
            Some((k, d0, d1)) if k == key => (d0, d1),
            _ => {
                let d = (heading_dir(self.headings[i] as f64), heading_dir(self.headings[i + 1] as f64));
                cache.0 = Some((key, d.0, d.1));
                d
            }
        };
        // tangents far off the chord (a kink in hand-made samples) would make the curve loop
        let c = chord / len;
        if d0.dot(c) < 0.8 || d1.dot(c) < 0.8 {
            return lin;
        }
        let (t2, t3) = (t * t, t * t * t);
        let xy = a.truncate() * (2.0 * t3 - 3.0 * t2 + 1.0) + d0 * len * (t3 - 2.0 * t2 + t) + b.truncate() * (3.0 * t2 - 2.0 * t3) + d1 * len * (t3 - t2);
        DVec3::new(xy.x, xy.y, lin.z)
    }

    /// The heading of `at_ext_with`, without the point.
    fn heading_at_ext(&self, s: f32) -> f32 {
        if s < 0.0 {
            self.start_heading()
        } else if s > self.length() {
            self.end_heading()
        } else {
            self.heading_at(s)
        }
    }

    fn at_ext_with(&self, s: f32, cache: &mut Tangents) -> (DVec3, f32) {
        if s < 0.0 {
            let h = self.start_heading();
            let d = heading_dir(h as f64) * s as f64;
            (self.start() + DVec3::new(d.x, d.y, 0.0), h)
        } else if s > self.length() {
            let h = self.end_heading();
            let d = heading_dir(h as f64) * (s - self.length()) as f64;
            (self.end() + DVec3::new(d.x, d.y, 0.0), h)
        } else {
            self.at_with(s, cache)
        }
    }

    /// Curvature at `s` (1/m, positive = right).
    pub fn curvature_at(&self, s: f32) -> f32 {
        if self.points.len() < 2 || self.curvature.len() != self.points.len() {
            return 0.0;
        }
        let (i, t) = self.locate(s);
        self.curvature[i] + (self.curvature[i + 1] - self.curvature[i]) * t
    }

    pub fn start(&self) -> DVec3 {
        self.points[0]
    }
    pub fn end(&self) -> DVec3 {
        *self.points.last().unwrap()
    }
    pub fn start_heading(&self) -> f32 {
        self.headings[0]
    }
    pub fn end_heading(&self) -> f32 {
        *self.headings.last().unwrap()
    }
}

/// Speed limit of a flight path without a `[rule] speedlimit` (km/h): none, the aircraft
/// flies at its own speed.
pub const AIR_NO_LIMIT_KMH: f32 = 1000.0;

/// Builds lanes from analytic descriptions.
pub struct LaneBuilder;

impl LaneBuilder {
    /// Sample an arc/straight lane: start position, heading (deg), length, radius (0 = straight,
    /// > 0 right turn), height change over the length.
    pub fn arc(start: DVec3, heading_deg: f64, length: f64, radius: f64, dz: f64, kind: LaneKind, width: f32) -> Lane {
        let n = ((length / 2.0).ceil() as usize).clamp(1, 400);
        let mut points = Vec::with_capacity(n + 1);
        let mut headings = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let s = length * i as f64 / n as f64;
            let (p, h) = arc_point(start, heading_deg, s, radius);
            points.push(DVec3::new(p.x, p.y, start.z + dz * s / length.max(1e-6)));
            headings.push(h as f32);
        }
        let k = if radius.abs() < 1e-6 { 0.0 } else { (1.0 / radius) as f32 };
        Self::curve(points, headings, vec![k; n + 1], kind, width)
    }

    /// Lane from sampled points with their tangent headings and curvatures.
    /// A flight path has no speed limit of its own unless the map gives it one with a
    /// `[rule] speedlimit`; the street default of 50 km/h had the Tegel approach flown at
    /// walking pace for an airliner.
    pub fn curve(points: Vec<DVec3>, headings: Vec<f32>, curvature: Vec<f32>, kind: LaneKind, width: f32) -> Lane {
        let dist = cumulative(&points);
        let speed_limit_kmh = if kind == LaneKind::Air { AIR_NO_LIMIT_KMH } else { 50.0 };
        Lane { key: None, reversed: false, one_way: false, kind, width, points, headings, curvature, dist, speed_limit_kmh, next: Vec::new(), traffic_light: None, turn: 0, source: 0, offset: 0.0, name: String::new(), invisible: false, density: 1.0, group_density: Vec::new(), no_cars: false, rule_bus: false, rule_trucks: false, no_unsched: false, left: None, right: None, priority: DEFAULT_PRIORITY, blocks: Vec::new(), no_overtaking: false, drivable: (width * 0.5, width * 0.5), loaded_edge: false }
    }

    /// Lane from bare points: headings from the neighbouring points on both sides (the
    /// tangent there, not the chord ahead), curvature from how the heading changes.
    pub fn polyline(points: Vec<DVec3>, kind: LaneKind, width: f32) -> Lane {
        let n = points.len();
        let mut headings = Vec::with_capacity(n);
        for i in 0..n {
            let (a, b) = match (i.checked_sub(1), (i + 1 < n).then_some(i + 1)) {
                (Some(p), Some(q)) => (points[p], points[q]),
                (None, Some(q)) => (points[i], points[q]),
                (Some(p), None) => (points[p], points[i]),
                (None, None) => (points[i], points[i] + DVec3::Y),
            };
            let d = b - a;
            headings.push((d.x.atan2(d.y)).to_degrees() as f32);
        }
        let dist = cumulative(&points);
        let curvature = (0..n)
            .map(|i| {
                let (p, q) = (i.saturating_sub(1), (i + 1).min(n.saturating_sub(1)));
                let ds = dist.get(q).copied().unwrap_or(0.0) - dist.get(p).copied().unwrap_or(0.0);
                if ds > 1e-3 { wrap_deg(headings[q] - headings[p]).to_radians() / ds } else { 0.0 }
            })
            .collect();
        Self::curve(points, headings, curvature, kind, width)
    }
}

/// Cumulative distance along a sequence of points.
fn cumulative(points: &[DVec3]) -> Vec<f32> {
    let mut acc = 0.0f32;
    points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if i > 0 {
                acc += (*p - points[i - 1]).length() as f32;
            }
            acc
        })
        .collect()
}

/// An angle difference in degrees brought into -180..180.
pub fn wrap_deg(d: f32) -> f32 {
    (d + 180.0).rem_euclid(360.0) - 180.0
}

/// Unit vector (east, north) of a heading in degrees.
fn heading_dir(h: f64) -> DVec2 {
    let r = h.to_radians();
    DVec2::new(r.sin(), r.cos())
}

/// Point and heading after travelling `s` metres on an arc starting at `start`.
pub fn arc_point(start: DVec3, heading_deg: f64, s: f64, radius: f64) -> (DVec2, f64) {
    let h = heading_deg.to_radians();
    let d = DVec2::new(h.sin(), h.cos());
    if radius.abs() < 1e-6 {
        return (start.truncate() + d * s, heading_deg);
    }
    let r = radius.abs();
    let turn = radius.signum();
    let right = DVec2::new(d.y, -d.x);
    let centre = start.truncate() + right * r * turn;
    let ang = -turn * s / r;
    let p = start.truncate() - centre;
    let (sa, ca) = ang.sin_cos();
    (centre + DVec2::new(p.x * ca - p.y * sa, p.x * sa + p.y * ca), heading_deg + turn * (s / r).to_degrees())
}

/// The whole network.
#[derive(Default)]
pub struct Network {
    pub lanes: Vec<Lane>,
    /// Lanes by map identity (both directions of a two-way path share a key).
    pub by_key: HashMap<LaneKey, Vec<usize>>,
    /// Per lane: the lanes it has a meeting place with (`crossings` without the places).
    pub conflicts: Vec<Vec<usize>>,
    /// Per lane: the lanes that lead into it (the reverse of `next`), for walking a
    /// one-way path backwards.
    pub prev: Vec<Vec<usize>>,
    /// Lanes by 50 m grid cell (every cell a lane's points touch), so that a nearest-lane
    /// query looks at a handful of lanes instead of all 17 000 of Spandau.
    pub grid: HashMap<(i32, i32), Vec<usize>>,
    /// Lanes by the cell containing their start. Population only needs lane starts near
    /// a viewer; the geometry grid above can contain the same long lane in many cells.
    pub start_grid: HashMap<(i32, i32), Vec<usize>>,
    /// Per street or rail lane: where another lane meets it - the street and rail lanes of its
    /// junction object that cross it or run into its end (`conflicts_from`), lane joins and the
    /// lanes converging before them (`compile_joins`), swept paths (`compile_swept`) and the lanes
    /// crossing it or running beside it outside any one object (`compile_lane_crossings`,
    /// `compile_outside_converging`).
    pub crossings: Vec<Vec<Crossing>>,
    /// Per street lane: the footpaths of its junction object that cross it (zebra and
    /// signalled crossings): (footpath lane, distance along the street lane, along the path).
    pub walks: Vec<Vec<(usize, f32, f32)>>,
    /// Per lane: how far a vehicle can drive from its start before the network ends (m, at
    /// most `REACH_MAX`; lanes closed to cars do not count as a way on).
    pub reach: Vec<f32>,
    /// The map drives on the left (`global.cfg` `[lht]`): priority to the left, the
    /// oncoming lane on the right, turning right across the oncoming traffic.
    pub left_hand: bool,
}

/// Lit stubs: an approach lane is not linked past its short lit stub into the lane after
/// it, and two pieces of one way (one leading into the other, or two short stubs side by
/// side at one place) are no merge of each other (`Network::apply_stub_links`). A stub that
/// carries a light is at most this long (m).
pub const STUB_MAX: f32 = 1.5;
/// The lane after a stub starts within this distance of the stub's end (m)...
const STUB_JOINT: f64 = 0.5;
/// ... heading as the stub ends, within this many degrees.
const STUB_HEADING: f32 = 20.0;

/// `Network::reach` is counted up to this far (m).
pub const REACH_MAX: f32 = 600.0;
/// A way on that ends within this distance is a dead end to a driver who has a choice (m).
pub const DEAD_END: f32 = 500.0;

/// Right of way between paths of equal `[rule] priority` (`Network::right_of_way`): a path
/// whose approach carries at most this `[rule] trafficdensity` is a way out of a yard, a car
/// park, a garage or a bus bay.
pub const YARD_DENSITY: f32 = 0.01;
/// ... more heading change than this (degrees) over the approach is a turn.
pub const TURN_DEG: f32 = 30.0;
/// ... at a merge, a way bending this much more than the other (degrees) merges into it.
pub const MERGE_DEG: f32 = 15.0;
/// ... the approach is the last this many metres of the way to the meeting place (its turn);
pub const APPROACH_WINDOW: f32 = 20.0;
/// ... and the road it comes off (a yard, the road class) the first spline lane within this
/// many metres back (case05's side road: its yard spline lies 34 m before the meeting place).
pub const SPLINE_WINDOW: f32 = 40.0;
/// ... a road whose density is this many times another's is the major one ...
pub const CLASS_DENSITY_RATIO: f32 = 3.0;
/// ... and, between spline approaches, one with this many times the speed limit.
pub const CLASS_SPEED_RATIO: f32 = 1.5;

/// How a vehicle comes to a meeting place (`Network::approach_to`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Approach {
    /// The turn flag of the lane at the meeting place.
    pub turn: i32,
    /// Degrees between the heading at the meeting place and `APPROACH_WINDOW` metres back.
    pub heading_change: f32,
    /// The heading `APPROACH_WINDOW` metres back (degrees).
    pub entry_heading: f32,
    /// The first spline lane (source 1) of the approach, the road it comes off: None when the
    /// approach runs on junction pieces only.
    pub spline: Option<SplineClass>,
}

/// The road an approach comes off (`Approach::spline`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplineClass {
    /// Lanes of the road in its direction (`Network::road_class`).
    pub lanes: u32,
    /// The mean `[rule] trafficdensity` and speed limit of those lanes.
    pub density: f32,
    pub speed_kmh: f32,
}

/// Where two lanes of a junction meet.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Crossing {
    pub other: usize,
    /// Distance along this lane and along the other one to the meeting point.
    pub at: f32,
    pub other_at: f32,
    /// Both lanes end in the same place (one road runs into the other) rather than crossing.
    pub merge: bool,
    /// The meeting place: how far before and after `at` this lane's centre line stays
    /// closer than `MEET_DIST` to the other's (where two vehicles on them would touch),
    /// and the same along the other lane. Two paths that cross at a shallow angle, or two
    /// turns that bend towards each other, meet over metres, not at a point.
    pub before: f32,
    pub after: f32,
    pub other_before: f32,
    pub other_after: f32,
    /// A merge of two lanes running side by side closer than a car is wide, that neither
    /// cross nor run into one another (`Network::add_converging`): the stretch is where they
    /// run together, not a joint.
    #[serde(default)]
    pub side: bool,
    /// A crossing of two vehicle lanes that no one crossing object compiles
    /// (`Network::compile_lane_crossings`): `Junctions::extend` puts the two into one junction.
    #[serde(default)]
    pub outside: bool,
}

/// Two vehicles whose centre lines come closer than this (m) touch: the half widths of two
/// buses.
pub const MEET_DIST: f64 = 2.6;
/// Two vehicle lanes crossing with their centre lines at most this far apart in height (m) cross
/// at one level (`Network::lane_crossings`); farther, one passes over the other on a bridge. The
/// traffic's same-level band for a vehicle near the player (the app's player box and
/// `perception`'s player test, 4 m: a bus under a bridge held up the traffic on it, #753); a road
/// under a deck lies at least its 4.5 m clearance below.
pub const LEVEL_DZ: f64 = 4.0;
/// A meeting place reaches at most this far either side of the crossing point (m).
const MEET_MAX: f32 = 14.0;

/// How far before and after distance `at` of lane `a` its centre line stays within
/// `MEET_DIST` of lane `b`'s (in half-metre steps, at least the half metre around the point).
fn meeting_extent(a: &Lane, at: f32, b: &Lane) -> (f32, f32) {
    let near = |s: f32| b.nearest_point(a.at(s).0).map(|(_, d)| d < MEET_DIST).unwrap_or(false);
    let mut before = 0.5f32;
    while before < MEET_MAX && at - before >= 0.0 && near(at - before) {
        before += 0.5;
    }
    let mut after = 0.5f32;
    while after < MEET_MAX && at + after <= a.length() && near(at + after) {
        after += 0.5;
    }
    (before, after)
}

/// Two lanes of one object that neither cross nor run into one another but whose centre lines
/// come this close (m) over a stretch meet there: two cars on them touch (`converging_stretch`).
/// (Lanes of a two-lane road lie 2.3-2.6 m apart: they stay apart.)
pub const CONVERGE_DIST: f64 = MEET_DIST - 0.6;
/// ... or whose edges overlap by this much (m), given the two lanes' widths ...
pub const CONVERGE_EDGE: f64 = 0.3;
/// ... running the same way there (degrees between their headings at most).
pub const CONVERGE_HEADING: f32 = 60.0;

/// Where lanes `a` and `b` run side by side closer than `MEET_DIST`: the stretch along `a`
/// (from, to), the one along `b`, and the least distance between the centre lines - when they
/// come within `CONVERGE_DIST`, or their edges overlap by `CONVERGE_EDGE`, somewhere on it.
/// Sampled every half metre of each lane. (Two lanes of one crossing object that run together
/// before they part, or converge without ending in one place: case07's 5229/5231, 3769/3770,
/// 1430/1179; the network compiled no meeting place for them and cars met side by side.)
pub fn converging_stretch(a: &Lane, b: &Lane) -> Option<((f32, f32), (f32, f32), f64)> {
    if a.points.len() < 2 || b.points.len() < 2 {
        return None;
    }
    let stretch = |x: &Lane, y: &Lane| -> Option<(f32, f32, f64)> {
        let (mut s0, mut s1, mut least) = (f32::INFINITY, f32::NEG_INFINITY, f64::INFINITY);
        let len = x.length();
        let mut s = 0.0f32;
        loop {
            let at = s.min(len);
            if let Some((_, d)) = y.nearest_point(x.at(at).0) {
                least = least.min(d);
                if d < MEET_DIST {
                    s0 = s0.min(at);
                    s1 = s1.max(at);
                }
            }
            if at >= len {
                break;
            }
            s += 0.5;
        }
        s0.is_finite().then_some((s0, s1, least))
    };
    let (a0, a1, least_a) = stretch(a, b)?;
    let (b0, b1, least_b) = stretch(b, a)?;
    let least = least_a.min(least_b);
    let edges = (a.width as f64 + b.width as f64) * 0.5 - least;
    (least < CONVERGE_DIST || edges >= CONVERGE_EDGE).then_some(((a0, a1), (b0, b1), least))
}

/// Where two polylines cross: the distance along each (ends excluded).
fn polyline_crossing(a: &Lane, b: &Lane) -> Option<(f32, f32)> {
    for (i, pa) in a.points.windows(2).enumerate() {
        for (j, pb) in b.points.windows(2).enumerate() {
            let (p, r) = (pa[0].truncate(), (pa[1] - pa[0]).truncate());
            let (q, s) = (pb[0].truncate(), (pb[1] - pb[0]).truncate());
            let den = r.perp_dot(s);
            if den.abs() < 1e-9 {
                continue;
            }
            let t = (q - p).perp_dot(s) / den;
            let u = (q - p).perp_dot(r) / den;
            if !(0.0..=1.0).contains(&t) || !(0.0..=1.0).contains(&u) {
                continue;
            }
            let sa = a.dist[i] + (a.dist[i + 1] - a.dist[i]) * t as f32;
            let sb = b.dist[j] + (b.dist[j + 1] - b.dist[j]) * u as f32;
            // meeting at an end is a joint (or a fork), not a crossing
            let inner = |s: f32, l: &Lane| s > 0.3 && s < l.length() - 0.3;
            if inner(sa, a) && inner(sb, b) {
                return Some((sa, sb));
            }
        }
    }
    None
}

/// Grid cell size of `Network::grid` (m).
pub const GRID_CELL: f64 = 50.0;

impl Network {
    /// Does the road from lane `n` on end at the edge of the loaded tiles: followed
    /// along every way on, it reaches only lanes with `loaded_edge` and no way on (at a fork,
    /// when every branch does: a road splitting into parallel lanes that all end at the
    /// edge). The map's own end, a loop, or any branch leading elsewhere is not that edge.
    pub fn ends_at_loaded_edge(&self, n: usize) -> bool {
        // (each lane walked once: a lane met again gives its answer, one still being walked - a
        // loop - is not the edge)
        // ... within `EDGE_LOOKAHEAD`: farther on it is not taken for the edge
        fn walk(net: &Network, l: usize, from: f64, memo: &mut std::collections::HashMap<usize, Option<bool>>) -> bool {
            let Some(lane) = net.lanes.get(l) else { return false };
            if lane.next.is_empty() {
                return lane.loaded_edge;
            }
            if from > EDGE_LOOKAHEAD {
                return false;
            }
            match memo.get(&l) {
                Some(Some(r)) => return *r,
                Some(None) => return false,
                None => {}
            }
            memo.insert(l, None);
            let to = from + lane.length() as f64;
            let r = lane.next.iter().all(|&m| walk(net, m, to, memo));
            memo.insert(l, Some(r));
            r
        }
        walk(self, n, 0.0, &mut std::collections::HashMap::new())
    }
}

impl Network {
    /// The sign of a lateral offset (positive to the right) towards the oncoming lane:
    /// -1 on the right-hand side of the road, +1 on a left-hand-traffic map.
    pub fn oncoming_sign(&self) -> f32 {
        if self.left_hand {
            1.0
        } else {
            -1.0
        }
    }

    /// Lane with the given key, preferring the direction flag.
    pub fn find(&self, key: LaneKey, reversed: Option<bool>) -> Option<usize> {
        let list = self.by_key.get(&key)?;
        match reversed {
            Some(r) => list.iter().copied().find(|&i| self.lanes[i].reversed == r).or_else(|| list.first().copied()),
            None => list.first().copied(),
        }
    }

    /// Does `b` run beside `a` from about where `a` starts - another lane of the same spline,
    /// or the other branch of a fork in the same junction - without `a` leading into it? A
    /// timetable track that lists two such paths one after the other changes lanes there
    /// (Spandau's line 92 moves from lane 5 to lane 6 of a six-lane Falkenseer Chaussee
    /// piece that way); driving `a` to its end first and then starting `b` sent the bus
    /// 30 m back and sideways.
    ///
    /// Or `b` runs beside a stretch of `a` without starting where it does: a bus lane or
    /// stop lane of a junction object that begins before or after the lane the route came
    /// in on (Spandau's Klosterstraße, Heerstraße and Altstädter Ring junctions: the station
    /// link reaches the stop on the through lane, the next one leaves from the stop lane
    /// beside it). Driven one after the other the bus jumped up to 58 m back.
    pub fn parallel(&self, a: usize, b: usize) -> bool {
        let (Some(la), Some(lb)) = (self.lanes.get(a), self.lanes.get(b)) else { return false };
        if a == b || la.next.contains(&b) {
            return false;
        }
        let same = match (la.key, lb.key) {
            (Some(x), Some(y)) => x.tile == y.tile && x.id == y.id && x.path != y.path,
            _ => false,
        };
        if same && (lb.start() - la.start()).truncate().length() < 8.0 && (lb.start().z - la.start().z).abs() < 1.0 && wrap_deg(lb.start_heading() - la.start_heading()).abs() < 45.0 {
            return true;
        }
        // (a lane of another spline beside it counts as well: mod maps lay a bus lane or a
        // second carriageway as splines of their own, and Novi Sad's tracks change onto
        // them 30 m before the lane they come in on ends)
        self.beside(la, lb)
    }

    /// Does lane `b` run beside `a` (a lane's width or so to the side, the same way) along
    /// at least 8 m of it?
    fn beside(&self, la: &Lane, lb: &Lane) -> bool {
        let n = 8;
        let (mut along, step) = (0.0f32, la.length() / n as f32);
        for i in 0..=n {
            let s = la.length() * i as f32 / n as f32;
            let (p, h) = la.at(s);
            let Some((sb, d)) = lb.nearest_point(p) else { continue };
            let inside = sb > 0.5 && sb < lb.length() - 0.5;
            // (on the same level: a track or road under a bridge lies a few metres off in 3D
            // too, and a train "changed lanes" down onto the line under its viaduct)
            let (q, hb) = lb.at(sb);
            let level = (q.z - p.z).abs() < 1.0;
            if inside && level && d > 0.8 && d < 5.5 && wrap_deg(hb - h).abs() < 30.0 {
                along += step;
            }
        }
        along >= 8.0 - 1e-3
    }

    /// Where on lane `b`, beside `a` (see `parallel`), the car at distance `s` along `a` is:
    /// the same distance on lanes that start together, else the point of `b` beside it.
    pub fn beside_s(&self, a: usize, b: usize, s: f32) -> f32 {
        let (la, lb) = (&self.lanes[a], &self.lanes[b]);
        if (lb.start() - la.start()).truncate().length() < 8.0 {
            let frac = if la.length() > 0.0 { s / la.length() } else { 0.0 };
            return frac * lb.length();
        }
        (s + self.beside_delta(a, b)).clamp(0.0, lb.length())
    }

    /// How much further along `b` than along `a` a point beside both lies (see `parallel`):
    /// 0 for lanes that start together, the distance `b` starts before `a` (positive) or
    /// after it (negative) otherwise.
    pub fn beside_delta(&self, a: usize, b: usize) -> f32 {
        let (la, lb) = (&self.lanes[a], &self.lanes[b]);
        if (lb.start() - la.start()).truncate().length() < 8.0 {
            return 0.0;
        }
        match la.nearest_point(lb.start()) {
            Some((sa, _)) if sa > 0.5 => -sa,
            _ => lb.nearest_point(la.start()).map(|(sb, _)| sb).unwrap_or(0.0),
        }
    }

    /// Point on a lane sequence closest to `p`: (index into `route`, distance along that lane).
    pub fn project_on_route(&self, route: &[usize], p: DVec3) -> Option<(usize, f32)> {
        self.project_on_route_lateral(route, p).map(|(ri, s, _)| (ri, s))
    }

    /// Like `project_on_route`, plus the lateral offset of `p` from the lane (m, right of
    /// the driving direction positive).
    pub fn project_on_route_lateral(&self, route: &[usize], p: DVec3) -> Option<(usize, f32, f32)> {
        let mut best: Option<(usize, f32, f64, f32)> = None;
        for (ri, &li) in route.iter().enumerate() {
            let l = &self.lanes[li];
            for k in 0..l.points.len().saturating_sub(1) {
                let a = l.points[k];
                let b = l.points[k + 1];
                let ab = b - a;
                let t = ((p - a).dot(ab) / ab.length_squared().max(1e-9)).clamp(0.0, 1.0);
                let q = a + ab * t;
                let d = (q - p).truncate().length();
                if best.map(|b| d < b.2).unwrap_or(true) {
                    let dir = ab.truncate().normalize_or_zero();
                    let rel = (p - q).truncate();
                    // right vector of (dir.x, dir.y) is (dir.y, -dir.x)
                    let lateral = (rel.x * dir.y - rel.y * dir.x) as f32;
                    best = Some((ri, l.dist[k] + (l.dist[k + 1] - l.dist[k]) * t as f32, d, lateral));
                }
            }
        }
        best.map(|(ri, s, _, lat)| (ri, s, lat))
    }

    /// Where a bus stop at `p` lies on `route`: like [`Network::project_on_route_lateral`],
    /// but among the points within `reach` (when given) and not before route index
    /// `from`, a lane with the stop on its kerb side (right, or left with left-hand traffic)
    /// goes before a nearer one with the stop across it - a route back along the same
    /// street passes each stop twice, once from the other side, and the bus stopped at the
    /// stop across the road on its way out. Falls back to the nearest point.
    pub fn project_stop_on_route(&self, route: &[usize], p: DVec3, reach: Option<f64>, from: usize) -> Option<(usize, f32, f32)> {
        // (index, s, distance, lateral) of the best on the kerb side, and of any
        let mut kerb: Option<(usize, f32, f64, f32)> = None;
        let mut any: Option<(usize, f32, f64, f32)> = None;
        for (ri, &li) in route.iter().enumerate().skip(from.min(route.len())) {
            let l = &self.lanes[li];
            for k in 0..l.points.len().saturating_sub(1) {
                let a = l.points[k];
                let b = l.points[k + 1];
                let ab = b - a;
                let t = ((p - a).dot(ab) / ab.length_squared().max(1e-9)).clamp(0.0, 1.0);
                let q = a + ab * t;
                let d = (q - p).truncate().length();
                if reach.is_some_and(|r| d > r) {
                    continue;
                }
                let dir = ab.truncate().normalize_or_zero();
                let rel = (p - q).truncate();
                let lateral = (rel.x * dir.y - rel.y * dir.x) as f32;
                let cand = (ri, l.dist[k] + (l.dist[k + 1] - l.dist[k]) * t as f32, d, lateral);
                let kerb_side = if self.left_hand { lateral < -0.3 } else { lateral > 0.3 };
                if kerb_side && kerb.map(|b| d < b.2).unwrap_or(true) {
                    kerb = Some(cand);
                }
                if any.map(|b| d < b.2).unwrap_or(true) {
                    any = Some(cand);
                }
            }
        }
        match kerb.or(any) {
            Some((ri, s, _, lat)) => Some((ri, s, lat)),
            None if from > 0 => self.project_stop_on_route(route, p, reach, 0),
            None => None,
        }
    }

    /// Geometric conflicts between the lanes of each crossing object (paths that intersect
    /// or end at the same point, with where they meet), and the footpaths crossing its
    /// street lanes; the lane joins, swept paths, and lane crossings and merges outside the
    /// objects.
    /// AI cars sort out who goes first from these (`Network::must_yield`).
    pub fn compute_conflicts(&mut self) {
        self.conflicts = vec![Vec::new(); self.lanes.len()];
        self.crossings = vec![Vec::new(); self.lanes.len()];
        self.walks = vec![Vec::new(); self.lanes.len()];
        let (n, walks) = self.conflicts_from(0);
        let joins = self.compile_joins() + self.compile_swept();
        let t = std::time::Instant::now();
        let outside = self.compile_lane_crossings(0);
        let beside = self.compile_outside_converging(0);
        log::info!("path network: {n} conflicting lane pairs at crossings, {joins} lane joins, {outside} lane crossings and {beside} merges outside the objects ({:.1} ms), {walks} footpath crossings", t.elapsed().as_secs_f64() * 1e3);
    }

    /// `conflicts`, `crossings` and `walks` of the junction objects whose lanes start at
    /// index `first` (all lanes of an object come with its tile, so a grown network only
    /// needs its new objects looked at). Returns the pairs and footpath crossings found.
    fn conflicts_from(&mut self, first: usize) -> (usize, usize) {
        let len = self.lanes.len();
        self.conflicts.resize(len, Vec::new());
        self.crossings.resize(len, Vec::new());
        self.walks.resize(len, Vec::new());
        let mut by_object: HashMap<((i32, i32), i64), Vec<usize>> = HashMap::new();
        let mut walks_by_object: HashMap<((i32, i32), i64), Vec<usize>> = HashMap::new();
        for (i, l) in self.lanes.iter().enumerate().skip(first) {
            if l.source != 2 {
                continue;
            }
            let Some(k) = l.key else { continue };
            match l.kind {
                // a level crossing is a junction of a road and a railway
                LaneKind::Street | LaneKind::Rail => by_object.entry((k.tile, k.id)).or_default().push(i),
                LaneKind::Sidewalk => walks_by_object.entry((k.tile, k.id)).or_default().push(i),
                LaneKind::Air => {}
            }
        }
        let mut n = 0;
        for (key, lanes) in &by_object {
            for (x, &i) in lanes.iter().enumerate() {
                for &j in &lanes[x + 1..] {
                    let (a, b) = (&self.lanes[i], &self.lanes[j]);
                    let same_path = a.key.map(|k| k.path) == b.key.map(|k| k.path);
                    if same_path || (a.kind == LaneKind::Rail && b.kind == LaneKind::Rail) {
                        continue;
                    }
                    // one runs into the other: the same end, or ends a car's width apart that
                    // lead into the same lane (two turns converging on one exit)
                    let ends = (a.end() - b.end()).truncate().length();
                    let joint = ends < 1.5 || (ends < MEET_DIST && a.next.iter().any(|n| b.next.contains(n)));
                    // a lead-in and its own stub end a metre apart and start far apart: two
                    // pieces of one way, not two ways running into one
                    let pieces = self.one_way_pieces(i, j);
                    let merge = joint && (a.start() - b.start()).truncate().length() > 3.0 && !pieces;
                    let place = if merge { Some((a.length(), b.length())) } else { polyline_crossing(a, b) };
                    // `[blockpath]`: the object says the two are in each other's way even
                    // where their lines do not cross - the whole of both is the meeting place
                    let (pa, pb) = (a.key.map(|k| k.path).unwrap_or(u16::MAX), b.key.map(|k| k.path).unwrap_or(u16::MAX));
                    if place.is_none() && (a.blocks.contains(&pb) || b.blocks.contains(&pa)) {
                        let (la, lb) = (a.length(), b.length());
                        self.conflicts[i].push(j);
                        self.conflicts[j].push(i);
                        self.crossings[i].push(Crossing { other: j, at: la * 0.5, other_at: lb * 0.5, merge: false, before: la * 0.5, after: la * 0.5, other_before: lb * 0.5, other_after: lb * 0.5, side: false, outside: false });
                        self.crossings[j].push(Crossing { other: i, at: lb * 0.5, other_at: la * 0.5, merge: false, before: lb * 0.5, after: lb * 0.5, other_before: la * 0.5, other_after: la * 0.5, side: false, outside: false });
                        n += 1;
                        continue;
                    }
                    if let Some((sa, sb)) = place {
                        // (after a joint the two are in one lane, and follow each other)
                        let (ab, aa) = if merge { (meeting_extent(a, sa, b).0, 0.5) } else { meeting_extent(a, sa, b) };
                        let (bb, ba) = if merge { (meeting_extent(b, sb, a).0, 0.5) } else { meeting_extent(b, sb, a) };
                        self.conflicts[i].push(j);
                        self.conflicts[j].push(i);
                        self.crossings[i].push(Crossing { other: j, at: sa, other_at: sb, merge, before: ab, after: aa, other_before: bb, other_after: ba, side: false, outside: false });
                        self.crossings[j].push(Crossing { other: i, at: sb, other_at: sa, merge, before: bb, after: ba, other_before: ab, other_after: aa, side: false, outside: false });
                        n += 1;
                    } else if self.add_converging(i, j) {
                        n += 1;
                    }
                }
            }
            if let Some(walks) = walks_by_object.get(key) {
                for &i in lanes.iter().filter(|&&i| self.lanes[i].kind == LaneKind::Street) {
                    for &w in walks {
                        if let Some((sa, sw)) = polyline_crossing(&self.lanes[i], &self.lanes[w]) {
                            self.walks[i].push((w, sa, sw));
                        }
                    }
                }
            }
        }
        let walks: usize = self.walks[first..].iter().map(|w| w.len()).sum();
        (n, walks)
    }

    /// Must a vehicle on lane `a` let one on lane `b` go first where they meet? In order:
    /// a train always goes first; the path with the higher `[rule] priority` goes first;
    /// between equals (BeijingCity sets none at all) a turn across the oncoming traffic waits
    /// for it, then the fallbacks of [`Network::right_of_way`]: a way out of a yard or one
    /// merging into the other's lane yields, a turning movement yields to the one going
    /// straight on, the minor road yields to the major one; only then right before left (the
    /// German "rechts vor links" the stock maps rely on wherever they set no priorities,
    /// mirrored on a left-hand-traffic map).
    pub fn must_yield(&self, a: usize, b: usize) -> bool {
        let (la, lb) = (&self.lanes[a], &self.lanes[b]);
        if la.kind == LaneKind::Rail {
            return false;
        }
        if lb.kind == LaneKind::Rail {
            return true;
        }
        if (la.priority - lb.priority).abs() > 0.5 {
            return la.priority < lb.priority;
        }
        let rel = wrap_deg(lb.start_heading() - la.start_heading());
        // (the turn across the oncoming traffic: left, or right when driving on the left)
        let across = if self.left_hand { 2 } else { 1 };
        // oncoming: the turn across the other's way waits for it, before any fallback (the
        // yard and road-class steps made a straight way yield to an oncoming left turner at
        // 27 of 1237 BeijingCity junction pairs)
        if rel.abs() > 135.0 && (la.turn == across) != (lb.turn == across) {
            return la.turn == across;
        }
        if let Some(y) = self.right_of_way(a, b) {
            return y;
        }
        if rel.abs() > 135.0 {
            return false;
        }
        // `b` comes from the right when it travels to the left of `a`'s way (from the left,
        // travelling to its right, on a left-hand-traffic map)
        let rel = if self.left_hand { -rel } else { rel };
        if rel < -30.0 {
            return true;
        }
        if rel > 30.0 {
            return false;
        }
        // side by side from the same direction: the one turning across the other waits
        la.turn != 0 && lb.turn == 0
    }

    /// Where lanes `a` and `b` meet, along `a` (the crossing point, or `a`'s end at a merge;
    /// `a`'s end when the network does not know the place), and whether it is a merge.
    pub fn meeting_on(&self, a: usize, b: usize) -> (f32, bool) {
        match self.crossings.get(a).and_then(|v| v.iter().find(|c| c.other == b)) {
            Some(c) => (c.at, c.merge),
            None => (self.lanes[a].length(), false),
        }
    }

    /// How a vehicle comes to distance `at` of lane `a`: over the last `APPROACH_WINDOW`
    /// metres of its way there (back along `a` and the lanes leading into it, each time the
    /// one that continues it most smoothly).
    pub fn approach_to(&self, a: usize, at: f32) -> Approach {
        let la = &self.lanes[a];
        let at = at.clamp(0.0, la.length());
        let h_meet = la.at(at).1;
        let mut ap = Approach { turn: la.turn, heading_change: 0.0, entry_heading: 0.0, spline: None };
        let note = |ap: &mut Approach, l: usize| {
            let lane = &self.lanes[l];
            // the road from the first spline lane of the approach only (a yard, the road
            // class): the pieces of a crossing object carry their author's spawn weights (0,
            // 0.001, 0.1, 1.0), no road class (a piece of 0.001 behind a main road's lit stub
            // is no yard). (A lane closed to cars - the app's reverse twins and
            // timetable connectors - is no road.)
            if lane.source == 1 && !lane.no_cars && ap.spline.is_none() {
                ap.spline = Some(self.road_class(l));
            }
        };
        note(&mut ap, a);
        // (back from the meeting place: `left` of the turn's window, `back` the distance from
        // the meeting place to the current lane's start, for the spline's)
        let mut left = APPROACH_WINDOW - at;
        let mut back = at;
        let mut cur = a;
        let mut h_back = la.at((at - APPROACH_WINDOW).max(0.0)).1;
        for _ in 0..16 {
            if left <= 0.0 && (ap.spline.is_some() || back >= SPLINE_WINDOW) {
                break;
            }
            let start_h = self.lanes[cur].start_heading();
            let Some(&p) = self.prev.get(cur).and_then(|v| v.iter().min_by(|&&x, &&y| wrap_deg(self.lanes[x].end_heading() - start_h).abs().total_cmp(&wrap_deg(self.lanes[y].end_heading() - start_h).abs()))) else { break };
            let lp = &self.lanes[p];
            note(&mut ap, p);
            if left > 0.0 {
                h_back = lp.at((lp.length() - left).max(0.0)).1;
                left -= lp.length();
            }
            back += lp.length();
            cur = p;
        }
        ap.heading_change = wrap_deg(h_meet - h_back).abs();
        ap.entry_heading = h_back;
        ap
    }

    /// The class of the road spline lane `l` belongs to, in its direction: its street lanes
    /// whose heading lies within a right angle of `l`'s, their number and the mean of their
    /// `[rule] trafficdensity` and speed limits, as SUMO ranks a node's incoming edges by the
    /// edge's priority, speed and lane number (sumo.md 1.3, `rightOfWay` default) - the road's,
    /// not one lane's: a lane past a median or a bus bay has no linked neighbour, and the map
    /// gives single lanes their own spawn weight (viz case01 AndingR_noside: 723743/9 counted as a
    /// one-lane road, 191/6 of density 3 beside lanes of 1 as a major road, and one crossing of
    /// two roads had two right-of-way orders). A spline's paths are numbered from 0 without gaps
    /// (the spline file's `[path]` list); a lane of no spline is a road of its own `left`/`right`
    /// chain with its own density and limit.
    pub fn road_class(&self, l: usize) -> SplineClass {
        let lane = &self.lanes[l];
        let Some(k) = lane.key.filter(|_| lane.source == 1) else {
            let mut n = 1;
            for side in [true, false] {
                let mut cur = l;
                for _ in 0..self.lanes.len() {
                    let next = if side { self.lanes[cur].left } else { self.lanes[cur].right };
                    match next {
                        Some(x) if x != l && x < self.lanes.len() => {
                            n += 1;
                            cur = x;
                        }
                        _ => break,
                    }
                }
            }
            return SplineClass { lanes: n, density: lane.density, speed_kmh: lane.speed_limit_kmh };
        };
        let h = lane.start_heading();
        let (mut n, mut density, mut speed) = (0u32, 0.0f32, 0.0f32);
        for path in 0..=u16::MAX {
            let Some(list) = self.by_key.get(&LaneKey { path, ..k }) else { break };
            for &x in list {
                let o = &self.lanes[x];
                if o.kind == LaneKind::Street && wrap_deg(o.start_heading() - h).abs() < 90.0 {
                    n += 1;
                    density += o.density;
                    speed += o.speed_limit_kmh;
                }
            }
        }
        if n == 0 {
            return SplineClass { lanes: 1, density: lane.density, speed_kmh: lane.speed_limit_kmh };
        }
        SplineClass { lanes: n, density: density / n as f32, speed_kmh: speed / n as f32 }
    }

    /// The right of way between two paths of equal `[rule] priority` before right before
    /// left: Some(true) when `a` yields to `b`, Some(false) when `b` yields to `a`, None when
    /// none of these says. In order:
    /// 1. a way out of a yard, a car park, a garage or a bus bay (the spline lane it comes off
    ///    within `SPLINE_WINDOW` carries `[rule] trafficdensity` <= `YARD_DENSITY`: the stock and
    ///    BeijingCity paths off the street say so) yields to
    ///    the street; at a merge, a way coming in at an angle to the way out (`MERGE_DEG` more
    ///    than the other, which comes within `TURN_DEG` of it) yields to the other;
    /// 2. a turning movement (a turn flag, or more than `TURN_DEG` of heading change over its
    ///    approach) yields to one going on straight (not between oncoming ways: there the
    ///    left turn waits, step 4);
    /// 3. the minor road yields to the major one, read from the spline lane each approach
    ///    comes off within the window (none for one of them: no road class): 2+ lanes in its
    ///    direction against 1, else a mean `[rule] trafficdensity` more than `CLASS_DENSITY_RATIO`
    ///    times lower, else a mean speed limit `CLASS_SPEED_RATIO` times lower - of the road's
    ///    lanes in that direction (`road_class`).
    pub fn right_of_way(&self, a: usize, b: usize) -> Option<bool> {
        let (at_a, merge) = self.meeting_on(a, b);
        let (at_b, _) = self.meeting_on(b, a);
        let (pa, pb) = (self.approach_to(a, at_a), self.approach_to(b, at_b));
        let decided = |ya: bool, yb: bool| -> Option<bool> {
            match (ya, yb) {
                (true, false) => Some(true),
                (false, true) => Some(false),
                _ => None,
            }
        };
        // 1. yards (from the spline the approach comes off within the window: none, no yard;
        // a density of 0 says no random traffic, not a yard); merging into the other's lane
        let yard = |p: &Approach| p.spline.is_some_and(|c| c.density > 0.0 && c.density <= YARD_DENSITY);
        if let Some(y) = decided(yard(&pa), yard(&pb)) {
            return Some(y);
        }
        if merge {
            // (how far off the way out each comes in: the lane both lead into, else the
            // middle of their two ends)
            let (la, lb) = (&self.lanes[a], &self.lanes[b]);
            let exit = la.next.iter().find(|n| lb.next.contains(n)).and_then(|&n| self.lanes.get(n)).map(|n| n.start_heading());
            let exit = exit.unwrap_or(la.end_heading() + wrap_deg(lb.end_heading() - la.end_heading()) * 0.5);
            let (oa, ob) = (wrap_deg(exit - pa.entry_heading).abs(), wrap_deg(exit - pb.entry_heading).abs());
            if let Some(y) = decided(oa >= ob + MERGE_DEG && ob <= TURN_DEG, ob >= oa + MERGE_DEG && oa <= TURN_DEG) {
                return Some(y);
            }
        }
        // 2. turning against straight on (oncoming ways: `must_yield`)
        let rel = wrap_deg(self.lanes[b].start_heading() - self.lanes[a].start_heading());
        if rel.abs() <= 135.0 {
            let turning = |p: &Approach| p.turn != 0 || p.heading_change > TURN_DEG;
            if let Some(y) = decided(turning(&pa), turning(&pb)) {
                return Some(y);
            }
        }
        // 3. road class, from the spline approaches
        let (Some(ca), Some(cb)) = (pa.spline, pb.spline) else { return None };
        if let Some(y) = decided(ca.lanes == 1 && cb.lanes >= 2, cb.lanes == 1 && ca.lanes >= 2) {
            return Some(y);
        }
        // (a density of 0 is no random traffic, no road class: BeijingCity's main road spline
        // before case05's junction carries 0)
        let (da, db) = (ca.density, cb.density);
        if da > 0.0 && db > 0.0 {
            if let Some(y) = decided(db > da * CLASS_DENSITY_RATIO, da > db * CLASS_DENSITY_RATIO) {
                return Some(y);
            }
        }
        decided(cb.speed_kmh >= ca.speed_kmh * CLASS_SPEED_RATIO, ca.speed_kmh >= cb.speed_kmh * CLASS_SPEED_RATIO)
    }

    /// The lane beside `lane` at `s` that carries the traffic the other way (the other half
    /// of a two-way street): (lane, distance along it at the same place, how far its middle
    /// lies over to the oncoming side - the left, or the right on a left-hand-traffic map;
    /// `oncoming_sign` turns it into a lateral offset).
    pub fn opposite(&self, lane: usize, s: f32) -> Option<(usize, f32, f32)> {
        let l = self.lanes.get(lane)?;
        let (p, h) = l.at(s);
        let hr = (h as f64).to_radians();
        let left = DVec3::new(-hr.cos(), hr.sin(), 0.0);
        let left = if self.left_hand { -left } else { left };
        let cx = (p.x / GRID_CELL).floor() as i32;
        let cy = (p.y / GRID_CELL).floor() as i32;
        let mut best: Option<(usize, f32, f32)> = None;
        let mut seen: Vec<usize> = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                for &i in self.grid.get(&(cx + dx, cy + dy)).map(|v| v.as_slice()).unwrap_or(&[]) {
                    if i == lane || seen.contains(&i) {
                        continue;
                    }
                    seen.push(i);
                    let o = &self.lanes[i];
                    if o.kind != LaneKind::Street || o.no_cars {
                        continue;
                    }
                    let Some((os, d)) = o.nearest_point(p) else { continue };
                    if d > 7.0 || d < 1.5 {
                        continue;
                    }
                    let (q, oh) = o.at(os);
                    let side = (q - p).dot(left) as f32;
                    if side < 1.5 || wrap_deg(oh - h).abs() < 150.0 {
                        continue;
                    }
                    if best.map(|b| side < b.2).unwrap_or(true) {
                        best = Some((i, os, side));
                    }
                }
            }
        }
        best
    }

    /// The lanes traffic comes from towards distance `s` of `lane`, walked backwards over
    /// joints and through junctions for up to `within` metres: (lane, offset, the lane it
    /// leads into on the way to `lane`). A vehicle at distance `x` along such a lane is at
    /// `x + offset` in `lane`'s own distances (negative before its start); the first entry
    /// is `lane` itself with offset 0. Where several ways lead to one lane, the shortest
    /// counts. At most `max` lanes.
    pub fn upstream(&self, lane: usize, s: f32, within: f32, max: usize) -> Vec<(usize, f32, Option<usize>)> {
        let mut out: Vec<(usize, f32, Option<usize>)> = vec![(lane, 0.0, None)];
        if lane >= self.lanes.len() {
            return out;
        }
        // nearest first (a junction's lanes can be reached two ways)
        let mut done = vec![false];
        while let Some(k) = (0..out.len()).filter(|&k| !done[k]).max_by(|&a, &b| out[a].1.total_cmp(&out[b].1)) {
            done[k] = true;
            let (l, off, _) = out[k];
            // this lane starts `s - off` metres before the place: nothing before it is in reach
            if s - off > within {
                continue;
            }
            for &p in self.prev.get(l).map(|v| v.as_slice()).unwrap_or(&[]) {
                if p == lane || self.lanes[p].kind != self.lanes[lane].kind {
                    continue;
                }
                let o = off - self.lanes[p].length();
                match out.iter().position(|e| e.0 == p) {
                    Some(e) => {
                        if o > out[e].1 && !done[e] {
                            out[e] = (p, o, Some(l));
                        }
                    }
                    None => {
                        if out.len() < max {
                            out.push((p, o, Some(l)));
                            done.push(false);
                        }
                    }
                }
            }
        }
        out
    }

    /// Lane `s` is a short lit stub: at most `STUB_MAX` long, under a light.
    pub fn is_lit_stub(&self, s: usize) -> bool {
        self.lanes.get(s).is_some_and(|l| l.traffic_light.is_some() && l.length() <= STUB_MAX)
    }

    /// The links of lane `a` that go round one of its short lit stubs: `b` in
    /// `a`'s `next` together with a lit stub `s` (`is_lit_stub`) that leads into `b`, with
    /// `b` starting within half a metre of the stub's end and heading as the stub ends. The
    /// way is `a` → `s` → `b`; `a` → `b` (the 1.5 m linking distance reaches past a 1 m
    /// stub) drives past the light. Returns (b, s) pairs.
    pub fn stub_bypasses(&self, a: usize) -> Vec<(usize, usize)> {
        let Some(la) = self.lanes.get(a) else { return Vec::new() };
        let mut out = Vec::new();
        for &st in la.next.iter().filter(|&&n| n != a && self.is_lit_stub(n)) {
            let ls = &self.lanes[st];
            for &b in &la.next {
                if b == st || b == a || !ls.next.contains(&b) {
                    continue;
                }
                let lb = &self.lanes[b];
                if lb.points.is_empty() || (lb.start() - ls.end()).truncate().length() > STUB_JOINT || wrap_deg(lb.start_heading() - ls.end_heading()).abs() > STUB_HEADING {
                    continue;
                }
                if !out.iter().any(|&(x, _)| x == b) {
                    out.push((b, st));
                }
            }
        }
        out
    }

    /// Drop the links of `lanes` that go round a lit stub (`stub_bypasses`),
    /// with their `prev` entries. Returns the links dropped, (from, to).
    fn drop_stub_bypasses(&mut self, lanes: &[usize]) -> Vec<(usize, usize)> {
        let mut dropped = Vec::new();
        for &a in lanes {
            let by: Vec<usize> = self.stub_bypasses(a).into_iter().map(|(b, _)| b).collect();
            if by.is_empty() {
                continue;
            }
            self.lanes[a].next.retain(|n| !by.contains(n));
            for b in by {
                if let Some(p) = self.prev.get_mut(b) {
                    p.retain(|&x| x != a);
                }
                dropped.push((a, b));
            }
        }
        dropped
    }

    /// Lanes `i` and `j` are pieces of one way, not two ways that meet: one
    /// leads into the other (and no other stream's way runs through the piece, `fed_past`),
    /// directly or over at most two short pieces (`STUB_MAX`: a
    /// lead-in and its stub, whose ends lie a metre apart; a lead-in and the 0.4 m piece
    /// after its stub), or both are short stubs side by side from one place (a lit stub and
    /// an unlit one of 0.3 m at the end of one lead-in: a fork).
    pub fn one_way_pieces(&self, i: usize, j: usize) -> bool {
        let (a, b) = (&self.lanes[i], &self.lanes[j]);
        if a.length() <= STUB_MAX && b.length() <= STUB_MAX && !a.points.is_empty() && !b.points.is_empty() && (a.start() - b.start()).truncate().length() <= STUB_JOINT {
            return true;
        }
        // (not when the piece is another stream's way too, `fed_past`)
        if self.leads_into(i, j) {
            return !self.fed_past(i, j);
        }
        self.leads_into(j, i) && !self.fed_past(j, i)
    }

    /// Compile the meeting place of two lanes `i` and `j` of one crossing object that neither
    /// cross nor run into one another but run side by side closer than a car is wide
    /// (`converging_stretch`): a merge-kind meeting place over the stretch, as far as its end
    /// (where they part, or where they end together), the right of way read as for any merge
    /// (`must_yield`; neither yields: a zipper). Not between pieces of one way
    /// (`one_way_pieces`, nor one leading into the other within three links), nor two
    /// branches forking from one point (starts within half a metre), nor two lanes of one path
    /// (one key: both directions of a two-way path) or of one spline (a road's lanes side by
    /// side), nor two that have a meeting place already; lanes of no one object only where
    /// their centre lines come within `CONVERGE_DIST` (the edges' overlap is a test for pieces
    /// of one object). True when one was added.
    pub fn add_converging(&mut self, i: usize, j: usize) -> bool {
        let (Some(a), Some(b)) = (self.lanes.get(i), self.lanes.get(j)) else { return false };
        if i == j || a.points.len() < 2 || b.points.len() < 2 || a.kind != LaneKind::Street || b.kind != LaneKind::Street {
            return false;
        }
        let spline = |l: &Lane| l.key.filter(|_| l.source == 1).map(|k| (k.tile, k.id));
        if (a.key.is_some() && a.key == b.key) || (spline(a).is_some() && spline(a) == spline(b)) || self.crossings.get(i).is_some_and(|c| c.iter().any(|c| c.other == j)) {
            return false;
        }
        if (a.start() - b.start()).truncate().length() <= 0.5 || self.one_way_pieces(i, j) || self.links_within(i, j, 3) || self.links_within(j, i, 3) {
            return false;
        }
        // (lanes whose boxes lie further apart than that never come so close)
        let bbox = |l: &Lane| l.points.iter().fold((f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY), |(x0, y0, x1, y1), p| (x0.min(p.x), y0.min(p.y), x1.max(p.x), y1.max(p.y)));
        let (ba, bb) = (bbox(a), bbox(b));
        if ba.0 > bb.2 + MEET_DIST || bb.0 > ba.2 + MEET_DIST || ba.1 > bb.3 + MEET_DIST || bb.1 > ba.3 + MEET_DIST {
            return false;
        }
        let Some(((a0, a1), (b0, b1), least)) = converging_stretch(a, b) else { return false };
        // (the edges' overlap only between pieces of one object, drawn as one junction; lanes of
        // different roads by their centre lines: two splines' lanes side by side 2.2-2.6 m apart
        // with 3 m widths are a road's lanes, `CONVERGE_DIST`)
        let object = |l: &Lane| l.key.filter(|_| l.source == 2).map(|k| (k.tile, k.id));
        if least >= CONVERGE_DIST && !(object(a).is_some() && object(a) == object(b)) {
            return false;
        }
        // (side by side the same way: two lanes of opposite directions close together are a
        // narrow road, not two ways running into one)
        if wrap_deg(a.at((a0 + a1) * 0.5).1 - b.at((b0 + b1) * 0.5).1).abs() > CONVERGE_HEADING {
            return false;
        }
        let (la, lb) = (a.length(), b.length());
        let (aa, ba) = ((la - a1).min(0.5), (lb - b1).min(0.5));
        self.conflicts[i].push(j);
        self.conflicts[j].push(i);
        self.crossings[i].push(Crossing { other: j, at: a1, other_at: b1, merge: true, before: a1 - a0 + 0.5, after: aa, other_before: b1 - b0 + 0.5, other_after: ba, side: true, outside: false });
        self.crossings[j].push(Crossing { other: i, at: b1, other_at: a1, merge: true, before: b1 - b0 + 0.5, after: ba, other_before: a1 - a0 + 0.5, other_after: aa, side: true, outside: false });
        true
    }

    /// Lane `from` reaches `to` over at most `n` links.
    fn links_within(&self, from: usize, to: usize, n: usize) -> bool {
        let mut at = vec![from];
        for _ in 0..n {
            at = at.iter().flat_map(|&l| self.lanes[l].next.iter().copied()).collect();
            if at.contains(&to) {
                return true;
            }
            if at.is_empty() {
                return false;
            }
        }
        false
    }

    /// The crossings of two vehicle lanes (street, or a street and a track) that no one crossing
    /// object compiles: SUMO makes every crossing of two connections a foe in its node's request
    /// matrix, whether or not the map draws a node (sumo.md 1.3); OMSI maps may draw two roads
    /// across each other with splines alone, or a spline across a piece of an object, and the
    /// crossing is then the junction. Two lanes whose polylines cross at an inner point
    /// (`polyline_crossing`) at the same level (`LEVEL_DZ`: farther apart in height one passes
    /// over the other), not two lanes of one crossing object (`conflicts_from` compiles those)
    /// nor of one spline, not two tracks (as inside an object: the railway's signals keep them
    /// apart), not pieces of one way (`one_way_pieces`), and with no meeting place yet. Only the
    /// pairs with a lane from index `first` on (a grown network looks at its new lanes,
    /// `outside_pairs`). Each pair once, lower index first: (a, b, distance along a, distance
    /// along b).
    pub fn lane_crossings(&self, first: usize) -> Vec<(usize, usize, f32, f32)> {
        let mut out = Vec::new();
        for (j, i) in self.outside_pairs(first, 0.0) {
            let (a, b) = (&self.lanes[i], &self.lanes[j]);
            if a.kind == LaneKind::Rail && b.kind == LaneKind::Rail {
                continue;
            }
            let Some((sa, sb)) = polyline_crossing(a, b) else { continue };
            if (a.at(sa).0.z - b.at(sb).0.z).abs() > LEVEL_DZ || self.one_way_pieces(i, j) {
                continue;
            }
            out.push((j, i, sb, sa));
        }
        out
    }

    /// The candidate pairs of vehicle lanes (street or rail) for a meeting place outside any one
    /// crossing object: not two lanes of one object nor of one spline nor of one path, with no
    /// meeting place yet, their bounding boxes within `reach` metres of each other, and one of
    /// them from index `first` on (found through `grid`). Each pair once, (lower, higher index),
    /// sorted.
    fn outside_pairs(&self, first: usize, reach: f64) -> Vec<(usize, usize)> {
        let vehicle = |l: &Lane| matches!(l.kind, LaneKind::Street | LaneKind::Rail) && l.points.len() >= 2;
        let object = |l: &Lane| l.key.filter(|_| l.source == 2).map(|k| (k.tile, k.id));
        let spline = |l: &Lane| l.key.filter(|_| l.source == 1).map(|k| (k.tile, k.id));
        let bbox = |l: &Lane| l.points.iter().fold((f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY), |(x0, y0, x1, y1), p| (x0.min(p.x), y0.min(p.y), x1.max(p.x), y1.max(p.y)));
        let mut out = Vec::new();
        for i in first..self.lanes.len() {
            let a = &self.lanes[i];
            if !vehicle(a) {
                continue;
            }
            let ba = bbox(a);
            let mut near: Vec<usize> = Vec::new();
            for c in Self::lane_cells(a) {
                if let Some(list) = self.grid.get(&c) {
                    near.extend(list.iter().copied().filter(|&j| j < i));
                }
            }
            near.sort_unstable();
            near.dedup();
            for j in near {
                let b = &self.lanes[j];
                if !vehicle(b) || (object(a).is_some() && object(a) == object(b)) || (spline(a).is_some() && spline(a) == spline(b)) || (a.key.is_some() && a.key == b.key) {
                    continue;
                }
                let bb = bbox(b);
                if ba.0 > bb.2 + reach || bb.0 > ba.2 + reach || ba.1 > bb.3 + reach || bb.1 > ba.3 + reach {
                    continue;
                }
                if self.crossings.get(i).is_some_and(|c| c.iter().any(|c| c.other == j)) {
                    continue;
                }
                out.push((j, i));
            }
        }
        out.sort_unstable();
        out
    }

    /// Compile a crossing meeting place for each of `lane_crossings(first)`, as `conflicts_from`
    /// compiles one inside an object: over the stretch either side of the point where the two
    /// centre lines stay within `MEET_DIST` (`meeting_extent`: the two bodies' half widths).
    /// Marked `outside`: `Junctions::extend` puts the two lanes into one junction table, where the
    /// right of way is `must_yield`'s as for every compiled foe. Returns how many were added.
    pub fn compile_lane_crossings(&mut self, first: usize) -> usize {
        let len = self.lanes.len();
        self.conflicts.resize(len, Vec::new());
        self.crossings.resize(len, Vec::new());
        let found = self.lane_crossings(first);
        for &(i, j, sa, sb) in &found {
            let (a, b) = (&self.lanes[i], &self.lanes[j]);
            let (ab, aa) = meeting_extent(a, sa, b);
            let (bb, ba) = meeting_extent(b, sb, a);
            self.conflicts[i].push(j);
            self.conflicts[j].push(i);
            self.crossings[i].push(Crossing { other: j, at: sa, other_at: sb, merge: false, before: ab, after: aa, other_before: bb, other_after: ba, side: false, outside: true });
            self.crossings[j].push(Crossing { other: i, at: sb, other_at: sa, merge: false, before: bb, after: ba, other_before: ab, other_after: aa, side: false, outside: true });
        }
        found.len()
    }

    /// The merges without an object: two street lanes of no one crossing object that run side
    /// by side closer than a car is wide without crossing or running into one another get the
    /// side-merge meeting place of `add_converging`, as lanes of one object do (SUMO: two ways
    /// whose shapes overlap meet at a node, a zipper where nothing else orders them; OMSI maps
    /// draw such overlaps with splines alone - X10's parallel ramp splines 0.5-1.8 m apart, a
    /// spline under the first metres of an object's piece). Not two lanes leaving one lane, or
    /// one leaving a lane the other's way leaves within three links (a fork: one lane before the
    /// split keeps them in queue order, as `same_root` inside an object), nor at different
    /// levels (`LEVEL_DZ`), nor neighbours (`left`/`right`: lanes of one road). Only pairs with a
    /// lane from index `first` on (`outside_pairs`). Returns how many were added.
    pub fn compile_outside_converging(&mut self, first: usize) -> usize {
        let len = self.lanes.len();
        self.conflicts.resize(len, Vec::new());
        self.crossings.resize(len, Vec::new());
        let mut n = 0;
        for (j, i) in self.outside_pairs(first, MEET_DIST) {
            let (a, b) = (&self.lanes[i], &self.lanes[j]);
            if a.kind != LaneKind::Street || b.kind != LaneKind::Street || a.left == Some(j) || a.right == Some(j) || b.left == Some(i) || b.right == Some(i) {
                continue;
            }
            let prev = |l: usize| self.prev.get(l).map(|v| v.as_slice()).unwrap_or(&[]);
            let fork = prev(i).iter().any(|p| prev(j).contains(p)) || prev(i).iter().any(|&p| self.links_within(p, j, 3)) || prev(j).iter().any(|&p| self.links_within(p, i, 3));
            if fork {
                continue;
            }
            let Some(((a0, a1), (b0, b1), _)) = converging_stretch(a, b) else { continue };
            if (a.at((a0 + a1) * 0.5).0.z - b.at((b0 + b1) * 0.5).0.z).abs() > LEVEL_DZ {
                continue;
            }
            if self.add_converging(i, j) {
                n += 1;
            }
        }
        n
    }

    /// The meeting place of every lane join, as SUMO makes every join a junction node with a
    /// response matrix or a zipper (`NBNode`, `NBRequest`), because in OMSI only crossing objects
    /// come with a table: two street
    /// lanes that run into one lane (both in its `prev`) and have none yet - spline lanes outside any crossing
    /// object, a spline lane joining an object's way out, two pieces of one object that start
    /// within 3 m of each other (which `conflicts_from` leaves out) - a merge at their ends, as
    /// `conflicts_from` compiles one (`meeting_extent` back from the end, half a metre on).
    /// Not between pieces of one way (`one_way_pieces`: one leads into the other; the queue
    /// order decides there). Returns how many were added.
    pub fn compile_joins(&mut self) -> usize {
        let len = self.lanes.len();
        self.conflicts.resize(len, Vec::new());
        self.crossings.resize(len, Vec::new());
        let street = |l: &Lane| l.kind == LaneKind::Street && l.points.len() >= 2;
        let mut n = 0;
        for to in 0..len.min(self.prev.len()) {
            if !street(&self.lanes[to]) {
                continue;
            }
            let prev: Vec<usize> = self.prev[to].iter().copied().filter(|&p| p < len && p != to && street(&self.lanes[p])).collect();
            for (x, &i) in prev.iter().enumerate() {
                for &j in &prev[x + 1..] {
                    if self.crossings[i].iter().any(|c| c.other == j) || self.one_way_pieces(i, j) || self.same_root(i, j) {
                        continue;
                    }
                    let (a, b) = (&self.lanes[i], &self.lanes[j]);
                    let (sa, sb) = (a.length(), b.length());
                    let ab = meeting_extent(a, sa, b).0;
                    let bb = meeting_extent(b, sb, a).0;
                    self.conflicts[i].push(j);
                    self.conflicts[j].push(i);
                    self.crossings[i].push(Crossing { other: j, at: sa, other_at: sb, merge: true, before: ab, after: 0.5, other_before: bb, other_after: 0.5, side: false, outside: false });
                    self.crossings[j].push(Crossing { other: i, at: sb, other_at: sa, merge: true, before: bb, after: 0.5, other_before: ab, other_after: 0.5, side: false, outside: false });
                    n += 1;
                }
            }
        }
        // lanes converging side by side before the join, without a shared next lane (viz case06: 4115 runs beside 3745 and reaches 3747 through 4126):
        // the lanes up to two links back from each side of a join, converging closer than a
        // car is wide, get the side-merge meeting place of `add_converging`
        for (i, j) in self.join_approaches() {
            if !self.same_root(i, j) && self.add_converging(i, j) {
                n += 1;
            }
        }
        n
    }

    /// The swept-path meeting places of the crossing objects: a piece of an
    /// object and every lane of the object or leading into it whose last body length
    /// (`SWEPT_BUS_LENGTH`) before its end lies in the piece's swept path - the two lanes' half
    /// widths (their vehicles' room) and the off-tracking of a long vehicle in the piece's
    /// tightest bend (`SWEPT_WHEELBASE`^2 / 2R, R no less than `SWEPT_MIN_RADIUS`) - when nothing else is compiled
    /// between the two: a meeting place over the two stretches, so a vehicle standing there
    /// counts as inside it and a refused vehicle stops before it - in the junction table where
    /// both lanes are movements of the object (a piece, or a spline lane grouped into it by a
    /// join); a lane leading in that is no movement of it keeps the place in the network only
    /// (`Junctions::extend`; `traffic_cases crossings` counts them). SUMO: any conflict between
    /// connection shapes is a foe (sumo.md 1.3); OMSI: a stop line can sit inside a tight
    /// turn's sweep (x10 case03: bus #587 turning right on 2577 swept #3221 standing at
    /// 2605's end, no meeting place between them). Returns how many were added.
    pub fn compile_swept(&mut self) -> usize {
        let len = self.lanes.len();
        self.conflicts.resize(len, Vec::new());
        self.crossings.resize(len, Vec::new());
        let street = |l: &Lane| l.kind == LaneKind::Street && l.points.len() >= 2;
        let mut by_object: std::collections::BTreeMap<((i32, i32), i64), Vec<usize>> = std::collections::BTreeMap::new();
        for (i, l) in self.lanes.iter().enumerate() {
            if l.source == 2 && street(l) {
                if let Some(k) = l.key {
                    by_object.entry((k.tile, k.id)).or_default().push(i);
                }
            }
        }
        let mut n = 0;
        for pieces in by_object.values() {
            // (the object's lanes and the lanes leading into them)
            let mut near: Vec<usize> = pieces.clone();
            for &p in pieces {
                for &q in self.prev.get(p).map(|v| v.as_slice()).unwrap_or(&[]) {
                    if !near.contains(&q) && self.lanes.get(q).is_some_and(street) {
                        near.push(q);
                    }
                }
            }
            for &pi in pieces {
                for &bi in &near {
                    if pi == bi || self.crossings[pi].iter().any(|c| c.other == bi) {
                        continue;
                    }
                    if self.one_way_pieces(pi, bi) || self.same_root(pi, bi) || self.links_within(pi, bi, 3) || self.links_within(bi, pi, 3) {
                        continue;
                    }
                    let (p, b) = (&self.lanes[pi], &self.lanes[bi]);
                    let bend = p.curvature.iter().fold(0.0f32, |m, c| m.max(c.abs()));
                    // (the two lanes' half widths - two lanes of a road side by side are no meeting
                    // place, round 4b x1 - and the piece's off-tracking)
                    let radius = if bend > 0.0 { (1.0 / bend).max(SWEPT_MIN_RADIUS) } else { f32::INFINITY };
                    let reach = 0.5 * (p.width + b.width) + SWEPT_WHEELBASE * SWEPT_WHEELBASE / (2.0 * radius);
                    let lb = b.length();
                    let (mut b0, mut b1, mut p0, mut p1) = (f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY);
                    // (the last body length: where a vehicle stands at the lane's end, the bus and
                    // the metre a car comes to rest short of its stop, `plan`'s `REST_SHORT`)
                    let mut sb = (lb - SWEPT_BUS_LENGTH - 1.0).max(0.0);
                    loop {
                        let at = sb.min(lb);
                        if let Some((sp, d)) = p.nearest_point(b.at(at).0) {
                            if (d as f32) < reach && sp > 0.05 && sp < p.length() - 0.05 {
                                b0 = b0.min(at);
                                b1 = b1.max(at);
                                p0 = p0.min(sp);
                                p1 = p1.max(sp);
                            }
                        }
                        if at >= lb {
                            break;
                        }
                        sb += 0.5;
                    }
                    if !b0.is_finite() {
                        continue;
                    }
                    // (the way on from `b` meets the piece there already - within half a bus of
                    // the swept stretch, where a bus standing at `b`'s end reaches: the lead-in
                    // of a way that crosses the turn a few metres on, its claim's chain carries
                    // that meeting place, round 4 rule E)
                    let covered = self.lanes[bi].next.iter().any(|&nx| self.crossings.get(nx).is_some_and(|cs| cs.iter().any(|c| c.other == pi && c.other_at + c.other_after >= p0 - SWEPT_BUS_LENGTH * 0.5 && c.other_at - c.other_before <= p1 + SWEPT_BUS_LENGTH * 0.5)));
                    if covered {
                        continue;
                    }
                    let (pa, ba) = ((p0 + p1) * 0.5, (b0 + b1) * 0.5);
                    let (ph, bh) = (((p1 - p0) * 0.5).max(0.5), ((b1 - b0) * 0.5).max(0.5));
                    self.conflicts[pi].push(bi);
                    self.conflicts[bi].push(pi);
                    self.crossings[pi].push(Crossing { other: bi, at: pa, other_at: ba, merge: false, before: ph, after: ph, other_before: bh, other_after: bh, side: false, outside: false });
                    self.crossings[bi].push(Crossing { other: pi, at: ba, other_at: pa, merge: false, before: bh, after: bh, other_before: ph, other_after: ph, side: false, outside: false });
                    n += 1;
                }
            }
        }
        n
    }

    /// Lanes `i` and `j` are pieces of ONE crossing object that
    /// leave one lane together (a lane forking into twin pieces of the object that rejoin, viz3
    /// case03: 1106/1110 leave 1100 and rejoin into 1477/1481): pieces of ONE way, as SUMO
    /// makes the connections of one lane that re-merge one internal edge - no meeting place
    /// between them, the queue order decides. Spline lanes leaving one lane and converging
    /// (3126/3129 off 2746, abreast over their last 14 m into 79), and lanes whose common root
    /// is farther back, keep their meeting place.
    pub fn same_root(&self, i: usize, j: usize) -> bool {
        let object = |l: usize| self.lanes.get(l).filter(|l| l.source == 2).and_then(|l| l.key).map(|k| (k.tile, k.id));
        if object(i).is_none() || object(i) != object(j) {
            return false;
        }
        let (pi, pj) = (self.prev.get(i), self.prev.get(j));
        pi.is_some_and(|a| pj.is_some_and(|b| a.iter().any(|r| b.contains(r))))
    }

    /// The pairs of street lanes that may converge before a lane join
    /// without sharing a next lane: for every two lanes running into one lane, the lanes up to
    /// two links back from each (themselves included), one from each side, of different paths.
    pub fn join_approaches(&self) -> Vec<(usize, usize)> {
        let len = self.lanes.len().min(self.prev.len());
        let street = |l: usize| self.lanes.get(l).is_some_and(|l| l.kind == LaneKind::Street && l.points.len() >= 2);
        let back = |from: usize| -> Vec<usize> {
            let mut out = vec![from];
            let mut at = vec![from];
            for _ in 0..2 {
                at = at.iter().flat_map(|&l| self.prev.get(l).cloned().unwrap_or_default()).filter(|&l| street(l) && !out.contains(&l)).collect();
                out.extend(at.iter().copied());
            }
            out
        };
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for to in 0..len {
            if !street(to) {
                continue;
            }
            let prev: Vec<usize> = self.prev[to].iter().copied().filter(|&p| p != to && street(p)).collect();
            for (x, &a) in prev.iter().enumerate() {
                for &b in &prev[x + 1..] {
                    let (ba, bb) = (back(a), back(b));
                    for &i in &ba {
                        for &j in &bb {
                            if i != j && !ba.contains(&j) && !bb.contains(&i) {
                                pairs.push((i.min(j), i.max(j)));
                            }
                        }
                    }
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }

    /// The converging lanes of every crossing object (`add_converging`) on a network whose
    /// meeting places were derived before them (a recording's header): the pairs with no
    /// meeting place yet, the lane joins (`compile_joins`), the swept paths (`compile_swept`) and
    /// the lane crossings and merges outside the objects (`compile_lane_crossings`,
    /// `compile_outside_converging`). Returns how many were added.
    pub fn compile_converging(&mut self) -> usize {
        self.conflicts.resize(self.lanes.len(), Vec::new());
        self.crossings.resize(self.lanes.len(), Vec::new());
        let mut by_object: std::collections::BTreeMap<((i32, i32), i64), Vec<usize>> = std::collections::BTreeMap::new();
        for (i, l) in self.lanes.iter().enumerate() {
            if l.source != 2 || l.kind != LaneKind::Street {
                continue;
            }
            if let Some(k) = l.key {
                by_object.entry((k.tile, k.id)).or_default().push(i);
            }
        }
        let mut n = 0;
        for lanes in by_object.values() {
            for (x, &i) in lanes.iter().enumerate() {
                for &j in &lanes[x + 1..] {
                    if self.add_converging(i, j) {
                        n += 1;
                    }
                }
            }
        }
        n + self.compile_joins() + self.compile_swept() + self.compile_lane_crossings(0) + self.compile_outside_converging(0)
    }

    /// Lane `b` (in `a`'s `next`) goes on from `a`'s end: it ends further along than `a`
    /// does. (The 1.5 m linking distance also links a lane to a short piece that starts a
    /// metre before its end and ends where it ends: the last piece of another way running
    /// into the same place, a merge, not the way on.)
    fn continues(&self, a: usize, b: usize) -> bool {
        let (la, lb) = (&self.lanes[a], &self.lanes[b]);
        if la.points.is_empty() || lb.points.is_empty() {
            return false;
        }
        let h = (la.end_heading() as f64).to_radians();
        (lb.end() - la.end()).truncate().dot(DVec2::new(h.sin(), h.cos())) > 0.1
    }

    /// The piece `y` that lane `x` leads into is also the way of another stream that `x`
    /// can drive past: a feeder of `y` that is not one of `x`'s own pieces nor a stub beside
    /// `y` from the same place, while `x` links past `y` into a lane after it (case07
    /// 5229/5230: 5231 feeds the 1.1 m piece 5230 and 5229 links past it into 5228; the
    /// merge 5229/5230 is where the two streams meet). The merge stays.
    fn fed_past(&self, x: usize, y: usize) -> bool {
        let (lx, ly) = (&self.lanes[x], &self.lanes[y]);
        let skips = lx.next.iter().any(|&m| m != y && ly.next.contains(&m));
        if !skips || ly.points.is_empty() {
            return false;
        }
        self.prev.get(y).is_some_and(|feeders| {
            feeders.iter().any(|&k| {
                let lk = &self.lanes[k];
                k != x && !self.leads_into(x, k) && !lk.points.is_empty() && (lk.start() - ly.start()).truncate().length() > STUB_JOINT
            })
        })
    }

    /// Lane `from` leads into `to`: `to` goes on from its end (`continues`), or from the end
    /// of a short piece (`STUB_MAX`) that goes on from it, two such pieces at most.
    fn leads_into(&self, from: usize, to: usize) -> bool {
        let mut at = vec![from];
        for step in 0..3 {
            let mut on = Vec::new();
            for &l in &at {
                for &n in self.lanes[l].next.iter().filter(|&&n| self.continues(l, n)) {
                    if n == to {
                        return true;
                    }
                    if step < 2 && n != from && self.lanes.get(n).is_some_and(|x| x.length() <= STUB_MAX) && !on.contains(&n) {
                        on.push(n);
                    }
                }
            }
            if on.is_empty() {
                return false;
            }
            at = on;
        }
        false
    }

    /// The lit stubs on a linked network (one linked before them, a recording's from
    /// before them): drop the links round the lit stubs and the merges of two pieces of one
    /// way (`one_way_pieces`) from the meeting places. The rest of what was derived (`prev`,
    /// `reach`, the other meeting places) stays as it was. Returns (links dropped, merge
    /// pairs dropped, each pair once with the lower index first). Does nothing on a network
    /// that has them already.
    pub fn apply_stub_links(&mut self) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
        let all: Vec<usize> = (0..self.lanes.len()).collect();
        let links = self.drop_stub_bypasses(&all);
        let mut pairs = Vec::new();
        for i in 0..self.crossings.len().min(self.lanes.len()) {
            let gone: Vec<usize> = self.crossings[i].iter().filter(|c| c.merge && c.other < self.lanes.len() && self.one_way_pieces(i, c.other)).map(|c| c.other).collect();
            if gone.is_empty() {
                continue;
            }
            self.crossings[i].retain(|c| !(c.merge && gone.contains(&c.other)));
            for o in gone {
                if !self.crossings[i].iter().any(|c| c.other == o) {
                    if let Some(c) = self.conflicts.get_mut(i) {
                        c.retain(|&x| x != o);
                    }
                }
                if i < o {
                    pairs.push((i, o));
                }
            }
        }
        (links, pairs)
    }

    /// Connect lane ends to lane starts that lie within `tol` metres with a compatible heading.
    pub fn link(&mut self, tol: f64) {
        self.by_key.clear();
        for (i, l) in self.lanes.iter().enumerate() {
            if let Some(k) = l.key {
                self.by_key.entry(k).or_default().push(i);
            }
        }
        self.build_grid();
        // neighbouring lanes: paths of one spline running the same way, next to each other
        let mut by_spline: HashMap<((i32, i32), i64, bool), Vec<usize>> = HashMap::new();
        for (i, l) in self.lanes.iter().enumerate() {
            if let Some(k) = l.key {
                if l.source == 1 && matches!(l.kind, LaneKind::Street) {
                    by_spline.entry((k.tile, k.id as i64, l.reversed)).or_default().push(i);
                }
            }
        }
        let mut neighbours = 0usize;
        for (_, mut list) in by_spline {
            list.sort_by(|a, b| self.lanes[*a].offset.total_cmp(&self.lanes[*b].offset));
            for w in list.windows(2) {
                let (a, b) = (w[0], w[1]);
                if Lane::beside_each_other(&self.lanes[a], &self.lanes[b]) {
                    // b lies at the larger offset: to the right when driving along the spline
                    if neighbours == 0 {
                        let p = self.lanes[a].start();
                        log::info!("first neighbouring lane pair at ({:.1}, {:.1}), heading {:.0}", p.x, p.y, self.lanes[a].start_heading());
                    }
                    neighbours += 1;
                    let reversed = self.lanes[a].reversed;
                    if reversed {
                        self.lanes[a].left = Some(b);
                        self.lanes[b].right = Some(a);
                    } else {
                        self.lanes[a].right = Some(b);
                        self.lanes[b].left = Some(a);
                    }
                }
            }
        }
        log::info!("lanes: {} neighbouring lane pairs (lane changes)", neighbours);
        self.clip_drivable(0);
        let cell = 4.0;
        let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        for (i, l) in self.lanes.iter().enumerate() {
            let s = l.start();
            grid.entry(((s.x / cell).floor() as i64, (s.y / cell).floor() as i64)).or_default().push(i);
        }
        let mut links = 0;
        for i in 0..self.lanes.len() {
            let e = self.lanes[i].end();
            let eh = self.lanes[i].end_heading();
            let kind = self.lanes[i].kind;
            let cx = (e.x / cell).floor() as i64;
            let cy = (e.y / cell).floor() as i64;
            let mut found = Vec::new();
            for dx in -1..=1 {
                for dy in -1..=1 {
                    if let Some(list) = grid.get(&(cx + dx, cy + dy)) {
                        for &j in list {
                            if j == i {
                                continue;
                            }
                            let l = &self.lanes[j];
                            if l.kind != kind {
                                continue;
                            }
                            let d = (l.start() - e).truncate().length();
                            let dh = wrap_deg(l.start_heading() - eh).abs();
                            if d <= tol && dh < 40.0 && (l.start().z - e.z).abs() < 3.0 {
                                found.push(j);
                            }
                        }
                    }
                }
            }
            if found.is_empty() && omsi_cfg::env::var_os("OMSI_DEBUG_UNLINKED").is_some() {
                // a lane end with a start of its kind near it that was not taken
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for &j in grid.get(&(cx + dx, cy + dy)).map(|v| v.as_slice()).unwrap_or(&[]) {
                            let l = &self.lanes[j];
                            if j != i && l.kind == kind {
                                let d = (l.start() - e).truncate().length();
                                let dh = wrap_deg(l.start_heading() - eh).abs();
                                if d < 4.0 && dh < 40.0 {
                                    log::info!("unlinked: lane {i} {:?} end ({:.2}, {:.2}, {:.2}) -> lane {j} {:?} start {:.2} m away, dz {:.2}, dh {:.1}", self.lanes[i].key, e.x, e.y, e.z, l.key, d, l.start().z - e.z, dh);
                                }
                            }
                        }
                    }
                }
            }
            links += found.len();
            self.lanes[i].next = found;
        }
        {
            let all: Vec<usize> = (0..self.lanes.len()).collect();
            let dropped = self.drop_stub_bypasses(&all);
            links -= dropped.len();
            log::info!("path network: {} links past a lit stub dropped", dropped.len());
        }
        log::info!("path network: {} lanes, {} links", self.lanes.len(), links);
        self.prev = vec![Vec::new(); self.lanes.len()];
        for i in 0..self.lanes.len() {
            for &n in &self.lanes[i].next {
                if n < self.lanes.len() {
                    self.prev[n].push(i);
                }
            }
        }
        self.compute_conflicts();
        self.compute_reach();
    }

    /// `reach` of every lane: its length plus the best reach of the lanes after it. The lanes
    /// are settled from the dead ends backwards (a lane once all the lanes after it are); a
    /// lane that never settles has a loop ahead of it and can be driven on for ever.
    pub fn compute_reach(&mut self) {
        let dead = self.update_reach();
        log::info!("path network: {dead} street lanes lead into a dead end within {DEAD_END} m");
    }

    /// `compute_reach` without the log line; returns the street lanes that lead into a dead
    /// end.
    fn update_reach(&mut self) -> usize {
        let n = self.lanes.len();
        // a lane closed to cars is no way on for one that is open
        let counts = |i: usize, j: usize| !self.lanes[j].no_cars || self.lanes[i].no_cars;
        let mut waiting: Vec<usize> = (0..n).map(|i| self.lanes[i].next.iter().filter(|&&j| counts(i, j)).count()).collect();
        let mut reach = vec![REACH_MAX; n];
        let mut settled = vec![false; n];
        let mut ready: Vec<usize> = (0..n).filter(|&i| waiting[i] == 0).collect();
        while let Some(j) = ready.pop() {
            let l = &self.lanes[j];
            let best = l.next.iter().filter(|&&k| counts(j, k)).map(|&k| reach[k]).fold(0.0f32, f32::max);
            reach[j] = (l.length() + best).min(REACH_MAX);
            settled[j] = true;
            for &p in self.prev.get(j).map(|v| v.as_slice()).unwrap_or(&[]) {
                if counts(p, j) && !settled[p] {
                    waiting[p] = waiting[p].saturating_sub(1);
                    if waiting[p] == 0 {
                        ready.push(p);
                    }
                }
            }
        }
        let dead = self.lanes.iter().zip(&reach).filter(|(l, r)| l.kind == LaneKind::Street && **r < DEAD_END).count();
        self.reach = reach;
        dead
    }

    /// Nearest street lane and distance along it to a world point.
    /// Sort the lanes into the grid (called by `link`; call again after adding lanes).
    pub fn build_grid(&mut self) {
        self.grid.clear();
        self.start_grid.clear();
        for (i, l) in self.lanes.iter().enumerate() {
            self.start_grid.entry(Self::grid_cell(l.start())).or_default().push(i);
            let mut cells: Vec<(i32, i32)> = Vec::new();
            for w in l.points.windows(2) {
                // every cell along the segment, sampled finer than a cell
                let n = ((w[1] - w[0]).truncate().length() / (GRID_CELL * 0.5)).ceil().max(1.0) as usize;
                for k in 0..=n {
                    let q = w[0].lerp(w[1], k as f64 / n as f64);
                    let c = ((q.x / GRID_CELL).floor() as i32, (q.y / GRID_CELL).floor() as i32);
                    if !cells.contains(&c) {
                        cells.push(c);
                    }
                }
            }
            if l.points.len() == 1 {
                cells.push(((l.points[0].x / GRID_CELL).floor() as i32, (l.points[0].y / GRID_CELL).floor() as i32));
            }
            for c in cells {
                self.grid.entry(c).or_default().push(i);
            }
        }
    }

    /// Lane indices whose starts lie in cells intersecting a circle around `p`.
    /// Callers still apply their own exact distance and lane-kind tests. Sorting keeps
    /// their traversal (and seeded traffic selection) in map order.
    pub fn lanes_starting_near(&self, p: DVec3, radius: f64) -> Vec<usize> {
        let min_x = ((p.x - radius) / GRID_CELL).floor() as i32;
        let max_x = ((p.x + radius) / GRID_CELL).floor() as i32;
        let min_y = ((p.y - radius) / GRID_CELL).floor() as i32;
        let max_y = ((p.y + radius) / GRID_CELL).floor() as i32;
        let cells = (max_x as i64 - min_x as i64 + 1) * (max_y as i64 - min_y as i64 + 1);
        if self.start_grid.is_empty() || cells > self.start_grid.len() as i64 * 2 {
            return (0..self.lanes.len()).collect();
        }
        let mut out = Vec::new();
        for x in min_x..=max_x {
            for y in min_y..=max_y {
                if let Some(lanes) = self.start_grid.get(&(x, y)) {
                    out.extend_from_slice(lanes);
                }
            }
        }
        out.sort_unstable();
        out
    }

    /// The nearest lane of `kind` to `p`: (lane, distance along it, distance to it). With
    /// the grid built, only lanes within about a cell of `p` are considered - enough for
    /// everything that asks where a vehicle or a person stands.
    pub fn nearest_lane(&self, p: DVec3, kind: LaneKind) -> Option<(usize, f32, f64)> {
        if !self.grid.is_empty() {
            if let Some(best) = self.nearest_lane_near(p, kind) {
                return Some(best);
            }
        }
        let mut best: Option<(usize, f32, f64)> = None;
        for (i, l) in self.lanes.iter().enumerate() {
            if l.kind != kind {
                continue;
            }
            for k in 0..l.points.len().saturating_sub(1) {
                let a = l.points[k];
                let b = l.points[k + 1];
                let ab = b - a;
                let t = ((p - a).dot(ab) / ab.length_squared().max(1e-9)).clamp(0.0, 1.0);
                let q = a + ab * t;
                let d = (q - p).length();
                if best.map(|b| d < b.2).unwrap_or(true) {
                    let s = l.dist[k] + (l.dist[k + 1] - l.dist[k]) * t as f32;
                    best = Some((i, s, d));
                }
            }
        }
        best
    }

    /// Find shortest lane path from `start` to `target` using Dijkstra's algorithm.
    pub fn shortest_path(&self, start: usize, target: usize) -> Option<Vec<usize>> {
        if start >= self.lanes.len() || target >= self.lanes.len() {
            return None;
        }
        if start == target {
            return Some(vec![start]);
        }
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        #[derive(Copy, Clone, PartialEq)]
        struct State {
            cost: f32,
            position: usize,
        }
        impl Eq for State {}
        impl Ord for State {
            fn cmp(&self, other: &Self) -> Ordering {
                other.cost.total_cmp(&self.cost)
            }
        }
        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut dist = vec![f32::INFINITY; self.lanes.len()];
        let mut parent = vec![None; self.lanes.len()];
        let mut heap = BinaryHeap::new();

        dist[start] = 0.0;
        heap.push(State { cost: 0.0, position: start });

        while let Some(State { cost, position }) = heap.pop() {
            if position == target {
                let mut path = Vec::new();
                let mut curr = Some(target);
                while let Some(node) = curr {
                    path.push(node);
                    curr = parent[node];
                }
                path.reverse();
                return Some(path);
            }

            if cost > dist[position] {
                continue;
            }

            for &next_idx in &self.lanes[position].next {
                if next_idx >= self.lanes.len() {
                    continue;
                }
                let next_len = self.lanes[next_idx].length();
                let next_cost = cost + next_len;
                if next_cost < dist[next_idx] {
                    dist[next_idx] = next_cost;
                    parent[next_idx] = Some(position);
                    heap.push(State { cost: next_cost, position: next_idx });
                }
            }
        }
        None
    }
}

/// Lanes of other splines or objects running alongside a lane are within this many degrees
/// of its heading (either way).
const ALONGSIDE_DEG: f32 = 35.0;

impl Network {
    /// [round 5 L1] The drivable extent of every street lane from `from` on is cut short of
    /// any other street lane running alongside it, whatever spline or object
    /// it belongs to (the cross-section knew its own spline's paths alone; BeijingCity lays
    /// a 10 m "noside" asphalt under a separate carriageway spline): the strip beyond a
    /// lane's own edge that a squeeze or a shift may use is the drawn road that belongs to
    /// no lane. The lanes already there are cut by the new ones too.
    pub fn clip_drivable(&mut self, from: usize) {
        let n = self.lanes.len();
        let mut clips: Vec<(usize, f32, f32)> = Vec::new();
        // (what cuts a street lane: another street lane; a footpath of another spline - a
        // raised pavement spline laid beside the road, BeijingCity's L_Roadside_5m - at its
        // painted edge, its kerb; a footpath of the lane's own spline lies on the same asphalt
        // and is no edge, the brief's JiananDong case)
        let edge_for = |lane: &Lane, other: &Lane| -> bool {
            match other.kind {
                LaneKind::Street => true,
                LaneKind::Sidewalk => match (lane.key, other.key) {
                    (Some(ka), Some(kb)) => ka.tile != kb.tile || ka.id != kb.id || lane.source != other.source,
                    _ => true,
                },
                _ => false,
            }
        };
        for i in from..n {
            let a = &self.lanes[i];
            if !matches!(a.kind, LaneKind::Street | LaneKind::Sidewalk) || a.points.len() < 2 {
                continue;
            }
            let mut cand: Vec<usize> = Vec::new();
            for c in Self::lane_cells(a) {
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        if let Some(list) = self.grid.get(&(c.0 + dx, c.1 + dy)) {
                            for &j in list {
                                if j != i && !cand.contains(&j) {
                                    cand.push(j);
                                }
                            }
                        }
                    }
                }
            }
            for j in cand {
                let b = &self.lanes[j];
                if b.points.len() < 2 {
                    continue;
                }
                if a.kind == LaneKind::Street && edge_for(a, b) {
                    if let Some((l, r)) = Self::clip_by(a, b) {
                        clips.push((i, l, r));
                    }
                }
                if j < from && b.kind == LaneKind::Street && edge_for(b, a) {
                    if let Some((l, r)) = Self::clip_by(b, a) {
                        clips.push((j, l, r));
                    }
                }
            }
        }
        for (i, l, r) in clips {
            let cur = self.lanes[i].drivable();
            self.lanes[i].drivable = (cur.0.min(l), cur.1.min(r));
        }
    }

    /// Lane `a`'s drivable extent cut short of `b` where `b` runs alongside it (a car's half
    /// width and `NEIGHBOUR_KEEP` off its axis, `omsi_scenery::profile::neighbour_edge`): the
    /// new (left, right), if `b` cuts anything.
    fn clip_by(a: &Lane, b: &Lane) -> Option<(f32, f32)> {
        let half = a.width * 0.5;
        let (mut l, mut r) = a.drivable();
        if l <= half + 0.01 && r <= half + 0.01 {
            return None;
        }
        let reach = (l.max(r) + b.width * 0.5 + 0.5) as f64;
        let mut changed = false;
        // (b sampled every 3 m along its segments: a straight lane has points at its ends only)
        let mut samples: Vec<(DVec3, f32)> = Vec::new();
        for w in b.points.windows(2) {
            let seg = w[1] - w[0];
            let hb = (seg.x.atan2(seg.y).to_degrees()) as f32;
            let n = (seg.truncate().length() / 3.0).ceil().max(1.0) as usize;
            for k in 0..=n {
                samples.push((w[0].lerp(w[1], k as f64 / n as f64), hb));
            }
        }
        for (q, hb) in samples {
            let Some((s, d)) = a.nearest_point(q) else { continue };
            // (beside the lane, not beyond its ends: the lane ahead or behind is no edge)
            if d > reach || s <= 0.5 || s >= a.length() - 0.5 || (q.z - a.at(s).0.z).abs() > 3.0 {
                continue;
            }
            let (p, h) = a.at(s);
            let dh = wrap_deg(hb - h).abs();
            if dh > ALONGSIDE_DEG && dh < 180.0 - ALONGSIDE_DEG {
                continue;
            }
            let hr = (h as f64).to_radians();
            let right = DVec2::new(hr.cos(), -hr.sin());
            let lat = (q - p).truncate().dot(right) as f32;
            // (a street lane: a car's half width and 0.3 off its axis; a footpath: its edge)
            let edge = if b.kind == LaneKind::Street { omsi_scenery::profile::neighbour_edge(lat.abs(), b.width) } else { lat.abs() - b.width * 0.5 }.max(half);
            if lat > 0.0 && edge < r {
                r = edge;
                changed = true;
            } else if lat < 0.0 && edge < l {
                l = edge;
                changed = true;
            }
        }
        changed.then_some((l, r))
    }
}

/// Growing the network while the map streams in: the lanes of newly loaded tiles are
/// appended (existing indices stay valid for the cars and routes that hold them) and linked
/// to what is there, with the same rules as [`Network::link`].
impl Network {
    /// Append `new` lanes and link them. Returns the range of their indices.
    pub fn extend(&mut self, new: Vec<Lane>, tol: f64) -> std::ops::Range<usize> {
        let first = self.lanes.len();
        if new.is_empty() {
            return first..first;
        }
        if first == 0 {
            self.lanes = new;
            self.link(tol);
            return 0..self.lanes.len();
        }
        self.lanes.extend(new);
        let end = self.lanes.len();
        for i in first..end {
            if let Some(k) = self.lanes[i].key {
                self.by_key.entry(k).or_default().push(i);
            }
            self.start_grid.entry(Self::grid_cell(self.lanes[i].start())).or_default().push(i);
            for c in Self::lane_cells(&self.lanes[i]) {
                self.grid.entry(c).or_default().push(i);
            }
        }
        // neighbouring lanes of the new splines
        let mut by_spline: HashMap<((i32, i32), i64, bool), Vec<usize>> = HashMap::new();
        for i in first..end {
            let l = &self.lanes[i];
            if let (Some(k), 1, LaneKind::Street) = (l.key, l.source, l.kind) {
                by_spline.entry((k.tile, k.id, l.reversed)).or_default().push(i);
            }
        }
        for (_, mut list) in by_spline {
            list.sort_by(|a, b| self.lanes[*a].offset.total_cmp(&self.lanes[*b].offset));
            for w in list.windows(2) {
                let (a, b) = (w[0], w[1]);
                if Lane::beside_each_other(&self.lanes[a], &self.lanes[b]) {
                    if self.lanes[a].reversed {
                        self.lanes[a].left = Some(b);
                        self.lanes[b].right = Some(a);
                    } else {
                        self.lanes[a].right = Some(b);
                        self.lanes[b].left = Some(a);
                    }
                }
            }
        }
        // links: ends of new lanes to any start, ends of old lanes to new starts
        let joins = |net: &Network, from: usize, to: usize| -> bool {
            let (a, b) = (&net.lanes[from], &net.lanes[to]);
            if from == to || a.kind != b.kind {
                return false;
            }
            let (e, s) = (a.end(), b.start());
            let dh = ((b.start_heading() - a.end_heading() + 540.0) % 360.0 - 180.0).abs();
            (s - e).truncate().length() <= tol && dh < 40.0 && (s.z - e.z).abs() < 3.0
        };
        let near = |net: &Network, p: DVec3| -> Vec<usize> {
            let c = ((p.x / GRID_CELL).floor() as i32, (p.y / GRID_CELL).floor() as i32);
            let mut out: Vec<usize> = Vec::new();
            for dx in -1..=1 {
                for dy in -1..=1 {
                    if let Some(list) = net.grid.get(&(c.0 + dx, c.1 + dy)) {
                        out.extend(list.iter().copied());
                    }
                }
            }
            out.sort_unstable();
            out.dedup();
            out
        };
        let mut added: Vec<(usize, usize)> = Vec::new();
        for i in first..end {
            let e = self.lanes[i].end();
            let found: Vec<usize> = near(self, e).into_iter().filter(|&j| joins(self, i, j)).collect();
            for &j in &found {
                added.push((i, j));
            }
            self.lanes[i].next = found;
            let s = self.lanes[i].start();
            for j in near(self, s) {
                if j < first && joins(self, j, i) && !self.lanes[j].next.contains(&i) {
                    self.lanes[j].next.push(i);
                    added.push((j, i));
                }
            }
        }
        self.prev.resize(end, Vec::new());
        {
            // (no links round a lit stub: the new lanes and the old ones linked to them, as
            // a stub and the lane past it may come in one batch, or the lane past it later)
            let mut touched: Vec<usize> = added.iter().map(|&(a, _)| a).collect();
            touched.sort_unstable();
            touched.dedup();
            let dropped = self.drop_stub_bypasses(&touched);
            added.retain(|p| !dropped.contains(p));
        }
        for (from, to) in added {
            if !self.prev[to].contains(&from) {
                self.prev[to].push(from);
            }
        }
        // meeting places and footpath crossings inside the new junction objects, the joins,
        // swept paths, and lane crossings and merges outside the objects with a new lane, and how far every
        // lane now leads (a dead end may go on into the new tiles)
        self.conflicts_from(first);
        self.compile_joins();
        self.compile_swept();
        let t = std::time::Instant::now();
        let outside = self.compile_lane_crossings(first);
        let beside = self.compile_outside_converging(first);
        log::debug!("network grew by {} lanes: {outside} lane crossings and {beside} merges outside the objects ({:.2} ms)", end - first, t.elapsed().as_secs_f64() * 1e3);
        self.update_reach_from(first);
        self.clip_drivable(first);
        first..end
    }

    /// Appending lanes can only change the reach of those lanes and their predecessors.
    /// An outgoing edge into the rest of the network uses its already settled reach. This
    /// avoids rewalking every loaded tile whenever the streamer adds a small batch.
    fn update_reach_from(&mut self, first: usize) {
        if self.reach.len() != first {
            self.update_reach();
            return;
        }
        let mut affected: hashbrown::HashSet<usize> = (first..self.lanes.len()).collect();
        let mut pending: Vec<usize> = affected.iter().copied().collect();
        while let Some(i) = pending.pop() {
            // The vector-based full pass is cheaper once much of the network leads
            // into this tile (for example a strongly connected city grid).
            if affected.len() > self.lanes.len() / 4 {
                self.update_reach();
                return;
            }
            for &p in &self.prev[i] {
                if affected.insert(p) {
                    pending.push(p);
                }
            }
        }
        self.reach.resize(self.lanes.len(), REACH_MAX);
        let counts = |i: usize, j: usize| !self.lanes[j].no_cars || self.lanes[i].no_cars;
        let mut waiting: HashMap<usize, usize> = affected.iter().map(|&i| {
            (i, self.lanes[i].next.iter().filter(|&&j| affected.contains(&j) && counts(i, j)).count())
        }).collect();
        let mut ready: Vec<usize> = waiting.iter().filter_map(|(&i, &n)| (n == 0).then_some(i)).collect();
        for &i in &affected {
            self.reach[i] = REACH_MAX;
        }
        while let Some(i) = ready.pop() {
            let l = &self.lanes[i];
            let best = l.next.iter().filter(|&&j| counts(i, j)).map(|&j| self.reach[j]).fold(0.0f32, f32::max);
            self.reach[i] = (l.length() + best).min(REACH_MAX);
            for &p in &self.prev[i] {
                if counts(p, i) {
                    if let Some(n) = waiting.get_mut(&p) {
                        *n = n.saturating_sub(1);
                        if *n == 0 {
                            ready.push(p);
                        }
                    }
                }
            }
        }
    }

    /// Like [`nearest_lane`](Self::nearest_lane), but only among the lanes in the grid cells
    /// around `p` (a cell is [`GRID_CELL`]): None when there are none there, never a search
    /// of the whole network.
    pub fn nearest_lane_near(&self, p: DVec3, kind: LaneKind) -> Option<(usize, f32, f64)> {
        let (cx, cy) = Self::grid_cell(p);
        let mut best: Option<(usize, f32, f64)> = None;
        let mut seen: Vec<usize> = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                let Some(list) = self.grid.get(&(cx + dx, cy + dy)) else { continue };
                for &i in list {
                    if seen.contains(&i) {
                        continue;
                    }
                    seen.push(i);
                    let l = &self.lanes[i];
                    if l.kind != kind {
                        continue;
                    }
                    if let Some((s, d)) = l.nearest_point(p) {
                        if best.map(|b| d < b.2).unwrap_or(true) {
                            best = Some((i, s, d));
                        }
                    }
                }
            }
        }
        best
    }

    /// The nearest lane of `kind` within `max_dist` of `p` that runs within `max_turn`
    /// degrees of `heading` there (a vehicle's own lane, not the one beside it going the
    /// other way): (lane, distance along it, distance to it).
    pub fn lane_along(&self, p: DVec3, heading: f64, kind: LaneKind, max_dist: f64, max_turn: f64) -> Option<(usize, f32, f64)> {
        let (cx, cy) = Self::grid_cell(p);
        let mut best: Option<(usize, f32, f64)> = None;
        let mut seen: Vec<usize> = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                let Some(list) = self.grid.get(&(cx + dx, cy + dy)) else { continue };
                for &i in list {
                    if seen.contains(&i) {
                        continue;
                    }
                    seen.push(i);
                    let l = &self.lanes[i];
                    if l.kind != kind {
                        continue;
                    }
                    let Some((s, d)) = l.nearest_point(p) else { continue };
                    if d > max_dist || best.map(|b| d >= b.2).unwrap_or(false) {
                        continue;
                    }
                    let turn = (l.at(s).1 as f64 - heading + 540.0).rem_euclid(360.0) - 180.0;
                    if turn.abs() <= max_turn {
                        best = Some((i, s, d));
                    }
                }
            }
        }
        best
    }

    /// The grid cell `p` lies in.
    pub fn grid_cell(p: DVec3) -> (i32, i32) {
        ((p.x / GRID_CELL).floor() as i32, (p.y / GRID_CELL).floor() as i32)
    }

    /// The grid cells a lane's points touch (see [`Network::build_grid`]).
    pub fn lane_cells(l: &Lane) -> Vec<(i32, i32)> {
        let mut cells: Vec<(i32, i32)> = Vec::new();
        for w in l.points.windows(2) {
            let n = ((w[1] - w[0]).truncate().length() / (GRID_CELL * 0.5)).ceil().max(1.0) as usize;
            for k in 0..=n {
                let q = w[0].lerp(w[1], k as f64 / n as f64);
                let c = ((q.x / GRID_CELL).floor() as i32, (q.y / GRID_CELL).floor() as i32);
                if !cells.contains(&c) {
                    cells.push(c);
                }
            }
        }
        if l.points.len() == 1 {
            cells.push(((l.points[0].x / GRID_CELL).floor() as i32, (l.points[0].y / GRID_CELL).floor() as i32));
        }
        cells
    }
}

#[cfg(test)]
mod extend_tests {
    use super::*;

    fn straight(x: f64, y0: f64, y1: f64, id: i64, tile: (i32, i32)) -> Lane {
        let mut l = LaneBuilder::polyline(vec![DVec3::new(x, y0, 0.0), DVec3::new(x, y1, 0.0)], LaneKind::Street, 3.0);
        l.key = Some(LaneKey { tile, id, path: 0 });
        l.source = 1;
        l
    }

    #[test]
    fn extend_links_like_link() {
        let all = vec![straight(0.0, 0.0, 100.0, 1, (0, 0)), straight(0.0, 100.0, 200.0, 2, (0, 0)), straight(0.0, 200.0, 300.0, 3, (0, 1)), straight(0.0, 300.0, 400.0, 4, (0, 1))];
        let mut whole = Network { lanes: all.clone(), ..Default::default() };
        whole.link(1.5);
        let mut grown = Network { lanes: all[..2].to_vec(), ..Default::default() };
        grown.link(1.5);
        let r = grown.extend(all[2..].to_vec(), 1.5);
        assert_eq!(r, 2..4);
        for i in 0..4 {
            assert_eq!(grown.lanes[i].next, whole.lanes[i].next, "lane {i}");
            assert_eq!(grown.prev[i], whole.prev[i], "lane {i}");
        }
        assert_eq!(grown.reach, whole.reach);
        assert_eq!(grown.crossings.len(), 4);
        assert_eq!(grown.find(LaneKey { tile: (0, 1), id: 3, path: 0 }, None), Some(2));
        assert_eq!(grown.nearest_lane(DVec3::new(0.5, 350.0, 0.0), LaneKind::Street).map(|n| n.0), Some(3));
    }

    #[test]
    fn start_grid_tracks_added_lanes_in_map_order() {
        let first = vec![straight(0.0, 0.0, 100.0, 1, (0, 0)), straight(300.0, 0.0, 100.0, 2, (1, 0))];
        let mut net = Network { lanes: first, ..Default::default() };
        net.link(1.5);
        let near = |net: &Network| net.lanes_starting_near(DVec3::ZERO, 60.0).into_iter()
            .filter(|&i| net.lanes[i].start().truncate().length() < 60.0).collect::<Vec<_>>();
        assert_eq!(near(&net), vec![0]);
        net.extend(vec![straight(25.0, 0.0, 100.0, 3, (0, 1))], 1.5);
        assert_eq!(near(&net), vec![0, 2]);
    }

    #[test]
    fn added_lanes_only_change_reach_of_their_predecessors() {
        let old = vec![
            straight(0.0, 0.0, 100.0, 1, (0, 0)),
            straight(0.0, 100.0, 200.0, 2, (0, 0)),
            straight(500.0, 0.0, 100.0, 3, (2, 0)),
        ];
        let extra = vec![straight(0.0, 200.0, 300.0, 4, (0, 1))];
        let mut grown = Network { lanes: old.clone(), ..Default::default() };
        grown.link(1.5);
        let distant_reach = grown.reach[2];
        grown.extend(extra.clone(), 1.5);
        let mut whole = Network { lanes: old.into_iter().chain(extra).collect(), ..Default::default() };
        whole.link(1.5);
        assert_eq!(grown.reach, whole.reach);
        assert_eq!(grown.reach[2], distant_reach);
    }
}

#[cfg(test)]
mod stub_links_tests {
    use super::*;

    /// A path of crossing object 9 (`source` 2) along y.
    fn piece(x: f64, y0: f64, y1: f64, path: u16) -> Lane {
        let mut l = LaneBuilder::polyline(vec![DVec3::new(x, y0, 0.0), DVec3::new(x, y1, 0.0)], LaneKind::Street, 3.0);
        l.key = Some(LaneKey { tile: (0, 0), id: 9, path });
        l.source = 2;
        l
    }

    /// BeijingCity's way through a signalled crossing: a 20 m lead-in (0), the 1 m stub
    /// with the light (1), the piece across (2) starting where the stub ends; beside it a
    /// second lead-in (3) whose lit stub (4) has an unlit 0.3 m sibling (5) from the same
    /// place, which leads into a turn (6); the straight piece (7) after the lit stub.
    fn crossing() -> Vec<Lane> {
        let mut lanes = vec![piece(0.0, 0.0, 20.0, 0)];
        let mut stub = piece(0.0, 20.0, 21.0, 1);
        stub.traffic_light = Some((0, 0));
        lanes.push(stub);
        lanes.push(piece(0.0, 21.0, 51.0, 2));
        lanes.push(piece(3.0, 0.0, 20.0, 3));
        let mut stub = piece(3.0, 20.0, 21.0, 4);
        stub.traffic_light = Some((0, 0));
        lanes.push(stub);
        lanes.push(piece(3.0, 20.0, 20.3, 5));
        let mut turn = LaneBuilder::polyline(vec![DVec3::new(3.0, 20.3, 0.0), DVec3::new(3.6, 22.0, 0.0), DVec3::new(6.0, 24.0, 0.0), DVec3::new(10.0, 25.0, 0.0)], LaneKind::Street, 3.0);
        turn.key = Some(LaneKey { tile: (0, 0), id: 9, path: 6 });
        turn.source = 2;
        lanes.push(turn);
        lanes.push(piece(3.0, 21.0, 51.0, 7));
        lanes
    }

    fn linked() -> Network {
        let mut net = Network { lanes: crossing(), ..Default::default() };
        net.link(1.5);
        net
    }

    fn merges(net: &Network, a: usize, b: usize) -> bool {
        net.crossings[a].iter().any(|c| c.other == b && c.merge)
    }

    #[test]
    fn stub_links_the_lead_in_leads_through_its_lit_stub_only() {
        let net = linked();
        assert_eq!(net.lanes[0].next, vec![1], "the 1.5 m linking distance reaches past the 1 m stub, but not round it");
        assert_eq!(net.lanes[1].next, vec![2]);
        assert_eq!(net.prev[2], vec![1]);
        // the lead-in with the sibling: the lit stub, the 0.3 m unlit stub, and the turn
        // after it (an unlit stub carries no light to be driven round)
        assert!(!net.lanes[3].next.contains(&7), "{:?}", net.lanes[3].next);
        assert!(net.lanes[3].next.contains(&4) && net.lanes[3].next.contains(&5));
        // no merge of a way with its own pieces, nor of the two stubs side by side
        for (a, b) in [(0, 1), (3, 4), (3, 5), (4, 5), (0, 2), (3, 7)] {
            assert!(!merges(&net, a, b) && !merges(&net, b, a), "{a}/{b}: {:?}", net.crossings[a]);
        }
        assert!(net.one_way_pieces(4, 5), "sibling stubs are a fork");
        assert!(net.one_way_pieces(0, 2), "through a short stub");
        assert!(!net.one_way_pieces(0, 3));
    }

    #[test]
    fn stub_links_two_ways_into_one_lane_are_still_a_merge() {
        // two lead-ins converge on one exit: the same end, starts 6 m apart
        let lanes = vec![piece(0.0, 0.0, 20.0, 0), {
            let mut l = LaneBuilder::polyline(vec![DVec3::new(6.0, 0.0, 0.0), DVec3::new(0.0, 20.0, 0.0)], LaneKind::Street, 3.0);
            l.key = Some(LaneKey { tile: (0, 0), id: 9, path: 1 });
            l.source = 2;
            l
        }, piece(0.0, 20.0, 40.0, 2)];
        let mut net = Network { lanes: lanes.clone(), ..Default::default() };
        net.link(1.5);
        assert!(merges(&net, 0, 1));
        // (BeijingCity case07 5229/5230/5231) one way's last piece is 1.1 m long and ends
        // where the other way ends; the other is linked to that piece too (it starts 1.1 m
        // before its end): no way on, the two ways' meeting place
        let slant = |x0: f64, y0: f64, x1: f64, y1: f64, path: u16| {
            let mut l = LaneBuilder::polyline(vec![DVec3::new(x0, y0, 0.0), DVec3::new(x1, y1, 0.0)], LaneKind::Street, 3.0);
            l.key = Some(LaneKey { tile: (0, 0), id: 9, path });
            l.source = 2;
            l
        };
        let lanes = vec![slant(-2.3, 0.0, -0.12, 10.9, 0), slant(-0.12, 10.9, 0.0, 12.0, 1), slant(0.0, 0.0, 0.0, 12.0, 2), piece(0.0, 12.0, 20.0, 3)];
        let mut net = Network { lanes: lanes.clone(), ..Default::default() };
        net.link(1.5);
        assert!(net.lanes[2].next.contains(&1), "linked back to the other way's last piece");
        assert!(merges(&net, 1, 2), "{:?}", net.crossings[1]);
        // (review4) the first way links past its last piece into the exit too, and the
        // other way's stream runs through that piece: their meeting place stays
        assert!(net.lanes[0].next.contains(&3) && net.lanes[0].next.contains(&1));
        assert!(merges(&net, 0, 1), "{:?}", net.crossings[0]);
    }

    #[test]
    fn stub_links_unlit_or_long_stubs_and_lanes_further_off_keep_their_links() {
        let mut lanes = crossing();
        lanes[1].traffic_light = None;
        let mut net = Network { lanes, ..Default::default() };
        net.link(1.5);
        assert_eq!(net.lanes[0].next, vec![1, 2], "an unlit stub: nothing to drive round");
        // a lit piece of 2 m is no stub
        let mut lanes = crossing();
        lanes[1] = piece(0.0, 20.0, 22.0, 1);
        lanes[1].traffic_light = Some((0, 0));
        lanes[2] = piece(0.0, 21.0, 51.0, 2);
        let a = Network { lanes: lanes.clone(), ..Default::default() };
        assert!(!a.is_lit_stub(1));
    }

    #[test]
    fn stub_links_lanes_added_later_are_linked_with_it() {
        let all = crossing();
        let mut whole = Network { lanes: all.clone(), ..Default::default() };
        whole.link(1.5);
        // the lead-ins first, the stubs and the pieces after them with a later tile (as
        // `LanesAdded` brings them: unlinked); and the other way round
        for split in [3, 1, 2, 5, 7] {
            let mut grown = Network { lanes: all[..split].to_vec(), ..Default::default() };
            grown.link(1.5);
            grown.extend(all[split..].to_vec(), 1.5);
            for i in 0..all.len() {
                let (mut g, mut w) = (grown.lanes[i].next.clone(), whole.lanes[i].next.clone());
                g.sort_unstable();
                w.sort_unstable();
                assert_eq!(g, w, "split {split}, lane {i}");
                let (mut g, mut w) = (grown.prev[i].clone(), whole.prev[i].clone());
                g.sort_unstable();
                w.sort_unstable();
                assert_eq!(g, w, "split {split}, prev of lane {i}");
            }
            for (a, b) in [(0, 1), (3, 4), (3, 5)] {
                assert!(!merges(&grown, a, b) && !merges(&grown, b, a), "split {split}: {a}/{b}");
            }
        }
    }

    #[test]
    fn stub_links_applied_to_a_linked_network_is_linking_with_it() {
        // the crossing, and beside it two ways of the same object that run into one exit
        // (8 and 9: ends together, starts 6 m apart, a merge of two streams)
        let with_streams = || {
            let mut lanes = crossing();
            lanes.push(piece(20.0, 0.0, 20.0, 8));
            let mut slant = LaneBuilder::polyline(vec![DVec3::new(26.0, 0.0, 0.0), DVec3::new(20.0, 20.0, 0.0)], LaneKind::Street, 3.0);
            slant.key = Some(LaneKey { tile: (0, 0), id: 9, path: 9 });
            slant.source = 2;
            lanes.push(slant);
            lanes.push(piece(20.0, 20.0, 40.0, 10));
            let mut net = Network { lanes, ..Default::default() };
            net.link(1.5);
            net
        };
        let on = with_streams();
        assert!(merges(&on, 8, 9) && merges(&on, 9, 8), "two streams into one exit: {:?}", on.crossings[8]);
        // the network as a recording from before the lit stubs has it: linked round them
        // (the 0.3 m sibling is linked to the lit stub beside it, which starts 0.3 m back,
        // and past it to the straight piece: round the light too), the lead-in a merge of
        // its own stub and the other lead-in of its sibling stub; put back by hand
        let mut net = with_streams();
        for (a, b) in [(0, 2), (3, 7), (5, 7)] {
            net.lanes[a].next.push(b);
            net.prev[b].push(a);
        }
        for (i, j) in [(0, 1), (3, 5)] {
            let (li, lj) = (net.lanes[i].length(), net.lanes[j].length());
            net.crossings[i].push(Crossing { other: j, at: li, other_at: lj, merge: true, before: 2.0, after: 0.0, other_before: lj.min(2.0), other_after: 0.0, side: false, outside: false });
            net.crossings[j].push(Crossing { other: i, at: lj, other_at: li, merge: true, before: lj.min(2.0), after: 0.0, other_before: 2.0, other_after: 0.0, side: false, outside: false });
            net.conflicts[i].push(j);
            net.conflicts[j].push(i);
        }
        assert!(merges(&net, 0, 1) && merges(&net, 3, 5));
        let (links, pairs) = net.apply_stub_links();
        assert_eq!(links, vec![(0, 2), (3, 7), (5, 7)]);
        // the lead-in / stub merges go (each pair once), the two streams' merge stays
        assert_eq!(pairs, vec![(0, 1), (3, 5)]);
        for (a, b) in [(0, 1), (3, 5)] {
            assert!(!merges(&net, a, b) && !merges(&net, b, a), "{a}/{b}");
            assert!(!net.conflicts[a].contains(&b) && !net.conflicts[b].contains(&a), "{a}/{b}");
        }
        assert!(merges(&net, 8, 9) && merges(&net, 9, 8) && net.conflicts[8].contains(&9));
        // and the network is as linked with them
        for i in 0..net.lanes.len() {
            assert_eq!(net.lanes[i].next, on.lanes[i].next, "lane {i}");
            assert_eq!(net.prev[i], on.prev[i], "prev of lane {i}");
            assert_eq!(net.crossings[i], on.crossings[i], "crossings of lane {i}");
            assert_eq!(net.conflicts[i], on.conflicts[i], "conflicts of lane {i}");
        }
        // on a network linked with them it does nothing
        let mut again = with_streams();
        assert_eq!(again.apply_stub_links(), (Vec::new(), Vec::new()));
        for i in 0..again.lanes.len() {
            assert_eq!(again.lanes[i].next, on.lanes[i].next, "lane {i}");
            assert_eq!(again.crossings[i], on.crossings[i], "crossings of lane {i}");
        }
    }
}

/// A `[traffic_light_stop]` / `[traffic_light_jump]` of a light program (`TAmpelStop`:
/// ampel, time, jumptime, ifAnf). When the cycle clock reaches `time` and the condition
/// holds, the clock waits there (stop) or continues at `jump_to` (jump). With `if_request`
/// set the condition is "nobody is asking at `light`": the stock railway crossings keep the
/// road green at 1 s until a train approaches, the bus loop of Heerstraße skips its bus
/// phase when no bus is waiting, the depot gates jump over their phases when nothing comes.
/// Without it the condition is "somebody is asking": a gate stays green while buses keep
/// coming, a barrier stays closed while the train is still there.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LightStop {
    pub light: usize,
    pub time: f32,
    pub if_request: bool,
    pub jump_to: Option<f32>,
}

/// How far before its stop line a vehicle asks a light for green when the program gives no
/// `[approachdist]` (m): about four seconds at town speed.
pub const DEFAULT_APPROACH: f32 = 50.0;

/// The `TrafficLightPhase` of an object no crossing light drives: a signal without a
/// `[varparent]`, one whose crossing has no program, any other scripted object. Omsi.exe
/// binds every placed object's `TrafficLightPhase` and `TrafficLightApproach` to a light
/// of its parent crossing ("RefreshAmpelParenting", 0x77d460) and, failing that, to a
/// shared dummy variable in .bss (0x8619a4), which reads 0: red, not "no light".
pub const UNLINKED_PHASE: i32 = 0;

/// Traffic light program of a crossing object instance (`TAmpelGroup`): its lights'
/// phases and one cycle clock that runs on game time. The clock is state, not a function
/// of the time of day, because the program may wait at a stop point or jump.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrafficLightController {
    /// Per light: phases (state, duration).
    pub lights: Vec<Vec<(i32, f32)>>,
    /// Cycle length (`[traffic_lights_group]`); 0 = the longest light's phases.
    pub cycle: f32,
    pub offset: f32,
    /// `[approachdist]` per light.
    pub approach: Vec<Option<f32>>,
    pub stops: Vec<LightStop>,
    /// Position in the cycle (s).
    pub time: f64,
    /// Per light: a vehicle (or a pedestrian) is asking for it this frame.
    pub request: Vec<bool>,
    /// The clock waits at a stop point.
    pub held: bool,
    /// Points already let past at this instant. Remember all of them so coincident
    /// inactive points cannot keep making one another eligible again.
    passed: Vec<usize>,
    /// A short backwards jump has replayed its stretch once. Do not take it again before
    /// the clock has moved past its source time.
    rewound: Option<usize>,
    started: bool,
}

impl TrafficLightController {
    pub fn new(lights: Vec<Vec<(i32, f32)>>, cycle: f32) -> TrafficLightController {
        let n = lights.len();
        TrafficLightController { lights, cycle, offset: 0.0, approach: vec![None; n], stops: Vec::new(), time: 0.0, request: vec![false; n], held: false, passed: Vec::new(), rewound: None, started: false }
    }

    /// From the `[traffic_light]` program of a crossing object: (per light: name, phases
    /// as (state, seconds), `[approachdist]`), the cycle, the stop and jump points.
    pub fn from_program(lights: Vec<(Vec<(i32, f32)>, Option<f32>)>, cycle: Option<f32>, stops: &[[f32; 3]], jumps: &[[f32; 4]]) -> TrafficLightController {
        let approach = lights.iter().map(|l| l.1).collect();
        let mut c = TrafficLightController::new(lights.into_iter().map(|l| l.0).collect(), cycle.unwrap_or(0.0));
        c.approach = approach;
        for s in stops {
            c.stops.push(LightStop { light: s[0].max(0.0) as usize, time: s[1], if_request: s[2] > 0.5, jump_to: None });
        }
        for j in jumps {
            c.stops.push(LightStop { light: j[0].max(0.0) as usize, time: j[1], if_request: j[2] > 0.5, jump_to: Some(j[3]) });
        }
        c
    }

    /// Length of the cycle in seconds.
    pub fn cycle_len(&self) -> f64 {
        if self.cycle > 0.0 {
            return self.cycle as f64;
        }
        self.lights.iter().map(|p| p.iter().map(|x| x.1).sum::<f32>()).fold(0.0f32, f32::max).max(1.0) as f64
    }

    /// Set the clock from the time of day (s) the first time the program runs: crossings
    /// with the same cycle length then run in step, as a coordinated street would.
    pub fn start(&mut self, day_time: f64) {
        if !self.started {
            self.time = (day_time + self.offset as f64).rem_euclid(self.cycle_len());
            self.started = true;
        }
    }

    /// Request distance of light `i` (m).
    pub fn approach_dist(&self, i: usize) -> f32 {
        self.approach.get(i).copied().flatten().unwrap_or(DEFAULT_APPROACH)
    }

    /// Run the cycle clock on by `dt` seconds of game time, honouring the stop and jump
    /// points with this frame's requests (`request`, set by the caller before).
    pub fn advance(&mut self, dt: f32) {
        let cycle = self.cycle_len();
        let mut left = dt.max(0.0) as f64;
        self.held = false;
        let clear_rewind = |this: &mut Self, move_by: f64| {
            let Some(k) = this.rewound else { return };
            let source = this.stops[k].time as f64;
            if ((this.time - source).abs() < 1e-6 && move_by > 1e-6)
                || (this.time < source && this.time + move_by > source + 1e-6)
            {
                this.rewound = None;
            }
        };
        // Allow every coincident inactive point without losing this frame's time,
        // but still bound authored jump loops.
        for _ in 0..16 + self.stops.len() {
            let mut best: Option<(usize, f64)> = None;
            for (k, p) in self.stops.iter().enumerate() {
                if self.rewound == Some(k) {
                    continue;
                }
                let d = (p.time as f64 - self.time).rem_euclid(cycle);
                let d = if d > cycle - 1e-6 { 0.0 } else { d };
                if d < 1e-6 && self.passed.contains(&k) {
                    continue;
                }
                if d <= left && best.map(|b| d < b.1).unwrap_or(true) {
                    best = Some((k, d));
                }
            }
            let Some((k, d)) = best else {
                if left > 0.0 {
                    clear_rewind(self, left);
                    self.passed.clear();
                }
                self.time = (self.time + left).rem_euclid(cycle);
                return;
            };
            if d > 1e-6 {
                self.passed.clear();
            }
            clear_rewind(self, d);
            self.time = (self.time + d).rem_euclid(cycle);
            left -= d;
            let p = self.stops[k];
            let asked = self.request.get(p.light).copied().unwrap_or(false);
            let active = if p.if_request { !asked } else { asked };
            if !active {
                self.passed.push(k);
                continue;
            }
            match p.jump_to {
                Some(to) => {
                    // A jump a couple of seconds back extends the current phase. It is not
                    // a loop: after replaying that small stretch, continue through it.
                    self.rewound = (to > 1e-6 && to < p.time - 1e-6).then_some(k);
                    self.time = (to as f64).rem_euclid(cycle);
                    self.passed.clear();
                    if left <= 0.0 {
                        return;
                    }
                }
                None => {
                    self.time = p.time as f64;
                    self.held = true;
                    self.passed.clear();
                    return;
                }
            }
        }
    }

    /// State of light `i` at position `x` of the cycle (s).
    pub fn state_at(&self, i: usize, x: f64) -> i32 {
        let phases = match self.lights.get(i) {
            Some(p) if !p.is_empty() => p,
            _ => return 6,
        };
        let mut x = x.rem_euclid(self.cycle_len()) as f32;
        for (state, dur) in phases {
            if x < *dur {
                return *state;
            }
            x -= dur;
        }
        // after the last phase its state holds until the cycle starts again
        phases.last().map(|p| p.0).unwrap_or(0)
    }

    /// Current state of light `i` (the `TrafficLightPhase` value of its lamps).
    pub fn state(&self, i: usize) -> i32 {
        self.state_at(i, self.time)
    }

    /// Seconds until light `i` lets vehicles go (0 while it does), if the clock keeps
    /// running; None when it never does within a cycle.
    pub fn time_until_go(&self, i: usize) -> Option<f32> {
        if Self::allows_go(self.state(i)) {
            return Some(0.0);
        }
        let cycle = self.cycle_len();
        let mut t = 0.25;
        while t <= cycle {
            if Self::allows_go(self.state_at(i, self.time + t)) {
                return Some(t as f32);
            }
            t += 0.25;
        }
        None
    }

    /// Seconds until light `i` leaves the state it shows now while the clock runs on
    /// (phases in a row with the same state count as one; a stop point may hold it
    /// longer). A pedestrian starts across only when the green lasts.
    pub fn remaining(&self, i: usize) -> f32 {
        let phases = match self.lights.get(i) {
            Some(p) if !p.is_empty() => p,
            _ => return f32::INFINITY,
        };
        let cycle = self.cycle_len() as f32;
        let x = self.time.rem_euclid(self.cycle_len()) as f32;
        // the schedule of one cycle: (state, start, end); a last phase of length 0 (or a
        // cycle longer than the phases) lasts until the cycle ends
        let mut spans: Vec<(i32, f32, f32)> = Vec::new();
        let mut start = 0.0;
        for (k, (state, dur)) in phases.iter().enumerate() {
            let end = if k + 1 == phases.len() { cycle.max(start + dur) } else { start + dur };
            spans.push((*state, start, end));
            start = end;
        }
        let Some(k) = spans.iter().position(|s| x < s.2).or(Some(spans.len() - 1)) else { return f32::INFINITY };
        let state = spans[k].0;
        if spans.iter().all(|s| s.0 == state) {
            return f32::INFINITY;
        }
        // run on through the following phases (wrapping round) while the state stays
        let mut left = spans[k].2 - x;
        let mut j = (k + 1) % spans.len();
        while spans[j].0 == state && j != k {
            left += spans[j].2 - spans[j].1;
            j = (j + 1) % spans.len();
        }
        left
    }

    /// OMSI light states as the stock lamp scripts read them: 0..2 red, 3..5 red and
    /// yellow, 6..8 green (8: the GDR's green with yellow), 9..11 yellow, 12 and above dark.
    /// (The code once read 8 as yellow and 9 as all-red: every stock program's yellow, 9,
    /// showed red, so the lights went from green straight to red.)
    pub fn aspect(state: i32) -> Aspect {
        match state {
            0..=2 => Aspect::Red,
            3..=5 => Aspect::RedYellow,
            6 | 7 => Aspect::Green,
            8 => Aspect::GreenYellow,
            9..=11 => Aspect::Yellow,
            _ => Aspect::Dark,
        }
    }

    /// Lights `a` and `b` of this program never let vehicles in at the same time: at no
    /// point of the cycle (sampled every quarter second) do both show anything but red or
    /// red-and-yellow (yellow counts as letting in: a driver may still take the line). Two
    /// signal groups of conflicting streams; a light is never exclusive of itself.
    pub fn never_together(&self, a: usize, b: usize) -> bool {
        if a == b || a >= self.lights.len() || b >= self.lights.len() {
            return false;
        }
        let lets_in = |s: i32| !matches!(Self::aspect(s), Aspect::Red | Aspect::RedYellow);
        let cycle = self.cycle_len();
        let mut x = 0.0;
        while x < cycle {
            if lets_in(self.state_at(a, x)) && lets_in(self.state_at(b, x)) {
                return false;
            }
            x += 0.25;
        }
        true
    }

    /// May a vehicle drive over the stop line now (not counting yellow, which is the
    /// driver's decision)?
    pub fn allows_go(state: i32) -> bool {
        matches!(Self::aspect(state), Aspect::Green | Aspect::Dark)
    }

    /// Lamp variables (red, yellow, green) of a state, for lamp objects without a script
    /// (the rules of the stock `ampel1_ddr.osc`, which agree with `ampel1.osc` for the
    /// states the West Berlin programs use).
    pub fn lamps(state: i32) -> (bool, bool, bool) {
        (matches!(state, 0..=5), matches!(state, 3..=5 | 8..=11), matches!(state, 6..=8))
    }
}

/// What a traffic light shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aspect {
    Red,
    RedYellow,
    Green,
    GreenYellow,
    Yellow,
    Dark,
}

/// How far ahead (m) a car decides which way it goes at least: far enough to signal a turn in
/// good time and for the steering to look beyond the junction; [round 7 A8] and as far as its
/// junction claim reaches (`claim_horizon`), so that the junction is seen there.
const PLAN_AHEAD: f32 = 90.0;
/// At most this many lanes are planned ahead (junction pieces can be a few metres long).
// [ROLLBACK lights-130] 20 lanes (was 10): on a road of short pieces ten lanes were not
// ninety metres, and a light further ahead was not seen in time to stop; [round 7 A8] 40 for
// the claim horizon's 220 m
const PLAN_LANES: usize = 40;

/// [round 7 A8] How far before its first meeting place (m, from its front) a vehicle at
/// `speed` m/s files its junction claim - eight seconds, at least 60 m, at most 220 m - and
/// so how far ahead it plans its way and looks for the junction.
pub fn claim_horizon(speed: f32) -> f32 {
    (8.0 * speed + 30.0).clamp(60.0, 220.0)
}
/// Linked lanes may end up to 1.5 m apart; over this many metres on either side of a joint
/// the way is bent so that it has no step (half the gap on each side).
const JOINT_BLEND: f32 = 6.0;
/// Seconds of signalling before a lane change begins to move sideways.
// [ROLLBACK lanechange-108] two seconds (was 1.2) before the car moves over
pub const SIGNAL_BEFORE_CHANGE: f32 = 2.0;

/// One AI vehicle moving on the network: where it is along its lanes and which way it is
/// going. The body that follows this way is `ai_motion::AiBody`. (Read from a recording, a
/// field it lacks takes `AiState::default()`'s: `omsi_traffic::record::VERSION`'s rule.)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct AiState {
    pub traffic_pool: Option<(usize, std::sync::Arc<Vec<i32>>)>,
    /// The vehicle's `[ai_veh_type]` (0 car, 1 taxi, 2 bus, 3 truck; -1 a timetable bus):
    /// which lanes it may take, see [`Lane::allows`].
    pub veh_type: i32,
    pub lane: usize,
    pub s: f32,
    pub speed: f32,
    pub max_speed_kmh: f32,
    pub accel: f32,
    pub decel: f32,
    pub length: f32,
    pub rng: u64,
    pub blinker: i32,
    pub braking: bool,
    /// Distance travelled, for wheel animation.
    pub odometer: f32,
    /// Lane chosen for after the current one.
    pub planned_next: Option<usize>,
    /// The lanes after `planned_next`, decided `PLAN_AHEAD` metres in advance.
    pub ahead: Vec<usize>,
    /// During a lane change: the way on from the lane the car is moving over to, chosen
    /// when the change begins, so that the steering and the speed for the bends already
    /// know it.
    pub change_plan: Vec<usize>,
    /// The lane the car came from (the way behind it, for the rear axle).
    pub prev_lane: Option<usize>,
    /// Fixed lane sequence (timetable track); empty = wander randomly.
    pub route: Vec<usize>,
    /// Position in `route` of the current lane.
    pub route_index: usize,
    /// Lane change in progress.
    pub change: Option<LaneChange>,
    /// Seconds until the next lane change may start.
    pub change_cooldown: f32,
    /// Lateral offset from the lane (m, positive = right): bus bays, swerving round a
    /// parked car. It moves towards `lateral_target` along an S-curve over the distance
    /// driven (`lateral_ramp`), so that the car is straight again when it gets there and
    /// nothing moves while it stands.
    pub lateral: f32,
    pub lateral_target: f32,
    /// The S-curve in progress: (from, to, odometer at its start, its length in m).
    pub lateral_ramp: (f32, f32, f32, f32),
    /// [ROLLBACK deform-105] A way bent round something, laid out by the traffic planner:
    /// (odometer, offset) points, every metre or so; the offset between them is straight
    /// between the points. While the odometer is short of the last point this is the offset
    /// (the S-curve above holds from there on, and the planner leaves it at 0 to 0).
    pub lateral_curve: Vec<(f32, f32)>,
    /// Indicator the driver sets for a manoeuvre of its own (pulling away from a stop),
    /// held while `signal_time` runs.
    pub signal: i32,
    pub signal_time: f32,
    /// Sideways acceleration the driver accepts in a bend (m/s²): the speed through a
    /// curve of radius r is at most sqrt(this × r).
    pub lat_accel: f32,
    /// The driver (`TPathInfo`'s rowdy_factor & co): how fast they like to go relative to
    /// the limit, the time gap they keep to the car ahead (s), the distance they stop
    /// behind it (m) and how long they take to move off when the way clears (s) (the gap
    /// kept to a foe at a junction is `reservation::JM_TIMEGAP_MINOR`). `accel` is how hard they pull
    /// away and `decel` how hard they like to brake.
    pub desire: f32,
    pub headway: f32,
    pub min_gap: f32,
    pub reaction: f32,
    /// From the vehicle's origin (the lane position `s`) to its front and rear bumper (m).
    pub front: f32,
    pub rear: f32,
    /// Standing still and held there (by a car ahead, a light, a stop): moving off again
    /// starts after `reaction`; `start_timer` counts it down.
    pub held: bool,
    pub start_timer: f32,
    /// Acceleration of the last step (m/s²).
    pub acc: f32,
    /// At most this much acceleration for now (m/s²): edging out round something standing
    /// close ahead.
    pub accel_cap: Option<f32>,
    /// A frozen lane change was settled where the body stood (`settle_change`) and the
    /// offset it left has not eased off yet: standing, the traffic eases it in place. Only
    /// `settle_change` sets it (a fresh car, a pull-out from a bay or a parking strip, a
    /// swerve have offsets too, and must not slide).
    #[serde(default)]
    pub settled_offset: bool,
}

/// What a car keeps its distance to: the gap from its front bumper to the thing (m), how
/// fast that is moving along the car's way (m/s) and how it is speeding up (m/s²).
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct Lead {
    pub gap: f32,
    pub speed: f32,
    pub acc: f32,
}

impl Lead {
    /// The nearer (more constraining) of two.
    pub fn min(a: Option<Lead>, b: Option<Lead>) -> Option<Lead> {
        match (a, b) {
            (Some(x), Some(y)) => Some(if y.gap < x.gap { y } else { x }),
            (x, None) => x,
            (None, y) => y,
        }
    }
}

/// Seconds a vehicle needs to cover `dist` metres from speed `v`, speeding up at `a` to at
/// most `v_max` (the earliest it can be there when nothing holds it back).
pub fn arrival_time(dist: f32, v: f32, a: f32, v_max: f32) -> f32 {
    if dist <= 0.0 {
        return 0.0;
    }
    let (v, a) = (v.max(0.0), a.max(0.05));
    let v_max = v_max.max(v).max(0.1);
    // speeding up to v_max takes (v_max - v) / a seconds and that much road
    let t_up = (v_max - v) / a;
    let d_up = (v + v_max) * 0.5 * t_up;
    if dist <= d_up {
        (-v + (v * v + 2.0 * a * dist).sqrt()) / a
    } else {
        t_up + (dist - d_up) / v_max
    }
}

/// The road a lane change at `speed` (m/s) moves over along (m): 2.8 to 4 s at that speed,
/// at least 12 m ([ROLLBACK lanechange-108] a longer, gentler move: three to four seconds).
pub fn change_length(speed: f32) -> f32 {
    (speed * (4.0 - speed * 0.04).clamp(2.8, 4.0)).max(12.0)
}

/// The road (m) an S-curve sideways by `offset` metres takes (`AiState::lateral_ramp`): 8 m
/// per metre across, 8 m at least, 30 m at most.
pub fn s_curve_len(offset: f32) -> f32 {
    (offset.abs() * 8.0).clamp(8.0, 30.0)
}

/// Why a lane is closed to a vehicle (`open_to`; the debug overlay).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Closed {
    /// Not a lane of the vehicle's kind (a footpath, a rail, a flight path for a car).
    Kind,
    /// `[rule] no_cars`, `bus`, `trucks` by the vehicle's `[ai_veh_type]`.
    Rules,
    /// No random traffic of this vehicle's group drives here (`trafficdensity 0`).
    Density,
}

/// [round 7 A1] Is `lane` open to a vehicle with the way `st`, driving on lanes of `kind`
/// (street, rail, air), for driving along? One rule for the population, the way on at a
/// fork, a lane change, a pass on the oncoming lane and the lane beside
/// (`omsi_traffic::access`). A timetable vehicle on its route is open anywhere its route
/// goes; this answers for the driver's own choices.
pub fn open_to(lane: &Lane, st: &AiState, kind: LaneKind) -> Result<(), Closed> {
    if lane.kind != kind || kind == LaneKind::Sidewalk {
        return Err(Closed::Kind);
    }
    // (a rail or a flight path is its vehicle's whatever the road rules and the traffic
    // densities say: those are the street traffic's)
    if kind != LaneKind::Street {
        return Ok(());
    }
    if !lane.allows(st.veh_type) {
        return Err(Closed::Rules);
    }
    // a scheduled bus goes where its route goes; everybody else needs the lane to carry
    // traffic of their group
    if st.route.is_empty() {
        // (a vehicle of no group is a normal car: where the path names its groups, the
        // first group's share counts, not the most any group gets - `density` is the
        // greatest over all groups, so that a lane stays open to whoever is asked for)
        let density = match st.traffic_pool.as_ref() {
            Some((pool, defaults)) => lane.pool_density(defaults, *pool),
            None if lane.group_density.is_empty() => lane.density,
            None => lane.pool_density(&[1], 0),
        };
        if density <= 0.0 {
            return Err(Closed::Density);
        }
    }
    // (no test on the path's width: a tenth of the stock maps' street paths are drawn a
    // metre wide where the lane is a full one - Spandau's ten-lane Altstädter Ring is
    // all 1 m paths - and BeijingCity's bus lane is as wide as its cycle lane. What keeps
    // a car off a cycle lane beyond a strip is the strip: `Lane::beside_each_other`.)
    Ok(())
}

/// [fix pass 2, 2; pass 3, 9] The swept path (`Network::compile_swept`), from the stock
/// timetable bus (`Vehicles/NEOMAN_Overhaul_V3/_12a_MAN_A21_2D.bus`): its length, the
/// `[boundingbox]`'s 11.8 m ...
pub const SWEPT_BUS_LENGTH: f32 = 11.8;
/// ... its wheelbase, the two `[newachse]` `achse_long` 3.149 and -2.665 m apart, for the
/// off-tracking wheelbase^2 / 2R in a bend of radius R (the piece's own curvature) ...
pub const SWEPT_WHEELBASE: f32 = 3.149 + 2.665;
/// ... and the tightest turn it makes, the 12.5 m outer turning circle every bus must manage
/// (Directive 97/27/EC, annex I 7.6), which caps the off-tracking.
pub const SWEPT_MIN_RADIUS: f32 = 12.5;

/// [round 7 A1] Ways on chosen by a tier below "open to me and leading on"
/// (`AiState::choose_after`), per tier: [open but every way on from it closed, closed by my
/// group's density but driven by another group, driven by no group, closed by the rules but
/// not `no_cars`, `no_cars`, none at all]. Per call (a plan is extended frame after frame).
/// A diagnostic for `traffic_cases ab` (whole process).
pub static FORK_FALLBACKS: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

/// Hardest braking of an AI driver (m/s²): an emergency stop.
pub const MAX_BRAKE: f32 = 8.0;
/// Gap a car leaves before a stop line or a stop point (m).
const STOP_LINE_GAP: f32 = 0.6;
/// [Q8-1] A car backing up out of a packed queue does so at walking pace (m/s,
/// `AiState::reverse_step`): [fix pass 2] 5 km/h, which the Austrian Supreme Court's decision
/// OGH 23.03.2007, 2Ob262/05a equates with "Schrittgeschwindigkeit" in the lower court's findings
/// (https://ris.bka.gv.at/Dokumente/Justiz/JJT_20070323_OGH0002_0020OB00262_05A0000_000/JJT_20070323_OGH0002_0020OB00262_05A0000_000.html),
/// not as a definition; for Germany de.wikipedia ("Schrittgeschwindigkeit") gives the courts'
/// 7-10 km/h. (SUMO has no reversing of road vehicles.)
pub const REVERSE_SPEED: f32 = 5.0 / 3.6;

/// A lane change: the car moves over from its lane to `to` along `length` metres of road
/// (by distance, not by time: a car that has to stop halfway stands still, and so does its
/// way).
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct LaneChange {
    pub to: usize,
    /// Progress 0..1.
    pub t: f32,
    pub length: f32,
    /// Position along `to`.
    pub s_to: f32,
    /// 1 left, 2 right.
    pub dir: i32,
    /// Seconds of indicating left before the car starts to move over.
    pub wait: f32,
    /// Pulling out from behind something standing: the car moves off sideways from a
    /// standstill, and what it leaves behind in its lane no longer holds it.
    pub bypass: bool,
}

/// A vehicle's heading is that of the chord of its way this far behind and ahead of it (m,
/// `Simulation::update_positions`).
pub const HEADING_CHORD: f32 = 1.5;

/// Smoothstep 0..1.
pub fn smooth01(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The tangents of the lane segment `Lane::hermite` used last ((lane, segment), at its start,
/// at its end): a run of points along a lane mostly falls in one segment, and the two sines
/// and cosines were most of a point's cost.
#[derive(Debug, Clone, Copy, Default)]
struct Tangents(Option<((usize, usize), DVec2, DVec2)>);

/// A short sequence of lanes the way runs along, with the lane the car is on at `cur`.
struct LaneSeq {
    lanes: [usize; PLAN_LANES + 2],
    n: usize,
    cur: usize,
}

impl LaneSeq {
    /// Gap from the end of `lanes[i]` to the start of `lanes[i + 1]`, if they are linked
    /// (a gap of a few metres means they are not one road, and nothing is bent).
    fn gap(&self, net: &Network, i: usize) -> Option<DVec3> {
        let (a, b) = (&net.lanes[self.lanes[i]], &net.lanes[self.lanes[i + 1]]);
        let g = b.start() - a.end();
        (g.length() < 3.0).then_some(g)
    }

    /// Point and heading `u` metres into `lanes[i]` (beyond the ends of the first and last
    /// lane: straight on), with the joints on both sides smoothed.
    fn at(&self, net: &Network, i: usize, u: f32, cache: &mut Tangents) -> (DVec3, f32) {
        let l = &net.lanes[self.lanes[i]];
        let (mut p, h) = l.at_ext_with(u, cache);
        let w = |x: f32| 1.0 - smooth01(x / JOINT_BLEND);
        if i > 0 {
            if let Some(g) = self.gap(net, i - 1) {
                p -= g * (0.5 * w(u.max(0.0))) as f64;
            }
        }
        if i + 1 < self.n {
            if let Some(g) = self.gap(net, i) {
                p += g * (0.5 * w((l.length() - u).max(0.0))) as f64;
            }
        }
        (p, h)
    }

    /// Lane index in `lanes` and distance along it, `d` metres from distance `s` of the
    /// current lane.
    fn locate(&self, net: &Network, s: f32, d: f32) -> (usize, f32) {
        let mut i = self.cur;
        let mut u = s + d;
        while u < 0.0 && i > 0 {
            i -= 1;
            u += net.lanes[self.lanes[i]].length();
        }
        while i + 1 < self.n && u > net.lanes[self.lanes[i]].length() {
            u -= net.lanes[self.lanes[i]].length();
            i += 1;
        }
        (i, u)
    }

    /// Point and heading `d` metres from distance `s` of the current lane.
    fn point(&self, net: &Network, s: f32, d: f32) -> (DVec3, f32) {
        self.point_with(net, s, d, &mut Tangents::default())
    }

    /// `point`, the tangents of the lane segment kept in `cache` for the next one.
    fn point_with(&self, net: &Network, s: f32, d: f32, cache: &mut Tangents) -> (DVec3, f32) {
        let (i, u) = self.locate(net, s, d);
        self.at(net, i, u, cache)
    }

    /// The heading of `point`, without the point (the joints' smoothing moves the point only).
    fn heading(&self, net: &Network, s: f32, d: f32) -> f32 {
        let (i, u) = self.locate(net, s, d);
        net.lanes[self.lanes[i]].heading_at_ext(u)
    }
}

impl Default for AiState {
    /// A vehicle on lane 0 at its start, standing (`AiState::new`): what a field a recording
    /// lacks is read as.
    fn default() -> AiState {
        AiState::new(0, 0.0, 1)
    }
}

impl AiState {
    pub fn new(lane: usize, s: f32, seed: u64) -> AiState {
        AiState { traffic_pool: None, veh_type: 0, lane, s, speed: 0.0, max_speed_kmh: 50.0, accel: 1.2, decel: 3.0, length: 5.0, rng: seed | 1, blinker: 0, braking: false, odometer: 0.0, planned_next: None, ahead: Vec::new(), change_plan: Vec::new(), prev_lane: None, route: Vec::new(), route_index: 0, change: None, change_cooldown: 5.0, lateral: 0.0, lateral_target: 0.0, lateral_ramp: (0.0, 0.0, 0.0, 1.0), lateral_curve: Vec::new(), signal: 0, signal_time: 0.0, lat_accel: 2.8, desire: 1.0, headway: 1.4, min_gap: 2.0, reaction: 0.7, front: 2.5, rear: 2.5, held: false, start_timer: 0.0, acc: 0.0, accel_cap: None, settled_offset: false }
    }

    fn rand(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// The lanes ahead, nearest first: `planned_next`, then the rest of the plan.
    pub fn upcoming(&self) -> impl Iterator<Item = usize> + '_ {
        self.planned_next.into_iter().chain(self.ahead.iter().copied())
    }

    /// A random way on from the end of `lane`: one open to this vehicle by the access rule
    /// (`open_to`: the path's rules, `[rule] no_cars`/`bus`/`trucks` by its type, and the
    /// traffic density of its group - filtering them only at spawn still let cars turn into a
    /// pedestrian street or a depot yard from next door); a car of a traffic pool takes them
    /// by its pool's density. [round 7 A1] The ways on are ranked, the best tier taken (the
    /// lower ones counted, `FORK_FALLBACKS`): open to me and leading on (a dead end too); open
    /// to me but every way on from it closed (SUMO's bestLanes look-ahead, one step); closed
    /// by my group's density but driven by another group; driven by no group; closed by the
    /// rules but not `no_cars`; `no_cars` - a car already on a lane whose only continuation is
    /// closed drives on through it, as before round 7, and is never taken off moving; only a
    /// lane with no way of its kind at all ends the network for it. A way that ends the
    /// network just ahead (`Network::reach`) is taken like any other; [round 9 A4] one whose
    /// road ends at the loaded tiles' edge (`Network::ends_at_loaded_edge`) only when no other
    /// way of its tier is left.
    fn choose_after(
        &mut self,
        net: &Network,
        lane: usize,
        avoid: &dyn Fn(usize) -> bool,
    ) -> Option<usize> {
        let l = &net.lanes[lane];
        // [ROLLBACK cycle-135] the ways that lead back round a loop of short lanes are not
        // among the choices (the others are: a street goes on where it has a way on)
        let next: Vec<usize> = l.next.iter().copied().filter(|&n| !avoid(n)).collect();
        let kind = l.kind;
        let closed = |n: usize| open_to(&net.lanes[n], self, kind).err();
        // (SUMO's bestLanes, one step: a lane whose ways on are all closed to me is no way on
        // for me - a car entered X10's 20754 and 20737, whose only continuation is no_cars)
        let leads_on = |n: usize| {
            let ln = &net.lanes[n];
            ln.next.is_empty() || ln.next.iter().any(|&m| closed(m).is_none())
        };
        let tier = |n: usize| -> usize {
            match closed(n) {
                None if leads_on(n) => 0,
                None => 1,
                Some(Closed::Density) if net.lanes[n].density > 0.0 => 2,
                Some(Closed::Density) => 3,
                Some(Closed::Rules) if !net.lanes[n].no_cars => 4,
                Some(Closed::Rules) => 5,
                Some(Closed::Kind) => 6,
            }
        };
        let best = next.iter().map(|&n| tier(n)).min().unwrap_or(6);
        let mut choices: Vec<usize> = if best < 6 { next.iter().copied().filter(|&n| tier(n) == best).collect() } else { Vec::new() };
        // [round 9 A4] not into the edge of the loaded tiles when another way of the same tier
        // leads on: a random car has no trip, and the loaded-tile edge is not a place on the map,
        // only where what is loaded ends - a car turning in there stood in a queue that reached
        // back 400 m towards the player (r9 jzk). (OMSI geometry, not SUMO: SUMO's vehicles have
        // routes to real destinations and never aim at the network's edge.) With no other way it
        // drives on as before. The flag follows the tiles (`Lane::loaded_edge`, set by the game
        // as they load), so a branch opens again once its tile is loaded.
        if choices.len() > 1 {
            let open: Vec<usize> = choices.iter().copied().filter(|&n| !net.ends_at_loaded_edge(n)).collect();
            if !open.is_empty() && open.len() < choices.len() {
                choices = open;
            }
        }
        let weighted = best == 0 && self.traffic_pool.is_some();
        if best > 0 && !next.is_empty() {
            FORK_FALLBACKS[best.min(6) - 1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if choices.is_empty() {
            None
        } else if best == 2 {
            // ([fix pass 2, 7] lanes another group drives: by how much traffic they carry, a
            // lane at 0.001 not as likely as one at 1.0)
            let total: f32 = choices.iter().map(|&n| net.lanes[n].density).sum();
            let mut pick = (self.rand() >> 32) as f32 / (u32::MAX as f32 + 1.0) * total;
            for &n in &choices {
                if pick < net.lanes[n].density {
                    return Some(n);
                }
                pick -= net.lanes[n].density;
            }
            choices.last().copied()
        } else {
            if let Some((pool, defaults)) = self.traffic_pool.clone().filter(|_| weighted) {
                let total: f32 = choices.iter().map(|&n| net.lanes[n].pool_density(&defaults, pool)).sum();
                let mut pick = (self.rand() >> 32) as f32 / (u32::MAX as f32 + 1.0) * total;
                for &n in &choices {
                    let weight = net.lanes[n].pool_density(&defaults, pool);
                    if pick < weight { return Some(n); }
                    pick -= weight;
                }
                choices.last().copied()
            } else { Some(choices[(self.rand() % choices.len() as u64) as usize]) }
        }
    }

    /// How far ahead (m) the way is planned: `PLAN_AHEAD`, or the claim horizon at this speed.
    fn plan_ahead(&self) -> f32 {
        PLAN_AHEAD.max(claim_horizon(self.speed) + self.front)
    }

    /// Decide the way on: the lane after the current one and enough lanes after it to
    /// cover `PLAN_AHEAD` metres (a timetable route simply is that plan).
    pub fn plan_next(&mut self, net: &Network) {
        if !self.route.is_empty() {
            // a route step onto a lane beside this one is a lane change, not the way on
            let mut k = self.route_index + 1;
            if self.route.get(k).map(|&b| net.parallel(self.lane, b)).unwrap_or(false) {
                k += 1;
            }
            self.planned_next = self.route.get(k).copied();
            self.ahead = self.route.iter().skip(k + 1).take(PLAN_LANES).copied().collect();
            return;
        }
        if self.planned_next.is_none() {
            self.ahead.clear();
        }
        let planned: f32 = self.upcoming().map(|l| net.lanes[l].length()).sum();
        let rest = net.lanes[self.lane].length() - self.s;
        if self.planned_next.is_some() && (rest + planned >= self.plan_ahead() || self.ahead.len() >= PLAN_LANES) {
            return;
        }
        let mut plan: Vec<usize> = self.upcoming().collect();
        self.extend_plan(net, self.lane, rest, &mut plan);
        self.planned_next = plan.first().copied();
        self.ahead = plan.into_iter().skip(1).collect();
    }

    /// Choose lanes after `from` (which has `rest` metres left) until `plan` covers
    /// `PLAN_AHEAD` metres.
    fn extend_plan(&mut self, net: &Network, from: usize, rest: f32, plan: &mut Vec<usize>) {
        let mut dist = rest + plan.iter().map(|&l| net.lanes[l].length()).sum::<f32>();
        let ahead = self.plan_ahead();
        while dist < ahead && plan.len() <= PLAN_LANES {
            let last = plan.last().copied().unwrap_or(from);
            // [ROLLBACK cycle-135] not round a loop of short lanes to fill the plan (two
            // lanes of no length leading into each other made a plan of twenty), nor back
            // onto the lane the car came from: those ways are left out of the choice, so the
            // car still takes any other way on; where there is none, the plan ends and the
            // car drives to the end of the paths and is gone (as in OMSI)
            let (lane0, prev0) = (self.lane, self.prev_lane);
            let len = |x: usize| net.lanes[x].length();
            let avoid = |n: usize| -> bool {
                if n == from {
                    return len(from) + plan.iter().map(|&x| len(x)).sum::<f32>() < 20.0;
                }
                if let Some(q) = plan.iter().position(|&x| x == n) {
                    return plan[q..].iter().map(|&x| len(x)).sum::<f32>() < 20.0;
                }
                plan.is_empty() && from == lane0 && prev0 == Some(n) && len(from) + len(n) < 20.0
            };
            match self.choose_after(net, last, &avoid) {
                Some(n) => {
                    dist += net.lanes[n].length();
                    plan.push(n);
                }
                None => break,
            }
        }
    }

    /// A way planned on a network linked without the lit stubs (a recorded vehicle's
    /// route, its plan, and the way on of a lane change under way, `change_plan`) takes the
    /// lit stubs it went round: where one lane of it is no longer
    /// linked to the next and a lit stub between them is, the stub goes in. Returns whether
    /// anything changed.
    pub fn route_through_stubs(&mut self, net: &Network) -> bool {
        let through = |way: &mut Vec<usize>, keep: Option<&mut usize>| -> bool {
            let mut changed = false;
            let mut k = 0;
            let mut keep = keep;
            while k + 1 < way.len() {
                let (a, b) = (way[k], way[k + 1]);
                if let Some(la) = net.lanes.get(a) {
                    if !la.next.contains(&b) {
                        if let Some(&st) = la.next.iter().find(|&&st| net.is_lit_stub(st) && net.lanes[st].next.contains(&b)) {
                            way.insert(k + 1, st);
                            if let Some(i) = keep.as_deref_mut() {
                                if *i > k {
                                    *i += 1;
                                }
                            }
                            changed = true;
                        }
                    }
                }
                k += 1;
            }
            changed
        };
        let mut route = std::mem::take(&mut self.route);
        let mut ri = self.route_index;
        let mut changed = through(&mut route, Some(&mut ri));
        self.route = route;
        self.route_index = ri;
        let mut plan: Vec<usize> = std::iter::once(self.lane).chain(self.upcoming()).collect();
        if through(&mut plan, None) {
            changed = true;
            self.planned_next = plan.get(1).copied();
            self.ahead = plan.into_iter().skip(2).collect();
        }
        // [round 7 E3 fix] a lane change under way: the way on from the lane it moves to
        // (`change_plan`, chosen when the change began) takes the stubs too (viz case13 #45279
        // crossed lit stub 9114's red mid-change)
        if let Some(c) = self.change {
            let mut plan: Vec<usize> = std::iter::once(c.to).chain(self.change_plan.iter().copied()).collect();
            if through(&mut plan, None) {
                changed = true;
                self.change_plan = plan.into_iter().skip(1).collect();
            }
        }
        changed
    }

    /// Put the car on a fixed route starting at its first lane.
    pub fn set_route(&mut self, net: &Network, route: Vec<usize>, s: f32) {
        self.route = route;
        self.route_index = 0;
        if let Some(&l) = self.route.first() {
            self.lane = l;
            self.s = s;
        }
        self.prev_lane = None;
        self.plan_next(net);
    }

    /// Begin a lane change onto the neighbour `to` (`dir` 1 left, 2 right): the indicator
    /// goes on at once, the car moves over after a moment.
    pub fn start_change(&mut self, net: &Network, to: usize, dir: i32) {
        let Some(l) = net.lanes.get(self.lane) else { return };
        let Some(t) = net.lanes.get(to) else { return };
        let length = change_length(self.speed);
        // (the same share of a lane that starts beside this one, else the point beside the car)
        let s_to = net.beside_s(self.lane, to, self.s.min(l.length()));
        let rest = t.length() - s_to;
        self.change = Some(LaneChange { to, t: 0.0, length, s_to, dir, wait: SIGNAL_BEFORE_CHANGE, bypass: false });
        self.blinker = dir;
        let mut plan = Vec::new();
        if self.route.is_empty() {
            self.extend_plan(net, to, rest, &mut plan);
        } else {
            plan.extend(self.route.iter().skip(self.route_index + 2).take(PLAN_LANES).copied());
        }
        self.change_plan = plan;
    }

    /// How far the car still has to drive to distance `ss` of route lane `ri` (a lane it
    /// changes over from counts as the lane beside it).
    pub fn route_distance(&self, net: &Network, ri: usize, ss: f32) -> f32 {
        let (mut k, mut d) = (self.route_index, -self.s);
        if let Some(c) = self.change {
            if self.route.get(k + 1) == Some(&c.to) {
                k += 1;
                d = -c.s_to;
            }
        }
        while k < ri && k + 1 < self.route.len() {
            if !net.parallel(self.route[k], self.route[k + 1]) {
                d += net.lanes[self.route[k]].length();
            } else {
                // a lane beside that starts earlier is further along at the same place
                d -= net.beside_delta(self.route[k], self.route[k + 1]);
            }
            k += 1;
        }
        d + ss
    }

    /// A timetable route that moves over to the lane beside the current one next: that lane
    /// and the side it lies on (1 left, 2 right). The driver starts the move as soon as the
    /// lane is free (`start_route_change`).
    pub fn route_change_due(&self, net: &Network) -> Option<(usize, i32)> {
        if self.route.is_empty() || self.change.is_some() {
            return None;
        }
        let &b = self.route.get(self.route_index + 1)?;
        if !net.parallel(self.lane, b) {
            return None;
        }
        // a lane beside that begins further on: not before the car is level with its start
        let delta = net.beside_delta(self.lane, b);
        if delta < 0.0 && self.s < -delta + 1.0 {
            return None;
        }
        let (la, lb) = (&net.lanes[self.lane], &net.lanes[b]);
        // (compared where the car is: a lane beside may start well before or after this one)
        let (pa, ha) = la.at(self.s.min(la.length()));
        let pb = lb.at(net.beside_s(self.lane, b, self.s)).0;
        let h = (ha as f64).to_radians();
        let side = (pb - pa).truncate().dot(DVec2::new(h.cos(), -h.sin()));
        Some((b, if side > 0.0 || (side.abs() < 0.5 && lb.offset > la.offset) { 2 } else { 1 }))
    }

    /// During a lane change: the places on its own lane and on the lane it moves to that the car
    /// is level with, (point, heading in degrees) each.
    pub fn change_anchors(&self, net: &Network) -> Option<((DVec3, f32), (DVec3, f32))> {
        let c = self.change?;
        if c.to >= net.lanes.len() {
            return None;
        }
        Some((self.seq().point(net, self.s, 0.0), self.change_seq(c.to).point(net, c.s_to, 0.0)))
    }

    /// During a lane change: how far it has got (the S-curve's share of the way across, 0..1)
    /// and where the body is, as lateral offsets (m, + right) from its own lane and from the
    /// lane it moves to, at the place along each the car is level with.
    pub fn change_offsets(&self, net: &Network) -> Option<(f32, f32, f32)> {
        let c = self.change?;
        if c.to >= net.lanes.len() {
            return None;
        }
        let (ps, hs) = self.seq().point(net, self.s, 0.0);
        let (pt, ht) = self.change_seq(c.to).point(net, c.s_to, 0.0);
        let k = smooth01(c.t - c.wait * self.speed.max(2.0) / c.length);
        let p = ps.lerp(pt, k as f64);
        let off = |q: DVec3, h: f32| {
            let hr = (h as f64).to_radians();
            (p - q).truncate().dot(DVec2::new(hr.cos(), -hr.sin())) as f32
        };
        Some((k, off(ps, hs) + self.lateral, off(pt, ht) + self.lateral))
    }

    /// A lane change that stands frozen part-way (the car cannot move on, so neither does the
    /// change, which goes by distance): settled where the body is, `done` in place (the car is
    /// on the lane it moved to, offset by what was left of the move) or given up (back on its
    /// own lane, offset by what it had moved). Either way the offset eases off over the S-curve
    /// of a bay as the car drives on (`lateral_ramp`), so the body does not jump. (viz3 case09:
    /// 13 cars that began a change at the tail of a red queue stood diagonally across the lines
    /// for 40-95 s; case01 #4436 for 45 s.) Which: see the traffic's `settle_choice`.
    pub fn settle_change(&mut self, net: &Network, done: bool) {
        let Some(c) = self.change else { return };
        let Some((_, from_own, from_target)) = self.change_offsets(net) else {
            self.change = None;
            self.change_plan.clear();
            return;
        };
        let off = if done { from_target } else { from_own };
        if done {
            if self.route.get(self.route_index + 1) == Some(&c.to) {
                self.route_index += 1;
            }
            let to_len = net.lanes[c.to].length();
            self.lane = c.to;
            self.s = c.s_to.min(to_len);
            self.prev_lane = None;
            let plan = std::mem::take(&mut self.change_plan);
            self.planned_next = plan.first().copied();
            self.ahead = plan.into_iter().skip(1).collect();
            if self.route.is_empty() && self.planned_next.is_none() {
                self.plan_next(net);
            }
            self.change_cooldown = 6.0;
        } else {
            self.change_plan.clear();
            // (not straight into the same change again)
            self.change_cooldown = 10.0;
        }
        self.change = None;
        self.lateral_curve.clear();
        self.lateral = off;
        self.lateral_target = 0.0;
        self.lateral_ramp = (off, 0.0, self.odometer, s_curve_len(off));
        self.settled_offset = true;
    }

    /// Pull out into `to` from behind a standing obstacle: a short, steep move over
    /// `length` metres (8, less from close behind it).
    pub fn start_bypass(&mut self, net: &Network, to: usize, dir: i32, length: f32) {
        self.start_change(net, to, dir);
        if let Some(c) = self.change.as_mut() {
            c.length = length;
            c.bypass = true;
            // (pulling out round something: the old signal time)
            c.wait = c.wait.min(1.2);
        }
    }

    /// Start the lane change a route asks for: at once (the indicator has been on while the
    /// driver waited for a gap) and quick enough to be done well before a fork's branches part.
    pub fn start_route_change(&mut self, net: &Network, to: usize, dir: i32) {
        self.start_change(net, to, dir);
        let left = (net.lanes[self.lane].length() - self.s).max(0.0);
        if let Some(c) = self.change.as_mut() {
            c.wait = 0.0;
            c.length = c.length.min(left * 0.7).max(6.0);
        }
    }

    /// The lane sequence the car is driving along: where it came from, where it is, the plan.
    fn seq(&self) -> LaneSeq {
        let mut q = LaneSeq { lanes: [0; PLAN_LANES + 2], n: 0, cur: 0 };
        if let Some(p) = self.prev_lane {
            q.lanes[0] = p;
            q.n = 1;
        }
        q.cur = q.n;
        q.lanes[q.n] = self.lane;
        q.n += 1;
        for l in self.upcoming() {
            if q.n == q.lanes.len() {
                break;
            }
            q.lanes[q.n] = l;
            q.n += 1;
        }
        q
    }

    /// The lane sequence of the lane the car is changing to, with the way on it chose.
    fn change_seq(&self, to: usize) -> LaneSeq {
        let mut q = LaneSeq { lanes: [0; PLAN_LANES + 2], n: 1, cur: 0 };
        q.lanes[0] = to;
        for &l in &self.change_plan {
            if q.n == q.lanes.len() {
                break;
            }
            q.lanes[q.n] = l;
            q.n += 1;
        }
        q
    }

    /// The point of the car's way `d` metres ahead of it (behind for negative `d`): along
    /// its lanes with the joints smoothed, over to the new lane during a lane change, and
    /// out to the side where it pulls into a bay or round a parked car. The steering
    /// follows this curve; it has no steps, so neither does the car.
    pub fn way_point(&self, net: &Network, d: f32) -> DVec3 {
        let change = self.change.filter(|c| c.to < net.lanes.len()).map(|c| self.change_seq(c.to));
        self.way_point_on(net, &self.seq(), change.as_ref(), d, &mut [Tangents::default(); 2])
    }

    /// `way_point` at each of `ds`: the same points, the lane sequences built once and the
    /// tangents of a lane segment kept while the points fall in it (the perception samples
    /// a hundred metres of way for every car each frame).
    pub fn way_points(&self, net: &Network, ds: impl Iterator<Item = f32>) -> Vec<DVec3> {
        let (seq, change) = (self.seq(), self.change.filter(|c| c.to < net.lanes.len()).map(|c| self.change_seq(c.to)));
        let mut cache = [Tangents::default(); 2];
        ds.map(|d| self.way_point_on(net, &seq, change.as_ref(), d, &mut cache)).collect()
    }

    fn way_point_on(&self, net: &Network, seq: &LaneSeq, change_seq: Option<&LaneSeq>, d: f32, cache: &mut [Tangents; 2]) -> DVec3 {
        let (mut p, mut h) = seq.point_with(net, self.s, d, &mut cache[0]);
        if let (Some(c), Some(cs)) = (self.change, change_seq) {
            let (pt, ht) = cs.point_with(net, c.s_to, d, &mut cache[1]);
            // how far the change will have got by the time the car is `d` metres on
            let tau = (d - c.wait * self.speed.max(2.0)) / c.length;
            let k = smooth01(c.t + tau);
            p = p.lerp(pt, k as f64);
            h += wrap_deg(ht - h) * k;
        }
        let lat = self.lateral_at(self.odometer + d);
        if lat.abs() > 1e-3 {
            let hr = (h as f64).to_radians();
            p += DVec3::new(hr.cos(), -hr.sin(), 0.0) * lat as f64;
        }
        p
    }

    /// How fast the car may be going now so that it can take every bend of the next stretch
    /// of its way at `lat_accel`, braking gently (2 m/s²) into the tight ones. Without this
    /// the cars swept round a junction at 40 km/h - well over a g. The bend is the lane's
    /// own curvature or the turn of the way over 6 m, whichever is sharper: lanes are linked
    /// with up to 40° between them, and such a kink is a bend too.
    pub fn curve_speed(&self, net: &Network) -> f32 {
        let mut best = self.curve_speed_on(net, &self.seq(), self.s);
        if let Some(c) = self.change {
            if c.to < net.lanes.len() {
                best = best.min(self.curve_speed_on(net, &self.change_seq(c.to), c.s_to));
            }
        }
        best
    }

    fn curve_speed_on(&self, net: &Network, q: &LaneSeq, s: f32) -> f32 {
        let reach = (self.speed * self.speed / 4.0 + 12.0).min(90.0);
        let mut best = f32::MAX;
        // The samples lie at fixed places of the road (every 2.5 m of the odometer), not at
        // fixed distances ahead of the car: moving with the car, the sample that caught a
        // bend jumped 2.5 m nearer every 2.5 m, the allowed speed fell in steps of 4 m²/s²
        // and the car braked hard, let go and braked hard again with nothing in front of it.
        let first = 2.5 - self.odometer.rem_euclid(2.5);
        let mut d = 0.0f32;
        let mut next = first;
        while d <= reach {
            let (i, u) = q.locate(net, s, d);
            let turn = wrap_deg(q.heading(net, s, d + 3.0) - q.heading(net, s, d - 3.0)).abs().to_radians() / 6.0;
            let k = net.lanes[q.lanes[i]].curvature_at(u).abs().max(turn);
            if k > 1e-4 {
                let v = (self.lat_accel / k).sqrt().max(2.5);
                // (a bend may begin anywhere up to a sample's spacing before the sample
                // that finds it: the speed is taken from there, or the car met the start of
                // a tight turn half a metre after its profile had allowed 1 m/s more)
                best = best.min((v * v + 2.0 * 2.0 * (d - 2.5).max(0.0)).sqrt());
            }
            d = next;
            next += 2.5;
        }
        best
    }

    /// The sideways offset of the way `d` metres ahead of the car (m, + = right).
    pub fn lateral_ahead(&self, d: f32) -> f32 {
        self.lateral_at(self.odometer + d)
    }

    /// The sideways offset of the way when the odometer reads `x`.
    fn lateral_at(&self, x: f32) -> f32 {
        // [ROLLBACK deform-105] a bent way laid out point by point
        if let Some(&(x_last, _)) = self.lateral_curve.last() {
            if x < x_last {
                let c = &self.lateral_curve;
                let k = c.partition_point(|p| p.0 <= x);
                if k == 0 {
                    return c[0].1;
                }
                let (a, b) = (c[k - 1], c[k]);
                let t = (x - a.0) / (b.0 - a.0).max(1e-3);
                return a.1 + (b.1 - a.1) * t;
            }
        }
        let (from, to, x0, len) = self.lateral_ramp;
        from + (to - from) * smooth01((x - x0) / len.max(0.1))
    }

    /// The first turn (1 left, 2 right) on the way within `within` metres, with the
    /// distance to the start of the turning lane (0 while on it).
    pub fn turn_ahead(&self, net: &Network, within: f32) -> Option<(i32, f32)> {
        let here = &net.lanes[self.lane];
        if here.turn != 0 {
            return Some((here.turn, 0.0));
        }
        let mut dist = here.length() - self.s;
        for l in self.upcoming() {
            if dist > within {
                break;
            }
            let lane = &net.lanes[l];
            if lane.turn != 0 {
                return Some((lane.turn, dist));
            }
            dist += lane.length();
        }
        None
    }

    /// Set the indicator for what the car is doing or about to do: a lane change, its own
    /// manoeuvre (pulling away from a stop), moving sideways into a bay or out round a
    /// parked car and back, and a turn at the junction ahead - from about four seconds
    /// before it (at least 25 m, at most 60 m) until the turn is done.
    pub fn update_blinker(&mut self, net: &Network) {
        self.blinker = if let Some(c) = self.change {
            c.dir
        } else if self.signal != 0 && self.signal_time > 0.0 {
            self.signal
        } else if (self.lateral_target - self.lateral).abs() > 0.3 && self.speed > 1.0 {
            // [ROLLBACK blinker-130] (not standing: a car held in a queue beside a parked car
            // whose side kept changing flashed left and right in turns)
            if self.lateral_target > self.lateral { 2 } else { 1 }
        } else {
            let within = (self.speed * 4.0).clamp(25.0, 60.0);
            self.turn_ahead(net, within).map(|t| t.0).unwrap_or(0)
        };
    }

    /// [Q8-1] Back up along the way behind by at most `left` m this step (the traffic's queue
    /// escape, `omsi_traffic::plan`): from rest at `ai_motion::PULL_OUT_ACCEL` to `REVERSE_SPEED`,
    /// slowing at the driver's comfortable brake to stand where `left` runs out; the speed is
    /// negative meanwhile, the odometer runs back (the wheels turn backwards) and the lateral
    /// offset stays where it is (the ramps run by the odometer: the caller freezes them,
    /// `freeze_lateral`). [fix pass 1] Never over the start of its own lane: it stops there.
    /// Returns the distance moved back.
    pub fn reverse_step(&mut self, net: &Network, dt: f32, left: f32) -> f32 {
        self.change_cooldown = (self.change_cooldown - dt).max(0.0);
        let left = left.max(0.0);
        let b = self.decel.max(0.5);
        let want = -REVERSE_SPEED.min((2.0 * b * left).sqrt());
        let v0 = self.speed.min(0.0);
        let v1 = if want < v0 { (v0 - crate::ai_motion::PULL_OUT_ACCEL * dt).max(want) } else { (v0 + b * dt).min(want) };
        let mut back = (-(v0 + v1) * 0.5 * dt).clamp(0.0, left);
        let mut v1 = if back >= left { 0.0 } else { v1 };
        self.s -= back;
        // ([fix pass 1] never back over the start of its own lane: the escape keeps the body on it)
        if self.s < 0.0 {
            back += self.s;
            self.s = 0.0;
            v1 = 0.0;
        }
        let _ = net;
        self.odometer -= back;
        self.acc = if dt > 0.0 { (v1 - v0) / dt } else { 0.0 };
        self.speed = v1;
        self.braking = false;
        self.held = false;
        back
    }

    /// [Q8-1] The lateral offset stays where it is while the odometer runs back (`reverse_step`):
    /// no ramp or bent way left to follow, a new one starts from here as the car drives on.
    pub fn freeze_lateral(&mut self) {
        let lat = self.lateral;
        self.lateral_curve.clear();
        self.lateral_ramp = (lat, lat, self.odometer, 0.1);
    }

    /// Advance along the network. `obstacle` = distance from the car's origin to the rear of
    /// a standing vehicle ahead (m), `stop_at` = distance from its origin to a stop line
    /// (m). False at a dead end (the car is taken off the road). (The tests' plain driver.)
    #[cfg(test)]
    pub fn advance(&mut self, net: &Network, dt: f32, obstacle: Option<f32>, stop_at: Option<f32>) -> bool {
        let lead = obstacle.map(|d| Lead { gap: d - self.front, speed: 0.0, acc: 0.0 });
        self.drive(net, dt, lead, stop_at)
    }

    /// The acceleration the driver wants (the Intelligent Driver Model): towards the
    /// desired speed on a free road, and a braking term for what is ahead that is gentle
    /// (`decel`) while there is room and only as hard as it must be when there is not.
    /// `lead` is the vehicle ahead, `stop` the distance from the car's origin to where it
    /// has to stop (a light, a junction it gives way at, a bus stop).
    pub fn desired_accel(&self, net: &Network, lead: Option<Lead>, stop: Option<f32>) -> f32 {
        let Some(lane) = net.lanes.get(self.lane) else { return 0.0 };
        let (a, b) = (self.accel.max(0.1), self.decel.max(0.5));
        let v = self.speed;
        let limit = |l: &Lane| (l.speed_limit_kmh * self.desire).min(self.max_speed_kmh).max(3.0) / 3.6;
        let mut v0 = limit(lane);
        // a lower limit on the lanes ahead (a junction's turning lanes, a 30 zone) is
        // reached at that speed, slowing gently (1.2 m/s²) from where it has to: taken only
        // on entering the lane, the new limit threw the model into a -3 m/s² stop at every
        // junction, with nothing in front of the car
        {
            let reach = (v * v / 2.4 + 10.0).min(120.0);
            let mut d = lane.length() - self.s;
            for l in self.upcoming() {
                if d > reach {
                    break;
                }
                if let Some(nl) = net.lanes.get(l) {
                    let vl = limit(nl);
                    if vl < v0 {
                        v0 = v0.min((vl * vl + 2.0 * 1.2 * d.max(0.0)).sqrt());
                    }
                    d += nl.length();
                }
            }
        }
        let mut acc = a * (1.0 - (v / v0).powi(4)).max(-1.5 * b / a);
        // bends: follow the speed profile `curve_speed` lays out (it assumes 2 m/s² of
        // braking), blending in over the last metre per second above it
        let bend = self.curve_speed(net);
        if bend < v0 && v > bend - 1.0 {
            let track = -2.0 + (bend - v) / 0.6;
            let k = ((v - (bend - 1.0)) / 1.0).clamp(0.0, 1.0);
            acc = acc.min(acc + (track - acc) * k);
        }
        let interaction = |gap: f32, lead_speed: f32, s0: f32, headway: f32| -> f32 {
            let s_star = s0 + (v * headway + v * (v - lead_speed) / (2.0 * (a * b).sqrt())).max(0.0);
            -a * (s_star / gap.max(0.05)).powi(2)
        };
        let mut out = acc;
        if let Some(l) = lead {
            let idm = acc + interaction(l.gap, l.speed.max(0.0), self.min_gap, self.headway);
            // The constant-acceleration heuristic (the "ACC" variant of the model): how hard
            // the driver has to brake if the car ahead keeps doing what it does. A car that
            // has just cut in close ahead but drives almost as fast calls for a firm brake
            // and a gap that opens again over the next seconds, not an emergency stop.
            let (vl, s, al) = (l.speed.max(0.0), l.gap.max(0.05), l.acc.min(a));
            let den = vl * vl - 2.0 * s * al;
            let cah = if vl * (v - vl) <= -2.0 * s * al && den > 1e-3 { v * v * al / den } else { al - (v - vl).max(0.0).powi(2) / (2.0 * s) };
            // (a car beside or just ahead that drives away faster gives the model a gap of
            // nothing: its braking term is bounded so that the heuristic decides)
            let idm = idm.max(-2.0 * MAX_BRAKE);
            let acc_lead = if idm >= cah { idm } else { 0.01 * idm + 0.99 * (cah + b * ((idm - cah) / b).tanh()) };
            out = out.min(acc_lead);
        }
        if let Some(d) = stop {
            // A stop line: far away the model's braking term (without a time gap, the line
            // does not move off), gently; once the constant deceleration that stops the car
            // at the line reaches half the driver's comfortable braking, that deceleration.
            // The model alone braked twice as hard as needed when a light turned yellow 25 m
            // ahead, and then eased off.
            let room = (d - self.front - STOP_LINE_GAP).max(0.02);
            let need = v * v / (2.0 * room);
            let idm = (acc + interaction(d - self.front, 0.0, STOP_LINE_GAP, 0.2)).max(-0.5 * b);
            let constant = -need * 1.05;
            let k = ((need - 0.35 * b) / (0.15 * b)).clamp(0.0, 1.0);
            let a_stop = idm + (constant - idm) * k;
            out = out.min(a_stop);
        }
        out.clamp(-MAX_BRAKE, a)
    }

    /// Advance along the network with the car ahead (`lead`) and a stop point (`stop`,
    /// distance from the car's origin). False at a dead end.
    pub fn drive(&mut self, net: &Network, dt: f32, lead: Option<Lead>, stop: Option<f32>) -> bool {
        self.change_cooldown = (self.change_cooldown - dt).max(0.0);
        self.signal_time = (self.signal_time - dt).max(0.0);
        if self.signal_time <= 0.0 {
            self.signal = 0;
        }
        if net.lanes.get(self.lane).is_none() {
            return false;
        }
        let mut acc = self.desired_accel(net, lead, stop);
        if let Some(cap) = self.accel_cap {
            acc = acc.min(cap);
        }
        // standing: a driver who is held there moves off only after a moment when the way
        // clears (the wave that runs down a queue at a green light)
        if self.speed < 0.05 {
            if acc <= 0.05 {
                // (a hold of a frame or two - a junction that is free and not free by turns
                // as the cars on the ring come and go - winds the reaction back only a
                // little: set back whole every frame, it never ran out, and the car stood at
                // an empty roundabout for minutes, "about to go")
                self.start_timer = if self.held { (self.start_timer + 3.0 * dt).min(self.reaction) } else { self.reaction };
                self.held = true;
                acc = acc.min(0.0);
            } else if self.held {
                // [ROLLBACK queuestart-100] the cars of a queue do not wait for the one ahead to
                // be gone: once it rolls, the reaction runs out three times as fast (the wave
                // down the queue at a green light), but for the slow drivers (the longest
                // reactions, about one in five) who keep waiting as before
                let rolling = lead.map(|l| l.speed > 0.5).unwrap_or(false);
                let quick = rolling && self.reaction < 1.05;
                let k = if quick { 3.0 } else { 1.0 };
                self.start_timer -= k * dt;
                if self.start_timer > 0.0 {
                    acc = 0.0;
                } else {
                    self.held = false;
                }
            }
        } else if self.speed > 0.5 {
            self.held = false;
        }
        let v0 = self.speed;
        let v1 = (v0 + acc * dt).max(0.0);
        self.speed = v1;
        self.acc = if dt > 0.0 { (v1 - v0) / dt } else { 0.0 };
        // brake lights: braking, or holding the car on the brake
        self.braking = acc < -0.6 || (v1 < 0.3 && self.held);
        let ds = (v0 + v1) * 0.5 * dt;
        self.s += ds;
        self.odometer += ds;
        // lane change: signal, glide over to the neighbour, then continue there
        if let Some(mut c) = self.change {
            c.s_to += ds;
            if c.wait > 0.0 {
                c.wait = (c.wait - dt).max(0.0);
            } else {
                c.t += ds / c.length;
            }
            let from_len = net.lanes[self.lane].length();
            let mut to_len = net.lanes.get(c.to).map(|l| l.length()).unwrap_or(0.0);
            // A move that is not over where a lane ends carries on across the joint: the
            // lane beside goes on into the way chosen on it, and this lane into its plan
            // (below). Finished at the joint instead, the car jumped sideways by what was
            // left of the move - so a lane change was only started 20 m and more before a
            // lane's end, and on a road of short spline pieces a car stopped behind a parked
            // car just past a joint never got round it. (A timetable vehicle's route moves
            // over well before the end, as before.)
            let carry_on = self.route.is_empty();
            while carry_on && c.t < 1.0 && c.s_to >= to_len && !self.change_plan.is_empty() {
                c.s_to -= to_len;
                c.to = self.change_plan.remove(0);
                to_len = net.lanes.get(c.to).map(|l| l.length()).unwrap_or(0.0);
            }
            let from_ends = self.s >= from_len && (self.planned_next.is_none() || !carry_on);
            // (a bypass's short S-curve ends once the heading's chord behind the car is off it
            // too: ended at its end, the heading jumped back 40 degrees in a frame, review r6l)
            let tail = if c.bypass { HEADING_CHORD / c.length.max(0.1) } else { 0.0 };
            if c.t >= 1.0 + tail || from_ends || c.s_to >= to_len {
                // arrived on the new lane: it has no joint behind the car, and the way on
                // is planned afresh from there
                if self.route.get(self.route_index + 1) == Some(&c.to) {
                    self.route_index += 1;
                }
                self.lane = c.to;
                self.s = c.s_to.min(to_len);
                self.change = None;
                self.change_cooldown = 6.0;
                self.prev_lane = None;
                let plan = std::mem::take(&mut self.change_plan);
                self.planned_next = plan.first().copied();
                self.ahead = plan.into_iter().skip(1).collect();
            } else {
                self.change = Some(c);
            }
        }
        // bay offset: a new target starts an S-curve from where the car is now, eight metres
        // long for every metre sideways (at least 8, at most 30)
        let curve_end = self.lateral_curve.last().map(|l| l.0);
        if curve_end.is_some_and(|x| self.odometer >= x) {
            self.lateral_curve.clear();
        }
        if (self.lateral_target - self.lateral_ramp.1).abs() > 1e-3 {
            let from = self.lateral_at(self.odometer - ds);
            let len = s_curve_len(self.lateral_target - from);
            self.lateral_ramp = (from, self.lateral_target, self.odometer - ds, len);
        }
        self.lateral = self.lateral_at(self.odometer);
        if self.planned_next.is_none() {
            self.plan_next(net);
        }
        while (self.change.is_none() || self.route.is_empty()) && self.s >= net.lanes[self.lane].length() {
            let l = &net.lanes[self.lane];
            let Some(next) = self.planned_next else {
                // the end of a flight path: the aircraft flies on straight (the way runs on
                // along the end tangent) until the traffic takes it away out of sight
                if l.kind == LaneKind::Air {
                    break;
                }
                return false; // dead end: despawn
            };
            self.s -= l.length();
            self.prev_lane = Some(self.lane);
            self.lane = next;
            if self.route.is_empty() {
                self.planned_next = if self.ahead.is_empty() { None } else { Some(self.ahead.remove(0)) };
            } else {
                // the plan may have stepped over a lane-change entry of the route
                let from = self.route_index + 1;
                self.route_index = (from..(from + 3).min(self.route.len())).find(|&k| self.route[k] == next).unwrap_or(from);
            }
            self.plan_next(net);
        }
        // keep the plan PLAN_AHEAD metres long as the car eats into it (a route's plan
        // only moves when the car enters its next lane)
        if self.route.is_empty() {
            self.plan_next(net);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_stop_is_matched_to_the_lane_it_stands_beside() {
        // out along y = 0 (east), back along y = 6 (west); the stop stands north of the
        // way back: on its right, across the road from the way out
        let out = LaneBuilder::polyline(vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(100.0, 0.0, 0.0)], LaneKind::Street, 3.0);
        let back = LaneBuilder::polyline(vec![DVec3::new(100.0, 6.0, 0.0), DVec3::new(0.0, 6.0, 0.0)], LaneKind::Street, 3.0);
        let net = Network { lanes: vec![out, back], ..Default::default() };
        let stop = DVec3::new(50.0, 9.0, 0.0);
        // nearer to the way back anyway: matched there
        assert_eq!(net.project_stop_on_route(&[0, 1], stop, Some(25.0), 0).unwrap().0, 1);
        // a stop on the right of the way out, nearer the middle of the road
        let stop2 = DVec3::new(50.0, -2.0, 0.0);
        assert_eq!(net.project_stop_on_route(&[0, 1], stop2, Some(25.0), 0).unwrap().0, 0);
        // the route out, back and out again: a stop on the way out, once the trip is past
        // its first leg, is the one on the second way out
        assert_eq!(net.project_stop_on_route(&[0, 1, 0], stop2, Some(25.0), 1).unwrap().0, 2);
    }

    use super::*;

    /// A straight lane of 60 m running north, then a right-hand bend of radius 14 m over
    /// 60°, then straight on again.
    fn junction() -> Network {
        let a = LaneBuilder::arc(DVec3::ZERO, 0.0, 60.0, 0.0, 0.0, LaneKind::Street, 3.0);
        let bend = LaneBuilder::arc(a.end(), 0.0, 14.0 * 60f64.to_radians(), 14.0, 0.0, LaneKind::Street, 3.0);
        let c = LaneBuilder::arc(bend.end(), 60.0, 80.0, 0.0, 0.0, LaneKind::Street, 3.0);
        let mut net = Network { lanes: vec![a, bend, c], ..Default::default() };
        net.link(1.5);
        net
    }

    #[test]
    fn path_rules_open_lanes_by_vehicle_type() {
        let mut l = LaneBuilder::polyline(vec![DVec3::ZERO, DVec3::new(0.0, 50.0, 0.0)], LaneKind::Street, 3.0);
        // no rules: cars and taxis, no AI buses or trucks (Omsi.exe 0x71d714)
        assert!(l.allows(0) && l.allows(1) && !l.allows(2) && !l.allows(3) && l.allows(-1));
        l.rule_trucks = true;
        assert!(l.allows(0) && l.allows(3) && !l.allows(2));
        l.no_cars = true;
        assert!(!l.allows(0) && l.allows(1) && l.allows(3));
        l.rule_trucks = false;
        assert!(!l.allows(0) && !l.allows(1));
        l.rule_bus = true;
        assert!(!l.allows(0) && l.allows(1) && l.allows(2) && !l.allows(3));
    }

    #[test]
    fn slows_down_for_a_bend() {
        let net = junction();
        assert_eq!(net.lanes[0].next, vec![1]);
        let mut car = AiState::new(0, 0.0, 7);
        car.speed = 13.9;
        car.plan_next(&net);
        assert_eq!(car.upcoming().collect::<Vec<_>>(), vec![1, 2]);
        // the bend allows sqrt(2.8 × 14) ≈ 6.3 m/s; 60 m before it the car may still go fast
        let far = car.curve_speed(&net);
        assert!(far > 13.0, "60 m before the bend: {far}");
        let dt = 1.0 / 30.0;
        let mut entered = None;
        for _ in 0..600 {
            if !car.advance(&net, dt, None, None) {
                break; // the end of the test road
            }
            if car.lane == 1 && entered.is_none() {
                entered = Some(car.speed);
            }
        }
        let v = entered.expect("reached the bend");
        assert!(v < 7.5, "entered the bend at {v} m/s");
    }

    /// A pull-out started a few metres before the car's lane ends (a road of short spline
    /// pieces) carries on across the joint and ends on the lane beside's next piece, without
    /// a jump: it used to be finished at the joint, the car moved sideways in one frame.
    #[test]
    fn a_lane_change_carries_on_across_a_joint() {
        let lane = |x: f64, y0: f64, y1: f64| LaneBuilder::polyline(vec![DVec3::new(x, y0, 0.0), DVec3::new(x, y1, 0.0)], LaneKind::Street, 3.0);
        let mut net = Network { lanes: vec![lane(0.0, 0.0, 20.0), lane(-3.5, 0.0, 20.0), lane(0.0, 20.0, 70.0), lane(-3.5, 20.0, 70.0)], ..Default::default() };
        net.link(1.5);
        net.lanes[0].left = Some(1);
        net.lanes[2].left = Some(3);
        let mut car = AiState::new(0, 16.0, 7);
        car.speed = 3.0;
        car.plan_next(&net);
        assert_eq!(car.planned_next, Some(2));
        car.start_bypass(&net, 1, 1, 8.0);
        let dt = 1.0 / 30.0;
        let mut at = car.way_point(&net, 0.0);
        for _ in 0..300 {
            assert!(car.drive(&net, dt, None, None));
            let p = car.way_point(&net, 0.0);
            let step = (p - at).length();
            assert!(step < car.speed as f64 * dt as f64 + 0.05, "a jump of {step:.2} m on lane {} s {:.1} change {:?}", car.lane, car.s, car.change);
            at = p;
            if car.change.is_none() {
                break;
            }
        }
        assert!(car.change.is_none(), "the move is over");
        assert_eq!(car.lane, 3, "on the lane beside's next piece");
        assert!((at.x + 3.5).abs() < 0.05, "over in the lane beside: {at:?}");
    }

    #[test]
    fn a_turning_lane_is_signalled_in_advance() {
        let mut net = junction();
        net.lanes[1].turn = 2;
        let mut car = AiState::new(0, 0.0, 7);
        car.speed = 10.0;
        car.plan_next(&net);
        let dt = 1.0 / 30.0;
        let mut first_on = None;
        for _ in 0..900 {
            car.advance(&net, dt, None, None);
            car.update_blinker(&net);
            if car.blinker == 2 && first_on.is_none() {
                first_on = Some((car.lane, net.lanes[0].length() - car.s));
            }
            if car.lane == 2 && car.s > 10.0 {
                assert_eq!(car.blinker, 0, "indicator off after the turn");
                break;
            }
        }
        let (lane, before) = first_on.expect("indicator on");
        assert_eq!(lane, 0);
        assert!(before >= 24.0, "indicator on only {before} m before the turn");
    }

    /// Einm_erzgebirgs.sco's "Main" light: red and yellow 2 s, green 11 s, yellow 3 s,
    /// red for the rest of the 38 s cycle.
    fn erzgebirgs() -> TrafficLightController {
        TrafficLightController::from_program(vec![(vec![(3, 2.0), (6, 11.0), (9, 3.0), (0, 0.0)], None)], Some(38.0), &[], &[])
    }

    #[test]
    fn a_light_shows_every_aspect_for_its_time_at_any_frame_rate() {
        for dt in [1.0 / 144.0, 1.0 / 30.0, 0.1, 0.25] {
            let mut c = erzgebirgs();
            c.start(0.0);
            // (aspect, seconds) as the lamps show it over two cycles
            let mut seen: Vec<(Aspect, f32)> = Vec::new();
            let mut t = 0.0f32;
            while t < 76.0 {
                let a = TrafficLightController::aspect(c.state(0));
                match seen.last_mut() {
                    Some(last) if last.0 == a => last.1 += dt,
                    _ => seen.push((a, dt)),
                }
                c.advance(dt);
                t += dt;
            }
            let order: Vec<Aspect> = seen.iter().map(|s| s.0).collect();
            assert_eq!(&order[..5], &[Aspect::RedYellow, Aspect::Green, Aspect::Yellow, Aspect::Red, Aspect::RedYellow], "dt {dt}");
            let tol = dt + 1e-3;
            assert!((seen[0].1 - 2.0).abs() <= tol, "red-yellow {} at dt {dt}", seen[0].1);
            assert!((seen[1].1 - 11.0).abs() <= tol, "green {} at dt {dt}", seen[1].1);
            assert!((seen[2].1 - 3.0).abs() <= tol, "yellow {} at dt {dt}", seen[2].1);
            assert!((seen[3].1 - 22.0).abs() <= tol, "red {} at dt {dt}", seen[3].1);
        }
    }

    #[test]
    fn stock_state_codes_mean_what_the_lamp_scripts_show() {
        assert_eq!(TrafficLightController::lamps(0), (true, false, false));
        assert_eq!(TrafficLightController::lamps(3), (true, true, false));
        assert_eq!(TrafficLightController::lamps(6), (false, false, true));
        assert_eq!(TrafficLightController::lamps(9), (false, true, false));
        assert_eq!(TrafficLightController::lamps(12), (false, false, false));
        assert!(TrafficLightController::allows_go(6) && !TrafficLightController::allows_go(9) && !TrafficLightController::allows_go(3));
        // the clock starts from the time of day: two crossings with one cycle run in step
        let (mut a, mut b) = (erzgebirgs(), erzgebirgs());
        a.start(8.0 * 3600.0 + 10.0);
        b.start(8.0 * 3600.0 + 10.0);
        assert_eq!(a.time, b.time);
        // 28 810 s = 758 cycles and 6 s: green
        assert!((a.time - 6.0).abs() < 1e-6);
        assert_eq!(a.state(0), 6);
    }

    #[test]
    fn a_level_crossing_waits_for_its_train() {
        // bue_falks_ohe.sco: road green until a train asks at light 0 (stop at 1 s), barrier
        // down while it is still there (stop at 16 s)
        let mut c = TrafficLightController::from_program(
            vec![(vec![(0, 15.0), (6, 4.0), (0, 1.0)], None), (vec![(6, 2.0), (9, 12.0), (0, 5.0), (3, 1.0)], None)],
            Some(22.0),
            &[[0.0, 1.0, 1.0], [0.0, 16.0, 0.0]],
            &[],
        );
        c.start(0.0);
        for _ in 0..600 {
            c.advance(0.1);
        }
        assert!((c.time - 1.0).abs() < 1e-6 && c.held, "holding at {}", c.time);
        assert_eq!(c.state(1), 6, "road green while no train comes");
        c.request[0] = true;
        let mut t = 0.0;
        while t < 20.0 {
            c.advance(0.1);
            t += 0.1;
        }
        assert!((c.time - 16.0).abs() < 1e-6 && c.held, "the train is still there: holding at {}", c.time);
        assert_eq!(c.state(0), 6);
        assert_eq!(c.state(1), 0, "road red while the train passes");
        c.request[0] = false;
        c.advance(1.0);
        assert!((c.time - 17.0).abs() < 1e-3, "{}", c.time);
    }

    #[test]
    fn a_bus_phase_is_skipped_when_no_bus_comes() {
        // Kreuz_Heerstr_Pillnitzer_Reimer.sco: the bus light 3 jumps from 51.5 to 61.5 s
        let mut c = TrafficLightController::from_program(vec![(vec![(0, 52.0), (3, 2.0), (6, 4.0), (9, 3.0), (0, 0.0)], Some(10.0))], Some(64.0), &[], &[[0.0, 51.5, 1.0, 61.5]]);
        c.start(50.0);
        c.advance(2.0);
        assert!((c.time - 62.0).abs() < 1e-3, "jumped: {}", c.time);
        c.start(0.0);
        c.time = 50.0;
        c.request[0] = true;
        c.advance(3.0);
        assert!((c.time - 53.0).abs() < 1e-3);
        assert_eq!(c.state(0), 3, "the bus gets its phase");
    }

    #[test]
    fn a_backwards_jump_replays_its_phase_once() {
        // win-wit.sco and similar crossings use a small rewind to extend a green. Taking
        // that jump again on every pass locked the whole program in that stretch.
        let mut c = TrafficLightController::from_program(
            vec![(vec![(0, 2.0), (3, 2.0), (6, 4.0), (9, 2.0), (0, 0.0)], None)],
            Some(12.0),
            &[],
            &[[0.0, 8.0, 0.0, 4.0]],
        );
        c.time = 7.9;
        c.request[0] = true;
        for _ in 0..50 {
            c.advance(0.1);
        }
        assert_eq!(c.state(0), 9, "the clock left the replayed green");
    }

    #[test]
    fn simultaneous_inactive_events_do_not_stall_the_cycle() {
        let mut c = TrafficLightController::from_program(
            vec![(vec![(0, 4.0), (6, 16.0)], None); 2],
            Some(20.0),
            &[[0.0, 4.0, 0.0]],
            &[[1.0, 4.0, 0.0, 12.0]],
        );
        c.start(0.0);
        c.advance(5.0);
        assert_eq!(c.time, 5.0);
        assert_eq!(c.state(0), 6);
        for _ in 0..40 {
            c.advance(1.0);
        }
        assert_eq!(c.time, 5.0, "the clock completes later cycles too");
        assert!(!c.held);
    }

    #[test]
    fn simultaneous_inactive_events_do_not_hide_a_later_active_jump() {
        let mut c = TrafficLightController::from_program(
            vec![(vec![(0, 4.0), (6, 16.0)], None); 3],
            Some(20.0),
            &[[0.0, 4.0, 0.0]],
            &[[1.0, 4.0, 0.0, 9.0], [2.0, 4.0, 0.0, 12.0]],
        );
        c.request[2] = true;
        c.start(0.0);
        c.advance(5.0);
        assert_eq!(c.time, 13.0, "take the jump and consume the remaining second");
        assert!(!c.held);
    }

    #[test]
    fn simultaneous_stops_release_and_are_checked_on_the_next_cycle() {
        let mut c = TrafficLightController::from_program(
            vec![(vec![(0, 4.0), (6, 16.0)], None); 3],
            Some(20.0),
            &[[0.0, 4.0, 0.0], [1.0, 4.0, 0.0], [2.0, 4.0, 0.0]],
            &[],
        );
        c.request[2] = true;
        c.start(0.0);
        c.advance(5.0);
        assert_eq!(c.time, 4.0);
        assert!(c.held);
        c.advance(1.0);
        assert_eq!(c.time, 4.0, "an active stop remains held next frame");
        c.request[2] = false;
        c.advance(1.0);
        assert_eq!(c.time, 5.0);
        assert!(!c.held);
        c.advance(16.0);
        assert_eq!(c.time, 1.0);
        c.request[2] = true;
        c.advance(4.0);
        assert_eq!(c.time, 4.0, "the released stop is evaluated after wrapping");
        assert!(c.held);
    }

    #[test]
    fn many_simultaneous_inactive_events_consume_the_frame_time() {
        let mut c = TrafficLightController::new(vec![vec![(0, 4.0), (6, 16.0)]], 20.0);
        c.stops = vec![
            LightStop {
                light: 0,
                time: 4.0,
                if_request: false,
                jump_to: None,
            };
            33
        ];
        c.start(4.0);
        c.advance(0.25);
        assert_eq!(c.time, 4.25, "the inactive group must not exhaust the event budget");
        assert_eq!(c.state(0), 6);
        assert!(!c.held);
    }

    #[test]
    fn a_car_stops_at_the_line_without_braking_hard() {
        let net = junction();
        let mut car = AiState::new(0, 0.0, 3);
        car.speed = 13.9;
        car.accel = 1.5;
        car.decel = 2.2;
        car.plan_next(&net);
        let dt = 1.0 / 60.0;
        let line = 55.0;
        let mut hardest = 0.0f32;
        for _ in 0..1200 {
            let stop = line - car.s;
            car.drive(&net, dt, None, Some(stop));
            hardest = hardest.min(car.acc);
        }
        let front = car.s + car.front;
        assert!(car.speed < 0.01, "stopped: {}", car.speed);
        assert!(front <= line && front > line - 1.5, "front at {front}, line at {line}");
        assert!(hardest > -3.5, "braked at {hardest} m/s²");
    }

    #[test]
    fn a_follower_keeps_its_distance_when_the_leader_brakes() {
        let net = junction();
        let mut car = AiState::new(0, 0.0, 5);
        car.speed = 12.0;
        car.plan_next(&net);
        let (mut lead_s, mut lead_v) = (30.0f32, 12.0f32);
        let dt = 1.0 / 60.0;
        let mut closest = f32::MAX;
        for k in 0..900 {
            // the leader brakes hard after a second and stays standing
            if k > 60 {
                lead_v = (lead_v - 6.0 * dt).max(0.0);
            }
            lead_s += lead_v * dt;
            let gap = lead_s - 2.5 - (car.s + car.front);
            closest = closest.min(gap);
            car.drive(&net, dt, Some(Lead { gap, speed: lead_v, acc: if k > 60 && lead_v > 0.0 { -6.0 } else { 0.0 } }), None);
        }
        assert!(closest > 0.8, "came within {closest} m");
        assert!(car.speed < 0.05);
    }

    #[test]
    fn right_of_way_between_paths() {
        // a crossing: a from the south going north, b from the east going west, c from the
        // north going south and turning left (east)
        let a = LaneBuilder::arc(DVec3::new(0.0, -10.0, 0.0), 0.0, 20.0, 0.0, 0.0, LaneKind::Street, 3.0);
        let b = LaneBuilder::arc(DVec3::new(10.0, 0.0, 0.0), 270.0, 20.0, 0.0, 0.0, LaneKind::Street, 3.0);
        let mut c = LaneBuilder::arc(DVec3::new(-1.5, 10.0, 0.0), 180.0, 10.0 * std::f64::consts::FRAC_PI_2, -10.0, 0.0, LaneKind::Street, 3.0);
        c.turn = 1;
        let mut net = Network { lanes: vec![a, b, c], ..Default::default() };
        net.link(1.5);
        // equal priority: b comes from a's right
        assert!(net.must_yield(0, 1));
        assert!(!net.must_yield(1, 0));
        // the left turn waits for the oncoming car
        assert!(net.must_yield(2, 0));
        assert!(!net.must_yield(0, 2));
        // a [rule] priority beats the geometry
        net.lanes[0].priority = 192.0;
        net.lanes[1].priority = 64.0;
        assert!(!net.must_yield(0, 1));
        assert!(net.must_yield(1, 0));
    }

    #[test]
    fn right_of_way_on_the_left() {
        // the same crossing on a left-hand-traffic map: b (from a's right) now waits for a
        // (from b's left), and a right turn across the oncoming traffic waits, a left one not
        let a = LaneBuilder::arc(DVec3::new(0.0, -10.0, 0.0), 0.0, 20.0, 0.0, 0.0, LaneKind::Street, 3.0);
        let b = LaneBuilder::arc(DVec3::new(10.0, 0.0, 0.0), 270.0, 20.0, 0.0, 0.0, LaneKind::Street, 3.0);
        let mut c = LaneBuilder::arc(DVec3::new(-1.5, 10.0, 0.0), 180.0, 10.0 * std::f64::consts::FRAC_PI_2, -10.0, 0.0, LaneKind::Street, 3.0);
        c.turn = 2;
        let mut net = Network { lanes: vec![a, b, c], left_hand: true, ..Default::default() };
        net.link(1.5);
        assert!(!net.must_yield(0, 1));
        assert!(net.must_yield(1, 0));
        assert!(net.must_yield(2, 0));
        net.lanes[2].turn = 1;
        assert!(!net.must_yield(2, 0));
        assert_eq!(net.oncoming_sign(), 1.0);
    }

    /// Lanes of one crossing object (source 2, one key), linked, with their meeting places.
    fn object_net(lanes: Vec<Lane>) -> Network {
        let mut net = Network { lanes, ..Default::default() };
        for (k, l) in net.lanes.iter_mut().enumerate() {
            l.source = 2;
            l.key = Some(LaneKey { tile: (0, 0), id: 7, path: k as u16 });
        }
        net.link(1.5);
        net.compute_conflicts();
        net
    }

    fn line(pts: &[(f64, f64)]) -> Lane {
        LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), LaneKind::Street, 3.0)
    }

    #[test]
    fn right_of_way_fallbacks_between_equal_priorities() {
        // a northbound on x = 0; b westbound on y = 0, coming from a's right: right before
        // left, a yields to b
        let a = || line(&[(0.0, -40.0), (0.0, 40.0)]);
        let b = || line(&[(40.0, 0.0), (-40.0, 0.0)]);
        let net = object_net(vec![a(), b()]);
        assert!(net.must_yield(0, 1) && !net.must_yield(1, 0), "right before left");
        // 1. (yards: below, from the spline approaches) a piece of density 0.001 with no
        // spline behind it is no yard: right before left
        let mut piece_only = b();
        piece_only.density = 0.001;
        let net = object_net(vec![a(), piece_only]);
        assert!(net.must_yield(0, 1) && !net.must_yield(1, 0), "a piece's density is no yard");
        // 2. b turns 45 degrees right before it crosses (no turn flag): it yields to a, which
        // goes straight on
        let turn = line(&[(40.0, 0.0), (10.0, 0.0), (0.0, 10.0), (-30.0, 40.0)]);
        let net = object_net(vec![a(), turn]);
        assert!(net.approach_to(1, net.crossings[1][0].at).heading_change > TURN_DEG);
        assert!(!net.must_yield(0, 1) && net.must_yield(1, 0), "the turner yields");
        // ... with a turn flag on a straight path just the same
        let mut flagged = b();
        flagged.turn = 1;
        let net = object_net(vec![a(), flagged]);
        assert!(!net.must_yield(0, 1) && net.must_yield(1, 0), "the flagged turner yields");
        // 3. road class, from the spline lanes the approaches come off (pieces 10 m long
        // meeting at x = y = 0, spline approaches 0 (north), 1 (west) before them)
        let class_net = |main: (f32, bool), side: f32| -> Network {
            let mut sa = line(&[(0.0, -200.0), (0.0, -10.0)]);
            sa.density = main.0;
            let mut sb = line(&[(200.0, 0.0), (10.0, 0.0)]);
            sb.density = side;
            let mut net = object_net(vec![line(&[(0.0, -10.0), (0.0, 40.0)]), line(&[(10.0, 0.0), (-40.0, 0.0)]), sa, sb, line(&[(3.5, -200.0), (3.5, -10.0)])]);
            for l in 2..5 {
                net.lanes[l].source = 1;
                net.lanes[l].key = Some(LaneKey { tile: (0, 0), id: 100 + l as i64, path: 0 });
            }
            // (two lanes of the main road: its spline's second path, beside the first)
            if main.1 {
                net.lanes[4].key = Some(LaneKey { tile: (0, 0), id: 102, path: 1 });
                net.by_key.clear();
                for (i, l) in net.lanes.iter().enumerate() {
                    if let Some(k) = l.key {
                        net.by_key.entry(k).or_default().push(i);
                    }
                }
            }
            net
        };
        // 1. b comes off a yard spline (density 0.001): it yields to the street
        let net = class_net((1.0, false), 0.001);
        assert!(!net.must_yield(0, 1) && net.must_yield(1, 0), "the yard yields");
        // ... a [rule] priority still decides first
        let mut net = class_net((1.0, false), 0.001);
        net.lanes[0].priority = 64.0;
        assert!(net.must_yield(0, 1) && !net.must_yield(1, 0), "[rule] priority first");
        // ... a spline of density 0 (no random traffic) is no yard
        let net = class_net((1.0, false), 0.0);
        assert!(net.must_yield(0, 1) && !net.must_yield(1, 0), "density 0 is no yard");
        let net = class_net((1.0, true), 1.0);
        assert!(!net.must_yield(0, 1) && net.must_yield(1, 0), "one lane yields to two");
        // ... density 3.0 against 0.5 (more than 3 times)
        let net = class_net((3.0, false), 0.5);
        assert!(!net.must_yield(0, 1) && net.must_yield(1, 0), "the minor road yields");
        // ... 0.5 against 1.0 is no clear difference: right before left
        let net = class_net((1.0, false), 0.5);
        assert!(net.must_yield(0, 1) && !net.must_yield(1, 0));
        // ... and the densities of the junction pieces themselves are no road class
        let mut main = a();
        main.density = 3.0;
        let mut side = b();
        side.density = 0.5;
        let net = object_net(vec![main, side]);
        assert!(net.must_yield(0, 1) && !net.must_yield(1, 0), "piece densities are spawn weights");
    }

    /// The oncoming case: a straight way through and a left turn across it from the other
    /// direction. BeijingCity's object pieces carry `[rule] trafficdensity` 0.0-0.1 or 1.0
    /// as the author set them (spawn weights, not road classes); 27 such pairs in the viz3
    /// fixtures now make the straight yield to the oncoming left turner (density ratio > 3 or
    /// yard <= 0.01, e.g. junction 32 lanes 12903/12938, junction 29 lanes 11329/11571).
    #[test]
    fn review7_right_of_way_the_oncoming_left_turn_still_waits_for_the_straight() {
        // a northbound on x = 0 (density 0.1 on its piece); b southbound on x = -3.5 turning
        // left (east) across a (turn flag 1, density 1.0)
        let mut a = line(&[(0.0, -40.0), (0.0, 40.0)]);
        a.density = 0.1;
        let mut b = line(&[(-3.5, 40.0), (-3.5, 5.0), (0.0, 0.0), (5.0, -3.5), (40.0, -3.5)]);
        b.turn = 1;
        let net = object_net(vec![a, b]);
        assert!(!net.crossings[0].is_empty(), "they cross");
        assert!(net.must_yield(1, 0) && !net.must_yield(0, 1), "the left turner must still yield to the oncoming straight (fails on r4a: the straight yields)");
    }

    #[test]
    fn right_of_way_a_yard_spline_34_m_back_is_still_a_yard() {
        // as case05's side road: the side way (b) runs 10 m on its last piece and 24 m on the
        // piece before it before its spline (density 0.001) ends; the main way (a) comes off a
        // spline of 3.0 right behind its piece
        let mut sa = line(&[(0.0, -200.0), (0.0, -10.0)]);
        sa.density = 3.0;
        let mut sb = line(&[(200.0, 0.0), (34.0, 0.0)]);
        sb.density = 0.001;
        let mut net = object_net(vec![line(&[(0.0, -10.0), (0.0, 40.0)]), line(&[(10.0, 0.0), (-40.0, 0.0)]), line(&[(34.0, 0.0), (10.0, 0.0)]), sa, sb]);
        for l in 3..5 {
            net.lanes[l].source = 1;
            net.lanes[l].key = Some(LaneKey { tile: (0, 0), id: 100 + l as i64, path: 0 });
        }
        assert!(net.approach_to(1, net.meeting_on(1, 0).0).spline.is_some_and(|c| c.density == 0.001));
        assert!(!net.must_yield(0, 1) && net.must_yield(1, 0), "the side way off the yard yields");
    }

    #[test]
    fn right_of_way_a_way_merging_in_at_an_angle_yields_to_the_one_going_on() {
        // a straight on into the exit; b comes in from the right at about 35 degrees, neither
        // with a turn of TURN_DEG over its approach
        let a = line(&[(0.0, -40.0), (0.0, 0.0)]);
        let b = line(&[(25.0, -35.0), (6.0, -10.0), (0.0, 0.0)]);
        let exit = line(&[(0.0, 0.0), (0.0, 30.0)]);
        let net = object_net(vec![a, b, exit]);
        assert!(net.crossings[0].iter().any(|c| c.merge && c.other == 1), "{:?}", net.crossings[0]);
        assert!(net.approach_to(1, net.lanes[1].length()).heading_change <= TURN_DEG);
        assert!(!net.must_yield(0, 1) && net.must_yield(1, 0), "the merging way yields");
    }

    #[test]
    fn a_shallow_crossing_is_a_long_meeting_place() {
        // one junction object (source 2, same key): a straight lane and two lanes crossing
        // it, one square, one at 20°
        let key = |path: u16| Some(LaneKey { tile: (0, 0), id: 1, path });
        let mut a = LaneBuilder::arc(DVec3::new(0.0, -20.0, 0.0), 0.0, 40.0, 0.0, 0.0, LaneKind::Street, 3.0);
        let mut b = LaneBuilder::arc(DVec3::new(20.0, 0.0, 0.0), 270.0, 40.0, 0.0, 0.0, LaneKind::Street, 3.0);
        let h = 20f64.to_radians();
        let mut c = LaneBuilder::arc(DVec3::new(-20.0 * h.sin(), -20.0 * h.cos(), 0.0), 20.0, 40.0, 0.0, 0.0, LaneKind::Street, 3.0);
        for (l, k) in [(&mut a, 0), (&mut b, 1), (&mut c, 2)] {
            l.source = 2;
            l.key = key(k);
        }
        let mut net = Network { lanes: vec![a, b, c], ..Default::default() };
        net.link(1.5);
        let square = net.crossings[0].iter().find(|x| x.other == 1).expect("square crossing");
        let shallow = net.crossings[0].iter().find(|x| x.other == 2).expect("shallow crossing");
        assert!((square.at - 20.0).abs() < 0.5 && (shallow.at - 20.0).abs() < 0.5);
        // square: the bodies touch within a car's width or so of the point
        assert!(square.before <= 3.0 && square.after <= 3.0, "{square:?}");
        // at 20° the centre lines stay within 2.6 m for 2.6 / sin 20° ≈ 7.6 m either side
        assert!(shallow.before >= 7.0 && shallow.after >= 7.0, "{shallow:?}");
    }

    #[test]
    fn a_car_does_not_turn_back_onto_a_short_lane_it_came_from() {
        // [ROLLBACK cycle-134] two short lanes that lead into each other: the car drives on
        // to the second and, with no way on, is gone at its end
        let st = LaneKind::Street;
        let mut a = LaneBuilder::arc(DVec3::ZERO, 0.0, 3.0, 0.0, 0.0, st, 3.0);
        let mut b = LaneBuilder::arc(DVec3::new(0.0, 3.0, 0.0), 0.0, 3.0, 0.0, 0.0, st, 3.0);
        a.next = vec![1];
        b.next = vec![0];
        let net = Network {
            lanes: vec![a, b],
            ..Default::default()
        };
        let mut car = AiState::new(0, 0.0, 1);
        car.plan_next(&net);
        assert_eq!(car.planned_next, Some(1));
        car.lane = 1;
        car.prev_lane = Some(0);
        car.planned_next = None;
        car.ahead.clear();
        car.plan_next(&net);
        assert_eq!(car.planned_next, None);
    }

    #[test]
    fn a_driver_with_a_choice_takes_a_dead_end_like_any_other_way() {
        // a lane that forks: one way ends after 50 m, the other runs round a long loop;
        // `reach` tells them apart, but the driver takes either (as in OMSI, which spawns on
        // and drives into every path and takes the car away where it runs off the end)
        let a = LaneBuilder::arc(DVec3::ZERO, 0.0, 30.0, 0.0, 0.0, LaneKind::Street, 3.0);
        let dead = LaneBuilder::arc(DVec3::new(0.0, 30.0, 0.0), 10.0, 50.0, 0.0, 0.0, LaneKind::Street, 3.0);
        let on = LaneBuilder::arc(DVec3::new(0.0, 30.0, 0.0), 350.0, 700.0, 0.0, 0.0, LaneKind::Street, 3.0);
        let mut net = Network { lanes: vec![a, dead, on], ..Default::default() };
        net.link(1.5);
        assert_eq!(net.lanes[0].next.len(), 2);
        assert!(net.reach[1] < DEAD_END && net.reach[0] >= DEAD_END && net.reach[2] >= DEAD_END);
        let (mut into_dead, mut on_round) = (0, 0);
        for seed in 1..40 {
            let mut car = AiState::new(0, 0.0, seed);
            car.plan_next(&net);
            match car.planned_next {
                Some(1) => into_dead += 1,
                Some(2) => on_round += 1,
                other => panic!("seed {seed}: {other:?}"),
            }
        }
        assert!(into_dead > 0 && on_round > 0, "{into_dead} into the dead end, {on_round} on round");
    }

    #[test]
    fn the_way_has_no_steps_at_a_lane_joint() {
        // two lanes that meet 1.2 m apart: the way bends over the joint instead of jumping
        let a = LaneBuilder::arc(DVec3::ZERO, 0.0, 30.0, 0.0, 0.0, LaneKind::Street, 3.0);
        let b = LaneBuilder::arc(DVec3::new(1.2, 30.0, 0.0), 0.0, 30.0, 0.0, 0.0, LaneKind::Street, 3.0);
        let mut net = Network { lanes: vec![a, b], ..Default::default() };
        net.link(1.5);
        let mut car = AiState::new(0, 20.0, 1);
        car.plan_next(&net);
        let mut last = car.way_point(&net, -10.0);
        let mut d = -10.0;
        while d < 25.0 {
            d += 0.25;
            let p = car.way_point(&net, d);
            assert!((p - last).length() < 0.3, "step of {} m at {d}", (p - last).length());
            last = p;
        }
    }

    /// The southbound half of a road through a junction: a lane B (60 m) into the junction's
    /// straight path J1 (10 m), a left turn J2 from the east into the same exit, and the lane
    /// A (50 m) after the junction.
    fn oncoming_road() -> Network {
        let a = LaneBuilder::arc(DVec3::new(-3.0, 150.0, 0.0), 180.0, 50.0, 0.0, 0.0, LaneKind::Street, 3.0);
        let j1 = LaneBuilder::arc(DVec3::new(-3.0, 160.0, 0.0), 180.0, 10.0, 0.0, 0.0, LaneKind::Street, 3.0);
        let j2 = LaneBuilder::arc(DVec3::new(5.0, 158.0, 0.0), 270.0, 8.0 * std::f64::consts::FRAC_PI_2, -8.0, 0.0, LaneKind::Street, 3.0);
        let b = LaneBuilder::arc(DVec3::new(-3.0, 220.0, 0.0), 180.0, 60.0, 0.0, 0.0, LaneKind::Street, 3.0);
        assert!((j2.end() - DVec3::new(-3.0, 150.0, 0.0)).length() < 0.05, "{:?}", j2.end());
        let mut net = Network { lanes: vec![a, j1, j2, b], ..Default::default() };
        net.link(1.5);
        net
    }

    #[test]
    fn upstream_walks_back_through_the_junction() {
        let net = oncoming_road();
        assert_eq!(net.prev[0].len(), 2, "{:?}", net.prev);
        // 20 m into A, looking 100 m back: A itself, both junction paths and the lane before
        let up = net.upstream(0, 20.0, 100.0, 16);
        let find = |l: usize| up.iter().find(|e| e.0 == l).copied();
        assert_eq!(up[0], (0, 0.0, None));
        let (_, off1, into1) = find(1).expect("J1");
        assert!((off1 + 10.0).abs() < 0.01 && into1 == Some(0), "{up:?}");
        let (_, off2, into2) = find(2).expect("J2");
        assert!((off2 + 8.0 * std::f32::consts::FRAC_PI_2).abs() < 0.05 && into2 == Some(0), "{up:?}");
        let (_, off3, into3) = find(3).expect("B");
        assert!((off3 + 70.0).abs() < 0.01 && into3 == Some(1), "{up:?}");
        // a car 15 m into B is at 15 - 70 = -55 in A's distances: 75 m before the place
        // looking only 25 m back, the lanes that end within reach are there, B is not
        let near = net.upstream(0, 20.0, 25.0, 16);
        assert!(near.iter().any(|e| e.0 == 1) && !near.iter().any(|e| e.0 == 3), "{near:?}");
        // and the list is capped
        assert_eq!(net.upstream(0, 20.0, 100.0, 2).len(), 2);
    }

    #[test]
    fn an_acceleration_cap_holds_a_car_back() {
        let net = junction();
        let mut car = AiState::new(0, 0.0, 7);
        car.accel = 2.5;
        car.plan_next(&net);
        car.accel_cap = Some(1.0);
        for _ in 0..30 {
            car.drive(&net, 1.0 / 30.0, None, None);
        }
        assert!(car.speed > 0.9 && car.speed < 1.0 + 1e-3, "{}", car.speed);
        car.accel_cap = None;
        for _ in 0..30 {
            car.drive(&net, 1.0 / 30.0, None, None);
        }
        assert!(car.speed > 3.0, "{}", car.speed);
    }

    #[test]
    fn arrival_times() {
        assert_eq!(arrival_time(-1.0, 5.0, 1.0, 10.0), 0.0);
        // at a steady 10 m/s
        assert!((arrival_time(100.0, 10.0, 1.0, 10.0) - 10.0).abs() < 1e-4);
        // from a standstill at 2 m/s² without reaching the limit: sqrt(2 d / a)
        assert!((arrival_time(50.0, 0.0, 2.0, 20.0) - 50f32.sqrt()).abs() < 1e-3);
        // 5 s up to 10 m/s over 25 m, then 125 m at 10 m/s
        assert!((arrival_time(150.0, 0.0, 2.0, 10.0) - 17.5).abs() < 1e-3);
        // a car faster than the limit keeps its speed
        assert!((arrival_time(60.0, 15.0, 2.0, 10.0) - 4.0).abs() < 1e-3);
    }

}

#[cfg(test)]
mod light_tests {
    use super::TrafficLightController;

    #[test]
    fn remaining_green_of_a_pedestrian_light() {
        let at = |mut c: TrafficLightController, t: f64| {
            c.time = t;
            c
        };
        // Einm_Stresow_Obermeier "Main_Ped": green 8 s, then red to the end of a 38 s cycle
        let ped = TrafficLightController::new(vec![vec![(6, 8.0), (0, 0.0)]], 38.0);
        let c = at(ped.clone(), 3.0);
        assert_eq!(c.state(0), 6);
        assert!((c.remaining(0) - 5.0).abs() < 1e-4);
        let c = at(ped, 20.0);
        assert_eq!(c.state(0), 0);
        assert!((c.remaining(0) - 18.0).abs() < 1e-4);
        // red running over the end of the cycle into a red start
        let d = at(TrafficLightController::new(vec![vec![(0, 19.0), (3, 2.0), (6, 11.0), (9, 3.0), (0, 0.0)]], 38.0), 36.0);
        assert!((d.remaining(0) - 21.0).abs() < 1e-4, "{}", d.remaining(0));
        assert!(at(TrafficLightController::new(vec![vec![(6, 5.0)]], 0.0), 1.0).remaining(0).is_infinite());
    }

    #[test]
    fn two_signal_groups_of_one_program_are_never_green_together_or_overlap() {
        // 0: green 0-20, yellow 20-23, red to 40; 1: red 0-25, red-yellow 25-26, green 26-37,
        // yellow 37-40; 2: green 15-35 (overlaps both)
        let c = TrafficLightController::new(
            vec![vec![(6, 20.0), (9, 3.0), (0, 17.0)], vec![(0, 25.0), (3, 1.0), (6, 11.0), (9, 3.0)], vec![(0, 15.0), (6, 20.0), (0, 5.0)]],
            40.0,
        );
        assert!(c.never_together(0, 1) && c.never_together(1, 0));
        assert!(!c.never_together(0, 2) && !c.never_together(1, 2));
        assert!(!c.never_together(0, 0), "a light is not exclusive of itself");
        assert!(!c.never_together(0, 7));
    }
}

#[cfg(test)]
mod aurora_pool_tests {
    use super::*;
    #[test]
    fn independent_rules_and_default_references() {
        let mut lane = LaneBuilder::arc(DVec3::ZERO, 0.0, 100.0, 0.0, 0.0, LaneKind::Street, 3.0);
        lane.group_density = vec![(0, 1.0), (1, 0.1), (3, 0.001), (5, 0.0)];
        let defaults = [1, 0, 1, 0, 0, 0];
        assert_eq!(lane.pool_density(&defaults, 1), 0.1);
        assert_eq!(lane.pool_density(&defaults, 2), 1.0);
        assert_eq!(lane.pool_density(&defaults, 3), 0.001);
        assert_eq!(lane.pool_density(&defaults, 4), 0.0);
        assert_eq!(lane.pool_density(&defaults, 5), 0.0);
        lane.group_density.push((1, 0.5));
        assert_eq!(lane.pool_density(&defaults, 1), 0.5);
    }
}
