//! The offscreen harness's moving player and what is watched round it.
//!
//! `--follow-line` (`LineFollower`): the player's bus drives its duty's trip as a timetable
//! bus of the AI drives one. The player's vehicle is a physics vehicle outside the traffic's
//! cars, so the AI controller cannot drive it directly: a timetable bus of the departure's AI
//! group (a player's type may know no AI doors: the MAN A21 of the gate runs) takes the
//! player's departure on where the player's bus stands (`Schedule::drive_player_trip`), and
//! every frame the player's vehicle is put where that
//! bus's body is (`pose_player`: the AI's own `AiBody` pose, its speed, its rear sections).
//! So the player's bus stops at its stops for a dwell with the doors open, obeys the lights
//! and gives way as an AI timetable bus does (`bus_service`, `omsi_traffic`), and moves as
//! the AI's bodies move (`ai_motion`). The physics and scripts of the player's vehicle do not
//! run meanwhile; its doors stay shut, and the people see the AI bus in its place, not the
//! player's bus nor the duty (`Humans::tick` gets no player's bus while it drives): they
//! board it and wait for it at its stops as for any timetable bus. The other
//! drivers see the AI bus, not a player's outline (`Traffic::tick` gets no player box while
//! it drives): what they do round the player's bus is what they do round a timetable bus.
//! When the bus is lost (taken off the road, its trip over and let go) the player's bus
//! stands where it was left.
//!
//! `--watch-player` (implied by `--follow-line`; `PlayerWatch`): what happens round the
//! player while the run goes on, for a moving player and a standing one alike, printed in the
//! run's summary: the player's position and speed every `POSITION_EVERY` s; cars and people
//! appearing or vanishing in the view cone within `WATCH_RADIUS` of the player, and vanishing
//! in the cone farther off, each told apart as in sight, behind cover or beyond its horizon
//! by its own kind's test (`Sight`: a car by the traffic's, a person by the people's,
//! `Humans::figure_sight`), the ones in sight logged with their id and time; the people taken
//! in sight by the people's own test (`Humans::gone`); people held at a kerb in sight; the
//! nearest unloaded ground (or end of the loaded lanes) along the AI bus's whole route ahead;
//! and the bus's own holds by reason, with their times. People put down
//! in plain view past `humans::APPEAR_IN_VIEW_BEYOND` (the people's own rule) are counted
//! apart and not listed. A car that parks or pulls out of a parked space (its parked object
//! put down or taken up where it stands) and anything while the map loads
//! (`Traffic::loading_phase`) is not counted. The AI bodies as the game draws them
//! (`BodyLane`): how far each swings out of its own footprint along its way and where two of
//! them overlap.

use crate::traffic::Traffic;
use glam::{DVec3, Vec3};
use omsi_sim::ai_motion::AiBody;
use omsi_sim::VehicleInstance;
use std::collections::{BTreeMap, HashMap, HashSet};

/// Radius (m) round the player within which no car or person may appear or vanish in the
/// view cone (round 11 area H brief, H-2).
pub(crate) const WATCH_RADIUS: f64 = 150.0;
/// Seconds between the lines with the player's position and speed (H-2: "per 10 s").
const POSITION_EVERY: f32 = 10.0;
/// Seconds between two tries to put the AI bus on the player's trip when it could not be put
/// there (a tile not loaded yet, the lane taken): the timetable's own retry
/// (`Placed::Busy` waits 3 s).
const RETRY_EVERY: f32 = 3.0;
/// Seconds between the probes of the ground along the route ahead: in a second the bus
/// covers a few metres of a tile hundreds of metres across (`omsi_map::tile_size`), and a tile
/// takes seconds to load (`tiles::Streamer`).
const TILES_EVERY: f32 = 1.0;

/// The player's vehicle put where the AI's body is: its pose (`AiBody::apply`, as the AI
/// vehicles get it), its speed and its steering, its rigid body placed there moving at that
/// speed, and the rear sections where the AI vehicle's are (`ai_rear`).
pub(crate) fn pose_player(body: &AiBody, speed: f32, ai_rear: &[(DVec3, f64)], v: &mut VehicleInstance) {
    body.apply(v);
    v.physics.speed = speed;
    v.physics.steer_deg = body.steer;
    let (position, heading) = (v.position, v.heading);
    if let Some(rb) = v.rigid.as_mut() {
        rb.place(position, heading);
        rb.velocity = rb.orientation.mul_vec3(Vec3::Y) * speed;
    }
    for (part, &(p, h)) in v.trailers.iter_mut().zip(ai_rear) {
        part.set_pose(p, h);
    }
}

/// `--follow-line`: the AI bus that drives the player's trip, and the player's vehicle
/// following it.
#[derive(Default)]
pub(crate) struct LineFollower {
    /// The AI bus's car id once it is on the road.
    pub car: Option<u64>,
    /// When to try again to put it there (run seconds).
    next_try: f32,
    /// Why it was not put there the last time.
    pub refused: Option<String>,
    /// The run second it was lost, and why.
    pub lost: Option<(f32, &'static str)>,
}

impl LineFollower {
    /// Whether the player's vehicle is driven by the AI bus now.
    pub fn driving(&self) -> bool {
        self.car.is_some() && self.lost.is_none()
    }

    /// Put the AI bus on the player's trip, unless it is there or was lost; a try every
    /// `RETRY_EVERY` s.
    #[allow(clippy::too_many_arguments)]
    pub fn place(
        &mut self,
        t_s: f32,
        schedule: &mut crate::schedule::Schedule,
        duty: &crate::schedule::PlayerDuty,
        world: &crate::scene::World,
        traffic: &mut Traffic,
        renderer: &omsi_render::Renderer,
        scene: &mut omsi_render::Scene,
        bus: &VehicleInstance,
    ) {
        if self.car.is_some() || self.lost.is_some() || t_s < self.next_try {
            return;
        }
        self.next_try = t_s + RETRY_EVERY;
        match schedule.drive_player_trip(world, traffic, renderer, scene, traffic.day_time, duty, bus) {
            Ok(id) => {
                self.car = Some(id);
                self.refused = None;
            }
            Err(e) => {
                if self.refused.as_deref() != Some(e.as_str()) {
                    log::info!("follow line t={t_s:.1}: {e}");
                }
                self.refused = Some(e);
            }
        }
    }

    /// The player's vehicle put where the AI bus is this frame (`pose_player`); true while
    /// it drives it.
    pub fn pose(&mut self, t_s: f32, traffic: &Traffic, v: &mut VehicleInstance) -> bool {
        let Some(id) = self.car else { return false };
        if self.lost.is_some() {
            return false;
        }
        let Some(c) = traffic.cars.iter().find(|c| c.id == id) else {
            self.lost = Some((t_s, "taken off the road"));
            log::info!("follow line t={t_s:.1}: bus {id} taken off the road; the player's bus stands where it was left");
            return false;
        };
        if c.released || !c.serves_stops() {
            self.lost = Some((t_s, "trip over"));
            log::info!("follow line t={t_s:.1}: bus {id} let go after its trip; the player's bus stands where it was left");
            return false;
        }
        let rear: Vec<(DVec3, f64)> = c.vehicle.trailers.iter().map(|t| (t.position, t.heading)).collect();
        pose_player(&c.body, c.ai.state.speed, &rear, v);
        true
    }
}

/// One appearance or disappearance seen.
struct Event {
    t: f32,
    what: &'static str,
    id: u64,
    how: &'static str,
    dist: f64,
    pos: DVec3,
    sight: Sight,
}

/// Where something stands for the eye, by the test of its own kind: a car by the traffic's
/// (the view cone `traffic::Viewer::frames`, its perceptual horizon
/// `traffic::Viewer::beyond_horizon`, cover `Traffic::occluded`), a person by the people's
/// (`Humans::figure_sight`: the view cone, the horizon of a `humans::FIGURE_HEIGHT` figure,
/// `may_appear`'s ray to the head). Out of the view cone, in it behind cover, in it beyond the
/// horizon, a person in plain view past `humans::APPEAR_IN_VIEW_BEYOND` (where the people's
/// rule lets somebody be put down), or in sight.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Sight {
    Out,
    Hidden,
    Beyond,
    Far,
    In,
}

impl From<crate::humans::FigureSight> for Sight {
    fn from(f: crate::humans::FigureSight) -> Sight {
        use crate::humans::FigureSight as F;
        match f {
            F::Out => Sight::Out,
            F::Beyond => Sight::Beyond,
            F::Hidden => Sight::Hidden,
            F::Far => Sight::Far,
            F::In => Sight::In,
        }
    }
}

impl Sight {
    fn name(self) -> &'static str {
        match self {
            Sight::Out => "out of the view cone",
            Sight::Hidden => "behind cover",
            Sight::Beyond => "beyond its horizon",
            Sight::Far => "in plain view past the people's in-view appearance distance",
            Sight::In => "in sight",
        }
    }
}

/// The AI bus's hold that is going on: since when, why, the vehicle waited for or followed,
/// its lane.
struct Hold {
    since: f32,
    why: &'static str,
    other: Option<u64>,
    lane: usize,
}

/// `--watch-player`: see the module's doc.
#[derive(Default)]
pub(crate) struct PlayerWatch {
    started: bool,
    /// Cars last frame: id -> (position, radius, body height).
    cars: HashMap<u64, (DVec3, f64, f64)>,
    /// Every car id seen so far (a car woken from its sleep out of range comes back under it).
    cars_ever: HashSet<u64>,
    /// Parked objects within the watch's reach last frame: key -> position.
    parked: HashMap<i64, DVec3>,
    /// People last frame: id -> (position, on the ground).
    people: HashMap<u32, (DVec3, bool)>,
    /// Counted events: appeared / vanished in the cone within `WATCH_RADIUS`, taken away
    /// in the cone at any distance.
    events: Vec<Event>,
    /// Appearances and disappearances not counted, by kind ("car parked", ...).
    uncounted: BTreeMap<&'static str, usize>,
    /// The people's own count of people taken in sight (`Humans::gone`).
    people_taken_seen: usize,
    /// People held at a kerb in the cone within `WATCH_RADIUS`: person-seconds and who.
    held_s: f64,
    held_ids: HashSet<u32>,
    /// The nearest unloaded ground along the AI bus's route ahead (m, its whole route from
    /// its lane on, not its plan), the least over the run, with its run second, where, and
    /// what it was (no ground, or the end of the route's lanes the loaded tiles have);
    /// and how often it was probed.
    tiles_min: Option<(f64, f32, DVec3, &'static str)>,
    tiles_probes: usize,
    tiles_next: f32,
    /// The AI bus's holds: seconds by reason, the episodes, and the one going on.
    holds_s: BTreeMap<&'static str, f64>,
    episodes: Vec<(f32, f32, &'static str, Option<u64>, usize)>,
    hold: Option<Hold>,
    /// The position lines.
    lines: Vec<String>,
    next_line: f32,
    /// The AI bodies against their lanes and each other.
    bodies: BodyLane,
}

/// Step (m) of the way's points a body is measured against: the polyline between them is
/// within a millimetre of a 25 m bend (the sagitta of a 0.5 m chord).
const BODY_WAY_STEP: f32 = 0.5;
/// How far past an edge (m) the signed measure counts a side as over it: 5 cm, a body truly
/// over the line and not the pure pursuit's steady error.
const BODY_SIDE_OVER: f64 = 0.05;
/// Upper bounds (m) of the radius classes the bodies are told by (the bends of the unit
/// test, `ai_motion`'s 25/40/80 m, and the rest).
const BODY_RADIUS_CLASSES: [f64; 3] = [30.0, 60.0, 120.0];
/// Swing (m) from which a body counts as out of its footprint: the way's polyline and the
/// pure pursuit's steady error on a straight stay within millimetres; a centimetre is a body
/// truly turned off its way.
const BODY_COUNTED: f64 = 0.01;

/// The AI road bodies as the game draws them, every frame outside the map's loading: per
/// vehicle section the point of its outline farthest outside a lane as wide as the body
/// itself (the four corners and the sides abreast of its turning line, the inner side's
/// nearest point to a bend's centre): its distance from the vehicle's way
/// (`AiState::way_point`, with its lane change and sideways moves) less the section's half
/// width, i.e. how far the body leaves its own footprint along the way; for a lane W wide it
/// is outside the lane by this less (W/2 - half width). (The lanes' own `width` is the path's,
/// 1 m on most BeijingCity lanes, not the painted lane.) Front section and coupled rear
/// sections apart, told by the radius of the way under the section (the circle through the
/// way's points at its rear bumper, origin and front bumper), as percentiles of the
/// vehicle-frames (a body put back on its way after a jump of the traffic's state stands
/// metres off it for a frame: the maximum alone is that). And the pairs of bodies (every
/// section) whose outlines overlap (`collision::Obb::separation`), each overlap listed once
/// over with both vehicles as they were when it began (position, heading, speed, lane, place,
/// sideways offset, the lane's radius there).
#[derive(Default)]
struct BodyLane {
    /// Per section kind (front, rear) and radius class (and beyond the last): every
    /// vehicle-frame's swing (m) and its frame time (s); the most with its second, id, type.
    swing: [[Vec<(f32, f32)>; 4]; 2],
    /// The same vehicle-frames, signed against the bend: how far the body's inner side and its
    /// outer side leave the vehicle's lane (m, past its edges on the bend's inside and outside,
    /// `Network::lane_edges`, from the way; - = inside it): (inner, outer, frame time); and
    /// the vehicle-seconds on a lane whose edges the map does not tell, and those moving
    /// sideways off their lane's axis (a lane change, a pull-out, a sideways move: the way is
    /// not on the lane's axis the edges are measured from), not measured.
    sides: [[Vec<(f32, f32, f32)>; 4]; 2],
    sides_unknown_s: [[f64; 4]; 2],
    sides_sideways_s: [[f64; 4]; 2],
    /// Where the inner side passes the lane's edge by more than `BODY_SIDE_OVER`: vehicle-seconds
    /// by the lane's source (spline lane, connector: a junction path or an editor-only lane) and the vehicle (bus or not), and per
    /// vehicle its deepest frame (m, what it was doing then).
    inner_over_s: HashMap<(&'static str, bool), f64>,
    inner_worst: HashMap<u64, (f32, String)>,
    /// The street lanes by where their edges come from (`Network::lane_edges`), counted
    /// whenever the network has grown: (lanes counted, spline lanes with both edges, connectors
    /// - junction paths and editor-only splines' lanes - from the pieces they join, spline lanes
    /// not known, connectors not known), and
    /// the known lanes' widths (left + right edge, m).
    lane_edges: (usize, usize, usize, usize, usize, Vec<f32>),
    /// The files of the lanes whose edges are not known, with their counts.
    edges_unknown: HashMap<String, usize>,
    most: [[(f64, f32, u64, String); 4]; 2],
    /// Overlapping pair-seconds, the pairs, the deepest (m) with its second and ids.
    overlap_s: f64,
    pairs: HashSet<(u64, u64)>,
    deepest: (f64, f32, u64, u64),
    /// The overlaps going on: pair -> (since, deepest, both vehicles as they were then).
    open: HashMap<(u64, u64), (f32, f64, String)>,
    /// Each overlap once over: its line.
    episodes: Vec<String>,
}

impl BodyLane {
    fn frame(&mut self, t_s: f32, dt: f32, traffic: &Traffic) {
        use glam::DVec2;
        use omsi_sim::ai_motion::MotionKind;
        use omsi_sim::collision::Obb;
        let net = &traffic.net;
        if self.lane_edges.0 != net.lanes.len() {
            let mut e = (net.lanes.len(), 0, 0, 0, 0, Vec::new());
            self.edges_unknown.clear();
            for (i, l) in net.lanes.iter().enumerate().filter(|(_, l)| l.kind == omsi_sim::traffic::LaneKind::Street) {
                match (net.lane_edges(i), l.source == 2 || l.invisible) {
                    (Some(w), junction) => {
                        if junction { e.2 += 1 } else { e.1 += 1 }
                        e.5.push(w.0 + w.1);
                    }
                    (None, junction) => {
                        if junction { e.4 += 1 } else { e.3 += 1 }
                        *self.edges_unknown.entry(format!("{}{}", if l.invisible { "invisible " } else { "" }, l.name)).or_default() += 1;
                    }
                }
            }
            self.lane_edges = e;
        }
        let mut boxes: Vec<(u64, Obb, f64)> = Vec::new();
        // per vehicle: what it is and does, for the overlap lines
        let mut what: HashMap<u64, String> = HashMap::new();
        for c in traffic.cars.iter().filter(|c| c.body.kind == MotionKind::Road) {
            let st = &c.ai.state;
            if st.lane >= net.lanes.len() {
                continue;
            }
            let n = ((st.rear + st.front + 4.0) / BODY_WAY_STEP).ceil() as usize;
            let first = -st.rear - 2.0;
            let way: Vec<DVec2> = st.way_points(net, (0..=n).map(|i| first + i as f32 * BODY_WAY_STEP)).into_iter().map(|p| p.truncate()).collect();
            // the signed lateral distance of a point from the way (+ = left of its direction)
            let lateral = |p: DVec2| {
                way.windows(2)
                    .map(|w| {
                        let (a, b) = (w[0], w[1]);
                        let t = ((p - a).dot(b - a) / (b - a).length_squared().max(1e-12)).clamp(0.0, 1.0);
                        let c = a + (b - a) * t;
                        ((p - c).length(), (b - a).perp_dot(p - c).signum() * (p - c).length())
                    })
                    .fold((f64::MAX, 0.0), |m, x| if x.0 < m.0 { x } else { m })
                    .1
            };
            let off_way = |p: DVec2| {
                way.windows(2)
                    .map(|w| {
                        let (a, b) = (w[0], w[1]);
                        let t = ((p - a).dot(b - a) / (b - a).length_squared().max(1e-12)).clamp(0.0, 1.0);
                        (p - (a + (b - a) * t)).length()
                    })
                    .fold(f64::MAX, f64::min)
            };
            // the way at `d` m from the vehicle's origin (between its samples)
            let at = |d: f32| {
                let x = ((d - first) / BODY_WAY_STEP).clamp(0.0, n as f32);
                let i = (x.floor() as usize).min(n.saturating_sub(1));
                way[i].lerp(way[(i + 1).min(n)], (x - i as f32) as f64)
            };
            what.insert(
                c.id,
                format!(
                    "#{} {} at ({:.1}, {:.1}) heading {:.0} {:.1} m/s lane {} s {:.1} lateral {:.2} radius {:.0}",
                    c.id,
                    c.vehicle.ty.def.path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default(),
                    c.vehicle.position.x,
                    c.vehicle.position.y,
                    c.vehicle.heading,
                    st.speed,
                    st.lane,
                    st.s,
                    st.lateral,
                    1.0 / net.lanes[st.lane].curvature_at(st.s).abs().max(1e-4)
                ),
            );
            let mut sections: Vec<(DVec3, f64, (f32, f32, f32), f32, f32)> = Vec::new();
            let ext = crate::traffic::extents(&c.vehicle.ty, st.length);
            sections.push((c.vehicle.position, c.vehicle.heading, ext, omsi_sim::ai_motion::rotation_point(&c.vehicle.ty.def).0, 0.0));
            for t in &c.vehicle.trailers {
                // (where along the way the section stands: its origin's distance behind the front's)
                let back = (t.position - c.vehicle.position).truncate().length() as f32;
                sections.push((t.position, t.heading, crate::traffic::extents(&t.ty, st.length), omsi_sim::ai_motion::rotation_point(&t.ty.def).0, -back));
            }
            for (i, &(pos, heading, (front, rear, half), turning, along)) in sections.iter().enumerate() {
                let h = heading.to_radians();
                let fwd = DVec2::new(h.sin(), h.cos());
                let right = DVec2::new(fwd.y, -fwd.x);
                let o = pos.truncate();
                let corners = [(front, half), (front, -half), (-rear, half), (-rear, -half), (turning, half), (turning, -half)];
                let worst = corners
                    .iter()
                    .map(|&(long, lat)| off_way(o + fwd * long as f64 + right * lat as f64) - half as f64)
                    .fold(f64::MIN, f64::max);
                // the way's radius under the section: the circle through three of its points
                let (a, b, q) = (at(along - rear), at(along), at(along + front));
                let cross = (b - a).perp_dot(q - a).abs();
                let k = 2.0 * cross / ((b - a).length() * (q - b).length() * (q - a).length()).max(1e-9);
                let class = BODY_RADIUS_CLASSES.iter().position(|r| k * r >= 1.0).unwrap_or(3);
                let kind = (i > 0) as usize;
                self.swing[kind][class].push((worst as f32, dt));
                // the lane under the section's own origin: a trailer still on the piece before
                // the front's is measured against that piece's edges
                let (sec_lane, sec_s) = st.lane_at(net, along);
                let edges = net.lane_edges(sec_lane);
                // the bend's side: + = it bends left, so its inside is the way's left
                let left = (b - a).perp_dot(q - a) >= 0.0;
                if st.lateral != 0.0 || st.change.is_some() {
                    self.sides_sideways_s[kind][class] += dt as f64;
                } else if let Some((el, er)) = edges {
                    let (inward, inner_edge, outer_edge) = if left { (1.0, el, er) } else { (-1.0, er, el) };
                    let (mut inn, mut out) = (f64::MIN, f64::MIN);
                    for &(long, lat) in &corners {
                        let l = lateral(o + fwd * long as f64 + right * lat as f64) * inward;
                        inn = inn.max(l - inner_edge as f64);
                        out = out.max(-l - outer_edge as f64);
                    }
                    self.sides[kind][class].push((inn as f32, out as f32, dt));
                    if inn > BODY_SIDE_OVER && class < 3 {
                        let ln = &net.lanes[sec_lane];
                        let junction = ln.source == 2 || ln.invisible;
                        let bus = c.vehicle.ty.def.path.extension().is_some_and(|e| e.eq_ignore_ascii_case("bus"));
                        *self.inner_over_s.entry((if junction { "connector" } else { "spline lane" }, bus)).or_default() += dt as f64;
                        let w = self.inner_worst.entry(c.id).or_insert((f32::MIN, String::new()));
                        if inn as f32 > w.0 {
                            let ty = if i == 0 { &c.vehicle.ty } else { &c.vehicle.trailers[i - 1].ty };
                            *w = (
                                inn as f32,
                                format!(
                                    "t={t_s:.1} #{} {} section {i} lane {} ({} {}) s {:.1} of {:.1} edges ({el:.2}, {er:.2}) bend {} R {:.0} speed {:.1}",
                                    c.id,
                                    ty.def.path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default(),
                                    sec_lane,
                                    if ln.invisible { "editor-only lane" } else if junction { "junction path" } else { "spline lane" },
                                    ln.name,
                                    sec_s,
                                    ln.length(),
                                    if left { "left" } else { "right" },
                                    1.0 / k.max(1e-4),
                                    st.speed
                                ),
                            );
                        }
                    }
                } else {
                    self.sides_unknown_s[kind][class] += dt as f64;
                }
                let m = &mut self.most[kind][class];
                if worst > m.0 || m.3.is_empty() {
                    let ty = if i == 0 { &c.vehicle.ty } else { &c.vehicle.trailers[i - 1].ty };
                    let name = ty.def.path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
                    *m = (worst, t_s, c.id, name);
                }
                boxes.push((c.id, Obb::vehicle(o, heading, front as f64, rear as f64, half as f64), front.max(rear).hypot(half) as f64));
            }
        }
        let mut seen: HashSet<(u64, u64)> = HashSet::new();
        for (i, (a, ba, ra)) in boxes.iter().enumerate() {
            for (b, bb, rb) in boxes.iter().skip(i + 1) {
                if a == b || (ba.center - bb.center).length() > ra + rb {
                    continue;
                }
                let sep = ba.separation(bb);
                if sep < 0.0 {
                    let pair = (*a.min(b), *a.max(b));
                    if seen.insert(pair) {
                        self.overlap_s += dt as f64;
                        self.pairs.insert(pair);
                        let e = self.open.entry(pair).or_insert_with(|| (t_s, 0.0, format!("{} / {}", what.get(&pair.0).cloned().unwrap_or_default(), what.get(&pair.1).cloned().unwrap_or_default())));
                        e.1 = e.1.max(-sep);
                    } else if let Some(e) = self.open.get_mut(&pair) {
                        e.1 = e.1.max(-sep);
                    }
                    if -sep > self.deepest.0 {
                        self.deepest = (-sep, t_s, pair.0, pair.1);
                    }
                }
            }
        }
        let over: Vec<(u64, u64)> = self.open.keys().filter(|k| !seen.contains(k)).copied().collect();
        for k in over {
            if let Some((since, depth, who)) = self.open.remove(&k) {
                self.episodes.push(format!("t={since:.1} for {:.1} s, deepest {depth:.2} m: {who}", t_s - since));
            }
        }
    }

    fn report(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        for (i, part) in ["front section", "rear sections"].iter().enumerate() {
            for class in 0..4 {
                let v = &mut self.swing[i][class];
                if v.is_empty() {
                    continue;
                }
                let name = match class {
                    0 => format!("R <= {:.0} m", BODY_RADIUS_CLASSES[0]),
                    3 => format!("R > {:.0} m", BODY_RADIUS_CLASSES[2]),
                    c => format!("R {:.0}-{:.0} m", BODY_RADIUS_CLASSES[c - 1], BODY_RADIUS_CLASSES[c]),
                };
                let secs: f64 = v.iter().map(|x| x.1 as f64).sum();
                let out_s: f64 = v.iter().filter(|x| x.0 as f64 > BODY_COUNTED).map(|x| x.1 as f64).sum();
                let mean = v.iter().map(|x| x.0 as f64).sum::<f64>() / v.len() as f64;
                v.sort_by(|a, b| a.0.total_cmp(&b.0));
                let pct = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize].0;
                let m = &self.most[i][class];
                out.push(format!(
                    "body lane: {part} {name}: {secs:.0} vehicle-s, out of its own footprint {out_s:.0} vehicle-s; swing mean {mean:.2} p50 {:.2} p95 {:.2} p99 {:.2} max {:.2} m (t={:.1} #{} {})",
                    pct(0.5), pct(0.95), pct(0.99), m.0, m.1, m.2, m.3
                ));
            }
        }
        for (i, part) in ["front section", "rear sections"].iter().enumerate() {
            for class in 0..3 {
                let v = &self.sides[i][class];
                if v.is_empty() {
                    continue;
                }
                let secs: f64 = v.iter().map(|x| x.2 as f64).sum();
                let over = |f: &dyn Fn(&(f32, f32, f32)) -> f32| v.iter().filter(|x| f(x) as f64 > BODY_SIDE_OVER).map(|x| x.2 as f64).sum::<f64>();
                let stats = |f: &dyn Fn(&(f32, f32, f32)) -> f32| {
                    let mut x: Vec<f32> = v.iter().map(f).collect();
                    x.sort_by(|a, b| a.total_cmp(b));
                    (x.iter().map(|&y| y as f64).sum::<f64>() / x.len() as f64, x[((x.len() - 1) as f64 * 0.95).round() as usize], x[x.len() - 1])
                };
                let (im, ip, ix) = stats(&|x| x.0);
                let (om, op, ox) = stats(&|x| x.1);
                let name = if class == 0 { format!("R <= {:.0} m", BODY_RADIUS_CLASSES[0]) } else { format!("R {:.0}-{:.0} m", BODY_RADIUS_CLASSES[class - 1], BODY_RADIUS_CLASSES[class]) };
                out.push(format!(
                    "body lane signed: {part} {name}: {secs:.0} vehicle-s (and {:.0} on lanes of no known edges, {:.0} moving sideways); past the lane's inner edge by mean {im:.2} p95 {ip:.2} max {ix:.2} m ({:.0} vehicle-s over 5 cm); past its outer edge by mean {om:.2} p95 {op:.2} max {ox:.2} m ({:.0} vehicle-s over 5 cm)",
                    self.sides_unknown_s[i][class],
                    self.sides_sideways_s[i][class],
                    over(&|x| x.0),
                    over(&|x| x.1)
                ));
            }
        }
        {
            let (n, spline, junction, spline_unknown, junction_unknown, w) = &mut self.lane_edges;
            w.sort_by(|a, b| a.total_cmp(b));
            let pct = |p: f64| w.get(((w.len().max(1) - 1) as f64 * p).round() as usize).copied().unwrap_or(0.0);
            out.push(format!(
                "body lane: lane edges of the {} street lanes of {n} lanes: spline lanes from the profile {spline}, junction paths and editor-only splines' lanes from the pieces they join {junction}, not known: spline lanes {spline_unknown}, junction paths and editor-only lanes {junction_unknown}; known widths p5 {:.2} p50 {:.2} p95 {:.2} m",
                *spline + *junction + *spline_unknown + *junction_unknown,
                pct(0.05),
                pct(0.5),
                pct(0.95)
            ));
            let mut names: Vec<(&String, &usize)> = self.edges_unknown.iter().collect();
            names.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
            let list: Vec<String> = names.iter().take(12).map(|(n, c)| format!("{n} {c}")).collect();
            out.push(format!("body lane: lane edges not known, by file: {}", list.join(", ")));
        }
        {
            let mut by: Vec<String> = self.inner_over_s.iter().map(|((src, bus), s)| format!("{src} {} {s:.0}", if *bus { "bus" } else { "other" })).collect();
            by.sort();
            out.push(format!("body lane signed: inner side past the lane's edge by over {BODY_SIDE_OVER} m in bends (R <= {:.0} m), vehicle-s by lane and vehicle: {}", BODY_RADIUS_CLASSES[2], by.join(", ")));
            let mut worst: Vec<&(f32, String)> = self.inner_worst.values().collect();
            worst.sort_by(|a, b| b.0.total_cmp(&a.0));
            for (d, w) in worst.iter().take(12) {
                out.push(format!("body lane signed:   inner side {d:.2} m past: {w}"));
            }
        }
        let still: Vec<String> = self.open.values().map(|(since, depth, who)| format!("t={since:.1} still at the end, deepest {depth:.2} m: {who}")).collect();
        for e in self.episodes.iter().chain(&still) {
            out.push(format!("body lane:   overlap {e}"));
        }
        out.push(format!("body lane: AI bodies overlapping as drawn: {:.1} pair-s, {} pairs, deepest {:.2} m at t={:.1} #{}/#{}", self.overlap_s, self.pairs.len(), self.deepest.0, self.deepest.1, self.deepest.2, self.deepest.3));
        out
    }
}

impl PlayerWatch {
    /// One frame of `dt` s at run second `t_s`: `player` the player's vehicle, `car` the AI
    /// bus driving it (`LineFollower`), `load_radius` how far the tiles are loaded round the
    /// player (the route ahead is probed that far).
    #[allow(clippy::too_many_arguments)]
    pub fn frame(
        &mut self,
        t_s: f32,
        dt: f32,
        traffic: &Traffic,
        humans: Option<&crate::humans::Humans>,
        world: &crate::scene::World,
        player: &VehicleInstance,
        car: Option<u64>,
        load_radius: f64,
    ) {
        let me = player.position;
        let viewer = traffic.viewer;
        let car_sight = |p: DVec3, r: f64, height: f64| match viewer {
            Some(v) if v.frames(p, r) => {
                if v.beyond_horizon((p - v.pos).length(), height, r) {
                    Sight::Beyond
                } else if traffic.occluded(world, &v, p, r) {
                    Sight::Hidden
                } else {
                    Sight::In
                }
            }
            _ => Sight::Out,
        };
        let near = |p: DVec3| (p - me).length() <= WATCH_RADIUS;
        // (a parked object put down or taken up where a car vanished or appeared: the
        // car parked or pulled out, the picture does not change)
        let parked: HashMap<i64, DVec3> = world
            .parked_objects
            .lock()
            .iter()
            .filter(|(_, o)| (o.pos - me).length() <= WATCH_RADIUS * 2.0)
            .map(|(k, o)| (*k, o.pos))
            .collect();
        let cars: HashMap<u64, (DVec3, f64, f64)> = traffic
            .cars
            .iter()
            .filter(|c| Some(c.id) != car)
            .map(|c| (c.id, (c.vehicle.position, crate::traffic::body_radius(&c.vehicle.ty), crate::traffic::body_height(&c.vehicle.ty))))
            .collect();
        let people: HashMap<u32, (DVec3, bool)> = humans.map(|h| h.everyone().map(|(id, p, ground)| (id, (p, ground))).collect()).unwrap_or_default();
        let loading = traffic.loading_phase();
        if self.started {
            let parked_near = |set: &HashMap<i64, DVec3>, other: &HashMap<i64, DVec3>, p: DVec3, r: f64| {
                set.iter().any(|(k, q)| !other.contains_key(k) && (*q - p).truncate().length() <= r)
            };
            for (&id, &(p, r, height)) in &cars {
                if self.cars.contains_key(&id) {
                    continue;
                }
                let how = if self.cars_ever.contains(&id) {
                    "woken"
                } else if parked_near(&self.parked, &parked, p, r) {
                    "pulled out of its space"
                } else {
                    "spawned"
                };
                self.see(t_s, loading, "car", id, how, p, car_sight(p, r, height), near(p), me, true, how == "pulled out of its space");
            }
            let before: Vec<(u64, (DVec3, f64, f64))> = self.cars.iter().map(|(k, v)| (*k, *v)).collect();
            for (id, (p, r, height)) in before {
                if cars.contains_key(&id) {
                    continue;
                }
                let parked_here = parked_near(&parked, &self.parked, p, r);
                let how = if parked_here {
                    "parked"
                } else if traffic.dormant.iter().any(|d| d.id == id) {
                    "put to sleep"
                } else {
                    "taken off the road"
                };
                self.see(t_s, loading, "car", id, how, p, car_sight(p, r, height), near(p), me, false, parked_here);
            }
            // (people by the people's own sight test)
            let person_sight = |p: DVec3| humans.map(|h| Sight::from(h.figure_sight(world, p))).unwrap_or(Sight::Out);
            for (&id, &(p, ground)) in &people {
                if ground && !self.people.contains_key(&id) {
                    self.see(t_s, loading, "person", id as u64, "put down", p, person_sight(p), near(p), me, true, false);
                }
            }
            let before: Vec<(u32, (DVec3, bool))> = self.people.iter().map(|(k, v)| (*k, *v)).collect();
            for (id, (p, ground)) in before {
                if ground && !people.contains_key(&id) {
                    self.see(t_s, loading, "person", id as u64, "taken away", p, person_sight(p), near(p), me, false, false);
                }
            }
        }
        self.started = true;
        self.cars_ever.extend(cars.keys().copied());
        self.cars = cars;
        self.parked = parked;
        self.people = people;
        if let Some(h) = humans {
            self.people_taken_seen += h.gone().iter().filter(|g| g.1).count();
        }
        // people held at a kerb in sight within the radius (`Traffic::people`: the people's
        // own `at_kerb`; in sight by the people's own test, `Humans::figure_sight`)
        for &(p, _, _, _, _, at_kerb, _, id, _) in &traffic.people {
            let seen = humans.is_some_and(|h| matches!(h.figure_sight(world, p), crate::humans::FigureSight::In | crate::humans::FigureSight::Far));
            if at_kerb && near(p) && seen {
                self.held_s += dt as f64;
                self.held_ids.insert(id);
            }
        }
        let bus = car.and_then(|id| traffic.cars.iter().find(|c| c.id == id));
        // the AI bus's holds
        let standing = bus.map(|c| c.ai.state.speed.abs() < omsi_traffic::driver::STANDING_SPEED);
        let now = match (bus, standing) {
            (Some(c), Some(true)) => {
                let why = match c.ai.last.why.name() {
                    "" if c.at_stop() => "stop",
                    "" => "none",
                    w => w,
                };
                Some((why, c.ai.last.wait_for.or(c.ai.last.lead_id), c.ai.state.lane))
            }
            _ => None,
        };
        let same = matches!((&self.hold, now), (Some(h), Some((w, _, _))) if h.why == w);
        if !same {
            if let Some(h) = self.hold.take() {
                *self.holds_s.entry(h.why).or_default() += (t_s - h.since) as f64;
                self.episodes.push((h.since, t_s - h.since, h.why, h.other, h.lane));
            }
            if let Some((why, other, lane)) = now {
                self.hold = Some(Hold { since: t_s, why, other, lane });
            }
        }
        // the ground along the route ahead
        if let Some(c) = bus {
            if t_s >= self.tiles_next {
                self.tiles_next = t_s + TILES_EVERY;
                self.tiles_probes += 1;
                let net = &traffic.net;
                let st = &c.ai.state;
                // the bus's whole route from its lane on, each lane from where it begins
                // along the way (`AiState::route_distance`: a lane changed over to counts from
                // the lane beside it), as far as the load radius
                let way: Vec<(usize, f32)> = (st.route_index..st.route.len())
                    .map(|k| (st.route[k], st.route_distance(net, k, 0.0)))
                    .take_while(|&(_, d0)| (d0 as f64) < load_radius)
                    .collect();
                let mut edge: Option<(f64, DVec3, &'static str)> = None;
                'lanes: for &(l, d0) in &way {
                    let lane = &net.lanes[l];
                    let mut along = 0.0f64;
                    for w in lane.points.windows(2) {
                        along += (w[1] - w[0]).length();
                        let d = d0 as f64 + along;
                        if d > 0.0 && !world.has_ground(w[1].x, w[1].y) {
                            edge = Some((d, w[1], "no ground"));
                            break 'lanes;
                        }
                    }
                }
                // (the route holds the trip's lanes as far as the loaded tiles have them,
                // `BusSetup::route`; while the trip goes on past them, `route_open`, its last
                // lane's end is the edge of what is loaded)
                if edge.is_none() && c.route_open() && way.len() == st.route.len() - st.route_index {
                    if let Some(&(l, d0)) = way.last() {
                        let d = d0 as f64 + net.lanes[l].length() as f64;
                        if d < load_radius {
                            edge = Some((d, net.lanes[l].end(), "the end of the route's loaded lanes"));
                        }
                    }
                }
                if let Some((d, p, what)) = edge {
                    if self.tiles_min.is_none_or(|m| d < m.0) {
                        self.tiles_min = Some((d, t_s, p, what));
                    }
                }
            }
        }
        if !loading {
            self.bodies.frame(t_s, dt, traffic);
        }
        if t_s >= self.next_line {
            self.next_line = t_s + POSITION_EVERY;
            let held = self.hold.as_ref().map(|h| format!(", held ({}) {:.0} s", h.why, t_s - h.since)).unwrap_or_default();
            self.lines.push(format!(
                "t={t_s:.0} at ({:.1}, {:.1}) heading {:.0} {:.1} km/h{}{held}",
                me.x,
                me.y,
                player.heading,
                player.physics.speed.abs() * 3.6,
                bus.map(|c| format!(", lane {} s {:.1}", c.ai.state.lane, c.ai.state.s)).unwrap_or_default()
            ));
        }
    }

    /// An appearance (`came`) or disappearance of `what` #`id` at `p`, its `sight` for the
    /// eye, `near` within `WATCH_RADIUS`: counted when in the view cone within the radius,
    /// or a disappearance in the cone at any distance; neither while the map loads
    /// (`Traffic::loading_phase`), nor a car that parked or pulled out (`unchanged`: its
    /// parked object stands where it was). One in sight (inside its horizon) is logged. A
    /// person put down in plain view past `humans::APPEAR_IN_VIEW_BEYOND` (`Sight::Far`) is
    /// counted apart, not logged; one taken away there is in sight (the people's rule for
    /// taking somebody away has no such distance, `figure_in_sight`).
    #[allow(clippy::too_many_arguments)]
    fn see(&mut self, t: f32, loading: bool, what: &'static str, id: u64, how: &'static str, p: DVec3, sight: Sight, near: bool, me: DVec3, came: bool, unchanged: bool) {
        let kind: &'static str = match (what, came) {
            ("car", true) => "car appeared",
            ("car", false) => "car vanished",
            (_, true) => "person appeared",
            (_, false) => "person vanished",
        };
        let sight = if sight == Sight::Far && !came { Sight::In } else { sight };
        if loading || unchanged || sight == Sight::Out || (came && !near) {
            *self.uncounted.entry(kind).or_default() += 1;
            return;
        }
        let dist = (p - me).length();
        if sight == Sight::In {
            log::info!("player watch t={t:.1}: {kind} IN SIGHT #{id} ({how}) {dist:.0} m from the player at ({:.1}, {:.1})", p.x, p.y);
        }
        self.events.push(Event { t, what: kind, id, how, dist, pos: p, sight });
    }

    /// The summary: the lines for the run's log.
    pub fn report(&mut self, t_end: f32, driven: Option<&LineFollower>) -> Vec<String> {
        if let Some(h) = self.hold.take() {
            *self.holds_s.entry(h.why).or_default() += (t_end - h.since) as f64;
            self.episodes.push((h.since, t_end - h.since, h.why, h.other, h.lane));
        }
        let mut out = Vec::new();
        match driven {
            Some(f) => out.push(format!(
                "player watch: moving (--follow-line), AI bus {:?}{}{}",
                f.car,
                f.lost.map(|(t, why)| format!(", lost at {t:.1} s ({why})")).unwrap_or_default(),
                f.refused.as_ref().filter(|_| f.car.is_none()).map(|e| format!(", never put on the trip: {e}")).unwrap_or_default()
            )),
            None => out.push("player watch: standing".to_string()),
        }
        out.extend(self.lines.iter().map(|l| format!("player watch: {l}")));
        let count = |kind: &str, near: bool, sight: Sight| self.events.iter().filter(|e| e.what == kind && (e.dist <= WATCH_RADIUS) == near && e.sight == sight).count();
        for sight in [Sight::In, Sight::Hidden, Sight::Beyond] {
            out.push(format!(
                "player watch: in the view cone {}, within {WATCH_RADIUS:.0} m: cars appeared {} vanished {}, people appeared {} vanished {}; farther: cars vanished {} people vanished {}",
                sight.name(),
                count("car appeared", true, sight),
                count("car vanished", true, sight),
                count("person appeared", true, sight),
                count("person vanished", true, sight),
                count("car vanished", false, sight),
                count("person vanished", false, sight),
            ));
        }
        // (the people's rule lets somebody be put down in plain view past
        // `humans::APPEAR_IN_VIEW_BEYOND`)
        let past = self.events.iter().filter(|e| e.sight == Sight::Far).count();
        out.push(format!("player watch: of the people appeared in sight, {past} past the people's in-view appearance distance ({:.0} m, not listed)", crate::humans::APPEAR_IN_VIEW_BEYOND));
        out.push(format!("player watch: people taken in sight by the people's own test (`Humans::gone`): {}", self.people_taken_seen));
        out.push(format!("player watch: not counted (out of the view cone, appeared beyond the radius, a car parked or pulled out, the map loading): {:?}", self.uncounted));
        for e in self.events.iter().filter(|e| e.sight == Sight::In) {
            out.push(format!("player watch:   t={:.1} {} #{} ({}) {:.0} m at ({:.1}, {:.1})", e.t, e.what, e.id, e.how, e.dist, e.pos.x, e.pos.y));
        }
        out.push(format!("player watch: people held at a kerb in sight within {WATCH_RADIUS:.0} m: {:.0} person-s, {} people", self.held_s, self.held_ids.len()));
        out.push(match self.tiles_min {
            Some((d, t, p, what)) => format!("player watch: unloaded ground along the route ahead: nearest {d:.0} m at {t:.0} s ({what}, at ({:.0}, {:.0})), {} probes", p.x, p.y, self.tiles_probes),
            None if self.tiles_probes > 0 => format!("player watch: unloaded ground along the route ahead: none within the load radius, {} probes", self.tiles_probes),
            None => "player watch: unloaded ground along the route ahead: no route (standing)".to_string(),
        });
        if !self.holds_s.is_empty() {
            out.push(format!("player watch: the bus's holds (s by reason): {}", self.holds_s.iter().map(|(k, v)| format!("{k} {v:.1}")).collect::<Vec<_>>().join(", ")));
            for &(since, secs, why, other, lane) in &self.episodes {
                out.push(format!("player watch:   held t={since:.1} for {secs:.1} s: {why}{} lane {lane}", other.map(|o| format!(" #{o}")).unwrap_or_default()));
            }
        }
        out.extend(self.bodies.report());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus_service::{BusService, Ctx, Stop};
    use omsi_sim::ai_motion::MotionKind;
    use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network, TrafficLightController};
    use omsi_traffic::{Simulation, Vehicle, VehicleClass, Why};

    const DT: f32 = 1.0 / 30.0;

    fn lane(points: Vec<DVec3>, id: i64, source: u32) -> omsi_sim::traffic::Lane {
        let mut l = LaneBuilder::polyline(points, LaneKind::Street, 3.5);
        l.source = source;
        l.key = Some(LaneKey { tile: (0, 0), id, path: 0 });
        l
    }

    /// North 150 m to a light, 20 m through it, a right bend of 30 m radius, east 2 km.
    fn route() -> Network {
        let mut net = Network::default();
        net.lanes.push(lane(vec![DVec3::ZERO, DVec3::new(0.0, 150.0, 0.0)], 1, 1));
        let mut light = lane(vec![DVec3::new(0.0, 150.0, 0.0), DVec3::new(0.0, 170.0, 0.0)], 9, 2);
        light.priority = 128.0;
        light.traffic_light = Some((0, 0));
        net.lanes.push(light);
        let r = 30.0;
        let bend: Vec<DVec3> = (0..=24)
            .map(|k| {
                let a = k as f64 / 24.0 * std::f64::consts::FRAC_PI_2;
                DVec3::new(r - r * a.cos(), 170.0 + r * a.sin(), 0.0)
            })
            .collect();
        net.lanes.push(lane(bend, 2, 1));
        net.lanes.push(lane(vec![DVec3::new(r, 200.0, 0.0), DVec3::new(2000.0, 200.0, 0.0)], 3, 1));
        net.link(1.5);
        net.compute_conflicts();
        net
    }

    /// A two-axle bus, 12 m long, as the player's `VehicleInstance` and the AI's body see it.
    fn bus() -> VehicleInstance {
        use omsi_vehicle::vehicle::Axle;
        let mut def = omsi_vehicle::Vehicle { mass: 11.0, inv_min_turn_radius: 0.1, cog_height: 1.0, ..Default::default() };
        def.axles = vec![Axle { long: 3.0, max_width: 2.0, spring: 300.0, damper: 20.0, wheel_diameter: 1.0, ..Default::default() }, Axle { long: -2.9, max_width: 2.0, spring: 300.0, damper: 20.0, wheel_diameter: 1.0, ..Default::default() }];
        let program = omsi_script::compile(&omsi_script::CompileInput::default());
        let ty = std::sync::Arc::new(omsi_sim::VehicleType {
            def,
            model: Default::default(),
            model_dir: std::path::PathBuf::new(),
            program: std::sync::Arc::new(program),
            meshes: Vec::new(),
            paint_schemes: Vec::new(),
            texchanges: Vec::new(),
            wheel_meshes: Vec::new(),
            suspension_axles: Vec::new(),
            missing_packs: Vec::new(),
            mesh_bounds: Vec::new(),
            mesh_boxes: Vec::new(),
        });
        VehicleInstance::new(ty, omsi_sim::VehicleHost::new(Default::default()))
    }

    /// The player's bus put where an AI timetable bus driving the route is (`pose_player`),
    /// the bus driven as `Traffic::tick` drives one (its service's stop, pace and bay into
    /// the driver, `Simulation::tick`, its body stepped along its way): it keeps within its
    /// lane on the straight and through the bend, stops at its stop with the doors open and
    /// at the red light, and goes on when the light turns green.
    #[test]
    fn the_player_follows_its_route_in_its_lane_and_stops_at_its_stop_and_at_a_red_light() {
        let net = route();
        // red for 90 s, then green
        let ctl = TrafficLightController::new(vec![vec![(0, 90.0), (6, 90.0)]], 180.0);
        let mut sim = Simulation::new(&net, vec![ctl]);
        let mut st = AiState::new(0, 10.0, 1);
        (st.front, st.rear, st.length, st.max_speed_kmh, st.veh_type) = (6.0, 6.0, 12.0, 50.0, 0);
        st.route = vec![0, 1, 2, 3];
        let mut ai = vec![Vehicle::new(1, VehicleClass::ScheduledBus, st, 1.25, 7919)];
        ai[0].fresh = 0.0;
        // (its stop on the first straight, and the next one far down the last lane: the trip
        // goes on past the bend)
        let stop = Stop { ri: 0, s: 60.0, bay: 0.0, depart: 0.0, id: 7, side: 0.0 };
        let mut service = BusService::new(vec![stop, Stop { ri: 3, s: 1500.0, id: 8, ..stop }]);
        let mut ai_vehicle = bus();
        let mut player = bus();
        let mut body = AiBody::new(&player.ty.def, MotionKind::Road);
        let half_lane = 3.5 / 2.0;
        let (mut worst, mut served_at, mut stood_at_red, mut on_red) = (0.0f64, None, false, false);
        let (mut t, mut stopped) = (0.0f64, 0.0f32);
        while t < 200.0 && !(ai[0].state.lane == 3 && ai[0].state.s > 50.0) {
            let way = omsi_traffic::perception::way_lanes(&net, &ai[0].state, 200.0);
            let ctx = Ctx { net: &net, way: &way, day_time: t, dt: DT, id: 1, stopped, passing: false, lead_close: false, follower_close: false, wanted: None, debug: false, roadside: false, queue_behind_secs: 0.0, v_nom: 13.9, people_hold: false, close_behind: false, covers_through: false };
            let stop_at = service.step(&mut ai[0].state, &mut ai_vehicle, &ctx);
            ai[0].external.stop_at = stop_at;
            ai[0].external.at_stop = service.at_stop();
            ai[0].external.pace = Some(service.pace);
            ai[0].external.bay = service.bay_wish.or(service.at_stop().then_some(ai[0].state.lateral_target));
            sim.tick(&net, DT, &mut ai);
            assert!(sim.removed.is_empty(), "the bus was taken off its route at {t:.1} s");
            let (state, speed) = (&ai[0].state, ai[0].state.speed);
            let way_point = |d: f32| state.way_point(&net, d);
            if t == 0.0 {
                body.place(&way_point, None, None, speed);
            } else {
                body.step(DT, speed, &way_point, None, None);
            }
            pose_player(&body, speed, &[], &mut player);
            stopped = if speed < 0.3 { stopped + DT } else { 0.0 };
            // within its lane: the player's origin off the middle line of the lane under it
            let off = state.route.iter().filter_map(|&l| net.lanes[l].nearest_point(player.position)).map(|(_, d)| d).fold(f64::MAX, f64::min);
            worst = worst.max(off);
            assert!((player.physics.speed - speed).abs() < 1e-6);
            let red = sim.lights[0].state(0) < 6;
            if service.at_station() && player.physics.speed < 0.3 && served_at.is_none() {
                served_at = Some((t, player.position.y));
            }
            if red && state.lane == 0 && speed < 0.3 && ai[0].last.why == Why::Light {
                stood_at_red = true;
            }
            if red && state.lane != 0 {
                on_red = true;
            }
            t += DT as f64;
        }
        assert!(worst <= half_lane, "{worst:.2} m off its lane's middle line");
        let (when, y) = served_at.expect("never stood at its stop (at_station, speed under 0.3 m/s)");
        // (at its stop point, where `BusService::step` brings it to rest)
        assert!((y - 60.0).abs() < 2.0, "served its stop at y {y:.1} ({when:.1} s)");
        assert!(stood_at_red, "never stood at the red light");
        assert!(!on_red, "drove past the red light");
        assert!(ai[0].state.lane == 3, "never got through the bend after the green ({t:.0} s, lane {})", ai[0].state.lane);
    }

    /// End to end as the harness runs it: `LineFollower::place` asks the timetable for the
    /// player's trip (`Schedule::drive_player_trip`), which puts the departure's AI vehicle
    /// on the trip's lane under the player's bus and hands it the departure there
    /// (`spawn_departure(.., here = true)`: the route from the lane it stands on, the stops
    /// from there on, the station behind it none of them); `Traffic::tick` drives it (its
    /// `BusService`, the light, its body) and `LineFollower::pose` puts the player's bus where
    /// it is: within its lane, at its stop with the doors open, at the red light, through the
    /// bend after the green. The traffic's network, the light and the stop objects come from
    /// a world as its tiles bring them (`World::lanes`, `traffic_lights`, `object_positions`).
    #[test]
    fn the_follower_puts_the_players_trip_on_where_its_bus_stands_and_drives_it_to_its_stop_and_the_light() {
        use crate::scene::World;
        let dir = std::env::temp_dir().join(format!("openomsi-follow-line-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("global.cfg"), "[name]\nFollow\n").unwrap();
        let world = World::open(&dir, &dir.join("global.cfg"), 20261001).unwrap();
        world.lanes.lock().extend(route().lanes);
        world.lane_tiles.lock().push((0, 0));
        // red for 90 s, then green
        world.traffic_lights.lock().push(TrafficLightController::new(vec![vec![(0, 90.0), (6, 90.0)]], 180.0));
        // stations at the kerb (on the right): 30 m behind the player's bus, 30 m ahead of it, far east
        let positions = [(6, DVec3::new(2.0, 0.0, 0.0)), (7, DVec3::new(2.0, 60.0, 0.0)), (8, DVec3::new(1500.0, 198.0, 0.0))];
        for (id, p) in positions {
            world.object_positions.lock().insert(id, (p, [0.0; 3]));
        }
        let mut traffic = Traffic::new(&dir, &world, 0).unwrap();
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::NOOP;
        descriptor.backend_options.noop = wgpu::NoopBackendOptions { enable: true };
        let instance = wgpu::Instance::new(descriptor);
        let renderer = pollster::block_on(omsi_render::Renderer::new_with(&instance, None, Some(wgpu::TextureFormat::Rgba8UnormSrgb), omsi_render::RenderOptions { msaa: 1, shadow_size: 1024, ..Default::default() })).expect("noop renderer");
        let mut scene = renderer.new_scene();
        let departure = 8.0 * 3600.0;
        let steps = [(1, 0, 150.0), (9, 0, 20.0), (2, 0, 47.0), (3, 0, 1970.0)];
        let (mut schedule, duty) = crate::schedule::tests::one_trip(&world, "t104", &steps, &[6, 7, 8], departure, bus().ty.clone());
        // (late on its trip: it leaves its stop once its people are aboard)
        traffic.day_time = departure + 400.0;
        let net = &traffic.net;
        let lane_of = |id: i64| net.by_key[&LaneKey { tile: (0, 0), id, path: 0 }][0];
        let (first, last) = (lane_of(1), lane_of(3));
        let mut player = bus();
        let (p, h) = net.lanes[first].at(30.0);
        (player.position, player.heading) = (p, h as f64);
        let mut follower = LineFollower::default();
        follower.place(0.0, &mut schedule, &duty, &world, &mut traffic, &renderer, &mut scene, &player);
        let id = follower.car.unwrap_or_else(|| panic!("not put on the trip: {:?}", follower.refused));
        let c = traffic.cars.iter().find(|c| c.id == id).unwrap();
        // on the lane under the player's bus, where it stands, its route from there
        assert_eq!((c.ai.state.lane, c.ai.state.route.first()), (first, Some(&first)));
        assert!((c.ai.state.s - 30.0).abs() < 0.5, "put at s {:.1}", c.ai.state.s);
        let stops: Vec<i64> = c.bus.as_ref().unwrap().stops.iter().map(|s| s.id).collect();
        assert_eq!(stops, vec![7, 8], "the stops from the bus on");
        let half_lane = 3.5 / 2.0;
        let (mut worst, mut served_at, mut stood_at_red, mut on_red) = (0.0f64, None, false, false);
        let mut t = 0.0f32;
        loop {
            traffic.tick(DT, None);
            assert!(follower.pose(t, &traffic, &mut player), "lost at {t:.1} s: {:?}", follower.lost);
            let c = traffic.cars.iter().find(|c| c.id == id).unwrap();
            let st = &c.ai.state;
            if st.lane == last && st.s > 50.0 || t > 300.0 {
                break;
            }
            let off = st.route.iter().filter_map(|&l| traffic.net.lanes[l].nearest_point(player.position)).map(|(_, d)| d).fold(f64::MAX, f64::min);
            worst = worst.max(off);
            assert!((player.physics.speed - st.speed).abs() < 1e-6);
            if c.at_stop() && player.physics.speed < 0.3 && served_at.is_none() {
                served_at = Some((t, player.position.y));
            }
            let red = traffic.sim.lights[0].state(0) < 6;
            if red && st.lane == first && st.speed < 0.3 && c.ai.last.why == Why::Light {
                stood_at_red = true;
            }
            if red && st.lane != first {
                on_red = true;
            }
            t += DT;
        }
        let _ = std::fs::remove_dir_all(&dir);
        assert!(t <= 300.0, "never got through the bend after the green");
        assert!(worst <= half_lane, "{worst:.2} m off its lane's middle line");
        let (when, y) = served_at.expect("never stood at its stop (at_station, speed under 0.3 m/s)");
        assert!((y - 60.0).abs() < 3.0, "served its stop at y {y:.1} ({when:.1} s)");
        assert!(stood_at_red, "never stood at the red light");
        assert!(!on_red, "drove past the red light");
    }
}
