//! What a driver sees, as plain values: the vehicle ahead with its gap, the next signal,
//! the next junction, the lanes beside with their gaps, people in the way, the oncoming
//! lane and how long it stays clear. Read-only over the whole traffic; the plan reads
//! nothing else.
//!
//! The car's way ahead is sampled once per frame (`WayPts`) and every test against a
//! body, a player or a person reuses the samples; the vehicles and the people near the
//! car come from the simulation's grids, not from a scan of everything.

use crate::junction::Junctions;
use crate::sim::{Simulation, Vehicle};
use glam::{DVec2, DVec3};
use omsi_sim::collision::Obb;
use omsi_sim::traffic::{arrival_time, AiState, Aspect, LaneKind, Network, TrafficLightController};

/// What a headway is to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Seen {
    /// Another AI vehicle on the way (index, this frame only).
    Vehicle(usize),
    /// Another AI vehicle's body across the way (index, this frame only), not on the
    /// way's lanes.
    Body(usize),
    Player(u32),
    Parked,
}

/// The thing ahead: the gap from the front bumper to it, how it moves along the way.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Headway {
    pub gap: f32,
    pub speed: f32,
    pub acc: f32,
    pub what: Seen,
    /// Seconds the thing has stood (a parked car: for ever).
    pub standing_for: f32,
    /// Its half width (for passing it).
    pub half_width: f32,
    /// From its rear to its front (m).
    pub length: f32,
    /// It is itself held by something (a lead, a light, a junction): a queue, not an
    /// obstacle to pass.
    pub held: bool,
    /// A bus serving its stop.
    pub at_stop: bool,
    /// Where its middle lies across my lane (m, + right of the lane's middle).
    pub lateral: f32,
}

impl Headway {
    fn nearer(a: Option<Headway>, b: Option<Headway>) -> Option<Headway> {
        match (a, b) {
            (Some(x), Some(y)) => Some(if y.gap < x.gap { y } else { x }),
            (x, None) => x,
            (None, y) => y,
        }
    }

    pub fn lead(&self) -> omsi_sim::traffic::Lead {
        omsi_sim::traffic::Lead { gap: self.gap, speed: self.speed, acc: self.acc }
    }

    /// Something that stands and will go on standing: worth going round.
    pub fn standing_obstacle(&self) -> bool {
        if self.speed >= 0.3 {
            return false;
        }
        match self.what {
            Seen::Parked => true,
            Seen::Player(_) => self.standing_for > 4.0,
            Seen::Vehicle(_) | Seen::Body(_) => (self.at_stop && self.standing_for > 0.5) || (!self.held && self.standing_for > 4.0) || self.standing_for > 25.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignalView {
    /// Distance from the origin to the stop line.
    pub dist: f32,
    pub ctl: usize,
    pub light: usize,
    pub aspect: Aspect,
    pub until_go: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JunctionView {
    pub junction: usize,
    pub movement: usize,
    /// Distance from the origin to the start of the movement's lane (negative inside).
    pub entry: f32,
    pub inside: bool,
    /// The pieces of the junction the way runs through, each with how far along the way
    /// through its start lies (`Junctions::chain_along`).
    pub chain: Vec<(usize, f32)>,
    /// Distance from the origin to the first meeting place on the way through (the end
    /// of the way through when it has no foe).
    pub conflict_start: f32,
    /// Room past the junction's conflicts for this vehicle, measured along its own way.
    pub exit_room: bool,
    /// Distance from the origin to the first place where the way meets another way (not
    /// one of its own pieces, nor a piece forking where one of them starts): a light that
    /// holds the car at this junction stands before it (the end of the way through when
    /// it meets nobody).
    pub light_line: f32,
    /// The light at this junction's line: (controller, light, distance from the origin to
    /// the lit lane's start, negative once passed). The first lit lane of the way, from the
    /// lane leading in, before `light_line`; for a car inside the junction past it, the
    /// lit lane it came through (looked for back along the pieces of the junction).
    pub light: Option<(usize, usize, f32)>,
    /// The front is over that light's line (by more than half a metre), or the lit lane lies
    /// behind the car.
    pub past_line: bool,
    /// Past its line (passed while the light let it in), the light no longer green, and unable
    /// to stop before the first meeting place with another way or already at it: in the box
    /// when its light turned, it clears it. That light holds it no more (`clears_past`; a later line on the
    /// way, a left-turn waiting area's under another light of the controller, holds it as any)
    /// and it goes on through, waiting only for foes committed or clearing themselves (the
    /// table decides between those); a fresh green waits for it (`Claim::clearing`).
    pub clearing: bool,
    /// The way through's length (m, along the chain's pieces).
    pub way_len: f32,
}

impl JunctionView {
    /// Clearing the box, the car is not held by `sig` when that is the light whose line it
    /// passed and the line lies behind its front (`front`: from the origin to the front).
    /// (Review7: it ignored every light of the controller on its way through and drove over
    /// the red of a left-turn waiting area's second line, 41 such ways in viz3.)
    pub fn clears_past(&self, sig: &SignalView, front: f32) -> bool {
        self.clearing && self.light.is_some_and(|(ctl, li, _)| (ctl, li) == (sig.ctl, sig.light)) && sig.dist < front
    }
}

/// The light of junction `j` that a car on its pieces came through: back from `lane` along
/// the lanes leading into it (the car's own previous lane first), within the junction and the
/// lane leading into it, at most eight pieces or 80 m. (controller, light, distance from the
/// origin to the lit lane's start; `back` is the distance to `lane`'s start, <= 0.) None when
/// none is found or two different lights are (the way it came is not known).
fn light_behind(net: &Network, js: &Junctions, j: usize, lane: usize, back: f32, prev_lane: Option<usize>) -> Option<(usize, usize, f32)> {
    let mut found: Option<(usize, usize, f32)> = None;
    let mut ambiguous = false;
    // (lane, distance from the origin to its start, steps)
    let mut stack: Vec<(usize, f32, usize)> = vec![(lane, back, 0)];
    while let Some((l, d, n)) = stack.pop() {
        if n >= 8 || d < -80.0 {
            continue;
        }
        let prevs = net.prev.get(l).map(|v| v.as_slice()).unwrap_or(&[]);
        // the way it came: its previous lane, when it is one of them
        let only: Option<usize> = if n == 0 { prev_lane.filter(|p| prevs.contains(p)) } else { None };
        for &p in prevs {
            if only.is_some_and(|o| o != p) {
                continue;
            }
            let Some(pl) = net.lanes.get(p) else { continue };
            let dp = d - pl.length();
            if let Some((ctl, li)) = pl.traffic_light {
                match found {
                    Some((c, i, _)) if (c, i) != (ctl, li) => ambiguous = true,
                    Some(_) => {}
                    None => found = Some((ctl, li, dp)),
                }
                continue;
            }
            if js.of_lane.get(&p).map(|x| x.0) == Some(j) {
                stack.push((p, dp, n + 1));
            }
        }
    }
    if ambiguous {
        None
    } else {
        found
    }
}

/// A lane beside the current one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeighbourView {
    pub lane: usize,
    /// 1 left, 2 right.
    pub dir: i32,
    /// Nearest vehicle ahead there: (gap from my front, its speed).
    pub ahead: Option<(f32, f32)>,
    /// Nearest vehicle behind there: (gap to my rear, its speed).
    pub behind: Option<(f32, f32)>,
    /// How far the lane and its successors run on from beside me (m).
    pub runs_on: f32,
    /// A turn (1 left, 2 right) the lane leads into within 150 m, 0 none.
    pub turn: i32,
    /// A vehicle is passing on this lane against its direction.
    pub foreign: bool,
    /// This vehicle may drive there (the path's rules); a route may still lead there.
    pub allowed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OncomingView {
    pub lane: usize,
    /// Distance along it level with me.
    pub s: f32,
    /// Lateral offset of its middle from my lane (signed, m).
    pub side: f32,
    /// Seconds before the first oncoming vehicle is level with me.
    pub clear_for: f32,
    /// Something parked or standing on it in the stretch a pass would use.
    pub blocked: bool,
}

#[derive(Debug, Clone, Default)]
pub struct Perception {
    /// Lanes ahead: (lane, distance from the origin to its start).
    pub way: Vec<(usize, f32)>,
    pub leader: Option<Headway>,
    /// A vehicle's body across the way that held the car (its id), for mutual waits.
    pub body_block: Option<u64>,
    pub signal: Option<SignalView>,
    pub junction: Option<JunctionView>,
    pub left: Option<NeighbourView>,
    pub right: Option<NeighbourView>,
    /// Distance from the origin to stop for a person in the way.
    pub people_stop: Option<f32>,
    /// Distance to stop for a pedestrian at a crossing of the junction.
    pub zebra_stop: Option<f32>,
    /// A route lane change due: (lane, dir).
    pub route_change: Option<(usize, i32)>,
    /// Metres the way runs on before it ends.
    pub road_left: f32,
    /// A parked car the way should swerve round: the lateral offset needed.
    pub kerb_swerve: Option<f32>,
    pub oncoming: Option<OncomingView>,
    /// A player's vehicle ahead indicating to pull out towards the traffic: the distance
    /// from my origin to its rear.
    pub letting_out: Option<f32>,
    /// A vehicle on another lane that reaches the joint ahead first.
    pub merging: Option<Headway>,
    /// The current lane's limit at the driver's desire (m/s).
    pub v_want: f32,
}

/// The lanes ahead along the car's way with the distance from the origin to each one's
/// start, within `within` metres. The lane being changed onto replaces the current one
/// once the move is under way.
pub fn way_lanes(net: &Network, st: &AiState, within: f32) -> Vec<(usize, f32)> {
    let mut out = Vec::new();
    let committed = st.change.map(|c| c.t > 0.4 || (c.bypass && c.wait <= 0.0)).unwrap_or(false);
    let (first, s, rest): (usize, f32, Vec<usize>) = match (committed, st.change) {
        (true, Some(c)) => (c.to, c.s_to, st.change_plan.clone()),
        _ => (st.lane, st.s, st.upcoming().collect()),
    };
    let Some(l) = net.lanes.get(first) else { return out };
    let mut d = -s;
    out.push((first, d));
    d += l.length();
    for n in rest {
        if d > within {
            break;
        }
        let Some(nl) = net.lanes.get(n) else { break };
        out.push((n, d));
        d += nl.length();
    }
    out
}

/// The car's way ahead, sampled: (distance from the origin, point, forward direction), with
/// the height of each sample.
struct WayPts {
    pts: Vec<(f32, DVec2, DVec2)>,
    zs: Vec<f64>,
}

/// Sample spacing of the way (m) ...
const WAY_STEP: f32 = 1.5;
/// ... over this many metres, twice that beyond (a lead far ahead).
const WAY_FINE: f32 = 60.0;

/// Margin round my corridor (m): a body that comes within it of where my body will be is a
/// lead (`lead_by_bodies`).
pub const LEAD_MARGIN: f64 = 0.3;

impl WayPts {
    fn sample(net: &Network, st: &AiState, upto: f32) -> WayPts {
        let n = (upto.min(WAY_FINE) / WAY_STEP).ceil() as usize + ((upto - WAY_FINE).max(0.0) / (2.0 * WAY_STEP)).ceil() as usize + 2;
        let mut pts = Vec::with_capacity(n);
        let mut zs = Vec::with_capacity(n);
        let mut prev = st.way_point(net, -WAY_STEP).truncate();
        let mut d = 0.0f32;
        for _ in 0..n {
            let q = st.way_point(net, d);
            zs.push(q.z);
            let q2 = q.truncate();
            let fwd = (q2 - prev).normalize_or_zero();
            pts.push((d, q2, fwd));
            prev = q2;
            d += if d < WAY_FINE { WAY_STEP } else { 2.0 * WAY_STEP };
        }
        WayPts { pts, zs }
    }

    /// Where box `b` comes into the corridor of half width `w` round the way (no further than
    /// `upto` along it): (distance from the origin along the way to its near edge, the index
    /// of the sample whose slice it enters, the slice's forward direction). Each slice of the
    /// way (between two samples) clips the box in its own frame, so a box beside the way,
    /// however near, is no hit, and one across it is met at its near edge.
    fn corridor_hit(&self, b: &Obb, w: f64, upto: f32) -> Option<(f32, usize, DVec2)> {
        let corners = b.corners();
        let reach = b.half.length() + w;
        for k in 0..self.pts.len().saturating_sub(1) {
            let (d, q, _) = self.pts[k];
            if d > upto {
                break;
            }
            let (d1, q1, _) = self.pts[k + 1];
            let step = (d1 - d) as f64;
            if (b.center - q).length() > reach + step + 0.5 {
                continue;
            }
            let f = (q1 - q).normalize_or_zero();
            if f == DVec2::ZERO {
                continue;
            }
            let r = DVec2::new(f.y, -f.x);
            let poly: Vec<DVec2> = corners.iter().map(|c| {
                let v = *c - q;
                DVec2::new(v.dot(f), v.dot(r))
            }).collect();
            let Some((x0, x1)) = clip_strip(&poly, w) else { continue };
            // (the first slice reaches back behind the origin: a body beside the car)
            let lo = if k == 0 { f64::NEG_INFINITY } else { 0.0 };
            if x1 < lo || x0 > step {
                continue;
            }
            return Some((d + x0 as f32, k, f));
        }
        None
    }
}

/// The x extent (min, max) of the polygon `poly` cut to the strip |y| <= `w`, if anything of it
/// lies in the strip.
fn clip_strip(poly: &[DVec2], w: f64) -> Option<(f64, f64)> {
    fn clip(pts: &[DVec2], keep: impl Fn(DVec2) -> f64) -> Vec<DVec2> {
        let mut out = Vec::with_capacity(pts.len() + 2);
        for k in 0..pts.len() {
            let (a, b) = (pts[k], pts[(k + 1) % pts.len()]);
            let (fa, fb) = (keep(a), keep(b));
            if fa >= 0.0 {
                out.push(a);
            }
            if (fa >= 0.0) != (fb >= 0.0) {
                out.push(a + (b - a) * (fa / (fa - fb)));
            }
        }
        out
    }
    let upper = clip(poly, |p| w - p.y);
    if upper.is_empty() {
        return None;
    }
    let both = clip(&upper, |p| p.y + w);
    if both.is_empty() {
        return None;
    }
    Some(both.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), p| (a.min(p.x), b.max(p.x))))
}

fn look_ahead(v: f32) -> f32 {
    (v * v / 3.0 + 2.0 * v + 20.0).clamp(70.0, 150.0)
}

/// The vehicle ahead on the way, through the lanes ahead (by lane registration: an aircraft's,
/// whose way has no body test).
fn leader_on_way(sim: &Simulation, vehicles: &[Vehicle], i: usize, way: &[(usize, f32)], look: f32) -> Option<Headway> {
    let me = &vehicles[i];
    let st = &me.state;
    let signalling = st.change.map(|c| c.wait > 0.0 && c.t <= 0.0 && !c.bypass).unwrap_or(false);
    let passing = me.passing.filter(|p| !p.aborted);
    let mut best: Option<Headway> = None;
    let consider = |best: &mut Option<Headway>, lane_d0: f32, occ: &crate::sim::Occupant, limit: f32| {
        if occ.vehicle == i {
            return;
        }
        let o = &vehicles[occ.vehicle];
        let d_rear = lane_d0 + occ.s - o.state.rear;
        let gap = d_rear - st.front;
        if gap < -0.5 || gap > limit {
            return;
        }
        // side by side counts only when the two overlap sideways
        let my_lat = st.lateral_ahead((gap + st.front).max(0.0));
        if (my_lat - occ.lateral).abs() > me.half_width + o.half_width + 0.3 {
            return;
        }
        // the one behind that follows me is not my lead
        if o.lead_id == Some(me.id) {
            let rel = (o.pos - me.pos).truncate();
            let h = (me.heading as f64).to_radians();
            if rel.dot(DVec2::new(h.sin(), h.cos())) < 0.0 {
                return;
            }
        }
        if let Some((id, until)) = me.ignore_lead {
            if id == o.id && sim.time < until {
                return;
            }
        }
        // the standing thing I am passing
        if let Some(p) = passing {
            let standing = o.state.speed < 0.3;
            if standing && st.odometer + gap + st.front < p.until + 1.0 {
                return;
            }
        }
        let standing_for = if o.external.at_stop { o.external.standing_for.max(o.stopped) } else { o.stopped };
        let held = o.last.held || o.last.stop.is_some();
        let h = Headway { gap, speed: if occ.foreign { -o.state.speed } else { o.state.speed }, acc: o.state.acc, what: Seen::Vehicle(occ.vehicle), standing_for, half_width: o.half_width, length: o.length(), held, at_stop: o.external.at_stop, lateral: occ.lateral };
        if best.map(|b| h.gap < b.gap).unwrap_or(true) {
            *best = Some(h);
        }
    };
    for &(l, d0) in way {
        if d0 > look {
            break;
        }
        for occ in sim.occupancy.get(&l).map(|v| v.as_slice()).unwrap_or(&[]) {
            consider(&mut best, d0, occ, look);
        }
    }
    // the lane being changed onto, before the move is under way
    if let Some(c) = st.change {
        if !way.iter().any(|w| w.0 == c.to) {
            let limit = if signalling { 6.0 } else { look };
            for occ in sim.occupancy.get(&c.to).map(|v| v.as_slice()).unwrap_or(&[]) {
                consider(&mut best, -c.s_to, occ, limit);
            }
        }
    }
    best
}

/// The vehicles ahead by their bodies where they really are (round 4b): every vehicle whose box
/// comes into my corridor (my half width and `LEAD_MARGIN` either side of the way my body will
/// take) ahead of my front, the gap measured along my way to its near edge. Lane registration
/// is only the index the candidates are found by: the occupants of the lanes of my way within
/// `look` (and of the lane I change onto) are leads of the kind `Seen::Vehicle`, the others
/// near me (within `body_look`) `Seen::Body` (across the way: crossing a junction, pulling
/// out, passing). A car changing lanes is where its body is, not on its target lane at its
/// middle from the first frame (viz3 case01 #4519 stood behind #4436 3.5 m to its side, a
/// frozen change). Not my lead: my follower that has me as its lead, a lead I ignore for a
/// while (`ignore_lead`), the standing thing I am passing, and, of two bodies blocking each
/// other, the one with the lower id's for the higher. (nearest of the lane kind, nearest body)
fn lead_by_bodies(net: &Network, sim: &Simulation, vehicles: &[Vehicle], i: usize, lanes: &[(usize, f32)], way: &WayPts, look: f32, body_look: f32) -> (Option<Headway>, Option<Headway>) {
    let me = &vehicles[i];
    let st = &me.state;
    // (once it pulls out: while it signals the thing is its lead)
    let passing = me.passing.filter(|p| !p.aborted && p.wait <= 0.0);
    // (candidates: index, found on my way's lanes)
    let mut cand: Vec<(usize, bool)> = Vec::new();
    for &(l, d0) in lanes {
        if d0 > look {
            break;
        }
        for occ in sim.occupancy.get(&l).map(|v| v.as_slice()).unwrap_or(&[]) {
            cand.push((occ.vehicle, true));
        }
    }
    if let Some(c) = st.change {
        if !lanes.iter().any(|w| w.0 == c.to) {
            for occ in sim.occupancy.get(&c.to).map(|v| v.as_slice()).unwrap_or(&[]) {
                cand.push((occ.vehicle, true));
            }
        }
    }
    for j in sim.vehicles_near(me.pos, body_look as f64 + 12.0) {
        cand.push((j, false));
    }
    // (once each, on-way first)
    cand.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
    cand.dedup_by_key(|c| c.0);
    let w = me.half_width as f64 + LEAD_MARGIN;
    let h = (me.heading as f64).to_radians();
    let my_fwd = DVec2::new(h.sin(), h.cos());
    let (mut on_lane, mut body): (Option<Headway>, Option<Headway>) = (None, None);
    for (j, on_way) in cand {
        if j == i {
            continue;
        }
        let o = &vehicles[j];
        let rel = (o.pos - me.pos).truncate();
        let reach = if on_way { look } else { body_look };
        if rel.length() > reach as f64 + o.length() as f64 + 2.0 {
            continue;
        }
        // (two bodies blocking each other: the higher id goes)
        if !on_way && o.last.body_block == Some(me.id) && me.id > o.id {
            continue;
        }
        // (one changing into a lane of my way: also where its body is going, at that lane's
        // middle - it signals, I let it in ahead of me, as one in that lane; the lane-based
        // lead had it there from its first frame. Not one beside me: it waits for its gap)
        let into_mine = o.state.change.filter(|c| lanes.iter().any(|w| w.0 == c.to)).and_then(|_| o.state.change_anchors(net)).map(|(_, (p, h))| Obb::vehicle(p.truncate(), h as f64, o.state.front as f64, o.state.rear as f64, o.half_width as f64));
        let hit = way.corridor_hit(&o.obb(), w, reach).or_else(|| into_mine.and_then(|b| way.corridor_hit(&b, w, reach).filter(|h| h.0 - st.front >= 0.0)));
        let Some((near, k, f)) = hit else { continue };
        // (not one on another level: a bridge over the way, the road under it)
        if !on_way && (o.pos.z - way.zs[k]).abs() > 3.0 {
            continue;
        }
        let gap = near - st.front;
        if gap < -0.5 || gap > reach {
            continue;
        }
        // the one behind that follows me is not my lead
        if o.lead_id == Some(me.id) && rel.dot(my_fwd) < 0.0 {
            continue;
        }
        if let Some((id, until)) = me.ignore_lead {
            if id == o.id && sim.time < until {
                continue;
            }
        }
        // the standing thing I am passing
        if let Some(p) = passing {
            if o.state.speed < 0.3 && st.odometer + gap + st.front < p.until + 1.0 {
                continue;
            }
        }
        let oh = (o.heading as f64).to_radians();
        let o_fwd = DVec2::new(oh.sin(), oh.cos());
        let along = o_fwd.dot(f) as f32;
        let speed = o.state.speed * along;
        let acc = if along > 0.7 { o.state.acc * along } else { 0.0 };
        let (q, right) = (way.pts[k].1, DVec2::new(f.y, -f.x));
        let lateral = (o.pos.truncate() - q).dot(right) as f32 + st.lateral_ahead(way.pts[k].0);
        let standing_for = if o.external.at_stop { o.external.standing_for.max(o.stopped) } else { o.stopped };
        let held = o.last.held || o.last.stop.is_some();
        let hw = Headway { gap, speed, acc, what: if on_way { Seen::Vehicle(j) } else { Seen::Body(j) }, standing_for, half_width: o.half_width, length: o.length(), held, at_stop: o.external.at_stop, lateral };
        if on_way {
            on_lane = Headway::nearer(on_lane, Some(hw));
        } else {
            body = Headway::nearer(body, Some(hw));
        }
    }
    // changing lanes myself: the cars in the lane I move to, ahead of where I will be in it,
    // whose bodies lie in that lane (the corridor bends over only as I drive: a slower car just
    // ahead there, not yet in it, was cut in on, viz2 case07 #10906)
    if let Some(c) = st.change {
        for occ in sim.occupancy.get(&c.to).map(|v| v.as_slice()).unwrap_or(&[]) {
            if occ.vehicle == i {
                continue;
            }
            let o = &vehicles[occ.vehicle];
            let gap = occ.s - c.s_to - o.state.rear - st.front;
            if gap < -0.5 || gap > look || occ.lateral.abs() > me.half_width + o.half_width + LEAD_MARGIN as f32 {
                continue;
            }
            if o.lead_id == Some(me.id) || me.ignore_lead.is_some_and(|(id, until)| id == o.id && sim.time < until) {
                continue;
            }
            let standing_for = if o.external.at_stop { o.external.standing_for.max(o.stopped) } else { o.stopped };
            let hw = Headway { gap, speed: if occ.foreign { -o.state.speed } else { o.state.speed }, acc: o.state.acc, what: Seen::Vehicle(occ.vehicle), standing_for, half_width: o.half_width, length: o.length(), held: o.last.held || o.last.stop.is_some(), at_stop: o.external.at_stop, lateral: occ.lateral };
            on_lane = Headway::nearer(on_lane, Some(hw));
        }
    }
    (on_lane, body)
}

/// A vehicle on another lane into the same next lane that gets to the joint first.
fn merging_ahead(net: &Network, sim: &Simulation, vehicles: &[Vehicle], i: usize) -> Option<Headway> {
    let me = &vehicles[i];
    let st = &me.state;
    let next = st.planned_next?;
    let my_lane = net.lanes.get(st.lane)?;
    let d_me = my_lane.length() - st.s;
    if d_me > look_ahead(st.speed) {
        return None;
    }
    let limit = my_lane.speed_limit_kmh / 3.6;
    let t_me = arrival_time(d_me, st.speed, st.accel, limit);
    let mut best: Option<Headway> = None;
    for &p in net.prev.get(next)? {
        if p == st.lane {
            continue;
        }
        let pl = &net.lanes[p];
        for occ in sim.occupancy.get(&p).map(|v| v.as_slice()).unwrap_or(&[]) {
            if occ.vehicle == i || occ.foreign {
                continue;
            }
            let o = &vehicles[occ.vehicle];
            // into the same next lane; or out of a lane ending where mine does, whichever
            // lane it goes on into (two lanes converging on one joint whose starts lie too
            // close for the network to call them a merge, and for the junction to know them
            // as foes, met side by side at the joint)
            let same_joint = (pl.end() - my_lane.end()).truncate().length() < 1.5;
            if o.state.planned_next != Some(next) && !same_joint {
                continue;
            }
            // someone held at a light or a junction never arrives
            if o.state.speed < 0.5 && o.last.stop.is_some() {
                continue;
            }
            let d_them = pl.length() - occ.s;
            let t_them = arrival_time(d_them, o.state.speed, o.state.accel, limit);
            // (a near tie keeps whoever was first last frame: by the lower id)
            let first = t_them + 0.4 < t_me || ((t_them - t_me).abs() <= 0.4 && o.id < me.id);
            if !first {
                continue;
            }
            let gap = (d_me - d_them - o.state.rear - st.front).max(0.3);
            let h = Headway { gap, speed: o.state.speed, acc: o.state.acc, what: Seen::Vehicle(occ.vehicle), standing_for: o.stopped, half_width: o.half_width, length: o.length(), held: true, at_stop: false, lateral: 0.0 };
            best = Headway::nearer(best, Some(h));
        }
    }
    best
}

/// A player's vehicle on the way.
fn player_on_way(sim: &Simulation, me: &Vehicle, way: &WayPts, look: f32) -> Option<Headway> {
    let st = &me.state;
    let mut best: Option<Headway> = None;
    for p in &sim.players {
        if (p.pos - me.pos).truncate().length() > 180.0 || (p.pos.z - me.pos.z).abs() > 4.0 {
            continue;
        }
        // The box as it is; a vehicle with the right of way (`TrafficPriority`) crossing
        // the way also claims the five seconds of road ahead of it. (For any other player
        // the junction grants settle who goes first: a box stretched a second and a half
        // ahead swept over the car's way as the bus went by and drew an emergency stop
        // from a car that was rightly waiting behind.)
        let same_way = omsi_sim::traffic::wrap_deg(p.heading - me.heading).abs() < 60.0;
        let ahead = if p.priority && !same_way && p.speed.abs() > 0.5 { (p.speed.abs() * 5.0) as f64 } else { 0.0 };
        // (its body where it is, in my corridor as any vehicle's: `lead_by_bodies`; the box
        // grown by half a metre all round and my half width, as before, kept a follower 1.4 m
        // further back than the gap it says, viz3 case13)
        let b = p.obb(ahead, 0.0);
        if let Some((d, k, fwd)) = way.corridor_hit(&b, me.half_width as f64 + LEAD_MARGIN, look) {
            let gap = d - st.front;
            if gap < -0.5 {
                continue;
            }
            let speed = (p.fwd() * p.speed as f64).dot(fwd) as f32;
            let right = DVec2::new(fwd.y, -fwd.x);
            let q = way.pts[k].1;
            let lateral = (p.pos.truncate() - q).dot(right) as f32 + st.lateral_ahead(way.pts[k].0);
            let hw = Headway { gap, speed, acc: 0.0, what: Seen::Player(p.id), standing_for: p.still, half_width: p.half_width, length: p.half_len * 2.0, held: false, at_stop: false, lateral };
            best = Headway::nearer(best, Some(hw));
        }
    }
    best
}

/// Parked cars on the way: a blocking one as a headway, a protruding one as a swerve.
fn parked_on_way(sim: &Simulation, me: &Vehicle, way: &[(usize, f32)]) -> (Option<Headway>, Option<f32>) {
    let st = &me.state;
    let mut lead: Option<Headway> = None;
    let mut swerve: Option<f32> = None;
    for &(l, d0) in way {
        if d0 > 100.0 {
            break;
        }
        for p in sim.parked.get(&l).map(|v| v.as_slice()).unwrap_or(&[]) {
            let along = d0 + p.s;
            if !(-6.0..=100.0).contains(&along) {
                continue;
            }
            let my_lat = st.lateral_ahead(along.max(0.0));
            let lat = p.lateral - my_lat;
            let clearance = me.half_width + p.half_width + 0.35;
            if lat.abs() >= clearance {
                continue;
            }
            let need = (clearance - lat.abs()) * -lat.signum();
            if need.abs() <= 1.1 && along < 30.0 && me.passing.is_none() {
                if swerve.map(|s| need.abs() > s.abs()).unwrap_or(true) {
                    swerve = Some(need);
                }
            } else if need.abs() > 1.1 {
                let gap = along - p.half_len - st.front - 2.0;
                let h = Headway { gap, speed: 0.0, acc: 0.0, what: Seen::Parked, standing_for: f32::INFINITY, half_width: p.half_width, length: p.half_len * 2.0, held: false, at_stop: false, lateral: p.lateral };
                lead = Headway::nearer(lead, Some(h));
            }
        }
    }
    (lead, swerve)
}

/// Someone on foot in the strip ahead.
fn people_ahead(sim: &Simulation, me: &Vehicle, way: &WayPts) -> Option<f32> {
    let st = &me.state;
    let v = st.speed;
    let reach = (v * v / 5.0 + v + 6.0).clamp(8.0, 45.0);
    let radius = (st.front + reach + 6.0) as f64;
    let mut best: Option<f32> = None;
    for k in sim.people_near(me.pos, radius) {
        let p = &sim.people[k];
        let rel = p.pos - me.pos;
        if rel.truncate().length() > radius || rel.z.abs() > 2.5 {
            continue;
        }
        let half = if p.on_crossing {
            me.half_width + 1.5
        } else if p.waiting_for_bus && me.is_bus() {
            me.half_width - 0.3
        } else {
            me.half_width + 0.3
        };
        let t = ((rel.truncate().length() as f32) / v.max(1.0)).min(2.0);
        let later = p.pos.truncate() + p.vel * t as f64;
        for &(d, q, fwd) in &way.pts {
            if d > reach + st.front {
                break;
            }
            let right = DVec2::new(fwd.y, -fwd.x);
            for pos in [p.pos.truncate(), later] {
                let r = pos - q;
                let along = r.dot(fwd);
                let across = r.dot(right).abs();
                if along.abs() <= WAY_STEP as f64 * 0.5 + 0.05 && (across as f32) < half {
                    // [ROLLBACK r4c-L4] somebody waiting at the kerb to cross, outside my own
                    // footprint and its 0.3 m (to the 0.02 m beyond their body radius: the kerb spot is half a lane and
                    // 0.3 m off the lane's axis, and a bus is as wide as a lane), holds nobody:
                    // with a stream on the crossing they let the vehicle go first (P4)
                    if p.at_kerb && !p.on_crossing && (across as f32) >= me.half_width + 0.28 {
                        continue;
                    }
                    // someone who has stood at the edge for a while does not hold a standing car
                    if !p.on_crossing && v < 0.3 && p.still > 3.0 && (across as f32) > half - 0.9 {
                        continue;
                    }
                    // (where the person really is relative to the sample, so that the stop
                    // point does not jump with the sampling)
                    let stop = d + along as f32 - 1.5;
                    if best.map(|b| stop < b).unwrap_or(true) {
                        best = Some(stop);
                    }
                }
            }
        }
    }
    best
}

/// The next light on the way within 150 m. A second light of the same junction object on a
/// straight run holds nobody: the car has passed its line.
fn signal_ahead(net: &Network, sim: &Simulation, way: &[(usize, f32)], rear: f32) -> Option<SignalView> {
    let mut seen_object: Option<(((i32, i32), i64), f32)> = None;
    for &(l, d0) in way {
        if d0 > 150.0 {
            break;
        }
        let lane = &net.lanes[l];
        let Some((ctl, li)) = lane.traffic_light else { continue };
        let key = lane.key.map(|k| (k.tile, k.id));
        if let (Some(k), Some((sk, sh))) = (key, seen_object) {
            if k == sk && omsi_sim::traffic::wrap_deg(lane.start_heading() - sh).abs() < 45.0 {
                continue;
            }
        }
        if let Some(k) = key {
            if seen_object.is_none() {
                seen_object = Some((k, lane.start_heading()));
            }
        }
        if d0 < -(rear + 0.5) {
            // the whole car is past the line (one standing half over it waits like one
            // behind it: put down there, it drove through the junction on red)
            continue;
        }
        let Some(c) = sim.lights.get(ctl) else { continue };
        return Some(SignalView { dist: d0, ctl, light: li, aspect: TrafficLightController::aspect(c.state(li)), until_go: c.time_until_go(li) });
    }
    None
}

/// The next junction on the way (or the one the car is in), with the room past its
/// conflicts measured along the car's own way: a vehicle standing inside the piece past the
/// meeting place, or on the lanes after it within 40 m, leaves no room ("keep clear").
fn junction_ahead(net: &Network, sim: &Simulation, vehicles: &[Vehicle], i: usize, way: &[(usize, f32)]) -> Option<JunctionView> {
    let me = &vehicles[i];
    let js: &Junctions = &sim.junctions;
    for (k, &(l, d0)) in way.iter().enumerate() {
        if d0 > 120.0 {
            break;
        }
        let Some(&(j, m)) = js.of_lane.get(&l) else { continue };
        let inside = d0 <= 0.0;
        // the pieces my way takes through the object, and where their meeting places
        // begin and end in my way's distances (they may lie on the pieces after this one)
        let chain = js.chain_along(j, &way[k..]);
        let foes = js.foes_along(j, &chain);
        let way_len: f32 = chain.iter().map(|&(mm, _)| net.lanes[js.nodes[j].movements[mm].lane].length()).sum();
        let (first, last) = foes.iter().fold((f32::INFINITY, 0.0f32), |(s, e), f| (s.min(f.mine.0), e.max(f.mine.1)));
        let conflict_start = d0 + if first.is_finite() { first } else { way_len };
        let clear_from = d0 + if last > 0.0 { last } else { way_len };
        let need = me.length() + me.state.min_gap;
        let mut room = f32::INFINITY;
        for &(wl, wd) in &way[k..] {
            if wd > clear_from + 40.0 {
                break;
            }
            for occ in sim.occupancy.get(&wl).map(|v| v.as_slice()).unwrap_or(&[]) {
                if occ.vehicle == i || occ.foreign {
                    continue;
                }
                let o = &vehicles[occ.vehicle];
                let tail = wd + occ.s - o.state.rear;
                if o.state.speed < 1.5 && tail > clear_from - 0.5 {
                    room = room.min(tail - clear_from);
                }
            }
            for &(k, s, _) in sim.player_occupancy.get(&wl).map(|v| v.as_slice()).unwrap_or(&[]) {
                let p = &sim.players[k];
                let tail = wd + s - p.half_len;
                if p.speed.abs() < 1.5 && tail > clear_from - 0.5 {
                    room = room.min(tail - clear_from);
                }
            }
        }
        // the light at this junction's line: on the way before the first meeting place with
        // another way, wherever the object begins (BeijingCity lights a 1 m stub inside the
        // crossing object, past its lead-in piece). (The lead-in and its own stub end 1 m
        // apart and compile as a merge of each other: a meeting place of the way with itself,
        // before the light, which is no meeting with anybody else's way; nor is one with a
        // piece forking from where one of my own pieces starts: an unlit 0.3 m stub beside the
        // lit one at the end of the lead-in.)
        let light_line = {
            let node = &js.nodes[j];
            let starts: Vec<DVec3> = chain.iter().filter_map(|&(mm, _)| net.lanes.get(node.movements[mm].lane)).filter(|l| !l.points.is_empty()).map(|l| l.start()).collect();
            let fork = |mm: usize| net.lanes.get(node.movements[mm].lane).is_some_and(|l| !l.points.is_empty() && starts.iter().any(|p| (l.start() - *p).truncate().length() <= 0.5));
            // (nor where it runs side by side with another lane before the light, `Foe::side`:
            // the queue at the light stands there on both, round 4b)
            let first = foes.iter().filter(|f| !chain.iter().any(|&(mm, _)| mm == f.movement) && !fork(f.movement) && !f.side).map(|f| f.mine.0).fold(f32::INFINITY, f32::min);
            let any = foes.iter().filter(|f| !f.side).map(|f| f.mine.0).fold(f32::INFINITY, f32::min);
            if first.is_finite() {
                d0 + first
            } else if any.is_finite() {
                d0 + any
            } else {
                d0 + way_len
            }
        };
        let mut light = None;
        for &(wl, wd) in &way[k.saturating_sub(1)..] {
            if wd > light_line + 1.0 {
                break;
            }
            if let Some((ctl, li)) = net.lanes[wl].traffic_light {
                light = Some((ctl, li, wd));
                break;
            }
        }
        if light.is_none() && inside && k == 0 {
            light = light_behind(net, js, j, l, d0, me.state.prev_lane);
        }
        let past_line = light.is_some_and(|(_, _, d)| d + 0.5 < me.state.front);
        let turned = light.is_some_and(|(ctl, li, _)| sim.lights.get(ctl).is_some_and(|c| !matches!(TrafficLightController::aspect(c.state(li)), Aspect::Green | Aspect::GreenYellow | Aspect::Dark)));
        // (it passed the line while the light let it in: `Vehicle::passed_on_go`)
        // (and only once it cannot stop before its first meeting place, or is at it: a car
        // over its line on yellow 16 m short of it, that could still stop, was committed and
        // went in front of a car it had been refused for; review9)
        let to_line = light_line - me.state.front - 0.6;
        let cannot_stop = me.state.speed > 1.0 && to_line < me.state.speed * me.state.speed / (2.0 * omsi_sim::traffic::MAX_BRAKE * 0.7);
        let clearing = past_line && turned && light.map(|(c, l, _)| (c, l)) == me.passed_on_go && (cannot_stop || light_line - me.state.front <= 0.5);
        return Some(JunctionView { junction: j, movement: m, entry: d0, inside, chain, conflict_start, exit_room: room >= need, light_line, light, past_line, clearing, way_len });
    }
    None
}

/// A pedestrian at a crossing of the junction ahead.
fn zebra_ahead(net: &Network, sim: &Simulation, st: &AiState, way: &[(usize, f32)]) -> Option<f32> {
    let mut best: Option<f32> = None;
    for &(l, d0) in way {
        if d0 > 40.0 {
            break;
        }
        for &(w, sa, sw) in net.walks.get(l).map(|v| v.as_slice()).unwrap_or(&[]) {
            let near = sim.walkers.iter().any(|&(wl, ws)| wl == w && (ws - sw).abs() < 3.0);
            if !near {
                continue;
            }
            let point = d0 + sa;
            if point - st.front < 30.0 && point - st.front > -1.0 {
                let stop = point - 2.5;
                if best.map(|b| stop < b).unwrap_or(true) {
                    best = Some(stop);
                }
            }
        }
    }
    best
}

fn neighbour(net: &Network, sim: &Simulation, vehicles: &[Vehicle], i: usize, nb: usize, dir: i32) -> Option<NeighbourView> {
    let me = &vehicles[i];
    let st = &me.state;
    let lane = net.lanes.get(nb)?;
    let allowed = crate::access::is_open(lane, me);
    let s_b = net.beside_s(st.lane, nb, st.s);
    let mut ahead: Option<(f32, f32)> = None;
    let mut behind: Option<(f32, f32)> = None;
    let mut foreign = false;
    // (d_rel: the other's origin relative to the place beside me; front/rear its extents)
    let mut note = |d_rel: f32, front: f32, rear: f32, speed: f32, occ_foreign: bool| {
        if occ_foreign {
            foreign = true;
        }
        if d_rel >= 0.0 {
            let gap = d_rel - rear - st.front;
            if ahead.map(|a| gap < a.0).unwrap_or(true) {
                ahead = Some((gap, speed));
            }
        } else {
            let gap = -d_rel - front - st.rear;
            if behind.map(|b| gap < b.0).unwrap_or(true) {
                behind = Some((gap, speed));
            }
        }
    };
    let lane_users = |l: usize, offset: f32, note: &mut dyn FnMut(f32, f32, f32, f32, bool)| {
        for occ in sim.occupancy.get(&l).map(|v| v.as_slice()).unwrap_or(&[]) {
            if occ.vehicle == i {
                continue;
            }
            let o = &vehicles[occ.vehicle];
            note(occ.s + offset - s_b, o.state.front, o.state.rear, o.state.speed, occ.foreign);
        }
        // the players' vehicles are there like anybody else
        for &(k, s, _) in sim.player_occupancy.get(&l).map(|v| v.as_slice()).unwrap_or(&[]) {
            let p = &sim.players[k];
            note(s + offset - s_b, p.half_len, p.half_len, p.speed, false);
        }
    };
    lane_users(nb, 0.0, &mut note);
    // behind, on the lanes leading in (within 80 m)
    for &p in net.prev.get(nb).map(|v| v.as_slice()).unwrap_or(&[]) {
        lane_users(p, -net.lanes[p].length(), &mut note);
    }
    // ahead, on the first lane after
    if let Some(&n) = lane.next.first() {
        lane_users(n, lane.length(), &mut note);
    }
    let runs_on = net.reach.get(nb).copied().unwrap_or(lane.length()) - s_b;
    // a turn the lane leads into within 150 m
    let mut turn = lane.turn;
    let mut d = lane.length() - s_b;
    let mut cur = nb;
    while turn == 0 && d < 150.0 {
        let Some(&n) = net.lanes[cur].next.first() else { break };
        turn = net.lanes[n].turn;
        d += net.lanes[n].length();
        cur = n;
    }
    Some(NeighbourView { lane: nb, dir, ahead, behind, runs_on, turn, foreign, allowed })
}

/// The oncoming lane and how long it stays clear (only looked for behind a standing thing).
///
/// The oncoming lane runs the other way: along it, travel goes from small distances to
/// large ones, `os` is level with me, so an oncoming vehicle still to come is at a distance
/// *below* `os` (physically ahead of me) and one past me above it. The stretch a pass uses
/// lies ahead of me: oncoming distances `os - pass_len .. os`.
fn oncoming(net: &Network, sim: &Simulation, vehicles: &[Vehicle], i: usize, pass_len: f32) -> Option<OncomingView> {
    let me = &vehicles[i];
    let st = &me.state;
    let (opp, os, side) = net.opposite(st.lane, st.s)?;
    let ol = &net.lanes[opp];
    // a lane the car could not drive on anyway (a footpath, a bus lane closed to its
    // group), or one beyond a strip wider than two lanes' edges ever are, is no lane to
    // pass on
    let my_lane = &net.lanes[st.lane];
    let edge_gap = side.abs() - (my_lane.width.max(2.0) + ol.width.max(2.0)) * 0.5;
    if crate::access::open_to(ol, me).is_err() || edge_gap > omsi_sim::traffic::NEIGHBOUR_EDGE_GAP {
        return None;
    }
    let limit = ol.speed_limit_kmh / 3.6;
    let look = (limit.clamp(8.0, 20.0) * 12.0 + 20.0).min(300.0);
    let mut clear_for = f32::INFINITY;
    let mut blocked = false;
    let behind_me = os + st.rear + 1.0;
    // (pos, speed, accel, held at a red light)
    let mut note = |pos: f32, speed: f32, accel: f32, at_red: bool| {
        // past me already
        if pos > behind_me {
            return;
        }
        // standing in or just beyond the stretch the pass uses: in the way, not coming
        if speed < 0.5 && pos > os - pass_len - 15.0 {
            blocked = true;
            clear_for = 0.0;
            return;
        }
        // not yet level: the distance it still has to me
        let dist = (os - pos).max(0.0);
        let mut t = arrival_time(dist, speed, accel, limit);
        // a car standing at a red light with time to go: counts from when it goes
        if at_red && speed < 0.5 {
            t += 5.0;
        }
        // inside the stretch I will use: no time at all
        if pos > os - pass_len {
            t = 0.0;
        }
        clear_for = clear_for.min(t);
    };
    for (l, off, _) in net.upstream(opp, os, look, 48) {
        for occ in sim.occupancy.get(&l).map(|v| v.as_slice()).unwrap_or(&[]) {
            if occ.vehicle == i {
                continue;
            }
            let o = &vehicles[occ.vehicle];
            note(occ.s + off, o.state.speed, o.state.accel, o.last.why == crate::plan::Why::Light);
        }
        for &(k, s, _) in sim.player_occupancy.get(&l).map(|v| v.as_slice()).unwrap_or(&[]) {
            let p = &sim.players[k];
            note(s + off, p.speed.max(0.0), 1.2, false);
        }
    }
    if sim.parked.get(&opp).map(|v| v.iter().any(|p| p.lateral.abs() <= 1.6 && p.s > os - pass_len - 10.0 && p.s < os + 10.0)).unwrap_or(false) {
        blocked = true;
    }
    let side = side * sim.oncoming_sign();
    Some(OncomingView { lane: opp, s: os, side, clear_for, blocked })
}

/// A player's vehicle ahead on my side of the road, indicating to pull out.
fn letting_out(sim: &Simulation, me: &Vehicle) -> Option<f32> {
    let st = &me.state;
    let out_side = sim.out_side() as u8;
    let hr = (me.heading as f64).to_radians();
    let fwd = DVec2::new(hr.sin(), hr.cos());
    let right = DVec2::new(fwd.y, -fwd.x);
    let mut best: Option<f32> = None;
    for p in &sim.players {
        if p.blinker != out_side || p.signalling > 20.0 || p.speed.abs() > 3.0 {
            continue;
        }
        let rel = (p.pos - me.pos).truncate();
        let along = rel.dot(fwd) as f32;
        let across = rel.dot(right).abs() as f32;
        if along < 0.0 || along > 120.0 || p.fwd().dot(fwd) < 0.8 {
            continue;
        }
        if across > me.half_width + p.half_width + 2.5 {
            continue;
        }
        let d = along - p.half_len - st.front;
        if d > 0.5 && d >= st.speed * st.speed / (2.0 * st.decel.max(0.5)) {
            if best.map(|b| d < b).unwrap_or(true) {
                best = Some(d);
            }
        }
    }
    best
}

/// Everything vehicle `i` sees this frame.
pub fn perceive(net: &Network, sim: &Simulation, vehicles: &[Vehicle], i: usize) -> Perception {
    let me = &vehicles[i];
    let st = &me.state;
    let mut p = Perception::default();
    let Some(lane) = net.lanes.get(st.lane) else { return p };
    p.v_want = (lane.speed_limit_kmh * st.desire).min(st.max_speed_kmh).max(3.0) / 3.6;
    let look = look_ahead(st.speed);
    p.way = way_lanes(net, st, 220.0);
    p.road_left = net.reach.get(st.lane).copied().unwrap_or(lane.length()) - st.s;
    let air = me.class == crate::driver::VehicleClass::Air;

    p.merging = merging_ahead(net, sim, vehicles, i);
    if air {
        p.leader = Headway::nearer(leader_on_way(sim, vehicles, i, &p.way, look), p.merging);
    } else {
        // the way ahead, sampled once for the bodies, the players and the people
        let body_look = (st.speed * st.speed / (2.0 * st.decel.max(0.5)) + 2.0 * st.speed + 12.0).clamp(12.0, 70.0);
        let people_reach = (st.speed * st.speed / 5.0 + st.speed + 6.0).clamp(8.0, 45.0) + st.front;
        let player_near = sim.players.iter().any(|pl| (pl.pos - me.pos).truncate().length() < 180.0);
        let upto = body_look.max(people_reach).max(look);
        let way = WayPts::sample(net, st, upto);
        let (on_lane, body) = lead_by_bodies(net, sim, vehicles, i, &p.way, &way, look, body_look);
        p.leader = Headway::nearer(on_lane, p.merging);
        if player_near {
            p.leader = Headway::nearer(p.leader, player_on_way(sim, me, &way, look));
        }
        if let Some(h) = body {
            // a body across the way replaces the lead only when clearly nearer
            if p.leader.map(|l| h.gap < l.gap - 0.5).unwrap_or(true) {
                if let Seen::Body(j) = h.what {
                    p.body_block = Some(vehicles[j].id);
                }
                p.leader = Some(h);
            }
        }
        let (parked, swerve) = parked_on_way(sim, me, &p.way);
        p.leader = Headway::nearer(p.leader, parked);
        p.kerb_swerve = swerve;
        p.people_stop = people_ahead(sim, me, &way);
        p.signal = signal_ahead(net, sim, &p.way, st.rear);
        p.junction = junction_ahead(net, sim, vehicles, i, &p.way);
        p.zebra_stop = zebra_ahead(net, sim, st, &p.way);
        p.letting_out = letting_out(sim, me);
        if lane.kind == LaneKind::Street && !me.is_rail() {
            if let Some(l) = lane.left {
                p.left = neighbour(net, sim, vehicles, i, l, 1);
            }
            if let Some(r) = lane.right {
                p.right = neighbour(net, sim, vehicles, i, r, 2);
            }
            // the oncoming lane matters behind something standing, or while out on it
            let standing = p.leader.map(|l| l.standing_obstacle() && l.gap < 45.0).unwrap_or(false);
            if (standing || me.passing.is_some()) && lane.left.is_none() && lane.right.is_none() {
                let obstacle = p.leader.map(|l| l.gap + l.length).unwrap_or(10.0);
                let pass_len = obstacle + st.front + st.rear + 6.0;
                p.oncoming = oncoming(net, sim, vehicles, i, pass_len);
            }
        }
    }
    p.route_change = st.route_change_due(net);
    p
}

#[allow(dead_code)]
fn _unused(_: DVec3) {}
