//! What a driver sees, as plain values: the vehicle ahead with its gap, the next signal,
//! the next junction, the lanes beside with their gaps, people in the way, the oncoming
//! lane and how long it stays clear. Read-only over the whole traffic; the plan reads
//! nothing else.
//!
//! The car's way ahead is sampled once per frame (`WayPts`) and every test against a
//! body, a player or a person reuses the samples; the vehicles and the people near the
//! car come from the simulation's grids, not from a scan of everything.

use crate::driver::STANDING_SPEED;
use crate::junction::{FoeKind, Junctions};
use crate::plan::QUEUE_OBSTACLE_STAND;
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
    /// [round5 C3] Where its rear comes to rest (m from my origin) when it brakes for a stop of
    /// its own (its plan's `stop`): a queue forming ahead of me, before it stands.
    pub rest: Option<f32>,
}

impl Headway {
    /// Where vehicle `o`'s rear comes to rest, `gap` ahead of my front (`front`): at the stop of
    /// its own it brakes for, or behind its own standing lead at its queue gap, whichever is
    /// nearer (x10 case15: the lead braked for its exit-full line but stood 8 m short of it
    /// behind a bus at its stop); `None` with neither.
    fn rest_of(o: &Vehicle, gap: f32, front: f32) -> Option<f32> {
        let by_stop = o.last.stop.map(|s| (s - o.state.front).max(0.0));
        let by_lead = o.last.lead.filter(|l| l.speed < STANDING_SPEED).map(|l| (l.gap - o.state.min_gap).max(0.0));
        let on = match (by_stop, by_lead) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        on.map(|d| gap + front + d)
    }

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

    /// Something that stands and will go on standing: worth going round. (Parked; a bus at
    /// its stop after `AT_STOP_OBSTACLE_STAND`; one held by nothing, or the player, after
    /// `QUEUE_OBSTACLE_STAND`; anything after `HELD_OBSTACLE_STAND`.)
    pub fn standing_obstacle(&self) -> bool {
        if self.speed >= STANDING_SPEED {
            return false;
        }
        match self.what {
            Seen::Parked => true,
            Seen::Player(_) => self.standing_for > QUEUE_OBSTACLE_STAND,
            Seen::Vehicle(_) | Seen::Body(_) => (self.at_stop && self.standing_for > AT_STOP_OBSTACLE_STAND) || (!self.held && self.standing_for > QUEUE_OBSTACLE_STAND) || self.standing_for > HELD_OBSTACLE_STAND,
        }
    }
}

/// A bus at its stop is an obstacle worth going round once it has stood this long there (s).
pub const AT_STOP_OBSTACLE_STAND: f32 = 0.5;
/// Anything standing this long (s) is an obstacle, whatever holds it (its hold is not
/// going to lift soon).
pub const HELD_OBSTACLE_STAND: f32 = 25.0;

/// How far ahead of its front (m) a driver at `v` m/s looks for people on its way: a stop
/// and some, 8 m at least, 45 m at most (`people_ahead`, `crossing_ahead`, `perceive`).
pub fn people_reach(v: f32) -> f32 {
    (v * v / 5.0 + v + 6.0).clamp(8.0, 45.0)
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
    /// Distance from the origin to the end of the last meeting place on the way through
    /// (the end of the way through when it has no foe).
    pub conflict_end: f32,
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
fn light_behind(net: &Network, sim: &Simulation, js: &Junctions, j: usize, lane: usize, back: f32, prev_lane: Option<usize>) -> Option<(usize, usize, f32)> {
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
            if let Some((ctl, li)) = stop_light(net, sim, p) {
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
    /// A route lane change due: (lane, dir).
    pub route_change: Option<(usize, i32)>,
    /// Metres the way runs on before it ends.
    pub road_left: f32,
    /// A parked car the way should swerve round, or [round 5 L3] people on a kerbless
    /// pavement strip beside the lane: the lateral offset needed.
    pub kerb_swerve: Option<f32>,
    /// [round 5 L3] `kerb_swerve` is (also) the shift past people beside the lane: the car
    /// eases past them.
    pub people_swerve: bool,
    pub oncoming: Option<OncomingView>,
    /// A player's vehicle ahead indicating to pull out towards the traffic: the distance
    /// from my origin to its rear.
    pub letting_out: Option<f32>,
    /// A vehicle on another lane that reaches the joint ahead first.
    pub merging: Option<Headway>,
    /// Every body in my corridor ahead (vehicles, players, parked cars): (gap from my front to
    /// its near edge, to its far edge, its speed along my way), nearest first. What lies
    /// beyond the lead (a pass needs room to come back in, `plan`).
    pub ahead: Vec<(f32, f32, f32)>,
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
    /// Bounding circles of runs of `WAY_CHUNK` slices: (centre, radius, first sample, one past
    /// the last slice), so that a box far from a run skips its slices at one test.
    chunks: Vec<(DVec2, f64, usize, usize)>,
}

/// Half the length of the longest vehicle (m), at most (an articulated bus is 18 m): how far
/// before or past its reference point a body on my way can reach.
const LONGEST_HALF: f32 = 15.0;

/// Slices of the way per bounding circle (`WayPts::chunks`).
const WAY_CHUNK: usize = 8;

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
        let mut ds = Vec::with_capacity(n + 1);
        ds.push(-WAY_STEP);
        let mut d = 0.0f32;
        for _ in 0..n {
            ds.push(d);
            d += if d < WAY_FINE { WAY_STEP } else { 2.0 * WAY_STEP };
        }
        let qs = st.way_points(net, ds.iter().copied());
        let mut prev = qs[0].truncate();
        for (&d, q) in ds.iter().zip(&qs).skip(1) {
            zs.push(q.z);
            let q2 = q.truncate();
            let fwd = (q2 - prev).normalize_or_zero();
            pts.push((d, q2, fwd));
            prev = q2;
        }
        let slices = pts.len().saturating_sub(1);
        let chunks = (0..slices).step_by(WAY_CHUNK).map(|a| {
            let e = (a + WAY_CHUNK).min(slices);
            // (the run's slices span samples a..=e)
            let c = pts[a..=e].iter().map(|p| p.1).sum::<DVec2>() / (e - a + 1) as f64;
            let r = pts[a..=e].iter().map(|p| (p.1 - c).length()).fold(0.0, f64::max);
            (c, r, a, e)
        }).collect();
        WayPts { pts, zs, chunks }
    }

    /// Whether a slice of the way starts within `r` of `c`.
    fn any_within(&self, c: DVec2, r: f64) -> bool {
        self.chunks.iter().any(|&(cc, rc, a, e)| (c - cc).length() <= rc + r && self.pts[a..e].iter().any(|p| (p.1 - c).length() <= r))
    }

    /// The sample (as the start of its slice) nearest to `c`, if one is within `within`.
    fn nearest(&self, c: DVec2, within: f64) -> Option<usize> {
        let mut best: Option<(f64, usize)> = None;
        for &(cc, r, a, e) in &self.chunks {
            let lower = (c - cc).length() - r;
            if lower > within || best.is_some_and(|b| lower > b.0) {
                continue;
            }
            for k in a..e {
                let d = (self.pts[k].1 - c).length();
                if d <= within && best.is_none_or(|b| d < b.0 || (d == b.0 && k < b.1)) {
                    best = Some((d, k));
                }
            }
        }
        best.map(|b| b.1)
    }

    /// Where box `b` comes into the corridor of half width `w` round the way (no further than
    /// `upto` along it): (distance from the origin along the way to its near edge, the index
    /// of the sample whose slice it enters, the slice's forward direction). Each slice of the
    /// way (between two samples) clips the box in its own frame, so a box beside the way,
    /// however near, is no hit, and one across it is met at its near edge.
    fn corridor_hit(&self, b: &Obb, w: f64, upto: f32) -> Option<(f32, usize, DVec2)> {
        self.corridor_span(b, w, upto).map(|(near, _, k, f)| (near, k, f))
    }

    /// `corridor_hit` with the far edge of the box along the way too (in the frame of the
    /// slice it enters): (near, far, sample, direction).
    fn corridor_span(&self, b: &Obb, w: f64, upto: f32) -> Option<(f32, f32, usize, DVec2)> {
        let corners = b.corners();
        let [ax_r, ax_f] = b.axes();
        let reach = b.half.length() + w;
        let slices = self.chunks.iter().filter(|&&(cc, r, _, _)| (b.center - cc).length() <= r + reach + 2.0 * WAY_STEP as f64 + 0.5).flat_map(|&(_, _, a, e)| a..e);
        for k in slices {
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
            // (the box's extent across and along the slice first: clear of the strip or of
            // the slice, it is not clipped)
            let lo = if k == 0 { f64::NEG_INFINITY } else { 0.0 };
            let v = b.center - q;
            let (cx, cy) = (v.dot(f), v.dot(r));
            let ex = b.half.x * ax_r.dot(f).abs() + b.half.y * ax_f.dot(f).abs();
            let ey = b.half.x * ax_r.dot(r).abs() + b.half.y * ax_f.dot(r).abs();
            if cy.abs() - ey > w + 1e-6 || cx + ex < lo - 1e-6 || cx - ex > step + 1e-6 {
                continue;
            }
            let poly: [DVec2; 4] = corners.map(|c| {
                let v = c - q;
                DVec2::new(v.dot(f), v.dot(r))
            });
            let Some((x0, x1)) = clip_strip(&poly, w) else { continue };
            // (the first slice reaches back behind the origin: a body beside the car)
            if x1 < lo || x0 > step {
                continue;
            }
            return Some((d + x0 as f32, d + x1 as f32, k, f));
        }
        None
    }
}

/// The x extent (min, max) of the polygon `poly` cut to the strip |y| <= `w`, if anything of it
/// lies in the strip.
fn clip_strip(poly: &[DVec2; 4], w: f64) -> Option<(f64, f64)> {
    // (a quadrilateral cut by two lines has at most 6 corners; on the stack, it is cut for
    // every slice near every body each frame)
    fn clip(pts: &[DVec2], keep: impl Fn(DVec2) -> f64, out: &mut [DVec2; 8]) -> usize {
        let mut n = 0;
        for k in 0..pts.len() {
            let (a, b) = (pts[k], pts[(k + 1) % pts.len()]);
            let (fa, fb) = (keep(a), keep(b));
            if fa >= 0.0 {
                out[n] = a;
                n += 1;
            }
            if (fa >= 0.0) != (fb >= 0.0) {
                out[n] = a + (b - a) * (fa / (fa - fb));
                n += 1;
            }
        }
        n
    }
    let (mut upper, mut both) = ([DVec2::ZERO; 8], [DVec2::ZERO; 8]);
    let nu = clip(poly, |p| w - p.y, &mut upper);
    if nu == 0 {
        return None;
    }
    let nb = clip(&upper[..nu], |p| p.y + w, &mut both);
    if nb == 0 {
        return None;
    }
    Some(both[..nb].iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), p| (a.min(p.x), b.max(p.x))))
}

fn look_ahead(v: f32) -> f32 {
    (v * v / 3.0 + 2.0 * v + 20.0).clamp(70.0, 150.0)
}

/// The vehicle ahead on the way, through the lanes ahead (by lane registration: an aircraft's,
/// whose way has no body test; an aircraft never passes, `plan`'s `can_pass`).
fn leader_on_way(sim: &Simulation, vehicles: &[Vehicle], i: usize, way: &[(usize, f32)], look: f32) -> Option<Headway> {
    let me = &vehicles[i];
    let st = &me.state;
    let signalling = st.change.map(|c| c.wait > 0.0 && c.t <= 0.0 && !c.bypass).unwrap_or(false);
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
        let standing_for = if o.external.at_stop { o.external.standing_for.max(o.stopped) } else { o.stopped };
        let held = o.last.held || o.last.stop.is_some();
        let h = Headway { gap, speed: if occ.foreign { -o.state.speed } else { o.state.speed }, acc: o.state.acc, what: Seen::Vehicle(occ.vehicle), standing_for, half_width: o.half_width, length: o.length(), held, at_stop: o.external.at_stop, lateral: occ.lateral, rest: Headway::rest_of(o, gap, st.front) };
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

/// The vehicles `lead_by_bodies` looks at: (index, found on my way's lanes, how far along my
/// way its middle is by those lanes, NaN if not known), once each; with how far along my way
/// (m) the farthest one found there reaches, at most `look`: the way need be sampled no
/// further than that, `body_look` and the people's and the players' reach (review12).
fn body_candidates(sim: &Simulation, vehicles: &[Vehicle], i: usize, lanes: &[(usize, f32)], look: f32, body_look: f32) -> (Vec<(usize, bool, f32)>, f32) {
    let me = &vehicles[i];
    let st = &me.state;
    let mut far = 0.0f32;
    // (once each, the first found: on my way's lanes, nearest lane first, then the others near
    // me; a vehicle is in one grid cell, so those are once each already)
    let mut cand: Vec<(usize, bool, f32)> = Vec::with_capacity(64);
    for &(l, d0) in lanes {
        if d0 > look {
            break;
        }
        for occ in sim.occupancy.get(&l).map(|v| v.as_slice()).unwrap_or(&[]) {
            if cand.iter().any(|c| c.0 == occ.vehicle) {
                continue;
            }
            cand.push((occ.vehicle, true, d0 + occ.s));
            // (its front at most `LONGEST_HALF` on: no look at the vehicle itself)
            far = far.max(d0 + occ.s + LONGEST_HALF);
        }
    }
    if let Some(c) = st.change {
        if !lanes.iter().any(|w| w.0 == c.to) {
            for occ in sim.occupancy.get(&c.to).map(|v| v.as_slice()).unwrap_or(&[]) {
                if !cand.iter().any(|c| c.0 == occ.vehicle) {
                    cand.push((occ.vehicle, true, f32::NAN));
                }
                far = look;
            }
        }
    }
    let on_way = cand.len();
    for j in sim.vehicles_near(me.pos, body_look as f64 + 12.0) {
        if !cand[..on_way].iter().any(|c| c.0 == j) {
            cand.push((j, false, f32::NAN));
        }
    }
    // (a slice or two more: the box is met in the slice its near edge enters)
    (cand, (far + 2.0 * WAY_STEP).min(look))
}

/// The vehicles ahead by their bodies where they really are (round 4b): every vehicle whose box
/// comes into my corridor (my half width and `LEAD_MARGIN` either side of the way my body will
/// take) ahead of my front, the gap measured along my way to its near edge. Lane registration
/// is only the index the candidates are found by: the occupants of the lanes of my way within
/// `look` (and of the lane I change onto) are leads of the kind `Seen::Vehicle`, the others
/// near me (within `body_look`) `Seen::Body` (across the way: crossing a junction, pulling
/// out, passing). A car changing lanes is where its body is, not on its target lane at its
/// middle from the first frame (viz3 case01 #4519 stood behind #4436 3.5 m to its side, a
/// frozen change). Not my lead: my follower that has me as its lead, the standing thing I am
/// passing, and, of two bodies blocking each other, the one with the lower id's for the
/// higher. (nearest of the lane kind, nearest body)
fn lead_by_bodies(net: &Network, sim: &Simulation, vehicles: &[Vehicle], i: usize, lanes: &[(usize, f32)], way: &WayPts, look: f32, body_look: f32, cand: Vec<(usize, bool, f32)>, ahead: &mut Vec<(f32, f32, f32)>) -> (Option<Headway>, Option<Headway>) {
    let me = &vehicles[i];
    let st = &me.state;
    // (once it pulls out: while it signals the thing is its lead)
    let passing = me.passing.filter(|p| !p.aborted && p.wait <= 0.0);
    let w = me.half_width as f64 + LEAD_MARGIN;
    let h = (me.heading as f64).to_radians();
    let my_fwd = DVec2::new(h.sin(), h.cos());
    let (mut on_lane, mut body): (Option<Headway>, Option<Headway>) = (None, None);
    for (j, on_way, along) in cand {
        if j == i {
            continue;
        }
        // (on my way's lanes far behind me or beyond what I look at, whatever its length: no
        // lead, and no need to look at it)
        if on_way && (along + LONGEST_HALF < -(st.rear + 2.0) || along - LONGEST_HALF - st.front > look + 3.0) {
            continue;
        }
        let o = &vehicles[j];
        let rel = (o.pos - me.pos).truncate();
        let reach = if on_way { look } else { body_look };
        if rel.length() > reach as f64 + o.length() as f64 + 2.0 {
            continue;
        }
        // (on my way's lanes wholly behind me, or beyond what I look at: no lead)
        if on_way && (along + o.state.front < -(st.rear + 2.0) || along - o.state.rear - st.front > look + 3.0) {
            continue;
        }
        // (two bodies blocking each other: the higher id goes)
        // TODO(review11, pre-existing): of two bodies blocking each other head-on, the higher id
        // ignores the lower one's body and drives into it (-1.82 m on both the baseline and
        // round 4b); it should keep the body as an obstacle and leave the pair to the
        // negotiation of a mutual hold instead.
        if !on_way && o.last.body_block == Some(me.id) && me.id > o.id {
            continue;
        }
        // (one changing into a lane of my way: also where its body is going, at that lane's
        // middle - it signals, I let it in ahead of me, as one in that lane; the lane-based
        // lead had it there from its first frame. Not one beside me: it waits for its gap)
        let ob = o.obb();
        let into_mine = o.state.change.filter(|c| lanes.iter().any(|w| w.0 == c.to)).and_then(|_| o.state.change_anchors(net)).map(|(_, (p, h))| Obb::vehicle(p.truncate(), h as f64, o.state.front as f64, o.state.rear as f64, o.half_width as f64));
        // (nowhere near my way, however far it moves in, `cut_in_hit`, or steps back, `back_in`
        // below: one bounding circle test instead of the slices', `WayPts::corridor_span`)
        let moves_in = if o.state.speed >= 0.5 { (o.state.speed as f64 * CUT_IN_TIME as f64).min(3.0) } else { 0.0 };
        let steps_back = if o.passing.is_some_and(|p| p.aborted || o.stopped > 2.0) && o.state.lateral.abs() > 0.5 { o.state.lateral.abs() as f64 } else { 0.0 };
        if into_mine.is_none() && !way.any_within(ob.center, ob.half.length() + w + 2.0 * WAY_STEP as f64 + 0.5 + moves_in.max(steps_back)) {
            continue;
        }
        let oh = (o.heading as f64).to_radians();
        let o_fwd = DVec2::new(oh.sin(), oh.cos());
        let hit = way.corridor_span(&ob, w, reach).or_else(|| into_mine.and_then(|b| way.corridor_span(&b, w, reach).filter(|h| h.0 - st.front >= 0.0)));
        // (moving into my corridor: where it will be, if nearer, `cut_in_hit`)
        // (one keeping its lane - no change, no pass - comes no nearer sideways than its lane's
        // line lets it, `lane_room`: a car drifting half a metre in its own lane made the one
        // beside it brake hard, review11 re-check)
        let lane_room = (o.state.change.is_none() && o.passing.is_none()).then(|| {
            let lh = (net.lanes[o.state.lane].at(o.state.s).1 as f64).to_radians();
            (o.state.lateral_target - o.state.lateral, DVec2::new(lh.sin(), lh.cos()))
        });
        let hit = match (hit, cut_in_hit(way, &ob, o_fwd, o.state.speed, o.last.stop.map(|sp| sp - o.state.front), lane_room, w, reach)) {
            (Some(h), Some(c)) if c.0 < h.0 => Some((c.0, h.1, c.1, c.2)),
            (None, Some(c)) => Some((c.0, c.0 + o.length(), c.1, c.2)),
            (h, _) => h,
        };
        // (a pass standing out beside the lane, given up or held: also where its body steps back
        // to, so that I keep my gap behind it there and it can; review12, case15 #47012 closed up
        // to 0.23 m behind #47238's rear and it never could)
        let back_in = o.passing.filter(|p| p.aborted || o.stopped > 2.0).filter(|_| o.state.lateral.abs() > 0.5).map(|_| {
            let r = DVec2::new(o_fwd.y, -o_fwd.x);
            Obb { center: ob.center - r * o.state.lateral as f64, ..ob }
        });
        // (half a metre more than the gap kept: at least two metres behind its rear)
        let back_in = back_in.and_then(|b| way.corridor_span(&b, w, reach).filter(|h| h.0 - st.front >= -0.5)).map(|h| (h.0 - BACK_IN_MARGIN, h.1, h.2, h.3));
        let hit = match (hit, back_in) {
            (Some(h), Some(c)) if c.0 < h.0 => Some(c),
            (None, Some(c)) => Some(c),
            (h, _) => h,
        };
        let Some((near, far, k, f)) = hit else { continue };
        // (not one on another level: a bridge over the way, the road under it)
        if !on_way && (o.pos.z - way.zs[k]).abs() > 3.0 {
            continue;
        }
        let mut gap = near - st.front;
        let mut cut_in = false;
        if gap > reach {
            continue;
        }
        // beside me: a lead only when it is moving into my corridor (`cutting_in`), at gap 0
        if gap < -0.5 {
            if !cutting_in(o_fwd, o.state.speed, f, o.pos.truncate() - way.pts[k].1, lane_room, w) {
                continue;
            }
            gap = 0.0;
            cut_in = true;
        }
        // the one behind that follows me is not my lead
        if o.lead_id == Some(me.id) && rel.dot(my_fwd) < 0.0 {
            continue;
        }
        {
            let oh = (o.heading as f64).to_radians();
            ahead.push((gap, far - st.front, o.state.speed * DVec2::new(oh.sin(), oh.cos()).dot(f) as f32));
        }
        let (q, right) = (way.pts[k].1, DVec2::new(f.y, -f.x));
        let lateral = (o.pos.truncate() - q).dot(right) as f32 + st.lateral_ahead(way.pts[k].0);
        // the standing thing I am passing (in my lane, nearer its middle than the line of the
        // pass: one standing on the line the pass takes, an oncoming car, is in my way)
        if let Some(p) = passing {
            if o.state.speed < STANDING_SPEED && st.odometer + gap + st.front < p.until + 1.0 && lateral.abs() < (lateral - p.side).abs() {
                continue;
            }
        }
        let along = o_fwd.dot(f) as f32;
        // (cutting in beside me: held as standing there, so that I drop back behind it)
        let speed = if cut_in { 0.0 } else { o.state.speed * along };
        let acc = if along > 0.7 { o.state.acc * along } else { 0.0 };
        let standing_for = if o.external.at_stop { o.external.standing_for.max(o.stopped) } else { o.stopped };
        let held = o.last.held || o.last.stop.is_some();
        let hw = Headway { gap, speed, acc, what: if on_way { Seen::Vehicle(j) } else { Seen::Body(j) }, standing_for, half_width: o.half_width, length: o.length(), held, at_stop: o.external.at_stop, lateral, rest: Headway::rest_of(o, gap, st.front) };
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
            if o.lead_id == Some(me.id) {
                continue;
            }
            let standing_for = if o.external.at_stop { o.external.standing_for.max(o.stopped) } else { o.stopped };
            let hw = Headway { gap, speed: if occ.foreign { -o.state.speed } else { o.state.speed }, acc: o.state.acc, what: Seen::Vehicle(occ.vehicle), standing_for, half_width: o.half_width, length: o.length(), held: o.last.held || o.last.stop.is_some(), at_stop: o.external.at_stop, lateral: occ.lateral, rest: Headway::rest_of(o, gap, st.front) };
            on_lane = Headway::nearer(on_lane, Some(hw));
        }
    }
    (on_lane, body)
}

/// A vehicle on another lane into the same next lane that gets to the joint first: the merge
/// rule of a lane join the junctions know no meeting place of (`Junctions::foe_between`:
/// [round 7 A2] where they do - a compiled merge - its grant and the zipper decide, and this
/// first-come rule, which contradicted them, is not asked). Of the joins of the recorded
/// networks, X10's had 28 such pairs outside junctions and 231 with one lane in one
/// (`traffic_cases joins`).
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
        if p == st.lane || sim.junctions.foe_between(st.lane, p).is_some() {
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
            if o.state.speed < STANDING_SPEED && o.last.stop.is_some() {
                continue;
            }
            let d_them = pl.length() - occ.s;
            let t_them = arrival_time(d_them, o.state.speed, o.state.accel, limit);
            // (a near tie, within 0.4 s, goes to the lower id)
            let first = t_them + 0.4 < t_me || ((t_them - t_me).abs() <= 0.4 && o.id < me.id);
            if !first {
                continue;
            }
            let gap = (d_me - d_them - o.state.rear - st.front).max(0.3);
            let h = Headway { gap, speed: o.state.speed, acc: o.state.acc, what: Seen::Vehicle(occ.vehicle), standing_for: o.stopped, half_width: o.half_width, length: o.length(), held: true, at_stop: false, lateral: 0.0, rest: Headway::rest_of(o, gap, st.front) };
            best = Headway::nearer(best, Some(h));
        }
    }
    best
}

/// A body beside me (`rel`: from the way's sample to its middle, `f`: the way's direction
/// there) that overlaps my corridor and is moving into it (or is in it already, its middle
/// within my corridor's half width `w`): its middle ahead of mine, moving
/// (over 0.5 m/s) with its heading more than `CUT_IN_DEG` towards my way or its sideways speed
/// towards it over `CUT_IN_SPEED`. Such a body is a lead at gap 0 (I hold back and let it
/// in, braking as for one standing there until it is ahead of me); one merely beside me and
/// parallel is not, nor one keeping its lane whose line lets it come no nearer (`lane_room`:
/// how far its lane's line still takes it sideways, + = its right; see `lane_room_toward`).
fn cutting_in(o_fwd: DVec2, speed: f32, f: DVec2, rel: DVec2, lane_room: Option<(f32, DVec2)>, w: f64) -> bool {
    let right = DVec2::new(f.y, -f.x);
    let (along, lat) = (rel.dot(f), rel.dot(right));
    if along <= 0.0 {
        return false;
    }
    // (or in already: its middle inside my corridor, `w` either side of my way)
    if lat.abs() <= w {
        return true;
    }
    if speed < 0.5 {
        return false;
    }
    // (one keeping its lane: not unless its line still takes it towards me, or its lane itself
    // runs into my way)
    if let Some((room, lane_fwd)) = lane_room {
        let toward_lane = -(lane_fwd.dot(right)) * lat.signum();
        if lane_room_toward(room, lat) < 0.05 && toward_lane <= (CUT_IN_DEG as f64).to_radians().sin() {
            return false;
        }
    }
    let toward = -(o_fwd.dot(right)) * lat.signum();
    if o_fwd.dot(f) < CUT_IN_MAX_DEG.to_radians().cos() as f64 {
        return false;
    }
    toward > (CUT_IN_DEG as f64).to_radians().sin() || speed as f64 * toward > CUT_IN_SPEED as f64
}

/// A body moving sideways into my corridor (`cutting_in`'s heading or sideways speed, measured
/// at the sample of my way nearest its middle): where its box would come into my corridor
/// moved sideways by what it covers in `CUT_IN_TIME` (at most 3 m, and no further than `room`, the
/// way its front has to the stop it brakes for), as `WayPts::corridor_hit`.
/// It is my lead from there: seen only where it is, a bus cutting in at 6-15 degrees left the
/// car half a second to brake, and its near edge in my lane ran back towards the car faster
/// than its speed said (review11).
fn cut_in_hit(way: &WayPts, b: &Obb, o_fwd: DVec2, speed: f32, room: Option<f32>, lane_room: Option<(f32, DVec2)>, w: f64, upto: f32) -> Option<(f32, usize, DVec2)> {
    if speed < 0.5 || way.pts.len() < 2 {
        return None;
    }
    // (the sample nearest its middle: none near enough, the box moved 3 m meets no slice)
    let k = way.nearest(b.center, b.half.length() + w + 3.0 + 2.0 * WAY_STEP as f64 + 0.5)?;
    let f = (way.pts[k + 1].1 - way.pts[k].1).normalize_or_zero();
    let right = DVec2::new(f.y, -f.x);
    let lat = (b.center - way.pts[k].1).dot(right);
    if lat.abs() < 1e-3 {
        return None;
    }
    let toward = -(o_fwd.dot(right)) * lat.signum();
    // (my way, about: crossing traffic is the junctions' to sort out)
    if o_fwd.dot(f) < CUT_IN_MAX_DEG.to_radians().cos() as f64 {
        return None;
    }
    if !(toward > (CUT_IN_DEG as f64).to_radians().sin() || speed as f64 * toward > CUT_IN_SPEED as f64) {
        return None;
    }
    // (no further than the stop it is braking for lets it: a car creeping up to the line it
    // waits at comes no nearer)
    let shift = (speed as f64 * toward * CUT_IN_TIME as f64).min(room.map(|r| r.max(0.0) as f64 * toward).unwrap_or(f64::INFINITY)).clamp(0.0, 3.0);
    // (one keeping its lane: its lane's own run towards my way in full, `toward_lane`, and
    // the rest - its drift within the lane - no further than its line takes it)
    let shift = match lane_room {
        Some((room, lane_fwd)) => {
            let toward_lane = (-(lane_fwd.dot(right)) * lat.signum()).max(0.0);
            let by_lane = speed as f64 * toward_lane * CUT_IN_TIME as f64;
            let drift = (speed as f64 * (toward - toward_lane) * CUT_IN_TIME as f64).max(0.0).min(lane_room_toward(room, lat));
            shift.min(by_lane + drift)
        }
        None => shift,
    };
    if shift < 0.05 {
        return None;
    }
    let moved = Obb { center: b.center - right * (lat.signum() * shift), ..*b };
    way.corridor_hit(&moved, w, upto)
}

/// How much further a body keeping its lane can drift towards my way within it (m): its way's
/// sideways offset still to go, `room` (+ = its right, `AiState::lateral_target - lateral`),
/// taken towards my way, which lies on its left when its offset from my way `lat` is positive
/// (the two run within 45 degrees of each other, so its right is mine, about).
fn lane_room_toward(room: f32, lat: f64) -> f64 {
    (room as f64 * -lat.signum()).max(0.0)
}

/// Metres more than its gap a car keeps behind where a pass standing out beside the lane steps
/// back in to.
pub const BACK_IN_MARGIN: f32 = 0.5;

/// `cut_in_hit`: the seconds of sideways movement looked ahead.
pub const CUT_IN_TIME: f32 = 2.5;

/// A body beside me heading more than this many degrees towards my way is cutting in ...
pub const CUT_IN_DEG: f32 = 5.0;
/// ... heading within this many degrees of my way (one going across it is crossing traffic) ...
pub const CUT_IN_MAX_DEG: f32 = 45.0;
/// ... or moving towards it sideways faster than this (m/s).
pub const CUT_IN_SPEED: f32 = 0.2;

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
        let w = me.half_width as f64 + LEAD_MARGIN;
        let hit = match (way.corridor_hit(&b, w, look), cut_in_hit(way, &b, p.fwd(), p.speed.abs(), None, None, w, look)) {
            (Some(h), Some(c)) if c.0 < h.0 => Some(c),
            (None, Some(c)) => Some(c),
            (h, _) => h,
        };
        if let Some((d, k, fwd)) = hit {
            let mut gap = d - st.front;
            let mut cut_in = false;
            // (beside me: a lead only when it is moving into my corridor, at gap 0; review11: the
            // bus cutting in from the next lane at 6-15 degrees was dropped once part of it was
            // beside the car, and the car drove through it at 8 m/s)
            if gap < -0.5 {
                if !cutting_in(p.fwd(), p.speed.abs(), fwd, p.pos.truncate() - way.pts[k].1, None, me.half_width as f64 + LEAD_MARGIN) {
                    continue;
                }
                gap = 0.0;
                cut_in = true;
            }
            let speed = if cut_in { 0.0 } else { (p.fwd() * p.speed as f64).dot(fwd) as f32 };
            let right = DVec2::new(fwd.y, -fwd.x);
            let q = way.pts[k].1;
            let lateral = (p.pos.truncate() - q).dot(right) as f32 + st.lateral_ahead(way.pts[k].0);
            let hw = Headway { gap, speed, acc: 0.0, what: Seen::Player(p.id), standing_for: p.still, half_width: p.half_width, length: p.half_len * 2.0, held: false, at_stop: false, lateral, rest: None };
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
                let h = Headway { gap, speed: 0.0, acc: 0.0, what: Seen::Parked, standing_for: f32::INFINITY, half_width: p.half_width, length: p.half_len * 2.0, held: false, at_stop: false, lateral: p.lateral, rest: None };
                lead = Headway::nearer(lead, Some(h));
            }
        }
    }
    (lead, swerve)
}

/// Someone on foot in the strip ahead.
/// [ROLLBACK r4d-B4] The clearance (m) a bus at its stop keeps round its footprint from the
/// people standing at the kerb.
const KERB_CLEAR: f32 = 0.3;

/// A person's body radius (m, `humans::BODY_OUTSIDE`).

/// How far ahead (s) the step of a person about to enter my lane is followed, at most: the
/// time I need to reach them, at least 2 s (review-r5c-safety: at `dist / v` alone, 0.15 s at
/// 13 m/s, a stepper beside the bumper was never braked for); 4 s covers my reach of 45 m at
/// 11 m/s.
const STEP_LOOK: f32 = 4.0;
/// Who is about to enter my lane: moving toward it from within this (m) of my people strip,
/// a step and a half (a kerb waiter half a step behind the spot is 1.15 m off the strip).
/// Farther off, a walker heading for the road is followed as before (2 s at most): most
/// stop at the kerb, and followed for 4 s they held cars 30 m away for nothing (viz2
/// people_wait_s +214 s, viz3 +290 s against the 2 s rule).
const STEP_NEAR: f32 = 1.5;

/// [round 5 L3] A car shifts inside the drawn road to pass people on a kerbless pavement
/// strip beside its lane (BeijingCity's Wuluju stop, round5 case01: walkers on the inner half
/// of a pavement path laid on the asphalt, 0.3-0.5 m off the car's flank, held #2827 for 50 s)
/// rather than standing: at least this far (m) - half a metre clears walkers at 1.3-1.5 m of
/// the axis even with the two-second projection of their shuffle ...
const PEOPLE_SWERVE_MIN: f32 = 0.5;
/// ... and at most this far (the brief: a metre; beyond that it is a lane change).
const PEOPLE_SWERVE_MAX: f32 = 1.0;
/// What the shifted body keeps off the drawn road's far edge on the free side (m).
const SWERVE_EDGE_KEEP: f32 = 0.3;
/// Over how many metres of way the shift builds up (`AiState::lateral_target`'s S-curve is
/// 8 m at least): a moving car's shift has got this far at a person ahead.
const SWERVE_RAMP: f32 = 8.0;
/// Below this speed (m/s) the car is creeping: the shift it has planned counts in full
/// (standing, the ramp cannot build up at all).
const SWERVE_CREEP: f32 = 0.5;

/// Somebody in the strip of the car's (bent) way: where the stop point is, how far along,
/// where they are across the lane's axis (signed, + right) and whether a shift inside the
/// drawn road could clear them.
struct Holder {
    stop: f32,
    along: f32,
    axis: f32,
    swervable: bool,
}

/// The shift of `swerve` the car has `along` metres on before its S-curve is laid
/// (`SWERVE_RAMP`): the whole of it level with the body or creeping, the curve's share ahead.
fn shift_at(swerve: f32, along: f32, v: f32) -> f32 {
    if v < SWERVE_CREEP || along <= 0.0 {
        return swerve;
    }
    let t = (along / SWERVE_RAMP).clamp(0.0, 1.0);
    swerve * t * t * (3.0 - 2.0 * t)
}

/// How far beside my lane's footprint (m) somebody on a crossing's walk still counts as on the
/// crossing (`people_ahead`): the kerb spot (0.3 m off the lane's edge) and half a step back
/// from it (review-r5c-safety: a stander 0.3 m behind the spot who stepped on 1.5-2.2 s before
/// a 14 m/s car was counted too late at 0.5); one farther off - a lane width away, on the far
/// half of the crossing - does not (review-r5x merged list 4: within 3 m along the walk lane
/// either side, a kerb waiter 1.9 m from the axis held a car 56 s).
const ZEBRA_BESIDE: f32 = 0.8;
/// Crossings this far along my way (m, from my origin) are looked at (`zebra_cuts`), and of
/// them those whose cut lies at most `ZEBRA_AHEAD` before my front.
const ZEBRA_WAY: f32 = 40.0;
const ZEBRA_AHEAD: f32 = 30.0;

/// A pedestrian crossing's cut of my way (`Network::walks`): its walk lane, where it cuts
/// the lane's axis, the lane's right there and how far from that axis it holds.
struct ZebraCut {
    walk: usize,
    at: DVec2,
    right: DVec2,
    half: f32,
}

impl ZebraCut {
    /// Somebody at `pos` on this crossing's walk (within its width and a body) within `half`
    /// of the lane's axis.
    fn holds(&self, net: &Network, pos: DVec3) -> bool {
        let Some(walk) = net.lanes.get(self.walk) else { return false };
        walk.nearest_point(pos).is_some_and(|(_, d)| (d as f32) < walk.width * 0.5 + crate::plan::PERSON_RADIUS) && ((pos.truncate() - self.at).dot(self.right).abs() as f32) < self.half
    }
}

/// The crossings cutting my way ahead (`ZEBRA_WAY`, their cut from a metre behind my front to
/// `ZEBRA_AHEAD` before it), each holding within my lane's footprint (my own width where the
/// lane is narrower) and `ZEBRA_BESIDE` either side of it.
fn zebra_cuts(net: &Network, me: &Vehicle, lanes: &[(usize, f32)]) -> Vec<ZebraCut> {
    let st = &me.state;
    let mut out = Vec::new();
    for &(l, d0) in lanes {
        if d0 > ZEBRA_WAY {
            break;
        }
        let Some(lane) = net.lanes.get(l) else { continue };
        let half = (lane.width * 0.5).max(me.half_width) + ZEBRA_BESIDE;
        for &(w, sa, _) in net.walks.get(l).map(|v| v.as_slice()).unwrap_or(&[]) {
            let point = d0 + sa;
            if point - st.front >= ZEBRA_AHEAD || point - st.front <= -1.0 {
                continue;
            }
            let (q, h) = lane.at(sa);
            let hr = (h as f64).to_radians();
            out.push(ZebraCut { walk: w, at: q.truncate(), right: DVec2::new(hr.cos(), -hr.sin()), half });
        }
    }
    out
}

/// The people in the strip ahead: where to stop for them, and the shift inside the drawn
/// road that lets the car past the ones beside its lane instead, with where to stop once
/// shifted (for whom the shift does not clear). `parked_swerve`: a swerve round a parked car
/// already asked for (a shift the other way is not added to it).
///
/// The way is the car's bent way (`AiState::way_points` lays its lateral curve on): a person
/// holds the car where its body will really go. The shift is planned on the lane's axis, so
/// that it does not wander with the way it bends.
fn people_ahead(net: &Network, sim: &Simulation, me: &Vehicle, lanes: &[(usize, f32)], way: &WayPts, parked_swerve: Option<f32>) -> (Option<f32>, Option<(f32, Option<f32>)>) {
    let st = &me.state;
    let v = st.speed;
    let reach = people_reach(v);
    let radius = (st.front + reach + 6.0) as f64;
    // who holds the car on its way, and who would on the lane's axis (the shift's candidates)
    let mut holders: Vec<Holder> = Vec::new();
    let mut on_axis: Vec<Holder> = Vec::new();
    // [ROLLBACK r4d-B4] a bus coming up to its own stop pulls in beside the people standing
    // at the kerb: only one inside its footprint and `KERB_CLEAR` round it holds it, where
    // they stand (where their shuffle would take them stopped a bus 7.4 m short of its pole),
    // and with no "stood at the edge a while" exemption (it crept on and stopped by turns)
    let own_stop = me.serves_stops() && me.external.stop_at.is_some_and(|s| s - st.front < 40.0);
    // the current lane's footprint and drawn road (the shift is planned on it alone)
    let lane = net.lanes.get(st.lane);
    let lane_half = lane.map(|l| l.width * 0.5).unwrap_or(f32::INFINITY);
    let (drivable_left, drivable_right) = lane.map(|l| l.drivable()).unwrap_or((0.0, 0.0));
    let lane_left = lane.map(|l| l.length() - st.s).unwrap_or(0.0);
    let zebras = zebra_cuts(net, me, lanes);
    for k in sim.people_near(me.pos, radius) {
        let p = &sim.people[k];
        let rel = p.pos - me.pos;
        if rel.truncate().length() > radius || rel.z.abs() > 2.5 {
            continue;
        }
        let standing = p.vel.length() < 0.3;
        // [round 7 A3] somebody on the walk of a crossing ahead, within `ZEBRA_BESIDE` of my
        // lane's footprint, not a kerb waiter (`at_kerb`: they let the vehicle go first, P4),
        // is held for as one on the crossing: the one people test (round 4's zebra check,
        // which held for them from the walkers' list, is gone; review-r5c-safety C2: a stander
        // 0.3 m behind the kerb spot stepping on 1.5-2.5 s before the car)
        let on_crossing = p.on_crossing || (!p.at_kerb && zebras.iter().any(|z| z.holds(net, p.pos)));
        let at_kerb_of_stop = own_stop && !on_crossing && (standing || p.waiting_for_bus);
        let half = if on_crossing {
            me.half_width + 1.5
        } else if p.waiting_for_bus && me.serves_stops() {
            me.half_width - 0.3
        } else if at_kerb_of_stop {
            // (its footprint and 0.3 m clear of it)
            me.half_width + KERB_CLEAR
        } else {
            me.half_width + 0.3
        };
        // (one about to step into my lane - moving toward it from within STEP_NEAR of my
        // strip - is followed for the time I need to reach them, 2 s at least; anyone else
        // for that time at most)
        let dist = rel.truncate().length() as f32;
        let hr = (me.heading as f64).to_radians();
        let my_right = DVec2::new(hr.cos(), -hr.sin());
        let off = rel.truncate().dot(my_right);
        let stepping_in = p.vel.dot(my_right) * off < -0.01 && (off.abs() as f32) < half + STEP_NEAR;
        let t = if stepping_in { (dist / v.max(1.0)).clamp(2.0, STEP_LOOK) } else { (dist / v.max(1.0)).min(2.0) };
        let later = p.pos.truncate() + p.vel * t as f64;
        // (where somebody waiting for a bus is, not where their shuffle would take them)
        let positions: &[DVec2] = if at_kerb_of_stop || (p.waiting_for_bus && me.serves_stops()) { &[p.pos.truncate()] } else { &[p.pos.truncate(), later] };
        for &(d, q, fwd) in &way.pts {
            if d > reach + st.front {
                break;
            }
            let right = DVec2::new(fwd.y, -fwd.x);
            // (the way's offset from the lane's axis here: across the axis = across the way + it)
            let lat = st.lateral_ahead(d);
            let now_way = (p.pos.truncate() - q).dot(right) as f32;
            let now_axis = now_way + lat;
            let later_axis = (later - q).dot(right) as f32 + lat;
            // (where their body is now, off my way: the exemptions below are for somebody
            // outside my footprint, which a projected shuffle must not decide)
            let outside = now_way.abs() >= me.half_width + crate::plan::PERSON_RADIUS;
            // [round 5 L3] one beside the lane - outside its footprint now and after their
            // step, their body outside mine (C2 holds one inside it), inside the drawn road -
            // on the car's own lane: a shift inside the drawn road may clear them (a bus pulls
            // in beside its people; one on a crossing, or stepping into the lane, holds the car)
            let strip = if now_axis < 0.0 { drivable_left } else { drivable_right };
            let swervable = |along: f32| {
                !me.serves_stops()
                    && !on_crossing
                    && outside
                    && along <= lane_left
                    && now_axis.abs() > lane_half
                    && later_axis.abs() > lane_half
                    && now_axis.abs() < strip
                    && now_axis.signum() == later_axis.signum()
            };
            for &pos in positions {
                let r = pos - q;
                let along = r.dot(fwd);
                let across_way = r.dot(right) as f32;
                let across = across_way.abs();
                if along.abs() > WAY_STEP as f64 * 0.5 + 0.05 {
                    continue;
                }
                // [ROLLBACK r4c-L4] somebody waiting at the kerb to cross, outside my own
                // footprint and its 0.3 m (to the 0.02 m beyond their body radius: the kerb spot is half a lane and
                // 0.3 m off the lane's axis, and a bus is as wide as a lane), holds nobody:
                // with a stream on the crossing they let the vehicle go first (P4)
                if p.at_kerb && !on_crossing && across >= me.half_width + crate::plan::PERSON_RADIUS {
                    continue;
                }
                // [round5 C2] somebody level with my body (their body behind my front
                // plane, now and where their step takes them, `later`) and outside my
                // footprint holds nothing: my front never reaches them; one ahead of the
                // front, or stepping there, holds as before (viz4 case01 #2827: eleven on
                // the pavement 0.3 m off its flank, level with its body, held it 34 s). At
                // any speed, not only standing: let go below 0.3 m/s alone, the car was
                // held again the moment it rolled and crept at 0.1 m/s (test c2_*: 2.3 s
                // held of 6). One level with my body INSIDE my footprint is being hit:
                // held as before (review-r5c-safety: 25 new contacts at across 0.7-1.5 m,
                // 1-7 m/s). Not a bus pulling in at its stop (B4).
                if !at_kerb_of_stop && outside && d + (along as f32) < st.front - crate::plan::PERSON_RADIUS {
                    continue;
                }
                // someone who has stood at the edge for a while does not hold a standing car
                // (a bus at its stop: one inside its footprint does, or it crept on and
                // stopped by turns) - at the edge: outside my footprint and their body
                // (review-r5c-safety: at `half - 0.9` a red-light waiter standing 0.5-1.1 m
                // off my axis was crawled through at 0.25 m/s once the zebra check no longer
                // listed them)
                if !on_crossing && !at_kerb_of_stop && v < STANDING_SPEED && p.still > 3.0 && outside {
                    continue;
                }
                // (where the person really is relative to the sample, so that the stop
                // point does not jump with the sampling)
                let stop = d + along as f32 - 1.5;
                let axis = across_way + lat;
                let h = Holder { stop, along: d + along as f32, axis, swervable: swervable(d + along as f32) };
                if across < half {
                    holders.push(Holder { ..h });
                }
                if axis.abs() < half {
                    on_axis.push(h);
                }
            }
        }
    }
    if holders.is_empty() && on_axis.is_empty() {
        return (None, None);
    }
    // the shift: everybody who would hold the car on the axis is beside the lane on one side,
    // there is room for it on the other, and no swerve round a parked car goes the other way
    let swerve = (|| {
        if on_axis.is_empty() || !on_axis.iter().all(|h| h.swervable) {
            return None;
        }
        let side = on_axis[0].axis.signum();
        if on_axis.iter().any(|h| h.axis.signum() != side) {
            return None;
        }
        let half = me.half_width + 0.3;
        let need = on_axis.iter().map(|h| half - h.axis.abs()).fold(0.0f32, f32::max).clamp(PEOPLE_SWERVE_MIN, PEOPLE_SWERVE_MAX);
        let room = if side > 0.0 { drivable_left } else { drivable_right };
        if room - me.half_width - need < SWERVE_EDGE_KEEP {
            return None;
        }
        let swerve = -side * need;
        if parked_swerve.is_some_and(|s| s.signum() != swerve.signum() && s.abs() > 0.05) {
            return None;
        }
        Some(swerve)
    })();
    // (with the shift, whoever it clears holds nothing: level with the body or creeping, by
    // the shift as planned; ahead of a moving car by the way as it is bent already, or by the
    // curve's share while the curve is not laid yet)
    let half = me.half_width + 0.3;
    let nearest = |stops: &mut dyn Iterator<Item = f32>| {
        let s = stops.fold(f32::INFINITY, f32::min);
        s.is_finite().then_some(s)
    };
    let stop = nearest(&mut holders.iter().map(|h| h.stop));
    let shifted = swerve.map(|sw| {
        let laid = (st.lateral_target - sw).abs() < 0.01 && (st.lateral_ramp.1 - sw).abs() < 0.01;
        let cleared = |h: &Holder| h.swervable && (v < SWERVE_CREEP || h.along <= 0.0 || !laid) && (h.axis - shift_at(sw, h.along, v)).abs() >= half;
        (sw, nearest(&mut holders.iter().filter(|h| !cleared(h)).map(|h| h.stop)))
    });
    (stop, shifted)
}

/// How far ahead (m from the origin) the farthest person within the car's strip stands:
/// what a shift past them has to cover.
fn reach_of_people(sim: &Simulation, me: &Vehicle, way: &WayPts) -> f32 {
    let st = &me.state;
    let v = st.speed;
    let reach = people_reach(v);
    let radius = (st.front + reach + 6.0) as f64;
    let half = (me.half_width + 0.3) as f64;
    let mut far = 0.0f32;
    for k in sim.people_near(me.pos, radius) {
        let p = &sim.people[k];
        if (p.pos - me.pos).truncate().length() > radius {
            continue;
        }
        for &(d, q, fwd) in &way.pts {
            if d > reach + st.front {
                break;
            }
            let r = p.pos.truncate() - q;
            let right = DVec2::new(fwd.y, -fwd.x);
            if r.dot(fwd).abs() <= WAY_STEP as f64 * 0.5 + 0.05 && r.dot(right).abs() < half {
                far = far.max(d + r.dot(fwd) as f32);
            }
        }
    }
    far
}

/// Light `li` of controller `ctl` is a detector, not a stop line: its program is one constant
/// green phase (X10's "Loop", "HVZ" and "Nacht" lights on 1 m paths before the real line, that
/// the program's `[traffic_light_jump]` points test through the lane's `[approachdist]` request).
/// Nobody stops at it and it hides no later line (x10 case04: 14 of the 16 red crossings of
/// 54 min had one 1 m before the real line, which came into view only over the bumper). The
/// lane's request is raised as for any lit lane (`Simulation::update_lights`).
pub fn detector_light(sim: &Simulation, ctl: usize, li: usize) -> bool {
    sim.lights.get(ctl).and_then(|c| c.lights.get(li)).is_some_and(|phases| phases.len() == 1 && matches!(TrafficLightController::aspect(phases[0].0), Aspect::Green))
}

/// The light on lane `l`, unless it is a detector (`detector_light`).
pub fn stop_light(net: &Network, sim: &Simulation, l: usize) -> Option<(usize, usize)> {
    net.lanes.get(l)?.traffic_light.filter(|&(ctl, li)| !detector_light(sim, ctl, li))
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
        let Some((ctl, li)) = stop_light(net, sim, l) else { continue };
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

/// A vehicle slower than this (m/s) past the junction's conflicts is standing there as far
/// as the room on the exit goes (a car moving off is past 2 m/s within a second; a crawl is
/// not moving off). Was 1.5: x10 case06, a third left-turner entered behind one crawling at
/// 0.5-2 m/s, stopped with its tail across the through lane and held it 11 s.
const EXIT_SLOW: f32 = 2.0;
/// The exit's first metres past the conflicts: a vehicle crawling there, or rolling up to a
/// standing lead there, is a queue forming, and the box is not entered behind it (x10 case06;
/// the figure `heads` reports the exit's state over).
const EXIT_NEAR: f32 = 15.0;

/// The next junction on the way (or the one the car is in), up to the claim horizon
/// (`sim::claim_horizon`, [round 7 A8]: a fixed 120 m started the claims 5.4 s out at
/// 22 m/s, not 8), with the room past its conflicts measured along the car's own way: a vehicle standing inside the piece past the
/// meeting place, or on the lanes after it within 40 m, leaves no room ("keep clear").
fn junction_ahead(net: &Network, sim: &Simulation, vehicles: &[Vehicle], i: usize, way: &[(usize, f32)]) -> Option<JunctionView> {
    let me = &vehicles[i];
    let js: &Junctions = &sim.junctions;
    let horizon = crate::sim::claim_horizon(me.state.speed);
    for (k, &(l, d0)) in way.iter().enumerate() {
        if d0 - me.state.front > horizon {
            break;
        }
        let Some(&(j, m)) = js.of_lane.get(&l) else { continue };
        let inside = d0 <= 0.0;
        // the pieces my way takes through the object, and where their meeting places
        // begin and end in my way's distances (they may lie on the pieces after this one)
        let chain = js.chain_along(j, &way[k..]);
        let foes = js.foes_along(j, &chain);
        let way_len: f32 = chain.iter().map(|&(mm, _)| net.lanes[js.nodes[j].movements[mm].lane].length()).sum();
        // (the room past the conflicts is measured from the end of the last crossing or
        // joint; a stretch where the way runs side by side with another lane, `Foe::side`, is
        // a queueing lane, not a box to keep clear: counting it put the "box" of BeijingCity's
        // twin stub chains 60 m on, where the exit queue's tail closed it for 25 s with the
        // crossing itself clear, viz3 case03 #4436; the resolver's side-merge rules and the
        // roll-up stop, `plan::roll_up_point`, keep a car off the one standing beside)
        let first = foes.iter().map(|f| f.mine.0).fold(f32::INFINITY, f32::min);
        let last = foes.iter().filter(|f| !f.side).map(|f| f.mine.1).fold(0.0f32, f32::max);
        let conflict_start = d0 + if first.is_finite() { first } else { way_len };
        let clear_from = d0 + if last > 0.0 { last } else { way_len };
        let need = me.length() + me.state.min_gap;
        // [round6 J6] a box to keep clear: a crossing on the way through, or a zebra cut; a
        // merge-only lane join has no box, and the queue-place term below (`EXIT_NEAR`) does not
        // apply there, only the ordinary room (case10: taxis stood 22 s each at the j31 join,
        // exit-full for a queue forming 15 m on)
        let crosses = foes.iter().any(|f| f.kind == FoeKind::Cross) || chain.iter().any(|&(mm, _)| net.walks.get(js.nodes[j].movements[mm].lane).is_some_and(|w| !w.is_empty()));
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
                if o.state.speed < EXIT_SLOW && tail > clear_from - 0.5 {
                    room = room.min(tail - clear_from);
                }
                // a queue forming in the exit's first metres: one crawling there (rolling under
                // `EXIT_SLOW`) leaves no room behind it whatever the gap to its tail now (not a
                // bus serving its stop there: it leaves on its own schedule);
                // one standing there, or rolling up to a standing lead, closes the box when
                // the place I would queue in behind it - my need behind its tail where it stops,
                // `predicted_tail` - lies inside those first metres: x10 case06 #11146 entered
                // behind #11131 rolling at 4.6 m/s towards a queue 18 m ahead, and stood with
                // its tail across the through lane; a queue tail 40 m past a crossing, viz3
                // case03/16, leaves the box open)
                let crawling = o.state.speed >= STANDING_SPEED && o.state.speed < EXIT_SLOW;
                let standing_lead = o.last.lead.filter(|l| l.speed < STANDING_SPEED);
                let queueing = o.state.speed < STANDING_SPEED || standing_lead.is_some();
                let predicted_tail = tail + standing_lead.map(|l| (l.gap - o.state.min_gap).max(0.0)).unwrap_or(0.0);
                if crosses && !o.external.at_stop && tail > clear_from - 0.5 && ((crawling && tail - clear_from < EXIT_NEAR) || (queueing && predicted_tail - clear_from < EXIT_NEAR + need)) {
                    room = 0.0;
                }
            }
            for &(k, s, lateral) in sim.player_occupancy.get(&wl).map(|v| v.as_slice()).unwrap_or(&[]) {
                let p = &sim.players[k];
                // (a player snapped to this lane from beside it, its body clear of the lane by
                // the kerb margin, as `Simulation::player_parked_beside`: in a queue 3.5 m to the
                // left of the exit lane, the player's bus closed the box for 25 s, viz3 case03)
                if crate::sim::beside_lane(net, wl, lateral, p.half_width) {
                    continue;
                }
                let tail = wd + s - p.half_len;
                if p.speed.abs() < EXIT_SLOW && tail > clear_from - 0.5 {
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
            if let Some((ctl, li)) = stop_light(net, sim, wl) {
                light = Some((ctl, li, wd));
                break;
            }
        }
        if light.is_none() && inside && k == 0 {
            light = light_behind(net, sim, js, j, l, d0, me.state.prev_lane);
        }
        let past_line = light.is_some_and(|(_, _, d)| d + 0.5 < me.state.front);
        let turned = light.is_some_and(|(ctl, li, _)| sim.lights.get(ctl).is_some_and(|c| !matches!(TrafficLightController::aspect(c.state(li)), Aspect::Green | Aspect::GreenYellow | Aspect::Dark)));
        // (it passed the line while the light let it in: `Vehicle::passed_on_go`)
        // (and only once it cannot stop before its first meeting place, or is at it: a car
        // over its line on yellow 16 m short of it, that could still stop, was committed and
        // went in front of a car it had been refused for; review9)
        let to_line = light_line - me.state.front - 0.6;
        let cannot_stop = crate::plan::cannot_stop(me.state.speed, to_line, crate::plan::HARD_BRAKE);
        let clearing = past_line && turned && light.map(|(c, l, _)| (c, l)) == me.passed_on_go && (cannot_stop || light_line - me.state.front <= 0.5);
        return Some(JunctionView { junction: j, movement: m, entry: d0, inside, chain, conflict_start, conflict_end: clear_from, exit_room: room >= need, light_line, light, past_line, clearing, way_len });
    }
    None
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
        if speed < STANDING_SPEED && pos > os - pass_len - 15.0 {
            blocked = true;
            clear_for = 0.0;
            return;
        }
        // not yet level: the distance it still has to me
        let dist = (os - pos).max(0.0);
        let mut t = arrival_time(dist, speed, accel, limit);
        // a car standing at a red light with time to go: counts from when it goes
        if at_red && speed < STANDING_SPEED {
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
        let people_reach = people_reach(st.speed) + st.front;
        let player_near = sim.players.iter().any(|pl| (pl.pos - me.pos).truncate().length() < 180.0);
        let (cand, far) = body_candidates(sim, vehicles, i, &p.way, look, body_look);
        let upto = body_look.max(people_reach).max(far).max(if player_near { look } else { 0.0 });
        let way = WayPts::sample(net, st, upto);
        let (on_lane, body) = lead_by_bodies(net, sim, vehicles, i, &p.way, &way, look, body_look, cand, &mut p.ahead);
        p.leader = Headway::nearer(on_lane, p.merging);
        if player_near {
            let pl = player_on_way(sim, me, &way, look);
            if let Some(h) = pl {
                p.ahead.push((h.gap, h.gap + h.length, h.speed));
            }
            p.leader = Headway::nearer(p.leader, pl);
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
        if let Some(h) = parked {
            p.ahead.push((h.gap, h.gap + h.length, 0.0));
        }
        p.ahead.sort_by(|a, b| a.0.total_cmp(&b.0));
        p.leader = Headway::nearer(p.leader, parked);
        let (people_stop, shifted) = people_ahead(net, sim, me, &p.way, &way, swerve);
        // [round 5 L3] the shift past people beside the lane, towards the middle only while
        // the oncoming lane is clear for its length (the shifted body may touch the oncoming
        // lane's painted edge)
        let shifted = shifted.filter(|&(sw, _)| {
            if sw.signum() != sim.oncoming_sign() {
                return true;
            }
            let pass_len = reach_of_people(sim, me, &way) + me.length() + 6.0;
            match oncoming(net, sim, vehicles, i, pass_len) {
                None => true,
                Some(o) => {
                    let v_cap = p.v_want.clamp(4.0, 14.0);
                    !o.blocked && o.clear_for > arrival_time(pass_len, st.speed, st.accel, v_cap) + 1.5
                }
            }
        });
        p.kerb_swerve = match (swerve, shifted) {
            (Some(a), Some((b, _))) => Some(if b.abs() > a.abs() { b } else { a }),
            (a, b) => a.or(b.map(|s| s.0)),
        };
        p.people_swerve = shifted.is_some();
        p.people_stop = match shifted {
            Some((_, stop)) => stop,
            None => people_stop,
        };
        p.signal = signal_ahead(net, sim, &p.way, st.rear);
        p.junction = junction_ahead(net, sim, vehicles, i, &p.way);
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

#[cfg(test)]
mod cut_in_tests {
    use super::{cutting_in, lane_room_toward};
    use glam::DVec2;

    /// A car beside me, 2.6 m to my right, heading 8 degrees towards my way at 9 m/s, with its
    /// middle 3 m ahead of my way's sample.
    fn beside() -> (DVec2, f32, DVec2, DVec2) {
        let a = 8f64.to_radians();
        (DVec2::new(-a.sin(), a.cos()), 9.0, DVec2::Y, DVec2::new(2.6, 3.0))
    }

    /// Review11 re-check: a car keeping its lane whose line takes it no nearer is not cutting
    /// in, whatever its heading says; one changing lanes, passing, or the player (no lane room)
    /// is; and one whose line still brings it a metre towards me is.
    #[test]
    fn a_car_keeping_its_lane_is_not_cutting_in() {
        let (o_fwd, speed, f, rel) = beside();
        assert!(cutting_in(o_fwd, speed, f, rel, None, 1.2));
        // its lane runs parallel to my way
        let par = DVec2::Y;
        // it sits 0.6 m left of its lane's line, drifted towards me, and goes back to it (I
        // am on its left: its line lies 0.6 m to its right)
        assert!(!cutting_in(o_fwd, speed, f, rel, Some((0.6, par)), 1.2));
        // its line is where it is: it stays
        assert!(!cutting_in(o_fwd, speed, f, rel, Some((0.0, par)), 1.2));
        // its line lies 1 m to its left, towards me: it comes
        assert!(cutting_in(o_fwd, speed, f, rel, Some((-1.0, par)), 1.2));
        // its line lies 1 m to its right, away from me: it does not
        assert!(!cutting_in(o_fwd, speed, f, rel, Some((1.0, par)), 1.2));
        // its lane itself runs 8 degrees into my way (lanes converging): it comes, whatever
        // its line's offset says
        assert!(cutting_in(o_fwd, speed, f, rel, Some((0.0, o_fwd)), 1.2));
        assert!(cutting_in(o_fwd, speed, f, rel, Some((1.0, o_fwd)), 1.2));
        assert_eq!(lane_room_toward(-1.0, 2.6), 1.0);
        assert_eq!(lane_room_toward(1.0, 2.6), 0.0);
        assert_eq!(lane_room_toward(1.0, -2.6), 1.0);
        // its middle already inside my corridor: in, whatever its line says
        assert!(cutting_in(o_fwd, speed, f, DVec2::new(1.0, 3.0), Some((0.0, par)), 1.2));
    }
}
