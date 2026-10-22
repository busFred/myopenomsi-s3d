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

/// [round 7 A3 fix] Somebody moving at this pace (m/s) or more is stepping, and is judged
/// where their step takes them as well as where they are (`people_ahead`); slower is a shuffle
/// (round 5 C2: 0.1 m/s beside a standing car's body).
const STEP_SPEED: f32 = 0.5;

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
    /// [round 7 A2-full] Distance from the origin to where the junction begins on this way:
    /// the movement's lane's start (`entry`), but for a lane that is a movement only by
    /// joining another way (a spline lane, `Junctions::extend`) a metre before its first
    /// meeting place - up to there it is an approach lane like any lane leading in (the
    /// claim horizon applies, the line is there), from there on it is inside.
    pub line: f32,
    /// The origin is past `line`.
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
    /// [J7-3] That vehicle's index, when it is an AI vehicle (the one asked to make room,
    /// `plan`'s cooperation).
    pub behind_id: Option<usize>,
    /// [J7-3, fix pass 2] That vehicle's standing gap (m; a player's: the timetable bus preset's):
    /// the gap a change needs from it at rest (`plan::gap_needed_behind`).
    pub behind_min_gap: f32,
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
    /// [round 7 A7] The swerve round a parked car alone (`kerb_swerve` without the shift
    /// past people): what a timetable bus between its stops takes.
    pub parked_swerve: Option<f32>,
    pub oncoming: Option<OncomingView>,
    /// A player's vehicle ahead indicating to pull out towards the traffic: the distance
    /// from my origin to its rear.
    pub letting_out: Option<f32>,
    /// Every body in my corridor ahead (vehicles, players, parked cars): (gap from my front to
    /// its near edge, to its far edge, its speed along my way), nearest first. What lies
    /// beyond the lead (a pass needs room to come back in, `plan`).
    pub ahead: Vec<(f32, f32, f32)>,
    /// The current lane's limit at the driver's desire (m/s).
    pub v_want: f32,
    /// [J7-3] Where my lane ends for me ahead (`lane_end`).
    pub lane_end: Option<LaneEnd>,
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

/// What two neighbouring samples' spans overlap by (m, round 4's people test: half a step and
/// this), so that somebody exactly between them is in one of them.
const SAMPLE_OVERLAP: f64 = 0.05;

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

    /// [P8-5] How far behind and ahead of sample `i` (m, along its heading) a point belongs to
    /// it: half the real distance to each neighbour and the 5 cm overlap of round 4 (half a
    /// `WAY_STEP` where there is no neighbour). The samples are `WAY_STEP` apart along the way,
    /// but not on the ground: over a junction's short pieces two of them lay 2.5 m apart, and a
    /// walker between them was seen in one frame and not in the next (viz3 case09 #20705 hit a
    /// walker 1.14 m deep at 5 m/s with its people stop on and off frame by frame).
    fn sample_span(&self, i: usize) -> (f64, f64) {
        let half = |a: usize, b: usize| self.pts.get(a).zip(self.pts.get(b)).map(|(p, q)| (q.1 - p.1).length() * 0.5).unwrap_or(WAY_STEP as f64 * 0.5);
        let back = if i == 0 { WAY_STEP as f64 * 0.5 } else { half(i - 1, i) };
        (back + SAMPLE_OVERLAP, half(i, i + 1) + SAMPLE_OVERLAP)
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
        if o.last.lead_id == Some(me.id) {
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
        // (two bodies blocking each other, both moving: the higher id goes. [round 7 A2-full]
        // Not past one that stands: a standing body is the lead of whoever rolls into it,
        // whatever it waits for - viz2 case11's cars rolled at 4-5 m/s into a car standing at
        // the join of 9844/9845 and 10340, 1.28 m deep, as it named them its body block; two
        // that both stand are left to the negotiation of a mutual hold, `Simulation`'s
        // `mutual`, review11's TODO.)
        if !on_way && o.last.body_block == Some(me.id) && me.id > o.id && o.state.speed >= STANDING_SPEED {
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
        if o.last.lead_id == Some(me.id) && rel.dot(my_fwd) < 0.0 {
            continue;
        }
        // [J7-1] a body crossing my way that is out of my corridor before my front can be where
        // it is: no lead - SUMO's follower case of `MSLink::blockedByFoe` at a crossing (its
        // leave time before my arrival), the junction leader that has passed the conflict point
        // by the time ego arrives (`MSVehicle::adaptToJunctionLeader`, `MSLink::getLeaderInfo`).
        // (Its leave time is where its body leaves my corridor at its sideways speed: OMSI's
        // junction pieces give the body, not SUMO's conflict point.) (The junction let the other through 1 s ahead
        // of me by the time test; braking for its body as a lead standing across the way, a
        // 13 m/s main road car braked 5 m/s² for a side car out of its way 1.3 s before it
        // came, adversarial2 F5b.)
        // (only one on its way across: not one held - a stop of its own or a lead near, its
        // `Decision::held` - which may come to rest in my corridor; SUMO's `willPass`. A car held
        // by its lead 4.9 m on, crossing at 1.4 m/s, was dropped and stopped in my corridor,
        // viz case01 #1674/#118)
        if !on_way && !cut_in && o.state.speed >= STANDING_SPEED && !o.last.held {
            let right = DVec2::new(f.y, -f.x);
            let q = way.pts[k].1;
            let lats = ob.corners().map(|c| (c - q).dot(right));
            let (lo, hi) = (lats.iter().copied().fold(f64::INFINITY, f64::min), lats.iter().copied().fold(f64::NEG_INFINITY, f64::max));
            let v_lat = o.state.speed as f64 * o_fwd.dot(right);
            let t_clear = if v_lat > 1e-3 { (w - lo) / v_lat } else if v_lat < -1e-3 { (hi + w) / -v_lat } else { f64::INFINITY };
            // (my limit as the claim's, `Simulation::tick`)
            let v_max = (net.lanes[st.lane].speed_limit_kmh / 3.6 * st.desire).min(st.max_speed_kmh / 3.6);
            let t_me = omsi_sim::traffic::arrival_time(gap.max(0.0), st.speed, st.accel, v_max);
            if (t_clear as f32) < t_me {
                continue;
            }
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
            if o.last.lead_id == Some(me.id) {
                continue;
            }
            let standing_for = if o.external.at_stop { o.external.standing_for.max(o.stopped) } else { o.stopped };
            let hw = Headway { gap, speed: if occ.foreign { -o.state.speed } else { o.state.speed }, acc: o.state.acc, what: Seen::Vehicle(occ.vehicle), standing_for, half_width: o.half_width, length: o.length(), held: o.last.held || o.last.stop.is_some(), at_stop: o.external.at_stop, lateral: occ.lateral, rest: Headway::rest_of(o, gap, st.front) };
            on_lane = Headway::nearer(on_lane, Some(hw));
        }
    }
    (on_lane, body)
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
    // (where the plan's lateral is the kerb swerve: no pass, no lane change, no bay)
    let swerve_holds = me.passing.is_none() && st.change.is_none() && me.external.bay.is_none();
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
            // ([fix pass 1, 13] against the offset the car holds without the kerb swerve - the
            // lane's axis where the swerve is the plan's lateral, `plan` rule 9: measured
            // against `lateral_ahead`, which includes the swerve itself, the need flickered
            // 0 / -0.35 / 0 frame by frame on 37 cars of a run)
            let my_lat = if swerve_holds { 0.0 } else { st.lateral_ahead(along.max(0.0)) };
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
    /// Distance from my origin along my way to the cut.
    along: f32,
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

/// [P8-5] SUMO's `MSPModel::SAFETY_GAP` (1 m, `MSPModel.cpp`): the band a vehicle asks the
/// crossing about is its width and this (`MSLink::getLeaderInfo`: `vehWidth = width +
/// SAFETY_GAP`).
const CROSSING_SAFETY_GAP: f32 = 1.0;
/// SUMO's `jmCrossingGap` default (10 m, `JM_CROSSING_GAP_DEFAULT` in `MSLink.cpp`): somebody
/// walking on a crossing towards my band holds me from this far along the crossing.
const JM_CROSSING_GAP: f32 = 10.0;
/// SUMO's `blockedAtDist` (`MSPModel_Interacting.cpp`): somebody approaching the band holds
/// the vehicle only while they have waited less than this (s); one standing longer holds it
/// only from inside the band.
const CROSSING_STANDING_WAIT: f32 = 2.0;

impl ZebraCut {
    /// [P8-5] SUMO's `blockedAtDist` for somebody on this crossing's walk: they hold a vehicle
    /// whose band (its width and `CROSSING_SAFETY_GAP`, centred `lat` off the lane's axis) they
    /// have not completely passed, when they are inside it, or walk towards it from at most
    /// `JM_CROSSING_GAP` and have not stood `CROSSING_STANDING_WAIT`. Their walking direction is
    /// their step across my lane; somebody without one (standing) holds only from inside.
    fn blocked_by(&self, net: &Network, p: &crate::sim::Person, half_width: f32, lat: f32) -> bool {
        let Some(walk) = net.lanes.get(self.walk) else { return false };
        if !walk.nearest_point(p.pos).is_some_and(|(_, d)| (d as f32) < walk.width * 0.5 + crate::plan::PERSON_RADIUS) {
            return false;
        }
        let r = crate::plan::PERSON_RADIUS;
        let band = half_width + CROSSING_SAFETY_GAP * 0.5;
        // (across my lane: where they are from the band's middle, and their step)
        let off = (p.pos.truncate() - self.at).dot(self.right) as f32 - lat;
        let step = p.vel.dot(self.right) as f32;
        if off.abs() < band + r {
            return true;
        }
        // (approaching: only somebody already on the crossing - the game's crossing flag, SUMO's
        // pedestrian on the crossing lane - not one walking up to it on the pavement over the
        // walk's ends: SUMO's walking area is no crossing)
        if !p.on_crossing || step.abs() < 0.01 || p.still >= CROSSING_STANDING_WAIT {
            return false;
        }
        // (the distance along their walk from their front to the band's near edge: negative
        // once they are past it, the band then behind them)
        let front_dist = -off * step.signum() - band - r;
        front_dist >= 0.0 && front_dist <= JM_CROSSING_GAP
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
            out.push(ZebraCut { walk: w, along: point, at: q.truncate(), right: DVec2::new(hr.cos(), -hr.sin()), half });
        }
    }
    out
}

/// [P8-1] Is `pos` level with where my body will stand at the stop I am coming up to
/// (`External::stop_at`, my front's place there): along my way between my rear and my front
/// as they will stand, a body's radius either side - the stop's own places.
fn at_stop_place(me: &Vehicle, way: &WayPts, pos: DVec2) -> bool {
    let Some(stop) = me.external.stop_at else { return false };
    let Some(k) = way.nearest(pos, f64::INFINITY) else { return false };
    let (d, q, fwd) = way.pts[k];
    let along = d + (pos - q).dot(fwd) as f32;
    let r = crate::plan::PERSON_RADIUS;
    along >= stop - me.state.front - me.state.rear - r && along <= stop + r
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
    // ([fix pass 3, 3] my body already over a crossing my way cuts: I clear it forward - the
    // people on it treat me as an obstacle, SUMO's pedestrian side, sumo.md §4.3)
    let covering = lanes.iter().any(|&(l, d0)| net.walks.get(l).is_some_and(|ws| ws.iter().any(|&(_, sa, _)| d0 + sa >= -st.rear && d0 + sa <= st.front)));
    // [P8-5] SUMO's crossing test (`MSLink::getLeaderInfo` with `blockedAtDist`, sumo.md 4.3):
    // for every crossing my way cuts whose zebra stop my front has not passed (SUMO keeps
    // driving once on top of it: `distToCrossing - jmStoplineCrossingGap >= -SAFETY_GAP`;
    // here once past the stop before it, the strip and body tests below decide, so that a
    // car that could not stop short does not stand on the crossing), anybody on its
    // walk who is in my band or walking towards it from within `JM_CROSSING_GAP` holds me
    // short of the crossing, whatever their pace (viz4 case04 #18965 moved off at a crossing
    // while a walker 3.5 m off its axis came at 1.7 m/s; the strip test saw them in its
    // 2 s step only once they were 0.4 m from its flank). Kerb waiters (on the walking area,
    // not the crossing) and bus waiters are no crossing users.
    let sideways = me.passing.is_some() || st.change.is_some_and(|c| c.bypass);
    let on_walks = if zebras.is_empty() { Vec::new() } else { sim.people_near(me.pos, radius + (JM_CROSSING_GAP + me.half_width) as f64) };
    for z in zebras.iter().filter(|z| z.along - crate::plan::ZEBRA_STOP > st.front) {
        let lat = st.lateral_ahead(z.along);
        if on_walks.iter().any(|&k| {
            let p = &sim.people[k];
            !p.at_kerb && !p.waiting_for_bus && (p.pos.z - me.pos.z).abs() <= 2.5 && z.blocked_by(net, p, me.half_width, lat)
        }) {
            let stop = z.along - crate::plan::ZEBRA_STOP;
            holders.push(Holder { stop, along: z.along, axis: 0.0, swervable: false });
        }
    }
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
        let cut = zebras.iter().find(|z| z.holds(net, p.pos));
        let on_crossing = p.on_crossing || (!p.at_kerb && cut.is_some());
        // (somebody on the crossing my body covers - level with my body or within the
        // crossing's width past my front - holds me only inside my body now; one farther
        // ahead is on the next crossing and holds as any)
        let hd = (me.heading as f64).to_radians();
        let ahead_of_front = rel.truncate().dot(DVec2::new(hd.sin(), hd.cos())) as f32 - st.front;
        if covering && on_crossing && ahead_of_front < crate::plan::ZEBRA_PAST + crate::plan::PERSON_RADIUS {
            if omsi_sim::collision::Obb::point(p.pos, crate::plan::PERSON_RADIUS as f64).separation(&me.obb()) < 0.0 {
                holders.push(Holder { stop: st.front, along: st.front, axis: 0.0, swervable: false });
            }
            continue;
        }
        let at_kerb_of_stop = own_stop && !on_crossing && (standing || p.waiting_for_bus);
        // [P8-1] somebody waiting for a bus at the stop this bus is coming up to, level with
        // where its body will stand there (`at_stop_place`): round 4d's pull-in beside them
        // holds for them only inside its footprint. Any other waiter - at another stop, or
        // for a bus going by - is held for by the car rule (viz5 case02: timetable buses drove
        // through waiters standing 0.3 m inside their flank away from their own stop)
        let my_waiter = p.waiting_for_bus && own_stop && at_stop_place(me, way, p.pos.truncate());
        let half = if on_crossing {
            me.half_width + 1.5
        } else if my_waiter {
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
        let positions: &[DVec2] = if at_kerb_of_stop || my_waiter { &[p.pos.truncate()] } else { &[p.pos.truncate(), later] };
        // ([fix pass 1, 2; pass 2, 3] anybody whose body is over mine - beside it or behind
        // the front plane, now or where their step takes them - holds the car where it stands
        // until they are clear of it (SUMO: vehicles brake for persons on their lane): the
        // way's samples start at the front and let a car move off into a walker crossing
        // through its flank, x10 case03 #2786 0.85-1.36 m, and a coach creep through somebody
        // 0.9 m behind its reference point, viz3 case08 #11190)
        {
            let body = me.obb();
            // (where they are; where their step takes them only for one stepping - a shuffle
            // is no step, round 5 C2)
            // ([fix pass 3, 3] and only somebody inside it NOW holds a car whose body already
            // covers a crossing: it clears the crossing forward - viz3 case20 #59845 stopped at
            // 3.7 m/s with the crossing under it and stood 78 s with a walker's projected step
            // inside its body)
            let stepping = (on_crossing || p.vel.length() >= STEP_SPEED as f64) && !covering;
            let at: &[DVec2] = if stepping { positions } else { &positions[..1] };
            if at.iter().any(|&pos| omsi_sim::collision::Obb::point(pos.extend(p.pos.z), crate::plan::PERSON_RADIUS as f64).separation(&body) < 0.0) {
                holders.push(Holder { stop: st.front, along: st.front, axis: 0.0, swervable: false });
            }
        }
        // [P8 pass 1, S2] a pass or bypass moving sideways: the body as it will go on the bent
        // way over one body length against where they are and where their step takes them,
        // held short of the first place it would cover them (SUMO checks the target lane's
        // pedestrians for every change, MSLaneChanger: `targetLane->hasPedestrians()`): the
        // strip test exempts somebody level with the body and outside its footprint, whom the
        // sideways move reaches (viz3 case11 #29300 passed out into a walker level with it)
        if sideways {
            let mut safe = 0.0f32;
            for &(d, q, fwd) in way.pts.iter().skip(1) {
                if d > st.front + st.rear {
                    break;
                }
                let pose = omsi_sim::collision::Obb::vehicle(q, fwd.x.atan2(fwd.y).to_degrees(), st.front as f64, st.rear as f64, me.half_width as f64);
                if positions.iter().any(|&pos| omsi_sim::collision::Obb::point(pos.extend(p.pos.z), crate::plan::PERSON_RADIUS as f64).separation(&pose) < 0.0) {
                    holders.push(Holder { stop: st.front + safe, along: st.front + safe, axis: 0.0, swervable: false });
                    break;
                }
                safe = d;
            }
        }
        for (i, &(d, q, fwd)) in way.pts.iter().enumerate() {
            if d > reach + st.front {
                break;
            }
            let (span_back, span_ahead) = way.sample_span(i);
            let right = DVec2::new(fwd.y, -fwd.x);
            // (the way's offset from the lane's axis here: across the axis = across the way + it)
            let lat = st.lateral_ahead(d);
            let now_way = (p.pos.truncate() - q).dot(right) as f32;
            let now_axis = now_way + lat;
            let later_axis = (later - q).dot(right) as f32 + lat;
            // (where their body is now, off my way: the shift below is for somebody outside my
            // footprint, which a projected shuffle must not decide)
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
                // ([round 7 A3 fix] outside my footprint at this position - now or where their
                // step takes them - for the exemptions below: one stepping into my body from
                // beside it holds the car; review-cleanup-crate-safety: x10 case15 bus #33316
                // swept a walker on a crossing 1.20 m deep at 4.4 m/s)
                // (a shuffle is no step: round 5 C2's people shuffling at 0.1 m/s beside a
                // standing car's body do not hold it; anybody on a crossing is taken where
                // their step takes them, whatever their pace)
                let stepping = on_crossing || p.vel.length() >= STEP_SPEED as f64;
                let outside_here = outside && (!stepping || across >= me.half_width + crate::plan::PERSON_RADIUS);
                if along < -span_back || along > span_ahead {
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
                if !at_kerb_of_stop && outside_here && d + (along as f32) < st.front - crate::plan::PERSON_RADIUS {
                    continue;
                }
                // someone who has stood at the edge for a while does not hold a standing car
                // (a bus at its stop: one inside its footprint does, or it crept on and
                // stopped by turns) - at the edge: outside my footprint and their body
                // (review-r5c-safety: at `half - 0.9` a red-light waiter standing 0.5-1.1 m
                // off my axis was crawled through at 0.25 m/s once the zebra check no longer
                // listed them)
                if !on_crossing && !at_kerb_of_stop && v < STANDING_SPEED && p.still > 3.0 && outside_here {
                    continue;
                }
                // (where the person really is relative to the sample, so that the stop
                // point does not jump with the sampling)
                // (somebody on a crossing my way cuts: short of the crossing, `plan::ZEBRA_STOP`
                // before the cut, as SUMO's vehicle does not take a crossing link a person
                // occupies; elsewhere 1.5 m short of them)
                // (a crossing whose zebra stop the front has passed: where the person is, as
                // anywhere else - the car does not drive on into them)
                let stop = match cut {
                    Some(z) if on_crossing && z.along - crate::plan::ZEBRA_STOP > st.front => z.along - crate::plan::ZEBRA_STOP,
                    _ => d + along as f32 - 1.5,
                };
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
        for (i, &(d, q, fwd)) in way.pts.iter().enumerate() {
            if d > reach + st.front {
                break;
            }
            let r = p.pos.truncate() - q;
            let right = DVec2::new(fwd.y, -fwd.x);
            let (back, ahead) = way.sample_span(i);
            if (-back..=ahead).contains(&r.dot(fwd)) && r.dot(right).abs() < half {
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
        let approach = match net.lanes.get(l) {
            Some(lane) if lane.source != 2 => {
                let first = js.nodes[j].movements[m].conflict_start;
                if first.is_finite() { (first - 1.0).max(0.0) } else { 0.0 }
            }
            _ => 0.0,
        };
        let line = d0 + approach;
        let inside = line <= 0.0;
        // the pieces my way takes through the object, and where their meeting places
        // begin and end in my way's distances (they may lie on the pieces after this one)
        let chain = js.chain_along(j, &way[k..]);
        let foes = js.foes_along(j, &chain);
        let way_len: f32 = chain.iter().map(|&(mm, _)| net.lanes[js.nodes[j].movements[mm].lane].length()).sum();
        // (the room past the conflicts is measured from the end of the last crossing or
        // joint; a stretch where the way runs side by side with another lane, `Foe::side`, is
        // a queueing lane, not a box to keep clear: counting it put the "box" of BeijingCity's
        // twin stub chains 60 m on, where the exit queue's tail closed it for 25 s with the
        // crossing itself clear, viz3 case03 #4436; the zipper of a side merge, [J7-2], keeps a
        // car off the one standing beside)
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
        return Some(JunctionView { junction: j, movement: m, entry: d0, line, inside, chain, conflict_start, conflict_end: clear_from, exit_room: room >= need, light_line, light, past_line, clearing, way_len });
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
    let mut behind_id: Option<usize> = None;
    let mut behind_min_gap = crate::driver::DriverParams::preset(crate::sim::PLAYER_PRESET).min_gap;
    let mut foreign = false;
    // (d_rel: the other's origin relative to the place beside me; front/rear its extents; the
    // vehicle's index, None for a player)
    let mut note = |d_rel: f32, front: f32, rear: f32, speed: f32, occ_foreign: bool, who: Option<usize>| {
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
                behind_id = who;
                if let Some(o) = who.and_then(|k| vehicles.get(k)) {
                    behind_min_gap = o.params.min_gap;
                }
            }
        }
    };
    let lane_users = |l: usize, offset: f32, note: &mut dyn FnMut(f32, f32, f32, f32, bool, Option<usize>)| {
        for occ in sim.occupancy.get(&l).map(|v| v.as_slice()).unwrap_or(&[]) {
            if occ.vehicle == i {
                continue;
            }
            let o = &vehicles[occ.vehicle];
            note(occ.s + offset - s_b, o.state.front, o.state.rear, o.state.speed, occ.foreign, Some(occ.vehicle));
        }
        // the players' vehicles are there like anybody else
        for &(k, s, _) in sim.player_occupancy.get(&l).map(|v| v.as_slice()).unwrap_or(&[]) {
            let p = &sim.players[k];
            note(s + offset - s_b, p.half_len, p.half_len, p.speed, false, None);
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
    Some(NeighbourView { lane: nb, dir, ahead, behind, behind_id, behind_min_gap, runs_on, turn, foreign, allowed })
}

/// [J7-3] Why my lane ends for me (`LaneEnd`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneEndKind {
    /// The way runs into a lane closed to me (all the ways on at its fork are: the fork's
    /// ranked choice took a closed one).
    Closed,
    /// The way ends (no lane on).
    DeadEnd,
    /// The way runs into a merge where it gives way to a way the lane beside me leads to.
    Merge,
}

/// [J7-3] Where my lane ends for me ahead (SUMO LC2013's dead lane / best lanes, sumo.md 3.2):
/// the distance from my front (to the closed lane's start, the way's end, or the start of the
/// merge stretch I give way at), why, and for a merge the lane beside me whose way the merge
/// joins (`toward`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaneEnd {
    pub dist: f32,
    pub kind: LaneEndKind,
    pub toward: Option<usize>,
}

/// `from` is `target`, or leads into it within a few lanes (`next`, every branch).
fn leads_to(net: &Network, from: usize, target: usize) -> bool {
    let mut at = vec![from];
    for _ in 0..4 {
        if at.contains(&target) {
            return true;
        }
        at = at.iter().flat_map(|&l| net.lanes[l].next.iter().copied()).collect();
    }
    at.contains(&target)
}

/// [J7-3] The first place along my way (`way`, within `look` m of my front) where my lane ends
/// for me (`LaneEnd`).
fn lane_end(net: &Network, sim: &Simulation, me: &Vehicle, way: &[(usize, f32)], look: f32) -> Option<LaneEnd> {
    let st = &me.state;
    let cur = net.lanes.get(st.lane)?;
    let beside = [cur.left, cur.right];
    // ([fix pass 2] the way as far as the strategic look-ahead, past the planned way where it does
    // not fork - SUMO's best lanes look far ahead: the planned way, eight seconds, ended on a
    // junction piece 170 m short of the closed lane, and the car found the end only on the piece,
    // where it has no neighbour)
    let mut way: Vec<(usize, f32)> = way.to_vec();
    while let Some(&(l, d0)) = way.last() {
        let lane = &net.lanes[l];
        let next = d0 + lane.length();
        if next - st.front > look || lane.next.len() != 1 || way.len() > 64 {
            break;
        }
        way.push((lane.next[0], next));
    }
    for (k, &(l, d0)) in way.iter().enumerate() {
        if d0 - st.front > look {
            break;
        }
        let lane = &net.lanes[l];
        // (a junction piece has no rules of its own: the way through is the junction's)
        if k > 0 && lane.source != 2 && !crate::access::is_open(lane, me) {
            return Some(LaneEnd { dist: d0 - st.front, kind: LaneEndKind::Closed, toward: None });
        }
        if let Some(&(j, m)) = sim.junctions.of_lane.get(&l) {
            let node = &sim.junctions.nodes[j];
            for f in &node.movements[m].foes {
                if f.kind != FoeKind::Merge || !f.yields || f.side {
                    continue;
                }
                let at = d0 + f.mine.0 - st.front;
                if !(0.0..=look).contains(&at) {
                    continue;
                }
                let fl = node.movements[f.movement].lane;
                if let Some(nb) = beside.iter().flatten().copied().find(|&nb| leads_to(net, nb, fl)) {
                    return Some(LaneEnd { dist: at, kind: LaneEndKind::Merge, toward: Some(nb) });
                }
            }
        }
        if k + 1 == way.len() && lane.next.is_empty() {
            let end = d0 + lane.length() - st.front;
            if end <= look {
                return Some(LaneEnd { dist: end, kind: LaneEndKind::DeadEnd, toward: None });
            }
        }
    }
    None
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
    // (pos, speed, accel, seconds until the red light it stands at lets it go)
    let mut note = |pos: f32, speed: f32, accel: f32, red_for: Option<f32>| {
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
        // a car standing at a red light: counts from when it goes ([round 7 A9] the light's
        // own `until_go`, not a flat 5 s; never within a cycle: not coming)
        if let (Some(wait), true) = (red_for, speed < STANDING_SPEED) {
            t += wait;
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
            let red_for = (o.last.why == crate::plan::Why::Light).then(|| signal_ahead(net, sim, &way_lanes(net, &o.state, 160.0), o.state.rear).and_then(|s| s.until_go).unwrap_or(f32::INFINITY));
            note(occ.s + off, o.state.speed, o.state.accel, red_for);
        }
        for &(k, s, _) in sim.player_occupancy.get(&l).map(|v| v.as_slice()).unwrap_or(&[]) {
            let p = &sim.players[k];
            note(s + off, p.speed.max(0.0), sim.player_accel(p.id), None);
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

    if air {
        p.leader = leader_on_way(sim, vehicles, i, &p.way, look);
    } else {
        // the way ahead, sampled once for the bodies, the players and the people
        let body_look = (st.speed * st.speed / (2.0 * st.decel.max(0.5)) + 2.0 * st.speed + 12.0).clamp(12.0, 70.0);
        let people_reach = people_reach(st.speed) + st.front;
        let player_near = sim.players.iter().any(|pl| (pl.pos - me.pos).truncate().length() < 180.0);
        let (cand, far) = body_candidates(sim, vehicles, i, &p.way, look, body_look);
        let upto = body_look.max(people_reach).max(far).max(if player_near { look } else { 0.0 });
        let way = WayPts::sample(net, st, upto);
        let (on_lane, body) = lead_by_bodies(net, sim, vehicles, i, &p.way, &way, look, body_look, cand, &mut p.ahead);
        p.leader = on_lane;
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
        p.parked_swerve = swerve;
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
    // [J7-3] (within the strategic look-ahead towards the overtaking side, the longer one)
    if lane.kind == LaneKind::Street && !me.is_rail() && (lane.left.is_some() || lane.right.is_some()) {
        let look = crate::plan::lane_end_look(me.look_ahead_speed.max(st.speed), true, me);
        p.lane_end = lane_end(net, sim, me, &p.way, look);
    }
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

#[cfg(test)]
mod crossing_tests {
    use super::ZebraCut;
    use crate::sim::Person;
    use glam::{DVec2, DVec3};
    use omsi_sim::traffic::{LaneBuilder, LaneKind, Network};

    /// [P8-5] SUMO's `blockedAtDist` on a crossing walked along x (3 m wide) that cuts my lane
    /// (heading north) at the origin: a car 0.9 m half-wide, band 1.4 m half-wide.
    #[test]
    fn a_walker_on_the_crossing_holds_the_car_until_past_its_band() {
        let net = Network { lanes: vec![LaneBuilder::polyline(vec![DVec3::new(-12.0, 0.0, 0.0), DVec3::new(12.0, 0.0, 0.0)], LaneKind::Sidewalk, 3.0)], ..Default::default() };
        let z = ZebraCut { walk: 0, along: 20.0, at: DVec2::ZERO, right: DVec2::X, half: 2.5 };
        let p = |x: f64, vx: f64, on: bool, still: f32| Person { pos: DVec3::new(x, 0.0, 0.0), vel: DVec2::new(vx, 0.0), on_crossing: on, still, waiting_for_bus: false, at_kerb: false };
        // coming in from 4 m at 1.8 m/s: held (2.3 m from the band, within 10 m)
        assert!(z.blocked_by(&net, &p(4.0, -1.8, true, 0.0), 0.9, 0.0));
        // the same walking away, past the band: not
        assert!(!z.blocked_by(&net, &p(-4.0, -1.8, true, 0.0), 0.9, 0.0));
        // inside the band, standing for long: held
        assert!(z.blocked_by(&net, &p(0.5, 0.0, true, 30.0), 0.9, 0.0));
        // standing 4 m off for 3 s: not (SUMO: standing walkers hold only from inside)
        assert!(!z.blocked_by(&net, &p(4.0, -0.4, true, 3.0), 0.9, 0.0));
        // coming in but not yet on the crossing (on the pavement over the walk's end): not
        assert!(!z.blocked_by(&net, &p(9.0, -1.4, false, 0.0), 0.9, 0.0));
        // off the walk altogether: not
        assert!(!z.blocked_by(&net, &p(0.0, 0.0, true, 0.0).with_y(5.0), 0.9, 0.0));
    }

    trait WithY {
        fn with_y(self, y: f64) -> Self;
    }
    impl WithY for Person {
        fn with_y(mut self, y: f64) -> Self {
            self.pos.y = y;
            self
        }
    }
}
