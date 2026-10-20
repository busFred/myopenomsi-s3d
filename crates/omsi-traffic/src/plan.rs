//! The decision: from what the driver sees to one plan for this frame. A plan is the
//! lead to follow, a stop point, a lateral target and at most one manoeuvre to start.
//! No plan reads another vehicle's plan; the only cross-vehicle input is the junction
//! grant.

use crate::driver::{DriverParams, VehicleClass, STANDING_SPEED};
use crate::perception::{JunctionView, NeighbourView, Perception, Seen, SignalView};
use crate::reservation::{Grant, Hold};
use crate::sim::{Passing, Simulation, Vehicle};
use glam::DVec2;
use omsi_sim::collision::Obb;
use omsi_sim::traffic::{arrival_time, Aspect, Lead, Network};

/// Why the car holds back, the nearest reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum Why {
    #[default]
    None,
    Lead,
    Light,
    Junction,
    ExitFull,
    People,
    /// A pedestrian at a crossing (round 4's zebra check; [round 7 A3] not produced any more:
    /// `People` holds for them. Kept for the recordings that name it.)
    Zebra,
    Service,
    Merge,
    PullOut,
    LetOut,
    End,
}

impl Why {
    pub fn name(self) -> &'static str {
        match self {
            Why::None => "",
            Why::Lead => "lead",
            Why::Light => "light",
            Why::Junction => "yield",
            Why::ExitFull => "exit",
            Why::People => "people",
            Why::Zebra => "zebra",
            Why::Service => "service",
            Why::Merge => "merge",
            Why::PullOut => "pull_out",
            Why::LetOut => "let_out",
            Why::End => "end",
        }
    }
}

/// A manoeuvre to begin this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Start {
    Change { to: usize, dir: i32 },
    /// Round something standing closer than the usual 8 m move: over `out_len` m.
    Bypass { to: usize, dir: i32, out_len: f32 },
    RouteChange { to: usize, dir: i32 },
    Pass(Passing),
    AbortPass { hold: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Plan {
    pub lead: Option<Lead>,
    pub lead_seen: Option<Seen>,
    /// Distance from the origin to where the car stops.
    pub stop: Option<f32>,
    pub why: Why,
    pub lateral: Option<f32>,
    pub start: Option<Start>,
    pub accel_cap: Option<f32>,
    /// Held by something (the start reaction applies when it clears).
    pub held: bool,
    pub hold: Option<Hold>,
    /// The claim index of the vehicle waited for at a junction (a player: `sim::player_claim`).
    pub wait_for: Option<usize>,
    /// Id of the vehicle whose body across the way holds this one.
    pub body_block: Option<u64>,
    /// The decision at the light ahead taken at its amber onset (`light_call`): (light,
    /// through, the gap from the front to its line).
    pub light_decision: Option<((usize, usize), bool, f32)>,
}

fn min_stop(stop: &mut Option<f32>, why: &mut Why, d: Option<f32>, w: Why) {
    if let Some(d) = d {
        if stop.map(|s| d < s).unwrap_or(true) {
            *stop = Some(d);
            *why = w;
        }
    }
}

/// Is the gap in the lane beside accepted: the follower there must not have to brake
/// hard, and there is room to settle ahead.
fn gap_accepted(me: &Vehicle, nb: &NeighbourView) -> bool {
    let (st, dp) = (&me.state, &me.params);
    let v = st.speed;
    let ok_behind = nb.behind.map(|(g, vb)| g > 2.0 + me.impatience.gap(dp.lc_gap) * vb + (vb - v).max(0.0).powi(2) / (2.0 * dp.comfort_brake)).unwrap_or(true);
    let ok_ahead = nb.ahead.map(|(g, va)| g > 2.0 + (v - va).max(0.0) * 1.5).unwrap_or(true);
    ok_behind && ok_ahead && !nb.foreign
}

/// The lane-change desire towards a neighbour, 0..1, and whether the gap there is
/// accepted (LMRS: the route, a standing obstacle, speed gain, keeping to the side).
/// `near_junction`: within the last stretch before a junction, where the lines are solid
/// and only the turn lane and a standing obstacle are reasons to move over.
fn change_desire(net: &Network, sim: &Simulation, me: &Vehicle, p: &Perception, nb: &NeighbourView, near_junction: bool) -> (f32, bool) {
    let st = &me.state;
    let dp = &me.params;
    let lane = &net.lanes[st.lane];
    if !may_change_to(net, me, nb) {
        return (0.0, false);
    }
    let mut desire = 0.0f32;
    let my_turn = st.turn_ahead(net, TURN_LANE_KEEP).map(|t| t.0).unwrap_or(0);
    if lane.turn == 0 && my_turn != 0 && nb.turn == my_turn && lane.turn != my_turn {
        desire = desire.max(1.0 * dp.lc_strategic);
    }
    // behind something standing: round it
    if let Some(l) = p.leader {
        let standing = l.standing_obstacle() && l.gap < 25.0;
        if standing && me.stopped >= 4.0 && nb.ahead.map(|a| a.0 > l.gap + l.length + 6.0).unwrap_or(true) {
            desire = desire.max(0.8);
        }
        // a faster lane
        let slow = l.speed < 0.7 * p.v_want && l.gap < 45.0 && l.gap > 8.0 && l.speed < st.speed + 1.0;
        if slow && !near_junction && st.speed >= 4.0 && nb.ahead.map(|a| a.0 > 50.0).unwrap_or(true) && open_on(net, me, nb.lane, st.s) {
            desire = desire.max(dp.lc_speed_gain);
        }
    }
    // keep to the side when the road is clear (the target lane reaching `KEEP_SIDE_REACH` on,
    // and open to me along the move and on: `open_on`; the target piece's own run-on of 40 m
    // cut keep-right 21-34 %, 201 of 446 went onto pieces under 40 m of lanes cut into short
    // spline pieces, review r6l)
    let keep_side = if sim.left_hand { 1 } else { 2 };
    if nb.dir == keep_side && !near_junction && p.leader.map(|l| l.gap > 60.0).unwrap_or(true) && nb.ahead.map(|a| a.0 > 70.0).unwrap_or(true) && nb.runs_on > KEEP_SIDE_REACH && open_on(net, me, nb.lane, st.s) {
        desire = desire.max(dp.lc_keep_right);
    }
    (desire, gap_accepted(me, nb))
}

/// A turn lane is never left within this many metres (m) of its junction, nor one taken
/// that leads into a turn the driver does not take before it runs on this far.
const TURN_LANE_KEEP: f32 = 150.0;
/// A lane beside that runs on less than this (m, `NeighbourView::runs_on`) is not changed
/// onto.
const CHANGE_RUNS_ON: f32 = 40.0;

/// [round 7 D5] May the driver change onto `nb` of its own accord (a change for a reason of
/// its own, a bypass; `change_desire` and `could_bypass` ask the same): the path's rules
/// allow it there; no solid line (`[rule] overtaking_prohib`) on either lane, not on a
/// junction object's piece; a turn lane is never left within `TURN_LANE_KEEP` of its
/// junction, a turn ahead is taken from the lane that leads into it, and a turn lane is not
/// taken for a turn the driver does not take; and `nb` runs on `CHANGE_RUNS_ON` at least.
fn may_change_to(net: &Network, me: &Vehicle, nb: &NeighbourView) -> bool {
    let lane = &net.lanes[me.state.lane];
    if !nb.allowed || lane.no_overtaking || net.lanes[nb.lane].no_overtaking || lane.source == 2 || nb.runs_on < CHANGE_RUNS_ON {
        return false;
    }
    let my_turn = me.state.turn_ahead(net, TURN_LANE_KEEP).map(|t| t.0).unwrap_or(0);
    !((my_turn != 0 && nb.turn != my_turn) || (nb.turn != 0 && my_turn == 0 && nb.runs_on < TURN_LANE_KEEP))
}

/// A keep-to-the-side change needs the target lane to reach this far on (m,
/// `NeighbourView::runs_on`: round 5's rule) ...
pub const KEEP_SIDE_REACH: f32 = 200.0;
/// ... and the lanes it runs into (one after the other, while each has a single next) open
/// to the car for the move's length and this many metres more (m: what a driver sees of the
/// lane markings ahead; viz5 case04 #5644 changed onto 3470's bus lane, which runs into
/// 3474's, closed to it).
pub const KEEP_SIDE_OPEN_AHEAD: f32 = 60.0;

/// Is `lane` (from beside distance `s` of my lane) and the chain of lanes it runs into open
/// to me (`access::is_open`) for a lane change's length and `KEEP_SIDE_OPEN_AHEAD` more? The
/// chain ends at a fork or a dead end; a junction object's piece (no rules: source 2) is not
/// asked, the way through is the junction's.
fn open_on(net: &Network, me: &Vehicle, lane: usize, s: f32) -> bool {
    let st = &me.state;
    // (`AiState::start_change`'s move)
    let need = omsi_sim::traffic::change_length(st.speed) + KEEP_SIDE_OPEN_AHEAD;
    let mut cur = lane;
    let mut d = -net.beside_s(st.lane, lane, s);
    for _ in 0..16 {
        let l = &net.lanes[cur];
        if l.source != 2 && !crate::access::is_open(l, me) {
            return false;
        }
        d += l.length();
        if d >= need || l.next.len() != 1 {
            return true;
        }
        cur = l.next[0];
    }
    true
}

/// Metres a pass needs past the obstacle besides the car's own length and the way back in
/// (`obstacle_run`).
pub const PASS_ROOM: f32 = 4.0;
/// A pass goes round at most this many metres of things standing one behind the other.
pub const PASS_MAX_OBSTACLE: f32 = 30.0;
/// Seconds a pass may stand out there with its way back in blocked before it is given up.
pub const PASS_STRANDED: f32 = 8.0;

/// The obstacle a pass would go round: the standing `lead` and what stands (or crawls under
/// 2 m/s) less than `need` beyond it, one after the other, at most `PASS_MAX_OBSTACLE` long;
/// and whether there is `need` of free lane beyond that (`Perception::ahead`). (Its length, room.)
fn obstacle_run(p: &Perception, lead: &crate::perception::Headway, need: f32) -> (f32, bool) {
    let mut end = lead.gap + lead.length;
    for &(near, far, speed) in &p.ahead {
        if near <= lead.gap + 0.5 || far <= end {
            continue;
        }
        if speed >= 2.0 {
            continue;
        }
        if near - end >= need {
            break;
        }
        if far - lead.gap > PASS_MAX_OBSTACLE {
            return (end - lead.gap, false);
        }
        end = far;
    }
    // (and the road goes on that far)
    (end - lead.gap, p.road_left > end + need + 2.0)
}

/// The vehicle `id` passed (a player's: `PLAYER_ID_BASE` + its id): (a bus, indicating out
/// towards the traffic, moving). None when it is gone.
fn obstacle_bus(sim: &Simulation, vehicles: &[Vehicle], id: u64) -> Option<(bool, bool, bool)> {
    let out = sim.out_side();
    if id >= crate::sim::PLAYER_ID_BASE {
        let p = sim.players.iter().find(|p| p.id as u64 == id - crate::sim::PLAYER_ID_BASE)?;
        return Some((true, p.blinker as i32 == out, p.speed.abs() > 0.5));
    }
    let o = vehicles.iter().find(|o| o.id == id)?;
    Some((o.serves_stops(), o.blinker_shown.0 == out, o.state.speed > 0.5))
}

/// Within this many metres of a junction (from the front to where its object begins) a car
/// does not pull out to pass, nor keeps back to be able to.
pub const JUNCTION_KEEP: f32 = 40.0;
/// An in-lane squeeze is not kept `JUNCTION_KEEP` from a junction when what it squeezes past
/// stands within this many metres (Sobol's `sq_junction_m`: 25 m beyond the bus's tail).
pub const SQUEEZE_NEAR: f32 = 25.0;
/// No light or junction within this many metres past the thing an in-lane squeeze goes
/// round (Sobol's 25 m; the pass on the oncoming lane keeps 60 m behind the player).
pub const SQUEEZE_ZONE: f32 = 25.0;
/// A pass on the oncoming lane round the player keeps this far (m) from a light or a
/// junction past it; round anything else `SQUEEZE_ZONE`, as a squeeze.
pub const PASS_ZONE_PLAYER: f32 = 60.0;
/// Seconds the player's bus has stood before a car squeezes past it inside the lane
/// (Sobol's `player_squeeze` fired at 1.5 s; the oncoming-lane pass waits for 4 s and 3 s
/// stopped itself).
pub const PLAYER_SQUEEZE_STAND: f32 = 1.5;
/// A squeeze past the player's bus starts once the car has slowed to this (m/s): closing
/// up behind it, not swerving out at speed 40 m back.
pub const SQUEEZE_START_SPEED: f32 = 3.0;
/// Seconds of indicator before an in-lane squeeze pulls out (round 5 L2: Sobol's squeeze
/// showed none; a real pass signals `SIGNAL_BEFORE_CHANGE`).
pub const SQUEEZE_SIGNAL: f32 = 0.5;
/// Metres the driver-side wheels may go over the line into a free neighbour lane when the
/// drawn road beside the obstacle is not quite enough for the squeeze ("the driver side
/// wheels step into the left neighbouring lane while most of the car remains in the
/// current lane"; round 5 L2).
pub const WHEELS_OVER_LINE: f32 = 0.6;

/// The clearance (m) a squeeze's bent corridor keeps from every body beside the lane - a car
/// standing on the strip, a bus in its bay, a parked car, the thing passed itself where it
/// stands askew: a squeeze does not start into one, and the thing passed holds the car until
/// the bent way really clears it (the BeijingCity game run of round 5 L: squeezes on the
/// AndingR strip ran cars into each other).
pub const SQUEEZE_CLEARANCE: f32 = 0.3;

/// A person's body radius (m), for the squeeze corridor (`humans::PERSON_RADIUS` is the
/// app's; the traffic model has no people geometry of its own).
pub const PERSON_RADIUS: f32 = 0.28;

/// The thing passed must have settled sideways: not itself out on a pass (stepping back from
/// an aborted squeeze, its lateral still moving), and its lateral where it means to be.
fn lateral_settled(vehicles: &[Vehicle], l: &crate::perception::Headway) -> bool {
    match l.what {
        // (standing, its offset cannot change: the ramps run by the odometer)
        Seen::Vehicle(j) | Seen::Body(j) => vehicles.get(j).is_none_or(|o| o.passing.is_none() && (o.state.speed < STANDING_SPEED || (o.state.lateral - o.state.lateral_target).abs() <= 0.05)),
        Seen::Player(_) | Seen::Parked => true,
    }
}

/// My body on my way `d` metres from the origin, shifted `side` metres sideways.
fn shifted_body(net: &Network, me: &Vehicle, d: f32, side: f32) -> Obb {
    let st = &me.state;
    let p = st.way_point(net, d).truncate();
    let p2 = st.way_point(net, d + 1.0).truncate();
    let f = (p2 - p).normalize_or_zero();
    let hr = (me.heading as f64).to_radians();
    let f = if f.length_squared() < 0.5 { DVec2::new(hr.sin(), hr.cos()) } else { f };
    let h = f.x.atan2(f.y).to_degrees();
    let right = DVec2::new(f.y, -f.x);
    Obb::vehicle(p + right * side as f64, h, st.front as f64, st.rear as f64, me.half_width as f64)
}

/// Is the corridor of a squeeze clear of every other body (vehicles, players, parked cars,
/// people) by `SQUEEZE_CLEARANCE`: from where the car stands, pulling `side` metres out over
/// the first `out_len` metres of way (the bent way starts at the car, `Passing::curve`),
/// level with the obstacle to `level` metres ahead of the origin, and easing back to the axis
/// over `back` metres more? (Review r5l safety, pass 3: sampled from `real - out_len` only,
/// a car standing level or just ahead on the squeeze side was missed; BJ #3878 yawed into
/// #4203 standing 0.2 m beside it.)
fn squeeze_clear(net: &Network, sim: &Simulation, vehicles: &[Vehicle], me: &Vehicle, way: &[(usize, f32)], side: f32, out_len: f32, level: f32, back: f32) -> bool {
    let to = level + back;
    let from = 0.0f32;
    let near = sim.vehicles_near(me.pos, (to + 30.0) as f64);
    let clearance = SQUEEZE_CLEARANCE as f64;
    let mut parked: Vec<Obb> = Vec::new();
    for &(l, d0) in way {
        if d0 > to + 5.0 {
            break;
        }
        let Some(lane) = net.lanes.get(l) else { continue };
        for pk in sim.parked.get(&l).map(|v| v.as_slice()).unwrap_or(&[]) {
            if d0 + pk.s < from - 15.0 || d0 + pk.s > to + 15.0 {
                continue;
            }
            let (q, h) = lane.at(pk.s.min(lane.length()));
            let hr = (h as f64).to_radians();
            let right = DVec2::new(hr.cos(), -hr.sin());
            parked.push(Obb::vehicle(q.truncate() + right * pk.lateral as f64, h as f64, pk.half_len as f64, pk.half_len as f64, pk.half_width as f64));
        }
    }
    let people = sim.people_near(me.pos, (to + 15.0) as f64);
    let mut d = from;
    loop {
        let side_d = if d < out_len {
            side * (d / out_len.max(0.1))
        } else if d <= level {
            side
        } else {
            side * (1.0 - (d - level) / back.max(0.1)).max(0.0)
        };
        let b = shifted_body(net, me, d, side_d);
        // (nobody on foot in the corridor: a squeeze stopped for people mid-ramp stands half
        // out, its body in the car it passes - the BeijingCity 08:16 pair #376/#450)
        for &k in &people {
            let p = &sim.people[k];
            if (p.pos.z - me.pos.z).abs() <= 2.5 && Obb::point(p.pos, PERSON_RADIUS as f64).separation(&b) < clearance {
                return false;
            }
        }
        for &j in &near {
            let o = &vehicles[j];
            if o.id == me.id || (o.pos.z - me.pos.z).abs() > 3.0 {
                continue;
            }
            if o.obb().separation(&b) < clearance {
                return false;
            }
        }
        if sim.players.iter().any(|p| (p.pos.z - me.pos.z).abs() <= 3.0 && p.obb(0.0, 0.0).separation(&b) < clearance) {
            return false;
        }
        if parked.iter().any(|o| o.separation(&b) < clearance) {
            return false;
        }
        if d >= to {
            return true;
        }
        d = (d + 2.0).min(to);
        if d >= to && d - 2.0 >= to {
            return true;
        }
    }
}

/// Does my body on the bent way, `side` out, really clear the thing passed, `l` (by
/// `SQUEEZE_CLEARANCE`)? A parked car: by the lateral arithmetic alone.
fn shift_clears(net: &Network, sim: &Simulation, vehicles: &[Vehicle], me: &Vehicle, side: f32, l: &crate::perception::Headway) -> bool {
    let other = match l.what {
        Seen::Vehicle(j) | Seen::Body(j) => vehicles.get(j).map(|o| o.obb()),
        Seen::Player(id) => sim.players.iter().find(|p| p.id == id).map(|p| p.obb(0.0, 0.0)),
        Seen::Parked => return true,
    };
    let Some(other) = other else { return true };
    let d = l.gap + me.state.front + l.length * 0.5;
    let b = shifted_body(net, me, d, side);
    other.separation(&b) >= SQUEEZE_CLEARANCE as f64
}

/// The player's bus pausing in the lane: squeezed past inside the lane after
/// `PLAYER_SQUEEZE_STAND` (`Headway::standing_obstacle` waits 4 s for the pass on the
/// oncoming lane).
fn pausing_player(l: &crate::perception::Headway) -> bool {
    matches!(l.what, Seen::Player(_)) && l.standing_for > PLAYER_SQUEEZE_STAND
}

/// Could the car get round the standing `lead` into the lane beside, `nb` (`Start::Bypass`,
/// the lane-change rules of `change_desire` but for having stood a while): a lane it may take,
/// no solid line, not a junction's piece, a turn ahead that stays its own, room there past
/// the thing standing and nobody passing on it the wrong way. (What the keep-back asks.)
fn could_bypass(net: &Network, me: &Vehicle, nb: &NeighbourView, lead: &crate::perception::Headway) -> bool {
    if !may_change_to(net, me, nb) {
        return false;
    }
    // (the gap there is the start's to accept, `change_desire`, and a car driving on ahead
    // there leaves the room: one going by in that lane now does not make the car roll up to 2 m behind the obstacle, from where it can never
    // steer out - fix pass 2, the queue behind a bus at its stop)
    nb.ahead.map(|a| a.1 > QUEUE_SPEED * 2.0 || a.0 > lead.gap + lead.length + 6.0).unwrap_or(true) && !nb.foreign
}

/// The move of a bypass from rest (m, `AiState::start_bypass`'s): a short, steep move.
pub const BYPASS_LENGTH: f32 = 8.0;
/// What the move keeps short of the obstacle's rear (m): the pass's `gap - 0.3`.
pub const BYPASS_GAP_MARGIN: f32 = 0.3;
/// A car's wheelbase as a share of its length (a 4.5 m saloon's 2.7 m; a 12 m bus's 6-7 m).
pub const WHEELBASE_SHARE: f32 = 0.6;
/// The largest road-wheel angle a driver steers when pulling out from rest (degrees; a
/// passenger car's full lock is about 35-40, its turning circle about 10.5-11 m).
pub const MAX_STEER_DEG: f32 = 35.0;

/// The shortest move over `side` metres the body can drive (m): the S-curve's tightest
/// bends, at its ends, have a radius of length² / (6 |side|), no tighter than the turning
/// radius `wheelbase / tan(MAX_STEER_DEG)`. (Round 6 L2 fix pass 1: a 2 m move over 3.5 m
/// pivoted the car 40-70 degrees on the spot, viz5 case08 #45474 178 -> 138 degrees in
/// 0.5 m; for a 4.6 m car over 3.5 m this is 9.1 m, its steepest yaw 30 degrees.)
pub fn bypass_min_len(length: f32, side: f32) -> f32 {
    let radius = WHEELBASE_SHARE * length / MAX_STEER_DEG.to_radians().tan();
    (6.0 * side.abs() * radius).sqrt()
}

/// The bypass round the standing `lead` into `nb`, from rest: the move ends `BYPASS_GAP_MARGIN`
/// short of its rear, no longer than `BYPASS_LENGTH` and no shorter than the body can steer
/// (`bypass_min_len`), and it starts only when that move clears the obstacle and every other
/// body: `squeeze_clear` (a car standing in the lane beside level or just ahead, people)
/// and `bypass_sweep_clear` (the body turned along the curve, the obstacle's corner too);
/// where the gap is too short for it, the car does not set out (the 8 m move from 1.5 m
/// behind a bus ran its way into the bus's corner and froze six times, viz5 case08 #45474;
/// a car does not reverse out). Not round a bus indicating out of its stop or a vehicle
/// moving off (it is let out).
fn bypass_start(net: &Network, sim: &Simulation, vehicles: &[Vehicle], me: &Vehicle, p: &Perception, nb: &NeighbourView, lead: &crate::perception::Headway) -> Option<Start> {
    let st = &me.state;
    let obstacle = match lead.what {
        Seen::Vehicle(j) | Seen::Body(j) => vehicles.get(j).map(|o| o.id),
        Seen::Player(id) => Some(crate::sim::PLAYER_ID_BASE + id as u64),
        Seen::Parked => None,
    };
    if obstacle.and_then(|ob| obstacle_bus(sim, vehicles, ob)).is_some_and(|(bus, signals, moves)| moves || (bus && signals)) {
        return None;
    }
    let target = &net.lanes[nb.lane];
    let (ts, _) = target.nearest_point(me.pos)?;
    let hr = (me.heading as f64).to_radians();
    let right = DVec2::new(hr.cos(), -hr.sin());
    let side = (target.at(ts).0 - me.pos).truncate().dot(right) as f32;
    let shortest = bypass_min_len(me.length(), side);
    let out_len = (lead.gap - BYPASS_GAP_MARGIN).clamp(shortest.min(BYPASS_LENGTH), BYPASS_LENGTH).max(shortest);
    let (obstacle_len, _) = obstacle_run(p, lead, me.length() + out_len + PASS_ROOM);
    let level = lead.gap + obstacle_len + st.front + st.rear + 2.0;
    (squeeze_clear(net, sim, vehicles, me, &p.way, side, out_len, level, 0.0) && bypass_sweep_clear(net, sim, vehicles, me, side, out_len, level)).then_some(Start::Bypass { to: nb.lane, dir: nb.dir, out_len })
}

/// The gap (m) behind the standing `lead` from which a bypass into `nb` clears it as the body
/// can steer (`bypass_min_len`): the S-curve has moved the body off the obstacle by both half
/// widths and `SQUEEZE_CLEARANCE`, and `BYPASS_TURN_MARGIN` for the corner of a turned body,
/// by the time the front is level with its rear. A car that closes up nearer cannot get out
/// without reversing (round 6 L2 fix pass 1: cars rolled up to 1.9 m behind and never could).
fn bypass_room(net: &Network, me: &Vehicle, nb: &NeighbourView, lead: &crate::perception::Headway) -> Option<f32> {
    let target = &net.lanes[nb.lane];
    let (ts, _) = target.nearest_point(me.pos)?;
    let hr = (me.heading as f64).to_radians();
    let right = DVec2::new(hr.cos(), -hr.sin());
    let side = ((target.at(ts).0 - me.pos).truncate().dot(right) as f32).abs();
    let need = me.half_width + lead.half_width + SQUEEZE_CLEARANCE + BYPASS_TURN_MARGIN;
    if side <= need {
        return None;
    }
    // (smoothstep inverted by halving: the share of the move at which it is `need` across)
    let frac = need / side;
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    for _ in 0..24 {
        let t = 0.5 * (lo + hi);
        if t * t * (3.0 - 2.0 * t) < frac {
            lo = t;
        } else {
            hi = t;
        }
    }
    Some(hi * bypass_min_len(me.length(), side))
}

/// A car closing in on the room a bypass needs holds there from this far inside it (m: the
/// stop lands within half a metre of where it was aimed).
pub const BYPASS_HOLD_SLACK: f32 = 0.5;

/// Seconds a vehicle at the head of a queue stands, held by nothing, before the queue counts
/// as one behind an obstacle; `Headway::standing_obstacle` takes one standing unheld, or the player, as one after it.
pub const QUEUE_OBSTACLE_STAND: f32 = 4.0;
/// Vehicles of a queue looked along for its head (`behind_an_obstacle`).
pub const QUEUE_LOOK: usize = 6;

/// Is the standing `lead` the tail of a queue whose head is an obstacle - a bus held at its
/// stop (`External::at_stop`, `Why::Service`), a vehicle at the end of its way, or one that
/// stands `QUEUE_OBSTACLE_STAND` s held by nothing - rather than a queue that is the road (a
/// red light, a junction, a zebra: their heads are held by them)? The car approaching it is
/// then kept back at the room a bypass needs before the queue packs to 2 m (round 6 L2 fix
/// pass 2: behind bus #22368 at its stop, viz5 case07, four cars closed up to 1.8-2.2 m and
/// stood 60 s and more, none able to steer out).
fn behind_an_obstacle(vehicles: &[Vehicle], lead: &crate::perception::Headway) -> bool {
    if lead.speed >= STANDING_SPEED {
        return false;
    }
    let (Seen::Vehicle(mut j) | Seen::Body(mut j)) = lead.what else { return false };
    for _ in 0..QUEUE_LOOK {
        let Some(o) = vehicles.get(j) else { return false };
        if o.state.speed >= STANDING_SPEED || o.passing.is_some() || o.state.change.is_some() {
            return false;
        }
        let why = o.last.why;
        if o.external.at_stop || matches!(why, Why::Service | Why::End) || (!o.last.held && why != Why::Lead && o.stopped >= QUEUE_OBSTACLE_STAND) {
            return o.blinker_shown.0 == 0;
        }
        if why != Why::Lead {
            return false;
        }
        let Some(next) = o.last.lead_id.and_then(|id| vehicles.iter().position(|v| v.id == id)) else { return false };
        j = next;
    }
    false
}

/// What the corner of a body turned on the bypass's curve reaches beyond its shifted box (m):
/// half a metre at 30 degrees for a 2 m front overhang.
pub const BYPASS_TURN_MARGIN: f32 = 0.5;

/// Does the body clear every other body by `SQUEEZE_CLEARANCE` on the bypass's S-curve
/// (`AiState::start_bypass`'s move: `side` metres over `out_len`, the body turned along the
/// curve, not only shifted as `squeeze_clear` has it), up to `level` metres on?
fn bypass_sweep_clear(net: &Network, sim: &Simulation, vehicles: &[Vehicle], me: &Vehicle, side: f32, out_len: f32, level: f32) -> bool {
    let st = &me.state;
    let near = sim.vehicles_near(me.pos, (level + 20.0) as f64);
    let clearance = SQUEEZE_CLEARANCE as f64;
    let mut d = 0.0f32;
    while d <= level {
        let t = (d / out_len.max(0.1)).clamp(0.0, 1.0);
        let lat = side * t * t * (3.0 - 2.0 * t);
        let slope = side * 6.0 * t * (1.0 - t) / out_len.max(0.1);
        let p = st.way_point(net, d).truncate();
        let p2 = st.way_point(net, d + 1.0).truncate();
        let f = (p2 - p).normalize_or_zero();
        let hr = (me.heading as f64).to_radians();
        let f = if f.length_squared() < 0.5 { DVec2::new(hr.sin(), hr.cos()) } else { f };
        let right = DVec2::new(f.y, -f.x);
        let dir = (f + right * slope as f64).normalize();
        let b = Obb::vehicle(p + right * lat as f64, dir.x.atan2(dir.y).to_degrees(), st.front as f64, st.rear as f64, me.half_width as f64);
        for &j in &near {
            let o = &vehicles[j];
            if o.id != me.id && (o.pos.z - me.pos.z).abs() <= 3.0 && o.obb().separation(&b) < clearance {
                return false;
            }
        }
        if sim.players.iter().any(|pl| (pl.pos.z - me.pos.z).abs() <= 3.0 && pl.obb(0.0, 0.0).separation(&b) < clearance) {
            return false;
        }
        d += 0.5;
    }
    true
}

/// What the driver does about the light ahead this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightCall {
    /// Over the line now (no stop for the light).
    pub go: bool,
    /// The decision taken at this light's amber onset - (light, through, the gap from the
    /// front to its line) - kept until the light is no longer the one ahead or turns green
    /// (`Vehicle::light_decision`).
    pub decision: Option<((usize, usize), bool, f32)>,
}

/// [round 7 D2] The braking (m/s²) a driver reckons with when asking whether it can still
/// stop before a line or a meeting place: hard, short of an emergency stop (`MAX_BRAKE`).
/// One figure for the light (`light_call`), the claim's committed and clearing tests and the
/// player's red (before: 0.8 of `MAX_BRAKE` at the light, 0.7 at the junction).
pub const HARD_BRAKE: f32 = omsi_sim::traffic::MAX_BRAKE * 0.8;

/// [round 7 D2] Too late to stop within `room` metres at `v` m/s braking `brake` (m/s²;
/// `HARD_BRAKE` but inside a junction); under 1 m/s a car can always stop.
pub fn cannot_stop(v: f32, room: f32, brake: f32) -> bool {
    v > 1.0 && room < v * v / (2.0 * brake)
}

/// [fix pass 3, 1] The braking (m/s²) a driver always accepts to stop at an amber line,
/// SUMO's `--tls.yellow.min-decel` (3 m/s², sumo.md §7: "the deceleration needed is compared
/// against" it).
pub const YELLOW_MIN_DECEL: f32 = 3.0;

/// [fix pass 2, 4; pass 3, 1] Can the driver stop within `gap` (m) at `v` (m/s) on amber -
/// SUMO's yellow: the vehicle stops when v²/2gap is within its decel or `YELLOW_MIN_DECEL`,
/// whichever is more? Standing it can. (Its comfortable brake alone, 2 m/s², committed a car
/// 56 m out at 14.86 m/s, needing 1.95, that crossed 1.8 s into red: viz case17 #60501.)
pub fn stops_comfortably(v: f32, gap: f32, dp: &DriverParams) -> bool {
    v <= STANDING_SPEED || (gap > 0.0 && v * v / (2.0 * gap) <= dp.comfort_brake.max(YELLOW_MIN_DECEL))
}

/// [round 7 fix pass 1] The driver's call at the light, SUMO's yellow and red rule: through
/// on green; at amber onset the driver decides once whether it can stop with comfortable
/// braking (`comfortable_stop`) - if not it is committed and goes, if so it stops - and keeps
/// that decision while this light is the one ahead (`Vehicle::light_decision`); on red only a
/// car that cannot stop at all (`cannot_stop`, `HARD_BRAKE`) or one committed at amber onset
/// and still rolling crosses. Nothing is decided on green, no decision is extended past
/// what the onset allowed, and a committed car that came to a stand waits for its green
/// (round 6 J3: one standing short of its line set off over it on red, viz case07 #18558).
/// (This replaces round 4's amber mark: taken on green 3 m out, extended 6 m on yellow,
/// carried over frames and retired by `stale_amber` and a 2 s stand; x10 case03 #3145
/// crossed its line on red on it.)
pub fn light_call(me: &Vehicle, p: &Perception, sig: &SignalView) -> LightCall {
    let st = &me.state;
    let dp = &me.params;
    let v = st.speed;
    let gap = sig.dist - st.front;
    let queued = v < 1.0 && p.leader.map(|l| l.gap < 8.0).unwrap_or(false);
    let light = (sig.ctl, sig.light);
    // (that light's line: the same light lights several approaches, viz3 case01's 1.0 three)
    let line = st.odometer + gap;
    let held = me.light_decision.filter(|d| d.0 == light && (d.2 - line).abs() < 2.0).map(|d| d.1);
    match sig.aspect {
        Aspect::Green | Aspect::GreenYellow | Aspect::Dark => LightCall { go: true, decision: None },
        Aspect::Yellow => {
            // (and never from a standstill or a queue: at rest nothing is "too late to stop";
            // [fix pass 2, 4] nor into a box whose exit is not free - SUMO's no-block check,
            // sumo.md: viz5 case05 #16535 took its amber with people 2.8 m past the line and
            // stood 85.8 s inside - unless it cannot stop at all)
            let exit_free = p.junction.as_ref().is_none_or(|j| j.exit_room);
            let go = held.unwrap_or_else(|| v > STANDING_SPEED && !queued && !stops_comfortably(v, gap, dp) && (exit_free || cannot_stop(v, gap + 0.5, HARD_BRAKE)));
            LightCall { go: go && v > STANDING_SPEED, decision: Some((light, go, gap)) }
        }
        Aspect::Red | Aspect::RedYellow => {
            let committed = held == Some(true) && v > STANDING_SPEED;
            LightCall { go: committed || cannot_stop(v, gap + 0.5, HARD_BRAKE), decision: held.map(|h| (light, h, gap)) }
        }
    }
}

/// The zebra band a car keeps its body out of: its front ZEBRA_STOP m before a pedestrian
/// crossing's cut of its way (the zebra stop) to ZEBRA_PAST m past it (the crossing's width
/// and a step; `hold_point`'s band since round 4). (Somebody on the crossing is held for by
/// `perception::people_ahead`, the one people test since round 7 A3.)
pub const ZEBRA_STOP: f32 = 2.5;
pub const ZEBRA_PAST: f32 = 2.0;

/// A lead slower than this (m/s) is a queue I come to rest behind, not one I follow through
/// (`light_call`'s `queued`: under 1 m/s behind a lead).
const QUEUE_SPEED: f32 = 1.0;
/// A lead under this speed (m/s) that is itself held (a lead, a light, a junction) is coming
/// to rest in a queue too, and so is one braking for a stop of its own at any speed
/// (`Headway::rest`; x10 case15: the lead rolled at 1.4-3 m/s past the cut to its exit-full
/// line while I was a car length short of it, and stood the moment I was on the zebra).
const QUEUE_COMING: f32 = 3.0;
/// The brake (m/s²) such a lead is taken to come to rest with, for where its rear will stand
/// (a comfortable brake, `AiState::decel`'s 3 m/s² default is its hardest liking).
const QUEUE_BRAKE: f32 = 2.0;

/// Where a car refused for a foe stops (rule D): its front a metre before the first stretch
/// it waits for (`waited`, along the way through), pulled back before any other meeting
/// place ahead or pedestrian crossing its body would stand in ("keep the junction clear"; a
/// meeting place it is in already it cannot leave but forward). A distance from the origin.
fn hold_point(net: &Network, sim: &Simulation, me: &Vehicle, j: &JunctionView, waited: f32) -> f32 {
    pulled_back(&hold_bands(net, sim, j), me, j.entry + waited - 1.0)
}

/// The bands of the way through a held car keeps its body out of (`hold_point`): the meeting
/// places (from a metre before), the pedestrian crossings and the convergences.
fn hold_bands(net: &Network, sim: &Simulation, j: &JunctionView) -> Vec<(f32, f32)> {
    let mut bands: Vec<(f32, f32)> = sim.junctions.foes_along(j.junction, &j.chain).iter().map(|f| (j.entry + f.mine.0 - 1.0, j.entry + f.mine.1)).collect();
    bands.extend(walk_bands(net, sim, j));
    bands.extend(converge_bands(sim, j).into_iter().map(|(a, b, _)| (a, b)));
    bands
}

/// [round6 J2] A car comes to rest up to this far (m) short of its stop (the controller's 0.6 m
/// `STOP_LINE_GAP` and a creep): `pulled_back` keeps that much more of the body out of a band
/// (review-r6j: the rear stood 0.6 m inside a convergence it was stopped behind).
const REST_SHORT: f32 = 1.0;

/// `stop` (a distance from the origin, the front's) pulled back before every band of
/// `bands` (from, to) the body would stand in, a band at a time.
fn pulled_back(bands: &[(f32, f32)], me: &Vehicle, mut stop: f32) -> f32 {
    let front = me.state.front;
    let len = me.length();
    for _ in 0..32 {
        let hit = bands.iter().filter(|b| b.0 > front - 0.3 && b.0 < stop && b.1 > stop - len - REST_SHORT).map(|b| b.0).fold(f32::INFINITY, f32::min);
        if !hit.is_finite() {
            break;
        }
        stop = hit;
    }
    stop
}

/// The pedestrian crossings cutting the way through, as bands the body may not stand in:
/// 2.5 m before the cut (the zebra stop) to 2 m past it, distances from the origin.
fn walk_bands(net: &Network, sim: &Simulation, j: &JunctionView) -> Vec<(f32, f32)> {
    let node = &sim.junctions.nodes[j.junction];
    let mut out = Vec::new();
    for &(m, off) in &j.chain {
        for &(_, sa, _) in net.walks.get(node.movements[m].lane).map(|v| v.as_slice()).unwrap_or(&[]) {
            let point = j.entry + off + sa;
            out.push((point - ZEBRA_STOP, point + ZEBRA_PAST));
        }
    }
    out
}

/// The stretches of the way through where it converges with a lane of the junction that is no
/// foe (`Movement::converge`), as (from a metre before, to, the movement converged with)
/// distances from the origin.
fn converge_bands(sim: &Simulation, j: &JunctionView) -> Vec<(f32, f32, usize)> {
    let node = &sim.junctions.nodes[j.junction];
    j.chain.iter().flat_map(|&(m, off)| node.movements[m].converge.iter().zip(&node.movements[m].converge_with).map(move |(&(a, b), &with)| (j.entry + off + a - 1.0, j.entry + off + b, with))).collect()
}

/// The lanes outside junction `j` from which movement `m` is reached: its own entries and
/// those of the inner pieces leading into it (a way through chains pieces, the later ones
/// have no entry of their own).
fn entry_roots(sim: &Simulation, j: usize, m: usize) -> Vec<usize> {
    let node = &sim.junctions.nodes[j];
    let mut roots: Vec<usize> = Vec::new();
    let mut seen = vec![m];
    let mut frontier = vec![m];
    for _ in 0..8 {
        let mut next = Vec::new();
        for &x in &frontier {
            for &e in &node.movements[x].entries {
                if !roots.contains(&e) {
                    roots.push(e);
                }
            }
            for (p, mv) in node.movements.iter().enumerate() {
                if mv.continues.contains(&x) && !seen.contains(&p) {
                    seen.push(p);
                    next.push(p);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    roots.sort_unstable();
    roots
}

/// [round6 J3] Under this speed (m/s) an eased-off car stands, and is pinned where it is
/// (`Claim::standing`'s 0.3 m/s).
const EASE_STANDING: f32 = 0.3;

/// A standing vehicle this far (m) off a piece of my way (its origin to the piece's centre
/// line) is on a lane running side by side with it, and its rear is projected on my way.
const BESIDE_MAX: f64 = 4.0;

/// [round6 J2] A convergence of the way through (`converge_bands`) beginning within this many
/// metres past the front (the line while short of it) and the gentle stopping distance stops a
/// refused car at once; one farther on it rolls up to (review-r6 fix 4: the coach of case01 froze
/// across the player's lane for a stretch 90 m on).
const CONVERGE_LOOK: f32 = 20.0;

/// [round6 J2] The body of the vehicle waited for (`Grant::wait_for`, the claims' index; a
/// player's `sim::player_claim`) projected on the way through: (rear, front), distances from my
/// origin, when its origin lies within `BESIDE_MAX` of a piece of the way.
fn waited_body_on_way(net: &Network, sim: &Simulation, vehicles: &[Vehicle], j: &JunctionView, w: usize) -> Option<(f32, f32)> {
    let (pos, rear, front) = if let Some(id) = crate::sim::claim_player(w) {
        let p = sim.players.iter().find(|p| p.id == id)?;
        (p.pos, p.half_len, p.half_len)
    } else {
        let o = vehicles.get(w)?;
        (o.pos, o.state.rear, o.state.front)
    };
    let node = &sim.junctions.nodes[j.junction];
    j.chain
        .iter()
        .filter_map(|&(m, off)| net.lanes.get(node.movements[m].lane)?.nearest_point(pos).filter(|&(_, d)| d < BESIDE_MAX).map(|(s, d)| (d, j.entry + off + s)))
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, s)| (s - rear, s + front))
}

/// A car refused for a side-merge stretch (`Foe::side`) that a vehicle stands held in rolls up
/// to it instead of standing at the line, when every band between the line and that stretch
/// is a side merge or a convergence with a piece of its own fork: a piece entered from the
/// same lanes as its own way (`entry_roots`), so that nobody who is not already behind it in
/// its lane can come in beside it there. Its stop: the lesser of a metre before the stretch
/// and its gap (`min_gap`) behind the rear of every standing vehicle on those pieces,
/// projected on its way; then pulled back before any crossing. None where the rule does not
/// apply: the line as before. (viz4 case02: twin 1 m stub chains forking from one lead-in
/// run side by side 39 m to the two exit lanes; a car waiting for the queue in the other
/// chain stood at the object's entry, 48 m behind the car it waited for, level with the
/// player's rear bumper, 57 s; viz3 case03: a stretch shared with turn pockets that have
/// their own entries keeps the line.)
fn roll_up_point(net: &Network, sim: &Simulation, vehicles: &[Vehicle], me: &Vehicle, j: &JunctionView, waited: f32) -> Option<f32> {
    let js = &sim.junctions;
    let node = &js.nodes[j.junction];
    let foes = js.foes_along_pieces(j.junction, &j.chain);
    let mine = entry_roots(sim, j.junction, j.movement);
    if mine.is_empty() {
        return None;
    }
    let own_fork = |m: usize| m != j.movement && !j.chain.iter().any(|&(c, _)| c == m) && entry_roots(sim, j.junction, m) == mine;
    // the stretch waited for: a side merge with a piece of my own fork, a vehicle standing in it
    let waited_side = foes.iter().any(|(_, f)| (f.mine.0 - waited).abs() < 0.05 && f.side && own_fork(f.movement));
    if !waited_side {
        return None;
    }
    let me_k = me_index(vehicles, me);
    let standing_in = |m: usize, from: f32, to: f32| -> bool {
        sim.claims.iter().any(|(c, _)| c.vehicle != me_k && c.speed < STANDING_SPEED && c.junction == j.junction && c.chain.iter().any(|&(cm, off)| cm == m && c.s - off + c.front >= from && c.s - off - c.rear <= to))
    };
    if !foes.iter().any(|(_, f)| (f.mine.0 - waited).abs() < 0.05 && f.side && standing_in(f.movement, f.theirs.0, f.theirs.1)) {
        return None;
    }
    // every band between the line and the stretch: a side merge or a convergence with my own fork
    let mut partners: Vec<usize> = Vec::new();
    for (_, f) in &foes {
        if f.mine.0 < waited - 0.05 && f.mine.1 > 0.0 {
            if !(f.side && own_fork(f.movement)) {
                return None;
            }
            partners.push(f.movement);
        } else if (f.mine.0 - waited).abs() < 0.05 {
            partners.push(f.movement);
        }
    }
    for (a, _, with) in converge_bands(sim, j) {
        if a < j.entry + waited - 1.0 {
            if !own_fork(with) {
                return None;
            }
            partners.push(with);
        }
    }
    // the stop: a metre before the stretch, and behind every standing vehicle on the
    // partner pieces projected on my way (the queue's tail, as a lead would be)
    let mut stop = j.entry + waited - 1.0;
    let rear_on_my_way = |pos: glam::DVec3, rear: f32| -> Option<f32> {
        j.chain.iter().filter_map(|&(m, off)| net.lanes.get(node.movements[m].lane)?.nearest_point(pos).filter(|&(_, d)| d < BESIDE_MAX).map(|(s, _)| j.entry + off + s - rear)).fold(None, |a: Option<f32>, b| Some(a.map_or(b, |a| a.min(b))))
    };
    for &m in &partners {
        let lane = node.movements[m].lane;
        for occ in sim.occupancy.get(&lane).map(|v| v.as_slice()).unwrap_or(&[]) {
            let o = &vehicles[occ.vehicle];
            if o.id == me.id || o.state.speed >= STANDING_SPEED {
                continue;
            }
            if let Some(r) = rear_on_my_way(o.pos, o.state.rear) {
                stop = stop.min(r - me.state.min_gap);
            }
        }
        for &(k, _, _) in sim.player_occupancy.get(&lane).map(|v| v.as_slice()).unwrap_or(&[]) {
            let p = &sim.players[k];
            if p.speed.abs() < STANDING_SPEED {
                if let Some(r) = rear_on_my_way(p.pos, p.half_len) {
                    stop = stop.min(r - me.state.min_gap);
                }
            }
        }
    }
    Some(pulled_back(&walk_bands(net, sim, j), me, stop))
}

/// The index of `me` among `vehicles` (the claims name vehicles by index).
fn me_index(vehicles: &[Vehicle], me: &Vehicle) -> usize {
    vehicles.iter().position(|v| v.id == me.id).unwrap_or(usize::MAX)
}

/// Decide for vehicle `i`.
/// [round 7 D3] Where a car its grant refuses stops (a distance from its origin): one ordered
/// list of cases, the first that applies.
///
/// 0. `first`: at the line, where it can still stop there gently or stands (`line_stop`),
///    else - already on the lead-in piece of a long object, or a line too near to stop at
///    gently, whose bumper is over it - a metre short of the first meeting place of the way
///    through (the claim counts it so; a line the bumper was over drew an emergency stop).
///    Every hold but one for a foe at a known stretch stops there.
/// 1. [round6 J1] held only by a foe standing for good far ahead (`Grant::roll_up`): up to
///    its stretch (`hold_point`).
/// 2. a side-merge stretch of my own fork that a vehicle stands in: up to it, behind that
///    vehicle (`roll_up_point`).
/// 3. refused for a foe: just before the first stretch it waits for (`hold_point`), not
///    short of the first meeting place of its way (viz3 case03: a car waiting for a stretch
///    22 m on stood 2.8 m in, on the fork shared with the turn pockets, and blocked the
///    turners behind it) - except where its way converges with another lane short of that
///    stretch that the junction knows no foe on (review9: refused cars stood beside, and
///    rolled into, the very car they waited for, on 5229/5231, 1430/1179, 10814/10625):
///    [round6 J2] only a convergence beginning within the stopping distance (`CONVERGE_LOOK`
///    at least, counted from the line while short of it) stops the car at `first`; a
///    farther one it rolls up to (the stop is pulled back before it), unless the vehicle
///    waited for stands in it. (case01 #4252 froze across the player's lane for a
///    convergence 90 m on, case11 #54242 mid-turn.) In order:
///    a. one farther on only (`far_only`), or - past the line, or no longer able to stop
///       there gently - one come near that lies `CONVERGE_LOOK` past the line or farther:
///       before the convergence (`at`), not short of the first meeting place, inside the
///       convergence or on a zebra cut (review-r6j);
///    b. `first`, when it lies before the stretch's stop or a convergence is near, and the
///       car can still stop there;
///    c. a convergence near, past the line already: as soon as it can stop gently;
///    d. the stretch's stop.
fn refused_stop(net: &Network, sim: &Simulation, vehicles: &[Vehicle], me: &Vehicle, j: &JunctionView, g: &Grant) -> f32 {
    let stop = refused_stop_cases(net, sim, vehicles, me, j, g);
    // [fix pass 3, 4] off a crossing: a body that would stand over a crossing band at the
    // stop (`pulled_back` only pulls back before bands ahead of the front, so a body already
    // partly in one stopped there: x10 case04 #3222 at s 4.0 with its rear over the crossing
    // at 2.2) goes forward until its rear is past the crossing and `ZEBRA_PAST` where the
    // meeting place it waits for allows, else back to `ZEBRA_STOP` before the crossing while
    // it can still stop there
    let limit = match (g.hold, g.at) {
        (Some(Hold::Foe), Some(waited)) => j.entry + waited - 1.0,
        _ => j.conflict_start - 1.0,
    };
    off_crossing(net, sim, me, j, stop, limit.max(stop))
}

/// [fix pass 3, 4] `stop` moved off the crossing bands of `j`'s way (`walk_bands`) the body
/// would stand over there: forward, the rear past the band, no farther than `limit`; else
/// back before it while the front has not passed its start.
fn off_crossing(net: &Network, sim: &Simulation, me: &Vehicle, j: &JunctionView, mut stop: f32, limit: f32) -> f32 {
    let len = me.length();
    for w in walk_bands(net, sim, j) {
        if w.0 < stop && w.1 > stop - len - REST_SHORT {
            let forward = w.1 + len + REST_SHORT;
            if forward <= limit {
                stop = forward;
            } else if w.0 > me.state.front - 0.3 {
                stop = stop.min(w.0);
            }
        }
    }
    stop
}

/// The stop of `refused_stop` before it is moved off a crossing.
fn refused_stop_cases(net: &Network, sim: &Simulation, vehicles: &[Vehicle], me: &Vehicle, j: &JunctionView, g: &Grant) -> f32 {
    let st = &me.state;
    let dp = &me.params;
    let v = st.speed;
    let at_line = j.line - 0.3;
    let gently = v * v / (2.0 * (at_line - st.front - 0.6).max(0.05)) <= dp.comfort_brake * 1.5;
    let line_stop = !j.inside && (gently || v < 0.5);
    let first = if line_stop { at_line } else { j.conflict_start - 1.0 };
    let (Some(Hold::Foe), Some(waited)) = (g.hold, g.at) else { return first };
    if g.roll_up {
        return hold_point(net, sim, me, j, waited);
    }
    if let Some(at) = roll_up_point(net, sim, vehicles, me, j, waited) {
        return at;
    }
    let at = hold_point(net, sim, me, j, waited);
    let look = st.front.max(j.line) + CONVERGE_LOOK + v * v / (2.0 * dp.comfort_brake);
    let waited_body = g.wait_for.and_then(|w| waited_body_on_way(net, sim, vehicles, j, w));
    let bands = converge_bands(sim, j);
    let ahead: Vec<&(f32, f32, usize)> = bands.iter().filter(|b| b.0 < j.entry + waited && b.1 > -st.rear).collect();
    let converges = ahead.iter().any(|b| b.0 < look || waited_body.is_some_and(|(lo, hi)| lo < b.1 && hi > b.0));
    let far_only = !converges && !ahead.is_empty();
    let past_line_look = !ahead.is_empty() && ahead.iter().all(|b| b.0 >= j.line + CONVERGE_LOOK);
    if (far_only || (converges && !line_stop && past_line_look)) && at > st.front - 0.3 {
        at
    } else if (at < first || converges) && first > st.front - 0.3 {
        first
    } else if converges {
        at.min(st.front + v * v / (2.0 * dp.comfort_brake) + 0.3)
    } else {
        at
    }
}

pub fn plan(net: &Network, sim: &Simulation, vehicles: &[Vehicle], i: usize, p: &Perception, grant: Option<Grant>) -> Plan {
    let me = &vehicles[i];
    let st = &me.state;
    let dp = &me.params;
    let mut out = Plan::default();
    let Some(lane) = net.lanes.get(st.lane) else { return out };
    let v = st.speed;

    // 1. the lead
    if let Some(l) = p.leader {
        out.lead = Some(l.lead());
        out.lead_seen = Some(l.what);
        out.body_block = p.body_block;
        // [round5 C3] a queue over a pedestrian crossing: behind a lead standing past a cut of
        // my way, where closing up to it would put my body in the zebra band, I stop at the
        // zebra stop instead, as `hold_point` keeps a refused car off it (x10 case15 #35611
        // followed its lead onto the crossing and stood there, the stream through the 2.2 m
        // gap in front of it)
        let coming = l.rest.is_some() || l.speed < QUEUE_SPEED || (l.held && l.speed < QUEUE_COMING);
        if coming && !matches!(l.what, Seen::Parked) {
            // (its rear where it comes to rest: at the stop of its own it brakes for, else
            // where a comfortable brake puts it)
            let lead_rear = l.rest.unwrap_or(st.front + l.gap + l.speed.max(0.0).powi(2) / (2.0 * QUEUE_BRAKE));
            let my_front = lead_rear - st.min_gap;
            for &(wl, d0) in &p.way {
                if d0 > lead_rear {
                    break;
                }
                for &(_, sa, _) in net.walks.get(wl).map(|v| v.as_slice()).unwrap_or(&[]) {
                    let cut = d0 + sa;
                    if !(cut < lead_rear && my_front > cut - ZEBRA_STOP && my_front - me.length() < cut + ZEBRA_PAST) {
                        continue;
                    }
                    // (past the zebra stop already: where it can stop gently, while that is
                    // still short of the cut - as `hold_point` does for a refused car)
                    let mut at = cut - ZEBRA_STOP;
                    if at <= st.front - 0.3 {
                        at = st.front + v * v / (2.0 * dp.comfort_brake) + 0.3;
                    }
                    if at > st.front - 0.3 && at <= cut - 0.3 {
                        min_stop(&mut out.stop, &mut out.why, Some(at), Why::Lead);
                    }
                }
            }
        }
    }

    // 2. lights (not the one whose line it passed while it clears the box: in it when that
    // light turned, it drives on; a later line, of that controller or not, holds it as any)
    if let Some(sig) = p.signal.filter(|s| !p.junction.as_ref().is_some_and(|j| j.clears_past(s, st.front))) {
        let call = light_call(me, p, &sig);
        out.light_decision = call.decision;
        if !call.go {
            min_stop(&mut out.stop, &mut out.why, Some(sig.dist), Why::Light);
        }
    }

    // 3. the junction
    if let (Some(j), Some(g)) = (p.junction.as_ref(), grant) {
        if !g.go {
            let at = refused_stop(net, sim, vehicles, me, j, &g);
            if let (Some((gap, speed)), Some(w)) = (g.follow, g.wait_for.filter(|&w| w < vehicles.len())) {
                // second at a zipper merge, both rolling: in behind the other at my gap
                out.hold = g.hold;
                out.wait_for = g.wait_for;
                let behind = Lead { gap, speed, acc: 0.0 };
                if out.lead.map(|l| gap < l.gap).unwrap_or(true) {
                    out.lead = Some(behind);
                    out.lead_seen = Some(Seen::Vehicle(w));
                }
            } else if at > st.front - 0.3 {
                out.hold = g.hold;
                out.wait_for = g.wait_for;
                let w = if g.hold == Some(Hold::ExitFull) { Why::ExitFull } else { Why::Junction };
                min_stop(&mut out.stop, &mut out.why, Some(at), w);
            }
        }
    }

    // (eased off in a mutual hold: it stays where it stands until the other is through)
    // [round6 J3] pinned there only while it stands and its grant offers no follow; with one
    // (the other rolls on into a merge of the two ahead) it rolls in behind it, the follow
    // branch above; rolling with none, it stops gently (case03: #5424 stood pinned 13 s with
    // the follow behind #8417 ready 9 of them; case06 #13911 10 s on the zebra)
    if let (Some(_), Some(g)) = (me.ease_for, grant) {
        if !g.go {
            out.hold = g.hold;
            out.wait_for = g.wait_for;
            if g.follow.is_none() {
                let at = if v < EASE_STANDING { st.front + 0.1 } else { st.front + v * v / (2.0 * dp.comfort_brake) + 0.3 };
                min_stop(&mut out.stop, &mut out.why, Some(at), Why::Junction);
            }
        }
    }

    // 4. people
    min_stop(&mut out.stop, &mut out.why, p.people_stop, Why::People);

    // 5. the app's own holds
    min_stop(&mut out.stop, &mut out.why, me.external.stop_at, Why::Service);
    if me.pull_out > 0.0 {
        min_stop(&mut out.stop, &mut out.why, Some(st.front + 0.1), Why::PullOut);
    }
    if let Some(d) = p.letting_out {
        min_stop(&mut out.stop, &mut out.why, Some(st.front + (d - 1.0).max(0.0)), Why::LetOut);
    }

    // 6. a route's lane change: now if the gap is there, else wait at the lane's end (a
    //    route may lead into a lane the path's rules close to the vehicle's class: only
    //    the gap counts)
    if let Some((to, dir)) = p.route_change {
        let nb = if dir == 1 { p.left } else { p.right };
        let ok = nb.map(|n| gap_accepted(me, &n)).unwrap_or(true);
        if ok && st.change.is_none() && st.change_cooldown <= 0.0 {
            out.start = Some(Start::RouteChange { to, dir });
        } else if st.change.is_none() {
            min_stop(&mut out.stop, &mut out.why, Some(lane.length() - 1.0), Why::Merge);
        }
    }

    // 7. a lane change of the driver's own: round something standing, for speed, to keep
    //    to the side, into the lane of the turn ahead
    let free_to_change = st.change.is_none() && st.route.is_empty() && st.change_cooldown <= 0.0 && me.passing.is_none() && out.start.is_none() && !me.is_rail() && lane.length() - st.s > 9.0;
    if free_to_change {
        let near_junction = p.junction.as_ref().map(|j| !j.inside && j.line < JUNCTION_KEEP).unwrap_or(false);
        let mut best: Option<(f32, NeighbourView)> = None;
        for nb in [p.left, p.right].into_iter().flatten() {
            let (d, ok) = change_desire(net, sim, me, p, &nb, near_junction);
            if ok && d > 0.0 && best.map(|b| d > b.0).unwrap_or(true) {
                best = Some((d, nb));
            }
        }
        let threshold = 0.4 - 0.2 * me.impatience.value;
        if let Some((_, nb)) = best.filter(|b| b.0 >= threshold) {
            let standing = p.leader.filter(|l| l.standing_obstacle() && l.gap < 25.0);
            out.start = match standing {
                Some(l) if v < 1.0 => bypass_start(net, sim, vehicles, me, p, &nb, &l),
                _ => Some(Start::Change { to: nb.lane, dir: nb.dir }),
            };
        }
    }

    // 8. passing on the oncoming lane, or squeezing past inside the lane
    let out_side = sim.out_side();
    if let Some(pass) = me.passing {
        // out there: come back when the oncoming traffic is nearer than the time needed
        if !pass.aborted && pass.lane.is_some() {
            let remaining = pass.until + pass.back_len - st.odometer;
            if let Some(o) = p.oncoming {
                let v_cap = (p.v_want).clamp(4.0, 14.0);
                let t_need = arrival_time(remaining.max(0.0), v, st.accel, v_cap) + 0.5;
                let shallow = st.lateral.abs() < pass.side.abs() - me.half_width - 1.45;
                if o.clear_for < t_need && shallow && st.odometer + v * v / 7.0 + 0.4 < pass.block {
                    out.start = Some(Start::AbortPass { hold: pass.block - 2.0 });
                }
            }
        }
        // passing a bus that indicates out of its stop or moves off: back in behind it at the
        // gap the driver keeps, while it still can (it pulls out in front of the car; the
        // player's bus is let out of its stop so, `letting_out`)
        if !pass.aborted && out.start.is_none() {
            if let Some((true, signals, moves)) = pass.obstacle.and_then(|ob| obstacle_bus(sim, vehicles, ob)) {
                let shallow = pass.wait > 0.0 || st.lateral.abs() < pass.side.abs() - me.half_width - 1.45;
                if (signals || moves) && shallow && st.odometer + v * v / 7.0 + 0.4 < pass.block {
                    out.start = Some(Start::AbortPass { hold: pass.block + st.front + 0.6 - st.min_gap });
                }
            }
        }
        // standing out there with the way back in blocked: given up, back in behind the obstacle
        // at the gap kept (where it is still behind it), else into the lane where it is, and the
        // obstacle is a lead again (viz case09 #30816 stood 110 s on the oncoming side)
        // (past it, on its way back in, it is stranded too while it stands still out there, held
        // by a red light or a queue: review12, #20946 stood out on a scheduled bus's lane 84 s)
        let held_out = st.lateral.abs() > 1.0;
        if !pass.aborted && out.start.is_none() && pass.wait <= 0.0 && me.stopped > PASS_STRANDED && (st.odometer < pass.until || held_out) {
            // (beside it or past it already: it stays where it is and steps back in as soon as
            // its body there touches nobody, `Simulation::sidestep`)
            let hold = if st.odometer + 0.5 < pass.block { pass.block + st.front + 0.6 - st.min_gap } else { st.odometer };
            out.start = Some(Start::AbortPass { hold });
        }
        // (given up, it holds where it said, until what it went round has left)
        let gone = pass.obstacle.is_some_and(|ob| obstacle_bus(sim, vehicles, ob).is_none_or(|(_, _, moves)| moves));
        if pass.aborted && !gone {
            min_stop(&mut out.stop, &mut out.why, Some((pass.hold - st.odometer).max(0.0)), Why::Lead);
        } else if pass.aborted {
            // (its obstacle gone: nothing holds it here, the pass is over, `Simulation::act`)
        } else if pass.wait > 0.0 {
            // signalling: still behind it, with the room to pull out
            min_stop(&mut out.stop, &mut out.why, Some(st.front + (pass.block - st.odometer - pass.out_len - 1.0).max(0.0)), Why::Lead);
        } else {
            if st.odometer < pass.block + 2.0 {
                // edging out: gently
                out.accel_cap = Some(omsi_sim::ai_motion::PULL_OUT_ACCEL);
            }
            // the thing being passed no longer holds the car (the bent way clears it)
            if let Some(l) = p.leader {
                // (not one standing on the line the pass takes: that holds it)
                let passing_it = (l.standing_obstacle() || pausing_player(&l)) && st.odometer + l.gap + st.front < pass.until + 1.0 && l.lateral.abs() < (l.lateral - pass.side).abs() && shift_clears(net, sim, vehicles, me, pass.side, &l);
                if passing_it {
                    out.lead = None;
                    out.lead_seen = None;
                }
            }
        }
    } else if let Some(l) = p.leader.filter(|l| l.standing_obstacle() || pausing_player(l) || behind_an_obstacle(vehicles, l)) {
        // (the player's bus pausing 1.5 s is squeezed past inside the lane; everything else
        // waits for it to be a standing obstacle; a car standing in a queue behind one is only
        // kept back from, for a bypass round them, `behind_an_obstacle`)
        let queue_only = !(l.standing_obstacle() || pausing_player(&l));
        let standing = l.standing_obstacle() || queue_only;
        let can_pass = !me.is_rail() && me.class != VehicleClass::Air && st.change.is_none() && out.start.is_none();
        // (the oncoming lane is used only on a road with one lane each way; not across a
        // solid line: a pass there stays a squeeze within the drawn road)
        let two_lane = lane.left.is_none() && lane.right.is_none();
        let solid = lane.no_overtaking;
        let player = matches!(l.what, Seen::Player(_));
        let player_out = player && sim.players.iter().any(|pl| pl.blinker as i32 == out_side && pl.signalling < 1.0);
        // the thing standing, and a bus of it that indicates out of its stop (not passed: it
        // is let out, Sobol's rule)
        let obstacle_id = match l.what {
            Seen::Vehicle(j) | Seen::Body(j) => vehicles.get(j).map(|o| o.id),
            Seen::Player(id) => Some(crate::sim::PLAYER_ID_BASE + id as u64),
            Seen::Parked => None,
        };
        let bus_out = obstacle_id.and_then(|ob| obstacle_bus(sim, vehicles, ob)).is_some_and(|(bus, signals, _)| bus && signals);
        // a bus does not pass what stands at its own next stop
        let own_stop = me.external.stop_at.map(|s| s < l.gap + l.length + st.front + 15.0).unwrap_or(false);
        let real = l.gap + if player { 0.5 } else { 0.0 };
        // how far the car has to swing out: past the thing inside the drawn road if that
        // leaves room beside where it stands (the lane's drivable extent on that side, L1:
        // to the kerb, the pavement or the next lane's painted edge), else onto the
        // oncoming lane
        let need = me.half_width + l.half_width + 0.35;
        let towards_middle = sim.oncoming_sign();
        let squeeze_side = if l.lateral * towards_middle > 0.3 { l.lateral - towards_middle * need } else { l.lateral + towards_middle * need };
        let (drivable_left, drivable_right) = lane.drivable();
        // (and never into the traffic of the lane beside or the oncoming lane: a car's half
        // width and 0.3 off their axes, whatever the stored extent says - a recording or a
        // piece without edge data, review r5l safety S3/S4; beyond that it is the wheels over
        // the line into a free neighbour lane, or the pass on the oncoming lane)
        let traffic_cap = |sign: f32| -> f32 {
            let mut cap = f32::INFINITY;
            let nb = if sign < 0.0 { p.left } else { p.right };
            if let Some(nl) = nb.and_then(|nb| net.lanes.get(nb.lane)) {
                if let Some((_, d)) = nl.nearest_point(me.pos) {
                    cap = cap.min(omsi_scenery::profile::neighbour_edge(d as f32, nl.width));
                }
            }
            if sign == towards_middle {
                if let Some((opp, _, oside)) = net.opposite(st.lane, st.s) {
                    cap = cap.min(omsi_scenery::profile::neighbour_edge(oside.abs(), net.lanes[opp].width));
                }
            }
            cap
        };
        let extent = (if squeeze_side < 0.0 { drivable_left } else { drivable_right }).min(traffic_cap(squeeze_side.signum()));
        let reach = squeeze_side.abs() + me.half_width;
        // not quite enough drawn road: the driver-side wheels may go `WHEELS_OVER_LINE` over
        // the line into the neighbour lane on that side when it is the car's to drive, nobody
        // passes on it the wrong way, no solid line, and it is free for the squeeze's length
        // (ahead past what is passed and the way back in, behind the gap the driver accepts)
        let squeeze_out_len = Passing::out_len_for(squeeze_side);
        let (squeeze_obstacle_len, _) = obstacle_run(p, &l, me.length() + squeeze_out_len + PASS_ROOM);
        let over_line = if reach > extent && reach <= extent + WHEELS_OVER_LINE && !solid {
            let nb = if squeeze_side < 0.0 { p.left } else { p.right };
            nb.filter(|nb| nb.allowed && !nb.foreign && !net.lanes[nb.lane].no_overtaking && nb.ahead.is_none_or(|a| a.0 > real + squeeze_obstacle_len + squeeze_out_len + PASS_ROOM) && gap_accepted(me, nb))
        } else {
            None
        };
        // (and the bent corridor - out, past what stands and back - touches no body now, no
        // person stops the car within it, and the thing passed has settled sideways)
        let squeeze_end = real + squeeze_obstacle_len + st.front + st.rear + 2.0;
        let squeeze = (reach <= extent || over_line.is_some())
            && !player_out
            && lateral_settled(vehicles, &l)
            && p.people_stop.is_none_or(|s| s > squeeze_end + Passing::back_len_for(squeeze_side))
            && squeeze_clear(net, sim, vehicles, me, &p.way, squeeze_side, squeeze_out_len, squeeze_end, Passing::back_len_for(squeeze_side));
        let side = if squeeze { squeeze_side } else { p.oncoming.map(|o| o.side).unwrap_or(0.0) };
        let out_len = Passing::out_len_for(side);
        let room = out_len + 1.0;
        // what a pass would need, now: no light or junction just past the obstacle (within
        // `JUNCTION_KEEP` of the front, a junction), the road going on, and the oncoming lane
        // (or the lane's own drawn road) free for it
        // what stands right behind it is passed with it, and beyond them the lane must have
        // room to come back in: my length, the way to pull back in and 4 m (`PASS_ROOM`; a car
        // or player standing or slower than 2 m/s there takes it; viz case09 #30816 pulled out
        // round two standing cars with three more beyond and stood out there 110 s)
        let (obstacle_len, room_beyond) = obstacle_run(p, &l, me.length() + out_len + PASS_ROOM);
        let pass_len = real + obstacle_len + st.front + st.rear + 6.0;
        let zone = if squeeze || !player { SQUEEZE_ZONE } else { PASS_ZONE_PLAYER };
        let light_near = p.signal.map(|s| s.dist < real + st.front + obstacle_len + zone).unwrap_or(false);
        // (an in-lane squeeze is not kept `JUNCTION_KEEP` from a junction when the obstacle
        // stands within `SQUEEZE_NEAR`, and inside the junction already it may go where its
        // stretch - out, past the obstacle and back in - keeps clear of the foe stretches)
        let squeeze_run = real + obstacle_len + st.front + st.rear + 2.0 + Passing::back_len_for(side);
        let junction_near = p.junction.as_ref().map(|j| {
            if matches!(l.what, Seen::Parked) {
                return false;
            }
            // (a junction object's pieces carry no edge data - their extent is their half
            // width - so a squeeze inside a junction needs the obstacle mostly off the piece;
            // in practice this branch waits for pieces with real extents)
            if squeeze && j.inside {
                return squeeze_run > j.conflict_start && j.conflict_end > -st.rear;
            }
            let keep = j.line - st.front < JUNCTION_KEEP && !(squeeze && real < SQUEEZE_NEAR);
            j.line < real + st.front + obstacle_len + zone || keep
        }).unwrap_or(false);
        let road_ok = p.road_left > pass_len + real + 30.0;
        let oncoming = p.oncoming.filter(|o| !solid && o.side.abs() >= 2.3 && o.side.abs() <= 5.5 && !o.blocked);
        // TODO(review11, pre-existing): `clear_for` is the oncoming car's time to the passer's
        // current position, compared with the pass time + 1.5 s; a car coming at 13 m/s meets
        // the passer before that point. It should be the time to where the pass ends.
        let oncoming_free = oncoming.is_some_and(|o| {
            let v_cap = (p.v_want).clamp(4.0, 14.0);
            o.clear_for > arrival_time(pass_len + Passing::back_len_for(side), v, st.accel, v_cap) + 1.5
        });
        let pass_possible = !queue_only && can_pass && !own_stop && !player_out && !bus_out && room_beyond && real >= 0.3 && !light_near && !junction_near && road_ok && (squeeze || (two_lane && standing && oncoming_free));
        // (or round it into the lane beside, `Start::Bypass`)
        let bypass_nb = [p.left, p.right].into_iter().flatten().find(|nb| could_bypass(net, me, nb, &l));
        let bypass_possible = standing && !light_near && !junction_near && bypass_nb.is_some();
        // (a bypass needs the room its body can steer round the obstacle in, `bypass_room`)
        let room = match bypass_nb.filter(|_| bypass_possible && !pass_possible) {
            Some(nb) => bypass_room(net, me, &nb, &l).unwrap_or(room),
            None => room,
        };
        // keep back: room to pull out again, where the car can still stop there gently - only
        // where a pass (or a bypass) could start now; behind a standing player at a red 40 m
        // on, or a queue at a junction, the car closes up to its queue gap (viz3 review: 4785
        // keyframe rows 6 m back behind AI leads and 56 behind the player where no pass could
        // ever start)
        // (a bypass's room is held once reached: closing in from 0.5 m beyond it, the car rolled
        // on to 2 m behind a bus at its stop and could never steer out, viz5 case07 #37625)
        let bypass_room_held = bypass_possible && !pass_possible;
        let slack = if bypass_room_held { -BYPASS_HOLD_SLACK } else { 0.5 };
        if (pass_possible || bypass_possible) && !(me.serves_stops() && l.at_stop) && l.gap > room + slack {
            let dist = (l.gap - room).max(0.0);
            let need_brake = v * v / (2.0 * dist.max(0.1));
            if need_brake <= dp.comfort_brake * 1.5 {
                min_stop(&mut out.stop, &mut out.why, Some(st.front + dist), Why::Lead);
            }
        }
        // (a car that ended up too close - the thing stopped in front of it - pulls out
        // over what room it has once it has stood a while, rather than for ever)
        let too_close = l.gap < room - 1.0;
        // a squeeze past the player's bus starts as soon as the bus has paused and the car
        // has closed up (no 3 s stop of its own); a pass waits for 3 s stopped
        let early = squeeze && player && v < SQUEEZE_START_SPEED;
        let ready = (me.stopped >= 3.0 || early || matches!(l.what, Seen::Parked)) && (!too_close || me.stopped >= 10.0) && l.gap >= 1.5;
        if pass_possible && ready {
            let out_len = if too_close { (l.gap - 0.3).clamp(2.0, out_len) } else { out_len };
            let until = st.odometer + real + obstacle_len + st.front + st.rear + 2.0;
            let lane = if squeeze { None } else { oncoming.map(|o| o.lane) };
            let wait = if squeeze { SQUEEZE_SIGNAL } else { omsi_sim::traffic::SIGNAL_BEFORE_CHANGE };
            out.start = Some(Start::Pass(Passing { lane, side, until, block: st.odometer + real, out_len, back_len: Passing::back_len_for(side), aborted: false, hold: 0.0, wait, obstacle: obstacle_id, beside: over_line.map(|nb| nb.lane) }));
        }
    }

    // 9. where the car sits in its lane: a bay, a swerve round a parked car, or the middle
    if me.passing.is_none() && st.change.is_none() {
        // ([round 7 A7] a timetable bus has a bay only at or into its stop: between stops it
        // swerves round a parked car as everybody does - not towards the middle past the
        // people at the kerb, whom it serves: the round-5 L3 shift took bus 267 0.5 m out
        // beside the waiters of its next stop, Anwai 300 s run)
        if let Some(bay) = me.external.bay {
            out.lateral = Some(bay);
        } else if me.serves_stops() {
            out.lateral = Some(p.parked_swerve.unwrap_or(0.0));
        } else {
            out.lateral = Some(p.kerb_swerve.unwrap_or(0.0));
            // [round 5 L3] easing past people beside the lane: gently, as a car edging out
            if p.people_swerve && v < 3.0 && out.accel_cap.is_none() {
                out.accel_cap = Some(omsi_sim::ai_motion::PULL_OUT_ACCEL);
            }
        }
    }

    // 10. the end of a fixed route: stop at its end. [ROLLBACK r4d-B1] Not where the road
    //     itself ends (a street lane that runs into nothing, no stop of the bus's own on it):
    //     there the bus keeps its speed and is taken off as it reaches the end (`AiState::
    //     drive` finds no way on), as a plain car at a dead end; the timetable puts it back
    //     where its track goes on. Held 0.5 m before the end, two buses on the Anwai bridge
    //     (BeijingCity) stood for up to half an hour with fifty cars behind them.
    //     [round 6 B2] Nor where its trip is over at the end of its track on a lane with a way
    //     on (`External::trip_over`): T11's bus stood 11.7 min at Why::End on lane 10871 with
    //     its next trip never handed over (review-r6-cases case08).
    if !st.route.is_empty() && st.route_index + 1 >= st.route.len() && st.planned_next.is_none() {
        let end = lane.length() - st.s - 0.5;
        let dead_end = lane.next.is_empty() && lane.kind == omsi_sim::traffic::LaneKind::Street;
        // ([fix pass 3, 10] nor where the stop it is asked for lies beyond the end of its
        // route: the route's next part is not loaded yet, it waits at the end - a bus whose
        // stop lay 76 m past its route's end drove off it, x10 case11 #18542)
        let stop_beyond = me.external.stop_at.is_none_or(|sa| sa > end + st.front + 0.5);
        if end < 150.0 && stop_beyond && me.class == VehicleClass::ScheduledBus && !dead_end && !me.external.trip_over {
            min_stop(&mut out.stop, &mut out.why, Some(end.max(0.0)), Why::End);
        }
    }

    // 11. [fix pass 3, 2] keep the box clear, on every aspect (SUMO's no-block heuristic,
    //     sumo.md §2): a car short of its junction's lit line does not cross it while where it
    //     would come to rest - its stop (people, a refusal) or behind a lead standing or coming
    //     to rest - lies between the line and the way's exit from the box; it holds at the line,
    //     pulled back before a crossing there (viz5 case05 #16535 crossed on green into a box
    //     its people stop lay in and stood 2.8 m in, over the crossing)
    if let (Some(j), Some(sig)) = (p.junction.as_ref(), p.signal.as_ref()) {
        let line = sig.dist;
        if !j.inside && sig.dist <= j.light_line.max(j.conflict_start) + 1.0 && line - st.front > 0.3 {
            let lead_rest = p.leader.filter(|l| l.speed < QUEUE_SPEED || l.rest.is_some()).map(|l| st.front + l.gap.min(l.rest.map(|r| r - st.front).unwrap_or(f32::INFINITY)));
            let rest = [out.stop, lead_rest].into_iter().flatten().fold(f32::INFINITY, f32::min);
            let exit = j.entry + j.way_len;
            if rest > line + 0.5 && rest < exit {
                let hold = pulled_back(&walk_bands(net, sim, j), me, line);
                out.stop = Some(hold);
                out.why = Why::ExitFull;
            }
        }
    }

    out.held = out.stop.is_some() || p.leader.map(|l| l.gap < 12.0).unwrap_or(false);
    if out.why == Why::None && out.lead.is_some() {
        out.why = Why::Lead;
    }
    out
}
