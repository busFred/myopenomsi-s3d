//! The decision: from what the driver sees to one plan for this frame. A plan is the
//! lead to follow, a stop point, a lateral target and at most one manoeuvre to start.
//! No plan reads another vehicle's plan; the only cross-vehicle input is the junction
//! grant.

use crate::driver::{DriverParams, VehicleClass, STANDING_SPEED};
use crate::perception::{JunctionView, LaneEndKind, NeighbourView, Perception, Seen, SignalView};
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
    /// Refused by the junction for want of room on its exit lane (`Hold::ExitFull`, SUMO's
    /// junction-blocking test at the link).
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
    /// Held at its line so as not to come to rest in the box (step 11 of `plan`, SUMO's
    /// keep-clear rule): where it would stop lies between the line and the way's exit and is
    /// no waiting place.
    KeepClear,
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
            Why::KeepClear => "keep_clear",
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
    /// [J7-3] SUMO LC2013's cooperation (`informFollower`): the follower in the lane I must
    /// change into is asked to make room - (its index, the gap from its front to my rear, my
    /// speed, the gap my acceptance needs from it; `Vehicle::help_for`).
    pub help: Option<(usize, f32, f32, f32)>,
    /// [J7-3] What the lane change of the driver's own it starts is for (`ChangeWhy`).
    pub change_why: Option<ChangeWhy>,
    /// [Q8-4] The last-resort stop before a lane's end placed this frame (the odometer where the
    /// front stops, `Vehicle::lane_end_stop`).
    pub lane_end_stop: Option<f32>,
    /// [Q8-1] Backing up out of a packed queue (`queue_escape`): [fix pass 1] (the odometer where it
    /// is to stand, stopping: a committed stop - the strip behind no longer clear or the queue
    /// moving).
    pub reverse: Option<(f32, bool)>,
    /// [Q8-3] SUMO LC2013's keep-right accumulation this frame (per second, `keep_right_rate`):
    /// `Vehicle::keep_right` grows by it; None where no lane beside on the keep side is one the
    /// driver may change onto (the wish is left as it is).
    pub keep_right_rate: Option<f32>,
    /// [round 8 H8-1] A car leaving a parking space is held where it is parked because its
    /// lane has no room behind it: its `pull_out` clock waits.
    pub pull_out_held: bool,
    /// [P8 pass 1, S1] Held standing out of its lane on the way back in (`reentry_clear`):
    /// the ramp back is not begun while held (`Simulation` keeps its start ahead of the body).
    pub reentry_held: bool,
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
    let (ok_behind, ok_ahead) = gap_parts(me, nb);
    ok_behind && ok_ahead && !nb.foreign
}

/// The gap (m) the change needs from a follower at `vb` m/s there while I go at `v`: [`gap_accepted`]'s
/// (the follower need not brake hard). (`informFollower` asks for it, [J7-3].)
fn gap_needed_behind(me: &Vehicle, nb: &NeighbourView, vb: f32, v: f32) -> f32 {
    accepted_gap(me, nb.behind_min_gap, vb, v)
}

/// The gap (m) I accept in front of a follower whose standing gap is `min_gap`, coming at `vb`
/// m/s while I go at `v`: its standing gap, my accepted time gap at its speed and what it
/// sheds braking at my comfort to my speed (`gap_needed_behind`'s; [merge P8/H8] the way back
/// into the lane from out of it asks each follower's own secure gap instead, `pull_out_clear`).
fn accepted_gap(me: &Vehicle, min_gap: f32, vb: f32, v: f32) -> f32 {
    let dp = &me.params;
    // ([fix pass 2] the follower's own standing gap, not a fixed 2 m: the helper opens exactly
    // this gap, `informFollower`, one function on both sides; a follower resting at its 1.57 m
    // standing gap behind the changer was never accepted, and both stood)
    min_gap + me.impatience.gap(dp.lc_gap) * vb + (vb - v).max(0.0).powi(2) / (2.0 * dp.comfort_brake)
}

/// [`gap_accepted`]'s two halves: (the follower there need not brake hard, there is room ahead).
fn gap_parts(me: &Vehicle, nb: &NeighbourView) -> (bool, bool) {
    let v = me.state.speed;
    let ok_behind = nb.behind.map(|(g, vb)| g >= gap_needed_behind(me, nb, vb, v)).unwrap_or(true);
    let ok_ahead = nb.ahead.map(|(g, va)| g > 2.0 + (v - va).max(0.0) * 1.5).unwrap_or(true);
    (ok_behind, ok_ahead)
}

/// [round 8 H8-1/H8-8, fix pass 1] A vehicle behind a car about to pull out of a parking space,
/// on the lane it pulls out onto (or a lane leading into it): the gap from its front to the
/// pull-out's rear along the lane (m), its speed and its own driver.
#[derive(Debug, Clone, Copy)]
pub struct PullOutFollower {
    pub gap: f32,
    pub speed: f32,
    pub params: DriverParams,
}

/// SUMO's insertion test of the follower (`MSLane::isInsertionSuccess`: the follower's own
/// `getSecureGap`, its headway and its braking, behind a vehicle put down at speed 0, plus its
/// minGap): the gap that follower needs (m). No impatience: an insertion has none.
pub fn secure_gap_behind(f: &PullOutFollower) -> f32 {
    f.params.min_gap + f.speed * f.params.tau + f.speed * f.speed / (2.0 * f.params.comfort_brake.max(0.1))
}

/// The vehicles behind lane `lane` at `s` (the pull-out's origin, `rear` behind it to its
/// rear): on that lane and back over the lanes leading into it as far as the slowest-braking
/// preset at the lane's limit needs (`secure_gap_behind`), the players too (their preset's
/// driver). `on_lane(l)` gives the AI vehicles on lane `l`: (s, front, speed, driver).
pub fn pull_out_followers(net: &Network, sim: &Simulation, lane: usize, s: f32, rear: f32, on_lane: &dyn Fn(usize) -> Vec<(f32, f32, f32, DriverParams)>) -> Vec<PullOutFollower> {
    let Some(l) = net.lanes.get(lane) else { return Vec::new() };
    let v_lim = l.speed_limit_kmh.max(0.0) / 3.6;
    let reach = secure_gap_behind(&PullOutFollower { gap: 0.0, speed: v_lim, params: DriverParams::preset(VehicleClass::ScheduledBus) });
    let player = DriverParams::preset(crate::sim::PLAYER_PRESET);
    let mut out = Vec::new();
    // (lane, how far its start lies behind the pull-out's origin)
    let mut frontier: Vec<(usize, f32)> = vec![(lane, -s)];
    let mut seen: Vec<usize> = vec![lane];
    while let Some((cur, start)) = frontier.pop() {
        let mut add = |os: f32, front: f32, speed: f32, params: DriverParams| {
            let at = start + os;
            if cur == lane && at >= 0.0 {
                return;
            }
            out.push(PullOutFollower { gap: -at - front - rear, speed, params });
        };
        for (os, front, speed, params) in on_lane(cur) {
            add(os, front, speed, params);
        }
        for &(k, ps, _) in sim.player_occupancy.get(&cur).map(|v| v.as_slice()).unwrap_or(&[]) {
            if let Some(p) = sim.players.get(k) {
                add(ps, p.half_len, p.speed, player);
            }
        }
        if -start > reach {
            continue;
        }
        for &pl in net.prev.get(cur).map(|v| v.as_slice()).unwrap_or(&[]) {
            if !seen.contains(&pl) {
                seen.push(pl);
                frontier.push((pl, start - net.lanes[pl].length()));
            }
        }
    }
    out
}

/// [round 8 H8-8, fix pass 1] Nobody is on the strip a pull-out's body sweeps from its parking
/// offset onto its lane: the body at lane `lane`, `s` + d, `lateral` eased to 0 over `ramp_len`
/// m as `AiState::lateral_ramp` does, for d from where it stands to the ramp's end (a walker
/// passing through the parked body holds it until through - leaving them out, the car drove off
/// with a walker inside its body, viz case15 #54577, a probe; the walkers on the pavement beside
/// and behind it, which held Jiangzhaikou #1913 at its offset, are not on the strip); each person
/// where they are and where they will be
/// when the body is at that metre, driving from rest at `PULL_OUT_ACCEL` (SUMO's insertion fails
/// on a person on the footprint, `MSLane::checkForPedestrians`; on its way it brakes for persons
/// on its lane).
pub fn pull_out_strip_clear(net: &Network, sim: &Simulation, lane: usize, s: f32, lateral: f32, front: f32, rear: f32, half_width: f32, ramp_len: f32, v: f32) -> bool {
    let Some(l) = net.lanes.get(lane) else { return true };
    let (q, _) = l.at(s.min(l.length()));
    !sim.people_near(q, (ramp_len + front + rear) as f64 + 10.0).into_iter().any(|k| on_pull_out_strip(net, lane, s, lateral, front, rear, half_width, ramp_len, v, &sim.people[k]))
}

/// [round 8 H8-8, fix pass 2] Whether person `p` is on the strip (`pull_out_strip_clear`): where
/// they are, or where they will be when the body is at that metre - the car driving on from `v`
/// m/s at `PULL_OUT_ACCEL` up to the lane's limit (`arrival_time`; pass 1 timed it from rest even
/// when moving).
pub fn on_pull_out_strip(net: &Network, lane: usize, s: f32, lateral: f32, front: f32, rear: f32, half_width: f32, ramp_len: f32, v: f32, p: &crate::sim::Person) -> bool {
    let Some(l) = net.lanes.get(lane) else { return false };
    let len = l.length();
    let v_lim = l.speed_limit_kmh.max(1.0) / 3.6;
    let mut d = 0.0f32;
    while d <= ramp_len + 0.01 {
        let at = (s + d).min(len);
        let (q, h) = l.at(at);
        let hr = (h as f64).to_radians();
        let lat = lateral * (1.0 - omsi_sim::traffic::smooth01(d / ramp_len.max(0.1)));
        let c = q.truncate() + DVec2::new(hr.cos(), -hr.sin()) * lat as f64;
        let body = Obb::vehicle(c, h as f64, front as f64, rear as f64, half_width as f64);
        let later = p.pos.truncate() + p.vel * arrival_time(d, v.max(0.0), omsi_sim::ai_motion::PULL_OUT_ACCEL, v_lim) as f64;
        if [p.pos.truncate(), later].iter().any(|&x| Obb::point(x.extend(p.pos.z), PERSON_RADIUS as f64).separation(&body) < 0.0) {
            return true;
        }
        d += 1.0;
    }
    false
}

/// [round 8 H8-8, fix pass 2] The rest of a pull-out's way out (m): the steerable move from rest
/// while it has not left (`bypass_min_len`), else what is left of its way back (`way_back`).
pub fn pull_out_ramp_left(me: &Vehicle) -> f32 {
    let st = &me.state;
    let (_, _, x0, len) = way_back(me);
    if st.speed < STANDING_SPEED && me.pull_out > 0.0 {
        bypass_min_len(me.length(), st.lateral)
    } else {
        (x0 + len - st.odometer).max(1.0)
    }
}

/// [round 8 H8-8, fix pass 2] The one exemption the people test makes for a car pulling out of a
/// parking space (`people_ahead` asks it for every person): somebody under or beside its parked
/// body - level with it, from its rear to its front - who is not on the strip it sweeps onto its
/// lane (`on_pull_out_strip`) is not in its way (the walkers passing over the pavement at its
/// spot, Jiangzhaikou #1913). Everybody else is tested as by any car (pass 1 switched the test
/// off while the strip was clear: probes hit people just past the strip at 3-4 m/s).
pub fn pull_out_exempt(net: &Network, me: &Vehicle, p: &crate::sim::Person) -> bool {
    if !me.pulling_out {
        return false;
    }
    let st = &me.state;
    let hr = (me.heading as f64).to_radians();
    let along = (p.pos - me.pos).truncate().dot(DVec2::new(hr.sin(), hr.cos())) as f32;
    let level = along >= -st.rear - PERSON_RADIUS && along <= st.front + PERSON_RADIUS;
    level && !on_pull_out_strip(net, st.lane, st.s, st.lateral, st.front, st.rear, me.half_width, pull_out_ramp_left(me), st.speed, p)
}

/// [round 8 H8-1/H8-8, fix pass 1] The one pull-out rule, for the parked object the app takes
/// off the kerb, for a recorded pull-out standing at its offset and ([merge P8/H8], P8-4) for any
/// vehicle standing out of its lane on its way back in, a bus leaving its bay: every moving
/// follower has its secure gap (`secure_gap_behind`), a standing one only room for its body, and nobody is on the strip its body sweeps
/// (`pull_out_strip_clear`). SUMO re-inserts a parked vehicle only when `isInsertionSuccess`
/// passes (`MSVehicleTransfer`); OMSI draws the body, so the car then drives from its offset.
pub fn pull_out_clear(net: &Network, sim: &Simulation, followers: &[PullOutFollower], lane: usize, s: f32, lateral: f32, front: f32, rear: f32, half_width: f32, ramp_len: f32) -> bool {
    // ([merge P8/H8] a standing follower reaches only its own body, SUMO's secure gap at v = 0
    // being 0: one queued right behind a car out of its lane held it for good, viz5 case06 #18756)
    followers.iter().all(|f| if f.speed < STANDING_SPEED { f.gap >= 0.0 } else { f.gap >= secure_gap_behind(f) }) && pull_out_strip_clear(net, sim, lane, s, lateral, front, rear, half_width, ramp_len, 0.0)
}

/// [J7-3] SUMO LC2013's strategic look-ahead (`MSLCM_LC2013::LOOK_FORWARD`, s): a lane that
/// ends for the driver raises its wish to leave it from this many seconds of its speed out.
pub const LC_LOOK_FORWARD: f32 = 10.0;
/// [J7-3, fix pass 1] SUMO LC2013's `LOOK_AHEAD_SPEED_MEMORY`: the look-ahead speed falls by this
/// factor per second (`Vehicle::look_ahead_speed`).
pub const LOOK_AHEAD_SPEED_MEMORY: f32 = 0.9;
/// [J7-3] ... times this towards the overtaking side (SUMO's `lcLookaheadLeft` default; the
/// side mirrors on a left-hand-traffic map).
pub const LC_LOOKAHEAD_LEFT: f32 = 2.0;
/// [J7-3] SUMO LC2013's `HELP_DECEL_FACTOR`: a follower asked to make room for a lane change
/// brakes no harder than its comfortable brake times this.
pub const HELP_DECEL_FACTOR: f32 = 1.0;

/// [J7-3, fix pass 1] The road a lane change started at `v` m/s takes (m): the signalling before
/// it moves (`SIGNAL_BEFORE_CHANGE`, speeding up at `a` meanwhile) and the move
/// (`change_length`).
pub fn change_space(v: f32, a: f32) -> f32 {
    let w = omsi_sim::traffic::SIGNAL_BEFORE_CHANGE;
    v * w + 0.5 * a.max(0.0) * w * w + omsi_sim::traffic::change_length(v)
}

/// [J7-3] SUMO LC2013's strategic distance (`laDist`): the look-ahead speed times
/// `LC_LOOK_FORWARD` (times `LC_LOOKAHEAD_LEFT` towards the overtaking side) plus two of the
/// vehicle's lengths with its standing gap.
pub fn lane_end_look(v: f32, overtaking_side: bool, me: &Vehicle) -> f32 {
    v * LC_LOOK_FORWARD * if overtaking_side { LC_LOOKAHEAD_LEFT } else { 1.0 } + 2.0 * (me.length() + me.state.min_gap)
}

/// [J7-3] May the driver change onto `nb` because its own lane ends for it (`Perception::
/// lane_end`): the path's rules allow it there and no solid line holds either lane; on a
/// junction object's piece only onto a piece of the same object beside it that it meets in no
/// crossing (two lanes of one carriageway running side by side: SUMO changes lanes on an edge,
/// never across an internal junction's conflict). The turn-lane rules of `may_change_to` do not
/// hold: the lane ends.
fn may_change_for_lane_end(net: &Network, sim: &Simulation, me: &Vehicle, nb: &NeighbourView) -> bool {
    let lane = &net.lanes[me.state.lane];
    if !nb.allowed || lane.no_overtaking || net.lanes[nb.lane].no_overtaking {
        return false;
    }
    if lane.source != 2 {
        return true;
    }
    match (sim.junctions.of_lane.get(&me.state.lane), sim.junctions.of_lane.get(&nb.lane)) {
        (Some(&(j, m)), Some(&(j2, m2))) if j == j2 => !sim.junctions.nodes[j].movements[m].foes.iter().any(|f| f.movement == m2 && f.kind != crate::junction::FoeKind::Merge),
        _ => false,
    }
}

/// [J7-3] Somebody on the lane `nb` over the move a change would take (from beside my rear to
/// the move's length past my front): no change into people.
fn people_on_move(net: &Network, sim: &Simulation, me: &Vehicle, nb: &NeighbourView) -> bool {
    let st = &me.state;
    let Some(lane) = net.lanes.get(nb.lane) else { return false };
    let s_b = net.beside_s(st.lane, nb.lane, st.s);
    let (from, to) = (s_b - st.rear, s_b + st.front + omsi_sim::traffic::change_length(st.speed));
    let reach = (to - from) as f64 * 0.5 + lane.width as f64;
    let mid = lane.at(((from + to) * 0.5).clamp(0.0, lane.length())).0;
    sim.people_near(mid, reach).into_iter().any(|k| {
        let q = &sim.people[k];
        !q.at_kerb && lane.nearest_point(q.pos).is_some_and(|(s, d)| s >= from && s <= to && d < (lane.width * 0.5 + 0.5) as f64)
    })
}

/// [J7-3] The lane `lane` beside me runs on `need` m past my front (along its first ways on;
/// `NeighbourView::runs_on` is the network's reach from the lane's start, capped).
fn runs_past(net: &Network, me: &Vehicle, lane: usize, need: f32) -> bool {
    let st = &me.state;
    let mut d = net.lanes[lane].length() - net.beside_s(st.lane, lane, st.s) - st.front;
    let mut cur = lane;
    for _ in 0..16 {
        if d >= need {
            return true;
        }
        let Some(&n) = net.lanes[cur].next.first() else { return false };
        d += net.lanes[n].length();
        cur = n;
    }
    d >= need
}

/// [J7-3] SUMO LC2013's strategic wish to leave a lane that ends for the driver (sumo.md 3.2),
/// towards `nb`: 0 far off, rising with the nearness of the end over the strategic distance
/// (`lane_end_look`) to `lc_strategic` two vehicle lengths before it - towards the lane the
/// merge joins (`LaneEnd::toward`), or for a closed lane or the way's end one that runs on past
/// it and is open to me (`open_on`).
fn lane_end_desire(net: &Network, sim: &Simulation, me: &Vehicle, p: &Perception, nb: &NeighbourView) -> f32 {
    let st = &me.state;
    let Some(e) = p.lane_end else { return 0.0 };
    let suits = match e.kind {
        LaneEndKind::Merge => e.toward == Some(nb.lane),
        LaneEndKind::Closed | LaneEndKind::DeadEnd => runs_past(net, me, nb.lane, e.dist + me.length()) && open_on(net, me, nb.lane, st.s),
    };
    if !suits || !may_change_for_lane_end(net, sim, me, nb) {
        return 0.0;
    }
    // (SUMO LC2013's `currentDistDisallows`: urgent once the distance left per change needed - one
    // here, the lane beside - is under the strategic distance; a step, not a ramp)
    let overtaking = nb.dir == if sim.left_hand { 2 } else { 1 };
    let la = lane_end_look(me.look_ahead_speed.max(st.speed), overtaking, me);
    if e.dist < la {
        me.params.lc_strategic
    } else {
        0.0
    }
}

/// The lane-change desire towards a neighbour, 0..1, and whether the gap there is
/// accepted (LMRS: the route, a standing obstacle, speed gain, keeping to the side).
/// `near_junction`: within the last stretch before a junction, where the lines are solid
/// and only the turn lane and a standing obstacle are reasons to move over.
fn change_desire(net: &Network, sim: &Simulation, me: &Vehicle, p: &Perception, nb: &NeighbourView, near_junction: bool) -> (f32, bool, ChangeWhy) {
    let st = &me.state;
    let dp = &me.params;
    let lane = &net.lanes[st.lane];
    // [J7-3] the lane ends for me: the strategic wish to leave it, whatever else holds
    let end = lane_end_desire(net, sim, me, p, nb);
    // (nor a move that does not fit before the end: SUMO's `LCA_INSUFFICIENT_SPACE`; the car
    // slows to its last-resort stop, which leaves the room to move over from standing)
    // (the move's whole road, `change_space`, within the end, less what the car's stop control
    // lands either side of a stop, `REST_SHORT`: the last-resort stop leaves the space from rest)
    let fits = p.lane_end.is_none_or(|e| e.kind == LaneEndKind::Merge || change_space(st.speed, st.accel) <= e.dist + REST_SHORT);
    let end_ok = |ok: bool| ok && !(end > 0.0 && (people_on_move(net, sim, me, nb) || !fits));
    if !may_change_to(net, me, nb) {
        return (end, end > 0.0 && end_ok(gap_accepted(me, nb)), ChangeWhy::LaneEnd);
    }
    let (mut desire, mut why) = (end, ChangeWhy::LaneEnd);
    let mut wish = |d: f32, w: ChangeWhy| {
        if d > desire {
            desire = d;
            why = w;
        }
    };
    let my_turn = st.turn_ahead(net, TURN_LANE_KEEP).map(|t| t.0).unwrap_or(0);
    if lane.turn == 0 && my_turn != 0 && nb.turn == my_turn && lane.turn != my_turn {
        wish(1.0 * dp.lc_strategic, ChangeWhy::Turn);
    }
    // behind something standing: round it
    if let Some(l) = p.leader {
        let standing = l.standing_obstacle() && l.gap < 25.0;
        if standing && me.stopped >= 4.0 && nb.ahead.map(|a| a.0 > l.gap + l.length + 6.0).unwrap_or(true) {
            wish(0.8, ChangeWhy::Obstacle);
        }
        // a faster lane
        let slow = l.speed < 0.7 * p.v_want && l.gap < 45.0 && l.gap > 8.0 && l.speed < st.speed + 1.0;
        if slow && !near_junction && st.speed >= 4.0 && nb.ahead.map(|a| a.0 > 50.0).unwrap_or(true) && open_on(net, me, nb.lane, st.s) {
            wish(dp.lc_speed_gain, ChangeWhy::SpeedGain);
        }
    }
    // [Q8-3] keep to the side: SUMO LC2013's keepRight layer - the accumulated wish
    // (`Vehicle::keep_right`, `keep_right_rate`) past its threshold, unless SUMO's strategic stays
    // hold (`stays_on_best`, over the lanes open to me: [fix pass 1] no `open_on` window); a speed-
    // gain, obstacle or turn wish, or a lane-end wish over `lc_keep_right`, outweighs it (round 5's gates -
    // lead over 60 m, the lane beside free 70 m, not near a junction, running on 200 m - are gone:
    // how long the driver could keep its speed there is the accumulator's rate)
    if nb.dir == keep_side(sim) && me.keep_right >= KEEP_RIGHT_THRESHOLD && !stays_on_best(net, sim, me, nb) {
        wish(dp.lc_keep_right, ChangeWhy::KeepSide);
    }
    (desire, end_ok(gap_accepted(me, nb)), why)
}

/// [J7-3] The motivation that won a lane change of the driver's own (`change_desire`; counted per
/// start, `Simulation::changes_by_why`, `traffic_cases ab`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeWhy {
    LaneEnd = 0,
    Turn = 1,
    Obstacle = 2,
    SpeedGain = 3,
    KeepSide = 4,
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

/// A keep-to-the-side change needs the lanes the target runs into (one after the other, while
/// each has a single next) open to the car for the move's length and this many metres more (m:
/// what a driver sees of the lane markings ahead; viz5 case04 #5644 changed onto 3470's bus
/// lane, which runs into 3474's, closed to it).
pub const KEEP_SIDE_OPEN_AHEAD: f32 = 60.0;

/// [Q8-3] SUMO LC2013's `KEEP_RIGHT_TIME` (s): "the number of seconds after which a vehicle
/// should move to the right lane" - with the lane beside free for its whole acceptance time
/// the wish reaches its threshold in this long.
pub const KEEP_RIGHT_TIME: f32 = 5.0;
/// [Q8-3] The accumulated keep-right wish (`Vehicle::keep_right`) asks for the change once it
/// reaches this: SUMO's `myKeepRightProbability * lcKeepRight < -myChangeProbThresholdRight`
/// with the default `lcKeepRight` 1, the accumulator counted in units of the threshold.
pub const KEEP_RIGHT_THRESHOLD: f32 = 1.0;
/// [Q8-3] SUMO LC2013's legacy acceptance time (`lcKeepRightAcceptanceTime` -1, its default):
/// `7 * roadSpeedFactor * max(1, v)` seconds.
pub const KEEP_RIGHT_ACCEPTANCE: f32 = 7.0;
/// [Q8-3] SUMO LC2013 accumulates the keep-right wish only while the current lane is not faster
/// than the lane beside by more than this (m/s: `thisLaneVSafe - 5 / 3.6 > neighLaneVSafe`).
pub const KEEP_RIGHT_FASTER: f32 = 5.0 / 3.6;
/// [Q8-3] SUMO LC2013's `TURN_LANE_DIST` (m): "the distance at which a lane leading elsewhere is
/// considered to be a turn-lane that must be avoided" - a driver on the lane that runs on does not
/// leave it for a keep-right onto one that runs on less than this (`stays_on_best`).
pub const TURN_LANE_DIST: f32 = 200.0;
/// [Q8-3] SUMO's `lcStrategicLookahead` (m, default 3000): how far LC2013's best lanes, and so
/// the distance a lane beside runs on (`neighDist`), are looked along.
pub const LC_STRATEGIC_LOOKAHEAD: f32 = 3000.0;
/// [Q8-3] SUMO's `MSVehicle::congested` (m/s, 60 km/h): below it overtaking on the keep side is
/// allowed (`avoidOvertakeRight` is false), above it the time on the lane beside is bounded by
/// catching up with my own leader.
pub const CONGESTED_SPEED: f32 = 60.0 / 3.6;

/// The side a driver keeps to: 2 right, 1 left on a left-hand-traffic map.
fn keep_side(sim: &Simulation) -> i32 {
    if sim.left_hand {
        1
    } else {
        2
    }
}

/// [Q8-3] SUMO LC2013's strategic stays that come before its keep-right layer, for the lane beside
/// `nb` when my own lane runs on further (SUMO: `bestLaneOffset == 0`): VARIANT_21 `stayOnBest`,
/// the lane beside runs on less than `TURN_LANE_DIST`; and "could not change back and forth in
/// time", twice its run is under the strategic distance (`lane_end_look`).
fn stays_on_best(net: &Network, sim: &Simulation, me: &Vehicle, nb: &NeighbourView) -> bool {
    let st = &me.state;
    let overtaking = nb.dir == if sim.left_hand { 2 } else { 1 };
    let short = TURN_LANE_DIST.max(0.5 * lane_end_look(me.look_ahead_speed.max(st.speed), overtaking, me));
    let theirs = open_run(net, me, nb.lane, net.beside_s(st.lane, nb.lane, st.s), short);
    theirs < short && open_run(net, me, st.lane, st.s, short) > theirs
}

/// [Q8-3] How far (m, at most `cap`) a car at `s` on `lane` can drive on along it and the lanes it
/// runs into while they are open to it (`access::is_open`; a junction object's piece is the
/// junction's): SUMO's best-lane length (`neighDist`, `currentDist`) for a car with no route -
/// at a fork the way on that runs furthest. (`NeighbourView::runs_on` is the network's reach,
/// which counts no closed lane out and stops counting at `REACH_MAX`.)
fn open_run(net: &Network, me: &Vehicle, lane: usize, s: f32, cap: f32) -> f32 {
    // (at most `OPEN_RUN_LANES` lanes looked at: past that the way is taken to run on)
    fn walk(net: &Network, me: &Vehicle, lane: usize, d: f32, cap: f32, budget: &mut usize) -> f32 {
        let Some(l) = net.lanes.get(lane) else { return d };
        if *budget == 0 {
            return cap;
        }
        *budget -= 1;
        if l.source != 2 && !crate::access::is_open(l, me) {
            return d;
        }
        let end = d + l.length();
        if end >= cap {
            return cap;
        }
        let mut best = end;
        for &n in &l.next {
            best = best.max(walk(net, me, n, end, cap, budget));
            if best >= cap {
                break;
            }
        }
        best
    }
    let Some(l) = net.lanes.get(lane) else { return 0.0 };
    let mut budget = OPEN_RUN_LANES;
    walk(net, me, lane, -s.min(l.length()), cap, &mut budget).min(cap)
}

/// [Q8-3] `open_run` looks at no more than this many lanes (the network's own reach is counted
/// over the same chains, `Network::compute_reach`).
const OPEN_RUN_LANES: usize = 64;

/// [Q8-3] SUMO's Krauss `maximumSafeFollowSpeed` (on insertion: no reaction step): the highest
/// speed from which the driver, braking at `b` after `tau`, stays behind a lead `gap` m ahead at
/// `vl` m/s braking as hard; a stop `gap` m ahead with `vl` 0 (`maximumSafeStopSpeed`).
fn safe_speed(gap: f32, vl: f32, b: f32, tau: f32) -> f32 {
    let b = b.max(0.1);
    let room = gap.max(0.0) + vl.max(0.0).powi(2) / (2.0 * b);
    -b * tau + ((b * tau).powi(2) + 2.0 * b * room).sqrt()
}

/// [Q8-3] SUMO LC2013's keep-right accumulation per second towards the lane beside on the keep
/// side, `nb` (`MSLCM_LC2013::_wantsChange`, "honor the obligation to keep right"): 0 while my
/// lane is faster than that one by `KEEP_RIGHT_FASTER` (the anticipated safe speeds behind each
/// lane's leader, `anticipateFollowSpeed`); else the share of the acceptance time
/// (`KEEP_RIGHT_ACCEPTANCE`) the driver could drive at its desired speed there - bounded by how far
/// the lane runs on less its braking distance, by closing on a slower leader there to its secure
/// gap, and above `CONGESTED_SPEED` by catching up with my own slower leader (no overtaking on the
/// keep side) - over `KEEP_RIGHT_TIME`.
pub fn keep_right_rate(net: &Network, vehicles: &[Vehicle], me: &Vehicle, p: &Perception, nb: &NeighbourView) -> f32 {
    let st = &me.state;
    let dp = &me.params;
    let (b, tau) = (dp.comfort_brake, dp.tau);
    let v_max = p.v_want.max(0.1);
    // ([fix pass 1] SUMO's `currentDist` and `neighDist`: the best lanes' length, the lanes open to
    // the car, up to `lcStrategicLookahead` - `open_run`, not the network's capped reach)
    let mine = open_run(net, me, st.lane, st.s, LC_STRATEGIC_LOOKAHEAD);
    let runs = open_run(net, me, nb.lane, net.beside_s(st.lane, nb.lane, st.s), LC_STRATEGIC_LOOKAHEAD);
    let this_safe = v_max.min(match p.leader {
        Some(l) => safe_speed(l.gap, l.speed, b, tau),
        None => safe_speed(mine, 0.0, b, tau),
    });
    let nb_max = net.lanes.get(nb.lane).map_or(v_max, |l| ((l.speed_limit_kmh * st.desire).min(st.max_speed_kmh).max(3.0) / 3.6).max(0.1));
    let nb_safe = nb_max.min(match nb.ahead {
        Some((g, vl)) => safe_speed(g, vl, b, tau),
        None => safe_speed(runs, 0.0, b, tau),
    });
    if this_safe - KEEP_RIGHT_FASTER > nb_safe {
        return 0.0;
    }
    let limit = net.lanes.get(st.lane).map_or(v_max, |l| (l.speed_limit_kmh / 3.6).max(0.1));
    let acceptance = KEEP_RIGHT_ACCEPTANCE * (v_max / limit) * st.speed.max(1.0);
    let brake_gap = v_max * tau + v_max * v_max / (2.0 * b.max(0.1));
    let mut full_gap = (runs - brake_gap).max(0.0);
    let mut seconds = acceptance.min(full_gap / v_max);
    if let Some((g, vl)) = nb.ahead.filter(|a| a.1 < v_max) {
        let secure = (v_max * tau + v_max * v_max / (2.0 * b.max(0.1)) - vl.max(0.0).powi(2) / (2.0 * b.max(0.1))).max(0.0);
        full_gap = full_gap.min(g - secure).max(0.0);
        seconds = seconds.min(full_gap / (v_max - vl));
    }
    // ([fix pass 1] SUMO compares my desired speed with the leader's maximum on its lane,
    // `getVehicleMaxSpeed`, not its speed now: a player's vehicle or a parked car is taken at my
    // lane's limit)
    let leader_max = |l: &crate::perception::Headway| match l.what {
        Seen::Vehicle(j) | Seen::Body(j) => vehicles.get(j).map_or(limit, |o| net.lanes.get(o.state.lane).map_or(limit, |ol| (ol.speed_limit_kmh * o.state.desire).min(o.state.max_speed_kmh).max(3.0) / 3.6)),
        Seen::Player(_) | Seen::Parked => limit,
    };
    if let Some(l) = p.leader.filter(|l| st.speed >= CONGESTED_SPEED && leader_max(l) < v_max) {
        full_gap = full_gap.min(l.gap);
        seconds = seconds.min(full_gap / (v_max - l.speed));
    }
    KEEP_RIGHT_THRESHOLD * (seconds / acceptance.max(0.1)) / KEEP_RIGHT_TIME
}

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
fn behind_an_obstacle(net: &Network, sim: &Simulation, vehicles: &[Vehicle], lead: &crate::perception::Headway) -> bool {
    queue_head(net, sim, vehicles, lead).is_some()
}

/// [Q8-1 fix pass 1] What heads the queue the standing `lead` is the tail of (`QUEUE_LOOK`
/// vehicles on at most, linked by their leads; the one definition of `behind_an_obstacle` and the
/// escape, `queue_escape`): `AtStop`, a bus held at its stop; `ForGood`, a vehicle at the end of
/// its way, or one standing `QUEUE_OBSTACLE_STAND` held by nothing - an AI vehicle not indicating,
/// a player's vehicle not indicating and not held by its red (its claim's `light_red`), [fix pass
/// 3] the standing counted as continuous time held by nothing; None, a
/// queue that is the road (a red light, a junction, people: their heads are held by them) or
/// anything moving or manoeuvring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QueueHead {
    AtStop,
    ForGood,
}

/// A lead nearer than this (m) holds the car (`Plan::held`: the start reaction applies when it
/// clears); [Q8-1 follow-up] the same test holds a player's vehicle for `queue_head`.
pub const HELD_LEAD_GAP: f32 = 12.0;

/// [Q8-1 follow-up] Is player `id`'s vehicle blocked by a vehicle ahead - an AI vehicle on a lane it
/// stands on, or on a lane that one runs into, whose rear is within `HELD_LEAD_GAP` of its front -
/// as an AI head is held by a lead near (`Plan::held`)?
fn player_blocked(net: &Network, sim: &Simulation, vehicles: &[Vehicle], id: u32) -> bool {
    let Some(k) = sim.players.iter().position(|p| p.id == id) else { return false };
    let half = sim.players[k].half_len;
    for (&l, list) in &sim.player_occupancy {
        for &(pk, ps, _) in list.iter().filter(|o| o.0 == k) {
            let _ = pk;
            let front = ps + half;
            let ahead = |lane: usize, offset: f32| {
                sim.occupancy.get(&lane).is_some_and(|occ| {
                    occ.iter().any(|o| {
                        let rear = o.s + offset - vehicles.get(o.vehicle).map_or(0.0, |v| v.state.rear);
                        rear >= front - 0.5 && rear - front < HELD_LEAD_GAP
                    })
                })
            };
            let len = net.lanes.get(l).map_or(0.0, |ln| ln.length());
            if ahead(l, 0.0) || net.lanes.get(l).is_some_and(|ln| ln.next.iter().any(|&n| ahead(n, len))) {
                return true;
            }
        }
    }
    false
}

fn queue_head(net: &Network, sim: &Simulation, vehicles: &[Vehicle], lead: &crate::perception::Headway) -> Option<QueueHead> {
    if lead.speed.abs() >= STANDING_SPEED {
        return None;
    }
    // ([fix pass 3] standing for good counts the time standing held by nothing, continuous
    // (`Vehicle::unheld_for`, `Simulation::player_wait`): a head held by its red, people or the
    // junction is not standing for good on the frame it is let go - SUMO's waiting time runs on
    // under people and junction holds)
    let player_head = |id: u32| {
        let held_by_red = sim.claims.iter().any(|(c, _)| crate::sim::claim_player(c.vehicle) == Some(id) && c.light_red);
        let waited = sim.player_wait.get(&id).copied().unwrap_or(0.0);
        let blocked = player_blocked(net, sim, vehicles, id);
        sim.players.iter().find(|p| p.id == id).is_some_and(|p| p.speed.abs() < STANDING_SPEED && waited >= QUEUE_OBSTACLE_STAND && p.blinker == 0 && !held_by_red && !blocked).then_some(QueueHead::ForGood)
    };
    let mut j = match lead.what {
        Seen::Vehicle(j) | Seen::Body(j) => j,
        Seen::Player(id) => return player_head(id),
        Seen::Parked => return None,
    };
    for _ in 0..QUEUE_LOOK {
        let o = vehicles.get(j)?;
        if o.state.speed.abs() >= STANDING_SPEED || o.passing.is_some() || o.state.change.is_some() || o.reversing.is_some() {
            return None;
        }
        let why = o.last.why;
        if o.external.at_stop || why == Why::Service {
            return (o.blinker_shown.0 == 0).then_some(QueueHead::AtStop);
        }
        if why == Why::End || (!o.last.held && why != Why::Lead && o.unheld_for >= QUEUE_OBSTACLE_STAND) {
            return (o.blinker_shown.0 == 0).then_some(QueueHead::ForGood);
        }
        if why != Why::Lead {
            return None;
        }
        if let Some(Seen::Player(id)) = o.last.lead_seen {
            return player_head(id);
        }
        j = o.last.lead_id.and_then(|id| vehicles.iter().position(|v| v.id == id))?;
    }
    None
}

/// [Q8-1] A car stood this long (s) behind a queue whose head is an obstacle backs up to the room
/// a bypass needs (`queue_escape`): as long as anything standing takes to be an obstacle whatever
/// holds it (`perception::HELD_OBSTACLE_STAND`, the cleanup's one definition).
pub const QUEUE_ESCAPE_STAND: f32 = crate::perception::HELD_OBSTACLE_STAND;

/// [Q8-1 fix pass 1] A vehicle's brake gap at `v` (m): SUMO's `MSCFModel::brakeGap` - the
/// reaction distance `v * tau` and the stop `v^2 / 2b` at its comfortable brake.
fn brake_gap(v: f32, tau: f32, b: f32) -> f32 {
    let v = v.abs();
    v * tau + v * v / (2.0 * b.max(0.1))
}

/// [Q8-1 fix pass 1] How far behind me (m) a vehicle can come from before it would stand: the
/// brake gap at my lane's limit of the road preset that brakes longest (the lorry's: tau 1.8 s,
/// 1.5 m/s², `DriverParams::preset`).
fn rear_look(net: &Network, me: &Vehicle) -> f32 {
    let lorry = DriverParams::preset(VehicleClass::Truck);
    let limit = net.lanes.get(me.state.lane).map_or(0.0, |l| l.speed_limit_kmh / 3.6);
    brake_gap(limit, lorry.tau, lorry.comfort_brake)
}

/// [Q8-1] Is the strip behind me clear for backing up `back` m and keeping my standing gap
/// there: my body moved back along my way over every half metre of `back` + `min_gap` touches no
/// vehicle, player's vehicle, parked car or person by `SQUEEZE_CLEARANCE`; [fix pass 1] a vehicle
/// on the move is swept along its way from where it is to where it would stand (its SUMO brake
/// gap, `brake_gap`), and looked for as far back as one can come from (`rear_look`).
fn rear_clear(net: &Network, sim: &Simulation, vehicles: &[Vehicle], me: &Vehicle, back: f32) -> bool {
    let st = &me.state;
    let reach = back + st.min_gap;
    let clearance = SQUEEZE_CLEARANCE as f64;
    let near = sim.vehicles_near(me.pos, (reach + st.rear + rear_look(net, me)) as f64);
    let mut others: Vec<Obb> = Vec::new();
    for &j in &near {
        let o = &vehicles[j];
        if o.id == me.id || (o.pos.z - me.pos.z).abs() > 3.0 {
            continue;
        }
        let v = o.state.speed;
        if v.abs() < STANDING_SPEED {
            others.push(o.obb());
            continue;
        }
        // (from where it is to where it stands, a metre at a time)
        let gap = brake_gap(v, o.params.tau, o.params.comfort_brake) * v.signum();
        let n = (gap.abs().ceil() as usize).max(1);
        for k in 0..=n {
            let d = gap * k as f32 / n as f32;
            let p = o.state.way_point(net, d).truncate();
            let p2 = o.state.way_point(net, d + v.signum()).truncate();
            let f = ((p2 - p) * v.signum() as f64).normalize_or_zero();
            let h = if f.length_squared() < 0.5 { o.heading as f64 } else { f.x.atan2(f.y).to_degrees() };
            others.push(Obb::vehicle(p, h, o.state.front as f64, o.state.rear as f64, o.half_width as f64));
        }
    }
    let people = sim.people_near(me.pos, (reach + st.rear + 5.0) as f64);
    let mut parked: Vec<Obb> = Vec::new();
    for (&l, list) in sim.parked.iter() {
        let Some(lane) = net.lanes.get(l) else { continue };
        for pk in list {
            let (q, h) = lane.at(pk.s.min(lane.length()));
            if (q - me.pos).truncate().length() > (reach + st.rear + 10.0) as f64 {
                continue;
            }
            let hr = (h as f64).to_radians();
            let right = DVec2::new(hr.cos(), -hr.sin());
            parked.push(Obb::vehicle(q.truncate() + right * pk.lateral as f64, h as f64, pk.half_len as f64, pk.half_len as f64, pk.half_width as f64));
        }
    }
    let mut x = 0.5f32.min(reach);
    loop {
        let b = shifted_body(net, me, -x, 0.0);
        if others.iter().chain(parked.iter()).any(|o| o.separation(&b) < clearance) {
            return false;
        }
        if sim.players.iter().any(|pl| (pl.pos.z - me.pos.z).abs() <= 3.0 && pl.obb(0.0, 0.0).separation(&b) < clearance) {
            return false;
        }
        if people.iter().any(|&k| (sim.people[k].pos.z - me.pos.z).abs() <= 2.5 && Obb::point(sim.people[k].pos, PERSON_RADIUS as f64).separation(&b) < clearance) {
            return false;
        }
        if x >= reach {
            return true;
        }
        x = (x + 0.5).min(reach);
    }
}

/// [Q8-1 fix pass 1] Where a car backed up to `s` on its own lane would wait: SUMO's waiting
/// positions (J7-4's `legal_before`), its body on no band of its lane - a pedestrian crossing's
/// zebra band (`ZEBRA_STOP` before the cut to `ZEBRA_PAST` past it), and on a lane that is a
/// junction's movement, its meeting places (from a metre before) and convergences.
fn rest_clear_of_bands(net: &Network, sim: &Simulation, me: &Vehicle, s: f32) -> bool {
    let st = &me.state;
    let (from, to) = (s - st.rear - REST_SHORT, s + st.front);
    let mut bands: Vec<(f32, f32)> = net.walks.get(st.lane).map(|w| w.iter().map(|&(_, sa, _)| (sa - ZEBRA_STOP, sa + ZEBRA_PAST)).collect()).unwrap_or_default();
    if let Some(&(j, m)) = sim.junctions.of_lane.get(&st.lane) {
        let mv = &sim.junctions.nodes[j].movements[m];
        bands.extend(mv.foes.iter().map(|f| (f.mine.0 - 1.0, f.mine.1)));
        bands.extend(mv.converge.iter().map(|&(a, b)| (a - 1.0, b)));
    }
    !bands.iter().any(|b| b.0 < to && b.1 > from)
}

/// [Q8-1] The packed-queue escape: a car that has stood `QUEUE_ESCAPE_STAND` behind a queue whose
/// head stands for good (`queue_head`: `ForGood`, the same head the round-6 keep-back and bypass
/// read), held by nothing but its lead, with a lane beside it may bypass into (`could_bypass`) but
/// nearer its lead than the room the bypass needs (`bypass_room`), backs up to that room when the
/// strip behind holds the move and its standing gap (`rear_clear`; so the rearmost goes first and
/// the queue dissolves from the back), then pulls out with the round-6 bypass. [fix pass 1] Only
/// on a lane of a road (not a junction object's piece), its body staying on its own lane, and
/// to a place where it may wait (`rest_clear_of_bands`). The metres to back up. (SUMO has no
/// reversing: its vehicles are points on a lane that change lanes from any gap; an OMSI car's
/// body must steer round the car in front, and a queue packed to its standing gap behind a
/// vehicle standing for good stood for ever - the user's viz5 case08, five cars 1.9-2.3 m apart
/// behind T11 #44422 for 11 minutes.)
fn queue_escape(net: &Network, sim: &Simulation, vehicles: &[Vehicle], me: &Vehicle, p: &Perception) -> Option<f32> {
    let st = &me.state;
    if st.speed.abs() >= STANDING_SPEED || me.stopped < QUEUE_ESCAPE_STAND || st.change.is_some() || me.passing.is_some() || !st.route.is_empty() || me.is_rail() || me.class == VehicleClass::Air || me.serves_stops() {
        return None;
    }
    if net.lanes.get(st.lane).is_none_or(|l| l.source == 2) {
        return None;
    }
    let l = p.leader?;
    if queue_head(net, sim, vehicles, &l) != Some(QueueHead::ForGood) {
        return None;
    }
    let need = [p.left, p.right].into_iter().flatten().filter(|nb| could_bypass(net, me, nb, &l)).filter_map(|nb| bypass_room(net, me, &nb, &l)).fold(f32::INFINITY, f32::min);
    if !need.is_finite() || l.gap >= need - BYPASS_HOLD_SLACK {
        return None;
    }
    let back = need - l.gap;
    // (the body stays on its own lane, and rests where it may wait)
    if st.s - st.rear - back < 0.0 || !rest_clear_of_bands(net, sim, me, st.s - back) {
        return None;
    }
    rear_clear(net, sim, vehicles, me, back).then_some(back)
}

/// [P8-4] The pull-back into the lane is sampled at this spacing (m), as the bypass's sweep
/// (`bypass_sweep_clear`).
const REENTRY_STEP: f32 = 0.5;

/// [round 8 fix, re-entry] The way back into the lane of a vehicle out of it, as a sideways
/// ramp (`AiState::lateral_ramp`: from, to, start odometer, length): the ramp laid if it ends at
/// the offset the vehicle is asked for (a bay, the bus service sends 0 as it leaves one; else
/// its target), else the S-curve the motion lays for a new target (`AiState` step,
/// `s_curve_len`) from where the body is now. The ramp laid can still be the way IN for a frame:
/// the bay's target turns back before the service's way out is laid, and read as the way back
/// it swept the bay outward and, its start moved on by the hold, put the body on the lane's
/// axis (viz2 case10 bus #31063 at t=1418.2, 3.48 m sideways onto #34056 passing at 4.4 m/s).
pub fn way_back(me: &Vehicle) -> (f32, f32, f32, f32) {
    let st = &me.state;
    let to = me.external.bay.unwrap_or(st.lateral_target);
    let laid = st.lateral_ramp;
    if (laid.1 - to).abs() <= 1e-3 {
        laid
    } else {
        (st.lateral, to, st.odometer, omsi_sim::traffic::s_curve_len(to - st.lateral))
    }
}

/// [P8-4] A vehicle out of its lane on its way back in: its body clear of the lane's axis (a
/// car passing in the lane gets by it), no pass or change of its own, the offset it is asked
/// for (a bay, the bus service sends 0 as it leaves one; else its target) nearer the lane
/// than where it is, and its way back (`way_back`: the ramp the bus service's `depart` or a
/// parked car's pull-out laid, else one from where it is) not run out yet.
fn reentering(me: &Vehicle) -> bool {
    let st = &me.state;
    let (_, to, x0, len) = way_back(me);
    // (not a settled lane change's offset easing in place, `Simulation::sidestep`: it moves
    // sideways standing, not back into the traffic)
    !st.settled_offset && me.passing.is_none() && st.change.is_none() && st.lateral.abs() > me.half_width && to.abs() < st.lateral.abs() && x0 + len > st.odometer
}

/// [P8-4] Is the way back into the lane clear of bodies standing on it? My body on the bent way
/// over what is left of the ramp (`shifted_body`, the way carries the ramp), turned along it as
/// the bypass's sweep, against every other vehicle's and player's box where it is now; a
/// vehicle going my way with its rear ahead of my front is a lead, not a threat. [merge P8/H8]
/// No gap of its own: who comes up behind and must be able to stop for me is the one pull-out
/// rule's (`pull_out_clear`: each follower's own secure gap, the players too); this sweep adds
/// only the bodies the way back in would run into (a van standing beside my nose). (x10 case08:
/// timetable bus #10940 swung out of its bay across the lane beside, into #11059 coming at
/// 8.9 m/s 14 m away, 0.58 m deep.)
fn reentry_clear(net: &Network, sim: &Simulation, vehicles: &[Vehicle], me: &Vehicle) -> bool {
    // ([round 8 fix, re-entry] the poses on the way back, `way_back`: a ramp laid that ends
    // elsewhere is not where the body will move)
    let back = way_back(me);
    let on_way_back;
    let me = if back == me.state.lateral_ramp {
        me
    } else {
        let mut w = me.clone();
        w.state.lateral_curve.clear();
        w.state.lateral_ramp = back;
        on_way_back = w;
        &on_way_back
    };
    let st = &me.state;
    let (_, _, x0, len) = back;
    let left = (x0 + len - st.odometer).max(0.0);
    let n = (left / REENTRY_STEP).ceil() as usize;
    // ([pass 2, A1] the poses it has not reached yet: where it stands now is no move into
    // anybody's way - a car queued behind it stands there already)
    let poses: Vec<Obb> = (1..=n.max(1)).map(|k| shifted_body(net, me, (k as f32 * REENTRY_STEP).min(left), 0.0)).collect();
    let hr = (me.heading as f64).to_radians();
    let fwd = DVec2::new(hr.sin(), hr.cos());
    let reach = (left + st.front + st.rear) as f64;
    for o in vehicles {
        if o.id == me.id || (o.pos.z - me.pos.z).abs() > 3.0 {
            continue;
        }
        if (o.pos - me.pos).truncate().length() > reach + (o.state.front + o.half_width) as f64 {
            continue;
        }
        let oh = (o.heading as f64).to_radians();
        let same_way = DVec2::new(oh.sin(), oh.cos()).dot(fwd) > 0.0;
        if same_way && (o.pos - me.pos).truncate().dot(fwd) - o.state.rear as f64 > st.front as f64 {
            continue;
        }
        let body = o.obb();
        if poses.iter().any(|b| body.separation(b) < 0.0) {
            return false;
        }
    }
    !sim.players.iter().any(|pl| {
        if (pl.pos.z - me.pos.z).abs() > 3.0 {
            return false;
        }
        let b = pl.obb(0.0, 0.0);
        poses.iter().any(|q| b.separation(q) < 0.0)
    })
}

/// What the corner of a body turned on the bypass's curve reaches beyond its shifted box (m):
/// half a metre at 30 degrees for a 2 m front overhang.
pub const BYPASS_TURN_MARGIN: f32 = 0.5;

/// Does the body clear every other body by `SQUEEZE_CLEARANCE` on the bypass's S-curve
/// (`AiState::start_bypass`'s move: `side` metres over `out_len`, the body turned along the
/// curve, not only shifted as `squeeze_clear` has it), up to `level` metres on?
fn bypass_sweep_clear(net: &Network, sim: &Simulation, vehicles: &[Vehicle], me: &Vehicle, side: f32, out_len: f32, level: f32) -> bool {
    body_sweep_clear(net, sim, vehicles, me, &[], &[], side, out_len, level)
}

/// [P8 pass 1, S2] The parked cars within `reach` metres of my origin as boxes, on any lane
/// (one parked at the far kerb of the oncoming lane is in a pass's path), as `squeeze_clear`
/// lays them.
fn parked_boxes(net: &Network, sim: &Simulation, me: &Vehicle, reach: f32) -> Vec<Obb> {
    let mut out = Vec::new();
    for (&l, list) in &sim.parked {
        let Some(lane) = net.lanes.get(l) else { continue };
        for pk in list {
            let (q, h) = lane.at(pk.s.min(lane.length()));
            if (q - me.pos).truncate().length() > (reach + pk.half_len + pk.half_width) as f64 {
                continue;
            }
            let hr = (h as f64).to_radians();
            let right = DVec2::new(hr.cos(), -hr.sin());
            out.push(Obb::vehicle(q.truncate() + right * pk.lateral as f64, h as f64, pk.half_len as f64, pk.half_len as f64, pk.half_width as f64));
        }
    }
    out
}

/// `bypass_sweep_clear` against parked cars (`parked`) and people's discs (`people`) as well.
#[allow(clippy::too_many_arguments)]
fn body_sweep_clear(net: &Network, sim: &Simulation, vehicles: &[Vehicle], me: &Vehicle, parked: &[Obb], people: &[Obb], side: f32, out_len: f32, level: f32) -> bool {
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
        if parked.iter().chain(people).any(|o| o.separation(&b) < clearance) {
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
/// on green; on amber the driver asks every frame whether it can stop with comfortable
/// braking (`stops_comfortably`) - if not it goes, if so it stops ([J8-2] SUMO re-tests it every
/// step; the last amber answer is kept as `Vehicle::light_decision`); on red only a
/// car that cannot stop at all (`cannot_stop`, `HARD_BRAKE`) or one committed at amber onset
/// and still rolling crosses. Nothing is decided on green, no decision is extended past
/// what the onset allowed, and a committed car that came to a stand waits for its green
/// (round 6 J3: one standing short of its line set off over it on red, viz case07 #18558).
/// (This replaces round 4's amber mark: taken on green 3 m out, extended 6 m on yellow,
/// carried over frames and retired by `stale_amber` and a 2 s stand; x10 case03 #3145
/// crossed its line on red on it.) [P8 map data] "Cannot stop" is reckoned to the line
/// itself, as SUMO's red: the half metre past the line it allowed (round 4's `possible -
/// 0.5`, carried into round 7's D2) had no source and let a stop lie over the line.
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
            // ([J8-2] SUMO's yellow rule, re-tested every step (`MSVehicle::planMoveInternal`:
            // `yellow && canBrake`, canBrake = the brake gap at the vehicle's decel - here its
            // comfortable brake or `--tls.yellow.min-decel` 3 m/s², whichever is more - within the
            // distance to the line): a car that slowed for anything and can then stop, stops; the
            // decision is not kept from the amber onset. The line is the stop place: no 0.5 m past
            // it. viz3 case18 #52423 decided "go" once at the onset 26 m out, braked for people and
            // crossed at red at 2 m/s)
            let _ = held;
            let go = v > STANDING_SPEED && !queued && !stops_comfortably(v, gap, dp) && (exit_free || cannot_stop(v, gap, HARD_BRAKE));
            LightCall { go: go && v > STANDING_SPEED, decision: Some((light, go, gap)) }
        }
        Aspect::Red | Aspect::RedYellow => {
            let committed = held == Some(true) && v > STANDING_SPEED;
            LightCall { go: committed || cannot_stop(v, gap, HARD_BRAKE), decision: held.map(|h| (light, h, gap)) }
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
pub(crate) const REST_SHORT: f32 = 1.0;

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

/// [round6 J3] Under this speed (m/s) an eased-off car stands, and is pinned where it is
/// (`Claim::standing`'s 0.3 m/s).
const EASE_STANDING: f32 = 0.3;

/// [J7-4] Where a car its grant refuses stops (a distance from its origin), one rule: refused
/// for a foe, just before the first stretch it waits for (`Grant::at`), pulled back before every
/// other meeting place, pedestrian crossing or convergence its body would stand in
/// (`hold_point`) - SUMO's waiting position before the conflict on the internal lane - then off
/// any crossing band its body would stand over (`off_crossing`); held for anything else (a full
/// exit, SUMO's keep-clear), at the line where it can still stop there gently, else a metre short
/// of the first meeting place of its way (`first`). A foe it is the follower or the leader of
/// lets it go (no stop), a zipper caps its speed (`Grant::zipper`). (Round 6's special cases are
/// gone: the roll-up to a far foe standing for good and the side-merge roll-up behind a vehicle
/// standing there are this rule itself - the zipper takes the side merges - and the
/// `CONVERGE_LOOK` choice between the line and the stretch is `hold_point`'s convergence bands.)
fn refused_stop(net: &Network, sim: &Simulation, me: &Vehicle, j: &JunctionView, g: &Grant) -> f32 {
    let st = &me.state;
    let dp = &me.params;
    let v = st.speed;
    let at_line = j.line - 0.3;
    let gently = v * v / (2.0 * (at_line - st.front - 0.6).max(0.05)) <= dp.comfort_brake * 1.5;
    let line_stop = !j.inside && (gently || v < 0.5);
    let bands = wait_bands(net, sim, j);
    let first = if line_stop { at_line } else { j.conflict_start - 1.0 };
    // ([fix pass 2] inside and past that metre already: the first band still ahead of the front -
    // a car held for a full exit with the first meeting place behind it was stopped where it
    // was and rolled onto the walk ahead, viz case11 #38790/#38867)
    let first = if first < st.front - 0.3 { bands.iter().map(|b| b.0).filter(|&a| a > st.front - 0.3).fold(f32::INFINITY, f32::min).min(j.conflict_end) } else { first };
    let target = match (g.hold, g.at) {
        (Some(Hold::Foe), Some(waited)) => j.entry + waited - 1.0,
        _ => first,
    };
    // the line (SUMO's link end) while the car is short of it, else the last spot short of the
    // target where the body lies on no band; none left (the body is over a band it cannot leave
    // before the target - the resolver lets such a car go, `Claim::next_wait`, unless the foe is
    // in the meeting place already): the hold point as before
    // ([fix pass 2] the line too only where the body there lies on no band: a car held for a
    // full exit stood at its line over a walk, viz case11 #38790/#38867)
    let len = me.length();
    let line = (line_stop && at_line <= target && !bands.iter().any(|b| b.0 < at_line && b.1 > at_line - len - REST_SHORT)).then_some(at_line);
    match legal_before(&bands, me, target) {
        Some(p) => line.map_or(p, |l| p.max(l)),
        // ([fix pass 2] held for anything but a foe - a full exit -, past its last place to wait:
        // on to the next one, `next_wait`, as the claim's commitment says; the lead still stops
        // it behind a queue)
        None if g.hold != Some(Hold::Foe) => line.unwrap_or_else(|| next_wait(net, sim, me, j, if j.inside { st.decel * 1.5 } else { HARD_BRAKE })),
        None => line.unwrap_or_else(|| hold_point(net, sim, me, j, target - j.entry + 1.0)),
    }
}

/// [J7-4, fix pass 1] The bands a waiting car's body may not lie on (SUMO waits at link ends and
/// internal junctions, never on a crossing link): the crossings with other ways (`Cross`, `Block`
/// meeting places, from a metre before), the pedestrian crossings and the convergences - not the
/// merges, where the zipper puts the two one behind the other ([J7-2]; a queue stands in a side
/// merge as in a lane, viz4 case02's twin chains).
pub(crate) fn wait_bands(net: &Network, sim: &Simulation, j: &JunctionView) -> Vec<(f32, f32)> {
    let mut bands: Vec<(f32, f32)> = sim.junctions.foes_along(j.junction, &j.chain).iter().filter(|f| f.kind != crate::junction::FoeKind::Merge).map(|f| (j.entry + f.mine.0 - 1.0, j.entry + f.mine.1)).collect();
    bands.extend(walk_bands(net, sim, j));
    bands.extend(converge_bands(sim, j).into_iter().map(|(a, b, _)| (a, b)));
    bands
}

/// [J7-4, fix pass 1] SUMO's waiting positions: the last place at or before `limit` (the front's
/// distance from the origin, no further back than where the front is) where the body lies on no
/// band of the way through (`wait_bands`). None: no such place ahead of the front.
pub fn legal_before(bands: &[(f32, f32)], me: &Vehicle, limit: f32) -> Option<f32> {
    let front = me.state.front;
    let len = me.length();
    let mut p = limit;
    for _ in 0..64 {
        if p < front - 0.3 {
            return None;
        }
        match bands.iter().filter(|b| b.0 < p && b.1 > p - len - REST_SHORT).map(|b| b.0).fold(f32::INFINITY, f32::min) {
            x if x.is_finite() => p = x,
            _ => return Some(p),
        }
    }
    None
}

/// [J7-4, fix pass 1] SUMO's waiting positions: the first place the car can come to rest at
/// with its body on no band of the way through (`wait_bands`), braking at `brake` (the claim's,
/// `Simulation::tick`) - a distance from the origin. Until there it cannot wait: it is committed
/// to every meeting place short of it (`Claim::next_wait`, SUMO's `willPass`).
pub fn next_wait(net: &Network, sim: &Simulation, me: &Vehicle, j: &JunctionView, brake: f32) -> f32 {
    let st = &me.state;
    // ([fix pass 2] from its stopping place at any speed - its go took it there: a car standing
    // with its body on a band cannot wait there, viz case04 #9059 stood 435 frames on a walk; the
    // resolver still holds it for a foe in or arriving at the meeting place, `Claim::next_wait`)
    let len = me.length();
    let bands = wait_bands(net, sim, j);
    let mut p = st.front + st.speed * st.speed / (2.0 * brake.max(0.1));
    for _ in 0..64 {
        match bands.iter().filter(|b| b.0 < p && b.1 > p - len - REST_SHORT).map(|b| b.1).fold(f32::NEG_INFINITY, f32::max) {
            x if x.is_finite() => p = x + len + REST_SHORT,
            _ => return p,
        }
    }
    p
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
            let at = refused_stop(net, sim, me, j, &g);
            if let (Some((gap, speed)), Some(w)) = (g.follow, g.wait_for.filter(|&w| w < vehicles.len())) {
                // eased off for the other of a mutual hold rolling on into a merge of the two
                // ahead: in behind it at my gap (`reservation::eased_follow`)
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

    // [J7-2] the zipper speed (SUMO's `getZipperSpeed`) for the foe ahead at a merge that crosses
    // nothing: whatever the grant otherwise says, a cap on the acceleration - the gentler of
    // following that foe where it stands along the merged way (the model's own answer for it as a
    // lead) and arriving `tau` after its rear, no harder than the blended brake - or, the foe
    // standing at the join, the queue-entry stop behind it (my lead)
    if let Some(z) = grant.and_then(|g| g.zipper) {
        let behind = Lead { gap: z.gap, speed: z.speed, acc: 0.0 };
        if z.stop {
            if out.lead.map(|l| z.gap < l.gap).unwrap_or(true) {
                out.lead = Some(behind);
                out.lead_seen = (z.foe < vehicles.len()).then_some(Seen::Vehicle(z.foe));
            }
        } else {
            let follow = st.desired_accel(net, Some(behind), None);
            let cap = follow.max(z.arrive).max(z.floor);
            if cap < st.accel {
                out.accel_cap = Some(out.accel_cap.map_or(cap, |c| c.min(cap)));
            }
        }
        // [fix pass 2] the position bound: short of the stretch until the foe's rear has been past
        // its start `tau`, the car waits at the start (`ZipCap::hold`), at a waiting position
        if let (Some(d), Some(j)) = (z.hold, p.junction.as_ref()) {
            if let Some(at) = legal_before(&wait_bands(net, sim, j), me, st.front + d) {
                min_stop(&mut out.stop, &mut out.why, Some(at.max(st.front - 0.3)), Why::Junction);
            }
        }
        // [J7-4, fix pass 1] SUMO's waiting positions: where following the foe ahead brings the car
        // to rest - behind it at the join, or, the foe in the stretch, behind where it comes to
        // rest itself while it is coming to rest (`QUEUE_COMING`, `QUEUE_BRAKE`) - it rests at the
        // last place short of that with its body on no band, never on a crossing (viz case17
        // #60670 stood on a cut). (Not behind a foe still short of the join in its own lane: that
        // one is no lead, the cap alone follows it; a taxi stood 26 s for one, viz5 case04 #15985)
        if let Some(j) = p.junction.as_ref().filter(|_| z.stop || (z.inside && z.speed < QUEUE_COMING)) {
            let rest = st.front + z.gap + z.speed * z.speed / (2.0 * QUEUE_BRAKE) - st.min_gap;
            if let Some(at) = legal_before(&wait_bands(net, sim, j), me, rest) {
                min_stop(&mut out.stop, &mut out.why, Some(at.max(st.front - 0.3)), Why::Junction);
            }
        }
    }

    // [J7-3] asked to make room for a car whose lane ends beside me (SUMO's cooperative
    // follower, `informFollower`): its rear my lead, braking no harder than my comfortable
    // brake times `HELP_DECEL_FACTOR` (`Vehicle::help_for`)
    // ([fix pass 1] it keeps its own standing gap behind the place the changer's acceptance
    // needs it to be: its lead is the changer's rear less that gap - following the rear itself it
    // came to rest at its standing gap, which the acceptance at rest never takes, and both stood)
    if let Some((_, gap, v_r, need)) = me.help_for.filter(|h| h.3 <= 0.0) {
        // (asked only not to speed up: the changer is faster and cuts in as the gap opens)
        let _ = (gap, v_r, need);
        out.accel_cap = Some(out.accel_cap.map_or(0.0, |c| c.min(0.0)));
    } else if let Some((_, gap, v_r, need)) = me.help_for {
        // (it rests at its own standing gap behind the place the acceptance needs, so the gap it
        // opens is the one the changer accepts, `gap_needed_behind`)
        let lead = Lead { gap: (gap - (need - st.min_gap)).max(0.05), speed: v_r, acc: 0.0 };
        let cap = st.desired_accel(net, Some(lead), None).max(-dp.comfort_brake * HELP_DECEL_FACTOR);
        if cap < st.accel {
            out.accel_cap = Some(out.accel_cap.map_or(cap, |c| c.min(cap)));
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

    // 4. people ([round 8 H8-8, fix pass 2] a pull-out too: `people_ahead` leaves out only the
    // people under or beside its parked body off its way out, `pull_out_exempt`)
    min_stop(&mut out.stop, &mut out.why, p.people_stop, Why::People);

    // 5. the app's own holds
    min_stop(&mut out.stop, &mut out.why, me.external.stop_at, Why::Service);
    // [round 8 H8-1] a car leaving a parking space starts only when its lane has room behind
    // it: the follower there not within the gap a lane change needs from it (`gap_needed_behind`:
    // its standing gap, the time gap and the braking distance of the speed it closes in at) -
    // SUMO's parked vehicle re-enters by insertion, which checks the follower's secure gap on the
    // lane it enters (`MSLane::isInsertionSuccess`, `getSecureGap`). Until then it stays parked
    // and its clock waits (x10 case08: #11387 pulled out with bus #11125 coming up at 12 m/s)
    // [P8-4, merge P8/H8] standing out of the lane and about to move back in - a parked car
    // pulling out, a bus leaving its bay (`reentering`) - is the same case: one rule,
    // `pull_out_clear` (every follower up its lane, the players too, beyond its own secure gap, a
    // standing one reaching only its own body; nobody on the strip its body sweeps), over what is
    // left of its way out, and no body standing on that way (`reentry_clear`)
    let leaving = v < STANDING_SPEED && (me.pulling_out || reentering(me));
    let leave_held = leaving && {
        let on_lane = |l: usize| -> Vec<(f32, f32, f32, DriverParams)> {
            sim.occupancy.get(&l).map(|v| v.as_slice()).unwrap_or(&[]).iter().filter(|o| o.vehicle != i).filter_map(|o| vehicles.get(o.vehicle).map(|x| (o.s, x.state.front, x.state.speed, x.params))).collect()
        };
        let followers = pull_out_followers(net, sim, st.lane, st.s, st.rear, &on_lane);
        let ramp = if me.pulling_out { bypass_min_len(me.length(), st.lateral) } else { pull_out_ramp_left(me) };
        !pull_out_clear(net, sim, &followers, st.lane, st.s, st.lateral, st.front, st.rear, me.half_width, ramp) || !reentry_clear(net, sim, vehicles, me)
    };
    let pull_out_held = leave_held && me.pulling_out;
    if me.pull_out > 0.0 || leave_held {
        min_stop(&mut out.stop, &mut out.why, Some(st.front + 0.1), Why::PullOut);
    }
    out.pull_out_held = pull_out_held;
    // (held out of its lane on the way back in, not parked: the ramp back waits for the hold)
    out.reentry_held = leave_held && !me.pulling_out;
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

    // [Q8-1] backing up out of a packed queue (`queue_escape`): [fix pass 1] to the place fixed when
    // it began, the speed braking to it (`AiState::reverse_step`); every frame the strip behind is
    // swept again (`rear_clear`) and the queue looked at: any change and the car stops, a committed
    // stop over its brake distance from where it is, kept to the end
    if let Some((target, stopping)) = me.reversing {
        let left = st.odometer - target;
        let queue_stands = p.leader.is_none_or(|l| l.speed.abs() < STANDING_SPEED);
        let go_on = !stopping && left > 0.0 && queue_stands && rear_clear(net, sim, vehicles, me, left);
        out.reverse = Some(if stopping || go_on { (target, stopping) } else { (st.odometer - st.speed * st.speed / (2.0 * st.decel.max(0.5)), true) });
        out.held = true;
        out.why = if out.why == Why::None { Why::Lead } else { out.why };
        return out;
    }
    if out.stop.is_none() && out.start.is_none() && p.people_stop.is_none() {
        if let Some(back) = queue_escape(net, sim, vehicles, me, p) {
            out.reverse = Some((st.odometer - back, false));
            out.held = true;
            out.why = if out.why == Why::None { Why::Lead } else { out.why };
            return out;
        }
    }

    // [Q8-3] SUMO LC2013's keep-right wish accumulates every step the driver is not changing
    // lanes (`wantsChange` runs each step), towards a lane beside on the keep side it may change
    // onto
    if st.change.is_none() && me.passing.is_none() && !me.is_rail() && st.route.is_empty() {
        // (a strategic stay comes first and leaves the wish as it is)
        if let Some(nb) = [p.left, p.right].into_iter().flatten().find(|nb| nb.dir == keep_side(sim) && may_change_to(net, me, nb) && !stays_on_best(net, sim, me, nb)) {
            out.keep_right_rate = Some(keep_right_rate(net, vehicles, me, p, &nb));
        }
    }

    // 7. a lane change of the driver's own: round something standing, for speed, to keep
    //    to the side, into the lane of the turn ahead
    let free_to_change = st.change.is_none() && st.route.is_empty() && st.change_cooldown <= 0.0 && me.passing.is_none() && out.start.is_none() && !me.is_rail() && lane.length() - st.s > 9.0;
    if free_to_change {
        let near_junction = p.junction.as_ref().map(|j| !j.inside && j.line < JUNCTION_KEEP).unwrap_or(false);
        let mut best: Option<(f32, NeighbourView, ChangeWhy)> = None;
        // [J7-3] the strongest lane-end wish whose gap is not there
        let mut blocked_end: Option<(f32, NeighbourView)> = None;
        for nb in [p.left, p.right].into_iter().flatten() {
            let (d, ok, why) = change_desire(net, sim, me, p, &nb, near_junction);
            if ok && d > 0.0 && best.map(|b| d > b.0).unwrap_or(true) {
                best = Some((d, nb, why));
            }
            let end = lane_end_desire(net, sim, me, p, &nb);
            if !ok && end > 0.0 && blocked_end.map(|b| end > b.0).unwrap_or(true) {
                blocked_end = Some((end, nb));
            }
        }
        let threshold = 0.4 - 0.2 * me.impatience.value;
        if let Some((_, nb, why)) = best.filter(|b| b.0 >= threshold) {
            let standing = p.leader.filter(|l| l.standing_obstacle() && l.gap < 25.0);
            out.start = match standing {
                Some(l) if v < 1.0 => bypass_start(net, sim, vehicles, me, p, &nb, &l),
                _ => Some(Start::Change { to: nb.lane, dir: nb.dir }),
            };
            out.change_why = Some(why);
        } else if let Some((_, nb)) = blocked_end.filter(|b| b.0 >= threshold) {
            // [J7-3] SUMO LC2013's cooperation (`informFollower`, sumo.md 3.3): the follower there,
            // when only it is in the way (room ahead there), is asked to make room
            // the gap my acceptance needs from it (`gap_needed_behind`); one level with me or ahead
            // (no gap behind my rear) is not asked to brake: it must be overtaken - I slow so it
            // passes by the lane's end (LC2013's `needDV = overtakeDist / remainingSeconds`)
            let (ok_behind, ok_ahead) = gap_parts(me, &nb);
            // (to fall in behind a vehicle there: how far I must drop back, and its speed)
            let mut drop_back: Option<(f32, f32)> = None;
            if let (false, Some((g, va))) = (ok_ahead, nb.ahead) {
                // the one ahead there too close (LC2013 `informLeader`: I cannot overtake it, I
                // stay behind it)
                drop_back = Some(((2.0 + (v - va).max(0.0) * 1.5) - g, va));
            } else if let (false, Some(f), Some((gap, vb))) = (ok_behind, nb.behind_id, nb.behind) {
                // ([fix pass 3] SUMO's `neededGap`/`decelGap` test before the `helpDecel` branch:
                // asked to brake only if it can still open the gap from where it is - braking at
                // its comfortable brake (times `HELP_DECEL_FACTOR`) to rest while I go on to my
                // last stop before the lane's end, the gap at rest is then its standing gap;
                // else it is told not to brake and I change in behind it. Asked when it could
                // not, the second follower of a chain stood 1.56 m behind my rear against the
                // 1.82 m the acceptance needs at rest, and all three stood, review r7j)
                let reach = p.lane_end.map_or(f32::INFINITY, |e| {
                    let side = (lane.width + net.lanes[nb.lane].width) * 0.5;
                    (e.dist - change_space(0.0, st.accel).max(bypass_min_len(me.length(), side))).max(0.0)
                });
                let b_f = vehicles.get(f).map_or(dp.comfort_brake, |o| o.params.comfort_brake) * HELP_DECEL_FACTOR;
                let can_open = gap + reach - vb * vb / (2.0 * b_f.max(0.1)) >= nb.behind_min_gap;
                if gap > 0.0 && can_open {
                    // (SUMO's first branch: faster than the follower so that the gap opens by itself
                    // before the lane ends, I cut in - it is only asked not to speed up, `need` 0;
                    // else it is asked to open the gap)
                    let need = gap_needed_behind(me, &nb, vb, v);
                    let opens = p.lane_end.is_some_and(|e| v > vb && gap + (v - vb) * e.dist / v.max(STANDING_SPEED) >= need);
                    let need = if opens { 0.0 } else { need };
                    out.help = Some((f, gap, v, need));
                } else if let Some(o) = vehicles.get(f) {
                    drop_back = Some((me.length() + o.length() + gap + st.min_gap, vb));
                }
            }
            if let (Some((overtake, v_other)), Some(e)) = (drop_back, p.lane_end) {
                let remaining = e.dist / v.max(STANDING_SPEED);
                let target = (v_other - overtake / remaining).max(0.0);
                if target < v {
                    let cap = ((target - v) / st.headway.max(0.1)).max(-dp.comfort_brake * HELP_DECEL_FACTOR);
                    out.accel_cap = Some(out.accel_cap.map_or(cap, |c| c.min(cap)));
                }
            }
            // ... and the last resort where the lane really ends (a merge is the zipper's): the
            // car stops before the end with the room to move over from standing (the cleanup's
            // keep-back for a bypass: a lane change's shortest move, or what the body needs to
            // steer the lane's width over, whichever is longer) - braking for it only once it
            // has to, within its comfortable stopping distance (SUMO's vehicle stops at a dead
            // lane's end as late as its model lets it: braking earlier, it slowed beside the
            // follower that braked to make room, and the gap never opened)
            // ([Q8-4] once placed it is kept - a latch, `Vehicle::lane_end_stop` - until the change
            // starts or the car stands: placed and dropped by the brake test frame by frame, it
            // flicked on and off in the last metre, an 8 m/s² brake-light flicker)
            if let Some(e) = p.lane_end.filter(|e| e.kind != LaneEndKind::Merge) {
                let side = (lane.width + net.lanes[nb.lane].width) * 0.5;
                let room = change_space(0.0, st.accel).max(bypass_min_len(me.length(), side));
                let latched = me.lane_end_stop.filter(|_| v >= STANDING_SPEED).map(|o| st.front + (o - st.odometer));
                let at = latched.unwrap_or(st.front + e.dist - room);
                if latched.is_some() || (at > st.front - 0.3 && at - st.front <= v * v / (2.0 * dp.comfort_brake)) {
                    min_stop(&mut out.stop, &mut out.why, Some(at), Why::Merge);
                    out.lane_end_stop = Some(st.odometer + (at - st.front));
                }
            }
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
                out.accel_cap = Some(out.accel_cap.map_or(omsi_sim::ai_motion::PULL_OUT_ACCEL, |c| c.min(omsi_sim::ai_motion::PULL_OUT_ACCEL)));
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
    } else if let Some(l) = p.leader.filter(|l| l.standing_obstacle() || pausing_player(l) || behind_an_obstacle(net, sim, vehicles, l)) {
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
        // [P8-2] a pass on the oncoming lane starts only when its path is swept as the
        // bypass's is (`bypass_sweep_clear`): the body turned along its S-curve (the out
        // length it will drive) out and level with what it passes, clear of every vehicle's
        // and player's body by `SQUEEZE_CLEARANCE` - not on the oncoming lane's gap alone.
        // Only the start: the keep-back still reads the pass as possible (gating it too made a
        // car close up and later start where it should not - round4b x6), and not the
        // squeeze's linear corridor (`squeeze_clear`: from a queue gap behind a parked car its
        // straight ramp cut the car passed and the pass never came - adversarial b1c).
        let pass_out_len = if too_close { (l.gap - 0.3).clamp(2.0, out_len) } else { out_len };
        // ([pass 1, S2] and the parked cars along the way and the people near it, as the
        // squeeze's corridor has them)
        let pass_swept = || {
            if squeeze {
                return true;
            }
            let level = real + obstacle_len + st.front + st.rear + 2.0;
            let parked = parked_boxes(net, sim, me, level + st.front);
            let people: Vec<Obb> = sim.people_near(me.pos, (level + st.front) as f64).into_iter().map(|k| &sim.people[k]).filter(|q| (q.pos.z - me.pos.z).abs() <= 2.5).map(|q| Obb::point(q.pos, PERSON_RADIUS as f64)).collect();
            body_sweep_clear(net, sim, vehicles, me, &parked, &people, side, pass_out_len, level)
        };
        if pass_possible && ready && pass_swept() {
            let out_len = pass_out_len;
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

    // 10b. [round 8 H8-7, fix pass 1] the edge of the loaded tiles (`Lane::loaded_edge`): the
    //      road goes on into a tile that is not loaded; the car stops before the lane's end, as a
    //      bus at its loaded route's end, and the game takes it off once nobody sees it (a
    //      dead-end removal) or lets it on when the tile comes
    //      [round 9 A2 fix pass 2] unless the edge is its trip's end (`External::edge_trip_end`,
    //      the app's: beyond the horizon, lamps dark): then it drives off the lane's end as at the
    //      network's end, still moving, and no brake lamp lights
    if let Some(&(el, d0)) = p.way.iter().filter(|_| !me.external.edge_trip_end).find(|&&(l, _)| net.lanes.get(l).is_some_and(|x| x.loaded_edge && x.next.is_empty())) {
        let end = d0 + net.lanes[el].length() - 0.5;
        min_stop(&mut out.stop, &mut out.why, Some(end.max(st.front - 0.3)), Why::End);
    }

    // 11. keep the box clear, on every aspect (SUMO's keep-clear rule, sumo.md 1.6: a vehicle does
    //     not drive onto a junction where it would get stuck): a car short of its junction's lit
    //     line does not cross it while where it would come to rest lies between the line and the
    //     way's exit and is no waiting place - behind a lead standing or coming to rest there (one
    //     car to a waiting place), or its own stop (a refusal, people) with its body on a band (a
    //     crossing meeting place, a walk, a convergence: `wait_bands`), where it would block a
    //     foe's way or a crossing. Its own stop at a waiting place inside (the body on no band:
    //     `refused_stop`'s waiting place, or the zebra stop short of a walk) is a legal place:
    //     SUMO's vehicle enters on green and waits at the internal junction (sumo.md 1.2), or
    //     before the crossing on its internal lane (sumo.md 4.3), and at the end of its green it
    //     clears first (`reservation::will_pass`). Held, it waits at the line,
    //     pulled back before a walk there (viz5 case05 #16535 crossed on green into a box its
    //     people stop lay in and stood 2.8 m in, over the crossing). A way through that crosses
    //     nothing - no crossing meeting place, walk or convergence, only merges, which the zipper
    //     orders - has nothing to keep clear: SUMO applies no keep-clear at such geometry-like
    //     nodes (sumo.md 1.6), and its queue stands in it as in a lane (viz5 case10: cars
    //     held at the line of a merge-only join behind a queue standing in it).
    if let (Some(j), Some(sig)) = (p.junction.as_ref(), p.signal.as_ref()) {
        let line = sig.dist;
        let bands = if !j.inside && sig.dist <= j.light_line.max(j.conflict_start) + 1.0 && line - st.front > 0.3 { wait_bands(net, sim, j) } else { Vec::new() };
        if !bands.is_empty() {
            let lead_rest = p.leader.filter(|l| l.speed < QUEUE_SPEED || l.rest.is_some()).map(|l| st.front + l.gap.min(l.rest.map(|r| r - st.front).unwrap_or(f32::INFINITY)));
            let len = me.length();
            let waiting_place = |r: f32| !bands.iter().any(|b| b.0 < r && b.1 > r - len - REST_SHORT);
            let own = out.stop.filter(|&r| !(matches!(out.why, Why::Junction | Why::People) && waiting_place(r)));
            let rest = [own, lead_rest].into_iter().flatten().fold(f32::INFINITY, f32::min);
            let exit = j.entry + j.way_len;
            if rest > line + 0.5 && rest < exit {
                let hold = pulled_back(&walk_bands(net, sim, j), me, line);
                out.stop = Some(hold);
                out.why = Why::KeepClear;
            }
        }
    }

    out.held = out.stop.is_some() || p.leader.map(|l| l.gap < HELD_LEAD_GAP).unwrap_or(false);
    if out.why == Why::None && out.lead.is_some() {
        out.why = Why::Lead;
    }
    out
}
