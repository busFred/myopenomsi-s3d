//! The decision: from what the driver sees to one plan for this frame. A plan is the
//! lead to follow, a stop point, a lateral target and at most one manoeuvre to start.
//! No plan reads another vehicle's plan; the cross-vehicle inputs are the junction grant and
//! the follower's request to make room (`Vehicle::help_for`, set by the last frame's acts).

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
    /// A pedestrian at a crossing: not produced (`People` holds for them); kept for the
    /// recordings that name it.
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
    /// Round something standing closer than the usual 8 m move: over `out_len` m, at full lock
    /// (`full_lock_len`) where the smoothstep cannot steer it (`bypass_min_len`); round the
    /// vehicle `round` (`LaneChange::round`: `None` round a parked car).
    Bypass { to: usize, dir: i32, out_len: f32, full_lock: bool, round: Option<u64> },
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
    /// SUMO LC2013's cooperation (`informFollower`): the follower in the lane I must
    /// change into is asked to make room - (its index, the gap from its front to my rear, my
    /// speed, the gap my acceptance needs from it; `Vehicle::help_for`).
    pub help: Option<(usize, f32, f32, f32)>,
    /// What the lane change of the driver's own it starts is for (`ChangeWhy`).
    pub change_why: Option<ChangeWhy>,
    /// The lane change of the driver's own that signals is dropped before it moves over: its
    /// gap rule fails again (`gap_accepted` against `change_target_view`) or somebody is on the
    /// move (`people_on_move`; counted, `sim::CHANGES_ABORTED`).
    pub abort_change: bool,
    /// The last-resort stop before a lane's end placed this frame (the odometer where the
    /// front stops, `Vehicle::lane_end_stop`).
    pub lane_end_stop: Option<f32>,
    /// Backing up out of a packed queue (`queue_escape`): (the odometer where it
    /// is to stand, stopping: a committed stop - the strip behind no longer clear or the queue
    /// moving).
    pub reverse: Option<(f32, bool)>,
    /// SUMO LC2013's keep-right accumulation this frame (per second, `keep_right_rate`):
    /// `Vehicle::keep_right` grows by it; None where no lane beside on the keep side is one the
    /// driver may change onto (the wish is left as it is).
    pub keep_right_rate: Option<f32>,
    /// SUMO LC2013's speed-gain step this frame per side ([0] left, [1] right; `speed_gain_step`):
    /// how `Vehicle::speed_gain` changes; None where no lane lies beside on that side.
    pub speed_gain_step: [Option<SpeedGainStep>; 2],
    /// A car leaving a parking space is held where it is parked because its
    /// lane has no room behind it: its `pull_out` clock waits.
    pub pull_out_held: bool,
    /// Held standing out of its lane on the way back in (`reentry_clear`):
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
/// (the follower need not brake hard). (`informFollower` asks for it, .)
fn gap_needed_behind(me: &Vehicle, nb: &NeighbourView, vb: f32, v: f32) -> f32 {
    accepted_gap(me, nb.behind_min_gap, vb, v)
}

/// The gap (m) I accept in front of a follower whose standing gap is `min_gap`, coming at `vb`
/// m/s while I go at `v`: its standing gap, my accepted time gap at its speed and what it
/// sheds braking at my comfort to my speed (`gap_needed_behind`'s; the way back
/// into the lane from out of it asks each follower's own secure gap instead, `pull_out_clear`).
pub(crate) fn accepted_gap(me: &Vehicle, min_gap: f32, vb: f32, v: f32) -> f32 {
    let dp = &me.params;
    // (the follower's own standing gap, not a fixed 2 m: the helper opens exactly
    // this gap, `informFollower`, one function on both sides; a follower resting at its 1.57 m
    // standing gap behind the changer was never accepted, and both stood)
    min_gap + me.impatience.gap(dp.lc_gap) * vb + (vb - v).max(0.0).powi(2) / (2.0 * dp.comfort_brake)
}

/// The lane beside that a signalling change of my own onto `to` (side `dir`) is checked
/// against (`gap_accepted`, the follower looked up back over the lanes leading in,
/// `perception::neighbour`): the one the change goes to, or where the move has carried on past a
/// joint (`AiState::change_plan`) the one on its side; None where no lane beside is seen there
/// (nothing to check against: the change stands).
fn change_target_view(p: &Perception, to: usize, dir: i32) -> Option<NeighbourView> {
    let views = [p.left, p.right];
    views.iter().flatten().find(|n| n.lane == to).or_else(|| views.iter().flatten().find(|n| n.dir == dir)).copied()
}

/// [`gap_accepted`]'s two halves: (the follower there need not brake hard, there is room ahead).
fn gap_parts(me: &Vehicle, nb: &NeighbourView) -> (bool, bool) {
    let v = me.state.speed;
    let ok_behind = nb.behind.map(|(g, vb)| g >= gap_needed_behind(me, nb, vb, v)).unwrap_or(true);
    let ok_ahead = nb.ahead.map(|(g, va)| g > 2.0 + (v - va).max(0.0) * 1.5).unwrap_or(true);
    (ok_behind, ok_ahead)
}

/// A vehicle behind a car about to pull out of a parking space,
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

/// The lane's limit (m/s).
fn lane_limit(net: &Network, lane: usize) -> f32 {
    net.lanes.get(lane).map_or(0.0, |l| l.speed_limit_kmh.max(0.0) / 3.6)
}

/// SUMO's `MSLane::getMaximumBrakeDist` on lane `lane` (m) over the drivers on the road
/// (`FleetBounds`: the largest chosen speed factor, the least deceleration, the largest minGap).
pub fn max_brake_dist(net: &Network, sim: &Simulation, lane: usize) -> f32 {
    sim.fleet.max_brake_dist(lane_limit(net, lane))
}

/// How far back (m) a follower can matter behind a place on lane `lane` for a pull-out: SUMO's
/// bound (`max_brake_dist`, which `getFollowersOnConsecutive` searches to), and no less than the
/// secure gap a pull-out asks of a follower (`secure_gap_behind`) bounded over the drivers on the
/// road (`FleetBounds`: the largest minGap and time gap, the least deceleration, coming at the
/// lane's limit times the largest speed factor).
pub fn follower_reach(net: &Network, sim: &Simulation, lane: usize) -> f32 {
    let f = &sim.fleet;
    let v = lane_limit(net, lane) * f.speed_factor;
    // (the bounds as one driver: `secure_gap_behind` reads only these three)
    let bound = DriverParams { tau: f.tau, min_gap: f.min_gap, comfort_brake: f.decel, ..DriverParams::preset(VehicleClass::Car) };
    secure_gap_behind(&PullOutFollower { gap: 0.0, speed: v, params: bound }).max(max_brake_dist(net, sim, lane))
}

/// How far back (m) a follower can matter behind the place beside me on lane `lane` for a lane
/// change of `me`'s: `follower_reach`, and no less than the gap my change accepts from rest
/// (`accepted_gap`) from a follower with the largest minGap on the road at the lane's limit times
/// the largest speed factor (`FleetBounds`).
pub fn change_follower_reach(net: &Network, sim: &Simulation, lane: usize, me: &Vehicle) -> f32 {
    let f = &sim.fleet;
    follower_reach(net, sim, lane).max(accepted_gap(me, f.min_gap, lane_limit(net, lane) * f.speed_factor, 0.0))
}

/// Walks back from lane `lane` at `s` over the lanes leading into it, as SUMO's
/// `MSLane::getFollowersOnConsecutive` does, while a lane's start lies within `reach` m behind
/// `s`: `visit(l, start)` once per lane, `start` the lane's start relative to `s` (negative;
/// `-s` for `lane` itself), and the walk goes on behind that lane where it returns true.
pub fn walk_back(net: &Network, lane: usize, s: f32, reach: f32, mut visit: impl FnMut(usize, f32) -> bool) {
    if lane >= net.lanes.len() {
        return;
    }
    let mut frontier: Vec<(usize, f32)> = vec![(lane, -s)];
    let mut seen: Vec<usize> = vec![lane];
    while let Some((cur, start)) = frontier.pop() {
        if !visit(cur, start) || -start > reach {
            continue;
        }
        for &pl in net.prev.get(cur).map(|v| v.as_slice()).unwrap_or(&[]) {
            if !seen.contains(&pl) {
                seen.push(pl);
                frontier.push((pl, start - net.lanes[pl].length()));
            }
        }
    }
}

/// The vehicles behind lane `lane` at `s` (the pull-out's origin, `rear` behind it to its
/// rear): on that lane and back over the lanes leading into it as far as a follower can matter
/// (`walk_back` to `follower_reach`), the players too (their preset's driver). `on_lane(l)` gives
/// the AI vehicles on lane `l`: (s, front, speed, driver).
pub fn pull_out_followers(net: &Network, sim: &Simulation, lane: usize, s: f32, rear: f32, on_lane: &dyn Fn(usize) -> Vec<(f32, f32, f32, DriverParams)>) -> Vec<PullOutFollower> {
    let player = DriverParams::preset(crate::sim::PLAYER_PRESET);
    let mut out = Vec::new();
    walk_back(net, lane, s, follower_reach(net, sim, lane), |cur, start| {
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
        true
    });
    out
}

/// Nobody is on the strip a pull-out's body sweeps from its parking
/// offset onto its lane: the body at lane `lane`, `s` + d, `lateral` eased to 0 over `ramp_len`
/// m as `AiState::lateral_ramp` does, for d from where it stands to the ramp's end (a walker
/// passing through the parked body holds it until through; the walkers on the pavement beside
/// and behind it are not on the strip); each person
/// where they are and where they will be
/// when the body is at that metre, driving from rest at `PULL_OUT_ACCEL` (SUMO's insertion fails
/// on a person on the footprint, `MSLane::checkForPedestrians`; on its way it brakes for persons
/// on its lane).
pub fn pull_out_strip_clear(net: &Network, sim: &Simulation, lane: usize, s: f32, lateral: f32, front: f32, rear: f32, half_width: f32, ramp_len: f32, v: f32) -> bool {
    let Some(l) = net.lanes.get(lane) else { return true };
    let (q, _) = l.at(s.min(l.length()));
    !sim.people_near(q, (ramp_len + front + rear) as f64 + 10.0).into_iter().any(|k| on_pull_out_strip(net, lane, s, lateral, front, rear, half_width, ramp_len, v, &sim.people[k]))
}

/// Whether person `p` is on the strip (`pull_out_strip_clear`): where
/// they are, or where they will be when the body is at that metre - the car driving on from `v`
/// m/s at `PULL_OUT_ACCEL` up to the lane's limit (`arrival_time`; timed from `v`, not from rest).
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

/// The rest of a pull-out's way out (m): the steerable move from rest
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

/// The one exemption the people test makes for a car pulling out of a
/// parking space (`people_ahead` asks it for every person): somebody under or beside its parked
/// body - level with it, from its rear to its front - who is not on the strip it sweeps onto its
/// lane (`on_pull_out_strip`) is not in its way (the walkers passing over the pavement at its
/// spot). Everybody else is tested as by any car (people just past the strip are met at 3-4 m/s).
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

/// The one pull-out rule, for the parked object the app takes
/// off the kerb, for a recorded pull-out standing at its offset and for any
/// vehicle standing out of its lane on its way back in, a bus leaving its bay: every moving
/// follower has its secure gap (`secure_gap_behind`), a standing one only room for its body, and nobody is on the strip its body sweeps
/// (`pull_out_strip_clear`). SUMO re-inserts a parked vehicle only when `isInsertionSuccess`
/// passes (`MSVehicleTransfer`); OMSI draws the body, so the car then drives from its offset.
pub fn pull_out_clear(net: &Network, sim: &Simulation, followers: &[PullOutFollower], lane: usize, s: f32, lateral: f32, front: f32, rear: f32, half_width: f32, ramp_len: f32) -> bool {
    // (a standing follower reaches only its own body, SUMO's secure gap at v = 0
    // being 0: one queued right behind a car out of its lane would hold it for good)
    followers.iter().all(|f| if f.speed < STANDING_SPEED { f.gap >= 0.0 } else { f.gap >= secure_gap_behind(f) }) && pull_out_strip_clear(net, sim, lane, s, lateral, front, rear, half_width, ramp_len, 0.0)
}

/// SUMO LC2013's strategic look-ahead (`MSLCM_LC2013::LOOK_FORWARD`, s): a lane that
/// ends for the driver raises its wish to leave it from this many seconds of its speed out.
pub const LC_LOOK_FORWARD: f32 = 10.0;
/// SUMO LC2013's `LOOK_AHEAD_SPEED_MEMORY`: the look-ahead speed falls by this
/// factor per second (`Vehicle::look_ahead_speed`).
pub const LOOK_AHEAD_SPEED_MEMORY: f32 = 0.9;
/// ... times this towards the overtaking side (SUMO's `lcLookaheadLeft` default; the
/// side mirrors on a left-hand-traffic map).
pub const LC_LOOKAHEAD_LEFT: f32 = 2.0;
/// SUMO LC2013's `HELP_DECEL_FACTOR`: a follower asked to make room for a lane change
/// brakes no harder than its comfortable brake times this.
pub const HELP_DECEL_FACTOR: f32 = 1.0;

/// The road a lane change started at `v` m/s takes (m): the signalling before
/// it moves (`SIGNAL_BEFORE_CHANGE`, speeding up at `a` meanwhile) and the move
/// (`change_length`).
pub fn change_space(v: f32, a: f32) -> f32 {
    let w = omsi_sim::traffic::SIGNAL_BEFORE_CHANGE;
    v * w + 0.5 * a.max(0.0) * w * w + omsi_sim::traffic::change_length(v)
}

/// SUMO LC2013's strategic distance (`laDist`): the look-ahead speed times
/// `LC_LOOK_FORWARD` (times `LC_LOOKAHEAD_LEFT` towards the overtaking side) plus two of the
/// vehicle's lengths with its standing gap.
pub fn lane_end_look(v: f32, overtaking_side: bool, me: &Vehicle) -> f32 {
    v * LC_LOOK_FORWARD * if overtaking_side { LC_LOOKAHEAD_LEFT } else { 1.0 } + 2.0 * (me.length() + me.state.min_gap)
}

/// May the driver change onto `nb` because its own lane ends for it (`Perception::
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

/// Somebody on the lane `nb` over the move a change would take (from beside my rear to
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

/// The lane `lane` beside me runs on `need` m past my front (along its first ways on;
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

/// SUMO LC2013's strategic wish to leave a lane that ends for the driver (sumo.md 3.2),
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
/// accepted: SUMO LC2013's motivations (the lane's end and the turn - strategic -, speed gain,
/// keeping to the side) and an obstacle motive of ours (a standing obstacle).
/// `near_junction`: within the last stretch before a junction, where the lines are solid
/// and only the turn lane and a standing obstacle are reasons to move over. `ant`: (mine, beside)
/// from `anticipated_speeds`.
fn change_desire(net: &Network, sim: &Simulation, me: &Vehicle, p: &Perception, nb: &NeighbourView, near_junction: bool, ant: (f32, f32)) -> (f32, bool, ChangeWhy) {
    let st = &me.state;
    let dp = &me.params;
    let lane = &net.lanes[st.lane];
    // the lane ends for me: the strategic wish to leave it, whatever else holds
    let end = lane_end_desire(net, sim, me, p, nb);
    // (nor a move that does not fit before the end: SUMO's `LCA_INSUFFICIENT_SPACE`; the car
    // slows to its last-resort stop, which leaves the room to move over from standing)
    // (the move's whole road, `change_space`, within the end, less what the car's stop control
    // lands either side of a stop, `REST_SHORT`: the last-resort stop leaves the space from rest)
    let fits = p.lane_end.is_none_or(|e| e.kind == LaneEndKind::Merge || change_space(st.speed, st.accel) <= e.dist + REST_SHORT);
    let end_ok = |ok: bool| ok && !(end > 0.0 && (people_on_move(net, sim, me, nb) || !fits));
    if !may_change_to(net, sim, me, nb) {
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
    // behind something standing: round it (ours: SUMO LC2013's nearest rule, the strategic
    // change round a leader stopped at a stop within `REACT_TO_STOPPED_DISTANCE`, MSLCM_LC2013.cpp
    // l.1400, takes only vehicles at a stop; here any standing obstacle near, stood behind a while)
    if let Some(l) = p.leader {
        let standing = l.standing_obstacle() && l.gap < 25.0;
        if standing && me.stopped >= 4.0 && nb.ahead.map(|a| a.0 > l.gap + l.length + 6.0).unwrap_or(true) {
            wish(0.8, ChangeWhy::Obstacle);
        }
    }
    // a faster lane: SUMO LC2013's speed gain - the accumulated wish (`Vehicle::speed_gain`,
    // `speed_gain_step`: the relative gain of the speeds the driver can anticipate on each lane,
    // `anticipated_speeds`) past its threshold (`SPEED_GAIN_THRESHOLD`, towards the keep side
    // over `SPEED_GAIN_RIGHT`), away from the keep side only while the lane beside is still the
    // faster, onto a lane the car can stay on `SPEED_GAIN_REMAIN_TIME` at its speed - whatever
    // its own speed: a car standing in a queue beside an open lane has the gain too; a queue
    // standing ahead on the lane beside counts only through the safe speed behind it
    // (`anticipateFollowSpeed`), as in SUMO, no bar
    // (and not into a gap behind its leader there shorter than my secure gap, SUMO's
    // `LCA_BLOCKED_BY_LEFT/RIGHT_LEADER`: standing gap, headway, the braking to its speed)
    let (this_safe, nb_safe) = ant;
    let secure = |vl: f32| st.min_gap + st.speed.max(0.0) * dp.tau + (st.speed.max(0.0).powi(2) - vl.max(0.0).powi(2)).max(0.0) / (2.0 * dp.comfort_brake.max(0.1));
    let blocked = nb.ahead.is_some_and(|(g, vl)| g < secure(vl));
    let towards_keep = nb.dir == keep_side(sim);
    let threshold = if towards_keep { SPEED_GAIN_THRESHOLD / SPEED_GAIN_RIGHT } else { SPEED_GAIN_THRESHOLD };
    let gained = me.speed_gain[(nb.dir as usize).clamp(1, 2) - 1] > threshold && (towards_keep || nb_safe > this_safe);
    let remains = nb.runs_on / st.speed.max(0.1) > SPEED_GAIN_REMAIN_TIME;
    if gained && remains && !blocked && !near_junction && open_on(net, me, nb.lane, st.s) {
        wish(dp.lc_speed_gain, ChangeWhy::SpeedGain);
    }
    // keep to the side: SUMO LC2013's keepRight layer - the accumulated wish
    // (`Vehicle::keep_right`, `keep_right_rate`) past its threshold, unless SUMO's strategic stays
    // hold (`stays_on_best`, over the lanes open to me: no `open_on` window); a speed-
    // gain, obstacle or turn wish, or a lane-end wish over `lc_keep_right`, outweighs it (how long the driver could keep
    // its speed there is the accumulator's rate)
    if nb.dir == keep_side(sim) && me.keep_right >= KEEP_RIGHT_THRESHOLD && !stays_on_best(net, sim, me, nb) {
        wish(dp.lc_keep_right, ChangeWhy::KeepSide);
    }
    (desire, end_ok(gap_accepted(me, nb)), why)
}

/// The motivation that won a lane change of the driver's own (`change_desire`; counted per
/// start, `sim::CHANGES_BY_WHY`, `traffic_cases ab`).
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

/// May the driver change onto `nb` of its own accord (a change for a reason of
/// its own, a bypass; `change_desire` and `could_bypass` ask the same): the path's rules
/// allow it there; no solid line (`[rule] overtaking_prohib`) on either lane; on a junction
/// object's piece only where the move is clear of the junction (`piece_change_clear`); a turn
/// lane is never left within `TURN_LANE_KEEP` of its junction, a turn ahead is taken from the
/// lane that leads into it, and a turn lane is not taken for a turn the driver does not take;
/// and `nb` runs on `CHANGE_RUNS_ON` at least.
fn may_change_to(net: &Network, sim: &Simulation, me: &Vehicle, nb: &NeighbourView) -> bool {
    let lane = &net.lanes[me.state.lane];
    if !nb.allowed || lane.no_overtaking || net.lanes[nb.lane].no_overtaking || nb.runs_on < CHANGE_RUNS_ON {
        return false;
    }
    if lane.source == 2 && !piece_change_clear(net, sim, me, nb) {
        return false;
    }
    let my_turn = me.state.turn_ahead(net, TURN_LANE_KEEP).map(|t| t.0).unwrap_or(0);
    !((my_turn != 0 && nb.turn != my_turn) || (nb.turn != 0 && my_turn == 0 && nb.runs_on < TURN_LANE_KEEP))
}

/// A change of the driver's own from a junction object's piece onto the piece beside it
/// (`Network::piece_beside`) is a change on a road: SUMO changes lanes on an edge and never on
/// an internal lane, whose stretch is where the ways of the junction meet (`mayChange`, sumo.md
/// 3.1). An OMSI object also lays plain road as paths, so the move is allowed where it touches no
/// meeting place: the stretch the body covers - from beside my rear to my front, plus what the car
/// drives while it signals (`SIGNAL_BEFORE_CHANGE` at its speed, as `change_clear` lays it) and the
/// move's length (`change_length`) - must stay on both pieces and touch, on either piece, no foe's
/// stretch (`Foe::mine`), no stretch converging with a third way (`Movement::converge`) and no
/// people's crossing (`Network::walks`, the cuts the perception's zebra rule reads; SUMO's
/// crossings are internal lanes too), and `nb` must still lie beside where the move ends
/// (parallel for the move's length).
fn piece_change_clear(net: &Network, sim: &Simulation, me: &Vehicle, nb: &NeighbourView) -> bool {
    let st = &me.state;
    let (Some(&(j, m)), Some(&(j2, m2))) = (sim.junctions.of_lane.get(&st.lane), sim.junctions.of_lane.get(&nb.lane)) else { return false };
    if j != j2 {
        return false;
    }
    let pre = st.speed.max(0.0) * omsi_sim::traffic::SIGNAL_BEFORE_CHANGE;
    let mv = omsi_sim::traffic::change_length(st.speed);
    let s_b = net.beside_s(st.lane, nb.lane, st.s);
    let mine = (st.s - st.rear, st.s + st.front + pre + mv);
    let theirs = (s_b - st.rear, s_b + st.front + pre + mv);
    if mine.0 < 0.0 || mine.1 > net.lanes[st.lane].length() || theirs.0 < 0.0 || theirs.1 > net.lanes[nb.lane].length() {
        return false;
    }
    // (the two pieces' own nearness is the lane beside, not a meeting: narrow parallel pieces
    // come within `MEET_DIST` of each other)
    let meets = |mi: usize, other: usize, lane: usize, (a, b): (f32, f32)| {
        let mm = &sim.junctions.nodes[j].movements[mi];
        mm.foes.iter().any(|f| f.mine.0 < b && f.mine.1 > a)
            || mm.converge.iter().zip(&mm.converge_with).any(|(c, &w)| w != other && c.0 < b && c.1 > a)
            || net.walks.get(lane).is_some_and(|w| w.iter().any(|&(fp, sa, _)| {
                // (the walk's band: its half width either side of where its axis cuts my lane)
                let half = net.lanes.get(fp).map_or(0.0, |l| l.width * 0.5);
                sa + half > a && sa - half < b
            }))
    };
    if meets(m, m2, st.lane, mine) || meets(m2, m, nb.lane, theirs) {
        return false;
    }
    net.piece_beside(st.lane, mine.1, nb.dir) == Some(nb.lane)
}

/// A keep-to-the-side change needs the lanes the target runs into (one after the other, while
/// each has a single next) open to the car for the move's length and this many metres more (m:
/// what a driver sees of the lane markings ahead: not onto a bus lane that runs into a lane
/// closed to the car).
pub const KEEP_SIDE_OPEN_AHEAD: f32 = 60.0;

/// SUMO LC2013's `KEEP_RIGHT_TIME` (s): "the number of seconds after which a vehicle
/// should move to the right lane" - with the lane beside free for its whole acceptance time
/// the wish reaches its threshold in this long.
pub const KEEP_RIGHT_TIME: f32 = 5.0;
/// The accumulated keep-right wish (`Vehicle::keep_right`) asks for the change once it
/// reaches this: SUMO's `myKeepRightProbability * lcKeepRight < -myChangeProbThresholdRight`
/// with the default `lcKeepRight` 1, the accumulator counted in units of the threshold.
pub const KEEP_RIGHT_THRESHOLD: f32 = 1.0;
/// SUMO LC2013's legacy acceptance time (`lcKeepRightAcceptanceTime` -1, its default):
/// `7 * roadSpeedFactor * max(1, v)` seconds.
pub const KEEP_RIGHT_ACCEPTANCE: f32 = 7.0;
/// SUMO LC2013's `RELGAIN_NORMALIZATION_MIN_SPEED` (m/s): the speed gain is relative to the
/// faster of the anticipated speed beside and this.
pub const RELGAIN_NORMALIZATION_MIN_SPEED: f32 = 10.0;
/// SUMO LC2013's speed-gain threshold away from the keep side (`myChangeProbThresholdLeft`,
/// `0.2 / lcSpeedGain` with the default `lcSpeedGain` 1), in seconds of full relative gain.
pub const SPEED_GAIN_THRESHOLD: f32 = 0.2;
/// SUMO's `lcSpeedGainRight` (default 0.1): towards the keep side the threshold is
/// `SPEED_GAIN_THRESHOLD / SPEED_GAIN_RIGHT` (2.0), so a car does not swing back and forth.
pub const SPEED_GAIN_RIGHT: f32 = 0.1;
/// SUMO's `lcSpeedGainRemainTime` (s): a change for speed only onto a lane the car can stay on
/// this long at its speed (`neighDist / max(0.1, v)`).
pub const SPEED_GAIN_REMAIN_TIME: f32 = 20.0;

/// How SUMO LC2013 moves a speed-gain wish (`Vehicle::speed_gain`) per second.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpeedGainStep {
    /// Grows by this relative gain per second (negative: shrinks).
    Grow(f32),
    /// Fades by this factor per second (0.5 where the own lane is better, 0.8 where they are
    /// equal).
    Fade(f32),
}

/// SUMO LC2013's speed-gain step towards the lane beside, `nb` (`_wantsChange`): the relative
/// gain of the speeds the driver can anticipate (`anticipated_speeds`), `(beside - mine) /
/// max(beside, RELGAIN_NORMALIZATION_MIN_SPEED)`. Towards the keep side (SUMO's right): faded by
/// half a second while the own lane is faster by `KEEP_RIGHT_FASTER`, else grown by the gain;
/// away from it (SUMO's left): faded by half while the own lane is faster, by 0.8 while equal,
/// else grown. `ant`: (mine, beside) from `anticipated_speeds`.
pub fn speed_gain_step(sim: &Simulation, nb: &NeighbourView, ant: (f32, f32)) -> SpeedGainStep {
    let (this_safe, nb_safe) = ant;
    let rel = (nb_safe - this_safe) / nb_safe.max(RELGAIN_NORMALIZATION_MIN_SPEED);
    if nb.dir == keep_side(sim) {
        if this_safe - KEEP_RIGHT_FASTER > nb_safe {
            SpeedGainStep::Fade(0.5)
        } else {
            SpeedGainStep::Grow(rel)
        }
    } else if this_safe > nb_safe {
        SpeedGainStep::Fade(0.5)
    } else if this_safe == nb_safe {
        SpeedGainStep::Fade(0.8)
    } else {
        SpeedGainStep::Grow(rel)
    }
}

/// SUMO LC2013 accumulates the keep-right wish only while the current lane is not faster
/// than the lane beside by more than this (m/s: `thisLaneVSafe - 5 / 3.6 > neighLaneVSafe`).
pub const KEEP_RIGHT_FASTER: f32 = 5.0 / 3.6;
/// SUMO LC2013's `TURN_LANE_DIST` (m): "the distance at which a lane leading elsewhere is
/// considered to be a turn-lane that must be avoided" - a driver on the lane that runs on does not
/// leave it for a keep-right onto one that runs on less than this (`stays_on_best`).
pub const TURN_LANE_DIST: f32 = 200.0;
/// SUMO's `lcStrategicLookahead` (m, default 3000): how far LC2013's best lanes, and so
/// the distance a lane beside runs on (`neighDist`), are looked along.
pub const LC_STRATEGIC_LOOKAHEAD: f32 = 3000.0;
/// SUMO's `MSVehicle::congested` (m/s, 60 km/h): below it overtaking on the keep side is
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

/// SUMO LC2013's strategic stays that come before its keep-right layer, for the lane beside
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

/// How far (m, at most `cap`) a car at `s` on `lane` can drive on along it and the lanes it
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
        if l.source != 2 && !crate::access::is_open_at(net, lane, me) {
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

/// `open_run` looks at no more than this many lanes (the network's own reach is counted
/// over the same chains, `Network::compute_reach`).
const OPEN_RUN_LANES: usize = 64;

/// SUMO's Krauss `maximumSafeFollowSpeed` (on insertion: no reaction step): the highest
/// speed from which the driver, braking at `b` after `tau`, stays behind a lead `gap` m ahead at
/// `vl` m/s braking as hard; a stop `gap` m ahead with `vl` 0 (`maximumSafeStopSpeed`).
fn safe_speed(gap: f32, vl: f32, b: f32, tau: f32) -> f32 {
    let b = b.max(0.1);
    let room = gap.max(0.0) + vl.max(0.0).powi(2) / (2.0 * b);
    -b * tau + ((b * tau).powi(2) + 2.0 * b * room).sqrt()
}

/// The speeds (m/s) the driver can anticipate on its own lane and on the lane beside, `nb`
/// (SUMO LC2013's `thisLaneVSafe` / `neighLaneVSafe`, `anticipateFollowSpeed`): the safe speed
/// behind each lane's leader (none: before the end of the lanes open to it there, `open_run`,
/// up to `LC_STRATEGIC_LOOKAHEAD`), capped by its desired speed and on the lane beside by that
/// lane's limit at its desire; and on each lane by the stop before the nearest person ahead in
/// the car's width there (`person_ahead`; LC2013's `adaptSpeedToPedestrians`).
///
/// `own` keeps the speed on my own lane between the calls for the two lanes beside (it does not
/// depend on the lane beside).
fn anticipated_speeds(net: &Network, sim: &Simulation, me: &Vehicle, p: &Perception, nb: &NeighbourView, own: &mut Option<f32>) -> (f32, f32) {
    let st = &me.state;
    let (b, tau) = (me.params.comfort_brake, me.params.tau);
    let v_max = p.v_want.max(0.1);
    let s_b = net.beside_s(st.lane, nb.lane, st.s);
    let runs = open_run(net, me, nb.lane, s_b, LC_STRATEGIC_LOOKAHEAD);
    // (SUMO's `stopSpeed` before the person less the driver's standing gap)
    let person_cap = |lane: usize, s: f32| person_ahead(net, sim, me, lane, s).map_or(f32::INFINITY, |d| safe_speed(d - st.min_gap, 0.0, b, tau));
    let this_safe = *own.get_or_insert_with(|| {
        v_max.min(person_cap(st.lane, st.s)).min(match p.leader {
            Some(l) => safe_speed(l.gap, l.speed, b, tau),
            None => safe_speed(open_run(net, me, st.lane, st.s, LC_STRATEGIC_LOOKAHEAD), 0.0, b, tau),
        })
    });
    let nb_max = net.lanes.get(nb.lane).map_or(v_max, |l| ((l.speed_limit_kmh * st.desire).min(st.max_speed_kmh).max(3.0) / 3.6).max(0.1));
    let nb_safe = nb_max.min(person_cap(nb.lane, s_b)).min(match nb.ahead {
        Some((g, vl)) => safe_speed(g, vl, b, tau),
        None => safe_speed(runs, 0.0, b, tau),
    });
    (this_safe, nb_safe)
}

/// The distance (m) from my front to the nearest person on `lane` from beside my rear past `s`
/// to `REACT_TO_STOPPED_DISTANCE` ahead (its own stretch) whose body reaches into the width I
/// would have there (my half width and `PERSON_RADIUS` round the lane's axis; one beside my body:
/// 0): SUMO's `MSLane::nextBlocking` over the vehicle's width, which LC2013's
/// `adaptSpeedToPedestrians` reads for the speed it anticipates on each lane. It looks only at
/// the people in a box round that stretch of the lane, as `nextBlocking` loops only over the
/// pedestrians of its lane.
fn person_ahead(net: &Network, sim: &Simulation, me: &Vehicle, lane: usize, s: f32) -> Option<f32> {
    let l = net.lanes.get(lane)?;
    let reach = (l.length() - s).min(crate::perception::REACT_TO_STOPPED_DISTANCE);
    if reach <= 0.0 {
        return None;
    }
    let band = me.half_width + PERSON_RADIUS;
    // (only somebody within the band of the lane's stretch from my rear to the reach can count:
    // the box round that stretch's points, widened by the band and `BOX_ROUNDING`, holds them all)
    let (from, to) = (s - me.state.rear - BOX_ROUNDING, s + reach + BOX_ROUNDING);
    let (mut lo, mut hi) = (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY));
    for k in 0..l.points.len().saturating_sub(1) {
        if l.dist[k + 1] < from || l.dist[k] > to {
            continue;
        }
        for q in [l.points[k], l.points[k + 1]] {
            lo = lo.min(q.truncate());
            hi = hi.max(q.truncate());
        }
    }
    let widen = DVec2::splat((band + BOX_ROUNDING) as f64);
    let (lo, hi) = (lo - widen, hi + widen);
    let mut best: Option<f32> = None;
    for k in sim.people_near_in_box(me.pos, (reach + me.state.front) as f64, lo, hi) {
        let q = &sim.people[k];
        if (q.pos.z - me.pos.z).abs() > 2.5 || q.pos.x < lo.x || q.pos.x > hi.x || q.pos.y < lo.y || q.pos.y > hi.y {
            continue;
        }
        let Some((qs, d)) = l.nearest_point(q.pos) else { continue };
        // (one beside my body counts as at my front: a change sweeps the body over it)
        if qs - s < -me.state.rear || qs - s > reach || d as f32 > band {
            continue;
        }
        let ahead = (qs - s - me.state.front).max(0.0);
        if best.is_none_or(|x| ahead < x) {
            best = Some(ahead);
        }
    }
    best
}

/// `person_ahead`'s box round a stretch of lane is this much larger on every side (m) than the
/// stretch and the band need, so that the f32 rounding of distances along the lane and across it
/// never leaves out somebody the exact test keeps (the box only spares the exact test).
const BOX_ROUNDING: f32 = 0.01;

/// SUMO LC2013's keep-right accumulation per second towards the lane beside on the keep
/// side, `nb` (`MSLCM_LC2013::_wantsChange`, "honor the obligation to keep right"): 0 while my
/// lane is faster than that one by `KEEP_RIGHT_FASTER` (the anticipated safe speeds behind each
/// lane's leader, `anticipateFollowSpeed`); else the share of the acceptance time
/// (`KEEP_RIGHT_ACCEPTANCE`) the driver could drive at its desired speed there - bounded by how far
/// the lane runs on less its braking distance, by closing on a slower leader there to its secure
/// gap, and above `CONGESTED_SPEED` by catching up with my own slower leader (no overtaking on the
/// keep side) - over `KEEP_RIGHT_TIME`. `ant`: (mine, beside) from `anticipated_speeds`.
pub fn keep_right_rate(net: &Network, vehicles: &[Vehicle], me: &Vehicle, p: &Perception, nb: &NeighbourView, ant: (f32, f32)) -> f32 {
    let st = &me.state;
    let dp = &me.params;
    let (b, tau) = (dp.comfort_brake, dp.tau);
    let v_max = p.v_want.max(0.1);
    // (SUMO's `currentDist` and `neighDist`: the best lanes' length, the lanes open to
    // the car, up to `lcStrategicLookahead` - `open_run`, not the network's capped reach)
    let runs = open_run(net, me, nb.lane, net.beside_s(st.lane, nb.lane, st.s), LC_STRATEGIC_LOOKAHEAD);
    let (this_safe, nb_safe) = ant;
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
    // (SUMO compares my desired speed with the leader's maximum on its lane,
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
        if l.source != 2 && !crate::access::is_open_at(net, cur, me) {
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
    obstacle_run_in(&p.ahead, p.road_left, lead, need)
}

/// `obstacle_run` over the bodies `ahead` in a corridor (nearest first) with `road_left` metres
/// of way.
fn obstacle_run_in(ahead: &[(f32, f32, f32)], road_left: f32, lead: &crate::perception::Headway, need: f32) -> (f32, bool) {
    let mut end = lead.gap + lead.length;
    for &(near, far, speed) in ahead {
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
    (end - lead.gap, road_left > end + need + 2.0)
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
/// Seconds of indicator before an in-lane squeeze pulls out (a real pass signals
/// `SIGNAL_BEFORE_CHANGE`).
pub const SQUEEZE_SIGNAL: f32 = 0.5;
/// Metres the driver-side wheels may go over the line into a free neighbour lane when the
/// drawn road beside the obstacle is not quite enough for the squeeze ("the driver side
/// wheels step into the left neighbouring lane while most of the car remains in the
/// current lane").
pub const WHEELS_OVER_LINE: f32 = 0.6;

/// The clearance (m) a squeeze's bent corridor keeps from every body beside the lane - a car
/// standing on the strip, a bus in its bay, a parked car, the thing passed itself where it
/// stands askew: a squeeze does not start into one, and the thing passed holds the car until
/// the bent way really clears it (squeezing along a strip with bodies on it runs cars into each other).
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

/// SUMO's sublane `minGapLat` default (m, sumo.md 3.4; SL2015 `updateGaps`): the lateral gap an
/// in-lane squeeze keeps from the body it passes, measured from that body's corners
/// (`body_across`). SUMO keeps it to a body beside; where the drawn road leaves less room the car
/// does not squeeze and passes on the oncoming lane or waits.
pub const MIN_GAP_LAT: f32 = 0.6;

/// Where the body of the thing ahead, `l`, lies across my way (m from my lane's axis, + right):
/// the least and greatest lateral of its box's corners in the way's frame where it is met
/// (a parked car: its middle and half width).
fn body_across(net: &Network, sim: &Simulation, vehicles: &[Vehicle], me: &Vehicle, l: &crate::perception::Headway) -> (f32, f32) {
    let st = &me.state;
    let obb = match l.what {
        Seen::Vehicle(j) | Seen::Body(j) => vehicles.get(j).map(|o| o.obb()),
        Seen::Player(id) => sim.players.iter().find(|p| p.id == id).map(|p| p.obb(0.0, 0.0)),
        Seen::Parked => None,
    };
    let Some(b) = obb else { return (l.lateral - l.half_width, l.lateral + l.half_width) };
    let d = l.gap + st.front;
    let q = st.way_point(net, d).truncate();
    let f = (st.way_point(net, d + 1.0).truncate() - q).normalize_or_zero();
    let hr = (me.heading as f64).to_radians();
    let f = if f.length_squared() < 0.5 { DVec2::new(hr.sin(), hr.cos()) } else { f };
    let right = DVec2::new(f.y, -f.x);
    // (the way's point carries my own planned offset: back to the lane's axis)
    let base = st.lateral_ahead(d);
    b.corners().iter().map(|&c| (c - q).dot(right) as f32 + base).fold((f32::INFINITY, f32::NEG_INFINITY), |(a, z), x| (a.min(x), z.max(x)))
}

/// The out length of an in-lane squeeze `side` metres over past `l`: the usual move
/// (`Passing::out_len_for`), or from nearer than that the shortest move the body can steer
/// (`bypass_min_len`) within the gap less `BYPASS_GAP_MARGIN`.
fn squeeze_out(me: &Vehicle, l: &crate::perception::Headway, side: f32) -> f32 {
    let usual = Passing::out_len_for(side);
    let lateral = side - me.state.lateral;
    (l.gap - BYPASS_GAP_MARGIN).clamp(bypass_min_len(me.length(), lateral).min(usual), usual)
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
/// over `back` metres more? (Sampled from the start of the way, so a car standing level or just
/// ahead on the squeeze side is met.)
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
        // out, its body in the car it passes)
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
/// no solid line, on a junction's piece a move clear of the junction (`may_change_to`), a turn
/// ahead that stays its own, room there past
/// the thing standing and nobody passing on it the wrong way. (What the keep-back asks.)
fn could_bypass(net: &Network, sim: &Simulation, me: &Vehicle, nb: &NeighbourView, lead: &crate::perception::Headway) -> bool {
    if !may_change_to(net, sim, me, nb) {
        return false;
    }
    // (the gap there is the start's to accept, `change_desire`, and a car driving on ahead
    // there leaves the room: one going by in that lane now does not make the car roll up to 2 m behind the obstacle, from where it can never
    // steer out - the queue behind a bus at its stop)
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

/// The turning radius (m) of a body `length` long at `MAX_STEER_DEG`: its wheelbase
/// (`WHEELBASE_SHARE`) over the tangent of the lock.
pub fn turn_radius(length: f32) -> f32 {
    WHEELBASE_SHARE * length / MAX_STEER_DEG.to_radians().tan()
}

/// The shortest smoothstep move over `side` metres the body can drive (m, a ramp back into the
/// lane, `AiState::lateral_ramp`): its tightest bends, at its ends, have a radius of length² /
/// (6 |side|), no tighter than `turn_radius`. (For a 4.6 m car over 3.5 m this is 9.1 m, its
/// steepest yaw 30 degrees.)
pub fn bypass_min_len(length: f32, side: f32) -> f32 {
    (6.0 * side.abs() * turn_radius(length)).sqrt()
}

/// The shortest move over `side` metres a body `length` long can steer from rest (m): two arcs
/// of its `turn_radius`, out at full lock and back at full lock the other way
/// (`omsi_sim::traffic::full_lock01`, the bypass's move). For a 4.5 m car over 3.05 m: 6.1 m,
/// against the smoothstep's 8.4 m (`bypass_min_len`), whose bends ease off from full lock.
pub fn full_lock_len(length: f32, side: f32) -> f32 {
    let r = turn_radius(length);
    let h = (side.abs() * 0.5).min(r);
    2.0 * (r * r - (r - h) * (r - h)).max(0.0).sqrt()
}

/// The bypass's move at `d` metres along it (`full_lock01` over `out_len`, `side` across):
/// (the lateral, its slope).
fn full_lock_at(d: f32, out_len: f32, side: f32) -> (f32, f32) {
    let f = |x: f32| side * omsi_sim::traffic::full_lock01(x / out_len.max(0.1), out_len, side);
    let e = 0.05;
    (f(d), (f(d + e) - f(d - e)) / (2.0 * e))
}

/// The bypass round the standing `lead` into `nb`, from rest: the move ends `BYPASS_GAP_MARGIN`
/// short of its rear, no longer than `BYPASS_LENGTH` and no shorter than the body can steer
/// (`bypass_min_len` on the smoothstep; where that move does not clear, `full_lock_len` on the
/// two arcs at full lock), and it starts only when that move - the body
/// turned along it - clears the thing passed and every other vehicle, player, parked car and
/// person by `SQUEEZE_CLEARANCE` (`body_sweep_clear`); where the gap is too short for it, the
/// car does not set out (a car does not reverse out). Not round a bus indicating out of its
/// stop or a vehicle moving off (it is let out). What stands beyond the lead is read from the
/// bodies `ahead` in my corridor (`Perception::ahead`, `obstacle_run`). With `kind` only that
/// move is tried (`Some(true)` the two arcs at full lock, `Some(false)` the smoothstep).
fn bypass_start(net: &Network, sim: &Simulation, vehicles: &[Vehicle], me: &Vehicle, ahead: &[(f32, f32, f32)], road_left: f32, nb: &NeighbourView, lead: &crate::perception::Headway, kind: Option<bool>) -> Option<Start> {
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
    // (the gentle smoothstep first; the two arcs at full lock only where it does not clear;
    // the move begins after the signalling, `BYPASS_SIGNAL`, the car stopping meanwhile where it
    // can, `pre` on; its way must take the lead out of my corridor before my front is there,
    // `bypass_need`, and the body turned along it must clear every body)
    let pre = st.speed.max(0.0).powi(2) / (2.0 * me.params.comfort_brake.max(0.1));
    let need = bypass_need(me, lead, side.signum());
    let moves = [(bypass_min_len(me.length(), side), false), (full_lock_len(me.length(), side), true)];
    for (shortest, full_lock) in moves.into_iter().filter(|m| kind.is_none_or(|k| k == m.1)) {
        let out_len = (lead.gap - pre - BYPASS_GAP_MARGIN).clamp(shortest.min(BYPASS_LENGTH), BYPASS_LENGTH).max(shortest);
        if bypass_lateral(lead.gap + st.front - pre, out_len, side.abs(), full_lock) < need {
            continue;
        }
        let profile = if full_lock { Profile::FullLock { pre } } else { Profile::Smooth { pre } };
        let (obstacle_len, _) = obstacle_run_in(ahead, road_left, lead, me.length() + out_len + PASS_ROOM);
        let level = lead.gap + obstacle_len + st.front + st.rear + 2.0;
        let parked = parked_boxes(net, sim, me, level + st.front);
        let people: Vec<Obb> = sim.people_near(me.pos, (level + st.front) as f64).into_iter().map(|k| &sim.people[k]).filter(|q| (q.pos.z - me.pos.z).abs() <= 2.5).map(|q| Obb::point(q.pos, PERSON_RADIUS as f64)).collect();
        if body_sweep_clear(net, sim, vehicles, me, &parked, &people, side, out_len, level, profile, Bodies::All) {
            return Some(Start::Bypass { to: nb.lane, dir: nb.dir, out_len, full_lock, round: obstacle });
        }
    }
    None
}

/// Does a lane change into `nb` started now, at speed, clear what is near: the body turned
/// along the change's smoothstep (`change_length`) after the signalling (`SIGNAL_BEFORE_CHANGE`
/// at this speed), swept by `SQUEEZE_CLEARANCE` (`body_sweep_clear`) past the parked cars, the
/// people and the standing vehicles and players near, and with a `lead` standing within the move
/// past every body (the moving ones are otherwise the gap rule's). OMSI geometry: the body turns
/// as it moves over, its rear swinging out the other way - into a car standing on the lane or
/// piece on that side - where SUMO's change is a jump.
fn change_clear(net: &Network, sim: &Simulation, vehicles: &[Vehicle], me: &Vehicle, p: &Perception, nb: &NeighbourView, lead: Option<&crate::perception::Headway>) -> bool {
    let st = &me.state;
    let v = st.speed.max(0.0);
    let Some((ts, _)) = net.lanes[nb.lane].nearest_point(me.pos) else { return false };
    let hr = (me.heading as f64).to_radians();
    let right = DVec2::new(hr.cos(), -hr.sin());
    let side = (net.lanes[nb.lane].at(ts).0 - me.pos).truncate().dot(right) as f32;
    let pre = v * omsi_sim::traffic::SIGNAL_BEFORE_CHANGE;
    let out_len = omsi_sim::traffic::change_length(v);
    let level = match lead {
        Some(lead) => lead.gap + obstacle_run(p, lead, me.length() + PASS_ROOM).0 + st.front + st.rear + 2.0,
        None => pre + out_len,
    };
    let parked = parked_boxes(net, sim, me, level + st.front);
    let people: Vec<Obb> = sim.people_near(me.pos, (level + st.front) as f64).into_iter().map(|k| &sim.people[k]).filter(|q| (q.pos.z - me.pos.z).abs() <= 2.5).map(|q| Obb::point(q.pos, PERSON_RADIUS as f64)).collect();
    body_sweep_clear(net, sim, vehicles, me, &parked, &people, side, out_len, level, Profile::Smooth { pre }, if lead.is_some() { Bodies::All } else { Bodies::Standing })
}

/// How far across (m) a bypass `side` metres over `len` metres long has got `x` metres along
/// it: the two arcs at full lock (`full_lock01`) or the smoothstep.
fn bypass_lateral(x: f32, len: f32, side: f32, full_lock: bool) -> f32 {
    let t = (x / len.max(0.1)).clamp(0.0, 1.0);
    side * if full_lock { omsi_sim::traffic::full_lock01(t, len, side) } else { t * t * (3.0 - 2.0 * t) }
}

/// The lateral (m, towards the move) the way of a bypass must have reached where the standing
/// `lead`'s rear is for that lead to be out of my corridor (`perception::lead_by_bodies`: my half
/// width and `LEAD_MARGIN` round my way) with `SQUEEZE_CLEARANCE` to spare for the corridor's
/// straight slices between its samples: the lead's near edge towards the move, my half width,
/// both.
fn bypass_need(me: &Vehicle, lead: &crate::perception::Headway, dir: f32) -> f32 {
    dir * lead.lateral + lead.half_width + me.half_width + crate::perception::LEAD_MARGIN as f32 + SQUEEZE_CLEARANCE
}

/// The gap (m) behind the standing `lead` from which a bypass into `nb` - the shortest move of
/// its kind, the smoothstep (`bypass_min_len`) or with `full_lock` the two arcs
/// (`full_lock_len`) - takes the lead out of my corridor before my front gets there (where its
/// way reaches `bypass_need` across, less my front) and my body's front corners, turned along
/// it, clear the lead's box by `SQUEEZE_CLEARANCE`. A car that closes up nearer than the
/// full-lock room cannot get out without reversing.
fn bypass_room(net: &Network, me: &Vehicle, nb: &NeighbourView, lead: &crate::perception::Headway, full_lock: bool) -> Option<f32> {
    let st = &me.state;
    let target = &net.lanes[nb.lane];
    let (ts, _) = target.nearest_point(me.pos)?;
    let hr = (me.heading as f64).to_radians();
    let right = DVec2::new(hr.cos(), -hr.sin());
    let side = ((target.at(ts).0 - me.pos).truncate().dot(right) as f32).abs();
    let dir = ((target.at(ts).0 - me.pos).truncate().dot(right) as f32).signum();
    let need = bypass_need(me, lead, dir);
    if side <= need {
        return None;
    }
    let len = if full_lock { full_lock_len(me.length(), side) } else { bypass_min_len(me.length(), side) };
    // (the profile rises: halving finds where it is `need` across)
    let (mut lo, mut hi) = (0.0f32, len);
    for _ in 0..24 {
        let x = 0.5 * (lo + hi);
        if bypass_lateral(x, len, side, full_lock) < need {
            lo = x;
        } else {
            hi = x;
        }
    }
    // (and the body turned along the move - its heading the chord of its way
    // `HEADING_CHORD` behind and ahead, as the simulation turns it - keeps `SQUEEZE_CLEARANCE`
    // from the lead's box standing straight at its lateral: the least gap that does, halved for)
    let c = omsi_sim::traffic::HEADING_CHORD as f64;
    let at = |x: f64| bypass_lateral(x as f32, len, side, full_lock) as f64;
    let clears = |gap: f32| {
        let lead_box = Obb::vehicle(DVec2::new(dir as f64 * lead.lateral as f64, (st.front + gap) as f64 + lead.length as f64 * 0.5), 0.0, lead.length as f64 * 0.5, lead.length as f64 * 0.5, lead.half_width as f64);
        let mut d = 0.0f64;
        let mut prev = None;
        while d <= len as f64 + c {
            let slope = (at(d + c) - at(d - c)) / (2.0 * c);
            let b = Obb::vehicle(DVec2::new(at(d), d), slope.atan().to_degrees(), st.front as f64, st.rear as f64, me.half_width as f64);
            if !sample_clear(&mut prev, b, b.separation(&lead_box), SQUEEZE_CLEARANCE as f64) {
                return false;
            }
            d += 0.1;
        }
        true
    };
    let (mut lo, mut hi2) = (0.0f32, len + st.front);
    if !clears(hi2) {
        return None;
    }
    for _ in 0..16 {
        let g = 0.5 * (lo + hi2);
        if clears(g) {
            hi2 = g;
        } else {
            lo = g;
        }
    }
    let corner = hi2 + st.front;
    Some((hi.max(corner) - st.front).max(0.0))
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
/// then kept back at the room a bypass needs before the queue packs to 2 m,
/// from where none could steer out.
fn behind_an_obstacle(net: &Network, sim: &Simulation, vehicles: &[Vehicle], lead: &crate::perception::Headway) -> bool {
    queue_head(net, sim, vehicles, lead).is_some()
}

/// What heads the queue the standing `lead` is the tail of (`QUEUE_LOOK`
/// vehicles on at most, linked by their leads; the one definition of `behind_an_obstacle` and the
/// escape, `queue_escape`): `AtStop`, a bus held at its stop; `ForGood`, a vehicle at the end of
/// its way, or one standing `QUEUE_OBSTACLE_STAND` held by nothing - an AI vehicle not indicating,
/// a player's vehicle not indicating and not held by its red (its claim's `light_red`), the
/// standing counted as continuous time held by nothing; None, a
/// queue that is the road (a red light, a junction, people: their heads are held by them) or
/// anything moving or manoeuvring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QueueHead {
    AtStop,
    ForGood,
}

/// A lead nearer than this (m) holds the car (`Plan::held`: the start reaction applies when it
/// clears); the same test holds a player's vehicle for `queue_head`.
pub const HELD_LEAD_GAP: f32 = 12.0;

/// Is player `id`'s vehicle blocked by a vehicle ahead - an AI vehicle on a lane it
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
    // (standing for good counts the time standing held by nothing, continuous
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

/// A car stood this long (s) behind a queue whose head is an obstacle backs up to the room
/// a bypass needs (`queue_escape`): as long as anything standing takes to be an obstacle whatever
/// holds it (`perception::HELD_OBSTACLE_STAND`, the cleanup's one definition).
pub const QUEUE_ESCAPE_STAND: f32 = crate::perception::HELD_OBSTACLE_STAND;

/// A vehicle's brake gap at `v` (m): SUMO's `MSCFModel::brakeGap` - the
/// reaction distance `v * tau` and the stop `v^2 / 2b` at its comfortable brake.
fn brake_gap(v: f32, tau: f32, b: f32) -> f32 {
    let v = v.abs();
    v * tau + v * v / (2.0 * b.max(0.1))
}

/// How far behind me (m) a vehicle can come from before it would stand: the
/// brake gap at my lane's limit of the road preset that brakes longest (the lorry's: tau 1.8 s,
/// 1.5 m/s², `DriverParams::preset`).
fn rear_look(net: &Network, me: &Vehicle) -> f32 {
    let lorry = DriverParams::preset(VehicleClass::Truck);
    let limit = net.lanes.get(me.state.lane).map_or(0.0, |l| l.speed_limit_kmh / 3.6);
    brake_gap(limit, lorry.tau, lorry.comfort_brake)
}

/// Is the strip behind me clear for backing up `back` m and keeping my standing gap
/// there: my body moved back along my way over every half metre of `back` + `min_gap` touches no
/// vehicle, player's vehicle, parked car or person by `SQUEEZE_CLEARANCE`; a vehicle
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

/// Where a car backed up to `s` on its own lane would wait: SUMO's waiting
/// positions (`legal_before`), its body on no band of its lane - a pedestrian crossing's
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

/// The packed-queue escape: a car that has stood `QUEUE_ESCAPE_STAND` behind a queue whose
/// head stands for good (`queue_head`: `ForGood`, the same head the keep-back and the bypass
/// read), held by nothing but its lead, with a lane beside it may bypass into (`could_bypass`) but
/// nearer its lead than the room the bypass at full lock needs (`bypass_room`), backs up to that room when the
/// strip behind holds the move and its standing gap (`rear_clear`; so the rearmost goes first and
/// the queue dissolves from the back), then pulls out with the bypass. Only
/// on a lane of a road (not a junction object's piece), its body staying on its own lane, and
/// to a place where it may wait (`rest_clear_of_bands`). The metres to back up. (SUMO has no
/// reversing: its vehicles are points on a lane that change lanes from any gap; an OMSI car's
/// body must steer round the car in front, so a queue packed to its standing gap behind a
/// vehicle standing for good closer than the full-lock room would stand for ever.)
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
    let need = [p.left, p.right].into_iter().flatten().filter(|nb| could_bypass(net, sim, me, nb, &l)).filter_map(|nb| bypass_room(net, me, &nb, &l, true)).fold(f32::INFINITY, f32::min);
    // (any short of the room: the bypass starts from the room, `bypass_start`, not from inside it)
    if !need.is_finite() || l.gap >= need {
        return None;
    }
    // (nor where the bypass already clears from here: the room is found by halving and the
    // backing lands on it to the float, so the bypass itself decides, `bypass_start`)
    if [p.left, p.right].into_iter().flatten().any(|nb| could_bypass(net, sim, me, &nb, &l) && bypass_start(net, sim, vehicles, me, &p.ahead, p.road_left, &nb, &l, None).is_some()) {
        return None;
    }
    let back = need - l.gap;
    // (the body stays on its own lane, and rests where it may wait)
    if st.s - st.rear - back < 0.0 || !rest_clear_of_bands(net, sim, me, st.s - back) {
        return None;
    }
    rear_clear(net, sim, vehicles, me, back).then_some(back)
}

/// The pull-back into the lane is sampled at this spacing (m), as the swept moves
/// (`body_sweep_clear`).
const REENTRY_STEP: f32 = 0.5;

/// The way back into the lane of a vehicle out of it, as a sideways
/// ramp (`AiState::lateral_ramp`: from, to, start odometer, length): the ramp laid if it ends at
/// the offset the vehicle is asked for (a bay, the bus service sends 0 as it leaves one; else
/// its target), else the S-curve the motion lays for a new target (`AiState` step,
/// `s_curve_len`) from where the body is now. The ramp laid can still be the way IN for a frame:
/// the bay's target turns back before the service's way out is laid, and read as the way back
/// it swept the bay outward and, its start moved on by the hold, put the body on the lane's
/// axis (sideways onto a car passing the bay).
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

/// A vehicle out of its lane on its way back in: its body clear of the lane's axis (a
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

/// Is the way back into the lane clear of bodies standing on it? My body on the bent way
/// over what is left of the ramp (`shifted_body`, the way carries the ramp), turned along it as
/// the bypass's sweep, against every other vehicle's and player's box where it is now; a
/// vehicle going my way with its rear ahead of my front is a lead, not a threat.
/// No gap of its own: who comes up behind and must be able to stop for me is the one pull-out
/// rule's (`pull_out_clear`: each follower's own secure gap, the players too); this sweep adds
/// only the bodies the way back in would run into (a van standing beside my nose).
fn reentry_clear(net: &Network, sim: &Simulation, vehicles: &[Vehicle], me: &Vehicle) -> bool {
    // (the poses on the way back, `way_back`: a ramp laid that ends
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
    // (the poses it has not reached yet: where it stands now is no move into
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

/// The parked cars within `reach` metres of my origin as boxes, on any lane
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

/// The lateral profile a swept move follows (`body_sweep_clear`).
#[derive(Debug, Clone, Copy, PartialEq)]
enum Profile {
    /// A smoothstep over the out length after `pre` metres straight on (a pass's, a lane
    /// change's after its signalling).
    Smooth { pre: f32 },
    /// Two arcs at one lock (`full_lock01`, a bypass's) after `pre` metres straight on.
    FullLock { pre: f32 },
}

/// One more sample of a swept move (`body_sweep_clear`, `bypass_room`): the body's pose `b`, its
/// least separation `sep` from what is swept, after the sample `prev` (updated). Between two
/// samples no point of the body moves further than the largest corner displacement `shift`
/// between the two poses (a rigid move displaces a box's points most at a corner), so a pose a
/// share u of the way is at least `psep - u shift` and `sep - (1 - u) shift` clear: the poses
/// between keep `clearance` where `(psep + sep - shift) / 2` does. A sweep sampled every few
/// centimetres holds between its samples too.
fn sample_clear(prev: &mut Option<(Obb, f64)>, b: Obb, sep: f64, clearance: f64) -> bool {
    let ok = match *prev {
        Some((pb, psep)) => {
            let shift = pb.corners().iter().zip(b.corners().iter()).map(|(a, c)| a.distance(*c)).fold(0.0f64, f64::max);
            0.5 * (psep + sep - shift) >= clearance
        }
        None => sep >= clearance,
    };
    *prev = Some((b, sep));
    ok
}

/// Which vehicles and players a swept move is checked against (`body_sweep_clear`).
#[derive(Debug, Clone, Copy, PartialEq)]
enum Bodies {
    /// Every one near, where it is now.
    All,
    /// Only those standing (under `STANDING_SPEED`): where they are now is where they will be
    /// while the move is made; the moving ones are the gap rule's.
    Standing,
}

/// Does my body, turned along a move `side` metres over `out_len` metres (`profile`), clear
/// the vehicles and players near that `bodies` names, the parked cars (`parked`) and people's
/// discs (`people`) by `SQUEEZE_CLEARANCE`, up to `level` metres on?
#[allow(clippy::too_many_arguments)]
fn body_sweep_clear(net: &Network, sim: &Simulation, vehicles: &[Vehicle], me: &Vehicle, parked: &[Obb], people: &[Obb], side: f32, out_len: f32, level: f32, profile: Profile, bodies: Bodies) -> bool {
    let st = &me.state;
    let counts = |speed: f32| bodies == Bodies::All || speed.abs() < STANDING_SPEED;
    let near: Vec<usize> = sim.vehicles_near(me.pos, (level + 20.0) as f64).into_iter().filter(|&j| counts(vehicles[j].state.speed)).collect();
    let clearance = SQUEEZE_CLEARANCE as f64;
    // (the last sample's pose and its least separation)
    let mut prev: Option<(Obb, f64)> = None;
    let mut d = 0.0f32;
    while d <= level {
        let (lat, slope) = match profile {
            Profile::Smooth { pre } => {
                let t = ((d - pre) / out_len.max(0.1)).clamp(0.0, 1.0);
                (side * t * t * (3.0 - 2.0 * t), side * 6.0 * t * (1.0 - t) / out_len.max(0.1))
            }
            // (the body's heading as the simulation gives it, the chord of its way
            // `HEADING_CHORD` behind and ahead: on the arcs' tight bends it lags the tangent)
            Profile::FullLock { pre } => {
                let c = omsi_sim::traffic::HEADING_CHORD;
                let at = |x: f32| full_lock_at((x - pre).clamp(0.0, out_len), out_len, side).0;
                (at(d), (at(d + c) - at(d - c)) / (2.0 * c))
            }
        };
        let p = st.way_point(net, d).truncate();
        let p2 = st.way_point(net, d + 1.0).truncate();
        let f = (p2 - p).normalize_or_zero();
        let hr = (me.heading as f64).to_radians();
        let f = if f.length_squared() < 0.5 { DVec2::new(hr.sin(), hr.cos()) } else { f };
        let right = DVec2::new(f.y, -f.x);
        let dir = (f + right * slope as f64).normalize();
        let b = Obb::vehicle(p + right * lat as f64, dir.x.atan2(dir.y).to_degrees(), st.front as f64, st.rear as f64, me.half_width as f64);
        // (the least separation of this pose from what is swept)
        let mut sep = f64::INFINITY;
        for &j in &near {
            let o = &vehicles[j];
            if o.id != me.id && (o.pos.z - me.pos.z).abs() <= 3.0 {
                sep = sep.min(o.obb().separation(&b));
            }
        }
        for pl in sim.players.iter().filter(|pl| counts(pl.speed) && (pl.pos.z - me.pos.z).abs() <= 3.0) {
            sep = sep.min(pl.obb(0.0, 0.0).separation(&b));
        }
        for o in parked.iter().chain(people) {
            sep = sep.min(o.separation(&b));
        }
        if !sample_clear(&mut prev, b, sep, clearance) {
            return false;
        }
        // (the arcs' bends turn the body fast: sampled finer)
        d += if matches!(profile, Profile::FullLock { .. }) { 0.05 } else { 0.5 };
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

/// The braking (m/s²) a driver reckons with when asking whether it can still
/// stop before a line or a meeting place: hard, short of an emergency stop (`MAX_BRAKE`).
/// One figure for the light (`light_call`), the claim's committed and clearing tests and the
/// player's red (before: 0.8 of `MAX_BRAKE` at the light, 0.7 at the junction).
pub const HARD_BRAKE: f32 = omsi_sim::traffic::MAX_BRAKE * 0.8;

/// Too late to stop within `room` metres at `v` m/s braking `brake` (m/s²;
/// `HARD_BRAKE` but inside a junction); under 1 m/s a car can always stop.
pub fn cannot_stop(v: f32, room: f32, brake: f32) -> bool {
    v > 1.0 && room < v * v / (2.0 * brake)
}

/// The braking (m/s²) a driver always accepts to stop at an amber line,
/// SUMO's `--tls.yellow.min-decel` (3 m/s², sumo.md §7: "the deceleration needed is compared
/// against" it).
pub const YELLOW_MIN_DECEL: f32 = 3.0;

/// Can the driver stop within `gap` (m) at `v` (m/s) on amber -
/// SUMO's yellow: the vehicle stops when v²/2gap is within its decel or `YELLOW_MIN_DECEL`,
/// whichever is more? Standing it can. (Its comfortable brake alone, 2 m/s², would commit a car
/// 56 m out at 14.86 m/s, needing 1.95, that crosses 1.8 s into red.)
pub fn stops_comfortably(v: f32, gap: f32, dp: &DriverParams) -> bool {
    v <= STANDING_SPEED || (gap > 0.0 && v * v / (2.0 * gap) <= dp.comfort_brake.max(YELLOW_MIN_DECEL))
}

/// The driver's call at the light, SUMO's yellow and red rule: through
/// on green; on amber the driver asks every frame whether it can stop with comfortable
/// braking (`stops_comfortably`) - if not it goes, if so it stops (SUMO re-tests it every
/// step; the last amber answer is kept as `Vehicle::light_decision`); on red only a
/// car that cannot stop at all (`cannot_stop`, `HARD_BRAKE`) or one committed at amber onset
/// and still rolling crosses. Nothing is decided on green, no decision is extended past
/// what the onset allowed, and a committed car that came to a stand waits for its green
/// (one standing short of its line must not set off over it on red). "Cannot stop" is
/// reckoned to the line itself, as SUMO's red: a stop must not lie over the line.
pub fn light_call(me: &Vehicle, p: &Perception, sig: &SignalView) -> LightCall {
    let st = &me.state;
    let dp = &me.params;
    let v = st.speed;
    let gap = sig.dist - st.front;
    let queued = v < 1.0 && p.leader.map(|l| l.gap < 8.0).unwrap_or(false);
    let light = (sig.ctl, sig.light);
    // (that light's line: the same light lights several approaches)
    let line = st.odometer + gap;
    let held = me.light_decision.filter(|d| d.0 == light && (d.2 - line).abs() < 2.0).map(|d| d.1);
    match sig.aspect {
        Aspect::Green | Aspect::GreenYellow | Aspect::Dark => LightCall { go: true, decision: None },
        Aspect::Yellow => {
            // (and never from a standstill or a queue: at rest nothing is "too late to stop";
            // nor into a box whose exit is not free - SUMO's no-block check,
            // sumo.md - unless it cannot stop at all)
            let exit_free = p.junction.as_ref().is_none_or(|j| j.exit_room);
            // (SUMO's yellow rule, re-tested every step (`MSVehicle::planMoveInternal`:
            // `yellow && canBrake`, canBrake = the brake gap at the vehicle's decel - here its
            // comfortable brake or `--tls.yellow.min-decel` 3 m/s², whichever is more - within the
            // distance to the line): a car that slowed for anything and can then stop, stops; the
            // decision is not kept from the amber onset. The line is the stop place: no 0.5 m past
            // it)
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
/// and a step; `hold_point`'s band). (Somebody on the crossing is held for by
/// `perception::people_ahead`, the one people test.)
pub const ZEBRA_STOP: f32 = 2.5;
pub const ZEBRA_PAST: f32 = 2.0;

/// A lead slower than this (m/s) is a queue I come to rest behind, not one I follow through
/// (`light_call`'s `queued`: under 1 m/s behind a lead).
const QUEUE_SPEED: f32 = 1.0;
/// A lead under this speed (m/s) that is itself held (a lead, a light, a junction) is coming
/// to rest in a queue too, and so is one braking for a stop of its own at any speed
/// (`Headway::rest`).
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

/// A car comes to rest up to this far (m) short of its stop (the controller's 0.6 m
/// `STOP_LINE_GAP` and a creep): `pulled_back` keeps that much more of the body out of a band
/// (the rear would otherwise stand inside a convergence the car is stopped behind).
pub(crate) const REST_SHORT: f32 = 1.0;
/// A refused car whose waiting place lies more than this behind its front (m) is past it: it is
/// given no stop there and rolls on (`plan`'s refusal stop; `reservation::rests_short` reads
/// the same).
pub(crate) const STOP_BEHIND_FRONT: f32 = 0.3;

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

/// Under this speed (m/s) an eased-off car stands, and is pinned where it is
/// (`Claim::standing`'s 0.3 m/s).
const EASE_STANDING: f32 = 0.3;

/// Where a car its grant refuses stops (a distance from its origin), one rule: refused
/// for a foe, just before the first stretch it waits for (`Grant::at`); held for anything else (a
/// full exit, SUMO's keep-clear), at the line where it can still stop there gently, else a metre
/// short of the first meeting place of its way (`first`). The stop is the line (SUMO's link end)
/// while the car is short of it, else the last place short of that target where its body lies on
/// no band (`legal_before`: SUMO's waiting position before the conflict on the internal lane;
/// none left, the hold point or `next_wait`). A foe it is the follower or the leader of lets it
/// go (no stop), a zipper caps its speed (`Grant::zipper`); the zipper takes the side merges.
fn refused_stop(net: &Network, sim: &Simulation, me: &Vehicle, j: &JunctionView, g: &Grant) -> f32 {
    let st = &me.state;
    let dp = &me.params;
    let v = st.speed;
    let at_line = j.line - 0.3;
    let gently = v * v / (2.0 * (at_line - st.front - 0.6).max(0.05)) <= dp.comfort_brake * 1.5;
    let line_stop = !j.inside && (gently || v < 0.5);
    let bands = wait_bands(net, sim, j);
    let first = if line_stop { at_line } else { j.conflict_start - 1.0 };
    // (inside and past that metre already: the first band still ahead of the front -
    // a car held for a full exit with the first meeting place behind it was stopped where it
    // was and rolled onto the walk ahead)
    let first = if first < st.front - 0.3 { bands.iter().map(|b| b.0).filter(|&a| a > st.front - 0.3).fold(f32::INFINITY, f32::min).min(j.conflict_end) } else { first };
    let target = match (g.hold, g.at) {
        (Some(Hold::Foe), Some(waited)) => j.entry + waited - 1.0,
        _ => first,
    };
    // the line (SUMO's link end) while the car is short of it, else the last spot short of the
    // target where the body lies on no band; none left (the body is over a band it cannot leave
    // before the target - the resolver lets such a car go, `Claim::next_wait`, unless the foe is
    // in the meeting place already): the hold point as before
    // (the line too only where the body there lies on no band)
    let len = me.length();
    let line = (line_stop && at_line <= target && !bands.iter().any(|b| b.0 < at_line && b.1 > at_line - len - REST_SHORT)).then_some(at_line);
    match legal_before(&bands, me, target) {
        Some(p) => line.map_or(p, |l| p.max(l)),
        // (held for anything but a foe - a full exit -, past its last place to wait:
        // on to the next one, `next_wait`, as the claim's commitment says; the lead still stops
        // it behind a queue)
        None if g.hold != Some(Hold::Foe) => line.unwrap_or_else(|| next_wait(net, sim, me, j, if j.inside { st.decel * 1.5 } else { HARD_BRAKE })),
        None => line.unwrap_or_else(|| hold_point(net, sim, me, j, target - j.entry + 1.0)),
    }
}

/// The bands a waiting car's body may not lie on (SUMO waits at link ends and
/// internal junctions, never on a crossing link): the crossings with other ways (`Cross`, `Block`
/// meeting places, from a metre before), the pedestrian crossings and the convergences - not the
/// merges, where the zipper puts the two one behind the other; a queue stands in a side
/// merge as in a lane).
pub(crate) fn wait_bands(net: &Network, sim: &Simulation, j: &JunctionView) -> Vec<(f32, f32)> {
    let mut bands: Vec<(f32, f32)> = sim.junctions.foes_along(j.junction, &j.chain).iter().filter(|f| f.kind != crate::junction::FoeKind::Merge).map(|f| (j.entry + f.mine.0 - 1.0, j.entry + f.mine.1)).collect();
    bands.extend(walk_bands(net, sim, j));
    bands.extend(converge_bands(sim, j).into_iter().map(|(a, b, _)| (a, b)));
    bands
}

/// SUMO's waiting positions: the last place at or before `limit` (the front's
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

/// SUMO's waiting positions: the first place the car can come to rest at
/// with its body on no band of the way through (`wait_bands`), braking at `brake` (the claim's,
/// `Simulation::tick`) - a distance from the origin. Until there it cannot wait: it is committed
/// to every meeting place short of it (`Claim::next_wait`, SUMO's `willPass`).
pub fn next_wait(net: &Network, sim: &Simulation, me: &Vehicle, j: &JunctionView, brake: f32) -> f32 {
    let st = &me.state;
    // (from its stopping place at any speed - its go took it there: a car standing
    // with its body on a band cannot wait there; the
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
        // a queue over a pedestrian crossing: behind a lead standing past a cut of
        // my way, where closing up to it would put my body in the zebra band, I stop at the
        // zebra stop instead, as `hold_point` keeps a refused car off it
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
            } else if at > st.front - STOP_BEHIND_FRONT {
                out.hold = g.hold;
                out.wait_for = g.wait_for;
                let w = if g.hold == Some(Hold::ExitFull) { Why::ExitFull } else { Why::Junction };
                min_stop(&mut out.stop, &mut out.why, Some(at), w);
            }
        }
    }

    // the zipper speed (SUMO's `getZipperSpeed`) for the foe ahead at a merge that crosses
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
        // the position bound: short of the stretch until the foe's rear has been past
        // its start `tau`, the car waits at the start (`ZipCap::hold`), at a waiting position
        // (the resolver gives a cap from a foe short of the stretch only within the car's zipper
        // reach, `reservation::zipper_reach`: farther off there is no hold)
        if let (Some(d), Some(j)) = (z.hold, p.junction.as_ref()) {
            if let Some(at) = legal_before(&wait_bands(net, sim, j), me, st.front + d) {
                min_stop(&mut out.stop, &mut out.why, Some(at.max(st.front - 0.3)), Why::Junction);
            }
        }
        // SUMO's waiting positions: where following the foe ahead brings the car
        // to rest - behind it at the join, or, the foe in the stretch, behind where it comes to
        // rest itself while it is coming to rest (`QUEUE_COMING`, `QUEUE_BRAKE`) - it rests at the
        // last place short of that with its body on no band, never on a crossing.
        // (Not behind a foe still short of the join in its own lane: that
        // one is no lead, the cap alone follows it)
        if let Some(j) = p.junction.as_ref().filter(|_| z.stop || (z.inside && z.speed < QUEUE_COMING)) {
            let rest = st.front + z.gap + z.speed * z.speed / (2.0 * QUEUE_BRAKE) - st.min_gap;
            if let Some(at) = legal_before(&wait_bands(net, sim, j), me, rest) {
                min_stop(&mut out.stop, &mut out.why, Some(at.max(st.front - 0.3)), Why::Junction);
            }
        }
    }

    // asked to make room for a car whose lane ends beside me (SUMO's cooperative
    // follower, `informFollower`): its rear my lead, braking no harder than my comfortable
    // brake times `HELP_DECEL_FACTOR` (`Vehicle::help_for`)
    // (it keeps its own standing gap behind the place the changer's acceptance
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
    // pinned there only while it stands and its grant offers no follow; with one
    // (the other rolls on into a merge of the two ahead) it rolls in behind it, the follow
    // branch above; rolling with none, it stops gently
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

    // 4. people (a pull-out too: `people_ahead` leaves out only the
    // people under or beside its parked body off its way out, `pull_out_exempt`)
    min_stop(&mut out.stop, &mut out.why, p.people_stop, Why::People);

    // 5. the app's own holds
    min_stop(&mut out.stop, &mut out.why, me.external.stop_at, Why::Service);
    // a car leaving a parking space starts only when its lane has room behind
    // it: the follower there not within the gap a lane change needs from it (`gap_needed_behind`:
    // its standing gap, the time gap and the braking distance of the speed it closes in at) -
    // SUMO's parked vehicle re-enters by insertion, which checks the follower's secure gap on the
    // lane it enters (`MSLane::isInsertionSuccess`, `getSecureGap`). Until then it stays parked
    // and its clock waits
    // standing out of the lane and about to move back in - a parked car
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

    // backing up out of a packed queue (`queue_escape`): to the place fixed when
    // it began, the speed braking to it (`AiState::reverse_step`); every frame the strip behind is
    // swept again (`rear_clear`) and the queue looked at: any change and the car stops, a committed
    // stop over its brake distance from where it is (never past the place fixed), kept to the end
    if let Some((target, stopping)) = me.reversing {
        let left = st.odometer - target;
        let queue_stands = p.leader.is_none_or(|l| l.speed.abs() < STANDING_SPEED);
        let go_on = !stopping && left > 0.0 && queue_stands && rear_clear(net, sim, vehicles, me, left);
        // (the committed stop is never past the place fixed: backing stops there at the latest)
        out.reverse = Some(if stopping || go_on { (target, stopping) } else { ((st.odometer - st.speed * st.speed / (2.0 * st.decel.max(0.5))).max(target), true) });
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

    // SUMO LC2013's keep-right wish accumulates every step the driver is not changing
    // lanes (`wantsChange` runs each step), towards a lane beside on the keep side it may change
    // onto
    // (the speeds anticipated on my lane and each lane beside, `anticipated_speeds`, are worked
    // out once a frame: the keep-right rate, the speed-gain step and the desire read the same)
    let mut ant_memo: [Option<(usize, (f32, f32))>; 2] = [None, None];
    let mut ant_own: Option<f32> = None;
    let mut ant = |nb: &NeighbourView| -> (f32, f32) {
        let k = (nb.dir as usize).clamp(1, 2) - 1;
        match ant_memo[k] {
            Some((l, a)) if l == nb.lane => a,
            _ => {
                let a = anticipated_speeds(net, sim, me, p, nb, &mut ant_own);
                ant_memo[k] = Some((nb.lane, a));
                a
            }
        }
    };
    if st.change.is_none() && me.passing.is_none() && !me.is_rail() && st.route.is_empty() {
        // (a strategic stay comes first and leaves the wish as it is)
        if let Some(nb) = [p.left, p.right].into_iter().flatten().find(|nb| nb.dir == keep_side(sim) && may_change_to(net, sim, me, nb) && !stays_on_best(net, sim, me, nb)) {
            out.keep_right_rate = Some(keep_right_rate(net, vehicles, me, p, &nb, ant(&nb)));
        }
        // (and the speed-gain wishes towards each lane beside, `speed_gain_step`)
        for nb in [p.left, p.right].into_iter().flatten() {
            out.speed_gain_step[(nb.dir as usize).clamp(1, 2) - 1] = Some(speed_gain_step(sim, &nb, ant(&nb)));
        }
    }

    // a lane change of the driver's own (a bypass too) checks its gap again on every frame of
    // its signal, and that nobody is on the target lane over the move (`people_on_move`) - a
    // bypass also that its move is still clear (below) - and is dropped if any fails before the
    // car moves over (SUMO's `checkChange` runs every step
    // until the change is carried out, and blocks it for a pedestrian on the target lane within
    // the brake gap, `MSLaneChanger::checkChange` `nextBlocking`); a change the lane's end asks for keeps
    // informing the follower there (LC2013's `informFollower`, every step the strategic change
    // is wanted and not yet carried out: SUMO carries it out in the step it is accepted, the
    // OMSI driver signals first, and the follower that made room is not let go meanwhile)
    if let Some(c) = st.change.filter(|c| c.wait > 0.0 && c.t <= 0.0 && st.route.is_empty()) {
        if let Some(nb) = change_target_view(p, c.to, c.dir) {
            if let (true, Some(f), Some((gap, vb))) = (lane_end_desire(net, sim, me, p, &nb) > 0.0, nb.behind_id, nb.behind) {
                if gap > 0.0 {
                    out.help = Some((f, gap, v, gap_needed_behind(me, &nb, vb, v)));
                }
            }
            out.abort_change = !gap_accepted(me, &nb) || people_on_move(net, sim, me, &nb);
            // (a bypass from rest also sweeps its move again, as it did at its start,
            // `bypass_start`: every body near, not only the target lane's follower - one that came
            // to stand in the swept band or alongside since; SUMO checks every sublane it moves
            // into on each step. The sweep runs as at the start, on my own lane's way without the
            // change laid on it and from the pose the car has on that way (`sim::way_pose`: late
            // in the signal the chord heading turns into the move laid ahead, a pose the move's
            // own samples sweep), round the thing the bypass was started round with what stands
            // beyond it there (`Perception::bypassed`; not the lead, which with the change laid
            // on the way can be the target lane's car), while it stands; it sweeps the kind of
            // move started, the smoothstep or the two arcs at full lock, as the car drives that one
            // and not whichever clears now. Once that thing has moved off or gone from my way, the
            // car still drives the bypass move it started, checked only by the gap and people
            // rules above, as it always was when its lead left)
            if !out.abort_change && c.bypass {
                if let Some(b) = p.bypassed.as_ref().filter(|b| b.lead.speed < STANDING_SPEED) {
                    let mut before = me.clone();
                    before.state.drop_change();
                    (before.pos, before.heading) = crate::sim::way_pose(net, &before.state);
                    out.abort_change = bypass_start(net, sim, vehicles, &before, &b.ahead, p.road_left, &nb, &b.lead, Some(c.full_lock)).is_none();
                }
            }
        }
    }

    // a bypass signalling stands where it is until it moves over (the room it was started
    // with is the room it has: rolled on meanwhile, the lead stayed in its corridor and held it
    // half out)
    if st.change.is_some_and(|c| c.bypass && c.wait > 0.0) {
        min_stop(&mut out.stop, &mut out.why, Some(st.front + v * v / (2.0 * dp.comfort_brake.max(0.1))), Why::Lead);
    }

    // 7. a lane change of the driver's own: round something standing, for speed, to keep
    //    to the side, into the lane of the turn ahead
    let free_to_change = st.change.is_none() && st.route.is_empty() && st.change_cooldown <= 0.0 && me.passing.is_none() && out.start.is_none() && !me.is_rail() && lane.length() - st.s > 9.0;
    if free_to_change {
        let near_junction = p.junction.as_ref().map(|j| !j.inside && j.line < JUNCTION_KEEP).unwrap_or(false);
        let mut best: Option<(f32, NeighbourView, ChangeWhy)> = None;
        // the strongest lane-end wish whose gap is not there
        let mut blocked_end: Option<(f32, NeighbourView)> = None;
        for nb in [p.left, p.right].into_iter().flatten() {
            let (d, ok, why) = change_desire(net, sim, me, p, &nb, near_junction, ant(&nb));
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
            // (from rest behind a lead within a change's move - an obstacle or a queue - the car
            // pulls out at full lock, `bypass_start`: SUMO's change is instantaneous, an OMSI
            // body steers out round the one in front)
            // (under way the move itself must clear the parked cars and people near, and with a
            // lead standing within it every body near, `change_clear`; one moving on is followed
            // during the change)
            let near = p.leader.filter(|l| l.speed < STANDING_SPEED && l.gap < omsi_sim::traffic::change_length(v.max(0.0)) + v * omsi_sim::traffic::SIGNAL_BEFORE_CHANGE);
            out.start = match near {
                Some(l) if v < 1.0 => bypass_start(net, sim, vehicles, me, &p.ahead, p.road_left, &nb, &l, None),
                Some(l) => change_clear(net, sim, vehicles, me, p, &nb, Some(&l)).then_some(Start::Change { to: nb.lane, dir: nb.dir }),
                None => change_clear(net, sim, vehicles, me, p, &nb, None).then_some(Start::Change { to: nb.lane, dir: nb.dir }),
            };
            out.change_why = Some(why);
        } else if let Some((_, nb)) = blocked_end.filter(|b| b.0 >= threshold) {
            // SUMO LC2013's cooperation (`informFollower`, sumo.md 3.3): the follower there,
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
                // (SUMO's `neededGap`/`decelGap` test before the `helpDecel` branch:
                // asked to brake only if it can still open the gap from where it is - braking at
                // its comfortable brake (times `HELP_DECEL_FACTOR`) to rest while I go on to my
                // last stop before the lane's end, the gap at rest is then its standing gap;
                // else it is told not to brake and I change in behind it. Asked when it could
                // not, the second follower of a chain would stand behind my rear closer than the
                // gap the acceptance needs at rest, and all would stand)
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
            // (once placed it is kept - a latch, `Vehicle::lane_end_stop` - until the change
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
        // obstacle is a lead again
        // (past it, on its way back in, it is stranded too while it stands still out there, held
        // by a red light or a queue)
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
        let two_lane = p.left.is_none() && p.right.is_none();
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
        // and no AI vehicle whose indicator shows the side the pass would take (`side` m, + right):
        // it has announced a move into that way - a step-out, a change or a turn - and is not
        // overtaken there (SUMO: a vehicle does not overtake within the lane one that has
        // announced a change into its path, the sublane model's blinker; as the bus and the
        // player above)
        let vehicle_out = |side: f32| match l.what {
            Seen::Vehicle(j) | Seen::Body(j) => side != 0.0 && vehicles.get(j).is_some_and(|o| o.blinker_shown.0 == if side < 0.0 { 1 } else { 2 }),
            Seen::Player(_) | Seen::Parked => false,
        };
        // a bus does not pass what stands at its own next stop
        let own_stop = me.external.stop_at.map(|s| s < l.gap + l.length + st.front + 15.0).unwrap_or(false);
        let real = l.gap + if player { 0.5 } else { 0.0 };
        // how far the car has to swing out: past the thing inside the drawn road if that
        // leaves room beside where it stands (the lane's drivable extent on that side, L1:
        // to the kerb, the pavement or the next lane's painted edge), else onto the
        // oncoming lane
        // (my body keeps `MIN_GAP_LAT` from the thing's body where it really is - its corners
        // across my way, `body_across`, SUMO's sublane model: a body standing askew or reaching
        // only its corner into my width is passed by that much, not by its middle)
        let towards_middle = sim.oncoming_sign();
        let (lo, hi) = body_across(net, sim, vehicles, me, &l);
        let away = if l.lateral * towards_middle > 0.3 { -towards_middle } else { towards_middle };
        let squeeze_side = if away < 0.0 { lo - MIN_GAP_LAT - me.half_width } else { hi + MIN_GAP_LAT + me.half_width };
        // (a junction object's piece has no edge data: its lane reaches halfway to the piece
        // laid beside it, `Network::piece_beside` - the paths tile the carriageway - and no
        // further than its half width where none lies beside)
        let (drivable_left, drivable_right) = if lane.source == 2 {
            let half = |nb: Option<NeighbourView>| nb.and_then(|nb| net.lanes[nb.lane].nearest_point(me.pos)).map_or(lane.width * 0.5, |(_, d)| (d as f32 * 0.5).max(lane.width * 0.5));
            (half(p.left), half(p.right))
        } else {
            lane.drivable()
        };
        // (and never into the traffic of the lane beside or the oncoming lane: a car's half
        // width and 0.3 off their axes, whatever the stored extent says - a recording or a
        // piece without edge data; beyond that it is the wheels over
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
        // (the move the squeeze starts with: the usual one, or from nearer than that the
        // shortest the body can steer, `bypass_min_len` - the out length the start uses below)
        let squeeze_out_len = squeeze_out(me, &l, squeeze_side);
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
            && !vehicle_out(squeeze_side)
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
        // or player standing or slower than 2 m/s there takes it)
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
            // (inside a junction a squeeze keeps off the stretch where the ways meet: its
            // body out of its lane's middle there would be where the junction's foes are
            // not reckoned to be)
            if squeeze && j.inside {
                return squeeze_run > j.conflict_start && j.conflict_end > -st.rear;
            }
            let keep = j.line - st.front < JUNCTION_KEEP && !(squeeze && real < SQUEEZE_NEAR);
            j.line < real + st.front + obstacle_len + zone || keep
        }).unwrap_or(false);
        let road_ok = p.road_left > pass_len + real + 30.0;
        let oncoming = p.oncoming.filter(|o| !solid && o.side.abs() >= 2.3 && o.side.abs() <= 5.5 && !o.blocked);
        // TODO: `clear_for` is the oncoming car's time to the passer's
        // current position, compared with the pass time + 1.5 s; a car coming at 13 m/s meets
        // the passer before that point. It should be the time to where the pass ends.
        let oncoming_free = oncoming.is_some_and(|o| {
            let v_cap = (p.v_want).clamp(4.0, 14.0);
            o.clear_for > arrival_time(pass_len + Passing::back_len_for(side), v, st.accel, v_cap) + 1.5
        });
        let pass_possible = !queue_only && can_pass && !own_stop && !player_out && !bus_out && !vehicle_out(side) && room_beyond && real >= 0.3 && !light_near && !junction_near && road_ok && (squeeze || (two_lane && standing && oncoming_free));
        // (or round it into the lane beside, `Start::Bypass`)
        let bypass_nb = [p.left, p.right].into_iter().flatten().find(|nb| could_bypass(net, sim, me, nb, &l));
        let bypass_possible = standing && !light_near && !junction_near && bypass_nb.is_some();
        // (a bypass needs the room its body can steer round the obstacle in, `bypass_room`)
        let room = match bypass_nb.filter(|_| bypass_possible && !pass_possible) {
            Some(nb) => bypass_room(net, me, &nb, &l, false).unwrap_or(room),
            None => room,
        };
        // keep back: room to pull out again, where the car can still stop there gently - only
        // where a pass (or a bypass) could start now; behind a standing player at a red 40 m
        // on, or a queue at a junction, the car closes up to its queue gap
        // where no pass could ever start
        // (a bypass's room is held once reached: closing in from 0.5 m beyond it, the car rolled
        // on to 2 m behind a bus at its stop and could never steer out)
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
        // a pass on the oncoming lane starts only when its path is swept as the
        // bypass's is (`body_sweep_clear`): the body turned along its S-curve (the out
        // length it will drive) out and level with what it passes, clear of every vehicle's
        // and player's body by `SQUEEZE_CLEARANCE` - not on the oncoming lane's gap alone.
        // Only the start: the keep-back still reads the pass as possible (gating it too makes a
        // car close up and later start where it should not), and not the
        // squeeze's linear corridor (`squeeze_clear`: from a queue gap behind a parked car its
        // straight ramp cuts the car passed and the pass never comes).
        let pass_out_len = if squeeze { squeeze_out(me, &l, side) } else if too_close { (l.gap - 0.3).clamp(2.0, out_len) } else { out_len };
        // (and the parked cars along the way and the people near it, as the
        // squeeze's corridor has them)
        let pass_swept = || {
            if squeeze {
                return true;
            }
            let level = real + obstacle_len + st.front + st.rear + 2.0;
            let parked = parked_boxes(net, sim, me, level + st.front);
            let people: Vec<Obb> = sim.people_near(me.pos, (level + st.front) as f64).into_iter().map(|k| &sim.people[k]).filter(|q| (q.pos.z - me.pos.z).abs() <= 2.5).map(|q| Obb::point(q.pos, PERSON_RADIUS as f64)).collect();
            body_sweep_clear(net, sim, vehicles, me, &parked, &people, side, pass_out_len, level, Profile::Smooth { pre: 0.0 }, Bodies::All)
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
        // (a timetable bus has a bay only at or into its stop: between stops it
        // swerves round a parked car as everybody does - not towards the middle past the
        // people at the kerb, whom it serves)
        if let Some(bay) = me.external.bay {
            out.lateral = Some(bay);
        } else if me.serves_stops() {
            out.lateral = Some(p.parked_swerve.unwrap_or(0.0));
        } else {
            out.lateral = Some(p.kerb_swerve.unwrap_or(0.0));
            // easing past people beside the lane: gently, as a car edging out
            if p.people_swerve && v < 3.0 && out.accel_cap.is_none() {
                out.accel_cap = Some(omsi_sim::ai_motion::PULL_OUT_ACCEL);
            }
        }
    }

    // 10. the end of a fixed route: stop at its end. Not where the road
    //     itself ends (a street lane that runs into nothing, no stop of the bus's own on it):
    //     there the bus keeps its speed and is taken off as it reaches the end (`AiState::
    //     drive` finds no way on), as a plain car at a dead end; the timetable puts it back
    //     where its track goes on. Held 0.5 m before the end it would stand there for good with its
    //     queue behind it.
    // Nor where its trip is over at the end of its track on a lane with a way
    //     on (`External::trip_over`): it would stand at Why::End with its next trip never
    //     handed over.
    if !st.route.is_empty() && st.route_index + 1 >= st.route.len() && st.planned_next.is_none() {
        let end = lane.length() - st.s - 0.5;
        let dead_end = lane.next.is_empty() && lane.kind == omsi_sim::traffic::LaneKind::Street;
        // (nor where the stop it is asked for lies beyond the end of its
        // route: the route's next part is not loaded yet, it waits at the end - a bus whose
        // stop lies past its route's end would otherwise drive off it)
        let stop_beyond = me.external.stop_at.is_none_or(|sa| sa > end + st.front + 0.5);
        if end < 150.0 && stop_beyond && me.class == VehicleClass::ScheduledBus && !dead_end && !me.external.trip_over {
            min_stop(&mut out.stop, &mut out.why, Some(end.max(0.0)), Why::End);
        }
    }

    // 10b. the edge of the loaded tiles (`Lane::loaded_edge`): the
    //      road goes on into a tile that is not loaded; the car stops before the lane's end, as a
    //      bus at its loaded route's end, and the game takes it off once nobody sees it (a
    //      dead-end removal) or lets it on when the tile comes
    // unless the edge is its trip's end (`External::edge_trip_end`,
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
    //     pulled back before a walk there. A way through that crosses
    //     nothing has no box to keep clear (`perception::junction_ahead`).
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
                // (the hold replaces a stop further on - a refusal, people at a waiting place
                // inside - never one nearer: a person already under the front keeps the car
                // standing, it is not pulled on to the line)
                if out.stop.is_none_or(|s| s >= hold) {
                    out.stop = Some(hold);
                    out.why = Why::KeepClear;
                }
            }
        }
    }

    out.held = out.stop.is_some() || p.leader.map(|l| l.gap < HELD_LEAD_GAP).unwrap_or(false);
    if out.why == Why::None && out.lead.is_some() {
        out.why = Why::Lead;
    }
    out
}

#[cfg(test)]
mod piece_change_tests {
    use super::*;
    use glam::DVec3;
    use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind};

    /// A path of junction object 9, `kind`, from `a` to `b`.
    fn path(a: (f64, f64), b: (f64, f64), n: u16, kind: LaneKind) -> omsi_sim::traffic::Lane {
        let mut l = LaneBuilder::polyline(vec![DVec3::new(a.0, a.1, 0.0), DVec3::new(b.0, b.1, 0.0)], kind, 3.0);
        l.source = 2;
        l.key = Some(LaneKey { tile: (0, 0), id: 9, path: n });
        l
    }

    /// Two 150 m pieces of one object 3 m apart running east (0 at y 0, 1 at y -3) and, at x 120,
    /// what crosses both: a street path (a foe) or a footpath (a people's crossing).
    fn object(cross: LaneKind) -> Network {
        let mut net = Network::default();
        net.lanes.push(path((0.0, 0.0), (150.0, 0.0), 0, LaneKind::Street));
        net.lanes.push(path((0.0, -3.0), (150.0, -3.0), 1, LaneKind::Street));
        net.lanes.push(path((120.0, -20.0), (120.0, 20.0), 2, cross));
        net.link(1.5);
        net.compute_conflicts();
        net
    }

    fn car_at(s: f32, speed: f32) -> Vehicle {
        let mut st = AiState::new(0, s, 1);
        st.speed = speed;
        st.front = 2.0;
        st.rear = 2.0;
        st.length = 4.0;
        Vehicle::new(1, VehicleClass::Car, st, 0.9, 7919)
    }

    fn beside(net: &Network) -> NeighbourView {
        let dir = [1, 2].into_iter().find(|&d| net.piece_beside(0, 50.0, d) == Some(1)).expect("piece 1 lies beside piece 0");
        NeighbourView { lane: 1, dir, ahead: None, behind: None, behind_id: None, behind_min_gap: 2.0, runs_on: 150.0, turn: 0, foreign: false, allowed: true }
    }

    /// Where the crossing first touches piece 0 (m), whatever crosses it.
    fn first_meeting(net: &Network, sim: &Simulation) -> f32 {
        let (j, m) = sim.junctions.of_lane[&0];
        let foe = sim.junctions.nodes[j].movements[m].foes.iter().map(|f| f.mine.0).fold(f32::INFINITY, f32::min);
        let walk = net.walks[0].iter().map(|w| w.1).fold(f32::INFINITY, f32::min);
        foe.min(walk)
    }

    /// The window runs past my front by what the car drives while it signals and the move's
    /// length: at 10 m/s (20 m of signal, a 36 m move) a meeting 50 m past the car's middle bars
    /// the change, one 70 m on does not; the move alone (38 m past the middle) would have missed it.
    fn signal_run_counts(cross: LaneKind) {
        let net = object(cross);
        let sim = Simulation::new(&net, vec![]);
        let nb = beside(&net);
        let meet = first_meeting(&net, &sim);
        assert!(meet.is_finite() && meet > 100.0, "the crossing meets piece 0 near x 120 ({meet})");
        let v = 10.0;
        let reach = 2.0 + v * omsi_sim::traffic::SIGNAL_BEFORE_CHANGE + omsi_sim::traffic::change_length(v);
        assert!(reach > 50.0 && 2.0 + omsi_sim::traffic::change_length(v) < 50.0 - 1.0);
        assert!(!piece_change_clear(&net, &sim, &car_at(meet - 50.0, v), &nb), "the meeting within the signal's run bars the change");
        assert!(piece_change_clear(&net, &sim, &car_at(meet - reach - 2.0, v), &nb), "past the move's end it does not");
    }

    /// S2: the window counts the signal's run (a crossing street path, a foe of both pieces).
    #[test]
    fn a_foe_within_the_signal_run_bars_a_piece_change() {
        signal_run_counts(LaneKind::Street);
    }

    /// S3: a people's crossing on the pieces bars the change like a foe (`Network::walks`).
    #[test]
    fn a_people_crossing_within_the_move_bars_a_piece_change() {
        let net = object(LaneKind::Sidewalk);
        assert!(net.walks[0].iter().any(|w| w.0 == 2) && net.walks[1].iter().any(|w| w.0 == 2), "the footpath cuts both pieces");
        let sim = Simulation::new(&net, vec![]);
        let (j, m) = sim.junctions.of_lane[&0];
        assert!(sim.junctions.nodes[j].movements[m].foes.is_empty(), "a footpath is no foe of the pieces");
        signal_run_counts(LaneKind::Sidewalk);
    }

    /// S3: the people's crossing counts with its width (3 m here): a window ending a metre short of
    /// the walk's axis, or a car whose rear is a metre past it, has the walk's band in it.
    #[test]
    fn a_people_crossing_counts_with_its_width() {
        let net = object(LaneKind::Sidewalk);
        let sim = Simulation::new(&net, vec![]);
        let nb = beside(&net);
        let axis = net.walks[0].iter().find(|w| w.0 == 2).map(|w| w.1).unwrap();
        let half = net.lanes[2].width * 0.5;
        let reach = 2.0 + omsi_sim::traffic::change_length(0.0);
        assert!(!piece_change_clear(&net, &sim, &car_at(axis - reach - 1.0, 0.0), &nb), "the band reaches into the window's end");
        assert!(!piece_change_clear(&net, &sim, &car_at(axis + 2.0 + 1.0, 0.0), &nb), "the band reaches under the rear");
        assert!(piece_change_clear(&net, &sim, &car_at(axis - reach - half - 1.0, 0.0), &nb), "a metre short of the band");
        assert!(piece_change_clear(&net, &sim, &car_at(axis + 2.0 + half + 1.0, 0.0), &nb), "a metre past the band");
    }
}
