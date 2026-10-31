//! Lane changes of the drivers' own, round 11 L3: a step-out swept as committed once what it went
//! round has moved off or gone, a plain change from rest projecting the cars beside it and swept
//! again while it signals, and SUMO LC2013's strategic change towards a random car's trip end.
use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network};
use omsi_traffic::{Simulation, Vehicle, VehicleClass};

const DT: f32 = 1.0 / 30.0;

fn road(pts: &[(f64, f64)], id: i64, path: u16, offset: f32) -> omsi_sim::traffic::Lane {
    let mut l = LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), LaneKind::Street, 3.0);
    l.source = 1;
    l.key = Some(LaneKey { tile: (0, 0), id, path });
    l.offset = offset;
    l
}

fn car(id: u64, lane: usize, s: f32, speed: f32) -> Vehicle {
    let mut st = AiState::new(lane, s, id);
    st.speed = speed;
    st.front = 2.0;
    st.rear = 2.0;
    st.length = 4.0;
    st.max_speed_kmh = 100.0;
    st.veh_type = 0;
    let mut v = Vehicle::new(id, VehicleClass::Car, st, 0.9, id * 7919);
    v.fresh = 0.0;
    v
}

/// Two eastbound lanes of one road to x 600: 0 the left at y 0, 1 the right at y -3.5.
fn two_lanes() -> Network {
    let mut net = Network::default();
    net.lanes.push(road(&[(0.0, 0.0), (600.0, 0.0)], 1, 0, 0.0));
    net.lanes.push(road(&[(0.0, -3.5), (600.0, -3.5)], 1, 1, 3.5));
    net.link(1.5);
    net.compute_conflicts();
    net.compute_reach();
    net
}

/// A car standing for good (its own plan never moves it off).
fn standing(id: u64, lane: usize, s: f32) -> Vehicle {
    let mut o = car(id, lane, s, 0.0);
    o.pull_out = 1000.0;
    o.state.change_cooldown = 1000.0;
    o
}

/// Car 3 standing 2.2 m right of the right lane's axis at x `s`: beside car 1's rear, where a
/// full-lock step-out to the left swings it.
fn kerb_car(s: f32) -> Vehicle {
    let lat = 2.2;
    let mut c3 = standing(3, 1, s);
    c3.state.lateral = lat;
    c3.state.lateral_target = lat;
    c3.state.lateral_ramp = (lat, lat, 0.0, 1.0);
    c3
}

/// What a signalling step-out does once the car it went round is no longer standing.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Obstacle {
    /// Taken off the road the frame after the step-out started.
    Gone,
    /// Driven off at 2 m/s the frame after the step-out started.
    MovesOff,
}

/// Car 1 at rest on the right lane 2.5 m behind car 2 steps out at full lock (as round11l's
/// `from_rest`); the frame after it started, car 2 goes (`what`), and with `intruder` car 3 is put
/// beside car 1's rear on the kerb side: (the change started - t, full lock -, whether it was
/// dropped with car 1 still on its lane, the least gap of car 1's body to car 3).
fn step_out_after_its_obstacle_left(what: Obstacle, intruder: bool) -> (Option<(f32, bool)>, bool, f32) {
    let net = two_lanes();
    let mut sim = Simulation::new(&net, vec![]);
    let mut me = car(1, 1, 100.0, 0.0);
    me.state.change_cooldown = 0.0;
    let mut cars = vec![me, standing(2, 1, 106.5)];
    let (mut started, mut dropped, mut least, mut done) = (None, false, f32::INFINITY, false);
    let mut t = 0.0f32;
    while t < 12.0 {
        sim.tick(&net, DT, &mut cars);
        let me = cars.iter().find(|c| c.id == 1).unwrap().clone();
        if started.is_none() {
            started = me.state.change.map(|c| (t, c.full_lock));
        } else if !done {
            done = true;
            match what {
                Obstacle::Gone => cars.retain(|c| c.id != 2),
                Obstacle::MovesOff => {
                    let c2 = cars.iter_mut().find(|c| c.id == 2).unwrap();
                    c2.pull_out = 0.0;
                    c2.state.speed = 2.0;
                }
            }
            if intruder {
                cars.push(kerb_car(99.5));
            }
        }
        if started.is_some() && me.state.change.is_none() && me.state.lane == 1 {
            dropped = true;
        }
        if let Some(o) = cars.iter().find(|o| o.id == 3) {
            least = least.min(me.obb().separation(&o.obb()) as f32);
        }
        t += DT;
    }
    (started, dropped, least)
}

/// L3-1: a step-out whose obstacle has gone, or moved off, still drives the full-lock move it
/// committed to, so that move is swept on every frame of its signal (SUMO checks the manoeuvre
/// each step): a car put beside its rear where the move swings it drops the change before the car
/// moves over, and nothing is touched; with nobody put there the change is not dropped.
#[test]
fn a_step_out_whose_obstacle_has_left_still_sweeps_its_committed_move() {
    for what in [Obstacle::Gone, Obstacle::MovesOff] {
        let (started, dropped, _) = step_out_after_its_obstacle_left(what, false);
        assert!(started.is_some_and(|s| s.1) && !dropped, "{what:?}, nobody beside: started {started:?}, dropped {dropped}");
        let (started, dropped, least) = step_out_after_its_obstacle_left(what, true);
        assert!(started.is_some_and(|s| s.1) && dropped && least > 0.3, "{what:?}, car beside the rear: started {started:?}, dropped {dropped}, least {least:.2}");
    }
}

/// `two_lanes` with a third lane, 2, on the right of lane 1 (y -7).
fn three_lanes() -> Network {
    let mut net = Network::default();
    net.lanes.push(road(&[(0.0, 0.0), (600.0, 0.0)], 1, 0, 0.0));
    net.lanes.push(road(&[(0.0, -3.5), (600.0, -3.5)], 1, 1, 3.5));
    net.lanes.push(road(&[(0.0, -7.0), (600.0, -7.0)], 1, 2, 7.0));
    net.link(1.5);
    net.compute_conflicts();
    net.compute_reach();
    net
}

/// A long body (front 3.5 m, rear 8.5 m) at 4 m/s on lane 1 signals a plain change left onto
/// lane 0 at t 0; `after` s into the signal a car rolling at the same speed is put level with it
/// on lane 2, 0.4 m beside its body (or 2.5 m), where its rear swings as it turns left:
/// (dropped with the body still on lane 1, the least gap between the two bodies).
fn plain_change_with_a_car_coming_beside(after: f32, gap: f32) -> (bool, f32) {
    let net = three_lanes();
    assert_eq!(net.lanes[1].left, Some(0));
    let mut sim = Simulation::new(&net, vec![]);
    let mut me = car(1, 1, 100.0, 4.0);
    me.state.front = 3.5;
    me.state.rear = 8.5;
    me.state.length = 12.0;
    me.state.max_speed_kmh = 14.4;
    me.state.start_change(&net, 0, 1);
    let mut far = car(2, 2, 100.0 + 0.75, 4.0);
    far.state.max_speed_kmh = 14.4;
    far.state.change_cooldown = 1000.0;
    // (lane 2's axis is 3.5 m right of lane 1's; the far car's body 0.9 m wide either side)
    let lat = -(3.5 - 0.9 - 0.9 - gap);
    far.state.lateral = lat;
    far.state.lateral_target = lat;
    far.state.lateral_ramp = (lat, lat, 0.0, 1.0);
    let mut cars = vec![me];
    let (mut dropped, mut least, mut placed) = (false, f32::INFINITY, false);
    let mut t = 0.0f32;
    while t < 8.0 {
        if !placed && t >= after {
            let s = cars[0].state.s;
            far.state.s = s + 0.75;
            cars.push(far.clone());
            placed = true;
        }
        sim.tick(&net, DT, &mut cars);
        let me = cars.iter().find(|c| c.id == 1).unwrap().clone();
        if me.state.change.is_none() && me.state.lane == 1 {
            dropped = true;
        }
        if let Some(o) = cars.iter().find(|o| o.id == 2) {
            least = least.min(me.obb().separation(&o.obb()) as f32);
        }
        t += DT;
    }
    (dropped, least)
}

/// L3-2: a plain change sweeps its move again on every frame of its signal (as `change_clear`
/// swept it at the start; SUMO checks the manoeuvre each step): a car that comes level with the
/// body on the far side, 0.4 m beside it, where its rear swings out as it turns (without the sweep
/// it went on and swung to 0.30 m, down to `SQUEEZE_CLEARANCE`), drops the change before the body
/// moves over; 2.5 m beside it does not.
#[test]
fn a_plain_change_is_swept_again_while_it_signals() {
    let (dropped, least) = plain_change_with_a_car_coming_beside(0.5, 0.4);
    assert!(dropped && least > 0.3, "a car 0.4 m beside: dropped {dropped}, least {least:.2}");
    let (dropped, least) = plain_change_with_a_car_coming_beside(0.5, 2.5);
    assert!(!dropped && least > 0.3, "a car 2.5 m beside: dropped {dropped}, least {least:.2}");
}

/// Fix 1-2: a car signalling a plain change whose leader comes to rest ahead keeps behind it the
/// room the body needs to turn out of the lane from rest (`change_keep_room`: the bypass's
/// full-lock room) and the change goes on as the step-out from there, not dropped: car 1 at 6 m/s
/// on the right lane of `two_lanes` signals a change left behind car 2 rolling at 6 m/s 15 m ahead;
/// half a second later car 2 stands. Car 1 stops behind it with more than a standing gap, moves
/// over without the change dropped, and its body never comes within `SQUEEZE_CLEARANCE` (0.3 m) of
/// car 2's.
#[test]
fn a_car_signalling_a_change_keeps_the_room_to_turn_out_behind_its_leader() {
    let net = two_lanes();
    let mut sim = Simulation::new(&net, vec![]);
    let mut me = car(1, 1, 100.0, 6.0);
    me.state.change_cooldown = 0.0;
    me.params.lc_speed_gain = 0.0;
    me.params.lc_keep_right = 0.0;
    me.state.start_change(&net, 0, 1);
    let mut ahead = car(2, 1, 119.0, 6.0);
    ahead.state.change_cooldown = 1000.0;
    ahead.params.lc_speed_gain = 0.0;
    ahead.params.lc_keep_right = 0.0;
    let mut cars = vec![me, ahead];
    let (mut dropped, mut least, mut gap_least) = (false, f32::INFINITY, f32::INFINITY);
    let (mut t, mut stopped) = (0.0f32, false);
    while t < 15.0 {
        if !stopped && t >= 0.5 {
            let k = cars.iter().position(|c| c.id == 2).unwrap();
            cars[k] = standing(2, 1, cars[k].state.s);
            stopped = true;
        }
        sim.tick(&net, DT, &mut cars);
        let me = cars.iter().find(|c| c.id == 1).unwrap().clone();
        let o = cars.iter().find(|o| o.id == 2).unwrap();
        if me.state.change.is_none() && me.state.lane == 1 {
            dropped = true;
        }
        if me.state.lane == 1 && me.state.change.is_none_or(|c| c.t <= 0.0) {
            gap_least = gap_least.min((o.state.s - o.state.rear) - (me.state.s + me.state.front));
        }
        least = least.min(me.obb().separation(&o.obb()) as f32);
        t += DT;
    }
    let lane = cars.iter().find(|c| c.id == 1).unwrap().state.lane;
    assert!(!dropped && lane == 0, "the change goes through: dropped {dropped}, lane {lane}, least gap before the move {gap_least:.2}");
    assert!(gap_least > 2.5 && least > 0.3, "room kept behind the leader: least gap before the move {gap_least:.2}, least {least:.2}");
}

/// A two-lane eastbound approach (lane 0 the left at y 0, lane 1 the right at y -3.5, x 0..300)
/// to a fork at x 300: lane 0 runs on straight (lane 2, to x 600) or turns left (lane 5); lane 1
/// turns right (lane 3, the trip end, ending at (400, -100)) or runs on straight (lane 4). Lane 2
/// forks again at x 600 (lanes 6, 7), lane 5 at (400, 100) (lanes 8, 9); those and 4 end.
fn fork_ahead() -> Network {
    let mut net = Network::default();
    net.lanes.push(road(&[(0.0, 0.0), (300.0, 0.0)], 1, 0, 0.0));
    net.lanes.push(road(&[(0.0, -3.5), (300.0, -3.5)], 1, 1, 3.5));
    net.lanes.push(road(&[(300.0, 0.0), (600.0, 0.0)], 2, 0, 0.0));
    net.lanes.push(road(&[(300.0, -3.5), (330.0, -10.0), (400.0, -100.0)], 3, 0, 0.0));
    net.lanes.push(road(&[(300.0, -3.5), (600.0, -3.5)], 4, 0, 0.0));
    net.lanes.push(road(&[(300.0, 0.0), (330.0, 7.0), (400.0, 100.0)], 5, 0, 0.0));
    net.lanes.push(road(&[(600.0, 0.0), (900.0, 0.0)], 6, 0, 0.0));
    net.lanes.push(road(&[(600.0, 0.0), (630.0, 7.0), (700.0, 100.0)], 7, 0, 0.0));
    net.lanes.push(road(&[(400.0, 100.0), (400.0, 400.0)], 8, 0, 0.0));
    net.lanes.push(road(&[(400.0, 100.0), (380.0, 130.0), (300.0, 200.0)], 9, 0, 0.0));
    net.link(1.5);
    net.compute_conflicts();
    net.compute_reach();
    net
}

/// The counters `traffic_cases ab` prints for the trip ends and lane use (whole process).
fn counters() -> (u64, u64, u64) {
    let g = |a: &std::sync::atomic::AtomicU64| a.load(std::sync::atomic::Ordering::Relaxed);
    (g(&omsi_traffic::sim::CHANGES_BY_WHY[omsi_traffic::plan::ChangeWhy::Strategic as usize]), g(&omsi_sim::traffic::STRATEGIC_MISSED), g(&omsi_sim::traffic::GOAL_UNREACHABLE))
}

/// Car 1 on the left lane at x `start`, 13 m/s, its trip end lane 3 (off the right-hand way), with
/// no wish of its own to change for speed or to keep right (`lc_speed_gain`, `lc_keep_right` 0) and
/// `lc_strategic` as given; with `platoon` a row of cars on the right lane from 60 m behind it, at
/// its speed, too close one behind the other for any gap. Runs 40 s: (the lanes it drove, in order,
/// the strategic changes, missed changes and goal redraws for unreachability counted meanwhile,
/// its trip end at the end, the longest it stood, s).
fn to_the_trip_end_off_the_right(lc_strategic: f32, platoon: bool, start: f32) -> (Vec<usize>, (u64, u64, u64), Option<usize>, f32) {
    let net = fork_ahead();
    assert_eq!(net.lanes[0].right, Some(1));
    assert_eq!(net.lanes[0].next.len(), 2);
    let mut sim = Simulation::new(&net, vec![]);
    let mut me = car(1, 0, start, 13.0);
    me.state.change_cooldown = 0.0;
    me.state.goal = Some(3);
    me.params.lc_speed_gain = 0.0;
    me.params.lc_keep_right = 0.0;
    me.params.lc_strategic = lc_strategic;
    let mut cars = vec![me];
    if platoon {
        for (k, s) in (0..14).map(|k| (k, start - 60.0 + 9.0 * k as f32)).filter(|&(_, s)| s >= 0.0) {
            let mut o = car(10 + k, 1, s, 13.0);
            o.state.change_cooldown = 1000.0;
            o.params.lc_speed_gain = 0.0;
            o.params.lc_keep_right = 0.0;
            o.state.max_speed_kmh = 13.0 * 3.6;
            o.state.goal = Some(4);
            cars.push(o);
        }
        cars[0].state.max_speed_kmh = 13.0 * 3.6;
    }
    let before = counters();
    let mut lanes = vec![0usize];
    let (mut t, mut stood, mut longest) = (0.0f32, 0.0f32, 0.0f32);
    while t < 60.0 {
        sim.tick(&net, DT, &mut cars);
        let Some(me) = cars.iter().find(|c| c.id == 1) else { break };
        if lanes.last() != Some(&me.state.lane) {
            lanes.push(me.state.lane);
        }
        stood = if me.state.speed < 0.1 { stood + DT } else { 0.0 };
        longest = longest.max(stood);
        t += DT;
    }
    let after = counters();
    let goal = cars.iter().find(|c| c.id == 1).and_then(|c| c.state.goal);
    (lanes, (after.0 - before.0, after.1 - before.1, after.2 - before.2), goal, longest)
}

/// L3-3, SUMO LC2013's strategic change: a car on the left lane whose trip end lies off the
/// right-hand way at the fork changes right before the fork (counted strategic) and takes the way
/// to its trip end; with `lc_strategic` -1 (SUMO: no strategic changes) it stays left and drives
/// through the fork on its own lane's way. With a row on the right lane level with it, too close for
/// any gap, the row is asked to make room (LC2013's `informFollower`) and it changes right in time
/// and takes the turn, no miss. Only past the point of no return - the row level with it 25 m
/// before the fork, where it can no longer stop at its comfortable brake - does it drive on
/// through the fork (counted `STRATEGIC_MISSED`) and its trip end is drawn again
/// (`GOAL_UNREACHABLE` at its next fork).
#[test]
fn a_car_changes_lanes_towards_its_trip_end_before_the_fork_or_misses_it_and_draws_again() {
    let (lanes, (strategic, _, _), goal, _) = to_the_trip_end_off_the_right(1.0, false, 40.0);
    assert!(lanes.starts_with(&[0, 1, 3]) && strategic >= 1, "changes right and takes the turn: lanes {lanes:?}, strategic {strategic}, goal {goal:?}");
    let (lanes, _, _, _) = to_the_trip_end_off_the_right(-1.0, false, 40.0);
    assert!(lanes.len() >= 2 && lanes[1] != 1 && lanes[1] != 3, "no strategic change: lanes {lanes:?}");
    let (lanes, (strategic, _, _), goal, _) = to_the_trip_end_off_the_right(1.0, true, 40.0);
    assert!(lanes.starts_with(&[0, 1, 3]) && strategic >= 1, "a row beside: it makes room, the car changes: lanes {lanes:?}, strategic {strategic}, goal {goal:?}");
    let (lanes, (_, missed, unreachable), goal, _) = to_the_trip_end_off_the_right(1.0, true, 275.0);
    assert!(lanes.len() >= 3 && (lanes[1] == 2 || lanes[1] == 5), "past the point of no return: drives on through the fork on its lane: lanes {lanes:?}");
    assert!(missed >= 1, "the miss is counted: {missed}");
    assert!(unreachable >= 1 && goal != Some(3), "drawn again at the next fork: unreachable {unreachable}, goal {goal:?}");
}

/// A blocked urgent strategic change brings the car to a stop before the end of its dead lane
/// (SUMO LC2013: the vehicle adapts its speed to stop at the lane's end while the change is
/// blocked, sumo.md 3.2), where it waits for a gap: car 1 on the left lane of `fork_ahead` at x 200,
/// 13 m/s, its trip end lane 3 off the right-hand way, beside a jam standing on the right lane
/// from x 220 to the fork, stands before x 300 and is still on its lane with its trip end after
/// 30 s (no miss); the jam gone, it changes right and takes the turn.
#[test]
fn a_blocked_strategic_change_stops_before_the_dead_lanes_end_and_waits() {
    let net = fork_ahead();
    let mut sim = Simulation::new(&net, vec![]);
    let mut me = car(1, 0, 200.0, 13.0);
    me.state.change_cooldown = 0.0;
    me.state.goal = Some(3);
    me.params.lc_speed_gain = 0.0;
    me.params.lc_keep_right = 0.0;
    let mut cars = vec![me];
    for k in 0..12u64 {
        cars.push(standing(10 + k, 1, 220.0 + 6.5 * k as f32));
    }
    let mut lanes = vec![0usize];
    let mut t = 0.0f32;
    while t < 50.0 {
        if t >= 30.0 {
            cars.retain(|c| c.id == 1);
        }
        sim.tick(&net, DT, &mut cars);
        let me = cars.iter().find(|c| c.id == 1).expect("car 1");
        if lanes.last() != Some(&me.state.lane) {
            lanes.push(me.state.lane);
        }
        if (t - 29.9).abs() < DT * 0.5 {
            assert!(me.state.lane == 0 && me.state.speed < 0.1 && me.state.s + me.state.front < 300.0 && me.state.goal == Some(3), "blocked: stands before the dead lane's end: lane {} s {:.1} v {:.1} goal {:?}", me.state.lane, me.state.s, me.state.speed, me.state.goal);
        }
        t += DT;
    }
    assert!(lanes.starts_with(&[0, 1, 3]), "the jam gone, it changes and takes the turn: lanes {lanes:?}");
}

/// Car 1 standing on the left lane of `fork_ahead` at x `at`, its trip end lane 3 off the right-hand
/// way; a row of 14 cars on the right lane rolls past at 3 m/s from x 190, each at its standing gap
/// behind the one ahead (6.5 m apart), none with a wish to change. Runs 40 s: (the time car 1 moved
/// over onto lane 1, the least gap between car 1's body and any body of the row while it did).
fn into_a_slow_row(at: f32) -> (Option<f32>, f32) {
    let net = fork_ahead();
    let mut sim = Simulation::new(&net, vec![]);
    let mut me = car(1, 0, at, 0.0);
    me.state.change_cooldown = 0.0;
    me.state.goal = Some(3);
    me.params.lc_speed_gain = 0.0;
    me.params.lc_keep_right = 0.0;
    let mut cars = vec![me];
    for k in 0..14u64 {
        let mut o = car(10 + k, 1, 190.0 + 6.5 * k as f32, 3.0);
        o.state.change_cooldown = 1000.0;
        o.params.lc_speed_gain = 0.0;
        o.params.lc_keep_right = 0.0;
        o.state.max_speed_kmh = 3.0 * 3.6;
        o.state.goal = Some(4);
        cars.push(o);
    }
    let (mut t, mut moved, mut least) = (0.0f32, None, f32::INFINITY);
    while t < 40.0 {
        sim.tick(&net, DT, &mut cars);
        sim.update_positions(&net, &mut cars);
        let me = cars.iter().find(|c| c.id == 1).expect("car 1");
        if moved.is_none() && me.state.lane != 0 {
            moved = Some(t);
        }
        let b = me.obb();
        for o in cars.iter().filter(|o| o.id != 1) {
            least = least.min(o.obb().separation(&b) as f32);
        }
        t += DT;
    }
    (moved, least)
}

/// Fix 2-2, SUMO LC2013's speed advice for a blocked urgent change (`_wantsChange`: `informLeader`,
/// then `informFollower` with the speed planned): a car waiting before its dead lane's end beside a
/// slow row informs the follower there also while the leader there is still too close ahead, and
/// that follower holds back (`LCA_AMBLOCKINGFOLLOWER`, `LC_COOPERATIVE` 1.0): car 1 at rest at x 270
/// beside a row rolling past at 3 m/s, each car at its standing gap, moves over within 20 s (14.6 s;
/// asking only a follower behind a leader there that was clear, it had not moved over after 40 s),
/// and no body comes within 0.3 m of another.
#[test]
fn a_car_waiting_for_its_strategic_change_beside_a_slow_row_is_let_in() {
    let (moved, least) = into_a_slow_row(270.0);
    let t = moved.expect("car 1 never moved over");
    assert!(t < 20.0, "car 1 moved over only at {t:.1} s");
    assert!(least >= 0.3, "bodies {least:.2} m apart");
}

/// The trip end of a car with none is drawn at the first fork within SUMO's strategic look-ahead
/// (`LC_STRATEGIC_LOOKAHEAD`), not only once its plan (`PLAN_AHEAD`) reaches the fork: car 1 260 m
/// before the fork of `fork_ahead` has one after its first frame, and over 40 cars some draw lane 3,
/// the end reached only from the lane beside.
#[test]
fn a_trip_end_is_drawn_where_the_strategic_look_ahead_sees_the_fork() {
    let net = fork_ahead();
    let mut beside = 0;
    for id in 1..=40u64 {
        let mut sim = Simulation::new(&net, vec![]);
        let mut cars = vec![car(id, 0, 40.0, 13.0)];
        sim.tick(&net, DT, &mut cars);
        let goal = cars[0].state.goal;
        assert!(goal.is_some(), "car {id}: no trip end after its first frame, 260 m before the fork");
        beside += (goal == Some(3)) as usize;
    }
    assert!(beside > 0, "lane 3, off the lane beside, never drawn");
}

/// A lane beside only on a short last piece before the fork does not count for the draw: the
/// left lane (lane 0, x 0..285) runs into lane 1 (285..300) with lane 2 beside it only there
/// (y -3.5), lane 2 leading to lane 3 (an end); lane 1 forks into lanes 4 and 5 (ends). A car at
/// 13 m/s on lane 0 runs beside lane 2 for 15 m, less than a change takes (`change_space`): over 40
/// cars none draws lane 3.
#[test]
fn a_lane_beside_only_on_a_short_last_piece_is_no_way_to_the_trip_end() {
    let mut net = Network::default();
    net.lanes.push(road(&[(0.0, 0.0), (285.0, 0.0)], 1, 0, 0.0));
    net.lanes.push(road(&[(285.0, 0.0), (300.0, 0.0)], 2, 0, 0.0));
    net.lanes.push(road(&[(285.0, -3.5), (300.0, -3.5)], 2, 1, 3.5));
    net.lanes.push(road(&[(300.0, -3.5), (330.0, -10.0), (400.0, -100.0)], 3, 0, 0.0));
    net.lanes.push(road(&[(300.0, 0.0), (600.0, 0.0)], 4, 0, 0.0));
    net.lanes.push(road(&[(300.0, 0.0), (330.0, 7.0), (400.0, 100.0)], 5, 0, 0.0));
    net.link(1.5);
    net.compute_conflicts();
    net.compute_reach();
    assert_eq!(net.lanes[1].right, Some(2));
    assert_eq!(net.lanes[1].next.len(), 2);
    assert!(omsi_sim::traffic::change_space(13.0, 1.0, omsi_sim::traffic::SIGNAL_BEFORE_CHANGE) > 15.0);
    for id in 1..=40u64 {
        let mut sim = Simulation::new(&net, vec![]);
        let mut cars = vec![car(id, 0, 40.0, 13.0)];
        sim.tick(&net, DT, &mut cars);
        let goal = cars[0].state.goal;
        assert!(goal.is_some_and(|g| g == 4 || g == 5), "car {id}: trip end {goal:?}, not one its own lane's ways reach");
    }
}

/// Car 1 at rest on the right lane at s 100 behind car 2 standing at 106.5 (close enough that only
/// the full-lock move steers), car 5 standing at 112.5 behind car 2 with `queue` (the obstacle run
/// spans both), car 4 standing on the left lane at 148.5; each frame `each(t, car 1, people)` may
/// put people down. Runs 12 s: (the changes started - t, full lock, round -, the drops with car 1
/// still on its lane - t -, the least gap of car 1's body to any other body or person, whether it
/// ended on the left lane, the largest heading difference while signalling between car 1's pose
/// and the pose on its way without the change, `sim::way_pose`).
#[allow(clippy::type_complexity)]
fn step_out_round_a_queue(queue: bool, mut each: impl FnMut(f32, &Vehicle, &mut Vec<omsi_traffic::Person>)) -> (Vec<(f32, bool, Option<u64>)>, Vec<f32>, f32, bool, f32) {
    let net = two_lanes();
    let mut sim = Simulation::new(&net, vec![]);
    let mut me = car(1, 1, 100.0, 0.0);
    me.state.change_cooldown = 0.0;
    let mut cars = vec![me, standing(2, 1, 106.5), standing(4, 0, 148.5)];
    if queue {
        cars.push(standing(5, 1, 112.5));
    }
    let mut people: Vec<omsi_traffic::Person> = Vec::new();
    let (mut starts, mut drops, mut least, mut turned) = (Vec::new(), Vec::new(), f32::INFINITY, 0.0f32);
    let mut was = false;
    let mut t = 0.0f32;
    while t < 12.0 {
        sim.people = people.clone();
        sim.tick(&net, DT, &mut cars);
        let me = cars.iter().find(|c| c.id == 1).unwrap().clone();
        let now = me.state.change;
        if let (Some(c), false) = (now, was) {
            starts.push((t, c.full_lock, c.round));
        }
        if was && now.is_none() && me.state.lane == 1 {
            drops.push(t);
        }
        if let Some(c) = now.filter(|c| c.wait > 0.0 && c.t <= 0.0) {
            let _ = c;
            let mut before = me.state.clone();
            before.drop_change();
            let (_, h) = omsi_traffic::sim::way_pose(&net, &before);
            turned = turned.max(((me.heading - h + 540.0) % 360.0 - 180.0).abs());
        }
        was = now.is_some();
        for o in cars.iter().filter(|o| o.id != 1) {
            least = least.min(me.obb().separation(&o.obb()) as f32);
        }
        for q in &people {
            least = least.min((omsi_sim::collision::Obb::point(q.pos, omsi_traffic::plan::PERSON_RADIUS as f64).separation(&me.obb())) as f32);
        }
        each(t, &me, &mut people);
        t += DT;
    }
    (starts, drops, least, cars.iter().find(|c| c.id == 1).unwrap().state.lane == 0, turned)
}

/// The final recheck's test A: a queue of two standing in front (cars 2 and 5) is stepped out
/// round at full lock, the move swept on the change-less corridor with the committed kind, round
/// car 2: one start, never dropped, ends on the left lane, nothing within `SQUEEZE_CLEARANCE`.
#[test]
fn a_step_out_round_a_queue_of_two_starts_once_and_is_not_dropped() {
    let (starts, drops, least, left, _) = step_out_round_a_queue(true, |_, _, _| {});
    assert!(starts.len() == 1 && starts[0].1 && starts[0].2 == Some(2), "one full-lock start round car 2: {starts:?}");
    assert!(drops.is_empty() && left && least > 0.3, "drops {drops:?}, on the left lane {left}, least {least:.2}");
}

/// The final recheck's test B: (a) late in the signal the body's heading turns into the move laid
/// ahead (more than a degree off the pose on its way without the change), and the step-out swept
/// from the change-less pose is not dropped for it; (b) a person put down at 0.9 of the signal
/// 0.15 m outside the un-turned body's front corner on the target side (left) - inside the turned
/// body's corner there, the lagged chord of the move laid ahead - drops the step-out before the
/// car moves over, and the car's body never comes nearer them than it was when they were put
/// down (the turn eases back as the change is dropped).
#[test]
fn a_step_out_late_in_its_signal_is_swept_from_its_change_less_pose() {
    let (starts, drops, least, left, turned) = step_out_round_a_queue(false, |_, _, _| {});
    assert!(starts.len() == 1 && starts[0].1, "one full-lock start: {starts:?}");
    assert!(turned > 1.0 && drops.is_empty() && left && least > 0.3, "turned {turned:.2} deg, drops {drops:?}, on the left lane {left}, least {least:.2}");
    let net = two_lanes();
    let signal = omsi_sim::traffic::BYPASS_SIGNAL;
    let (mut start, mut placed) = (None::<f32>, false);
    let mut seps: Vec<f32> = Vec::new();
    let (starts, drops, _, _, _) = step_out_round_a_queue(false, |t, me, people| {
        if let Some(q) = people.first() {
            seps.push(omsi_sim::collision::Obb::point(q.pos, omsi_traffic::plan::PERSON_RADIUS as f64).separation(&me.obb()) as f32);
        }
        if start.is_none() && me.state.change.is_some() {
            start = Some(t);
        }
        if let (Some(t0), false) = (start, placed) {
            if t - t0 >= 0.9 * signal && me.state.change.is_some_and(|c| c.t <= 0.0) {
                let mut before = me.state.clone();
                before.drop_change();
                let (pos, h) = omsi_traffic::sim::way_pose(&net, &before);
                let body = omsi_sim::collision::Obb::vehicle(pos.truncate(), h as f64, me.state.front as f64, me.state.rear as f64, me.half_width as f64);
                // (the un-turned body's front corner on the left, and 0.15 m further left)
                let hr = (h as f64).to_radians();
                let (fwd, left) = (glam::DVec2::new(hr.sin(), hr.cos()), glam::DVec2::new(-hr.cos(), hr.sin()));
                let corner = body.corners().into_iter().max_by(|a, b| ((*a - pos.truncate()).dot(fwd) + (*a - pos.truncate()).dot(left)).total_cmp(&((*b - pos.truncate()).dot(fwd) + (*b - pos.truncate()).dot(left)))).unwrap();
                let q = corner + left * 0.15;
                people.push(omsi_traffic::Person { pos: glam::DVec3::new(q.x, q.y, 0.0), vel: glam::DVec2::ZERO, on_crossing: false, still: 10.0, waiting_for_bus: false, at_kerb: false, waits_at: None, id: None, approach: None });
                placed = true;
            }
        }
    });
    assert!(placed && starts.first().is_some_and(|s| s.1), "placed {placed}, starts {starts:?}");
    let t0 = starts[0].0;
    let first = seps.first().copied().unwrap_or(f32::NAN);
    let least = seps.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(drops.first().is_some_and(|&d| d <= t0 + signal + 0.05) && least >= first - 1e-3, "dropped {drops:?} (signal ends {:.2}), gap to the person when put down {first:.2}, least {least:.2}", t0 + signal);
}

/// A lane of crossing object 9 (source 2), 1 m wide as the drive's 12690.
fn piece(pts: &[(f64, f64)], path: u16) -> omsi_sim::traffic::Lane {
    let mut l = LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), LaneKind::Street, 1.0);
    l.source = 2;
    l.key = Some(LaneKey { tile: (0, 0), id: 9, path });
    l.priority = 128.0;
    l
}

/// Fix 1-3, the drive's c0914 (09:14) geometry: car 1 (4.6 m, 1.84 m wide) stands on a 1 m
/// crossing piece (lane 0, eastbound at y 0, x 100..200) behind car 2, which stands 10 degrees
/// askew on a lane of its own (lane 5, heading 100 as #33376 beside #33433's 89), its rear-left
/// corner 0.87 m right of car 1's axis: a squeeze to the left keeping `MIN_GAP_LAT` needs 1.57 m
/// left of the axis. The object's piece beside on the left (lane 1, y 3.05) is closed to cars, an
/// open one lies 3.2 m right; halfway to the piece beside gives 1.52 m. The object's paths are
/// drawn on its road, and the closed piece, empty over the squeeze, lends its width (the body
/// keeps off its axis): from 7.7 m behind car 2's rear the squeeze goes at the least offset
/// (0.65 m) and never comes within `SQUEEZE_CLEARANCE` of a body. With a car on the closed piece
/// beside car 1 it lends nothing and car 1 does not squeeze. From 1.7 m behind (as in the drive)
/// the least offset's shortest move meets car 2's corner before the body is out, and the offset
/// is the least one only (the user's decision L3-D1: SUMO's minGapLat is the only bound): car 1
/// does not squeeze, it waits, clear of car 2.
#[test]
fn on_a_junction_piece_a_closed_empty_piece_beside_lends_its_width_to_a_squeeze() {
    let mut net = Network::default();
    net.lanes.push(piece(&[(100.0, 0.0), (200.0, 0.0)], 0)); // 0
    let mut closed = piece(&[(100.0, 3.05), (200.0, 3.05)], 1);
    closed.no_cars = true;
    net.lanes.push(closed); // 1
    net.lanes.push(piece(&[(100.0, -3.2), (200.0, -3.2)], 2)); // 2
    net.lanes.push(road(&[(0.0, 0.0), (100.0, 0.0)], 1, 0, 0.0)); // 3
    net.lanes.push(road(&[(200.0, 0.0), (600.0, 0.0)], 2, 0, 0.0)); // 4
    net.link(1.5);
    net.compute_conflicts();
    net.compute_reach();
    // (lane 5: car 2's way, heading 100 - 10 degrees askew of lane 0 - through (152, -2.1), as
    // #33376 stood on 12587 at heading 100 beside #33433's 89: its rear-left corner 0.87 m right
    // of car 1's axis)
    let slope = (10.0f64).to_radians().tan();
    net.lanes.push(road(&[(140.0, -2.1 + 12.0 * slope), (170.0, -2.1 - 18.0 * slope)], 3, 0, 0.0)); // 5
    net.build_grid();
    assert_eq!((net.piece_beside(0, 50.0, 1), net.piece_beside(0, 50.0, 2)), (Some(1), Some(2)));
    for (s0, occupied) in [(40.0f32, false), (40.0, true), (46.0, false)] {
        let mut sim = Simulation::new(&net, vec![]);
        let mut me = car(1, 0, s0, 0.0);
        me.half_width = 0.92;
        me.state.front = 2.4;
        me.state.rear = 2.2;
        me.state.length = 4.6;
        me.stopped = 80.0;
        me.state.max_speed_kmh = 30.0;
        let mut ahead = standing(2, 5, 12.0 / (10.0f32).to_radians().cos());
        ahead.stopped = 90.0;
        let mut cars = vec![me, ahead];
        if occupied {
            cars.push(standing(3, 1, 46.0));
        }
        let (mut pass, mut least) = (None, f32::INFINITY);
        let mut t = 0.0f32;
        while t < 15.0 {
            sim.tick(&net, DT, &mut cars);
            let me = cars.iter().find(|c| c.id == 1).unwrap();
            if pass.is_none() {
                pass = me.passing;
            }
            for o in cars.iter().filter(|o| o.id != 1) {
                least = least.min(me.obb().separation(&o.obb()) as f32);
            }
            t += DT;
        }
        let x = cars.iter().find(|c| c.id == 1).unwrap().pos.x;
        if occupied {
            assert!(pass.is_none() && x < 150.0, "the piece beside taken: no squeeze: {pass:?}, x {x:.1}");
        } else if s0 > 45.0 {
            assert!(pass.is_none() && x < 150.0 && least > 0.3, "from 1.7 m behind the least offset does not clear: no squeeze, it waits: {pass:?}, x {x:.1}, least {least:.2}");
        } else {
            let side = pass.map(|p| p.side);
            assert!(pass.is_some_and(|p| p.lane.is_none() && (p.side + 0.65).abs() < 0.02), "squeezes to the left inside the junction at the least offset: {pass:?}");
            assert!(least > 0.3 && x > 160.0, "clear past car 2: least {least:.2}, x {x:.1}, side {side:?}");
        }
    }
}

/// Fix 3, item 1: the dead lane's run beside the lane that leads (`dead_run`) is walked along the
/// lanes beside, so a piece of the lane beside that is shorter than the dead lane's own piece - the
/// inner lane of a bend - does not end it. A two-lane road (left lane on path 0, right on path 1
/// of each spline): straight x 0..100, a bend to the right through 60 degrees (radius 200 on the
/// left lane, 196.5 on the right: the right one 3.7 m shorter), straight again for 300 m to a fork
/// where the left lane turns off left or right (lanes 6, 7) and the right lane runs on to the trip
/// end (lane 5). A car on the left lane at x 10, 13 m/s, its trip end lane 5, with no wish to
/// change for speed or to keep right: its urgent strategic change begins only where the distance
/// left to the fork comes under its strategic distance (`laDist`, at 13 m/s some 143 m towards the
/// right, more with its room to stop), on the last straight - not in the bend, as when the run
/// ended with the bend's inner lane - and it takes the way to its trip end.
#[test]
fn a_dead_lanes_run_beside_the_lane_that_leads_goes_on_round_a_bend() {
    let arc = |r: f64| -> Vec<(f64, f64)> { (0..=12).map(|k| { let a = (k as f64 * 5.0).to_radians(); (100.0 + r * a.sin(), -200.0 + r * a.cos()) }).collect() };
    let (end_l, end_r) = (*arc(200.0).last().unwrap(), *arc(196.5).last().unwrap());
    let dir = (-(60.0f64).to_radians().sin(), (60.0f64).to_radians().cos());
    let along = |p: (f64, f64), d: f64| (p.0 + dir.1 * d, p.1 + dir.0 * d);
    let mut net = Network::default();
    net.lanes.push(road(&[(0.0, 0.0), (100.0, 0.0)], 1, 0, 0.0)); // 0
    net.lanes.push(road(&[(0.0, -3.5), (100.0, -3.5)], 1, 1, 3.5)); // 1
    net.lanes.push(road(&arc(200.0), 2, 0, 0.0)); // 2
    net.lanes.push(road(&arc(196.5), 2, 1, 3.5)); // 3
    net.lanes.push(road(&[end_l, along(end_l, 300.0)], 3, 0, 0.0)); // 4
    net.lanes.push(road(&[end_r, along(end_r, 300.0), along(end_r, 600.0)], 3, 1, 3.5)); // 5
    let fork = along(end_l, 300.0);
    // (the left lane's ways on at the fork: off to the left, or straight on beside lane 5's way)
    let (h, n) = ((dir.1, dir.0), (-dir.0, dir.1));
    let at = |a: f64, b: f64| (fork.0 + h.0 * a + n.0 * b, fork.1 + h.1 * a + n.1 * b);
    net.lanes.push(road(&[fork, at(20.0, 2.0), at(40.0, 30.0)], 4, 0, 0.0)); // 6
    net.lanes.push(road(&[fork, at(20.0, 8.0), at(300.0, 30.0)], 5, 0, 0.0)); // 7
    net.link(1.5);
    net.compute_conflicts();
    net.compute_reach();
    assert_eq!((net.lanes[2].right, net.lanes[4].right, net.lanes[4].next.len()), (Some(3), Some(5), 2));
    assert!(net.lanes[3].length() + 1.0 < net.lanes[2].length(), "the bend's right lane is the shorter");
    // (a car whose trip end, drawn on its first frame where the look-ahead sees the fork, is lane 5)
    let (mut sim, mut cars) = (1..=60u64)
        .map(|id| {
            let mut sim = Simulation::new(&net, vec![]);
            let mut me = car(id, 0, 10.0, 13.0);
            me.state.change_cooldown = 0.0;
            me.state.max_speed_kmh = 13.0 * 3.6;
            me.params.lc_speed_gain = 0.0;
            me.params.lc_keep_right = 0.0;
            let mut cars = vec![me];
            sim.tick(&net, DT, &mut cars);
            (sim, cars)
        })
        .find(|(_, cars)| cars[0].state.goal == Some(5))
        .expect("a car drawing lane 5");
    let id = cars[0].id;
    let (mut started_at, mut lanes, mut t) = (None, vec![0usize], 0.0f32);
    while t < 60.0 {
        sim.tick(&net, DT, &mut cars);
        let Some(me) = cars.iter().find(|c| c.id == id) else { break };
        if started_at.is_none() && me.state.change.is_some() {
            started_at = Some((me.state.lane, me.state.s));
        }
        if lanes.last() != Some(&me.state.lane) {
            lanes.push(me.state.lane);
        }
        t += DT;
    }
    let (lane, s) = started_at.expect("a strategic change");
    assert!(lane == 4, "the change begins on the last straight, within the strategic distance of the fork - not in the bend, where the run ended at the bend's inner lane: lane {lane} s {s:.1}");
    assert!(lanes.contains(&5), "takes the way to its trip end: lanes {lanes:?}");
}

/// L3-D3: a blocked urgent strategic change is given up after the driver's own patience. Car 1 on
/// the left lane of `fork_ahead`, its trip end lane 3 off the right-hand way, beside a jam standing
/// on the right lane to the fork: with `strategic_patience` 5 s it stands at its stop before the
/// dead lane's end, then drives on through the fork on its own lane's way (lane 2 or 5) within the
/// 30 s the jam stands, counted `STRATEGIC_GAVE_UP` (not `STRATEGIC_MISSED`), and its trip end is
/// drawn again; with a patience longer than the jam stands it waits and takes the turn (as
/// `a_blocked_strategic_change_stops_before_the_dead_lanes_end_and_waits`).
#[test]
fn a_blocked_strategic_change_is_given_up_after_the_drivers_patience() {
    for patience in [5.0f32, 1000.0] {
        let net = fork_ahead();
        let mut sim = Simulation::new(&net, vec![]);
        let mut me = car(1, 0, 200.0, 13.0);
        me.state.change_cooldown = 0.0;
        me.state.goal = Some(3);
        me.params.lc_speed_gain = 0.0;
        me.params.lc_keep_right = 0.0;
        me.params.strategic_patience = patience;
        let mut cars = vec![me];
        for k in 0..12u64 {
            cars.push(standing(10 + k, 1, 220.0 + 6.5 * k as f32));
        }
        let g = |a: &std::sync::atomic::AtomicU64| a.load(std::sync::atomic::Ordering::Relaxed);
        let (gave0, missed0) = (g(&omsi_sim::traffic::STRATEGIC_GAVE_UP), g(&omsi_sim::traffic::STRATEGIC_MISSED));
        let (mut lanes, mut stood, mut t) = (vec![0usize], 0.0f32, 0.0f32);
        while t < 50.0 {
            if t >= 30.0 {
                cars.retain(|c| c.id == 1);
            }
            sim.tick(&net, DT, &mut cars);
            let Some(me) = cars.iter().find(|c| c.id == 1) else { break };
            if lanes.last() != Some(&me.state.lane) {
                lanes.push(me.state.lane);
            }
            if me.state.speed < 0.1 && t < 30.0 {
                stood += DT;
            }
            t += DT;
        }
        let (gave, missed) = (g(&omsi_sim::traffic::STRATEGIC_GAVE_UP) - gave0, g(&omsi_sim::traffic::STRATEGIC_MISSED) - missed0);
        if patience < 30.0 {
            assert!(lanes.len() >= 2 && (lanes[1] == 2 || lanes[1] == 5) && stood >= patience - 0.5 && gave >= 1, "gives up after {patience} s and drives on its own lane's way: lanes {lanes:?}, stood {stood:.1}, gave up {gave}, missed {missed}");
        } else {
            assert!(lanes.starts_with(&[0, 1, 3]) && gave == 0, "waits and takes the turn: lanes {lanes:?}, gave up {gave}");
        }
    }
}

/// Fix 5-2: a car signalling a plain change follows the nearer of its own lane's leader and the
/// target lane's (SUMO's `MSVehicle::planMoveInternal` adapts to the shadow lane's leaders too while
/// a change is under way). Car 1 at 10 m/s on the left lane of `two_lanes`, signalling a change to
/// the free right lane, a car crawling at 2 m/s 14 m ahead on its own lane: it brakes behind that
/// car while it signals (following only the target lane's leader, none, it drove on at 10 m/s) and
/// its body never touches it.
#[test]
fn a_signalling_car_follows_its_own_lanes_leader() {
    let net = two_lanes();
    let mut sim = Simulation::new(&net, vec![]);
    let mut me = car(1, 0, 100.0, 10.0);
    me.params.lc_speed_gain = 0.0;
    me.params.lc_keep_right = 0.0;
    me.state.start_change(&net, 1, 2);
    assert!(me.state.change.is_some_and(|c| c.wait > 0.0));
    let mut slow = car(2, 0, 118.0, 2.0);
    slow.state.max_speed_kmh = 2.0 * 3.6;
    slow.state.change_cooldown = 1000.0;
    let cars_init = vec![me, slow];
    let mut cars = cars_init;
    let (mut least, mut t, mut sig_least) = (f32::INFINITY, 0.0f32, f32::INFINITY);
    while t < 8.0 {
        sim.tick(&net, DT, &mut cars);
        sim.update_positions(&net, &mut cars);
        let a = cars.iter().find(|c| c.id == 1).unwrap();
        let b = cars.iter().find(|c| c.id == 2).unwrap();
        let sep = a.obb().separation(&b.obb()) as f32;
        least = least.min(sep);
        if a.state.change.is_some_and(|c| c.wait > 0.0) {
            sig_least = sig_least.min(sep);
        }
        t += DT;
    }
    assert!(least > 0.0 && sig_least > 0.5, "no contact with the leader standing on its own lane: least {least:.2}, while signalling {sig_least:.2}");
}

/// L3-D6: the in-lane squeeze (least offset keeping `MIN_GAP_LAT`, L3-D1) applies on a junction's
/// pieces too. Lane 0 (x 0..100) runs into piece 1 of a junction object (x 100..160, 3.5 m wide,
/// crossed by piece 3 at x 130) and on to lane 2. A car stands on piece 1 at x 104, just inside the
/// junction, 2.0 m off its axis, its body 1.1 m from the axis - in car 1's way with the corridor's
/// margin, as the lorry's nose at viz2 j17 was; car 1 starts from rest on lane 0 at x 88, outside
/// the junction, and is held there by it (without L3-D6 it waits: the squeeze was kept off a
/// junction ahead).
/// It gets round the standing car inside the junction at the least offset, never within
/// `SQUEEZE_CLEARANCE` of it and not past the piece's drawn road, and goes on along lane 2.
#[test]
fn a_car_squeezes_round_a_body_across_its_way_inside_a_junction() {
    let wide_piece = |pts: &[(f64, f64)], path: u16| {
        let mut l = LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), LaneKind::Street, 3.5);
        l.source = 2;
        l.key = Some(LaneKey { tile: (0, 0), id: 9, path });
        l.priority = 128.0;
        l
    };
    let mut net = Network::default();
    net.lanes.push(road(&[(0.0, 0.0), (100.0, 0.0)], 1, 0, 0.0)); // 0
    net.lanes.push(wide_piece(&[(100.0, 0.0), (160.0, 0.0)], 0)); // 1
    net.lanes.push(road(&[(160.0, 0.0), (400.0, 0.0)], 2, 0, 0.0)); // 2
    net.lanes.push(wide_piece(&[(130.0, -40.0), (130.0, 40.0)], 1)); // 3
    net.link(1.5);
    net.compute_conflicts();
    net.compute_reach();
    net.build_grid();
    let mut sim = Simulation::new(&net, vec![]);
    let mut me = car(1, 0, 88.0, 0.0);
    me.params.lc_speed_gain = 0.0;
    me.params.lc_keep_right = 0.0;
    let mut ob = standing(2, 1, 4.0);
    ob.stopped = 60.0;
    ob.state.lateral = 2.0;
    ob.state.lateral_target = 2.0;
    ob.state.lateral_ramp = (2.0, 2.0, 0.0, 1.0);
    let mut cars = vec![me, ob];
    let (mut squeezed, mut least, mut max_lat, mut t) = (false, f32::INFINITY, 0.0f32, 0.0f32);
    while t < 40.0 {
        sim.tick(&net, DT, &mut cars);
        sim.update_positions(&net, &mut cars);
        let me = cars.iter().find(|c| c.id == 1).unwrap();
        let o = cars.iter().find(|c| c.id == 2).unwrap();
        squeezed |= me.passing.is_some_and(|p| p.lane.is_none());
        least = least.min(me.obb().separation(&o.obb()) as f32);
        max_lat = max_lat.max(me.state.lateral.abs());
        t += DT;
    }
    let me = cars.iter().find(|c| c.id == 1).unwrap();
    assert!(squeezed && me.state.lane == 2, "squeezes round the body inside the junction and goes on: squeezed {squeezed}, lane {}", me.state.lane);
    assert!(least > 0.3 && max_lat < 1.75 - 0.9, "least {least:.2} m to the body, swung {max_lat:.2} m (drawn road 1.75 m, half width 0.9)");
}

/// Fix 7-2: the rule against overtaking a vehicle on the side it indicates is about passing it
/// alongside (road rule, ours); a car whose way crosses in front of the indicating vehicle's nose is
/// not barred by it. As viz2 j17: car 1 goes east along lane 0 into a junction piece; a car stands on
/// the crossing piece heading south, its nose 1.1 m left of car 1's axis, indicating right (west, its
/// turn). Car 1 squeezes right past its nose at the least offset (L3-D6) and goes on, never within
/// `SQUEEZE_CLEARANCE` of it.
#[test]
fn a_car_passes_in_front_of_a_turning_vehicles_nose_whatever_it_indicates() {
    let wide_piece = |pts: &[(f64, f64)], path: u16| {
        let mut l = LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), LaneKind::Street, 3.5);
        l.source = 2;
        l.key = Some(LaneKey { tile: (0, 0), id: 9, path });
        l.priority = 128.0;
        l
    };
    let mut net = Network::default();
    net.lanes.push(road(&[(0.0, 0.0), (100.0, 0.0)], 1, 0, 0.0)); // 0
    net.lanes.push(wide_piece(&[(100.0, 0.0), (160.0, 0.0)], 0)); // 1
    net.lanes.push(road(&[(160.0, 0.0), (400.0, 0.0)], 2, 0, 0.0)); // 2
    net.lanes.push(wide_piece(&[(106.0, 40.0), (106.0, -40.0)], 1)); // 3, southbound
    net.link(1.5);
    net.compute_conflicts();
    net.compute_reach();
    net.build_grid();
    let mut sim = Simulation::new(&net, vec![]);
    let mut me = car(1, 0, 88.0, 0.0);
    me.params.lc_speed_gain = 0.0;
    me.params.lc_keep_right = 0.0;
    // (its front 2.0 m ahead of its middle: the nose at y 1.1)
    let mut ob = standing(2, 3, 40.0 - 3.1);
    ob.stopped = 60.0;
    ob.blinker_shown = (2, 0.0);
    let mut cars = vec![me, ob];
    let (mut squeezed, mut least, mut t) = (false, f32::INFINITY, 0.0f32);
    while t < 70.0 {
        sim.tick(&net, DT, &mut cars);
        sim.update_positions(&net, &mut cars);
        if let Some(o) = cars.iter_mut().find(|c| c.id == 2) {
            o.blinker_shown = (2, 0.0);
        }
        let me = cars.iter().find(|c| c.id == 1).unwrap();
        let o = cars.iter().find(|c| c.id == 2).unwrap();
        squeezed |= me.passing.is_some_and(|p| p.lane.is_none());
        least = least.min(me.obb().separation(&o.obb()) as f32);
        t += DT;
    }
    let me = cars.iter().find(|c| c.id == 1).unwrap();
    assert!(squeezed && me.state.lane == 2, "passes in front of the nose: squeezed {squeezed}, lane {}", me.state.lane);
    assert!(least > 0.3, "least {least:.2} m to the indicating car");
}

/// A blocked urgent change shows its side while it waits for the gap (SUMO: the blocked wish stays
/// in the vehicle's own state, `MSLaneChanger::change`'s `setOwnState`, and
/// `MSVehicle::setBlinkerInformation` lights the blinker for it), and what it has shown counts
/// towards the driver's signalling time: car 1 of `a_blocked_strategic_change_stops_before_the_dead_lanes_end_and_waits`,
/// standing before the dead lane's end beside the jam, indicates right; the jam gone after 30 s -
/// far longer than its `signal_before_change` - its change starts with nothing left to signal and
/// moves over at once, where a fresh change signals its whole time first.
#[test]
fn a_blocked_urgent_change_indicates_while_it_waits_and_moves_over_as_the_gap_opens() {
    let net = fork_ahead();
    let mut sim = Simulation::new(&net, vec![]);
    let mut me = car(1, 0, 200.0, 13.0);
    me.state.change_cooldown = 0.0;
    me.state.goal = Some(3);
    me.params.lc_speed_gain = 0.0;
    me.params.lc_keep_right = 0.0;
    assert!(me.params.signal_before_change > 0.5, "the driver signals before a fresh change: {}", me.params.signal_before_change);
    let mut cars = vec![me];
    for k in 0..12u64 {
        cars.push(standing(10 + k, 1, 220.0 + 6.5 * k as f32));
    }
    let (mut t, mut shown_waiting, mut started) = (0.0f32, None, None);
    while t < 40.0 {
        if t >= 30.0 {
            cars.retain(|c| c.id == 1);
        }
        sim.tick(&net, DT, &mut cars);
        let me = cars.iter().find(|c| c.id == 1).expect("car 1");
        if (t - 29.9).abs() < DT * 0.5 {
            assert!(me.state.change.is_none() && me.state.speed < 0.1, "still waiting for the gap: change {:?} v {:.1}", me.state.change, me.state.speed);
            shown_waiting = Some(me.blinker_shown);
        }
        if let (None, Some(c)) = (started, me.state.change) {
            started = Some((t, c.dir, c.wait));
        }
        t += DT;
    }
    let (shown, since) = shown_waiting.expect("sampled while waiting");
    let (at, dir, wait) = started.expect("the jam gone, the change starts");
    assert!(shown != 0 && shown == dir, "indicates the side of its change while it waits: shown {shown}, change dir {dir}");
    assert!(since < 25.0, "and has done so for long: since {since:.1} s");
    assert!(at >= 30.0 && wait == 0.0, "the change starts once the gap is there, nothing left to signal: at {at:.2} s, wait {wait:.2} s");
}

/// Final review S1: the road a change takes (`change_space`, SUMO's `LCA_INSUFFICIENT_SPACE` test,
/// the lane-end fit, the trip end's lanes beside) reads the driver's own signalling time: at
/// 14 m/s a driver who signals 3.6 s needs 14 * 1.6 + 1.0 * (3.6^2 - 2^2) / 2 = 26.9 m more road
/// than one who signals 2 s; a spawned car's state carries its driver's time.
#[test]
fn the_road_a_change_takes_reads_the_drivers_own_signalling_time() {
    let d = omsi_sim::traffic::change_space(14.0, 1.0, 3.6) - omsi_sim::traffic::change_space(14.0, 1.0, 2.0);
    assert!((d - (14.0 * 1.6 + 0.5 * (3.6f32 * 3.6 - 4.0))).abs() < 1e-3, "{d}");
    let c = car(1, 0, 0.0, 0.0);
    assert_eq!(c.state.signal_before_change, c.params.signal_before_change);
}

/// Final review item 6: inside a junction a squeeze keeps off a meeting stretch a foe is coming to
/// before the body has left it (SUMO: an approaching foe by its arrival, `MSLink::blockedAtTime`),
/// not only one moving in it. The junction of
/// `a_car_squeezes_round_a_body_across_its_way_inside_a_junction`, the car standing on piece 1 at
/// x 124, so the squeeze round it runs over the crossing with piece 3 (x 130); car 1 from rest at
/// x 98, and car 3 coming north on lane 4 into piece 3 at 8 m/s from y -104. Without the rule car
/// 1 squeezed at 9.6 s, before car 3 was through (13.6 s); now it begins no squeeze while car 3
/// would reach the crossing before car 1 has cleared it, and goes on along lane 2 after.
#[test]
fn a_squeeze_inside_a_junction_waits_for_a_foe_coming_to_the_meeting_stretch() {
    let wide_piece = |pts: &[(f64, f64)], path: u16| {
        let mut l = LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), LaneKind::Street, 3.5);
        l.source = 2;
        l.key = Some(LaneKey { tile: (0, 0), id: 9, path });
        l.priority = 128.0;
        l
    };
    let mut net = Network::default();
    net.lanes.push(road(&[(0.0, 0.0), (100.0, 0.0)], 1, 0, 0.0)); // 0
    net.lanes.push(wide_piece(&[(100.0, 0.0), (160.0, 0.0)], 0)); // 1
    net.lanes.push(road(&[(160.0, 0.0), (400.0, 0.0)], 2, 0, 0.0)); // 2
    net.lanes.push(wide_piece(&[(130.0, -40.0), (130.0, 40.0)], 1)); // 3
    net.lanes.push(road(&[(130.0, -200.0), (130.0, -40.0)], 3, 0, 0.0)); // 4, leads into 3
    net.lanes.push(road(&[(130.0, 40.0), (130.0, 300.0)], 4, 0, 0.0)); // 5
    net.link(1.5);
    net.compute_conflicts();
    net.compute_reach();
    net.build_grid();
    let mut sim = Simulation::new(&net, vec![]);
    let mut me = car(1, 0, 98.0, 0.0);
    me.params.lc_speed_gain = 0.0;
    me.params.lc_keep_right = 0.0;
    let mut ob = standing(2, 1, 24.0);
    ob.stopped = 60.0;
    ob.state.lateral = 2.0;
    ob.state.lateral_target = 2.0;
    ob.state.lateral_ramp = (2.0, 2.0, 0.0, 1.0);
    let mut foe = car(3, 4, 96.0, 8.0);
    foe.state.max_speed_kmh = 8.0 * 3.6;
    foe.state.change_cooldown = 1000.0;
    let mut cars = vec![me, ob, foe];
    let (mut first_squeeze, mut foe_through, mut least, mut t) = (None, None, f32::INFINITY, 0.0f32);
    while t < 40.0 {
        sim.tick(&net, DT, &mut cars);
        sim.update_positions(&net, &mut cars);
        let me = cars.iter().find(|c| c.id == 1).unwrap();
        if first_squeeze.is_none() && me.passing.is_some_and(|p| p.lane.is_none()) {
            first_squeeze = Some(t);
        }
        if let Some(f) = cars.iter().find(|c| c.id == 3) {
            if foe_through.is_none() && f.pos.y - f.state.rear as f64 > 1.75 + 0.9 {
                foe_through = Some(t);
            }
            least = least.min(me.obb().separation(&f.obb()) as f32);
        }
        t += DT;
    }
    let me = cars.iter().find(|c| c.id == 1).unwrap();
    let (sq, through) = (first_squeeze.expect("squeezes in the end"), foe_through.expect("the foe goes through"));
    assert!(sq >= through - 0.5, "no squeeze before the coming foe is through: squeeze at {sq:.2} s, foe through at {through:.2} s");
    assert!(least > 0.3 && me.state.lane == 2, "least {least:.2} m to the foe, lane {}", me.state.lane);
}

/// L3-D4 (final review M2): the lane-change impatience grows over the driver's `strategic_patience`
/// while its urgent change is blocked and is 0 again once the change has moved over (ours: no
/// growth for speed-gain changes and no slow decay, where SL2015 grows it for both and lets it
/// decay at the same rate). Car 1 of `a_blocked_strategic_change_stops_before_the_dead_lanes_end_and_waits`
/// with a patience of 40 s: after 20 s blocked its impatience is near half; the jam gone, it
/// changes, and once on lane 1 its impatience is 0.
#[test]
fn lane_change_impatience_grows_while_an_urgent_change_is_blocked_and_resets_after() {
    let net = fork_ahead();
    let mut sim = Simulation::new(&net, vec![]);
    let mut me = car(1, 0, 200.0, 13.0);
    me.state.change_cooldown = 0.0;
    me.state.goal = Some(3);
    me.params.lc_speed_gain = 0.0;
    me.params.lc_keep_right = 0.0;
    me.params.strategic_patience = 40.0;
    let mut cars = vec![me];
    for k in 0..12u64 {
        cars.push(standing(10 + k, 1, 220.0 + 6.5 * k as f32));
    }
    let (mut blocked_since, mut mid, mut after, mut t) = (None, None, None, 0.0f32);
    while t < 45.0 {
        if t >= 30.0 {
            cars.retain(|c| c.id == 1);
        }
        sim.tick(&net, DT, &mut cars);
        let me = cars.iter().find(|c| c.id == 1).expect("car 1");
        if blocked_since.is_none() && me.lc_impatience > 0.0 {
            blocked_since = Some(t);
        }
        if let Some(b) = blocked_since {
            if mid.is_none() && t - b >= 20.0 {
                mid = Some(me.lc_impatience);
            }
        }
        if me.state.lane != 0 && me.state.change.is_none() && after.is_none() {
            after = Some(me.lc_impatience);
        }
        t += DT;
    }
    let mid = mid.expect("blocked 20 s");
    assert!((mid - 0.5).abs() < 0.05, "half its patience blocked: impatience {mid:.3}");
    assert_eq!(after, Some(0.0), "moved over: impatience back to 0");
}
