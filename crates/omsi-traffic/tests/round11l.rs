//! Lane changes of the drivers' own: the gap checked again while the change signals (SUMO's
//! `checkChange` every step), the follower looked up back over the lanes leading in
//! (`getFollowersOnConsecutive`).
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

/// The lane changed from, 0 (x 304..600 at y -3.5, fed by lane 1 from x 0), beside lane 2 on
/// its left (x 304..600 at y 0, one spline with lane 0). Lane 2 is fed by a 4 m piece, lane 3
/// (x 300..304), which is fed by lane 4 (x 0..300): a follower on lane 4 is two lanes back of
/// the target.
fn pieces() -> Network {
    let mut net = Network::default();
    net.lanes.push(road(&[(304.0, -3.5), (600.0, -3.5)], 1, 1, 3.5));
    net.lanes.push(road(&[(0.0, -3.5), (304.0, -3.5)], 2, 0, 0.0));
    net.lanes.push(road(&[(304.0, 0.0), (600.0, 0.0)], 1, 0, 0.0));
    net.lanes.push(road(&[(300.0, 0.0), (304.0, 0.0)], 3, 0, 0.0));
    net.lanes.push(road(&[(0.0, 0.0), (300.0, 0.0)], 4, 0, 0.0));
    net.link(1.5);
    net.compute_conflicts();
    net
}

/// Runs the changer (index 0) with its change onto lane 2 signalled at t 0 and the follower
/// (index 1) on lane 4 at `follower_s`, 13 m/s; returns (the change dropped before the car moved
/// over - when, s - the change carried out onto lane 2).
fn signalled_change(follower_s: f32) -> (Option<f32>, bool) {
    let net = pieces();
    assert_eq!(net.lanes[0].left, Some(2), "lane 2 lies left of lane 0");
    assert!(net.prev[2].contains(&3) && net.prev[3].contains(&4), "lane 4 feeds lane 3 feeds lane 2");
    let mut sim = Simulation::new(&net, vec![]);
    let mut me = car(1, 0, 6.0, 4.0);
    me.state.start_change(&net, 2, 1);
    let mut f = car(2, 4, follower_s, 13.0);
    f.state.change_cooldown = 1000.0;
    let mut cars = vec![me, f];
    let (mut dropped, mut done) = (None, false);
    let mut t = 0.0f32;
    while t < 5.0 {
        sim.tick(&net, DT, &mut cars);
        let st = &cars[0].state;
        if st.change.is_none() && st.lane == 0 && !done {
            dropped = Some(t);
            break;
        }
        if st.lane == 2 {
            done = true;
        }
        t += DT;
    }
    (dropped, done)
}

/// A follower two lanes back of the target (lane 4, beyond the 4 m piece), 15 m behind the
/// changer's rear and closing at 13 m/s, needs far more than that gap (its standing gap, the
/// changer's time gap at its speed, the braking to the changer's speed): the signalled change is
/// dropped at once, while the follower is still on lane 4, before the car moves over. With the
/// follower 250 m back the same change goes.
#[test]
fn follower_two_lanes_back_aborts_a_signalled_change() {
    // (the changer's rear at x 308; the follower's front at x 293, on lane 4 for 0.5 s more)
    let (dropped, done) = signalled_change(291.0);
    assert!(dropped.is_some_and(|t| t < 0.2) && !done, "the change into the closing follower's way is dropped at once (dropped at {dropped:?}, done {done})");
    let (dropped, done) = signalled_change(50.0);
    assert!(dropped.is_none() && done, "with the follower far back the change is carried out (dropped at {dropped:?}, done {done})");
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

/// Car 1 at rest on the right lane 6.5 m (front to rear: 2.5 m, just over the full-lock room of
/// about 2.3-2.4 m for a 4 m car over 3.5 m, `plan::bypass_room`) behind car 2 standing, with
/// `beside` standing on the left lane if given; runs `secs`: (when its speed-gain wish to the left
/// first passed `SPEED_GAIN_THRESHOLD`, the change it started - (t, full lock) -, the least gap of
/// its body to any other, whether it ended on the left lane).
fn from_rest(beside: Option<f32>, secs: f32) -> (Option<f32>, Option<(f32, bool)>, f32, bool) {
    let net = two_lanes();
    let mut sim = Simulation::new(&net, vec![]);
    let mut me = car(1, 1, 100.0, 0.0);
    me.state.change_cooldown = 0.0;
    let mut cars = vec![me, standing(2, 1, 106.5)];
    if let Some(s) = beside {
        cars.push(standing(3, 0, s));
    }
    let (mut wished, mut started, mut least) = (None, None, f32::INFINITY);
    let mut t = 0.0f32;
    while t < secs {
        sim.tick(&net, DT, &mut cars);
        let me = cars.iter().find(|c| c.id == 1).unwrap();
        if wished.is_none() && me.speed_gain[0] > omsi_traffic::plan::SPEED_GAIN_THRESHOLD {
            wished = Some(t);
        }
        if started.is_none() {
            if let Some(c) = me.state.change {
                started = Some((t, c.full_lock));
            }
        }
        for o in cars.iter().filter(|o| o.id != 1) {
            least = least.min(me.obb().separation(&o.obb()) as f32);
        }
        t += DT;
    }
    let me = cars.iter().find(|c| c.id == 1).unwrap();
    (wished, started, least, me.state.lane == 0)
}

/// SUMO LC2013's speed-gain wish accumulates from rest: standing behind a standing car with the
/// left lane empty, the relative gain is about 1 per second (the safe speed beside over
/// `RELGAIN_NORMALIZATION_MIN_SPEED`, its own 0), so the wish passes its threshold within half
/// a second, well before the obstacle motive's 4 s of standing; the car then steps out from 2.5 m
/// behind at full lock and ends on the left lane without coming within `SQUEEZE_CLEARANCE` of a
/// body.
#[test]
fn the_speed_gain_wish_accumulates_from_rest_and_the_car_steps_out_at_full_lock() {
    let (wished, started, least, left) = from_rest(None, 12.0);
    let t = wished.expect("the speed-gain wish never passed its threshold");
    assert!(t < 0.5, "the wish passed its threshold only at {t:.2} s");
    let (ts, full_lock) = started.expect("never stepped out");
    assert!(ts < 4.0 && full_lock, "stepped out at {ts:.2} s, full lock {full_lock}");
    assert!(least > 0.3 && left, "least {least:.2}, on the left lane {left}");
}

/// The step-out from rest at full lock sweeps its turned body past the bodies near: with a car
/// standing on the left lane level with it, a metre or three ahead, it does not set out (and
/// touches nothing).
#[test]
fn no_step_out_from_rest_into_a_car_standing_in_the_target_lane() {
    for s in [100.0f32, 103.0, 106.0] {
        let (_, started, least, left) = from_rest(Some(s), 12.0);
        assert!(started.is_none() && !left, "car beside at s {s}: set out {started:?}");
        assert!(least > 0.3, "car beside at s {s}: least {least:.2}");
    }
}

/// Two eastbound lanes 4 m wide, 6 m apart (0 the left at y 0, 1 the right at y -6): the right
/// lane's drawn road reaches 3.7 m to its left (to the left lane's traffic, 6 - 2 - 0.3), so a car
/// squeezes past a car standing in its middle inside the lane (it needs 0.9 + 0.6 + 0.9 + 0.9 =
/// 3.3 m).
fn wide_lanes() -> Network {
    let mut net = Network::default();
    let mut l0 = road(&[(0.0, 0.0), (600.0, 0.0)], 1, 0, 0.0);
    l0.width = 4.0;
    net.lanes.push(l0);
    let mut l1 = road(&[(0.0, -6.0), (600.0, -6.0)], 1, 1, 6.0);
    l1.width = 4.0;
    l1.drivable = (3.7, 2.0);
    net.lanes.push(l1);
    net.link(1.5);
    net.compute_conflicts();
    net.compute_reach();
    net
}

/// Car 1 (no lane changes of its own) rolls up at 8 m/s from x 120 on the right lane behind car 3,
/// standing for good at x 193.5, 2.5 m behind car 2 standing for good: car 3 is a standing
/// obstacle, and car 1 squeezes past it on the left inside the lane. Car 3's own changes start
/// once `step_out_after` s have passed (its change cooldown); with `indicator` it shows its left
/// indicator throughout. Runs 40 s: (when car 1 started its squeeze, when car 3 started a change
/// and whether at full lock, the least gap between the two bodies).
fn squeezer_and_step_out(step_out_after: f32, indicator: bool) -> (Option<f32>, Option<(f32, bool)>, f32) {
    let net = wide_lanes();
    assert_eq!(net.lanes[1].left, Some(0), "lane 0 lies left of lane 1");
    let mut sim = Simulation::new(&net, vec![]);
    let mut o = standing(2, 1, 200.0);
    o.stopped = 30.0;
    let mut a = standing(3, 1, 193.5);
    a.stopped = 30.0;
    a.state.change_cooldown = step_out_after;
    let mut b = car(1, 1, 120.0, 8.0);
    b.params.lc_speed_gain = 0.0;
    b.state.change_cooldown = 1000.0;
    let mut cars = vec![b, a, o];
    let (mut squeezed, mut stepped, mut least) = (None, None, f32::INFINITY);
    let mut t = 0.0f32;
    while t < 40.0 {
        if indicator {
            let a = cars.iter_mut().find(|c| c.id == 3).unwrap();
            a.state.signal = 1;
            a.state.signal_time = 0.25;
        }
        sim.tick(&net, DT, &mut cars);
        let b = cars.iter().find(|c| c.id == 1).unwrap();
        let a = cars.iter().find(|c| c.id == 3).unwrap();
        least = least.min(b.obb().separation(&a.obb()) as f32);
        if squeezed.is_none() && b.passing.is_some() {
            squeezed = Some(t);
        }
        if stepped.is_none() {
            stepped = a.state.change.map(|c| (t, c.full_lock));
        }
        t += DT;
    }
    (squeezed, stepped, least)
}

/// A car does not squeeze past an AI car whose indicator shows the side it would pass on (SUMO: no
/// overtaking within the lane of one that has announced a move into that path), as it does not
/// past a bus or the player signalling out; with no indicator shown it squeezes past. And a
/// squeezer and a full-lock step-out from rest of the car it squeezes past never touch, whichever
/// starts first.
#[test]
fn no_squeeze_past_a_car_indicating_to_that_side_and_a_squeezer_never_touches_a_step_out() {
    let (squeezed, _, least) = squeezer_and_step_out(1000.0, false);
    assert!(squeezed.is_some() && least > 0.55, "no indicator: squeezed {squeezed:?}, least {least:.2}");
    let (squeezed, _, least) = squeezer_and_step_out(1000.0, true);
    assert!(squeezed.is_none() && least > 0.3, "left indicator: squeezed {squeezed:?}, least {least:.2}");
    let mut full_lock = 0;
    for k in 0..=12 {
        let after = 11.0 + 0.5 * k as f32;
        let (squeezed, stepped, least) = squeezer_and_step_out(after, false);
        full_lock += stepped.is_some_and(|s| s.1) as usize;
        assert!(least > 0.0, "step-out after {after} s: squeezed {squeezed:?}, stepped {stepped:?}, least {least:.2}");
    }
    assert!(full_lock > 0, "no full-lock step-out among the runs");
}

/// `from_rest`'s step-out, with car 3 standing beside car 1's rear on the kerb side, 2.2 m right of
/// the right lane's axis at x `s3`, where the rear swings as the full-lock move turns the body
/// left: (put there from the start, or in the frame after the change started) -> (the change
/// started - when, full lock -, when it was dropped with the car still on its lane, the least gap
/// between the bodies, whether it ended on the left lane).
fn step_out_with_a_body_beside(s3: f32, from_start: bool) -> (Option<(f32, bool)>, Option<f32>, f32, bool) {
    let net = two_lanes();
    let mut sim = Simulation::new(&net, vec![]);
    let mut me = car(1, 1, 100.0, 0.0);
    me.state.change_cooldown = 0.0;
    let mut cars = vec![me, standing(2, 1, 106.5)];
    let lat = 2.2;
    let mut c3 = standing(3, 1, s3);
    c3.state.lateral = lat;
    c3.state.lateral_target = lat;
    c3.state.lateral_ramp = (lat, lat, 0.0, 1.0);
    if from_start {
        cars.push(c3.clone());
    }
    let (mut started, mut dropped, mut least, mut placed) = (None, None, f32::INFINITY, from_start);
    let mut t = 0.0f32;
    while t < 12.0 {
        sim.tick(&net, DT, &mut cars);
        let me = cars.iter().find(|c| c.id == 1).unwrap().clone();
        if started.is_none() {
            started = me.state.change.map(|c| (t, c.full_lock));
        }
        if started.is_some() && dropped.is_none() && me.state.change.is_none() && me.state.lane == 1 {
            dropped = Some(t);
        }
        if !placed && started.is_some() {
            cars.push(c3.clone());
            placed = true;
        }
        for o in cars.iter().filter(|o| o.id != 1) {
            least = least.min(me.obb().separation(&o.obb()) as f32);
        }
        t += DT;
    }
    (started, dropped, least, cars.iter().find(|c| c.id == 1).unwrap().state.lane == 0)
}

/// A step-out from rest sweeps its move again on every frame of its signal (as `bypass_start`
/// swept it at the start, SUMO checking each sublane it moves into every step): a car that comes
/// to stand beside its rear, where the full-lock move swings it, once the signal has begun drops
/// the step-out before the car moves (it would have swung its rear into it: least -0.16 m without
/// the sweep); the same car there from the start keeps it from setting out at all.
#[test]
fn a_step_out_is_swept_again_on_every_frame_of_its_signal() {
    let (started, _, least, left) = step_out_with_a_body_beside(99.5, true);
    assert!(started.is_none() && !left && least > 0.3, "there from the start: started {started:?}, least {least:.2}");
    for s3 in [98.0f32, 99.5, 100.0] {
        let (started, dropped, least, left) = step_out_with_a_body_beside(s3, false);
        assert!(started.is_some_and(|s| s.1), "car 3 at {s3}: no full-lock step-out {started:?}");
        assert!(dropped.is_some_and(|d| d < started.unwrap().0 + 0.1) && !left && least > 0.3, "car 3 at {s3}: dropped {dropped:?}, least {least:.2}, on the left lane {left}");
    }
}

/// `pieces` with a second lane, 5, leading into the 4 m piece (lane 3): from (0, 30) to (300, 0),
/// merging with lane 4 there.
fn two_branches() -> Network {
    let mut net = pieces();
    net.lanes.push(road(&[(0.0, 30.0), (300.0, 0.0)], 5, 0, 0.0));
    net.link(1.5);
    net.compute_conflicts();
    net
}

/// SUMO's `getFollowersOnConsecutive` keeps, over every lane leading in, the follower missing the
/// most of its rear gap, not the nearest: with a slow car 21 m behind on lane 4 (it needs 5 m) and a
/// fast one 40 m behind on the merging branch, lane 5 (at 14 m/s it needs 48 m), the signalled
/// change onto lane 2 is dropped; with the fast one 90 m back it goes.
#[test]
fn the_follower_missing_most_of_its_gap_on_any_lane_leading_in_is_the_one_checked() {
    for (fast_x, dropped_expected) in [(266.0f64, true), (216.0, false)] {
        let net = two_branches();
        assert!(net.prev[3].contains(&4) && net.prev[3].contains(&5), "lanes 4 and 5 lead into lane 3");
        let mut sim = Simulation::new(&net, vec![]);
        let mut me = car(1, 0, 6.0, 4.0);
        me.state.start_change(&net, 2, 1);
        let mut slow = car(2, 4, 285.0, 2.0);
        slow.state.change_cooldown = 1000.0;
        let (fast_s, _) = net.lanes[5].nearest_point(DVec3::new(fast_x, 30.0 * (1.0 - fast_x / 300.0), 0.0)).unwrap();
        let mut fast = car(3, 5, fast_s, 14.0);
        fast.state.change_cooldown = 1000.0;
        let mut cars = vec![me, slow, fast];
        let (mut dropped, mut t) = (false, 0.0f32);
        while t < 1.0 {
            sim.tick(&net, DT, &mut cars);
            let st = &cars[0].state;
            if st.change.is_none() && st.lane == 0 {
                dropped = true;
                break;
            }
            t += DT;
        }
        assert_eq!(dropped, dropped_expected, "fast follower at x {fast_x}: dropped {dropped}");
    }
}
