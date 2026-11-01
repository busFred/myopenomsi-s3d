//! Round 11 area W: a walker crossing a two-lane road mid-block from the far side, timed as
//! SUMO times an approaching person (`perception::walk_across`): the car in the near lane
//! brakes for them at its comfortable deceleration as soon as their path meets its window, and
//! goes on when its rear clears their line before they reach its strip.

use glam::{DVec2, DVec3};
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network};
use omsi_traffic::{Person, Simulation, Vehicle, VehicleClass, Why};

const DT: f32 = 1.0 / 30.0;

/// Two eastbound spline lanes 3.5 m wide (0 the left one at y = 0, 1 the right one at y = -3.5)
/// limited to `kmh`, and a crossing object (9) laid over them at x = `cross_x`: its street piece
/// (lane 2, y = 8, westbound) and its footpath (lane 3, from y = 12 to y = -12) across the street
/// piece and the two spline lanes. The footpath is a piece of a junction's crossing
/// (`Junctions::of_foot_lane`) whose cut of the spline lanes the network does not list (the
/// spline lanes are no lanes of the object), as viz2 j19's crossing 0 across #19609's way.
fn two_lanes(kmh: f32, cross_x: f64) -> Network {
    let mut net = Network::default();
    for (k, y) in [0.0, -3.5].into_iter().enumerate() {
        let mut l = LaneBuilder::polyline(vec![DVec3::new(0.0, y, 0.0), DVec3::new(600.0, y, 0.0)], LaneKind::Street, 3.5);
        l.source = 1;
        l.key = Some(LaneKey { tile: (0, 0), id: 1, path: k as u16 });
        l.offset = -y as f32;
        l.speed_limit_kmh = kmh;
        net.lanes.push(l);
    }
    let piece = |pts: &[(f64, f64)], path: u16, kind: LaneKind| {
        let mut l = LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), kind, 3.0);
        l.source = 2;
        l.key = Some(LaneKey { tile: (0, 0), id: 9, path });
        l.priority = 128.0;
        l
    };
    net.lanes.push(piece(&[(cross_x + 20.0, 8.0), (cross_x - 20.0, 8.0)], 0, LaneKind::Street));
    net.lanes.push(piece(&[(cross_x, 12.0), (cross_x, -12.0)], 1, LaneKind::Sidewalk));
    net.link(1.5);
    net.build_grid();
    net.compute_conflicts();
    net
}

/// The crossing's footpath lane in `two_lanes`.
const FOOTPATH: usize = 3;

/// A sedan (4.62 x 1.94 m) on the right lane at `s` m with `speed` m/s; its top speed 100 km/h
/// as the app gives a car (the lane's limit is what it drives at).
fn sedan(s: f32, speed: f32) -> Vehicle {
    let mut st = AiState::new(1, s, 1);
    st.speed = speed;
    st.front = 2.42;
    st.rear = 2.2;
    st.length = 4.62;
    st.max_speed_kmh = 100.0;
    st.veh_type = 0;
    let mut v = Vehicle::new(1, VehicleClass::Car, st, 0.97, 7919);
    v.fresh = 0.0;
    v
}

/// Somebody at (`x`, `y`) stepping at `vel` along footpath `leg`.
fn walker(x: f64, y: f64, vel: DVec2, leg: Option<usize>) -> Person {
    Person { pos: DVec3::new(x, y, 0.0), vel, on_crossing: false, still: 0.0, waiting_for_bus: false, at_kerb: false, waits_at: None, id: None, approach: None, leg }
}

struct Seen {
    /// At the first frame the people test held the car: its speed and the road from its front
    /// to the walker's line.
    first: Option<(f32, f32)>,
    least_speed: f32,
    /// The least separation of the car's body from the walker's (m).
    least_sep: f32,
    /// The hardest the car braked (m/s^2).
    hardest: f32,
}

/// The sedan rolls for `secs` on lanes limited to `kmh` with one walker stepping at `vel` from
/// `from` along footpath `leg` (the crossing's footpath laid at x = `from.x`, `two_lanes`).
/// A walker coming down to a kerb at `kerb_y` stops there.
fn drive(me: Vehicle, kmh: f32, from: DVec2, vel: DVec2, leg: Option<usize>, kerb_y: Option<f64>, secs: f32) -> Seen {
    let net = two_lanes(kmh, from.x);
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![me];
    let mut p = walker(from.x, from.y, vel, leg);
    let mut seen = Seen { first: None, least_speed: f32::INFINITY, least_sep: f32::INFINITY, hardest: 0.0 };
    let mut t = 0.0;
    while t < secs {
        sim.people = vec![p.clone()];
        let before = cars[0].state.speed;
        sim.tick(&net, DT, &mut cars);
        let me = &cars[0];
        seen.hardest = seen.hardest.max((before - me.state.speed) / DT);
        seen.least_speed = seen.least_speed.min(me.state.speed);
        seen.least_sep = seen.least_sep.min(me.obb().separation(&omsi_sim::collision::Obb::point(p.pos, 0.28)) as f32);
        if seen.first.is_none() && me.last.why == Why::People {
            seen.first = Some((before, (from.x - me.pos.x) as f32 - me.state.front));
        }
        if kerb_y.is_some_and(|k| p.pos.y <= k) {
            p.vel = DVec2::ZERO;
        }
        p.pos += (p.vel * DT as f64).extend(0.0);
        t += DT;
    }
    seen
}

#[test]
fn w1_a_walker_crossing_from_the_far_side_is_braked_for_at_the_comfortable_deceleration() {
    // viz2 case08 #19609: 12.3 m/s in the right lane; a walker steps across both lanes at
    // 1.2 m/s from beyond the left one. Their path enters the car's strip (half width and 0.3 m)
    // 3.9 s on, while the car would be on their line: it brakes from where it can still stop
    // short at its comfortable deceleration and never touches them
    let v0 = 12.3;
    let me = sedan(100.0, v0);
    let comfort = me.params.comfort_brake;
    let seen = drive(me, v0 * 3.6, DVec2::new(100.0 + 2.42 + 60.0, 2.5), DVec2::new(0.0, -1.2), Some(FOOTPATH), None, 12.0);
    let (v, gap) = seen.first.expect("never braked for the walker crossing its way");
    assert!(gap - 1.5 >= v * v / (2.0 * comfort), "first held {gap:.1} m before their line at {v:.1} m/s: inside its comfortable stop ({:.1} m)", v * v / (2.0 * comfort));
    assert!(seen.least_sep > 0.0, "touched the walker: {:.2} m", seen.least_sep);
    assert!(seen.hardest <= comfort, "braked {:.2} m/s^2 (comfortable {comfort:.2})", seen.hardest);
}

#[test]
fn w1_a_car_whose_rear_clears_the_walkers_line_before_they_reach_its_strip_goes_on() {
    // the walker in the left lane's middle (3.5 m off the car's axis) stepping across at
    // 1.2 m/s, 13 m ahead of the car's front: they reach its strip in 1.9 s, its rear clears
    // their line in 1.5 s. It passes first, not stopping late inside their path
    let v0 = 12.3;
    let seen = drive(sedan(100.0, v0), v0 * 3.6, DVec2::new(100.0 + 2.42 + 13.0, 0.0), DVec2::new(0.0, -1.2), Some(FOOTPATH), None, 3.0);
    assert!(seen.least_speed > 0.9 * v0, "braked to {:.1} m/s for a walker it passes first", seen.least_speed);
    assert!(seen.least_sep > 0.0, "touched the walker: {:.2} m", seen.least_sep);
}

#[test]
fn w1_a_slow_car_waits_for_a_walker_crossing_ahead() {
    // at 3 m/s, a walker 2 m off its strip crossing 10 m ahead: it reaches their line after
    // they step in, and stops short of them
    let seen = drive(sedan(100.0, 3.0), 10.8, DVec2::new(100.0 + 2.42 + 10.0, -3.5 + 3.3), DVec2::new(0.0, -1.2), Some(FOOTPATH), None, 6.0);
    assert!(seen.first.is_some(), "never held for the walker");
    assert!(seen.least_sep > 0.0, "touched the walker: {:.2} m", seen.least_sep);
}

#[test]
fn w1_a_walker_along_the_pavement_drifting_in_does_not_hold_the_car() {
    // pass 2: a walker on the pavement 2.6 m right of the near lane's axis walking along the road
    // towards the car at 1.3 m/s and drifting in at 0.3 m/s (more along it than across): SUMO's
    // person on the lane, held for where they are; they never reach the strip while the car is
    // there, so it neither brakes for them nor touches them
    let v0 = 10.0;
    let seen = drive(sedan(100.0, v0), v0 * 3.6, DVec2::new(100.0 + 2.42 + 25.0, -3.5 - 2.6), DVec2::new(-1.3, 0.3), None, None, 3.0);
    assert!(seen.first.is_none(), "held for a walker walking along the pavement");
    assert!(seen.least_sep > 0.0, "touched the walker: {:.2} m", seen.least_sep);
}

#[test]
fn w1_the_people_reach_covers_a_comfortable_stop_at_city_speed() {
    // pass 2: at 13.9 m/s (50 km/h) the comfortable stop at 2 m/s^2 is 48 m: a walker crossing
    // 60 m ahead of the front whose path meets the car's window is braked for at the comfortable
    // deceleration, never harder
    let v0 = 13.9;
    let me = sedan(100.0, v0);
    let comfort = me.params.comfort_brake;
    let seen = drive(me, v0 * 3.6, DVec2::new(100.0 + 2.42 + 75.0, 2.5), DVec2::new(0.0, -1.2), Some(FOOTPATH), None, 14.0);
    let (v, gap) = seen.first.expect("never braked for the walker crossing its way");
    assert!(gap - 1.5 >= v * v / (2.0 * comfort), "first held {gap:.1} m before their line at {v:.1} m/s: inside its comfortable stop");
    assert!(seen.hardest <= comfort, "braked {:.2} m/s^2 (comfortable {comfort:.2})", seen.hardest);
    assert!(seen.least_sep > 0.0, "touched the walker: {:.2} m", seen.least_sep);
}

/// An eastbound road through crossing object 9's piece (lane 1, x -15..45) whose zebra lies at
/// x = 23 (lane 3, 3.5 m wide, y -8..8): the walk's near edge at x = 21.25, its zebra stop
/// (`plan::ZEBRA_STOP`, 2.5 m before the cut) at x = 20.5.
fn road_with_a_zebra() -> Network {
    let lane = |pts: &[(f64, f64)], kind: LaneKind| LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), kind, 3.5);
    let road = |pts: &[(f64, f64)], id: i64| {
        let mut l = lane(pts, LaneKind::Street);
        l.source = 1;
        l.key = Some(LaneKey { tile: (0, 0), id, path: 0 });
        l
    };
    let piece = |pts: &[(f64, f64)], path: u16, kind: LaneKind| {
        let mut l = lane(pts, kind);
        l.source = 2;
        l.key = Some(LaneKey { tile: (0, 0), id: 9, path });
        l.priority = 128.0;
        l
    };
    let mut net = Network::default();
    net.lanes.push(road(&[(-300.0, 0.0), (-15.0, 0.0)], 1));
    net.lanes.push(piece(&[(-15.0, 0.0), (45.0, 0.0)], 0, LaneKind::Street));
    net.lanes.push(road(&[(45.0, 0.0), (400.0, 0.0)], 2));
    net.lanes.push(piece(&[(23.0, -8.0), (23.0, 8.0)], 3, LaneKind::Sidewalk));
    net.link(1.5);
    net.build_grid();
    net.compute_conflicts();
    net
}

#[test]
fn w2_a_car_standing_just_past_its_zebra_stop_is_not_let_onto_the_walk_while_somebody_walks_onto_it() {
    // viz2 case11 #40365: standing just past its zebra stop, the crossing test was off and the
    // strip test let the car roll onto the walk while people walked onto it. SUMO's vehicle asks
    // blockedAtDist until its front is on the crossing: somebody on the crossing walking towards
    // its band from within jmCrossingGap (10 m) holds it. Here 7 m off its axis at 1 m/s; the
    // car's front stays short of the walk
    let net = road_with_a_zebra();
    let mut st = AiState::new(1, (20.6 - 2.0 + 15.0) as f32, 1);
    st.front = 2.0;
    st.rear = 2.0;
    st.length = 4.0;
    st.max_speed_kmh = 50.0;
    let mut me = Vehicle::new(1, VehicleClass::Car, st, 0.9, 7919);
    me.fresh = 0.0;
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![me];
    let vel = DVec2::new(0.0, -1.0);
    let mut p = walker(23.0, 7.0, vel, None);
    p.on_crossing = true;
    let mut furthest = f64::NEG_INFINITY;
    let mut t = 0.0;
    while t < 3.0 {
        sim.people = vec![p.clone()];
        sim.tick(&net, DT, &mut cars);
        let me = &cars[0];
        let h = (me.heading as f64).to_radians();
        furthest = furthest.max(me.pos.x + h.sin() * me.state.front as f64);
        p.pos += (vel * DT as f64).extend(0.0);
        t += DT;
    }
    assert!(furthest < 21.25, "rolled onto the walk with somebody walking onto it: front at x {furthest:.2}");
}

#[test]
fn w3_a_pavement_walker_heading_for_the_kerb_is_not_timed() {
    // review F3: somebody on the pavement (no crossing's footpath piece their leg) walking down
    // to the kerb 0.45 m left of the left lane's edge, 45 m ahead of a car at 12.3 m/s: their
    // step points at the car's strip, but they stop at the kerb, and SUMO times only persons whose
    // route uses a crossing (`registerCrossingApproach`). The car drives on at its speed
    let v0 = 12.3;
    let seen = drive(sedan(100.0, v0), v0 * 3.6, DVec2::new(100.0 + 2.42 + 45.0, 3.0), DVec2::new(0.0, -1.2), None, Some(2.2), 5.0);
    assert!(seen.first.is_none(), "braked for a pavement walker who stops at the kerb");
    assert!(seen.least_speed > 0.9 * v0, "slowed to {:.1} m/s", seen.least_speed);
}

#[test]
fn w3_a_slow_car_on_a_limited_lane_does_not_stop_across_the_walkers_line() {
    // review F1 (`review-r11-w-probe.rs`): a car at 3 m/s on an 11 km/h lane, its top speed
    // 100 km/h; a walker crosses from the middle of the left lane at 1.2 m/s, their line 2-2.5 m
    // ahead of its front. Timed at its top speed the car "cleared first", rolled on and stood across
    // their line (the walker 1.08-1.23 m into its body); timed at its arrival up to its lane's
    // limit (`crossing_time`; SUMO's `getMinimalArrivalTime` to the speed it may pass at) it
    // stands short of them
    for g in [2.0f64, 2.5] {
        let net = two_lanes(11.0, 100.0 + 2.42 + g);
        let mut sim = Simulation::new(&net, vec![]);
        let mut cars = vec![sedan(100.0, 3.0)];
        let vel = DVec2::new(0.0, -1.2);
        let mut p = walker(100.0 + 2.42 + g, 0.0, vel, Some(FOOTPATH));
        let mut least = f32::INFINITY;
        let mut t = 0.0;
        while t < 5.0 {
            sim.people = vec![p.clone()];
            sim.tick(&net, DT, &mut cars);
            least = least.min(cars[0].obb().separation(&omsi_sim::collision::Obb::point(p.pos, 0.28)) as f32);
            p.pos += (vel * DT as f64).extend(0.0);
            t += DT;
        }
        assert!(least > 0.0, "line {g} m ahead: the walker came {:.2} m into the car's body", -least);
    }
}
