//! Round 8 P8 (people and buses): pull back into the lane (P8-4), buses and waiters (P8-1),
//! the pass's sweep (P8-2), the people strip on a bend (P8-5).
use glam::{DVec2, DVec3};
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKind, Network};
use omsi_traffic::{Person, Simulation, Vehicle, VehicleClass, Why};

const DT: f32 = 1.0 / 30.0;

fn lane(pts: &[(f64, f64)]) -> omsi_sim::traffic::Lane {
    LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), LaneKind::Street, 3.5)
}

fn finish(lanes: Vec<omsi_sim::traffic::Lane>) -> Network {
    let mut net = Network { lanes, ..Default::default() };
    net.link(1.5);
    net.build_grid();
    net.compute_conflicts();
    net
}

fn vehicle(id: u64, class: VehicleClass, route: Vec<usize>, s: f32, speed: f32, half_len: f32, half_width: f32) -> Vehicle {
    let mut st = AiState::new(route[0], s, id);
    st.speed = speed;
    st.front = half_len;
    st.rear = half_len;
    st.length = 2.0 * half_len;
    st.max_speed_kmh = 60.0;
    st.route = route;
    let mut v = Vehicle::new(id, class, st, half_width, id * 7919);
    v.fresh = 0.0;
    v
}

fn bus(id: u64, route: Vec<usize>, s: f32, speed: f32) -> Vehicle {
    vehicle(id, VehicleClass::ScheduledBus, route, s, speed, 6.0, 1.25)
}

fn car(id: u64, route: Vec<usize>, s: f32, speed: f32) -> Vehicle {
    vehicle(id, VehicleClass::Car, route, s, speed, 2.25, 0.9)
}

/// A bus standing 2.8 m out in a bay on a straight road east, its service just sent it back
/// (bay 0, a ramp of `ramp` m laid), a car coming in the lane at `v` from `gap` behind the
/// bus's rear: (when the bus moves off, its largest heading off the lane while standing, the
/// least distance between the two, the car's hardest braking).
fn pull_out(gap: f32, v: f32, ramp: f32) -> (f32, f32, f64, f32) {
    let net = finish(vec![lane(&[(0.0, 0.0), (400.0, 0.0)])]);
    let mut sim = Simulation::new(&net, vec![]);
    let mut b = bus(1, vec![0], 200.0, 0.0);
    b.state.lateral = 2.8;
    b.state.lateral_target = 0.0;
    b.external.bay = Some(0.0);
    let odo = b.state.odometer;
    b.state.lateral_ramp = (2.8, 0.0, odo, ramp);
    let c = car(2, vec![0], 200.0 - 6.0 - gap - 2.25, v);
    let mut vs = vec![b, c];
    let (mut min, mut maxdec, mut go_t, mut hd) = (f64::INFINITY, 0.0f32, -1.0f32, 0.0f32);
    let mut prev = v;
    let mut t = 0.0;
    sim.update_positions(&net, &mut vs);
    while t < 20.0 {
        sim.tick(&net, DT, &mut vs);
        sim.update_positions(&net, &mut vs);
        if go_t < 0.0 && vs[0].state.speed > 0.3 {
            go_t = t;
        }
        if go_t < 0.0 {
            hd = hd.max((vs[0].heading - 90.0).abs());
        }
        maxdec = maxdec.max((prev - vs[1].state.speed) / DT);
        prev = vs[1].state.speed;
        min = min.min(vs[0].obb().separation(&vs[1].obb()));
        t += DT;
    }
    (go_t, hd, min, maxdec)
}

/// [P8-4, pass 1 S1] A car at 13.9 m/s 40 m behind is not cut up: the bus waits for it (the
/// gap a lane change accepts in front of it), standing straight in its bay meanwhile, and the
/// car does not brake for it at all; a car standing 5 m behind does not keep the bus in.
#[test]
fn p84_a_bus_leaving_its_bay_waits_for_the_car_coming_and_stands_straight() {
    let (go, hd, min, dec) = pull_out(40.0, 13.9, 12.0);
    eprintln!("40 m at 13.9: goes {go:.1} s, heading off {hd:.1} deg, min sep {min:.2}, car's braking {dec:.2}");
    assert!(go > 2.0, "the bus pulled out in front of the car at once ({go:.1} s)");
    assert!(hd < 1.0, "the bus stood turned {hd:.1} degrees into the lane");
    assert!(min > 0.5 && dec < 1.0, "the car had to brake for the bus: {dec:.2} m/s², {min:.2} m");
    let (go, _, min, _) = pull_out(5.0, 0.0, 12.0);
    assert!(go >= 0.0 && go < 3.0 && min > 0.3, "a car standing behind kept the bus in: goes {go:.1}, {min:.2} m");
}

/// [P8-4] One clock for everybody: a car held standing out of its lane accrues the give-up
/// waiting time like any held car.
#[test]
fn p84_the_hold_runs_the_waiting_clock() {
    let net = finish(vec![lane(&[(0.0, 0.0), (400.0, 0.0)])]);
    let mut sim = Simulation::new(&net, vec![]);
    let mut p = car(1, vec![0], 200.0, 0.0);
    p.state.lateral = 2.5;
    p.state.lateral_target = 0.0;
    let odo = p.state.odometer;
    p.state.lateral_ramp = (2.5, 0.0, odo, 12.0);
    // a van stopped in the lane beside its nose for good (held by its own stop)
    let mut van = car(2, vec![0], 204.0, 0.0);
    van.external.stop_at = Some(0.1);
    let mut vs = vec![p, van];
    let mut t = 0.0;
    while t < 6.0 {
        vs[1].external.stop_at = Some(vs[1].state.front + 0.1);
        sim.tick(&net, DT, &mut vs);
        sim.update_positions(&net, &mut vs);
        t += DT;
    }
    assert_eq!(vs[0].last.why, Why::PullOut);
    assert!(vs[0].waiting > 5.0, "waiting clock {:.1}", vs[0].waiting);
}

/// A two-way road east (lane 0, y = 0) and back (lane 1, y = 3.5), 3 m lanes, a car parked in
/// lane 0 at 200 m; `far`: another parked at the oncoming lane's far kerb level with it, 1.8 m
/// off that lane's axis: its flank 0.0 m from where the passing body's flank will be.
fn pass_round_parked(far: bool) -> bool {
    let mut a = lane(&[(0.0, 0.0), (2000.0, 0.0)]);
    a.width = 3.0;
    let mut b = lane(&[(2000.0, 3.5), (0.0, 3.5)]);
    b.width = 3.0;
    let net = finish(vec![a, b]);
    let mut sim = Simulation::new(&net, vec![]);
    sim.parked.insert(0, vec![omsi_traffic::Parked { s: 200.0, lateral: 0.0, half_width: 0.9, half_len: 2.2 }]);
    if far {
        sim.parked.insert(1, vec![omsi_traffic::Parked { s: 1800.0, lateral: 1.8, half_width: 0.9, half_len: 2.2 }]);
    }
    let mut vs = vec![car(1, vec![0], 150.0, 8.0)];
    let mut started = false;
    let mut t = 0.0;
    while t < 30.0 {
        sim.tick(&net, DT, &mut vs);
        sim.update_positions(&net, &mut vs);
        started |= vs[0].passing.is_some_and(|p| p.lane.is_some());
        t += DT;
    }
    started
}

/// [P8-2, pass 1 S2] The pass on the oncoming lane is swept against parked cars too: round a
/// car parked in the lane it goes; with another parked at the far kerb flush with the passing
/// body, the path is not clear by SQUEEZE_CLEARANCE and it does not start (before: it did).
#[test]
fn p82_no_pass_into_a_car_parked_in_the_oncoming_lane() {
    assert!(pass_round_parked(false), "no pass round a lone parked car");
    assert!(!pass_round_parked(true), "a pass started into a car parked in the oncoming lane");
}

/// A timetable bus serving stops `serves` drives east at 8 m/s past somebody waiting for a bus
/// at stop `waits_at`, standing at x 120 `off` m right of its way: (held for them, least
/// distance between their disc and its body).
fn bus_past_waiter(serves: [Option<i64>; 2], waits_at: Option<i64>, off: f64) -> (bool, f64) {
    let net = finish(vec![lane(&[(0.0, 0.0), (300.0, 0.0)])]);
    let mut sim = Simulation::new(&net, vec![]);
    let mut vs = vec![bus(1, vec![0], 60.0, 8.0)];
    sim.people = vec![Person { pos: DVec3::new(120.0, -off, 0.0), vel: DVec2::ZERO, on_crossing: false, still: 10.0, waiting_for_bus: true, at_kerb: false, waits_at, id: None, approach: None }];
    let (mut held, mut min, mut t) = (false, f64::INFINITY, 0.0);
    sim.update_positions(&net, &mut vs);
    while t < 15.0 {
        vs[0].external.serves = serves;
        sim.tick(&net, DT, &mut vs);
        sim.update_positions(&net, &mut vs);
        held |= vs[0].last.why == Why::People;
        min = min.min(omsi_sim::collision::Obb::point(sim.people[0].pos, 0.28).separation(&vs[0].obb()));
        t += DT;
    }
    (held, min)
}

/// [P8-1, pass 1 B1] A bus pulls in beside the waiters of the stops it serves - the one it
/// comes up to and the one it served last, known by the stop they wait at - held only by one
/// whose body is over its footprint; a waiter of any other stop is held for by the car rule.
/// No contact either way (round 4d's 0.3 m inside the flank is gone).
#[test]
fn p81_a_bus_pulls_in_beside_its_own_stops_waiters_only() {
    // 1.54 m off the way: clear of a 1.25 m half-wide body and a 0.28 m disc, inside the car
    // rule's 1.25 + 0.3
    let own = bus_past_waiter([Some(7), None], Some(7), 1.54);
    let last = bus_past_waiter([Some(8), Some(7)], Some(7), 1.54);
    let other = bus_past_waiter([Some(8), None], Some(7), 1.54);
    let inside = bus_past_waiter([Some(7), None], Some(7), 1.0);
    eprintln!("own {own:?} last {last:?} other {other:?} inside {inside:?}");
    assert!(!own.0 && own.1 > 0.0, "held by (or touched) a waiter of its next stop beside it: {own:?}");
    assert!(!last.0 && last.1 > 0.0, "held by (or touched) a waiter of the stop it served: {last:?}");
    assert!(other.0 && other.1 > 0.0, "a waiter of another stop not held for by the car rule: {other:?}");
    assert!(inside.0 && inside.1 > 0.0, "drove into a waiter of its own stop inside its footprint: {inside:?}");
}

/// [P8 map data, light_call] On red a car that cannot stop before its line braking hard goes
/// on; "cannot stop" is reckoned to the line itself, no half metre past it: at 10 m/s the hard
/// stop takes 7.81 m, so 7.6 m before the line it goes (before: it stopped 0.2 m over the
/// line), 8.0 m before it stops.
#[test]
fn light_call_reckons_to_the_line() {
    use omsi_sim::traffic::Aspect;
    use omsi_traffic::perception::{Perception, SignalView};
    let mut c = car(1, vec![0], 50.0, 10.0);
    c.state.speed = 10.0;
    let p = Perception::default();
    let at = |gap: f32| SignalView { dist: c.state.front + gap, ctl: 0, light: 0, aspect: Aspect::Red, until_go: None };
    assert!((10.0f32 * 10.0 / (2.0 * omsi_traffic::plan::HARD_BRAKE) - 7.8125).abs() < 1e-3);
    assert!(omsi_traffic::plan::light_call(&c, &p, &at(7.6)).go, "stops over the line instead of clearing it");
    assert!(!omsi_traffic::plan::light_call(&c, &p, &at(8.0)).go, "runs a red it can stop for");
}

/// [P8-4, pass 2 A1] A car out of its lane with a car queued right behind it in the lane
/// (stood up to it as its lead, 1 m off its rear) moves back in: a standing follower needs no
/// gap (SUMO's secure gap at v = 0 is 0), and where it stands now is no move (viz5 case06
/// #18756 stood 90 s held by the car queued behind it).
#[test]
fn p84_a_car_queued_behind_does_not_keep_it_out_for_good() {
    let net = finish(vec![lane(&[(0.0, 0.0), (400.0, 0.0)])]);
    let mut sim = Simulation::new(&net, vec![]);
    let mut p = car(1, vec![0], 200.0, 0.0);
    p.state.lateral = -1.2;
    p.state.lateral_target = 0.0;
    let odo = p.state.odometer;
    p.state.lateral_ramp = (-1.2, 0.0, odo, 9.6);
    let behind = car(2, vec![0], 200.0 - 4.5 - 1.0, 0.0);
    let mut vs = vec![p, behind];
    let (mut go, mut t) = (-1.0f32, 0.0);
    while t < 10.0 {
        sim.tick(&net, DT, &mut vs);
        sim.update_positions(&net, &mut vs);
        if go < 0.0 && vs[0].state.speed > 0.3 {
            go = t;
        }
        t += DT;
    }
    assert!(go >= 0.0 && go < 3.0, "held by the car queued behind it: goes at {go:.1}");
}

/// [round 8 fix, re-entry] A bus standing 3.48 m out in its bay, part-way along the bay's way
/// IN (the ramp laid is still 0 -> 4.1 m over 30 m, 22.6 m of it driven) when its bay is sent
/// back to 0 - the service's way out comes a frame later - with a car passing it in the lane at
/// 4.4 m/s, its front 7.7 m past the bus's rear (inside its secure gap: the bus is held). Held,
/// the bus stays where it stands: moving the start of the way-in ramp put it on the lane's
/// axis, 3.48 m sideways onto the car (viz2 case10 #31063/#34056, t=1418.2, 2.14 m deep).
#[test]
fn reentry_hold_on_a_stale_way_in_keeps_the_bus_where_it_stands() {
    let net = finish(vec![lane(&[(0.0, 0.0), (400.0, 0.0)])]);
    let mut sim = Simulation::new(&net, vec![]);
    let mut b = bus(1, vec![0], 200.0, 0.0);
    let odo = b.state.odometer;
    b.state.lateral_ramp = (0.0, 4.1, odo - 22.6, 30.0);
    b.state.lateral = 4.1 * omsi_sim::traffic::smooth01(22.6 / 30.0);
    b.state.lateral_target = 0.0;
    b.external.bay = Some(0.0);
    let lat0 = b.state.lateral;
    let c = car(2, vec![0], 200.0 - 6.0 + 7.7 - 2.25, 4.4);
    let mut vs = vec![b, c];
    sim.update_positions(&net, &mut vs);
    let (mut min, mut jump, mut held, mut t) = (f64::INFINITY, 0.0f32, false, 0.0);
    while t < 6.0 {
        sim.tick(&net, DT, &mut vs);
        sim.update_positions(&net, &mut vs);
        held |= vs[0].last.why == Why::PullOut;
        if vs[0].state.speed < 0.1 {
            jump = jump.max((vs[0].state.lateral - lat0).abs());
        }
        min = min.min(vs[0].obb().separation(&vs[1].obb()));
        t += DT;
    }
    eprintln!("held {held}, standing offset moved {jump:.2} m, min sep {min:.2}");
    assert!(held, "the bus was not held for the car beside it");
    assert!(jump < 0.05, "the held bus moved {jump:.2} m sideways where it stood");
    assert!(min > 0.0, "the bus and the car went into each other: {min:.2} m");
}

/// A pass round a parked car (as `pass_round_parked`); once the car is `trig` m out, a
/// person appears `dx` ahead of its front at y0 and walks across at vy: the least distance
/// between them and whether the car was held for them.
fn pass_person(trig: f32, dx: f64, y0: f64, vy: f64) -> (f64, bool) {
    let mut a = lane(&[(0.0, 0.0), (2000.0, 0.0)]);
    a.width = 3.0;
    let mut b = lane(&[(2000.0, 3.5), (0.0, 3.5)]);
    b.width = 3.0;
    let net = finish(vec![a, b]);
    let mut sim = Simulation::new(&net, vec![]);
    sim.parked.insert(0, vec![omsi_traffic::Parked { s: 200.0, lateral: 0.0, half_width: 0.9, half_len: 2.2 }]);
    let mut vs = vec![car(1, vec![0], 150.0, 8.0)];
    sim.update_positions(&net, &mut vs);
    let (mut t, mut spawned, mut min, mut held) = (0.0f32, false, f64::INFINITY, false);
    while t < 40.0 {
        sim.tick(&net, DT, &mut vs);
        sim.update_positions(&net, &mut vs);
        let c = &vs[0];
        if !spawned && c.passing.is_some() && c.state.lateral.abs() >= trig {
            spawned = true;
            sim.people = vec![Person { pos: DVec3::new(c.pos.x + c.state.front as f64 + dx, y0, 0.0), vel: DVec2::new(0.0, vy), on_crossing: false, still: 0.0, waiting_for_bus: false, at_kerb: false, waits_at: None, id: None, approach: None }];
        }
        if spawned {
            let pp = &mut sim.people[0];
            pp.pos.y += pp.vel.y * DT as f64;
            held |= vs[0].last.why == Why::People;
            min = min.min(omsi_sim::collision::Obb::point(sim.people[0].pos, 0.28).separation(&vs[0].obb()));
        }
        t += DT;
    }
    (min, held)
}

/// [P8-2, pass 2 R1] A walker steps out across the oncoming lane 2 m ahead of a car out on a
/// pass at 7 m/s: the car cannot stop short of them, so it is not stopped across their line -
/// it goes on (the reviewer's probe: the untimed hold stopped it there, -1.21 m); one far
/// enough ahead to stop for is stopped for, clear of them.
#[test]
fn p82_a_passing_car_is_never_stopped_across_a_walkers_line() {
    let (near, _) = pass_person(2.5, 2.0, 6.5, -1.4);
    let (far, far_held) = pass_person(2.5, 12.0, 6.5, -1.4);
    eprintln!("near {near:.2}, far {far:.2} held {far_held}");
    assert!(near > -0.28, "stopped across the walker's line: {near:.2} m");
    assert!(far > 0.0, "touched a walker it could stop for: {far:.2} m");
}
