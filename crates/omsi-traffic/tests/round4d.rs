//! Headless scenarios for the timetable buses of round 4d: a bus at the dead end of its
//! road, a bus stopping at its pole with people beside its way, its standing time.

use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKind, Network};
use omsi_traffic::{Simulation, Vehicle, VehicleClass, Why};

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

/// A timetable bus (12 m) on `route`, `s` into its first lane, at `speed`.
fn bus(id: u64, route: Vec<usize>, s: f32, speed: f32) -> Vehicle {
    let mut st = AiState::new(route[0], s, id);
    st.speed = speed;
    st.front = 6.0;
    st.rear = 6.0;
    st.length = 12.0;
    st.max_speed_kmh = 60.0;
    st.route = route;
    let mut v = Vehicle::new(id, VehicleClass::ScheduledBus, st, 1.25, id * 7919);
    v.fresh = 0.0;
    v
}

/// Run `secs`, taking off what the simulation removes; `each` sees every frame. Returns the
/// ids removed, with the time and their speed then.
fn run(net: &Network, sim: &mut Simulation, vehicles: &mut Vec<Vehicle>, secs: f32, mut each: impl FnMut(f32, &mut Simulation, &mut Vec<Vehicle>)) -> Vec<(u64, f32, f32)> {
    let mut gone = Vec::new();
    let mut t = 0.0;
    while t < secs {
        each(t, sim, vehicles);
        sim.tick(net, DT, vehicles);
        for &i in &sim.removed {
            gone.push((vehicles[i].id, t, vehicles[i].state.speed));
            vehicles.remove(i);
        }
        t += DT;
    }
    gone
}

/// [ROLLBACK r4d-B1] A timetable bus whose route ends where the road does (the lane runs
/// into nothing) drives on at its speed and is taken off at the end; where the road goes on
/// it stops at the end of its route as before.
#[test]
fn a_timetable_bus_drives_off_a_dead_end_and_stops_at_a_route_end_that_goes_on() {
    // lane 0 east 0..200 m, nothing after it
    let net = finish(vec![lane(&[(0.0, 0.0), (200.0, 0.0)])]);
    let mut sim = Simulation::new(&net, vec![]);
    let mut vs = vec![bus(1, vec![0], 40.0, 10.0)];
    let mut whys = Vec::new();
    let gone = run(&net, &mut sim, &mut vs, 40.0, |_, _, vs| whys.extend(vs.iter().map(|v| v.last.why)));
    assert_eq!(gone.len(), 1, "taken off at the dead end");
    assert!(gone[0].2 > 6.0, "at its speed, not stopped first: {:?}", gone[0]);
    assert!(!whys.contains(&Why::End), "never held at the end");

    // the road goes on into lane 1: held at the end of its route, not taken off
    let net = finish(vec![lane(&[(0.0, 0.0), (200.0, 0.0)]), lane(&[(200.0, 0.0), (300.0, 0.0)])]);
    assert_eq!(net.lanes[0].next, vec![1]);
    let mut sim = Simulation::new(&net, vec![]);
    let mut vs = vec![bus(1, vec![0], 40.0, 10.0)];
    let gone = run(&net, &mut sim, &mut vs, 60.0, |_, _, _| {});
    assert!(gone.is_empty());
    assert_eq!(vs[0].last.why, Why::End);
    assert!(vs[0].state.speed < 0.1 && vs[0].state.s + vs[0].state.front > 198.0, "stands at the end: s {}", vs[0].state.s);

    // a stop of its own on the dead-end lane: it stops there (the service holds it)
    let net = finish(vec![lane(&[(0.0, 0.0), (200.0, 0.0)])]);
    let mut sim = Simulation::new(&net, vec![]);
    let mut vs = vec![bus(1, vec![0], 40.0, 10.0)];
    let gone = run(&net, &mut sim, &mut vs, 60.0, |_, _, vs| {
        if let Some(v) = vs.first_mut() {
            v.external.stop_at = Some(150.0 - v.state.s);
        }
    });
    assert!(gone.is_empty());
    assert!((vs[0].state.s + vs[0].state.front - 150.0).abs() < 1.0, "at its stop: s {}", vs[0].state.s);
}

/// [round 6 B2] A timetable bus whose trip is over at the end of its track
/// (`External::trip_over`: no stop left, the route complete) is not held at the end of its
/// route where the road goes on (case08: T11's bus stood 11.7 min at Why::End on lane
/// 10871, next [10468]): it drives on as off a dead end; without the flag (a route still
/// open at a tile edge, a stop pending) it stops there as before.
#[test]
fn b2_a_bus_whose_trip_is_over_at_its_track_end_is_not_held_there() {
    let net = finish(vec![lane(&[(0.0, 0.0), (200.0, 0.0)]), lane(&[(200.0, 0.0), (300.0, 0.0)])]);
    assert_eq!(net.lanes[0].next, vec![1]);
    let mut sim = Simulation::new(&net, vec![]);
    let mut vs = vec![bus(1, vec![0], 40.0, 10.0)];
    let mut whys = Vec::new();
    let gone = run(&net, &mut sim, &mut vs, 40.0, |_, _, vs| {
        for v in vs.iter_mut() {
            v.external.trip_over = true;
        }
        whys.extend(vs.iter().map(|v| v.last.why));
    });
    assert!(!whys.contains(&Why::End), "never held at the end");
    assert_eq!(gone.len(), 1, "off the end of its route");
    assert!(gone[0].2 > 6.0, "at its speed: {:?}", gone[0]);
    // (the flag is the planner's only exemption: a stop of the service still holds it)
    let mut sim = Simulation::new(&net, vec![]);
    let mut vs = vec![bus(1, vec![0], 40.0, 10.0)];
    let gone = run(&net, &mut sim, &mut vs, 60.0, |_, _, vs| {
        if let Some(v) = vs.first_mut() {
            v.external.trip_over = true;
            v.external.stop_at = Some(150.0 - v.state.s);
        }
    });
    assert!(gone.is_empty());
    assert!((vs[0].state.s + vs[0].state.front - 150.0).abs() < 1.0, "at its stop: s {}", vs[0].state.s);
}

/// A bus on lane 0 (east along y = 0) coming up to its stop: its front comes to rest at
/// x = 150, in the bay 1.23 m right of the lane (as the service asks).
fn to_the_stop(v: &mut Vehicle) {
    let front = v.state.s + v.state.front;
    v.external.stop_at = Some(150.0 - v.state.s);
    if 150.0 - front < 30.0 {
        v.external.bay = Some(1.23);
    }
}

fn person(x: f64, y: f64, waiting: bool) -> omsi_traffic::Person {
    omsi_traffic::Person { pos: DVec3::new(x, y, 0.0), vel: glam::DVec2::ZERO, on_crossing: false, still: 10.0, waiting_for_bus: waiting, at_kerb: false, waits_at: None, id: None, approach: None }
}

/// [ROLLBACK r4d-B4] Four people standing at the kerb 2.5 m beside the bus's nose (1.3 m
/// from its middle in the bay) do not hold it short of its pole; one inside its footprint
/// does, and once they are gone the bus creeps up to the pole. A creep under 0.5 m/s does
/// not set its standing time back.
#[test]
fn a_bus_pulls_up_to_its_pole_beside_the_people_and_only_one_in_its_way_holds_it() {
    let net = finish(vec![lane(&[(0.0, 0.0), (300.0, 0.0)])]);
    let mut sim = Simulation::new(&net, vec![]);
    let mut vs = vec![bus(1, vec![0], 40.0, 8.0)];
    // (1.6-1.7 m from the middle of the bus in the bay: its footprint (1.25 m) and 0.3 m clear)
    sim.people = vec![person(149.0, -2.85, true), person(148.0, -2.85, true), person(146.0, -2.95, false), person(144.0, -2.9, false)];
    run(&net, &mut sim, &mut vs, 40.0, |_, _, vs| to_the_stop(&mut vs[0]));
    let b = &vs[0];
    assert!((b.state.s + b.state.front - 150.0).abs() < 0.6, "at its pole: front at {}", b.state.s + b.state.front);
    assert!(b.state.speed < 0.1 && (b.state.lateral - 1.23).abs() < 0.1, "in the bay: {}", b.state.lateral);

    // one standing 0.2 m outside its footprint, inside the clearance: held short of them
    let mut sim = Simulation::new(&net, vec![]);
    let mut vs = vec![bus(1, vec![0], 40.0, 8.0)];
    sim.people = vec![person(146.0, -1.23 - 1.45, false)];
    run(&net, &mut sim, &mut vs, 40.0, |_, _, vs| to_the_stop(&mut vs[0]));
    let front = vs[0].state.s + vs[0].state.front;
    assert!(front < 145.0, "kept 0.3 m clear: front at {front}");
    // one on a zebra beside its way (2 m from its middle): held, as any car is
    let mut sim = Simulation::new(&net, vec![]);
    let mut vs = vec![bus(1, vec![0], 40.0, 8.0)];
    sim.people = vec![omsi_traffic::Person { on_crossing: true, ..person(130.0, -1.23 - 2.0, false) }];
    run(&net, &mut sim, &mut vs, 40.0, |_, _, vs| to_the_stop(&mut vs[0]));
    let front = vs[0].state.s + vs[0].state.front;
    assert!(front < 129.0, "stops for the zebra: front at {front}");

    // somebody standing inside its footprint 10 m before the pole: held short of them
    let mut sim = Simulation::new(&net, vec![]);
    let mut vs = vec![bus(1, vec![0], 40.0, 8.0)];
    sim.people = vec![person(140.0, -1.2, false)];
    let (mut held_at, mut prev_stopped, mut reset_by_creep) = (None, 0.0f32, false);
    run(&net, &mut sim, &mut vs, 80.0, |t, sim, vs| {
        to_the_stop(&mut vs[0]);
        let b = &vs[0];
        if b.state.speed >= 0.3 && b.state.speed < 0.5 && b.stopped < prev_stopped {
            reset_by_creep = true;
        }
        prev_stopped = b.stopped;
        if held_at.is_none() && b.state.speed < 0.05 && b.stopped > 8.0 {
            held_at = Some((t, b.state.s + b.state.front, b.last.why));
            sim.people.clear();
        }
    });
    let (_, front, why) = held_at.expect("held");
    assert!(front < 139.0 && front > 130.0, "short of them: front at {front}");
    assert_eq!(why, Why::People);
    let b = &vs[0];
    assert!((b.state.s + b.state.front - 150.0).abs() < 0.6, "crept up to its pole: front at {}", b.state.s + b.state.front);
    assert!(!reset_by_creep);
}

/// [round 6 B3] Bus waiters' places keep `max(w, 2.6)/2 + 0.58` m off a lane's axis (humans.rs
/// `clear_of_lanes`): an 11.6 m coach of the random traffic (car class, 1.26 m half width:
/// BeijingCity's Daba HHF.ovh) in the middle of a 1 m bus lane is held by a waiter whose body
/// overlaps its width ([P2] SUMO's nextBlocking: 1.0 and 1.46 m off its way, case02's waiters
/// held it 148 s) and passes one whose body is clear of its flank (1.55 m: 0.01 m clear; the
/// 0.3 m margin round the footprint that held it there too is gone) or standing where the
/// places are moved to, 1.88 m.
#[test]
fn b3_a_coach_passes_bus_waiters_standing_clear_of_the_lane() {
    let net = finish(vec![lane(&[(0.0, 0.0), (300.0, 0.0)])]);
    let pass = |off: f64| {
        let mut st = AiState::new(0, 40.0, 1);
        st.speed = 8.0;
        st.front = 5.8;
        st.rear = 5.8;
        st.length = 11.6;
        st.max_speed_kmh = 50.0;
        let mut v = Vehicle::new(1, VehicleClass::Car, st, 1.26, 7919);
        v.fresh = 0.0;
        let mut sim = Simulation::new(&net, vec![]);
        sim.people = (0..9).map(|k| person(80.0 + k as f64 * 0.8, -off, true)).collect();
        let mut vs = vec![v];
        let mut held = false;
        run(&net, &mut sim, &mut vs, 12.0, |_, _, vs| held |= vs[0].last.why == Why::People);
        (held, vs[0].state.s)
    };
    for off in [1.0, 1.46] {
        let (held, s) = pass(off);
        assert!(held && s + 5.8 < 80.0, "held by them {off} m off: s {s}");
    }
    for off in [1.55, 1.3 + 0.28 + 0.3] {
        let (held, s) = pass(off);
        assert!(!held && s > 90.0, "past them at {off} m: s {s}");
    }
}
