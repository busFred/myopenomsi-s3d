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
