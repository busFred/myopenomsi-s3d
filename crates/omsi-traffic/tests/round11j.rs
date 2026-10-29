//! Round 11 area J3: two paths converging at a shallow angle are a meeting place where the
//! bodies on them would touch (J11-2).
//! `cargo test --release -p omsi-traffic --test round11j -- --nocapture`

use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network};
use omsi_traffic::{Simulation, Vehicle, VehicleClass};

const DT: f32 = 1.0 / 30.0;

fn lane(pts: &[(f64, f64)], kind: LaneKind, width: f32) -> omsi_sim::traffic::Lane {
    LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), kind, width)
}

/// A spline lane (source 1).
fn road(pts: &[(f64, f64)], id: i64) -> omsi_sim::traffic::Lane {
    let mut l = lane(pts, LaneKind::Street, 3.0);
    l.source = 1;
    l.key = Some(LaneKey { tile: (0, 0), id, path: 0 });
    l
}

/// A lane of crossing object 9 (source 2), 1 m wide as BeijingCity's pieces are.
fn piece(pts: &[(f64, f64)], path: u16) -> omsi_sim::traffic::Lane {
    let mut l = lane(pts, LaneKind::Street, 1.0);
    l.source = 2;
    l.key = Some(LaneKey { tile: (0, 0), id: 9, path });
    l.priority = 128.0;
    l
}

fn finish(mut net: Network) -> Network {
    net.link(1.5);
    net.compute_conflicts();
    net.compute_reach();
    net
}

fn vehicle(id: u64, class: VehicleClass, route: Vec<usize>, s: f32, speed: f32, half_len: f32, half_width: f32) -> Vehicle {
    let mut st = AiState::new(route[0], s, id);
    st.speed = speed;
    st.front = half_len;
    st.rear = half_len;
    st.length = 2.0 * half_len;
    // (5 m/s at most: they go through side by side at the speed they come)
    st.max_speed_kmh = 18.0;
    st.veh_type = if class == VehicleClass::Car { 0 } else { 2 };
    st.route = route;
    let mut v = Vehicle::new(id, class, st, half_width, id * 7919);
    v.fresh = 0.0;
    v
}

/// Run `secs`; returns the least separation of the two bodies and how far each got (odometer).
fn run(net: &Network, sim: &mut Simulation, cars: &mut Vec<Vehicle>, secs: f32) -> (f32, Vec<f32>) {
    let mut least = f32::INFINITY;
    let mut t = 0.0;
    sim.update_positions(net, cars);
    while t < secs {
        sim.tick(net, DT, cars);
        for &i in &sim.removed {
            cars.remove(i);
        }
        sim.update_positions(net, cars);
        for (a, va) in cars.iter().enumerate() {
            for vb in cars.iter().skip(a + 1) {
                least = least.min(va.obb().separation(&vb.obb()) as f32);
            }
        }
        if std::env::var("J3_DEBUG").is_ok() && ((t / 0.5).fract() < DT / 0.5) {
            for c in cars.iter() {
                println!("t {t:.1} #{} lane {} s {:.1} v {:.2} why {:?} pos ({:.1},{:.1}) hd {:.1}", c.id, c.state.lane, c.state.s, c.state.speed, c.last.why, c.pos.x, c.pos.y, c.heading);
            }
        }
        t += DT;
    }
    (least, cars.iter().map(|c| c.state.odometer).collect())
}

/// Two turn pieces of object 9 into a two-lane exit: piece 2 straight along y = 0, piece 3
/// coming in from y = 4.4 at a shallow angle (3 degrees) to end 2.3 m beside it, each into its
/// own exit lane (4 and 5, 2.3 m apart: a road's lanes, which stay apart). Lanes 0 and 1 are the
/// approaches. The centre lines come no closer than 2.3 m, under two buses' half widths
/// (2 x 1.25 m).
fn converging_turns() -> Network {
    let mut net = Network::default();
    net.lanes.push(road(&[(-100.0, 0.0), (0.0, 0.0)], 1));
    net.lanes.push(road(&[(-100.0, 4.4), (0.0, 4.4)], 2));
    net.lanes.push(piece(&[(0.0, 0.0), (40.0, 0.0)], 0));
    net.lanes.push(piece(&[(0.0, 4.4), (40.0, 2.3)], 1));
    net.lanes.push(road(&[(40.0, 0.0), (200.0, 0.0)], 3));
    net.lanes.push(road(&[(40.0, 2.3), (200.0, 2.3)], 4));
    finish(net)
}

/// [J11-2] Two buses side by side at 5 m/s on two paths converging at a shallow angle: the
/// paths are a side merge (one comes in to the other, `Converging::comes_in`), and the two
/// buses' half widths (2 x 1.25 m) are more than the centre lines' 2.3 m (`Crossing::apart`), so
/// they meet there: one goes behind the other and they never touch. The exit lanes, parallel from
/// their start, stay apart. (With centre lines alone under `CONVERGE_DIST` there was no meeting
/// place and the two touched, -0.43 m. A car beside the bus is held already: the bus's body comes
/// into its corridor ahead of it, its lead.)
#[test]
fn j11_2_two_bodies_on_paths_converging_at_a_shallow_angle_do_not_touch() {
    let net = converging_turns();
    let meet = net.crossings[2].iter().find(|c| c.other == 3).copied();
    let exits = net.crossings[4].iter().find(|c| c.other == 5).copied();
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![
        vehicle(1, VehicleClass::Bus, vec![1, 3, 5], 80.0, 5.0, 5.9, 1.25),
        vehicle(2, VehicleClass::Bus, vec![0, 2, 4], 80.0, 5.0, 5.9, 1.25),
    ];
    let (least, odo) = run(&net, &mut sim, &mut cars, 30.0);
    println!("meeting {meet:?} exits {exits:?} least {least:.2} odometers {odo:?}");
    assert!(meet.is_some_and(|m| m.merge && m.side && (m.apart - 2.3).abs() < 0.05), "no side merge between the converging paths: {meet:?}");
    assert_eq!(exits, None, "the exit lanes run parallel from their start and stay apart");
    assert!(least > 0.0, "the two buses touched: {least:.2} m");
    assert!(odo.iter().all(|&o| o > 100.0), "both go through: {odo:?}");
}

/// [J11-2] Two cars side by side on the same two paths: their half widths (2 x 0.9 m) are less
/// than the centre lines' 2.3 m, so they do not meet at the side merge: both go through abreast
/// at their speed, neither waits for the other, and they do not touch.
#[test]
fn j11_2_two_cars_on_paths_converging_to_more_than_their_widths_go_through_abreast() {
    let net = converging_turns();
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![
        vehicle(1, VehicleClass::Car, vec![1, 3, 5], 80.0, 5.0, 2.2, 0.9),
        vehicle(2, VehicleClass::Car, vec![0, 2, 4], 80.0, 5.0, 2.2, 0.9),
    ];
    let (least, odo) = run(&net, &mut sim, &mut cars, 30.0);
    println!("least {least:.2} odometers {odo:?}");
    assert!(least > 0.0, "the two cars touched: {least:.2} m");
    assert!((odo[0] - odo[1]).abs() < 2.0, "one waited for the other: odometers {odo:?}");
}
