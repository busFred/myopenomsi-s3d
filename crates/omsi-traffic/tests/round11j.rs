//! Round 11 area J3: two paths converging at a shallow angle are a meeting place where the
//! bodies on them would touch (J11-2), and the ease of a mutual hold (J11-6c).
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

/// Object 9: piece 0 eastbound along y = 0 and piece 1 northbound along x = 0 crossing it at the
/// origin, each with its way on (lanes 2 east, 3 north).
fn crossing_ways() -> Network {
    let mut net = Network::default();
    let mut east = piece(&[(-40.0, 0.0), (20.0, 0.0)], 0);
    east.width = 3.5;
    let mut north = piece(&[(0.0, -40.0), (0.0, 20.0)], 1);
    north.width = 3.5;
    net.lanes.push(east);
    net.lanes.push(north);
    net.lanes.push(road(&[(20.0, 0.0), (300.0, 0.0)], 2));
    net.lanes.push(road(&[(0.0, 20.0), (0.0, 300.0)], 3));
    finish(net)
}

/// [J11-6c, end to end] The three-car ring of viz5 case02 through `Simulation::tick`: A (id 1)
/// eased off for B (id 2), B standing behind its lead C (id 3), C eased off for A. B cannot move
/// off before A (its lead C, C's ease for A); A is held back for B by nothing physical: A's ease
/// is the one that closes the ring, it ends, and C's stays while A crosses. A goes, then C and B
/// follow; nobody waits out `EASE_MAX`, nobody touches, no new ease is started.
#[test]
fn j11_6c_a_three_car_ring_of_eases_and_a_lead_opens_end_to_end() {
    let net = crossing_ways();
    let mut sim = Simulation::new(&net, vec![]);
    let j = sim.junctions.of_lane[&0].0;
    assert_eq!(sim.junctions.of_lane[&1].0, j, "one junction");
    // (fronts: C 3.5 m short of the crossing, B 2 m behind C, A 3.5 m short on the other way)
    let mut c = vehicle(3, VehicleClass::Car, vec![0, 2], 40.0 - 3.5 - 2.2, 0.0, 2.2, 0.9);
    let mut b = vehicle(2, VehicleClass::Car, vec![0, 2], 40.0 - 3.5 - 2.2 - 4.4 - 2.0, 0.0, 2.2, 0.9);
    let mut a = vehicle(1, VehicleClass::Car, vec![1, 3], 40.0 - 3.5 - 2.2, 0.0, 2.2, 0.9);
    for v in [&mut a, &mut b, &mut c] {
        v.state.max_speed_kmh = 30.0;
    }
    a.ease_for = Some((2, j, 0.0));
    c.ease_for = Some((1, j, 0.0));
    let mut cars = vec![a, b, c];
    let mut t = 0.0;
    let (mut a_ease_end, mut c_ease_end, mut least, mut eases) = (None, None, f32::INFINITY, 0usize);
    let mut eased: Vec<u64> = vec![1, 3];
    let mut through = [false; 3];
    sim.update_positions(&net, &mut cars);
    while t < 20.0 {
        sim.tick(&net, DT, &mut cars);
        sim.update_positions(&net, &mut cars);
        for (k, v) in cars.iter().enumerate() {
            for w in cars.iter().skip(k + 1) {
                least = least.min(v.obb().separation(&w.obb()) as f32);
            }
            match (v.ease_for.is_some(), eased.contains(&v.id)) {
                (true, false) => {
                    eases += 1;
                    eased.push(v.id);
                }
                (false, true) => {
                    eased.retain(|&x| x != v.id);
                    if v.id == 1 && a_ease_end.is_none() {
                        a_ease_end = Some(t);
                    }
                    if v.id == 3 && c_ease_end.is_none() {
                        c_ease_end = Some(t);
                    }
                }
                _ => {}
            }
            through[(v.id - 1) as usize] |= v.state.lane >= 2;
        }
        t += DT;
    }
    println!("A's ease ended {a_ease_end:?}, C's {c_ease_end:?}, new eases {eases}, least {least:.2}, through {through:?}");
    let a_end = a_ease_end.expect("A's ease never ended");
    assert!(a_end < 0.5, "A's ease closing the ring ended at {a_end:.2} s");
    assert!(c_ease_end.is_some_and(|e| e > a_end), "C's ease ended before A's: {c_ease_end:?}");
    assert!(through.iter().all(|&x| x), "not everybody went through: {through:?}");
    assert_eq!(eases, 0, "a new ease was started");
    assert!(least > 0.0, "touched: {least:.2}");
}

/// Two cars standing refused for each other (`wait_for` both ways) at the crossing of
/// `crossing_ways`: A (id 1) on piece 0 with its front `a_short` m short of the crossing, B (id 2)
/// on piece 1 with its front `b_front` m past it. One frame to make the claims, then
/// `Simulation::negotiate`: who eased off.
fn negotiated(a_short: f32, b_front: f32) -> Option<(u64, u64)> {
    let net = crossing_ways();
    let mut sim = Simulation::new(&net, vec![]);
    let a = vehicle(1, VehicleClass::Car, vec![0, 2], 40.0 - a_short - 2.2, 0.0, 2.2, 0.9);
    let b = vehicle(2, VehicleClass::Car, vec![1, 3], 40.0 + b_front - 2.2, 0.0, 2.2, 0.9);
    let mut cars = vec![a, b];
    sim.update_positions(&net, &mut cars);
    sim.tick(&net, DT, &mut cars);
    let claim_of: Vec<Option<usize>> = (0..cars.len()).map(|i| sim.claims.iter().position(|(c, _)| c.vehicle == i)).collect();
    assert!(claim_of.iter().all(|c| c.is_some()), "both claim the junction");
    for k in 0..2 {
        let other = cars[1 - k].id;
        cars[k].state.speed = 0.0;
        cars[k].last.wait_for = Some(other);
        sim.claims[claim_of[k].unwrap()].0.speed = 0.0;
    }
    sim.negotiate(&mut cars, &claim_of, DT);
    cars.iter().find_map(|c| c.ease_for.map(|(o, _, _)| (c.id, o)))
}

/// [J11-6c] An ease its own end test would undo at once is not started: B's rear is already past
/// the crossing, the meeting place of the two (`Simulation::ease_holds`: the other past the first
/// meeting place beyond the eased car's front ends the ease), so neither eases off; with B still
/// short of it, one does.
#[test]
fn j11_6c_an_ease_that_would_end_at_once_is_not_started() {
    assert!(negotiated(3.5, -3.5).is_some(), "two short of the crossing, refused for each other: nobody eased off");
    // (B's front 8 m past the crossing: its rear 3.6 m past, the meeting place behind it)
    assert_eq!(negotiated(3.5, 8.0), None, "an ease was started that ends at once");
}
