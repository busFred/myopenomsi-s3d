//! Headless scenarios for round 6 area J (junctions): the bounded convergence stop (J2), the
//! ease pin (J3), the player's lit lane and lead (J4, J5), the box room at merge-only joins
//! (J6) and the sticky grant (J7). (J1, the far inside foe, is a resolver test in
//! `reservation.rs`.)

use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network, TrafficLightController};
#[allow(unused_imports)]
use omsi_traffic::sim::PLAYER_ID_BASE;
#[allow(unused_imports)]
use omsi_traffic::{Junctions, PlayerView, Simulation, Vehicle, VehicleClass, Why};

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

/// A lane of crossing object 9 (source 2).
fn piece(pts: &[(f64, f64)], path: u16, priority: f32, width: f32) -> omsi_sim::traffic::Lane {
    let mut l = lane(pts, LaneKind::Street, width);
    l.source = 2;
    l.key = Some(LaneKey { tile: (0, 0), id: 9, path });
    l.priority = priority;
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

fn finish(mut net: Network) -> Network {
    net.link(1.5);
    net.compute_conflicts();
    net
}

/// Run `secs`; `each` sees every frame. Returns the smallest gap between two bodies.
fn run(net: &Network, sim: &mut Simulation, vehicles: &mut Vec<Vehicle>, secs: f32, mut each: impl FnMut(f32, &mut Simulation, &mut [Vehicle])) -> f32 {
    let mut min_gap = f32::INFINITY;
    let mut t = 0.0;
    while t < secs {
        sim.tick(net, DT, vehicles);
        for &i in &sim.removed {
            vehicles.remove(i);
        }
        for (a, va) in vehicles.iter().enumerate() {
            for vb in vehicles.iter().skip(a + 1) {
                min_gap = min_gap.min(va.obb().separation(&vb.obb()) as f32);
            }
        }
        each(t, sim, vehicles.as_mut_slice());
        t += DT;
    }
    min_gap
}

fn front_x(v: &Vehicle) -> f32 {
    v.pos.x as f32 + v.state.front
}

/// J2: an eastbound way through object 9 (piece 1, x 0..60) crossed at x = 50 by a main road
/// (piece 5, priority 192) carrying a stream; a piece of the object (3) runs beside it 2.4 m
/// off over x `beside.0..beside.1`, no meeting place (a convergence, `Movement::converge`).
/// Lanes: 0 the approach, 1 the piece, 2 the exit, 3 the piece beside, 4/5/6 the main road.
fn converge_far(beside: (f64, f64)) -> Network {
    let mut net = Network::default();
    net.lanes.push(road(&[(-300.0, 0.0), (0.0, 0.0)], 1)); // 0
    net.lanes.push(piece(&[(0.0, 0.0), (60.0, 0.0)], 0, 128.0, 2.3)); // 1
    net.lanes.push(road(&[(60.0, 0.0), (400.0, 0.0)], 2)); // 2
    net.lanes.push(piece(&[(beside.0, 2.4), (beside.1, 2.4)], 1, 128.0, 2.3)); // 3
    net.lanes.push(road(&[(50.0, -400.0), (50.0, -15.0)], 5)); // 4
    net.lanes.push(piece(&[(50.0, -15.0), (50.0, 15.0)], 2, 192.0, 3.0)); // 5
    net.lanes.push(road(&[(50.0, 15.0), (50.0, 600.0)], 6)); // 6
    finish(net)
}

/// Where the eastbound car (front x) first stands held by the junction, and the least gap.
fn converge_far_stand(beside: (f64, f64)) -> (f32, f32) {
    let net = converge_far(beside);
    let js = Junctions::compile(&net);
    let (_, way) = js.movement(1).unwrap();
    assert!(!way.converge.is_empty(), "no convergence compiled: {:?}", way.foes);
    let mut sim = Simulation::new(&net, vec![]);
    // (standing 3 m short of the line, the stream coming: refused from the start)
    let mut cars = vec![car(1, 0, 295.0, 0.0)];
    for k in 0..12u64 {
        cars.push(car(10 + k, 4, 370.0 - 22.0 * k as f32, 12.0));
    }
    let mut stood_at: Option<f32> = None;
    let min_gap = run(&net, &mut sim, &mut cars, 25.0, |_, _, cars| {
        let me = cars.iter().find(|c| c.id == 1).unwrap();
        if me.state.speed < 0.1 && me.last.why == Why::Junction && stood_at.is_none() {
            stood_at = Some(front_x(me));
        }
    });
    (stood_at.expect("the car never stood for the main road"), min_gap)
}

#[test]
fn j2_a_refused_car_rolls_up_past_a_convergence_far_ahead_and_stops_at_once_for_a_near_one() {
    // the convergence 28-42 m on (its band from x 27): the car rolls up, its body wholly out of
    // the convergence (before it or past it) and short of the main road at 50 less its half
    // width (review-r6j: no body inside the band)
    let (at, gap) = converge_far_stand((28.0, 42.0));
    assert!((at > 20.0 && at <= 27.0) || (at - 4.0 >= 42.0 && at < 48.6), "stood with its front at x = {at:.1}");
    assert!(gap > 0.3, "touched: {gap:.2} m");
    // the convergence 5-20 m on, within CONVERGE_LOOK of the line: at once, at the line, as
    // before (review9 5229/5231, 10814/10625: near bands unchanged)
    let (at, _) = converge_far_stand((5.0, 20.0));
    assert!(at <= 0.0 && at > -1.5, "stood with its front at x = {at:.1}, not at the line");
}

/// J2: my way (piece 1, x 0..70, priority 64) is crossed at x ~ 43 by piece 5 (priority 192),
/// which first runs beside it 2.8 m to the left from x 0 to 40; a converging piece of the
/// object (3, no foe) runs 2.4 m to the right over x 26..36. A queue on piece 5 crawls at
/// `stream_v` from `stream_from` on: where my car, refused for it, first stands (front x).
fn waited_beside(stream_from: f32, stream_v: f32) -> f32 {
    let mut net = Network::default();
    net.lanes.push(road(&[(-300.0, 0.0), (0.0, 0.0)], 1)); // 0
    net.lanes.push(piece(&[(0.0, 0.0), (70.0, 0.0)], 0, 64.0, 2.3)); // 1
    net.lanes.push(road(&[(70.0, 0.0), (400.0, 0.0)], 2)); // 2
    net.lanes.push(piece(&[(26.0, -2.4), (36.0, -2.4)], 1, 128.0, 2.3)); // 3
    net.lanes.push(road(&[(-300.0, 2.8), (0.0, 2.8)], 3)); // 4
    net.lanes.push(piece(&[(0.0, 2.8), (40.0, 2.8), (55.0, -15.0)], 2, 192.0, 2.3)); // 5
    net.lanes.push(road(&[(55.0, -15.0), (55.0, -300.0)], 4)); // 6
    let net = finish(net);
    let js = Junctions::compile(&net);
    assert!(!js.movement(1).unwrap().1.converge.is_empty(), "no convergence compiled");
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![car(1, 0, 295.0, 0.0)];
    for k in 0..4u64 {
        let mut s = car(10 + k, 5, stream_from - 9.0 * k as f32, stream_v);
        s.state.max_speed_kmh = stream_v * 3.6;
        cars.push(s);
    }
    let mut stood = f32::NAN;
    run(&net, &mut sim, &mut cars, 12.0, |_, _, cars| {
        let me = cars.iter().find(|c| c.id == 1).unwrap();
        if me.state.speed < 0.1 && me.last.why == Why::Junction && stood.is_nan() {
            stood = front_x(me);
        }
    });
    stood
}

#[test]
fn j2_the_waited_vehicle_standing_in_a_far_convergence_stops_the_car_at_its_line() {
    // the car waited for crawls at 0.3 m/s with its body in the stretch beside mine where the
    // convergence lies (x 26..36, 26 m past the line): the car stops at once, at its line, not
    // rolling up beside it (review9: refused cars rolled into the very car they waited for)
    let at = waited_beside(36.0, 0.3);
    assert!(at <= 0.0 && at > -1.5, "stood with its front at x = {at:.1}, not at the line");
    // the queue rolling at 2 m/s further on: the car rolls up, its body out of the convergence
    let at = waited_beside(30.0, 2.0);
    assert!(at > 5.0 && at <= 25.0, "stood with its front at x = {at:.1}");
}
