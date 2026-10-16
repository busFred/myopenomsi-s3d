//! Headless scenarios: a small road network, a few vehicles, a fixed step and a seed, and
//! assertions on what happened: nobody touched, nobody stood for ever, nobody braked
//! like an emergency. These are the regression gate for the traffic.

use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network, TrafficLightController};
use omsi_traffic::{PlayerView, Simulation, Vehicle, VehicleClass, Why};

const DT: f32 = 1.0 / 30.0;

/// A spline lane (source 1) between two points.
fn road(from: (f64, f64), to: (f64, f64), id: i64) -> omsi_sim::traffic::Lane {
    let mut l = LaneBuilder::polyline(vec![DVec3::new(from.0, from.1, 0.0), DVec3::new(to.0, to.1, 0.0)], LaneKind::Street, 3.5);
    l.source = 1;
    l.key = Some(LaneKey { tile: (0, 0), id, path: 0 });
    l
}

/// A lane of crossing object `obj` (source 2).
fn piece(from: (f64, f64), to: (f64, f64), obj: i64, path: u16, priority: f32) -> omsi_sim::traffic::Lane {
    let mut l = LaneBuilder::polyline(vec![DVec3::new(from.0, from.1, 0.0), DVec3::new(to.0, to.1, 0.0)], LaneKind::Street, 3.5);
    l.source = 2;
    l.key = Some(LaneKey { tile: (0, 0), id: obj, path });
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

/// Run `secs` and keep the worst of everything.
struct Run {
    min_gap: f32,
    hardest_brake: f32,
    longest_stand: f32,
    removed: usize,
}

fn run(net: &Network, sim: &mut Simulation, vehicles: &mut Vec<Vehicle>, secs: f32, mut each: impl FnMut(f32, &Simulation, &mut [Vehicle])) -> Run {
    let mut r = Run { min_gap: f32::INFINITY, hardest_brake: 0.0, longest_stand: 0.0, removed: 0 };
    let mut t = 0.0;
    while t < secs {
        sim.tick(net, DT, vehicles);
        for &i in &sim.removed {
            vehicles.remove(i);
            r.removed += 1;
        }
        for v in vehicles.iter() {
            r.hardest_brake = r.hardest_brake.max(-v.state.acc);
            r.longest_stand = r.longest_stand.max(v.stopped);
        }
        for (a, va) in vehicles.iter().enumerate() {
            for vb in vehicles.iter().skip(a + 1) {
                let d = va.obb().separation(&vb.obb()) as f32;
                r.min_gap = r.min_gap.min(d);
            }
        }
        each(t, sim, vehicles.as_mut_slice());
        t += DT;
    }
    r
}

fn straight_road() -> Network {
    let mut net = Network::default();
    net.lanes.push(road((0.0, 0.0), (300.0, 0.0), 1));
    net.lanes.push(road((300.0, 0.0), (2000.0, 0.0), 2));
    net.link(1.5);
    net.compute_conflicts();
    net
}

#[test]
fn a_follower_keeps_its_distance_and_a_queue_moves_off_in_a_wave() {
    let net = straight_road();
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars: Vec<Vehicle> = (0..5).map(|k| car(k + 1, 0, 100.0 - 8.0 * k as f32, 0.0)).collect();
    // the leader is held for ten seconds (a stop of its own), then is free to go
    cars[0].external.stop_at = Some(2.1);
    let mut frames = 0;
    let r = run(&net, &mut sim, &mut cars, 40.0, |t, _, cars| {
        if t > 10.0 {
            cars[0].external.stop_at = None;
        }
        frames += 1;
        if std::env::var_os("PROBE").is_some() && frames % 15 == 0 {
            let line: Vec<String> = cars.iter().map(|c| format!("[{} s={:.1} v={:.1} lead={:?} why={:?} stop={:?}]", c.id, c.state.s, c.state.speed, c.last.lead.map(|l| (l.gap * 10.0).round() / 10.0), c.last.why, c.last.stop)).collect();
            eprintln!("t={:.1} {}", t, line.join(" "));
        }
    });
    assert!(r.min_gap > 0.3, "cars touched: nearest {:.2} m", r.min_gap);
    assert!(r.hardest_brake < 4.0, "braked {:.1} m/s²", r.hardest_brake);
    assert!(r.removed == 0);
    // all moving at the end, and in order
    for w in cars.windows(2) {
        assert!(w[0].state.s > w[1].state.s);
        assert!(w[1].state.speed > 5.0, "a follower never got going");
    }
}

#[test]
fn cars_stop_at_a_red_light_and_go_on_green() {
    let mut net = Network::default();
    net.lanes.push(road((0.0, 0.0), (200.0, 0.0), 1));
    let mut p = piece((200.0, 0.0), (230.0, 0.0), 9, 0, 128.0);
    p.traffic_light = Some((0, 0));
    net.lanes.push(p);
    net.lanes.push(road((230.0, 0.0), (500.0, 0.0), 2));
    net.link(1.5);
    net.compute_conflicts();
    // red for 20 s, green for 20 s
    let ctl = TrafficLightController::new(vec![vec![(0, 20.0), (6, 20.0)]], 40.0);
    let mut sim = Simulation::new(&net, vec![ctl]);
    let mut cars = vec![car(1, 0, 120.0, 13.0), car(2, 0, 90.0, 13.0)];
    let mut stood_at_red = false;
    let mut crossed_on_red = false;
    let r = run(&net, &mut sim, &mut cars, 45.0, |_, sim, cars| {
        let red = sim.lights[0].state(0) < 6;
        for c in cars {
            if c.state.lane == 0 && c.state.speed < 0.3 && c.last.why == Why::Light {
                stood_at_red = true;
            }
            if red && c.state.lane == 1 && c.state.s < 3.0 && c.state.speed > 2.0 && !c.amber {
                crossed_on_red = true;
            }
        }
    });
    assert!(stood_at_red, "nobody stopped at the red light");
    assert!(!crossed_on_red, "a car drove through the red light");
    assert!(r.hardest_brake < 4.5, "braked {:.1} m/s² for the light", r.hardest_brake);
    assert!(r.min_gap > 0.3);
    assert!(cars.iter().all(|c| c.state.lane == 2 || c.state.s > 200.0), "both went on after the green");
}

/// A main road along x with a side road from the south, meeting in a crossing object.
fn priority_junction() -> Network {
    let mut net = Network::default();
    net.lanes.push(road((-200.0, 0.0), (-15.0, 0.0), 1)); // 0 west approach
    net.lanes.push(piece((-15.0, 0.0), (15.0, 0.0), 9, 0, 192.0)); // 1 main through
    net.lanes.push(road((15.0, 0.0), (1500.0, 0.0), 2)); // 2 east exit
    net.lanes.push(road((0.0, -200.0), (0.0, -15.0), 3)); // 3 south approach
    net.lanes.push(piece((0.0, -15.0), (0.0, 15.0), 9, 1, 64.0)); // 4 side through
    net.lanes.push(road((0.0, 15.0), (0.0, 1500.0), 4)); // 5 north exit
    net.link(1.5);
    net.compute_conflicts();
    net
}

#[test]
fn the_side_road_gives_way_to_a_stream_on_the_main_road_and_gets_through_a_gap() {
    let net = priority_junction();
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = Vec::new();
    // a stream on the main road, 30 m apart, then a gap
    for k in 0..4 {
        cars.push(car(10 + k, 0, 150.0 - 30.0 * k as f32, 13.0));
    }
    cars.push(car(1, 3, 150.0, 10.0));
    let mut waited = false;
    let r = run(&net, &mut sim, &mut cars, 40.0, |_, _, cars| {
        if let Some(c) = cars.iter().find(|c| c.id == 1) {
            if c.state.speed < 0.3 && c.last.why == Why::Junction {
                waited = true;
            }
        }
    });
    assert!(waited, "the side road car never gave way");
    assert!(r.min_gap > 0.5, "vehicles met in the junction: {:.2} m", r.min_gap);
    let side = cars.iter().find(|c| c.id == 1).expect("the side road car is still there");
    assert_eq!(side.state.lane, 5, "the side road car got across after the stream");
    assert!(r.hardest_brake < 4.5, "braked {:.1} m/s²", r.hardest_brake);
}

#[test]
fn right_before_left_lets_exactly_one_go_first_and_nobody_waits_for_ever() {
    let mut net = Network::default();
    net.lanes.push(road((-200.0, 0.0), (-15.0, 0.0), 1));
    net.lanes.push(piece((-15.0, 0.0), (15.0, 0.0), 9, 0, 128.0));
    net.lanes.push(road((15.0, 0.0), (1500.0, 0.0), 2));
    net.lanes.push(road((0.0, -200.0), (0.0, -15.0), 3));
    net.lanes.push(piece((0.0, -15.0), (0.0, 15.0), 9, 1, 128.0));
    net.lanes.push(road((0.0, 15.0), (0.0, 1500.0), 4));
    net.link(1.5);
    net.compute_conflicts();
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![car(1, 0, 160.0, 12.0), car(2, 3, 160.0, 12.0)];
    let r = run(&net, &mut sim, &mut cars, 30.0, |_, _, _| {});
    assert!(r.min_gap > 0.5, "met in the junction: {:.2} m", r.min_gap);
    assert!(r.longest_stand < 15.0, "someone stood {:.0} s", r.longest_stand);
    assert!(cars.iter().all(|c| c.state.lane == 2 || c.state.lane == 5), "both got through");
}

#[test]
fn a_car_stops_behind_the_standing_player_and_does_not_drive_through_it() {
    let net = straight_road();
    let mut sim = Simulation::new(&net, vec![]);
    sim.players.push(PlayerView { id: 7, pos: DVec3::new(200.0, 0.0, 0.0), heading: 90.0, half_len: 6.0, half_width: 1.25, speed: 0.0, still: 100.0, blinker: 0, signalling: 0.0, priority: false });
    let mut cars = vec![car(1, 0, 100.0, 13.0)];
    let r = run(&net, &mut sim, &mut cars, 12.0, |_, _, _| {});
    let c = &cars[0];
    assert!(r.hardest_brake < 4.0, "braked {:.1} m/s² for the bus", r.hardest_brake);
    // either standing behind the bus (its rear is at x = 194) or past it on a pass
    let behind = c.state.s + c.state.front < 194.0 - 0.5;
    assert!(behind || c.state.lateral.abs() > 1.5, "car at {:.1} with lateral {:.2}", c.state.s, c.state.lateral);
}

#[test]
fn the_network_may_grow_while_the_traffic_runs() {
    let mut net = straight_road();
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![car(1, 0, 100.0, 13.0)];
    run(&net, &mut sim, &mut cars, 5.0, |_, _, _| {});
    let first = net.lanes.len();
    let mut p = piece((600.0, 0.0), (630.0, 0.0), 9, 0, 128.0);
    p.key = Some(LaneKey { tile: (1, 0), id: 9, path: 0 });
    net.extend(vec![p], 1.5);
    sim.network_grew(&net, first);
    assert!(sim.junctions.is_inside(first));
    let r = run(&net, &mut sim, &mut cars, 40.0, |_, _, _| {});
    assert!(r.hardest_brake < 4.0);
}
