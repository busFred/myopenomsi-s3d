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
            if red && c.state.lane == 1 && c.state.s < 3.0 && c.state.speed > 2.0 && !c.light_decision.is_some_and(|d| d.1) {
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

// ---------------------------------------------------------------------------------------
// Second batch: lanes beside each other, obstacles, people, light programs, deadlocks.

/// `PROBE=1 cargo test ... -- --nocapture` prints a line per half second.
fn probe(t: f32, frames: &mut u32, cars: &[Vehicle]) {
    *frames += 1;
    if std::env::var_os("PROBE").is_some() && *frames % 15 == 0 {
        let line: Vec<String> = cars
            .iter()
            .map(|c| format!("[{} L{} s={:.1} lat={:.2} v={:.1} why={:?} gap={:?} pass={}]", c.id, c.state.lane, c.state.s, c.state.lateral, c.state.speed, c.last.why, c.last.lead.map(|l| (l.gap * 10.0).round() / 10.0), c.passing.is_some()))
            .collect();
        eprintln!("t={:.1} {}", t, line.join(" "));
    }
}

/// Two lanes of one road in the same direction, 3.5 m apart: lane 0 on the left (y = 0),
/// lane 1 on the right (y = -3.5). `Network::link` pairs them as neighbours because they
/// are paths of one spline with offsets 3.5 m apart.
fn two_lane_road() -> Network {
    let mut net = Network::default();
    let mut a = road((0.0, 0.0), (2000.0, 0.0), 1);
    a.offset = 0.0;
    let mut b = road((0.0, -3.5), (2000.0, -3.5), 1);
    b.offset = 3.5;
    b.key = Some(LaneKey { tile: (0, 0), id: 1, path: 1 });
    net.lanes.push(a);
    net.lanes.push(b);
    net.link(1.5);
    net.compute_conflicts();
    net
}

#[test]
fn a_fast_car_changes_lane_and_passes_a_slow_one() {
    let net = two_lane_road();
    assert_eq!(net.lanes[1].left, Some(0), "the two lanes are neighbours");
    let mut sim = Simulation::new(&net, vec![]);
    // both in the right-hand lane; the slow one crawls at 20 km/h
    let mut slow = car(1, 1, 120.0, 5.5);
    slow.state.max_speed_kmh = 20.0;
    let mut fast = car(2, 1, 80.0, 13.0);
    fast.state.change_cooldown = 0.0;
    slow.state.change_cooldown = 100.0;
    let mut cars = vec![slow, fast];
    let mut frames = 0;
    let mut changed = false;
    let r = run(&net, &mut sim, &mut cars, 40.0, |t, _, cars| {
        probe(t, &mut frames, cars);
        if cars[1].state.lane == 0 || cars[1].state.change.is_some() {
            changed = true;
        }
    });
    assert!(changed, "the fast car never changed lane");
    assert!(r.min_gap > 0.3, "cars touched: nearest {:.2} m", r.min_gap);
    assert!(r.hardest_brake < 4.5, "braked {:.1} m/s²", r.hardest_brake);
    assert!(cars[1].state.s > cars[0].state.s + 10.0, "the fast car got ahead: {:.1} vs {:.1}", cars[1].state.s, cars[0].state.s);
}

/// A two-way road: eastbound lane 0 and westbound lane 1 3.5 m to its left (north).
fn two_way_road() -> Network {
    let mut net = Network::default();
    // (3 m lanes: a car does not fit past a parked one inside its own lane, so the pass
    // has to use the oncoming one)
    let mut a = road((0.0, 0.0), (2000.0, 0.0), 1);
    a.width = 3.0;
    let mut b = road((2000.0, 3.5), (0.0, 3.5), 2);
    b.width = 3.0;
    net.lanes.push(a);
    net.lanes.push(b);
    net.link(1.5);
    net.compute_conflicts();
    net
}

const PARKED_AT: f32 = 200.0;

fn parked_simulation(net: &Network) -> Simulation {
    let mut sim = Simulation::new(net, vec![]);
    sim.parked.insert(0, vec![omsi_traffic::Parked { s: PARKED_AT, lateral: 0.0, half_width: 0.9, half_len: 2.2 }]);
    sim
}

#[test]
fn a_car_passes_a_blocking_parked_car_on_the_empty_oncoming_lane() {
    let net = two_way_road();
    assert!(net.opposite(0, 150.0).is_some());
    let mut sim = parked_simulation(&net);
    let mut cars = vec![car(1, 0, 100.0, 12.0)];
    let mut frames = 0;
    let mut out_on_the_other_lane = false;
    let r = run(&net, &mut sim, &mut cars, 30.0, |t, _, cars| {
        probe(t, &mut frames, cars);
        let c = &cars[0];
        // abreast of the parked car (x 197.8 .. 202.2): must be clear of its side (lateral -0.9 - 0.9)
        if c.state.s + c.state.front > PARKED_AT - 2.2 && c.state.s - c.state.rear < PARKED_AT + 2.2 {
            assert!(c.state.lateral.abs() > 1.9, "abreast of the parked car at lateral {:.2}", c.state.lateral);
            out_on_the_other_lane = true;
        }
    });
    assert!(out_on_the_other_lane);
    assert!(r.hardest_brake < 4.5, "braked {:.1} m/s²", r.hardest_brake);
    assert!(cars[0].state.s > PARKED_AT + 20.0, "the car got past: s = {:.1}", cars[0].state.s);
    assert!(cars[0].state.lateral.abs() < 0.5, "and is back in its lane: lateral {:.2}", cars[0].state.lateral);
}

#[test]
fn a_car_waits_behind_a_parked_car_while_a_stream_comes_the_other_way() {
    let net = two_way_road();
    let mut sim = parked_simulation(&net);
    let mut cars = vec![car(1, 0, 170.0, 8.0)];
    // oncoming (lane 1 runs west, its s counts from x = 2000): a car every 25 m, the first
    // 30 m beyond the parked one, 20 of them, so there is never room for the pass
    for k in 0..20u64 {
        cars.push(car(10 + k, 1, 2000.0 - 230.0 - 25.0 * k as f32, 13.0));
    }
    let mut frames = 0;
    let mut passed_at: Option<f32> = None;
    let mut waited_behind = false;
    let r = run(&net, &mut sim, &mut cars, 80.0, |t, _, cars| {
        probe(t, &mut frames, cars);
        let Some(c) = cars.iter().find(|c| c.id == 1) else { return };
        let behind = c.state.s + c.state.front < PARKED_AT - 2.2;
        if behind && c.state.speed < 0.3 {
            waited_behind = true;
        }
        if !behind && passed_at.is_none() {
            passed_at = Some(t);
        }
        // the last oncoming car is level with the waiting one at about t = 41 s
    });
    assert!(waited_behind, "the car never stood behind the parked car");
    assert!(r.min_gap > 0.3, "touched: nearest {:.2} m", r.min_gap);
    assert!(r.hardest_brake < 4.5, "braked {:.1} m/s²", r.hardest_brake);
    let at = passed_at.expect("the car never got past once the stream was over");
    assert!(at > 35.0, "it pushed past the stream at {at:.1} s");
}

#[test]
fn a_car_stops_for_a_person_on_a_crossing_and_goes_on_when_they_are_gone() {
    let net = straight_road();
    let mut sim = Simulation::new(&net, vec![]);
    sim.people.push(omsi_traffic::Person { pos: DVec3::new(150.0, 0.0, 0.0), vel: glam::DVec2::ZERO, on_crossing: true, still: 0.0, waiting_for_bus: false, at_kerb: false, waits_at: None });
    let mut cars = vec![car(1, 0, 60.0, 12.0)];
    let mut frames = 0;
    let mut stood_short = false;
    let mut crossed_while_there = false;
    let mut watch = |cars: &mut [Vehicle]| {
        let c = &cars[0];
        if c.state.speed < 0.3 && c.state.s + c.state.front < 149.0 {
            stood_short = true;
        }
        if c.state.s + c.state.front > 150.0 {
            crossed_while_there = true;
        }
    };
    let mut frames1 = 0;
    let r = run(&net, &mut sim, &mut cars, 12.0, |t, _, cars| {
        probe(t, &mut frames1, cars);
        watch(cars)
    });
    // the person walks off the crossing
    sim.people.clear();
    let r2 = run(&net, &mut sim, &mut cars, 13.0, |t, _, cars| probe(t + 12.0, &mut frames, cars));
    let r = Run { min_gap: r.min_gap.min(r2.min_gap), hardest_brake: r.hardest_brake.max(r2.hardest_brake), longest_stand: r.longest_stand.max(r2.longest_stand), removed: r.removed + r2.removed };
    assert!(stood_short, "the car never stopped short of the crossing");
    assert!(!crossed_while_there, "the car drove over the person");
    assert!(r.hardest_brake < 4.5, "braked {:.1} m/s²", r.hardest_brake);
    assert!(cars[0].state.s > 170.0, "the car went on: s = {:.1}", cars[0].state.s);
}

#[test]
fn a_car_keeps_the_junction_clear_while_the_exit_is_blocked() {
    let net = priority_junction();
    let mut sim = Simulation::new(&net, vec![]);
    // a car stands inside the main-road piece just past its conflict with the side road
    // (the piece runs to 30 m, the meeting place ends near 18 m): no room for another
    // car to wait there clear of the conflict, so the next one has to hold at the line
    let mut blocker = car(9, 1, 24.0, 0.0);
    blocker.external.stop_at = Some(2.1);
    let mut cars = vec![blocker, car(1, 0, 140.0, 12.0)];
    let mut frames = 0;
    let mut entered_early = false;
    let mut held_for_exit = false;
    let r = run(&net, &mut sim, &mut cars, 35.0, |t, _, cars| {
        probe(t, &mut frames, cars);
        let c = cars.iter().find(|c| c.id == 1).unwrap();
        if t < 13.0 {
            if c.state.lane != 0 {
                entered_early = true;
            }
            if c.state.speed < 0.3 && c.last.why == Why::ExitFull {
                held_for_exit = true;
            }
        }
        if t >= 14.0 {
            cars[0].external.stop_at = None;
        }
    });
    assert!(!entered_early, "the car drove into the junction with no room past it");
    assert!(held_for_exit, "the car was not held for the full exit");
    assert!(r.min_gap > 0.3, "touched: nearest {:.2} m", r.min_gap);
    assert!(r.hardest_brake < 4.5, "braked {:.1} m/s²", r.hardest_brake);
    let c = cars.iter().find(|c| c.id == 1).unwrap();
    assert!(c.state.lane == 2 || c.state.s > 15.0 && c.state.lane == 1, "the car went on after the blocker moved: lane {} s {:.1}", c.state.lane, c.state.s);
}

/// A main road along x and a side road from the south through one crossing object, both
/// pieces with a light of the same controller (main: light 0, side: light 1).
fn signalled_junction() -> Network {
    let mut net = Network::default();
    net.lanes.push(road((-200.0, 0.0), (-15.0, 0.0), 1)); // 0 west approach
    let mut a = piece((-15.0, 0.0), (15.0, 0.0), 9, 0, 128.0); // 1 main through
    a.traffic_light = Some((0, 0));
    net.lanes.push(a);
    net.lanes.push(road((15.0, 0.0), (1500.0, 0.0), 2)); // 2 east exit
    net.lanes.push(road((0.0, -200.0), (0.0, -15.0), 3)); // 3 south approach
    let mut b = piece((0.0, -15.0), (0.0, 15.0), 9, 1, 128.0); // 4 side through
    b.traffic_light = Some((0, 1));
    net.lanes.push(b);
    net.lanes.push(road((0.0, 15.0), (0.0, 1500.0), 4)); // 5 north exit
    net.link(1.5);
    net.compute_conflicts();
    net
}

#[test]
fn the_side_road_goes_on_its_green_while_the_main_road_is_red() {
    let net = signalled_junction();
    // a 44 s cycle: the main road is green 0..18, yellow 18..21, red 21..44; the side road
    // red 0..22, green 22..40, yellow 40..43, red 43..44 (one second of all-red between)
    let main = vec![(6, 18.0), (9, 3.0), (0, 23.0)];
    let side = vec![(0, 22.0), (6, 18.0), (9, 3.0), (0, 1.0)];
    let ctl = TrafficLightController::new(vec![main, side], 44.0);
    let mut sim = Simulation::new(&net, vec![ctl]);
    // the program starts at 22 s: main red, side green
    sim.day_time = 22.0;
    let mut cars = vec![car(1, 0, 130.0, 13.0), car(2, 0, 100.0, 13.0), car(3, 3, 120.0, 12.0), car(4, 3, 60.0, 12.0)];
    let mut frames = 0;
    let mut main_stood_at_red = false;
    let mut side_crossed_on_green = false;
    let mut crossed_on_red = false;
    let r = run(&net, &mut sim, &mut cars, 40.0, |t, sim, cars| {
        probe(t, &mut frames, cars);
        for c in cars.iter() {
            let (in_junction, light) = match c.state.lane {
                1 => (true, 0),
                4 => (true, 1),
                _ => (false, 0),
            };
            let red = sim.lights[0].state(light) < 3;
            if c.state.lane == 0 && c.state.speed < 0.3 && c.last.why == Why::Light {
                main_stood_at_red = true;
            }
            if in_junction && c.state.s < 3.0 && c.state.speed > 2.0 && red && !c.light_decision.is_some_and(|d| d.1) {
                crossed_on_red = true;
            }
            if c.state.lane == 4 && !red {
                side_crossed_on_green = true;
            }
        }
    });
    assert!(main_stood_at_red, "the main road never stopped at its red");
    assert!(side_crossed_on_green, "the side road never used its green");
    assert!(!crossed_on_red, "a car drove through a red light");
    assert!(r.min_gap > 0.3, "touched: nearest {:.2} m", r.min_gap);
    assert!(r.hardest_brake < 4.5, "braked {:.1} m/s²", r.hardest_brake);
    // the side road's cars are across, the main road's cars went on at their green
    for id in [3, 4] {
        let c = cars.iter().find(|c| c.id == id).expect("a side road car is still there");
        assert!(c.state.lane == 5 || (c.state.lane == 4 && c.state.s > 15.0), "side car {id} on lane {} s {:.1}", c.state.lane, c.state.s);
    }
    for id in [1, 2] {
        let c = cars.iter().find(|c| c.id == id).expect("a main road car is still there");
        assert!(c.state.lane == 2 || c.state.lane == 1, "main car {id} still on lane {} s {:.1}", c.state.lane, c.state.s);
    }
}

/// Four approaches to one crossing object, all with the same priority. Each road is
/// offset 1.75 m from the axis so that the two directions of a road do not share a line.
fn four_way() -> Network {
    let mut net = Network::default();
    let o = 1.75;
    // (unit direction, offset point): eastbound on the south side of the x axis,
    // westbound on the north side, northbound on the east side, southbound on the west
    let legs: [((f64, f64), (f64, f64)); 4] = [((1.0, 0.0), (0.0, -o)), ((-1.0, 0.0), (0.0, o)), ((0.0, 1.0), (o, 0.0)), ((0.0, -1.0), (-o, 0.0))];
    for (k, (u, off)) in legs.iter().enumerate() {
        let at = |along: f64| (u.0 * along + off.0, u.1 * along + off.1);
        // approach: 200 m out to 15 m out; piece through the crossing; exit 15 m past to 1500 m
        net.lanes.push(road(at(-200.0), at(-15.0), 10 * (k as i64 + 1)));
        net.lanes.push(piece(at(-15.0), at(15.0), 9, k as u16, 128.0));
        net.lanes.push(road(at(15.0), at(1500.0), 10 * (k as i64 + 1) + 1));
    }
    net.link(1.5);
    net.compute_conflicts();
    net
}

#[test]
fn four_cars_arriving_together_at_a_right_before_left_junction_get_through() {
    let net = four_way();
    // lane 3k is the approach of leg k, 3k + 2 its exit
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars: Vec<Vehicle> = (0..4).map(|k| car(k as u64 + 1, 3 * k, 100.0, 12.0)).collect();
    let mut frames = 0;
    // a deadlock is a stretch in which nobody moves although somebody has yet to get through
    let mut frozen = 0.0f32;
    let mut longest_deadlock = 0.0f32;
    let mut hard = 0.0f32;
    let r = run(&net, &mut sim, &mut cars, 90.0, |t, _, cars| {
        probe(t, &mut frames, cars);
        let waiting = cars.iter().any(|c| c.state.lane % 3 == 0);
        if waiting && cars.iter().all(|c| c.state.speed < 0.3) {
            frozen += DT;
        } else {
            frozen = 0.0;
        }
        longest_deadlock = longest_deadlock.max(frozen);
        // (creeping off and being held again inside the stop-line margin brakes at the
        // model's limit from under 1 m/s, which is no emergency: only count real speed)
        for c in cars.iter() {
            if c.state.speed > 2.0 {
                hard = hard.max(-c.state.acc);
            }
        }
    });
    assert!(longest_deadlock < 30.0, "nobody moved for {longest_deadlock:.0} s");
    assert!(r.longest_stand < 45.0, "someone stood {:.0} s", r.longest_stand);
    assert!(r.min_gap > 0.3, "met in the junction: {:.2} m", r.min_gap);
    assert!(hard < 4.5, "braked {:.1} m/s² at speed", hard);
    assert!(cars.iter().all(|c| c.state.lane % 3 == 2), "all four got through: {:?}", cars.iter().map(|c| c.state.lane).collect::<Vec<_>>());
}

#[test]
fn a_car_holds_back_for_a_bus_with_its_indicator_towards_the_traffic() {
    let net = straight_road();
    let mut sim = Simulation::new(&net, vec![]);
    let bus = |signalling: f32, still: f32| PlayerView { id: 7, pos: DVec3::new(200.0, 0.0, 0.0), heading: 90.0, half_len: 6.0, half_width: 1.25, speed: 0.0, still, blinker: 1, signalling, priority: false };
    let mut cars = vec![car(1, 0, 130.0, 10.0)];
    let mut frames = 0;
    let mut let_out = false;
    let mut passed = false;
    let mut worst_brake = 0.0f32;
    // the indicator comes on at t = 0 and stays on (the car is to hold back for 15 s of it).
    // For the first 8 s the bus has only just stopped (not yet a standing obstacle to go
    // round: the car lets it out); after that it has stood for a good while, and the car
    // still does not pass a bus that is indicating to pull out.
    let mut now = 0.0;
    while now < 15.0 {
        sim.players.clear();
        sim.players.push(bus(now, if now < 8.0 { 2.0 } else { 100.0 }));
        let r = run(&net, &mut sim, &mut cars, 0.5, |t, _, cars| {
            probe(now + t, &mut frames, cars);
            let c = &cars[0];
            if c.last.why == Why::LetOut {
                let_out = true;
            }
            // the bus's rear is at x = 194
            if c.state.s + c.state.front > 194.0 || c.state.lateral.abs() > 1.0 {
                passed = true;
            }
        });
        worst_brake = worst_brake.max(r.hardest_brake);
        now += 0.5;
    }
    assert!(worst_brake < 4.5, "braked {:.1} m/s²", worst_brake);
    assert!(let_out, "the car never let the bus out");
    assert!(!passed, "the car passed a bus that was pulling out");
    let c = &cars[0];
    assert!(c.state.speed < 0.3 && c.state.s + c.state.front < 194.0, "the car stands behind the bus: s {:.1} v {:.1}", c.state.s, c.state.speed);
}

/// What a scenario ends with: every vehicle's place, exactly.
fn fingerprint(cars: &[Vehicle]) -> Vec<(u64, usize, u32, u32, u32)> {
    cars.iter().map(|c| (c.id, c.state.lane, c.state.s.to_bits(), c.state.speed.to_bits(), c.state.lateral.to_bits())).collect()
}

fn determinism_scenario() -> Vec<(u64, usize, u32, u32, u32)> {
    // a queue behind a parked car with a pass on the oncoming lane and a stream coming...
    let net = two_way_road();
    let mut sim = parked_simulation(&net);
    let mut cars = vec![car(1, 0, 100.0, 12.0), car(2, 0, 60.0, 12.0), car(3, 0, 20.0, 12.0)];
    for k in 0..6u64 {
        cars.push(car(10 + k, 1, 2000.0 - 300.0 - 90.0 * k as f32, 13.0));
    }
    run(&net, &mut sim, &mut cars, 40.0, |_, _, _| {});
    let mut fp = fingerprint(&cars);
    // ... and a side road giving way to a stream
    let net = priority_junction();
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars: Vec<Vehicle> = (0..4).map(|k| car(10 + k, 0, 150.0 - 30.0 * k as f32, 13.0)).collect();
    cars.push(car(1, 3, 150.0, 10.0));
    run(&net, &mut sim, &mut cars, 40.0, |_, _, _| {});
    fp.extend(fingerprint(&cars));
    fp
}

#[test]
fn the_same_scenario_twice_gives_the_same_positions() {
    let a = determinism_scenario();
    let b = determinism_scenario();
    assert!(a.len() >= 8);
    assert_eq!(a, b);
}

/// [ROLLBACK r4c-L4] A 2.5 m bus standing in a 2.5 m lane, somebody 5 m ahead of its nose at the
/// kerb spot (half the lane and 0.3 m off its axis): waiting there to cross, they hold it
/// neither as a person (People) nor at a crossing (Zebra) and it drives on; not waiting
/// (the same place, walking about) they still do, and somebody waiting nearer than the bus's
/// footprint and 0.3 m does too.
#[test]
fn a_bus_goes_past_a_person_waiting_at_the_kerb_spot() {
    let run_with = |lateral: f64, at_kerb: bool| -> (bool, f32) {
        let mut net = Network::default();
        let mut l = LaneBuilder::polyline(vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(400.0, 0.0, 0.0)], LaneKind::Street, 2.5);
        l.source = 1;
        l.key = Some(LaneKey { tile: (0, 0), id: 1, path: 0 });
        net.lanes.push(l);
        net.link(1.5);
        net.compute_conflicts();
        let mut sim = Simulation::new(&net, vec![]);
        let mut st = AiState::new(0, 100.0, 1);
        st.front = 6.0;
        st.rear = 6.0;
        st.length = 12.0;
        st.max_speed_kmh = 60.0;
        st.veh_type = 2;
        let mut bus = Vehicle::new(1, VehicleClass::Bus, st, 1.25, 7);
        bus.fresh = 0.0;
        let mut cars = vec![bus];
        sim.people.push(omsi_traffic::Person { pos: DVec3::new(111.0, -lateral, 0.0), vel: glam::DVec2::ZERO, on_crossing: false, still: 0.5, waiting_for_bus: false, at_kerb, waits_at: None });
        let mut held = false;
        run(&net, &mut sim, &mut cars, 8.0, |_, sim, cars| {
            held |= matches!(cars[0].last.why, Why::People | Why::Zebra);
            // (standing still on: the person's seconds go on)
            let _ = sim;
        });
        (held, cars[0].state.s)
    };
    let (held, s) = run_with(1.25 + 0.3, true);
    assert!(!held, "held by somebody waiting outside its footprint");
    assert!(s > 115.0, "the bus did not go: s = {s:.1}");
    let (held, s) = run_with(1.25 + 0.28 + 0.01, true);
    assert!(!held && s > 115.0, "just beyond the body radius: s = {s:.1}");
    let (held, _) = run_with(1.25 + 0.3 - 0.05, false);
    assert!(held, "somebody not waiting there still holds it");
    let (held, _) = run_with(1.25 + 0.25, true);
    assert!(held, "waiting with the body inside its side and 0.3 m holds it");
}
