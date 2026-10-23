//! Adversarial scenarios (review): situations the regression gate does not cover. Each
//! test asserts what a safe, sane traffic should do; a failure is a demonstrated problem.
//! `cargo test -p omsi-traffic --test adversarial -- --nocapture` prints the numbers.

use glam::{DVec2, DVec3};
use omsi_sim::collision::Obb;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network, TrafficLightController};
use omsi_traffic::{DriverParams, Person, PlayerView, Simulation, Vehicle, VehicleClass, Why};

const DT: f32 = 1.0 / 30.0;

fn road(from: (f64, f64), to: (f64, f64), id: i64) -> omsi_sim::traffic::Lane {
    let mut l = LaneBuilder::polyline(vec![DVec3::new(from.0, from.1, 0.0), DVec3::new(to.0, to.1, 0.0)], LaneKind::Street, 3.5);
    l.source = 1;
    l.key = Some(LaneKey { tile: (0, 0), id, path: 0 });
    l
}

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

fn bus(id: u64, lane: usize, s: f32, speed: f32) -> Vehicle {
    let mut st = AiState::new(lane, s, id);
    st.speed = speed;
    st.front = 6.0;
    st.rear = 6.0;
    st.length = 12.0;
    st.max_speed_kmh = 80.0;
    st.veh_type = 2;
    let mut v = Vehicle::new(id, VehicleClass::Bus, st, 1.25, id * 7919);
    v.fresh = 0.0;
    v
}

struct Run {
    min_gap: f32,
    min_pair: (u64, u64),
    hardest_brake: f32,
    longest_stand: f32,
    removed: usize,
}

/// Like the gate's `run`, but the callback may change the simulation and the vehicles.
fn run(net: &Network, sim: &mut Simulation, vehicles: &mut Vec<Vehicle>, secs: f32, mut each: impl FnMut(f32, &mut Simulation, &mut Vec<Vehicle>)) -> Run {
    let mut r = Run { min_gap: f32::INFINITY, min_pair: (0, 0), hardest_brake: 0.0, longest_stand: 0.0, removed: 0 };
    let mut t = 0.0;
    while t < secs {
        sim.tick(net, DT, vehicles);
        let mut last = usize::MAX;
        for &i in &sim.removed {
            assert!(i < last, "removed indices not strictly descending: {:?}", sim.removed);
            last = i;
            vehicles.remove(i);
            r.removed += 1;
        }
        for v in vehicles.iter() {
            assert!(v.state.speed.is_finite() && v.state.s.is_finite(), "non-finite state for {}", v.id);
            r.hardest_brake = r.hardest_brake.max(-v.state.acc);
            r.longest_stand = r.longest_stand.max(v.stopped);
        }
        for (a, va) in vehicles.iter().enumerate() {
            for vb in vehicles.iter().skip(a + 1) {
                let d = va.obb().separation(&vb.obb()) as f32;
                if d < r.min_gap {
                    r.min_gap = d;
                    r.min_pair = (va.id, vb.id);
                }
            }
        }
        each(t, sim, vehicles);
        t += DT;
    }
    r
}

fn hard_brakes(t: f32, cars: &[Vehicle]) {
    for c in cars {
        if c.state.acc < -6.0 {
            eprintln!("  t={t:.2} HARD BRAKE car {} acc {:.1} v {:.1} lane {} s {:.1} why {:?} stop {:?} lead {:?} seen {:?}", c.id, c.state.acc, c.state.speed, c.state.lane, c.state.s, c.last.why, c.last.stop, c.last.lead.map(|l| l.gap), c.last.lead_seen);
        }
    }
}

fn probe(t: f32, cars: &[Vehicle]) {
    hard_brakes(t, cars);
    let line: Vec<String> = cars
        .iter()
        .map(|c| format!("[{} L{} s={:.1} lat={:.2} v={:.1} why={:?} gap={:?} pass={}]", c.id, c.state.lane, c.state.s, c.state.lateral, c.state.speed, c.last.why, c.last.lead.map(|l| (l.gap * 10.0).round() / 10.0), c.passing.is_some()))
        .collect();
    eprintln!("t={:.1} {}", t, line.join(" "));
}

fn straight_road() -> Network {
    let mut net = Network::default();
    net.lanes.push(road((0.0, 0.0), (300.0, 0.0), 1));
    net.lanes.push(road((300.0, 0.0), (2000.0, 0.0), 2));
    net.link(1.5);
    net.compute_conflicts();
    net
}

/// Eastbound lane 0 and westbound lane 1 (s = 2000 - x), 3 m wide, 3.5 m apart.
fn two_way_road() -> Network {
    let mut net = Network::default();
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

/// `n` lanes of one spline eastbound, lane 0 the leftmost (y = 0), each next 3.5 m right.
fn multi_lane_road(n: usize) -> Network {
    let mut net = Network::default();
    for k in 0..n {
        let mut l = road((0.0, -3.5 * k as f64), (3000.0, -3.5 * k as f64), 1);
        l.offset = 3.5 * k as f32;
        l.key = Some(LaneKey { tile: (0, 0), id: 1, path: k as u16 });
        net.lanes.push(l);
    }
    net.link(1.5);
    net.compute_conflicts();
    net
}

fn person_box(p: DVec3) -> Obb {
    Obb::vehicle(p.truncate(), 0.0, 0.25, 0.25, 0.25)
}

// ---------------------------------------------------------------------------------------
// B1: head-on on a pass

#[test]
fn b1a_a_pass_is_given_up_when_a_car_appears_on_the_oncoming_lane() {
    let net = two_way_road();
    let mut sim = parked_simulation(&net);
    let mut cars = vec![car(1, 0, 100.0, 12.0)];
    let mut spawned = false;
    let mut aborted = false;
    let r = run(&net, &mut sim, &mut cars, 30.0, |t, _, cars| {
        if !spawned {
            if let Some(c) = cars.iter().find(|c| c.id == 1) {
                if c.passing.is_some() {
                    // a car comes round the bend 70 m ahead at 50 km/h
                    let x = c.state.s + 70.0;
                    eprintln!("t={t:.1}: pass started at s={:.1} lat={:.2}; oncoming spawned at x={x:.1}", c.state.s, c.state.lateral);
                    cars.push(car(50, 1, 2000.0 - x, 13.9));
                    spawned = true;
                }
            }
        }
        if cars.iter().any(|c| c.passing.is_some_and(|p| p.aborted)) {
            aborted = true;
        }
        if std::env::var_os("PROBE").is_some() && (t * 30.0) as u32 % 15 == 0 {
            probe(t, cars);
        }
    });
    eprintln!("b1a: spawned {spawned} aborted {aborted} min_gap {:.2} ({:?}) brake {:.1}", r.min_gap, r.min_pair, r.hardest_brake);
    assert!(spawned, "no pass was started");
    assert!(r.min_gap > 0.3, "head-on: nearest {:.2} m between {:?}", r.min_gap, r.min_pair);
}

#[test]
fn b1b_no_pass_is_started_into_a_car_standing_on_the_oncoming_lane() {
    let net = two_way_road();
    let mut sim = parked_simulation(&net);
    // an oncoming car stands (held by a stop of its own) on the oncoming lane 12 m beyond
    // the parked car, right where the pass would go
    let mut stander = car(50, 1, 2000.0 - (PARKED_AT + 14.0), 0.0);
    stander.external.stop_at = Some(2.1);
    let mut cars = vec![car(1, 0, 120.0, 10.0), stander];
    let mut out_and_stuck = 0.0f32;
    let mut started = false;
    let r = run(&net, &mut sim, &mut cars, 40.0, |t, _, cars| {
        let c = cars.iter().find(|c| c.id == 1).unwrap();
        if c.passing.is_some() && !started {
            started = true;
            eprintln!("t={t:.1}: pass started at s={:.1}, oncoming clear_for misjudged", c.state.s);
        }
        if c.state.lateral.abs() > 1.2 && c.state.speed < 0.3 {
            out_and_stuck += DT;
        }
        if std::env::var_os("PROBE").is_some() && (t * 30.0) as u32 % 15 == 0 {
            probe(t, cars);
        }
    });
    eprintln!("b1b: pass started {started}, stood out on the oncoming lane {out_and_stuck:.1} s, min_gap {:.2} {:?}", r.min_gap, r.min_pair);
    assert!(r.min_gap > 0.3, "drove into the standing oncoming car: {:.2} m", r.min_gap);
    assert!(!started, "a pass was started towards a car standing on the oncoming lane");
}

#[test]
fn b1c_after_a_bad_pass_the_two_face_to_face_resolve_without_touching() {
    // b1b, then the oncoming car's hold ends at t = 15 s
    let net = two_way_road();
    let mut sim = parked_simulation(&net);
    let mut stander = car(50, 1, 2000.0 - (PARKED_AT + 14.0), 0.0);
    stander.external.stop_at = Some(2.1);
    let mut cars = vec![car(1, 0, 120.0, 10.0), stander];
    let mut both_still = 0.0f32;
    let r = run(&net, &mut sim, &mut cars, 60.0, |t, _, cars| {
        if t > 15.0 {
            if let Some(o) = cars.iter_mut().find(|c| c.id == 50) {
                o.external.stop_at = None;
            }
        }
        if t > 16.0 && cars.iter().all(|c| c.state.speed < 0.3) {
            both_still += DT;
        }
        if std::env::var_os("PROBE").is_some() && (t * 30.0) as u32 % 45 == 0 {
            probe(t, cars);
        }
    });
    let c1 = cars.iter().find(|c| c.id == 1).map(|c| (c.state.s, c.state.lateral, c.last.lead_seen, c.last.body_block));
    let c50 = cars.iter().find(|c| c.id == 50).map(|c| (c.state.s, c.state.lateral, c.last.lead_seen, c.last.body_block));
    eprintln!("b1c: min_gap {:.2} {:?}; both standing {both_still:.1} s after the release; car 1 {:?} car 50 {:?}", r.min_gap, r.min_pair, c1, c50);
    assert!(r.min_gap > 0.3, "head-on after the release: {:.2} m", r.min_gap);
    assert!(both_still < 10.0, "face-to-face deadlock for {both_still:.1} s");
}

#[test]
fn b4d_simultaneous_change_with_seeded_drivers_and_offset_positions() {
    // b4b without identical drivers: seeded personalities, 2 m apart
    let net = multi_lane_road(3);
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = Vec::new();
    for (k, lane, s) in [(0u64, 0usize, 120.0f32), (1, 2, 118.0)] {
        let mut slow = car(10 + k, lane, 160.0, 5.5);
        slow.state.max_speed_kmh = 20.0;
        slow.state.change_cooldown = 1000.0;
        cars.push(slow);
        let mut f = car(3 + k, lane, s, 12.0);
        f.params.lc_speed_gain = f.params.lc_speed_gain.max(0.7);
        f.state.change_cooldown = 0.0;
        cars.push(f);
    }
    let r = run(&net, &mut sim, &mut cars, 25.0, |_, _, _| {});
    eprintln!("b4d: min_gap {:.2} {:?}", r.min_gap, r.min_pair);
    assert!(r.min_gap > 0.3, "the two met in the middle lane: {:.2} m {:?}", r.min_gap, r.min_pair);
}

// ---------------------------------------------------------------------------------------
// B2: a bus at its stop, cars behind

#[test]
fn b2_cars_behind_a_bus_at_its_stop_wait_or_pass_safely_and_follow_it_off() {
    let net = two_way_road();
    let mut sim = Simulation::new(&net, vec![]);
    let mut b = bus(9, 0, 200.0, 0.0);
    b.external.stop_at = Some(6.1);
    b.external.at_stop = true;
    let mut cars = vec![b, car(1, 0, 150.0, 12.0), car(2, 0, 110.0, 12.0)];
    // a sparse oncoming stream
    for k in 0..4u64 {
        cars.push(car(20 + k, 1, 2000.0 - 420.0 - 160.0 * k as f32, 13.0));
    }
    let mut stood_for = 0.0f32;
    let mut passed = Vec::new();
    let r = run(&net, &mut sim, &mut cars, 45.0, |t, _, cars| {
        stood_for += DT;
        hard_brakes(t, cars);
        if let Some(b) = cars.iter_mut().find(|c| c.id == 9) {
            if t < 20.0 {
                b.external.standing_for = stood_for;
            } else {
                b.external.stop_at = None;
                b.external.at_stop = false;
                b.external.standing_for = 0.0;
            }
        }
        let bus_s = cars.iter().find(|c| c.id == 9).map(|b| b.state.s).unwrap_or(0.0);
        for c in cars.iter() {
            if (c.id == 1 || c.id == 2) && c.state.lane == 0 && c.state.s > bus_s && !passed.contains(&c.id) {
                passed.push(c.id);
                eprintln!("t={t:.1}: car {} got past the bus", c.id);
            }
        }
        if std::env::var_os("PROBE").is_some() && (t * 30.0) as u32 % 15 == 0 {
            probe(t, cars);
        }
    });
    eprintln!("b2: min_gap {:.2} {:?} brake {:.1} passed {:?}", r.min_gap, r.min_pair, r.hardest_brake, passed);
    assert!(r.min_gap > 0.3, "touched: {:.2} m {:?}", r.min_gap, r.min_pair);
    assert!(r.hardest_brake < 4.5, "braked {:.1}", r.hardest_brake);
}

// ---------------------------------------------------------------------------------------
// B3: entering a junction behind a car that stands inside it

#[test]
fn b3_a_car_does_not_follow_into_the_box_behind_a_car_standing_in_it() {
    let net = priority_junction();
    let mut sim = Simulation::new(&net, vec![]);
    // A stands in the main road piece past the conflict (front at s 26 of 30)
    let mut a = car(9, 1, 24.0, 0.0);
    a.external.stop_at = Some(2.1);
    let mut cars = vec![a, car(1, 0, 140.0, 12.0), car(2, 3, 150.0, 10.0)];
    let mut b_in_conflict = 0.0f32;
    let mut c_waited_for_b = 0.0f32;
    let js = sim.junctions.clone();
    let mv = js.movement(1).unwrap().1.clone();
    eprintln!("main movement conflict {:.1}..{:.1}, exits {:?}", mv.conflict_start, mv.conflict_end, mv.exits);
    let r = run(&net, &mut sim, &mut cars, 45.0, |t, _, cars| {
        if t > 20.0 {
            if let Some(a) = cars.iter_mut().find(|c| c.id == 9) {
                a.external.stop_at = None;
            }
        }
        let b = cars.iter().find(|c| c.id == 1).unwrap();
        let inside = b.state.lane == 1 && b.state.s - b.state.rear < mv.conflict_end && b.state.s + b.state.front > mv.conflict_start;
        if t < 20.0 && inside && b.state.speed < 0.3 {
            b_in_conflict += DT;
        }
        if let Some(c) = cars.iter().find(|c| c.id == 2) {
            if c.state.speed < 0.3 && c.last.why == Why::Junction && t < 20.0 {
                c_waited_for_b += DT;
            }
        }
        if std::env::var_os("PROBE").is_some() && (t * 30.0) as u32 % 15 == 0 {
            probe(t, cars);
        }
    });
    eprintln!("b3: B stood in the conflict area {b_in_conflict:.1} s while A blocked; C held at the side road {c_waited_for_b:.1} s; min_gap {:.2} {:?}", r.min_gap, r.min_pair);
    assert!(r.min_gap > 0.3, "touched {:.2} {:?}", r.min_gap, r.min_pair);
    assert!(b_in_conflict < 0.5, "B queued inside the junction's conflict area for {b_in_conflict:.1} s (keep-clear only looks at the exit lanes)");
}

// ---------------------------------------------------------------------------------------
// B4: lane changes

#[test]
fn b4a_no_lane_change_into_a_faster_car_in_the_blind_spot() {
    let net = multi_lane_road(2);
    assert_eq!(net.lanes[1].left, Some(0));
    let mut sim = Simulation::new(&net, vec![]);
    let mut slow = car(1, 1, 150.0, 5.5);
    slow.state.max_speed_kmh = 20.0;
    slow.state.change_cooldown = 1000.0;
    let mut x = car(2, 1, 125.0, 12.0);
    x.params.lc_speed_gain = 0.9;
    x.state.change_cooldown = 0.0;
    // Y in the left lane, 6 m behind X's rear and 3 m/s faster
    let mut y = car(3, 0, 115.0, 15.0);
    y.state.change_cooldown = 1000.0;
    let mut cars = vec![slow, x, y];
    let r = run(&net, &mut sim, &mut cars, 25.0, |t, _, cars| {
        if std::env::var_os("PROBE").is_some() && (t * 30.0) as u32 % 15 == 0 {
            probe(t, cars);
        }
    });
    eprintln!("b4a: min_gap {:.2} {:?} brake {:.1}", r.min_gap, r.min_pair, r.hardest_brake);
    assert!(r.min_gap > 0.3, "touched {:.2} {:?}", r.min_gap, r.min_pair);
    assert!(r.hardest_brake < 4.5, "braked {:.1}", r.hardest_brake);
}

#[test]
fn b4b_two_cars_do_not_change_into_the_same_gap_from_both_sides() {
    let net = multi_lane_road(3);
    assert_eq!(net.lanes[1].left, Some(0));
    assert_eq!(net.lanes[1].right, Some(2));
    let mut sim = Simulation::new(&net, vec![]);
    let mut p = DriverParams::preset(VehicleClass::Car);
    p.lc_speed_gain = 0.9;
    let mut cars = Vec::new();
    for (k, lane) in [(0u64, 0usize), (1, 2)] {
        let mut slow = car(10 + k, lane, 160.0, 5.5);
        slow.state.max_speed_kmh = 20.0;
        slow.state.change_cooldown = 1000.0;
        cars.push(slow);
        let mut f = car(1 + k, lane, 120.0, 12.0);
        f.params = p;
        p.apply(&mut f.state);
        f.state.change_cooldown = 0.0;
        cars.push(f);
    }
    let mut both_changing = false;
    let r = run(&net, &mut sim, &mut cars, 25.0, |t, _, cars| {
        let n = cars.iter().filter(|c| c.state.change.is_some_and(|ch| ch.to == 1)).count();
        if n == 2 && !both_changing {
            both_changing = true;
            eprintln!("t={t:.1}: both fast cars are changing into the middle lane at once");
        }
        if std::env::var_os("PROBE").is_some() && (t * 30.0) as u32 % 15 == 0 {
            probe(t, cars);
        }
    });
    eprintln!("b4b: both_changing {both_changing} min_gap {:.2} {:?}", r.min_gap, r.min_pair);
    assert!(r.min_gap > 0.3, "the two met in the middle lane: {:.2} m {:?}", r.min_gap, r.min_pair);
}

#[test]
fn b4c_every_driver_eventually_overtakes_a_crawler_on_a_two_lane_road() {
    let net = multi_lane_road(2);
    let mut never = Vec::new();
    for seed_id in 1..=20u64 {
        let mut sim = Simulation::new(&net, vec![]);
        let mut slow = car(1000, 1, 150.0, 5.5);
        slow.state.max_speed_kmh = 20.0;
        slow.state.change_cooldown = 1000.0;
        let mut f = car(seed_id, 1, 110.0, 12.0);
        f.state.change_cooldown = 0.0;
        let gain = f.params.lc_speed_gain;
        let mut cars = vec![slow, f];
        let mut changed = false;
        run(&net, &mut sim, &mut cars, 60.0, |_, _, cars| {
            if cars.iter().any(|c| c.id == seed_id && (c.state.lane == 0 || c.state.change.is_some())) {
                changed = true;
            }
        });
        if !changed {
            never.push((seed_id, gain));
        }
    }
    eprintln!("b4c: drivers who never overtook a 20 km/h car in 60 s: {} of 20 {:?}", never.len(), never);
    assert!(never.is_empty(), "{} of 20 drivers sat behind a 20 km/h car for a minute on an empty two-lane road: (id, lc_speed_gain) {:?}", never.len(), never);
}

// ---------------------------------------------------------------------------------------
// B5: a queue at a red light starts on green

#[test]
fn b5_a_queue_at_red_starts_on_green_without_touching() {
    let mut net = Network::default();
    net.lanes.push(road((0.0, 0.0), (200.0, 0.0), 1));
    let mut p = piece((200.0, 0.0), (230.0, 0.0), 9, 0, 128.0);
    p.traffic_light = Some((0, 0));
    net.lanes.push(p);
    net.lanes.push(road((230.0, 0.0), (1500.0, 0.0), 2));
    net.link(1.5);
    net.compute_conflicts();
    // red 15 s, green 25 s
    let ctl = TrafficLightController::new(vec![vec![(0, 15.0), (6, 25.0)]], 40.0);
    let mut sim = Simulation::new(&net, vec![ctl]);
    let mut cars: Vec<Vehicle> = (0..6).map(|k| car(k + 1, 0, 150.0 - 18.0 * k as f32, 10.0)).collect();
    let mut crossed_at: Vec<(u64, f32)> = Vec::new();
    let mut on_red = false;
    let r = run(&net, &mut sim, &mut cars, 40.0, |t, sim, cars| {
        let red = sim.lights[0].state(0) < 6;
        for c in cars.iter() {
            if c.state.lane != 0 && !crossed_at.iter().any(|x| x.0 == c.id) {
                crossed_at.push((c.id, t));
                if red {
                    on_red = true;
                }
            }
        }
        if std::env::var_os("PROBE").is_some() && (t * 30.0) as u32 % 15 == 0 {
            probe(t, cars);
        }
    });
    eprintln!("b5: crossed {:?} min_gap {:.2} brake {:.1}", crossed_at, r.min_gap, r.hardest_brake);
    assert!(!on_red);
    assert!(r.min_gap > 0.3, "touched {:.2} {:?}", r.min_gap, r.min_pair);
    assert_eq!(crossed_at.len(), 6, "not all six got across in 25 s of green");
    let last = crossed_at.iter().map(|x| x.1).fold(0.0, f32::max);
    assert!(last < 15.0 + 20.0, "the queue took {:.1} s to clear", last - 15.0);
}

// ---------------------------------------------------------------------------------------
// B6: pedestrians stepping out

#[test]
fn b6a_a_pedestrian_walking_out_from_the_kerb_is_not_run_over() {
    let net = straight_road();
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![car(1, 0, 100.0, 13.0)];
    // 28 m ahead of the front bumper, 3 m to the right, walking across at 1.4 m/s
    let mut pos = DVec3::new(130.0, -3.0, 0.0);
    let vel = DVec2::new(0.0, 1.4);
    let mut hit = f32::INFINITY;
    let r = run(&net, &mut sim, &mut cars, 12.0, |_, sim, cars| {
        if pos.y < 5.0 {
            pos += (vel * DT as f64).extend(0.0);
        }
        sim.people = vec![Person { pos, vel: if pos.y < 5.0 { vel } else { DVec2::ZERO }, on_crossing: false, still: 0.0, waiting_for_bus: false, at_kerb: false, waits_at: None }];
        hit = hit.min(cars[0].obb().separation(&person_box(pos)) as f32);
    });
    eprintln!("b6a: nearest car-person {:.2} m, brake {:.1}", hit, r.hardest_brake);
    assert!(hit > 0.2, "the car came within {hit:.2} m of the pedestrian");
}

#[test]
fn b6b_a_pedestrian_stepping_out_2m_ahead_of_a_crawling_car() {
    let net = straight_road();
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![car(1, 0, 100.0, 3.0)];
    let mut hit = f32::INFINITY;
    let mut placed = false;
    let r = run(&net, &mut sim, &mut cars, 8.0, |t, sim, cars| {
        if t > 1.0 && !placed {
            // steps out 2 m in front of the bumper, into the lane
            let x = cars[0].state.s as f64 + cars[0].state.front as f64 + 2.0;
            sim.people = vec![Person { pos: DVec3::new(x, -0.5, 0.0), vel: DVec2::new(0.0, 1.2), on_crossing: false, still: 0.0, waiting_for_bus: false, at_kerb: false, waits_at: None }];
            placed = true;
        }
        if let Some(p) = sim.people.first() {
            hit = hit.min(cars[0].obb().separation(&person_box(p.pos)) as f32);
        }
    });
    eprintln!("b6b: nearest car-person {:.2} m, brake {:.1}", hit, r.hardest_brake);
    assert!(hit > 0.0, "drove into the pedestrian");
}

// ---------------------------------------------------------------------------------------
// B7: dead end

#[test]
fn b7_vehicles_reaching_a_dead_end_are_removed_one_by_one_without_panics() {
    let mut net = Network::default();
    net.lanes.push(road((0.0, 0.0), (300.0, 0.0), 1));
    net.link(1.5);
    net.compute_conflicts();
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars: Vec<Vehicle> = (0..4).map(|k| car(k + 1, 0, 250.0 - 9.0 * k as f32, 12.0)).collect();
    let mut speeds_at_end = Vec::new();
    let mut prev: Vec<(u64, f32)> = Vec::new();
    let r = run(&net, &mut sim, &mut cars, 30.0, |_, sim, cars| {
        for (id, v) in &prev {
            if !cars.iter().any(|c| c.id == *id) {
                speeds_at_end.push((*id, *v));
            }
        }
        let _ = sim;
        prev = cars.iter().map(|c| (c.id, c.state.speed)).collect();
    });
    eprintln!("b7: removed {} speeds at the end of the road {:?}, min_gap {:.2}", r.removed, speeds_at_end, r.min_gap);
    assert_eq!(r.removed, 4);
    assert!(r.min_gap > 0.3);
}

#[test]
fn b7b_two_vehicles_removed_in_the_same_frame() {
    // two parallel dead-end lanes, two cars reaching the end together
    let mut net = Network::default();
    net.lanes.push(road((0.0, 0.0), (100.0, 0.0), 1));
    net.lanes.push(road((0.0, 50.0), (100.0, 50.0), 2));
    net.link(1.5);
    net.compute_conflicts();
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![car(1, 0, 90.0, 10.0), car(2, 1, 90.0, 10.0), car(3, 0, 10.0, 0.0)];
    cars[2].external.stop_at = Some(2.1);
    let r = run(&net, &mut sim, &mut cars, 5.0, |_, _, _| {});
    assert_eq!(r.removed, 2);
    assert_eq!(cars.len(), 1);
    assert_eq!(cars[0].id, 3);
}

// ---------------------------------------------------------------------------------------
// B8: two lanes into one

fn y_joint() -> Network {
    let mut net = Network::default();
    net.lanes.push(road((0.0, 0.0), (200.0, 0.0), 1)); // 0 straight
    net.lanes.push(road((0.0, -25.0), (200.0, 0.0), 2)); // 1 slip, ~7 degrees
    net.lanes.push(road((200.0, 0.0), (2000.0, 0.0), 3)); // 2 the one lane on
    net.link(1.5);
    net.compute_conflicts();
    net
}

#[test]
fn b8a_two_cars_arriving_together_at_a_y_joint_do_not_touch() {
    let net = y_joint();
    assert_eq!(net.lanes[0].next, vec![2]);
    assert_eq!(net.lanes[1].next, vec![2]);
    let mut sim = Simulation::new(&net, vec![]);
    let s1 = net.lanes[1].length() - 100.0;
    let mut cars = vec![car(1, 0, 100.0, 12.0), car(2, 1, s1, 12.0)];
    let r = run(&net, &mut sim, &mut cars, 25.0, |t, _, cars| {
        if std::env::var_os("PROBE").is_some() && (t * 30.0) as u32 % 15 == 0 {
            probe(t, cars);
        }
    });
    eprintln!("b8a: min_gap {:.2} {:?} brake {:.1}", r.min_gap, r.min_pair, r.hardest_brake);
    assert!(r.min_gap > 0.3, "met at the joint: {:.2}", r.min_gap);
    assert!(r.hardest_brake < 4.5, "braked {:.1}", r.hardest_brake);
}

#[test]
fn b8b_a_car_on_the_slip_gets_into_a_stream() {
    let net = y_joint();
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars: Vec<Vehicle> = (0..8).map(|k| car(10 + k, 0, 150.0 - 22.0 * k as f32, 12.0)).collect();
    let s1 = net.lanes[1].length() - 40.0;
    cars.push(car(1, 1, s1, 10.0));
    let mut merged_at = None;
    let r = run(&net, &mut sim, &mut cars, 40.0, |t, _, cars| {
        if merged_at.is_none() && cars.iter().any(|c| c.id == 1 && c.state.lane == 2) {
            merged_at = Some(t);
        }
    });
    eprintln!("b8b: merged at {:?}, min_gap {:.2} {:?}, brake {:.1}", merged_at, r.min_gap, r.min_pair, r.hardest_brake);
    assert!(r.min_gap > 0.3, "touched at the joint {:.2} {:?}", r.min_gap, r.min_pair);
    assert!(merged_at.is_some());
}

// ---------------------------------------------------------------------------------------
// The player at a junction (the old module had tests for this)

#[test]
fn p7_a_side_road_car_gives_way_to_the_players_bus_on_the_main_road() {
    let net = priority_junction();
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![car(1, 3, 160.0, 10.0)];
    // the player's bus on the main road, eastbound at 11 m/s, reaching the middle at ~t=5.5 s
    let mut x = -60.0f64;
    let mut nearest = f32::INFINITY;
    let mut waited = false;
    let r = run(&net, &mut sim, &mut cars, 15.0, |_, sim, cars| {
        x += 11.0 * DT as f64;
        hard_brakes(0.0, cars);
        if let Some(c) = cars.first() { if std::env::var_os("PROBE").is_some() { eprintln!("  x_player={x:.1} car L{} s={:.1} v={:.1} why={:?} stop={:?} lead={:?}", c.state.lane, c.state.s, c.state.speed, c.last.why, c.last.stop, c.last.lead.map(|l| l.gap)); } }
        let pv = PlayerView { id: 0, pos: DVec3::new(x, 0.0, 0.0), heading: 90.0, half_len: 6.0, half_width: 1.25, speed: 11.0, still: 0.0, blinker: 0, signalling: 0.0, priority: false };
        sim.players = vec![pv];
        if let Some(c) = cars.first() {
            nearest = nearest.min(c.obb().separation(&pv.obb(0.0, 0.0)) as f32);
            if c.state.speed < 0.5 && c.state.lane == 3 {
                waited = true;
            }
        }
    });
    eprintln!("p7: side road car waited {waited}, nearest to the player's bus {nearest:.2} m, brake {:.1}", r.hardest_brake);
    assert!(nearest > 0.5, "the side road car met the player's bus: {nearest:.2} m");
    assert!(r.hardest_brake < 4.5, "braked {:.1} for the player's bus", r.hardest_brake);
}

// ---------------------------------------------------------------------------------------
// D: cost of a frame with 300 vehicles and 200 people

#[test]
#[ignore = "a timing bound: run by hand, not on a loaded CI machine"]
fn d_cost_of_a_frame_with_300_vehicles_and_200_people() {
    // 12 lanes of 1 km, 25 cars on each (40 m apart), people along the kerbs
    let mut net = Network::default();
    for k in 0..12 {
        net.lanes.push(road((0.0, 30.0 * k as f64), (1000.0, 30.0 * k as f64), 100 + k as i64));
    }
    net.link(1.5);
    net.compute_conflicts();
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = Vec::new();
    for k in 0..12usize {
        for j in 0..25usize {
            cars.push(car((k * 100 + j + 1) as u64, k, 20.0 + 38.0 * j as f32, 10.0));
        }
    }
    let people: Vec<Person> = (0..200).map(|i| {
        let lane = i % 12;
        Person { pos: DVec3::new(10.0 + 5.0 * (i / 12) as f64 * 12.0 % 990.0, 30.0 * lane as f64 - 4.5, 0.0), vel: DVec2::ZERO, on_crossing: false, still: 10.0, waiting_for_bus: false, at_kerb: false, waits_at: None }
    }).collect();
    sim.people = people;
    // warm up
    for _ in 0..10 {
        sim.tick(&net, DT, &mut cars);
    }
    let n = 120;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        sim.tick(&net, DT, &mut cars);
    }
    let per = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let mut sim2 = Simulation::new(&net, vec![]);
    for _ in 0..10 {
        sim2.tick(&net, DT, &mut cars);
    }
    let t1 = std::time::Instant::now();
    for _ in 0..n {
        sim2.tick(&net, DT, &mut cars);
    }
    let per_no_people = t1.elapsed().as_secs_f64() * 1000.0 / n as f64;
    eprintln!("D: {} vehicles, 200 people: {per:.2} ms per tick; without people {per_no_people:.2} ms", cars.len());
    assert!(per < 4.0, "{per:.2} ms per traffic tick");
}

#[test]
#[ignore = "a timing bound: run by hand, not on a loaded CI machine"]
fn d2_cost_split_by_stage() {
    // the same 300 cars on lanes 30 m apart vs 400 m apart (no other body within reach),
    // and the perception stage alone
    for spacing in [30.0f64, 400.0] {
        let mut net = Network::default();
        for k in 0..12 {
            net.lanes.push(road((0.0, spacing * k as f64), (1000.0, spacing * k as f64), 100 + k as i64));
        }
        net.link(1.5);
        net.compute_conflicts();
        let mut sim = Simulation::new(&net, vec![]);
        let mut cars = Vec::new();
        for k in 0..12usize {
            for j in 0..25usize {
                cars.push(car((k * 100 + j + 1) as u64, k, 20.0 + 38.0 * j as f32, 10.0));
            }
        }
        for _ in 0..10 {
            sim.tick(&net, DT, &mut cars);
        }
        let n = 60;
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            sim.tick(&net, DT, &mut cars);
        }
        let tick = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let t1 = std::time::Instant::now();
        for _ in 0..n {
            for i in 0..cars.len() {
                std::hint::black_box(omsi_traffic::perception::perceive(&net, &sim, &cars, i));
            }
        }
        let perc = t1.elapsed().as_secs_f64() * 1000.0 / n as f64;
        eprintln!("D2: spacing {spacing} m: tick {tick:.2} ms, perception alone {perc:.2} ms");
    }
}
