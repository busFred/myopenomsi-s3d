//! Review4 counter-examples for round 3 (the competitor filter widened), driven in the
//! simulation: who goes first, how hard the car with the right of way brakes, how close the
//! bodies come. `cargo test --release -p omsi-traffic --test review4 -- --nocapture`

use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network, TrafficLightController};
use omsi_traffic::{Simulation, Vehicle, VehicleClass};

const DT: f32 = 1.0 / 30.0;

fn road(from: (f64, f64), to: (f64, f64), id: i64) -> omsi_sim::traffic::Lane {
    let mut l = LaneBuilder::polyline(vec![DVec3::new(from.0, from.1, 0.0), DVec3::new(to.0, to.1, 0.0)], LaneKind::Street, 3.5);
    l.source = 1;
    l.key = Some(LaneKey { tile: (0, 0), id, path: 0 });
    l
}

fn piece(from: (f64, f64), to: (f64, f64), path: u16, priority: f32) -> omsi_sim::traffic::Lane {
    let mut l = LaneBuilder::polyline(vec![DVec3::new(from.0, from.1, 0.0), DVec3::new(to.0, to.1, 0.0)], LaneKind::Street, 3.5);
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

/// (first car into `exit`, min box separation, min acceleration of car `who` after `after` s)
fn drive(net: &Network, sim: &mut Simulation, cars: &mut [Vehicle], secs: f32, past: &dyn Fn(&Vehicle) -> bool, who: u64, after: f32) -> (Option<u64>, f32, f32) {
    let (mut t, mut first, mut min_gap, mut min_acc) = (0.0f32, None, f32::INFINITY, f32::INFINITY);
    let mut last_v: Vec<f32> = cars.iter().map(|c| c.state.speed).collect();
    while t < secs {
        sim.tick(net, DT, cars);
        for (a, va) in cars.iter().enumerate() {
            for vb in cars.iter().skip(a + 1) {
                min_gap = min_gap.min(va.obb().separation(&vb.obb()) as f32);
            }
            if first.is_none() && past(va) {
                first = Some(va.id);
            }
            if va.id == who && t >= after {
                min_acc = min_acc.min((va.state.speed - last_v[a]) / DT);
            }
        }
        last_v = cars.iter().map(|c| c.state.speed).collect();
        if std::env::var_os("DBG").is_some() && (t * 30.0).round() as i32 % 15 == 0 {
            let line: Vec<String> = cars.iter().map(|c| format!("[{} l{} s={:.1} v={:.1} why={:?} wf={:?} hold={:?}]", c.id, c.state.lane, c.state.s, c.state.speed, c.last.why, c.last.wait_for, c.last.hold)).collect();
            eprintln!("t={t:.1} {}", line.join(" "));
        }
        t += DT;
    }
    (first, min_gap, min_acc)
}

/// The side car's place: its front 0.5 m before the merge's meeting place on A.
fn side_s() -> f32 {
    let net = merge_net();
    let js = omsi_traffic::Junctions::compile(&net);
    let (_, a) = js.movement(3).unwrap();
    a.conflict_start - 0.5 - 2.0
}

/// A side lane (A, gives way) merges into the main lane (B) at the origin; exit east.
fn merge_net() -> Network {
    let mut net = Network::default();
    net.lanes.push(road((-400.0, 0.0), (-10.0, 0.0), 1)); // 0 main approach
    net.lanes.push(piece((-10.0, 0.0), (0.0, 0.0), 0, 192.0)); // 1 B
    net.lanes.push(road((-168.0, 126.0), (-8.0, 6.0), 2)); // 2 side approach (from the left)
    net.lanes.push(piece((-8.0, 6.0), (0.0, 0.0), 1, 64.0)); // 3 A
    net.lanes.push(road((0.0, 0.0), (1500.0, 0.0), 3)); // 4 exit
    net.link(0.5);
    net.compute_conflicts();
    net
}

/// (C2) The side car stands with its nose at the merge; the main road car comes at 13.5 m/s.
/// For each distance: who is first into the exit, the closest the boxes came, and the
/// hardest braking of the main road car (which has the right of way).
#[test]
fn review4_merging_behind_at_speed() {
    let net = merge_net();
    let mut worst = (0.0f32, 0.0f32, f32::INFINITY);
    println!("D(m) | first  min_gap  main_min_acc");
    for d in std::env::var("ONLY_D").ok().map(|x| x.parse::<usize>().unwrap()).map(|x| x..=x).unwrap_or(40..=140).step_by(5) {
        let mut sim = Simulation::new(&net, vec![]);
        let mut side = car(1, 2, net.lanes[2].length() - 2.0 - 0.3, 0.0);
        side.state.planned_next = Some(3);
        side.state.ahead = vec![4];
        let _ = side_s;
        let main_s = net.lanes[0].length() - (d as f32 - 10.0) - 2.0;
        let mut main = car(2, 0, main_s, 13.5);
        main.state.planned_next = Some(1);
        main.state.ahead = vec![4];
        let mut cars = vec![side, main];
        let (first, gap, acc) = drive(&net, &mut sim, &mut cars, 14.0, &|v: &Vehicle| v.state.lane == 4, 2, 0.5);
        println!("{d:>4} | {first:?} {gap:>6.2} {acc:>7.2}");
        if acc < worst.2 {
            worst = (d as f32, gap, acc);
        }
    }
    assert!(worst.2 > -3.0, "the main road car braked {:.1} m/s² (D {} m, min gap {:.2} m)", worst.2, worst.0, worst.1);
}

/// (C3) A main road car (right of way, west) stands at its red, held by the light; the light
/// turns green at 5 s. A side road car (south, green, gives way) comes up at 8 m/s. The
/// standing main road car is "blocked" (why=Light) and dropped when the side car is out
/// before it could be in "from rest after 1.5 s"; the real start reaction is 0.3-1.3 s.
#[test]
fn review4_blocked_foe_whose_light_turns_green_now() {
    let mut net = Network::default();
    net.lanes.push(road((-200.0, 0.0), (-20.0, 0.0), 1)); // 0 west approach
    net.lanes.push(piece((-20.0, 0.0), (-12.0, 0.0), 0, 192.0)); // 1 lead-in
    let mut stub = piece((-12.0, 0.0), (-11.0, 0.0), 1, 192.0); // 2 lit stub
    stub.traffic_light = Some((0, 0));
    net.lanes.push(stub);
    net.lanes.push(piece((-11.0, 0.0), (15.0, 0.0), 2, 192.0)); // 3 across
    net.lanes.push(road((15.0, 0.0), (1500.0, 0.0), 2)); // 4 east exit
    net.lanes.push(road((0.0, -200.0), (0.0, -20.0), 3)); // 5 south approach
    net.lanes.push(piece((0.0, -20.0), (0.0, -12.0), 3, 64.0)); // 6 lead-in
    let mut stub = piece((0.0, -12.0), (0.0, -11.0), 4, 64.0); // 7 lit stub
    stub.traffic_light = Some((0, 1));
    net.lanes.push(stub);
    net.lanes.push(piece((0.0, -11.0), (0.0, 15.0), 5, 64.0)); // 8 across
    net.lanes.push(road((0.0, 15.0), (0.0, 1500.0), 4)); // 9 north exit
    net.link(0.5);
    net.compute_conflicts();
    let mut worst = (0.0f32, f32::INFINITY, f32::INFINITY, None);
    println!("green | first  min_gap  west_min_acc (west reaction)");
    for k in 0..=36 {
        let green = 3.0 + 0.5 * k as f32;
        let d = 30.0f32;
        let ctl = TrafficLightController::new(vec![vec![(0, green), (6, 100.0)], vec![(6, 200.0)]], 100.0 + green);
        let mut sim = Simulation::new(&net, vec![ctl]);
        let mut west = car(1, 1, 8.0 - 2.0 - 0.3, 0.0);
        west.state.planned_next = Some(2);
        west.state.ahead = vec![3, 4];
        // south car: front `d` m before the crossing point at 8 m/s at t = 0
        let mut south = car(2, 5, 200.0 - d - 2.0, 8.0);
        south.state.max_speed_kmh = 29.0;
        south.state.planned_next = Some(6);
        south.state.ahead = vec![7, 8, 9];
        let reaction = west.state.reaction;
        let mut cars = vec![west, south];
        let (first, gap, acc) = drive(&net, &mut sim, &mut cars, green + 14.0, &|v: &Vehicle| (v.state.lane == 3 || v.state.lane == 8) && v.state.s - v.state.rear > 13.0, 1, green);
        println!("green {green:>5.1} | {first:?} {gap:>6.2} {acc:>7.2} ({reaction:.2})");
        if gap < worst.1 || acc < worst.2 {
            worst = (green, gap.min(worst.1), acc.min(worst.2), first);
        }
    }
    assert!(worst.2 > -3.0 && worst.1 > 0.0, "worst west braking {:.1} m/s², min gap {:.2} m (green at {} s)", worst.2, worst.1, worst.0);
}
