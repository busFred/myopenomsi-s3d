//! Headless scenarios for the junction rules of round 4: where a car held at a junction
//! stops (before the stretch it waits for), the player's claim over the pieces ahead, and a
//! car inside the box when its light turns red.

use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network};
use omsi_traffic::{Simulation, Vehicle, VehicleClass};

const DT: f32 = 1.0 / 30.0;

fn lane(pts: &[(f64, f64)], kind: LaneKind) -> omsi_sim::traffic::Lane {
    LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), kind, 3.5)
}

/// A spline lane (source 1).
fn road(pts: &[(f64, f64)], id: i64) -> omsi_sim::traffic::Lane {
    let mut l = lane(pts, LaneKind::Street);
    l.source = 1;
    l.key = Some(LaneKey { tile: (0, 0), id, path: 0 });
    l
}

/// A lane of crossing object 9 (source 2).
fn piece(pts: &[(f64, f64)], path: u16, priority: f32) -> omsi_sim::traffic::Lane {
    let mut l = lane(pts, LaneKind::Street);
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

/// Review7: the zipper's "who went last" lives in `Simulation::signal_groups`, which no keyframe
/// carries. A simulation restored from a keyframe (the vehicles, as `Replay::at` and the reset
/// replay do) and the one that made it part ways at the next level tie.
#[test]
fn review7_zipper_memory_survives_a_keyframe_restore() {
    // two parallel approaches 4 m apart merge into one exit inside object 9
    let mut net = Network::default();
    net.lanes.push(road(&[(-400.0, -2.0), (-10.0, -2.0)], 1)); // 0
    net.lanes.push(road(&[(-400.0, 2.0), (-10.0, 2.0)], 2)); // 1
    net.lanes.push(piece(&[(-10.0, -2.0), (0.0, 0.0)], 0, 128.0)); // 2
    net.lanes.push(piece(&[(-10.0, 2.0), (0.0, 0.0)], 1, 128.0)); // 3
    net.lanes.push(piece(&[(0.0, 0.0), (10.0, 0.0)], 2, 128.0)); // 4
    net.lanes.push(road(&[(10.0, 0.0), (2000.0, 0.0)], 3)); // 5
    let net = finish(net);
    let js = omsi_traffic::Junctions::compile(&net);
    let (_, m) = js.movement(2).unwrap();
    assert!(m.foes.iter().any(|f| f.kind == omsi_traffic::junction::FoeKind::Merge && !f.yields), "{:?}", m.foes);
    // level pairs, one on each approach, every 14 m, rolling at 8 m/s
    let mut cars: Vec<Vehicle> = Vec::new();
    for k in 0..14u64 {
        cars.push(car(100 + k, 0, 380.0 - 14.0 * k as f32, 8.0));
        cars.push(car(200 + k, 1, 380.0 - 14.0 * k as f32 + 0.3, 8.0));
    }
    let mut sim = Simulation::new(&net, vec![]);
    let mut snap: Option<(f64, Vec<Vehicle>, omsi_traffic::reservation::Context)> = None;
    let mut t = 0.0f32;
    let mut a_traj: Vec<Vec<(u64, f64, f64)>> = Vec::new();
    while t < 60.0 {
        sim.tick(&net, DT, &mut cars);
        for &i in &sim.removed { cars.remove(i); }
        t += DT;
        if snap.is_none() && t >= 20.0 && !sim.signal_groups.zipper.is_empty() {
            snap = Some((sim.time, cars.clone(), sim.signal_groups.clone()));
        }
        if snap.is_some() { a_traj.push(cars.iter().map(|v| (v.id, v.pos.x, v.pos.y)).collect()); }
    }
    let (time, cars_at, groups) = snap.expect("no zipper tie in 20 s");
    let drift = |with_memory: bool| -> f64 {
    let mut cars2 = cars_at.clone();
    let mut sim2 = Simulation::new(&net, vec![]);
    sim2.time = time;
    if with_memory { sim2.signal_groups = groups.clone(); }
    let mut worst = 0.0f64;
    for frame in a_traj.iter().skip(1) {
        sim2.tick(&net, DT, &mut cars2);
        for &i in &sim2.removed { cars2.remove(i); }
        for &(id, x, y) in frame {
            if let Some(v) = cars2.iter().find(|v| v.id == id) {
                worst = worst.max(((v.pos.x - x).powi(2) + (v.pos.y - y).powi(2)).sqrt());
            }
        }
    }
    worst
    };
    let control = drift(true);
    assert!(control < 1e-6, "control (zipper memory copied too) drifted {control:.3} m");
    let worst = drift(false);
    assert!(worst < 1e-6, "the restored simulation (vehicles only, as a keyframe) drifted {worst:.2} m from the one it was restored from; with the zipper memory copied: {control}");
}
