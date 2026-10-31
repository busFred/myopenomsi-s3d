//! Round 11 area R: a car files approaches (`Claim::foe_only`) at the junctions after its first
//! within `perception::ENTRY_SIGHT_TIME` of its speed, and none while it was refused at its first
//! junction and can still stop there (`sim::files_approaches`).
use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network};
use omsi_traffic::{Simulation, Vehicle, VehicleClass};

fn lane(pts: &[(f64, f64)], source: u32, id: i64, path: u16, priority: f32) -> omsi_sim::traffic::Lane {
    let mut l = LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), LaneKind::Street, 3.0);
    l.source = source;
    l.key = Some(LaneKey { tile: (0, 0), id, path });
    l.priority = priority;
    l
}

/// A straight road through two merge objects 20 m apart: J1 (object 9) with a side road coming in
/// at (0, 0), J2 (object 10) with one at (40, 0).
fn two_merges() -> Network {
    let mut net = Network::default();
    net.lanes.push(lane(&[(-300.0, 0.0), (-20.0, 0.0)], 1, 1, 0, 0.0)); // 0
    net.lanes.push(lane(&[(-20.0, 0.0), (0.0, 0.0)], 2, 9, 0, 192.0)); // 1 J1 main
    net.lanes.push(lane(&[(-20.0, -12.0), (0.0, 0.0)], 2, 9, 1, 64.0)); // 2 J1 side
    net.lanes.push(lane(&[(0.0, 0.0), (20.0, 0.0)], 1, 2, 0, 0.0)); // 3
    net.lanes.push(lane(&[(20.0, 0.0), (40.0, 0.0)], 2, 10, 0, 192.0)); // 4 J2 main
    net.lanes.push(lane(&[(20.0, -12.0), (40.0, 0.0)], 2, 10, 1, 64.0)); // 5 J2 side
    net.lanes.push(lane(&[(40.0, 0.0), (400.0, 0.0)], 1, 3, 0, 0.0)); // 6
    net.lanes.push(lane(&[(-120.0, -72.0), (-20.0, -12.0)], 1, 4, 0, 0.0)); // 7 to J1 side
    net.lanes.push(lane(&[(-80.0, -72.0), (20.0, -12.0)], 1, 5, 0, 0.0)); // 8 to J2 side
    net.link(1.5);
    net.compute_conflicts();
    net
}

fn car(id: u64, lane: usize, s: f32, speed: f32) -> Vehicle {
    let mut st = AiState::new(lane, s, id);
    st.speed = speed;
    st.front = 2.2;
    st.rear = 2.2;
    st.length = 4.4;
    st.max_speed_kmh = 100.0;
    st.veh_type = 0;
    let mut v = Vehicle::new(id, VehicleClass::Car, st, 0.9, id * 7919);
    v.fresh = 0.0;
    v
}

#[test]
fn r11r_a_car_files_an_approach_at_the_next_junction_unless_refused_at_its_first() {
    let net = two_merges();
    let mut sim = Simulation::new(&net, vec![]);
    assert_eq!(sim.junctions.nodes.len(), 2);
    // 13.9 m/s, its front ~26 m before J1's object and ~60 m before J2's merge: J2 within 5 s x 13.9 m/s
    let mut cars = vec![car(1, 0, 252.0, 13.9)];
    let mut filed = Vec::new();
    for _ in 0..30 {
        sim.tick(&net, 1.0 / 30.0, &mut cars);
        filed.push((cars[0].state.lane, cars[0].state.s, sim.approaches_filed, sim.claims.len()));
    }
    assert!(filed.iter().any(|f| f.2 >= 1), "no approach filed at J2: {filed:?}");
    // refused at J1 last frame and still able to stop: none, the frame after one that filed
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![car(1, 0, 252.0, 13.9)];
    for _ in 0..30 {
        sim.tick(&net, 1.0 / 30.0, &mut cars);
        if sim.approaches_filed >= 1 {
            break;
        }
    }
    assert!(sim.approaches_filed >= 1);
    cars[0].refused_last = true;
    sim.tick(&net, 1.0 / 30.0, &mut cars);
    assert_eq!(sim.approaches_filed, 0);
}
