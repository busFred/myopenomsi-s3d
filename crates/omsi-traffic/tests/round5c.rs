//! Headless scenarios for round 5 area C (cars versus people): who on a crossing holds a car
//! at the zebra, who beside a standing car holds it, and where a car queues up behind a
//! lead standing past a crossing.

use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network};
use omsi_traffic::{Person, Simulation, Vehicle, VehicleClass, Why};

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
fn piece(pts: &[(f64, f64)], path: u16, kind: LaneKind) -> omsi_sim::traffic::Lane {
    let mut l = lane(pts, kind);
    l.source = 2;
    l.key = Some(LaneKey { tile: (0, 0), id: 9, path });
    l.priority = 128.0;
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
    net.build_grid();
    net.compute_conflicts();
    net
}

fn run(net: &Network, sim: &mut Simulation, vehicles: &mut Vec<Vehicle>, secs: f32, mut each: impl FnMut(f32, &mut Simulation, &mut [Vehicle])) {
    let mut t = 0.0;
    while t < secs {
        each(t, sim, vehicles.as_mut_slice());
        sim.tick(net, DT, vehicles);
        for &i in &sim.removed {
            vehicles.remove(i);
        }
        t += DT;
    }
}

/// The front bumper's x of a car heading east.
fn front_x(v: &Vehicle) -> f32 {
    v.pos.x as f32 + v.state.front
}

/// An eastbound road (lane 0, x -300..-15) into object 9's piece (lane 1, x -15..45, 3.5 m
/// wide) and on (lane 2); `zebra`: a crossing of the object at x = 23 (lane 3, y -8..8).
fn road_with_a_zebra(zebra: bool) -> Network {
    let mut net = Network::default();
    net.lanes.push(road(&[(-300.0, 0.0), (-15.0, 0.0)], 1));
    net.lanes.push(piece(&[(-15.0, 0.0), (45.0, 0.0)], 0, LaneKind::Street));
    net.lanes.push(road(&[(45.0, 0.0), (400.0, 0.0)], 2));
    if zebra {
        net.lanes.push(piece(&[(23.0, -8.0), (23.0, 8.0)], 3, LaneKind::Sidewalk));
    }
    finish(net)
}

fn person(x: f64, y: f64) -> Person {
    Person { pos: DVec3::new(x, y, 0.0), vel: glam::DVec2::ZERO, on_crossing: false, still: 0.0, waiting_for_bus: false, at_kerb: false }
}

// C1 ----------------------------------------------------------------------------------

/// [round5 C1] A walker on the crossing 2.8 m or a lane width from my axis (the far half of
/// the zebra; the old rule held for anyone within 3 m along it) holds nobody at the zebra;
/// one within my lane's footprint and half a metre (2.25 m) stops me 2.5 m short of it.
#[test]
fn c1_a_walker_on_the_far_half_of_the_zebra_holds_nobody_one_in_my_lane_does() {
    let net = road_with_a_zebra(true);
    assert!(net.walks[1].iter().any(|&(w, _, _)| w == 3), "the zebra cuts the piece: {:?}", net.walks[1]);
    for (y, holds) in [(-3.5f32, false), (-2.8, false), (2.8, false), (-1.5, true), (1.5, true), (-2.1, true)] {
        let mut sim = Simulation::new(&net, vec![]);
        // (the zebra runs south to north from y = -8: the walker `y + 8` along it)
        sim.walkers = vec![(3, y + 8.0)];
        let mut cars = vec![car(1, 0, 230.0, 10.0)];
        let mut zebra_s = 0.0f32;
        let mut stood_at: Option<f32> = None;
        run(&net, &mut sim, &mut cars, 25.0, |_, _, cars| {
            let me = &cars[0];
            if me.last.why == Why::Zebra {
                zebra_s += DT;
                if me.state.speed < 0.1 && stood_at.is_none() {
                    stood_at = Some(front_x(me));
                }
            }
        });
        if holds {
            let at = stood_at.unwrap_or_else(|| panic!("walker at y {y}: never stood for the zebra ({zebra_s:.1} s why=zebra)"));
            assert!(at > 18.0 && at <= 23.0 - 2.5 + 0.2, "walker at y {y}: stood with its front at x = {at:.1}");
        } else {
            assert!(zebra_s < 0.5, "walker at y {y}: held {zebra_s:.1} s for the zebra");
            assert!(front_x(&cars[0]) > 100.0, "walker at y {y}: did not pass: front at {:.1}", front_x(&cars[0]));
        }
    }
}
