//! Round 8 area H8 (junction and compile hazards).
//! `cargo test --release -p omsi-traffic --test round8h -- --nocapture`

use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network};
use omsi_traffic::{Simulation, Vehicle, VehicleClass};

const DT: f32 = 1.0 / 30.0;

fn road(from: (f64, f64), to: (f64, f64), id: i64, path: u16) -> omsi_sim::traffic::Lane {
    let mut l = LaneBuilder::polyline(vec![DVec3::new(from.0, from.1, 0.0), DVec3::new(to.0, to.1, 0.0)], LaneKind::Street, 3.5);
    l.source = 1;
    l.key = Some(LaneKey { tile: (0, 0), id, path });
    l
}

fn car(id: u64, lane: usize, s: f32, speed: f32) -> Vehicle {
    let mut st = AiState::new(lane, s, id);
    st.speed = speed;
    st.front = 2.0;
    st.rear = 2.0;
    st.length = 4.0;
    st.max_speed_kmh = 50.0;
    st.veh_type = 0;
    st.change_cooldown = 0.0;
    let mut v = Vehicle::new(id, VehicleClass::Car, st, 0.9, id * 7919);
    v.fresh = 0.0;
    v
}

/// One straight lane along +y (heading 0) from y = 0 to y = 600.
fn one_lane() -> Network {
    let mut net = Network::default();
    net.lanes.push(road((0.0, 0.0), (0.0, 600.0), 1, 0));
    net.link(1.5);
    net.compute_conflicts();
    net.compute_reach();
    net
}

/// A parked car leaving its space as the app puts it there (`Traffic::pull_out_parked`): on its
/// lane at `s`, `lat` m to the right, its way ramped back to the axis over 8-16 m, held for its
/// `pull_out` seconds, indicating.
fn pulled_out(id: u64, s: f32, lat: f32) -> Vehicle {
    let mut c = car(id, 0, s, 0.0);
    c.state.lateral = lat;
    c.state.lateral_target = 0.0;
    c.state.lateral_ramp = (lat, 0.0, c.state.odometer, (lat * 6.0).clamp(8.0, 16.0));
    c.pull_out = 2.0;
    c.state.blinker = 1;
    c
}

/// Run `secs`; before each frame `before(t, sim)` (the people), after it `each(t, sim, cars)`.
/// The least separation of any two boxes.
fn run(net: &Network, sim: &mut Simulation, cars: &mut Vec<Vehicle>, secs: f32, mut before: impl FnMut(f32, &mut Simulation), mut each: impl FnMut(f32, &Simulation, &[Vehicle])) -> f32 {
    let mut least = f32::INFINITY;
    let mut t = 0.0;
    sim.update_positions(net, cars);
    while t < secs {
        before(t, sim);
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
        each(t, sim, cars);
        t += DT;
    }
    least
}

fn nobody(_: f32, _: &mut Simulation) {}

/// [H8-1] SUMO's parked vehicle re-enters by insertion, which checks the follower's secure gap
/// on the lane it enters: a car leaving its space with a car coming up its lane at 12 m/s
/// 40 m behind (inside the gap a change needs from it) stays parked until that car has gone
/// by, then leaves; nobody brakes hard for it and nobody touches.
#[test]
fn h8_1_a_pull_out_waits_for_the_follower_in_its_lane() {
    let net = one_lane();
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![pulled_out(1, 100.0, 3.0), car(2, 0, 60.0, 12.0)];
    let (mut moved_before_passed, mut moved, mut hardest) = (false, false, 0.0f32);
    let least = run(&net, &mut sim, &mut cars, 20.0, nobody, |_, _, cars| {
        let (Some(p), Some(f)) = (cars.iter().find(|c| c.id == 1), cars.iter().find(|c| c.id == 2)) else { return };
        let moving = p.state.speed > 0.3;
        moved |= moving;
        moved_before_passed |= moving && f.state.s < p.state.s;
        hardest = hardest.max(-f.state.acc);
    });
    assert!(!moved_before_passed, "pulled out in front of the car coming up its lane");
    assert!(moved, "never left its space");
    assert!(hardest < 1.0, "the follower braked {hardest:.1} m/s² for it");
    assert!(least > 0.0, "touched: {least:.2}");
}

/// [H8-1] The counter-case: the follower far back (200 m) leaves the gap: the car leaves its
/// space when its `pull_out` has run, as before.
#[test]
fn h8_1_a_pull_out_with_room_behind_leaves_on_time() {
    let net = one_lane();
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![pulled_out(1, 300.0, 3.0), car(2, 0, 100.0, 12.0)];
    let mut left_at = None;
    run(&net, &mut sim, &mut cars, 8.0, nobody, |t, _, cars| {
        if left_at.is_none() && cars.iter().any(|c| c.id == 1 && c.state.speed > 0.3) {
            left_at = Some(t);
        }
    });
    let t = left_at.expect("never left its space");
    assert!(t < 3.0, "left at {t:.1} s, its pull_out is 2 s");
}
