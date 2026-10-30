//! Round 11 area W: a walker crossing a two-lane road mid-block from the far side, timed as
//! SUMO times an approaching person (`perception::walk_across`): the car in the near lane
//! brakes for them at its comfortable deceleration as soon as their path meets its window, and
//! goes on when its rear clears their line before they reach its strip.

use glam::{DVec2, DVec3};
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network};
use omsi_traffic::{Person, Simulation, Vehicle, VehicleClass, Why};

const DT: f32 = 1.0 / 30.0;

/// Two eastbound lanes 3.5 m wide: 0 the left one at y = 0, 1 the right one at y = -3.5.
fn two_lanes() -> Network {
    let mut net = Network::default();
    for (k, y) in [0.0, -3.5].into_iter().enumerate() {
        let mut l = LaneBuilder::polyline(vec![DVec3::new(0.0, y, 0.0), DVec3::new(600.0, y, 0.0)], LaneKind::Street, 3.5);
        l.source = 1;
        l.key = Some(LaneKey { tile: (0, 0), id: 1, path: k as u16 });
        l.offset = -y as f32;
        net.lanes.push(l);
    }
    net.link(1.5);
    net.compute_conflicts();
    net
}

/// A sedan (4.62 x 1.94 m) on the right lane at `s` m with `speed` m/s, wanting 45 km/h.
fn sedan(s: f32, speed: f32) -> Vehicle {
    let mut st = AiState::new(1, s, 1);
    st.speed = speed;
    st.front = 2.42;
    st.rear = 2.2;
    st.length = 4.62;
    st.max_speed_kmh = speed * 3.6;
    st.veh_type = 0;
    let mut v = Vehicle::new(1, VehicleClass::Car, st, 0.97, 7919);
    v.fresh = 0.0;
    v
}

fn walker(x: f64, y: f64, vel: DVec2) -> Person {
    Person { pos: DVec3::new(x, y, 0.0), vel, on_crossing: false, still: 0.0, waiting_for_bus: false, at_kerb: false, waits_at: None, id: None, approach: None }
}

struct Seen {
    /// At the first frame the people test held the car: its speed and the road from its front
    /// to the walker's line.
    first: Option<(f32, f32)>,
    least_speed: f32,
    /// The least separation of the car's body from the walker's (m).
    least_sep: f32,
    /// The hardest the car braked (m/s^2).
    hardest: f32,
}

/// The sedan rolls for `secs` with one walker stepping at `vel` from `from` (their line across
/// the road at x = `from.x`).
fn drive(me: Vehicle, from: DVec2, vel: DVec2, secs: f32) -> Seen {
    let net = two_lanes();
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![me];
    let mut p = walker(from.x, from.y, vel);
    let mut seen = Seen { first: None, least_speed: f32::INFINITY, least_sep: f32::INFINITY, hardest: 0.0 };
    let mut t = 0.0;
    while t < secs {
        sim.people = vec![p.clone()];
        let before = cars[0].state.speed;
        sim.tick(&net, DT, &mut cars);
        let me = &cars[0];
        seen.hardest = seen.hardest.max((before - me.state.speed) / DT);
        seen.least_speed = seen.least_speed.min(me.state.speed);
        seen.least_sep = seen.least_sep.min(me.obb().separation(&omsi_sim::collision::Obb::point(p.pos, 0.28)) as f32);
        if seen.first.is_none() && me.last.why == Why::People {
            seen.first = Some((before, (from.x - me.pos.x) as f32 - me.state.front));
        }
        p.pos += (vel * DT as f64).extend(0.0);
        t += DT;
    }
    seen
}

#[test]
fn w1_a_walker_crossing_from_the_far_side_is_braked_for_at_the_comfortable_deceleration() {
    // viz2 case08 #19609: 12.3 m/s in the right lane; a walker steps across both lanes at
    // 1.2 m/s from beyond the left one. Their path enters the car's strip (half width and 0.3 m)
    // 3.9 s on, while the car would be on their line: it brakes from where it can still stop
    // short at its comfortable deceleration and never touches them
    let v0 = 12.3;
    let me = sedan(100.0, v0);
    let comfort = me.params.comfort_brake;
    let seen = drive(me, DVec2::new(100.0 + 2.42 + 60.0, 2.5), DVec2::new(0.0, -1.2), 12.0);
    let (v, gap) = seen.first.expect("never braked for the walker crossing its way");
    assert!(gap - 1.5 >= v * v / (2.0 * comfort), "first held {gap:.1} m before their line at {v:.1} m/s: inside its comfortable stop ({:.1} m)", v * v / (2.0 * comfort));
    assert!(seen.least_sep > 0.0, "touched the walker: {:.2} m", seen.least_sep);
    assert!(seen.hardest <= comfort, "braked {:.2} m/s^2 (comfortable {comfort:.2})", seen.hardest);
}

#[test]
fn w1_a_car_whose_rear_clears_the_walkers_line_before_they_reach_its_strip_goes_on() {
    // the walker in the left lane's middle (3.5 m off the car's axis) stepping across at
    // 1.2 m/s, 13 m ahead of the car's front: they reach its strip in 1.9 s, its rear clears
    // their line in 1.5 s. It passes first, not stopping late inside their path
    let v0 = 12.3;
    let seen = drive(sedan(100.0, v0), DVec2::new(100.0 + 2.42 + 13.0, 0.0), DVec2::new(0.0, -1.2), 3.0);
    assert!(seen.least_speed > 0.9 * v0, "braked to {:.1} m/s for a walker it passes first", seen.least_speed);
    assert!(seen.least_sep > 0.0, "touched the walker: {:.2} m", seen.least_sep);
}

#[test]
fn w1_a_slow_car_waits_for_a_walker_crossing_ahead() {
    // at 3 m/s, a walker 2 m off its strip crossing 10 m ahead: it reaches their line after
    // they step in, and stops short of them
    let seen = drive(sedan(100.0, 3.0), DVec2::new(100.0 + 2.42 + 10.0, -3.5 + 3.3), DVec2::new(0.0, -1.2), 6.0);
    assert!(seen.first.is_some(), "never held for the walker");
    assert!(seen.least_sep > 0.0, "touched the walker: {:.2} m", seen.least_sep);
}
