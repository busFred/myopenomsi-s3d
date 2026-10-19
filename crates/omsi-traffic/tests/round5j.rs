//! Headless scenarios for round 5 area J (junctions and signals): detector lights, the parked
//! player's claim, the dice at a mutual hold, a queue rolling up behind a car that waits for
//! me, box entry, the side-merge hold, and the yielding merger's window.

use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network, TrafficLightController};
#[allow(unused_imports)]
use omsi_traffic::sim::PLAYER_ID_BASE;
#[allow(unused_imports)]
use omsi_traffic::{PlayerView, Simulation, Vehicle, VehicleClass, Why};

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
fn piece(pts: &[(f64, f64)], path: u16, priority: f32) -> omsi_sim::traffic::Lane {
    let mut l = lane(pts, LaneKind::Street, 3.0);
    l.source = 2;
    l.key = Some(LaneKey { tile: (0, 0), id: 9, path });
    l.priority = priority;
    l
}

fn finish(mut net: Network) -> Network {
    net.link(1.5);
    net.compute_conflicts();
    net
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

/// Run `secs`; `each` sees every frame. Returns the smallest gap between two bodies.
#[allow(dead_code)]
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

// ---- J1: detector lights ----

/// X10's Buergertreff approach (case04): the approach road (0), a 1 m piece with light 0 (1),
/// the 9.7 m piece with light 1 whose line is the real stop line (2), the box (3), the exit (4).
fn detector_approach() -> Network {
    let mut net = Network::default();
    net.lanes.push(road(&[(-300.0, 0.0), (-11.0, 0.0)], 1));
    let mut loop_lane = piece(&[(-11.0, 0.0), (-10.0, 0.0)], 0, 128.0);
    loop_lane.traffic_light = Some((0, 0));
    net.lanes.push(loop_lane);
    let mut line = piece(&[(-10.0, 0.0), (-0.3, 0.0)], 1, 128.0);
    line.traffic_light = Some((0, 1));
    net.lanes.push(line);
    net.lanes.push(piece(&[(-0.3, 0.0), (20.0, 0.0)], 2, 128.0));
    net.lanes.push(road(&[(20.0, 0.0), (300.0, 0.0)], 2));
    finish(net)
}

/// Light 0 with `program`, light 1 red for good: where the car stops (its front's x), whether
/// the request of light 0 was ever raised, and the first signal the car saw from 60 m out.
fn runs_up_to(program: Vec<(i32, f32)>) -> (f32, bool, Option<(usize, usize)>) {
    let net = detector_approach();
    let ctl = TrafficLightController::new(vec![program, vec![(0, 100.0)]], 100.0);
    let mut sim = Simulation::new(&net, vec![ctl]);
    let mut cars = vec![car(1, 0, 230.0, 10.0)];
    let mut requested = false;
    let mut first_seen: Option<(usize, usize)> = None;
    run(&net, &mut sim, &mut cars, 20.0, |_, sim, cars| {
        requested |= sim.lights[0].request.first().copied().unwrap_or(false);
        if first_seen.is_none() && cars[0].pos.x > -60.0 {
            let p = omsi_traffic::perception::perceive(&net, sim, cars, 0);
            first_seen = p.signal.map(|s| (s.ctl, s.light));
        }
    });
    (front_x(&cars[0]), requested, first_seen)
}

#[test]
fn j1_a_constant_green_detector_before_the_line_hides_no_red_and_keeps_its_request() {
    // the Loop light: one phase, green, 1 s (X10 Buergertreff 0.13)
    let (front, requested, seen) = runs_up_to(vec![(6, 1.0)]);
    assert_eq!(seen, Some((0, 1)), "from 60 m out the car sees the real line, not the detector");
    assert!(front <= -10.0 + 0.3, "stood with its front at x = {front:.1}, over the red line at -10 (was: at -0.8 with the detector green in front of it)");
    assert!(front > -16.0, "stood at x = {front:.1}: far short of the line");
    assert!(requested, "the detector's request was never raised");
}

#[test]
fn j1_a_multi_phase_light_held_green_by_a_stop_point_is_a_stop_line_not_a_detector() {
    // a `[traffic_light_stop]`-held green: two phases, the clock held in the green by a stop
    // point until a request comes (the program's green lasts, but it is a real signal)
    let net = detector_approach();
    let mut ctl = TrafficLightController::new(vec![vec![(6, 10.0), (0, 90.0)], vec![(0, 100.0)]], 100.0);
    ctl.stops.push(omsi_sim::traffic::LightStop { light: 0, time: 5.0, if_request: false, jump_to: None });
    let sim = Simulation::new(&net, vec![ctl]);
    assert!(!omsi_traffic::perception::detector_light(&sim, 0, 0));
    assert!(omsi_traffic::perception::detector_light(&sim, 0, 1) == false, "a constant red is no detector");
    // a dark lane (state 12, one phase) is no detector either: it is reported as before
    let dark = Simulation::new(&net, vec![TrafficLightController::new(vec![vec![(12, 1.0)], vec![(0, 100.0)]], 100.0)]);
    assert!(!omsi_traffic::perception::detector_light(&dark, 0, 0));
    // and the car under a two-phase green first line runs up to the second as before: the
    // second light of the same object on a straight run is the line it has passed (leftarrow-1)
    let (front, _, seen) = runs_up_to(vec![(6, 10.0), (0, 90.0)]);
    assert_eq!(seen, Some((0, 0)), "a real first light is the one seen");
    assert!(front > -10.0, "with a real green first line it drives on past the second (same object, straight): front {front:.1}");
}
