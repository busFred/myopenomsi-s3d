//! Signals as part of the right of way, and the lit stubs of a signalled crossing (rule
//! changes once tried behind switches, `signal_foes` and `stub_links`, now the rules).

use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network, TrafficLightController};
use omsi_traffic::{Simulation, Vehicle, VehicleClass, Why};

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

// lanes of `beijing_junction`
const WEST_IN: usize = 1;
const WEST_STUB: usize = 2;
const WEST_EXIT: usize = 4;
const SOUTH_IN: usize = 6;
const SOUTH_STUB: usize = 7;
const SOUTH_THROUGH: usize = 8;

/// A crossing object as BeijingCity builds them: each way through is a lead-in piece, a
/// 1 m stub that carries the light, and the piece across; the lights stand 8 m inside
/// the object. West → east (light 0) and south → north (light 1) cross at the origin.
fn beijing_junction(west_prio: f32, south_prio: f32) -> Network {
    let mut net = Network::default();
    net.lanes.push(road((-200.0, 0.0), (-20.0, 0.0), 1)); // 0 west approach
    net.lanes.push(piece((-20.0, 0.0), (-12.0, 0.0), 0, west_prio)); // 1 lead-in
    let mut stub = piece((-12.0, 0.0), (-11.0, 0.0), 1, west_prio); // 2 lit stub
    stub.traffic_light = Some((0, 0));
    net.lanes.push(stub);
    net.lanes.push(piece((-11.0, 0.0), (15.0, 0.0), 2, west_prio)); // 3 across
    net.lanes.push(road((15.0, 0.0), (1500.0, 0.0), 2)); // 4 east exit
    net.lanes.push(road((0.0, -200.0), (0.0, -20.0), 3)); // 5 south approach
    net.lanes.push(piece((0.0, -20.0), (0.0, -12.0), 3, south_prio)); // 6 lead-in
    let mut stub = piece((0.0, -12.0), (0.0, -11.0), 4, south_prio); // 7 lit stub
    stub.traffic_light = Some((0, 1));
    net.lanes.push(stub);
    net.lanes.push(piece((0.0, -11.0), (0.0, 15.0), 5, south_prio)); // 8 across
    net.lanes.push(road((0.0, 15.0), (0.0, 1500.0), 4)); // 9 north exit
    // (joints of 0.5 m: at 1.5 m the lead-in would reach past the stub straight into the
    // piece across too, the way round the light that the lit stubs keep it from)
    net.link(0.5);
    net.compute_conflicts();
    net
}

/// `PROBE=1`: the cars once a second.
fn probe(t: f32, cars: &[Vehicle]) {
    if std::env::var_os("PROBE").is_some() && (t * 30.0).round() as i32 % 30 == 0 {
        let line: Vec<String> = cars.iter().map(|c| format!("[{} l{} s={:.1} v={:.1} why={:?} wait={:?} stop={:?}]", c.id, c.state.lane, c.state.s, c.state.speed, c.last.why, c.last.wait_for, c.last.stop)).collect();
        eprintln!("t={:.1} {}", t, line.join(" "));
    }
}

/// Every frame: (time, the simulation, the cars).
fn run(net: &Network, sim: &mut Simulation, cars: &mut [Vehicle], secs: f32, mut each: impl FnMut(f32, &Simulation, &[Vehicle])) -> f32 {
    let mut t = 0.0;
    let mut min_gap = f32::INFINITY;
    while t < secs {
        sim.tick(net, DT, cars);
        assert!(sim.removed.is_empty());
        for (a, va) in cars.iter().enumerate() {
            for vb in cars.iter().skip(a + 1) {
                min_gap = min_gap.min(va.obb().separation(&vb.obb()) as f32);
            }
        }
        each(t, sim, cars);
        t += DT;
    }
    min_gap
}

/// West is green throughout; south is red for `red` seconds, then green. A south car
/// stands at its red on the lead-in piece (inside the object), a west car comes up to the
/// junction: by right before left it gives way to the south car. Returns (seconds the first
/// west car waited for the south car while south was red, the first west car crossed before
/// south went green, the second west car waited for the south car once it was green, the
/// south car ran its red).
fn foe_at_a_red_on_a_stub() -> (f32, bool, bool, bool) {
    foe_at_a_red_on_a_stub_in(beijing_junction(128.0, 128.0))
}

/// `beijing_junction` with an unlit 0.3 m stub (10) beside the south lit stub, from the
/// same place at the end of the lead-in (BeijingCity case04 1102/1409/1410, case07
/// 5233/5109/5110): linked without the lit stubs, the lead-in merged with it a few metres
/// before the light.
fn beijing_junction_with_a_sibling_stub() -> Network {
    let mut net = beijing_junction(128.0, 128.0);
    net.lanes.push(piece((0.0, -12.0), (0.0, -11.7), 6, 128.0));
    net.link(0.5);
    net.compute_conflicts();
    net
}

fn foe_at_a_red_on_a_stub_in(net: Network) -> (f32, bool, bool, bool) {
    let js = omsi_traffic::Junctions::compile(&net);
    let (_, across) = js.movement(3).unwrap();
    assert!(across.foes.iter().any(|f| f.yields), "west gives way to south");
    let red = 20.0;
    let ctl = TrafficLightController::new(vec![vec![(6, 60.0)], vec![(0, red), (6, 60.0 - red)]], 60.0);
    let mut sim = Simulation::new(&net, vec![ctl]);
    // south: standing at its stub on the lead-in; west: one rolling up now, and one that
    // comes up as south's light goes green
    let mut late = car(3, 0, 0.0, 8.0);
    late.state.max_speed_kmh = 30.0;
    let mut cars = vec![car(1, SOUTH_IN, 5.5, 0.0), car(2, 0, 140.0, 10.0), late];
    // (the south car's way is through the lit stub)
    cars[0].state.planned_next = Some(SOUTH_STUB);
    cars[0].state.ahead = vec![SOUTH_THROUGH, 9];
    let (mut waited, mut crossed_on_red, mut waited_on_green, mut ran_red) = (0.0f32, false, false, false);
    run(&net, &mut sim, &mut cars, red + 12.0, |t, sim, cars| {
        let south_red = sim.lights[0].state(1) < 3;
        let south = &cars[0];
        if south_red && (south.state.lane == SOUTH_THROUGH || (south.state.lane == SOUTH_STUB && south.state.speed > 0.5)) {
            ran_red = true;
        }
        // the first west car comes up 20 s before south's green, the second (slow) one
        // as it turns green
        if south_red && cars[1].last.wait_for == Some(1) {
            waited += DT;
        }
        if !south_red && cars[2].last.wait_for == Some(1) {
            waited_on_green = true;
        }
        if south_red && cars[1].state.lane == WEST_EXIT {
            crossed_on_red = true;
        }
        probe(t, cars);
    });
    (waited, crossed_on_red, waited_on_green, ran_red)
}

#[test]
fn signal_foes_a_foe_at_its_red_on_a_stub_inside_the_object_does_not_hold_a_green_claimant() {
    let (waited, crossed, waited_on_green, ran_red) = foe_at_a_red_on_a_stub();
    assert!(!ran_red, "the south car ran its red");
    assert!(waited < 1.0, "the west car waited {waited:.1} s for a car standing at its red");
    assert!(crossed, "the west car did not get across while south was red");
    assert!(waited_on_green, "once its light is green the south car counts again");
}

#[test]
fn signal_foes_an_unlit_sibling_stub_at_the_lead_in_end_does_not_hide_the_light() {
    let net = beijing_junction_with_a_sibling_stub();
    // the lead-in and the sibling are pieces of one way (a fork), no merge before the light
    // (linked without the lit stubs, they merged there, and the merge was the line)
    let js = omsi_traffic::Junctions::compile(&net);
    let (_, lead_in) = js.movement(SOUTH_IN).unwrap();
    assert!(lead_in.conflict_start.is_infinite(), "{:?}", lead_in.foes);
    let (waited, crossed, waited_on_green, ran_red) = foe_at_a_red_on_a_stub_in(net);
    assert!(!ran_red, "the south car ran its red");
    assert!(waited < 1.0, "the west car waited {waited:.1} s for a car standing at its red");
    assert!(crossed && waited_on_green);
}

/// The south car gives way (lower priority) and stands at its red on the stub; a west car
/// crosses in front of it. When the foe is gone the junction lets the south car go, the
/// light does not.
fn claimant_at_its_own_red() -> (bool, bool, bool) {
    let net = beijing_junction(192.0, 64.0);
    let ctl = TrafficLightController::new(vec![vec![(6, 60.0)], vec![(0, 60.0)]], 60.0);
    let mut sim = Simulation::new(&net, vec![ctl]);
    let mut cars = vec![car(1, SOUTH_IN, 5.5, 0.0), car(2, 0, 150.0, 10.0)];
    let mut ran_red = false;
    run(&net, &mut sim, &mut cars, 25.0, |t, _, cars| {
        probe(t, cars);
        let south = &cars[0];
        if south.state.lane == SOUTH_THROUGH || (south.state.lane == SOUTH_STUB && south.state.speed > 0.5) {
            ran_red = true;
        }
    });
    let at_red = cars[0].state.lane == SOUTH_IN && cars[0].state.speed < 0.3 && cars[0].last.why == Why::Light;
    let west_gone = cars[1].state.lane == WEST_EXIT;
    (ran_red, at_red, west_gone)
}

#[test]
fn signal_foes_a_claimant_at_its_own_red_stays_held_when_its_foe_is_gone() {
    let (ran_red, at_red, west_gone) = claimant_at_its_own_red();
    assert!(west_gone, "the west car got across");
    assert!(!ran_red, "the south car ran its red once the west car was gone");
    assert!(at_red, "the south car stands at its red");
}

#[test]
fn the_stub_lanes_are_where_the_test_says() {
    let net = beijing_junction(128.0, 128.0);
    assert_eq!(net.lanes[WEST_STUB].traffic_light, Some((0, 0)));
    assert_eq!(net.lanes[SOUTH_STUB].traffic_light, Some((0, 1)));
    assert_eq!(net.lanes[WEST_IN].next, vec![WEST_STUB]);
    assert_eq!(net.lanes[SOUTH_IN].next, vec![SOUTH_STUB]);
}

/// `beijing_junction` linked as the game links (1.5 m): the lead-in would reach past its
/// 1 m stub into the piece across too, but for the lit stub.
fn beijing_junction_as_linked() -> Network {
    let mut net = beijing_junction(128.0, 128.0);
    net.link(1.5);
    net
}

/// West is red for 40 s; a west car comes up at 10 m/s. Returns, over 24 drivers (each its
/// own random pick at the fork), how many drove into the piece across on red, and how
/// many stood at the line (why=Light, the front within a metre of the stub).
fn red_runners() -> (usize, usize) {
    let net = beijing_junction_as_linked();
    let (mut ran, mut at_line) = (0, 0);
    for id in 1..=24u64 {
        let ctl = TrafficLightController::new(vec![vec![(0, 40.0), (6, 20.0)], vec![(6, 60.0)]], 60.0);
        let mut sim = Simulation::new(&net, vec![ctl]);
        let mut cars = vec![car(id, 0, 100.0, 10.0)];
        let mut crossed = false;
        run(&net, &mut sim, &mut cars, 30.0, |_, _, cars| {
            if cars[0].state.lane == 3 {
                crossed = true;
            }
        });
        let c = &cars[0];
        ran += crossed as usize;
        let to_stub = if c.state.lane == WEST_IN { net.lanes[WEST_IN].length() - c.state.s - c.state.front } else { f32::INFINITY };
        at_line += (c.state.speed < 0.3 && c.last.why == Why::Light && to_stub.abs() < 1.0) as usize;
    }
    (ran, at_line)
}

#[test]
fn stub_links_every_driver_stands_at_its_red() {
    let net = beijing_junction_as_linked();
    assert_eq!(net.lanes[WEST_IN].next, vec![WEST_STUB]);
    // the lead-in is no merge of its own stub: its first meeting place is the crossing
    let js = omsi_traffic::Junctions::compile(&net);
    let (_, lead_in) = js.movement(WEST_IN).unwrap();
    assert!(lead_in.conflict_start.is_infinite(), "{:?}", lead_in.foes);
    let (ran, at_line) = red_runners();
    assert_eq!(ran, 0);
    assert_eq!(at_line, 24);
}
