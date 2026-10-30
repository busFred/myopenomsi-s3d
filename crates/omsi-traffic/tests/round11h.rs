//! Round 11 area H2: an AI vehicle that drives the player's vehicle (`StandIn`, the offscreen
//! harness's `--follow-line`) is seen by the others as the player - its player claim, its
//! box - and not as an AI vehicle besides; it does not yield to itself, its player's claim is
//! filed as a user's (from the indicator) and it drives by the grant of its own claim, which
//! only it reads (resolved against the others' claims, its player's left out). `cargo test -p omsi-traffic --test round11h -- --nocapture`.

use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network};
use omsi_traffic::sim::{player_claim, PLAYER_ID_BASE};
use omsi_traffic::{PlayerView, Simulation, StandIn, Vehicle, VehicleClass, Why};

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

/// A main road west-east (lanes 0, 1, 2) with priority over a side road south-north (3, 4, 5),
/// crossing at the origin (as `adversarial.rs`' `priority_junction`).
fn priority_junction() -> Network {
    let mut net = Network::default();
    net.lanes.push(road((-200.0, 0.0), (-15.0, 0.0), 1));
    net.lanes.push(piece((-15.0, 0.0), (15.0, 0.0), 9, 0, 192.0));
    net.lanes.push(road((15.0, 0.0), (1500.0, 0.0), 2));
    net.lanes.push(road((0.0, -200.0), (0.0, -15.0), 3));
    net.lanes.push(piece((0.0, -15.0), (0.0, 15.0), 9, 1, 64.0));
    net.lanes.push(road((0.0, 15.0), (0.0, 1500.0), 4));
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

/// The player's vehicle put where the stand-in is, as the app gives it to the traffic: its box
/// (centre, heading, half extents), its speed, the indicator of its driver.
fn player_of(v: &Vehicle, still: f32) -> PlayerView {
    let h = (v.heading as f64).to_radians();
    let fwd = DVec3::new(h.sin(), h.cos(), 0.0);
    let centre = v.pos + fwd * ((v.state.front - v.state.rear) * 0.5) as f64;
    PlayerView { id: 0, pos: centre, heading: v.heading, half_len: (v.state.front + v.state.rear) * 0.5, half_width: v.half_width, speed: v.state.speed, still, blinker: v.state.blinker as u8, signalling: 0.0, priority: false }
}

struct Seen {
    /// Frames the other car waited for the player's claim, and for the stand-in as an AI vehicle.
    other_waits_player: usize,
    other_waits_vehicle: usize,
    /// Frames the stand-in waited for its own player claim; for the other car.
    own_waits_self: usize,
    own_waits_other: usize,
    /// Frames with the player's claim filed, and with a claim of the stand-in's own index.
    player_claims: usize,
    own_claims: usize,
    /// The nearest the two bodies came (m), the stand-in's lowest speed, the other's.
    nearest: f32,
    own_min_speed: f32,
    other_min_speed: f32,
    /// Frames the player's claim (filed from the indicator, as a user's) and the stand-in's
    /// own named another junction or way, or only one was filed (`stand_in_unmatched`).
    unmatched: u32,
}

/// Run the stand-in `own` (index 0) and the other car (index 1) for `secs`, the player's view
/// built from the stand-in every frame (`stand_in`; else `own` is a plain AI vehicle and there
/// is no player).
fn run(net: &Network, mut vehicles: Vec<Vehicle>, secs: f32, stand_in: bool) -> Seen {
    let mut sim = Simulation::new(net, vec![]);
    let own_id = vehicles[0].id;
    let other_id = vehicles[1].id;
    sim.stand_in = stand_in.then_some(StandIn { vehicle: own_id, player: 0 });
    let mut seen = Seen { other_waits_player: 0, other_waits_vehicle: 0, own_waits_self: 0, own_waits_other: 0, player_claims: 0, own_claims: 0, nearest: f32::INFINITY, own_min_speed: f32::INFINITY, other_min_speed: f32::INFINITY, unmatched: 0 };
    let mut still = 0.0f32;
    let mut t = 0.0;
    // (the pose the first view is built from)
    sim.update_positions(net, &mut vehicles);
    while t < secs {
        let own = vehicles.iter().find(|v| v.id == own_id).expect("the stand-in is on the road");
        still = if own.state.speed.abs() < 0.3 { still + DT } else { 0.0 };
        sim.players = if stand_in { vec![player_of(own, still)] } else { Vec::new() };
        sim.tick(net, DT, &mut vehicles);
        assert!(sim.removed.is_empty(), "nobody leaves the road here");
        let own = vehicles.iter().find(|v| v.id == own_id).unwrap();
        let other = vehicles.iter().find(|v| v.id == other_id).unwrap();
        seen.other_waits_player += usize::from(other.last.wait_for == Some(PLAYER_ID_BASE));
        seen.other_waits_vehicle += usize::from(other.last.wait_for == Some(own_id));
        seen.own_waits_self += usize::from(own.last.wait_for == Some(PLAYER_ID_BASE));
        seen.own_waits_other += usize::from(own.last.wait_for == Some(other_id) && own.last.why == Why::Junction);
        seen.player_claims += usize::from(sim.claims.iter().any(|(c, _)| c.vehicle == player_claim(0)));
        seen.own_claims += usize::from(sim.claims.iter().any(|(c, _)| c.vehicle == 0));
        seen.nearest = seen.nearest.min(own.obb().separation(&other.obb()) as f32);
        // (only while the two are still to meet: past the junction both may stop at nothing)
        if own.pos.length() < 60.0 {
            seen.own_min_speed = seen.own_min_speed.min(own.state.speed);
        }
        if other.pos.length() < 60.0 {
            seen.other_min_speed = seen.other_min_speed.min(other.state.speed);
        }
        t += DT;
    }
    seen.unmatched = sim.stand_in_unmatched;
    seen
}

/// The stand-in on the main road: the side road car gives way to it as it gives way to the
/// player's bus (`adversarial.rs` p7) - waiting for the player's claim, never for an AI
/// vehicle - and the stand-in, with its own player's claim let go, drives through without
/// waiting for itself; nobody sees a claim of its own.
#[test]
fn h2_a_side_road_car_gives_way_to_the_stand_in_as_to_the_player() {
    let net = priority_junction();
    // (the bus reaches the middle at ~5.5 s, the car would at ~5 s)
    let vehicles = vec![bus(1, 0, net.lanes[0].length() - 45.0, 11.0), car(2, 3, net.lanes[3].length() - 40.0, 10.0)];
    let plain = run(&net, vehicles.clone(), 15.0, false);
    assert!(plain.other_waits_vehicle > 0 && plain.own_claims > 0, "as a plain AI bus it is waited for as one");
    let s = run(&net, vehicles, 15.0, true);
    eprintln!(
        "h2 main: other waits for the player {} / the AI bus {} frames, own waits for itself {} / the car {}, player claims {} own {}, nearest {:.2} m, own min speed {:.1}, other min speed {:.1}, claims apart {} frames",
        s.other_waits_player, s.other_waits_vehicle, s.own_waits_self, s.own_waits_other, s.player_claims, s.own_claims, s.nearest, s.own_min_speed, s.other_min_speed, s.unmatched
    );
    assert!(s.other_waits_player > 0, "the side road car gave way to the player's claim");
    assert_eq!(s.other_waits_vehicle, 0, "the stand-in is no AI vehicle to the others");
    assert_eq!(s.own_waits_self, 0, "the stand-in never waits for itself");
    assert!(s.player_claims > 0, "the player's claim is filed");
    assert_eq!(s.own_claims, 0, "no AI claim of the stand-in's own");
    assert!(s.own_min_speed > 8.0, "the stand-in went through on its right of way: {:.1}", s.own_min_speed);
    assert!(s.other_min_speed < 8.0, "the side road car slowed for it: {:.1}", s.other_min_speed);
    assert!(s.nearest > 0.5, "the two met: {:.2} m", s.nearest);
}

/// The stand-in on the side road: it gives way to the car on the main road by the grant of its
/// own claim and the car does not wait for it.
#[test]
fn h2_the_stand_in_gives_way_by_its_own_grant() {
    let net = priority_junction();
    let vehicles = vec![bus(1, 3, net.lanes[3].length() - 45.0, 10.0), car(2, 0, net.lanes[0].length() - 40.0, 11.0)];
    let s = run(&net, vehicles, 15.0, true);
    eprintln!(
        "h2 side: other waits for the player {} / the AI bus {} frames, own waits for itself {} / the car {}, player claims {} own {}, nearest {:.2} m, own min speed {:.1}, other min speed {:.1}, claims apart {} frames",
        s.other_waits_player, s.other_waits_vehicle, s.own_waits_self, s.own_waits_other, s.player_claims, s.own_claims, s.nearest, s.own_min_speed, s.other_min_speed, s.unmatched
    );
    assert!(s.own_waits_other > 0, "the stand-in waited for the car at the junction");
    assert_eq!(s.own_waits_self, 0, "the stand-in never waits for itself");
    assert_eq!(s.other_waits_player + s.other_waits_vehicle, 0, "the car on the main road waits for nobody");
    assert_eq!(s.own_claims, 0, "no AI claim of the stand-in's own among the others'");
    assert!(s.other_min_speed > 8.0, "the car went through: {:.1}", s.other_min_speed);
    assert!(s.nearest > 0.5, "the two met: {:.2} m", s.nearest);
}


/// The bus on the side road standing at its line while a car comes along the main road and is
/// stopped short of the junction by somebody standing in its lane (stuck: it is not coming while
/// they stand there): the second at which the bus's front enters the main road (its near edge,
/// 1.75 m south of the axis), as a plain AI bus and as the stand-in.
fn crossing_past_a_stuck_foe(stand_in: bool, secs: f32) -> Option<f32> {
    let net = priority_junction();
    let mut vehicles = vec![bus(1, 3, net.lanes[3].length() - 7.0, 0.0), car(2, 0, net.lanes[0].length() - 60.0, 9.0)];
    let mut sim = Simulation::new(&net, vec![]);
    sim.stand_in = stand_in.then_some(StandIn { vehicle: 1, player: 0 });
    // (in the main road's lane 21 m short of the crossing's middle, standing)
    sim.people = vec![omsi_traffic::sim::Person { pos: DVec3::new(-21.0, 0.0, 0.0), vel: glam::DVec2::ZERO, on_crossing: false, still: 30.0, waiting_for_bus: false, at_kerb: false, waits_at: None, id: Some(1), approach: None }];
    sim.update_positions(&net, &mut vehicles);
    let (mut t, mut still) = (0.0f32, 0.0f32);
    while t < secs {
        let own = vehicles.iter().find(|v| v.id == 1).expect("the bus is on the road");
        if own.pos.y + own.state.front as f64 > -1.75 {
            return Some(t);
        }
        still = if own.state.speed.abs() < 0.3 { still + DT } else { 0.0 };
        sim.players = if stand_in { vec![player_of(own, still)] } else { Vec::new() };
        sim.tick(&net, DT, &mut vehicles);
        t += DT;
    }
    let own = vehicles.iter().find(|v| v.id == 1).unwrap();
    let other = vehicles.iter().find(|v| v.id == 2).unwrap();
    eprintln!("  not across: bus lane {} s {:.1} why {:?} wait_for {:?}; car lane {} s {:.1} v {:.1} why {:?}", own.state.lane, own.state.s, own.last.why, own.last.wait_for, other.state.lane, other.state.s, other.state.speed, other.last.why);
    None
}

/// The stand-in waiting on the side road decides as the AI driver it is (its own claim, with
/// its own patience and why it stands): a main road car that people stop short of the junction
/// is not coming (`reservation`'s held-off foe), and the stand-in crosses as the plain AI bus
/// does. Driven by its player's grant (a user's state: standing over `PLAYER_STILL` is blocked,
/// which drops nobody; no impatience) it waited for that car for good.
#[test]
fn h2_the_stand_in_crosses_past_a_stuck_foe_as_the_ai_bus() {
    let plain = crossing_past_a_stuck_foe(false, 40.0);
    let own = crossing_past_a_stuck_foe(true, 40.0);
    eprintln!("h2 stuck foe: the plain AI bus enters at {plain:?} s, the stand-in at {own:?} s");
    let plain = plain.expect("the plain AI bus crosses in front of the stopped car");
    let own = own.expect("the stand-in crosses in front of the stopped car");
    assert!(own <= plain, "the stand-in no later than the plain AI bus: {own:.1} s against {plain:.1} s");
}

