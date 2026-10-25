//! Round 10 area P: the crate's side of the people rules.

use glam::{DVec2, DVec3};
use omsi_sim::traffic::{LaneBuilder, LaneKind, Network};
use omsi_traffic::{Claim, Grant, Impatience, Junction, Junctions, Movement, Simulation};

/// A junction of one movement (lane 0, along x from 0 to 20 m, 3 m wide) whose crossing cuts
/// it 12 m in, and a claim on it with its origin `s` along the movement, front/rear 2 m.
fn sim_with(s: f32, speed: f32) -> (Network, Simulation) {
    sim_on(0, s, speed)
}

/// `sim_with` with the claim on movement `m`: 0 the lane the crossing cuts, 1 another arm of
/// the junction (lane 1, along y at x = 30, which the crossing does not cut).
fn sim_on(m: usize, s: f32, speed: f32) -> (Network, Simulation) {
    let mut net = Network::default();
    net.lanes.push(LaneBuilder::polyline(vec![DVec3::ZERO, DVec3::new(20.0, 0.0, 0.0)], LaneKind::Street, 3.0));
    net.lanes.push(LaneBuilder::polyline(vec![DVec3::new(30.0, -20.0, 0.0), DVec3::new(30.0, 20.0, 0.0)], LaneKind::Street, 3.0));
    net.build_grid();
    let mut sim = Simulation::new(&net, Vec::new());
    let arm = |lane: usize| Movement {
        lane,
        kind: LaneKind::Street,
        entries: Vec::new(),
        exits: Vec::new(),
        continues: Vec::new(),
        light: None,
        foes: Vec::new(),
        conflict_end: 20.0,
        conflict_start: 0.0,
        converge: Vec::new(),
        converge_with: Vec::new(),
    };
    let mut js = Junctions::default();
    js.nodes.push(Junction { key: ((0, 0), 1), movements: vec![arm(0), arm(1)], crossings: Vec::new() });
    js.of_lane.insert(0, (0, 0));
    js.of_lane.insert(1, (0, 1));
    sim.junctions = js;
    let claim = Claim {
        yields_now: None,
        clear_ahead: f32::INFINITY,
        brake: 5.6,
        vehicle: 0,
        junction: 0,
        movement: m,
        s,
        speed,
        accel: 0.0,
        v_max: 13.0,
        front: 2.0,
        rear: 2.0,
        decel: 2.5,
        waiting: 0.0,
        lead_gap: None,
        zip_behind: None,
        body_block: None,
        next_wait: f32::NEG_INFINITY,
        tau: 1.3,
        min_gap: 2.0,
        red_at: f32::INFINITY,
        committed: false,
        exit_room: true,
        impatience: Impatience::default(),
        absolute: false,
        held: false,
        light_red: false,
        until_go: None,
        chain: vec![(m, 0.0)],
        lead: None,
        was_standing: false,
        blocked: false,
        lanes: Vec::new(),
        light: None,
        id: 77,
        stuck: false,
        clearing: false,
    };
    sim.claims = vec![(claim, Grant::go(0))];
    (net, sim)
}

/// The crossing's cut of lane 0, and a point of the walk beside the lane (no cut).
const CUT: DVec2 = DVec2::new(12.0, 0.0);
const BESIDE: DVec2 = DVec2::new(12.0, 2.0);

fn passing(s: f32, speed: f32, cuts: &[DVec2]) -> Vec<(u64, f32)> {
    let (net, sim) = sim_with(s, speed);
    sim.vehicles_passing_crossing(&net, cuts)
}

#[test]
fn d7_a_vehicle_that_cannot_stop_short_of_the_crossing_is_passing_it() {
    // SUMO addCrossingVehs on a prioritised crossing: front 2 m at s = 0, the cut at 12 m;
    // at 8 m/s its brake gap at 2.5 m/s^2 is 12.8 m: it reaches the cut, it passes
    let got = passing(0.0, 8.0, &[CUT]);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].0, 77);
    assert!((got[0].1 - 10.0).abs() < 1e-4, "{got:?}");
    // at 5 m/s (5 m) it can still brake for them: not listed
    assert!(passing(0.0, 5.0, &[CUT]).is_empty());
    // standing over the cut: the kerb's own rule, not listed
    assert!(passing(12.0, 0.0, &[CUT]).is_empty());
    // its rear past the cut: gone by
    assert!(passing(14.5, 8.0, &[CUT]).is_empty());
    // over it, moving: listed, the front past the cut
    let over = passing(12.0, 3.0, &[CUT]);
    assert!(over.len() == 1 && over[0].1 < 0.0, "{over:?}");
    // a point off the lane's footprint is no cut of its way
    assert!(passing(0.0, 8.0, &[BESIDE]).is_empty());
    // a vehicle on another arm of the junction, at speed: its way does not run over the cut
    let (net, sim) = sim_on(1, 0.0, 12.0);
    assert!(sim.vehicles_passing_crossing(&net, &[CUT]).is_empty());
}

// P9-2 ---------------------------------------------------------------------------------

const DT: f32 = 1.0 / 30.0;

fn lane(pts: &[(f64, f64)], kind: LaneKind) -> omsi_sim::traffic::Lane {
    LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), kind, 3.5)
}

/// An eastbound road (lane 0, x -300..-15, a spline lane) into crossing object 9's piece (lane 1,
/// x -15..45) and on (lane 2), with the object's zebra at x = 23 (lane 3, y -8..8).
fn road_with_a_zebra() -> Network {
    let road = |pts: &[(f64, f64)], id: i64| {
        let mut l = lane(pts, LaneKind::Street);
        l.source = 1;
        l.key = Some(omsi_sim::traffic::LaneKey { tile: (0, 0), id, path: 0 });
        l
    };
    let piece = |pts: &[(f64, f64)], path: u16, kind: LaneKind| {
        let mut l = lane(pts, kind);
        l.source = 2;
        l.key = Some(omsi_sim::traffic::LaneKey { tile: (0, 0), id: 9, path });
        l.priority = 128.0;
        l
    };
    let mut net = Network::default();
    net.lanes.push(road(&[(-300.0, 0.0), (-15.0, 0.0)], 1));
    net.lanes.push(piece(&[(-15.0, 0.0), (45.0, 0.0)], 0, LaneKind::Street));
    net.lanes.push(road(&[(45.0, 0.0), (400.0, 0.0)], 2));
    net.lanes.push(piece(&[(23.0, -8.0), (23.0, 8.0)], 3, LaneKind::Sidewalk));
    net.link(1.5);
    net.build_grid();
    net.compute_conflicts();
    net
}

fn car_at(front_x: f64, speed: f32) -> omsi_traffic::Vehicle {
    // (on the object's piece, lane 1, which starts at x = -15)
    let mut st = omsi_sim::traffic::AiState::new(1, (front_x - 2.0 + 15.0) as f32, 1);
    st.speed = speed;
    st.front = 2.0;
    st.rear = 2.0;
    st.length = 4.0;
    st.max_speed_kmh = 100.0;
    let mut v = omsi_traffic::Vehicle::new(1, omsi_traffic::VehicleClass::Car, st, 0.9, 7919);
    v.fresh = 0.0;
    v
}

/// The car with its front at `front_x` coming east at `speed` past somebody standing at the
/// zebra's south kerb who will step onto it in `arrive` s (and be across in `arrive + 12` s):
/// the furthest its front got in `secs`, and whether it stood for people.
fn past_an_approaching_person(front_x: f64, speed: f32, arrive: Option<f32>, secs: f32) -> (f64, bool) {
    let net = road_with_a_zebra();
    let mut sim = Simulation::new(&net, Vec::new());
    let mut cars = vec![car_at(front_x, speed)];
    let (mut far, mut stood) = (f64::MIN, false);
    let mut t = 0.0f32;
    while t < secs {
        let approach = arrive.map(|a| (DVec2::new(23.0, -8.0), (a - t).max(0.0), a + 12.0 - t));
        sim.people = vec![omsi_traffic::Person { pos: DVec3::new(23.0, -8.6, 0.0), vel: DVec2::new(0.0, 0.0), on_crossing: false, still: 0.0, waiting_for_bus: false, at_kerb: false, waits_at: None, id: Some(3), approach }];
        sim.tick(&net, DT, &mut cars);
        far = far.max(cars[0].pos.x + 2.0);
        stood |= cars[0].last.why == omsi_traffic::Why::People && cars[0].state.speed < 0.1;
        t += DT;
    }
    (far, stood)
}

#[test]
fn p92_a_car_is_held_short_of_the_zebra_for_a_person_about_to_step_onto_it() {
    // SUMO's approaching persons (MSLink::blockedAtTime): the person steps on in 1 s, the car 20 m
    // short at 8 m/s would be on the crossing in their window and can still brake: held at the
    // zebra stop (2.5 m before the cut at x = 23)
    let (far, stood) = past_an_approaching_person(0.0, 8.0, Some(1.0), 6.0);
    assert!(stood && far < 23.0 - 2.0, "front reached {far:.2}, stood {stood}");
    // nobody approaching: it drives on over the zebra
    let (far, _) = past_an_approaching_person(0.0, 8.0, None, 6.0);
    assert!(far > 30.0, "front reached {far:.2}");
    // stepping on only in 20 s, long after the car has gone over it: not held
    let (far, _) = past_an_approaching_person(0.0, 8.0, Some(20.0), 6.0);
    assert!(far > 30.0, "front reached {far:.2}");
}

#[test]
fn p92_a_far_kerb_approacher_on_a_wide_crossing_is_seen() {
    // SUMO registers approaching persons on the crossing's link whatever its width: one stepping
    // on 12 m across from my lane's cut (beyond jmCrossingGap), 16 m of crossing ahead of them,
    // still holds the car while their window overlaps its own
    let net = road_with_a_zebra();
    let mut sim = Simulation::new(&net, Vec::new());
    let mut cars = vec![car_at(0.0, 8.0)];
    let (mut far, mut t) = (f64::MIN, 0.0f32);
    while t < 6.0 {
        // (the crossing runs y -8..8; they step on at its far end, y = 12 being 12 m off my cut)
        let approach = Some((DVec2::new(23.0, 12.0), (1.0 - t).max(0.0), 1.0 + 16.0 / 1.3 - t));
        sim.people = vec![omsi_traffic::Person { pos: DVec3::new(23.0, 12.6, 0.0), vel: DVec2::ZERO, on_crossing: false, still: 0.0, waiting_for_bus: false, at_kerb: false, waits_at: None, id: Some(4), approach }];
        sim.tick(&net, DT, &mut cars);
        far = far.max(cars[0].pos.x + 2.0);
        t += DT;
    }
    assert!(far < 23.0 - 2.0, "front reached {far:.2}");
}

#[test]
fn p92_a_car_past_the_zebra_stop_brakes_for_a_person_who_steps_on_first() {
    // MSLink::getLeaderInfo's "breaking for approaching person": already past the zebra stop,
    // 2 m short of the cut at 2 m/s (1 s to it), the person steps on in 0.2 s: it stops 1 m
    // before the crossing (jmStoplineCrossingGap)
    let (far, stood) = past_an_approaching_person(21.0, 2.0, Some(0.2), 4.0);
    assert!(stood && far < 22.6, "front reached {far:.2}, stood {stood}");
}
