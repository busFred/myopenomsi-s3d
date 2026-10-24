//! Round 10 area P: the crate's side of the people rules.

use glam::{DVec2, DVec3};
use omsi_sim::traffic::{LaneBuilder, LaneKind, Network};
use omsi_traffic::{Claim, Grant, Impatience, Junction, Junctions, Movement, Simulation};

/// A junction of one movement (lane 0, along x from 0 to 20 m, 3 m wide) whose crossing cuts
/// it 12 m in, and a claim on it with its origin `s` along the movement, front/rear 2 m.
fn sim_with(s: f32, speed: f32) -> (Network, Simulation) {
    let mut net = Network::default();
    net.lanes.push(LaneBuilder::polyline(vec![DVec3::ZERO, DVec3::new(20.0, 0.0, 0.0)], LaneKind::Street, 3.0));
    let mut sim = Simulation::new(&net, Vec::new());
    let movement = Movement {
        lane: 0,
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
    js.nodes.push(Junction { key: ((0, 0), 1), movements: vec![movement], crossings: Vec::new() });
    js.of_lane.insert(0, (0, 0));
    sim.junctions = js;
    let claim = Claim {
        yields_now: None,
        clear_ahead: f32::INFINITY,
        brake: 5.6,
        vehicle: 0,
        junction: 0,
        movement: 0,
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
        chain: vec![(0, 0.0)],
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
}
