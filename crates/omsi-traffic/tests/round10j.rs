//! Headless checks for round 10 area J (foe eligibility): the crate predicates other areas read.

use glam::DVec3;
use omsi_sim::traffic::{LaneBuilder, LaneKey, LaneKind, Network};
use omsi_traffic::reservation::{Claim, Grant};
use omsi_traffic::{Impatience, Simulation};

fn piece(pts: &[(f64, f64)], path: u16, kind: LaneKind) -> omsi_sim::traffic::Lane {
    let mut l = LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), kind, 3.5);
    l.source = 2;
    l.key = Some(LaneKey { tile: (0, 0), id: 9, path });
    l.priority = 128.0;
    l
}

/// A crossing object: eastbound piece 0 (x -15..45), northbound piece 1 (x = 0), and a zebra
/// (footpath 2) across piece 0 at x = 23.
fn cross_with_a_zebra() -> Network {
    let mut net = Network::default();
    net.lanes.push(piece(&[(-15.0, 0.0), (45.0, 0.0)], 0, LaneKind::Street));
    net.lanes.push(piece(&[(0.0, -15.0), (0.0, 15.0)], 1, LaneKind::Street));
    net.lanes.push(piece(&[(23.0, -8.0), (23.0, 8.0)], 2, LaneKind::Sidewalk));
    net.link(1.5);
    net.build_grid();
    net.compute_conflicts();
    net
}

fn claim_on(sim: &Simulation, lane: usize, s: f32, in_box_at_turn: bool) -> Claim {
    let (j, m) = sim.junctions.of_lane[&lane];
    Claim { chain: vec![(m, 0.0)], vehicle: 0, junction: j, movement: m, s, speed: 0.0, accel: 1.8, v_max: 13.0, front: 2.0, rear: 2.0, decel: 2.0, waiting: 0.0, lead_gap: None, zip_behind: None, body_block: None, next_wait: f32::NEG_INFINITY, tau: 1.3, min_gap: 2.0, committed: false, exit_room: true, impatience: Impatience::default(), absolute: false, held: false, light_red: false, until_go: None, red_at: f32::INFINITY, lead: None, was_standing: true, blocked: false, stuck: false, clearing: false, in_box_at_turn, lanes: Vec::new(), light: Some((0, 0)), id: 1, yields_now: None, clear_ahead: f32::INFINITY, brake: 5.6 }
}

/// [round 10 D7] `Simulation::clearing_across`: a vehicle in the box at the end of its green whose
/// way still crosses the zebra ahead of its rear holds the zebra's walkers; once its rear is past
/// the zebra, or when it is not in the box at the end of its green, it does not.
#[test]
fn d7_walkers_see_a_vehicle_clearing_the_box_across_their_crossing() {
    let net = cross_with_a_zebra();
    let mut sim = Simulation::new(&net, vec![]);
    assert!(sim.junctions.of_lane.contains_key(&0), "piece 0 is a junction piece");
    let zebra_at = net.walks[0].iter().find(|w| w.0 == 2).expect("the zebra cuts piece 0").1;
    for (s, in_box, expect) in [(5.0f32, true, true), (zebra_at + 1.0, true, true), (zebra_at + 2.5, true, false), (5.0, false, false)] {
        sim.claims = vec![(claim_on(&sim, 0, s, in_box), Grant::go(0))];
        assert_eq!(sim.clearing_across(&net, 2), expect, "s {s} in the box at the turn {in_box}");
    }
}
