//! Round 10 area M (density semantics): the density floor of the access rule and the lane-end
//! rule at closed stop-line pieces.
//! `cargo test --release -p omsi-traffic --test round10m`

use glam::DVec3;
use omsi_sim::traffic::{density_is_open, AiState, Closed, LaneBuilder, LaneKey, LaneKind, Network, OPEN_DENSITY_FLOOR};
use omsi_traffic::{Simulation, Vehicle, VehicleClass};

const DT: f32 = 1.0 / 30.0;

fn road(from: (f64, f64), to: (f64, f64), id: i64, path: u16, offset: f32) -> omsi_sim::traffic::Lane {
    let mut l = LaneBuilder::polyline(vec![DVec3::new(from.0, from.1, 0.0), DVec3::new(to.0, to.1, 0.0)], LaneKind::Street, 3.5);
    l.source = 1;
    l.offset = offset;
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

/// The editor's density classes: 0 and 0.001 are closed, 0.1 and above are open.
#[test]
fn the_density_floor_separates_the_editors_classes() {
    for closed in [0.0, 0.001] {
        assert!(!density_is_open(closed), "{closed}");
    }
    for open in [0.1, 0.5, 1.0, 3.0] {
        assert!(density_is_open(open), "{open}");
    }
    assert!(0.001 < OPEN_DENSITY_FLOOR && OPEN_DENSITY_FLOOR < 0.1);
}

/// B9-1: BeijingCity's bus path: the normal cars' density 0 on the bus lane, 0.001 on its feeder,
/// the buses' 1; allcar 0.5. A car of group 0 is refused both, the buses' group is not.
#[test]
fn a_near_closure_is_closed_to_the_group_by_its_own_density() {
    let defaults = [1, 2, 1, 1, 1, 1, 1];
    let mut feeder = road((0.0, 0.0), (40.0, 0.0), 1, 0, 0.0);
    feeder.group_density = vec![(0, 0.001), (3, 0.5), (6, 1.0)];
    feeder.density = 1.0;
    assert_eq!(feeder.closed_to(0, Some((&defaults, 0)), true), Some(Closed::Density));
    assert_eq!(feeder.closed_to(0, Some((&defaults, 3)), true), None);
    assert_eq!(feeder.closed_to(0, Some((&defaults, 6)), true), None);
    // a vehicle of no group takes the first group's share, not the most any group has
    assert_eq!(feeder.closed_to(0, None, true), Some(Closed::Density));
    // a scheduled bus (a route) is not asked for density
    assert_eq!(feeder.closed_to(-1, Some((&defaults, 0)), false), None);
    // the path's rules come first
    feeder.no_cars = true;
    assert_eq!(feeder.closed_to(0, Some((&defaults, 3)), true), Some(Closed::Rules));
}

/// B9-1: the way on at a fork never takes a way at 0.001 for a group while an open way exists;
/// with none open it drives on (counted by `FORK_FALLBACKS`), as it does through a density 0 way.
#[test]
fn the_fork_never_routes_a_group_over_a_near_closure_when_a_way_is_open() {
    let defaults = std::sync::Arc::new(vec![1, 2, 1, 1]);
    let fork = |open_density: f32| {
        let mut lanes = vec![road((0.0, 0.0), (100.0, 0.0), 1, 0, 0.0), road((100.0, 0.0), (200.0, 10.0), 2, 0, 0.0), road((100.0, 0.0), (200.0, -10.0), 3, 0, 0.0)];
        lanes[1].group_density = vec![(0, 0.001), (3, 1.0)];
        lanes[2].group_density = vec![(0, open_density), (3, 1.0)];
        let mut net = Network { lanes, ..Default::default() };
        net.link(1.5);
        net
    };
    let ways = |net: &Network| -> Vec<Option<usize>> {
        (0..40u64)
            .map(|seed| {
                let mut st = AiState::new(0, 95.0, seed + 1);
                st.traffic_pool = Some((0, defaults.clone()));
                st.plan_next(net);
                st.planned_next
            })
            .collect()
    };
    assert!(ways(&fork(0.1)).iter().all(|w| *w == Some(2)), "{:?}", ways(&fork(0.1)));
    // none open: it drives on over one of them
    assert!(ways(&fork(0.001)).iter().all(|w| w.is_some()));
}

/// M9-1 (#31323): a 3 m stop-line piece (shorter than the 4 m body) of density 0 for the normal cars, the only way on from its
/// lane and leading on into open road, is no lane end: the car does not set out to move over (and
/// stand 16 m short of the piece where the lane beside is full), it drives on over the piece and the open lane after it. The other lane has its own
/// open piece.
#[test]
fn a_closed_stop_line_piece_that_leads_on_is_no_lane_end() {
    let mut net = Network::default();
    net.lanes.push(road((0.0, 0.0), (600.0, 0.0), 1, 0, 0.0));
    net.lanes.push(road((0.0, -3.5), (600.0, -3.5), 1, 1, 3.5));
    net.lanes.push(road((600.0, 0.0), (603.0, 0.0), 2, 0, 0.0));
    net.lanes.push(road((600.0, -3.5), (603.0, -3.5), 2, 1, 3.5));
    net.lanes.push(road((603.0, 0.0), (1500.0, 0.0), 3, 0, 0.0));
    net.lanes.push(road((603.0, -3.5), (1500.0, -3.5), 3, 1, 3.5));
    net.lanes[3].group_density = vec![(0, 0.0), (5, 1.0)];
    net.link(1.5);
    net.compute_conflicts();
    net.compute_reach();
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![car(1, 1, 400.0, 12.0)];
    let (mut stood, mut reached, mut moved_over) = (false, false, false);
    let mut t = 0.0;
    while t < 40.0 {
        sim.tick(&net, DT, &mut cars);
        sim.update_positions(&net, &mut cars);
        stood |= cars[0].state.speed < 0.3;
        reached |= cars[0].state.lane == 5;
        moved_over |= cars[0].state.change.is_some() || [0, 2, 4].contains(&cars[0].state.lane);
        t += DT;
    }
    assert!(!stood, "stood");
    assert!(!moved_over, "left its lane for the closed piece at its end");
    assert!(reached, "never drove over the piece: lane {} s {:.0}", cars[0].state.lane, cars[0].state.s);
}

/// M9-1: the same piece one body long (a 30 m closed run) is a closure: the car moves over before it,
/// as the J7-3 lane drop; and the group's longest body (`Network::group_body`) decides: a 12 m
/// run is a marker for a group with an 18 m bus, a closure for one whose longest body is 4 m.
#[test]
fn a_closed_run_as_long_as_a_body_is_a_closure_and_the_groups_longest_body_decides() {
    let build = |piece_len: f64| {
        let mut net = Network::default();
        net.lanes.push(road((0.0, 0.0), (600.0, 0.0), 1, 0, 0.0));
        net.lanes.push(road((600.0, 0.0), (600.0 + piece_len, 0.0), 2, 0, 0.0));
        net.lanes.push(road((600.0 + piece_len, 0.0), (1500.0, 0.0), 3, 0, 0.0));
        net.lanes[1].group_density = vec![(0, 0.0), (5, 1.0)];
        net.link(1.5);
        net
    };
    let pool = Some((0usize, std::sync::Arc::new(vec![1, 1, 1, 1, 1, 1])));
    let mut st = AiState::new(0, 100.0, 1);
    st.length = 4.0;
    st.traffic_pool = pool;
    let mut net = build(12.0);
    net.group_body = vec![18.0; 6];
    assert!(net.closed_marker(1, &st), "12 m < an 18 m bus: a marker");
    net.group_body = vec![4.0; 6];
    assert!(!net.closed_marker(1, &st), "12 m >= 4 m: a closure");
    let net = build(3.0);
    assert!(net.closed_marker(1, &st), "3 m < 4 m: a marker");
}
