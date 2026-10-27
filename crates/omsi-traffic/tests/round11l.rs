//! Lane changes of the drivers' own: the gap checked again while the change signals (SUMO's
//! `checkChange` every step), the follower looked up back over the lanes leading in
//! (`getFollowersOnConsecutive`).
use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network};
use omsi_traffic::{Simulation, Vehicle, VehicleClass};

const DT: f32 = 1.0 / 30.0;

fn road(pts: &[(f64, f64)], id: i64, path: u16, offset: f32) -> omsi_sim::traffic::Lane {
    let mut l = LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), LaneKind::Street, 3.0);
    l.source = 1;
    l.key = Some(LaneKey { tile: (0, 0), id, path });
    l.offset = offset;
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

/// The lane changed from, 0 (x 304..600 at y -3.5, fed by lane 1 from x 0), beside lane 2 on
/// its left (x 304..600 at y 0, one spline with lane 0). Lane 2 is fed by a 4 m piece, lane 3
/// (x 300..304), which is fed by lane 4 (x 0..300): a follower on lane 4 is two lanes back of
/// the target.
fn pieces() -> Network {
    let mut net = Network::default();
    net.lanes.push(road(&[(304.0, -3.5), (600.0, -3.5)], 1, 1, 3.5));
    net.lanes.push(road(&[(0.0, -3.5), (304.0, -3.5)], 2, 0, 0.0));
    net.lanes.push(road(&[(304.0, 0.0), (600.0, 0.0)], 1, 0, 0.0));
    net.lanes.push(road(&[(300.0, 0.0), (304.0, 0.0)], 3, 0, 0.0));
    net.lanes.push(road(&[(0.0, 0.0), (300.0, 0.0)], 4, 0, 0.0));
    net.link(1.5);
    net.compute_conflicts();
    net
}

/// Runs the changer (index 0) with its change onto lane 2 signalled at t 0 and the follower
/// (index 1) on lane 4 at `follower_s`, 13 m/s; returns (the change dropped before the car moved
/// over - when, s - the change carried out onto lane 2).
fn signalled_change(follower_s: f32) -> (Option<f32>, bool) {
    let net = pieces();
    assert_eq!(net.lanes[0].left, Some(2), "lane 2 lies left of lane 0");
    assert!(net.prev[2].contains(&3) && net.prev[3].contains(&4), "lane 4 feeds lane 3 feeds lane 2");
    let mut sim = Simulation::new(&net, vec![]);
    let mut me = car(1, 0, 6.0, 4.0);
    me.state.start_change(&net, 2, 1);
    let mut f = car(2, 4, follower_s, 13.0);
    f.state.change_cooldown = 1000.0;
    let mut cars = vec![me, f];
    let (mut dropped, mut done) = (None, false);
    let mut t = 0.0f32;
    while t < 5.0 {
        sim.tick(&net, DT, &mut cars);
        let st = &cars[0].state;
        if st.change.is_none() && st.lane == 0 && !done {
            dropped = Some(t);
            break;
        }
        if st.lane == 2 {
            done = true;
        }
        t += DT;
    }
    (dropped, done)
}

/// A follower two lanes back of the target (lane 4, beyond the 4 m piece), 15 m behind the
/// changer's rear and closing at 13 m/s, needs far more than that gap (its standing gap, the
/// changer's time gap at its speed, the braking to the changer's speed): the signalled change is
/// dropped at once, while the follower is still on lane 4, before the car moves over. With the
/// follower 250 m back the same change goes.
#[test]
fn follower_two_lanes_back_aborts_a_signalled_change() {
    // (the changer's rear at x 308; the follower's front at x 293, on lane 4 for 0.5 s more)
    let (dropped, done) = signalled_change(291.0);
    assert!(dropped.is_some_and(|t| t < 0.2) && !done, "the change into the closing follower's way is dropped at once (dropped at {dropped:?}, done {done})");
    let (dropped, done) = signalled_change(50.0);
    assert!(dropped.is_none() && done, "with the follower far back the change is carried out (dropped at {dropped:?}, done {done})");
}
