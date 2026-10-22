//! Round 7 area J7 (the junction core): the lane-end change with cooperation (J7-3).
//! `cargo test --release -p omsi-traffic --test round7j -- --nocapture`

use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network};
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

/// Two lanes of one road (0 left at y = 0, 1 right at y = -3.5) to x = 600, each running on
/// into its own lane of the next spline (2 left, 3 right) to x = 1500; `close` closes lanes of
/// that list to cars (`no_cars`).
fn lane_drop(close: &[usize]) -> Network {
    let mut net = Network::default();
    net.lanes.push(road((0.0, 0.0), (600.0, 0.0), 1, 0, 0.0));
    net.lanes.push(road((0.0, -3.5), (600.0, -3.5), 1, 1, 3.5));
    net.lanes.push(road((600.0, 0.0), (1500.0, 0.0), 2, 0, 0.0));
    net.lanes.push(road((600.0, -3.5), (1500.0, -3.5), 2, 1, 3.5));
    for &c in close {
        net.lanes[c].no_cars = true;
    }
    net.link(1.5);
    net.compute_conflicts();
    net.compute_reach();
    net
}

/// Run `secs`; per frame `each`. The least separation of any two boxes.
fn run(net: &Network, sim: &mut Simulation, cars: &mut Vec<Vehicle>, secs: f32, mut each: impl FnMut(&Simulation, &[Vehicle])) -> f32 {
    let mut least = f32::INFINITY;
    let mut t = 0.0;
    while t < secs {
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
        each(sim, cars);
        t += DT;
    }
    least
}

/// [J7-3] SUMO LC2013's strategic change: the right lane runs into a lane closed to cars; the car
/// on it moves over into the open lane before the end and never takes the closed one, without
/// standing.
#[test]
fn j73_a_car_leaves_a_lane_that_runs_into_a_closed_one_before_its_end() {
    let net = lane_drop(&[3]);
    assert_eq!(net.lanes[1].left, Some(0), "neighbours");
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![car(1, 1, 100.0, 12.0)];
    let (mut on_closed, mut stood, mut moved_at) = (false, false, None);
    run(&net, &mut sim, &mut cars, 60.0, |_, cars| {
        let c = &cars[0];
        on_closed |= c.state.lane == 3;
        stood |= c.state.speed < 0.3;
        if moved_at.is_none() && (c.state.lane == 0 || c.state.change.is_some()) {
            moved_at = Some(c.pos.x);
        }
    });
    assert!(!on_closed, "drove into the closed lane");
    assert!(!stood, "stood");
    let x = moved_at.expect("never moved over");
    assert!(x < 600.0 && x > 300.0, "moved over at x = {x:.0} (the strategic distance is ~130 m at 12 m/s)");
}

/// [J7-3, fix pass 1] SUMO LC2013's cooperation (`informFollower`): a car in the open lane just
/// behind the changer is asked to open the gap the changer's acceptance needs, brakes no harder
/// than comfortably, and the changer moves over before the end; nobody touches and nobody is
/// left standing on the open lane. (Pass 1: the test never asserted the change, and both cars
/// stood.)
#[test]
fn j73_the_follower_in_the_target_lane_makes_room() {
    let net = lane_drop(&[3]);
    let mut sim = Simulation::new(&net, vec![]);
    // the follower 3 m behind the changer's rear in the open lane, at the same speed (first in
    // the order: the changer's request, made as it acts, is there to see after the frame)
    let mut cars = vec![car(2, 0, 393.0, 12.0), car(1, 1, 400.0, 12.0)];
    cars[0].state.change_cooldown = 100.0;
    let (mut asked, mut on_closed, mut hardest, mut changed) = (false, false, 0.0f32, false);
    let least = run(&net, &mut sim, &mut cars, 40.0, |_, cars| {
        if let Some(f) = cars.iter().find(|c| c.id == 2) {
            asked |= f.help_for.is_some_and(|h| h.0 == 1);
            hardest = hardest.max(-f.state.acc);
        }
        on_closed |= cars.iter().any(|c| c.id == 1 && c.state.lane == 3);
        changed |= cars.iter().any(|c| c.id == 1 && (c.state.lane == 0 || c.state.lane == 2));
    });
    assert!(asked, "the follower was never asked to make room");
    assert!(changed, "the changer never moved over: {:?}", cars.iter().map(|c| (c.id, c.state.lane, c.state.s, c.state.speed)).collect::<Vec<_>>());
    assert!(!on_closed, "drove into the closed lane");
    assert!(hardest <= 2.0 * 1.2 + 0.3, "the follower braked {hardest:.1} m/s²");
    assert!(least > 0.3, "touched: {least:.2} m");
    assert!(cars.iter().all(|c| c.state.speed > 1.0), "left standing: {:?}", cars.iter().map(|c| (c.id, c.state.lane, c.state.speed)).collect::<Vec<_>>());
}

/// [J7-3, fix pass 1-2] The safety review's chain, 36 runs: one or two followers at 0.6-1 s
/// behind each other in the open lane, level with the changer or behind it, at 8-12 m/s: in every
/// run the changer moves over before the end, nobody touches, nobody is left standing.
#[test]
fn j73_a_chain_of_followers_lets_the_changer_in() {
    let net = lane_drop(&[3]);
    let mut bad = Vec::new();
    for n in [1usize, 2] {
        for headway in [0.6f32, 1.0] {
            for off in [-4.0f32, -1.0, 2.0] {
                for v in [8.0f32, 10.0, 12.0] {
                    let mut sim = Simulation::new(&net, vec![]);
                    let mut cars = Vec::new();
                    for k in 0..n {
                        let mut f = car(10 + k as u64, 0, 400.0 + off - k as f32 * (4.0 + headway * v + 2.0), v);
                        f.state.change_cooldown = 100.0;
                        cars.push(f);
                    }
                    cars.push(car(1, 1, 400.0, v));
                    let (mut changed, mut on_closed) = (false, false);
                    let least = run(&net, &mut sim, &mut cars, 60.0, |_, cars| {
                        changed |= cars.iter().any(|c| c.id == 1 && (c.state.lane == 0 || c.state.lane == 2));
                        on_closed |= cars.iter().any(|c| c.id == 1 && c.state.lane == 3);
                    });
                    let standing = cars.iter().any(|c| c.state.speed < 1.0);
                    if !changed || on_closed || least < 0.3 || standing {
                        bad.push(format!("n {n} headway {headway} off {off} v {v}: changed {changed} closed {on_closed} least {least:.2} end {:?}", cars.iter().map(|c| (c.id, c.state.lane, c.state.s as i32, c.state.speed)).collect::<Vec<_>>()));
                    }
                }
            }
        }
    }
    assert!(bad.is_empty(), "{} of 36 runs: {:#?}", bad.len(), bad);
}

/// [J7-3, fix pass 3] The reviewer's chain (review r7j): the changer 100 m before its lane runs
/// into a closed one, three followers in the open lane, the first beside it (-6..5 m), each
/// `head` s behind the one ahead. When the first passes, the second may be too close to open
/// the acceptance gap at its comfortable brake (standing 1.56 m behind the changer's rear
/// against 1.82 m at rest): it is then not asked to brake and the changer goes in behind it
/// (SUMO's `neededGap`/`decelGap` test before `helpDecel`). Over 36 runs nobody is left standing
/// at 30 s and nobody touches; the changer never takes the closed lane.
#[test]
fn j73_a_follower_asked_to_brake_can_still_open_the_gap() {
    let net = lane_drop(&[3]);
    let mut bad = Vec::new();
    for k in 0..12 {
        let off = -6.0 + k as f32;
        for &(v, head) in &[(12.0f32, 1.0f32), (14.0, 0.8), (8.0, 0.6)] {
            let mut sim = Simulation::new(&net, vec![]);
            let gap = v * head + 4.0;
            let mut cars = vec![car(2, 0, 500.0 + off, v), car(3, 0, 500.0 + off - gap, v), car(4, 0, 500.0 + off - 2.0 * gap, v), car(1, 1, 500.0, v)];
            for c in cars.iter_mut() {
                c.state.max_speed_kmh = 60.0;
                c.state.change_cooldown = if c.id == 1 { 0.0 } else { 100.0 };
            }
            let mut on_closed = false;
            let least = run(&net, &mut sim, &mut cars, 30.0, |_, cars| {
                on_closed |= cars.iter().any(|c| c.id == 1 && c.state.lane == 3);
            });
            let standing: Vec<u64> = cars.iter().filter(|c| c.state.speed < 0.3).map(|c| c.id).collect();
            if !standing.is_empty() || on_closed || least < 0.3 {
                bad.push(format!("off {off} v {v} head {head}: standing {standing:?} closed {on_closed} least {least:.2}"));
            }
        }
    }
    assert!(bad.is_empty(), "{} of 36 runs: {:#?}", bad.len(), bad);
}

/// [J7-3] viz5 case04: a lane closed to the car (a bus lane) is never the target, whatever its
/// own lane runs into; with no open lane beside it the car drives on as before (no stop).
#[test]
fn j73_a_closed_lane_beside_is_never_the_target() {
    let net = lane_drop(&[0, 2, 3]);
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![car(1, 1, 300.0, 12.0)];
    let (mut on_bus_lane, mut stood) = (false, false);
    run(&net, &mut sim, &mut cars, 40.0, |_, cars| {
        if let Some(c) = cars.first() {
            on_bus_lane |= c.state.lane == 0 || c.state.lane == 2 || c.state.change.is_some();
            stood |= c.state.speed < 0.3;
        }
    });
    assert!(!on_bus_lane, "changed onto the closed lane");
    assert!(!stood, "stood although no open lane was there");
}

/// [J7-3, fix pass 2] A lane that ends past a junction object whose two pieces run side by side,
/// a third way of the object crossing them: object pieces are no neighbours to the network
/// (`Network::link` pairs the paths of one spline only), so the in-object branch of
/// `may_change_for_lane_end` is not reached on OMSI data; the car moves over on the spline
/// approach before the object, as LC2013 does on the edge before a junction, and never takes the
/// closed lane. (The safety review's probe gave the pieces no neighbours: so does every map.)
#[test]
fn j73_a_lane_ending_past_an_object_is_left_on_the_approach() {
    let piece = |pts: &[(f64, f64)], path: u16, offset: f32| {
        let mut l = LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), LaneKind::Street, 3.5);
        l.source = 2;
        l.offset = offset;
        l.priority = 128.0;
        l.key = Some(LaneKey { tile: (0, 0), id: 9, path });
        l
    };
    let mut net = Network::default();
    net.lanes.push(road((-400.0, 0.0), (0.0, 0.0), 1, 0, 0.0)); // 0 approach left
    net.lanes.push(road((-400.0, -3.5), (0.0, -3.5), 1, 1, 3.5)); // 1 approach right
    net.lanes.push(piece(&[(0.0, 0.0), (150.0, 0.0)], 0, 0.0)); // 2 piece left (open on)
    net.lanes.push(piece(&[(0.0, -3.5), (150.0, -3.5)], 1, 3.5)); // 3 piece right (into a closed lane)
    net.lanes.push(road((150.0, 0.0), (900.0, 0.0), 2, 0, 0.0)); // 4
    let mut closed = road((150.0, -3.5), (900.0, -3.5), 2, 1, 3.5); // 5
    closed.no_cars = true;
    net.lanes.push(closed);
    net.lanes.push(piece(&[(120.0, -40.0), (120.0, 40.0)], 2, 0.0)); // 6 the crossing way
    net.link(1.5);
    net.compute_conflicts();
    net.compute_reach();
    assert_eq!((net.lanes[3].left, net.lanes[2].right), (None, None), "object pieces are no neighbours");
    assert_eq!(net.lanes[1].left, Some(0), "the approach lanes are");
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![car(1, 1, 380.0, 10.0)];
    let (mut changed, mut on_closed) = (false, false);
    let mut fr = 0;
    let least = run(&net, &mut sim, &mut cars, 40.0, |sim, cars| {
        fr += 1;
        if std::env::var_os("PROBE").is_some() && fr % 15 == 0 && fr < 300 {
            let p = omsi_traffic::perception::perceive(&net, sim, cars, 0);
            eprintln!("L{} s{:.1} v{:.1} why {:?} end {:?} left {:?} way {:?}", cars[0].state.lane, cars[0].state.s, cars[0].state.speed, cars[0].last.why, p.lane_end, p.left.map(|l| (l.lane, l.allowed, l.ahead, l.behind)), p.way);
        }
        if let Some(c) = cars.first() {
            changed |= c.state.lane == 0 || c.state.lane == 2 || c.state.lane == 4;
            on_closed |= c.state.lane == 5;
        }
    });
    assert!(changed && !on_closed, "changed {changed}, on the closed lane {on_closed}: {:?}", cars.iter().map(|c| (c.state.lane, c.state.s)).collect::<Vec<_>>());
    assert!(least > 0.3 || cars.len() < 2);
}

