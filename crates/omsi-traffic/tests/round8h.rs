//! Round 8 area H8 (junction and compile hazards).
//! `cargo test --release -p omsi-traffic --test round8h -- --nocapture`

use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network};
use omsi_traffic::{Simulation, Vehicle, VehicleClass};

const DT: f32 = 1.0 / 30.0;

fn road(from: (f64, f64), to: (f64, f64), id: i64, path: u16) -> omsi_sim::traffic::Lane {
    let mut l = LaneBuilder::polyline(vec![DVec3::new(from.0, from.1, 0.0), DVec3::new(to.0, to.1, 0.0)], LaneKind::Street, 3.5);
    l.source = 1;
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

/// One straight lane along +y (heading 0) from y = 0 to y = 600.
fn one_lane() -> Network {
    let mut net = Network::default();
    net.lanes.push(road((0.0, 0.0), (0.0, 600.0), 1, 0));
    net.link(1.5);
    net.compute_conflicts();
    net.compute_reach();
    net
}

/// A parked car leaving its space as the app puts it there (`Traffic::pull_out_parked`): on its
/// lane at `s`, `lat` m to the right, its way ramped back to the axis over 8-16 m, held for its
/// `pull_out` seconds, indicating.
fn pulled_out(id: u64, s: f32, lat: f32) -> Vehicle {
    let mut c = car(id, 0, s, 0.0);
    c.state.lateral = lat;
    c.state.lateral_target = 0.0;
    c.state.lateral_ramp = (lat, 0.0, c.state.odometer, (lat * 6.0).clamp(8.0, 16.0));
    c.pull_out = 2.0;
    c.state.blinker = 1;
    c
}

/// Run `secs`; before each frame `before(t, sim)` (the people), after it `each(t, sim, cars)`.
/// The least separation of any two boxes.
fn run(net: &Network, sim: &mut Simulation, cars: &mut Vec<Vehicle>, secs: f32, mut before: impl FnMut(f32, &mut Simulation), mut each: impl FnMut(f32, &Simulation, &[Vehicle])) -> f32 {
    let mut least = f32::INFINITY;
    let mut t = 0.0;
    sim.update_positions(net, cars);
    while t < secs {
        before(t, sim);
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
        each(t, sim, cars);
        t += DT;
    }
    least
}

fn nobody(_: f32, _: &mut Simulation) {}

/// [H8-1] SUMO's parked vehicle re-enters by insertion, which checks the follower's secure gap
/// on the lane it enters: a car leaving its space with a car coming up its lane at 12 m/s
/// 40 m behind (inside the gap a change needs from it) stays parked until that car has gone
/// by, then leaves; nobody brakes hard for it and nobody touches.
#[test]
fn h8_1_a_pull_out_waits_for_the_follower_in_its_lane() {
    let net = one_lane();
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![pulled_out(1, 100.0, 3.0), car(2, 0, 60.0, 12.0)];
    let (mut moved_before_passed, mut moved, mut hardest) = (false, false, 0.0f32);
    let least = run(&net, &mut sim, &mut cars, 20.0, nobody, |_, _, cars| {
        let (Some(p), Some(f)) = (cars.iter().find(|c| c.id == 1), cars.iter().find(|c| c.id == 2)) else { return };
        let moving = p.state.speed > 0.3;
        moved |= moving;
        moved_before_passed |= moving && f.state.s < p.state.s;
        hardest = hardest.max(-f.state.acc);
    });
    assert!(!moved_before_passed, "pulled out in front of the car coming up its lane");
    assert!(moved, "never left its space");
    assert!(hardest < 1.0, "the follower braked {hardest:.1} m/s² for it");
    assert!(least > 0.0, "touched: {least:.2}");
}

/// [H8-1] The counter-case: the follower far back (200 m) leaves the gap: the car leaves its
/// space when its `pull_out` has run, as before.
#[test]
fn h8_1_a_pull_out_with_room_behind_leaves_on_time() {
    let net = one_lane();
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![pulled_out(1, 300.0, 3.0), car(2, 0, 100.0, 12.0)];
    let mut left_at = None;
    run(&net, &mut sim, &mut cars, 8.0, nobody, |t, _, cars| {
        if left_at.is_none() && cars.iter().any(|c| c.id == 1 && c.state.speed > 0.3) {
            left_at = Some(t);
        }
    });
    let t = left_at.expect("never left its space");
    assert!(t < 3.0, "left at {t:.1} s, its pull_out is 2 s");
}

/// A lane of crossing object 9 (source 2).
fn piece(pts: &[(f64, f64)], path: u16, kind: LaneKind, priority: f32) -> omsi_sim::traffic::Lane {
    let mut l = LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), kind, 3.5);
    l.source = 2;
    l.key = Some(LaneKey { tile: (0, 0), id: 9, path });
    l.priority = priority;
    l
}

/// [H8-4] Zebra-then-crossing inside a long object: an eastbound car already inside object 9
/// (its piece x -15..60, so its line is behind it) is refused for a stream on the northbound
/// priority piece at x = 25; the object's zebra cuts its way at x = 20, so the stretch it waits
/// for begins right after the zebra. J7-4's waiting positions (`legal_before` over
/// `wait_bands`, the walk bands among them) pull the refused stop back before the zebra: it
/// comes to rest refused (`Why::Junction`) with its front at most `ZEBRA_STOP` short of the cut,
/// and its body is never over the zebra band (cut - ZEBRA_STOP .. cut + ZEBRA_PAST) while the
/// stream passes.
#[test]
fn h8_4_a_refused_car_inside_a_long_object_stops_before_the_zebra_short_of_the_crossing() {
    use omsi_traffic::plan::{ZEBRA_PAST, ZEBRA_STOP};
    let mut net = Network::default();
    net.lanes.push(road((-300.0, 0.0), (-15.0, 0.0), 1, 0));
    net.lanes.push(piece(&[(-15.0, 0.0), (60.0, 0.0)], 0, LaneKind::Street, 0.0));
    net.lanes.push(road((60.0, 0.0), (400.0, 0.0), 2, 0));
    net.lanes.push(piece(&[(25.0, -40.0), (25.0, 40.0)], 1, LaneKind::Street, 10.0));
    net.lanes.push(piece(&[(20.0, -8.0), (20.0, 8.0)], 2, LaneKind::Sidewalk, 0.0));
    net.link(1.5);
    net.build_grid();
    net.compute_conflicts();
    net.compute_reach();
    assert!(net.walks[1].iter().any(|&(w, _, _)| w == 4), "the zebra cuts the piece: {:?}", net.walks[1]);
    assert!(net.crossings[1].iter().any(|c| c.other == 3 && !c.merge), "the pieces cross: {:?}", net.crossings[1]);
    let mut sim = Simulation::new(&net, vec![]);
    // (a stream of nine on the priority piece, 14 m apart at 9 m/s; the car rolls in at 7 m/s
    // 5 m inside the object)
    let mut cars = vec![car(1, 1, 7.0, 7.0)];
    cars.extend((0..9u64).map(|k| car(10 + k, 3, 28.0 - 14.0 * k as f32, 9.0)));
    for c in cars.iter_mut().skip(1) {
        c.state.prev_lane = None;
    }
    let cut = 20.0 + 15.0; // along the piece
    let (mut over_zebra, mut rest) = (false, None);
    run(&net, &mut sim, &mut cars, 15.0, nobody, |_, _, cars| {
        let me = cars.iter().find(|c| c.id == 1).unwrap();
        if std::env::var_os("H8_DEBUG").is_some() {
            eprintln!("lane {} s {:.1} v {:.2} why {} stop {:?} wf {:?} | {}", me.state.lane, me.state.s, me.state.speed, me.last.why.name(), me.last.stop, me.last.wait_for, cars.iter().skip(1).map(|c| format!("{}:{}/{:.0}/{:.1}", c.id, c.state.lane, c.state.s, c.state.speed)).collect::<Vec<_>>().join(" "));
        }
        let (front, rear) = (me.state.s + me.state.front, me.state.s - me.state.rear);
        if me.state.lane == 1 && front > cut - ZEBRA_STOP && rear < cut + ZEBRA_PAST {
            over_zebra = true;
        }
        if me.state.speed < 0.05 && rest.is_none() {
            rest = Some((me.state.lane, front, me.last.why.name()));
        }
    });
    let (lane, front, why) = rest.expect("never came to rest");
    eprintln!("rests on {lane} with its front {:.2} m short of the cut, why {why}", cut - front);
    assert_eq!(lane, 1, "rest on the piece");
    assert!(!over_zebra, "its body was over the zebra band");
    assert_eq!(why, "yield", "refused at the junction (Why::Junction)");
    assert!(front <= cut - ZEBRA_STOP + 0.05 && front > cut - ZEBRA_STOP - 3.0, "rests with its front {:.1} m short of the cut ({why})", cut - front);
}

