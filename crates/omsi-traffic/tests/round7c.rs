//! Round 7 cleanup, crate side: the give-up clock (B1) and the timetable/random bus split (A4).

use omsi_sim::traffic::AiState;
use omsi_traffic::{plan::Why, Vehicle, VehicleClass};

const DT: f32 = 0.1;

fn standing(class: VehicleClass) -> Vehicle {
    let mut v = Vehicle::new(1, class, AiState::new(0, 10.0, 1), 1.0, 7919);
    v.state.speed = 0.0;
    v
}

/// Run the clock for `secs` with the decision `why`; whether the vehicle gave up.
fn stand(v: &mut Vehicle, why: Why, secs: f32) -> bool {
    v.last.why = why;
    for _ in 0..(secs / DT).round() as usize {
        v.give_up_clock(DT);
    }
    v.gone
}

/// [round 7 B1] A random car gives up only after standing held by nothing for
/// `GIVE_UP_UNSEEN` at a stretch; any hold restarts the clock; moving takes it back.
#[test]
fn a_random_car_gives_up_only_held_by_nothing_for_minutes_and_comes_back_when_it_moves() {
    let give_up = omsi_traffic::sim::GIVE_UP_UNSEEN;
    assert!(give_up >= 300.0);
    // at a red light, in a queue, held by a junction or a person: never, however long
    let mut v = standing(VehicleClass::Car);
    for why in [Why::Light, Why::Lead, Why::Junction, Why::ExitFull, Why::People, Why::Merge] {
        assert!(!stand(&mut v, why, give_up + 60.0), "gave up held by {why:?}");
    }
    // held by nothing: not after a minute, after the give-up time
    assert!(!stand(&mut v, Why::None, 60.0));
    // (a hold in between restarts the clock)
    assert!(!stand(&mut v, Why::Light, 1.0));
    assert!(!stand(&mut v, Why::None, give_up - 1.0), "the clock did not restart on a hold");
    assert!(stand(&mut v, Why::None, 2.0), "held by nothing for the give-up time");
    // a creep is no progress; moving off takes the request back
    v.state.speed = 0.05;
    assert!(stand(&mut v, Why::None, 1.0));
    v.state.speed = 0.5;
    assert!(!stand(&mut v, Why::None, DT), "still given up while it drives on");
    assert_eq!(v.held_by_nothing, 0.0);
}

/// [round 7 A4] A random bus gives up like any random car; a timetable bus never does (its
/// `gone` is the app's: the crate neither sets nor clears it).
#[test]
fn a_random_bus_gives_up_a_timetable_bus_never() {
    let give_up = omsi_traffic::sim::GIVE_UP_UNSEEN;
    let mut bus = standing(VehicleClass::Bus);
    assert!(bus.heavy() && !bus.serves_stops());
    assert!(stand(&mut bus, Why::None, give_up + 1.0));
    let mut tt = standing(VehicleClass::ScheduledBus);
    assert!(tt.heavy() && tt.serves_stops());
    assert!(!stand(&mut tt, Why::None, give_up + 1.0));
    // (the app's own `gone` on a timetable bus, its trip over: kept while it drives)
    tt.gone = true;
    tt.state.speed = 5.0;
    assert!(stand(&mut tt, Why::None, 1.0));
}

/// [round 7 A1] The way on at a fork follows the access rule (`open_to`), in tiers: an open
/// lane leading on whenever there is one; with none open, one closed only by density before
/// one closed by the rules; a `no_cars` lane only when nothing else goes on (a car already
/// there drives through, it is never taken off moving).
#[test]
fn the_way_on_at_a_fork_takes_open_lanes_then_density_closed_never_no_cars() {
    use glam::DVec3;
    use omsi_sim::traffic::{LaneBuilder, LaneKind, Network};
    let lane = |a: (f64, f64), b: (f64, f64)| LaneBuilder::polyline(vec![DVec3::new(a.0, a.1, 0.0), DVec3::new(b.0, b.1, 0.0)], LaneKind::Street, 3.5);
    // lane 0 into a fork of three: 1 (the case's "open"), 2 (density 0), 3 (no_cars)
    let fork = |open: bool| {
        let mut lanes = vec![lane((0.0, 0.0), (100.0, 0.0)), lane((100.0, 0.0), (200.0, 10.0)), lane((100.0, 0.0), (200.0, 0.0)), lane((100.0, 0.0), (200.0, -10.0))];
        if !open {
            lanes[1].no_cars = true;
        }
        lanes[2].density = 0.0;
        lanes[3].no_cars = true;
        let mut net = Network { lanes, ..Default::default() };
        net.link(1.5);
        net
    };
    let ways = |net: &Network| -> Vec<Option<usize>> {
        (0..40u64)
            .map(|seed| {
                let mut st = AiState::new(0, 95.0, seed + 1);
                st.plan_next(net);
                st.planned_next
            })
            .collect()
    };
    let net = fork(true);
    assert_eq!(net.lanes[0].next.len(), 3);
    assert!(ways(&net).iter().all(|w| *w == Some(1)), "only the open way: {:?}", ways(&net));
    // none open: the density-closed lane, never the no_cars one
    let net = fork(false);
    assert!(ways(&net).iter().all(|w| *w == Some(2)), "the density-closed way: {:?}", ways(&net));
    // only no_cars lanes: through one of them, as before round 7
    let mut net = fork(false);
    net.lanes[2].no_cars = true;
    assert!(ways(&net).iter().all(|w| w.is_some()), "{:?}", ways(&net));
}

/// [round 7 A1] SUMO's bestLanes, one step: an open lane whose only way on is closed to the
/// car is not taken while another open lane leads on.
#[test]
fn an_open_lane_leading_only_into_a_closed_one_is_not_taken() {
    use glam::DVec3;
    use omsi_sim::traffic::{LaneBuilder, LaneKind, Network};
    let lane = |a: (f64, f64), b: (f64, f64)| LaneBuilder::polyline(vec![DVec3::new(a.0, a.1, 0.0), DVec3::new(b.0, b.1, 0.0)], LaneKind::Street, 3.5);
    // 0 forks into 1 (on into 3, no_cars) and 2 (on into 4, open)
    let mut lanes = vec![lane((0.0, 0.0), (100.0, 0.0)), lane((100.0, 0.0), (130.0, 10.0)), lane((100.0, 0.0), (130.0, -10.0)), lane((130.0, 10.0), (300.0, 10.0)), lane((130.0, -10.0), (300.0, -10.0))];
    lanes[3].no_cars = true;
    let mut net = Network { lanes, ..Default::default() };
    net.link(1.5);
    for seed in 0..40u64 {
        let mut st = AiState::new(0, 95.0, seed + 1);
        st.plan_next(&net);
        assert_eq!(st.planned_next, Some(2), "seed {seed}");
    }
}

/// [round 7 fix pass 1] SUMO's yellow rule: at amber onset a car that cannot stop
/// comfortably is committed and goes, on red too while it rolls; one that can stops; a
/// committed car standing at its red waits; on red without a decision only a car that cannot
/// stop at all goes.
#[test]
fn the_decision_at_amber_onset_is_kept_and_a_standing_car_waits() {
    use omsi_sim::traffic::Aspect;
    use omsi_traffic::{plan::light_call, Perception, SignalView};
    let p = Perception::default();
    let mut v = standing(VehicleClass::Car);
    v.state.front = 2.0;
    v.state.speed = 12.0;
    let sig = |v: &Vehicle, gap: f32, aspect| SignalView { dist: v.state.front + gap, ctl: 0, light: 0, aspect, until_go: Some(20.0) };
    // yellow 15 m out at 12 m/s: no comfortable stop (27 m) - committed
    let call = light_call(&v, &p, &sig(&v, 15.0, Aspect::Yellow));
    assert!(call.go && call.decision.is_some_and(|d| d.1), "{call:?}");
    v.light_decision = call.decision.map(|(l, go, gap)| (l, go, v.state.odometer + gap));
    // red 10 m out, could stop hard (11.25 m at 6.4 m/s2 is more: cannot) - and committed anyway
    assert!(light_call(&v, &p, &sig(&v, 10.0, Aspect::Red)).go);
    // standing with the decision 0.5 m short: waits
    v.state.speed = 0.0;
    assert!(!light_call(&v, &p, &sig(&v, 10.0, Aspect::Red)).go);
    // yellow 40 m out at 12 m/s: stops, and keeps stopping as it comes nearer
    let mut w = standing(VehicleClass::Car);
    w.state.front = 2.0;
    w.state.speed = 12.0;
    let call = light_call(&w, &p, &sig(&w, 40.0, Aspect::Yellow));
    assert!(!call.go && call.decision.is_some_and(|d| !d.1));
    w.light_decision = call.decision.map(|(l, go, gap)| (l, go, w.state.odometer + gap));
    w.state.speed = 8.0;
    w.state.odometer += 25.0;
    assert!(!light_call(&w, &p, &sig(&w, 15.0, Aspect::Yellow)).go, "changed its mind");
    // red, no decision, 30 m out at 12 m/s: stops
    let x = Vehicle { light_decision: None, ..w.clone() };
    assert!(!light_call(&x, &p, &sig(&x, 30.0, Aspect::Red)).go);
}

/// [round 7 A2-full] Two spline lanes (no crossing object) converging into one lane have a
/// compiled meeting place - a merge, the side road giving way by the right-of-way order - so
/// the junction table decides there; two cars arriving together do not meet in the joint.
#[test]
fn two_spline_lanes_converging_into_one_have_a_compiled_merge_and_do_not_meet() {
    use glam::DVec3;
    use omsi_sim::traffic::{LaneBuilder, LaneKind, Network};
    use omsi_traffic::{FoeKind, Junctions, Simulation};
    let lane = |pts: &[(f64, f64)], prio: f32| {
        let mut l = LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), LaneKind::Street, 3.5);
        l.priority = prio;
        l
    };
    // 0: the main road, 1: a side road joining it from the right at 30 degrees, 2: on
    let lanes = vec![lane(&[(0.0, 0.0), (200.0, 0.0)], 128.0), lane(&[(30.0, -90.0), (120.0, -52.0), (200.0, 0.0)], 64.0), lane(&[(200.0, 0.0), (400.0, 0.0)], 128.0)];
    let mut net = Network { lanes, ..Default::default() };
    net.link(1.5);
    net.build_grid();
    net.compute_conflicts();
    assert_eq!(net.prev[2].len(), 2, "both run into lane 2");
    let js = Junctions::compile(&net);
    let side = js.foe_between(1, 0).expect("the join has a meeting place");
    assert_eq!(side.kind, FoeKind::Merge);
    assert!(side.yields, "the side road gives way to the main road");
    assert!(!js.foe_between(0, 1).expect("both ways").yields);
    // two cars timed to reach the joint together: the side road's waits, nobody touches
    let car = |id: u64, l: usize, s: f32| {
        let mut st = AiState::new(l, s, id);
        st.speed = 12.0;
        st.front = 2.0;
        st.rear = 2.0;
        st.length = 4.0;
        st.max_speed_kmh = 60.0;
        let mut v = Vehicle::new(id, VehicleClass::Car, st, 0.9, id * 7919);
        v.fresh = 0.0;
        v.state.plan_next(&net);
        v
    };
    let side_len = net.lanes[1].length();
    let mut vs = vec![car(1, 0, 100.0), car(2, 1, side_len - 100.0)];
    let mut sim = Simulation::new(&net, vec![]);
    let mut closest = f64::INFINITY;
    let mut side_held = false;
    for _ in 0..(20.0 / DT) as usize {
        sim.tick(&net, DT, &mut vs);
        sim.update_positions(&net, &mut vs);
        closest = closest.min(vs[0].obb().separation(&vs[1].obb()));
        side_held |= vs[1].last.why == Why::Junction;
    }
    assert!(side_held, "the side road's car never gave way");
    assert!(closest > 0.3, "the two came within {closest:.2} m");
}
