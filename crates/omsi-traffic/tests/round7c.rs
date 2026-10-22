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

/// [round 7 B1] SUMO's waiting time: a random car standing in a jam, at a yield, for people or
/// held by nothing gives up after `GIVE_UP_UNSEEN`; at a red light the clock pauses (it does
/// not restart); moving restarts it and takes the request back.
#[test]
fn a_random_car_gives_up_only_held_by_nothing_for_minutes_and_comes_back_when_it_moves() {
    let give_up = omsi_traffic::sim::GIVE_UP_UNSEEN;
    assert!(give_up >= 300.0);
    // at a red light or for people: never, however long
    let mut v = standing(VehicleClass::Car);
    assert!(!stand(&mut v, Why::Light, give_up + 60.0), "gave up at a red light");
    assert_eq!(v.waiting, 0.0);
    // a jam, a yield: they count, and a red in between only pauses the clock
    assert!(!stand(&mut v, Why::Lead, give_up * 0.5));
    assert!(!stand(&mut v, Why::Light, 200.0));
    assert!(!stand(&mut v, Why::People, give_up * 0.5 - 1.0));
    assert!(stand(&mut v, Why::None, 2.0), "waited the give-up time");
    // a creep is no progress; moving off takes the request back
    v.state.speed = 0.05;
    assert!(stand(&mut v, Why::None, 1.0));
    v.state.speed = 0.5;
    assert!(!stand(&mut v, Why::None, DT), "still given up while it drives on");
    assert_eq!(v.waiting, 0.0);
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
        // ([J7-2] the side road's car gives way by the zipper: capped by the main road car ahead
        // of it, or refused)
        side_held |= vs[1].last.why == Why::Junction || sim.claims.iter().any(|(c, g)| c.vehicle == 1 && g.zipper.is_some_and(|z| z.foe == 0));
    }
    assert!(side_held, "the side road's car never gave way");
    assert!(closest > 0.3, "the two came within {closest:.2} m");
}

/// [fix pass 1, 13] The swerve round a parked car protruding into the lane is measured from the
/// lane's axis, not from the swerved way: its target holds steady while the car passes.
#[test]
fn the_swerve_round_a_parked_car_does_not_flicker() {
    use glam::DVec3;
    use omsi_sim::traffic::{LaneBuilder, LaneKind, Network};
    use omsi_traffic::{Parked, Simulation};
    let mut net = Network { lanes: vec![LaneBuilder::polyline(vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(400.0, 0.0, 0.0)], LaneKind::Street, 3.5)], ..Default::default() };
    net.link(1.5);
    net.build_grid();
    net.compute_conflicts();
    let mut sim = Simulation::new(&net, vec![]);
    // its flank 0.25 m into the lane's strip of a 0.9 m car
    sim.parked.insert(0, vec![Parked { s: 150.0, lateral: 2.0, half_width: 0.9, half_len: 2.2 }]);
    let mut st = AiState::new(0, 100.0, 1);
    st.speed = 8.0;
    st.front = 2.0;
    st.rear = 2.0;
    st.length = 4.0;
    let mut v = Vehicle::new(1, VehicleClass::Car, st, 0.9, 7919);
    v.fresh = 0.0;
    v.state.plan_next(&net);
    let mut vs = vec![v];
    let mut targets: Vec<f32> = Vec::new();
    for _ in 0..(8.0 / DT) as usize {
        sim.tick(&net, DT, &mut vs);
        if (vs[0].state.s - 150.0).abs() < 25.0 {
            targets.push(vs[0].state.lateral_target);
        }
    }
    let flips = targets.windows(2).filter(|w| (w[0] - w[1]).abs() > 0.05).count();
    assert!(targets.iter().any(|&t| t < -0.05), "no swerve: {targets:?}");
    assert!(flips <= 2, "the swerve target changed {flips} times: {targets:?}");
}

/// [round 7 A8] The way is planned as far as the claim horizon: at 22 m/s a car plans 206 m
/// and more ahead (the junction claim is filed eight seconds out), not 90 m.
#[test]
fn the_way_is_planned_to_the_claim_horizon() {
    use glam::DVec3;
    use omsi_sim::traffic::{LaneBuilder, LaneKind, Network};
    let lanes: Vec<_> = (0..16).map(|k| LaneBuilder::polyline(vec![DVec3::new(30.0 * k as f64, 0.0, 0.0), DVec3::new(30.0 * (k + 1) as f64, 0.0, 0.0)], LaneKind::Street, 3.5)).collect();
    let mut net = Network { lanes, ..Default::default() };
    net.link(1.5);
    let mut st = AiState::new(0, 5.0, 1);
    st.speed = 22.0;
    st.plan_next(&net);
    let planned: f32 = 25.0 + st.upcoming().map(|l| net.lanes[l].length()).sum::<f32>();
    assert!(planned >= omsi_traffic::sim::claim_horizon(22.0), "planned {planned:.0} m");
    let mut slow = AiState::new(0, 5.0, 1);
    slow.speed = 5.0;
    slow.plan_next(&net);
    let planned: f32 = 25.0 + slow.upcoming().map(|l| net.lanes[l].length()).sum::<f32>();
    assert!(planned < 150.0, "slow: planned {planned:.0} m");
}

/// [fix pass 1, 15] A vehicle record with a driver field the code no longer knows still
/// reads (`yellow_brake`, removed in round 7 A5, is in every recording before it), and one
/// missing a driver field reads too, the field as the car preset's.
#[test]
fn a_vehicle_record_without_a_removed_driver_field_still_reads() {
    let v = standing(VehicleClass::Car);
    let text = omsi_traffic::finite::to_string(&v).expect("written");
    let old = text.replacen("\"tau\":", "\"yellow_brake\":2.5,\"tau\":", 1);
    assert!(old.contains("yellow_brake"), "an old record carries the field");
    let back: Vehicle = omsi_traffic::finite::from_str(&old).expect("reads with yellow_brake");
    assert_eq!(back.params, v.params);
    let without = text.replacen("\"impatience_max\":1.0,", "", 1).replacen(",\"impatience_max\":1.0", "", 1);
    assert!(!without.contains("impatience_max"), "the field is gone from the text");
    let mut back: Vehicle = omsi_traffic::finite::from_str(&without).expect("reads without impatience_max");
    back.params.fill_missing(back.class);
    assert_eq!(back.params.impatience_max, omsi_traffic::DriverParams::preset(VehicleClass::Car).impatience_max);
    // a bus's missing field is a bus's (`fill_missing` takes the class's preset)
    let bus = standing(VehicleClass::ScheduledBus);
    let text = omsi_traffic::finite::to_string(&bus).expect("written");
    let needle = format!("\"comfort_brake\":{}", bus.params.comfort_brake);
    let without = text.replacen(&format!("{needle},"), "", 1).replacen(&format!(",{needle}"), "", 1);
    assert!(!without.contains("comfort_brake"), "the field is gone from the text");
    let mut back: Vehicle = omsi_traffic::finite::from_str(&without).expect("reads without comfort_brake");
    back.params.fill_missing(back.class);
    assert_eq!(back.params.comfort_brake, omsi_traffic::DriverParams::preset(VehicleClass::ScheduledBus).comfort_brake);
    // and a vehicle missing whole fields of its own and of its way reads too
    let bare: Vehicle = omsi_traffic::finite::from_str("{\"id\":7,\"class\":\"Bus\",\"state\":{\"lane\":3,\"s\":12.5}}").expect("reads a bare vehicle");
    assert_eq!((bare.id, bare.state.lane), (7, 3));
    let with_more = text.replacen("{", "{\"a_field_from_elsewhere\":1,", 1);
    let _: Vehicle = omsi_traffic::finite::from_str(&with_more).expect("reads with an unknown field");
}

/// [round 7 A9] The player's pulling away is measured: 2 m/s² shown, the claim reckons 2;
/// it fades back towards `PLAYER_ACCEL` once the player cruises.
#[test]
fn the_players_pulling_away_is_measured() {
    use glam::DVec3;
    use omsi_sim::traffic::{LaneBuilder, LaneKind, Network};
    use omsi_traffic::{PlayerView, Simulation};
    let mut net = Network { lanes: vec![LaneBuilder::polyline(vec![DVec3::ZERO, DVec3::new(500.0, 0.0, 0.0)], LaneKind::Street, 3.5)], ..Default::default() };
    net.link(1.5);
    net.build_grid();
    net.compute_conflicts();
    let mut sim = Simulation::new(&net, vec![]);
    let player = |speed: f32| PlayerView { id: 3, pos: DVec3::new(50.0, -1.0, 0.0), heading: 90.0, half_len: 6.0, half_width: 1.25, speed, still: 0.0, blinker: 0, signalling: 0.0, priority: false };
    let mut vs: Vec<Vehicle> = Vec::new();
    assert_eq!(sim.player_accel(3), omsi_traffic::sim::PLAYER_ACCEL);
    let mut speed = 0.0f32;
    for _ in 0..(5.0 / DT) as usize {
        speed += 2.0 * DT;
        sim.players = vec![player(speed)];
        sim.tick(&net, DT, &mut vs);
    }
    assert!((sim.player_accel(3) - 2.0).abs() < 0.2, "measured {:.2}", sim.player_accel(3));
    for _ in 0..(60.0 / DT) as usize {
        sim.players = vec![player(speed)];
        sim.tick(&net, DT, &mut vs);
    }
    assert!(sim.player_accel(3) < 1.4, "faded to {:.2}", sim.player_accel(3));
}

/// [fix pass 2, 2] A lane whose stop line lies in a tight turn's swept path (the turn's axis
/// passing 3 m from its end: two 3 m lanes' half widths and a bus's off-tracking) meets the
/// turn there: a meeting place is compiled between them.
#[test]
fn a_stop_line_in_a_tight_turns_sweep_is_a_meeting_place() {
    use glam::DVec3;
    use omsi_sim::traffic::{LaneBuilder, LaneKey, LaneKind, Network};
    let piece = |pts: &[(f64, f64)], path: u16| {
        let mut l = LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), LaneKind::Street, 3.0);
        l.source = 2;
        l.key = Some(LaneKey { tile: (0, 0), id: 1, path });
        l
    };
    // a right turn of 10 m radius from (0, 0) heading north to (10, 10) heading east; a lane
    // of the same object ending at (3.5, 4.0), heading north-east, beside the turn's inside
    let turn: Vec<(f64, f64)> = (0..=9).map(|k| {
        let a = std::f64::consts::FRAC_PI_2 * k as f64 / 9.0;
        (10.0 - 10.0 * a.cos(), 10.0 * a.sin())
    }).collect();
    let lanes = vec![piece(&turn, 0), piece(&[(-6.0, -6.0), (3.5, 4.0)], 1)];
    let mut net = Network { lanes, ..Default::default() };
    net.link(1.5);
    net.build_grid();
    net.compute_conflicts();
    assert!(net.crossings[0].iter().any(|c| c.other == 1), "no meeting place: {:?}", net.crossings[0]);
}
