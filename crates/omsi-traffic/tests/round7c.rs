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

/// [round 7 A1] The way on at a fork follows the access rule (`open_to`): an open lane
/// whenever there is one; with none open, one closed only by density before one closed by
/// the rules; never a `no_cars` lane.
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
    // only no_cars lanes: no way on
    let mut net = fork(false);
    net.lanes[2].no_cars = true;
    assert!(ways(&net).iter().all(|w| w.is_none()), "{:?}", ways(&net));
}

/// [round 7 A6] The mark of a line taken on green or yellow carries a car over its red only
/// while it rolls (round 6 J3): standing with the mark half a metre short of the line it is not "marked
/// through", so its claim files the red like any car standing there.
#[test]
fn a_marked_car_standing_at_its_red_is_not_marked_through() {
    use omsi_sim::traffic::Aspect;
    use omsi_traffic::{plan::light_call, Perception, SignalView};
    let mut v = standing(VehicleClass::Car);
    v.amber = Some((0, 0));
    v.state.front = 2.0;
    let sig = |aspect| SignalView { dist: v.state.front + 0.5, ctl: 0, light: 0, aspect, until_go: Some(20.0) };
    let p = Perception::default();
    for aspect in [Aspect::Red, Aspect::Yellow] {
        let call = light_call(&v, &p, &sig(aspect));
        assert!(!call.go && !call.marked_through, "standing at {aspect:?}: {call:?}");
    }
    // rolling at 2 m/s there: through
    v.state.speed = 2.0;
    let call = light_call(&v, &p, &sig(Aspect::Red));
    assert!(call.go && call.marked_through, "rolling: {call:?}");
}
