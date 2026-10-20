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
