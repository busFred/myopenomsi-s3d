//! Headless scenarios for round 5 area J (junctions and signals): detector lights, the parked
//! player's claim, the dice at a mutual hold, a queue rolling up behind a car that waits for
//! me, box entry, the side-merge hold, and the yielding merger's window.

use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network, TrafficLightController};
#[allow(unused_imports)]
use omsi_traffic::sim::PLAYER_ID_BASE;
#[allow(unused_imports)]
use omsi_traffic::{PlayerView, Simulation, Vehicle, VehicleClass, Why};

const DT: f32 = 1.0 / 30.0;

fn lane(pts: &[(f64, f64)], kind: LaneKind, width: f32) -> omsi_sim::traffic::Lane {
    LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), kind, width)
}

/// A spline lane (source 1).
fn road(pts: &[(f64, f64)], id: i64) -> omsi_sim::traffic::Lane {
    let mut l = lane(pts, LaneKind::Street, 3.0);
    l.source = 1;
    l.key = Some(LaneKey { tile: (0, 0), id, path: 0 });
    l
}

/// A lane of crossing object 9 (source 2).
fn piece(pts: &[(f64, f64)], path: u16, priority: f32) -> omsi_sim::traffic::Lane {
    let mut l = lane(pts, LaneKind::Street, 3.0);
    l.source = 2;
    l.key = Some(LaneKey { tile: (0, 0), id: 9, path });
    l.priority = priority;
    l
}

fn finish(mut net: Network) -> Network {
    net.link(1.5);
    net.compute_conflicts();
    net
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

/// Run `secs`; `each` sees every frame. Returns the smallest gap between two bodies.
#[allow(dead_code)]
fn run(net: &Network, sim: &mut Simulation, vehicles: &mut Vec<Vehicle>, secs: f32, mut each: impl FnMut(f32, &mut Simulation, &mut [Vehicle])) -> f32 {
    let mut min_gap = f32::INFINITY;
    let mut t = 0.0;
    while t < secs {
        sim.tick(net, DT, vehicles);
        for &i in &sim.removed {
            vehicles.remove(i);
        }
        for (a, va) in vehicles.iter().enumerate() {
            for vb in vehicles.iter().skip(a + 1) {
                min_gap = min_gap.min(va.obb().separation(&vb.obb()) as f32);
            }
        }
        each(t, sim, vehicles.as_mut_slice());
        t += DT;
    }
    min_gap
}

fn front_x(v: &Vehicle) -> f32 {
    v.pos.x as f32 + v.state.front
}

// ---- J1: detector lights ----

/// X10's Buergertreff approach (case04): the approach road (0), a 1 m piece with light 0 (1),
/// the 9.7 m piece with light 1 whose line is the real stop line (2), the box (3), the exit (4).
fn detector_approach() -> Network {
    let mut net = Network::default();
    net.lanes.push(road(&[(-300.0, 0.0), (-11.0, 0.0)], 1));
    let mut loop_lane = piece(&[(-11.0, 0.0), (-10.0, 0.0)], 0, 128.0);
    loop_lane.traffic_light = Some((0, 0));
    net.lanes.push(loop_lane);
    let mut line = piece(&[(-10.0, 0.0), (-0.3, 0.0)], 1, 128.0);
    line.traffic_light = Some((0, 1));
    net.lanes.push(line);
    net.lanes.push(piece(&[(-0.3, 0.0), (20.0, 0.0)], 2, 128.0));
    net.lanes.push(road(&[(20.0, 0.0), (300.0, 0.0)], 2));
    finish(net)
}

/// Light 0 with `program`, light 1 red for good: where the car stops (its front's x), whether
/// the request of light 0 was ever raised, and the first signal the car saw from 60 m out.
fn runs_up_to(program: Vec<(i32, f32)>) -> (f32, bool, Option<(usize, usize)>) {
    let net = detector_approach();
    let ctl = TrafficLightController::new(vec![program, vec![(0, 100.0)]], 100.0);
    let mut sim = Simulation::new(&net, vec![ctl]);
    let mut cars = vec![car(1, 0, 230.0, 10.0)];
    let mut requested = false;
    let mut first_seen: Option<(usize, usize)> = None;
    run(&net, &mut sim, &mut cars, 20.0, |_, sim, cars| {
        requested |= sim.lights[0].request.first().copied().unwrap_or(false);
        if first_seen.is_none() && cars[0].pos.x > -60.0 {
            let p = omsi_traffic::perception::perceive(&net, sim, cars, 0);
            first_seen = p.signal.map(|s| (s.ctl, s.light));
        }
    });
    (front_x(&cars[0]), requested, first_seen)
}

#[test]
fn j1_a_constant_green_detector_before_the_line_hides_no_red_and_keeps_its_request() {
    // the Loop light: one phase, green, 1 s (X10 Buergertreff 0.13)
    let (front, requested, seen) = runs_up_to(vec![(6, 1.0)]);
    assert_eq!(seen, Some((0, 1)), "from 60 m out the car sees the real line, not the detector");
    assert!(front <= -10.0 + 0.3, "stood with its front at x = {front:.1}, over the red line at -10 (was: at -0.8 with the detector green in front of it)");
    assert!(front > -16.0, "stood at x = {front:.1}: far short of the line");
    assert!(requested, "the detector's request was never raised");
}

#[test]
fn j1_a_multi_phase_light_held_green_by_a_stop_point_is_a_stop_line_not_a_detector() {
    // a `[traffic_light_stop]`-held green: two phases, the clock held in the green by a stop
    // point until a request comes (the program's green lasts, but it is a real signal)
    let net = detector_approach();
    let mut ctl = TrafficLightController::new(vec![vec![(6, 10.0), (0, 90.0)], vec![(0, 100.0)]], 100.0);
    ctl.stops.push(omsi_sim::traffic::LightStop { light: 0, time: 5.0, if_request: false, jump_to: None });
    let sim = Simulation::new(&net, vec![ctl]);
    assert!(!omsi_traffic::perception::detector_light(&sim, 0, 0));
    assert!(omsi_traffic::perception::detector_light(&sim, 0, 1) == false, "a constant red is no detector");
    // a dark lane (state 12, one phase) is no detector either: it is reported as before
    let dark = Simulation::new(&net, vec![TrafficLightController::new(vec![vec![(12, 1.0)], vec![(0, 100.0)]], 100.0)]);
    assert!(!omsi_traffic::perception::detector_light(&dark, 0, 0));
    // and under a two-phase green first line the second line, carrying another light, is its
    // own link (a BeijingCity lead-in line and its arrow line, CHANGELOG leftarrow-1): its red is
    // the one seen from 60 m out (a green line before a red one holds nobody), and the car
    // stops at it
    let (front, _, seen) = runs_up_to(vec![(6, 10.0), (0, 90.0)]);
    assert_eq!(seen, Some((0, 1)), "the red second line is the one seen");
    assert!(front <= -10.0 + 0.3 && front > -16.0, "stops at the second line's red: front {front:.1}");
}

// ---- J2: a parked player beside the lane makes no junction claim ----

/// A side road piece (3) merging into the main road's piece (1) at its end, giving way to it;
/// the main road approach (0) and exit (2), the side approach (4).
fn side_merge() -> Network {
    let mut net = Network::default();
    net.lanes.push(road(&[(-200.0, 0.0), (-10.0, 0.0)], 1)); // 0
    net.lanes.push(piece(&[(-10.0, 0.0), (10.0, 0.0)], 0, 192.0)); // 1
    net.lanes.push(road(&[(10.0, 0.0), (300.0, 0.0)], 2)); // 2
    net.lanes.push(piece(&[(-10.0, -8.0), (10.0, 0.0)], 1, 64.0)); // 3
    net.lanes.push(road(&[(-200.0, -8.0), (-10.0, -8.0)], 3)); // 4
    finish(net)
}

/// The player's bus stands (10 s) at `(x, y)` heading east; the merger rolls up the side road.
/// (how far east the merger got in 20 s, whether it ever waited for the player)
fn merger_past_standing_player(x: f64, y: f64) -> (f32, bool) {
    let net = side_merge();
    let js = omsi_traffic::Junctions::compile(&net);
    let (_, side) = js.movement(3).unwrap();
    assert!(side.foes.iter().any(|f| f.yields && f.kind == omsi_traffic::FoeKind::Merge), "{:?}", side.foes);
    let mut sim = Simulation::new(&net, vec![]);
    let player = PlayerView { id: 7, pos: DVec3::new(x, y, 0.0), heading: 90.0, half_len: 6.0, half_width: 1.25, speed: 0.0, still: 10.0, blinker: 0, signalling: 0.0, priority: false };
    sim.players = vec![player];
    let mut cars = vec![car(1, 4, 130.0, 8.0)];
    let mut waited = false;
    run(&net, &mut sim, &mut cars, 20.0, |_, sim, cars| {
        // ([J7-2] at a merge that crosses nothing the bus is the zipper's foe ahead: the
        // queue-entry stop behind it, or a refusal for it)
        waited |= cars[0].last.wait_for == Some(PLAYER_ID_BASE + 7) || sim.claims.iter().any(|(c, g)| c.vehicle == 0 && g.zipper.is_some_and(|z| z.stop && z.foe == omsi_traffic::sim::player_claim(7)));
        sim.players = vec![player];
    });
    (cars[0].pos.x as f32, waited)
}

#[test]
fn j2_a_player_standing_3m4_beside_the_merging_piece_holds_no_merger() {
    // the bus's centre 3.4 m left of the main piece's axis (its right side 2.15 m off, the
    // lane's edge at 1.5): parked beside the lane, as x10 case10's bus on the cycle track
    // (left, away from the side road: its body is in nobody's corridor)
    let (x, waited) = merger_past_standing_player(5.0, 3.4);
    assert!(!waited, "the merger waited for the parked player");
    assert!(x > 30.0, "the merger got only to x = {x:.1}");
    // (and further off)
    let (x, waited) = merger_past_standing_player(5.0, 4.5);
    assert!(!waited && x > 30.0, "4.5 m off: waited {waited}, x = {x:.1}");
}

#[test]
fn j2_a_player_stopped_in_the_box_keeps_its_claim() {
    // the bus stands with its body over the lane (0.8 m off the axis): the merger waits for
    // it and does not come in beside it
    let (x, waited) = merger_past_standing_player(5.0, -0.8);
    assert!(waited, "the merger never waited for the bus standing in the lane");
    assert!(x < 8.0, "the merger drove on to x = {x:.1} with the bus standing in the merge");
    // body clear of the lane's edge by less than PARKED_CLEAR: still a claim
    let (_, waited) = merger_past_standing_player(5.0, 1.5 + 1.25 + 0.2);
    assert!(waited, "a bus 0.2 m off the lane's edge is not parked beside it");
}


// ---- J5: box entry behind a queue forming on the exit ----

/// A main road (approach 0, piece 1 x -15..15, exit 2) crossed by a side road (approach 3,
/// piece 4 y -15..15, exit 5); the main road has the right of way.
fn cross_with_exit() -> Network {
    let mut net = Network::default();
    net.lanes.push(road(&[(-300.0, 0.0), (-15.0, 0.0)], 1));
    net.lanes.push(piece(&[(-15.0, 0.0), (15.0, 0.0)], 0, 192.0));
    net.lanes.push(road(&[(15.0, 0.0), (600.0, 0.0)], 2));
    net.lanes.push(road(&[(0.0, -300.0), (0.0, -15.0)], 3));
    net.lanes.push(piece(&[(0.0, -15.0), (0.0, 15.0)], 1, 64.0));
    net.lanes.push(road(&[(0.0, 15.0), (0.0, 300.0)], 4));
    finish(net)
}

/// A vehicle on the exit lane with its tail at x = `tail` (the crossing's conflict ends about
/// 2.6 m past the middle; the exit's first 15 m reach x = 17.6) at `kmh` (its top speed, so
/// it keeps crawling), standing when `stands`, a bus at its stop when `at_stop`, with another
/// car standing `lead_gap` m ahead of it when given; the car comes up the main road at
/// 10 m/s, 40 m out. (held for the full exit at all, the car's x at the end)
fn enters_behind(tail: f32, kmh: f32, stands: bool, at_stop: bool, lead_gap: Option<f32>) -> (bool, f32) {
    enters_behind_from(40.0, tail, kmh, stands, at_stop, lead_gap)
}

/// `enters_behind` with the car `out` m before the box.
fn enters_behind_from(out: f32, tail: f32, kmh: f32, stands: bool, at_stop: bool, lead_gap: Option<f32>) -> (bool, f32) {
    let net = cross_with_exit();
    let mut sim = Simulation::new(&net, vec![]);
    let mut ahead = car(9, 2, tail - 15.0 + 2.0, kmh / 3.6);
    ahead.state.max_speed_kmh = kmh;
    ahead.state.desire = 1.0;
    if stands {
        ahead.external.at_stop = at_stop;
        ahead.external.stop_at = Some(0.1);
        ahead.state.speed = 0.0;
    }
    let mut cars = vec![ahead, car(1, 0, 285.0 - out - 2.0, 10.0)];
    if let Some(gap) = lead_gap {
        let mut lead = car(8, 2, tail - 15.0 + 2.0 + 2.0 + gap + 2.0, 0.0);
        lead.external.stop_at = Some(0.1);
        cars.push(lead);
    }
    let mut held_exit = false;
    run(&net, &mut sim, &mut cars, 25.0, |_, _, cars| {
        let c = cars.iter().find(|c| c.id == 1).unwrap();
        held_exit |= c.state.speed < 0.3 && c.last.why == Why::ExitFull;
    });
    (held_exit, cars.iter().find(|c| c.id == 1).unwrap().pos.x as f32)
}

#[test]
fn j5_a_car_crawling_in_the_exits_first_metres_closes_the_box_a_bus_at_its_stop_does_not() {
    // crawling at 1 m/s, its tail 7.4 m past the conflict when the car is 40 m out (room
    // enough by the gap alone, 7.4 >= 4 + 2): the box is not entered until it has crawled out
    // of the first 15 m
    let (held, x) = enters_behind(10.0, 3.6, false, false, None);
    assert!(held, "the car entered the box behind a crawler (x = {x:.1})");
    assert!(x > 15.0, "the car never went on after the crawler left: x = {x:.1}");
    // at 1.7 m/s too (EXIT_SLOW 2.0, was 1.5): its tail 6.1 m past the conflict (room by the
    // gap alone), the car 20 m out; when the car is at the line the crawler is 12 m past
    let (held, _) = enters_behind_from(20.0, 8.7, 1.7 * 3.6, false, false, None);
    assert!(held, "the car entered the box behind a 1.7 m/s crawler");
    // rolling at 3 m/s: a queue moving off, entered as before
    let (held, _) = enters_behind(10.0, 10.8, false, false, None);
    assert!(!held, "held for the exit behind a car rolling at 3 m/s");
    // a bus serving its stop 12 m past the conflict: entered as before
    let (held, _) = enters_behind(15.0, 0.0, true, true, None);
    assert!(!held, "held for the exit behind a bus at its stop");
}

#[test]
fn j5_a_car_standing_where_my_queue_place_would_lie_in_the_exits_first_metres_closes_the_box() {
    // standing 12.4 m past the conflict: 12.4 m of room by the gap, but queued behind it my
    // tail would stand 6 m back, inside the exit's first 15 m: no room (x10 case06 #11146
    // entered behind a car stopping 13 m past and stood with its tail across the through lane)
    let (held, _) = enters_behind(15.0, 0.0, true, false, None);
    assert!(held, "the car entered the box behind a car standing 12 m past the conflict");
    // standing 8 m past: the same
    let (held, _) = enters_behind(10.6, 0.0, true, false, None);
    assert!(held, "the car entered the box behind a car standing 8 m past the conflict");
    // standing 25 m past (my queue place 19 m past, beyond the first 15 m): entered, as before
    let (held, _) = enters_behind(27.6, 0.0, true, false, None);
    assert!(!held, "held for the exit behind a car standing 25 m past the conflict");
    // rolling at 3 m/s 25 m past towards a car standing 5 m ahead of it: it stops with its
    // tail 28 m past, no queue in the first metres, entered
    let (held, _) = enters_behind(27.6, 10.8, false, false, Some(5.0));
    assert!(!held, "held for the exit behind a queue forming 25 m past the conflict");
    // standing 4 m past: no room by the gap, as before
    let (held, _) = enters_behind(6.6, 0.0, true, false, None);
    assert!(held, "the car entered the box with 4 m of room past the conflict");
}

#[test]
fn j5_the_players_bus_beside_the_exit_lane_does_not_close_the_box_in_it_does() {
    // the player's bus standing with its tail 4 m past the conflict, 3.5 m left of the exit
    // lane's axis (a queue in the lane beside, snapped to this one): room; in the lane: none
    for (lateral, expect_held) in [(3.5f64, false), (0.0, true)] {
        let net = cross_with_exit();
        let mut sim = Simulation::new(&net, vec![]);
        let player = PlayerView { id: 7, pos: DVec3::new(6.6 + 6.0, lateral, 0.0), heading: 90.0, half_len: 6.0, half_width: 1.25, speed: 0.0, still: 10.0, blinker: 0, signalling: 0.0, priority: false };
        sim.players = vec![player];
        let mut cars = vec![car(1, 0, 285.0 - 40.0 - 2.0, 10.0)];
        let mut held_exit = false;
        run(&net, &mut sim, &mut cars, 12.0, |_, sim, cars| {
            held_exit |= cars[0].state.speed < 0.3 && cars[0].last.why == Why::ExitFull;
            sim.players = vec![player];
        });
        assert_eq!(held_exit, expect_held, "player {lateral} m beside the exit lane: held for the exit {held_exit}");
    }
}

// ---- J6: the side-merge hold rolls up behind the queue in the twin chain ----

/// viz4 case02's object: the lead-in piece (1) forks into two chains running side by side to
/// two exit lanes: A (2, 3, 4: x 0..6, 6..20, 20..60, bending right to y = -2.8) and B (6, 7,
/// 8, straight along y = 0; the chains 1.6 m apart, linked along themselves only); approach
/// 0, exits 5 (A) and 9 (B). With `pocket`, a side road (10) comes in from the north into B's
/// last piece: another entry into the chain.
fn twin_chains(pocket: bool) -> Network {
    let mut net = Network::default();
    net.lanes.push(road(&[(-200.0, 0.0), (-10.0, 0.0)], 1)); // 0
    net.lanes.push(piece(&[(-10.0, 0.0), (0.0, 0.0)], 0, 128.0)); // 1
    net.lanes.push(piece(&[(0.0, 0.0), (6.0, -1.6)], 1, 128.0)); // 2
    net.lanes.push(piece(&[(6.0, -1.6), (20.0, -1.6)], 2, 128.0)); // 3
    net.lanes.push(piece(&[(20.0, -1.6), (60.0, -2.8)], 3, 128.0)); // 4
    net.lanes.push(road(&[(60.0, -2.8), (300.0, -2.8)], 2)); // 5
    net.lanes.push(piece(&[(0.0, 0.0), (6.0, 0.0)], 4, 128.0)); // 6
    net.lanes.push(piece(&[(6.0, 0.0), (20.0, 0.0)], 5, 128.0)); // 7
    net.lanes.push(piece(&[(20.0, 0.0), (60.0, 0.0)], 6, 128.0)); // 8
    net.lanes.push(road(&[(60.0, 0.0), (300.0, 0.0)], 3)); // 9
    if pocket {
        net.lanes.push(road(&[(10.0, 200.0), (16.0, 1.5), (20.0, 0.0)], 4)); // 10
    }
    finish(net)
}

/// A car stands held in B's last piece 12 m in; the car on the approach, bound for A, comes
/// up at 8 m/s: where its front stands at the end (x), and whether it ever waited for the
/// standing one.
fn rolls_up_in_twin_chains(pocket: bool) -> (f32, bool) {
    let net = twin_chains(pocket);
    let js = omsi_traffic::Junctions::compile(&net);
    let (_, a3) = js.movement(4).unwrap();
    assert!(a3.foes.iter().any(|f| f.side && js.nodes[0].movements[f.movement].lane == 8), "A3/B3 a side merge: {:?}", a3.foes);
    let (_, a2) = js.movement(3).unwrap();
    assert!(a2.foes.iter().any(|f| f.side && js.nodes[0].movements[f.movement].lane == 7), "A2/B2 a side merge: {:?}", a2.foes);
    let mut sim = Simulation::new(&net, vec![]);
    let mut blocker = car(9, 8, 12.0, 0.0);
    blocker.external.stop_at = Some(0.1);
    let mut mine = car(1, 0, 120.0, 8.0);
    mine.state.planned_next = Some(1);
    mine.state.ahead = vec![2, 3, 4, 5];
    let mut cars = vec![blocker, mine];
    let mut waited = false;
    let min_gap = run(&net, &mut sim, &mut cars, 25.0, |_, sim, cars| {
        let c = cars.iter().find(|c| c.id == 1).unwrap();
        // ([J7-2] the standing car is the zipper's foe ahead: the queue-entry stop behind it)
        let (me, it) = (cars.iter().position(|c| c.id == 1).unwrap(), cars.iter().position(|c| c.id == 9).unwrap());
        waited |= c.last.wait_for == Some(9) || sim.claims.iter().any(|(cl, g)| cl.vehicle == me && g.zipper.is_some_and(|z| z.stop && z.foe == it));
        cars.iter_mut().find(|c| c.id == 9).unwrap().state.speed = 0.0;
    });
    assert!(min_gap > 0.3, "touched: {min_gap:.2} m");
    let c = cars.iter().find(|c| c.id == 1).unwrap();
    (front_x(c), waited)
}

#[test]
fn j6_refused_for_the_twin_chains_queue_the_car_rolls_up_behind_it_not_to_the_line() {
    let (x, waited) = rolls_up_in_twin_chains(false);
    assert!(waited, "never refused for the standing car");
    // ([J7-2] the A3/B3 side merge is a zipper: the standing car is the foe ahead, and the car
    // joins its queue behind it, its standing gap behind its rear at x = 30 - well past where it
    // stood before round 6, x = 5; round 6 stopped it a metre before the stretch, x = 19)
    assert!(x > 26.0 && x <= 28.5, "stood with its front at x = {x:.1}");
}

#[test]
fn j6_a_chain_with_another_entry_queues_behind_the_standing_car_too() {
    // (viz3 case03: a stretch shared with turn pockets that have their own entries)
    // ([J7-2] the zipper's queue behind the standing car whatever comes into the chains: the
    // pocket's cars are zipper foes of mine at the same stretch, one file behind it, so the hold
    // at the line round 6 kept here for them is gone)
    let (x, waited) = rolls_up_in_twin_chains(true);
    assert!(waited, "never refused for the standing car");
    assert!(x > 26.0 && x <= 28.5, "stood with its front at x = {x:.1}");
}


// ---- follow-up: a standing car at an amber line stays ----

#[test]
fn a_car_standing_half_a_metre_before_an_amber_line_does_not_take_it() {
    // one lit piece after the approach; the light yellow for 6 s, then red; the car stands
    // with its front 0.5 m before the line (released there from a hold)
    let mut net = Network::default();
    net.lanes.push(road(&[(-300.0, 0.0), (-10.0, 0.0)], 1));
    let mut lit = piece(&[(-10.0, 0.0), (10.0, 0.0)], 0, 128.0);
    lit.traffic_light = Some((0, 0));
    net.lanes.push(lit);
    net.lanes.push(road(&[(10.0, 0.0), (300.0, 0.0)], 2));
    let net = finish(net);
    let ctl = TrafficLightController::new(vec![vec![(9, 6.0), (0, 60.0)]], 66.0);
    let mut sim = Simulation::new(&net, vec![ctl]);
    let mut cars = vec![car(1, 0, 290.0 - 0.5 - 2.0, 0.0)];
    run(&net, &mut sim, &mut cars, 12.0, |_, _, _| {});
    let front = front_x(&cars[0]);
    assert!(front <= -10.0 + 0.3, "took the amber from a standstill: front at x = {front:.1}");
    // rolling up at 8 m/s with 10 m to go (comfortable stop ~12 m): it takes the yellow
    let mut sim = Simulation::new(&net, vec![TrafficLightController::new(vec![vec![(9, 6.0), (0, 60.0)]], 66.0)]);
    let mut cars = vec![car(1, 0, 290.0 - 10.0 - 2.0, 8.0)];
    run(&net, &mut sim, &mut cars, 6.0, |_, _, _| {});
    assert!(front_x(&cars[0]) > 10.0, "a rolling car in the dilemma zone stopped: x = {:.1}", front_x(&cars[0]));
}

/// `twin_chains` with a side road (10) crossing the lead-in piece at x = -5: a crossing
/// before the chains, the only conflict that is no side merge.
fn twin_chains_crossed() -> Network {
    let mut net = twin_chains(false);
    net.lanes.push(piece(&[(-5.0, -20.0), (-5.0, 20.0)], 7, 128.0)); // 10
    net.lanes.push(road(&[(-5.0, 20.0), (-5.0, 200.0)], 5)); // 11
    net.lanes.push(road(&[(-5.0, -200.0), (-5.0, -20.0)], 6)); // 12
    finish(net)
}

#[test]
fn j5_the_room_past_the_crossing_counts_a_car_just_past_it_not_the_side_stretch_queue_40_m_on() {
    // the crossing's stretch on the lead-in ends about 1.3 m past x = -5; a car standing in
    // my chain's first piece (x 0..6) with its tail at x = -1 leaves 2.3 m past the crossing:
    // no room; one standing in my chain's last piece with its tail 60 m past the crossing
    // does not close the box (before, the "box" reached to the end of the 40 m side stretch,
    // 1.5 m short of that tail, and the car stood at the entry)
    for (lane, s, expect_held) in [(2usize, 1.0f32, true), (4, 38.0, false)] {
        let net = twin_chains_crossed();
        let mut sim = Simulation::new(&net, vec![]);
        let mut blocker = car(9, lane, s, 0.0);
        blocker.external.stop_at = Some(0.1);
        let mut mine = car(1, 0, 120.0, 8.0);
        mine.state.planned_next = Some(1);
        mine.state.ahead = vec![2, 3, 4, 5];
        let mut cars = vec![blocker, mine];
        let mut held_exit = false;
        run(&net, &mut sim, &mut cars, 20.0, |_, _, cars| {
            let c = cars.iter().find(|c| c.id == 1).unwrap();
            held_exit |= c.state.speed < 0.3 && c.last.hold == Some(omsi_traffic::Hold::ExitFull);
            cars.iter_mut().find(|c| c.id == 9).unwrap().state.speed = 0.0;
        });
        assert_eq!(held_exit, expect_held, "blocker on lane {lane} s {s}: held for the exit {held_exit}");
    }
}

#[test]
fn j5_a_side_merge_stretch_is_no_box_to_keep_clear() {
    // the twin chains (J6's object): a car stands in B's last piece near its END (s 38 of 40),
    // past the A3/B3 side stretch, its tail 1.5 m past that stretch's end: before, the exit
    // was "full" (clear_from = the side stretch's end, 1.5 m of room) and the car stood at the
    // entry; now the side stretch is no box: no exit hold, it drives on along A
    let net = twin_chains(false);
    let mut sim = Simulation::new(&net, vec![]);
    let mut blocker = car(9, 8, 38.0, 0.0);
    blocker.external.stop_at = Some(0.1);
    let mut mine = car(1, 0, 120.0, 8.0);
    mine.state.planned_next = Some(1);
    mine.state.ahead = vec![2, 3, 4, 5];
    let mut cars = vec![blocker, mine];
    let mut held_exit = false;
    run(&net, &mut sim, &mut cars, 25.0, |_, _, cars| {
        let c = cars.iter().find(|c| c.id == 1).unwrap();
        held_exit |= c.last.hold == Some(omsi_traffic::Hold::ExitFull);
        cars.iter_mut().find(|c| c.id == 9).unwrap().state.speed = 0.0;
    });
    assert!(!held_exit, "held for a full exit by a car standing beside the side stretch's end");
    let c = cars.iter().find(|c| c.id == 1).unwrap();
    assert!(front_x(c) > 40.0, "did not drive on: front at x = {:.1}", front_x(c));
}
