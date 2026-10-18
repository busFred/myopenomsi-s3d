//! Headless scenarios for the junction rules of round 4: where a car held at a junction
//! stops (before the stretch it waits for), the player's claim over the pieces ahead, and a
//! car inside the box when its light turns red.

use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network, TrafficLightController};
use omsi_traffic::sim::PLAYER_ID_BASE;
use omsi_traffic::{PlayerView, Simulation, Vehicle, VehicleClass, Why};

const DT: f32 = 1.0 / 30.0;

fn lane(pts: &[(f64, f64)], kind: LaneKind) -> omsi_sim::traffic::Lane {
    LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), kind, 3.5)
}

/// A spline lane (source 1).
fn road(pts: &[(f64, f64)], id: i64) -> omsi_sim::traffic::Lane {
    let mut l = lane(pts, LaneKind::Street);
    l.source = 1;
    l.key = Some(LaneKey { tile: (0, 0), id, path: 0 });
    l
}

/// A lane of crossing object 9 (source 2).
fn piece(pts: &[(f64, f64)], path: u16, priority: f32) -> omsi_sim::traffic::Lane {
    let mut l = lane(pts, LaneKind::Street);
    l.source = 2;
    l.key = Some(LaneKey { tile: (0, 0), id: 9, path });
    l.priority = priority;
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

fn finish(mut net: Network) -> Network {
    net.link(1.5);
    net.compute_conflicts();
    net
}

/// Run `secs`; `each` sees every frame. Returns the smallest gap between two bodies.
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

/// The front bumper's x of a car heading east.
fn front_x(v: &Vehicle) -> f32 {
    v.pos.x as f32 + v.state.front
}

/// An eastbound way through object 9 (piece 1, x -15..45) crossed by an unlit side road at
/// x = 0 (piece 4, nobody on it), a zebra at x = 23 and a main road at x = 30 (piece 7,
/// priority 192: the eastbound way gives way to it). Lanes: 0 the approach, 1 the piece,
/// 2 the exit; 3/4/5 the side road; 6/7/8 the main road; 9 the zebra.
fn long_way_through(lit: bool) -> Network {
    let mut net = Network::default();
    net.lanes.push(road(&[(-300.0, 0.0), (-15.0, 0.0)], 1));
    let mut p = piece(&[(-15.0, 0.0), (45.0, 0.0)], 0, 128.0);
    if lit {
        p.traffic_light = Some((0, 0));
    }
    net.lanes.push(p);
    net.lanes.push(road(&[(45.0, 0.0), (400.0, 0.0)], 2));
    net.lanes.push(road(&[(0.0, -200.0), (0.0, -15.0)], 3));
    net.lanes.push(piece(&[(0.0, -15.0), (0.0, 15.0)], 1, 128.0));
    net.lanes.push(road(&[(0.0, 15.0), (0.0, 300.0)], 4));
    net.lanes.push(road(&[(30.0, -400.0), (30.0, -15.0)], 5));
    net.lanes.push(piece(&[(30.0, -15.0), (30.0, 15.0)], 2, 192.0));
    net.lanes.push(road(&[(30.0, 15.0), (30.0, 600.0)], 6));
    let mut zebra = lane(&[(23.0, -8.0), (23.0, 8.0)], LaneKind::Sidewalk);
    zebra.source = 2;
    zebra.key = Some(LaneKey { tile: (0, 0), id: 9, path: 3 });
    net.lanes.push(zebra);
    finish(net)
}

#[test]
fn rule_d_a_held_car_stops_just_before_the_stretch_it_waits_for_and_off_the_zebra() {
    let net = long_way_through(false);
    let js = omsi_traffic::Junctions::compile(&net);
    let (_, mv) = js.movement(1).unwrap();
    assert_eq!(mv.foes.len(), 2, "{:?}", mv.foes);
    assert!(mv.foes.iter().all(|f| f.yields), "{:?}", mv.foes);
    let mut sim = Simulation::new(&net, vec![]);
    // the eastbound car rolls up; a main road stream (a car every 30 m at 12 m/s) keeps it
    // waiting for the crossing at x = 30
    let mut cars = vec![car(1, 0, 230.0, 10.0)];
    for k in 0..8u64 {
        cars.push(car(10 + k, 6, 330.0 - 30.0 * k as f32, 12.0));
    }
    let mut stood_at: Option<f32> = None;
    let min_gap = run(&net, &mut sim, &mut cars, 30.0, |_, _, cars| {
        let me = cars.iter().find(|c| c.id == 1).unwrap();
        if me.state.speed < 0.1 && me.last.why == Why::Junction && stood_at.is_none() {
            stood_at = Some(front_x(me));
        }
    });
    let at = stood_at.expect("the eastbound car never stood for the main road");
    // past the side road's meeting place (x = 0, 2.6 m either side), its body off the zebra
    // (x = 23, 2 m either side), short of the main road's stretch (x = 30 - 2.6)
    assert!(at > 2.6 + 4.0 && at <= 23.0 - 2.5 + 0.2, "stood with its front at x = {at:.1}");
    assert!(min_gap > 0.3, "touched: {min_gap:.2} m");
}

#[test]
fn rule_d_on_red_the_held_car_stops_at_its_line() {
    let net = long_way_through(true);
    // the eastbound light red for good
    let ctl = TrafficLightController::new(vec![vec![(0, 60.0)]], 60.0);
    let mut sim = Simulation::new(&net, vec![ctl]);
    let mut cars = vec![car(1, 0, 230.0, 10.0)];
    for k in 0..8u64 {
        cars.push(car(10 + k, 6, 330.0 - 30.0 * k as f32, 12.0));
    }
    let mut furthest = f32::NEG_INFINITY;
    run(&net, &mut sim, &mut cars, 30.0, |_, _, cars| {
        let me = cars.iter().find(|c| c.id == 1).unwrap();
        furthest = furthest.max(front_x(me));
    });
    assert!(furthest <= -15.0 + 0.3, "its front got to x = {furthest:.1}, past its red line at -15");
}

/// Northbound (x = 2): approach 0, lead-in piece 1 (y -20..-5, meets nobody), crossing piece
/// 2 (y -5..20), exit 3. Southbound (x = -2): approach 4, a left turn 5 (south, then east
/// across the northbound way at y = 1, flagged turn 1), exit 6 (east).
fn oncoming_left_turn() -> Network {
    let mut net = Network::default();
    net.lanes.push(road(&[(2.0, -300.0), (2.0, -20.0)], 1));
    net.lanes.push(piece(&[(2.0, -20.0), (2.0, -5.0)], 0, 128.0));
    net.lanes.push(piece(&[(2.0, -5.0), (2.0, 20.0)], 1, 128.0));
    net.lanes.push(road(&[(2.0, 20.0), (2.0, 300.0)], 2));
    net.lanes.push(road(&[(-2.0, 200.0), (-2.0, 20.0)], 3));
    let mut turn = piece(&[(-2.0, 20.0), (-2.0, 5.0), (5.0, -2.0), (20.0, -2.0)], 2, 128.0);
    turn.turn = 1;
    net.lanes.push(turn);
    net.lanes.push(road(&[(20.0, -2.0), (300.0, -2.0)], 4));
    finish(net)
}

#[test]
fn rule_e_the_oncoming_left_turner_yields_to_the_player_going_straight() {
    let net = oncoming_left_turn();
    let js = omsi_traffic::Junctions::compile(&net);
    let (_, lead_in) = js.movement(1).unwrap();
    assert!(lead_in.foes.is_empty(), "{:?}", lead_in.foes);
    let (_, turner) = js.movement(5).unwrap();
    assert!(turner.foes.iter().any(|f| f.yields && js.nodes[0].movements[f.movement].lane == 2), "{:?}", turner.foes);
    let mut sim = Simulation::new(&net, vec![]);
    // the turner stands at its line; the player's bus comes up at 10 m/s, 60 m out
    let mut cars = vec![car(1, 4, 178.0, 0.0)];
    let mut player = PlayerView { id: 7, pos: DVec3::new(2.0, -80.0, 0.0), heading: 0.0, half_len: 6.0, half_width: 1.25, speed: 10.0, still: 0.0, blinker: 0, signalling: 0.0, priority: false };
    let (mut waited, mut entered_before) = (false, false);
    run(&net, &mut sim, &mut cars, 10.0, |_, sim, cars| {
        let t = &cars[0];
        waited |= t.last.wait_for == Some(PLAYER_ID_BASE + 7);
        // the turner's front across the northbound way (x > 0.5) while the bus has not passed
        let front = t.state.way_point(&net, t.state.front);
        if front.x > 0.5 && player.pos.y - (player.half_len as f64) < 4.0 {
            entered_before = true;
        }
        player.pos.y += player.speed as f64 * DT as f64;
        sim.players = vec![player];
    });
    assert!(waited, "the turner never waited for the player");
    assert!(!entered_before, "the turner turned across in front of the player");
}

/// A signalled crossing: eastbound (approach 0, piece 1 x -20..20 lit (0, 0), exit 2) and
/// northbound (approach 3, piece 4 y -20..20 lit (0, 1), exit 5), equal priorities (the
/// northbound way comes from the eastbound's right). Eastbound green 0-10 s, yellow 10-13,
/// red 13-40; northbound red 0-15, green 15-37, yellow 37-40.
fn signalled_cross() -> (Network, TrafficLightController) {
    let mut net = Network::default();
    net.lanes.push(road(&[(-300.0, 0.0), (-20.0, 0.0)], 1));
    let mut a = piece(&[(-20.0, 0.0), (20.0, 0.0)], 0, 128.0);
    a.traffic_light = Some((0, 0));
    net.lanes.push(a);
    net.lanes.push(road(&[(20.0, 0.0), (300.0, 0.0)], 2));
    net.lanes.push(road(&[(0.0, -300.0), (0.0, -20.0)], 3));
    let mut b = piece(&[(0.0, -20.0), (0.0, 20.0)], 1, 128.0);
    b.traffic_light = Some((0, 1));
    net.lanes.push(b);
    net.lanes.push(road(&[(0.0, 20.0), (0.0, 300.0)], 4));
    let ctl = TrafficLightController::new(vec![vec![(6, 10.0), (9, 3.0), (0, 27.0)], vec![(0, 15.0), (6, 22.0), (9, 3.0)]], 40.0);
    (finish(net), ctl)
}

#[test]
fn rule_f_a_car_in_the_box_when_its_light_turns_clears_it_and_the_fresh_green_waits() {
    let (net, ctl) = signalled_cross();
    let mut sim = Simulation::new(&net, vec![ctl]);
    // the eastbound car crawls (7 km/h) over its line at about 9 s, on green; it is still
    // short of the crossing when its light is red and the northbound one green
    let mut slow = car(1, 0, 280.0 - 2.0 - 1.95 * 9.0, 1.95);
    slow.state.max_speed_kmh = 7.0;
    slow.state.desire = 1.0;
    // the northbound car stands at its red line
    let fresh = car(2, 3, 280.0 - 2.0 - 0.5, 0.0);
    let mut cars = vec![slow, fresh];
    let (mut held_in_box, mut fresh_waited, mut t_cleared, mut t_fresh_in) = (false, false, None, None);
    let min_gap = run(&net, &mut sim, &mut cars, 30.0, |t, _, cars| {
        let (x, y) = (&cars[0], &cars[1]);
        let in_box = x.pos.x > -20.0 && x.pos.x - (x.state.rear as f64) < 20.0;
        if t > 13.0 && in_box && x.state.speed < 0.3 {
            held_in_box = true;
        }
        fresh_waited |= y.last.wait_for == Some(1);
        // the slow car's rear past the northbound way's meeting place (x = 2.6); the fresh
        // car's front into it (y = -2.6)
        if t_cleared.is_none() && x.pos.x as f32 - x.state.rear > 2.6 {
            t_cleared = Some(t);
        }
        if t_fresh_in.is_none() && y.pos.y as f32 + y.state.front > -2.6 {
            t_fresh_in = Some(t);
        }
    });
    assert!(!held_in_box, "the car in the box stood after its light went red");
    let cleared = t_cleared.expect("the slow car never cleared the crossing");
    assert!(cleared > 15.0, "(the scenario: still in the box at the fresh green, cleared at {cleared:.1} s)");
    assert!(fresh_waited, "the fresh green never waited for the car in the box");
    assert!(t_fresh_in.map(|t| t > cleared).unwrap_or(true), "the fresh green went in at {t_fresh_in:?} s, before the box was clear at {cleared:.1} s");
    assert!(min_gap > 0.3, "touched: {min_gap:.2} m");
}

#[test]
fn rule_f_the_mark_of_a_line_taken_on_green_does_not_carry_to_the_next_line_of_its_light() {
    // two lit pieces in a row under the same light (0, 0): object 9 at x -10..10, object 10
    // at x 110..130 (BeijingCity lights several approaches of one object with one light)
    let mut net = Network::default();
    net.lanes.push(road(&[(-300.0, 0.0), (-10.0, 0.0)], 1));
    let mut a = piece(&[(-10.0, 0.0), (10.0, 0.0)], 0, 128.0);
    a.traffic_light = Some((0, 0));
    net.lanes.push(a);
    net.lanes.push(road(&[(10.0, 0.0), (110.0, 0.0)], 2));
    let mut b = piece(&[(110.0, 0.0), (130.0, 0.0)], 0, 128.0);
    b.key = Some(LaneKey { tile: (0, 0), id: 10, path: 0 });
    b.traffic_light = Some((0, 0));
    net.lanes.push(b);
    net.lanes.push(road(&[(130.0, 0.0), (600.0, 0.0)], 3));
    let net = finish(net);
    // green 0-6 s, yellow 6-9, red 9-60
    let ctl = TrafficLightController::new(vec![vec![(6, 6.0), (9, 3.0), (0, 51.0)]], 60.0);
    let mut sim = Simulation::new(&net, vec![ctl]);
    // over the first line at about 3 s, on green, at 10 m/s
    let mut cars = vec![car(1, 0, 258.0, 10.0)];
    let (mut marked, mut furthest) = (false, f32::NEG_INFINITY);
    run(&net, &mut sim, &mut cars, 40.0, |_, _, cars| {
        marked |= cars[0].amber.is_some();
        furthest = furthest.max(front_x(&cars[0]));
    });
    assert!(marked, "it never took the first line on green");
    assert!(furthest <= 110.3, "its front got to x = {furthest:.1}, over the second line at 110 on red");
}

#[test]
fn rule_e_the_player_standing_at_its_red_does_not_hold_a_green_stream() {
    // the player's bus stands at its northbound red line (red until 15 s, never green with
    // the eastbound light), for a second only (not blocked); the eastbound car has its green
    // and must give way to it by the table (it comes from the right)
    let (net, ctl) = signalled_cross();
    let mut sim = Simulation::new(&net, vec![ctl]);
    let mut cars = vec![car(1, 0, 280.0 - 2.0 - 60.0, 10.0)];
    let player = PlayerView { id: 7, pos: DVec3::new(0.0, -20.0 - 6.2, 0.0), heading: 0.0, half_len: 6.0, half_width: 1.25, speed: 0.0, still: 1.0, blinker: 0, signalling: 0.0, priority: false };
    let (mut waited, mut through) = (false, None);
    run(&net, &mut sim, &mut cars, 12.0, |t, sim, cars| {
        waited |= cars[0].last.wait_for == Some(PLAYER_ID_BASE + 7);
        if through.is_none() && front_x(&cars[0]) > 20.0 {
            through = Some(t);
        }
        sim.players = vec![player];
    });
    assert!(!waited, "the green car waited for the player standing at its red");
    assert!(through.is_some_and(|t| t < 10.0), "through the box at {through:?}");
}

#[test]
fn rule_d_a_held_car_stops_at_its_line_where_its_way_converges_with_another_lane_first() {
    // lead-in piece 1 (x -20..0) and a second lead-in 4 starting 2.4 m beside it and running
    // into its end (no merge to the network: their starts lie under 3 m apart), then piece 2
    // (x 0..40) crossed at x = 20 by a main road (piece 7, priority 192) carrying a stream
    let mut net = Network::default();
    net.lanes.push(road(&[(-300.0, 0.0), (-20.0, 0.0)], 1)); // 0
    net.lanes.push(piece(&[(-20.0, 0.0), (0.0, 0.0)], 0, 128.0)); // 1
    net.lanes.push(piece(&[(0.0, 0.0), (40.0, 0.0)], 1, 128.0)); // 2
    net.lanes.push(road(&[(40.0, 0.0), (400.0, 0.0)], 2)); // 3
    net.lanes.push(piece(&[(-20.0, 2.4), (-10.0, 1.6), (0.0, 0.5)], 3, 128.0)); // 4
    net.lanes.push(road(&[(20.0, -400.0), (20.0, -15.0)], 5)); // 5
    net.lanes.push(road(&[(20.0, 15.0), (20.0, 600.0)], 6)); // 6
    net.lanes.push(piece(&[(20.0, -15.0), (20.0, 15.0)], 2, 192.0)); // 7
    let net = finish(net);
    let js = omsi_traffic::Junctions::compile(&net);
    let (_, lead_in) = js.movement(1).unwrap();
    assert!(lead_in.foes.iter().all(|f| js.nodes[0].movements[f.movement].lane != 4), "no foe between the two lead-ins: {:?}", lead_in.foes);
    assert!(!lead_in.converge.is_empty(), "the convergence is known");
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![car(1, 0, 230.0, 10.0)];
    for k in 0..8u64 {
        cars.push(car(10 + k, 5, 330.0 - 30.0 * k as f32, 12.0));
    }
    let mut stood_at: Option<f32> = None;
    run(&net, &mut sim, &mut cars, 30.0, |_, _, cars| {
        let me = cars.iter().find(|c| c.id == 1).unwrap();
        if me.state.speed < 0.1 && me.last.why == Why::Junction && stood_at.is_none() {
            stood_at = Some(front_x(me));
        }
    });
    let at = stood_at.expect("the car never stood for the main road");
    assert!(at <= -19.5, "stood with its front at x = {at:.1}, past the line into the converging lead-ins");
}
