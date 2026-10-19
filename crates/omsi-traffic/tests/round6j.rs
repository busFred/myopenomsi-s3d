//! Headless scenarios for round 6 area J (junctions): the bounded convergence stop (J2), the
//! ease pin (J3), the player's lit lane and lead (J4, J5), the box room at merge-only joins
//! (J6) and the sticky grant (J7). (J1, the far inside foe, is a resolver test in
//! `reservation.rs`.)

use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network, TrafficLightController};
#[allow(unused_imports)]
use omsi_traffic::sim::PLAYER_ID_BASE;
#[allow(unused_imports)]
use omsi_traffic::{Junctions, PlayerView, Simulation, Vehicle, VehicleClass, Why};

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
fn piece(pts: &[(f64, f64)], path: u16, priority: f32, width: f32) -> omsi_sim::traffic::Lane {
    let mut l = lane(pts, LaneKind::Street, width);
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

fn front_x(v: &Vehicle) -> f32 {
    v.pos.x as f32 + v.state.front
}

/// J2: an eastbound way through object 9 (piece 1, x 0..60) crossed at x = 50 by a main road
/// (piece 5, priority 192) carrying a stream; a piece of the object (3) runs beside it 2.4 m
/// off over x `beside.0..beside.1`, no meeting place (a convergence, `Movement::converge`).
/// Lanes: 0 the approach, 1 the piece, 2 the exit, 3 the piece beside, 4/5/6 the main road.
fn converge_far(beside: (f64, f64)) -> Network {
    let mut net = Network::default();
    net.lanes.push(road(&[(-300.0, 0.0), (0.0, 0.0)], 1)); // 0
    net.lanes.push(piece(&[(0.0, 0.0), (60.0, 0.0)], 0, 128.0, 2.3)); // 1
    net.lanes.push(road(&[(60.0, 0.0), (400.0, 0.0)], 2)); // 2
    net.lanes.push(piece(&[(beside.0, 2.4), (beside.1, 2.4)], 1, 128.0, 2.3)); // 3
    net.lanes.push(road(&[(50.0, -400.0), (50.0, -15.0)], 5)); // 4
    net.lanes.push(piece(&[(50.0, -15.0), (50.0, 15.0)], 2, 192.0, 3.0)); // 5
    net.lanes.push(road(&[(50.0, 15.0), (50.0, 600.0)], 6)); // 6
    finish(net)
}

/// Where the eastbound car (front x) first stands held by the junction, and the least gap.
fn converge_far_stand(beside: (f64, f64)) -> (f32, f32) {
    let net = converge_far(beside);
    let js = Junctions::compile(&net);
    let (_, way) = js.movement(1).unwrap();
    assert!(!way.converge.is_empty(), "no convergence compiled: {:?}", way.foes);
    let mut sim = Simulation::new(&net, vec![]);
    // (standing 3 m short of the line, the stream coming: refused from the start)
    let mut cars = vec![car(1, 0, 295.0, 0.0)];
    for k in 0..12u64 {
        cars.push(car(10 + k, 4, 370.0 - 22.0 * k as f32, 12.0));
    }
    let mut stood_at: Option<f32> = None;
    let min_gap = run(&net, &mut sim, &mut cars, 25.0, |_, _, cars| {
        let me = cars.iter().find(|c| c.id == 1).unwrap();
        if me.state.speed < 0.1 && me.last.why == Why::Junction && stood_at.is_none() {
            stood_at = Some(front_x(me));
        }
    });
    (stood_at.expect("the car never stood for the main road"), min_gap)
}

#[test]
fn j2_a_refused_car_rolls_up_past_a_convergence_far_ahead_and_stops_at_once_for_a_near_one() {
    // the convergence 28-42 m on (its band from x 27): the car rolls up, its body wholly out of
    // the convergence (before it or past it) and short of the main road at 50 less its half
    // width (review-r6j: no body inside the band)
    let (at, gap) = converge_far_stand((28.0, 42.0));
    assert!((at > 20.0 && at <= 27.0) || (at - 4.0 >= 42.0 && at < 48.6), "stood with its front at x = {at:.1}");
    assert!(gap > 0.3, "touched: {gap:.2} m");
    // the convergence 5-20 m on, within CONVERGE_LOOK of the line: at once, at the line, as
    // before (review9 5229/5231, 10814/10625: near bands unchanged)
    let (at, _) = converge_far_stand((5.0, 20.0));
    assert!(at <= 0.0 && at > -1.5, "stood with its front at x = {at:.1}, not at the line");
}

/// J2: my way (piece 1, x 0..70, priority 64) is crossed at x ~ 43 by piece 5 (priority 192),
/// which first runs beside it 2.8 m to the left from x 0 to 40; a converging piece of the
/// object (3, no foe) runs 2.4 m to the right over x 26..36. A queue on piece 5 crawls at
/// `stream_v` from `stream_from` on: where my car, refused for it, first stands (front x).
fn waited_beside(stream_from: f32, stream_v: f32) -> f32 {
    let mut net = Network::default();
    net.lanes.push(road(&[(-300.0, 0.0), (0.0, 0.0)], 1)); // 0
    net.lanes.push(piece(&[(0.0, 0.0), (70.0, 0.0)], 0, 64.0, 2.3)); // 1
    net.lanes.push(road(&[(70.0, 0.0), (400.0, 0.0)], 2)); // 2
    net.lanes.push(piece(&[(26.0, -2.4), (36.0, -2.4)], 1, 128.0, 2.3)); // 3
    net.lanes.push(road(&[(-300.0, 2.8), (0.0, 2.8)], 3)); // 4
    net.lanes.push(piece(&[(0.0, 2.8), (40.0, 2.8), (55.0, -15.0)], 2, 192.0, 2.3)); // 5
    net.lanes.push(road(&[(55.0, -15.0), (55.0, -300.0)], 4)); // 6
    let net = finish(net);
    let js = Junctions::compile(&net);
    assert!(!js.movement(1).unwrap().1.converge.is_empty(), "no convergence compiled");
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![car(1, 0, 295.0, 0.0)];
    for k in 0..4u64 {
        let mut s = car(10 + k, 5, stream_from - 9.0 * k as f32, stream_v);
        s.state.max_speed_kmh = stream_v * 3.6;
        cars.push(s);
    }
    let mut stood = f32::NAN;
    run(&net, &mut sim, &mut cars, 12.0, |_, _, cars| {
        let me = cars.iter().find(|c| c.id == 1).unwrap();
        if me.state.speed < 0.1 && me.last.why == Why::Junction && stood.is_nan() {
            stood = front_x(me);
        }
    });
    stood
}

#[test]
fn j2_the_waited_vehicle_standing_in_a_far_convergence_stops_the_car_at_its_line() {
    // the car waited for crawls at 0.3 m/s with its body in the stretch beside mine where the
    // convergence lies (x 26..36, 26 m past the line): the car stops at once, at its line, not
    // rolling up beside it (review9: refused cars rolled into the very car they waited for)
    let at = waited_beside(36.0, 0.3);
    assert!(at <= 0.0 && at > -1.5, "stood with its front at x = {at:.1}, not at the line");
    // the queue rolling at 2 m/s further on: the car rolls up, its body out of the convergence
    let at = waited_beside(30.0, 2.0);
    assert!(at > 5.0 && at <= 25.0, "stood with its front at x = {at:.1}");
}

/// J3: two lanes of object 9 merging at x = 0 (piece 1 from the south-west, piece 4 from the
/// north-west, 20 m each, then piece 2 to x = 30). Lanes: 0 and 3 the approaches, 1 and 4 the
/// merging pieces, 2 the merged way, 5 the exit.
fn merge_pair() -> Network {
    let mut net = Network::default();
    net.lanes.push(road(&[(-300.0, -6.0), (-20.0, -6.0)], 1)); // 0
    net.lanes.push(piece(&[(-20.0, -6.0), (0.0, 0.0)], 0, 128.0, 3.0)); // 1
    net.lanes.push(piece(&[(0.0, 0.0), (30.0, 0.0)], 2, 128.0, 3.0)); // 2
    net.lanes.push(road(&[(-300.0, 6.0), (-20.0, 6.0)], 3)); // 3
    net.lanes.push(piece(&[(-20.0, 6.0), (0.0, 0.0)], 1, 128.0, 3.0)); // 4
    net.lanes.push(road(&[(30.0, 0.0), (400.0, 0.0)], 4)); // 5
    finish(net)
}

/// Car 1 on piece 1, eased off for car 2 on piece 4 (2 m nearer the merge, at `speed`): (how
/// far car 1 moved in `secs`, when it first moved 0.3 m, the least gap between the bodies).
fn eased_for_partner(speed: f32, secs: f32) -> (f32, Option<f32>, f32) {
    let net = merge_pair();
    let mut sim = Simulation::new(&net, vec![]);
    let mut a = car(1, 1, 8.0, 0.0);
    a.ease_for = Some((2, 0, 0.0));
    let mut b = car(2, 4, 10.0, speed);
    if speed == 0.0 {
        // (standing for good: something of its own holds it, a parked lead ahead on its piece)
        b.state.max_speed_kmh = 0.0;
    }
    let mut cars = vec![a, b];
    let start = cars[0].state.odometer;
    let mut moved_at = None;
    let min_gap = run(&net, &mut sim, &mut cars, secs, |t, _, cars| {
        let a = cars.iter().find(|c| c.id == 1).unwrap();
        if moved_at.is_none() && a.state.odometer - start > 0.3 {
            moved_at = Some(t);
        }
    });
    let a = cars.iter().find(|c| c.id == 1).unwrap();
    (a.state.odometer - start, moved_at, min_gap)
}

#[test]
fn j3_an_eased_car_follows_its_rolling_partner_in_and_stays_pinned_for_a_standing_one() {
    // the partner rolls on into the merge at 3 m/s: the eased car rolls in behind it within 2 s
    // (it was pinned at front + 0.1 until the partner had left the whole object, review-r6
    // case03 #5424 13 s), keeping off its body
    let (moved, at, gap) = eased_for_partner(3.0, 8.0);
    assert!(at.is_some_and(|t| t < 2.0), "moved off at {at:?} s");
    assert!(moved > 5.0, "moved {moved:.1} m");
    assert!(gap > 0.3, "touched: {gap:.2} m");
    // a standing partner (x10 case03 #587/#3221): no follow, it stays where it stands
    let (moved, _, _) = eased_for_partner(0.0, 4.0);
    assert!(moved < 0.3, "moved {moved:.2} m off its pin with the partner standing");
}

#[test]
fn j3_a_marked_car_standing_at_a_red_line_waits_for_its_green() {
    // one lit piece after the approach, red for 60 s; the car stands 0.5 m before the line with
    // the light's mark (it took it rolling up behind a partner it followed in, then braked to a
    // stand: viz case07 #18558 set off over the line on red, review-r6j)
    let mut net = Network::default();
    net.lanes.push(road(&[(-300.0, 0.0), (-10.0, 0.0)], 1));
    let mut lit = piece(&[(-10.0, 0.0), (10.0, 0.0)], 0, 128.0, 3.0);
    lit.traffic_light = Some((0, 0));
    net.lanes.push(lit);
    net.lanes.push(road(&[(10.0, 0.0), (300.0, 0.0)], 2));
    let net = finish(net);
    let mut sim = Simulation::new(&net, vec![TrafficLightController::new(vec![vec![(0, 60.0), (6, 6.0)]], 66.0)]);
    // (just come to a stand, 0.1 m/s; the mark given after the first frame, once the light is
    // seen, as a car that took it rolling has it)
    let mut cars = vec![car(1, 0, 290.0 - 0.5 - 2.0, 0.1)];
    run(&net, &mut sim, &mut cars, 10.0, |t, _, cars| {
        if t < DT * 0.5 {
            cars[0].amber = Some((0, 0));
            cars[0].amber_line = Some(cars[0].state.odometer + 0.5);
        }
    });
    let front = front_x(&cars[0]);
    assert!(front <= -10.0 + 0.3, "set off over its red line from a stand with the mark: front at x = {front:.1}");
}

/// J4: an eastbound way (approach 0, lit piece 1 x -20..20 under light (0, 0), exit 2) crossed
/// by a northbound one (approach 3, lit piece 4 y -20..20 under (0, 1), exit 5), equal
/// priorities (the northbound way comes from the eastbound's right). Eastbound green 0-10 s,
/// yellow 10-13, red 13-40; northbound red 0-15, green 15-37, yellow 37-40.
fn signalled_cross() -> (Network, TrafficLightController) {
    let mut net = Network::default();
    net.lanes.push(road(&[(-300.0, 0.0), (-20.0, 0.0)], 1));
    let mut a = piece(&[(-20.0, 0.0), (20.0, 0.0)], 0, 128.0, 3.5);
    a.traffic_light = Some((0, 0));
    net.lanes.push(a);
    net.lanes.push(road(&[(20.0, 0.0), (300.0, 0.0)], 2));
    net.lanes.push(road(&[(0.0, -300.0), (0.0, -20.0)], 3));
    let mut b = piece(&[(0.0, -20.0), (0.0, 20.0)], 1, 128.0, 3.5);
    b.traffic_light = Some((0, 1));
    net.lanes.push(b);
    net.lanes.push(road(&[(0.0, 20.0), (0.0, 300.0)], 4));
    let ctl = TrafficLightController::new(vec![vec![(6, 10.0), (9, 3.0), (0, 27.0)], vec![(0, 15.0), (6, 22.0), (9, 3.0)]], 40.0);
    (finish(net), ctl)
}

/// The player's bus standing at its northbound red with its front `past` m over the lit
/// piece's start, still `still` s; the eastbound car on its green: (whether the car waited for
/// the player, when it was through the box, the player's claim's light_red).
fn player_over_its_line(past: f64, still: f32) -> (bool, Option<f32>, bool) {
    let (net, ctl) = signalled_cross();
    let mut sim = Simulation::new(&net, vec![ctl]);
    let mut cars = vec![car(1, 0, 280.0 - 2.0 - 60.0, 10.0)];
    let player = PlayerView { id: 7, pos: DVec3::new(0.0, -20.0 + past - 6.0, 0.0), heading: 0.0, half_len: 6.0, half_width: 1.25, speed: 0.0, still, blinker: 0, signalling: 0.0, priority: false };
    let (mut waited, mut through, mut red) = (false, None, false);
    sim.players = vec![player];
    run(&net, &mut sim, &mut cars, 12.0, |t, sim, cars| {
        waited |= cars[0].last.wait_for == Some(PLAYER_ID_BASE + 7);
        if through.is_none() && front_x(&cars[0]) > 20.0 {
            through = Some(t);
        }
        red |= sim.claims.iter().any(|(c, _)| c.vehicle >= usize::MAX - 64 && c.light_red);
        sim.players = vec![player];
    });
    (waited, through, red)
}

#[test]
fn j4_the_player_keeps_its_red_until_its_rear_is_past_its_line() {
    // its front 0.8 m past the red stub, standing 3 s (blocked): still held by its red, the
    // green car does not wait for it (case07: the bus lost its light and held #37076 21 s)
    let (waited, through, red) = player_over_its_line(0.8, 3.0);
    assert!(red, "the player's claim has no red light");
    assert!(!waited, "the green car waited for the player at its red");
    assert!(through.is_some_and(|t| t < 10.0), "through the box at {through:?}");
    // at its line (round 4's rule E, unchanged)
    let (waited, _, red) = player_over_its_line(-0.2, 3.0);
    assert!(red && !waited);
    // its rear past the line (it drove through on red; put down there): no light, the
    // ordinary rules (blocked, standing, from rest when it moves)
    let (_, _, red) = player_over_its_line(12.6, 3.0);
    assert!(!red, "a player whose rear is past its line kept its red");
}

/// J5: the player's northbound way (approach 0 to y = -20, piece 1 to the merge at (0, 40),
/// priority 192) and a piece of the AI's (2, from (3.5, -20), 3.5 m to the right to y = 0, then
/// bending in to the merge, priority 64: it gives way; they meet from y = 10), then the way on
/// 3 and 4 (outside the object: the two ways share no piece, as case01's); the AI's approach 5.
fn diagonal_merge() -> Network {
    let mut net = Network::default();
    net.lanes.push(road(&[(0.0, -300.0), (0.0, -20.0)], 1)); // 0
    net.lanes.push(piece(&[(0.0, -20.0), (0.0, 40.0)], 0, 192.0, 2.0)); // 1
    net.lanes.push(piece(&[(3.5, -20.0), (3.5, 0.0), (0.0, 40.0)], 1, 64.0, 2.0)); // 2
    net.lanes.push(road(&[(0.0, 40.0), (0.0, 60.0)], 3)); // 3
    net.lanes.push(road(&[(0.0, 60.0), (0.0, 400.0)], 4)); // 4
    net.lanes.push(road(&[(3.5, -300.0), (3.5, -20.0)], 5)); // 5
    finish(net)
}

/// The AI car standing on piece 2 with its rear at y = 4 (its centre 2.2 m right of the
/// player's axis, x = 0.9); the player's bus (12 m) standing `still` s (or rolling at `speed`)
/// with its front `gap` m behind that rear: whether the car
/// waited for the player in `secs`, and the player's claim's lead at the first frame.
fn behind_the_tail(gap: f64, still: f32, speed: f32, secs: f32) -> (bool, Option<usize>) {
    let net = diagonal_merge();
    let mut sim = Simulation::new(&net, vec![]);
    // (a slow one, 10 km/h: out of the merge only well after the player could be in it)
    let mut me = car(1, 2, 24.0 + 2.0, 0.0);
    me.state.max_speed_kmh = 10.0;
    let mut cars = vec![me];
    let player = PlayerView { id: 7, pos: DVec3::new(0.9, 4.0 - gap - 6.0, 0.0), heading: 0.0, half_len: 6.0, half_width: 1.25, speed, still, blinker: 0, signalling: 0.0, priority: false };
    sim.players = vec![player];
    let (mut waited, mut lead, mut first) = (false, None, true);
    run(&net, &mut sim, &mut cars, secs, |_, sim, cars| {
        waited |= cars[0].last.wait_for == Some(PLAYER_ID_BASE + 7);
        if first {
            lead = sim.claims.iter().find(|(c, _)| c.vehicle >= usize::MAX - 64).and_then(|(c, _)| c.lead);
            first = false;
        }
        sim.players = vec![player];
    });
    (waited, lead)
}

#[test]
fn j5_a_standing_player_behind_a_car_s_tail_names_it_as_its_lead() {
    // the player stands 1.5 m behind the tail of the car beside-ahead of it (its centre 2 m
    // across): the car, which gives way to the player's way at their merge, does not wait for
    // it (case01: coach #4252 waited for PLAYER0 standing behind its tail)
    let (waited, lead) = behind_the_tail(1.5, 5.0, 0.0, 2.0);
    assert!(!waited, "the car waited for the player standing behind it");
    assert_eq!(lead, Some(0), "the player's claim names no lead");
    // standing for a second only (not blocked yet): no lead
    let (_, lead) = behind_the_tail(1.5, 1.0, 0.0, 0.5);
    assert_eq!(lead, None);
    // rolling (viz3 case11: a rolling player still meets the foes it meets): no lead
    let (_, lead) = behind_the_tail(1.5, 0.0, 4.0, 0.2);
    assert_eq!(lead, None);
    // 4 m behind the tail: out of reach, no lead
    let (_, lead) = behind_the_tail(4.0, 5.0, 0.0, 0.5);
    assert_eq!(lead, None);
}
