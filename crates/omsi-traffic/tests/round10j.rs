//! Headless checks for round 10 area J (foe eligibility): the crate predicates other areas read.

use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network, TrafficLightController};
use omsi_traffic::reservation::{Claim, Grant};
use omsi_traffic::{Impatience, Simulation, Vehicle, VehicleClass, Why};

const DT: f32 = 1.0 / 30.0;

fn road(pts: &[(f64, f64)], id: i64) -> omsi_sim::traffic::Lane {
    let mut l = LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), LaneKind::Street, 3.5);
    l.source = 1;
    l.key = Some(LaneKey { tile: (0, 0), id, path: 0 });
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

/// A lane of crossing object 9 (source 2).
fn piece(pts: &[(f64, f64)], path: u16, kind: LaneKind) -> omsi_sim::traffic::Lane {
    let mut l = LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), kind, 3.5);
    l.source = 2;
    l.key = Some(LaneKey { tile: (0, 0), id: 9, path });
    l.priority = 128.0;
    l
}

/// A crossing object: eastbound piece 0 (x -15..45), northbound piece 1 (x = 0), and a zebra
/// (footpath 2) across piece 0 at x = 23.
fn cross_with_a_zebra() -> Network {
    let mut net = Network::default();
    net.lanes.push(piece(&[(-15.0, 0.0), (45.0, 0.0)], 0, LaneKind::Street));
    net.lanes.push(piece(&[(0.0, -15.0), (0.0, 15.0)], 1, LaneKind::Street));
    net.lanes.push(piece(&[(23.0, -8.0), (23.0, 8.0)], 2, LaneKind::Sidewalk));
    net.link(1.5);
    net.build_grid();
    net.compute_conflicts();
    net
}

fn claim_on(sim: &Simulation, lane: usize, s: f32, in_box_at_turn: bool) -> Claim {
    let (j, m) = sim.junctions.of_lane[&lane];
    Claim { chain: vec![(m, 0.0)], vehicle: 0, junction: j, movement: m, s, speed: 0.0, accel: 1.8, v_max: 13.0, front: 2.0, rear: 2.0, decel: 2.0, waiting: 0.0, lead_gap: None, zip_behind: None, refused_last: false, body_block: None, next_wait: f32::NEG_INFINITY, tau: 1.3, min_gap: 2.0, committed: false, exit_room: true, impatience: Impatience::default(), absolute: false, held: false, light_red: false, until_go: None, red_at: f32::INFINITY, lead: None, was_standing: true, blocked: false, stuck: false, clearing: false, in_box_at_turn, lanes: Vec::new(), light: Some((0, 0)), id: 1, yields_now: None, clear_ahead: f32::INFINITY, serving: false, brake: 5.6 }
}

/// [round 10 D7] `Simulation::clearing_across`: a vehicle in the box at the end of its green whose
/// way still crosses the zebra ahead of its rear holds the zebra's walkers; once its rear is past
/// the zebra, or when it is not in the box at the end of its green, it does not.
#[test]
fn d7_walkers_see_a_vehicle_clearing_the_box_across_their_crossing() {
    let net = cross_with_a_zebra();
    let mut sim = Simulation::new(&net, vec![]);
    assert!(sim.junctions.of_lane.contains_key(&0), "piece 0 is a junction piece");
    let zebra_at = net.walks[0].iter().find(|w| w.0 == 2).expect("the zebra cuts piece 0").1;
    for (s, in_box, expect) in [(5.0f32, true, true), (zebra_at + 1.0, true, true), (zebra_at + 2.5, true, false), (5.0, false, false)] {
        sim.claims = vec![(claim_on(&sim, 0, s, in_box), Grant::go(0))];
        assert_eq!(sim.clearing_across(&net, 2), expect, "s {s} in the box at the turn {in_box}");
    }
    // (stuck - held by people or a standing lead - it does not clear: walkers are not held for it)
    let mut c = claim_on(&sim, 0, 5.0, true);
    c.stuck = true;
    sim.claims = vec![(c, Grant::go(0))];
    assert!(!sim.clearing_across(&net, 2), "a stuck car in the box holds the walkers");
}

/// A signalled crossing (review9's): northbound 0 -> lit stub 1 -> 2 -> 3; the southbound left
/// turn 4 -> lit stub 5 -> 6 (turns east across the northbound way) -> 7; the westbound cross street
/// 8 -> lit stub 9 -> 10 -> 11 at y = 17 crosses the turn 8 m into it. North and south share a green
/// (lights 0 and 1), the cross street has its own (light 2).
fn left_turn_cross() -> Network {
    let mut net = Network::default();
    net.lanes.push(road(&[(2.0, -400.0), (2.0, -20.0)], 1));
    let mut l1 = piece(&[(2.0, -20.0), (2.0, -19.0)], 0, LaneKind::Street);
    l1.traffic_light = Some((0, 0));
    net.lanes.push(l1);
    net.lanes.push(piece(&[(2.0, -19.0), (2.0, 25.0)], 1, LaneKind::Street));
    net.lanes.push(road(&[(2.0, 25.0), (2.0, 400.0)], 2));
    net.lanes.push(road(&[(-2.0, 300.0), (-2.0, 26.0)], 3));
    let mut l5 = piece(&[(-2.0, 26.0), (-2.0, 25.0)], 2, LaneKind::Street);
    l5.traffic_light = Some((0, 1));
    net.lanes.push(l5);
    let mut turn = piece(&[(-2.0, 25.0), (-2.0, 5.0), (5.0, -2.0), (25.0, -2.0)], 3, LaneKind::Street);
    turn.turn = 1;
    net.lanes.push(turn);
    net.lanes.push(road(&[(25.0, -2.0), (300.0, -2.0)], 4));
    net.lanes.push(road(&[(300.0, 17.0), (26.0, 17.0)], 5));
    let mut l9 = piece(&[(26.0, 17.0), (25.0, 17.0)], 4, LaneKind::Street);
    l9.traffic_light = Some((0, 2));
    net.lanes.push(l9);
    net.lanes.push(piece(&[(25.0, 17.0), (-25.0, 17.0)], 5, LaneKind::Street));
    net.lanes.push(road(&[(-25.0, 17.0), (-300.0, 17.0)], 6));
    net.link(1.5);
    net.compute_conflicts();
    net
}

/// [round 10 J9-2] A left-turner facing an oncoming stream on its green enters the box and waits
/// at its waiting place inside (past the cross street, short of the oncoming way; SUMO's internal
/// junction), not at its line: the keep-clear rule holds a car at the line only when its rest
/// inside is no waiting place. When the green ends it turns, before the cross street's green.
/// Nobody touches. (Before: held at the line 29 s of the 30 s green.)
#[test]
fn j92_a_left_turner_waits_inside_on_green_not_at_its_line() {
    let net = left_turn_cross();
    // north/south green 0..30, yellow 3 s, red; the cross street green from 36 s
    let ctl = TrafficLightController::new(vec![vec![(6, 30.0), (9, 3.0), (0, 27.0)], vec![(6, 30.0), (9, 3.0), (0, 27.0)], vec![(0, 36.0), (6, 24.0)]], 60.0);
    let mut sim = Simulation::new(&net, vec![ctl]);
    // the turner 40 m before its line at 10 m/s; an oncoming stream every 25 m at 12 m/s (2 s gaps)
    let mut cars = vec![car(1, 4, 274.0 - 2.0 - 40.0, 10.0)];
    for k in 0..16u64 {
        cars.push(car(10 + k, 0, 380.0 - 2.0 - 25.0 * k as f32, 12.0));
    }
    // two cross-street cars standing at their red
    cars.push(car(40, 8, 274.0 - 2.0 - 1.0, 0.0));
    cars.push(car(41, 8, 274.0 - 2.0 - 8.0, 0.0));
    let (mut waited_inside, mut kept_clear_on_green, mut through_at, mut cross_in_at) = (false, 0.0f32, None, None);
    let min_gap = run(&net, &mut sim, &mut cars, 50.0, |t, _, cars| {
        if let Some(tr) = cars.iter().find(|c| c.id == 1) {
            if t < 30.0 && tr.state.lane == 6 && tr.state.s > 9.0 && tr.state.speed < 0.3 && tr.last.why == Why::Junction {
                waited_inside = true;
            }
            if t < 30.0 && tr.last.why == Why::KeepClear {
                kept_clear_on_green += DT;
            }
            if through_at.is_none() && tr.state.lane == 7 {
                through_at = Some(t);
            }
        }
        if cross_in_at.is_none() && cars.iter().any(|c| c.id == 40 && c.state.lane == 10 && c.state.s > 15.0) {
            cross_in_at = Some(t);
        }
    });
    assert!(min_gap > 0.3, "touched: {min_gap:.2} m");
    assert!(waited_inside, "the turner waited at its line, kept clear {kept_clear_on_green:.1} s");
    assert!(kept_clear_on_green < 1.0, "held at the line to keep the box clear on green for {kept_clear_on_green:.1} s");
    let (Some(a), Some(b)) = (through_at, cross_in_at) else { panic!("turner through {through_at:?}, cross street in {cross_in_at:?}") };
    assert!(a < b, "the cross street went first: turner through at {a:.1} s, cross street over the turn at {b:.1} s");
}

/// [round 10 Q9-1] The player's time standing held by nothing starts, after its red, from minus
/// the start reaction of its preset (0.8 s), as an AI head let go starts within its reaction: a
/// bus that stands 4 s after its green is not yet standing for good (`QUEUE_OBSTACLE_STAND`).
#[test]
fn q91_the_players_standing_clock_after_its_red_starts_after_the_start_reaction() {
    let lane = |y: f64, x0: f64, x1: f64| {
        let mut l = LaneBuilder::polyline(vec![DVec3::new(x0, y, 0.0), DVec3::new(x1, y, 0.0)], LaneKind::Street, 3.5);
        l.source = 1;
        l
    };
    let mut net = Network::default();
    net.lanes.push(lane(-3.5, 0.0, 200.0));
    net.lanes.push(lane(-3.5, 200.0, 400.0));
    net.lanes[1].traffic_light = Some((0, 0));
    net.link(1.5);
    net.build_grid();
    net.compute_conflicts();
    // (red 10 s, then green)
    let ctl = TrafficLightController::new(vec![vec![(0, 10.0), (6, 100.0)]], 110.0);
    let mut sim = Simulation::new(&net, vec![ctl]);
    sim.players.push(omsi_traffic::PlayerView { id: 0, pos: DVec3::new(193.0, -3.5, 0.0), heading: 90.0, half_len: 6.0, half_width: 1.25, speed: 0.0, still: 60.0, blinker: 0, signalling: 0.0, priority: false });
    let mut cars: Vec<Vehicle> = Vec::new();
    let mut at = |secs: f32, sim: &mut Simulation| {
        run(&net, sim, &mut cars, secs, |_, _, _| {});
        sim.player_wait.get(&0).copied().unwrap_or(0.0)
    };
    let w_red = at(9.0, &mut sim);
    assert!(w_red < 0.0, "held by its red: {w_red}");
    // (green from 10 s: 4 s later the clock has run 4 s less the reaction)
    let w4 = at(5.0, &mut sim);
    assert!(w4 < omsi_traffic::plan::QUEUE_OBSTACLE_STAND && w4 > 2.0, "4 s after the green: {w4}");
    let w6 = at(2.0, &mut sim);
    assert!(w6 >= omsi_traffic::plan::QUEUE_OBSTACLE_STAND, "6 s after the green: {w6}");
}

/// [round 10 H8-5] A lit way through that crosses nothing (a merge-only join: no crossing meeting
/// place, walk or convergence) has nothing to keep clear - SUMO applies no keep-clear at such
/// geometry-like nodes - so a car on green follows the queue standing in it as in a lane; with a
/// walk across the join's exit piece, it is held at its line as before (here by the exit's room).
#[test]
fn h85_a_merge_only_join_is_no_box_to_keep_clear() {
    for walk in [false, true] {
        let mut net = Network::default();
        net.lanes.push(road(&[(-300.0, 0.0), (-21.0, 0.0)], 1)); // 0 approach
        let mut stub = piece(&[(-21.0, 0.0), (-20.0, 0.0)], 0, LaneKind::Street); // 1 lit stub
        stub.traffic_light = Some((0, 0));
        net.lanes.push(stub);
        net.lanes.push(piece(&[(-20.0, 0.0), (0.0, 0.0)], 1, LaneKind::Street)); // 2
        net.lanes.push(piece(&[(-14.0, -8.0), (0.0, 0.0)], 2, LaneKind::Street)); // 3 merging in
        net.lanes.push(piece(&[(0.0, 0.0), (30.0, 0.0)], 3, LaneKind::Street)); // 4 exit piece
        net.lanes.push(road(&[(30.0, 0.0), (300.0, 0.0)], 2)); // 5
        if walk {
            net.lanes.push(piece(&[(20.0, -8.0), (20.0, 8.0)], 4, LaneKind::Sidewalk)); // 6
        }
        net.link(1.5);
        net.build_grid();
        net.compute_conflicts();
        let ctl = TrafficLightController::new(vec![vec![(6, 1000.0)]], 1000.0);
        let mut sim = Simulation::new(&net, vec![ctl]);
        // (a queue standing on the exit piece: its tail at x 10; the car 40 m before the line)
        let mut q = car(1, 4, 12.0, 0.0);
        q.external.stop_at = Some(q.state.front + 0.1);
        let mut cars = vec![q, car(2, 0, 279.0 - 40.0, 8.0)];
        let mut kept = 0.0f32;
        run(&net, &mut sim, &mut cars, 20.0, |_, _, cars| {
            if cars.iter().any(|c| c.id == 2 && c.last.why == Why::KeepClear) {
                kept += DT;
            }
        });
        let c = cars.iter().find(|c| c.id == 2).unwrap();
        if walk {
            assert!(c.state.lane <= 1 && matches!(c.last.why, Why::KeepClear | Why::ExitFull), "a walk across the exit: held at the line (lane {} s {:.1} why {:?})", c.state.lane, c.state.s, c.last.why);
        } else {
            assert!(kept == 0.0 && c.state.lane >= 2, "merge only: kept clear {kept:.1} s, lane {} s {:.1}", c.state.lane, c.state.s);
        }
    }
}

/// Three ways through one point (object 9): X eastbound at y = 5 (low priority), Y northbound at
/// x = 0 (middle), Z eastbound at y = -5 (high): Y crosses both, X and Z nothing of each other.
fn three_ways() -> Network {
    let mut net = Network::default();
    let mut x = piece(&[(-10.0, 5.0), (10.0, 5.0)], 0, LaneKind::Street);
    x.priority = 64.0;
    let mut z = piece(&[(-10.0, -5.0), (10.0, -5.0)], 2, LaneKind::Street);
    z.priority = 192.0;
    net.lanes.push(x);
    net.lanes.push(piece(&[(0.0, -10.0), (0.0, 10.0)], 1, LaneKind::Street));
    net.lanes.push(z);
    net.link(1.5);
    net.compute_conflicts();
    net
}

/// [round 10 pass 1, 8464ed00] A car in the box at the end of its green that stands refused for a
/// third vehicle is not committed for the fresh green across its way: Y (in the box, light 1)
/// stands refused for Z rolling on its way; X on the fresh green (light 0, never green with 1),
/// which the table makes give way to Y, goes. With Y not refused (no Z), X waits for it.
#[test]
fn pass1_a_car_in_the_box_refused_for_a_third_does_not_hold_the_fresh_green() {
    use omsi_traffic::reservation::{resolve_with, Context};
    let net = three_ways();
    let sim = Simulation::new(&net, vec![]);
    let js = &sim.junctions;
    let start = |a: usize, b: usize| {
        let (_, ma) = js.of_lane[&a];
        let (_, mb) = js.of_lane[&b];
        js.nodes[0].movements[ma].foes.iter().find(|f| f.movement == mb).unwrap().mine.0
    };
    let mut ctx = Context::default();
    ctx.exclusive.insert((0, 0, 1), true);
    let mut x = claim_on(&sim, 0, start(0, 1) - 1.0 - 2.0, false);
    x.vehicle = 1;
    x.id = 1;
    x.light = Some((0, 0));
    let mut y = claim_on(&sim, 1, start(1, 2) - 1.0 - 2.0, true);
    y.vehicle = 2;
    y.id = 2;
    y.light = Some((0, 1));
    y.refused_last = true;
    let mut z = claim_on(&sim, 2, start(2, 1) - 8.0 - 2.0, false);
    z.vehicle = 3;
    z.id = 3;
    z.speed = 8.0;
    z.was_standing = false;
    z.light = None;
    let grant = |g: &[Grant], v: usize| *g.iter().find(|g| g.vehicle == v).unwrap();
    let g = resolve_with(js, &[x.clone(), y.clone(), z], &ctx);
    assert_eq!(grant(&g, 2).wait_for, Some(3), "Y waits for Z: {g:?}");
    assert!(grant(&g, 1).go, "X waits for Y, which stands refused in the box: {g:?}");
    let g = resolve_with(js, &[x, y], &ctx);
    assert_eq!(grant(&g, 1).wait_for, Some(2), "Y clears the box first: {g:?}");
}

/// [round 10 pass 1, dbc48899] The ease of a mutual refusal starts only once the resolver has seen
/// both claims standing: two vehicles standing after `act`, refused for each other, one of whose
/// claims was filed while it still crawled, are not eased that frame; filed standing, they are.
#[test]
fn pass1_a_mutual_refusal_eases_only_once_both_claims_stand() {
    let net = cross_with_a_zebra();
    for crawl in [true, false] {
        let mut sim = Simulation::new(&net, vec![]);
        let mut a = car(1, 0, 3.0, 0.0);
        let mut b = car(2, 1, 3.0, 0.0);
        a.last.wait_for = Some(2);
        b.last.wait_for = Some(1);
        let mut ca = claim_on(&sim, 0, 3.0, false);
        ca.vehicle = 0;
        ca.id = 1;
        ca.light = None;
        ca.was_standing = false;
        ca.speed = if crawl { 0.5 } else { 0.0 };
        let mut cb = claim_on(&sim, 1, 3.0, false);
        cb.vehicle = 1;
        cb.id = 2;
        cb.light = None;
        sim.claims = vec![(ca, Grant::refused(0, Some(1), omsi_traffic::Hold::Foe)), (cb, Grant::refused(1, Some(0), omsi_traffic::Hold::Foe))];
        let mut vs = vec![a, b];
        sim.negotiate(&mut vs, &[Some(0), Some(1)], DT);
        let eased = vs.iter().any(|v| v.ease_for.is_some());
        assert_eq!(eased, !crawl, "claim filed crawling {crawl}: eased {eased}");
    }
}

/// [round 10 pass 1, J9-2 for people] A car on green whose rest inside the box is the zebra stop of
/// a walk with somebody on it, at a waiting place (its body on no band), enters and waits there,
/// as SUMO's vehicle waits before the crossing on its internal lane; it is not held at its line.
/// (A guard of the behaviour: in this geometry the rule before pass 1 let it in after 1.0 s too.)
#[test]
fn pass1_a_car_waiting_for_people_at_a_walk_inside_the_box_enters_on_green() {
    let mut net = Network::default();
    net.lanes.push(road(&[(-300.0, 0.0), (-21.0, 0.0)], 1)); // 0 approach
    let mut stub = piece(&[(-21.0, 0.0), (-20.0, 0.0)], 0, LaneKind::Street); // 1 lit stub
    stub.traffic_light = Some((0, 0));
    net.lanes.push(stub);
    net.lanes.push(piece(&[(-20.0, 0.0), (30.0, 0.0)], 1, LaneKind::Street)); // 2
    net.lanes.push(road(&[(30.0, 0.0), (300.0, 0.0)], 2)); // 3
    net.lanes.push(piece(&[(-12.0, -15.0), (-12.0, 15.0)], 2, LaneKind::Street)); // 4 cross street
    net.lanes.push(piece(&[(6.0, -8.0), (6.0, 8.0)], 3, LaneKind::Sidewalk)); // 5 walk at x 6
    net.link(1.5);
    net.build_grid();
    net.compute_conflicts();
    let ctl = TrafficLightController::new(vec![vec![(6, 900.0), (0, 100.0)]], 1000.0);
    let mut sim = Simulation::new(&net, vec![ctl]);
    sim.people.push(omsi_traffic::Person { pos: DVec3::new(6.0, 0.3, 0.0), vel: glam::DVec2::ZERO, on_crossing: true, still: 60.0, waiting_for_bus: false, at_kerb: false, waits_at: None, id: None, approach: None });
    let mut cars = vec![car(1, 0, 279.0 - 40.0, 8.0)];
    let mut kept = 0.0f32;
    run(&net, &mut sim, &mut cars, 20.0, |_, _, cars| {
        if cars[0].last.why == Why::KeepClear {
            kept += DT;
        }
    });
    let c = &cars[0];
    let front = c.pos.x as f32 + c.state.front;
    // (at the green's start the keep-clear test meets the people stop for a moment, while it is
    // still placed before the walk's band is in reach; the car never waits at its line)
    assert!(kept < 1.0 && c.state.lane == 2 && c.last.why == Why::People, "held at the line {kept:.1} s; lane {} why {:?}", c.state.lane, c.last.why);
    assert!(front <= 6.0 - omsi_traffic::plan::ZEBRA_STOP + 0.1 && front > -7.0, "waits at the zebra stop inside, past the cross street: front {front:.1}");
}
