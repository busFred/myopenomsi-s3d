//! Headless scenarios for the junction rules of round 4: where a car held at a junction
//! stops (before the stretch it waits for), the player's claim over the pieces ahead, and a
//! car inside the box when its light turns red.

use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network, TrafficLightController};
use omsi_traffic::sim::PLAYER_ID_BASE;
use omsi_traffic::{PlayerView, Simulation, Vehicle, VehicleClass};

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


/// Review7: the zipper's "who went last" lives in `Simulation::signal_groups`, which no keyframe
/// carries. A simulation restored from a keyframe (the vehicles, as `Replay::at` and the reset
/// replay do) and the one that made it part ways at the next level tie.
#[test]
fn review7_zipper_memory_survives_a_keyframe_restore() {
    // two parallel approaches 4 m apart merge into one exit inside object 9
    let mut net = Network::default();
    net.lanes.push(road(&[(-400.0, -2.0), (-10.0, -2.0)], 1)); // 0
    net.lanes.push(road(&[(-400.0, 2.0), (-10.0, 2.0)], 2)); // 1
    net.lanes.push(piece(&[(-10.0, -2.0), (0.0, 0.0)], 0, 128.0)); // 2
    net.lanes.push(piece(&[(-10.0, 2.0), (0.0, 0.0)], 1, 128.0)); // 3
    net.lanes.push(piece(&[(0.0, 0.0), (10.0, 0.0)], 2, 128.0)); // 4
    net.lanes.push(road(&[(10.0, 0.0), (2000.0, 0.0)], 3)); // 5
    let net = finish(net);
    let js = omsi_traffic::Junctions::compile(&net);
    let (_, m) = js.movement(2).unwrap();
    assert!(m.foes.iter().any(|f| f.kind == omsi_traffic::junction::FoeKind::Merge && !f.yields), "{:?}", m.foes);
    // level pairs, one on each approach, every 14 m, rolling at 8 m/s
    let mut cars: Vec<Vehicle> = Vec::new();
    for k in 0..14u64 {
        cars.push(car(100 + k, 0, 380.0 - 14.0 * k as f32, 8.0));
        cars.push(car(200 + k, 1, 380.0 - 14.0 * k as f32 + 0.3, 8.0));
    }
    let mut sim = Simulation::new(&net, vec![]);
    let mut snap: Option<(f64, Vec<Vehicle>, omsi_traffic::reservation::Context)> = None;
    let mut t = 0.0f32;
    let mut a_traj: Vec<Vec<(u64, f64, f64)>> = Vec::new();
    while t < 60.0 {
        sim.tick(&net, DT, &mut cars);
        for &i in &sim.removed { cars.remove(i); }
        t += DT;
        if snap.is_none() && t >= 20.0 && !sim.signal_groups.zipper.is_empty() {
            snap = Some((sim.time, cars.clone(), sim.signal_groups.clone()));
        }
        if snap.is_some() { a_traj.push(cars.iter().map(|v| (v.id, v.pos.x, v.pos.y)).collect()); }
    }
    let (time, cars_at, groups) = snap.expect("no zipper tie in 20 s");
    let drift = |with_memory: bool| -> f64 {
    let mut cars2 = cars_at.clone();
    let mut sim2 = Simulation::new(&net, vec![]);
    sim2.time = time;
    if with_memory { sim2.signal_groups = groups.clone(); }
    let mut worst = 0.0f64;
    for frame in a_traj.iter().skip(1) {
        sim2.tick(&net, DT, &mut cars2);
        for &i in &sim2.removed { cars2.remove(i); }
        for &(id, x, y) in frame {
            if let Some(v) = cars2.iter().find(|v| v.id == id) {
                worst = worst.max(((v.pos.x - x).powi(2) + (v.pos.y - y).powi(2)).sqrt());
            }
        }
    }
    worst
    };
    let control = drift(true);
    assert!(control < 1e-6, "control (zipper memory copied too) drifted {control:.3} m");
    let worst = drift(false);
    assert!(worst < 1e-6, "the restored simulation (vehicles only, as a keyframe) drifted {worst:.2} m from the one it was restored from; with the zipper memory copied: {control}");
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


/// Review7 (rule E): the player's bus dwells at a stop on the lead-in piece, its front 8 m short of
/// the piece that crosses the left turn, standing for a minute. An AI bus standing so is
/// `blocked` and dropped by `START_DELAY`; the player's claim is never `blocked`, and now that it
/// covers the crossing piece too the turner waits out the whole dwell (viz3: AI vehicle-seconds
/// held for a standing player more than 15 m away 1.8 s at the baseline, 124.8 s with r4a).
#[test]
fn review7_rule_e_a_dwelling_player_bus_does_not_hold_the_turner() {
    let net = oncoming_left_turn();
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![car(1, 4, 178.0, 0.0)];
    let player = PlayerView { id: 7, pos: DVec3::new(2.0, -19.0, 0.0), heading: 0.0, half_len: 6.0, half_width: 1.25, speed: 0.0, still: 60.0, blinker: 0, signalling: 0.0, priority: false };
    let mut held = 0.0f32;
    run(&net, &mut sim, &mut cars, 12.0, |_, sim, cars| {
        if cars[0].last.wait_for == Some(PLAYER_ID_BASE + 7) {
            held += DT;
        }
        sim.players = vec![player];
    });
    assert!(held < 3.0, "the turner waited {held:.1} s for a bus standing at its stop");
}

/// Review7 (rule F): one way through object 9 under two lights of one controller, as at
/// BeijingCity's left-turn waiting areas (viz3 junction 1: lane 1124 lit 3.2, then 1255 lit
/// 3.3; 41 such ways in the viz3 fixtures). Eastbound: approach 0, piece 1 (x -20..-19, lit
/// (0, 0)), piece 2 (x -19..10), piece 3 (x 10..11, lit (0, 1), the waiting area's line), piece 4
/// (x 11..40), exit 5; the northbound way (piece 8) crosses piece 2 at x = 0, before the waiting
/// area's line, as 1234/1237 cross lane 1250 before 1255. Light 0 green 0-10 s then
/// red; light 1 red throughout; light 2 (northbound) green from 14 s. A car crawling at 2 m/s
/// passes light 0 on green and is in the waiting area when light 0 turns red: "clearing", the
/// plan drops every signal of controller 0 on its way through, and it drives over light 1's red.
#[test]
fn review7_rule_f_clearing_does_not_run_the_second_red_of_its_controller() {
    let mut net = Network::default();
    net.lanes.push(road(&[(-300.0, 0.0), (-20.0, 0.0)], 1)); // 0
    let mut p1 = piece(&[(-20.0, 0.0), (-19.0, 0.0)], 0, 128.0);
    p1.traffic_light = Some((0, 0));
    net.lanes.push(p1); // 1
    net.lanes.push(piece(&[(-19.0, 0.0), (10.0, 0.0)], 1, 128.0)); // 2
    let mut p3 = piece(&[(10.0, 0.0), (11.0, 0.0)], 2, 128.0);
    p3.traffic_light = Some((0, 1));
    net.lanes.push(p3); // 3
    net.lanes.push(piece(&[(11.0, 0.0), (40.0, 0.0)], 3, 128.0)); // 4
    net.lanes.push(road(&[(40.0, 0.0), (400.0, 0.0)], 2)); // 5
    net.lanes.push(road(&[(0.0, -300.0), (0.0, -20.0)], 3)); // 6
    let mut n1 = piece(&[(0.0, -20.0), (0.0, -19.0)], 4, 128.0);
    n1.traffic_light = Some((0, 2));
    net.lanes.push(n1); // 7
    net.lanes.push(piece(&[(0.0, -19.0), (0.0, 20.0)], 5, 128.0)); // 8
    net.lanes.push(road(&[(0.0, 20.0), (0.0, 300.0)], 4)); // 9
    let net = finish(net);
    let ctl = TrafficLightController::new(vec![vec![(6, 10.0), (0, 50.0)], vec![(0, 60.0)], vec![(0, 14.0), (6, 46.0)]], 60.0);
    let mut sim = Simulation::new(&net, vec![ctl]);
    let mut slow = car(1, 0, 276.0, 2.0);
    slow.state.max_speed_kmh = 7.2;
    slow.state.desire = 1.0;
    let mut cars = vec![slow];
    let mut furthest = f32::NEG_INFINITY;
    let mut red0_at_x = None;
    run(&net, &mut sim, &mut cars, 30.0, |t, sim, cars| {
        let c = &cars[0];
        let x = c.pos.x as f32 + c.state.front;
        furthest = furthest.max(x);
        if red0_at_x.is_none() && sim.lights[0].state(0) < 3 { red0_at_x = Some((t, x)); }
    });
    let (t0, x0) = red0_at_x.unwrap();
    assert!(x0 > -19.0 && x0 < 9.0, "set-up: in the waiting area when light 0 turned red ({t0:.1} s, x {x0:.1})");
    assert!(furthest <= 10.3, "drove over the second line (x = 10, light 1 red throughout) to x = {furthest:.1}");
}
