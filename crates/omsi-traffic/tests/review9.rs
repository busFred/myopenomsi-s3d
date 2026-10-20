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

fn cross(main_prio: f32, side_prio: f32) -> Network {
    // eastbound side road (approach 0, piece 1 x -20..20, exit 2); northbound main road
    // (approach 3, piece 4 y -20..20, exit 5)
    let mut net = Network::default();
    net.lanes.push(road(&[(-300.0, 0.0), (-20.0, 0.0)], 1));
    net.lanes.push(piece(&[(-20.0, 0.0), (20.0, 0.0)], 0, side_prio));
    net.lanes.push(road(&[(20.0, 0.0), (300.0, 0.0)], 2));
    net.lanes.push(road(&[(0.0, -400.0), (0.0, -20.0)], 3));
    net.lanes.push(piece(&[(0.0, -20.0), (0.0, 20.0)], 1, main_prio));
    net.lanes.push(road(&[(0.0, 20.0), (0.0, 400.0)], 4));
    finish(net)
}

/// Rule E: the player's bus drives north on the main road at 10 m/s, `out` m before the
/// meeting place; the AI car stands at its side road line (it gives way by the table). The AI
/// car must not be in the main road's way while the bus passes.
#[test]
fn review9_e_player_moving_at_10_is_not_crossed() {
    for out in [30.0f64, 45.0, 60.0, 90.0] {
        let net = cross(192.0, 64.0);
        let mut sim = Simulation::new(&net, vec![]);
        let mut cars = vec![car(1, 0, 280.0 - 2.0 - 0.5, 0.0)];
        let mut player = PlayerView { id: 7, pos: DVec3::new(0.0, -2.6 - out - 6.0, 0.0), heading: 0.0, half_len: 6.0, half_width: 1.25, speed: 10.0, still: 0.0, blinker: 0, signalling: 0.0, priority: false };
        let mut worst = f64::INFINITY;
        let mut waited = false;
        run(&net, &mut sim, &mut cars, 20.0, |_, sim, cars| {
            let a = &cars[0];
            waited |= a.last.wait_for == Some(PLAYER_ID_BASE + 7);
            // axis-aligned bodies: the AI eastbound, the bus northbound
            let (ax0, ax1, ay0, ay1) = (a.pos.x - a.state.rear as f64, a.pos.x + a.state.front as f64, a.pos.y - 0.95, a.pos.y + 0.95);
            let (px0, px1, py0, py1) = (player.pos.x - 1.25, player.pos.x + 1.25, player.pos.y - 6.0, player.pos.y + 6.0);
            let gap = (px0 - ax1).max(ax0 - px1).max(py0 - ay1).max(ay0 - py1);
            worst = worst.min(gap);
            player.pos.y += player.speed as f64 * DT as f64;
            sim.players = vec![player];
        });
        eprintln!("E {out}: worst {worst:.2} waited {waited} ai x {:.1}", cars[0].pos.x);
        assert!(worst > 0.5, "{out} m out: the AI car came within {worst:.2} m of the moving player's bus (waited for it: {waited})");
    }
}

/// Zipper: two parallel approaches 4 m apart merge into one exit inside object 9 (review7's
/// network). Pairs arriving abreast at the same speed, and with the one behind faster.
fn zipper_net() -> Network {
    let mut net = Network::default();
    net.lanes.push(road(&[(-400.0, -2.0), (-10.0, -2.0)], 1)); // 0
    net.lanes.push(road(&[(-400.0, 2.0), (-10.0, 2.0)], 2)); // 1
    net.lanes.push(piece(&[(-10.0, -2.0), (0.0, 0.0)], 0, 128.0)); // 2
    net.lanes.push(piece(&[(-10.0, 2.0), (0.0, 0.0)], 1, 128.0)); // 3
    net.lanes.push(piece(&[(0.0, 0.0), (10.0, 0.0)], 2, 128.0)); // 4
    net.lanes.push(road(&[(10.0, 0.0), (2000.0, 0.0)], 3)); // 5
    finish(net)
}

#[test]
fn review9_b_zipper_abreast_and_a_faster_loser_never_touch() {
    // (front offset of B relative to A, speed A, speed B)
    for (off, va, vb) in [(0.0f32, 8.0f32, 8.0f32), (0.3, 8.0, 8.0), (-2.5, 6.0, 10.0), (-1.0, 5.0, 12.0), (-4.0, 4.0, 13.0), (2.1, 8.0, 3.0)] {
        let net = zipper_net();
        let mut sim = Simulation::new(&net, vec![]);
        let mut cars = vec![car(1, 0, 340.0, va), car(2, 1, 340.0 + off, vb)];
        let mut log = String::new();
        let min_gap = run(&net, &mut sim, &mut cars, 25.0, |t, _, cars| { if (t * 30.0).round() as i32 % 30 == 0 && t > 3.0 && t < 12.0 { log += &format!(" t{t:.0}:{:.1}/{:.1}@{:.1},{:.1}", cars[0].pos.x, cars[1].pos.x, cars[0].state.speed, cars[1].state.speed); } });
        eprintln!("B {off} {va} {vb}: min {min_gap:.2}{log}");
        assert!(min_gap > 0.3, "offset {off} m, A {va} m/s, B {vb} m/s: touched, {min_gap:.2} m");
    }
}

/// Rule F: the eastbound car crawls over its line on green and is still in the box when its
/// light turns red; the northbound car arrives rolling at 8 m/s soon after its green.
#[test]
fn review9_f_clearing_car_and_a_rolling_fresh_green_do_not_meet() {
    for arrive in [15.5f32, 16.5, 18.0] {
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
        let net = finish(net);
        // eastbound green 0-10, yellow 10-13, red 13-40; northbound red 0-15, green 15-37
        let ctl = TrafficLightController::new(vec![vec![(6, 10.0), (9, 3.0), (0, 27.0)], vec![(0, 15.0), (6, 22.0), (9, 3.0)]], 40.0);
        let mut sim = Simulation::new(&net, vec![ctl]);
        let mut slow = car(1, 0, 280.0 - 2.0 - 1.95 * 9.0, 1.95);
        slow.state.max_speed_kmh = 7.0;
        slow.state.desire = 1.0;
        // the northbound car at 8 m/s, its front at the line at `arrive` s if it kept its speed
        let fresh = car(2, 3, 280.0 - 2.0 - 8.0 * arrive, 8.0);
        let mut cars = vec![slow, fresh];
        let mut tin = (None, None);
        let min_gap = run(&net, &mut sim, &mut cars, 35.0, |t, _, cars| { if tin.0.is_none() && cars[0].pos.x as f32 - cars[0].state.rear > 2.6 { tin.0 = Some(t); } if tin.1.is_none() && cars[1].pos.y as f32 + cars[1].state.front > -2.6 { tin.1 = Some(t); } });
        eprintln!("F {arrive}: min {min_gap:.2} slow cleared {:?} fresh in {:?}", tin.0, tin.1);
        assert!(min_gap > 0.3, "fresh green arriving at {arrive} s: touched the clearing car, {min_gap:.2} m");
    }
}

/// Rule F: a left turner waits in the box (past its line and past its first meeting place,
/// the westbound street) for the oncoming stream; the shared green turns yellow. The turner is
/// now "clearing" (committed, no hold); the oncoming cars within "comfortable" take the yellow.
#[test]
fn review9_f_a_left_turner_clearing_on_amber_meets_the_oncoming_amber_runner() {
    let mut net = Network::default();
    net.lanes.push(road(&[(2.0, -400.0), (2.0, -20.0)], 1)); // 0 northbound approach
    let mut l1 = piece(&[(2.0, -20.0), (2.0, -19.0)], 0, 128.0); // 1 lit stub
    l1.traffic_light = Some((0, 0));
    net.lanes.push(l1);
    net.lanes.push(piece(&[(2.0, -19.0), (2.0, 25.0)], 1, 128.0)); // 2
    net.lanes.push(road(&[(2.0, 25.0), (2.0, 400.0)], 2)); // 3
    net.lanes.push(road(&[(-2.0, 300.0), (-2.0, 26.0)], 3)); // 4 southbound approach
    let mut l5 = piece(&[(-2.0, 26.0), (-2.0, 25.0)], 2, 128.0); // 5 lit stub
    l5.traffic_light = Some((0, 1));
    net.lanes.push(l5);
    let mut turn = piece(&[(-2.0, 25.0), (-2.0, 5.0), (5.0, -2.0), (25.0, -2.0)], 3, 128.0); // 6
    turn.turn = 1;
    net.lanes.push(turn);
    net.lanes.push(road(&[(25.0, -2.0), (300.0, -2.0)], 4)); // 7
    // westbound cross street at y = 17 (nobody on it), its own light
    net.lanes.push(road(&[(300.0, 17.0), (26.0, 17.0)], 5)); // 8
    let mut l9 = piece(&[(26.0, 17.0), (25.0, 17.0)], 4, 128.0); // 9
    l9.traffic_light = Some((0, 2));
    net.lanes.push(l9);
    net.lanes.push(piece(&[(25.0, 17.0), (-25.0, 17.0)], 5, 128.0)); // 10
    net.lanes.push(road(&[(-25.0, 17.0), (-300.0, 17.0)], 6)); // 11
    let net = finish(net);
    {
        let js = omsi_traffic::Junctions::compile(&net);
        for l in [1usize, 2, 5, 6] {
            if let Some((_, m)) = js.movement(l) { eprintln!("lane {l}: cs {} foes {:?}", m.conflict_start, m.foes.iter().map(|f| (js.nodes[0].movements[f.movement].lane, f.mine, f.kind, f.yields)).collect::<Vec<_>>()); }
        }
    }
    for (gap, phase) in [(30.0f32, 20.0f32), (24.0, 20.0), (36.0, 20.0), (30.0, 21.0), (30.0, 22.0)] {
        // north/south green 0..phase, yellow 3 s, red; westbound green later
        let ctl = TrafficLightController::new(vec![vec![(6, phase), (9, 3.0), (0, 60.0 - phase - 3.0)], vec![(6, phase), (9, 3.0), (0, 60.0 - phase - 3.0)], vec![(0, phase + 6.0), (6, 60.0 - phase - 6.0)]], 60.0);
        let mut sim = Simulation::new(&net, vec![ctl]);
        eprintln!("RUN {gap} {phase}");
        let mut cars = vec![car(1, 4, 274.0 - 2.0 - 60.0, 10.0)];
        for k in 0..14u64 {
            cars.push(car(10 + k, 0, 380.0 - 2.0 - gap * k as f32, 12.0));
        }
        let mut clearing_at = None;
        let mut turner_in_box_at_amber = false;
        let mut log = String::new();
        let min_gap = run(&net, &mut sim, &mut cars, 35.0, |t, _, cars| {
            let tr = cars.iter().find(|c| c.id == 1);
            if let Some(tr) = tr {
                if (t - phase).abs() < DT && tr.state.lane == 6 {
                    turner_in_box_at_amber = true;
                }
                if clearing_at.is_none() && t > phase && tr.state.lane == 6 && tr.state.speed > 0.5 {
                    clearing_at = Some(t);
                }
                if t > phase - 1.0 && t < phase + 5.0 && (t * 30.0).round() as i32 % 6 == 0 {
                    log += &format!("\n t {t:.1} turner L{} s {:.1} v {:.1} {:?} wf {:?}", tr.state.lane, tr.state.s, tr.state.speed, tr.last.why, tr.last.wait_for);
                    for c in cars.iter().filter(|c| c.id != 1 && (c.pos.y > -40.0 && c.pos.y < 10.0)) {
                        log += &format!(" | #{} y {:.1} v {:.1} {:?} wf {:?} hold {:?} stop {:?} amber {:?}", c.id, c.pos.y, c.state.speed, c.last.why, c.last.wait_for, c.last.hold, c.last.stop, c.light_decision);
                    }
                }
            }
        });
        eprintln!("F-turner gap {gap} phase {phase}: min {min_gap:.2} turner in box at amber {turner_in_box_at_amber} moved off at {clearing_at:?}");
        assert!(min_gap > 0.3, "gap {gap} phase {phase}: touched, {min_gap:.2} m{log}");
    }
}
