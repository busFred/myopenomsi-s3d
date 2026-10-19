//! Headless scenarios for the access and priority rules: who may use which lane (bus lanes
//! by group density, medians, solid lines, the near side of a junction, a player beside),
//! and who gives way to whom (a green arrow against a red oncoming stream, a stream on the
//! main road, a yellow in a queue, the light of a pedestrian crossing).

use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network, TrafficLightController};
use omsi_traffic::{PlayerView, Simulation, Vehicle, VehicleClass, Why};

const DT: f32 = 1.0 / 30.0;

/// A spline lane (source 1) between two points.
fn road(from: (f64, f64), to: (f64, f64), id: i64) -> omsi_sim::traffic::Lane {
    let mut l = LaneBuilder::polyline(vec![DVec3::new(from.0, from.1, 0.0), DVec3::new(to.0, to.1, 0.0)], LaneKind::Street, 3.5);
    l.source = 1;
    l.key = Some(LaneKey { tile: (0, 0), id, path: 0 });
    l
}

/// A lane of crossing object `obj` (source 2).
fn piece(from: (f64, f64), to: (f64, f64), obj: i64, path: u16, priority: f32) -> omsi_sim::traffic::Lane {
    let mut l = LaneBuilder::polyline(vec![DVec3::new(from.0, from.1, 0.0), DVec3::new(to.0, to.1, 0.0)], LaneKind::Street, 3.5);
    l.source = 2;
    l.key = Some(LaneKey { tile: (0, 0), id: obj, path });
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

/// Run `secs` and keep the worst of everything.
struct Run {
    min_gap: f32,
    hardest_brake: f32,
    longest_stand: f32,
    removed: usize,
}

fn run(net: &Network, sim: &mut Simulation, vehicles: &mut Vec<Vehicle>, secs: f32, mut each: impl FnMut(f32, &Simulation, &mut [Vehicle])) -> Run {
    let mut r = Run { min_gap: f32::INFINITY, hardest_brake: 0.0, longest_stand: 0.0, removed: 0 };
    let mut t = 0.0;
    while t < secs {
        sim.tick(net, DT, vehicles);
        for &i in &sim.removed {
            vehicles.remove(i);
            r.removed += 1;
        }
        for v in vehicles.iter() {
            r.hardest_brake = r.hardest_brake.max(-v.state.acc);
            r.longest_stand = r.longest_stand.max(v.stopped);
        }
        for (a, va) in vehicles.iter().enumerate() {
            for vb in vehicles.iter().skip(a + 1) {
                let d = va.obb().separation(&vb.obb()) as f32;
                r.min_gap = r.min_gap.min(d);
            }
        }
        each(t, sim, vehicles.as_mut_slice());
        t += DT;
    }
    r
}

/// `PROBE=1 cargo test ... -- --nocapture` prints a line per half second.
fn probe(t: f32, frames: &mut u32, cars: &[Vehicle]) {
    *frames += 1;
    if std::env::var_os("PROBE").is_some() && *frames % 15 == 0 {
        let line: Vec<String> = cars
            .iter()
            .map(|c| format!("[{} L{} s={:.1} lat={:.2} v={:.1} why={:?} gap={:?} chg={} pass={}]", c.id, c.state.lane, c.state.s, c.state.lateral, c.state.speed, c.last.why, c.last.lead.map(|l| (l.gap * 10.0).round() / 10.0), c.state.change.is_some(), c.passing.is_some()))
            .collect();
        eprintln!("t={:.1} {}", t, line.join(" "));
    }
}

/// Lanes of one spline (id 1, running along +x from `x0` to `x1`) at the given (offset,
/// width) pairs, offset to the right, i.e. towards -y. `Network::link` pairs the ones
/// whose edges are close as neighbours.
fn spline_lanes(net: &mut Network, x0: f64, x1: f64, lanes: &[(f32, f32)]) {
    for (k, &(offset, width)) in lanes.iter().enumerate() {
        let mut l = road((x0, -(offset as f64)), (x1, -(offset as f64)), 1);
        l.offset = offset;
        l.width = width;
        l.key = Some(LaneKey { tile: (0, 0), id: 1, path: k as u16 });
        net.lanes.push(l);
    }
}

fn finish(mut net: Network) -> Network {
    net.link(1.5);
    net.compute_conflicts();
    net
}

/// A fast car (id 2) 40 m behind a slow one (id 1, 20 km/h) on lane `lane`. Returns
/// whether the fast car ever started a change (or left its lane) and the run.
fn behind_a_slow_car(net: &Network, lane: usize, secs: f32) -> (bool, Run, Vec<Vehicle>) {
    let mut sim = Simulation::new(net, vec![]);
    let mut slow = car(1, lane, 120.0, 5.5);
    slow.state.max_speed_kmh = 20.0;
    slow.state.change_cooldown = 100.0;
    let mut fast = car(2, lane, 80.0, 13.0);
    fast.state.change_cooldown = 0.0;
    let mut cars = vec![slow, fast];
    let mut frames = 0;
    let mut changed = false;
    let r = run(net, &mut sim, &mut cars, secs, |t, _, cars| {
        probe(t, &mut frames, cars);
        changed |= cars[1].state.lane != lane || cars[1].state.change.is_some();
    });
    (changed, r, cars)
}

fn two_lanes(left: impl Fn(&mut omsi_sim::traffic::Lane), right: impl Fn(&mut omsi_sim::traffic::Lane)) -> Network {
    let mut net = Network::default();
    spline_lanes(&mut net, 0.0, 2000.0, &[(0.0, 3.0), (3.5, 3.0)]);
    left(&mut net.lanes[0]);
    right(&mut net.lanes[1]);
    finish(net)
}

// 1 -----------------------------------------------------------------------------------

#[test]
fn a_lane_closed_to_the_group_by_density_is_no_lane_to_change_into() {
    // BeijingCity's bus lane: no `no_cars`, but `trafficdensity 0` for the normal cars
    let net = two_lanes(|l| l.density = 0.0, |_| {});
    assert_eq!(net.lanes[1].left, Some(0), "they are neighbours all the same");
    let (changed, r, cars) = behind_a_slow_car(&net, 1, 25.0);
    assert!(!changed, "the car changed into the bus lane");
    assert_eq!(cars[1].state.lane, 1);
    assert!(r.min_gap > 0.3, "touched: {:.2} m", r.min_gap);
    assert!(r.hardest_brake < 4.5, "braked {:.1} m/s²", r.hardest_brake);
    // control: the same road with traffic on the left lane
    let net = two_lanes(|_| {}, |_| {});
    let (changed, r, _) = behind_a_slow_car(&net, 1, 25.0);
    assert!(changed, "the control car never changed lane");
    assert!(r.min_gap > 0.3 && r.hardest_brake < 4.5);
}

// 2 -----------------------------------------------------------------------------------

#[test]
fn a_median_is_not_a_neighbour_and_a_cycle_lane_across_it_is_no_lane_to_pass_on() {
    let mut net = Network::default();
    spline_lanes(&mut net, 0.0, 2000.0, &[(0.0, 2.36), (3.8, 2.36), (9.4, 1.0)]);
    let net = finish(net);
    assert_eq!((net.lanes[0].right, net.lanes[1].left), (Some(1), Some(0)), "lanes 0 and 1 are neighbours");
    assert_eq!((net.lanes[1].right, net.lanes[2].left), (None, None), "the median keeps lane 2 apart");
    // a one-lane road (2.36 m: no room to squeeze past inside it) with an oncoming lane at
    // (offset, width), a car behind a parked blocker: does it go out on that lane?
    let blocked = |offset: f64, width: f32| {
        let mut net = Network::default();
        let mut a = road((0.0, 0.0), (2000.0, 0.0), 1);
        a.width = 2.36;
        let mut b = road((2000.0, offset), (0.0, offset), 2);
        b.width = width;
        b.reversed = true;
        net.lanes.push(a);
        net.lanes.push(b);
        let net = finish(net);
        assert!(net.opposite(0, 150.0).is_some(), "the oncoming lane is found");
        let mut sim = Simulation::new(&net, vec![]);
        sim.parked.insert(0, vec![omsi_traffic::Parked { s: 200.0, lateral: 0.0, half_width: 0.9, half_len: 2.2 }]);
        let mut cars = vec![car(1, 0, 100.0, 12.0)];
        let mut went_out = false;
        let r = run(&net, &mut sim, &mut cars, 40.0, |_, _, cars| went_out |= cars[0].passing.map(|p| p.lane.is_some()).unwrap_or(false));
        assert!(r.min_gap > 0.3 && r.hardest_brake < 4.5, "gap {:.2} brake {:.1}", r.min_gap, r.hardest_brake);
        (went_out, cars[0].state.s + cars[0].state.front)
    };
    assert!(blocked(3.5, 3.0).0, "control: the car passes on a normal oncoming lane");
    // (a 1 m path right beside the lane is a lane drawn narrow, as Spandau draws its
    // full lanes: a car passes on it; one beyond a 5.6 m strip is a cycle lane behind a
    // median, whatever its width)
    assert!(blocked(3.5, 1.0).0, "control: a narrow path right beside the lane is still the oncoming lane");
    let (went_out, front) = blocked(5.6, 1.0);
    assert!(!went_out, "the car went out on the lane beyond the median");
    assert!(front < 197.8, "the car waits behind the parked one: front at {front:.1}");
}

// 3 -----------------------------------------------------------------------------------

#[test]
fn a_solid_line_forbids_the_change_and_the_pass_on_the_oncoming_lane() {
    // on the current lane, then on the lane beside: no change for speed either way
    for (cur, other) in [(true, false), (false, true)] {
        let net = two_lanes(|l| l.no_overtaking = other, |l| l.no_overtaking = cur);
        let (changed, r, _) = behind_a_slow_car(&net, 1, 25.0);
        assert!(!changed, "changed across a solid line (current {cur}, beside {other})");
        assert!(r.min_gap > 0.3 && r.hardest_brake < 4.5);
    }
    // a one-lane two-way road with 3 m lanes: the pass has to use the oncoming lane
    let pass = |solid: bool| {
        let mut net = Network::default();
        let mut a = road((0.0, 0.0), (2000.0, 0.0), 1);
        a.width = 3.0;
        a.no_overtaking = solid;
        let mut b = road((2000.0, 3.5), (0.0, 3.5), 2);
        b.width = 3.0;
        net.lanes.push(a);
        net.lanes.push(b);
        let net = finish(net);
        let mut sim = Simulation::new(&net, vec![]);
        sim.parked.insert(0, vec![omsi_traffic::Parked { s: 200.0, lateral: 0.0, half_width: 0.9, half_len: 2.2 }]);
        let mut cars = vec![car(1, 0, 100.0, 12.0)];
        let mut went_out = false;
        let r = run(&net, &mut sim, &mut cars, 30.0, |_, _, cars| went_out |= cars[0].passing.is_some() || cars[0].state.lateral.abs() > 1.0);
        (went_out, cars[0].state.s, r)
    };
    let (out, _, _) = pass(false);
    assert!(out, "control: without the solid line the car passes on the oncoming lane");
    let (out, s, r) = pass(true);
    assert!(!out, "the car passed across a solid line");
    assert!(s < 197.8, "the car waits behind the parked one: s {s:.1}");
    assert!(r.min_gap > 0.3 && r.hardest_brake < 4.5);
}

// 4 -----------------------------------------------------------------------------------

#[test]
fn a_car_does_not_change_into_the_lane_the_players_bus_is_driving_along_beside_it() {
    let net = two_lanes(|_| {}, |_| {});
    let drive = |with_bus: bool| {
        let mut sim = Simulation::new(&net, vec![]);
        if with_bus {
            // heading 90 is east; the bus is on the left lane (y = 0), 5 m behind the car
            sim.players.push(PlayerView { id: 7, pos: DVec3::new(95.0, 0.0, 0.0), heading: 90.0, half_len: 6.0, half_width: 1.25, speed: 12.0, still: 0.0, blinker: 0, signalling: 0.0, priority: false });
        }
        let mut slow = car(1, 1, 140.0, 5.5);
        slow.state.max_speed_kmh = 20.0;
        slow.state.change_cooldown = 100.0;
        let mut fast = car(2, 1, 100.0, 13.0);
        fast.state.change_cooldown = 0.0;
        let mut cars = vec![slow, fast];
        let (mut worst, mut hard, mut changed_at, mut beside_frames, mut ever) = (f32::INFINITY, 0.0f32, None, 0, false);
        let mut frames = 0;
        for k in 0..(20.0 / DT) as u32 {
            if let Some(p) = sim.players.first_mut() {
                p.pos.x += 12.0 * DT as f64;
            }
            let r = run(&net, &mut sim, &mut cars, DT, |_, _, _| {});
            worst = worst.min(r.min_gap);
            hard = hard.max(r.hardest_brake);
            probe(k as f32 * DT, &mut frames, &cars);
            ever |= cars[1].state.change.is_some() || cars[1].state.lane != 1;
            let dx = sim.players.first().map(|p| (p.pos.x as f32 - (cars[1].pos.x as f32)).abs());
            if dx.map(|d| d < 15.0).unwrap_or(false) {
                beside_frames += 1;
                if cars[1].state.change.is_some() || cars[1].state.lane != 1 {
                    changed_at = Some(k);
                }
            }
        }
        (changed_at, beside_frames, ever, worst, hard)
    };
    let (at, beside, _, gap, hard) = drive(true);
    assert!(beside > 30, "the bus was never beside the car ({beside} frames)");
    assert_eq!(at, None, "the car changed into the bus's lane at frame {at:?}");
    assert!(gap > 0.3 && hard < 4.5, "gap {gap:.2} brake {hard:.1}");
    let (_, _, ever, _, _) = drive(false);
    assert!(ever, "control: without the bus the car changes lane");
}

// 5 -----------------------------------------------------------------------------------

/// A lane of crossing object 9 through the given points (x, y).
fn bent_piece(pts: &[(f64, f64)], path: u16) -> omsi_sim::traffic::Lane {
    let mut l = LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), LaneKind::Street, 3.5);
    l.source = 2;
    l.key = Some(LaneKey { tile: (0, 0), id: 9, path });
    l
}

/// The west approach turns left (north) across the oncoming (westbound) straight movement.
/// Equal priorities: between oncoming movements the left turn gives way. Each has its own
/// light of controller 0: the straight one (light 0), the turn (light 1, the green arrow).
fn arrow_junction() -> Network {
    let mut net = Network::default();
    net.lanes.push(road((-200.0, -1.75), (-15.0, -1.75), 1)); // 0 west approach
    let mut turn = bent_piece(&[(-15.0, -1.75), (0.0, -1.75), (1.75, 0.0), (1.75, 15.0)], 0);
    turn.turn = 1;
    turn.traffic_light = Some((0, 1));
    net.lanes.push(turn); // 1 the left turn
    net.lanes.push(road((1.75, 15.0), (1.75, 1500.0), 2)); // 2 north exit
    net.lanes.push(road((200.0, 1.75), (15.0, 1.75), 3)); // 3 east approach (westbound)
    let mut straight = bent_piece(&[(15.0, 1.75), (-15.0, 1.75)], 1);
    straight.traffic_light = Some((0, 0));
    net.lanes.push(straight); // 4 oncoming straight
    net.lanes.push(road((-15.0, 1.75), (-1500.0, 1.75), 4)); // 5 west exit
    finish(net)
}

#[test]
fn a_left_turn_on_its_green_arrow_does_not_wait_for_an_oncoming_stream_held_at_red() {
    let net = arrow_junction();
    let js = omsi_traffic::Junctions::compile(&net);
    let (_, turn) = js.movement(1).unwrap();
    assert!(turn.foes.iter().any(|f| f.yields), "the left turn gives way to the oncoming straight");
    // light 0 (straight): red for 30 s then green; light 1 (arrow): green throughout
    let ctl = TrafficLightController::new(vec![vec![(0, 30.0), (6, 30.0)], vec![(6, 60.0)]], 60.0);
    let mut sim = Simulation::new(&net, vec![ctl]);
    // two oncoming cars stand at their red line (the line is at 185 m, they are 2 m long ahead)
    let mut cars = vec![car(1, 0, 140.0, 10.0), car(2, 3, 180.0, 0.0), car(3, 3, 172.0, 0.0)];
    let (mut held_for, mut longest_hold, mut frames) = (0.0f32, 0.0f32, 0);
    let r = run(&net, &mut sim, &mut cars, 20.0, |t, sim, cars| {
        probe(t, &mut frames, cars);
        assert!(sim.lights[0].state(0) < 3, "the oncoming light is red throughout");
        held_for = if cars[0].last.why == Why::Junction { held_for + DT } else { 0.0 };
        longest_hold = longest_hold.max(held_for);
        // the oncoming cars stand at their line all the time
        assert!(cars[1].state.s < 185.0 && cars[2].state.s < 185.0, "an oncoming car ran its red");
    });
    assert!(longest_hold <= 10.0, "the left turner was held {longest_hold:.1} s by the red oncoming queue");
    assert_eq!(cars[0].state.lane, 2, "the left turner got to its exit: lane {} s {:.1}", cars[0].state.lane, cars[0].state.s);
    assert!(r.min_gap > 0.3, "touched: {:.2} m", r.min_gap);
    assert!(r.hardest_brake < 4.5, "braked {:.1} m/s²", r.hardest_brake);
}

// 6 -----------------------------------------------------------------------------------

#[test]
fn the_side_road_never_squeezes_into_a_continuous_main_road_stream() {
    // main road (priority 192) along x, side road (64) from the south; a long main approach
    // so that the stream (a car every 25 m at 13 m/s, 31 cars: 60 s of it) is all there
    let mut net = Network::default();
    net.lanes.push(road((-1000.0, 0.0), (-15.0, 0.0), 1)); // 0
    net.lanes.push(piece((-15.0, 0.0), (15.0, 0.0), 9, 0, 192.0)); // 1
    net.lanes.push(road((15.0, 0.0), (4000.0, 0.0), 2)); // 2
    net.lanes.push(road((0.0, -200.0), (0.0, -15.0), 3)); // 3
    net.lanes.push(piece((0.0, -15.0), (0.0, 15.0), 9, 1, 64.0)); // 4
    net.lanes.push(road((0.0, 15.0), (0.0, 1500.0), 4)); // 5
    let net = finish(net);
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars: Vec<Vehicle> = (0..31u64).map(|k| car(10 + k, 0, 950.0 - 25.0 * k as f32, 13.0)).collect();
    cars.push(car(1, 3, 150.0, 10.0));
    let (mut frames, mut hardest_stream, mut side_enters) = (0, 0.0f32, None);
    let mut stood = false;
    let r = run(&net, &mut sim, &mut cars, 95.0, |t, _, cars| {
        probe(t, &mut frames, &cars[cars.len().saturating_sub(2)..]);
        for c in cars.iter().filter(|c| c.id >= 10) {
            hardest_stream = hardest_stream.max(-c.state.acc);
        }
        let side = cars.iter().find(|c| c.id == 1).unwrap();
        stood |= side.state.speed < 0.3 && side.last.why == Why::Junction;
        // (round 4, rule D: it waits inside the object at the edge of the main road, its
        // front a metre short of the meeting place, rather than at the object's entry: it
        // "enters" when its front reaches the meeting place, 2.6 m short of the main lane)
        if side.pos.y as f32 + side.state.front > -2.6 && side_enters.is_none() {
            side_enters = Some(t);
        }
    });
    assert!(stood, "the side road car never stood at its line");
    let at = side_enters.expect("the side road car never got across once the stream was over");
    if std::env::var_os("PROBE").is_some() {
        eprintln!("side car entered at {at:.1} s, stream braked {hardest_stream:.2} m/s²");
    }
    // the last stream car reaches the line at about 61 s and is across at about 64 s
    assert!(at > 60.0, "the side road car pushed into the stream at {at:.1} s");
    assert!(hardest_stream < 3.0, "a main road car braked {hardest_stream:.1} m/s² for the side road");
    assert!(r.min_gap > 0.3, "touched: {:.2} m", r.min_gap);
    assert!(r.hardest_brake < 4.5, "braked {:.1} m/s²", r.hardest_brake);
}

// 7 -----------------------------------------------------------------------------------

#[test]
fn in_a_queue_rolling_up_to_a_yellow_the_first_goes_and_the_third_and_fourth_stop() {
    let mut net = Network::default();
    net.lanes.push(road((0.0, 0.0), (200.0, 0.0), 1));
    let mut p = piece((200.0, 0.0), (230.0, 0.0), 9, 0, 128.0);
    p.traffic_light = Some((0, 0));
    net.lanes.push(p);
    net.lanes.push(road((230.0, 0.0), (2000.0, 0.0), 2));
    let net = finish(net);
    // green 20 s, yellow 3 s, red 20 s
    let ctl = TrafficLightController::new(vec![vec![(6, 20.0), (9, 3.0), (0, 20.0)]], 43.0);
    let mut sim = Simulation::new(&net, vec![ctl]);
    // four cars rolling at 5 m/s (18 km/h is all they want), 8 m apart; the first one's
    // front is 100 m from the line, so it gets there as the green turns to yellow
    let mut cars: Vec<Vehicle> = (0..4u64)
        .map(|k| {
            let mut c = car(k + 1, 0, 98.0 - 8.0 * k as f32, 5.0);
            c.state.max_speed_kmh = 18.0;
            c
        })
        .collect();
    let mut frames = 0;
    // (time, light state) when each car first stands inside the junction piece
    let mut entered: [Option<(f32, i32)>; 4] = [None; 4];
    let mut stopped_for_light = [false; 4];
    let r = run(&net, &mut sim, &mut cars, 60.0, |t, sim, cars| {
        probe(t, &mut frames, cars);
        for c in cars.iter() {
            let k = c.id as usize - 1;
            if c.state.lane != 0 && entered[k].is_none() {
                entered[k] = Some((t, sim.lights[0].state(0)));
            }
            stopped_for_light[k] |= c.state.lane == 0 && c.state.speed < 0.3 && c.last.why == Why::Light;
        }
    });
    if std::env::var_os("PROBE").is_some() {
        eprintln!("entered (time, light state): {entered:?}, stopped for the light: {stopped_for_light:?}");
    }
    let (t1, state1) = entered[0].expect("the first car never went");
    assert!(t1 < 23.0 && state1 != 0, "the first car went at {t1:.1} s on light state {state1}");
    for k in [2, 3] {
        let (t, _) = entered[k].expect("a car in the queue never got through");
        assert!(stopped_for_light[k], "car {} did not stop for the red", k + 1);
        assert!(t >= 43.0, "car {} crossed the line at {t:.1} s, while the light was yellow or red", k + 1);
    }
    assert!(r.min_gap > 0.3, "touched: {:.2} m", r.min_gap);
    assert!(r.hardest_brake < 4.5, "braked {:.1} m/s²", r.hardest_brake);
}

// 8 -----------------------------------------------------------------------------------

#[test]
fn an_unlit_crossing_piece_is_governed_by_the_lit_stub_at_the_kerb() {
    let mut net = Network::default();
    net.lanes.push(bent_piece(&[(-15.0, 0.0), (15.0, 0.0)], 0)); // 0 the street piece
    let walk = |pts: &[(f64, f64)], path: u16| {
        let mut l = bent_piece(pts, path);
        l.kind = LaneKind::Sidewalk;
        l
    };
    // a 1 m stub at the kerb with the light, linked end to start to the 6 m piece across
    let mut stub = walk(&[(0.0, -4.0), (0.0, -3.0)], 1);
    stub.traffic_light = Some((0, 1));
    net.lanes.push(stub); // 1
    net.lanes.push(walk(&[(0.0, -3.0), (0.0, 3.0)], 2)); // 2 unlit, crosses the street piece
    let net = finish(net);
    assert!(net.lanes[1].next.contains(&2), "the stub is linked to the piece across");
    let ctl = TrafficLightController::new(vec![vec![(0, 10.0)], vec![(0, 10.0)]], 10.0);
    let sim = Simulation::new(&net, vec![ctl]);
    assert_eq!(sim.junctions.nodes.len(), 1);
    assert_eq!(sim.junctions.nodes[0].crossings.len(), 1, "stub and piece are one crossing");
    assert_eq!(sim.crossing_light(&net, 1), Some((0, 1)), "the stub has its own light");
    assert_eq!(sim.crossing_light(&net, 2), Some((0, 1)), "the unlit piece is governed by the stub's");
    assert_eq!(sim.crossing_light(&net, 0), None, "the street piece is no crossing's");
}

/// A crossing chain over two streets of one object (along x at y 0 and y 9): an unlit
/// 2 m piece at the kerb, a 1.005 m stub lit `a`, the piece over the first street, a 2 m
/// piece on the island, a 1 m stub lit `b`, the piece over the second street.
/// Lanes: 0, 1 the streets; 2 the kerb piece; 3 stub a; 4 across; 5 island; 6 stub b; 7 across.
fn two_stub_chain(a: Option<(usize, usize)>, b: Option<(usize, usize)>) -> (Network, Simulation) {
    let mut net = Network::default();
    net.lanes.push(bent_piece(&[(-15.0, 0.0), (15.0, 0.0)], 0));
    net.lanes.push(bent_piece(&[(15.0, 9.0), (-15.0, 9.0)], 1));
    let walk = |pts: &[(f64, f64)], path: u16, light: Option<(usize, usize)>| {
        let mut l = bent_piece(pts, path);
        l.kind = LaneKind::Sidewalk;
        l.traffic_light = light;
        l
    };
    net.lanes.push(walk(&[(0.0, -6.005), (0.0, -4.005)], 2, None));
    net.lanes.push(walk(&[(0.0, -4.005), (0.0, -3.0)], 3, a));
    net.lanes.push(walk(&[(0.0, -3.0), (0.0, 3.0)], 4, None));
    net.lanes.push(walk(&[(0.0, 3.0), (0.0, 5.0)], 5, None));
    net.lanes.push(walk(&[(0.0, 5.0), (0.0, 6.0)], 6, b));
    net.lanes.push(walk(&[(0.0, 6.0), (0.0, 12.0)], 7, None));
    let net = finish(net);
    let ctl = TrafficLightController::new(vec![vec![(0, 10.0)], vec![(0, 10.0)], vec![(0, 10.0)]], 10.0);
    let sim = Simulation::new(&net, vec![ctl]);
    (net, sim)
}

#[test]
fn each_crossing_piece_takes_the_light_of_the_lit_stub_before_it() {
    // Taijichang j28 (case09): one chain of 56 pieces under lights 18.0..18.3, and every
    // unlit piece got the first, 18.1
    let (net, sim) = two_stub_chain(Some((0, 1)), Some((0, 2)));
    assert_eq!(sim.junctions.nodes.len(), 1);
    let xs = &sim.junctions.nodes[0].crossings;
    assert_eq!(xs.len(), 1, "one chain: {xs:?}");
    assert_eq!(xs[0].foot_lanes.len(), 6);
    let light = |l: usize| sim.crossing_light(&net, l);
    assert_eq!(light(3), Some((0, 1)));
    assert_eq!(light(4), Some((0, 1)), "the piece after stub a");
    assert_eq!(light(5), Some((0, 1)), "the island piece after it");
    assert_eq!(light(6), Some((0, 2)));
    assert_eq!(light(7), Some((0, 2)), "the piece after stub b, not a's");
    assert_eq!(light(2), None, "before every stub in a chain of two groups");
    // one group: every piece has it, the kerb piece too (as before)
    let (net, sim) = two_stub_chain(Some((0, 1)), Some((0, 1)));
    for l in 2..8 {
        assert_eq!(sim.crossing_light(&net, l), Some((0, 1)), "piece {l}");
    }
    // a zebra: no light anywhere
    let (net, sim) = two_stub_chain(None, None);
    assert_eq!(sim.junctions.nodes[0].crossings.len(), 1);
    for l in 2..8 {
        assert_eq!(sim.crossing_light(&net, l), None, "piece {l}");
    }
    // only the second stub lit: what lies before it has its light (one group)
    let (net, sim) = two_stub_chain(None, Some((0, 2)));
    assert_eq!(sim.crossing_light(&net, 4), Some((0, 2)));
    assert_eq!(sim.crossing_light(&net, 7), Some((0, 2)));
}

#[test]
fn a_long_crossing_of_two_metre_pieces_between_lit_stubs_stays_one_chain() {
    // X10 j530: 1.35 m stubs lit at both kerbs and a 33 m crossing of 2 m pieces over four
    // lanes
    let mut net = Network::default();
    for (k, y) in [-12.0, -4.0, 4.0, 12.0].into_iter().enumerate() {
        net.lanes.push(bent_piece(&[(-20.0, y), (20.0, y)], k as u16));
    }
    let walk = |a: f64, b: f64, path: u16| {
        let mut l = bent_piece(&[(0.0, a), (0.0, b)], path);
        l.kind = LaneKind::Sidewalk;
        l
    };
    let first = net.lanes.len();
    let mut stub = walk(-17.85, -16.5, 10);
    stub.traffic_light = Some((0, 0));
    net.lanes.push(stub);
    for k in 0..16 {
        net.lanes.push(walk(-16.5 + 2.0 * k as f64, -14.5 + 2.0 * k as f64, 11 + k as u16));
    }
    let mut far = walk(15.5, 16.85, 30);
    far.traffic_light = Some((0, 0));
    net.lanes.push(far);
    let last = net.lanes.len() - 1;
    let net = finish(net);
    let ctl = TrafficLightController::new(vec![vec![(0, 10.0)]], 10.0);
    let sim = Simulation::new(&net, vec![ctl]);
    let xs = &sim.junctions.nodes[0].crossings;
    assert_eq!(xs.len(), 1, "one chain");
    assert_eq!(xs[0].foot_lanes.len(), 18);
    for l in first..=last {
        assert_eq!(sim.crossing_light(&net, l), Some((0, 0)), "piece {l}");
    }
}

#[test]
fn a_long_lit_piece_governs_itself_and_the_piece_after_it() {
    // Grundorf: an 11.6 m lit crossing, then a short unlit piece onto the kerb
    let mut net = Network::default();
    net.lanes.push(bent_piece(&[(-15.0, 0.0), (15.0, 0.0)], 0));
    let mut lit = bent_piece(&[(0.0, -5.8), (0.0, 5.8)], 1);
    lit.kind = LaneKind::Sidewalk;
    lit.traffic_light = Some((0, 0));
    net.lanes.push(lit);
    let mut kerb = bent_piece(&[(0.0, 5.8), (0.0, 7.0)], 2);
    kerb.kind = LaneKind::Sidewalk;
    net.lanes.push(kerb);
    let net = finish(net);
    let ctl = TrafficLightController::new(vec![vec![(0, 10.0)]], 10.0);
    let sim = Simulation::new(&net, vec![ctl]);
    assert_eq!(sim.junctions.nodes[0].crossings.len(), 1);
    assert_eq!(sim.crossing_light(&net, 1), Some((0, 0)));
    assert_eq!(sim.crossing_light(&net, 2), Some((0, 0)));
}

// 9 -----------------------------------------------------------------------------------

/// Two lanes of a road running to x = `end`, where each leads into a junction piece
/// (its own object path, 30 m) and on to an exit road.
fn road_into_a_junction(end: f64) -> Network {
    let mut net = Network::default();
    spline_lanes(&mut net, 0.0, end, &[(0.0, 3.0), (3.5, 3.0)]);
    for (k, y) in [0.0, -3.5].into_iter().enumerate() {
        net.lanes.push(piece((end, y), (end + 30.0, y), 9, k as u16, 128.0));
        net.lanes.push(road((end + 30.0, y), (end + 2000.0, y), 10 + k as i64));
    }
    finish(net)
}

#[test]
fn no_change_for_speed_in_the_last_forty_metres_before_a_junction() {
    // the cars are 30 m (the slow one 10 m) from the junction piece, or 200 m
    for (end, expect_change) in [(300.0, false), (470.0, true)] {
        let net = road_into_a_junction(end);
        assert_eq!(net.lanes[1].left, Some(0));
        let mut sim = Simulation::new(&net, vec![]);
        let mut slow = car(1, 1, 290.0 - 2.0, 5.5);
        slow.state.max_speed_kmh = 20.0;
        slow.state.change_cooldown = 100.0;
        let mut fast = car(2, 1, 270.0 - 2.0, 11.0);
        fast.state.change_cooldown = 0.0;
        // (a spawned car has its way on planned: the junction is in sight from the first frame)
        slow.state.plan_next(&net);
        fast.state.plan_next(&net);
        let mut cars = vec![slow, fast];
        let (mut frames, mut changed) = (0, false);
        let r = run(&net, &mut sim, &mut cars, if expect_change { 15.0 } else { 6.0 }, |t, _, cars| {
            probe(t, &mut frames, cars);
            changed |= cars[1].state.change.is_some() || cars[1].state.lane == 0;
        });
        assert_eq!(changed, expect_change, "junction at {end} m: the car changed lane = {changed}");
        assert!(r.min_gap > 0.3 && r.hardest_brake < 4.5, "gap {:.2} brake {:.1}", r.min_gap, r.hardest_brake);
    }
}
