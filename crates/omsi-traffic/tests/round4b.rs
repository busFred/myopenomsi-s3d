//! Headless scenarios for round 4b: converging lanes of one object compiled as meeting
//! places (X1), leads by the body's real place (X2), frozen lane changes (X3), the keep-back
//! only where a pass can start (X4), passes that signal and give way to a bus pulling out
//! (X5), passes that need room past the obstacle (X6), and the negotiation of a prolonged
//! mutual hold (X7).

use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network};
use omsi_traffic::junction::FoeKind;
#[allow(unused_imports)]
use omsi_traffic::{Junctions, PlayerView, Simulation, Vehicle, VehicleClass};

#[allow(dead_code)]
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

/// A lane of crossing object 9 (source 2), 1 m wide as BeijingCity's pieces are.
fn piece(pts: &[(f64, f64)], path: u16) -> omsi_sim::traffic::Lane {
    let mut l = lane(pts, LaneKind::Street, 1.0);
    l.source = 2;
    l.key = Some(LaneKey { tile: (0, 0), id: 9, path });
    l.priority = 128.0;
    l
}

fn finish(mut net: Network) -> Network {
    net.link(1.5);
    net.compute_conflicts();
    net
}

#[allow(dead_code)]
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

/// The meeting place the network compiled between lanes `a` and `b`, if any.
fn meeting(net: &Network, a: usize, b: usize) -> Option<omsi_sim::traffic::Crossing> {
    net.crossings.get(a).and_then(|c| c.iter().find(|c| c.other == b)).copied()
}

/// Two lanes of object 9 that start 2.2 m apart and run into one end point, as case07's
/// 5229/5231 and 3769/3770 do (starts under 3 m apart: no merge to the old rule), with the
/// way on after them. Lanes: 0 and 1 the approaches, 2 the straight piece, 3 the one bending
/// in, 4 the way on.
fn converging(turn_b: i32) -> Network {
    let mut net = Network::default();
    net.lanes.push(road(&[(-100.0, 0.0), (0.0, 0.0)], 1));
    net.lanes.push(road(&[(-100.0, 2.2), (0.0, 2.2)], 2));
    net.lanes.push(piece(&[(0.0, 0.0), (30.0, 0.0)], 0));
    let mut b = piece(&[(0.0, 2.2), (15.0, 1.2), (30.0, 0.0)], 1);
    b.turn = turn_b;
    net.lanes.push(b);
    net.lanes.push(road(&[(30.0, 0.0), (200.0, 0.0)], 3));
    finish(net)
}

#[test]
fn x1_two_lanes_converging_from_starts_two_metres_apart_meet_over_the_whole_stretch() {
    let net = converging(0);
    let m = meeting(&net, 2, 3).expect("the converging pair has a meeting place");
    assert!(m.merge, "{m:?}");
    // the stretch: from where they come within 2.6 m (the start: 2.2 m apart) to their end
    assert!((m.at - 30.0).abs() < 0.6, "{m:?}");
    assert!(m.at - m.before <= 0.5, "{m:?}");
    let js = Junctions::compile(&net);
    let (_, mv) = js.movement(2).unwrap();
    let foe = mv.foes.iter().find(|f| js.nodes[0].movements[f.movement].lane == 3).expect("a foe");
    assert_eq!(foe.kind, FoeKind::Merge);
    assert!(foe.mine.0 < 0.6 && foe.mine.1 > 29.5, "{foe:?}");
    // neither turns nor comes off another road: no rule separates them, a zipper
    let mi = js.nodes[0].movement_of(2).unwrap();
    assert!(js.is_zipper(0, mi, foe), "{foe:?}");
    // compiling them again (a replay recompiles a recording's header) gives the same places
    let mut again = converging(0);
    again.compute_conflicts();
    assert_eq!(meeting(&again, 2, 3), meeting(&net, 2, 3));
}

#[test]
fn x1_the_right_of_way_decides_a_converging_pair_the_turner_yields() {
    let net = converging(2);
    let js = Junctions::compile(&net);
    let (_, straight) = js.movement(2).unwrap();
    let (_, turner) = js.movement(3).unwrap();
    assert!(!straight.foes[0].yields, "{:?}", straight.foes);
    assert!(turner.foes[0].yields, "{:?}", turner.foes);
}

#[test]
fn x1_a_header_network_without_them_gets_them_compiled() {
    let net = converging(0);
    let mut old = converging(0);
    for c in old.crossings.iter_mut() {
        c.clear();
    }
    for c in old.conflicts.iter_mut() {
        c.clear();
    }
    old.compute_conflicts();
    assert_eq!(meeting(&old, 2, 3), meeting(&net, 2, 3));
}

#[test]
fn x1_lanes_of_a_two_lane_road_side_by_side_are_no_meeting_place() {
    for (gap, width) in [(2.3, 2.3f32), (2.5, 2.3), (2.6, 2.5), (2.3, 1.0)] {
        let mut net = Network::default();
        net.lanes.push(road(&[(-100.0, 0.0), (0.0, 0.0)], 1));
        net.lanes.push(road(&[(-100.0, gap), (0.0, gap)], 2));
        let mut a = piece(&[(0.0, 0.0), (30.0, 0.0)], 0);
        a.width = width;
        let mut b = piece(&[(0.0, gap), (30.0, gap)], 1);
        b.width = width;
        net.lanes.push(a);
        net.lanes.push(b);
        let net = finish(net);
        assert_eq!(meeting(&net, 2, 3), None, "{gap} m apart, {width} m wide");
    }
    // edges overlapping by 0.4 m: they meet
    let mut net = Network::default();
    let mut a = piece(&[(0.0, 0.0), (30.0, 0.0)], 0);
    a.width = 2.7;
    let mut b = piece(&[(0.0, 2.3), (30.0, 2.3)], 1);
    b.width = 2.7;
    net.lanes.push(a);
    net.lanes.push(b);
    let net = finish(net);
    assert!(meeting(&net, 0, 1).is_some_and(|m| m.merge));
}

#[test]
fn x1_pieces_of_one_way_forks_and_oncoming_lanes_stay_out() {
    let mut net = Network::default();
    // a lead-in, its 1 m lit stub and the piece after it: one way
    net.lanes.push(road(&[(-100.0, 0.0), (-20.0, 0.0)], 1));
    net.lanes.push(piece(&[(-20.0, 0.0), (0.0, 0.0)], 0));
    let mut stub = piece(&[(0.0, 0.0), (1.0, 0.0)], 1);
    stub.traffic_light = Some((0, 0));
    net.lanes.push(stub);
    net.lanes.push(piece(&[(1.0, 0.0), (30.0, 0.0)], 2));
    // a fork from the piece's end: straight on (3) and bending away (4)
    net.lanes.push(piece(&[(1.0, 0.0), (15.0, 1.0), (30.0, 6.0)], 3));
    // an oncoming lane 1.5 m beside the piece
    net.lanes.push(piece(&[(30.0, 1.5), (1.0, 1.5)], 4));
    let net = finish(net);
    assert_eq!(meeting(&net, 1, 2), None);
    assert_eq!(meeting(&net, 1, 3), None);
    assert_eq!(meeting(&net, 3, 4), None, "a fork from one point");
    assert_eq!(meeting(&net, 3, 5), None, "the oncoming lane");
    assert!(net.one_way_pieces(1, 2));
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

#[test]
fn x1_two_lead_ins_side_by_side_before_their_lights_leave_the_queue_held_by_its_light() {
    // lead-ins 1 and 6 run 1.8 m apart (a meeting place now) to their lit stubs at x = 0, then
    // on across a main road at x = 20: the light at the line is the car's, though the first
    // meeting place of its way, the side-by-side stretch, lies before it
    let mut net = Network::default();
    net.lanes.push(road(&[(-300.0, 0.0), (-20.0, 0.0)], 1)); // 0
    net.lanes.push(piece(&[(-20.0, 0.0), (0.0, 0.0)], 0)); // 1
    let mut stub = piece(&[(0.0, 0.0), (1.0, 0.0)], 1); // 2
    stub.traffic_light = Some((0, 0));
    net.lanes.push(stub);
    net.lanes.push(piece(&[(1.0, 0.0), (40.0, 0.0)], 2)); // 3
    net.lanes.push(road(&[(40.0, 0.0), (400.0, 0.0)], 2)); // 4
    net.lanes.push(road(&[(-300.0, 1.8), (-20.0, 1.8)], 3)); // 5
    net.lanes.push(piece(&[(-20.0, 1.8), (0.0, 1.8)], 3)); // 6
    let mut stub_b = piece(&[(0.0, 1.8), (1.0, 1.8)], 4); // 7
    stub_b.traffic_light = Some((0, 1));
    net.lanes.push(stub_b);
    net.lanes.push(piece(&[(1.0, 1.8), (10.0, 1.8), (40.0, 20.0)], 5)); // 8
    net.lanes.push(road(&[(20.0, -400.0), (20.0, -15.0)], 4)); // 9
    let mut main = piece(&[(20.0, -15.0), (20.0, 30.0)], 6); // 10
    main.priority = 192.0;
    net.lanes.push(main);
    net.lanes.push(road(&[(20.0, 30.0), (20.0, 400.0)], 5)); // 11
    let net = finish(net);
    assert!(meeting(&net, 1, 6).is_some_and(|m| m.merge && m.side), "{:?}", meeting(&net, 1, 6));
    let ctl = omsi_sim::traffic::TrafficLightController::new(vec![vec![(0, 60.0)], vec![(0, 60.0)]], 60.0);
    let mut sim = Simulation::new(&net, vec![ctl]);
    let mut cars = vec![car(1, 0, 240.0, 10.0)];
    let mut held_by_light = false;
    run(&net, &mut sim, &mut cars, 20.0, |_, sim, cars| {
        if cars[0].state.speed < 0.1 {
            held_by_light |= sim.claims.iter().any(|(c, _)| c.id == 1 && c.light_red);
        }
    });
    assert!(cars[0].pos.x < 0.0, "it ran its red: x = {:.1}", cars[0].pos.x);
    assert!(held_by_light, "standing at its red, its claim was not held by its light");
}

#[test]
fn x1_a_car_standing_at_the_second_red_of_a_long_object_is_held_by_it_not_clearing() {
    // light 0 at the object's entry (x = -40, green 0-4 s, then red), light 1 at a second line
    // inside it (x = 0, red throughout) before the main road at x = 20: the car takes the
    // first line on green and stands at the second; once the first turns red it is in the
    // box past its line, but the second red holds it: its claim is held by that light, not
    // clearing the box (and so committed to all who wait for it; viz case06 #4599, 50 s)
    let mut net = Network::default();
    net.lanes.push(road(&[(-300.0, 0.0), (-41.0, 0.0)], 1)); // 0
    let mut stub0 = piece(&[(-41.0, 0.0), (-40.0, 0.0)], 0); // 1
    stub0.traffic_light = Some((0, 0));
    net.lanes.push(stub0);
    net.lanes.push(piece(&[(-40.0, 0.0), (0.0, 0.0)], 1)); // 2
    let mut stub1 = piece(&[(0.0, 0.0), (1.0, 0.0)], 2); // 3
    stub1.traffic_light = Some((0, 1));
    net.lanes.push(stub1);
    net.lanes.push(piece(&[(1.0, 0.0), (40.0, 0.0)], 3)); // 4
    net.lanes.push(road(&[(40.0, 0.0), (400.0, 0.0)], 2)); // 5
    net.lanes.push(road(&[(20.0, -400.0), (20.0, -15.0)], 3)); // 6
    let mut main = piece(&[(20.0, -15.0), (20.0, 30.0)], 4); // 7
    main.priority = 192.0;
    net.lanes.push(main);
    net.lanes.push(road(&[(20.0, 30.0), (20.0, 400.0)], 4)); // 8
    // (an unlit side road across the way at x = -20, between the two lines: the first meeting
    // place of the way, behind the car standing at the second line)
    net.lanes.push(road(&[(-20.0, -400.0), (-20.0, -15.0)], 5)); // 9
    net.lanes.push(piece(&[(-20.0, -15.0), (-20.0, 15.0)], 5)); // 10
    net.lanes.push(road(&[(-20.0, 15.0), (-20.0, 400.0)], 6)); // 11
    let net = finish(net);
    let ctl = omsi_sim::traffic::TrafficLightController::new(vec![vec![(6, 4.0), (9, 2.0), (0, 54.0)], vec![(0, 60.0)]], 60.0);
    let mut sim = Simulation::new(&net, vec![ctl]);
    let mut cars = vec![car(1, 0, 255.0, 9.0)];
    let (mut clearing, mut held) = (false, false);
    run(&net, &mut sim, &mut cars, 25.0, |t, sim, cars| {
        if t > 8.0 && cars[0].state.speed < 0.1 {
            for (c, _) in sim.claims.iter().filter(|(c, _)| c.id == 1) {
                clearing |= c.clearing;
                held |= c.light_red;
            }
        }
    });
    assert!(cars[0].pos.x < 0.0 && cars[0].pos.x > -40.0, "x = {:.1}", cars[0].pos.x);
    assert!(cars[0].passed_on_go.is_some(), "it did not take the first line on green");
    assert!(held && !clearing, "held by the light {held}, clearing {clearing}");
}

/// Two lanes of one road eastbound, lane 0 at y = 0 and lane 1 at y = -3.5 (on its right).
fn two_lanes() -> Network {
    let mut net = Network::default();
    let mut a = road(&[(0.0, 0.0), (300.0, 0.0)], 1);
    a.offset = 0.0;
    let mut b = road(&[(0.0, -3.5), (300.0, -3.5)], 1);
    b.key = Some(LaneKey { tile: (0, 0), id: 1, path: 1 });
    b.offset = 3.5;
    net.lanes.push(a);
    net.lanes.push(b);
    finish(net)
}

/// What vehicle `i` sees, the bodies placed and the lanes occupied as a frame does.
fn seen(net: &Network, vehicles: &mut [Vehicle], i: usize) -> omsi_traffic::Perception {
    let mut sim = Simulation::new(net, vec![]);
    for v in vehicles.iter_mut() {
        v.state.plan_next(net);
    }
    sim.update_positions(net, vehicles);
    sim.build_occupancy(net, vehicles);
    omsi_traffic::perception::perceive(net, &sim, vehicles, i)
}

#[test]
fn x2_a_car_frozen_most_of_the_way_into_the_next_lane_is_not_the_lead_of_the_one_behind() {
    let net = two_lanes();
    let mut follower = car(1, 0, 80.0, 0.0);
    follower.stopped = 10.0;
    // still registered on lane 0, its body 85 % over into lane 1 (3.3 m to the side)
    let mut changer = car(2, 0, 100.0, 0.0);
    changer.state.change = Some(omsi_sim::traffic::LaneChange { to: 1, t: 0.85, length: 12.0, s_to: 100.0, dir: 2, wait: 0.0, bypass: false, full_lock: false });
    let mut cars = vec![follower, changer];
    let p = seen(&net, &mut cars, 0);
    assert!((cars[1].pos.y + 3.3).abs() < 0.2, "the changer stands at y = {:.2}", cars[1].pos.y);
    assert!(p.leader.is_none(), "a phantom lead: {:?}", p.leader);
    // counter-case: one 30 % over is still in my corridor
    cars[1].state.change = Some(omsi_sim::traffic::LaneChange { to: 1, t: 0.3, length: 12.0, s_to: 100.0, dir: 2, wait: 0.0, bypass: false, full_lock: false });
    let p = seen(&net, &mut cars, 0);
    let l = p.leader.expect("the car half in my lane is my lead");
    assert!((l.gap - 16.0).abs() < 0.3, "gap {:.2}", l.gap);
}

#[test]
fn x2_a_body_in_my_corridor_is_my_lead_whatever_lane_it_is_registered_on() {
    let net = two_lanes();
    // registered on lane 1, but standing 3.5 m to its left: in lane 0, 16 m ahead of my front
    let mut other = car(2, 1, 100.0, 0.0);
    other.state.lateral = -3.5;
    other.state.lateral_target = -3.5;
    other.state.lateral_ramp = (-3.5, -3.5, 0.0, 1.0);
    let mut cars = vec![car(1, 0, 80.0, 8.0), other];
    let p = seen(&net, &mut cars, 0);
    assert!((cars[1].pos.y).abs() < 0.05, "it stands at y = {:.2}", cars[1].pos.y);
    let l = p.leader.expect("the body ahead in my lane is my lead");
    assert!((l.gap - 16.0).abs() < 0.3, "gap {:.2}", l.gap);
    // and the one in its own lane beside mine is not: lateral 0, 3.5 m to my right
    cars[1].state.lateral = 0.0;
    cars[1].state.lateral_target = 0.0;
    cars[1].state.lateral_ramp = (0.0, 0.0, 0.0, 1.0);
    let p = seen(&net, &mut cars, 0);
    assert!(p.leader.is_none(), "{:?}", p.leader);
    // a car straight ahead in my lane: the bumper gap, as the lane-based lead had it
    let mut cars = vec![car(1, 0, 80.0, 8.0), car(3, 0, 110.0, 5.0)];
    let p = seen(&net, &mut cars, 0);
    let l = p.leader.expect("the car ahead");
    assert!((l.gap - 26.0).abs() < 0.3 && (l.speed - 5.0).abs() < 0.1, "{l:?}");
    assert!(matches!(l.what, omsi_traffic::Seen::Vehicle(1)), "{:?}", l.what);
}

#[test]
fn x2_changing_lanes_the_car_ahead_in_the_lane_i_move_to_is_my_lead() {
    // I move over into lane 1 at 10 m/s; a car 12 m ahead there at 4 m/s: my corridor reaches
    // its lane only metres on, but it is my lead now (viz2 case07 #10906 cut in on one)
    let net = two_lanes();
    let mut me = car(1, 0, 100.0, 10.0);
    me.state.start_change(&net, 1, 2);
    if let Some(c) = me.state.change.as_mut() {
        c.wait = 0.0;
    }
    let mut cars = vec![me, car(2, 1, 112.0, 4.0)];
    let p = seen(&net, &mut cars, 0);
    let l = p.leader.expect("the car in the lane I move to");
    assert!(matches!(l.what, omsi_traffic::Seen::Vehicle(1)) && (l.gap - 8.0).abs() < 0.5, "{l:?}");
    // and the car changing into my lane ahead of me, from the lane beside: my lead too
    let mut other = car(3, 1, 112.0, 4.0);
    other.state.start_change(&net, 0, 1);
    let mut cars = vec![car(1, 0, 100.0, 10.0), other];
    let p = seen(&net, &mut cars, 0);
    assert!(p.leader.is_some_and(|l| (l.gap - 8.0).abs() < 0.5), "{:?}", p.leader);
}

#[test]
fn x2_my_follower_that_has_me_as_its_lead_is_not_my_lead() {
    // my way goes east, round a tight U-turn and back west 5 m north: the car behind me on
    // the way back (its body in the corridor my way takes after the turn) follows me
    let mut net = Network::default();
    net.lanes.push(road(&[(0.0, 0.0), (50.0, 0.0)], 1));
    net.lanes.push(road(&[(50.0, 0.0), (51.8, 0.7), (52.5, 2.5), (51.8, 4.3), (50.0, 5.0)], 2));
    net.lanes.push(road(&[(50.0, 5.0), (0.0, 5.0)], 3));
    let net = finish(net);
    let mut follower = car(2, 2, 20.0, 0.0);
    follower.last.lead_id = Some(1);
    let mut cars = vec![car(1, 0, 40.0, 3.0), follower];
    let p = seen(&net, &mut cars, 0);
    assert!(p.leader.is_none(), "{:?}", p.leader);
    // counter-case: not following me, it is in my way
    cars[1].last.lead_id = None;
    let p = seen(&net, &mut cars, 0);
    assert!(p.leader.is_some_and(|l| l.gap > 20.0), "{:?}", p.leader);
}

/// A car (id 1) on lane 0 at x = 100 changing over to lane 1 with its change at `t`, standing
/// behind a car (id 2) standing in lane 1 just ahead of it and one (id 3) standing in lane 0:
/// the change cannot go on.
fn frozen_change(t: f32) -> (Network, Vec<Vehicle>) {
    let net = two_lanes();
    let mut me = car(1, 0, 100.0, 0.0);
    me.state.change = Some(omsi_sim::traffic::LaneChange { to: 1, t, length: 12.0, s_to: 100.0, dir: 2, wait: 0.0, bypass: false, full_lock: false });
    let mut ahead_target = car(2, 1, 105.5, 0.0);
    ahead_target.pull_out = 1000.0;
    let mut ahead_own = car(3, 0, 105.5, 0.0);
    ahead_own.pull_out = 1000.0;
    (net, vec![me, ahead_target, ahead_own])
}

#[test]
fn x3_a_change_frozen_more_than_half_across_is_done_in_place_and_one_less_given_up() {
    for (t, done) in [(0.8f32, true), (0.25, false)] {
        let (net, mut cars) = frozen_change(t);
        let mut sim = Simulation::new(&net, vec![]);
        sim.update_positions(&net, &mut cars);
        let start = cars[0].pos;
        let mut jump = 0.0f64;
        let mut prev = start;
        let mut settled_at: Option<(f32, f64, f32)> = None;
        run(&net, &mut sim, &mut cars, 8.0, |time, _, cars| {
            let me = cars.iter().find(|c| c.id == 1).unwrap();
            jump = jump.max((me.pos - prev).truncate().length());
            prev = me.pos;
            if me.state.change.is_none() && settled_at.is_none() {
                settled_at = Some((time, me.pos.y, me.state.lateral));
            }
        });
        let me = cars.iter().find(|c| c.id == 1).unwrap();
        let (at, y, lat) = settled_at.expect("the change was never settled");
        assert!(at > 4.9 && at < 5.5, "settled at {at:.2} s");
        assert_eq!(me.state.lane, if done { 1 } else { 0 }, "t {t}");
        // where the body was when settled: on lane 1 offset to the left, or on lane 0 offset
        // to the right (since round 6 L3 the offset then eases in place at SETTLE_CREEP while
        // the car stands, so it is no longer where it was at the end of the run)
        assert!((y - start.y).abs() < 0.05, "the body moved from y {:.2} to {:.2}", start.y, y);
        assert!(jump < 0.05, "a jump of {jump:.2} m");
        let expect = if done { -(y + 3.5) as f32 } else { -y as f32 };
        assert!((lat - expect).abs() < 0.05, "lateral {lat:.2}, expected {expect:.2}");
        assert!(me.state.lateral.abs() < lat.abs() - 0.3, "the offset {lat:.2} did not ease in place: {:.2}", me.state.lateral);
        // the occupancy follows the body
        sim.build_occupancy(&net, &cars);
        let i = cars.iter().position(|c| c.id == 1).unwrap();
        let occ = sim.occupancy.get(&me.state.lane).unwrap().iter().find(|o| o.vehicle == i).unwrap();
        assert!((occ.lateral - me.state.lateral).abs() < 0.05, "registered at lateral {:.2}", occ.lateral);
        let other = if done { 0 } else { 1 };
        assert!(sim.occupancy.get(&other).map(|v| v.iter().all(|o| o.vehicle != i)).unwrap_or(true));
    }
}

#[test]
fn x3_a_frozen_change_is_not_done_into_a_car_standing_beside_it() {
    // lanes 4.5 m apart; the change 55 % across, a car standing in the target lane level with
    // it (viz case06: done in place, the two stood side by side in one lane and touched as the
    // offset eased off): given up instead, back on its own lane
    let mut net = Network::default();
    let a = road(&[(0.0, 0.0), (300.0, 0.0)], 1);
    let mut b = road(&[(0.0, -4.5), (300.0, -4.5)], 1);
    b.key = Some(LaneKey { tile: (0, 0), id: 1, path: 1 });
    b.offset = 4.5;
    net.lanes.push(a);
    net.lanes.push(b);
    let net = finish(net);
    let mut me = car(1, 0, 100.0, 0.0);
    me.state.change = Some(omsi_sim::traffic::LaneChange { to: 1, t: 0.53, length: 12.0, s_to: 100.0, dir: 2, wait: 0.0, bypass: false, full_lock: false });
    let mut beside = car(2, 1, 97.0, 0.0);
    beside.pull_out = 1000.0;
    let mut ahead_target = car(3, 1, 105.5, 0.0);
    ahead_target.pull_out = 1000.0;
    let mut ahead_own = car(4, 0, 105.5, 0.0);
    ahead_own.pull_out = 1000.0;
    let mut cars = vec![me, beside, ahead_target, ahead_own];
    let mut sim = Simulation::new(&net, vec![]);
    let min_gap = run(&net, &mut sim, &mut cars, 7.0, |_, _, _| {});
    let me = cars.iter().find(|c| c.id == 1).unwrap();
    assert!(me.state.change.is_none(), "never settled: {:?} still {:?} v {:.2} pos {:?} others {:?}", me.state.change, me.change_still, me.state.speed, me.pos, cars.iter().map(|c| (c.id, c.pos.x as i32, c.pos.y)).collect::<Vec<_>>());
    assert_eq!(me.state.lane, 0, "done into the lane where a car stands beside it");
    assert!(min_gap > 0.1, "bodies {min_gap:.2} m apart");
}

#[test]
fn x3_mid_change_the_body_is_registered_where_it_is_on_both_lanes() {
    let (net, mut cars) = frozen_change(0.5);
    let mut sim = Simulation::new(&net, vec![]);
    sim.update_positions(&net, &mut cars);
    sim.build_occupancy(&net, &cars);
    let y = cars[0].pos.y as f32;
    let own = sim.occupancy[&0].iter().find(|o| o.vehicle == 0).unwrap();
    let target = sim.occupancy[&1].iter().find(|o| o.vehicle == 0).unwrap();
    assert!((own.lateral + y).abs() < 0.05 && (target.lateral + y + 3.5).abs() < 0.05, "{:.2} {:.2} at y {y:.2}", own.lateral, target.lateral);
}

#[test]
fn x3_a_change_begun_at_the_tail_of_a_red_queue_does_not_stand_across_the_lines() {
    // viz3 case09: queues stand at a red in both lanes; a car rolls up to the tail of its own
    // lane's queue at 9 m/s, starts over to the other lane 14 m short of it and has to stop
    // part-way behind that lane's queue
    let net = two_lanes();
    let mut cars = Vec::new();
    for k in 0..4u64 {
        let mut a = car(10 + k, 0, 200.0 - 6.5 * k as f32, 0.0);
        a.pull_out = 1000.0;
        cars.push(a);
        let mut b = car(20 + k, 1, 196.0 - 6.5 * k as f32, 0.0);
        b.pull_out = 1000.0;
        cars.push(b);
    }
    let mut me = car(1, 0, 160.0, 9.0);
    me.state.start_change(&net, 1, 2);
    if let Some(c) = me.state.change.as_mut() {
        c.wait = 0.0;
    }
    cars.push(me);
    let mut sim = Simulation::new(&net, vec![]);
    let (mut stood, mut diagonal_after) = (0.0f32, 0.0f32);
    let min_gap = run(&net, &mut sim, &mut cars, 20.0, |_, _, cars| {
        let me = cars.iter().find(|c| c.id == 1).unwrap();
        if me.state.speed < 0.1 {
            stood += DT;
            if stood > 6.0 && me.state.change.is_some() {
                diagonal_after += DT;
            }
        } else {
            stood = 0.0;
        }
    });
    let me = cars.iter().find(|c| c.id == 1).unwrap();
    assert!(stood > 6.0, "it never came to stand ({stood:.1} s)");
    assert_eq!(diagonal_after, 0.0, "still mid-change {diagonal_after:.1} s after 6 s standing");
    assert!(me.state.change.is_none());
    assert!(min_gap > 0.2, "bodies {min_gap:.2} m apart");
}

/// A two-way road: eastbound lane 0 (x 0..238), a 1 m stub (lit by light 0.0 when `lit`) and
/// lane 2 (x 239..600); the westbound lane 3 3 m to the north.
fn two_way(lit: bool) -> Network {
    let mut net = Network::default();
    net.lanes.push(road(&[(0.0, 0.0), (238.0, 0.0)], 1));
    let mut stub = road(&[(238.0, 0.0), (239.0, 0.0)], 2);
    if lit {
        stub.traffic_light = Some((0, 0));
    }
    net.lanes.push(stub);
    net.lanes.push(road(&[(239.0, 0.0), (600.0, 0.0)], 3));
    net.lanes.push(road(&[(600.0, 3.0), (0.0, 3.0)], 4));
    finish(net)
}

fn standing_player(x: f64) -> omsi_traffic::PlayerView {
    omsi_traffic::PlayerView { id: 7, pos: DVec3::new(x, 0.0, 0.0), heading: 90.0, half_len: 6.0, half_width: 1.25, speed: 0.0, still: 30.0, blinker: 0, signalling: 0.0, priority: false }
}

/// The bumper gap at which car 1 first stands behind the standing player bus at x = 200.
fn stands_behind_player(net: &Network, lights: Vec<omsi_sim::traffic::TrafficLightController>) -> f32 {
    let mut sim = Simulation::new(net, lights);
    let player = standing_player(200.0);
    sim.players = vec![player];
    let mut cars = vec![car(1, 0, 120.0, 10.0)];
    let mut gap: Option<f32> = None;
    run(net, &mut sim, &mut cars, 20.0, |_, sim, cars| {
        sim.players = vec![player];
        if gap.is_none() && cars[0].state.speed < 0.05 {
            gap = Some((200.0 - 6.0) - (cars[0].pos.x as f32 + cars[0].state.front));
        }
    });
    gap.expect("it never stood")
}

#[test]
fn x4_behind_a_standing_player_at_a_red_the_car_closes_up_and_keeps_room_only_where_it_can_pass() {
    // a red 38 m past the bus's front: no pass can start, so no room is kept
    let red = omsi_sim::traffic::TrafficLightController::new(vec![vec![(0, 60.0)]], 60.0);
    let at_red = stands_behind_player(&two_way(true), vec![red]);
    assert!(at_red > 1.5 && at_red < 3.0, "behind the player at the red: {at_red:.2} m");
    // no light, the oncoming lane free: room to pull out (out_len 9 m + 1 m)
    let free = stands_behind_player(&two_way(false), vec![]);
    assert!(free > 8.5 && free < 11.5, "behind the player, free to pass: {free:.2} m");
}

/// A car standing for good on lane 0 at x = 200 (nothing holds it: an obstacle after 4 s).
fn broken_down(id: u64, s: f32) -> Vehicle {
    let mut o = car(id, 0, s, 0.0);
    o.stopped = 30.0;
    o.pull_out = 1000.0;
    o
}

#[test]
fn x5_a_pass_signals_the_side_it_moves_to_two_seconds_ahead_and_until_it_is_back() {
    let net = two_way(false);
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![car(1, 0, 150.0, 8.0), broken_down(2, 200.0)];
    let (mut out_at, mut lead_time, mut wrong, mut started, mut back) = (None::<f32>, 0.0f32, 0, false, false);
    let min_gap = run(&net, &mut sim, &mut cars, 40.0, |t, _, cars| {
        let me = cars.iter().find(|c| c.id == 1).unwrap();
        started |= me.passing.is_some();
        if out_at.is_none() && me.state.lateral.abs() > 0.3 {
            out_at = Some(t);
            // the left indicator, shown since
            assert_eq!(me.blinker_shown.0, 1, "pulled out showing {}", me.blinker_shown.0);
            lead_time = t - me.blinker_shown.1 as f32;
        }
        if me.passing.is_some() && me.state.lateral.abs() > 1.0 && me.blinker_shown.0 != 1 {
            wrong += 1;
        }
        back |= out_at.is_some() && me.passing.is_none() && me.state.lateral.abs() < 0.05 && me.pos.x > 215.0;
    });
    assert!(started && out_at.is_some(), "it never passed");
    assert!(lead_time > 1.9, "signalled {lead_time:.2} s before pulling out");
    assert_eq!(wrong, 0, "frames out on the left showing another indicator");
    assert!(back, "never back in its lane past the obstacle");
    assert!(min_gap > 0.3, "bodies {min_gap:.2} m apart");
    let me = cars.iter().find(|c| c.id == 1).unwrap();
    assert_eq!(me.blinker_shown.0, 0, "the indicator is still on");
}

/// A bus serving its stop on lane 0 at x = 200, and a car rolling up behind it.
fn bus_at_stop() -> (Network, Vec<Vehicle>) {
    let net = two_way(false);
    let mut st = AiState::new(0, 200.0, 2);
    st.front = 6.0;
    st.rear = 6.0;
    st.length = 12.0;
    st.veh_type = 2;
    let mut bus = Vehicle::new(2, VehicleClass::ScheduledBus, st, 1.25, 2 * 7919);
    bus.fresh = 0.0;
    bus.external.at_stop = true;
    bus.external.standing_for = 30.0;
    bus.external.stop_at = Some(6.1);
    (net, vec![car(1, 0, 150.0, 8.0), bus])
}

#[test]
fn x5_a_pass_of_a_bus_at_its_stop_is_given_up_when_it_indicates_or_moves_off() {
    // the bus indicates out of its stop (or moves off) while the car signals to pass it
    for moves in [false, true] {
        let (net, mut cars) = bus_at_stop();
        let mut sim = Simulation::new(&net, vec![]);
        let (mut started_at, mut aborted, mut out) = (None::<f32>, false, 0.0f32);
        run(&net, &mut sim, &mut cars, 25.0, |t, _, cars| {
            let me = cars.iter().find(|c| c.id == 1).unwrap();
            if started_at.is_none() && me.passing.is_some() {
                started_at = Some(t);
            }
            // (given up: the pass is over before it pulled out)
            aborted |= started_at.is_some() && me.passing.is_none();
            out = out.max(me.state.lateral.abs());
            let pulled = started_at.is_some_and(|s| t > s + 1.0);
            let bus = cars.iter_mut().find(|c| c.id == 2).unwrap();
            if pulled {
                bus.state.signal = 1;
                bus.state.signal_time = 5.0;
                if moves {
                    bus.external.stop_at = None;
                    bus.external.at_stop = false;
                }
            }
        });
        assert!(started_at.is_some(), "it never set out to pass the bus");
        assert!(aborted, "the pass went on (moves {moves})");
        assert!(out < 0.5, "it pulled out {out:.2} m (moves {moves})");
        let me = cars.iter().find(|c| c.id == 1).unwrap();
        let bus = cars.iter().find(|c| c.id == 2).unwrap();
        assert!(me.passing.is_none(), "still passing");
        assert!(me.pos.x as f32 + me.state.front < bus.pos.x as f32 - bus.state.rear, "not behind the bus");
    }
    // a bus that already indicates is not passed at all
    let (net, mut cars) = bus_at_stop();
    cars[1].state.signal = 1;
    cars[1].state.signal_time = 60.0;
    let mut sim = Simulation::new(&net, vec![]);
    let mut started = false;
    run(&net, &mut sim, &mut cars, 15.0, |_, _, cars| {
        started |= cars.iter().find(|c| c.id == 1).unwrap().passing.is_some();
    });
    assert!(!started, "it set out to pass a bus indicating out of its stop");
}

#[test]
fn x6_no_pass_round_a_standing_queue_with_no_room_to_come_back_in() {
    // viz case09 #30816: standing cars one behind the other (2.5 m apart), nowhere to come back
    // in for 30 m and more: no pass; one car standing alone: passed
    for (n, expect) in [(6u64, false), (1, true)] {
        let net = two_way(false);
        let mut sim = Simulation::new(&net, vec![]);
        let mut cars = vec![car(1, 0, 150.0, 8.0)];
        for k in 0..n {
            cars.push(broken_down(10 + k, 200.0 + 6.5 * k as f32));
        }
        let mut started = false;
        run(&net, &mut sim, &mut cars, 25.0, |_, _, cars| {
            started |= cars.iter().find(|c| c.id == 1).unwrap().passing.is_some();
        });
        assert_eq!(started, expect, "{n} standing");
    }
    // two standing and a gap too short beyond them (12 m: my 4 m, 9 m back in and 4 m need 17)
    let net = two_way(false);
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![car(1, 0, 150.0, 8.0), broken_down(10, 200.0), broken_down(11, 206.5), broken_down(12, 222.5)];
    let mut started = false;
    run(&net, &mut sim, &mut cars, 25.0, |_, _, cars| {
        started |= cars.iter().find(|c| c.id == 1).unwrap().passing.is_some();
    });
    assert!(!started, "passed two with 12 m beyond");
}

#[test]
fn x6_a_pass_comes_back_in_only_where_nobody_is_beside_it() {
    // out on the oncoming lane past what it passed, about to come back in, with a car in its
    // lane drawn level with it at its speed (viz3 case11 #28490 came back in on top of one)
    let net = two_way(false);
    let mut sim = Simulation::new(&net, vec![]);
    let mut me = car(1, 0, 100.0, 8.0);
    me.state.max_speed_kmh = 30.0;
    let odo = me.state.odometer;
    me.passing = Some(omsi_traffic::Passing { lane: Some(3), side: -3.0, until: odo, block: odo - 12.0, out_len: 9.0, back_len: 18.0, aborted: false, hold: 0.0, wait: 0.0, obstacle: None, beside: None });
    me.state.lateral = -3.0;
    me.state.lateral_curve = vec![(odo, -3.0), (odo + 18.0, 0.0)];
    let mut beside = car(2, 0, 99.0, 8.0);
    beside.state.max_speed_kmh = 29.0;
    let mut cars = vec![me, beside];
    let mut back = false;
    let min_gap = run(&net, &mut sim, &mut cars, 25.0, |_, _, cars| {
        let me = cars.iter().find(|c| c.id == 1).unwrap();
        back |= me.passing.is_none() && me.state.lateral.abs() < 0.05;
    });
    assert!(min_gap > 0.2, "came back in on the car beside it: {min_gap:.2} m");
    assert!(back, "never came back in");
}

#[test]
fn x6_a_pass_stranded_out_there_is_given_up_and_the_obstacle_is_the_lead_again() {
    // as the car pulls out round the car standing at x = 200 (its rear at 198), an oncoming car
    // stands head-on in its way (lane 3, facing west, its front at x = 195): the car stops
    // half out, still short of the obstacle, and its way back in past it never opens
    let net = two_way(false);
    let mut sim = Simulation::new(&net, vec![]);
    let mut spare = car(999, 3, 10.0, 0.0);
    spare.pull_out = 1000.0;
    let mut cars = vec![car(1, 0, 150.0, 8.0), broken_down(2, 200.0), spare];
    let (mut out, mut stood_out, mut gave_up_after, mut most) = (None::<f32>, None::<f32>, None::<f32>, 0.0f32);
    let mut placed = false;
    let min_gap = run(&net, &mut sim, &mut cars, 45.0, |t, _, cars| {
        let me = cars.iter().find(|c| c.id == 1).unwrap().clone();
        if out.is_none() && me.state.lateral < -1.5 {
            out = Some(t);
        }
        if out.is_some() && !placed {
            placed = true;
            let mut o = car(3, 3, 600.0 - 197.0, 0.0);
            o.pull_out = 1000.0;
            cars_push(cars, o);
        }
        most = most.max(-me.state.lateral);
        if stood_out.is_none() && out.is_some() && me.state.speed < 0.1 && me.passing.is_some() {
            stood_out = Some(t);
        }
        if stood_out.is_some() && gave_up_after.is_none() && me.passing.is_none() {
            gave_up_after = Some(t - stood_out.unwrap());
        }
    });
    let after = gave_up_after.expect("the stranded pass was never given up");
    assert!(most > 1.0, "it never got out ({most:.2} m)");
    assert!(after > 8.0 && after < 16.0, "given up {after:.1} s after it stood out there");
    let me = cars.iter().find(|c| c.id == 1).unwrap();
    assert!(me.state.lateral.abs() < 0.1 && me.pos.y.abs() < 0.1, "not back in its lane: lateral {:.2}", me.state.lateral);
    assert!(me.pos.x < 196.0, "not behind the obstacle: x {:.1}", me.pos.x);
    assert!(me.last.lead_id == Some(2) || me.last.lead_id == Some(3) || me.last.lead.is_some(), "no lead: {:?}", me.last);
    assert!(min_gap > 0.2, "bodies {min_gap:.2} m apart");
}

/// (a vehicle added during a run: the slice of `run` cannot grow, so it is put in for the
/// first one that is gone; tests that use it keep one spare)
fn cars_push(cars: &mut [Vehicle], v: Vehicle) {
    if let Some(slot) = cars.iter_mut().find(|c| c.id == 999) {
        *slot = v;
    }
}

/// A crossing of object 9: the side road northbound (lanes 0/1/2, the piece 1 along x = 0,
/// y -15..15) gives way to the main road eastbound (lanes 3/4/5, the piece 4 along y = 0,
/// priority 192). Car 1 stands at the side road's line, car 2 at the main road's line with its
/// front `b_front` short of the middle.
fn side_and_main(b_front: f64) -> (Network, Vec<Vehicle>) {
    let mut net = Network::default();
    net.lanes.push(road(&[(0.0, -300.0), (0.0, -15.0)], 1));
    net.lanes.push(piece(&[(0.0, -15.0), (0.0, 15.0)], 0));
    net.lanes.push(road(&[(0.0, 15.0), (0.0, 300.0)], 2));
    net.lanes.push(road(&[(-300.0, 0.0), (-15.0, 0.0)], 3));
    let mut main = piece(&[(-15.0, 0.0), (15.0, 0.0)], 1);
    main.priority = 192.0;
    net.lanes.push(main);
    net.lanes.push(road(&[(15.0, 0.0), (300.0, 0.0)], 4));
    let net = finish(net);
    let a = car(1, 1, 15.0 - 9.0 - 2.0, 0.0);
    let b = car(2, 4, (15.0 - b_front - 2.0) as f32, 0.0);
    (net, vec![a, b])
}

/// The two holding each other, standing, for `secs`, then `Simulation::negotiate` each frame:
/// who eased off. Car `k` (ids 1 and 2) waits for the other at the junction when `wait_for`
/// says so; `blocked` names the cars held by the other's body, for `body_on` s of every
/// `body_on + body_off` s (a flickering lead).
fn held_each_other_as(red: bool, wait_for: &[u64], blocked: &[u64], body_on: f32, body_off: f32, secs: f32) -> Option<(u64, u64, usize, f32)> {
    held_each_other_led(red, wait_for, blocked, &[], body_on, body_off, secs)
}

/// `held_each_other_as` with the cars in `led` following the other as their lead (`lead_id`).
fn held_each_other_led(red: bool, wait_for: &[u64], blocked: &[u64], led: &[u64], body_on: f32, body_off: f32, secs: f32) -> Option<(u64, u64, usize, f32)> {
    let (net, mut cars) = side_and_main(6.0);
    let mut sim = Simulation::new(&net, vec![]);
    sim.tick(&net, DT, &mut cars);
    let claim_of: Vec<Option<usize>> = (0..cars.len()).map(|i| sim.claims.iter().position(|(c, _)| c.vehicle == i)).collect();
    assert!(claim_of.iter().all(|c| c.is_some()), "both claim the junction");
    for (c, _) in sim.claims.iter_mut() {
        c.light_red = red;
    }
    let mut t = 0.0;
    while t < secs {
        let body_now = (t % (body_on + body_off)) < body_on;
        for k in 0..2 {
            let other = cars[1 - k].id;
            let me = &mut cars[k];
            me.state.speed = 0.0;
            me.last.wait_for = wait_for.contains(&me.id).then_some(other);
            me.last.body_block = (blocked.contains(&me.id) && body_now).then_some(other);
            me.last.lead_id = led.contains(&me.id).then_some(other);
        }
        sim.negotiate(&mut cars, &claim_of, DT);
        if let Some(e) = cars.iter().find_map(|c| c.ease_for.map(|(o, j, _)| (c.id, o, j, t))) {
            return Some(e);
        }
        t += DT;
    }
    None
}

/// Both refused for each other by the junction (`wait_for` both ways), `blocked` held by the
/// other's body too.
fn held_each_other(red: bool, blocked: &[u64], secs: f32) -> Option<(u64, u64, usize)> {
    held_each_other_as(red, &[1, 2], blocked, 1.0, 0.0, secs).map(|(a, b, j, _)| (a, b, j))
}

#[test]
fn x7_a_mutual_refusal_is_negotiated_at_once_by_the_pair_s_hash() {
    use omsi_traffic::sim::eases_off;
    // (round 5: at the first frame both stand refused for each other, not after 6 s)
    let (who, other, j) = held_each_other(false, &[], 0.2).expect("nobody eased off");
    assert_eq!(who, eases_off(1, 2, j), "not the hash's pick");
    assert_eq!(other, 3 - who);
    // the same in every run
    assert_eq!(held_each_other(false, &[], 0.2), Some((who, other, j)));
    // of two where only one is held by the other's body, that one eases off (the other waits
    // for it at the junction only)
    for blocked in [1u64, 2] {
        assert_eq!(held_each_other(false, &[blocked], 0.2).map(|e| e.0), Some(blocked));
    }
    // never while either is held by a red light
    assert_eq!(held_each_other(true, &[], 20.0), None, "a negotiation at a red light");
    // of two refused for each other where one follows the other as its lead (nose to tail), the
    // one behind eases off, whatever the hash says (viz case14 #46608/#50444)
    for led in [1u64, 2] {
        assert_eq!(held_each_other_led(false, &[1, 2], &[], &[led], 1.0, 0.0, 0.2).map(|e| e.0), Some(led));
    }
}

#[test]
fn x7_a_pair_held_by_a_body_on_one_side_negotiates_after_the_jittered_hold() {
    use omsi_traffic::sim::{mutual_jitter, MUTUAL_HOLD, MUTUAL_JITTER};
    // car 1 waits for car 2 at the junction; car 2 is held by car 1's body only: no refusal
    // both ways, the clock decides, scaled by the lower id's jitter
    let hold = MUTUAL_HOLD * mutual_jitter(1);
    assert!(hold >= MUTUAL_HOLD * MUTUAL_JITTER.0 && hold <= MUTUAL_HOLD * MUTUAL_JITTER.1);
    assert_eq!(held_each_other_as(false, &[1], &[2], 1.0, 0.0, hold - 0.5), None, "too early");
    let (who, _, _, at) = held_each_other_as(false, &[1], &[2], 1.0, 0.0, hold + 1.0).expect("nobody eased off");
    assert!((at - hold).abs() < 0.2, "eased at {at:.2} s, the hold is {hold:.2} s");
    assert_eq!(who, 2, "the body-blocked one eases off");
    // the body seen for 0.6 s of every second (a flickering lead): the clock runs on over the
    // gaps (MUTUAL_GAP) and it eases off at about the same time
    let (who, _, _, at) = held_each_other_as(false, &[1], &[2], 0.6, 0.4, hold + 2.0).expect("the flicker reset the clock");
    assert!(who == 2 && at < hold + 1.0, "eased at {at:.2} s, who {who}");
    // a gap longer than MUTUAL_GAP resets it: body seen 0.5 s of every 2 s, nothing in 3 holds
    assert_eq!(held_each_other_as(false, &[1], &[2], 0.5, 1.5, 3.0 * hold), None, "a hold gapped 1.5 s at a time eased off");
    // the jitter varies between drivers and stays within its range
    let js: Vec<f32> = (1..200u64).map(mutual_jitter).collect();
    assert!(js.iter().all(|&j| (MUTUAL_JITTER.0..=MUTUAL_JITTER.1).contains(&j)));
    assert!(js.iter().any(|&j| j < 0.8) && js.iter().any(|&j| j > 1.2));
}

/// The side road car (1) and the main road car (2, eased off for it, `b_front` short of the
/// middle) for `secs`: (the least gap between the bodies, car 1's front y at the end, car 2's
/// ease at the end).
fn eased_main(b_front: f64, secs: f32, eased: bool) -> (f32, f32, Option<(u64, usize, f32)>, f32) {
    let (net, mut cars) = side_and_main(b_front);
    if eased {
        cars[1].ease_for = Some((1, 0, 0.0));
    }
    let mut sim = Simulation::new(&net, vec![]);
    // (how far the main road car had moved by the time the side road car was across)
    let mut moved_before = 0.0f32;
    let start = -(b_front as f32) - 2.0;
    let min_gap = run(&net, &mut sim, &mut cars, secs, |_, _, cars| {
        let a = cars.iter().find(|c| c.id == 1).unwrap();
        let b = cars.iter().find(|c| c.id == 2).unwrap();
        if a.pos.y < 3.0 {
            moved_before = moved_before.max(b.pos.x as f32 - start);
        }
    });
    let a = cars.iter().find(|c| c.id == 1).unwrap();
    (min_gap, a.pos.y as f32 + a.state.front, cars.iter().find(|c| c.id == 2).unwrap().ease_for, moved_before)
}

#[test]
fn x7_the_one_eased_off_waits_and_the_other_goes_keeping_off_its_body() {
    // the main road car 6 m short of the middle, eased off: the side road car, which would
    // give way to it, goes; the main road car stays until it is through, then takes part again
    let (_, _, _, moved) = eased_main(6.0, 12.0, false);
    assert!(moved > 3.0, "not eased off, the main road car goes first: it moved {moved:.1} m");
    let (min_gap, a_front, ease, moved) = eased_main(6.0, 12.0, true);
    assert!(a_front > 10.0, "the side road car did not go: front at y = {a_front:.1}");
    assert!(moved < 0.3, "the main road car moved {moved:.1} m before the side road car was across");
    assert!(min_gap > 0.3, "bodies {min_gap:.2} m apart");
    assert!(ease.is_none(), "still eased off once the other has left: {ease:?}");
    // with its nose in the side road car's corridor (its front 0.7 m short of the side road's
    // middle line, 0.2 m inside the corridor): the side road car is let go by the junction
    // but not by the body, and stops short of it
    let (min_gap, a_front, _, _) = eased_main(0.7, 12.0, true);
    assert!(min_gap > 0.2, "it drove into the body: {min_gap:.2} m");
    assert!(a_front < -0.9, "front at y = {a_front:.1}, past the body's edge");
}

#[test]
fn x7_who_eases_off_is_a_hash_of_the_pair_and_the_junction() {
    use omsi_traffic::sim::eases_off;
    for (a, b, j) in [(1u64, 2u64, 0usize), (15110, 10813, 6), (7, 900, 31)] {
        let e = eases_off(a, b, j);
        assert!(e == a || e == b);
        assert_eq!(e, eases_off(b, a, j), "not symmetric");
        assert_eq!(e, eases_off(a, b, j), "not the same twice");
    }
    // (both sides come up)
    let picks: Vec<bool> = (0..64usize).map(|j| eases_off(3, 4, j) == 3).collect();
    assert!(picks.iter().any(|&p| p) && picks.iter().any(|&p| !p));
}

/// X2: a player's bus beside an AI car on lanes `apart` m apart cuts in: from abreast (the
/// bus's rear `behind` m behind the car's front), it turns `angle` degrees into the car's lane
/// over 1.5 s at 7 m/s, the car doing 8 m/s.
#[test]
fn r11_x2_player_bus_beside_cuts_in() {
    let mut rows = Vec::new();
    let mut worst = f32::INFINITY;
    for apart in [2.4f64, 3.0, 3.5] {
        for behind in [0.0f32, 1.0, 2.0, 4.0] {
            for angle in [6.0f64, 10.0, 15.0] {
                let mut net = Network::default();
                let mut a = road(&[(0.0, 0.0), (600.0, 0.0)], 1);
                a.offset = 0.0;
                let mut b = road(&[(0.0, -apart), (600.0, -apart)], 1);
                b.key = Some(LaneKey { tile: (0, 0), id: 1, path: 1 });
                b.offset = apart as f32;
                net.lanes.push(a);
                net.lanes.push(b);
                let net = finish(net);
                let mut sim = Simulation::new(&net, vec![]);
                let mut me = car(1, 0, 100.0, 8.0);
                me.state.max_speed_kmh = 29.0;
                let mut cars = vec![me];
                // the bus: centre x so that its rear is `behind` m behind the car's front (102)
                let mut bx = 102.0 - behind as f64 + 6.0;
                let mut by = -apart;
                let mut h = 90.0f64;
                let mut least = f32::INFINITY;
                let mut t_in = None;
                let mut at_contact: Option<String> = None;
                let mut v_after = 0.0f32;
                let mut lost = 0;
                run(&net, &mut sim, &mut cars, 8.0, |t, sim, cars| {
                    let v = 7.0f64;
                    if t > 0.5 && h > 90.0 - angle {
                        h -= angle * DT as f64 / 1.5;
                    }
                    // straighten once over the line
                    if by > -0.3 && h < 90.0 {
                        h = (h + 20.0 * DT as f64).min(90.0);
                    }
                    let hr = h.to_radians();
                    bx += hr.sin() * v * DT as f64;
                    by += hr.cos() * v * DT as f64;
                    let pl = PlayerView { id: 7, pos: DVec3::new(bx, by, 0.0), heading: h as f32, half_len: 6.0, half_width: 1.25, speed: 7.0, still: 0.0, blinker: 1, signalling: 3.0 + t, priority: false };
                    let s = pl.obb(0.0, 0.0).separation(&cars[0].obb()) as f32;
                    if s < least {
                        least = s;
                    }
                    if s < 0.0 && at_contact.is_none() {
                        at_contact = Some(format!("t {t:.2} AI v {:.1} lead {:?} why {:?}", cars[0].state.speed, cars[0].last.lead.map(|l| (l.gap * 10.0).round() / 10.0), cars[0].last.why));
                    }
                    if at_contact.is_some() {
                        v_after = v_after.max(cars[0].state.speed);
                        if cars[0].last.lead_seen.is_none() {
                            lost += 1;
                        }
                    }
                    if t_in.is_none() && cars[0].last.lead_seen.is_some() {
                        t_in = Some(t);
                    }
                    sim.players = vec![pl];
                });
                rows.push(format!("apart {apart} behind {behind} angle {angle}: least {least:.2}, AI saw a lead from {t_in:?}; first contact {at_contact:?}, AI top speed after contact {v_after:.1}, frames after contact with no lead {lost}"));
                worst = worst.min(least);
            }
        }
    }
    eprintln!("X2 player bus cuts in:\n  {}", rows.join("\n  "));
    assert!(worst > 0.0, "the AI car drove into the bus cutting in: {worst:.2}");
}

/// A car (an AI car, or the player's bus) driving beside me in the next lane, parallel, at my
/// speed or a little faster: no lead, I am not held.
#[test]
fn x2_a_parallel_vehicle_beside_me_does_not_hold_me() {
    for apart in [2.6f64, 3.0, 3.5] {
        for player in [false, true] {
            let mut net = Network::default();
            let mut a = road(&[(0.0, 0.0), (600.0, 0.0)], 1);
            a.offset = 0.0;
            let mut b = road(&[(0.0, -apart), (600.0, -apart)], 1);
            b.key = Some(LaneKey { tile: (0, 0), id: 1, path: 1 });
            b.offset = apart as f32;
            net.lanes.push(a);
            net.lanes.push(b);
            let net = finish(net);
            let mut sim = Simulation::new(&net, vec![]);
            let mut me = car(1, 0, 100.0, 8.0);
            me.state.max_speed_kmh = 29.0;
            let mut cars = vec![me];
            if !player {
                let mut o = car(2, 1, 103.0, 8.5);
                o.state.max_speed_kmh = 31.0;
                cars.push(o);
            }
            let mut bx = 106.0f64;
            let mut slowest = f32::INFINITY;
            let mut held = 0;
            run(&net, &mut sim, &mut cars, 6.0, |t, sim, cars| {
                bx += 8.5 * DT as f64;
                if player {
                    sim.players = vec![omsi_traffic::PlayerView { id: 7, pos: DVec3::new(bx, -apart, 0.0), heading: 90.0, half_len: 6.0, half_width: 1.25, speed: 8.5, still: 0.0, blinker: 0, signalling: 0.0, priority: false }];
                }
                let me = cars.iter().find(|c| c.id == 1).unwrap();
                if t > 0.5 {
                    slowest = slowest.min(me.state.speed);
                    held += me.last.lead.is_some_and(|l| l.gap < 3.0) as i32;
                }
            });
            assert!(held == 0 && slowest > 7.5, "apart {apart} player {player}: held {held} frames, slowest {slowest:.1}");
        }
    }
}

/// An AI car beside me in the next lane, a little ahead, changes into my lane (no gap for it):
/// it is moving into my corridor, I drop back and do not drive into it.
#[test]
fn x2_an_ai_car_beside_me_cutting_in_is_let_in() {
    for (apart, ahead) in [(3.0f64, 2.0f32), (3.5, 0.5), (3.5, 3.0)] {
        let mut net = Network::default();
        let mut a = road(&[(0.0, 0.0), (600.0, 0.0)], 1);
        a.offset = 0.0;
        let mut b = road(&[(0.0, -apart), (600.0, -apart)], 1);
        b.key = Some(LaneKey { tile: (0, 0), id: 1, path: 1 });
        b.offset = apart as f32;
        net.lanes.push(a);
        net.lanes.push(b);
        let net = finish(net);
        let mut sim = Simulation::new(&net, vec![]);
        let mut me = car(1, 0, 100.0, 8.0);
        me.state.max_speed_kmh = 29.0;
        let mut other = car(2, 1, 100.0 + ahead, 7.0);
        other.state.max_speed_kmh = 25.2;
        other.state.start_change(&net, 0, 1);
        if let Some(c) = other.state.change.as_mut() {
            c.wait = 0.0;
        }
        let mut cars = vec![me, other];
        let min_gap = run(&net, &mut sim, &mut cars, 8.0, |_, _, _| {});
        assert!(min_gap > 0.0, "apart {apart} ahead {ahead}: drove into it, {min_gap:.2} m");
    }
}

/// A car (id 1) at `s` on lane 0, out `lat` m on a pass (on the oncoming lane 3), standing.
fn out_on_a_pass(s: f32, lat: f32, aborted: bool, until: f32, obstacle: Option<u64>) -> Vehicle {
    let mut me = car(1, 0, s, 0.0);
    let odo = me.state.odometer;
    me.passing = Some(omsi_traffic::Passing { lane: Some(3), side: -3.0, until: odo + until, block: odo - 4.0, out_len: 9.0, back_len: 18.0, aborted, hold: odo, wait: 0.0, obstacle, beside: None });
    me.state.lateral = lat;
    me.state.lateral_target = lat;
    me.state.lateral_ramp = (lat, lat, 0.0, 1.0);
    me
}

#[test]
fn x6_a_follower_does_not_close_up_on_a_given_up_pass_so_it_can_step_back() {
    // review12, case15: #47238 gave up its pass 2.1 m out; #47012 closed up to 0.23 m behind its
    // rear and it never got the 0.2 m it needs to step back
    let net = two_way(false);
    let mut sim = Simulation::new(&net, vec![]);
    let mut follower = car(3, 0, 80.0, 5.0);
    follower.state.max_speed_kmh = 30.0;
    let mut cars = vec![out_on_a_pass(100.0, -2.1, true, 20.0, Some(2)), broken_down(2, 106.5), follower];
    let (mut least_behind, mut back, mut least_lat) = (f32::INFINITY, false, f32::INFINITY);
    let min_gap = run(&net, &mut sim, &mut cars, 20.0, |_, _, cars| {
        let me = cars.iter().find(|c| c.id == 1).unwrap();
        let f = cars.iter().find(|c| c.id == 3).unwrap();
        // (while it stands out there; back in, the follower keeps its own gap)
        if me.passing.is_some_and(|p| p.aborted) {
            least_behind = least_behind.min((me.pos.x as f32 - me.state.rear) - (f.pos.x as f32 + f.state.front));
        }
        // (back in, the pass over; it may set out on a new one later)
        back |= me.passing.is_none() && me.state.lateral.abs() < 0.1;
        least_lat = least_lat.min(me.state.lateral.abs());
    });
    assert!(least_behind >= 2.0, "the follower closed up to {least_behind:.2} m behind its rear");
    assert!(back, "never stepped back: lateral down to {least_lat:.2}");
    assert!(min_gap > 0.2, "bodies {min_gap:.2} m apart");
}

#[test]
fn x6_a_given_up_pass_beside_its_obstacle_drives_on_once_the_obstacle_leaves() {
    // abreast of the car it went round (it cannot step back into it); at 3 s that car moves
    // off: the pass is over, the car drives on and eases back into its lane ahead
    let net = two_way(false);
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![out_on_a_pass(106.0, -3.0, true, 10.0, Some(2)), broken_down(2, 105.0)];
    let mut gone_at: Option<f32> = None;
    let mut over_at: Option<f32> = None;
    let (mut turned, mut last_heading) = (0.0f32, 90.0f32);
    let min_gap = run(&net, &mut sim, &mut cars, 25.0, |t, _, cars| {
        if t > 3.0 {
            cars.iter_mut().find(|c| c.id == 2).unwrap().pull_out = 0.0;
        }
        let o = cars.iter().find(|c| c.id == 2).unwrap().state.speed;
        if gone_at.is_none() && o > 1.0 {
            gone_at = Some(t);
        }
        let me = cars.iter().find(|c| c.id == 1).unwrap();
        if over_at.is_none() && !me.passing.is_some_and(|p| p.aborted) {
            over_at = Some(t);
        }
        // (turning where it stands: the review12 glitch, 20-30 degrees)
        if me.state.speed < 0.1 {
            turned += (me.heading - last_heading).abs();
        }
        last_heading = me.heading;
    });
    let (gone, over) = (gone_at.expect("the obstacle never left"), over_at.expect("the pass was never over"));
    assert!(over - gone < 1.0, "the pass was over {:.1} s after the obstacle left", over - gone);
    let me = cars.iter().find(|c| c.id == 1).unwrap();
    assert!(me.state.lateral.abs() < 0.1 && me.pos.x > 120.0, "not on in its lane: lateral {:.2} x {:.1}", me.state.lateral, me.pos.x);
    assert!(turned < 3.0, "the body turned {turned:.1} degrees standing");
    // (it stepped back to 0.2 m beside it; no nearer as they move off)
    assert!(min_gap > 0.18, "bodies {min_gap:.2} m apart");
}

#[test]
fn x6_a_pass_held_out_there_by_a_red_light_is_given_up() {
    // past what it passed, still out 3 m, standing at the red 13 m on (review12: #20946 stood
    // out on a scheduled bus's lane at a red 84 s): given up after 8 s, it steps back in
    let net = two_way(true);
    let red = omsi_sim::traffic::TrafficLightController::new(vec![vec![(0, 120.0)]], 120.0);
    let mut sim = Simulation::new(&net, vec![red]);
    let mut cars = vec![out_on_a_pass(223.0, -3.0, false, -2.0, None)];
    let (mut back_at, mut stood_at): (Option<f32>, Option<f32>) = (None, None);
    run(&net, &mut sim, &mut cars, 25.0, |t, _, cars| {
        let me = &cars[0];
        if stood_at.is_none() && me.state.speed < 0.1 && t > 1.0 {
            stood_at = Some(t);
        }
        if back_at.is_none() && me.passing.is_none() && me.state.lateral.abs() < 0.1 {
            back_at = Some(t);
        }
    });
    let at = back_at.expect("still out on the pass at the red") - stood_at.expect("never stood at the red");
    // (8 s stranded, then 3 m back at a walking pace)
    assert!(at > 8.0 && at < 16.0, "back in {at:.1} s after it stood");
    assert!(cars[0].pos.x < 238.0, "ran the red");
}

#[test]
fn x3_a_frozen_change_with_both_lanes_taken_beside_it_waits_where_it_is() {
    // lanes 4.5 m apart; the change 53 % across, a car standing level with it in each lane:
    // neither done nor given up (either puts its body into a car), it stays where it is; once
    // the one in its own lane (its nose 1 m past its rear) has driven off, it is given up into
    // that lane
    let mut net = Network::default();
    let a = road(&[(0.0, 0.0), (300.0, 0.0)], 1);
    let mut b = road(&[(0.0, -4.5), (300.0, -4.5)], 1);
    b.key = Some(LaneKey { tile: (0, 0), id: 1, path: 1 });
    b.offset = 4.5;
    net.lanes.push(a);
    net.lanes.push(b);
    let net = finish(net);
    let mut me = car(1, 0, 100.0, 0.0);
    me.state.change = Some(omsi_sim::traffic::LaneChange { to: 1, t: 0.53, length: 12.0, s_to: 100.0, dir: 2, wait: 0.0, bypass: false, full_lock: false });
    let mut cars = vec![me];
    for (id, lane, s) in [(2u64, 1usize, 97.0f32), (3, 0, 103.0), (4, 1, 105.5)] {
        let mut o = car(id, lane, s, 0.0);
        o.pull_out = 1000.0;
        cars.push(o);
    }
    let mut sim = Simulation::new(&net, vec![]);
    sim.update_positions(&net, &mut cars);
    let start = cars[0].pos;
    let (mut moved, mut waiting_at_10) = (0.0f64, false);
    let min_gap = run(&net, &mut sim, &mut cars, 20.0, |t, _, cars| {
        let me = cars.iter().find(|c| c.id == 1).unwrap();
        if t < 10.0 {
            moved = moved.max((me.pos - start).truncate().length());
            waiting_at_10 = me.state.change.is_some();
        }
        // (at 10 s the car beside it in its own lane drives off)
        if t > 10.0 {
            cars.iter_mut().find(|c| c.id == 3).unwrap().pull_out = 0.0;
        }
    });
    assert!(waiting_at_10, "settled into a lane where a car stands beside it");
    assert!(moved < 0.05, "the body moved {moved:.2} m while it waited");
    let me = cars.iter().find(|c| c.id == 1).unwrap();
    assert!(me.state.change.is_none() && me.state.lane == 0, "not given up into its own lane once free: {:?} lane {}", me.state.change, me.state.lane);
    assert!(min_gap > 0.1, "bodies {min_gap:.2} m apart");
}

/// The eastbound two-way road of `two_way(false)` with a crossing of object 9 at x = `jx`: the
/// eastbound and westbound roads run through it as pieces (x `jx` - 10..`jx` + 10), a
/// northbound main road (priority 192) crosses them. Lanes: 0 eastbound to the crossing, 3
/// westbound beyond it, 4 westbound after it.
fn two_way_with_crossing(jx: f64) -> Network {
    let mut net = Network::default();
    net.lanes.push(road(&[(0.0, 0.0), (jx - 10.0, 0.0)], 1)); // 0
    net.lanes.push(piece(&[(jx - 10.0, 0.0), (jx + 10.0, 0.0)], 0)); // 1
    net.lanes.push(road(&[(jx + 10.0, 0.0), (600.0, 0.0)], 2)); // 2
    net.lanes.push(road(&[(600.0, 3.0), (jx + 10.0, 3.0)], 3)); // 3
    net.lanes.push(road(&[(jx - 10.0, 3.0), (0.0, 3.0)], 4)); // 4
    net.lanes.push(piece(&[(jx + 10.0, 3.0), (jx - 10.0, 3.0)], 1)); // 5
    net.lanes.push(road(&[(jx, -300.0), (jx, -15.0)], 5)); // 6
    let mut main = piece(&[(jx, -15.0), (jx, 15.0)], 2);
    main.priority = 192.0;
    net.lanes.push(main); // 7
    net.lanes.push(road(&[(jx, 15.0), (jx, 300.0)], 6)); // 8
    finish(net)
}

/// The bumper gap at which car 1, coming at 10 m/s, first stands behind a car standing for
/// good at x = 200 (rear at 198), the oncoming lane `oncoming`'s traffic given.
fn stands_behind_broken_down(net: &Network, oncoming: &[Vehicle]) -> f32 {
    let mut sim = Simulation::new(net, vec![]);
    let mut cars = vec![car(1, 0, 120.0, 10.0), broken_down(2, 200.0)];
    cars.extend(oncoming.iter().cloned());
    let mut gap: Option<f32> = None;
    run(net, &mut sim, &mut cars, 20.0, |t, _, cars| {
        let me = cars.iter().find(|c| c.id == 1).unwrap();
        if gap.is_none() && t > 1.0 && me.state.speed < 0.05 {
            gap = Some(198.0 - (me.pos.x as f32 + me.state.front));
        }
    });
    gap.expect("it never stood")
}

#[test]
fn x4_no_room_is_kept_where_no_pass_can_start_a_junction_near_or_the_oncoming_lane_taken() {
    // free to pass: room to pull out again (out_len 9 m + 1 m)
    let free = stands_behind_broken_down(&two_way(false), &[]);
    assert!(free > 8.5 && free < 11.5, "free to pass: {free:.2} m");
    // a crossing 18 m past the standing car's front (within `JUNCTION_KEEP` of mine): no
    // pass can start, it closes up
    let net = two_way_with_crossing(220.0);
    let js = omsi_traffic::Junctions::compile(&net);
    assert!(js.movement(1).is_some(), "the crossing is a junction");
    let near_junction = stands_behind_broken_down(&net, &[]);
    assert!(near_junction > 1.0 && near_junction < 3.5, "a junction ahead: {near_junction:.2} m");
    // the oncoming lane taken by a stream (a car every 25 m at 12 m/s): it closes up
    let stream: Vec<Vehicle> = (0..14u64).map(|k| car(20 + k, 3, 150.0 + 25.0 * k as f32, 12.0)).collect();
    let taken = stands_behind_broken_down(&two_way(false), &stream);
    assert!(taken > 1.0 && taken < 3.5, "the oncoming lane taken: {taken:.2} m");
}

/// The side road car (1) standing for good at its line, the main road car (2) at its line
/// eased off for it from the first frame both claim the crossing, run through the whole
/// simulation for `secs` (the main road's light `light` given: on the main road's piece, its
/// line at x = -15): when car 2's ease ended, how far car 2 had moved by then (from where it
/// was eased off) and by the end, and the least gap between the bodies.
fn eased_for_a_car_that_never_goes(light: Option<omsi_sim::traffic::TrafficLightController>, secs: f32) -> (Option<f32>, f32, f32, f32) {
    let mut net = Network::default();
    net.lanes.push(road(&[(0.0, -300.0), (0.0, -15.0)], 1));
    net.lanes.push(piece(&[(0.0, -15.0), (0.0, 15.0)], 0));
    net.lanes.push(road(&[(0.0, 15.0), (0.0, 300.0)], 2));
    net.lanes.push(road(&[(-300.0, 0.0), (-15.0, 0.0)], 3));
    let mut main = piece(&[(-15.0, 0.0), (15.0, 0.0)], 1);
    main.priority = 192.0;
    if light.is_some() {
        main.traffic_light = Some((0, 0));
    }
    net.lanes.push(main);
    net.lanes.push(road(&[(15.0, 0.0), (300.0, 0.0)], 4));
    let net = finish(net);
    let mut a = car(1, 0, 285.0 - 2.5, 0.0);
    a.pull_out = 1000.0;
    let b = car(2, 3, 285.0 - 2.5, 0.0);
    let mut cars = vec![a, b];
    let mut sim = Simulation::new(&net, light.into_iter().collect());
    let (mut eased_at, mut ended, mut moved_eased) = (None::<f32>, None::<f32>, 0.0f32);
    let min_gap = run(&net, &mut sim, &mut cars, secs, |t, sim, cars| {
        if eased_at.is_none() && sim.claims.len() == 2 {
            let b = cars.iter_mut().find(|c| c.id == 2).unwrap();
            b.ease_for = Some((1, 0, 0.0));
            eased_at = Some(b.pos.x as f32);
            return;
        }
        let Some(start) = eased_at else { return };
        let b = cars.iter().find(|c| c.id == 2).unwrap();
        if ended.is_none() && b.ease_for.is_none() {
            ended = Some(t);
        }
        if ended.is_none() {
            moved_eased = moved_eased.max(b.pos.x as f32 - start);
        }
    });
    let start = eased_at.expect("the two never both claimed the crossing");
    let b = cars.iter().find(|c| c.id == 2).unwrap();
    (ended, moved_eased, b.pos.x as f32 - start, min_gap)
}

#[test]
fn x7_an_ease_off_ends_after_thirty_seconds_or_at_a_red_light_end_to_end() {
    use omsi_traffic::sim::EASE_MAX;
    // the one it eased off for never goes: after `EASE_MAX` the ease ends and it goes (it has
    // the right of way)
    let (ended, moved_eased, moved, min_gap) = eased_for_a_car_that_never_goes(None, EASE_MAX + 8.0);
    let ended = ended.expect("the ease never ended");
    assert!((ended - EASE_MAX).abs() < 1.0, "the ease ended at {ended:.1} s");
    assert!(moved_eased < 0.3, "it moved {moved_eased:.1} m while eased off");
    assert!(moved > 5.0, "it did not go once the ease ended: {moved:.1} m");
    assert!(min_gap > 0.3, "bodies {min_gap:.2} m apart");
    // its light goes red at 4 s: the ease ends there (the light holds it now), well before
    // `EASE_MAX`, and it does not move
    let red_at_4 = omsi_sim::traffic::TrafficLightController::new(vec![vec![(6, 4.0), (0, 100.0)]], 104.0);
    let (ended, moved_eased, moved, _) = eased_for_a_car_that_never_goes(Some(red_at_4), 12.0);
    let ended = ended.expect("the ease did not end at the red");
    assert!(ended > 3.5 && ended < 5.0, "the ease ended at {ended:.1} s");
    assert!(moved_eased < 0.3 && moved < 0.3, "it moved {moved:.1} m");
}
