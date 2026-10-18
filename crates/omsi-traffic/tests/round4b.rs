//! Headless scenarios for round 4b: converging lanes of one object compiled as meeting
//! places (X1), leads by the body's real place (X2), frozen lane changes (X3), the keep-back
//! only where a pass can start (X4), passes that signal and give way to a bus pulling out
//! (X5), passes that need room past the obstacle (X6), and the negotiation of a prolonged
//! mutual hold (X7).

use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network};
use omsi_traffic::junction::FoeKind;
#[allow(unused_imports)]
use omsi_traffic::{Junctions, Simulation, Vehicle, VehicleClass};

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
    // compiling them again (a recording's header) adds nothing
    let mut again = converging(0);
    assert_eq!(again.compile_converging(), 0);
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
    assert_eq!(old.compile_converging(), 1);
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
    changer.state.change = Some(omsi_sim::traffic::LaneChange { to: 1, t: 0.85, length: 12.0, s_to: 100.0, dir: 2, wait: 0.0, bypass: false });
    let mut cars = vec![follower, changer];
    let p = seen(&net, &mut cars, 0);
    assert!((cars[1].pos.y + 3.3).abs() < 0.2, "the changer stands at y = {:.2}", cars[1].pos.y);
    assert!(p.leader.is_none(), "a phantom lead: {:?}", p.leader);
    // counter-case: one 30 % over is still in my corridor
    cars[1].state.change = Some(omsi_sim::traffic::LaneChange { to: 1, t: 0.3, length: 12.0, s_to: 100.0, dir: 2, wait: 0.0, bypass: false });
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
    follower.lead_id = Some(1);
    let mut cars = vec![car(1, 0, 40.0, 3.0), follower];
    let p = seen(&net, &mut cars, 0);
    assert!(p.leader.is_none(), "{:?}", p.leader);
    // counter-case: not following me, it is in my way
    cars[1].lead_id = None;
    let p = seen(&net, &mut cars, 0);
    assert!(p.leader.is_some_and(|l| l.gap > 20.0), "{:?}", p.leader);
}
