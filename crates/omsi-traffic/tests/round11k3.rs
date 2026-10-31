//! Swing-aware swept places: the swept-path meeting places per pair of body classes
//! (`omsi_sim::traffic::SWEPT_CLASSES`), from the bodies' swept bands on the careful driver's line
//! (`SweptBody::sides`), so a car is not held where only a bus's swing reaches
//! (`Junctions::for_bodies`); crossings and side merges stay one stretch for every class.
use glam::DVec3;
use omsi_sim::traffic::{LaneBuilder, LaneKey, LaneKind, Network, SWEPT_ARTICULATED, SWEPT_CAR, SWEPT_RIGID};
use omsi_traffic::reservation::{resolve, Claim};
use omsi_traffic::{Impatience, Junctions};

/// A lane of crossing object 9 (`source` 2) or of a spline (`source` 1), 2.5 m wide.
fn keyed(mut l: omsi_sim::traffic::Lane, source: u32, id: i64, path: u16) -> omsi_sim::traffic::Lane {
    l.source = source;
    l.key = Some(LaneKey { tile: (0, 0), id, path });
    l
}

fn line(from: (f64, f64), to: (f64, f64)) -> omsi_sim::traffic::Lane {
    LaneBuilder::polyline(vec![DVec3::new(from.0, from.1, 0.0), DVec3::new(to.0, to.1, 0.0)], LaneKind::Street, 2.5)
}

/// A right turn of radius `r` from `start` heading north (a quarter circle).
fn turn(start: (f64, f64), r: f64) -> omsi_sim::traffic::Lane {
    LaneBuilder::arc(DVec3::new(start.0, start.1, 0.0), 0.0, r * std::f64::consts::FRAC_PI_2, r, 0.0, LaneKind::Street, 2.5)
}

/// A claim of a standing vehicle of body class `body` on movement `m`, its origin at `s`.
fn standing(js: &Junctions, lane: usize, vehicle: usize, s: f32, body: usize) -> Claim {
    let (j, m) = js.of_lane[&lane];
    Claim { foe_only: false, pass_speed: f32::INFINITY, half_width: omsi_sim::traffic::SWEPT_CLASSES[body].half_width, chain: vec![(m, 0.0)], vehicle, junction: j, movement: m, s, speed: 0.0, accel: 1.8, v_max: 13.0, front: 2.0, rear: 2.0, decel: 2.0, waiting: 0.0, lead_gap: None, zip_behind: None, refused_last: false, body_block: None, next_wait: f32::NEG_INFINITY, tau: 1.3, min_gap: 2.0, committed: false, exit_room: true, impatience: Impatience::default(), absolute: false, held: false, light_red: false, until_go: None, red_at: f32::INFINITY, lead: None, was_standing: true, blocked: false, stuck: false, clearing: false, in_box_at_turn: false, lanes: Vec::new(), light: None, id: vehicle as u64, yields_now: None, clear_ahead: f32::INFINITY, serving: false, brake: 5.6, body }
}

/// The turn (lane 0, piece of object 9: R 15 right from (0, 0) north, fed by lane 1) and a spline
/// lane (2) whose end lies 3.6 m outside the turn at 45 degrees, tangent to it, leading into a
/// piece of the object (3) that runs on straight, 30 m (its last body lengths beyond any swing:
/// the place is lane 2's own, not carried by the way on). Lanes 2.5 m wide: their vehicles have 2.5 m side by side; a
/// car's swing out of the turn (0.41 m) does not reach lane 2's end, a rigid bus's front (1.36 m,
/// BJ6123's reach) does.
fn swept_turn() -> (Network, Junctions) {
    swept_turn_lit(false)
}

/// `swept_turn`, the piece lane 2 leads into (3) lit by light 0 of controller 0 when `lit`.
fn swept_turn_lit(lit: bool) -> (Network, Junctions) {
    let r = 15.0f64;
    let (c, k) = ((r, 0.0), std::f64::consts::FRAC_1_SQRT_2);
    let end = (c.0 - (r + 3.6) * k, c.1 + (r + 3.6) * k);
    let mut net = Network::default();
    net.lanes.push(keyed(turn((0.0, 0.0), r), 2, 9, 0));
    net.lanes.push(keyed(line((0.0, -30.0), (0.0, 0.0)), 1, 1, 0));
    net.lanes.push(keyed(line((end.0 - 20.0 * k, end.1 - 20.0 * k), end), 1, 2, 0));
    net.lanes.push(keyed(line(end, (end.0 + 30.0 * k, end.1 + 30.0 * k)), 2, 9, 1));
    if lit {
        net.lanes[3].traffic_light = Some((0, 0));
    }
    net.link(1.5);
    net.compute_conflicts();
    let js = Junctions::compile(&net);
    (net, js)
}

/// The swept place of the turn over lane 2's end is a bus's and not a car's: a car standing at the
/// end of lane 2 is not in the way of a car turning (no meeting place for the pair), it is of a
/// rigid or articulated bus turning; and lane 2 is a movement of the object (N5: its line before
/// the place).
#[test]
fn k3_the_swept_place_of_a_turn_is_the_turning_bodys() {
    let (net, js) = swept_turn();
    let c = net.crossings[0].iter().find(|c| c.other == 2).expect("the turn's swept place over lane 2's end");
    assert!(c.swept && c.classes.is_some(), "{c:?}");
    let (j, _) = *js.of_lane.get(&2).expect("lane 2 is a movement of the object");
    let foe = js.foes_along(j, &[(js.of_lane[&0].1, 0.0)]).into_iter().find(|f| js.nodes[j].movements[f.movement].lane == 2).expect("a foe in the table");
    assert!(js.for_bodies(j, &foe, SWEPT_CAR, Some(SWEPT_CAR)).is_none(), "two cars: no meeting place");
    for bus in [SWEPT_RIGID, SWEPT_ARTICULATED] {
        let f = js.for_bodies(j, &foe, bus, Some(SWEPT_CAR)).unwrap_or_else(|| panic!("class {bus} turning meets a car at lane 2's end"));
        assert!(f.theirs.1 > net.lanes[2].length() - 1.0, "at lane 2's end: {f:?}");
    }
    // (lane 2's line lies before the place: the first meeting place of a car there is the bus's)
    let mine = js.foes_along_for(j, &[(js.of_lane[&2].1, 0.0)], SWEPT_CAR, omsi_traffic::junction::EVERY_BODY);
    assert!(mine.iter().any(|f| f.mine.0 < net.lanes[2].length() - 1.0), "{mine:?}");
}

/// A car is not held by a bus-only swept place, a bus is: a car standing at the end of lane 2 is
/// refused for a bus standing in the turn's meeting place, and goes for a car standing there; a bus
/// claiming the turn is refused for the car standing at lane 2's end, a car claiming it goes.
#[test]
fn k3_a_car_is_not_held_by_a_bus_only_swept_place_and_a_bus_is() {
    let (net, js) = swept_turn();
    let (j, m2) = js.of_lane[&2];
    let m0 = js.of_lane[&0].1;
    let foe = js.nodes[j].movements[m2].foes.iter().find(|f| f.movement == m0).copied().expect("lane 2 meets the turn");
    let pair = js.for_bodies(j, &foe, SWEPT_CAR, Some(SWEPT_RIGID)).expect("a car at lane 2's end meets a bus turning");
    // the car standing short of the stretch, the turning vehicle in the middle of its own
    let me_s = pair.mine.0 - 2.0 - 0.5;
    let their_s = (pair.theirs.0 + pair.theirs.1) * 0.5;
    for (body, held) in [(SWEPT_RIGID, true), (SWEPT_ARTICULATED, true), (SWEPT_CAR, false)] {
        let mut other = standing(&js, 0, 2, their_s, body);
        other.held = true;
        let g = resolve(&js, &[standing(&js, 2, 1, me_s, SWEPT_CAR), other]);
        let g = g.iter().find(|g| g.vehicle == 1).unwrap();
        assert_eq!(!g.go && g.wait_for == Some(2), held, "car at lane 2's end, class {body} in the turn: {g:?}");
    }
    // the other way round: the car stands at lane 2's end, inside the bus's place
    let car_s = net.lanes[2].length() - 2.0;
    let bus_pair = js.for_bodies(j, js.nodes[j].movements[m0].foes.iter().find(|f| f.movement == m2).unwrap(), SWEPT_RIGID, Some(SWEPT_CAR)).unwrap();
    let turner_s = bus_pair.mine.0 - 2.0 - 3.0;
    for (body, held) in [(SWEPT_RIGID, true), (SWEPT_CAR, false)] {
        let mut car = standing(&js, 2, 2, car_s, SWEPT_CAR);
        car.held = true;
        let g = resolve(&js, &[standing(&js, 0, 1, turner_s, body), car]);
        let g = g.iter().find(|g| g.vehicle == 1).unwrap();
        assert_eq!(!g.go && g.wait_for == Some(2), held, "class {body} turning, a car at lane 2's end: {g:?}");
    }
}

/// Crossings and side merges are one stretch for every body class (the classes' own places are
/// the swept paths'): two turns converging in a bend, 3 m apart at the closest, get no place for any
/// class (their centre lines never within `CONVERGE_DIST`), and a crossing carries no class table.
#[test]
fn k3_crossings_and_side_merges_are_one_stretch_for_every_class() {
    let (r, k) = (15.0f64, std::f64::consts::FRAC_1_SQRT_2);
    let near = (r - (r + 3.0) * k, (r + 3.0) * k);
    let mut net = Network::default();
    net.lanes.push(keyed(turn((0.0, 0.0), r), 2, 9, 0));
    net.lanes.push(keyed(line((near.0 - 15.0 * k, near.1 - 15.0 * k), (near.0 + 15.0 * k, near.1 + 15.0 * k)), 2, 9, 1));
    net.lanes.push(keyed(line((0.0, -30.0), (0.0, 0.0)), 1, 1, 0));
    net.lanes.push(keyed(line((-20.0, 5.0), (20.0, 5.0)), 2, 9, 2));
    net.link(1.5);
    net.compute_conflicts();
    assert!(!net.crossings[0].iter().any(|c| c.other == 1 && c.side), "{:?}", net.crossings[0]);
    let cross = net.crossings[0].iter().find(|c| c.other == 3).expect("the turn crosses lane 3");
    assert!(cross.classes.is_none() && !cross.swept, "{cross:?}");
}

/// On straights the class meeting places are the old ones: two lanes crossing at right angles
/// meet over the same stretch for every pair of classes (`MEET_DIST` either side, no swing).
#[test]
fn k3_on_straights_every_class_meets_alike() {
    let mut net = Network::default();
    net.lanes.push(keyed(line((-20.0, 0.0), (20.0, 0.0)), 2, 9, 0));
    net.lanes.push(keyed(line((0.0, -20.0), (0.0, 20.0)), 2, 9, 1));
    net.link(1.5);
    net.compute_conflicts();
    let js = Junctions::compile(&net);
    let (j, m0) = js.of_lane[&0];
    let foe = js.nodes[j].movements[m0].foes[0];
    for a in 0..omsi_sim::traffic::SWEPT_CLASSES.len() {
        for b in 0..omsi_sim::traffic::SWEPT_CLASSES.len() {
            let f = js.for_bodies(j, &foe, a, Some(b)).expect("a crossing meets every pair");
            assert_eq!((f.mine, f.theirs), (foe.mine, foe.theirs), "classes {a} {b}");
        }
    }
}

/// A lane leading in that only a larger body's sweep made a movement of the object is an approach
/// for a car to its end: a car on lane 2 (which only a bus turning sweeps) stops at the red light of
/// the piece it leads into (lane 3's line, lane 2's end), as on any lit approach.
#[test]
fn k3_a_car_on_a_lane_only_a_bus_sweeps_stops_at_its_red_line() {
    use omsi_sim::traffic::{AiState, TrafficLightController};
    use omsi_traffic::{Simulation, Vehicle, VehicleClass};
    let (net, _) = swept_turn_lit(true);
    assert!(Junctions::compile(&net).of_lane.contains_key(&2), "lane 2 is a movement (N5)");
    let ctl = TrafficLightController::new(vec![vec![(0, 60.0)]], 60.0);
    let mut sim = Simulation::new(&net, vec![ctl]);
    let mut st = AiState::new(2, 2.0, 1);
    st.speed = 8.0;
    st.front = 2.0;
    st.rear = 2.0;
    st.length = 4.0;
    st.max_speed_kmh = 50.0;
    let mut v = Vehicle::new(1, VehicleClass::Car, st, 0.9, 7919);
    v.fresh = 0.0;
    let mut cars = vec![v];
    let len = net.lanes[2].length();
    let mut furthest = 0.0f32;
    for _ in 0..(15.0 * 30.0) as usize {
        sim.tick(&net, 1.0 / 30.0, &mut cars);
        let c = &cars[0];
        let front = if c.state.lane == 2 { c.state.s + c.state.front } else { len + c.state.s + c.state.front };
        furthest = furthest.max(front);
    }
    assert!(furthest <= len + 0.3, "its front got {:.1} m past lane 2's start, its line at {len:.1}", furthest);
}

/// A hand-made junction: a lead-in (movement 0, lane 0, 105.2 m) whose last half metre a turn's
/// swept path covers, the way on (movement 1, lane 1) and the turn (movement 2, lane 2), which
/// gives way to both: it meets the lead-in at its start (0..0.7) and the way on 2..4.5 m on.
fn lead_in_and_turn() -> Junctions {
    use omsi_traffic::junction::{Foe, FoeKind, Junction, Movement};
    let foe = |movement: usize, mine: (f32, f32), theirs: (f32, f32), yields: bool| Foe { movement, kind: FoeKind::Cross, mine, theirs, yields, side: false, apart: 0.0, classes: None, shift: 0.0 };
    let mv = |lane: usize, continues: Vec<usize>, foes: Vec<Foe>| {
        let (start, end) = foes.iter().fold((f32::INFINITY, 0.0f32), |(s, e), f| (s.min(f.mine.0), e.max(f.mine.1)));
        Movement { lane, kind: LaneKind::Street, entries: Vec::new(), exits: Vec::new(), continues, light: None, foes, conflict_end: end, conflict_start: start, converge: Vec::new(), converge_with: Vec::new(), wide: Vec::new() }
    };
    let movements = vec![
        mv(0, vec![1], vec![foe(2, (104.7, 105.2), (0.0, 0.7), false)]),
        mv(1, Vec::new(), vec![foe(2, (0.0, 3.0), (2.0, 4.5), false)]),
        mv(2, Vec::new(), vec![foe(0, (0.0, 0.7), (104.7, 105.2), true), foe(1, (2.0, 4.5), (0.0, 3.0), true)]),
    ];
    let mut js = Junctions::default();
    js.nodes.push(Junction { key: ((0, 0), 9), movements, crossings: Vec::new(), class_meets: Vec::new() });
    for m in 0..3 {
        js.of_lane.insert(m, (0, m));
    }
    js
}

/// SUMO's `willPass`: a turner standing with its front in the swept place over a lead-in's end,
/// refused there for the stream it gives way to further on, does not wait for a car standing up
/// the lead-in: that car cannot reach the crossing further on before the turner has left the
/// place at the lead-in's end (x10 j787: #34764 at 33467's end, #33787 13 m up lane 33466, both
/// refused for each other and one eased off).
#[test]
fn k3_a_turner_in_a_lead_ins_swept_place_does_not_wait_for_the_lead_ins_queue() {
    let js = lead_in_and_turn();
    let mut up = standing(&js, 0, 1, 91.3, SWEPT_CAR);
    up.chain = vec![(0, 0.0), (1, 105.2)];
    let turner = standing(&js, 2, 2, -0.9, SWEPT_CAR);
    let g = resolve(&js, &[up.clone(), turner.clone()]);
    let (gu, gt) = (g.iter().find(|g| g.vehicle == 1).unwrap(), g.iter().find(|g| g.vehicle == 2).unwrap());
    assert!(gt.go, "the turner waits for the car up the lead-in: {gt:?}");
    assert!(gu.go || gu.wait_for == Some(2), "{gu:?}");
    // (the turner short of the place: the car up the lead-in comes first, as the table says)
    let short = standing(&js, 2, 2, -4.0, SWEPT_CAR);
    let g = resolve(&js, &[up, short]);
    assert_eq!(g.iter().find(|g| g.vehicle == 2).unwrap().wait_for, Some(1), "{g:?}");
}
