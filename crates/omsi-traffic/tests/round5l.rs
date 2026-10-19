//! Headless scenarios for round 5 area L: the drawn road's drivable extent as the room for an
//! in-lane squeeze (L2: Sobol's squeeze past the player's bus on any road, the wheels over
//! the line into a free neighbour lane, an early start, no squeeze across a lit stub or into
//! a junction's foe stretch) and the shift inside the drawn road past people on a kerbless
//! pavement (L3).

use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network};
use omsi_traffic::{Person, PlayerView, Simulation, Vehicle, VehicleClass};

const DT: f32 = 1.0 / 30.0;

fn lane(pts: &[(f64, f64)], kind: LaneKind, width: f32) -> omsi_sim::traffic::Lane {
    LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), kind, width)
}

/// A spline lane (source 1), 3 m wide, its drivable extent its half width either side.
fn road(pts: &[(f64, f64)], id: i64) -> omsi_sim::traffic::Lane {
    let mut l = lane(pts, LaneKind::Street, 3.0);
    l.source = 1;
    l.key = Some(LaneKey { tile: (0, 0), id, path: 0 });
    l
}

/// A lane of crossing object 9 (source 2), `width` m wide.
fn piece(pts: &[(f64, f64)], path: u16, width: f32) -> omsi_sim::traffic::Lane {
    let mut l = lane(pts, LaneKind::Street, width);
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

/// A car 4 m long, 1.8 m wide.
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

/// The player's bus (12 x 2.5 m) heading east at `x`, `lateral` m right of the lane axis at
/// `y_axis`, standing `still` seconds.
fn bus(x: f64, y_axis: f64, lateral: f64, still: f32) -> PlayerView {
    PlayerView { id: 7, pos: DVec3::new(x, y_axis - lateral, 0.0), heading: 90.0, half_len: 6.0, half_width: 1.25, speed: 0.0, still, blinker: 0, signalling: 0.0, priority: false }
}

/// A road with two lanes eastbound (0 the left one at y = 0, 1 the right one at y = -3.5),
/// 3 m wide; the right lane's drivable extent given (left, right).
fn two_lanes(drivable: (f32, f32), solid: bool) -> Network {
    let mut net = Network::default();
    let mut a = road(&[(0.0, 0.0), (600.0, 0.0)], 1);
    a.offset = 0.0;
    let mut b = road(&[(0.0, -3.5), (600.0, -3.5)], 1);
    b.key = Some(LaneKey { tile: (0, 0), id: 1, path: 1 });
    b.offset = 3.5;
    b.drivable = drivable;
    b.no_overtaking = solid;
    net.lanes.push(a);
    net.lanes.push(b);
    finish(net)
}

/// A car standing for good (nothing holds it: an obstacle after 4 s).
fn broken_down(id: u64, lane: usize, s: f32) -> Vehicle {
    let mut o = car(id, lane, s, 0.0);
    o.stopped = 30.0;
    o.pull_out = 1000.0;
    o
}

/// Car 1 on lane `lane` rolls up behind the player's bus (its box) and we watch 25 s: (the
/// squeeze it started, the least its body came to the bus, the lanes it was registered on
/// while out, how far it got).
struct Squeezed {
    pass: Option<omsi_traffic::Passing>,
    least: f32,
    on_lanes: Vec<usize>,
    x_end: f64,
    stopped_at_start: f32,
    still_at_start: f32,
    blinker_lead: f32,
}

fn squeeze_past(net: &Network, lane: usize, player: PlayerView, others: Vec<Vehicle>, still_ramps: bool) -> Squeezed {
    squeeze_past_from(net, lane, (150.0, 8.0), player, others, still_ramps)
}

fn squeeze_past_from(net: &Network, lane: usize, start: (f32, f32), player: PlayerView, others: Vec<Vehicle>, still_ramps: bool) -> Squeezed {
    let mut sim = Simulation::new(net, vec![]);
    sim.players = vec![player];
    let mut cars = vec![car(1, lane, start.0, start.1)];
    cars.extend(others);
    let mut r = Squeezed { pass: None, least: f32::INFINITY, on_lanes: Vec::new(), x_end: 0.0, stopped_at_start: 0.0, still_at_start: 0.0, blinker_lead: 0.0 };
    let mut out_at: Option<f32> = None;
    run(net, &mut sim, &mut cars, 25.0, |t, sim, cars| {
        let mut p = player;
        if still_ramps {
            p.still = t;
        }
        sim.players = vec![p];
        let me = cars.iter().find(|c| c.id == 1).unwrap();
        if r.pass.is_none() {
            if let Some(ps) = me.passing {
                r.pass = Some(ps);
                r.stopped_at_start = me.stopped;
                r.still_at_start = p.still;
            }
        }
        if out_at.is_none() && me.state.lateral.abs() > 0.3 {
            out_at = Some(t);
            r.blinker_lead = t - me.blinker_shown.1 as f32;
        }
        r.least = r.least.min(me.obb().separation(&p.obb(0.0, 0.0)) as f32);
        if me.state.lateral.abs() > 0.3 {
            for (l, occ) in &sim.occupancy {
                if occ.iter().any(|o| o.vehicle == 0) && !r.on_lanes.contains(l) {
                    r.on_lanes.push(*l);
                }
            }
        }
        r.x_end = me.pos.x;
    });
    r
}

#[test]
fn l2_a_car_squeezes_past_the_players_bus_half_out_of_its_lane_on_a_road_with_a_neighbour_lane() {
    // the right lane's drawn road reaches the left lane's painted edge (2.0 m) and a kerb at
    // its own right edge; the bus stands 1.6 m right of the axis (half on the kerb): the
    // squeeze needs 0.9 m to the left, 1.8 m of road from the axis - inside the drawn road
    let net = two_lanes((2.0, 1.5), false);
    let r = squeeze_past(&net, 1, bus(200.0, -3.5, 1.6, 30.0), vec![], false);
    let pass = r.pass.expect("no squeeze on a road with a neighbour lane");
    assert!(pass.lane.is_none() && pass.beside.is_none(), "{pass:?}");
    assert!((pass.side + 0.9).abs() < 0.05, "squeeze side {}", pass.side);
    assert!(r.least > 0.25, "came within {:.2} m of the bus", r.least);
    assert!(r.x_end > 215.0, "never got past the bus: x {:.1}", r.x_end);
    assert_eq!(r.on_lanes, vec![1], "registered on {:?}", r.on_lanes);
}

#[test]
fn l2_the_driver_side_wheels_go_over_the_line_into_a_free_neighbour_lane_not_a_taken_one_nor_a_solid_line() {
    // the drawn road ends at the lane's own edge (1.5 m): 0.3 m short for the squeeze, within
    // the 0.6 m the wheels may go over the line; the left lane free
    let net = two_lanes((1.5, 1.5), false);
    let r = squeeze_past(&net, 1, bus(200.0, -3.5, 1.6, 30.0), vec![], false);
    let pass = r.pass.expect("no squeeze with the wheels over the line");
    assert_eq!(pass.beside, Some(0), "{pass:?}");
    assert!(pass.lane.is_none());
    assert!(r.least > 0.25 && r.x_end > 215.0, "least {:.2} x {:.1}", r.least, r.x_end);
    let mut on = r.on_lanes.clone();
    on.sort();
    assert_eq!(on, vec![0, 1], "while out its body is registered on both lanes: {:?}", r.on_lanes);
    // a car standing in the left lane level with the bus: that lane is not free, no squeeze
    let r = squeeze_past(&net, 1, bus(200.0, -3.5, 1.6, 30.0), vec![broken_down(5, 0, 200.0)], false);
    assert!(r.pass.is_none(), "squeezed with a car in the lane its wheels go into: {:?}", r.pass);
    assert!(r.x_end < 194.0, "x {:.1}", r.x_end);
    // a solid line: the wheels stay on their side, no squeeze
    let net = two_lanes((1.5, 1.5), true);
    let r = squeeze_past(&net, 1, bus(200.0, -3.5, 1.6, 30.0), vec![], false);
    assert!(r.pass.is_none(), "squeezed over a solid line: {:?}", r.pass);
    // more than 0.6 m short (the bus 1.2 m right of the axis: 2.2 m needed of 1.5): no squeeze
    let net = two_lanes((1.5, 1.5), false);
    let r = squeeze_past(&net, 1, bus(200.0, -3.5, 1.2, 30.0), vec![], false);
    assert!(r.pass.is_none(), "squeezed 0.7 m over the line: {:?}", r.pass);
}

/// One lane each way: 0 eastbound at y = 0 to x = 600, 1 westbound at y = 3; the eastbound
/// lane's drawn road `right` m to its right (a parking strip, no kerb) and to the oncoming
/// lane's painted edge on its left; a lit stub at `lit` if given.
fn two_way(right: f32, lit: Option<f64>) -> Network {
    let mut net = Network::default();
    match lit {
        Some(x) => {
            let mut a = road(&[(0.0, 0.0), (x, 0.0)], 1);
            a.drivable = (1.5, right);
            let mut stub = road(&[(x, 0.0), (x + 1.0, 0.0)], 2);
            stub.traffic_light = Some((0, 0));
            stub.drivable = (1.5, right);
            let mut on = road(&[(x + 1.0, 0.0), (600.0, 0.0)], 3);
            on.drivable = (1.5, right);
            net.lanes.push(a);
            net.lanes.push(stub);
            net.lanes.push(on);
        }
        None => {
            let mut a = road(&[(0.0, 0.0), (600.0, 0.0)], 1);
            a.drivable = (1.5, right);
            net.lanes.push(a);
        }
    }
    net.lanes.push(road(&[(600.0, 3.0), (0.0, 3.0)], 4));
    finish(net)
}

#[test]
fn l2_the_players_bus_is_squeezed_past_as_soon_as_it_has_paused_with_half_a_second_of_indicator() {
    // the bus pulls in 2.0 m right of the axis onto the parking strip; the squeeze needs
    // 0.5 m to the left (1.4 m of road from the axis, inside the 1.5 m). The bus's standing
    // clock starts with the car already closed up 8 m behind it: the car goes round it
    // after the bus has paused 1.5 s, not the 4 s a pass waits, and without standing 3 s
    let net = two_way(3.0, None);
    let r = squeeze_past_from(&net, 0, (184.0, 0.0), bus(200.0, 0.0, 2.0, 0.0), vec![], true);
    let pass = r.pass.expect("no squeeze past the pausing bus");
    assert!(pass.lane.is_none() && pass.beside.is_none(), "{pass:?}");
    assert!(r.stopped_at_start < 3.0, "it stood {:.1} s before the squeeze", r.stopped_at_start);
    assert!(r.still_at_start > 1.5 && r.still_at_start < 2.5, "the bus had paused {:.1} s", r.still_at_start);
    // (and rolling up from 50 m back it goes round the bus as soon as it has closed up)
    let r = squeeze_past(&net, 0, bus(200.0, 0.0, 2.0, 30.0), vec![], false);
    assert!(r.pass.is_some() && r.stopped_at_start < 3.0 && r.x_end > 215.0, "{:?} stood {:.1} x {:.1}", r.pass, r.stopped_at_start, r.x_end);
    assert!(r.blinker_lead > 0.45, "signalled {:.2} s before pulling out", r.blinker_lead);
    assert!(r.least > 0.25 && r.x_end > 215.0, "least {:.2} x {:.1}", r.least, r.x_end);
}

#[test]
fn l2_no_squeeze_across_a_lit_stub_or_into_a_junction() {
    // a lit stub 19 m past the bus's front (within the 25 m zone): no squeeze
    let net = two_way(3.0, Some(225.0));
    let red = omsi_sim::traffic::TrafficLightController::new(vec![vec![(0, 60.0)]], 60.0);
    let mut sim = Simulation::new(&net, vec![red]);
    let player = bus(200.0, 0.0, 2.0, 30.0);
    sim.players = vec![player];
    let mut cars = vec![car(1, 0, 150.0, 8.0)];
    let mut started = false;
    run(&net, &mut sim, &mut cars, 20.0, |_, sim, cars| {
        sim.players = vec![player];
        started |= cars[0].passing.is_some();
    });
    assert!(!started, "squeezed towards the lit stub");
    assert!(cars[0].pos.x < 194.0, "x {:.1}", cars[0].pos.x);
    // a junction 14 m past the bus's front (the control at 60 m: squeezed)
    for (jx, expect) in [(220.0f64, false), (270.0, true)] {
        let mut net = Network::default();
        let mut a = road(&[(0.0, 0.0), (jx, 0.0)], 1);
        a.drivable = (1.5, 3.0);
        net.lanes.push(a); // 0
        net.lanes.push(piece(&[(jx, 0.0), (jx + 20.0, 0.0)], 0, 3.0)); // 1
        net.lanes.push(road(&[(jx + 20.0, 0.0), (600.0, 0.0)], 2)); // 2
        net.lanes.push(road(&[(jx + 10.0, -300.0), (jx + 10.0, -15.0)], 5)); // 3
        let mut main = piece(&[(jx + 10.0, -15.0), (jx + 10.0, 15.0)], 2, 3.0);
        main.priority = 192.0;
        net.lanes.push(main); // 4
        net.lanes.push(road(&[(jx + 10.0, 15.0), (jx + 10.0, 300.0)], 6)); // 5
        let net = finish(net);
        let r = squeeze_past(&net, 0, bus(200.0, 0.0, 2.0, 30.0), vec![], false);
        assert_eq!(r.pass.is_some(), expect, "junction at {jx}: {:?}", r.pass);
    }
}

/// A long crossing piece (3 m wide, source 2) from x = 100 to 200 with a main road across it
/// at `mx`; the way on to 600.
fn inside_junction(mx: f64) -> Network {
    let mut net = Network::default();
    net.lanes.push(road(&[(0.0, 0.0), (100.0, 0.0)], 1)); // 0
    net.lanes.push(piece(&[(100.0, 0.0), (200.0, 0.0)], 0, 3.0)); // 1
    net.lanes.push(road(&[(200.0, 0.0), (600.0, 0.0)], 2)); // 2
    net.lanes.push(road(&[(mx, -300.0), (mx, -15.0)], 5)); // 3
    let mut main = piece(&[(mx, -15.0), (mx, 15.0)], 2, 3.0);
    main.priority = 192.0;
    net.lanes.push(main); // 4
    net.lanes.push(road(&[(mx, 15.0), (mx, 300.0)], 6)); // 5
    finish(net)
}

#[test]
fn l2_inside_a_junction_the_squeeze_keeps_clear_of_the_foe_stretch() {
    // the bus pauses on the junction's piece 2 m right of its axis, the car behind it on the
    // piece too (`JUNCTION_KEEP` moot inside): with the main road 60 m on the squeeze ends
    // well before the foe stretch and goes; 20 m on, it would run into it and does not
    for (mx, expect) in [(190.0f64, true), (150.0, false)] {
        let net = inside_junction(mx);
        let mut sim = Simulation::new(&net, vec![]);
        let player = bus(130.0, 0.0, 2.0, 30.0);
        sim.players = vec![player];
        let mut me = car(1, 1, 5.0, 3.0);
        me.state.max_speed_kmh = 30.0;
        let mut cars = vec![me];
        let (mut pass, mut least) = (None, f32::INFINITY);
        run(&net, &mut sim, &mut cars, 20.0, |_, sim, cars| {
            sim.players = vec![player];
            if pass.is_none() {
                pass = cars[0].passing;
            }
            least = least.min(cars[0].obb().separation(&player.obb(0.0, 0.0)) as f32);
        });
        assert_eq!(pass.is_some(), expect, "main road at {mx}: {pass:?}");
        if expect {
            assert!(pass.unwrap().lane.is_none(), "{pass:?}");
            assert!(least > 0.25, "least {least:.2}");
            assert!(cars[0].pos.x > 140.0, "x {:.1}", cars[0].pos.x);
        } else {
            assert!(cars[0].pos.x < 124.0, "x {:.1}", cars[0].pos.x);
        }
    }
}

#[allow(dead_code)]
fn walker(x: f64, y: f64, vx: f64, vy: f64) -> Person {
    Person { pos: DVec3::new(x, y, 0.0), vel: glam::DVec2::new(vx, vy), on_crossing: false, still: 0.0, waiting_for_bus: false, at_kerb: false }
}
