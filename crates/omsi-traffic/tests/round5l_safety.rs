//! Round 5 L safety review (review-r5l-safety.md) scenarios S1-S8, a car moving sideways into
//! something, plus the regression review's asks: tests that fail with `shift_clears`,
//! `SQUEEZE_SIGNAL` and `SQUEEZE_NEAR` off, and the footpath clip.
use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network};
use omsi_traffic::{Person, PlayerView, Simulation, Vehicle, VehicleClass};

const DT: f32 = 1.0 / 30.0;

fn lane(pts: &[(f64, f64)], kind: LaneKind, width: f32) -> omsi_sim::traffic::Lane {
    LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), kind, width)
}
fn road(pts: &[(f64, f64)], id: i64) -> omsi_sim::traffic::Lane {
    let mut l = lane(pts, LaneKind::Street, 3.0);
    l.source = 1;
    l.key = Some(LaneKey { tile: (0, 0), id, path: 0 });
    l
}
fn finish(mut net: Network) -> Network {
    net.link(1.5);
    net.compute_conflicts();
    net
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
fn van(id: u64, lane: usize, s: f32, speed: f32) -> Vehicle {
    let mut v = car(id, lane, s, speed);
    v.half_width = 1.25;
    v
}
fn ai_bus(id: u64, lane: usize, s: f32, speed: f32) -> Vehicle {
    let mut st = AiState::new(lane, s, id);
    st.speed = speed;
    st.front = 6.0;
    st.rear = 6.0;
    st.length = 12.0;
    st.max_speed_kmh = 60.0;
    st.veh_type = 2;
    let mut v = Vehicle::new(id, VehicleClass::Bus, st, 1.25, id * 7919);
    v.fresh = 0.0;
    v
}
fn bus(x: f64, y_axis: f64, lateral: f64, still: f32) -> PlayerView {
    PlayerView { id: 7, pos: DVec3::new(x, y_axis - lateral, 0.0), heading: 90.0, half_len: 6.0, half_width: 1.25, speed: 0.0, still, blinker: 0, signalling: 0.0, priority: false }
}
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
/// Two same-direction 1.9 m lanes 3.6 m apart (BeijingCity's narrow lanes): 0 left at y=0, 1
/// right at y=-3.6; the right lane's extent to the left lane's painted edge (3.6-0.95=2.65).
fn two_narrow() -> Network {
    let mut net = Network::default();
    let mut a = lane(&[(0.0, 0.0), (600.0, 0.0)], LaneKind::Street, 1.9);
    a.source = 1;
    a.key = Some(LaneKey { tile: (0, 0), id: 1, path: 0 });
    a.offset = 0.0;
    a.drivable = (0.95, 2.65);
    let mut b = lane(&[(0.0, -3.6), (600.0, -3.6)], LaneKind::Street, 1.9);
    b.source = 1;
    b.key = Some(LaneKey { tile: (0, 0), id: 1, path: 1 });
    b.offset = 3.6;
    b.drivable = (2.65, 0.95);
    net.lanes.push(a);
    net.lanes.push(b);
    finish(net)
}
fn jianandong() -> Network {
    let mut net = Network::default();
    let mut a = lane(&[(0.0, 0.0), (600.0, 0.0)], LaneKind::Street, 1.9);
    a.source = 1;
    a.key = Some(LaneKey { tile: (0, 0), id: 1, path: 1 });
    a.offset = 1.8;
    a.drivable = (2.65, 4.6);
    net.lanes.push(a);
    let mut b = lane(&[(600.0, 3.6), (0.0, 3.6)], LaneKind::Street, 1.9);
    b.source = 1;
    b.key = Some(LaneKey { tile: (0, 0), id: 1, path: 0 });
    b.offset = 1.8;
    b.drivable = (2.65, 3.6);
    net.lanes.push(b);
    finish(net)
}
fn sedan(id: u64, lane: usize, s: f32, speed: f32) -> Vehicle {
    let mut v = car(id, lane, s, speed);
    v.half_width = 0.97;
    v.state.front = 2.42;
    v.state.rear = 2.2;
    v.state.length = 4.62;
    v
}
fn walker(x: f64, y: f64, vx: f64, vy: f64) -> Person {
    Person { pos: DVec3::new(x, y, 0.0), vel: glam::DVec2::new(vx, vy), on_crossing: false, still: 0.0, waiting_for_bus: false, at_kerb: false }
}
fn walkers_beside(x0: f64) -> Vec<Person> {
    (0..4).map(|k| walker(x0 + k as f64 * 0.7, -1.3 - 0.07 * k as f64, 0.0, 0.1)).collect()
}

struct Out {
    pass: Option<omsi_traffic::Passing>,
    least_other: f32,
    least_bus: f32,
    max_lat: f32,
    x_end: f64,
    out_at: Option<f32>,
    least_other_t: f32,
}

/// Car 1 (`me`) with the player's bus and `others`; watch `secs`: the squeeze it started, the
/// least its body came to any other vehicle / to the bus, its largest lateral offset.
fn watch(net: &Network, me: Vehicle, player: Option<PlayerView>, others: Vec<Vehicle>, people: Vec<Person>, secs: f32) -> Out {
    let mut sim = Simulation::new(net, vec![]);
    sim.players = player.into_iter().collect();
    sim.people = people.clone();
    let mut cars = vec![me];
    cars.extend(others);
    let mut r = Out { pass: None, least_other: f32::INFINITY, least_bus: f32::INFINITY, max_lat: 0.0, x_end: 0.0, out_at: None, least_other_t: 0.0 };
    let mut t = 0.0f32;
    while t < secs {
        sim.players = player.into_iter().collect();
        sim.people = people.clone();
        sim.tick(net, DT, &mut cars);
        for &i in &sim.removed {
            cars.remove(i);
        }
        let Some(me) = cars.iter().find(|c| c.id == 1) else { break };
        if r.pass.is_none() {
            r.pass = me.passing;
        }
        if r.out_at.is_none() && me.state.lateral.abs() > 0.3 {
            r.out_at = Some(t);
        }
        if me.state.lateral.abs() > r.max_lat.abs() {
            r.max_lat = me.state.lateral;
        }
        for o in cars.iter().filter(|c| c.id != 1) {
            let s = me.obb().separation(&o.obb()) as f32;
            if s < r.least_other {
                r.least_other = s;
                r.least_other_t = t;
            }
        }
        if let Some(p) = player {
            r.least_bus = r.least_bus.min(me.obb().separation(&p.obb(0.0, 0.0)) as f32);
        }
        r.x_end = me.pos.x;
        t += DT;
    }
    r
}

#[test]
fn s1_wheels_over_the_line_with_a_car_arriving_in_the_neighbour_lane() {
    // the implementer's wheels-over case (two_lanes 1.5/1.5, bus 1.6 right): a car in the left
    // lane arriving at 14 m/s from various distances behind
    let net = two_lanes((1.5, 1.5), false);
    for start in [0.0f32, 30.0, 60.0, 90.0, 110.0, 130.0] {
        let other = car(5, 0, start, 14.0);
        let r = watch(&net, car(1, 1, 150.0, 8.0), Some(bus(200.0, -3.5, 1.6, 30.0)), vec![other], vec![], 30.0);
        eprintln!("S1 neighbour from s={start}: pass {:?} beside {:?} out_at {:?} least_other {:.2} at t {:.1} least_bus {:.2} max_lat {:.2} x_end {:.1}", r.pass.map(|p| p.side), r.pass.and_then(|p| p.beside), r.out_at, r.least_other, r.least_other_t, r.least_bus, r.max_lat, r.x_end);
        assert!(r.least_other > 0.0, "contact with the neighbour-lane car (start {start}): {:.2}", r.least_other);
    }
}

#[test]
fn s2_bus_lane_neighbour_with_an_ai_bus_coming_at_12() {
    let mut net = two_lanes((1.5, 1.5), false);
    net.lanes[0].rule_bus = true;
    net.lanes[0].no_cars = true;
    let net = finish(net);
    for start in [0.0f32, 60.0, 100.0] {
        let r = watch(&net, car(1, 1, 150.0, 8.0), Some(bus(200.0, -3.5, 1.6, 30.0)), vec![ai_bus(5, 0, start, 12.0)], vec![], 30.0);
        eprintln!("S2 bus lane, bus from s={start}: pass {:?} beside {:?} least_other {:.2} at t {:.1} max_lat {:.2} x_end {:.1}", r.pass.map(|p| p.side), r.pass.and_then(|p| p.beside), r.least_other, r.least_other_t, r.max_lat, r.x_end);
        assert!(r.pass.and_then(|p| p.beside).is_none(), "wheels over into a bus lane: {:?}", r.pass);
        assert!(r.least_other > 0.0, "contact with the bus: {:.2}", r.least_other);
    }
}

#[test]
fn s3_squeeze_to_the_extent_on_narrow_lanes_oncoming_van_at_14() {
    // JiananDong: the bus 0.8 m right of the axis of a 1.9 m lane: the squeeze needs 2.5 m
    // left of it = 1.7 m, reach 2.6 <= extent 2.65 (the oncoming lane's painted edge); an
    // oncoming van (2.5 m wide) centred on its 1.9 m lane reaches 2.35 m from my axis
    let net = jianandong();
    let mut cases: Vec<(f32, f32, f64)> = vec![(350.0, 1.25, 0.9), (420.0, 1.25, 0.9), (350.0, 0.97, 0.9), (420.0, 0.97, 0.9)];
    for from in [260.0f32, 270.0, 280.0, 290.0, 300.0, 310.0, 320.0, 330.0, 340.0] {
        cases.push((from, 1.25, 1.0));
        cases.push((from, 0.97, 1.0));
        cases.push((from, 1.25, 0.9));
    }
    let mut contacts = Vec::new();
    for (from, hw, bl) in cases {
        let mut on = van(5, 1, from, 14.0);
        on.half_width = hw;
        let r = watch(&net, sedan(1, 0, 150.0, 8.0), Some(bus(200.0, 0.0, bl, 30.0)), vec![on], vec![], 30.0);
        eprintln!("S3 bus lat {bl} oncoming hw {hw} from s={from} (x={}): pass {:?} beside {:?} out_at {:?} least_other {:.2} at t {:.1} least_bus {:.2} max_lat {:.2} x_end {:.1}", 600.0 - from, r.pass.map(|p| (p.side, p.lane)), r.pass.and_then(|p| p.beside), r.out_at, r.least_other, r.least_other_t, r.least_bus, r.max_lat, r.x_end);
        if r.least_other <= 0.0 {
            contacts.push((bl, hw, from, r.least_other));
        }
    }
    assert!(contacts.is_empty(), "contacts with the oncoming vehicle (bus lat, hw, from, overlap): {contacts:?}");
    // and with no oncoming car at all: the squeeze itself
    let r = watch(&net, sedan(1, 0, 150.0, 8.0), Some(bus(200.0, 0.0, 0.9, 30.0)), vec![], vec![], 30.0);
    eprintln!("S3 no oncoming: pass {:?} least_bus {:.2} max_lat {:.2} x_end {:.1}", r.pass, r.least_bus, r.max_lat, r.x_end);
}

#[test]
fn s4_squeeze_to_the_extent_on_narrow_same_direction_lanes_car_arrives_beside() {
    // two 1.9 m lanes 3.6 m apart; the bus 0.8 m right of the right lane's axis; the squeeze
    // reaches 2.6 m left (extent 2.65 = the left lane's painted edge, 0.95 from its axis);
    // a car in the left lane arrives at 14 m/s
    let net = two_narrow();
    let mut cases: Vec<(f32, f32, f64)> = vec![(0.0, 1.25, 0.8), (60.0, 1.25, 0.8), (100.0, 1.25, 0.8), (60.0, 0.97, 0.8)];
    for start in [80.0f32, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0] {
        cases.push((start, 1.25, 0.9));
        cases.push((start, 0.97, 0.9));
    }
    let mut contacts = Vec::new();
    for (start, hw, bl) in cases {
        let mut o = car(5, 0, start, 14.0);
        o.half_width = hw;
        let r = watch(&net, sedan(1, 1, 150.0, 8.0), Some(bus(200.0, -3.6, bl, 30.0)), vec![o], vec![], 30.0);
        eprintln!("S4 bus lat {bl} neighbour hw {hw} from s={start}: pass {:?} beside {:?} out_at {:?} least_other {:.2} at t {:.1} least_bus {:.2} max_lat {:.2} x_end {:.1}", r.pass.map(|p| p.side), r.pass.and_then(|p| p.beside), r.out_at, r.least_other, r.least_other_t, r.least_bus, r.max_lat, r.x_end);
        if r.least_other <= 0.0 {
            contacts.push((bl, hw, start, r.least_other));
        }
    }
    assert!(contacts.is_empty(), "contacts with the neighbour-lane car (bus lat, hw, start, overlap): {contacts:?}");
}

#[test]
fn s5_l3_shift_towards_the_middle_with_an_oncoming_car_at_14() {
    let net = jianandong();
    for from_x in [250.0f64, 310.0, 400.0] {
        let on = van(5, 1, (600.0 - from_x) as f32, 14.0);
        let r = watch(&net, sedan(1, 0, 190.0, 4.0), None, vec![on], walkers_beside(205.0), 20.0);
        eprintln!("S5 oncoming van from x={from_x} ({} m ahead): least_other {:.2} at t {:.1} max_lat {:.2} x_end {:.1}", from_x - 190.0, r.least_other, r.least_other_t, r.max_lat, r.x_end);
        assert!(r.least_other > 0.0, "contact with the oncoming van (from {from_x}): {:.2}", r.least_other);
    }
    // both directions with walkers beside: both shift towards the middle
    let mut people = walkers_beside(205.0);
    people.extend((0..4).map(|k| walker(212.0 + k as f64 * 0.7, 3.6 + 1.3 + 0.07 * k as f64, 0.0, -0.1)));
    let on = sedan(5, 1, 600.0 - 230.0, 4.0);
    let r = watch(&net, sedan(1, 0, 190.0, 4.0), None, vec![on], people, 20.0);
    eprintln!("S5 both sides: least_other {:.2} at t {:.1} max_lat {:.2} x_end {:.1}", r.least_other, r.least_other_t, r.max_lat, r.x_end);
    assert!(r.least_other > 0.0, "two shifting cars met: {:.2}", r.least_other);
}

#[test]
fn s6_player_pulling_in_with_the_right_indicator_and_just_paused() {
    // the bus pulled in to the right, right indicator still on, paused 1.6 s: the squeeze goes
    // left; what happens (the rule as written allows it: player_out is the out side only)
    let net = two_lanes((2.0, 1.5), false);
    let mut p = bus(200.0, -3.5, 1.6, 1.6);
    p.blinker = 2;
    p.signalling = 3.0;
    let r = watch(&net, car(1, 1, 184.0, 0.0), Some(p), vec![], vec![], 15.0);
    eprintln!("S6 right indicator: pass {:?} out_at {:?} least_bus {:.2} x_end {:.1}", r.pass.map(|p| p.side), r.out_at, r.least_bus, r.x_end);
    assert!(r.least_bus > 0.25, "touched the bus: {:.2}", r.least_bus);
    // hazard lights (doors open at a stop): squeezed too?
    let mut p = bus(200.0, -3.5, 1.6, 1.6);
    p.blinker = 3;
    p.signalling = 3.0;
    let r = watch(&net, car(1, 1, 184.0, 0.0), Some(p), vec![], vec![], 15.0);
    eprintln!("S6 hazard: pass {:?} out_at {:?} least_bus {:.2} x_end {:.1}", r.pass.map(|p| p.side), r.out_at, r.least_bus, r.x_end);
    assert!(r.least_bus > 0.25, "touched the bus: {:.2}", r.least_bus);
    // the bus in the lane (no bay), pausing 1.6 s in a queue behind an AI car standing 20 m on
    let net = two_lanes((2.0, 1.5), false);
    let mut ahead = car(5, 1, 230.0, 0.0);
    ahead.stopped = 30.0;
    ahead.pull_out = 1000.0;
    let r = watch(&net, car(1, 1, 184.0, 0.0), Some(bus(200.0, -3.5, 0.0, 1.6)), vec![ahead], vec![], 15.0);
    eprintln!("S6 bus centred in lane, paused 1.6 s: pass {:?} out_at {:?} least_bus {:.2} least_other {:.2} x_end {:.1}", r.pass.map(|p| p.side), r.out_at, r.least_bus, r.least_other, r.x_end);
    assert!(r.pass.is_none(), "squeezed past a bus centred in a queue: {:?}", r.pass);
    assert!(r.least_bus > 0.25 && r.least_other > 0.25);
}

#[test]
fn s7_clip_distance_a_lane_alongside_9_m_away() {
    // a 1.0 m lane with 9.3 m of drawn road to its right (AndingR_noside path4) and a 3 m
    // carriageway lane of another spline 8 m to its right: is the extent cut?
    let mut net = Network::default();
    let mut a = lane(&[(0.0, 0.0), (300.0, 0.0)], LaneKind::Street, 1.0);
    a.source = 1;
    a.key = Some(LaneKey { tile: (0, 0), id: 1, path: 0 });
    a.drivable = (4.22, 9.3);
    net.lanes.push(a);
    let mut b = road(&[(0.0, -8.0), (300.0, -8.0)], 2);
    b.drivable = (1.5, 1.5);
    net.lanes.push(b);
    let net = finish(net);
    eprintln!("S7 extent with a lane 8 m right: {:?}", net.lanes[0].drivable());
    // cut a car's half width + 0.3 short of the lane's axis: 8 - 1.5 - 0.3 = 6.2
    assert!((net.lanes[0].drivable().1 - 6.2).abs() < 1e-4, "{:?}", net.lanes[0].drivable());
    for dist in [4.0f64, 5.0, 6.0, 7.0, 8.0] {
        let mut net = Network::default();
        let mut a = lane(&[(0.0, 0.0), (300.0, 0.0)], LaneKind::Street, 1.0);
        a.source = 1;
        a.key = Some(LaneKey { tile: (0, 0), id: 1, path: 0 });
        a.drivable = (4.22, 9.3);
        net.lanes.push(a);
        net.lanes.push(road(&[(0.0, -dist), (300.0, -dist)], 2));
        let net = finish(net);
        eprintln!("S7 lane {dist} m right: {:?}", net.lanes[0].drivable());
        let want = (dist as f32 - 1.5 - 0.3).max(0.5);
        assert!((net.lanes[0].drivable().1 - want).abs() < 1e-4, "{dist}: {:?}", net.lanes[0].drivable());
    }
}

/// AndingR_noside lane 1435: a 1.0 m lane with 4.22 m of drawn road left and 9.3 m right (an
/// asphalt-textured pedestrian area); a car (#376) stands for people in the lane; the car behind
/// (#450) squeezes; people stand on the strip a few metres on.
#[test]
fn s8_squeeze_onto_the_anding_strip_stopped_by_people_mid_ramp() {
    let mut net = Network::default();
    let mut a = lane(&[(0.0, 0.0), (600.0, 0.0)], LaneKind::Street, 1.0);
    a.source = 1;
    a.key = Some(LaneKey { tile: (0, 0), id: 1, path: 0 });
    a.drivable = (4.22, 9.3);
    net.lanes.push(a);
    let net = finish(net);
    // the passed car: a sedan standing 1.2 m left of the axis for a person 3 m in front of
    // it, so the squeeze goes right (1.0 m), onto the strip where the other people stand
    let mut ahead = sedan(5, 0, 200.0, 0.0);
    ahead.stopped = 30.0;
    ahead.pull_out = 1000.0;
    ahead.state.lateral = -1.2;
    ahead.state.lateral_target = -1.2;
    ahead.state.lateral_ramp = (-1.2, -1.2, 0.0, 1.0);
    ahead.external.bay = Some(-1.2);
    // (expected: no squeeze while the person the passed car stands for is in the lane ahead
    // - the squeeze's stretch runs past them 0.3 m off - nor while the people on the strip
    // are in its corridor; with nobody in the lane and the strip's people more than 3 m right
    // of the axis the squeeze goes)
    for (people_x, people_y, in_lane, expect) in [(206.0f64, -2.0f64, true, false), (203.0, -2.4, true, false), (208.0, -3.0, true, false), (212.0, -2.0, true, false), (300.0, -2.0, true, false), (206.0, -2.0, false, false), (203.0, -2.4, false, false), (208.0, -3.0, false, true), (300.0, -2.0, false, true)] {
        // somebody in the passed car's lane (what holds it) and others on the strip beside
        let mut people = if in_lane { vec![walker(206.0, 0.0, 0.0, 0.0)] } else { vec![] };
        people.push(walker(people_x, people_y, 0.0, 0.0));
        people.push(walker(people_x + 0.6, people_y - 0.3, 0.0, 0.0));
        let r = watch(&net, car(1, 0, 150.0, 8.0), None, vec![ahead.clone()], people, 40.0);
        eprintln!("S8 people at ({people_x}, {people_y}) in_lane {in_lane}: pass {:?} out_at {:?} least_other {:.2} at t {:.1} max_lat {:.2} x_end {:.1}", r.pass.map(|p| (p.side, p.lane, p.beside)), r.out_at, r.least_other, r.least_other_t, r.max_lat, r.x_end);
        assert!(r.least_other > 0.0, "people at ({people_x}, {people_y}): contact with the passed car {:.2}", r.least_other);
        assert_eq!(r.pass.is_some(), expect, "people at ({people_x}, {people_y}): {:?}", r.pass);
        if let Some(p) = r.pass {
            assert!(p.side > 0.0 && p.lane.is_none(), "{p:?}");
        }
    }
}


/// Car 1 squeezes past a car standing 1.7 m right of the axis of a lane with 3 m of drawn road
/// to its right (the obstacle half on the strip) and nothing on the left but the oncoming lane.
fn one_lane_strip(obstacle_lateral: f32) -> (Network, Vehicle) {
    let mut net = Network::default();
    let mut a = road(&[(0.0, 0.0), (600.0, 0.0)], 1);
    a.drivable = (1.5, 3.0);
    net.lanes.push(a);
    net.lanes.push(road(&[(600.0, 3.5), (0.0, 3.5)], 4));
    let net = finish(net);
    let mut o = car(5, 0, 200.0, 0.0);
    o.stopped = 30.0;
    o.pull_out = 1000.0;
    o.state.lateral = obstacle_lateral;
    o.state.lateral_target = obstacle_lateral;
    o.state.lateral_ramp = (obstacle_lateral, obstacle_lateral, 0.0, 1.0);
    // (held there as in a bay: its own plan would otherwise aim back at the axis and the
    // standing car would read as askew by its planned ramp)
    o.external.bay = Some(obstacle_lateral);
    (net, o)
}

#[test]
fn r1_the_thing_passed_holds_the_car_until_the_bent_way_really_clears_it() {
    // (fails with `shift_clears` off) a squeeze left round a car standing 1.7 m right; while
    // the car is out, a second car is put down in the corridor 14 m past the obstacle, in the
    // lane's middle: the lateral arithmetic says "passed" (it is near the axis, the side is
    // left), the bodies say no: it stays the lead and the car stops behind it
    let (net, o) = one_lane_strip(1.7);
    let mut sim = Simulation::new(&net, vec![]);
    let mut spare = car(999, 1, 10.0, 0.0);
    spare.pull_out = 1000.0;
    let mut cars = vec![car(1, 0, 150.0, 8.0), o, spare];
    let (mut out, mut placed, mut least) = (false, false, f32::INFINITY);
    let mut t = 0.0f32;
    while t < 30.0 {
        sim.tick(&net, DT, &mut cars);
        let me = cars.iter().find(|c| c.id == 1).unwrap().clone();
        out |= me.state.lateral < -0.3;
        if out && !placed {
            placed = true;
            let mut b = car(3, 0, 214.0, 0.0);
            b.stopped = 30.0;
            b.pull_out = 1000.0;
            if let Some(slot) = cars.iter_mut().find(|c| c.id == 999) {
                *slot = b;
            }
        }
        if placed {
            if let Some(b) = cars.iter().find(|c| c.id == 3) {
                least = least.min(me.obb().separation(&b.obb()) as f32);
            }
        }
        t += DT;
    }
    assert!(out, "it never squeezed out");
    assert!(least > 0.2, "drove into the car put down in its corridor: {least:.2} m");
    let me = cars.iter().find(|c| c.id == 1).unwrap();
    assert!(me.pos.x < 212.0, "x {:.1}", me.pos.x);
}

#[test]
fn r2_a_squeeze_signals_half_a_second_not_two() {
    // (fails with SQUEEZE_SIGNAL at 2 s) the player's bus 2 m right of the axis, the car
    // closed up behind it: the indicator leads the pull-out by 0.5-1.0 s
    let mut net = Network::default();
    let mut a = road(&[(0.0, 0.0), (600.0, 0.0)], 1);
    a.drivable = (1.5, 3.0);
    net.lanes.push(a);
    net.lanes.push(road(&[(600.0, 3.5), (0.0, 3.5)], 4));
    let net = finish(net);
    let mut sim = Simulation::new(&net, vec![]);
    let player = bus(200.0, 0.0, 2.0, 30.0);
    sim.players = vec![player];
    let mut cars = vec![car(1, 0, 184.0, 0.0)];
    let (mut wait, mut started_at, mut out_at) = (None, None, None);
    let mut t = 0.0f32;
    while t < 15.0 {
        sim.players = vec![player];
        sim.tick(&net, DT, &mut cars);
        let me = &cars[0];
        if let (None, Some(p)) = (wait, me.passing) {
            wait = Some(p.wait);
            started_at = Some(t);
            assert!(p.lane.is_none(), "{p:?}");
        }
        if out_at.is_none() && me.state.lateral.abs() > 0.1 {
            out_at = Some(t);
        }
        t += DT;
    }
    let wait = wait.expect("never squeezed");
    assert!(wait > 0.4 && wait < 1.0, "the squeeze signals {wait:.2} s before pulling out");
    // (and it is out within three seconds of deciding: 0.5 s of indicator, then the ramp)
    let (s0, o) = (started_at.unwrap(), out_at.expect("never pulled out"));
    assert!(o - s0 < 3.0, "out {:.2} s after deciding", o - s0);
}

#[test]
fn r3_a_squeeze_past_an_obstacle_within_25_m_is_not_kept_40_m_from_a_junction() {
    // (fails with SQUEEZE_NEAR at 0) a car standing 1.7 m right of the axis, a junction
    // piece 39 m from where the car stops behind it: JUNCTION_KEEP (40 m) would forbid the
    // pass, the squeeze's own zone (25 m past the obstacle) does not
    let mut net = Network::default();
    let mut a = road(&[(0.0, 0.0), (229.0, 0.0)], 1);
    a.drivable = (1.5, 3.0);
    net.lanes.push(a); // 0
    let mut piece = lane(&[(229.0, 0.0), (249.0, 0.0)], LaneKind::Street, 3.0);
    piece.source = 2;
    piece.key = Some(LaneKey { tile: (0, 0), id: 9, path: 0 });
    net.lanes.push(piece); // 1
    net.lanes.push(road(&[(249.0, 0.0), (600.0, 0.0)], 2)); // 2
    net.lanes.push(road(&[(239.0, -300.0), (239.0, -15.0)], 5)); // 3
    let mut main = lane(&[(239.0, -15.0), (239.0, 15.0)], LaneKind::Street, 3.0);
    main.source = 2;
    main.key = Some(LaneKey { tile: (0, 0), id: 9, path: 2 });
    main.priority = 192.0;
    net.lanes.push(main); // 4
    net.lanes.push(road(&[(239.0, 15.0), (239.0, 300.0)], 6)); // 5
    net.lanes.push(road(&[(600.0, 3.5), (0.0, 3.5)], 4)); // 6
    let net = finish(net);
    let mut o = car(5, 0, 200.0, 0.0);
    o.stopped = 30.0;
    o.pull_out = 1000.0;
    o.state.lateral = 1.7;
    o.state.lateral_target = 1.7;
    o.state.lateral_ramp = (1.7, 1.7, 0.0, 1.0);
    o.external.bay = Some(1.7);
    let r = watch(&net, car(1, 0, 150.0, 8.0), None, vec![o], vec![], 25.0);
    eprintln!("R3: pass {:?} out_at {:?} least_other {:.2} x_end {:.1}", r.pass.map(|p| (p.side, p.lane)), r.out_at, r.least_other, r.x_end);
    let pass = r.pass.expect("no squeeze past the car standing 6 m before the 40 m junction keep");
    assert!(pass.lane.is_none(), "{pass:?}");
    assert!(r.least_other > 0.25 && r.x_end > 210.0, "least {:.2} x {:.1}", r.least_other, r.x_end);
}

#[test]
fn r4_a_footpath_of_another_spline_alongside_cuts_the_extent_at_its_edge() {
    // JiananDong_1 lanes 83/87: 4.6 m of asphalt to the right, but a raised pavement spline
    // (L_Roadside_5m) runs alongside with its footpath axis 2.75 m out, 1.5 m wide: the
    // extent stops at its edge, 2.0 m; the lane's own pavement path (same spline, laid on the
    // asphalt) cuts nothing
    let mut net = Network::default();
    let mut a = lane(&[(0.0, 0.0), (300.0, 0.0)], LaneKind::Street, 1.9);
    a.source = 1;
    a.key = Some(LaneKey { tile: (0, 0), id: 1, path: 1 });
    a.drivable = (2.3, 4.6);
    net.lanes.push(a);
    let mut own = lane(&[(0.0, -3.845), (300.0, -3.845)], LaneKind::Sidewalk, 1.567);
    own.source = 1;
    own.key = Some(LaneKey { tile: (0, 0), id: 1, path: 2 });
    net.lanes.push(own);
    let net = finish(net);
    assert_eq!(net.lanes[0].drivable(), (2.3, 4.6));
    let mut foreign = lane(&[(0.0, -2.75), (300.0, -2.75)], LaneKind::Sidewalk, 1.5);
    foreign.source = 1;
    foreign.key = Some(LaneKey { tile: (0, 0), id: 2, path: 0 });
    let mut net = net;
    net.extend(vec![foreign], 1.5);
    assert_eq!(net.lanes[0].drivable(), (2.3, 2.0));
}

#[test]
fn s10_a_car_standing_level_or_just_ahead_on_the_squeeze_side_stops_the_squeeze() {
    // BJ new3 08:21 #3878/#4203: a car stranded 1.96 m out stood 0.21 m beside #3878, which
    // started a squeeze to that side past a lead 19 m ahead and yawed into it. Here: the lead
    // stands 1.7 m right of the axis 6 m ahead of the car's front (the squeeze goes left), and
    // another car stands out on the left, 1.96 m from the axis, level with the car / 3 m / 6 m
    // ahead: no squeeze starts into it; without it (the control) the squeeze goes. (A car
    // stands on the oncoming lane 40 m on, so that no pass on the oncoming lane is possible:
    // that pass has no body sweep of its own - pre-existing, outside this review's ask.)
    for ahead in [Some(0.0f32), Some(3.0), Some(6.0), None] {
        let (net, lead) = one_lane_strip(1.7);
        let mut me = car(1, 0, 190.0, 0.0);
        me.stopped = 5.0;
        let mut oncoming = car(7, 1, 600.0 - 230.0, 0.0);
        oncoming.stopped = 30.0;
        oncoming.pull_out = 1000.0;
        let mut others = vec![lead, oncoming];
        if let Some(a) = ahead {
            let mut beside = car(6, 0, 190.0 + a, 0.0);
            beside.stopped = 30.0;
            beside.pull_out = 1000.0;
            beside.state.lateral = -1.96;
            beside.state.lateral_target = -1.96;
            beside.state.lateral_ramp = (-1.96, -1.96, 0.0, 1.0);
            beside.external.bay = Some(-1.96);
            others.push(beside);
        }
        let r = watch(&net, me, None, others, vec![], 20.0);
        eprintln!("S10 beside {ahead:?} m ahead: pass {:?} out_at {:?} least_other {:.2} at t {:.1} max_lat {:.2} x_end {:.1}", r.pass.map(|p| (p.side, p.lane)), r.out_at, r.least_other, r.least_other_t, r.max_lat, r.x_end);
        assert!(r.least_other > 0.0, "beside {ahead:?} m ahead: contact {:.2}", r.least_other);
        match ahead {
            Some(_) => assert!(r.pass.is_none(), "beside {ahead:?} m ahead: set out into it: {:?}", r.pass),
            None => assert!(r.pass.is_some_and(|p| p.lane.is_none() && p.side < 0.0) && r.x_end > 210.0, "control: {:?} x {:.1}", r.pass, r.x_end),
        }
    }
}
