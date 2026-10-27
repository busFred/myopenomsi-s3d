//! Round 6 area L (review-r6-cases.md section 5, fixes 12-14): the bypass from rest behind a
//! standing obstacle, the settled offset of a frozen change, keep-right into closed pieces.
use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network};
use omsi_traffic::{Person, Simulation, Vehicle, VehicleClass};

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
fn walker(x: f64, y: f64, vx: f64, vy: f64) -> Person {
    Person { pos: DVec3::new(x, y, 0.0), vel: glam::DVec2::new(vx, vy), on_crossing: false, still: 0.0, waiting_for_bus: false, at_kerb: false, waits_at: None, id: None, approach: None }
}


/// One frame of the run: (time, the cars).
fn run(net: &Network, sim: &mut Simulation, cars: &mut Vec<Vehicle>, secs: f32, mut each: impl FnMut(f32, &Simulation, &[Vehicle])) {
    let mut t = 0.0f32;
    while t < secs {
        sim.tick(net, DT, cars);
        for &i in &sim.removed {
            cars.remove(i);
        }
        each(t, sim, cars);
        t += DT;
    }
}

/// A standing 12 m bus (id 5) on the right lane of `two_lanes` at s 200, waited behind by
/// car 1 standing `gap` m behind its rear.
fn behind_a_standing_bus(gap: f32) -> (Network, Vec<Vehicle>) {
    let net = two_lanes((1.5, 1.5), false);
    let mut b = ai_bus(5, 1, 200.0, 0.0);
    b.stopped = 60.0;
    b.pull_out = 1000.0;
    let mut me = car(1, 1, 200.0 - 6.0 - gap - 2.0, 0.0);
    me.stopped = 20.0;
    (net, vec![me, b])
}

/// What car 1 did round the standing bus: (when it was on the lane beside with its change
/// done, the changes it started, the least its body came to any other, the least to a person).
struct Round {
    done: Option<f32>,
    starts: usize,
    least: f32,
    least_person: f32,
}

fn round_the_bus(net: &Network, mut cars: Vec<Vehicle>, people: Vec<Person>, secs: f32) -> Round {
    let mut sim = Simulation::new(net, vec![]);
    sim.people = people.clone();
    let mut r = Round { done: None, starts: 0, least: f32::INFINITY, least_person: f32::INFINITY };
    let mut was = false;
    run(net, &mut sim, &mut cars, secs, |t, sim, cars| {
        let Some(me) = cars.iter().find(|c| c.id == 1) else { return };
        for o in cars.iter().filter(|c| c.id != 1) {
            r.least = r.least.min(me.obb().separation(&o.obb()) as f32);
        }
        for q in &sim.people {
            r.least_person = r.least_person.min((me.obb().separation(&omsi_sim::collision::Obb::point(q.pos, 0.28))) as f32);
        }
        let on = me.state.change.is_some();
        if on && !was {
            r.starts += 1;
        }
        was = on;
        if r.done.is_none() && me.state.lane == 0 && !on {
            r.done = Some(t);
        }
    });
    r
}

#[test]
fn r6l2_a_bypass_from_rest_steers_as_a_car_can_and_never_sets_out_into_the_obstacle() {
    // viz5 case08 #45474 stood 1.5-2.3 m behind the T11 bus #44422 and started its bypass six
    // times (blinker 5 s on, 10 s off): the 8 m move ran its way into the bus's corner. A 2 m
    // move instead pivoted the car 40-70 degrees on the spot (review r6l). Now: no start
    // where the turning radius cannot clear the bus; round it from further back, at no more
    // than about 30 degrees of yaw and with no jump of the heading
    for gap in [1.5f32, 2.5, 10.0, 14.0] {
        let (net, mut cars) = behind_a_standing_bus(gap);
        let mut sim = Simulation::new(&net, vec![]);
        let (mut done, mut starts, mut was, mut least) = (None, 0, false, f32::INFINITY);
        let (mut max_yaw, mut max_step, mut prev_h) = (0.0f32, 0.0f32, None::<f32>);
        run(&net, &mut sim, &mut cars, 25.0, |t, _, cars| {
            let (Some(me), Some(b)) = (cars.iter().find(|c| c.id == 1), cars.iter().find(|c| c.id == 5)) else { return };
            least = least.min(me.obb().separation(&b.obb()) as f32);
            let on = me.state.change.is_some();
            starts += (on && !was) as usize;
            was = on;
            if done.is_none() && me.state.lane == 0 && !on {
                done = Some(t);
            }
            // (the lanes run along x: heading 90)
            if me.pos.x < 260.0 {
                max_yaw = max_yaw.max((me.heading - 90.0).abs());
                if let Some(h) = prev_h {
                    max_step = max_step.max((me.heading - h).abs());
                }
                prev_h = Some(me.heading);
            }
        });
        eprintln!("gap {gap}: done {done:?} starts {starts} least {least:.2} max yaw {max_yaw:.1} max step {max_step:.2}");
        assert!(least > 0.3, "gap {gap}: came {least:.2} m to the bus");
        assert!(max_yaw < 33.0, "gap {gap}: yawed {max_yaw:.1} degrees");
        assert!(max_step < 2.0, "gap {gap}: the heading jumped {max_step:.2} degrees in a frame");
        if gap < 3.0 {
            assert_eq!(starts, 0, "gap {gap}: set out with no room to steer round");
        } else {
            let done = done.unwrap_or_else(|| panic!("gap {gap}: never got round ({starts} starts)"));
            assert!(done < 15.0, "gap {gap}: round only at {done:.1} s");
        }
    }
}

#[test]
fn r6l2_no_bypass_into_a_car_standing_level_or_just_ahead_in_the_lane_beside() {
    // (r5l S10 for the bypass) a car standing in the left lane level with car 1 or 3 m ahead
    // of its front: no bypass starts into it, nobody is touched
    for ahead in [0.0f32, 3.0] {
        let (net, mut cars) = behind_a_standing_bus(1.5);
        let mut beside = car(6, 0, cars[0].state.s + ahead, 0.0);
        beside.stopped = 60.0;
        beside.pull_out = 1000.0;
        cars.push(beside);
        let r = round_the_bus(&net, cars, vec![], 20.0);
        eprintln!("beside {ahead}: starts {} least {:.2}", r.starts, r.least);
        assert_eq!(r.starts, 0, "beside {ahead} m ahead: started a change into it");
        assert!(r.least > 0.0);
    }
}

/// A bus held at its stop (`at_stop`, `stop_at` here) on the right lane at s 200 of
/// `two_lanes`, and cars 1, 2, 3 coming up behind it at 10 m/s, 30 m apart (viz5 case07,
/// the queue behind #22368). (bus lateral: 0 in the lane, 3 in a bay)
fn queue_behind_a_bus_at_its_stop(bus_lateral: f32) -> (Network, Vec<Vehicle>) {
    let net = two_lanes((1.5, if bus_lateral != 0.0 { 4.5 } else { 1.5 }), false);
    let mut b = ai_bus(5, 1, 200.0, 0.0);
    b.stopped = 30.0;
    b.pull_out = 1000.0;
    b.external.at_stop = true;
    b.external.standing_for = 30.0;
    if bus_lateral != 0.0 {
        b.state.lateral = bus_lateral;
        b.state.lateral_target = bus_lateral;
        b.state.lateral_ramp = (bus_lateral, bus_lateral, 0.0, 1.0);
        b.external.bay = Some(bus_lateral);
    }
    let mut cars = vec![b];
    for k in 0..3u64 {
        let mut c = car(1 + k, 1, 120.0 - 30.0 * k as f32, 10.0);
        c.state.change_cooldown = 0.0;
        cars.push(c);
    }
    (net, cars)
}

#[test]
fn r6l2_a_queue_coming_up_behind_a_bus_at_its_stop_keeps_the_room_to_steer_round_it() {
    let (net, mut cars) = queue_behind_a_bus_at_its_stop(0.0);
    let mut sim = Simulation::new(&net, vec![]);
    // (per car: the least gap to the vehicle ahead in its lane while standing, whether it
    // got round onto the left lane, the least to any body)
    let mut closest = [f32::INFINITY; 3];
    let mut round = [false; 3];
    let mut least = f32::INFINITY;
    run(&net, &mut sim, &mut cars, 40.0, |_, _, cars| {
        for k in 0..3u64 {
            let Some(me) = cars.iter().find(|c| c.id == 1 + k) else { continue };
            for o in cars.iter().filter(|o| o.id != me.id) {
                least = least.min(me.obb().separation(&o.obb()) as f32);
            }
            round[k as usize] |= me.state.lane == 0;
            if me.state.lane == 1 && me.state.speed < 0.1 {
                if let Some(l) = me.last.lead {
                    closest[k as usize] = closest[k as usize].min(l.gap);
                }
            }
        }
    });
    eprintln!("queue: closest standing gaps {closest:?} round {round:?} least {least:.2}");
    assert!(least > 0.3, "came {least:.2} m to a body");
    assert!(round[0], "the first car never went round the bus");
    assert!(closest.iter().all(|g| !g.is_finite() || *g > 4.0), "the queue packed: {closest:?}");
}

#[test]
fn r6l2_a_bus_at_its_stop_in_a_bay_is_driven_past_not_kept_back_from() {
    let (net, mut cars) = queue_behind_a_bus_at_its_stop(3.0);
    let mut sim = Simulation::new(&net, vec![]);
    let mut slowest = f32::INFINITY;
    run(&net, &mut sim, &mut cars, 15.0, |_, _, cars| {
        if let Some(me) = cars.iter().find(|c| c.id == 1) {
            if me.pos.x > 150.0 && me.pos.x < 210.0 {
                slowest = slowest.min(me.state.speed);
            }
        }
    });
    assert!(slowest > 5.0, "slowed to {slowest:.1} m/s past a bus in its bay");
}

#[test]
fn r6l2_a_queue_at_a_red_light_is_closed_up_to_not_kept_back_from() {
    // the right lane ends at a red at x 200 (lit lane beyond): cars 1 and 2 come to stand in
    // its queue; car 3 comes up and closes to its queue gap, not a bypass's room back
    let mut net = Network::default();
    let mut a = road(&[(0.0, 0.0), (200.0, 0.0)], 1);
    a.offset = 0.0;
    let mut b = road(&[(0.0, -3.5), (200.0, -3.5)], 1);
    b.key = Some(LaneKey { tile: (0, 0), id: 1, path: 1 });
    b.offset = 3.5;
    let mut a2 = road(&[(200.0, 0.0), (600.0, 0.0)], 2);
    a2.offset = 0.0;
    a2.traffic_light = Some((0, 0));
    let mut b2 = road(&[(200.0, -3.5), (600.0, -3.5)], 2);
    b2.key = Some(LaneKey { tile: (0, 0), id: 2, path: 1 });
    b2.offset = 3.5;
    b2.traffic_light = Some((0, 0));
    net.lanes.extend([a, b, a2, b2]);
    let net = finish(net);
    let ctl = omsi_sim::traffic::TrafficLightController::new(vec![vec![(0, 1000.0)]], 1000.0);
    let mut sim = Simulation::new(&net, vec![ctl]);
    let mut cars = vec![car(1, 1, 120.0, 10.0), car(2, 1, 80.0, 10.0), car(3, 1, 20.0, 10.0)];
    for c in cars.iter_mut() {
        c.state.change_cooldown = 0.0;
    }
    let mut gap3 = None;
    run(&net, &mut sim, &mut cars, 60.0, |_, _, cars| {
        if let Some(me) = cars.iter().find(|c| c.id == 3) {
            if me.state.speed < 0.1 {
                gap3 = me.last.lead.map(|l| l.gap);
            }
        }
    });
    eprintln!("red queue: car 3 stood {gap3:?} behind");
    assert!(gap3.is_some_and(|g| g < 3.5), "kept back {gap3:?} from a red-light queue");
}

#[test]
fn r6l2_the_rear_of_a_steep_bypass_does_not_swing_into_a_car_standing_close_on_the_other_side() {
    // (x10 case09 #14309, case14 #34514) a car standing level with car 1, 0.5 m off its
    // right side on the strip: moving off left at a steep yaw swings car 1's rear right into
    // it; shifted sideways only, the corridor looked clear
    let (net, mut cars) = behind_a_standing_bus(1.5);
    let mut beside = car(7, 1, cars[0].state.s, 0.0);
    beside.pull_out = 1000.0;
    beside.stopped = 60.0;
    beside.state.lateral = 2.3;
    beside.state.lateral_target = 2.3;
    beside.state.lateral_ramp = (2.3, 2.3, 0.0, 1.0);
    beside.external.bay = Some(2.3);
    cars.push(beside);
    let r = round_the_bus(&net, cars, vec![], 15.0);
    eprintln!("beside right: done {:?} starts {} least {:.2}", r.done, r.starts, r.least);
    assert!(r.least > 0.1, "came {:.2} m to a car", r.least);
}

#[test]
fn r6l2_a_slower_car_ahead_in_the_lane_beside_is_not_cut_in_front_of() {
    // (viz2 case07 #10906) a car crawling at 2 m/s in the left lane with its rear 2 m past car
    // 1's front: no bypass into it while it is there, and no contact
    let (net, mut cars) = behind_a_standing_bus(1.5);
    let mut slow = car(6, 0, cars[0].state.s + 6.0, 2.0);
    slow.state.max_speed_kmh = 7.2;
    cars.push(slow);
    let r = round_the_bus(&net, cars, vec![], 20.0);
    eprintln!("slow: done {:?} starts {} least {:.2}", r.done, r.starts, r.least);
    assert!(r.least > 0.3, "came {:.2} m to it", r.least);
}

#[test]
fn r6l2_a_bus_indicating_out_of_its_stop_is_let_out_not_gone_round() {
    let (net, mut cars) = behind_a_standing_bus(1.5);
    cars[1].state.signal = 1;
    cars[1].state.signal_time = 1000.0;
    let mut sim = Simulation::new(&net, vec![]);
    let mut starts = 0;
    let mut shown = false;
    run(&net, &mut sim, &mut cars, 15.0, |_, _, cars| {
        shown |= cars[1].blinker_shown.0 == 1;
        starts += cars[0].state.change.is_some_and(|c| c.bypass) as usize;
    });
    assert!(shown, "the bus never showed its indicator");
    assert_eq!(starts, 0, "went round a bus indicating out");
}

#[test]
fn r6l2_no_bypass_with_somebody_on_foot_in_its_corridor() {
    // a person standing in the left lane beside the bus's rear
    let (net, cars) = behind_a_standing_bus(1.5);
    let x = cars[1].state.s as f64 - 6.5;
    let r = round_the_bus(&net, cars, vec![walker(x, -0.3, 0.0, 0.0)], 15.0);
    eprintln!("person: starts {} least_person {:.2}", r.starts, r.least_person);
    assert_eq!(r.starts, 0, "started round into the person");
    assert!(r.least_person > 0.3);
}

/// Car 1 on the right lane of `two_lanes` changing over to the left one, frozen 55 % of
/// the way behind car 2 standing in the right lane just ahead and car 4 in the left lane,
/// which drives off after 6 s: settled done in place on the left lane, 1.5 m right of its
/// middle, car 2's corner in its corridor, the left lane free (viz5 case05 #18836 behind
/// #20220, which had given up its own change into that lane).
fn settled_beside() -> (Network, Vec<Vehicle>) {
    // (case05's lanes 4916/4919 lie 2.9 m apart)
    let mut net = Network::default();
    let mut a = road(&[(0.0, 0.0), (600.0, 0.0)], 1);
    a.offset = 0.0;
    let mut b = road(&[(0.0, -2.9), (600.0, -2.9)], 1);
    b.key = Some(LaneKey { tile: (0, 0), id: 1, path: 1 });
    b.offset = 2.9;
    net.lanes.push(a);
    net.lanes.push(b);
    let net = finish(net);
    let mut me = car(1, 1, 100.0, 0.0);
    me.state.change = Some(omsi_sim::traffic::LaneChange { to: 0, t: 0.55, length: 12.0, s_to: 100.0, dir: 1, wait: 0.0, bypass: false, full_lock: false });
    me.state.change_cooldown = 0.0;
    let mut ahead = car(2, 1, 105.5, 0.0);
    ahead.pull_out = 1000.0;
    ahead.stopped = 60.0;
    let mut leaves = car(4, 0, 105.5, 0.0);
    leaves.pull_out = 6.0;
    (net, vec![me, ahead, leaves])
}

#[test]
fn r6l3_a_change_settled_standing_eases_its_offset_in_place_and_drives_on() {
    let (net, mut cars) = settled_beside();
    let mut sim = Simulation::new(&net, vec![]);
    let (mut settled, mut least, mut gone_at, mut lat_at_settle) = (None, f32::INFINITY, None, 0.0f32);
    run(&net, &mut sim, &mut cars, 30.0, |t, _, cars| {
        let Some(me) = cars.iter().find(|c| c.id == 1) else { return };
        let other = cars.iter().find(|c| c.id == 2).unwrap();
        least = least.min(me.obb().separation(&other.obb()) as f32);
        if settled.is_none() && me.state.change.is_none() {
            settled = Some(t);
            lat_at_settle = me.state.lateral;
        }
        if gone_at.is_none() && me.pos.x > 115.0 {
            gone_at = Some(t);
        }
    });
    let settled = settled.expect("never settled");
    eprintln!("settled at {settled:.1} s, lateral {lat_at_settle:.2}; past the car at {gone_at:?}; least {least:.2}");
    assert!(lat_at_settle.abs() > 1.0, "settled at lateral {lat_at_settle:.2}");
    let gone = gone_at.expect("stood behind the car beside with its lane free");
    assert!(gone - settled < 10.0, "drove on only {:.1} s after settling", gone - settled);
    assert!(least > 0.15, "came {least:.2} m to the car beside");
}

#[test]
fn r6l3_the_offset_eases_no_nearer_than_0_2_m_to_a_car_beside_in_the_lane() {
    // (viz case06) two cars side by side in one lane: once car 1 has settled 1.5 m right of
    // the left lane's middle (car 4 ahead stays), car 3 stands beside it 1 m left of that
    // middle: the offset eases in place until the bodies are 0.2 m apart, no further
    let (net, mut cars) = settled_beside();
    cars[2].pull_out = 1000.0;
    let mut sim = Simulation::new(&net, vec![]);
    let mut t = 0.0f32;
    while cars[0].state.change.is_some() && t < 8.0 {
        sim.tick(&net, DT, &mut cars);
        t += DT;
    }
    assert!(cars[0].state.change.is_none() && cars[0].state.lane == 0, "not settled into the left lane");
    let mut beside = car(3, 0, cars[0].state.s, 0.0);
    beside.pull_out = 1000.0;
    beside.stopped = 60.0;
    beside.state.lateral = -1.0;
    beside.state.lateral_target = -1.0;
    beside.state.lateral_ramp = (-1.0, -1.0, 0.0, 1.0);
    beside.external.bay = Some(-1.0);
    cars.push(beside);
    let mut least = f32::INFINITY;
    run(&net, &mut sim, &mut cars, 15.0, |_, _, cars| {
        let me = cars.iter().find(|c| c.id == 1).unwrap();
        let other = cars.iter().find(|c| c.id == 3).unwrap();
        least = least.min(me.obb().separation(&other.obb()) as f32);
    });
    let me = cars.iter().find(|c| c.id == 1).unwrap();
    eprintln!("lane {} lateral {:.2} least {least:.2}", me.state.lane, me.state.lateral);
    assert!(least > 0.15, "eased to {least:.2} m of the car beside");
    assert!(me.state.lateral < 1.45, "no easing at all: lateral {:.2}", me.state.lateral);
}

#[test]
fn r6l3_a_car_pulling_out_of_the_strip_with_its_offset_does_not_slide() {
    // (review r6l, x10 case08 #11387) a car put down standing 2.9 m right of its lane's
    // middle in the parking strip (a fresh car: `AiState::new`'s lane-change cooldown), held
    // by a car standing in the strip 2 m ahead: its offset is no settled change's, it stays
    let net = two_lanes((1.5, 4.0), false);
    let mut me = car(1, 1, 100.0, 0.0);
    me.state.lateral = 2.9;
    me.state.lateral_target = 0.0;
    me.state.lateral_ramp = (2.9, 0.0, 0.0, 30.0);
    me.pull_out = 4.0;
    assert!(me.state.change_cooldown > 0.0);
    let mut ahead = car(2, 1, 106.0, 0.0);
    ahead.state.lateral = 2.9;
    ahead.state.lateral_target = 2.9;
    ahead.state.lateral_ramp = (2.9, 2.9, 0.0, 1.0);
    ahead.external.bay = Some(2.9);
    ahead.pull_out = 1000.0;
    ahead.stopped = 60.0;
    let mut cars = vec![me, ahead];
    let mut sim = Simulation::new(&net, vec![]);
    run(&net, &mut sim, &mut cars, 5.0, |_, _, _| {});
    assert!((cars[0].state.lateral - 2.9).abs() < 0.01, "slid to lateral {:.2}", cars[0].state.lateral);
}

#[test]
fn r6l3_the_offset_does_not_ease_into_a_person_beside() {
    // (review r6l) car 1 settled 1.2 m off the left lane's middle, car 4 ahead stays; a
    // person stands 0.4 m off the side it eases towards: no nearer than 0.2 m
    let (net, mut cars) = settled_beside();
    cars[2].pull_out = 1000.0;
    let mut sim = Simulation::new(&net, vec![]);
    let mut t = 0.0f32;
    while cars[0].state.change.is_some() && t < 8.0 {
        sim.tick(&net, DT, &mut cars);
        t += DT;
    }
    let me = &cars[0];
    assert!(me.state.settled_offset && me.state.lateral.abs() > 1.0, "not settled with an offset");
    let h = (me.heading as f64).to_radians();
    let right = glam::DVec2::new(h.cos(), -h.sin());
    let lat = me.state.lateral as f64;
    let q = me.pos.truncate() - right * (lat.signum() * (me.half_width as f64 + 0.28 + 0.4));
    sim.people = vec![walker(q.x, q.y, 0.0, 0.0)];
    let mut least = f32::INFINITY;
    run(&net, &mut sim, &mut cars, 10.0, |_, sim, cars| {
        let me = cars.iter().find(|c| c.id == 1).unwrap();
        for q in &sim.people {
            least = least.min(me.obb().separation(&omsi_sim::collision::Obb::point(q.pos, 0.28)) as f32);
        }
    });
    let me = cars.iter().find(|c| c.id == 1).unwrap();
    eprintln!("person: lateral {:.2} least {least:.2}", me.state.lateral);
    assert!(least > 0.15, "eased to {least:.2} m of the person");
}

/// Three pieces of a two-lane road along x (BeijingCity's AnwaiDajie splines 3466, 3470,
/// 3474): the left lane (y 0) open to all, the right lane (y -3.5) a bus lane closed to the
/// normal cars by its density on the first piece (0..200) and the last (221..600), but the
/// 21 m piece between (3470) without rules. Lanes: 0/1 the first piece's left/right, 2/3 the
/// rule-less piece's, 4/5 the last piece's.
fn bus_lane_with_a_ruleless_piece() -> Network {
    let mut net = Network::default();
    for (k, (x0, x1)) in [(0.0, 200.0), (200.0, 221.0), (221.0, 600.0)].into_iter().enumerate() {
        let mut a = road(&[(x0, 0.0), (x1, 0.0)], 10 + k as i64);
        a.offset = 0.0;
        let mut b = road(&[(x0, -3.5), (x1, -3.5)], 10 + k as i64);
        b.key = Some(LaneKey { tile: (0, 0), id: 10 + k as i64, path: 1 });
        b.offset = 3.5;
        if k != 1 {
            b.density = 0.0;
        }
        net.lanes.push(a);
        net.lanes.push(b);
    }
    finish(net)
}

/// A car on the left lane of the first piece at `s` and 14 m/s, keen to keep right ([Q8-3] its
/// LC2013 keep-right wish accumulated to the threshold already: these tests are about where it
/// may go, not when).
fn keen(s: f32) -> Vehicle {
    let mut me = car(1, 0, s, 14.0);
    me.state.change_cooldown = 0.0;
    me.params.lc_keep_right = 1.0;
    me.keep_right = omsi_traffic::plan::KEEP_RIGHT_THRESHOLD;
    me.state.max_speed_kmh = 50.4;
    me
}

#[test]
fn r6l4_no_keep_right_onto_a_ruleless_piece_that_runs_into_a_closed_one() {
    // viz5 case04 #5644: from spline 3470's left lane over onto its rule-less bus lane, which
    // runs into 3474's, closed to it
    let net = bus_lane_with_a_ruleless_piece();
    assert!(net.lanes[2].right == Some(3) || net.lanes[3].left == Some(2), "the pieces' lanes are neighbours");
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![keen(150.0)];
    let mut onto = Vec::new();
    run(&net, &mut sim, &mut cars, 12.0, |_, _, cars| {
        let Some(me) = cars.iter().find(|c| c.id == 1) else { return };
        if let Some(c) = me.state.change {
            onto.push(c.to);
        }
        if me.state.lane % 2 == 1 {
            onto.push(me.state.lane);
        }
    });
    onto.dedup();
    assert!(onto.is_empty(), "changed onto the bus lane: {onto:?}");
}

#[test]
fn r6l4_keep_right_still_goes_onto_an_open_lane_that_runs_on() {
    // the control: the bus lane's rules dropped from the last piece too, it runs on open:
    // the car keeps right onto it (now from the first lane whose own piece runs on 40 m)
    let mut net = bus_lane_with_a_ruleless_piece();
    net.lanes[5].density = 1.0;
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![keen(150.0)];
    let mut right = false;
    run(&net, &mut sim, &mut cars, 20.0, |_, _, cars| {
        let Some(me) = cars.iter().find(|c| c.id == 1) else { return };
        right |= me.state.lane == 5 && me.state.change.is_none();
    });
    assert!(right, "no keep-right onto the open lane");
}

#[test]
fn r6l4_keep_right_goes_onto_a_short_piece_of_a_lane_that_runs_on_open() {
    // (review r6l: 201 of 446 keep-right changes of viz5/viz4 went onto a target piece under
    // 40 m of a lane cut into short spline pieces) the bus lane opened on the last piece too:
    // the 21 m piece is taken as the lane it is, open for 400 m on
    let mut net = bus_lane_with_a_ruleless_piece();
    net.lanes[5].density = 1.0;
    let mut sim = Simulation::new(&net, vec![]);
    let mut me = keen(190.0);
    me.state.lane = 2;
    me.state.s = 1.0;
    let mut cars = vec![me];
    let mut first_target = None;
    run(&net, &mut sim, &mut cars, 1.0, |_, _, cars| {
        let Some(me) = cars.iter().find(|c| c.id == 1) else { return };
        if first_target.is_none() {
            first_target = me.state.change.map(|c| c.to);
        }
    });
    assert_eq!(first_target, Some(3), "no keep-right onto the 21 m piece");
}

#[test]
fn r6l4_a_change_whose_target_runs_on_into_a_closed_lane_is_given_up() {
    // a change begun on the rule-less piece, its target moving on into the closed one before
    // the body is half over: given up, back on the left lane
    let net = bus_lane_with_a_ruleless_piece();
    let mut me = keen(12.0);
    me.state.lane = 2;
    me.state.s = 12.0;
    me.params.lc_keep_right = 0.0;
    let mut sim = Simulation::new(&net, vec![]);
    sim.update_positions(&net, std::slice::from_mut(&mut me));
    me.state.start_change(&net, 3, 2);
    me.state.change.as_mut().unwrap().wait = 0.0;
    let mut cars = vec![me];
    let (mut on_closed, mut least_k_seen) = (false, 0.0f32);
    run(&net, &mut sim, &mut cars, 6.0, |_, _, cars| {
        let Some(me) = cars.iter().find(|c| c.id == 1) else { return };
        on_closed |= me.state.lane == 5;
        if let Some(c) = me.state.change {
            if c.to == 5 {
                least_k_seen = least_k_seen.max(c.t);
            }
        }
    });
    let me = cars.iter().find(|c| c.id == 1).unwrap();
    assert!(!on_closed, "ended on the closed lane");
    assert!(me.state.change.is_none() && me.state.lane % 2 == 0, "lane {} change {:?}", me.state.lane, me.state.change);
}

#[test]
fn r6l4_a_timetable_bus_takes_its_route_onto_the_closed_lane() {
    // a scheduled bus's route runs onto the bus lane: open anywhere its route goes
    let net = bus_lane_with_a_ruleless_piece();
    let mut bus = ai_bus(1, 0, 150.0, 10.0);
    bus.state.set_route(&net, vec![0, 2, 3, 5], 150.0);
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![bus];
    let mut reached = false;
    run(&net, &mut sim, &mut cars, 15.0, |_, _, cars| {
        let Some(me) = cars.iter().find(|c| c.id == 1) else { return };
        reached |= me.state.lane == 5;
    });
    assert!(reached, "the bus never got onto its route's bus lane");
}

#[test]
fn r6l4_a_strategic_change_into_a_junction_piece_stays_allowed() {
    // a double right turn: both lanes run into right-turn pieces of a crossing object (source
    // 2, no rules); the car on the left lane moves over for the turn as before
    let mut net = Network::default();
    let mut a = road(&[(0.0, 0.0), (200.0, 0.0)], 1);
    a.offset = 0.0;
    let mut b = road(&[(0.0, -3.5), (200.0, -3.5)], 1);
    b.key = Some(LaneKey { tile: (0, 0), id: 1, path: 1 });
    b.offset = 3.5;
    net.lanes.push(a);
    net.lanes.push(b);
    for (k, y) in [(0u16, 0.0), (1, -3.5)] {
        let mut j = lane(&[(200.0, y), (210.0, y - 2.0), (212.0, y - 12.0)], LaneKind::Street, 3.0);
        j.source = 2;
        j.turn = 2;
        j.key = Some(LaneKey { tile: (0, 0), id: 9, path: k });
        net.lanes.push(j);
    }
    let net = finish(net);
    let mut me = car(1, 0, 60.0, 10.0);
    me.state.change_cooldown = 0.0;
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars = vec![me];
    let mut changed = false;
    run(&net, &mut sim, &mut cars, 12.0, |_, _, cars| {
        let Some(me) = cars.iter().find(|c| c.id == 1) else { return };
        changed |= me.state.lane == 1 || me.state.lane == 3;
    });
    assert!(changed, "no strategic change for the turn");
}

/// A junction object's merge (viz5 case01, AnwaiDajie at object 384): the south-bound
/// carriageway piece (lane 1, after a lit stub, lane 0) and the ramp piece from its right
/// (lane 2, no light) run into one way out (lane 3). `ramp_priority`: the ramp piece's
/// `[rule] priority` (128 the default).
fn ramp_merge(ramp_priority: f32) -> Network {
    let piece = |pts: &[(f64, f64)], path: u16| {
        let mut l = lane(pts, LaneKind::Street, 3.0);
        l.source = 2;
        l.key = Some(LaneKey { tile: (0, 0), id: 384, path });
        l
    };
    let mut net = Network::default();
    let mut stub = piece(&[(0.0, 31.0), (0.0, 30.0)], 0);
    stub.traffic_light = Some((0, 0));
    net.lanes.push(stub); // 0
    net.lanes.push(piece(&[(0.0, 30.0), (0.0, 0.0)], 1)); // 1 carriageway
    let mut ramp = piece(&[(-20.0, 20.0), (-8.0, 8.0), (0.0, 0.0)], 2);
    ramp.priority = ramp_priority;
    net.lanes.push(ramp); // 2 ramp, from the carriageway's right
    net.lanes.push(road(&[(0.0, 0.0), (0.0, -200.0)], 5)); // 3 way out
    net.lanes.push(road(&[(0.0, 200.0), (0.0, 31.0)], 6)); // 4 carriageway in
    net.lanes.push(road(&[(-200.0, 200.0), (-20.0, 20.0)], 7)); // 5 ramp in
    finish(net)
}

/// Whether movement `a` gives way to `b` in the compiled junction.
fn yields(sim: &Simulation, a: usize, b: usize) -> Option<bool> {
    let (j, mv) = sim.junctions.movement(a)?;
    mv.foes.iter().find(|f| j.movements[f.movement].lane == b).map(|f| f.yields)
}

#[test]
fn r6l5_a_ramp_priority_rule_makes_the_unlit_ramp_give_way_and_an_unlit_line_alone_does_not() {
    // equal priorities: the ramp from the right goes first, stop line or not (no generic
    // "an unlit stop line gives way": unlit stop lines inside lit objects are common, viz5
    // case06's lane 5060 would yield its own green)
    let net = ramp_merge(128.0);
    let sim = Simulation::new(&net, vec![]);
    assert_eq!(yields(&sim, 1, 2), Some(true), "the carriageway gives way to the ramp from its right");
    assert_eq!(yields(&sim, 2, 1), Some(false));
    // the map's [rule] priority 64 on the ramp piece (the patch for AnwaiDajie's ramp): the
    // ramp gives way
    let net = ramp_merge(64.0);
    let sim = Simulation::new(&net, vec![]);
    assert_eq!(yields(&sim, 1, 2), Some(false), "the carriageway goes first");
    assert_eq!(yields(&sim, 2, 1), Some(true), "the ramp gives way");
}
