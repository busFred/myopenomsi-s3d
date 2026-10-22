//! Round 8 area Q8 (queues and lane use): the packed-queue escape (Q8-1), the bypass inside a
//! junction piece (Q8-2), LC2013's keep-right layer (Q8-3), J7's stop placement after attribution
//! (Q8-4).
use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network};
use omsi_traffic::{Person, Simulation, Vehicle, VehicleClass};

const DT: f32 = 1.0 / 30.0;

fn lane(pts: &[(f64, f64)], kind: LaneKind, width: f32) -> omsi_sim::traffic::Lane {
    LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), kind, width)
}
fn road(pts: &[(f64, f64)], id: i64, path: u16) -> omsi_sim::traffic::Lane {
    let mut l = lane(pts, LaneKind::Street, 3.0);
    l.source = 1;
    l.key = Some(LaneKey { tile: (0, 0), id, path });
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

/// Two same-direction lanes, 0 left at y=0 and 1 right at y=-3.5, 0..210 m, running on into 2
/// and 3 to 600 m.
fn two_lanes() -> Network {
    let mut net = Network::default();
    let mut a = road(&[(0.0, 0.0), (210.0, 0.0)], 1, 0);
    a.offset = 0.0;
    let mut b = road(&[(0.0, -3.5), (210.0, -3.5)], 1, 1);
    b.offset = 3.5;
    b.drivable = (1.5, 1.5);
    let mut a2 = road(&[(210.0, 0.0), (600.0, 0.0)], 2, 0);
    a2.offset = 0.0;
    let mut b2 = road(&[(210.0, -3.5), (600.0, -3.5)], 2, 1);
    b2.offset = 3.5;
    b2.drivable = (1.5, 1.5);
    net.lanes.extend([a, b, a2, b2]);
    finish(net)
}


/// Car 1 on the left lane at 12 m/s; `slow`: a car 45 m ahead of it on the right lane at that
/// speed. (When it began to move onto the right lane, the most its keep-right wish reached, and
/// whether it was past the slower car then.)
fn keep_right_run(slow: Option<f32>, secs: f32) -> (Option<f32>, f32, bool) {
    let net = two_lanes();
    let mut sim = Simulation::new(&net, vec![]);
    let mut me = car(1, 0, 20.0, 12.0);
    me.state.change_cooldown = 0.0;
    me.state.max_speed_kmh = 43.2;
    let mut cars = vec![me];
    if let Some(v) = slow {
        let mut o = car(2, 1, 69.0, v);
        o.state.max_speed_kmh = v * 3.6;
        o.state.change_cooldown = 100.0;
        cars.push(o);
    }
    let (mut right, mut most, mut past) = (None, 0.0f32, true);
    run(&net, &mut sim, &mut cars, secs, |t, _, cars| {
        let Some(me) = cars.iter().find(|c| c.id == 1) else { return };
        most = most.max(me.keep_right);
        if right.is_none() && (me.state.lane == 1 || me.state.lane == 3 || me.state.change.is_some_and(|c| c.to == 1 || c.to == 3)) {
            right = Some(t);
            past = cars.iter().find(|c| c.id == 2).is_none_or(|o| o.pos.x + 2.0 < me.pos.x - 2.0);
        }
    });
    (right, most, past)
}

#[test]
fn q83_keep_right_is_lc2013s_accumulated_wish_not_a_distance_gate() {
    // SUMO LC2013's keepRight layer: the wish accumulates by the share of the acceptance time the
    // driver could keep its speed on the lane beside, over KEEP_RIGHT_TIME; the change comes once
    // it reaches the threshold - not at once, and not while a slower car there would soon have to
    // be overtaken (round 5 asked for 70 m free there and 60 m to the own lead)
    let (free, _, _) = keep_right_run(None, 30.0);
    let (slow, most, past) = keep_right_run(Some(6.0), 30.0);
    eprintln!("keep right: free lane at {free:?}; slower car 45 m ahead there: {slow:?} (wish {most:.2}, past it {past})");
    let t = free.expect("never kept right onto a free lane");
    assert!(t >= omsi_traffic::plan::KEEP_RIGHT_TIME, "kept right at {t:.1} s, before the wish could accumulate");
    // (it overtakes the slower car first, then keeps right in front of it)
    assert!(slow.is_some() && past, "kept right before passing the slower car ({slow:?})");
}

/// A timetable bus (id 5) standing for good at the end of its route on the right lane (front at
/// x 209.5: `Why::End`, the T11 #44422 of viz5 case08), and `n` cars packed behind it at `gap` m
/// (ids 1.., car 1 nearest), stood `stood` s.
fn packed_queue(net: &Network, n: u64, gap: f32, stood: f32) -> Vec<Vehicle> {
    let mut st = AiState::new(1, 203.0, 5);
    st.front = 6.0;
    st.rear = 6.0;
    st.length = 12.0;
    st.max_speed_kmh = 60.0;
    st.veh_type = -1;
    st.route = vec![1];
    st.route_index = 0;
    let mut b = Vehicle::new(5, VehicleClass::ScheduledBus, st, 1.25, 5 * 7919);
    b.fresh = 0.0;
    b.stopped = 300.0;
    let _ = net;
    let mut cars = vec![b];
    let mut front = 203.0 - 6.0 - gap;
    for k in 0..n {
        let mut c = car(1 + k, 1, front - 2.0, 0.0);
        c.stopped = stood;
        c.state.change_cooldown = 0.0;
        cars.push(c);
        front -= 4.0 + gap;
    }
    cars
}

/// What a run did: per car id 1..=n when it was on the left lane, its least gap to any body, the
/// most it backed up (m), and whether it moved backwards at all.
struct Escape {
    round: Vec<Option<f32>>,
    least: f32,
    backed: Vec<f32>,
    least_person: f32,
}

fn escape(net: &Network, mut cars: Vec<Vehicle>, people: Vec<Person>, n: u64, secs: f32) -> Escape {
    let mut sim = Simulation::new(net, vec![]);
    sim.people = people;
    // (the bodies are placed by the first frame: start from there)
    sim.update_positions(net, &mut cars);
    let start: Vec<f64> = (1..=n).map(|id| cars.iter().find(|c| c.id == id).map_or(0.0, |c| c.pos.x)).collect();
    let mut e = Escape { round: vec![None; n as usize], least: f32::INFINITY, backed: vec![0.0; n as usize], least_person: f32::INFINITY };
    run(net, &mut sim, &mut cars, secs, |t, sim, cars| {
        for k in 0..n {
            let Some(me) = cars.iter().find(|c| c.id == 1 + k) else { continue };
            e.backed[k as usize] = e.backed[k as usize].max((start[k as usize] - me.pos.x) as f32);
            if e.round[k as usize].is_none() && (me.state.lane == 0 || me.state.lane == 2) && me.state.change.is_none() {
                e.round[k as usize] = Some(t);
            }
            for o in cars.iter().filter(|o| o.id != me.id) {
                e.least = e.least.min(me.obb().separation(&o.obb()) as f32);
            }
            for q in &sim.people {
                e.least_person = e.least_person.min(me.obb().separation(&omsi_sim::collision::Obb::point(q.pos, 0.28)) as f32);
            }
        }
    });
    e
}

#[test]
fn q81_a_packed_queue_behind_a_vehicle_standing_for_good_dissolves_from_the_back() {
    // viz5 case08: five cars 1.9-2.3 m apart behind T11 #44422 standing at the end of its track
    // for 11 minutes, none able to steer round. Now the rearmost backs up to the room the
    // bypass needs and goes round, then the next, the queue dissolving from the back within
    // 60 s; nobody is touched
    let net = two_lanes();
    // (the bodies are placed on the first frame: start positions from a settled first tick)
    let cars = packed_queue(&net, 3, 2.0, 30.0);
    let e = escape(&net, cars, vec![], 3, 60.0);
    eprintln!("packed: round {:?} backed {:?} least {:.2}", e.round, e.backed, e.least);
    assert!(e.least > 0.3, "came {:.2} m to a body", e.least);
    for (k, r) in e.round.iter().enumerate() {
        assert!(r.is_some_and(|t| t < 60.0), "car {} never got round: {:?}", k + 1, e.round);
    }
    // (rearmost first, each backing up to the room the bypass needs)
    assert!(e.round[2] <= e.round[1] && e.round[1] <= e.round[0], "not from the back: {:?}", e.round);
    assert!(e.backed.iter().all(|&b| b > 1.0), "went round without backing up: {:?}", e.backed);
}

#[test]
fn q81_a_bus_at_its_stop_is_kept_back_from_not_backed_away_from() {
    // a bus serving its stop, doors open: the cars behind keep back (round 6), none backs up
    let net = two_lanes();
    let mut cars = packed_queue(&net, 2, 2.0, 30.0);
    cars[0].external.at_stop = true;
    cars[0].external.standing_for = 60.0;
    cars[0].state.route.clear();
    cars[0].pull_out = 1000.0;
    let e = escape(&net, cars, vec![], 2, 30.0);
    eprintln!("at stop: backed {:?}", e.backed);
    assert!(e.backed.iter().all(|&b| b < 0.05), "backed up behind a bus at its stop: {:?}", e.backed);
}

#[test]
fn q81_no_backing_up_into_a_person_or_a_car_behind() {
    // somebody standing behind the rearmost car, or a car standing 3 m behind it: no backing up
    // into them (the strip behind must hold the move and the standing gap)
    let net = two_lanes();
    let cars = packed_queue(&net, 2, 2.0, 30.0);
    let tail = cars.iter().find(|c| c.id == 2).map(|c| c.state.s).unwrap();
    let person = Person { pos: DVec3::new((tail - 2.0 - 2.5) as f64, -3.5, 0.0), vel: glam::DVec2::ZERO, on_crossing: false, still: 30.0, waiting_for_bus: false, at_kerb: false };
    let e = escape(&net, cars, vec![person], 2, 20.0);
    eprintln!("person behind: backed {:?} least person {:.2}", e.backed, e.least_person);
    assert!(e.backed[1] < 0.05, "backed up towards the person: {:?}", e.backed);
    assert!(e.least_person > 0.3);
    let mut cars = packed_queue(&net, 2, 2.0, 30.0);
    let mut c = car(9, 1, tail - 4.0 - 3.0, 0.0);
    c.stopped = 30.0;
    c.pull_out = 1000.0;
    cars.push(c);
    let e = escape(&net, cars, vec![], 2, 20.0);
    eprintln!("car behind: backed {:?} least {:.2}", e.backed, e.least);
    assert!(e.least > 0.3, "came {:.2} m to a body", e.least);
}

#[test]
fn q81_a_red_light_queue_never_backs_up() {
    // the right lane ends at a red at x 210: a queue packed behind it never backs up
    let mut net = two_lanes();
    net.lanes[3].traffic_light = Some((0, 0));
    net.lanes[2].traffic_light = Some((0, 0));
    let ctl = omsi_sim::traffic::TrafficLightController::new(vec![vec![(0, 1000.0)]], 1000.0);
    let mut sim = Simulation::new(&net, vec![ctl]);
    let mut cars = vec![car(1, 1, 120.0, 10.0), car(2, 1, 80.0, 10.0), car(3, 1, 40.0, 10.0)];
    for c in cars.iter_mut() {
        c.state.change_cooldown = 0.0;
    }
    let mut back = false;
    run(&net, &mut sim, &mut cars, 90.0, |_, _, cars| {
        for c in cars.iter().filter(|c| c.state.speed < -0.01 || c.reversing.is_some()) {
            if !back {
                eprintln!("#{} backs up: why {:?} lead {:?} s {:.1}; all: {:?}", c.id, c.last.why, c.last.lead_id, c.state.s, cars.iter().map(|o| (o.id, o.last.why, o.state.s, o.stopped, o.last.held)).collect::<Vec<_>>());
            }
            back = true;
        }
    });
    assert!(!back, "a red-light queue backed up");
    // (it was one: the head at its red, the others standing behind it)
    assert_eq!(cars[0].last.why, omsi_traffic::Why::Light);
    assert!(cars.iter().all(|c| c.stopped > 30.0), "{:?}", cars.iter().map(|c| c.stopped).collect::<Vec<_>>());
}
