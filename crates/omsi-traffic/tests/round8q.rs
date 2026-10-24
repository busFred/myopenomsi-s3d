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
    let person = Person { pos: DVec3::new((tail - 2.0 - 2.5) as f64, -3.5, 0.0), vel: glam::DVec2::ZERO, on_crossing: false, still: 30.0, waiting_for_bus: false, at_kerb: false, waits_at: None, id: None, approach: None };
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

/// [Q8-1 fix pass 1] The tail of a packed queue of two (car 2) backs up; at the frame it first
/// backs faster than 0.5 m/s `inject` changes the scene. (How far it backed beyond the place fixed
/// when it began, the least distance from it to any other body, to any person, whether it backed.)
fn backing_with(inject: impl Fn(&mut Simulation, &mut Vec<Vehicle>, &Vehicle), secs: f32) -> (f32, f32, f32, bool) {
    let net = two_lanes();
    let mut cars = packed_queue(&net, 2, 2.0, 30.0);
    let mut sim = Simulation::new(&net, vec![]);
    sim.update_positions(&net, &mut cars);
    let (mut planned, mut done, mut over, mut least, mut least_p, mut backed) = (None::<f32>, false, 0.0f32, f32::INFINITY, f32::INFINITY, false);
    let mut t = 0.0;
    while t < secs {
        sim.tick(&net, DT, &mut cars);
        let Some(k) = cars.iter().position(|c| c.id == 2) else { break };
        let me = cars[k].clone();
        if planned.is_none() {
            planned = me.reversing.map(|r| r.0);
        }
        backed |= me.state.speed < -0.05;
        if !done && me.state.speed < -0.5 {
            inject(&mut sim, &mut cars, &me);
            done = true;
        }
        if let Some(p) = planned.filter(|_| me.reversing.is_some() || me.state.speed < 0.0) {
            over = over.max(p - me.state.odometer);
        }
        for o in cars.iter().filter(|o| o.id != 2) {
            least = least.min(me.obb().separation(&o.obb()) as f32);
        }
        for q in &sim.people {
            least_p = least_p.min(me.obb().separation(&omsi_sim::collision::Obb::point(q.pos, 0.28)) as f32);
        }
        t += DT;
    }
    (over, least, least_p, backed)
}

/// Where car `me`'s rear is, `d` metres further back along x (the lanes run along x).
fn behind(me: &Vehicle, d: f64) -> f64 {
    me.pos.x - me.state.rear as f64 - d
}

#[test]
fn q81_fix1_a_backing_car_stops_for_a_person_or_a_car_that_appears_behind_it() {
    // (safety B1: it held its speed and backed into them, -1.18 m and -1.80 m)
    let (over, _, least_p, backed) = backing_with(|sim, _, me| {
        sim.people.push(Person { pos: DVec3::new(behind(me, 1.5), -3.5, 0.0), vel: glam::DVec2::ZERO, on_crossing: false, still: 30.0, waiting_for_bus: false, at_kerb: false, waits_at: None, id: None, approach: None });
    }, 25.0);
    eprintln!("person 1.5 m behind: over {over:.2} least person {least_p:.2}");
    assert!(backed && least_p > 0.3, "backed into the person: {least_p:.2}");
    assert!(over <= 0.0, "backed {over:.2} m past the place fixed");
    let (over, least, _, backed) = backing_with(|_, cars, me| {
        let mut c = car(9, 1, (behind(me, 1.5) - 2.0) as f32, 0.0);
        c.pull_out = 1000.0;
        cars.push(c);
    }, 25.0);
    eprintln!("car 1.5 m behind: over {over:.2} least {least:.2}");
    assert!(backed && least > 0.3, "backed into the car: {least:.2}");
}

#[test]
fn q81_fix1_a_backing_car_stops_when_the_queue_moves_and_never_backs_past_the_place_fixed() {
    // (safety B1: the head left, it backed 6 m more at -0.9 m/s; x10 case12 #18778 15 m for 6.6)
    let (over, least, _, backed) = backing_with(|_, cars, _| {
        cars.retain(|c| c.id != 5);
    }, 25.0);
    eprintln!("head gone: over {over:.2} least {least:.2}");
    assert!(backed && over <= 0.0, "backed {over:.2} m past the place fixed");
}

#[test]
fn q81_fix1_followers_rolling_up_during_the_whole_backing_are_not_backed_into() {
    // (safety S2) a car, a bus and a motorbike rolling up behind the tail at 10 m/s from 40 m
    for (class, len, hw) in [(VehicleClass::Car, 4.0f32, 0.9f32), (VehicleClass::Bus, 12.0, 1.25), (VehicleClass::Car, 2.0, 0.4)] {
        let (over, least, _, backed) = backing_with(move |_, cars, me| {
            let mut st = AiState::new(1, (behind(me, 40.0) - len as f64 * 0.5) as f32, 9);
            st.speed = 10.0;
            st.front = len * 0.5;
            st.rear = len * 0.5;
            st.length = len;
            st.max_speed_kmh = 50.0;
            st.change_cooldown = 100.0;
            let mut f = Vehicle::new(9, class, st, hw, 9 * 7919);
            f.fresh = 0.0;
            cars.push(f);
        }, 30.0);
        eprintln!("{class:?} {len} m follower: backed {backed} over {over:.2} least {least:.2}");
        assert!(least > 0.3, "{class:?} {len} m: touched {least:.2}");
        assert!(over <= 0.0);
    }
}

#[test]
fn q81_fix1_no_backing_off_its_own_lane_or_onto_a_crossing() {
    // (safety B3) the tail stands 1 m from its lane's start (a junction piece behind it): no
    // backing over the lane's start; and a zebra where it would come to rest: no backing either
    let net = two_lanes();
    let mut cars = packed_queue(&net, 1, 2.0, 30.0);
    // (car 1 moved up to 1 m from the start of lane 1: a queue of one behind a bus parked there)
    cars[0].state.s = 2.0 + 2.0 + 6.0 + 1.0;
    cars[1].state.s = 2.0 + 1.0;
    let mut sim = Simulation::new(&net, vec![]);
    let mut backed = false;
    run(&net, &mut sim, &mut cars, 20.0, |_, _, cars| {
        backed |= cars.iter().any(|c| c.id == 1 && c.state.speed < -0.05);
    });
    assert!(!backed, "backed off its own lane");
    let mut net = two_lanes();
    // (a crossing cut 3 m behind the tail's rear, ped crossing on lane 1)
    let tail = packed_queue(&net, 1, 2.0, 30.0)[1].state.s;
    net.walks[1].push((0, tail - 2.0 - 3.0, 0.0));
    let mut cars = packed_queue(&net, 1, 2.0, 30.0);
    let mut sim = Simulation::new(&net, vec![]);
    let mut backed = false;
    run(&net, &mut sim, &mut cars, 20.0, |_, _, cars| {
        backed |= cars.iter().any(|c| c.id == 1 && c.state.speed < -0.05);
    });
    assert!(!backed, "backed onto a zebra band");
}

#[test]
fn q81_fix1_a_player_held_by_its_red_or_indicating_heads_no_queue_to_back_out_of() {
    // (safety B2: behind the player standing at a red with its indicator on, the seventh car backed
    // up 2 s after the green and never went round) the player's bus stands where the bus stood:
    // indicating, no car backs up; not indicating and held by nothing, the tail does
    for blinker in [1u8, 0] {
        let net = two_lanes();
        let mut cars = packed_queue(&net, 2, 2.0, 30.0);
        cars.retain(|c| c.id != 5);
        let mut sim = Simulation::new(&net, vec![]);
        sim.players.push(omsi_traffic::PlayerView { id: 0, pos: DVec3::new(203.0, -3.5, 0.0), heading: 90.0, half_len: 6.0, half_width: 1.25, speed: 0.0, still: 60.0, blinker, signalling: 30.0, priority: false });
        let mut backed = false;
        run(&net, &mut sim, &mut cars, 20.0, |_, _, cars| {
            backed |= cars.iter().any(|c| c.state.speed < -0.05);
        });
        eprintln!("player blinker {blinker}: backed {backed}");
        assert_eq!(backed, blinker == 0, "blinker {blinker}");
    }
}

#[test]
fn q81_fix2_a_queue_behind_a_head_at_a_red_does_not_back_up_at_the_green() {
    // (safety recheck S1: viz case12 head #45205 stood at its red until the green and read 57 s
    // "held by nothing" there; #45372 backed up 0.74 s before the queue moved) the head's standing
    // is SUMO's waiting time, which the red paused: at the green it is no queue head
    let mut net = two_lanes();
    net.lanes[3].traffic_light = Some((0, 0));
    net.lanes[2].traffic_light = Some((0, 0));
    let ctl = omsi_sim::traffic::TrafficLightController::new(vec![vec![(0, 60.0), (6, 60.0)]], 120.0);
    let mut sim = Simulation::new(&net, vec![ctl]);
    let mut cars = vec![car(1, 1, 120.0, 10.0), car(2, 1, 80.0, 10.0), car(3, 1, 40.0, 10.0)];
    // (one lane's queue: no change of their own on the way up; the escape reads no cooldown)
    for c in cars.iter_mut() {
        c.state.change_cooldown = 1000.0;
    }
    let (mut back, mut green_seen) = (false, false);
    run(&net, &mut sim, &mut cars, 75.0, |t, _, cars| {
        back |= cars.iter().any(|c| c.state.speed < -0.01 || c.reversing.is_some());
        green_seen |= t > 61.0 && cars.iter().any(|c| c.state.speed > 1.0);
    });
    assert!(green_seen, "the queue never moved off at the green");
    assert!(!back, "a car backed up at the green");
}

#[test]
fn q81_fix3_a_head_let_go_by_people_is_not_a_head_on_the_frame_it_is_let_go() {
    // (safety recheck pass 2, S2: viz3 case10 #26738 backed on the frame its head #26737, held by
    // people, was let go) the head's standing for good is its continuous time held by nothing
    let net = two_lanes();
    let mut cars = vec![car(1, 1, 150.0, 0.0), car(2, 1, 144.0, 0.0), car(3, 1, 138.0, 0.0)];
    for c in cars.iter_mut() {
        c.stopped = 60.0;
        c.waiting = 60.0;
        c.state.change_cooldown = 1000.0;
    }
    let mut sim = Simulation::new(&net, vec![]);
    sim.people.push(Person { pos: DVec3::new(155.0, -3.5, 0.0), vel: glam::DVec2::ZERO, on_crossing: false, still: 60.0, waiting_for_bus: false, at_kerb: false, waits_at: None, id: None, approach: None });
    let (mut back, mut people_held) = (false, false);
    let mut t = 0.0;
    // (released once all have stood the escape's 25 s)
    while t < 55.0 {
        if (t - 40.0f32).abs() < DT * 0.5 {
            sim.people.clear();
        }
        sim.tick(&net, DT, &mut cars);
        people_held |= cars[0].last.why == omsi_traffic::Why::People;
        back |= cars.iter().any(|c| c.state.speed < -0.01 || c.reversing.is_some());
        t += DT;
    }
    assert!(people_held, "the head was never held by the person");
    assert!(!back, "a car backed up when the head was let go");
}

#[test]
fn q81_fix3_a_player_whose_way_leads_into_a_red_heads_no_queue() {
    // (safety recheck pass 2, S1: the player stood 40 m short of its light with no claim, its own
    // lane unlit, and its clock ran on) the lane its way leads into is lit red: no head
    let mut net = two_lanes();
    net.lanes[3].traffic_light = Some((0, 0));
    net.lanes[2].traffic_light = Some((0, 0));
    let ctl = omsi_sim::traffic::TrafficLightController::new(vec![vec![(0, 1000.0)]], 1000.0);
    // (the player's bus stands 60 m short of the line, two cars packed behind it)
    let mut cars = vec![car(1, 1, 140.0, 0.0), car(2, 1, 134.0, 0.0)];
    for c in cars.iter_mut() {
        c.stopped = 30.0;
        c.state.change_cooldown = 0.0;
    }
    let mut sim = Simulation::new(&net, vec![ctl]);
    sim.players.push(omsi_traffic::PlayerView { id: 0, pos: DVec3::new(150.0, -3.5, 0.0), heading: 90.0, half_len: 6.0, half_width: 1.25, speed: 0.0, still: 60.0, blinker: 0, signalling: 0.0, priority: false });
    let mut back = false;
    run(&net, &mut sim, &mut cars, 50.0, |_, _, cars| {
        back |= cars.iter().any(|c| c.state.speed < -0.05);
    });
    assert!(!back, "backed up behind a player whose way leads into its red");
}

#[test]
fn j82_a_car_that_goes_at_amber_onset_and_then_slows_stops_at_the_line() {
    // [J8-2] (viz3 case18 #52423) at the amber onset 26 m from the line at 13.4 m/s it cannot stop
    // within 3 m/s² and goes; a person on the road short of the line makes it brake; able to stop
    // then, it stops at the line when the person has gone (SUMO re-tests `yellow && canBrake` every
    // step), never over it at red
    let mut net = two_lanes();
    net.lanes[3].traffic_light = Some((0, 0));
    // (green 10 s, amber 3 s, red)
    let ctl = omsi_sim::traffic::TrafficLightController::new(vec![vec![(6, 10.0), (10, 3.0), (0, 60.0)]], 73.0);
    let mut sim = Simulation::new(&net, vec![ctl]);
    // (the line at x 210: at 13.4 m/s, 26 m out after 10 s: start 10 s * 13.4 m further back)
    let mut me = car(1, 1, 210.0 - 2.0 - 26.0 - 134.0, 13.4);
    me.state.max_speed_kmh = 48.24;
    me.state.change_cooldown = 1000.0;
    let mut cars = vec![me];
    // (somebody on the road 6 m short of the line from the amber onset to 1 s into the red)
    let mut over_at_red = false;
    let mut t = 0.0f32;
    while t < 30.0 {
        if (t - 10.0).abs() < DT * 0.5 {
            sim.people.push(Person { pos: DVec3::new(204.0, -3.5, 0.0), vel: glam::DVec2::ZERO, on_crossing: false, still: 60.0, waiting_for_bus: false, at_kerb: false, waits_at: None, id: None, approach: None });
        }
        if (t - 14.0).abs() < DT * 0.5 {
            sim.people.clear();
        }
        sim.tick(&net, DT, &mut cars);
        let c = &cars[0];
        let front_x = c.pos.x + c.state.front as f64;
        over_at_red |= t > 13.0 && c.state.lane == 1 && front_x > 210.0 + 0.05;
        t += DT;
    }
    assert!(!over_at_red, "crossed its line at red");
}

#[test]
fn q81_followup_a_player_blocked_by_a_vehicle_ahead_heads_no_queue() {
    // (safety final recheck) the player's bus 2 m behind an AI car held where it stands (pulling
    // out), indicator off, no light: it is held by its lead as an AI head would be (`Plan::held`'s
    // lead near), so the two cars packed behind it do not back up
    let net = two_lanes();
    let mut held = car(9, 1, 203.0, 0.0);
    held.pull_out = 1000.0;
    held.stopped = 60.0;
    let mut cars = vec![held, car(1, 1, 183.0, 0.0), car(2, 1, 177.0, 0.0)];
    for c in cars.iter_mut().skip(1) {
        c.stopped = 30.0;
        c.state.change_cooldown = 0.0;
    }
    let mut sim = Simulation::new(&net, vec![]);
    sim.players.push(omsi_traffic::PlayerView { id: 0, pos: DVec3::new(193.0, -3.5, 0.0), heading: 90.0, half_len: 6.0, half_width: 1.25, speed: 0.0, still: 60.0, blinker: 0, signalling: 0.0, priority: false });
    let mut back = false;
    run(&net, &mut sim, &mut cars, 30.0, |_, _, cars| {
        back |= cars.iter().any(|c| c.state.speed < -0.05);
    });
    assert!(!back, "backed up behind a player held by the car in front of it");
}
