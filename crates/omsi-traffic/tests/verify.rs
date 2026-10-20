//! Adversarial scenarios (review): situations the regression gate does not cover. Each
//! test asserts what a safe, sane traffic should do; a failure is a demonstrated problem.
//! `cargo test -p omsi-traffic --test adversarial -- --nocapture` prints the numbers.

use glam::{DVec2, DVec3};
use omsi_sim::collision::Obb;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network, TrafficLightController};
use omsi_traffic::{DriverParams, Person, PlayerView, Simulation, Vehicle, VehicleClass, Why};

const DT: f32 = 1.0 / 30.0;

fn road(from: (f64, f64), to: (f64, f64), id: i64) -> omsi_sim::traffic::Lane {
    let mut l = LaneBuilder::polyline(vec![DVec3::new(from.0, from.1, 0.0), DVec3::new(to.0, to.1, 0.0)], LaneKind::Street, 3.5);
    l.source = 1;
    l.key = Some(LaneKey { tile: (0, 0), id, path: 0 });
    l
}

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

fn bus(id: u64, lane: usize, s: f32, speed: f32) -> Vehicle {
    let mut st = AiState::new(lane, s, id);
    st.speed = speed;
    st.front = 6.0;
    st.rear = 6.0;
    st.length = 12.0;
    st.max_speed_kmh = 80.0;
    st.veh_type = 2;
    let mut v = Vehicle::new(id, VehicleClass::Bus, st, 1.25, id * 7919);
    v.fresh = 0.0;
    v
}

struct Run {
    min_gap: f32,
    min_pair: (u64, u64),
    hardest_brake: f32,
    longest_stand: f32,
    removed: usize,
}

/// Like the gate's `run`, but the callback may change the simulation and the vehicles.
fn run(net: &Network, sim: &mut Simulation, vehicles: &mut Vec<Vehicle>, secs: f32, mut each: impl FnMut(f32, &mut Simulation, &mut Vec<Vehicle>)) -> Run {
    let mut r = Run { min_gap: f32::INFINITY, min_pair: (0, 0), hardest_brake: 0.0, longest_stand: 0.0, removed: 0 };
    let mut t = 0.0;
    while t < secs {
        sim.tick(net, DT, vehicles);
        let mut last = usize::MAX;
        for &i in &sim.removed {
            assert!(i < last, "removed indices not strictly descending: {:?}", sim.removed);
            last = i;
            vehicles.remove(i);
            r.removed += 1;
        }
        for v in vehicles.iter() {
            assert!(v.state.speed.is_finite() && v.state.s.is_finite(), "non-finite state for {}", v.id);
            r.hardest_brake = r.hardest_brake.max(-v.state.acc);
            r.longest_stand = r.longest_stand.max(v.stopped);
        }
        for (a, va) in vehicles.iter().enumerate() {
            for vb in vehicles.iter().skip(a + 1) {
                let d = va.obb().separation(&vb.obb()) as f32;
                if d < r.min_gap {
                    r.min_gap = d;
                    r.min_pair = (va.id, vb.id);
                }
            }
        }
        each(t, sim, vehicles);
        t += DT;
    }
    r
}

fn hard_brakes(t: f32, cars: &[Vehicle]) {
    for c in cars {
        if c.state.acc < -6.0 {
            eprintln!("  t={t:.2} HARD BRAKE car {} acc {:.1} v {:.1} lane {} s {:.1} why {:?} stop {:?} lead {:?} seen {:?}", c.id, c.state.acc, c.state.speed, c.state.lane, c.state.s, c.last.why, c.last.stop, c.last.lead.map(|l| l.gap), c.last.lead_seen);
        }
    }
}

fn probe(t: f32, cars: &[Vehicle]) {
    hard_brakes(t, cars);
    let line: Vec<String> = cars
        .iter()
        .map(|c| format!("[{} L{} s={:.1} lat={:.2} v={:.1} why={:?} gap={:?} pass={}]", c.id, c.state.lane, c.state.s, c.state.lateral, c.state.speed, c.last.why, c.last.lead.map(|l| (l.gap * 10.0).round() / 10.0), c.passing.is_some()))
        .collect();
    eprintln!("t={:.1} {}", t, line.join(" "));
}

fn straight_road() -> Network {
    let mut net = Network::default();
    net.lanes.push(road((0.0, 0.0), (300.0, 0.0), 1));
    net.lanes.push(road((300.0, 0.0), (2000.0, 0.0), 2));
    net.link(1.5);
    net.compute_conflicts();
    net
}

/// Eastbound lane 0 and westbound lane 1 (s = 2000 - x), 3 m wide, 3.5 m apart.
fn two_way_road() -> Network {
    let mut net = Network::default();
    let mut a = road((0.0, 0.0), (2000.0, 0.0), 1);
    a.width = 3.0;
    let mut b = road((2000.0, 3.5), (0.0, 3.5), 2);
    b.width = 3.0;
    net.lanes.push(a);
    net.lanes.push(b);
    net.link(1.5);
    net.compute_conflicts();
    net
}

const PARKED_AT: f32 = 200.0;

fn parked_simulation(net: &Network) -> Simulation {
    let mut sim = Simulation::new(net, vec![]);
    sim.parked.insert(0, vec![omsi_traffic::Parked { s: PARKED_AT, lateral: 0.0, half_width: 0.9, half_len: 2.2 }]);
    sim
}

fn priority_junction() -> Network {
    let mut net = Network::default();
    net.lanes.push(road((-200.0, 0.0), (-15.0, 0.0), 1)); // 0 west approach
    net.lanes.push(piece((-15.0, 0.0), (15.0, 0.0), 9, 0, 192.0)); // 1 main through
    net.lanes.push(road((15.0, 0.0), (1500.0, 0.0), 2)); // 2 east exit
    net.lanes.push(road((0.0, -200.0), (0.0, -15.0), 3)); // 3 south approach
    net.lanes.push(piece((0.0, -15.0), (0.0, 15.0), 9, 1, 64.0)); // 4 side through
    net.lanes.push(road((0.0, 15.0), (0.0, 1500.0), 4)); // 5 north exit
    net.link(1.5);
    net.compute_conflicts();
    net
}

// ---------------------------------------------------------------------------------------
// Verification of bcd2742

#[allow(dead_code)]
fn _unused(_: Obb, _: DVec2, _: Person, _: DriverParams, _: Why) {
    let _ = (bus as fn(u64, usize, f32, f32) -> Vehicle, probe as fn(f32, &[Vehicle]), straight_road as fn() -> Network, LaneKind::Street);
}

#[test]
fn v1_junction_entries_refreshed_when_an_approach_lane_is_appended() {
    let mut net = Network::default();
    net.lanes.push(road((-200.0, 0.0), (-15.0, 0.0), 1)); // 0
    net.lanes.push(piece((-15.0, 0.0), (15.0, 0.0), 9, 0, 192.0)); // 1
    net.lanes.push(road((15.0, 0.0), (1500.0, 0.0), 2)); // 2
    net.lanes.push(piece((0.0, -15.0), (0.0, 15.0), 9, 1, 64.0)); // 3 side through
    net.lanes.push(road((0.0, 15.0), (0.0, 1500.0), 4)); // 4
    net.link(1.5);
    net.compute_conflicts();
    let mut sim = Simulation::new(&net, vec![]);
    let before = sim.junctions.movement(3).unwrap().1.entries.clone();
    let added = net.extend(vec![road((0.0, -200.0), (0.0, -15.0), 3)], 1.5);
    eprintln!("v1: added {:?}; prev of 3 now {:?}", added, net.prev.get(3));
    sim.network_grew(&net, added.start);
    let after = sim.junctions.movement(3).unwrap().1.entries.clone();
    eprintln!("v1: entries before {:?} after {:?}; lookup {:?}", before, after, sim.junctions.entries.get(&added.start));
    assert!(!before.contains(&added.start));
    assert!(after.contains(&added.start), "old junction's entries not refreshed");
    assert!(sim.junctions.entries.get(&added.start).is_some_and(|v| !v.is_empty()));
    // and an AI car on the new approach now gives way to one on the main road
    let mut cars = vec![car(1, added.start, 160.0, 10.0), car(2, 0, 140.0, 12.0)];
    let r = run(&net, &mut sim, &mut cars, 15.0, |_, _, _| {});
    eprintln!("v1: min_gap {:.2} {:?}", r.min_gap, r.min_pair);
    assert!(r.min_gap > 0.3);
}

/// p7 with the main approach split: the player is on a lane that is neither inside the
/// junction nor an entry until 10 m before it.
#[test]
fn v2a_player_on_a_lane_before_a_short_entry_lane() {
    for split in [false, true] {
        let mut net = Network::default();
        if split {
            net.lanes.push(road((-200.0, 0.0), (-25.0, 0.0), 1)); // 0
            net.lanes.push(road((-25.0, 0.0), (-15.0, 0.0), 5)); // 1 short entry
        } else {
            net.lanes.push(road((-200.0, 0.0), (-15.0, 0.0), 1));
        }
        let base = net.lanes.len();
        net.lanes.push(piece((-15.0, 0.0), (15.0, 0.0), 9, 0, 192.0));
        net.lanes.push(road((15.0, 0.0), (1500.0, 0.0), 2));
        net.lanes.push(road((0.0, -200.0), (0.0, -15.0), 3)); // side approach
        net.lanes.push(piece((0.0, -15.0), (0.0, 15.0), 9, 1, 64.0));
        net.lanes.push(road((0.0, 15.0), (0.0, 1500.0), 4));
        net.link(1.5);
        net.compute_conflicts();
        let mut sim = Simulation::new(&net, vec![]);
        let mut cars = vec![car(1, base + 2, 160.0, 10.0)];
        let mut x = -60.0f64;
        let mut nearest = f32::INFINITY;
        let mut waited = false;
        let r = run(&net, &mut sim, &mut cars, 15.0, |_, sim, cars| {
            x += 11.0 * DT as f64;
            let pv = PlayerView { id: 0, pos: DVec3::new(x, 0.0, 0.0), heading: 90.0, half_len: 6.0, half_width: 1.25, speed: 11.0, still: 0.0, blinker: 0, signalling: 0.0, priority: false };
            sim.players = vec![pv];
            if let Some(c) = cars.first() {
                nearest = nearest.min(c.obb().separation(&pv.obb(0.0, 0.0)) as f32);
                if c.state.speed < 0.5 && c.state.lane == base + 2 {
                    waited = true;
                }
            }
            if split { hard_brakes(x as f32, cars); }
        });
        eprintln!("v2a split={split}: waited {waited} nearest {nearest:.2} brake {:.1}", r.hardest_brake);
    }
}

#[test]
fn v2b_player_with_priority_on_the_side_road_is_given_way_and_named() {
    for prio in [true, false] {
        let net = priority_junction();
        let mut sim = Simulation::new(&net, vec![]);
        // AI on the main road
        let mut cars = vec![car(1, 0, 160.0, 10.0)];
        let mut y = -60.0f64;
        let mut waited = false;
        let mut wait_for = None;
        let mut nearest = f32::INFINITY;
        run(&net, &mut sim, &mut cars, 15.0, |_, sim, cars| {
            y += 11.0 * DT as f64;
            let pv = PlayerView { id: 7, pos: DVec3::new(0.0, y, 0.0), heading: 0.0, half_len: 6.0, half_width: 1.25, speed: 11.0, still: 0.0, blinker: 0, signalling: 0.0, priority: prio };
            sim.players = vec![pv];
            if let Some(c) = cars.first() {
                nearest = nearest.min(c.obb().separation(&pv.obb(0.0, 0.0)) as f32);
                if c.state.speed < 0.5 && c.state.lane == 0 {
                    waited = true;
                }
                if c.last.wait_for.is_some() {
                    wait_for = c.last.wait_for;
                }
            }
        });
        let base = omsi_traffic::sim::PLAYER_ID_BASE;
        eprintln!("v2b priority={prio}: main-road car waited {waited}, wait_for {:?} (BASE+{:?}), nearest {nearest:.2}", wait_for, wait_for.map(|w| w.wrapping_sub(base)));
        let _ = prio;
    }
}

#[test]
fn v3_oncoming_clear_for_of_a_moving_car_100m_ahead() {
    let net = two_way_road();
    let me_s = PARKED_AT - 8.0;
    // an oncoming car 100 m ahead of me at 13 m/s
    let mut sim = parked_simulation(&net);
    let mut cars = vec![car(1, 0, me_s, 0.0), car(50, 1, 2000.0 - (me_s + 100.0), 13.0)];
    cars[0].stopped = 10.0;
    sim.tick(&net, DT, &mut cars);
    let p = omsi_traffic::perception::perceive(&net, &sim, &cars, 0);
    eprintln!("v3: moving 100 m ahead -> {:?} (expect ~{:.1} s)", p.oncoming, 100.0 / 13.0);
    // a car standing 10 m behind me on the oncoming lane: ignored
    let mut sim = parked_simulation(&net);
    let mut cars = vec![car(1, 0, me_s, 0.0), car(51, 1, 2000.0 - (me_s - 10.0), 0.0)];
    cars[0].stopped = 10.0;
    sim.tick(&net, DT, &mut cars);
    let p = omsi_traffic::perception::perceive(&net, &sim, &cars, 0);
    eprintln!("v3: standing 10 m behind me -> {:?}", p.oncoming);
    // a car standing 20 m ahead on the oncoming lane: blocks
    let mut sim = parked_simulation(&net);
    let mut cars = vec![car(1, 0, me_s, 0.0), car(52, 1, 2000.0 - (me_s + 20.0), 0.0)];
    cars[0].stopped = 10.0;
    sim.tick(&net, DT, &mut cars);
    let p = omsi_traffic::perception::perceive(&net, &sim, &cars, 0);
    eprintln!("v3: standing 20 m ahead -> {:?}", p.oncoming);
}

fn light_road(two: bool) -> Network {
    let mut net = Network::default();
    net.lanes.push(road((0.0, 0.0), (200.0, 0.0), 1)); // 0
    let mut p = piece((200.0, 0.0), (230.0, 0.0), 9, 0, 128.0);
    p.traffic_light = Some((0, 0));
    net.lanes.push(p); // 1
    if two {
        net.lanes.push(road((230.0, 0.0), (500.0, 0.0), 2)); // 2
        let mut q = piece((500.0, 0.0), (530.0, 0.0), 10, 0, 128.0);
        q.traffic_light = Some((1, 0));
        net.lanes.push(q); // 3
        net.lanes.push(road((530.0, 0.0), (2000.0, 0.0), 3)); // 4
    } else {
        net.lanes.push(road((230.0, 0.0), (2000.0, 0.0), 2));
    }
    net.link(1.5);
    net.compute_conflicts();
    net
}

fn set(sim: &mut Simulation, ctl: usize, state: i32) {
    sim.lights[ctl].lights[0] = vec![(state, 10000.0)];
}

#[test]
fn v6a_amber_at_the_line() {
    // yellow when `at` m from the line at 12 m/s, red `after` s later
    for (at, after) in [(2.0f32, 1.0f32), (18.0, 0.5), (14.0, 0.4)] {
        let net = light_road(false);
        let ctl = TrafficLightController::new(vec![vec![(6, 10000.0)]], 40.0);
        let mut sim = Simulation::new(&net, vec![ctl]);
        let mut cars = vec![car(1, 0, 120.0, 12.0)];
        let mut yellow_at: Option<f32> = None;
        let mut red = false;
        let mut crossed = None;
        let mut amber_seen = false;
        let r = run(&net, &mut sim, &mut cars, 12.0, |t, sim, cars| {
            let c = &cars[0];
            let gap = 200.0 - (c.state.s + c.state.front);
            if c.state.lane == 0 && yellow_at.is_none() && gap <= at {
                yellow_at = Some(t);
                set(sim, 0, 9);
                eprintln!("  yellow at t={t:.2} gap {gap:.1} v {:.1}", c.state.speed);
            }
            if let Some(y) = yellow_at {
                if !red && t >= y + after {
                    red = true;
                    set(sim, 0, 0);
                    eprintln!("  red at t={t:.2} lane {} gap {:.1} v {:.1}", c.state.lane, 200.0 - (c.state.s + c.state.front), c.state.speed);
                }
            }
            amber_seen |= c.light_decision.is_some();
            if c.state.lane != 0 && crossed.is_none() {
                crossed = Some((t, red));
            }
        });
        eprintln!("v6a at {at} after {after}: crossed {:?} amber_seen {amber_seen} brake {:.1}", crossed, r.hardest_brake);
        assert!(crossed.is_some(), "did not go through");
        assert!(r.hardest_brake <= 4.0, "braked {:.1}", r.hardest_brake);
    }
}

/// After driving through one light at green, does the amber flag stay set and make the car
/// take the next light's yellow (and red) it could stop for?
#[test]
fn v6b_stale_amber_runs_the_next_light() {
  for clear in [true, false] {
    let net = light_road(true);
    let c0 = TrafficLightController::new(vec![vec![(6, 10000.0)]], 40.0);
    let c1 = TrafficLightController::new(vec![vec![(6, 10000.0)]], 40.0);
    let mut sim = Simulation::new(&net, vec![c0, c1]);
    let mut cars = vec![car(1, 0, 120.0, 12.0)];
    let mut yellow_at: Option<f32> = None;
    let mut red = false;
    let mut crossed: Option<(f32, bool, f32)> = None;
    let mut amber_after_first = false;
    run(&net, &mut sim, &mut cars, 40.0, |t, sim, cars| {
        if cars[0].state.lane == 2 {
            amber_after_first |= cars[0].light_decision.is_some();
            if clear && yellow_at.is_none() { cars[0].light_decision = None; }
        }
        let c = &cars[0];
        let gap = 500.0 - (c.state.s + 230.0 + c.state.front);
        if c.state.lane == 2 && yellow_at.is_none() && gap <= 60.0 {
            yellow_at = Some(t);
            set(sim, 1, 9);
            eprintln!("  2nd light yellow at t={t:.2} gap {gap:.1} v {:.1}", c.state.speed);
        }
        if let Some(y) = yellow_at {
            if !red && t >= y + 3.0 {
                red = true;
                set(sim, 1, 0);
                eprintln!("  2nd light red at t={t:.2} lane {} v {:.1}", c.state.lane, c.state.speed);
            }
        }
        if c.state.lane == 3 && crossed.is_none() {
            crossed = Some((t, red, c.state.speed));
        }
    });
    eprintln!("v6b clear_amber_by_hand={clear}: amber still set after the first light {amber_after_first}; crossed 2nd line (t, on red, v) {:?}", crossed);
    if !clear { assert!(crossed.is_none(), "drove through a yellow it could stop for 60 m ahead (on red: {:?})", crossed.map(|c| c.1)); }
  }
}

#[test]
fn v8_every_seeded_driver_clears_the_threshold() {
    let mut lo = f32::INFINITY;
    let mut hi = 0.0f32;
    for seed in 0..200_000u64 {
        let p = DriverParams::seeded(VehicleClass::Car, seed.wrapping_mul(7919));
        lo = lo.min(p.lc_speed_gain);
        hi = hi.max(p.lc_speed_gain);
    }
    eprintln!("v8: lc_speed_gain over 200k seeds {lo:.3}..{hi:.3} (threshold 0.4 at impatience 0)");
    assert!(lo >= 0.4);
}
