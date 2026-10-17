//! Recording the traffic and replaying it: a scenario run with the recorder on, read back
//! and replayed headless from its first keyframe, must drive as it did (the simulation is
//! deterministic: the same inputs and steps give the same positions).

use glam::{DVec2, DVec3};
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network, TrafficLightController};
use omsi_traffic::record::{Header, PersonRecord, Recorder, Steer};
use omsi_traffic::replay::Replay;
use omsi_traffic::{Parked, Person, PlayerView, Simulation, Vehicle, VehicleClass};
use std::borrow::Cow;
use std::path::PathBuf;

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

/// A main road along x with a side road from the south, meeting in a crossing object.
fn priority_junction() -> Network {
    let mut net = Network::default();
    net.lanes.push(road((-200.0, 0.0), (-15.0, 0.0), 1)); // 0 west approach
    net.lanes.push(piece((-15.0, 0.0), (15.0, 0.0), 9, 0, 192.0)); // 1 main through
    net.lanes.push(road((15.0, 0.0), (1500.0, 0.0), 2)); // 2 east exit
    net.lanes.push(road((0.0, -200.0), (0.0, -15.0), 3)); // 3 south approach
    net.lanes.push(piece((0.0, -15.0), (0.0, 15.0), 9, 1, 64.0)); // 4 side through
    net.lanes.push(road((0.0, 15.0), (0.0, 1500.0), 4)); // 5 north exit
    net.link(1.5);
    net
}

/// The same with lights on both ways through.
fn signalled_junction() -> (Network, Vec<TrafficLightController>) {
    let mut net = priority_junction();
    net.lanes[1].traffic_light = Some((0, 0));
    net.lanes[4].traffic_light = Some((0, 1));
    net.lanes[1].priority = 128.0;
    net.lanes[4].priority = 128.0;
    let ctl = TrafficLightController::new(vec![vec![(6, 15.0), (0, 15.0)], vec![(0, 15.0), (6, 15.0)]], 30.0);
    (net, vec![ctl])
}

fn temp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("omsi-replay-{}-{name}.rec", std::process::id()))
}

/// What the game does between frames in the run below.
type Hook<'a> = &'a mut dyn FnMut(f32, &mut Network, &mut Simulation, &mut Vec<Vehicle>, &mut Recorder);

/// Run `secs` with the recorder on (into `path` from the start), as the game runs a frame:
/// the inputs, the record, the frame, the vehicles at a dead end taken off.
fn record(mut net: Network, lights: Vec<TrafficLightController>, mut cars: Vec<Vehicle>, secs: f32, path: &PathBuf, hook: Hook) -> (Recorder, Network, Simulation) {
    let mut sim = Simulation::new(&net, lights);
    let mut rec = Recorder::new();
    rec.start_full(path, &net, &sim, "test").expect("the recording starts");
    let mut t = 0.0;
    while t < secs {
        hook(t, &mut net, &mut sim, &mut cars, &mut rec);
        let people: Vec<PersonRecord> = sim
            .people
            .iter()
            .map(|p| PersonRecord { state: Cow::Borrowed("WalkStreet"), decision: Some(Cow::Borrowed("go")), ..PersonRecord::from_person(p) })
            .collect();
        let steer: Vec<(u64, Steer)> = cars.iter().filter(|c| c.id == 1).map(|c| (c.id, Steer::of(&c.state))).collect();
        rec.frame(&sim, DT, &cars, &people, &steer, &|i| format!("car {}", cars[i].id));
        sim.tick(&net, DT, &mut cars);
        for &i in &sim.removed {
            cars.swap_remove(i);
        }
        t += DT;
    }
    rec.stop_full();
    rec.flush();
    (rec, net, sim)
}

/// Replay `path` from its first keyframe to the end: every keyframe within 5 cm.
fn replays_within(path: &PathBuf, keyframes_at_least: usize) -> Replay {
    let mut r = Replay::load(path).expect("the recording reads");
    assert!(r.skipped.is_empty(), "lines not read: {:?}", r.skipped.first());
    assert!(!r.truncated);
    assert!(r.keyframe_count() >= keyframes_at_least, "{} keyframes", r.keyframe_count());
    assert!(r.at(0));
    let mut frames = 0;
    while r.step() {
        frames += 1;
    }
    assert!(frames > 100, "{frames} frames replayed");
    assert!(r.divergence.len() + 1 >= keyframes_at_least, "{} keyframes compared", r.divergence.len());
    for &(t, ds, id) in &r.divergence {
        assert!(ds < 0.05, "at {t:.1} s vehicle {id} is {ds:.3} m from where the game had it");
    }
    assert!(r.missing.is_empty(), "vehicles missing at keyframes: {:?}", r.missing);
    r
}

#[test]
fn a_recorded_junction_replays_as_it_ran() {
    let net = priority_junction();
    let mut cars = Vec::new();
    for k in 0..4 {
        cars.push(car(10 + k, 0, 150.0 - 30.0 * k as f32, 13.0));
    }
    cars.push(car(1, 3, 150.0, 10.0));
    let path = temp("junction");
    let mut hook = |t: f32, _: &mut Network, sim: &mut Simulation, cars: &mut Vec<Vehicle>, rec: &mut Recorder| {
        // a car comes onto the road, one leaves it, a player stands at the side road
        if (t - 8.0).abs() < DT * 0.5 {
            cars.push(car(20, 3, 60.0, 8.0));
        }
        if (t - 14.0).abs() < DT * 0.5 {
            cars.retain(|c| c.id != 11);
        }
        // the game changes a car's way between frames
        if (t - 17.0).abs() < DT * 0.5 {
            if let Some(c) = cars.iter_mut().find(|c| c.id == 12) {
                c.state.max_speed_kmh = 40.0;
                rec.vehicle_changed(sim.time, c);
            }
        }
        sim.players = vec![PlayerView { id: 0, pos: DVec3::new(6.0, -40.0, 0.0), heading: 180.0, half_len: 6.0, half_width: 1.25, speed: 0.0, still: t, blinker: 0, signalling: 0.0, priority: false }];
    };
    record(net, vec![], cars, 30.0, &path, &mut hook);
    let r = replays_within(&path, 6);
    // (the measure is not blind: replayed with longer steps, the cars get elsewhere)
    let mut off = Replay::new(r.header.clone(), r.records.clone(), Vec::new(), false);
    for rec in &mut off.records {
        if let omsi_traffic::record::Record::Frame(f) = rec {
            f.dt *= 1.1;
        }
    }
    assert!(off.at(0));
    while off.step() {}
    assert!(off.worst().is_some_and(|w| w.1 > 0.5), "a different step went unnoticed: {:?}", off.worst());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn lights_people_parked_cars_and_new_lanes_replay_too() {
    let (net, lights) = signalled_junction();
    let cars = vec![car(1, 0, 120.0, 12.0), car(2, 0, 90.0, 12.0), car(3, 3, 140.0, 10.0), car(4, 3, 110.0, 10.0)];
    let path = temp("lights");
    let mut hook = |t: f32, net: &mut Network, sim: &mut Simulation, cars: &mut Vec<Vehicle>, rec: &mut Recorder| {
        let _ = cars;
        // somebody crosses the east exit for a while
        sim.people = if (3.0..12.0).contains(&t) {
            vec![Person { pos: DVec3::new(60.0, -4.0 + (t as f64 - 3.0) * 0.9, 0.0), vel: DVec2::new(0.0, 0.9), on_crossing: true, still: 0.0, waiting_for_bus: false }]
        } else {
            Vec::new()
        };
        if (t - 6.0).abs() < DT * 0.5 {
            sim.parked.insert(2, vec![Parked { s: 300.0, lateral: 2.5, half_width: 0.9, half_len: 2.3 }]);
        }
        // a tile brings a crossing object at the end of the east exit (a way through it
        // that meets nobody: its first conflict is at infinity, which a header must keep)
        if (t - 9.0).abs() < DT * 0.5 {
            let first = net.lanes.len();
            let new = vec![piece((1500.0, 0.0), (1530.0, 0.0), 7, 0, 128.0)];
            rec.lanes_added(sim.time, first, &new);
            net.extend(new, 1.5);
            sim.network_grew(net, first);
        }
    };
    let (mut rec, net, sim) = record(net, lights, cars, 75.0, &path, &mut hook);
    let r = replays_within(&path, 14);
    assert_eq!(r.net.lanes.len(), net.lanes.len(), "the lanes added were replayed");
    // the memory holds the last minute (from the keyframe before it): saved, it replays
    // the same way, with the parked cars of before it and the network of now
    let ring = temp("ring");
    rec.dump_ring(&net, &sim, "test", &ring, "manual").expect("the minute is saved");
    rec.flush();
    let mut r2 = replays_within(&ring, 12);
    assert_eq!(r2.header.trigger.as_deref(), Some("manual"));
    assert!(r2.keyframe_time(0).is_some_and(|t| (t - 15.0).abs() < 0.1), "the minute starts at {:?}", r2.keyframe_time(0));
    assert!(r2.at(0));
    assert_eq!(r2.sim.parked.get(&2).map(|p| p.len()), Some(1), "the parked cars of before the minute");
    assert_eq!(r2.net.lanes.len(), net.lanes.len());
    assert!(r2.sim.junctions.nodes.iter().any(|j| j.movements.iter().any(|m| m.conflict_start == f32::INFINITY)), "an infinity in the header read back");
    // a fixture cut from it replays too
    let fixture = temp("fixture");
    r.scenario(12.0, 30.0, &fixture).expect("the fixture is written");
    let mut f = Replay::load(&fixture).expect("the fixture reads");
    assert!(f.at(0));
    assert!(f.keyframe_time(0).is_some_and(|t| t <= 12.0 && t > 5.0));
    while f.step() {}
    assert!(f.divergence.iter().all(|d| d.1 < 0.05), "{:?}", f.worst());
    assert_eq!(f.net.lanes.len(), net.lanes.len());
    for p in [path, ring, fixture] {
        let _ = std::fs::remove_file(p);
    }
}

#[test]
fn a_header_reads_back_as_it_was_written() {
    let (net, lights) = signalled_junction();
    let h = Header { version: omsi_traffic::record::VERSION, map: "test".into(), left_hand: false, lanes: net.lanes.clone(), lights, started_day_time: 8.0 * 3600.0, trigger: None, derived: None, switches: None };
    let text = serde_json::to_string(&h).expect("written");
    let back: Header = serde_json::from_str(&text).expect("read");
    assert_eq!(serde_json::to_string(&back).expect("written again"), text);
    assert_eq!(back.lanes.len(), net.lanes.len());
    for (a, b) in back.lanes.iter().zip(&net.lanes) {
        assert_eq!(a.points, b.points);
        assert_eq!(a.headings, b.headings);
        assert_eq!(a.dist, b.dist);
        assert_eq!(a.next, b.next);
        assert_eq!(a.key, b.key);
        assert_eq!(a.traffic_light, b.traffic_light);
        assert_eq!(a.priority, b.priority);
    }
}

/// The recorder's cost per frame at 300 vehicles and 200 people (`cargo test --release -p
/// omsi-traffic --test replay -- --ignored --nocapture`).
#[test]
#[ignore]
fn the_recorder_costs_little_per_frame() {
    let mut net = Network::default();
    for r in 0..30 {
        let y = r as f64 * 20.0;
        net.lanes.push(road((0.0, y), (1000.0, y), 100 + r));
        net.lanes.push(road((1000.0, y), (3000.0, y), 200 + r));
    }
    net.link(1.5);
    let mut sim = Simulation::new(&net, vec![TrafficLightController::new(vec![vec![(6, 15.0), (0, 15.0)]; 4], 30.0); 40]);
    let mut cars: Vec<Vehicle> = (0..300).map(|k| car(k as u64 + 1, (k % 30) * 2, 20.0 + (k / 30) as f32 * 40.0, 10.0)).collect();
    sim.people = (0..200).map(|k| Person { pos: DVec3::new(k as f64 * 3.1, -5.0, 0.0), vel: DVec2::new(1.1, 0.0), on_crossing: false, still: 0.0, waiting_for_bus: false }).collect();
    let mut people: Vec<PersonRecord> = sim.people.iter().map(|p| PersonRecord { state: Cow::Borrowed("WalkStreet"), leg: Some((3, 12.5, 40.0)), decision: Some(Cow::Borrowed("go")), ..PersonRecord::from_person(p) }).collect();
    sim.tick(&net, DT, &mut cars);
    let mut rec = Recorder::new();
    let file = std::env::temp_dir().join(format!("omsi-replay-{}-cost.rec", std::process::id()));
    rec.start_full(&file, &net, &sim, "test").expect("the recording starts");
    let frames = 600;
    let (mut spent, mut worst) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
    for _ in 0..frames {
        let t0 = std::time::Instant::now();
        rec.frame(&sim, DT, &cars, &people, &[], &|_| "car".into());
        spent += t0.elapsed();
        worst = worst.max(t0.elapsed());
        // (everybody moves: the file's size is that of a real frame's)
        sim.tick(&net, DT, &mut cars);
        for (p, r) in sim.people.iter_mut().zip(&mut people) {
            p.pos += (p.vel * DT as f64).extend(0.0);
            r.set_person(p);
        }
    }
    rec.stop_full();
    rec.flush();
    let ms = spent.as_secs_f64() * 1000.0 / frames as f64;
    let kb = std::fs::metadata(&file).map(|m| m.len()).unwrap_or(0) as f64 / 1024.0;
    let _ = std::fs::remove_file(&file);
    println!("the file: {:.0} KiB per second of traffic at {} frames a second", kb / (frames as f64 * DT as f64), (1.0 / DT).round());
    println!("recorder.frame: {ms:.3} ms per frame, at most {:.3} ms (a keyframe; 300 vehicles, 200 people)", worst.as_secs_f64() * 1000.0);
}

/// Replay a recording made in the game and say how it went: `OMSI_REPLAY=<file.rec> cargo
/// test --release -p omsi-traffic --test replay -- --ignored a_recording_from_the_game
/// --nocapture` (`OMSI_REPLAY_FROM=<keyframe>` starts later).
#[test]
#[ignore]
fn a_recording_from_the_game_replays() {
    use omsi_traffic::record::Record;
    let Some(path) = std::env::var_os("OMSI_REPLAY") else { return };
    let mut r = Replay::load(std::path::Path::new(&path)).expect("the recording reads");
    let mut kinds: std::collections::BTreeMap<&str, (usize, usize)> = Default::default();
    for rec in &r.records {
        let k = match rec {
            Record::Frame(_) => "frame",
            Record::Keyframe { .. } => "keyframe",
            Record::Spawn { .. } => "spawn",
            Record::Remove { .. } => "remove",
            Record::VehicleChanged { .. } => "vehicle changed",
            Record::LanesAdded { .. } => "lanes added",
            Record::LightsAdded { .. } => "lights added",
            Record::Parked { .. } => "parked",
            Record::Note { .. } => "note",
        };
        let e = kinds.entry(k).or_default();
        e.0 += 1;
        e.1 += serde_json::to_string(rec).map(|s| s.len()).unwrap_or(0);
    }
    println!("{}: map {}, {} lanes, {} lights, trigger {:?}, {} lines unread, cut short {}", path.to_string_lossy(), r.header.map, r.header.lanes.len(), r.header.lights.len(), r.header.trigger, r.skipped.len(), r.truncated);
    println!("records (count, bytes): {kinds:?}; keyframes from {:?} to {:?} s", r.keyframe_time(0), r.keyframe_count().checked_sub(1).and_then(|k| r.keyframe_time(k)));
    let from = std::env::var("OMSI_REPLAY_FROM").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    assert!(r.at(from), "no keyframe {from}");
    while r.step() {}
    for (t, ds, id) in &r.divergence {
        println!("  {t:7.1} s: {ds:.4} m (vehicle {id})");
    }
    println!("worst {:?}, {} missing", r.worst(), r.missing.len());
}

/// A free run keeps its own vehicles at later keyframes (and still measures how far they
/// are from the recorded ones); a normal replay goes back to the recorded ones.
#[test]
fn a_free_run_keeps_its_vehicles_at_keyframes() {
    let net = priority_junction();
    let cars = vec![car(10, 0, 150.0, 13.0), car(11, 0, 100.0, 13.0), car(1, 3, 150.0, 10.0)];
    let path = temp("free");
    let mut hook = |t: f32, _: &mut Network, sim: &mut Simulation, cars: &mut Vec<Vehicle>, _: &mut Recorder| {
        if (t - 8.0).abs() < DT * 0.5 {
            cars.push(car(20, 3, 60.0, 8.0));
        }
        sim.players = vec![PlayerView { id: 0, pos: DVec3::new(6.0, -40.0, 0.0), heading: 180.0, half_len: 6.0, half_width: 1.25, speed: 0.0, still: t, blinker: 0, signalling: 0.0, priority: false }];
    };
    record(net, vec![], cars, 20.0, &path, &mut hook);
    let base = Replay::load(&path).expect("the recording reads");
    let _ = std::fs::remove_file(&path);
    // the game had car 11 five metres further on at the second keyframe
    let mut records = base.records.clone();
    let second = records.iter().enumerate().filter(|(_, r)| matches!(r, omsi_traffic::record::Record::Keyframe { .. })).nth(1).map(|(i, _)| i).expect("two keyframes");
    let kt = records[second].time();
    if let omsi_traffic::record::Record::Keyframe { vehicles, .. } = &mut records[second] {
        vehicles.iter_mut().find(|v| v.id == 11).expect("car 11").state.s += 5.0;
    }
    let run = |free: bool| {
        let mut r = Replay::new(base.header.clone(), records.clone(), Vec::new(), false);
        r.free_run = free;
        assert!(r.at(0));
        r.run_until(kt + 0.05);
        let s = r.vehicles.iter().find(|v| v.id == 11).expect("car 11 is on the road").state.s;
        let worst = r.divergence.first().copied().expect("the keyframe was compared");
        (s, worst, r)
    };
    let (s_reset, d_reset, _) = run(false);
    let (s_free, d_free, mut r) = run(true);
    assert!((d_reset.1 - 5.0).abs() < 0.1 && d_reset.2 == 11, "{d_reset:?}");
    assert!((d_free.1 - 5.0).abs() < 0.1 && d_free.2 == 11, "{d_free:?}");
    assert!((s_reset - s_free - 5.0).abs() < 0.5, "reset {s_reset}, free {s_free}");
    // the spawn at 8 s did not duplicate anything, and the run goes on to the end
    while r.step() {}
    assert_eq!(r.vehicles.iter().filter(|v| v.id == 20).count(), 1);
    // switches are kept across `at`
    r.set_switches(omsi_traffic::sim::Switches::from_list("pass_room,kerb_wait").unwrap());
    assert!(r.at(0));
    assert_eq!(r.sim.switches.list(), "pass_room,kerb_wait");
}

/// A free run that has moved on does not put a spawn onto a vehicle (it waits for room, and
/// gives up after a while) and does not snap a re-routed vehicle back to where the game had it.
#[test]
fn a_free_run_defers_spawns_and_keeps_changed_vehicles_where_they_are() {
    use omsi_traffic::record::Record;
    let net = priority_junction();
    let cars = vec![car(10, 0, 150.0, 13.0), car(1, 3, 150.0, 10.0)];
    let path = temp("defer");
    let mut hook = |t: f32, _: &mut Network, sim: &mut Simulation, cars: &mut Vec<Vehicle>, rec: &mut Recorder| {
        if (t - 8.0).abs() < DT * 0.5 {
            cars.push(car(20, 3, 60.0, 8.0));
        }
        if (t - 10.0).abs() < DT * 0.5 {
            if let Some(c) = cars.iter_mut().find(|c| c.id == 10) {
                c.state.max_speed_kmh = 40.0;
                rec.vehicle_changed(sim.time, c);
            }
        }
        sim.players = vec![PlayerView { id: 0, pos: DVec3::new(6.0, -40.0, 0.0), heading: 180.0, half_len: 6.0, half_width: 1.25, speed: 0.0, still: t, blinker: 0, signalling: 0.0, priority: false }];
    };
    record(net, vec![], cars, 30.0, &path, &mut hook);
    let base = Replay::load(&path).expect("the recording reads");
    let _ = std::fs::remove_file(&path);
    let mut records = base.records.clone();
    // the game spawned car 20 at 8 s; here two more spawns at the same moment and place,
    // cars 30 and 31; car 10's change, as the game had it 50 m further on; and a change of
    // a vehicle that is not on the road (99)
    let spawn = records.iter().position(|r| matches!(r, Record::Spawn { vehicle, .. } if vehicle.id == 20)).expect("the spawn");
    let (time, v20) = match &records[spawn] {
        Record::Spawn { time, vehicle, .. } => (*time, vehicle.clone()),
        _ => unreachable!(),
    };
    for id in [31, 30] {
        let mut v = v20.clone();
        v.id = id;
        records.insert(spawn + 1, Record::Spawn { time, vehicle: v, type_name: "car".into() });
    }
    let changed = records.iter().position(|r| matches!(r, Record::VehicleChanged { .. })).expect("the change");
    let ct = records[changed].time();
    let mut far = match &records[changed] {
        Record::VehicleChanged { vehicle, .. } => vehicle.clone(),
        _ => unreachable!(),
    };
    far.state.s += 50.0;
    let far_s = far.state.s;
    let mut v99 = far.clone();
    v99.id = 99;
    v99.state.lane = 3;
    v99.state.s = 30.0;
    records[changed] = Record::VehicleChanged { time: ct, vehicle: far };
    records.insert(changed + 1, Record::VehicleChanged { time: ct, vehicle: v99 });
    // (the game's order of vehicles would take off those it does not know)
    for rec in &mut records {
        if let Record::Frame(f) = rec {
            f.order = None;
        }
    }
    // (and the free run is off from the recording at the keyframe before: car 1 was
    // further on in the game)
    let kf = records.iter().position(|r| matches!(r, Record::Keyframe { time, .. } if *time > 4.0 && *time < 6.0)).expect("a keyframe at 5 s");
    if let Record::Keyframe { vehicles, .. } = &mut records[kf] {
        vehicles.iter_mut().find(|v| v.id == 1).expect("car 1").state.s += 3.0;
    }
    let make = |free: bool, patience: f64| {
        let mut r = Replay::new(base.header.clone(), records.clone(), Vec::new(), false);
        r.free_run = free;
        r.spawn_patience = patience;
        assert!(r.at(0));
        r
    };
    let has = |r: &Replay, id: u64| r.vehicles.iter().any(|v| v.id == id);
    let overlaps = |r: &mut Replay| {
        r.sim.update_positions(&r.net, &mut r.vehicles);
        (0..r.vehicles.len()).flat_map(|i| (i + 1..r.vehicles.len()).map(move |j| (i, j))).filter(|&(i, j)| r.vehicles[i].obb().overlaps_plan(&r.vehicles[j].obb())).count()
    };
    // in step with the recording a free run puts spawns where the recording does
    let mut r = Replay::new(base.header.clone(), base.records.clone(), Vec::new(), false);
    r.free_run = true;
    assert!(r.at(0));
    r.run_until(time + 0.2);
    assert!(r.in_sync && r.deferred_spawns == 0);
    // as before when it is not a free run: everything where the recording says
    let mut r = make(false, 30.0);
    r.run_until(time + 0.2);
    assert!(has(&r, 30) && has(&r, 31));
    assert_eq!((r.deferred_spawns, r.dropped_spawns), (0, 0));
    // a free run: 30 and 31 wait for car 20 to drive off, 99 (not on the road) comes at its
    // place, car 10 stays where it is and takes the new way
    let mut r = make(true, 30.0);
    r.run_until(time + 0.2);
    assert!(has(&r, 20) && !has(&r, 30) && !has(&r, 31), "spawned onto others");
    assert_eq!(overlaps(&mut r), 0);
    assert_eq!(r.deferred_spawns, 2);
    r.run_until(ct + 0.1);
    assert!(has(&r, 99), "a change of a vehicle not on the road is a spawn");
    let c10 = r.vehicles.iter().find(|v| v.id == 10).unwrap();
    assert!(c10.state.s < far_s - 20.0 || c10.state.lane != 0, "car 10 snapped to where the game had it: s {}", c10.state.s);
    assert!((c10.state.max_speed_kmh - 40.0).abs() < 1e-3, "it took the new way: {}", c10.state.max_speed_kmh);
    r.run_until(time + 15.0);
    assert!(has(&r, 30) && has(&r, 31), "30: {}, 31: {}", has(&r, 30), has(&r, 31));
    assert_eq!(overlaps(&mut r), 0, "30 and 31 were put onto others");
    while r.step() {}
    assert_eq!((r.deferred_spawns, r.dropped_spawns), (2, 0));
    // with no patience they are given up
    let mut r = make(true, 0.5);
    while r.step() {}
    assert!(!has(&r, 30) && !has(&r, 31));
    assert_eq!((r.deferred_spawns, r.dropped_spawns), (2, 2));
    assert!(r.pending_spawns.is_empty());
}

/// The recordings made in the game: `OMSI_TRAFFIC_FIXTURES` or `~/.openomsi/traffic/viz`.
fn fixture(name: &str) -> Option<PathBuf> {
    let dir = std::env::var_os("OMSI_TRAFFIC_FIXTURES").map(PathBuf::from).or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".openomsi/traffic/viz")))?;
    let p = dir.join(name);
    p.exists().then_some(p)
}

/// Free run with every rule as recorded must be the recording: the first 30 s of the
/// smallest fixture (`cargo test --release -p omsi-traffic --test replay free_run --
/// --nocapture`; skipped when the fixtures are not on this machine).
#[test]
fn a_free_run_without_switches_reproduces_a_game_recording() {
    let Some(path) = fixture("case01-0814.rec") else {
        eprintln!("no fixture case01-0814.rec: skipped");
        return;
    };
    let mut r = Replay::load(&path).expect("the fixture reads");
    r.free_run = true;
    r.set_switches(Default::default());
    assert!(r.at(0));
    let t0 = r.sim.time;
    r.run_until(t0 + 30.0);
    assert!(r.divergence.len() >= 5, "{} keyframes compared", r.divergence.len());
    let worst = r.divergence.iter().map(|d| d.1).fold(0.0f32, f32::max);
    println!("free run, first 30 s of case01: {} keyframes, worst {:.4} m, {} vehicles missing", r.divergence.len(), worst, r.missing.len());
    for (t, ds, id) in &r.divergence {
        println!("  {:6.1} s: {ds:.4} m (vehicle {id})", t - t0);
    }
    assert!(worst < 0.01, "the free run drifted {worst:.3} m from the game in 30 s: {:?}", r.worst());
    assert!(r.missing.is_empty(), "{:?}", r.missing.first());
}

/// The standing hysteresis of `competitor_filter` (`Vehicle::claim_standing`) is the
/// vehicle's: a replay started at any keyframe has it as the game had it, and drives on
/// as the game did. (It lived in the simulation, and a replay started at a keyframe began
/// with nobody standing: a car creeping off at 0.76 m/s after a stand was taken for a
/// moving one; viz2 case07 drifted 20 m by the next keyframe.)
#[test]
fn the_claim_standing_hysteresis_is_kept_with_the_vehicle_across_keyframes() {
    let net = priority_junction();
    // a queue on the side road at the main road, a stream on the main road
    let mut cars = Vec::new();
    for k in 0..5 {
        cars.push(car(10 + k, 0, 170.0 - 25.0 * k as f32, 12.0));
    }
    for k in 0..4 {
        cars.push(car(30 + k, 3, 182.0 - 7.0 * k as f32, 0.0));
    }
    let path = temp("standing");
    let mut hook = |t: f32, _: &mut Network, sim: &mut Simulation, cars: &mut Vec<Vehicle>, _: &mut Recorder| {
        sim.switches.competitor_filter = true;
        if (t - 9.0).abs() < DT * 0.5 {
            cars.push(car(20, 0, 60.0, 12.0));
        }
    };
    record(net, vec![], cars, 40.0, &path, &mut hook);
    let base = Replay::load(&path).expect("the recording reads");
    let _ = std::fs::remove_file(&path);
    assert_eq!(base.header.version, omsi_traffic::record::VERSION);
    // the keyframes carry the mark
    let marked = |rec: &omsi_traffic::record::Record| matches!(rec, omsi_traffic::record::Record::Keyframe { vehicles, .. } if vehicles.iter().any(|v| v.claim_standing));
    assert!(base.records.iter().any(marked), "nobody was standing at a claim");
    // (and somebody creeps off with it at a keyframe: where it matters)
    let creeping = |rec: &omsi_traffic::record::Record| matches!(rec, omsi_traffic::record::Record::Keyframe { vehicles, .. } if vehicles.iter().any(|v| v.claim_standing && v.state.speed >= 0.3));
    assert!(base.records.iter().any(creeping), "nobody creeps off at a keyframe");
    // started at every keyframe in turn, a replay drives as the game did up to the next one
    let n = base.keyframe_count();
    assert!(n >= 6, "{n} keyframes");
    for k in 0..n - 1 {
        let mut r = Replay::new(base.header.clone(), base.records.clone(), Vec::new(), false);
        r.free_run = true;
        r.set_switches(omsi_traffic::sim::Switches { competitor_filter: true, ..Default::default() });
        assert!(r.at(k));
        r.run_until(base.keyframe_time(k + 1).unwrap() + 0.01);
        let d = r.divergence.first().copied().expect("the next keyframe was compared");
        assert!(d.1 == 0.0, "started at keyframe {k}: {d:?}");
    }
}

/// A vehicle written before `claim_standing` existed reads as not standing.
#[test]
fn a_vehicle_without_the_standing_mark_reads_as_not_standing() {
    let mut v = car(5, 0, 10.0, 0.5);
    v.claim_standing = true;
    let mut json: serde_json::Value = serde_json::to_value(&v).unwrap();
    assert_eq!(json["claim_standing"], serde_json::Value::Bool(true));
    let back: Vehicle = serde_json::from_value(json.clone()).unwrap();
    assert!(back.claim_standing);
    json.as_object_mut().unwrap().remove("claim_standing");
    let old: Vehicle = serde_json::from_value(json).unwrap();
    assert!(!old.claim_standing);
}

/// The header keeps the switches the game ran with (stub_links as its network was linked);
/// the replay runs with them, lanes added later linked the same way, unless told otherwise.
#[test]
fn the_recording_keeps_its_switches_and_the_replay_runs_with_them() {
    let mut net = priority_junction();
    net.stub_links = true;
    let mut sim = Simulation::new(&net, vec![]);
    sim.switches.competitor_filter = true;
    let path = temp("switches");
    let mut rec = Recorder::new();
    rec.start_full(&path, &net, &sim, "test").expect("the recording starts");
    let mut cars = vec![car(10, 0, 100.0, 12.0)];
    for _ in 0..200 {
        rec.frame(&sim, DT, &cars, &[], &[], &|i| format!("car {}", cars[i].id));
        sim.tick(&net, DT, &mut cars);
    }
    rec.stop_full();
    rec.flush();
    let r = Replay::load(&path).expect("the recording reads");
    let _ = std::fs::remove_file(&path);
    assert_eq!(r.header.switches.as_deref(), Some("competitor_filter,stub_links"));
    assert_eq!(r.switches.list(), "competitor_filter,stub_links");
    assert!(r.net.stub_links, "lanes added later are linked with stub_links");
    // told otherwise
    let mut off = Replay::new(r.header.clone(), r.records.clone(), Vec::new(), false);
    off.set_switches(Default::default());
    assert!(off.at(0));
    assert!(!off.net.stub_links && off.sim.switches.list().is_empty());
    // a recording without them (format 1): all off
    let mut h = r.header.clone();
    h.switches = None;
    assert_eq!(Replay::new(h, Vec::new(), Vec::new(), false).switches.list(), "");
}
