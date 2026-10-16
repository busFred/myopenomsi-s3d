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
    let h = Header { version: omsi_traffic::record::VERSION, map: "test".into(), left_hand: false, lanes: net.lanes.clone(), lights, started_day_time: 8.0 * 3600.0, trigger: None, derived: None };
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
