//! Replaying a traffic recording ([`crate::record`]) headless: the network and the light
//! programs from the header, the vehicles from a keyframe, then frame after frame with
//! the recorded inputs. The simulation is deterministic, so the replay drives as the game
//! did; at every further keyframe it measures how far it drifted and goes on from the
//! recorded positions, so that one difference does not snowball.
//!
//! ```no_run
//! let mut r = omsi_traffic::replay::Replay::load(std::path::Path::new("jam.rec")).unwrap();
//! r.at(0);
//! while r.step() {}
//! for (t, ds, id) in &r.divergence {
//!     println!("{t:.1} s: {ds:.3} m (vehicle {id})");
//! }
//! ```

use crate::junction::Junctions;
use crate::record::{read_lines, write_file, Frame, Header, Record};
use crate::sim::{External, Simulation, Switches, Vehicle};
use hashbrown::HashMap;
use omsi_sim::traffic::Network;
use std::io;
use std::path::Path;

/// How far apart lane ends may be and still link (as the game links them).
const LINK_TOL: f64 = 1.5;

pub struct Replay {
    pub header: Header,
    pub records: Vec<Record>,
    /// Lines that could not be read: (line number, why).
    pub skipped: Vec<(usize, String)>,
    /// The recording ends without its last line (the game stopped mid-write).
    pub truncated: bool,
    pub net: Network,
    pub sim: Simulation,
    pub vehicles: Vec<Vehicle>,
    /// The vehicles' type names (from their `Spawn`).
    pub types: HashMap<u64, String>,
    /// The next record to apply.
    pub cursor: usize,
    /// At each keyframe replayed into: (time, the largest difference between where a
    /// vehicle got to and where the game had it, that vehicle's id). The difference is
    /// along the lane, or the distance between the two places when they are on different
    /// lanes.
    pub divergence: Vec<(f64, f32, u64)>,
    /// Vehicles of a keyframe the replay did not have (or had and the keyframe did not).
    pub missing: Vec<(f64, u64)>,
    /// Free run: after the starting keyframe the replay no longer goes back to the recorded
    /// vehicles at later keyframes. They are still compared (`divergence`, `missing`), and
    /// the replay goes on with its own. `Spawn` adds a vehicle whose id is not on the road,
    /// `Remove` takes one off, the frame's inputs apply as always. This is how a rule
    /// change is tried: what the recording does differently with it is not undone by the
    /// next keyframe.
    pub free_run: bool,
    /// The rule changes under test; kept across `at`, which builds the simulation anew.
    pub switches: Switches,
    /// Indices of the keyframes in `records`.
    keyframes: Vec<usize>,
}

impl Replay {
    /// Read a recording. The state is the header's network with no vehicles; `at` puts
    /// it at a keyframe.
    pub fn load(path: &Path) -> io::Result<Replay> {
        let (lines, truncated) = read_lines(path)?;
        let Some(first) = lines.first() else {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "an empty recording"));
        };
        let header: Header = crate::finite::from_str(first).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("the header: {e}")))?;
        let mut records = Vec::with_capacity(lines.len());
        let mut skipped = Vec::new();
        for (n, l) in lines.iter().enumerate().skip(1) {
            match crate::finite::from_str::<Record>(l) {
                Ok(r) => records.push(r),
                Err(e) => skipped.push((n + 1, e.to_string())),
            }
        }
        if !skipped.is_empty() {
            log::warn!("replay: {} lines of {} could not be read (the first: line {}: {})", skipped.len(), path.display(), skipped[0].0, skipped[0].1);
        }
        Ok(Replay::new(header, records, skipped, truncated))
    }

    pub fn new(header: Header, records: Vec<Record>, skipped: Vec<(usize, String)>, truncated: bool) -> Replay {
        let keyframes = records.iter().enumerate().filter(|(_, r)| matches!(r, Record::Keyframe { .. })).map(|(i, _)| i).collect();
        let (net, sim) = build(&header);
        Replay { header, records, skipped, truncated, net, sim, vehicles: Vec::new(), types: HashMap::new(), cursor: 0, divergence: Vec::new(), missing: Vec::new(), free_run: false, switches: Switches::default(), keyframes }
    }

    /// Set the rule changes under test (now, and after every `at`).
    pub fn set_switches(&mut self, switches: Switches) {
        self.switches = switches;
        self.sim.switches = switches;
    }

    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    /// The time of keyframe `k`.
    pub fn keyframe_time(&self, k: usize) -> Option<f64> {
        self.keyframes.get(k).map(|&i| self.records[i].time())
    }

    /// The state at keyframe `k`: the network with the lanes and lights added before it,
    /// the parked cars of then, the keyframe's vehicles. False when there is no such
    /// keyframe.
    pub fn at(&mut self, k: usize) -> bool {
        let Some(&at) = self.keyframes.get(k) else { return false };
        let (net, sim) = build(&self.header);
        self.net = net;
        self.sim = sim;
        self.sim.switches = self.switches;
        self.types.clear();
        self.divergence.clear();
        self.missing.clear();
        for i in 0..at {
            match &self.records[i] {
                Record::LanesAdded { .. } | Record::LightsAdded { .. } | Record::Parked { .. } => self.apply(i),
                Record::Spawn { vehicle, type_name, .. } => {
                    self.types.insert(vehicle.id, type_name.clone());
                }
                _ => {}
            }
        }
        if let Record::Keyframe { time, vehicles } = &self.records[at] {
            self.vehicles = vehicles.clone();
            self.sim.time = *time;
        }
        self.cursor = at + 1;
        true
    }

    /// Apply records up to and including the next frame. False at the end.
    pub fn step(&mut self) -> bool {
        while self.cursor < self.records.len() {
            let i = self.cursor;
            self.cursor += 1;
            let frame = matches!(self.records[i], Record::Frame(_));
            self.apply(i);
            if frame {
                return true;
            }
        }
        false
    }

    /// Replay up to time `t` (the frames that start before it).
    pub fn run_until(&mut self, t: f64) {
        while self.records.get(self.cursor).is_some_and(|r| r.time() < t) && self.step() {}
    }

    /// The largest divergence measured.
    pub fn worst(&self) -> Option<(f64, f32, u64)> {
        self.divergence.iter().copied().max_by(|a, b| a.1.total_cmp(&b.1))
    }

    fn apply(&mut self, i: usize) {
        // (lent out while it is applied, and put back: the replay can start over)
        let rec = std::mem::replace(&mut self.records[i], Record::Remove { time: 0.0, id: u64::MAX });
        self.apply_record(&rec);
        self.records[i] = rec;
    }

    fn apply_record(&mut self, rec: &Record) {
        match rec {
            Record::Frame(f) => self.frame(f),
            Record::Keyframe { time, vehicles } => {
                self.compare(*time, vehicles);
                if !self.free_run {
                    self.vehicles = vehicles.clone();
                }
            }
            Record::Spawn { vehicle, type_name, .. } => {
                self.types.insert(vehicle.id, type_name.clone());
                let v = vehicle.clone();
                match self.vehicles.iter_mut().find(|o| o.id == v.id) {
                    // (a free run keeps the one it has)
                    Some(_) if self.free_run => {}
                    Some(o) => *o = v,
                    None => self.vehicles.push(v),
                }
            }
            Record::Remove { id, .. } => {
                let id = *id;
                self.vehicles.retain(|v| v.id != id);
            }
            Record::VehicleChanged { vehicle, .. } => {
                let v = vehicle.clone();
                match self.vehicles.iter_mut().find(|o| o.id == v.id) {
                    Some(o) => *o = v,
                    None => self.vehicles.push(v),
                }
            }
            Record::LanesAdded { time, first, lanes } => {
                if *first != self.net.lanes.len() {
                    log::warn!("replay: lanes added at {time:.1} s from index {first}, but the network has {}", self.net.lanes.len());
                }
                let added = self.net.extend(lanes.clone(), LINK_TOL);
                self.sim.network_grew(&self.net, added.start);
            }
            Record::LightsAdded { first, lights, .. } => {
                let from = (*first).min(self.sim.lights.len());
                let skip = self.sim.lights.len() - from;
                self.sim.lights.extend(lights.iter().skip(skip).cloned());
            }
            Record::Parked { parked, .. } => self.sim.parked = parked.clone(),
            Record::Note { .. } => {}
        }
    }

    fn frame(&mut self, f: &Frame) {
        let sim = &mut self.sim;
        sim.time = f.time;
        sim.day_time = f.day_time;
        sim.time_scale = f.time_scale;
        sim.players.clone_from(&f.players);
        sim.people = f.people.iter().map(|p| p.person()).collect();
        sim.walkers.clone_from(&f.walkers);
        if let Some(order) = &f.order {
            let mut by_id: HashMap<u64, Vehicle> = self.vehicles.drain(..).map(|v| (v.id, v)).collect();
            for id in order {
                match by_id.remove(id) {
                    Some(v) => self.vehicles.push(v),
                    None => log::warn!("replay: vehicle {id} is in the game's order at {:.1} s but not here", f.time),
                }
            }
        }
        let ext: HashMap<u64, External> = f.externals.iter().copied().collect();
        let steer: HashMap<u64, crate::record::Steer> = f.steer.iter().copied().collect();
        for v in &mut self.vehicles {
            v.external = ext.get(&v.id).copied().unwrap_or_default();
            if let Some(s) = steer.get(&v.id) {
                s.apply(&mut v.state);
            }
        }
        for (c, (time, held, _)) in sim.lights.iter_mut().zip(&f.light_clocks) {
            c.time = *time;
            c.held = *held;
        }
        sim.tick(&self.net, f.dt, &mut self.vehicles);
        // (as the game takes them off: from the back, each swapped with the last)
        for &i in &sim.removed {
            if i < self.vehicles.len() {
                self.vehicles.swap_remove(i);
            }
        }
    }

    /// How far the replay is from the recorded keyframe.
    fn compare(&mut self, time: f64, recorded: &[Vehicle]) {
        let mut worst = (0.0f32, 0u64);
        let here: HashMap<u64, usize> = self.vehicles.iter().enumerate().map(|(i, v)| (v.id, i)).collect();
        for r in recorded {
            let Some(&i) = here.get(&r.id) else {
                self.missing.push((time, r.id));
                continue;
            };
            let v = &self.vehicles[i];
            let d = if v.state.lane == r.state.lane {
                (v.state.s - r.state.s).abs()
            } else if v.state.lane < self.net.lanes.len() && r.state.lane < self.net.lanes.len() {
                (v.state.way_point(&self.net, 0.0) - r.state.way_point(&self.net, 0.0)).length() as f32
            } else {
                f32::INFINITY
            };
            if d > worst.0 {
                worst = (d, r.id);
            }
        }
        for v in &self.vehicles {
            if !recorded.iter().any(|r| r.id == v.id) {
                self.missing.push((time, v.id));
            }
        }
        self.divergence.push((time, worst.0, worst.1));
    }

    /// Write a fixture for a regression test: the header, what came before (the lanes and
    /// lights added, the parked cars of then), the keyframe at or before `t0` and the
    /// records from it up to `t1`.
    pub fn scenario(&self, t0: f64, t1: f64, out: &Path) -> io::Result<()> {
        let Some(&k) = self.keyframes.iter().rev().find(|&&i| self.records[i].time() <= t0).or(self.keyframes.first()) else {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "the recording has no keyframe"));
        };
        let mut out_records: Vec<&Record> = Vec::new();
        let mut parked: Option<&Record> = None;
        for r in &self.records[..k] {
            match r {
                Record::LanesAdded { .. } | Record::LightsAdded { .. } => out_records.push(r),
                Record::Parked { .. } => parked = Some(r),
                _ => {}
            }
        }
        out_records.extend(parked);
        out_records.extend(self.records[k..].iter().take_while(|r| r.time() <= t1));
        write_file(out, &self.header, &out_records)
    }
}

/// The network and the simulation the header describes.
fn build(h: &Header) -> (Network, Simulation) {
    let mut net = Network { lanes: h.lanes.clone(), left_hand: h.left_hand, ..Default::default() };
    for (i, l) in net.lanes.iter().enumerate() {
        if let Some(k) = l.key {
            net.by_key.entry(k).or_default().push(i);
        }
    }
    net.build_grid();
    let n = net.lanes.len();
    match &h.derived {
        Some(d) => {
            net.prev = d.prev.clone();
            net.conflicts = d.conflicts.clone();
            net.crossings = d.crossings.clone();
            net.walks = d.walks.clone();
            net.reach = d.reach.clone();
            // (a header cut down by hand: fill up)
            net.prev.resize(n, Vec::new());
            net.conflicts.resize(n, Vec::new());
            net.crossings.resize(n, Vec::new());
            net.walks.resize(n, Vec::new());
            net.reach.resize(n, omsi_sim::traffic::REACH_MAX);
        }
        None => {
            net.prev = vec![Vec::new(); n];
            for i in 0..n {
                for k in 0..net.lanes[i].next.len() {
                    let to = net.lanes[i].next[k];
                    if to < n {
                        net.prev[to].push(i);
                    }
                }
            }
            net.compute_conflicts();
            net.compute_reach();
        }
    }
    let junctions = match &h.derived {
        Some(d) => d.junctions.clone(),
        None => Junctions::compile(&net),
    };
    // (the junctions are the recorded ones: nothing to compile)
    let mut sim = Simulation::new(&Network::default(), h.lights.clone());
    sim.junctions = junctions;
    sim.left_hand = net.left_hand;
    (net, sim)
}
