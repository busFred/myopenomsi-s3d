//! Replaying a traffic recording ([`crate::record`]) headless: the network and the light
//! programs from the header, the vehicles from a keyframe, then frame after frame with
//! the recorded inputs. The simulation is deterministic, so the replay drives as the game
//! did; at every further keyframe it measures how far it drifted and goes on from the
//! recorded positions, so that one difference does not snowball.
//!
//! The replay runs with the current rules. The header names the rules the game ran with by
//! their hash (`Header::rules`, format 4); a recording made under other rules - another
//! hash, or a format before 4, which names none - loads with a warning, as it will not
//! reproduce bit-exactly.
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
use crate::record::{read_lines, rules_hash, write_file, Frame, Header, Record};
use crate::sim::{External, Simulation, Vehicle};
use hashbrown::HashMap;
use omsi_sim::traffic::Network;
use std::io;
use std::path::Path;

/// How far apart lane ends may be and still link (as the game links them).
const LINK_TOL: f64 = 1.5;

/// A free run puts a spawn off for this long (s) while its place is taken; then drops it.
pub const SPAWN_PATIENCE: f64 = 30.0;

/// A free run is where the recording has it when no vehicle is further off than this (m).
const IN_SYNC: f32 = 0.01;

/// A spawn is put off when a vehicle is this close ahead on its lane (m, bumper to bumper).
const SPAWN_GAP: f32 = 6.0;

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
    /// the replay goes on with its own. `Spawn` adds a vehicle whose id is not on the road
    /// (when its place is free: else it waits for up to `SPAWN_PATIENCE` s, then is
    /// dropped), `VehicleChanged` gives a vehicle its new way without moving it (unless it is
    /// where the game has it), `Remove` is not applied (the game's removals were made under
    /// the game's rules, 25ce52f; it only drops a spawn still waiting), and the run's own
    /// network end takes vehicles off. Of the frame's inputs the players and the people
    /// apply as recorded; the light clocks run on with the run's own demand after the first
    /// frame, and the game's stops and lateral ramps are laid from the run's own vehicles
    /// ([round 7 E2], `free_inputs`). (The people are the game's: they walk by the game's
    /// light phases while the run's vehicles see the run's own clocks - a pedestrian may
    /// cross on a phase the run's light no longer shows.) This is how a rule change is tried: what the recording
    /// does differently with it is not undone by the next keyframe.
    pub free_run: bool,
    /// Free run: spawns that did not fit on the road as the free run has it (a vehicle
    /// there already, or one just ahead): (give up at, the vehicle). They are tried again
    /// every frame at the recorded lane and place.
    pub pending_spawns: Vec<(f64, Vehicle)>,
    /// Free run: the last keyframe found the replay where the recording had it (every vehicle
    /// within `IN_SYNC` metres, none missing or extra). While it does, spawns go where the
    /// recording puts them, even onto another vehicle (the game does that too); once it has
    /// drifted, they wait for room. (Known only at keyframes: a drift shows up late.)
    pub in_sync: bool,
    /// Free run: how long (s) a spawn waits for its place (`SPAWN_PATIENCE`).
    pub spawn_patience: f64,
    /// Free run: spawns put off at least once (they may have been placed later).
    pub deferred_spawns: u64,
    /// Free run: spawns given up after `SPAWN_PATIENCE` seconds.
    pub dropped_spawns: u64,
    /// Indices of the keyframes in `records`.
    keyframes: Vec<usize>,
    /// The vehicles the run's own network end took off (`Simulation::removed`): (time, id,
    /// speed, place). Appended frame by frame; a caller may drain it.
    pub ended: Vec<(f64, u64, f32, glam::DVec3)>,
    /// [round 7 E2] Free run: the light clocks were taken from the recording (the first
    /// frame after the starting keyframe); from then on they run on with the run's own
    /// demand (`Simulation::update_lights`).
    clocks_taken: bool,
    /// [round 7 E2] Free run, per vehicle: the stop the game's bus service last asked for,
    /// (its distance as recorded, the run's odometer reading at it) - see `free_inputs`.
    free_stops: HashMap<u64, (f32, f32)>,
    /// [round 7 E2] Free run, per vehicle: the lateral ramp the game had last (`Steer`).
    free_ramps: HashMap<u64, (f32, f32, f32, f32)>,
}

/// [round 7 E2] A recorded stop distance (`External::stop_at`) growing by more than this (m)
/// from one frame to the next is a new stop: the game's bus approaching one sees it shrink by
/// its travel, at most 0.6 m a frame (18 m/s at 30 frames a second), and a stop it serves
/// stays put; a jump up of a metre is the next stop (`bus_service::step` hands it on).
const NEW_STOP: f32 = 1.0;

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
        if let Some(why) = other_rules(&header) {
            log::warn!("replay: {} was made under other rules ({why}): the replay runs with the current ones ({}) and will not reproduce it bit-exactly", path.display(), rules_hash());
        }
        if header.version > crate::record::VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{} is of format {}, later than this tool's {}: read it with the tool of its branch (`record::VERSION`)", path.display(), header.version, crate::record::VERSION)));
        }
        if header.version < 4 {
            log::warn!("replay: {} is of format {}: fields added since (`record::VERSION`) read as their defaults where it lacks them", path.display(), header.version);
        }
        Ok(Replay::new(header, records, skipped, truncated))
    }

    pub fn new(header: Header, mut records: Vec<Record>, skipped: Vec<(usize, String)>, truncated: bool) -> Replay {
        // ([fix pass 2, 5] a driver field a record lacked: its class's preset)
        for rec in &mut records {
            match rec {
                Record::Keyframe { vehicles, .. } => vehicles.iter_mut().for_each(|v| v.params.fill_missing(v.class)),
                Record::Spawn { vehicle, .. } | Record::VehicleChanged { vehicle, .. } => vehicle.params.fill_missing(vehicle.class),
                _ => {}
            }
        }
        let keyframes = records.iter().enumerate().filter(|(_, r)| matches!(r, Record::Keyframe { .. })).map(|(i, _)| i).collect();
        let (net, sim) = build(&header);
        let mut r = Replay { header, records, skipped, truncated, net, sim, vehicles: Vec::new(), types: HashMap::new(), cursor: 0, divergence: Vec::new(), missing: Vec::new(), free_run: false, pending_spawns: Vec::new(), in_sync: true, spawn_patience: SPAWN_PATIENCE, deferred_spawns: 0, dropped_spawns: 0, keyframes, ended: Vec::new(), clocks_taken: false, free_stops: HashMap::new(), free_ramps: HashMap::new() };
        r.stub_links();
        r
    }

    /// The lit stubs on the recorded network: the links round them and the merges of one
    /// way's pieces dropped (a recording from before them has them; one since, none), the
    /// converging lanes compiled, the junctions compiled from what is left (lanes added later are linked and compiled with
    /// them, `Network::extend`).
    fn stub_links(&mut self) {
        self.net.apply_stub_links();
        // (and the lanes of one object converging side by side, that a header from before
        // them has no meeting place for: `Network::add_converging`)
        self.net.compile_converging();
        self.sim.junctions = Junctions::compile(&self.net);
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
        self.stub_links();
        self.types.clear();
        self.divergence.clear();
        self.missing.clear();
        self.pending_spawns.clear();
        self.in_sync = true;
        self.clocks_taken = false;
        self.free_stops.clear();
        self.free_ramps.clear();
        self.deferred_spawns = 0;
        self.dropped_spawns = 0;
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
            self.through_stubs_all();
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
                    let mut vs = vehicles.clone();
                    self.keep_standing(&mut vs);
                    self.vehicles = vs;
                    self.through_stubs_all();
                }
            }
            Record::Spawn { time, vehicle, type_name } => {
                self.types.insert(vehicle.id, type_name.clone());
                let mut v = vehicle.clone();
                v.state.route_through_stubs(&self.net);
                if self.free_run {
                    // (a free run keeps the one it has, and does not put one onto another)
                    if !self.vehicles.iter().any(|o| o.id == v.id) {
                        self.spawn_free(*time, v);
                    }
                    return;
                }
                match self.vehicles.iter_mut().find(|o| o.id == v.id) {
                    Some(o) => {
                        let mut v = v;
                        if self.header.version < 2 {
                            v.claim_standing = o.claim_standing;
                        }
                        *o = v
                    }
                    None => self.vehicles.push(v),
                }
            }
            Record::Remove { id, .. } => {
                let id = *id;
                // (a free run drives its own vehicles: the game's removal of one, made
                // under the game's rules at the game's time, is not applied to a vehicle
                // this run still has - a car the game deleted the second its light turned
                // green vanished here while rolling into its turn under the new rules; the
                // run's own network end takes a vehicle off, as `drive` does)
                if !self.free_run {
                    self.vehicles.retain(|v| v.id != id);
                }
                self.pending_spawns.retain(|p| p.1.id != id);
            }
            Record::VehicleChanged { time, vehicle } => {
                let mut v = vehicle.clone();
                v.state.route_through_stubs(&self.net);
                if self.free_run {
                    match self.vehicles.iter().position(|o| o.id == v.id) {
                        // (the way changes, not where the vehicle is; one where the game
                        // has it is the game's)
                        Some(i) if !in_step(&self.vehicles[i], &v) => take_way(&mut self.vehicles[i], &v, &self.net),
                        Some(i) => self.vehicles[i] = v,
                        None => self.spawn_free(*time, v),
                    }
                    return;
                }
                match self.vehicles.iter_mut().find(|o| o.id == v.id) {
                    Some(o) => {
                        let mut v = v;
                        if self.header.version < 2 {
                            v.claim_standing = o.claim_standing;
                        }
                        *o = v
                    }
                    None => self.vehicles.push(v),
                }
            }
            Record::LanesAdded { time, first, lanes } => {
                if *first != self.net.lanes.len() {
                    log::warn!("replay: lanes added at {time:.1} s from index {first}, but the network has {}", self.net.lanes.len());
                }
                let added = self.net.extend(lanes.clone(), LINK_TOL);
                self.sim.network_grew(&self.net, added.start);
                self.through_stubs_all();
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

    /// A recording before format 2 has no `claim_standing` in its vehicles (the game kept it
    /// in the simulation, where it lived on over keyframes and changed ways): the vehicles
    /// put back keep the replay's own.
    fn keep_standing(&self, vs: &mut [Vehicle]) {
        if self.header.version >= 2 {
            return;
        }
        let here: HashMap<u64, bool> = self.vehicles.iter().map(|v| (v.id, v.claim_standing)).collect();
        for v in vs {
            if let Some(&s) = here.get(&v.id) {
                v.claim_standing = s;
            }
        }
    }

    /// [round 7 E2] Free run: the game's inputs of a frame on the run's own vehicles, which
    /// are not where the game's were. The flags and the bay (`External` but `stop_at`) and
    /// the indicator and lateral target (`Steer`) as recorded (`free_ramps`: the game's last
    /// ramp); a stop distance is the game
    /// bus's, from where the game had it: the run keeps the stop where it lay from its own
    /// bus when the game asked for it (`free_stops`: a new one when the recorded distance
    /// appears or grows by `NEW_STOP`), and so counts the distance from its own place; a
    /// lateral ramp the game laid (`Steer::lateral_ramp`, other than the vehicle's) is laid
    /// from the run's own place and offset, not the game's odometer.
    fn free_inputs(vehicles: &mut [Vehicle], free_stops: &mut HashMap<u64, (f32, f32)>, free_ramps: &mut HashMap<u64, (f32, f32, f32, f32)>, ext: &HashMap<u64, External>, steer: &HashMap<u64, crate::record::Steer>) {
        let mut stops: HashMap<u64, (f32, f32)> = HashMap::new();
        for v in vehicles.iter_mut() {
            let mut e = ext.get(&v.id).copied().unwrap_or_default();
            if let Some(d) = e.stop_at {
                let odo = v.state.odometer;
                let at = match free_stops.get(&v.id) {
                    Some(&(last, at)) if d <= last + NEW_STOP => at,
                    _ => odo + d,
                };
                stops.insert(v.id, (d, at));
                e.stop_at = Some(at - odo);
            }
            v.external = e;
            if let Some(s) = steer.get(&v.id) {
                let st = &mut v.state;
                st.signal = s.signal;
                st.signal_time = s.signal_time;
                st.lateral_target = s.lateral_target;
                // (a ramp the game laid since it last had one: the first seen is the vehicle's
                // own when it came from the recording)
                let last = free_ramps.insert(v.id, s.lateral_ramp).unwrap_or(st.lateral_ramp);
                if s.lateral_ramp != last {
                    let (_, to, _, len) = s.lateral_ramp;
                    st.lateral_ramp = (st.lateral, to, st.odometer, len);
                }
            }
        }
        *free_stops = stops;
    }

    /// [round 7 E3] A recording from before the lit stubs: its vehicles planned their ways on
    /// the network as the game linked it, and the ways round the lit stubs take the stubs
    /// (`AiState::route_through_stubs`) - once, as a vehicle comes from the recording
    /// (keyframe, spawn, changed way) and when the network grows, not every frame: a
    /// vehicle plans on on the replay's own network.
    fn through_stubs_all(&mut self) {
        for v in self.vehicles.iter_mut().chain(self.pending_spawns.iter_mut().map(|p| &mut p.1)) {
            v.state.route_through_stubs(&self.net);
        }
    }

    /// Free run: put `v` on the road if its place is free, else wait for it.
    fn spawn_free(&mut self, time: f64, v: Vehicle) {
        if self.pending_spawns.iter().any(|p| p.1.id == v.id) {
            return;
        }
        if self.in_sync || self.fits(&v) {
            self.vehicles.push(v);
        } else {
            self.deferred_spawns += 1;
            self.pending_spawns.push((time + self.spawn_patience, v));
        }
    }

    /// Free run: try the put-off spawns again; give up on those that waited too long.
    fn retry_spawns(&mut self, now: f64) {
        for (deadline, v) in std::mem::take(&mut self.pending_spawns) {
            if self.vehicles.iter().any(|o| o.id == v.id) {
                continue;
            }
            if self.fits(&v) {
                self.vehicles.push(v);
            } else if now > deadline {
                self.dropped_spawns += 1;
            } else {
                self.pending_spawns.push((deadline, v));
            }
        }
    }

    /// Whether `v`, at the lane and place it has, would overlap a vehicle on the road (their
    /// boxes in the plan, at about the same height) or stand less than `SPAWN_GAP` behind
    /// one on its lane.
    fn fits(&mut self, v: &Vehicle) -> bool {
        let mut v = v.clone();
        self.sim.update_positions(&self.net, std::slice::from_mut(&mut v));
        self.sim.update_positions(&self.net, &mut self.vehicles);
        let b = v.obb();
        self.vehicles.iter().all(|o| {
            if (o.pos.z - v.pos.z).abs() <= 3.0 && o.obb().overlaps_plan(&b) {
                return false;
            }
            let ahead = o.state.lane == v.state.lane && o.state.s >= v.state.s && (o.state.s - o.state.rear) - (v.state.s + v.state.front) < SPAWN_GAP;
            !ahead
        })
    }

    fn frame(&mut self, f: &Frame) {
        if self.free_run && !self.pending_spawns.is_empty() {
            self.retry_spawns(f.time);
        }
        let sim = &mut self.sim;
        sim.time = f.time;
        sim.day_time = f.day_time;
        sim.time_scale = f.time_scale;
        sim.players.clone_from(&f.players);
        sim.people = f.people.iter().map(|p| p.person()).collect();
        // [round5 C1] the walkers as `Humans::strollers()` lists them now, rebuilt from the
        // people records (their footpath leg; not a kerb waiter on a red light): the recorded
        // list is the game's at recording time, so a change of that rule showed in no replay
        // (a recording without the legs keeps its own list)
        if f.people.iter().any(|p| p.leg.is_some()) {
            sim.walkers = f.people.iter().filter(|p| !(p.at_kerb && p.decision.as_deref() == Some(crate::record::RED_LIGHT))).filter_map(|p| p.leg.map(|(l, s, _)| (l, s))).collect();
        } else {
            sim.walkers.clone_from(&f.walkers);
        }
        if let Some(order) = &f.order {
            let mut by_id: HashMap<u64, Vehicle> = self.vehicles.drain(..).map(|v| (v.id, v)).collect();
            for id in order {
                match by_id.remove(id) {
                    Some(v) => self.vehicles.push(v),
                    None => log::warn!("replay: vehicle {id} is in the game's order at {:.1} s but not here", f.time),
                }
            }
            // (a free run keeps the vehicles the game no longer lists: the game took them
            // off under its rules at its time - a car deleted the frame its light turned
            // green, standing 100 s - and this reorder silently dropped them, so every
            // page showed the game's deletions on both sides whatever the rules did)
            if self.free_run && !by_id.is_empty() {
                let mut rest: Vec<Vehicle> = by_id.into_values().collect();
                rest.sort_by_key(|v| v.id);
                self.vehicles.extend(rest);
            }
        }
        let ext: HashMap<u64, External> = f.externals.iter().copied().collect();
        let steer: HashMap<u64, crate::record::Steer> = f.steer.iter().copied().collect();
        if self.free_run {
            Self::free_inputs(&mut self.vehicles, &mut self.free_stops, &mut self.free_ramps, &ext, &steer);
        } else {
            for v in &mut self.vehicles {
                v.external = ext.get(&v.id).copied().unwrap_or_default();
                if let Some(s) = steer.get(&v.id) {
                    s.apply(&mut v.state);
                }
            }
        }
        // ([round 7 E2] a free run takes the clocks once and runs them on itself: the demand
        // phases follow its own vehicles, not the game's)
        if !self.free_run || !self.clocks_taken {
            for (c, (time, held, _)) in sim.lights.iter_mut().zip(&f.light_clocks) {
                c.time = *time;
                c.held = *held;
            }
            self.clocks_taken = true;
        }
        sim.tick(&self.net, f.dt, &mut self.vehicles);
        // (as the game takes them off: from the back, each swapped with the last)
        for &i in &sim.removed {
            if i < self.vehicles.len() {
                let v = self.vehicles.swap_remove(i);
                self.ended.push((sim.time, v.id, v.state.speed, v.pos));
            }
        }
    }

    /// How far the replay is from the recorded keyframe.
    fn compare(&mut self, time: f64, recorded: &[Vehicle]) {
        let mut worst = (0.0f32, 0u64);
        let missing_before = self.missing.len();
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
        self.in_sync = worst.0 <= IN_SYNC && self.missing.len() == missing_before;
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

/// Whether the vehicle is where the recording has `rec` (same lane, within `IN_SYNC` m).
fn in_step(o: &Vehicle, rec: &Vehicle) -> bool {
    o.state.lane == rec.state.lane && (o.state.s - rec.state.s).abs() <= IN_SYNC
}

/// A free run's vehicle `o` gets the way the game changed (`rec`): the route and where in it
/// the vehicle is, its class, driver, size and external inputs. It stays where it is: its
/// place, speed, odometer and lane changes in progress; [round 7 E2] a new route is planned
/// afresh from where it is, as the game's `set_route` does (bus_service.rs), not along the
/// lanes the old one had planned.
fn take_way(o: &mut Vehicle, rec: &Vehicle, net: &Network) {
    o.class = rec.class;
    o.params = rec.params.clone();
    o.half_width = rec.half_width;
    o.external = rec.external;
    let (st, r) = (&mut o.state, &rec.state);
    st.traffic_pool = r.traffic_pool.clone();
    st.veh_type = r.veh_type;
    st.max_speed_kmh = r.max_speed_kmh;
    st.accel = r.accel;
    st.decel = r.decel;
    st.length = r.length;
    st.front = r.front;
    st.rear = r.rear;
    st.desire = r.desire;
    st.headway = r.headway;
    st.min_gap = r.min_gap;
    st.reaction = r.reaction;
    st.lat_accel = r.lat_accel;
    let new_route = st.route != r.route;
    st.route = r.route.clone();
    // (where it is in the new route: its lane's place in it when it is there)
    st.route_index = st.route.iter().position(|&l| l == st.lane).unwrap_or(r.route_index);
    if new_route && !st.route.is_empty() {
        st.planned_next = None;
        st.ahead.clear();
        st.plan_next(net);
    }
}

/// Why the recording was made under other rules than the current ones ([`rules_hash`]), if
/// it was: another hash, or a format before 4, which names its rules only by the frozen
/// switch list (`record::RULES`, unchanged since round 4 while the rules changed). Such a
/// recording does not replay bit-exactly.
pub fn other_rules(h: &Header) -> Option<String> {
    let now = rules_hash();
    match h.rules.as_deref() {
        _ if h.version < 4 => Some(format!("format {}, before the rules were named by their hash", h.version)),
        None => Some("no rules recorded".into()),
        Some(r) if r == now => None,
        Some(r) => Some(format!("rules {r}, now {now}")),
    }
}

/// The network and the simulation as the header has them: the recorded lanes, links and
/// derived data (the junction table too), without the lit stubs applied (`Replay` applies
/// them).
pub fn recorded(h: &Header) -> (Network, Simulation) {
    build(h)
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
    // (the recorded junctions: `Replay::new` compiles its own after the lit stubs; these are
    // the game's, for `recorded`)
    let mut sim = Simulation::new(&Network::default(), h.lights.clone());
    sim.junctions = junctions;
    sim.left_hand = net.left_hand;
    (net, sim)
}
