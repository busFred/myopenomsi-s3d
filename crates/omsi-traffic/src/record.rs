//! Recording the traffic ("a rosbag"): everything the frame of [`Simulation::tick`] reads
//! from outside, frame by frame, so that a deadlock or a frozen pedestrian seen in the
//! game can be replayed headless ([`crate::replay`]) and turned into a regression test.
//!
//! A recording is a zlib stream of JSON lines. The first line is a [`Header`]: the map,
//! the lanes with everything the network and the junction table derived from them, and
//! the light programs. Every further line is one [`Record`]: a frame's inputs (the
//! players, the people, the bus services' wishes, the light clocks), a keyframe of every
//! vehicle every [`KEYFRAME_SECS`], and what the game changed between frames (a car put
//! on or taken off the road, a timetable bus given its next trip, tiles bringing lanes).
//!
//! The [`Recorder`] keeps the last [`RING_SECS`] in memory all the time, so that the
//! minute before something went wrong can be saved after the fact, and streams to a file
//! when asked to. Writing, compressing and saving happen on a thread of their own: the
//! frame only serializes its record once.

use crate::finite;
use crate::junction::Junctions;
use crate::sim::{External, Parked, Person, PlayerView, Simulation, Vehicle};
use flate2::write::ZlibEncoder;
use flate2::Compression;
use glam::{DVec2, DVec3};
use hashbrown::{HashMap, HashSet};
use omsi_sim::traffic::{AiState, Crossing, Lane, Network, TrafficLightController};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::VecDeque;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;

/// The format's version (the header's first field). 2: vehicles carry their claim
/// standing hysteresis (`Vehicle::claim_standing`); before, the simulation kept it. 3: the
/// lanes carry their drivable extent (`Lane::drivable`, round 5 L1); a recording without it
/// reads back the lanes' half widths. 4 ([round 7 E1]): the header names the rules by their
/// hash (`Header::rules`, [`rules_hash`]), and every field added with a `serde(default)`
/// since 3 is written - `Vehicle::passed_on_go`, `amber_line`, `zip_won`, `change_still`,
/// `mutual`, `mutual_gap`, `ease_for`, `held_by_nothing`, `External::trip_over`,
/// `Frame::steer`, `Frame::order`; a format-3 recording may lack some (read as their
/// defaults, `Replay::load` warns).
pub const VERSION: u32 = 4;
/// The rule switches a recording names (`Header::switches`), for information: the rule
/// changes once tried behind switches and accepted, now always on, and the junction rules of
/// round 4. Frozen since round 4: which rules a recording was made under is
/// [`rules_hash`]'s (format 4).
pub const RULES: &str = "competitor_filter,signal_foes,stub_links,lights_first,right_of_way,blocked_foe,stop_placement,player_chain,amber_clear";
/// [round 7 E1] The rules the traffic runs with, as a hash: FNV-1a (64 bit) over the
/// sources of the rules - the driver, perception, plan, junctions, resolver, simulation,
/// access rule (omsi-traffic) and the lane model (omsi-sim's traffic.rs) - with comment
/// lines and blank lines left out, so that a changed constant or rule changes it and a
/// reworded comment does not. A recording names it (`Header::rules`); one naming another
/// does not replay bit-exactly (`replay::other_rules`).
pub fn rules_hash() -> String {
    static HASH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    HASH.get_or_init(|| {
        const SOURCES: [&str; 9] = [
            include_str!("driver.rs"),
            include_str!("perception.rs"),
            include_str!("plan.rs"),
            include_str!("junction.rs"),
            include_str!("reservation.rs"),
            include_str!("sim.rs"),
            include_str!("access.rs"),
            include_str!("../../omsi-sim/src/traffic.rs"),
            include_str!("../../omsi-sim/src/ai_motion.rs"),
        ];
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for src in SOURCES {
            for line in src.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with("//")) {
                for b in line.bytes().chain(std::iter::once(b'\n')) {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x0100_0000_01b3);
                }
            }
        }
        format!("{h:016x}")
    })
    .clone()
}

/// The memory keeps this many seconds (and the keyframe before them).
pub const RING_SECS: f64 = 60.0;
/// A keyframe of every vehicle this often (s): a replay starts at one, and a replay that
/// drifted is put back on the recorded positions there.
pub const KEYFRAME_SECS: f64 = 5.0;

/// The first line of a recording.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Header {
    pub version: u32,
    /// The map's folder name.
    pub map: String,
    pub left_hand: bool,
    /// The lanes as the game had them, linked (`next`, `left`, `right` are kept).
    pub lanes: Vec<Lane>,
    /// The light programs with their clocks.
    pub lights: Vec<TrafficLightController>,
    /// The time of day the recording began at (s since midnight).
    pub started_day_time: f64,
    /// What made the game save it ("manual", "stuck", "overlap"), None for a recording
    /// that was asked for.
    pub trigger: Option<String>,
    /// What the network and the junction table derived from the lanes. The game grows its
    /// network tile by tile and some of it (a timetable's reverse twins of a junction's
    /// paths) comes out differently from linking all the lanes at once, so the replay
    /// takes the game's own. Without it (a fixture written by hand) it is worked out.
    #[serde(default)]
    pub derived: Option<Derived>,
    /// (format 2) The rules the game ran with, for information: [`RULES`] since the rule
    /// switches were removed; a recording made while they were switches names those that
    /// were on. The replay always runs with the current rules; one made under others (none
    /// recorded: format 1, all off; or a list without all of `RULES`) does not reproduce
    /// bit-exactly (`Replay::load` warns).
    #[serde(default)]
    pub switches: Option<String>,
    /// (format 4) The rules the game ran with, by their hash ([`rules_hash`]).
    #[serde(default)]
    pub rules: Option<String>,
}

/// The network's and the junction table's derived data (see [`Header::derived`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Derived {
    pub prev: Vec<Vec<usize>>,
    pub conflicts: Vec<Vec<usize>>,
    pub crossings: Vec<Vec<Crossing>>,
    pub walks: Vec<Vec<(usize, f32, f32)>>,
    pub reach: Vec<f32>,
    pub junctions: Junctions,
}

/// One person on foot as the traffic saw them, with what the game knew about them
/// besides (their task, the footpath they walk, their last decision at a kerb), for
/// finding out why somebody froze. (The people are most of a frame's line: what is
/// false, zero or none is left out.)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PersonRecord {
    pub pos: DVec3,
    pub vel: DVec2,
    /// What they do: a stroller "WalkStreet", a passenger their task ("aboard" or
    /// "waiting" added).
    pub state: Cow<'static, str>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub still: f32,
    #[serde(default, skip_serializing_if = "is_false")]
    pub on_crossing: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub waiting_for_bus: bool,
    /// The footpath leg they walk: (lane, distance along it, the leg's length).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub leg: Option<(usize, f32, f32)>,
    /// The last decision at a kerb: "go", or why they wait ("red light", ...), or what
    /// they do on a crossing ("caught by the change: ...").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision: Option<Cow<'static, str>>,
    /// The pace they walk at now (m/s; 0 when not recorded).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub pace: f32,
    /// Waiting at a kerb to cross (`Person::at_kerb`).
    #[serde(default, skip_serializing_if = "is_false")]
    pub at_kerb: bool,
}

fn is_zero(v: &f32) -> bool {
    // (a negative zero is written: it reads back as itself only so)
    v.to_bits() == 0
}

fn is_false(v: &bool) -> bool {
    !*v
}

impl PersonRecord {
    /// The person as the frame sees them.
    pub fn person(&self) -> Person {
        Person { pos: self.pos, vel: self.vel, on_crossing: self.on_crossing, still: self.still, waiting_for_bus: self.waiting_for_bus, at_kerb: self.at_kerb }
    }

    /// A record of somebody the game told nothing more about.
    pub fn from_person(p: &Person) -> PersonRecord {
        PersonRecord { pos: p.pos, vel: p.vel, state: Cow::Borrowed(""), still: p.still, on_crossing: p.on_crossing, waiting_for_bus: p.waiting_for_bus, leg: None, decision: None, pace: 0.0, at_kerb: p.at_kerb }
    }

    /// Take the frame's own values over (the extras stay).
    pub fn set_person(&mut self, p: &Person) {
        self.pos = p.pos;
        self.vel = p.vel;
        self.on_crossing = p.on_crossing;
        self.still = p.still;
        self.waiting_for_bus = p.waiting_for_bus;
        self.at_kerb = p.at_kerb;
    }
}

/// What the game sets on a vehicle's way itself before the frame: a timetable bus at its
/// stops and a car parking work the indicator and the move into the bay.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Steer {
    pub signal: i32,
    pub signal_time: f32,
    pub lateral_target: f32,
    pub lateral_ramp: (f32, f32, f32, f32),
}

impl Steer {
    pub fn of(st: &AiState) -> Steer {
        Steer { signal: st.signal, signal_time: st.signal_time, lateral_target: st.lateral_target, lateral_ramp: st.lateral_ramp }
    }

    pub fn apply(&self, st: &mut AiState) {
        st.signal = self.signal;
        st.signal_time = self.signal_time;
        st.lateral_target = self.lateral_target;
        st.lateral_ramp = self.lateral_ramp;
    }
}

/// A frame's inputs, as they were just before the game ran it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frame {
    /// `Simulation::time` before the frame.
    pub time: f64,
    pub dt: f32,
    pub day_time: f64,
    pub time_scale: f64,
    pub players: Vec<PlayerView>,
    pub people: Vec<PersonRecord>,
    pub walkers: Vec<(usize, f32)>,
    /// The vehicles whose `External` is not the default, by id (the rest have none).
    pub externals: Vec<(u64, External)>,
    #[serde(default)]
    pub steer: Vec<(u64, Steer)>,
    /// Per light program: its cycle clock, whether it waits at a stop point, and the
    /// requests of the frame before.
    pub light_clocks: Vec<(f64, bool, Vec<bool>)>,
    /// The vehicles' ids in the game's order, when it changed since the last frame (the
    /// order matters: the first of two cars to ask for a gap gets it).
    #[serde(default)]
    pub order: Option<Vec<u64>>,
}

/// One line of a recording after the header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Record {
    Frame(Frame),
    /// Every vehicle, before the frame of the same time.
    Keyframe { time: f64, vehicles: Vec<Vehicle> },
    Spawn { time: f64, vehicle: Vehicle, type_name: String },
    Remove { time: f64, id: u64 },
    /// The game changed a vehicle's way outside the frame (a timetable bus given its next
    /// trip or let go, its route extended).
    VehicleChanged { time: f64, vehicle: Vehicle },
    /// Lanes handed to `Network::extend` (as they were before it linked them), the first
    /// of them at index `first`.
    LanesAdded { time: f64, first: usize, lanes: Vec<Lane> },
    LightsAdded { time: f64, first: usize, lights: Vec<TrafficLightController> },
    /// The parked cars beside the lanes, whenever they changed.
    Parked { time: f64, parked: HashMap<usize, Vec<Parked>> },
    /// Free text: what triggered a save, a stuck car's report.
    Note { time: f64, text: String },
}

impl Record {
    pub fn time(&self) -> f64 {
        match self {
            Record::Frame(f) => f.time,
            Record::Keyframe { time, .. }
            | Record::Spawn { time, .. }
            | Record::Remove { time, .. }
            | Record::VehicleChanged { time, .. }
            | Record::LanesAdded { time, .. }
            | Record::LightsAdded { time, .. }
            | Record::Parked { time, .. }
            | Record::Note { time, .. } => *time,
        }
    }
}

/// [`Record`] borrowing its parts, for writing without copying them first. Serializes
/// exactly as `Record` does.
#[derive(Serialize)]
enum RecordRef<'a> {
    Frame(FrameRef<'a>),
    Keyframe { time: f64, vehicles: &'a [Vehicle] },
    Spawn { time: f64, vehicle: &'a Vehicle, type_name: &'a str },
    Remove { time: f64, id: u64 },
    VehicleChanged { time: f64, vehicle: &'a Vehicle },
    LanesAdded { time: f64, first: usize, lanes: &'a [Lane] },
    LightsAdded { time: f64, first: usize, lights: &'a [TrafficLightController] },
    Parked { time: f64, parked: &'a HashMap<usize, Vec<Parked>> },
    Note { time: f64, text: &'a str },
}

#[derive(Serialize)]
struct FrameRef<'a> {
    time: f64,
    dt: f32,
    day_time: f64,
    time_scale: f64,
    players: &'a [PlayerView],
    people: &'a [PersonRecord],
    walkers: &'a [(usize, f32)],
    externals: &'a [(u64, External)],
    steer: &'a [(u64, Steer)],
    light_clocks: LightClocks<'a>,
    order: Option<&'a [u64]>,
}

/// The light programs' clocks as `Frame::light_clocks` has them.
struct LightClocks<'a>(&'a [TrafficLightController]);

impl Serialize for LightClocks<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.0.len()))?;
        for c in self.0 {
            seq.serialize_element(&(c.time, c.held, &c.request))?;
        }
        seq.end()
    }
}

#[derive(Serialize)]
struct DerivedRef<'a> {
    prev: &'a [Vec<usize>],
    conflicts: &'a [Vec<usize>],
    crossings: &'a [Vec<Crossing>],
    walks: &'a [Vec<(usize, f32, f32)>],
    reach: &'a [f32],
    junctions: &'a Junctions,
}

/// What a line is, for the memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Keyframe,
    Parked,
    /// Kept in memory with the rest.
    Other,
    /// Only for the file: the memory's header is written from the network as it is when
    /// it is saved, which has these lanes and lights already.
    FileOnly,
}

enum Msg {
    Line { time: f64, kind: Kind, text: String },
    StartFull { file: std::fs::File, header: String },
    StopFull,
    Dump { path: PathBuf, header: String },
    Flush(mpsc::Sender<()>),
}

/// The traffic recorder: the last minute in memory always, everything into a file while
/// a recording runs (both can be on).
pub struct Recorder {
    tx: Option<mpsc::Sender<Msg>>,
    thread: Option<std::thread::JoinHandle<()>>,
    /// The file being recorded to.
    full: Option<PathBuf>,
    last_keyframe: f64,
    /// The vehicles' ids in the order of the last frame, and as a set.
    order: Vec<u64>,
    known: HashSet<u64>,
    ids: Vec<u64>,
    /// A fingerprint of the parked cars last written.
    parked_print: Option<u64>,
    /// The network part of a header (`"lanes":…,"derived":…`) for a network of this many
    /// lanes: saving the minute again does not write the whole map out again.
    net_cache: Option<(usize, std::sync::Arc<String>)>,
    externals: Vec<(u64, External)>,
}

impl Default for Recorder {
    fn default() -> Self {
        Recorder::new()
    }
}

impl Recorder {
    pub fn new() -> Recorder {
        let (tx, rx) = mpsc::channel::<Msg>();
        let thread = std::thread::Builder::new().name("traffic-recorder".into()).spawn(move || writer_thread(rx)).ok();
        Recorder {
            tx: thread.as_ref().map(|_| tx),
            thread,
            full: None,
            last_keyframe: f64::NEG_INFINITY,
            order: Vec::new(),
            known: HashSet::new(),
            ids: Vec::new(),
            parked_print: None,
            net_cache: None,
            externals: Vec::new(),
        }
    }

    fn send(&self, m: Msg) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(m);
        }
    }

    fn line(&self, time: f64, kind: Kind, rec: &RecordRef) {
        match finite::to_string(rec) {
            Ok(text) => self.send(Msg::Line { time, kind, text }),
            Err(e) => log::warn!("traffic recording: a record could not be written: {e}"),
        }
    }

    /// Record everything into `path` from now on. The file starts with the network as it
    /// is and a keyframe at the next frame.
    pub fn start_full(&mut self, path: &Path, net: &Network, sim: &Simulation, map: &str) -> io::Result<()> {
        self.stop_full();
        let header = self.header_text(net, sim, map, None);
        let file = std::fs::File::create(path)?;
        self.send(Msg::StartFull { file, header });
        self.full = Some(path.to_path_buf());
        // (the file needs a keyframe and the parked cars to start from)
        self.last_keyframe = f64::NEG_INFINITY;
        self.parked_print = None;
        log::info!("traffic: recording to {}", path.display());
        Ok(())
    }

    pub fn stop_full(&mut self) {
        if let Some(p) = self.full.take() {
            self.send(Msg::StopFull);
            log::info!("traffic: recording to {} ended", p.display());
        }
    }

    pub fn is_recording(&self) -> bool {
        self.full.is_some()
    }

    /// The file being recorded to.
    pub fn recording(&self) -> Option<&Path> {
        self.full.as_deref()
    }

    /// The frame about to run: `sim` with its inputs set, the vehicles as they go into
    /// it. Writes what changed in the fleet (`type_name` names vehicle `i` of a new one)
    /// and in the parked cars, a keyframe when one is due, and the frame's inputs.
    /// `people` has the frame's people in `sim.people`'s order (with what the game knows
    /// about them besides); `steer`, the ways the game set itself this frame.
    pub fn frame(&mut self, sim: &Simulation, dt: f32, vehicles: &[Vehicle], people: &[PersonRecord], steer: &[(u64, Steer)], type_name: &dyn Fn(usize) -> String) {
        let time = sim.time;
        let keyframe_due = time - self.last_keyframe >= KEYFRAME_SECS;
        // the fleet: vehicles new since the last frame, and gone (at the very first frame
        // the keyframe has them all)
        self.ids.clear();
        self.ids.extend(vehicles.iter().map(|v| v.id));
        let order_changed = self.ids != self.order;
        if order_changed {
            let now: HashSet<u64> = self.ids.iter().copied().collect();
            if !self.known.is_empty() {
                for (i, v) in vehicles.iter().enumerate() {
                    if !self.known.contains(&v.id) {
                        let name = type_name(i);
                        self.line(time, Kind::Other, &RecordRef::Spawn { time, vehicle: v, type_name: &name });
                    }
                }
                let mut gone: Vec<u64> = self.known.difference(&now).copied().collect();
                gone.sort_unstable();
                for id in gone {
                    self.line(time, Kind::Other, &RecordRef::Remove { time, id });
                }
            }
            self.known = now;
            std::mem::swap(&mut self.order, &mut self.ids);
        }
        let print = parked_print(&sim.parked);
        if self.parked_print != Some(print) {
            self.parked_print = Some(print);
            self.line(time, Kind::Parked, &RecordRef::Parked { time, parked: &sim.parked });
        }
        if keyframe_due {
            self.keyframe(time, vehicles);
        }
        self.externals.clear();
        self.externals.extend(vehicles.iter().filter(|v| v.external != External::default()).map(|v| (v.id, v.external)));
        let people_fallback: Vec<PersonRecord>;
        let people = if people.len() == sim.people.len() {
            people
        } else {
            people_fallback = sim.people.iter().map(PersonRecord::from_person).collect();
            &people_fallback
        };
        let frame = FrameRef {
            time,
            dt,
            day_time: sim.day_time,
            time_scale: sim.time_scale,
            players: &sim.players,
            people,
            walkers: &sim.walkers,
            externals: &self.externals,
            steer,
            light_clocks: LightClocks(&sim.lights),
            order: order_changed.then_some(self.order.as_slice()),
        };
        self.line(time, Kind::Other, &RecordRef::Frame(frame));
    }

    /// Every vehicle as it is now.
    pub fn keyframe(&mut self, time: f64, vehicles: &[Vehicle]) {
        self.last_keyframe = time;
        self.line(time, Kind::Keyframe, &RecordRef::Keyframe { time, vehicles });
    }

    /// A record the game writes itself (a vehicle changed, a note).
    pub fn event(&mut self, rec: Record) {
        let kind = match rec {
            Record::LanesAdded { .. } | Record::LightsAdded { .. } => Kind::FileOnly,
            Record::Keyframe { .. } => Kind::Keyframe,
            Record::Parked { .. } => Kind::Parked,
            _ => Kind::Other,
        };
        let time = rec.time();
        match finite::to_string(&rec) {
            Ok(text) => self.send(Msg::Line { time, kind, text }),
            Err(e) => log::warn!("traffic recording: a record could not be written: {e}"),
        }
    }

    /// The game changed vehicle `v`'s way between frames.
    pub fn vehicle_changed(&mut self, time: f64, v: &Vehicle) {
        self.line(time, Kind::Other, &RecordRef::VehicleChanged { time, vehicle: v });
    }

    pub fn note(&mut self, time: f64, text: &str) {
        self.line(time, Kind::Other, &RecordRef::Note { time, text });
    }

    /// Lanes about to be handed to `Network::extend` (as they are before it links them),
    /// the first to be at index `first`. Only a recording to a file needs them: call this
    /// only while [`is_recording`](Self::is_recording).
    pub fn lanes_added(&mut self, time: f64, first: usize, lanes: &[Lane]) {
        if self.full.is_some() {
            self.line(time, Kind::FileOnly, &RecordRef::LanesAdded { time, first, lanes });
        }
    }

    pub fn lights_added(&mut self, time: f64, first: usize, lights: &[TrafficLightController]) {
        if self.full.is_some() {
            self.line(time, Kind::FileOnly, &RecordRef::LightsAdded { time, first, lights });
        }
    }

    /// Save the memory (the last minute) to `path`, with a header from the network as it
    /// is now. The writing happens on the recorder's thread: the file is complete after
    /// [`flush`](Self::flush) (or a moment later); a failure is logged.
    pub fn dump_ring(&mut self, net: &Network, sim: &Simulation, map: &str, path: &Path, trigger: &str) -> io::Result<PathBuf> {
        if self.tx.is_none() {
            return Err(io::Error::other("the traffic recorder is not running"));
        }
        let header = self.header_text(net, sim, map, Some(trigger));
        self.send(Msg::Dump { path: path.to_path_buf(), header });
        Ok(path.to_path_buf())
    }

    /// Wait until everything sent so far is written.
    pub fn flush(&self) {
        let (tx, rx) = mpsc::channel();
        self.send(Msg::Flush(tx));
        let _ = rx.recv();
    }

    /// The header line for the network and the lights as they are now.
    fn header_text(&mut self, net: &Network, sim: &Simulation, map: &str, trigger: Option<&str>) -> String {
        let net_part = match &self.net_cache {
            Some((n, text)) if *n == net.lanes.len() => text.clone(),
            _ => {
                let derived = DerivedRef { prev: &net.prev, conflicts: &net.conflicts, crossings: &net.crossings, walks: &net.walks, reach: &net.reach, junctions: &sim.junctions };
                let text = std::sync::Arc::new(format!(
                    "\"lanes\":{},\"derived\":{}",
                    finite::to_string(&net.lanes).unwrap_or_else(|_| "[]".into()),
                    finite::to_string(&derived).unwrap_or_else(|_| "null".into())
                ));
                self.net_cache = Some((net.lanes.len(), text.clone()));
                text
            }
        };
        format!(
            "{{\"version\":{VERSION},\"map\":{},\"left_hand\":{},\"lights\":{},\"started_day_time\":{},\"trigger\":{},\"switches\":{},\"rules\":{},{}}}",
            json(&map),
            net.left_hand,
            json(&sim.lights),
            json(&sim.day_time),
            json(&trigger),
            json(RULES),
            json(&rules_hash()),
            net_part
        )
    }
}

/// A value as JSON text.
fn json<T: Serialize + ?Sized>(v: &T) -> String {
    finite::to_string(v).unwrap_or_else(|_| "null".into())
}

impl Drop for Recorder {
    fn drop(&mut self) {
        // (the channel closes: the thread finishes the file and ends)
        self.tx = None;
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Changes when the parked cars do (order-independent: the map's order is not the same
/// from run to run).
fn parked_print(parked: &HashMap<usize, Vec<Parked>>) -> u64 {
    let mut sum = parked.len() as u64;
    for (&lane, cars) in parked {
        let mut h = (lane as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for p in cars {
            h = h.rotate_left(7) ^ (p.s.to_bits() as u64) ^ ((p.lateral.to_bits() as u64) << 32);
            h = h.wrapping_mul(0x2545_F491_4F6C_DD1D);
        }
        sum = sum.wrapping_add(h ^ cars.len() as u64);
    }
    sum
}

/// The memory: the lines since one keyframe up to the next as one compressed block.
struct Block {
    start: f64,
    data: Vec<u8>,
    /// The last parked-cars line in it.
    parked: Option<String>,
}

struct Building {
    start: f64,
    text: String,
    parked: Option<String>,
}

#[derive(Default)]
struct Ring {
    blocks: VecDeque<Block>,
    current: Option<Building>,
    /// The parked cars as of the oldest block's start.
    parked_before: Option<String>,
}

impl Ring {
    fn push(&mut self, time: f64, kind: Kind, text: &str) {
        if kind == Kind::Keyframe {
            if let Some(b) = self.current.take() {
                self.blocks.push_back(Block { start: b.start, data: compress(b.text.as_bytes()), parked: b.parked });
            }
            self.current = Some(Building { start: time, text: String::with_capacity(1 << 20), parked: None });
        }
        match self.current.as_mut() {
            Some(b) => {
                b.text.push_str(text);
                b.text.push('\n');
                if kind == Kind::Parked {
                    b.parked = Some(text.to_string());
                }
            }
            // (before the first keyframe only the parked cars count)
            None if kind == Kind::Parked => self.parked_before = Some(text.to_string()),
            None => {}
        }
        // keep the newest block that starts before the window, and everything after it
        let window = time - RING_SECS;
        while !self.blocks.is_empty() {
            let next_start = self.blocks.get(1).map(|b| b.start).or(self.current.as_ref().map(|b| b.start));
            if !next_start.is_some_and(|s| s <= window) {
                break;
            }
            if let Some(old) = self.blocks.pop_front() {
                if old.parked.is_some() {
                    self.parked_before = old.parked;
                }
            }
        }
    }

    fn dump(&self, path: &Path, header: &str) -> io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut out = ZlibEncoder::new(io::BufWriter::new(std::fs::File::create(path)?), Compression::default());
        writeln!(out, "{header}")?;
        if let Some(p) = &self.parked_before {
            writeln!(out, "{p}")?;
        }
        for b in &self.blocks {
            let mut d = flate2::read::ZlibDecoder::new(b.data.as_slice());
            io::copy(&mut d, &mut out)?;
        }
        if let Some(b) = &self.current {
            out.write_all(b.text.as_bytes())?;
        }
        out.finish()?.flush()
    }
}

fn compress(bytes: &[u8]) -> Vec<u8> {
    let mut e = ZlibEncoder::new(Vec::with_capacity(bytes.len() / 4), Compression::default());
    let _ = e.write_all(bytes);
    e.finish().unwrap_or_default()
}

type FileOut = ZlibEncoder<io::BufWriter<std::fs::File>>;

fn writer_thread(rx: mpsc::Receiver<Msg>) {
    let mut file: Option<FileOut> = None;
    let mut ring = Ring::default();
    let finish = |f: Option<FileOut>| {
        if let Some(f) = f {
            if let Err(e) = f.finish().and_then(|mut w| w.flush()) {
                log::warn!("traffic recording: {e}");
            }
        }
    };
    for m in rx {
        match m {
            Msg::Line { time, kind, text } => {
                if let Some(f) = file.as_mut() {
                    // (a keyframe flushes what came before: a recording of a game that
                    // crashed reads up to its last keyframe)
                    let ok = if kind == Kind::Keyframe { f.flush().is_ok() } else { true } && f.write_all(text.as_bytes()).is_ok() && f.write_all(b"\n").is_ok();
                    if !ok {
                        log::warn!("traffic recording: the file cannot be written any more, the recording stops");
                        file = None;
                    }
                }
                if kind != Kind::FileOnly {
                    ring.push(time, kind, &text);
                }
            }
            Msg::StartFull { file: f, header } => {
                finish(file.take());
                let mut f = ZlibEncoder::new(io::BufWriter::new(f), Compression::default());
                if f.write_all(header.as_bytes()).and_then(|_| f.write_all(b"\n")).is_ok() {
                    file = Some(f);
                }
            }
            Msg::StopFull => finish(file.take()),
            Msg::Dump { path, header } => match ring.dump(&path, &header) {
                Ok(()) => log::info!("traffic: the last minute saved to {}", path.display()),
                Err(e) => log::warn!("traffic: saving the last minute to {} failed: {e}", path.display()),
            },
            Msg::Flush(reply) => {
                if let Some(f) = file.as_mut() {
                    let _ = f.flush();
                }
                let _ = reply.send(());
            }
        }
    }
    finish(file.take());
}

/// The lines of a recording (header first). A recording cut short (the game crashed)
/// reads up to where it ends; the second value says so.
pub fn read_lines(path: &Path) -> io::Result<(Vec<String>, bool)> {
    use std::io::BufRead;
    let f = std::fs::File::open(path)?;
    let mut r = io::BufReader::new(flate2::read::ZlibDecoder::new(io::BufReader::new(f)));
    let mut lines = Vec::new();
    let mut buf = String::new();
    loop {
        buf.clear();
        match r.read_line(&mut buf) {
            Ok(0) => return Ok((lines, false)),
            Ok(_) if buf.ends_with('\n') => lines.push(buf.trim_end().to_string()),
            // a last line without its end: cut short
            Ok(_) => return Ok((lines, true)),
            Err(_) if !lines.is_empty() => return Ok((lines, true)),
            Err(e) => return Err(e),
        }
    }
}

/// Write a recording: the header and the records, compressed.
pub fn write_file(path: &Path, header: &Header, records: &[&Record]) -> io::Result<()> {
    let mut out = ZlibEncoder::new(io::BufWriter::new(std::fs::File::create(path)?), Compression::default());
    finite::to_writer(&mut out, header)?;
    out.write_all(b"\n")?;
    for r in records {
        finite::to_writer(&mut out, r)?;
        out.write_all(b"\n")?;
    }
    out.finish()?.flush()
}
