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
use crate::removal::RemoveRule;
use crate::sim::{External, Parked, Person, PlayerView, Simulation, Vehicle};
use flate2::write::ZlibEncoder;
use flate2::Compression;
use glam::{DVec2, DVec3};
use hashbrown::{HashMap, HashSet};
use omsi_sim::traffic::{AiState, Crossing, Lane, Network, TrafficLightController};
#[cfg(test)]
use serde::ser::SerializeSeq;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::VecDeque;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;

/// The format's version (the header's first field). 2: vehicles carry their claim
/// standing hysteresis (`Vehicle::claim_standing`); before, the simulation kept it. 3: the
/// lanes carry their drivable extent (`Lane::drivable`); a recording without it
/// reads back the lanes' half widths. 4: the header names the rules by their
/// hash (`Header::rules`, [`rules_hash`]), and every field added with a `serde(default)`
/// since 3 is written - `Vehicle::passed_on_go`, `change_still`,
/// `mutual`, `mutual_gap`, `ease_for`, `waiting`, `External::trip_over`,
/// `Frame::steer`, `Frame::order`; a format-3 recording may lack some (read as their
/// defaults, `Replay::load` warns). 5: a timetable bus's stand at a stop is written when it
/// pulls away ([`Record::Dwell`]); a tool of format 4 cannot read that line.
///
/// Compatibility rule: a recording is read by the tools of its format and
/// later ones. On read, a field the recording has and the code no longer knows is ignored
/// (no `deny_unknown_fields` anywhere), and a field the code has and the recording lacks
/// takes its container's default - `Vehicle`, `AiState`, `External` and `DriverParams` are
/// `serde(default)` (`Vehicle::default`, `AiState::default`: a car standing at the start of
/// lane 0; `External::default`: no input) - except a driver field, which reads as unknown
/// and then takes the preset of the vehicle's class (`DriverParams::fill_missing`, in
/// `Replay::new`: a bus's missing `comfort_brake` is a bus's). A recording of a later format than the tool's (`header.version > VERSION`,
/// e.g. format 4 in a tool of format 3) is not for that tool: `Replay::load` refuses it with that
/// message, as fields it requires may be missing; read it with the branch's own tool.
pub const VERSION: u32 = 5;
/// The rule switches a recording names (`Header::switches`), for information: the rule
/// changes once tried behind switches and accepted, now always on, and the earlier junction
/// rules. Frozen: which rules a recording was made under is [`rules_hash`]'s (format 4).
pub const RULES: &str = "competitor_filter,signal_foes,stub_links,lights_first,right_of_way,blocked_foe,stop_placement,player_chain,amber_clear";
/// The rules the traffic runs with, as a hash: FNV-1a (64 bit) over the
/// sources of the rules - the driver, perception, plan, junctions, resolver, simulation,
/// access rule (omsi-traffic) and the lane model (omsi-sim's traffic.rs and ai_motion.rs) - with comment
/// lines, blank lines and the test modules left out, so that a changed constant or rule changes it and a
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
            // (the rules only: a test module, `#[cfg(test)] mod`, and what follows are left out)
            let mut lines = src.lines().map(str::trim).peekable();
            let mut rules: Vec<&str> = Vec::new();
            while let Some(l) = lines.next() {
                if l == "#[cfg(test)]" && lines.peek().is_some_and(|n| n.starts_with("mod ")) {
                    break;
                }
                rules.push(l);
            }
            for line in rules.into_iter().filter(|l| !l.is_empty() && !l.starts_with("//")) {
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

/// The decision a person waiting at the kerb for a red light records
/// (`PersonRecord::decision`): the replay leaves them out of the walkers (the game's
/// `Humans::strollers`).
pub const RED_LIGHT: &str = "red light";

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
    /// takes the game's own links, conflicts and walks; the junction table it compiles
    /// itself after the lit stubs (`Replay::new`), the recorded one is the game's for
    /// comparison (`replay::recorded`, `traffic_cases stubs`). Without it (a fixture written
    /// by hand) it is worked out.
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
    /// The stop they wait at (`Person::waits_at`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub waits_at: Option<i64>,
    /// Their id in the game (`Person::id`; None in a recording from before it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<u32>,
    /// Walking up to a crossing to step onto it (`Person::approach`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approach: Option<(DVec2, f32, f32)>,
    /// Waiting at a kerb where they stand, with no place in its queue (the app's last fallback,
    /// off the queue grid).
    #[serde(default, skip_serializing_if = "is_false")]
    pub kerb_fallback: bool,
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
        Person { pos: self.pos, vel: self.vel, on_crossing: self.on_crossing, still: self.still, waiting_for_bus: self.waiting_for_bus, at_kerb: self.at_kerb, waits_at: self.waits_at, id: self.id, approach: self.approach }
    }

    /// A record of somebody the game told nothing more about.
    pub fn from_person(p: &Person) -> PersonRecord {
        PersonRecord { pos: p.pos, vel: p.vel, state: Cow::Borrowed(""), still: p.still, on_crossing: p.on_crossing, waiting_for_bus: p.waiting_for_bus, leg: None, decision: None, pace: 0.0, at_kerb: p.at_kerb, waits_at: p.waits_at, id: p.id, approach: p.approach, kerb_fallback: false }
    }

    /// Take the frame's own values over (the extras stay).
    pub fn set_person(&mut self, p: &Person) {
        self.pos = p.pos;
        self.vel = p.vel;
        self.on_crossing = p.on_crossing;
        self.still = p.still;
        self.waiting_for_bus = p.waiting_for_bus;
        self.at_kerb = p.at_kerb;
        self.waits_at = p.waits_at;
        self.id = p.id;
        self.approach = p.approach;
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

/// A timetable bus's stand at one of its stops, as the game saw it (format 5): what was
/// planned when it came to rest, how long it stood, and who was left on the wrong side of
/// its doors when it pulled away (`traffic_cases dwells`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Dwell {
    /// The stop (its map object id).
    pub stop: i64,
    /// The day time it came to rest there (s).
    pub arrived: f64,
    /// Seconds from then to the doors closing as planned on arrival: its own boarding time
    /// or the timetable's wait, whichever is longer.
    pub planned: f32,
    /// Seconds from arrival to the doors closing as it was.
    pub closed: f32,
    /// Seconds from arrival to pulling away.
    pub left: f32,
    /// The doors closed at the boarding cap with somebody still said to be at them.
    pub at_cap: bool,
    /// Riders aboard when it came to rest and when it pulled away.
    pub aboard_arrival: u32,
    pub aboard_departure: u32,
    /// Riders aboard still on their way out when it pulled away (carried past their stop).
    pub bound_left: u32,
    /// People who had come up for it at the stop (at the gather point or walking to its
    /// doors) and were not aboard when it pulled away.
    pub boarders_left: u32,
    /// Where it stood when it pulled away (its origin, x and y).
    pub pos: (f64, f64),
}

/// One line of a recording after the header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Record {
    Frame(Frame),
    /// Every vehicle, before the frame of the same time.
    Keyframe { time: f64, vehicles: Vec<Vehicle> },
    Spawn { time: f64, vehicle: Vehicle, type_name: String },
    /// With the rule the game took it off by (`None`: an older
    /// recording, or a removal nobody named), whether the eye saw it (in frame and not
    /// behind a building, as the game judged; the traffic's own network end: in frame) and
    /// its waiting time then (`Vehicle::waiting`).
    Remove {
        time: f64,
        id: u64,
        #[serde(default)]
        rule: Option<RemoveRule>,
        #[serde(default)]
        seen: bool,
        #[serde(default)]
        waiting: f32,
        /// Its distance from the eye (m) when the population took it off (`None`:
        /// an older recording, or not a population removal).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        from_eye: Option<f32>,
    },
    /// The game changed a vehicle's way outside the frame (a timetable bus given its next
    /// trip or let go, its route extended).
    VehicleChanged { time: f64, vehicle: Vehicle },
    /// Lanes handed to `Network::extend` (as they were before it linked them), the first
    /// of them at index `first`.
    LanesAdded { time: f64, first: usize, lanes: Vec<Lane> },
    LightsAdded { time: f64, first: usize, lights: Vec<TrafficLightController> },
    /// Lanes whose `loaded_edge` or `edge_in_sight` the game changed: (lane, loaded_edge,
    /// edge_in_sight), as they are from then on (the tiles loaded, the eye moved).
    LaneFlags { time: f64, flags: Vec<(usize, bool, bool)> },
    /// The parked cars beside the lanes, whenever they changed.
    Parked { time: f64, parked: HashMap<usize, Vec<Parked>> },
    /// Free text: what triggered a save, a stuck car's report.
    Note { time: f64, text: String },
    /// Timetable bus `id` pulled away from a stop (format 5).
    Dwell { time: f64, id: u64, dwell: Dwell },
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
            | Record::LaneFlags { time, .. }
            | Record::Parked { time, .. }
            | Record::Note { time, .. }
            | Record::Dwell { time, .. } => *time,
        }
    }
}

/// [`Record`] borrowing its parts, for writing without copying them first. Serializes
/// exactly as `Record` does.
#[derive(Serialize)]
enum RecordRef<'a> {
    /// (only the test: a frame is written from an owned `Frame`, `Msg::Frame`; this is what
    /// it must serialise to, byte for byte)
    #[cfg(test)]
    Frame(FrameRef<'a>),
    Keyframe { time: f64, vehicles: &'a [Vehicle] },
    Spawn { time: f64, vehicle: &'a Vehicle, type_name: &'a str },
    Remove {
        time: f64,
        id: u64,
        rule: Option<RemoveRule>,
        seen: bool,
        waiting: f32,
        #[serde(skip_serializing_if = "Option::is_none")]
        from_eye: Option<f32>,
    },
    VehicleChanged { time: f64, vehicle: &'a Vehicle },
    LanesAdded { time: f64, first: usize, lanes: &'a [Lane] },
    LightsAdded { time: f64, first: usize, lights: &'a [TrafficLightController] },
    LaneFlags { time: f64, flags: &'a [(usize, bool, bool)] },
    Parked { time: f64, parked: &'a HashMap<usize, Vec<Parked>> },
    Note { time: f64, text: &'a str },
    Dwell { time: f64, id: u64, dwell: &'a Dwell },
}

#[cfg(test)]
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
#[cfg(test)]
struct LightClocks<'a>(&'a [TrafficLightController]);

#[cfg(test)]
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
    /// A frame's inputs, owned, written as a line by the recorder's thread (the people are
    /// most of a frame's text: serialising them was the game thread's, now it is this one's).
    Frame { time: f64, frame: Box<Frame> },
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
    /// The rule (and seen, waiting) of each vehicle the game is taking
    /// off, named by `note_remove` before the frame that finds it gone.
    remove_rules: HashMap<u64, (RemoveRule, bool, f32, Option<f32>)>,
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
            remove_rules: HashMap::new(),
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

    /// The game is taking vehicle `id` off by `rule` (`seen`: the eye
    /// saw it; `waiting`: its waiting time): written with its `Record::Remove`.
    pub fn note_remove(&mut self, id: u64, rule: RemoveRule, seen: bool, waiting: f32) {
        self.note_remove_from(id, rule, seen, waiting, None);
    }

    /// `note_remove` with its distance from the eye (m), for the population's
    /// removals (the rule and the distance band of each, `Record::Remove::from_eye`).
    pub fn note_remove_from(&mut self, id: u64, rule: RemoveRule, seen: bool, waiting: f32, from_eye: Option<f32>) {
        // (the ring of the last minute records too: every removal is named)
        self.remove_rules.insert(id, (rule, seen, waiting, from_eye));
    }

    /// `note_remove` unless a rule was named already (the general ways off the road).
    pub fn note_remove_unnamed(&mut self, id: u64, rule: RemoveRule, seen: bool, waiting: f32, from_eye: Option<f32>) {
        self.remove_rules.entry(id).or_insert((rule, seen, waiting, from_eye));
    }

    pub fn is_recording(&self) -> bool {
        self.full.is_some()
    }

    /// The file being recorded to.
    pub fn recording(&self) -> Option<&Path> {
        self.full.as_deref()
    }

    /// The sizes of what the recorder keeps between frames: (vehicles known, removal rules
    /// waiting for their frame), for the long-run profile (`OMSI_PROFILE`).
    pub fn sizes(&self) -> (usize, usize) {
        (self.known.len(), self.remove_rules.len())
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
                    let (rule, seen, waiting, from_eye) = match self.remove_rules.remove(&id) {
                        Some((r, s, w, d)) => (Some(r), s, w, d),
                        None => (None, false, 0.0, None),
                    };
                    self.line(time, Kind::Other, &RecordRef::Remove { time, id, rule, seen, waiting, from_eye });
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
        // (copied, not written: the recorder's thread serialises it, `Record::Frame` as
        // `RecordRef::Frame` would have)
        let frame = Frame {
            time,
            dt,
            day_time: sim.day_time,
            time_scale: sim.time_scale,
            players: sim.players.clone(),
            people: people.to_vec(),
            // (not with the people's legs: the replay rebuilds the list from them)
            walkers: if people.iter().any(|p| p.leg.is_some()) { Vec::new() } else { sim.walkers.clone() },
            externals: self.externals.clone(),
            steer: steer.to_vec(),
            light_clocks: sim.lights.iter().map(|c| (c.time, c.held, c.request.clone())).collect(),
            order: order_changed.then(|| self.order.clone()),
        };
        self.send(Msg::Frame { time, frame: Box::new(frame) });
    }

    /// Every vehicle as it is now.
    pub fn keyframe(&mut self, time: f64, vehicles: &[Vehicle]) {
        self.last_keyframe = time;
        self.line(time, Kind::Keyframe, &RecordRef::Keyframe { time, vehicles });
    }

    /// The game changed vehicle `v`'s way between frames.
    pub fn vehicle_changed(&mut self, time: f64, v: &Vehicle) {
        self.line(time, Kind::Other, &RecordRef::VehicleChanged { time, vehicle: v });
    }

    pub fn note(&mut self, time: f64, text: &str) {
        self.line(time, Kind::Other, &RecordRef::Note { time, text });
    }

    /// Timetable bus `id` pulled away from a stop after `dwell`.
    pub fn dwell(&mut self, time: f64, id: u64, dwell: &Dwell) {
        self.line(time, Kind::Other, &RecordRef::Dwell { time, id, dwell });
    }

    /// Lanes about to be handed to `Network::extend` (as they are before it links them),
    /// the first to be at index `first`. Only a recording to a file needs them: call this
    /// only while [`is_recording`](Self::is_recording).
    pub fn lanes_added(&mut self, time: f64, first: usize, lanes: &[Lane]) {
        if self.full.is_some() {
            self.line(time, Kind::FileOnly, &RecordRef::LanesAdded { time, first, lanes });
        }
    }

    /// Lanes whose `loaded_edge`/`edge_in_sight` changed (lane, loaded_edge, edge_in_sight): a
    /// replay sets them at that time, so a recording replays the flags as they were.
    pub fn lane_flags(&mut self, time: f64, flags: &[(usize, bool, bool)]) {
        if self.full.is_some() && !flags.is_empty() {
            self.line(time, Kind::FileOnly, &RecordRef::LaneFlags { time, flags });
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

/// A line of text to the file being recorded (if any) and to the ring.
fn write_line(file: &mut Option<FileOut>, ring: &mut Ring, time: f64, kind: Kind, text: &str) {
    if let Some(f) = file.as_mut() {
        // (a keyframe flushes what came before: a recording of a game that
        // crashed reads up to its last keyframe)
        let ok = if kind == Kind::Keyframe { f.flush().is_ok() } else { true } && f.write_all(text.as_bytes()).is_ok() && f.write_all(b"\n").is_ok();
        if !ok {
            log::warn!("traffic recording: the file cannot be written any more, the recording stops");
            *file = None;
        }
    }
    if kind != Kind::FileOnly {
        ring.push(time, kind, text);
    }
}

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
            Msg::Frame { time, frame } => match finite::to_string(&Record::Frame(*frame)) {
                Ok(text) => write_line(&mut file, &mut ring, time, Kind::Other, &text),
                Err(e) => log::warn!("traffic recording: a record could not be written: {e}"),
            },
            Msg::Line { time, kind, text } => write_line(&mut file, &mut ring, time, kind, &text),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame written from its owned copy on the recorder's thread is the line the game
    /// thread used to write from the borrowed parts, byte for byte.
    #[test]
    fn an_owned_frame_serialises_as_the_borrowed_one_did() {
        let mut light = TrafficLightController::new(vec![vec![(1, 5.0), (3, 7.5)], vec![(3, 5.0)]], 12.5);
        light.time = 3.25;
        light.held = true;
        light.request = vec![true, false];
        let lights = vec![light];
        let players = vec![PlayerView { id: 0, pos: DVec3::new(1.5, -2.25, 0.1), heading: 33.3, half_len: 6.0, half_width: 1.2, speed: -0.7, still: 0.0, blinker: 2, signalling: 1.5, priority: true }];
        let mut walker = PersonRecord::from_person(&Person { pos: DVec3::new(10.1, 20.2, 0.0), vel: DVec2::new(0.3, -0.1), on_crossing: true, still: 0.4, waiting_for_bus: false, at_kerb: true, waits_at: Some(7), id: None, approach: None });
        walker.leg = Some((3, 1.25, 8.5));
        walker.decision = Some(Cow::Borrowed("red light"));
        let people = vec![walker, PersonRecord::from_person(&Person { pos: DVec3::new(-5.0, 6.0, 0.0), vel: DVec2::ZERO, on_crossing: false, still: 0.0, waiting_for_bus: true, at_kerb: false, waits_at: None, id: None, approach: None })];
        let externals = vec![(7u64, External { priority: true, ..Default::default() })];
        let steer = vec![(9u64, Steer { signal: 1, signal_time: 0.5, lateral_target: -1.5, lateral_ramp: (0.1, 0.2, 0.3, 0.4) })];
        let order = vec![3u64, 1, 2];
        for (order, walkers) in [(None, vec![(2usize, 0.5f32)]), (Some(order.clone()), Vec::new())] {
            let borrowed = FrameRef {
                time: 12.125,
                dt: 0.033333335,
                day_time: 30000.5,
                time_scale: 1.0,
                players: &players,
                people: &people,
                walkers: &walkers,
                externals: &externals,
                steer: &steer,
                light_clocks: LightClocks(&lights),
                order: order.as_deref(),
            };
            let before = finite::to_string(&RecordRef::Frame(borrowed)).unwrap();
            let owned = Frame {
                time: 12.125,
                dt: 0.033333335,
                day_time: 30000.5,
                time_scale: 1.0,
                players: players.clone(),
                people: people.clone(),
                walkers: walkers.clone(),
                externals: externals.clone(),
                steer: steer.clone(),
                light_clocks: lights.iter().map(|c| (c.time, c.held, c.request.clone())).collect(),
                order,
            };
            assert_eq!(finite::to_string(&Record::Frame(owned)).unwrap(), before);
        }
    }
}

#[cfg(test)]
mod remove_line_tests {
    use super::{Record, RecordRef};
    use crate::removal::RemoveRule;

    /// A removal's line carries its rule, whether the eye saw it and its
    /// distance from the eye; one with no distance is written as before, and an older line (no
    /// `from_eye`) reads as None.
    #[test]
    fn a2_the_remove_line_carries_its_distance() {
        let line = |from_eye| serde_json::to_string(&RecordRef::Remove { time: 56.0, id: 85, rule: Some(RemoveRule::FarDeadEnd), seen: true, waiting: 0.0, from_eye }).unwrap();
        let far = line(Some(869.0));
        assert!(far.contains("\"rule\":\"FarDeadEnd\"") && far.contains("\"from_eye\":869.0"), "{far}");
        assert!(!line(None).contains("from_eye"));
        match serde_json::from_str::<Record>(&far).unwrap() {
            Record::Remove { rule, seen, from_eye, .. } => assert!(rule == Some(RemoveRule::FarDeadEnd) && seen && from_eye == Some(869.0)),
            _ => panic!("not a removal"),
        }
        let old = r#"{"Remove":{"time":1.7,"id":111,"rule":"NetworkEnd","seen":false,"waiting":0.0}}"#;
        assert!(matches!(serde_json::from_str::<Record>(old).unwrap(), Record::Remove { from_eye: None, .. }));
    }

    /// A change of the lanes' edge flags is a line of its own, read back as written
    /// (a replay sets the flags at its time).
    #[test]
    fn s4_lane_flag_changes_are_recorded() {
        let flags = [(149usize, true, false), (151, false, true)];
        let line = serde_json::to_string(&RecordRef::LaneFlags { time: 12.5, flags: &flags }).unwrap();
        match serde_json::from_str::<Record>(&line).unwrap() {
            Record::LaneFlags { time, flags: f } => assert!(time == 12.5 && f == flags.to_vec()),
            _ => panic!("not lane flags: {line}"),
        }
    }

    /// A bus's stand at a stop is a line of its own (format 5), read back as written.
    #[test]
    fn b4_a_dwell_is_recorded() {
        let dwell = super::Dwell { stop: 2296442, arrived: 30338.4, planned: 9.0, closed: 120.0, left: 123.3, at_cap: true, aboard_arrival: 31, aboard_departure: 29, bound_left: 3, boarders_left: 1, pos: (461.1, -1596.2) };
        let line = serde_json::to_string(&RecordRef::Dwell { time: 952.0, id: 3127, dwell: &dwell }).unwrap();
        match serde_json::from_str::<Record>(&line).unwrap() {
            Record::Dwell { time, id, dwell: d } => assert!(time == 952.0 && id == 3127 && d == dwell),
            _ => panic!("not a dwell: {line}"),
        }
    }
}
