//! People on foot: passengers and pedestrians as agents with a goal.
//!
//! A passenger comes along the pavement (or already stands at the stop when the map
//! starts), waits at a free waiting place of the stop - the `[passpos]` points of the
//! map's `people_standing_*` markers and shelters, else places spread along the back of
//! the platform - and when a bus opens its doors there, queues at the nearest open
//! `[entry]` (a passenger who still has to buy a ticket only at one with a cash desk),
//! steps in when the doorway is free, pays or shows a pass at the desk, walks the cabin's
//! `paths.cfg` network to a free `[passpos]` (a standing place once the seats are gone),
//! rides, presses the stop button before their stop, walks to the nearest `[exit]` when
//! the bus stands there, steps out and walks away along the pavement - or waits at the
//! stop for another bus. Timetable (AI) buses carry their passengers the same way.
//! Nobody is taken away while the player can see them.
//!
//! An articulated bus is one cabin: the sections' path networks, seats and exits are put
//! together in the front section's frame with the sections straight behind each other, and
//! the front section's `[linkToPrevVeh]` point is joined to the rear section's
//! `[linkToNextVeh]` point, so people walk through the bellows to the seats and exits at the
//! back. Entries and exits are numbered front section first, which is how the stock door
//! scripts count them (the GN92's rear door is `PAX_Exit2`/`PAX_Exit3`). A point behind a
//! joint is carried by its own section, whatever the angle of the bend.
//!
//! Movement is a crowd: everybody on the same floor (the ground, or one bus) avoids
//! everybody else with the anticipatory model of `omsi_sim::crowd`, does not push into
//! somebody standing in front, speeds up, slows down and turns at a human pace, and keeps
//! to the aisle inside a bus. Doorways and the cash desk are taken one at a time, people
//! getting off go first, and somebody pressed against another for seconds slips past.
//! Every waiting state has a way out, and `OMSI_DEBUG_PAX=1` logs every change of state
//! and why somebody stands still.
//!
//! Pedestrians walk the map's pavement paths as one network (path ends that meet are
//! joined whatever their heading), wait at the kerb for a pedestrian light's green - and
//! only start across when it lasts long enough - and for approaching cars where there is
//! no light; nobody stops in the middle of the road.
//!
//! The map streams: stops, waiting places and pavements come with their tiles. The
//! pavement network grows as the traffic network does, a stop is set up again when its
//! neighbourhood changed and nobody uses it, a stop whose tile went takes its people with
//! it, and nobody stands or walks where the ground is not loaded.

use crate::ambience;
use crate::scene::World;
use crate::traffic::Traffic;
use glam::{DVec2, DVec3, Mat4, Vec3};
use hashbrown::{HashMap, HashSet};
use omsi_render::{AlphaMode, Camera, MaterialId, MeshId, Renderer, Scene};
use omsi_sim::crowd::{self, Block, CrowdParams, PathGraph, Walker};
use omsi_sim::human::{skin, Activity, HumanType};
use omsi_sim::human_omsi::{AnimInput, OmsiAnim};
use omsi_sim::traffic::{LaneKind, Network};
use omsi_sim::VehicleInstance;
use omsi_vehicle::PassengerCabin;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[path = "humans_pax.rs"]
mod pax;
use pax::*;

/// The map's traffic keeps left (its stops are on the left): see the doors of `Cabin`.
pub(crate) static LEFT_HAND: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// How far outside the bus side somebody stands at a door (m).
const DOOR_OUT: f32 = 0.5;
/// Entries and exits with door variables of their own in Omsi.exe (`PAX_Entry0..7` ...).
const OMSI_PAX_DOORS: usize = 8;
/// Body radius for the crowd outside (m): shoulders and swinging arms. With the cabin's
/// radius people on the pavement came within 0.46 m, and two walking past each other or a
/// group crossing the road merged into one another in the picture.
const BODY_OUTSIDE: f64 = 0.28;
/// [ROLLBACK pedhold-1] How far off their pavement (m) a stroller is still drawn back onto
/// it by its corridor (`path_corridor_band`).
const STROLL_BAND: f64 = 2.0;

/// [ROLLBACK around-86] The two corners (world) to walk through to get round the box `b` from
/// `here` going in `dir`: the corner at the nearer end on the side one stands at, then the one on
/// the far side. None when the way does not lead into the box, or one stands in it.
fn around_box(b: &Block, here: DVec2, dir: DVec2) -> Option<[DVec2; 2]> {
    const MARGIN: f64 = 0.8;
    let (s, c) = (b.heading.sin(), b.heading.cos());
    let (r, f) = (DVec2::new(c, -s), DVec2::new(s, c));
    if [0.9, 1.8, 3.0].iter().all(|t| !b.closest(here + dir * *t).1) {
        return None;
    }
    let d = here - b.center;
    let (x0, y0) = (d.dot(r), d.dot(f));
    if x0.abs() < b.half.x && y0.abs() < b.half.y {
        return None;
    }
    let side = if x0 < 0.0 { -1.0 } else { 1.0 };
    let dy = dir.dot(f);
    // the end that is nearer, leaning to the one the way points to
    let cost = |ye: f64| (ye - y0).abs() - 4.0 * ye.signum() * dy;
    let (hy, hx) = (b.half.y + MARGIN, b.half.x + MARGIN);
    let ye = if cost(hy) <= cost(-hy) { hy } else { -hy };
    let at = |x: f64, y: f64| b.center + r * x + f * y;
    Some([at(side * hx, ye), at(-side * hx, ye)])
}
/// [ROLLBACK r4c-P5] How the way leads past a standing box.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Along {
    /// It lies along the way: through these two points beside it (its near end, past its far
    /// end) on its free side.
    Route([DVec2; 2]),
    /// It lies along the way and leaves no room beside it within the path.
    NoRoom,
    /// It lies across the way (or not in it): round its end (`around_box`).
    Across,
}

/// [ROLLBACK r4c-P5] Past a box `b` that lies along the way `dir` from `here`: the walker's line
/// moved sideways by the box's overlap and half a metre, to the side where that stays within
/// the path (`corridor`: its axis and half width), the nearer if both do.
// TODO(r4c-L6, review 2026-10-18, low): `fits` checks only the lateral offset at `here`; the
// second waypoint is the box's length + 0.8 m ahead along the bus axis, unchecked against a
// path that bends; with no corridor any side is taken. The turn cap (P5) makes pacers stand
// (vel 0) up to 5 s instead of turning, still inside a bus's strip if they were there.
fn along_box(b: &Block, here: DVec2, dir: DVec2, corridor: Option<(DVec2, DVec2, f64)>) -> Along {
    const CLEAR: f64 = 0.5;
    let f = DVec2::new(b.heading.sin(), b.heading.cos());
    if f.dot(dir).abs() < 0.7 {
        return Along::Across;
    }
    if [0.9, 1.8, 3.0].iter().all(|t| !b.closest(here + dir * *t).1) && !b.near(here + dir * 0.9, BODY_OUTSIDE + 0.15) {
        return Along::Across;
    }
    // (in the box's own axes: the way along it, and across)
    let fd = if f.dot(dir) < 0.0 { -f } else { f };
    let n = DVec2::new(fd.y, -fd.x);
    let rel = b.center - here;
    let (c_n, c_d) = (rel.dot(n), rel.dot(fd));
    let (e_n, e_d) = (b.half.x, b.half.y);
    if c_d + e_d < 0.0 {
        return Along::Across;
    }
    let fits = |x: f64| match corridor {
        Some((a, bb, dev)) => {
            let axis = crowd::project_on_segment(here + n * x, a, bb).0;
            (here + n * x - axis).length() <= dev + 0.2
        }
        None => true,
    };
    let mut sides = [c_n - e_n - CLEAR, c_n + e_n + CLEAR];
    sides.sort_by(|a, b| a.abs().total_cmp(&b.abs()));
    let Some(x) = sides.into_iter().find(|x| fits(*x)) else { return Along::NoRoom };
    let near = (c_d - e_d).max(0.3);
    Along::Route([here + n * x + fd * near, here + n * x + fd * (c_d + e_d + 0.8)])
}

/// [ROLLBACK r4c-P5] Seconds between two turns round beside a standing vehicle, at least.
const PED_TURN_EVERY: f32 = 5.0;

/// [ROLLBACK r4c-P5] Whether somebody who last turned round `since` s ago may turn again.
fn may_turn(since: f32) -> bool {
    since >= PED_TURN_EVERY
}

/// [ROLLBACK r4c-P5] Whether the box `b` covers the path from `from` along `dir` within `ahead`
/// m (a walker's body wide).
// TODO(r4c-L5, review 2026-10-18, low): margin mismatch - this uses BODY_OUTSIDE 0.28 m, the
// crowd's in_way check 0.43 m (BODY_OUTSIDE + 0.15); with a box 0.28-0.43 m from the line in
// the first 1.5 m the kerb says go, the crowd blocks, car_wait 2 s -> turn_back -> kerb -> go:
// a 3-4 s oscillation (reviewer's probe D: covers false, in_way true). Not seen in game.
fn covers_path(b: &Block, from: DVec2, dir: DVec2, ahead: f64) -> bool {
    (0..=6).any(|k| b.near(from + dir * (ahead * k as f64 / 6.0), BODY_OUTSIDE))
}

/// Stops within this distance of the player have their people (Omsi.exe: the stop's tile
/// and the eight round the camera's, sub_61bf94).
const STOP_RANGE: f64 = 450.0;
/// Pedestrians stroll within this distance of the player (m).
const STROLL_RADIUS: f64 = 200.0;
/// [ROLLBACK strollers-2] Further off than this (m) somebody may be put down in plain view.
const APPEAR_IN_VIEW_BEYOND: f64 = 120.0;
/// How far in front of a seat's hip point somebody stands to sit down - where the feet
/// stay while seated (m).
const SEAT_FRONT: f32 = 0.34;
/// Over this distance on either side of a joint (m) a point of an articulated bus's cabin
/// moves from the frame of the section in front to the one behind.
const JOINT_BLEND: f32 = 0.5;

fn debug_pax() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        omsi_cfg::env::var_os("OMSI_DEBUG_PAX").is_some()
            || omsi_cfg::env::var_os("OMSI_DEBUG_HUMANS").is_some()
    })
}


/// Where the player looks from, for "nobody appears or vanishes in sight".
#[derive(Debug, Clone, Copy)]
pub struct Eye {
    pub pos: DVec3,
    pub fwd: DVec3,
    /// Cosine of half the diagonal field of view, with a margin.
    pub cos_half: f64,
}

impl Eye {
    pub fn of(cam: &Camera, aspect: f32) -> Eye {
        let half_v = (cam.fov_deg as f64 * 0.5).to_radians();
        let half_diag = (half_v.tan() * (1.0 + (aspect as f64).powi(2)).sqrt()).atan();
        Eye {
            pos: cam.position,
            fwd: cam.forward().as_dvec3().normalize_or_zero(),
            cos_half: (half_diag + 10f64.to_radians())
                .min(89f64.to_radians())
                .cos(),
        }
    }

    /// A wider picture than the camera's own (a triple screen's side panels): the tangents
    /// of its half-angles, horizontal and vertical.
    pub fn widened(mut self, extent: Option<(f64, f64)>) -> Eye {
        if let Some((tan_x, tan_y)) = extent {
            let half_diag = tan_x.hypot(tan_y).atan();
            self.cos_half = self.cos_half.min((half_diag + 10f64.to_radians()).min(89f64.to_radians()).cos());
        }
        self
    }
}

/// What the passengers tell a bus's scripts in a frame: about its doors, entry by entry and
/// exit by exit, and its places (see [`Humans::write_door_requests`]).
#[derive(Debug, Clone, Default)]
pub struct DoorWants {
    /// `PAX_Entry<i>_Req` / `PAX_Exit<i>_Req`: somebody wants in / out there.
    pub entry_req: Vec<bool>,
    pub exit_req: Vec<bool>,
    /// `PAX_Entry<i>_Busy` / `PAX_Exit<i>_Busy`: somebody stands in that doorway.
    pub entry_busy: Vec<bool>,
    pub exit_busy: Vec<bool>,
    /// The occupancy variables the `[passpos]` places name (#721): whether somebody is on
    /// each.
    pub places: Vec<(String, bool)>,
}

/// A bus as the passengers know it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BusId {
    Player,
    Ai(u64),
}

impl BusId {
}

/// An `[entry]` or `[exit]` of a cabin, in the bus frame.
#[derive(Debug, Clone)]
struct Door {
    /// The door's path point (the threshold) and its index.
    inside: Vec3,
    point: Option<usize>,
    /// Where somebody stands just outside, at ground level.
    outside: Vec3,
    /// +1 on the right side of the bus, -1 on the left.
    side: f32,
    /// Direction along the bus (+1 forwards) in which the queue at this door runs.
    queue_dir: f32,
    /// A passenger who still has to buy a ticket may board here (no `{noticketsale}`).
    sells: bool,
    /// `{withbutton}`: a door the passenger opens with the request button, worth walking to
    /// while it is still shut.
    button: bool,
    /// Where people getting off wait for the door to open: the path point next to it.
    wait: Vec3,
    /// The sections it belongs to that people walk between (see `Cabin::groups`).
    group: usize,
}

impl Door {
    /// Somebody standing at `p` (bus frame) is in the doorway (`PAX_Entry<n>_Busy`,
    /// `PAX_Exit<n>_Busy`, #720): within a body's width of the way through the door, from
    /// its threshold point across the bus side to the step outside - where a door's light
    /// barrier sees them. Not the queue waiting outside a shut door, nor the deck above it.
    /// It ends 0.4 m out from the bus side, clear of the line 0.5 m out that the people
    /// walking along a bus keep to (`clamp_x`), so they do not set it as they pass.
    fn in_doorway(&self, p: Vec3) -> bool {
        const BODY: f32 = 0.3;
        if p.z > self.inside.z + 1.0 {
            return false;
        }
        let a = self.inside.truncate();
        let b = glam::Vec2::new(self.outside.x - self.side * (DOOR_OUT - 0.1), self.inside.y);
        let ab = b - a;
        let t = if ab.length_squared() > 1e-6 { ((p.truncate() - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0) } else { 0.0 };
        (a + ab * t - p.truncate()).length() < BODY
    }
}

#[derive(Debug, Clone)]
struct Seat {
    /// The `[passpos]` point: a seated passenger's hip, a standing one's feet.
    pos: Vec3,
    /// The floor in front of it, where the feet go (and where a seated passenger stands
    /// before sitting down and after getting up).
    floor: Vec3,
    rot: f32,
    seated: bool,
    /// The `[passpos]`'s seat height (+0x20; 0: a standing place).
    height: f32,
    /// Its number for the scripts (`GetHumanCountOnSeat`): Omsi.exe's place in the file
    /// among the `[passpos]` and `[drivpos]`, the sections behind counted on after those
    /// in front (0x7d39a4 asks the next one for a number past its own places).
    omsi_seat: usize,
    /// openOMSI's variables of the place (#721): the one that switches it on and off, and the
    /// one its occupancy is written into.
    switch_var: Option<String>,
    taken_var: Option<String>,
    /// The sections it lies in that people walk between (see `Cabin::groups`).
    group: usize,
}

/// [ROLLBACK aidebug-89] One person for the AI debug overlays of the developer tools: where
/// the text goes, its lines, the colour, and the way ahead.
pub struct NpcDebug {
    pub at: DVec3,
    pub lines: Vec<String>,
    pub rgba: [u8; 4],
    pub path: Vec<DVec3>,
}

fn takes_text(t: &Takes) -> String {
    match t {
        Takes::Terminus => "Terminus (line records listing the sign)".to_string(),
        Takes::Nobody => "Nobody".to_string(),
        Takes::Duty { trip, next, done } => format!("Duty trip '{}' to '{}', next stop {next}, done {done}, public {}", trip.name.trim(), trip.terminus.trim(), trip.public),
    }
}

fn bus_text(id: BusId) -> String {
    match id {
        BusId::Player => "player's bus".to_string(),
        BusId::Ai(n) => format!("AI bus {n}"),
    }
}

impl Humans {
    /// The people within `range` of `from`, with what they are doing, where they go and how
    /// (the lines and the planned way, for the AI debug overlays).
    pub fn debug_npcs(&self, net: Option<&Network>, from: DVec3, range: f64) -> Vec<NpcDebug> {
        let mut out = Vec::new();
        for p in &self.people {
            if p.puppet.is_some() || (p.position - from).length() > range {
                continue;
            }
            let at = p.position + DVec3::new(0.0, 0.0, 2.1);
            let mut lines = vec![p.label()];
            let mut path: Vec<DVec3> = Vec::new();
            let rgba;
            match &p.state {
                State::Pax(x) => {
                    rgba = match x.task {
                        Task::WaitingForBus | Task::WalkingToBusstop => [255, 215, 60, 255],
                        Task::ToBus | Task::WalkingToBus => [80, 230, 255, 255],
                        Task::InBusToExit => [255, 120, 220, 255],
                        _ => [120, 255, 140, 255],
                    };
                    lines.push(format!("{} (state {}){}", x.task.name(), x.st, if p.remote { " [remote]" } else { "" }));
                    let stop = x.stop.and_then(|s| self.stops.get(&s).map(|st| (s, st)));
                    match stop {
                        Some((sid, st)) => {
                            lines.push(format!("stop {sid} '{}' -> {}", st.name.trim(), x.dest.as_deref().unwrap_or("(no destination)")));
                            match x.line.and_then(|k| st.lines.get(k).map(|l| (k, l))) {
                                Some((k, (name, termini))) => {
                                    let mut t: Vec<&String> = termini.iter().collect();
                                    t.sort();
                                    lines.push(format!("expects line record #{k} '{}', termini {:?}", name.trim(), t));
                                }
                                None => lines.push("no line record: takes the first bus listed".to_string()),
                            }
                            let listed: Vec<String> = st.buses.iter().map(|(b, inbox)| format!("{}{}", bus_text(*b), if *inbox { " (in the box)" } else { "" })).collect();
                            lines.push(format!("listed at the stop: {}", if listed.is_empty() { "none".to_string() } else { listed.join(", ") }));
                        }
                        None => {
                            if let Some(d) = &x.dest {
                                lines.push(format!("-> {d}"));
                            }
                        }
                    }
                    if let Some(b) = x.bus.or(x.inside) {
                        match self.last_buses.iter().find(|bn| bn.id == b) {
                            Some(bn) => lines.push(format!("{}: sign {:?}, takes {}", bus_text(b), bn.terminus, takes_text(&bn.takes))),
                            None => lines.push(format!("{}: gone", bus_text(b))),
                        }
                    }
                    if x.seat.is_some() || x.door.is_some() {
                        lines.push(format!("place {:?}, door {:?}", x.seat, x.door));
                    }
                    if x.inside.is_none() && x.st != 0 && !x.target_bus {
                        path = vec![p.position + DVec3::new(0.0, 0.0, 0.3), x.target + DVec3::new(0.0, 0.0, 0.3)];
                    }
                }
                State::Strolling(walk) => {
                    rgba = [190, 190, 190, 255];
                    lines.push(format!("stroll: leg {}/{}{}{}", walk.leg + 1, walk.legs.len(), if walk.roam { ", roams" } else { "" }, if walk.held > 0.0 { format!(", held {:.0} s", walk.held) } else { String::new() }));
                    if let Some(leg) = walk.legs.get(walk.leg) {
                        lines.push(format!("lane {} {:.0} -> {:.0}, at {:.1} m", leg.lane, leg.a, leg.b, walk.s));
                    }
                    if !p.why.is_empty() {
                        lines.push(format!("why: {}", p.why));
                    }
                    if let Some(net) = net {
                        for (k, leg) in walk.legs.iter().enumerate().skip(walk.leg).take(8) {
                            let len = leg.len();
                            let mut d = if k == walk.leg { walk.s.min(len) } else { 0.0 };
                            while d < len && path.len() < 160 {
                                path.push(leg.at(net, d).0 + DVec3::new(0.0, 0.0, 0.3));
                                d += 2.0;
                            }
                            if path.len() < 160 {
                                path.push(leg.at(net, len).0 + DVec3::new(0.0, 0.0, 0.3));
                            }
                        }
                    }
                }
                State::Standing => {
                    rgba = [160, 160, 255, 255];
                    lines.push("standing, looks for a pavement".to_string());
                }
                State::Idle => {
                    rgba = [160, 160, 255, 255];
                    lines.push("idle".to_string());
                }
            }
            if let Some((tgt, left)) = p.step_out {
                lines.push(format!("steps out to ({:.1}, {:.1}), {left:.1} s", tgt.x, tgt.y));
            }
            if p.around.is_some() {
                lines.push("goes round a bus or a lorry".to_string());
            }
            if p.stuck > 0.5 {
                lines.push(format!("blocked for {:.1} s", p.stuck));
            }
            out.push(NpcDebug { at, lines, rgba, path });
        }
        out
    }

    /// What the player's bus shows and how the people at the nearest stop take it (the AI
    /// debug page of the developer tools).
    pub fn debug_sign(&self, v: &VehicleInstance) -> Vec<String> {
        let mut out = Vec::new();
        let hof = v.host.hof.as_ref();
        match v.var("target_index_int") {
            Some(i) => {
                let row = if i.is_finite() && i >= 0.0 { hof.and_then(|h| h.termini.get(i.round() as usize)) } else { None };
                match row {
                    Some(t) => out.push(format!("target_index_int = {i}  ->  hof row {}: '{}'{}  (code {})", i.round() as i64, t.texture_id.trim(), if t.all_exit { "  [allexit]" } else { "" }, t.code)),
                    None => out.push(format!("target_index_int = {i}  ->  no hof row")),
                }
            }
            None => out.push("target_index_int: this vehicle has no such variable".to_string()),
        }
        let num = |n: &str| v.var(n).map(|x| format!("{x}")).unwrap_or_else(|| "-".to_string());
        out.push(format!("IBIS_TerminusIndex = {}, IBIS_TerminusCode = {}, AI_target_index = {}", num("IBIS_TerminusIndex"), num("IBIS_TerminusCode"), num("AI_target_index")));
        match self.last_buses.iter().find(|b| b.id == BusId::Player) {
            Some(bn) => {
                out.push(format!("the game takes the sign to be: {:?}", bn.terminus));
                out.push(format!("takes: {}", takes_text(&bn.takes)));
            }
            None => out.push("the player's bus is not among the buses people know yet".to_string()),
        }
        match &self.duty {
            Some((trip, next, done)) => out.push(format!("duty: trip '{}' to '{}', {} stops, next {next}, done {done}, public {}", trip.name.trim(), trip.terminus.trim(), trip.stops.len(), trip.public)),
            None => out.push("duty: none (free drive)".to_string()),
        }
        let nearest = self.stops.iter().min_by(|a, b| (a.1.pos - v.position).length().total_cmp(&(b.1.pos - v.position).length()));
        if let Some((id, s)) = nearest.filter(|(_, s)| (s.pos - v.position).length() < 120.0) {
            let waiting = self.people.iter().filter(|p| matches!(&p.state, State::Pax(x) if x.stop == Some(*id) && x.inside.is_none())).count();
            out.push(format!("nearest stop {id} '{}' {:.0} m away, length {:.0} m, {waiting} people there, {} places", s.name.trim(), (s.pos - v.position).length(), s.length, s.spots.len()));
            let listed: Vec<String> = s.buses.iter().map(|(b, inbox)| format!("{}{}", bus_text(*b), if *inbox { " in the box" } else { " not in the box" })).collect();
            out.push(format!("  listed: {}", if listed.is_empty() { "none (not near, wrong way, or out of service)".to_string() } else { listed.join(", ") }));
            for (name, termini) in s.lines.iter().take(8) {
                let mut t: Vec<&String> = termini.iter().collect();
                t.sort();
                out.push(format!("  line record '{}': termini {:?}", name.trim(), t));
            }
        }
        out
    }
}

/// A cabin for the developer tools' picture (see `Humans::debug_cabin`).
pub struct CabinDebug {
    pub points: Vec<DVec3>,
    pub links: Vec<(usize, usize)>,
    /// Each place's point and whether it is a seat (`[passpos]` height above 0) or a standing place.
    pub seats: Vec<(DVec3, bool)>,
    /// What each path point is for (same order as `points`).
    pub roles: Vec<PointRole>,
    /// [ROLLBACK pathid-117] each point's section (0 the front one) and its number in that section's
    /// own `paths.cfg` (what the file calls it, not counted from the front of the train).
    pub sections: Vec<usize>,
    pub local: Vec<usize>,
    /// The cabin's devices: the `[ticket_sale]`, `[stamper]`, `[ticket_sale_money_point]` and
    /// `[ticket_sale_change_point]` places (with `_2`): the world point, the kind and a note.
    pub devices: Vec<(DVec3, &'static str, String)>,
}

/// What a path point is used for besides being part of the paths.
#[derive(Clone, Default)]
pub struct PointRole {
    /// `[entry]` doors ending here: (the door's number, `{noticketsale}`, `{withbutton}`).
    pub entries: Vec<(usize, bool, bool)>,
    /// `[exit]` doors ending here (their numbers).
    pub exits: Vec<usize>,
    /// `[linkToNextVeh]` / `[linkToPrevVeh]` of a section (articulated buses, trailers).
    pub link_next: bool,
    pub link_prev: bool,
    /// Where a passenger stands to buy a ticket (`[ticket_sale]`), at a validator (`[stamper]`).
    pub sale: bool,
    pub stamper: bool,
}

/// A device of the cabin kept for the developer tools: kind, place in the cabin's frame, the
/// path point it belongs to (if any) and a note.
struct DebugPoint {
    kind: &'static str,
    pos: Vec3,
    point: Option<usize>,
    note: String,
}

/// What a section's files said when the cabin was made: its `[passengercabin]` and its paths.
type CabinSource = (Option<PassengerCabin>, Option<omsi_vehicle::VehiclePaths>);

/// What a cabin made again keeps from the one it replaces (see `Humans::reload_cabin`).
#[derive(Clone, Copy)]
enum Keep<'a> {
    Nothing,
    /// The paths as they were: only the `[passengercabin]` files are read again.
    Paths(&'a [CabinSource]),
    /// The `[passengercabin]` as it was: only the paths are read again.
    Cabin(&'a [CabinSource]),
}

/// What the developer tools can have read again while the game runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CabinReload {
    /// The `[passengercabin]` file: places, doors, validators, the cash desk.
    Cabin,
    /// The `paths.cfg`: the network the passengers walk on.
    Paths,
}

/// What passengers need to know about one vehicle type's cabin.
struct Cabin {
    /// Each section's files as they were read (for `Keep`).
    sources: Vec<CabinSource>,
    data: PassengerCabin,
    graph: PathGraph,
    links: Vec<(i32, i32, bool)>,
    /// Each link's footstep sounds: its section's `[stepsoundpack]` named by the link's
    /// `[next_stepsound]` (index into `step_packs`), none where the paths.cfg gives none -
    /// Omsi.exe hears no steps there - and on the joint between two sections.
    link_pack: Vec<Option<usize>>,
    step_packs: Vec<Arc<[String]>>,
    entries: Vec<Door>,
    exits: Vec<Door>,
    /// Where a passenger stands at the cash desk, its path point, and the heading (bus
    /// frame) they face: between the desk top, where the money goes, and the driver.
    desk: Option<(Vec3, Option<usize>, f64)>,
    seats: Vec<Seat>,
    /// The sections (one for a rigid bus), front first; everything above is in the
    /// unfolded frame of the front section.
    parts: Vec<CabinPart>,
    /// How many groups of sections people walk between (one but where a trailer hangs on
    /// that nobody walks into from the bus, #718), and each path point's group: a passenger
    /// gets in, rides and gets out within one.
    groups: usize,
    point_group: Vec<usize>,
    /// [ROLLBACK pathid-117] each path point's section (front first) and its number in that section's
    /// own `paths.cfg`, for the developer tools
    point_section: Vec<usize>,
    point_local: Vec<usize>,
    /// Each link's room height (`[next_roomheight]`; 2 m before any).
    link_room: Vec<f32>,
    /// The routing tables of the path network (sub_72410c).
    routes: Vec<Vec<RouteLink>>,
    /// The validators and the cash desk: (path point, device). Omsi.exe keeps one of each,
    /// the last of the file (cabin +0x14/+0x18, +0x28/+0x2c); every `[stamper]` of every
    /// section is kept here, and a passenger stamps at the one nearest the door they came
    /// in by (`Cabin::nearest_stamper`, #722) - with one, that one.
    stampers: Vec<(Option<usize>, Vec3)>,
    sale: Option<(Option<usize>, Vec3)>,
    /// Where the money goes (+0x38) and where the change is taken from (+0x58), with the
    /// money point's spread.
    money_point: Option<Vec3>,
    money_var: Option<(Vec3, [f32; 2])>,
    change_point: Option<Vec3>,
    /// For the developer tools: the path points that link to the next / previous vehicle (the
    /// bool: next), and every ticket sale, stamper, money and change point of every section.
    link_marks: Vec<(usize, bool)>,
    debug_points: Vec<DebugPoint>,
}

/// The people on each seat by the scripts' numbers (`Seat::omsi_seat`, the `[drivpos]`
/// counted with the `[passpos]`), from the places (indices into `seats`) taken by people
/// sitting there. (Counted by the `[passpos]` alone, every seat of a cabin with the
/// driver's place first was one off: a tip-up seat folded down under the next one.)
fn seat_numbers(seats: &[Seat], sitting: impl Iterator<Item = usize>) -> Vec<u32> {
    let n = seats.iter().map(|s| s.omsi_seat + 1).max().unwrap_or(0);
    let mut out = vec![0u32; n];
    for k in sitting {
        if let Some(c) = seats.get(k).and_then(|s| out.get_mut(s.omsi_seat)) {
            *c += 1;
        }
    }
    out
}

/// The places of a bus its scripts have switched off: a `[passpos]` naming a variable of
/// its own that is 0 now (#721), by index into the cabin's.
/// (A place in a section of its own without an entry or without an exit is off too: nobody
/// could get there, or out of it again, #718.)
fn places_off(v: &VehicleInstance, cabin: &Cabin) -> Vec<bool> {
    let unreached = |g: usize| cabin.groups > 1 && (!cabin.entries.iter().any(|e| e.group == g) || !cabin.exits.iter().any(|e| e.group == g));
    cabin
        .seats
        .iter()
        .map(|s| s.switch_var.as_deref().and_then(|n| v.var(n)).is_some_and(|x| x == 0.0) || unreached(s.group))
        .collect()
}

/// The occupancy variables of the places of bus `bn` that name one (#721), and whether
/// somebody is on each: a rider at that place (`sitting`: the places of its riders there).
fn places_taken(bn: &BusNow, sitting: &[(BusId, usize)]) -> Vec<(String, bool)> {
    bn.cabin
        .seats
        .iter()
        .enumerate()
        .filter_map(|(k, s)| s.taken_var.as_ref().map(|n| (n.clone(), sitting.contains(&(bn.id, k)))))
        .collect()
}

/// Which doorways of bus `bn` somebody stands in (entries, exits; see `Door::in_doorway`):
/// of the people `at` (inside a bus and where in its frame, or where in the world), its
/// riders and those outside close enough to it.
fn doorways_taken(bn: &BusNow, at: &[(Option<BusId>, DVec3)]) -> (Vec<bool>, Vec<bool>) {
    let mut entries = vec![false; bn.cabin.entries.len()];
    let mut exits = vec![false; bn.cabin.exits.len()];
    // (beyond the bus's own length round its origin nobody outside can be in a doorway)
    let reach = bn.half.length() + bn.centre.length() + bn.trailers.iter().map(|t| t.offset.length() as f64).fold(0.0, f64::max) + 2.0;
    for (inside, pos) in at {
        let local = match inside {
            Some(b) if *b == bn.id => pos.as_vec3(),
            Some(_) => continue,
            None if (*pos - bn.pos).truncate().length() < reach => bn.to_local(*pos),
            None => continue,
        };
        for (k, d) in bn.cabin.entries.iter().enumerate() {
            entries[k] |= d.in_doorway(local);
        }
        for (k, d) in bn.cabin.exits.iter().enumerate() {
            exits[k] |= d.in_doorway(local);
        }
    }
    (entries, exits)
}

/// A section of an articulated bus in its cabin's unfolded frame.
#[derive(Debug, Clone, Copy)]
struct CabinPart {
    /// Where the section's own origin lies.
    offset: Vec3,
    /// The unfolded y of the joint in front of it (the front section: none, +inf).
    joint_y: f32,
}

/// One vehicle of a coupled train as a cabin is put together from it: its definition, its
/// origin in the front vehicle's unfolded frame, and the unfolded y of its front joint.
type TrainPart<'a> = (&'a omsi_vehicle::Vehicle, Vec3, f32);

/// The sections of `v` passengers can walk through, front first: the vehicle and every
/// coupled part straight behind it (a part coupled the wrong way round and all behind it
/// are left out).
fn train_parts(v: &VehicleInstance) -> Vec<TrainPart<'_>> {
    let mut out: Vec<TrainPart<'_>> = vec![(&v.ty.def, Vec3::ZERO, f32::INFINITY)];
    let mut offset = Vec3::ZERO;
    for t in &v.trailers {
        if t.reversed {
            break;
        }
        let (back, front) = t.couplings();
        let joint_y = offset.y + back.y;
        offset += back - front;
        out.push((&t.ty.def, offset, joint_y));
    }
    out
}

impl Cabin {
    /// The cabin of a train of vehicles (see [`train_parts`]): the front one's, with the
    /// sections behind joined on as far as they have a cabin and a path network.
    fn load_train(parts: &[TrainPart<'_>]) -> Option<Cabin> {
        Self::load_train_with(parts, Keep::Nothing)
    }

    /// As [`Cabin::load_train`], keeping from an earlier cabin what `keep` says.
    fn load_train_with(parts: &[TrainPart<'_>], keep: Keep<'_>) -> Option<Cabin> {
        let (lead, _, _) = parts.first()?;
        let mut sources: Vec<CabinSource> = Vec::new();
        let load_cabin = |k: usize, def: &omsi_vehicle::Vehicle| -> Option<PassengerCabin> {
            if let Keep::Cabin(old) = keep {
                if let Some(Some(c)) = old.get(k).map(|s| &s.0) {
                    return Some(c.clone());
                }
            }
            let rel = def.passenger_cabin.as_ref()?;
            PassengerCabin::load(&omsi_cfg::resolve_path(def.dir(), rel))
                .map_err(|e| log::warn!("{e}"))
                .ok()
        };
        let load_paths = |k: usize, def: &omsi_vehicle::Vehicle| {
            if let Keep::Paths(old) = keep {
                if let Some(Some(p)) = old.get(k).map(|s| &s.1) {
                    return Some(p.clone());
                }
            }
            def.paths.as_ref().and_then(|rel| {
                omsi_vehicle::VehiclePaths::load(&omsi_cfg::resolve_path(def.dir(), rel))
                    .map_err(|e| log::warn!("{e}"))
                    .ok()
            })
        };
        let data = load_cabin(0, lead)?;
        let mut points: Vec<Vec3> = Vec::new();
        let mut links: Vec<(i32, i32, bool)> = Vec::new();
        let mut link_pack: Vec<Option<usize>> = Vec::new();
        let mut link_room: Vec<f32> = Vec::new();
        let mut step_packs: Vec<Arc<[String]>> = Vec::new();
        // (merged path point or -1, sells tickets, {withbutton}, half width of the section,
        // group)
        let mut entry_points: Vec<(i32, bool, bool, f32, usize)> = Vec::new();
        let mut exit_points: Vec<(i32, f32, usize)> = Vec::new();
        let mut places: Vec<(omsi_vehicle::cabin::PassPos, Vec3, usize, usize)> = Vec::new();
        let mut group = 0usize;
        let mut point_group: Vec<usize> = Vec::new();
        let mut point_section: Vec<usize> = Vec::new();
        let mut point_local: Vec<usize> = Vec::new();
        // (the script seat numbers of the sections in front)
        let mut seat_base = 0usize;
        let mut cabin_parts: Vec<CabinPart> = Vec::new();
        let mut stampers: Vec<(Option<usize>, Vec3)> = Vec::new();
        let mut link_marks: Vec<(usize, bool)> = Vec::new();
        let mut debug_points: Vec<DebugPoint> = Vec::new();
        // the point of the section in front that leads on to the next one, and that one as
        // its cabin gives it (`[linkToPrevVeh]`)
        let mut rear_link: Option<usize> = None;
        let mut rear_declared: Option<usize> = None;
        for (k, (def, offset, joint_y)) in parts.iter().enumerate() {
            let cab = if k == 0 {
                Some(data.clone())
            } else {
                load_cabin(k, def)
            };
            let Some(cab) = cab else { break };
            let paths = load_paths(k, def);
            sources.push((Some(cab.clone()), paths.clone()));
            let (own, own_links, own_steps, own_packs, own_rooms): (Vec<Vec3>, Vec<(i32, i32, bool)>, Vec<i32>, Vec<Vec<String>>, Vec<f32>) = match paths {
                Some(p) => (
                    p.points
                        .iter()
                        .map(|q| Vec3::from(q.pos) + *offset)
                        .collect(),
                    p.links,
                    p.link_step_sound,
                    p.step_sound_packs,
                    p.link_room_height,
                ),
                None => (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()),
            };
            let base = points.len();
            let valid = |i: i32| (i >= 0 && (i as usize) < own.len()).then_some(base + i as usize);
            let end = |front: bool| {
                (0..own.len())
                    .filter(|i| own[*i].x.abs() < 0.6)
                    .max_by(|a, b| {
                        if front {
                            own[*a].y.total_cmp(&own[*b].y)
                        } else {
                            own[*b].y.total_cmp(&own[*a].y)
                        }
                    })
                    .map(|i| base + i)
            };
            if k > 0 {
                // through the joint: from the front section's [linkToPrevVeh] point to this
                // one's [linkToNextVeh] point (the frontmost aisle point when it has none). A
                // trailer's hitch (`[coupling_front_character]` type 0, a lorry's coupling) is
                // no bellows: walked through only where both cabins give those points, as
                // Omsi.exe joins coupled cabins (0x7d172c) - joined at the aisle's ends, the
                // passengers of a bus walked through its back wall into the trailer behind
                let hitch = def.coupling_front_character.is_some_and(|c| c[3] == 0.0);
                let front = cab.link_to_next_veh.and_then(valid).or_else(|| if hitch { None } else { end(true) });
                // [ROLLBACK pathid-117] a bellows is walked through where one of the two cabins names its
                // joint (`[linkToNextVeh]` / `[linkToPrevVeh]`): with neither named, the sections are
                // separate (before: joined at the aisle's ends all the same, so a link taken out of the
                // files was still there). OMSI_PAX_GUESS_LINK=1: as before.
                let named =
                    rear_declared.is_some() || cab.link_to_next_veh.and_then(valid).is_some();
                let guess = named || omsi_cfg::env::var_os("OMSI_PAX_GUESS_LINK").is_some();
                match (
                    if hitch {
                        rear_declared
                    } else if guess {
                        rear_link
                    } else {
                        None
                    },
                    front,
                ) {
                    (Some(a), Some(b)) => {
                        links.push((a as i32, b as i32, false));
                        link_pack.push(None);
                        link_room.push(2.0);
                    }
                    // no way through: a section of its own, which people get into and out
                    // of by its own doors and ride in (#718; it stayed empty)
                    _ if !own.is_empty() => group += 1,
                    // nowhere to walk in it
                    _ => break,
                }
            }
            points.extend(own.iter().copied());
            point_group.extend(std::iter::repeat_n(group, own.len()));
            point_section.extend(std::iter::repeat_n(k, own.len()));
            point_local.extend(0..own.len());
            links.extend(
                own_links
                    .iter()
                    .map(|(a, b, o)| (a + base as i32, b + base as i32, *o)),
            );
            let pack_base = step_packs.len();
            link_pack.extend((0..own_links.len()).map(|i| {
                let n = own_steps.get(i).copied().unwrap_or(-1);
                (n >= 0 && (n as usize) < own_packs.len()).then(|| pack_base + n as usize)
            }));
            step_packs.extend(own_packs.into_iter().map(Arc::from));
            link_room.extend((0..own_links.len()).map(|i| own_rooms.get(i).copied().unwrap_or(2.0)));
            rear_declared = cab.link_to_prev_veh.and_then(valid);
            rear_link = rear_declared.or_else(|| end(false));
            let half = def
                .bounding_box
                .map(|b| b[0] * 0.5)
                .unwrap_or_else(|| own.iter().map(|p| p.x.abs()).fold(1.2, f32::max));
            let shift = |i: i32| valid(i).map(|m| m as i32).unwrap_or(-1);
            if let Some(i) = cab.link_to_next_veh.and_then(valid) {
                link_marks.push((i, true));
            }
            if let Some(i) = cab.link_to_prev_veh.and_then(valid) {
                link_marks.push((i, false));
            }
            for t in &cab.ticket_sales {
                debug_points.push(DebugPoint { kind: "ticket_sale", pos: Vec3::from(t.pos) + *offset, point: valid(t.path_point), note: String::new() });
            }
            for t in &cab.stampers {
                debug_points.push(DebugPoint { kind: "stamper", pos: Vec3::from(t.pos) + *offset, point: valid(t.path_point), note: String::new() });
            }
            for (list, base_kind) in [(&cab.money_points, "ticket_sale_money_point"), (&cab.change_points, "ticket_sale_change_point")] {
                for m in list {
                    let kind = match (base_kind, m.parent.is_some()) {
                        ("ticket_sale_money_point", false) => "ticket_sale_money_point",
                        ("ticket_sale_money_point", true) => "ticket_sale_money_point_2",
                        (_, false) => "ticket_sale_change_point",
                        (_, true) => "ticket_sale_change_point_2",
                    };
                    let note = format!("var {} {}{}", m.var[0], m.var[1], m.parent.as_deref().map(|p| format!(", parent {p}")).unwrap_or_default());
                    debug_points.push(DebugPoint { kind, pos: Vec3::from(m.pos) + *offset, point: None, note });
                }
            }
            entry_points.extend(
                cab.entries
                    .iter()
                    .map(|e| (shift(e.path_point), !e.no_ticket_sale, e.with_button, half, group)),
            );
            exit_points.extend(cab.exits.iter().map(|e| (shift(*e), half, group)));
            places.extend(cab.pass_positions.iter().map(|p| (p.clone(), *offset, seat_base + p.file_index, group)));
            stampers.extend(cab.stampers.iter().map(|st| (valid(st.path_point), Vec3::from(st.pos) + *offset)));
            seat_base += cab.pass_positions.len() + cab.driver_positions.len();
            cabin_parts.push(CabinPart {
                offset: *offset,
                joint_y: *joint_y,
            });
        }
        let graph = PathGraph::new(points.clone(), &links);
        // (the side of the road the stops are on: where a door's own point does not tell)
        let kerb = if LEFT_HAND.load(std::sync::atomic::Ordering::Relaxed) { -1.0f32 } else { 1.0 };
        let door = |pp: i32, sells: bool, button: bool, half_width: f32, group: usize| -> Door {
            let point = (pp >= 0 && (pp as usize) < points.len()).then_some(pp as usize);
            let inside = point
                .map(|i| points[i])
                .unwrap_or(Vec3::new(kerb * (half_width - 0.1), 4.0, 0.4));
            // A door's side is the side of its entry point; one in the middle of the aisle
            // (or none) is taken to open to the kerb - on the left where the traffic keeps
            // left. (Always the right: a UK bus whose entry point lies on the aisle had the
            // people come to its door from the road side, round the bus.)
            let side = if inside.x.abs() < 0.6 { kerb } else if inside.x >= 0.0 { 1.0 } else { -1.0 };
            let outside = Vec3::new(side * (half_width + DOOR_OUT), inside.y, 0.0);
            // the aisle point next to the door: its neighbour nearest the middle
            let wait_point = point
                .and_then(|i| {
                    graph
                        .neighbours(i)
                        .into_iter()
                        .min_by(|a, b| points[*a].x.abs().total_cmp(&points[*b].x.abs()))
                })
                .filter(|&w| (points[w].x - inside.x).abs() > 0.3);
            // (no aisle point linked beside the door - the W906's door steps lead straight on
            // along it: the nearest path point off the door's line, else a step inwards)
            let wait = wait_point.map(|w| points[w]).unwrap_or_else(|| {
                points
                    .iter()
                    .filter(|p| (p.x - inside.x).abs() > 0.3 && (p.truncate() - inside.truncate()).length() < 1.2 && (p.z - inside.z).abs() < 0.6)
                    .min_by(|a, b| (a.truncate() - inside.truncate()).length().total_cmp(&(b.truncate() - inside.truncate()).length()))
                    .copied()
                    .unwrap_or(Vec3::new(inside.x - side * 0.7, inside.y, inside.z))
            });
            Door {
                inside,
                point,
                outside,
                side,
                queue_dir: -1.0,
                sells,
                button,
                wait,
                group,
            }
        };
        let mut entries: Vec<Door> = entry_points
            .iter()
            .map(|(pp, sells, button, half, g)| door(*pp, *sells, *button, *half, *g))
            .collect();
        let exits: Vec<Door> = exit_points
            .iter()
            .map(|(pp, half, g)| door(*pp, false, false, *half, *g))
            .collect();
        // two leaves of one door: the queue of the front leaf runs forwards, the other's back,
        // so that the two lines do not stand in each other
        for i in 0..entries.len() {
            let partner = (0..entries.len()).find(|&j| {
                j != i
                    && entries[j].side == entries[i].side
                    && (entries[j].inside.y - entries[i].inside.y).abs() < 1.4
            });
            entries[i].queue_dir = match partner {
                Some(j) if entries[j].inside.y < entries[i].inside.y => 1.0,
                _ => -1.0,
            };
        }
        let desk = data.ticket_sales.first().map(|ts| {
            let top = Vec3::from(ts.pos);
            let by_point = usize::try_from(ts.path_point)
                .ok()
                .and_then(|i| points.get(i).map(|p| (i, *p)))
                .filter(|(_, p)| (top.truncate() - p.truncate()).length() < 3.0);
            let (stand, pi) = match by_point {
                Some((i, p)) => (p, Some(i)),
                None => {
                    // no usable path point: the nearest one on the entry floor, else the floor by the desk
                    let floor = entries.first().map(|e| e.inside.z).unwrap_or(0.4);
                    let near = points
                        .iter()
                        .enumerate()
                        .filter(|(_, p)| (p.z - floor).abs() < 0.6)
                        .min_by(|a, b| {
                            (a.1.truncate() - top.truncate())
                                .length()
                                .total_cmp(&(b.1.truncate() - top.truncate()).length())
                        });
                    match near {
                        Some((i, p)) => (*p, Some(i)),
                        None => (Vec3::new(top.x + 0.4, top.y, floor), None),
                    }
                }
            };
            let target = match data.driver_positions.first() {
                Some(d) => (top + Vec3::from(d.pos)) * 0.5,
                None => top,
            };
            let d = target - stand;
            let face = if d.truncate().length() > 0.05 {
                (d.x as f64).atan2(d.y as f64).to_degrees()
            } else {
                -90.0
            };
            (stand, pi, face)
        });
        let seats = places
            .iter()
            .map(|(p, offset, omsi_seat, group)| {
                let pos = Vec3::from(p.pos) + *offset;
                let seated = p.height > 0.01;
                let floor = if seated {
                    let r = p.rot.to_radians();
                    Vec3::new(
                        pos.x + r.sin() * SEAT_FRONT,
                        pos.y + r.cos() * SEAT_FRONT,
                        pos.z - p.height,
                    )
                } else {
                    pos
                };
                Seat {
                    pos,
                    floor,
                    rot: p.rot,
                    seated,
                    height: p.height,
                    omsi_seat: *omsi_seat,
                    switch_var: p.switch_var.clone(),
                    taken_var: p.taken_var.clone(),
                    group: *group,
                }
            })
            .collect();
        let routes = build_routes(graph.points.len(), &links);
        let point_of = |i: i32| usize::try_from(i).ok().filter(|i| *i < graph.points.len());
        let sale = data.ticket_sales.last().map(|st| (point_of(st.path_point), Vec3::from(st.pos)));
        let money_point = data.money_points.last().map(|m| Vec3::from(m.pos));
        let money_var = data.money_points.last().map(|m| (Vec3::from(m.pos), m.var));
        let change_point = data.change_points.last().map(|m| Vec3::from(m.pos));
        Some(Cabin {
            sources,
            data,
            graph,
            links,
            link_pack,
            step_packs,
            entries,
            exits,
            desk,
            seats,
            parts: cabin_parts,
            groups: group + 1,
            point_group,
            point_section,
            point_local,
            link_room,
            routes,
            stampers,
            sale,
            money_point,
            money_var,
            change_point,
            link_marks,
            debug_points,
        })
    }

    /// Every point of the path network (Omsi.exe's list +0xc of the paths).
    fn all_points(&self) -> Vec<Option<usize>> {
        (0..self.graph.points.len()).map(Some).collect()
    }











}

/// Whether the straight way from `a` to `b` goes over a carriageway: across the centre
/// line of a street lane (walking along the kerb on the carriageway's edge does not).
fn crosses_street(net: &Network, a: DVec2, b: DVec2) -> bool {
    let mut cells: Vec<(i32, i32)> = Vec::new();
    for p in [a, b, (a + b) * 0.5] {
        let c = Network::grid_cell(p.extend(0.0));
        if !cells.contains(&c) {
            cells.push(c);
        }
    }
    let mut seen: Vec<usize> = Vec::new();
    for c in cells {
        for &i in net.grid.get(&c).map(|v| v.as_slice()).unwrap_or(&[]) {
            if seen.contains(&i) {
                continue;
            }
            seen.push(i);
            let l = &net.lanes[i];
            if l.kind != LaneKind::Street {
                continue;
            }
            if l
                .points
                .windows(2)
                .any(|w| segments_cross(a, b, w[0].truncate(), w[1].truncate()))
            {
                return true;
            }
        }
    }
    false
}


/// Whether the segments `a`-`b` and `c`-`d` cross.
fn segments_cross(a: DVec2, b: DVec2, c: DVec2, d: DVec2) -> bool {
    let side = |p: DVec2, q: DVec2, r: DVec2| (q - p).perp_dot(r - p);
    let (d1, d2) = (side(c, d, a), side(c, d, b));
    let (d3, d4) = (side(a, b, c), side(a, b, d));
    d1 * d2 < 0.0 && d3 * d4 < 0.0
}

/// A bus as the passengers see it this frame.
#[derive(Clone)]
struct BusNow {
    id: BusId,
    next_stop: Option<RequestStop>,
    cabin: Arc<Cabin>,
    pos: DVec3,
    rot: Mat4,
    heading: f64,
    /// m/s, forwards.
    speed: f64,
    entry_open: Vec<bool>,
    exit_open: Vec<bool>,
    /// The doors a walker may use (another player's bus: its doors as they are, while
    /// `entry_open` stays shut for the passengers here); None: as `entry_open`/`exit_open`.
    walk_open: Option<(Vec<bool>, Vec<bool>)>,
    interior: f32,
    /// The saloon's air and the light outside, for what boarding passengers say.
    air: CabinAir,
    /// Half extents across / along and the centre of its bounding box (bus frame).
    half: DVec2,
    centre: DVec2,
    /// Acceleration of the floor (bus frame: x to the right, y forwards; m/s²).
    accel: DVec2,
    /// The sections behind the front one (the cabin's parts after the first).
    trailers: Vec<PartFrame>,
    /// The terminus it shows, by name (Omsi.exe's bus +0x7bc). None: "$allexit$" - the
    /// scripts' `target_index_int` names a hof terminus added with `[addterminus_allexit]`
    /// ("Nicht einsteigen", a works trip) - or none; no timetable target has it.
    terminus: Option<String>,
    /// Whom it takes on at the stops.
    takes: Takes,
    /// The places its scripts have switched off (see `places_off`; empty: none).
    places_off: Vec<bool>,
}

/// What passengers feel stepping into a bus (OMSI reads the same fields: the vehicle's
/// `Cabinair_Temp` and `Cabinair_relHum`, the weather's temperature and the daylight).
#[derive(Debug, Clone, Copy, Default)]
struct CabinAir {
    /// °C, when the bus keeps its cabin air (every bus does: its script or the engine).
    temp: Option<f32>,
    /// Relative humidity, a fraction.
    rel_hum: f32,
    /// The temperature outside (°C).
    outside: f32,
    /// `Envir_Brightness`: the daylight, 0 dark .. 1.
    brightness: f32,
}

impl CabinAir {
    fn of(v: &VehicleInstance) -> CabinAir {
        CabinAir {
            temp: v.var("Cabinair_Temp").filter(|t| t.is_finite()),
            rel_hum: v.var("Cabinair_relHum").filter(|h| h.is_finite()).unwrap_or(0.0),
            outside: v.host.temperature,
            brightness: v.var("Envir_Brightness").unwrap_or(1.0),
        }
    }
}

/// Where a rear section of a bus is this frame, with its place in the cabin.
#[derive(Debug, Clone, Copy)]
struct PartFrame {
    pos: DVec3,
    rot: Mat4,
    heading: f64,
    offset: Vec3,
    joint_y: f32,
    /// Half extents across / along and the centre of its bounding box (own frame).
    half: DVec2,
    centre: DVec2,
}

/// The rear sections of `v` that are parts of `cabin`, as they stand now.
fn part_frames(v: &VehicleInstance, cabin: &Cabin) -> Vec<PartFrame> {
    cabin
        .parts
        .iter()
        .skip(1)
        .zip(&v.trailers)
        .map(|(cp, t)| {
            let bb =
                t.ty.def
                    .bounding_box
                    .unwrap_or([2.5, 7.0, 3.0, 0.0, 0.0, 1.5]);
            PartFrame {
                pos: t.position,
                rot: t.body_rotation(),
                heading: t.heading,
                offset: cp.offset,
                joint_y: cp.joint_y,
                half: DVec2::new(bb[0] as f64 * 0.5, bb[1] as f64 * 0.5),
                centre: DVec2::new(bb[3] as f64, bb[4] as f64),
            }
        })
        .collect()
}

/// How far into the frame of the section behind joint `t` a cabin point `y` lies: 0 in
/// front of the joint's blend, 1 behind it.
fn behind(t: &PartFrame, y: f32) -> f32 {
    ((JOINT_BLEND - (y - t.joint_y)) / (2.0 * JOINT_BLEND)).clamp(0.0, 1.0)
}

/// Where a point of a cabin (unfolded frame) is in the world: the front section carries
/// what lies ahead of the first joint, a rear section what lies behind its joint, and near
/// a joint the two are blended, so that somebody walking through the bellows moves on
/// smoothly however far the bus is bent.
fn train_point(pos: DVec3, rot: &Mat4, trailers: &[PartFrame], local: Vec3) -> DVec3 {
    let mut here = pos + rot.transform_point3(local).as_dvec3();
    for t in trailers {
        let w = behind(t, local.y);
        if w <= 0.0 {
            break;
        }
        let there = t.pos + t.rot.transform_point3(local - t.offset).as_dvec3();
        here = here.lerp(there, w as f64);
        if w < 1.0 {
            break;
        }
    }
    here
}

/// The heading of the floor at a point of a cabin (see [`train_point`]).
fn train_heading(heading: f64, trailers: &[PartFrame], local: Vec3) -> f64 {
    let mut here = heading;
    for t in trailers {
        let w = behind(t, local.y);
        if w <= 0.0 {
            break;
        }
        here += crowd::angle_diff(here, t.heading) * w as f64;
        if w < 1.0 {
            break;
        }
    }
    here
}

impl BusNow {
    fn world(&self, local: Vec3) -> DVec3 {
        train_point(self.pos, &self.rot, &self.trailers, local)
    }
    /// A world point in the cabin's frame (the inverse of `world`): the front section's,
    /// or a rear section's for a point behind its joint.
    fn to_local(&self, w: DVec3) -> Vec3 {
        let mut l = self.rot.inverse().transform_point3((w - self.pos).as_vec3());
        for t in &self.trailers {
            if l.y > t.joint_y {
                break;
            }
            l = t.rot.inverse().transform_point3((w - t.pos).as_vec3()) + t.offset;
        }
        l
    }
    /// The tilt (pitch and bank, in the world's axes, no heading) of the section a point of
    /// the cabin is in.
    fn tilt_at(&self, local: Vec3) -> Mat4 {
        let mut rot = self.rot;
        let mut heading = self.heading;
        for t in &self.trailers {
            if behind(t, local.y) < 0.5 {
                break;
            }
            rot = t.rot;
            heading = t.heading;
        }
        rot * Mat4::from_rotation_z(heading.to_radians() as f32)
    }
    /// The heading of the section a point of the cabin is in.
    fn heading_at(&self, local: Vec3) -> f64 {
        train_heading(self.heading, &self.trailers, local)
    }
    fn fwd(&self) -> DVec2 {
        let h = self.heading.to_radians();
        DVec2::new(h.sin(), h.cos())
    }
    /// The bodies people on the ground walk round: the bus and its rear sections.
    fn blocks(&self) -> Vec<Block> {
        let block = |pos: DVec3, heading: f64, half: DVec2, centre: DVec2| {
            let h = heading.to_radians();
            let (fwd, right) = (DVec2::new(h.sin(), h.cos()), DVec2::new(h.cos(), -h.sin()));
            Block {
                center: pos.truncate() + right * centre.x + fwd * centre.y,
                half,
                heading: h,
                vel: fwd * self.speed,
            }
        };
        let mut out = vec![block(self.pos, self.heading, self.half, self.centre)];
        out.extend(
            self.trailers
                .iter()
                .map(|t| block(t.pos, t.heading, t.half, t.centre)),
        );
        out
    }
}

/// One piece of a walk along a pavement path: lane `lane` from distance `a` to `b`.
#[derive(Debug, Clone, Copy)]
struct Leg {
    lane: usize,
    a: f32,
    b: f32,
}

impl Leg {
    fn len(&self) -> f32 {
        (self.b - self.a).abs()
    }
    fn dist(&self, p: f32) -> f32 {
        if self.b >= self.a {
            self.a + p
        } else {
            self.a - p
        }
    }
    /// Point and walking heading `p` metres into the leg.
    fn at(&self, net: &Network, p: f32) -> (DVec3, f64) {
        let (q, h) = net.lanes[self.lane].at(self.dist(p.clamp(0.0, self.len())));
        (
            q,
            if self.b >= self.a {
                h as f64
            } else {
                h as f64 + 180.0
            },
        )
    }
    /// How far into the leg the point nearest `pos` lies, looking around `hint`.
    fn project(&self, net: &Network, pos: DVec3, hint: f32) -> f32 {
        let (lo, hi) = ((hint - 1.5).max(0.0), (hint + 3.0).min(self.len()));
        let mut best = (hint, f64::MAX);
        let mut p = lo;
        while p <= hi + 1e-3 {
            let d = (self.at(net, p).0 - pos).truncate().length_squared();
            if d < best.1 {
                best = (p, d);
            }
            p += 0.2;
        }
        best.0
    }
    /// Whether the leg starts at an end of its lane (at a kerb or a junction).
    fn from_end(&self, net: &Network) -> bool {
        self.a < 0.05 || self.a > net.lanes[self.lane].length() - 0.05
    }
}

/// A walk along the pavement network.
#[derive(Debug, Clone)]
struct PedWalk {
    legs: Vec<Leg>,
    leg: usize,
    /// Metres walked into the current leg.
    s: f32,
    /// A stroll: goes on at random when the legs run out.
    roam: bool,
    /// Keep-right offset (m).
    side: f32,
    /// Seconds spent waiting at the kerb before the current leg.
    held: f32,
    /// [ROLLBACK r4c-P1] The crossing being walked: its pieces among `legs`, from kerb to kerb.
    cross: Option<Cross>,
    /// [ROLLBACK r4c-P1] Back at the near kerb after turning round on a crossing put down on
    /// (no pavement leg to stand on): the next leg is decided as at a kerb.
    kerb_hold: bool,
    /// [ROLLBACK r4c-P3] Where they wait at this kerb: (leg, next lane, spot).
    spot: Option<(usize, usize, DVec2)>,
}

impl PedWalk {
    fn new(legs: Vec<Leg>, roam: bool, side: f32) -> PedWalk {
        PedWalk {
            legs,
            leg: 0,
            s: 0.0,
            roam,
            side,
            held: 0.0,
            cross: None,
            kerb_hold: false,
            spot: None,
        }
    }

    /// On to the next leg.
    fn step_on(&mut self) {
        let len = self.legs.get(self.leg).map(|l| l.len()).unwrap_or(0.0);
        self.s = (self.s - len).max(0.0);
        self.leg += 1;
        self.held = 0.0;
        self.spot = None;
        self.kerb_hold = false;
    }

    /// [ROLLBACK r4c-P1] Caught by the change before the middle of the crossing: back to the
    /// near kerb the way they came (then the crossing again, at the next green).
    fn turn_back(&mut self, net: &Network) {
        if let Some(c) = self.cross {
            self.turn_back_to(net, c.from);
        }
    }

    /// [ROLLBACK r5-P2] Back the way they came to the near end of piece `upto` of the crossing
    /// (its first piece: the near kerb; a later one: the refuge before it). "Back" is always
    /// the lane end behind that piece's start, never a point inside it: somebody put down
    /// mid-piece had the drop point in the road for a kerb and stood there (X10 case09, 26
    /// vehicles 66 s; case12).
    fn turn_back_to(&mut self, net: &Network, upto: usize) {
        let Some(c) = self.cross else { return };
        let k = self.leg;
        if !c.on(k) || k >= self.legs.len() || upto < c.from || upto > k {
            return;
        }
        let cur = self.legs[k];
        let first = self.legs[upto];
        let node = if first.b >= first.a { 0.0 } else { net.lanes[first.lane].length() };
        let mut back = vec![Leg { lane: cur.lane, a: cur.dist(self.s.min(cur.len())), b: if k == upto { node } else { cur.a } }];
        for j in (upto..k).rev() {
            let l = self.legs[j];
            back.push(Leg { lane: l.lane, a: l.b, b: if j == upto { node } else { l.a } });
        }
        // (the pavement leg they stepped off, entered at its end; the crossing again from that
        // lane end, and on)
        let kerb = if upto == c.from { c.kerb } else { None };
        let mut rest: Vec<Leg> = kerb.into_iter().collect();
        rest.push(Leg { lane: first.lane, a: node, b: first.b });
        rest.extend_from_slice(&self.legs[upto + 1..]);
        let n = back.len();
        let total = back.iter().map(|l| l.len()).sum();
        self.legs.truncate(k);
        self.legs.extend(back);
        self.legs.extend(rest);
        self.s = 0.0;
        self.held = 0.0;
        self.spot = None;
        self.cross = Some(Cross { from: k, to: k + n - 1, total, lead: 0.0, back: true, caught: true, kerb, ..c });
    }
}

/// [ROLLBACK r4c-P1] A crossing: the footpath pieces from one kerb to the other (the lit stub,
/// the pieces over each lane, a median between them), walked as one: the light applies at
/// the kerb only, and nobody stops on the carriageway.
#[derive(Debug, Clone, Copy)]
struct Cross {
    /// Indices into the walk's `legs` of the first piece and the last.
    from: usize,
    to: usize,
    /// Length of all the pieces (m).
    total: f32,
    /// [ROLLBACK r5-P2] Metres from the lane end behind the first piece's start to that
    /// start: zero from a kerb, the drop distance for somebody put down mid-piece (so that
    /// the middle and the way back are measured from the kerb, not the drop point).
    lead: f32,
    /// Kerb to kerb (m): the pieces and the steps between them (pieces of 1 m each join the
    /// next a metre on: a walk leaves every other one out).
    span: f32,
    /// The light that governs it (a piece's own, or that of the crossing object's crossing).
    light: Option<(usize, usize)>,
    /// Walking back to the near kerb (the pieces `from..=to` are the way back).
    back: bool,
    /// The light changed while they were on the carriageway.
    caught: bool,
    /// The pavement leg that led to the near kerb.
    kerb: Option<Leg>,
    /// The crossing's key: its lowest lane (the same both ways).
    key: usize,
}

impl Cross {
    fn on(&self, leg: usize) -> bool {
        leg >= self.from && leg <= self.to
    }
    /// Metres walked from the kerb.
    fn walked(&self, walk: &PedWalk) -> f32 {
        let done: f32 = walk.legs[self.from.min(walk.legs.len())..walk.leg.min(walk.legs.len())].iter().map(|l| l.len()).sum();
        done + walk.legs.get(walk.leg).map(|l| walk.s.min(l.len())).unwrap_or(0.0)
    }
}

/// [ROLLBACK r5-P2] Where somebody caught by the change clears the carriageway to: back to the
/// near end of piece `i` of the crossing (0: the near kerb), or on to the far end of piece
/// `i` (the last: the far kerb).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Clear {
    Back(usize),
    On(usize),
}

/// [ROLLBACK r5-P2] A piece of a crossing at least this long (m) lying in no carriageway lane's
/// footprint is a refuge to clear to (a median); a 1 m painted island is not: a body (0.28 m)
/// on it stands within KERB_CLEAR of both lanes' edges.
const PED_REFUGE_MIN: f32 = 2.0;

/// [ROLLBACK r5-P2] Whether the piece `leg` is a refuge: PED_REFUGE_MIN long at least, and off
/// every carriageway lane's footprint along its length (the half metre at each end, which
/// lies at the kerbs, excepted).
fn refuge(net: &Network, leg: &Leg) -> bool {
    let len = leg.len();
    if len < PED_REFUGE_MIN {
        return false;
    }
    let mut s = 0.5f32;
    while s <= len - 0.5 + 1e-3 {
        if on_carriageway(net, leg.at(net, s).0) {
            return false;
        }
        s += 0.5;
    }
    true
}

/// [ROLLBACK r5-P3] Seconds of the clearance somebody going on across at the change leaves
/// unused: the two seconds `pedestrian_window` allows where no car light is programmed, a
/// step onto the far kerb.
const PED_CLEAR_MARGIN: f32 = 2.0;

/// [ROLLBACK r5-P3] Seconds until a vehicle may drive over the footpath lanes `paths` by the
/// lights of the carriageway lanes they cross and of the lanes leading into those (the
/// soonest; 0 with one showing green now); None where none of them carries a light - the
/// program gives no clearance.
fn cars_go_in(ped: &mut PedNet, net: &Network, traffic: &dyn PedLights, paths: &[usize]) -> Option<f32> {
    let crossed: Vec<usize> = paths.iter().flat_map(|&p| ped.crossed_lanes(net, p).to_vec()).collect();
    let mut soonest: Option<f32> = None;
    for &s in &crossed {
        let feeding = net.prev.get(s).map(|p| p.as_slice()).unwrap_or(&[]);
        for &l in std::iter::once(&s).chain(feeding) {
            let Some((c, li)) = net.lanes[l].traffic_light else {
                continue;
            };
            if let Some(g) = traffic.light_until_go(c, li) {
                soonest = Some(soonest.map(|v: f32| v.min(g)).unwrap_or(g));
            }
        }
    }
    soonest
}

/// [ROLLBACK r4c-P1] Caught by the change at `walked` metres (from the first piece's start,
/// `lead` m from the kerb behind it) of a crossing of `pieces` (length, whether a refuge):
/// [ROLLBACK r5-P3] on across when the far kerb is reached in the clearance (seconds until
/// the cars go, seconds the rest takes at the hurried pace) less PED_CLEAR_MARGIN; else, or
/// with no clearance programmed, to the nearest safe place of the near kerb, a refuge of
/// the crossing and the far kerb (the far kerb on a tie: the old rule, back before the middle).
fn change_turns_back(walked: f32, lead: f32, pieces: &[(f32, bool)], clearance: Option<(f32, f32)>) -> Clear {
    let last = pieces.len().saturating_sub(1);
    let total: f32 = pieces.iter().map(|p| p.0).sum();
    if let Some((cars_in, rest_takes)) = clearance {
        if rest_takes <= cars_in - PED_CLEAR_MARGIN {
            return Clear::On(last);
        }
    }
    let mut options = vec![(total - walked, Clear::On(last)), (walked + lead, Clear::Back(0))];
    let mut r0 = 0.0f32;
    for (j, &(len, is_refuge)) in pieces.iter().enumerate() {
        let r1 = r0 + len;
        if is_refuge && j < last {
            options.push(if walked < r0 {
                (r0 - walked, Clear::On(j))
            } else if walked > r1 {
                (walked - r1, Clear::Back(j + 1))
            } else {
                (0.0, Clear::On(j))
            });
        }
        r0 = r1;
    }
    let mut best = options[0];
    for o in &options[1..] {
        if o.0 < best.0 {
            best = *o;
        }
    }
    best.1
}

/// What the walk asks the traffic lights (the traffic, or a test's).
trait PedLights {
    /// State of light `li` of controller `c` and the seconds it keeps showing it.
    fn light_state(&self, c: usize, li: usize) -> Option<(i32, f32)>;
    /// Seconds until the light lets traffic go (0 while it does).
    fn light_until_go(&self, c: usize, li: usize) -> Option<f32>;
    /// The light governing footpath lane `lane` (its own, or its crossing's).
    fn crossing_light(&self, lane: usize) -> Option<(usize, usize)>;
    /// Whether footpath lane `lane` is a piece of a crossing object's pedestrian crossing.
    fn crossing_piece(&self, lane: usize) -> bool;
}

impl PedLights for Traffic {
    fn light_state(&self, c: usize, li: usize) -> Option<(i32, f32)> {
        Traffic::light_state(self, c, li)
    }
    fn light_until_go(&self, c: usize, li: usize) -> Option<f32> {
        Traffic::light_until_go(self, c, li)
    }
    fn crossing_light(&self, lane: usize) -> Option<(usize, usize)> {
        Traffic::crossing_light(self, lane)
    }
    fn crossing_piece(&self, lane: usize) -> bool {
        self.sim.junctions.crossing_of(lane).is_some()
    }
}

/// The id the player's vehicle goes by among the `SeenCar`s (its rear sections count down).
const PLAYER_CAR: u64 = u64::MAX - 16;

/// A vehicle as the people on foot see it.
#[derive(Debug, Clone, Copy)]
struct SeenCar {
    id: u64,
    pos: DVec2,
    vel: DVec2,
    half_len: f64,
    /// Its box on the ground.
    body: Block,
    /// Stopping for somebody on foot (a car that yields is no threat to the one it yields to).
    yielding: bool,
    /// Seconds it has stood (under 0.5 m/s).
    stood: f32,
}

/// What the walk needs of the person (copied in and out of `Person`).
#[derive(Debug, Clone)]
struct Me {
    id: u32,
    pos: DVec2,
    pace: f64,
    on_crossing: bool,
    decision: Option<&'static str>,
    why: &'static str,
    /// The pace walked this frame.
    walk_pace: f64,
}

/// What a stroller sees this frame.
struct Around<'a> {
    net: &'a Network,
    lights: Option<&'a dyn PedLights>,
    cars: &'a [SeenCar],
    /// [ROLLBACK r4c-P4] The people on the carriageway of each crossing (by its key).
    streams: &'a Streams,
    time: f64,
}

fn xorshift(x: &mut u64) -> u64 {
    let mut v = *x;
    v ^= v >> 12;
    v ^= v << 25;
    v ^= v >> 27;
    *x = v;
    v.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

/// A piece of a crossing proper: it cuts a carriageway, carries a light, or is one of the
/// pieces of a crossing object's pedestrian crossing (BeijingCity lays its crossings in 1 m
/// pieces, most of which cut no lane).
fn core_piece(ped: &mut PedNet, net: &Network, lights: Option<&dyn PedLights>, lane: usize) -> bool {
    net.lanes[lane].traffic_light.is_some() || lights.map(|t| t.crossing_piece(lane)).unwrap_or(false) || !ped.crossings(net, lane).is_empty()
}

/// [ROLLBACK r5-P6] How strollers are put down in groups and spaced (/tmp/round5x/streams-report.md:
/// the spawner sprinkled one person at a time with no spacing law, so the gaps between
/// walkers were exponential, 21 % under a metre, and nobody walked in a pair or a knot).
/// The size of a group: round(exp(N(0, PED_GROUP_SIGMA))), 1..PED_GROUP_MAX (median 1, mean
/// about 1.4, a tail to six).
const PED_GROUP_SIGMA: f64 = 0.8;
const PED_GROUP_MAX: usize = 6;
/// The gap from one member of a group to the next along the pavement (m): log-normal with
/// this median and sigma (a pair walks a metre apart).
const PED_GROUP_GAP_MEDIAN: f64 = 1.0;
const PED_GROUP_GAP_SIGMA: f64 = 0.6;
/// The members' paces lie within this share of the leader's (they walk together).
const PED_GROUP_PACE_SPREAD: f64 = 0.05;
/// A candidate is refused while another stroller on its lane walking the same way is nearer
/// than a log-normal gap of sigma PED_GAP_SIGMA (m) whose median is `gap_median`: at most
/// PED_GAP_MEDIAN, at least PED_GAP_MIN - right-skewed gaps between groups, tight ones inside.
/// [ROLLBACK r5-P6b] (a fixed 6 m median accepted 1-4 % of the draws at the density the
/// `pedestrians` setting asks for and halved the strollers within 200 m: review r5p)
const PED_GAP_MEDIAN: f64 = 6.0;
const PED_GAP_MIN: f64 = 0.8;
const PED_GAP_SIGMA: f64 = 1.0;

/// [ROLLBACK r5-P6b] The median gap (m) for `target` strollers on `pavement_m` metres of pavement
/// in the circle: half the pavement per person wanted, within PED_GAP_MIN..PED_GAP_MEDIAN, so
/// that the count the setting asks for is reached while the gaps keep their skew. At the top
/// of the `pedestrians` slider the median sits on the 0.8 m floor and the law is off by
/// design: the slider is the density, and only the group draw shapes the gaps there.
fn gap_median(pavement_m: f32, target: usize) -> f64 {
    (pavement_m as f64 / (2.0 * target.max(1) as f64)).clamp(PED_GAP_MIN, PED_GAP_MEDIAN)
}

/// [ROLLBACK r5-P6] The size of a group from a standard normal deviate `z`.
fn group_size(z: f64) -> usize {
    ((PED_GROUP_SIGMA * z).exp().round() as usize).clamp(1, PED_GROUP_MAX)
}

/// [ROLLBACK r5-P6] A log-normal value of `median` and `sigma` from a standard normal deviate `z`.
fn lognormal(median: f64, sigma: f64, z: f64) -> f64 {
    median * (sigma * z).exp()
}

/// [ROLLBACK r5-P1] Whether a stroller may be put down on pavement lane `l`: a path of the
/// walking network longer than 4 m that unscheduled people may use, and no piece of a
/// crossing: nothing that cuts a carriageway or carries a light. Strollers put down on a
/// crossing piece walked off mid-road with no kerb decision, and once turned back by the
/// change their "kerb" was the drop point in the road (a bus and 26 cars held 66 s by one
/// man, X10 case09; BeijingCity case04; 22 drop-ins in 93 s at X10 j530, case12). A kerb
/// piece that cuts no lane stays a candidate (the pavements of junction objects).
fn stroll_candidate(ped: &mut PedNet, net: &Network, lights: Option<&dyn PedLights>, l: usize) -> bool {
    ped.ends.contains_key(&l)
        && net.lanes[l].length() > 4.0
        && !net.lanes[l].no_unsched // [ROLLBACK unsched-93]
        && !core_piece(ped, net, lights, l)
}

/// [ROLLBACK r4c-P1] The crossing starting with `walk.legs[start]`: the pieces on from it that
/// cut a carriageway or carry a light, with short pieces between them (a median); a stroll
/// is planned on straight across where its legs run out. None when no such piece follows.
fn crossing_span(ped: &mut PedNet, net: &Network, lights: Option<&dyn PedLights>, walk: &mut PedWalk, start: usize) -> Option<Cross> {
    const SHORT: f32 = 6.0;
    if start >= walk.legs.len() {
        return None;
    }
    let planned = walk.legs.len();
    let mut last_core: Option<usize> = None;
    // (the short pieces since the last piece proper, in metres: a median)
    let mut pending = 0.0f32;
    // [ROLLBACK r4c-L2] the light of the crossing so far: a piece with another (the second
    // street beyond an island, a median with its own light) is another crossing, decided at
    // its own kerb (merged, the second street was walked on its red)
    let piece_light = |l: usize| net.lanes[l].traffic_light.or_else(|| lights.and_then(|t| t.crossing_light(l)));
    let mut group: Option<(usize, usize)> = None;
    let mut k = start;
    while k < start + 64 {
        if k >= walk.legs.len() {
            if !walk.roam || k == 0 {
                break;
            }
            let prev = walk.legs[k - 1];
            let Some(n) = ped.straight_on(net, &prev, 60.0, |p, l| core_piece(p, net, lights, l) || net.lanes[l].length() <= SHORT) else { break };
            walk.legs.push(n);
        }
        let leg = walk.legs[k];
        if let Some(g) = piece_light(leg.lane) {
            if group.map(|h| h != g).unwrap_or(false) {
                break;
            }
            // a lit stub at a kerb after unlit crossing pieces (a corner piece, a slip lane
            // before the island) begins its own crossing, decided at the stub (Taijichang's
            // 18.2 stub after four unlit pieces, case09); a lit stub at the far kerb, onto
            // the pavement, ends the crossing it is on
            if group.is_none() && last_core.is_some() && net.lanes[leg.lane].traffic_light.is_some() {
                let next = net.lanes[leg.lane].next.clone();
                if next.into_iter().any(|n| core_piece(ped, net, lights, n)) {
                    break;
                }
            }
            group = Some(g);
        }
        if core_piece(ped, net, lights, leg.lane) {
            last_core = Some(k);
            pending = 0.0;
        } else if pending + leg.len() <= SHORT {
            pending += leg.len();
        } else {
            break;
        }
        k += 1;
    }
    // (what was planned past the far kerb only to look at it goes again)
    walk.legs.truncate(planned.max(last_core.map(|e| e + 1).unwrap_or(0)));
    let end = last_core?;
    let pieces = &walk.legs[start..=end];
    let light = group;
    let first = pieces[0];
    Some(Cross {
        from: start,
        to: end,
        total: pieces.iter().map(|l| l.len()).sum(),
        lead: if first.b >= first.a { first.a } else { net.lanes[first.lane].length() - first.a },
        span: pieces.iter().map(|l| l.len()).sum::<f32>()
            + pieces.windows(2).map(|w| (w[0].at(net, w[0].len()).0 - w[1].at(net, 0.0).0).truncate().length() as f32).sum::<f32>(),
        light,
        back: false,
        caught: false,
        kerb: None,
        key: pieces.iter().map(|l| l.lane).min().unwrap_or(0),
    })
}

/// Path ends closer than this (m) are one node of the walking network (the grid of
/// `PedNet::cells` is 1.5 m, so a search of the 3x3 cells around finds them all).
const PED_NODE_MERGE: f64 = 1.2;
/// The far end of a short lit piece joins only a node this close (m): a 1.005 m stub keeps
/// its two ends apart (the review of case09; the stock maps' stubs meet the next piece
/// end to start, within centimetres).
const LIT_FAR_MERGE: f64 = 0.3;

/// The pavement paths as a walking network: path ends closer than a metre are one
/// junction, whatever their heading (the road network joins lane ends only when they
/// continue in the same direction, which leaves every pavement corner open).
struct PedNet {
    /// Per pavement lane: its start and end junction.
    ends: HashMap<usize, (usize, usize)>,
    /// Per junction: (lane, walked forwards) leaving it.
    out: Vec<Vec<(usize, bool)>>,
    /// Where each pavement lane crosses a carriageway (lazily), and the carriageway lanes
    /// it crosses.
    crossings: HashMap<usize, Vec<DVec2>>,
    crossed: HashMap<usize, Vec<usize>>,
    /// Pavement lanes by 50 m cell.
    grid: HashMap<(i32, i32), Vec<usize>>,
    /// The junctions and a 1.5 m grid of them, for joining the paths of tiles loaded later.
    nodes: Vec<DVec3>,
    cells: HashMap<(i64, i64), Vec<usize>>,
    /// How many lanes of the traffic network are in (the network only grows: tiles bring
    /// their lanes and the indices stay).
    built: usize,
}

impl PedNet {
    fn build(net: &Network) -> PedNet {
        let mut p = PedNet {
            ends: HashMap::new(),
            out: Vec::new(),
            crossings: HashMap::new(),
            crossed: HashMap::new(),
            grid: HashMap::new(),
            nodes: Vec::new(),
            cells: HashMap::new(),
            built: 0,
        };
        p.extend(net);
        log::info!(
            "pavement network: {} paths, {} junctions",
            p.ends.len(),
            p.nodes.len()
        );
        p
    }

    /// Take in the lanes the network gained since the last call (tiles streamed in).
    ///
    /// The lit pieces go first, each short one with a node of its own at its far end, so
    /// that a 1 m lit stub is walked and its light is met (dropped as a piece with both
    /// ends in one node, BeijingCity's crossings were walked from the kerb straight onto
    /// the piece beyond the stub: 44 of its 84 lit foot pieces are under 1.2 m, case09).
    fn extend(&mut self, net: &Network) -> usize {
        let from = self.built.min(net.lanes.len());
        let before = self.ends.len();
        let walkable = |l: &omsi_sim::traffic::Lane| l.kind == LaneKind::Sidewalk && l.points.len() >= 2 && l.length() >= 0.3;
        let lit: Vec<usize> = (from..net.lanes.len()).filter(|&i| walkable(&net.lanes[i]) && net.lanes[i].traffic_light.is_some()).collect();
        let mut nodes: HashMap<usize, (usize, usize)> = HashMap::new();
        for &i in &lit {
            let l = &net.lanes[i];
            let a = self.node_of(l.start(), PED_NODE_MERGE, None);
            let far = if (l.length() as f64) < 2.0 * PED_NODE_MERGE { LIT_FAR_MERGE } else { PED_NODE_MERGE };
            let b = self.node_of(l.end(), far, Some(a));
            nodes.insert(i, (a, b));
        }
        for i in from..net.lanes.len() {
            let l = &net.lanes[i];
            if l.kind == LaneKind::Street && from > 0 {
                // a new carriageway may cross pavement paths that are in already
                self.crossings.clear();
            }
            if !walkable(l) {
                continue;
            }
            let (a, b) = match nodes.get(&i) {
                Some(&ab) => ab,
                None => (self.node_of(l.start(), PED_NODE_MERGE, None), self.node_of(l.end(), PED_NODE_MERGE, None)),
            };
            if a == b && l.length() < 3.0 && l.traffic_light.is_none() {
                continue;
            }
            self.ends.insert(i, (a, b));
            // [ROLLBACK pedoneway-1] a one-way pavement path with a light, or one that
            // crosses a carriageway, is walked its way only: the lit stub at the kerb is then
            // the only way onto its crossing, and nobody steps onto the unlit pieces from
            // the far kerb or from the side and crosses on red (every path was walkable
            // both ways; the ordinary pavements still are)
            let keeps_way = l.one_way && (l.traffic_light.is_some() || !self.crossings(net, i).is_empty());
            self.out[a].push((i, true));
            if !keeps_way {
                self.out[b].push((i, false));
            }
            let mut seen: Vec<(i32, i32)> = Vec::new();
            for p in &l.points {
                let c = ((p.x / 50.0).floor() as i32, (p.y / 50.0).floor() as i32);
                if !seen.contains(&c) {
                    seen.push(c);
                    self.grid.entry(c).or_default().push(i);
                }
            }
        }
        self.built = net.lanes.len();
        self.ends.len() - before
    }

    /// The junction at `p`: the nearest within `radius` (but `not`), a new one when there
    /// is none. (The first found within reach, not the nearest, joined the piece beyond a
    /// 1 m lit stub to the stub's kerb end.)
    fn node_of(&mut self, p: DVec3, radius: f64, not: Option<usize>) -> usize {
        let (cx, cy) = ((p.x / 1.5).floor() as i64, (p.y / 1.5).floor() as i64);
        let mut best: Option<(f64, usize)> = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(list) = self.cells.get(&(cx + dx, cy + dy)) {
                    for &n in list {
                        let d = (self.nodes[n] - p).truncate().length();
                        if d < radius && (self.nodes[n].z - p.z).abs() < 2.5 && Some(n) != not && best.is_none_or(|(bd, _)| d < bd) {
                            best = Some((d, n));
                        }
                    }
                }
            }
        }
        if let Some((_, n)) = best {
            return n;
        }
        self.nodes.push(p);
        self.out.push(Vec::new());
        self.cells
            .entry((cx, cy))
            .or_default()
            .push(self.nodes.len() - 1);
        self.nodes.len() - 1
    }

    /// The pavement lane nearest `p` within `reach` that can be reached without going over
    /// a carriageway: (lane, distance along it, distance to it). The plain nearest one was
    /// often the pavement across the road - a passenger off a bus then walked straight
    /// over the carriageway through the traffic to it, or joined a crossing in the middle.
    fn nearest(&self, net: &Network, p: DVec3, reach: f64) -> Option<(usize, f32, f64)> {
        let (cx, cy) = ((p.x / 50.0).floor() as i32, (p.y / 50.0).floor() as i32);
        let mut cands: Vec<(usize, f32, f64)> = Vec::new();
        let mut seen = HashSet::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                for &i in self
                    .grid
                    .get(&(cx + dx, cy + dy))
                    .map(|v| v.as_slice())
                    .unwrap_or(&[])
                {
                    if !seen.insert(i) {
                        continue;
                    }
                    if let Some((s, d)) = net.lanes[i].nearest_point(p) {
                        if d < reach {
                            cands.push((i, s, d));
                        }
                    }
                }
            }
        }
        cands.sort_by(|a, b| a.2.total_cmp(&b.2));
        // [ROLLBACK r5-P6b] never a piece of a crossing: one that carries a light or cuts a
        // street (somebody off a bus beside a zebra got it for their first leg and walked
        // off mid-piece with no kerb decision: the j530 appearances of review r5p)
        let cuts_street = |i: usize| {
            let pts = &net.lanes[i].points;
            pts.windows(2).any(|w| !street_cuts(net, w[0], w[1]).is_empty())
        };
        cands.retain(|&(i, _, _)| net.lanes[i].traffic_light.is_none() && !cuts_street(i));
        let first = cands.first().copied();
        cands
            .into_iter()
            .find(|&(i, s, _)| {
                let (q, _) = net.lanes[i].at(s);
                !crosses_street(net, p.truncate(), q.truncate())
            })
            // on an island between carriageways: the nearest after all
            .or(first)
    }

    /// The junction a leg ends at, when it ends at one.
    fn end_node(&self, net: &Network, leg: &Leg) -> Option<usize> {
        let (a, b) = *self.ends.get(&leg.lane)?;
        let len = net.lanes[leg.lane].length();
        if leg.b < 0.05 {
            Some(a)
        } else if leg.b > len - 0.05 {
            Some(b)
        } else {
            None
        }
    }

    /// A leg leaving junction `node`, not back along `came` (unless it is a dead end).
    fn next_leg(&self, net: &Network, node: usize, came: usize, pick: u64) -> Option<Leg> {
        let back = self.ends.get(&came).copied();
        let twin = |l: usize| -> bool {
            l == came
                || matches!((self.ends.get(&l), back), (Some(&(a, b)), Some((c, d))) if a == d && b == c && (net.lanes[l].length() - net.lanes[came].length()).abs() < 1.0)
        };
        let list: Vec<(usize, bool)> = self
            .out
            .get(node)?
            .iter()
            .copied()
            .filter(|(l, _)| !twin(*l))
            .collect();
        // the way on rather than back: a path leaving the junction within 110° of the way
        // the walker came (there usually is one - a pavement goes on past a side street),
        // else any. Picked from all, a stroller would turn round at every corner and walk
        // back the way they came, which looked like a change of mind for no reason.
        let heading_in = {
            let l = &net.lanes[came];
            let (a, _) = back.unwrap_or((usize::MAX, usize::MAX));
            // arriving at `node` along `came`: forwards if its end is the node
            if a == node {
                wrap_heading(l.start_heading() as f64 + 180.0)
            } else {
                l.end_heading() as f64
            }
        };
        let leaving = |&(l, fwd): &(usize, bool)| -> f64 {
            let lane = &net.lanes[l];
            if fwd {
                lane.start_heading() as f64
            } else {
                wrap_heading(lane.end_heading() as f64 + 180.0)
            }
        };
        let onward: Vec<(usize, bool)> = list
            .iter()
            .copied()
            .filter(|o| angle_between(heading_in, leaving(o)) <= 110.0)
            .collect();
        let list = if onward.is_empty() { list } else { onward };
        let (lane, fwd) = if list.is_empty() {
            // a dead end: turn round
            let (a, _) = back?;
            (came, a == node)
        } else {
            list[(pick as usize) % list.len()]
        };
        let len = net.lanes[lane].length();
        Some(if fwd {
            Leg {
                lane,
                a: 0.0,
                b: len,
            }
        } else {
            Leg {
                lane,
                a: len,
                b: 0.0,
            }
        })
    }


    /// [ROLLBACK r4c-P1] The leg on from the end of `leg` that turns least (within `max_turn`
    /// degrees) among the lanes `ok` accepts; not back along `leg`'s lane or its twin.
    fn straight_on(&mut self, net: &Network, leg: &Leg, max_turn: f64, mut ok: impl FnMut(&mut PedNet, usize) -> bool) -> Option<Leg> {
        let node = self.end_node(net, leg)?;
        let (_, h_in) = leg.at(net, leg.len());
        let back = self.ends.get(&leg.lane).copied();
        let list: Vec<(usize, bool)> = self.out.get(node)?.clone();
        let mut best: Option<(f64, Leg)> = None;
        for (l, fwd) in list {
            if l == leg.lane {
                continue;
            }
            if let (Some(&(a, b)), Some((c, d))) = (self.ends.get(&l), back) {
                if a == d && b == c && (net.lanes[l].length() - net.lanes[leg.lane].length()).abs() < 1.0 {
                    continue;
                }
            }
            if !ok(self, l) {
                continue;
            }
            let len = net.lanes[l].length();
            let cand = if fwd { Leg { lane: l, a: 0.0, b: len } } else { Leg { lane: l, a: len, b: 0.0 } };
            let (_, h) = cand.at(net, 0.0);
            let turn = angle_between(h_in, h);
            if turn <= max_turn && best.map(|b| turn < b.0).unwrap_or(true) {
                best = Some((turn, cand));
            }
        }
        best.map(|b| b.1)
    }

    /// Where pavement lane `lane` crosses a carriageway.
    fn crossings(&mut self, net: &Network, lane: usize) -> &[DVec2] {
        if !self.crossings.contains_key(&lane) {
            let l = &net.lanes[lane];
            let mut cand: Vec<usize> = Vec::new();
            for p in &l.points {
                if let Some(list) = net
                    .grid
                    .get(&((p.x / 50.0).floor() as i32, (p.y / 50.0).floor() as i32))
                {
                    for &i in list {
                        if net.lanes[i].kind == LaneKind::Street && !cand.contains(&i) {
                            cand.push(i);
                        }
                    }
                }
            }
            let mut out = Vec::new();
            let mut crossed = Vec::new();
            for i in cand {
                let o = &net.lanes[i];
                for w in l.points.windows(2) {
                    for v in o.points.windows(2) {
                        if (w[0].z - v[0].z).abs() > 3.0 {
                            continue;
                        }
                        if let Some(x) = seg_cross(
                            w[0].truncate(),
                            w[1].truncate(),
                            v[0].truncate(),
                            v[1].truncate(),
                        ) {
                            if !crossed.contains(&i) {
                                crossed.push(i);
                            }
                            if !out.iter().any(|q: &DVec2| (*q - x).length() < 1.5) {
                                out.push(x);
                            }
                        }
                    }
                }
            }
            self.crossings.insert(lane, out);
            self.crossed.insert(lane, crossed);
        }
        &self.crossings[&lane]
    }

    /// The carriageway lanes a pavement lane crosses.
    fn crossed_lanes(&mut self, net: &Network, lane: usize) -> &[usize] {
        self.crossings(net, lane);
        &self.crossed[&lane]
    }
}

/// Seconds a pedestrian starting across `path` now has before a vehicle may drive over it:
/// until the first light of a carriageway lane it crosses (or of a lane leading into one)
/// turns green once the pedestrian green is over. Lanes that have green now, or get it
/// while the pedestrians still have theirs, are turning traffic that gives way. Without
/// such a light, the pedestrian green `green_left` and two seconds.
fn pedestrian_window(
    mut ped: Option<&mut PedNet>,
    net: &Network,
    traffic: &dyn PedLights,
    paths: &[usize],
    green_left: f32,
) -> f32 {
    let mut window = f32::MAX;
    let crossed: Vec<usize> = match ped.as_deref_mut() {
        Some(ped) => paths.iter().flat_map(|&p| ped.crossed_lanes(net, p).to_vec()).collect(),
        None => Vec::new(),
    };
    {
        for &s in &crossed {
            let feeding = net.prev.get(s).map(|p| p.as_slice()).unwrap_or(&[]);
            for &l in std::iter::once(&s).chain(feeding) {
                let Some((c, li)) = net.lanes[l].traffic_light else {
                    continue;
                };
                if let Some(g) = traffic
                    .light_until_go(c, li)
                    .filter(|g| *g > 0.0 && *g >= green_left)
                {
                    window = window.min(g);
                }
            }
        }
    }
    if window == f32::MAX {
        green_left + 2.0
    } else {
        window
    }
}

/// One frame of a walk along the pavement network: where `me` goes, and what they decide at
/// the kerbs.
#[allow(clippy::too_many_arguments)]
fn stroll(mut ped: Option<&mut PedNet>, at: &Around, me: &mut Me, walk: &mut PedWalk, rng: &mut u64, yields: &mut Yields, dt: f32) -> Want {
    let net = at.net;
    let pos2 = me.pos;
    me.walk_pace = me.pace;
    if walk.leg >= walk.legs.len() {
        me.on_crossing = false;
        return Want::stand(None, Activity::Stand);
    }
    // [ROLLBACK r4c-P1] the light changes while they are on the carriageway: back before the
    // middle of the crossing, on past it (nobody stops in the road)
    if let (Some(c), Some(lights)) = (walk.cross, at.lights) {
        if c.on(walk.leg) && !c.back && !c.caught {
            if let Some((state, _)) = c.light.and_then(|(ctl, li)| lights.light_state(ctl, li)) {
                if !omsi_sim::traffic::TrafficLightController::allows_go(state) {
                    // [ROLLBACK r5-P2] to the nearest safe place: the near kerb, a median that
                    // lies in no lane, the far kerb - [ROLLBACK r5-P3] or on across, hurrying,
                    // when the clearance the controller gives allows it (at a 5.6 s green with a
                    // 12 s clearance everybody who had started on green was turned back, and ten
                    // stood on the zebra with "go" through the cars' green: X10 case13)
                    let pieces: Vec<(f32, bool)> = walk.legs[c.from..=c.to].iter().map(|l| (l.len(), refuge(net, l))).collect();
                    let paths: Vec<usize> = walk.legs[c.from..=c.to].iter().map(|l| l.lane).collect();
                    let walked = c.walked(walk);
                    let rest = (c.total - walked).max(0.0) * if c.total > 0.1 { c.span / c.total } else { 1.0 };
                    let clearance = ped.as_deref_mut().and_then(|p| cars_go_in(p, net, lights, &paths)).map(|g| (g, rest / hurried(me.id, me.pace) as f32));
                    match change_turns_back(walked, c.lead, &pieces, clearance) {
                        Clear::Back(i) => {
                            walk.turn_back_to(net, c.from + i);
                            me.decision = Some(TURNS_BACK);
                        }
                        Clear::On(i) => {
                            if let Some(c) = walk.cross.as_mut() {
                                c.caught = true;
                                c.to = c.from + i;
                            }
                            me.decision = Some(HURRIES_ON);
                        }
                    }
                }
            }
        }
    }
    let leg = walk.legs[walk.leg];
    if walk.s >= leg.len() - 0.35 || walk.held > 0.0 {
        match walk.cross {
            // between two pieces of the crossing: on, no decision on the carriageway
            Some(c) if walk.leg >= c.from && walk.leg < c.to => walk.step_on(),
            _ => {
                // the crossing walked: at a kerb (turned back, they stand on the pavement at
                // the near kerb again)
                if let Some(c) = walk.cross.take() {
                    if c.back && c.kerb.is_some() && walk.leg + 1 < walk.legs.len() {
                        walk.leg += 1;
                        walk.s = walk.legs[walk.leg].len();
                        walk.held = 0.0;
                        walk.spot = None;
                        me.on_crossing = false;
                        return Want::stand(None, Activity::Stand);
                    }
                    walk.kerb_hold = true;
                }
                if walk.leg + 1 >= walk.legs.len() {
                    if !walk.roam {
                        walk.leg += 1;
                        me.on_crossing = false;
                        return Want::stand(None, Activity::Stand);
                    }
                    let pick = xorshift(rng);
                    let next = ped
                        .as_ref()
                        .and_then(|p| p.end_node(net, &leg).and_then(|n| p.next_leg(net, n, leg.lane, pick)))
                        .unwrap_or_else(|| {
                            // a leg that ends in the middle of its path (the point of a stop
                            // somebody got off at) goes on to one of its ends: turned round
                            // there, the people off a bus were sent back to the same point
                            // every frame and milled round each other at the stop (#913)
                            let lane_len = net.lanes[leg.lane].length();
                            if leg.b > 0.05 && leg.b < lane_len - 0.05 {
                                let fwd = if leg.len() > 0.05 { leg.b > leg.a } else { pick % 2 == 0 };
                                Leg { lane: leg.lane, a: leg.b, b: if fwd { lane_len } else { 0.0 } }
                            } else {
                                Leg { lane: leg.lane, a: leg.b, b: leg.a }
                            }
                        });
                    walk.legs.push(next);
                    if walk.leg > 6 {
                        walk.legs.drain(..walk.leg);
                        walk.leg = 0;
                        walk.spot = None;
                    }
                }
                let next = walk.legs[walk.leg + 1];
                // (on a crossing piece, not at a kerb: at the end of a crossing's last piece
                // every crosser is here - `walk.cross` was just taken - and the span from this
                // piece steps them off it without a kerb decision; once also somebody put
                // down on a crossing, which no longer happens since r5-P1/P6b)
                let mid = !walk.kerb_hold && ped.as_deref_mut().map(|p| core_piece(p, net, at.lights, leg.lane)).unwrap_or(false);
                // [ROLLBACK r5-P2] (back at the near end after turning round on a crossing with
                // no pavement leg to stand on: decided as at a kerb - the light and the cars
                // looked at - not "go" without a verdict, which looped on red: case04/09/12)
                let at_kerb = (next.from_end(net) || walk.kerb_hold) && !mid;
                let span = match ped.as_deref_mut() {
                    Some(p) => crossing_span(p, net, at.lights, walk, if mid { walk.leg } else { walk.leg + 1 }),
                    None => None,
                };
                match span {
                    Some(c) if at_kerb => {
                        let verdict = match ped.as_deref_mut() {
                            Some(p) => kerb_verdict(p, at, walk, &c, me, yields),
                            None => Ok(()),
                        };
                        match verdict {
                            Ok(()) => {
                                me.decision = Some("go");
                                if walk.held > 0.0 && debug_pax() {
                                    log::info!("t={:.1} pax #{} crosses path {} after waiting {:.0} s", at.time, me.id, next.lane, walk.held);
                                }
                                let from_pavement = !ped.as_deref_mut().map(|p| core_piece(p, net, at.lights, leg.lane)).unwrap_or(false);
                                walk.cross = Some(Cross { kerb: from_pavement.then_some(leg), ..c });
                                walk.step_on();
                            }
                            Err(why) => {
                                walk.held += dt;
                                me.why = why;
                                me.decision = Some(why);
                                // a light that stays red (nobody crosses on red any more): a
                                // stroller gives up after three minutes and walks back the way
                                // they came
                                if walk.roam && why == "red light" && walk.held > 180.0 {
                                    if debug_pax() {
                                        log::info!("t={:.1} pax #{} gives up waiting at the red light and turns back", at.time, me.id);
                                    }
                                    walk.legs.truncate(walk.leg + 1);
                                    walk.legs.push(Leg { lane: leg.lane, a: leg.b, b: leg.a });
                                    walk.held = 0.0;
                                    walk.spot = None;
                                    walk.kerb_hold = false;
                                    return Want::stand(None, Activity::Stand);
                                }
                                me.on_crossing = false;
                                return wait_at_kerb(ped.as_deref_mut(), at, me, walk, &leg, &next);
                            }
                        }
                    }
                    Some(c) => {
                        me.decision = Some("go");
                        walk.cross = Some(c);
                        walk.step_on();
                    }
                    None => {
                        me.decision = Some("go");
                        walk.step_on();
                    }
                }
            }
        }
    }
    let leg = walk.legs[walk.leg];
    let len = leg.len();
    let (p, h) = leg.at(net, (walk.s + 1.3).min(len));
    let lane = &net.lanes[leg.lane];
    let width = (lane.width as f64).max(1.0);
    let crosses = ped.as_ref().map(|p| p.crossings.get(&leg.lane).map(|x| !x.is_empty()).unwrap_or(false)).unwrap_or(false);
    // [ROLLBACK r4c-P1] on the crossing from the first steps off the kerb to the far kerb, on
    // every piece of it (the cars of the lanes ahead stop for them)
    me.on_crossing = match walk.cross {
        Some(c) if c.on(walk.leg) => walk.leg > c.from || walk.s > 0.3 || c.back,
        _ => false,
    };
    let crossing = lane.traffic_light.is_some() || crosses || me.on_crossing;
    // [ROLLBACK r4c-P6] faster across when caught by the change, or with a vehicle standing for
    // them close by, until off the carriageway
    if me.on_crossing && (walk.cross.map(|c| c.caught).unwrap_or(false) || yielded_to(at.cars, pos2)) {
        me.walk_pace = hurried(me.id, me.pace);
    }
    let pace = me.walk_pace;
    // keep to the right of the pavement (less so on a crossing) - the left where the
    // traffic drives on the left
    let side = if crossing { (walk.side.abs() as f64).min(0.3) } else { (walk.side.abs() as f64).min(width * 0.5 - 0.3).max(0.0) };
    let side = if net.left_hand { -side } else { side };
    let hr = h.to_radians();
    let right = DVec2::new(hr.cos(), -hr.sin());
    let target = p.truncate() + right * side;
    let vel = (target - pos2).normalize_or_zero() * pace;
    // stay on the path: the lane locally, as wide as it is
    let (a, _) = leg.at(net, (walk.s - 2.0).max(0.0));
    let (b, _) = leg.at(net, (walk.s + 2.5).min(len));
    let (m, _) = leg.at(net, (walk.s + 0.25).min(len));
    let bow = crowd::project_on_segment(m.truncate(), a.truncate(), b.truncate()).0.distance(m.truncate());
    let corridor = ((b - a).truncate().length() > 0.5).then(|| (a.truncate() + right * side, b.truncate() + right * side, (width * 0.5 - side).max(0.35) + bow));
    let corridor = path_corridor_band(pos2, corridor, STROLL_BAND);
    if me.why != "queueing behind somebody" {
        me.why = "";
    }
    Want { vel, face: None, give: 1.0, corridor, idle: Activity::Stand }
}

/// [ROLLBACK r4c-P2] Whether somebody at the far kerb `arrival` s after stepping on makes it:
/// past the middle of the crossing before the pedestrian green ends in `green_left` s (the
/// change turns the others back, P1) and across before the cars get their green (`window`
/// s): the light must not turn red before the arrival less a margin, the shorter of half
/// the crossing and the clearance time.
fn makes_it(green_left: f32, window: f32, arrival: f64) -> bool {
    let clearance = (window - green_left).max(0.0) as f64;
    let margin = (arrival * 0.5).min(clearance);
    green_left as f64 >= arrival - margin
}

/// [ROLLBACK r4c-P2] The share of people who step onto a lit crossing they cannot make before
/// the red (each person at each crossing, seeded: always the same ones).
const PED_LATE_CHANCE: f64 = 0.08;

/// A number in 0..1 fixed by `a` and `b`.
fn hash01(a: u64, b: u64) -> f64 {
    let mut x = a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_add(0x632B_E59B_D9B4_E019).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// [ROLLBACK r4c-P2] Whether person `id` steps onto crossing `key` late anyway.
fn late_stepper(id: u32, key: usize) -> bool {
    hash01(id as u64, key as u64) < PED_LATE_CHANCE
}

/// [ROLLBACK r4c-P4] Seconds a vehicle stands at an unlit crossing before the people arriving
/// let it go first (when a stream of them would never leave it a gap).
const PED_YIELD_AFTER: f32 = 8.0;
/// [ROLLBACK r4c-P4] Seconds the people arriving at the kerb wait for such a vehicle, at most.
const PED_YIELD_MAX: f32 = 6.0;
/// [ROLLBACK r4c-L4] All the yields to one standing vehicle lie within this many seconds of the
/// first (3 x PED_YIELD_MAX).
const PED_YIELD_CAP: f32 = 3.0 * PED_YIELD_MAX;
/// [ROLLBACK r5-P5] A stream of this many people per metre of the crossing is dense: the
/// arrivals let the standing vehicle go for twice PED_YIELD_MAX (six seconds could not empty
/// a 10 m piece with 26-37 people on it: X10 case01, the player's bus 20 s).
const PED_DENSE_STREAM: f32 = 1.0;
/// [ROLLBACK r5-P5] A vehicle's stood clock survives a creep: it resets only once the vehicle
/// has moved this far (m) from where it last stood, or ...
const PED_CREEP_DIST: f64 = 2.0;
/// ... exceeds this speed (m/s). A car rolling through a zebra at 1-1.5 m/s covers the 2 m
/// in under two seconds and earns no yield (PED_YIELD_AFTER is 8 s); a driver inching
/// forward five times in 20 s (X10 case01: the yield came at 17 s instead of 8) keeps it.
const PED_CREEP_SPEED: f64 = 1.5;

/// [ROLLBACK r5-P5] The stood clock of a vehicle at `pos` moving at `speed` after `dt` s, from
/// its previous reading and the place it last stood: (seconds stood, that place). Under 0.5
/// m/s it stands; a creep from there keeps the clock running until PED_CREEP_DIST or
/// PED_CREEP_SPEED, which reset it.
fn stood_clock(prev: Option<(f32, DVec2)>, pos: DVec2, speed: f64, dt: f32) -> (f32, DVec2) {
    let standing = speed < 0.5;
    match prev {
        Some((s, anchor)) if s > 0.0 && speed <= PED_CREEP_SPEED && (pos - anchor).length() <= PED_CREEP_DIST => (s + dt, if standing { pos } else { anchor }),
        _ => (if standing { dt } else { 0.0 }, pos),
    }
}
/// The people on the carriageway of each crossing (by its key): id, position, velocity.
type Streams = HashMap<usize, Vec<(u32, DVec2, DVec2)>>;
/// The vehicles yielded to: the start of the current window and of the first.
type Yields = HashMap<u64, (f64, f64)>;
/// [ROLLBACK r4c-P4] A stream: somebody else on the crossing within this distance (m).
const PED_STREAM_GAP: f64 = 3.0;
const YIELDS: &str = "yields to a standing vehicle";

/// [ROLLBACK r4c-P4] Whether a vehicle stands at the crossing whose carriageway cuts are `cuts`:
/// on it, or with its front just before it, in the lane.
fn stands_at(car: &SeenCar, cuts: &[DVec2]) -> bool {
    let h = car.body.heading;
    let fwd = DVec2::new(h.sin(), h.cos());
    cuts.iter().any(|x| {
        let rel = *x - car.body.center;
        let along = rel.dot(fwd);
        along > -car.body.half.y && along < car.body.half.y + 6.0 && rel.perp_dot(fwd).abs() < 2.5
    })
}

/// [ROLLBACK r5-P4] The median of people's free walking pace (m/s). Bohannon 1997 (Age and
/// Ageing 26(1):15-19, doi:10.1093/ageing/26.1.15) measured comfortable means of 1.27-1.46
/// m/s by age and sex; the MUTCD (2009, 4E.06) designs pedestrian clearances for 3.5 ft/s
/// = 1.07 m/s, a slow walker.
const PED_PACE_MEDIAN: f64 = 1.35;
/// [ROLLBACK r5-P4] The spread of the free pace in log space (sigma of ln pace): about 15 %
/// either way for two thirds of the people, the spread of Bohannon's cohorts.
const PED_PACE_SIGMA: f64 = 0.2;
/// [ROLLBACK r5-P4] The slowest and the fastest free pace (m/s): the MUTCD design walker a
/// quarter under, Bohannon's slowest "maximum" means (1.75) a little under.
const PED_PACE_MIN: f64 = 0.8;
const PED_PACE_MAX: f64 = 1.85;
/// [ROLLBACK r5-P4] The factor a person hurries by when caught by the change or yielded to:
/// 1.3..1.8 of their own pace, capped at PED_HURRY_MAX (Bohannon's maximum walking speeds
/// 1.75-2.53 m/s are 1.4-1.7 times the comfortable ones).
const PED_HURRY_MIN: f64 = 1.3;
const PED_HURRY_MAX_FACTOR: f64 = 1.8;
const PED_HURRY_MAX: f64 = 2.5;
/// [ROLLBACK r5-P4] The share of people who jog across when caught, at PED_JOG_MIN..PED_JOG_MAX
/// m/s (the top of Bohannon's maximum speeds and a little over: a jog, not a walk).
const PED_JOG_SHARE: f64 = 0.05;
const PED_JOG_MIN: f64 = 2.5;
const PED_JOG_MAX: f64 = 3.0;

/// [ROLLBACK r5-P4] A standard normal deviate fixed by `a` and `b` (Box-Muller on two hashes).
fn hash_normal(a: u64, b: u64) -> f64 {
    let u1 = hash01(a, b).max(1e-12);
    let u2 = hash01(a, b ^ 0x9E37_79B9);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// [ROLLBACK r4c-P6] A person's own walking pace (m/s), fixed by their id: [ROLLBACK r5-P4]
/// log-normal about PED_PACE_MEDIAN with PED_PACE_SIGMA, within PED_PACE_MIN..PED_PACE_MAX
/// (it was uniform 1.0..1.4, and nobody was as quick or as slow as people are).
fn pace_of(id: u32) -> f64 {
    (PED_PACE_MEDIAN * (PED_PACE_SIGMA * hash_normal(id as u64, 0x5EED_9ACE)).exp()).clamp(PED_PACE_MIN, PED_PACE_MAX)
}

/// [ROLLBACK r5-P4] Whether person `id` jogs across when caught by the change.
fn jogger(id: u32) -> bool {
    hash01(id as u64, 0x1066_D0D6) < PED_JOG_SHARE
}

/// [ROLLBACK r4c-P6] The pace across in a hurry: [ROLLBACK r5-P4] the person's own hurry
/// factor (PED_HURRY_MIN..PED_HURRY_MAX_FACTOR, by id) times their pace, at most
/// PED_HURRY_MAX; a jogger (PED_JOG_SHARE) at PED_JOG_MIN..PED_JOG_MAX. Never slower than the
/// own pace.
fn hurried(id: u32, pace: f64) -> f64 {
    let h = hash01(id as u64, 0x4A6B_5E5A);
    if jogger(id) {
        return (PED_JOG_MIN + (PED_JOG_MAX - PED_JOG_MIN) * h).max(pace);
    }
    let factor = PED_HURRY_MIN + (PED_HURRY_MAX_FACTOR - PED_HURRY_MIN) * h;
    (pace * factor).min(PED_HURRY_MAX).max(pace)
}

/// [ROLLBACK r4c-P6] Whether a vehicle stands for somebody at `pos` within 6 m (one stopping for
/// people).
fn yielded_to(cars: &[SeenCar], pos: DVec2) -> bool {
    cars.iter().any(|c| c.yielding && c.vel.length() < 0.5 && (c.body.closest(pos).0 - pos).length() < 6.0)
}

/// [ROLLBACK r4c-P6] Seconds somebody who has decided to go may stand still wanting to walk
/// before they are planned again.
const PED_GO_FROZEN: f32 = 2.0;

/// [ROLLBACK r4c-P6] Whether somebody is frozen with "go": standing `still` s while wanting to walk
/// at `wants` m/s.
fn replan_frozen(decision: Option<&str>, still: f32, wants: f64) -> bool {
    decision == Some("go") && still > PED_GO_FROZEN && wants > 0.2
}

const BACK_TO_KERB: &str = "a vehicle in the way: back to the kerb";
/// A vehicle's box over the way at the kerb ([ROLLBACK r4c-P5]) or over a cut of the
/// crossing near the kerb ([ROLLBACK r5-P6]).
const STANDS_ON: &str = "a vehicle stands on the crossing";
/// [ROLLBACK r5-P6b] Metres of the way across within which a vehicle standing on a cut holds
/// the kerb: a kerb stub and a two-lane carriageway (the far lane's cut lies 7.25 m from the
/// kerb of the test crossing) - the kerb holds for its first two lanes; a vehicle on a
/// farther lane is met mid-crossing and dealt with by P1/P5. A junction object's crossing
/// has cuts across the whole junction, and 1050 person-s were refused at one kerb for cars
/// standing on cuts 20 m away (BJ 0821, review r5p).
const PED_OCCUPIED_REACH: f64 = 8.0;

/// [ROLLBACK r5-P6b] Whether a vehicle stopping for `why` is one stopping for people on foot: for
/// a person in front or for the zebra (a car holding for the zebra with its nose over a cut
/// counted as "not yielding" and held the kerb 20 s while both waited: X10 0837, cars 68/129).
fn yields_to_people(why: omsi_traffic::Why) -> bool {
    matches!(why, omsi_traffic::Why::People | omsi_traffic::Why::Zebra)
}
/// [ROLLBACK r5-P6] Seconds of a vehicle's standing on the crossing after which the kerb no
/// longer waits for it (people step round it): a car abandoned on a zebra
/// (X10 case15, a 90 s queue) does not hold the kerb for good.
const PED_OCCUPIED_CAP: f32 = 20.0;

/// [ROLLBACK r4c-P1] Decisions on a crossing as the recording shows them.
const TURNS_BACK: &str = "caught by the change: back to the kerb";
const HURRIES_ON: &str = "caught by the change: on across";

/// Waiting at the kerb before `next`: facing the way across, spread along it and a step back,
/// [ROLLBACK r4c-P3] off the carriageway (see `kerb_spot`).
fn wait_at_kerb(mut ped: Option<&mut PedNet>, at: &Around, me: &Me, walk: &mut PedWalk, _leg: &Leg, next: &Leg) -> Want {
    let net = at.net;
    let (_, h) = next.at(net, 0.3);
    let id = me.id;
    let spread = ((id % 5) as f64 - 2.0) * 0.45;
    let back = 0.25 + (id % 3) as f64 * 0.45;
    let spot = match walk.spot {
        Some((l, n, q)) if l == walk.leg && n == next.lane => q,
        _ => {
            // the legs walked up to the kerb, the current one first, as long as they join
            let mut way: Vec<Leg> = Vec::new();
            for k in (0..=walk.leg.min(walk.legs.len().saturating_sub(1))).rev() {
                let l = walk.legs[k];
                // [ROLLBACK r4c-L2] (never back onto a crossing just walked: at an island between
                // two lights the spot was backed across the first street)
                if ped.as_deref_mut().map(|p| core_piece(p, net, at.lights, l.lane)).unwrap_or(false) {
                    break;
                }
                if let Some(after) = way.last() {
                    let joins = (l.at(net, l.len()).0 - after.at(net, 0.0).0).truncate().length() < 1.2;
                    let on_cross = walk.cross.map(|c| c.on(k)).unwrap_or(false) || net.lanes[l.lane].traffic_light.is_some();
                    if !joins || on_cross || way.iter().map(|w| w.len()).sum::<f32>() > 12.0 {
                        break;
                    }
                }
                way.push(l);
            }
            // [ROLLBACK r4c-L3] (and on behind the first of them along its lane, as far as it goes:
            // somebody put down on the kerb piece had walked 0.3 m of it)
            if let Some(l) = way.last_mut() {
                let lane_len = net.lanes[l.lane].length();
                l.a = if l.b >= l.a { 0.0 } else { lane_len };
            }
            let q = kerb_spot(ped.as_deref(), net, &way, next, spread, back);
            walk.spot = Some((walk.leg, next.lane, q));
            q
        }
    };
    Want { vel: arrive(me.pos, spot, me.pace * 0.6), face: Some(h), give: 0.5, corridor: None, idle: Activity::Stand }
}

/// [ROLLBACK r4c-P3] How far beyond half a carriageway lane's width from its axis somebody
/// waiting at the kerb keeps (m).
const KERB_CLEAR: f64 = 0.3;

/// [ROLLBACK r5-P7] The outermost of the crowd's spread spots at a kerb (m either side of the
/// way: `wait_at_kerb` spreads by (id % 5 - 2) x 0.45).
const KERB_SPREAD_MAX: f64 = 0.9;

/// [ROLLBACK r4c-P3] The width a carriageway lane is taken to have at least (m): the pieces
/// inside junctions say 1 m, and a bus is 2.5 m wide.
const LANE_MIN_WIDTH: f64 = 2.5;

/// [ROLLBACK r4c-P3] Whether `q` lies inside the footprint of a carriageway lane: within half
/// its width and `KERB_CLEAR` of its axis.
fn on_carriageway(net: &Network, q: DVec3) -> bool {
    within_lanes(net, q, &|l| (l.width as f64).max(LANE_MIN_WIDTH) * 0.5 + KERB_CLEAR)
}

/// Whether `q` lies too close to a carriageway lane for somebody to wait there: within
/// `waiter_reach` of its axis, the reach of the bus waiters ([round 6 B3]): a crossing's kerb
/// spot at 1.55 m held a 2.6 m coach, which lets people go only beyond 1.58 m.
fn in_waiter_reach(net: &Network, q: DVec3) -> bool {
    within_lanes(net, q, &waiter_reach)
}

/// Whether `q` lies within `reach(lane)` of the axis of a street lane near it.
fn within_lanes(net: &Network, q: DVec3, reach: &dyn Fn(&omsi_sim::traffic::Lane) -> f64) -> bool {
    let (cx, cy) = ((q.x / 50.0).floor() as i32, (q.y / 50.0).floor() as i32);
    let mut seen: Vec<usize> = Vec::new();
    for dx in -1..=1 {
        for dy in -1..=1 {
            for &i in net.grid.get(&(cx + dx, cy + dy)).map(|v| v.as_slice()).unwrap_or(&[]) {
                let l = &net.lanes[i];
                if l.kind != LaneKind::Street || seen.contains(&i) {
                    continue;
                }
                seen.push(i);
                if l.nearest_point(q).map(|(_, d)| d < reach(l)).unwrap_or(false) {
                    return true;
                }
            }
        }
    }
    false
}

/// [round 6 B3] Somebody waiting for a bus keeps clear of a street lane's axis by half the
/// lane's width, or half the widest random vehicle's when that is wider (m: BeijingCity's
/// Daba coaches are 2.52 m, a few trucks 2.55), ...
const WAITER_VEHICLE_WIDTH: f64 = 2.6;
/// ... plus a body's radius (`perception::BODY_RADIUS`) and `KERB_CLEAR`: a car-class coach
/// passing in the middle of the lane is then clear of them by 0.3 m and holds for nobody
/// there (review-r6-cases case02: nine waiters 1.46-1.55 m off the axis of bus lane 622,
/// width 1 m, held two 11.6 m coaches 148 s, in the 2 cm band between the hold at hw+0.3 and
/// the release at hw+0.28).
const WAITER_BODY: f64 = 0.28;
/// [round 6 B3] A waiting place is moved at most this far (m) to clear the lanes; one that
/// needs more (a platform between two lanes) stays where the map has it.
const WAITER_PUSH_MAX: f64 = 1.0;

/// [round 6 B3] How far from the axis of street lane `l` somebody waiting for a bus keeps.
fn waiter_reach(l: &omsi_sim::traffic::Lane) -> f64 {
    (l.width as f64).max(WAITER_VEHICLE_WIDTH) * 0.5 + WAITER_BODY + KERB_CLEAR
}

/// [round 6 B3] `q`, moved straight away from the axes of the street lanes it is closer to
/// than `waiter_reach` (at its height); `q` itself when nothing is that close, or when
/// clearing them would take it more than `WAITER_PUSH_MAX` or into another lane.
pub(crate) fn clear_of_lanes(net: &Network, q: DVec3) -> DVec3 {
    // (the lane's polyline point nearest to `p`, and how far: as `Lane::nearest_point`)
    let nearest = |l: &omsi_sim::traffic::Lane, p: DVec3| -> Option<(DVec3, f64)> {
        let mut best: Option<(DVec3, f64)> = None;
        for w in l.points.windows(2) {
            let ab = w[1] - w[0];
            let t = ((p - w[0]).truncate().dot(ab.truncate()) / ab.truncate().length_squared().max(1e-9)).clamp(0.0, 1.0);
            let c = w[0] + ab * t;
            let d = (c - p).truncate().length();
            if (c.z - p.z).abs() < 2.5 && best.is_none_or(|b| d < b.1) {
                best = Some((c, d));
            }
        }
        best
    };
    // (the street lane it is deepest in, by how much)
    let worst = |p: DVec3| -> Option<(DVec3, f64, f64)> {
        let (cx, cy) = ((p.x / 50.0).floor() as i32, (p.y / 50.0).floor() as i32);
        let mut out: Option<(DVec3, f64, f64)> = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for &i in net.grid.get(&(cx + dx, cy + dy)).map(|v| v.as_slice()).unwrap_or(&[]) {
                    let l = &net.lanes[i];
                    if l.kind != LaneKind::Street {
                        continue;
                    }
                    let reach = waiter_reach(l);
                    if let Some((c, d)) = nearest(l, p) {
                        if d < reach - 1e-3 && out.is_none_or(|o| reach - d > o.2 - o.1) {
                            out = Some((c, d, reach));
                        }
                    }
                }
            }
        }
        out
    };
    let mut p = q;
    for _ in 0..4 {
        let Some((c, _, reach)) = worst(p) else {
            return if (p - q).truncate().length() <= WAITER_PUSH_MAX { p } else { q };
        };
        let away = (p - c).truncate();
        if away.length() < 1e-3 {
            return q;
        }
        let to = c.truncate() + away.normalize() * (reach + 0.01);
        p = DVec3::new(to.x, to.y, p.z);
        if (p - q).truncate().length() > WAITER_PUSH_MAX {
            return q;
        }
    }
    if worst(p).is_none() { p } else { q }
}

/// [round 6 B3] The waiting places of `stop` (not the seats: a bench stays where it is)
/// and its point, cleared of the street lanes (`clear_of_lanes`): the places that moved.
fn clear_stop_of_lanes(stop: &mut PaxStop, net: &Network) -> Vec<usize> {
    let mut moved = Vec::new();
    for (k, sp) in stop.spots.iter_mut().enumerate() {
        if sp.height != 0.0 {
            continue;
        }
        let to = clear_of_lanes(net, sp.pos);
        if (to - sp.pos).length() > 1e-3 {
            sp.pos = to;
            moved.push(k);
        }
    }
    stop.kerb_shift = clear_of_lanes(net, stop.pos) - stop.pos;
    stop.cleared = true;
    moved
}

/// [ROLLBACK r4c-P3] Where somebody waits before stepping onto `next` at the end of `way` (the
/// legs walked up to the kerb, the last first): a step back from the kerb and spread along
/// it, outside every street lane's waiter reach (`in_waiter_reach`). A kerb inside one (a
/// short piece at the kerb that lies in the lane: people waited 0.4 m in front of a bumper)
/// is backed away from along the legs walked until the spot is outside.
fn kerb_spot(ped: Option<&PedNet>, net: &Network, way: &[Leg], next: &Leg, spread: f64, back: f64) -> DVec2 {
    let (start, _) = next.at(net, 0.0);
    let (_, h) = next.at(net, 0.3);
    let hr = h.to_radians();
    let (fwd, right) = (DVec2::new(hr.sin(), hr.cos()), DVec2::new(hr.cos(), -hr.sin()));
    let end = way.first().map(|l| l.at(net, l.len()).0).unwrap_or(start);
    let first = end + (right * spread - fwd * back).extend(0.0);
    if !in_waiter_reach(net, first) {
        return first.truncate();
    }
    // [ROLLBACK r5-P7] the other side of the way first - the pavement's, where the spread
    // put them into a junction piece (2-5 people at "red light" 2-4 m inside the west arm
    // of X10 j392 lost a car its green, case07; case01b's j3 kerbs)
    let mirrored = end + (-right * spread - fwd * back).extend(0.0);
    if !in_waiter_reach(net, mirrored) {
        return mirrored.truncate();
    }
    let mut last = end;
    let mut d = 0.25f32;
    for l in way {
        let len = l.len();
        while d <= len {
            let (q, _) = l.at(net, len - d);
            last = q;
            // (own spread, halved, mirrored; then the crowd's outermost spread spots, each
            // tested: a lane beside the way takes every spot on its side)
            for off in [spread * 0.5, 0.0, -spread * 0.5, -spread, -KERB_SPREAD_MAX, KERB_SPREAD_MAX] {
                let c = q + (right * off).extend(0.0);
                if !in_waiter_reach(net, c) {
                    return c.truncate();
                }
            }
            d += 0.25;
        }
        d -= len;
    }
    // [ROLLBACK r4c-L3] the whole way in a lane: the nearest point of a footpath outside them
    ped.and_then(|p| nearest_off_carriageway(p, net, end, 15.0)).unwrap_or(last.truncate())
}

/// [ROLLBACK r4c-L3] The point of a footpath within `reach` of `at` (and at its height) nearest to
/// it outside every street lane's waiter reach (`in_waiter_reach`).
fn nearest_off_carriageway(ped: &PedNet, net: &Network, at: DVec3, reach: f64) -> Option<DVec2> {
    let (cx, cy) = ((at.x / 50.0).floor() as i32, (at.y / 50.0).floor() as i32);
    let mut best: Option<(f64, DVec2)> = None;
    let mut seen: Vec<usize> = Vec::new();
    for dx in -1..=1 {
        for dy in -1..=1 {
            for &i in ped.grid.get(&(cx + dx, cy + dy)).map(|v| v.as_slice()).unwrap_or(&[]) {
                if seen.contains(&i) {
                    continue;
                }
                seen.push(i);
                let l = &net.lanes[i];
                let len = l.length();
                let mut s = 0.0f32;
                while s <= len {
                    let (q, _) = l.at(s);
                    let d = (q - at).truncate().length();
                    if d <= reach && (q.z - at.z).abs() < 2.0 && best.map(|b| d < b.0).unwrap_or(true) && !in_waiter_reach(net, q) {
                        best = Some((d, q.truncate()));
                    }
                    s += 0.5;
                }
            }
        }
    }
    best.map(|b| b.1)
}

/// May a pedestrian at the kerb start across crossing `c` (from `walk.legs[c.from]` on)? A
/// pedestrian light must show green, and the time left to get across - the green and then
/// the clearance until a light of the carriageway turns green - must do; without a light no
/// car may be about to pass the crossing. Somebody who has waited very long takes any green
/// (never a red).
fn kerb_verdict(ped: &mut PedNet, at: &Around, walk: &PedWalk, c: &Cross, me: &Me, yields: &mut Yields) -> Result<(), &'static str> {
    let net = at.net;
    let next = walk.legs[c.from];
    let pace = me.pace;
    let held = walk.held;
    // [ROLLBACK r4c-L1] the whole crossing, kerb to kerb (the first piece alone was timed, and
    // only where it cut a lane: a kerb stub cuts none, and nobody looked at the cars at all)
    let t_cross = c.span as f64 / pace.max(0.5) + 1.0;
    let cuts = span_cuts(ped, net, &walk.legs[c.from..=c.to]);
    // [ROLLBACK r5-P6] nobody starts across while a vehicle stands on the crossing - its box
    // over any of its cuts, under 0.5 m/s, not one stopping for people - for PED_OCCUPIED_CAP
    // of its standing at most; those on the carriageway keep P1 and P5. (At a lit crossing
    // on green nobody looked at the vehicles at all: 339 step-ons in a minute while a bus
    // stood with two pieces under its body, X10 streams report; a lead-following car on the
    // zebra had 118 step-ons round its nose.)
    // ([ROLLBACK r5-P6b] the cuts of the first metres of the way only, and a vehicle creeping
    // onto one at under PED_CREEP_SPEED counts as standing on it: a queue car rolling onto the
    // crossing at 0.5-0.7 m/s at a lit green passed, X10 #485, 25 step-ons)
    let start = next.at(net, 0.0).0.truncate();
    let near_cuts: Vec<DVec2> = cuts.iter().copied().filter(|x| (*x - start).length() <= PED_OCCUPIED_REACH).collect();
    // ([ROLLBACK r4c-P5] or its box over the first 1.5 m of the way off the kerb, where no cut
    // lies: a bus on the kerb stub; P5 held for 5 s of the person's own waiting, then they
    // stepped round it - one rule now, review r7)
    let (from, _) = next.at(net, 0.0);
    let (ahead, _) = next.at(net, 1.5);
    let way = (ahead - from).truncate().normalize_or_zero();
    let occupied = || {
        at.cars
            .iter()
            // (the "abandoned car" exemption past the cap is for one that stands: a car that
            // stood 19 s on the cut and then crept 2.2 m at 0.4-0.7 m/s kept its clock past
            // the cap and 56 people stepped on at a lit green, X10 0837 car 485)
            .any(|car| {
                let speed = car.vel.length();
                speed < PED_CREEP_SPEED
                    && !car.yielding
                    && (car.stood < PED_OCCUPIED_CAP || speed >= 0.5)
                    && (near_cuts.iter().any(|x| car.body.near(*x, KERB_CLEAR)) || covers_path(&car.body, from.truncate(), way, 1.5))
            })
    };
    if let (Some((ctl, li)), Some(t)) = (c.light, at.lights) {
        if let Some((state, left)) = t.light_state(ctl, li) {
            if !omsi_sim::traffic::TrafficLightController::allows_go(state) {
                return Err("red light");
            }
            if held <= 150.0 {
                // A pedestrian green is short (8 s at Grundorf for an 11.6 m crossing that
                // takes 10.7 s): who starts on green crosses in the clearance time after
                // it, until the cars get their green. Only the green alone was counted, so
                // nobody ever started on green and everybody went across on red after
                // 150 s, in front of moving cars.
                // [ROLLBACK r4c-P2] the whole crossing, kerb to kerb (the next piece alone was
                // counted: people stepped onto a 28 s crossing 1-4 s before the red)
                let arrival = c.span as f64 / pace.max(0.5) + 1.0;
                let paths: Vec<usize> = walk.legs[c.from..=c.to].iter().map(|l| l.lane).collect();
                let window = pedestrian_window(Some(&mut *ped), net, t, &paths, left);
                // (a few go anyway, so that a crowd does not freeze all at once; the change
                // then turns them back or sends them on, P1)
                if !makes_it(left, window, arrival) && !(left > 2.0 && late_stepper(me.id, c.key)) {
                    return Err("the green ends before they would be across");
                }
            }
            if occupied() {
                return Err(STANDS_ON);
            }
            return Ok(());
        }
    }
    // somebody who has waited long accepts a shorter gap (down to the time the crossing
    // takes, never less): a steady stream does not hold them for ever, but nobody walks
    // out in front of a car that is about to be there (after 45 s they used to ignore
    // the cars altogether)
    let margin = if held > 45.0 { 0.0 } else if held > 20.0 { 1.0 } else { 2.5 };
    // [ROLLBACK r4c-P4] a vehicle that has stood at the crossing for a while, held by a stream
    // of people that never gaps (a station's crossing: heads stood 75 and 90 s): the people
    // arriving let it pass, for a few seconds; those on the carriageway go on across
    let stream = at.streams.get(&c.key).map(|v| v.iter().any(|(id, q, _)| *id != me.id && (*q - me.pos).length() < PED_STREAM_GAP)).unwrap_or(false);
    // [ROLLBACK r5-P5] a dense stream yields for longer: until the vehicle is through the
    // cut (it is no longer `stands_at` then), 2 x PED_YIELD_MAX at most, under PED_YIELD_CAP
    let dense = at.streams.get(&c.key).map(|v| v.len() as f32 >= PED_DENSE_STREAM * c.total.max(1.0)).unwrap_or(false);
    let window = if dense { 2.0 * PED_YIELD_MAX } else { PED_YIELD_MAX };
    for car in at.cars {
        if car.vel.length() >= 0.5 || car.stood <= PED_YIELD_AFTER || !stands_at(car, &cuts) {
            continue;
        }
        // (somebody of the stream still on the crossing in front of it: the vehicle's seconds
        // start once they are off it - with the window running from the first yield the
        // crossing was clear for a second of the six, and the queue stood 90 s)
        // (only somebody walking towards the cut: one lingering there held the clock for good)
        let in_front = at.streams.get(&c.key).map(|v| {
            v.iter().any(|(id, q, vel)| {
                *id != me.id && cuts.iter().any(|x| (*q - *x).length() < 2.0 && vel.length() > 0.2 && vel.dot(*x - *q) > 0.0 && stands_at(car, std::slice::from_ref(x)))
            })
        });
        let in_front = in_front.unwrap_or(false);
        match yields.get(&car.id).copied() {
            // (all the yields to one vehicle, resets and renewals too, within PED_YIELD_CAP of
            // the first)
            Some((_, first)) if at.time - first >= PED_YIELD_CAP as f64 => {}
            Some((since, first)) if at.time - since < window as f64 => {
                if in_front {
                    yields.insert(car.id, (at.time, first));
                }
                return Err(YIELDS);
            }
            // [ROLLBACK r4c-L4] still standing after the people's own turn (as long again): it
            // is let go again (once, the people on a long crossing outlasted the window and
            // the vehicle stood for good)
            Some((since, first)) if stream && at.time - since >= 2.0 * PED_YIELD_MAX as f64 => {
                yields.insert(car.id, (at.time, first));
                return Err(YIELDS);
            }
            Some(_) => {}
            None if stream => {
                yields.insert(car.id, (at.time, at.time));
                return Err(YIELDS);
            }
            None => {}
        }
    }
    if occupied() {
        return Err(STANDS_ON);
    }
    // [ROLLBACK r4c-P5] a vehicle moving across the first 1.5 m of the way holds them; one
    // standing there is `occupied`'s (above: up to PED_OCCUPIED_CAP of its standing, then
    // they step round it); one beside the way, or the player's bus with its nose short of
    // it, does not hold (nine walkers were held 5 s each 1-1.5 m from the nose of a
    // standing bus)
    for car in at.cars {
        let speed = car.vel.length();
        if speed >= 0.5 && !car.yielding && covers_path(&car.body, from.truncate(), way, 1.5) {
            return Err(STANDS_ON);
        }
    }
    for x in &cuts {
        for car in at.cars {
            // [ROLLBACK pedyield-1] a car stopping for people stops: when the car yields,
            // the pedestrian passes (a child and a taxi stood for each other for minutes,
            // the child waiting for the taxi's creep, the taxi for the child)
            if car.yielding {
                continue;
            }
            let rel = *x - car.pos;
            let dist = rel.length();
            if dist > 90.0 {
                continue;
            }
            let speed = car.vel.length();
            let half = car.half_len;
            if speed < 0.5 {
                continue;
            }
            let dir = car.vel / speed;
            let along = rel.dot(dir);
            let lateral = rel.perp_dot(dir).abs();
            if along > -half && lateral < 3.5 && along / speed < t_cross + margin {
                return Err("waits for a car to pass");
            }
        }
    }
    Ok(())
}

/// [ROLLBACK r4c-L1] Where the pieces `legs` of a crossing cut a carriageway: the pieces' own
/// cuts and those of the steps between them (a walk leaves every other 1 m piece out, and
/// the piece over the lane may be one of those).
fn span_cuts(ped: &mut PedNet, net: &Network, legs: &[Leg]) -> Vec<DVec2> {
    let mut out: Vec<DVec2> = Vec::new();
    let mut add = |x: DVec2, out: &mut Vec<DVec2>| {
        if !out.iter().any(|q| (*q - x).length() < 1.5) {
            out.push(x);
        }
    };
    for l in legs {
        for x in ped.crossings(net, l.lane).to_vec() {
            add(x, &mut out);
        }
    }
    for w in legs.windows(2) {
        let (a, b) = (w[0].at(net, w[0].len()).0, w[1].at(net, 0.0).0);
        if (b - a).truncate().length() > 0.05 {
            for x in street_cuts(net, a, b) {
                add(x, &mut out);
            }
        }
    }
    out
}

/// Where the segment `a`-`b` crosses a carriageway lane.
fn street_cuts(net: &Network, a: DVec3, b: DVec3) -> Vec<DVec2> {
    let m = (a + b) * 0.5;
    let (cx, cy) = ((m.x / 50.0).floor() as i32, (m.y / 50.0).floor() as i32);
    let mut seen: Vec<usize> = Vec::new();
    let mut out = Vec::new();
    for dx in -1..=1 {
        for dy in -1..=1 {
            for &i in net.grid.get(&(cx + dx, cy + dy)).map(|v| v.as_slice()).unwrap_or(&[]) {
                let l = &net.lanes[i];
                if l.kind != LaneKind::Street || seen.contains(&i) {
                    continue;
                }
                seen.push(i);
                for v in l.points.windows(2) {
                    if (a.z - v[0].z).abs() > 3.0 {
                        continue;
                    }
                    if let Some(x) = seg_cross(a.truncate(), b.truncate(), v[0].truncate(), v[1].truncate()) {
                        out.push(x);
                    }
                }
            }
        }
    }
    out
}

fn seg_cross(a: DVec2, b: DVec2, c: DVec2, d: DVec2) -> Option<DVec2> {
    let r = b - a;
    let s = d - c;
    let den = r.perp_dot(s);
    if den.abs() < 1e-9 {
        return None;
    }
    let t = (c - a).perp_dot(s) / den;
    let u = (c - a).perp_dot(r) / den;
    (t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0).then(|| a + r * t)
}

/// [ROLLBACK doorwait-68] How long (s) the people wait for a door that is opening before they
/// look for another.
const DOOR_OPENING_GRACE: f64 = 3.0;

/// [ROLLBACK solidcars-1] People on foot collide with nothing (not each other, not the
/// scenery: [ROLLBACK nocoll-93], they only hindered) except vehicles, which are solid to
/// people on the ground:
/// somebody with a car in front of them stops (and waits, then goes round it) instead of
/// walking into it. People still pass through each other and the scenery as before.
const VEHICLE_COLLISIONS: bool = true;

/// How far (m) from where they stand somebody left standing looks for a pavement to walk on.
const STANDING_REACH: f64 = 10.0; // [ROLLBACK reach-93] was 30.0
/// [ROLLBACK respawn-61] How long (s) somebody who got off and finds no pavement stands before
/// being respawned: sooner when nobody sees them.
const STANDING_GIVE_UP: f32 = 1.0;
/// [ROLLBACK party-70] The share of the waiting who wait with others (two to four).
const PARTY_CHANCE: f32 = 0.22;

#[derive(Debug, Clone)]
enum State {
    /// A pedestrian strolling the pavements (Omsi.exe's task 8, `WalkStreet`).
    Strolling(PedWalk),
    /// Moved by somebody else: an avatar, or one of a LAN host's people.
    Idle,
    /// Task 8 without a path (+0x2f0 = -1): somebody who got off where no pavement is
    /// stands where they are until the player is gone.
    Standing,
    /// A passenger (see `humans_pax`).
    Pax(Box<Pax>),
}

impl State {
    fn name(&self) -> &'static str {
        match self {
            State::Strolling(_) => "WalkStreet",
            State::Idle => "Idle",
            State::Standing => "WalkStreet",
            State::Pax(p) => p.task.name(),
        }
    }
    fn bus(&self) -> Option<BusId> {
        match self {
            State::Pax(p) => p.inside.or(p.bus),
            _ => None,
        }
    }
}

/// Where a person is: on the ground, or inside a bus at a point of its frame.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Place {
    Ground,
    Bus(BusId, Vec3),
}



/// [round5 C2] Below this speed (m/s) somebody counts as standing and `Person::still` runs:
/// standers pressed by a crowd jitter at 0.05-0.1 m/s (viz4 case01: eleven on a pavement
/// beside a car kept `still` at 0 for 34 s at the old 0.05, and the car's "stood at the edge
/// 3 s" exemption never fired).
const STILL_BELOW: f32 = 0.15;

/// `Person::still` after a frame of `dt` at `speed`.
fn still_after(still: f32, speed: f32, dt: f32) -> f32 {
    if speed < STILL_BELOW {
        still + dt
    } else {
        0.0
    }
}

pub struct Person {
    id: u32,
    ty: Arc<HumanType>,
    /// Clothing variant (`HumanType::variant_texture`).
    variant: usize,
    meshes: Vec<(MeshId, usize)>,
    position: DVec3,
    heading: f64,
    /// Heading in the bus frame while inside one.
    lheading: f64,
    place: Place,
    /// Velocity in the plane the person walks in (ground or bus floor).
    vel: DVec2,
    pace: f64,
    /// [ROLLBACK r4c-P6] The pace walked now (faster across when caught by the change or with
    /// a vehicle standing for them), for the recording.
    walk_pace: f64,
    activity: Activity,
    /// The animation: Omsi.exe's walk phase and joint angles (sub_626ae8).
    anim: OmsiAnim,
    state: State,
    /// Seconds in the current state.
    t_state: f32,
    /// Skinned positions and normals, per mesh.
    skins: Vec<(Vec<Vec3>, Vec<Vec3>)>,
    /// The bones the skins were made with, and whether this frame's pose changed them
    /// (somebody standing still keeps the mesh of the frame before: skinning and uploading
    /// thirty waiting people every frame took 2 ms of the frame at a bus station).
    skin_bones: Option<[glam::Affine3A; omsi_sim::human::SLOTS]>,
    pose_changed: bool,
    /// Interior light of the bus the person is in (0 outside).
    interior: f32,
    /// The interior light as drawn: it follows `interior` over a moment (stepping through
    /// the door, people lit up and went dark again from one frame to the next).
    lit: f32,
    /// The tilt of the floor the person stands on (a bus pitching under the brakes and
    /// leaning in a bend), without its heading: riders are drawn with it. Upright on the
    /// ground; drawn upright in a tilted bus, their feet sank through the floor on one side.
    tilt: Mat4,
    /// Age in years: the `.hum`'s `[age]`, else 40 as in OMSI. The
    /// ticket pack's tickets have age ranges (the reduced fare is for 6..13).
    age: f32,
    /// Seconds without getting nearer the goal while wanting to move; seconds left
    /// passing through others.
    stuck: f32,
    /// [ROLLBACK pedhold-1] Seconds standing still (under 0.05 m/s).
    still: f32,
    /// [ROLLBACK pedyield-1] On a crossing, off the kerb (a stroller on a path that crosses
    /// a carriageway, past its first steps): the cars of the lanes still ahead stop for
    /// them as for somebody stepping in.
    on_crossing: bool,
    /// The last decision at a kerb (`may_cross`): "go", or why they wait - for the traffic
    /// recording, which shows why somebody froze.
    last_decision: Option<&'static str>,
    /// [ROLLBACK stepout-70] Just off a bus: where to walk straight to (1.8 m off its side) and the
    /// seconds left for it, before the pavement's path takes over; no waiting for the standing bus.
    step_out: Option<(DVec2, f32)>,
    /// Seconds a standing vehicle has stood in the way (see the crowd step).
    car_wait: f32,
    /// Seconds left going round something in the way off the pavement's line (a lamp post
    /// on the path): the corridor does not pull them back into it meanwhile.
    detour: f32,
    /// [ROLLBACK r4c-P5] The way they wanted to walk last, and the seconds since they last turned
    /// round (beside a standing vehicle: at most once in `PED_TURN_EVERY`).
    want_dir: DVec2,
    turned: f32,
    /// [ROLLBACK around-86] Going round a standing bus or lorry: the two corners of its box to
    /// walk through (the near end, the far side of it), the one on its way and the seconds left.
    around: Option<([DVec2; 2], usize, f32)>,
    /// Why the person is standing, for `OMSI_DEBUG_PAX`.
    why: &'static str,
    /// Whether this person has ever been posed (an unposed model is the file's T-pose).
    skinned: bool,
    /// Frames since the last pose and where the person stood then (the
    /// feet of a mesh posed a frame ago stay on the floor when it is drawn there).
    since_posed: u32,
    posed_at: (DVec3, f64),
    /// Ankles of the last pose (model frame), for `OMSI_TRACE_PAX`.
    ankles: [Vec3; 2],
    /// A scripted test person (`OMSI_PAX_GALLERY`).
    puppet: Option<Puppet>,
    /// LAN play: one of the host's people, drawn where the host says (`mirror_set`).
    remote: bool,
}

impl Person {
    pub fn state_name(&self) -> String {
        format!("#{} {} ({})", self.id, self.state.name(), self.why)
    }
    pub fn position(&self) -> DVec3 {
        self.position
    }
    fn inside(&self, bus: BusId) -> bool {
        matches!(self.place, Place::Bus(b, _) if b == bus)
    }
    /// [ROLLBACK r4c-L4] Waiting at a kerb to cross (held there by `may_cross`'s verdict), off the
    /// carriageway: the cars' people check takes them as clear outside their footprint.
    fn at_kerb(&self) -> bool {
        matches!(&self.state, State::Strolling(w) if w.held > 0.0) && !self.on_crossing && self.last_decision.map(|d| d != "go").unwrap_or(false)
    }
    fn label(&self) -> String {
        format!(
            "#{} {}",
            self.id,
            self.ty
                .def
                .path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
        )
    }
}

/// What a person wants this frame.
struct Want {
    vel: DVec2,
    /// Heading to turn to when standing (world on the ground, bus frame inside).
    face: Option<f64>,
    give: f64,
    corridor: Option<(DVec2, DVec2, f64)>,
    /// What they do when not walking.
    idle: Activity,
}

impl Want {
    fn stand(face: Option<f64>, idle: Activity) -> Want {
        Want {
            vel: DVec2::ZERO,
            face,
            give: 0.35,
            corridor: None,
            idle,
        }
    }
}

/// Velocity towards `to`, easing into the stop over the last metre.
/// The corridor a walker at `pos` keeps to: none while they are more than `band` off it.
/// Eased into it at up to 0.6 m/s (twice a step) on top of walking there, the people
/// getting off a bus slid sideways from its door to the pavement at twice their pace
/// (#1033, #1079). [ROLLBACK pedhold-1] A stroller nudged off the
/// pavement by the others (three abreast at a kerb) is drawn back from up to `STROLL_BAND`
/// away; at 0.1 m they walked on along the kerb line in the carriageway, and a car stood
/// behind them for good.
fn path_corridor_band(pos: DVec2, corridor: Option<(DVec2, DVec2, f64)>, band: f64) -> Option<(DVec2, DVec2, f64)> {
    corridor.filter(|&(a, b, dev)| (crowd::clamp_to_corridor(pos, a, b, dev) - pos).length() <= band)
}

fn arrive(from: DVec2, to: DVec2, pace: f64) -> DVec2 {
    let d = to - from;
    let dist = d.length();
    if dist < 0.1 {
        return DVec2::ZERO;
    }
    let speed = (pace * dist.min(1.0)).max(if dist > 0.3 { 0.25 } else { 0.0 });
    d / dist * speed
}

/// [ROLLBACK paxvoice-99] The voice files of the passengers' complaints (the `pax_complain` setting).
const COMPLAINTS: [&str; 7] = [
    "TooBad_",
    "TooDark_",
    "TooLate_",
    "TooHot_",
    "TooCold_",
    "TooWet_",
    "BadChange_",
];

/// Seconds after one passenger's greeting or complaint before anybody says another.
const CHAT_PAUSE: f64 = 12.0;

pub struct Humans {
    types: Vec<Arc<HumanType>>,
    pub people: Vec<Person>,
    rng: u64,
    next_id: u32,
    /// Seconds since the start.
    time: f64,
    /// Passenger cabins by vehicle files (the front vehicle and its coupled parts).
    cabins: HashMap<Vec<PathBuf>, Option<Arc<Cabin>>>,
    player_cabin: Option<Arc<Cabin>>,
    player_next_stop: Option<RequestStop>,
    /// Which places of each bus are taken.
    seats: HashMap<BusId, Vec<bool>>,
    /// The bus stops as Omsi.exe keeps them for the people (see `humans_pax`).
    stops: HashMap<i64, PaxStop>,
    /// Kilometres each bus has driven (the odometer the riders read, +0x430).
    odometer: HashMap<BusId, f64>,
    /// The `PAX_Entry<n>_Req` / `PAX_Exit<n>_Req` of each bus this frame.
    pax_req: HashMap<BusId, (Vec<bool>, Vec<bool>)>,
    /// Its `PAX_Entry<n>_Busy` / `PAX_Exit<n>_Busy`: somebody in that doorway.
    pax_busy: HashMap<BusId, (Vec<bool>, Vec<bool>)>,
    /// [ROLLBACK doorwait-66] Doors (entries, exits) of each bus that are opening: the door
    /// variable is rising and the script has not flagged the entry / exit open yet. Whoever
    /// chooses a door counts them as open, so nobody turns away from the door being opened.
    door_moving: HashMap<BusId, (Vec<bool>, Vec<bool>)>,
    /// The last reading of a door variable: (bus, exit?, index).
    door_prev: HashMap<(BusId, bool, usize), f32>,
    /// [ROLLBACK doorwait-68] When each door began to open (bus, exit?, index): the wait for a
    /// door that is being opened ends after `DOOR_OPENING_GRACE` seconds.
    door_since: HashMap<(BusId, bool, usize), f64>,
    /// The occupancy variables its places name, and whether somebody is on each (#721).
    pax_places: HashMap<BusId, Vec<(String, bool)>>,
    /// Who is at the player's cash desk (+0x7a8), how often the driver has been asked
    /// again (0x859bc4) and the most of that in this sale (0x859df4).
    desk_busy: Option<u32>,
    pardons: u8,
    pardon_max: u8,
    /// The first populate put people at the stops; later stops fill on foot when in sight.
    started: bool,
    ped: Option<PedNet>,
    /// [ROLLBACK r4c-P4] Seconds each vehicle near has stood, by id.
    /// [ROLLBACK r5-P5] Per vehicle: its stood clock and where it last stood (`stood_clock`).
    car_stood: HashMap<u64, (f32, DVec2)>,
    /// [ROLLBACK r4c-P4] The vehicles people at an unlit crossing yield to, and since when.
    yields: Yields,
    /// [ROLLBACK r4c-P4] The people on the carriageway of each crossing (by its key) this frame.
    streams: Streams,
    hidden: Vec<usize>,
    /// GPU side of the human types, shared by everyone of a type: textures by file and the
    /// materials of every (type, mesh) - each person used to upload its own copies - and
    /// the meshes and instances of the people who have gone, taken over by the next person
    /// of the same type (the skinned vertices are rewritten anyway). Without that every
    /// passenger who ever appeared kept a mesh, its textures and materials on the GPU.
    gpu_textures: HashMap<PathBuf, Option<omsi_render::TextureId>>,
    /// Per (type, clothing variant, mesh): its materials, and the meshes and instances of
    /// people who have gone, kept for the next person dressed alike.
    gpu_materials: HashMap<(usize, usize, usize), Vec<MaterialId>>,
    spare: HashMap<(usize, usize, usize), Vec<(MeshId, usize)>>,
    /// Stop the player's bus is serving (standing at it).
    served_stop: Option<i64>,
    /// Timetable buses at a stop: id → (stop, time the visit began).
    ai_visits: HashMap<u64, (i64, f64)>,
    /// When each bus last had a door open (the passengers' clock).
    last_door_open: HashMap<BusId, f64>,
    /// Timetable buses to keep at their stop for a few seconds more (for the traffic): the
    /// bus, the stop it must be serving for it (none: any), the seconds.
    holds: Vec<(u64, Option<i64>, f32)>,
    /// Door requests for the timetable buses' scripts.
    ai_requests: Vec<(u64, DoorWants)>,
    pub tickets: Option<Arc<omsi_content::tickets::TicketPack>>,
    /// Current ticket request at the player's cash desk: (ticket name, value).
    pub request: Option<(String, f32)>,
    /// Payment on the desk: (paid, ticket value), and the change still owed after the ticket.
    pub paid: Option<(f32, f32)>,
    pub change_due: Option<f32>,
    pub money: Option<crate::money::Money>,
    /// A rider pressed the stop button for the next stop (the app fires the vehicle trigger `int_haltewunsch`).
    pub stop_request: bool,
    /// Tickets sold at the cash desk this session and what they were worth.
    pub tickets_sold: u32,
    pub ticket_cash: f32,
    /// Passengers that reached the cash desk, and those the driver served there.
    pub boarded: u32,
    pub served: u32,
    /// OMSI's rating counters: people who stepped into the player's bus
    /// and of those who had nothing to complain about (comfort = content / stepped in);
    /// tickets asked for and the points for selling them, two for the right change, one
    /// for the wrong (ticket selling = points / 2 × asked).
    pub stepped_in: u32,
    pub content: u32,
    pub ticket_requests: u32,
    pub ticket_points: u32,
    /// `PAX_Entry<i>_Req`: somebody at the kerb wants in through entry `i`.
    pub entry_req: Vec<bool>,
    /// `PAX_Exit<i>_Req`: somebody inside wants out through exit `i`.
    pub exit_req: Vec<bool>,
    /// `PAX_Entry<i>_Busy` / `PAX_Exit<i>_Busy`: somebody stands in that doorway.
    pub entry_busy: Vec<bool>,
    pub exit_busy: Vec<bool>,
    sync_frame: u32,
    /// Feet put down since the app last collected them (see [`Humans::take_footfalls`]).
    footfalls: Vec<ambience::Footfall>,
    /// `[trafficdensity_passenger]` factor for the current hour (set by the app).
    pub density: f32,
    /// The clock's time of day in seconds (set by the app): day tickets sell by it.
    pub time_of_day: f64,
    /// How late the player's bus is on its duty (s; set by the app): over five minutes,
    /// boarding passengers may say so.
    pub delay: f64,
    /// The game's folder (the ticket pack's voices are found from it).
    root: std::path::PathBuf,
    /// What passengers said since the app last collected it (see `take_voice_lines`).
    voice_lines: Vec<VoiceLine>,
    /// When each voice file was last said (seconds of `time`): OMSI keeps such a list
    /// and says a greeting or a complaint only when that
    /// very file has not been heard for 10 s - without it every boarding passenger said
    /// "Hallo" one after the other.
    voice_said: HashMap<std::path::PathBuf, f64>,
    /// [ROLLBACK paxvoice-99] What passengers may say (the `pax_greet`, `pax_ticket` and
    /// `pax_complain` settings): bit 0 the greetings, bit 1 the ticket they ask for (and
    /// what goes with buying it), bit 2 the complaints.
    pub voices: u8,
    /// [ROLLBACK seatpick-69] How much passengers want to sit (the `pax_sit` setting): 0 the
    /// original's random place, else the odds of looking for a seat are scaled by it.
    pub sit_bias: f32,
    seat_clock: f32,
    /// When a passenger last changed places (by person id), and when a seat was last given up
    /// in a bus.
    seat_moved: HashMap<u32, f64>,
    seat_gift: HashMap<BusId, f64>,
    /// The party being made at a stop (stop, party, destination, line, ride, people still to come),
    /// the next party number, and the seats planned for a party in a bus (bus, places, when).
    party_open: Option<(i64, u32, Option<String>, Option<usize>, f32, u8, u8)>,
    party_next: u32,
    party_plan: HashMap<u32, (BusId, Vec<usize>, f64)>,
    /// When anybody last greeted or complained (seconds of `time`).
    last_chat: f64,
    /// Avatars (the player on foot, other players' walkers): key → person id, and what
    /// the game wants of each this frame.
    avatars: HashMap<u32, u32>,
    avatar_cmds: HashMap<u32, AvatarCmd>,
    /// Avatars not drawn (the first-person view), by person id.
    avatar_hidden: HashMap<u32, bool>,
    /// The buses of the last tick (for the avatars' seats and doors).
    last_buses: Vec<BusNow>,
    /// [ROLLBACK payload-121] The people the developer tools put into the player's bus (`fill_places`).
    fake: HashSet<u32>,
    /// Only avatars: nobody else is put on the map (the passengers are off).
    pub avatar_only: bool,
    /// The player has got up and left the wheel: a standing bus with a door open is left
    /// by its riders as at a terminus (see `ALL_OUT_STOP`).
    pub driver_away: bool,
    /// Per bus stop, Omsi.exe's station targets (0x61cb18, `Schedule::stop_targets`): the
    /// stops the trips go on to, each with the termini of those trips. A person waiting there
    /// wants one of them and boards only a bus showing one of its termini; at a stop no trip
    /// goes on from, anybody takes the first bus (0x61c33c).
    pub stop_targets: Option<HashMap<i64, Vec<(String, HashSet<String>)>>>,
    /// [ROLLBACK soon-90] Per stop, the destinations a bus is about to leave for (within the
    /// next 20 minutes, see `Schedule::soon_targets`), renewed every few seconds of the clock; the
    /// people who appear there draw their destination from these. None: not known (any).
    pub soon: Option<HashMap<i64, HashSet<String>>>,
    pub soon_clock: f64,
    /// [ROLLBACK reshuffle-91] The clock as of the last look (a jump of it is a new time set), and
    /// the stops are to be filled up at once (the people were sent away for a new time).
    pub clock_seen: f64,
    refill_all: bool,
    /// The timetable's name of each stop object (`Schedule::stop_names`), the names the
    /// targets above are made of.
    pub stop_names: Option<HashMap<i64, String>>,
    /// [ROLLBACK ridemin-71] Minutes by the timetable between two stops by name (`Schedule::ride_minutes`).
    pub ride_minutes: Option<HashMap<(String, String), f32>>,
    /// The player's duty (`set_duty`): its trip, the stop of it the duty is due at, and
    /// whether the trip has reached its last stop. None: free drive, and nobody waiting
    /// boards the player's bus.
    duty: Option<(Arc<DutyTrip>, usize, bool)>,
    /// Buses whose validator somebody used since the app last looked (`take_stamped`).
    stamped: Vec<BusId>,
    /// How many people to keep strolling the pavements about the player (the `pedestrians`
    /// setting), scaled by `density` and by how much pavement is around, within the pool.
    pub pedestrians: usize,
    /// Omsi.exe's people (`[AIMaxCountRandom]`'s second line, the `ai_max_humans` setting):
    /// it makes that many at the start (0x709274) and draws everybody waiting at a stop,
    /// walking the pavements or riding from them - never more; at most half of them walk
    /// the pavements (0x62463c). Here people are made as they are wanted, so they are
    /// counted against it instead.
    pub max_people: usize,
    stroll_timer: f32,
    /// Passengers pay the exact fare: no change is ever due.
    pub exact_fare: bool,
    /// How passengers board (`boarding` in the settings): `auto` - pay and take the
    /// ticket by themselves after a moment; `pay` - wait at the desk for the driver to
    /// sell it (and show a pass after `PAY_PATIENCE`); `walk` - no cash desk at all.
    pub boarding: String,
    /// The driver pressed the ticket key (`ticket_give`): sell the requested ticket.
    pub give_ticket: bool,
    /// The driver pressed `change_give`: all the change owed goes on the tray at once.
    pub give_change_all: bool,
    /// The key that sells a ticket, as the HUD names it.
    pub ticket_key: String,
    /// Where the player looks from (set by the app every frame).
    pub eye: Option<Eye>,
    /// Around where people are kept (the player's bus, else the camera).
    center: DVec3,
    /// A line for the HUD about something that just happened.
    message: Option<String>,
    /// Frames ticked, total and longest tick (ms).
    tick_stats: (u32, f64, f64),
    /// Where the time of this tick went (stage, ms since the one before), for the slow
    /// ticks OMSI_PROFILE reports.
    tick_stages: Vec<(&'static str, f64)>,
    /// Speed, heading and floor acceleration of each bus last frame (for the riders' balance).
    bus_motion: HashMap<BusId, (f64, f64, DVec2)>,
    /// `types` has been cut down to the map's `humans.txt`.
    map_humans_done: bool,
    /// Simulation time of the last `sync`.
    last_sync: f64,
    /// Frames synced, people posed and skinned, the time that took and the part of it spent
    /// uploading (ms), in total.
    pose_stats: (u32, usize, f64, f64),
    /// `OMSI_TRACE_PAX=<csv>`: every person near the eye, every frame (see `sync`).
    trace: Option<std::io::BufWriter<std::fs::File>>,
    /// `World::tiles_generation` the stops were last checked against.
    tiles_seen: u64,
    /// LAN play: this game draws the host's people instead of its own (`lan_world`).
    mirror: bool,
    /// LAN play: where the other players are (host): people are kept around them too.
    pub lan_centers: Vec<DVec3>,
    /// LAN play: the other players' buses this frame (`set_remote_buses`), for their riders
    /// to sit in. Nobody of ours boards them: their doors count as shut.
    remote_now: Vec<BusNow>,
    /// The vehicles the player placed and is not driving now (`placed_bus_id`): their
    /// riders stay in them when the player drives another.
    placed_now: Vec<BusNow>,
    /// LAN play (client): waiting people our bus could take, to ask the host for, and when
    /// each was last asked for.
    claims_out: Vec<u32>,
    claimed: HashMap<u32, f64>,
    /// The host's people waiting at a stop (client): (stop, waiting place).
    mirror_wait: HashMap<u32, (i64, usize)>,
    /// How the player's bus is driven, for its riders' complaints.
    comfort: RideComfort,
    /// LAN play (host): the waiting people handed over to another player's bus, by stop and
    /// that bus. They count among the people of the stop while the bus stands there, as
    /// the people who board a bus of ours keep their stop until it has left: without them
    /// the stop filled up again at once - one more person a frame once its 10..15 s were
    /// up - and the client's bus took them all, one stream of passengers that never ended
    /// (#842, #840, #830).
    handed: Vec<(i64, u64)>,
}

/// Resolve each map entry directly, including human packs with nested folders.
/// Keep duplicate entries as spawn weights, but load each definition only once.
fn map_human_types(root: &Path, list: &[String]) -> Vec<Arc<HumanType>> {
    // (keyed case-blind: OMSI paths are, and the lists spell one file several ways)
    let mut loaded: HashMap<String, Option<Arc<HumanType>>> = HashMap::new();
    let mut picked = Vec::new();
    for line in list {
        let rel = line.trim().replace('\\', "/");
        // Lists normally include Humans/, but also accept paths relative to that folder.
        let rel = if rel.to_ascii_lowercase().starts_with("humans/") {
            rel
        } else {
            format!("Humans/{rel}")
        };
        let path = omsi_cfg::resolve_path(root, &rel);
        let ty = loaded.entry(path.to_string_lossy().to_lowercase()).or_insert_with(|| {
            match HumanType::load(&path) {
                Ok(t) => Some(Arc::new(t)),
                Err(e) => {
                    log::warn!("map human {}: {e:#}", path.display());
                    None
                }
            }
        });
        if let Some(t) = ty {
            picked.push(t.clone());
        }
    }
    picked
}

impl Humans {
    /// LAN uses the room id as the shared source of randomness.  This keeps the
    /// initial pedestrian selection and their generated identities identical on
    /// the host and clients; subsequent movement remains simulation-local.
    pub fn set_lan_seed(&mut self, seed: u64) {
        self.rng = (seed ^ 0xA5A5_5A5A_1F2E_3D4C) as u64 | 1;
    }

    pub fn new(root: &Path) -> Humans {
        let mut types = Vec::new();
        // `Humans/<group>/*.hum` of every content root (an installed map or mod brings its
        // own people); a file of the same group and name higher up replaces the stock one
        let mut roots = omsi_cfg::content_dirs("Humans");
        if roots.is_empty() {
            roots.push(root.join("Humans"));
        }
        // (group, file name, path), sorted by group and name as the single folder used to be
        let mut found: Vec<(std::ffi::OsString, std::ffi::OsString, std::path::PathBuf)> =
            Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for r in &roots {
            for (group, is_dir) in omsi_cfg::vfs::list_dir(r).unwrap_or_default() {
                if !is_dir {
                    continue;
                }
                let d = r.join(&group);
                for (n, _) in omsi_cfg::vfs::list_dir(&d).unwrap_or_default() {
                    let lower = n.to_string_lossy().to_ascii_lowercase();
                    if !lower.ends_with(".hum") || lower.contains("driver") {
                        continue;
                    }
                    if seen.insert(format!(
                        "{}/{lower}",
                        group.to_string_lossy().to_ascii_lowercase()
                    )) {
                        found.push((group.clone(), n.clone(), d.join(&n)));
                    }
                }
            }
        }
        found.sort();
        let files: Vec<std::path::PathBuf> = found.into_iter().map(|(_, _, p)| p).collect();
        for f in files {
            match HumanType::load(&f) {
                Ok(t) => types.push(Arc::new(t)),
                Err(e) => log::warn!("human {}: {e:#}", f.display()),
            }
        }
        log::info!("humans: {} types", types.len());
        if omsi_cfg::env::var_os("OMSI_DEBUG_HUMANS").is_some() {
            for t in &types {
                let (mut lo, mut hi) = (f32::MAX, f32::MIN);
                for m in &t.meshes {
                    for v in &m.data.positions {
                        lo = lo.min(v.z);
                        hi = hi.max(v.z);
                    }
                }
                log::info!(
                    "  {} z {lo:.2}..{hi:.2}",
                    t.def.path.file_name().unwrap_or_default().to_string_lossy()
                );
            }
        }
        // (the people pool and the pedestrian share, from the settings file once)
        let settings = crate::settings::Settings::load();
        Humans {
            types,
            people: Vec::new(),
            rng: 0x1234_5678_9ABC_DEF1,
            next_id: 1,
            time: 0.0,
            cabins: HashMap::new(),
            player_cabin: None,
            player_next_stop: None,
            seats: HashMap::new(),
            stops: HashMap::new(),
            odometer: HashMap::new(),
            pax_req: HashMap::new(),
            pax_busy: HashMap::new(),
            door_moving: HashMap::new(),
            door_prev: HashMap::new(),
            door_since: HashMap::new(),
            pax_places: HashMap::new(),
            desk_busy: None,
            pardons: 0,
            pardon_max: 0,
            started: false,
            ped: None,
            car_stood: HashMap::new(),
            yields: HashMap::new(),
            streams: HashMap::new(),
            hidden: Vec::new(),
            gpu_textures: HashMap::new(),
            gpu_materials: HashMap::new(),
            spare: HashMap::new(),
            served_stop: None,
            ai_visits: HashMap::new(),
            last_door_open: HashMap::new(),
            holds: Vec::new(),
            ai_requests: Vec::new(),
            tickets: None,
            request: None,
            paid: None,
            change_due: None,
            money: None,
            stop_request: false,
            tickets_sold: 0,
            ticket_cash: 0.0,
            boarded: 0,
            served: 0,
            stepped_in: 0,
            content: 0,
            ticket_requests: 0,
            ticket_points: 0,
            entry_req: Vec::new(),
            exit_req: Vec::new(),
            entry_busy: Vec::new(),
            exit_busy: Vec::new(),
            sync_frame: 0,
            footfalls: Vec::new(),
            density: 1.0,
            time_of_day: 12.0 * 3600.0,
            delay: 0.0,
            root: root.to_path_buf(),
            voice_lines: Vec::new(),
            voice_said: HashMap::new(),
            voices: 0b011,
            sit_bias: 1.0,
            seat_clock: 0.0,
            seat_moved: HashMap::new(),
            seat_gift: HashMap::new(),
            party_open: None,
            party_next: 1,
            party_plan: HashMap::new(),
            last_chat: -1e9,
            avatars: HashMap::new(),
            avatar_cmds: HashMap::new(),
            avatar_hidden: HashMap::new(),
            last_buses: Vec::new(),
            fake: HashSet::new(),
            avatar_only: false,
            driver_away: false,
            stop_targets: None,
            soon: None,
            soon_clock: f64::NEG_INFINITY,
            clock_seen: f64::NAN,
            refill_all: false,
            stop_names: None,
            ride_minutes: None,
            duty: None,
            stamped: Vec::new(),
            pedestrians: settings.pedestrians as usize,
            max_people: settings.ai_max_humans.max(1) as usize,
            stroll_timer: 0.0,
            exact_fare: true,
            boarding: "auto".into(),
            give_ticket: false,
            give_change_all: false,
            ticket_key: "T".into(),
            eye: None,
            center: DVec3::ZERO,
            message: None,
            tick_stats: (0, 0.0, 0.0),
            tick_stages: Vec::new(),
            bus_motion: HashMap::new(),
            map_humans_done: false,
            last_sync: 0.0,
            pose_stats: (0, 0, 0.0, 0.0),
            trace: omsi_cfg::env::var("OMSI_TRACE_PAX").ok().and_then(|f| std::fs::File::create(f).ok()).map(|f| {
                use std::io::Write;
                let mut w = std::io::BufWriter::new(f);
                let _ = writeln!(w, "t,id,state,ground,posed,x,y,z,heading,lx,ly,lz,rx,ry,rz,vx,vy");
                w
            }),
            tiles_seen: 0,
            mirror: false,
            lan_centers: Vec::new(),
            remote_now: Vec::new(),
            placed_now: Vec::new(),
            claims_out: Vec::new(),
            claimed: HashMap::new(),
            mirror_wait: HashMap::new(),
            comfort: RideComfort::default(),
            handed: Vec::new(),
        }
    }

    /// The map's tiles changed: a stop gone with its tile takes the people waiting there
    /// with it; a stop nobody waits at is set up again with what its tiles hold now (its
    /// waiting places come with the objects round it, sub_620c0c).
    fn tiles_changed(&mut self, world: &World) {
        let present: HashSet<i64> = world.bus_stops.lock().iter().map(|s| s.0).collect();
        let bound = |st: &State| -> Option<i64> {
            match st {
                State::Pax(p) if p.inside.is_none() => p.stop,
                _ => None,
            }
        };
        let used: HashSet<i64> = self.people.iter().filter_map(|p| bound(&p.state)).collect();
        let gone: Vec<i64> = self.stops.keys().copied().filter(|id| !present.contains(id)).collect();
        let mut removed = 0usize;
        for i in (0..self.people.len()).rev() {
            let p = &self.people[i];
            let lost_stop = bound(&p.state).is_some_and(|s| gone.contains(&s));
            let lost_ground = p.place == Place::Ground && p.puppet.is_none() && !world.has_ground(p.position.x, p.position.y);
            if lost_stop || lost_ground {
                self.release(i);
                let p = self.people.swap_remove(i);
                if debug_pax() {
                    log::info!("t={:.1} pax {} taken away with its tile ({})", self.time, p.label(), p.state.name());
                }
                self.retire(&p);
                removed += 1;
            }
        }
        for id in &gone {
            self.stops.remove(id);
        }
        let idle: Vec<i64> = self.stops.keys().copied().filter(|id| !used.contains(id)).collect();
        let rebuilt = idle.len();
        for id in idle {
            self.stops.remove(&id);
        }
        if debug_pax() || (omsi_cfg::env::var_os("OMSI_PROFILE").is_some() && (removed > 0 || !gone.is_empty())) {
            log::info!("people: tiles changed: {} stops gone, {rebuilt} set up again, {removed} people taken away", gone.len());
        }
    }

    /// The buses somebody stamped a ticket in since the last call (the app fires their
    /// `ev_Stamper` sound trigger): `None` the player's, else the AI car's id.
    pub fn take_stamped(&mut self) -> Vec<Option<u64>> {
        std::mem::take(&mut self.stamped)
            .into_iter()
            .map(|b| match b {
                BusId::Ai(id) => Some(id),
                _ => None,
            })
            .collect()
    }

    /// Lines passengers said since the last call (the app plays them where they stand).
    pub fn take_voice_lines(&mut self) -> Vec<VoiceLine> {
        std::mem::take(&mut self.voice_lines)
    }


    /// `limited`: said only when the same file has not been said for 10 s (greetings and
    /// complaints; the ticket asked for, "thanks" and the missing change always are).
    fn say_ex(&mut self, i: usize, name: &str, limited: bool) {
        // the player may have silenced them (settings), all but the ticket they ask for
        let kind = if name.starts_with("Hello_")
            || name.starts_with("GoodMorning_")
            || name.starts_with("GoodEvening_")
        {
            1
        } else if COMPLAINTS.iter().any(|p| name.starts_with(p)) {
            4
        } else {
            2
        };
        if self.voices & kind == 0 {
            return;
        }
        // Greetings and complaints: one at a time for the whole bus. OMSI only keeps
        // the same file from being said twice within 10 s, and with a dozen people
        // boarding every other one said hello - the saloon never stopped talking, which
        // is not how the original sounds: a few words now and then.
        if limited && self.time - self.last_chat < CHAT_PAUSE && self.time >= self.last_chat {
            return;
        }
        // (without a `[voicepath]` the pack's own folder: Berlin_1 and Berlin_86 carry the
        // voices themselves and name no path; the later packs point at theirs)
        let Some(base) = self.tickets.as_ref().and_then(|t| match &t.voice_path {
            Some(vp) if !vp.trim().is_empty() => Some(omsi_cfg::resolve_path(&self.root, vp.trim())),
            _ => t.path.parent().map(|p| p.to_path_buf()),
        }) else {
            return;
        };
        let voice = self.people[i].ty.def.voice.trim().to_string();
        if voice.is_empty() {
            return;
        }
        let dir = omsi_cfg::resolve_path(&base, &voice);
        let path = omsi_cfg::resolve_path(&dir, &format!("{name}.wav"));
        if !omsi_cfg::vfs::is_file(&path) {
            return;
        }
        if limited {
            if let Some(&t) = self.voice_said.get(&path) {
                if self.time - t < 10.0 && self.time >= t {
                    return;
                }
            }
        }
        self.voice_said.insert(path.clone(), self.time);
        if limited {
            self.last_chat = self.time;
        }
        if debug_pax() {
            log::info!("t={:.1} pax {} says {name}", self.time, self.people[i].label());
        }
        self.voice_lines.push(VoiceLine { position: self.people[i].position + DVec3::new(0.0, 0.0, 1.6), path });
    }


    fn rand(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn rand_f(&mut self) -> f64 {
        (self.rand() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// [ROLLBACK r5-P6] A standard normal deviate (Box-Muller on two draws).
    fn rand_normal(&mut self) -> f64 {
        let u1 = self.rand_f().max(1e-12);
        let u2 = self.rand_f();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// [ROLLBACK strollers-2] May somebody be put down at `p` without the player seeing it:
    /// out of the view (`seen`), or far enough off that a figure is a few pixels (over
    /// `APPEAR_IN_VIEW_BEYOND`), or behind a building (a ray from the eye to the person's
    /// head blocked by a collision box, as the traffic asks for its cars). Until now only
    /// the view cone counted, with no building ever hiding anybody: while the bus drove,
    /// nobody could appear on the street ahead, and people were put down only beside and
    /// behind it, where walkers never catch a bus up.
    fn may_appear(&self, world: &World, p: DVec3) -> bool {
        if !self.seen(p) {
            return true;
        }
        let Some(e) = self.eye else { return false };
        let d = (p - e.pos).length();
        if d > APPEAR_IN_VIEW_BEYOND {
            return true;
        }
        if d < 25.0 {
            return false;
        }
        world.collision.lock().ray_blocker(e.pos, p + DVec3::Z * 1.2, 2.5, 2.5).is_some()
    }

    /// Whether the player could see somebody standing at `p`.
    fn seen(&self, p: DVec3) -> bool {
        match self.eye {
            None => (p - self.center).length() < 150.0,
            Some(e) => {
                let d = p + DVec3::Z * 0.9 - e.pos;
                let dist = d.length();
                if dist > 230.0 {
                    return false;
                }
                dist < 3.0 || d.dot(e.fwd) / dist > e.cos_half
            }
        }
    }

    /// The people of OMSI's pool there are now (everybody but avatars and other players'
    /// people mirrored here).
    fn pool_used(&self) -> usize {
        self.people.iter().filter(|p| p.puppet.is_none() && !p.remote).count()
    }

    /// Room in the pool for one more person; when it is full somebody walking the street out
    /// of sight is taken for it, as Omsi.exe takes a task-8 person for a stop (0x61bd44).
    fn pool_room(&mut self) -> bool {
        if self.pool_used() < self.max_people {
            return true;
        }
        let free = (0..self.people.len()).find(|&i| {
            let p = &self.people[i];
            p.puppet.is_none() && !p.remote && matches!(p.state, State::Strolling(_) | State::Standing) && !self.seen(p.position)
        });
        match free {
            Some(i) => {
                self.release(i);
                let p = self.people.swap_remove(i);
                self.retire(&p);
                true
            }
            None => false,
        }
    }

    /// The cabin of a vehicle with the parts coupled behind it.
    fn cabin_for(&mut self, v: &VehicleInstance) -> Option<Arc<Cabin>> {
        let parts = train_parts(v);
        let key: Vec<PathBuf> = parts.iter().map(|p| p.0.path.clone()).collect();
        if let Some(c) = self.cabins.get(&key) {
            return c.clone();
        }
        let cabin = Cabin::load_train(&parts).map(Arc::new);
        if let Some(c) = cabin.as_ref().filter(|c| c.parts.len() > 1) {
            log::info!("passenger cabin of {}: {} sections joined ({} places, {} entries, {} exits, {} path points)", v.ty.def.path.file_name().unwrap_or_default().to_string_lossy(), c.parts.len(), c.seats.len(), c.entries.len(), c.exits.len(), c.graph.points.len());
        }
        self.cabins.insert(key, cabin.clone());
        cabin
    }

    /// The footsteps taken since the last call, for the environment sounds. They pile up
    /// only between two frames; a run without audio never looks at them, so the list is
    /// dropped once it grows past a crowd's worth of steps.
    /// `OMSI_TRACE_PAX` is writing a trace.
    pub fn tracing(&self) -> bool {
        self.trace.is_some()
    }

    pub fn take_footfalls(&mut self) -> Vec<ambience::Footfall> {
        if self.footfalls.len() > 256 {
            self.footfalls.clear();
        }
        std::mem::take(&mut self.footfalls)
    }

    /// Everybody: (walking, waiting at a stop, in a bus).
    pub fn counts(&self) -> (usize, usize, usize) {
        let (mut walking, mut waiting, mut aboard) = (0, 0, 0);
        for p in &self.people {
            match (&p.place, &p.state) {
                (Place::Bus(..), _) => aboard += 1,
                (Place::Ground, State::Pax(x)) if x.inside.is_none() => waiting += 1,
                (Place::Ground, _) => walking += 1,
            }
        }
        (walking, waiting, aboard)
    }

    /// People currently in the player's bus.
    pub fn riding(&self) -> usize {
        self.people
            .iter()
            .filter(|p| p.inside(BusId::Player))
            .count()
    }

    /// People walking the footpaths of the traffic network: (lane, distance along it). The
    /// traffic gives way to them at crossings and presses the pedestrian lights' buttons
    /// for them.
    /// Everybody on foot on the ground, for the traffic to stop for: position, velocity
    /// and whether they wait at a stop (a bus pulls up right beside those).
    pub fn on_foot(&self) -> Vec<(DVec3, DVec2, bool, f32, bool, bool)> {
        self.people
            .iter()
            .filter(|p| p.place == Place::Ground)
            .map(|p| {
                let waiting = matches!(&p.state, State::Pax(x) if x.inside.is_none());
                (p.position, p.vel, waiting, p.still, p.on_crossing, p.at_kerb())
            })
            .collect()
    }

    /// `on_foot` (same people, same order) with what they do besides: their task, the
    /// footpath leg they walk (lane, distance along it, the leg's length) and their last
    /// decision at a kerb - for the traffic recording.
    pub fn records(&self) -> Vec<omsi_traffic::record::PersonRecord> {
        use std::borrow::Cow;
        self.people
            .iter()
            .filter(|p| p.place == Place::Ground)
            .map(|p| {
                let waiting = matches!(&p.state, State::Pax(x) if x.inside.is_none());
                let state: Cow<'static, str> = match &p.state {
                    State::Pax(x) => Cow::Owned(format!("{} ({})", p.state.name(), if x.inside.is_some() { "aboard" } else { "waiting" })),
                    s => Cow::Borrowed(s.name()),
                };
                let leg = match &p.state {
                    State::Strolling(walk) => walk.legs.get(walk.leg).map(|leg| (leg.lane, leg.dist(walk.s), leg.len())),
                    _ => None,
                };
                omsi_traffic::record::PersonRecord {
                    pos: p.position,
                    vel: p.vel,
                    state,
                    still: p.still,
                    on_crossing: p.on_crossing,
                    waiting_for_bus: waiting,
                    leg,
                    decision: p.last_decision.map(Cow::Borrowed),
                    pace: if matches!(p.state, State::Strolling(_)) { p.walk_pace as f32 } else { p.pace as f32 },
                    at_kerb: p.at_kerb(),
                }
            })
            .collect()
    }

    /// Who walks the footpaths, for the cars' zebra check (`perception::zebra_ahead`): the
    /// lane and distance along it of each stroller's leg. Not somebody standing at a kerb on
    /// a red light: off the carriageway, they hold nobody (one at a 3 m kerb stub held the
    /// exit of a green junction 56 s, viz4 case01 #138; x10 case14 #35715 its whole green);
    /// one standing inside a lane's footprint the cars' people check still catches.
    pub fn strollers(&self) -> Vec<(usize, f32)> {
        self.people
            .iter()
            .filter(|p| !(p.at_kerb() && p.last_decision == Some("red light")))
            .filter_map(|p| match &p.state {
                State::Strolling(walk) => walk
                    .legs
                    .get(walk.leg)
                    .map(|leg| (leg.lane, leg.dist(walk.s))),
                _ => None,
            })
            .collect()
    }

    /// Seats of the player's bus from its `[passengercabin]`, and the engine's side of the
    /// ticket printer: `GivenTicket` is -1 until the driver hands a ticket over (the stock
    /// `Ticketprinter.osc` never sets it, OMSI starts it at -1 - left at 0 the first
    /// passenger took ticket 0 without the driver doing anything).
    pub fn set_cabin(&mut self, vehicle: &mut VehicleInstance) {
        vehicle.set_engine_var("GivenTicket", -1.0);
        match self.cabin_for(vehicle) {
            Some(c) => {
                log::info!("passenger cabin: {} places ({} seats), {} entries, {} exits, {} path points, desk {:?}", c.seats.len(), c.seats.iter().filter(|s| s.seated).count(), c.entries.len(), c.exits.len(), c.graph.points.len(), c.desk.map(|d| d.0));
                if debug_pax() {
                    for (i, e) in c.entries.iter().enumerate() {
                        log::info!(
                            "  entry {i}: inside {:?} wait {:?} sells {}",
                            e.inside,
                            e.wait,
                            e.sells
                        );
                    }
                    for (i, e) in c.exits.iter().enumerate() {
                        log::info!("  exit {i}: inside {:?} wait {:?}", e.inside, e.wait);
                    }
                    for (i, s) in c.seats.iter().enumerate() {
                        log::info!(
                            "  seat {i}: pos {:?} floor {:?} rot {:.0} seated {}{}{}",
                            s.pos,
                            s.floor,
                            s.rot,
                            s.seated,
                            s.switch_var.as_ref().map(|n| format!(" switched by {n} ({:?})", vehicle.var(n))).unwrap_or_default(),
                            s.taken_var.as_ref().map(|n| format!(" occupancy into {n}")).unwrap_or_default()
                        );
                    }
                }
                self.seats.insert(BusId::Player, vec![false; c.seats.len()]);
                self.entry_req = vec![false; c.entries.len().max(1)];
                self.exit_req = vec![false; c.exits.len().max(1)];
                self.entry_busy = vec![false; c.entries.len()];
                self.exit_busy = vec![false; c.exits.len()];
                self.player_cabin = Some(c);
            }
            None => log::info!("{}: no passenger cabin", vehicle.ty.def.path.display()),
        }
    }





    fn spawn(
        &mut self,
        world: &World,
        renderer: &Renderer,
        scene: &mut Scene,
        position: DVec3,
        heading: f64,
        state: State,
    ) -> Option<usize> {
        self.spawn_as(world, renderer, scene, position, heading, state, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn spawn_as(
        &mut self,
        world: &World,
        renderer: &Renderer,
        scene: &mut Scene,
        position: DVec3,
        heading: f64,
        state: State,
        kind: Option<usize>,
    ) -> Option<usize> {
        self.use_map_humans(world);
        if self.types.is_empty() {
            return None;
        }
        // on the surface they will walk on, not on the bare terrain under a pavement
        // (they stood in the asphalt and climbed out of it when they started walking)
        let mut position = position;
        if let Some(z) = world.walk_height_near(position.x, position.y, position.z) {
            if (z - position.z).abs() < 3.0 {
                position.z = z;
            }
        }
        // Not a twin of somebody standing near: two of the same figure in the same clothes
        // side by side at a stop was the first thing one noticed. A few tries for a figure
        // nobody near wears (then at least other clothes); with few figures installed some
        // repeat anyway.
        let near: Vec<(usize, usize)> = self
            .people
            .iter()
            .filter(|q| (q.position - position).truncate().length() < 30.0)
            .map(|q| (Arc::as_ptr(&q.ty) as usize, q.variant))
            .collect();
        let mut choice: Option<(usize, usize)> = None;
        for attempt in 0..10 {
            let pick = (self.rand() % self.types.len() as u64) as usize;
            let idx = kind.map(|k| k % self.types.len()).unwrap_or(pick);
            let t = &self.types[idx];
            let tk = Arc::as_ptr(t) as usize;
            // the default clothes or one of the `.cti` variants, alike likely
            let n_var = t.variants.len() as u64 + 1;
            let v0 = (self.rand() % n_var) as usize;
            // a clothing variant nobody near wears in this figure
            let var = (0..n_var as usize).map(|k| (v0 + k) % n_var as usize).find(|v| !near.contains(&(tk, *v)));
            let figure_free = !near.iter().any(|n| n.0 == tk);
            match var {
                Some(v) if figure_free || attempt >= 6 || kind.is_some() => {
                    choice = Some((idx, v));
                    break;
                }
                Some(v) if choice.is_none() => choice = Some((idx, v)),
                None if choice.is_none() && attempt == 9 => choice = Some((idx, v0)),
                _ => {}
            }
        }
        let (idx, variant) = choice.unwrap_or((0, 0));
        let ty = self.types[idx].clone();
        let tkey = Arc::as_ptr(&ty) as usize;
        let mut meshes = Vec::new();
        for (mi, hm) in ty.meshes.iter().enumerate() {
            let key = (tkey, variant, mi);
            // somebody of this type has gone: their mesh and instance
            if let Some((id, inst)) = self.spare.get_mut(&key).and_then(|v| v.pop()) {
                self.hidden.retain(|h| *h != inst);
                renderer.set_transform(scene, inst, position, Mat4::IDENTITY);
                renderer.set_params(scene, inst, &[], true, &[]);
                meshes.push((id, inst));
                continue;
            }
            if !self.gpu_materials.contains_key(&key) {
                let dirs = ty.texture_dirs(&world.root);
                let mut mats = Vec::new();
                for (k, m) in hm.materials.iter().enumerate() {
                    // the variant's texture from its own folder first, else the default
                    let (name, first) = ty.variant_texture(&m.texture, variant);
                    let mut look: Vec<&Path> = first.into_iter().collect();
                    look.extend(dirs.iter().map(|p| p.as_path()));
                    let found = omsi_texture::find_texture(name, &look)
                        .or_else(|| omsi_texture::find_texture(&m.texture, &look));
                    if found.is_none() && !m.texture.trim().is_empty() {
                        log::warn!(
                            "human {}: texture {} not found",
                            ty.def.path.display(),
                            m.texture
                        );
                    }
                    let tex = match found {
                        Some(path) => match self.gpu_textures.get(&path) {
                            Some(t) => *t,
                            None => {
                                let t = world
                                    .textures
                                    .get_gpu_fast(&path)
                                    .map(|(img, _)| renderer.add_texture_data(scene, &img));
                                world.textures.release(&path);
                                self.gpu_textures.insert(path, t);
                                t
                            }
                        },
                        None => None,
                    };
                    let alpha = match hm.alpha.get(k).copied().unwrap_or(0) {
                        1 => AlphaMode::Test,
                        2 => AlphaMode::Blend,
                        _ => AlphaMode::Opaque,
                    };
                    mats.push(renderer.add_material(scene, tex, alpha, [1.0; 4], false));
                }
                self.gpu_materials.insert(key, mats);
            }
            let mats = self.gpu_materials[&key].clone();
            let id = renderer.add_mesh(scene, &hm.data);
            let inst = renderer.add_instance(scene, id, position, Mat4::IDENTITY, mats);
            meshes.push((id, inst));
        }
        let age = ty.def.age.map(|a| a as f32).unwrap_or(40.0);
        let id = self.next_id;
        self.next_id += 1;
        // [ROLLBACK r4c-P6] walking pace by the person, for their life (Omsi.exe draws 1.1 m/s
        // +- 0.2 for everybody, 0x625758: sub_7f08b0(0.2, 1.1)); `[walk_param]` holds the
        // stride, not a speed; [ROLLBACK r5-P4] log-normal, see `pace_of`
        let pace = pace_of(id);
        if debug_pax() {
            log::info!(
                "pax #{id} ({}) appears at ({:.1}, {:.1}, {:.1}): {}{}",
                ty.def
                    .path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                position.x,
                position.y,
                position.z,
                state.name(),
                if self.seen(position) { " IN SIGHT" } else { "" }
            );
        }
        self.people.push(Person {
            id,
            ty,
            variant,
            meshes,
            position,
            heading,
            lheading: 0.0,
            place: Place::Ground,
            vel: DVec2::ZERO,
            pace,
            walk_pace: pace,
            activity: Activity::Stand,
            anim: OmsiAnim::default(),
            state,
            t_state: 0.0,
            skins: Vec::new(),
            skin_bones: None,
            pose_changed: false,
            interior: 0.0,
            lit: 0.0,
            tilt: Mat4::IDENTITY,

            age,
            stuck: 0.0,
            still: 0.0,
            on_crossing: false,
            last_decision: None,
            step_out: None,
            car_wait: 0.0,
            detour: 0.0,
            around: None,
            want_dir: DVec2::ZERO,
            turned: PED_TURN_EVERY,
            why: "",
            skinned: false,
            since_posed: 0,
            posed_at: (position, heading),
            ankles: [Vec3::ZERO; 2],
            puppet: None,
            remote: false,
        });
        Some(self.people.len() - 1)
    }

    /// Someone has gone: hidden, and their meshes kept for the next person of the type.
    fn retire(&mut self, p: &Person) {
        let tkey = Arc::as_ptr(&p.ty) as usize;
        for (mi, m) in p.meshes.iter().enumerate() {
            self.hidden.push(m.1);
            self.spare.entry((tkey, p.variant, mi)).or_default().push(*m);
        }
    }


    /// A ticket of the pack for a passenger of `age`: those whose age
    /// range holds it, weighted by their probability - a day ticket's by the time of day
    /// as well (`day_ticket_factor`). `max_stations` plays no part in the choice.
    fn pick_ticket(&mut self, age: f32) -> Option<usize> {
        let r = self.rand_f() as f32;
        let day = day_ticket_factor(self.time_of_day);
        let t = self.tickets.as_ref()?;
        let weight = |tk: &omsi_content::tickets::Ticket| {
            if (tk.age_min as f32) > age || (tk.age_max as f32) < age {
                0.0
            } else if tk.day_ticket {
                tk.probability.max(0.0) * day
            } else {
                tk.probability.max(0.0)
            }
        };
        let total: f32 = t.tickets.iter().map(weight).sum();
        if total <= 0.0 {
            return None;
        }
        let mut x = r * total;
        for (i, tk) in t.tickets.iter().enumerate() {
            let w = weight(tk);
            if w > 0.0 && x < w {
                return Some(i);
            }
            x -= w;
        }
        None
    }



    fn free_seat(&mut self, bus: BusId, seat: usize) {
        if let Some(t) = self.seats.get_mut(&bus).and_then(|v| v.get_mut(seat)) {
            *t = false;
        }
    }



    /// Put people at the bus stops near `center`: at the start everywhere, later only at
    /// stops out of sight (the others fill with people walking up).
    pub fn populate(
        &mut self,
        world: &World,
        renderer: &Renderer,
        scene: &mut Scene,
        center: DVec3,
    ) {
        if self.avatar_only {
            return;
        }
        // whoever stands under the surface there (its tile's pavements and roads came
        // after them), or anyone standing still off it: onto it
        for p in self.people.iter_mut() {
            if matches!(p.place, Place::Ground) {
                if let Some(z) = world.walk_height_near(p.position.x, p.position.y, p.position.z) {
                    let d = z - p.position.z;
                    let still = p.vel.length() < 0.05;
                    if d.abs() < 3.0 && (d > 0.02 || (still && d.abs() > 0.02)) {
                        p.position.z = z;
                    }
                }
            }
        }
        self.populate_with(world, None, renderer, scene, center);
    }

    fn populate_with(
        &mut self,
        world: &World,
        net: Option<&Network>,
        _renderer: &Renderer,
        _scene: &mut Scene,
        center: DVec3,
    ) {
        self.center = center;
        let list: Vec<(i64, DVec3, f64, String)> = world
            .bus_stops
            .lock()
            .iter()
            .filter(|s| (s.1 - center).length() < STOP_RANGE + 100.0)
            .map(|s| (s.0, s.1, s.2, s.3.clone()))
            .collect();
        for (id, pos, rot, name) in list {
            // only once the ground under the stop is there
            if world.walk_height(pos.x, pos.y).is_none() || self.stops.contains_key(&id) {
                continue;
            }
            let st = self.build_pax_stop(world, net, id, pos, rot, &name);
            self.stops.insert(id, st);
        }
        // [round 6 B3] the stops set up before the network was there: their places cleared
        // of the street lanes now, and whoever is at a place that moved walks to it
        if let Some(net) = net {
            let mut ids: Vec<i64> = self.stops.iter().filter(|(_, s)| !s.cleared).map(|(&id, _)| id).collect();
            ids.sort_unstable();
            for id in ids {
                let moved = clear_stop_of_lanes(self.stops.get_mut(&id).unwrap(), net);
                if moved.is_empty() {
                    continue;
                }
                let spots: Vec<DVec3> = self.stops[&id].spots.iter().map(|sp| sp.pos).collect();
                for p in self.people.iter_mut() {
                    if let State::Pax(x) = &mut p.state {
                        let Some(k) = x.spot.filter(|k| moved.contains(k)) else { continue };
                        if x.stop == Some(id) && x.inside.is_none() && matches!(x.task, Task::WaitingForBus | Task::WalkingToBusstop) && x.st <= 1 {
                            // (the few steps there, as to any waiting place)
                            x.task = Task::WalkingToBusstop;
                            x.target = spots[k];
                            x.target_bus = false;
                            x.st = 1;
                        }
                    }
                }
            }
        }
        self.started = true;
    }

    /// A stop as Omsi.exe sets it up (sub_620058, sub_620c0c, sub_61c604): its waiting
    /// places are the `[passpos]` of every object near it - within 10 m to the platform's
    /// side and from 10 m behind to the stop's length ahead of it -, the gather point a
    /// metre to the kerb and a metre ahead, the destinations of the trips leaving it.
    fn build_pax_stop(&mut self, world: &World, net: Option<&Network>, id: i64, pos: DVec3, heading: f64, name: &str) -> PaxStop {
        let length = world.stop_length(id);
        let side = world.stop_side(id).round().clamp(0.0, 255.0) as u8;
        let left = LEFT_HAND.load(std::sync::atomic::Ordering::Relaxed);
        let (xmax, xmin) = (if (side == 1) != left { 0.0 } else { 10.0 }, if (side == 0) != left { 0.0 } else { -10.0 });
        let h = heading.to_radians();
        let objects = world.object_positions.lock();
        let mut spots: Vec<WaitSpot> = Vec::new();
        for (obj, p, face, height) in world.waiting_places.lock().iter() {
            let Some((opos, _)) = objects.get(obj) else {
                if debug_pax() && (*p - pos).length() < 30.0 {
                    log::info!("stop {id}: waiting place of object {obj} at {p:?}: object position unknown");
                }
                continue;
            };
            // (sub_7f0db8 / sub_7f0d3c: the stop less the object)
            let v = pos - *opos;
            // (Omsi.exe looks at every object of the stop's tile and the ones round it; the
            // region below reaches the stop's length ahead, which a long bus station stop
            // takes past 40 m)
            if v.length() > 40.0_f64.max(length as f64 + 15.0) {
                continue;
            }
            // (0x7efb08 with -heading: in the stop's frame)
            let lat = h.cos() * v.x - h.sin() * v.y;
            let along = h.cos() * v.y + h.sin() * v.x;
            if !(-along < 10.0 && -along > -(length as f64).max(10.0) && -lat < xmax && -lat > xmin) {
                if debug_pax() && v.length() < 30.0 {
                    log::info!("stop {id}: object {obj} (waiting place {p:?}) not the stop's: across {:.1}, along {:.1}", -lat, -along);
                }
                continue;
            }
            spots.push(WaitSpot { pos: *p, face: *face, height: *height });
        }
        drop(objects);
        // the gather point (+0x48): (1, 0, 1) or (-1, 0, 1) through the stop's turn
        let x = if (side == 1) == left { 1.0 } else { -1.0 };
        let (fwd, right) = (DVec2::new(h.sin(), h.cos()), DVec2::new(h.cos(), -h.sin()));
        let g = pos.truncate() + right * x + fwd * 1.0;
        let gather = DVec3::new(g.x, g.y, pos.z);
        let lane = net.and_then(|n| self.ped.as_ref().and_then(|pn| pn.nearest(n, pos, 12.0))).map(|(l, s, _)| (l, s));
        let (enter_max, enter_min) = world.stop_enter(id);
        // the destinations: the stops the trips from here go on to, as likely as people
        // get off there; each with the termini of the buses that go there
        let lines: Vec<(String, HashSet<String>)> = self.stop_targets.as_ref().and_then(|m| m.get(&id)).cloned().unwrap_or_default();
        // [ROLLBACK paxweight-122] The weight of a destination is the exit number of the stop
        // object(s) it names - by the object's label or by its name in the timetable (the two
        // can differ, then the stop was not found and got the default 0.5: a technical stop
        // with 0 exiting passengers was chosen like any other). Of several objects of the one
        // name (the two sides of a stop) the largest counts; 0 stays 0.
        let weights: Vec<f32> = lines
            .iter()
            .map(|(n, _)| {
                let n = n.trim();
                let stops = world.bus_stops.lock();
                let mut best: Option<f32> = None;
                for st in stops.iter() {
                    let by_name = self.stop_names.as_ref().and_then(|m| m.get(&st.0)).is_some_and(|a| a.trim() == n);
                    if by_name || st.3.trim() == n {
                        let w = world.stop_exit_weight(st.0);
                        best = Some(best.map_or(w, |b| b.max(w)));
                    }
                }
                best.unwrap_or(0.5)
            })
            .collect();
        let total: f32 = weights.iter().sum();
        let dests: Vec<(String, f32)> = if total > 0.0 {
            lines.iter().zip(&weights).map(|((n, _), w)| (n.clone(), w / total)).collect()
        } else {
            Vec::new()
        };
        if debug_pax() {
            log::info!("stop {id} '{name}' at ({:.1}, {:.1}, {:.2}) heading {heading:.0}: {} waiting places, length {length}, side {side}, {} destinations", pos.x, pos.y, pos.z, spots.len(), dests.len());
        }
        let n = spots.len();
        // what the timetable calls it - its id when the timetable does not know it, as the
        // targets then do
        let alias = match &self.stop_names {
            Some(n) => n.get(&id).cloned().unwrap_or_else(|| id.to_string()),
            None => String::new(),
        };
        let mut stop = PaxStop {
            name: name.to_string(),
            alias,
            pos,
            heading,
            gather,
            spots,
            taken: vec![false; n],
            enter_max,
            enter_min,
            length,
            lane,
            was_near: false,
            near: false,
            clock_ms: 0.0,
            want: 0,
            factor: 1.0,
            buses: Vec::new(),
            dests,
            lines,
            cleared: false,
            kerb_shift: DVec3::ZERO,
        };
        if let Some(net) = net {
            clear_stop_of_lanes(&mut stop, net);
        }
        stop
    }

    /// The stops near the player fill with people (sub_61bf94, every frame): a stop coming
    /// into range gets its people at once, one in range another one every 10..15 s while
    /// it has fewer than it should - its pass_enter mean times its own random factor
    /// times the passenger density, at most one per waiting place. A stop going out of
    /// range loses the people waiting there.
    fn stops_tick(&mut self, dt: f32, world: &World, renderer: &Renderer, scene: &mut Scene) {
        if self.mirror || self.avatar_only {
            return;
        }
        let mut ids: Vec<i64> = self.stops.keys().copied().collect();
        ids.sort_unstable();
        let forced = omsi_cfg::env::var("OMSI_PAX_WAITING").ok().and_then(|v| v.parse::<usize>().ok());
        // the people handed over to another player's bus stop counting once it has left
        // their stop (or the session)
        if !self.handed.is_empty() {
            let (stops, remote) = (&self.stops, &self.remote_now);
            self.handed.retain(|(stop, bus)| {
                let Some(s) = stops.get(stop) else { return false };
                remote.iter().any(|b| b.id == BusId::Ai(*bus) && (b.pos - s.pos).length() < 60.0)
            });
        }
        // [ROLLBACK reshuffle-91] after a new time was set: every stop is filled up at once
        let burst = std::mem::take(&mut self.refill_all);
        for id in ids {
            let center = self.center;
            let near = {
                let s = &self.stops[&id];
                (s.pos - center).length() < STOP_RANGE || self.lan_centers.iter().any(|c| (s.pos - *c).length() < STOP_RANGE)
            };
            let changed = {
                let s = self.stops.get_mut(&id).unwrap();
                s.was_near = s.near;
                s.near = near;
                s.was_near != s.near
            };
            if !near {
                if changed {
                    // (sub_61be80) the people waiting there go
                    for i in (0..self.people.len()).rev() {
                        let here = matches!(&self.people[i].state, State::Pax(p) if p.stop == Some(id) && p.inside.is_none() && matches!(p.task, Task::WaitingForBus | Task::WalkingToBusstop));
                        if here {
                            self.release(i);
                            let p = self.people.swap_remove(i);
                            self.retire(&p);
                        }
                    }
                    for t in self.stops.get_mut(&id).unwrap().taken.iter_mut() {
                        *t = false;
                    }
                }
                continue;
            }
            if changed {
                let r = self.rand_f() as f32;
                let s = self.stops.get_mut(&id).unwrap();
                let mean = (s.enter_max + s.enter_min) / 2.0;
                let k = if s.enter_max == 0.0 {
                    0.0
                } else if mean == 0.0 {
                    (s.enter_max - s.enter_min) / (s.enter_max * 2.0)
                } else {
                    (s.enter_max - s.enter_min) / (mean * 2.0)
                };
                s.factor = (r * 2.0 - 1.0) * k + 1.0;
            }
            let count = self.people.iter().filter(|p| matches!(&p.state, State::Pax(x) if x.stop == Some(id))).count() + self.handed.iter().filter(|h| h.0 == id).count();
            let want = {
                let s = &self.stops[&id];
                let mean = (s.enter_max + s.enter_min) / 2.0;
                // (0x61bf94: with a timetable, times the share of the trips due there - at a
                // stop no trip leaves from, nobody)
                let served = if (self.stop_targets.is_some() && s.lines.is_empty()) || (!s.lines.is_empty() && s.dests.is_empty()) || self.nothing_soon(id) { 0.0 } else { 1.0 };
                let w = (self.density.max(0.0) * mean * s.factor * served).round().max(0.0) as usize;
                forced.unwrap_or(w).min(s.spots.len())
            };
            let s = self.stops.get_mut(&id).unwrap();
            s.want = want;
            s.clock_ms += dt * 1000.0;
            if !changed && !burst {
                let r = self.rand_f() as f32;
                if self.stops[&id].clock_ms <= r * 5000.0 + 10000.0 {
                    continue;
                }
            }
            let mut count = count;
            while count < want {
                if self.spawn_waiting(world, renderer, scene, id).is_none() {
                    break;
                }
                count += 1;
                if !changed && !burst {
                    break;
                }
            }
            if count >= want {
                self.stops.get_mut(&id).unwrap().clock_ms = 0.0;
            }
        }
    }

    /// A destination drawn from stop `id`'s (sub_61baa8): by weight; none when the weights
    /// leave the draw over. Also the stop's line record it matched.
    /// [ROLLBACK reshuffle-91] The time was set to another: the people waiting at the stops for
    /// a destination no bus leaves for any more (now, with `soon` renewed) go, and the stops
    /// are filled up with people for the buses that do leave. Those already boarding stay.
    pub fn reshuffle_waiting(&mut self) {
        let gone: Vec<usize> = (0..self.people.len())
            .rev()
            .filter(|&i| {
                let p = &self.people[i];
                if p.remote {
                    return false;
                }
                match (&p.state, &self.soon) {
                    (State::Pax(x), Some(m)) => {
                        x.inside.is_none()
                            && matches!(x.task, Task::WaitingForBus | Task::WalkingToBusstop)
                            && match (x.stop, x.dest.as_ref()) {
                                (Some(s), Some(d)) => !m.get(&s).is_some_and(|set| set.contains(d)),
                                _ => false,
                            }
                    }
                    _ => false,
                }
            })
            .collect();
        for i in gone {
            self.release(i);
            let p = self.people.swap_remove(i);
            self.retire(&p);
        }
        self.refill_all = true;
    }

    /// [ROLLBACK soon-90] Whether stop `id` has destinations but no bus leaves for any of them
    /// in the next minutes (see `soon`): nobody appears there.
    fn nothing_soon(&self, id: i64) -> bool {
        match (&self.soon, self.stops.get(&id)) {
            (Some(m), Some(s)) => !s.dests.is_empty() && !s.dests.iter().any(|(n, w)| *w > 0.0 && m.get(&id).is_some_and(|x| x.contains(n))),
            _ => false,
        }
    }

    fn draw_dest(&mut self, id: i64) -> (Option<String>, Option<usize>) {
        let mut r = self.rand_f() as f32;
        let mut dest: Option<String> = None;
        let Some(stop) = self.stops.get(&id) else { return (None, None) };
        // [ROLLBACK soon-90] only the destinations a bus is about to leave for, by their weights
        // [ROLLBACK paxsoon-124] never one that is this very stop, and with a timetable only those
        // a bus leaves for soon (no falling back to all of them: that waited for the night line)
        let soon_here = self.soon.as_ref().and_then(|m| m.get(&id));
        let timetable = self.soon.is_some() || self.stop_targets.is_some();
        let dests: Vec<(&String, f32)> = stop
            .dests
            .iter()
            .filter(|(n, _)| {
                !stop.is_named(n) && (!timetable || soon_here.is_some_and(|s| s.contains(n)))
            })
            .map(|(n, w)| (n, *w))
            .collect();
        // [ROLLBACK paxweight-122] by the weights alone: a destination of weight 0 is never drawn,
        // and the draw never "leaves over" (a person without a destination boards any bus)
        let total: f32 = dests.iter().map(|d| d.1.max(0.0)).sum();
        if total > 0.0 {
            r *= total;
            for (n, w) in dests.iter().filter(|d| d.1 > 0.0) {
                r -= w;
                dest = Some((*n).clone());
                if r <= 0.0 {
                    break;
                }
            }
        }
        let line = dest.as_ref().and_then(|d| stop.lines.iter().position(|(n, _)| n.trim() == d.trim()));
        (dest, line)
    }

    /// A person put at a free waiting place of stop `id` (sub_626044) with a destination
    /// drawn from the stop's (sub_61baa8); they settle there as task 6 does.
    fn spawn_waiting(&mut self, world: &World, renderer: &Renderer, scene: &mut Scene, id: i64) -> Option<usize> {
        if self.stops.get(&id)?.taken.iter().all(|t| *t) || !self.pool_room() || self.nothing_soon(id) {
            return None;
        }
        let k = self.take_spot(id)?;
        let sp = self.stops[&id].spots[k].clone();
        // [ROLLBACK party-70] some wait in twos to fours: one destination, one ride, together
        let mut party = (0u32, 0u8, 0.0f32);
        let (dest, line) = match self.party_open.take() {
            Some((st, pid, d, l, ride, n, left)) if st == id && left > 0 => {
                party = (pid, n, ride);
                if left > 1 {
                    self.party_open = Some((st, pid, d.clone(), l, ride, n, left - 1));
                }
                (d, l)
            }
            _ => {
                let (d, l) = self.draw_dest(id);
                // [ROLLBACK paxweight-122] a stop with a line record has nobody waiting whose
                // draw gave no destination (all its destinations take 0 passengers)
                if d.is_none() && self.stops.get(&id).is_some_and(|s| !s.lines.is_empty()) {
                    self.free_spot(id, k);
                    return None;
                }
                if self.sit_bias > 0.0 && (self.rand_f() as f32) < PARTY_CHANCE {
                    let r = self.rand_f();
                    let n: u8 = if r < 0.6 { 2 } else if r < 0.88 { 3 } else { 4 };
                    let pid = self.party_next;
                    self.party_next += 1;
                    let ride = self.rand_f() as f32 * 19.0 + 1.0;
                    party = (pid, n, ride);
                    self.party_open = Some((id, pid, d.clone(), l, ride, n, n - 1));
                }
                (d, l)
            }
        };
        let walk = 1.1 + (self.rand_f() as f32 * 2.0 - 1.0) * 0.2;
        let mut pax = Pax::new(walk, self.rand_f());
        pax.party = party.0;
        pax.party_n = party.1;
        pax.stop = Some(id);
        pax.spot = Some(k);
        pax.pos = sp.pos;
        pax.yaw = sp.face.to_radians();
        pax.dest = dest;
        pax.line = line;
        pax.st = 0;
        let Some(i) = self.spawn(world, renderer, scene, sp.pos, sp.face, State::Pax(Box::new(pax))) else {
            self.free_spot(id, k);
            return None;
        };
        let dummy_b: Vec<BusNow> = Vec::new();
        let dummy_ix: HashMap<BusId, usize> = HashMap::new();
        self.set_task(i, Task::WalkingToBusstop, &dummy_b, &dummy_ix, world);
        if party.0 != 0 {
            if let Some(p) = self.pax_mut(i) {
                p.ride_km = party.2;
            }
        }
        // [ROLLBACK ridemin-71] the ride by the timetable
        let from = self.stops.get(&id).map(|s| if s.alias.is_empty() { s.name.trim().to_string() } else { s.alias.trim().to_string() });
        let mins = match (from, self.pax(i).and_then(|p| p.dest.clone()), self.ride_minutes.as_ref()) {
            (Some(f), Some(d), Some(t)) => t.get(&(f, d.trim().to_string())).copied(),
            _ => None,
        };
        if let (Some(m), Some(p)) = (mins, self.pax_mut(i)) {
            p.ride_min = m;
        }
        if debug_pax() {
            let d = self.pax(i).and_then(|p| p.dest.clone());
            log::info!("t={:.1} pax {} waits at stop {id} place {k}, for {:?}", self.time, self.people[i].label(), d);
        }
        Some(i)
    }

    /// Keep strollers on the pavements near the player, and people walking up to the stops.
    fn populate_on_foot(
        &mut self,
        world: &World,
        net: &Network,
        lights: Option<&dyn PedLights>,
        renderer: &Renderer,
        scene: &mut Scene,
        dt: f32,
    ) {
        let Some(mut ped) = self.ped.take() else { return };
        self.populate_on_foot_with(&mut ped, world, net, lights, renderer, scene, dt);
        self.ped = Some(ped);
    }

    #[allow(clippy::too_many_arguments)]
    fn populate_on_foot_with(
        &mut self,
        ped: &mut PedNet,
        world: &World,
        net: &Network,
        lights: Option<&dyn PedLights>,
        renderer: &Renderer,
        scene: &mut Scene,
        _dt: f32,
    ) {
        let center = self.center;
        let radius = STROLL_RADIUS * 0.9;
        // strollers: as many as the `pedestrians` setting asks for, times the hour's
        // passenger curve, scaled down (never up) where there is little pavement around,
        // and never past the people pool (`ai_max_humans`; OMSI lets at most half of its
        // pool walk, 0x62463c, which the setting's default of 100 for a pool of 200 keeps).
        // [ROLLBACK strollers-2] The pavements are those that pass within the radius (found
        // by the pavement grid), not those that start within it: a long pavement spline
        // along a boulevard that began 250 m back was no candidate at all, and only the
        // junction objects with their hundreds of short pieces were - so a depot or a big
        // junction filled up and the streets between them stayed empty. How much pavement
        // there is, is its metres inside the circle, not its number of pieces.
        let cell = |v: f64| (v / 50.0).floor() as i32;
        let (cx, cy) = (cell(center.x), cell(center.y));
        let reach = (radius / 50.0).ceil() as i32 + 1;
        let mut lanes: Vec<usize> = Vec::new();
        for gx in (cx - reach)..=(cx + reach) {
            for gy in (cy - reach)..=(cy + reach) {
                if let Some(v) = ped.grid.get(&(gx, gy)) {
                    lanes.extend(v.iter().copied());
                }
            }
        }
        lanes.sort_unstable();
        lanes.dedup();
        // per candidate: the metres of it inside the circle (as a weight) and the point
        // indices inside, to place somebody on that part
        let inside = |l: usize| -> (f32, Vec<usize>) {
            let lane = &net.lanes[l];
            let mut metres = 0.0f32;
            let mut ks = Vec::new();
            for k in 0..lane.points.len() {
                if (lane.points[k] - center).truncate().length() < radius {
                    ks.push(k);
                    if k + 1 < lane.points.len() {
                        metres += lane.dist[k + 1] - lane.dist[k];
                    }
                }
            }
            (metres, ks)
        };
        let ahead = self.eye.map(|e| DVec3::new(e.fwd.x, e.fwd.y, 0.0).normalize_or_zero());
        let candidates: Vec<(usize, f32, Vec<usize>)> = lanes
            .iter()
            .copied()
            .filter(|&l| stroll_candidate(ped, net, lights, l))
            .filter_map(|l| {
                let (metres, ks) = inside(l);
                (metres > 0.5 && !ks.is_empty()).then_some((l, metres, ks))
            })
            .collect();
        let metres: f32 = candidates.iter().map(|c| c.1).sum();
        // (700 m of pavement in the circle - a street with a pavement on each side - is the
        // full measure)
        let pavement = (metres / 700.0).clamp(0.3, 1.0);
        let target = (self.pedestrians as f32 * self.density.clamp(0.0, 3.0) * pavement).round() as usize;
        // (never past the pool)
        let strolling = self.people.iter().filter(|p| matches!(p.state, State::Strolling(_))).count();
        let target = target.min((self.max_people + strolling).saturating_sub(self.pool_used()));
        // (around this player only: whoever walked off behind the bus no longer counts
        // against the people wanted about it)
        let have = self
            .people
            .iter()
            .filter(|p| matches!(p.state, State::Strolling(_)) && (p.position - center).length() < STROLL_RADIUS)
            .count();
        let (mut placed, mut rej_seen, mut rej_near, mut rej_ground, mut rej_spawn, mut rej_gap) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        if have < target && !candidates.is_empty() {
            // weighted by the metres inside, and by lying ahead of the player: somebody put
            // down beside or behind a moving bus never gets in front of it, and the street
            // ahead looked empty until the bus stopped
            let weights: Vec<f32> = candidates
                .iter()
                .map(|(l, m, ks)| {
                    let mid = net.lanes[*l].points[ks[ks.len() / 2]];
                    let lead = ahead
                        .map(|a| ((mid - center).truncate().normalize_or_zero().extend(0.0)).dot(a).max(0.0) as f32)
                        .unwrap_or(0.0);
                    m * (1.0 + 2.0 * lead)
                })
                .collect();
            let total: f32 = weights.iter().sum();
            // (12 a second at most, the members of a group counted)
            let cap = (target - have).min(12);
            for _ in 0..((target - have) * 3).min(36) {
                if placed >= cap {
                    break;
                }
                let mut x = self.rand_f() as f32 * total;
                let mut ci = candidates.len() - 1;
                for (k, w) in weights.iter().enumerate() {
                    if x < *w {
                        ci = k;
                        break;
                    }
                    x -= *w;
                }
                let (lane, _, ks) = &candidates[ci];
                let lane = *lane;
                let l = &net.lanes[lane];
                let k = ks[(self.rand() as usize) % ks.len()];
                let seg = if k + 1 < l.points.len() { l.dist[k + 1] - l.dist[k] } else { 0.0 };
                let len = l.length();
                let s = (l.dist[k] + self.rand_f() as f32 * seg).clamp(0.5, (len - 0.5).max(0.5));
                let (p, h) = l.at(s);
                let d = (p - center).length();
                if d < 20.0 || d > radius {
                    rej_near += 1;
                    continue;
                }
                if !self.may_appear(world, p) {
                    rej_seen += 1;
                    continue;
                }
                if !world.has_ground(p.x, p.y) {
                    rej_ground += 1;
                    continue;
                }
                let fwd = self.rand_f() < 0.5;
                // [ROLLBACK r5-P6] spaced: not within a log-normal gap of another stroller on
                // this lane walking the same way
                let z = self.rand_normal();
                let gap = lognormal(gap_median(metres, target), PED_GAP_SIGMA, z);
                let too_near = self.people.iter().any(|q| {
                    matches!(&q.state, State::Strolling(w) if w.legs.get(w.leg).map(|l| l.lane == lane && (l.b >= l.a) == fwd).unwrap_or(false))
                        && (q.position - p).length() < gap
                });
                if too_near {
                    rej_gap += 1;
                    continue;
                }
                let heading = if fwd { h as f64 } else { h as f64 + 180.0 };
                // [ROLLBACK r5-P6] a group: the leader and its members behind, on the same
                // lane the same way, at log-normal gaps, their paces within a few per cent
                // of the leader's
                let z = self.rand_normal();
                let members = group_size(z);
                let mut at_s = s;
                let mut leader_pace: Option<f64> = None;
                for k in 0..members {
                    if placed >= cap {
                        break;
                    }
                    if k > 0 {
                        let z = self.rand_normal();
                        let off = lognormal(PED_GROUP_GAP_MEDIAN, PED_GROUP_GAP_SIGMA, z) as f32;
                        at_s = if fwd { at_s - off } else { at_s + off };
                        if at_s < 0.5 || at_s > len - 0.5 {
                            break;
                        }
                    }
                    let (p, _) = l.at(at_s);
                    if k > 0 && (!self.may_appear(world, p) || !world.has_ground(p.x, p.y)) {
                        continue;
                    }
                    let leg = if fwd {
                        Leg { lane, a: at_s, b: len }
                    } else {
                        Leg { lane, a: at_s, b: 0.0 }
                    };
                    let side = 0.3 + self.rand_f() as f32 * 0.4;
                    if let Some(i) = self.spawn(
                        world,
                        renderer,
                        scene,
                        p,
                        heading,
                        State::Strolling(PedWalk::new(vec![leg], true, side)),
                    ) {
                        self.people[i].activity = Activity::Walk;
                        placed += 1;
                        match leader_pace {
                            None => leader_pace = Some(self.people[i].pace),
                            Some(lp) => {
                                let spread = (2.0 * hash01(self.people[i].id as u64, 0x06E0_D09A) - 1.0) * PED_GROUP_PACE_SPREAD;
                                let pace = (lp * (1.0 + spread)).clamp(PED_PACE_MIN, PED_PACE_MAX);
                                self.people[i].pace = pace;
                                self.people[i].walk_pace = pace;
                            }
                        }
                    } else {
                        rej_spawn += 1;
                    }
                }
            }
        }
        if debug_pax() && (have < target || placed > 0) {
            log::info!(
                "t={:.1} strollers: {have} of {target} wanted ({} asked, curve {:.2}, pavement {:.2} of {:.0} m on {} paths), {placed} put down; refused: seen {rej_seen}, distance {rej_near}, no ground {rej_ground}, too near another {rej_gap}, no figure {rej_spawn}",
                self.time, self.pedestrians, self.density, pavement, metres, candidates.len()
            );
        }
        // OMSI_PAX_CROSS=x,y: a few pedestrians sent across the signalised crossing nearest that point
        if let Some((x, y)) = omsi_cfg::env::var("OMSI_PAX_CROSS").ok().and_then(|v| {
            let mut it = v.split(',').filter_map(|t| t.trim().parse::<f64>().ok());
            Some((it.next()?, it.next()?))
        }) {
            let want = DVec3::new(x, y, center.z);
            let placed = self
                .people
                .iter()
                .filter(|p| matches!(p.state, State::Strolling(ref w) if !w.roam || w.side < 0.0))
                .count();
            let lane = ped
                .ends
                .keys()
                .copied()
                .filter(|&l| net.lanes[l].traffic_light.is_some())
                .min_by(|a, b| {
                    (net.lanes[*a].start() - want)
                        .truncate()
                        .length()
                        .total_cmp(&(net.lanes[*b].start() - want).truncate().length())
                });
            if let (Some(cross), 0) = (lane, placed) {
                let (start_node, _) = ped.ends[&cross];
                let feeders: Vec<(usize, bool)> = ped.out[start_node]
                    .iter()
                    .copied()
                    .filter(|(l, _)| *l != cross)
                    .collect();
                log::info!(
                    "OMSI_PAX_CROSS: crossing path {cross} light {:?}, {} paths lead to it",
                    net.lanes[cross].traffic_light,
                    feeders.len()
                );
                for k in 0..6 {
                    let Some(&(lane, fwd)) = feeders.get(k % feeders.len().max(1)) else {
                        break;
                    };
                    let len = net.lanes[lane].length();
                    let back = (3.0 + k as f32 * 1.6).min(len);
                    let first = if fwd {
                        Leg {
                            lane,
                            a: back,
                            b: 0.0,
                        }
                    } else {
                        Leg {
                            lane,
                            a: len - back,
                            b: len,
                        }
                    };
                    let over = Leg {
                        lane: cross,
                        a: 0.0,
                        b: net.lanes[cross].length(),
                    };
                    let (p, h) = first.at(net, 0.0);
                    let mut walk = PedWalk::new(vec![first, over], true, 0.4);
                    // marked so that the knob spawns them once
                    walk.side = -0.4;
                    if let Some(i) =
                        self.spawn(world, renderer, scene, p, h, State::Strolling(walk))
                    {
                        self.people[i].activity = Activity::Walk;
                    }
                }
            }
        }
    }

    /// Whether the bus script reports `name`: it writes it (`PAX_*` are engine variables every
    /// vehicle has, so stock scripts set them without a varlist entry) or declares it.
    fn script_reports(v: &VehicleInstance, name: &str) -> bool {
        v.has_script_var(name) || v.ty.program.var(name).is_some_and(|id| v.ty.program.stores(id))
    }

    /// Whether entry or exit `i` (`kind` "Entry" or "Exit") has variables of its own: one of
    /// Omsi.exe's eight, or one past them whose `PAX_<kind><i>_Open` the script reports (#719).
    /// The others open with the eighth, as in Omsi.exe (0x62d31e reads the open state of door
    /// min(i, 7)), and ask through its `_Req` ([`Humans::write_door_requests`]).
    fn own_pax_door(v: &VehicleInstance, kind: &str, i: usize) -> bool {
        i < OMSI_PAX_DOORS || (i < omsi_sim::vehicle::PAX_DOORS && Self::script_reports(v, &format!("PAX_{kind}{i}_Open")))
    }

    /// The passengers' requests into the bus's `PAX_Entry<n>_Req` / `PAX_Exit<n>_Req`, and
    /// who stands in its doorways into their `_Busy` (#720): an entry or exit without
    /// variables of its own asks through the eighth's ([`Humans::own_pax_door`]), where
    /// Omsi.exe drops them (its request arrays have eight slots). Past the eighth they were
    /// written to no variable at all.
    pub(crate) fn write_door_requests(v: &mut VehicleInstance, doors: &DoorWants) {
        for (kind, flags, what) in [
            ("Entry", &doors.entry_req, "Req"),
            ("Exit", &doors.exit_req, "Req"),
            ("Entry", &doors.entry_busy, "Busy"),
            ("Exit", &doors.exit_busy, "Busy"),
        ] {
            let mut slots = vec![false; flags.len().min(omsi_sim::vehicle::PAX_DOORS)];
            for (i, r) in flags.iter().enumerate() {
                let slot = if Self::own_pax_door(v, kind, i) { i } else { OMSI_PAX_DOORS - 1 };
                if let Some(s) = slots.get_mut(slot) {
                    *s |= *r;
                }
            }
            for (i, r) in slots.iter().enumerate() {
                v.set_var(&format!("PAX_{kind}{i}_{what}"), if *r { 1.0 } else { 0.0 });
            }
        }
        for (name, on) in &doors.places {
            v.set_var(name, if *on { 1.0 } else { 0.0 });
        }
    }

    /// `PAX_Entry<i>_Open` / `PAX_Exit<i>_Open` as the bus script reports them. A bus whose
    /// script never sets them (or only sets some of them) falls back to its physical `door_<i>`
    /// or `door<i>` animations; an entry or exit past the eighth without variables of its own
    /// is open with the eighth ([`Humans::own_pax_door`]).
    fn doors_open(v: &VehicleInstance, n_entry: usize, n_exit: usize) -> (Vec<bool>, Vec<bool>) {
        let door_val = |k: usize| -> bool {
            v.var(&format!("door_{k}"))
                .or_else(|| v.var(&format!("door{k}")))
                .unwrap_or(0.0)
                > 0.5
        };
        // Exits in standard OMSI city buses (2 or more front door leaves) begin at door_2 (middle door),
        // while coaches with a single front door leaf begin at door_1. Exits must not be offset by
        // n_entry, because buses with all doors configured as entries (e.g. 3-door buses with 6 entries)
        // still place middle-door exits at door_2/3 and rear-door exits at door_4/5.
        let exit_door_base = if n_entry <= 1 { 1 } else { 2 };
        let states = |kind: &str, n: usize, door: &dyn Fn(usize) -> bool| -> Vec<bool> {
            let mut out: Vec<bool> = Vec::with_capacity(n);
            for i in 0..n {
                let name = format!("PAX_{kind}{i}_Open");
                let open = if !Self::own_pax_door(v, kind, i) {
                    out[OMSI_PAX_DOORS - 1]
                } else if Self::script_reports(v, &name) {
                    v.var(&name).unwrap_or(0.0) > 0.5
                } else {
                    door(i)
                };
                out.push(open);
            }
            out
        };
        let entry = states("Entry", n_entry, &door_val);
        let exit = states("Exit", n_exit, &|i| door_val((exit_door_base + i).min(7)));
        (entry, exit)
    }


    /// [ROLLBACK doorwait-66] Note which doors of `v` are opening now (see `door_moving`).
    fn note_door_motion(&mut self, id: BusId, v: &VehicleInstance, entry_open: &[bool], exit_open: &[bool]) {
        let pos = |k: usize| v.var(&format!("door_{k}")).or_else(|| v.var(&format!("door{k}")));
        let exit_base = if entry_open.len() <= 1 { 1 } else { 2 };
        let mut moving = (vec![false; entry_open.len()], vec![false; exit_open.len()]);
        // [ROLLBACK doorwait-68] only the entries: whoever leaves goes to an exit when it is flagged
        // open, as before (they walked into the exit point of a door still shut)
        for (exit, flags, out) in [(false, entry_open, &mut moving.0)] {
            for k in 0..flags.len() {
                let d = if exit { (exit_base + k).min(7) } else { k.min(7) };
                if let Some(x) = pos(d) {
                    let prev = self.door_prev.insert((id, exit, k), x).unwrap_or(x);
                    let rising = !flags[k] && x > 0.02 && x >= prev - 1e-4;
                    // (and only for a few seconds: a door that never gets flagged is given up)
                    let since = *self.door_since.entry((id, exit, k)).or_insert(self.time);
                    if !rising {
                        self.door_since.remove(&(id, exit, k));
                    }
                    out[k] = rising && self.time - since < DOOR_OPENING_GRACE;
                }
            }
        }
        self.door_moving.insert(id, moving);
    }

    pub fn set_player_next_stop(&mut self, stop: Option<&crate::schedule::PlannedStop>) {
        // (called once a frame: worked out again only when the duty moves on to another stop)
        if let (Some(s), Some(cur)) = (stop, self.player_next_stop.as_ref()) {
            if cur.id == s.object_id {
                return;
            }
        }
        self.player_next_stop = stop
            .and_then(|stop| self.request_stop(stop.object_id, Some(&stop.name), stop.position));
    }

    fn request_stop(
        &self,
        id: i64,
        name: Option<&str>,
        position: Option<DVec3>,
    ) -> Option<RequestStop> {
        let loaded = self.stops.get(&id);
        let name = name
            .filter(|name| !name.trim().is_empty())
            .map(str::to_string)
            .or_else(|| loaded.map(|stop| stop.name.clone()))
            .unwrap_or_else(|| id.to_string());
        let alias = loaded
            .map(|stop| {
                if name.trim() == stop.name.trim() {
                    stop.alias.clone()
                } else {
                    stop.name.clone()
                }
            })
            .unwrap_or_default();
        Some(RequestStop {
            id,
            name,
            alias,
            pos: position.or_else(|| loaded.map(|stop| stop.pos))?,
        })
    }

    fn vehicle_next_stop(&self, vehicle: &VehicleInstance) -> Option<RequestStop> {
        if let Ok(index) = usize::try_from(vehicle.host.tt_busstop_index) {
            if let Some(id) = vehicle.host.tt_stop_ids.get(index) {
                let name = vehicle.host.tt_stops.get(index).map(|stop| stop.0.as_str());
                if let Some(stop) = self.request_stop(*id, name, None) {
                    return Some(stop);
                }
            }
        }
        let name = vehicle.str_var("act_busstop");
        if name.trim().is_empty() {
            return None;
        }
        let (id, _) = self
            .stops
            .iter()
            .filter(|(_, stop)| stop.is_named(&name))
            .min_by(|(_, a), (_, b)| {
                let back =
                    |stop: &PaxStop| crowd::angle_diff(vehicle.heading, stop.heading).abs() > 100.0;
                back(a).cmp(&back(b)).then_with(|| {
                    (a.pos - vehicle.position)
                        .length_squared()
                        .total_cmp(&(b.pos - vehicle.position).length_squared())
                })
            })?;
        self.request_stop(*id, Some(&name), None)
    }

    /// The buses passengers deal with this frame.
    fn gather_buses(
        &mut self,
        world: &World,
        bus: Option<&VehicleInstance>,
        traffic: Option<&Traffic>,
    ) -> Vec<BusNow> {
        let mut out = Vec::new();
        let stops: Vec<(i64, DVec3, f64)> =
            world.bus_stops.lock().iter().map(|s| (s.0, s.1, s.2)).collect();
        // The stop a bus serves: the nearest in reach - but one facing the way the bus goes
        // before one facing the other way. The two stops of a street often lie within
        // reach of each other, and the people of the stop across the road then walked over
        // the carriageway, through the traffic, to a bus that was not theirs.
        let serving = |pos: DVec3, heading: f64, reach: f64| -> Option<i64> {
            stops
                .iter()
                .filter(|s| (s.1 - pos).length() < reach)
                .min_by(|a, c| {
                    let back = |s: &(i64, DVec3, f64)| crowd::angle_diff(heading, s.2).abs() > 100.0;
                    back(a)
                        .cmp(&back(c))
                        .then((a.1 - pos).length().total_cmp(&(c.1 - pos).length()))
                })
                .map(|s| s.0)
        };
        let bb_of = |v: &VehicleInstance| {
            let bb =
                v.ty.def
                    .bounding_box
                    .unwrap_or([2.5, 11.0, 3.0, 0.0, 0.0, 1.5]);
            (
                DVec2::new(bb[0] as f64 * 0.5, bb[1] as f64 * 0.5),
                DVec2::new(bb[3] as f64, bb[4] as f64),
            )
        };
        if let (Some(b), Some(cabin)) = (bus, self.player_cabin.clone()) {
            let speed = b.physics.velocity_kmh() as f64 / 3.6;
            let (entry_open, exit_open) =
                Self::doors_open(b, cabin.entries.len(), cabin.exits.len());
            self.note_door_motion(BusId::Player, b, &entry_open, &exit_open);
            let (half, centre) = bb_of(b);
            let trailers = part_frames(b, &cabin);
            // [ROLLBACK boarding-88] the sign's terminus: the scripts' `target_index_int` row;
            // a bus without that variable (an IBIS bus whose sign takes its row from other
            // variables) by the IBIS's terminus, else - on a duty - by its trip's: it read
            // "shows nothing" there, so it was out of service and took nobody
            let duty_terminus = self.duty.as_ref().filter(|(trip, ..)| trip.public).map(|(trip, ..)| trip.terminus.trim().to_string());
            let terminus = match (b.var("target_index_int"), b.host.hof.as_ref()) {
                (Some(i), Some(hof)) if i.is_finite() && i >= 0.0 => hof
                    .termini
                    .get(i.round() as usize)
                    .filter(|t| !t.all_exit)
                    .map(|t| t.texture_id.trim().to_string()),
                (_, hof) => {
                    let ibis = hof.and_then(|hof| {
                        let code = b.var("IBIS_TerminusCode").unwrap_or(-1.0) as i32;
                        let at = b.var("IBIS_TerminusIndex").filter(|i| i.is_finite() && *i >= 0.0).map(|i| i.round() as usize);
                        at.filter(|&k| hof.termini.get(k).is_some_and(|t| t.code == code))
                            .or_else(|| hof.termini.iter().position(|t| t.code == code && code > 0))
                            .and_then(|k| hof.termini.get(k))
                            .filter(|t| !t.all_exit)
                            .map(|t| t.texture_id.trim().to_string())
                    });
                    ibis.filter(|s| !s.is_empty()).or(duty_terminus)
                }
            };
            // (on a duty the people its trip takes where they are going get on, however the
            // depot file names the terminus; in free drive nobody)
            let takes = match &self.duty {
                Some((trip, next, done)) => Takes::Duty { trip: trip.clone(), next: *next, done: *done },
                // [ROLLBACK boarding-87] free drive: those whose line record lists the shown
                // destination get on (Omsi.exe's test); a bus showing none still takes nobody
                None => Takes::Terminus,
            };
            out.push(BusNow {
                terminus,
                takes,
                id: BusId::Player,
                next_stop: self
                    .player_next_stop
                    .clone()
                    .or_else(|| self.vehicle_next_stop(b)),
                walk_open: None,
                places_off: places_off(b, &cabin),
                cabin,
                pos: b.position,
                rot: b.body_rotation(),
                heading: b.heading,
                speed,
                entry_open,
                exit_open,
                interior: b.interior_light(),
                air: CabinAir::of(b),
                half,
                centre,
                accel: DVec2::ZERO,
                trailers,
            });
        }
        if let Some(t) = traffic {
            let near = self.center;
            let riding: HashSet<u64> = self
                .people
                .iter()
                .filter_map(|p| match p.state.bus() {
                    Some(BusId::Ai(id)) => Some(id),
                    _ => None,
                })
                .collect();
            let mut visits = HashMap::new();
            for c in t.cars.iter().filter(|c| c.serves_stops()) {
                let from_eye = self.eye.map(|e| (c.vehicle.position - e.pos).length()).unwrap_or(f64::MAX);
                if (c.vehicle.position - near).length().min(from_eye) > 400.0 && !riding.contains(&c.id) {
                    continue;
                }
                let Some(cabin) = self.cabin_for(&c.vehicle) else {
                    continue;
                };
                let speed = c.ai.state.speed as f64;
                let stop = if c.at_station() && speed.abs() < 0.3 {
                    serving(c.vehicle.position, c.vehicle.heading, 18.0)
                } else {
                    None
                };
                let since = match stop {
                    Some(s) => {
                        let v = match self.ai_visits.get(&c.id) {
                            Some(&(vs, t0)) if vs == s => (vs, t0),
                            _ => (s, self.time),
                        };
                        visits.insert(c.id, v);
                        self.time - v.1
                    }
                    None => 0.0,
                };
                let open = stop.is_some();
                let (mut entry_open, mut exit_open) = (
                    vec![false; cabin.entries.len()],
                    vec![false; cabin.exits.len()],
                );
                if open {
                    if Self::script_reports(&c.vehicle, "PAX_Entry0_Open") || c.vehicle.var("door_0").is_some() || c.vehicle.var("door0").is_some() {
                        let (e, x) =
                            Self::doors_open(&c.vehicle, cabin.entries.len(), cabin.exits.len());
                        entry_open = e;
                        exit_open = x;
                    } else if since > 2.5 {
                        // the script does not say: the doors are open while the bus boards
                        entry_open
                            .iter_mut()
                            .chain(exit_open.iter_mut())
                            .for_each(|o| *o = true);
                    }
                }
                self.seats
                    .entry(BusId::Ai(c.id))
                    .or_insert_with(|| vec![false; cabin.seats.len()]);
                self.note_door_motion(BusId::Ai(c.id), &c.vehicle, &entry_open, &exit_open);
                let (half, centre) = bb_of(&c.vehicle);
                let trailers = part_frames(&c.vehicle, &cabin);
                out.push(BusNow {
                    terminus: c.bus.as_ref().map(|b| b.terminus.trim().to_string()).filter(|t| !t.is_empty()),
                    takes: Takes::Terminus,
                    id: BusId::Ai(c.id),
                    next_stop: c
                        .bus
                        .as_ref()
                        .and_then(|bus| bus.stops.front())
                        .and_then(|stop| {
                            let position = c
                                .ai.state
                                .route
                                .get(stop.ri)
                                .and_then(|lane| t.net.lanes.get(*lane))
                                .map(|lane| lane.at(stop.s).0);
                            self.request_stop(stop.id, None, position)
                        }),
                    walk_open: None,
                    places_off: places_off(&c.vehicle, &cabin),
                    cabin,
                    pos: c.vehicle.position,
                    rot: c.vehicle.body_rotation(),
                    heading: c.vehicle.heading,
                    speed,
                    entry_open,
                    exit_open,
                    interior: c.vehicle.interior_light(),
                    air: CabinAir::of(&c.vehicle),
                    half,
                    centre,
                    accel: DVec2::ZERO,
                    trailers,
                });
            }
            self.ai_visits = visits;
            let alive: HashSet<u64> = t.cars.iter().map(|c| c.id).chain(self.remote_now.iter().chain(self.placed_now.iter()).filter_map(|b| match b.id {
                BusId::Ai(id) => Some(id),
                BusId::Player => None,
            })).collect();
            self.seats.retain(|k, _| match k {
                BusId::Ai(id) => alive.contains(id),
                BusId::Player => true,
            });
        }
        for b in self.remote_now.iter().chain(self.placed_now.iter()) {
            self.seats.entry(b.id).or_insert_with(|| vec![false; b.cabin.seats.len()]);
            out.push(b.clone());
        }
        out
    }

    /// The vehicles standing in the world that the player placed and does not drive now:
    /// (their `Player::uid`, the vehicle). Their riders stay aboard; nobody new boards them.
    pub fn set_placed_buses<'a>(&mut self, buses: impl Iterator<Item = (u64, &'a VehicleInstance)>) {
        let mut out = Vec::new();
        for (uid, v) in buses {
            if let Some(b) = self.parked_bus(BusId::Ai(placed_bus_id(uid)), v) {
                out.push(b);
            }
        }
        self.placed_now = out;
    }



    /// A bus people may be in but do not board here (another player's, one the player left).
    fn parked_bus(&mut self, id: BusId, v: &VehicleInstance) -> Option<BusNow> {
        let cabin = self.cabin_for(v)?;
        let bb = v.ty.def.bounding_box.unwrap_or([2.5, 11.0, 3.0, 0.0, 0.0, 1.5]);
        let trailers = part_frames(v, &cabin);
        let walk_open = Self::doors_open(v, cabin.entries.len(), cabin.exits.len());
        Some(BusNow {
            terminus: None,
            takes: Takes::Nobody,
            places_off: Vec::new(),
            id,
            next_stop: None,
            entry_open: vec![false; cabin.entries.len()],
            exit_open: vec![false; cabin.exits.len()],
            walk_open: Some(walk_open),
            cabin,
            pos: v.position,
            rot: v.body_rotation(),
            heading: v.heading,
            speed: v.physics.velocity_kmh() as f64 / 3.6,
            interior: v.interior_light(),
            air: CabinAir::of(v),
            half: DVec2::new(bb[0] as f64 * 0.5, bb[1] as f64 * 0.5),
            centre: DVec2::new(bb[3] as f64, bb[4] as f64),
            accel: DVec2::ZERO,
            trailers,
        })
    }

    /// The player now drives vehicle `new_uid` and left `old_uid`: whoever rode in the one
    /// left stays in it (it is one of the placed vehicles now), whoever rode in the one
    /// taken over is the player's bus's, and the player's cabin is the new vehicle's own.
    /// (Riders followed the player into the next bus, and people boarding it took the old
    /// bus's seats - places in the air round a minibus's bonnet.)
    pub fn player_bus_swapped(&mut self, old_uid: u64, new_uid: u64, new_vehicle: &mut VehicleInstance) {
        let old = BusId::Ai(placed_bus_id(old_uid));
        let new = BusId::Ai(placed_bus_id(new_uid));
        // (through a free id: Player -> old, new -> Player)
        let tmp = BusId::Ai(u64::MAX);
        self.remap_bus(BusId::Player, tmp);
        self.remap_bus(new, BusId::Player);
        self.remap_bus(tmp, old);
        let kept = self.seats.remove(&BusId::Player);
        self.player_cabin = None;
        self.served_stop = None;
        self.set_cabin(new_vehicle);
        if let (Some(k), Some(now)) = (kept, self.seats.get_mut(&BusId::Player)) {
            if k.len() == now.len() {
                *now = k;
            }
        }
    }

    /// Developer tools: the player's passenger cabin is read from its files again - the
    /// `[passengercabin]` or the paths - and put in the place of the one in use, without
    /// loading the session again. (Done between two frames; the game does not need to stop.)
    /// When the new cabin has as many places, doors, validators and path points as the old
    /// one, the riders stay where they are and walk on the new paths. Otherwise their
    /// numbers mean nothing in it, and everybody inside a bus (or on the way to one) is put
    /// out on the ground. A cabin that cannot be read leaves the old one as it is.
    pub(crate) fn reload_cabin(&mut self, what: CabinReload, vehicle: &mut VehicleInstance) -> String {
        let parts = train_parts(vehicle);
        let key: Vec<PathBuf> = parts.iter().map(|p| p.0.path.clone()).collect();
        let old = self.cabins.get(&key).cloned().flatten();
        let fresh = {
            let keep = match (&old, what) {
                (Some(o), CabinReload::Cabin) => Keep::Paths(&o.sources),
                (Some(o), CabinReload::Paths) => Keep::Cabin(&o.sources),
                (None, _) => Keep::Nothing,
            };
            Cabin::load_train_with(&parts, keep).map(Arc::new)
        };
        let name = match what {
            CabinReload::Cabin => "passenger cabin",
            CabinReload::Paths => "paths",
        };
        let Some(new) = fresh else {
            return format!("Reloading the {name}: the vehicle has no cabin that could be read (the old one stays)");
        };
        let same = old.as_ref().is_some_and(|o| {
            o.seats.len() == new.seats.len()
                && o.entries.len() == new.entries.len()
                && o.exits.len() == new.exits.len()
                && o.graph.points.len() == new.graph.points.len()
                && o.parts.len() == new.parts.len()
                && o.stampers.len() == new.stampers.len()
        });
        self.cabins.insert(key, Some(new.clone()));
        let summary = format!("{} places, {} entries, {} exits, {} path points", new.seats.len(), new.entries.len(), new.exits.len(), new.graph.points.len());
        if same {
            self.player_cabin = Some(new);
            log::info!("developer tools: {name} read again: {summary}");
            return format!("Reloaded the {name}: {summary}; the riders stay");
        }
        let mut buses: HashSet<BusId> = self.seats.keys().copied().collect();
        for p in &self.people {
            if let Place::Bus(b, _) = p.place {
                buses.insert(b);
            }
            if let State::Pax(x) = &p.state {
                buses.extend(x.bus);
                buses.extend(x.inside);
            }
        }
        let before = self.people.len();
        for b in buses {
            self.evict_bus(b);
        }
        let put_out = before - self.people.len();
        self.set_cabin(vehicle);
        log::info!("developer tools: {name} read again: {summary}; {put_out} people put out");
        format!("Reloaded the {name}: {summary}; {put_out} people put out of the buses")
    }

    /// Bus `bus` is gone (the player removed it): whoever was in it stands where they were,
    /// on the ground, and walks off.
    pub fn evict(&mut self, bus: BusId, world: &World) {
        let _ = world;
        self.evict_bus(bus);
    }

    fn evict_bus(&mut self, bus: BusId) {
        for i in (0..self.people.len()).rev() {
            let p = &self.people[i];
            let theirs = matches!(p.place, Place::Bus(b, _) if b == bus) || matches!(&p.state, State::Pax(x) if x.bus == Some(bus) || x.inside == Some(bus));
            if theirs {
                self.release(i);
                let p = self.people.swap_remove(i);
                self.retire(&p);
            }
        }
        self.seats.remove(&bus);
        self.bus_motion.remove(&bus);
        if bus == BusId::Player {
            self.player_cabin = None;
            self.served_stop = None;
        }
    }

    /// Everyone and everything that belongs to bus `from` belongs to `to` now.
    fn remap_bus(&mut self, from: BusId, to: BusId) {
        let fix = |b: &mut BusId| {
            if *b == from {
                *b = to;
            }
        };
        for p in &mut self.people {
            if let Place::Bus(b, _) = &mut p.place {
                fix(b);
            }
            if let State::Pax(x) = &mut p.state {
                if let Some(b) = x.bus.as_mut() {
                    fix(b);
                }
                if let Some(b) = x.inside.as_mut() {
                    fix(b);
                }
            }
        }
        if let Some(v) = self.seats.remove(&from) {
            self.seats.insert(to, v);
        }
        if let Some(v) = self.bus_motion.remove(&from) {
            self.bus_motion.insert(to, v);
        }
    }

    /// LAN play: the other players' buses this frame, by player id (their riders are drawn
    /// in them, see `remote_bus_id`).
    pub fn set_remote_buses<'a>(&mut self, buses: impl Iterator<Item = (u32, &'a VehicleInstance)>) {
        let mut out = Vec::new();
        for (player, v) in buses {
            let Some(cabin) = self.cabin_for(v) else { continue };
            let bb = v.ty.def.bounding_box.unwrap_or([2.5, 11.0, 3.0, 0.0, 0.0, 1.5]);
            let trailers = part_frames(v, &cabin);
            // (their doors as their game has them: a walker gets in only where one is open;
            // the passengers here never board it - that bus's own game boards them)
            let walk_open = Self::doors_open(v, cabin.entries.len(), cabin.exits.len());
            out.push(BusNow {
                terminus: None,
                takes: Takes::Nobody,
                places_off: Vec::new(),
                id: BusId::Ai(remote_bus_id(player)),
                next_stop: None,
                entry_open: vec![false; cabin.entries.len()],
                exit_open: vec![false; cabin.exits.len()],
                walk_open: Some(walk_open),
                cabin,
                pos: v.position,
                rot: v.body_rotation(),
                heading: v.heading,
                speed: v.physics.velocity_kmh() as f64 / 3.6,
                interior: v.interior_light(),
                air: CabinAir::of(v),
                half: DVec2::new(bb[0] as f64 * 0.5, bb[1] as f64 * 0.5),
                centre: DVec2::new(bb[3] as f64, bb[4] as f64),
                accel: DVec2::ZERO,
                trailers,
            });
        }
        self.remote_now = out;
    }

    /// The player's duty this frame; None in free drive, where the people waiting take the
    /// player's bus by the destination it shows. (Set after `stop_names`: the trip's stops are named by it.)
    pub fn set_duty(&mut self, duty: Option<&crate::schedule::PlayerDuty>) {
        let Some(d) = duty else {
            self.duty = None;
            return;
        };
        // the trip the IBIS is given: on a works trip from the depot the next one with a line
        let (t, next) = d.trip_for_ibis();
        let done = std::ptr::eq(t, d.trip()) && d.trip_done();
        let trip = match self.duty.take() {
            Some((trip, ..)) if trip.name == t.name && trip.departure == t.departure => trip,
            _ => Arc::new(DutyTrip::of(t, self.stop_names.as_ref())),
        };
        self.duty = Some((trip, next, done));
    }

    /// Is `bus` among the buses people can be in this frame (an AI bus, or another player's)?
    pub fn knows_bus(&self, bus: u64) -> bool {
        self.remote_now.iter().any(|b| b.id == BusId::Ai(bus)) || self.seats.contains_key(&BusId::Ai(bus))
    }

    /// Is person `id` (one drawn for another game) here?
    pub fn has_mirror(&self, id: u32) -> bool {
        self.people.iter().any(|p| p.id == id && p.remote)
    }

    /// The riders of our own bus, for the other LAN players to see: (id, type, place in the
    /// bus frame, heading there, seat, activity).
    pub fn lan_riders(&self) -> Vec<LanPerson> {
        self.people
            .iter()
            .filter(|p| p.puppet.is_none() && !p.remote)
            .filter_map(|p| match p.place {
                Place::Bus(BusId::Player, l) => Some(LanPerson {
                    id: p.id,
                    ty: p.ty.clone(),
                    pos: p.position,
                    heading: p.heading,
                    speed: 0.0,
                    activity: p.activity,
                    aboard: Some((
                        0,
                        l,
                        p.lheading,
                        match &p.state {
                            State::Pax(x) if x.task == Task::SittingInBus => x.seat,
                            _ => None,
                        },
                    )),
                    waiting: None,
                }),
                _ => None,
            })
            .collect()
    }



    /// Keep only the people the map's `humans.txt` names, an entry listed twice counting
    /// twice, as OMSI draws a map's pedestrians and passengers from that list alone. A map
    /// without the file, or whose list names nobody to be found, keeps everybody.
    fn use_map_humans(&mut self, world: &World) {
        if self.map_humans_done {
            return;
        }
        self.map_humans_done = true;
        let path = omsi_cfg::resolve_path(&world.map_dir, "humans.txt");
        let list = omsi_map::ailists::load_list(&path);
        if list.is_empty() {
            return;
        }
        // the people installed already (any content root, mods too), matched by the path
        // below `Humans/`; an entry not among them (a pack nested deeper than the scan) is
        // loaded from its own path
        let key = |p: &str| -> String {
            let p = p.replace('\\', "/").to_ascii_lowercase();
            match p.rfind("humans/") {
                Some(k) => p[k + 7..].to_string(),
                None => p,
            }
        };
        let mut picked: Vec<Arc<HumanType>> = Vec::new();
        for line in &list {
            let want = key(line.trim());
            match self.types.iter().find(|t| key(&t.def.path.to_string_lossy()) == want) {
                Some(t) => picked.push(t.clone()),
                None => picked.extend(map_human_types(&world.root, std::slice::from_ref(line))),
            }
        }
        // (a list that names nobody to be found keeps everybody: a map without people
        // looked broken)
        if picked.is_empty() {
            log::warn!("humans.txt of the map names nobody installed: keeping all people");
            return;
        }
        log::info!(
            "humans: {} of {} map entries loaded from {}",
            picked.len(),
            list.len(),
            path.display()
        );
        self.types = picked;
    }

    /// People the moving bus has just knocked down. OMSI counts them in the driver's
    /// personnel file; they are only counted once and then walk away.
    pub fn run_over(&mut self, bus: &VehicleInstance) -> u32 {
        if bus.physics.velocity_kmh().abs() < 5.0 {
            return 0;
        }
        let Some(bb) = bus.ty.def.bounding_box else {
            return 0;
        };
        let (half_x, half_y) = ((bb[0] - bb[3]).abs() / 2.0, (bb[1] - bb[4]).abs() / 2.0);
        let inv = bus.body_rotation().transpose();
        let mut knocked = Vec::new();
        for (i, p) in self.people.iter().enumerate() {
            // (sub_62a6a0 at 0x62dc6c: the people waiting at a stop and those on the pavements)
            let counts = match &p.state {
                State::Pax(x) => x.task == Task::WaitingForBus,
                State::Strolling(_) | State::Standing => true,
                State::Idle => false,
            };
            if p.place != Place::Ground || !counts {
                continue;
            }
            let local = inv.transform_vector3((p.position - bus.position).as_vec3());
            if local.x.abs() < half_x + 0.2 && local.y.abs() < half_y + 0.2 && local.z.abs() < 3.0 {
                knocked.push(i);
            }
        }
        let mut gone = Vec::new();
        for &i in knocked.iter().rev() {
            // a waiting passenger knocked down leaves the stop and walks off (sub_626818)
            if let State::Pax(x) = &self.people[i].state {
                let (at, h, stop) = (x.pos, x.yaw.to_degrees(), x.stop);
                self.release(i);
                let world = None::<&World>;
                let _ = world;
                self.walk_street_plain(i, at, h, stop, &mut gone);
            }
        }
        gone.sort_unstable();
        for i in gone.into_iter().rev() {
            let p = self.people.swap_remove(i);
            self.retire(&p);
        }
        knocked.len() as u32
    }

    /// `--riders n`: n passengers already in their places in the player's bus (a test
    /// start; OMSI's buses start empty), without a destination - they ride 1..20 km.
    pub fn seed_riders(&mut self, n: usize, bus: &VehicleInstance, world: &World, renderer: &Renderer, scene: &mut Scene) {
        let Some(cabin) = self.cabin_for(bus) else { return };
        let trailers = part_frames(bus, &cabin);
        let rot = bus.body_rotation();
        let off = places_off(bus, &cabin);
        for _ in 0..n {
            let Some(k) = self.reserve_place(BusId::Player, cabin.seats.len(), &off) else { break };
            if !self.seat_rider(k, bus, &cabin, &trailers, &rot, world, renderer, scene) {
                break;
            }
        }
    }

    /// [ROLLBACK payload-121] Somebody on place `k` of the player's bus, sitting or standing there as the
    /// people `seed_riders` makes. False: nobody could be made (the place is given back).
    #[allow(clippy::too_many_arguments)]
    fn seat_rider(
        &mut self,
        k: usize,
        bus: &VehicleInstance,
        cabin: &Cabin,
        trailers: &[PartFrame],
        rot: &Mat4,
        world: &World,
        renderer: &Renderer,
        scene: &mut Scene,
    ) -> bool {
        let walk = 1.1 + (self.rand_f() as f32 * 2.0 - 1.0) * 0.2;
        let r = self.rand_f() as f32;
        let mut pax = Pax::new(walk, self.rand_f());
        pax.bus = Some(BusId::Player);
        pax.inside = Some(BusId::Player);
        pax.seat = Some(k);
        pax.ride_km = r * 19.0 + 1.0;
        pax.task = Task::InBusToPlace;
        let at = train_point(bus.position, rot, trailers, cabin.seats[k].pos);
        let Some(i) = self.spawn(
            world,
            renderer,
            scene,
            at,
            bus.heading,
            State::Pax(Box::new(pax)),
        ) else {
            self.free_seat(BusId::Player, k);
            return false;
        };
        let s = cabin.seats[k].clone();
        let seatheight = self.people[i].ty.def.seat_height;
        if let Some(p) = self.pax_mut(i) {
            p.task = Task::SittingInBus;
            p.st = 0;
            if s.seated {
                p.seat_h = s.height;
                p.pos = (s.pos - Vec3::Z * seatheight).as_dvec3();
                p.pax_state = 2.0;
            } else {
                p.pos = s.pos.as_dvec3();
            }
            p.yaw = (s.rot as f64).to_radians();
        }
        self.people[i].place = Place::Bus(BusId::Player, s.pos);
        true
    }

    /// [ROLLBACK payload-121] Developer tools: people on every free standing place (`standing`) or every free
    /// seat of the player's bus. Returns how many were put in.
    pub fn fill_places(
        &mut self,
        standing: bool,
        bus: &VehicleInstance,
        world: &World,
        renderer: &Renderer,
        scene: &mut Scene,
    ) -> usize {
        let Some(cabin) = self.cabin_for(bus) else {
            return 0;
        };
        let trailers = part_frames(bus, &cabin);
        let rot = bus.body_rotation();
        let off = places_off(bus, &cabin);
        let n = cabin.seats.len();
        let mut put = 0;
        for k in 0..n {
            if cabin.seats[k].seated == standing || off.get(k).copied().unwrap_or(false) {
                continue;
            }
            let taken = self
                .seats
                .entry(BusId::Player)
                .or_insert_with(|| vec![false; n]);
            if taken.len() < n {
                taken.resize(n, false);
            }
            if taken[k] {
                continue;
            }
            taken[k] = true;
            let before = self.people.len();
            if !self.seat_rider(k, bus, &cabin, &trailers, &rot, world, renderer, scene) {
                break;
            }
            if self.people.len() > before {
                let id = self.people[self.people.len() - 1].id;
                self.fake.insert(id);
            }
            put += 1;
        }
        put
    }

    /// [ROLLBACK payload-121] Developer tools: the people `fill_places` put in leave without a trace (the
    /// others, real passengers, stay). Returns how many.
    pub fn clear_fake(&mut self) -> usize {
        let mut gone = 0;
        for i in (0..self.people.len()).rev() {
            if !self.fake.contains(&self.people[i].id) {
                continue;
            }
            self.release(i);
            let p = self.people.swap_remove(i);
            self.retire(&p);
            gone += 1;
        }
        self.fake.clear();
        gone
    }

    /// [ROLLBACK intsep-122] The section of `bus` a world point is in (0 the front one, `n + 1` the
    /// coupled part `n`), as the walkers tell it; `None` when the bus is not known.
    pub fn section_at(&self, bus: BusId, world: DVec3) -> Option<usize> {
        let bn = self.last_buses.iter().find(|b| b.id == bus)?;
        let mut l = bn.rot.inverse().transform_point3((world - bn.pos).as_vec3());
        let mut sec = 0;
        for t in &bn.trailers {
            if l.y > t.joint_y {
                break;
            }
            l = t.rot.inverse().transform_point3((world - t.pos).as_vec3()) + t.offset;
            sec += 1;
        }
        Some(sec)
    }

    /// [ROLLBACK payload-121] The passengers inside bus `bus` for its physics: per section (the front one
    /// first) their mass (kg, the `.hum`'s `[mass]`, 75 where it says none) and the centre of it in that
    /// section's own model frame.
    pub fn payload_of(&self, bus: BusId) -> Vec<(f32, Vec3)> {
        let Some(bn) = self.last_buses.iter().find(|b| b.id == bus) else {
            return Vec::new();
        };
        let mut sum = vec![(0.0f32, Vec3::ZERO); 1 + bn.trailers.len()];
        for p in &self.people {
            let Place::Bus(b, local) = p.place else {
                continue;
            };
            if b != bus {
                continue;
            }
            // (the section whose joint the point lies behind)
            let mut sec = 0;
            for (k, t) in bn.trailers.iter().enumerate() {
                if behind(t, local.y) >= 0.5 {
                    sec = k + 1;
                } else {
                    break;
                }
            }
            let own = if sec == 0 {
                local
            } else {
                local - bn.trailers[sec - 1].offset
            };
            let kg = if p.ty.def.mass > 1.0 {
                p.ty.def.mass
            } else {
                75.0
            };
            let e = &mut sum[sec];
            e.1 += own * kg;
            e.0 += kg;
        }
        sum.into_iter()
            .map(|(kg, m)| {
                if kg > 0.0 {
                    (kg, m / kg)
                } else {
                    (0.0, Vec3::ZERO)
                }
            })
            .collect()
    }

    /// Give back what a person holds (a waiting place, a seat) before they change plans.
    fn release(&mut self, i: usize) {
        let id = self.people[i].id;
        let State::Pax(x) = &mut self.people[i].state else { return };
        let (stop, spot, bus, seat) = (x.stop, x.spot.take(), x.bus.or(x.inside), x.seat.take());
        if let (Some(s), Some(k)) = (stop, spot) {
            self.free_spot(s, k);
        }
        if let (Some(b), Some(k)) = (bus, seat) {
            self.free_seat(b, k);
        }
        if self.desk_busy == Some(id) {
            self.desk_busy = None;
            self.request = None;
        }
    }

    /// OMSI_CHECK_WALLS: everybody inside a bus who stands away from its walkways (more
    /// than 0.45 m from every path link, not on a seat): through a seat back or a wall.
    fn check_walls(&self) {
        for p in &self.people {
            let Place::Bus(bus, local) = p.place else { continue };
            if matches!(&p.state, State::Pax(x) if x.task == Task::SittingInBus || x.st == 9) {
                continue;
            }
            let Some(bn) = self.last_buses.iter().find(|b| b.id == bus) else { continue };
            let pts = &bn.cabin.graph.points;
            let mut best = f32::INFINITY;
            for &(a, b, _) in &bn.cabin.links {
                let (Some(pa), Some(pb)) = (pts.get(a.max(0) as usize), pts.get(b.max(0) as usize)) else { continue };
                let ab = *pb - *pa;
                let t = if ab.length_squared() > 1e-6 { ((local - *pa).dot(ab) / ab.length_squared()).clamp(0.0, 1.0) } else { 0.0 };
                let q = *pa + ab * t;
                best = best.min((q.truncate() - local.truncate()).length() + (q.z - local.z).abs());
            }
            let near_seat = bn.cabin.seats.iter().any(|s| (s.floor - local).truncate().length() < 0.35 || (s.pos - local).truncate().length() < 0.35);
            if best > 0.45 && !near_seat && !bn.cabin.links.is_empty() {
                log::warn!("t={:.1} person {} in bus {:?} off the walkways by {best:.2} m at ({:.2}, {:.2}, {:.2}), state {}", self.time, p.id, bus, local.x, local.y, local.z, p.state.name());
            }
        }
    }

    /// Report the waiting and alighting passengers to the bus script, the way OMSI does.
    pub fn write_pax_vars(&self, b: &mut VehicleInstance) {
        let doors = DoorWants {
            entry_req: self.entry_req.clone(),
            exit_req: self.exit_req.clone(),
            entry_busy: self.entry_busy.clone(),
            exit_busy: self.exit_busy.clone(),
            places: self.pax_places.get(&BusId::Player).cloned().unwrap_or_default(),
        };
        Self::write_door_requests(b, &doors);
    }

    /// Who wants a timetable bus to stop, as Omsi.exe asks before it lets one pull in
    /// (0x7da91f): the AI buses with somebody aboard on the way to a door to get off
    /// (task 5), and the stops where somebody is waiting for a bus or walking to one
    /// (tasks 1 to 3).
    pub fn stop_wishes(&self) -> (HashSet<u64>, HashSet<i64>) {
        let (mut alighting, mut waiting) = (HashSet::new(), HashSet::new());
        for p in &self.people {
            let State::Pax(x) = &p.state else { continue };
            match x.task {
                Task::InBusToExit => {
                    if let Some(BusId::Ai(id)) = x.inside {
                        alighting.insert(id);
                    }
                }
                Task::WaitingForBus | Task::ToBus | Task::WalkingToBus => {
                    if let Some(s) = x.stop {
                        waiting.insert(s);
                    }
                }
                _ => {}
            }
        }
        (alighting, waiting)
    }

    /// Timetable buses to hold at their stop, for the traffic.
    pub fn take_holds(&mut self) -> Vec<(u64, Option<i64>, f32)> {
        std::mem::take(&mut self.holds)
    }

    /// Door requests for the timetable buses, for the traffic to hand to their scripts.
    pub fn take_ai_requests(&mut self) -> Vec<(u64, DoorWants)> {
        std::mem::take(&mut self.ai_requests)
    }

    /// A line for the HUD about something that just happened.
    pub fn take_message(&mut self) -> Option<String> {
        self.message.take()
    }

    /// What the driver should do now, for the HUD: a passenger waiting at the cash desk
    /// for the ticket (only when the driver has to sell it).
    pub fn hint(&self) -> Option<String> {
        if !self.boarding.eq_ignore_ascii_case("pay") {
            return None;
        }
        self.people.iter().find(|p| matches!(&p.state, State::Pax(x) if x.inside == Some(BusId::Player) && x.ticket == TICKET_BUY && x.sub == 5))?;
        let (name, value) = self.request.clone()?;
        Some(format!("Passenger waiting for a ticket: {name} {value:.2} - press {}", self.ticket_key))
    }

    /// People sitting on each `[passpos]` of the player's bus, for `GetHumanCountOnSeat`
    /// (the BVG Citaro folds its tip-up seats down when somebody sits on them).
    pub fn seat_counts(&self) -> Vec<u32> {
        let Some(cabin) = self.player_cabin.as_ref() else {
            return Vec::new();
        };
        let sitting = self.people.iter().filter_map(|p| match &p.state {
            State::Pax(x) if x.inside == Some(BusId::Player) && x.task == Task::SittingInBus => x.seat,
            _ => None,
        });
        seat_numbers(&cabin.seats, sitting)
    }

    /// How many people stand on each `paths.cfg` link inside the player's bus, for the
    /// scripts' `GetHumanCountOnPathLink` (the NL/NG uses it for the fare gate).
    pub fn path_link_counts(&self) -> Vec<u32> {
        let Some(cabin) = self.player_cabin.as_ref() else {
            return Vec::new();
        };
        // (the link each walker is on, +0x698)
        let mut out = vec![0u32; cabin.links.len()];
        for p in &self.people {
            if let State::Pax(x) = &p.state {
                if x.inside == Some(BusId::Player) && (x.st == 1 || x.st == 5 || x.st == 9) {
                    if let Some(c) = x.link.and_then(|l| out.get_mut(l)) {
                        *c += 1;
                    }
                }
            }
        }
        out
    }

    /// Where everybody is, for logs.
    pub fn positions(&self) -> Vec<(String, DVec3)> {
        self.people
            .iter()
            .map(|p| (p.state.name().to_string(), p.position))
            .collect()
    }

    /// Count of people per state, for logs.
    pub fn summary(&self) -> String {
        let mut counts: std::collections::BTreeMap<&'static str, usize> = Default::default();
        for p in &self.people {
            *counts.entry(p.state.name()).or_default() += 1;
        }
        let mut out = counts
            .iter()
            .map(|(k, v)| format!("{v} {k}"))
            .collect::<Vec<_>>()
            .join(", ");
        let (n, total, worst) = self.tick_stats;
        if n > 0 {
            out.push_str(&format!(
                "; {:.2} ms a frame, longest {worst:.1} ms",
                total / n as f64
            ));
        }
        let (frames, posed, ms, up) = self.pose_stats;
        if frames > 0 {
            out.push_str(&format!(
                "; posing {:.2} ms a frame ({:.1} people, {:.2} ms of it uploading and placing)",
                ms / frames as f64,
                posed as f64 / frames as f64,
                up / frames as f64
            ));
        }
        out
    }

    /// Advance everybody. `bus`: the player's vehicle; `traffic`: the timetable buses,
    /// the traffic lights and the cars pedestrians wait for. Returns true when a passenger
    /// took the printed ticket (the caller resets `GivenTicket`).
    pub fn tick(
        &mut self,
        dt: f32,
        world: &World,
        bus: Option<&VehicleInstance>,
        traffic: Option<&Traffic>,
        renderer: &Renderer,
        scene: &mut Scene,
    ) -> bool {
        let started = std::time::Instant::now();
        self.tick_stages.clear();
        let took = self.tick_inner(dt, world, bus, traffic, renderer, scene);
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        if (debug_pax() || omsi_cfg::env::var_os("OMSI_PROFILE").is_some()) && ms > 30.0 {
            log::info!(
                "t={:.1} slow people tick: {ms:.1} ms ({} people): {}",
                self.time,
                self.people.len(),
                self.tick_stages.iter().filter(|s| s.1 >= 1.0).map(|(n, t)| format!("{n} {t:.1}")).collect::<Vec<_>>().join(", ")
            );
        }
        if omsi_cfg::env::var_os("OMSI_CHECK_WALLS").is_some() {
            self.check_walls();
        }
        // OMSI_CHECK_GROUND=1: people on foot with a walkable surface over their heads'
        // reach above them, every two seconds (people "in the ground")
        if omsi_cfg::env::var_os("OMSI_CHECK_GROUND").is_some() && (self.time / 2.0).floor() != ((self.time - dt as f64) / 2.0).floor() {
            for p in &self.people {
                if !matches!(p.place, Place::Ground) || p.puppet.is_some() {
                    continue;
                }
                // the floor under the feet: the highest face up to a step (0.5 m) over them
                let floor = world.walk_height_near(p.position.x, p.position.y, p.position.z);
                if let Some(f) = floor {
                    if f - p.position.z > 0.05 {
                        let detail = match &p.state {
                            State::Pax(x) => format!(" st {} pax_state {} task {:?} pos.z {:.2}", x.st, x.pax_state, x.task, x.pos.z),
                            _ => String::new(),
                        };
                        log::warn!("t={:.1} person {} ({}) {:.2} m under the floor at ({:.1}, {:.1}, {:.2}), top surface {:?}{detail}", self.time, p.id, p.state.name(), f - p.position.z, p.position.x, p.position.y, p.position.z, world.walk_height(p.position.x, p.position.y));
                    }
                }
            }
        }
        self.tick_stats.0 += 1;
        self.tick_stats.1 += ms;
        self.tick_stats.2 = self.tick_stats.2.max(ms);
        took
    }

    #[allow(unused_assignments)]
    fn tick_inner(
        &mut self,
        dt: f32,
        world: &World,
        bus: Option<&VehicleInstance>,
        traffic: Option<&Traffic>,
        renderer: &Renderer,
        scene: &mut Scene,
    ) -> bool {
        let mut mark = std::time::Instant::now();
        macro_rules! stage {
            ($name:expr) => {{
                let now = std::time::Instant::now();
                self.tick_stages.push(($name, (now - mark).as_secs_f64() * 1000.0));
                mark = now;
            }};
        }
        self.use_map_humans(world);
        self.time += dt as f64;
        let net = traffic.map(|t| &t.net);
        if let Some(b) = bus {
            self.center = b.position;
        } else if let Some(e) = self.eye {
            self.center = e.pos;
        }
        let generation = world.tiles_generation.load(std::sync::atomic::Ordering::Relaxed);
        if generation != self.tiles_seen {
            self.tiles_seen = generation;
            self.tiles_changed(world);
        }
        stage!("tiles");
        // tiles brought lanes: their pavements join the network, and stops without one look again
        if let (Some(pn), Some(n)) = (self.ped.as_mut(), net) {
            if pn.built < n.lanes.len() {
                let added = pn.extend(n);
                if added > 0 {
                    let ids: Vec<(i64, DVec3)> = self.stops.iter().filter(|(_, s)| s.lane.is_none()).map(|(k, s)| (*k, s.pos)).collect();
                    for (id, pos) in ids {
                        let lane = self.ped.as_ref().and_then(|pn| pn.nearest(n, pos, 12.0)).map(|(l, s, _)| (l, s));
                        self.stops.get_mut(&id).unwrap().lane = lane;
                    }
                }
            }
        }
        if self.ped.is_none() {
            if let Some(n) = net {
                self.ped = Some(PedNet::build(n));
                let ids: Vec<(i64, DVec3)> = self.stops.iter().map(|(k, s)| (*k, s.pos)).collect();
                for (id, pos) in ids {
                    let lane = self.ped.as_ref().and_then(|pn| pn.nearest(n, pos, 12.0)).map(|(l, s, _)| (l, s));
                    self.stops.get_mut(&id).unwrap().lane = lane;
                }
            }
        }
        stage!("pedestrian network");
        if let Some(n) = net {
            self.stroll_timer -= dt;
            if self.stroll_timer <= 0.0 {
                self.stroll_timer = 1.0;
                let c = self.center;
                self.populate_with(world, Some(n), renderer, scene, c);
                if !self.mirror {
                    let lights = traffic.map(|t| t as &dyn PedLights);
                    self.populate_on_foot(world, n, lights, renderer, scene, 1.0);
                    self.populate_lan_centers(world, n, lights, renderer, scene);
                }
            }
        }
        stage!("populate");
        let mut buses = self.gather_buses(world, bus, traffic);
        for b in &buses {
            if b.entry_open.iter().chain(b.exit_open.iter()).any(|o| *o) {
                self.last_door_open.insert(b.id, self.time);
            }
        }
        // how the floor of each bus accelerates (for the drawing of its riders)
        if dt > 1e-4 {
            let mut motion = HashMap::new();
            for bn in buses.iter_mut() {
                let accel = match self.bus_motion.get(&bn.id) {
                    Some(&(v0, h0, a0)) => {
                        let yaw_rate = crowd::angle_diff(h0, bn.heading).to_radians() / dt as f64;
                        let raw = DVec2::new(bn.speed * yaw_rate, (bn.speed - v0) / dt as f64).clamp(DVec2::splat(-6.0), DVec2::splat(6.0));
                        a0 + (raw - a0) * (1.0 - (-(dt as f64) / 0.2).exp())
                    }
                    None => DVec2::ZERO,
                };
                bn.accel = accel;
                motion.insert(bn.id, (bn.speed, bn.heading, accel));
            }
            self.bus_motion = motion;
        }
        let buses = buses;
        self.last_buses = buses.clone();
        let bus_ix: HashMap<BusId, usize> = buses.iter().enumerate().map(|(i, b)| (b.id, i)).collect();
        // the stops: which buses stand at them (sub_61f93c), who waits there (sub_61bf94)
        let at_stops = self.register_buses(&buses, dt);
        self.claim_waiting();
        self.ride_comfort(dt, bus, &buses, &bus_ix, world);
        if !self.avatar_only {
            self.stops_tick(dt, world, renderer, scene);
        }
        stage!("stops");
        // the passengers (sub_6ffc7c)
        let mut taken_ticket = false;
        let mut remove: Vec<usize> = Vec::new();
        self.pax_frame(dt, world, &buses, &bus_ix, &at_stops, bus, renderer, scene, &mut taken_ticket, &mut remove);
        stage!("passengers");
        // the pedestrians: a crowd on the pavements
        let mut cars: Vec<SeenCar> = Vec::new();
        let mut blocks: Vec<Block> = Vec::new();
        if let Some(t) = traffic {
            for c in &t.cars {
                if (c.vehicle.position - self.center).length() > 320.0 {
                    continue;
                }
                let h = c.vehicle.heading.to_radians();
                let fwd = DVec2::new(h.sin(), h.cos());
                let bb = c.vehicle.ty.def.bounding_box.unwrap_or(crate::traffic::DEFAULT_BOX);
                let o = omsi_sim::collision::Obb::from_box(bb, c.vehicle.position, c.vehicle.heading);
                let body = Block { center: o.center, half: o.half, heading: o.heading, vel: fwd * c.ai.state.speed as f64 };
                // [ROLLBACK pedyield-1] (and whether it is stopping for somebody on foot:
                // a car that yields is no threat to the one it yields to)
                cars.push(SeenCar {
                    id: c.id,
                    pos: c.vehicle.position.truncate(),
                    vel: fwd * c.ai.state.speed as f64,
                    half_len: bb[1] as f64 * 0.5,
                    body,
                    yielding: yields_to_people(c.ai.last.why),
                    stood: 0.0,
                });
                if !matches!(c.vehicle.ty.def.kind, omsi_vehicle::VehicleKind::Other(3)) {
                    blocks.push(body);
                    for t in &c.vehicle.trailers {
                        let tb = t.ty.def.bounding_box.unwrap_or([2.5, 7.0, 3.0, 0.0, 0.0, 1.5]);
                        let o = omsi_sim::collision::Obb::from_box(tb, t.position, t.heading);
                        let th = t.heading.to_radians();
                        blocks.push(Block { center: o.center, half: o.half, heading: o.heading, vel: DVec2::new(th.sin(), th.cos()) * c.ai.state.speed as f64 });
                    }
                }
            }
        }
        for o in world.parked_boxes.lock().iter() {
            if (o.center - self.center.truncate()).length() < 320.0 {
                blocks.push(Block { center: o.center, half: o.half, heading: o.heading, vel: DVec2::ZERO });
            }
        }
        if let Some(pb) = bus_ix.get(&BusId::Player).map(|&i| &buses[i]) {
            let bodies = pb.blocks();
            cars.push(SeenCar { id: PLAYER_CAR, pos: pb.pos.truncate(), vel: pb.fwd() * pb.speed, half_len: pb.half.y, body: bodies[0], yielding: false, stood: 0.0 });
            for (k, t) in pb.trailers.iter().enumerate() {
                let h = t.heading.to_radians();
                let body = bodies.get(k + 1).copied().unwrap_or(bodies[0]);
                cars.push(SeenCar { id: PLAYER_CAR - 1 - k as u64, pos: t.pos.truncate(), vel: DVec2::new(h.sin(), h.cos()) * pb.speed, half_len: t.half.y, body, yielding: false, stood: 0.0 });
            }
            blocks.extend(bodies);
        }
        // [ROLLBACK r4c-P4] how long each vehicle has stood; the yields to those that move again end
        // ([ROLLBACK r5-P5] a creep does not count as moving again: `stood_clock`)
        {
            let mut stood: HashMap<u64, (f32, DVec2)> = HashMap::new();
            for c in cars.iter_mut() {
                let (s, anchor) = stood_clock(self.car_stood.get(&c.id).copied(), c.pos, c.vel.length(), dt);
                c.stood = s;
                if s > 0.0 {
                    stood.insert(c.id, (s, anchor));
                }
            }
            self.car_stood = stood;
            let car_stood = &self.car_stood;
            self.yields.retain(|id, _| car_stood.contains_key(id));
            // and who is on the carriageway of which crossing
            let mut streams: Streams = HashMap::new();
            for p in &self.people {
                if let State::Strolling(w) = &p.state {
                    if let Some(c) = w.cross.filter(|c| c.on(w.leg) && p.on_crossing) {
                        streams.entry(c.key).or_default().push((p.id, p.position.truncate(), p.vel));
                    }
                }
            }
            self.streams = streams;
        }
        let mut wants: Vec<Want> = Vec::with_capacity(self.people.len());
        for i in 0..self.people.len() {
            self.people[i].t_state += dt;
            let w = if self.people[i].puppet.is_some() || matches!(self.people[i].state, State::Pax(_)) {
                Want::stand(None, Activity::Stand)
            } else if self.people[i].remote {
                self.mirror_want(i, &buses)
            } else {
                self.decide(i, dt, world, net, traffic, &cars, &mut remove)
            };
            wants.push(w);
        }
        // a standing vehicle in the way: wait, then go round it
        for i in 0..self.people.len() {
            let p = &self.people[i];
            if !VEHICLE_COLLISIONS || remove.contains(&i) || p.puppet.is_some() || p.remote || !matches!(p.state, State::Strolling(_)) || p.step_out.is_some() {
                continue;
            }
            let want = wants[i].vel;
            let speed = want.length();
            if speed < 0.2 {
                self.people[i].car_wait = 0.0;
                continue;
            }
            // [ROLLBACK around-86] on the way round a big box already: on to its next corner
            let here = p.position.truncate();
            if let Some((wp, k, left)) = self.people[i].around {
                let (k, left) = if (wp[k] - here).length() < 0.45 { (k + 1, left) } else { (k, left - dt) };
                if k >= 2 || left <= 0.0 {
                    self.people[i].around = None;
                } else {
                    self.people[i].around = Some((wp, k, left));
                    self.people[i].detour = self.people[i].detour.max(0.3);
                    wants[i].vel = (wp[k] - here).normalize_or_zero() * speed;
                    self.people[i].car_wait = 0.0;
                    continue;
                }
            }
            let ahead = here + want / speed * 0.9;
            let in_way = blocks.iter().any(|b| {
                b.vel.length() < 0.5 && b.near(ahead, BODY_OUTSIDE + 0.15) && {
                    let (q, inside) = b.closest(ahead);
                    inside || (ahead - q).length() < BODY_OUTSIDE + 0.15
                }
            });
            if !in_way {
                self.people[i].car_wait = 0.0;
                continue;
            }
            // [ROLLBACK around-86] a bus or a lorry (not a car or a post) that the way leads
            // into: round its nearer end by its bounding box, not into its side and waiting
            let dir = want / speed;
            // [ROLLBACK r4c-P5] one that lies along the way (a bus standing half on the pavement):
            // along its free side within the path's width; round its end it was across the
            // way, and the people paced to and fro in front of its nose (the bus held by them)
            let big = |b: &&Block| b.vel.length() < 0.5 && b.half.max_element() >= 2.0;
            let mut no_room = false;
            let mut route = None;
            for b in blocks.iter().filter(big) {
                match along_box(b, here, dir, wants[i].corridor) {
                    Along::Route(wp) => {
                        route = Some(wp);
                        break;
                    }
                    Along::NoRoom => no_room = true,
                    Along::Across => {}
                }
            }
            if route.is_none() && !no_room {
                route = blocks.iter().filter(big).find_map(|b| around_box(b, here, dir));
            }
            if let Some(wp) = route {
                self.people[i].around = Some((wp, 0, 12.0));
                self.people[i].detour = self.people[i].detour.max(0.3);
                wants[i].vel = (wp[0] - here).normalize_or_zero() * speed;
                self.people[i].car_wait = 0.0;
                continue;
            }
            // [ROLLBACK r4c-P6] (not for long: somebody who has decided to go and stands is
            // planned again after PED_GO_FROZEN - back to the kerb when just off it, else
            // round the vehicle; after 8 s, people stood frozen with "go" in the road)
            self.people[i].car_wait += dt;
            if self.people[i].car_wait > PED_GO_FROZEN {
                self.people[i].car_wait = 0.0;
                let p = &mut self.people[i];
                if let (State::Strolling(walk), Some(net)) = (&mut p.state, net) {
                    if let Some(c) = walk.cross.filter(|c| c.on(walk.leg) && !c.back) {
                        if c.walked(walk) < 1.5 {
                            walk.turn_back(net);
                            p.last_decision = Some(BACK_TO_KERB);
                            wants[i].vel = DVec2::ZERO;
                            continue;
                        }
                    }
                }
                p.detour = p.detour.max(4.0);
            } else if self.people[i].detour <= 0.0 {
                wants[i].vel = DVec2::ZERO;
            }
        }
        // [ROLLBACK r4c-P6] nobody stands frozen with "go": released from the path's corridor for
        // a moment, the crowd takes them round what holds them
        for i in 0..self.people.len() {
            let p = &mut self.people[i];
            if matches!(p.state, State::Strolling(_)) && p.detour <= 0.0 && replan_frozen(p.last_decision, p.still, wants[i].vel.length()) {
                p.detour = 2.0;
            }
        }
        // [ROLLBACK r4c-P5] beside a standing vehicle nobody turns round more than once in
        // PED_TURN_EVERY seconds (they stand instead)
        for i in 0..self.people.len() {
            if !matches!(self.people[i].state, State::Strolling(_)) || remove.contains(&i) {
                continue;
            }
            let here = self.people[i].position.truncate();
            let near_box = blocks.iter().any(|b| b.vel.length() < 0.5 && b.near(here, 3.0));
            let p = &mut self.people[i];
            p.turned += dt;
            let v = wants[i].vel;
            let speed = v.length();
            if speed < 0.2 {
                continue;
            }
            let dir = v / speed;
            if p.want_dir.dot(dir) < -0.2 {
                // (not on a crossing: turned back by the change, they go at once, P1)
                if near_box && !p.on_crossing && !may_turn(p.turned) {
                    wants[i].vel = DVec2::ZERO;
                    continue;
                }
                p.turned = 0.0;
            }
            p.want_dir = dir;
        }
        // the crowd of the pavements (passengers outside stand in it as they are)
        let mut walkers: Vec<Walker> = Vec::with_capacity(self.people.len());
        let mut who: Vec<usize> = Vec::with_capacity(self.people.len());
        for (i, p) in self.people.iter().enumerate() {
            if remove.contains(&i) || p.puppet.is_some() || p.remote || p.place != Place::Ground {
                continue;
            }
            let fixed = matches!(p.state, State::Pax(_));
            let w = &wants[i];
            walkers.push(Walker {
                pos: p.position.truncate(),
                vel: p.vel,
                radius: BODY_OUTSIDE,
                want: w.vel,
                give: w.give,
                space: 0,
                fixed,
                // [ROLLBACK nocoll-93] people on foot touch nothing but the vehicles (`solid`):
                // not each other, not the scenery
                ghost: true,
                solid: VEHICLE_COLLISIONS,
                corridor: if p.detour > 0.0 { None } else { w.corridor },
            });
            who.push(i);
        }
        let near_blocks: Vec<Block> = blocks.into_iter().filter(|b| walkers.iter().any(|w| b.near(w.pos, 25.0))).collect();
        let mut g = walkers.clone();
        crowd::step(&mut g, &near_blocks, &CrowdParams::default(), dt as f64);
        let ground: Vec<(usize, Walker)> = g.into_iter().enumerate().collect();
        let mut moved = vec![false; self.people.len()];
        for (k, w) in ground {
            let i = who[k];
            if matches!(self.people[i].state, State::Pax(_)) {
                // [round5 C2] (the passengers' engine moves them; their standing time runs
                // here: a bus waiter's `still` stayed 0 for ever, and a car beside their stop
                // was never let go, viz4 case03 #12135 88 s)
                let p = &mut self.people[i];
                p.still = still_after(p.still, p.vel.length() as f32, dt);
                continue;
            }
            moved[i] = true;
            self.apply(i, &w, &wants[i], dt, world, net, &buses, &bus_ix);
        }
        for i in 0..self.people.len() {
            if !moved[i] && !matches!(self.people[i].state, State::Pax(_)) {
                self.carry(i, dt, &buses, &bus_ix, wants[i].face);
            }
        }
        self.animate(dt, world, &buses, &bus_ix);
        remove.sort_unstable();
        remove.dedup();
        for i in remove.into_iter().rev() {
            self.release(i);
            let p = self.people.swap_remove(i);
            if debug_pax() {
                log::info!("t={:.1} pax {} taken away ({}){}", self.time, p.label(), p.state.name(), if self.seen(p.position) { " IN SIGHT" } else { "" });
            }
            self.retire(&p);
        }
        self.give_ticket = false;
        stage!("pedestrians");
        taken_ticket
    }

    /// What pedestrian `i` wants this frame (task 8, `WalkStreet`).
    #[allow(clippy::too_many_arguments)]
    fn decide(
        &mut self,
        i: usize,
        dt: f32,
        world: &World,
        net: Option<&Network>,
        traffic: Option<&Traffic>,
        cars: &[SeenCar],
        remove: &mut Vec<usize>,
    ) -> Want {
        let state = self.people[i].state.clone();
        let pos2 = self.people[i].position.truncate();
        // somebody walking on towards ground that is not loaded goes
        if self.people[i].place == Place::Ground && self.people[i].vel.length_squared() > 1e-4 && !world.has_ground(pos2.x, pos2.y) {
            remove.push(i);
            return Want::stand(None, Activity::Stand);
        }
        match state {
            State::Strolling(mut walk) => {
                let seen = self.seen(self.people[i].position);
                // (a stroller goes only once well out of everybody's range and out of sight)
                let far = self.far_from_players(self.people[i].position, STROLL_RADIUS * 2.0);
                let Some(net) = net else {
                    remove.push(i);
                    return Want::stand(None, Activity::Stand);
                };
                if far && !seen {
                    remove.push(i);
                    return Want::stand(None, Activity::Stand);
                }
                // [ROLLBACK stepout-70] off a bus: first straight away from its side
                if let Some((tgt, left)) = self.people[i].step_out {
                    let pos2 = self.people[i].position.truncate();
                    let d = tgt - pos2;
                    if d.length() < 0.25 || left <= 0.0 {
                        self.people[i].step_out = None;
                    } else {
                        self.people[i].step_out = Some((tgt, left - dt));
                        self.people[i].state = State::Strolling(walk);
                        let pace = self.people[i].pace;
                        return Want { vel: d.normalize() * pace, face: None, give: 1.0, corridor: None, idle: Activity::Stand };
                    }
                }
                let w = self.walk_want(i, &mut walk, net, traffic, cars, dt);
                self.people[i].state = State::Strolling(walk);
                w
            }
            State::Standing => {
                // somebody who got off with no pavement known beside the stop (or off a stop)
                // looks for one about once a second - those at a stop whose pavement was not
                // found, or not loaded yet, stood there for good - and walks on along it
                let t = self.people[i].t_state;
                if let (Some(net), true) = (net, t % 1.0 < dt || t == 0.0) {
                    let at = self.people[i].position;
                    let found = self.ped.as_ref().and_then(|pn| pn.nearest(net, at, STANDING_REACH));
                    // [ROLLBACK reach-93] a pavement across the street does not count (they
                    // walked into the bus to reach it); with none within reach they are gone
                    let found = found.filter(|&(lane, s, _)| {
                        let (q, _) = net.lanes[lane].at(s);
                        !crosses_street(net, at.truncate(), q.truncate())
                    });
                    if let Some((lane, s, _)) = found {
                        let leg = Leg { lane, a: s, b: s };
                        self.people[i].state = State::Strolling(PedWalk::new(vec![leg], true, 0.0));
                        self.people[i].t_state = 0.0;
                        return Want::stand(None, Activity::Stand);
                    }
                    if self.ped.is_some() {
                        remove.push(i);
                        return Want::stand(None, Activity::Stand);
                    }
                }
                // [ROLLBACK respawn-61] as in OMSI 2, somebody who finds no pavement is respawned
                // (taken away; the spawner puts a pedestrian where one belongs): at once when out
                // of sight, after about a second, in sight or not (STANDING_GIVE_UP)
                let gone = !self.seen(self.people[i].position) && self.far_from_players(self.people[i].position, STROLL_RADIUS);
                if gone || t >= STANDING_GIVE_UP {
                    remove.push(i);
                }
                Want::stand(None, Activity::Stand)
            }
            _ => Want::stand(None, Activity::Stand),
        }
    }

    /// Where a walk along the pavement takes somebody next.
    fn walk_want(
        &mut self,
        i: usize,
        walk: &mut PedWalk,
        net: &Network,
        traffic: Option<&Traffic>,
        cars: &[SeenCar],
        dt: f32,
    ) -> Want {
        let mut ped = self.ped.take();
        let p = &self.people[i];
        let mut me = Me { id: p.id, pos: p.position.truncate(), pace: p.pace, on_crossing: p.on_crossing, decision: p.last_decision, why: p.why, walk_pace: p.pace };
        let at = Around { net, lights: traffic.map(|t| t as &dyn PedLights), cars, streams: &self.streams, time: self.time };
        let mut rng = self.rng;
        let w = stroll(ped.as_mut(), &at, &mut me, walk, &mut rng, &mut self.yields, dt);
        self.rng = rng;
        self.ped = ped;
        let p = &mut self.people[i];
        p.on_crossing = me.on_crossing;
        p.last_decision = me.decision;
        p.why = me.why;
        p.walk_pace = me.walk_pace;
        w
    }

    /// Take over where the crowd moved person `i`.
    #[allow(clippy::too_many_arguments)]
    fn apply(
        &mut self,
        i: usize,
        w: &Walker,
        want: &Want,
        dt: f32,
        world: &World,
        net: Option<&Network>,
        buses: &[BusNow],
        bus_ix: &HashMap<BusId, usize>,
    ) {
        let dt64 = dt as f64;
        let p = &mut self.people[i];
        let speed = w.vel.length();
        p.still = still_after(p.still, speed as f32, dt);
        // somebody pressed against somebody else for seconds slips past them
        if want.vel.length() > 0.2 && speed < 0.08 {
            p.stuck += dt;
        } else if speed > 0.2 {
            p.stuck = 0.0;
        }
        p.detour = (p.detour - dt).max(0.0);
        p.vel = w.vel;
        match p.place {
            Place::Ground => {
                p.position.x = w.pos.x;
                p.position.y = w.pos.y;
                // [ROLLBACK bridge-1] a pavement path over the road carries the walker at its
                // own height: a footbridge is an editor-only spline placed at height with no
                // floor of its own (BeijingCity's `Human_path.sli`, its drawn profile 100 m
                // under ground, the bridge mesh `[nocollision]`), so the probe found the road
                // six metres below and lowered the people onto it, where they crossed the
                // junction on the bridge's line, through the traffic and past the lights.
                // Omsi.exe walks its people at the path's height.
                let path_z = match (&p.state, net) {
                    (State::Strolling(walk), Some(net)) => walk.legs.get(walk.leg).and_then(|leg| {
                        let l = net.lanes.get(leg.lane)?;
                        let s = if leg.b >= leg.a { leg.a + walk.s } else { leg.a - walk.s };
                        Some(l.at(s.clamp(0.0, l.length())).0.z)
                    }),
                    _ => None,
                };
                let floor = world
                    .walk_height_near(p.position.x, p.position.y, p.position.z)
                    .map(|z| match path_z {
                        Some(pz) if pz - z > 1.0 => pz,
                        _ => z,
                    })
                    .or(path_z);
                if let Some(z) = floor {
                    // up a kerb quickly, down it smoothly (the feet find the kerb themselves);
                    // more than a kerb below the surface is no step but a wrong height (the
                    // pavement's tile came after them): straight onto it
                    // (and whoever stands still simply stands on it: waiting people sank
                    // into a pavement that came after them and rose only when they walked)
                    p.position.z = if z - p.position.z > 0.35 || speed < 0.05 {
                        z
                    } else if z > p.position.z {
                        z.min(p.position.z + 1.5 * dt64)
                    } else {
                        z.max(p.position.z - 2.0 * dt64)
                    };
                }
            }
            Place::Bus(b, l) => {
                let here = Vec3::new(w.pos.x as f32, w.pos.y as f32, l.z);
                let z = l.z;
                let local = Vec3::new(here.x, here.y, z);
                p.place = Place::Bus(b, local);
                if let Some(bn) = bus_ix.get(&b).map(|k| &buses[*k]) {
                    p.position = bn.world(local);
                    p.interior = bn.interior;
                    p.tilt = bn.tilt_at(local);
                }
            }
        }
        // progress along the pavement
        if let Some(net) = net {
            let pos = p.position;
            match &mut p.state {
                State::Strolling(walk) => {
                    if let Some(leg) = walk.legs.get(walk.leg) {
                        walk.s = leg.project(net, pos, walk.s).max(walk.s - 0.3);
                    }
                }
                _ => {}
            }
        }
        let walking = if p.activity == Activity::Walk {
            speed > 0.12
        } else {
            speed > 0.3
        };
        let activity = if walking { Activity::Walk } else { want.idle };
        let inside = matches!(p.place, Place::Bus(..));
        let bus_heading = match p.place {
            Place::Bus(b, l) => bus_ix
                .get(&b)
                .map(|k| buses[*k].heading_at(l))
                .unwrap_or(0.0),
            Place::Ground => 0.0,
        };
        let current = if inside { p.lheading } else { p.heading };
        let target = if speed > 0.25 {
            Some(crowd::heading_of(w.vel))
        } else {
            want.face
        };
        // turning eases in and out (a constant rate started and stopped with a jerk): the
        // rate follows the angle still to go, up to the most a walker or a stander turns
        let turned = match target {
            Some(t) => {
                let left = crowd::angle_diff(current, t);
                let max_rate = if walking { 260.0 } else { 140.0 };
                let rate = (left.abs() * 5.0).min(max_rate).max(12.0);
                crowd::turn_towards(current, t, rate, dt64)
            }
            None => current,
        };
        if inside {
            p.lheading = turned;
            p.heading = bus_heading + turned;
        } else {
            p.heading = turned;
        }
        p.activity = activity;
    }

    /// People carried by a bus in their seat.
    fn carry(
        &mut self,
        i: usize,
        dt: f32,
        buses: &[BusNow],
        bus_ix: &HashMap<BusId, usize>,
        face: Option<f64>,
    ) {
        let p = &mut self.people[i];
        let Place::Bus(b, l) = p.place else { return };
        let Some(bn) = bus_ix.get(&b).map(|k| &buses[*k]) else {
            return;
        };
        p.position = bn.world(l);
        p.tilt = bn.tilt_at(l);
        p.vel = DVec2::ZERO;
        if let Some(f) = face {
            p.lheading = crowd::turn_towards(p.lheading, f, 150.0, dt as f64);
        }
        p.heading = bn.heading_at(l) + p.lheading;
        p.interior = bn.interior;
    }

    /// OMSI's `change_take`: the driver takes back the coins lying on the change tray.
    pub fn take_change_tray(&mut self) {
        if let Some(m) = self.money.as_mut() {
            m.clear(true);
        }
    }

    /// Coins the driver handed out (from the host's GiveChangeCoin list) onto the change point.
    pub fn give_change(
        &mut self,
        world: &World,
        renderer: &Renderer,
        scene: &mut Scene,
        coins: &[usize],
    ) {
        if coins.is_empty() {
            return;
        }
        let point = self.player_cabin.as_ref().and_then(|c| {
            c.data
                .change_points
                .first()
                .or(c.data.money_points.first())
                .cloned()
        });
        if let (Some(m), Some(pt)) = (self.money.as_mut(), point) {
            m.place(
                world,
                renderer,
                scene,
                coins,
                Vec3::from(pt.pos),
                pt.var,
                true,
            );
        }
    }

    pub fn sync_money(&mut self, renderer: &Renderer, scene: &mut Scene, bus: &VehicleInstance) {
        if let Some(m) = self.money.as_mut() {
            m.sync(renderer, scene, bus);
        }
    }

    /// Everybody's animation this frame (sub_626ae8): the passengers from what their task
    /// says (`PAX_State`, speed, the room height, the seat, the hand and the head), the
    /// pedestrians from their walk.
    fn animate(&mut self, dt: f32, world: &World, buses: &[BusNow], bus_ix: &HashMap<BusId, usize>) {
        let dt_ms = dt * 1000.0;
        for i in 0..self.people.len() {
            if let Some(pp) = self.people[i].puppet {
                if pp.mode == PuppetMode::Avatar {
                    self.animate_avatar(i, dt, world, buses, bus_ix);
                }
                continue;
            }
            let p = &self.people[i];
            let (input, footstep) = match &p.state {
                State::Pax(x) => {
                    let bn = x.inside.and_then(|b| bus_ix.get(&b).map(|k| &buses[*k]));
                    // a point of the bus in the person's own frame, Direct3D's axes
                    let own = |q: Vec3| -> Vec3 {
                        let v = q.as_dvec3() - x.pos;
                        let (s, c) = x.yaw.sin_cos();
                        let local = Vec3::new((v.x * c - v.y * s) as f32, (v.x * s + v.y * c) as f32, v.z as f32);
                        omsi_sim::human_omsi::d3d(local)
                    };
                    let reach = (x.reach && x.inside.is_some()).then(|| own(x.reach_at));
                    let look = match (x.look_driver, bn) {
                        (true, Some(b)) => b.cabin.data.driver_positions.first().map(|d| own(Vec3::from(d.pos) + Vec3::Z * 0.65)),
                        _ => None,
                    };
                    let kind = x.pax_state.round().clamp(0.0, 2.0) as u8;
                    let pack = match (x.step_pack, bn) {
                        (Some(k), Some(b)) => b.cabin.step_packs.get(k).cloned().map(|pk| (b.id, pk)),
                        _ => None,
                    };
                    (
                        AnimInput {
                            kind,
                            speed: x.speed,
                            moved: x.moved,
                            room_height: x.room,
                            seat_height: x.seat_h,
                            reach,
                            look,
                            smooth: x.smooth,
                            dt_ms,
                        },
                        pack,
                    )
                }
                _ => {
                    let v = p.vel.length() as f32;
                    (
                        AnimInput {
                            kind: if v > 0.05 { 1 } else { 0 },
                            speed: v,
                            moved: v * dt,
                            room_height: pax::OUTSIDE_ROOM,
                            dt_ms,
                            ..Default::default()
                        },
                        None,
                    )
                }
            };
            let p = &mut self.people[i];
            let ev = p.anim.advance(&p.ty.omsi, &input);
            // a foot down inside a vehicle: the link's step sound (outside there are none)
            if let (true, Some((bus, pack))) = (ev.step, footstep) {
                self.footfalls.push(ambience::Footfall { position: p.position, inside: true, own_bus: bus == BusId::Player, pack: Some(pack) });
            }
        }
    }

    /// Skin the people due for a new pose and push transforms to the renderer. Near people
    /// are posed every frame, far ones every few frames and people out of view rarely; the
    /// posing and skinning run in parallel.
    pub fn sync(&mut self, renderer: &Renderer, scene: &mut Scene, camera: DVec3) {
        for inst in self.hidden.drain(..) {
            renderer.set_params(scene, inst, &[], false, &[]);
        }
        let started = std::time::Instant::now();
        self.sync_frame = self.sync_frame.wrapping_add(1);
        let eye = self.eye;
        let from = eye.map(|e| e.pos).unwrap_or(camera);
        // synced only now and then (offscreen snapshots): everybody is posed afresh
        let all = self.time - self.last_sync > 0.12;
        let sdt = (self.time - self.last_sync).clamp(0.0, 0.5) as f32;
        self.last_sync = self.time;
        let mut due: Vec<bool> = Vec::with_capacity(self.people.len());
        for (k, p) in self.people.iter_mut().enumerate() {
            p.since_posed = p.since_posed.saturating_add(1);
            let d = p.position + DVec3::Z * 0.9 - from;
            let dist = d.length();
            let visible = match eye {
                Some(e) => dist < 4.0 || d.dot(e.fwd) / dist.max(1e-3) > e.cos_half - 0.15,
                None => true,
            };
            // everybody the eye can make out is posed every frame: a pose every other
            // frame at 12-30 m moved walkers in steps and made planted feet shiver
            // (within 30 m everybody, seen or not: the mirrors show the people behind the
            // bus, who were posed every twelfth frame and moved in jerks there)
            let every = if dist < 30.0 {
                1
            } else if !visible {
                12
            } else if dist < 45.0 {
                1
            } else if dist < 90.0 {
                2
            } else if dist < 160.0 {
                3
            } else {
                6
            };
            let every = if p.vel.length_squared() < 1e-4 && dist > 20.0 { every * 2 } else { every };
            // spread the far ones over the frames
            let turn = (self.sync_frame + k as u32) % every == 0;
            due.push(
                !p.skinned
                    || all
                    || (p.since_posed >= every && (turn || p.since_posed >= 2 * every)),
            );
        }
        let n_due = due.iter().filter(|d| **d).count();
        let pose_one = |p: &mut Person| {
            let Person { anim, ty, skins, skin_bones, pose_changed, .. } = p;
            *pose_changed = false;
            let bones = omsi_sim::human::slots_from_omsi(&anim.bones(&ty.omsi));
            if bones.iter().any(|b| !b.is_finite()) && !skins.is_empty() {
                // keep the last good mesh (the rest pose would be the file's T-pose)
                return;
            }
            // (the same bones as the mesh was made with: nothing to skin or upload)
            if skins.len() == ty.meshes.len() && skin_bones.as_ref().is_some_and(|b| b.iter().zip(&bones).all(|(a, c)| a.abs_diff_eq(*c, 1e-6))) {
                return;
            }
            skins.resize_with(ty.meshes.len(), Default::default);
            for (k, m) in ty.meshes.iter().enumerate() {
                let (pos, nrm) = &mut skins[k];
                skin(m, &bones, pos, nrm);
            }
            *skin_bones = Some(bones);
            *pose_changed = true;
        };
        // a handful is quicker on this thread than handed to the pool
        if n_due >= 8 {
            self.people
                .par_iter_mut()
                .zip(due.par_iter())
                .with_min_len(2)
                .filter(|(_, go)| **go)
                .for_each(|(p, _)| pose_one(p));
        } else {
            self.people
                .iter_mut()
                .zip(&due)
                .filter(|(_, go)| **go)
                .for_each(|(p, _)| pose_one(p));
        }
        let upload = std::time::Instant::now();
        for (p, &go) in self.people.iter_mut().zip(&due) {
            if go {
                if p.pose_changed || !p.skinned {
                    for (k, (id, _)) in p.meshes.iter().enumerate() {
                        if let Some((pos, nrm)) = p.skins.get(k) {
                            renderer.update_mesh(scene, *id, pos, nrm, &p.ty.meshes[k].data.uvs);
                        }
                    }
                }
                p.skinned = true;
                p.since_posed = 0;
                p.posed_at = (p.position, p.heading);
            }
            // riders go with their bus; on the ground a mesh not posed this frame goes on
            // with the body too (left where it was posed, a far walker moved in jerks -
            // its feet slide a few centimetres instead, which nobody sees at that distance)
            let (at, heading) = match (p.puppet, p.place) {
                (_, Place::Ground) if go => p.posed_at,
                _ => (p.position, p.heading),
            };
            // (riders with the tilt of their floor)
            let tilt = if matches!(p.place, Place::Bus(..)) { p.tilt } else { Mat4::IDENTITY };
            let xf = tilt * Mat4::from_rotation_z((-heading).to_radians() as f32);
            let lit_to = if matches!(p.place, Place::Bus(..)) { p.interior } else { 0.0 };
            p.lit += (lit_to - p.lit) * (sdt / 0.4).min(1.0);
            for (_, inst) in &p.meshes {
                renderer.set_transform(scene, *inst, at, xf);
                renderer.set_interior(scene, *inst, p.lit * 0.5);
            }
            if self.avatar_hidden.contains_key(&p.id) && omsi_cfg::env::var_os("OMSI_DEBUG_FOOT").is_some() && self.sync_frame % 30 == 0 {
                log::info!("avatar drawn at ({:.2}, {:.2}, {:.2}) heading {:.0} place {:?} go {}", at.x, at.y, at.z, heading, matches!(p.place, Place::Ground), go);
            }
            if let Some(hide) = self.avatar_hidden.get_mut(&p.id) {
                // (the first-person view: the avatar's own body out of the picture; set
                // every frame, the posing would show it again)
                for (_, inst) in &p.meshes {
                    renderer.set_params(scene, *inst, &[], !*hide, &[]);
                }
            }
            if let Some(t) = self.trace.as_mut() {
                // OMSI_TRACE_PAX: where the mesh is drawn and where its ankles are, per frame
                if (at - from).length() < 40.0 {
                    use std::io::Write;
                    let a = |k: usize| at + (xf.transform_vector3(p.ankles[k])).as_dvec3();
                    let (l, r) = (a(0), a(1));
                    let _ = writeln!(
                        t,
                        "{:.4},{},{},{},{},{:.4},{:.4},{:.4},{:.2},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.3},{:.3}",
                        self.time,
                        p.id,
                        p.state.name(),
                        matches!(p.place, Place::Ground) as u8,
                        go as u8,
                        at.x,
                        at.y,
                        at.z,
                        heading,
                        l.x,
                        l.y,
                        l.z,
                        r.x,
                        r.y,
                        r.z,
                        p.vel.x,
                        p.vel.y
                    );
                }
            }
        }
        // OMSI_CHECK_TPOSE=1: everybody drawn with the arms out (the file's rest pose): the
        // skinned mesh wider than 1.3 m from hand to hand
        if omsi_cfg::env::var_os("OMSI_CHECK_TPOSE").is_some() {
            for p in &self.people {
                let Some((pos, _)) = p.skins.first() else {
                    log::info!("t-pose? {} {}: never skinned", p.label(), p.state_name());
                    continue;
                };
                let (lo, hi) = pos.iter().fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(v.x), hi.max(v.x)));
                if hi - lo > 1.3 {
                    let pax = match &p.state {
                        State::Pax(x) => format!("pax_state {} speed {:.2} seat_h {:.2} room {:.2} st {}", x.pax_state, x.speed, x.seat_h, x.room, x.st),
                        _ => String::new(),
                    };
                    log::info!("t-pose: {} {} width {:.2} skinned {} {pax}", p.label(), p.state_name(), hi - lo, p.skinned);
                }
            }
        }
        self.pose_stats.0 += 1;
        self.pose_stats.1 += n_due;
        self.pose_stats.2 += started.elapsed().as_secs_f64() * 1000.0;
        self.pose_stats.3 += upload.elapsed().as_secs_f64() * 1000.0;
    }


}


#[derive(Debug, Clone, Copy, PartialEq)]
enum PuppetMode {
    /// The player on foot (or another player's walker): moved by the game, see `avatar`.
    Avatar,
}

/// A person the game moves itself (the player on foot).
#[derive(Debug, Clone, Copy)]
struct Puppet {
    mode: PuppetMode,
}

// ---------------------------------------------------------------------------------------
// Avatars: the player got up from the seat (`on_foot`), or another player walks about. The
// game moves them; the people's animation poses them - the gait and its feet on the
// ground, sitting down on a seat and getting up - so every change is eased, never a jump.

/// What the game wants of an avatar this frame.
#[derive(Debug, Clone, Copy)]
pub struct AvatarCmd {
    /// The feet (on foot), in the world.
    pub pos: DVec3,
    /// Facing (degrees, OMSI's).
    pub heading: f64,
    /// Velocity over the ground (m/s).
    pub vel: DVec2,
    /// How high the feet are over the ground (a jump).
    pub lift: f64,
    /// Sitting on this seat of this bus.
    pub seat: Option<(BusId, usize)>,
    /// Standing on a vehicle's floor at this height rather than on the ground (walking
    /// inside a bus: the feet stay on its floor, not reaching down to the road).
    pub floor: Option<f64>,
    /// Standing or walking inside this bus at this point of its cabin (bus frame): placed
    /// in the bus's frame as it is this frame, as its passengers are (a world point taken
    /// a frame earlier left the figure trembling behind the moving bus).
    pub aboard: Option<(BusId, Vec3)>,
}

/// A seat an avatar may take: which, in which bus.
#[derive(Debug, Clone, Copy)]
pub struct SeatSpot {
    pub bus: BusId,
    pub seat: usize,
}

impl Humans {
    /// Put avatar `key` where `cmd` says (made on its first call, of figure `kind`).
    pub fn avatar(&mut self, key: u32, world: &World, renderer: &Renderer, scene: &mut Scene, cmd: AvatarCmd, kind: u64) {
        let known = self.avatars.get(&key).copied().filter(|id| self.people.iter().any(|p| p.id == *id));
        if known.is_none() {
            let state = State::Idle;
            let n = self.types.len().max(1) as u64;
            let Some(i) = self.spawn_as(world, renderer, scene, cmd.pos, cmd.heading, state, Some((kind % n) as usize)) else { return };
            self.people[i].puppet = Some(Puppet { mode: PuppetMode::Avatar });
            self.avatars.insert(key, self.people[i].id);
        }
        // a seat taken is kept from the passengers; one left is theirs again
        let before = self.avatar_cmds.get(&key).and_then(|c| c.seat);
        if before != cmd.seat {
            if let Some((b, k)) = before {
                self.free_seat(b, k);
            }
            if let Some((b, k)) = cmd.seat {
                if let Some(t) = self.seats.get_mut(&b).and_then(|v| v.get_mut(k)) {
                    *t = true;
                }
            }
        }
        self.avatar_cmds.insert(key, cmd);
    }

    /// Take avatar `key` away.
    pub fn avatar_remove(&mut self, key: u32) {
        if let Some(c) = self.avatar_cmds.remove(&key) {
            if let Some((b, k)) = c.seat {
                self.free_seat(b, k);
            }
        }
        if let Some(id) = self.avatars.remove(&key) {
            if let Some(i) = self.people.iter().position(|p| p.id == id) {
                let p = self.people.swap_remove(i);
                self.retire(&p);
            }
        }
    }

    /// Draw avatar `key` or not (the first-person view looks out of its eyes).
    pub fn avatar_show(&mut self, key: u32, show: bool) {
        if let Some(id) = self.avatars.get(&key) {
            self.avatar_hidden.insert(*id, !show);
        }
    }

    /// Where avatar `key` is drawn: its feet, facing, and its eyes.
    pub fn avatar_body(&self, key: u32) -> Option<(DVec3, f64, DVec3)> {
        let id = self.avatars.get(&key)?;
        let p = self.people.iter().find(|p| p.id == *id)?;
        let rig = &p.ty.rig;
        let eye_h = (rig.head_top - 0.11 * rig.scale) as f64;
        let eye = match (p.place, self.avatar_cmds.get(&key).and_then(|c| c.seat)) {
            (Place::Bus(b, _), Some((_, k))) => {
                let bn = self.last_buses.iter().find(|x| x.id == b)?;
                let s = bn.cabin.seats.get(k)?;
                // sitting: the eyes over the hip, a little back
                let r = s.rot.to_radians();
                bn.world(s.pos + Vec3::new(-r.sin() * 0.05, -r.cos() * 0.05, (eye_h - rig.hip[0].z as f64) as f32 + 0.04))
            }
            _ => p.position + DVec3::new(0.0, 0.0, eye_h),
        };
        Some((p.position, p.heading, eye))
    }

    /// The seat nearest `at` with a door of its bus within `reach` of it (people and
    /// the other avatars' seats taken), among the buses of the last tick; `only` limits it
    /// to one bus.
    pub fn seat_near(&self, at: DVec3, reach: f64, only: Option<BusId>) -> Option<SeatSpot> {
        let mut best: Option<(f64, SeatSpot)> = None;
        for bn in &self.last_buses {
            if only.map(|o| o != bn.id).unwrap_or(false) {
                continue;
            }
            // the nearest door (entries and exits: any door will do to get in)
            let door = bn
                .cabin
                .entries
                .iter()
                .chain(bn.cabin.exits.iter())
                .map(|d| bn.world(d.outside))
                .min_by(|a, b| (*a - at).length().total_cmp(&(*b - at).length()));
            let Some(door) = door else { continue };
            let d = (door - at).truncate().length();
            if d > reach {
                continue;
            }
            let taken = self.seats.get(&bn.id);
            let seat = bn
                .cabin
                .seats
                .iter()
                .enumerate()
                .filter(|(k, s)| s.seated && !taken.and_then(|t| t.get(*k)).copied().unwrap_or(false))
                .min_by(|a, b| (bn.world(a.1.floor) - door).length().total_cmp(&(bn.world(b.1.floor) - door).length()))
                .map(|(k, _)| k);
            let Some(seat) = seat else { continue };
            if best.map(|b| d < b.0).unwrap_or(true) {
                best = Some((d, SeatSpot { bus: bn.id, seat }));
            }
        }
        best.map(|b| b.1)
    }

    /// Where the doors of a bus are now (outside, in the world).
    pub fn bus_doors(&self, bus: BusId) -> Vec<DVec3> {
        self.last_buses
            .iter()
            .find(|b| b.id == bus)
            .map(|bn| bn.cabin.entries.iter().chain(bn.cabin.exits.iter()).map(|d| bn.world(d.outside)).collect())
            .unwrap_or_default()
    }

    /// The door of vehicle `v` nearest its driver's seat (outside, in the world): where the
    /// driver gets in and out.
    pub fn vehicle_driver_door(&mut self, v: &VehicleInstance) -> Option<DVec3> {
        // (a van's own cab door first: its driver does not climb in through the sliding door)
        if let Some(d) = self.vehicle_cab_door(v) {
            return Some(d);
        }
        let cabin = self.cabin_for(v)?;
        let seat = cabin.data.driver_positions.first().map(|d| Vec3::from(d.pos)).unwrap_or(Vec3::new(-0.8, 4.5, 1.0));
        let door = cabin.entries.iter().chain(cabin.exits.iter()).min_by(|a, b| (a.outside - seat).truncate().length().total_cmp(&(b.outside - seat).truncate().length()))?;
        let trailers = part_frames(v, &cabin);
        Some(train_point(v.position, &v.body_rotation(), &trailers, door.outside))
    }

    /// A door of the driver's own beside the driver's seat (a van's or a coach's cab door:
    /// on the driver's side, level with the seat), in the world, outside.
    pub fn vehicle_cab_door(&mut self, v: &VehicleInstance) -> Option<DVec3> {
        let cabin = self.cabin_for(v)?;
        let seat = cabin.data.driver_positions.first().map(|d| Vec3::from(d.pos))?;
        let door = cabin
            .entries
            .iter()
            .chain(cabin.exits.iter())
            .filter(|d| d.outside.x * seat.x > 0.0 && (d.outside.y - seat.y).abs() < 1.5)
            .min_by(|a, b| (a.outside - seat).truncate().length().total_cmp(&(b.outside - seat).truncate().length()))
            .map(|d| d.outside);
        // a van or minibus (the W906: its cabin knows only the sliding door, the passengers'):
        // the driver's door beside the seat, which every such vehicle has
        let door = door.or_else(|| {
            let bb = v.ty.def.bounding_box?;
            (bb[1] < 8.5 && seat.x.abs() > 0.2).then(|| Vec3::new(seat.x.signum() * (bb[0] * 0.5 + bb[3] * seat.x.signum() + 0.45), seat.y, 0.0))
        })?;
        let trailers = part_frames(v, &cabin);
        Some(train_point(v.position, &v.body_rotation(), &trailers, door))
    }

    /// Put `ty` among the figures (once) and give its index: the player's own figure.
    pub fn type_index(&mut self, ty: Arc<HumanType>) -> usize {
        if let Some(i) = self.types.iter().position(|t| Arc::ptr_eq(t, &ty) || t.def.path == ty.def.path) {
            return i;
        }
        self.types.push(ty);
        self.types.len() - 1
    }

    /// Where the doors of vehicle `v` are now (outside, in the world), entries first.
    pub fn vehicle_doors(&mut self, v: &VehicleInstance) -> Vec<DVec3> {
        let Some(cabin) = self.cabin_for(v) else { return Vec::new() };
        let trailers = part_frames(v, &cabin);
        let rot = v.body_rotation();
        cabin.entries.iter().chain(cabin.exits.iter()).map(|d| train_point(v.position, &rot, &trailers, d.outside)).collect()
    }

    /// A walker inside bus `bus` moving from cabin point `local` by `step` (bus frame,
    /// metres): kept within a corridor round the cabin's own path network (the aisles,
    /// the door areas, the space by the driver) and on its floor. Gives the new cabin point
    /// and where that is in the world now.
    pub fn cabin_walk(&self, bus: BusId, local: Vec3, step: glam::Vec2) -> Option<(Vec3, DVec3)> {
        const WIDTH: f32 = 0.3;
        let bn = self.last_buses.iter().find(|b| b.id == bus)?;
        let pts = &bn.cabin.graph.points;
        let want = glam::Vec2::new(local.x + step.x, local.y + step.y);
        let mut best: Option<(f32, glam::Vec2, f32)> = None;
        for &(a, b, _) in &bn.cabin.links {
            let (Some(pa), Some(pb)) = (pts.get(a.max(0) as usize), pts.get(b.max(0) as usize)) else { continue };
            let (a2, b2) = (pa.truncate(), pb.truncate());
            let ab = b2 - a2;
            let t = if ab.length_squared() > 1e-6 { ((want - a2).dot(ab) / ab.length_squared()).clamp(0.0, 1.0) } else { 0.0 };
            let q = a2 + ab * t;
            let d = (want - q).length() + (local.z - (pa.z + (pb.z - pa.z) * t)).abs();
            if best.map(|x| d < x.0).unwrap_or(true) {
                best = Some((d, q, pa.z + (pb.z - pa.z) * t));
            }
        }
        if best.is_none() {
            for pt in pts {
                let d = (want - pt.truncate()).length();
                if best.map(|x| d < x.0).unwrap_or(true) {
                    best = Some((d, pt.truncate(), pt.z));
                }
            }
        }
        let (d, q, z) = best?;
        let xy = if d > WIDTH { q + (want - q) / d * WIDTH } else { want };
        // not through the seats and the driver's place: no nearer to one than 0.38 m
        // (walking away from one that close is let be)
        let from = local.truncate();
        let solid = bn.cabin.seats.iter().filter(|s| s.seated).map(|s| s.pos.truncate()).chain(bn.cabin.data.driver_positions.iter().map(|d| glam::Vec2::new(d.pos[0], d.pos[1])));
        for c in solid {
            let (dn, d0) = ((xy - c).length(), (from - c).length());
            if dn < 0.38 && dn < d0 {
                return Some((local, bn.world(local)));
            }
        }
        let l = Vec3::new(xy.x, xy.y, z);
        Some((l, bn.world(l)))
    }

    /// The doors of bus `bus`: the threshold in the cabin, where one stands outside (world),
    /// which side of the bus (+1 right) and whether it is open now.
    pub fn cabin_doors(&self, bus: BusId) -> Vec<(Vec3, DVec3, f32, bool)> {
        let Some(bn) = self.last_buses.iter().find(|b| b.id == bus) else { return Vec::new() };
        let (eo, xo) = bn.walk_open.as_ref().map(|w| (&w.0, &w.1)).unwrap_or((&bn.entry_open, &bn.exit_open));
        let entries = bn.cabin.entries.iter().enumerate().map(|(k, d)| (d, eo.get(k).copied().unwrap_or(false)));
        let exits = bn.cabin.exits.iter().enumerate().map(|(k, d)| (d, xo.get(k).copied().unwrap_or(false)));
        entries.chain(exits).map(|(d, open)| (d.inside, bn.world(d.outside), d.side, open)).collect()
    }

    /// The buses of the last tick within `r` of `at`, the own first.
    pub fn bus_ids_near(&self, at: DVec3, r: f64) -> Vec<BusId> {
        let mut v: Vec<(BusId, f64)> = self.last_buses.iter().map(|b| (b.id, (b.pos - at).truncate().length())).filter(|x| x.1 < r).collect();
        v.sort_by(|a, b| (a.0 != BusId::Player).cmp(&(b.0 != BusId::Player)).then(a.1.total_cmp(&b.1)));
        v.into_iter().map(|x| x.0).collect()
    }

    /// Where the cabin point `local` of bus `bus` is in the world now, and the bus's heading.
    pub fn cabin_world(&self, bus: BusId, local: Vec3) -> Option<(DVec3, f64)> {
        let bn = self.last_buses.iter().find(|b| b.id == bus)?;
        Some((bn.world(local), bn.heading))
    }

    /// The free seat of bus `bus` nearest the world point `at` (for a walker inside it).
    pub fn seat_nearest(&self, bus: BusId, at: DVec3, reach: f64) -> Option<usize> {
        let bn = self.last_buses.iter().find(|b| b.id == bus)?;
        let taken = self.seats.get(&bn.id);
        bn.cabin
            .seats
            .iter()
            .enumerate()
            .filter(|(k, s)| s.seated && !taken.and_then(|t| t.get(*k)).copied().unwrap_or(false))
            .map(|(k, s)| (k, (bn.world(s.floor) - at).truncate().length()))
            .filter(|(_, d)| *d < reach)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|x| x.0)
    }

    /// The cabin path point nearest seat `seat` of bus `bus` (where one stands up to).
    pub fn seat_stand(&self, bus: BusId, seat: usize) -> Option<Vec3> {
        let bn = self.last_buses.iter().find(|b| b.id == bus)?;
        let s = bn.cabin.seats.get(seat)?;
        // (on the seat's own deck: in a double-decker the nearest point in plan could be
        // the one straight above or below it)
        let d = |a: &Vec3| (a.truncate() - s.floor.truncate()).length() + (a.z - s.floor.z).abs() * 3.0;
        bn.cabin.graph.points.iter().copied().min_by(|a, b| d(a).total_cmp(&d(b)))
    }

    /// The cabin of vehicle `v` as the developer tools draw it, in the world now: the path
    /// points, the links between them (indices into the points) and the places (the
    /// `[passpos]` point, seated or standing).
    pub fn debug_cabin(&mut self, v: &VehicleInstance) -> Option<CabinDebug> {
        let cabin = self.cabin_for(v)?;
        let trailers = part_frames(v, &cabin);
        let rot = v.body_rotation();
        let world = |p: Vec3| train_point(v.position, &rot, &trailers, p);
        let mut roles = vec![PointRole::default(); cabin.graph.points.len()];
        for (n, d) in cabin.entries.iter().enumerate() {
            if let Some(r) = d.point.and_then(|p| roles.get_mut(p)) {
                r.entries.push((n, !d.sells, d.button));
            }
        }
        for (n, d) in cabin.exits.iter().enumerate() {
            if let Some(r) = d.point.and_then(|p| roles.get_mut(p)) {
                r.exits.push(n);
            }
        }
        for (i, next) in &cabin.link_marks {
            if let Some(r) = roles.get_mut(*i) {
                if *next {
                    r.link_next = true;
                } else {
                    r.link_prev = true;
                }
            }
        }
        let mut devices = Vec::new();
        for dp in &cabin.debug_points {
            if let Some(r) = dp.point.and_then(|p| roles.get_mut(p)) {
                match dp.kind {
                    "ticket_sale" => r.sale = true,
                    "stamper" => r.stamper = true,
                    _ => {}
                }
            }
            devices.push((world(dp.pos), dp.kind, dp.note.clone()));
        }
        Some(CabinDebug {
            points: cabin.graph.points.iter().map(|p| world(*p)).collect(),
            links: cabin.links.iter().filter(|l| l.0 >= 0 && l.1 >= 0).map(|l| (l.0 as usize, l.1 as usize)).collect(),
            seats: cabin.seats.iter().map(|s| (world(s.pos), s.seated)).collect(),
            roles,
            sections: cabin.point_section.clone(),
            local: cabin.point_local.clone(),
            devices,
        })
    }

    /// Cabin point `local` of vehicle `v` in the world (before the buses' first tick).
    pub fn vehicle_cabin_world(&mut self, v: &VehicleInstance, local: Vec3) -> Option<DVec3> {
        let cabin = self.cabin_for(v)?;
        let trailers = part_frames(v, &cabin);
        Some(train_point(v.position, &v.body_rotation(), &trailers, local))
    }

    /// Where the driver stands up in vehicle `v`'s cabin: the cabin's path point nearest the
    /// driver's seat (bus frame).
    pub fn driver_stand(&mut self, v: &VehicleInstance) -> Option<Vec3> {
        let cabin = self.cabin_for(v)?;
        let seat = cabin.data.driver_positions.first().map(|d| Vec3::from(d.pos)).unwrap_or(Vec3::new(-0.8, 4.5, 1.0));
        // the driver's position is the hip, half a metre over the cab floor: a double
        // decker's upper deck lies straight over the cab and was as near in plan, and the
        // driver who got up stood in the roof over the windscreen
        let d = |a: &Vec3| (a.truncate() - seat.truncate()).length() + (a.z - (seat.z - 0.5)).abs() * 3.0;
        cabin.graph.points.iter().copied().min_by(|a, b| d(a).total_cmp(&d(b)))
    }

    /// How many people are in (or boarding, riding, leaving) bus `bus`.
    pub fn people_in(&self, bus: BusId) -> usize {
        self.people.iter().filter(|p| matches!(p.place, Place::Bus(b, _) if b == bus) || p.state.bus() == Some(bus)).count()
    }

    /// Where bus `bus` stands (its origin), as of the last tick.
    pub fn bus_center(&self, bus: BusId) -> Option<DVec3> {
        self.last_buses.iter().find(|b| b.id == bus).map(|b| b.pos)
    }

    /// Is `bus` among the buses of the last tick?
    pub fn bus_here(&self, bus: BusId) -> bool {
        self.last_buses.iter().any(|b| b.id == bus)
    }

    /// The player's (or another player's) body on foot, animated as Omsi.exe animates its
    /// people: sitting on a seat (its hip on the `[passpos]`), walking or standing.
    fn animate_avatar(&mut self, i: usize, dt: f32, world: &World, buses: &[BusNow], bus_ix: &HashMap<BusId, usize>) {
        let id = self.people[i].id;
        let Some(key) = self.avatars.iter().find(|(_, v)| **v == id).map(|(k, _)| *k) else { return };
        let Some(cmd) = self.avatar_cmds.get(&key).copied() else { return };
        let dt_ms = dt * 1000.0;
        let seated = cmd.seat.and_then(|(b, k)| {
            let bn = bus_ix.get(&b).map(|x| &buses[*x])?;
            let s = bn.cabin.seats.get(k)?.clone();
            Some((b, s, bn))
        });
        let seatheight = self.people[i].ty.def.seat_height;
        let p = &mut self.people[i];
        let input = match seated {
            Some((b, s, bn)) => {
                // on the seat, in its bus's frame (set_task(7): the feet the human's seat
                // height under the seat point, facing the way the seat does)
                let l = if s.seated { s.pos - Vec3::Z * seatheight } else { s.pos };
                p.place = Place::Bus(b, l);
                p.lheading = s.rot as f64;
                p.position = bn.world(l);
                p.tilt = bn.tilt_at(l);
                p.heading = bn.heading_at(l) + p.lheading;
                p.interior = bn.interior;
                p.vel = DVec2::ZERO;
                p.activity = if s.seated { Activity::Sit } else { Activity::Stand };
                AnimInput { kind: if s.seated { 2 } else { 0 }, seat_height: s.height, room_height: pax::OUTSIDE_ROOM, dt_ms, ..Default::default() }
            }
            None if cmd.aboard.is_some_and(|(b, _)| bus_ix.contains_key(&b)) => {
                let (b, l) = cmd.aboard.unwrap();
                let bn = &buses[bus_ix[&b]];
                let bh = bn.heading_at(l);
                p.place = Place::Bus(b, l);
                p.lheading = wrap_heading(cmd.heading - bh);
                p.position = bn.world(l);
                p.tilt = bn.tilt_at(l);
                p.heading = cmd.heading;
                p.interior = bn.interior;
                p.vel = cmd.vel;
                let v = cmd.vel.length() as f32;
                p.activity = if v > 0.05 { Activity::Walk } else { Activity::Stand };
                AnimInput { kind: (v > 0.05) as u8, speed: v, moved: v * dt, room_height: pax::OUTSIDE_ROOM, dt_ms, ..Default::default() }
            }
            None => {
                p.place = Place::Ground;
                p.tilt = Mat4::IDENTITY;
                p.interior = 0.0;
                let ground = cmd.floor.unwrap_or_else(|| world.walk_height(cmd.pos.x, cmd.pos.y).unwrap_or(cmd.pos.z));
                let origin = DVec3::new(cmd.pos.x, cmd.pos.y, if cmd.floor.is_some() { ground } else { cmd.pos.z.max(ground) } + cmd.lift.max(0.0));
                p.position = origin;
                p.heading = cmd.heading;
                p.vel = cmd.vel;
                let v = cmd.vel.length() as f32;
                p.activity = if v > 0.05 { Activity::Walk } else { Activity::Stand };
                AnimInput { kind: (v > 0.05) as u8, speed: v, moved: v * dt, room_height: pax::OUTSIDE_ROOM, dt_ms, ..Default::default() }
            }
        };
        p.anim.advance(&p.ty.omsi, &input);
    }
}

// ---------------------------------------------------------------------------------------
// LAN play (see `lan_world`): a host keeps people around every player, tells the clients
// where they are and hands the waiting ones over to a client's bus; a client draws the
// host's people instead of its own and simulates only those who board its bus.

/// Where one of the host's people is this frame, as a client draws them.
#[derive(Debug, Clone, Copy)]
pub struct MirrorPose {
    pub pos: DVec3,
    pub heading: f64,
    pub vel: DVec2,
    pub activity: Activity,
    /// Aboard a timetable bus: (its id, the point of its frame, heading in its frame, the
    /// seat or standing place, if known).
    pub aboard: Option<(u64, Vec3, f64, Option<usize>)>,
    /// Waiting at a stop: (the stop object, the waiting place).
    pub waiting: Option<(i64, usize)>,
}

/// The bus id (`BusId::Ai`) another LAN player's bus has among the buses here: far above
/// the traffic's car ids.
pub fn remote_bus_id(player: u32) -> u64 {
    (1 << 40) | player as u64
}

/// The bus id (`BusId::Ai`) of a vehicle the player placed (`Player::uid`).
pub fn placed_bus_id(uid: u64) -> u64 {
    (2 << 40) | uid
}

/// The player whose bus `remote_bus_id` gave this id (None for a traffic bus).
pub fn remote_bus_player(bus: u64) -> Option<u32> {
    (bus >> 40 == 1).then_some((bus & 0xFFFF_FFFF) as u32)
}

/// One of the host's people as it tells the clients.
pub struct LanPerson {
    pub id: u32,
    pub ty: Arc<HumanType>,
    pub pos: DVec3,
    pub heading: f64,
    pub speed: f64,
    pub activity: Activity,
    pub aboard: Option<(u64, Vec3, f64, Option<usize>)>,
    pub waiting: Option<(i64, usize)>,
}

impl Humans {
    /// Is `p` further than `r` from us and from every other LAN player?
    fn far_from_players(&self, p: DVec3, r: f64) -> bool {
        (p - self.center).length() > r && self.lan_centers.iter().all(|c| (p - *c).length() > r)
    }

    /// The stops and pavements around the other players of a LAN session (host).
    fn populate_lan_centers(
        &mut self,
        world: &World,
        net: &Network,
        lights: Option<&dyn PedLights>,
        renderer: &Renderer,
        scene: &mut Scene,
    ) {
        if self.lan_centers.is_empty() {
            return;
        }
        let mine = self.center;
        for c in self.lan_centers.clone() {
            if (c - mine).length() < 150.0 {
                continue;
            }
            self.populate_with(world, Some(net), renderer, scene, c);
            self.populate_on_foot(world, net, lights, renderer, scene, 1.0);
        }
        self.center = mine;
    }

    /// Everybody within `radius` of `near` the clients may see (host): on foot, waiting at
    /// a stop, or aboard a timetable bus - not the riders of our own bus, which the others
    /// see from outside only.
    pub fn lan_people(&self, near: DVec3, radius: f64) -> Vec<LanPerson> {
        let r2 = radius * radius;
        self.people
            .iter()
            .filter(|p| p.puppet.is_none() && !p.remote)
            .filter(|p| (p.position - near).length_squared() < r2)
            .filter_map(|p| {
                let aboard = match p.place {
                    Place::Bus(BusId::Player, _) => return None,
                    Place::Bus(BusId::Ai(bus), l) => Some((
                        bus,
                        l,
                        p.lheading,
                        match &p.state {
                            State::Pax(x) if x.task == Task::SittingInBus => x.seat,
                            _ => None,
                        },
                    )),
                    Place::Ground => None,
                };
                let waiting = match &p.state {
                    State::Pax(x) if aboard.is_none() && x.task == Task::WaitingForBus => x.stop.zip(x.spot),
                    _ => None,
                };
                Some(LanPerson {
                    id: p.id,
                    ty: p.ty.clone(),
                    pos: p.position,
                    heading: p.heading,
                    speed: if aboard.is_some() { 0.0 } else { p.vel.length() },
                    activity: p.activity,
                    aboard,
                    waiting,
                })
            })
            .collect()
    }

    /// A client's bus takes these waiting people (host): those still waiting leave our
    /// world (they are the client's now); returns them. Somebody who has meanwhile walked
    /// up to another bus stays ours.
    pub fn hand_over(&mut self, player: u32, ids: &[u32]) -> Vec<u32> {
        let mut out = Vec::new();
        for id in ids {
            let Some(i) = self.people.iter().position(|p| p.id == *id) else {
                continue;
            };
            if !matches!(&self.people[i].state, State::Pax(x) if x.task == Task::WaitingForBus) || self.people[i].remote {
                continue;
            }
            // (still counted at their stop while that bus stands there, as the people
            // boarding a bus of ours are: see `handed`)
            if let Some(stop) = self.pax(i).and_then(|x| x.stop) {
                self.handed.push((stop, remote_bus_id(player)));
            }
            self.release(i);
            let p = self.people.swap_remove(i);
            self.retire(&p);
            out.push(*id);
        }
        out
    }

    /// Draw the host's people from now on (`on`), or simulate our own again. Everybody who
    /// is not getting on, riding or getting off our bus goes (the host's come instead; the
    /// host's copies cannot walk on by themselves).
    pub fn set_mirror(&mut self, on: bool) {
        if self.mirror == on {
            return;
        }
        self.mirror = on;
        let keep = |p: &Person| !p.remote && p.state.bus() == Some(BusId::Player);
        let mut i = 0;
        while i < self.people.len() {
            if keep(&self.people[i]) || self.people[i].puppet.is_some() {
                i += 1;
                continue;
            }
            self.release(i);
            let p = self.people.swap_remove(i);
            self.retire(&p);
        }
        // our own people from now on are numbered far above the host's (those riding with
        // us already too)
        if on {
            self.next_id = self.next_id.max(1 << 30);
            for p in self.people.iter_mut().filter(|p| p.puppet.is_none()) {
                p.id = self.next_id;
                self.next_id += 1;
            }
        }
        for s in self.stops.values_mut() {
            for t in s.taken.iter_mut() {
                *t = false;
            }
        }
        self.claims_out.clear();
        self.claimed.clear();
        self.mirror_wait.clear();
    }

    /// The human type of a file relative to a content root (`Humans/…/x.hum`).
    pub fn type_by_file(&self, file: &str) -> Option<usize> {
        let want = file.replace('\\', "/").to_ascii_lowercase();
        self.types.iter().position(|t| {
            t.def
                .path
                .to_string_lossy()
                .replace('\\', "/")
                .to_ascii_lowercase()
                .ends_with(&want)
        })
    }

    /// The file of a human type relative to its content root (`Humans/…/x.hum`).
    pub fn type_file(ty: &HumanType) -> String {
        let p = ty.def.path.to_string_lossy().replace('\\', "/");
        match p.to_ascii_lowercase().rfind("/humans/") {
            Some(k) => p[k + 1..].to_string(),
            None => p,
        }
    }

    /// One of the host's people appears here (client).
    pub fn mirror_add(
        &mut self,
        world: &World,
        renderer: &Renderer,
        scene: &mut Scene,
        id: u32,
        ty: usize,
        pose: &MirrorPose,
    ) -> bool {
        if self.people.iter().any(|p| p.id == id) {
            return false;
        }
        let state = State::Idle;
        let Some(i) =
            self.spawn_as(world, renderer, scene, pose.pos, pose.heading, state, Some(ty))
        else {
            return false;
        };
        self.next_id -= 1;
        let p = &mut self.people[i];
        p.id = id;
        p.anim = OmsiAnim::default();
        p.remote = true;
        self.mirror_set(id, pose);
        true
    }

    /// Where one of the host's people is this frame (client).
    pub fn mirror_set(&mut self, id: u32, pose: &MirrorPose) {
        let Some(p) = self.people.iter_mut().find(|p| p.id == id && p.remote) else {
            return;
        };
        p.position = pose.pos;
        p.heading = pose.heading;
        p.vel = pose.vel;
        p.activity = pose.activity;
        match pose.waiting {
            Some(w) => {
                self.mirror_wait.insert(id, w);
            }
            None => {
                self.mirror_wait.remove(&id);
            }
        }
        match pose.aboard {
            Some((bus, local, lheading, _)) => {
                p.place = Place::Bus(BusId::Ai(bus), local);
                p.lheading = lheading;
                p.vel = DVec2::ZERO;
            }
            None => p.place = Place::Ground,
        }
    }

    /// One of the host's people has gone (client).
    pub fn mirror_remove(&mut self, id: u32) {
        if let Some(i) = self.people.iter().position(|p| p.id == id && p.remote) {
            let p = self.people.swap_remove(i);
            self.retire(&p);
        }
        self.claimed.remove(&id);
        self.mirror_wait.remove(&id);
    }

    /// A remote person this frame: they stand where the host put them.
    fn mirror_want(&mut self, i: usize, _buses: &[BusNow]) -> Want {
        Want::stand(None, self.people[i].activity)
    }

    /// The type of one of our people (host).
    pub fn lan_people_by_id(&self, id: u32) -> Option<Arc<HumanType>> {
        self.people
            .iter()
            .find(|p| p.id == id && !p.remote)
            .map(|p| p.ty.clone())
    }

    /// Where the host's people are drawn (client; `OMSI_LAN_TRACE`).
    pub fn mirror_positions(&self) -> Vec<(u32, DVec3)> {
        self.people
            .iter()
            .filter(|p| p.remote && p.place == Place::Ground)
            .map(|p| (p.id, p.position))
            .collect()
    }

    /// Waiting people to ask the host for (client).
    pub fn take_claims(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.claims_out)
    }

    /// The host handed this waiting person over to our bus (client). (The people of a LAN
    /// game stay the host's: nothing is taken over.)
    pub fn grant(&mut self, id: u32) -> bool {
        self.claimed.remove(&id);
        let Some((stop, spot)) = self.mirror_wait.remove(&id) else { return false };
        let Some(i) = self.people.iter().position(|p| p.id == id && p.remote) else { return false };
        if !self.stops.contains_key(&stop) {
            return false;
        }
        // ours from now on: waiting at that place, for the bus that stands there (the first
        // listed, as Omsi.exe takes it without a line record)
        let sp = self.stops[&stop].spots.get(spot).cloned();
        let seatheight = self.people[i].ty.def.seat_height;
        let walk = 1.1 + (self.rand_f() as f32 * 2.0 - 1.0) * 0.2;
        let mut pax = Pax::new(walk, self.rand_f());
        pax.task = Task::WaitingForBus;
        pax.stop = Some(stop);
        // what a waiting person of ours has (sub_626044 and task 6): a destination drawn
        // from the stop (both sides load the same map) and a distance to ride without one;
        // with neither they would get off again at once (#813)
        let (dest, line) = self.draw_dest(stop);
        pax.dest = dest;
        pax.line = line;
        pax.ride_km = self.rand_f() as f32 * 19.0 + 1.0;
        pax.pos = self.people[i].position;
        pax.yaw = self.people[i].heading.to_radians();
        if let Some(sp) = sp {
            pax.spot = Some(spot);
            if let Some(t) = self.stops.get_mut(&stop).unwrap().taken.get_mut(spot) {
                *t = true;
            }
            if sp.height != 0.0 {
                pax.seat_h = sp.height;
                pax.pos = sp.pos - DVec3::Z * seatheight as f64;
                pax.pax_state = 2.0;
            }
            pax.yaw = sp.face.to_radians();
        }
        let p = &mut self.people[i];
        p.remote = false;
        p.state = State::Pax(Box::new(pax));
        true
    }

    /// The host's people waiting at the stop our bus is listed at (client): ask for them.
    fn claim_waiting(&mut self) {
        if !self.mirror {
            return;
        }
        let now = self.time;
        self.claimed.retain(|_, t| now - *t < 10.0);
        let at: Vec<i64> = self
            .stops
            .iter()
            .filter(|(_, s)| s.buses.iter().any(|b| b.0 == BusId::Player))
            .map(|(id, _)| *id)
            .collect();
        if at.is_empty() {
            return;
        }
        for (id, (stop, _)) in &self.mirror_wait {
            if at.contains(stop) && !self.claimed.contains_key(id) {
                self.claimed.insert(*id, now);
                self.claims_out.push(*id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Off a bus 3 m from the pavement's path (along y), walking onto it at 1.1 m/s: hardly
    /// faster over any quarter of a second (pulled by the corridor all the way, over 1.5
    /// m/s), and on the path in the end.
    #[test]
    fn off_a_bus_people_walk_onto_the_pavement_at_their_pace() {
        // (a band of 0.1 m: the corridor holds once they are on the path)
        const OFF_A_BUS: f64 = 0.1;
        let walk = |held: bool| {
            let pace = 1.1;
            let mut w = Walker::new(DVec2::new(3.0, 0.0), 0.25, 0);
            let params = CrowdParams::default();
            let dt = 1.0 / 60.0;
            let mut track = vec![w.pos];
            for _ in 0..360 {
                let s = w.pos.y.max(0.0);
                let (a, b) = (DVec2::new(0.0, s - 2.0), DVec2::new(0.0, s + 2.5));
                w.want = (DVec2::new(0.0, s + 1.3) - w.pos).normalize_or_zero() * pace;
                let c = Some((a, b, 0.5));
                w.corridor = if held { c } else { path_corridor_band(w.pos, c, OFF_A_BUS) };
                crowd::step(std::slice::from_mut(&mut w), &[], &params, dt);
                track.push(w.pos);
            }
            let fastest = track.windows(16).map(|k| (k[15] - k[0]).length() / (15.0 * dt)).fold(0.0, f64::max);
            (fastest, w.pos)
        };
        let (fastest, end) = walk(false);
        assert!(fastest < 1.3, "{fastest:.2} m/s");
        assert!(end.x.abs() < 0.55, "{end:?}");
        let held = walk(true).0;
        assert!(held > 1.5, "{held:.2} m/s");
        // on the path the corridor holds again
        assert!(path_corridor_band(DVec2::new(0.55, 0.0), Some((DVec2::ZERO, DVec2::Y, 0.5)), OFF_A_BUS).is_some());
        assert!(path_corridor_band(DVec2::new(3.0, 0.0), Some((DVec2::ZERO, DVec2::Y, 0.5)), OFF_A_BUS).is_none());
    }

    #[test]
    fn map_humans_load_nested_paths_and_preserve_weights() {
        let root = std::env::temp_dir().join(format!(
            "omsi-map-human-paths-{}", std::process::id()
        ));
        let nested = root.join("Humans/JP_Test/Child_1");
        std::fs::create_dir_all(&nested).unwrap();
        // Synthetic definitions: no original passenger assets are required.
        std::fs::write(nested.join("Child_1.hum"), "[model]\nmodel.cfg\n").unwrap();
        std::fs::write(nested.join("model.cfg"), "").unwrap();
        let other = root.join("Humans/Other");
        std::fs::create_dir_all(&other).unwrap();
        std::fs::write(other.join("Man.hum"), "[model]\nmodel.cfg\n").unwrap();
        std::fs::write(other.join("model.cfg"), "").unwrap();
        let list = vec![
            "humans\\jp_test\\child_1\\child_1.hum".into(),
            "Humans/JP_Test/Child_1/Child_1.hum".into(),
            "JP_Test/Child_1/Child_1.hum".into(),
            "Humans/JP_Test/Missing.hum".into(),
        ];
        let picked = map_human_types(&root, &list);
        assert_eq!(picked.len(), 3);
        assert!(Arc::ptr_eq(&picked[0], &picked[1]));
        assert!(Arc::ptr_eq(&picked[1], &picked[2]));
        // (case-blind: a case-insensitive disk keeps the list's own spelling; and
        // separator-blind: on Windows `nested` keeps the slashes it was joined with, while
        // the resolved path is built with backslashes)
        let lower = |p: &Path| p.to_string_lossy().to_lowercase().replace('\\', "/");
        assert!(picked.iter().all(|t| lower(&t.def.path).starts_with(&lower(&nested))));
        assert!(map_human_types(&root, &["Humans/Missing/None.hum".into()]).is_empty());
        std::fs::remove_dir_all(root).unwrap();
    }

    /// Berlin 1991's pack: full fare, short haul, day ticket (adults), and two reduced
    /// fares for 6..13.
    fn berlin_91() -> omsi_content::tickets::TicketPack {
        let t = |name: &str, age: (i32, i32), day: bool, p: f32| omsi_content::tickets::Ticket {
            name: name.into(),
            age_min: age.0,
            age_max: age.1,
            day_ticket: day,
            probability: p,
            ..Default::default()
        };
        omsi_content::tickets::TicketPack {
            stamper_prop: 0.3,
            ticketbuy_prop: 0.2,
            tickets: vec![
                t("Fahrschein", (14, 200), false, 1.0),
                t("Kurzstrecke", (14, 200), false, 0.4),
                t("Tageskarte", (14, 200), true, 0.2),
                t("Ermaessigt", (6, 13), false, 1.0),
                t("Kurzstrecke Erm", (6, 13), false, 0.4),
            ],
            ..Default::default()
        }
    }

    #[test]
    fn seats_counted_by_the_scripts_numbers() {
        let seat = |omsi_seat: usize| Seat { pos: Vec3::ZERO, floor: Vec3::ZERO, rot: 0.0, seated: true, height: 0.45, omsi_seat, switch_var: None, taken_var: None, group: 0 };
        // the driver's place is seat 0, a second section's numbers follow the first's
        let seats = [seat(1), seat(2), seat(4), seat(6)];
        assert_eq!(seat_numbers(&seats, [0, 2, 2, 3].into_iter()), [0, 1, 0, 0, 2, 0, 1]);
        assert!(seat_numbers(&[], [0].into_iter()).is_empty());
    }

    #[test]
    fn tickets_by_age_and_time() {
        let mut h = Humans::new(Path::new("/nonexistent"));
        h.tickets = Some(Arc::new(berlin_91()));
        let count = |h: &mut Humans, age: f32| {
            let mut n = [0usize; 5];
            for _ in 0..4000 {
                n[h.pick_ticket(age).unwrap()] += 1;
            }
            n
        };
        // an adult (OMSI's default age of 40) never gets a reduced fare, a child only those
        h.time_of_day = 9.0 * 3600.0;
        let adult = count(&mut h, 40.0);
        assert_eq!(adult[3] + adult[4], 0);
        assert!(adult[2] > 300, "{adult:?}");
        let child = count(&mut h, 10.0);
        assert_eq!(child[0] + child[1] + child[2], 0);
        // day tickets sell best at 9:00, little early in the morning and late at night
        h.time_of_day = 1.0 * 3600.0;
        let early = count(&mut h, 40.0);
        assert!(early[2] * 4 < adult[2], "{early:?} vs {adult:?}");
        assert!(day_ticket_factor(9.0 * 3600.0) > 0.99);
        assert!(day_ticket_factor(0.0) < 0.01);
        assert!((day_ticket_factor(20.0 * 3600.0) - (1.0 - 39_600.0 / 56_376.0) as f32).abs() < 1e-3);
        // nobody in the age range: no ticket
        assert_eq!(h.pick_ticket(3.0), None);
    }

    fn lane(points: Vec<DVec3>, kind: LaneKind) -> omsi_sim::traffic::Lane {
        omsi_sim::traffic::LaneBuilder::polyline(points, kind, 2.5)
    }

    #[test]
    fn pavement_corners_are_joined_and_routed() {
        // an L of pavement: the two paths meet at a right angle, which the road network's
        // heading rule leaves unlinked
        let mut net = Network::default();
        net.lanes.push(lane(
            vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(0.0, 20.0, 0.0)],
            LaneKind::Sidewalk,
        ));
        net.lanes.push(lane(
            vec![DVec3::new(0.5, 20.3, 0.0), DVec3::new(30.0, 20.3, 0.0)],
            LaneKind::Sidewalk,
        ));
        net.lanes.push(lane(
            vec![DVec3::new(-5.0, 10.0, 0.0), DVec3::new(5.0, 10.0, 0.0)],
            LaneKind::Street,
        ));
        net.link(1.5);
        assert!(
            net.lanes[0].next.is_empty(),
            "the road rule does not join the corner"
        );
        let ped = PedNet::build(&net);
        // the corner is one junction: walking up the first path goes on round it
        let up = Leg { lane: 0, a: 5.0, b: 20.0 };
        let corner = ped.end_node(&net, &up).unwrap();
        assert!(ped.out[corner].iter().any(|&(l, fwd)| l == 1 && fwd), "{:?}", ped.out[corner]);
        // a dead end turns round
        let n = ped
            .end_node(
                &net,
                &Leg {
                    lane: 1,
                    a: 0.0,
                    b: net.lanes[1].length(),
                },
            )
            .unwrap();
        let turn = ped.next_leg(&net, n, 1, 7).unwrap();
        assert_eq!(turn.lane, 1);
        assert!(turn.a > turn.b);
    }

    #[test]
    fn crossings_of_a_pavement_path_are_found() {
        let mut net = Network::default();
        net.lanes.push(lane(
            vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(0.0, 8.0, 0.0)],
            LaneKind::Sidewalk,
        ));
        net.lanes.push(lane(
            vec![DVec3::new(-30.0, 4.0, 0.0), DVec3::new(30.0, 4.0, 0.0)],
            LaneKind::Street,
        ));
        net.link(1.5);
        let mut ped = PedNet::build(&net);
        let x = ped.crossings(&net, 0).to_vec();
        assert_eq!(x.len(), 1);
        assert!((x[0] - DVec2::new(0.0, 4.0)).length() < 1e-6);
    }

    /// An articulated bus: the front section's cabin and the rear section's (which only has
    /// exits and a seat) become one network through the joint, numbered front first, and a
    /// walk through the bent joint moves on without a jump.
    #[test]
    fn articulated_cabins_are_joined_through_the_bellows() {
        let dir = std::env::temp_dir().join(format!("omsi-humans-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let write = |name: &str, text: &str| std::fs::write(dir.join(name), text).unwrap();
        // front: door 0 at the front right, exit 1 in the middle, link to the rear at 3
        write("paths_a.cfg", "[pathpnt]\n1.2\n4\n0.4\n[pathpnt]\n0\n4\n0.5\n[pathpnt]\n0\n0\n0.5\n[pathpnt]\n0\n-4.2\n0.6\n[pathpnt]\n1.2\n0\n0.4\n[pathlink]\n0\n1\n[pathlink]\n1\n2\n[pathlink]\n2\n3\n[pathlink]\n2\n4\n");
        write(
            "cabin_a.cfg",
            "[entry]\n0\n[exit]\n4\n[linkToPrevVeh]\n3\n[passpos]\n-0.5\n2\n1.0\n0.45\n0\n",
        );
        // rear: only exits, a seat and the link to the front at point 0
        write("paths_b.cfg", "[pathpnt]\n0\n3.6\n0.6\n[pathpnt]\n0\n0\n0.6\n[pathpnt]\n1.2\n0\n0.4\n[pathpnt]\n0\n-2\n0.6\n[pathlink]\n0\n1\n[pathlink]\n1\n2\n[pathlink]\n1\n3\n");
        write(
            "cabin_b.cfg",
            "[exit]\n2\n[linkToNextVeh]\n0\n[passpos]\n-0.5\n-2\n1.1\n0.45\n0\n",
        );
        let def = |cabin: &str, paths: &str| omsi_vehicle::Vehicle {
            path: dir.join("bus.bus"),
            passenger_cabin: Some(cabin.into()),
            paths: Some(paths.into()),
            bounding_box: Some([2.5, 9.0, 3.0, 0.0, 0.0, 1.5]),
            ..Default::default()
        };
        let (front, rear) = (
            def("cabin_a.cfg", "paths_a.cfg"),
            def("cabin_b.cfg", "paths_b.cfg"),
        );
        // couplings: the front's at y -4.3, the rear's own at y 4.0
        let (back, own) = (Vec3::new(0.0, -4.3, 0.3), Vec3::new(0.0, 4.0, 0.3));
        let offset = back - own;
        let cabin =
            Cabin::load_train(&[(&front, Vec3::ZERO, f32::INFINITY), (&rear, offset, back.y)])
                .expect("cabin");
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(cabin.parts.len(), 2);
        assert_eq!(cabin.graph.points.len(), 9);
        assert_eq!(
            (cabin.entries.len(), cabin.exits.len(), cabin.seats.len()),
            (1, 2, 2)
        );
        // exit 1 is the rear section's door, where the rear file puts it
        assert!(
            (cabin.exits[1].inside - Vec3::new(1.2, -8.3, 0.4)).length() < 1e-4,
            "{:?}",
            cabin.exits[1].inside
        );
        // the seat in the rear is reached from the front door through the joint, along the
        // routing tables Omsi.exe builds over the joined network
        let seat = &cabin.seats[1];
        assert!((seat.pos.y + 10.3).abs() < 1e-4);
        let all: Vec<Option<usize>> = (0..cabin.graph.points.len()).map(Some).collect();
        let to = cabin.omsi_nearest(seat.floor, &all, false, false, None, None).unwrap();
        let mut at = cabin.entries[0].point.unwrap();
        let mut route = vec![cabin.graph.points[at]];
        while at != to {
            at = cabin.route_next(at, to).expect("a way on").0;
            route.push(cabin.graph.points[at]);
            assert!(route.len() < 20, "{route:?}");
        }
        assert!(
            route.iter().any(|p| (p.y + 4.2).abs() < 1e-4)
                && route.iter().any(|p| (p.y + 4.7).abs() < 1e-4),
            "{route:?}"
        );
        // and the nearest exit from there is the rear one
        let exit = cabin.omsi_nearest(seat.floor, &cabin.exit_points(), false, false, None, None);
        assert_eq!(exit, cabin.exits[1].point);
        // the rear section bent 30 degrees about the coupling: walking down the aisle moves on
        // smoothly, and the frames agree with the sections away from the joint
        let lead_rot = Mat4::IDENTITY;
        let bent = 30.0f64;
        let rot = Mat4::from_rotation_z((-bent).to_radians() as f32);
        let pos = back.as_dvec3() - rot.transform_point3(own).as_dvec3();
        let frames = [PartFrame {
            pos,
            rot,
            heading: bent,
            offset,
            joint_y: back.y,
            half: DVec2::new(1.25, 4.5),
            centre: DVec2::ZERO,
        }];
        // beside the aisle the two frames disagree by 0.31 m at the joint itself
        let at_joint = Vec3::new(0.6, back.y, 0.5);
        let rear_frame = pos + rot.transform_point3(at_joint - offset).as_dvec3();
        assert!((rear_frame - at_joint.as_dvec3()).length() > 0.3);
        let mut last = train_point(DVec3::ZERO, &lead_rot, &frames, Vec3::new(0.6, 0.0, 0.5));
        for k in 1..=100 {
            let y = -(k as f32) * 0.1;
            let p = train_point(DVec3::ZERO, &lead_rot, &frames, Vec3::new(0.6, y, 0.5));
            assert!(
                (p - last).length() < 0.14,
                "a jump of {:.3} m at y {y}",
                (p - last).length()
            );
            last = p;
        }
        let ahead = train_point(DVec3::ZERO, &lead_rot, &frames, Vec3::new(1.0, -1.0, 0.5));
        assert!((ahead - DVec3::new(1.0, -1.0, 0.5)).length() < 1e-4);
        let behind_joint = Vec3::new(1.0, -9.0, 0.5);
        let p = train_point(DVec3::ZERO, &lead_rot, &frames, behind_joint);
        assert!(
            (p - (pos + rot.transform_point3(behind_joint - offset).as_dvec3())).length() < 1e-4
        );
        assert!((train_heading(0.0, &frames, behind_joint) - bent).abs() < 1e-9);
        assert!(
            (train_heading(0.0, &frames, Vec3::new(0.0, back.y, 0.5)) - bent * 0.5).abs() < 1e-9
        );
    }

    /// The SD200's footsteps as its paths.cfg gives them to the links (#311): the stairs
    /// sound as stairs, the front of the upper deck as its own floor, the aisle below as
    /// the plain floor.
    #[test]
    fn footsteps_come_from_the_links_step_sound_pack() {
        let root = omsi_cfg::env::var_os("OMSI_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("../../../OMSI 2 Original"));
        let bus = root.join("Vehicles/MAN_SD200/MAN_SD80.bus");
        if !bus.exists() {
            eprintln!("skipped: no {}", bus.display());
            return;
        }
        let def = omsi_vehicle::Vehicle::load(&bus).expect("SD200");
        let cabin = Cabin::load_train(&[(&def, Vec3::ZERO, f32::INFINITY)]).expect("cabin");
        // the link nearest the point, and its pack
        let first = |p: Vec3| {
            let pts = &cabin.graph.points;
            let d = |l: &(i32, i32, bool)| {
                let (a, b) = (pts[l.0 as usize], pts[l.1 as usize]);
                let ab = b - a;
                let t = ((p - a).dot(ab) / ab.length_squared().max(1e-6)).clamp(0.0, 1.0);
                (a + ab * t - p).length()
            };
            let l = (0..cabin.links.len()).min_by(|&x, &y| d(&cabin.links[x]).total_cmp(&d(&cabin.links[y])))?;
            cabin.link_pack[l].map(|k| cabin.step_packs[k][0].to_ascii_lowercase())
        };
        assert_eq!(first(Vec3::new(-0.89, -1.61, 1.63)).as_deref(), Some("step_st_01.wav"), "the rear stairs");
        assert_eq!(first(Vec3::new(0.0, 4.35, 2.5)).as_deref(), Some("step_ov_01.wav"), "the upper deck's front");
        assert_eq!(first(Vec3::new(0.0, 0.84, 0.57)).as_deref(), Some("step_01.wav"), "the aisle below");
    }

    /// The SD202's cabin: the stairs down from the upper deck end beside the rear exits, and
    /// the walk from up there to an exit goes down the stairs.
    #[test]
    fn double_decker_exits_are_reached_down_the_stairs() {
        let root = omsi_cfg::env::var_os("OMSI_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("../../../OMSI 2 Original"));
        let bus = root.join("Vehicles/MAN_SD202/MAN_D92.bus");
        if !bus.exists() {
            eprintln!("skipped: no {}", bus.display());
            return;
        }
        let def = omsi_vehicle::Vehicle::load(&bus).expect("SD202");
        let cabin = Cabin::load_train(&[(&def, Vec3::ZERO, f32::INFINITY)]).expect("cabin");
        assert_eq!(cabin.exits.len(), 2);
        let exit = &cabin.exits[1];
        assert!(
            (exit.wait - Vec3::new(0.806, -1.26, 0.505)).length() < 1e-3,
            "{:?}",
            exit.wait
        );
        // from the upper deck the routing tables lead down both flights to the exit
        let all: Vec<Option<usize>> = (0..cabin.graph.points.len()).map(Some).collect();
        let upstairs = cabin.omsi_nearest(Vec3::new(0.0, -1.8, 2.46), &all, false, false, None, None).unwrap();
        assert!((cabin.graph.points[upstairs].z - 2.46).abs() < 0.1);
        let to = exit.point.unwrap();
        let mut at = upstairs;
        let mut route = vec![cabin.graph.points[at]];
        while at != to {
            at = cabin.route_next(at, to).expect("a way down").0;
            route.push(cabin.graph.points[at]);
            assert!(route.len() < 60, "{route:?}");
        }
        assert!(
            route.iter().any(|p| (p.z - 1.82).abs() < 0.01)
                && route.iter().any(|p| (p.z - 1.205).abs() < 0.01),
            "{route:?}"
        );
    }

    #[test]
    fn legs_run_both_ways() {
        let mut net = Network::default();
        net.lanes.push(lane(
            vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(0.0, 10.0, 0.0)],
            LaneKind::Sidewalk,
        ));
        let back = Leg {
            lane: 0,
            a: 8.0,
            b: 2.0,
        };
        let (p, h) = back.at(&net, 1.0);
        assert!((p.y - 7.0).abs() < 1e-6);
        assert!((h - 180.0).abs() < 1e-6);
        assert!((back.project(&net, DVec3::new(0.3, 5.0, 0.0), 2.5) - 3.0).abs() < 0.11);
    }

    #[test]
    fn doors_open_falls_back_when_exit_vars_are_undeclared() {
        let dir = std::env::temp_dir().join(format!("omsi-doors-open-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("test.bus"),
            "[model]\nmodel.cfg\n[varnamelist]\n1\nvars.txt\n[script]\n1\nmain.osc\n",
        )
        .unwrap();
        std::fs::write(dir.join("model.cfg"), "").unwrap();
        // Front door leaf 0 uses PAX_Entry0_Open. Rear door (door_2) has no PAX_Exit0_Open in varlist.
        std::fs::write(dir.join("vars.txt"), "door_0\ndoor_1\ndoor_2\nPAX_Entry0_Open\n").unwrap();
        std::fs::write(dir.join("main.osc"), "{init}\n{end}\n").unwrap();

        let ty = std::sync::Arc::new(omsi_sim::VehicleType::load(&dir, &dir.join("test.bus")).unwrap());
        let mut v = VehicleInstance::new(ty, omsi_sim::VehicleHost::new(Default::default()));

        // Initially both entries and exit closed
        let (e, x) = Humans::doors_open(&v, 2, 1);
        assert_eq!(e, vec![false, false]);
        assert_eq!(x, vec![false]);

        // Front door leaf 0 opens via PAX_Entry0_Open
        v.set_var("PAX_Entry0_Open", 1.0);
        let (e, x) = Humans::doors_open(&v, 2, 1);
        assert_eq!(e, vec![true, false]);
        assert_eq!(x, vec![false]);

        // Rear door leaf 2 opens (falls back to door_2 since PAX_Exit0_Open is not in varlist)
        v.set_var("door_2", 1.0);
        let (e, x) = Humans::doors_open(&v, 2, 1);
        assert_eq!(e, vec![true, false]);
        assert_eq!(x, vec![true]);

        // Front door leaf 1 opens via door_1 fallback
        v.set_var("door_1", 1.0);
        let (e, x) = Humans::doors_open(&v, 2, 1);
        assert_eq!(e, vec![true, true]);
        assert_eq!(x, vec![true]);

        std::fs::remove_dir_all(&dir).ok();
    }


    #[test]
    fn doors_open_reads_pax_vars_the_script_writes_without_declaring() {
        let dir = std::env::temp_dir().join(format!("omsi-doors-open-undeclared-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("test.bus"),
            "[model]\nmodel.cfg\n[varnamelist]\n1\nvars.txt\n[script]\n1\nmain.osc\n",
        )
        .unwrap();
        std::fs::write(dir.join("model.cfg"), "").unwrap();
        std::fs::write(dir.join("vars.txt"), "door_0\n").unwrap();
        std::fs::write(dir.join("main.osc"), "{frame}\n1 (S.L.PAX_Entry0_Open)\n{end}\n").unwrap();

        let ty = std::sync::Arc::new(omsi_sim::VehicleType::load(&dir, &dir.join("test.bus")).unwrap());
        let mut v = VehicleInstance::new(ty, omsi_sim::VehicleHost::new(Default::default()));
        v.set_var("door_0", 0.0);
        v.set_var("PAX_Entry0_Open", 1.0);
        let (e, _) = Humans::doors_open(&v, 1, 0);
        assert_eq!(e, vec![true]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn doors_open_3door_bus_handles_middle_and_rear_exits() {
        let dir = std::env::temp_dir().join(format!("omsi-doors-3door-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("test.bus"),
            "[model]\nmodel.cfg\n[varnamelist]\n1\nvars.txt\n[script]\n1\nmain.osc\n",
        )
        .unwrap();
        std::fs::write(dir.join("model.cfg"), "").unwrap();
        std::fs::write(dir.join("vars.txt"), "door_0\ndoor_1\ndoor_2\ndoor_3\ndoor_4\ndoor_5\n").unwrap();
        std::fs::write(dir.join("main.osc"), "{init}\n{end}\n").unwrap();

        let ty = std::sync::Arc::new(omsi_sim::VehicleType::load(&dir, &dir.join("test.bus")).unwrap());
        let mut v = VehicleInstance::new(ty, omsi_sim::VehicleHost::new(Default::default()));

        // 3-door bus: 6 entries (all 3 doors), 4 exits (middle door leaves 2,3; rear door leaves 4,5)
        let (e, x) = Humans::doors_open(&v, 6, 4);
        assert_eq!(e, vec![false; 6]);
        assert_eq!(x, vec![false; 4]);

        // Middle doors (door_2 and door_3) open
        v.set_var("door_2", 1.0);
        v.set_var("door_3", 1.0);
        let (e, x) = Humans::doors_open(&v, 6, 4);
        assert_eq!(e, vec![false, false, true, true, false, false]);
        assert_eq!(x, vec![true, true, false, false]);

        // Rear doors (door_4 and door_5) open
        v.set_var("door_4", 1.0);
        v.set_var("door_5", 1.0);
        let (e, x) = Humans::doors_open(&v, 6, 4);
        assert_eq!(e, vec![false, false, true, true, true, true]);
        assert_eq!(x, vec![true, true, true, true]);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// #718: a trailer on a lorry's hitch (`[coupling_front_character]` type 0) whose cabin
    /// gives no `[linkToNextVeh]` is not walked into from the bus: it is a section of its
    /// own, boarded, ridden in and left by its own doors. A bus joint (type 1) without the
    /// points is still walked through at the aisle's ends.
    #[test]
    fn a_trailer_nobody_walks_into_is_boarded_by_its_own_doors() {
        let dir = std::env::temp_dir().join(format!("omsi-trailer-cabin-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let write = |name: &str, text: &str| std::fs::write(dir.join(name), text).unwrap();
        // the bus: door at the front right, an aisle, a seat; the trailer: a door, an aisle,
        // a seat and an exit, nothing that names the bus
        write("paths_bus.cfg", "[pathpnt]\n1.1\n4\n0.4\n[pathpnt]\n0\n4\n0.5\n[pathpnt]\n0\n-4\n0.5\n[pathlink]\n0\n1\n[pathlink]\n1\n2\n");
        write("cabin_bus.cfg", "[entry]\n0\n[exit]\n0\n[stamper]\n1\n0.5\n4\n1.5\n[passpos]\n-0.5\n2\n1.0\n0.45\n0\n");
        write("paths_trail.cfg", "[pathpnt]\n1.1\n2\n0.4\n[pathpnt]\n0\n2\n0.5\n[pathpnt]\n0\n-2\n0.5\n[pathlink]\n0\n1\n[pathlink]\n1\n2\n");
        write("cabin_trail.cfg", "[entry]\n0\n[exit]\n0\n[passpos]\n-0.5\n-1\n1.0\n0.45\n0\n");
        // trailers with a way in and none out, and the other way round
        write("cabin_trail_in.cfg", "[entry]\n0\n[passpos]\n-0.5\n-1\n1.0\n0.45\n0\n");
        write("cabin_trail_out.cfg", "[exit]\n0\n[passpos]\n-0.5\n-1\n1.0\n0.45\n0\n");
        // (a bus for its script's variables)
        write("vars.bus", "[model]\nmodel.cfg\n[varnamelist]\n1\nvars.txt\n[script]\n1\nmain.osc\n");
        write("model.cfg", "");
        write("vars.txt", "door_0\n");
        write("main.osc", "{init}\n{end}\n");
        let def = |cabin: &str, paths: &str, coupling: Option<f32>| omsi_vehicle::Vehicle {
            path: dir.join("bus.bus"),
            passenger_cabin: Some(cabin.into()),
            paths: Some(paths.into()),
            bounding_box: Some([2.5, 9.0, 3.0, 0.0, 0.0, 1.5]),
            coupling_front_character: coupling.map(|kind| [30.0, -10.0, 10.0, kind]),
            ..Default::default()
        };
        let bus = def("cabin_bus.cfg", "paths_bus.cfg", None);
        let offset = Vec3::new(0.0, -10.0, 0.0);
        let cabin_of = |cabin: &str, kind: Option<f32>| {
            let trail = def(cabin, "paths_trail.cfg", kind);
            Cabin::load_train(&[(&bus, Vec3::ZERO, f32::INFINITY), (&trail, offset, -5.0)]).expect("cabin")
        };
        let hitched = cabin_of("cabin_trail.cfg", Some(0.0));
        let jointed = cabin_of("cabin_trail.cfg", Some(1.0));
        let plain = cabin_of("cabin_trail.cfg", None);
        let no_exit = cabin_of("cabin_trail_in.cfg", Some(0.0));
        let no_entry = cabin_of("cabin_trail_out.cfg", Some(0.0));
        let jointed_no_exit = cabin_of("cabin_trail_in.cfg", Some(1.0));
        let ty = std::sync::Arc::new(omsi_sim::VehicleType::load(&dir, &dir.join("vars.bus")).unwrap());
        std::fs::remove_dir_all(&dir).ok();
        let v = VehicleInstance::new(ty, omsi_sim::VehicleHost::new(Default::default()));
        assert_eq!((jointed.groups, plain.groups), (1, 1), "a bus joint is walked through");
        assert_eq!(jointed.links.len(), 5);
        assert_eq!(hitched.groups, 2);
        assert_eq!(hitched.links.len(), 4, "nothing joins the bus to its trailer");
        assert_eq!(hitched.parts.len(), 2);
        // its doors and its seat are the trailer's, behind the bus
        assert_eq!((hitched.entries.len(), hitched.exits.len(), hitched.seats.len()), (2, 2, 2));
        assert_eq!((hitched.entries[1].group, hitched.exits[1].group, hitched.seats[1].group), (1, 1, 1));
        assert!((hitched.entries[1].inside - Vec3::new(1.1, -8.0, 0.4)).length() < 1e-4);
        // somebody for the trailer's seat walks to the trailer's door, a rider in it to its exit
        let entries = hitched.in_group(hitched.entry_points(), Some(1));
        assert_eq!(entries, [None, Some(3)]);
        let near_bus_door = Vec3::new(1.6, 4.0, 0.0);
        assert_eq!(hitched.omsi_nearest(near_bus_door, &entries, false, false, None, None), Some(3));
        assert_eq!(hitched.in_group(hitched.exit_points(), hitched.group_at(Some(5))), [None, Some(3)]);
        assert_eq!(hitched.in_group(hitched.exit_points(), Some(0)), [Some(0), None]);
        // with the trailer's doors shut (and no button) they wait at those, not at nothing:
        // the bus's door, the first of the list, is left out of it
        let here = Vec3::new(0.0, -8.0, 0.5);
        let exits = hitched.in_group(hitched.exit_points(), Some(1));
        assert_eq!(hitched.omsi_nearest(here, &exits, false, false, None, Some(&[false, false])), Some(3));
        let flags = hitched.entry_flags();
        for avoid in [false, true] {
            assert_eq!(hitched.omsi_nearest(near_bus_door, &entries, avoid, false, Some(&flags), Some(&[false, false])), Some(3));
        }
        // (one group: the list's first, as Omsi.exe)
        assert_eq!(jointed.omsi_nearest(here, &[None, Some(3)], false, false, None, Some(&[false, false])), None);
        // a trailer of its own nobody can get into, or out of again, has its places off
        assert_eq!(places_off(&v, &hitched), [false, false]);
        assert_eq!(places_off(&v, &no_exit), [false, true], "nobody could get out of it");
        assert_eq!(places_off(&v, &no_entry), [false, true], "nobody could get into it");
        assert_eq!(places_off(&v, &jointed_no_exit), [false, false], "out through the bus");
        // the bus's validator is out of a trailer passenger's reach
        assert_eq!(hitched.nearest_stamper(Vec3::new(1.1, -8.0, 0.4)), None);
        assert_eq!(hitched.nearest_stamper(Vec3::new(1.1, 4.0, 0.4)), Some(0));
        assert_eq!(jointed.nearest_stamper(Vec3::new(1.1, -8.0, 0.4)), Some(0));
    }

    /// #722: a validator by each door - every `[stamper]` is kept, and a passenger stamps at
    /// the one nearest where they came in (Omsi.exe kept the file's last one only).
    #[test]
    fn passengers_stamp_at_the_validator_nearest_their_door() {
        let dir = std::env::temp_dir().join(format!("omsi-stampers-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("test.bus"), "[passengercabin]\ncabin.cfg\n\n[paths]\npaths.cfg\n").unwrap();
        // an aisle from the front door (y 5) to the rear one (y -4)
        let pts: String = [5.0, 4.0, -3.0, -4.0].iter().map(|y| format!("[pathpnt]\n0\n{y}\n0.5\n\n")).collect();
        std::fs::write(dir.join("paths.cfg"), format!("{pts}[pathlink]\n0\n1\n\n[pathlink]\n1\n2\n\n[pathlink]\n2\n3\n")).unwrap();
        std::fs::write(
            dir.join("cabin.cfg"),
            "[entry]\n0\n\n[entry]\n3\n\n[stamper]\n1\n0.5\n4\n1.5\n\n[stamper]\n2\n0.5\n-3\n1.5\n\n[passpos]\n0\n0\n0.5\n0\n0\n",
        )
        .unwrap();
        let def = omsi_vehicle::Vehicle::load(&dir.join("test.bus")).unwrap();
        let cabin = Cabin::load_train(&[(&def, Vec3::ZERO, f32::INFINITY)]).expect("cabin");
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(cabin.stampers.len(), 2);
        assert_eq!(cabin.stampers[1], (Some(2), Vec3::new(0.5, -3.0, 1.5)));
        assert_eq!(cabin.nearest_stamper(Vec3::new(0.0, 5.0, 0.5)), Some(0), "in by the front door");
        assert_eq!(cabin.nearest_stamper(Vec3::new(0.0, -4.0, 0.5)), Some(1), "in by the rear door");
    }

    /// #719: a five-door bus with two paths a door has ten entries. Past Omsi.exe's eight,
    /// an entry whose `PAX_Entry<n>_Open` the script gives is a door of its own, and its
    /// people ask through its `PAX_Entry<n>_Req`; one the script does not know opens with
    /// the eighth, as in Omsi.exe, and asks through it (Omsi.exe drops that request).
    #[test]
    fn doors_past_the_eighth_have_variables_of_their_own_or_go_with_the_eighth() {
        let bus = |name: &str, vars: &str| {
            let dir = std::env::temp_dir().join(format!("omsi-doors-past-eight-{name}-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("test.bus"), "[model]\nmodel.cfg\n[varnamelist]\n1\nvars.txt\n[script]\n1\nmain.osc\n").unwrap();
            std::fs::write(dir.join("model.cfg"), "").unwrap();
            std::fs::write(dir.join("vars.txt"), vars).unwrap();
            std::fs::write(dir.join("main.osc"), "{init}\n{end}\n").unwrap();
            let ty = std::sync::Arc::new(omsi_sim::VehicleType::load(&dir, &dir.join("test.bus")).unwrap());
            std::fs::remove_dir_all(&dir).ok();
            VehicleInstance::new(ty, omsi_sim::VehicleHost::new(Default::default()))
        };
        let mut req = vec![false; 10];
        req[9] = true;

        // the script has the ten doors' variables
        let all: String = (0..10).map(|i| format!("PAX_Entry{i}_Open\n")).collect();
        let mut v = bus("own", &all);
        v.set_var("PAX_Entry9_Open", 1.0);
        let (e, _) = Humans::doors_open(&v, 10, 0);
        assert_eq!(e, [false, false, false, false, false, false, false, false, false, true]);
        Humans::write_door_requests(&mut v, &DoorWants { entry_req: req.clone(), ..Default::default() });
        assert_eq!((v.var("PAX_Entry9_Req"), v.var("PAX_Entry7_Req")), (Some(1.0), Some(0.0)));

        // an older bus knows eight: the ninth and tenth go with the eighth
        let eight: String = (0..8).map(|i| format!("PAX_Entry{i}_Open\n")).collect();
        let mut v = bus("eight", &eight);
        v.set_var("PAX_Entry7_Open", 1.0);
        let (e, _) = Humans::doors_open(&v, 10, 0);
        assert_eq!(e, [false, false, false, false, false, false, false, true, true, true]);
        Humans::write_door_requests(&mut v, &DoorWants { entry_req: req.clone(), ..Default::default() });
        assert_eq!((v.var("PAX_Entry9_Req"), v.var("PAX_Entry7_Req")), (Some(0.0), Some(1.0)));
    }

    /// #721: a `[passpos]` that names a variable of its own is offered to passengers only
    /// while the script holds that variable off 0, and one that names an occupancy variable
    /// has it set while a rider is on it - a tip-up seat follows the person on it, a second
    /// seating layout comes and goes with a setvar.
    #[test]
    fn places_follow_their_own_variables() {
        let dir = std::env::temp_dir().join(format!("omsi-place-vars-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("test.bus"),
            "[model]\nmodel.cfg\n[varnamelist]\n1\nvars.txt\n[script]\n1\nmain.osc\n\n[passengercabin]\ncabin.cfg\n\n[paths]\npaths.cfg\n",
        )
        .unwrap();
        std::fs::write(dir.join("model.cfg"), "").unwrap();
        std::fs::write(dir.join("vars.txt"), "layout_long\nfold_seat_down\n").unwrap();
        std::fs::write(dir.join("main.osc"), "{init}\n{end}\n").unwrap();
        std::fs::write(dir.join("paths.cfg"), "[pathpnt]\n0\n4\n0.5\n\n[pathpnt]\n0\n-3\n0.5\n\n[pathlink]\n0\n1\n").unwrap();
        // a seat of the long layout, a tip-up seat, a plain standing place, and a place whose
        // variable the script does not have
        std::fs::write(
            dir.join("cabin.cfg"),
            "[entry]\n0\n\n[passpos]\n-0.5\n2\n0.9\n0.45\n0\nlayout_long\n\n[passpos]\n0.5\n1\n0.9\n0.45\n0\nlayout_any\nfold_seat_down\n\n\
             [passpos]\n0\n0\n0.5\n0\n0\n\n[passpos]\n0\n-1\n0.5\n0\n0\nno_such_var\n",
        )
        .unwrap();
        let ty = std::sync::Arc::new(omsi_sim::VehicleType::load(&dir, &dir.join("test.bus")).unwrap());
        let def = omsi_vehicle::Vehicle::load(&dir.join("test.bus")).unwrap();
        let cabin = Cabin::load_train(&[(&def, Vec3::ZERO, f32::INFINITY)]).expect("cabin");
        std::fs::remove_dir_all(&dir).ok();
        let mut v = VehicleInstance::new(ty, omsi_sim::VehicleHost::new(Default::default()));
        v.set_var("layout_long", 0.0);
        let off = places_off(&v, &cabin);
        assert_eq!(off, [true, false, false, false], "only the place whose variable is 0 is off");
        let mut h = Humans::new(Path::new("/nonexistent"));
        for _ in 0..40 {
            let k = h.reserve_place(BusId::Player, 4, &off).expect("a free place");
            assert_ne!(k, 0, "nobody takes a place that is switched off");
            h.free_seat(BusId::Player, k);
        }
        assert_eq!(h.reserve_place(BusId::Player, 4, &[true; 4]), None, "all off: nobody gets on");
        v.set_var("layout_long", 1.0);
        assert_eq!(places_off(&v, &cabin), [false; 4]);
        // somebody sits on the tip-up seat (place 1) of this bus, somebody on place 1 of another
        let bn = BusNow {
            id: BusId::Player,
            cabin: Arc::new(cabin),
            pos: DVec3::ZERO,
            rot: Mat4::IDENTITY,
            heading: 0.0,
            speed: 0.0,
            entry_open: vec![false],
            exit_open: Vec::new(),
            walk_open: None,
            interior: 0.0,
            air: CabinAir::default(),
            half: DVec2::new(1.25, 6.0),
            centre: DVec2::ZERO,
            accel: DVec2::ZERO,
            trailers: Vec::new(),
            terminus: None,
            takes: Takes::Terminus,
            next_stop: None,
            places_off: Vec::new(),
        };
        let taken = places_taken(&bn, &[(BusId::Player, 1), (BusId::Ai(2), 2)]);
        assert_eq!(taken, [("fold_seat_down".to_string(), true)]);
        Humans::write_door_requests(&mut v, &DoorWants { places: taken, ..Default::default() });
        assert_eq!(v.var("fold_seat_down"), Some(1.0));
        Humans::write_door_requests(&mut v, &DoorWants { places: places_taken(&bn, &[]), ..Default::default() });
        assert_eq!(v.var("fold_seat_down"), Some(0.0), "up again once they have got up");
    }

    /// #720: `PAX_Entry<n>_Busy` / `PAX_Exit<n>_Busy` tell a door script that somebody stands
    /// in that doorway - on the threshold or in the opening, not in the queue outside a shut
    /// door, the aisle or the deck above - and go with the frame like the requests.
    #[test]
    fn a_doorway_is_busy_while_somebody_stands_in_it() {
        let dir = std::env::temp_dir().join(format!("omsi-doorway-busy-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("test.bus"), "[boundingbox]\n2.5\n12\n3\n0\n0\n1.5\n\n[passengercabin]\ncabin.cfg\n\n[paths]\npaths.cfg\n").unwrap();
        // the front door's threshold on the right (x 1.1), an aisle point beside it, the
        // rear door's threshold at y -3
        std::fs::write(dir.join("paths.cfg"), "[pathpnt]\n1.1\n4\n0.4\n\n[pathpnt]\n0\n4\n0.5\n\n[pathpnt]\n0\n-3\n0.5\n\n[pathpnt]\n1.1\n-3\n0.4\n\n[pathlink]\n0\n1\n\n[pathlink]\n1\n2\n\n[pathlink]\n2\n3\n").unwrap();
        std::fs::write(dir.join("cabin.cfg"), "[entry]\n0\n\n[exit]\n3\n\n[passpos]\n-0.5\n0\n0.9\n0.45\n0\n").unwrap();
        let def = omsi_vehicle::Vehicle::load(&dir.join("test.bus")).unwrap();
        let cabin = Cabin::load_train(&[(&def, Vec3::ZERO, f32::INFINITY)]).expect("cabin");
        let door = &cabin.entries[0];
        assert!(door.in_doorway(Vec3::new(1.1, 4.0, 0.4)), "on the threshold");
        assert!(door.in_doorway(Vec3::new(1.35, 4.1, 0.0)), "in the opening");
        assert!(!door.in_doorway(Vec3::new(1.8, 4.0, 0.0)), "waiting outside the shut door");
        // (the bus is 2.5 m wide: walking past along the line 0.5 m out from its side)
        assert!(!door.in_doorway(Vec3::new(1.25 + 0.5, 4.0, 0.0)), "walking past the door");
        assert!(!door.in_doorway(Vec3::new(1.25 + 0.5, 4.0, 0.4)), "walking past the door");
        assert!(!door.in_doorway(Vec3::new(0.0, 4.0, 0.5)), "in the aisle");
        assert!(!door.in_doorway(Vec3::new(1.1, 4.0, 2.4)), "on the deck above");
        assert!(!door.in_doorway(Vec3::new(1.1, 2.5, 0.4)), "beside the door");
        assert!(cabin.exits[0].in_doorway(Vec3::new(1.2, -3.0, 0.4)));
        assert!(!cabin.exits[0].in_doorway(Vec3::new(1.2, 4.0, 0.4)));
        // a bus standing at (100, 200) heading east: somebody stepping in from the kerb (in
        // the world) and a rider at the rear door (in its frame)
        let rot = Mat4::from_rotation_z(-90f32.to_radians());
        let bn = BusNow {
            id: BusId::Ai(3),
            cabin: Arc::new(cabin),
            pos: DVec3::new(100.0, 200.0, 0.0),
            rot,
            heading: 90.0,
            speed: 0.0,
            entry_open: vec![true],
            exit_open: vec![true],
            walk_open: None,
            interior: 0.0,
            air: CabinAir::default(),
            half: DVec2::new(1.25, 6.0),
            centre: DVec2::ZERO,
            accel: DVec2::ZERO,
            trailers: Vec::new(),
            terminus: None,
            takes: Takes::Terminus,
            next_stop: None,
            places_off: Vec::new(),
        };
        let step_in = bn.world(Vec3::new(1.35, 4.0, 0.0));
        assert_eq!(doorways_taken(&bn, &[(None, step_in)]), (vec![true], vec![false]));
        let rear = DVec3::new(1.1, -3.0, 0.4);
        assert_eq!(doorways_taken(&bn, &[(Some(BusId::Ai(3)), rear)]), (vec![false], vec![true]));
        // a rider of another bus, somebody waiting further out on the pavement: nobody here
        let waiting = bn.world(Vec3::new(2.6, 4.0, 0.0));
        assert_eq!(doorways_taken(&bn, &[(Some(BusId::Player), rear), (None, waiting)]), (vec![false], vec![false]));
        // into the script's variables, a door past the eighth with the eighth's
        std::fs::write(dir.join("vars.bus"), "[model]\nmodel.cfg\n[varnamelist]\n1\nvars.txt\n[script]\n1\nmain.osc\n").unwrap();
        std::fs::write(dir.join("model.cfg"), "").unwrap();
        std::fs::write(dir.join("vars.txt"), "door_0\n").unwrap();
        std::fs::write(dir.join("main.osc"), "{init}\n{end}\n").unwrap();
        let ty = std::sync::Arc::new(omsi_sim::VehicleType::load(&dir, &dir.join("vars.bus")).unwrap());
        std::fs::remove_dir_all(&dir).ok();
        let mut v = VehicleInstance::new(ty, omsi_sim::VehicleHost::new(Default::default()));
        let mut entry_busy = vec![false; 10];
        entry_busy[9] = true;
        Humans::write_door_requests(&mut v, &DoorWants { entry_busy, exit_busy: vec![false, true], ..Default::default() });
        assert_eq!((v.var("PAX_Entry7_Busy"), v.var("PAX_Entry0_Busy")), (Some(1.0), Some(0.0)));
        assert_eq!((v.var("PAX_Exit1_Busy"), v.var("PAX_Exit0_Busy")), (Some(1.0), Some(0.0)));
        // nobody there in the next frame: gone after the scripts' frame, as the requests
        v.update(0.02);
        assert_eq!((v.var("PAX_Entry7_Busy"), v.var("PAX_Exit1_Busy")), (Some(0.0), Some(0.0)));
    }

    // [ROLLBACK seatpick-69]
    #[test]
    fn seat_choice_follows_ride_age_and_room() {
        // a long ride sits more than a short one, the elderly nearly always, a full bus less
        let short = sit_wish(2.0, 40.0, 10, 20, 1.0);
        let long = sit_wish(15.0, 40.0, 10, 20, 1.0);
        assert!(short < 0.35 && long > 0.7, "{short} {long}");
        assert!(sit_wish(2.0, 80.0, 10, 20, 1.0) > 0.85);
        // the last seats go to those who ride far, not to the ones with a short ride
        assert!(sit_wish(2.0, 40.0, 1, 20, 1.0) < 0.1 && sit_wish(15.0, 40.0, 1, 20, 1.0) > 0.5);
        assert!(sit_wish(15.0, 40.0, 1, 20, 1.0) < long);
        // the setting scales the odds either way
        assert!(sit_wish(5.0, 40.0, 10, 20, 2.0) > sit_wish(5.0, 40.0, 10, 20, 1.0));
        assert!(sit_wish(5.0, 40.0, 10, 20, 0.5) < sit_wish(5.0, 40.0, 10, 20, 1.0));
        // weights pick in proportion, the last one at the end of the range
        assert_eq!(pick_weighted(&[3, 7], &[1.0, 3.0], 0.1), 3);
        assert_eq!(pick_weighted(&[3, 7], &[1.0, 3.0], 0.9), 7);
        assert_eq!(pick_weighted(&[3, 7], &[1.0, 3.0], 0.999_999), 7);
    }

    /// [round5 C2] Standing time runs under a crowd's jitter (0.1 m/s), not under a walk; a
    /// person who stood 3 s and shuffles at 0.1 m/s keeps counting.
    #[test]
    fn c2_still_runs_under_the_crowds_jitter() {
        assert_eq!(still_after(0.0, 0.1, 0.5), 0.5);
        assert_eq!(still_after(3.0, 0.14, 0.5), 3.5);
        assert_eq!(still_after(3.0, 0.2, 0.5), 0.0);
        assert_eq!(still_after(3.0, 1.2, 0.5), 0.0);
    }
}

/// [ROLLBACK r4c] The rules of the people at crossings (round 4c, P1..P6).
#[cfg(test)]
mod ped_tests {
    use super::*;
    use std::cell::Cell;

    /// Pedestrian lights the test sets: one light, its state and the seconds it keeps it.
    pub(super) struct FakeLights {
        pub state: Cell<i32>,
        pub left: Cell<f32>,
        /// The pieces of a crossing object's crossing (lit by light (0, 0) unless `unlit`).
        pub pieces: Vec<usize>,
        pub unlit: bool,
        /// The state of light (0, 1), a second group.
        pub second: Cell<i32>,
        /// Seconds until a car light goes green while the pedestrians' light is not.
        pub car_go: Cell<f32>,
    }

    pub(super) const GREEN: i32 = 6;
    pub(super) const RED: i32 = 0;

    impl FakeLights {
        pub fn new(state: i32, left: f32) -> FakeLights {
            FakeLights { state: Cell::new(state), left: Cell::new(left), pieces: Vec::new(), unlit: false, second: Cell::new(state), car_go: Cell::new(20.0) }
        }
    }

    impl PedLights for FakeLights {
        fn light_state(&self, _c: usize, li: usize) -> Option<(i32, f32)> {
            Some((if li == 1 { self.second.get() } else { self.state.get() }, self.left.get()))
        }
        fn light_until_go(&self, _c: usize, _li: usize) -> Option<f32> {
            Some(if omsi_sim::traffic::TrafficLightController::allows_go(self.state.get()) { 0.0 } else { self.car_go.get() })
        }
        fn crossing_light(&self, lane: usize) -> Option<(usize, usize)> {
            (!self.unlit && self.pieces.contains(&lane)).then_some((0, 0))
        }
        fn crossing_piece(&self, lane: usize) -> bool {
            self.pieces.contains(&lane)
        }
    }

    fn lane(points: Vec<DVec3>, kind: LaneKind, width: f32) -> omsi_sim::traffic::Lane {
        omsi_sim::traffic::LaneBuilder::polyline(points, kind, width)
    }

    fn p(x: f64, y: f64) -> DVec3 {
        DVec3::new(x, y, 0.0)
    }

    /// A street of two 3.5 m lanes along x (axes y = -1.75 and +1.75, kerbs at -3.5 and
    /// 3.5) and a crossing over it at x = 0 from the pavement at y = -5.5 to the one at y =
    /// 5.5: a lit 1.5 m stub, a piece over each lane, a lit stub at the far kerb (11 m in
    /// all). Lanes: 0, 1 the street; 2 the near pavement (-20, -5.5) to (0, -5.5); 3 the
    /// stub; 4, 5 the pieces; 6 the far stub; 7 the far pavement (0, 5.5) to (20, 5.5).
    pub(super) fn crossing(lit: bool) -> (Network, PedNet) {
        let mut net = Network::default();
        let street = |y: f64, from: f64, to: f64| (0..=20).map(|k| p(from + (to - from) * k as f64 / 20.0, y)).collect::<Vec<_>>();
        net.lanes.push(lane(street(-1.75, -60.0, 60.0), LaneKind::Street, 3.5));
        net.lanes.push(lane(street(1.75, 60.0, -60.0), LaneKind::Street, 3.5));
        net.lanes.push(lane(vec![p(-20.0, -5.5), p(0.0, -5.5)], LaneKind::Sidewalk, 2.5));
        net.lanes.push(lane(vec![p(0.0, -5.5), p(0.0, -4.0)], LaneKind::Sidewalk, 2.5));
        net.lanes.push(lane(vec![p(0.0, -4.0), p(0.0, 0.0)], LaneKind::Sidewalk, 2.5));
        net.lanes.push(lane(vec![p(0.0, 0.0), p(0.0, 4.0)], LaneKind::Sidewalk, 2.5));
        net.lanes.push(lane(vec![p(0.0, 4.0), p(0.0, 5.5)], LaneKind::Sidewalk, 2.5));
        net.lanes.push(lane(vec![p(0.0, 5.5), p(20.0, 5.5)], LaneKind::Sidewalk, 2.5));
        if lit {
            net.lanes[3].traffic_light = Some((0, 0));
            net.lanes[6].traffic_light = Some((0, 0));
        }
        net.link(1.5);
        let ped = PedNet::build(&net);
        (net, ped)
    }

    /// Across the crossing of `crossing` from 10 m before the near kerb.
    pub(super) fn across() -> PedWalk {
        PedWalk::new(
            vec![
                Leg { lane: 2, a: 10.0, b: 20.0 },
                Leg { lane: 3, a: 0.0, b: 1.5 },
                Leg { lane: 4, a: 0.0, b: 4.0 },
                Leg { lane: 5, a: 0.0, b: 4.0 },
                Leg { lane: 6, a: 0.0, b: 1.5 },
                Leg { lane: 7, a: 0.0, b: 20.0 },
            ],
            false,
            0.0,
        )
    }

    pub(super) fn me(id: u32, pos: DVec2, pace: f64) -> Me {
        Me { id, pos, pace, on_crossing: false, decision: None, why: "", walk_pace: pace }
    }

    /// One frame of the walk as the game moves them (the crowd left out): (velocity).
    pub(super) fn frame(net: &Network, ped: &mut PedNet, lights: Option<&dyn PedLights>, cars: &[SeenCar], t: f64, me: &mut Me, walk: &mut PedWalk, dt: f32) -> DVec2 {
        frame_with(net, ped, lights, cars, &HashMap::new(), &mut HashMap::new(), t, me, walk, dt)
    }

    /// `frame` with the streams on the crossings and the yields to vehicles.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn frame_with(net: &Network, ped: &mut PedNet, lights: Option<&dyn PedLights>, cars: &[SeenCar], streams: &Streams, yields: &mut Yields, t: f64, me: &mut Me, walk: &mut PedWalk, dt: f32) -> DVec2 {
        let mut rng = 0x9E37_79B9_7F4A_7C15u64 ^ me.id as u64;
        let at = Around { net, lights, cars, streams, time: t };
        let w = stroll(Some(ped), &at, me, walk, &mut rng, yields, dt);
        me.pos += w.vel * dt as f64;
        if let Some(leg) = walk.legs.get(walk.leg) {
            walk.s = leg.project(net, me.pos.extend(0.0), walk.s).max(walk.s - 0.3);
        }
        w.vel
    }

    /// The carriageway of `crossing` (inside the kerbs).
    pub(super) fn in_road(q: DVec2) -> bool {
        q.y.abs() < 3.5
    }

    /// Walks the crossing with the light green, turning red once they are `red_at` m across;
    /// returns where they end up after `secs`, the longest they stood on the carriageway and
    /// whether they were `on_crossing` everywhere between the kerbs.
    fn caught_at(red_at: f32, secs: f32) -> (DVec2, f32, bool, Option<&'static str>) {
        caught_at_with(red_at, secs, None)
    }

    /// `caught_at` with the near lane lit by a car light that goes green `car_go` s after the
    /// pedestrians' red (None: no car light programmed, as `crossing` has it).
    fn caught_at_with(red_at: f32, secs: f32, car_go: Option<f32>) -> (DVec2, f32, bool, Option<&'static str>) {
        let (mut net, mut ped) = crossing(true);
        if car_go.is_some() {
            net.lanes[0].traffic_light = Some((0, 5));
        }
        let lights = FakeLights::new(GREEN, 60.0);
        lights.car_go.set(car_go.unwrap_or(20.0));
        let mut walk = across();
        let mut m = me(3, DVec2::new(-10.0, -5.5), 1.2);
        let dt = 0.05;
        let (mut t, mut still, mut longest, mut flagged) = (0.0f32, 0.0f32, 0.0f32, true);
        while t < secs {
            if walk.cross.map(|c| c.walked(&walk) >= red_at && !c.back).unwrap_or(false) {
                lights.state.set(RED);
                lights.left.set(30.0);
            }
            let v = frame(&net, &mut ped, Some(&lights), &[], t as f64, &mut m, &mut walk, dt);
            if in_road(m.pos) {
                still = if v.length() < 0.05 { still + dt } else { 0.0 };
                longest = longest.max(still);
                // (between the kerbs, past the first steps off it)
                if m.pos.y.abs() < 4.9 && !m.on_crossing {
                    flagged = false;
                }
            }
            t += dt;
        }
        (m.pos, longest, flagged, m.decision)
    }

    /// BeijingCity: a crossing object's crossing in 1 m pieces over the street of `crossing`
    /// (y -5.5 to 5.5), most cutting no lane and none lit itself; the walk leaves every other
    /// piece out (both ends of one are within the 1.2 m that make one node). Lane 2 the near
    /// pavement; returns the pieces too.
    pub(super) fn metre_pieces() -> (Network, PedNet, Vec<usize>) {
        let mut net = Network::default();
        let street = |y: f64, from: f64, to: f64| (0..=20).map(|k| p(from + (to - from) * k as f64 / 20.0, y)).collect::<Vec<_>>();
        net.lanes.push(lane(street(-1.75, -60.0, 60.0), LaneKind::Street, 3.5));
        net.lanes.push(lane(street(1.75, 60.0, -60.0), LaneKind::Street, 3.5));
        net.lanes.push(lane(vec![p(-20.0, -5.5), p(0.0, -5.5)], LaneKind::Sidewalk, 2.5));
        let mut pieces = Vec::new();
        for k in 0..11 {
            pieces.push(net.lanes.len());
            net.lanes.push(lane(vec![p(0.0, -5.5 + k as f64), p(0.0, -4.5 + k as f64)], LaneKind::Sidewalk, 2.0));
        }
        net.lanes.push(lane(vec![p(0.0, 5.5), p(20.0, 5.5)], LaneKind::Sidewalk, 2.5));
        net.link(1.5);
        let ped = PedNet::build(&net);
        (net, ped, pieces)
    }

    /// At the unlit kerb of `metre_pieces` with `car`, moving at its speed: the time of the
    /// first "go", and the car's x then.
    fn unlit_with_car(car: SeenCar) -> (Option<f32>, f64, Vec<&'static str>) {
        let (net, mut ped, pieces) = metre_pieces();
        let mut lights = FakeLights::new(GREEN, 60.0);
        lights.pieces = pieces.clone();
        lights.unlit = true;
        let first = Leg { lane: 2, a: 19.5, b: 20.0 };
        let mut walk = PedWalk::new(vec![first], true, 0.0);
        let node = ped.end_node(&net, &first).unwrap();
        let (l0, _) = *ped.out[node].iter().find(|(l, _)| pieces.contains(l)).unwrap();
        walk.legs.push(Leg { lane: l0, a: 0.0, b: 1.0 });
        let mut m = me(5, DVec2::new(-0.5, -5.5), 1.2);
        let mut car = car;
        let (mut went, mut car_x, mut said) = (None, 0.0, Vec::new());
        for k in 0..200 {
            let t = k as f32 * 0.05;
            frame(&net, &mut ped, Some(&lights), &[car], t as f64, &mut m, &mut walk, 0.05);
            if let Some(d) = m.decision {
                if !said.contains(&d) {
                    said.push(d);
                }
                if d == "go" && went.is_none() {
                    went = Some(t);
                    car_x = car.pos.x;
                }
            }
            car.pos += car.vel * 0.05;
            car.body.center = car.pos;
        }
        (went, car_x, said)
    }

    #[test]
    fn l1_a_car_coming_holds_them_at_an_unlit_crossing_of_metre_pieces() {
        let moving = |x: f64| {
            let body = Block { center: DVec2::new(x, -1.75), half: DVec2::new(0.9, 2.25), heading: std::f64::consts::FRAC_PI_2, vel: DVec2::new(12.0, 0.0) };
            SeenCar { id: 3, pos: body.center, vel: body.vel, half_len: 2.25, body, yielding: false, stood: 0.0 }
        };
        // 30 m away at 12 m/s: they wait for it to pass, and go once it has
        let (went, car_x, said) = unlit_with_car(moving(-30.0));
        assert_eq!(said.first(), Some(&"waits for a car to pass"), "{said:?}");
        let went = went.expect("they never went");
        assert!(went > 2.0 && car_x > 2.25, "went at {went:.1} s with the car at x {car_x:.1}");
        // far off (and gone past): across at once
        let (went, _, said) = unlit_with_car(moving(-300.0));
        assert_eq!(said.first(), Some(&"go"), "{said:?}");
        assert!(went.unwrap() < 0.5);
        let (went, _, _) = unlit_with_car(moving(10.0));
        assert!(went.unwrap() < 0.5);
    }

    #[test]
    fn l2_two_light_groups_beyond_an_island_are_two_crossings() {
        // the street of `crossing`, a 6 m island (y 3.5..9.5, its piece 4 m), and a second
        // street (lanes at y 11.75 and 15.75) whose crossing has a light of its own, (0, 1)
        let (mut net, _) = crossing(true);
        net.lanes.truncate(6);
        let street = |y: f64, from: f64, to: f64| (0..=20).map(|k| p(from + (to - from) * k as f64 / 20.0, y)).collect::<Vec<_>>();
        net.lanes.push(lane(vec![p(0.0, 4.0), p(0.0, 8.0)], LaneKind::Sidewalk, 2.5)); // 6 island
        net.lanes.push(lane(vec![p(0.0, 8.0), p(0.0, 9.5)], LaneKind::Sidewalk, 2.5)); // 7 stub
        net.lanes.push(lane(vec![p(0.0, 9.5), p(0.0, 13.75)], LaneKind::Sidewalk, 2.5)); // 8
        net.lanes.push(lane(vec![p(0.0, 13.75), p(0.0, 18.0)], LaneKind::Sidewalk, 2.5)); // 9
        net.lanes.push(lane(vec![p(0.0, 18.0), p(20.0, 18.0)], LaneKind::Sidewalk, 2.5)); // 10
        net.lanes.push(lane(street(11.75, -60.0, 60.0), LaneKind::Street, 3.5));
        net.lanes.push(lane(street(15.75, 60.0, -60.0), LaneKind::Street, 3.5));
        net.lanes[7].traffic_light = Some((0, 1));
        net.link(1.5);
        let mut ped = PedNet::build(&net);
        let lights = FakeLights::new(GREEN, 60.0);
        lights.second.set(RED);
        let legs = vec![
            Leg { lane: 2, a: 10.0, b: 20.0 },
            Leg { lane: 3, a: 0.0, b: 1.5 },
            Leg { lane: 4, a: 0.0, b: 4.0 },
            Leg { lane: 5, a: 0.0, b: 4.0 },
            Leg { lane: 6, a: 0.0, b: 4.0 },
            Leg { lane: 7, a: 0.0, b: 1.5 },
            Leg { lane: 8, a: 0.0, b: 4.25 },
            Leg { lane: 9, a: 0.0, b: 4.25 },
            Leg { lane: 10, a: 0.0, b: 20.0 },
        ];
        let mut walk = PedWalk::new(legs.clone(), false, 0.0);
        let c = crossing_span(&mut ped, &net, Some(&lights), &mut walk, 1).unwrap();
        assert_eq!((c.to, c.light), (3, Some((0, 0))), "the first street only");
        let c = crossing_span(&mut ped, &net, Some(&lights), &mut walk, 4).unwrap();
        assert_eq!((c.from, c.to, c.light), (4, 7, Some((0, 1))), "the island and the second street");
        // walked: across the first on its green, then at the island for the second's red
        let mut walk = PedWalk::new(legs, false, 0.0);
        let mut m = me(2, DVec2::new(-10.0, -5.5), 1.2);
        let mut on_second = false;
        for k in 0..800 {
            frame(&net, &mut ped, Some(&lights), &[], k as f64 * 0.05, &mut m, &mut walk, 0.05);
            on_second |= m.pos.y > 10.0;
        }
        assert!(!on_second, "walked onto the second street on its red");
        assert!(m.pos.y > 3.5 && m.pos.y < 9.5, "waits on the island: {:?} ({:?})", m.pos, m.decision);
        assert_eq!(m.decision, Some("red light"));
        lights.second.set(GREEN);
        for k in 800..1400 {
            frame(&net, &mut ped, Some(&lights), &[], k as f64 * 0.05, &mut m, &mut walk, 0.05);
        }
        assert!(m.pos.y > 17.5, "across on its green: {:?}", m.pos);
    }

    /// The lights as the traffic has them: the crossing pieces' lights from the compiled
    /// junctions (`Junctions::crossing_light`), each light's state set by the test.
    struct JunctionLights<'a> {
        net: &'a Network,
        junctions: omsi_traffic::Junctions,
        states: Vec<Cell<i32>>,
    }

    impl PedLights for JunctionLights<'_> {
        fn light_state(&self, _c: usize, li: usize) -> Option<(i32, f32)> {
            Some((self.states[li].get(), 60.0))
        }
        fn light_until_go(&self, _c: usize, _li: usize) -> Option<f32> {
            Some(20.0)
        }
        fn crossing_light(&self, lane: usize) -> Option<(usize, usize)> {
            self.junctions.crossing_light(self.net, lane)
        }
        fn crossing_piece(&self, lane: usize) -> bool {
            self.junctions.crossing_of(lane).is_some()
        }
    }

    #[test]
    fn r6l1_a_metre_lit_stub_is_walked_and_its_light_holds_them_at_its_kerb() {
        // Taijichang j28 (case09): a kerb corner piece, a 1.005 m stub lit 18.2, the pieces
        // over the road; the chain's other side is lit 18.1 (here: a second street beyond an
        // island, its stub lit (0, 1)). The stub's two ends lay within the 1.2 m of one node,
        // so it was dropped, and the stream walked from the corner piece onto the road on
        // the other side's light.
        let mut net = Network::default();
        let street = |y: f64, from: f64, to: f64| (0..=20).map(|k| p(from + (to - from) * k as f64 / 20.0, y)).collect::<Vec<_>>();
        let object = |mut l: omsi_sim::traffic::Lane, path: u16| {
            l.source = 2;
            l.key = Some(omsi_sim::traffic::LaneKey { tile: (0, 0), id: 9, path });
            l
        };
        net.lanes.push(object(lane(street(-1.75, -15.0, 15.0), LaneKind::Street, 3.5), 0)); // 0
        net.lanes.push(object(lane(street(1.75, 15.0, -15.0), LaneKind::Street, 3.5), 1)); // 1
        net.lanes.push(lane(vec![p(-20.0, -7.005), p(0.0, -7.005)], LaneKind::Sidewalk, 2.5)); // 2 pavement
        let piece = |a: (f64, f64), b: (f64, f64), path: u16| object(lane(vec![p(a.0, a.1), p(b.0, b.1)], LaneKind::Sidewalk, 2.0), path);
        net.lanes.push(piece((0.0, -7.005), (0.0, -5.005), 2)); // 3 corner piece, unlit
        net.lanes.push(piece((0.0, -5.005), (0.0, -4.0), 3)); // 4 the 1.005 m stub, lit (0, 2)
        net.lanes.push(piece((0.0, -4.0), (0.0, 4.0), 4)); // 5 across the street
        net.lanes.push(piece((0.0, 4.0), (0.0, 7.0), 5)); // 6 island
        net.lanes.push(piece((0.0, 7.0), (0.0, 8.0), 6)); // 7 stub lit (0, 1)
        net.lanes.push(piece((0.0, 8.0), (0.0, 16.0), 7)); // 8 across the second street
        net.lanes.push(object(lane(street(10.25, -15.0, 15.0), LaneKind::Street, 3.5), 8)); // 9
        net.lanes.push(object(lane(street(13.75, 15.0, -15.0), LaneKind::Street, 3.5), 9)); // 10
        net.lanes.push(lane(vec![p(0.0, 16.0), p(20.0, 16.0)], LaneKind::Sidewalk, 2.5)); // 11 far pavement
        for l in [3, 4, 5, 6, 7, 8] {
            net.lanes[l].one_way = true;
        }
        net.lanes[4].traffic_light = Some((0, 2));
        net.lanes[7].traffic_light = Some((0, 1));
        net.link(1.5);
        net.compute_conflicts();
        let junctions = omsi_traffic::Junctions::compile(&net);
        assert_eq!(junctions.nodes.len(), 1);
        assert_eq!(junctions.crossing_light(&net, 5), Some((0, 2)), "the piece after the stub has the stub's light");
        assert_eq!(junctions.crossing_light(&net, 8), Some((0, 1)));
        let mut ped = PedNet::build(&net);
        let (a, b) = ped.ends[&4];
        assert_ne!(a, b, "the 1.005 m lit stub keeps its two ends");
        assert_eq!(ped.ends[&3].1, a, "the corner piece ends where the stub starts");
        assert_eq!(ped.ends[&5].0, b, "the piece across starts where the stub ends");
        assert!(ped.out[a].iter().any(|&(l, fwd)| l == 4 && fwd));
        assert!(!ped.out[a].iter().any(|&(l, _)| l == 5), "no way past the stub onto the road");
        let lights = JunctionLights { net: &net, junctions, states: (0..3).map(|_| Cell::new(GREEN)).collect() };
        lights.states[2].set(RED);
        let legs = vec![
            Leg { lane: 2, a: 10.0, b: 20.0 },
            Leg { lane: 3, a: 0.0, b: 2.0 },
            Leg { lane: 4, a: 0.0, b: 1.005 },
            Leg { lane: 5, a: 0.0, b: 8.0 },
            Leg { lane: 6, a: 0.0, b: 3.0 },
            Leg { lane: 7, a: 0.0, b: 1.0 },
            Leg { lane: 8, a: 0.0, b: 8.0 },
            Leg { lane: 11, a: 0.0, b: 20.0 },
        ];
        // the corner piece is a crossing of its own (unlit, it cuts nothing); the stub begins
        // the crossing of the first street under its light
        let mut walk = PedWalk::new(legs.clone(), false, 0.0);
        let c = crossing_span(&mut ped, &net, Some(&lights), &mut walk, 1).unwrap();
        assert_eq!((c.from, c.to, c.light), (1, 1, None));
        let c = crossing_span(&mut ped, &net, Some(&lights), &mut walk, 2).unwrap();
        assert_eq!((c.from, c.to, c.light), (2, 4, Some((0, 2))), "to the island");
        // walked: they wait at the stub's kerb on (0, 2)'s red with (0, 1) green
        let mut walk = PedWalk::new(legs, false, 0.0);
        let mut m = me(5, DVec2::new(-10.0, -7.005), 1.2);
        let mut on_road = false;
        for k in 0..800 {
            frame(&net, &mut ped, Some(&lights), &[], k as f64 * 0.05, &mut m, &mut walk, 0.05);
            on_road |= m.pos.y > -3.5;
        }
        assert!(!on_road, "walked onto the street on its red: {:?}", m.pos);
        assert!(m.pos.y > -7.5 && m.pos.y < -3.5, "waits at the kerb: {:?} ({:?})", m.pos, m.decision);
        assert_eq!(m.decision, Some("red light"));
        lights.states[2].set(GREEN);
        for k in 800..1600 {
            frame(&net, &mut ped, Some(&lights), &[], k as f64 * 0.05, &mut m, &mut walk, 0.05);
        }
        assert!(m.pos.y > 15.5, "across on the greens: {:?}", m.pos);
    }

    #[test]
    fn l3_put_down_on_the_kerb_piece_they_still_wait_off_the_carriageway() {
        // the kerb piece of `p3_a_kerb_inside...` (to y = -3.2, in the lane's footprint) with
        // only its last 0.3 m walked; and a stub wholly in the lane beside a pavement outside
        let mut net = Network::default();
        let street = |y: f64, from: f64, to: f64| (0..=20).map(|k| p(from + (to - from) * k as f64 / 20.0, y)).collect::<Vec<_>>();
        net.lanes.push(lane(street(-1.75, -60.0, 60.0), LaneKind::Street, 3.5));
        net.lanes.push(lane(street(1.75, 60.0, -60.0), LaneKind::Street, 3.5));
        net.lanes.push(lane(vec![p(0.0, -12.0), p(0.0, -3.2)], LaneKind::Sidewalk, 2.5)); // 2
        net.lanes.push(lane(vec![p(0.0, -3.2), p(0.0, 3.2)], LaneKind::Sidewalk, 2.5)); // 3
        net.lanes.push(lane(vec![p(20.0, -3.3), p(25.0, -3.3)], LaneKind::Sidewalk, 2.5)); // 4: in the lane
        net.lanes.push(lane(vec![p(25.0, -3.3), p(25.0, 3.3)], LaneKind::Sidewalk, 2.5)); // 5
        net.lanes.push(lane(vec![p(15.0, -7.0), p(35.0, -7.0)], LaneKind::Sidewalk, 2.5)); // 6: outside
        net.lanes[3].traffic_light = Some((0, 0));
        net.lanes[5].traffic_light = Some((0, 0));
        net.link(1.5);
        let mut ped = PedNet::build(&net);
        let lights = FakeLights::new(RED, 30.0);
        for id in 0..6u32 {
            let mut walk = PedWalk::new(vec![Leg { lane: 2, a: 8.5, b: 8.8 }, Leg { lane: 3, a: 0.0, b: 6.4 }], false, 0.0);
            let mut m = me(id, DVec2::new(0.0, -3.5), 1.2);
            for k in 0..300 {
                frame(&net, &mut ped, Some(&lights), &[], k as f64 * 0.05, &mut m, &mut walk, 0.05);
            }
            assert_eq!(m.decision, Some("red light"));
            assert!(!on_carriageway(&net, m.pos.extend(0.0)), "#{id} at {:?}", m.pos);
            assert!(m.pos.y > -5.5, "#{id} at {:?}", m.pos);
            let mut walk = PedWalk::new(vec![Leg { lane: 4, a: 4.7, b: 5.0 }, Leg { lane: 5, a: 0.0, b: 6.6 }], false, 0.0);
            let mut m = me(id, DVec2::new(24.7, -3.3), 1.2);
            for k in 0..300 {
                frame(&net, &mut ped, Some(&lights), &[], k as f64 * 0.05, &mut m, &mut walk, 0.05);
            }
            assert!(!on_carriageway(&net, m.pos.extend(0.0)), "#{id} at {:?}", m.pos);
            assert!((m.pos - DVec2::new(25.0, -3.3)).length() < 4.0, "#{id} near the kerb: {:?}", m.pos);
        }
    }

    #[test]
    fn r5p1_nobody_is_put_down_on_a_crossing_piece_or_a_lit_one() {
        // the street of `crossing`; pavements, a 5 m kerb piece inside a lane's footprint
        // that cuts no lane, an 8 m unlit piece across both lanes, a 6 m lit piece that
        // cuts nothing, and a crossing object's 1 m pieces
        let (mut net, _) = crossing(true);
        net.lanes.push(lane(vec![p(20.0, -3.3), p(25.0, -3.3)], LaneKind::Sidewalk, 2.5)); // 8: kerb piece in the lane
        net.lanes.push(lane(vec![p(30.0, -4.0), p(30.0, 4.0)], LaneKind::Sidewalk, 2.0)); // 9: unlit, cuts both lanes
        net.lanes.push(lane(vec![p(40.0, -12.0), p(40.0, -6.0)], LaneKind::Sidewalk, 2.5)); // 10: lit, cuts nothing
        net.lanes[10].traffic_light = Some((0, 2));
        let mut pieces = Vec::new();
        for k in 0..11 {
            pieces.push(net.lanes.len());
            net.lanes.push(lane(vec![p(50.0, -5.5 + k as f64), p(50.0, -4.5 + k as f64)], LaneKind::Sidewalk, 2.0));
        }
        net.lanes.push(lane(vec![p(20.0, -12.0), p(60.0, -12.0)], LaneKind::Sidewalk, 2.5)); // a boulevard pavement
        let boulevard = net.lanes.len() - 1;
        net.link(1.5);
        let mut ped = PedNet::build(&net);
        let mut lights = FakeLights::new(GREEN, 60.0);
        lights.pieces = pieces.clone();
        let ok = |ped: &mut PedNet, l: usize| stroll_candidate(ped, &net, Some(&lights), l);
        // long pavements and junction-object kerb pieces that cut no lane: candidates
        for l in [2usize, 7, 8, boulevard] {
            assert!(ok(&mut ped, l), "lane {l} should take strollers");
        }
        // lit stubs, pieces over the lanes, an unlit piece across the street, a lit piece
        // that cuts nothing, a crossing object's pieces: never
        for l in [3usize, 4, 5, 6, 9, 10] {
            assert!(!ok(&mut ped, l), "lane {l} is a crossing piece");
        }
        for l in pieces {
            assert!(!ok(&mut ped, l), "lane {l} is a crossing object's piece");
        }
        // without the traffic's crossing table the lanes' own cuts and lights still tell
        assert!(!stroll_candidate(&mut ped, &net, None, 9) && !stroll_candidate(&mut ped, &net, None, 10));
        assert!(stroll_candidate(&mut ped, &net, None, 8));
    }

    #[test]
    fn p1_a_crossing_of_one_metre_pieces_is_one_crossing() {
        // BeijingCity: the crossing object's crossing in 1 m pieces, most cutting no lane and
        // none lit itself (the light is the crossing's); the walk leaves every other piece
        // out (both ends of one are within the 1.2 m that make one node)
        let mut net = Network::default();
        let street = |y: f64, from: f64, to: f64| (0..=20).map(|k| p(from + (to - from) * k as f64 / 20.0, y)).collect::<Vec<_>>();
        net.lanes.push(lane(street(-1.75, -60.0, 60.0), LaneKind::Street, 3.5));
        net.lanes.push(lane(street(1.75, 60.0, -60.0), LaneKind::Street, 3.5));
        net.lanes.push(lane(vec![p(-20.0, -5.5), p(0.0, -5.5)], LaneKind::Sidewalk, 2.5));
        let mut pieces = Vec::new();
        for k in 0..11 {
            pieces.push(net.lanes.len());
            net.lanes.push(lane(vec![p(0.0, -5.5 + k as f64), p(0.0, -4.5 + k as f64)], LaneKind::Sidewalk, 2.0));
        }
        net.lanes.push(lane(vec![p(0.0, 5.5), p(20.0, 5.5)], LaneKind::Sidewalk, 2.5));
        net.link(1.5);
        let mut ped = PedNet::build(&net);
        let mut lights = FakeLights::new(GREEN, 60.0);
        lights.pieces = pieces.clone();
        let first = Leg { lane: 2, a: 10.0, b: 20.0 };
        let node = ped.end_node(&net, &first).unwrap();
        let (l0, fwd) = *ped.out[node].iter().find(|(l, _)| pieces.contains(l)).unwrap();
        assert!(fwd);
        let mut walk = PedWalk::new(vec![first, Leg { lane: l0, a: 0.0, b: 1.0 }], true, 0.0);
        let c = crossing_span(&mut ped, &net, Some(&lights), &mut walk, 1).unwrap();
        let last = walk.legs[c.to];
        assert!(last.at(&net, last.len()).0.y > 4.4, "to the far kerb: {:?}", walk.legs);
        assert!(c.span > 8.5 && c.span < 10.5, "{}", c.span);
        assert_eq!(c.light, Some((0, 0)));
        // and walked: on_crossing everywhere between the kerbs, the red at the kerb only
        let mut m = me(9, DVec2::new(-10.0, -5.5), 1.2);
        let mut walk = PedWalk::new(vec![first], true, 0.0);
        walk.legs.push(Leg { lane: l0, a: 0.0, b: 1.0 });
        let (mut flagged, mut stood, mut longest) = (true, 0.0f32, 0.0f32);
        for k in 0..600 {
            let v = frame(&net, &mut ped, Some(&lights), &[], k as f64 * 0.05, &mut m, &mut walk, 0.05);
            if walk.cross.map(|c| c.walked(&walk) > c.total * 0.6).unwrap_or(false) {
                lights.state.set(RED);
            }
            if in_road(m.pos) {
                if !m.on_crossing {
                    flagged = false;
                }
                stood = if v.length() < 0.05 { stood + 0.05 } else { 0.0 };
                longest = longest.max(stood);
            }
        }
        assert!(flagged);
        assert!(longest < 0.2, "{longest}");
        assert!(m.pos.y > 5.0, "{:?}", m.pos);
    }

    #[test]
    fn p1_caught_past_the_middle_goes_on_to_the_far_kerb() {
        let (end, stood, flagged, _) = caught_at(6.0, 40.0);
        assert!(end.y > 5.3, "on the far pavement: {end:?}");
        assert!(stood < 0.2, "stood {stood:.2} s on the carriageway");
        assert!(flagged, "on_crossing on every piece between the kerbs");
    }

    #[test]
    fn p1_caught_before_the_middle_goes_back_to_the_near_kerb() {
        let (end, stood, flagged, decision) = caught_at(3.0, 40.0);
        assert!(end.y < -5.0, "back on the near side: {end:?}");
        assert!(!in_road(end));
        assert!(stood < 0.2, "stood {stood:.2} s on the carriageway");
        assert!(flagged);
        // and waits there for the green
        assert_eq!(decision, Some("red light"));
    }

    #[test]
    fn r5p2_put_down_mid_piece_goes_back_to_the_kerb_not_the_drop_point() {
        // somebody dropped 1.5 m into the piece over the near lane (y -2.5, in lane 0's
        // footprint) with the light red: at the piece's end the change sends them back, and
        // "back" is the kerb at y -4, where they wait with "red light" off the carriageway -
        // not the drop point, where they looped "go" / "back to the kerb" (case04/09/12)
        let (net, mut ped) = crossing(true);
        let lights = FakeLights::new(RED, 60.0);
        let mut walk = PedWalk::new(vec![Leg { lane: 4, a: 1.5, b: 4.0 }], true, 0.0);
        let mut m = me(3, DVec2::new(0.0, -2.5), 1.2);
        let (mut still, mut longest, mut turned, mut decisions) = (0.0f32, 0.0f32, false, Vec::new());
        for k in 0..600 {
            let v = frame(&net, &mut ped, Some(&lights), &[], k as f64 * 0.05, &mut m, &mut walk, 0.05);
            turned |= m.decision == Some(TURNS_BACK);
            if let Some(d) = m.decision {
                if decisions.last() != Some(&d) {
                    decisions.push(d);
                }
            }
            if turned && in_road(m.pos) {
                still = if v.length() < 0.05 { still + 0.05 } else { 0.0 };
                longest = longest.max(still);
            }
        }
        assert!(turned, "{decisions:?}");
        assert!(longest < 0.2, "stood {longest:.2} s in the road after turning back ({decisions:?})");
        assert!(!on_carriageway(&net, m.pos.extend(0.0)) && m.pos.y < -3.8, "{:?} {decisions:?}", m.pos);
        assert_eq!(m.decision, Some("red light"), "{decisions:?}");
        // and no looping: back once, then the wait
        assert_eq!(decisions.iter().filter(|d| **d == TURNS_BACK).count(), 1, "{decisions:?}");
    }

    /// A two-stage crossing under ONE light group: the street of `crossing`, a median piece
    /// `m` long (y 3.5 .. 3.5 + m, in no lane), a second street of two lanes beyond it, lit
    /// stubs (0, 0) at both kerbs. Lanes: 0, 1 and 2, 3 the streets; 4 the near pavement; 5
    /// stub; 6, 7 pieces; 8 the median; 9, 10 pieces; 11 stub; 12 the far pavement.
    fn two_stage(m: f64) -> (Network, PedNet, Vec<Leg>) {
        let mut net = Network::default();
        let street = |y: f64, from: f64, to: f64| (0..=20).map(|k| p(from + (to - from) * k as f64 / 20.0, y)).collect::<Vec<_>>();
        let far = 3.5 + m;
        net.lanes.push(lane(street(-1.75, -60.0, 60.0), LaneKind::Street, 3.5));
        net.lanes.push(lane(street(1.75, 60.0, -60.0), LaneKind::Street, 3.5));
        net.lanes.push(lane(street(far + 1.75, -60.0, 60.0), LaneKind::Street, 3.5));
        net.lanes.push(lane(street(far + 5.25, 60.0, -60.0), LaneKind::Street, 3.5));
        net.lanes.push(lane(vec![p(-20.0, -5.5), p(0.0, -5.5)], LaneKind::Sidewalk, 2.5));
        net.lanes.push(lane(vec![p(0.0, -5.5), p(0.0, -4.0)], LaneKind::Sidewalk, 2.5));
        net.lanes.push(lane(vec![p(0.0, -4.0), p(0.0, 0.0)], LaneKind::Sidewalk, 2.5));
        net.lanes.push(lane(vec![p(0.0, 0.0), p(0.0, 3.5)], LaneKind::Sidewalk, 2.5));
        net.lanes.push(lane(vec![p(0.0, 3.5), p(0.0, far)], LaneKind::Sidewalk, 2.5));
        net.lanes.push(lane(vec![p(0.0, far), p(0.0, far + 3.5)], LaneKind::Sidewalk, 2.5));
        net.lanes.push(lane(vec![p(0.0, far + 3.5), p(0.0, far + 7.0)], LaneKind::Sidewalk, 2.5));
        net.lanes.push(lane(vec![p(0.0, far + 7.0), p(0.0, far + 8.5)], LaneKind::Sidewalk, 2.5));
        net.lanes.push(lane(vec![p(0.0, far + 8.5), p(20.0, far + 8.5)], LaneKind::Sidewalk, 2.5));
        net.lanes[5].traffic_light = Some((0, 0));
        net.lanes[11].traffic_light = Some((0, 0));
        net.link(1.5);
        let ped = PedNet::build(&net);
        let legs = vec![
            Leg { lane: 4, a: 10.0, b: 20.0 },
            Leg { lane: 5, a: 0.0, b: 1.5 },
            Leg { lane: 6, a: 0.0, b: 4.0 },
            Leg { lane: 7, a: 0.0, b: 3.5 },
            Leg { lane: 8, a: 0.0, b: m as f32 },
            Leg { lane: 9, a: 0.0, b: 3.5 },
            Leg { lane: 10, a: 0.0, b: 3.5 },
            Leg { lane: 11, a: 0.0, b: 1.5 },
            Leg { lane: 12, a: 0.0, b: 20.0 },
        ];
        (net, ped, legs)
    }

    /// Walks `two_stage(m)` on green, red once `red_at` m are walked: where they end up after
    /// `secs`, their decision, the longest they stood on a carriageway.
    fn caught_two_stage(m: f64, red_at: f32, secs: f32) -> (DVec2, Option<&'static str>, f32) {
        let (net, mut ped, legs) = two_stage(m);
        let lights = FakeLights::new(GREEN, 60.0);
        let mut walk = PedWalk::new(legs, false, 0.0);
        let mut me_ = me(3, DVec2::new(-10.0, -5.5), 1.2);
        let (mut t, mut still, mut longest) = (0.0f32, 0.0f32, 0.0f32);
        while t < secs {
            if walk.cross.map(|c| c.walked(&walk) >= red_at && !c.back).unwrap_or(false) {
                lights.state.set(RED);
                lights.left.set(30.0);
            }
            let v = frame(&net, &mut ped, Some(&lights), &[], t as f64, &mut me_, &mut walk, 0.05);
            if on_carriageway(&net, me_.pos.extend(0.0)) {
                still = if v.length() < 0.05 { still + 0.05 } else { 0.0 };
                longest = longest.max(still);
            }
            t += 0.05;
        }
        (me_.pos, me_.decision, longest)
    }

    #[test]
    fn r5p2_caught_near_a_median_clears_to_it() {
        let (net, _, legs) = two_stage(3.0);
        assert!(refuge(&net, &legs[4]), "a 3 m median in no lane is a refuge");
        assert!(!refuge(&net, &legs[3]) && !refuge(&net, &legs[5]), "the pieces over the lanes are not");
        let (_, _, legs1) = two_stage(1.0);
        assert!(!refuge(&net, &legs1[4]), "a 1 m painted island is not");
        let pieces = |m: f32| vec![(1.5, false), (4.0, false), (3.5, false), (m, m >= 2.0), (3.5, false), (3.5, false), (1.5, false)];
        // 8 m of 20.5 walked, the median 1 m on: on to it (before: back 8 m to the near kerb)
        assert_eq!(change_turns_back(8.0, 0.0, &pieces(3.0), None), Clear::On(3));
        // 13 m walked, the median 1 m behind: back to its far end
        assert_eq!(change_turns_back(13.0, 0.0, &pieces(3.0), None), Clear::Back(4));
        // on the median itself: stay on it (to its end)
        assert_eq!(change_turns_back(10.0, 0.0, &pieces(3.0), None), Clear::On(3));
        // 2 m walked: the near kerb; 19 m: the far kerb; no refuge with a 1 m island
        assert_eq!(change_turns_back(2.0, 0.0, &pieces(3.0), None), Clear::Back(0));
        assert_eq!(change_turns_back(19.0, 0.0, &pieces(3.0), None), Clear::On(6));
        assert_eq!(change_turns_back(8.0, 0.0, &pieces(1.0), None), Clear::Back(0));
        // put down 6 m into a 12 m piece and caught 0.3 m on: 6.3 of 12 from the kerb, on
        assert_eq!(change_turns_back(0.3, 6.0, &[(6.0, false)], None), Clear::On(0));
        // walked: caught 8 m across, they wait on the median with "red light", in no lane
        let (end, decision, stood) = caught_two_stage(3.0, 8.0, 40.0);
        assert!(end.y > 3.8 && end.y < 6.2, "on the median: {end:?}");
        assert_eq!(decision, Some("red light"));
        assert!(stood < 0.2, "stood {stood:.2} s in a lane");
        // caught 13 m across: back to the median
        let (end, decision, stood) = caught_two_stage(3.0, 13.0, 40.0);
        assert!(end.y > 3.8 && end.y < 6.2, "back on the median: {end:?}");
        assert_eq!(decision, Some("red light"));
        assert!(stood < 0.2, "stood {stood:.2} s in a lane");
        // a 1 m island: back to the near kerb as before
        let (end, _, stood) = caught_two_stage(1.0, 8.0, 40.0);
        assert!(end.y < -3.8, "back on the near pavement: {end:?}");
        assert!(stood < 0.2, "stood {stood:.2} s in a lane");
        // and from the median, the second street on its green
        let (net, mut ped, legs) = two_stage(3.0);
        let lights = FakeLights::new(GREEN, 60.0);
        let mut walk = PedWalk::new(legs, false, 0.0);
        let mut m = me(3, DVec2::new(-10.0, -5.5), 1.2);
        for k in 0..800 {
            if walk.cross.map(|c| c.walked(&walk) >= 8.0 && !c.back).unwrap_or(false) && k < 400 {
                lights.state.set(RED);
            }
            if k == 400 {
                lights.state.set(GREEN);
            }
            frame(&net, &mut ped, Some(&lights), &[], k as f64 * 0.05, &mut m, &mut walk, 0.05);
        }
        assert!(m.pos.y > 14.5, "across the second street on its green: {:?} {:?}", m.pos, m.decision);
    }

    #[test]
    fn r5p3_only_who_cannot_clear_in_the_clearance_turns_back() {
        // caught 3 m into the 11 m crossing: 8 m at the hurried pace (1.56-2.16 m/s for a walker) take 3.7-5.1 s
        // (me(3) walks 1.2 m/s). With the cars' green 12 s off, on across; 5 s off, back;
        // with no car light programmed (`caught_at`), back before the middle as before
        let (end, stood, flagged, decision) = caught_at_with(3.0, 40.0, Some(12.0));
        assert!(end.y > 5.3, "on across within the clearance: {end:?} {decision:?}");
        assert!(stood < 0.2 && flagged);
        let (end, stood, _, decision) = caught_at_with(3.0, 40.0, Some(5.0));
        assert!(end.y < -5.0, "back, the clearance too short: {end:?} {decision:?}");
        assert!(stood < 0.2);
        assert_eq!(decision, Some("red light"));
        let (end, ..) = caught_at_with(3.0, 40.0, None);
        assert!(end.y < -5.0, "back, no clearance programmed: {end:?}");
        // the rule itself: 4 s needed, 7 s given: on; 5.9 s given (margin 2): back
        let pieces = [(1.5, false), (4.0, false), (4.0, false), (1.5, false)];
        assert_eq!(change_turns_back(3.0, 0.0, &pieces, Some((7.0, 4.0))), Clear::On(3));
        assert_eq!(change_turns_back(3.0, 0.0, &pieces, Some((5.9, 4.0))), Clear::Back(0));
        assert_eq!(change_turns_back(3.0, 0.0, &pieces, Some((0.0, 4.0))), Clear::Back(0), "the cars have green already");
        assert_eq!(change_turns_back(3.0, 0.0, &pieces, None), Clear::Back(0));
        assert_eq!(change_turns_back(6.0, 0.0, &pieces, None), Clear::On(3));
    }

    #[test]
    fn p1_the_light_counts_at_the_kerb_only() {
        // red at the kerb: they wait on the pavement; on the pieces past the stub the light
        // is not asked again (it is green throughout here, so nothing changes on the way)
        let (net, mut ped) = crossing(true);
        let lights = FakeLights::new(RED, 30.0);
        let mut walk = across();
        let mut m = me(4, DVec2::new(-10.0, -5.5), 1.2);
        for k in 0..400 {
            frame(&net, &mut ped, Some(&lights), &[], k as f64 * 0.05, &mut m, &mut walk, 0.05);
        }
        assert!(!in_road(m.pos) && m.pos.y < -5.0, "{:?}", m.pos);
        assert_eq!(m.decision, Some("red light"));
        assert_eq!(walk.leg, 0);
        // the crossing is all four pieces, 11 m, with the light of its stubs
        let c = crossing_span(&mut ped, &net, None, &mut walk, 1).unwrap();
        assert_eq!((c.from, c.to), (1, 4));
        assert!((c.total - 11.0).abs() < 1e-3);
        assert_eq!(c.light, Some((0, 0)));
        // a stroll is planned straight across
        let mut roam = PedWalk::new(vec![Leg { lane: 2, a: 10.0, b: 20.0 }, Leg { lane: 3, a: 0.0, b: 1.5 }], true, 0.0);
        let c = crossing_span(&mut ped, &net, None, &mut roam, 1).unwrap();
        assert_eq!(roam.legs.iter().map(|l| l.lane).collect::<Vec<_>>(), [2, 3, 4, 5, 6]);
        assert_eq!(c.to, 4);
    }

    #[test]
    fn p6_everybody_walks_at_their_own_pace() {
        // [ROLLBACK r5-P4] (the uniform 1.0..1.4 and the fixed 1.4 x hurry became a log-normal
        // pace and a hurry of the person's own: the bounds and the middle are checked, not
        // the old numbers)
        let paces: Vec<f64> = (1..2001u32).map(pace_of).collect();
        assert!(paces.iter().all(|p| (PED_PACE_MIN..=PED_PACE_MAX).contains(p)));
        let (lo, hi) = paces.iter().fold((9.0f64, 0.0f64), |(a, b), p| (a.min(*p), b.max(*p)));
        assert!(lo < 0.9 && hi > 1.75, "{lo} {hi}");
        let mut sorted = paces.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let median = sorted[sorted.len() / 2];
        assert!((median - PED_PACE_MEDIAN).abs() < 0.03, "median {median}");
        // two thirds within one sigma (15 %) of the median
        let within = paces.iter().filter(|p| (p.ln() - PED_PACE_MEDIAN.ln()).abs() < PED_PACE_SIGMA).count() as f64 / paces.len() as f64;
        assert!(within > 0.62 && within < 0.74, "{within}");
        // the same person always the same
        assert_eq!(pace_of(77), pace_of(77));
        assert_eq!(hurried(77, pace_of(77)), hurried(77, pace_of(77)));
        // hurrying: 1.3..1.8 times one's own, at most 2.5; joggers 2.5..3.0, about 5 %; and
        // nobody below 0.8 or above 3.0 ever
        let mut joggers = 0;
        for id in 1..2001u32 {
            let (p, h) = (pace_of(id), hurried(id, pace_of(id)));
            assert!((PED_PACE_MIN..=PED_JOG_MAX).contains(&h), "#{id} hurries at {h}");
            if jogger(id) {
                joggers += 1;
                assert!((PED_JOG_MIN..=PED_JOG_MAX).contains(&h), "#{id} jogs at {h}");
            } else {
                assert!(h >= p * PED_HURRY_MIN - 1e-9 && h <= (p * PED_HURRY_MAX_FACTOR).min(PED_HURRY_MAX) + 1e-9, "#{id} {p} -> {h}");
            }
        }
        assert!((60..=140).contains(&joggers), "{joggers} joggers of 2000");
    }

    #[test]
    fn p6_caught_by_the_change_or_yielded_to_they_hurry() {
        let (net, mut ped) = crossing(true);
        let lights = FakeLights::new(GREEN, 60.0);
        let mut walk = across();
        let mut m = me(3, DVec2::new(-10.0, -5.5), 1.2);
        let (mut before, mut after, mut off) = (Vec::new(), Vec::new(), Vec::new());
        for k in 0..600 {
            if walk.cross.map(|c| c.walked(&walk) >= 6.0).unwrap_or(false) {
                lights.state.set(RED);
            }
            frame(&net, &mut ped, Some(&lights), &[], k as f64 * 0.05, &mut m, &mut walk, 0.05);
            let caught = lights.state.get() == RED;
            match (m.on_crossing, caught) {
                (true, false) => before.push(m.walk_pace),
                (true, true) => after.push(m.walk_pace),
                (false, true) => off.push(m.walk_pace),
                _ => {}
            }
        }
        assert!(!before.is_empty() && before.iter().all(|p| (*p - 1.2).abs() < 1e-9));
        assert!(!after.is_empty() && after.iter().all(|p| (*p - hurried(3, 1.2)).abs() < 1e-9), "{after:?}");
        // off the carriageway their own pace again
        assert!(off.iter().all(|p| (*p - 1.2).abs() < 1e-9));
        // a car standing for them close by: faster across too, on green
        let (net, mut ped) = crossing(true);
        let lights = FakeLights::new(GREEN, 60.0);
        let mut walk = across();
        let mut m = me(3, DVec2::new(-10.0, -5.5), 1.2);
        let mut car = car_before(3.0);
        car.body.center.x = -4.0;
        let mut hurried_on = 0;
        for k in 0..400 {
            frame(&net, &mut ped, Some(&lights), &[car], k as f64 * 0.05, &mut m, &mut walk, 0.05);
            if m.on_crossing && (m.walk_pace - hurried(3, 1.2)).abs() < 1e-9 {
                hurried_on += 1;
            }
        }
        assert!(hurried_on > 20, "{hurried_on}");
    }

    #[test]
    fn p6_nobody_stands_frozen_with_go() {
        assert!(replan_frozen(Some("go"), 2.1, 1.2));
        assert!(!replan_frozen(Some("go"), 1.5, 1.2));
        // waiting by decision, or not wanting to walk, is not frozen
        assert!(!replan_frozen(Some("red light"), 30.0, 1.2));
        assert!(!replan_frozen(Some("go"), 30.0, 0.0));
        // just off the kerb with a vehicle in the way: back to the kerb, never left on the
        // carriageway
        let (net, mut ped) = crossing(false);
        let mut walk = across();
        let mut m = me(3, DVec2::new(-10.0, -5.5), 1.2);
        for k in 0..400 {
            frame(&net, &mut ped, None, &[], k as f64 * 0.05, &mut m, &mut walk, 0.05);
            if walk.leg == 2 {
                break;
            }
        }
        assert!(walk.cross.map(|c| c.walked(&walk) < 1.6).unwrap_or(false));
        walk.turn_back(&net);
        for k in 0..200 {
            frame(&net, &mut ped, None, &[], k as f64 * 0.05, &mut m, &mut walk, 0.05);
            if walk.cross.is_none() {
                break;
            }
        }
        assert!(!in_road(m.pos) && m.pos.y < -5.0, "{:?}", m.pos);
    }

    /// At the unlit kerb of `crossing` with `car` standing: the first decision.
    fn kerb_with(car: SeenCar) -> Option<&'static str> {
        let (net, mut ped) = crossing(false);
        let mut walk = across();
        walk.legs[0] = Leg { lane: 2, a: 19.5, b: 20.0 };
        let mut m = me(1, DVec2::new(-0.5, -5.5), 1.2);
        let mut first = None;
        for k in 0..20 {
            frame(&net, &mut ped, None, &[car], k as f64 * 0.05, &mut m, &mut walk, 0.05);
            first = first.or(m.decision);
        }
        first
    }

    /// At the lit kerb of `crossing` on green with `car` standing: the first decision.
    fn lit_kerb_with(car: SeenCar) -> Option<&'static str> {
        let (net, mut ped) = crossing(true);
        let lights = FakeLights::new(GREEN, 60.0);
        let mut walk = across();
        walk.legs[0] = Leg { lane: 2, a: 19.5, b: 20.0 };
        let mut m = me(1, DVec2::new(-0.5, -5.5), 1.2);
        let mut first = None;
        for k in 0..20 {
            frame(&net, &mut ped, Some(&lights), &[car], k as f64 * 0.05, &mut m, &mut walk, 0.05);
            first = first.or(m.decision);
        }
        first
    }

    #[test]
    fn r5p6_nobody_starts_onto_an_occupied_crossing() {
        // a car standing 5 s with its box over the near lane's cut (0, -1.75): refused at a
        // lit kerb on green (where no vehicle was looked at before) and at an unlit one
        let mut on = car_before(5.0);
        on.yielding = false;
        on.body.center = DVec2::new(0.0, -1.75);
        on.pos = on.body.center;
        assert_eq!(lit_kerb_with(on), Some(STANDS_ON));
        assert_eq!(kerb_with(on), Some(STANDS_ON));
        // over the far lane's cut as well
        let mut far = on;
        far.body.center = DVec2::new(0.0, 1.75);
        far.pos = far.body.center;
        assert_eq!(lit_kerb_with(far), Some(STANDS_ON));
        // standing there 25 s (abandoned): go, round it
        let mut old = on;
        old.stood = 25.0;
        assert_eq!(lit_kerb_with(old), Some("go"));
        // one stopping for people (pedyield-1): go
        let mut yielding = on;
        yielding.yielding = true;
        assert_eq!(lit_kerb_with(yielding), Some("go"));
        // its front 0.95 m short of the cut (p5): go
        let mut short = on;
        short.body.center = DVec2::new(-3.2, -1.75);
        short.pos = short.body.center;
        assert_eq!(lit_kerb_with(short), Some("go"));
        assert_eq!(kerb_with(short), Some("go"));
    }

    #[test]
    fn r5p6b_the_kerb_rule_counts_zebra_holds_creepers_and_near_cuts_only() {
        // a car holding for the zebra is one stopping for people
        assert!(yields_to_people(omsi_traffic::Why::People) && yields_to_people(omsi_traffic::Why::Zebra));
        assert!(!yields_to_people(omsi_traffic::Why::Lead) && !yields_to_people(omsi_traffic::Why::Light));
        // a queue car creeping onto the near cut at 0.7 m/s: refused, lit and unlit
        let mut creep = car_before(5.0);
        creep.yielding = false;
        creep.body.center = DVec2::new(0.0, -1.75);
        creep.pos = creep.body.center;
        creep.vel = DVec2::new(0.7, 0.0);
        creep.body.vel = creep.vel;
        assert_eq!(lit_kerb_with(creep), Some(STANDS_ON));
        assert_eq!(kerb_with(creep), Some(STANDS_ON));
        // one that stood 25 s on the cut and now creeps on at 0.6 m/s: the abandoned-car
        // exemption is for a car that stands, so still refused; standing 25 s: go (round it)
        let mut rolls = creep;
        rolls.stood = 25.0;
        rolls.vel = DVec2::new(0.6, 0.0);
        rolls.body.vel = rolls.vel;
        assert_eq!(lit_kerb_with(rolls), Some(STANDS_ON));
        let mut stands = rolls;
        stands.vel = DVec2::ZERO;
        stands.body.vel = DVec2::ZERO;
        assert_eq!(lit_kerb_with(stands), Some("go"));
        // a car on a cut 13.75 m down the way (the second street of `two_stage`): go
        let (net, mut ped, legs) = two_stage(3.0);
        let lights = FakeLights::new(GREEN, 60.0);
        let mut far = car_before(5.0);
        far.yielding = false;
        far.body.center = DVec2::new(0.0, 3.5 + 3.0 + 1.75);
        far.pos = far.body.center;
        let mut walk = PedWalk::new(legs, false, 0.0);
        walk.legs[0] = Leg { lane: 4, a: 19.5, b: 20.0 };
        let mut m = me(1, DVec2::new(-0.5, -5.5), 1.2);
        let mut first = None;
        for k in 0..20 {
            frame(&net, &mut ped, Some(&lights), &[far], k as f64 * 0.05, &mut m, &mut walk, 0.05);
            first = first.or(m.decision);
        }
        assert_eq!(first, Some("go"), "a car on a far cut does not hold the kerb");
        // the same car on the near street's far lane (3.5 m in): refused
        far.body.center = DVec2::new(0.0, 1.75);
        far.pos = far.body.center;
        let (net, mut ped, legs) = two_stage(3.0);
        let mut walk = PedWalk::new(legs, false, 0.0);
        walk.legs[0] = Leg { lane: 4, a: 19.5, b: 20.0 };
        let mut m = me(1, DVec2::new(-0.5, -5.5), 1.2);
        let mut first = None;
        for k in 0..20 {
            frame(&net, &mut ped, Some(&lights), &[far], k as f64 * 0.05, &mut m, &mut walk, 0.05);
            first = first.or(m.decision);
        }
        assert_eq!(first, Some(STANDS_ON));
    }

    #[test]
    fn r5p6b_the_gap_median_follows_the_strollers_wanted() {
        // 4852 m of pavement for 20000 wanted (pedestrians=19000): the floor; 700 m for 50:
        // the 6 m ceiling; 2000 m for 500: 2 m
        assert_eq!(gap_median(4852.0, 20000), PED_GAP_MIN);
        assert_eq!(gap_median(700.0, 50), PED_GAP_MEDIAN);
        assert!((gap_median(2000.0, 500) - 2.0).abs() < 1e-9);
        assert_eq!(gap_median(100.0, 0), PED_GAP_MEDIAN);
    }

    #[test]
    fn r5p6b_the_nearest_pavement_is_never_a_crossing_piece() {
        // off a bus 0.3 m from the piece over the near lane of `crossing`, 1 m from the lit
        // stub and 2.5 m from the pavement: the pavement
        let (net, ped) = crossing(true);
        let (lane, _, _) = ped.nearest(&net, DVec3::new(0.3, -3.0, 0.0), 12.0).unwrap();
        assert_eq!(lane, 2);
        // in the middle of the road: a pavement still, never a piece
        let (lane, _, _) = ped.nearest(&net, DVec3::new(0.2, 0.3, 0.0), 12.0).unwrap();
        assert!(lane == 2 || lane == 7, "{lane}");
        // an unlit crossing's piece over the lane is no candidate either (its unlit kerb
        // stub, which cuts nothing, is a pavement)
        let (net, ped) = crossing(false);
        let (lane, _, _) = ped.nearest(&net, DVec3::new(0.3, -3.0, 0.0), 12.0).unwrap();
        assert!(lane == 2 || lane == 3, "{lane}");
    }

    #[test]
    fn r5p6_groups_and_gaps_follow_their_laws() {
        let zs: Vec<f64> = (1..4001u64).map(|k| hash_normal(k, 7)).collect();
        let mean = zs.iter().sum::<f64>() / zs.len() as f64;
        let sd = (zs.iter().map(|z| (z - mean) * (z - mean)).sum::<f64>() / zs.len() as f64).sqrt();
        assert!(mean.abs() < 0.06 && (sd - 1.0).abs() < 0.06, "normal: {mean} {sd}");
        let sizes: Vec<usize> = zs.iter().map(|z| group_size(*z)).collect();
        assert!(sizes.iter().all(|s| (1..=PED_GROUP_MAX).contains(s)));
        let mean = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;
        assert!(mean > 1.25 && mean < 1.6, "group mean {mean}");
        // round(exp(0.8 z)) is 1 for z < 0.507: 69 %
        let ones = sizes.iter().filter(|s| **s == 1).count() as f64 / sizes.len() as f64;
        assert!(ones > 0.62 && ones < 0.76, "singles {ones}");
        assert!(sizes.iter().any(|s| *s >= 4) && sizes.iter().filter(|s| **s == 6).count() < 100);
        let mut gaps: Vec<f64> = zs.iter().map(|z| lognormal(PED_GAP_MEDIAN, PED_GAP_SIGMA, *z)).collect();
        gaps.sort_by(|a, b| a.total_cmp(b));
        let median = gaps[gaps.len() / 2];
        assert!((median - 6.0).abs() < 0.5, "gap median {median}");
        assert!(gaps[0] < 1.0 && gaps[gaps.len() - 1] > 40.0, "{} {}", gaps[0], gaps[gaps.len() - 1]);
        let mut offs: Vec<f64> = zs.iter().map(|z| lognormal(PED_GROUP_GAP_MEDIAN, PED_GROUP_GAP_SIGMA, *z)).collect();
        offs.sort_by(|a, b| a.total_cmp(b));
        assert!((offs[offs.len() / 2] - 1.0).abs() < 0.1);
    }

    #[test]
    fn p5_only_a_box_over_the_way_holds_at_the_kerb() {
        // a car standing in lane 0 with its front 0.95 m short of the crossing (its centre
        // 3.2 m from where the crossing cuts the lane: within half its length and 1.5 m,
        // which held them before): beside the way, no hold
        let mut car = car_before(1.0);
        car.yielding = false;
        car.body.center = DVec2::new(-3.2, -1.75);
        car.pos = car.body.center;
        assert_eq!(kerb_with(car), Some("go"));
        // the player's bus standing with its nose 1 m short of the way off the kerb: no hold
        let bus = Block { center: DVec2::new(-7.25, -4.6), half: DVec2::new(1.25, 6.0), heading: std::f64::consts::FRAC_PI_2, vel: DVec2::ZERO };
        let player = SeenCar { id: PLAYER_CAR, pos: bus.center, vel: DVec2::ZERO, half_len: 6.0, body: bus, yielding: false, stood: 30.0 };
        assert_eq!(kerb_with(player), Some("go"));
        // with its nose over the way it holds them while it has stood under PED_OCCUPIED_CAP
        // (the r5-P6 rule); one standing longer is stepped round at once (P5's own 5 s hold
        // went, review r7)
        let mut over = player;
        over.body.center.x = -5.0;
        over.stood = PED_OCCUPIED_CAP - 10.0;
        assert_eq!(kerb_with(over), Some("a vehicle stands on the crossing"));
        over.stood = PED_OCCUPIED_CAP + 10.0;
        assert_eq!(kerb_with(over), Some("go"));
    }

    /// A walker going north along a pavement (axis x = 0, 2.1 m wide) past a bus standing
    /// half on it (its box x -2.15..0.35, y -6..6): the game's steps round a standing box in
    /// small (the route along its side, the turns capped), the crowd keeping them out of it.
    /// Returns where they end up and how often they turned round.
    fn past_the_bus(corridor_half: f64) -> (DVec2, usize, bool) {
        let bus = Block { center: DVec2::new(-0.9, 0.0), half: DVec2::new(1.25, 6.0), heading: 0.0, vel: DVec2::ZERO };
        let mut w = Walker::new(DVec2::new(0.0, -9.0), BODY_OUTSIDE, 0);
        w.solid = true;
        let params = CrowdParams::default();
        let dt = 0.05;
        let (mut around, mut want_dir, mut turned, mut turns, mut routed) = (None::<([DVec2; 2], usize)>, DVec2::ZERO, PED_TURN_EVERY, 0usize, false);
        for _ in 0..800 {
            let here = w.pos;
            let corridor = Some((DVec2::new(0.0, here.y - 2.0), DVec2::new(0.0, here.y + 2.5), corridor_half));
            let mut want = (DVec2::new(0.0, here.y + 1.3) - here).normalize_or_zero() * 1.2;
            if let Some((wp, k)) = around {
                let k = if (wp[k] - here).length() < 0.45 { k + 1 } else { k };
                around = (k < 2).then_some((wp, k));
                if let Some((wp, k)) = around {
                    want = (wp[k] - here).normalize_or_zero() * 1.2;
                }
            } else {
                let dir = want.normalize_or_zero();
                let ahead = here + dir * 0.9;
                if bus.near(ahead, BODY_OUTSIDE + 0.15) {
                    match along_box(&bus, here, dir, corridor) {
                        Along::Route(wp) => {
                            routed = true;
                            around = Some((wp, 0));
                            want = (wp[0] - here).normalize_or_zero() * 1.2;
                        }
                        _ => {
                            if let Some(wp) = around_box(&bus, here, dir) {
                                around = Some((wp, 0));
                                want = (wp[0] - here).normalize_or_zero() * 1.2;
                            }
                        }
                    }
                }
            }
            turned += dt as f32;
            if want.length() > 0.2 {
                let d = want.normalize();
                if want_dir.dot(d) < -0.2 {
                    if !may_turn(turned) {
                        want = DVec2::ZERO;
                    } else {
                        turned = 0.0;
                        turns += 1;
                    }
                }
                if want.length() > 0.2 {
                    want_dir = d;
                }
            }
            w.want = want;
            w.corridor = None;
            crowd::step(std::slice::from_mut(&mut w), &[bus], &params, dt);
        }
        (w.pos, turns, routed)
    }

    #[test]
    fn p5_a_bus_along_the_pavement_is_walked_past_on_its_free_side() {
        let (end, turns, routed) = past_the_bus(1.05);
        assert!(routed, "{end:?} {turns}");
        assert!(end.y > 7.0, "past the bus: {end:?}");
        assert_eq!(turns, 0, "no pacing in front of its nose");
        // the route keeps to the pavement: beside the box by half a metre
        let bus = Block { center: DVec2::new(-0.9, 0.0), half: DVec2::new(1.25, 6.0), heading: 0.0, vel: DVec2::ZERO };
        let Along::Route(wp) = along_box(&bus, DVec2::new(0.0, -7.0), DVec2::Y, Some((DVec2::new(0.0, -9.0), DVec2::new(0.0, -4.5), 1.05))) else { panic!() };
        assert!((wp[0].x - 0.85).abs() < 1e-9 && (wp[1].x - 0.85).abs() < 1e-9, "{wp:?}");
        assert!(wp[1].y > 6.0);
        // no room beside it within the path: not round its nose either
        assert_eq!(along_box(&bus, DVec2::new(0.0, -7.0), DVec2::Y, Some((DVec2::new(0.0, -9.0), DVec2::new(0.0, -4.5), 0.4))), Along::NoRoom);
        // a box across the way is gone round its end as before
        let across = Block { heading: std::f64::consts::FRAC_PI_2, ..bus };
        assert_eq!(along_box(&across, DVec2::new(0.0, -9.0), DVec2::Y, None), Along::Across);
    }

    #[test]
    fn p5_turning_round_is_capped_beside_a_standing_vehicle() {
        assert!(!may_turn(1.0));
        assert!(!may_turn(4.9));
        assert!(may_turn(5.0));
        // a pavement too narrow to pass: the walker may turn once in five seconds at most,
        // round the nose as before (which paced 10-14 times in 24 s)
        let (_, turns, _) = past_the_bus(0.4);
        assert!(turns <= 40 / 5 + 1, "{turns} turns in 40 s");
    }

    /// A car standing in lane 0 with its front 2.75 m before the crossing of `crossing`.
    pub(super) fn car_before(stood: f32) -> SeenCar {
        let body = Block { center: DVec2::new(-5.0, -1.75), half: DVec2::new(0.9, 2.25), heading: std::f64::consts::FRAC_PI_2, vel: DVec2::ZERO };
        SeenCar { id: 7, pos: body.center, vel: DVec2::ZERO, half_len: 2.25, body, yielding: true, stood }
    }

    /// Somebody arriving at the kerb with another person just ahead of them on the crossing
    /// (`stream`) and the car of `car_before` standing there `stood` s: their decisions over
    /// `secs` (time, decision), where they end up and the yields.
    fn arrive_at(lit: bool, stood: f32, stream: bool, secs: f32) -> (Vec<(f32, Option<&'static str>)>, Me, Yields) {
        arrive_with(lit, stood, if stream { 1 } else { 0 }, secs)
    }

    /// `arrive_at` with `stream_n` people on the crossing: one just ahead at the kerb, the
    /// rest spread over the far lane's piece (not in front of the car).
    fn arrive_with(lit: bool, stood: f32, stream_n: usize, secs: f32) -> (Vec<(f32, Option<&'static str>)>, Me, Yields) {
        let (net, mut ped) = crossing(lit);
        let lights = FakeLights::new(GREEN, 60.0);
        let mut walk = across();
        walk.legs[0] = Leg { lane: 2, a: 19.5, b: 20.0 };
        let mut m = me(1, DVec2::new(-0.5, -5.5), 1.2);
        let mut streams: Streams = HashMap::new();
        if stream_n > 0 {
            let mut v = vec![(50, DVec2::new(0.0, -4.5), DVec2::new(0.0, 1.2))];
            for k in 1..stream_n {
                v.push((50 + k as u32, DVec2::new(0.0, 0.2 + 3.6 * k as f64 / stream_n as f64), DVec2::new(0.0, 1.2)));
            }
            streams.insert(3, v);
        }
        let mut yields = HashMap::new();
        let mut car = car_before(stood);
        let mut out = Vec::new();
        let mut t = 0.0f32;
        while t < secs {
            frame_with(&net, &mut ped, lit.then_some(&lights as &dyn PedLights), &[car], &streams, &mut yields, t as f64, &mut m, &mut walk, 0.05);
            out.push((t, m.decision));
            car.stood += 0.05;
            t += 0.05;
        }
        (out, m, yields)
    }

    #[test]
    fn p4_after_eight_seconds_a_stream_lets_the_standing_vehicle_go() {
        let (seen, m, yields) = arrive_at(false, 10.0, true, 20.0);
        let at = |t: f32| seen.iter().find(|(s, _)| *s >= t).unwrap().1;
        assert_eq!(at(1.0), Some(YIELDS));
        assert_eq!(at(5.0), Some(YIELDS));
        // for six seconds at most, then across
        assert_eq!(at(6.5), Some("go"));
        assert!(m.pos.y > 4.0, "{:?}", m.pos);
        assert!(yields.contains_key(&7));
        // waiting, they were off the crossing and outside the car's strip (its half width and
        // the cars' 0.3 m): the car's people check lets it go
        let (seen, m, _) = arrive_at(false, 10.0, true, 3.0);
        assert_eq!(seen.last().unwrap().1, Some(YIELDS));
        assert!(!m.on_crossing);
        assert!((m.pos.y + 1.75).abs() > 0.9 + 0.3, "{:?}", m.pos);
    }

    #[test]
    fn r5p5_the_stood_clock_survives_a_creep_but_not_a_roll_through() {
        let o = DVec2::ZERO;
        // standing 8 s, then a creep of 0.7 m/s for a second (0.7 m), standing again: the
        // clock runs on
        let mut c = (0.0f32, o);
        for _ in 0..160 {
            c = stood_clock(Some(c), o, 0.0, 0.05);
        }
        assert!((c.0 - 8.0).abs() < 1e-3);
        let mut pos = o;
        for _ in 0..20 {
            pos += DVec2::new(0.7 * 0.05, 0.0);
            c = stood_clock(Some(c), pos, 0.7, 0.05);
        }
        assert!(c.0 > 8.9, "the creep reset it: {}", c.0);
        c = stood_clock(Some(c), pos, 0.0, 0.05);
        assert!(c.0 > 8.9 && (c.1 - pos).length() < 1e-9, "stands at the new place: {c:?}");
        // then 1.2 m/s for 2.5 s (3 m): reset
        for _ in 0..50 {
            pos += DVec2::new(1.2 * 0.05, 0.0);
            c = stood_clock(Some(c), pos, 1.2, 0.05);
        }
        assert_eq!(c.0, 0.0);
        // a car rolling through at 1.0-1.5 m/s never earns PED_YIELD_AFTER
        for speed in [1.0f64, 1.2, 1.5] {
            let mut c = (0.0f32, o);
            let mut pos = o;
            let mut most = 0.0f32;
            for _ in 0..400 {
                pos += DVec2::new(speed * 0.05, 0.0);
                c = stood_clock(Some(c), pos, speed, 0.05);
                most = most.max(c.0);
            }
            assert!(most < 2.5, "{speed} m/s: {most} s");
        }
        // a speed over PED_CREEP_SPEED resets at once
        let c = stood_clock(Some((8.0, o)), o, 1.6, 0.05);
        assert_eq!(c.0, 0.0);
        // from nothing: standing starts the clock, moving does not
        assert!(stood_clock(None, o, 0.1, 0.05).0 > 0.0);
        assert_eq!(stood_clock(None, o, 0.7, 0.05).0, 0.0);
    }

    #[test]
    fn r5p5_a_dense_stream_yields_until_the_cap() {
        // twelve on the 11 m crossing: the arrivals let the car go for 2 x 6 s, renewed under
        // the 18 s cap, then across; one person ahead (the thin stream): 6 s as before
        let (seen, m, _) = arrive_with(false, 10.0, 12, 25.0);
        let at = |t: f32| seen.iter().find(|(s, _)| *s >= t).unwrap().1;
        assert_eq!(at(1.0), Some(YIELDS));
        assert_eq!(at(8.0), Some(YIELDS));
        assert_eq!(at(11.5), Some(YIELDS));
        assert_eq!(at(14.0), Some(YIELDS));
        assert_eq!(at(18.5), Some("go"));
        assert!(m.pos.y > 2.0, "{:?}", m.pos);
        let (seen, ..) = arrive_with(false, 10.0, 1, 10.0);
        let at = |t: f32| seen.iter().find(|(s, _)| *s >= t).unwrap().1;
        assert_eq!(at(5.0), Some(YIELDS));
        assert_eq!(at(6.5), Some("go"));
    }

    #[test]
    fn l4_the_yield_renews_while_the_vehicle_keeps_standing() {
        // a car standing on, a stream on: those arriving let it go for 6 s, cross themselves
        // for as long, then let it go again (once, it stood for good after the first 6 s)
        let (net, mut ped) = crossing(false);
        let mut streams: Streams = HashMap::new();
        streams.insert(3, vec![(50, DVec2::new(0.0, -4.5), DVec2::new(0.0, 1.2))]);
        let mut yields = HashMap::new();
        let decide = |t: f64, yields: &mut Yields, ped: &mut PedNet| {
            let mut walk = across();
            walk.legs[0] = Leg { lane: 2, a: 19.5, b: 20.0 };
            let mut m = me(1, DVec2::new(-0.5, -5.5), 1.2);
            for _ in 0..6 {
                frame_with(&net, ped, None, &[car_before(10.0 + t as f32)], &streams, yields, t, &mut m, &mut walk, 0.05);
            }
            m.decision
        };
        assert_eq!(decide(0.0, &mut yields, &mut ped), Some(YIELDS));
        assert_eq!(decide(5.0, &mut yields, &mut ped), Some(YIELDS));
        assert_eq!(decide(8.0, &mut yields, &mut ped), Some("go"));
        assert_eq!(decide(11.5, &mut yields, &mut ped), Some("go"));
        assert_eq!(decide(12.5, &mut yields, &mut ped), Some(YIELDS));
        assert_eq!(decide(17.0, &mut yields, &mut ped), Some(YIELDS));
        // while somebody of the stream is still on the crossing in front of the car, its six
        // seconds wait for them to be off it
        let mut yields = HashMap::new();
        let mut streams: Streams = HashMap::new();
        streams.insert(3, vec![(50, DVec2::new(0.0, -2.5), DVec2::new(0.0, 1.2)), (51, DVec2::new(0.0, -4.5), DVec2::new(0.0, 1.2))]);
        let decide = |t: f64, yields: &mut Yields, ped: &mut PedNet, streams: &Streams| {
            let mut walk = across();
            walk.legs[0] = Leg { lane: 2, a: 19.5, b: 20.0 };
            let mut m = me(1, DVec2::new(-0.5, -5.5), 1.2);
            for _ in 0..6 {
                frame_with(&net, ped, None, &[car_before(10.0 + t as f32)], streams, yields, t, &mut m, &mut walk, 0.05);
            }
            m.decision
        };
        assert_eq!(decide(0.0, &mut yields, &mut ped, &streams), Some(YIELDS));
        assert_eq!(decide(5.0, &mut yields, &mut ped, &streams), Some(YIELDS));
        assert_eq!(decide(9.0, &mut yields, &mut ped, &streams), Some(YIELDS), "still somebody in front of it");
        streams.insert(3, vec![(50, DVec2::new(0.0, 3.0), DVec2::new(0.0, 1.2))]);
        assert_eq!(decide(13.0, &mut yields, &mut ped, &streams), Some(YIELDS), "six seconds once they are off");
        assert_eq!(decide(15.5, &mut yields, &mut ped, &streams), Some("go"));
    }

    #[test]
    fn l4_somebody_lingering_at_the_cut_holds_arrivals_18_s_at_most() {
        // a car standing on, the stream on, and one person standing in front of it at the cut
        // for 60 s: the arrivals let the car go for 18 s at most in all
        let (net, mut ped) = crossing(false);
        let mut streams: Streams = HashMap::new();
        streams.insert(3, vec![(50, DVec2::new(0.0, -1.75), DVec2::ZERO), (51, DVec2::new(0.0, -4.5), DVec2::new(0.0, 1.2))]);
        let mut yields: Yields = HashMap::new();
        let (mut held, mut last_held) = (0.0f64, 0.0f64);
        for k in 0..60 {
            let t = k as f64;
            let mut walk = across();
            walk.legs[0] = Leg { lane: 2, a: 19.5, b: 20.0 };
            let mut m = me(1, DVec2::new(-0.5, -5.5), 1.2);
            for _ in 0..6 {
                frame_with(&net, &mut ped, None, &[car_before(10.0 + t as f32)], &streams, &mut yields, t, &mut m, &mut walk, 0.05);
            }
            if m.decision == Some(YIELDS) {
                held += 1.0;
                last_held = t;
            }
        }
        assert!(held >= 6.0 && held <= 18.0, "held {held} s");
        assert!(last_held < 18.0, "still held at {last_held} s");
    }

    #[test]
    fn p4_no_yield_too_soon_without_a_stream_or_at_a_light() {
        // the car has stood 5 s only
        let (seen, ..) = arrive_at(false, 5.0, true, 1.0);
        assert!(seen.iter().all(|(_, d)| *d != Some(YIELDS)));
        // nobody ahead of them on the crossing: no stream, no yield (the car gets its gap)
        let (seen, ..) = arrive_at(false, 10.0, false, 1.0);
        assert!(seen.iter().all(|(_, d)| *d != Some(YIELDS)));
        // a lit crossing on green: across
        let (seen, m, _) = arrive_at(true, 10.0, true, 15.0);
        assert!(seen.iter().all(|(_, d)| *d != Some(YIELDS)));
        assert!(m.pos.y > 4.0, "{:?}", m.pos);
    }

    #[test]
    fn p3_they_wait_off_the_carriageway() {
        // the kerb of `crossing` lies outside the lanes: the spot is the kerb's, a step back
        let (net, mut ped) = crossing(true);
        let lights = FakeLights::new(RED, 30.0);
        for id in 0..15u32 {
            let mut walk = across();
            let mut m = me(id, DVec2::new(-10.0, -5.5), 1.2);
            for k in 0..400 {
                frame(&net, &mut ped, Some(&lights), &[], k as f64 * 0.05, &mut m, &mut walk, 0.05);
            }
            assert!(!on_carriageway(&net, m.pos.extend(0.0)), "#{id} at {:?}", m.pos);
            assert!(m.pos.y < -5.0 && m.pos.y > -7.0, "#{id} at {:?}", m.pos);
        }
        assert!(on_carriageway(&net, DVec3::new(0.0, -3.6, 0.0)), "within the lane and 0.3 m");
        assert!(!on_carriageway(&net, DVec3::new(0.0, -3.9, 0.0)));
    }

    #[test]
    fn r5p7_the_kerb_spot_keeps_out_of_a_junction_piece_beside_the_way() {
        // the street of `p3_they_wait_off...` with its kerb outside the lanes (y -4), and a
        // 1 m junction piece (counted 2.7 m wide here: its footprint reaches 1.65 m) running
        // beside the pavement 1.6 m to the right of it: every spot on that side and the
        // pavement's own line lie in its footprint, the far side does not. Before, the way
        // walked was backed along for 8 m to its end for the spot; now a step to the other
        // side does
        let mut net = Network::default();
        let street = |y: f64, from: f64, to: f64| (0..=20).map(|k| p(from + (to - from) * k as f64 / 20.0, y)).collect::<Vec<_>>();
        net.lanes.push(lane(street(-1.75, -60.0, 60.0), LaneKind::Street, 3.5));
        net.lanes.push(lane(street(1.75, 60.0, -60.0), LaneKind::Street, 3.5));
        net.lanes.push(lane(vec![p(0.0, -20.0), p(0.0, -4.0)], LaneKind::Sidewalk, 2.5)); // 2
        net.lanes.push(lane(vec![p(0.0, -4.0), p(0.0, 4.0)], LaneKind::Sidewalk, 2.5)); // 3
        net.lanes.push(lane(vec![p(1.6, -4.2), p(1.6, -12.0)], LaneKind::Street, 2.7)); // 4: the junction piece
        net.lanes[3].traffic_light = Some((0, 0));
        net.link(1.5);
        let mut ped = PedNet::build(&net);
        let lights = FakeLights::new(RED, 30.0);
        assert!(on_carriageway(&net, DVec3::new(0.0, -5.0, 0.0)) && !on_carriageway(&net, DVec3::new(-0.9, -5.0, 0.0)));
        for id in 0..10u32 {
            let mut walk = PedWalk::new(vec![Leg { lane: 2, a: 4.0, b: 16.0 }, Leg { lane: 3, a: 0.0, b: 8.0 }], false, 0.0);
            let mut m = me(id, DVec2::new(0.0, -16.0), 1.2);
            for k in 0..400 {
                frame(&net, &mut ped, Some(&lights), &[], k as f64 * 0.05, &mut m, &mut walk, 0.05);
            }
            assert_eq!(m.decision, Some("red light"));
            assert!(!on_carriageway(&net, m.pos.extend(0.0)), "#{id} at {:?}", m.pos);
            assert!(m.pos.y > -6.0 && m.pos.y < -3.9, "#{id} near the kerb, not backed away: {:?}", m.pos);
            assert!(m.pos.x < -0.3, "#{id} on the pavement's side: {:?}", m.pos);
        }
    }

    #[test]
    fn p3_a_kerb_inside_the_lane_is_backed_away_from() {
        // a pavement piece that runs on into the lane's footprint (to y = -3.2, the lane's
        // edge at -3.5, its footprint to -3.8) and the crossing from there
        let mut net = Network::default();
        let street = |y: f64, from: f64, to: f64| (0..=20).map(|k| p(from + (to - from) * k as f64 / 20.0, y)).collect::<Vec<_>>();
        net.lanes.push(lane(street(-1.75, -60.0, 60.0), LaneKind::Street, 3.5));
        net.lanes.push(lane(street(1.75, 60.0, -60.0), LaneKind::Street, 3.5));
        net.lanes.push(lane(vec![p(0.0, -12.0), p(0.0, -3.2)], LaneKind::Sidewalk, 2.5));
        net.lanes.push(lane(vec![p(0.0, -3.2), p(0.0, 3.2)], LaneKind::Sidewalk, 2.5));
        net.lanes[3].traffic_light = Some((0, 0));
        net.link(1.5);
        let mut ped = PedNet::build(&net);
        let lights = FakeLights::new(RED, 30.0);
        for id in 0..6u32 {
            let mut walk = PedWalk::new(vec![Leg { lane: 2, a: 0.0, b: 8.8 }, Leg { lane: 3, a: 0.0, b: 6.4 }], false, 0.0);
            let mut m = me(id, DVec2::new(0.0, -12.0), 1.2);
            for k in 0..400 {
                frame(&net, &mut ped, Some(&lights), &[], k as f64 * 0.05, &mut m, &mut walk, 0.05);
            }
            assert_eq!(m.decision, Some("red light"));
            assert!(!on_carriageway(&net, m.pos.extend(0.0)), "#{id} at {:?}", m.pos);
            // (no further back than needed)
            assert!(m.pos.y > -5.0, "#{id} at {:?}", m.pos);
        }
        // the old spot, at the kerb a step back, was in the lane
        let spot = kerb_spot(Some(&ped), &net, &[Leg { lane: 2, a: 0.0, b: 8.8 }], &Leg { lane: 3, a: 0.0, b: 6.4 }, 0.0, 0.25);
        assert!(spot.y < -3.8 && spot.y > -4.3, "{spot:?}");
    }

    /// At the kerb with the green on for `left` s more: what person `id` decides, and (walked
    /// on for 30 s with the red coming when the green is over) where they end up and the
    /// longest they stood on the carriageway.
    fn at_green(id: u32, left: f32) -> (Option<&'static str>, DVec2, f32) {
        let (net, mut ped) = crossing(true);
        let lights = FakeLights::new(GREEN, left);
        let mut walk = across();
        walk.legs[0] = Leg { lane: 2, a: 19.5, b: 20.0 };
        let mut m = me(id, DVec2::new(-0.5, -5.5), 1.2);
        let dt = 0.05;
        let (mut first, mut still, mut longest) = (None, 0.0f32, 0.0f32);
        let mut t = 0.0f32;
        while t < 30.0 {
            let rest = left - t;
            if rest <= 0.0 {
                lights.state.set(RED);
                lights.left.set(30.0);
            } else {
                lights.left.set(rest);
            }
            let v = frame(&net, &mut ped, Some(&lights), &[], t as f64, &mut m, &mut walk, dt);
            if first.is_none() && m.decision.is_some() {
                first = m.decision;
            }
            if in_road(m.pos) {
                still = if v.length() < 0.05 { still + dt } else { 0.0 };
                longest = longest.max(still);
            }
            t += dt;
        }
        (first, m.pos, longest)
    }

    #[test]
    fn p2_the_whole_crossing_is_timed_at_the_kerb() {
        // 11 m at 1.2 m/s: at the far kerb 10.2 s after stepping on; with no light for the
        // cars the clearance is 2 s
        assert!(makes_it(9.0, 11.0, 10.2));
        assert!(!makes_it(4.0, 6.0, 10.2));
        // past the middle before the red, across in the clearance
        assert!(makes_it(6.0, 20.0, 10.2));
        assert!(!makes_it(4.0, 20.0, 10.2));
        let id = (1..200u32).find(|&id| !late_stepper(id, 2) && !late_stepper(id, 3)).unwrap();
        // not enough green left: they wait at the kerb, off the carriageway
        let (first, end, _) = at_green(id, 4.0);
        assert_eq!(first, Some("the green ends before they would be across"));
        assert!(end.y < -5.0, "{end:?}");
        // enough: across
        let (first, end, stood) = at_green(id, 20.0);
        assert_eq!(first, Some("go"));
        assert!(end.y > 5.3, "{end:?}");
        assert!(stood < 0.2);
    }

    #[test]
    fn p2_a_late_stepper_goes_and_the_change_clears_them() {
        let share = (1..2001u32).filter(|&id| late_stepper(id, 3)).count() as f64 / 2000.0;
        assert!((share - PED_LATE_CHANCE).abs() < 0.03, "{share}");
        let id = (1..400u32).find(|&id| late_stepper(id, 3)).unwrap();
        let (first, end, stood) = at_green(id, 4.0);
        assert_eq!(first, Some("go"));
        // caught before the middle: back on the near pavement, never standing in the road
        assert!(end.y < -5.0, "{end:?}");
        assert!(stood < 0.2, "{stood}");
    }

    /// [round 6 B3] A bus waiter's place keeps `max(w, 2.6)/2 + 0.58` from a street lane's
    /// axis (case02: waiters 1.46-1.55 m off bus lane 622, 1 m wide, held 11.6 m coaches
    /// 148 s): a place in that band moves out of it; one clear of it, one on another level,
    /// and one between two lanes (a platform) stay where they are; seats stay.
    #[test]
    fn b3_bus_waiters_stand_clear_of_the_widest_vehicle_in_the_lane() {
        use omsi_sim::traffic::LaneBuilder;
        let p = |x: f64, y: f64| DVec3::new(x, y, 0.0);
        let road = |width: f32, other: bool| {
            let mut lanes = vec![LaneBuilder::polyline(vec![p(-50.0, 0.0), p(50.0, 0.0)], LaneKind::Street, width)];
            if other {
                lanes.push(LaneBuilder::polyline(vec![p(50.0, 3.5), p(-50.0, 3.5)], LaneKind::Street, width));
            }
            let mut net = Network { lanes, ..Default::default() };
            net.link(1.5);
            net
        };
        let net = road(1.0, false);
        let reach = 1.3 + 0.28 + 0.3;
        for y in [1.46, 1.5, 1.55, -1.5] {
            let q = clear_of_lanes(&net, p(3.0, y));
            assert!((q.y.abs() - reach).abs() < 0.02 && q.y.signum() == f64::signum(y) && (q.x - 3.0).abs() < 1e-6, "{y} -> {q:?}");
        }
        assert_eq!(clear_of_lanes(&net, p(3.0, 2.2)), p(3.0, 2.2), "clear already");
        assert_eq!(clear_of_lanes(&net, DVec3::new(3.0, 1.5, 6.0)), DVec3::new(3.0, 1.5, 6.0), "on a bridge above");
        // a wide lane: its own half width counts
        assert!((clear_of_lanes(&road(3.5, false), p(0.0, 2.0)).y - (1.75 + 0.58)).abs() < 0.02);
        // between two lanes 3.5 m apart there is no clear place within a metre: it stays
        assert_eq!(clear_of_lanes(&road(1.0, true), p(0.0, 1.75)), p(0.0, 1.75));
        // a stop's places: the standing one moves, the bench does not
        let mut stop = PaxStop {
            name: String::new(),
            alias: String::new(),
            pos: p(0.0, 1.5),
            heading: 90.0,
            gather: p(0.0, 1.5),
            spots: vec![WaitSpot { pos: p(1.0, 1.5), face: 0.0, height: 0.0 }, WaitSpot { pos: p(2.0, 1.5), face: 0.0, height: 0.45 }, WaitSpot { pos: p(3.0, 3.0), face: 0.0, height: 0.0 }],
            taken: vec![false; 3],
            enter_max: 1.0,
            enter_min: 0.0,
            length: 30.0,
            lane: None,
            was_near: false,
            near: false,
            clock_ms: 0.0,
            want: 0,
            factor: 1.0,
            buses: Vec::new(),
            dests: Vec::new(),
            lines: Vec::new(),
            cleared: false,
            kerb_shift: DVec3::ZERO,
        };
        assert_eq!(clear_stop_of_lanes(&mut stop, &net), vec![0]);
        assert!(stop.cleared && (stop.spots[0].pos.y - reach).abs() < 0.02 && stop.spots[1].pos.y == 1.5 && stop.spots[2].pos.y == 3.0);
        assert!((stop.kerb_shift.y - (reach - 1.5)).abs() < 0.02);
    }
}


/// A heading in 0..360 degrees.
/// A line a passenger says: the sample and where they stand.
pub struct VoiceLine {
    pub position: DVec3,
    pub path: std::path::PathBuf,

}

/// How much of its probability a day ticket keeps at a time of day (seconds): rising from
/// nothing at midnight to all of it at 9:00, as the ticket packs describe it, then falling
/// on OMSI's line.
fn day_ticket_factor(t: f64) -> f32 {
    let t = t.rem_euclid(86_400.0);
    let rise = t / 32_400.0;
    let fall = 1.0 - (t - 32_400.0) / (88_776.0 - 32_400.0);
    rise.min(fall).clamp(0.0, 1.0) as f32
}

fn wrap_heading(h: f64) -> f64 {
    h.rem_euclid(360.0)
}

/// The angle between two headings (degrees, 0..180).
fn angle_between(a: f64, b: f64) -> f64 {
    ((b - a + 540.0).rem_euclid(360.0) - 180.0).abs()
}
