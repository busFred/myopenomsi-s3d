//! The frame: positions, occupancy, lights, then perception → claims → plan → act for
//! every vehicle. `Simulation` owns the junction table, the light programs and the
//! frame's inputs (players, people, parked cars); the vehicles and the network are the
//! caller's, so the app keeps its own vehicle instances beside them and a test needs
//! neither.

use crate::driver::{DriverParams, Impatience, VehicleClass, STANDING_SPEED};
use crate::junction::Junctions;
use crate::perception::{self, Perception, Seen};
use crate::plan::{self, Plan, Start, Why};
use crate::reservation::{self, Claim, Grant, Hold};
use glam::{DVec2, DVec3};
use hashbrown::HashMap;
use omsi_sim::collision::Obb;
use omsi_sim::traffic::{AiState, LaneKind, Lead, Network, TrafficLightController};

/// A player's (or another LAN player's) vehicle as the traffic sees it.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PlayerView {
    pub id: u32,
    pub pos: DVec3,
    /// Degrees, clockwise from north.
    pub heading: f32,
    pub half_len: f32,
    pub half_width: f32,
    /// Signed along the heading (m/s, negative reversing).
    pub speed: f32,
    /// Seconds standing (|speed| < 0.3).
    pub still: f32,
    /// 0 off, 1 left, 2 right, 3 hazard.
    pub blinker: u8,
    /// Seconds the indicator has been on.
    pub signalling: f32,
    /// `TrafficPriority` set: the vehicle has the right of way everywhere.
    pub priority: bool,
}

impl PlayerView {
    pub fn fwd(&self) -> DVec2 {
        let h = (self.heading as f64).to_radians();
        DVec2::new(h.sin(), h.cos())
    }

    /// The box, reaching `ahead` metres further forward.
    pub fn obb(&self, ahead: f64, margin: f64) -> Obb {
        let f = self.fwd();
        let c = self.pos.truncate() + f * (ahead * 0.5);
        Obb { center: c, half: DVec2::new(self.half_width as f64 + margin, self.half_len as f64 + ahead * 0.5 + margin), heading: (self.heading as f64).to_radians(), z0: self.pos.z - 1.0, z1: self.pos.z + 3.0, velocity: f * self.speed as f64, mass: 0.0, pole: None, id: -1 }
    }
}

/// A person on or by the road.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Person {
    pub pos: DVec3,
    pub vel: DVec2,
    /// On a crossing (a zebra, a signalled crossing): cars stop for them further away.
    pub on_crossing: bool,
    /// Seconds standing still.
    pub still: f32,
    /// Waiting at a bus stop: a bus keeps well in, a car drives by.
    pub waiting_for_bus: bool,
    /// Waiting at a kerb to cross (a red light, a car to pass, yielding to a vehicle), off
    /// the carriageway: clear of a vehicle whose footprint they stand outside.
    #[serde(default)]
    pub at_kerb: bool,
}

/// A parked car beside a lane.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Parked {
    pub s: f32,
    /// Lateral offset of its middle from the lane (m, + right).
    pub lateral: f32,
    pub half_width: f32,
    pub half_len: f32,
}

/// A squeeze is registered on the neighbour lane its wheels go over the line into from this
/// much lateral offset (m): out of its lane's middle enough for the body to be over the line.
const WHEELS_OUT: f32 = 0.3;

/// Overtaking on the oncoming lane, or squeezing past something inside the lane.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Passing {
    /// The oncoming lane used, None when the car stays in its own lane.
    pub lane: Option<usize>,
    /// The lateral offset driven (m, signed).
    pub side: f32,
    /// Odometer reading at which the car is past the obstacle and may come back.
    pub until: f32,
    /// Odometer reading of the obstacle's near end.
    pub block: f32,
    /// Metres of way over which the car pulls out, and over which it comes back in.
    pub out_len: f32,
    pub back_len: f32,
    /// The oncoming traffic came: stop short of the obstacle and come back in.
    pub aborted: bool,
    /// Odometer reading at which an aborted pass holds the car.
    pub hold: f32,
    /// Seconds of indicating left before the car pulls out (it signals the side it moves to
    /// `SIGNAL_BEFORE_CHANGE` ahead, as a lane change does, and until it is back in its lane).
    #[serde(default)]
    pub wait: f32,
    /// The vehicle passed (its id; a player's: `PLAYER_ID_BASE` + its id), if one.
    #[serde(default)]
    pub obstacle: Option<u64>,
    /// The neighbour lane the driver-side wheels go over the line into for an in-lane squeeze
    /// (`plan::WHEELS_OVER_LINE`): the body is registered there too while it is out.
    #[serde(default)]
    pub beside: Option<usize>,
}

impl Passing {
    /// The way over which a car pulls `side` metres out: steep, three metres of way per
    /// metre sideways (the bay ramp of eight is for comfort, not for a car edging out
    /// from behind something).
    pub fn out_len_for(side: f32) -> f32 {
        (side.abs() * 3.0).clamp(5.0, 12.0)
    }

    pub fn back_len_for(side: f32) -> f32 {
        (side.abs() * 6.0).clamp(8.0, 20.0)
    }

    /// The indicator of the side the car moves to (1 left, 2 right).
    pub fn blinker(&self) -> i32 {
        if self.side > 0.0 {
            2
        } else {
            1
        }
    }

    /// The bent way, as (odometer, offset) points from `odo` and the offset `from` there.
    pub fn curve(&self, odo: f32, from: f32) -> Vec<(f32, f32)> {
        let level = self.until.max(odo + self.out_len + 0.5);
        vec![(odo, from), (odo + self.out_len, self.side), (level, self.side), (level + self.back_len, 0.0)]
    }
}

/// What the app tells the planner about this vehicle this frame.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct External {
    /// A stop the bus service asks for (distance from the origin).
    pub stop_at: Option<f32>,
    /// Lateral offset into a bay.
    pub bay: Option<f32>,
    /// Indicator off whatever the planner says (doors open).
    pub blinker_off: bool,
    /// At its stop: standing there is service, not a jam.
    pub at_stop: bool,
    /// Seconds the bus has stood at its stop (others pass it after a while).
    pub standing_for: f32,
    /// `TrafficPriority`: goes first at junctions.
    pub priority: bool,
    /// [round 6 B2] A timetable bus whose trip is over at the end of its track (no stop left,
    /// the route complete, its last lane with a way on): it drives on off its route's end
    /// rather than standing there (`plan` rule 10).
    #[serde(default)]
    pub trip_over: bool,
}

/// The last decision, kept for consumers and the debug overlay. Other vehicles are named
/// by id: the caller may take vehicles off the road between frames.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct Decision {
    pub why: Why,
    pub lead: Option<Lead>,
    pub lead_seen: Option<Seen>,
    /// The vehicle followed (its id).
    pub lead_id: Option<u64>,
    pub stop: Option<f32>,
    pub held: bool,
    pub hold: Option<Hold>,
    /// Vehicle id waited for at a junction (a player: `PLAYER_ID_BASE` + its id).
    pub wait_for: Option<u64>,
    /// Another vehicle's body (not on the way) that held this one (its id).
    pub body_block: Option<u64>,
}

/// Ids of the players in `Decision::wait_for`, above any vehicle's.
pub const PLAYER_ID_BASE: u64 = 1 << 62;

/// The lane a player's vehicle goes on into from `lane`: the only one, or at a fork the one
/// its indicator points to (the furthest turn that way, a lane flagged with that turn first),
/// else the straightest (a lane without a turn flag first).
fn player_next(net: &Network, lane: usize, blinker: u8) -> Option<usize> {
    let l = net.lanes.get(lane)?;
    let nexts: Vec<usize> = l.next.iter().copied().filter(|&n| net.lanes.get(n).is_some_and(|x| x.kind == LaneKind::Street && !x.points.is_empty())).collect();
    if nexts.len() <= 1 {
        return nexts.first().copied();
    }
    let h = l.end_heading();
    // (heading change into and along each, degrees: negative to the left)
    let bend = |n: usize| omsi_sim::traffic::wrap_deg(net.lanes[n].end_heading() - h);
    let want = blinker as i32;
    let key = |n: usize| -> (u8, f32) {
        let flag = net.lanes[n].turn;
        match want {
            1 => (u8::from(flag != 1), bend(n)),
            2 => (u8::from(flag != 2), -bend(n)),
            _ => (u8::from(flag != 0), bend(n).abs()),
        }
    };
    nexts.into_iter().min_by(|&a, &b| {
        let (ka, kb) = (key(a), key(b));
        ka.0.cmp(&kb.0).then(ka.1.total_cmp(&kb.1))
    })
}

/// One AI vehicle in the traffic: its way (`state`), its driver and its manoeuvres.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Vehicle {
    pub id: u64,
    pub class: VehicleClass,
    pub state: AiState,
    pub params: DriverParams,
    pub impatience: Impatience,
    pub half_width: f32,
    /// Seconds standing (not at a bus stop).
    pub stopped: f32,
    /// (odometer at the last progress, seconds without any).
    pub progress: (f32, f32),
    /// A random vehicle that stood `GIVE_UP_UNSEEN` held by nothing: it asks to be taken off
    /// the road once out of sight. The crate sets it and clears it when the vehicle moves
    /// again (`Vehicle::give_up_clock`); a timetable bus's is the app's.
    pub gone: bool,
    /// [round 7 B1] Seconds at a stretch it has stood held by nothing: no reason in its plan
    /// (lead, light, yield, exit full, people, merge, ...), no lead close, no junction hold,
    /// no foe or body it waits for, and not moving faster than `UNHELD_SPEED`.
    #[serde(default)]
    pub held_by_nothing: f32,
    /// Just put on the road: its speed is trimmed to what is ahead.
    pub fresh: f32,
    /// Seconds before a car leaving a parking space moves.
    pub pull_out: f32,
    pub passing: Option<Passing>,
    /// Past the line on green/yellow: goes through on red too.
    pub amber: Option<(usize, usize)>,
    pub external: External,
    pub last: Decision,
    /// Id of the vehicle followed last frame.
    pub lead_id: Option<u64>,
    /// Indicator shown, and since when (debounced).
    pub blinker_shown: (i32, f64),
    /// World position of the origin and heading (degrees), from the way.
    pub pos: DVec3,
    pub heading: f32,
    /// Standing as of its last claim at a junction, with hysteresis:
    /// from under 0.3 m/s until over 1 m/s (`Claim::standing`). Kept with the vehicle, so a
    /// keyframe restores it (recordings before format 2 lack it: false).
    #[serde(default)]
    pub claim_standing: bool,
    /// The light (controller, light) whose line the vehicle passed while it let vehicles in
    /// (green or yellow), as long as it is past that line: in the box when it turned, the
    /// vehicle clears it (`JunctionView::clearing`). One put down past a red line, or that ran
    /// it, has none.
    #[serde(default)]
    pub passed_on_go: Option<(usize, usize)>,
    /// With `amber`: the odometer reading at the line it marks.
    #[serde(default)]
    pub amber_line: Option<f32>,
    /// The zipper merge it last went first at: (the merge, its piece there, the time). The
    /// junction grants' memory of who went last (`reservation::Context::zipper`) is built
    /// from these, so a keyframe restores it.
    #[serde(default)]
    pub zip_won: Option<((usize, usize, usize), usize, f64)>,
    /// A lane change under way: (its progress when it last moved on, seconds since while
    /// standing). Frozen for `FROZEN_CHANGE` it is settled where the body is
    /// (`AiState::settle_change`).
    #[serde(default)]
    pub change_still: (f32, f32),
    /// Holding the vehicle with this id at a junction while it holds this one (each the
    /// other's `wait_for` or `body_block`), and for how many seconds (`MUTUAL_HOLD`).
    #[serde(default)]
    pub mutual: Option<(u64, f32)>,
    /// Seconds since the mutual hold (`mutual`) was last seen: the partner is kept over a gap
    /// shorter than `MUTUAL_GAP` (a lead flickering between a body and none).
    #[serde(default)]
    pub mutual_gap: f32,
    /// Eased off for the vehicle with this id at this junction, since then (s): it stays held
    /// and its claim does not stand in the other's way (`Claim::yields_now`) until the other
    /// has left the meeting places of the two.
    #[serde(default)]
    pub ease_for: Option<(u64, usize, f32)>,
}

/// [ROLLBACK r4d-B4] A timetable bus creeping up to its stop slower than this (m/s) has not
/// moved off: its standing time (`Vehicle::stopped`) runs on (`STANDING_SPEED` would reset
/// it in a 0.3-0.4 m/s creep).
pub const CREPT_OFF_SPEED: f32 = 0.5;

/// [round 7 B1] A random vehicle that has stood held by nothing (`Vehicle::held_by_nothing`)
/// this long at a stretch (s) gives up (`Vehicle::gone`): five minutes, SUMO's default
/// `--time-to-teleport` of 300 s for a vehicle that waited too long (sumo.md 5.3,
/// https://sumo.dlr.de/docs/Simulation/Why_Vehicles_are_teleporting.html).
pub const GIVE_UP_UNSEEN: f32 = 5.0 * 60.0;
/// Faster than this (m/s) a vehicle is moving: its give-up clock restarts and a vehicle that
/// had given up is back in the traffic (SUMO's waiting time counts below 0.1 m/s, same page).
pub const UNHELD_SPEED: f32 = 0.1;

/// Two vehicles holding each other at a junction where they have a meeting place negotiate:
/// one eases off (`Vehicle::ease_for`) at the first frame both stand refused for each other
/// by the junction (`Decision::wait_for` both ways: the resolver's answer will not change by
/// itself); a pair held by a body (`body_block`) on one side negotiates after this long (s),
/// times the driver's `mutual_jitter`. (Round 4's 6 s clock for every pair never fired in a
/// drive: four pairs stood 157 pair-seconds creeping at 0.2 m/s, review r5.)
pub const MUTUAL_HOLD: f32 = 6.0;
/// A mutual hold not seen for less than this (s) keeps its partner and its clock: a lead
/// flickering every 0.4 s between a body and none restarted the clock each time (x10 case03,
/// the dice rolled once in 54 min).
pub const MUTUAL_GAP: f32 = 1.0;
/// `MUTUAL_HOLD` is scaled per driver by a factor in this range (the lower id's, seeded by
/// its id) so that unrelated pairs do not tick in lockstep.
pub const MUTUAL_JITTER: (f32, f32) = (0.5, 1.5);

/// The factor of `MUTUAL_JITTER` for the driver with vehicle id `id`, the same in every run.
pub fn mutual_jitter(id: u64) -> f32 {
    let mut x = id.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    let u = (x >> 40) as f32 / (1u64 << 24) as f32;
    MUTUAL_JITTER.0 + (MUTUAL_JITTER.1 - MUTUAL_JITTER.0) * u
}
/// An eased-off car takes part again after this long (s) whatever the other does.
pub const EASE_MAX: f32 = 30.0;

/// Which of two vehicles (ids `a`, `b`) in a mutual hold at junction `j` eases off: a hash of
/// the lower id, the higher and the junction, the same in every run.
pub fn eases_off(a: u64, b: u64, j: usize) -> u64 {
    let (lo, hi) = (a.min(b), a.max(b));
    let mut x = lo.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ hi.rotate_left(21) ^ (j as u64).rotate_left(42);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    if x & 1 == 0 {
        lo
    } else {
        hi
    }
}

/// [round6 J5] A standing player's claim names as its lead the AI vehicle in its corridor
/// within this many metres of its front (review-r6 fix 7: a queue's gap, 2-3 m).
pub const PLAYER_LEAD_REACH: f32 = 3.0;
/// [round6 J5] ... its centre within this many metres of the player's axis (`traffic_cases
/// heads`' 2.5 m corridor for the queue in front of the player).
pub const PLAYER_LEAD_ACROSS: f32 = 2.5;
/// [round6 J5] ... or within this many metres when it moves off ahead of the player (`heads`'
/// 12 m from one car of a queue to the next).
pub const PLAYER_LEAD_AWAY: f32 = 12.0;
/// [round6 J5] ... and its box reaching no more than this far (m) back past the player's front
/// (review-r6j: a car alongside counted as the lead).
pub const PLAYER_LEAD_OVERLAP: f32 = 0.5;
/// [round6 J4] A player keeps the light of its lit lane until its rear is this far (m) past the
/// lane's start (`perception::signal_ahead`'s half metre for an AI car).
pub const PLAYER_LINE_PAST: f32 = 0.5;
/// Seconds a player stands before it counts as held where it is: blocked to the junction
/// claim (`Claim::blocked`), parked when beside the lane (`Simulation::player_parked_beside`).
pub const PLAYER_STILL: f32 = 2.0;
/// A standing player whose body is this far (m) clear of the lane's edge makes no junction
/// claim: what a car keeps off a kerb (`perception::KERB_CLEAR`).
pub const PARKED_CLEAR: f32 = 0.3;

/// The lateral offset (m, right positive) of player `p` from lane `lane` at `s` along it.
fn player_lateral(net: &Network, lane: usize, s: f32, p: &PlayerView) -> f32 {
    let (q, h) = net.lanes[lane].at(s.min(net.lanes[lane].length()));
    let hr = (h as f64).to_radians();
    let right = DVec2::new(hr.cos(), -hr.sin());
    (p.pos - q).truncate().dot(right) as f32
}

/// An aborted pass standing out there steps back into its lane at this pace (m/s) ...
pub const SIDESTEP_SPEED: f32 = 0.5;
/// ... until it is this near its lane's middle (m).
pub const SIDESTEP_DONE: f32 = 0.05;

/// A settled lane change's offset eases in place at this pace (m/s) while the car stands
/// (round 6 L3: it eased by distance only, and viz5 case05 #18836, settled 1.38 m out over
/// the lane it came from, stood 46 s behind #20220 with 66 m free ahead; a car edging over,
/// slower than `SIDESTEP_SPEED`'s step back of a pass given up).
pub const SETTLE_CREEP: f32 = 0.3;

/// Whether `v`'s offset is one a settled change left (`AiState::settled_offset`, set by
/// `settle_change` alone, kept in keyframes) and still easing off: no change, pass, bay,
/// pull-out or swerve since, and not a car just put on the road. (Round 6 L3 fix pass 1: the
/// lane-change cooldown a new car starts with let every fresh car standing with an offset -
/// 139 of 416, the pull-outs from the parking strips - slide sideways; x10 case08 #11387 slid,
/// started its pull-out pass 6 s late, and #11125 ran into it.)
fn settling(v: &Vehicle) -> bool {
    let st = &v.state;
    st.settled_offset && offset_left(v) && v.fresh <= 0.0 && v.pull_out <= 0.0 && v.external.bay.is_none()
}

/// A settled offset still there to ease off (`settling` without the vehicle's own state).
fn offset_left(v: &Vehicle) -> bool {
    let st = &v.state;
    st.change.is_none() && v.passing.is_none() && st.lateral.abs() > SIDESTEP_DONE && st.lateral_target == 0.0 && st.lateral_ramp.1 == 0.0
}

/// Seconds a lane change may stand without moving on before it is settled where the car is:
/// done in place, or given up (`AiState::settle_change`).
pub const FROZEN_CHANGE: f32 = 5.0;

impl Vehicle {
    pub fn new(id: u64, class: VehicleClass, state: AiState, half_width: f32, seed: u64) -> Vehicle {
        let mut state = state;
        let params = DriverParams::seeded(class, seed);
        params.apply(&mut state);
        Vehicle {
            id,
            class,
            state,
            params,
            impatience: Impatience::default(),
            half_width,
            stopped: 0.0,
            progress: (0.0, 0.0),
            gone: false,
            held_by_nothing: 0.0,
            fresh: 1.5,
            pull_out: 0.0,
            passing: None,
            amber: None,
            external: External::default(),
            last: Decision::default(),
            lead_id: None,
            blinker_shown: (0, 0.0),
            pos: DVec3::ZERO,
            heading: 0.0,
            claim_standing: false,
            passed_on_go: None,
            amber_line: None,
            zip_won: None,
            change_still: (0.0, 0.0),
            mutual: None,
            mutual_gap: 0.0,
            ease_for: None,
        }
    }

    /// An empty vehicle to stand in while one is lent out.
    pub fn placeholder() -> Vehicle {
        Vehicle::new(0, VehicleClass::Car, AiState::new(0, 0.0, 0), 0.0, 0)
    }

    pub fn length(&self) -> f32 {
        self.state.front + self.state.rear
    }

    /// The body on the ground.
    pub fn obb(&self) -> Obb {
        Obb::vehicle(self.pos.truncate(), self.heading as f64, self.state.front as f64, self.state.rear as f64, self.half_width as f64)
    }

    /// [round 7 B1] The give-up clock, after this frame's decision (`last`): it runs while
    /// the vehicle stands held by nothing and restarts on any hold or on moving faster than
    /// `UNHELD_SPEED`; a random vehicle whose clock passes `GIVE_UP_UNSEEN` gives up
    /// (`gone`), and is back in the traffic once it moves. (Its gridlock is not a reason: a
    /// queue or a junction hold restarts the clock.)
    pub fn give_up_clock(&mut self, dt: f32) {
        let d = &self.last;
        let held = d.why != Why::None || d.held || d.hold.is_some() || d.wait_for.is_some() || d.body_block.is_some() || self.external.at_stop;
        let moving = self.state.speed > UNHELD_SPEED;
        if moving || held {
            self.held_by_nothing = 0.0;
        } else {
            self.held_by_nothing += dt;
        }
        if self.serves_stops() || self.is_rail() {
            return;
        }
        if moving {
            self.gone = false;
        } else if self.held_by_nothing > GIVE_UP_UNSEEN {
            self.gone = true;
        }
    }

    /// Standing still, held by something.
    pub fn is_held(&self) -> bool {
        self.state.speed < STANDING_SPEED && self.last.held
    }

    /// [round 7 A4] A timetable bus (`VehicleClass::serves_stops`).
    pub fn serves_stops(&self) -> bool {
        self.class.serves_stops()
    }

    /// A heavy vehicle: any bus or a truck (`VehicleClass::heavy`); a random heavy vehicle
    /// is `heavy() && !serves_stops()`.
    pub fn heavy(&self) -> bool {
        self.class.heavy()
    }

    pub fn is_rail(&self) -> bool {
        self.class == VehicleClass::Rail
    }
}

/// A vehicle's place on a lane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Occupant {
    pub vehicle: usize,
    pub s: f32,
    pub lateral: f32,
    /// Passing on this lane against its direction.
    pub foreign: bool,
}

/// Grid cell size of the vehicle and people grids (m).
const GRID: f64 = 50.0;

fn cell(p: DVec3) -> (i32, i32) {
    ((p.x / GRID).floor() as i32, (p.y / GRID).floor() as i32)
}

/// The traffic's shared state and this frame's inputs.
#[derive(Debug, Clone)]
pub struct Simulation {
    pub junctions: Junctions,
    pub lights: Vec<TrafficLightController>,
    pub time: f64,
    pub day_time: f64,
    pub time_scale: f64,
    pub players: Vec<PlayerView>,
    pub people: Vec<Person>,
    /// Pedestrians on footpath lanes: (lane, distance), for push buttons and zebras.
    pub walkers: Vec<(usize, f32)>,
    pub parked: HashMap<usize, Vec<Parked>>,
    pub occupancy: HashMap<usize, Vec<Occupant>>,
    /// The players' vehicles on the lanes, as of this frame: per lane, (player index,
    /// distance along it, lateral offset). The lanes beside, the room past a junction and
    /// the oncoming lane see them as they see any vehicle.
    pub player_occupancy: HashMap<usize, Vec<(usize, f32, f32)>>,
    /// Vehicles and people by 50 m cell, as of this frame.
    vehicle_grid: HashMap<(i32, i32), Vec<usize>>,
    people_grid: HashMap<(i32, i32), Vec<usize>>,
    /// Lane changes begun this frame: (lane changed onto, distance along it). A change
    /// into the same stretch later in the same frame is refused (`act`): the perceptions
    /// were taken before anybody moved, so two cars would move into one gap at once.
    pub changes_started: Vec<(usize, f32)>,
    pub left_hand: bool,
    /// Vehicles that reached a dead end this frame (indices, descending).
    pub removed: Vec<usize>,
    /// Vehicles counted standing at a red light this frame (diagnostics).
    pub held_at_red: usize,
    /// What the junction grants read besides the claims: the pairs of lights that are
    /// never green together, filled in as claimants meet.
    pub signal_groups: reservation::Context,
    /// This frame's claims and their grants (diagnostics: `traffic_cases watch`).
    pub claims: Vec<(Claim, Grant)>,
}

/// How far on (m) a pass coming back in looks for a body beside it, and runs straight when
/// one is: half a metre past its front (at least 2 m; its body turns with the way there).
fn return_look(st: &AiState) -> f32 {
    (st.front + 0.5).max(2.0)
}

impl Simulation {
    pub fn new(net: &Network, lights: Vec<TrafficLightController>) -> Simulation {
        Simulation {
            junctions: Junctions::compile(net),
            lights,
            time: 0.0,
            day_time: 0.0,
            time_scale: 1.0,
            players: Vec::new(),
            people: Vec::new(),
            walkers: Vec::new(),
            parked: HashMap::new(),
            occupancy: HashMap::new(),
            player_occupancy: HashMap::new(),
            vehicle_grid: HashMap::new(),
            people_grid: HashMap::new(),
            changes_started: Vec::new(),
            left_hand: net.left_hand,
            removed: Vec::new(),
            held_at_red: 0,
            signal_groups: reservation::Context::default(),
            claims: Vec::new(),
        }
    }

    /// Lanes from index `first` on are new: compile their junctions.
    pub fn network_grew(&mut self, net: &Network, first: usize) {
        self.junctions.extend(net, first);
    }

    /// The side the oncoming traffic is on as a lateral sign: -1 (left) driving on the
    /// right, +1 on a left-hand-traffic map.
    pub fn oncoming_sign(&self) -> f32 {
        if self.left_hand {
            1.0
        } else {
            -1.0
        }
    }

    /// The indicator a vehicle shows when pulling out towards the traffic.
    pub fn out_side(&self) -> i32 {
        if self.left_hand {
            2
        } else {
            1
        }
    }

    /// Indices of the vehicles within `r` metres of `p` (by grid cell: a few more).
    pub fn vehicles_near(&self, p: DVec3, r: f64) -> Vec<usize> {
        let (cx, cy) = cell(p);
        let n = (r / GRID).ceil() as i32;
        let mut out = Vec::new();
        for dx in -n..=n {
            for dy in -n..=n {
                if let Some(v) = self.vehicle_grid.get(&(cx + dx, cy + dy)) {
                    out.extend_from_slice(v);
                }
            }
        }
        out
    }

    /// Indices of the people within `r` metres of `p` (by grid cell: a few more).
    pub fn people_near(&self, p: DVec3, r: f64) -> Vec<usize> {
        let (cx, cy) = cell(p);
        let n = (r / GRID).ceil() as i32;
        let mut out = Vec::new();
        for dx in -n..=n {
            for dy in -n..=n {
                if let Some(v) = self.people_grid.get(&(cx + dx, cy + dy)) {
                    out.extend_from_slice(v);
                }
            }
        }
        out
    }

    /// Place every vehicle on the ground from its way.
    pub fn update_positions(&self, net: &Network, vehicles: &mut [Vehicle]) {
        for v in vehicles.iter_mut() {
            if net.lanes.get(v.state.lane).is_none() {
                continue;
            }
            let p = v.state.way_point(net, 0.0);
            let a = v.state.way_point(net, -omsi_sim::traffic::HEADING_CHORD);
            let b = v.state.way_point(net, omsi_sim::traffic::HEADING_CHORD);
            let d = (b - a).truncate();
            v.pos = p;
            if d.length_squared() > 1e-6 {
                v.heading = d.x.atan2(d.y).to_degrees() as f32;
            } else {
                v.heading = net.lanes[v.state.lane].at(v.state.s).1;
            }
        }
    }

    /// Which vehicle is where on which lane, and the grids.
    pub fn build_occupancy(&mut self, net: &Network, vehicles: &[Vehicle]) {
        self.occupancy.clear();
        self.vehicle_grid.clear();
        self.people_grid.clear();
        for (i, v) in vehicles.iter().enumerate() {
            let st = &v.state;
            if net.lanes.get(st.lane).is_none() {
                continue;
            }
            self.vehicle_grid.entry(cell(v.pos)).or_default().push(i);
            // (a car changing lanes is on both, where its body is from each)
            let offsets = st.change_offsets(net);
            self.occupancy.entry(st.lane).or_default().push(Occupant { vehicle: i, s: st.s, lateral: offsets.map(|o| o.1).unwrap_or(st.lateral), foreign: false });
            if let Some(p) = st.prev_lane {
                if st.s < st.rear + 1.0 && st.change.is_none() {
                    if let Some(pl) = net.lanes.get(p) {
                        self.occupancy.entry(p).or_default().push(Occupant { vehicle: i, s: pl.length() + st.s, lateral: st.lateral, foreign: false });
                    }
                }
            }
            if let Some(c) = st.change {
                self.occupancy.entry(c.to).or_default().push(Occupant { vehicle: i, s: c.s_to, lateral: offsets.map(|o| o.2).unwrap_or(0.0), foreign: false });
            }
            if let Some(p) = v.passing {
                if let (Some(opp), true) = (p.lane, st.lateral.abs() > 1.2) {
                    if let Some(ol) = net.lanes.get(opp) {
                        if let Some((os, _)) = ol.nearest_point(v.pos) {
                            self.occupancy.entry(opp).or_default().push(Occupant { vehicle: i, s: os, lateral: 0.0, foreign: true });
                        }
                    }
                }
                // (a squeeze with its wheels over the line: on the neighbour lane too, where its
                // body is from that lane's axis, so the cars there see it)
                if let (Some(b), true) = (p.beside, st.lateral.abs() > WHEELS_OUT) {
                    if let Some(bl) = net.lanes.get(b) {
                        if let Some((bs, _)) = bl.nearest_point(v.pos) {
                            let (q, h) = bl.at(bs.min(bl.length()));
                            let hr = (h as f64).to_radians();
                            let right = DVec2::new(hr.cos(), -hr.sin());
                            let lateral = (v.pos - q).truncate().dot(right) as f32;
                            self.occupancy.entry(b).or_default().push(Occupant { vehicle: i, s: bs, lateral, foreign: false });
                        }
                    }
                }
            }
        }
        for occ in self.occupancy.values_mut() {
            occ.sort_by(|a, b| a.s.total_cmp(&b.s));
        }
        for (k, p) in self.people.iter().enumerate() {
            self.people_grid.entry(cell(p.pos)).or_default().push(k);
        }
        self.player_occupancy.clear();
        for k in 0..self.players.len() {
            let p = self.players[k];
            let Some((lane, s)) = self.player_lane(net, &p) else { continue };
            self.player_occupancy.entry(lane).or_default().push((k, s, player_lateral(net, lane, s, &p)));
        }
    }

    /// The light that governs footpath lane `lane`: its own, else that of the pedestrian
    /// crossing it is a piece of.
    pub fn crossing_light(&self, net: &Network, lane: usize) -> Option<(usize, usize)> {
        self.junctions.crossing_light(net, lane)
    }

    /// Lanes starting just ahead of a player, and the ones after them, within `within`
    /// metres: what the player's vehicle will drive along.
    fn lanes_ahead_of(&self, net: &Network, p: &PlayerView, within: f32) -> Vec<(usize, f32)> {
        let f = p.fwd();
        let mut out: Vec<(usize, f32)> = Vec::new();
        for l in net.lanes_starting_near(p.pos, 25.0) {
            let lane = &net.lanes[l];
            if lane.kind != LaneKind::Street {
                continue;
            }
            let rel = (lane.start() - p.pos).truncate();
            let along = rel.dot(f);
            let across = rel.perp_dot(f).abs();
            if !(0.0..=25.0).contains(&along) || across > 6.0 || (lane.start().z - p.pos.z).abs() > 4.0 {
                continue;
            }
            if omsi_sim::traffic::wrap_deg(lane.start_heading() - p.heading).abs() > 60.0 {
                continue;
            }
            out.push((l, along as f32));
        }
        // and on from them
        let mut k = 0;
        while k < out.len() && out.len() < 24 {
            let (l, d) = out[k];
            let end = d + net.lanes[l].length();
            if end < within {
                for &n in &net.lanes[l].next {
                    if !out.iter().any(|o| o.0 == n) {
                        out.push((n, end));
                    }
                }
            }
            k += 1;
        }
        out
    }

    /// The lane a player's vehicle is on: (lane, distance along it), a street lane within
    /// 6 m heading the same way.
    fn player_lane(&self, net: &Network, p: &PlayerView) -> Option<(usize, f32)> {
        let (l, s, _) = net.lane_along(p.pos, p.heading as f64, LaneKind::Street, 6.0, 45.0)?;
        Some((l, s))
    }

    /// A player standing (`PLAYER_STILL`) with its body wholly beside the lane it is snapped
    /// to, by more than `PARKED_CLEAR`: parked beside the lanes, it is a body to perception
    /// (`player_on_way`) but no junction contender. (x10 case10: a bus 0.75 m clear of the
    /// lane, snapped to the right-turn piece 3.3 m away, was "standing in the merge" and held
    /// 9 cars 106 s.)
    fn player_parked_beside(net: &Network, lane: usize, s: f32, p: &PlayerView) -> bool {
        let lateral = player_lateral(net, lane, s, p);
        p.still > PLAYER_STILL && p.speed.abs() < STANDING_SPEED && lateral.abs() - p.half_width > net.lanes[lane].width * 0.5 + PARKED_CLEAR
    }

    fn update_lights(&mut self, net: &Network, vehicles: &[Vehicle], dt: f32) {
        for c in &mut self.lights {
            c.request.iter_mut().for_each(|r| *r = false);
        }
        let ask = |lights: &mut Vec<TrafficLightController>, ctl: usize, li: usize, gap: f32| {
            if let Some(c) = lights.get_mut(ctl) {
                if gap <= c.approach_dist(li) {
                    if let Some(r) = c.request.get_mut(li) {
                        *r = true;
                    }
                }
            }
        };
        for v in vehicles {
            for (l, d0) in perception::way_lanes(net, &v.state, 160.0) {
                if let Some((ctl, li)) = net.lanes[l].traffic_light {
                    ask(&mut self.lights, ctl, li, d0 - v.state.front);
                }
            }
        }
        for p in &self.players {
            for (l, d0) in self.lanes_ahead_of(net, p, 160.0) {
                if let Some((ctl, li)) = net.lanes[l].traffic_light {
                    ask(&mut self.lights, ctl, li, d0 - p.half_len);
                }
            }
        }
        // a walker presses the button of a lit pedestrian lane they stand on or at the end
        // of (the lanes before and after theirs too: a crossing's lit stub is short)
        let press = |lights: &mut Vec<TrafficLightController>, ctl: usize, li: usize, dist: f32| {
            if let Some(c) = lights.get_mut(ctl) {
                if dist <= c.approach_dist(li).min(10.0) {
                    if let Some(r) = c.request.get_mut(li) {
                        *r = true;
                    }
                }
            }
        };
        for &(l, s) in &self.walkers {
            let Some(lane) = net.lanes.get(l) else { continue };
            let to_end = (lane.length() - s).max(0.0);
            if let Some((ctl, li)) = lane.traffic_light {
                press(&mut self.lights, ctl, li, to_end.min(s.max(0.0)));
            }
            for &n in &lane.next {
                if let Some((ctl, li)) = net.lanes[n].traffic_light {
                    press(&mut self.lights, ctl, li, to_end);
                }
            }
            for &p in net.prev.get(l).map(|v| v.as_slice()).unwrap_or(&[]) {
                if let Some((ctl, li)) = net.lanes[p].traffic_light {
                    press(&mut self.lights, ctl, li, s.max(0.0));
                }
            }
        }
        for c in &mut self.lights {
            c.start(self.day_time);
            c.advance(dt);
        }
    }

    /// One frame of `dt` seconds for every vehicle.
    pub fn tick(&mut self, net: &Network, dt: f32, vehicles: &mut [Vehicle]) {
        self.time += dt as f64;
        self.day_time += dt as f64 * self.time_scale;
        self.removed.clear();
        self.changes_started.clear();
        self.held_at_red = 0;
        self.update_positions(net, vehicles);
        self.build_occupancy(net, vehicles);
        self.update_lights(net, vehicles, dt);

        // perception: read-only over everything
        let perceptions: Vec<Perception> = (0..vehicles.len()).map(|i| perception::perceive(net, self, vehicles, i)).collect();

        // claims and grants at the junctions
        let mut claims: Vec<Claim> = Vec::new();
        let mut claim_of: Vec<Option<usize>> = vec![None; vehicles.len()];
        // the claimants' standing hysteresis as of this claim, written
        // back once all claims are filed
        let mut standing_now: Vec<(usize, bool)> = Vec::new();
        // the line each vehicle has passed on green or yellow, written back with them
        let mut passed_now: Vec<(usize, Option<(usize, usize)>)> = Vec::new();
        let mut stale_amber: Vec<usize> = Vec::new();
        for (i, (v, p)) in vehicles.iter().zip(&perceptions).enumerate() {
            let passed = p.junction.as_ref().and_then(|j| {
                let (ctl, li, _) = j.light.filter(|_| j.past_line)?;
                let lets_in = self.lights.get(ctl).is_some_and(|c| !matches!(TrafficLightController::aspect(c.state(li)), omsi_sim::traffic::Aspect::Red | omsi_sim::traffic::Aspect::RedYellow));
                if lets_in || v.passed_on_go == Some((ctl, li)) {
                    Some((ctl, li))
                } else {
                    None
                }
            });
            if passed != v.passed_on_go {
                passed_now.push((i, passed));
            }
            // the mark of a line taken on green or yellow (`Vehicle::amber`) is for that line:
            // gone once neither the light ahead nor the junction's light is that light, or once
            // the car is past that line and a line of the same light lies ahead (it stayed on
            // after the junction, and the next approach under the same signal group went over
            // its red on it: viz3 case01, light 1.0 lights three approaches)
            if let Some(a) = v.amber {
                let ahead = p.signal.filter(|s| (s.ctl, s.light) == a);
                let here = p.junction.as_ref().and_then(|j| j.light).map(|(c, l, _)| (c, l));
                let left = ahead.is_none() && here != Some(a);
                let another = ahead.is_some_and(|s| s.dist > v.state.front + 0.5) && v.amber_line.is_some_and(|at| v.state.odometer > at + v.length() + 1.0);
                // (or the line ahead lies further than a mark is ever taken at this speed:
                // `plan::light_call`, a yellow within "comfortable", kept `MARK_REACH` beyond it)
                let far = ahead.is_some_and(|s| s.dist - v.state.front > plan::comfortable_stop(v.state.speed, &v.params) + plan::MARK_REACH);
                if left || another || far {
                    stale_amber.push(i);
                }
            }
            let Some(j) = p.junction.as_ref() else { continue };
            let st = &v.state;
            // a claim is filed from eight seconds before the meeting place (at least 60 m,
            // at most 220 m): the drivers who give way to this one see it coming in time
            // to stop, not only once it is a braking distance from its own line
            let horizon = (8.0 * st.speed + 30.0).clamp(60.0, 220.0);
            if !j.inside && j.conflict_start - st.front > horizon {
                continue;
            }
            // committed: too close to the next meeting place it has not entered to stop before
            // it even braking hard, or past every one it has (in the last it is in). A driver
            // who can still stop, if not comfortably, gives way as the rules say (two cars
            // arriving together at an unmarked junction both took themselves for committed
            // and met in the middle); one on the lead-in piece of a long object is inside but
            // not yet committed; one standing in the box short of a later stretch it waits
            // for (`plan`: the stop before the stretch waited for) is committed to the ones it
            // is in, not to that one (`reservation`: a meeting place I am in). (at the line a
            // driver stops hard if it must; one already inside does not slam on in the middle
            // of the junction, it goes on unless it can stop gently)
            let next_ahead = self.junctions.foes_along(j.junction, &j.chain).iter().map(|f| j.entry + f.mine.0).filter(|&d| d - st.front > 0.5).fold(f32::INFINITY, f32::min);
            let next = if next_ahead.is_finite() { next_ahead } else { j.conflict_start };
            let room = next - st.front - 0.6;
            let brake = if j.inside { st.decel * 1.5 } else { plan::HARD_BRAKE };
            let cannot_stop = plan::cannot_stop(st.speed, room, brake);
            let committed = (next_ahead.is_infinite() && j.conflict_start - st.front <= 0.5) || cannot_stop;
            let limit = net.lanes.get(st.lane).map(|l| l.speed_limit_kmh / 3.6 * st.desire).unwrap_or(13.0);
            // the light at this junction's line, if the driver is held by one: any light on
            // the way before the first meeting place with another way, wherever the object
            // begins (`JunctionView::light_line`: the old test, the light within 1 m of the
            // object's entry, never saw a car standing at BeijingCity's lit stub as held by it)
            let light_line = j.light_line;
            let (light_red, until_go) = match p.signal {
                // (clearing the box: the light it passed holds it no more,
                // `JunctionView::clears_past`)
                Some(sig) if j.clears_past(&sig, st.front) => (false, None),
                // (or, clearing the box, a later line of the way through: it holds the car as any,
                // `JunctionView::clears_past`, and the car standing there is held by it, not
                // clearing; viz case06 #4599 stood 50 s at the second red of a long object,
                // committed to the cars waiting for it)
                Some(sig) if (sig.dist <= light_line + 1.0 || (j.clearing && sig.dist < j.entry + j.way_len)) && !matches!(sig.aspect, omsi_sim::traffic::Aspect::Green | omsi_sim::traffic::Aspect::GreenYellow | omsi_sim::traffic::Aspect::Dark) => {
                    // not held by a light it goes through: the plan's own
                    // call (too late to stop, the yellow it takes), or the mark of one
                    // that took the line on green or yellow and is close to it (queued
                    // there, it goes on as soon as its leader moves, whatever the light)
                    let call = plan::light_call(v, p, &sig);
                    let through = call.go || call.marked_through;
                    if through {
                        (false, None)
                    } else {
                        (true, sig.until_go)
                    }
                }
                _ => (false, None),
            };
            // standing at its red short of the first meeting place with
            // another way, a car is committed to nothing: the merge of the lead-in with its
            // own stub put its "first meeting place" behind its bumper, and the foes took
            // the car waiting at its red for one already on its way across
            let committed = committed && !(light_red && st.speed < 0.5 && light_line - st.front > 0.5);
            // standing, with hysteresis (`Claim::standing`):
            // from under 0.3 m/s until over 1 m/s, remembered from one claim to the next
            let was_standing = v.claim_standing;
            standing_now.push((i, st.speed < STANDING_SPEED || (was_standing && st.speed <= 1.0)));
            claim_of[i] = Some(claims.len());
            claims.push(Claim {
                chain: j.chain.clone(),
                was_standing,
                light_red,
                until_go,
                lead: match p.leader.map(|l| l.what) {
                    Some(Seen::Vehicle(o)) | Some(Seen::Body(o)) => Some(o),
                    _ => None,
                },
                vehicle: i,
                junction: j.junction,
                movement: j.movement,
                s: -j.entry,
                speed: st.speed,
                accel: st.accel,
                v_max: limit.min(st.max_speed_kmh / 3.6),
                front: st.front,
                rear: st.rear,
                gap: v.impatience.gap(st.accept_gap),
                committed,
                exit_room: j.exit_room,
                impatience: v.impatience,
                absolute: v.is_rail() || v.external.priority,
                held: v.is_held(),
                // (held by its lead, people, a zebra, a light, ... rather than by a junction's
                // answer or a full exit)
                blocked: v.is_held() && !matches!(v.last.why, Why::None | Why::Junction | Why::ExitFull),
                // (people or a zebra in front of it, or a lead that stands too: nothing that
                // lets it go on a schedule, as a light or a lead moving off does)
                // in the box when its light turned: it clears it (rule F)
                clearing: j.clearing && !light_red,
                stuck: v.is_held()
                    && match v.last.why {
                        Why::People | Why::Zebra => true,
                        Why::Lead => p.leader.is_some_and(|l| l.speed < STANDING_SPEED && l.gap < 12.0),
                        _ => false,
                    },
                lanes: perception::way_lanes(net, st, 150.0),
                light: j.light.map(|(c, l, _)| (c, l)),
                id: v.id,
                yields_now: None,
                // (where people, a zebra or a standing lead will stop the front)
                clear_ahead: [p.people_stop, p.zebra_stop, p.leader.filter(|l| l.speed < STANDING_SPEED).map(|l| st.front + l.gap)].into_iter().flatten().fold(f32::INFINITY, f32::min),
                // (held by its service, or standing `FAR_FOE_STOOD` s or more)
                // (not a stop requested ahead, `stop_at`: a bus held by its lead in the box has one
                // from its first frame, review-r6j)
                standing_long: st.speed < STANDING_SPEED && (v.external.at_stop || v.last.why == Why::Service || v.stopped >= reservation::FAR_FOE_STOOD),
                brake,
            });
        }
        // the players' vehicles claim their way like anybody else (their grants are not
        // read: the player drives): the junction on the way ahead of the lane under the
        // vehicle (its `next` lanes, straight on at a fork unless the indicator says
        // otherwise), the claim over every piece of it the way takes (`chain_along`, as for an
        // AI vehicle: a claim of the first piece alone never met the oncoming left turner
        // on the piece after it, viz3 case11); off the lanes, the junction on the lanes
        // starting ahead of it, on its first piece
        for (k, p) in self.players.iter().enumerate() {
            let v = p.speed.max(0.0);
            let horizon = (8.0 * v + 30.0).clamp(60.0, 220.0);
            // the way: (lane, distance from the vehicle's origin to the lane's start)
            let mut way: Vec<(usize, f32)> = Vec::new();
            if let Some((lane, s)) = self.player_lane(net, p) {
                // (parked beside the lane: no claim, `player_parked_beside`)
                if Self::player_parked_beside(net, lane, s, p) {
                    continue;
                }
                way.push((lane, -s));
                let mut d = net.lanes[lane].length() - s;
                let mut cur = lane;
                for _ in 0..16 {
                    if d > horizon + 40.0 {
                        break;
                    }
                    let Some(n) = player_next(net, cur, p.blinker) else { break };
                    way.push((n, d));
                    d += net.lanes[n].length();
                    cur = n;
                }
            }
            let on_way = way.iter().position(|(l, _)| self.junctions.of_lane.contains_key(l)).map(|at| {
                let j = self.junctions.of_lane[&way[at].0].0;
                (j, self.junctions.chain_along(j, &way[at..]), -way[at].1)
            });
            let hit = on_way.or_else(|| {
                // off the lanes: the way through chosen by the indicator, else the straightest
                let pick = |entries: &Vec<(usize, usize)>| {
                    entries.iter().copied().min_by_key(|&(j, m)| {
                        let turn = net.lanes[self.junctions.nodes[j].movements[m].lane].turn;
                        let want = p.blinker as i32;
                        if want != 0 && turn == want {
                            0
                        } else if turn == 0 {
                            1
                        } else {
                            2
                        }
                    })
                };
                let mut found: Option<(usize, usize, f32)> = None;
                for (lane, d0) in self.lanes_ahead_of(net, p, 80.0).into_iter().map(|(l, d)| (l, d - p.half_len)) {
                    let len = net.lanes[lane].length();
                    let hit = if let Some(&(j, m)) = self.junctions.of_lane.get(&lane) {
                        Some((j, m, -d0))
                    } else if let Some(entries) = self.junctions.entries.get(&lane) {
                        pick(entries).map(|(j, m)| (j, m, -(d0 + len)))
                    } else {
                        None
                    };
                    if let Some(h) = hit {
                        if found.map(|f| h.2 > f.2).unwrap_or(true) {
                            found = Some(h);
                        }
                    }
                }
                found.map(|(j, m, s_on)| (j, vec![(m, 0.0)], s_on))
            });
            let Some((j, chain, s_on)) = hit else { continue };
            let Some(&(m, _)) = chain.first() else { continue };
            // where its meeting places begin along the way through (the end of the way when
            // it meets nobody), and the next one its front has not entered
            let foes = self.junctions.foes_along(j, &chain);
            let way_len: f32 = chain.iter().map(|&(mm, _)| net.lanes[self.junctions.nodes[j].movements[mm].lane].length()).sum();
            let cs = foes.iter().map(|f| f.mine.0).fold(f32::INFINITY, f32::min);
            let cs = if cs.is_finite() { cs } else { way_len };
            let front_at = s_on + p.half_len;
            if cs - front_at > horizon {
                continue;
            }
            let next_ahead = foes.iter().map(|f| f.mine.0).filter(|&m0| m0 - front_at > 0.5).fold(f32::INFINITY, f32::min);
            let room = if next_ahead.is_finite() { next_ahead } else { cs } - front_at - 0.6;
            let cannot_stop = plan::cannot_stop(v, room, plan::HARD_BRAKE);
            let committed = (next_ahead.is_infinite() && cs - front_at <= 0.5) || cannot_stop;
            // its light, as an AI car's: the first lit lane of its way before the first
            // meeting place, red (or red and yellow) at its line with room to stop there
            // (not a stretch it runs side by side with another lane before its light, `Foe::side`;
            // nor a detector, `perception::detector_light`)
            let cs_line = foes.iter().filter(|f| !f.side).map(|f| f.mine.0).fold(f32::INFINITY, f32::min);
            let line_limit = -s_on + if cs_line.is_finite() { cs_line } else { way_len } + 1.0;
            // ([round6 J4] until its REAR is half a metre past the lit lane's start, as an AI
            // car's `signal_ahead`: the front 0.8 m past its stub, the player's bus lost its red,
            // stood blocked and held a green stream, case07 #37076 21 s)
            let lit = way.iter().filter(|&&(_, d)| d <= line_limit && d + p.half_len > -PLAYER_LINE_PAST).find_map(|&(l, d)| perception::stop_light(net, self, l).map(|lt| (lt, d)));
            let (light, light_red, until_go) = match lit.and_then(|((ctl, li), d)| self.lights.get(ctl).map(|c| ((ctl, li), d, c))) {
                Some((lt, d, c)) => {
                    let red = matches!(TrafficLightController::aspect(c.state(lt.1)), omsi_sim::traffic::Aspect::Red | omsi_sim::traffic::Aspect::RedYellow);
                    let gap = d - p.half_len;
                    // (as an AI car's `light_call`)
                    let can_stop = !plan::cannot_stop(v, gap + 0.5, plan::HARD_BRAKE);
                    if red && can_stop {
                        (Some(lt), true, c.time_until_go(lt.1))
                    } else {
                        (Some(lt), false, None)
                    }
                }
                None => (None, false, None),
            };
            // standing more than 2 s (at a stop, in a queue, waiting to turn): blocked, as an
            // AI car held by its lead; it comes from rest when it moves off (an AI car waited
            // 26 s for the player's bus standing 47 m away, viz3 case08 #21332)
            let blocked = p.still > PLAYER_STILL && p.speed.abs() < STANDING_SPEED && !light_red;
            // [round6 J5] standing, its lead: the AI vehicle in its corridor, its box
            // within `PLAYER_LEAD_REACH` ahead of the front and its centre within
            // `PLAYER_LEAD_ACROSS` of the axis (`traffic_cases heads`' queue in front of the
            // player), so that a foe the player stands behind does not wait for it (`reservation`:
            // a standing foe whose lead is me; case01: coach #4252 waited for PLAYER0 standing
            // 1 m behind its tail)
            let player_foes: Vec<usize> = foes.iter().map(|f| f.movement).collect();
            let meets = |k: usize| claim_of[k].is_some_and(|ck| claims[ck].junction == j && !claims[ck].chain.iter().any(|&(cm, _)| cm == m) && claims[ck].chain.iter().any(|&(cm, _)| player_foes.contains(&cm)));
            let lead = (p.still > PLAYER_STILL && p.speed.abs() < STANDING_SPEED)
                .then(|| {
                    let fwd = p.fwd();
                    let right = DVec2::new(fwd.y, -fwd.x);
                    let o = p.pos.truncate();
                    // (at any speed of its own: the coach creeping off in front of the standing
                    // player was refused for it again at 0.6-1.8 m/s, stop and go)
                    self.vehicles_near(p.pos, (p.half_len + PLAYER_LEAD_REACH) as f64 + 20.0)
                        .into_iter()
                        .filter_map(|k| {
                            let b = vehicles[k].obb();
                            let along = b.corners().map(|c| (c - o).dot(fwd) - p.half_len as f64);
                            let lo = along.iter().copied().fold(f64::INFINITY, f64::min);
                            // (one moving off from in front of it keeps it as its lead for
                            // `PLAYER_LEAD_AWAY`: refused again 3 m on, it stood and went)
                            let hk = (vehicles[k].heading as f64).to_radians();
                            let away = vehicles[k].state.speed >= STANDING_SPEED && DVec2::new(hk.sin(), hk.cos()).dot(fwd) > 0.0;
                            let reach = if away { PLAYER_LEAD_AWAY } else { PLAYER_LEAD_REACH };
                            // (wholly ahead of its front, but for half a metre: not a car alongside
                            // in a narrow piece beside it with its front just past the player's)
                            (lo > -PLAYER_LEAD_OVERLAP as f64 && lo < reach as f64 && (b.center - o).dot(right).abs() < PLAYER_LEAD_ACROSS as f64).then_some((k, lo.max(0.0)))
                        })
                        // (of two side by side in its corridor, one whose way meets the player's
                        // at this junction and is not on the player's own piece (one ahead on its
                        // way comes first anyway, `reservation::behind_on_my_way`), else the
                        // nearest: case01, the player stood 2 m behind #5348 on its own piece and
                        // 1-2 m behind the tail of coach #4252 beside it, which waited for the
                        // player at their zipper)
                        .min_by(|a, b| (!meets(a.0), a.1).partial_cmp(&(!meets(b.0), b.1)).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(k, _)| k)
                })
                .flatten();
            claims.push(Claim {
                chain,
                was_standing: false,
                light_red,
                until_go,
                lead,
                vehicle: usize::MAX - k,
                junction: j,
                movement: m,
                s: s_on,
                speed: v,
                accel: 1.2,
                v_max: 14.0,
                front: p.half_len,
                rear: p.half_len,
                gap: 4.0,
                committed,
                exit_room: true,
                impatience: Impatience::default(),
                absolute: p.priority,
                held: p.still > PLAYER_STILL,
                blocked,
                stuck: false,
                clearing: false,
                lanes: way,
                light,
                id: PLAYER_ID_BASE + p.id as u64,
                yields_now: None,
                clear_ahead: f32::INFINITY,
                standing_long: false,
                brake: plan::HARD_BRAKE,
            });
        }
        for (i, standing) in standing_now {
            vehicles[i].claim_standing = standing;
        }
        for (i, passed) in passed_now {
            vehicles[i].passed_on_go = passed;
        }
        for i in stale_amber {
            vehicles[i].amber = None;
            vehicles[i].amber_line = None;
        }
        // the signal groups the claimants' lights belong to: which pairs never let vehicles
        // in together (worked out once per pair of lights, from the programs)
        for a in 0..claims.len() {
            let Some((ctl, la)) = claims[a].light else { continue };
            for b in a + 1..claims.len() {
                let Some((ctl_b, lb)) = claims[b].light else { continue };
                if ctl_b != ctl || la == lb || claims[b].junction != claims[a].junction {
                    continue;
                }
                let key = (ctl, la.min(lb), la.max(lb));
                if !self.signal_groups.exclusive.contains_key(&key) {
                    let ex = self.lights.get(ctl).is_some_and(|c| c.never_together(la, lb));
                    self.signal_groups.exclusive.insert(key, ex);
                }
            }
        }
        // who went first last at each zipper merge, from the vehicles (the latest per merge)
        let mut last_won: HashMap<(usize, usize, usize), (f64, usize, u64)> = HashMap::new();
        for v in vehicles.iter() {
            if let Some((key, piece, t)) = v.zip_won {
                if last_won.get(&key).map(|w| t > w.0 || (t == w.0 && v.id < w.2)).unwrap_or(true) {
                    last_won.insert(key, (t, piece, v.id));
                }
            }
        }
        self.signal_groups.zipper = last_won.into_iter().map(|(k, (_, piece, id))| (k, (piece, id))).collect();
        self.eased_claims(vehicles, &mut claims, &claim_of, dt);
        let grants: Vec<Grant> = reservation::resolve_with(&self.junctions, &claims, &self.signal_groups);
        self.claims = claims.iter().cloned().zip(grants.iter().copied()).collect();
        // who went first at a zipper merge: it keeps going first while the two are engaged,
        // and the next tie there is the other side's ([round6 J3] written at the first
        // decision, not only on go: refused for a third, the winner stays first, and the pair's
        // order does not flip as their distances change)
        for (g, c) in grants.iter().zip(&claims) {
            if let Some((key, piece)) = g.zip {
                // (a player's win is not remembered: no keyframe would carry it)
                if let Some(v) = vehicles.get_mut(c.vehicle) {
                    v.zip_won = Some((key, piece, self.time));
                    self.signal_groups.zipper.insert(key, (piece, c.id));
                }
            }
        }

        // plan and act, one vehicle at a time
        for i in 0..vehicles.len() {
            let grant = claim_of[i].map(|k| grants[k]);
            let plan = plan::plan(net, self, vehicles, i, &perceptions[i], grant);
            self.act(net, vehicles, i, plan, dt);
        }
        self.negotiate(vehicles, &claim_of, dt);
        self.removed.sort_unstable_by(|a, b| b.cmp(a));
    }

    /// The claims of the cars eased off in a mutual hold (`Vehicle::ease_for`): marked
    /// `yields_now` for the other while it has not left the meeting places of the two; the
    /// ease ends when it has, when either is gone or no longer claims the junction, when either
    /// is held by a red light, or after `EASE_MAX`.
    fn eased_claims(&self, vehicles: &mut [Vehicle], claims: &mut [Claim], claim_of: &[Option<usize>], dt: f32) {
        for i in 0..vehicles.len() {
            let Some((oid, j, since)) = vehicles[i].ease_for else { continue };
            let keep = (|| {
                let k = claim_of[i]?;
                let o = vehicles.iter().position(|v| v.id == oid)?;
                let ko = claim_of[o]?;
                let (mine, theirs) = (&claims[k], &claims[ko]);
                if mine.junction != j || theirs.junction != j || mine.light_red || theirs.light_red || since > EASE_MAX {
                    return None;
                }
                // (the other still short of, or in, the first meeting place of the two beyond
                // where I stand, its rear not yet `CLEAR_MARGIN` s of its speed past that
                // stretch's end. [round6 J3] Not every meeting place of the chain: two merging
                // run on along one way, and the ease held the one behind until the other had
                // left the whole object, review-r6 case03/06)
                let pieces = self.junctions.foes_along_pieces(j, &mine.chain);
                let my_front = mine.s + mine.front;
                let first = pieces.iter().filter(|(_, f)| f.mine.1 > my_front).find_map(|(_, f)| theirs.chain.iter().find(|&&(m, _)| m == f.movement).map(|&(_, off)| (f, off)));
                let inside = first.is_some_and(|(f, off)| theirs.s - off - theirs.rear <= f.theirs.1 + reservation::CLEAR_MARGIN * theirs.speed.max(0.0));
                inside.then_some((k, o))
            })();
            match keep {
                Some((k, o)) => {
                    claims[k].yields_now = Some(o);
                    if let Some(e) = vehicles[i].ease_for.as_mut() {
                        e.2 += dt;
                    }
                }
                None => vehicles[i].ease_for = None,
            }
        }
    }

    /// Two vehicles that hold each other (each the other's `wait_for` or `body_block`), both
    /// standing, at a junction where their ways meet, neither at a red light: one eases off
    /// (`Vehicle::ease_for`), the other goes on by the ordinary rules - bodies, other foes and
    /// all (round 4a's standoff break let the winner ignore the other's body and claim, and it
    /// squeezed past at -0.01 m, review9). When: at the first frame both are refused for each
    /// other by the junction (`wait_for` both ways; the resolver's answer will not change by
    /// itself, and it took 6 s before: `MUTUAL_HOLD`), else, one of them held by the other's
    /// body only, after `MUTUAL_HOLD` times the lower id's `mutual_jitter`; a hold not seen
    /// for less than `MUTUAL_GAP` keeps its clock. The one that eases off: of two where only
    /// one is body-blocked by the other, that one (it cannot go anyway; the other waits for it
    /// at the junction); else by `eases_off`, the same in every run.
    pub fn negotiate(&mut self, vehicles: &mut [Vehicle], claim_of: &[Option<usize>], dt: f32) {
        let by_id: HashMap<u64, usize> = vehicles.iter().enumerate().map(|(i, v)| (v.id, i)).collect();
        let holds = |v: &Vehicle, id: u64| v.last.wait_for == Some(id) || v.last.body_block == Some(id);
        let mut timers: Vec<(Option<(u64, f32)>, f32)> = Vec::with_capacity(vehicles.len());
        for v in vehicles.iter() {
            // (both standing: one rolling up to its stop is not held yet)
            let other = [v.last.wait_for, v.last.body_block].into_iter().flatten().find(|&id| by_id.get(&id).is_some_and(|&o| holds(&vehicles[o], v.id) && vehicles[o].state.speed < STANDING_SPEED)).filter(|_| v.state.speed < STANDING_SPEED);
            timers.push(match (other, v.mutual) {
                (Some(o), Some((p, t))) if p == o => (Some((o, t + dt)), 0.0),
                (Some(o), _) => (Some((o, dt)), 0.0),
                // (not seen this frame: kept over a short gap while it stands)
                (None, Some((p, t))) if v.mutual_gap + dt < MUTUAL_GAP && v.state.speed < STANDING_SPEED => (Some((p, t + dt)), v.mutual_gap + dt),
                (None, _) => (None, 0.0),
            });
        }
        for (v, (t, gap)) in vehicles.iter_mut().zip(&timers) {
            v.mutual = *t;
            v.mutual_gap = *gap;
        }
        for a in 0..vehicles.len() {
            let Some((oid, t)) = vehicles[a].mutual else { continue };
            let Some(&b) = by_id.get(&oid) else { continue };
            if vehicles[a].id > oid || vehicles[a].ease_for.is_some() || vehicles[b].ease_for.is_some() {
                continue;
            }
            if vehicles[a].state.speed >= STANDING_SPEED || vehicles[b].state.speed >= STANDING_SPEED {
                continue;
            }
            // (only at a frame the hold is seen: who is body-blocked is read from this frame)
            if vehicles[a].mutual_gap > 0.0 || vehicles[b].mutual_gap > 0.0 {
                continue;
            }
            // the first mutual refusal, or the body-held pair's clock
            let refused = vehicles[a].last.wait_for == Some(oid) && vehicles[b].last.wait_for == Some(vehicles[a].id);
            if !refused && t < MUTUAL_HOLD * mutual_jitter(vehicles[a].id) {
                continue;
            }
            let (Some(ka), Some(kb)) = (claim_of[a], claim_of[b]) else { continue };
            let (ca, cb) = (&self.claims[ka].0, &self.claims[kb].0);
            if ca.junction != cb.junction || ca.light_red || cb.light_red || ca.vehicle != a || cb.vehicle != b {
                continue;
            }
            let j = ca.junction;
            let meet = self.junctions.foes_along_pieces(j, &ca.chain).iter().any(|(_, f)| cb.chain.iter().any(|&(m, _)| m == f.movement));
            if !meet {
                continue;
            }
            // (behind the other: held by its body, or following it as its lead - that one
            // cannot go anyway; the hash eased the one in front of a nose-to-tail pair, and it
            // stood `EASE_MAX` for the one behind it, viz case14 #46608/#50444)
            let (ida, idb) = (vehicles[a].id, vehicles[b].id);
            let blocked_a = vehicles[a].last.body_block == Some(idb) || vehicles[a].lead_id == Some(idb);
            let blocked_b = vehicles[b].last.body_block == Some(ida) || vehicles[b].lead_id == Some(ida);
            let ease = if blocked_a != blocked_b {
                if blocked_a {
                    a
                } else {
                    b
                }
            } else if eases_off(ida, idb, j) == ida {
                a
            } else {
                b
            };
            let other = if ease == a { idb } else { ida };
            vehicles[ease].ease_for = Some((other, j, 0.0));
            vehicles[a].mutual = None;
            vehicles[b].mutual = None;
            vehicles[a].mutual_gap = 0.0;
            vehicles[b].mutual_gap = 0.0;
        }
    }

    fn act(&mut self, net: &Network, vehicles: &mut [Vehicle], i: usize, plan: Plan, dt: f32) {
        let now = self.time;
        let sidestep = self.sidestep(vehicles, i, dt);
        let creep = self.settle_creep(vehicles, i, dt);
        // (a pass given up whose obstacle has left: over, the car drives on where it is and the
        // offset eases off as it goes, re-entering its lane ahead; review12, case15 #47238 stood
        // 88 s out on the oncoming side, its stop behind it, after the obstacle had gone)
        let obstacle_gone = vehicles[i].passing.filter(|p| p.aborted).and_then(|p| p.obstacle).is_some_and(|id| {
            if id >= PLAYER_ID_BASE {
                self.players.iter().find(|p| p.id as u64 == id - PLAYER_ID_BASE).is_none_or(|p| p.speed.abs() > 1.0)
            } else {
                vehicles.iter().find(|o| o.id == id).is_none_or(|o| o.state.speed > 1.0)
            }
        });
        let return_blocked = self.return_blocked(vehicles, i);
        let settle = if vehicles[i].state.change.is_some() && vehicles[i].change_still.1 + dt > FROZEN_CHANGE { self.settle_choice(net, vehicles, i) } else { None };
        let wait_for = plan.wait_for.map(|k| if k >= usize::MAX - 64 { PLAYER_ID_BASE + self.players.get(usize::MAX - k).map(|p| p.id as u64).unwrap_or(0) } else { vehicles.get(k).map(|o| o.id).unwrap_or(0) });
        let lead_id = match plan.lead_seen {
            Some(Seen::Vehicle(j)) | Some(Seen::Body(j)) => vehicles.get(j).map(|o| o.id),
            _ => None,
        };
        let v = &mut vehicles[i];
        v.last = Decision { why: plan.why, lead: plan.lead, lead_seen: plan.lead_seen, lead_id, stop: plan.stop, held: plan.held, hold: plan.hold, wait_for, body_block: plan.body_block };
        v.lead_id = lead_id;
        if plan.why == Why::Light && v.state.speed < STANDING_SPEED {
            self.held_at_red += 1;
        }
        if let Some(light) = plan.amber {
            v.amber = Some(light);
            v.amber_line = plan.amber_line.map(|gap| v.state.odometer + gap);
        }
        if let Some(lat) = plan.lateral {
            v.state.lateral_target = lat;
        }
        // a lane change begins only when nobody started one into the same stretch this
        // frame (the perception was taken before anybody moved)
        let mut change_into = |to: usize, s_to: f32| -> bool {
            if self.changes_started.iter().any(|&(l, s)| l == to && (s - s_to).abs() < 25.0) {
                return false;
            }
            self.changes_started.push((to, s_to));
            true
        };
        match plan.start {
            Some(Start::Change { to, dir }) => {
                if change_into(to, net.beside_s(v.state.lane, to, v.state.s)) {
                    v.state.start_change(net, to, dir);
                }
            }
            Some(Start::Bypass { to, dir, out_len }) => {
                if change_into(to, net.beside_s(v.state.lane, to, v.state.s)) {
                    v.state.start_bypass(net, to, dir, out_len);
                    v.stopped = 0.0;
                }
            }
            Some(Start::RouteChange { to, dir }) => {
                if change_into(to, net.beside_s(v.state.lane, to, v.state.s)) {
                    v.state.start_route_change(net, to, dir);
                }
            }
            Some(Start::Pass(p)) => {
                v.passing = Some(p);
                // (it pulls out once it has signalled, below)
                if p.wait <= 0.0 {
                    v.state.lateral_curve = p.curve(v.state.odometer, v.state.lateral);
                }
                v.state.lateral_target = 0.0;
                v.stopped = 0.0;
            }
            Some(Start::AbortPass { hold }) => {
                if let Some(p) = v.passing.as_mut() {
                    p.aborted = true;
                    p.hold = hold;
                }
                // back in over what room there is, at least two metres (standing: over a bay's
                // S-curve as it drives on, else the way bent over two metres turned the body
                // 20-30 degrees where it stood, review12)
                let odo = v.state.odometer;
                let lat = v.state.lateral;
                if v.state.speed < 0.5 {
                    v.state.lateral_curve.clear();
                    v.state.lateral_ramp = (lat, 0.0, odo, (lat.abs() * 8.0).clamp(8.0, 30.0));
                } else {
                    let back = (hold - odo - 0.5).clamp(2.0, 8.0);
                    v.state.lateral_curve = vec![(odo, lat), (odo + back, 0.0)];
                }
                v.state.lateral_target = 0.0;
            }
            None => {}
        }
        // a pass pulls out once it has signalled for `SIGNAL_BEFORE_CHANGE` (counted while its
        // indicator shows that side: from the other side's it takes a moment to change)
        let shown = v.blinker_shown.0;
        if let Some(p) = v.passing.as_mut() {
            if !p.aborted && p.wait > 0.0 && shown == p.blinker() {
                p.wait = (p.wait - dt).max(0.0);
                if p.wait <= 0.0 {
                    let pc = *p;
                    v.state.lateral_curve = pc.curve(v.state.odometer, v.state.lateral);
                    v.state.lateral_target = 0.0;
                }
            }
        }
        // coming back in: not into a body beside it - it stays out a metre more (a pass recorded
        // under way at a fixture's start came back in on top of the car that had drawn level,
        // viz3 case11 #28490, -1.77 m)
        // (straight on past its front: the body follows the way under its front, and bent
        // nearer than that it turned towards the body beside it, review12)
        if let (true, Some(p)) = (return_blocked, v.passing.as_mut()) {
            let (odo, lat) = (v.state.odometer, v.state.lateral);
            let straight = return_look(&v.state);
            p.until = p.until.max(odo + 1.0);
            let back = (p.back_len * (lat.abs() / p.side.abs().max(0.1))).max(2.0);
            v.state.lateral_curve = vec![(odo, lat), (odo + straight, lat), (odo + straight + back, 0.0)];
        }
        v.state.accel_cap = plan.accel_cap;
        // a fresh car: no harder braking than its driver likes at the start
        if v.fresh > 0.0 {
            let mut n = 0;
            while n < 16 && v.state.desired_accel(net, plan.lead, plan.stop) < -v.state.decel {
                v.state.speed *= 0.8;
                n += 1;
            }
        }
        let alive = v.state.drive(net, dt, plan.lead, plan.stop);
        // a change of the driver's own whose target lane (moving on along its plan) is closed
        // to it: given up while the body is less than half over (viz5 case04 #5644 went on from
        // 3470's rule-less bus lane piece into 3474's)
        let closed = v.state.change.is_some_and(|c| v.state.route.is_empty() && net.lanes.get(c.to).is_some_and(|l| l.source != 2 && !crate::access::is_open(l, v)));
        if closed && v.state.change_offsets(net).is_some_and(|(k, _, _)| k < 0.5) {
            v.state.settle_change(net, false);
            v.change_still = (0.0, 0.0);
        }
        // a lane change that stands frozen part-way is settled where the body is
        match v.state.change {
            Some(c) => {
                if v.state.speed >= STANDING_SPEED || (c.t - v.change_still.0).abs() > 0.01 {
                    v.change_still = (c.t, 0.0);
                } else {
                    v.change_still.1 += dt;
                    if let (true, Some(done)) = (v.change_still.1 > FROZEN_CHANGE, settle) {
                        v.state.settle_change(net, done);
                        v.change_still = (0.0, 0.0);
                    }
                }
            }
            None => v.change_still = (0.0, 0.0),
        }
        // timers
        v.fresh = (v.fresh - dt).max(0.0);
        v.pull_out = (v.pull_out - dt).max(0.0);
        let standing = v.state.speed < STANDING_SPEED;
        // [ROLLBACK r4d-B4] a bus creeping up under `CREPT_OFF_SPEED` has not moved off: its
        // standing time runs on from there (reset by a 0.2-0.4 m/s creep, the bus held short of
        // its stop by people served the stop where it stood only after 12.7 s instead of 6)
        let moving = if v.serves_stops() { v.state.speed >= CREPT_OFF_SPEED } else { !standing };
        if standing && !v.external.at_stop {
            v.stopped += dt;
        } else if moving {
            v.stopped = 0.0;
        }
        if (v.state.odometer - v.progress.0).abs() > 5.0 {
            v.progress = (v.state.odometer, 0.0);
        } else if !v.external.at_stop {
            v.progress.1 += dt;
        }
        let held = standing && plan.held;
        // A driver who is let go but has hardly rolled yet (the start reaction takes a moment,
        // then the car creeps off) keeps the impatience that earned the go. It used to cool
        // the moment the hold lifted for a frame, which took the squeeze away again before
        // the car was committed, so four cars at an equal junction held each other for
        // ever, each in turn fed up for one frame (and then creeping and stopping again).
        if !(v.state.speed < 2.0 && !plan.held) {
            v.impatience.update(&v.params, dt, held);
        }
        if v.stopped > 2.0 {
            v.amber = None;
        }
        // a pass runs its course: the bent way brings the car back by itself
        if obstacle_gone {
            // (a pass on its way back in from where it is: the bent way eases the offset off
            // as it drives, held out while a body beside it is in the way, `return_blocked`)
            let (odo, lat) = (v.state.odometer, v.state.lateral);
            let back = (lat.abs() * 8.0).clamp(8.0, 30.0);
            if let Some(p) = v.passing.as_mut() {
                *p = Passing { side: lat, until: odo, block: odo, aborted: false, hold: odo, back_len: back, wait: 0.0, obstacle: None, ..*p };
            }
            // (straight on past its front first: bent under it where it stands, the body
            // turned on the spot)
            let straight = return_look(&v.state);
            v.state.lateral_curve = vec![(odo, lat), (odo + straight, lat), (odo + straight + back, 0.0)];
            v.state.lateral_target = 0.0;
        }
        // a settled change standing with its offset: eased in place towards the lane's middle
        // while its body there touches nobody (it eased only as the car drove on)
        if !offset_left(v) {
            v.state.settled_offset = false;
        }
        if settling(v) {
            if let Some(lat) = creep.filter(|_| v.state.speed < 0.1) {
                v.state.lateral_curve.clear();
                v.state.lateral = lat;
                v.state.lateral_ramp = (lat, 0.0, v.state.odometer, (lat.abs() * 8.0).clamp(8.0, 30.0));
            }
        }
        if let Some(p) = v.passing {
            if p.aborted {
                // standing out there, it steps back into its lane where it stands, at a walking
                // pace, while its body there touches nobody (backing up a little and pulling
                // in: the way bends only as the car drives); once back the pass is over
                if v.state.speed < 0.1 {
                    if let Some(lat) = sidestep {
                        v.state.lateral_curve.clear();
                        v.state.lateral = lat;
                        v.state.lateral_target = 0.0;
                        v.state.lateral_ramp = (lat, 0.0, v.state.odometer, (lat.abs() * 8.0).clamp(8.0, 30.0));
                    }
                }
                if v.state.speed < 0.1 && v.state.lateral.abs() <= SIDESTEP_DONE && (v.state.odometer >= p.hold - 0.3 || v.stopped > 1.0) {
                    v.passing = None;
                    // (from where it is: the rest of the way back eases off as it drives on)
                    let lat = v.state.lateral;
                    v.state.lateral_curve.clear();
                    v.state.lateral_target = 0.0;
                    v.state.lateral_ramp = (lat, 0.0, v.state.odometer, 8.0);
                }
            } else if v.state.odometer >= p.until + p.back_len {
                v.passing = None;
            } else if v.stopped > 8.0 && v.state.lateral.abs() < p.side.abs() * 0.5 {
                // never got out: give it up where it stands (the offset easing off as it drives)
                v.passing = None;
                let (odo, lat) = (v.state.odometer, v.state.lateral);
                v.state.lateral_curve.clear();
                v.state.lateral_target = 0.0;
                v.state.lateral_ramp = (lat, 0.0, odo, (lat.abs() * 8.0).clamp(8.0, 30.0));
            }
        }
        // a random car that stands held by nothing for minutes asks to be taken away when
        // nobody looks
        v.give_up_clock(dt);
        // indicators: what the car does, debounced so a flicker of the plan does not flash
        // (a pass shows the side it moves to from before it pulls out until it is back in its
        // lane: the bent way's offset alone showed the side it would come back to, the right
        // while out on the left, viz3 review: 47 of 47)
        if let Some(p) = v.passing {
            if !p.aborted {
                v.state.signal = p.blinker();
                v.state.signal_time = 0.25;
            } else if v.state.lateral.abs() > 0.3 {
                // (given up: the side back into its lane, while it is out)
                v.state.signal = if v.state.lateral > 0.0 { 1 } else { 2 };
                v.state.signal_time = 0.25;
            }
        }
        v.state.update_blinker(net);
        let want = if v.external.blinker_off { 0 } else { v.state.blinker };
        let (shown, since) = v.blinker_shown;
        let age = (now - since) as f32;
        let allowed = if want == shown {
            true
        } else if want == 0 {
            age >= 0.5
        } else if shown == 0 {
            true
        } else {
            age >= 1.5
        };
        if allowed && want != shown {
            v.blinker_shown = (want, now);
        }
        if !alive {
            self.removed.push(i);
        }
    }

    /// How a frozen lane change of vehicle `i` is settled (`AiState::settle_change`): done in
    /// place with the body more than half over the lane it moves to, else given up - on the lane
    /// where its body, at that lane's middle, would come within 0.2 m of nobody; the other if
    /// that one is taken (the change froze because a car stood beside it in the lane it went
    /// for: done in place, the two stood side by side in one lane and touched as the offset
    /// eased off, viz case06); neither: it stays as it is for now. True: done.
    fn settle_choice(&self, net: &Network, vehicles: &[Vehicle], i: usize) -> Option<bool> {
        let v = &vehicles[i];
        let (k, _, _) = v.state.change_offsets(net)?;
        let (own, target) = v.state.change_anchors(net)?;
        let near = self.vehicles_near(v.pos, 25.0);
        // (its body at that lane's middle, along it)
        let free = |(p, h): (DVec3, f32)| {
            let b = Obb::vehicle(p.truncate(), h as f64, v.state.front as f64, v.state.rear as f64, v.half_width as f64);
            near.iter().filter(|&&j| j != i).all(|&j| (vehicles[j].pos.z - v.pos.z).abs() > 3.0 || vehicles[j].obb().separation(&b) > 0.2)
                && self.players.iter().all(|p| (p.pos.z - v.pos.z).abs() > 3.0 || p.obb(0.0, 0.0).separation(&b) > 0.2)
        };
        let first = k > 0.5;
        let place = |done: bool| if done { target } else { own };
        if free(place(first)) {
            Some(first)
        } else if free(place(!first)) {
            Some(!first)
        } else {
            None
        }
    }

    /// A pass coming back into its lane whose bent way would bring its body, `return_look` on, within 0.2 m
    /// of another vehicle (and nearer than it is now): one beside it in the lane.
    fn return_blocked(&self, vehicles: &[Vehicle], i: usize) -> bool {
        let v = &vehicles[i];
        let Some(p) = v.passing.filter(|p| !p.aborted && p.wait <= 0.0) else { return false };
        let (odo, lat) = (v.state.odometer, v.state.lateral);
        // (on its way back in, or about to be: where the way puts it a little past its front)
        let look = return_look(&v.state);
        let next = v.state.lateral_ahead(look);
        if odo + look < p.until || (next.abs() >= lat.abs() - 0.01) {
            return false;
        }
        let h = (v.heading as f64).to_radians();
        let right = DVec2::new(h.cos(), -h.sin());
        let now = v.obb();
        let b = Obb::vehicle(v.pos.truncate() + right * (next - lat) as f64, v.heading as f64, v.state.front as f64, v.state.rear as f64, v.half_width as f64);
        // (a body that the move brings nearer than 0.2 m, and nearer than it is now)
        self.vehicles_near(v.pos, 25.0).into_iter().filter(|&j| j != i).any(|j| {
            let o = vehicles[j].obb();
            (vehicles[j].pos.z - v.pos.z).abs() <= 3.0 && o.separation(&b) < 0.2 && o.separation(&b) < o.separation(&now)
        })
    }

    /// An aborted pass standing out there: the lateral offset one step (`SIDESTEP_SPEED` for
    /// `dt`) back towards its lane, if its body there would come within 0.2 m of nobody.
    fn sidestep(&self, vehicles: &[Vehicle], i: usize, dt: f32) -> Option<f32> {
        let v = &vehicles[i];
        if !v.passing.is_some_and(|p| p.aborted) || v.state.speed >= 0.1 || v.state.lateral.abs() <= SIDESTEP_DONE {
            return None;
        }
        let lat = v.state.lateral;
        let new = lat - (SIDESTEP_SPEED * dt).min(lat.abs()) * lat.signum();
        let h = (v.heading as f64).to_radians();
        let right = DVec2::new(h.cos(), -h.sin());
        let b = Obb::vehicle(v.pos.truncate() - right * (lat - new) as f64, v.heading as f64, v.state.front as f64, v.state.rear as f64, v.half_width as f64);
        let clear_of_cars = self.vehicles_near(v.pos, 25.0).into_iter().filter(|&j| j != i).all(|j| (vehicles[j].pos.z - v.pos.z).abs() > 3.0 || vehicles[j].obb().separation(&b) > 0.2);
        let clear_of_players = self.players.iter().all(|p| (p.pos.z - v.pos.z).abs() > 3.0 || p.obb(0.0, 0.0).separation(&b) > 0.2);
        (clear_of_cars && clear_of_players).then_some(new)
    }

    /// A settled change standing with its offset (`settling`): the offset one step
    /// (`SETTLE_CREEP` for `dt`) towards its lane's middle, if its body there comes within
    /// 0.2 m of nobody - vehicle, player or person on foot (`PERSON_RADIUS`) - it does not
    /// move away from (the one it settled beside, viz case06, must not be touched; one it
    /// leaves behind on the other side does not hold it; review r6l: a person standing
    /// 0.4 m off its inner side was eased 0.88 m into).
    fn settle_creep(&self, vehicles: &[Vehicle], i: usize, dt: f32) -> Option<f32> {
        let v = &vehicles[i];
        if !settling(v) || v.state.speed >= 0.1 {
            return None;
        }
        let lat = v.state.lateral;
        let new = lat - (SETTLE_CREEP * dt).min(lat.abs()) * lat.signum();
        let h = (v.heading as f64).to_radians();
        let right = DVec2::new(h.cos(), -h.sin());
        let now = v.obb();
        let b = Obb::vehicle(v.pos.truncate() - right * (lat - new) as f64, v.heading as f64, v.state.front as f64, v.state.rear as f64, v.half_width as f64);
        let ok = |o: &Obb| o.separation(&b) > 0.2 || o.separation(&b) > o.separation(&now);
        let clear_of_cars = self.vehicles_near(v.pos, 25.0).into_iter().filter(|&j| j != i).all(|j| (vehicles[j].pos.z - v.pos.z).abs() > 3.0 || ok(&vehicles[j].obb()));
        let clear_of_players = self.players.iter().all(|p| (p.pos.z - v.pos.z).abs() > 3.0 || ok(&p.obb(0.0, 0.0)));
        let clear_of_people = self.people_near(v.pos, 15.0).into_iter().all(|k| {
            let q = &self.people[k];
            (q.pos.z - v.pos.z).abs() > 2.5 || ok(&Obb::point(q.pos, crate::plan::PERSON_RADIUS as f64))
        });
        (clear_of_cars && clear_of_players && clear_of_people).then_some(new)
    }

    /// The indicator a vehicle shows now (0 off, 1 left, 2 right).
    pub fn blinker(v: &Vehicle) -> i32 {
        v.blinker_shown.0
    }

    /// Light `li` of controller `ctl`: (state code, seconds the state lasts).
    pub fn light_state(&self, ctl: usize, li: usize) -> Option<(i32, f32)> {
        let c = self.lights.get(ctl)?;
        Some((c.state(li), c.remaining(li)))
    }

    pub fn light_until_go(&self, ctl: usize, li: usize) -> Option<f32> {
        self.lights.get(ctl)?.time_until_go(li)
    }
}
