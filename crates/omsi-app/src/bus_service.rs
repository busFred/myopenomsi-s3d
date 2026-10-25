//! Timetable buses as traffic. A timetable bus is one of the town's AI cars (`AiCar`,
//! created by the same `Traffic::create_car` as every other car, driven by the same
//! following, lights, junctions, lane changes and passing) that carries a `BusService`:
//! the stops of its trip, the pull into the bay, the doors at each stop, the wait for the
//! departure time, the blinker and the pull back out, the end of the trip, and the people
//! aboard. Nothing of the player's bus is involved: no driver inputs, no throttle and brake
//! for the script to turn into motion - the car moves, the script only animates it.

use omsi_sim::traffic::{AiState, LaneKind, Network};
use omsi_sim::VehicleInstance;
use std::collections::VecDeque;

/// One stop of the trip, on the car's route.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stop {
    /// Index into the car's route of the lane the stop is on.
    pub ri: usize,
    /// Where the front of the bus comes to rest, along that lane (m).
    pub s: f32,
    /// How far right of the lane's middle the stop's bay lies (m).
    pub bay: f32,
    /// Timetable departure (seconds of the day).
    pub depart: f64,
    /// The stop's map object (its `[busstop]` strings weigh who gets off there).
    pub id: i64,
    /// The side the platform lies on (see `tiles::stop_side`): 0 = right, 1 = the other,
    /// 2 = both. A bus whose doors are on both sides opens only these (it reads the value
    /// as `AI_Scheduled_AtStation_Side`).
    pub side: f32,
}

impl Stop {
    #[allow(clippy::type_complexity)]
    pub fn from_tuple(t: (usize, f32, f32, f64, i64, f32)) -> Stop {
        Stop { ri: t.0, s: t.1, bay: t.2, depart: t.3, id: t.4, side: t.5 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// On the way to the next stop (or with none left, to the end of the route).
    Running,
    /// At the stop with the doors open: people get off and on.
    Boarding,
    /// At the stop with the doors shut, waiting for the departure time (a layover at the
    /// first stop, or a bus that is early).
    Waiting,
    /// The doors close and the stop brake comes off, the blinker goes on: about to pull out.
    Closing,
    /// At the end of the trip: it stands until the timetable hands it the tour's next trip
    /// or lets it go.
    TripDone,
}

#[derive(Debug, Clone)]
pub struct BusService {
    /// [round 7 A7] The offset it steers to at or into a stop (its bay, a place parked away):
    /// the planner's `External::bay`. None between stops, where the traffic's own rules
    /// (the kerb swerve, the settle creep) place it in its lane.
    pub bay_wish: Option<f32>,
    /// [P8 pass 1, B1] The stop it served last (its map object id): with the front stop, the
    /// stops whose waiters it pulls in beside (`External::serves`).
    pub served: Option<i64>,
    pub stops: VecDeque<Stop>,
    pub phase: Phase,
    /// Seconds in the current phase.
    pub phase_t: f32,
    /// Seconds of boarding left (the people at the doors hold it open: `hold`).
    pub boarding: f32,
    /// When it may leave the stop (seconds of the day).
    pub leave_at: f64,
    /// The doors have been open at this stop already (a layover bus opens them only for
    /// the last minute before its departure).
    pub boarded: bool,
    /// Seconds behind (positive) or ahead of the timetable, as of the last stop.
    pub delay: f64,
    /// Put out before its departure: it waits for it at its first stop.
    pub layover: bool,
    /// [ROLLBACK aiparked-73] The wait at this stop is a long one (`PARKED_WAIT`): the engine stops.
    long_wait: bool,
    /// This stop is a layover on a through lane: the doors are open for the whole wait, and
    /// the bus goes early when a queue builds up behind it.
    roadside_layover: bool,
    /// The timetable still carries the route on as tiles bring their lanes: at the end of
    /// what it has, it waits for more.
    pub route_open: bool,
    /// The terminus of its trip, the name the waiting people read off it (Omsi.exe's bus
    /// +0x7bc) to see whether it goes their way.
    pub terminus: String,
    /// How far the front stop was last frame (m; infinite when not measured yet).
    near_d: f32,
    /// Whether it serves the front stop, once that is settled (`SKIP_DECIDE`).
    serve: Option<bool>,
    /// The trip's last station: always served.
    pub last_stop: Option<i64>,
    /// Stops the timetable has it serve in any case, and those it serves when it would be
    /// more than `EARLY_STOP_SHORT` early (`schedule::TripTimes::kinds`).
    pub always: Vec<i64>,
    pub serve_early: Vec<i64>,
    /// [ROLLBACK waitstop-136] Stops where the timetable has it wait for the departure time
    /// (`[profile_otherstopping]` 3 and 4, `schedule::WAIT_KINDS`); at any other it leaves
    /// when the people are off and on, early or not.
    pub wait_stops: Vec<i64>,
    /// [ROLLBACK r4d-B2] Away from the stop it waits at (round the block, or to a place with
    /// room off the through lane), to be back for its departure.
    pub away: Option<Away>,
    /// A way off decided while the doors were open: taken once they are shut.
    pending: Option<Detour>,
    /// When it last looked for a way off (seconds of the day), and the stop it found none at
    /// (logged once).
    detour_looked: f64,
    detour_none: Option<i64>,
    /// The wait at the front stop is its layover (as `arrive` had it).
    this_layover: bool,
    /// Its route was changed this frame (for the recording).
    pub rerouted: bool,
    /// [ROLLBACK r4d-B3] Seconds ahead of its timetable since it left a stop early for the
    /// traffic it held up (the roadside layover's early start): it drives slower
    /// (`pace`, the factor on its usual speed) until that is made up, and waits out what is
    /// left at its next stop.
    pub ahead_by: f64,
    pub pace: f32,
    /// [ROLLBACK r4d-B3] When it is to be at its next stop to have made the lead up (seconds
    /// of the day): the lead is that less when it would be there at its usual speed, every
    /// frame - time stood at a light or in a queue counts it off as well as time driven slow.
    pace_target: Option<f64>,
    /// It leaves this stop early for the traffic behind it.
    left_early: bool,
    /// [ROLLBACK r4d-B4] Held short of its stop by somebody in its way: it creeps up to the
    /// pole once they are clear rather than serving the stop where it stands.
    short_by_people: bool,
    /// [ROLLBACK r4d-B5] How far past its stop point it pulls on for a bus behind it for the
    /// same stop (kept until it is at the stop: a follower lost from view for a frame made the
    /// bus 10 m past its point take the stop for missed), and, standing at the stop with the
    /// doors shut, the odometer reading it pulls on to (once a stop).
    pull: f32,
    pull_to: Option<f32>,
    pulled_here: bool,
    /// [round 6 B2] Its trip ended at the end of its route with no stop there (a track that
    /// ends on a lane with a way on): it drives on until the timetable hands it the tour's
    /// next trip or lets it go, rather than standing at the lane's end.
    track_end: bool,
}

/// [ROLLBACK r4d-B2] A bus away from its stop until its departure.
#[derive(Debug, Clone)]
pub struct Away {
    /// Parked somewhere with room: where, and what it comes back by.
    park: Option<ParkAway>,
}

#[derive(Debug, Clone)]
struct ParkAway {
    /// The place: route index, where its origin comes to rest, how far right of the lane.
    ri: usize,
    s: f32,
    lat: f32,
    /// The way back to the lane it left (that lane last), the route on from there and the
    /// stops (their route indices from the lane it left).
    back: Vec<usize>,
    rest: Vec<usize>,
    stops: Vec<Stop>,
    /// When it sets off back (seconds of the day).
    leave_at: f64,
    arrived: bool,
}

/// [ROLLBACK r4d-B2] A way off a stop for a bus that blocks a through lane: `route` starts
/// on the lane it stands on.
#[derive(Debug, Clone, PartialEq)]
pub enum Detour {
    /// Round the block and back onto that lane (`route` ends on it again).
    Loop { route: Vec<usize>, len: f32 },
    /// To a place with room on the last lane of `route` (its origin at `s`, `lat` right of
    /// the lane), and back from there by `back` (ending on the lane it left).
    Park { route: Vec<usize>, s: f32, lat: f32, len: f32, back: Vec<usize>, back_len: f32 },
}

/// The side the bus pulls out towards: left (1) from a bay on the right, else right (2).
fn out_signal(bay: f32) -> i32 {
    if bay < -0.1 {
        2
    } else {
        1
    }
}

/// Seconds the doors stay open at a stop without anyone holding them.
fn boarding_time(id: u64) -> f32 {
    7.0 + (id % 5) as f32
}

/// An early bus waits at its stop until this long before its departure, a train at its
/// station until `EARLY_LEAVE_RAIL` before (Omsi.exe 0x7d9bdc: it stands while it is more
/// than 20 s, a train 120 s, early). Capped at 40 s, the buses no longer waited for their
/// times at the stops where a timetable holds them all for a connection (#1012).
const EARLY_LEAVE: f64 = 20.0;
const EARLY_LEAVE_RAIL: f64 = 120.0;
/// A layover waits for the departure however long (a tour's bus in on its previous trip).
const LAYOVER_WAIT: f64 = 1800.0;

/// How long a bus arriving at `now` stands at a stop it is to leave at `depart`.
fn early_wait(depart: f64, now: f64, layover: bool, rail: bool) -> f64 {
    let lead = if layover {
        0.0
    } else if rail {
        EARLY_LEAVE_RAIL
    } else {
        EARLY_LEAVE
    };
    (depart - lead - now).clamp(0.0, LAYOVER_WAIT)
}
/// [ROLLBACK aiparked-73] A wait at a stop this long (3.5 minutes) stops the engine; it starts
/// again this long (s) before the departure.
// [ROLLBACK engine-95] was 210 s / 75 s: the engine stops for any wait over two minutes and starts a minute before
const PARKED_WAIT: f64 = 120.0;
const PARKED_START: f64 = 60.0;
/// [ROLLBACK terminus-130] Seconds after its trip is over that a bus stands before the
/// engine stops (the passengers are off by then).
const TRIP_DONE_PARK: f32 = 20.0;
/// [ROLLBACK stopstuck-130] Seconds of boarding past the departure after which the doors
/// close whoever is still said to be at them.
const BOARDING_MAX: f32 = 120.0;
/// On a layover, the doors open this long before the departure.
const LAYOVER_BOARDING: f64 = 45.0;
/// A layover bus on a through lane leaves early once a vehicle has stood behind it this
/// long (s), if its departure is more than `ROADSIDE_EARLY_MIN` away: the queue costs the
/// town more than a few minutes' lead on the timetable.
const ROADSIDE_QUEUE: f32 = 20.0;
const ROADSIDE_EARLY_MIN: f64 = 30.0;
/// [ROLLBACK r4d-B3] A bus ahead of its timetable drives no slower than this share of its
/// usual speed while it makes the lead up; a lead under a second is none.
pub const PACE_MIN: f32 = 0.6;
/// [ROLLBACK r4d-B3] With a vehicle closer than this behind it (m) a bus ahead of its
/// timetable drives at its usual speed.
pub const PACE_FOLLOWER: f32 = 15.0;

/// [ROLLBACK r4d-B3] The share of its usual speed at which a bus `ahead_by` seconds early,
/// `dist` metres before its next stop at `v_nom`, takes the lead up by the time it is there
/// (at least `PACE_MIN`).
pub fn pace_for(ahead_by: f64, dist: f32, v_nom: f32) -> f32 {
    if ahead_by < 1.0 || dist <= 0.0 {
        return 1.0;
    }
    let t = dist / v_nom.max(1.0);
    (t / (t + ahead_by as f32)).clamp(PACE_MIN, 1.0)
}
/// [ROLLBACK r4d-B2] A bus standing out a long wait at its stop (or queueing for an occupied
/// berth) with its body over a through lane leaves once a vehicle has stood behind it this
/// long (s), or more than `LAYOVER_QUEUE_CARS` stand queued behind it: round the shortest
/// loop back to the stop of at most `LAYOVER_LOOP_MAX` m that it drives in the time to its
/// departure less `DETOUR_MARGIN`, else to the nearest place with room within
/// `LAYOVER_PARK_MAX` m, standing there until it is time to drive back.
pub const LAYOVER_QUEUE: f32 = 20.0;
pub const LAYOVER_QUEUE_CARS: usize = 3;
pub const LAYOVER_LOOP_MAX: f32 = 1200.0;
/// [ROLLBACK r4d-B2] The least way round the block (m) from the end of the lane back to its
/// start that counts as one.
pub const LAYOVER_LOOP_MIN: f32 = 100.0;
pub const LAYOVER_PARK_MAX: f32 = 400.0;
const DETOUR_MARGIN: f64 = 60.0;
/// The pace a detour is timed at (m/s: town traffic with its lights and junctions), and the
/// least wait worth one (s).
const DETOUR_SPEED: f32 = 6.0;
const DETOUR_MIN_LEFT: f64 = 90.0;
/// Seconds between two looks for a way off at the same stop.
const DETOUR_RETRY: f64 = 30.0;
/// Room a parked bus needs beyond its length (m).
pub const PARK_ROOM: f32 = 4.0;
/// [ROLLBACK r4d-B2] A bus taken off the road as the last resort comes back this long (s)
/// before its departure.
pub const RESPAWN_LEAD: f64 = 60.0;
/// [ROLLBACK r4d-B2] Placing a bus taken off the road back on it is tried again this often (s)
/// until `RESPAWN_GIVE_UP` after its departure, when the trip is given up; and the tour's bus
/// is not taken off the road again within `RESPAWN_COOLDOWN` of the last time (s).
pub const RESPAWN_RETRY: f64 = 3.0;
pub const RESPAWN_GIVE_UP: f64 = 120.0;
pub const RESPAWN_COOLDOWN: f64 = 300.0;

/// [ROLLBACK r4d-B2] A timetable bus taken off the road as the last resort: its trip comes back
/// at stop `stop` at time `at`, for its departure `depart` from there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Respawned {
    pub car: u64,
    /// When it was taken off (seconds of the day).
    pub removed: f64,
    pub at: f64,
    pub depart: f64,
    pub stop: i64,
}

/// [ROLLBACK r4d-B2] Seconds a detour of `len` metres takes.
fn detour_time(len: f32) -> f64 {
    (len / DETOUR_SPEED) as f64
}

/// [ROLLBACK r4d-B2] A street lane that is a through lane (`ThroughLanes::through`): other
/// vehicles drive it.
pub fn covers_through_lane(net: &Network, lane: usize, lateral: f32, half_width: f32, through: bool) -> bool {
    through && net.lanes.get(lane).is_some_and(|l| l.kind == LaneKind::Street) && covers_lane(net, lane, lateral, half_width)
}

/// [ROLLBACK r4d-B2] A bus `lateral` metres right of lane `lane` covers it: its body
/// (`half_width`) is not moved off the lane into a bay (a bus half on the pavement still
/// holds up a car).
pub fn covers_lane(net: &Network, lane: usize, lateral: f32, half_width: f32) -> bool {
    net.lanes.get(lane).is_some_and(|l| lateral.abs() < half_width + l.width * 0.5 - 0.5)
}

/// [round 6 B1] A bus standing across the fan of lanes of a junction object holds up the
/// traffic of the lanes beside its own (review-r6-cases case11: the layover bus on junction
/// piece 12074 stood over the stretches of j30 while B2 looked at its own lane only). A lane
/// of its object is covered when a car driving on its axis would come within this much (m)
/// of the bus's body: `covers_lane`'s 0.5 m margin ...
pub const FAN_COVER: f32 = 0.5;
/// [round 6 B1] ... with the car this wide either side of the axis (m: half a typical town
/// car's 1.8 m, the box every parked car had until round 7). A lane 2.7 m or more off the axis of a 2.5 m bus is
/// then not covered (a bus in a bay beside the through lane stays).
pub const FAN_CAR_HALF: f32 = 0.9;
/// [round 6 B1] A vehicle standing on a covered lane, or whose way runs into one within
/// this far of its front (m), and held by the bus (it waits for it, follows it or is held
/// by its body), queues behind the bus (the gap list's 40 m horizon of a junction approach,
/// review-r6-cases section 1 case11).
pub const FAN_QUEUE_REACH: f32 = 40.0;
/// [round 6 B1] Lane samples are tested against the body this far apart (m).
const FAN_STEP: f64 = 0.5;

/// [round 6 B1] The other street lanes of the junction object whose piece `own` the bus
/// stands on that its body covers: a car of `FAN_CAR_HALF` on the lane's axis would come
/// within `FAN_COVER` of the body (origin `pos`, `heading` degrees clockwise from north,
/// `front`/`rear` from the origin, `half_width`), and `through` says other vehicles drive
/// the lane. None on a lane of a spline: the lanes beside it there are its own road's
/// (`ThroughLanes::through` and `covers_lane` already judge it).
#[allow(clippy::too_many_arguments)]
pub fn fan_lanes_covered(net: &Network, own: usize, pos: glam::DVec2, heading: f32, front: f32, rear: f32, half_width: f32, through: &dyn Fn(usize) -> bool) -> Vec<usize> {
    let Some(here) = net.lanes.get(own) else { return Vec::new() };
    let Some(key) = here.key.filter(|_| here.source == 2) else { return Vec::new() };
    let h = (heading as f64).to_radians();
    let (fwd, right) = (glam::DVec2::new(h.sin(), h.cos()), glam::DVec2::new(h.cos(), -h.sin()));
    let (mid, half_len) = (((front - rear) * 0.5) as f64, ((front + rear) * 0.5) as f64);
    // (a car's flank on the axis within FAN_COVER of the body: the axis within this of it)
    let reach = (FAN_CAR_HALF + FAN_COVER) as f64;
    // (the distance of a point from the body's rectangle)
    let near_body = |q: glam::DVec2| {
        let r = q - pos;
        let along = (r.dot(fwd) - mid).abs() - half_len;
        let across = r.dot(right).abs() - half_width as f64;
        glam::DVec2::new(along.max(0.0), across.max(0.0)).length() < reach
    };
    let span = half_len + mid.abs() + reach + 1.0;
    let cell = |v: f64| (v / 50.0).floor() as i32;
    let mut out: Vec<usize> = Vec::new();
    for cx in cell(pos.x - span)..=cell(pos.x + span) {
        for cy in cell(pos.y - span)..=cell(pos.y + span) {
            for &i in net.grid.get(&(cx, cy)).map(|v| v.as_slice()).unwrap_or(&[]) {
                let l = &net.lanes[i];
                if i == own || out.contains(&i) || l.kind != LaneKind::Street || l.source != 2 || l.key.is_none_or(|k| (k.tile, k.id) != (key.tile, key.id)) {
                    continue;
                }
                let covered = l.points.windows(2).any(|w| {
                    let (a, b) = (w[0].truncate(), w[1].truncate());
                    let n = ((b - a).length() / FAN_STEP).ceil().max(1.0) as usize;
                    (0..=n).any(|k| near_body(a + (b - a) * (k as f64 / n as f64)))
                });
                if covered && through(i) {
                    out.push(i);
                }
            }
        }
    }
    out.sort_unstable();
    out
}

/// [round 6 B1] One of the others for `fan_queue`.
#[derive(Debug, Clone, Default)]
pub struct FanOther {
    pub id: u64,
    /// The lane it is on, and its way's lanes with the distance of each one's start from
    /// its front.
    pub lane: usize,
    pub way: Vec<(usize, f32)>,
    pub speed: f32,
    /// Seconds it has stood.
    pub stood: f32,
    /// The vehicle it follows.
    pub lead: Option<u64>,
    /// It stands at a stop of its own (a timetable bus serving its berth, or done with its
    /// trip there): it waits for nobody.
    pub serving: bool,
    /// The bus is what holds it: it waits for the bus at a junction, follows it, or the
    /// bus's body holds it (`Decision::wait_for` / `lead_id` / `body_block`).
    pub held_by_bus: bool,
}

/// [round 6 B1] The queue at the lanes a bus covers (`fan_lanes_covered`): of the vehicles
/// `others` the ones standing on a covered lane or whose way runs into one within
/// `FAN_QUEUE_REACH` and that the bus holds, and the ones standing queued behind those (as
/// far back as 30): the longest any of the first has stood, and how many stand in all.
/// One standing there for anything else - a red light, a give-way, people - is no queue of
/// the bus's (a bus off its stop over two fan lanes had five standing 60 s behind a light
/// there, none for it: review-r6b-regression b1c-after bus 252), nor a bus at a stop of its
/// own (two layovers at the berths of one fan would send each other round the block).
pub fn fan_queue(fan: &[usize], others: &[FanOther]) -> (f32, usize) {
    if fan.is_empty() {
        return (0.0, 0);
    }
    let standing = |o: &FanOther| o.speed < 0.3 && !o.serving;
    let mut chain: Vec<u64> = Vec::new();
    let mut secs = 0.0f32;
    for o in others.iter().filter(|o| standing(o) && o.held_by_bus) {
        if fan.contains(&o.lane) || o.way.iter().any(|&(l, d)| d <= FAN_QUEUE_REACH && fan.contains(&l)) {
            chain.push(o.id);
            secs = secs.max(o.stood);
        }
    }
    let mut k = 0;
    while k < chain.len() && chain.len() <= 30 {
        let id = chain[k];
        for o in others.iter().filter(|o| standing(o)) {
            if o.lead == Some(id) && !chain.contains(&o.id) {
                chain.push(o.id);
            }
        }
        k += 1;
    }
    (secs, chain.len())
}

/// [round 6 B1] A layover bus blocks the traffic and the queue asks it to go
/// (`Traffic::bus_detour`): it covers its own through lane (`covers_through_lane`) and the
/// queue behind it asks (`queue_asks_to_go`: `behind_secs` stood behind it, `behind` in the
/// queue), or it covers lanes of its fan and the queue held there asks (`fan_queue`'s
/// `fan_secs`, `fan_queued`).
pub fn layover_asked_to_go(covers_own: bool, behind_secs: f32, behind: usize, fan: &[usize], fan_secs: f32, fan_queued: usize) -> bool {
    (covers_own && queue_asks_to_go(behind_secs, behind)) || (!fan.is_empty() && queue_asks_to_go(fan_secs, fan_queued))
}

/// [ROLLBACK r4d-B2] The through lanes, as the traffic has driven them: a street lane is one
/// when a vehicle of the random traffic has been on it, when another timetable vehicle's
/// route runs over it, or when it has a lane beside it (a road of several lanes). The lane
/// density says nothing: 25 street lanes of BeijingCity with density 0 carried 287 cars in
/// ten minutes, and some with density 1 at the edge of the loaded map none.
#[derive(Debug, Default, Clone)]
pub struct ThroughLanes {
    /// Lanes the random traffic has been on.
    pub driven: std::collections::HashSet<usize>,
    /// Lanes on the timetable routes (as they are now), with the vehicles whose route they
    /// are on.
    pub routes: std::collections::HashMap<usize, Vec<u64>>,
}

impl ThroughLanes {
    /// Lane `l` is a through lane for vehicle `me`.
    pub fn through(&self, net: &Network, l: usize, me: u64) -> bool {
        let Some(lane) = net.lanes.get(l) else { return false };
        lane.kind == LaneKind::Street && (self.driven.contains(&l) || lane.left.is_some() || lane.right.is_some() || self.routes.get(&l).is_some_and(|v| v.iter().any(|&o| o != me)))
    }

    /// Lane `l` is on the way of any vehicle at all (its own route counts too): nobody parks
    /// on it.
    pub fn used(&self, l: usize) -> bool {
        self.driven.contains(&l) || self.routes.contains_key(&l)
    }
}

/// [ROLLBACK r4d-B2] Where on a lane a bus `len` metres long and `half_width` wide stands out
/// of the through traffic, nearest the start of the lane first: its origin's place and its
/// offset right of the lane. `lane_len`, `width`: the lane; `free(a, b)`: nobody on the lane
/// between `a` and `b`; `on_way`: some vehicle's way uses the lane; `bays`: other stops'
/// places off the lane (origin, offset); `parked`: the cars parked beside it (their boxes in
/// the lane's frame, [round 7 A10]);
/// `keep_off`: places along the lane a bus must not stand on (junction mouths, crossings).
/// Candidates, in order: another stop's bay, the lane itself when nobody's way uses it (a
/// lay-by, a stand), a gap in a parked row or the kerb before or after the row.
#[allow(clippy::too_many_arguments)]
pub fn room_on_lane(lane_len: f32, width: f32, len: f32, half_width: f32, on_way: bool, bays: &[(f32, f32)], parked: &[omsi_traffic::Parked], keep_off: &[f32], free: &dyn Fn(f32, f32, f32) -> bool) -> Option<(f32, f32)> {
    let need = len + PARK_ROOM;
    // off the lane: the body clear of the lane by the margin `covers_lane` asks
    let off = |lat: f32| lat.abs() >= half_width + width * 0.5 - 0.5;
    let clear = |mid: f32| mid - need * 0.5 >= 2.0 && mid + need * 0.5 <= lane_len - 2.0 && keep_off.iter().all(|&k| (k - mid).abs() > need * 0.5 + 4.0);
    // (a bus's origin is about its middle)
    for &(mid, lat) in bays {
        if off(lat) && clear(mid) && free(mid - need * 0.5, mid + need * 0.5, lat) {
            return Some((mid, lat));
        }
    }
    if !on_way && lane_len >= need + 4.0 {
        let s = lane_len - 2.0 - need * 0.5;
        if clear(s) && free(0.0, lane_len, 0.0) {
            return Some((s, 0.0));
        }
    }
    // (the bus in a row stands with its inner side where the cars' inner sides are, each car
    // by its own box)
    let at = |p: &omsi_traffic::Parked| p.lateral + p.lateral.signum() * (half_width - p.half_width);
    let mut row: Vec<omsi_traffic::Parked> = parked.iter().copied().filter(|p| off(at(p))).collect();
    row.sort_by(|a, b| a.s.total_cmp(&b.s));
    let mut spots: Vec<(f32, f32)> = Vec::new();
    if let (Some(first), Some(last)) = (row.first(), row.last()) {
        // the kerb before the row and after it
        spots.push((first.s - first.half_len - 1.0 - need * 0.5, at(first)));
        spots.push((last.s + last.half_len + 1.0 + need * 0.5, at(last)));
    }
    for w in row.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        if a.lateral.signum() == b.lateral.signum() && b.s - a.s - a.half_len - b.half_len >= need {
            spots.push(((a.s + b.s) * 0.5, (at(a) + at(b)) * 0.5));
        }
    }
    spots.sort_by(|a, b| a.0.total_cmp(&b.0));
    spots.into_iter().find(|&(mid, lat)| clear(mid) && free(mid - need * 0.5, mid + need * 0.5, lat))
}

/// [ROLLBACK r4d-B2] The queue behind a bus asks it to go: somebody has stood behind it for
/// longer than `LAYOVER_QUEUE`, or more than `LAYOVER_QUEUE_CARS` stand in it.
pub fn queue_asks_to_go(queue_behind_secs: f32, queued: usize) -> bool {
    queue_behind_secs > LAYOVER_QUEUE || queued > LAYOVER_QUEUE_CARS
}

/// [ROLLBACK r4d-B2] A way off for a bus at `s` on `lane` that is to be back at `back_to`
/// on that lane (where its origin stands at the stop) within `time_left` seconds: the
/// shortest loop through the network back onto the lane, or - if none fits - the nearest
/// place with room (`room`: where on a lane a bus can stand off the through traffic, its
/// origin's place and offset) with a way back. None when nothing fits.
pub fn plan_detour(net: &Network, lane: usize, s: f32, back_to: f32, time_left: f64, room: &dyn Fn(usize) -> Option<(f32, f32)>) -> Option<Detour> {
    let here = net.lanes.get(lane)?;
    let budget = time_left - DETOUR_MARGIN;
    let rest = here.length() - s;
    let way_len = |p: &[usize]| p[..p.len().saturating_sub(1)].iter().map(|&l| net.lanes[l].length()).sum::<f32>();
    // round the block
    let mut best: Option<(f32, Vec<usize>)> = None;
    for &n in &here.next {
        if n == lane || net.lanes[n].kind != LaneKind::Street {
            continue;
        }
        let Some(p) = net.shortest_path(n, lane) else { continue };
        let len = rest + way_len(&p) + back_to;
        // (a loop of a few metres is a fault in the lanes, not a block: BeijingCity has 1-2 m
        // "loops" of two lane pieces linked into each other)
        let around = len - rest - back_to;
        if around >= LAYOVER_LOOP_MIN && len <= LAYOVER_LOOP_MAX && detour_time(len) <= budget && best.as_ref().is_none_or(|b| len < b.0) {
            best = Some((len, p));
        }
    }
    if let Some((len, p)) = best {
        let route = std::iter::once(lane).chain(p).collect();
        return Some(Detour::Loop { route, len });
    }
    // to a place with room nearby: the lanes ahead nearest first
    use std::cmp::Reverse;
    use std::collections::{BinaryHeap, HashMap};
    let mm = |m: f32| (m.max(0.0) * 1000.0) as u64;
    let mut parent: HashMap<usize, usize> = HashMap::new();
    let mut seen: HashMap<usize, f32> = HashMap::new();
    let mut heap = BinaryHeap::new();
    for &n in &here.next {
        heap.push((Reverse(mm(rest)), n, lane));
    }
    while let Some((Reverse(c), l, from)) = heap.pop() {
        let c = c as f32 / 1000.0;
        if c > LAYOVER_PARK_MAX || seen.contains_key(&l) || l == lane {
            continue;
        }
        seen.insert(l, c);
        parent.insert(l, from);
        let ln = &net.lanes[l];
        if ln.kind != LaneKind::Street {
            continue;
        }
        if let Some((ps, lat)) = room(l) {
            let len = c + ps;
            let back = net.shortest_path(l, lane);
            if let (true, Some(back)) = (len <= LAYOVER_PARK_MAX, back) {
                let back_len = ln.length() - ps + way_len(&back[1..]) + back_to;
                // (the way back is bounded by the time it takes only)
                if detour_time(len) + detour_time(back_len) <= budget {
                    let mut route = vec![l];
                    let mut at = l;
                    while let Some(&p) = parent.get(&at) {
                        route.push(p);
                        if p == lane {
                            break;
                        }
                        at = p;
                    }
                    route.reverse();
                    return Some(Detour::Park { route, s: ps, lat, len, back, back_len });
                }
            }
        }
        let c2 = c + ln.length();
        for &n in &ln.next {
            if !seen.contains_key(&n) {
                heap.push((Reverse(mm(c2)), n, l));
            }
        }
    }
    None
}

/// [ROLLBACK r4d-B2] Put `route` (starting on the lane the bus is on) on the bus's way.
fn set_route(st: &mut AiState, net: &Network, route: Vec<usize>) {
    st.route = route;
    st.route_index = 0;
    st.change = None;
    st.planned_next = None;
    st.ahead.clear();
    st.plan_next(net);
}

/// Pull into the bay over this distance before the stop: the stop's docking distance,
/// 30 m unless its object strings say otherwise (Omsi.exe 0x620058, string 4; the bus
/// moves over once it is that near, 0x7dac5e).
const BAY_REACH: f32 = 30.0;
/// Pulling out: at least this long after the doors were told to close (s), at most this
/// long waiting for the script to say they are shut.
const CLOSE_MIN: f32 = 1.5;
const CLOSE_MAX: f32 = 12.0;
/// Brake for the stop from this far.
const STOP_REACH: f32 = 80.0;
/// This near its stop a timetable bus settles whether it stops there at all (Omsi.exe
/// 0x7da5b5: 50 m): not when nobody aboard wants to get off and nobody waits there,
/// unless it is the trip's first or last stop or the bus is more than `EARLY_STOP` early.
const SKIP_DECIDE: f32 = 50.0;
const EARLY_STOP: f64 = 120.0;
const EARLY_STOP_SHORT: f64 = 20.0;
/// [ROLLBACK r4d-B4] A bus held short of its pole by people serves the stop where it stands
/// only after this long (s); before, it creeps up to the pole once they are clear.
const PEOPLE_SERVE_MAX: f32 = 30.0;

/// [ROLLBACK r4d-B5] The room on the bus's way past a point `from` metres ahead of its origin
/// (its front at its stop point), less 2 m: up to the first junction after it, the meeting
/// place of a junction on the lane it is on (its mouth), a footpath crossing (a zebra: its
/// near edge), or the end of the way it knows; at a stop with a bay (`bay` over 0.3 m) no
/// further than `BAY_PULL` - where the bay ends is not known, and past it the bus would stand
/// across the lane.
fn room_past(ctx: &Ctx, from: f32, bay: f32) -> f32 {
    let mut limit = ctx.way.last().map(|&(l, dl)| dl + ctx.net.lanes[l].length()).unwrap_or(from);
    for &(l, dl) in ctx.way {
        let crossings = ctx.net.crossings.get(l).map(|c| c.as_slice()).unwrap_or(&[]);
        if dl > from && !crossings.is_empty() {
            limit = limit.min(dl);
        }
        for c in crossings {
            let at = dl + c.at - c.before;
            if at > from {
                limit = limit.min(at);
            }
        }
        for &(_, at, _) in ctx.net.walks.get(l).map(|w| w.as_slice()).unwrap_or(&[]) {
            let edge = dl + at - ZEBRA_HALF;
            if dl + at + ZEBRA_HALF > from {
                limit = limit.min(edge);
            }
        }
    }
    let room = (limit - from - 2.0).max(0.0);
    if bay.abs() > 0.3 {
        room.min(BAY_PULL)
    } else {
        room
    }
}

/// [ROLLBACK r4d-B5] Half the width of a footpath crossing (m), and how far on a bus pulls
/// at most in a bay (m).
const ZEBRA_HALF: f32 = 2.0;
const BAY_PULL: f32 = 4.0;

/// What the service needs of the world this frame.
pub struct Ctx<'a> {
    pub net: &'a Network,
    /// The car's way ahead: (lane, distance from its origin to the lane's start).
    pub way: &'a [(usize, f32)],
    pub day_time: f64,
    pub dt: f32,
    pub id: u64,
    /// Seconds it has stood still without a stop of its own.
    pub stopped: f32,
    /// Passing something (the passing manoeuvre owns the lateral position).
    pub passing: bool,
    /// [ROLLBACK bus-106] Another bus stands at the same stop close ahead of it (its tail within
    /// 9 m): this one, once in its bay and still, serves the stop where it stands.
    pub lead_close: bool,
    /// [ROLLBACK bus-108] Another bus comes up behind it for the same stop: it pulls on a
    /// few metres past its stop point, to leave the one behind room in the bay.
    pub follower_close: bool,
    /// Somebody aboard wants to get off at the front stop or somebody waits there (None:
    /// nobody knows - no passengers run - and every stop is served).
    pub wanted: Option<bool>,
    pub debug: bool,
    /// The front stop lies on a through lane (random traffic runs on it) with no bay: a bus
    /// standing there blocks the lane.
    pub roadside: bool,
    /// Seconds a vehicle has stood right behind this bus (0 when none does).
    pub queue_behind_secs: f32,
    /// [ROLLBACK r4d-B3] Its usual speed on the lane it is on (m/s: the limit at its driver's
    /// speed factor).
    pub v_nom: f32,
    /// [ROLLBACK r4d-B4] What holds it is somebody in its way (`Why::People`).
    pub people_hold: bool,
    /// [ROLLBACK r4d-B3] A vehicle follows it within `PACE_FOLLOWER` m: it does not drive
    /// slower than the traffic.
    pub close_behind: bool,
    /// [ROLLBACK r4d-B3] It stands over a through lane (`covers_through_lane`, as the layover
    /// rules have it).
    pub covers_through: bool,
}

impl BusService {
    pub fn new(stops: Vec<Stop>) -> BusService {
        BusService {
            stops: stops.into(),
            phase: Phase::Running,
            phase_t: 0.0,
            boarding: 0.0,
            leave_at: 0.0,
            boarded: false,
            delay: 0.0,
            layover: false,
            long_wait: false,
            roadside_layover: false,
            route_open: false,
            terminus: String::new(),
            near_d: f32::INFINITY,
            serve: None,
            last_stop: None,
            always: Vec::new(),
            serve_early: Vec::new(),
            wait_stops: Vec::new(),
            away: None,
            pending: None,
            detour_looked: f64::NEG_INFINITY,
            detour_none: None,
            this_layover: false,
            rerouted: false,
            ahead_by: 0.0,
            bay_wish: None,
            served: None,
            pace: 1.0,
            pace_target: None,
            left_early: false,
            short_by_people: false,
            pull: 0.0,
            pull_to: None,
            pulled_here: false,
            track_end: false,
        }
    }

    /// Doors open for people (`AI_Scheduled_AtStation`).
    pub fn at_station(&self) -> bool {
        self.phase == Phase::Boarding
    }

    /// The side's doors the bus opens at the stop it is at (`AI_Scheduled_AtStation_Side`):
    /// the front stop's while it boards, else 0 (nobody at a stop, nothing to open).
    pub fn at_station_side(&self) -> f32 {
        if self.phase == Phase::Boarding {
            self.stops.front().map(|s| s.side).unwrap_or(0.0)
        } else {
            0.0
        }
    }

    pub fn trip_done(&self) -> bool {
        self.phase == Phase::TripDone
    }

    /// [round 6 B2] Its trip is over at the end of its track, and it drives on off the end
    /// of its route (the planner does not hold it there: `External::trip_over`).
    pub fn drives_off(&self) -> bool {
        self.phase == Phase::TripDone && self.track_end
    }

    /// [round 6 B2] Driving with no stop left, it has reached the last lane of its route,
    /// a street lane with a way on, and the route is complete (not `route_open`) and its own
    /// (not round the block or to a place with room).
    fn track_ends(&self, st: &AiState, net: &Network) -> bool {
        self.phase == Phase::Running
            && self.stops.is_empty()
            && !self.route_open
            && self.away.is_none()
            && self.pending.is_none()
            && !st.route.is_empty()
            && st.route_index + 1 >= st.route.len()
            && st.route.last() == Some(&st.lane)
            && net.lanes.get(st.lane).is_some_and(|l| l.kind == LaneKind::Street && !l.next.is_empty())
    }

    /// [ROLLBACK aiparked-73] Standing out a long layover with the engine off: it starts again
    /// `PARKED_START` before the doors open for the departure.
    pub fn parked(&self, day_time: f64) -> bool {
        // [ROLLBACK terminus-130] a bus that has finished its trip and stands at the end of
        // the line waits for its next trip with the engine off, as one placed there does
        // (it stood with the engine running until the next trip was handed to it)
        if self.phase == Phase::TripDone {
            return !self.track_end && self.phase_t > TRIP_DONE_PARK;
        }
        matches!(self.phase, Phase::Waiting | Phase::Boarding) && self.long_wait && self.leave_at - day_time > PARKED_START
    }

    /// Standing at a stop (boarding, waiting or pulling out).
    pub fn at_stop(&self) -> bool {
        matches!(self.phase, Phase::Boarding | Phase::Waiting | Phase::Closing)
    }

    /// Seconds it expects to stand where it is yet (for the traffic behind: worth going
    /// round, or worth waiting for).
    pub fn standing_for(&self, day_time: f64) -> f32 {
        let wait = (self.leave_at - day_time).max(0.0) as f32;
        match self.phase {
            Phase::Running => 0.0,
            Phase::Boarding => self.boarding.max(0.0) + 2.0 + wait,
            Phase::Waiting => wait + 2.0,
            Phase::Closing => 1.0,
            Phase::TripDone if self.track_end => 0.0,
            Phase::TripDone => 600.0,
        }
    }

    /// Somebody is still at the doors: keep them open for `secs` more - for somebody
    /// coming from stop `stop` only while the bus serves that stop (Omsi.exe 0x7d9f1d: the
    /// person's stop is the bus's), not a stop it stands next to.
    pub fn hold(&mut self, stop: Option<i64>, secs: f32) {
        let here = stop.is_none_or(|s| self.stops.front().is_some_and(|f| f.id == s));
        if self.phase == Phase::Boarding && here {
            self.boarding = self.boarding.max(secs);
        }
    }

    /// A new trip (the tour's next, or the rest of a trip).
    pub fn restart(&mut self, stops: Vec<Stop>, layover: bool) {
        self.near_d = f32::INFINITY;
        self.serve = None;
        self.short_by_people = false;
        self.pace_target = None;
        self.pull = 0.0;
        self.stops = stops.into();
        self.phase = Phase::Running;
        self.phase_t = 0.0;
        self.boarding = 0.0;
        self.boarded = false;
        self.layover = layover;
        self.roadside_layover = false;
        self.away = None;
        self.pending = None;
        // [ROLLBACK r4d-B3] a new trip has a timetable of its own
        self.ahead_by = 0.0;
        self.pace = 1.0;
        self.pace_target = None;
        self.left_early = false;
        self.track_end = false;
    }

    fn set_phase(&mut self, p: Phase) {
        self.phase = p;
        self.phase_t = 0.0;
    }

    /// A stop it serves whoever wants it or not: the trip's first (a layover) and last, the
    /// ones its timetable says it always serves, and any stop it would reach more than
    /// `EARLY_STOP` early (`EARLY_STOP_SHORT` at a stop marked for it).
    fn must_serve(&self, stop: &Stop, day_time: f64) -> bool {
        let last = (self.stops.len() == 1 && !self.route_open) || self.last_stop == Some(stop.id);
        let early = stop.depart - day_time;
        last || self.layover
            || (early > EARLY_STOP && self.wait_stops.contains(&stop.id))
            || self.always.contains(&stop.id)
            || (early > EARLY_STOP_SHORT && self.serve_early.contains(&stop.id))
    }

    /// Arrived at the front stop: what now.
    fn arrive(&mut self, ctx: &Ctx, depart: f64, id: i64, at: (usize, f32)) {
        if omsi_cfg::env::var_os("OMSI_DEBUG_STOPS").is_some() {
            log::info!("t={:.1}: timetable bus {} serves its stop {:?}", ctx.day_time, ctx.id, self.stops.front().map(|s| s.id));
        }
        let layover = std::mem::take(&mut self.layover);
        // [ROLLBACK r4d-B2] back from round the block (or never away)
        self.away = None;
        self.this_layover = layover;
        self.short_by_people = false;
        self.pace_target = None;
        self.pull = 0.0;
        self.pulled_here = false;
        self.pull_to = None;
        let rail = ctx.net.lanes.get(at.0).is_some_and(|l| l.kind == LaneKind::Rail);
        // [ROLLBACK waitstop-136] it stands out its time where the timetable says so, on its
        // layover and (a train) at its stations; elsewhere it goes when the people are done
        let wait = if layover || rail || self.wait_stops.contains(&id) {
            early_wait(depart, ctx.day_time, layover, rail)
        } else {
            0.0
        };
        // [ROLLBACK r4d-B3] still ahead of its timetable from an early start: it waits the
        // rest out here - where it is out of the way (not at a stop on a through lane: the
        // next stop of a terminus a few metres on had it wait out the whole lead again in
        // the lane it had left early to free; there it keeps making the lead up on the way)
        let wait = if self.ahead_by >= 1.0 && !ctx.covers_through {
            let rest = std::mem::take(&mut self.ahead_by).min((depart - ctx.day_time).max(0.0));
            log::info!("t={:.1}: timetable bus {} at its stop {id} {rest:.0} s early after its early start: it waits", ctx.day_time, ctx.id);
            wait.max(rest)
        } else {
            wait
        };
        self.pace = 1.0;
        self.pace_target = None;
        self.leave_at = ctx.day_time + wait;
        self.long_wait = wait >= PARKED_WAIT;
        self.boarding = boarding_time(ctx.id);
        self.boarded = false;
        // a layover opens the doors for the last minute only (out of the way, at a terminus
        // or in a depot); anywhere else people get off straight away. At a roadside stop on
        // a through lane the doors open at once: people wait outside, and the bus blocks the
        // lane whatever it does, so it may as well serve them
        self.roadside_layover = layover && ctx.roadside;
        if layover && !ctx.roadside && wait > LAYOVER_BOARDING + 10.0 {
            self.set_phase(Phase::Waiting);
        } else {
            self.set_phase(Phase::Boarding);
            self.boarded = true;
        }
        self.delay = (ctx.day_time + (self.boarding as f64).max(wait)) - depart;
        if ctx.debug {
            log::info!(
                "t={:.1}: timetable bus {} at its stop, {:.0} s to its departure ({:?}) at {:?}",
                ctx.day_time,
                ctx.id,
                depart - ctx.day_time,
                self.phase,
                ctx.net.lanes.get(at.0).map(|l| { let p = l.at(at.1).0; (p.x.round(), p.y.round()) })
            );
        }
    }

    /// Off from the stop: the next one is the front.
    fn depart(&mut self, st: &mut AiState, vehicle: &mut VehicleInstance, ctx: &Ctx) {
        let bay = self.stops.front().map(|s| s.bay).unwrap_or(0.0);
        // [ROLLBACK r4d-B3] off early for the traffic behind: the lead is made up on the way
        if std::mem::take(&mut self.left_early) {
            if let Some(stop) = self.stops.front() {
                self.ahead_by = (stop.depart - ctx.day_time).max(0.0);
                log::info!("t={:.1}: timetable bus {} leaves its stop {} {:.0} s early for the traffic behind it", ctx.day_time, ctx.id, stop.id, self.ahead_by);
            }
        }
        if let Some(d) = self.pending.take() {
            // [ROLLBACK r4d-B2] off round the block (or to a place with room): the stop stays
            // the next one
            self.take_detour(st, ctx.net, d, ctx.day_time, ctx.id);
        } else {
            self.served = self.stops.pop_front().map(|s| s.id);
            crate::traffic::ibis_to_next_stop(vehicle, self.stops.len());
        }
        self.near_d = f32::INFINITY;
        self.serve = None;
        self.short_by_people = false;
        self.pace_target = None;
        self.pull = 0.0;
        let air = ctx.net.lanes[st.lane].kind == LaneKind::Air;
        if self.stops.is_empty() && !self.route_open && !air && self.away.is_none() {
            self.set_phase(Phase::TripDone);
            st.signal = 0;
            st.signal_time = 0.0;
            return;
        }
        self.set_phase(Phase::Running);
        // the blinker stays on while it pulls out
        st.signal = out_signal(bay);
        st.signal_time = 2.5;
        // out of the bay, and back in the lane before the next junction
        let lat = st.lateral;
        if lat.abs() > 0.05 && !ctx.passing {
            let junction = ctx
                .way
                .iter()
                .find(|&&(l, dl)| dl > 0.0 && !ctx.net.crossings[l].is_empty())
                .map(|w| (w.1 - st.front - 1.0).max(4.0));
            let len = (lat.abs() * 8.0).clamp(8.0, 30.0).min(junction.unwrap_or(f32::MAX));
            st.lateral_target = 0.0;
            st.lateral_ramp = (lat, 0.0, st.odometer, len);
        }
        self.bay_wish = None;
        if ctx.debug {
            log::info!(
                "t={:.1}: timetable bus {} pulls away, IBIS stop {:?}",
                ctx.day_time,
                ctx.id,
                vehicle.var("IBIS_busstop")
            );
        }
    }

    /// [ROLLBACK r4d-B2] The seconds left before the bus may leave the front stop, when it
    /// stands out a long wait there - or queues for it (`queued`, standing behind what holds
    /// the stop) - and a way off is worth looking for now.
    pub fn detour_wanted(&self, speed: f32, day_time: f64, queued: bool) -> Option<f64> {
        if self.away.is_some() || self.pending.is_some() || day_time - self.detour_looked < DETOUR_RETRY {
            return None;
        }
        let stop = self.stops.front()?;
        let left = match self.phase {
            // (only its layover - at a terminus, at the first stop of a trip -, not the wait
            // at a stop on the way where the timetable holds it, nor a lead waited out)
            Phase::Waiting | Phase::Boarding if self.this_layover => self.leave_at - day_time,
            Phase::Running if queued && speed < 0.3 => stop.depart - day_time,
            _ => return None,
        };
        (left > DETOUR_MIN_LEFT).then_some(left)
    }

    /// [ROLLBACK r4d-B2] Looked for a way off and found none: it stays (said once a stop).
    pub fn no_detour(&mut self, day_time: f64, id: u64) {
        self.detour_looked = day_time;
        let stop = self.stops.front().map(|s| s.id);
        if self.detour_none != stop {
            self.detour_none = stop;
            log::info!("t={day_time:.1}: timetable bus {id} blocks a through lane at its stop {stop:?}, and no loop or place with room fits before its departure: it stays");
        }
    }

    /// [ROLLBACK r4d-B2] Leave the stop (or the queue for it) on `d`: at once while driving,
    /// at a stop once the doors are shut (`depart`).
    pub fn start_detour(&mut self, st: &mut AiState, net: &Network, d: Detour, day_time: f64, id: u64) {
        self.detour_looked = day_time;
        match self.phase {
            Phase::Running => self.take_detour(st, net, d, day_time, id),
            // the doors shut: off once they are; open: once the people at them are in or out
            // (`board`: the passengers holding them are waited for as at any stop)
            Phase::Waiting => {
                self.pending = Some(d);
                self.leave_at = day_time;
                self.set_phase(Phase::Closing);
            }
            Phase::Boarding => {
                self.pending = Some(d);
                self.leave_at = day_time;
            }
            _ => {}
        }
    }

    fn take_detour(&mut self, st: &mut AiState, net: &Network, d: Detour, day_time: f64, id: u64) {
        let cur = st.route_index;
        let rest: Vec<usize> = st.route.iter().skip(cur + 1).copied().collect();
        let layover = self.this_layover || self.layover;
        let depart = self.stops.front().map(|s| s.depart).unwrap_or(day_time);
        match d {
            Detour::Loop { route, len } => {
                log::info!("t={day_time:.1}: timetable bus {id} goes round the block ({len:.0} m, {} lanes) to be back for {:.0} s from now", route.len(), depart - day_time);
                let k = route.len() - 1;
                for s in self.stops.iter_mut() {
                    s.ri = s.ri.saturating_sub(cur) + k;
                }
                let mut r = route;
                r.extend(rest);
                set_route(st, net, r);
                self.away = Some(Away { park: None });
            }
            Detour::Park { route, s, lat, len, back, back_len } => {
                let leave_at = depart - detour_time(back_len) - DETOUR_MARGIN * 0.5;
                log::info!("t={day_time:.1}: timetable bus {id} parks {len:.0} m off at lane {} until {:.0} s before its departure", route.last().copied().unwrap_or(0), depart - leave_at);
                let stops = self
                    .stops
                    .drain(..)
                    .map(|mut x| {
                        x.ri = x.ri.saturating_sub(cur);
                        x
                    })
                    .collect();
                let ri = route.len() - 1;
                set_route(st, net, route);
                self.away = Some(Away { park: Some(ParkAway { ri, s, lat, back, rest, stops, leave_at, arrived: false }) });
            }
        }
        // (back at the stop it waits for its departure as on its layover)
        self.layover = layover;
        self.near_d = f32::INFINITY;
        self.serve = None;
        self.short_by_people = false;
        self.pace_target = None;
        self.pull = 0.0;
        self.rerouted = true;
        self.set_phase(Phase::Running);
    }

    /// [ROLLBACK r4d-B2] Driving to the place with room, standing there, and setting off back
    /// for the stop when it is time: where it has to stop, if anywhere.
    fn park_away(&mut self, st: &mut AiState, ctx: &Ctx) -> Option<f32> {
        let away = self.away.as_mut()?;
        let p = away.park.as_mut()?;
        if !p.arrived {
            let d = st.route_distance(ctx.net, p.ri, p.s);
            if !ctx.passing {
                if d < BAY_REACH {
                    st.lateral_target = p.lat;
                }
                self.bay_wish = (d < BAY_REACH).then_some(p.lat);
            }
            if st.speed < 0.3 && (d < 1.5 || (ctx.stopped > 5.0 && d < 15.0 && (st.lateral - p.lat).abs() < 0.5)) {
                p.arrived = true;
            }
            return Some(d.max(0.0) + st.front - 0.3);
        }
        if ctx.day_time < p.leave_at {
            return Some(st.front);
        }
        // back: the way to the lane it left, and the route on from there
        let p = away.park.take()?;
        let k = p.back.len() - 1;
        let mut route = p.back;
        route.extend(p.rest);
        set_route(st, ctx.net, route);
        self.stops = p.stops.into_iter().map(|mut x| {
            x.ri += k;
            x
        }).collect();
        self.rerouted = true;
        st.signal = out_signal(p.lat);
        st.signal_time = 2.5;
        st.lateral_target = 0.0;
        self.bay_wish = None;
        if ctx.debug {
            log::info!("t={:.1}: timetable bus {} sets off back for its stop", ctx.day_time, ctx.id);
        }
        None
    }

    /// One frame at a stop with the doors shut: they open for the last minute.
    fn wait(&mut self, ctx: &Ctx) {
        let board_at = self.leave_at - LAYOVER_BOARDING;
        if !self.boarded && ctx.day_time >= board_at {
            self.boarded = true;
            self.boarding = self.boarding.max((self.leave_at - ctx.day_time) as f32 - 3.0);
            self.set_phase(Phase::Boarding);
        } else if self.boarded && ctx.day_time >= self.leave_at {
            self.set_phase(Phase::Closing);
        }
    }

    /// One frame at a stop with the doors open.
    fn board(&mut self, ctx: &Ctx) {
        self.boarding -= ctx.dt;
        // a layover on a through lane with a queue behind it gives up the rest of its wait;
        // the closing sequence below then runs as usual
        if self.roadside_layover && ctx.queue_behind_secs > ROADSIDE_QUEUE && self.leave_at - ctx.day_time > ROADSIDE_EARLY_MIN {
            self.leave_at = ctx.day_time;
            self.left_early = true;
        }
        // an early bus waits for its departure with the doors open, as drivers do
        // (shut, it stood at the stop for half a minute for no reason anyone could
        // see); the doors close when it is time to go
        if self.boarding <= 0.0 && ctx.day_time >= self.leave_at {
            self.set_phase(Phase::Closing);
        } else if ctx.day_time >= self.leave_at && self.phase_t > BOARDING_MAX {
            // [ROLLBACK stopstuck-130] somebody always said to be at the doors (a
            // passenger who never gets in) kept the bus at its stop for good
            self.set_phase(Phase::Closing);
        }
    }

    /// One frame of the service: where the bus has to stop (distance from its origin along
    /// its way), if anywhere; it also sets its blinker and its place across the lane.
    pub fn step(&mut self, st: &mut AiState, vehicle: &mut VehicleInstance, ctx: &Ctx) -> Option<f32> {
        self.phase_t += ctx.dt;
        let here = Some(st.front);
        match self.phase {
            // [round 6 B2] (at the end of its track it drives on: no stop)
            Phase::TripDone if self.track_end => None,
            Phase::TripDone => here,
            Phase::Waiting => {
                self.wait(ctx);
                // [ROLLBACK r4d-B5] doors shut, a bus behind it for this stop: it pulls on by
                // its length (as far as the road before the next junction lets it) to leave
                // the one behind the pole
                if ctx.follower_close && !self.pulled_here && self.phase == Phase::Waiting && !ctx.passing {
                    self.pulled_here = true;
                    let bay = self.stops.front().map_or(0.0, |x| x.bay);
                    let pull = (st.front + st.rear).min(room_past(ctx, st.front, bay));
                    if pull >= 3.0 {
                        self.pull_to = Some(st.odometer + pull);
                    }
                }
                match self.pull_to {
                    Some(to) if to - st.odometer > 0.3 && self.phase == Phase::Waiting => Some(st.front + (to - st.odometer)),
                    _ => {
                        self.pull_to = None;
                        here
                    }
                }
            }
            Phase::Boarding => {
                self.board(ctx);
                here
            }
            Phase::Closing => {
                // blinker on towards the traffic while the doors close and the stop brake
                // comes off. The bus leaves when its script says the doors are shut (it
                // writes AI_Scheduled_AtStation back to 0), as OMSI's buses do; leaving
                // after a fixed 3.5 s drove buses off with their doors still closing, and
                // each type at its own moment. A script that never answers is not waited
                // for longer than CLOSE_MAX.
                let bay = self.stops.front().map(|s| s.bay).unwrap_or(0.0);
                st.signal = out_signal(bay);
                st.signal_time = st.signal_time.max(1.0);
                let answered = vehicle.station_released() && vehicle.var("AI_Scheduled_AtStation").map(|v| v.abs() < 0.5).unwrap_or(true);
                if (answered && self.phase_t >= CLOSE_MIN) || self.phase_t >= CLOSE_MAX {
                    self.depart(st, vehicle, ctx);
                    if self.phase == Phase::TripDone {
                        return here;
                    }
                    return None;
                }
                here
            }
            Phase::Running => self.approach(st, vehicle, ctx),
        }
    }

    /// Driving: brake for the next stop, pull into its bay.
    fn approach(&mut self, st: &mut AiState, vehicle: &mut VehicleInstance, ctx: &Ctx) -> Option<f32> {
        if self.away.as_ref().is_some_and(|a| a.park.is_some()) {
            return self.park_away(st, ctx);
        }
        loop {
            let Some(stop) = self.stops.front().copied() else {
                // ([round 7 A7] out of any bay: where it sits in its lane is the traffic's -
                // the target is not written here every frame)
                self.bay_wish = None;
                // [round 6 B2] the end of its track with no stop left: its trip is over here
                // (the timetable hands it the tour's next trip where it is, or lets it go),
                // as at its last stop - not on at a tile edge (`route_open`: it waits for the
                // tiles), nor round the block, nor at a dead end (the road ends: it drives
                // off it and its trip comes back past it, r4d-B1)
                if self.track_ends(st, ctx.net) {
                    self.track_end = true;
                    self.set_phase(Phase::TripDone);
                    st.signal = 0;
                    st.signal_time = 0.0;
                    log::info!("t={:.1}: timetable bus {} at the end of its track on lane {} with no stop left: its trip is over", ctx.day_time, ctx.id, st.lane);
                }
                return None;
            };
            let speed = st.speed;
            // A bus creeping the last metres to its stop point is at its stop, even when the
            // point lies past the end of the stop's lane and the lane ahead takes over first:
            // measured from the new lane the stop was suddenly 20-30 m behind, and the bus -
            // blinker on, pulled into the bay - drove on without opening its doors.
            let crept_past = self.near_d < 12.0 && speed < 4.0;
            if stop.ri < st.route_index {
                if crept_past {
                    self.near_d = f32::INFINITY;
                    self.arrive(ctx, stop.depart, stop.id, (st.lane, st.s));
                    return Some(st.front);
                }
                // behind it already (the route was cut short)
                self.stops.pop_front();
                self.near_d = f32::INFINITY;
                self.serve = None;
                self.short_by_people = false;
                self.pace_target = None;
                self.pull = 0.0;
                continue;
            }
            let d = st.route_distance(ctx.net, stop.ri, stop.s);
            // [ROLLBACK r4d-B3] slower while ahead of the timetable: the lead goes by the time
            // it is at the stop (the time it drives slower is time it is behind its usual pace)
            if self.ahead_by >= 1.0 || self.pace_target.is_some() {
                let eta = ctx.day_time + (d.max(0.0) / ctx.v_nom.max(1.0)) as f64;
                let target = *self.pace_target.get_or_insert(eta + self.ahead_by);
                self.ahead_by = (target - eta).max(0.0);
                if self.ahead_by < 1.0 {
                    self.ahead_by = 0.0;
                    self.pace_target = None;
                }
                // (never slower than the traffic behind it)
                self.pace = if ctx.close_behind { 1.0 } else { pace_for(self.ahead_by, d, ctx.v_nom) };
            } else {
                self.pace = 1.0;
            }
            // near enough to see whether anybody wants it (a train keeps to its stations)
            if self.serve.is_none() && d < SKIP_DECIDE {
                let rail = ctx.net.lanes.get(st.lane).is_some_and(|l| l.kind == LaneKind::Rail);
                self.serve = Some(rail || self.must_serve(&stop, ctx.day_time) || ctx.wanted.unwrap_or(true));
            }
            if self.serve == Some(false) {
                if ctx.debug || omsi_cfg::env::var_os("OMSI_DEBUG_STOPS").is_some() {
                    log::info!("t={:.1}: timetable bus {} passes its stop {}: nobody gets off or on", ctx.day_time, ctx.id, stop.id);
                }
                self.stops.pop_front();
                self.near_d = f32::INFINITY;
                self.serve = None;
                self.short_by_people = false;
                self.pace_target = None;
                self.pull = 0.0;
                crate::traffic::ibis_to_next_stop(vehicle, self.stops.len());
                // ([round 7 A7] out of any bay: where it sits in its lane is the traffic's -
                // the target is not written here every frame)
                self.bay_wish = None;
                continue;
            }
            // into the bay over the last metres, but only once no junction lies between
            // the bus and its stop: the meeting places of a junction are laid out for
            // vehicles in the middle of their lane. Not where the stop lies too close
            // behind the junction for the S-curve into the bay (`AiState::lateral`):
            // the bus came to rest in the middle of its lane, too far from the pole for
            // anybody to get on, and stood there for good (DBC_Map, Grand and Peshtigo)
            if !ctx.passing {
                let junction_end = ctx
                    .way
                    .iter()
                    .filter(|&&(l, dl)| dl < d && !ctx.net.crossings[l].is_empty())
                    .map(|&(l, dl)| dl + ctx.net.lanes[l].length())
                    .reduce(f32::max);
                let ramp = ((stop.bay - st.lateral).abs() * 8.0).clamp(8.0, 30.0);
                let junction_first = junction_end.is_some_and(|e| d - e >= ramp);
                let entering = d < BAY_REACH && d > -25.0 && !junction_first;
                if entering {
                    st.lateral_target = stop.bay;
                }
                self.bay_wish = entering.then_some(stop.bay);
            }
            // queued behind something standing at its stop (another bus, the player's):
            // after a while it serves the stop where it stands, as drivers do
            let in_bay = (st.lateral - stop.bay).abs() < 0.5;
            // (a bus with another one behind it for this stop pulls on past the stop point:
            // [ROLLBACK r4d-B5] by its length, as far as the road before the next junction
            // lets it, so that the one behind serves the pole)
            if ctx.follower_close && d > -2.0 {
                let want = (st.front + st.rear).min(room_past(ctx, d.max(0.0) + st.front, stop.bay));
                self.pull = self.pull.max(want);
            }
            let pull = self.pull;
            let close = ctx.lead_close
                && in_bay
                && speed < 0.5
                && ctx.stopped > 0.8
                && d < 45.0
                && d >= 2.0;
            // [ROLLBACK r4d-B4] (not when people held it short: it creeps up to the pole once
            // they are clear, unless it is within 2 m of it)
            if ctx.people_hold && d >= 2.0 {
                self.short_by_people = true;
            }
            let people_short = self.short_by_people && ctx.stopped < PEOPLE_SERVE_MAX;
            // [ROLLBACK r4d-B2] (a bus coming to its layover queues for the berth instead: it
            // stands out its wait there, not in the lane behind - only right behind another
            // bus at the stop, `close`, it serves it where it stands; [P3-1] and held short of the
            // berth by people for PEOPLE_SERVE_MAX, as any bus: they are its own waiters, who
            // wait for it to come - d24 #194 stood 290 s before its first stop, its waiters on a
            // 3.1 m island 0.03 m clear of its body on its lane and inside it on the way into the
            // bay)
            let queued = close || (speed < 0.3 && ctx.stopped > 6.0 && d < 45.0 && d >= 2.0 && !people_short && (!self.layover || self.short_by_people));
            if (d < 2.0 && speed < 0.3) || queued || (d < -2.0 - pull && crept_past) {
                self.near_d = f32::INFINITY;
                self.arrive(ctx, stop.depart, stop.id, (st.lane, st.s));
                return Some(st.front);
            }
            self.near_d = d;
            if d < -2.0 - pull {
                // missed it
                if ctx.debug || omsi_cfg::env::var_os("OMSI_DEBUG_STOPS").is_some() {
                    log::info!("t={:.1}: timetable bus {} missed its stop ({:.1} m past, {:.1} m/s, stood {:.1} s, passing {})", ctx.day_time, ctx.id, -d, speed, ctx.stopped, ctx.passing);
                }
                self.stops.pop_front();
                self.near_d = f32::INFINITY;
                self.serve = None;
                self.short_by_people = false;
                self.pace_target = None;
                self.pull = 0.0;
                continue;
            }
            if d < STOP_REACH {
                // the stop point is where the front of the bus comes to rest
                return Some((d + pull).max(0.0) + st.front - 0.3);
            }
            return None;
        }
    }
}

/// How far before a station's point a vehicle stops with its origin, as Omsi.exe measures
/// the way to the station (0x7da4e3): from the origin of the vehicle that leads, less half
/// its length for a train (its front comes to rest at the station, not its middle - the
/// S-Bahn stopped with half a car past the end of the platform), plus the holding point
/// offset of its `[ai_brakeperformance]` ("to correct unprecise braking").
pub fn stop_shift(ty: &omsi_sim::VehicleType, rail: bool) -> f32 {
    let hold = ty.def.ai_brake_performance.map(|b| b[4]).unwrap_or(0.0);
    let half = if rail { ty.half_length().unwrap_or(0.0) } else { 0.0 };
    half - hold
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Somebody coming from another stop than the one the bus serves does not keep it there
    /// (#767); somebody on the way out does, wherever.
    #[test]
    fn a_hold_counts_at_the_stop_served() {
        let stop = Stop { ri: 0, s: 0.0, bay: 0.0, depart: 0.0, id: 42, side: 0.0 };
        let mut s = BusService::new(vec![stop]);
        s.phase = Phase::Boarding;
        s.boarding = 0.0;
        s.hold(Some(41), 2.5);
        assert_eq!(s.boarding, 0.0);
        s.hold(Some(42), 2.5);
        assert_eq!(s.boarding, 2.5);
        s.boarding = 0.0;
        s.hold(None, 2.5);
        assert_eq!(s.boarding, 2.5);
    }

    #[test]
    fn an_early_bus_waits_for_its_time() {
        // five minutes early: until 20 s before its departure (a train: 2 min)
        assert_eq!(early_wait(400.0, 100.0, false, false), 280.0);
        assert_eq!(early_wait(400.0, 100.0, false, true), 180.0);
        // late, or nearly on time: off at once
        assert_eq!(early_wait(400.0, 390.0, false, false), 0.0);
        assert_eq!(early_wait(400.0, 500.0, false, false), 0.0);
        // a layover: to the departure itself
        assert_eq!(early_wait(400.0, 100.0, true, false), 300.0);
    }

    #[test]
    fn a_bus_at_the_end_of_its_line_stops_the_engine() {
        // [ROLLBACK terminus-130]
        let mut s = BusService::new(vec![]);
        s.phase = Phase::TripDone;
        s.phase_t = 5.0;
        assert!(!s.parked(0.0));
        s.phase_t = TRIP_DONE_PARK + 1.0;
        assert!(s.parked(0.0));
        s.phase = Phase::Running;
        assert!(!s.parked(0.0));
    }

    #[test]
    fn standing_time() {
        let mut s = BusService::new(vec![]);
        assert_eq!(s.standing_for(0.0), 0.0);
        s.phase = Phase::Waiting;
        s.leave_at = 100.0;
        assert!((s.standing_for(40.0) - 62.0).abs() < 1e-3);
        s.phase = Phase::TripDone;
        assert!(s.standing_for(0.0) > 100.0);
    }

    #[test]
    fn only_the_ends_of_the_trip_and_an_early_bus_stop_for_nobody() {
        let stop = |id: i64, depart: f64| Stop::from_tuple((0, 0.0, 0.0, depart, id, 0.0));
        let mut s = BusService::new(vec![stop(1, 100.0), stop(2, 200.0), stop(3, 300.0)]);
        s.last_stop = Some(3);
        // on time at a stop in the middle: only if somebody wants it
        assert!(!s.must_serve(&stop(2, 200.0), 150.0));
        // over two minutes early: it stops and waits, where the timetable says it waits
        assert!(!s.must_serve(&stop(2, 200.0), 70.0));
        s.wait_stops = vec![2];
        assert!(s.must_serve(&stop(2, 200.0), 70.0));
        s.wait_stops.clear();
        // the trip's terminus, and the last stop it knows of with the route complete
        assert!(s.must_serve(&stop(3, 300.0), 300.0));
        s.stops = vec![stop(2, 200.0)].into();
        assert!(s.must_serve(&stop(2, 200.0), 200.0));
        s.route_open = true;
        assert!(!s.must_serve(&stop(2, 200.0), 200.0));
        // its first stop, where it stands out its layover
        s.layover = true;
        assert!(s.must_serve(&stop(2, 200.0), 200.0));
        s.layover = false;
        // `[profile_otherstopping]` 1/4 (always) and 3 (when early)
        s.always = vec![2];
        assert!(s.must_serve(&stop(2, 200.0), 200.0));
        s.always.clear();
        s.serve_early = vec![2];
        assert!(!s.must_serve(&stop(2, 200.0), 190.0));
        assert!(s.must_serve(&stop(2, 200.0), 170.0));
    }

    /// A bus that has just arrived at its first stop 200 s before its departure (the layover).
    fn layover_bus(roadside: bool, queue_behind_secs: f32) -> (BusService, Ctx<'static>) {
        let net: &'static Network = Box::leak(Box::default());
        let stop = Stop { ri: 0, s: 0.0, bay: 0.0, depart: 300.0, id: 7, side: 0.0 };
        let mut s = BusService::new(vec![stop]);
        s.layover = true;
        let ctx = Ctx {
            net,
            way: &[],
            day_time: 100.0,
            dt: 0.1,
            id: 1,
            stopped: 0.0,
            passing: false,
            lead_close: false,
            follower_close: false,
            wanted: None,
            debug: false,
            roadside,
            queue_behind_secs,
            v_nom: 13.9,
            people_hold: false,
            close_behind: false,
            covers_through: roadside,
        };
        s.arrive(&ctx, 300.0, 7, (0, 0.0));
        (s, ctx)
    }

    #[test]
    fn a_layover_off_the_road_waits_with_its_doors_shut() {
        let (mut s, mut ctx) = layover_bus(false, 0.0);
        assert_eq!(s.phase, Phase::Waiting);
        assert!(!s.boarded);
        assert!(!s.at_station());
        assert_eq!(s.leave_at, 300.0);
        // a queue behind it changes nothing: it is out of the way
        ctx.queue_behind_secs = 60.0;
        s.wait(&ctx);
        assert_eq!(s.phase, Phase::Waiting);
        assert_eq!(s.leave_at, 300.0);
        // the doors open 45 s before the departure
        ctx.day_time = 254.0;
        s.wait(&ctx);
        assert_eq!(s.phase, Phase::Waiting);
        ctx.day_time = 255.0;
        s.wait(&ctx);
        assert_eq!(s.phase, Phase::Boarding);
    }

    #[test]
    fn a_layover_on_a_through_lane_opens_its_doors_at_once() {
        let (mut s, mut ctx) = layover_bus(true, 0.0);
        assert_eq!(s.phase, Phase::Boarding);
        assert!(s.boarded);
        assert!(s.at_station());
        // it leaves on time, not before
        assert_eq!(s.leave_at, 300.0);
        ctx.day_time = 200.0;
        s.board(&ctx);
        assert_eq!(s.phase, Phase::Boarding);
        ctx.day_time = 300.0;
        s.boarding = 0.0;
        s.board(&ctx);
        assert_eq!(s.phase, Phase::Closing);
    }

    #[test]
    fn a_layover_on_a_through_lane_leaves_early_for_a_queue_behind_it() {
        // a short queue: it stays
        let (mut s, mut ctx) = layover_bus(true, 0.0);
        ctx.queue_behind_secs = 10.0;
        s.board(&ctx);
        assert_eq!(s.leave_at, 300.0);
        // a long one: the departure is now, and once the people are in the doors close
        ctx.queue_behind_secs = 25.0;
        s.board(&ctx);
        assert_eq!(s.leave_at, ctx.day_time);
        assert_eq!(s.phase, Phase::Boarding);
        s.boarding = 0.0;
        s.board(&ctx);
        assert_eq!(s.phase, Phase::Closing);
        // with the departure under 30 s away it just waits it out
        let (mut s, mut ctx) = layover_bus(true, 25.0);
        ctx.day_time = 275.0;
        s.board(&ctx);
        assert_eq!(s.leave_at, 300.0);
    }

    #[test]
    fn a_later_stop_is_not_a_roadside_layover() {
        let (mut s, ctx) = layover_bus(true, 25.0);
        assert!(s.roadside_layover);
        let stop = Stop { ri: 0, s: 0.0, bay: 0.0, depart: 500.0, id: 8, side: 0.0 };
        s.restart(vec![stop], false);
        s.wait_stops = vec![8];
        s.arrive(&ctx, 500.0, 8, (0, 0.0));
        assert!(!s.roadside_layover);
        s.board(&ctx);
        // 20 s before its time, as any stop where the timetable holds a bus
        assert_eq!(s.leave_at, 480.0);
    }

    /// [ROLLBACK r4d-B2] A ring road of four quarter lanes (0..3, 100 m each, clockwise) and
    /// a road on from the end of lane 0 (lane 4): a block to go round.
    fn ring() -> Network {
        ring_of(400.0)
    }

    fn ring_of(around: f64) -> Network {
        use omsi_sim::traffic::LaneBuilder;
        let r = around / std::f64::consts::TAU;
        let arc = |q: usize| {
            let pts = (0..=24)
                .map(|k| {
                    let a = (q as f64 + k as f64 / 24.0) * std::f64::consts::FRAC_PI_2;
                    glam::DVec3::new(r * a.sin(), r * a.cos(), 0.0)
                })
                .collect();
            LaneBuilder::polyline(pts, LaneKind::Street, 3.5)
        };
        let end = glam::DVec3::new(r, 0.0, 0.0);
        let on = LaneBuilder::polyline(vec![end, end + glam::DVec3::new(0.0, -200.0, 0.0)], LaneKind::Street, 3.5);
        let mut net = Network { lanes: vec![arc(0), arc(1), arc(2), arc(3), on], ..Default::default() };
        net.link(1.5);
        net
    }

    fn test_vehicle() -> VehicleInstance {
        let program = omsi_script::compile(&omsi_script::CompileInput::default());
        let ty = std::sync::Arc::new(omsi_sim::VehicleType {
            def: Default::default(),
            model: Default::default(),
            model_dir: std::path::PathBuf::new(),
            program: std::sync::Arc::new(program),
            meshes: Vec::new(),
            paint_schemes: Vec::new(),
            texchanges: Vec::new(),
            wheel_meshes: Vec::new(),
            suspension_axles: Vec::new(),
            missing_packs: Vec::new(),
            mesh_bounds: Vec::new(),
            mesh_boxes: Vec::new(),
        });
        VehicleInstance::new(ty, omsi_sim::VehicleHost::new(Default::default()))
    }

    fn ctx_on<'a>(net: &'a Network, way: &'a [(usize, f32)], day_time: f64, stopped: f32) -> Ctx<'a> {
        Ctx { net, way, day_time, dt: 0.1, id: 1, stopped, passing: false, lead_close: false, follower_close: false, wanted: None, debug: false, roadside: true, queue_behind_secs: 0.0, v_nom: 13.9, people_hold: false, close_behind: false, covers_through: true }
    }

    /// The bus driven along its way by `AiState::drive` for `secs` from `t`, its service
    /// stepped every 0.1 s; returns the time of day then. `each` sees every step and ends
    /// the drive by answering true.
    fn drive(s: &mut BusService, st: &mut AiState, v: &mut VehicleInstance, net: &Network, mut t: f64, secs: f64, mut each: impl FnMut(f64, &BusService, &AiState) -> bool) -> f64 {
        let end = t + secs;
        let mut stopped = 0.0f32;
        while t < end {
            let way = omsi_traffic::perception::way_lanes(net, st, 200.0);
            let mut ctx = ctx_on(net, &way, t, stopped);
            ctx.roadside = s.stops.front().is_some_and(|x| x.bay.abs() < 0.3);
            ctx.covers_through = ctx.roadside;
            let stop = s.step(st, v, &ctx);
            st.desire = s.pace;
            assert!(st.drive(net, 0.1, None, stop), "still on the road");
            stopped = if st.speed < 0.3 { stopped + 0.1 } else { 0.0 };
            t += 0.1;
            if each(t, s, st) {
                break;
            }
        }
        t
    }

    #[test]
    fn a_layover_across_a_through_lane_with_a_queue_goes_round_the_block_and_is_back_for_its_departure() {
        let net = ring();
        assert_eq!(net.lanes[0].next.len(), 2, "the ring and the road on: {:?}", net.lanes[0].next);
        let mut v = test_vehicle();
        let mut st = AiState::new(0, 50.0, 1);
        st.front = 6.0;
        st.rear = 6.0;
        st.route = vec![0, 4];
        let stops = vec![Stop { ri: 0, s: 50.0, bay: 0.0, depart: 400.0, id: 7, side: 0.0 }, Stop { ri: 1, s: 150.0, bay: 0.0, depart: 600.0, id: 8, side: 0.0 }];
        let mut s = BusService::new(stops);
        s.layover = true;
        let ctx = ctx_on(&net, &[], 100.0, 0.0);
        s.arrive(&ctx, 400.0, 7, (0, 50.0));
        assert_eq!(s.phase, Phase::Boarding);
        // in the lane, four cars queued behind it: it looks for a way off
        assert!(covers_through_lane(&net, 0, st.lateral, 1.25, true));
        assert!(queue_asks_to_go(5.0, 4) && queue_asks_to_go(25.0, 0) && !queue_asks_to_go(5.0, 3));
        let left = s.detour_wanted(0.0, 100.0, false).expect("a long wait");
        assert!((left - 300.0).abs() < 1e-6);
        let d = plan_detour(&net, 0, 50.0, 50.0, left, &|_| None).expect("round the ring");
        let Detour::Loop { route, len } = d.clone() else { panic!("{d:?}") };
        assert_eq!(route, vec![0, 1, 2, 3, 0]);
        assert!((len - 400.0).abs() < 2.0, "{len}");
        s.start_detour(&mut st, &net, d, 100.0, 1);
        // (the people who just got on are in first: the doors shut after its boarding time)
        assert_eq!(s.phase, Phase::Boarding);
        // the doors shut, round the block, back at its stop before its departure
        let mut left_at = None;
        let back = drive(&mut s, &mut st, &mut v, &net, 100.0, 300.0, |t, s, _| {
            if s.phase == Phase::Running && left_at.is_none() {
                left_at = Some(t);
            }
            left_at.is_some() && s.at_stop()
        });
        assert!(left_at.is_some_and(|t| t < 115.0), "{left_at:?}");
        assert!(back < 400.0 - 30.0, "back at {back}");
        assert_eq!((st.lane, st.route_index), (0, 4));
        assert!((st.s - 50.0).abs() < 2.5, "at its stop: {}", st.s);
        assert_eq!(s.stops.front().map(|x| (x.id, x.ri)), Some((7, 4)));
        // it waits for its departure there, and goes on time
        assert_eq!(s.leave_at, 400.0);
        let gone = drive(&mut s, &mut st, &mut v, &net, back, 400.0, |_, s, _| s.phase == Phase::Running);
        assert!((400.0..410.0).contains(&gone), "off at {gone}");
        assert_eq!(s.stops.front().map(|x| (x.id, x.ri)), Some((8, 5)));
        assert_eq!(st.route, vec![0, 1, 2, 3, 0, 4]);

        // the same layover moved into a bay off the lane: nothing to go round for
        assert!(!covers_through_lane(&net, 0, 3.2, 1.25, true));
        // a "loop" of two lane pieces linked into each other is none
        let mut tiny = ring_of(400.0);
        tiny.lanes[0].next.push(5);
        tiny.lanes.push(omsi_sim::traffic::LaneBuilder::polyline(vec![tiny.lanes[0].end(), tiny.lanes[0].end() + glam::DVec3::new(0.5, 0.0, 0.0)], LaneKind::Street, 3.5));
        tiny.lanes[5].next = vec![0];
        let Some(Detour::Loop { route, .. }) = plan_detour(&tiny, 0, 50.0, 50.0, 300.0, &|_| None) else { panic!("the ring") };
        assert_eq!(route, vec![0, 1, 2, 3, 0], "round the ring, not the 0.5 m piece");
        // nor with too little time for the loop (67 s at 6 m/s, and a minute to spare)
        assert!(plan_detour(&net, 0, 50.0, 50.0, 120.0, &|_| None).is_none());
    }

    #[test]
    fn a_layover_with_no_loop_parks_where_there_is_room_and_drives_back_in_time() {
        // a ring of 1600 m (no loop under 1200 m), the bus 300 m into lane 0; lane 4 (no
        // random traffic) has room 150 m in, and a way back onto the ring (lane 1)
        let mut net = ring_of(1600.0);
        net.lanes[4].density = 0.0;
        net.lanes[4].next = vec![1];
        assert!(plan_detour(&net, 0, 300.0, 300.0, 600.0, &|_| None).is_none(), "the loop is too long");
        let room = |l: usize| (l == 4).then_some((150.0, 0.0));
        let d = plan_detour(&net, 0, 300.0, 300.0, 600.0, &room).expect("parks on lane 4");
        let Detour::Park { route, s: ps, back, .. } = d.clone() else { panic!("{d:?}") };
        assert_eq!((route, ps, back), (vec![0, 4], 150.0, vec![4, 1, 2, 3, 0]));
        // without the time to drive there and back, it stays
        assert!(plan_detour(&net, 0, 300.0, 300.0, 300.0, &room).is_none());
        let mut v = test_vehicle();
        let mut st = AiState::new(0, 300.0, 1);
        st.front = 6.0;
        st.rear = 6.0;
        st.route = vec![0, 1];
        let stops = vec![Stop { ri: 0, s: 300.0, bay: 0.0, depart: 700.0, id: 7, side: 0.0 }, Stop { ri: 1, s: 50.0, bay: 0.0, depart: 800.0, id: 8, side: 0.0 }];
        let mut s = BusService::new(stops);
        s.layover = true;
        let ctx = ctx_on(&net, &[], 100.0, 0.0);
        s.arrive(&ctx, 700.0, 7, (0, 300.0));
        s.start_detour(&mut st, &net, d, 100.0, 1);
        // there and standing; then back at the stop before the departure
        let (mut parked_at, mut set_off) = (None, None);
        let back = drive(&mut s, &mut st, &mut v, &net, 100.0, 600.0, |t, s, st| {
            if st.lane == 4 && st.speed < 0.05 && (st.s - 150.0).abs() < 2.0 && parked_at.is_none() {
                parked_at = Some(t);
            }
            if parked_at.is_some() && set_off.is_none() && st.speed > 0.5 {
                set_off = Some(t);
            }
            parked_at.is_some() && s.at_stop()
        });
        assert!(parked_at.is_some_and(|t| t < 150.0), "{parked_at:?}");
        // it stands there until the way back (1550 m at 6 m/s) and half a minute are left
        assert!(set_off.is_some_and(|t| (410.0..416.0).contains(&t)), "{set_off:?}");
        assert!(back < 700.0, "back at {back}");
        assert_eq!(st.lane, 0);
        assert_eq!(s.stops.front().map(|x| (x.id, x.ri)), Some((7, 4)));
        assert_eq!(s.leave_at, 700.0);
    }

    #[test]
    fn a_lead_is_taken_up_at_no_less_than_six_tenths_of_the_usual_speed() {
        // 60 s early, 2 km at 13.9 m/s (144 s): 0.7 of the usual speed
        let p = pace_for(60.0, 2000.0, 13.9);
        assert!((p - 144.0 / 204.0).abs() < 0.01, "{p}");
        // a lead too long for the way: no slower than 0.6
        assert_eq!(pace_for(600.0, 2000.0, 13.9), PACE_MIN);
        // on time: the usual speed
        assert_eq!(pace_for(0.0, 2000.0, 13.9), 1.0);
        // a lead from the last stop of a trip is not carried into the next trip
        let mut s = BusService::new(vec![]);
        s.ahead_by = 60.0;
        s.pace = 0.7;
        s.restart(vec![], true);
        assert_eq!((s.ahead_by, s.pace), (0.0, 1.0));
    }

    /// [ROLLBACK r4d-B3] A roadside layover that starts 60 s early for the queue behind it
    /// makes the lead up over the next 2 km to its next stop and is there on time, not a
    /// minute early.
    #[test]
    fn a_bus_that_leaves_a_minute_early_makes_it_up_over_the_next_two_kilometres() {
        use omsi_sim::traffic::LaneBuilder;
        let l = |a: f64, b: f64| LaneBuilder::polyline(vec![glam::DVec3::new(0.0, a, 0.0), glam::DVec3::new(0.0, b, 0.0)], LaneKind::Street, 3.5);
        let mut net = Network { lanes: vec![l(0.0, 100.0), l(100.0, 2100.0), l(2100.0, 2400.0)], ..Default::default() };
        net.link(1.5);
        let run = |queue: f32, bay: f32, at: f32| {
            let mut v = test_vehicle();
            let mut st = AiState::new(0, 50.0, 1);
            st.front = 6.0;
            st.rear = 6.0;
            st.route = vec![0, 1, 2];
            // the next stop 2 km on, timed at 150 s after the first
            let stops = vec![Stop { ri: 0, s: 50.0, bay: 0.0, depart: 160.0, id: 7, side: 0.0 }, Stop { ri: 1, s: at, bay, depart: 310.0, id: 8, side: 0.0 }, Stop { ri: 2, s: 200.0, bay: 0.0, depart: 400.0, id: 9, side: 0.0 }];
            let mut s = BusService::new(stops);
            s.layover = true;
            let mut ctx = ctx_on(&net, &[], 100.0, 0.0);
            s.arrive(&ctx, 160.0, 7, (0, 50.0));
            ctx.queue_behind_secs = queue;
            s.board(&ctx);
            let (mut off, mut there, mut slowest) = (None, None, 1.0f32);
            let t = drive(&mut s, &mut st, &mut v, &net, 100.0, 400.0, |t, s, _| {
                if off.is_none() && s.phase == Phase::Running {
                    off = Some(t);
                }
                if off.is_some() && there.is_none() && s.at_stop() {
                    there = Some(t);
                }
                slowest = slowest.min(s.pace);
                there.is_some() && s.phase == Phase::Running
            });
            (off.unwrap(), there.unwrap(), t, slowest, s.ahead_by)
        };
        // a queue: off a minute early, at about 0.7 of its speed, at the next stop on time
        // (the stop 2 km on in a bay, out of the way)
        let (off, there, gone, slowest, left) = run(25.0, 2.0, 1950.0);
        assert!(left < 1.0, "made up: {left}");
        // (once the people are in)
        assert!(off < 115.0, "off at {off}");
        assert!((0.6..0.8).contains(&slowest), "{slowest}");
        assert!((295.0..312.0).contains(&there), "there at {there}");
        assert!(gone >= 305.0, "on from there at {gone}");
        // no queue: it leaves on time and drives at its usual speed
        let (off, there, _, slowest, _) = run(0.0, 2.0, 1950.0);
        assert!(off >= 160.0, "off at {off}");
        assert_eq!(slowest, 1.0);
        assert!(there < 340.0, "{there}");
        // the next stop 300 m on, too near to make the minute up: in a bay the rest is
        // waited out there; in the lane, where it would hold the traffic up again, it is not
        // (but made up on the way on)
        let (_, _, gone, _, left) = run(25.0, 2.0, 250.0);
        assert!(left == 0.0 && gone > 175.0, "waited out: {left}, on at {gone}");
        let (_, _, gone, _, left) = run(25.0, 0.0, 250.0);
        assert!(left > 20.0 && gone < 170.0, "not waited out in the lane: {left}, on at {gone}");
    }

    /// [ROLLBACK r4d-B4] Held short of its pole by somebody in its way, a bus does not serve
    /// the stop where it stands after 6 s (as behind another bus) but creeps up to the pole
    /// once they are clear; only after `PEOPLE_SERVE_MAX` does it serve it where it is. Off
    /// the bay again it is back in the middle of its lane.
    #[test]
    fn a_bus_held_short_by_people_creeps_up_to_its_pole_and_leaves_the_bay_for_the_lane() {
        use omsi_sim::traffic::LaneBuilder;
        let mut net = Network { lanes: vec![LaneBuilder::polyline(vec![glam::DVec3::ZERO, glam::DVec3::new(0.0, 300.0, 0.0)], LaneKind::Street, 3.5)], ..Default::default() };
        net.link(1.5);
        let mut v = test_vehicle();
        let stop = || Stop { ri: 0, s: 100.0, bay: 1.2, depart: 500.0, id: 7, side: 0.0 };
        let at = |s: f32| {
            let mut st = AiState::new(0, s, 1);
            st.front = 6.0;
            st.rear = 6.0;
            st.route = vec![0];
            st
        };
        // 8 m short, standing 7 s, somebody in its way: not served here
        let mut s = BusService::new(vec![stop(), Stop { id: 8, s: 250.0, ..stop() }]);
        let mut st = at(92.0);
        let way = [(0usize, -92.0f32)];
        let mut ctx = ctx_on(&net, &way, 100.0, 7.0);
        ctx.people_hold = true;
        let target = s.step(&mut st, &mut v, &ctx).expect("its stop");
        assert_eq!(s.phase, Phase::Running);
        assert!((target - (8.0 + 6.0 - 0.3)).abs() < 0.01, "{target}");
        // [round 7 A7] into its stop it asks for its bay
        assert_eq!(s.bay_wish, Some(1.2));
        assert_eq!(st.lateral_target, 1.2);
        // [round 7 A7] far from it (beyond BAY_REACH) no bay, and its lateral target is the
        // traffic's: not written by the service
        let mut far = BusService::new(vec![stop()]);
        let mut st_far = at(20.0);
        st_far.lateral_target = -0.4;
        let way_far = [(0usize, -20.0f32)];
        far.step(&mut st_far, &mut v, &ctx_on(&net, &way_far, 100.0, 0.0));
        assert_eq!(far.bay_wish, None);
        assert_eq!(st_far.lateral_target, -0.4);
        // clear now, still standing (a creep does not reset the time): it goes on to the pole
        ctx.people_hold = false;
        ctx.stopped = 9.0;
        s.step(&mut st, &mut v, &ctx);
        assert_eq!(s.phase, Phase::Running);
        // there: served
        let mut st = at(99.0);
        s.step(&mut st, &mut v, &ctx);
        assert_eq!(s.phase, Phase::Boarding);
        // held for good: served where it stands after half a minute
        let mut s = BusService::new(vec![stop()]);
        let mut st = at(92.0);
        ctx.people_hold = true;
        s.step(&mut st, &mut v, &ctx);
        assert_eq!(s.phase, Phase::Running);
        ctx.stopped = PEOPLE_SERVE_MAX + 1.0;
        s.step(&mut st, &mut v, &ctx);
        assert_eq!(s.phase, Phase::Boarding);
        // [P3-1] coming to its layover (its first stop) and held short by its own waiters: served
        // where it stands after half a minute too, not queued for the berth for good (d24 #194)
        let mut s = BusService::new(vec![stop()]);
        s.layover = true;
        let mut st = at(92.0);
        ctx.people_hold = true;
        ctx.stopped = 7.0;
        s.step(&mut st, &mut v, &ctx);
        assert_eq!(s.phase, Phase::Running);
        ctx.stopped = PEOPLE_SERVE_MAX + 1.0;
        s.step(&mut st, &mut v, &ctx);
        assert_eq!(s.phase, Phase::Boarding);
        // the counter-case: held by a vehicle ahead, served where it stands after 6 s
        let mut s = BusService::new(vec![stop()]);
        let mut st = at(92.0);
        ctx.people_hold = false;
        ctx.stopped = 7.0;
        s.step(&mut st, &mut v, &ctx);
        assert_eq!(s.phase, Phase::Boarding);

        // in the bay at its pole; off again it is back in the middle of the lane
        let mut s = BusService::new(vec![stop(), Stop { id: 8, s: 280.0, ..stop() }]);
        let mut st = at(100.0);
        st.lateral = 1.2;
        st.lateral_target = 1.2;
        st.lateral_ramp = (1.2, 1.2, 0.0, 1.0);
        let ctx = ctx_on(&net, &way, 100.0, 0.0);
        s.arrive(&ctx, 500.0, 7, (0, 100.0));
        s.leave_at = 100.0;
        let mut lateral_at = Vec::new();
        drive(&mut s, &mut st, &mut v, &net, 100.0, 60.0, |_, _, st| {
            lateral_at.push((st.s, st.lateral));
            st.s > 160.0
        });
        assert_eq!(st.lateral_target, 0.0);
        assert_eq!(s.bay_wish, None, "between stops it asks for no bay");
        assert!(lateral_at.iter().any(|&(s, l)| s < 101.0 && l > 1.1), "starts in the bay");
        assert!(st.lateral.abs() < 0.05, "back in the lane: {}", st.lateral);
    }

    /// [ROLLBACK r4d-B5] Two buses, one stop: the first, knowing the second behind it,
    /// pulls on by its length past the pole (as far as the road lets it) and serves the stop
    /// there; the second comes up close behind it and serves the stop where it stands. A
    /// first one standing out its layover with the doors shut pulls on as well.
    #[test]
    fn two_buses_at_one_stop_the_first_pulls_on_and_the_second_serves_behind_it() {
        use omsi_sim::traffic::LaneBuilder;
        let straight = |len: f64| {
            let mut net = Network { lanes: vec![LaneBuilder::polyline(vec![glam::DVec3::ZERO, glam::DVec3::new(0.0, len, 0.0)], LaneKind::Street, 3.5)], ..Default::default() };
            net.link(1.5);
            net
        };
        let stop = Stop { ri: 0, s: 100.0, bay: 0.0, depart: 500.0, id: 7, side: 0.0 };
        let bus_at = |s: f32, speed: f32| {
            let mut st = AiState::new(0, s, 1);
            st.front = 6.0;
            st.rear = 6.0;
            st.route = vec![0];
            st.speed = speed;
            st
        };
        // the first, a follower behind it: where does it come to rest
        let first = |net: &Network, follower: bool, bay: f32| {
            let mut v = test_vehicle();
            let mut s = BusService::new(vec![Stop { bay, ..stop }, Stop { id: 8, s: 290.0, ..stop }]);
            let mut st = bus_at(40.0, 10.0);
            let mut t = 100.0;
            let mut stopped = 0.0;
            while t < 200.0 && !s.at_stop() {
                let way = omsi_traffic::perception::way_lanes(net, &st, 200.0);
                let mut ctx = ctx_on(net, &way, t, stopped);
                ctx.follower_close = follower;
                let target = s.step(&mut st, &mut v, &ctx);
                st.drive(net, 0.1, None, target);
                stopped = if st.speed < 0.3 { stopped + 0.1 } else { 0.0 };
                t += 0.1;
            }
            assert!(s.at_stop(), "served, not missed");
            st.s
        };
        let net = straight(300.0);
        let alone = first(&net, false, 0.0);
        assert!((alone - 100.0).abs() < 1.5, "at the pole: {alone}");
        let pulled = first(&net, true, 0.0);
        assert!((pulled - 112.0).abs() < 1.5, "a bus length on: {pulled}");
        // the road ends 6 m past the pole: only 4 m on
        let net = straight(112.0);
        let short = first(&net, true, 0.0);
        assert!((short - 104.0).abs() < 1.5, "as far as the road lets it: {short}");
        // a zebra 8 m past the pole (its front there at 106 m, the crossing at 114 m): it stops
        // 2 m short of the crossing's edge, not on it
        let mut net = straight(300.0);
        net.walks = vec![vec![(9, 114.0, 0.0)]];
        let zebra = first(&net, true, 0.0);
        assert!((zebra - 104.0).abs() < 1.5 && zebra + 6.0 <= 112.0 - 1.0, "off the zebra: {zebra}");
        // the mouth of a junction on its own lane 6 m past the pole: not into it
        let mut net = straight(300.0);
        net.crossings = vec![vec![omsi_sim::traffic::Crossing { other: 0, at: 113.0, other_at: 0.0, merge: false, before: 1.0, after: 1.0, other_before: 1.0, other_after: 1.0, side: false }]];
        let mouth = first(&net, true, 0.0);
        assert!((mouth - 104.0).abs() < 1.5, "short of the junction: {mouth}");
        // in a bay: no further on than the bay is known to reach
        let net = straight(300.0);
        let bay = first(&net, true, 2.0);
        assert!((bay - 104.0).abs() < 1.5, "within the bay: {bay}");

        // the second: close behind the first, standing a second at the pole's approach - it
        // serves the stop there; without the first ahead it drives on to the pole
        let net = straight(300.0);
        let mut v = test_vehicle();
        let way = omsi_traffic::perception::way_lanes(&net, &bus_at(86.0, 0.0), 200.0);
        let mut ctx = ctx_on(&net, &way, 100.0, 1.0);
        let mut s = BusService::new(vec![stop]);
        let mut st = bus_at(86.0, 0.0);
        s.step(&mut st, &mut v, &ctx);
        assert_eq!(s.phase, Phase::Running);
        ctx.lead_close = true;
        s.step(&mut st, &mut v, &ctx);
        assert_eq!(s.phase, Phase::Boarding);

        // a layover with the doors shut: a bus behind for its stop - it pulls on its length
        let mut s = BusService::new(vec![Stop { depart: 900.0, ..stop }]);
        s.layover = true;
        let mut st = bus_at(100.0, 0.0);
        let mut ctx = ctx_on(&net, &[], 100.0, 0.0);
        ctx.roadside = false;
        s.arrive(&ctx, 900.0, 7, (0, 100.0));
        assert_eq!(s.phase, Phase::Waiting);
        let mut t = 100.0;
        while t < 160.0 {
            let way = omsi_traffic::perception::way_lanes(&net, &st, 200.0);
            let mut ctx = ctx_on(&net, &way, t, 0.0);
            ctx.roadside = false;
            ctx.follower_close = t > 110.0;
            let target = s.step(&mut st, &mut v, &ctx);
            st.drive(&net, 0.1, None, target);
            t += 0.1;
        }
        assert_eq!(s.phase, Phase::Waiting);
        assert!((st.s - 112.0).abs() < 1.0 && st.speed < 0.1, "pulled on: {} at {}", st.s, st.speed);
    }

    /// [ROLLBACK r4d-B2] Only a layover (or a queue for an occupied berth) looks for a way off:
    /// not the wait at a stop on the way where the timetable holds the bus.
    #[test]
    fn only_a_layover_or_a_queue_for_the_berth_looks_for_a_way_off() {
        let (s, _) = layover_bus(true, 0.0);
        assert!(s.detour_wanted(0.0, 100.0, false).is_some(), "its layover");
        // a wait stop on the way, as long a wait
        let (mut s, ctx) = layover_bus(true, 0.0);
        s.restart(vec![Stop { ri: 0, s: 0.0, bay: 0.0, depart: 300.0, id: 8, side: 0.0 }], false);
        s.wait_stops = vec![8];
        s.arrive(&ctx, 300.0, 8, (0, 0.0));
        assert!(s.leave_at - ctx.day_time > 90.0);
        assert_eq!(s.detour_wanted(0.0, 100.0, false), None);
        // driving: only standing in the queue for the berth
        let mut s = BusService::new(vec![Stop { ri: 0, s: 50.0, bay: 0.0, depart: 300.0, id: 9, side: 0.0 }]);
        assert_eq!(s.detour_wanted(0.0, 100.0, false), None);
        assert!(s.detour_wanted(0.0, 100.0, true).is_some());
        assert_eq!(s.detour_wanted(5.0, 100.0, true), None, "moving");
        // looked a moment ago: not again for half a minute
        s.no_detour(100.0, 1);
        assert_eq!(s.detour_wanted(0.0, 110.0, true), None);
        assert!(s.detour_wanted(0.0, 131.0, true).is_some());
    }

    /// [ROLLBACK r4d-B2] A layover sent round the block while people get on keeps its doors
    /// open for them: it leaves only once nobody holds them any more.
    #[test]
    fn a_layover_sent_round_the_block_waits_for_the_people_at_its_doors() {
        let net = ring();
        let mut v = test_vehicle();
        let mut st = AiState::new(0, 50.0, 1);
        st.front = 6.0;
        st.rear = 6.0;
        st.route = vec![0, 4];
        let mut s = BusService::new(vec![Stop { ri: 0, s: 50.0, bay: 0.0, depart: 400.0, id: 7, side: 0.0 }]);
        s.layover = true;
        let ctx = ctx_on(&net, &[], 100.0, 0.0);
        s.arrive(&ctx, 400.0, 7, (0, 50.0));
        assert_eq!(s.phase, Phase::Boarding);
        let d = plan_detour(&net, 0, 50.0, 50.0, 300.0, &|_| None).unwrap();
        s.start_detour(&mut st, &net, d, 100.0, 1);
        assert_eq!(s.phase, Phase::Boarding, "the doors stay open");
        // somebody at the doors for 15 s
        let mut closed_at = None;
        drive(&mut s, &mut st, &mut v, &net, 100.0, 60.0, |t, s, _| {
            if s.phase == Phase::Closing && closed_at.is_none() {
                closed_at = Some(t);
            }
            s.phase == Phase::Running
        });
        // (`drive` cannot hold the doors: do it by hand)
        let mut s = BusService::new(vec![Stop { ri: 0, s: 50.0, bay: 0.0, depart: 400.0, id: 7, side: 0.0 }]);
        s.layover = true;
        let mut st = AiState::new(0, 50.0, 1);
        st.route = vec![0, 4];
        s.arrive(&ctx, 400.0, 7, (0, 50.0));
        let d = plan_detour(&net, 0, 50.0, 50.0, 300.0, &|_| None).unwrap();
        s.start_detour(&mut st, &net, d, 100.0, 1);
        let mut t = 100.0;
        let mut closed = None;
        while t < 160.0 {
            if t < 115.0 {
                s.hold(Some(7), 2.0);
            }
            let ctx = ctx_on(&net, &[], t, 0.0);
            s.step(&mut st, &mut v, &ctx);
            if s.phase != Phase::Boarding && closed.is_none() {
                closed = Some(t);
            }
            t += 0.1;
        }
        let closed = closed.expect("it goes in the end");
        assert!(closed >= 115.0 && closed < 118.0, "doors shut once nobody holds them: {closed}");
        assert!(closed_at.is_some_and(|t| t < 115.0), "nobody at the doors: off once boarding is over");
        assert_eq!(st.route, vec![0, 1, 2, 3, 0, 4], "and round the block");
    }

    /// [ROLLBACK r4d-B2] The through lanes are the lanes the traffic drives, not those of a
    /// density: a density-0 lane the random cars use is one, a density-1 lane nobody drives
    /// is not; another bus's route makes one, the bus's own does not; so does a lane beside.
    #[test]
    fn a_through_lane_is_one_the_traffic_drives() {
        use omsi_sim::traffic::LaneBuilder;
        let l = |x: f64| LaneBuilder::polyline(vec![glam::DVec3::new(x, 0.0, 0.0), glam::DVec3::new(x, 100.0, 0.0)], LaneKind::Street, 3.5);
        let mut net = Network { lanes: vec![l(0.0), l(10.0), l(20.0), l(30.0)], ..Default::default() };
        net.lanes[0].density = 0.0;
        net.lanes[3].left = Some(2);
        let mut t = ThroughLanes::default();
        t.driven.insert(0);
        t.routes.insert(1, vec![5]);
        t.routes.insert(2, vec![5, 6]);
        assert!(t.through(&net, 0, 5), "density 0, driven by cars");
        assert!(!t.through(&net, 1, 5), "only its own route");
        assert!(t.through(&net, 1, 6), "another bus's route");
        assert!(t.through(&net, 2, 5));
        assert!(t.through(&net, 3, 5), "a lane beside it");
        assert!(t.used(1) && !t.used(3));
        // a bus in the middle of a driven lane covers it; moved 3.2 m over into a bay it does not
        assert!(covers_through_lane(&net, 0, 0.0, 1.25, true));
        assert!(!covers_through_lane(&net, 0, 3.2, 1.25, true));
        assert!(!covers_through_lane(&net, 0, 0.0, 1.25, false));
    }

    /// [ROLLBACK r4d-B2] Where a 12 m bus stands out of the way on a lane: another stop's bay,
    /// the lane itself only when no vehicle's way uses it, the kerb of a parked row (a gap,
    /// before it, after it) - never on a lane in use, never over a crossing, never where a
    /// vehicle is.
    #[test]
    fn a_bus_parks_in_a_bay_on_an_unused_lane_or_at_the_kerb_of_a_parked_row() {
        let none = |_: f32, _: f32, _: f32| true;
        // (a town car's box, the one every parked car had until round 7)
        let car_at = |s: f32, lateral: f32| omsi_traffic::Parked { s, lateral, half_width: 0.9, half_len: 2.3 };
        // a lane in use, nothing beside it: nowhere
        assert_eq!(room_on_lane(200.0, 3.5, 12.0, 1.25, true, &[], &[], &[], &none), None);
        // nobody's way: the lane's end
        assert_eq!(room_on_lane(200.0, 3.5, 12.0, 1.25, false, &[], &[], &[], &none), Some((190.0, 0.0)));
        // ... unless a crossing is there
        assert_eq!(room_on_lane(200.0, 3.5, 12.0, 1.25, false, &[], &[], &[195.0], &none), None);
        // another stop's bay 3.2 m over
        assert_eq!(room_on_lane(200.0, 3.5, 12.0, 1.25, true, &[(80.0, 3.2)], &[], &[], &none), Some((80.0, 3.2)));
        // a bay that leaves the bus over the lane is none
        assert_eq!(room_on_lane(200.0, 3.5, 12.0, 1.25, true, &[(80.0, 1.2)], &[], &[], &none), None);
        // a parked row 3 m over from 60 to 90 m, cars 5 m apart: the kerb before it
        let row: Vec<omsi_traffic::Parked> = (0..7).map(|k| car_at(60.0 + 5.0 * k as f32, 3.0)).collect();
        let (s, lat) = room_on_lane(200.0, 3.5, 12.0, 1.25, true, &[], &row, &[], &none).expect("the kerb before the row");
        assert!(s < 60.0 - 2.3 - 8.0 && (lat - 3.35).abs() < 0.01, "{s} {lat}");
        // [round 7 A10] a row of vans (2.2 x 6 m) 3.2 m over: the bus lines up with their
        // inner sides and keeps clear of the first van's own length
        let vans: Vec<omsi_traffic::Parked> = (0..5).map(|k| omsi_traffic::Parked { s: 60.0 + 7.0 * k as f32, lateral: 3.2, half_width: 1.1, half_len: 3.0 }).collect();
        let (s, lat) = room_on_lane(200.0, 3.5, 12.0, 1.25, true, &[], &vans, &[], &none).expect("the kerb before the vans");
        assert!(s <= 60.0 - 3.0 - 1.0 - 8.0 + 0.01 && (lat - 3.35).abs() < 0.01, "{s} {lat}");
        // the stretch before the row too short (row from 15 m): after it
        let row: Vec<omsi_traffic::Parked> = (0..7).map(|k| car_at(15.0 + 5.0 * k as f32, 3.0)).collect();
        let (s, _) = room_on_lane(200.0, 3.5, 12.0, 1.25, true, &[], &row, &[], &none).expect("after the row");
        assert!(s > 45.0 + 2.3 + 8.0, "{s}");
        // a gap of 30 m in the row (no room before it or after it on an 80 m lane): the gap,
        // unless somebody stands there
        let mut row: Vec<omsi_traffic::Parked> = (0..4).map(|k| car_at(15.0 + 5.0 * k as f32, 3.0)).collect();
        row.extend((0..4).map(|k| car_at(60.0 + 5.0 * k as f32, 3.0)));
        assert_eq!(room_on_lane(80.0, 3.5, 12.0, 1.25, true, &[], &row, &[], &none), Some((45.0, 3.35)));
        assert_eq!(room_on_lane(80.0, 3.5, 12.0, 1.25, true, &[], &row, &[], &|_, _, _| false), None);
        // a row so close to the lane that a bus there would still cover it: none
        let row: Vec<omsi_traffic::Parked> = (0..7).map(|k| car_at(60.0 + 5.0 * k as f32, 1.8)).collect();
        assert_eq!(room_on_lane(200.0, 3.5, 12.0, 1.25, true, &[], &row, &[], &none), None);
    }

    /// [ROLLBACK r4d-B3] The lead counts off by the clock: a bus 50 s early held 40 s at a red
    /// light half-way has made the rest up by then and drives on at its usual speed - late at
    /// its next stop only by what it gave away before the light it could not foresee (counted
    /// off only while moving, the lead had kept it slow after the light too, 40 s more); with a
    /// vehicle close behind it, it keeps its usual speed (the lead is waited out at the stop).
    #[test]
    fn a_lead_counts_off_while_the_bus_is_held_and_a_follower_keeps_it_at_speed() {
        use omsi_sim::traffic::LaneBuilder;
        let l = |a: f64, b: f64| LaneBuilder::polyline(vec![glam::DVec3::new(0.0, a, 0.0), glam::DVec3::new(0.0, b, 0.0)], LaneKind::Street, 3.5);
        let mut net = Network { lanes: vec![l(0.0, 100.0), l(100.0, 2100.0), l(2100.0, 2400.0)], ..Default::default() };
        net.link(1.5);
        // off at 100 s 60 s early; the next stop (in a bay) 2 km on, timed for 310 s; a red
        // light at 1 km holding the bus 40 s from when it gets there
        let run = |red: bool, follower: bool| {
            let mut v = test_vehicle();
            let mut st = AiState::new(0, 50.0, 1);
            st.front = 6.0;
            st.rear = 6.0;
            st.route = vec![0, 1, 2];
            let stops = vec![Stop { ri: 0, s: 50.0, bay: 0.0, depart: 160.0, id: 7, side: 0.0 }, Stop { ri: 1, s: 1950.0, bay: 2.0, depart: 310.0, id: 8, side: 0.0 }, Stop { ri: 2, s: 200.0, bay: 0.0, depart: 400.0, id: 9, side: 0.0 }];
            let mut s = BusService::new(stops);
            s.layover = true;
            let mut ctx = ctx_on(&net, &[], 100.0, 0.0);
            s.arrive(&ctx, 160.0, 7, (0, 50.0));
            ctx.queue_behind_secs = 25.0;
            s.board(&ctx);
            let (mut t, mut stopped, mut held_from, mut there, mut slowest_after_red) = (100.0, 0.0f32, None::<f64>, None, 1.0f32);
            while t < 500.0 && there.is_none() {
                let way = omsi_traffic::perception::way_lanes(&net, &st, 200.0);
                let mut ctx = ctx_on(&net, &way, t, stopped);
                ctx.roadside = s.stops.front().is_some_and(|x| x.bay.abs() < 0.3);
                ctx.covers_through = ctx.roadside;
                ctx.close_behind = follower;
                let mut stop = s.step(&mut st, &mut v, &ctx);
                st.desire = s.pace;
                // the red light at x = 1100 (lane 1, 1000 m in)
                let to_light = st.route_distance(&net, 1, 1000.0) - st.front;
                if red && held_from.is_none_or(|h| t < h + 40.0) && to_light > -1.0 {
                    if to_light < 1.0 && st.speed < 0.1 && held_from.is_none() {
                        held_from = Some(t);
                    }
                    let light = to_light.max(0.0) + st.front;
                    stop = Some(stop.map_or(light, |x| x.min(light)));
                }
                st.drive(&net, 0.1, None, stop);
                stopped = if st.speed < 0.3 { stopped + 0.1 } else { 0.0 };
                t += 0.1;
                if held_from.is_some_and(|h| t > h + 45.0) {
                    slowest_after_red = slowest_after_red.min(s.pace);
                }
                if s.at_stop() && s.stops.front().is_some_and(|x| x.id == 8) {
                    there = Some(t);
                }
            }
            (there.expect("at the stop"), held_from, slowest_after_red, s.leave_at)
        };
        // no light: on time, as before
        let (there, _, _, _) = run(false, false);
        assert!((295.0..312.0).contains(&there), "{there}");
        // the light: held 40 s, nothing left to make up after it
        let (there, held, slowest, _) = run(true, false);
        assert!(held.is_some(), "held at the light");
        assert_eq!(slowest, 1.0, "after the light its usual speed");
        assert!(there < 310.0 + 35.0, "there at {there}");
        // a vehicle close behind: its usual speed all the way, the lead waited out at the stop
        let (there, _, _, leave) = run(false, true);
        assert!(there < 270.0, "early at the stop: {there}");
        assert!(leave >= there + 30.0, "and waits the lead out there: {leave} (there {there})");
    }

    /// [ROLLBACK r4d-B2] A bus put back 30 m before its layover stop, the berth taken: it
    /// queues, drives up once the berth frees, waits out its layover there and departs on
    /// time.
    #[test]
    fn a_bus_put_back_before_its_taken_berth_drives_up_once_it_frees_and_departs() {
        use omsi_sim::traffic::LaneBuilder;
        let mut net = Network { lanes: vec![LaneBuilder::polyline(vec![glam::DVec3::ZERO, glam::DVec3::new(0.0, 300.0, 0.0)], LaneKind::Street, 3.5)], ..Default::default() };
        net.link(1.5);
        let mut v = test_vehicle();
        let mut st = AiState::new(0, 70.0, 1);
        st.front = 6.0;
        st.rear = 6.0;
        st.route = vec![0];
        let mut s = BusService::new(vec![Stop { ri: 0, s: 100.0, bay: 0.0, depart: 400.0, id: 9, side: 0.0 }, Stop { ri: 0, s: 280.0, bay: 0.0, depart: 500.0, id: 10, side: 0.0 }]);
        // (as `Schedule::spawn_departure` puts a respawned bus back: on its layover)
        s.layover = true;
        let (mut t, mut stopped, mut at_stop, mut off) = (300.0, 0.0f32, None, None);
        while t < 460.0 && off.is_none() {
            let way = omsi_traffic::perception::way_lanes(&net, &st, 200.0);
            let ctx = ctx_on(&net, &way, t, stopped);
            let mut stop = s.step(&mut st, &mut v, &ctx);
            // the berth taken until 330: a bus there, its tail 15 m before the stop point
            if t < 330.0 {
                let tail = st.route_distance(&net, 0, 85.0) - st.front - 2.0;
                stop = Some(stop.map_or(tail.max(0.0) + st.front, |x| x.min(tail.max(0.0) + st.front)));
            }
            st.drive(&net, 0.1, None, stop);
            stopped = if st.speed < 0.3 { stopped + 0.1 } else { 0.0 };
            t += 0.1;
            if at_stop.is_none() && s.at_stop() {
                at_stop = Some((t, st.s));
            }
            if at_stop.is_some() && s.phase == Phase::Running {
                off = Some(t);
            }
        }
        let (there, where_) = at_stop.expect("at its stop");
        assert!(there > 330.0 && (where_ - 100.0).abs() < 2.5, "drove up once the berth freed: {there} at {where_}");
        let off = off.expect("off");
        assert!((400.0..410.0).contains(&off), "on time: {off}");
    }

    #[test]
    fn station_side_comes_from_the_stop_it_boards_at() {
        let stop = |side: f32| Stop::from_tuple((0, 0.0, 0.0, 0.0, 1, side));
        let mut s = BusService::new(vec![stop(1.0)]);
        // off a stop: nothing to open
        s.phase = Phase::Running;
        assert_eq!(s.at_station_side(), 0.0);
        // boarding: the front stop's side
        s.phase = Phase::Boarding;
        assert_eq!(s.at_station_side(), 1.0);
        // waiting to pull out (doors shut): the side is not asked for any more
        s.phase = Phase::Waiting;
        assert_eq!(s.at_station_side(), 0.0);
        // an empty queue answers 0, not a panic
        s.phase = Phase::Boarding;
        s.stops.clear();
        assert_eq!(s.at_station_side(), 0.0);
    }

    /// [round 6 B1] A fan of a junction object: lane 0 runs north from the origin, lane 1
    /// leaves the same point to the north-west, lane 2 lies 10 m to the east (all pieces of
    /// object 12074), lane 3 is another object's on top of lane 1, lane 4 a spline's.
    fn fan() -> Network {
        use omsi_sim::traffic::{LaneBuilder, LaneKey};
        let p = |x: f64, y: f64| glam::DVec3::new(x, y, 0.0);
        let piece = |pts: Vec<glam::DVec3>, id: i64, path: u16, source: u32| {
            let mut l = LaneBuilder::polyline(pts, LaneKind::Street, 3.0);
            l.key = Some(LaneKey { tile: (0, 0), id, path });
            l.source = source;
            l
        };
        let mut net = Network {
            lanes: vec![
                piece(vec![p(0.0, 0.0), p(0.0, 30.0)], 12074, 0, 2),
                piece(vec![p(0.0, 0.0), p(-8.0, 30.0)], 12074, 1, 2),
                piece(vec![p(10.0, 0.0), p(10.0, 30.0)], 12074, 2, 2),
                piece(vec![p(0.0, 0.0), p(-8.0, 30.0)], 999, 0, 2),
                piece(vec![p(0.0, -40.0), p(0.0, 0.0)], 5, 0, 1),
            ],
            ..Default::default()
        };
        net.link(1.5);
        net
    }

    #[test]
    fn b1_a_layover_bus_across_a_junction_fan_covers_the_lanes_beside_its_own() {
        let net = fan();
        let all = |_: usize| true;
        // a 12 m bus 8 m into lane 0: its body lies over lane 1 where that leaves lane 0
        let at = |x: f64, y: f64| glam::DVec2::new(x, y);
        assert_eq!(fan_lanes_covered(&net, 0, at(0.0, 8.0), 0.0, 6.0, 6.0, 1.25, &all), vec![1], "lane 1 of its object; not lane 2 10 m off, nor another object's");
        // the lane only counts when the traffic drives it
        assert!(fan_lanes_covered(&net, 0, at(0.0, 8.0), 0.0, 6.0, 6.0, 1.25, &|l| l != 1).is_empty());
        // pulled 3.2 m into a bay to the right: clear of lane 1 (counter-case: it stays)
        assert!(fan_lanes_covered(&net, 0, at(3.2, 8.0), 0.0, 6.0, 6.0, 1.25, &all).is_empty());
        // further up, where lane 1 is more than half a metre off its flank: nothing
        assert!(fan_lanes_covered(&net, 0, at(0.0, 26.0), 0.0, 4.0, 4.0, 1.25, &all).is_empty());
        // on a spline's lane: the lanes beside are its road's, `covers_lane` judges them
        assert!(fan_lanes_covered(&net, 4, at(0.0, -20.0), 0.0, 6.0, 6.0, 1.25, &all).is_empty());
        // a lane of its object alongside: covered while a car (0.9 m) on its axis would come
        // within 0.5 m of the body, not when it is 2.7 m or more off the bus's axis
        let beside = |x: f64| {
            use omsi_sim::traffic::{LaneBuilder, LaneKey};
            let piece = |x0: f64, path: u16| {
                let mut l = LaneBuilder::polyline(vec![glam::DVec3::new(x0, 0.0, 0.0), glam::DVec3::new(x0, 30.0, 0.0)], LaneKind::Street, 3.0);
                l.key = Some(LaneKey { tile: (0, 0), id: 12074, path });
                l.source = 2;
                l
            };
            let mut n = Network { lanes: vec![piece(0.0, 0), piece(x, 1)], ..Default::default() };
            n.link(1.5);
            fan_lanes_covered(&n, 0, at(0.0, 15.0), 0.0, 6.0, 6.0, 1.25, &all)
        };
        assert_eq!(beside(2.5), vec![1], "0.35 m from a car on it");
        assert!(beside(2.7).is_empty() && beside(2.9).is_empty(), "a car there is 0.55 m or more clear");
    }

    #[test]
    fn b1_the_queue_at_a_covered_fan_lane_counts_who_the_bus_holds_there_and_behind() {
        let fan = [1usize];
        let o = |id: u64, lane: usize, way: Vec<(usize, f32)>, speed: f32, stood: f32, lead: Option<u64>, held_by_bus: bool| FanOther { id, lane, way, speed, stood, lead, serving: false, held_by_bus };
        let others: Vec<FanOther> = vec![
            // waiting for the bus on the covered lane 25 s, and one behind it
            o(10, 1, vec![(1, -3.0)], 0.0, 25.0, None, true),
            o(11, 7, vec![(7, -2.0), (1, 6.0)], 0.0, 12.0, Some(10), false),
            // held by it 30 m before the lane on its way, and 50 m before it (too far)
            o(12, 8, vec![(8, -5.0), (1, 30.0)], 0.1, 8.0, None, true),
            o(13, 9, vec![(9, -5.0), (1, 50.0)], 0.0, 90.0, None, true),
            // driving onto it: no queue
            o(14, 9, vec![(9, -5.0), (1, 10.0)], 4.0, 0.0, None, true),
            // another bus at its own berth on the covered lane: waits for nobody
            FanOther { serving: true, ..o(15, 1, vec![(1, -12.0)], 0.0, 300.0, None, true) },
            // standing on the covered lane 60 s for a light, not for the bus (b1c-after bus 252)
            o(16, 1, vec![(1, -20.0)], 0.0, 60.0, None, false),
            o(17, 6, vec![(6, -2.0), (1, 4.0)], 0.0, 60.0, Some(16), false),
        ];
        let (secs, n) = fan_queue(&fan, &others);
        assert_eq!(n, 3, "10, 11 and 12");
        assert!((secs - 25.0).abs() < 1e-6, "{secs}");
        assert!(queue_asks_to_go(secs, n), "past LAYOVER_QUEUE");
        // nobody it holds at a covered lane: no queue (a roadside layover with nobody behind
        // stays), however many stand there for something else
        assert_eq!(fan_queue(&fan, &others[3..]), (0.0, 0), "nor a bus at its own berth there");
        assert_eq!(fan_queue(&[], &others), (0.0, 0));
        assert!(!queue_asks_to_go(0.0, 0));
    }

    /// [round 6 B1] `Traffic::bus_detour`'s ask: the own lane's queue only while the bus
    /// covers its own through lane, the fan's only with a fan; neither mixed into the other.
    #[test]
    fn b1_a_layover_is_asked_to_go_by_the_queue_of_what_it_covers() {
        // its own through lane, 25 s stood behind it
        assert!(layover_asked_to_go(true, 25.0, 1, &[], 0.0, 0));
        // in a bay (its own lane not covered): the queue behind is not its
        assert!(!layover_asked_to_go(false, 25.0, 4, &[], 0.0, 0));
        // its fan, the queue it holds there
        assert!(layover_asked_to_go(false, 0.0, 0, &[3], 21.0, 1));
        assert!(layover_asked_to_go(false, 0.0, 0, &[3], 5.0, 4));
        assert!(!layover_asked_to_go(false, 0.0, 0, &[3], 5.0, 3));
        // a fan queue without a fan, or the own lane's queue with only the fan covered: no
        assert!(!layover_asked_to_go(false, 0.0, 0, &[], 30.0, 5));
        assert!(!layover_asked_to_go(false, 30.0, 5, &[3], 0.0, 0));
    }

    /// [round 6 B2] Lane 0 north 200 m into lane 1 (100 m) and on into lane 3, and lane 2 that
    /// runs into nothing.
    fn track() -> Network {
        use omsi_sim::traffic::LaneBuilder;
        let p = |x: f64, y: f64| glam::DVec3::new(x, y, 0.0);
        let mut net = Network {
            lanes: vec![
                LaneBuilder::polyline(vec![p(0.0, 0.0), p(0.0, 200.0)], LaneKind::Street, 3.5),
                LaneBuilder::polyline(vec![p(0.0, 200.0), p(0.0, 300.0)], LaneKind::Street, 3.5),
                LaneBuilder::polyline(vec![p(50.0, 0.0), p(50.0, 200.0)], LaneKind::Street, 3.5),
                LaneBuilder::polyline(vec![p(0.0, 300.0), p(0.0, 400.0)], LaneKind::Street, 3.5),
            ],
            ..Default::default()
        };
        net.link(1.5);
        net
    }

    fn bus_on(route: Vec<usize>, s: f32) -> AiState {
        let mut st = AiState::new(route[0], s, 1);
        st.front = 6.0;
        st.rear = 6.0;
        st.speed = 8.0;
        st.max_speed_kmh = 50.0;
        st.route = route;
        st.plan_next(&track());
        st
    }

    #[test]
    fn b2_a_bus_at_the_end_of_its_track_with_no_stop_left_has_done_its_trip_and_drives_on() {
        let net = track();
        assert_eq!((net.lanes[0].next.as_slice(), net.lanes[1].next.as_slice()), (&[1][..], &[3][..]));
        let mut v = test_vehicle();
        // T11's case: no stop left, the route complete, the last lane goes on
        let mut s = BusService::new(Vec::new());
        let mut st = bus_on(vec![0, 1], 150.0);
        let t = drive(&mut s, &mut st, &mut v, &net, 100.0, 60.0, |_, s, _| s.trip_done());
        assert!(s.trip_done() && s.drives_off(), "trip over on its last lane: {:?} lane {}", s.phase, st.lane);
        assert_eq!(st.lane, 1);
        assert!(st.speed > 5.0, "no stop on the way: {}", st.speed);
        // it drives on (no stop asked for), engine running, nobody waits for it to go
        let way = omsi_traffic::perception::way_lanes(&net, &st, 200.0);
        let mut ctx = ctx_on(&net, &way, t + 30.0, 0.0);
        ctx.dt = 30.0;
        assert_eq!(s.step(&mut st, &mut v, &ctx), None);
        assert!(!s.parked(t + 30.0) && s.standing_for(t + 30.0) == 0.0);
        // the tour's next trip handed over: an ordinary trip again
        s.restart(vec![Stop { ri: 0, s: 50.0, bay: 0.0, depart: 400.0, id: 9, side: 0.0 }], true);
        assert!(!s.trip_done() && !s.drives_off());

        // counter-case: at the edge of the loaded map (`route_open`) it waits for the tiles
        let mut s = BusService::new(Vec::new());
        s.route_open = true;
        let mut st = bus_on(vec![0, 1], 150.0);
        drive(&mut s, &mut st, &mut v, &net, 100.0, 15.0, |_, _, st| st.lane == 1 && st.s > 30.0);
        assert_eq!((st.lane, s.phase), (1, Phase::Running));

        // counter-case: a dead end (lane 2 runs into nothing) - it drives off it, its trip
        // comes back past it (r4d-B1): not over
        let mut s = BusService::new(Vec::new());
        let mut st = bus_on(vec![2], 100.0);
        drive(&mut s, &mut st, &mut v, &net, 100.0, 10.0, |_, _, st| st.s > 160.0);
        assert_eq!(s.phase, Phase::Running);
    }

    #[test]
    fn b2_a_terminus_stop_on_the_last_lane_is_served_before_the_trip_is_over() {
        let net = track();
        let mut v = test_vehicle();
        let mut s = BusService::new(vec![Stop { ri: 1, s: 50.0, bay: 0.0, depart: 130.0, id: 9, side: 0.0 }]);
        let mut st = bus_on(vec![0, 1], 150.0);
        let mut served = false;
        drive(&mut s, &mut st, &mut v, &net, 100.0, 120.0, |_, s, _| {
            served |= s.at_stop();
            s.trip_done()
        });
        assert!(served, "it stopped at its terminus first");
        assert!(s.trip_done() && !s.drives_off(), "then its trip is over there, standing (as before round 6)");
        assert!(st.lane == 1 && (st.s - 50.0).abs() < 2.5 && st.speed < 0.3, "at its stop: s {} speed {}", st.s, st.speed);
    }
}
