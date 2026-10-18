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
    /// It leaves this stop early for the traffic behind it.
    left_early: bool,
    /// [ROLLBACK r4d-B4] Held short of its stop by somebody in its way: it creeps up to the
    /// pole once they are clear rather than serving the stop where it stands.
    short_by_people: bool,
}

/// [ROLLBACK r4d-B2] A bus away from its stop until its departure.
#[derive(Debug, Clone)]
pub struct Away {
    /// It comes back for its layover (it waits for the departure when it is back).
    layover: bool,
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

/// [ROLLBACK r4d-B2] Seconds a detour of `len` metres takes.
fn detour_time(len: f32) -> f64 {
    (len / DETOUR_SPEED) as f64
}

/// [ROLLBACK r4d-B2] A bus standing across a through lane: a street lane random traffic runs
/// on (not a stand of its own), its body (`half_width`) not moved off the lane into a bay
/// (a bus half on the pavement still holds up a car).
pub fn covers_through_lane(net: &Network, lane: usize, lateral: f32, half_width: f32) -> bool {
    net.lanes.get(lane).is_some_and(|l| l.kind == LaneKind::Street && l.density > 0.0 && lateral.abs() < half_width + l.width * 0.5 - 0.5)
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
        if len <= LAYOVER_LOOP_MAX && detour_time(len) <= budget && best.as_ref().is_none_or(|b| len < b.0) {
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
    /// Where it would swerve to round a car parked at the kerb.
    pub kerb_swerve: Option<f32>,
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
            pace: 1.0,
            left_early: false,
            short_by_people: false,
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

    /// [ROLLBACK aiparked-73] Standing out a long layover with the engine off: it starts again
    /// `PARKED_START` before the doors open for the departure.
    pub fn parked(&self, day_time: f64) -> bool {
        // [ROLLBACK terminus-130] a bus that has finished its trip and stands at the end of
        // the line waits for its next trip with the engine off, as one placed there does
        // (it stood with the engine running until the next trip was handed to it)
        if self.phase == Phase::TripDone {
            return self.phase_t > TRIP_DONE_PARK;
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
        self.left_early = false;
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
        let wait = if self.ahead_by >= 1.0 && !ctx.roadside {
            let rest = std::mem::take(&mut self.ahead_by).min((depart - ctx.day_time).max(0.0));
            log::info!("t={:.1}: timetable bus {} at its stop {id} {rest:.0} s early after its early start: it waits", ctx.day_time, ctx.id);
            wait.max(rest)
        } else {
            wait
        };
        self.pace = 1.0;
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
            self.stops.pop_front();
            crate::traffic::ibis_to_next_stop(vehicle, self.stops.len());
        }
        self.near_d = f32::INFINITY;
        self.serve = None;
        self.short_by_people = false;
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
            Phase::Waiting | Phase::Boarding => self.leave_at - day_time,
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
            Phase::Boarding | Phase::Waiting => {
                self.pending = Some(d);
                self.leave_at = day_time;
                self.set_phase(Phase::Closing);
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
                self.away = Some(Away { layover, park: None });
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
                self.away = Some(Away { layover, park: Some(ParkAway { ri, s, lat, back, rest, stops, leave_at, arrived: false }) });
            }
        }
        // (back at the stop it waits for its departure as on its layover)
        self.layover = layover;
        self.near_d = f32::INFINITY;
        self.serve = None;
        self.short_by_people = false;
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
                st.lateral_target = if d < BAY_REACH { p.lat } else { 0.0 };
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
            Phase::TripDone => here,
            Phase::Waiting => {
                self.wait(ctx);
                here
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
                if !ctx.passing {
                    st.lateral_target = ctx.kerb_swerve.unwrap_or(0.0);
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
                continue;
            }
            let d = st.route_distance(ctx.net, stop.ri, stop.s);
            // [ROLLBACK r4d-B3] slower while ahead of the timetable: the lead goes by the time
            // it is at the stop (the time it drives slower is time it is behind its usual pace)
            if self.ahead_by >= 1.0 {
                self.pace = pace_for(self.ahead_by, d, ctx.v_nom);
                if st.speed > 0.5 {
                    self.ahead_by = (self.ahead_by - ctx.dt as f64 * (1.0 - self.pace) as f64).max(0.0);
                }
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
                crate::traffic::ibis_to_next_stop(vehicle, self.stops.len());
                if !ctx.passing {
                    st.lateral_target = ctx.kerb_swerve.unwrap_or(0.0);
                }
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
                st.lateral_target = if d < BAY_REACH && d > -25.0 && !junction_first {
                    stop.bay
                } else {
                    ctx.kerb_swerve.unwrap_or(0.0)
                };
            }
            // queued behind something standing at its stop (another bus, the player's):
            // after a while it serves the stop where it stands, as drivers do
            let in_bay = (st.lateral - stop.bay).abs() < 0.5;
            // (a bus with another one behind it for this stop pulls on past the stop point)
            let pull = if ctx.follower_close { 2.5 } else { 0.0 };
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
            let queued = close || (speed < 0.3 && ctx.stopped > 6.0 && d < 45.0 && d >= 2.0 && !people_short);
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
            kerb_swerve: None,
            lead_close: false,
            follower_close: false,
            wanted: None,
            debug: false,
            roadside,
            queue_behind_secs,
            v_nom: 13.9,
            people_hold: false,
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
        Ctx { net, way, day_time, dt: 0.1, id: 1, stopped, passing: false, kerb_swerve: None, lead_close: false, follower_close: false, wanted: None, debug: false, roadside: true, queue_behind_secs: 0.0, v_nom: 13.9, people_hold: false }
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
        assert!(covers_through_lane(&net, 0, st.lateral, 1.25));
        assert!(queue_asks_to_go(5.0, 4) && queue_asks_to_go(25.0, 0) && !queue_asks_to_go(5.0, 3));
        let left = s.detour_wanted(0.0, 100.0, false).expect("a long wait");
        assert!((left - 300.0).abs() < 1e-6);
        let d = plan_detour(&net, 0, 50.0, 50.0, left, &|_| None).expect("round the ring");
        let Detour::Loop { route, len } = d.clone() else { panic!("{d:?}") };
        assert_eq!(route, vec![0, 1, 2, 3, 0]);
        assert!((len - 400.0).abs() < 2.0, "{len}");
        s.start_detour(&mut st, &net, d, 100.0, 1);
        assert_eq!(s.phase, Phase::Closing);
        // the doors shut, round the block, back at its stop before its departure
        let mut left_at = None;
        let back = drive(&mut s, &mut st, &mut v, &net, 100.0, 300.0, |t, s, _| {
            if s.phase == Phase::Running && left_at.is_none() {
                left_at = Some(t);
            }
            left_at.is_some() && s.at_stop()
        });
        assert!(left_at.is_some_and(|t| t < 105.0), "{left_at:?}");
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
        assert!(!covers_through_lane(&net, 0, 3.2, 1.25));
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
        assert!(left > 30.0 && gone < 170.0, "not waited out in the lane: {left}, on at {gone}");
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
        assert!(lateral_at.iter().any(|&(s, l)| s < 101.0 && l > 1.1), "starts in the bay");
        assert!(st.lateral.abs() < 0.05, "back in the lane: {}", st.lateral);
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
}
