//! Junction grants: the one place where vehicles decide against each other.
//!
//! Every frame each vehicle that is at or inside a junction files a [`Claim`] for its
//! movement: where it is, how fast it comes, the gap it accepts. [`resolve`] reads the
//! junction's foe table and answers each claim with a [`Grant`]: go, or wait for a named
//! vehicle. The answer is a pure function of the claims, so the stage can be checked in a
//! test with no vehicles at all.

use crate::driver::{Impatience, STANDING_SPEED};
use crate::junction::{FoeKind, Junctions};
use omsi_sim::traffic::{arrival_time, APPROACH_WINDOW, MAX_BRAKE};

/// A vehicle asking for its way through a junction this frame.
#[derive(Debug, Clone, PartialEq)]
pub struct Claim {
    /// The pieces of the junction the way runs through, each with how far along the way
    /// its start lies (`Junctions::chain_along`): the claim stands for all of them.
    pub chain: Vec<(usize, f32)>,
    /// The caller's index of the vehicle (echoed in the grant).
    pub vehicle: usize,
    pub junction: usize,
    pub movement: usize,
    /// Distance of the vehicle's origin along the movement's lane (m); negative while
    /// still on the entry lane (metres before the junction).
    pub s: f32,
    pub speed: f32,
    pub accel: f32,
    pub v_max: f32,
    /// From the origin to the front and the rear bumper (m).
    pub front: f32,
    pub rear: f32,
    /// The driver's comfortable braking (m/s², `DriverParams::comfort_brake`): what a foe
    /// is reckoned to brake with for me (SUMO's `unsafeMergeSpeeds`, the impatient arrival
    /// `computeFoeArrivalTimeBraking`; ).
    pub decel: f32,
    /// Its waiting time (s, SUMO's, `Vehicle::waiting`): its impatience grows with it
    /// (`impatience_of`).
    pub waiting: f32,
    /// Its lead (gap from its front to the lead's rear, the lead's speed), if any: its
    /// planned arrival (`planned_window`).
    pub lead_gap: Option<(f32, f32)>,
    /// The id of the foe it was capped behind at a zipper last frame
    /// (`Vehicle::zip_behind`): the pair's order, fixed once, is kept while both are engaged.
    pub zip_behind: Option<u64>,
    /// Its claim was refused last frame (`Vehicle::refused_last`). The resolver decides such a
    /// claimant, standing, before the others (`decide_in`), so that its foes read its answer of
    /// this frame: standing outside the meeting place and refused, it will not pass (SUMO's
    /// `willPass`, `will_pass`).
    pub refused_last: bool,
    /// The vehicle whose body across its way holds it this frame
    /// (`Perception::body_block`): at a zipper that one is ahead of it whatever the arrival
    /// rule says.
    pub body_block: Option<u64>,
    /// The first place along the way through (the chain's distances) where it
    /// can come to rest with its body on no band (`plan::next_wait`; -INFINITY: not known, a
    /// player): it cannot wait short of it, so it is in the way of whoever comes to the meeting
    /// places before it (SUMO's `willPass`), and it is refused only for those past it.
    pub next_wait: f32,
    /// The time gap it keeps to the car ahead (s, the driver's `tau`): at a zipper it
    /// arrives this long after the foe ahead (`getZipperSpeed`); and its standing gap (m,
    /// `min_gap`): with both, whether it could follow a foe that cuts in ahead (`zipper_cap`).
    pub tau: f32,
    pub min_gap: f32,
    /// Already inside the junction, or too close to the line to stop before it: holds its
    /// way whatever comes.
    pub committed: bool,
    /// There is room on an exit lane for this vehicle.
    pub exit_room: bool,
    pub impatience: Impatience,
    /// A train, or anything else that never gives way.
    pub absolute: bool,
    /// Standing, held by something of its own (a lead, a full exit): it is not about to
    /// move off, whatever its arrival time says.
    pub held: bool,
    /// Held at this junction's line by a red light, and the seconds until it shows green
    /// (None: not within a cycle).
    pub light_red: bool,
    pub until_go: Option<f32>,
    /// Where the red line that holds it lies along the way through (the chain's
    /// distances, as the foes' stretches; INFINITY: none): the light holds it only before the
    /// meeting places that begin past it.
    pub red_at: f32,
    /// The vehicle this one follows this frame (the caller's index), if any: a claimant
    /// does not wait for its own follower.
    pub lead: Option<usize>,
    /// Counted standing at its last claim: see [`Claim::standing`].
    pub was_standing: bool,
    /// Standing, held there by something other than a junction's answer: its lead, people,
    /// a zebra, a light. It cannot move off before that lets it go,
    /// whatever the junction says.
    pub blocked: bool,
    /// Blocked (`blocked`) by something that does not let it go on a schedule: people or a
    /// zebra in front of it, a lead that stands too. Not coming for as long as that holds,
    /// however near the meeting place it stands.
    pub stuck: bool,
    /// In the box when its light turned, unable to stop before its first meeting place or at
    /// it (`JunctionView::clearing`): it goes on through. To the others it is committed (a
    /// fresh green waits for it); itself it waits only for foes in the meeting place, committed
    /// or clearing, between which the table decides (two clearing claimants must not both go: a left
    /// turner in the box against an amber runner).
    pub clearing: bool,
    /// Past its line, crossed while its light let it in, and that light no longer green
    /// (`JunctionView::in_box_at_turn`): in the box at the end of its green, it clears first. A
    /// foe whose light is never green with its own (the fresh green across its way) waits for it
    /// and is no foe of its (`will_pass`; SUMO: a vehicle already on the junction is never blocked
    /// by a link that just opened).
    pub in_box_at_turn: bool,
    /// The vehicle's lanes from the one it is on: (lane, distance from its origin to the
    /// lane's start; the first is minus its place along it). Empty: not known (a player off
    /// the lanes). (For a car behind me in my own lane.)
    pub lanes: Vec<(usize, f32)>,
    /// The light at this junction's line (controller, light; `JunctionView::light`), None
    /// on an unlit way. With `light_red` false the car has its green (or takes the yellow).
    pub light: Option<(usize, usize)>,
    /// The vehicle's id (a player: `PLAYER_ID_BASE` + its id): who went last at a zipper.
    pub id: u64,
    /// Eased off for the vehicle with this index (the caller's) after the two held each other
    /// too long (`Simulation`'s negotiation): it is refused, waiting for that one, and that
    /// one's windows do not see it. Every other foe of either stays as it is.
    pub yields_now: Option<usize>,
    /// Distance from the origin to where the front will have to stop for people, a zebra or
    /// a standing lead ahead (INFINITY: nothing seen).
    pub clear_ahead: f32,
    /// Held for a stop it serves (`External::at_stop`, a timetable bus's dwell, or the bus
    /// service's stop hold, `Why::Service`): standing there it has no request at any link until
    /// its stop ends (`will_pass`).
    pub serving: bool,
    /// The brake (m/s²) its committed test reckons with (`Simulation::tick`: the vehicle's
    /// own `decel * 1.5` inside the object, `plan::HARD_BRAKE` before it).
    pub brake: f32,
}

/// What the resolver reads besides the claims: the light programs' signal groups.
#[derive(Debug, Clone, Default)]
pub struct Context {
    /// (controller, light a, light b) with a < b: the two lights never let vehicles in at
    /// the same time (`TrafficLightController::never_together`). A pair not in the map is
    /// not known to be exclusive.
    pub exclusive: hashbrown::HashMap<(usize, usize, usize), bool>,
}

impl Context {
    /// Lights `a` and `b` (controller, light) are never green together.
    pub fn exclusive(&self, a: (usize, usize), b: (usize, usize)) -> bool {
        a.0 == b.0 && a.1 != b.1 && self.exclusive.get(&(a.0, a.1.min(b.1), a.1.max(b.1))).copied().unwrap_or(false)
    }
}

impl Claim {
    /// Standing, with hysteresis: under 0.3 m/s, or still under 1 m/s after standing at the
    /// last claim (a car let go that creeps off is not refused again by the ordinary rules
    /// the moment it passes 0.3 m/s, to be let go again once it has stopped).
    pub fn standing(&self) -> bool {
        self.speed < STANDING_SPEED || (self.was_standing && self.speed <= 1.0)
    }

    /// Where its front comes to rest behind a standing lead - or a body across its way, which
    /// perception gives as its lead - in its origin's distances (`s` + `front` + the gap,
    /// `lead_gap`); INFINITY with no lead or one that moves. Not people: they clear on their own
    /// schedule (a stand for people is `stuck`'s, `held_off`).
    pub fn lead_rest(&self) -> f32 {
        match self.lead_gap {
            Some((gap, v)) if v < STANDING_SPEED => self.s + self.front + gap,
            _ => f32::INFINITY,
        }
    }
}

/// Why a vehicle waits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Hold {
    /// A foe with the right of way is coming (or is in the junction).
    Foe,
    /// No room past the junction.
    ExitFull,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grant {
    pub vehicle: usize,
    pub go: bool,
    /// The vehicle waited for, if any.
    pub wait_for: Option<usize>,
    pub hold: Option<Hold>,
    /// Eased off for the other of a mutual hold that rolls on into a merge of the two ahead:
    /// follow it in rather than stop, (gap from my front to its rear along the merged way, its
    /// speed; `eased_follow`).
    pub follow: Option<(f32, f32)>,
    /// The zipper speed for the foe ahead at a merge of two ways that do not cross
    /// (`getZipperSpeed`): never a refusal, a cap on the acceleration (`plan`), or the
    /// queue-entry stop behind it when it stands at the join.
    pub zipper: Option<ZipCap>,
    /// Refused for a foe: where the first stretch it waits for begins, along the way
    /// through (the chain's distances). The car stops just before it (`plan`).
    pub at: Option<f32>,
    /// Foes (claimants on a foe movement) this claim was let go for by SUMO's time test
    /// this frame: as the follower (the foe out of the stretch `JM_TIMEGAP_MINOR` before I
    /// am in it) and as the leader (I am out `JM_TIMEGAP_MINOR` before it is in). (`ab`'s
    /// `follower_go`/`leader_go`.)
    pub follower_go: u16,
    pub leader_go: u16,
}

impl Grant {
    pub fn go(vehicle: usize) -> Grant {
        Grant { vehicle, go: true, wait_for: None, hold: None, follow: None, zipper: None, at: None, follower_go: 0, leader_go: 0 }
    }

    pub fn refused(vehicle: usize, wait_for: Option<usize>, hold: Hold) -> Grant {
        Grant { vehicle, go: false, wait_for, hold: Some(hold), follow: None, zipper: None, at: None, follower_go: 0, leader_go: 0 }
    }
}

/// The time gap (s) a claimant keeps to a foe at a meeting place, as the follower (the
/// foe out before I am in) and as the leader (I am out before it is in): SUMO's
/// `jmTimegapMinor` default, the `lookAhead` of `MSLink::blockedByFoe` (sumo.md §1.5, §2.7).
pub const JM_TIMEGAP_MINOR: f32 = 1.0;
/// SUMO's `--time-to-impatience` default (s): the waiting time that makes a driver
/// wholly impatient (sumo.md §1.5).
pub const TIME_TO_IMPATIENCE: f32 = 180.0;
/// SUMO's vType `impatience` default: none before any waiting (sumo.md §2.7).
pub const IMPATIENCE_BASE: f32 = 0.0;
/// Impatience at which a driver goes first in front of a foe that stands and waits too.
const HALF_FED_UP: f32 = 0.5;
/// SUMO's `getZipperSpeed` blend (s): the braking a zipper cap may ask for goes from the
/// comfortable one, `ZIPPER_BLEND_TIME` or more from the merge, to the emergency one at it,
/// b = w b_comfort + (1 - w) b_emergency with w = min(1, t_e / this) (sumo.md 1.5).
pub const ZIPPER_BLEND_TIME: f32 = 10.0;

/// A zipper speed (`Grant::zipper`): the foe ahead (the caller's index), the gap from my
/// front to its rear along the merged way (the two ways aligned at the stretch's start), its
/// speed; the acceleration that brings me to the stretch's start `tau` after its rear
/// (`arrive`, -INFINITY: none, the cap is the following alone) and the hardest braking the cap
/// may ask (`floor`, negative).
/// `stop`: the queue-entry case - it stands at the join, my lead.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct ZipCap {
    pub foe: usize,
    pub gap: f32,
    pub speed: f32,
    pub arrive: f32,
    pub floor: f32,
    pub stop: bool,
    /// The distance from my front to the stretch's start while the foe's rear
    /// has not been past it `tau` (`getZipperSpeed`'s target): a position bound, I do not enter
    /// before my turn (a constant-deceleration bound would let the follower creep in). None for a
    /// car with the right of way over the foe's stream: it follows its link leader in.
    pub hold: Option<f32>,
    /// The foe is in the stretch already (its front past the start): a cap
    /// that brings me to rest behind it puts me at a waiting position (`plan`).
    pub inside: bool,
    /// From my front to the stretch's start (m; negative past it): against `zipper_reach`.
    pub dist: f32,
    /// The foe stops at its own red short of the stretch (ahead of me only when its green
    /// brings it there first): for the tool's `capped_by_red`.
    pub red_short: bool,
}

/// How far away (s) a foe has to be before a fed-up driver takes the way in front of it:
/// `driver::FED_UP_AFTER` for a foe that stands or crawls, twice that for one rolling,
/// and never one at town speed or more.
fn after_for(foe_speed: f32, squeeze: Option<f32>) -> f32 {
    let base = squeeze.unwrap_or(f32::INFINITY);
    if foe_speed < 2.0 {
        base
    } else if foe_speed < 8.0 {
        base * 2.0
    } else {
        f32::INFINITY
    }
}

/// When the claimant enters and leaves its stretch (from, to) along its lane.
fn window(c: &Claim, from: f32, to: f32) -> (f32, f32) {
    let t_in = arrival_time(from - (c.s + c.front), c.speed, c.accel, c.v_max);
    let t_out = arrival_time(to - (c.s - c.rear), c.speed, c.accel, c.v_max);
    (t_in, t_out)
}

/// My planned (in, out) at the stretch (from, to): `window`, no earlier than my lead lets
/// me (SUMO plans the arrival at a link with its leaders, `vLinkPass`): my front reaches a place
/// no sooner than `tau` after the lead's rear is there with my standing gap, the lead going on at
/// its speed up to my limit (`Claim::lead_gap`; a standing lead is `clear_ahead`'s).
fn planned_window(c: &Claim, from: f32, to: f32) -> (f32, f32) {
    let (mut t_in, mut t_out) = window(c, from, to);
    if let Some((gap, v_lead)) = c.lead_gap.filter(|l| l.1 >= STANDING_SPEED) {
        let front = c.s + c.front;
        // (when the lead's rear has gone `d` m from where it is: my front `d` m further on)
        let lead_at = |p: f32| arrival_time(p + c.min_gap - (front + gap), v_lead, c.accel, c.v_max.max(v_lead)) + c.tau;
        t_in = t_in.max(lead_at(from));
        t_out = t_out.max(lead_at(to + c.rear + c.front));
    }
    (t_in, t_out)
}

/// SUMO's impatience of a claimant (sumo.md §1.5): I = clamp(`IMPATIENCE_BASE` +
/// waiting / `TIME_TO_IMPATIENCE`, 0, 1).
pub fn impatience_of(c: &Claim) -> f32 {
    (IMPATIENCE_BASE + c.waiting / TIME_TO_IMPATIENCE).clamp(0.0, 1.0)
}

/// The speed (m/s) after `dist` m from `v` speeding up at `a` towards `v_max` (as
/// `arrival_time` reckons).
fn speed_after(dist: f32, v: f32, a: f32, v_max: f32) -> f32 {
    let v = v.max(0.0);
    (v * v + 2.0 * a.max(0.05) * dist.max(0.0)).sqrt().min(v_max.max(v))
}

/// SUMO's `MSLink::computeFoeArrivalTimeBraking`: when the foe brakes at its `decel`
/// (SUMO's `getMaxDecel`, our comfortable brake) times my impatience `imp` from the moment I
/// arrive (`my_in`), when it arrives at its stretch `dist` m on, planned at `their_in` from
/// now: (its arrival time, its arrival speed). Its speed is the average one of its plan,
/// dist / their_in. A foe that can stop short by then arrives 30 s later (SUMO's branch:
/// `foeArrivalTime + TIME2STEPS(30)`, speed 0).
fn foe_arrival_braking(my_in: f32, their_in: f32, dist: f32, decel: f32, imp: f32) -> (f32, f32) {
    let m = decel * imp;
    if !(their_in > my_in) || !their_in.is_finite() || m <= 0.0 || dist <= 0.0 {
        return (their_in, f32::NAN);
    }
    let dt = their_in - my_in;
    let d = dt * m;
    let a = dt * d / 2.0;
    let v = dist / their_in;
    let dist2 = dist - v * my_in;
    if 0.5 * v * v / m <= dist2 {
        return (their_in + 30.0, 0.0);
    }
    let disc = 4.0 * (v - d) * (v - d) - 8.0 * m * a;
    if disc < 0.0 {
        return (their_in, f32::NAN);
    }
    let x = (disc.sqrt() * -0.5 - d + v) / m;
    (their_in + x, (v - (dt + x) * m).max(0.0))
}

/// SUMO's `arrivalSpeedBraking`: the speed at `dist` m braking at `b` from `v`
/// (0 when it stops short).
fn arrival_speed_braking(dist: f32, v: f32, b: f32) -> f32 {
    (v.max(0.0).powi(2) - 2.0 * b.max(0.1) * dist.max(0.0)).max(0.0).sqrt()
}

/// SUMO's `unsafeMergeSpeeds`: two going on in one lane, the follower could not stop
/// behind the leader braking at their brakes (each its comfortable one, `Claim::decel`):
/// leader_v²/b_leader < follower_v²/b_follower. (SUMO's MSLink.h has `<=`. Ours is strict for
/// an OMSI reason: our follower's arrival speed is capped at the leader's leave speed, so
/// equal speeds are the normal planned following, already `JM_TIMEGAP_MINOR` behind; with
/// `<=` every planned follower would be blocked.)
fn unsafe_merge(leader_v: f32, follower_v: f32, leader_b: f32, follower_b: f32) -> bool {
    leader_v * leader_v / leader_b.max(0.1) < follower_v * follower_v / follower_b.max(0.1) && follower_v > STANDING_SPEED
}

/// How SUMO's time test (`MSLink::blockedByFoe`, sumo.md §1.5) saw a foe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum FoeCase {
    /// The foe is out of the stretch `JM_TIMEGAP_MINOR` before I am in it: I come after it.
    Follower,
    /// I am out of the stretch `JM_TIMEGAP_MINOR` before it is in (its arrival as impatience
    /// sees it): I go first.
    Leader,
    /// Neither: the two would be in the stretch together.
    Conflict,
}

/// The times of one claimant against one foe at one stretch (s from now; the foe's
/// arrival also as the claimant's impatience reckons it) and the case they make.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct FoeTimes {
    pub my_in: f32,
    pub my_out: f32,
    pub their_in: f32,
    pub their_out: f32,
    pub their_in_impatient: f32,
    pub case: FoeCase,
}

/// SUMO's `blockedByFoe` for the claimant `c` against the foe `o` at the stretch
/// `mine`/`theirs` (along each one's way): the follower case when the foe leaves before I
/// arrive by `JM_TIMEGAP_MINOR`, the leader case when it arrives after I leave by that gap -
/// its arrival re-estimated by my impatience I as (1-I) t_free + I t_braking, braking at its
/// comfortable brake (`computeFoeArrivalTimeBraking`); in one lane after a merge
/// (`same_lane`) either case needs speeds the follower can stop behind (`unsafe_merge`: a
/// merger that a 14 m/s main road car would have to brake hard behind is no leader).
/// `their_window` is the foe's (in, out) as the resolver has shifted it (a red, a stand).
fn foe_times(c: &Claim, o: &Claim, mine: (f32, f32), theirs: (f32, f32), their_window: (f32, f32), same_lane: bool) -> FoeTimes {
    let (mut my_in, mut my_out) = planned_window(c, mine.0, mine.1);
    // (my times as I will drive, not as I could - SUMO's arrival and leave times are the
    // vehicle's planned ones: stopped by people, a zebra or a standing lead ahead
    // (`Claim::clear_ahead`) short of the stretch I do not come, stopped inside it I do not
    // leave; a car held by people short of its stretch does not count itself in it)
    // (stopped inside it already in it or standing - held where I am, a refusal moves nothing -
    // I do not come either: a car standing behind its lead in the stretch does not wait for a
    // foe at a far-off red)
    let stop = c.s + c.clear_ahead;
    let held_inside = stop < mine.1 + c.rear + c.front && (c.s + c.front >= mine.0 || c.speed < STANDING_SPEED);
    if stop <= mine.0 || held_inside {
        my_in = f32::INFINITY;
        my_out = f32::INFINITY;
    } else if stop < mine.1 + c.rear + c.front {
        my_out = f32::INFINITY;
    }
    let (their_in, their_out) = their_window;
    // (one that does not come registers no approach in SUMO: no foe is in its way - the
    // follower case outright, never a conflict through `unsafeMergeSpeeds`)
    if my_in.is_infinite() {
        return FoeTimes { my_in, my_out, their_in, their_out, their_in_impatient: their_in, case: FoeCase::Follower };
    }
    // SUMO's impatience (`blockedByFoe`): only when I arrive before the foe, its arrival as if it
    // braked at its decel times my impatience from my arrival (`foe_arrival_braking`)
    let imp = impatience_of(c);
    let (t_brake, v_brake) = if imp > 0.0 && my_in < their_in && o.speed >= STANDING_SPEED { foe_arrival_braking(my_in, their_in, theirs.0 - (o.s + o.front), o.decel, imp) } else { (their_in, f32::NAN) };
    let their_in_impatient = (1.0 - imp) * their_in + imp * t_brake;
    // (SUMO's `arrivalSpeedBraking`: the foe's speed at the stretch if it brakes at its decel
    // from now - 0 when it can stop short, a foe at its red always; the free speed would make a Conflict
    // of a foe far off at its red)
    let their_arrive_v = if v_brake.is_nan() { arrival_speed_braking(theirs.0 - (o.s + o.front), o.speed, o.decel) } else { v_brake };
    // (my arrival speed behind it as planned: no faster than following it lets me - SUMO's ego
    // plans its link speed with its leaders - nor slower than I go now; free, a car standing
    // off would "arrive" at the limit behind a stream leaving slowly and never follow it in)
    let their_leave_v = speed_after(theirs.1 - (o.s - o.rear), o.speed, o.accel, o.v_max);
    let my_arrive_v = speed_after(mine.0 - (c.s + c.front), c.speed, c.accel, c.v_max).min(their_leave_v.max(c.speed));
    let my_leave_v = speed_after(mine.1 - (c.s - c.rear), c.speed, c.accel, c.v_max);
    // SUMO's `blockedByFoe`, in its order: the foe out before I am in - the follower, blocked
    // only into one lane with less than `JM_TIMEGAP_MINOR` between or unsafe speeds (a crossing
    // needs no gap: SUMO's `sameTargetLane` condition); else the foe in `JM_TIMEGAP_MINOR` after
    // I am out - the leader, blocked only into one lane at unsafe speeds; else a conflict
    let case = if their_out < my_in {
        if same_lane && (my_in - their_out < JM_TIMEGAP_MINOR || unsafe_merge(their_leave_v, my_arrive_v, o.decel, c.decel)) {
            FoeCase::Conflict
        } else {
            FoeCase::Follower
        }
    } else if their_in_impatient > my_out + JM_TIMEGAP_MINOR {
        if same_lane && unsafe_merge(my_leave_v, their_arrive_v, c.decel, o.decel) {
            FoeCase::Conflict
        } else {
            FoeCase::Leader
        }
    } else {
        FoeCase::Conflict
    };
    FoeTimes { my_in, my_out, their_in, their_out, their_in_impatient, case }
}

/// The room (m) the car-following model (IDM) needs behind a lead at `lead_speed` to follow it
/// braking no harder than comfortably: its braking a (s*/s)² is within the comfortable brake b
/// for s >= s* sqrt(a/b), s* = s0 + v T + v Δv / (2 sqrt(a b)) (SUMO's `couldBrakeForLeader` asks
/// the same of its Krauss model).
fn follow_room(x: &Claim, lead_speed: f32) -> f32 {
    let (a, b) = (x.accel.max(0.1), x.decel.max(0.1));
    let s_star = x.min_gap + (x.speed * x.tau + x.speed * (x.speed - lead_speed) / (2.0 * (a * b).sqrt())).max(0.0);
    s_star * (a / b).sqrt()
}

/// The zipper reach of the claimant `c` (m, from its front to the merge stretch's start): SUMO's
/// `getZipperSpeed` adapts nothing while ego is farther from the link than max(visibility,
/// brakeGap) (sumo.md 1.5). The brake gap is the room the car-following model needs to come to
/// rest comfortably behind a car standing at the stretch's start (`follow_room`, the measure the
/// zipper's "can brake" reads: farther off it can still stop behind whoever is there first, and
/// need not adapt yet); the visibility
/// is the approach the junction compile reads a meeting place's right of way over
/// (`omsi_sim::traffic::APPROACH_WINDOW`, the last metres of each way to it,
/// `Network::approach_to`): OMSI maps carry no connection `visibility`.
pub fn zipper_reach(c: &Claim) -> f32 {
    APPROACH_WINDOW.max(follow_room(c, 0.0))
}

/// SUMO's zipper (`MSLink::getZipperSpeed`, sumo.md 1.5) for the claimant `c` against `o`
/// at the merge `foe` of two ways that cross nowhere else: the foe ahead caps my speed, a foe
/// behind is none of mine. Aligned at the stretch's start (where two abreast would touch), one
/// in the stretch is ahead of one short of it (of two in it, the one further along); of two
/// short of it the one there first is ahead (`their_in` the foe's arrival as the resolver has shifted it, at its
/// green; a tie: the one further along, then the lower index). The table orders the rest: one
/// giving way (`i_yield`) also lets a later foe go first that could not stop behind it with its
/// comfortable brake (a 14 m/s main road car must not brake hard for a merger),
/// and the one with the right of way takes an earlier yielding foe as ahead only when it could
/// stop behind it so (the same pair from the other side: exactly one of the two is ahead);
/// between equals the order of arrival alone (SUMO: a later foe that can brake for me is
/// ignored). A car the table gives the right of way over the foe's stream (`yields_to_me`) is
/// not capped by a foe of that stream that can still stop short of the stretch: its link is open
/// (SUMO's `havePriority`, sumo.md 1.5); a foe already in the stretch ahead of it, or there first
/// and too close to stop short of it, is its link leader when it can brake for it (one it cannot
/// brake for gives way to it, as above), which it follows with no position hold at the stretch's
/// start (SUMO's `jmAdvance`, sumo.md 1.6). A foe
/// standing short of the stretch, or one whose own plan stops at its red short of it, can always
/// brake (SUMO's `couldBrakeForLeader` of a stopped vehicle; a car stopping for its red brakes
/// whatever I do), so it is ahead only when it arrives first: one at its red is ahead of me only
/// when its green and its start bring it there before me; and one whose plan stops behind a
/// standing lead short of the stretch (`Claim::lead_rest`) is no foe ahead at all (SUMO's
/// `willPass`: its planned stop is before the link). Outside its zipper reach
/// (`zipper_reach`) a car is capped by no foe short of the stretch (SUMO adapts nothing there);
/// a foe in the stretch, or standing at the join, is followed at any distance. For the foe
/// ahead: the queue-entry stop when it stands at the join (`ZipCap::stop`,
/// my lead), else the gentler of following it where it stands along the merged way and the
/// acceleration that brings my front to the start `tau` after its rear (`arrive`; inside the
/// stretch already or there late enough anyway, none: the cap is the following), never braking
/// harder than w b_comfort + (1 - w) b_emergency, w = min(1, t_e / `ZIPPER_BLEND_TIME`).
fn zipper_cap(c: &Claim, o: &Claim, foe: &crate::junction::Foe, their_in: f32, red_shift: f32, i_yield: bool, yields_to_me: bool) -> Option<ZipCap> {
    let o_len = o.front + o.rear;
    // (the fronts past the stretch's start; negative short of it)
    let p_e = (c.s + c.front) - foe.mine.0;
    let p_f = (o.s + o.front) - foe.theirs.0;
    let t_e = window(c, foe.mine.0, foe.mine.1).0;
    let t_f = their_in;
    // (short of the stretch, its plan stops it before it: at its red, or behind a standing lead,
    // `Claim::lead_rest`)
    let red_short = p_f < 0.0 && o.light_red && o.red_at <= foe.theirs.0;
    if p_f < 0.0 && o.lead_rest() <= foe.theirs.0 && c.body_block != Some(o.id) {
        return None;
    }
    // (outside my zipper reach, a foe short of the stretch adapts nothing of mine)
    if p_f < 0.0 && -p_e > zipper_reach(c) && c.body_block != Some(o.id) {
        return None;
    }
    // `x`, `behind` m further from the start than `y`, could follow `y` as its lead there
    // braking no harder than comfortably: the car-following model's (IDM's) braking for a lead
    // at that gap, a (s*/s)², is within its comfortable brake b - s >= s* sqrt(a/b), s* = s0 + v T
    // + v Δv / (2 sqrt(a b)). (SUMO's `couldBrakeForLeader` asks the same of its Krauss model;
    // a plain stop from x's speed let a merger in 55 m ahead of a 13.5 m/s main road car, which
    // then braked 8 m/s² in the model's following.) One standing short of the
    // stretch (`x_at` its front's place, negative) can always: its brake gap is 0, as SUMO's
    // `couldBrakeForLeader` finds for a stopped vehicle - so a foe standing at its red is ahead
    // only when it arrives first, its arrival counted from its green (`their_in`).
    let can_brake = |x: &Claim, x_at: f32, y: &Claim, behind: f32| -> bool {
        if (x.speed < STANDING_SPEED || (x.vehicle == o.vehicle && red_short)) && x_at < 0.0 {
            return true;
        }
        let room = behind - (y.front + y.rear);
        room > 0.0 && room >= follow_room(x, y.speed)
    };
    // (level: the table's, then the lower index)
    let tie_theirs = i_yield || (!yields_to_me && o.vehicle < c.vehicle);
    let first = t_f < t_e || (t_f == t_e && (p_f > p_e || (p_f == p_e && tie_theirs)));
    // the pair's order is fixed once and kept while both are engaged (the memory of
    // last frame's cap, `Claim::zip_behind`); fixed by arrival and `couldBrakeForLeader`: the foe
    // there first is ahead unless I cannot brake for it and it can brake for me; one there later
    // only when it cannot brake for me and I can for it (SUMO's zipper ignores a later foe that
    // can brake for ego; exactly one of a pair is ahead). In the stretch is
    // no rule of its own (it is arriving first): a flip there would put a fast car
    // under an emergency cap
    // (engaged: the foe on its way or in the stretch - a foe standing short of the stretch, held
    // elsewhere, or rolling up to its red short of it, keeps no order: a car would wait at the
    // start behind one stalled at its red)
    let engaged = (o.speed >= STANDING_SPEED && !red_short) || p_f >= 0.0;
    // (a body across the other's way is ahead of it, whatever the arrival rule says: the one
    // it holds cannot go first; held by its body, a car the cap put behind it stood with it
    // in a ring - the same rule as `negotiate`'s, where the
    // body-blocked one eases off)
    let ahead = if c.body_block == Some(o.id) {
        true
    } else if o.body_block == Some(c.id) {
        false
    } else if engaged && c.zip_behind == Some(o.id) {
        true
    } else if engaged && o.zip_behind == Some(c.id) {
        false
    } else if i_yield {
        first || !can_brake(o, p_f, c, p_e - p_f)
    } else if yields_to_me {
        // (my link is open: a foe of the yielding stream is ahead only when it is in the stretch
        // further along than I am, or there first and past stopping short of it (its committed
        // test, `plan::cannot_stop` at `Claim::brake`: SUMO's minor already on its link) - and
        // only when I can brake for it: one I cannot brake for gives way to me, the `i_yield` side)
        let on_link = (p_f >= 0.0 && p_f > p_e) || (first && crate::plan::cannot_stop(o.speed, -p_f, o.brake));
        on_link && can_brake(c, p_e, o, p_f - p_e)
    } else if first {
        // (unless I cannot brake for it and it can for me: then I go first)
        !(!can_brake(c, p_e, o, p_f - p_e) && can_brake(o, p_f, c, p_e - p_f))
    } else {
        // (the same pair from the other side: exactly one of the two is ahead)
        !can_brake(o, p_f, c, p_e - p_f) && can_brake(c, p_e, o, p_f - p_e)
    };
    if !ahead {
        return None;
    }
    let gap = (p_f - o_len) - p_e;
    if o.speed < STANDING_SPEED && p_f >= 0.0 {
        return Some(ZipCap { foe: o.vehicle, gap, speed: 0.0, arrive: f32::INFINITY, floor: -MAX_BRAKE, stop: true, hold: None, inside: true, dist: -p_e, red_short });
    }
    let d_e = -p_e;
    let t_target = arrival_time(o_len - p_f, o.speed, o.accel, o.v_max) + red_shift + c.tau;
    // (inside the stretch already, or there late enough anyway: no arrival bound, the cap is
    // the following of the foe where it stands along the merged way)
    let arrive = if d_e > 0.0 && t_e < t_target {
        let v = c.speed.max(0.0);
        (2.0 * (d_e - v * t_target) / (t_target * t_target)).max(-v * v / (2.0 * d_e))
    } else {
        f32::NEG_INFINITY
    };
    let w = (t_e / ZIPPER_BLEND_TIME).min(1.0);
    let floor = -(w * c.decel + (1.0 - w) * MAX_BRAKE);
    // (short of the start while the foe's rear has not been past it for `tau`: wait at the start;
    // not with the right of way over its stream: the foe is my link leader, and I advance behind
    // it, SUMO's `jmAdvance`, because a hold at the start would let a whole minor stream by)
    let hold = (!yields_to_me && d_e > 0.0 && o.speed >= STANDING_SPEED && o_len - p_f > -o.speed * c.tau).then_some(d_e);
    Some(ZipCap { foe: o.vehicle, gap, speed: o.speed, arrive, floor, stop: false, hold, inside: p_f >= 0.0, dist: -p_e, red_short })
}

/// The car-following model's (IDM's) acceleration for the claimant behind a lead `gap` m
/// ahead at `speed` (as `AiState::desired_accel` without the bend and stop-line terms): which of
/// several zipper foes ahead caps hardest.
fn idm_follow(c: &Claim, gap: f32, speed: f32) -> f32 {
    let (a, b) = (c.accel.max(0.1), c.decel.max(0.5));
    let v = c.speed.max(0.0);
    let v0 = c.v_max.max(0.1);
    let s_star = c.min_gap + (v * c.tau + v * (v - speed.max(0.0)) / (2.0 * (a * b).sqrt())).max(0.0);
    a * (1.0 - (v / v0).powi(4) - (s_star / gap.max(0.05)).powi(2))
}

/// One line of the resolver's account of a claim (`resolve_traced`; `traffic_cases
/// watch`): the foe claimant (the caller's index), the movement it is on, and the times and the
/// case of the time test, or what took it out of the test before it (`skip`).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct FoeTrace {
    pub claimant: usize,
    pub foe: usize,
    pub movement: usize,
    pub times: Option<FoeTimes>,
    pub skip: &'static str,
}

/// Decide every claim. Claims of different junctions do not interact.
///
/// A foe counts only if it can compete: one that follows the claimant, or is behind it on
/// a piece both ways take, is no foe in either pass (`will_pass`). In every pass the claimants
/// standing refused last frame are decided first, and a foe standing outside the meeting place
/// that this pass has refused for a stretch short of it is not coming (SUMO's `willPass`). Two
/// passes when somebody is fed up or the first refused anybody: the second lets the fed-up
/// drivers squeeze in, but not in front of a foe the first pass lets go and that stands at its
/// line (it moves off this very frame: equally far from the meeting place, the two would arrive
/// together); standing refused claimants that are each other's foes are decided in an order
/// (`standing_order`), so that of two who would drop each other only one goes.
pub fn resolve(js: &Junctions, claims: &[Claim]) -> Vec<Grant> {
    resolve_with(js, claims, &Context::default())
}

/// [`resolve`] knowing the light programs' signal groups (`ctx`).
pub fn resolve_with(js: &Junctions, claims: &[Claim], ctx: &Context) -> Vec<Grant> {
    resolve_traced(js, claims, ctx, None)
}

/// [`resolve_with`], with the account of every claim's foes in the pass whose grants are
/// returned (`FoeTrace`) when `trace` is given (`Simulation::trace_foes`, `traffic_cases watch`).
pub fn resolve_traced(js: &Junctions, claims: &[Claim], ctx: &Context, mut trace: Option<&mut Vec<FoeTrace>>) -> Vec<Grant> {
    let plain = decide_in(ctx, js, claims, None, None, trace.as_deref_mut());
    let fed_up = !claims.iter().all(|c| c.impatience.claims_after().is_none() && c.impatience.value < HALF_FED_UP);
    if !fed_up && plain.iter().all(|g| g.go) {
        return plain;
    }
    let standing = standing_order(js, claims, &plain);
    if let Some(t) = trace.as_deref_mut() {
        t.clear();
    }
    decide_in(ctx, js, claims, Some(&plain), Some(&standing), trace)
}

/// A claim as the second pass sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Standing {
    /// Not standing refused at its line (moving, let go, committed, a player).
    No,
    /// Standing, refused for a full exit and not squeezing in: refused in the second pass
    /// too, whatever the others do.
    Hard,
    /// Standing, refused for a foe: it may be let go in the second pass. Contenders of one
    /// junction are decided in this order, lowest first.
    Rank(usize),
}

/// The players' claims (`sim::player_claim`): their grants are not read (the player
/// drives), so a refused player is no proof of anything.
fn is_player(vehicle: usize) -> bool {
    crate::sim::claim_player(vehicle).is_some()
}

/// The standing refused claims of the first pass, and the order of the contenders among
/// them. Between two contenders that are foes, the one that does not give way by the table
/// goes first; when both give way or neither does, the lower vehicle index. A ring of these
/// (A before B before C before A: right before left all round) is cut at its lowest
/// vehicle index, so that one of a ring always goes first.
fn standing_order(js: &Junctions, claims: &[Claim], plain: &[Grant]) -> Vec<Standing> {
    let mut out = vec![Standing::No; claims.len()];
    // contenders by junction (sorted by vehicle index within each)
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for (k, c) in claims.iter().enumerate() {
        if plain[k].go || !c.standing() || c.committed || c.absolute || is_player(c.vehicle) {
            continue;
        }
        if plain[k].hold == Some(Hold::ExitFull) && c.impatience.claims_after().is_none() {
            out[k] = Standing::Hard;
            continue;
        }
        match groups.iter_mut().find(|g| g.0 == c.junction) {
            Some(g) => g.1.push(k),
            None => groups.push((c.junction, vec![k])),
        }
    }
    let mut rank = 0;
    for (j, mut members) in groups {
        members.sort_by_key(|&k| claims[k].vehicle);
        let n = members.len();
        // yields[a][b]: a gives way to b by the table (None: not foes)
        let foes: Vec<Vec<crate::junction::Foe>> = members.iter().map(|&k| js.foes_along(j, &claims[k].chain)).collect();
        let yields = |a: usize, b: usize| -> Option<bool> {
            let mut hit = None;
            for f in &foes[a] {
                if claims[members[b]].chain.iter().any(|&(m, _)| m == f.movement) {
                    hit = Some(hit.unwrap_or(false) || f.yields);
                }
            }
            hit
        };
        // beats[a][b]: a goes before b
        let mut beats = vec![vec![false; n]; n];
        for a in 0..n {
            for b in 0..n {
                if a == b {
                    continue;
                }
                beats[a][b] = match (yields(a, b), yields(b, a)) {
                    (None, None) => false,
                    (Some(ya), Some(yb)) if ya != yb => !ya,
                    // (members are sorted by vehicle index)
                    _ => a < b,
                };
            }
        }
        let mut left: Vec<usize> = (0..n).collect();
        while !left.is_empty() {
            // the first nobody left goes before, else (a ring) the lowest vehicle index
            let pick = left.iter().position(|&a| !left.iter().any(|&b| beats[b][a])).unwrap_or(0);
            let a = left.remove(pick);
            out[members[a]] = Standing::Rank(rank);
            rank += 1;
        }
    }
    out
}

/// The standing claim `k` waits, through the vehicle it follows, for `me`: its lead, or the
/// lead's lead, ... (or, in the second pass, the one such a lead waits for at the junction:
/// the first pass's `wait_for`) is `me`, within four steps. Standing behind a queue that
/// waits for me, it cannot come before I go (three-car rings: I wait for a car behind me on
/// another piece whose lead waits for me). The first step is its lead only: its own wait
/// for me at the junction is the second pass's ordering of standing claimants.
fn held_by(claims: &[Claim], claim_of: &hashbrown::HashMap<usize, usize>, plain: Option<&[Grant]>, k: usize, me: usize) -> bool {
    let mut at = k;
    for _ in 0..LEAD_CHAIN {
        let lead = claims[at].lead;
        let waits = if at == k { None } else { plain.and_then(|p| p[at].wait_for) };
        if lead == Some(me) || waits == Some(me) {
            return true;
        }
        match lead.or(waits).and_then(|v| claim_of.get(&v)) {
            Some(&next) if next != k => at = next,
            _ => return false,
        }
    }
    false
}

/// The standing claim `k` is refused (the first pass's `wait_for`) for a vehicle that itself waits
/// for `me` through its lead chain (`held_by` of that one, or its lead is me): what `k` waits for
/// cannot move before I go, so `k` cannot come before I go either (a ring of three: I wait for
/// `k`, `k` for one standing in its way, that one behind me). Second pass only.
fn waits_through(claims: &[Claim], claim_of: &hashbrown::HashMap<usize, usize>, plain: Option<&[Grant]>, k: usize, me: usize) -> bool {
    let Some(w) = plain.and_then(|p| p[k].wait_for) else { return false };
    if w == me {
        return false;
    }
    match claim_of.get(&w) {
        Some(&next) if next != k => claims[next].standing() && (claims[next].lead == Some(me) || held_by(claims, claim_of, plain, next, me)),
        _ => false,
    }
}

/// A rolling foe `k` is no competitor when it rolls up to a queue that waits for me: its lead
/// chain (up to `LEAD_CHAIN` links) ends at `me` through at least one standing vehicle (every
/// intermediate under 0.3 m/s). It cannot be at the meeting place before the standing one
/// ahead of it moves, and that one waits for me. A foe whose DIRECT lead is me (no standing
/// link between) stays a foe at any speed (`review_lead_is_me_drops_a_rolling_foe_2m_...`:
/// a lead is one frame's perception).
fn held_by_standing(claims: &[Claim], claim_of: &hashbrown::HashMap<usize, usize>, k: usize, me: usize) -> bool {
    let mut at = k;
    for step in 0..LEAD_CHAIN {
        let Some(lead) = claims[at].lead else { return false };
        if lead == me {
            return step >= 1;
        }
        let Some(&next) = claim_of.get(&lead) else { return false };
        if next == k || claims[next].speed >= STANDING_SPEED {
            return false;
        }
        at = next;
    }
    false
}

/// When the standing lead chain of claim `k` lets it go (s from now): the first link standing
/// at its red (`Claim::light_red`) is released at its green (`until_go`; INFINITY: not within a
/// cycle), from rest; a link stuck (people, a zebra, a standing lead: `Claim::stuck`) releases
/// it on no schedule (INFINITY, as `held_off`); a link standing behind a standing lead of
/// its own passes the question on, within `LEAD_CHAIN` links. None: a link held by anything
/// else, without a claim, or the chain too long - its release is not known, and the foe counts
/// as coming from rest now.
fn lead_release(claims: &[Claim], claim_of: &hashbrown::HashMap<usize, usize>, k: usize) -> Option<f32> {
    let mut at = k;
    for _ in 0..LEAD_CHAIN {
        let &next = claim_of.get(&claims[at].lead?)?;
        let l = &claims[next];
        if !l.standing() {
            return None;
        }
        // (one stuck - people, a zebra, a lead that stands too - goes on no schedule: not while
        // that holds, as `held_off` reads it)
        if l.stuck {
            return Some(f32::INFINITY);
        }
        if l.light_red {
            return Some(l.until_go.unwrap_or(f32::INFINITY));
        }
        if !l.lead_rest().is_finite() {
            return None;
        }
        at = next;
    }
    None
}

/// Links of a chain followed by `held_by`, `held_by_standing` and `held_off` (a three-car ring).
const LEAD_CHAIN: usize = 4;

/// The standing claim `k` cannot come while what holds it holds (not coming): it is stuck itself
/// (people, a zebra, a standing lead in front of it), or it stands refused by the junction
/// (the first pass's `wait_for`) for one that is, or for one refused for one that is, within
/// `LEAD_CHAIN` steps. (Not for `me`: a foe waiting for me is `held_by`'s.) Never a player (its
/// grant is not read: it moves off whenever its driver does), and no link is followed from a
/// member that is blocked without being stuck: its refusal says nothing (a blocked claimant
/// drops nobody), what holds it lets it go on a schedule, and the time test decides (a foe 1 m out
/// behind a lead moving off must not be dropped through a stuck third car).
fn held_off(claims: &[Claim], claim_of: &hashbrown::HashMap<usize, usize>, plain: Option<&[Grant]>, k: usize, me: usize) -> bool {
    let mut at = k;
    for step in 0..LEAD_CHAIN {
        let o = &claims[at];
        if !o.standing() || o.committed || is_player(o.vehicle) {
            return false;
        }
        if o.stuck {
            return true;
        }
        if o.blocked {
            return false;
        }
        let Some(w) = plain.and_then(|p| p[at].wait_for) else { return false };
        if w == me {
            return false;
        }
        match claim_of.get(&w) {
            Some(&next) if next != k && step + 1 < LEAD_CHAIN => at = next,
            _ => return false,
        }
    }
    false
}

/// The foe `o` (its claim as it stands on the piece of the meeting place `foe`;
/// `claim` as filed, along the first piece of its chain) comes after me, `c`, whatever its
/// speed or the table: one test, three cases in order -
/// 1. behind me in my own lane (`behind_in_my_lane`), wherever it is;
/// 2. not in the meeting place yet, merging in behind me (`merging_behind`);
/// 3. not in the meeting place yet nor committed to it, behind me on a piece both our ways
///    take (`behind_on_my_way`): one in it or committed is never dropped.
fn behind_me(c: &Claim, o: &Claim, claim: &Claim, foe: &crate::junction::Foe, their_inside: bool, zip_pair: bool) -> bool {
    behind_in_my_lane(c, o, foe.theirs.0 - o.s) || (!their_inside && ((!zip_pair && merging_behind(c, o, foe)) || (!o.committed && behind_on_my_way(c, claim))))
}

/// `o` is behind `c` on its own way: some piece of the junction is on both ways, `c`'s
/// front bumper has reached it (the piece starts at or before `c`'s current place), and
/// `o`'s origin lies further back along that piece than `c`'s. Positions along a piece
/// `p` of a chain are `s - offset(p)`: a claim's `s` is along the first piece of its chain
/// and each chain entry `(piece, offset)` says how far along the way that piece starts.
/// (Two ways that merge into one piece meet there; a car still on its merging piece
/// behind one already into the common piece comes in behind it.) A foe that stands is
/// behind me on a common piece as soon as it is further back along it than I am, before my
/// front reaches it: two cars in one queue before the object, whose ways part on a piece of
/// a fraction of a metre that merges back, are foes of each other by the table. Behind
/// means its front at least a metre behind my rear (its whole length and a metre more
/// behind my rear): two cars level at a merge, centimetres apart, are not one behind the
/// other, and both went.
fn behind_on_my_way(c: &Claim, o: &Claim) -> bool {
    c.chain.iter().any(|&(piece, my_off)| {
        let mine = c.s - my_off;
        (mine + c.front >= 0.0 || o.speed < STANDING_SPEED) && o.chain.iter().any(|&(their_piece, their_off)| their_piece == piece && o.s - their_off + o.front <= mine - c.rear - 1.0)
    })
}

/// `o` is behind `c` in `c`'s own lane: on the lane `c` is on, or on
/// a lane of its way that leads into it, its front at least a metre behind `c`'s rear. It
/// can only come after me, whatever its speed and wherever its way goes on (cars rolling up
/// behind the head of their own queue are "committed" to the merge of the lead-in with
/// its own stub, and must not hold the head at green).
/// Only where its way reaches my lane before its meeting place with me (`their_at`: the
/// distance from its origin to that meeting place): a way that comes back round into my
/// lane after crossing mine is no proof of anything.
fn behind_in_my_lane(c: &Claim, o: &Claim, their_at: f32) -> bool {
    let Some(&(my_lane, my_start)) = c.lanes.first() else { return false };
    // (my origin is `-my_start` along my lane; theirs `-d` where their way reaches it)
    let Some(&(_, d)) = o.lanes.iter().find(|&&(l, _)| l == my_lane) else { return false };
    d < their_at && -d + o.front <= -my_start - c.rear - 1.0
}

/// `o` comes into my way at the merge `foe` behind me: along the
/// merged way its front is at least a metre behind my rear, and it cannot be at the merge's
/// end before my rear is past it, with the margin: the gap I accept when it has the right of
/// way (a side car went in front of a main road car at 13.5 m/s 65 m off on the 0.6 s
/// margin, and the main road car braked 4.5 m/s²). A car merging in behind me follows me,
/// whether or not my front has reached a piece both ways share.
fn merging_behind(c: &Claim, o: &Claim, foe: &crate::junction::Foe) -> bool {
    if foe.kind != FoeKind::Merge {
        return false;
    }
    // (positions along the merged way, 0 at the merge's end)
    let my_rear = (c.s - c.rear) - foe.mine.1;
    let their_front = (o.s + o.front) - foe.theirs.1;
    if their_front > my_rear - 1.0 {
        return false;
    }
    // (not out while people, a zebra or a standing lead stop me inside it, as `foe_times`)
    let stop = c.s + c.clear_ahead;
    let my_out = if stop > foe.mine.0 && stop < foe.mine.1 + c.rear + c.front { f32::INFINITY } else { arrival_time(-my_rear, c.speed, c.accel, c.v_max) };
    let their_at = arrival_time(-their_front, o.speed, o.accel, o.v_max);
    // (SUMO's leader case, `JM_TIMEGAP_MINOR`)
    their_at >= my_out + JM_TIMEGAP_MINOR
}

/// The claimant `c`, eased off for `o` in a mutual hold, follows `o` in where the two merge
/// and `o` rolls (over the standing speed): (the gap from `c`'s front to `o`'s rear along the
/// merged way, `o`'s speed), as the second of a zipper does (`Grant::follow`), when that gap
/// is at least a metre. None at a crossing (nothing to follow in behind) or while `o` stands:
/// then it waits where it is. (The rule: "when a complete stop is expected, roll
/// earlier and let one slow down".)
fn eased_follow(my_foes: &[(usize, crate::junction::Foe)], c: &Claim, o: &Claim) -> Option<(f32, f32)> {
    if o.speed < STANDING_SPEED {
        return None;
    }
    let mut best: Option<(f32, f32)> = None;
    for (_, foe) in my_foes {
        if foe.kind != FoeKind::Merge {
            continue;
        }
        let Some(&(_, off)) = o.chain.iter().find(|&&(m, _)| m == foe.movement) else { continue };
        let gap = (foe.mine.1 - (c.s + c.front)) - (foe.theirs.1 - (o.s - off + o.front)) - (o.front + o.rear);
        if gap >= 1.0 && best.map(|b| gap < b.0).unwrap_or(true) {
            best = Some((gap, o.speed));
        }
    }
    best
}

/// One pass over the claims; `plain` is the first pass's answer in the second pass. Both
/// drop followers and those behind on the same way; `standing` (the second pass) drops
/// standing refused foes, the contenders decided in their order.
#[cfg(test)]
fn decide(js: &Junctions, claims: &[Claim], plain: Option<&[Grant]>, standing: Option<&[Standing]>) -> Vec<Grant> {
    decide_in(&Context::default(), js, claims, plain, standing, None)
}

/// Lights before the table: a foe that my green holds back. My light lets me in (`c` has a
/// light and is not held by it), the foe stands or rolls towards a red that holds it
/// (`light_red`: its driver stops there, wherever it is and however fast), and the two lights
/// never let vehicles in together: it does not come before its green, whatever its speed. Its
/// window is put at its green (`until_go`; no later than its own if it rolls) and it is a
/// competitor only when that still meets mine: some programs have no intergreen at all
/// (e.g. BeijingCity controllers 16 and 19), and a foe 0.2 s from its green at 10 m/s
/// is coming. (A stream braking towards a long red at 1-12 m/s is held back too, not only one
/// standing at it.)
fn red_foe(ctx: &Context, c: &Claim, o: &Claim) -> bool {
    match (c.light, o.light) {
        (Some(mine), Some(theirs)) => !c.light_red && o.light_red && ctx.exclusive(mine, theirs),
        _ => false,
    }
}

/// Does the refused claim `ok` (its grant `g`) come to rest short of `before` (a distance along
/// its own way through, the chain's)? It stops at the last place short of the stretch it waits for
/// (`Grant::at`) where its body lies on none of its crossing meeting places, from a metre before
/// each (`plan::refused_stop`'s waiting place, with the bands the resolver knows: a walk or a
/// convergence moves it further back still, so this errs towards "it comes"); no `at`: held at its
/// line or where it stands. A waiting place that lies behind its front gives it no stop (the plan
/// lets it roll on): it does not rest short. SUMO's willPass per link: a car that waits past a
/// link still asks for it.
fn rests_short(js: &Junctions, claims: &[Claim], ok: usize, g: &Grant, before: f32) -> bool {
    let Some(at) = g.at else { return true };
    let o = &claims[ok];
    let len = o.front + o.rear;
    let bands: Vec<(f32, f32)> = js.foes_along(o.junction, &o.chain).iter().filter(|f| f.kind != FoeKind::Merge).map(|f| (f.mine.0 - 1.0, f.mine.1)).collect();
    // (the front's place at rest, pulled back a band at a time, as `plan::legal_before`)
    let mut p = at - 1.0;
    for _ in 0..64 {
        match bands.iter().filter(|b| b.0 < p && b.1 > p - len - REST_SHORT_OF_STOP).map(|b| b.0).fold(f32::INFINITY, f32::min) {
            x if x.is_finite() => p = x,
            _ => break,
        }
        if p <= before {
            break;
        }
    }
    // (a waiting place behind its front is none: the plan gives it no stop there and it rolls
    // on, `plan::STOP_BEHIND_FRONT`)
    if p < o.s + o.front - crate::plan::STOP_BEHIND_FRONT {
        return false;
    }
    p <= before
}

/// The metre a car comes to rest short of its stop (`plan::REST_SHORT`): its body must lie on no
/// band that far back too.
const REST_SHORT_OF_STOP: f32 = 1.0;

/// What [`will_pass`] makes of a foe: not coming (why, `FoeTrace::skip`), or its window at the
/// stretch (in, out; s from now) as the resolver reckons it, and whether it is held at a red that
/// my green holds back (`red_foe`).
enum Pass {
    /// It has left the meeting place.
    Left,
    Not(&'static str),
    /// Its window, whether it is held at a red my green holds back, and whether its front is in
    /// the stretch.
    Window { their_in: f32, their_out: f32, at_red: bool, their_inside: bool },
}

/// The one foe-eligibility test of the resolver (SUMO's `MSLink::blockedByFoe`, whose first
/// line drops a foe that will not pass, `willPass`, and whose times are the foe's planned
/// arrival and leave): whether the foe `o` (its claim `ok`, shifted onto the piece of the meeting
/// place `foe`) can come into my, `c`'s, stretch before I am through, and when. Both the time
/// test and the zipper read its answer (`decide_in`): a foe that will not pass is nobody's
/// leader at a merge either. `their_inside`: its front is in the stretch; `me_in`: mine is (at a
/// crossing); `zip_pair`: the two ways merge and cross nowhere else.
#[allow(clippy::too_many_arguments)]
fn will_pass(ctx: &Context, js: &Junctions, claims: &[Claim], claim_of: &hashbrown::HashMap<usize, usize>, plain: Option<&[Grant]>, decided: &[Option<Grant>], c: &Claim, o: &mut Claim, ok: usize, foe: &crate::junction::Foe, me_in: bool, zip_pair: bool) -> Pass {
    // (its rear past the stretch: gone)
    if o.s - o.rear > foe.theirs.1 {
        return Pass::Left;
    }
    let their_inside = o.s + o.front >= foe.theirs.0;
    // committed is to a meeting place it cannot stop short of (SUMO's willPass is per link, from
    // the braking distance to that link): one standing outside this one is not on its way into
    // it, it comes as its window from standing says; and one rolling is committed only to the
    // meeting places short of its first waiting place (`Claim::next_wait`) - beyond that the
    // table and the time test decide (otherwise it waits for oncoming left-turners
    // its table gives way to, committed only to their first meeting place well before its own)
    if o.speed < STANDING_SPEED && !their_inside {
        o.committed = false;
    }
    let their_wait = claims[ok].next_wait - (claims[ok].s - o.s);
    if o.committed && !their_inside && their_wait.is_finite() && their_wait <= foe.theirs.0 {
        o.committed = false;
    }
    // clearing the box (`Claim::clearing`), or in the box at the end of its green against the
    // fresh green across its way (a foe whose light is never green with its own), it is
    // committed: SUMO's vehicle already on the junction is never blocked by a link that just
    // opened (the car in the box clears first). Not while it stands outside this meeting
    // place stuck (people, a standing lead: it is not coming while that holds), nor while it
    // stands outside it waiting for me through its lead or the junction's answer (`held_by`:
    // it is not coming before I go) - both made rings of mutual waits
    // (refused this pass, standing outside this meeting place, coming to rest short of it -
    // `rests_short`: a refused car stops at its last waiting place short of the stretch it waits
    // for, `Grant::at`; none, held at its line or where it stands - `refused`
    // below; and in the box at the end of its green it is no car clearing the box: one on its fresh green
    // would wait for a car itself refused for a queue beyond. One
    // refused for a stretch past this one still drives through this one to wait there, so it
    // counts, as SUMO's car allowed to wait at its internal junction still asks for the entry link)
    let stops_short = |g: &Grant| rests_short(js, claims, ok, g, foe.theirs.0 + (claims[ok].s - o.s));
    let refused_here = !their_inside && o.standing() && !is_player(o.vehicle) && !(o.blocked && !o.stuck) && decided[ok].is_some_and(|g| !g.go && stops_short(&g));
    let fresh_green = |a: &Claim, b: &Claim| matches!((a.light, b.light), (Some(x), Some(y)) if ctx.exclusive(x, y));
    if (o.clearing || (o.in_box_at_turn && fresh_green(c, o) && !refused_here)) && !(o.stuck && !their_inside) && !(o.speed < STANDING_SPEED && !their_inside && (held_by(claims, claim_of, plain, ok, c.vehicle) || waits_through(claims, claim_of, plain, ok, c.vehicle))) {
        o.committed = true;
    }
    // in the box at the end of my green, a foe on the fresh green across my way that is neither
    // in the meeting place nor unable to stop short of it waits for me: no foe of mine
    if c.in_box_at_turn && fresh_green(c, o) && !their_inside && !o.committed {
        return Pass::Not("clears_first");
    }
    let o = &*o;
    // clearing the box myself: only those in the meeting place, committed or clearing, hold me
    if c.clearing && !their_inside && !o.committed {
        return Pass::Not("clearing");
    }
    // lights before the table: held at a red that is never green with mine (short of its line,
    // `red_at`), it comes at its green (`red_foe`); with no clock, not at all
    let at_red = !their_inside && o.red_at <= foe.theirs.0 && red_foe(ctx, c, o);
    if at_red && o.until_go.is_none() {
        return Pass::Not("red");
    }
    if me_in && !their_inside {
        return Pass::Not("me_in");
    }
    // behind me (in my own lane, merging in behind me, or on a piece both our ways
    // take): it comes after me at any speed, `behind_me`
    if behind_me(c, o, &claims[ok], foe, their_inside, zip_pair) {
        return Pass::Not("behind");
    }
    let (mut their_in, mut their_out) = window(o, foe.theirs.0, foe.theirs.1);
    // my own follower comes after me whatever the table says, one in the meeting place or
    // committed to it excepted: standing with me as its lead (or waiting for me through the
    // chain, `held_by`) - standing with the claim's hysteresis, `Claim::standing`: one that has
    // just crept off after me under 1 m/s is still my follower, because counted as coming it would
    // refuse me while it follows me in and the two bodies lock; a lead is one frame's
    // perception, so one rolling faster is not taken on it - or rolling up to a queue whose lead
    // chain reaches me through a standing vehicle (`held_by_standing`)
    if !their_inside && !o.committed && ((o.standing() && (claims[ok].lead == Some(c.vehicle) || held_by(claims, claim_of, plain, ok, c.vehicle) || waits_through(claims, claim_of, plain, ok, c.vehicle))) || held_by_standing(claims, claim_of, ok, c.vehicle)) {
        return Pass::Not("waits_for_me");
    }
    // a foe at its red is not coming before its light goes (SUMO's foe arrival is its planned
    // one, and it plans to stop at its red): standing (creeping up under 1 m/s after a stand
    // counts, `Claim::was_standing`), from rest at its green; rolling, not before its green nor
    // before its own window says
    // (the seconds its start from rest is put off by: its own red's, below; its lead's release,
    // further below, adds only what lies beyond it)
    let mut released_at = 0.0f32;
    if at_red {
        let t = o.until_go.unwrap_or(f32::INFINITY);
        let standing = o.standing();
        let shift = if standing { t } else { (t - their_in).max(0.0) };
        their_in += shift;
        their_out += shift;
        released_at = shift;
    } else if o.light_red && !their_inside && o.red_at <= foe.theirs.0 {
        // (the same at a red that does not exclude mine, held short of this meeting place)
        match o.until_go {
            Some(t) => {
                let shift = if o.standing() { t } else { (t - their_in).max(0.0) };
                their_in += shift;
                their_out += shift;
                released_at = shift;
            }
            None => {
                return Pass::Not("red");
            }
        }
    }
    // a foe standing outside the meeting place at a stop it serves (`Claim::serving`) is not
    // approaching the link (SUMO: a vehicle stopped at a stop has no request until its stop
    // ends; its dwell's end is not known to the claim, so it comes, from rest, once it moves)
    if !their_inside && !o.committed && o.standing() && o.serving {
        return Pass::Not("at_stop");
    }
    // a foe standing outside the meeting place, held there by people, a zebra or a lead that
    // stands too, or standing refused for one so held (`held_off`), is not coming while that
    // holds, however near it stands; when it moves off it is a competitor again. Not when I am
    // held so myself: my window is no more real than its, and two such would let each other go
    if !their_inside && !o.committed && !c.blocked && held_off(claims, claim_of, plain, ok, c.vehicle) {
        return Pass::Not("held_off");
    }
    // SUMO's `setRequest` (MSVehicle.cpp: a link is requested only when the planned speed after
    // the leader constraint is above `NUMERICAL_EPS_SPEED`) and `willPass`, a statement of time
    // re-made every step: a foe standing outside the meeting place whose front comes to rest
    // short of it behind a standing lead or a body across its way (`Claim::lead_rest`), whatever
    // its why names (a car at its zipper waiting place 2 m behind a lead held at red is labelled
    // Junction, neither blocked nor stuck), starts no sooner than its lead chain is released
    // (`lead_release`: the first of the chain standing at a red, at its green, from rest): its
    // window from rest is put off to then, and the time test
    // decides. A chain held by anything else, or not known, lets it come from rest now; one whose
    // release never comes within a cycle is not coming
    if !their_inside && !o.committed && o.standing() && o.lead_rest() <= foe.theirs.0 {
        match lead_release(claims, claim_of, ok) {
            Some(t) if t.is_infinite() => return Pass::Not("stops_short"),
            Some(t) => {
                let extra = (t - released_at).max(0.0);
                their_in += extra;
                their_out += extra;
            }
            None => {}
        }
    }
    // SUMO's `willPass` (blockedByFoe drops a foe whose request is not to pass: its planned speed
    // at the link is 0): a foe standing outside the meeting place, not committed to it, that this
    // pass has refused already for a stretch at or short of this one (`refused_here`: decided
    // before me - the claimants standing refused last frame are decided first, `decide_in`) comes
    // to rest short of this place - it is not coming. One not decided yet counts (its answer may
    // be a go; also when it is refused afterwards for a third, a hold SUMO would not have: the
    // order cannot know it); a player's claim always counts (its grant is not read); so does one
    // blocked by what lets it go on a schedule (a lead moving off): its refusal may be its
    // block's (a blocked claimant drops nobody), the time test decides
    if refused_here && !o.committed {
        return Pass::Not("refused");
    }
    // (one held by something that lets it go on a schedule, a light or a lead moving off, counts
    // with its window from rest, as SUMO's `blockedByFoe` reckons a foe's arrival)
    // standing in a merge's meeting place behind me (its front further from the merge's end than
    // my rear) with me as its lead (through standing links, `held_by_standing`): it stopped behind
    // me and waits for me, it is not in my way; level with me or ahead, it holds me
    let behind_me = foe.kind == FoeKind::Merge && foe.theirs.1 - (o.s + o.front) > foe.mine.1 - (c.s - c.rear);
    if their_inside && o.speed < STANDING_SPEED && behind_me && (claims[ok].lead == Some(c.vehicle) || held_by_standing(claims, claim_of, ok, c.vehicle)) {
        return Pass::Not("stopped_behind_me");
    }
    // in the meeting place of a crossing, standing with my body as its lead or body block, while
    // its body is neither of mine: it is physically behind me and waits for me, not in my way -
    // SUMO's link leader of two on the junction is the one in front (MSLink::getLeaderInfo), and
    // my perception keeps me off its body. (Two long ways through one box each occupy the other's
    // next meeting place: one standing behind the other would hold it at the place it stands in
    // while it waits for the other at the next.)
    // (behind me as its perception has it: me its lead at a gap, or my body blocking its way)
    let its_lead_is_me = (claims[ok].lead == Some(c.vehicle) && claims[ok].lead_gap.is_some_and(|(gap, _)| gap >= 0.0)) || claims[ok].body_block == Some(c.id);
    let mine_is_it = c.lead == Some(o.vehicle) || c.body_block == Some(o.id);
    if their_inside && foe.kind != FoeKind::Merge && o.standing() && its_lead_is_me && !mine_is_it {
        return Pass::Not("stopped_behind_me");
    }
    // standing in the meeting place held by its plan (a lead, its stop): it does not leave on the
    // schedule its acceleration would give it - at its red past this place it leaves at its green
    // and its window from rest, else not (OMSI's reason: SUMO's foe carries its planned leave
    // time, which no claim knows)
    if their_inside && o.speed < STANDING_SPEED && o.held {
        their_out = match o.until_go.filter(|_| o.light_red && o.red_at > foe.theirs.0) {
            Some(t) => t + window(o, foe.theirs.1, foe.theirs.1).1,
            None => f32::INFINITY,
        };
    }
    Pass::Window { their_in, their_out, at_red, their_inside }
}

fn decide_in(ctx: &Context, js: &Junctions, claims: &[Claim], plain: Option<&[Grant]>, standing: Option<&[Standing]>, mut trace: Option<&mut Vec<FoeTrace>>) -> Vec<Grant> {
    let mut out: Vec<Option<Grant>> = vec![None; claims.len()];
    // (the contenders first, in their order: each sees whether the ones before it go)
    // (then the claimants standing refused last frame, in vehicle order: their answer of this pass
    // is known to the foes decided after them, `will_pass`; a foe not decided yet counts as
    // coming, so two that would drop each other never both go)
    let mut order: Vec<usize> = (0..claims.len()).collect();
    let refused_standing = |k: usize| claims[k].refused_last && claims[k].standing();
    order.sort_by_key(|&k| {
        let rank = match standing.map(|st| st[k]) {
            Some(Standing::Rank(r)) => r,
            _ => usize::MAX,
        };
        (rank, !refused_standing(k))
    });
    // a claim stands for the whole way through: the piece it names and the pieces that
    // one continues into, each as far along as its start lies (claim, piece, shift)
    // (by junction and piece, in claim order: each foe looks up its own; scanning them all for
    // every foe of every claim was most of the resolver's time once merges were foes too)
    let mut expanded: hashbrown::HashMap<(usize, usize), Vec<(usize, f32)>> = hashbrown::HashMap::new();
    for (k, c) in claims.iter().enumerate() {
        for &(m, off) in &c.chain {
            expanded.entry((c.junction, m)).or_default().push((k, off));
        }
    }
    // claim of each vehicle, for following a chain of leads
    let claim_of: hashbrown::HashMap<usize, usize> = claims.iter().enumerate().map(|(k, c)| (c.vehicle, k)).collect();
    for &k in &order {
        let c = &claims[k];
        let node = &js.nodes[c.junction];
        let my_foes = js.foes_along_pieces(c.junction, &c.chain);
        let squeeze = if plain.is_some() { c.impatience.claims_after() } else { None };
        // (held at my red I am held by my light, and no foe matters: the plan stops me at
        // the line, before any meeting place)
        // (eased off in a mutual hold: it waits for the other, where it stands; where the other
        // rolls on into a merge of the two ahead of it, it follows it in rather than standing,
        // `eased_follow`)
        // (the follow only when no other foe holds it: the rest of its foes are
        // decided as for anybody, below; one that refuses it keeps it where it stands.
        // Following its partner in, an eased car would roll into a third merging between them)
        let eased_for = c.yields_now;
        // held at a red line that lies before every meeting place still ahead of it, the
        // light holds it and no foe matters; a red past a meeting place holds it only before the
        // meeting places beyond its line (a car its own red holds must not go through the meeting places short of that red)
        let first_ahead = my_foes.iter().filter(|(_, f)| c.s - c.rear <= f.mine.1).map(|(_, f)| f.mine.0).fold(f32::INFINITY, f32::min);
        let red_before_all = c.light_red && c.red_at <= first_ahead;
        // (committed, it goes, but a zipper's foe ahead still caps its speed: the foes are
        // looked at for that alone)
        let go_anyway = eased_for.is_none() && c.committed && !c.absolute && !red_before_all;
        if eased_for.is_none() && !go_anyway {
            if c.committed || c.absolute || red_before_all {
                out[k] = Some(Grant::go(c.vehicle));
                continue;
            }
            if !c.exit_room && squeeze.is_none() {
                out[k] = Some(Grant::refused(c.vehicle, None, Hold::ExitFull));
                continue;
            }
        }
        let mut wait_for: Option<(usize, f32)> = None;
        // the zipper speed for the foe ahead at a merge that crosses nothing
        let mut zip_cap: Option<ZipCap> = None;
        // the first stretch waited for (the car stops just before it)
        let mut hold_at = f32::INFINITY;
        let (mut follower_go, mut leader_go) = (0u16, 0u16);
        for (my_piece, foe) in &my_foes {
            // a foe movement that is a piece of my own way through is ahead
            // of me on my own chain (a lane and its successor): whoever is on it is my lead,
            // never my junction foe (it would wait for a car standing ahead on its own successor)
            if c.chain.iter().any(|&(m, _)| m == foe.movement) {
                continue;
            }
            // past the red line that holds me: I stop at the line before it
            if c.light_red && foe.mine.0 >= c.red_at {
                continue;
            }
            let my_piece = *my_piece;
            // a meeting place I am in already: I am in the way of those coming, not they in
            // mine (a car standing in the box short of a later stretch it waits for is
            // committed to the stretches it is in, not to the later ones: `Claim::committed`)
            // (a crossing: two lanes running into one another negotiate who is first at the end
            // of their merge, however long the stretch where they converge)
            // (in it as a foe sees one in it, `their_inside`: a front half a metre short of it
            // counted as in, the foe saw it out, and both went)
            let me_in = foe.kind != FoeKind::Merge && c.s + c.front >= foe.mine.0 && c.s - c.rear <= foe.mine.1;
            // (one it cannot wait short of, `Claim::next_wait`: it holds its way
            // through it unless the foe is in it or arriving while it would be - then it waits)
            let wait_short_of = foe.kind != FoeKind::Merge && !me_in && foe.mine.0 < c.next_wait;
            let (my_in, _) = window(c, foe.mine.0, foe.mine.1);
            for &(ok, shift) in expanded.get(&(c.junction, foe.movement)).map(|v| v.as_slice()).unwrap_or(&[]).iter().filter(|&&(k, _)| claims[k].vehicle != c.vehicle && Some(claims[k].vehicle) != eased_for) {
                // (one eased off for me in a mutual hold: not in my way at the junction; its
                // body still is, to the plan)
                if claims[ok].yields_now == Some(c.vehicle) {
                    continue;
                }
                // the foe as it stands on this piece of its way
                let mut shifted = claims[ok].clone();
                shifted.s -= shift;
                shifted.red_at -= shift;
                // (a claimant refused for one standing in the meeting place stops at the stretch,
                // as every refused one does, `plan::refused_stop`, not at the line; the time test
                // has no reach: a foe is decided at any distance)
                // two ways running into one another (or side by side) that cross nowhere
                // else: SUMO's zipper (`getZipperSpeed`), never a refusal, the foe ahead caps my
                // speed (`zipper_cap`)
                let zip_pair = foe.kind == FoeKind::Merge && !my_foes.iter().any(|(_, g)| g.kind != FoeKind::Merge && claims[ok].chain.iter().any(|&(m, _)| m == g.movement));
                if go_anyway && !zip_pair {
                    continue;
                }
                let pass = will_pass(ctx, js, claims, &claim_of, plain, &out, c, &mut shifted, ok, foe, me_in, zip_pair);
                let o = &shifted;
                // (the account of this foe for `resolve_traced`)
                let mut note = |skip: &'static str, times: Option<FoeTimes>| {
                    if let Some(t) = trace.as_deref_mut() {
                        t.push(FoeTrace { claimant: c.vehicle, foe: o.vehicle, movement: foe.movement, times, skip });
                    }
                };
                let (their_in, their_out, at_red, their_inside) = match pass {
                    Pass::Left => continue,
                    Pass::Not(skip) => {
                        note(skip, None);
                        continue;
                    }
                    Pass::Window { their_in, their_out, at_red, their_inside } => (their_in, their_out, at_red, their_inside),
                };
                // SUMO's time test (`blockedByFoe`, `foe_times`): follower, leader or a
                // conflict, `JM_TIMEGAP_MINOR` either way, the foe's arrival as my impatience
                // sees it. (One held by its lead, people or a light outside the meeting place is
                // not through it before the other comes on the strength of a start from rest now:
                // it starts when what holds it lets it. Not the leader myself when I am held so,
                // nor the foe the one I follow when it is, or when it brakes for its red: a side
                // road car went behind one standing a metre from the meeting place, once the other
                // main road car it waited for was dropped; a car held by its lead, "first" by a
                // start it could not make, went past the contender before it in the order.)
                if zip_pair {
                    // (one order per pair of ways: the pair's first merge stretch
                    // that neither has left decides, for every stretch of the two - per-stretch
                    // answers put either first at two stretches of one pair)
                    let decisive = my_foes.iter().find(|(_, f)| f.kind == FoeKind::Merge && c.s - c.rear <= f.mine.1 && claims[ok].chain.iter().any(|&(m, off)| m == f.movement && claims[ok].s - off - claims[ok].rear <= f.theirs.1)).map(|(_, f)| *f);
                    if decisive.is_some_and(|d| d != *foe) {
                        note("zipper_later_stretch", None);
                        continue;
                    }
                    // (the foe's arrival as the resolver has it - at its green, from its stand -
                    // and the order the table gives, `zipper_cap`)
                    let yields_to_me = !foe.yields && node.movements[foe.movement].foes.iter().any(|f| f.movement == my_piece && f.yields);
                    let red_shift = their_in - window(o, foe.theirs.0, foe.theirs.1).0;
                    match zipper_cap(c, o, foe, their_in, red_shift, foe.yields, yields_to_me) {
                        Some(z) => {
                            note("zipper_ahead", None);
                            // (of several foes ahead, the one whose cap is the hardest: the cap as the
                            // plan will take it, with the car-following model's answer for the foe as
                            // a lead, `idm_follow`; the queue-entry stop first)
                            let cap = |z: &ZipCap| idm_follow(c, z.gap, z.speed).max(z.arrive).max(z.floor);
                            let tighter = match zip_cap {
                                None => true,
                                Some(w) => (z.stop && !w.stop) || (z.stop == w.stop && (if z.stop { z.gap < w.gap } else { cap(&z) < cap(&w) })),
                            };
                            if tighter {
                                zip_cap = Some(z);
                            }
                        }
                        None => note("zipper_behind", None),
                    }
                    continue;
                }
                let times = foe_times(c, o, foe.mine, foe.theirs, (their_in, their_out), foe.kind == FoeKind::Merge);
                if wait_short_of && !their_inside && times.case != FoeCase::Conflict {
                    note("committed_short_of", Some(times));
                    continue;
                }
                // (a blocked claimant too: its window comes from its lead's motion,
                // `planned_window`, or it does not come; otherwise any claimant would go ahead of a foe at
                // its red whose window it cleared)
                let i_am_first = times.case == FoeCase::Leader && !their_inside && !o.committed;
                // (not coming myself - stopped short of the stretch or held in it, `foe_times` -
                // nothing the foe does makes it a conflict: its schedule does not matter)
                // (and the foe passes before me only when it is on its way - SUMO counts only foes
                // that will pass, `willPass`: one standing outside the meeting place goes only if
                // the junction lets it, which this pass is deciding; for two such the table and
                // the standing order decide)
                let passing = o.speed >= STANDING_SPEED || their_inside || o.committed;
                let they_are_first = times.case == FoeCase::Follower && (times.my_in.is_infinite() || (passing && !(o.blocked && !their_inside) && !at_red));
                // the two are clear of each other in time whatever the rules say
                if i_am_first {
                    leader_go += 1;
                    note("", Some(times));
                    continue;
                }
                if they_are_first {
                    follower_go += 1;
                    note("", Some(times));
                    continue;
                }
                note("", Some(FoeTimes { case: FoeCase::Conflict, ..times }));
                // (the right of way: a train's, the table's, or between equals the first to come)
                let theirs_first = o.absolute || (foe.yields && !c.absolute) || (!foe.yields && !node.movements[foe.movement].foes.iter().any(|f| f.movement == my_piece && f.yields) && their_in < my_in);
                // (two fed-up drivers must not squeeze in on each other: all four at an equal
                // junction became fed up and went at once, into each other. The one with the
                // lower index takes it first, the other waits for it.)
                let goes_now = o.speed < 1.0 && !o.light_red && (plain.map(|p| p[ok].go).unwrap_or(false) || (standing.is_some() && out[ok].map(|g| g.go).unwrap_or(false)));
                // (never into a stream: a fed-up driver squeezes in front of a foe that is
                // slow or still well away, not across one doing town speed 35 m off. A
                // turner facing a green stream waits for it to stop at its red.)
                let far_enough = their_in > after_for(o.speed, squeeze);
                // (four cars arriving together at an unmarked junction all give way to
                // the right: when the foe stands and waits as well, half the patience is
                // enough to go first, as a nod across the junction settles it in life)
                let mutual_wait = plain.is_some() && c.impatience.value >= HALF_FED_UP && o.speed < STANDING_SPEED && !their_inside && !o.committed;
                let fed_up = (squeeze.is_some() && far_enough) || mutual_wait;
                let take_anyway = fed_up && !their_inside && !o.committed && !goes_now && !(o.impatience.value >= HALF_FED_UP && o.vehicle < c.vehicle);
                // (second pass) I stand refused at my line; a foe that
                // stands outside the meeting place and is refused itself is not coming: one
                // refused for a full exit stays refused, a contender after me in the order
                // waits for me if I go, one before me that stays refused does not go.
                // Only a contender before me that goes this pass holds me.
                if let Some(Standing::Rank(mine)) = standing.map(|st| st[k]) {
                    if o.standing() && !their_inside && !o.committed {
                        match standing.map(|st| st[ok]) {
                            // refused for a full exit: not coming while its exit is full
                            // this frame (it counts again the frame its exit has room)
                            Some(Standing::Hard) if !o.exit_room => continue,
                            Some(Standing::Rank(theirs)) => {
                                let me_inside = c.s + c.front >= foe.mine.0;
                                let ahead_goes = theirs < mine && out[ok].map(|g| g.go).unwrap_or(false);
                                // (one refused while it is blocked by what lets it go on a
                                // schedule, a lead moving off, is not "staying refused": the
                                // ordinary rules decide, its window from rest after its start)
                                let on_schedule = o.blocked && !o.stuck;
                                // (one before me that stays refused comes to rest short of the
                                // stretch it waits for, `Grant::at`: past this place, it still
                                // drives through this one to wait there, and counts)
                                let stops_short = theirs > mine || out[ok].is_some_and(|g| rests_short(js, claims, ok, &g, foe.theirs.0 + (claims[ok].s - o.s)));
                                if me_inside || (!ahead_goes && !on_schedule && stops_short) {
                                    continue;
                                }
                                if ahead_goes {
                                    if !take_anyway && wait_for.map(|w| their_in < w.1).unwrap_or(true) {
                                        wait_for = Some((o.vehicle, their_in));
                                    }
                                    if !take_anyway {
                                        hold_at = hold_at.min(foe.mine.0);
                                    }
                                    continue;
                                }
                            }
                            _ => {}
                        }
                    }
                }
                // two both inside the meeting place, each in the other's way: one goes first, the
                // other waits for it - the one the table gives the right of way, between equals (or
                // a pair that yields both ways) the one further along, less of its stretch left to
                // clear (SUMO's link leader of two on the junction, MSLink::getLeaderInfo), a tie
                // the lower index. Without it each would let the other out first and both stand (an
                // opposing left-turner standing in the box at the end of its green). Inside both is geometry, the same from
                // either side, so exactly one of the two goes.
                let me_inside = c.s + c.front >= foe.mine.0 && c.s - c.rear <= foe.mine.1;
                if me_inside && their_inside && !o.absolute && !c.absolute {
                    let they_yield = node.movements[foe.movement].foes.iter().any(|f| f.movement == my_piece && f.yields);
                    let (my_left, their_left) = (foe.mine.1 - (c.s - c.rear), foe.theirs.1 - (o.s - o.rear));
                    let i_lead = if foe.yields != they_yield { they_yield } else { my_left < their_left || (my_left == their_left && c.vehicle < o.vehicle) };
                    if i_lead {
                        continue;
                    }
                }
                if theirs_first && !take_anyway {
                    // the right of way is theirs, and the time test found neither me out before
                    // they come nor them out before I come (at a merge, one lane after it, SUMO's
                    // `unsafeMergeSpeeds` decides, `foe_times`)
                    if wait_for.map(|w| their_in < w.1).unwrap_or(true) {
                        wait_for = Some((o.vehicle, their_in));
                    }
                    hold_at = hold_at.min(foe.mine.0);
                } else if (their_inside || (o.committed && times.their_in <= times.my_out)) && !(c.clearing && o.clearing && !claims[ok].committed && !their_inside) {
                    // (committed but coming only after I am out: no foe of mine - SUMO's
                    // major link does not look at a minor one at all, and the minor one took the
                    // gap behind me as its follower; a 13 m/s main road car braked 7 m/s² for a
                    // side car following it 0.1 s behind)
                    // mine, but they are already in the way: let them out first (not, clearing
                    // the box myself, one clearing it too that is not in the meeting place: the
                    // table decides between two such, and it gives way to me; each let the other
                    // out first and both stood)
                    if wait_for.map(|w| their_out < w.1).unwrap_or(true) {
                        wait_for = Some((o.vehicle, their_out));
                    }
                    hold_at = hold_at.min(foe.mine.0);
                }
            }
        }
        if let Some(w) = eased_for {
            // (not with a zipper cap for a third either: it keeps the car where it stands)
            let follow = if wait_for.is_some() || zip_cap.is_some() { None } else { claim_of.get(&w).and_then(|&ok| eased_follow(&my_foes, c, &claims[ok])) };
            out[k] = Some(Grant { follow, ..Grant::refused(c.vehicle, Some(w), Hold::Foe) });
            continue;
        }
        out[k] = Some(match wait_for {
            Some((v, _)) => Grant { at: hold_at.is_finite().then_some(hold_at), zipper: zip_cap, ..Grant::refused(c.vehicle, Some(v), Hold::Foe) },
            None => Grant { zipper: zip_cap, ..Grant::go(c.vehicle) },
        });
        if let Some(g) = out[k].as_mut() {
            g.follower_go = follower_go;
            g.leader_go = leader_go;
        }
    }
    out.into_iter().map(|g| g.expect("every claim decided")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::{DriverParams, VehicleClass};
    use glam::DVec3;
    use omsi_sim::traffic::{LaneBuilder, LaneKey, LaneKind, Network};

    fn piece(from: (f64, f64), to: (f64, f64), path: u16, priority: f32) -> omsi_sim::traffic::Lane {
        let mut l = LaneBuilder::polyline(vec![DVec3::new(from.0, from.1, 0.0), DVec3::new(to.0, to.1, 0.0)], LaneKind::Street, 3.0);
        l.source = 2;
        l.key = Some(LaneKey { tile: (0, 0), id: 7, path });
        l.priority = priority;
        l
    }

    fn four_way(main_prio: f32, side_prio: f32) -> (Network, Junctions) {
        let mut net = Network::default();
        net.lanes.push(piece((-10.0, 0.0), (10.0, 0.0), 0, main_prio));
        net.lanes.push(piece((0.0, -10.0), (0.0, 10.0), 1, side_prio));
        net.link(1.5);
        net.compute_conflicts();
        let js = Junctions::compile(&net);
        (net, js)
    }

    fn claim(vehicle: usize, movement: usize, s: f32, speed: f32) -> Claim {
        Claim { yields_now: None, clear_ahead: f32::INFINITY, serving: false, brake: 5.6, vehicle, junction: 0, movement, s, speed, accel: 1.8, v_max: 13.0, front: 2.0, rear: 2.0, decel: 2.0, waiting: 0.0, lead_gap: None, zip_behind: None, refused_last: false, body_block: None, next_wait: f32::NEG_INFINITY, tau: 1.3, min_gap: 2.0, red_at: 0.0, committed: false, exit_room: true, impatience: Impatience::default(), absolute: false, held: false, light_red: false, until_go: None, chain: vec![(movement, 0.0)], lead: None, was_standing: false, blocked: false, lanes: Vec::new(), light: None, id: vehicle as u64, stuck: false, clearing: false, in_box_at_turn: false }
    }

    /// A crossing object of `pieces` (polylines; each with its turn), equal priorities.
    fn object(pieces: &[(&[(f64, f64)], i32)]) -> (Network, Junctions) {
        let mut net = Network::default();
        for (k, (pts, turn)) in pieces.iter().enumerate() {
            let mut l = LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), LaneKind::Street, 3.0);
            l.source = 2;
            l.key = Some(LaneKey { tile: (0, 0), id: 7, path: k as u16 });
            l.priority = 128.0;
            l.turn = *turn;
            net.lanes.push(l);
        }
        net.link(1.5);
        net.compute_conflicts();
        let js = Junctions::compile(&net);
        (net, js)
    }

    /// A (0) and B (1, from A's right at 37 degrees) merge where C (2) begins, and A gives
    /// way to B. (By the fallbacks of equal priorities B, coming in at an angle to
    /// the way out, would give way to A going straight on; B carries a higher `[rule]
    /// priority` here to keep the table these competitor-filter tests are about.)
    fn merge_a_gives_way() -> (Network, Junctions) {
        let (mut net, _) = object(&[(&[(-10.0, 0.0), (0.0, 0.0)], 0), (&[(-8.0, -6.0), (0.0, 0.0)], 0), (&[(0.0, 0.0), (10.0, 0.0)], 0)]);
        net.lanes[1].priority = 192.0;
        let js = Junctions::compile(&net);
        (net, js)
    }

    fn grant_of(g: &[Grant], vehicle: usize) -> Grant {
        *g.iter().find(|g| g.vehicle == vehicle).unwrap()
    }

    /// `v` goes ahead of nobody: refused, or let go only as the follower (behind them).
    fn not_ahead_of(g: &[Grant], v: usize) -> bool {
        let g = grant_of(g, v);
        !g.go || g.leader_go == 0
    }

    /// `v` gives way to `foe`: refused for it, or  capped by it at a zipper.
    fn gives_way(g: &[Grant], v: usize, foe: usize) -> bool {
        let g = grant_of(g, v);
        g.wait_for == Some(foe) || g.zipper.is_some_and(|z| z.foe == foe)
    }

    #[test]
    fn competitor_filter_my_follower_in_the_same_lane_is_no_foe() {
        // one lane forks inside the object: the straight way (0) and the left turn (1),
        // which gives way to it where the two overlap at the start
        let (_, js) = object(&[(&[(-10.0, 0.0), (10.0, 0.0)], 0), (&[(-10.0, 0.0), (-4.0, 0.0), (0.0, 4.0), (0.0, 10.0)], 1)]);
        assert!(js.nodes[0].movements[1].foes[0].yields);
        // the turner stands with its nose over the line, the car behind it (going
        // straight on, following the turner) 2 m behind its rear
        let turner = claim(1, 1, -1.0, 0.0);
        let mut follower = claim(2, 0, -7.0, 0.0);
        follower.lead = Some(1);
        let claims = [turner, follower];
        let g = resolve(&js, &claims);
        assert!(grant_of(&g, 1).go, "{:?}", g);
        // ... unless the follower is already in the meeting place
        let mut inside = claims.clone();
        inside[1].s = -1.5;
        let g = resolve(&js, &inside);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2));
    }

    #[test]
    fn competitor_filter_a_car_behind_on_a_merging_piece_is_no_foe() {
        // A (0) and B (1) merge where C (2) begins; A gives way to B
        let (_, js) = merge_a_gives_way();
        assert!(js.nodes[0].movements[0].foes[0].yields);
        // the car on A has its nose on C already; the one on B stands 9 m back along the
        // common way, short of the meeting place (no lead recorded: the way decides)
        let mut on_a = claim(1, 0, 8.5, 0.0);
        on_a.chain = vec![(0, 0.0), (2, 10.0)];
        let mut on_b = claim(2, 1, 1.0, 0.0);
        on_b.chain = vec![(1, 0.0), (2, 10.0)];
        let claims = [on_a, on_b];
        let g = resolve(&js, &claims);
        assert!(grant_of(&g, 1).go, "{:?}", g);
        // (at a merge that crosses nothing: capped by A, the zipper, not refused)
        assert!(gives_way(&g, 2, 1), "B still comes in behind A");
        // the one on B ahead along the common way: it is a foe
        let mut ahead = claims.clone();
        ahead[0].s = 4.0;
        ahead[1].s = 4.0;
        let g = resolve(&js, &ahead);
        assert!(gives_way(&g, 1, 2), "{g:?}");
    }

    #[test]
    fn competitor_filter_a_standing_refused_car_60_m_away_does_not_hold_a_car_at_the_junction() {
        let (_, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let at_line = claim(1, side, -1.0, 0.0);
        // standing in a queue 60 m out on the main road, refused itself (no room past)
        let mut far = claim(2, main, -60.0, 0.0);
        far.exit_room = false;
        far.held = true;
        let claims = [at_line, far];
        let g = resolve(&js, &claims);
        assert!(grant_of(&g, 1).go, "{:?}", g);
        assert!(!grant_of(&g, 2).go);
        // the same car rolling at town speed (it is coming: in the meeting place before I am out
        // with `JM_TIMEGAP_MINOR`) still counts (at 3 m/s it comes 3 s after I am out)
        let mut rolling = claims.clone();
        rolling[1].speed = 13.0;
        rolling[1].held = false;
        let g = resolve(&js, &rolling);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2));
    }

    #[test]
    fn competitor_filter_a_committed_foe_standing_outside_this_meeting_place_is_not_on_its_way_into_it() {
        let (_, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        // the side road car stands 15 m short of the crossing, "committed" (its first
        // meeting place, with its own stub, lies behind its bumper), held by its lead
        let mut far = claim(2, side, -15.0, 0.0);
        far.committed = true;
        far.held = true;
        let claims = [claim(1, main, -1.0, 0.0), far];
        let g = resolve(&js, &claims);
        assert!(grant_of(&g, 1).go, "{g:?}");
        // in the meeting place it is committed to it
        let mut inside = claims.clone();
        inside[1].s = 7.0;
        let g = resolve(&js, &inside);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2));
    }

    #[test]
    fn competitor_filter_a_standing_car_further_back_on_a_shared_piece_is_behind_before_my_front_reaches_it() {
        // A (0) and B (1) merge where C (2) begins; A gives way to B. Both stand short of the
        // object, the one for B further back along the common way.
        let (_, js) = merge_a_gives_way();
        let mut on_a = claim(1, 0, -3.0, 0.0);
        on_a.chain = vec![(0, 0.0), (2, 10.0)];
        let mut on_b = claim(2, 1, -9.0, 0.0);
        on_b.chain = vec![(1, 0.0), (2, 10.0)];
        let claims = [on_a, on_b];
        let g = resolve(&js, &claims);
        assert!(grant_of(&g, 1).go, "{g:?}");
        // rolling, it may be first at the merge: it counts (one of the two gives way to the
        // other at the zipper)
        let mut rolling = claims.clone();
        rolling[1].speed = 6.0;
        let g = resolve(&js, &rolling);
        assert!(gives_way(&g, 1, 2) != gives_way(&g, 2, 1), "{g:?}");
    }

    #[test]
    fn competitor_filter_a_foe_whose_lead_waits_for_me_is_no_foe() {
        // the side road car (1) stands at its line with its nose over the main road; the
        // main road car right at it (7) follows its body, the one behind that (2) follows 7
        let (_, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let mut first = claim(7, main, -3.0, 0.0);
        first.lead = Some(1);
        let mut second = claim(2, main, -10.0, 0.0);
        second.lead = Some(7);
        let claims = [claim(1, side, -1.0, 0.0), first, second];
        let g = resolve(&js, &claims);
        assert!(grant_of(&g, 1).go, "{g:?}");
        // the second car's lead is somebody else: it may come, it counts
        let mut other = claims.clone();
        other[2].lead = Some(99);
        let g = resolve(&js, &other);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2));
    }

    /// Three ways round a triangle, each crossing the other two, equal priorities: right
    /// before left gives way all round (0 to 2, 2 to 1, 1 to 0).
    fn triangle() -> (Network, Junctions) {
        let way = |h: f64| -> Vec<(f64, f64)> {
            let (dx, dy) = (h.to_radians().sin(), h.to_radians().cos());
            // 5 m off the middle, to the left of the way
            let (cx, cy) = (-5.0 * dy, 5.0 * dx);
            vec![(cx - 25.0 * dx, cy - 25.0 * dy), (cx + 25.0 * dx, cy + 25.0 * dy)]
        };
        let (a, b, c) = (way(90.0), way(210.0), way(330.0));
        object(&[(&a, 0), (&b, 0), (&c, 0)])
    }

    #[test]
    fn competitor_filter_a_ring_of_three_standing_claimants_lets_exactly_one_go_first() {
        let (_, js) = triangle();
        for m in 0..3 {
            let f = &js.nodes[0].movements[m].foes;
            assert_eq!(f.len(), 2, "movement {m}: {f:?}");
            assert_eq!(f.iter().filter(|f| f.yields).count(), 1, "movement {m} gives way to exactly one: {f:?}");
        }
        let claims: Vec<Claim> = (0..3)
            .map(|m| {
                let c = claim(10 + m, m, -1.0, 0.0);
                c
            })
            .collect();
        let g = resolve(&js, &claims);
        let going: Vec<usize> = g.iter().filter(|g| g.go).map(|g| g.vehicle).collect();
        assert_eq!(going, vec![10], "exactly one goes, the lowest index of the ring: {:?}", g);
        // (the others wait within the ring: for it, or - the one it gives way to having
        // been decided before it - for that one, which waits for it)
        assert!(g.iter().filter(|g| !g.go).all(|g| g.wait_for.is_some_and(|w| (10..13).contains(&w))), "the others wait in the ring: {:?}", g);
    }

    #[test]
    fn competitor_filter_of_two_standing_refused_claimants_the_one_with_the_right_of_way_goes() {
        // both stand refused at their lines (for whatever reason) and would drop each other
        let (_, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let refused = |v| Grant::refused(v, None, Hold::Foe);
        // the main road car has the higher index: the table decides, not the index
        let claims = [claim(5, main, -1.0, 0.0), claim(3, side, -1.0, 0.0)];
        let plain = [refused(5), refused(3)];
        let st = standing_order(&js, &claims, &plain);
        assert!(matches!((st[0], st[1]), (Standing::Rank(a), Standing::Rank(b)) if a < b), "{st:?}");
        let g = decide(&js, &claims, Some(&plain), Some(&st));
        assert!(g[0].go, "{g:?}");
        assert_eq!(g[1].wait_for, Some(5));
        // the one before in the order stays refused (held by something else): the other
        // does not wait for it
        let mut exit_full = claims.clone();
        exit_full[0].exit_room = false;
        let plain = [Grant::refused(5, None, Hold::ExitFull), refused(3)];
        let st = standing_order(&js, &exit_full, &plain);
        assert_eq!(st[0], Standing::Hard);
        let g = decide(&js, &exit_full, Some(&plain), Some(&st));
        assert!(!g[0].go);
        assert!(g[1].go, "{g:?}");
    }

    #[test]
    fn the_side_road_waits_for_the_main_road_and_goes_when_it_is_clear() {
        let (_, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        // both at their lines, the main road car still 20 m out
        let g = resolve(&js, &[claim(1, main, -20.0, 10.0), claim(2, side, -1.0, 0.0)]);
        assert!(g[0].go);
        assert!(!g[1].go);
        assert_eq!(g[1].wait_for, Some(1));
        // the main road car is 120 m out: the gap (its own, plus the time to clear) is enough to cross
        let g = resolve(&js, &[claim(1, main, -120.0, 10.0), claim(2, side, -1.0, 0.0)]);
        assert!(g[1].go);
        // no main road car at all
        let g = resolve(&js, &[claim(2, side, -1.0, 0.0)]);
        assert!(g[0].go);
    }

    #[test]
    fn right_before_left_between_equals_and_first_come_inside() {
        let (_, js) = four_way(128.0, 128.0);
        let (east, north) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let g = resolve(&js, &[claim(1, east, -1.0, 0.0), claim(2, north, -1.0, 0.0)]);
        // exactly one of the two goes
        assert_ne!(g[0].go, g[1].go);
        // the one already in the junction goes, whoever has the rules on their side
        let yields_first = !g[0].go;
        // (standing in the box, so it will not be gone before the other arrives)
        let inside = if yields_first { claim(1, east, 8.0, 0.0) } else { claim(2, north, 8.0, 0.0) };
        let other = if yields_first { claim(2, north, -1.0, 0.0) } else { claim(1, east, -1.0, 0.0) };
        let mut inside = inside;
        inside.committed = true;
        let g = resolve(&js, &[inside, other]);
        assert!(g[0].go);
        assert!(!g[1].go);
    }

    #[test]
    fn a_full_exit_holds_the_car_until_it_is_fed_up() {
        let (_, js) = four_way(192.0, 64.0);
        let main = js.nodes[0].movement_of(0).unwrap();
        let mut c = claim(1, main, -1.0, 0.0);
        c.exit_room = false;
        let g = resolve(&js, &[c.clone()]);
        assert_eq!(g[0].hold, Some(Hold::ExitFull));
        let p = DriverParams::preset(VehicleClass::Car);
        for _ in 0..1000 {
            c.impatience.update(&p, 0.1, true);
        }
        let g = resolve(&js, &[c]);
        assert!(g[0].go, "an impatient driver squeezes in");
    }

    #[test]
    fn a_train_is_never_held() {
        let (_, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let mut train = claim(9, side, -30.0, 15.0);
        train.absolute = true;
        let g = resolve(&js, &[claim(1, main, -1.0, 0.0), train]);
        assert!(!g[0].go);
        assert!(g[1].go);
    }

    // ---- counter-examples to the competitor filter and signals ----

    /// Pairs of claimants that both go, are foes at some meeting place, and neither of
    /// which is inside it: released into each other unless something else (car-following)
    /// keeps them apart.
    fn review_both_released(js: &Junctions, claims: &[Claim], g: &[Grant]) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for a in 0..claims.len() {
            for b in a + 1..claims.len() {
                if !(g[a].go && g[b].go) || claims[a].committed || claims[b].committed {
                    continue;
                }
                let foes = js.foes_along(claims[a].junction, &claims[a].chain);
                // and their windows there overlap (each from its own state; a car standing at
                // its red counts from its green, or not at all)
                let win = |c: &Claim, from: f32, to: f32, shift: f32| -> Option<(f32, f32)> {
                    let mut x = c.clone();
                    x.s -= shift;
                    let (i, o) = window(&x, from, to);
                    if c.light_red && c.speed < 0.5 {
                        return c.until_go.map(|t| (i + t, o + t));
                    }
                    Some((i, o))
                };
                let meet = foes.iter().any(|f| {
                    claims[b].chain.iter().any(|&(m, off)| {
                        m == f.movement
                            && claims[b].s - off + claims[b].front < f.theirs.0
                            && claims[a].s + claims[a].front < f.mine.0
                            && match (win(&claims[a], f.mine.0, f.mine.1, 0.0), win(&claims[b], f.theirs.0, f.theirs.1, off)) {
                                (Some((ai, ao)), Some((bi, bo))) => ai < bo + JM_TIMEGAP_MINOR && bi < ao + JM_TIMEGAP_MINOR,
                                _ => false,
                            }
                    })
                });
                if meet {
                    out.push((claims[a].vehicle, claims[b].vehicle));
                }
            }
        }
        out
    }

    /// Four straight ways of an unmarked cross, right-hand traffic, equal priorities.
    fn review_cross() -> (Network, Junctions) {
        object(&[
            (&[(-20.0, -2.0), (20.0, -2.0)], 0), // eastbound
            (&[(20.0, 2.0), (-20.0, 2.0)], 0),   // westbound
            (&[(2.0, -20.0), (2.0, 20.0)], 0),   // northbound
            (&[(-2.0, 20.0), (-2.0, -20.0)], 0), // southbound
        ])
    }

    #[test]
    fn review_two_crossing_standing_both_refused_exactly_one_goes_either_index_order() {
        let (_, js) = four_way(128.0, 128.0);
        let (east, north) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        for (ve, vn) in [(1usize, 2usize), (2, 1)] {
            let claims = [claim(ve, east, -1.0, 0.0), claim(vn, north, -1.0, 0.0)];
            let refused = |v| Grant::refused(v, None, Hold::Foe);
            let plain = [refused(ve), refused(vn)];
            let st = standing_order(&js, &claims, &plain);
            let g = decide(&js, &claims, Some(&plain), Some(&st));
            assert_eq!(g.iter().filter(|g| g.go).count(), 1, "{ve}/{vn}: {g:?}");
            let goer = g.iter().find(|g| g.go).unwrap().vehicle;
            assert!(g.iter().filter(|g| !g.go).all(|g| g.wait_for == Some(goer)), "{g:?}");
        }
    }

    #[test]
    fn review_three_ring_permuted_indices_and_one_member_far_back() {
        let (_, js) = triangle();
        for ids in [[30usize, 10, 20], [20, 30, 10], [10, 20, 30]] {
            for far in [None, Some(0usize), Some(1), Some(2)] {
                let claims: Vec<Claim> = (0..3)
                    .map(|m| {
                        let s = if far == Some(m) { -12.0 } else { -1.0 };
                        let c = claim(ids[m], m, s, 0.0);
                        c
                    })
                    .collect();
                let g = resolve(&js, &claims);
                assert!(g.iter().any(|g| g.go), "{ids:?} far {far:?}: nobody goes {g:?}");
                assert!(review_both_released(&js, &claims, &g).is_empty(), "{ids:?} far {far:?}: {g:?}");
            }
        }
    }

    #[test]
    fn review_four_ring_equal_priority_no_crossing_pair_released() {
        let (_, js) = review_cross();
        let yields: usize = (0..4).map(|m| js.nodes[0].movements[m].foes.iter().filter(|f| f.yields).count()).sum();
        let claims: Vec<Claim> = (0..4).map(|m| claim(10 + m, m, -1.0, 0.0)).collect();
        let g = resolve(&js, &claims);
        eprintln!("4-ring: yields in table {yields}; {g:?}");
        assert!(g.iter().any(|g| g.go), "{g:?}");
        assert!(review_both_released(&js, &claims, &g).is_empty(), "{g:?}");
        for p in [[3usize, 1, 0, 2], [1, 0, 3, 2], [2, 3, 1, 0]] {
            let claims: Vec<Claim> = (0..4).map(|m| claim(10 + p[m], m, -1.0, 0.0)).collect();
            let g = resolve(&js, &claims);
            assert!(g.iter().any(|g| g.go), "{p:?} {g:?}");
            assert!(review_both_released(&js, &claims, &g).is_empty(), "{p:?} {g:?}");
        }
    }

    #[test]
    fn review_standing_refused_foe_5m_out_is_still_first_once_it_moves() {
        // triangle: me gives way to the foe, the foe gives way to Z (rolling in fast), Z to me
        let (_, js) = triangle();
        let yields = |a: usize, b: usize| js.nodes[0].movements[a].foes.iter().any(|f| f.movement == b && f.yields);
        let (mut me_m, mut foe_m, mut z_m) = (0, 0, 0);
        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    if a != b && b != c && a != c && yields(a, b) && yields(b, c) {
                        (me_m, foe_m, z_m) = (a, b, c);
                    }
                }
            }
        }
        let mp = |m: usize, other: usize| js.nodes[0].movements[m].foes.iter().find(|f| f.movement == other).unwrap().mine.0;
        // my front 40 m from my meeting place with the foe; the foe's front 5 m from its first
        let me = claim(1, me_m, mp(me_m, foe_m) - 40.0 - 2.0, 0.0);
        let foe = claim(2, foe_m, mp(foe_m, me_m).min(mp(foe_m, z_m)) - 5.0 - 2.0, 0.0);
        let z = claim(3, z_m, mp(z_m, foe_m) - 2.0 - 15.0, 10.0);
        let claims = [me.clone(), foe.clone(), z.clone()];
        let g = resolve(&js, &claims);
        eprintln!("frame 1 {g:?}");
        // (held by Z, or behind it as SUMO's crossing follower)
        assert!(not_ahead_of(&g, 2), "the foe is held by Z: {g:?}");
        // frame 2: Z is gone; I have rolled off at 0.5 m/s. The foe must be first.
        let mut me2 = me.clone();
        me2.s += 0.3;
        me2.speed = 0.5;
        let g = resolve(&js, &[me2.clone(), foe.clone()]);
        assert!(grant_of(&g, 2).go, "{g:?}");
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "frame 2: I wait for the foe once it goes: {g:?}");
        // ... and standing still at 40 m myself: same
        let g = resolve(&js, &[me.clone(), foe.clone()]);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
    }

    #[test]
    fn review_foe_at_red_with_1s_to_go_counts_as_coming_in_1s() {
        let (_, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let me = claim(1, side, 4.0, 0.0);
        let mut red = claim(2, main, 1.0, 0.0);
        red.light_red = true;
        red.until_go = Some(1.0);
        let g = resolve(&js, &[me.clone(), red.clone()]);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
        // the same foe refused for a full exit (Hard): the second pass skips it outright,
        // whatever its light, though its exit may clear within the second
        let mut hard = red.clone();
        hard.exit_room = false;
        let g = resolve(&js, &[me.clone(), hard.clone()]);
        eprintln!("exit-full foe at red, 1 s to go: {g:?}");
    }

    #[test]
    fn review_moving_foe_10ms_40m_out_counts() {
        let (_, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let mine0 = js.nodes[0].movements[main].foes.iter().find(|f| f.movement == side).unwrap().mine.0;
        let me = claim(1, side, 4.0, 0.0);
        let mover = claim(2, main, mine0 - 40.0 - 2.0, 10.0);
        let g = resolve(&js, &[me.clone(), mover.clone()]);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
    }

    #[test]
    fn review_lead_is_me_drops_a_rolling_foe_2m_from_the_meeting_place() {
        // the main road car (right of way) rolls at 3 m/s, its front 2 m from the meeting
        // place, and names me (the side road car) as its lead
        let (_, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let mine0 = js.nodes[0].movements[main].foes.iter().find(|f| f.movement == side).unwrap().mine.0;
        let me = claim(1, side, 4.0, 0.0);
        let mut foe = claim(2, main, mine0 - 2.0 - 2.0, 3.0);
        foe.lead = Some(1);
        let g = resolve(&js, &[me.clone(), foe.clone()]);
        // (was: dropped on the strength of `lead` alone, both went) a rolling follower is
        // a foe: exactly one goes
        assert_eq!(g.iter().filter(|g| g.go).count(), 1, "{g:?}");
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
        assert!(review_both_released(&js, &[me, foe], &g).is_empty());
    }

    #[test]
    fn review_merge_both_standing_6cm_apart_both_released() {
        // A (0) and B (1) merge where C (2) begins; A gives way to B. Both stand short of
        // the merge, B 6 cm further back along the common way
        let (_, js) = object(&[(&[(-10.0, 0.0), (0.0, 0.0)], 0), (&[(-8.0, -6.0), (0.0, 0.0)], 0), (&[(0.0, 0.0), (10.0, 0.0)], 0)]);
        let mut on_a = claim(1, 0, -3.0, 0.0);
        on_a.chain = vec![(0, 0.0), (2, 10.0)];
        let mut on_b = claim(2, 1, -3.06, 0.0);
        on_b.chain = vec![(1, 0.0), (2, 10.0)];
        let claims = [on_a, on_b];
        let g = resolve(&js, &claims);
        eprintln!("merge 6 cm: {g:?}");
        // level cars are not one behind the other
        // (at a zipper: exactly one gives way to the other, capped)
        assert!(gives_way(&g, 1, 2) != gives_way(&g, 2, 1), "exactly one gives way: {g:?}");
    }

    // ---- the competitor filter and signals ----

    /// The main road car (right of way) stands `dist` m out, held by `blocked` (its lead,
    /// people, a light: not by the junction); the side road car rolls up at `speed`.
    fn held_main_road_foe(dist: f32, speed: f32, blocked: bool) -> (Network, Junctions, Vec<Claim>) {
        let (net, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let mine0 = js.nodes[0].movements[main].foes.iter().find(|f| f.movement == side).unwrap().mine.0;
        let me = claim(1, side, 2.0, speed);
        let mut foe = claim(2, main, mine0 - dist - 2.0, 0.0);
        foe.held = true;
        foe.blocked = blocked;
        (net, js, vec![me, foe])
    }

    #[test]
    fn competitor_filter_a_foe_standing_behind_its_lead_far_off_does_not_hold_a_moving_claimant() {
        let (_, js, claims) = held_main_road_foe(30.0, 3.0, true);
        let g = resolve(&js, &claims);
        assert!(grant_of(&g, 1).go, "{g:?}");
        // ... and standing at the line myself, not refused
        let mut standing = claims.clone();
        standing[0].speed = 0.0;
        assert!(grant_of(&resolve(&js, &standing), 1).go);
        // held by my own lead as well: standing behind it short of the stretch I do not
        // come (`clear_ahead`): no foe of mine, no refusal - SUMO registers no approach
        let mut both = claims.clone();
        both[0].speed = 0.0;
        both[0].held = true;
        both[0].blocked = true;
        both[0].clear_ahead = 1.0;
        let g = resolve(&js, &both);
        assert!(grant_of(&g, 1).go && grant_of(&g, 1).leader_go == 0, "{g:?}");
    }

    #[test]
    fn will_pass_a_foe_standing_behind_its_standing_lead_short_of_the_meeting_place_is_not_coming() {
        // 10 m out, standing, from rest it would be there before I am out (the test below) - but
        // its standing lead 0.5 m ahead of its front (`lead_rest`) stands at a red with 48 s
        // left: it starts no sooner than that green (`lead_release`), whatever its why names -
        // neither held, blocked nor stuck (a car at its zipper waiting place, labelled Junction).
        // SUMO's `setRequest`: no request while its planned speed is 0, re-made every step
        let (_, js, mut claims) = held_main_road_foe(10.0, 3.0, false);
        claims[1].held = false;
        claims[1].lead_gap = Some((0.5, 0.0));
        claims[1].lead = Some(3);
        let mut lead = claims[1].clone();
        lead.vehicle = 3;
        lead.id = 3;
        lead.s = claims[1].s + 6.5;
        lead.lead = None;
        lead.lead_gap = None;
        lead.light_red = true;
        lead.until_go = Some(48.0);
        lead.red_at = lead.s + lead.front + 0.5;
        claims.push(lead);
        let g = resolve(&js, &claims);
        assert!(grant_of(&g, 1).go, "{g:?}");
        // its lead's green half a second off: the queue comes from rest after it, before I am
        // out (I wait for its head)
        claims[2].until_go = Some(0.5);
        let g = resolve(&js, &claims);
        assert!(!grant_of(&g, 1).go, "{g:?}");
        // a lead standing for people (stuck): not while that holds (the 08:19 class: a foe
        // labelled Junction behind a head held by walkers)
        claims[2].until_go = None;
        claims[2].light_red = false;
        claims[2].stuck = true;
        claims[2].blocked = true;
        let g = resolve(&js, &claims);
        assert!(grant_of(&g, 1).go, "{g:?}");
        // a lead held by something not known (no claim): it counts from rest now
        claims.pop();
        let g = resolve(&js, &claims);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
        claims[1].lead = None;
        // its lead 3 m on, past the meeting place's start: it can come, and it counts
        claims[1].lead_gap = Some((3.0, 0.0));
        let g = resolve(&js, &claims);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
        // its lead 0.5 m on but moving off: it counts, its window from rest
        claims[1].lead_gap = Some((0.5, 1.0));
        let g = resolve(&js, &claims);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
        // people stopping it there: not this rule (they clear on their own schedule; `held_off`
        // drops a foe stuck for people)
        claims[1].lead_gap = None;
        claims[1].clear_ahead = claims[1].front + 0.5;
        let g = resolve(&js, &claims);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
        // rolling up to a standing lead it registers its request (SUMO: v > eps): the time test decides
        claims[1].clear_ahead = f32::INFINITY;
        claims[1].lead_gap = Some((0.5, 0.0));
        claims[1].speed = 6.0;
        let g = resolve(&js, &claims);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
    }

    /// J11-9: a bus standing at a stop it serves, 1 m out (from rest it would be there before
    /// I am out), is not approaching the link (SUMO: no request until its stop ends); once it
    /// moves, it counts again.
    #[test]
    fn will_pass_a_bus_serving_its_stop_short_of_the_meeting_place_is_not_coming() {
        let (_, js, mut claims) = held_main_road_foe(1.0, 3.0, true);
        claims[1].serving = true;
        assert!(grant_of(&resolve(&js, &claims), 1).go);
        claims[1].serving = false;
        assert_eq!(grant_of(&resolve(&js, &claims), 1).wait_for, Some(2));
        claims[1].serving = true;
        claims[1].speed = 2.0;
        claims[1].held = false;
        claims[1].blocked = false;
        assert_eq!(grant_of(&resolve(&js, &claims), 1).wait_for, Some(2));
    }

    #[test]
    fn competitor_filter_a_held_foe_that_could_be_there_before_i_am_out_still_counts() {
        // 1 m out: from rest after its start it is in the meeting place before I am out
        let (_, js, claims) = held_main_road_foe(1.0, 3.0, true);
        let g = resolve(&js, &claims);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
        // refused by the junction rather than held (it moves off as soon as it is let go):
        // the order of the second pass decides, not the timing (10 m out, from rest in
        // the meeting place 3.3 s on, before I am out with `JM_TIMEGAP_MINOR`: 30 m out, 5.8 s,
        // it comes after me by SUMO's time test)
        let (_, js, claims) = held_main_road_foe(10.0, 3.0, false);
        let g = resolve(&js, &claims);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
        let (_, js, claims) = held_main_road_foe(30.0, 3.0, false);
        assert!(grant_of(&resolve(&js, &claims), 1).go);
        // rolling (not standing): it is coming
        let (_, js, mut claims) = held_main_road_foe(10.0, 3.0, true);
        claims[1].speed = 6.0;
        claims[1].held = false;
        claims[1].blocked = false;
        let g = resolve(&js, &claims);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
    }

    #[test]
    fn competitor_filter_a_car_behind_me_in_my_own_lane_is_no_foe_at_any_speed() {
        // one lane forks inside the object: the straight way (0) and the left turn (1),
        // which gives way to it at the start; both cars on the entry lane 10 before it
        let (net, js) = object(&[(&[(-10.0, 0.0), (10.0, 0.0)], 0), (&[(-10.0, 0.0), (-4.0, 0.0), (0.0, 4.0), (0.0, 10.0)], 1)]);
        let mut turner = claim(1, 1, -1.0, 0.0);
        turner.lanes = vec![(10, -30.0), (net.lanes.len(), 1.0)];
        // the straight car rolls up behind it at 6 m/s, committed (it cannot stop before
        // the merge of the lead-in with its stub, it cannot stop before the merge), its front 8 m behind my rear
        let mut behind = claim(2, 0, -11.0, 6.0);
        behind.committed = true;
        behind.lanes = vec![(10, -20.0), (99, 10.0)];
        let claims = [turner, behind];
        let g = resolve(&js, &claims);
        assert!(grant_of(&g, 1).go, "{g:?}");
        // on a lane before mine that leads into it: behind me too
        let mut back = claims.clone();
        back[1].lanes = vec![(11, -20.0), (10, 2.0)];
        back[1].s = -33.0;
        assert!(grant_of(&resolve(&js, &back), 1).go);
        // ahead of me in my lane (it is the one in front): it counts
        let mut ahead = claims.clone();
        ahead[1].lanes = vec![(10, -34.0)];
        ahead[1].committed = false;
        ahead[1].s = 3.0;
        ahead[1].speed = 0.0;
        let g = resolve(&js, &ahead);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
    }

    #[test]
    fn competitor_filter_a_car_merging_in_behind_me_is_no_foe_before_my_front_reaches_the_shared_piece() {
        // A (0) and B (1) merge into the exit; B (from the left) gives way to A. No piece of
        // the object is on both ways (the exit lies outside it): the merge says who is behind.
        let (_, js) = object(&[(&[(-10.0, 0.0), (0.0, 0.0)], 0), (&[(-8.0, 6.0), (0.0, 0.0)], 0)]);
        let f = js.nodes[0].movements[0].foes[0];
        assert!(!f.yields && f.kind == FoeKind::Merge, "{f:?}");
        let on_a = claim(1, 0, 7.0, 3.0);
        // B rolls up behind at 5.8 m/s, its front 18 m from the merge's end, too close to
        // stop before its meeting place (committed) (14 m
        // off it was there 0.9 s after my rear, inside SUMO's `JM_TIMEGAP_MINOR`)
        let mut on_b = claim(2, 1, -10.0, 5.8);
        on_b.committed = true;
        let claims = [on_a, on_b];
        let g = resolve(&js, &claims);
        assert!(grant_of(&g, 1).go, "{g:?}");
        // fast, it would be at the merge's end before my rear is past: it counts (I am in
        // the stretch already, first at the zipper: it is capped behind me)
        let mut fast = claims.clone();
        fast[1].speed = 13.0;
        let g = resolve(&js, &fast);
        assert!(gives_way(&g, 2, 1), "{g:?}");
        // level with me: not behind (one of the two gives way to the other)
        let mut level = claims.clone();
        level[1].s = 5.0;
        level[1].speed = 3.0;
        let g = resolve(&js, &level);
        assert!(gives_way(&g, 1, 2) != gives_way(&g, 2, 1), "{g:?}");
        // giving way: B with the right of way 14 m from the merge's end at 5.8 m/s,
        // which could not follow me, goes first (the zipper; in the stretch
        // already, I am ahead of B only when B can brake for me - it cannot here, so I am capped)
        let (_, js) = object(&[(&[(-10.0, 0.0), (0.0, 0.0)], 0), (&[(-8.0, -6.0), (0.0, 0.0)], 0)]);
        let f = js.nodes[0].movements[0].foes[0];
        assert!(f.yields);
        let claims = [claim(1, 0, f.mine.0 - 1.0 - 2.0, 3.0), claim(2, 1, -6.0, 5.8)];
        let g = resolve(&js, &claims);
        assert!(gives_way(&g, 1, 2) && !gives_way(&g, 2, 1), "{g:?}");
        let claims = [claim(1, 0, 7.0, 3.0), claim(2, 1, -6.0, 5.8)];
        let g = resolve(&js, &claims);
        assert!(gives_way(&g, 1, 2) != gives_way(&g, 2, 1), "{g:?}");
    }

    #[test]
    fn signal_foes_a_foe_creeping_to_its_red_after_a_stand_is_not_coming() {
        let (_, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let mine0 = js.nodes[0].movements[main].foes.iter().find(|f| f.movement == side).unwrap().mine.0;
        let me = claim(1, side, 2.0, 0.0);
        // the main road car creeps up to its red at 0.8 m/s, 6 m out, 30 s to go
        let mut red = claim(2, main, mine0 - 6.0 - 2.0, 0.8);
        red.light_red = true;
        red.until_go = Some(30.0);
        red.was_standing = true;
        let g = resolve(&js, &[me.clone(), red.clone()]);
        assert!(grant_of(&g, 1).go, "{g:?}");
        // rolling at 0.8 m/s without having stood: held at its red short of the meeting
        // place, it does not come before its green either
        let mut rolling = red.clone();
        rolling.was_standing = false;
        assert!(grant_of(&resolve(&js, &[me, rolling.clone()]), 1).go);
        // its red past the meeting place: it comes through it first
        rolling.red_at = mine0 + 10.0;
        assert_eq!(grant_of(&resolve(&js, &[claim(1, side, 2.0, 0.0), rolling]), 1).wait_for, Some(2));
    }

    /// Pairs both let go into one meeting place whose windows overlap, a claimant held by
    /// its lead (`blocked`) counted from rest after its start (`TEST_START`): it does not
    /// move before that, whatever its grant says.
    /// The reaction (s) a claimant held by its lead takes to move off once let go, as the safety
    /// checks below model it (a test figure, no rule of the resolver).
    const TEST_START: f32 = 1.5;

    fn round3_both_released(js: &Junctions, claims: &[Claim], g: &[Grant]) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let win = |c: &Claim, from: f32, to: f32, shift: f32| -> Option<(f32, f32)> {
            let mut x = c.clone();
            x.s -= shift;
            let (mut i, mut o) = window(&x, from, to);
            if c.blocked {
                i += TEST_START;
                o += TEST_START;
            }
            // (a claimant held by its red is let go by the junction, its light
            // holds it before any meeting place: it comes at its green, whatever its speed)
            if c.light_red {
                return c.until_go.map(|t| (i + t, o + t));
            }
            Some((i, o))
        };
        for a in 0..claims.len() {
            for b in a + 1..claims.len() {
                if !(g[a].go && g[b].go) || claims[a].committed || claims[b].committed {
                    continue;
                }
                let foes = js.foes_along(claims[a].junction, &claims[a].chain);
                let meet = foes.iter().any(|f| {
                    claims[b].chain.iter().any(|&(m, off)| {
                        m == f.movement
                            && claims[b].s - off + claims[b].front < f.theirs.0
                            && claims[a].s + claims[a].front < f.mine.0
                            && match (win(&claims[a], f.mine.0, f.mine.1, 0.0), win(&claims[b], f.theirs.0, f.theirs.1, off)) {
                                (Some((ai, ao)), Some((bi, bo))) => ai < bo + JM_TIMEGAP_MINOR && bi < ao + JM_TIMEGAP_MINOR,
                                _ => false,
                            }
                    })
                });
                if meet {
                    out.push((claims[a].vehicle, claims[b].vehicle));
                }
            }
        }
        out
    }

    #[test]
    fn competitor_filter_fuzz_held_foes_never_release_two_claimants_into_each_other() {
        // random standing/rolling claimants, some held by their lead (`blocked`), some
        // creeping to their red after a stand, on crossing-only objects: no pair let go into
        // one meeting place that the competitor filter without the `blocked` and signal reads did not let go
        let objects: Vec<(Network, Junctions)> = vec![review_cross(), triangle(), four_way(128.0, 128.0), four_way(192.0, 64.0)];
        let mut seed: u64 = 0x9E3779B97F4A7C15;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let (mut cases, mut dropped, mut new_pairs, mut held_pairs) = (0, 0, 0, 0);
        for _ in 0..100_000 {
            let (_, js) = &objects[(rnd() * objects.len() as f64) as usize % objects.len()];
            let nm = js.nodes[0].movements.len();
            let n = 2 + (rnd() * 4.0) as usize;
            let mut claims: Vec<Claim> = Vec::new();
            for v in 0..n {
                let m = (rnd() * nm as f64) as usize % nm;
                let first = js.nodes[0].movements[m].foes.iter().map(|f| f.mine.0).fold(f32::INFINITY, f32::min);
                if !first.is_finite() {
                    continue;
                }
                let dist = [0.2f32, 1.0, 3.0, 6.0, 15.0, 40.0, 70.0][(rnd() * 7.0) as usize % 7];
                let speed = [0.0f32, 0.0, 0.0, 0.2, 0.8, 2.0, 6.0, 10.0][(rnd() * 8.0) as usize % 8];
                let mut c = claim(10 + v, m, first - dist - 2.0, speed);
                let room = dist - 0.6;
                c.committed = dist <= 0.5 || (speed > 1.0 && room < speed * speed / (2.0 * 8.0 * 0.7));
                c.exit_room = rnd() > 0.15;
                c.was_standing = speed <= 1.0 && rnd() < 0.5;
                c.held = speed < 0.3 && rnd() < 0.6;
                c.blocked = c.held && rnd() < 0.7;
                if rnd() < 0.15 {
                    c.light_red = true;
                    c.until_go = if rnd() < 0.5 { Some((rnd() * 5.0) as f32) } else { None };
                }
                if claims.iter().any(|o| o.movement == m && (o.s - c.s).abs() < 6.0) {
                    continue;
                }
                claims.push(c);
            }
            if claims.len() < 2 {
                continue;
            }
            cases += 1;
            // (the rules without what the claims read: the same claims with `blocked` cleared)
            let before: Vec<Claim> = claims.iter().map(|c| Claim { blocked: false, ..c.clone() }).collect();
            let old = resolve(js, &before);
            let g = resolve(js, &claims);
            dropped += old.iter().zip(&g).filter(|(o, n)| !o.go && n.go).count();
            let po = round3_both_released(js, &claims, &old);
            // (a claimant let go while held by its lead or standing at its red does not move
            // on its grant: it claims again, moving, the frame it can; such a pair is counted
            // apart)
            let idx = |v: usize| claims.iter().position(|c| c.vehicle == v).unwrap();
            let can_move = |k: usize| {
                let c = &claims[k];
                !(c.blocked || c.light_red)
            };
            for p in round3_both_released(js, &claims, &g) {
                if !po.contains(&p) {
                    let (a, b) = (idx(p.0), idx(p.1));
                    // (one of the pair held by its red is let go by the junction and
                    // claims again at its green, when the table decides between the two)
                    let at_red = [a, b].iter().any(|&k| claims[k].light_red);
                    let moved = !at_red && [a, b].iter().any(|&k| !old[k].go && g[k].go && can_move(k));
                    if !moved {
                        held_pairs += 1;
                        continue;
                    }
                    new_pairs += 1;
                    if new_pairs <= 4 {
                        eprintln!("NEW PAIR {p:?}\n  claims {:?}\n  old {old:?}\n  new {g:?}", claims.iter().map(|c| (c.vehicle, c.movement, c.s, c.speed, c.committed, c.held, c.blocked, c.light_red, c.until_go)).collect::<Vec<_>>());
                    }
                }
            }
        }
        eprintln!("round3 fuzz: {cases} cases, claimants let go that the rules before held {dropped}, new pairs released into each other {new_pairs} (and {held_pairs} let go while still held by their lead or light)");
        assert!(dropped > 0);
        assert_eq!(new_pairs, 0);
    }

    // ---- counter-examples to the merging-behind and following rules ----

    /// `merging_behind` takes the leader case's `JM_TIMEGAP_MINOR`: a side car standing at
    /// the merge goes in front of a main road car at 14 m/s whose front is 40.5 m from the
    /// merge end. At the moment the side car's rear clears the merge end the main car is
    /// 8.4 m behind it, closing at ~10 m/s: it must brake ~4-6 m/s² (with/without a 2 m standstill gap) for a car that
    /// had to give way to it.
    #[test]
    fn review4_merging_behind_bypasses_the_accepted_gap_at_14_ms() {
        let (_, js) = object(&[(&[(-10.0, 0.0), (0.0, 0.0)], 0), (&[(-8.0, -6.0), (0.0, 0.0)], 0)]);
        let f = js.nodes[0].movements[0].foes[0];
        assert!(f.yields && f.kind == FoeKind::Merge, "A gives way to B at the merge");
        // (its front 0.5 m short of the stretch where the two converge: in it, it is in the
        // main road car's way whatever it does; before, 0.5 m short of the merge's end)
        let side = claim(1, 0, f.mine.0 - 0.5 - 2.0, 0.0);
        let mut main = claim(2, 1, f.theirs.1 - 40.5 - 2.0, 14.0);
        main.v_max = 14.0;
        let my_out = arrival_time(4.5, 0.0, 1.8, 13.0);
        let their_at = arrival_time(40.5, 14.0, 1.8, 14.0);
        let claims = [side, main];
        let g = resolve(&js, &claims);
        let gap_at_clear = 14.0 * (their_at - my_out);
        let v_side = 1.8 * my_out;
        let need = (14.0 - v_side).powi(2) / (2.0 * gap_at_clear) - 1.8;
        let need_2m = (14.0 - v_side).powi(2) / (2.0 * (gap_at_clear - 2.0)) - 1.8;
        assert!(
            gives_way(&g, 1, 2) && !gives_way(&g, 2, 1),
            "side car let go {:.2} s before a 14 m/s main car (accepted gap {:.1} s): main car {:.1} m behind at clear, closing {:.1} m/s, must brake {:.1} m/s² ({:.1} with a 2 m gap)",
            their_at - my_out,
            JM_TIMEGAP_MINOR,
            gap_at_clear,
            14.0 - v_side,
            need,
            need_2m
        );
    }

    /// `behind_in_my_lane` takes the first lane of the foe's way that is mine and does
    /// not ask whether the foe passes the meeting place before it gets there: a main road
    /// car (right of way) 15 m out at 10 m/s whose 150 m way comes back round into my
    /// lane (a block loop, a turning circle) is "behind me in my lane".
    #[test]
    fn review4_behind_in_my_lane_ignores_where_the_foe_reaches_my_lane() {
        let (_, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let mine0 = js.nodes[0].movements[main].foes.iter().find(|f| f.movement == side).unwrap().mine.0;
        let mut me = claim(1, side, -1.0, 0.0);
        me.lanes = vec![(1, 1.0)];
        let mut foe = claim(2, main, mine0 - 15.0 - 2.0, 10.0);
        foe.lanes = vec![(0, -foe.s), (50, 20.0), (51, 60.0), (1, 120.0)];
        let claims = [me, foe];
        let g = resolve(&js, &claims);
        // (it may go behind it, SUMO's follower case of a crossing: the main road car is
        // out before the side car is in; never in front of it)
        assert!(not_ahead_of(&g, 1), "the side car goes in front of a 10 m/s main road car 15 m out: {g:?}");
    }

    // ---- junction right of way ----

    /// Lights (0, 0) on the main road and (0, 1) on the side road, never green together.
    fn exclusive_lights() -> Context {
        let mut ctx = Context::default();
        ctx.exclusive.insert((0, 0, 1), true);
        ctx
    }

    /// The side road car (1) has its green at the line; the main road car (2, right of way by
    /// the table) rolls at `speed` towards its red `dist` m out.
    fn green_side_red_main(dist: f32, speed: f32) -> (Network, Junctions, Vec<Claim>) {
        let (net, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let mine0 = js.nodes[0].movements[main].foes.iter().find(|f| f.movement == side).unwrap().mine.0;
        let mut me = claim(1, side, -1.0, 0.0);
        me.light = Some((0, 1));
        let mut foe = claim(2, main, mine0 - dist - 2.0, speed);
        foe.light = Some((0, 0));
        foe.light_red = true;
        foe.until_go = Some(40.0);
        (net, js, vec![me, foe])
    }

    #[test]
    fn rule_a_a_foe_rolling_towards_a_red_never_green_with_mine_is_no_competitor() {
        // at 10 m/s 20 m out, at 3 m/s 5 m out, standing 1 m out: whatever its speed
        for (dist, speed) in [(20.0, 10.0), (5.0, 3.0), (1.0, 0.0), (60.0, 13.0)] {
            let (_, js, claims) = green_side_red_main(dist, speed);
            let g = resolve_with(&js, &claims, &exclusive_lights());
            assert!(grant_of(&g, 1).go, "{dist} m at {speed} m/s: {g:?}");
            // the same phase (or not known to be exclusive): held at its red short of the
            // meeting place it does not come before its green 40 s on either (SUMO: its planned
            // arrival); before, a rolling one was waited for
            let g = resolve_with(&js, &claims, &Context::default());
            assert!(grant_of(&g, 1).go, "{dist} m at {speed} m/s, same phase: {g:?}");
        }
    }

    #[test]
    fn rule_a_a_foe_inside_the_box_is_kept_and_my_red_holds_me_not_the_foes() {
        let (_, js, mut claims) = green_side_red_main(0.0, 2.0);
        // its front in the meeting place (it took the line before its light went red)
        let main = claims[1].movement;
        let f = js.nodes[0].movements[main].foes[0];
        claims[1].s = f.mine.0 + 1.0 - 2.0;
        let g = resolve_with(&js, &claims, &exclusive_lights());
        // (waited for, or followed: out at 2 m/s before I am in, SUMO's crossing follower)
        assert!(not_ahead_of(&g, 1), "{g:?}");
        // standing there at its red, past its line ahead (`red_at` 30 m on): it leaves at its
        // green, I wait for it
        let mut standing = claims.clone();
        standing[1].speed = 0.0;
        standing[1].held = true;
        standing[1].red_at = f.mine.1 + 30.0;
        let g = resolve_with(&js, &standing, &exclusive_lights());
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
        // not held by its red (it goes through: light_red false): a competitor whatever the lights
        // (waited for, or followed as SUMO's crossing follower)
        let (_, js, mut claims) = green_side_red_main(20.0, 10.0);
        claims[1].light_red = false;
        let g = resolve_with(&js, &claims, &exclusive_lights());
        assert!(not_ahead_of(&g, 1), "{g:?}");
        // my own light red: held by it, the junction holds me for nobody
        let (_, js, mut claims) = green_side_red_main(20.0, 10.0);
        claims[1].light_red = false;
        claims[0].light_red = true;
        claims[0].until_go = Some(30.0);
        let g = resolve_with(&js, &claims, &exclusive_lights());
        assert!(grant_of(&g, 1).go && grant_of(&g, 1).wait_for.is_none(), "{g:?}");
        // an unlit foe is no red foe
        let (_, js, mut claims) = green_side_red_main(20.0, 10.0);
        claims[1].light = None;
        claims[1].light_red = false;
        let g = resolve_with(&js, &claims, &exclusive_lights());
        assert!(not_ahead_of(&g, 1), "{g:?}");
    }

    /// Every red on the way through is filed with its line (`red_at`): the
    /// claimant's own red holds it only before the meeting places past that line; short of it the
    /// meeting places are decided as for anybody (a car its own red holds must not go through them).
    #[test]
    fn j71_my_red_past_my_first_meeting_place_is_not_my_go() {
        let (_, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let f = *js.nodes[0].movements[side].foes.iter().find(|f| f.movement == main).unwrap();
        let mut me = claim(1, side, -1.0, 0.0);
        me.light_red = true;
        me.until_go = Some(30.0);
        // the main road car out at 10 m/s, in the meeting place while I would be (a conflict)
        let foe = claim(2, main, js.nodes[0].movements[main].foes[0].mine.0 - 22.0 - 2.0, 10.0);
        let t = foe_times(&me, &foe, f.mine, f.theirs, window(&foe, f.theirs.0, f.theirs.1), false);
        assert_eq!(t.case, FoeCase::Conflict, "{t:?}");
        // my red's line past the meeting place: refused for the foe, not let go by my light
        me.red_at = f.mine.1 + 5.0;
        let g = resolve(&js, &[me.clone(), foe.clone()]);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
        // my red's line before it: my light holds me there, the junction holds me for nobody
        me.red_at = f.mine.0 - 1.0;
        let g = resolve(&js, &[me, foe]);
        assert!(grant_of(&g, 1).go && grant_of(&g, 1).wait_for.is_none(), "{g:?}");
    }

    /// A foe rolling to its red whose line lies short
    /// of our meeting place (no signal group known: no `exclusive`) does not come before its green;
    /// one whose red lies past the meeting place comes through it first.
    #[test]
    fn j71_a_foe_rolling_to_its_red_short_of_the_meeting_place_comes_at_its_green() {
        let (_, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let m0 = js.nodes[0].movements[main].foes.iter().find(|f| f.movement == side).unwrap().mine.0;
        let me = claim(1, side, -1.0, 0.0);
        let mut foe = claim(2, main, m0 - 12.0 - 2.0, 6.0);
        foe.light_red = true;
        foe.until_go = Some(30.0);
        foe.red_at = m0 - 2.0;
        let g = resolve(&js, &[me.clone(), foe.clone()]);
        assert!(grant_of(&g, 1).go, "{g:?}");
        foe.red_at = m0 + 10.0;
        let g = resolve(&js, &[me, foe]);
        assert!(not_ahead_of(&g, 1) && !(grant_of(&g, 1).go && grant_of(&g, 1).follower_go == 0), "{g:?}");
    }

    /// SUMO's `computeFoeArrivalTimeBraking`: a 14 m/s foe 60 m out braking at
    /// decel 2.1 times I = 0.2 from my arrival at 1 s arrives 0.19 s later (SUMO: 4.33 s blended);
    /// a slow foe that can stop short by then is 30 s later.
    #[test]
    fn j71_impatience_brakes_the_foe_at_its_decel_times_i_from_my_arrival() {
        let (t, v) = foe_arrival_braking(1.0, 60.0 / 14.0, 60.0, 2.1, 0.2);
        let blended = 0.8 * (60.0 / 14.0) + 0.2 * t;
        assert!((blended - 4.33).abs() < 0.02 && v > 0.0 && v < 14.0, "{t} {v} {blended}");
        let (t, v) = foe_arrival_braking(1.0, 30.0, 60.0, 2.1, 0.2);
        assert!((t - 60.0).abs() < 1e-3 && v == 0.0, "{t} {v}");
        // arriving after the foe: no braking for me (SUMO's guard)
        assert_eq!(foe_arrival_braking(5.0, 4.0, 60.0, 2.1, 0.2).0, 4.0);
        // in the resolver: waiting 36 s (I = 0.2) the claimant reads the foe's arrival so
        let (_, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let f = *js.nodes[0].movements[side].foes.iter().find(|f| f.movement == main).unwrap();
        let mut me = claim(1, side, -1.0, 0.0);
        me.waiting = 36.0;
        let mut foe = claim(2, main, f.theirs.0 - 60.0 - 2.0, 14.0);
        foe.v_max = 14.0;
        foe.decel = 2.1;
        let t = foe_times(&me, &foe, f.mine, f.theirs, window(&foe, f.theirs.0, f.theirs.1), false);
        let (tb, _) = foe_arrival_braking(t.my_in, t.their_in, 60.0, 2.1, 0.2);
        assert!((t.their_in_impatient - (0.8 * t.their_in + 0.2 * tb)).abs() < 1e-3 && t.their_in_impatient > t.their_in, "{t:?}");
    }

    /// SUMO's leader case into one lane reads the foe's `arrivalSpeedBraking`: a
    /// foe at its red 60 s from green (it can stop short: 0 m/s) does not make a Conflict of a
    /// claimant out of the merge long before it comes.
    #[test]
    fn j71_the_leader_case_reads_the_foes_braking_arrival_speed() {
        let (_, js) = merge_a_gives_way();
        let f = *js.nodes[0].movements[0].foes.iter().find(|f| f.movement == 1).unwrap();
        let me = claim(1, 0, f.mine.0 - 4.0 - 2.0, 6.0);
        let foe = claim(2, 1, f.theirs.0 - 30.0 - 2.0, 8.0);
        let w = window(&foe, f.theirs.0, f.theirs.1);
        let t = foe_times(&me, &foe, f.mine, f.theirs, (w.0 + 60.0, w.1 + 60.0), true);
        assert_eq!(t.case, FoeCase::Leader, "{t:?}");
    }

    /// SUMO's follower case: at a crossing the foe out before I am in is enough;
    /// into one lane it needs `JM_TIMEGAP_MINOR` and safe speeds; one that does not come (stopped
    /// short) is the follower outright, never a conflict through `unsafeMergeSpeeds`.
    #[test]
    fn j71_the_follower_gap_is_for_one_lane_only_and_a_claimant_that_does_not_come_follows() {
        let (_, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let f = *js.nodes[0].movements[side].foes.iter().find(|f| f.movement == main).unwrap();
        let me = claim(1, side, f.mine.0 - 8.0 - 2.0, 5.0);
        // the foe out of the stretch 0.5 s before I am in
        let my_in = window(&me, f.mine.0, f.mine.1).0;
        let foe = claim(2, main, f.theirs.0 - 2.0, 8.0);
        let their = window(&foe, f.theirs.0, f.theirs.1);
        let shifted = (their.0 + (my_in - 0.5 - their.1), my_in - 0.5);
        assert_eq!(foe_times(&me, &foe, f.mine, f.theirs, shifted, false).case, FoeCase::Follower);
        assert_eq!(foe_times(&me, &foe, f.mine, f.theirs, shifted, true).case, FoeCase::Conflict);
        let mut stopped = me.clone();
        stopped.clear_ahead = 1.0;
        assert_eq!(foe_times(&stopped, &foe, f.mine, f.theirs, (0.0, f32::INFINITY), true).case, FoeCase::Follower);
    }

    /// A (0, from (-10, -2)) and B (1, from (-10, 2)) merge into C (2) at the same angle from
    /// either side: no rule of the table separates them.
    fn zipper_object() -> (Network, Junctions) {
        object(&[(&[(-10.0, -2.0), (0.0, 0.0)], 0), (&[(-10.0, 2.0), (0.0, 0.0)], 0), (&[(0.0, 0.0), (10.0, 0.0)], 0)])
    }

    /// A claim on piece `m` of the zipper object, its front `to_end` m before the merge's end.
    fn zip_claim(js: &Junctions, vehicle: usize, m: usize, to_end: f32, speed: f32) -> Claim {
        let end = js.nodes[0].movements[m].foes.iter().find(|f| f.kind == FoeKind::Merge).unwrap().mine.1;
        let mut c = claim(vehicle, m, end - to_end - 2.0, speed);
        c.chain = vec![(m, 0.0), (2, 10.2)];
        c
    }

    /// The zipper grant of claim `v` against `foe` (vehicle), if it is the cap's foe.
    fn capped_by(g: &[Grant], v: usize, foe: usize) -> bool {
        grant_of(g, v).zipper.is_some_and(|z| z.foe == foe)
    }

    /// SUMO's zipper: two merging that cross nowhere else are never refused for each
    /// other; the one there first goes on, the other is capped to arrive `tau` after its rear
    /// (`getZipperSpeed`), and of a pair exactly one is capped by the other.
    #[test]
    fn j72_a_zipper_never_refuses_and_exactly_one_of_a_pair_is_capped() {
        let (_, js) = zipper_object();
        let mut bad = Vec::new();
        for da in [0.5f32, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0] {
            for off in [-3.0f32, -2.1, -1.0, -0.2, 0.0, 0.2, 1.0, 2.1, 3.0] {
                for (va, vb) in [(8.0f32, 8.0f32), (5.0, 10.0), (10.0, 5.0), (0.0, 8.0), (8.0, 0.0), (1.4, 1.6), (0.0, 0.0)] {
                    let db = da + off;
                    if db < 0.2 {
                        continue;
                    }
                    let claims = [zip_claim(&js, 1, 0, da, va), zip_claim(&js, 2, 1, db, vb)];
                    let g = resolve(&js, &claims);
                    let refused = !grant_of(&g, 1).go || !grant_of(&g, 2).go;
                    let (a, b) = (capped_by(&g, 1, 2), capped_by(&g, 2, 1));
                    if refused || a == b {
                        bad.push(format!("A {da} m @{va}, B {db} m @{vb}: {g:?}"));
                    }
                }
            }
        }
        assert!(bad.is_empty(), "{} cases, e.g. {:?}", bad.len(), &bad[..bad.len().min(4)]);
        // the nearer (first) goes on, the later one is capped, braking for it
        let claims = [zip_claim(&js, 1, 0, 3.0, 6.0), zip_claim(&js, 2, 1, 6.0, 6.0)];
        let g = resolve(&js, &claims);
        let z = grant_of(&g, 2).zipper.expect("B capped");
        assert!(grant_of(&g, 1).zipper.is_none() && z.foe == 1 && z.arrive < 0.0 && !z.stop, "{g:?}");
    }

    /// J11-5 (SUMO's zipper reach and open priority link, `getZipperSpeed`, `havePriority`,
    /// `willPass`): a car with the right of way over the foe's stream, standing short of the
    /// merge, is not capped by a creeping foe of that stream that arrives first; a foe rolling up
    /// to its own red short of the stretch caps nobody before its green; a foe whose plan stops
    /// behind a standing lead short of the stretch caps nobody; and outside its zipper reach
    /// (`zipper_reach`) a car is capped by no foe short of the stretch.
    #[test]
    fn j115_the_zipper_caps_only_within_reach_never_a_priority_car_nor_for_a_foe_that_stops_short() {
        let (_, js) = merge_a_gives_way();
        let f = *js.nodes[0].movements[0].foes.iter().find(|f| f.kind == FoeKind::Merge).unwrap();
        assert!(f.yields);
        let len = f.mine.1 - f.mine.0;
        // (a) the main road car (way 1) stands 3 m short of the stretch; the merger (way 0)
        // creeps at 1 m/s 1 m short and arrives first: the merger is no cap of the main road car
        let main = zip_claim(&js, 1, 1, len + 3.0, 0.0);
        let creeper = zip_claim(&js, 2, 0, len + 1.0, 1.0);
        let g = resolve(&js, &[main.clone(), creeper.clone()]);
        assert!(!capped_by(&g, 1, 2), "{g:?}");
        // ... the merger in the stretch, rolling: the main road car's link leader - followed,
        // with no position hold at the start (SUMO's `jmAdvance`); between equals it holds
        let inside = zip_claim(&js, 2, 0, len - 2.0, 3.0);
        let g = resolve(&js, &[main.clone(), inside]);
        assert!(capped_by(&g, 1, 2) && grant_of(&g, 1).zipper.unwrap().hold.is_none(), "{g:?}");
        // ... the same pair between equals: the creeper, first, caps the standing car
        let (_, eq) = zipper_object();
        let leq = eq.nodes[0].movements[0].foes.iter().find(|f| f.kind == FoeKind::Merge).unwrap().mine;
        let g = resolve(&eq, &[zip_claim(&eq, 1, 1, leq.1 - leq.0 + 3.0, 0.0), zip_claim(&eq, 2, 0, leq.1 - leq.0 + 1.0, 1.0)]);
        assert!(capped_by(&g, 1, 2), "{g:?}");
        let g = resolve(&eq, &[zip_claim(&eq, 1, 1, leq.1 - leq.0 + 3.0, 0.0), zip_claim(&eq, 2, 0, leq.1 - leq.0 - 2.0, 3.0)]);
        assert!(grant_of(&g, 1).zipper.is_some_and(|z| z.foe == 2 && z.hold.is_some()), "{g:?}");
        // (b) between equals, a foe rolling at 6 m/s up to its red 0.5 m short of the stretch,
        // green in 60 s, caps nobody; green in 0.5 s and nearer, it does
        let me = zip_claim(&eq, 1, 1, leq.1 - leq.0 + 8.0, 0.0);
        let mut red = zip_claim(&eq, 2, 0, leq.1 - leq.0 + 6.0, 6.0);
        red.light_red = true;
        red.until_go = Some(60.0);
        red.red_at = red.s + red.front + 5.5;
        assert!(!capped_by(&resolve(&eq, &[me.clone(), red.clone()]), 1, 2));
        red.until_go = Some(0.5);
        assert!(capped_by(&resolve(&eq, &[me.clone(), red.clone()]), 1, 2));
        // (d) a foe rolling at 6 m/s up to a standing lead whose rear lies 1 m ahead of its
        // front, short of the stretch: no cap; the lead 10 m on, past the start: it caps me
        let mut queued = zip_claim(&eq, 2, 0, leq.1 - leq.0 + 6.0, 6.0);
        queued.lead_gap = Some((1.0, 0.0));
        assert!(!capped_by(&resolve(&eq, &[me.clone(), queued.clone()]), 1, 2));
        queued.lead_gap = Some((10.0, 0.0));
        assert!(capped_by(&resolve(&eq, &[me.clone(), queued.clone()]), 1, 2));
        // (c) standing 26 m short of the stretch (reach `APPROACH_WINDOW` at rest), a foe
        // rolling up nearer caps nothing of mine; 15 m short it does
        let far = zip_claim(&eq, 1, 1, leq.1 - leq.0 + 26.0, 0.0);
        assert!(zipper_reach(&far) < 26.0);
        let near_foe = zip_claim(&eq, 2, 0, leq.1 - leq.0 + 3.0, 5.0);
        assert!(!capped_by(&resolve(&eq, &[far, near_foe.clone()]), 1, 2));
        let near = zip_claim(&eq, 1, 1, leq.1 - leq.0 + 15.0, 0.0);
        assert!(capped_by(&resolve(&eq, &[near, near_foe]), 1, 2));
    }

    /// A zipper partner standing at its red is ahead only when its green and its
    /// start from rest bring it to the merge before me (SUMO's zipper ignores a later foe that can
    /// brake, and a stopped one can): a minute from green it caps nobody, whichever way the table
    /// orders the pair and however close to the merge it stands; half a second from green and
    /// nearer the merge, it is ahead - unless I have the right of way over its stream (my link is
    /// open, SUMO's `havePriority`); moving foes not stopping at a red keep the full zipper.
    #[test]
    fn j91_a_zipper_partner_standing_at_its_red_caps_me_only_when_its_green_brings_it_first() {
        // (equal ways, and a merge where the way on piece 0 gives way)
        for (_, js) in [zipper_object(), merge_a_gives_way()] {
            // (the fronts short of the merge's stretch: `len` from its start to its end)
            let f = *js.nodes[0].movements[0].foes.iter().find(|f| f.kind == FoeKind::Merge).unwrap();
            let len = f.mine.1 - f.mine.0;
            for (me, them) in [(0usize, 1usize), (1, 0)] {
                for (my_d, their_d) in [(len + 3.0, len + 5.0), (len + 3.0, len + 2.0), (len + 6.0, len + 0.5)] {
                    let a = zip_claim(&js, 1, me, my_d, 0.0);
                    let mut b = zip_claim(&js, 2, them, their_d, 0.0);
                    b.light_red = true;
                    b.until_go = Some(60.0);
                    b.red_at = b.s + b.front + 0.5;
                    let g = resolve(&js, &[a.clone(), b.clone()]);
                    assert!(!capped_by(&g, 1, 2), "me on {me} {my_d} m, red foe {their_d} m, 60 s: {g:?}");
                    // (its green before I can be there, and it nearer: it is ahead)
                    b.until_go = Some(0.5);
                    let g = resolve(&js, &[a.clone(), b.clone()]);
                    let my_priority = js.nodes[0].movements[them].foes.iter().any(|f| f.movement == me && f.yields);
                    if their_d + 1.0 < my_d {
                        assert_eq!(capped_by(&g, 1, 2), !my_priority, "me {my_d} m, foe {their_d} m green in 0.5 s: {g:?}");
                    }
                }
            }
            // (a moving foe not held by a red keeps the full zipper: the nearer moving one is ahead)
            let a = zip_claim(&js, 1, 0, len + 6.0, 0.0);
            let b = zip_claim(&js, 2, 1, len + 3.0, 5.0);
            assert!(capped_by(&resolve(&js, &[a, b]), 1, 2));
        }
    }

    /// SUMO's `willPass`: a foe standing short of the meeting place that this
    /// frame refuses (it gives way to a moving stream that does not cross me) is not coming, and I
    /// do not wait for it although it has the right of way over me; the order makes it safe: the
    /// claimants standing refused last frame are decided first, an undecided foe counts. Granted
    /// last frame (not decided first), it still counts. Two that would drop each other: exactly
    /// one goes.
    #[test]
    fn j81_a_foe_standing_refused_this_frame_short_of_the_meeting_place_does_not_hold_me() {
        // X eastbound at y = 5 (low), Y northbound at x = 0 (middle), Z eastbound at y = -5 (high):
        // Y crosses both, X and Z cross nothing of each other; X yields to Y, Y to Z
        let mut net = Network::default();
        net.lanes.push(piece((-10.0, 5.0), (10.0, 5.0), 0, 64.0));
        net.lanes.push(piece((0.0, -10.0), (0.0, 10.0), 1, 128.0));
        net.lanes.push(piece((-10.0, -5.0), (10.0, -5.0), 2, 192.0));
        net.link(1.5);
        net.compute_conflicts();
        let js = Junctions::compile(&net);
        let m = |l: usize| js.nodes[0].movement_of(l).unwrap();
        let start = |a: usize, b: usize| js.nodes[0].movements[m(a)].foes.iter().find(|f| f.movement == m(b)).unwrap().mine.0;
        // (X rolling up at 5 m/s, 12 m out: not one of the second pass's standing contenders)
        let mut x = claim(1, m(0), start(0, 1) - 12.0 - 2.0, 5.0);
        let mut y = claim(2, m(1), start(1, 2) - 1.0 - 2.0, 0.0);
        y.refused_last = true;
        // (Z rolling at 8 m/s, 10 m out: Y waits for it)
        let z = claim(3, m(2), start(2, 1) - 10.0 - 2.0, 8.0);
        let g = resolve(&js, &[x.clone(), y.clone(), z.clone()]);
        assert_eq!(grant_of(&g, 2).wait_for, Some(3), "Y waits for Z: {g:?}");
        assert!(grant_of(&g, 1).go, "X waits for Y, which stands refused: {g:?}");
        // (whichever order the claims come in)
        let g = resolve(&js, &[z.clone(), y.clone(), x.clone()]);
        assert!(grant_of(&g, 1).go && grant_of(&g, 2).wait_for == Some(3), "{g:?}");
        // (the control: Y granted last frame is not decided first - it counts, X waits)
        y.refused_last = false;
        let g = resolve(&js, &[x.clone(), y.clone(), z.clone()]);
        assert!(!grant_of(&g, 1).go || !grant_of(&g, 2).go, "{g:?}");
        // (no Z: Y goes, and X waits for it, refused last frame or not)
        for refused in [false, true] {
            y.refused_last = refused;
            x.refused_last = refused;
            let g = resolve(&js, &[x.clone(), y.clone()]);
            assert!(grant_of(&g, 2).go && grant_of(&g, 1).wait_for == Some(2), "refused last frame {refused}: {g:?}");
        }
    }

    /// A foe that cannot stop before its first meeting place (committed) but can wait
    /// short of a later one (`Claim::next_wait`) is not committed to that later one: the table
    /// decides there. (Otherwise it waits for oncoming left-turners its table gives way to,
    /// committed to their first meeting place well before the one with it.) With its next waiting
    /// place past my stretch, it still is.
    #[test]
    fn j4_a_foe_committed_to_its_first_meeting_place_is_not_committed_to_a_later_one_it_can_wait_short_of() {
        let (_, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let m0 = js.nodes[0].movements[main].foes.iter().find(|f| f.movement == side).unwrap().mine.0;
        let s0 = js.nodes[0].movements[side].foes.iter().find(|f| f.movement == main).unwrap().mine.0;
        // (me on the main road, standing 1 m out; the side road car rolling 6 m out at 5 m/s,
        // committed to something before: the two would meet)
        let me = claim(1, main, m0 - 1.0 - 2.0, 0.0);
        let mut foe = claim(2, side, s0 - 6.0 - 2.0, 5.0);
        foe.committed = true;
        foe.next_wait = s0 - 4.0;
        let g = resolve(&js, &[me.clone(), foe.clone()]);
        assert!(grant_of(&g, 1).go, "it can wait short of our stretch, and gives way to me: {g:?}");
        foe.next_wait = s0 + 3.0;
        let g = resolve(&js, &[me, foe]);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "it cannot wait short of it: {g:?}");
    }

    /// Two standing inside one meeting place, each in the other's way (two
    /// opposing left-turners in the box, clearing, or held there): exactly one goes first - the
    /// one the table gives the right of way, else the one further along - and the other waits for
    /// it; never a mutual wait for the ease to break.
    #[test]
    fn d7_two_inside_one_meeting_place_never_wait_for_each_other() {
        for (main, side) in [(192.0f32, 64.0f32), (128.0, 128.0)] {
            let (_, js) = four_way(main, side);
            let (a, b) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
            let fa = *js.nodes[0].movements[a].foes.iter().find(|f| f.movement == b).unwrap();
            let fb = *js.nodes[0].movements[b].foes.iter().find(|f| f.movement == a).unwrap();
            for (da, db) in [(0.5f32, 0.5f32), (0.5, 1.5), (1.5, 0.5)] {
                // (fronts `da`/`db` m into their stretches, standing, both clearing)
                let mut x = claim(1, a, fa.mine.0 + da - 2.0, 0.0);
                let mut y = claim(2, b, fb.mine.0 + db - 2.0, 0.0);
                x.clearing = true;
                y.clearing = true;
                let g = resolve(&js, &[x, y]);
                let goes = [grant_of(&g, 1).go, grant_of(&g, 2).go];
                assert_eq!(goes.iter().filter(|&&v| v).count(), 1, "{main}/{side} {da}/{db}: {g:?}");
                let waiter = if goes[0] { 2 } else { 1 };
                assert_eq!(grant_of(&g, waiter).wait_for, Some(3 - waiter), "{g:?}");
            }
        }
    }

    /// A foe standing inside a crossing's meeting place whose lead (at a gap) is
    /// me, while it is not mine, is behind me and waits for me: it does not hold me although it has
    /// the right of way and stands in the meeting place; without the gap (a lead that is one frame's
    /// perception at an overlap), or when it is my lead too, it does.
    #[test]
    fn d7_a_foe_standing_inside_behind_me_with_me_as_its_lead_does_not_hold_me() {
        let (_, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let m0 = js.nodes[0].movements[main].foes.iter().find(|f| f.movement == side).unwrap().mine.0;
        let s0 = js.nodes[0].movements[side].foes.iter().find(|f| f.movement == main).unwrap().mine.0;
        let me = claim(1, side, s0 - 1.0 - 2.0, 0.0);
        let mut foe = claim(2, main, m0 + 0.5 - 2.0, 0.0);
        foe.lead = Some(1);
        foe.lead_gap = Some((2.0, 0.0));
        assert!(grant_of(&resolve(&js, &[me.clone(), foe.clone()]), 1).go, "held by a foe standing behind me");
        foe.lead_gap = None;
        assert_eq!(grant_of(&resolve(&js, &[me.clone(), foe.clone()]), 1).wait_for, Some(2));
        foe.lead_gap = Some((2.0, 0.0));
        let mut me2 = me;
        me2.lead = Some(2);
        assert_eq!(grant_of(&resolve(&js, &[me2, foe]), 1).wait_for, Some(2));
    }

    /// A foe standing refused for a stretch PAST our meeting
    /// place still comes through ours to wait there (a refused car stops just before the stretch it
    /// waits for; inside the box at a legal waiting place): it is no "refused, not coming"
    /// foe. X eastbound at y = -5 (64), Y northbound (128) to y = 16, Z eastbound at y = 11 (192),
    /// 40 m out at 8 m/s: Y, refused last frame, waits for Z at a stretch past the Y-X one, so X
    /// waits for Y. With Z's stretch before X's (the j81 geometry) X goes.
    #[test]
    fn pass1_a_foe_refused_for_a_stretch_past_ours_still_comes_through_ours() {
        let mut net = Network::default();
        net.lanes.push(piece((-20.0, -5.0), (20.0, -5.0), 0, 64.0));
        net.lanes.push(piece((0.0, -10.0), (0.0, 16.0), 1, 128.0));
        net.lanes.push(piece((-60.0, 11.0), (20.0, 11.0), 2, 192.0));
        net.link(1.5);
        net.compute_conflicts();
        let js = Junctions::compile(&net);
        let m = |l: usize| js.nodes[0].movement_of(l).unwrap();
        let start = |a: usize, b: usize| js.nodes[0].movements[m(a)].foes.iter().find(|f| f.movement == m(b)).unwrap().mine.0;
        let x = claim(1, m(0), start(0, 1) - 1.0 - 2.0, 0.0);
        let mut y = claim(2, m(1), start(1, 0) - 1.0 - 2.0, 0.0);
        y.refused_last = true;
        let z = claim(3, m(2), start(2, 1) - 40.0 - 2.0, 8.0);
        let g = resolve(&js, &[x, y, z]);
        assert_eq!(grant_of(&g, 2).wait_for, Some(3), "Y waits for Z: {g:?}");
        assert!(grant_of(&g, 2).at.is_some_and(|a| a > start(1, 0)), "Y's stretch lies past the Y-X one: {g:?}");
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "X let into the place Y drives through: {g:?}");
        // (Z's way 4 m past X's: no place between the two where Y's body lies on neither - it comes
        // to rest short of X's, `rests_short`, and X goes)
        let mut net = Network::default();
        net.lanes.push(piece((-20.0, -5.0), (20.0, -5.0), 0, 64.0));
        net.lanes.push(piece((0.0, -10.0), (0.0, 16.0), 1, 128.0));
        net.lanes.push(piece((-60.0, -1.0), (20.0, -1.0), 2, 192.0));
        net.link(1.5);
        net.compute_conflicts();
        let js = Junctions::compile(&net);
        let m = |l: usize| js.nodes[0].movement_of(l).unwrap();
        let start = |a: usize, b: usize| js.nodes[0].movements[m(a)].foes.iter().find(|f| f.movement == m(b)).unwrap().mine.0;
        let x = claim(1, m(0), start(0, 1) - 1.0 - 2.0, 0.0);
        let mut y = claim(2, m(1), start(1, 0) - 1.0 - 2.0, 0.0);
        y.refused_last = true;
        let z = claim(3, m(2), start(2, 1) - 40.0 - 2.0, 8.0);
        let g = resolve(&js, &[x, y, z]);
        assert_eq!(grant_of(&g, 2).wait_for, Some(3), "Y waits for Z: {g:?}");
        assert!(grant_of(&g, 1).go, "Y rests short of X's place: {g:?}");
    }

    /// The queue-entry case: the foe ahead stands at the join (in its stretch): the
    /// capped one stops behind it (`ZipCap::stop`, its lead); a foe standing short of its
    /// stretch is not "at the join" (it may move off: an arrival cap, or none).
    #[test]
    fn j72_a_foe_standing_at_the_join_is_the_queue_entry_stop() {
        let (_, js) = zipper_object();
        let claims = [zip_claim(&js, 1, 0, 1.0, 0.0), zip_claim(&js, 2, 1, 12.0, 6.0)];
        let g = resolve(&js, &claims);
        let z = grant_of(&g, 2).zipper.expect("B behind the standing A");
        assert!(z.stop && z.foe == 1 && grant_of(&g, 2).go, "{g:?}");
    }

    /// Two standing side by side at the join, one held by the other's body
    /// across its way: the one holding it is ahead whatever the arrival rule says (the cap
    /// put the holder behind the held car and both stood to the end of the fixture).
    #[test]
    fn j72_a_car_held_by_the_others_body_is_the_capped_one() {
        let (_, js) = zipper_object();
        for (da, db) in [(1.0f32, 1.0f32), (0.5, 1.5), (1.5, 0.5)] {
            for held in [1usize, 2] {
                let mut claims = [zip_claim(&js, 1, 0, da, 0.0), zip_claim(&js, 2, 1, db, 0.0)];
                let holder = 3 - held;
                claims[held - 1].body_block = Some(holder as u64);
                let g = resolve(&js, &claims);
                assert!(capped_by(&g, held, holder) && !capped_by(&g, holder, held), "A {da} m, B {db} m, #{held} held: {g:?}");
            }
        }
    }

    /// At a zipper: a merger that gives way never goes in front of a main
    /// road car that could not stop behind it with its comfortable brake, even arriving first;
    /// with the main road car far enough to brake for it, the merger goes first and the main
    /// road car is the capped one. Exactly one of the two is capped either way.
    #[test]
    fn j72_a_yielding_merger_does_not_cut_in_front_of_a_main_road_car_that_cannot_brake_for_it() {
        let (mut net, _) = zipper_object();
        net.lanes[1].priority = 192.0;
        let js = Junctions::compile(&net);
        assert!(js.nodes[0].movements[0].foes[0].yields);
        // the merger (A) 1 m from the start at 4 m/s; the main road car (B) at 14 m/s
        let near_end = |d: f32| [zip_claim(&js, 1, 0, 8.0, 4.0), zip_claim(&js, 2, 1, d, 14.0)];
        // 12 m behind: B arrives later but needs (14² - 4²)/(2·2) = 45 m to come down to 4 m/s
        let g = resolve(&js, &near_end(20.0));
        assert!(capped_by(&g, 1, 2) && !capped_by(&g, 2, 1), "{g:?}");
        // 80 m behind: it can, the merger goes first; B is outside its zipper reach (it can still
        // come to rest behind a car at the start, `zipper_reach`) and adapts nothing yet
        let g = resolve(&js, &near_end(88.0));
        assert!(!capped_by(&g, 2, 1) && !capped_by(&g, 1, 2), "{g:?}");
        // within its reach and still able to brake for it: the merger (too close to stop short of
        // the stretch: on its link) is ahead and B is capped; in the stretch already, the same
        let len = { let f = js.nodes[0].movements[1].foes.iter().find(|f| f.kind == FoeKind::Merge).unwrap(); f.mine.1 - f.mine.0 };
        let g = resolve(&js, &near_end(len + 64.0));
        assert!(capped_by(&g, 2, 1) && !capped_by(&g, 1, 2), "{g:?}");
        let inside = [zip_claim(&js, 1, 0, 0.0, 4.0), zip_claim(&js, 2, 1, 88.0, 14.0)];
        let g = resolve(&js, &inside);
        assert!(capped_by(&g, 2, 1) && !capped_by(&g, 1, 2), "{g:?}");
    }

    /// Two ways side by side over two merge stretches (a side merge, then the
    /// merge into one): one order per pair, decided at the pair's first stretch, so the two
    /// stretches never cap each car for the other (per-stretch orders put each car first, and both stood).
    #[test]
    fn j3_one_zipper_order_per_pair_of_ways() {
        let (_, js) = object(&[(&[(-30.0, -1.4), (-12.0, -1.0)], 0), (&[(-12.0, -1.0), (0.0, 0.0)], 0), (&[(-30.0, 1.0), (-12.0, 0.9), (0.0, 0.0)], 0), (&[(0.0, 0.0), (10.0, 0.0)], 0)]);
        let n = &js.nodes[0].movements;
        assert!(n[2].foes.len() == 2 && n[2].foes.iter().all(|f| f.kind == FoeKind::Merge && js.is_zipper(0, 2, f)), "{:?}", n[2].foes);
        let mut both = Vec::new();
        for speed in [0.0f32, 3.0] {
            for k in 0..40 {
                let fa = 1.0 + 0.5 * k as f32;
                for lead in [-3.0f32, -1.0, 1.0, 3.0] {
                    let fb = fa + lead;
                    for (ia, ib) in [(1usize, 2usize), (2, 1)] {
                        let mut a = claim(ia, 0, fa - 2.0, speed);
                        a.chain = vec![(0, 0.0), (1, 18.0), (3, 30.0)];
                        let mut b = claim(ib, 2, fb - 2.0, speed);
                        b.chain = vec![(2, 0.0), (3, 30.0)];
                        let g = resolve(&js, &[a, b]);
                        if (capped_by(&g, ia, ib) && capped_by(&g, ib, ia)) || !grant_of(&g, ia).go || !grant_of(&g, ib).go {
                            both.push((speed, fa, fb, ia));
                        }
                    }
                }
            }
        }
        assert!(both.is_empty(), "both capped by (or refused for) each other at (speed, A's front, B's front, A's index): {both:?}");
    }



    #[test]
    fn rule_b_the_right_of_way_is_read_for_the_piece_of_my_way_the_foe_meets() {
        // my way: P1 (0) continues into P2 (1), which the side way Q (2, priority 64) merges
        // into before the exit C (3): Q gives way to P2. (The table is read for the piece of my way the foe meets, not for my
        // first piece, P1, which Q never meets: that reads "nobody gives way", and whoever came
        // first went; two such cars side by side waited for each other.)
        let (mut net, _) = object(&[(&[(-20.0, 0.0), (-10.0, 0.0)], 0), (&[(-10.0, 0.0), (0.0, 0.0)], 0), (&[(-10.0, -6.0), (0.0, 0.0)], 0), (&[(0.0, 0.0), (10.0, 0.0)], 0)]);
        net.lanes[2].priority = 64.0;
        let js = Junctions::compile(&net);
        let f = js.nodes[0].movements[2].foes.iter().find(|f| f.movement == 1).copied().unwrap();
        assert!(f.yields && f.kind == FoeKind::Merge, "{f:?}");
        let mut me = claim(1, 0, 8.0 - 2.0, 5.0);
        me.chain = vec![(0, 0.0), (1, 10.0), (3, 20.0)];
        // the side car nearer the merge, outside its meeting place, faster in
        let mut side = claim(2, 2, f.mine.0 - 1.0 - 2.0, 6.0);
        side.chain = vec![(2, 0.0), (3, net.lanes[2].length())];
        let g = resolve(&js, &[me, side]);
        assert!(grant_of(&g, 1).go && !gives_way(&g, 1, 2), "{g:?}");
        assert!(gives_way(&g, 2, 1), "{g:?}");
    }

    #[test]
    fn rule_c_a_foe_stuck_behind_people_is_dropped_whatever_my_clear_time() {
        // 1 m out, held by people (stuck): from rest after its start it would be in the
        // meeting place before I am out by the time test, but it is not coming
        let (_, js, mut claims) = held_main_road_foe(1.0, 3.0, true);
        claims[1].stuck = true;
        let g = resolve(&js, &claims);
        assert!(grant_of(&g, 1).go, "{g:?}");
        // standing at the line myself, a long bus: the same
        let mut bus = claims.clone();
        bus[0].speed = 0.0;
        bus[0].rear = 9.6;
        assert!(grant_of(&resolve(&js, &bus), 1).go);
        // it moves off: a competitor again
        let mut moving = claims.clone();
        moving[1].speed = 2.0;
        moving[1].stuck = false;
        moving[1].blocked = false;
        assert_eq!(grant_of(&resolve(&js, &moving), 1).wait_for, Some(2));
        // held by something that lets it go on a schedule (not stuck): the foe is waited for
        let (_, js, claims) = held_main_road_foe(1.0, 3.0, true);
        assert_eq!(grant_of(&resolve(&js, &claims), 1).wait_for, Some(2));
        // in the meeting place it is waited for, stuck or not
        let (_, js, mut claims) = held_main_road_foe(-3.0, 3.0, true);
        claims[1].stuck = true;
        assert_eq!(grant_of(&resolve(&js, &claims), 1).wait_for, Some(2));
        // held myself (blocked): not my call yet
        let (_, js, mut claims) = held_main_road_foe(1.0, 0.0, true);
        claims[1].stuck = true;
        claims[0].held = true;
        claims[0].blocked = true;
        assert_eq!(grant_of(&resolve(&js, &claims), 1).wait_for, Some(2));
    }

    #[test]
    fn rule_c_a_foe_refused_for_a_stuck_third_party_is_not_coming_and_one_that_moves_is() {
        // review_cross: 0 east, 1 west, 2 north, 3 south. Me northbound (2) at my line; the
        // westbound car (1) comes from my right and stands at its line, refused for the
        // southbound car (3) that stands in its meeting place held by people.
        let (_, js) = review_cross();
        let f = |m: usize, o: usize| js.nodes[0].movements[m].foes.iter().find(|f| f.movement == o).copied().unwrap();
        assert!(f(2, 1).yields && f(1, 3).yields, "{:?}", js.nodes[0].movements);
        let me = claim(1, 2, f(2, 1).mine.0 - 1.0 - 2.0, 0.0);
        let west = claim(2, 1, f(1, 2).mine.0.min(f(1, 3).mine.0) - 1.0 - 2.0, 0.0);
        let mut south = claim(3, 3, f(3, 1).mine.0 + 1.0 - 2.0, 0.0);
        south.held = true;
        south.blocked = true;
        south.stuck = true;
        let claims = [me, west, south];
        let g = resolve(&js, &claims);
        assert_eq!(grant_of(&g, 2).wait_for, Some(3), "the westbound car waits for the stuck one: {g:?}");
        assert!(grant_of(&g, 1).go, "it is not coming before the people let the other go: {g:?}");
        // the southbound car has gone on and the westbound car rolls off: a competitor again
        let mut moving = vec![claims[0].clone(), claims[1].clone()];
        moving[1].speed = 1.5;
        let g = resolve(&js, &moving);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
        // I am committed in the box when the westbound car moves off: it waits for me
        let mut inside = vec![claims[0].clone(), claims[1].clone()];
        inside[0].s = f(2, 1).mine.0 + 0.5 - 2.0;
        inside[0].speed = 3.0;
        inside[0].committed = true;
        inside[1].speed = 1.5;
        let g = resolve(&js, &inside);
        assert!(grant_of(&g, 1).go && grant_of(&g, 2).wait_for == Some(1), "{g:?}");
    }

    // ---- zero-intergreen counter-examples ----

    /// BeijingCity controller 16: light 2's yellow runs to the end of the 93 s cycle
    /// and light 0 turns green at 0 s, so the two are "never together" with an intergreen of
    /// zero. I take light 2's yellow at its last 0.2 s; the main road car (light 0, right of
    /// way) rolls at 10 m/s 12 m out, held by a red that turns green in 0.2 s. Dropping it
    /// "whatever its speed" sends me across a stream that gets its green before I am through.
    #[test]
    fn review7_rule_a_zero_intergreen_foe_rolling_at_its_red_is_still_a_competitor() {
        let ph = |v: &[(i32, f32)]| v.to_vec();
        let ctl = omsi_sim::traffic::TrafficLightController::new(
            vec![
                ph(&[(6, 50.0), (12, 0.5), (6, 0.5), (12, 0.5), (6, 0.5), (12, 0.5), (6, 0.5), (10, 3.0), (0, 1.0)]),
                ph(&[(0, 62.0), (6, 15.0), (0, 10.0)]),
                ph(&[(0, 62.0), (6, 25.0), (12, 0.5), (6, 0.5), (12, 0.5), (6, 0.5), (12, 0.5), (6, 0.5), (10, 3.0), (0, 1.0)]),
            ],
            93.0,
        );
        assert!(ctl.never_together(0, 2), "the program's pair is exclusive");
        let mut ctx = Context::default();
        ctx.exclusive.insert((0, 0, 2), ctl.never_together(0, 2));
        let (_, js, mut claims) = green_side_red_main(12.0, 10.0);
        claims[0].light = Some((0, 2));
        claims[1].light = Some((0, 0));
        claims[1].until_go = Some(0.2);
        let g = resolve_with(&js, &claims, &ctx);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "dropped a 10 m/s foe 0.2 s from its green: {g:?}");
    }


    // ---- safety counter-examples ----
    mod review9 {
        // counter-examples and probes of the safety rules
        use super::*;

        /// The side road car (1, light (0, 1), green, gives way by the table) `my_dist` m before its
        /// meeting place at `my_speed`; the main road car (2, light (0, 0)) `dist` m out at `speed`.
        fn side_and_main(my_dist: f32, my_speed: f32, dist: f32, speed: f32) -> (Network, Junctions, Vec<Claim>) {
            let (net, js) = four_way(192.0, 64.0);
            let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
            let m0 = js.nodes[0].movements[main].foes.iter().find(|f| f.movement == side).unwrap().mine.0;
            let s0 = js.nodes[0].movements[side].foes.iter().find(|f| f.movement == main).unwrap().mine.0;
            let mut me = claim(1, side, s0 - my_dist - 2.0, my_speed);
            me.light = Some((0, 1));
            let mut foe = claim(2, main, m0 - dist - 2.0, speed);
            foe.light = Some((0, 0));
            (net, js, vec![me, foe])
        }

        fn ex() -> Context {
            let mut ctx = Context::default();
            ctx.exclusive.insert((0, 0, 1), true);
            ctx
        }

        // ---------------- a foe's light ----------------

        #[test]
        fn review9_a_foe_light_turns_green_while_i_am_10m_out() {
            // its red turns green in 0.5 s; it rolls at 8 m/s 12 m out; I am 10 m out at 8 m/s
            let (_, js, mut c) = side_and_main(10.0, 8.0, 12.0, 8.0);
            c[1].light_red = true;
            c[1].until_go = Some(0.5);
            let g = resolve_with(&js, &c, &ex());
            assert_eq!(grant_of(&g, 1).wait_for, Some(2), "red foe 0.5 s from green dropped: {g:?}");
            // the next frame its light is green: an ordinary foe with right of way
            c[1].light_red = false;
            c[1].until_go = None;
            let g = resolve_with(&js, &c, &ex());
            assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
        }

        #[test]
        fn review9_a_amber_foe_that_can_stop_is_shifted_not_dropped() {
            // amber, it can stop (the sim marks it light_red); its next green is 3 s away (an
            // intergreen of 3 s); it rolls at 10 m/s 25 m out. I stand at my line (slow to clear).
            let (_, js, mut c) = side_and_main(1.0, 0.0, 25.0, 10.0);
            c[1].light_red = true;
            c[1].until_go = Some(3.0);
            let g = resolve_with(&js, &c, &ex());
            assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
            // a long red ahead of it: dropped (correct: it stops)
            c[1].until_go = Some(40.0);
            let g = resolve_with(&js, &c, &ex());
            assert!(grant_of(&g, 1).go, "{g:?}");
        }

        #[test]
        fn review9_a_two_controllers_at_one_junction_are_never_exclusive() {
            let (_, js, mut c) = side_and_main(1.0, 0.0, 20.0, 10.0);
            c[1].light = Some((1, 0));
            c[1].light_red = true;
            c[1].until_go = Some(40.0);
            let mut ctx = ex();
            ctx.exclusive.insert((1, 0, 1), true);
            let g = resolve_with(&js, &c, &ctx);
            // not exclusive (different controllers): held at its red short of the meeting
            // place for 40 s it does not come
            assert!(grant_of(&g, 1).go, "{g:?}");
        }

        #[test]
        fn review9_a_braking_foe_zero_intergreen_is_kept() {
            // slow and near: 4 m/s 5 m out, green in 0.2 s
            for (dist, speed, t) in [(5.0, 4.0, 0.2), (2.0, 1.5, 0.0), (12.0, 10.0, 0.25), (30.0, 13.0, 1.0)] {
                let (_, js, mut c) = side_and_main(1.0, 0.0, dist, speed);
                c[1].light_red = true;
                c[1].until_go = Some(t);
                let g = resolve_with(&js, &c, &ex());
                assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{dist} m {speed} m/s green in {t}: {g:?}");
            }
        }

        // ---------------- held-off chains ----------------

        /// review_cross: 0 east, 1 west, 2 north, 3 south. Me northbound (2) standing at my line;
        /// the westbound car (1, from my right: I give way to it) stands `west_out` m before our
        /// meeting place, BLOCKED by its lead (a lead that moves off on its own schedule; not stuck);
        /// the southbound car (3, it gives way to... the westbound gives way to it) stands stuck
        /// (people) OUTSIDE its meeting place with the westbound way.
        fn blocked_chain(west_out: f32, west_vehicle: usize) -> (Network, Junctions, Vec<Claim>) {
            let (net, js) = review_cross();
            let f = |m: usize, o: usize| js.nodes[0].movements[m].foes.iter().find(|f| f.movement == o).copied().unwrap();
            assert!(f(2, 1).yields && f(1, 3).yields, "{:?}", js.nodes[0].movements);
            let me = claim(1, 2, f(2, 1).mine.0 - 1.0 - 2.0, 0.0);
            let mut west = claim(west_vehicle, 1, f(1, 2).mine.0 - west_out - 2.0, 0.0);
            west.held = true;
            west.blocked = true;
            let mut south = claim(3, 3, f(3, 1).mine.0 - 3.0 - 2.0, 0.0);
            south.held = true;
            south.blocked = true;
            south.stuck = true;
            (net, js, vec![me, west, south])
        }

        #[test]
        fn review9_c_a_foe_blocked_by_a_moving_off_lead_and_refused_for_a_stuck_third_is_dropped() {
            let (_, js, claims) = blocked_chain(1.0, 2);
            let g = resolve(&js, &claims);
            // the westbound car's own refusal is only because it is blocked (a blocked claimant drops
            // nobody); its real holder is its lead, which lets it go on a schedule: the time test from
            // its rest says wait (it is 1 m out, I need seconds to clear from rest)
            let without_third = resolve(&js, &claims[..2]);
            assert_eq!(grant_of(&without_third, 1).wait_for, Some(2), "control: without the third car I wait for it: {without_third:?}");
            assert_eq!(grant_of(&g, 1).wait_for, Some(2), "dropped a foe 1 m out whose holder (its lead) moves off on a schedule: {g:?}");
        }

        #[test]
        fn review9_c_e_a_standing_player_refused_for_a_stuck_third_is_dropped() {
            // the same with the westbound vehicle a player (its grant is never read: is_player)
            let (_, js, claims) = blocked_chain(1.0, usize::MAX);
            let g = resolve(&js, &claims);
            assert_eq!(grant_of(&g, 1).wait_for, Some(usize::MAX), "dropped the standing player 1 m from the meeting place because its (unread) grant waits for a stuck car: {g:?}");
        }

        #[test]
        fn review9_c_held_off_chain_depth_and_a_moving_fourth() {
            // held_off over a synthetic first pass: k -> w1 -> w2 -> w3 (by wait_for)
            let mk = |v: usize, speed: f32, stuck: bool| {
                let mut c = claim(v, 0, -5.0, speed);
                c.stuck = stuck;
                c
            };
            let me = 99usize;
            let claims = vec![mk(10, 0.0, false), mk(11, 0.0, false), mk(12, 0.0, false), mk(13, 4.0, false), mk(14, 0.0, true), mk(me, 0.0, false)];
            let claim_of: hashbrown::HashMap<usize, usize> = claims.iter().enumerate().map(|(k, c)| (c.vehicle, k)).collect();
            let mut plain: Vec<Grant> = claims.iter().map(|c| Grant::go(c.vehicle)).collect();
            plain[0] = Grant::refused(10, Some(11), Hold::Foe);
            plain[1] = Grant::refused(11, Some(12), Hold::Foe);
            plain[2] = Grant::refused(12, Some(13), Hold::Foe);
            // the fourth (13) is moving: not held off
            assert!(!held_off(&claims, &claim_of, Some(&plain), 0, me), "a chain ending in a moving car");
            // the fourth stuck (via 13 -> 14 would be a fifth): 12 -> 14 directly, three steps
            plain[2] = Grant::refused(12, Some(14), Hold::Foe);
            assert!(held_off(&claims, &claim_of, Some(&plain), 0, me));
            // a ring back to k: not held off
            plain[2] = Grant::refused(12, Some(10), Hold::Foe);
            assert!(!held_off(&claims, &claim_of, Some(&plain), 0, me));
        }

        /// A frame-by-frame kinematic run of two claims through the four-way crossing: the side road
        /// car (me) obeys its grants (brakes up to 6 m/s^2 for a stop 1 m before the meeting place; no
        /// stop if it is past that), the main road car accelerates at 2 m/s^2 once its lead moves off
        /// at t = 0 (its lead's speed 2t; stuck while the lead is under 0.3 m/s and it stands) and
        /// obeys its grants as well. Returns the time both bodies are in the meeting place together.
        fn kinematic(my_dist: f32, my_speed: f32, foe_dist: f32, foe_obeys: bool) -> (f32, Vec<String>) {
            let (_, js, mut c) = side_and_main(my_dist, my_speed, foe_dist, 0.0);
            c[0].light = None;
            c[1].light = None;
            let side = c[0].movement;
            let main = c[1].movement;
            let fm = js.nodes[0].movements[side].foes.iter().find(|f| f.movement == main).copied().unwrap();
            let ff = js.nodes[0].movements[main].foes.iter().find(|f| f.movement == side).copied().unwrap();
            let dt = 1.0 / 30.0;
            let mut t = 0.0f32;
            let mut both = 0.0f32;
            let mut log = Vec::new();
            let inside = |c: &Claim, f: &crate::junction::Foe| c.s + c.front >= f.mine.0 && c.s - c.rear <= f.mine.1;
            let mut was_standing = [false, true];
            while t < 12.0 {
                let lead_v = 2.0 * t;
                // the foe: held by its lead until it moves; stuck while that lead stands
                c[1].held = lead_v < 0.3;
                c[1].blocked = c[1].speed < 0.3 && c[1].held;
                c[1].stuck = c[1].blocked && lead_v < 0.3;
                for k in 0..2 {
                    c[k].was_standing = was_standing[k];
                    let f = if k == 0 { fm } else { ff };
                    let room = f.mine.0 - (c[k].s + c[k].front) - 0.6;
                    c[k].committed = (c[k].s + c[k].front >= f.mine.0 - 0.5) && c[k].s - c[k].rear <= f.mine.1 && c[k].speed > 1.0
                        || (c[k].speed > 1.0 && room > -1.1 && room < c[k].speed * c[k].speed / (2.0 * 8.0 * 0.7));
                }
                let g = resolve(&js, &c);
                for k in 0..2 {
                    let gr = grant_of(&g, c[k].vehicle);
                    let f = if k == 0 { fm } else { ff };
                    let stop = f.mine.0 - 1.0 - (c[k].s + c[k].front);
                    let want = if k == 0 { 8.0f32.max(my_speed) } else { 12.0 };
                    let acc = if k == 1 && lead_v < 0.3 {
                        -10.0
                    } else if !gr.go && (k == 0 || foe_obeys) && stop > -0.3 {
                        // brake for the stop
                        let need = c[k].speed * c[k].speed / (2.0 * stop.max(0.05));
                        -(need.min(6.0)).max(0.0) - if stop < 0.2 { 6.0 } else { 0.0 }
                    } else if c[k].speed < want {
                        if k == 1 { 2.0 } else { 1.8 }
                    } else {
                        0.0
                    };
                    c[k].speed = (c[k].speed + acc * dt).max(0.0);
                    c[k].s += c[k].speed * dt;
                    was_standing[k] = c[k].speed < 0.3 || (was_standing[k] && c[k].speed <= 1.0);
                    if (t * 30.0).round() as i32 % 6 == 0 {
                        log.push(format!("t {t:.2} #{} s {:.2} v {:.2} go {} wf {:?} committed {} stuck {} blocked {}", c[k].vehicle, c[k].s, c[k].speed, gr.go, gr.wait_for, c[k].committed, c[k].stuck, c[k].blocked));
                    }
                }
                if inside(&c[0], &fm) && inside(&c[1], &ff) {
                    both += dt;
                }
                t += dt;
            }
            (both, log)
        }

        #[test]
        fn review9_c_stuck_foe_whose_lead_moves_off_with_me_20m_out_at_8() {
            // the brief's case: the foe 6 m from the meeting place, me 20 m out at 8 m/s
            let (both, log) = kinematic(20.0, 8.0, 6.0, true);
            assert!(both == 0.0, "both in the meeting place for {both:.2} s\n{}", log.join("\n"));
            // the foe does not read its grant (a human, or a grant ignored): when am I committed?
            let (both, log) = kinematic(20.0, 8.0, 6.0, false);
            assert!(both == 0.0, "(foe ignoring its grant) both in the meeting place for {both:.2} s\n{}", log.join("\n"));
        }

        // ---------------- clearing car against a fresh green ----------------

        #[test]
        fn review9_f_clearing_car_vs_a_fresh_green_foe_rolling_at_8() {
            // me on the side road (light (0, 1); I give way by the table) clearing: committed, in the
            // box 3 m short of the meeting place, crawling at 2 m/s; the main road foe (light (0, 0))
            // green for 3 s, rolling at 8 m/s: it must wait for me although it has the right of way
            let (_, js) = four_way(192.0, 64.0);
            let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
            let m0 = js.nodes[0].movements[main].foes.iter().find(|f| f.movement == side).unwrap().mine.0;
            let s0 = js.nodes[0].movements[side].foes.iter().find(|f| f.movement == main).unwrap().mine.0;
            for foe_out in [30.0f32, 20.0, 8.0] {
                let mut clr = claim(1, side, s0 - 3.0 - 2.0, 2.0);
                clr.committed = true;
                clr.light = Some((0, 1));
                let mut foe = claim(2, main, m0 - foe_out - 2.0, 8.0);
                foe.light = Some((0, 0));
                // (8 m out at 8 m/s it cannot stop any more: committed itself)
                foe.committed = 8.0 * 8.0 / (2.0 * 8.0 * 0.7) > foe_out - 0.6;
                let g = resolve_with(&js, &[clr, foe.clone()], &ex());
                assert!(grant_of(&g, 1).go, "{g:?}");
                if !foe.committed {
                    assert_eq!(grant_of(&g, 2).wait_for, Some(1), "{foe_out} m out: the fresh green does not wait for the clearing car: {g:?}");
                }
            }
        }

        /// A car in the box at the end of its green clears first: standing at a
        /// waiting place inside (it could stop, so it is not "clearing"), with its light now red,
        /// against the main road's fresh green standing at its line or rolling up: the fresh green
        /// waits for it, and it does not wait for the fresh green; with lights that may be green
        /// together, the table decides.
        #[test]
        fn d7_a_car_in_the_box_at_the_end_of_its_green_goes_before_the_fresh_green() {
            let (_, js) = four_way(192.0, 64.0);
            let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
            let m0 = js.nodes[0].movements[main].foes.iter().find(|f| f.movement == side).unwrap().mine.0;
            let s0 = js.nodes[0].movements[side].foes.iter().find(|f| f.movement == main).unwrap().mine.0;
            for (foe_out, foe_v) in [(1.0f32, 0.0f32), (15.0, 6.0), (30.0, 10.0)] {
                let mut inbox = claim(1, side, s0 - 2.0 - 2.0, 0.0);
                inbox.light = Some((0, 1));
                inbox.in_box_at_turn = true;
                let mut foe = claim(2, main, m0 - foe_out - 2.0, foe_v);
                foe.light = Some((0, 0));
                let g = resolve_with(&js, &[inbox.clone(), foe.clone()], &ex());
                assert!(grant_of(&g, 1).go, "{foe_out} m out at {foe_v}: the car in the box waits: {g:?}");
                assert_eq!(grant_of(&g, 2).wait_for, Some(1), "{foe_out} m out at {foe_v}: the fresh green goes: {g:?}");
                // (lights not known to be exclusive: the table - the main road - decides)
                let g = resolve_with(&js, &[inbox, foe], &Context::default());
                assert!(!grant_of(&g, 1).go || grant_of(&g, 2).wait_for.is_none(), "{g:?}");
            }
        }

        /// The chain of `blocked_chain` run frame by frame: I (northbound, from rest at my line) obey
        /// my grants; the westbound vehicle (a player: it reads no grant) stands 1 m out and moves
        /// off at `t_go` at 1.5 m/s^2 (a bus pulling away); the southbound car stays stuck.
        fn chain_kinematic(t_go: f32, west: usize) -> (f32, Vec<String>) {
            let (_, js, mut c) = blocked_chain(1.0, west);
            let f = |m: usize, o: usize| js.nodes[0].movements[m].foes.iter().find(|f| f.movement == o).copied().unwrap();
            let (fm, fw) = (f(2, 1), f(1, 2));
            let inside = |c: &Claim, f: &crate::junction::Foe| c.s + c.front >= f.mine.0 && c.s - c.rear <= f.mine.1;
            let dt = 1.0 / 30.0;
            let (mut t, mut both) = (0.0f32, 0.0f32);
            let mut log = Vec::new();
            let mut ws = [true, true, true];
            while t < 10.0 {
                for k in 0..3 {
                    c[k].was_standing = ws[k];
                }
                c[1].blocked = c[1].speed < 0.3 && t < t_go;
                c[1].held = c[1].blocked;
                let room = fm.mine.0 - (c[0].s + c[0].front) - 0.6;
                c[0].committed = c[0].speed > 1.0 && room < c[0].speed * c[0].speed / (2.0 * 8.0 * 0.7);
                let g = resolve(&js, &c);
                let gr = grant_of(&g, 1);
                let stop = fm.mine.0 - 1.0 - (c[0].s + c[0].front);
                let acc0 = if !gr.go && stop > -0.3 { -6.0 } else if c[0].speed < 10.0 { 1.8 } else { 0.0 };
                c[0].speed = (c[0].speed + acc0 * dt).max(0.0);
                c[0].s += c[0].speed * dt;
                if t >= t_go {
                    let gw = grant_of(&g, west);
                    let wstop = fw.mine.0 - 1.0 - (c[1].s + c[1].front);
                    // an AI car obeys its grant; a player does not
                    let a = if west != usize::MAX && !gw.go && wstop > -0.3 { -6.0 } else { 1.5 };
                    c[1].speed = (c[1].speed + a * dt).max(0.0);
                    c[1].s += c[1].speed * dt;
                }
                for k in 0..2 {
                    ws[k] = c[k].speed < 0.3 || (ws[k] && c[k].speed <= 1.0);
                }
                if (t * 30.0).round() as i32 % 6 == 0 {
                    log.push(format!("t {t:.2} me s {:.2} v {:.2} go {} wf {:?} | west s {:.2} v {:.2}", c[0].s, c[0].speed, gr.go, gr.wait_for, c[1].s, c[1].speed));
                }
                if inside(&c[0], &fm) && inside(&c[1], &fw) {
                    both += dt;
                }
                t += dt;
            }
            (both, log)
        }

        #[test]
        fn review9_e_a_standing_player_dropped_through_the_chain_moves_off_into_me() {
            for t_go in [1.0f32, 2.0, 3.0] {
                let (both, log) = chain_kinematic(t_go, usize::MAX);
                assert!(both == 0.0, "player moving off at {t_go} s: both in the meeting place for {both:.2} s\n{}", log.join("\n"));
            }
        }

        #[test]
        fn review9_c_the_same_chain_with_an_ai_car_that_obeys_its_grant() {
            for t_go in [1.0f32, 2.0, 3.0] {
                let (both, log) = chain_kinematic(t_go, 2);
                assert!(both == 0.0, "AI moving off at {t_go} s: both in the meeting place for {both:.2} s\n{}", log.join("\n"));
            }
        }


        #[test]
        fn review9_d_standing_half_a_metre_short_of_a_crossing_counts_as_in_it() {
            // (A side
            // road car stands, uncommitted, its front 0.4 m short of the main road's meeting place
            // (stopped by its lead, say); the main road car comes at 10 m/s 20 m out and sees it
            // outside the meeting place and not committed.
            let (_, js, mut c) = side_and_main(0.4, 0.0, 20.0, 10.0);
            c[0].light = None;
            c[1].light = None;
            let g = resolve(&js, &c);
            let both = grant_of(&g, 1).go && grant_of(&g, 2).go;
            assert!(!both, "both go: {g:?}");
        }
        #[test]
        fn review9_followup_two_clearing_claimants_the_table_decides_and_one_goes() {
            // both clearing the box (committed to the others), neither in the meeting place:
            // each let the other out first and both stood; the table decides between them
            let (_, js) = four_way(192.0, 64.0);
            let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
            let m0 = js.nodes[0].movements[main].foes.iter().find(|f| f.movement == side).unwrap().mine.0;
            let s0 = js.nodes[0].movements[side].foes.iter().find(|f| f.movement == main).unwrap().mine.0;
            let mut a = claim(1, main, m0 - 1.0 - 2.0, 2.0);
            a.clearing = true;
            let mut b = claim(2, side, s0 - 1.0 - 2.0, 2.0);
            b.clearing = true;
            let g = resolve(&js, &[a.clone(), b.clone()]);
            assert!(grant_of(&g, 1).go && grant_of(&g, 2).wait_for == Some(1), "the main road goes, the side road waits: {g:?}");
            // one of them committed by its own geometry (it cannot stop): the other waits for it
            let mut b2 = b.clone();
            b2.committed = true;
            let g = resolve(&js, &[a, b2]);
            assert!(grant_of(&g, 2).go && grant_of(&g, 1).wait_for == Some(2), "{g:?}");
        }
    }

    #[test]
    fn j7_a_yielding_merger_goes_when_its_rear_is_past_the_stretch_start_its_gap_ahead_of_the_foe() {
        // Two lanes converging over 14 m into one (A, 0, gives way to B, 1, priority 192).
        // The car on A rolls at 4.9 m/s, its front 2.6 m short of the stretch: its rear is
        // past the start in 1.2 s, past the end in 2.9 s. The through car on B comes at
        // 9.4 m/s (accel 1.2 to 14), its front 68 m from the start: 5.5 s, 6.5 s to the end
        // (3.1 + 4 > 6.6 by the end rule, 1.7 + 4 <= 6.6
        // by the start rule). The end rule refuses (2.9 + 4 > 5.5), `merging_behind` too
        // (6.5 < 6.9), the start rule grants (1.2 + 4 <= 5.5; out of the end 2.9 + 0.6 <= 6.5)
        let (mut net, _) = object(&[(&[(-30.0, 2.2), (0.0, 0.0)], 0), (&[(-30.0, -2.2), (0.0, 0.0)], 0), (&[(0.0, 0.0), (10.0, 0.0)], 0)]);
        net.lanes[1].priority = 192.0;
        let js = Junctions::compile(&net);
        let f = *js.nodes[0].movements[0].foes.iter().find(|f| f.movement == 1).unwrap();
        assert!(f.yields && f.kind == FoeKind::Merge && f.mine.1 - f.mine.0 > 10.0, "{f:?}");
        let (a0, b0, b1) = (f.mine.0, f.theirs.0, f.theirs.1);
        let mut merger = claim(1, 0, a0 - 2.6 - 2.0, 4.9);
        merger.chain = vec![(0, 0.0), (2, 30.0)];
        // the through car with its front `to_start` m before its stretch's start
        let through = |to_start: f32, v: f32| {
            let mut c = claim(2, 1, b0 - to_start - 2.0, v);
            c.chain = vec![(1, 0.0), (2, 30.0)];
            c.v_max = 14.0;
            c.accel = 1.2;
            c
        };
        let g = resolve(&js, &[merger.clone(), through(68.0, 9.4)]);
        assert!(grant_of(&g, 1).go, "refused with the through car 5.5 s off (the end rule's 2.9 + 4 > 5.5): {g:?}");
        // people, a zebra or a standing lead stopping it inside the stretch (its front 10 m
        // on, the stretch's end 18.6 m + its length): at a zipper it goes in (it arrives first
        // and the through car can brake for it), and the through car, with the right of way,
        // is not capped by it while it is short of the stretch (its link is open); it follows
        // it once it is in the stretch
        let mut blocked_ahead = merger.clone();
        blocked_ahead.clear_ahead = 10.0;
        let g = resolve(&js, &[blocked_ahead, through(68.0, 9.4)]);
        assert!(!gives_way(&g, 2, 1) && !gives_way(&g, 1, 2), "{g:?}");
        // ... and once it stands inside the stretch for them, the through car queues behind it
        // (the queue-entry stop, `ZipCap::stop`)
        let mut inside = merger.clone();
        inside.s = a0 + 2.0;
        inside.speed = 0.0;
        inside.clear_ahead = inside.front + 0.5;
        let g = resolve(&js, &[inside, through(40.0, 9.4)]);
        assert!(grant_of(&g, 2).zipper.is_some_and(|z| z.foe == 1 && z.stop), "{g:?}");
        // 2-3 s off: it gives way still (the turner against the player; capped by it
        // at the zipper)
        let g = resolve(&js, &[merger.clone(), through(25.0, 9.4)]);
        assert!(gives_way(&g, 1, 2) && !gives_way(&g, 2, 1), "{g:?}");
        // from rest it needs longer, but its rear is out of the end `JM_TIMEGAP_MINOR` before
        // the through car is there (7 s): it goes (SUMO's leader case; the driver's own
        // 4 s refused it: 2.7 s to the start + 4 s > 5.5)
        let mut rest = merger.clone();
        rest.speed = 0.0;
        let g = resolve(&js, &[rest, through(68.0, 9.4)]);
        assert!(grant_of(&g, 1).go, "{g:?}");
        // with its front a metre into the stretch already (let go by the window, stopped as
        // the foe sped up): it goes on when its rear is out of the end its gap ahead of the
        // foe's front there: from rest 17 m to the end, 4.3 s + 4 s; the foe 112 m from the
        // end, 8.6 s (7.6 s to the start: the end rule would refuse, 8.3 > 7.6)
        let mut inside = merger.clone();
        inside.s = a0 + 1.0 - 2.0;
        inside.speed = 0.0;
        let g = resolve(&js, &[inside.clone(), through(112.0 - (b1 - b0), 9.4)]);
        assert!(grant_of(&g, 1).go, "a merger inside the stretch stood for a foe 8.6 s from its end: {g:?}");
        // ... and with the foe 90 m from the end (7.0 s): out 4.3 s + `JM_TIMEGAP_MINOR` before
        // it, and the foe, braking comfortably, could stop behind it (`unsafe_merge`): it goes
        // (SUMO's leader case; refused by the driver's own 4 s it stood in the stretch).
        // Within 3 s it stands (a side car let go 0.5 m short of the end would have a
        // 14 m/s main car braking)
        let g = resolve(&js, &[inside.clone(), through(90.0 - (b1 - b0), 9.4)]);
        assert!(grant_of(&g, 1).go, "{g:?}");
        // (in the stretch already it is ahead of the through car short of it
        // only when that one can brake for it - 40 m out at 9.4 m/s it cannot follow it
        // comfortably, so it goes first and the merger is capped; exactly one of the two)
        let g = resolve(&js, &[inside, through(40.0 - (b1 - b0), 9.4)]);
        assert!(gives_way(&g, 1, 2) && !gives_way(&g, 2, 1), "{g:?}");
        // the guard: a longer stretch (14 m, two lanes converging) that the foe's front would
        // reach the end of before my rear is past it. A slow merger (3 m/s, its rear a metre
        // past the start, fed up: gap 1 s) and a through car at 13 m/s 40 m from the end: the
        // start window alone would let it go (2 s >= 0.3 + 1), the guard refuses (3.1 s to the
        // end, my rear past it in 5 s)
        let (mut n2, _) = object(&[(&[(-30.0, 2.2), (0.0, 0.0)], 0), (&[(-30.0, -2.2), (0.0, 0.0)], 0), (&[(0.0, 0.0), (10.0, 0.0)], 0)]);
        n2.lanes[1].priority = 192.0;
        let js2 = Junctions::compile(&n2);
        let f = js2.nodes[0].movements[0].foes.iter().find(|f| f.movement == 1).unwrap();
        assert!(f.yields && f.kind == FoeKind::Merge && f.mine.1 - f.mine.0 > 10.0, "{f:?}");
        let mut slow = claim(1, 0, f.mine.0 + 1.0, 3.0);
        slow.v_max = 3.0;
        slow.chain = vec![(0, 0.0), (2, 30.0)];
        let mut fast = claim(2, 1, f.theirs.1 - 40.0 - 2.0, 13.0);
        fast.chain = vec![(1, 0.0), (2, 30.0)];
        // (the merger in the stretch is ahead only when the through car can
        // brake for it: at 13 m/s 40 m from the end it cannot follow it comfortably, so it goes
        // first and the merger is capped behind it)
        let g = resolve(&js2, &[slow.clone(), fast.clone()]);
        assert!(gives_way(&g, 1, 2) && !gives_way(&g, 2, 1), "{g:?}");
        // the through car 90 m from the end (6.9 s): past the end before it, and the window
        // open: it goes
        fast.s = f.theirs.1 - 90.0 - 2.0;
        let g = resolve(&js2, &[slow, fast]);
        assert!(grant_of(&g, 1).go, "{g:?}");
    }

    #[test]
    fn j4_a_rolling_foe_behind_a_standing_queue_that_waits_for_me_is_no_foe() {
        // the side road car (1) stands at its line with its nose over the main road; the main
        // road car (7) stands right at it with the side car as its lead; a third (2) rolls up
        // at 3 m/s, 20 m out, behind 7
        let (_, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let mine0 = js.nodes[0].movements[main].foes.iter().find(|f| f.movement == side).unwrap().mine.0;
        let mut first = claim(7, main, mine0 - 3.0 - 2.0, 0.0);
        first.lead = Some(1);
        let mut rolling = claim(2, main, mine0 - 20.0 - 2.0, 3.0);
        rolling.lead = Some(7);
        let claims = [claim(1, side, -1.0, 0.0), first, rolling];
        let g = resolve(&js, &claims);
        assert!(grant_of(&g, 1).go, "{g:?}");
        // through two standing links (a fourth car rolling up behind the third, standing)
        let mut third = claims[2].clone();
        third.speed = 0.0;
        let mut fourth = claim(3, main, mine0 - 40.0 - 2.0, 6.0);
        fourth.lead = Some(2);
        let g = resolve(&js, &[claims[0].clone(), claims[1].clone(), third, fourth]);
        assert!(grant_of(&g, 1).go, "{g:?}");
        // the one between rolling too (the queue is moving off): a competitor
        let mut moving = claims.clone();
        moving[1].speed = 1.0;
        let g = resolve(&js, &moving);
        assert!(!grant_of(&g, 1).go, "{g:?}");
        // the chain ends at somebody else (the standing one 30 m out, held by its lead): the
        // rolling one is a competitor
        let mut other = claims.clone();
        other[1].lead = Some(99);
        other[1].s = mine0 - 30.0 - 2.0;
        other[1].held = true;
        other[1].blocked = true;
        let g = resolve(&js, &other);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
    }

    #[test]
    fn j4_a_foe_standing_in_the_meeting_place_behind_me_with_me_as_its_lead_is_no_foe() {
        // A (0) and B (1) merge into C (2); A gives way to B. The bus on B rolls at 2 m/s, its
        // front a metre short of the merge; the car on A stands held inside its stretch with
        // the bus as its lead (it rolled up to the bus's queue and stopped on its rear)
        let (_, js) = merge_a_gives_way();
        let f = js.nodes[0].movements[1].foes.iter().find(|f| f.movement == 0).unwrap();
        let (b0, a0) = (f.mine.0, f.theirs.0);
        // (the bus's rear 3 m past the stretch's start, the car's front a metre in: 2 m behind
        // that rear along the merged way)
        let mut bus = claim(1, 1, b0 + 3.0 + 2.0, 2.0);
        bus.chain = vec![(1, 0.0), (2, 10.0)];
        let mut car = claim(2, 0, a0 + 1.0 - 2.0, 0.0);
        car.chain = vec![(0, 0.0), (2, 10.0)];
        car.held = true;
        car.lead = Some(1);
        let g = resolve(&js, &[bus.clone(), car.clone()]);
        assert!(grant_of(&g, 1).go, "held by a car that waits for me: {g:?}");
        // level with me (its front past my rear): it holds me
        let mut level = car.clone();
        level.s = a0 + 4.0 - 2.0;
        let g = resolve(&js, &[bus.clone(), level]);
        // (level at a zipper: exactly one of the two gives way to the other)
        assert!(gives_way(&g, 1, 2) != gives_way(&g, 2, 1), "{g:?}");
        // through a standing link: the car's lead stands behind the bus with the bus as lead
        let mut link = claim(3, 0, a0 - 5.0 - 2.0, 0.0);
        link.chain = vec![(0, 0.0), (2, 10.0)];
        link.lead = Some(1);
        let mut car2 = car.clone();
        car2.lead = Some(3);
        let g = resolve(&js, &[bus.clone(), car2, link]);
        assert!(grant_of(&g, 1).go, "{g:?}");
        // (its lead somebody else, or rolling inside with me as its lead: at the zipper
        // the one further along in the stretch goes first and the other follows it, whatever
        // its lead; before, it held me as standing in the meeting place)
        let mut other = car.clone();
        other.lead = Some(99);
        let g = resolve(&js, &[bus.clone(), other]);
        assert!(gives_way(&g, 2, 1) && !gives_way(&g, 1, 2), "{g:?}");
        let mut rolling = car.clone();
        rolling.speed = 1.0;
        rolling.held = false;
        let g = resolve(&js, &[bus, rolling]);
        assert!(gives_way(&g, 2, 1) && !gives_way(&g, 1, 2), "{g:?}");
    }

    #[test]
    fn j3_a_clearing_foe_standing_outside_the_merge_that_waits_for_me_is_not_committed() {
        // A (0) and B (1) merge into C (2); A gives way to B. The bus on B stands held with its
        // front a metre into the merge; the car on A is past its line on amber ("clearing"),
        // standing a metre short of the merge with the bus's body as its lead (the car must not wait for the
        // bus inside the stretch while the bus waits for the car "committed" by its clearing)
        let (_, js) = merge_a_gives_way();
        let f = js.nodes[0].movements[1].foes.iter().find(|f| f.movement == 0).unwrap();
        let (b0, a0) = (f.mine.0, f.theirs.0);
        let mut bus = claim(1, 1, b0 + 1.0 - 2.0, 0.0);
        bus.held = true;
        bus.chain = vec![(1, 0.0), (2, 10.0)];
        let mut car = claim(2, 0, a0 - 1.0 - 2.0, 0.0);
        car.chain = vec![(0, 0.0), (2, 10.0)];
        car.clearing = true;
        car.lead = Some(1);
        let g = resolve(&js, &[bus.clone(), car.clone()]);
        assert!(grant_of(&g, 1).go, "the bus waits for a car that waits for it: {g:?}");
        assert!(gives_way(&g, 2, 1), "the car still waits for the bus: {g:?}");
        // (the same car not held by the bus, or rolling: at the zipper the bus in the
        // stretch is ahead of the car short of it, whatever the car's clearing; before, the car
        // clearing the box was committed to the others and the bus waited for it)
        let mut free = car.clone();
        free.lead = None;
        let g = resolve(&js, &[bus.clone(), free]);
        assert!(gives_way(&g, 2, 1) && !gives_way(&g, 1, 2), "{g:?}");
        let mut rolling = car.clone();
        rolling.speed = 2.0;
        let g = resolve(&js, &[bus, rolling]);
        assert!(gives_way(&g, 2, 1) && !gives_way(&g, 1, 2), "{g:?}");
    }

    #[test]
    fn j3_the_eased_one_follows_its_partner_into_a_merge_where_it_rolls_and_stands_elsewhere() {
        // A (0) and B (1) merge into C (2); A gives way to B. Both stood at the merge; A eased
        // off for B. B rolls on at 2 m/s, its front 3 m from the merge's end; A's front is 10 m
        // from it: 3 m behind B's rear along the merged way
        let (_, js) = merge_a_gives_way();
        let f = js.nodes[0].movements[0].foes.iter().find(|f| f.movement == 1).unwrap();
        let (a1, b1) = (f.mine.1, f.theirs.1);
        let mut on_a = claim(1, 0, a1 - 10.0 - 2.0, 0.0);
        on_a.chain = vec![(0, 0.0), (2, 10.0)];
        on_a.yields_now = Some(2);
        let mut on_b = claim(2, 1, b1 - 3.0 - 2.0, 2.0);
        on_b.chain = vec![(1, 0.0), (2, 10.0)];
        let g = resolve(&js, &[on_a.clone(), on_b.clone()]);
        let ga = grant_of(&g, 1);
        assert!(!ga.go && ga.wait_for == Some(2), "{g:?}");
        let (gap, speed) = ga.follow.expect("the eased one follows the rolling partner in");
        assert!((gap - 3.0).abs() < 0.3 && speed == 2.0, "follow {gap:.1} m at {speed} m/s");
        assert!(grant_of(&g, 2).go, "the partner goes: {g:?}");
        // the partner standing: the eased one waits where it is
        let mut standing = on_b.clone();
        standing.speed = 0.0;
        let g = resolve(&js, &[on_a.clone(), standing]);
        assert_eq!(grant_of(&g, 1).follow, None, "{g:?}");
        // the partner rolling but level with me (its rear not a metre ahead of my front): no follow
        let mut level = on_b.clone();
        level.s = b1 - 7.0 - 2.0;
        let g = resolve(&js, &[on_a, level]);
        assert_eq!(grant_of(&g, 1).follow, None, "{g:?}");
        // at a crossing there is nothing to follow in behind
        let (_, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let mut eased = claim(1, side, -3.0, 0.0);
        eased.yields_now = Some(2);
        let g = resolve(&js, &[eased, claim(2, main, -3.0, 2.0)]);
        assert!(!grant_of(&g, 1).go && grant_of(&g, 1).follow.is_none(), "{g:?}");
    }

    #[test]
    fn x7_a_claim_eased_off_for_another_waits_for_it_and_is_out_of_its_way_only() {
        // the side road car (1) gives way to the main road car (2) standing at its line; the
        // main road car eased off for it (a mutual hold negotiated): it is refused, waiting for
        // the side road car, which goes
        let (_, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let m0 = js.nodes[0].movements[main].foes.iter().find(|f| f.movement == side).unwrap().mine.0;
        let s0 = js.nodes[0].movements[side].foes.iter().find(|f| f.movement == main).unwrap().mine.0;
        let a = claim(1, side, s0 - 1.0 - 2.0, 0.0);
        let b = claim(2, main, m0 - 1.0 - 2.0, 0.0);
        let g = resolve(&js, &[a.clone(), b.clone()]);
        assert!(!grant_of(&g, 1).go && grant_of(&g, 2).go, "by the table: {g:?}");
        let mut eased = b.clone();
        eased.yields_now = Some(1);
        let g = resolve(&js, &[a.clone(), eased.clone()]);
        assert!(grant_of(&g, 1).go, "the other goes: {g:?}");
        assert!(!grant_of(&g, 2).go && grant_of(&g, 2).wait_for == Some(1), "the eased one waits for it: {g:?}");
        // a third car on the main road, coming at 10 m/s 25 m out: the side road car still
        // gives way to it (only the eased one is out of its way)
        let c = claim(3, main, m0 - 25.0 - 2.0, 10.0);
        let g = resolve(&js, &[a, eased, c]);
        assert!(!grant_of(&g, 1).go && grant_of(&g, 1).wait_for == Some(3), "{g:?}");
    }

    /// A foe standing for good in the meeting place (its service, or 30 s and more)
    /// holds claimants at any distance, refused for its stretch (`Grant::at`: the plan stops them
    /// there), whatever the distance.
    #[test]
    fn j1_a_claimant_is_held_by_a_foe_standing_for_good_inside_at_any_distance() {
        let (_, js) = four_way(128.0, 128.0);
        let stretch = js.nodes[0].movements[0].foes[0].mine.0;
        let theirs = js.nodes[0].movements[0].foes[0].theirs;
        // the foe stands in the middle of the meeting place, held there (its lead, its stop)
        let mut foe = claim(2, 1, (theirs.0 + theirs.1) * 0.5, 0.0);
        foe.held = true;
        let grant = |front_to_stretch: f32, speed: f32, brake: f32, foe: &Claim| -> Grant {
            let mut me = claim(1, 0, stretch - front_to_stretch - 2.0, speed);
            me.brake = brake;
            let g = resolve(&js, &[me, foe.clone()]);
            let g = grant_of(&g, 1);
            assert!(!g.go && g.wait_for == Some(2), "not refused for it: {g:?}");
            g
        };
        // standing or rolling, at any distance: refused for the stretch (never let go: a far
        // claimant let go would count as a foe for others)
        for (d, v, b) in [(100.0, 0.0, 5.6), (60.0, 8.0, 5.6), (30.0, 0.0, 5.6), (8.0, 0.0, 5.6), (50.0, 14.0, 5.6), (50.0, 13.0, 1.8)] {
            let g = grant(d, v, b, &foe);
            assert!(g.at == Some(stretch), "{d} m off at {v} m/s: {g:?}");
        }
    }


    /// An eased car follows its rolling partner into their merge only when no other
    /// foe holds it: a third coming into the same merge with the right of way keeps it where it
    /// stands (following its partner in, the eased car would roll into a third merging between).
    #[test]
    fn j3_the_eased_one_follows_its_partner_only_when_no_third_holds_it() {
        let (mut net, _) = object(&[(&[(-10.0, -3.0), (0.0, 0.0)], 0), (&[(-10.0, 3.0), (0.0, 0.0)], 0), (&[(-10.0, 0.0), (0.0, 0.0)], 0), (&[(0.0, 0.0), (10.0, 0.0)], 0)]);
        net.lanes[2].priority = 192.0;
        let js = Junctions::compile(&net);
        let end = |m: usize| js.nodes[0].movements[m].foes.iter().find(|f| f.kind == FoeKind::Merge).unwrap().mine.1;
        let at = |v: usize, m: usize, to_end: f32, speed: f32| {
            let mut c = claim(v, m, end(m) - to_end - 2.0, speed);
            c.chain = vec![(m, 0.0), (3, end(m) + 0.2)];
            c
        };
        let mut eased = at(1, 0, 8.0, 0.0);
        eased.yields_now = Some(2);
        let partner = at(2, 1, 3.0, 4.0);
        let g = resolve(&js, &[eased.clone(), partner.clone()]);
        assert!(grant_of(&g, 1).follow.is_some() && grant_of(&g, 1).wait_for == Some(2), "no follow behind the partner alone: {g:?}");
        // a third with the right of way 6 m from the merge at 6 m/s
        let third = at(3, 2, 6.0, 6.0);
        let g = resolve(&js, &[eased, partner, third]);
        let mine = grant_of(&g, 1);
        assert!(!mine.go && mine.follow.is_none() && mine.wait_for == Some(2), "followed in with a third coming: {g:?}");
    }
}
