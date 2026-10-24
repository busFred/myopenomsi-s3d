//! [round 7 R7-1 fix 1] When the game takes a vehicle off the road, as one shared rule: the
//! app's population asks it, and a free-run replay (`Replay`, the `ab` counts) asks the same
//! function, so a run counts the removals the game would make. The rule a removal was made
//! by is written into the recording (`Record::Remove`).

use crate::perception::Seen;
use crate::sim::{Decision, Vehicle, GIVE_UP_UNSEEN, PLAYER_ID_BASE};
use serde::{Deserialize, Serialize};

/// A car that gave up (or a timetable bus let go) goes in view too once the crate's give-up
/// clock (`Vehicle::waiting`, the one clock) has run this much (s) past `GIVE_UP_UNSEEN`,
/// beyond `GAVE_UP_DIST` (m) and not in a queue: kept until nobody saw it, a jam at a
/// junction the player watched never cleared. (The grace is the round-4 rule's three
/// minutes between giving up and going in view, 60 s and 240 s; the in-view removal is
/// never earlier than giving up.) (Moved from omsi-app's traffic.rs.)
pub const IN_VIEW_GRACE: f32 = 180.0;
/// (A distance window SUMO does not have: SUMO teleports the front car of a blocked queue
/// wherever it stands. Left as it is for the user to decide, round 8 H8-9.)
pub const GAVE_UP_DIST: f64 = 40.0;

/// [round5 C4] A lead standing within this (m) makes a car that gave up a queued car, not a
/// phantom against an invisible wall (x10 case15: the two cars in front of the player went at
/// 90 s with their leads 2.2 m ahead; a queue gap is 2-7 m, a car held by nothing has no lead).
pub const QUEUE_LEAD: f32 = 10.0;

/// Why the game took a vehicle off the road (`Record::Remove::rule`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RemoveRule {
    /// The traffic's own: the network's end, no way on (`Simulation::removed`).
    NetworkEnd,
    /// The tile under it was unloaded.
    GroundUnloaded,
    /// Out of range (beyond the spawn radius and the view), whether it sleeps on unseen or
    /// leaves the map.
    OutOfRange,
    /// Out of sight (nobody sees it: out of frame or behind a building).
    OutOfSight,
    /// Given up, in view after its grace ([`goes_in_view`]).
    InViewAfterGrace,
    /// At a dead end or the loaded route's end, far off, long, or with a queue behind it.
    DeadEnd,
    /// [round 9 A2, fix pass 1] At the loaded tiles' edge beyond the perceptual horizon, the
    /// first in line: its trip ends there ([`goes_far`]), in frame, a dot nobody can tell from
    /// nothing. Not counted as removed in view (`ab`'s `removed_far`).
    FarDeadEnd,
    /// A timetable bus taken off as the last resort (no loop or room), or its last trip over.
    BusLastResort,
    /// It parked: the parked object stands in its place.
    Parked,
    /// Anything else the game does (a clock jump, the server, the settings).
    Other,
}

impl RemoveRule {
    /// Whether this removal is the one `ab` gates as "removed in view": a rule of the
    /// population's, not the network's end (a lane end in view), nor a car that parked.
    /// ([round 8 H8-7] the ground going is gated too: it waits behind the app's `hidden()`
    /// like every other population rule.)
    pub fn gated(self) -> bool {
        !matches!(self, RemoveRule::NetworkEnd | RemoveRule::Parked)
    }

    /// [round 9 A2] Whether this removal is made beyond the perceptual horizon by its rule, so
    /// it is not "removed in view" even in frame (`ab` counts it as `removed_far`).
    pub fn beyond_horizon(self) -> bool {
        matches!(self, RemoveRule::FarDeadEnd)
    }
}

/// [round 7 B5] The id of what decision `d` follows: an AI vehicle's (`Decision::lead_id`) or
/// a player's, as `PLAYER_ID_BASE` + its id (the crate names a player lead in `lead_seen`
/// only; `plan::bypass_start` forms the id the same way).
pub fn lead_of(d: &Decision) -> Option<u64> {
    d.lead_id.or(match d.lead_seen {
        Some(Seen::Player(id)) => Some(PLAYER_ID_BASE + id as u64),
        _ => None,
    })
}

/// [round5 C4] Whether a car follows a vehicle within `QUEUE_LEAD`: in a queue, whatever its
/// own hold reads.
/// [round 7 B5] SUMO's rule for the in-view removal: only the car at the front of a blocked
/// queue goes; every car behind a vehicle stays, given up or not, the lead given up or not
/// (each link of the chain is asked of its own lead, so the chain is never cut in its
/// middle). A gridlock with no front car is removed only out of sight.
/// [round 8 H8-9] No speed test: SUMO's waiting time runs only for the vehicle first before
/// the junction (sumo.md 5.3, `MSLane::executeMovements` teleports the lane's front vehicle),
/// so a car with a vehicle lead on its approach is in a queue whether that lead stands or
/// creeps this frame (x10 c08 #1477, a mid-queue car, was taken off the frame its lead #1261
/// crept: the speed was sampled in one frame). ([fix pass 1] Stricter than SUMO, which
/// teleports the first vehicle of each LANE that is not moving (`MSLane::executeMovements`,
/// `firstNotStopped`), so also a car in mid-queue at a lane boundary; OMSI's spline lanes are
/// short - a queue of a few cars spans several of them, a lane boundary mid-queue is the rule -
/// so here a car with a vehicle lead in front of it is in the queue whatever lane that lead is
/// on.)
pub fn in_a_queue(d: &Decision) -> bool {
    lead_of(d).is_some() && d.lead.is_some_and(|l| l.gap < QUEUE_LEAD)
}

/// Whether vehicle `v`, done (`finished`: given up, or a timetable bus let go), goes in view
/// `from_eye` (m) from the eye: its waiting time past `GIVE_UP_UNSEEN + IN_VIEW_GRACE`,
/// beyond `GAVE_UP_DIST`, not in a queue (SUMO's time-to-teleport for the front of a jam,
/// sumo.md 5.3).
pub fn goes_in_view(finished: bool, v: &Vehicle, from_eye: f64) -> bool {
    finished && v.waiting > GIVE_UP_UNSEEN + IN_VIEW_GRACE && from_eye > GAVE_UP_DIST && !in_a_queue(&v.last)
}

/// [round 9 A2, fix pass 1] Beyond the perceptual horizon (the app's `Viewer::horizon`) the
/// edge of the loaded tiles is the trip's end, as the edge of a SUMO network is (a vehicle
/// leaves the simulation where its route ends): a random car that stands there (`at_dead_end`,
/// the crate stops it before the edge) as the first in line - no vehicle lead (`in_a_queue`) -
/// goes at once, in frame too. The queue behind it drives up and each car goes as it arrives, one
/// per arrival, so no row of cars vanishes at once. Inside the horizon this never fires: there
/// the edge stays a dead end (round 8 H8-7) and `hidden()` and `goes_in_view` are the ways off.
pub fn goes_far(at_dead_end: bool, beyond_horizon: bool, v: &Vehicle) -> bool {
    at_dead_end && beyond_horizon && !in_a_queue(&v.last)
}

#[cfg(test)]
mod tests {
    use super::*;
    use omsi_sim::traffic::Lead;

    /// The queue of fix2-jzk600 at t=595: player <- #150 <- #119 <- #163 (given up) <- #101
    /// <- #112, all 2-6 m apart. No car of it may go in view: #150 follows the player, the
    /// rest follow cars, the given-up #163 too and #101 behind it. With the player gone the
    /// front car #150 is the only one that may.
    #[test]
    fn b5_only_the_front_car_of_a_queue_may_go_in_view() {
        let gap = |g: f32| Some(Lead { gap: g, speed: 0.0, acc: 0.0 });
        let follows = |id: Option<u64>, player: Option<u32>| Decision { lead_id: id, lead_seen: player.map(Seen::Player), lead: gap(3.0), ..Default::default() };
        let mut chain = vec![(150u64, follows(None, Some(0))), (119, follows(Some(150), None)), (163, follows(Some(119), None)), (101, follows(Some(163), None)), (112, follows(Some(101), None))];
        assert_eq!(lead_of(&chain[0].1), Some(PLAYER_ID_BASE));
        let removable: Vec<u64> = chain.iter().filter(|(_, d)| !in_a_queue(d)).map(|c| c.0).collect();
        assert!(removable.is_empty(), "{removable:?}");
        chain[0].1 = follows(None, None);
        chain[0].1.lead = None;
        let removable: Vec<u64> = chain.iter().filter(|(_, d)| !in_a_queue(d)).map(|c| c.0).collect();
        assert_eq!(removable, vec![150]);
    }

    /// The no-lead car against an invisible wall still goes; one behind a vehicle within
    /// QUEUE_LEAD stays, one farther off goes.
    #[test]
    fn c4_a_car_behind_a_lead_is_queued_not_a_phantom() {
        let d = |id: Option<u64>, gap: Option<f32>| Decision { lead_id: id, lead: gap.map(|gap| Lead { gap, speed: 0.0, acc: 0.0 }), ..Default::default() };
        assert!(!in_a_queue(&d(None, None)));
        assert!(in_a_queue(&d(Some(7), Some(2.2))));
        assert!(in_a_queue(&d(Some(7), Some(QUEUE_LEAD - 0.5))));
        assert!(!in_a_queue(&d(Some(7), Some(QUEUE_LEAD + 0.5))));
    }

    /// [H8-9] x10 c08 #1477: a mid-queue car whose lead #1261 crept for a frame (`lead 1261 gap
    /// 4.0 standing false`) was taken for the front of the queue and removed. In a queue is a
    /// vehicle lead on its approach, whatever that lead's speed this frame.
    #[test]
    fn h8_9_a_creeping_lead_keeps_the_car_in_its_queue() {
        let creeping = Decision { lead_id: Some(1261), lead: Some(Lead { gap: 4.0, speed: 0.4, acc: 0.3 }), ..Default::default() };
        assert!(in_a_queue(&creeping));
        let mut v = Vehicle { last: creeping, waiting: GIVE_UP_UNSEEN + IN_VIEW_GRACE + 1.0, ..Default::default() };
        assert!(!goes_in_view(true, &v, 111.0));
        v.last = Decision::default();
        assert!(goes_in_view(true, &v, 111.0));
    }

    /// The give-up clock and the distance: before GIVE_UP_UNSEEN + IN_VIEW_GRACE it stays,
    /// after it goes beyond GAVE_UP_DIST, not nearer; not done, it never goes.
    #[test]
    fn r7_in_view_only_after_the_grace_and_beyond_the_distance() {
        let mut v = Vehicle::default();
        v.waiting = GIVE_UP_UNSEEN + IN_VIEW_GRACE - 1.0;
        assert!(!goes_in_view(true, &v, 100.0));
        v.waiting = GIVE_UP_UNSEEN + IN_VIEW_GRACE + 1.0;
        assert!(goes_in_view(true, &v, 100.0));
        assert!(!goes_in_view(true, &v, GAVE_UP_DIST - 1.0));
        assert!(!goes_in_view(false, &v, 100.0));
    }

    /// [round 9 A2, fix pass 1] r8-jzk t=520: #147 stood at the loaded tiles' edge 869 m from
    /// the eye. Beyond the horizon, first in line: its trip ends, whatever its waiting time;
    /// inside the horizon, behind a lead, or not at the edge it stays.
    #[test]
    fn a2_at_the_edge_beyond_the_horizon_the_trip_ends() {
        let mut v = Vehicle::default();
        assert!(goes_far(true, true, &v), "at once, no waiting time");
        assert!(!goes_far(true, false, &v), "inside the horizon");
        assert!(!goes_far(false, true, &v), "not at the edge");
        v.last = Decision { lead_id: Some(9), lead: Some(Lead { gap: 3.0, speed: 0.0, acc: 0.0 }), ..Default::default() };
        assert!(!goes_far(true, true, &v), "behind a lead");
        assert!(RemoveRule::FarDeadEnd.gated() && RemoveRule::FarDeadEnd.beyond_horizon());
        assert!(!RemoveRule::InViewAfterGrace.beyond_horizon() && !RemoveRule::DeadEnd.beyond_horizon());
    }
}
