//! Junction grants: the one place where vehicles decide against each other.
//!
//! Every frame each vehicle that is at or inside a junction files a [`Claim`] for its
//! movement: where it is, how fast it comes, the gap it accepts. [`resolve`] reads the
//! junction's foe table and answers each claim with a [`Grant`]: go, or wait for a named
//! vehicle. The answer is a pure function of the claims, so the stage can be checked in a
//! test with no vehicles at all.

use crate::driver::Impatience;
use crate::junction::{FoeKind, Junctions};
use omsi_sim::traffic::{arrival_time, Network};

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
    /// Gap (s) accepted to a foe that has the right of way, impatience applied.
    pub gap: f32,
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
}

/// Seconds a vehicle inside the junction keeps as its margin before a foe may follow it
/// into the meeting place.
const CLEAR_MARGIN: f32 = 0.6;
/// Impatience at which a driver goes first in front of a foe that stands and waits too.
const HALF_FED_UP: f32 = 0.5;

/// How far away (s) a foe has to be before a fed-up driver takes the way in front of it:
/// the driver's own figure for a foe that stands or crawls, twice that for one rolling,
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

/// Decide every claim. Claims of different junctions do not interact.
///
/// Two passes when somebody is fed up: the first with the rules only, the second lets the
/// fed-up drivers squeeze in, but not in front of a foe the first pass lets go and that
/// stands at its line (it moves off this very frame: equally far from the meeting place,
/// the two would arrive together).
pub fn resolve(js: &Junctions, net: &Network, claims: &[Claim]) -> Vec<Grant> {
    let plain = decide(js, net, claims, None);
    if claims.iter().all(|c| c.impatience.claims_after().is_none() && c.impatience.value < HALF_FED_UP) {
        return plain;
    }
    decide(js, net, claims, Some(&plain))
}

/// One pass over the claims; `plain` is the first pass's answer when fed-up drivers may
/// squeeze in.
fn decide(js: &Junctions, net: &Network, claims: &[Claim], plain: Option<&[Grant]>) -> Vec<Grant> {
    let mut out: Vec<Grant> = Vec::with_capacity(claims.len());
    // a claim stands for the whole way through: the piece it names and the pieces that
    // one continues into, each as far along as its start lies (claim, piece, shift)
    let mut expanded: Vec<(usize, usize, f32)> = Vec::new();
    for (k, c) in claims.iter().enumerate() {
        for &(m, off) in &c.chain {
            expanded.push((k, m, off));
        }
    }
    let _ = net;
    for c in claims {
        let node = &js.nodes[c.junction];
        let my_foes = js.foes_along(c.junction, &c.chain);
        let squeeze = if plain.is_some() { c.impatience.claims_after() } else { None };
        if c.committed || c.absolute {
            out.push(Grant { vehicle: c.vehicle, go: true, wait_for: None, hold: None });
            continue;
        }
        if !c.exit_room && squeeze.is_none() {
            out.push(Grant { vehicle: c.vehicle, go: false, wait_for: None, hold: Some(Hold::ExitFull) });
            continue;
        }
        let mut wait_for: Option<(usize, f32)> = None;
        for foe in &my_foes {
            let their_lane = node.movements[foe.movement].lane;
            let _ = net.lanes.get(their_lane);
            let (my_in, my_out) = window(c, foe.mine.0, foe.mine.1);
            for &(ok, _, shift) in expanded.iter().filter(|&&(k, m, _)| claims[k].junction == c.junction && m == foe.movement && claims[k].vehicle != c.vehicle) {
                // the foe as it stands on this piece of its way
                let mut shifted = claims[ok].clone();
                shifted.s -= shift;
                let o = &shifted;
                // a foe that has left the meeting place
                if o.s - o.rear > foe.theirs.1 {
                    continue;
                }
                let (mut their_in, mut their_out) = window(o, foe.theirs.0, foe.theirs.1);
                let their_inside = o.s + o.front >= foe.theirs.0;
                // a foe standing at its red light is not coming before its light goes:
                // the left-turner on its arrow turns in front of the oncoming queue
                if o.light_red && o.speed < 0.5 && !their_inside {
                    match o.until_go {
                        Some(t) => {
                            their_in += t;
                            their_out += t;
                        }
                        None => continue,
                    }
                }
                // standing in the meeting place, held by its own lead: it is not leaving
                // on the schedule its acceleration would give it
                if their_inside && o.held && o.speed < 0.3 {
                    their_out = f32::INFINITY;
                }
                // (one who has to give way is "first" only by the gap it accepts, not by
                // the margin a vehicle with the right of way keeps: a side-road car that
                // crossed 0.6 s ahead of the main road had the main road braking hard)
                let margin = if foe.yields && !c.absolute { c.gap.max(CLEAR_MARGIN) } else { CLEAR_MARGIN };
                let i_am_first = my_out + margin <= their_in;
                let they_are_first = their_out + CLEAR_MARGIN <= my_in;
                // the two are clear of each other in time whatever the rules say
                if !their_inside && i_am_first && !o.committed {
                    continue;
                }
                if they_are_first && (their_inside || o.committed || foe.yields || o.absolute) {
                    continue;
                }
                let theirs_first = o.absolute
                    || (foe.yields && !c.absolute)
                    || (!foe.yields && !node.movements[foe.movement].foes.iter().any(|f| f.movement == c.movement && f.yields) && their_in < my_in);
                // (two fed-up drivers must not squeeze in on each other: all four at an equal
                // junction became fed up and went at once, into each other. The one with the
                // lower index takes it first, the other waits for it.)
                let goes_now = o.speed < 1.0 && plain.map(|p| p[ok].go).unwrap_or(false);
                // (never into a stream: a fed-up driver squeezes in front of a foe that is
                // slow or still well away, not across one doing town speed 35 m off. A
                // turner facing a green stream waits for it to stop at its red.)
                let far_enough = their_in > after_for(o.speed, squeeze);
                // (four cars arriving together at an unmarked junction all give way to
                // the right: when the foe stands and waits as well, half the patience is
                // enough to go first, as a nod across the junction settles it in life)
                let mutual_wait = plain.is_some() && c.impatience.value >= HALF_FED_UP && o.speed < 0.3 && !their_inside && !o.committed;
                let fed_up = (squeeze.is_some() && far_enough) || mutual_wait;
                let take_anyway = fed_up && !their_inside && !o.committed && !goes_now && !(o.impatience.value >= HALF_FED_UP && o.vehicle < c.vehicle);
                if theirs_first && !take_anyway {
                    // the right of way is theirs: I need my gap after they are gone, or
                    // before they come
                    let gap_ok = (my_out + c.gap <= their_in && !their_inside && !o.committed) || they_are_first;
                    if !gap_ok {
                        if wait_for.map(|w| their_in < w.1).unwrap_or(true) {
                            wait_for = Some((o.vehicle, their_in));
                        }
                    }
                } else if their_inside || o.committed {
                    // mine, but they are already in the way: let them out first
                    if !they_are_first && wait_for.map(|w| their_out < w.1).unwrap_or(true) {
                        wait_for = Some((o.vehicle, their_out));
                    }
                }
                let _ = (foe.kind == FoeKind::Merge, their_out);
            }
        }
        match wait_for {
            Some((v, _)) => out.push(Grant { vehicle: c.vehicle, go: false, wait_for: Some(v), hold: Some(Hold::Foe) }),
            None => out.push(Grant { vehicle: c.vehicle, go: true, wait_for: None, hold: None }),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::{DriverParams, VehicleClass};
    use glam::DVec3;
    use omsi_sim::traffic::{LaneBuilder, LaneKey, LaneKind};

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
        Claim { vehicle, junction: 0, movement, s, speed, accel: 1.8, v_max: 13.0, front: 2.0, rear: 2.0, gap: 4.0, committed: false, exit_room: true, impatience: Impatience::default(), absolute: false, held: false, light_red: false, until_go: None, chain: vec![(movement, 0.0)] }
    }

    #[test]
    fn the_side_road_waits_for_the_main_road_and_goes_when_it_is_clear() {
        let (net, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        // both at their lines, the main road car still 20 m out
        let g = resolve(&js, &net, &[claim(1, main, -20.0, 10.0), claim(2, side, -1.0, 0.0)]);
        assert!(g[0].go);
        assert!(!g[1].go);
        assert_eq!(g[1].wait_for, Some(1));
        // the main road car is 120 m out: the gap (its own, plus the time to clear) is enough to cross
        let g = resolve(&js, &net, &[claim(1, main, -120.0, 10.0), claim(2, side, -1.0, 0.0)]);
        assert!(g[1].go);
        // no main road car at all
        let g = resolve(&js, &net, &[claim(2, side, -1.0, 0.0)]);
        assert!(g[0].go);
    }

    #[test]
    fn right_before_left_between_equals_and_first_come_inside() {
        let (net, js) = four_way(128.0, 128.0);
        let (east, north) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let g = resolve(&js, &net, &[claim(1, east, -1.0, 0.0), claim(2, north, -1.0, 0.0)]);
        // exactly one of the two goes
        assert_ne!(g[0].go, g[1].go);
        // the one already in the junction goes, whoever has the rules on their side
        let yields_first = !g[0].go;
        // (standing in the box, so it will not be gone before the other arrives)
        let inside = if yields_first { claim(1, east, 8.0, 0.0) } else { claim(2, north, 8.0, 0.0) };
        let other = if yields_first { claim(2, north, -1.0, 0.0) } else { claim(1, east, -1.0, 0.0) };
        let mut inside = inside;
        inside.committed = true;
        let g = resolve(&js, &net, &[inside, other]);
        assert!(g[0].go);
        assert!(!g[1].go);
    }

    #[test]
    fn a_full_exit_holds_the_car_until_it_is_fed_up() {
        let (net, js) = four_way(192.0, 64.0);
        let main = js.nodes[0].movement_of(0).unwrap();
        let mut c = claim(1, main, -1.0, 0.0);
        c.exit_room = false;
        let g = resolve(&js, &net, &[c.clone()]);
        assert_eq!(g[0].hold, Some(Hold::ExitFull));
        let p = DriverParams::preset(VehicleClass::Car);
        for _ in 0..1000 {
            c.impatience.update(&p, 0.1, true);
        }
        let g = resolve(&js, &net, &[c]);
        assert!(g[0].go, "an impatient driver squeezes in");
    }

    #[test]
    fn a_train_is_never_held() {
        let (net, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let mut train = claim(9, side, -30.0, 15.0);
        train.absolute = true;
        let g = resolve(&js, &net, &[claim(1, main, -1.0, 0.0), train]);
        assert!(!g[0].go);
        assert!(g[1].go);
    }
}
