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
    /// The vehicle this one follows this frame (the caller's index), if any: a claimant
    /// does not wait for its own follower (`competitor_filter`).
    pub lead: Option<usize>,
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
    resolve_with(js, net, claims, false)
}

/// [`resolve`], with the `competitor_filter` switch (see `Switches`): a foe counts only
/// if it can compete. Off, this is `resolve` exactly.
///
/// On, a foe that follows the claimant, or is behind it on a piece both ways take, is no
/// foe in either pass; and the second pass runs whenever the first refused anybody: a
/// claimant standing refused at its line no longer waits for a foe that stands outside
/// the meeting place and is itself refused (it is not coming). Standing refused claimants
/// that are each other's foes are put in an order first (`standing_order`), so that of two
/// who would drop each other only one goes.
pub fn resolve_with(js: &Junctions, net: &Network, claims: &[Claim], competitor_filter: bool) -> Vec<Grant> {
    let plain = decide(js, net, claims, None, competitor_filter, None);
    let fed_up = !claims.iter().all(|c| c.impatience.claims_after().is_none() && c.impatience.value < HALF_FED_UP);
    if competitor_filter {
        if !fed_up && plain.iter().all(|g| g.go) {
            return plain;
        }
        let standing = standing_order(js, claims, &plain);
        return decide(js, net, claims, Some(&plain), true, Some(&standing));
    }
    if !fed_up {
        return plain;
    }
    decide(js, net, claims, Some(&plain), false, None)
}

/// A claim as the second pass of `competitor_filter` sees it.
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

/// The players' claims (`usize::MAX - k`, see `Simulation::tick`): their grants are not
/// read (the player drives), so a refused player is no proof of anything.
fn is_player(vehicle: usize) -> bool {
    vehicle >= usize::MAX - 64
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
        if plain[k].go || c.speed >= 0.3 || c.committed || c.absolute || is_player(c.vehicle) {
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
    for _ in 0..4 {
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

/// `o` is behind `c` on its own way: some piece of the junction is on both ways, `c`'s
/// front bumper has reached it (the piece starts at or before `c`'s current place), and
/// `o`'s origin lies further back along that piece than `c`'s. Positions along a piece
/// `p` of a chain are `s - offset(p)`: a claim's `s` is along the first piece of its chain
/// and each chain entry `(piece, offset)` says how far along the way that piece starts.
/// (Two ways that merge into one piece meet there; a car still on its merging piece
/// behind one already into the common piece comes in behind it.) A foe that stands is
/// behind me on a common piece as soon as it is further back along it than I am, before my
/// front reaches it: two cars in one queue before the object, whose ways part on a piece of
/// a fraction of a metre that merges back, are foes of each other by the table.
fn behind_on_my_way(c: &Claim, o: &Claim) -> bool {
    c.chain.iter().any(|&(piece, my_off)| {
        let mine = c.s - my_off;
        (mine + c.front >= 0.0 || o.speed < 0.3) && o.chain.iter().any(|&(their_piece, their_off)| their_piece == piece && o.s - their_off < mine)
    })
}

/// One pass over the claims; `plain` is the first pass's answer when fed-up drivers may
/// squeeze in. `filter` (the `competitor_filter` switch) drops followers and those behind
/// on the same way; `standing` (its second pass) drops standing refused foes, the
/// contenders decided in their order.
fn decide(js: &Junctions, net: &Network, claims: &[Claim], plain: Option<&[Grant]>, filter: bool, standing: Option<&[Standing]>) -> Vec<Grant> {
    let mut out: Vec<Option<Grant>> = vec![None; claims.len()];
    // (the contenders first, in their order: each sees whether the ones before it go)
    let mut order: Vec<usize> = (0..claims.len()).collect();
    if let Some(st) = standing {
        order.sort_by_key(|&k| match st[k] {
            Standing::Rank(r) => r,
            _ => usize::MAX,
        });
    }
    // a claim stands for the whole way through: the piece it names and the pieces that
    // one continues into, each as far along as its start lies (claim, piece, shift)
    let mut expanded: Vec<(usize, usize, f32)> = Vec::new();
    for (k, c) in claims.iter().enumerate() {
        for &(m, off) in &c.chain {
            expanded.push((k, m, off));
        }
    }
    let _ = net;
    // (competitor_filter) claim of each vehicle, for following a chain of leads
    let claim_of: hashbrown::HashMap<usize, usize> = if filter { claims.iter().enumerate().map(|(k, c)| (c.vehicle, k)).collect() } else { hashbrown::HashMap::new() };
    for &k in &order {
        let c = &claims[k];
        let node = &js.nodes[c.junction];
        let my_foes = js.foes_along(c.junction, &c.chain);
        let squeeze = if plain.is_some() { c.impatience.claims_after() } else { None };
        if c.committed || c.absolute {
            out[k] = Some(Grant { vehicle: c.vehicle, go: true, wait_for: None, hold: None });
            continue;
        }
        if !c.exit_room && squeeze.is_none() {
            out[k] = Some(Grant { vehicle: c.vehicle, go: false, wait_for: None, hold: Some(Hold::ExitFull) });
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
                // a foe that has left the meeting place
                if shifted.s - shifted.rear > foe.theirs.1 {
                    continue;
                }
                let their_inside = shifted.s + shifted.front >= foe.theirs.0;
                // (competitor_filter) "committed" is to the foe's first meeting place: one
                // standing outside this one is not on its way into it whatever comes, it
                // comes as its window from standing says (a car standing in a queue on a
                // lead-in whose first meeting place, with its own stub or another way, lay
                // behind its bumper held the claimant ahead of it until it "left")
                if filter && shifted.speed < 0.3 && !their_inside {
                    shifted.committed = false;
                }
                let o = &shifted;
                let (mut their_in, mut their_out) = window(o, foe.theirs.0, foe.theirs.1);
                // (competitor_filter) my own follower, or one behind me on a piece both our
                // ways take, comes after me whatever the table says; one already in the
                // meeting place or committed to it is never dropped
                if filter && !their_inside && !o.committed && (claims[ok].lead == Some(c.vehicle) || behind_on_my_way(c, &claims[ok]) || (o.speed < 0.3 && held_by(claims, &claim_of, plain, ok, c.vehicle))) {
                    continue;
                }
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
                let goes_now = o.speed < 1.0 && (plain.map(|p| p[ok].go).unwrap_or(false) || (standing.is_some() && out[ok].map(|g| g.go).unwrap_or(false)));
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
                // (competitor_filter, second pass) I stand refused at my line; a foe that
                // stands outside the meeting place and is refused itself is not coming: one
                // refused for a full exit stays refused, a contender after me in the order
                // waits for me if I go, one before me that stays refused does not go.
                // Only a contender before me that goes this pass holds me.
                if let Some(Standing::Rank(mine)) = standing.map(|st| st[k]) {
                    if o.speed < 0.3 && !their_inside && !o.committed {
                        match standing.map(|st| st[ok]) {
                            Some(Standing::Hard) => continue,
                            Some(Standing::Rank(theirs)) => {
                                let me_inside = c.s + c.front >= foe.mine.0;
                                let ahead_goes = theirs < mine && out[ok].map(|g| g.go).unwrap_or(false);
                                if me_inside || !ahead_goes {
                                    continue;
                                }
                                if !take_anyway && wait_for.map(|w| their_in < w.1).unwrap_or(true) {
                                    wait_for = Some((o.vehicle, their_in));
                                }
                                continue;
                            }
                            _ => {}
                        }
                    }
                }
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
        out[k] = Some(match wait_for {
            Some((v, _)) => Grant { vehicle: c.vehicle, go: false, wait_for: Some(v), hold: Some(Hold::Foe) },
            None => Grant { vehicle: c.vehicle, go: true, wait_for: None, hold: None },
        });
    }
    out.into_iter().map(|g| g.expect("every claim decided")).collect()
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
        Claim { vehicle, junction: 0, movement, s, speed, accel: 1.8, v_max: 13.0, front: 2.0, rear: 2.0, gap: 4.0, committed: false, exit_room: true, impatience: Impatience::default(), absolute: false, held: false, light_red: false, until_go: None, chain: vec![(movement, 0.0)], lead: None }
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

    fn grant_of(g: &[Grant], vehicle: usize) -> Grant {
        *g.iter().find(|g| g.vehicle == vehicle).unwrap()
    }

    #[test]
    fn competitor_filter_my_follower_in_the_same_lane_is_no_foe() {
        // one lane forks inside the object: the straight way (0) and the left turn (1),
        // which gives way to it where the two overlap at the start
        let (net, js) = object(&[(&[(-10.0, 0.0), (10.0, 0.0)], 0), (&[(-10.0, 0.0), (-4.0, 0.0), (0.0, 4.0), (0.0, 10.0)], 1)]);
        assert!(js.nodes[0].movements[1].foes[0].yields);
        // the turner stands with its nose over the line, the car behind it (going
        // straight on, following the turner) 2 m behind its rear
        let turner = claim(1, 1, -1.0, 0.0);
        let mut follower = claim(2, 0, -7.0, 0.0);
        follower.lead = Some(1);
        let claims = [turner, follower];
        // the old rules: the turner waits for its own follower, which waits for the turner
        let g = resolve_with(&js, &net, &claims, false);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2));
        assert!(!grant_of(&g, 1).go);
        // filtered: the follower is no foe, the turner goes
        let g = resolve_with(&js, &net, &claims, true);
        assert!(grant_of(&g, 1).go, "{:?}", g);
        // ... unless the follower is already in the meeting place
        let mut inside = claims.clone();
        inside[1].s = -1.5;
        let g = resolve_with(&js, &net, &inside, true);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2));
    }

    #[test]
    fn competitor_filter_a_car_behind_on_a_merging_piece_is_no_foe() {
        // A (0) and B (1) merge where C (2) begins; A gives way to B
        let (net, js) = object(&[(&[(-10.0, 0.0), (0.0, 0.0)], 0), (&[(-8.0, -6.0), (0.0, 0.0)], 0), (&[(0.0, 0.0), (10.0, 0.0)], 0)]);
        assert!(js.nodes[0].movements[0].foes[0].yields);
        // the car on A has its nose on C already; the one on B stands 9 m back along the
        // common way, short of the meeting place (no lead recorded: the way decides)
        let mut on_a = claim(1, 0, 8.5, 0.0);
        on_a.chain = vec![(0, 0.0), (2, 10.0)];
        let mut on_b = claim(2, 1, 1.0, 0.0);
        on_b.chain = vec![(1, 0.0), (2, 10.0)];
        let claims = [on_a, on_b];
        let g = resolve_with(&js, &net, &claims, false);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "the old rules: A waits for the car behind it");
        assert_eq!(grant_of(&g, 2).wait_for, Some(1));
        let g = resolve_with(&js, &net, &claims, true);
        assert!(grant_of(&g, 1).go, "{:?}", g);
        assert_eq!(grant_of(&g, 2).wait_for, Some(1), "B still comes in behind A");
        // the one on B ahead along the common way: it is a foe as before
        let mut ahead = claims.clone();
        ahead[0].s = 4.0;
        ahead[1].s = 4.0;
        let g = resolve_with(&js, &net, &ahead, true);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2));
    }

    #[test]
    fn competitor_filter_a_standing_refused_car_60_m_away_does_not_hold_a_car_at_the_junction() {
        let (net, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let mut at_line = claim(1, side, -1.0, 0.0);
        at_line.gap = 6.0;
        // standing in a queue 60 m out on the main road, refused itself (no room past)
        let mut far = claim(2, main, -60.0, 0.0);
        far.exit_room = false;
        far.held = true;
        let claims = [at_line, far];
        let g = resolve_with(&js, &net, &claims, false);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "the old rules wait for it");
        let g = resolve_with(&js, &net, &claims, true);
        assert!(grant_of(&g, 1).go, "{:?}", g);
        assert!(!grant_of(&g, 2).go);
        // the same car rolling (it is coming) still counts
        let mut rolling = claims.clone();
        rolling[1].speed = 3.0;
        rolling[1].held = false;
        let g = resolve_with(&js, &net, &rolling, true);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2));
    }

    #[test]
    fn competitor_filter_a_committed_foe_standing_outside_this_meeting_place_is_not_on_its_way_into_it() {
        let (net, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        // the side road car stands 15 m short of the crossing, "committed" (its first
        // meeting place, with its own stub, lies behind its bumper), held by its lead
        let mut far = claim(2, side, -15.0, 0.0);
        far.committed = true;
        far.held = true;
        let claims = [claim(1, main, -1.0, 0.0), far];
        let g = resolve_with(&js, &net, &claims, false);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "the old rules wait for it to leave");
        let g = resolve_with(&js, &net, &claims, true);
        assert!(grant_of(&g, 1).go, "{g:?}");
        // in the meeting place it is committed to it
        let mut inside = claims.clone();
        inside[1].s = 7.0;
        let g = resolve_with(&js, &net, &inside, true);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2));
    }

    #[test]
    fn competitor_filter_a_standing_car_further_back_on_a_shared_piece_is_behind_before_my_front_reaches_it() {
        // A (0) and B (1) merge where C (2) begins; A gives way to B. Both stand short of the
        // object, the one for B further back along the common way.
        let (net, js) = object(&[(&[(-10.0, 0.0), (0.0, 0.0)], 0), (&[(-8.0, -6.0), (0.0, 0.0)], 0), (&[(0.0, 0.0), (10.0, 0.0)], 0)]);
        let mut on_a = claim(1, 0, -3.0, 0.0);
        on_a.chain = vec![(0, 0.0), (2, 10.0)];
        let mut on_b = claim(2, 1, -5.0, 0.0);
        on_b.chain = vec![(1, 0.0), (2, 10.0)];
        let claims = [on_a, on_b];
        let g = resolve_with(&js, &net, &claims, false);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "the old rules: A waits for it");
        let g = resolve_with(&js, &net, &claims, true);
        assert!(grant_of(&g, 1).go, "{g:?}");
        // rolling, it may be first at the merge: it counts
        let mut rolling = claims.clone();
        rolling[1].speed = 6.0;
        let g = resolve_with(&js, &net, &rolling, true);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2));
    }

    #[test]
    fn competitor_filter_a_foe_whose_lead_waits_for_me_is_no_foe() {
        // the side road car (1) stands at its line with its nose over the main road; the
        // main road car right at it (7) follows its body, the one behind that (2) follows 7
        let (net, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let mut first = claim(7, main, -3.0, 0.0);
        first.lead = Some(1);
        let mut second = claim(2, main, -10.0, 0.0);
        second.lead = Some(7);
        let claims = [claim(1, side, -1.0, 0.0), first, second];
        let g = resolve_with(&js, &net, &claims, false);
        assert!(!grant_of(&g, 1).go, "the old rules hold the side road car: {g:?}");
        let g = resolve_with(&js, &net, &claims, true);
        assert!(grant_of(&g, 1).go, "{g:?}");
        // the second car's lead is somebody else: it may come, it counts
        let mut other = claims.clone();
        other[2].lead = Some(99);
        let g = resolve_with(&js, &net, &other, true);
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
        let (net, js) = triangle();
        for m in 0..3 {
            let f = &js.nodes[0].movements[m].foes;
            assert_eq!(f.len(), 2, "movement {m}: {f:?}");
            assert_eq!(f.iter().filter(|f| f.yields).count(), 1, "movement {m} gives way to exactly one: {f:?}");
        }
        let claims: Vec<Claim> = (0..3)
            .map(|m| {
                let mut c = claim(10 + m, m, -1.0, 0.0);
                c.gap = 8.0;
                c
            })
            .collect();
        // the old rules: each waits for the one it gives way to, round the ring
        let g = resolve_with(&js, &net, &claims, false);
        assert!(g.iter().all(|g| !g.go), "{:?}", g);
        let g = resolve_with(&js, &net, &claims, true);
        let going: Vec<usize> = g.iter().filter(|g| g.go).map(|g| g.vehicle).collect();
        assert_eq!(going, vec![10], "exactly one goes, the lowest index of the ring: {:?}", g);
        assert!(g.iter().filter(|g| !g.go).all(|g| g.wait_for == Some(10)), "the others wait for it: {:?}", g);
    }

    #[test]
    fn competitor_filter_of_two_standing_refused_claimants_the_one_with_the_right_of_way_goes() {
        // both stand refused at their lines (for whatever reason) and would drop each other
        let (net, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let refused = |v| Grant { vehicle: v, go: false, wait_for: None, hold: Some(Hold::Foe) };
        // the main road car has the higher index: the table decides, not the index
        let claims = [claim(5, main, -1.0, 0.0), claim(3, side, -1.0, 0.0)];
        let plain = [refused(5), refused(3)];
        let st = standing_order(&js, &claims, &plain);
        assert!(matches!((st[0], st[1]), (Standing::Rank(a), Standing::Rank(b)) if a < b), "{st:?}");
        let g = decide(&js, &net, &claims, Some(&plain), true, Some(&st));
        assert!(g[0].go, "{g:?}");
        assert_eq!(g[1].wait_for, Some(5));
        // the one before in the order stays refused (held by something else): the other
        // does not wait for it
        let mut exit_full = claims.clone();
        exit_full[0].exit_room = false;
        let plain = [Grant { vehicle: 5, go: false, wait_for: None, hold: Some(Hold::ExitFull) }, refused(3)];
        let st = standing_order(&js, &exit_full, &plain);
        assert_eq!(st[0], Standing::Hard);
        let g = decide(&js, &net, &exit_full, Some(&plain), true, Some(&st));
        assert!(!g[0].go);
        assert!(g[1].go, "{g:?}");
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
