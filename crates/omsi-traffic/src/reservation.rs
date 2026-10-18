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
    /// does not wait for its own follower.
    pub lead: Option<usize>,
    /// Counted standing at its last claim: see [`Claim::standing`].
    pub was_standing: bool,
    /// Standing, held there by something other than a junction's answer: its lead, people,
    /// a zebra, a light. It cannot move off before that lets it go,
    /// whatever the junction says.
    pub blocked: bool,
    /// Standing as of its last claim (`was_standing`): creeping to its red
    /// under 1 m/s it is still held there (its claim shifted by `until_go`).
    pub red_hysteresis: bool,
    /// The vehicle's lanes from the one it is on: (lane, distance from its origin to the
    /// lane's start; the first is minus its place along it). Empty: not known (a player off
    /// the lanes). (For a car behind me in my own lane.)
    pub lanes: Vec<(usize, f32)>,
}

impl Claim {
    /// Standing, with hysteresis: under 0.3 m/s, or still under 1 m/s after standing at the
    /// last claim (a car let go that creeps off is not refused again by the ordinary rules
    /// the moment it passes 0.3 m/s, to be let go again once it has stopped).
    pub fn standing(&self) -> bool {
        self.speed < 0.3 || (self.was_standing && self.speed <= 1.0)
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
}

/// Seconds a vehicle inside the junction keeps as its margin before a foe may follow it
/// into the meeting place.
const CLEAR_MARGIN: f32 = 0.6;
/// Impatience at which a driver goes first in front of a foe that stands and waits too.
const HALF_FED_UP: f32 = 0.5;
/// Seconds a driver held by its lead, people or a light takes to move
/// off once let go: a foe held so is waited for only when it could be in the meeting place,
/// from rest after this, before the claimant is out of it.
const START_DELAY: f32 = 1.5;

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
/// A foe counts only if it can compete: one that follows the claimant, or is behind it on
/// a piece both ways take, is no foe in either pass. Two passes when somebody is fed up or
/// the first refused anybody: the second lets the fed-up drivers squeeze in, but not in
/// front of a foe the first pass lets go and that stands at its line (it moves off this
/// very frame: equally far from the meeting place, the two would arrive together); and a
/// claimant standing refused at its line no longer waits for a foe that stands outside the
/// meeting place and is itself refused (it is not coming). Standing refused claimants that
/// are each other's foes are put in an order first (`standing_order`), so that of two who
/// would drop each other only one goes.
pub fn resolve(js: &Junctions, net: &Network, claims: &[Claim]) -> Vec<Grant> {
    let plain = decide(js, net, claims, None, None);
    let fed_up = !claims.iter().all(|c| c.impatience.claims_after().is_none() && c.impatience.value < HALF_FED_UP);
    if !fed_up && plain.iter().all(|g| g.go) {
        return plain;
    }
    let standing = standing_order(js, claims, &plain);
    decide(js, net, claims, Some(&plain), Some(&standing))
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
/// a fraction of a metre that merges back, are foes of each other by the table. Behind
/// means its front at least a metre behind my rear (its whole length and a metre more
/// behind my rear): two cars level at a merge, centimetres apart, are not one behind the
/// other, and both went.
fn behind_on_my_way(c: &Claim, o: &Claim) -> bool {
    c.chain.iter().any(|&(piece, my_off)| {
        let mine = c.s - my_off;
        (mine + c.front >= 0.0 || o.speed < 0.3) && o.chain.iter().any(|&(their_piece, their_off)| their_piece == piece && o.s - their_off + o.front <= mine - c.rear - 1.0)
    })
}

/// `o` is behind `c` in `c`'s own lane: on the lane `c` is on, or on
/// a lane of its way that leads into it, its front at least a metre behind `c`'s rear. It
/// can only come after me, whatever its speed and wherever its way goes on (case03: cars
/// rolling up behind the head of their own queue, "committed" to the merge of the lead-in
/// with its own stub, held the head at green).
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
    let my_out = arrival_time(-my_rear, c.speed, c.accel, c.v_max);
    let their_at = arrival_time(-their_front, o.speed, o.accel, o.v_max);
    let margin = if foe.yields && !c.absolute { c.gap.max(CLEAR_MARGIN) } else { CLEAR_MARGIN };
    their_at >= my_out + margin
}

/// One pass over the claims; `plain` is the first pass's answer in the second pass. Both
/// drop followers and those behind on the same way; `standing` (the second pass) drops
/// standing refused foes, the contenders decided in their order.
fn decide(js: &Junctions, net: &Network, claims: &[Claim], plain: Option<&[Grant]>, standing: Option<&[Standing]>) -> Vec<Grant> {
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
    // claim of each vehicle, for following a chain of leads
    let claim_of: hashbrown::HashMap<usize, usize> = claims.iter().enumerate().map(|(k, c)| (c.vehicle, k)).collect();
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
                // "committed" is to the foe's first meeting place: one
                // standing outside this one is not on its way into it whatever comes, it
                // comes as its window from standing says (a car standing in a queue on a
                // lead-in whose first meeting place, with its own stub or another way, lay
                // behind its bumper held the claimant ahead of it until it "left")
                if shifted.speed < 0.3 && !their_inside {
                    shifted.committed = false;
                }
                let o = &shifted;
                // behind me in my own lane, or merging in behind me: it
                // comes after me at any speed, committed or not
                if behind_in_my_lane(c, &claims[ok], foe.theirs.0 - o.s) || (!their_inside && merging_behind(c, o, foe)) {
                    continue;
                }
                let (mut their_in, mut their_out) = window(o, foe.theirs.0, foe.theirs.1);
                // my own follower, or one behind me on a piece both our
                // ways take, comes after me whatever the table says; one already in the
                // meeting place or committed to it is never dropped
                // (the follower only while it stands: a lead is one frame's perception and
                // flickers between two cars side by side into a merge; one rolling at 3 m/s
                // 2 m from the meeting place was dropped on it, and the two touched)
                if !their_inside && !o.committed && (behind_on_my_way(c, &claims[ok]) || (o.speed < 0.3 && (claims[ok].lead == Some(c.vehicle) || held_by(claims, &claim_of, plain, ok, c.vehicle)))) {
                    continue;
                }
                // a foe standing at its red light is not coming before its light goes:
                // the left-turner on its arrow turns in front of the oncoming queue
                // (creeping up to its line under 1 m/s after a stand counts as
                // standing there, `Claim::red_hysteresis`)
                if o.light_red && (o.speed < 0.5 || (o.red_hysteresis && o.speed <= 1.0)) && !their_inside {
                    match o.until_go {
                        Some(t) => {
                            their_in += t;
                            their_out += t;
                        }
                        None => continue,
                    }
                }
                // a foe standing outside the meeting place, held there by
                // something other than a junction (its lead, people, a zebra, a light), is not
                // coming: not waited for when I am out of the meeting place before it could be
                // in it from rest after its start (`START_DELAY`), moving or not. (Not when I am
                // held so myself: my window is no more real than its, and two such would let
                // each other go; I am not going anywhere before my own lead lets me.)
                if o.blocked && !c.blocked && !c.light_red && o.standing() && !their_inside && !o.committed && my_out + CLEAR_MARGIN <= their_in + START_DELAY {
                    continue;
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
                // one held by its lead, people or a light outside the
                // meeting place is not through it before the other comes on the strength of a
                // start from rest now: it starts when what holds it lets it. Not "first" myself
                // when I am held so, nor the foe when it is (a side road car went behind one
                // standing a metre from the meeting place, once the other main road car it
                // waited for was dropped; a car held by its lead, "first" by a start it could
                // not make, went past the contender before it in the order)
                let i_am_first = my_out + margin <= their_in && !c.blocked;
                let they_are_first = their_out + CLEAR_MARGIN <= my_in && !(o.blocked && !their_inside);
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
        Claim { vehicle, junction: 0, movement, s, speed, accel: 1.8, v_max: 13.0, front: 2.0, rear: 2.0, gap: 4.0, committed: false, exit_room: true, impatience: Impatience::default(), absolute: false, held: false, light_red: false, until_go: None, chain: vec![(movement, 0.0)], lead: None, was_standing: false, blocked: false, red_hysteresis: false, lanes: Vec::new() }
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
        let g = resolve(&js, &net, &claims);
        assert!(grant_of(&g, 1).go, "{:?}", g);
        // ... unless the follower is already in the meeting place
        let mut inside = claims.clone();
        inside[1].s = -1.5;
        let g = resolve(&js, &net, &inside);
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
        let g = resolve(&js, &net, &claims);
        assert!(grant_of(&g, 1).go, "{:?}", g);
        assert_eq!(grant_of(&g, 2).wait_for, Some(1), "B still comes in behind A");
        // the one on B ahead along the common way: it is a foe as before
        let mut ahead = claims.clone();
        ahead[0].s = 4.0;
        ahead[1].s = 4.0;
        let g = resolve(&js, &net, &ahead);
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
        let g = resolve(&js, &net, &claims);
        assert!(grant_of(&g, 1).go, "{:?}", g);
        assert!(!grant_of(&g, 2).go);
        // the same car rolling (it is coming) still counts
        let mut rolling = claims.clone();
        rolling[1].speed = 3.0;
        rolling[1].held = false;
        let g = resolve(&js, &net, &rolling);
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
        let g = resolve(&js, &net, &claims);
        assert!(grant_of(&g, 1).go, "{g:?}");
        // in the meeting place it is committed to it
        let mut inside = claims.clone();
        inside[1].s = 7.0;
        let g = resolve(&js, &net, &inside);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2));
    }

    #[test]
    fn competitor_filter_a_standing_car_further_back_on_a_shared_piece_is_behind_before_my_front_reaches_it() {
        // A (0) and B (1) merge where C (2) begins; A gives way to B. Both stand short of the
        // object, the one for B further back along the common way.
        let (net, js) = object(&[(&[(-10.0, 0.0), (0.0, 0.0)], 0), (&[(-8.0, -6.0), (0.0, 0.0)], 0), (&[(0.0, 0.0), (10.0, 0.0)], 0)]);
        let mut on_a = claim(1, 0, -3.0, 0.0);
        on_a.chain = vec![(0, 0.0), (2, 10.0)];
        let mut on_b = claim(2, 1, -9.0, 0.0);
        on_b.chain = vec![(1, 0.0), (2, 10.0)];
        let claims = [on_a, on_b];
        let g = resolve(&js, &net, &claims);
        assert!(grant_of(&g, 1).go, "{g:?}");
        // rolling, it may be first at the merge: it counts
        let mut rolling = claims.clone();
        rolling[1].speed = 6.0;
        let g = resolve(&js, &net, &rolling);
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
        let g = resolve(&js, &net, &claims);
        assert!(grant_of(&g, 1).go, "{g:?}");
        // the second car's lead is somebody else: it may come, it counts
        let mut other = claims.clone();
        other[2].lead = Some(99);
        let g = resolve(&js, &net, &other);
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
        let g = resolve(&js, &net, &claims);
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
        let g = decide(&js, &net, &claims, Some(&plain), Some(&st));
        assert!(g[0].go, "{g:?}");
        assert_eq!(g[1].wait_for, Some(5));
        // the one before in the order stays refused (held by something else): the other
        // does not wait for it
        let mut exit_full = claims.clone();
        exit_full[0].exit_room = false;
        let plain = [Grant { vehicle: 5, go: false, wait_for: None, hold: Some(Hold::ExitFull) }, refused(3)];
        let st = standing_order(&js, &exit_full, &plain);
        assert_eq!(st[0], Standing::Hard);
        let g = decide(&js, &net, &exit_full, Some(&plain), Some(&st));
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

    // ---- review2 counter-examples (adversarial review of the competitor filter and signals) ----

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
                                (Some((ai, ao)), Some((bi, bo))) => ai < bo + CLEAR_MARGIN && bi < ao + CLEAR_MARGIN,
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
        let (net, js) = four_way(128.0, 128.0);
        let (east, north) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        for (ve, vn) in [(1usize, 2usize), (2, 1)] {
            let claims = [claim(ve, east, -1.0, 0.0), claim(vn, north, -1.0, 0.0)];
            let refused = |v| Grant { vehicle: v, go: false, wait_for: None, hold: Some(Hold::Foe) };
            let plain = [refused(ve), refused(vn)];
            let st = standing_order(&js, &claims, &plain);
            let g = decide(&js, &net, &claims, Some(&plain), Some(&st));
            assert_eq!(g.iter().filter(|g| g.go).count(), 1, "{ve}/{vn}: {g:?}");
            let goer = g.iter().find(|g| g.go).unwrap().vehicle;
            assert!(g.iter().filter(|g| !g.go).all(|g| g.wait_for == Some(goer)), "{g:?}");
        }
    }

    #[test]
    fn review_three_ring_permuted_indices_and_one_member_far_back() {
        let (net, js) = triangle();
        for ids in [[30usize, 10, 20], [20, 30, 10], [10, 20, 30]] {
            for far in [None, Some(0usize), Some(1), Some(2)] {
                let claims: Vec<Claim> = (0..3)
                    .map(|m| {
                        let s = if far == Some(m) { -12.0 } else { -1.0 };
                        let mut c = claim(ids[m], m, s, 0.0);
                        c.gap = 8.0;
                        c
                    })
                    .collect();
                let g = resolve(&js, &net, &claims);
                assert!(g.iter().any(|g| g.go), "{ids:?} far {far:?}: nobody goes {g:?}");
                assert!(review_both_released(&js, &claims, &g).is_empty(), "{ids:?} far {far:?}: {g:?}");
            }
        }
    }

    #[test]
    fn review_four_ring_equal_priority_no_crossing_pair_released() {
        let (net, js) = review_cross();
        let yields: usize = (0..4).map(|m| js.nodes[0].movements[m].foes.iter().filter(|f| f.yields).count()).sum();
        let claims: Vec<Claim> = (0..4).map(|m| claim(10 + m, m, -1.0, 0.0)).collect();
        let g = resolve(&js, &net, &claims);
        eprintln!("4-ring: yields in table {yields}; {g:?}");
        assert!(g.iter().any(|g| g.go), "{g:?}");
        assert!(review_both_released(&js, &claims, &g).is_empty(), "{g:?}");
        for p in [[3usize, 1, 0, 2], [1, 0, 3, 2], [2, 3, 1, 0]] {
            let claims: Vec<Claim> = (0..4).map(|m| claim(10 + p[m], m, -1.0, 0.0)).collect();
            let g = resolve(&js, &net, &claims);
            assert!(g.iter().any(|g| g.go), "{p:?} {g:?}");
            assert!(review_both_released(&js, &claims, &g).is_empty(), "{p:?} {g:?}");
        }
    }

    #[test]
    fn review_standing_refused_foe_5m_out_is_still_first_once_it_moves() {
        // triangle: me gives way to the foe, the foe gives way to Z (rolling in fast), Z to me
        let (net, js) = triangle();
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
        let g = resolve(&js, &net, &claims);
        eprintln!("frame 1 {g:?}");
        assert!(!grant_of(&g, 2).go, "the foe is held by Z: {g:?}");
        // frame 2: Z is gone; I have rolled off at 0.5 m/s. The foe must be first.
        let mut me2 = me.clone();
        me2.s += 0.3;
        me2.speed = 0.5;
        let g = resolve(&js, &net, &[me2.clone(), foe.clone()]);
        assert!(grant_of(&g, 2).go, "{g:?}");
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "frame 2: I wait for the foe once it goes: {g:?}");
        // ... and standing still at 40 m myself: same
        let g = resolve(&js, &net, &[me.clone(), foe.clone()]);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
    }

    #[test]
    fn review_foe_at_red_with_1s_to_go_counts_as_coming_in_1s() {
        let (net, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let me = claim(1, side, 4.0, 0.0);
        let mut red = claim(2, main, 1.0, 0.0);
        red.light_red = true;
        red.until_go = Some(1.0);
        let g = resolve(&js, &net, &[me.clone(), red.clone()]);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
        // the same foe refused for a full exit (Hard): the second pass skips it outright,
        // whatever its light, though its exit may clear within the second
        let mut hard = red.clone();
        hard.exit_room = false;
        let g = resolve(&js, &net, &[me.clone(), hard.clone()]);
        eprintln!("exit-full foe at red, 1 s to go: {g:?}");
    }

    #[test]
    fn review_moving_foe_10ms_40m_out_counts() {
        let (net, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let mine0 = js.nodes[0].movements[main].foes.iter().find(|f| f.movement == side).unwrap().mine.0;
        let me = claim(1, side, 4.0, 0.0);
        let mover = claim(2, main, mine0 - 40.0 - 2.0, 10.0);
        let g = resolve(&js, &net, &[me.clone(), mover.clone()]);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
    }

    #[test]
    fn review_lead_is_me_drops_a_rolling_foe_2m_from_the_meeting_place() {
        // the main road car (right of way) rolls at 3 m/s, its front 2 m from the meeting
        // place, and names me (the side road car) as its lead
        let (net, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let mine0 = js.nodes[0].movements[main].foes.iter().find(|f| f.movement == side).unwrap().mine.0;
        let me = claim(1, side, 4.0, 0.0);
        let mut foe = claim(2, main, mine0 - 2.0 - 2.0, 3.0);
        foe.lead = Some(1);
        let g = resolve(&js, &net, &[me.clone(), foe.clone()]);
        // (was: dropped on the strength of `lead` alone, both went) a rolling follower is
        // a foe: exactly one goes
        assert_eq!(g.iter().filter(|g| g.go).count(), 1, "{g:?}");
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
        assert!(review_both_released(&js, &[me, foe], &g).is_empty());
    }

    #[test]
    fn review_merge_both_standing_6cm_apart_both_released() {
        // A (0) and B (1) merge where C (2) begins; A gives way to B. Both stand short of
        // the merge, B 6 cm further back along the common way (cf. case10 #30596/#30816)
        let (net, js) = object(&[(&[(-10.0, 0.0), (0.0, 0.0)], 0), (&[(-8.0, -6.0), (0.0, 0.0)], 0), (&[(0.0, 0.0), (10.0, 0.0)], 0)]);
        let mut on_a = claim(1, 0, -3.0, 0.0);
        on_a.chain = vec![(0, 0.0), (2, 10.0)];
        let mut on_b = claim(2, 1, -3.06, 0.0);
        on_b.chain = vec![(1, 0.0), (2, 10.0)];
        let claims = [on_a, on_b];
        let g = resolve(&js, &net, &claims);
        eprintln!("merge 6 cm: {g:?}");
        // (was: both went, 6 cm apart, into one merge) level cars are not one behind the other
        assert_eq!(g.iter().filter(|g| g.go).count(), 1, "exactly one goes: {g:?}");
        assert!(review_both_released(&js, &claims, &g).is_empty(), "{g:?}");
    }

    // ---- round 3: the competitor filter and signals widened ----

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
        // (case10: a bus held after green by a queue member 61 m away)
        let (net, js, claims) = held_main_road_foe(30.0, 3.0, true);
        let g = resolve(&js, &net, &claims);
        assert!(grant_of(&g, 1).go, "{g:?}");
        // ... and standing at the line myself, not refused
        let mut standing = claims.clone();
        standing[0].speed = 0.0;
        assert!(grant_of(&resolve(&js, &net, &standing), 1).go);
        // held by my own lead as well: not my call yet
        let mut both = claims.clone();
        both[0].speed = 0.0;
        both[0].held = true;
        both[0].blocked = true;
        assert_eq!(grant_of(&resolve(&js, &net, &both), 1).wait_for, Some(2));
    }

    #[test]
    fn competitor_filter_a_held_foe_that_could_be_there_before_i_am_out_still_counts() {
        // 1 m out: from rest after its start it is in the meeting place before I am out
        let (net, js, claims) = held_main_road_foe(1.0, 3.0, true);
        let g = resolve(&js, &net, &claims);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
        // refused by the junction rather than held (it moves off as soon as it is let go):
        // the order of the second pass decides, not the timing
        let (net, js, claims) = held_main_road_foe(30.0, 3.0, false);
        let g = resolve(&js, &net, &claims);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
        // rolling (not standing): it is coming
        let (net, js, mut claims) = held_main_road_foe(30.0, 3.0, true);
        claims[1].speed = 6.0;
        claims[1].held = false;
        claims[1].blocked = false;
        let g = resolve(&js, &net, &claims);
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
        // the merge of the lead-in with its stub, case03), its front 8 m behind my rear
        let mut behind = claim(2, 0, -11.0, 6.0);
        behind.committed = true;
        behind.lanes = vec![(10, -20.0), (99, 10.0)];
        let claims = [turner, behind];
        let g = resolve(&js, &net, &claims);
        assert!(grant_of(&g, 1).go, "{g:?}");
        // on a lane before mine that leads into it: behind me too
        let mut back = claims.clone();
        back[1].lanes = vec![(11, -20.0), (10, 2.0)];
        back[1].s = -33.0;
        assert!(grant_of(&resolve(&js, &net, &back), 1).go);
        // ahead of me in my lane (it is the one in front): it counts
        let mut ahead = claims.clone();
        ahead[1].lanes = vec![(10, -34.0)];
        ahead[1].committed = false;
        ahead[1].s = 3.0;
        ahead[1].speed = 0.0;
        let g = resolve(&js, &net, &ahead);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
    }

    #[test]
    fn competitor_filter_a_car_merging_in_behind_me_is_no_foe_before_my_front_reaches_the_shared_piece() {
        // A (0) and B (1) merge into the exit; B (from the left) gives way to A. No piece of
        // the object is on both ways (the exit lies outside it): the merge says who is behind.
        let (net, js) = object(&[(&[(-10.0, 0.0), (0.0, 0.0)], 0), (&[(-8.0, 6.0), (0.0, 0.0)], 0)]);
        let f = js.nodes[0].movements[0].foes[0];
        assert!(!f.yields && f.kind == FoeKind::Merge, "{f:?}");
        let on_a = claim(1, 0, 7.0, 3.0);
        // B rolls up behind at 5.8 m/s, its front 14 m from the merge's end, too close to
        // stop before its meeting place (committed: the old rules let it out first)
        let mut on_b = claim(2, 1, -6.0, 5.8);
        on_b.committed = true;
        let claims = [on_a, on_b];
        let g = resolve(&js, &net, &claims);
        assert!(grant_of(&g, 1).go, "{g:?}");
        // fast, it would be at the merge's end before my rear is past: it counts
        let mut fast = claims.clone();
        fast[1].speed = 13.0;
        let g = resolve(&js, &net, &fast);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
        // level with me: not behind
        let mut level = claims.clone();
        level[1].s = 5.0;
        level[1].speed = 3.0;
        assert_eq!(grant_of(&resolve(&js, &net, &level), 1).wait_for, Some(2));
        // (review4) giving way, I keep my accepted gap, not the 0.6 s margin: B with the
        // right of way 14 m from the merge's end at 5.8 m/s is waited for
        let (net, js) = object(&[(&[(-10.0, 0.0), (0.0, 0.0)], 0), (&[(-8.0, -6.0), (0.0, 0.0)], 0)]);
        assert!(js.nodes[0].movements[0].foes[0].yields);
        let claims = [claim(1, 0, 7.0, 3.0), claim(2, 1, -6.0, 5.8)];
        assert_eq!(grant_of(&resolve(&js, &net, &claims), 1).wait_for, Some(2));
    }

    #[test]
    fn signal_foes_a_foe_creeping_to_its_red_after_a_stand_is_not_coming() {
        let (net, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let mine0 = js.nodes[0].movements[main].foes.iter().find(|f| f.movement == side).unwrap().mine.0;
        let me = claim(1, side, 2.0, 0.0);
        // the main road car creeps up to its red at 0.8 m/s, 6 m out, 30 s to go
        let mut red = claim(2, main, mine0 - 6.0 - 2.0, 0.8);
        red.light_red = true;
        red.until_go = Some(30.0);
        red.was_standing = true;
        // (without the hysteresis on the claim it is rolling, and coming)
        let g = resolve(&js, &net, &[me.clone(), red.clone()]);
        assert_eq!(grant_of(&g, 1).wait_for, Some(2), "{g:?}");
        let mut on = red.clone();
        on.red_hysteresis = true;
        let g = resolve(&js, &net, &[me.clone(), on]);
        assert!(grant_of(&g, 1).go, "{g:?}");
        // rolling at 0.8 m/s without having stood: coming
        let mut rolling = red.clone();
        rolling.was_standing = false;
        rolling.red_hysteresis = false;
        assert_eq!(grant_of(&resolve(&js, &net, &[me, rolling]), 1).wait_for, Some(2));
    }

    /// Pairs both let go into one meeting place whose windows overlap, a claimant held by
    /// its lead (`blocked`) counted from rest after its start (`START_DELAY`): it does not
    /// move before that, whatever its grant says.
    fn round3_both_released(js: &Junctions, claims: &[Claim], g: &[Grant]) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let win = |c: &Claim, from: f32, to: f32, shift: f32| -> Option<(f32, f32)> {
            let mut x = c.clone();
            x.s -= shift;
            let (mut i, mut o) = window(&x, from, to);
            if c.blocked {
                i += START_DELAY;
                o += START_DELAY;
            }
            if c.light_red && (c.speed < 0.5 || (c.red_hysteresis && c.speed <= 1.0)) {
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
                                (Some((ai, ao)), Some((bi, bo))) => ai < bo + CLEAR_MARGIN && bi < ao + CLEAR_MARGIN,
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
        // one meeting place that the competitor filter before round 3 did not let go
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
            let (net, js) = &objects[(rnd() * objects.len() as f64) as usize % objects.len()];
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
                    c.red_hysteresis = c.was_standing;
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
            // (the rules before round 3: the same claims without what they read)
            let before: Vec<Claim> = claims.iter().map(|c| Claim { blocked: false, red_hysteresis: false, ..c.clone() }).collect();
            let old = resolve(js, net, &before);
            let g = resolve(js, net, &claims);
            dropped += old.iter().zip(&g).filter(|(o, n)| !o.go && n.go).count();
            let po = round3_both_released(js, &claims, &old);
            // (a claimant let go while held by its lead or standing at its red does not move
            // on its grant: it claims again, moving, the frame it can; such a pair is counted
            // apart)
            let idx = |v: usize| claims.iter().position(|c| c.vehicle == v).unwrap();
            let can_move = |k: usize| {
                let c = &claims[k];
                !(c.blocked || (c.light_red && (c.speed < 0.5 || (c.red_hysteresis && c.speed <= 1.0))))
            };
            for p in round3_both_released(js, &claims, &g) {
                if !po.contains(&p) {
                    let (a, b) = (idx(p.0), idx(p.1));
                    let moved = [a, b].iter().any(|&k| !old[k].go && g[k].go && can_move(k));
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

    // ---- review4: counter-examples to round 3 ----

    /// (C2) `merging_behind` replaces the yielding driver's accepted gap (`Claim::gap`, 4 s
    /// here) by `CLEAR_MARGIN` (0.6 s): a side car standing at the merge goes in front of a
    /// main road car at 14 m/s whose front is 40.5 m from the merge end. At the moment the
    /// side car's rear clears the merge end the main car is 8.4 m behind it, closing at
    /// ~10 m/s: it must brake ~4-6 m/s² (with/without a 2 m standstill gap) for a car that
    /// had to give way to it.
    #[test]
    fn review4_merging_behind_bypasses_the_accepted_gap_at_14_ms() {
        let (net, js) = object(&[(&[(-10.0, 0.0), (0.0, 0.0)], 0), (&[(-8.0, -6.0), (0.0, 0.0)], 0)]);
        let f = js.nodes[0].movements[0].foes[0];
        assert!(f.yields && f.kind == FoeKind::Merge, "A gives way to B at the merge");
        let side = claim(1, 0, 7.5, 0.0); // front 0.5 m before the merge end, rear 4.5 m
        let mut main = claim(2, 1, f.theirs.1 - 40.5 - 2.0, 14.0);
        main.v_max = 14.0;
        let my_out = arrival_time(4.5, 0.0, 1.8, 13.0);
        let their_at = arrival_time(40.5, 14.0, 1.8, 14.0);
        let claims = [side, main];
        let g = resolve(&js, &net, &claims);
        let gap_at_clear = 14.0 * (their_at - my_out);
        let v_side = 1.8 * my_out;
        let need = (14.0 - v_side).powi(2) / (2.0 * gap_at_clear) - 1.8;
        let need_2m = (14.0 - v_side).powi(2) / (2.0 * (gap_at_clear - 2.0)) - 1.8;
        assert!(
            !grant_of(&g, 1).go,
            "side car let go {:.2} s before a 14 m/s main car (accepted gap {:.1} s): main car {:.1} m behind at clear, closing {:.1} m/s, must brake {:.1} m/s² ({:.1} with a 2 m gap)",
            their_at - my_out,
            claims[0].gap,
            gap_at_clear,
            14.0 - v_side,
            need,
            need_2m
        );
    }

    /// (C1) `behind_in_my_lane` takes the first lane of the foe's way that is mine and does
    /// not ask whether the foe passes the meeting place before it gets there: a main road
    /// car (right of way) 15 m out at 10 m/s whose 150 m way comes back round into my
    /// lane (a block loop, a turning circle) is "behind me in my lane".
    #[test]
    fn review4_behind_in_my_lane_ignores_where_the_foe_reaches_my_lane() {
        let (net, js) = four_way(192.0, 64.0);
        let (main, side) = (js.nodes[0].movement_of(0).unwrap(), js.nodes[0].movement_of(1).unwrap());
        let mine0 = js.nodes[0].movements[main].foes.iter().find(|f| f.movement == side).unwrap().mine.0;
        let mut me = claim(1, side, -1.0, 0.0);
        me.lanes = vec![(1, 1.0)];
        let mut foe = claim(2, main, mine0 - 15.0 - 2.0, 10.0);
        foe.lanes = vec![(0, -foe.s), (50, 20.0), (51, 60.0), (1, 120.0)];
        let claims = [me, foe];
        let g = resolve(&js, &net, &claims);
        assert!(!grant_of(&g, 1).go, "the side car goes in front of a 10 m/s main road car 15 m out: {g:?}");
    }
}
