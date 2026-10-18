//! The decision: from what the driver sees to one plan for this frame. A plan is the
//! lead to follow, a stop point, a lateral target and at most one manoeuvre to start.
//! No plan reads another vehicle's plan; the only cross-vehicle input is the junction
//! grant.

use crate::driver::VehicleClass;
use crate::perception::{JunctionView, NeighbourView, Perception, Seen, SignalView};
use crate::reservation::{Grant, Hold};
use crate::sim::{Passing, Simulation, Vehicle};
use omsi_sim::traffic::{arrival_time, Aspect, Lead, Network};

/// Why the car holds back, the nearest reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum Why {
    #[default]
    None,
    Lead,
    Light,
    Junction,
    ExitFull,
    People,
    Zebra,
    Service,
    Merge,
    PullOut,
    LetOut,
    End,
}

impl Why {
    pub fn name(self) -> &'static str {
        match self {
            Why::None => "",
            Why::Lead => "lead",
            Why::Light => "light",
            Why::Junction => "yield",
            Why::ExitFull => "exit",
            Why::People => "people",
            Why::Zebra => "zebra",
            Why::Service => "service",
            Why::Merge => "merge",
            Why::PullOut => "pull_out",
            Why::LetOut => "let_out",
            Why::End => "end",
        }
    }
}

/// A manoeuvre to begin this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Start {
    Change { to: usize, dir: i32 },
    Bypass { to: usize, dir: i32 },
    RouteChange { to: usize, dir: i32 },
    Pass(Passing),
    AbortPass { hold: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Plan {
    pub lead: Option<Lead>,
    pub lead_seen: Option<Seen>,
    /// Distance from the origin to where the car stops.
    pub stop: Option<f32>,
    pub why: Why,
    pub lateral: Option<f32>,
    pub start: Option<Start>,
    pub accel_cap: Option<f32>,
    /// Held by something (the start reaction applies when it clears).
    pub held: bool,
    pub hold: Option<Hold>,
    /// The claim index of the vehicle waited for at a junction (a player: `usize::MAX - k`).
    pub wait_for: Option<usize>,
    /// Id of the vehicle whose body across the way holds this one.
    pub body_block: Option<u64>,
    /// At the line on green or yellow: the driver goes through whatever it turns to.
    pub amber: Option<(usize, usize)>,
    /// With `amber`: the gap from the front to that line (m).
    pub amber_line: Option<f32>,
}

fn min_stop(stop: &mut Option<f32>, why: &mut Why, d: Option<f32>, w: Why) {
    if let Some(d) = d {
        if stop.map(|s| d < s).unwrap_or(true) {
            *stop = Some(d);
            *why = w;
        }
    }
}

/// Is the gap in the lane beside accepted: the follower there must not have to brake
/// hard, and there is room to settle ahead.
fn gap_accepted(me: &Vehicle, nb: &NeighbourView) -> bool {
    let (st, dp) = (&me.state, &me.params);
    let v = st.speed;
    let ok_behind = nb.behind.map(|(g, vb)| g > 2.0 + me.impatience.gap(dp.lc_gap) * vb + (vb - v).max(0.0).powi(2) / (2.0 * dp.comfort_brake)).unwrap_or(true);
    let ok_ahead = nb.ahead.map(|(g, va)| g > 2.0 + (v - va).max(0.0) * 1.5).unwrap_or(true);
    ok_behind && ok_ahead && !nb.foreign
}

/// The lane-change desire towards a neighbour, 0..1, and whether the gap there is
/// accepted (LMRS: the route, a standing obstacle, speed gain, keeping to the side).
/// `near_junction`: within the last stretch before a junction, where the lines are solid
/// and only the turn lane and a standing obstacle are reasons to move over.
fn change_desire(net: &Network, sim: &Simulation, me: &Vehicle, p: &Perception, nb: &NeighbourView, near_junction: bool) -> (f32, bool) {
    let st = &me.state;
    let dp = &me.params;
    let lane = &net.lanes[st.lane];
    if !nb.allowed {
        return (0.0, false);
    }
    // the solid line (`[rule] overtaking_prohib`) on either lane, and the pieces of a
    // junction object: no change of the driver's own
    if lane.no_overtaking || net.lanes[nb.lane].no_overtaking || lane.source == 2 {
        return (0.0, false);
    }
    let mut desire = 0.0f32;
    // a turn lane is never left within 150 m of its junction, and a turn ahead is taken
    // from the lane that leads into it
    let my_turn = st.turn_ahead(net, 150.0).map(|t| t.0).unwrap_or(0);
    if my_turn != 0 && nb.turn != my_turn {
        return (0.0, false);
    }
    if nb.turn != 0 && my_turn == 0 && nb.runs_on < 150.0 {
        return (0.0, false);
    }
    if lane.turn == 0 && my_turn != 0 && nb.turn == my_turn && lane.turn != my_turn {
        desire = desire.max(1.0 * dp.lc_strategic);
    }
    // behind something standing: round it
    if let Some(l) = p.leader {
        let standing = l.standing_obstacle() && l.gap < 25.0;
        if standing && me.stopped >= 4.0 && nb.ahead.map(|a| a.0 > l.gap + l.length + 6.0).unwrap_or(true) {
            desire = desire.max(0.8);
        }
        // a faster lane
        let slow = l.speed < 0.7 * p.v_want && l.gap < 45.0 && l.gap > 8.0 && l.speed < st.speed + 1.0;
        if slow && !near_junction && st.speed >= 4.0 && nb.ahead.map(|a| a.0 > 50.0).unwrap_or(true) {
            desire = desire.max(dp.lc_speed_gain);
        }
    }
    // keep to the side when the road is clear
    let keep_side = if sim.left_hand { 1 } else { 2 };
    if nb.dir == keep_side && !near_junction && p.leader.map(|l| l.gap > 60.0).unwrap_or(true) && nb.ahead.map(|a| a.0 > 70.0).unwrap_or(true) && nb.runs_on > 200.0 {
        desire = desire.max(dp.lc_keep_right);
    }
    if nb.runs_on < 40.0 {
        return (0.0, false);
    }
    (desire, gap_accepted(me, nb))
}

/// Metres a pass needs past the obstacle besides the car's own length and the way back in
/// (`obstacle_run`).
pub const PASS_ROOM: f32 = 4.0;
/// A pass goes round at most this many metres of things standing one behind the other.
pub const PASS_MAX_OBSTACLE: f32 = 30.0;
/// Seconds a pass may stand out there with its way back in blocked before it is given up.
pub const PASS_STRANDED: f32 = 8.0;

/// The obstacle a pass would go round: the standing `lead` and what stands (or crawls under
/// 2 m/s) less than `need` beyond it, one after the other, at most `PASS_MAX_OBSTACLE` long;
/// and whether there is `need` of free lane beyond that (`Perception::ahead`). (Its length, room.)
fn obstacle_run(p: &Perception, lead: &crate::perception::Headway, need: f32) -> (f32, bool) {
    let mut end = lead.gap + lead.length;
    for &(near, far, speed) in &p.ahead {
        if near <= lead.gap + 0.5 || far <= end {
            continue;
        }
        if speed >= 2.0 {
            continue;
        }
        if near - end >= need {
            break;
        }
        if far - lead.gap > PASS_MAX_OBSTACLE {
            return (end - lead.gap, false);
        }
        end = far;
    }
    // (and the road goes on that far)
    (end - lead.gap, p.road_left > end + need + 2.0)
}

/// The vehicle `id` passed (a player's: `PLAYER_ID_BASE` + its id): (a bus, indicating out
/// towards the traffic, moving). None when it is gone.
fn obstacle_bus(sim: &Simulation, vehicles: &[Vehicle], id: u64) -> Option<(bool, bool, bool)> {
    let out = sim.out_side();
    if id >= crate::sim::PLAYER_ID_BASE {
        let p = sim.players.iter().find(|p| p.id as u64 == id - crate::sim::PLAYER_ID_BASE)?;
        return Some((true, p.blinker as i32 == out, p.speed.abs() > 0.5));
    }
    let o = vehicles.iter().find(|o| o.id == id)?;
    Some((o.is_bus(), o.blinker_shown.0 == out, o.state.speed > 0.5))
}

/// Within this many metres of a junction (from the front to where its object begins) a car
/// does not pull out to pass, nor keeps back to be able to.
pub const JUNCTION_KEEP: f32 = 40.0;

/// Could the car get round the standing `lead` into the lane beside, `nb` (`Start::Bypass`,
/// the lane-change rules of `change_desire` but for having stood a while): a lane it may take,
/// no solid line, not a junction's piece, a turn ahead that stays its own, room there past
/// the thing standing and a gap it accepts.
fn could_bypass(net: &Network, me: &Vehicle, p: &Perception, nb: &NeighbourView, lead: &crate::perception::Headway) -> bool {
    let lane = &net.lanes[me.state.lane];
    if !nb.allowed || lane.no_overtaking || net.lanes[nb.lane].no_overtaking || lane.source == 2 || nb.runs_on < 40.0 {
        return false;
    }
    let my_turn = me.state.turn_ahead(net, 150.0).map(|t| t.0).unwrap_or(0);
    if (my_turn != 0 && nb.turn != my_turn) || (nb.turn != 0 && my_turn == 0 && nb.runs_on < 150.0) {
        return false;
    }
    let _ = p;
    nb.ahead.map(|a| a.0 > lead.gap + lead.length + 6.0).unwrap_or(true) && gap_accepted(me, nb)
}

/// What the driver does about the light ahead this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightCall {
    /// Over the line now (no stop for the light).
    pub go: bool,
    /// The light's mark is set (`Vehicle::amber`): the driver took the line on green or yellow.
    pub mark: bool,
    /// Holding the mark close to the line: it goes through as soon as nothing else holds
    /// it (a queue it stands in), whatever the light shows by then.
    pub marked_through: bool,
}

/// The driver's call at the light: through on green, the dilemma zone on yellow, on red
/// only with the mark close to the line or too late to stop.
pub fn light_call(me: &Vehicle, p: &Perception, sig: &SignalView) -> LightCall {
    let st = &me.state;
    let dp = &me.params;
    let v = st.speed;
    let gap = sig.dist - st.front;
    let comfortable = v * v / (2.0 * dp.comfort_brake * 1.4) + 1.0;
    let possible = v * v / (2.0 * omsi_sim::traffic::MAX_BRAKE * 0.8);
    let queued = v < 1.0 && p.leader.map(|l| l.gap < 8.0).unwrap_or(false);
    let marked = me.amber == Some((sig.ctl, sig.light));
    match sig.aspect {
        Aspect::Green | Aspect::GreenYellow | Aspect::Dark => {
            // at the line on green: through, whatever it turns to in a moment
            LightCall { go: true, mark: gap < 3.0 && !queued && v > 1.0, marked_through: marked && gap < comfortable }
        }
        Aspect::Yellow => {
            // the dilemma zone: through when stopping is no longer comfortable; a
            // driver who took the line on green just before goes a little further
            // out ("had the mark" alone let a car two places back in a queue go).
            // One already braking for the light stays on the brake: slowing, it
            // came within "comfortable" of the line again and set off across on red.
            let stopping = me.last.why == Why::Light && me.last.stop.is_some();
            let go = !queued && !stopping && (gap < comfortable || (marked && gap < comfortable + 6.0));
            LightCall { go, mark: go, marked_through: marked && gap < comfortable + 6.0 }
        }
        // (a car standing with its bumper over the line waits like one behind it: at a
        // standstill nothing is "too late to stop")
        Aspect::Red | Aspect::RedYellow => {
            let go = (marked && gap < comfortable && !queued) || (v > 1.0 && gap < possible - 0.5);
            LightCall { go, mark: false, marked_through: marked && gap < comfortable }
        }
    }
}

/// Where a car refused for a foe stops (rule D): its front a metre before the first stretch
/// it waits for (`waited`, along the way through), pulled back before any other meeting
/// place ahead or pedestrian crossing its body would stand in ("keep the junction clear"; a
/// meeting place it is in already it cannot leave but forward). A distance from the origin.
fn hold_point(net: &Network, sim: &Simulation, me: &Vehicle, j: &JunctionView, waited: f32) -> f32 {
    let front = me.state.front;
    let len = me.length();
    // the stretches where it may not stand, as (from, to) distances of its front's stop
    // and of the body: a meeting place a metre before it on, a crossing 2.5 m before it
    // (the zebra stop) to 2 m past it
    let mut bands: Vec<(f32, f32)> = sim.junctions.foes_along(j.junction, &j.chain).iter().map(|f| (j.entry + f.mine.0 - 1.0, j.entry + f.mine.1)).collect();
    let node = &sim.junctions.nodes[j.junction];
    for &(m, off) in &j.chain {
        for &(_, sa, _) in net.walks.get(node.movements[m].lane).map(|v| v.as_slice()).unwrap_or(&[]) {
            let point = j.entry + off + sa;
            bands.push((point - 2.5, point + 2.0));
        }
    }
    bands.extend(converge_bands(sim, j));
    let mut stop = j.entry + waited - 1.0;
    for _ in 0..32 {
        let hit = bands.iter().filter(|b| b.0 > front - 0.3 && b.0 < stop && b.1 > stop - len).map(|b| b.0).fold(f32::INFINITY, f32::min);
        if !hit.is_finite() {
            break;
        }
        stop = hit;
    }
    stop
}

/// The stretches of the way through where it converges with a lane of the junction that is no
/// foe (`Movement::converge`), as (from a metre before, to) distances from the origin.
fn converge_bands(sim: &Simulation, j: &JunctionView) -> Vec<(f32, f32)> {
    let node = &sim.junctions.nodes[j.junction];
    j.chain.iter().flat_map(|&(m, off)| node.movements[m].converge.iter().map(move |&(a, b)| (j.entry + off + a - 1.0, j.entry + off + b))).collect()
}

/// Decide for vehicle `i`.
pub fn plan(net: &Network, sim: &Simulation, vehicles: &[Vehicle], i: usize, p: &Perception, grant: Option<Grant>) -> Plan {
    let me = &vehicles[i];
    let st = &me.state;
    let dp = &me.params;
    let mut out = Plan::default();
    let Some(lane) = net.lanes.get(st.lane) else { return out };
    let v = st.speed;

    // 1. the lead
    if let Some(l) = p.leader {
        out.lead = Some(l.lead());
        out.lead_seen = Some(l.what);
        out.body_block = p.body_block;
    }

    // 2. lights (not the one whose line it passed while it clears the box: in it when that
    // light turned, it drives on; a later line, of that controller or not, holds it as any)
    if let Some(sig) = p.signal.filter(|s| !p.junction.as_ref().is_some_and(|j| j.clears_past(s, st.front))) {
        let call = light_call(me, p, &sig);
        if call.mark {
            out.amber = Some((sig.ctl, sig.light));
            out.amber_line = Some(sig.dist - st.front);
        }
        if !call.go {
            min_stop(&mut out.stop, &mut out.why, Some(sig.dist), Why::Light);
        }
    }

    // 3. the junction
    if let (Some(j), Some(g)) = (p.junction.as_ref(), grant) {
        if !g.go {
            // at the line, or, already on the lead-in piece of a long object, short of
            // the first meeting place of the way through
            // (a line too near to stop at gently any more is no stop either: the car
            // creeps on to the meeting place and stops short of that, as the claim counts
            // it; a line the bumper is already over drew an emergency stop for a frame)
            let at_line = j.entry - 0.3;
            let gently = v * v / (2.0 * (at_line - st.front - 0.6).max(0.05)) <= dp.comfort_brake * 1.5;
            let first = if !j.inside && (gently || v < 0.5) { at_line } else { j.conflict_start - 1.0 };
            // refused for a foe: just before the first stretch it waits for, not short of the
            // first meeting place of its way (viz3 case03: a car waiting for a stretch 22 m
            // on stood 2.8 m in, on the fork shared with the turn pockets, and blocked the
            // turners behind it)
            // (not where its way converges with another lane short of that stretch, that the
            // junction knows no foe on: there it stops at the line, or short of the first
            // meeting place, as before; review9: refused cars stood beside, and rolled into, the
            // very car they waited for, on 5229/5231, 1430/1179, 10814/10625)
            let at = match (g.hold, g.at) {
                (Some(Hold::Foe), Some(waited)) => {
                    let at = hold_point(net, sim, me, j, waited);
                    let converges = converge_bands(sim, j).iter().any(|b| b.0 < j.entry + waited && b.1 > -st.rear);
                    if (at < first || converges) && first > st.front - 0.3 {
                        first
                    } else if converges {
                        // (past the line already: as soon as it can stop gently)
                        at.min(st.front + v * v / (2.0 * dp.comfort_brake) + 0.3)
                    } else {
                        at
                    }
                }
                _ => first,
            };
            if let (Some((gap, speed)), Some(w)) = (g.follow, g.wait_for.filter(|&w| w < vehicles.len())) {
                // second at a zipper merge, both rolling: in behind the other at my gap
                out.hold = g.hold;
                out.wait_for = g.wait_for;
                let behind = Lead { gap, speed, acc: 0.0 };
                if out.lead.map(|l| gap < l.gap).unwrap_or(true) {
                    out.lead = Some(behind);
                    out.lead_seen = Some(Seen::Vehicle(w));
                }
            } else if at > st.front - 0.3 {
                out.hold = g.hold;
                out.wait_for = g.wait_for;
                let w = if g.hold == Some(Hold::ExitFull) { Why::ExitFull } else { Why::Junction };
                min_stop(&mut out.stop, &mut out.why, Some(at), w);
            }
        }
    }

    // (eased off in a mutual hold: it stays where it stands until the other is through)
    if let (Some(_), Some(g)) = (me.ease_for, grant) {
        if !g.go {
            out.hold = g.hold;
            out.wait_for = g.wait_for;
            min_stop(&mut out.stop, &mut out.why, Some(st.front + 0.1), Why::Junction);
        }
    }

    // 4. people
    min_stop(&mut out.stop, &mut out.why, p.people_stop, Why::People);
    min_stop(&mut out.stop, &mut out.why, p.zebra_stop, Why::Zebra);

    // 5. the app's own holds
    min_stop(&mut out.stop, &mut out.why, me.external.stop_at, Why::Service);
    if me.pull_out > 0.0 {
        min_stop(&mut out.stop, &mut out.why, Some(st.front + 0.1), Why::PullOut);
    }
    if let Some(d) = p.letting_out {
        min_stop(&mut out.stop, &mut out.why, Some(st.front + (d - 1.0).max(0.0)), Why::LetOut);
    }

    // 6. a route's lane change: now if the gap is there, else wait at the lane's end (a
    //    route may lead into a lane the path's rules close to the vehicle's class: only
    //    the gap counts)
    if let Some((to, dir)) = p.route_change {
        let nb = if dir == 1 { p.left } else { p.right };
        let ok = nb.map(|n| gap_accepted(me, &n)).unwrap_or(true);
        if ok && st.change.is_none() && st.change_cooldown <= 0.0 {
            out.start = Some(Start::RouteChange { to, dir });
        } else if st.change.is_none() {
            min_stop(&mut out.stop, &mut out.why, Some(lane.length() - 1.0), Why::Merge);
        }
    }

    // 7. a lane change of the driver's own: round something standing, for speed, to keep
    //    to the side, into the lane of the turn ahead
    let free_to_change = st.change.is_none() && st.route.is_empty() && st.change_cooldown <= 0.0 && me.passing.is_none() && out.start.is_none() && !me.is_rail() && lane.length() - st.s > 9.0;
    if free_to_change {
        let near_junction = p.junction.as_ref().map(|j| !j.inside && j.entry < 40.0).unwrap_or(false);
        let mut best: Option<(f32, NeighbourView)> = None;
        for nb in [p.left, p.right].into_iter().flatten() {
            let (d, ok) = change_desire(net, sim, me, p, &nb, near_junction);
            if ok && d > 0.0 && best.map(|b| d > b.0).unwrap_or(true) {
                best = Some((d, nb));
            }
        }
        let threshold = 0.4 - 0.2 * me.impatience.value;
        if let Some((_, nb)) = best.filter(|b| b.0 >= threshold) {
            let standing = p.leader.map(|l| l.standing_obstacle() && l.gap < 25.0).unwrap_or(false);
            out.start = Some(if standing && v < 1.0 { Start::Bypass { to: nb.lane, dir: nb.dir } } else { Start::Change { to: nb.lane, dir: nb.dir } });
        }
    }

    // 8. passing on the oncoming lane, or squeezing past inside the lane
    let out_side = sim.out_side();
    if let Some(pass) = me.passing {
        // out there: come back when the oncoming traffic is nearer than the time needed
        if !pass.aborted && pass.lane.is_some() {
            let remaining = pass.until + pass.back_len - st.odometer;
            if let Some(o) = p.oncoming {
                let v_cap = (p.v_want).clamp(4.0, 14.0);
                let t_need = arrival_time(remaining.max(0.0), v, st.accel, v_cap) + 0.5;
                let shallow = st.lateral.abs() < pass.side.abs() - me.half_width - 1.45;
                if o.clear_for < t_need && shallow && st.odometer + v * v / 7.0 + 0.4 < pass.block {
                    out.start = Some(Start::AbortPass { hold: pass.block - 2.0 });
                }
            }
        }
        // passing a bus that indicates out of its stop or moves off: back in behind it at the
        // gap the driver keeps, while it still can (it pulls out in front of the car; the
        // player's bus is let out of its stop so, `letting_out`)
        if !pass.aborted && out.start.is_none() {
            if let Some((true, signals, moves)) = pass.obstacle.and_then(|ob| obstacle_bus(sim, vehicles, ob)) {
                let shallow = pass.wait > 0.0 || st.lateral.abs() < pass.side.abs() - me.half_width - 1.45;
                if (signals || moves) && shallow && st.odometer + v * v / 7.0 + 0.4 < pass.block {
                    out.start = Some(Start::AbortPass { hold: pass.block + st.front + 0.6 - st.min_gap });
                }
            }
        }
        // standing out there with the way back in blocked: given up, back in behind the obstacle
        // at the gap kept (where it is still behind it), else into the lane where it is, and the
        // obstacle is a lead again (viz case09 #30816 stood 110 s on the oncoming side)
        // (past it, on its way back in, it is stranded too while it stands still out there, held
        // by a red light or a queue: review12, #20946 stood out on a scheduled bus's lane 84 s)
        let held_out = st.lateral.abs() > 1.0;
        if !pass.aborted && out.start.is_none() && pass.wait <= 0.0 && me.stopped > PASS_STRANDED && (st.odometer < pass.until || held_out) {
            // (beside it or past it already: it stays where it is and steps back in as soon as
            // its body there touches nobody, `Simulation::sidestep`)
            let hold = if st.odometer + 0.5 < pass.block { pass.block + st.front + 0.6 - st.min_gap } else { st.odometer };
            out.start = Some(Start::AbortPass { hold });
        }
        // (given up, it holds where it said, until what it went round has left)
        let gone = pass.obstacle.is_some_and(|ob| obstacle_bus(sim, vehicles, ob).is_none_or(|(_, _, moves)| moves));
        if pass.aborted && !gone {
            min_stop(&mut out.stop, &mut out.why, Some((pass.hold - st.odometer).max(0.0)), Why::Lead);
        } else if pass.aborted {
            // (its obstacle gone: nothing holds it here, the pass is over, `Simulation::act`)
        } else if pass.wait > 0.0 {
            // signalling: still behind it, with the room to pull out
            min_stop(&mut out.stop, &mut out.why, Some(st.front + (pass.block - st.odometer - pass.out_len - 1.0).max(0.0)), Why::Lead);
        } else {
            if st.odometer < pass.block + 2.0 {
                // edging out: gently
                out.accel_cap = Some(omsi_sim::ai_motion::PULL_OUT_ACCEL);
            }
            // the thing being passed no longer holds the car (the bent way clears it)
            if let Some(l) = p.leader {
                // (not one standing on the line the pass takes: that holds it)
                let passing_it = l.standing_obstacle() && st.odometer + l.gap + st.front < pass.until + 1.0 && l.lateral.abs() < (l.lateral - pass.side).abs();
                if passing_it {
                    out.lead = None;
                    out.lead_seen = None;
                }
            }
        }
    } else if let Some(l) = p.leader.filter(|l| l.standing_obstacle()) {
        // (not across a solid line: a pass there stays a squeeze within the lane)
        let can_pass = !me.is_rail() && me.class != VehicleClass::Air && st.change.is_none() && out.start.is_none() && lane.left.is_none() && lane.right.is_none();
        let solid = lane.no_overtaking;
        let player = matches!(l.what, Seen::Player(_));
        let player_out = player && sim.players.iter().any(|pl| pl.blinker as i32 == out_side && pl.signalling < 1.0);
        // the thing standing, and a bus of it that indicates out of its stop (not passed: it
        // is let out, Sobol's rule)
        let obstacle_id = match l.what {
            Seen::Vehicle(j) | Seen::Body(j) => vehicles.get(j).map(|o| o.id),
            Seen::Player(id) => Some(crate::sim::PLAYER_ID_BASE + id as u64),
            Seen::Parked => None,
        };
        let bus_out = obstacle_id.and_then(|ob| obstacle_bus(sim, vehicles, ob)).is_some_and(|(bus, signals, _)| bus && signals);
        // a bus does not pass what stands at its own next stop
        let own_stop = me.external.stop_at.map(|s| s < l.gap + l.length + st.front + 15.0).unwrap_or(false);
        // how far the car has to swing out: past the thing inside the lane if the lane
        // leaves room beside where it stands, else onto the oncoming lane
        let need = me.half_width + l.half_width + 0.35;
        let towards_middle = sim.oncoming_sign();
        let squeeze_side = if l.lateral * towards_middle > 0.3 { l.lateral - towards_middle * need } else { l.lateral + towards_middle * need };
        let squeeze = squeeze_side.abs() + me.half_width <= lane.width.max(3.0) * 0.5 + 0.3 && !player_out;
        let side = if squeeze { squeeze_side } else { p.oncoming.map(|o| o.side).unwrap_or(0.0) };
        let out_len = Passing::out_len_for(side);
        let room = out_len + 1.0;
        // what a pass would need, now: no light or junction just past the obstacle (within
        // `JUNCTION_KEEP` of the front, a junction), the road going on, and the oncoming lane
        // (or the lane's own width) free for it
        let real = l.gap + if player { 0.5 } else { 0.0 };
        // what stands right behind it is passed with it, and beyond them the lane must have
        // room to come back in: my length, the way to pull back in and 4 m (`PASS_ROOM`; a car
        // or player standing or slower than 2 m/s there takes it; viz case09 #30816 pulled out
        // round two standing cars with three more beyond and stood out there 110 s)
        let (obstacle_len, room_beyond) = obstacle_run(p, &l, me.length() + out_len + PASS_ROOM);
        let pass_len = real + obstacle_len + st.front + st.rear + 6.0;
        let zone = if player { 60.0 } else { 25.0 };
        let light_near = p.signal.map(|s| s.dist < real + st.front + obstacle_len + zone).unwrap_or(false);
        let junction_near = p.junction.as_ref().map(|j| (j.entry < real + st.front + obstacle_len + zone || j.entry - st.front < JUNCTION_KEEP) && !matches!(l.what, Seen::Parked)).unwrap_or(false);
        let road_ok = p.road_left > pass_len + real + 30.0;
        let oncoming = p.oncoming.filter(|o| !solid && o.side.abs() >= 2.3 && o.side.abs() <= 5.5 && !o.blocked);
        // TODO(review11, pre-existing): `clear_for` is the oncoming car's time to the passer's
        // current position, compared with the pass time + 1.5 s; a car coming at 13 m/s meets
        // the passer before that point. It should be the time to where the pass ends.
        let oncoming_free = oncoming.is_some_and(|o| {
            let v_cap = (p.v_want).clamp(4.0, 14.0);
            o.clear_for > arrival_time(pass_len + Passing::back_len_for(side), v, st.accel, v_cap) + 1.5
        });
        let pass_possible = can_pass && !own_stop && !player_out && !bus_out && room_beyond && real >= 0.3 && !light_near && !junction_near && road_ok && (squeeze || oncoming_free);
        // (or round it into the lane beside, `Start::Bypass`)
        let bypass_possible = !light_near && !junction_near && [p.left, p.right].into_iter().flatten().any(|nb| could_bypass(net, me, p, &nb, &l));
        // keep back: room to pull out again, where the car can still stop there gently - only
        // where a pass (or a bypass) could start now; behind a standing player at a red 40 m
        // on, or a queue at a junction, the car closes up to its queue gap (viz3 review: 4785
        // keyframe rows 6 m back behind AI leads and 56 behind the player where no pass could
        // ever start)
        if (pass_possible || bypass_possible) && !(me.is_bus() && l.at_stop) && l.gap > room + 0.5 {
            let dist = l.gap - room;
            let need_brake = v * v / (2.0 * dist.max(0.1));
            if need_brake <= dp.comfort_brake * 1.5 {
                min_stop(&mut out.stop, &mut out.why, Some(st.front + dist), Why::Lead);
            }
        }
        // (a car that ended up too close - the thing stopped in front of it - pulls out
        // over what room it has once it has stood a while, rather than for ever)
        let too_close = l.gap < room - 1.0;
        let ready = (me.stopped >= 3.0 || matches!(l.what, Seen::Parked)) && (!too_close || me.stopped >= 10.0) && l.gap >= 1.5;
        if pass_possible && ready {
            let out_len = if too_close { (l.gap - 0.3).clamp(2.0, out_len) } else { out_len };
            let until = st.odometer + real + obstacle_len + st.front + st.rear + 2.0;
            let lane = if squeeze { None } else { oncoming.map(|o| o.lane) };
            out.start = Some(Start::Pass(Passing { lane, side, until, block: st.odometer + real, out_len, back_len: Passing::back_len_for(side), aborted: false, hold: 0.0, wait: omsi_sim::traffic::SIGNAL_BEFORE_CHANGE, obstacle: obstacle_id }));
        }
    }

    // 9. where the car sits in its lane: a bay, a swerve round a parked car, or the middle
    if me.passing.is_none() && st.change.is_none() {
        if let Some(bay) = me.external.bay {
            out.lateral = Some(bay);
        } else if !me.is_bus() {
            out.lateral = Some(p.kerb_swerve.unwrap_or(0.0));
        } else {
            out.lateral = Some(0.0);
        }
    }

    // 10. the end of a fixed route: stop at its end. [ROLLBACK r4d-B1] Not where the road
    //     itself ends (a street lane that runs into nothing, no stop of the bus's own on it):
    //     there the bus keeps its speed and is taken off as it reaches the end (`AiState::
    //     drive` finds no way on), as a plain car at a dead end; the timetable puts it back
    //     where its track goes on. Held 0.5 m before the end, two buses on the Anwai bridge
    //     (BeijingCity) stood for up to half an hour with fifty cars behind them.
    if !st.route.is_empty() && st.route_index + 1 >= st.route.len() && st.planned_next.is_none() {
        let end = lane.length() - st.s - 0.5;
        let dead_end = lane.next.is_empty() && lane.kind == omsi_sim::traffic::LaneKind::Street;
        if end < 150.0 && me.external.stop_at.is_none() && me.class == VehicleClass::ScheduledBus && !dead_end {
            min_stop(&mut out.stop, &mut out.why, Some(end.max(0.0)), Why::End);
        }
    }

    out.held = out.stop.is_some() || p.leader.map(|l| l.gap < 12.0).unwrap_or(false);
    if out.why == Why::None && out.lead.is_some() {
        out.why = Why::Lead;
    }
    out
}
