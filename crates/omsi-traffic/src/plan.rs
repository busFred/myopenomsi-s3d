//! The decision: from what the driver sees to one plan for this frame. A plan is the
//! lead to follow, a stop point, a lateral target and at most one manoeuvre to start.
//! No plan reads another vehicle's plan; the only cross-vehicle input is the junction
//! grant.

use crate::driver::VehicleClass;
use crate::perception::{NeighbourView, Perception, Seen};
use crate::reservation::{Grant, Hold};
use crate::sim::{Passing, Simulation, Vehicle};
use omsi_sim::traffic::{arrival_time, Aspect, Lead, Network};

/// Why the car holds back, the nearest reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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

    // 2. lights
    if let Some(sig) = p.signal {
        let gap = sig.dist - st.front;
        let comfortable = v * v / (2.0 * dp.comfort_brake * 1.4) + 1.0;
        let possible = v * v / (2.0 * omsi_sim::traffic::MAX_BRAKE * 0.8);
        let queued = v < 1.0 && p.leader.map(|l| l.gap < 8.0).unwrap_or(false);
        let go = match sig.aspect {
            Aspect::Green | Aspect::GreenYellow | Aspect::Dark => {
                // at the line on green: through, whatever it turns to in a moment
                if gap < 3.0 && !queued && v > 1.0 {
                    out.amber = Some((sig.ctl, sig.light));
                }
                true
            }
            Aspect::Yellow => {
                // the dilemma zone: through when stopping is no longer comfortable; a
                // driver who took the line on green just before goes a little further
                // out ("had the mark" alone let a car two places back in a queue go)
                let marked = me.amber == Some((sig.ctl, sig.light));
                let go = !queued && (gap < comfortable || (marked && gap < comfortable + 6.0));
                if go {
                    out.amber = Some((sig.ctl, sig.light));
                }
                go
            }
            Aspect::Red | Aspect::RedYellow => (me.amber == Some((sig.ctl, sig.light)) && gap < comfortable && !queued) || gap < possible - 0.5,
        };
        if !go {
            min_stop(&mut out.stop, &mut out.why, Some(sig.dist), Why::Light);
        }
    }

    // 3. the junction
    if let (Some(j), Some(g)) = (p.junction, grant) {
        if !g.go {
            // at the line, or, already on the lead-in piece of a long object, short of
            // the first meeting place of the way through
            // (a line too near to stop at gently any more is no stop either: the car
            // creeps on to the meeting place and stops short of that, as the claim counts
            // it; a line the bumper is already over drew an emergency stop for a frame)
            let at_line = j.entry - 0.3;
            let gently = v * v / (2.0 * (at_line - st.front - 0.6).max(0.05)) <= dp.comfort_brake * 1.5;
            let at = if !j.inside && (gently || v < 0.5) { at_line } else { j.conflict_start - 1.0 };
            if at > st.front - 0.3 {
                out.hold = g.hold;
                out.wait_for = g.wait_for;
                let w = if g.hold == Some(Hold::ExitFull) { Why::ExitFull } else { Why::Junction };
                min_stop(&mut out.stop, &mut out.why, Some(at), w);
            }
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
        let near_junction = p.junction.map(|j| !j.inside && j.entry < 40.0).unwrap_or(false);
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
        if pass.aborted {
            min_stop(&mut out.stop, &mut out.why, Some((pass.hold - st.odometer).max(0.0)), Why::Lead);
        } else {
            if st.odometer < pass.block + 2.0 {
                // edging out: gently
                out.accel_cap = Some(omsi_sim::ai_motion::PULL_OUT_ACCEL);
            }
            // the thing being passed no longer holds the car (the bent way clears it)
            if let Some(l) = p.leader {
                let passing_it = l.standing_obstacle() && st.odometer + l.gap + st.front < pass.until + 1.0;
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
        // keep back: room to pull out again, where the car can still stop there gently
        let room = out_len + 1.0;
        if !(me.is_bus() && l.at_stop) && l.gap > room + 0.5 {
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
        if can_pass && !own_stop && !player_out && ready {
            let out_len = if too_close { (l.gap - 0.3).clamp(2.0, out_len) } else { out_len };
            let real = l.gap + if player { 0.5 } else { 0.0 };
            let obstacle_len = l.length;
            let pass_len = real + obstacle_len + st.front + st.rear + 6.0;
            let zone = if player { 60.0 } else { 25.0 };
            let light_near = p.signal.map(|s| s.dist < real + st.front + obstacle_len + zone).unwrap_or(false);
            let junction_near = p.junction.map(|j| j.entry < real + st.front + obstacle_len + zone && !matches!(l.what, Seen::Parked)).unwrap_or(false);
            let road_ok = p.road_left > pass_len + real + 30.0;
            let until = st.odometer + real + obstacle_len + st.front + st.rear + 2.0;
            if real >= 0.3 && !light_near && !junction_near && road_ok {
                if squeeze {
                    out.start = Some(Start::Pass(Passing { lane: None, side, until, block: st.odometer + real, out_len, back_len: Passing::back_len_for(side), aborted: false, hold: 0.0 }));
                } else if let Some(o) = p.oncoming.filter(|o| !solid && o.side.abs() >= 2.3 && o.side.abs() <= 5.5 && !o.blocked) {
                    let v_cap = (p.v_want).clamp(4.0, 14.0);
                    let t_need = arrival_time(pass_len + Passing::back_len_for(side), v, st.accel, v_cap) + 1.5;
                    if o.clear_for > t_need {
                        out.start = Some(Start::Pass(Passing { lane: Some(o.lane), side, until, block: st.odometer + real, out_len, back_len: Passing::back_len_for(side), aborted: false, hold: 0.0 }));
                    }
                }
            }
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

    // 10. the end of a fixed route: stop at its end
    if !st.route.is_empty() && st.route_index + 1 >= st.route.len() && st.planned_next.is_none() {
        let end = lane.length() - st.s - 0.5;
        if end < 150.0 && me.external.stop_at.is_none() && me.class == VehicleClass::ScheduledBus {
            min_stop(&mut out.stop, &mut out.why, Some(end.max(0.0)), Why::End);
        }
    }

    out.held = out.stop.is_some() || p.leader.map(|l| l.gap < 12.0).unwrap_or(false);
    if out.why == Why::None && out.lead.is_some() {
        out.why = Why::Lead;
    }
    out
}
