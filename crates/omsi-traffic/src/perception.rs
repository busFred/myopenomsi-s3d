//! What a driver sees, as plain values: the vehicle ahead with its gap, the next signal,
//! the next junction, the lanes beside with their gaps, people in the way, the oncoming
//! lane and how long it stays clear. Read-only over the whole traffic; the plan reads
//! nothing else.

use crate::junction::Junctions;
use crate::sim::{Simulation, Vehicle};
use glam::{DVec2, DVec3};
use omsi_sim::collision::Obb;
use omsi_sim::traffic::{arrival_time, AiState, Aspect, LaneKind, Network, TrafficLightController};

/// What a headway is to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seen {
    /// Another AI vehicle on the way (index).
    Vehicle(usize),
    /// Another AI vehicle's body across the way (index), not on the way's lanes.
    Body(usize),
    Player(u32),
    Parked,
}

/// The thing ahead: the gap from the front bumper to it, how it moves along the way.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Headway {
    pub gap: f32,
    pub speed: f32,
    pub acc: f32,
    pub what: Seen,
    /// Seconds the thing has stood (a parked car: for ever).
    pub standing_for: f32,
    /// Its half width (for passing it).
    pub half_width: f32,
    /// From its rear to its front (m).
    pub length: f32,
    /// It is itself held by something (a lead, a light, a junction): a queue, not an
    /// obstacle to pass.
    pub held: bool,
    /// A bus serving its stop.
    pub at_stop: bool,
}

impl Headway {
    /// Something that stands and will go on standing: worth going round.
    pub fn standing_obstacle(&self) -> bool {
        if self.speed >= 0.3 {
            return false;
        }
        match self.what {
            Seen::Parked => true,
            Seen::Player(_) => self.standing_for > 4.0,
            Seen::Vehicle(_) | Seen::Body(_) => (self.at_stop && self.standing_for > 3.0) || (!self.held && self.standing_for > 4.0) || self.standing_for > 25.0,
        }
    }
}

impl Headway {
    fn nearer(a: Option<Headway>, b: Option<Headway>) -> Option<Headway> {
        match (a, b) {
            (Some(x), Some(y)) => Some(if y.gap < x.gap { y } else { x }),
            (x, None) => x,
            (None, y) => y,
        }
    }

    pub fn lead(&self) -> omsi_sim::traffic::Lead {
        omsi_sim::traffic::Lead { gap: self.gap, speed: self.speed, acc: self.acc }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignalView {
    /// Distance from the origin to the stop line.
    pub dist: f32,
    pub ctl: usize,
    pub light: usize,
    pub aspect: Aspect,
    pub until_go: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JunctionView {
    pub junction: usize,
    pub movement: usize,
    /// Distance from the origin to the start of the movement's lane (negative inside).
    pub entry: f32,
    pub inside: bool,
    /// Room past the junction for this vehicle.
    pub exit_room: bool,
}

/// A lane beside the current one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeighbourView {
    pub lane: usize,
    /// 1 left, 2 right.
    pub dir: i32,
    /// Nearest vehicle ahead there: (gap from my front, its speed).
    pub ahead: Option<(f32, f32)>,
    /// Nearest vehicle behind there: (gap to my rear, its speed).
    pub behind: Option<(f32, f32)>,
    /// How far the lane and its successors run on from beside me (m).
    pub runs_on: f32,
    /// A turn (1 left, 2 right) the lane leads into within 150 m, 0 none.
    pub turn: i32,
    /// A vehicle is passing on this lane against its direction.
    pub foreign: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OncomingView {
    pub lane: usize,
    /// Distance along it level with me.
    pub s: f32,
    /// Lateral offset of its middle from my lane (signed, m).
    pub side: f32,
    /// Seconds before the first oncoming vehicle is level with me.
    pub clear_for: f32,
    /// Something parked on it ahead.
    pub blocked: bool,
}

#[derive(Debug, Clone, Default)]
pub struct Perception {
    /// Lanes ahead: (lane, distance from the origin to its start).
    pub way: Vec<(usize, f32)>,
    pub leader: Option<Headway>,
    /// A vehicle's body across the way that held the car (index), for mutual waits.
    pub body_block: Option<usize>,
    pub signal: Option<SignalView>,
    pub junction: Option<JunctionView>,
    pub left: Option<NeighbourView>,
    pub right: Option<NeighbourView>,
    /// Distance from the origin to stop for a person in the way.
    pub people_stop: Option<f32>,
    /// Distance to stop for a pedestrian at a crossing of the junction.
    pub zebra_stop: Option<f32>,
    /// A route lane change due: (lane, dir).
    pub route_change: Option<(usize, i32)>,
    /// Metres the way runs on before it ends.
    pub road_left: f32,
    /// A parked car the way should swerve round: the lateral offset needed.
    pub kerb_swerve: Option<f32>,
    pub oncoming: Option<OncomingView>,
    /// A player's vehicle ahead indicating to pull out towards the traffic: the distance
    /// from my origin to its rear.
    pub letting_out: Option<f32>,
    /// A vehicle on another lane that reaches the joint ahead first.
    pub merging: Option<Headway>,
    /// The current lane's limit at the driver's desire (m/s).
    pub v_want: f32,
}

/// The lanes ahead along the car's way with the distance from the origin to each one's
/// start, within `within` metres. The lane being changed onto replaces the current one
/// once the move is under way.
pub fn way_lanes(net: &Network, st: &AiState, within: f32) -> Vec<(usize, f32)> {
    let mut out = Vec::new();
    let committed = st.change.map(|c| c.t > 0.4 || (c.bypass && c.wait <= 0.0)).unwrap_or(false);
    let (first, s, rest): (usize, f32, Vec<usize>) = match (committed, st.change) {
        (true, Some(c)) => (c.to, c.s_to, st.change_plan.clone()),
        _ => (st.lane, st.s, st.upcoming().collect()),
    };
    let Some(l) = net.lanes.get(first) else { return out };
    let mut d = -s;
    out.push((first, d));
    d += l.length();
    for n in rest {
        if d > within {
            break;
        }
        let Some(nl) = net.lanes.get(n) else { break };
        out.push((n, d));
        d += nl.length();
    }
    out
}

/// Is `p` inside the box, in plan view, grown by `margin`?
fn inside(b: &Obb, p: DVec2, margin: f64) -> bool {
    let [r, f] = b.axes();
    let rel = p - b.center;
    rel.dot(r).abs() <= b.half.x + margin && rel.dot(f).abs() <= b.half.y + margin
}

fn look_ahead(v: f32) -> f32 {
    (v * v / 3.0 + 2.0 * v + 20.0).clamp(70.0, 150.0)
}

/// The vehicle ahead on the way, through the lanes ahead.
fn leader_on_way(_net: &Network, sim: &Simulation, vehicles: &[Vehicle], i: usize, way: &[(usize, f32)], look: f32) -> Option<Headway> {
    let me = &vehicles[i];
    let st = &me.state;
    let signalling = st.change.map(|c| c.wait > 0.0 && c.t <= 0.0 && !c.bypass).unwrap_or(false);
    let passing = me.passing.filter(|p| !p.aborted);
    let mut best: Option<Headway> = None;
    let consider = |best: &mut Option<Headway>, lane_d0: f32, occ: &crate::sim::Occupant, limit: f32| {
        if occ.vehicle == i {
            return;
        }
        let o = &vehicles[occ.vehicle];
        let d_rear = lane_d0 + occ.s - o.state.rear;
        let gap = d_rear - st.front;
        if gap < -0.5 || gap > limit {
            return;
        }
        // side by side counts only when the two overlap sideways
        let my_lat = st.lateral_ahead((gap + st.front).max(0.0));
        if (my_lat - occ.lateral).abs() > me.half_width + o.half_width + 0.3 {
            return;
        }
        // the one behind that follows me is not my lead
        if o.lead_car == Some(i) {
            let rel = (o.pos - me.pos).truncate();
            let h = (me.heading as f64).to_radians();
            if rel.dot(DVec2::new(h.sin(), h.cos())) < 0.0 {
                return;
            }
        }
        if let Some((j, until)) = me.ignore_lead {
            if j == occ.vehicle && sim.time < until {
                return;
            }
        }
        // the standing thing I am passing
        if let Some(p) = passing {
            let standing = o.state.speed < 0.3;
            if standing && st.odometer + gap + st.front < p.until + 1.0 {
                return;
            }
        }
        let standing_for = if o.external.at_stop { o.external.standing_for.max(o.stopped) } else { o.stopped };
        let held = o.last.held || o.last.stop.is_some();
        let h = Headway { gap, speed: if occ.foreign { -o.state.speed } else { o.state.speed }, acc: o.state.acc, what: Seen::Vehicle(occ.vehicle), standing_for, half_width: o.half_width, length: o.length(), held, at_stop: o.external.at_stop };
        if best.map(|b| h.gap < b.gap).unwrap_or(true) {
            *best = Some(h);
        }
    };
    for &(l, d0) in way {
        if d0 > look {
            break;
        }
        for occ in sim.occupancy.get(&l).map(|v| v.as_slice()).unwrap_or(&[]) {
            consider(&mut best, d0, occ, look);
        }
    }
    // the lane being changed onto, before the move is under way
    if let Some(c) = st.change {
        if !way.iter().any(|w| w.0 == c.to) {
            let limit = if signalling { 6.0 } else { look };
            for occ in sim.occupancy.get(&c.to).map(|v| v.as_slice()).unwrap_or(&[]) {
                consider(&mut best, -c.s_to, occ, limit);
            }
        }
    }
    best
}

/// A vehicle on another lane into the same next lane that gets to the joint first.
fn merging_ahead(net: &Network, sim: &Simulation, vehicles: &[Vehicle], i: usize) -> Option<Headway> {
    let me = &vehicles[i];
    let st = &me.state;
    let next = st.planned_next?;
    let my_lane = net.lanes.get(st.lane)?;
    let d_me = my_lane.length() - st.s;
    if d_me > look_ahead(st.speed) {
        return None;
    }
    let limit = my_lane.speed_limit_kmh / 3.6;
    let t_me = arrival_time(d_me, st.speed, st.accel, limit);
    let mut best: Option<Headway> = None;
    for &p in net.prev.get(next)? {
        if p == st.lane {
            continue;
        }
        let pl = &net.lanes[p];
        for occ in sim.occupancy.get(&p).map(|v| v.as_slice()).unwrap_or(&[]) {
            if occ.vehicle == i || occ.foreign {
                continue;
            }
            let o = &vehicles[occ.vehicle];
            if o.state.planned_next != Some(next) {
                continue;
            }
            // someone held at a light or a junction never arrives
            if o.state.speed < 0.5 && o.last.stop.is_some() {
                continue;
            }
            let d_them = pl.length() - occ.s;
            let t_them = arrival_time(d_them, o.state.speed, o.state.accel, limit);
            let first = t_them + 0.4 < t_me || ((t_them - t_me).abs() <= 0.4 && occ.vehicle < i);
            if !first {
                continue;
            }
            let gap = (d_me - d_them - o.state.rear - st.front).max(0.3);
            let h = Headway { gap, speed: o.state.speed, acc: o.state.acc, what: Seen::Vehicle(occ.vehicle), standing_for: o.stopped, half_width: o.half_width, length: o.length(), held: true, at_stop: false };
            best = Headway::nearer(best, Some(h));
        }
    }
    best
}

/// A player's vehicle on the way.
fn player_on_way(net: &Network, sim: &Simulation, me: &Vehicle, look: f32, widen: f64) -> Option<(Headway, u32)> {
    let st = &me.state;
    let mut best: Option<(Headway, u32)> = None;
    for p in &sim.players {
        if (p.pos - me.pos).truncate().length() > 180.0 {
            continue;
        }
        if (p.pos.z - me.pos.z).abs() > 4.0 {
            continue;
        }
        let horizon = if p.priority { 5.0 } else { 1.5 };
        let same_way = {
            let d = omsi_sim::traffic::wrap_deg(p.heading - me.heading).abs();
            d < 60.0
        };
        let ahead = if same_way || p.speed.abs() < 0.5 { 0.0 } else { (p.speed.abs() * horizon) as f64 };
        let b = p.obb(ahead, 0.5 + widen);
        let mut d = 0.0f32;
        while d <= look {
            let q = st.way_point(net, d).truncate();
            if inside(&b, q, me.half_width as f64) {
                // bisect back for the edge
                let (mut lo, mut hi) = ((d - 1.5).max(0.0), d);
                for _ in 0..5 {
                    let mid = (lo + hi) * 0.5;
                    if inside(&b, st.way_point(net, mid).truncate(), me.half_width as f64) {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                let gap = hi - st.front;
                let (_, h) = net.lanes[st.lane].at(st.s.min(net.lanes[st.lane].length()));
                let hr = (h as f64).to_radians();
                let along = DVec2::new(hr.sin(), hr.cos());
                let speed = (p.fwd() * p.speed as f64).dot(along) as f32;
                let hw = Headway { gap, speed, acc: 0.0, what: Seen::Player(p.id), standing_for: p.still, half_width: p.half_width, length: p.half_len * 2.0, held: false, at_stop: false };
                if best.map(|b| hw.gap < b.0.gap).unwrap_or(true) {
                    best = Some((hw, p.id));
                }
                break;
            }
            d += if d < 30.0 { 1.5 } else { 3.0 };
        }
    }
    best
}

/// Another vehicle's body across the way (crossing a junction, pulling out, passing).
fn body_on_way(net: &Network, vehicles: &[Vehicle], i: usize, way: &[(usize, f32)]) -> Option<Headway> {
    let me = &vehicles[i];
    let st = &me.state;
    if me.class == crate::driver::VehicleClass::Air {
        return None;
    }
    let look = (st.speed * st.speed / (2.0 * st.decel.max(0.5)) + 2.0 * st.speed + 12.0).clamp(12.0, 70.0);
    let mut best: Option<Headway> = None;
    for (j, o) in vehicles.iter().enumerate() {
        if j == i {
            continue;
        }
        let rel = (o.pos - me.pos).truncate();
        if rel.length() > look as f64 + o.length() as f64 || (o.pos.z - me.pos.z).abs() > 3.0 {
            continue;
        }
        // on my way: the occupancy has it
        if way.iter().any(|w| w.0 == o.state.lane) && o.passing.is_none() {
            continue;
        }
        // both blocking each other: the higher id goes
        if o.last.body_block == Some(i) && me.id > o.id {
            continue;
        }
        let b = o.obb();
        let mut d = 0.0f32;
        while d <= look {
            let q = st.way_point(net, d).truncate();
            if inside(&b, q, me.half_width as f64) {
                let gap = d - st.front;
                if gap > -0.5 {
                    let hr = (me.heading as f64).to_radians();
                    let along = DVec2::new(hr.sin(), hr.cos());
                    let speed = (b.velocity).dot(along) as f32;
                    let oh = (o.heading as f64).to_radians();
                    let ov = DVec2::new(oh.sin(), oh.cos()) * o.state.speed as f64;
                    let speed = if b.velocity.length_squared() > 0.0 { speed } else { ov.dot(along) as f32 };
                    let h = Headway { gap, speed, acc: 0.0, what: Seen::Body(j), standing_for: o.stopped, half_width: o.half_width, length: o.length(), held: o.last.held || o.last.stop.is_some(), at_stop: o.external.at_stop };
                    best = Headway::nearer(best, Some(h));
                }
                break;
            }
            d += if d < 20.0 { 0.75 } else { 1.5 };
        }
    }
    best
}

/// Parked cars on the way: a blocking one as a headway, a protruding one as a swerve.
fn parked_on_way(sim: &Simulation, me: &Vehicle, way: &[(usize, f32)]) -> (Option<Headway>, Option<f32>) {
    let st = &me.state;
    let mut lead: Option<Headway> = None;
    let mut swerve: Option<f32> = None;
    for &(l, d0) in way {
        if d0 > 100.0 {
            break;
        }
        for p in sim.parked.get(&l).map(|v| v.as_slice()).unwrap_or(&[]) {
            let along = d0 + p.s;
            if !(-6.0..=100.0).contains(&along) {
                continue;
            }
            let my_lat = st.lateral_ahead(along.max(0.0));
            let lat = p.lateral - my_lat;
            let clearance = me.half_width + p.half_width + 0.35;
            if lat.abs() >= clearance {
                continue;
            }
            let need = (clearance - lat.abs()) * -lat.signum();
            if need.abs() <= 1.1 && along < 30.0 && me.passing.is_none() {
                if swerve.map(|s| need.abs() > s.abs()).unwrap_or(true) {
                    swerve = Some(need);
                }
            } else if need.abs() > 1.1 {
                let gap = along - p.half_len - st.front - 2.0;
                let h = Headway { gap, speed: 0.0, acc: 0.0, what: Seen::Parked, standing_for: f32::INFINITY, half_width: p.half_width, length: p.half_len * 2.0, held: false, at_stop: false };
                lead = Headway::nearer(lead, Some(h));
            }
        }
    }
    (lead, swerve)
}

/// Someone on foot in the strip ahead.
fn people_ahead(net: &Network, sim: &Simulation, me: &Vehicle) -> Option<f32> {
    let st = &me.state;
    let v = st.speed;
    let reach = (v * v / 5.0 + v + 6.0).clamp(8.0, 45.0);
    let mut best: Option<f32> = None;
    for p in &sim.people {
        let rel = p.pos - me.pos;
        if rel.truncate().length() > (st.front + reach + 6.0) as f64 || rel.z.abs() > 2.5 {
            continue;
        }
        let half = if p.on_crossing {
            me.half_width + 1.5
        } else if p.waiting_for_bus && me.is_bus() {
            me.half_width - 0.3
        } else {
            me.half_width + 0.3
        };
        let t = ((rel.truncate().length() as f32) / v.max(1.0)).min(2.0);
        let later = p.pos + (p.vel * t as f64).extend(0.0);
        let mut d = 0.0f32;
        while d <= reach + st.front {
            let q = st.way_point(net, d);
            let q2 = st.way_point(net, d + 1.0);
            let fwd = (q2 - q).truncate().normalize_or_zero();
            let right = DVec2::new(fwd.y, -fwd.x);
            for pos in [p.pos, later] {
                let r = (pos - q).truncate();
                let along = r.dot(fwd);
                let across = r.dot(right).abs();
                if along.abs() <= 0.55 && (across as f32) < half {
                    // someone who has stood at the edge for a while does not hold a standing car
                    if !p.on_crossing && v < 0.3 && p.still > 3.0 && (across as f32) > half - 0.9 {
                        continue;
                    }
                    let stop = d - 1.5;
                    if best.map(|b| stop < b).unwrap_or(true) {
                        best = Some(stop);
                    }
                }
            }
            d += 1.0;
        }
    }
    best
}

/// The next light on the way within 150 m. A second light of the same junction object on a
/// straight run holds nobody: the car has passed its line.
fn signal_ahead(net: &Network, sim: &Simulation, st: &AiState, way: &[(usize, f32)]) -> Option<SignalView> {
    let mut seen_object: Option<(((i32, i32), i64), f32)> = None;
    for &(l, d0) in way {
        if d0 > 150.0 {
            break;
        }
        let lane = &net.lanes[l];
        let Some((ctl, li)) = lane.traffic_light else { continue };
        let key = lane.key.map(|k| (k.tile, k.id));
        if let (Some(k), Some((sk, sh))) = (key, seen_object) {
            if k == sk && omsi_sim::traffic::wrap_deg(lane.start_heading() - sh).abs() < 45.0 {
                continue;
            }
        }
        if let Some(k) = key {
            if seen_object.is_none() {
                seen_object = Some((k, lane.start_heading()));
            }
        }
        if d0 < -0.5 {
            // past the line already
            continue;
        }
        let Some(c) = sim.lights.get(ctl) else { continue };
        let _ = st;
        return Some(SignalView { dist: d0, ctl, light: li, aspect: TrafficLightController::aspect(c.state(li)), until_go: c.time_until_go(li) });
    }
    None
}

/// The next junction on the way (or the one the car is in).
fn junction_ahead(net: &Network, sim: &Simulation, vehicles: &[Vehicle], i: usize, way: &[(usize, f32)]) -> Option<JunctionView> {
    let me = &vehicles[i];
    let js: &Junctions = &sim.junctions;
    for &(l, d0) in way {
        if d0 > 120.0 {
            break;
        }
        let Some(&(j, m)) = js.of_lane.get(&l) else { continue };
        let inside = d0 <= 0.0;
        let node = &js.nodes[j];
        let mv = &node.movements[m];
        // room past the junction: the nearest standing vehicle on an exit lane
        let need = me.length() + me.state.min_gap;
        let mut exit_room = mv.exits.is_empty();
        for &e in &mv.exits {
            let mut room = f32::INFINITY;
            let mut d = 0.0;
            let mut lane = e;
            for _ in 0..4 {
                for occ in sim.occupancy.get(&lane).map(|v| v.as_slice()).unwrap_or(&[]) {
                    if occ.vehicle == i || occ.foreign {
                        continue;
                    }
                    let o = &vehicles[occ.vehicle];
                    if o.state.speed < 1.5 {
                        room = room.min(d + occ.s - o.state.rear);
                    }
                }
                d += net.lanes[lane].length();
                if d > 40.0 || room.is_finite() {
                    break;
                }
                let Some(&n) = net.lanes[lane].next.first() else { break };
                lane = n;
            }
            if room >= need {
                exit_room = true;
            }
        }
        return Some(JunctionView { junction: j, movement: m, entry: d0, inside, exit_room });
    }
    None
}

/// A pedestrian at a crossing of the junction ahead.
fn zebra_ahead(net: &Network, sim: &Simulation, st: &AiState, way: &[(usize, f32)]) -> Option<f32> {
    let mut best: Option<f32> = None;
    for &(l, d0) in way {
        if d0 > 40.0 {
            break;
        }
        for &(w, sa, sw) in net.walks.get(l).map(|v| v.as_slice()).unwrap_or(&[]) {
            let near = sim.walkers.iter().any(|&(wl, ws)| wl == w && (ws - sw).abs() < 3.0);
            if !near {
                continue;
            }
            let point = d0 + sa;
            if point - st.front < 30.0 && point - st.front > -1.0 {
                let stop = point - 2.5;
                if best.map(|b| stop < b).unwrap_or(true) {
                    best = Some(stop);
                }
            }
        }
    }
    best
}

fn neighbour(net: &Network, sim: &Simulation, vehicles: &[Vehicle], i: usize, nb: usize, dir: i32) -> Option<NeighbourView> {
    let me = &vehicles[i];
    let st = &me.state;
    let lane = net.lanes.get(nb)?;
    if lane.no_cars || !lane.allows(st.veh_type) {
        return None;
    }
    let s_b = net.beside_s(st.lane, nb, st.s);
    let mut ahead: Option<(f32, f32)> = None;
    let mut behind: Option<(f32, f32)> = None;
    let mut foreign = false;
    let mut note = |d_rel: f32, o: &Vehicle, occ_foreign: bool| {
        if occ_foreign {
            foreign = true;
        }
        if d_rel >= 0.0 {
            let gap = d_rel - o.state.rear - st.front;
            if ahead.map(|a| gap < a.0).unwrap_or(true) {
                ahead = Some((gap, o.state.speed));
            }
        } else {
            let gap = -d_rel - o.state.front - st.rear;
            if behind.map(|b| gap < b.0).unwrap_or(true) {
                behind = Some((gap, o.state.speed));
            }
        }
    };
    for occ in sim.occupancy.get(&nb).map(|v| v.as_slice()).unwrap_or(&[]) {
        if occ.vehicle == i {
            continue;
        }
        note(occ.s - s_b, &vehicles[occ.vehicle], occ.foreign);
    }
    // behind, on the lanes leading in (within 80 m)
    for &p in net.prev.get(nb).map(|v| v.as_slice()).unwrap_or(&[]) {
        let pl = &net.lanes[p];
        for occ in sim.occupancy.get(&p).map(|v| v.as_slice()).unwrap_or(&[]) {
            if occ.vehicle == i {
                continue;
            }
            let d_rel = occ.s - pl.length() - s_b;
            if d_rel > -80.0 {
                note(d_rel, &vehicles[occ.vehicle], occ.foreign);
            }
        }
    }
    // ahead, on the first lane after
    if let Some(&n) = lane.next.first() {
        for occ in sim.occupancy.get(&n).map(|v| v.as_slice()).unwrap_or(&[]) {
            if occ.vehicle == i {
                continue;
            }
            note(lane.length() - s_b + occ.s, &vehicles[occ.vehicle], occ.foreign);
        }
    }
    let runs_on = net.reach.get(nb).copied().unwrap_or(lane.length()) - s_b;
    // a turn the lane leads into within 150 m
    let mut turn = lane.turn;
    let mut d = lane.length() - s_b;
    let mut cur = nb;
    while turn == 0 && d < 150.0 {
        let Some(&n) = net.lanes[cur].next.first() else { break };
        turn = net.lanes[n].turn;
        d += net.lanes[n].length();
        cur = n;
    }
    Some(NeighbourView { lane: nb, dir, ahead, behind, runs_on, turn, foreign })
}

/// The oncoming lane and how long it stays clear (only looked for behind a standing thing).
fn oncoming(net: &Network, sim: &Simulation, vehicles: &[Vehicle], i: usize, pass_len: f32) -> Option<OncomingView> {
    let me = &vehicles[i];
    let st = &me.state;
    let (opp, os, side) = net.opposite(st.lane, st.s)?;
    let ol = &net.lanes[opp];
    let limit = ol.speed_limit_kmh / 3.6;
    let look = (limit.clamp(8.0, 20.0) * 12.0 + 20.0).min(300.0);
    let mut clear_for = f32::INFINITY;
    // an oncoming vehicle comes towards `os` from larger distances along `opp`... no: the
    // opposite lane runs the other way, so vehicles before `os` along it have passed me
    // and those at larger `s` are still to come? Along `opp`, travel goes from small s to
    // large s; level with me is `os`; a vehicle still to come is at s < os.
    for (l, off, _) in net.upstream(opp, os, look, 48) {
        for occ in sim.occupancy.get(&l).map(|v| v.as_slice()).unwrap_or(&[]) {
            if occ.vehicle == i {
                continue;
            }
            let o = &vehicles[occ.vehicle];
            let pos = occ.s + off;
            // past me (on the oncoming lane, further along it than my place)
            if pos > os + pass_len {
                continue;
            }
            // not yet level: the distance it still has to me
            let dist = (os - pos).max(0.0);
            // a car standing at a red light with time to go: counts from when it goes
            let mut t = arrival_time(dist, o.state.speed, o.state.accel, limit);
            if o.last.why == crate::plan::Why::Light && o.state.speed < 0.5 {
                if let Some(sig) = o.last.stop {
                    let _ = sig;
                    t += 5.0;
                }
            }
            // inside the stretch I will use: no time at all
            if pos > os - st.rear - 1.0 && pos < os + pass_len {
                t = 0.0;
            }
            clear_for = clear_for.min(t);
        }
    }
    let blocked = sim.parked.get(&opp).map(|v| v.iter().any(|p| p.lateral.abs() <= 1.6 && p.s > os - pass_len - 10.0 && p.s < os + 10.0)).unwrap_or(false);
    let side = side * sim.oncoming_sign();
    Some(OncomingView { lane: opp, s: os, side, clear_for, blocked })
}

/// A player's vehicle ahead on my side of the road, indicating to pull out.
fn letting_out(sim: &Simulation, me: &Vehicle) -> Option<f32> {
    let st = &me.state;
    let out_side = sim.out_side() as u8;
    let hr = (me.heading as f64).to_radians();
    let fwd = DVec2::new(hr.sin(), hr.cos());
    let right = DVec2::new(fwd.y, -fwd.x);
    let mut best: Option<f32> = None;
    for p in &sim.players {
        if p.blinker != out_side || p.signalling > 20.0 || p.speed.abs() > 3.0 {
            continue;
        }
        let rel = (p.pos - me.pos).truncate();
        let along = rel.dot(fwd) as f32;
        let across = rel.dot(right).abs() as f32;
        if along < 0.0 || along > 120.0 || p.fwd().dot(fwd) < 0.8 {
            continue;
        }
        if across > me.half_width + p.half_width + 2.5 {
            continue;
        }
        let d = along - p.half_len - st.front;
        if d > 0.5 && d >= st.speed * st.speed / (2.0 * st.decel.max(0.5)) {
            if best.map(|b| d < b).unwrap_or(true) {
                best = Some(d);
            }
        }
    }
    best
}

/// Everything vehicle `i` sees this frame.
pub fn perceive(net: &Network, sim: &Simulation, vehicles: &[Vehicle], i: usize) -> Perception {
    let me = &vehicles[i];
    let st = &me.state;
    let mut p = Perception::default();
    let Some(lane) = net.lanes.get(st.lane) else { return p };
    p.v_want = (lane.speed_limit_kmh * st.desire).min(st.max_speed_kmh).max(3.0) / 3.6;
    let look = look_ahead(st.speed);
    p.way = way_lanes(net, st, 220.0);
    p.road_left = net.reach.get(st.lane).copied().unwrap_or(lane.length()) - st.s;
    let air = me.class == crate::driver::VehicleClass::Air;

    p.leader = leader_on_way(net, sim, vehicles, i, &p.way, look);
    p.merging = merging_ahead(net, sim, vehicles, i);
    p.leader = Headway::nearer(p.leader, p.merging);
    if !air {
        if let Some((h, _)) = player_on_way(net, sim, me, look, 0.0) {
            p.leader = Headway::nearer(p.leader, Some(h));
        }
        if let Some(h) = body_on_way(net, vehicles, i, &p.way) {
            // a body across the way replaces the lead only when clearly nearer
            if p.leader.map(|l| h.gap < l.gap - 0.5).unwrap_or(true) {
                if let Seen::Body(j) = h.what {
                    p.body_block = Some(j);
                }
                p.leader = Some(h);
            }
        }
        let (parked, swerve) = parked_on_way(sim, me, &p.way);
        p.leader = Headway::nearer(p.leader, parked);
        p.kerb_swerve = swerve;
        p.people_stop = people_ahead(net, sim, me);
        p.signal = signal_ahead(net, sim, st, &p.way);
        p.junction = junction_ahead(net, sim, vehicles, i, &p.way);
        p.zebra_stop = zebra_ahead(net, sim, st, &p.way);
        p.letting_out = letting_out(sim, me);
        if lane.kind == LaneKind::Street && !me.is_rail() {
            if let Some(l) = lane.left {
                p.left = neighbour(net, sim, vehicles, i, l, 1);
            }
            if let Some(r) = lane.right {
                p.right = neighbour(net, sim, vehicles, i, r, 2);
            }
            // the oncoming lane matters behind something standing, or while out on it
            let standing = p.leader.map(|l| l.standing_obstacle() && l.gap < 45.0).unwrap_or(false);
            if (standing || me.passing.is_some()) && lane.left.is_none() && lane.right.is_none() {
                let obstacle = p.leader.map(|l| l.gap + l.length).unwrap_or(10.0);
                let pass_len = obstacle + st.front + st.rear + 6.0;
                p.oncoming = oncoming(net, sim, vehicles, i, pass_len);
            }
        }
    }
    p.route_change = st.route_change_due(net);
    p
}

#[allow(dead_code)]
fn _unused(_: DVec3) {}
