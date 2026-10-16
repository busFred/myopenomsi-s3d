//! The frame: positions, occupancy, lights, then perception → claims → plan → act for
//! every vehicle. `Simulation` owns the junction table, the light programs and the
//! frame's inputs (players, people, parked cars); the vehicles and the network are the
//! caller's, so the app keeps its own vehicle instances beside them and a test needs
//! neither.

use crate::driver::{DriverParams, Impatience, VehicleClass};
use crate::junction::Junctions;
use crate::perception::{self, Perception, Seen};
use crate::plan::{self, Plan, Start, Why};
use crate::reservation::{self, Claim, Grant, Hold};
use glam::{DVec2, DVec3};
use hashbrown::HashMap;
use omsi_sim::collision::Obb;
use omsi_sim::traffic::{AiState, Lead, Network, TrafficLightController};

/// A player's (or another LAN player's) vehicle as the traffic sees it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayerView {
    pub id: u32,
    pub pos: DVec3,
    /// Degrees, clockwise from north.
    pub heading: f32,
    pub half_len: f32,
    pub half_width: f32,
    /// Signed along the heading (m/s, negative reversing).
    pub speed: f32,
    /// Seconds standing (|speed| < 0.3).
    pub still: f32,
    /// 0 off, 1 left, 2 right, 3 hazard.
    pub blinker: u8,
    /// Seconds the indicator has been on.
    pub signalling: f32,
    /// `TrafficPriority` set: the vehicle has the right of way everywhere.
    pub priority: bool,
}

impl PlayerView {
    pub fn fwd(&self) -> DVec2 {
        let h = (self.heading as f64).to_radians();
        DVec2::new(h.sin(), h.cos())
    }

    /// The box, reaching `ahead` metres further forward.
    pub fn obb(&self, ahead: f64, margin: f64) -> Obb {
        let f = self.fwd();
        let c = self.pos.truncate() + f * (ahead * 0.5);
        Obb { center: c, half: DVec2::new(self.half_width as f64 + margin, self.half_len as f64 + ahead * 0.5 + margin), heading: (self.heading as f64).to_radians(), z0: self.pos.z - 1.0, z1: self.pos.z + 3.0, velocity: f * self.speed as f64, mass: 0.0, pole: None, id: -1 }
    }
}

/// A person on or by the road.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Person {
    pub pos: DVec3,
    pub vel: DVec2,
    /// On a crossing (a zebra, a signalled crossing): cars stop for them further away.
    pub on_crossing: bool,
    /// Seconds standing still.
    pub still: f32,
    /// Waiting at a bus stop: a bus keeps well in, a car drives by.
    pub waiting_for_bus: bool,
}

/// A parked car beside a lane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Parked {
    pub s: f32,
    /// Lateral offset of its middle from the lane (m, + right).
    pub lateral: f32,
    pub half_width: f32,
    pub half_len: f32,
}

/// Overtaking on the oncoming lane, or squeezing past something inside the lane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Passing {
    /// The oncoming lane used, None when the car stays in its own lane.
    pub lane: Option<usize>,
    /// The lateral offset driven (m, signed).
    pub side: f32,
    /// Odometer reading at which the car is past the obstacle and may come back.
    pub until: f32,
    /// Odometer reading of the obstacle's near end.
    pub block: f32,
    /// Metres of way over which the car pulls out, and over which it comes back in.
    pub out_len: f32,
    pub back_len: f32,
    /// The oncoming traffic came: stop short of the obstacle and come back in.
    pub aborted: bool,
    /// Odometer reading at which an aborted pass holds the car.
    pub hold: f32,
}

impl Passing {
    /// The way over which a car pulls `side` metres out: steep, three metres of way per
    /// metre sideways (the bay ramp of eight is for comfort, not for a car edging out
    /// from behind something).
    pub fn out_len_for(side: f32) -> f32 {
        (side.abs() * 3.0).clamp(5.0, 12.0)
    }

    pub fn back_len_for(side: f32) -> f32 {
        (side.abs() * 6.0).clamp(8.0, 20.0)
    }

    /// The bent way, as (odometer, offset) points from `odo` and the offset `from` there.
    pub fn curve(&self, odo: f32, from: f32) -> Vec<(f32, f32)> {
        let level = self.until.max(odo + self.out_len + 0.5);
        vec![(odo, from), (odo + self.out_len, self.side), (level, self.side), (level + self.back_len, 0.0)]
    }
}

/// What the app tells the planner about this vehicle this frame.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct External {
    /// A stop the bus service asks for (distance from the origin).
    pub stop_at: Option<f32>,
    /// Lateral offset into a bay.
    pub bay: Option<f32>,
    /// Indicator off whatever the planner says (doors open).
    pub blinker_off: bool,
    /// At its stop: standing there is service, not a jam.
    pub at_stop: bool,
    /// Seconds the bus has stood at its stop (others pass it after a while).
    pub standing_for: f32,
    /// `TrafficPriority`: goes first at junctions.
    pub priority: bool,
}

/// The last decision, kept for consumers and the debug overlay.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Decision {
    pub why: Why,
    pub lead: Option<Lead>,
    pub lead_seen: Option<Seen>,
    pub stop: Option<f32>,
    pub held: bool,
    pub hold: Option<Hold>,
    /// Vehicle index waited for at a junction.
    pub wait_for: Option<usize>,
    /// Another vehicle's body (not on the way) that held this one.
    pub body_block: Option<usize>,
}

/// One AI vehicle in the traffic: its way (`state`), its driver and its manoeuvres.
#[derive(Debug, Clone)]
pub struct Vehicle {
    pub id: u64,
    pub class: VehicleClass,
    pub state: AiState,
    pub params: DriverParams,
    pub impatience: Impatience,
    pub half_width: f32,
    /// Seconds standing (not at a bus stop).
    pub stopped: f32,
    /// (odometer at the last progress, seconds without any).
    pub progress: (f32, f32),
    /// A random car that stood too long: leaves once out of sight.
    pub gone: bool,
    /// Just put on the road: its speed is trimmed to what is ahead.
    pub fresh: f32,
    /// Seconds before a car leaving a parking space moves.
    pub pull_out: f32,
    pub passing: Option<Passing>,
    /// Past the line on green/yellow: goes through on red too.
    pub amber: bool,
    pub external: External,
    pub last: Decision,
    /// Index of the vehicle followed last frame.
    pub lead_car: Option<usize>,
    /// (vehicle, time until) a lead is ignored to break a mutual wait.
    pub ignore_lead: Option<(usize, f64)>,
    /// Indicator shown, and since when (debounced).
    pub blinker_shown: (i32, f64),
    /// World position of the origin and heading (degrees), from the way.
    pub pos: DVec3,
    pub heading: f32,
    /// Seconds in a row the lane change planner started one.
    pub change_streak: u8,
}

impl Vehicle {
    pub fn new(id: u64, class: VehicleClass, state: AiState, half_width: f32, seed: u64) -> Vehicle {
        let mut state = state;
        let params = DriverParams::seeded(class, seed);
        params.apply(&mut state);
        Vehicle {
            id,
            class,
            state,
            params,
            impatience: Impatience::default(),
            half_width,
            stopped: 0.0,
            progress: (0.0, 0.0),
            gone: false,
            fresh: 1.5,
            pull_out: 0.0,
            passing: None,
            amber: false,
            external: External::default(),
            last: Decision::default(),
            lead_car: None,
            ignore_lead: None,
            blinker_shown: (0, 0.0),
            pos: DVec3::ZERO,
            heading: 0.0,
            change_streak: 0,
        }
    }

    pub fn length(&self) -> f32 {
        self.state.front + self.state.rear
    }

    /// The body on the ground.
    pub fn obb(&self) -> Obb {
        Obb::vehicle(self.pos.truncate(), self.heading as f64, self.state.front as f64, self.state.rear as f64, self.half_width as f64)
    }

    /// Standing still, held by something.
    pub fn is_held(&self) -> bool {
        self.state.speed < 0.3 && self.last.held
    }

    pub fn is_bus(&self) -> bool {
        self.class.is_bus()
    }

    pub fn is_rail(&self) -> bool {
        self.class == VehicleClass::Rail
    }
}

/// A vehicle's place on a lane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Occupant {
    pub vehicle: usize,
    pub s: f32,
    pub lateral: f32,
    /// Passing on this lane against its direction.
    pub foreign: bool,
}

/// The traffic's shared state and this frame's inputs.
#[derive(Debug, Clone)]
pub struct Simulation {
    pub junctions: Junctions,
    pub lights: Vec<TrafficLightController>,
    pub time: f64,
    pub day_time: f64,
    pub time_scale: f64,
    pub players: Vec<PlayerView>,
    pub people: Vec<Person>,
    /// Pedestrians on footpath lanes: (lane, distance), for push buttons and zebras.
    pub walkers: Vec<(usize, f32)>,
    pub parked: HashMap<usize, Vec<Parked>>,
    pub occupancy: HashMap<usize, Vec<Occupant>>,
    pub left_hand: bool,
    /// Vehicles that reached a dead end this frame (indices, descending).
    pub removed: Vec<usize>,
    /// Vehicles counted standing at a red light this frame (diagnostics).
    pub held_at_red: usize,
}

impl Simulation {
    pub fn new(net: &Network, lights: Vec<TrafficLightController>) -> Simulation {
        Simulation {
            junctions: Junctions::compile(net),
            lights,
            time: 0.0,
            day_time: 0.0,
            time_scale: 1.0,
            players: Vec::new(),
            people: Vec::new(),
            walkers: Vec::new(),
            parked: HashMap::new(),
            occupancy: HashMap::new(),
            left_hand: net.left_hand,
            removed: Vec::new(),
            held_at_red: 0,
        }
    }

    /// Lanes from index `first` on are new: compile their junctions.
    pub fn network_grew(&mut self, net: &Network, first: usize) {
        self.junctions.extend(net, first);
    }

    /// The side the oncoming traffic is on as a lateral sign: -1 (left) driving on the
    /// right, +1 on a left-hand-traffic map.
    pub fn oncoming_sign(&self) -> f32 {
        if self.left_hand {
            1.0
        } else {
            -1.0
        }
    }

    /// The indicator a vehicle shows when pulling out towards the traffic.
    pub fn out_side(&self) -> i32 {
        if self.left_hand {
            2
        } else {
            1
        }
    }

    /// Place every vehicle on the ground from its way.
    pub fn update_positions(&self, net: &Network, vehicles: &mut [Vehicle]) {
        for v in vehicles.iter_mut() {
            if net.lanes.get(v.state.lane).is_none() {
                continue;
            }
            let p = v.state.way_point(net, 0.0);
            let a = v.state.way_point(net, -1.5);
            let b = v.state.way_point(net, 1.5);
            let d = (b - a).truncate();
            v.pos = p;
            if d.length_squared() > 1e-6 {
                v.heading = d.x.atan2(d.y).to_degrees() as f32;
            } else {
                v.heading = net.lanes[v.state.lane].at(v.state.s).1;
            }
        }
    }

    /// Which vehicle is where on which lane.
    pub fn build_occupancy(&mut self, net: &Network, vehicles: &[Vehicle]) {
        self.occupancy.clear();
        for (i, v) in vehicles.iter().enumerate() {
            let st = &v.state;
            let Some(lane) = net.lanes.get(st.lane) else { continue };
            let _ = lane;
            self.occupancy.entry(st.lane).or_default().push(Occupant { vehicle: i, s: st.s, lateral: st.lateral, foreign: false });
            if let Some(p) = st.prev_lane {
                if st.s < st.rear + 1.0 && st.change.is_none() {
                    if let Some(pl) = net.lanes.get(p) {
                        self.occupancy.entry(p).or_default().push(Occupant { vehicle: i, s: pl.length() + st.s, lateral: st.lateral, foreign: false });
                    }
                }
            }
            if let Some(c) = st.change {
                self.occupancy.entry(c.to).or_default().push(Occupant { vehicle: i, s: c.s_to, lateral: 0.0, foreign: false });
            }
            if let Some(p) = v.passing {
                if let (Some(opp), true) = (p.lane, st.lateral.abs() > 1.2) {
                    if let Some(ol) = net.lanes.get(opp) {
                        if let Some((os, _)) = ol.nearest_point(v.pos) {
                            self.occupancy.entry(opp).or_default().push(Occupant { vehicle: i, s: os, lateral: 0.0, foreign: true });
                        }
                    }
                }
            }
        }
        for occ in self.occupancy.values_mut() {
            occ.sort_by(|a, b| a.s.total_cmp(&b.s));
        }
    }

    /// Lanes starting just ahead of a player, and the ones after them, within `within`
    /// metres: what the player's vehicle will drive along.
    fn lanes_ahead_of(&self, net: &Network, p: &PlayerView, within: f32) -> Vec<(usize, f32)> {
        let f = p.fwd();
        let mut out: Vec<(usize, f32)> = Vec::new();
        for l in net.lanes_starting_near(p.pos, 25.0) {
            let lane = &net.lanes[l];
            if lane.kind != omsi_sim::traffic::LaneKind::Street {
                continue;
            }
            let rel = (lane.start() - p.pos).truncate();
            let along = rel.dot(f);
            let across = rel.perp_dot(f).abs();
            if !(0.0..=25.0).contains(&along) || across > 6.0 || (lane.start().z - p.pos.z).abs() > 4.0 {
                continue;
            }
            if omsi_sim::traffic::wrap_deg(lane.start_heading() - p.heading).abs() > 60.0 {
                continue;
            }
            out.push((l, along as f32));
        }
        // and on from them
        let mut k = 0;
        while k < out.len() && out.len() < 24 {
            let (l, d) = out[k];
            let end = d + net.lanes[l].length();
            if end < within {
                for &n in &net.lanes[l].next {
                    if !out.iter().any(|o| o.0 == n) {
                        out.push((n, end));
                    }
                }
            }
            k += 1;
        }
        out
    }

    fn update_lights(&mut self, net: &Network, vehicles: &[Vehicle], dt: f32) {
        for c in &mut self.lights {
            c.request.iter_mut().for_each(|r| *r = false);
        }
        let ask = |lights: &mut Vec<TrafficLightController>, ctl: usize, li: usize, gap: f32| {
            if let Some(c) = lights.get_mut(ctl) {
                if gap <= c.approach_dist(li) {
                    if let Some(r) = c.request.get_mut(li) {
                        *r = true;
                    }
                }
            }
        };
        for v in vehicles {
            for (l, d0) in perception::way_lanes(net, &v.state, 160.0) {
                if let Some((ctl, li)) = net.lanes[l].traffic_light {
                    ask(&mut self.lights, ctl, li, d0 - v.state.front);
                }
            }
        }
        for p in &self.players {
            for (l, d0) in self.lanes_ahead_of(net, p, 160.0) {
                if let Some((ctl, li)) = net.lanes[l].traffic_light {
                    ask(&mut self.lights, ctl, li, d0 - p.half_len);
                }
            }
        }
        for &(l, s) in &self.walkers {
            let Some(lane) = net.lanes.get(l) else { continue };
            if let Some((ctl, li)) = lane.traffic_light {
                if let Some(c) = self.lights.get_mut(ctl) {
                    let reach = c.approach_dist(li).min(10.0);
                    if lane.length() - s <= reach || s <= reach {
                        if let Some(r) = c.request.get_mut(li) {
                            *r = true;
                        }
                    }
                }
            }
        }
        for c in &mut self.lights {
            c.start(self.day_time);
            c.advance(dt);
        }
    }

    /// One frame of `dt` seconds for every vehicle.
    pub fn tick(&mut self, net: &Network, dt: f32, vehicles: &mut [Vehicle]) {
        self.time += dt as f64;
        self.day_time += dt as f64 * self.time_scale;
        self.removed.clear();
        self.held_at_red = 0;
        self.update_positions(net, vehicles);
        self.build_occupancy(net, vehicles);
        self.update_lights(net, vehicles, dt);

        // perception: read-only over everything
        let perceptions: Vec<Perception> = (0..vehicles.len()).map(|i| perception::perceive(net, self, vehicles, i)).collect();

        // claims and grants at the junctions
        let mut claims: Vec<Claim> = Vec::new();
        let mut claim_of: Vec<Option<usize>> = vec![None; vehicles.len()];
        for (i, (v, p)) in vehicles.iter().zip(&perceptions).enumerate() {
            let Some(j) = p.junction else { continue };
            let st = &v.state;
            let decide = (st.speed * st.speed / (2.0 * st.decel.max(0.5)) + 12.0).clamp(20.0, 70.0);
            if !j.inside && j.entry - st.front > decide {
                continue;
            }
            // committed: inside, or too close to the line to stop before it even braking
            // hard (a driver who can still stop, if not comfortably, gives way as the
            // rules say: two cars arriving together at an unmarked junction both took
            // themselves for committed and met in the middle)
            let room = j.entry - st.front - 0.6;
            let cannot_stop = st.speed > 1.0 && room < st.speed * st.speed / (2.0 * omsi_sim::traffic::MAX_BRAKE * 0.7);
            let committed = j.inside || cannot_stop;
            let limit = net.lanes.get(st.lane).map(|l| l.speed_limit_kmh / 3.6 * st.desire).unwrap_or(13.0);
            claim_of[i] = Some(claims.len());
            claims.push(Claim {
                vehicle: i,
                junction: j.junction,
                movement: j.movement,
                s: -j.entry,
                speed: st.speed,
                accel: st.accel,
                v_max: limit.min(st.max_speed_kmh / 3.6),
                front: st.front,
                rear: st.rear,
                gap: v.impatience.gap(st.accept_gap),
                committed,
                exit_room: j.exit_room,
                impatience: v.impatience,
                absolute: v.is_rail() || v.external.priority,
            });
        }
        let grants: Vec<Grant> = reservation::resolve(&self.junctions, net, &claims);

        // plan and act, one vehicle at a time
        for i in 0..vehicles.len() {
            let grant = claim_of[i].map(|k| grants[k]);
            let plan = plan::plan(net, self, vehicles, i, &perceptions[i], grant);
            self.act(net, vehicles, i, plan, dt);
        }
        self.removed.sort_unstable_by(|a, b| b.cmp(a));
    }

    fn act(&mut self, net: &Network, vehicles: &mut [Vehicle], i: usize, plan: Plan, dt: f32) {
        let v = &mut vehicles[i];
        let now = self.time;
        v.last = Decision { why: plan.why, lead: plan.lead, lead_seen: plan.lead_seen, stop: plan.stop, held: plan.held, hold: plan.hold, wait_for: plan.wait_for, body_block: plan.body_block };
        v.lead_car = match plan.lead_seen {
            Some(Seen::Vehicle(j)) | Some(Seen::Body(j)) => Some(j),
            _ => None,
        };
        if plan.why == Why::Light && v.state.speed < 0.5 {
            self.held_at_red += 1;
        }
        if let Some(lat) = plan.lateral {
            v.state.lateral_target = lat;
        }
        match plan.start {
            Some(Start::Change { to, dir }) => {
                v.state.start_change(net, to, dir);
                v.change_streak = v.change_streak.saturating_add(1);
            }
            Some(Start::Bypass { to, dir }) => {
                v.state.start_bypass(net, to, dir);
                v.stopped = 0.0;
            }
            Some(Start::RouteChange { to, dir }) => v.state.start_route_change(net, to, dir),
            Some(Start::Pass(p)) => {
                v.passing = Some(p);
                v.state.lateral_curve = p.curve(v.state.odometer, v.state.lateral);
                v.state.lateral_target = 0.0;
                v.stopped = 0.0;
            }
            Some(Start::AbortPass { hold }) => {
                if let Some(p) = v.passing.as_mut() {
                    p.aborted = true;
                    p.hold = hold;
                }
                // back in over what room there is, at least two metres
                let odo = v.state.odometer;
                let back = (hold - odo - 0.5).clamp(2.0, 8.0);
                v.state.lateral_curve = vec![(odo, v.state.lateral), (odo + back, 0.0)];
                v.state.lateral_target = 0.0;
            }
            None => {}
        }
        v.state.accel_cap = plan.accel_cap;
        // a fresh car: no harder braking than its driver likes at the start
        if v.fresh > 0.0 {
            let mut n = 0;
            while n < 16 && v.state.desired_accel(net, plan.lead, plan.stop) < -v.state.decel {
                v.state.speed *= 0.8;
                n += 1;
            }
        }
        let alive = v.state.drive(net, dt, plan.lead, plan.stop);
        // timers
        v.fresh = (v.fresh - dt).max(0.0);
        v.pull_out = (v.pull_out - dt).max(0.0);
        let standing = v.state.speed < 0.3;
        if standing && !v.external.at_stop {
            v.stopped += dt;
        } else if !standing {
            v.stopped = 0.0;
        }
        if (v.state.odometer - v.progress.0).abs() > 5.0 {
            v.progress = (v.state.odometer, 0.0);
        } else if !v.external.at_stop {
            v.progress.1 += dt;
        }
        let held = standing && plan.held;
        // A driver who is let go but has hardly rolled yet (the start reaction takes a moment,
        // then the car creeps off) keeps the impatience that earned the go. It used to cool
        // the moment the hold lifted for a frame, which took the squeeze away again before
        // the car was committed, so four cars at an equal junction held each other for
        // ever, each in turn fed up for one frame (and then creeping and stopping again).
        if !(v.state.speed < 2.0 && !plan.held) {
            v.impatience.update(&v.params, dt, held);
        }
        if v.stopped > 2.0 {
            v.amber = false;
        }
        // a pass runs its course: the bent way brings the car back by itself
        if let Some(p) = v.passing {
            if p.aborted {
                if v.state.speed < 0.1 && (v.state.odometer >= p.hold - 0.3 || v.stopped > 1.0) {
                    v.passing = None;
                    v.state.lateral_curve.clear();
                }
            } else if v.state.odometer >= p.until + p.back_len {
                v.passing = None;
            } else if v.stopped > 8.0 && v.state.lateral.abs() < p.side.abs() * 0.5 {
                // never got out: give it up where it stands
                v.passing = None;
                let odo = v.state.odometer;
                v.state.lateral_curve = vec![(odo, v.state.lateral), (odo + 4.0, 0.0)];
            }
        }
        // a random car that stands for ever asks to be taken away when nobody looks
        if !v.is_bus() && !v.is_rail() && plan.why != Why::Light {
            let stood = v.stopped.max(v.progress.1);
            if (stood > 60.0 && plan.why != Why::Junction) || stood > 150.0 {
                v.gone = true;
            }
        }
        // indicators: what the car does, debounced so a flicker of the plan does not flash
        v.state.update_blinker(net);
        let want = if v.external.blinker_off { 0 } else { v.state.blinker };
        let (shown, since) = v.blinker_shown;
        let age = (now - since) as f32;
        let allowed = if want == shown {
            true
        } else if want == 0 {
            age >= 0.5
        } else if shown == 0 {
            true
        } else {
            age >= 1.5
        };
        if allowed && want != shown {
            v.blinker_shown = (want, now);
        }
        if v.state.change.is_none() {
            v.change_streak = 0;
        }
        if !alive || v.change_streak >= 12 {
            self.removed.push(i);
        }
    }

    /// The indicator a vehicle shows now (0 off, 1 left, 2 right).
    pub fn blinker(v: &Vehicle) -> i32 {
        v.blinker_shown.0
    }

    /// Light `li` of controller `ctl`: (state code, seconds the state lasts).
    pub fn light_state(&self, ctl: usize, li: usize) -> Option<(i32, f32)> {
        let c = self.lights.get(ctl)?;
        Some((c.state(li), c.remaining(li)))
    }

    pub fn light_until_go(&self, ctl: usize, li: usize) -> Option<f32> {
        self.lights.get(ctl)?.time_until_go(li)
    }
}
