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
use omsi_sim::traffic::{AiState, LaneKind, Lead, Network, TrafficLightController};

/// A player's (or another LAN player's) vehicle as the traffic sees it.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Parked {
    pub s: f32,
    /// Lateral offset of its middle from the lane (m, + right).
    pub lateral: f32,
    pub half_width: f32,
    pub half_len: f32,
}

/// Overtaking on the oncoming lane, or squeezing past something inside the lane.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
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

/// The last decision, kept for consumers and the debug overlay. Other vehicles are named
/// by id: the caller may take vehicles off the road between frames.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct Decision {
    pub why: Why,
    pub lead: Option<Lead>,
    pub lead_seen: Option<Seen>,
    /// The vehicle followed (its id).
    pub lead_id: Option<u64>,
    pub stop: Option<f32>,
    pub held: bool,
    pub hold: Option<Hold>,
    /// Vehicle id waited for at a junction (a player: `PLAYER_ID_BASE` + its id).
    pub wait_for: Option<u64>,
    /// Another vehicle's body (not on the way) that held this one (its id).
    pub body_block: Option<u64>,
}

/// Ids of the players in `Decision::wait_for`, above any vehicle's.
pub const PLAYER_ID_BASE: u64 = 1 << 62;

/// One AI vehicle in the traffic: its way (`state`), its driver and its manoeuvres.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    pub amber: Option<(usize, usize)>,
    pub external: External,
    pub last: Decision,
    /// Id of the vehicle followed last frame.
    pub lead_id: Option<u64>,
    /// (vehicle id, time until) a lead is ignored to break a mutual wait.
    pub ignore_lead: Option<(u64, f64)>,
    /// Indicator shown, and since when (debounced).
    pub blinker_shown: (i32, f64),
    /// World position of the origin and heading (degrees), from the way.
    pub pos: DVec3,
    pub heading: f32,
    /// Standing as of its last claim at a junction, with hysteresis:
    /// from under 0.3 m/s until over 1 m/s (`Claim::standing`). Kept with the vehicle, so a
    /// keyframe restores it (recordings before format 2 lack it: false).
    #[serde(default)]
    pub claim_standing: bool,
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
            amber: None,
            external: External::default(),
            last: Decision::default(),
            lead_id: None,
            ignore_lead: None,
            blinker_shown: (0, 0.0),
            pos: DVec3::ZERO,
            heading: 0.0,
            claim_standing: false,
        }
    }

    /// An empty vehicle to stand in while one is lent out.
    pub fn placeholder() -> Vehicle {
        Vehicle::new(0, VehicleClass::Car, AiState::new(0, 0.0, 0), 0.0, 0)
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

/// Grid cell size of the vehicle and people grids (m).
const GRID: f64 = 50.0;

fn cell(p: DVec3) -> (i32, i32) {
    ((p.x / GRID).floor() as i32, (p.y / GRID).floor() as i32)
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
    /// The players' vehicles on the lanes, as of this frame: per lane, (player index,
    /// distance along it, lateral offset). The lanes beside, the room past a junction and
    /// the oncoming lane see them as they see any vehicle.
    pub player_occupancy: HashMap<usize, Vec<(usize, f32, f32)>>,
    /// Vehicles and people by 50 m cell, as of this frame.
    vehicle_grid: HashMap<(i32, i32), Vec<usize>>,
    people_grid: HashMap<(i32, i32), Vec<usize>>,
    /// Lane changes begun this frame: (lane changed onto, distance along it). A change
    /// into the same stretch later in the same frame is refused (`act`): the perceptions
    /// were taken before anybody moved, so two cars would move into one gap at once.
    pub changes_started: Vec<(usize, f32)>,
    pub left_hand: bool,
    /// Vehicles that reached a dead end this frame (indices, descending).
    pub removed: Vec<usize>,
    /// Vehicles counted standing at a red light this frame (diagnostics).
    pub held_at_red: usize,
    /// What the junction grants read besides the claims: the pairs of lights that are
    /// never green together, filled in as claimants meet.
    pub signal_groups: reservation::Context,
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
            player_occupancy: HashMap::new(),
            vehicle_grid: HashMap::new(),
            people_grid: HashMap::new(),
            changes_started: Vec::new(),
            left_hand: net.left_hand,
            removed: Vec::new(),
            held_at_red: 0,
            signal_groups: reservation::Context::default(),
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

    /// Indices of the vehicles within `r` metres of `p` (by grid cell: a few more).
    pub fn vehicles_near(&self, p: DVec3, r: f64) -> Vec<usize> {
        let (cx, cy) = cell(p);
        let n = (r / GRID).ceil() as i32;
        let mut out = Vec::new();
        for dx in -n..=n {
            for dy in -n..=n {
                if let Some(v) = self.vehicle_grid.get(&(cx + dx, cy + dy)) {
                    out.extend_from_slice(v);
                }
            }
        }
        out
    }

    /// Indices of the people within `r` metres of `p` (by grid cell: a few more).
    pub fn people_near(&self, p: DVec3, r: f64) -> Vec<usize> {
        let (cx, cy) = cell(p);
        let n = (r / GRID).ceil() as i32;
        let mut out = Vec::new();
        for dx in -n..=n {
            for dy in -n..=n {
                if let Some(v) = self.people_grid.get(&(cx + dx, cy + dy)) {
                    out.extend_from_slice(v);
                }
            }
        }
        out
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

    /// Which vehicle is where on which lane, and the grids.
    pub fn build_occupancy(&mut self, net: &Network, vehicles: &[Vehicle]) {
        self.occupancy.clear();
        self.vehicle_grid.clear();
        self.people_grid.clear();
        for (i, v) in vehicles.iter().enumerate() {
            let st = &v.state;
            if net.lanes.get(st.lane).is_none() {
                continue;
            }
            self.vehicle_grid.entry(cell(v.pos)).or_default().push(i);
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
        for (k, p) in self.people.iter().enumerate() {
            self.people_grid.entry(cell(p.pos)).or_default().push(k);
        }
        self.player_occupancy.clear();
        for k in 0..self.players.len() {
            let p = self.players[k];
            let Some((lane, s)) = self.player_lane(net, &p) else { continue };
            let (q, h) = net.lanes[lane].at(s.min(net.lanes[lane].length()));
            let hr = (h as f64).to_radians();
            let right = DVec2::new(hr.cos(), -hr.sin());
            let lateral = (p.pos - q).truncate().dot(right) as f32;
            self.player_occupancy.entry(lane).or_default().push((k, s, lateral));
        }
    }

    /// The light that governs footpath lane `lane`: its own, else that of the pedestrian
    /// crossing it is a piece of.
    pub fn crossing_light(&self, net: &Network, lane: usize) -> Option<(usize, usize)> {
        self.junctions.crossing_light(net, lane)
    }

    /// Lanes starting just ahead of a player, and the ones after them, within `within`
    /// metres: what the player's vehicle will drive along.
    fn lanes_ahead_of(&self, net: &Network, p: &PlayerView, within: f32) -> Vec<(usize, f32)> {
        let f = p.fwd();
        let mut out: Vec<(usize, f32)> = Vec::new();
        for l in net.lanes_starting_near(p.pos, 25.0) {
            let lane = &net.lanes[l];
            if lane.kind != LaneKind::Street {
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

    /// The lane a player's vehicle is on: (lane, distance along it), a street lane within
    /// 6 m heading the same way.
    fn player_lane(&self, net: &Network, p: &PlayerView) -> Option<(usize, f32)> {
        let (l, s, _) = net.lane_along(p.pos, p.heading as f64, LaneKind::Street, 6.0, 45.0)?;
        Some((l, s))
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
        // a walker presses the button of a lit pedestrian lane they stand on or at the end
        // of (the lanes before and after theirs too: a crossing's lit stub is short)
        let press = |lights: &mut Vec<TrafficLightController>, ctl: usize, li: usize, dist: f32| {
            if let Some(c) = lights.get_mut(ctl) {
                if dist <= c.approach_dist(li).min(10.0) {
                    if let Some(r) = c.request.get_mut(li) {
                        *r = true;
                    }
                }
            }
        };
        for &(l, s) in &self.walkers {
            let Some(lane) = net.lanes.get(l) else { continue };
            let to_end = (lane.length() - s).max(0.0);
            if let Some((ctl, li)) = lane.traffic_light {
                press(&mut self.lights, ctl, li, to_end.min(s.max(0.0)));
            }
            for &n in &lane.next {
                if let Some((ctl, li)) = net.lanes[n].traffic_light {
                    press(&mut self.lights, ctl, li, to_end);
                }
            }
            for &p in net.prev.get(l).map(|v| v.as_slice()).unwrap_or(&[]) {
                if let Some((ctl, li)) = net.lanes[p].traffic_light {
                    press(&mut self.lights, ctl, li, s.max(0.0));
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
        self.changes_started.clear();
        self.held_at_red = 0;
        self.update_positions(net, vehicles);
        self.build_occupancy(net, vehicles);
        self.update_lights(net, vehicles, dt);

        // perception: read-only over everything
        let perceptions: Vec<Perception> = (0..vehicles.len()).map(|i| perception::perceive(net, self, vehicles, i)).collect();

        // claims and grants at the junctions
        let mut claims: Vec<Claim> = Vec::new();
        let mut claim_of: Vec<Option<usize>> = vec![None; vehicles.len()];
        // the claimants' standing hysteresis as of this claim, written
        // back once all claims are filed
        let mut standing_now: Vec<(usize, bool)> = Vec::new();
        for (i, (v, p)) in vehicles.iter().zip(&perceptions).enumerate() {
            let Some(j) = p.junction.as_ref() else { continue };
            let st = &v.state;
            // a claim is filed from eight seconds before the meeting place (at least 60 m,
            // at most 220 m): the drivers who give way to this one see it coming in time
            // to stop, not only once it is a braking distance from its own line
            let horizon = (8.0 * st.speed + 30.0).clamp(60.0, 220.0);
            if !j.inside && j.conflict_start - st.front > horizon {
                continue;
            }
            // committed: past the first meeting place of the way through, or too close to
            // it to stop before it even braking hard. A driver who can still stop, if not
            // comfortably, gives way as the rules say (two cars arriving together at an
            // unmarked junction both took themselves for committed and met in the middle);
            // one on the lead-in piece of a long object is inside but not yet committed.
            // (at the line a driver stops hard if it must; one already inside does not
            // slam on in the middle of the junction, it goes on unless it can stop gently)
            let room = j.conflict_start - st.front - 0.6;
            let brake = if j.inside { st.decel * 1.5 } else { omsi_sim::traffic::MAX_BRAKE * 0.7 };
            let cannot_stop = st.speed > 1.0 && room < st.speed * st.speed / (2.0 * brake);
            let committed = j.conflict_start - st.front <= 0.5 || cannot_stop;
            let limit = net.lanes.get(st.lane).map(|l| l.speed_limit_kmh / 3.6 * st.desire).unwrap_or(13.0);
            // the light at this junction's line, if the driver is held by one: any light on
            // the way before the first meeting place with another way, wherever the object
            // begins (`JunctionView::light_line`: the old test, the light within 1 m of the
            // object's entry, never saw a car standing at BeijingCity's lit stub as held by it)
            let light_line = j.light_line;
            let (light_red, until_go) = match p.signal {
                Some(sig) if sig.dist <= light_line + 1.0 && !matches!(sig.aspect, omsi_sim::traffic::Aspect::Green | omsi_sim::traffic::Aspect::GreenYellow | omsi_sim::traffic::Aspect::Dark) => {
                    // not held by a light it goes through: the plan's own
                    // call (too late to stop, the yellow it takes), or the mark of one
                    // that took the line on green or yellow and is close to it (queued
                    // there, it goes on as soon as its leader moves, whatever the light)
                    let call = plan::light_call(v, p, &sig);
                    let through = call.go || call.marked_through;
                    if through {
                        (false, None)
                    } else {
                        (true, sig.until_go)
                    }
                }
                _ => (false, None),
            };
            // standing at its red short of the first meeting place with
            // another way, a car is committed to nothing: the merge of the lead-in with its
            // own stub put its "first meeting place" behind its bumper, and the foes took
            // the car waiting at its red for one already on its way across
            let committed = committed && !(light_red && st.speed < 0.5 && light_line - st.front > 0.5);
            // standing, with hysteresis (`Claim::standing`):
            // from under 0.3 m/s until over 1 m/s, remembered from one claim to the next
            let was_standing = v.claim_standing;
            standing_now.push((i, st.speed < 0.3 || (was_standing && st.speed <= 1.0)));
            claim_of[i] = Some(claims.len());
            claims.push(Claim {
                chain: j.chain.clone(),
                was_standing,
                light_red,
                until_go,
                lead: match p.leader.map(|l| l.what) {
                    Some(Seen::Vehicle(o)) | Some(Seen::Body(o)) => Some(o),
                    _ => None,
                },
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
                held: st.speed < 0.3 && v.last.held,
                // (held by its lead, people, a zebra, a light, ... rather than by a junction's
                // answer or a full exit)
                blocked: st.speed < 0.3 && v.last.held && !matches!(v.last.why, Why::None | Why::Junction | Why::ExitFull),
                // (people or a zebra in front of it, or a lead that stands too: nothing that
                // lets it go on a schedule, as a light or a lead moving off does)
                stuck: st.speed < 0.3
                    && v.last.held
                    && match v.last.why {
                        Why::People | Why::Zebra => true,
                        Why::Lead => p.leader.is_some_and(|l| l.speed < 0.3 && l.gap < 12.0),
                        _ => false,
                    },
                red_hysteresis: was_standing,
                lanes: perception::way_lanes(net, st, 150.0),
                light: j.light.map(|(c, l, _)| (c, l)),
                id: v.id,
            });
        }
        // the players' vehicles claim their way like anybody else (their grants are not
        // read: the player drives): the junction on the lane under the vehicle, or on the
        // lanes ahead of it within the deciding distance
        for (k, p) in self.players.iter().enumerate() {
            // the way through chosen by the indicator, else the straightest
            let pick = |entries: &Vec<(usize, usize)>| {
                entries.iter().copied().min_by_key(|&(j, m)| {
                    let turn = net.lanes[self.junctions.nodes[j].movements[m].lane].turn;
                    let want = p.blinker as i32;
                    if want != 0 && turn == want {
                        0
                    } else if turn == 0 {
                        1
                    } else {
                        2
                    }
                })
            };
            // candidate lanes: (lane, distance from the vehicle's origin to the lane's start)
            let mut lanes: Vec<(usize, f32)> = Vec::new();
            if let Some((lane, s)) = self.player_lane(net, p) {
                lanes.push((lane, -s));
                let mut d = net.lanes[lane].length() - s;
                let mut cur = lane;
                for _ in 0..6 {
                    if d > 80.0 {
                        break;
                    }
                    let Some(&n) = net.lanes[cur].next.first() else { break };
                    lanes.push((n, d));
                    d += net.lanes[n].length();
                    cur = n;
                }
            }
            lanes.extend(self.lanes_ahead_of(net, p, 80.0).into_iter().map(|(l, d)| (l, d - p.half_len)));
            let mut found: Option<(usize, usize, f32)> = None;
            for (lane, d0) in lanes {
                let len = net.lanes[lane].length();
                let hit = if let Some(&(j, m)) = self.junctions.of_lane.get(&lane) {
                    Some((j, m, -d0))
                } else if let Some(entries) = self.junctions.entries.get(&lane) {
                    pick(entries).map(|(j, m)| (j, m, -(d0 + len)))
                } else {
                    None
                };
                if let Some(h) = hit {
                    if found.map(|f| h.2 > f.2).unwrap_or(true) {
                        found = Some(h);
                    }
                }
            }
            let Some((j, m, s_on)) = found else { continue };
            if s_on < -70.0 {
                continue;
            }
            let v = p.speed.max(0.0);
            let cs = self.junctions.nodes[j].movements[m].conflict_start;
            let cs = if cs.is_finite() { cs } else { net.lanes[self.junctions.nodes[j].movements[m].lane].length() };
            let room = cs - s_on - p.half_len - 0.6;
            let cannot_stop = v > 1.0 && room < v * v / (2.0 * omsi_sim::traffic::MAX_BRAKE * 0.7);
            claims.push(Claim {
                chain: vec![(m, 0.0)],
                was_standing: false,
                light_red: false,
                until_go: None,
                lead: None,
                vehicle: usize::MAX - k,
                junction: j,
                movement: m,
                s: s_on,
                speed: v,
                accel: 1.2,
                v_max: 14.0,
                front: p.half_len,
                rear: p.half_len,
                gap: 4.0,
                committed: cs - s_on - p.half_len <= 0.5 || cannot_stop,
                exit_room: true,
                impatience: Impatience::default(),
                absolute: p.priority,
                held: p.still > 2.0,
                blocked: false,
                stuck: false,
                red_hysteresis: false,
                lanes: Vec::new(),
                light: None,
                id: PLAYER_ID_BASE + p.id as u64,
            });
        }
        for (i, standing) in standing_now {
            vehicles[i].claim_standing = standing;
        }
        // the signal groups the claimants' lights belong to: which pairs never let vehicles
        // in together (worked out once per pair of lights, from the programs)
        for a in 0..claims.len() {
            let Some((ctl, la)) = claims[a].light else { continue };
            for b in a + 1..claims.len() {
                let Some((ctl_b, lb)) = claims[b].light else { continue };
                if ctl_b != ctl || la == lb || claims[b].junction != claims[a].junction {
                    continue;
                }
                let key = (ctl, la.min(lb), la.max(lb));
                if !self.signal_groups.exclusive.contains_key(&key) {
                    let ex = self.lights.get(ctl).is_some_and(|c| c.never_together(la, lb));
                    self.signal_groups.exclusive.insert(key, ex);
                }
            }
        }
        let grants: Vec<Grant> = reservation::resolve_with(&self.junctions, net, &claims, &self.signal_groups);
        // who went first on a tie at a zipper merge: the next tie there is the other side's
        for (g, c) in grants.iter().zip(&claims) {
            if let (true, Some((key, piece))) = (g.go, g.zip) {
                self.signal_groups.zipper.insert(key, (piece, c.id));
            }
        }

        // plan and act, one vehicle at a time
        for i in 0..vehicles.len() {
            let grant = claim_of[i].map(|k| grants[k]);
            let plan = plan::plan(net, self, vehicles, i, &perceptions[i], grant);
            self.act(net, vehicles, i, plan, dt);
        }
        self.removed.sort_unstable_by(|a, b| b.cmp(a));
    }

    fn act(&mut self, net: &Network, vehicles: &mut [Vehicle], i: usize, plan: Plan, dt: f32) {
        let now = self.time;
        let wait_for = plan.wait_for.map(|k| if k >= usize::MAX - 64 { PLAYER_ID_BASE + self.players.get(usize::MAX - k).map(|p| p.id as u64).unwrap_or(0) } else { vehicles.get(k).map(|o| o.id).unwrap_or(0) });
        let lead_id = match plan.lead_seen {
            Some(Seen::Vehicle(j)) | Some(Seen::Body(j)) => vehicles.get(j).map(|o| o.id),
            _ => None,
        };
        let v = &mut vehicles[i];
        v.last = Decision { why: plan.why, lead: plan.lead, lead_seen: plan.lead_seen, lead_id, stop: plan.stop, held: plan.held, hold: plan.hold, wait_for, body_block: plan.body_block };
        v.lead_id = lead_id;
        if plan.why == Why::Light && v.state.speed < 0.5 {
            self.held_at_red += 1;
        }
        if let Some(light) = plan.amber {
            v.amber = Some(light);
        }
        if let Some(lat) = plan.lateral {
            v.state.lateral_target = lat;
        }
        // a lane change begins only when nobody started one into the same stretch this
        // frame (the perception was taken before anybody moved)
        let mut change_into = |to: usize, s_to: f32| -> bool {
            if self.changes_started.iter().any(|&(l, s)| l == to && (s - s_to).abs() < 25.0) {
                return false;
            }
            self.changes_started.push((to, s_to));
            true
        };
        match plan.start {
            Some(Start::Change { to, dir }) => {
                if change_into(to, net.beside_s(v.state.lane, to, v.state.s)) {
                    v.state.start_change(net, to, dir);
                }
            }
            Some(Start::Bypass { to, dir }) => {
                if change_into(to, net.beside_s(v.state.lane, to, v.state.s)) {
                    v.state.start_bypass(net, to, dir);
                    v.stopped = 0.0;
                }
            }
            Some(Start::RouteChange { to, dir }) => {
                if change_into(to, net.beside_s(v.state.lane, to, v.state.s)) {
                    v.state.start_route_change(net, to, dir);
                }
            }
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
            v.amber = None;
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
            if (stood > 60.0 && !matches!(plan.why, Why::Junction | Why::ExitFull)) || stood > 150.0 {
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
        if !alive {
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
