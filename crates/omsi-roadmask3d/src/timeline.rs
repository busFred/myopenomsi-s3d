//! The recording replayed once into what the overlay draws: the window's lanes with what
//! they are to the simulation (class, stop lines, kerb spots, junction pieces), and a
//! snapshot per recorded frame of the vehicles (their recorded box, state and reason), the
//! people (their decision) and the player's bus.

use crate::geometry::{LaneClass, PersonState, VehicleState};
use glam::DVec3;
use omsi_sim::traffic::{LaneKind, Network};
use omsi_traffic::record::{PersonRecord, Record};
use omsi_traffic::replay::Replay;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// One lane of the window.
pub struct LaneInfo {
    pub index: usize,
    pub class: LaneClass,
    pub width: f32,
    pub points: Vec<DVec3>,
    pub junction_piece: bool,
    /// A stop line across the lane's start (street lanes with a light or at a junction's
    /// entry): the start point, the heading there (degrees).
    pub stop_line: Option<(DVec3, f32)>,
}

#[derive(Clone)]
pub struct VehicleSnap {
    pub id: u64,
    pub pos: DVec3,
    pub heading: f32,
    pub front: f32,
    pub rear: f32,
    pub half_width: f32,
    pub class: String,
    pub type_name: String,
    pub speed: f32,
    pub lane: usize,
    pub s: f32,
    pub why: &'static str,
    pub state: VehicleState,
    pub lead_id: Option<u64>,
    pub wait_for: Option<u64>,
    pub stop_in: Option<f32>,
    /// The lanes it means to go on into, in order.
    pub ahead: Vec<usize>,
    pub standing_for: f32,
}

#[derive(Clone)]
pub struct PersonSnap {
    pub pos: DVec3,
    pub state: PersonState,
    pub decision: String,
    pub at_kerb: bool,
    pub on_crossing: bool,
    pub activity: String,
}

#[derive(Clone)]
pub struct PlayerSnap {
    pub pos: DVec3,
    pub heading: f32,
    pub half_len: f32,
    pub half_width: f32,
    pub speed: f32,
}

#[derive(Clone)]
pub struct FrameSnap {
    pub day_time: f64,
    pub vehicles: Vec<VehicleSnap>,
    pub people: Vec<PersonSnap>,
    pub players: Vec<PlayerSnap>,
}

pub struct Timeline {
    pub map: String,
    pub lanes: Vec<LaneInfo>,
    /// The ends of the crossings' chains of pieces: where a pedestrian steps off the kerb.
    pub kerb_spots: Vec<DVec3>,
    pub frames: Vec<FrameSnap>,
    /// The window's middle on the ground (the player's place at its middle, else the lanes').
    pub centre: DVec3,
    /// The lanes' extent (min, max) in world metres.
    pub extent: (DVec3, DVec3),
    pub net: Network,
}

impl Timeline {
    /// Replay `path` whole (a cut window) and keep what the overlay draws within `radius`
    /// metres of its middle.
    pub fn load(path: &Path, radius: f64, centre_override: Option<(f64, f64)>) -> Result<Timeline, Box<dyn std::error::Error>> {
        let mut r = Replay::load(path)?;
        if !r.at(0) {
            return Err("the recording has no keyframe".into());
        }
        let mid = window_mid(path, &r);
        let (cx, cy) = match centre_override {
            Some(c) => c,
            None => {
                let p = player_at(&r, mid).ok_or("no player in the recording: give --center x,y")?;
                (p.x, p.y)
            }
        };
        let near = |p: DVec3, rad: f64| ((p.x - cx).powi(2) + (p.y - cy).powi(2)).sqrt() <= rad;

        let mut frames = Vec::new();
        while r.step() {
            r.sim.update_positions(&r.net, &mut r.vehicles);
            let vehicles = r
                .vehicles
                .iter()
                .filter(|v| !v.gone && near(v.pos, radius + 20.0))
                .map(|v| {
                    let standing = v.state.speed.abs() < 0.05;
                    let why = v.last.why.name();
                    let mut ahead = Vec::new();
                    if let Some(n) = v.state.planned_next {
                        ahead.push(n);
                    }
                    for &l in &v.state.ahead {
                        if !ahead.contains(&l) {
                            ahead.push(l);
                        }
                    }
                    VehicleSnap {
                        id: v.id,
                        pos: v.pos,
                        heading: v.heading,
                        front: v.state.front,
                        rear: v.state.rear,
                        half_width: v.half_width,
                        class: format!("{:?}", v.class),
                        type_name: r.types.get(&v.id).cloned().unwrap_or_default(),
                        speed: v.state.speed,
                        lane: v.state.lane,
                        s: v.state.s,
                        why,
                        state: VehicleState::of(why, standing),
                        lead_id: v.last.lead_id.or(v.lead_id),
                        wait_for: v.last.wait_for,
                        stop_in: v.last.stop,
                        ahead,
                        standing_for: v.stopped,
                    }
                })
                .collect();
            let people = frame_people(&r)
                .iter()
                .filter(|p| !p.state.contains("aboard") && near(p.pos, radius + 5.0))
                .map(|p| PersonSnap {
                    pos: p.pos,
                    state: PersonState::of(p.decision.as_deref()),
                    decision: p.decision.as_deref().unwrap_or("").to_string(),
                    at_kerb: p.at_kerb,
                    on_crossing: p.on_crossing,
                    activity: p.state.to_string(),
                })
                .collect();
            let players = r
                .sim
                .players
                .iter()
                .map(|p| PlayerSnap { pos: p.pos, heading: p.heading, half_len: p.half_len, half_width: p.half_width, speed: p.speed })
                .collect();
            frames.push(FrameSnap { day_time: r.sim.day_time, vehicles, people, players });
        }
        if frames.is_empty() {
            return Err("the recording has no frames".into());
        }

        // the lanes of the window and what they are
        let net = &r.net;
        let junctions = &r.sim.junctions;
        let crossing_pieces: HashSet<usize> = junctions.of_foot_lane.keys().copied().collect();
        let mut lanes = Vec::new();
        let mut lo = DVec3::splat(f64::INFINITY);
        let mut hi = DVec3::splat(f64::NEG_INFINITY);
        for (l, lane) in net.lanes.iter().enumerate() {
            if lane.points.len() < 2 || !lane.points.iter().any(|&p| near(p, radius + 30.0)) {
                continue;
            }
            if lane.kind == LaneKind::Air {
                continue;
            }
            for p in &lane.points {
                lo = lo.min(*p);
                hi = hi.max(*p);
            }
            let class = LaneClass::of(lane.kind, lane.reversed, lane.no_cars, lane.density, lane.traffic_light.is_some(), lane.length(), crossing_pieces.contains(&l));
            let junction_piece = junctions.of_lane.contains_key(&l);
            let movement_start = junctions
                .of_lane
                .get(&l)
                .map(|&(j, m)| junctions.nodes[j].movements[m].lane == l && net.prev.get(l).map(|p| p.iter().all(|q| !junctions.of_lane.contains_key(q))).unwrap_or(true))
                .unwrap_or(false);
            let stop_line = if lane.kind == LaneKind::Street && (lane.traffic_light.is_some() || movement_start) {
                let (p, h) = lane.at(0.0);
                Some((p, h))
            } else {
                None
            };
            lanes.push(LaneInfo {
                index: l,
                class,
                width: lane.width,
                points: lane.points.clone(),
                junction_piece,
                stop_line,
            });
        }
        // the kerbs: the ends of a crossing's chain of pieces
        let mut kerb_spots = Vec::new();
        for &l in &crossing_pieces {
            let Some(lane) = net.lanes.get(l) else { continue };
            if lane.points.len() < 2 || !lane.points.iter().any(|&p| near(p, radius + 30.0)) {
                continue;
            }
            let prev_outside = net.prev.get(l).map(|p| p.is_empty() || p.iter().any(|q| !crossing_pieces.contains(q))).unwrap_or(true);
            let next_outside = lane.next.is_empty() || lane.next.iter().any(|q| !crossing_pieces.contains(q));
            if prev_outside {
                kerb_spots.push(lane.points[0]);
            }
            if next_outside {
                kerb_spots.push(*lane.points.last().unwrap());
            }
        }
        if !lo.x.is_finite() {
            lo = DVec3::new(cx - radius, cy - radius, 0.0);
            hi = DVec3::new(cx + radius, cy + radius, 0.0);
        }
        let z = frames.iter().flat_map(|f| f.players.iter().map(|p| p.pos.z)).next().unwrap_or((lo.z + hi.z) * 0.5);
        Ok(Timeline { map: r.header.map.clone(), lanes, kerb_spots, frames, centre: DVec3::new(cx, cy, z), extent: (lo, hi), net: r.net })
    }

    /// The index of the frame at or just before day time `t` (the first for an earlier t).
    pub fn frame_at(&self, t: f64) -> usize {
        match self.frames.binary_search_by(|f| f.day_time.partial_cmp(&t).unwrap_or(std::cmp::Ordering::Equal)) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1).min(self.frames.len() - 1),
        }
    }

    pub fn first_time(&self) -> f64 {
        self.frames[0].day_time
    }

    pub fn last_time(&self) -> f64 {
        self.frames[self.frames.len() - 1].day_time
    }

    /// The way ahead of a vehicle on its lane at `s`: the lane's points from there and on
    /// into the lanes it means to take, `dist` metres long.
    pub fn way_ahead(&self, lane: usize, s: f32, ahead: &[usize], dist: f32) -> Vec<DVec3> {
        let mut out = Vec::new();
        let mut left = dist;
        let mut lane_ix = Some(lane);
        let mut from = s;
        let mut chain = ahead.iter().copied();
        while let Some(l) = lane_ix {
            let Some(ln) = self.net.lanes.get(l) else { break };
            let len = ln.length();
            let mut at = from.max(0.0);
            while at <= len && left > 0.0 {
                out.push(ln.at(at).0);
                let step = 2.0f32.min(left);
                at += step;
                left -= step;
            }
            if left <= 0.0 {
                break;
            }
            out.push(ln.at(len).0);
            lane_ix = chain.next().or_else(|| ln.next.first().copied());
            from = 0.0;
            if out.len() > 400 {
                break;
            }
        }
        out
    }

    /// The vehicles seen anywhere in the window, by their first frame.
    pub fn vehicle_ids(&self) -> Vec<u64> {
        let mut seen = HashMap::new();
        for (i, f) in self.frames.iter().enumerate() {
            for v in &f.vehicles {
                seen.entry(v.id).or_insert(i);
            }
        }
        let mut ids: Vec<(u64, usize)> = seen.into_iter().collect();
        ids.sort_by_key(|x| (x.1, x.0));
        ids.into_iter().map(|x| x.0).collect()
    }
}

/// The window's middle (day time, s): the HHMM of a `caseNN-HHMM.rec` name, else the
/// middle of the recorded frames.
pub fn window_mid(path: &Path, r: &Replay) -> f64 {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    if let Some(hhmm) = stem.split('-').nth(1) {
        if hhmm.len() == 4 {
            if let (Ok(h), Ok(m)) = (hhmm[..2].parse::<f64>(), hhmm[2..].parse::<f64>()) {
                return h * 3600.0 + m * 60.0;
            }
        }
    }
    let days: Vec<f64> = r.records.iter().filter_map(|r| if let Record::Frame(f) = r { Some(f.day_time) } else { None }).collect();
    (days.first().copied().unwrap_or(0.0) + days.last().copied().unwrap_or(0.0)) * 0.5
}

/// Where the first player was nearest day time `t`.
fn player_at(r: &Replay, t: f64) -> Option<DVec3> {
    let mut best: Option<(f64, DVec3)> = None;
    for rec in &r.records {
        if let Record::Frame(f) = rec {
            if let Some(p) = f.players.first() {
                let d = (f.day_time - t).abs();
                if best.map(|b| d < b.0).unwrap_or(true) {
                    best = Some((d, p.pos));
                }
            }
        }
    }
    best.map(|b| b.1)
}

/// The people of the frame just replayed.
fn frame_people(r: &Replay) -> &[PersonRecord] {
    match r.cursor.checked_sub(1).and_then(|i| r.records.get(i)) {
        Some(Record::Frame(f)) => &f.people,
        _ => &[],
    }
}
