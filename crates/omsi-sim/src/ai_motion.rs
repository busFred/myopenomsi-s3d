//! How the body of an AI vehicle follows the way its planner lays out
//! (`traffic::AiState::way_point`): a road vehicle is a bicycle model steered by pure
//! pursuit and sitting on its springs, a rail vehicle rides on its two outer axles, an
//! aircraft flies along its path and banks into its turns.
//!
//! Placing the vehicle *on* the lane every frame, turned to the lane's heading, is what made
//! the traffic look wrong: the heading jumped at every lane sample and joint (the steering
//! was worked out from those jumps), the body turned about its middle so that the rear
//! wheels slid sideways through every bend, and a lane change was a sideways glide with a
//! made-up yaw. Here the heading only ever changes by speed × tan(steering) / wheelbase,
//! the rear axle never moves sideways, and the steering itself is rate limited.
//!
//! In a bend of its lane an AI road vehicle drives the careful driver's line
//! (`careful_line_offset`): its rear axle runs inside the way by as much as centres its swept
//! body on it, so the front no longer swings out over the next lane. For an articulated
//! vehicle the body is the whole of it: the rear sections that hang on joints off-track
//! further inside (`rear_off_tracking`), and the band to centre runs from the innermost point
//! of whichever runs innermost.
//! Where that band is wider than the lane the vehicle swings wide: the offset stops where the
//! innermost point reaches the lane's inner edge, and the outer side takes the rest.

use crate::vehicle::VehicleInstance;
use glam::{DVec2, DVec3, Vec3};
use omsi_vehicle::Vehicle;
use std::f32::consts::FRAC_PI_2;

/// What kind of way the vehicle follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionKind {
    Road,
    Rail,
    Air,
}

/// The line the vehicle turns about (`[rot_pnt_long]`, normally the rear axle) and the
/// distance from it to the front axle: (rotation point, wheelbase), both in metres along
/// the body.
pub fn rotation_point(def: &Vehicle) -> (f32, f32) {
    let front = def.axles.iter().map(|a| a.long).fold(f32::MIN, f32::max);
    let rear = def.axles.iter().map(|a| a.long).fold(f32::MAX, f32::min);
    if def.axles.is_empty() {
        return (-1.3, 2.6);
    }
    // a rotation point at (or in front of) the front axle is a file without one
    let rot = if def.axles.len() >= 2 && def.rot_pnt_long < front - 0.5 { def.rot_pnt_long } else { rear };
    (rot, (front - rot).max(0.8))
}

/// How far inside the centre line of a bend of curvature `k` (1/m, signed) the rotation
/// point of a road vehicle runs (m, same sign as `k`): the careful driver's line. `reach`
/// is the distance from the rotation point to the front bumper (the wheelbase and the
/// front overhang), `half_width` half the body's width. With the rotation point on a
/// circle of radius r the body sweeps the band from r - half_width (the inner side abreast
/// of the rear axle) to √((r + half_width)² + reach²) (the outer front corner); centring
/// that band on the bend's radius R gives r = R - reach² / (4 (R + half_width)), so the
/// offset is reach² · k / (4 (1 + half_width · |k|)), and 0 on a straight.
/// Why: a real vehicle off-tracks - in a bend its rear wheels run inside its front ones -
/// and a careful driver holds the front out so that the whole swept path stays in the
/// lane; SUMO keeps a vehicle's lateral extent within its lane and models no swing-out,
/// which is the lane the traffic simulation reasons about (its rigid box on the way).
/// `rear_pull` (`rear_off_tracking`) is the rear sections' extra off-tracking: their inner
/// side lies `rear_pull · |k| / 2` inside the band's inner side, which moves the band's
/// centre by `rear_pull · k / 4`; the offset is that much less (never past 0).
/// Where the band is wider than the lane the bus swings wide: the innermost point of the
/// body (the front section's inner side, or the rear sections' at `rear_pull · |k| / 2`
/// further in) stays on the lane's inner edge, `inner_edge` from the way on the bend's
/// inside, so the offset is at most `inner_edge - half_width - rear_pull · |k| / 2` (never
/// below 0) and the outer side takes all the excess (no rear wheels over the kerb on a right
/// turn, none into the oncoming lane in a left bend).
pub fn careful_line_offset(k: f32, reach: f32, half_width: f32, rear_pull: f32, inner_edge: f32) -> f32 {
    let off = reach * reach * k / (4.0 * (1.0 + half_width * k.abs())) - rear_pull * k / 4.0;
    if off * k < 0.0 {
        return 0.0;
    }
    let cap = (inner_edge - half_width - rear_pull * k.abs() * 0.5).max(0.0);
    off.abs().min(cap).copysign(off)
}

/// The off-tracking of the rear sections hung behind a road vehicle, for `AiBody::
/// keep_swept_in_lane_with`: `parts` are the vehicle's own type and then the types hung on
/// behind it, nearest first. In a steady bend the joint behind a section's rotation point
/// (`[coupling_back]`, `d` behind it) runs on a circle of radius √(r² + d²) when that point
/// is on one of radius r, and the next section's rotation point (`a` behind its
/// `[coupling_front]`) on √(r² + d² - a²): so section i runs inside the front one's rotation
/// circle by about G_i / (2r), G_i the sum of d² - a² over the joints up to it. The result is
/// the most of -G_i over the sections (m², 0 when none runs inside the front section's own
/// inner side).
pub fn rear_off_tracking(parts: &[&crate::vehicle::VehicleType]) -> f32 {
    let (mut g, mut pull) = (0.0f32, 0.0f32);
    for w in parts.windows(2) {
        let (back, front) = crate::vehicle::coupling_points(w[0], false, w[1], false);
        let d = rotation_point(&w[0].def).0 - back.y;
        let a = front.y - rotation_point(&w[1].def).0;
        g += d * d - a * a;
        pull = pull.max(-g);
    }
    pull
}

/// Steps the steering lays the lane's line in from its headings over the look-ahead
/// (`AiBody::drive`): with the midpoint rule the chord of a bend of R 10 over a 12 m look-ahead
/// is off by about a centimetre (the error goes as look³ / (24 R² steps²)).
const LANE_LINE_STEPS: usize = 8;

/// Clearance a vehicle pulling out keeps from the corner of what it goes round (m).
pub const PULL_OUT_CLEARANCE: f64 = 0.25;
/// Seconds a driver standing behind something turns the wheels before moving off round it
/// (the shortest reaction time of the AI drivers).
pub const PULL_OUT_WAIT: f32 = 0.4;
/// Acceleration (m/s²) of a driver edging out round something standing close ahead, until
/// the front is past its corner: pulling away briskly, a car covered too much ground while
/// the wheels were still turning and came within 0.2 m of the corner.
pub const PULL_OUT_ACCEL: f32 = 1.0;

/// Lengths (m) of the S-curve that takes a vehicle out round something standing `real`
/// metres ahead of its front bumper, gentlest first (`front`: origin to front bumper).
/// From a standstill a driver turns out steeply; rolling up to a parked car they move over
/// well before it.
pub fn pull_out_ramps(real: f32, front: f32, rolling: bool) -> Vec<f32> {
    let base = real + 0.5 * front;
    let (factors, max): (&[f32], f32) = if rolling { (&[1.0, 0.8, 0.6, 0.45], 20.0) } else { (&[1.5, 1.2, 1.0, 0.8, 0.6], 12.0) };
    let mut out: Vec<f32> = Vec::new();
    for f in factors {
        let r = (base * f).clamp(4.0, max);
        if !out.iter().any(|&o| (o - r).abs() < 0.3) {
            out.push(r);
        }
    }
    out
}

/// Sideways acceleration (m/s²) a driver allows moving back into the lane after passing.
pub const BACK_IN_LAT_ACCEL: f32 = 2.5;

/// Length (m) of the S-curve that takes a vehicle `offset` metres back into its lane at
/// `speed` (m/s) after passing: a smoothstep of length L over the offset D turns at most
/// 6·D·v²/L², so L = v·√(6·D/a) keeps it to `lat_accel` - about 2.8 s of travel for a lane
/// width. A fixed 8-14 m cut back 3.3 m in under a second at 40 km/h (4-5.6 m/s²).
pub fn back_in_ramp(offset: f32, speed: f32, lat_accel: f32) -> f32 {
    let geometric = (offset.abs() * 3.5).clamp(8.0, 14.0);
    let dynamic = speed.max(0.0) * (6.0 * offset.abs() / lat_accel.max(0.5)).sqrt();
    geometric.max(dynamic).min(60.0)
}

/// A straight way north from the origin that moves `side` metres to the left along an
/// S-curve (smoothstep) `ramp` metres long, as `AiState::way_point` lays out a pull-out.
pub fn straight_pull_out(side: f32, ramp: f32) -> impl Fn(f32) -> DVec3 {
    move |d: f32| {
        let t = (d / ramp.max(0.1)).clamp(0.0, 1.0);
        let lat = -side * t * t * (3.0 - 2.0 * t);
        DVec3::new(lat as f64, d as f64, 0.0)
    }
}

/// How far behind something standing in its lane a vehicle has to stop to get out round
/// it later from a standstill (m, from its front bumper to the other's body), with its own
/// wheelbase, steering lock and steering rate: the shortest gap from which one of the
/// `pull_out_ramps` takes it past the corner of a standing bus (`obstacle_half_width`,
/// in the middle of the lane, the oncoming lane `side` metres over) with
/// `PULL_OUT_CLEARANCE` to spare, and half a metre more for where it comes to rest.
/// `extent` is (origin to front bumper, to rear bumper, half width).
pub fn pull_out_room(def: &Vehicle, extent: (f32, f32, f32), obstacle_half_width: f64, side: f32) -> f32 {
    let (front, _, _) = extent;
    let bus_half_len = 6.0;
    let mut gap = 0.75f32;
    while gap <= 12.0 {
        let obstacle = crate::collision::Obb::vehicle(DVec2::new(0.0, (front + gap) as f64 + bus_half_len), 0.0, bus_half_len, bus_half_len, obstacle_half_width);
        for ramp in pull_out_ramps(gap, front, false) {
            let way = straight_pull_out(side, ramp);
            let mut body = AiBody::new(def, MotionKind::Road);
            body.place(&way, None, None, 0.0);
            if body.sweep_clearance(&way, 0.0, PULL_OUT_WAIT, PULL_OUT_ACCEL, 8.0, gap + 6.0, extent, &[obstacle]) >= PULL_OUT_CLEARANCE {
                return gap + 0.5;
            }
        }
        gap += 0.25;
    }
    12.5
}

/// Unit vector (east, north) of a heading in degrees.
fn dir(h: f64) -> DVec2 {
    let r = h.to_radians();
    DVec2::new(r.sin(), r.cos())
}

fn wrap(d: f64) -> f64 {
    (d + 180.0).rem_euclid(360.0) - 180.0
}

/// One wheel on its spring.
#[derive(Debug, Clone, Copy)]
struct Wheel {
    axle: usize,
    side: usize,
    /// Body frame: right and forward (m).
    lat: f32,
    long: f32,
    /// Spring (N/m) and damper (Ns/m) of this wheel.
    k: f32,
    c: f32,
}

/// How far over an AI car's wheel its ground is looked for (m): Omsi.exe casts the ray of its
/// ground query from 3 m over the point (0x7c40c8) and takes the highest face under that.
pub const AI_RAY_UP: f64 = 3.0;
/// [ROLLBACK aistepup-1] How far over its lane (m) a face still counts as the ground under
/// an AI wheel: a kerb, a junction plate or a road drawn up to a metre over its paths (a
/// map had 0.9 m); not a bridge deck, which passes higher than that wherever a road runs
/// under it.
pub const AI_STEP_UP: f64 = 1.0;
/// A face this far under the wheel's lane is not this wheel's ground (m): the road under the
/// bridge the lane crosses, or the ground beyond the edge of the road, which the wheel hangs
/// over on its spring rather than drops onto (the lane stays its height there).
const AI_DROP: f64 = 0.3;

#[derive(Debug, Clone)]
pub struct AiBody {
    pub kind: MotionKind,
    rot_long: f32,
    wheelbase: f32,
    front_long: f32,
    rear_long: f32,
    /// The body for the careful driver's line (`keep_swept_in_lane`): rotation point to
    /// front bumper and half the width (m); None drives the rotation point on the way.
    swept: Option<(f32, f32)>,
    /// How much further inside the rotation point's circle the rear sections hanging behind
    /// the vehicle run than its own inner side (m², `rear_off_tracking`); 0 for a rigid one.
    rear_pull: f32,
    /// Where the lanes the vehicle drives over end left and right of its way (m,
    /// `set_lane_edges`): the inner edge of a bend for the careful driver's line. Infinite
    /// until set: no inner edge.
    lane_edges: (f32, f32),
    /// Largest front wheel angle (deg) and how fast the driver turns towards it (deg/s).
    max_steer: f32,
    steer_rate: f32,
    wheels: Vec<Wheel>,
    axle_count: usize,
    mass: f32,
    cog: f32,
    /// Moments of inertia about the lateral (pitch) and longitudinal (roll) axes (kg m²).
    inertia: [f32; 2],
    /// Stiffness and damping of the body in heave (N/m), pitch and roll (Nm/rad).
    spring: [f32; 3],
    damper: [f32; 3],
    started: bool,
    /// World position of the rotation point (x east, y north).
    rear: DVec2,
    /// Heading (deg, clockwise from north).
    pub heading: f64,
    /// Front wheel angle of the bicycle model (deg, + = right), rate limited and smoothed.
    pub steer: f32,
    /// The feed-forward part of the last steering command (deg, + = right): the bicycle model's
    /// angle for the line's curvature at the rotation point (`drive`); for the harness.
    pub steer_ff: f32,
    steer_cmd: f32,
    last_speed: f32,
    /// Longitudinal and lateral acceleration (m/s², forward / right positive).
    pub a_long: f32,
    pub a_lat: f32,
    yaw_rate: f32,
    /// The body on its springs: height of the origin, pitch and roll (rad), their rates.
    z: f64,
    vz: f32,
    pitch: f32,
    vpitch: f32,
    roll: f32,
    vroll: f32,
    /// The plane the wheels stand on last frame (height, pitch, roll) and its rates.
    ground: Option<(f64, f32, f32)>,
    ground_rate: [f32; 3],
    /// Pose for the vehicle: origin, pitch and bank (deg, + = nose up / right side down).
    pub position: DVec3,
    pub pitch_deg: f32,
    pub bank_deg: f32,
    /// Per axle (left, right): how far the wheel hangs below its rest position against
    /// the body (m, the `Axle_Suspension_*` convention: negative = compressed).
    pub suspension: Vec<[f32; 2]>,
    /// Each wheel's contact height last frame (NaN until the first frame): where its next
    /// ground query starts from.
    contact_z: Vec<f64>,
}

impl AiBody {
    pub fn new(def: &Vehicle, kind: MotionKind) -> AiBody {
        let (rot_long, wheelbase) = rotation_point(def);
        let front_long = def.axles.iter().map(|a| a.long).fold(f32::MIN, f32::max);
        let rear_long = def.axles.iter().map(|a| a.long).fold(f32::MAX, f32::min);
        let (front_long, rear_long) = if def.axles.len() >= 2 { (front_long, rear_long) } else { (rot_long + wheelbase, rot_long) };
        // `[mass]` is in tonnes (a few files give kilograms), `[momentofintertia]` in the
        // same unit times m²: 1 t and 1.49 for a Golf, 10.9 t and 300 for an SD200
        let tonnes = def.mass < 100.0;
        let mass = (if tonnes { def.mass * 1000.0 } else { def.mass }).max(300.0);
        let unit = if tonnes { 1000.0 } else { 1.0 };
        let heavy = mass > 6000.0;
        // inv_min_turnradius = tan(max angle) / wheelbase
        let max_steer = if def.inv_min_turn_radius > 0.0 { (def.inv_min_turn_radius * wheelbase).atan().to_degrees().clamp(15.0, 55.0) } else { 35.0 };
        let mut wheels = Vec::new();
        for (i, a) in def.axles.iter().enumerate() {
            let half = (a.max_width * 0.5).max(0.4);
            for (side, lat) in [(0, -half), (1, half)] {
                // `achse_feder` / `achse_daempfer`: kN/m and kNs/m per side
                wheels.push(Wheel { axle: i, side, lat, long: a.long, k: a.spring.max(0.0) * 1000.0, c: a.damper.max(0.0) * 1000.0 });
            }
        }
        let n = wheels.len().max(1) as f32;
        if wheels.iter().any(|w| w.k <= 0.0) {
            // no springs in the file: 1.5 Hz with a third of critical damping
            let k_total = mass * (std::f32::consts::TAU * 1.5).powi(2);
            let c_total = 2.0 * 0.35 * (k_total * mass).sqrt();
            for w in wheels.iter_mut() {
                w.k = k_total / n;
                w.c = c_total / n;
            }
        }
        let mut spring = [0.0f32; 3];
        let mut damper = [0.0f32; 3];
        for w in &wheels {
            spring[0] += w.k;
            spring[1] += w.k * w.long * w.long;
            spring[2] += w.k * w.lat * w.lat;
            damper[0] += w.c;
            damper[1] += w.c * w.long * w.long;
            damper[2] += w.c * w.lat * w.lat;
        }
        let half_track = wheels.iter().map(|w| w.lat.abs()).fold(0.5, f32::max);
        // (pitch and roll: the first and the third value, as Omsi.exe reads them - see
        // `RigidBody::from_definition`)
        let mut inertia = [def.moment_of_inertia[0] * unit, def.moment_of_inertia[2] * unit];
        if inertia[0] <= 0.0 {
            inertia[0] = mass * (front_long * front_long + rear_long * rear_long) * 0.5;
        }
        if inertia[1] <= 0.0 {
            inertia[1] = mass * half_track * half_track * 1.2;
        }
        // keep every mode below ~6 Hz: the springs are integrated at a few hundred Hz at most
        for (i, j) in [(1usize, 0usize), (2, 1)] {
            let max_k = inertia[j] * 38.0f32.powi(2);
            if spring[i] > max_k {
                inertia[j] = spring[i] / 38.0f32.powi(2);
            }
        }
        // at least 0.8 of critical damping in every mode: a driven car settles after a
        // bump or a stop instead of swinging on (the files' dampers are tuned for the
        // player's physics, which damps through the tyres as well)
        let heave_c = 2.0 * 0.8 * (spring[0] * mass).sqrt();
        damper[0] = damper[0].max(heave_c);
        for (i, j) in [(1usize, 0usize), (2, 1)] {
            damper[i] = damper[i].max(2.0 * 0.8 * (spring[i] * inertia[j]).sqrt());
        }
        let axle_count = def.axles.len().max(1);
        AiBody {
            kind,
            rot_long,
            wheelbase,
            front_long,
            rear_long,
            swept: None,
            rear_pull: 0.0,
            lane_edges: (f32::INFINITY, f32::INFINITY),
            max_steer,
            steer_rate: if heavy { 25.0 } else { 40.0 },
            wheels,
            axle_count,
            mass,
            cog: if def.cog_height > 0.0 { def.cog_height } else { 0.6 },
            inertia,
            spring,
            damper,
            started: false,
            rear: DVec2::ZERO,
            heading: 0.0,
            steer: 0.0,
            steer_ff: 0.0,
            steer_cmd: 0.0,
            last_speed: 0.0,
            a_long: 0.0,
            a_lat: 0.0,
            yaw_rate: 0.0,
            z: 0.0,
            vz: 0.0,
            pitch: 0.0,
            vpitch: 0.0,
            roll: 0.0,
            vroll: 0.0,
            ground: None,
            ground_rate: [0.0; 3],
            position: DVec3::ZERO,
            pitch_deg: 0.0,
            bank_deg: 0.0,
            suspension: vec![[0.0; 2]; axle_count],
            contact_z: Vec::new(),
        }
    }

    /// Drive the careful driver's line in bends (`careful_line_offset`): `front` is the
    /// distance from the vehicle's origin to its front bumper, `half_width` half the body's
    /// width (the traffic's extent of the type). For AI road vehicles; a body without it
    /// keeps its rotation point on the way.
    pub fn keep_swept_in_lane(&mut self, front: f32, half_width: f32) {
        if self.kind == MotionKind::Road {
            self.swept = Some(((front - self.rot_long).max(0.0), half_width.max(0.0)));
        }
    }

    /// `keep_swept_in_lane` for a vehicle with sections hung behind it: `parts` as for
    /// `rear_off_tracking`. The swept band is the whole vehicle's, from the innermost point
    /// of its rear sections to the outer front corner.
    pub fn keep_swept_in_lane_with(&mut self, front: f32, half_width: f32, parts: &[&crate::vehicle::VehicleType]) {
        self.keep_swept_in_lane(front, half_width);
        self.rear_pull = rear_off_tracking(parts);
    }

    /// Where the lanes the vehicle drives over now end to the left and to the right of its way
    /// (m): the careful driver's line keeps the body's innermost point inside the edge on the
    /// bend's inside (`careful_line_offset`). The game gives the narrowest of the lanes from the
    /// vehicle's rear end to the end of `steer_span` (`traffic::AiState::lane_edges_between`),
    /// less the way's sideways offset towards each side, and 0 on a side the map does not tell
    /// (no inward offset there: the way's line). Infinite (as before it is set): no edge.
    pub fn set_lane_edges(&mut self, left: f32, right: f32) {
        self.lane_edges = (left, right);
    }

    /// Put the body onto its way, standing still and straight. `way(d)` is the point `d`
    /// metres along the way from the vehicle's origin.
    pub fn place(&mut self, way: &dyn Fn(f32) -> DVec3, ground: Option<&dyn Fn(f64, f64) -> Option<f64>>, contact: Option<&dyn crate::rigid::Ground>, speed: f32) {
        self.place_in_lane(way, None, ground, contact, speed);
    }

    /// `place`, with the heading of the vehicle's lane (see `step_in_lane`).
    pub fn place_in_lane(&mut self, way: &dyn Fn(f32) -> DVec3, lane: Option<&dyn Fn(f32) -> f32>, ground: Option<&dyn Fn(f64, f64) -> Option<f64>>, contact: Option<&dyn crate::rigid::Ground>, speed: f32) {
        self.started = false;
        self.ground = None;
        self.contact_z.clear();
        self.last_speed = speed;
        self.step_in_lane(0.0, speed, way, lane, ground, contact);
    }

    /// Follow the way for `dt` seconds at `speed` (m/s, the planner's speed).
    /// `contact`: what the wheels stand on (the faces under a height, as the player's wheels
    /// ask it); without it the plain height sampler `ground`.
    pub fn step(&mut self, dt: f32, speed: f32, way: &dyn Fn(f32) -> DVec3, ground: Option<&dyn Fn(f64, f64) -> Option<f64>>, contact: Option<&dyn crate::rigid::Ground>) {
        self.step_in_lane(dt, speed, way, None, ground, contact);
    }

    /// `step`, with `lane(d)` the heading (deg) of the vehicle's lanes `d` metres along (the
    /// way's without its lane change and its sideways moves, `AiState::lane_heading`): the
    /// careful driver's line takes the bend from the lane, so a pull-out's or a lane change's
    /// S-curve keeps the rotation point on the way. Without it the bend is the way's own.
    pub fn step_in_lane(&mut self, dt: f32, speed: f32, way: &dyn Fn(f32) -> DVec3, lane: Option<&dyn Fn(f32) -> f32>, ground: Option<&dyn Fn(f64, f64) -> Option<f64>>, contact: Option<&dyn crate::rigid::Ground>) {
        if dt > 0.0 {
            let a = (speed - self.last_speed) / dt;
            self.a_long += (a - self.a_long) * (dt / 0.15).min(1.0);
        }
        self.last_speed = speed;
        match self.kind {
            MotionKind::Road => {
                self.drive(dt, speed, way, lane);
                self.settle(dt, way, ground, contact);
            }
            MotionKind::Rail => self.ride(way),
            MotionKind::Air => self.fly(dt, speed, way),
        }
    }

    /// How far ahead of the rotation point the steering aims at `speed` (m): further the
    /// faster the car goes.
    fn look(&self, speed: f32) -> f32 {
        (1.2 * self.wheelbase).max(3.5) + 0.6 * speed.min(20.0)
    }

    /// The stretch of the way (m from the vehicle's origin) the careful driver's line is
    /// steered over at `speed`: from the rotation point to the aim. The lane edges it keeps to
    /// (`set_lane_edges`) are the narrowest of the lanes from the vehicle's rear end (its last
    /// section's) to the aim, the end of this stretch (`traffic::AiState::lane_edges_between`).
    pub fn steer_span(&self, speed: f32) -> (f32, f32) {
        (self.rot_long, self.rot_long + self.look(speed))
    }

    /// Bicycle model: the rotation point follows its line, the front wheels steer by the
    /// line's own curvature where the body is and correct towards a point further along it
    /// (feed-forward plus feedback, the standard path-tracking pair: J. M. Snider, "Automatic
    /// Steering Methods for Autonomous Automobile Path Tracking", CMU-RI-TR-09-08, 2009, the
    /// control law δ = -Kx + δ_ff of its "Optimal Control with Feed Forward Term", and its
    /// kinematic bicycle model's δ = atan(κL)). The feed-forward is the bicycle model's steer
    /// angle atan(wheelbase · k) for the curvature k of the lane's bend at the rotation point
    /// (its headings half a wheelbase either side of it, up to the way's own curvature there
    /// and 0 where the signs differ). The feedback is pure pursuit (looking further ahead the
    /// faster the car goes) less what pure pursuit would steer on the lane's bend itself (the
    /// rotation point on its line, the body along its tangent): on a bend of the lane it
    /// corrects only the tracking error, and pure pursuit's own look-ahead no longer cuts the
    /// inside where the bend tightens; a lane change's, a pull-out's or a swerve's S-curve
    /// (the way off the lane's axis) is steered by pure pursuit itself, which turns the wheel
    /// back in time. Without `lane` the bend is the way's.
    /// The line is the way, moved inside a bend of the lane (`lane`, else the way) by
    /// `careful_line_offset` where the body is given (`keep_swept_in_lane`); the bend fed
    /// forward is tighter by the offset (the parallel curve
    /// `side` inside: k / (1 - side · k)), which carries the rear sections' pull of an
    /// articulated vehicle (`rear_pull`) as the offset does.
    fn drive(&mut self, dt: f32, speed: f32, way: &dyn Fn(f32) -> DVec3, lane: Option<&dyn Fn(f32) -> f32>) {
        let look = self.look(speed);
        // The way's points the steering looks at: the rotation point's, half way to the aim
        // and the aim. The turn between the two chords says how tight the way bends over the
        // look-ahead (a circle through the three points has curvature 2·turn/look; + = left).
        let (a, b, c) = (way(self.rot_long).truncate(), way(self.rot_long + 0.5 * look).truncate(), way(self.rot_long + look).truncate());
        let (u, w) = (b - a, c - b);
        let turn_signed = (u.x * w.y - u.y * w.x).atan2(u.dot(w));
        let turn = turn_signed.abs() as f32;
        let bend_k = if look > 0.1 { 2.0 * turn / look } else { 0.0 };
        // The careful driver's line: the rotation point and the aim both moved inside the
        // bend by the offset that centres the swept body on the way, square to the way's
        // tangent there (on that circle the tangent at the first point is the first chord
        // turned back by half the turn, at the aim the second chord turned on by half). The
        // bend is the lane's: its heading's turn from the rotation point to the aim over the
        // look-ahead (the same mean curvature; + = left, headings turn clockwise), up to the
        // way's own curvature and 0 where their signs differ.
        let side = match self.swept {
            Some((reach, half_width)) if look > 0.1 => {
                // (the way's own: 2·turn/look, + = left)
                let k_way = 2.0 * turn_signed as f32 / look;
                let k = match lane {
                    Some(l) => {
                        let k_lane = -(((l(self.rot_long + look) - l(self.rot_long)) as f64 + 180.0).rem_euclid(360.0) - 180.0).to_radians() as f32 / look;
                        // no more of a bend than the way has, and none where the two turn
                        // different ways (a lane's heading that its points do not follow)
                        if k_lane * k_way > 0.0 { k_lane.abs().min(k_way.abs()).copysign(k_lane) } else { 0.0 }
                    }
                    None => k_way,
                };
                // (the bend's inside: its left for k > 0)
                let inner_edge = if k > 0.0 { self.lane_edges.0 } else { self.lane_edges.1 };
                careful_line_offset(k, reach, half_width, self.rear_pull, inner_edge) as f64
            }
            _ => 0.0,
        };
        let inside = |chord: DVec2, by: f64| {
            let t = DVec2::from_angle(by).rotate(chord.normalize_or_zero());
            DVec2::new(-t.y, t.x) * side
        };
        let (target, aim) = if side != 0.0 { (a + inside(u, -0.5 * turn_signed), c + inside(w, 0.5 * turn_signed)) } else { (a, c) };
        // The way's curvature at the rotation point (+ = left), from the circle through its
        // points half a wheelbase behind the rotation point, at it and half a wheelbase ahead
        // (the curvature where the rotation point is, on the body's own scale). On that circle
        // the tangent at the middle point lies along the chord between the outer two.
        let (p_back, p_ahead) = (way(self.rot_long - 0.5 * self.wheelbase).truncate(), way(self.rot_long + 0.5 * self.wheelbase).truncate());
        let k_way_here = {
            let (u1, w1) = (a - p_back, p_ahead - a);
            2.0 * (u1.x * w1.y - u1.y * w1.x).atan2(u1.dot(w1)) / self.wheelbase as f64
        };
        // The pure pursuit that a body standing on a line along it would steer (+ = right, as
        // `alpha` below): `g` from its rotation point to its aim, `t` its tangent there.
        let pursuit = |g: DVec2, t: DVec2| {
            let r = DVec2::new(t.y, -t.x);
            let alpha = (g.dot(r) as f32).atan2(g.dot(t) as f32).clamp(-FRAC_PI_2, FRAC_PI_2);
            2.0 * alpha.sin() / (g.length() as f32).max(1.0)
        };
        // The feed-forward and what it replaces of pure pursuit are the lane's bend only (its
        // headings `lane`: the way without its lane change and sideways moves), the parallel
        // line `side` inside it: a lane change's, a pull-out's or a swerve's S-curve is steered
        // by pure pursuit, whose look-ahead turns the wheel back before the S's second half (fed
        // forward, the rate-limited wheel lagged the S and the body ran on past it into the next
        // lane: a car changing lanes at 2.5 m/s at a red light stood 1.9 m over, beside the car
        // there). The lane's curvature counts as the careful line's does: up to the way's own
        // and 0 where their signs differ. Without `lane`, the way's.
        let (k_line, ideal) = match lane {
            Some(l) => {
                let (s0, hw) = (self.rot_long, 0.5 * self.wheelbase);
                let k_lane = -wrap((l(s0 + hw) - l(s0 - hw)) as f64).to_radians() / self.wheelbase as f64;
                let k = if k_lane * k_way_here > 0.0 { k_lane.abs().min(k_way_here.abs()).copysign(k_lane) } else { 0.0 };
                // the lane's line from the rotation point to the aim, laid from its headings
                // (midpoint rule over the steps of `LANE_LINE_STEPS`)
                let step = look / LANE_LINE_STEPS as f32;
                let along: DVec2 = (0..LANE_LINE_STEPS).map(|i| dir(l(s0 + (i as f32 + 0.5) * step) as f64) * step as f64).sum();
                let (t0, t1) = (dir(l(s0) as f64), dir(l(s0 + look) as f64));
                let left = |t: DVec2| DVec2::new(-t.y, t.x);
                let g = along + (left(t1) - left(t0)) * side;
                (k / (1.0 - side * k).max(f64::EPSILON), pursuit(g, t0))
            }
            None => (k_way_here / (1.0 - side * k_way_here).max(f64::EPSILON), pursuit(aim - target, (p_ahead - p_back).normalize_or_zero())),
        };
        if !self.started || (target - self.rear).length() > 8.0 {
            // spawned, or put somewhere else: stand on the line, straight along the way
            let ahead = way(self.rot_long + 2.0).truncate();
            let d = ahead - a;
            self.heading = if d.length() > 1e-3 { d.x.atan2(d.y).to_degrees() } else { self.heading };
            self.rear = target;
            self.steer = 0.0;
            self.steer_cmd = 0.0;
            self.started = true;
        }
        let fwd = dir(self.heading);
        let right = DVec2::new(fwd.y, -fwd.x);
        // the planner's position along the way leads; the body keeps up with it
        let along = (target - self.rear).dot(fwd) as f32;
        // ([Q8-1] backing up, `AiState::reverse_step`: the body follows backwards, the wheels straight)
        let reversing = speed < -0.02 || (speed <= 0.02 && along < -0.05);
        let v = if reversing {
            (speed + 1.5 * along).clamp(speed.min(0.0) - 3.0, 0.0)
        } else if speed < 0.02 && along.abs() < 0.05 {
            0.0
        } else {
            (speed + 1.5 * along).clamp(0.0, speed + 3.0)
        };
        let g = aim - self.rear;
        let alpha = (g.dot(right) as f32).atan2(g.dot(fwd) as f32).clamp(-FRAC_PI_2, FRAC_PI_2);
        let reach = (g.length() as f32).max(1.0);
        // The way itself says how tight it bends here: a junction's turn laid tighter than
        // the model's `[inv_min_turnradius]` still has to be followed - capped at the
        // model's lock the car ran wide of it, through the kerb, the corner house and the
        // people on the pavement (#249; OMSI's AI keeps to its path). The lock the bend
        // needs, and a little more, is allowed, and the wheel may turn that much faster.
        let need = (bend_k * self.wheelbase).atan().to_degrees() * 1.15;
        let limit = if std::env::var_os("OMSI_AI_MODEL_LOCK").is_some() { self.max_steer } else { self.max_steer.max(need.min(60.0)) };
        // OMSI_DEBUG_AI_WIDE: every tenth of a second a car stands over 1.5 m beside its way
        if std::env::var_os("OMSI_DEBUG_AI_WIDE").is_some() {
            let off = ((target - self.rear).dot(right)).abs();
            if off > 1.5 {
                static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n % 50 == 0 {
                    log::info!("ai wide: {off:.1} m beside its way at ({:.0}, {:.0}), bend needs {need:.0} deg, lock {:.0} ({} so far)", self.rear.x, self.rear.y, self.max_steer, n + 1);
                }
            }
        }
        // feed-forward (the line's curvature, + = right) plus pure pursuit's correction
        let k_cmd = -k_line as f32 + (2.0 * alpha.sin() / reach - ideal);
        self.steer_ff = (self.wheelbase * -k_line as f32).atan().to_degrees();
        let want = if reversing { 0.0 } else { (self.wheelbase * k_cmd).atan().to_degrees().clamp(-limit, limit) };
        if dt > 0.0 {
            let rate = self.steer_rate * dt * if limit > self.max_steer { 1.5 } else { 1.0 };
            self.steer_cmd += (want - self.steer_cmd).clamp(-rate, rate);
            self.steer += (self.steer_cmd - self.steer) * (dt / 0.08).min(1.0);
        }
        let k = self.steer.to_radians().tan() / self.wheelbase;
        let dpsi = (v * dt * k) as f64;
        let mid = dir(self.heading + dpsi.to_degrees() * 0.5);
        self.rear += mid * (v * dt) as f64;
        self.heading = (self.heading + dpsi.to_degrees()).rem_euclid(360.0);
        self.yaw_rate = if dt > 0.0 { (dpsi / dt as f64) as f32 } else { 0.0 };
        self.a_lat = v * v * k;
        let o = self.rear - dir(self.heading) * self.rot_long as f64;
        self.position.x = o.x;
        self.position.y = o.y;
    }

    /// The body on its springs over the ground under its wheels: the plane through the
    /// contact points is where the body wants to be, braking and cornering lean it off
    /// that plane, and the difference is how far each wheel moves in its arch.
    fn settle(&mut self, dt: f32, way: &dyn Fn(f32) -> DVec3, ground: Option<&dyn Fn(f64, f64) -> Option<f64>>, contact: Option<&dyn crate::rigid::Ground>) {
        let fwd = dir(self.heading);
        let right = DVec2::new(fwd.y, -fwd.x);
        let origin = self.position.truncate();
        // the way's own height at each axle: the road the map says is there (where nothing
        // is drawn under a wheel, and what tells another level from this one)
        let mut axle_z = vec![f64::NAN; self.axle_count];
        if self.contact_z.len() != self.wheels.len() {
            self.contact_z = vec![f64::NAN; self.wheels.len()];
        }
        let mut contacts: Vec<(f32, f32, f64)> = Vec::with_capacity(self.wheels.len());
        // per wheel: no face near its lane's height, and the face found deeper down
        let mut hung: Vec<bool> = Vec::with_capacity(self.wheels.len());
        let mut low: Vec<Option<f64>> = Vec::with_capacity(self.wheels.len());
        for w in self.wheels.iter() {
            if axle_z[w.axle].is_nan() {
                axle_z[w.axle] = way(w.long).z;
            }
            let path_z = axle_z[w.axle];
            let p = origin + right * w.lat as f64 + fwd * w.long as f64;
            // What the wheel stands on, as Omsi.exe stands an AI car's wheels: its AI cars
            // are bodies on the same wheel physics as the player's bus (0x7d5124 ->
            // 0x7e2574), asking the ground under each wheel (0x7aec3c -> 0x7a0814): a ray
            // from 3 m over the wheel (0x7c40c8 adds 3 to z) down into the terrain, the
            // splines and the first mesh of the `[surface]` objects, and the highest face it
            // meets is the ground. Each wheel on its own face: a car on a cambered road, a
            // kerb or a junction plate tilted on a hill leans with it. (Kept to the lane's
            // height, the least of an axle's wheels and only a few centimetres off that,
            // every car stood up to 5 cm in the asphalt wherever the road was drawn higher
            // than its lane or one wheel stood lower than the other, and one whose lane ran
            // under the drawn road drove through it with only its roof showing.) The ray
            // starts over the wheel's last contact, or over the lane where it has none yet.
            // Without the world's faces, the plain height sampler.
            // [ROLLBACK aistepup-1] The ray starts `AI_STEP_UP` over the lane, not 3 m over
            // the wheel's last contact: from 3 m up it met the deck of a bridge wherever
            // the deck passed less than 3 m over the lower road (its ramps and abutments),
            // the car was lifted onto the deck and came down again where the deck rose
            // out of reach (BeijingCity, Andingmennei). A kerb's height over the lane is
            // still ground (a junction plate drawn a little over its paths); the reach
            // down stays 3 m (`deeper`).
            let (drawn, deeper) = match contact {
                Some(c) => {
                    let below = c.probe(p.x, p.y, path_z + AI_STEP_UP).below;
                    (below.filter(|g| *g >= path_z - AI_DROP), below.filter(|g| *g < path_z - AI_DROP && *g > path_z - AI_RAY_UP))
                }
                None => (ground.and_then(|g| g(p.x, p.y)).filter(|g| *g > path_z - AI_RAY_UP && *g <= path_z + AI_STEP_UP), None),
            };
            low.push(deeper);
            contacts.push((w.lat, w.long, drawn.unwrap_or(path_z)));
            hung.push(drawn.is_none());
        }
        // A wheel hangs over a drop on its spring while the others carry the car; with no
        // wheel near its lane's height the car has nothing to hang from and comes down onto
        // what is drawn under it, as Omsi.exe's wheel physics drops it. (Kept at the lane's
        // height, a car whose junction's paths ran half a metre over the drawn road drove
        // through the junction in the air, #876.)
        if hung.iter().all(|h| *h) && low.iter().any(|l| l.is_some()) {
            for (c, l) in contacts.iter_mut().zip(&low) {
                if let Some(z) = l {
                    c.2 = *z;
                }
            }
        }
        for (k, c) in contacts.iter().enumerate() {
            self.contact_z[k] = c.2;
        }
        // least-squares plane h = a + b·long + c·lat (the wheels sit symmetrically, so the
        // two slopes separate)
        let n = contacts.len().max(1) as f64;
        let (ml, mt, mh) = contacts.iter().fold((0.0, 0.0, 0.0), |acc, c| (acc.0 + c.1 as f64 / n, acc.1 + c.0 as f64 / n, acc.2 + c.2 / n));
        let (mut sll, mut slh, mut stt, mut sth) = (0.0, 0.0, 0.0, 0.0);
        for c in &contacts {
            let (dl, dt_, dh) = (c.1 as f64 - ml, c.0 as f64 - mt, c.2 - mh);
            sll += dl * dl;
            slh += dl * dh;
            stt += dt_ * dt_;
            sth += dt_ * dh;
        }
        let b = if sll > 1e-6 { slh / sll } else { 0.0 };
        let c = if stt > 1e-6 { sth / stt } else { 0.0 };
        let zg = mh - b * ml - c * mt;
        // pitch up for a road rising ahead; bank positive = right side down (c < 0)
        let pg = b.atan() as f32;
        let rg = (-c).atan() as f32;
        match self.ground {
            Some((z0, p0, r0)) if (zg - self.z).abs() < 1.0 && dt > 0.0 => {
                let k = (dt / 0.1).min(1.0);
                let rates = [((zg - z0) / dt as f64) as f32, (pg - p0) / dt, (rg - r0) / dt];
                for (r, new) in self.ground_rate.iter_mut().zip(rates) {
                    *r += (new.clamp(-5.0, 5.0) - *r) * k;
                }
                let steps = 4;
                let h = dt / steps as f32;
                for _ in 0..steps {
                    let az = (self.spring[0] * (zg - self.z) as f32 + self.damper[0] * (self.ground_rate[0] - self.vz)) / self.mass;
                    let ap = (self.spring[1] * (pg - self.pitch) + self.damper[1] * (self.ground_rate[1] - self.vpitch) + self.mass * self.a_long * self.cog) / self.inertia[0];
                    let ar = (self.spring[2] * (rg - self.roll) + self.damper[2] * (self.ground_rate[2] - self.vroll) - self.mass * self.a_lat * self.cog) / self.inertia[1];
                    self.vz += az * h;
                    self.vpitch += ap * h;
                    self.vroll += ar * h;
                    self.z += (self.vz * h) as f64;
                    self.pitch += self.vpitch * h;
                    self.roll += self.vroll * h;
                }
                self.z = self.z.clamp(zg - 0.15, zg + 0.15);
                self.pitch = self.pitch.clamp(pg - 0.07, pg + 0.07);
                self.roll = self.roll.clamp(rg - 0.09, rg + 0.09);
            }
            _ => {
                // first frame, or the ground under the car changed completely
                self.z = zg;
                self.pitch = pg;
                self.roll = rg;
                self.vz = 0.0;
                self.vpitch = 0.0;
                self.vroll = 0.0;
                self.ground_rate = [0.0; 3];
            }
        }
        self.ground = Some((zg, pg, rg));
        self.position.z = self.z;
        self.pitch_deg = self.pitch.to_degrees();
        self.bank_deg = self.roll.to_degrees();
        let (tp, tr) = (self.pitch.tan(), self.roll.tan());
        for (w, c) in self.wheels.iter().zip(&contacts) {
            let body = self.z + (w.long * tp - w.lat * tr) as f64;
            if let Some(s) = self.suspension.get_mut(w.axle) {
                s[w.side] = ((body - c.2) as f32).clamp(-0.15, 0.15);
            }
        }
    }

    /// A rail vehicle: its outermost axles stand on the track, the body is the line
    /// between them.
    fn ride(&mut self, way: &dyn Fn(f32) -> DVec3) {
        let f = way(self.front_long);
        let r = way(self.rear_long);
        let d = f - r;
        let run = d.truncate().length();
        if run > 1e-3 {
            self.heading = d.x.atan2(d.y).to_degrees().rem_euclid(360.0);
            self.pitch_deg = (d.z / run).atan().to_degrees() as f32;
        }
        let span = (self.front_long - self.rear_long).max(0.1);
        self.position = r + d * ((0.0 - self.rear_long) / span) as f64;
        self.bank_deg = 0.0;
        self.steer = 0.0;
        self.started = true;
    }

    /// An aircraft: on its path, nose along it, banking into a turn as much as the turn
    /// asks for (tan bank = speed × turn rate / g).
    fn fly(&mut self, dt: f32, speed: f32, way: &dyn Fn(f32) -> DVec3) {
        let p = way(0.0);
        let d = way(15.0) - way(-15.0);
        let run = d.truncate().length();
        let (h_t, p_t) = if run > 1e-3 { (d.x.atan2(d.y).to_degrees(), (d.z / run).atan().to_degrees() as f32) } else { (self.heading, self.pitch_deg) };
        if !self.started || dt <= 0.0 {
            self.heading = h_t.rem_euclid(360.0);
            self.pitch_deg = p_t;
            self.bank_deg = 0.0;
            self.yaw_rate = 0.0;
            self.started = true;
        } else {
            let k = (dt / 0.6).min(1.0) as f64;
            let dh = wrap(h_t - self.heading) * k;
            self.heading = (self.heading + dh).rem_euclid(360.0);
            let rate = (dh.to_radians() / dt as f64) as f32;
            self.yaw_rate += (rate - self.yaw_rate) * (dt / 0.5).min(1.0);
            let bank = (speed * self.yaw_rate / 9.81).atan().to_degrees().clamp(-30.0, 30.0);
            let k = (dt / 0.8).min(1.0);
            self.bank_deg += (bank - self.bank_deg) * k;
            self.pitch_deg += (p_t - self.pitch_deg) * k;
        }
        self.position = p;
        self.steer = 0.0;
    }

    /// Drive a copy of this body along a way it has not taken yet and report how close it
    /// comes to `obstacles`: the smallest plan-view separation (m, negative = the bodies
    /// would touch) between the vehicle's box (`front`/`rear` from its origin, `half_width`)
    /// and the nearest of them while the vehicle covers `distance` metres. `way(d)` is the point `d`
    /// metres along the new way from where the vehicle is now (the same way `step` is given,
    /// e.g. with a pull-out round something standing ahead in it); the vehicle starts at
    /// `speed` (a standing one first turns its wheels for `wait` seconds, as a driver does
    /// before moving off) and speeds up at `accel` to at most `v_max`. The steering is the real one -
    /// its rate limit, its lock and the pure pursuit that cuts in on a steep S-bend - which
    /// is what decides whether a car standing a few metres behind a bus can get its front
    /// corner past the bus's. The way's S-curve is laid on a straight lane along the way's
    /// start (`drive`'s lane: a pull-out's S is steered by pure pursuit, as in the game).
    #[allow(clippy::too_many_arguments)]
    pub fn sweep_clearance(&self, way: &dyn Fn(f32) -> DVec3, speed: f32, wait: f32, accel: f32, v_max: f32, distance: f32, extent: (f32, f32, f32), obstacles: &[crate::collision::Obb]) -> f64 {
        let mut body = self.clone();
        let (front, rear, half_width) = extent;
        let dt = 1.0 / 20.0;
        let mut v = speed.max(0.0);
        let mut x = 0.0f32;
        let gap = |b: &AiBody| {
            let me = crate::collision::Obb::vehicle(b.position.truncate(), b.heading, front as f64, rear as f64, half_width as f64);
            obstacles.iter().map(|o| me.separation(o)).fold(f64::MAX, f64::min)
        };
        let mut best = gap(&body);
        let start = (way(1.0) - way(0.0)).truncate();
        let lane_heading = start.x.atan2(start.y).to_degrees() as f32;
        let lane = |_d: f32| lane_heading;
        if v < 0.05 {
            for _ in 0..(wait / dt).round() as usize {
                body.drive(dt, 0.0, way, Some(&lane));
            }
        }
        // (at most a minute of simulated time: a car that cannot move is not getting past)
        for _ in 0..1200 {
            if x >= distance {
                break;
            }
            v = (v + accel.max(0.3) * dt).min(v_max.max(0.5));
            let ds = v * dt;
            x += ds;
            let at = x;
            body.drive(dt, v, &|d| way(at + d), Some(&lane));
            best = best.min(gap(&body));
        }
        best
    }

    /// Hand the pose to the vehicle: position and attitude, the accelerations its scripts
    /// read and each wheel's travel.
    pub fn apply(&self, v: &mut VehicleInstance) {
        // A road vehicle's body stands on the contact plane of its wheels; its static pose
        // over that plane (`[ai_deltaheight]` and the model's wheel geometry) comes on top,
        // once: the springs here only add their travel about it. Rail and air vehicles
        // follow their path's own height.
        let (lift, delta) = if self.kind == MotionKind::Road { v.ai_rest_offset() } else { (0.0, 0.0) };
        v.position = self.position + DVec3::new(0.0, 0.0, lift as f64);
        v.heading = self.heading;
        v.pitch = self.pitch_deg;
        v.bank = self.bank_deg;
        v.physics.accel = Vec3::new(self.a_lat, self.a_long, 0.0);
        for (axle, s) in v.physics.wheels.iter_mut().zip(&self.suspension) {
            axle[0].suspension = s[0] + delta;
            axle[1].suspension = s[1] + delta;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use omsi_vehicle::vehicle::Axle;

    #[test]
    fn back_in_ramp_keeps_the_sideways_acceleration() {
        for (offset, kmh) in [(3.28f32, 40.0f32), (3.28, 44.0), (2.5, 34.0), (5.0, 50.0)] {
            let v = kmh / 3.6;
            let len = back_in_ramp(offset, v, BACK_IN_LAT_ACCEL);
            // the peak of the smoothstep's second derivative, at its ends
            let peak = 6.0 * offset * v * v / (len * len);
            assert!(peak <= BACK_IN_LAT_ACCEL + 1e-3, "{offset} m at {kmh} km/h: {len:.1} m, {peak:.2} m/s²");
            assert!(len / v >= 2.0, "{offset} m at {kmh} km/h: {len:.1} m is {:.2} s", len / v);
        }
        // creeping round something: the old geometric length
        assert_eq!(back_in_ramp(3.0, 1.0, BACK_IN_LAT_ACCEL), 10.5);
    }

    fn golf() -> Vehicle {
        let mut v = Vehicle { mass: 1.0, moment_of_inertia: [1.49, 0.40, 1.56], cog_height: 0.4, rot_pnt_long: -1.304, inv_min_turn_radius: 0.1904, ..Default::default() };
        v.axles = vec![Axle { long: 1.21, max_width: 1.606, spring: 40.0, damper: 2.0, wheel_diameter: 0.503, ..Default::default() }, Axle { long: -1.304, max_width: 1.606, spring: 50.0, damper: 2.0, wheel_diameter: 0.503, ..Default::default() }];
        v
    }

    /// A way: straight north for 50 m, then a right turn of radius `r`.
    fn bend(r: f64, s: f64) -> DVec3 {
        if s < 50.0 {
            DVec3::new(0.0, s, 10.0)
        } else {
            let a = (s - 50.0) / r;
            DVec3::new(r - r * a.cos(), 50.0 + r * a.sin(), 10.0)
        }
    }

    #[test]
    fn follows_a_bend_smoothly() {
        let def = golf();
        let mut body = AiBody::new(&def, MotionKind::Road);
        let speed = 6.0f32;
        let dt = 1.0 / 30.0;
        let mut s = 20.0f64;
        body.place(&|d| bend(12.0, s + d as f64), None, None, speed);
        let mut last_yaw: Option<f32> = None;
        let mut worst_jump = 0.0f32;
        let mut worst_off = 0.0f64;
        for _ in 0..(12.0 / dt) as usize {
            s += (speed * dt) as f64;
            let at = s;
            body.step(dt, speed, &|d| bend(12.0, at + d as f64), None, None);
            if let Some(y) = last_yaw {
                worst_jump = worst_jump.max((body.yaw_rate - y).abs().to_degrees());
            }
            last_yaw = Some(body.yaw_rate);
            // the rear axle stays on the way (a circle has no steady error; entering it costs a little)
            let rear = body.rear;
            let q = bend(12.0, at + def.rot_pnt_long as f64).truncate();
            worst_off = worst_off.max((rear - q).length());
        }
        // a quarter turn done at 6 m/s: the heading has turned right, the yaw rate never
        // changes between frames by more than the steering's own rate allows at the bend's
        // lock (speed / wheelbase · d tan(steer); the way turns from straight into the circle
        // at a point, and the steering takes the line's curvature at the steering's rate: 3.2
        // deg/s per frame, 0.6 with pure pursuit alone, which cut the corner), the car keeps
        // to its lane (the rear axle 0.20 m off the way, 0.47 with pure pursuit alone)
        assert!(body.heading > 60.0 && body.heading < 300.0, "heading {}", body.heading);
        let lock = (body.wheelbase / 12.0).atan();
        let allowed = (speed / body.wheelbase * (body.steer_rate * dt).to_radians() / lock.cos().powi(2)).to_degrees();
        assert!(worst_jump <= allowed, "yaw rate jumps by {worst_jump} deg/s in one frame (the steering allows {allowed})");
        assert!(worst_off < 0.25, "rear axle {worst_off} m off the way");
    }

    /// A BJ6123, the 12 m two-axle bus of BeijingCity (`Vehicles/BJ6180_BJC/6123.bus`): its
    /// axles and rotation point, and from its `[boundingbox]` (2.55 wide, 12.112 long, 0.478
    /// ahead) its front bumper 6.534 m ahead of the origin and its half width.
    fn bj6123() -> (Vehicle, f32, f32, f32) {
        let mut v = Vehicle { mass: 10.0, moment_of_inertia: [361.6, 89.6, 341.5], cog_height: 1.0, rot_pnt_long: -2.87237, inv_min_turn_radius: 0.195, ..Default::default() };
        v.axles = vec![Axle { long: 2.928, max_width: 2.37, spring: 300.0, damper: 20.0, wheel_diameter: 1.054, ..Default::default() }, Axle { long: -2.87237, max_width: 2.434, spring: 400.0, damper: 25.0, wheel_diameter: 1.054, ..Default::default() }];
        (v, 12.112 * 0.5 + 0.478, 12.112 * 0.5 - 0.478, 2.55 * 0.5)
    }

    /// A Golf with the traffic's extent of a car (front, rear, half width).
    fn golf_body() -> (Vehicle, f32, f32, f32) {
        (golf(), 2.1, 2.2, 0.85)
    }

    /// Drive `body` round the bend of `bend(r, _)` at `speed` and return, once it is well
    /// into the circle, how far its body leaves a lane `lane_width` wide centred on the
    /// way: (outward, inward), the most of each over half a circle (m, + = outside the
    /// lane's edge). The body is sampled at its four corners and at its sides abreast of
    /// the rotation point (where the inner side comes nearest the centre).
    fn swept_outside(body: &mut AiBody, (front, rear, half): (f32, f32, f32), r: f64, speed: f32, lane_width: f64) -> (f64, f64) {
        let dt = 1.0 / 30.0;
        let mut s = 30.0f64;
        // the lane's heading along the bend (deg, clockwise: a right turn)
        let heading = |x: f64| if x < 50.0 { 0.0 } else { ((x - 50.0) / r).to_degrees() as f32 };
        body.place_in_lane(&|d| bend(r, s + d as f64), Some(&|d| heading(s + d as f64)), None, None, speed);
        let centre = DVec2::new(r, 50.0);
        let (mut out, mut inn) = (f64::MIN, f64::MIN);
        let into = 50.0 + r * 1.5 * std::f64::consts::PI;
        while s < into + r * std::f64::consts::PI {
            s += (speed * dt) as f64;
            let at = s;
            body.step_in_lane(dt, speed, &|d| bend(r, at + d as f64), Some(&|d| heading(at + d as f64)), None, None);
            if s < into {
                continue;
            }
            let fwd = dir(body.heading);
            let right = DVec2::new(fwd.y, -fwd.x);
            let o = body.position.truncate();
            for (long, lat) in [(front, half), (front, -half), (-rear, half), (-rear, -half), (body.rot_long, half), (body.rot_long, -half)] {
                let p = o + fwd * long as f64 + right * lat as f64;
                let radius = (p - centre).length();
                out = out.max(radius - (r + lane_width * 0.5));
                inn = inn.max((r - lane_width * 0.5) - radius);
            }
        }
        (out, inn)
    }

    /// The careful driver's line: a 12 m bus and a car in steady bends of 25, 40 and 80 m.
    /// Before (the rotation point on the way) the outer front corner swings out over the lane's
    /// edge (the bus 0.6 m at 40 m); with the line the swept band is centred on the way, its
    /// excess over the lane shared equally by both sides, and the worse of the two sides is
    /// never more than the worse side was before (the inner side may be worse than it was).
    #[test]
    fn the_swept_body_is_centred_on_its_lane_in_a_bend() {
        let lane = 3.5;
        for (name, (def, front, rear, half)) in [("BJ6123", bj6123()), ("car", golf_body())] {
            for (r, speed) in [(25.0f64, 5.0f32), (40.0, 7.0), (80.0, 10.0)] {
                let mut before = AiBody::new(&def, MotionKind::Road);
                let (out0, in0) = swept_outside(&mut before, (front, rear, half), r, speed, lane);
                let mut after = AiBody::new(&def, MotionKind::Road);
                after.keep_swept_in_lane(front, half);
                let (out1, in1) = swept_outside(&mut after, (front, rear, half), r, speed, lane);
                // the band of the geometry: inner side at r_r - half, outer front corner
                let reach = (front - after.rot_long) as f64;
                let rr = r - careful_line_offset(1.0 / r as f32, reach as f32, half, 0.0, f32::INFINITY) as f64;
                let excess = ((rr + half as f64).hypot(reach) - (rr - half as f64) - lane) * 0.5;
                eprintln!("{name} R {r}: before out {out0:.3} in {in0:.3}, after out {out1:.3} in {in1:.3}, band excess {excess:.3}");
                assert!((out1 - in1).abs() < 0.05, "{name} R {r}: after out {out1:.3} in {in1:.3}");
                assert!((out1 - excess).abs() < 0.05 && (in1 - excess).abs() < 0.05, "{name} R {r}: after out {out1:.3} in {in1:.3}, band {excess:.3}");
                assert!(out1.max(in1) <= out0.max(in0) + 1e-3, "{name} R {r}: after {:.3} before {:.3}", out1.max(in1), out0.max(in0));
            }
        }
        // the bus at 40 m: the swung-out front of the rotation point on the way
        let (def, front, rear, half) = bj6123();
        let (out0, _) = swept_outside(&mut AiBody::new(&def, MotionKind::Road), (front, rear, half), 40.0, 7.0, lane);
        assert!(out0 > 0.5, "front {out0:.3} m out");
    }

    /// Where the band is wider than the lane the bus swings wide (K2-D2): BJ6123 in bends of 10,
    /// 15 and 25 m of a 3.5 m lane (edges 1.75 m either side): the inner side never passes the
    /// lane's inner edge, the offset is the one that puts it there (inner edge - half), and the
    /// outer front corner takes the rest of the band - more than centred, no more than with the
    /// rear axle on the way (before). Only the edge on the bend's inside counts (the right in
    /// these right turns): a lane whose right edge lies half the body from its axis keeps the
    /// bus on the way's line whatever its left edge, and one whose left edge does so drives as
    /// the 3.5 m lane.
    #[test]
    fn a_bus_in_a_tight_bend_swings_wide_and_keeps_its_inner_side_in_the_lane() {
        let lane = 3.5;
        let (def, front, rear, half) = bj6123();
        for (r, speed) in [(10.0f64, 3.0f32), (15.0, 4.0), (25.0, 5.0)] {
            let mut before = AiBody::new(&def, MotionKind::Road);
            let (out0, in0) = swept_outside(&mut before, (front, rear, half), r, speed, lane);
            let mut centred = AiBody::new(&def, MotionKind::Road);
            centred.keep_swept_in_lane(front, half);
            let (out_c, in_c) = swept_outside(&mut centred, (front, rear, half), r, speed, lane);
            let mut wide = AiBody::new(&def, MotionKind::Road);
            wide.keep_swept_in_lane(front, half);
            wide.set_lane_edges(lane as f32 * 0.5, lane as f32 * 0.5);
            let (out1, in1) = swept_outside(&mut wide, (front, rear, half), r, speed, lane);
            // the band of the geometry with the offset at its cap
            let reach = (front - wide.rot_long) as f64;
            let off = careful_line_offset(1.0 / r as f32, reach as f32, half, 0.0, lane as f32 * 0.5) as f64;
            let rr = r - off;
            let outer = (rr + half as f64).hypot(reach) - (r + lane * 0.5);
            eprintln!("BJ6123 R {r}: before out {out0:.3} in {in0:.3}; centred out {out_c:.3} in {in_c:.3}; swung wide out {out1:.3} in {in1:.3} (offset {off:.3}, band outer {outer:.3})");
            assert!(in_c > 0.05, "R {r}: the centred band's inner side is past the edge ({in_c:.3})");
            assert!((off - (lane * 0.5 - half as f64)).abs() < 1e-4, "R {r}: offset {off:.3}");
            assert!(in1 < 0.05, "R {r}: inner side {in1:.3} m past the inner edge");
            assert!((out1 - outer).abs() < 0.05, "R {r}: outer {out1:.3}, band {outer:.3}");
            assert!(out1 > out_c, "R {r}: outer side takes the rest ({out1:.3} against centred {out_c:.3})");
            assert!(out1 <= out0 + 0.05, "R {r}: outer {out1:.3} more than with the rear axle on the way ({out0:.3})");
            // the outer edge does not count
            let mut left_tight = AiBody::new(&def, MotionKind::Road);
            left_tight.keep_swept_in_lane(front, half);
            left_tight.set_lane_edges(half, lane as f32 * 0.5);
            let (out2, in2) = swept_outside(&mut left_tight, (front, rear, half), r, speed, lane);
            assert!((out2 - out1).abs() < 1e-6 && (in2 - in1).abs() < 1e-6, "R {r}: ({out2:.3}, {in2:.3}) against ({out1:.3}, {in1:.3})");
        }
        // a lane no wider than the body itself: no room to move in, the way's own line
        let (mut tight, mut plain) = (AiBody::new(&def, MotionKind::Road), AiBody::new(&def, MotionKind::Road));
        tight.keep_swept_in_lane(front, half);
        tight.set_lane_edges(lane as f32 * 0.5, half);
        let a = swept_outside(&mut tight, (front, rear, half), 15.0, 4.0, lane);
        let b = swept_outside(&mut plain, (front, rear, half), 15.0, 4.0, lane);
        assert!((a.0 - b.0).abs() < 0.02 && (a.1 - b.1).abs() < 0.02, "{a:?} against {b:?}");
    }

    /// Entering a bend from the straight (K2-D3): BJ6123 swinging wide in a 3.5 m lane at R 10,
    /// 15 and 25 and a car at R 12, over the whole pass from the straight into the circle: the
    /// body's inner side never passes the lane's inner edge by more than the steady state's
    /// (the steering takes the line's own curvature as it comes, so pure pursuit's look-ahead
    /// does not cut the inside where the bend begins).
    #[test]
    fn entering_a_bend_the_inner_side_keeps_to_the_edge() {
        let lane = 3.5f64;
        for (name, (def, front, rear, half), r, speed) in [("BJ6123", bj6123(), 10.0f64, 3.0f32), ("BJ6123", bj6123(), 15.0, 4.0), ("BJ6123", bj6123(), 25.0, 5.0), ("car", golf_body(), 12.0, 6.0)] {
            let mut body = AiBody::new(&def, MotionKind::Road);
            body.keep_swept_in_lane(front, half);
            body.set_lane_edges(lane as f32 * 0.5, lane as f32 * 0.5);
            let dt = 1.0 / 30.0;
            let mut s = 20.0f64;
            let heading = |x: f64| if x < 50.0 { 0.0 } else { ((x - 50.0) / r).to_degrees() as f32 };
            body.place_in_lane(&|d| bend(r, s + d as f64), Some(&|d| heading(s + d as f64)), None, None, speed);
            let centre = DVec2::new(r, 50.0);
            let mut inn = f64::MIN; let mut out = f64::MIN;
            while s < 50.0 + r * std::f64::consts::FRAC_PI_2 {
                s += (speed * dt) as f64;
                let at = s;
                body.step_in_lane(dt, speed, &|d| bend(r, at + d as f64), Some(&|d| heading(at + d as f64)), None, None);
                let fwd = dir(body.heading);
                let right = DVec2::new(fwd.y, -fwd.x);
                let o = body.position.truncate();
                for (long, lat) in [(front, half), (front, -half), (-rear, half), (-rear, -half), (body.rot_long, half), (body.rot_long, -half)] {
                    let p = o + fwd * long as f64 + right * lat as f64;
                    // (the inner edge: x = lane/2 beside the straight, r - lane/2 round the centre)
                    let past = if p.y < 50.0 { p.x - lane * 0.5 } else { (r - lane * 0.5) - (p - centre).length() };
                    inn = inn.max(past);
                    let po = if p.y < 50.0 { -p.x - lane * 0.5 } else { (p - centre).length() - (r + lane * 0.5) };
                    out = out.max(po);
                }
            }
            eprintln!("{name} R {r}: inner side past the edge at most {inn:.3} m, outer {out:.3}");
            assert!(inn < 0.05, "{name} R {r}: inner side {inn:.3} m past the edge entering the bend");
        }
    }

    /// A straight north to y = 50, a quarter turn right of radius `r` (centre (r, 50)), then a
    /// straight east: the point `s` m along it.
    fn quarter(r: f64, s: f64) -> DVec3 {
        let arc = r * std::f64::consts::FRAC_PI_2;
        if s < 50.0 + arc {
            bend(r, s)
        } else {
            DVec3::new(r + (s - 50.0 - arc), 50.0 + r, 10.0)
        }
    }

    /// Drive `body` through `quarter(r, _)` at `speed` in a lane `lane` m wide with its edges
    /// set: how far the body's inner and outer sides pass the lane's edges (m, + = past) while
    /// the rotation point is before the bend's end and after it: ((inner, outer) entering and
    /// on the arc, (inner, outer) leaving it).
    fn through_a_quarter_turn(body: &mut AiBody, (front, rear, half): (f32, f32, f32), r: f64, speed: f32, lane: f64) -> ((f64, f64), (f64, f64)) {
        let dt = 1.0 / 30.0;
        let arc = r * std::f64::consts::FRAC_PI_2;
        let mut s = 20.0f64;
        let heading = move |x: f64| if x < 50.0 { 0.0 } else if x < 50.0 + arc { ((x - 50.0) / r).to_degrees() as f32 } else { 90.0 };
        body.place_in_lane(&|d| quarter(r, s + d as f64), Some(&|d| heading(s + d as f64)), None, None, speed);
        let centre = DVec2::new(r, 50.0);
        let (mut on, mut off) = ((f64::MIN, f64::MIN), (f64::MIN, f64::MIN));
        while s < 50.0 + arc + 40.0 {
            s += (speed * dt) as f64;
            let at = s;
            body.step_in_lane(dt, speed, &|d| quarter(r, at + d as f64), Some(&|d| heading(at + d as f64)), None, None);
            let fwd = dir(body.heading);
            let right = DVec2::new(fwd.y, -fwd.x);
            let o = body.position.truncate();
            let leaving = s + body.rot_long as f64 >= 50.0 + arc;
            for (long, lat) in [(front, half), (front, -half), (-rear, half), (-rear, -half), (body.rot_long, half), (body.rot_long, -half)] {
                let p = o + fwd * long as f64 + right * lat as f64;
                // (beside the first straight: x = +-lane/2; round the centre; beside the last
                // straight: y = 50 + r -+ lane/2, the inside to the south)
                let (inner, outer) = if p.y < 50.0 {
                    (p.x - lane * 0.5, -p.x - lane * 0.5)
                } else if p.x < r {
                    let d = (p - centre).length();
                    ((r - lane * 0.5) - d, d - (r + lane * 0.5))
                } else {
                    ((50.0 + r - lane * 0.5) - p.y, p.y - (50.0 + r + lane * 0.5))
                };
                let m = if leaving { &mut off } else { &mut on };
                m.0 = m.0.max(inner);
                m.1 = m.1.max(outer);
            }
        }
        (on, off)
    }

    /// Through a bend and out of it (K2-D3): BJ6123 swinging wide in a 3.5 m lane at R 10, 15,
    /// 25 and 40 and a car at R 12, a straight, a quarter turn and a straight: entering and on
    /// the arc the inner side never passes the lane's inner edge, and leaving it the body
    /// comes back to its lane without running over either edge by more than it did on the arc
    /// (the steering takes the line's curvature at the rotation point, `AiBody::drive`; taken
    /// ahead by as far as the vehicle runs while the wheel turns half way to the bend's lock at
    /// its rate, it turned in early and put the BJ6123's inner side 0.43 m past at R 10).
    #[test]
    fn through_a_bend_and_out_the_inner_side_keeps_to_the_edge() {
        let lane = 3.5f64;
        for (name, body_def, r, speed) in [("BJ6123", bj6123(), 10.0f64, 3.0f32), ("BJ6123", bj6123(), 15.0, 4.0), ("BJ6123", bj6123(), 25.0, 5.0), ("BJ6123", bj6123(), 40.0, 7.0), ("BJ6123", bj6123(), 40.0, 11.0), ("car", golf_body(), 12.0, 6.0), ("car", golf_body(), 25.0, 10.0), ("car", golf_body(), 60.0, 15.0)] {
            let (def, front, rear, half) = body_def;
            let mut body = AiBody::new(&def, MotionKind::Road);
            body.keep_swept_in_lane(front, half);
            body.set_lane_edges(lane as f32 * 0.5, lane as f32 * 0.5);
            let (on, off) = through_a_quarter_turn(&mut body, (front, rear, half), r, speed, lane);
            eprintln!("{name} R {r} at {speed} m/s: entering and on the arc inner {:.3} outer {:.3}; leaving inner {:.3} outer {:.3}", on.0, on.1, off.0, off.1);
            assert!(on.0 < 0.05, "{name} R {r}: inner side {:.3} m past the edge entering the bend", on.0);
            assert!(off.0 < 0.05, "{name} R {r}: inner side {:.3} m past the edge leaving the bend", off.0);
            assert!(off.1 <= on.1.max(0.0) + 0.05, "{name} R {r}: outer side {:.3} m past the edge leaving the bend, {:.3} on it", off.1, on.1);
        }
    }

    /// The careful driver's line takes the lane's curvature only up to the way's own and 0
    /// where their signs differ: a lane heading that turns left while the way's points go
    /// straight or right, and one that turns more than the way, do not move the body.
    #[test]
    fn the_lane_bend_counts_only_as_far_as_the_way_has_it() {
        let (def, front, _, half) = bj6123();
        let straight = |d: f32| DVec3::new(0.0, d as f64, 0.0);
        let right = |d: f32| bend(40.0, 50.0 + d as f64);
        let place = |way: &dyn Fn(f32) -> DVec3, lane: &dyn Fn(f32) -> f32| {
            let mut b = AiBody::new(&def, MotionKind::Road);
            b.keep_swept_in_lane(front, half);
            b.place_in_lane(way, Some(lane), None, None, 5.0);
            b.rear
        };
        // headings turning right 40 m radius / left 40 m radius / right 10 m radius
        let r40 = |d: f32| (d as f64 / 40.0).to_degrees() as f32;
        let l40 = |d: f32| -(d as f64 / 40.0).to_degrees() as f32;
        let r10 = |d: f32| (d as f64 / 10.0).to_degrees() as f32;
        let north = |_d: f32| 0.0f32;
        // a straight way: a lane bend of either sign moves nothing
        let on_way = place(&straight, &north);
        assert_eq!(place(&straight, &r40), on_way);
        assert_eq!(place(&straight, &l40), on_way);
        // a way bending right: a lane bending left moves nothing, the same bend moves it
        // inward, a tighter lane bend (10 m) no more than the way's (40 m)
        let way_alone = place(&right, &north);
        assert_eq!(place(&right, &l40), way_alone);
        let same = place(&right, &r40);
        assert!((same - way_alone).length() > 0.3, "a 40 m bend moves the body {:.3} m", (same - way_alone).length());
        assert!((place(&right, &r10) - same).length() < 1e-3, "{:?} against {:?}", place(&right, &r10), same);
    }

    /// The installed BJ6180: its front section and the rear section hung on it (None when
    /// BeijingCity is not installed).
    fn bj6180() -> Option<(crate::vehicle::VehicleType, crate::vehicle::VehicleType)> {
        let root = omsi_cfg::env::var_os("OMSI_ROOT")
            .map(std::path::PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join("Game/omsi2")))?;
        let load = |f: &str| {
            let path = root.join("Vehicles/BJ6180_BJC").join(f);
            path.exists().then(|| crate::vehicle::VehicleType::load_ai(&root, &path).expect("the bus loads"))
        };
        Some((load("6180.bus")?, load("6180_T.bus")?))
    }

    /// `swept_outside` for an articulated bus: the (outward, inward) of the front section and
    /// of the rear section, which hangs on its joint (its rotation point `a` behind the joint
    /// is pulled towards it, the joint `d` behind the front's rotation point).
    fn swept_outside_articulated(body: &mut AiBody, front_box: (f32, f32, f32), rear_box: (f32, f32), (d, a): (f32, f32), r: f64, speed: f32, lane_width: f64) -> [(f64, f64); 2] {
        let (front, rear, half) = front_box;
        let (t_front, t_rear) = rear_box;
        let dt = 1.0 / 30.0;
        let mut s = 30.0f64;
        let heading = |x: f64| if x < 50.0 { 0.0 } else { ((x - 50.0) / r).to_degrees() as f32 };
        body.place_in_lane(&|d| bend(r, s + d as f64), Some(&|d| heading(s + d as f64)), None, None, speed);
        let centre = DVec2::new(r, 50.0);
        let joint_of = |b: &AiBody| b.rear - dir(b.heading) * d as f64;
        let mut axle = joint_of(body) - dir(body.heading) * a as f64;
        let mut worst = [(f64::MIN, f64::MIN); 2];
        let into = 50.0 + r * 1.5 * std::f64::consts::PI;
        while s < into + r * std::f64::consts::PI {
            s += (speed * dt) as f64;
            let at = s;
            body.step_in_lane(dt, speed, &|d| bend(r, at + d as f64), Some(&|d| heading(at + d as f64)), None, None);
            // the rear section: its rotation point is pulled towards the joint
            let joint = joint_of(body);
            axle = joint - (joint - axle).normalize() * a as f64;
            if s < into {
                continue;
            }
            let fwd = dir(body.heading);
            let right = DVec2::new(fwd.y, -fwd.x);
            let o = body.position.truncate();
            let t_fwd = (joint - axle).normalize();
            let t_right = DVec2::new(t_fwd.y, -t_fwd.x);
            let sections = [
                (o, fwd, right, [(front, half), (front, -half), (-rear, half), (-rear, -half), (body.rot_long, half), (body.rot_long, -half)]),
                (axle, t_fwd, t_right, [(t_front, half), (t_front, -half), (-t_rear, half), (-t_rear, -half), (0.0, half), (0.0, -half)]),
            ];
            for (i, (origin, f, rt, corners)) in sections.iter().enumerate() {
                for &(long, lat) in corners {
                    // (the rear section's lengths are from its rotation point)
                    let p = *origin + *f * long as f64 + *rt * lat as f64;
                    let radius = (p - centre).length();
                    worst[i].0 = worst[i].0.max(radius - (r + lane_width * 0.5));
                    worst[i].1 = worst[i].1.max((r - lane_width * 0.5) - radius);
                }
            }
        }
        worst
    }

    /// The careful driver's line for the whole of an articulated bus, the installed BJ6180
    /// (`Vehicles/BJ6180_BJC`, a front section and a rear section on a joint) in steady bends
    /// of 25, 40 and 80 m of a 3.5 m lane: the swept band runs from the rear section's
    /// innermost point to the front's outer corner and is centred, so the two sides leave
    /// the lane by the same amount; with the front section alone centred the rear section's
    /// off-tracking leaves the inner edge by more than the front the outer one.
    #[test]
    fn an_articulated_bus_is_centred_on_its_lane_as_a_whole() {
        let Some((front_ty, rear_ty)) = bj6180() else {
            eprintln!("skipped: no BJ6180");
            return;
        };
        let lane = 3.5;
        let half = front_ty.def.bounding_box.unwrap()[0] * 0.5;
        let boxed = |ty: &crate::vehicle::VehicleType| {
            let bb = ty.def.bounding_box.unwrap();
            (bb[1] * 0.5 + bb[4], bb[1] * 0.5 - bb[4])
        };
        let (f_front, f_rear) = boxed(&front_ty);
        let (t_front, t_rear) = boxed(&rear_ty);
        let rot_t = rotation_point(&rear_ty.def).0;
        // the lengths of the rear section from its rotation point
        let rear_box = (t_front - rot_t, t_rear + rot_t);
        let (back, joint_front) = crate::vehicle::coupling_points(&front_ty, false, &rear_ty, false);
        let dims = (rotation_point(&front_ty.def).0 - back.y, joint_front.y - rot_t);
        let pull = rear_off_tracking(&[&front_ty, &rear_ty]);
        assert!((pull - (dims.1 * dims.1 - dims.0 * dims.0)).abs() < 1e-3, "pull {pull}");
        println!("BJ6180: joint {:.3} m behind the front's rotation point, the rear's {:.3} m behind it; pull {pull:.2} m2", dims.0, dims.1);
        println!("R   | rule    | front out / in   | rear out / in");
        for (r, speed) in [(25.0f64, 5.0f32), (40.0, 7.0), (80.0, 10.0)] {
            let mut rows = Vec::new();
            for with in [false, true] {
                let mut body = AiBody::new(&front_ty.def, MotionKind::Road);
                if with {
                    body.keep_swept_in_lane_with(f_front, half, &[&front_ty, &rear_ty]);
                } else {
                    body.keep_swept_in_lane(f_front, half);
                }
                let w = swept_outside_articulated(&mut body, (f_front, f_rear, half), rear_box, dims, r, speed, lane);
                println!("{r:<3} | {} | {:+.3} / {:+.3} | {:+.3} / {:+.3}", if with { "whole  " } else { "front  " }, w[0].0, w[0].1, w[1].0, w[1].1);
                rows.push(w);
            }
            let (alone, whole) = (rows[0], rows[1]);
            // the whole vehicle's band: the front's outer corner against the rear's inner side
            let inner = whole[0].1.max(whole[1].1);
            assert!((whole[0].0 - inner).abs() < 0.06, "R {r}: out {:.3} in {inner:.3}", whole[0].0);
            // never worse on either side than the front section alone centred
            assert!(whole[0].0.max(inner) <= alone[0].0.max(alone[0].1.max(alone[1].1)) + 1e-3, "R {r}");
            assert!(inner < alone[0].1.max(alone[1].1), "R {r}: the rear section's inner side moves out");
        }
    }

    /// On a straight lane the careful driver's line is the way: the pose is the same, bit
    /// for bit, as a body driven on the way without it (the same lane given to both) - on a
    /// straight way, and on a pull-out's S-curve (round a bus standing ahead) laid on a
    /// straight lane, whose bend is not the lane's.
    #[test]
    fn on_a_straight_lane_the_line_is_the_way() {
        let (def, front, _, half) = bj6123();
        let straight = |d: f32| DVec3::new(0.0, d as f64, 0.0);
        let north = |_d: f32| 0.0f32;
        let pull_out = straight_pull_out(3.3, 12.0);
        for way in [&straight as &dyn Fn(f32) -> DVec3, &pull_out] {
            let mut plain = AiBody::new(&def, MotionKind::Road);
            let mut line = AiBody::new(&def, MotionKind::Road);
            line.keep_swept_in_lane(front, half);
            plain.place_in_lane(way, Some(&north), None, None, 0.0);
            line.place_in_lane(way, Some(&north), None, None, 0.0);
            let (mut s, mut v) = (0.0f32, 0.0f32);
            for _ in 0..600 {
                v = (v + 0.05).min(12.0);
                s += v / 30.0;
                let at = s;
                plain.step_in_lane(1.0 / 30.0, v, &|d| way(at + d), Some(&north), None, None);
                line.step_in_lane(1.0 / 30.0, v, &|d| way(at + d), Some(&north), None, None);
                assert_eq!((plain.position, plain.heading, plain.steer), (line.position, line.heading, line.steer));
            }
        }
    }

    #[test]
    fn backs_up_straight_along_its_way() {
        // [Q8-1] the planner's place moves back 4 m at walking pace (negative speed): the body
        // follows backwards, keeps its heading and does not jump
        let def = golf();
        let mut body = AiBody::new(&def, MotionKind::Road);
        let way = |s: f64| move |d: f32| DVec3::new(0.0, s + d as f64, 10.0);
        let mut s = 20.0f64;
        body.place(&way(s), None, None, 0.0);
        let dt = 1.0 / 30.0;
        let y0 = body.position.y;
        let mut worst_step = 0.0f64;
        let mut last = body.position;
        for k in 0..240 {
            let v = if k < 120 { -crate::traffic::REVERSE_SPEED } else { 0.0 };
            if s > 16.0 {
                s = (s + (v * dt) as f64).max(16.0);
            }
            let speed = if s > 16.0 { v } else { 0.0 };
            body.step(dt, speed, &way(s), None, None);
            worst_step = worst_step.max((body.position - last).truncate().length());
            last = body.position;
        }
        assert!((body.position.y - (y0 - 4.0)).abs() < 0.2, "backed to {:.2} from {y0:.2}", body.position.y);
        assert!(body.position.x.abs() < 0.05, "drifted {:.2} m sideways", body.position.x);
        assert!(body.heading.rem_euclid(360.0) < 1.0 || body.heading.rem_euclid(360.0) > 359.0, "heading {}", body.heading);
        assert!(worst_step < 0.1, "jumped {worst_step:.2} m in a frame");
    }

    #[test]
    fn brakes_nose_down_and_rolls_out_of_a_turn() {
        let def = golf();
        let mut body = AiBody::new(&def, MotionKind::Road);
        let flat = |_x: f64, _y: f64| Some(10.0);
        body.place(&|d| DVec3::new(0.0, d as f64, 10.0), Some(&flat), None, 10.0);
        let dt = 1.0 / 30.0;
        let mut speed = 10.0f32;
        let mut s = 0.0f64;
        let mut min_pitch = 0.0f32;
        for _ in 0..60 {
            speed = (speed - 3.0 * dt).max(0.0);
            s += (speed * dt) as f64;
            let at = s;
            body.step(dt, speed, &|d| DVec3::new(0.0, at + d as f64, 10.0), Some(&flat), None);
            min_pitch = min_pitch.min(body.pitch_deg);
        }
        assert!(min_pitch < -0.1, "braking pitch {min_pitch}");
        assert!((body.position.z - 10.0).abs() < 0.1);
        // the front wheels come up in their arches while the nose dives
        assert!(body.suspension[0][0] < 0.0 && body.suspension[1][0] > 0.0, "{:?}", body.suspension);
    }

    /// A two-axle lorry: 4.5 m between the axles, a turning circle of 11 m radius.
    fn lorry() -> Vehicle {
        let mut v = Vehicle { mass: 12.0, moment_of_inertia: [40.0, 10.0, 45.0], cog_height: 1.2, rot_pnt_long: -2.0, inv_min_turn_radius: 0.09, ..Default::default() };
        v.axles = vec![Axle { long: 2.5, max_width: 2.0, spring: 300.0, damper: 20.0, wheel_diameter: 1.0, ..Default::default() }, Axle { long: -2.0, max_width: 1.8, spring: 400.0, damper: 25.0, wheel_diameter: 1.0, ..Default::default() }];
        v
    }

    /// A standing bus `half_width` × 2 wide, `gap` metres ahead of a vehicle whose front is `front`
    /// metres ahead of its origin at (0, 0) facing north.
    fn bus_ahead(front: f32, gap: f32, half_width: f64) -> crate::collision::Obb {
        crate::collision::Obb::vehicle(DVec2::new(0.0, (front + gap) as f64 + 6.0), 0.0, 6.0, 6.0, half_width)
    }

    #[test]
    fn a_car_too_close_behind_a_bus_cannot_steer_round_it() {
        let def = golf();
        let extent = (2.1, 2.2, 0.85);
        let best = |gap: f32| {
            pull_out_ramps(gap, extent.0, false)
                .into_iter()
                .map(|ramp| {
                    let way = straight_pull_out(3.3, ramp);
                    let mut body = AiBody::new(&def, MotionKind::Road);
                    body.place(&way, None, None, 0.0);
                    body.sweep_clearance(&way, 0.0, PULL_OUT_WAIT, PULL_OUT_ACCEL, 8.0, gap + 6.0, extent, &[bus_ahead(extent.0, gap, 1.25)])
                })
                .fold(f64::MIN, f64::max)
        };
        // half a metre behind the bus no S-curve gets the front corner past (the lock
        // is 25°, the steering takes a moment to get there); six metres back it is easy
        let close = best(0.5);
        assert!(close < PULL_OUT_CLEARANCE, "clearance {close} from 0.5 m");
        let far = best(6.0);
        assert!(far >= PULL_OUT_CLEARANCE, "clearance {far} from 6 m");
        // and the swept body does leave the lane: after the pull-out it is 3.3 m over
        let way = straight_pull_out(3.3, 8.0);
        let mut body = AiBody::new(&def, MotionKind::Road);
        body.place(&way, None, None, 0.0);
        let mut x = 0.0f32;
        let dt = 1.0 / 20.0;
        for _ in 0..400 {
            x += 4.0 * dt;
            let at = x;
            body.step(dt, 4.0, &|d| way(at + d), None, None);
        }
        assert!((body.position.x + 3.3).abs() < 0.2, "ended at x {}", body.position.x);
    }

    /// A lane change's S-curve is steered by pure pursuit, not fed forward: a car pulling
    /// away from a standstill into the next lane 3.1 m over in 4 m (the game's case at
    /// Jiangzhaikou, a car changing lanes at a red light) runs past the new lane's axis by
    /// 1.9 m when the S's curvature is fed forward (the rate-limited wheel lags the S's turn
    /// back) and stands there beside the car in that lane; steered by pure pursuit it comes to
    /// rest within `PULL_OUT_CLEARANCE` of the axis and never passes it.
    #[test]
    fn a_lane_change_s_curve_does_not_overshoot_the_new_lane() {
        let def = golf();
        let north = |_d: f32| 0.0f32;
        let side = 3.1f32;
        let way = straight_pull_out(side, 4.0);
        let run = |lane: Option<&dyn Fn(f32) -> f32>| {
            let mut body = AiBody::new(&def, MotionKind::Road);
            body.place_in_lane(&way, lane, None, None, 0.0);
            let (dt, mut s, mut v, mut past) = (1.0 / 30.0, 0.0f32, 0.0f32, f64::MIN);
            for i in 0..240 {
                // pulling away at 0.8 m/s² up to 3.3 m/s for 5 s, then standing
                v = if i < 150 { (v + 0.8 * dt).min(3.3) } else { 0.0 };
                s += v * dt;
                let at = s;
                body.step_in_lane(dt, v, &|d| way(at + d), lane, None, None);
                past = past.max(-body.rear.x - side as f64);
            }
            (past, -body.rear.x - side as f64)
        };
        let (past_pp, rest_pp) = run(Some(&north));
        let (past_ff, rest_ff) = run(None);
        eprintln!("lane change: past the new lane's axis at most {past_pp:.3} m, at rest {rest_pp:.3} (fed forward {past_ff:.3}, {rest_ff:.3})");
        assert!(past_pp <= 0.0, "past the new lane's axis by {past_pp:.3} m");
        assert!(rest_pp.abs() < PULL_OUT_CLEARANCE, "at rest {rest_pp:.3} m off the new lane's axis");
        assert!(past_ff > 1.0, "the fed-forward S no longer overshoots ({past_ff:.3}): the test lost its case");
    }

    #[test]
    fn room_to_pull_out_depends_on_the_steering() {
        let car = pull_out_room(&golf(), (2.1, 2.2, 0.85), 1.25, 3.3);
        let lorry = pull_out_room(&lorry(), (3.6, 3.4, 1.25), 1.25, 3.3);
        // a bus standing a little further over (or wider) needs more room; the lorry, with its
        // longer wheelbase and smaller lock, more again
        let wide = pull_out_room(&golf(), (2.1, 2.2, 0.85), 1.6, 3.3);
        eprintln!("room: car {car} wide {wide} lorry {lorry}");
        assert!(wide > car, "{wide} against {car}");
        assert!(car > 1.5 && car < 6.0, "car {car}");
        assert!(lorry > car + 0.5 && lorry < 12.5, "lorry {lorry}, car {car}");
        // what it found works: from that room less the half metre of slack one of the
        // curves clears, from a metre less none does
        let clears = |gap: f32| {
            pull_out_ramps(gap, 2.1, false).into_iter().any(|ramp| {
                let way = straight_pull_out(3.3, ramp);
                let mut body = AiBody::new(&golf(), MotionKind::Road);
                body.place(&way, None, None, 0.0);
                body.sweep_clearance(&way, 0.0, PULL_OUT_WAIT, PULL_OUT_ACCEL, 8.0, gap + 6.0, (2.1, 2.2, 0.85), &[bus_ahead(2.1, gap, 1.25)]) >= PULL_OUT_CLEARANCE
            })
        };
        assert!(clears(car - 0.5));
        assert!(!clears(car - 1.0));
    }

    #[test]
    fn pull_out_curves_run_gentle_to_steep() {
        let r = pull_out_ramps(3.0, 2.0, false);
        assert!(r.windows(2).all(|w| w[0] > w[1]), "{r:?}");
        assert!(r.iter().all(|&x| (4.0..=12.0).contains(&x)), "{r:?}");
        let rolling = pull_out_ramps(30.0, 2.0, true);
        assert_eq!(rolling[0], 20.0);
        assert!(rolling.iter().all(|&x| (4.0..=20.0).contains(&x)), "{rolling:?}");
    }

    /// Every wheel stands on the face drawn under it, as Omsi.exe's ground query stands an
    /// AI car's (the highest face under a ray from over the wheel; here from `AI_STEP_UP`
    /// over the lane, Omsi.exe's 3 m lifted the cars onto bridge decks): a road drawn 8 cm
    /// over the lane and cambered across it carries each wheel at its own height, and a lane
    /// laid 0.9 m under its road has the car on the road, not inside it.
    #[test]
    fn each_wheel_stands_on_the_face_under_it() {
        let def = golf();
        for (lift, camber) in [(0.08f64, 0.03f64), (0.9, 0.0), (0.08, -0.05)] {
            let road = move |x: f64, _y: f64| 10.0 + lift + camber * x;
            let ground = move |x: f64, y: f64, top: f64| {
                let z = road(x, y);
                if z <= top { crate::rigid::GroundProbe { below: Some(z), above: None } } else { crate::rigid::GroundProbe { below: Some(9.0), above: Some(z) } }
            };
            let way = |d: f32| DVec3::new(0.0, d as f64, 10.0);
            let mut body = AiBody::new(&def, MotionKind::Road);
            body.place(&way, None, Some(&ground), 0.0);
            let mut x = 0.0f32;
            for _ in 0..90 {
                x += 0.2;
                let at = x;
                body.step(1.0 / 30.0, 6.0, &|d| way(at + d), None, Some(&ground));
            }
            for (w, z) in body.wheels.iter().zip(&body.contact_z) {
                let p = body.position.truncate() + DVec2::new(w.lat as f64, w.long as f64);
                assert!((z - road(p.x, p.y)).abs() < 0.01, "lift {lift} camber {camber}: wheel at {z:.3}, road {:.3}", road(p.x, p.y));
            }
            assert!((body.position.z - (10.0 + lift)).abs() < 0.02, "lift {lift} camber {camber}: body at {:.3}", body.position.z);
            let lean = (-camber).atan().to_degrees() as f32;
            assert!((body.bank_deg - lean).abs() < 0.3, "camber {camber}: bank {} against {lean}", body.bank_deg);
        }
    }

    /// A lane laid half a metre over its drawn road (a junction's paths): with no wheel
    /// near the lane's height the car comes down onto the road instead of driving in the air;
    /// a road far below (under a bridge the lane crosses) still does not pull it down.
    #[test]
    fn a_car_over_a_lower_road_comes_down_onto_it() {
        let def = golf();
        for (drop, expect) in [(0.5f64, 9.5f64), (6.0, 10.0)] {
            let ground = move |_x: f64, _y: f64, top: f64| {
                let z = 10.0 - drop;
                crate::rigid::GroundProbe { below: (z <= top).then_some(z), above: None }
            };
            let way = |d: f32| DVec3::new(0.0, d as f64, 10.0);
            let mut body = AiBody::new(&def, MotionKind::Road);
            body.place(&way, None, Some(&ground), 0.0);
            let mut x = 0.0f32;
            for _ in 0..60 {
                x += 0.2;
                let at = x;
                body.step(1.0 / 30.0, 6.0, &|d| way(at + d), None, Some(&ground));
            }
            assert!((body.position.z - expect).abs() < 0.02, "drop {drop}: body at {:.3}", body.position.z);
        }
    }

    #[test]
    fn aircraft_keep_their_height() {
        let mut def = golf();
        def.axles.truncate(1);
        let mut body = AiBody::new(&def, MotionKind::Air);
        let way = |d: f32| DVec3::new(0.0, d as f64, 300.0 - d as f64 * 0.05);
        body.place(&way, None, None, 70.0);
        body.step(1.0 / 30.0, 70.0, &way, None, None);
        assert!((body.position.z - 300.0).abs() < 1e-6);
        assert!(body.pitch_deg < -2.0 && body.pitch_deg > -4.0, "pitch {}", body.pitch_deg);
    }
}
