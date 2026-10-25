//! AI road traffic: vehicles from `ailists.cfg` moving on the map's path network, the
//! traffic light programs of the junctions, and the population of cars around the player.

use crate::bus_service::{BusService, Phase};
use crate::scene::{VehicleRender, World};
use anyhow::Result;
use glam::{DVec2, DVec3};
use hashbrown::HashMap;
use omsi_render::{Renderer, Scene};
use omsi_sim::ai_motion::{AiBody, MotionKind};
use omsi_sim::traffic::{AiState, Aspect, LaneKind, Network, TrafficLightController};
use omsi_sim::vehicle::AiFrame;
use omsi_sim::{VehicleInstance, VehicleType};
use omsi_traffic::{External, Parked, Person, PlayerView, Seen, Vehicle as AiVehicle, VehicleClass, Why};
use std::path::Path;
use std::sync::Arc;

/// How many of a type's paint schemes the AI uses: every scheme is a full upload of the
/// bus's textures the first time it appears, which used to cost a frame of 100-200 ms
/// each and a minute of stutter after loading Spandau.
pub const AI_SCHEMES: usize = 4;

/// The share of the textures' budget the vehicles' textures (AI traffic, the timetable) may take
/// before the paints nobody wears are no longer kept on the GPU: they are never shrunk, so
/// what they hold the scenery lacks.
pub const VEHICLE_SHARE: f64 = 0.30;
const SCRIPT_UPLOAD_BUDGET: usize = 4 << 20;
const AI_JOB_SECS: f32 = 50e-6;

/// Start a vehicle of type `ty` once and throw it away: its `{init}` and its displays read
/// the files they need (depot data, fonts) into the caches before the first real one of the
/// type comes along in the middle of a drive.
pub fn warm_up(world: &World, ty: &Arc<VehicleType>, hof: Option<Arc<omsi_vehicle::Hof>>) {
    let t = std::time::Instant::now();
    let mut host = omsi_sim::VehicleHost::new(omsi_sim::SimClock::default());
    host.hof = hof;
    host.font_lib = Some(world.fonts.clone());
    let mut vehicle = VehicleInstance::new(ty.clone(), host);
    vehicle.init_text_textures(&mut world.fonts.lock(), &|p| {
        omsi_texture::decode_file(p)
            .ok()
            .map(|i| (i.width, i.height, i.rgba))
    });
    if omsi_cfg::env::var_os("OMSI_PROFILE").is_some() {
        log::info!(
            "  first start of {}: {:.1} ms",
            ty.def.path.display(),
            t.elapsed().as_secs_f64() * 1000.0
        );
    }
}

/// The frame's cars by id, and the cars each one leads (the cars whose `last.lead_id` it is, in
/// car order), built once before the per-car loop of `Traffic::tick`: the lookups there were
/// a scan of every car per bus (and per car queued behind it); the answers are the same ones,
/// found by the id (`lead_id` is set by `Simulation::tick`, not inside that loop).
#[derive(Default)]
struct CarLinks {
    by_id: HashMap<u64, usize>,
    followers: HashMap<u64, Vec<usize>>,
}

impl CarLinks {
    fn of(cars: &[AiCar]) -> Self {
        let mut links = CarLinks::default();
        for (k, c) in cars.iter().enumerate() {
            // (the first car of an id, as `iter().find` took it)
            links.by_id.entry(c.id).or_insert(k);
            if let Some(lead) = c.ai.last.lead_id {
                links.followers.entry(lead).or_default().push(k);
            }
        }
        links
    }

    /// The indices of the cars that have `id` for their lead, in car order.
    fn following(&self, id: u64) -> &[usize] {
        self.followers.get(&id).map_or(&[], |v| v.as_slice())
    }
}

/// Frames between two lines of the traffic profile (OMSI_PROFILE): 30 s at the 30 Hz of the
/// offscreen run.
const TICK_PROFILE_FRAMES: u32 = 900;

/// The chance per population pass (about every 2 s) that a car near the player parks in a
/// free space (`Traffic::park_in`; the value since the first release, by eye).
const PARK_IN_CHANCE: f64 = 0.04;
/// The chance per population pass (about every 2 s) that a parked car drives off
/// (`Traffic::pull_out_parked`): about one car a minute.
const PULL_OUT_CHANCE: f64 = 0.035;

/// How far ahead an emergency vehicle warns what holds it up (`TrafficPriorityWarningNeeded`,
/// m).
const PRIORITY_WARN_GAP: f32 = 60.0;

/// [round 7 B5] The removal rules of `Traffic::populate_seen`, named (their reasons are the
/// comments there). A car finished (given up, or a timetable bus let go) on the last lane of
/// the network goes after standing this long (s): at once, as Omsi.exe deletes a car with no
/// next segment the same frame (0x6fe3fc) ...
const AT_END_STAND: f32 = 0.5;
/// ... a timetable bus let go there after this long (s).
const AT_END_STAND_BUS: f32 = 20.0;
/// Standing this long (s) beyond `FAR_OFF_DIST` (m) from the eye, too far for the
/// renderer to draw it plainly: a bus stood with its passengers at the far end of a
/// straight road for good (r4d). For a timetable bus at its loaded route's end (`at_edge`)
/// and the dead-end rule only ([round 8 H8-7 fix pass 2] not for a car at the loaded tiles'
/// edge, which goes by the round-7 in-view rule).
const FAR_OFF_STAND: f32 = 8.0;
const FAR_OFF_DIST: f64 = 180.0;
/// Standing this long (s) at a dead end or the loaded route's end goes in any case.
const STOOD_LONG: f32 = 150.0;
/// At a dead end with a queue behind it, in view too after this long (s), beyond
/// `QUEUE_BEHIND_DIST` (m): a fire engine at the end of a dead-end street held fourteen cars
/// for two and a half minutes.
const QUEUE_BEHIND_STAND: f32 = 25.0;
const QUEUE_BEHIND_DIST: f64 = 25.0;
// ([round 7 R7-1 fix 1] IN_VIEW_GRACE, GAVE_UP_DIST, QUEUE_LEAD, `in_a_queue` and `lead_of`
// are the crate's, `omsi_traffic::removal`: the free-run replay asks the same rule)
use omsi_traffic::removal::{goes_far, goes_in_view, lead_of, in_a_queue, RemoveRule};
/// A random car goes out of range beyond `VISIBLE_RANGE` times this (and `NEAR_HIDE` from the
/// eye): a margin over what is drawn.
const OUT_OF_RANGE: f64 = 1.3;

#[cfg(test)]
mod spawn_queue_tests {
    use super::{may_put_on, queue_within_reach, spawn_queue_reach};

    /// A crossing object (9) with two pieces across each other, fed by plain roads, one of
    /// them lit: the lit approach and the piece with a foe are refused, a plain road is taken.
    #[test]
    fn b4_no_car_is_put_on_a_lit_lane_or_a_junction_piece_with_foes() {
        use omsi_sim::traffic::{LaneBuilder, LaneKey, LaneKind, Network};
        let lane = |pts: &[(f64, f64)], source: u32, id: i64, path: u16| {
            let mut l = LaneBuilder::polyline(pts.iter().map(|p| glam::DVec3::new(p.0, p.1, 0.0)).collect(), LaneKind::Street, 3.5);
            l.source = source;
            l.key = Some(LaneKey { tile: (0, 0), id, path });
            l
        };
        let mut net = Network::default();
        let mut lit = lane(&[(-200.0, 0.0), (-15.0, 0.0)], 1, 1, 0);
        lit.traffic_light = Some((0, 0));
        net.lanes.push(lit); // 0: the lit approach
        net.lanes.push(lane(&[(-15.0, 0.0), (15.0, 0.0)], 2, 9, 0)); // 1: across
        net.lanes.push(lane(&[(15.0, 0.0), (200.0, 0.0)], 1, 2, 0)); // 2: plain
        net.lanes.push(lane(&[(0.0, -200.0), (0.0, -15.0)], 1, 3, 0)); // 3: plain
        net.lanes.push(lane(&[(0.0, -15.0), (0.0, 15.0)], 2, 9, 1)); // 4: across the other way
        net.lanes.push(lane(&[(0.0, 15.0), (0.0, 200.0)], 1, 4, 0)); // 5: plain
        net.link(1.5);
        net.compute_conflicts();
        let js = omsi_traffic::Junctions::compile(&net);
        assert!(js.movement(1).is_some_and(|(_, m)| !m.foes.is_empty()), "the pieces meet");
        assert!(!may_put_on(&net, &js, 0), "the lit approach");
        assert!(!may_put_on(&net, &js, 1) && !may_put_on(&net, &js, 4), "a piece with a foe");
        assert!(may_put_on(&net, &js, 2) && may_put_on(&net, &js, 3), "a plain road");
    }
    use omsi_sim::traffic::{LaneBuilder, LaneKind, Network};

    /// Two 100 m lanes end to end: a standing car's rear 20 m ahead of the spot on the next
    /// lane keeps a car going 10 m/s from being put down (reach 40 m); a moving one does not,
    /// and hides one standing behind it; one standing 40 m ahead does not; one behind the spot
    /// does not count.
    #[test]
    fn b4_no_car_is_put_down_within_reach_of_a_standing_queue() {
        let p = |y: f64| glam::DVec3::new(0.0, y, 0.0);
        let mut net = Network { lanes: vec![LaneBuilder::polyline(vec![p(0.0), p(100.0)], LaneKind::Street, 3.5), LaneBuilder::polyline(vec![p(100.0), p(200.0)], LaneKind::Street, 3.5)], ..Default::default() };
        net.link(1.5);
        let reach = spawn_queue_reach(10.0);
        assert!((reach - 40.0).abs() < 0.01, "{reach}");
        let at = |cars: Vec<(usize, f32, bool)>| move |lane: usize| cars.iter().filter(|c| c.0 == lane).map(|c| (c.1, c.1 - 2.0, c.2)).collect::<Vec<_>>();
        assert!(queue_within_reach(&net, 0, 90.0, reach, &at(vec![(1, 12.0, true)])));
        assert!(!queue_within_reach(&net, 0, 90.0, reach, &at(vec![(1, 12.0, false), (1, 20.0, true)])));
        assert!(!queue_within_reach(&net, 0, 90.0, reach, &at(vec![(1, 32.0, true)])));
        assert!(!queue_within_reach(&net, 0, 90.0, reach, &at(vec![(0, 50.0, true)])));
        assert!(queue_within_reach(&net, 0, 50.0, reach, &at(vec![(0, 60.0, true)])));
    }
}

// ([round 7 R7-1 fix 1] the queue tests moved with the rule to omsi_traffic::removal)

/// [round 9 A1] How many of the cars `(position, timetable bus, on a lane of the kind filled)`
/// count against a target: every random car on the kind's lanes ([round 7 B2] a car that gave
/// up too: it stands on the road), within `near` (centre, radius) when given - near the centre
/// it holds the budget, far off it does not.
fn budget_count(cars: impl Iterator<Item = (DVec3, bool, bool)>, near: Option<(DVec3, f64)>) -> usize {
    cars.filter(|&(p, scheduled, of_kind)| !scheduled && of_kind && near.is_none_or(|(c, r)| (p - c).length() < r)).count()
}

/// A vehicle type's half length (m), the radius the renderer's object distance is reckoned with
/// (`Viewer::draws`): `VehicleType::half_length`, else `DEFAULT_BOX`'s.
fn body_radius(ty: &VehicleType) -> f64 {
    ty.half_length().map_or(DEFAULT_BOX[1] as f64 * 0.5, |h| h as f64)
}

/// [round 9 A2 fix pass 1] A vehicle type's body height (m), the size its perceptual horizon
/// is reckoned for: its `[boundingbox]`'s, else its model's, else `DEFAULT_BOX`'s (bus-sized).
fn body_height(ty: &VehicleType) -> f64 {
    match ty.def.bounding_box {
        Some(bb) if bb[2] > 0.5 => bb[2] as f64,
        _ => ty.model_box().map(|(lo, hi)| (hi.z - lo.z) as f64).filter(|h| *h > 0.5).unwrap_or(DEFAULT_BOX[2] as f64),
    }
}

/// [round 9 A2 fix pass 1] The random cars `(id, standing at the loaded tiles' edge, beyond its
/// horizon, its state)` whose trip ends at the edge now (`removal::goes_far`): only the one at
/// the edge, first in line; the queue behind it drives up and each car goes as it arrives.
fn far_trip_ends<'a>(cars: impl Iterator<Item = (u64, bool, bool, &'a omsi_traffic::Vehicle)>) -> std::collections::HashSet<u64> {
    cars.filter(|(_, at_edge, beyond, v)| goes_far(*at_edge, *beyond, v)).map(|c| c.0).collect()
}

/// [round 9 A2 fix pass 3] What `edge_trip_end` decides on, for one random car this frame.
#[derive(Debug, Clone, Copy)]
struct EdgeFlag {
    /// Metres from its front to where `plan` rule 10b would stop it at the loaded tiles' edge,
    /// if its way ahead ends there (None: not heading for the edge).
    d_end: Option<f32>,
    /// Beyond its perceptual horizon (`Viewer::beyond_horizon`; with the lamps lit only beyond
    /// the draw distance).
    beyond: bool,
    blinker: i32,
    braking: bool,
    /// Its flag of the frame before.
    prev: bool,
    speed: f32,
    /// Its driver's comfortable braking (`DriverParams::comfort_brake`, m/s²).
    comfort_brake: f32,
}

/// [round 9 A2 fix pass 2, 3] Whether the loaded tiles' edge is the car's trip's end
/// (`External::edge_trip_end`: it drives off the lane's end moving instead of stopping, so no
/// brake lamp lights where a dot goes). Set while it heads for the edge beyond the horizon with
/// no indicator, not braking - or already set (a brake for a lead or a light on the way lights
/// its lamps there, not at the edge). [pass 3] Within its comfortable braking distance of the end
/// it holds only while not braking: braking there it stops at the edge instead (lamps lit, it
/// stays, round 8). And once set it holds, whatever the horizon (a zoom, a fast approach) or its
/// lamps, while the car can no longer stop before the end even at `MAX_BRAKE`: cleared there it
/// would drive off anyway, as an uncounted network end inside the horizon.
fn edge_trip_end(e: EdgeFlag) -> bool {
    let Some(d) = e.d_end else { return false };
    let v2 = e.speed * e.speed;
    if e.prev && d < v2 / (2.0 * omsi_sim::traffic::MAX_BRAKE) {
        return true;
    }
    if !e.beyond || e.blinker != 0 {
        return false;
    }
    if d < v2 / (2.0 * e.comfort_brake.max(0.1)) {
        return !e.braking;
    }
    e.prev || !e.braking
}

/// [round 9 A2 fix pass 2] The rule a car the traffic took off at the network's end is recorded
/// by: a timetable bus's last resort, a trip ended at the loaded tiles' edge (`edge_trip_end`:
/// the far trip end, counted as `removed_far`), else the network's own end.
fn network_end_rule(bus_last_resort: bool, on_edge_lane: bool, edge_trip_end: bool) -> RemoveRule {
    if bus_last_resort {
        RemoveRule::BusLastResort
    } else if on_edge_lane && edge_trip_end {
        RemoveRule::FarDeadEnd
    } else {
        RemoveRule::NetworkEnd
    }
}

/// [round 7 R7-1 fix 1] The recorded rule of a population removal, by the reason it logs.
fn removal_rule(why: &str) -> RemoveRule {
    match why {
        "its ground unloaded" => RemoveRule::GroundUnloaded,
        "out of range" => RemoveRule::OutOfRange,
        "finished, in view after its grace" => RemoveRule::InViewAfterGrace,
        "at the loaded tiles' edge, beyond the horizon" => RemoveRule::FarDeadEnd,
        w if w.starts_with("at the loaded tiles' edge, in view") => RemoveRule::InViewAfterGrace,
        w if w.starts_with("at the loaded tiles' edge") => RemoveRule::DeadEnd,
        w if w.ends_with("out of sight") => RemoveRule::OutOfSight,
        w if w.starts_with("at a dead end") || w.starts_with("at the loaded route's end") => RemoveRule::DeadEnd,
        _ => RemoveRule::Other,
    }
}

/// [round 8 H8-8, fix pass 2] A parked object that signals to pull out: which, its AI type, the
/// lane point it leaves onto and its offset there, its body's extents, when its signalling time
/// has run (`Traffic::time`), its seed.
#[derive(Clone)]
struct PendingPullOut {
    key: i64,
    ty: Arc<VehicleType>,
    lane: usize,
    s: f32,
    lat: f32,
    front: f32,
    rear: f32,
    half_w: f32,
    ready_at: f32,
    seed: u64,
}

/// What makes a new car a timetable bus (`Traffic::create_car`).
pub struct BusSetup {
    /// The trip's lanes, as far as the loaded tiles have them.
    pub route: Vec<usize>,
    pub stops: Vec<crate::bus_service::Stop>,
    /// Fleet number and registration (`number`, `ident` string variables).
    pub number: Option<(String, String)>,
    pub hof: Option<Arc<omsi_vehicle::Hof>>,
}

pub struct AiCar {
    /// Stable id for references from other systems (passengers).
    pub id: u64,
    /// The random seed it was made with and its paint scheme: a car that goes out of range
    /// and comes back is the same car (`DormantCar`).
    pub seed: u64,
    pub scheme: Option<usize>,
    /// The driver and its way: where the car is on the lanes, what it decided last frame,
    /// its manoeuvres (`omsi_traffic`). `ai.state` is the position along the lanes.
    pub ai: AiVehicle,
    pub vehicle: VehicleInstance,
    pub render: VehicleRender,
    pub trailer_renders: Vec<VehicleRender>,
    /// The body following the way `ai.state` lays out.
    pub body: AiBody,
    /// A timetable bus: its trip's stops, the doors, the layover, the people aboard (see
    /// `bus_service`). Everything else about it is this car's.
    pub bus: Option<Box<BusService>>,
    /// `[sound_ai]` set, created when the car comes near the listener.
    pub sounds: Option<omsi_audio::SoundSet>,
    /// Parking: the free space it drives into (see `Traffic::park_in`).
    pub park: Option<ParkPlan>,
    /// [round 7 B5] A timetable bus whose trip is over, let go (`Traffic::release`): it
    /// drives on as random traffic and leaves once nobody sees it. (It was the crate's
    /// `gone`, which means "stood too long" only.)
    pub released: bool,
    /// A rail vehicle: the track it has come along, (odometer, point), oldest first -
    /// where its rear bogie and its coupled cars and sections run (see `rail_behind`).
    pub rail_trail: std::collections::VecDeque<(f64, DVec3)>,
    /// Seconds its body and script took last frame (heavy ones get an AI job of their own).
    pub ai_secs: f32,
    /// A train turned round as a whole (its last car leads now): what a trip's
    /// `[trainreverse]` is compared with (Omsi.exe's vehicle +0x4e1).
    pub consist_reversed: bool,
}

/// A free parking space beside a lane that a car means to park in: the space of parked car
/// `key` that drove off (its object comes back when the car is in).
#[derive(Debug, Clone, Copy)]
pub struct ParkPlan {
    pub key: i64,
    pub lane: usize,
    /// The space's middle along the lane and its offset to the right of it (m).
    pub s: f32,
    pub lat: f32,
    /// Moving over into the space.
    pub ramped: bool,
    /// In the space and standing: the parked object takes its place at the next sync.
    pub done: bool,
}

impl AiCar {
    /// A timetable bus (in service or on its way off after its trip): it has a `BusService`,
    /// the class `ScheduledBus` (`omsi_traffic::Vehicle::serves_stops`). A bus of the random
    /// traffic (`[ai_veh_type]` 2) is random traffic here as everywhere (round 7 A4: it gives
    /// up and is removed like the other random vehicles; `Vehicle::heavy` is its class).
    pub fn serves_stops(&self) -> bool {
        self.bus.is_some()
    }

    /// Standing, as the traffic's own rules take it (`omsi_traffic::driver::STANDING_SPEED`):
    /// what every "queued behind" test here asks of a vehicle.
    pub fn standing(&self) -> bool {
        self.ai.state.speed.abs() < omsi_traffic::driver::STANDING_SPEED
    }

    /// Bound to rails (a train, a tram).
    pub fn is_rail(&self) -> bool {
        self.body.kind == MotionKind::Rail
    }

    /// Boarding at a stop: the script is told to open the doors (`AI_Scheduled_AtStation`).
    pub fn at_station(&self) -> bool {
        self.bus.as_ref().map(|b| b.at_station()).unwrap_or(false)
    }

    /// The side's doors to open at the stop it is boarding at (`AI_Scheduled_AtStation_Side`).
    pub fn at_station_side(&self) -> f32 {
        self.bus.as_ref().map(|b| b.at_station_side()).unwrap_or(0.0)
    }

    /// Standing at one of its stops (doors open, waiting for the departure, pulling out).
    pub fn at_stop(&self) -> bool {
        self.bus.as_ref().map(|b| b.at_stop()).unwrap_or(false)
    }

    pub fn trip_done(&self) -> bool {
        self.bus.as_ref().map(|b| b.trip_done()).unwrap_or(false)
    }

    pub fn route_open(&self) -> bool {
        self.bus.as_ref().map(|b| b.route_open).unwrap_or(false)
    }

    /// The next stop: (route index, distance along that lane).
    pub fn next_stop(&self) -> Option<(usize, f32)> {
        self.bus.as_ref()?.stops.front().map(|s| (s.ri, s.s))
    }

    /// Seconds it will still stand at its stop.
    pub fn standing_for(&self, day_time: f64) -> f32 {
        self.bus.as_ref().map(|b| b.standing_for(day_time)).unwrap_or(0.0)
    }
}

/// Where a vehicle's body stands, for the checks that go by geometry rather than by lanes:
/// the car it belongs to (a trailer or rear section counts as its own footprint), centre,
/// forward and right unit vectors, half length, half width and speed along its heading.
#[derive(Debug, Clone, Copy)]
struct Footprint {
    car: usize,
    center: DVec2,
    fwd: DVec2,
    right: DVec2,
    half_len: f64,
    half_w: f64,
    speed: f32,
    /// Height of the vehicle's origin (an aircraft overhead is not in a car's way).
    z: f64,
}

impl Footprint {
    fn from_obb(car: usize, b: &omsi_sim::collision::Obb, speed: f32) -> Footprint {
        let (sh, ch) = (b.heading.sin(), b.heading.cos());
        Footprint {
            car,
            center: b.center,
            fwd: DVec2::new(sh, ch),
            right: DVec2::new(ch, -sh),
            half_len: b.half.y,
            half_w: b.half.x,
            speed,
            z: b.z0,
        }
    }

    /// Does this footprint overlap `o`, both grown by `margin` (separating axes)?
    fn overlaps(&self, o: &Footprint, margin: f64) -> bool {
        let d = o.center - self.center;
        for axis in [self.fwd, self.right, o.fwd, o.right] {
            let extent = |f: &Footprint| {
                (f.fwd.dot(axis)).abs() * (f.half_len + margin)
                    + (f.right.dot(axis)).abs() * (f.half_w + margin)
            };
            if d.dot(axis).abs() > extent(self) + extent(o) {
                return false;
            }
        }
        true
    }
}

/// The player's vehicle as the traffic sees it: centre, heading (deg), half length, half
/// width, speed along the heading (m/s, negative when reversing).
pub type PlayerBox = (DVec3, f64, f32, f32, f32);

/// Where the player looks from, for putting cars on the road and taking them off only
/// where nobody sees it happen.
#[derive(Debug, Clone, Copy)]
pub struct Viewer {
    pub pos: DVec3,
    pub forward: DVec3,
    /// Tangents of half the horizontal and vertical field of view.
    pub tan_x: f64,
    pub tan_y: f64,
    /// Beyond this distance nothing shows (fog, or a car smaller than a pixel) (m).
    pub range: f64,
    /// What the renderer leaves out (see `omsi_render::RenderOptions`): objects smaller on
    /// the screen than `min_size` (the original's measure), and farther than `max_dist`
    /// (0 = no limit); `fov` is the vertical field of view (radians).
    pub min_size: f64,
    pub max_dist: f64,
    pub fov: f64,
    /// [round 9 A2] The picture's height in pixels (0: not known, see `Viewer::horizon`).
    pub px_height: f64,
    /// [round 9 A2 fix pass 1] The vertical field of view the horizon is reckoned with
    /// (radians): the camera's, or a triple screen's centre panel's when narrower (0: `fov`).
    pub horizon_fov: f64,
    /// [round 9 A2 fix pass 1] The vehicles' lamps are lit (`Viewer::with_lamps`): no two-pixel
    /// horizon, only the draw distance (`max_dist`).
    pub lamps_lit: bool,
}

/// A car further away than this is below two pixels on a 900-line screen (m).
const VISIBLE_RANGE: f64 = 900.0;

/// [round 9 A2] Fewer pixel rows than this and a car is a dot the eye cannot tell from nothing
/// (VISIBLE_RANGE's "two pixels").
const DOT_PIXELS: f64 = 2.0;
/// [round 9 A2] The height of a passenger car (m), the body VISIBLE_RANGE is documented for
/// (a saloon's 1.4-1.5 m; VISIBLE_RANGE's two pixels at 900 m on 900 lines is a 1.5 m body at
/// a 41 degree field of view): the horizon with the picture unknown scales from it.
const CAR_HEIGHT: f64 = 1.5;
/// [round 9 A2 fix pass 1] `Lighting::night` (0 day .. 1 night, the strength the coronas are
/// drawn with) above which the vehicles' lamps count as lit (safety review r9-a: coronas keep a
/// minimum of about three pixels at any distance, omsi-render corona.wgsl).
const LAMPS_LIT_NIGHT: f32 = 0.05;

/// Within this distance (m) of the camera a timetable bus has its driver at the wheel.
const DRIVER_NEAR: f64 = 70.0;

/// Within this distance of the camera no car appears or vanishes, seen or not: the mirrors
/// and a turn of the head see what is near (see [`Traffic::may_appear`]).
const NEVER_VANISH_WITHIN: f64 = 150.0;

/// Lane ends within this distance (m) of a lane start are linked (`Network::link`, the value
/// the network has always been linked with).
const LANE_LINK_TOL: f64 = 1.5;

/// [round 8 H8-7, fix pass 1] Whether world point `probe` lies on a tile of the map (`has_tile`)
/// whose lanes are not loaded (`lane_tiles`): the way on from a lane end there is not loaded.
fn loaded_edge_at(probe: DVec3, tile_size: f64, has_tile: &dyn Fn((i32, i32)) -> bool, lane_tiles: &hashbrown::HashSet<(i32, i32)>) -> bool {
    let key = ((probe.x / tile_size).floor() as i32, (probe.y / tile_size).floor() as i32);
    has_tile(key) && !lane_tiles.contains(&key)
}
/// Within this distance a vehicle appears or vanishes only behind something, wherever the
/// player looks (see `Traffic::hidden`).
const NEAR_HIDE: f64 = 350.0;

/// Within this distance of the camera an AI vehicle is animated and drawn even out of the
/// view (m): the mirrors look behind, and a car beside the view throws its shadow into it.
const UNSEEN_NEAR: f64 = 80.0;

/// How near an articulated AI bus has to be for its bellows to be reshaped as its joint
/// turns (m): the fold of the bend is a few centimetres, which is a screen pixel or more
/// within this range - farther out it is not worth a mesh update every frame it steers.
const SKIN_DISTANCE: f64 = 200.0;

impl Viewer {
    pub fn new(cam: &omsi_render::Camera, aspect: f64, fog_range: f64) -> Viewer {
        let tan_y = (cam.fov_deg as f64 * 0.5).to_radians().tan();
        Viewer {
            pos: cam.position,
            forward: cam.forward().as_dvec3().normalize_or_zero(),
            tan_x: tan_y * aspect.max(0.2),
            tan_y,
            range: fog_range.min(VISIBLE_RANGE).min(cam.far as f64),
            min_size: 0.0,
            max_dist: 0.0,
            fov: (cam.fov_deg as f64).to_radians(),
            px_height: 0.0,
            horizon_fov: 0.0,
            lamps_lit: false,
        }
    }

    /// [round 9 A2] The picture's height in pixels (the window's, or the offscreen image's),
    /// and [fix pass 1] a triple screen's centre panel's vertical field of view (degrees): the
    /// horizon is reckoned with the narrower of it and the camera's (the panels show the rig's
    /// field of view, not the camera's).
    pub fn with_screen(mut self, px_height: u32, panel_fov_deg: Option<f32>) -> Viewer {
        self.px_height = px_height as f64;
        self.horizon_fov = panel_fov_deg.map_or(self.fov, |f| (f as f64).to_radians().min(self.fov));
        self
    }

    /// [round 9 A2 fix pass 1] The vehicles' lamps are lit (dusk, night, gloom): a lamp glows at a
    /// few pixels at any distance, so there is no two-pixel distance; a lit car is out of the
    /// picture only where the renderer leaves it out, lamps included (`lights::lamps_drawn`).
    pub fn with_lamps(mut self, lit: bool) -> Viewer {
        self.lamps_lit = lit;
        self
    }

    /// The two-pixel distance (m) of a body `height` (m) high: farther off it covers fewer than
    /// `DOT_PIXELS` rows of this picture (`px_height / (2 tan(fov/2))` pixels per unit of the image
    /// plane, the vertical field of view `horizon_fov`) and is a dot the player cannot tell from
    /// nothing. With the picture's height unknown, the documented `VISIBLE_RANGE` scaled from a
    /// car's height; with the lamps lit, none (infinite).
    fn two_pixels(&self, height: f64) -> f64 {
        if self.lamps_lit {
            f64::INFINITY
        } else if self.px_height <= 0.0 {
            VISIBLE_RANGE * height / CAR_HEIGHT
        } else {
            let fov = if self.horizon_fov > 0.0 { self.horizon_fov } else { self.fov };
            height * self.px_height / (DOT_PIXELS * 2.0 * (fov * 0.5).tan().max(1e-3))
        }
    }

    /// The perceptual horizon (m) of a place where a body `height` (m) high may stand: the nearer
    /// of its two-pixel distance and the renderer's reach (`max_dist`, the object distance -
    /// `max_obj_dist`, the view distance when that is `auto` - held out to `MAX_OBJ_DIST_HOLD`;
    /// lamp glows are culled with their car there). For the loaded tiles' edge: whether it is in
    /// sight (`Traffic::mark_edges_in_sight`) and how far the tiles are prefetched
    /// (`prefetch_target`). A body itself is judged by `beyond_horizon`, with its own size.
    pub fn horizon(&self, height: f64) -> f64 {
        let two_pixels = self.two_pixels(height);
        if self.max_dist > 0.0 {
            two_pixels.min(self.max_dist * omsi_render::MAX_OBJ_DIST_HOLD as f64)
        } else {
            two_pixels
        }
    }

    /// [round 9 A2] Is a body `height` (m) high and `radius` (m) round, `dist` (m) from the eye,
    /// beyond its horizon: farther than its two-pixel distance, or where the renderer does not draw
    /// it (`draws`; its lamps go with it) - and ([fix pass 1]) never within `NEAR_HIDE`, where only
    /// cover hides a car whatever the picture (the mirrors, a turn of the head). One measure for
    /// every use: beyond it a random car may be woken in frame (`Traffic::may_appear_car`) and a
    /// car reaching the loaded tiles' edge ends its trip there in frame
    /// (`omsi_traffic::removal::goes_far`).
    pub fn beyond_horizon(&self, dist: f64, height: f64, radius: f64) -> bool {
        dist >= NEAR_HIDE && (dist > self.two_pixels(height) || !self.draws(dist, radius))
    }

    /// A wider picture than the camera's own (a triple screen's side panels): the tangents
    /// of its half-angles, horizontal and vertical. The size limit stays the camera's.
    pub fn with_extent(mut self, extent: Option<(f64, f64)>) -> Viewer {
        if let Some((tan_x, tan_y)) = extent {
            self.tan_x = self.tan_x.max(tan_x);
            self.tan_y = self.tan_y.max(tan_y);
        }
        self
    }

    /// The renderer's culling as well (`RenderOptions::min_obj_size`, `max_obj_dist`).
    pub fn with_culling(mut self, min_size: f32, max_dist: f32) -> Viewer {
        self.min_size = min_size.max(0.0) as f64;
        self.max_dist = max_dist.max(0.0) as f64;
        self
    }

    /// Would the renderer draw an object of radius `r` this far away at all (an object already
    /// drawn stays out to `MAX_OBJ_DIST_HOLD` times the object distance; too small on screen, under
    /// `min_size`, it is left out - but with the lamps lit its glows are not, so then only the
    /// distance counts)? Beyond that a car, its
    /// lamp glows included (`lights::lamps_drawn`), can come and go in plain view without anybody
    /// seeing it happen.
    pub fn draws(&self, dist: f64, r: f64) -> bool {
        // (the renderer measures a vehicle by a sphere about its origin, which may stand
        // well off its middle: half as much again, and a metre, to be sure)
        let r = r * 1.5 + 1.0;
        if self.max_dist > 0.0 && dist > self.max_dist * omsi_render::MAX_OBJ_DIST_HOLD as f64 + r {
            return false;
        }
        // (with the lamps lit only the distance counts: the renderer leaves a body under
        // `min_size` out, but its lamp glows are culled by the distance alone,
        // `lights::lamps_drawn`, and still show)
        self.lamps_lit || self.min_size <= 0.0 || 2.0 * r / (dist.max(0.01) * self.fov.max(1e-3)) >= self.min_size
    }

    /// Could a vehicle of radius `r` at `p` come or go unseen with nothing hiding it: the
    /// rule of `Traffic::hidden` without its "behind a building or the ground" branch, for
    /// where no world is at hand. Never within `NEVER_VANISH_WITHIN`; beyond that where
    /// the renderer leaves it out, or past `NEAR_HIDE` out of the picture.
    pub fn hides_plainly(&self, p: DVec3, r: f64) -> bool {
        let d = (p - self.pos).length();
        d >= NEVER_VANISH_WITHIN && (!self.draws(d, r) || (d >= NEAR_HIDE && !self.frames(p, r)))
    }

    /// Does a sphere of radius `r` at `p` lie within the view frustum and range?
    pub fn frames(&self, p: DVec3, r: f64) -> bool {
        let rel = p - self.pos;
        let dist = rel.length();
        if dist <= r {
            return true;
        }
        if dist - r > self.range {
            return false;
        }
        let f = self.forward;
        let mut right = f.cross(DVec3::Z);
        if right.length() < 1e-3 {
            right = DVec3::X;
        }
        let right = right.normalize();
        let up = right.cross(f);
        let z = rel.dot(f);
        if z < -r {
            return false;
        }
        let (x, y) = (rel.dot(right), rel.dot(up));
        x.abs() <= z * self.tan_x + r * (1.0 + self.tan_x * self.tan_x).sqrt()
            && y.abs() <= z * self.tan_y + r * (1.0 + self.tan_y * self.tan_y).sqrt()
    }
}

/// How far from the player cars are kept (m) and how far out of sight one may be before it
/// is taken off (m); in plain view a car stays until it is too small to see.
const DESPAWN_FACTOR: f64 = 1.6;
/// [ROLLBACK appearfar-1] Further off than this (m) a random car may appear in plain view.
const APPEAR_IN_VIEW_BEYOND: f64 = 300.0;

fn rel_len(p: DVec3, center: DVec3) -> f64 {
    (p - center).length()
}

/// A random car out of the player's range: still on the map and still driving, but without
/// a body, a script or a picture - a few numbers. It comes back as the same car (type,
/// paint, id) where it has got to when the player comes near, and it only ever leaves the
/// map at the end of the road network. Before, every car out of range was simply taken
/// away and new ones made up around the player: the traffic followed the player about, and
/// a car driven past was never seen again.
pub struct DormantCar {
    pub id: u64,
    pub ty: Arc<VehicleType>,
    pub kind: LaneKind,
    pub lane: usize,
    pub s: f32,
    pub speed: f32,
    pub seed: u64,
    pub scheme: Option<usize>,
    /// Its own dice for the turns it takes.
    pub walk: u64,
}

/// How many cars a whole map keeps at most, as a multiple of the number asked for around the
/// player (memory: a dormant car is a few dozen bytes, but each one woken is a full vehicle).
const MAP_POPULATION_FACTOR: f32 = 8.0;


fn street_lane_weight(l: &omsi_sim::traffic::Lane) -> Option<f64> {
    (l.kind == LaneKind::Street && !l.no_cars && l.density > 0.0 && l.length() >= 8.0)
        .then(|| l.length() as f64 * l.density.clamp(0.05, 4.0) as f64)
}

pub struct Traffic {
    pub net: Network,
    /// Sum of the spawn weights of every street lane, updated only as tiles add lanes.
    street_weight: f64,
    /// Parked cars standing in or beside a lane: per lane, their boxes in the lane's frame
    /// (distance along it and signed lateral offset of the box's middle, + = right, and its
    /// half extents along and across the lane). A car in the lane's middle is an obstacle to
    /// stop behind; one over the kerb side is passed with a swerve to the left.
    parked: HashMap<usize, Vec<Parked>>,
    /// Parked cars no lane has been found beside yet: the lane may come with a tile that is
    /// not loaded yet (a road spline often starts in the next tile). Most stand in car parks
    /// and stay here.
    parked_waiting: Vec<ParkedCar>,
    /// Tiles whose lanes the network has (lanes stay once they are in).
    pub lane_tiles: hashbrown::HashSet<(i32, i32)>,
    /// Counts the times tiles brought their lanes: whoever resolved something against the
    /// network and missed a part of it looks again when this changes.
    pub lanes_generation: u64,
    /// AI vehicle types with weight, the lane kind they run on (`[type]` 2 rail, 3 air)
    /// and their group in `groups`.
    types: Vec<(Arc<VehicleType>, f32, LaneKind, usize)>,
    /// The random traffic groups: name, `unsched_trafficdens.txt` factor and day curves.
    groups: Vec<omsi_map::ailists::UnschedGroup>,
    /// The map has an `unsched_trafficdens.txt` (else the global.cfg curve applies).
    group_curves: bool,
    /// Each group's place in `unsched_vehgroups.txt`, the number a path's `[rule]
    /// trafficdensity` names it by (None: the map has no such file, and every group drives
    /// wherever the lane's density lets traffic).
    group_uvg: Vec<Option<usize>>,
    /// The default density of every `unsched_vehgroups.txt` entry, in file order: 0 none, 1
    /// for the first entry its medium density, for any other that of the first entry, 2 of
    /// the second, and so on. It applies on the paths without a rule for the group.
    uvg_defaults: Arc<Vec<i32>>,
    pub cars: Vec<AiCar>,
    /// The random cars out of range (see `DormantCar`).
    pub dormant: Vec<DormantCar>,
    /// `time` when the dormant cars last moved on.
    dormant_time: f32,
    rng: u64,
    /// Target number of cars around the camera.
    pub target: usize,
    /// Made only so that the light programs run (no traffic, no timetable): nobody is put
    /// on the roads - no aircraft, no parked car pulling out - while `target` is 0.
    pub lights_only: bool,
    pub spawn_radius: f64,
    pub time: f32,
    /// Renders of cars that have gone, given back at the next `sync`.
    released: Vec<VehicleRender>,
    /// Where the camera is (the window sets it before `sync`): far cars show their script
    /// textures as stand-ins.
    pub camera: Option<DVec3>,
    /// Sound sets of despawned cars, stopped at the next audio update.
    orphan_sounds: Vec<omsi_audio::SoundSet>,
    /// The drivers' shared state: the junction tables, the light programs, this frame's
    /// inputs (the players, the people, the parked cars) and the frame itself.
    pub sim: omsi_traffic::Simulation,
    controller_of_object: HashMap<i64, usize>,
    /// Coupled vehicle types by file.
    trailer_types: HashMap<std::path::PathBuf, Option<Arc<VehicleType>>>,
    /// `[sound_ai]` configurations by file.
    sound_cfgs: HashMap<std::path::PathBuf, Option<Arc<omsi_vehicle::SoundCfg>>>,
    root: std::path::PathBuf,
    /// Who wants a timetable bus to stop (`Humans::stop_wishes`): the buses somebody
    /// aboard wants to get off, the stops where somebody waits. None without passengers:
    /// every bus then serves every stop.
    stop_wishes: Option<(hashbrown::HashSet<u64>, hashbrown::HashSet<i64>)>,
    /// Seconds the player's vehicle has been standing.
    player_still: f32,
    /// Time of day (seconds since midnight); light cycles and timetables run on it.
    pub day_time: f64,
    /// How fast the clock runs (the time speed): the timetable keeps to it.
    pub time_scale: f64,
    /// Day of the week (0 Monday … 6 Sunday) for the traffic density curves.
    pub weekday: i32,
    /// Street lights on → AI vehicles switch their lights on.
    pub night: bool,
    /// The light of the day, for the cars' `Envir_Brightness` (see `sync`).
    pub daylight: Option<omsi_sim::Daylight>,
    next_id: u64,
    /// The last car that started an overtake and when (for chase-camera debugging).
    pub last_overtaker: Option<(u64, f32)>,
    /// The first car that entered a turning lane and when (`--follow turn`).
    pub first_turner: Option<(u64, f32)>,
    /// The first car that stopped at a red light (`--follow red`), the first that gave way
    /// at a junction (`--follow yield`), the first that pulled out onto the other side of
    /// the road round an obstacle or squeezed past a bus at its stop (`--follow pass`).
    pub first_red: Option<(u64, f32)>,
    pub first_yield: Option<(u64, f32)>,
    pub first_passer: Option<(u64, f32)>,
    /// `[trafficdensity_road]` curve of the map: (hour, factor).
    pub density_curve: Vec<(f32, f32)>,
    /// The options' `[AIUnschedFactor]`: the share of the random traffic.
    pub unsched_factor: f32,
    /// The options' `[AIMaxCountScheduled]` (0 = no limit).
    pub max_scheduled: u32,
    /// Where the player looks from (set every frame).
    pub viewer: Option<Viewer>,
    /// Buildings that hide what is behind them (the player's collision world).
    pub occluders: Option<Arc<omsi_sim::collision::CollisionWorld>>,
    /// Pedestrians on the footpaths: (lane, distance along it), for giving way at crossings
    /// and for the pedestrian lights' request buttons.
    pub walkers: Vec<(usize, f32)>,
    /// Everybody on foot on the ground: position, velocity, whether they are waiting at a
    /// stop, and for how long they have stood still (set every frame) - the cars stop for
    /// anybody in their way, not only on a crossing.
    pub people: Vec<(DVec3, DVec2, bool, f32, bool, bool, Option<i64>)>,
    /// No car has been placed yet: the first population may fill the view.
    initial: bool,
    /// Seconds of the last tick (the lamp scripts run in `sync`).
    last_dt: f32,
    /// Game time since the lamps' scripts last ran (see `sync`).
    lamp_dt: f32,
    /// `OMSI_TRACE_AI=<file.csv>`: every car's pose, steering and speed, every frame.
    trace: Option<std::io::BufWriter<std::fs::File>>,
    /// Cars already reported for a hard bend (`OMSI_DEBUG_TRAFFIC`).
    logged_hard: hashbrown::HashSet<u64>,
    /// `OMSI_DEBUG_LIGHTS=all|near|<controller>,…`: which programs log their changes, and
    /// the states they showed last.
    light_log: Option<String>,
    light_prev: Vec<Vec<i32>>,
    /// `OMSI_DEBUG_POPULATION`: log where cars appear and vanish relative to the view.
    debug_population: bool,
    /// [round 9 A2] Since the start: cars that appeared in frame beyond the perceptual
    /// horizon, and cars taken off in frame there (`RemoveRule::FarDeadEnd`) (the population
    /// log line).
    appeared_far: usize,
    removed_far: usize,
    /// [round 9 A4] The lanes whose road ends at a loaded tiles' edge in sight
    /// (`mark_loaded_edges`, `mark_edges_in_sight`).
    edge_bound: hashbrown::HashSet<usize>,
    /// [round 10 S4] The lanes with `loaded_edge` (`mark_loaded_edges`).
    edge_lanes: Vec<usize>,
    /// Cars placed since the last look inside the view frustum (hidden behind something):
    /// (id, position). `OMSI_POPULATION_SHOTS` photographs them to check.
    pub framed_spawns: Vec<(u64, DVec3)>,
    /// The player's vehicle as of the last tick (nothing is put on the road on top of it).
    player: Option<PlayerBox>,
    /// The player's bus has right of way over the traffic (its script's `TrafficPriority`,
    /// OMSI: priority 1000 over the types' own): cars keep out of the way it is about
    /// to take for longer.
    pub player_priority: bool,
    /// The player's indicators (0 off, 1 left, 2 right, 3 hazard; `lan::indicator`), set
    /// before each `tick`.
    pub player_blinker: u8,
    /// How long the player's bus has been indicating towards the traffic (s).
    player_signalling: f32,
    /// The LAN players' vehicles (their session ids and boxes as for the player), set
    /// before each `tick`: the cars stop behind them and go round them as round the
    /// player's bus.
    pub others: Vec<(u32, PlayerBox)>,
    /// The drivers at the wheel of the timetable buses near the camera, by car id (see
    /// `driver.rs`; made within `DRIVER_NEAR` m of the camera, let go beyond twice that).
    drivers: HashMap<u64, crate::driver::DriverFigure>,
    /// Figures let go by their bus, hidden, for the next one (their GPU meshes stay).
    driver_pool: Vec<crate::driver::DriverFigure>,
    /// Where the last `tick` spent its time (s, OMSI_PROFILE): who is on which lane and the
    /// light programs, every car's plan, the bodies and scripts on the workers.
    pub tick_split: [f64; 3],
    /// Where the planning part of the last `tick` went (s, OMSI_PROFILE): the per-car
    /// pre-plan loop, the recorder's frame, `Simulation::tick`; the sums and the frame count
    /// since the last line logged.
    pub tick_inner: [f64; 3],
    /// The cars by id and by lead: valid for one frame only, built before the per-car loop of
    /// `tick` and cleared right after it (the indices go stale once cars are removed).
    links: CarLinks,
    tick_inner_sum: ([f64; 3], u32),
    /// Seconds each of them has stood still.
    others_still: HashMap<u32, f32>,
    /// Timetable buses taken off the road because the tile under them was unloaded (their
    /// ids), for the timetable to put them back when the tiles come again.
    pub removed_scheduled: Vec<u64>,
    /// [ROLLBACK r4d-B1] Timetable buses in service that drove off the end of a dead-end lane
    /// (their ids): the timetable puts the trip back where its track goes on.
    pub dead_ended: Vec<u64>,
    /// [ROLLBACK r4d-B1] Timetable buses taken off the road this frame, by any rule (their
    /// ids): whoever rides them is put out with them (`humans::Humans::evict`), so that a
    /// bus put back by the timetable carries nobody, and nobody is left riding a bus that
    /// is gone.
    pub riders_out: Vec<u64>,
    /// [ROLLBACK r4d-B2] The through lanes as the traffic drives them (for the timetable
    /// buses' layovers), and the timetable buses taken off the road as a last resort, with
    /// the time of day their trip is to be put on the road again.
    pub through: crate::bus_service::ThroughLanes,
    pub respawned: Vec<crate::bus_service::Respawned>,
    /// [ROLLBACK r4d-B2] Timetable buses not to be taken off the road as the last resort
    /// before this time of day (the tour's bus was, a moment ago: `RESPAWN_COOLDOWN`).
    pub respawn_block: HashMap<u64, f64>,
    /// [ROLLBACK r4d-B2] Why each timetable bus that nothing lets out of the way was not
    /// taken off the road either, as logged (once per reason).
    respawn_refused: HashMap<u64, &'static str>,
    /// One-way lanes that have had their reverse twin added (`add_reverse_twins`).
    twinned: hashbrown::HashSet<usize>,
    /// Bodies besides the AI vehicles' that no timetable vehicle may be put into: the
    /// player's vehicle and the LAN players' (set before each `Schedule::tick`).
    pub keep_clear: Vec<omsi_sim::collision::Obb>,
    /// LAN play: this game draws the host's traffic instead of its own (`lan_world`).
    mirror: bool,
    /// LAN play: where the other players are (host): the traffic is kept around them too.
    pub lan_centers: Vec<DVec3>,
    /// [ROLLBACK clockjump-traffic-1] The clock as of the last `note_clock` (NaN before the
    /// first), for telling a time set by hand from the clock running.
    clock_seen: f64,
    /// The traffic recorder (`omsi_traffic::record`): the last minute in memory always, a
    /// file while a recording runs (`OMSI_TRAFFIC_RECORD=<file>`, or the game menu).
    pub recorder: omsi_traffic::record::Recorder,
    /// What the people on foot do, in the order of `people` (set with it every frame), for
    /// the recording.
    pub people_records: Vec<omsi_traffic::record::PersonRecord>,
    /// The map's folder name: recordings and network dumps are named after it.
    pub map_name: String,
    /// Cars whose standing too long has been saved already (each once).
    stuck_saved: hashbrown::HashSet<u64>,
    /// The minutes saved by themselves this session, and when the last was (`time`): at
    /// most `AUTO_SAVES`, a minute apart.
    auto_saves: (u32, f32),
    /// When the cars were last looked over for two inside each other (`time`).
    overlap_checked: f32,
    /// [round 8 H8-8, fix pass 2] A parked object signalling to pull out (`advance_pull_out`).
    pending_pull_out: Option<PendingPullOut>,
}

/// At most this many minutes of traffic are saved by themselves in a session (a car stuck,
/// two cars inside each other); a jam of many would fill the disk otherwise.
const AUTO_SAVES: u32 = 3;

/// The folder recordings go to: `~/.openomsi/traffic` (made when missing).
pub fn recordings_dir() -> Option<std::path::PathBuf> {
    let dir = crate::settings::Settings::path()?.parent()?.join("traffic");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// The local time as `HHMMSS`, for naming a recording.
fn hhmmss() -> String {
    let s = crate::real_time::now().map(|n| n.secs as u32).unwrap_or(0);
    format!("{:02}{:02}{:02}", s / 3600, s / 60 % 60, s % 60)
}

/// Two bodies count as inside each other past this depth (m): a touch of 20 cm is the
/// bodies' rounding (`Traffic::overlaps`, the recordings' "overlap" trigger and the
/// `OMSI_CHECK_OVERLAP` log).
const OVERLAP_DEPTH: f64 = 0.2;

/// How far (m) a vehicle put on the road keeps its bodies from every other vehicle's
/// (`Traffic::spawn_clear`): the stricter of the two tests it replaces (footprints with 1 m
/// on each side, 2 m between; the timetable's bodies with 0.5 m).
const SPAWN_BODY_CLEAR: f64 = 2.0;

/// [round 7 B4] How far ahead of a spot a standing car keeps a new one from being put down
/// there (m): SUMO's insertion gap, its car-following safe gap to a standing leader - the
/// reaction (time gap `tau`) at the speed it is put down at, stopping from there with a town
/// car's comfortable braking, and its standing gap: v·tau + v²/2b + s0 (`DriverParams` car
/// preset; 40 m at 10 m/s).
fn spawn_queue_reach(v: f32) -> f32 {
    let car = omsi_traffic::DriverParams::preset(VehicleClass::Car);
    v * car.tau + v * v / (2.0 * car.comfort_brake) + car.min_gap
}

/// [round 7 B4] `Traffic::may_put_on` over the network and its compiled junctions: lane `l`
/// carries no light and is no junction piece with foes.
fn may_put_on(net: &Network, junctions: &omsi_traffic::Junctions, l: usize) -> bool {
    net.lanes.get(l).is_some_and(|lane| lane.traffic_light.is_none()) && !junctions.movement(l).is_some_and(|(_, m)| !m.foes.is_empty())
}

/// [round 7 B4] Whether the nearest vehicle ahead of `s` on lane `l`, or on any way on from it,
/// within `reach` (m) stands. `on(lane)`: the vehicles on a lane, (their origin and their
/// rear along the lane, standing). A branch ends at its first vehicle.
fn queue_within_reach(net: &Network, l: usize, s: f32, reach: f32, on: &dyn Fn(usize) -> Vec<(f32, f32, bool)>) -> bool {
    // (lanes to look at, with how far their start lies from the spot)
    let mut todo = vec![(l, -s)];
    let mut seen: hashbrown::HashSet<usize> = hashbrown::HashSet::new();
    while let Some((lane, start)) = todo.pop() {
        if start > reach || !seen.insert(lane) {
            continue;
        }
        // the nearest vehicle ahead of the spot on this lane: (distance to its rear, standing)
        let ahead = on(lane).into_iter().filter(|&(at, _, _)| lane != l || at > s).map(|(_, rear, standing)| (start + rear, standing)).min_by(|a, b| a.0.total_cmp(&b.0));
        match ahead {
            Some((d, standing)) => {
                if standing && d < reach {
                    return true;
                }
            }
            None => {
                let len = net.lanes.get(lane).map_or(0.0, |x| x.length());
                for &n in net.lanes.get(lane).map(|x| x.next.as_slice()).unwrap_or(&[]) {
                    todo.push((n, start + len));
                }
            }
        }
    }
    false
}

/// The share of the lane's limit a random car is put down at (`create_car`).
const SPAWN_SPEED_SHARE: f32 = 0.7;

/// A parked car a tile placed: its origin, heading (degrees) and `[boundingbox]`.
pub(crate) type ParkedCar = (DVec3, f64, Option<[f32; 6]>);

/// The box of a parked car whose object gives no `[boundingbox]`: 1.8 x 4.6 m, the fixed
/// box every parked car had until round 7 (half 0.9 x 2.3).
const PARKED_BOX: [f32; 6] = [1.8, 4.6, 1.5, 0.0, 0.0, 0.75];

/// `[boundingbox]` of a vehicle that gives none (bus-sized: the traffic's bodies and the
/// people's view of the AI cars, `Humans`, use this one value).
pub(crate) const DEFAULT_BOX: [f32; 6] = [2.5, 12.0, 3.0, 0.0, 0.0, 1.5];

/// The bodies of a vehicle and of the parts coupled to it, where they stand.
pub fn vehicle_bodies(v: &VehicleInstance) -> Vec<omsi_sim::collision::Obb> {
    let mut out = vec![omsi_sim::collision::Obb::from_box(
        v.ty.def.bounding_box.unwrap_or(DEFAULT_BOX),
        v.position,
        v.body_heading(),
    )];
    for t in &v.trailers {
        out.push(omsi_sim::collision::Obb::from_box(
            t.ty.def.bounding_box.unwrap_or(DEFAULT_BOX),
            t.position,
            t.body_heading(),
        ));
    }
    out
}

/// Cruising speed of an AI aircraft where its flight path sets no limit (km/h): an
/// airliner on its final approach.
const AIRCRAFT_KMH: f32 = 280.0;

/// A timetable bus's IBIS moves on to its next stop as the driver would press it on: the
/// stock scripts' interior displays, announcements and side displays read `IBIS_busstop`
/// (an index into the depot file's stop list of the route), which nothing moved on an AI
/// bus - its saloon display stood on the first stop for the whole trip. `remaining` is the
/// number of stops still to come.
pub(crate) fn ibis_to_next_stop(v: &mut VehicleInstance, remaining: usize) {
    let Some(ri) = v.var("IBIS_RouteIndex").filter(|r| *r >= 0.0) else { return };
    let Some(n) = v.host.hof.as_ref().and_then(|h| h.info_busstop_lists.get(ri as usize)).map(|l| l.len()) else { return };
    if n == 0 || v.var("IBIS_busstop").is_none() {
        return;
    }
    let idx = n.saturating_sub(remaining.max(1)).min(n - 1);
    v.set_var("IBIS_busstop", idx as f32);
}
/// How far ahead a car looks for other vehicles at least (m).
fn ai_ground(world: &World) -> Arc<dyn Fn(f64, f64) -> Option<f64> + Send + Sync> {
    let terrains = world.terrains.clone();
    let surfaces = world.surfaces.clone();
    Arc::new(move |x, y| {
        let tx = (x / omsi_map::tile_size()).floor() as i32;
        let ty = (y / omsi_map::tile_size()).floor() as i32;
        let lx = (x - tx as f64 * omsi_map::tile_size()) as f32;
        let ly = (y - ty as f64 * omsi_map::tile_size()) as f32;
        let surface = surfaces.read().get(&(tx, ty)).cloned();
        if let Some(h) = surface.and_then(|s| s.sample_road_smooth(lx, ly)) {
            return Some(h as f64);
        }
        let t = terrains.read();
        Some(t.get(&(tx, ty))?.sample(lx, ly) as f64)
    })
}

/// The bare ground at a point (no roads, decks or platforms on it).
fn terrain_height(world: &World, x: f64, y: f64) -> Option<f64> {
    let tx = (x / omsi_map::tile_size()).floor() as i32;
    let ty = (y / omsi_map::tile_size()).floor() as i32;
    let t = world.terrains.read();
    let terrain = t.get(&(tx, ty))?;
    Some(terrain.sample(
        (x - tx as f64 * omsi_map::tile_size()) as f32,
        (y - ty as f64 * omsi_map::tile_size()) as f32,
    ) as f64)
}

/// How a vehicle on lanes of `kind` moves.
fn motion_kind(kind: LaneKind) -> MotionKind {
    match kind {
        LaneKind::Air => MotionKind::Air,
        LaneKind::Rail => MotionKind::Rail,
        _ => MotionKind::Road,
    }
}

/// How far ahead of the player's bus centre a car looks for it (m): the bus's half length,
/// and where it will be in `horizon` seconds for a car whose way crosses the bus's. Not
/// for one going the same way (`way_dir` within 60 degrees of the bus's heading): with the
/// bus behind it, that stretch ahead of the bus reached over the car itself and it braked
/// for a bus that was only following it (#139).
const RAIL_TRAIL: f64 = 400.0;

/// Note where an AI rail vehicle is: `odometer` (m) and the point of its way there. A jump
/// (put somewhere else, turned round at a terminus) starts the trail afresh.
fn record_rail_trail(trail: &mut std::collections::VecDeque<(f64, DVec3)>, odometer: f64, here: DVec3) {
    if let Some(&(u, p)) = trail.back() {
        if (here - p).truncate().length() > (odometer - u).abs() + 2.0 {
            trail.clear();
        } else if (odometer - u).abs() <= 0.5 {
            return;
        }
    }
    // (backing up takes the trail back with it)
    while trail.back().is_some_and(|b| b.0 > odometer) {
        trail.pop_back();
    }
    trail.push_back((odometer, here));
    while trail.front().is_some_and(|f| odometer - f.0 > RAIL_TRAIL) {
        trail.pop_front();
    }
}

/// The point of an AI rail vehicle's track `d` metres behind its origin: on the trail it
/// came along. (Its way knows only the lane it came off; farther back it runs straight on,
/// and a train's last cars stood beside the track after a pair of points.) Where the trail
/// does not reach - the last half metre, a vehicle just put there - the way.
fn rail_behind(trail: &std::collections::VecDeque<(f64, DVec3)>, state: &AiState, net: &Network, d: f64) -> DVec3 {
    let u = state.odometer as f64 - d;
    let newest = trail.back().map_or(f64::MIN, |b| b.0);
    // [ROLLBACK trail-142] older than the trail reaches (it was just started, or started
    // afresh after a jump): the way, not the trail's oldest point - every car of a long train
    // behind the trail's end was put on that one point, one inside the other
    let oldest = trail.front().map_or(f64::MAX, |f| f.0);
    if u >= newest || u < oldest {
        return state.way_point(net, -d as f32);
    }
    crate::rail_drive::point_at(trail, u).unwrap_or_else(|| state.way_point(net, -d as f32))
}

/// A body for a vehicle that has just been put on the way `state` describes, with the
/// vehicle posed on it.
fn place_body(
    net: &Network,
    state: &AiState,
    vehicle: &mut VehicleInstance,
    kind: MotionKind,
) -> AiBody {
    let mut body = AiBody::new(&vehicle.ty.def, kind);
    let ground = vehicle.ground.clone();
    let contact = vehicle.contact.clone();
    body.place(
        &|d| state.way_point(net, d),
        ground
            .as_ref()
            .map(|g| g.as_ref() as &dyn Fn(f64, f64) -> Option<f64>),
        contact.as_deref(),
        state.speed,
    );
    body.apply(vehicle);
    body
}

/// The `OMSI_TRACE_AI` file, with its header written.
fn open_trace() -> Option<std::io::BufWriter<std::fs::File>> {
    use std::io::Write;
    let path = omsi_cfg::env::var_os("OMSI_TRACE_AI")?;
    let mut f = std::io::BufWriter::new(
        std::fs::File::create(&path)
            .map_err(|e| log::warn!("OMSI_TRACE_AI: {e}"))
            .ok()?,
    );
    writeln!(f, "t,id,type,x,y,z,heading,pitch,bank,steer,speed,lane,s,blinker,turn,lane_heading,lateral,at_station,acc,yielding,light_hold,passing,front,rear,half_width,scheduled,why,why_gap,phase,lane_z").ok()?;
    Some(f)
}

/// The vehicle's extent from its origin: (to the front bumper, to the rear bumper, half
/// the width) from its `[boundingbox]`.
fn extents(ty: &VehicleType, length: f32) -> (f32, f32, f32) {
    let (front, rear, width) = match ty.def.bounding_box {
        Some(bb) if bb[1] > 1.0 => (
            bb[1] * 0.5 + bb[4],
            bb[1] * 0.5 - bb[4],
            (bb[0] * 0.5).max(0.5),
        ),
        // without a `[boundingbox]` the model's own box, as Omsi.exe takes it (0x7b5da4):
        // the Berlin S-Bahn's cars, 18 m long, counted as 12 m ones
        _ => match ty.model_box() {
            Some((lo, hi)) if hi.y - lo.y > 1.0 => (hi.y.max(0.5), (-lo.y).max(0.5), (hi.x.max(-lo.x)).max(0.5)),
            _ => (length * 0.5, length * 0.5, 0.9),
        },
    };
    if omsi_sim::vehicle::body_reversed(&ty.def, false) {
        (rear, front, width)
    } else {
        (front, rear, width)
    }
}

/// The driver of a random car: how fast, how close, how patient (see `AiState`).
impl Traffic {
    /// Make the population deterministic for a LAN room.  The room's session id is
    /// shared by the host and every client, so the same map/time produces the same
    /// initial cars instead of each process inventing a different world.
    /// [ROLLBACK clockjump-traffic-1] Note the clock (`day_time`, set before): true when it
    /// was set to another time by hand - a step of over 90 s beyond what the time speed makes
    /// in a frame (as `Schedule::note_clock`).
    pub fn note_clock(&mut self, dt: f64, speed: f64) -> bool {
        let jumped = self.clock_seen.is_finite() && (self.day_time - self.clock_seen).abs() > 90.0 + dt.max(0.0) * speed.max(1.0) * 5.0;
        self.clock_seen = self.day_time;
        jumped
    }

    /// [ROLLBACK clockjump-traffic-1] The clock was set to another time: the next population
    /// pass fills the roads up as at load, cars appearing anywhere - the density of the new
    /// hour came at once, but the cars only where nobody saw them appear, and a jump from the
    /// early morning into the rush hour left the streets empty for minutes.
    pub fn clock_jumped(&mut self) {
        if !self.mirror {
            self.initial = true;
        }
    }

    pub fn set_lan_seed(&mut self, seed: u64) {
        self.rng = seed | 1;
    }

    /// Build the network from the lanes collected by `World::build_scene` and load the AI
    /// car types of the map's `ailists.cfg` (the `[aigroup_2]` groups that are not depots).
    pub fn new(root: &Path, world: &World, target: usize) -> Result<Traffic> {
        // (read once: the options' share of random traffic and count of timetable vehicles)
        let settings = crate::settings::Settings::load();
        let (lanes, parked_cars, lane_tiles) = take_from_tiles(world);
        let mut net = Network {
            lanes,
            // (`[lht]`: priority to the left, passing on the right, keeping left)
            left_hand: world.global.left_hand_traffic,
            ..Default::default()
        };
        if net.left_hand {
            log::info!("traffic: the map drives on the left");
        }
        net.link(LANE_LINK_TOL);
        let mut types = Vec::new();
        let mut groups: Vec<omsi_map::ailists::UnschedGroup> = Vec::new();
        let mut group_uvg: Vec<Option<usize>> = Vec::new();
        let mut uvg_defaults: Vec<i32> = Vec::new();
        // `unsched_trafficdens.txt`: per random group a factor and its density over the day
        // (by day of the week); the global.cfg curve is the fallback of maps without it
        let dens: Vec<omsi_map::ailists::UnschedGroup> =
            omsi_cfg::CfgFile::read(&world.map_dir.join("unsched_trafficdens.txt"))
                .ok()
                .map(|f| omsi_map::ailists::parse_unsched_trafficdens(&f))
                .unwrap_or_default();
        let group_curves = !dens.is_empty();
        {
            // `unsched_vehgroups.txt` names the groups the random traffic is made of. The
            // other `[aigroup_2]`s exist only for the timetable: on Berlin-Spandau "Pan Am"
            // and "Mi-8 Soviet AF" fly TXL.ttl and Relais.ttl, and taking them into the
            // random pool put airliners on the flight paths at any hour of the day. Its
            // number is the group's default density on the paths without a `[rule]
            // trafficdensity` for it (see `uvg_density`): 0 means only where the paths ask
            // for the group. Taken as "off", Spandau had no trucks and no Trabant at all,
            // though 865 paths ask for the one and 462 around Falkensee for the other.
            // `OMSI_TRAFFIC_ALL_GROUPS=1` lets such groups drive everywhere (and, on a map
            // without the file, every group, not only the default one).
            let all_groups = omsi_cfg::env::var_os("OMSI_TRAFFIC_ALL_GROUPS").is_some();
            let unscheduled: Option<Vec<(String, i32)>> =
                omsi_cfg::CfgFile::read(&world.map_dir.join("unsched_vehgroups.txt"))
                    .ok()
                    .map(|f| {
                        omsi_map::ailists::parse_unsched_vehgroups(&f)
                            .into_iter()
                            .map(|(n, c)| (n.trim().to_ascii_lowercase(), c))
                            .collect()
                    });
            if let Some(names) = &unscheduled {
                log::info!("random traffic groups (unsched_vehgroups.txt): {names:?}");
                uvg_defaults = names
                    .iter()
                    .map(|n| if all_groups && n.1 <= 0 { 1 } else { n.1 })
                    .collect();
            }
            let lists = &world.ailists;
            // Without `unsched_vehgroups.txt` the random traffic is the ailists' default group
            // alone (the first, or the one the `[ailist]` header names): Omsi.exe 0x785f98
            // makes one nameless group then, and a nameless group takes the default group.
            // Taking every group instead, a map whose ailists keep an ambulance (or a bus,
            // or a lorry) in a group of its own had one car in four of that kind (#1025).
            if unscheduled.is_none() && !all_groups {
                if let Some(g) = lists.groups.get(lists.default_group) {
                    log::info!("random traffic: no unsched_vehgroups.txt, only the default AI group {}", g.name);
                }
            }
            for (_, g) in lists.groups.iter().enumerate().filter(|(i, g)| {
                !g.is_depot
                    && g.hof.is_none()
                    && (unscheduled.is_some() || all_groups || *i == lists.default_group)
                    && !g
                        .vehicles
                        .iter()
                        .any(|v| v.file.to_ascii_lowercase().ends_with(".zug"))
            }) {
                let lname = g.name.trim().to_ascii_lowercase();
                let uvg = match &unscheduled {
                    Some(names) => match names.iter().position(|n| n.0 == lname) {
                        None => continue,
                        Some(u) => {
                            if uvg_defaults.get(u).copied().unwrap_or(0) <= 0 {
                                log::info!(
                                    "random traffic group {} drives only where its paths ask for it (unsched_vehgroups.txt)",
                                    g.name
                                );
                            }
                            Some(u)
                        }
                    },
                    None => None,
                };
                let gi = groups.len();
                group_uvg.push(uvg);
                groups.push(
                    dens.iter()
                        .find(|d| d.name.trim().eq_ignore_ascii_case(g.name.trim()))
                        .cloned()
                        .unwrap_or(omsi_map::ailists::UnschedGroup {
                            name: g.name.clone(),
                            factor: if group_curves { 0.0 } else { 1.0 },
                            densities: Vec::new(),
                        }),
                );
                for v in &g.vehicles {
                    let lower = v.file.to_ascii_lowercase();
                    if lower.ends_with(".zug")
                        || lower.contains("trains\\")
                        || lower.contains("trains/")
                    {
                        continue;
                    }
                    let path = omsi_cfg::resolve_path(root, &v.file);
                    match VehicleType::load_ai(root, &path) {
                        Ok(t) => {
                            // rail (only as scheduled trains), 3 = aircraft on flight paths
                            let rail = t.def.is_rail();
                            let air =
                                matches!(t.def.kind, omsi_vehicle::vehicle::VehicleKind::Other(3));
                            if rail {
                                log::debug!(
                                    "AI vehicle {} is rail-bound, not street traffic",
                                    v.file
                                );
                            } else {
                                types.push((
                                    Arc::new(t),
                                    v.weight.max(0.0),
                                    if air { LaneKind::Air } else { LaneKind::Street },
                                    gi,
                                ));
                            }
                        }
                        Err(e) => log::warn!("AI vehicle {}: {e}", v.file),
                    }
                }
            }
        }
        // No buses in the random road traffic: the depot groups of `ailists.cfg` are the
        // fleet the *timetable* drives, and OMSI puts a bus on a street only because a trip
        // of the map's TTData runs there. Mixing the depot fleet into the random pool put
        // the map's one bus type on every road of the map - on Grundorf that is a single
        // articulated GN92, which is why it seemed to be a type of our own choosing.
        if let Ok(list) = omsi_cfg::env::var("OMSI_DEBUG_LANES") {
            // lane indices, or `at:x,y,r` for the street lanes passing within r m of a point
            // (the indices change from run to run on a map whose tiles load in parallel)
            let chosen: Vec<usize> = match list.strip_prefix("at:") {
                Some(rest) => {
                    let v: Vec<f64> = rest
                        .split(',')
                        .filter_map(|x| x.trim().parse().ok())
                        .collect();
                    let (p, r) = (
                        DVec3::new(
                            v.first().copied().unwrap_or(0.0),
                            v.get(1).copied().unwrap_or(0.0),
                            0.0,
                        ),
                        v.get(2).copied().unwrap_or(10.0),
                    );
                    (0..net.lanes.len())
                        .filter(|&i| {
                            net.lanes[i].kind == LaneKind::Street
                                && net.lanes[i]
                                    .points
                                    .iter()
                                    .any(|q| (q.truncate() - p.truncate()).length() < r)
                        })
                        .collect()
                }
                None => list
                    .split(',')
                    .filter_map(|v| v.trim().parse::<usize>().ok())
                    .filter(|&i| i < net.lanes.len())
                    .collect(),
            };
            for i in chosen {
                let l = &net.lanes[i];
                let samples: Vec<String> = (0..l.points.len())
                    .step_by((l.points.len() / 8).max(1))
                    .map(|k| {
                        format!(
                            "[{:.1} m h {:.1} k {:.3}]",
                            l.dist[k],
                            l.headings[k],
                            l.curvature.get(k).copied().unwrap_or(0.0)
                        )
                    })
                    .collect();
                log::info!("lane {i}: {} {:?} rev {} turn {} prio {} len {:.1} start ({:.1}, {:.1}) end ({:.1}, {:.1}) next {:?} light {:?} crossings {:?} {}", l.name, l.key, l.reversed, l.turn, l.priority, l.length(), l.start().x, l.start().y, l.end().x, l.end().y, l.next, l.traffic_light, net.crossings.get(i), samples.join(" "));
            }
        }
        if omsi_cfg::env::var_os("OMSI_DEBUG_WHEELS").is_some() {
            for (t, ..) in &types {
                let v = VehicleInstance::new(
                    t.clone(),
                    omsi_sim::VehicleHost::new(omsi_sim::SimClock::default()),
                );
                for line in v.wheel_pivot_report() {
                    log::info!("wheel pivot: {line}");
                }
            }
        }
        let lights = world.traffic_lights.lock().clone();
        let controller_of_object = world.controller_of_object.lock().clone();
        let parked: HashMap<usize, Vec<Parked>> = HashMap::new();
        let turning = net.lanes.iter().filter(|l| l.turn != 0).count();
        let with_side = net
            .lanes
            .iter()
            .filter(|l| l.left.is_some() || l.right.is_some())
            .count();
        let turn_lanes = net
            .lanes
            .iter()
            .filter(|l| {
                (l.left.is_some() || l.right.is_some())
                    && l.next.iter().any(|&n| net.lanes[n].turn != 0)
            })
            .count();
        let closed = net
            .lanes
            .iter()
            .filter(|l| l.no_cars || l.density <= 0.0)
            .count();
        let closed_junctions = net
            .lanes
            .iter()
            .filter(|l| (l.no_cars || l.density <= 0.0) && l.source == 2)
            .count();
        let quiet = net
            .lanes
            .iter()
            .filter(|l| l.density < 1.0 && l.density > 0.0)
            .count();
        let prio = net
            .lanes
            .iter()
            .filter(|l| (l.priority - omsi_sim::traffic::DEFAULT_PRIORITY).abs() > 0.5)
            .count();
        log::info!("traffic: {} lanes ({turning} turning, {with_side} with a neighbour, {turn_lanes} where a turn lane applies, {closed} closed to cars of which {closed_junctions} are junctions, {quiet} with less traffic by [rule], {prio} with a [rule] priority), {} AI vehicle types in {} groups, {} light programs, {} lamps", net.lanes.len(), types.len(), groups.len(), lights.len(), world.light_objects.lock().len());
        // lights on paths a car reaches from a lit path of the same crossing (they hold
        // only a car that comes into the crossing there, see `light_at_entry`)
        let inner_lights = (0..net.lanes.len())
            .filter(|&l| {
                let lane = &net.lanes[l];
                lane.traffic_light.is_some()
                    && lane.source == 2
                    && net.prev.get(l).is_some_and(|ps| {
                        ps.iter().any(|&p| {
                            let q = &net.lanes[p];
                            q.source == 2 && q.traffic_light.is_some() && q.key.map(|k| (k.tile, k.id)) == lane.key.map(|k| (k.tile, k.id))
                        })
                    })
            })
            .count();
        log::info!("traffic: {inner_lights} lit paths inside crossings (a car already in the crossing is not held there again)");
        let light_log = omsi_cfg::env::var("OMSI_DEBUG_LIGHTS").ok();
        let light_prev = lights.iter().map(|c| vec![-100; c.lights.len()]).collect();
        let lanes = 0..net.lanes.len();
        let street_weight = net.lanes.iter().filter_map(street_lane_weight).sum();
        let sim = omsi_traffic::Simulation::new(&net, lights);
        let mut t = Traffic {
            net,
            street_weight,
            parked,
            parked_waiting: Vec::new(),
            lane_tiles: lane_tiles.into_iter().collect(),
            lanes_generation: 0,
            types,
            groups,
            group_curves,
            group_uvg,
            uvg_defaults: Arc::new(uvg_defaults),
            cars: Vec::new(),
            dormant: Vec::new(),
            dormant_time: 0.0,
            rng: 0x9E37_79B9_7F4A_7C15,
            target,
            lights_only: false,
            spawn_radius: 400.0,
            time: 0.0,
            released: Vec::new(),
            camera: None,
            orphan_sounds: Vec::new(),
            sim,
            controller_of_object,
            trailer_types: HashMap::new(),
            sound_cfgs: HashMap::new(),
            root: root.to_path_buf(),
            stop_wishes: None,
            player_still: 0.0,
            day_time: 0.0,
            time_scale: 1.0,
            weekday: 0,
            night: false,
            daylight: None,
            next_id: 1,
            last_overtaker: None,
            first_turner: None,
            first_red: None,
            first_yield: None,
            first_passer: None,
            density_curve: world.global.traffic_density_road.clone(),
            unsched_factor: settings.ai_unsched_factor,
            max_scheduled: settings.ai_max_scheduled,
            viewer: None,
            occluders: None,
            walkers: Vec::new(),
            people: Vec::new(),
            initial: true,
            last_dt: 0.0,
            lamp_dt: 0.0,
            trace: open_trace(),
            logged_hard: Default::default(),
            light_log,
            light_prev,
            debug_population: omsi_cfg::env::var_os("OMSI_DEBUG_POPULATION").is_some(),
            appeared_far: 0,
            removed_far: 0,
            edge_bound: Default::default(),
            edge_lanes: Vec::new(),
            framed_spawns: Vec::new(),
            player: None,
            player_priority: false,
            player_blinker: 0,
            player_signalling: 0.0,
            others: Vec::new(),
            drivers: HashMap::new(),
            driver_pool: Vec::new(),
            tick_split: [0.0; 3],
            tick_inner: [0.0; 3],
            links: CarLinks::default(),
            tick_inner_sum: ([0.0; 3], 0),
            others_still: HashMap::new(),
            removed_scheduled: Vec::new(),
            dead_ended: Vec::new(),
            riders_out: Vec::new(),
            through: Default::default(),
            respawned: Vec::new(),
            respawn_block: HashMap::new(),
            respawn_refused: HashMap::new(),
            twinned: Default::default(),
            keep_clear: Vec::new(),
            mirror: false,
            lan_centers: Vec::new(),
            clock_seen: f64::NAN,
            recorder: omsi_traffic::record::Recorder::new(),
            people_records: Vec::new(),
            map_name: world.map_dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "map".into()),
            stuck_saved: Default::default(),
            auto_saves: (0, 0.0),
            overlap_checked: 0.0,
            pending_pull_out: None,
        };
        t.sort_parked(parked_cars, lanes);
        t.mark_loaded_edges(world);
        if let Ok(path) = omsi_cfg::env::var("OMSI_TRAFFIC_RECORD") {
            if let Err(e) = t.start_recording(Path::new(&path)) {
                log::warn!("traffic: cannot record to {path}: {e}");
            }
        }
        t.dump_network();
        Ok(t)
    }

    /// Attach explicitly listed cars (a `.zug` train): (type, reversed).
    pub fn attach_cars(
        &mut self,
        world: &World,
        renderer: &Renderer,
        scene: &mut Scene,
        car: usize,
        cars: &[(Arc<VehicleType>, bool)],
    ) {
        let c = &mut self.cars[car];
        for (t, rev) in cars {
            c.trailer_renders
                .push(world.add_vehicle_shared(renderer, scene, t, None, Some(&c.render)));
            c.vehicle.attach_trailer_ex(t.clone(), *rev);
        }
    }

    /// The cars of car `ci`'s train, front to back, each with whether it is turned round
    /// (the first, the one that drives, is not).
    pub(crate) fn consist(&self, ci: usize) -> Vec<(Arc<VehicleType>, bool)> {
        let v = &self.cars[ci].vehicle;
        std::iter::once((v.ty.clone(), false)).chain(v.trailers.iter().map(|t| (t.ty.clone(), t.reversed))).collect()
    }

    /// Couple `cars` behind car `ci` instead of the ones it has (a train made up anew).
    pub(crate) fn set_trailers(&mut self, world: &World, renderer: &Renderer, scene: &mut Scene, ci: usize, cars: &[(Arc<VehicleType>, bool)]) {
        let c = &mut self.cars[ci];
        for r in c.trailer_renders.drain(..) {
            world.release_vehicle(renderer, scene, r);
        }
        c.vehicle.trailers.clear();
        self.attach_cars(world, renderer, scene, ci, cars);
    }

    /// Turn train `ci` round as Omsi.exe does for a trip whose `[trainreverse]` differs from
    /// how the train stands (0x613a98): the whole consist the other way, its last car
    /// leading - here the vehicle that drives is made anew as that car, standing where it
    /// stands (at `s` on `lane`, which runs the new way), and the others coupled behind it
    /// in the opposite order, each turned round. `behind`: the lanes the train has behind
    /// it now, nearest last, for the track its cars stand on. The car keeps its id and its
    /// service.
    pub(crate) fn turn_train(&mut self, world: &World, renderer: &Renderer, scene: &mut Scene, ci: usize, lane: usize, s: f32, behind: &[usize], reversed: bool) {
        let cars: Vec<(Arc<VehicleType>, bool)> = self.consist(ci).into_iter().rev().map(|(t, r)| (t, !r)).collect();
        let Some((lead, lead_turned)) = cars.first().cloned() else { return };
        if lead_turned {
            // (a lead car turned round is drawn facing the way: none of the stock trains has one)
            log::debug!("train {}: its last car leads turned round", self.cars[ci].id);
        }
        let (id, seed, scheme) = (self.cars[ci].id, self.cars[ci].seed, self.cars[ci].scheme);
        // (where its cars stood, front to back: they stand there still, the other way round)
        let before: Vec<DVec3> = std::iter::once(self.cars[ci].vehicle.position).chain(self.cars[ci].vehicle.trailers.iter().map(|t| t.position)).collect();
        let center = self.viewer.map(|v| v.pos).unwrap_or_default();
        let kind = self.net.lanes[lane].kind;
        self.create_car(world, renderer, scene, center, kind, lane, s, lead, seed, Some(scheme), Some(id), Some(0.0), None);
        let Some(mut new) = self.cars.pop() else { return };
        let old = &mut self.cars[ci];
        // its service goes with it (its line and destination are set for the trip it takes
        // on); the way it drives, from where it stands
        new.vehicle.host.hof = old.vehicle.host.hof.clone();
        new.bus = old.bus.take();
        new.ai.state.max_speed_kmh = old.ai.state.max_speed_kmh;
        new.ai.state.length = old.ai.state.length;
        new.ai.state.accel = old.ai.state.accel;
        new.ai.state.decel = old.ai.state.decel;
        new.ai.state.lat_accel = old.ai.state.lat_accel;
        new.ai.state.min_gap = old.ai.state.min_gap;
        new.ai.state.speed = 0.0;
        new.consist_reversed = reversed;
        let old = std::mem::replace(&mut self.cars[ci], new);
        self.orphan_sounds.extend(old.sounds);
        for r in std::iter::once(old.render).chain(old.trailer_renders) {
            world.release_vehicle(renderer, scene, r);
        }
        self.set_trailers(world, renderer, scene, ci, &cars[1..]);
        self.seed_rail_trail(ci, behind);
        let c = &mut self.cars[ci];
        let trail = &c.rail_trail;
        let (state, net) = (&c.ai.state, &self.net);
        c.vehicle.retrail(0.0, &|d| Some(rail_behind(trail, state, net, d)));
        let after: Vec<DVec3> = std::iter::once(c.vehicle.position).chain(c.vehicle.trailers.iter().map(|t| t.position)).collect();
        let moved = before.iter().rev().zip(&after).map(|(a, b)| (*a - *b).truncate().length()).fold(0.0f64, f64::max);
        log::info!(
            "train {id} turned round: {} (its cars {:.1} m at most from where they stood)",
            std::iter::once(&c.vehicle.ty).chain(c.vehicle.trailers.iter().map(|t| &t.ty)).map(|t| t.def.path.file_stem().unwrap_or_default().to_string_lossy().to_string()).collect::<Vec<_>>().join(" + "),
            moved
        );
    }

    /// The track behind rail car `ci` as it has just been put on its lane: back along its
    /// lane and then `behind` (the lanes before it, nearest last), for its coupled cars.
    fn seed_rail_trail(&mut self, ci: usize, behind: &[usize]) {
        let c = &mut self.cars[ci];
        let net = &self.net;
        let odo = c.ai.state.odometer as f64;
        let mut pts: Vec<(f64, DVec3)> = Vec::new();
        let (mut lane, mut s) = (c.ai.state.lane, c.ai.state.s);
        let mut back = behind.iter().rev();
        let mut d = 0.0f64;
        while d <= RAIL_TRAIL {
            pts.push((odo - d, net.lanes[lane].at(s.max(0.0)).0));
            s -= 1.0;
            if s < 0.0 {
                match back.next() {
                    Some(&l) => {
                        s += net.lanes[l].length();
                        lane = l;
                    }
                    None => break,
                }
            }
            d += 1.0;
        }
        c.rail_trail = pts.into_iter().rev().collect();
    }

    /// The vehicles coupled behind `ty` (its rear sections, trailers, the cars of a unit),
    /// each with whether it is turned round, loaded (once per file). As Omsi.exe builds a
    /// consist (0x70a174): towards the back of the train a vehicle goes on with its
    /// `[couple_back]`, or with its `[couple_front]` when it is itself turned round; the
    /// coupled one is turned round when the coupling's flag says so, against the one it
    /// hangs on; and a coupling back to the file it came from that turns nothing round is
    /// not followed. (Following `[couple_back]` whatever the way, the Berlin A3's unit -
    /// the S car and its K car turned round behind it, whose own `[couple_back]` names the
    /// S car again - went on S, K, S, K, S, none of them turned.)
    pub(crate) fn trailer_chain(&mut self, ty: &Arc<VehicleType>) -> Vec<(Arc<VehicleType>, bool)> {
        self.coupled_chain(ty, false, true)
    }

    /// See [`Traffic::trailer_chain`]: from `ty` (turned round: `rev`) towards the back of
    /// the train, or towards its front, the nearest first.
    pub(crate) fn coupled_chain(&mut self, ty: &Arc<VehicleType>, rev: bool, toward_back: bool) -> Vec<(Arc<VehicleType>, bool)> {
        let mut out = Vec::new();
        let (mut lead, mut lead_rev) = (ty.clone(), rev);
        let mut seen = vec![lead.def.path.clone()];
        for _ in 0..crate::spawn::MAX_COUPLED_PARTS {
            let Some((path, r)) = crate::spawn::next_coupled(&lead.def, lead_rev, toward_back) else {
                break;
            };
            // a consist that comes round again ends here (see `spawn::chain_has_part`)
            if crate::spawn::chain_has_part(&seen, &path) {
                break;
            }
            let root = self.root.clone();
            let t =
                self.trailer_types.entry(path.clone()).or_insert_with(
                    || match VehicleType::load_ai(&root, &path) {
                        Ok(t) => Some(Arc::new(t)),
                        Err(e) => {
                            log::warn!("trailer {}: {e}", path.display());
                            None
                        }
                    },
                );
            let Some(t) = t.clone() else { break };
            out.push((t.clone(), r));
            seen.push(path);
            lead = t;
            lead_rev = r;
        }
        out
    }

    /// Load and attach the `[couple_back]` chain of `vehicle`; returns the renders.
    fn attach_trailers(
        &mut self,
        world: &World,
        renderer: &Renderer,
        scene: &mut Scene,
        vehicle: &mut VehicleInstance,
        scheme: Option<usize>,
        lead: &VehicleRender,
    ) -> Vec<VehicleRender> {
        let mut renders = Vec::new();
        let ty = vehicle.ty.clone();
        for (t, rev) in self.trailer_chain(&ty) {
            renders.push(world.add_vehicle_shared(
                renderer,
                scene,
                &t,
                scheme.filter(|i| *i < t.paint_schemes.len()),
                Some(lead),
            ));
            vehicle.attach_trailer_ex(t, rev);
        }
        renders
    }

    fn rand(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn rand_f(&mut self) -> f64 {
        (self.rand() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// How much traffic group `g` makes now: its `unsched_trafficdens.txt` factor times its
    /// curve for this day of the week (+1 Monday to Friday, +2 Saturday, +4 Sunday, 0 every
    /// day).
    fn group_density(&self, g: usize) -> f32 {
        let Some(gr) = self.groups.get(g) else {
            return 0.0;
        };
        if !self.group_curves {
            return 1.0;
        }
        let bit = match self.weekday {
            0..=4 => 1,
            5 => 2,
            _ => 4,
        };
        let hour = (self.day_time.rem_euclid(86400.0) / 3600.0) as f32;
        match gr
            .densities
            .iter()
            .find(|(mask, _)| *mask == 0 || mask & bit != 0)
        {
            Some((_, curve)) => gr.factor * omsi_map::global::curve_at(curve, hour).max(0.0),
            None => 0.0,
        }
    }

    /// The street traffic density now, 1 = the map's normal level, times the options' share
    /// of random traffic.
    fn street_density(&self) -> f32 {
        self.street_density_map() * self.unsched_factor
    }

    fn street_density_map(&self) -> f32 {
        if !self.group_curves {
            return omsi_map::global::curve_at(
                &self.density_curve,
                (self.day_time.rem_euclid(86400.0) / 3600.0) as f32,
            )
            .clamp(0.0, 2.0);
        }
        let groups: Vec<usize> = (0..self.groups.len())
            .filter(|&g| {
                self.types
                    .iter()
                    .any(|t| t.3 == g && t.2 == LaneKind::Street)
            })
            .collect();
        let factors: f32 = groups.iter().map(|&g| self.groups[g].factor).sum();
        if factors <= 0.0 {
            return 0.0;
        }
        (groups.iter().map(|&g| self.group_density(g)).sum::<f32>() / factors).clamp(0.0, 2.0)
    }

    /// How much of group `g`'s traffic `lane` carries: the path's `[rule] trafficdensity`
    /// for the group, else the group's default (see `uvg_defaults`).
    fn lane_group_density(&self, lane: &omsi_sim::traffic::Lane, g: usize) -> f32 {
        match self.group_uvg.get(g).copied().flatten() {
            Some(u) => lane.pool_density(&self.uvg_defaults, u),
            None => lane.density,
        }
    }

    /// A random vehicle type for a lane of `kind`: on a street `lane`, of the groups that
    /// lane carries, as much as it carries of each.
    fn pick_type(&mut self, kind: LaneKind, lane: Option<usize>) -> Option<Arc<VehicleType>> {
        // a vehicle's share: its weight within its group times what the group makes now
        let group_weight: Vec<f32> = (0..self.groups.len())
            .map(|g| {
                self.types
                    .iter()
                    .filter(|t| t.3 == g && t.2 == kind)
                    .map(|t| t.1)
                    .sum::<f32>()
            })
            .collect();
        let dens: Vec<f32> = (0..self.groups.len())
            .map(|g| {
                if kind == LaneKind::Street {
                    let here = lane
                        .and_then(|i| self.net.lanes.get(i))
                        .map(|l| self.lane_group_density(l, g))
                        .unwrap_or(1.0);
                    self.group_density(g) * here
                } else {
                    1.0
                }
            })
            .collect();
        // (and only the vehicles the lane is open to: Grundorf's trucks where it says
        // `trucks`, Omsi.exe 0x71d714)
        let barred: Vec<*const VehicleType> = match lane.filter(|_| kind == LaneKind::Street).and_then(|i| self.net.lanes.get(i)) {
            Some(l) => self.types.iter().filter(|t| !l.allows(t.0.def.ai_veh_type)).map(|t| Arc::as_ptr(&t.0)).collect(),
            None => Vec::new(),
        };
        let weight = |t: &(Arc<VehicleType>, f32, LaneKind, usize)| -> f32 {
            let gw = group_weight.get(t.3).copied().unwrap_or(0.0);
            if gw <= 0.0 || barred.contains(&Arc::as_ptr(&t.0)) {
                0.0
            } else {
                t.1 / gw * dens.get(t.3).copied().unwrap_or(0.0)
            }
        };
        let total: f32 = self.types.iter().filter(|t| t.2 == kind).map(weight).sum();
        if total <= 0.0 {
            return None;
        }
        let mut x = self.rand_f() as f32 * total;
        for t in self.types.iter().filter(|t| t.2 == kind) {
            let w = weight(t);
            if x < w {
                return Some(t.0.clone());
            }
            x -= w;
        }
        self.types
            .iter()
            .filter(|t| t.2 == kind)
            .last()
            .map(|t| t.0.clone())
    }

    /// Put parked cars onto the lanes they stand in or beside: (distance along the lane,
    /// signed lateral offset, + = right). `cars` are the ones the tiles placed since the last
    /// call; the cars no lane was found for before are tried again where the lanes `added`
    /// just came in.
    fn sort_parked(&mut self, cars: Vec<ParkedCar>, added: std::ops::Range<usize>) {
        let mut todo: Vec<ParkedCar> = Vec::new();
        if !added.is_empty() && !self.parked_waiting.is_empty() {
            let cells: hashbrown::HashSet<(i32, i32)> = added
                .clone()
                .flat_map(|i| Network::lane_cells(&self.net.lanes[i]))
                .collect();
            let near = |p: &DVec3| {
                let (cx, cy) = Network::grid_cell(*p);
                (-1..=1).any(|dx| (-1..=1).any(|dy| cells.contains(&(cx + dx, cy + dy))))
            };
            let (retry, keep): (Vec<ParkedCar>, Vec<ParkedCar>) = std::mem::take(&mut self.parked_waiting)
                .into_iter()
                .partition(|p| near(&p.0));
            self.parked_waiting = keep;
            todo = retry;
        }
        let new_cars = cars.len();
        todo.extend(cars);
        if todo.is_empty() {
            return;
        }
        let mut on_lanes = 0usize;
        for car in todo {
            // [round 7 A10] its real box (`[boundingbox]`, as `World::parked_boxes` has it),
            // matched and placed by the box's middle
            let (origin, heading, bb) = car;
            let obb = omsi_sim::collision::Obb::from_box(bb.unwrap_or(PARKED_BOX), origin, heading);
            let p = obb.center.extend(origin.z);
            // a car beside a lane is within a few metres of it: the lanes of the cells
            // around it are enough, and a car in a car park finds none
            let beside = self
                .net
                .nearest_lane_near(p, LaneKind::Street)
                .filter(|(l, _, d)| *d <= (self.net.lanes[*l].width * 0.5).max(1.5) as f64 + 1.6);
            let Some((l, s, _)) = beside else {
                self.parked_waiting.push(car);
                continue;
            };
            let (q, h) = self.net.lanes[l].at(s);
            let hr = (h as f64).to_radians();
            let right = DVec3::new(hr.cos(), -hr.sin(), 0.0);
            let lat = (p - q).dot(right) as f32;
            if lat.abs() < 0.9 && omsi_cfg::env::var_os("OMSI_DEBUG_TRAFFIC").is_some() {
                let lane = &self.net.lanes[l];
                log::info!("parked car at ({:.1}, {:.1}) stands in lane {l} ({} {:?}, width {:.1}, heading {:.0} there, s {s:.1} of {:.1}, {lat:+.2} m to the side)", p.x, p.y, lane.name, lane.key, lane.width, h, lane.length());
            }
            // (its half extents along and across the lane: a car parked askew reaches further
            // into it)
            let rel = (obb.heading - hr).sin_cos();
            let (hw, hl) = (obb.half.x, obb.half.y);
            let half_len = (rel.1.abs() * hl + rel.0.abs() * hw) as f32;
            let half_width = (rel.0.abs() * hl + rel.1.abs() * hw) as f32;
            if bb.is_some_and(|b| b[1] > PARKED_BOX[1] + 0.5 || b[0] > PARKED_BOX[0] + 0.3) && omsi_cfg::env::var_os("OMSI_DEBUG_TRAFFIC").is_some() {
                log::info!("parked car at ({:.1}, {:.1}) beside lane {l} at s {s:.1}, {lat:+.2} m: box {:.2} x {:.2} m (half {half_width:.2} across, {half_len:.2} along)", p.x, p.y, 2.0 * hw, 2.0 * hl);
            }
            self.parked.entry(l).or_default().push(Parked { s, lateral: lat, half_width, half_len });
            on_lanes += 1;
        }
        if omsi_cfg::env::var_os("OMSI_DEBUG_TRAFFIC").is_some() {
            log::info!("traffic: {new_cars} parked cars placed, {on_lanes} more stand in or beside a lane ({} in all, {} not beside one)", self.parked.values().map(|v| v.len()).sum::<usize>(), self.parked_waiting.len());
        }
    }

    /// Is a vehicle of radius `r` at `p` hidden from the viewer by buildings (or a hill)?
    /// Every line of sight to it must be: to its middle, to both ends whichever way it
    /// points and over its roof. A single line to the middle let a car come and go half
    /// out from behind a corner, in plain view.
    fn occluded(&self, world: &World, v: &Viewer, p: DVec3, r: f64) -> bool {
        let rel = (p - v.pos).truncate();
        let across = if rel.length() > 1e-3 {
            DVec3::new(-rel.y, rel.x, 0.0).normalize()
        } else {
            DVec3::X
        };
        let along = DVec3::new(rel.x, rel.y, 0.0).normalize_or_zero();
        let reach = (r * 0.8).max(1.5);
        let mut targets = vec![
            p + DVec3::new(0.0, 0.0, 1.2),
            p + across * reach + DVec3::new(0.0, 0.0, 1.0),
            p - across * reach + DVec3::new(0.0, 0.0, 1.0),
            p - along * reach + DVec3::new(0.0, 0.0, 1.0),
        ];
        // the roof of a bus or a lorry shows over a wall a car hides behind
        if r > 4.0 {
            targets.push(p + DVec3::new(0.0, 0.0, 3.2));
        }
        targets.into_iter().all(|t| self.sight_blocked(world, v, t))
    }

    /// Is the line of sight from the viewer to the point `target` blocked by a building
    /// (or a hill)?
    fn sight_blocked(&self, world: &World, v: &Viewer, target: DVec3) -> bool {
        let p = target;
        let blocker = match &self.occluders {
            Some(c) => c.ray_blocker(v.pos, target, 2.5, 3.0),
            None => world.collision.lock().ray_blocker(v.pos, target, 2.5, 3.0),
        };
        if let Some(b) = blocker {
            if self.debug_population {
                log::info!("  line of sight to ({:.0}, {:.0}) blocked by a box at ({:.1}, {:.1}) {:.1} x {:.1} m, z {:.1}..{:.1}", p.x, p.y, b.center.x, b.center.y, b.half.x * 2.0, b.half.y * 2.0, b.z0, b.z1);
            }
            return true;
        }
        // the ground between: a crest or an embankment
        for k in 1..8 {
            let t = k as f64 / 8.0;
            let q = v.pos.lerp(target, t);
            if let Some(g) = terrain_height(world, q.x, q.y) {
                if g > q.z + 0.5 {
                    if self.debug_population {
                        log::info!("  line of sight to ({:.0}, {:.0}) blocked by the ground at ({:.0}, {:.0}): {:.1} over {:.1}", p.x, p.y, q.x, q.y, g, q.z);
                    }
                    return true;
                }
            }
        }
        false
    }

    /// May a vehicle be put on the road at `p` without the player seeing it appear? A
    /// timetable bus asks this before it spawns mid-route.
    pub fn may_appear(&self, world: &World, p: DVec3) -> bool {
        self.initial || self.hidden(world, p, 8.0)
    }

    /// [round 9 A2 fix pass 1] The vehicles' lamps are lit: the AI's lights are on (`night`) or
    /// the coronas glow (`Daylight::night` over `LAMPS_LIT_NIGHT`).
    pub fn lamps_lit(&self) -> bool {
        self.night || self.daylight.as_ref().is_some_and(|d| d.night > LAMPS_LIT_NIGHT)
    }

    /// [round 9 A2] May a random car of body `height` (m) be woken at `p` unseen: as
    /// `may_appear`, or beyond its perceptual horizon (`Viewer::horizon`), where it is a dot
    /// nobody can tell from nothing - the line a car reaching the loaded tiles' edge ends its
    /// trip beyond, too. (A timetable bus keeps `may_appear`; a new car keeps the round-6 rule,
    /// `APPEAR_IN_VIEW_BEYOND`, which already lets it appear in plain view beyond 300 m.)
    pub fn may_appear_car(&self, world: &World, p: DVec3, ty: &VehicleType) -> bool {
        self.may_appear(world, p) || self.viewer.is_some_and(|v| v.beyond_horizon((p - v.pos).length(), body_height(ty), body_radius(ty)))
    }

    /// [round 8 H8-7, fix pass 1] Car `c` stopped at the edge of the loaded tiles (`Lane::
    /// loaded_edge`: the crate stops it there, `Why::End`).
    fn at_loaded_edge(&self, c: &AiCar) -> bool {
        c.ai.last.why == Why::End
            && c.ai.stopped > AT_END_STAND
            && self.net.lanes.get(c.ai.state.lane).is_some_and(|l| l.loaded_edge && l.next.is_empty())
    }

    /// The world is still being built (the first populate, the first seconds): vehicles
    /// may be put anywhere.
    pub fn loading_phase(&self) -> bool {
        self.initial || self.time < 2.0
    }

    /// Could the player not see a vehicle (radius `r`) at `p` appear or vanish? Never close
    /// by: mirrors, a turn of the head and the gaps between houses see what is near,
    /// whatever the collision boxes say (a bus let appear 40 m away behind a box that stood
    /// for a building with a gateway in it was seen popping up in the middle of the street).
    /// Within `NEAR_HIDE` only behind a building or the ground, wherever the camera looks -
    /// the mirrors look back, and the head turns: buses appeared 160 m behind the player
    /// in plain sight of the mirrors, and a bus waiting at the edge of the loaded route
    /// vanished beside the player's bus because the camera was looking ahead. Further off:
    /// beyond what is drawn, out of the picture, or behind something.
    /// (`Viewer::hides_plainly`, or behind something past `NEVER_VANISH_WITHIN`: one rule.)
    pub fn hidden(&self, world: &World, p: DVec3, r: f64) -> bool {
        let Some(v) = self.viewer else { return true };
        v.hides_plainly(p, r) || ((p - v.pos).length() >= NEVER_VANISH_WITHIN && self.occluded(world, &v, p, r))
    }

    /// Spawn cars until `target` are within `spawn_radius` of `center`; despawn far ones.
    pub fn populate(
        &mut self,
        world: &World,
        renderer: &Renderer,
        scene: &mut Scene,
        center: DVec3,
    ) {
        self.populate_seen(world, renderer, scene, center, None);
    }

    /// Keep the population around the player: cars are taken off only where nobody sees
    /// it (far away and out of view, or behind a building), and new ones appear only there.
    /// `view` (a unit vector) stands in for the viewer when none has been set.
    pub fn populate_seen(
        &mut self,
        world: &World,
        renderer: &Renderer,
        scene: &mut Scene,
        center: DVec3,
        view: Option<DVec3>,
    ) {
        if self.mirror {
            return;
        }
        if self.viewer.is_none() {
            if let Some(f) = view {
                self.viewer = Some(Viewer {
                    pos: center,
                    forward: f,
                    tan_x: 1.2,
                    tan_y: 0.6,
                    range: VISIBLE_RANGE,
                    min_size: 0.0,
                    max_dist: 0.0,
                    fov: 1.0,
                    px_height: 0.0,
                    horizon_fov: 0.0,
                    lamps_lit: false,
                });
            }
        }
        let far = self.spawn_radius * DESPAWN_FACTOR;
        self.advance_dormant();
        // the cars somebody stands behind
        let queued: std::collections::HashSet<u64> = self.cars.iter().filter(|c| c.ai.stopped > 5.0).filter_map(|c| c.ai.last.lead_id).collect();
        // [round 9 A2, fix pass 1] beyond the perceptual horizon the loaded tiles' edge is the
        // trip's end: a random car that reaches it there, the first in line, goes at once
        // (`removal::goes_far`); the queue behind it drains one car per arrival
        let far_trip_end: std::collections::HashSet<u64> = match self.viewer {
            Some(v) if !self.initial => far_trip_ends(self.cars.iter().filter(|c| !c.serves_stops()).map(|c| {
                let beyond = v.beyond_horizon((c.vehicle.position - v.pos).length(), body_height(&c.vehicle.ty), body_radius(&c.vehicle.ty));
                (c.id, self.at_loaded_edge(c), beyond, &c.ai)
            })),
            _ => Default::default(),
        };
        let mut i = 0;
        let mut off_ground = 0usize;
        let mut asleep = 0usize;
        while i < self.cars.len() {
            let c = &self.cars[i];
            let p = c.vehicle.position;
            // (the nearest player: a LAN host keeps the traffic around the others too)
            let dist = self
                .lan_centers
                .iter()
                .fold((p - center).length(), |d, o| d.min((p - *o).length()));
            // every car on the road or the rails whose ground has been unloaded goes: the
            // lanes stay in the network when their tile goes, but nothing may drive over
            // ground that is not there (tiles only go well beyond the view, timetable buses
            // included; their trips come back with the tiles)
            let flying = self
                .net
                .lanes
                .get(c.ai.state.lane)
                .map(|l| l.kind == LaneKind::Air)
                .unwrap_or(false);
            let unloaded = !flying && !world.has_ground(p.x, p.y);
            off_ground += unloaded as usize;
            let random = !c.serves_stops() || c.released;
            // (done: given up by the traffic, or a timetable bus let go after its trip)
            let finished = c.ai.gone || c.released;
            let r = (c.ai.state.length as f64 * 0.5).max(2.0);
            // standing on the last lane of the network (the map's edge), given up: a random
            // car that reaches the end is taken off by the traffic itself as Omsi.exe does
            // (`AiState::drive` finds no way on: once 0x71dc9c finds no next segment
            // (0x612e10) its segment stays -1 and 0x6fe3fc deletes it the same frame,
            // 0x703bb0); one held short of it goes as below, out of sight or far off, after
            // a long while, or in view once others queue behind it
            let at_end = finished
                && c.ai.stopped > if c.serves_stops() { AT_END_STAND_BUS } else { AT_END_STAND }
                && c.ai.state.route.is_empty()
                && self.net.lanes[c.ai.state.lane].next.is_empty();
            let from_eye = self.viewer.map(|v| (p - v.pos).length()).unwrap_or(dist);
            // a timetable bus waiting where the loaded part of its route ends
            let at_edge = c.route_open()
                && c.ai.state.speed < 0.1
                && c.ai.state.route.last() == Some(&c.ai.state.lane)
                && c.ai.state.s > self.net.lanes[c.ai.state.lane].length() - 25.0;
            // [round 8 H8-7, fix pass 1] stopped at the edge of the loaded tiles
            let at_loaded_edge = self.at_loaded_edge(c);
            // a random car still on its way that goes out of range sleeps instead (see
            // `DormantCar`); only one whose trip is over leaves the map
            let sleeps_instead = !c.serves_stops() && !c.ai.gone && self.net.lanes.get(c.ai.state.lane).map(|l| l.kind == LaneKind::Street).unwrap_or(false);
            // (the first rule that takes it, for the log)
            // (the rules are asked in order, lazily: `hidden` casts rays)
            let first = |rules: &[(&dyn Fn() -> bool, &'static str)]| rules.iter().find(|r| r.0()).map(|r| r.1);
            let hidden = || self.hidden(world, p, r);
            // [round 8 H8-7] the ground going is no reason of its own to vanish in view: it
            // waits behind `hidden()` like every other removal (the tiles go at 170-200 m, and
            // 129 cars went within 200 m of the eye in 300 s at Jiangzhaikou, in frame); one in
            // sight stays until it is not, or another rule below takes it
            let reason: Option<&'static str> = if unloaded && hidden() {
                Some("its ground unloaded")
            } else if at_edge {
                // (its trip comes back with the tiles; kept until nobody saw it, a bus stood
                // with its passengers at the far end of a straight road for good)
                first(&[(&|| hidden(), "at the loaded route's end, out of sight"), (&|| c.ai.stopped > FAR_OFF_STAND && from_eye > FAR_OFF_DIST, "at the loaded route's end, far off"), (&|| c.ai.stopped > STOOD_LONG, "at the loaded route's end, long")])
            } else if !random {
                None
            } else if far_trip_end.contains(&c.id) {
                // [round 9 A2, fix pass 1] its trip ends at the edge beyond the horizon: once
                // nobody sees it (as before) or now, in frame, a dot
                first(&[(&|| hidden(), "at the loaded tiles' edge, out of sight"), (&|| true, "at the loaded tiles' edge, beyond the horizon")])
            } else if at_loaded_edge {
                // [round 8 H8-7, fix pass 2] standing at the edge of the loaded tiles (`Lane::
                // loaded_edge`, the crate stops it there): it goes once nobody sees it, or in view
                // by the round-7 rule every other car has (`goes_in_view`: given up, its grace
                // run, beyond GAVE_UP_DIST, the front of its queue) - nothing else; a dead-end
                // removal, gated and counted (it drove off the loaded network in view as "the
                // network's end": 607 in frame at 873 m, Jiangzhaikou). The bus route end's
                // FAR_OFF/STOOD_LONG are a timetable bus's alone.
                first(&[
                    (&|| hidden(), "at the loaded tiles' edge, out of sight"),
                    (&|| goes_in_view(c.ai.gone, &c.ai, from_eye), "at the loaded tiles' edge, in view after its grace"),
                ])
            } else if let Some(why) = at_end.then(|| first(&[(&|| hidden(), "at a dead end, out of sight"), (&|| c.ai.stopped > FAR_OFF_STAND && from_eye > FAR_OFF_DIST, "at a dead end, far off"), (&|| c.ai.stopped > STOOD_LONG, "at a dead end, long"), (&|| c.ai.stopped > QUEUE_BEHIND_STAND && queued.contains(&c.id) && from_eye > QUEUE_BEHIND_DIST, "at a dead end, a queue behind it")])).flatten() {
                // (and in view too once others wait behind it: a fire engine at the end of a
                // dead-end street held a queue of fourteen cars for two and a half minutes)
                // (taken at once it vanished in plain view 300 m ahead; but a car kept
                // until nobody could see it stood for good at the end of a long straight
                // road in view, and the queue behind it - timetable buses with their
                // passengers among them - never moved again)
                Some(why)
            } else if finished || dist > far {
                // in plain view a car stays until the renderer leaves it out anyway, and
                // close by (the mirrors, a turn of the head) it stays in any case
                // (one that gave up goes in view too, IN_VIEW_GRACE after giving up, unless
                // right beside the viewer: kept until nobody saw it, a jam at a junction the
                // player watched never cleared)
                // [round5 C4] (not one behind a standing vehicle within QUEUE_LEAD: a queue,
                // whatever its own hold reads - x10 case15: two cars 6 m in front of the
                // player, why=people behind leads 2.2 m ahead, vanished at 90 s)
                // [round 7 B1] (one clock, the crate's: `gone` at GIVE_UP_UNSEEN of
                // `waiting`, SUMO's waiting time: every hold but a red light and its own stop; the round-4
                // phantom rule's own 90 s in view is gone with the second clock)
                first(&[
                    (&|| dist > VISIBLE_RANGE * OUT_OF_RANGE && from_eye > NEAR_HIDE, "out of range"),
                    (&|| hidden(), if finished { "finished, out of sight" } else { "out of sight" }),
                    (&|| goes_in_view(finished, &c.ai, from_eye), "finished, in view after its grace"),
                ])
            } else {
                None
            };
            let remove = reason.is_some();
            if let Some(why) = reason {
                // [round 7 R7-1 fix 1] the rule into the recording, with whether the eye saw it
                let rule = removal_rule(why);
                let frames = || self.viewer.is_some_and(|v| v.frames(p, r));
                let seen = match rule {
                    RemoveRule::OutOfSight => false,
                    RemoveRule::InViewAfterGrace | RemoveRule::DeadEnd | RemoveRule::GroundUnloaded | RemoveRule::FarDeadEnd => frames() && !self.viewer.is_some_and(|v| self.occluded(world, &v, p, r)),
                    _ => frames(),
                };
                self.removed_far += (rule == RemoveRule::FarDeadEnd && seen) as usize;
                // ([round 9 A2] with its distance from the eye: the removals by distance band)
                self.recorder.note_remove_from(c.id, rule, seen, c.ai.waiting, Some(from_eye as f32));
            }
            // ([round 9 A2 fix pass 1] a trip ended at the edge does not drive on asleep)
            if remove && sleeps_instead && reason != Some("at the loaded tiles' edge, beyond the horizon") {
                let c = self.cars.swap_remove(i);
                asleep += 1;
                self.dormant.push(DormantCar {
                    id: c.id,
                    ty: c.vehicle.ty.clone(),
                    kind: LaneKind::Street,
                    lane: c.ai.state.lane,
                    s: c.ai.state.s,
                    speed: c.ai.state.speed.max(2.0),
                    seed: c.seed,
                    scheme: c.scheme,
                    walk: c.seed ^ c.id.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
                });
                self.orphan_sounds.extend(c.sounds);
                for r in std::iter::once(c.render).chain(c.trailer_renders) {
                    world.release_vehicle(renderer, scene, r);
                }
            } else if remove {
                let c = self.cars.swap_remove(i);
                if c.serves_stops() {
                    self.riders_out.push(c.id);
                }
                if c.serves_stops() && (unloaded || at_edge) {
                    self.removed_scheduled.push(c.id);
                }
                if self.debug_population {
                    let v = self.viewer;
                    // ([round 7 B1] with the rule that took it, its distance from the eye, its
                    // last hold and its waiting time, the give-up clock)
                    // ([round 7 R7-1] and what it follows: the lead's id and gap and whether
                    // its speed, the queue test of the in-view rule)
                    let lead = lead_of(&c.ai.last);
                    let in_queue = in_a_queue(&c.ai.last);
                    log::info!("population t={:.1}: car {} removed at ({:.0}, {:.0}), {:.0} m from the player, in frame {}, behind a building {}, {}; {:.0} m from the eye, why {}, stood {:.0} s, waiting {:.1} s, lead {} gap {} speed {}, in a queue {}", self.time, c.id, p.x, p.y, dist, v.map(|v| v.frames(p, r)).unwrap_or(false), v.map(|v| self.occluded(world, &v, p, r)).unwrap_or(false), reason.unwrap_or("-"), from_eye, c.ai.last.why.name(), c.ai.stopped, c.ai.waiting, lead.map(|l| l.to_string()).unwrap_or("-".into()), c.ai.last.lead.map(|l| format!("{:.1}", l.gap)).unwrap_or("-".into()), c.ai.last.lead.map(|l| format!("{:.1}", l.speed)).unwrap_or("-".into()), in_queue);
                }
                self.orphan_sounds.extend(c.sounds);
                for r in std::iter::once(c.render).chain(c.trailer_renders) {
                    world.release_vehicle(renderer, scene, r);
                }
            } else {
                i += 1;
            }
        }
        if off_ground > 0 && omsi_cfg::env::var_os("OMSI_DEBUG_TRAFFIC").is_some() {
            log::info!("traffic: {off_ground} vehicles taken away with the tiles under them");
        }
        if (asleep > 0 || !self.dormant.is_empty()) && self.debug_population {
            // ([round 9 A2] with the appearances and removals in frame beyond the horizon)
            log::info!("population t={:.1}: {asleep} cars went out of range and drive on unseen; {} on the map out of range, {} in range; beyond the horizon ({:.0} m for a car) in frame since the start: {} appeared, {} removed", self.time, self.dormant.len(), self.cars.len(), self.viewer.map(|v| v.horizon(CAR_HEIGHT)).unwrap_or(VISIBLE_RANGE), self.appeared_far, self.removed_far);
        }
        if self.types.is_empty() || self.net.lanes.is_empty() {
            self.initial = false;
            return;
        }
        // made only for the lights: nothing new while the target is 0, but the cars of a
        // target raised and lowered again go as they do anywhere (returning before the loop
        // above, they stood at the map's edge and drove over unloaded tiles for good)
        if self.lights_only && self.target == 0 {
            return;
        }
        // aircraft: a few on the flight paths, independent of the street target
        let has_air = self.types.iter().any(|t| t.2 == LaneKind::Air);
        // the map's traffic density by hour (and group) scales the street traffic ...
        let density = self.street_density();
        // ... and so does how much road there is around: the same number of cars looks
        // empty on a six-lane Berlin junction and crowded on a village lane, so the count
        // asked for is per a neighbourhood of about 250 lanes; and as Omsi spawns on each
        // path at a rate of its [rule] trafficdensity, paths of low density bring fewer cars
        // and those of density 0 (or kept clear of cars) none
        let near_density: Vec<f32> = self.net.lanes_starting_near(center, self.spawn_radius)
            .into_iter()
            .map(|i| &self.net.lanes[i])
            .filter(|l| {
                l.kind == LaneKind::Street
                    && l.points
                        .first()
                        .map(|p| (*p - center).length() < self.spawn_radius)
                        .unwrap_or(false)
            })
            .map(|l| if l.no_cars { 0.0 } else { l.density.clamp(0.0, 4.0) })
            .collect();
        let street_target = (self.target as f32 * density * road_scale(&near_density)).round() as usize;
        // the cars that come into range again, where they have got to
        self.wake_dormant(world, renderer, scene, center, street_target);
        // the whole map's population: as dense as around the player, on every street the
        // map has shown so far (sleeping where the player is not)
        self.fill_map(center, street_target);
        for (kind, target) in [
            (LaneKind::Street, street_target),
            (LaneKind::Air, if has_air { 3 } else { 0 }),
        ] {
            self.populate_kind(world, renderer, scene, center, kind, target);
        }
        self.populate_lan_centers(world, renderer, scene, center, street_target);
        if !self.initial {
            self.pull_out_parked(world, renderer, scene, center);
            self.park_in(world, center);
        }
        self.initial = false;
    }

    /// Now and then a car near the player parks: a space at the kerb that a parked car has
    /// left is taken by a car of the same kind driving up the lane beside it - it indicates,
    /// slows down, stops beside the space, moves over into it and stands there as the
    /// parked car it was. Called with every population pass (about every two seconds).
    pub fn park_in(&mut self, world: &World, center: DVec3) {
        if self.rand_f() >= PARK_IN_CHANCE {
            return;
        }
        let debug = omsi_cfg::env::var_os("OMSI_DEBUG_TRAFFIC").is_some();
        let taken: hashbrown::HashSet<i64> = self.cars.iter().filter_map(|c| c.park.map(|p| p.key)).collect();
        let mut spots = world.free_parking();
        spots.retain(|(k, p)| !taken.contains(k) && (30.0..260.0).contains(&(p.pos - center).truncate().length()));
        spots.sort_by_key(|s| s.0);
        let mut why: Vec<String> = Vec::new();
        while !spots.is_empty() {
            let (key, p) = spots.swap_remove(self.rand() as usize % spots.len());
            let Some((l, s, _)) = self.net.nearest_lane_near(p.pos, LaneKind::Street) else { continue };
            let lane = &self.net.lanes[l];
            if lane.no_cars || s < 8.0 || s > lane.length() - 4.0 {
                continue;
            }
            let (q, h) = lane.at(s);
            let hr = (h as f64).to_radians();
            let lat = (p.pos - q).dot(DVec3::new(hr.cos(), -hr.sin(), 0.0)) as f32;
            let mut dh = (p.heading - h as f64).rem_euclid(360.0);
            if dh > 180.0 {
                dh -= 360.0;
            }
            // beside the lane on the right and in line with it (a space across the kerb or in
            // a row is not driven into)
            if !(1.2..4.2).contains(&lat) || dh.abs() > 12.0 {
                why.push(format!("space {key}: {lat:+.1} m beside lane {l}, {dh:+.0} deg"));
                continue;
            }
            let folder = p.sco.parent().map(|d| d.to_string_lossy().to_lowercase());
            // a car of that kind coming up the lane, far enough off to slow down gently
            let mut best: Option<(usize, f32)> = None;
            for (i, c) in self.cars.iter().enumerate() {
                if c.serves_stops() || c.ai.gone || c.park.is_some() || c.ai.passing.is_some() || c.ai.state.change.is_some() || c.ai.pull_out > 0.0 {
                    continue;
                }
                if c.vehicle.ty.def.path.parent().map(|d| d.to_string_lossy().to_lowercase()) != folder {
                    continue;
                }
                if c.ai.state.speed > 15.0 || c.ai.state.lateral.abs() > 0.2 {
                    continue;
                }
                let Some(&(_, dl)) = omsi_traffic::perception::way_lanes(&self.net, &c.ai.state, 160.0).iter().find(|w| w.0 == l) else { continue };
                let d = dl + s;
                let need = c.ai.state.speed * c.ai.state.speed / 2.0 + 20.0;
                if d > need && d < 150.0 && best.map(|b| d < b.1).unwrap_or(true) {
                    best = Some((i, d));
                }
            }
            let Some((i, d)) = best else {
                why.push(format!("space {key}: no car of its kind coming up lane {l}"));
                continue;
            };
            let car = &mut self.cars[i];
            car.park = Some(ParkPlan { key, lane: l, s, lat, ramped: false, done: false });
            if debug {
                log::info!("car {} parks in the space of parked car {key} ({:.0} m ahead, {lat:.1} m right of lane {l})", car.id, d);
            }
            return;
        }
        if debug && !why.is_empty() {
            log::info!("park-in: none this time ({})", why.join("; "));
        }
    }

    /// Now and then a car parked at the kerb near the player drives off: the parked object
    /// goes (its space stays empty) and the AI car of the same folder takes its place, in
    /// the parking position, indicating, and pulls out into its lane once the road behind
    /// it is clear. Called with every population pass (about every two seconds).
    /// [round 8 H8-1/H8-8, fix pass 2] The parked object that signals to pull out
    /// (`pending_pull_out`): once its signalling time has run, the frame the one pull-out rule
    /// passes (`plan::pull_out_clear`: every vehicle driving up the lane, the players too, beyond
    /// its own secure gap; nobody on the strip its body sweeps from its offset onto the lane) and
    /// its lane point is clear (`spawn_clear`), it leaves the kerb: the AI car is made where it
    /// was parked, `pulling_out`, and drives out over the shortest move it can steer from rest.
    /// Until then it stays a parked object (SUMO: a parked vehicle enters only when
    /// `isInsertionSuccess` passes, MSVehicleTransfer.cpp ~135-149); no AI car waits at the offset.
    /// (The scenery object shows no indicator: the time runs, the lamps do not blink.)
    fn advance_pull_out(&mut self, world: &World, renderer: &Renderer, scene: &mut Scene, center: DVec3) {
        let Some(pp) = self.pending_pull_out.clone() else { return };
        let parked = world.parked_objects.lock().get(&pp.key).map(|p| p.pos);
        let Some(ppos) = parked else {
            self.pending_pull_out = None;
            return;
        };
        if self.time < pp.ready_at {
            return;
        }
        let ramp = omsi_traffic::plan::bypass_min_len(pp.front + pp.rear, pp.lat);
        let on_lane = |lane: usize| -> Vec<(f32, f32, f32, omsi_traffic::DriverParams)> {
            self.cars.iter().filter(|c| c.ai.state.lane == lane).map(|c| (c.ai.state.s, c.ai.state.front, c.ai.state.speed, c.ai.params)).collect()
        };
        let followers = omsi_traffic::plan::pull_out_followers(&self.net, &self.sim, pp.lane, pp.s, pp.rear, &on_lane);
        if !omsi_traffic::plan::pull_out_clear(&self.net, &self.sim, &followers, pp.lane, pp.s, pp.lat, pp.front, pp.rear, pp.half_w, ramp) {
            return;
        }
        let (q, h) = self.net.lanes[pp.lane].at(pp.s);
        if !self.spawn_clear(&pp.ty, q, h as f64) {
            return;
        }
        self.pending_pull_out = None;
        if world.depart_parked(renderer, scene, pp.key).is_none() {
            return;
        }
        if let Some(list) = self.parked.get_mut(&pp.lane) {
            if let Some(j) = (0..list.len()).min_by(|&a, &b| (list[a].s - pp.s).abs().total_cmp(&(list[b].s - pp.s).abs())) {
                if (list[j].s - pp.s).abs() < 3.0 {
                    list.swap_remove(j);
                }
            }
        }
        let id = self.create_car(world, renderer, scene, center, LaneKind::Street, pp.lane, pp.s, pp.ty.clone(), pp.seed, None, None, Some(0.0), None);
        let net = &self.net;
        if let Some(car) = self.cars.iter_mut().find(|c| c.id == id) {
            car.ai.state.lateral = pp.lat;
            car.ai.state.lateral_target = 0.0;
            car.ai.state.lateral_ramp = (pp.lat, 0.0, car.ai.state.odometer, ramp);
            car.body = place_body(net, &car.ai.state, &mut car.vehicle, MotionKind::Road);
            car.ai.pull_out = 0.0;
            car.ai.pulling_out = true;
            car.ai.state.blinker = 1;
        }
        if omsi_cfg::env::var_os("OMSI_DEBUG_TRAFFIC").is_some() {
            log::info!("parked car {} ({}) at ({:.1}, {:.1}) pulls out as car {id}, {:.1} m right of lane {}", pp.key, pp.ty.def.path.display(), ppos.x, ppos.y, pp.lat, pp.lane);
        }
    }

    pub fn pull_out_parked(&mut self, world: &World, renderer: &Renderer, scene: &mut Scene, center: DVec3) {
        if self.pending_pull_out.is_some() {
            self.advance_pull_out(world, renderer, scene, center);
            return;
        }
        if self.rand_f() >= PULL_OUT_CHANCE {
            return;
        }
        let debug = omsi_cfg::env::var_os("OMSI_DEBUG_TRAFFIC").is_some();
        let mut candidates: Vec<(i64, crate::scene::ParkedObject)> = world
            .parked_objects
            .lock()
            .iter()
            .filter(|(_, p)| {
                let d = (p.pos - center).truncate().length();
                (25.0..180.0).contains(&d)
            })
            .map(|(k, p)| (*k, p.clone()))
            .collect();
        candidates.sort_by_key(|c| c.0);
        if debug {
            log::info!("parked pull-out: {} parked cars in range ({} loaded)", candidates.len(), world.parked_objects.lock().len());
        }
        let mut why: Vec<String> = Vec::new();
        let mut tried = 0;
        while !candidates.is_empty() && tried < 8 {
            tried += 1;
            let (key, p) = candidates.swap_remove(self.rand() as usize % candidates.len());
            // the AI car of the same folder (a parked Golf is `parked_vw_golf_2.sco` next to
            // `ai_vw_golf_2.bus`)
            let folder = p.sco.parent().map(|d| d.to_string_lossy().to_lowercase());
            let Some(ty) = self
                .types
                .iter()
                .filter(|t| t.2 == LaneKind::Street)
                .find(|t| t.0.def.path.parent().map(|d| d.to_string_lossy().to_lowercase()) == folder)
                .map(|t| t.0.clone())
            else {
                why.push(format!("no AI car in {folder:?}"));
                continue;
            };
            let Some((l, s, _)) = self.net.nearest_lane_near(p.pos, LaneKind::Street) else {
                why.push("no lane".into());
                continue;
            };
            let lane = &self.net.lanes[l];
            if lane.no_cars || s < 4.0 || s > lane.length() - 4.0 {
                continue;
            }
            let (q, h) = lane.at(s);
            let hr = (h as f64).to_radians();
            let right = DVec3::new(hr.cos(), -hr.sin(), 0.0);
            let lat = (p.pos - q).dot(right) as f32;
            // beside the lane on the right, facing its way (one parked in the lane itself stands
            // bumper to bumper in a row it cannot steer out of)
            let mut dh = (p.heading - h as f64).rem_euclid(360.0);
            if dh > 180.0 {
                dh -= 360.0;
            }
            if !(0.8..4.5).contains(&lat) || dh.abs() > 30.0 {
                why.push(format!("{lat:+.1} m beside lane {l}, {dh:+.0} deg to it"));
                continue;
            }
            // nobody close by on the road
            if self.cars.iter().any(|c| (c.vehicle.position - p.pos).length() < 30.0) || !self.spawn_clear(&ty, q, h as f64) {
                why.push("road not clear".into());
                continue;
            }
            // [round 8 H8-8, fix pass 2] it signals as the parked object (SUMO's parked vehicle
            // indicates while parked, MSVehicleTransfer), and the car is made only when it moves
            // (`advance_pull_out`)
            let (front, rear, half_w) = extents(&ty, 12.0);
            let seed = self.rand();
            let ready_at = self.time + 2.0 + (seed % 1000) as f32 / 400.0;
            self.pending_pull_out = Some(PendingPullOut { key, ty, lane: l, s, lat, front, rear, half_w, ready_at, seed });
            if debug {
                log::info!("parked car {key} at ({:.1}, {:.1}) signals to pull out, {lat:.1} m right of lane {l}", p.pos.x, p.pos.y);
            }
            return;
        }
        if debug && !why.is_empty() {
            log::info!("parked pull-out: none this time ({})", why.join("; "));
        }
    }

    fn populate_kind(
        &mut self,
        world: &World,
        renderer: &Renderer,
        scene: &mut Scene,
        center: DVec3,
        kind: LaneKind,
        target: usize,
    ) {
        let radius = if kind == LaneKind::Air {
            self.spawn_radius * 6.0
        } else {
            self.spawn_radius
        };
        let nearby = self.net.lanes_starting_near(center, radius);
        // candidate lanes of this kind near the centre
        // (every lane of the kind, dead-end stubs included: a map maker leaves an open-ended
        // road on purpose and sets its [rule] trafficdensity, and OMSI spawns on it like any
        // other; a car that runs off the end of the network is taken away there by the
        // traffic, `AiState::drive`)
        let candidates: Vec<(usize, f32)> = nearby.iter().copied()
            .map(|i| (i, &self.net.lanes[i]))
            .filter(|(_, l)| {
                l.kind == kind
                    && l.length() > 8.0
                    && (l.start() - center).truncate().length() < radius
            })
            // lanes the map keeps clear of cars, and those whose [rule] trafficdensity is
            // zero, are not spawned on at all; a lower density makes a lane that much less
            // likely to be picked
            .filter(|(_, l)| !l.no_cars && l.density > 0.0)
            // nor on a lane with a stop line (a car put down past the line went through
            // the junction on red), nor on a piece of a junction that meets other ways
            .filter(|(i, _)| self.may_put_on(*i))
            // [round 9 A4] nor on a road that ends at the loaded tiles' edge before any choice of
            // way (`Network::ends_at_loaded_edge`): a car put there has nowhere to go but the
            // edge's queue (the flag follows the tiles, so the road opens again as they load)
            .filter(|(i, _)| kind != LaneKind::Street || !self.edge_bound.contains(i))
            // as many cars on a lane as metres of it (times its density): counted per
            // lane, the many short lanes of a junction drew the cars into the town's
            // tangles and left the long roads between them empty
            // [ROLLBACK crowd-96] and the nearer the player, the likelier: a low target
            // went out over the whole 400 m round and left the streets about the player
            // empty
            .map(|(i, l)| {
                let near = if kind == LaneKind::Street {
                    let d = ((l.start() - center).truncate().length() / 150.0) as f32;
                    1.0 / (1.0 + d * d)
                } else {
                    1.0
                };
                (i, l.length() * l.density.clamp(0.05, 4.0) * near)
            })
            .collect();
        if candidates.is_empty() {
            return;
        }
        let mut acc = 0.0f32;
        let cumulative: Vec<f32> = candidates.iter().map(|c| { acc += c.1; acc }).collect();
        let total_w = acc.max(1e-3);
        let mut attempts = 0;
        // [round 9 A1] the street target counts the cars within `spawn_radius` of the centre
        // only, for every host: counted over the whole map, queues standing at far tile edges
        // held the budget and nothing new appeared round the player (r8-jzk: the target met at
        // ~440 with 91 cars 700-900 m off). A LAN host's other players are counted the same
        // way round each of them (`populate_lan_centers`: counted over the whole map, the
        // traffic round the others met the host's target and the host drove through empty
        // streets, #342). Aircraft are few and far: counted over the whole map.
        let counted_near = (kind == LaneKind::Street).then_some((center, self.spawn_radius));
        let unscheduled = budget_count(
            self.cars.iter().map(|c| (c.vehicle.position, c.serves_stops(), self.net.lanes.get(c.ai.state.lane).is_some_and(|l| l.kind == kind))),
            counted_near,
        );
        let mut count = unscheduled;
        // ([round 7 B2] no global jam guard: SUMO has none, a car is put in only where its
        // lane has room - the per-spot tests below, `behind_a_queue` above all; the guard
        // counted every queue at a red, and with no car giving up there any more it held
        // every spawn at Jiangzhaikou from 60 s on)
        while count < target && attempts < target * 12 {
            attempts += 1;
            let x = self.rand_f() as f32 * total_w;
            let lane = candidates[cumulative.partition_point(|&c| c < x).min(candidates.len() - 1)].0;
            let mut s = (self.rand_f() * (self.net.lanes[lane].length() as f64 - 6.0)) as f32 + 3.0;
            let (mut p, _) = self.net.lanes[lane].at(s);
            // nobody may see it appear (the first population is the world as it loads) -
            // [ROLLBACK source-1] except near the start of a lane that nothing leads into: a
            // road the map begins out of nothing is where its cars come from, in OMSI in
            // plain view, and a map whose boulevard is a short stub in sight of the player
            // (BeijingCity) had no traffic on it at all until something drove in from the
            // side streets
            // [ROLLBACK appearfar-1] and beyond APPEAR_IN_VIEW_BEYOND in plain view: within
            // 350 m a car may appear only behind a building, and at a wide junction with the
            // view down the road nine attempts in ten found no such place (BeijingCity,
            // Ditan: 5 500 of 6 000 a pass; 70-100 cars about the player against a target of
            // 490; 200-260 with this, as measured headless with the player's view settings).
            // OMSI puts its cars down anywhere, at any distance.
            if kind == LaneKind::Street && !self.initial && !self.may_appear(world, p) {
                let source = self.net.prev.get(lane).is_some_and(|q| q.is_empty());
                let from_eye = self.viewer.map(|v| (p - v.pos).length()).unwrap_or(rel_len(p, center));
                if !source && from_eye <= APPEAR_IN_VIEW_BEYOND {
                    continue;
                }
                if source {
                    s = s.min(30.0);
                    p = self.net.lanes[lane].at(s).0;
                }
            }
            let rel = p - center;
            if rel.length() < 40.0 {
                continue; // not right next to the player
            }
            if self
                .cars
                .iter()
                .any(|c| (c.vehicle.position - p).length() < 14.0)
            {
                continue;
            }
            if kind == LaneKind::Street && self.behind_a_queue(lane, s, self.net.lanes[lane].speed_limit_kmh / 3.6 * SPAWN_SPEED_SHARE) {
                continue;
            }
            // only on loaded ground (a lane's tile may have gone again)
            if kind != LaneKind::Air && !world.has_ground(p.x, p.y) {
                continue;
            }
            let heading = self.net.lanes[lane].at(s).1 as f64;
            if self.in_front_of_someone(p) {
                continue;
            }
            if self
                .parked
                .get(&lane)
                .map(|l| {
                    l.iter()
                        .any(|p| (p.s - s).abs() < 8.0 && p.lateral.abs() < 1.5)
                })
                .unwrap_or(false)
            {
                continue; // not into a car parked in the lane
            }
            // (a lane may carry none of the groups that drive now: try another)
            let Some(ty) = self.pick_type(kind, Some(lane)) else {
                continue;
            };
            // (nor onto the rear section of an articulated bus, nor the player's bus)
            if kind != LaneKind::Air && !self.spawn_clear(&ty, p, heading) {
                continue;
            }
            let seed = self.rand();
            self.create_car(world, renderer, scene, center, kind, lane, s, ty, seed, None, None, None, None);
            count += 1;
        }
    }

    /// The cars out of range drive on: along their lanes at about the lanes' speed, taking
    /// a random way at every fork; one that reaches the end of the network has left the map.
    fn advance_dormant(&mut self) {
        let dt = (self.time - self.dormant_time).clamp(0.0, 10.0);
        self.dormant_time = self.time;
        if dt <= 0.0 || self.dormant.is_empty() {
            return;
        }
        let mut i = 0;
        while i < self.dormant.len() {
            let mut gone = false;
            {
                let d = &mut self.dormant[i];
                let lane = &self.net.lanes[d.lane];
                // (waits at lights and junctions taken as a quarter off the speed limit)
                d.speed = (lane.speed_limit_kmh.min(60.0) / 3.6 * 0.75).max(2.0);
                d.s += d.speed * dt;
                let mut guard = 0;
                while d.s > self.net.lanes[d.lane].length() && guard < 32 {
                    guard += 1;
                    let l = &self.net.lanes[d.lane];
                    // the ways a car on the road would take (`AiState::choose_after`): those
                    // of its group, else those open to cars, else any - only where the
                    // network ends does it leave the map (filtering by its group alone, the
                    // trucks of Spandau were gone at the first junction whose turn has no
                    // `trucks` rule, and hardly one of them ever came into range)
                    let pool = self.types.iter().find(|t| Arc::ptr_eq(&t.0, &d.ty)).and_then(|t| self.group_uvg[t.3]);
                    let same_kind = |n: &usize| self.net.lanes[*n].kind == d.kind;
                    let open = |pooled: bool| -> Vec<usize> {
                        l.next
                            .iter()
                            .copied()
                            .filter(same_kind)
                            .filter(|&n| {
                                let nl = &self.net.lanes[n];
                                let density = match pool.filter(|_| pooled) {
                                    Some(p) => nl.pool_density(&self.uvg_defaults, p),
                                    None => nl.density,
                                };
                                nl.allows(d.ty.def.ai_veh_type) && density > 0.0
                            })
                            .collect()
                    };
                    let mut options = if pool.is_some() { open(true) } else { Vec::new() };
                    if options.is_empty() {
                        options = open(false);
                    }
                    if options.is_empty() {
                        options = l.next.iter().copied().filter(same_kind).collect();
                    }
                    if options.is_empty() {
                        gone = true;
                        break;
                    }
                    d.s -= l.length();
                    d.walk = d.walk.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    d.lane = options[(d.walk >> 33) as usize % options.len()];
                }
            }
            if gone {
                self.dormant.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    /// May a random car be put on (or woken on) lane `l`: not a lane with a stop line (a car
    /// put down past the line went through the junction on red), nor a piece of a junction
    /// that meets other ways.
    fn may_put_on(&self, l: usize) -> bool {
        may_put_on(&self.net, &self.sim.junctions, l)
    }

    /// [ROLLBACK jamguard-1] [round 7 B4] A car put down at `s` on lane `l` going `v` m/s would
    /// come up behind a standing one: on the lane and the lanes it runs into (every branch), the
    /// nearest car ahead of the spot within `spawn_queue_reach(v)` stands. A car put down
    /// behind a queue only lengthens it; SUMO inserts a vehicle only where the safe gap to its
    /// leader holds (and the one to its follower, `in_front_of_someone`), it has no jam guard.
    fn behind_a_queue(&self, l: usize, s: f32, v: f32) -> bool {
        queue_within_reach(&self.net, l, s, spawn_queue_reach(v), &|lane| {
            self.cars.iter().filter(|c| c.ai.state.lane == lane).map(|c| (c.ai.state.s, c.ai.state.s - c.ai.state.rear, c.standing())).collect()
        })
    }

    /// Just in front of a car driving up to `p` (it would have to stop hard): within 20 m and
    /// one and a half braking distances ahead of it, 3 m to either side.
    fn in_front_of_someone(&self, p: DVec3) -> bool {
        self.cars.iter().any(|c| {
            let rel = p - c.vehicle.position;
            let h = c.vehicle.heading.to_radians();
            let (along, across) = (rel.x * h.sin() + rel.y * h.cos(), (rel.x * h.cos() - rel.y * h.sin()).abs());
            along > 0.0 && along < 20.0 + (c.ai.state.speed * c.ai.state.speed / (2.0 * c.ai.state.decel.max(1.0))) as f64 * 1.5 && across < 3.0
        })
    }

    /// The cars out of range that have come near again take their bodies back - where
    /// nobody sees it happen, up to a little over the number asked for around the player.
    fn wake_dormant(&mut self, world: &World, renderer: &Renderer, scene: &mut Scene, center: DVec3, target: usize) {
        if self.dormant.is_empty() {
            return;
        }
        let centers: Vec<DVec3> = std::iter::once(center).chain(self.lan_centers.iter().copied()).collect();
        // [round 9 A1] the cars round the players count, as for new cars (`populate_kind`):
        // counted over the whole map, far queues held this budget too
        let active = self.cars.iter().filter(|c| !c.serves_stops() && centers.iter().any(|o| (c.vehicle.position - *o).length() < self.spawn_radius)).count();
        let mut budget = (target as f32 * 1.25).ceil() as usize;
        budget = budget.saturating_sub(active);
        let mut i = 0;
        while i < self.dormant.len() && budget > 0 {
            let (p, h, ty, lane, s, speed) = {
                let d = &self.dormant[i];
                let l = &self.net.lanes[d.lane];
                let s = d.s.clamp(0.0, (l.length() - 0.1).max(0.0));
                let (p, h) = l.at(s);
                (p, h as f64, d.ty.clone(), d.lane, s, d.speed)
            };
            let near = centers.iter().any(|c| (p - *c).truncate().length() < self.spawn_radius);
            // (as a new car: never close by, where the mirrors and a turn of the head see
            // it - woken out of the picture 30-60 m from the bus, a car came into being in
            // the mirror or just round the corner)
            // ([round 7 B4] and where a new car may be put down: not past a stop line nor on a
            // junction piece with foes, not behind a queue, not just in front of somebody - a
            // dormant car drives everywhere, and one woken there appeared inside a junction)
            let ok = near
                && world.has_ground(p.x, p.y)
                && self.may_appear_car(world, p, &ty)
                && self.may_put_on(lane)
                // ([round 9 A4] nor where its road ends at the loaded tiles' edge: it sleeps on)
                && !self.edge_bound.contains(&lane)
                && !self.behind_a_queue(lane, s, speed)
                && !self.in_front_of_someone(p)
                && !self.cars.iter().any(|c| (c.vehicle.position - p).length() < 14.0)
                && self.spawn_clear(&ty, p, h);
            if ok {
                let d = self.dormant.swap_remove(i);
                self.create_car(world, renderer, scene, center, d.kind, d.lane, d.s, d.ty, d.seed, Some(d.scheme), Some(d.id), Some(d.speed), None);
                budget -= 1;
            } else {
                i += 1;
            }
        }
    }

    /// Fill the map: the streets the map has shown so far carry as many cars per metre as
    /// the ones around the player, the ones out of range as dormant cars (see `DormantCar`).
    fn fill_map(&mut self, center: DVec3, street_target: usize) {
        if street_target == 0 {
            return;
        }
        let far = self.spawn_radius * DESPAWN_FACTOR;
        let centers: Vec<DVec3> = std::iter::once(center).chain(self.lan_centers.iter().copied()).collect();
        let mut nearby: Vec<usize> = centers.iter()
            .flat_map(|&c| self.net.lanes_starting_near(c, self.spawn_radius))
            .collect();
        nearby.sort_unstable();
        nearby.dedup();
        let mut near = 0f64;
        for i in nearby {
            let l = &self.net.lanes[i];
            let Some(w) = street_lane_weight(l) else { continue };
            let d = centers.iter().map(|c| (l.start() - *c).truncate().length()).fold(f64::MAX, f64::min);
            if d < self.spawn_radius {
                near += w;
            }
        }
        if near < 50.0 {
            return;
        }
        let map_target = ((street_target as f64 * self.street_weight / near).min(street_target as f64 * MAP_POPULATION_FACTOR as f64)) as usize;
        let present = self.cars.iter().filter(|c| !c.serves_stops()).count() + self.dormant.len();
        if present >= map_target {
            return;
        }
        // The full outside list is only needed while replenishing the map population.
        let outside: Vec<(usize, f32)> = self.net.lanes.iter().enumerate()
            .filter_map(|(i, l)| {
                let w = street_lane_weight(l)?;
                let d = centers.iter().map(|c| (l.start() - *c).truncate().length()).fold(f64::MAX, f64::min);
                (d > far).then_some((i, w as f32))
            })
            .collect();
        if outside.is_empty() {
            return;
        }
        let mut acc = 0.0f32;
        let cumulative: Vec<f32> = outside.iter().map(|c| { acc += c.1; acc }).collect();
        for _ in 0..(map_target - present).min(64) {
            let x = self.rand_f() as f32 * acc;
            let lane = outside[cumulative.partition_point(|&c| c < x).min(outside.len() - 1)].0;
            let s = (self.rand_f() * (self.net.lanes[lane].length() as f64 - 4.0)) as f32 + 2.0;
            let Some(ty) = self.pick_type(LaneKind::Street, Some(lane)) else {
                continue;
            };
            let seed = self.rand();
            let scheme = if ty.paint_schemes.is_empty() { None } else { Some((seed >> 8) as usize % ty.paint_schemes.len().min(AI_SCHEMES)) };
            let id = self.next_id;
            self.next_id += 1;
            self.dormant.push(DormantCar { id, ty, kind: LaneKind::Street, lane, s, speed: 8.0, seed, scheme, walk: seed | 1 });
        }
    }

    /// Put a random car of type `ty` on `lane` at `s` metres into it and return its id:
    /// `scheme` Some = that paint scheme (a car coming back from out of range keeps its
    /// looks), `id` Some = that id, `speed` Some = at about that speed.
    #[allow(clippy::too_many_arguments)]
    fn create_car(
        &mut self,
        world: &World,
        renderer: &Renderer,
        scene: &mut Scene,
        center: DVec3,
        kind: LaneKind,
        lane: usize,
        s: f32,
        ty: Arc<VehicleType>,
        seed: u64,
        scheme: Option<Option<usize>>,
        id: Option<u64>,
        speed: Option<f32>,
        bus: Option<BusSetup>,
    ) -> u64 {
        let mut host = omsi_sim::VehicleHost::new(omsi_sim::SimClock::default());
        host.font_lib = Some(world.fonts.clone());
        if let Some(b) = &bus {
            host.hof = b.hof.clone();
        }
        // random paint scheme / advert (its variables there for the scripts' {init})
        let scheme = match scheme {
            Some(s) => s,
            None if ty.paint_schemes.is_empty() => None,
            None => Some((seed >> 8) as usize % ty.paint_schemes.len().min(AI_SCHEMES)),
        };
        host.paint_scheme = Some(scheme);
        let mut vehicle = VehicleInstance::new(ty.clone(), host);
        if let Some((num, reg)) = bus.as_ref().and_then(|b| b.number.clone()) {
            if let Some(i) = ty.program.str_var("number") {
                vehicle.state.str_vars[i as usize] = num;
            }
            if let Some(i) = ty.program.str_var("ident") {
                vehicle.state.str_vars[i as usize] = reg;
            }
        } else {
            // A vehicle of the random traffic with a `[number]` list takes a number of it at
            // random and the plate beside it, else the plate its mode makes of the number
            // (TRoadVehicleInst.virtual_11 at 0x7e7b51); a free plate is one of the map's
            // registrations.txt.
            let numbers = ty.def.numbers_with_plates();
            if !numbers.is_empty() {
                let (n, plate) = &numbers[(seed.rotate_left(29) % numbers.len() as u64) as usize];
                if let Some(i) = ty.program.str_var("number") {
                    vehicle.state.str_vars[i as usize] = n.clone();
                }
                if ty.def.registration_mode != 1 {
                    if let Some(i) = ty.program.str_var("ident") {
                        vehicle.state.str_vars[i as usize] = if plate.is_empty() { ty.def.plate_of_number(n) } else { plate.clone() };
                    }
                }
            }
            if ty.def.registration_mode == 1 {
                if let (Some(i), Some(reg)) = (ty.program.str_var("ident"), world.free_registration(seed.rotate_left(17))) {
                    vehicle.state.str_vars[i as usize] = reg;
                }
            }
        }
        // aircraft keep the height of their flight path: a ground sampler would pull
        // them down onto the streets
        vehicle.ground = if kind == LaneKind::Air {
            None
        } else {
            Some(ai_ground(world))
        };
        // and what its wheels stand on, asked as the player's are (see `AiBody::settle`); a
        // coupled part (an articulated bus's rear, a lorry's trailer) asks it too, with the
        // height it is at - the plain sampler gave it the deck of a bridge over its road
        vehicle.contact = (kind == LaneKind::Street).then(|| {
            std::sync::Arc::new(crate::scene::DriveGround {
                terrains: world.terrains.clone(),
                surfaces: world.surfaces.clone(),
                paint: world.ground_paint.clone(),
                wetness: world.road_wetness.clone(),
            }) as std::sync::Arc<dyn omsi_sim::rigid::Ground>
        });
        vehicle.apply_paint_vars(scheme);
        let render = world.add_vehicle_shared(renderer, scene, &ty, scheme, None);
        let trailer_renders =
            self.attach_trailers(world, renderer, scene, &mut vehicle, scheme, &render);
        if !ty.model.text_textures.is_empty() || bus.is_some() {
            vehicle.init_text_textures(&mut world.fonts.lock(), &|p| {
                omsi_texture::decode_file(p)
                    .ok()
                    .map(|i| (i.width, i.height, i.rgba))
            });
        }
        // a rear section's plates and numbers are `[texttexture]`s of its own reading the
        // leading vehicle's strings (`TrailerPart::update_text_textures`), so they need the
        // same fonts the front's do
        for t in vehicle.trailers.iter_mut() {
            t.init_text_textures(&mut world.fonts.lock(), &|p| {
                omsi_texture::decode_file(p)
                    .ok()
                    .map(|i| (i.width, i.height, i.rgba))
            });
        }
        let mut state = AiState::new(lane, s, seed);
        state.veh_type = if bus.is_some() { -1 } else { ty.def.ai_veh_type };
        if bus.is_none() {
            state.traffic_pool = self.types.iter().find(|t| Arc::ptr_eq(&t.0, &ty))
                .and_then(|t| self.group_uvg[t.3])
                .map(|pool| (pool, self.uvg_defaults.clone()));
        }
        state.plan_next(&self.net);
        // heavy vehicles (trucks, vans) cruise slower, which is what gets them overtaken
        let heavy = ty.def.mass > 6.0 || bus.is_some();
        state.max_speed_kmh = if kind == LaneKind::Air {
            AIRCRAFT_KMH
        } else if bus.is_some() {
            // a bus driver keeps to the limit (the town's 50) like the cars round him,
            // with a little more on the arterial roads
            56.0 + (seed % 7) as f32
        } else if heavy {
            // (a truck keeps to the limit like the cars, up to a truck's own 80-90 km/h: it
            // took 38-47 on every road, crawling along 80 km/h roads, #327)
            80.0 + (seed % 10) as f32
        } else if kind == LaneKind::Street && ty.def.mass > 0.0 && ty.def.mass <= 0.3 {
            // a bicycle (stock ones weigh exactly 0.3 t): 15-21 km/h, the `vmax` range their
            // script cuts the drive at (#327)
            15.0 + (seed % 7) as f32
        } else {
            100.0
        };
        let (front, rear, half_width) = extents(&ty, if bus.is_some() { 12.0 } else { 4.5 });
        // [round 7 fix pass 1, 12] the whole consist (SUMO's vehicle length): the last coupled
        // section's rear, not the front section's - followers stopped inside a BJ6180
        // articulated bus's rear section
        let rear = self.consist_rear(&ty).max(rear);
        state.front = front;
        state.rear = rear;
        state.length = front + rear;
        if let Some(b) = &bus {
            state.set_route(&self.net, b.route.clone(), s);
        }
        state.speed = (self.net.lanes[lane]
            .speed_limit_kmh
            .min(state.max_speed_kmh)
            / 3.6
            * SPAWN_SPEED_SHARE)
            .min(state.curve_speed(&self.net));
        if kind == LaneKind::Air {
            state.speed = self.net.lanes[lane]
                .speed_limit_kmh
                .min(state.max_speed_kmh)
                / 3.6;
        }
        // a bus put out at a stop stands there (in the bay, if it has one)
        let at_stop = bus
            .as_ref()
            .and_then(|b| b.stops.first())
            .filter(|st| st.ri == 0 && (st.s - s).abs() < 1.5);
        if let Some(st) = at_stop {
            state.speed = 0.0;
            if st.bay.abs() > 0.01 {
                state.lateral = st.bay;
                state.lateral_target = st.bay;
                state.lateral_ramp = (st.bay, st.bay, 0.0, 1.0);
            }
        }
        let body = place_body(&self.net, &state, &mut vehicle, motion_kind(kind));
        if omsi_cfg::env::var_os("OMSI_DEBUG_TRAFFIC").is_some() {
            let pos = vehicle.position;
            log::info!(
                "spawn {} on lane {lane} s={s:.1} at ({:.1}, {:.1}, {:.1}) heading {:.0}",
                ty.def.path.display(),
                pos.x,
                pos.y,
                pos.z,
                vehicle.heading
            );
        }
        let id = id.unwrap_or_else(|| {
            let id = self.next_id;
            self.next_id += 1;
            id
        });
        if let Some(v) = speed {
            state.speed = v.min(state.speed.max(v * 0.5));
        }
        // [round 9 A2] counted: a car put down in frame beyond the perceptual horizon
        if kind == LaneKind::Street && !self.initial {
            let pos = vehicle.position;
            self.appeared_far += self.viewer.is_some_and(|v| v.frames(pos, 2.5) && v.beyond_horizon((pos - v.pos).length(), body_height(&vehicle.ty), body_radius(&vehicle.ty))) as usize;
        }
        if self.debug_population && kind == LaneKind::Street {
            let v = self.viewer;
            let pos = vehicle.position;
            if !self.initial && v.map(|v| v.frames(pos, 2.5)).unwrap_or(false) {
                self.framed_spawns.push((id, pos));
            }
            log::info!("population t={:.1}: car {id} appears at ({:.0}, {:.0}), {:.0} m from the centre, {:.0} m from the camera, in frame {}, behind a building {}{}", self.time, pos.x, pos.y, (pos - center).length(), v.map(|v| (pos - v.pos).length()).unwrap_or(0.0), v.map(|v| v.frames(pos, 2.5)).unwrap_or(false), v.map(|v| self.occluded(world, &v, pos, 2.5)).unwrap_or(false), if self.initial { " (initial)" } else { "" });
        }
        // the driver: a parameter set of its class with a seeded personality, then the
        // few things the vehicle's own data says about it (its limit, how it takes bends)
        let class = if bus.is_some() { VehicleClass::ScheduledBus } else { VehicleClass::from_code(ty.def.ai_veh_type, kind) };
        // (its braking, pulling away and bends are its class's `DriverParams`, one value
        // each that the IDM and the plan both read)
        let (max_kmh, speed) = (state.max_speed_kmh, state.speed);
        let mut ai = AiVehicle::new(id, class, state, half_width, seed);
        // (a van of the car class over 6 t bends like a lorry: the preset by mass)
        ai.params = ai.params.by_mass(ty.def.mass);
        ai.params.apply(&mut ai.state);
        ai.state.max_speed_kmh = max_kmh;
        ai.state.speed = speed;
        self.cars.push(AiCar {
            id,
            ai,
            vehicle,
            render,
            trailer_renders,
            body,
            bus: bus.map(|b| Box::new(BusService::new(b.stops))),
            sounds: None,
            rail_trail: Default::default(),
            ai_secs: 0.0,
            consist_reversed: false,
            park: None,
            released: false,
            seed,
            scheme,
        });
        if kind != LaneKind::Air {
            // put down short of a red light: no faster than it can stop for it
            let i = self.cars.len() - 1;
            let way = omsi_traffic::perception::way_lanes(&self.net, &self.cars[i].ai.state, 200.0);
            let st = &self.cars[i].ai.state;
            let red = way.iter().skip(1).take_while(|w| w.1 <= 150.0).find_map(|&(l, d)| {
                let (c, li) = self.net.lanes[l].traffic_light?;
                let ctl = self.sim.lights.get(c)?;
                (!matches!(TrafficLightController::aspect(ctl.state(li)), Aspect::Green | Aspect::Dark)).then_some(d - st.front)
            });
            if let Some(gap) = red {
                let st = &mut self.cars[i].ai.state;
                st.speed = st.speed.min((2.0 * st.decel * (gap - 1.0).max(0.0)).sqrt());
            }
        }
        id
    }

    pub fn random_sets(&self) -> Vec<(Arc<VehicleType>, Option<usize>)> {
        // (the first paint of every type, then the second of every type that has one, and so on:
        // where the textures' budget does not hold them all, every type is there before any
        // type has all its paints)
        let mut sets: Vec<(Arc<VehicleType>, Option<usize>)> = Vec::new();
        let most = self.types.iter().map(|(ty, ..)| ty.paint_schemes.len().min(AI_SCHEMES)).max().unwrap_or(0).max(1);
        for k in 0..most {
            for (ty, ..) in &self.types {
                let n = ty.paint_schemes.len().min(AI_SCHEMES);
                let scheme = if n == 0 {
                    if k > 0 {
                        continue;
                    }
                    None
                } else if k < n {
                    Some(k)
                } else {
                    continue;
                };
                if !sets.iter().any(|(t, s)| t.def.path == ty.def.path && *s == scheme) {
                    sets.push((ty.clone(), scheme));
                }
            }
        }
        sets
    }

    pub fn precache_random(&mut self, world: &World, renderer: &Renderer, scene: &mut Scene) {
        let t0 = std::time::Instant::now();
        let sets = self.random_sets();
        let mut held = 0;
        for chunk in sets.chunks(3) {
            // (the vehicles' textures are never shrunk: past a third of the textures' budget the
            // rest of the paints are uploaded when a car first wears them, not ahead of time)
            if world.vehicle_pressure(scene, VEHICLE_SHARE) {
                log::info!("traffic: {held} of {} vehicle/paint sets uploaded ahead, the vehicles' textures take {:.0} MB of the budget; the rest when first used", sets.len(), world.vehicle_texture_bytes(scene) as f64 / 1e6);
                break;
            }
            world.prefetch_vehicle_sets(renderer, chunk);
            for (ty, scheme) in chunk {
                world.precache_vehicle(renderer, scene, ty, *scheme);
                held += 1;
            }
        }
        world.forget_prefetched();
        log::info!("traffic: {} vehicle/paint sets of the random traffic read and uploaded in {:.1} s", sets.len(), t0.elapsed().as_secs_f32());
    }

    /// Put a timetable bus on the road: an AI car like any other (`create_car`), on its
    /// trip's route at `s` metres into the first lane, with its service (stops).
    /// Returns the car index. `scheme`: the paint scheme to use (Some), or a random one.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_bus(
        &mut self,
        world: &World,
        renderer: &Renderer,
        scene: &mut Scene,
        ty: Arc<VehicleType>,
        route: Vec<usize>,
        s: f32,
        stops: Vec<(usize, f32, f32, f64, i64, f32)>,
        number: Option<(String, String)>,
        hof: Option<Arc<omsi_vehicle::Hof>>,
        scheme: Option<Option<usize>>,
    ) -> Option<usize> {
        let &lane = route.first()?;
        let kind = self.net.lanes.get(lane)?.kind;
        // the options' [AIMaxCountScheduled]: no more timetable vehicles than that at once
        if self.max_scheduled > 0 && self.cars.iter().filter(|c| c.serves_stops() || !c.ai.state.route.is_empty()).count() >= self.max_scheduled as usize {
            return None;
        }
        let seed = self.rand();
        let setup = BusSetup {
            route,
            stops: stops.into_iter().map(crate::bus_service::Stop::from_tuple).collect(),
            number,
            hof,
        };
        let center = self.viewer.map(|v| v.pos).unwrap_or_default();
        let id = self.create_car(world, renderer, scene, center, kind, lane, s, ty.clone(), seed, scheme, None, None, Some(setup));
        let ci = self.cars.iter().rposition(|c| c.id == id)?;
        if kind == LaneKind::Air {
            let p = self.cars[ci].vehicle.position;
            let ground = world
                .ground_height(p.x, p.y)
                .map(|g| format!("{:.0} m above the ground", p.z - g))
                .unwrap_or_else(|| "over unloaded ground".into());
            log::info!("aircraft {} on its flight path at ({:.0}, {:.0}), height {:.0} m, {ground}, {:.0} km/h", ty.def.path.file_name().unwrap_or_default().to_string_lossy(), p.x, p.y, p.z, self.cars[ci].ai.state.speed * 3.6);
        }
        Some(ci)
    }

    /// [ROLLBACK aispawn-137] An AI car of the random traffic on the street lane nearest to
    /// `p` (the tools window: a click on a street), driving on by itself from there.
    pub fn spawn_random_at(
        &mut self,
        world: &World,
        renderer: &Renderer,
        scene: &mut Scene,
        p: DVec3,
    ) -> Result<String, String> {
        let Some((lane, s, d)) = self.net.nearest_lane(p, LaneKind::Street) else {
            return Err("there is no street here".into());
        };
        if d > 8.0 {
            return Err(format!(
                "no street within reach ({d:.0} m to the nearest lane)"
            ));
        }
        let len = self.net.lanes[lane].length();
        let s = s.clamp(1.0, (len - 1.0).max(1.0));
        let (pos, h) = self.net.lanes[lane].at(s);
        let Some(ty) = self.pick_type(LaneKind::Street, Some(lane)) else {
            return Err(format!("no vehicle drives on lane {lane}"));
        };
        if !self.spawn_clear(&ty, pos, h as f64) {
            return Err("another vehicle stands there".into());
        }
        let seed = self.rand();
        let center = self.viewer.map(|v| v.pos).unwrap_or(pos);
        let name = format!("{} {}", ty.def.manufacturer, ty.def.type_name);
        let id = self.create_car(
            world,
            renderer,
            scene,
            center,
            LaneKind::Street,
            lane,
            s,
            ty,
            seed,
            None,
            None,
            None,
            None,
        );
        Ok(format!(
            "AI car #{id} ({}) put on lane {lane} at {s:.0} m",
            name.trim()
        ))
    }

    /// Nearest vehicle ahead of position `s` on `lane` (following the lanes `plan` has
    /// chosen after it, else the first `next`, for up to `look` m): (distance from `s` to
    /// its rear, its speed along the lane, its index). A vehicle beside the lane's middle
    /// far enough to be passed (a bus in its bay) does not count; one coming the other way
    /// round an obstacle does, standing.
    fn footprints(&self) -> Vec<Footprint> {
        let mut out = Vec::with_capacity(self.cars.len() + 8);
        for (i, c) in self.cars.iter().enumerate() {
            let st = &c.ai.state;
            let h = c.vehicle.heading.to_radians();
            let (fwd, right) = (DVec2::new(h.sin(), h.cos()), DVec2::new(h.cos(), -h.sin()));
            let center = c.vehicle.position.truncate() + fwd * ((st.front - st.rear) * 0.5) as f64;
            out.push(Footprint {
                car: i,
                center,
                fwd,
                right,
                half_len: ((st.front + st.rear) * 0.5) as f64,
                half_w: c.ai.half_width as f64,
                speed: st.speed,
                z: c.vehicle.position.z,
            });
            for t in &c.vehicle.trailers {
                if let Some(bb) = t.ty.def.bounding_box {
                    out.push(Footprint::from_obb(
                        i,
                        &omsi_sim::collision::Obb::from_box(bb, t.position, t.body_heading()),
                        st.speed,
                    ));
                }
            }
        }
        out
    }

    /// May a vehicle of `ty` be put on the road at `pos` facing `heading` (deg)? Not onto
    /// (or right up against) another vehicle or the player's: a timetable bus used to be
    /// checked only for a vehicle origin within 9 m, and a layover bus appeared inside the
    /// articulated bus waiting at the same stand.
    pub fn spawn_clear(&mut self, ty: &Arc<VehicleType>, pos: DVec3, heading: f64) -> bool {
        self.spawn_clear_except(ty, pos, heading, None)
    }

    /// `spawn_clear`, leaving out car `except` (a vehicle asking whether it could be put
    /// back where it stands). The bodies of the vehicle and of the parts coupled to it are
    /// compared with `SPAWN_BODY_CLEAR` to spare (not centres: an articulated bus reaches
    /// 6 m ahead of its origin and 12 m behind it, and one put down 9.3 m from the player's
    /// bus stood 1.8 m inside it) with every other vehicle's and `keep_clear`.
    pub fn spawn_clear_except(&mut self, ty: &Arc<VehicleType>, pos: DVec3, heading: f64, except: Option<usize>) -> bool {
        let bodies = self.consist_bodies(ty, pos, heading);
        let reach = bodies.iter().map(|b| (b.center - pos.truncate()).length() + b.half.length()).fold(0.0, f64::max) + 40.0;
        let touches = |o: &omsi_sim::collision::Obb| bodies.iter().any(|b| b.overlaps(o));
        let taken = self.keep_clear.iter().any(|o| touches(o))
            || self
                .cars
                .iter()
                .enumerate()
                .filter(|(i, c)| Some(*i) != except && (c.vehicle.position - pos).length() < reach)
                .any(|(_, c)| vehicle_bodies(&c.vehicle).iter().any(|o| touches(o)));
        if taken {
            return false;
        }
        let (front, rear, half_w) = extents(ty, 12.0);
        // nor just in front of a car driving up to that place (it would have to stop hard)
        let in_front = self.cars.iter().enumerate().filter(|(i, _)| Some(*i) != except).any(|(_, c)| {
            let rel = pos - c.vehicle.position;
            let h = c.vehicle.heading.to_radians();
            let (along, across) = (
                rel.x * h.sin() + rel.y * h.cos(),
                (rel.x * h.cos() - rel.y * h.sin()).abs(),
            );
            let v = c.ai.state.speed;
            along > 0.0
                && along
                    < (c.ai.state.front + rear + 15.0 + v * v / (2.0 * c.ai.state.decel.max(1.0)) * 1.5)
                        as f64
                && across < 3.0
                && (rel.z).abs() < 4.0
        });
        if in_front {
            return false;
        }
        // nor close to the player's (a wider margin: in plain sight)
        match self.player {
            Some((c, ph, hl, hw, _)) => {
                let h = heading.to_radians();
                let (fwd, right) = (DVec2::new(h.sin(), h.cos()), DVec2::new(h.cos(), -h.sin()));
                let me = Footprint { car: usize::MAX, center: pos.truncate() + fwd * ((front - rear) * 0.5) as f64, fwd, right, half_len: ((front + rear) * 0.5) as f64, half_w: half_w as f64, speed: 0.0, z: pos.z };
                let h = ph.to_radians();
                let p = Footprint {
                    car: usize::MAX,
                    center: c.truncate(),
                    fwd: DVec2::new(h.sin(), h.cos()),
                    right: DVec2::new(h.cos(), -h.sin()),
                    half_len: hl as f64,
                    half_w: hw as f64,
                    speed: 0.0,
                    z: c.z,
                };
                (p.z - me.z).abs() > 4.0 || !p.overlaps(&me, 1.5)
            }
            None => true,
        }
    }

    /// The aspect of every railway signal of `routes` (by signal object): 1 (clear), 2 (clear
    /// with a speed limit) when a train, AI or the player's (`player_rail`: its lane and
    /// direction), wants the signal's route and none stands on it yet; 0 (stop) otherwise.
    pub fn signal_aspects(&self, routes: &[omsi_map::ailists::SignalRoute], player_rail: Option<(usize, bool)>) -> hashbrown::HashMap<i64, f32> {
        let mut out: hashbrown::HashMap<i64, f32> = hashbrown::HashMap::new();
        if routes.is_empty() {
            return out;
        }
        // per train: the map ids it stands on and those of its next lanes
        let mut trains: Vec<(i64, Vec<i64>)> = self
            .cars
            .iter()
            .filter(|c| !c.ai.state.route.is_empty() && self.net.lanes.get(c.ai.state.lane).map(|l| l.kind == omsi_sim::traffic::LaneKind::Rail).unwrap_or(false))
            .map(|c| {
                let here = self.net.lanes[c.ai.state.lane].key.map(|k| k.id).unwrap_or(-1);
                let ahead = c.ai.state.route.iter().skip(c.ai.state.route_index + 1).take(40).filter_map(|&l| self.net.lanes.get(l).and_then(|l| l.key).map(|k| k.id)).collect();
                (here, ahead)
            })
            .collect();
        // the player's own train (driven on the rails): the lanes ahead of it the way it goes,
        // every branch at a fork (which it takes is not known yet) - its signals stayed at
        // stop, only an AI train ever cleared them
        if let Some((lane, along)) = player_rail.filter(|(l, _)| *l < self.net.lanes.len()) {
            let here = self.net.lanes[lane].key.map(|k| k.id).unwrap_or(-1);
            let mut ahead: Vec<i64> = Vec::new();
            let mut frontier = vec![lane];
            for _ in 0..6 {
                let mut next = Vec::new();
                for l in frontier {
                    let nb: Vec<usize> = if along { self.net.lanes[l].next.clone() } else { self.net.prev.get(l).cloned().unwrap_or_default() };
                    for n in nb {
                        if let Some(k) = self.net.lanes.get(n).and_then(|x| x.key) {
                            if !ahead.contains(&k.id) {
                                ahead.push(k.id);
                            }
                        }
                        next.push(n);
                    }
                }
                frontier = next;
                if frontier.len() > 32 {
                    break;
                }
            }
            trains.push((here, ahead));
        }
        for r in routes {
            let pieces: hashbrown::HashSet<i64> = r.entries.iter().map(|e| e[0]).collect();
            let occupied = trains.iter().any(|(here, _)| pieces.contains(here));
            let wanted = trains.iter().any(|(here, ahead)| !pieces.contains(here) && ahead.iter().any(|id| pieces.contains(id)));
            let aspect = if wanted && !occupied { if r.speed_limit.is_some() { 2.0 } else { 1.0 } } else { 0.0 };
            let e = out.entry(r.signal.0).or_insert(0.0);
            *e = e.max(aspect);
        }
        out
    }

    /// The switches the trains need set: (map object, `[path]` index) of the next lanes of
    /// every train's timetable route. A train throws the points ahead of it to the
    /// `[switchdir]` of the path its route takes (see `World::set_switches`).
    pub fn switch_requests(&self) -> Vec<(i64, u16)> {
        let mut out = Vec::new();
        for c in &self.cars {
            let st = &c.ai.state;
            if st.route.is_empty() || self.net.lanes.get(st.lane).map(|l| l.kind != omsi_sim::traffic::LaneKind::Rail).unwrap_or(true) {
                continue;
            }
            for &l in st.route.iter().skip(st.route_index).take(5) {
                if let Some(k) = self.net.lanes.get(l).and_then(|l| l.key) {
                    out.push((k.id, k.path));
                }
            }
        }
        out
    }

    /// Keep `player_kappa`: the player's turn per metre driven, from how its heading changed
    /// since the last tick.
    /// Two buses queued for one stop: this one close behind another standing at it (it
    /// serves the stop where it stands), and another one coming up behind this one for it
    /// (this one pulls on a few metres to leave the bay).
    fn bus_queue_flags(&self, i: usize, player: Option<PlayerBox>) -> (bool, bool) {
        let me = &self.cars[i];
        let Some(my_stop) = me.bus.as_ref().and_then(|b| b.stops.front()).map(|s| s.id) else { return (false, false) };
        let stop_of = |c: &AiCar| c.bus.as_ref().and_then(|b| b.stops.front()).map(|s| s.id);
        let lead_close = match (me.ai.last.lead_seen, me.ai.last.lead_id, me.ai.last.lead) {
            (Some(Seen::Player(_)), _, Some(l)) => l.gap < 9.0 && player.is_some_and(|p| p.4.abs() < omsi_traffic::driver::STANDING_SPEED),
            (_, Some(id), Some(l)) => l.gap < 9.0 && self.links.by_id.get(&id).map(|&k| &self.cars[k]).is_some_and(|o| o.standing() && o.at_stop() && stop_of(o) == Some(my_stop)),
            _ => false,
        };
        let follower_close = self.links.following(me.id).iter().any(|&j| {
            let o = &self.cars[j];
            j != i && o.ai.last.lead.is_some_and(|l| l.gap < 40.0) && stop_of(o) == Some(my_stop)
        });
        (lead_close, follower_close)
    }

    /// [ROLLBACK r4d-B2] How many vehicles stand queued behind car `i` (each standing with the
    /// one ahead of it as its lead), as far back as 30 of them.
    fn queued_behind(&self, i: usize) -> usize {
        let mut chain = vec![self.cars[i].id];
        let mut k = 0;
        while k < chain.len() && chain.len() <= 30 {
            let id = chain[k];
            for &j in self.links.following(id) {
                let o = &self.cars[j];
                if o.standing() && !chain.contains(&o.id) {
                    chain.push(o.id);
                }
            }
            k += 1;
        }
        chain.len() - 1
    }

    /// [round 6 B1] The lanes beside its own of the junction object timetable bus `i` stands
    /// on that its body covers (`bus_service::fan_lanes_covered`); none while it drives.
    fn fan_cover(&self, i: usize) -> Vec<usize> {
        let c = &self.cars[i];
        let st = &c.ai.state;
        if c.bus.is_none() || c.is_rail() || !c.standing() {
            return Vec::new();
        }
        let through = |l: usize| self.through.through(&self.net, l, c.id);
        crate::bus_service::fan_lanes_covered(&self.net, st.lane, c.ai.pos.truncate(), c.ai.heading, st.front, st.rear, c.ai.half_width, &through)
    }

    /// [round 6 B1] The queue at the lanes `fan` timetable bus `i` covers
    /// (`bus_service::fan_queue`): the longest anybody it holds has stood there, and how
    /// many stand.
    fn fan_queue(&self, i: usize, fan: &[usize]) -> (f32, usize) {
        if fan.is_empty() {
            return (0.0, 0);
        }
        let me = self.cars[i].id;
        let reach = crate::bus_service::FAN_QUEUE_REACH;
        let others: Vec<crate::bus_service::FanOther> = self
            .cars
            .iter()
            .filter(|o| o.id != me && o.standing())
            .map(|o| {
                let st = &o.ai.state;
                let way: Vec<(usize, f32)> = omsi_traffic::perception::way_lanes(&self.net, st, reach + st.front).into_iter().map(|(l, d)| (l, d - st.front)).collect();
                let last = &o.ai.last;
                crate::bus_service::FanOther {
                    id: o.id,
                    lane: st.lane,
                    way,
                    speed: st.speed,
                    stood: o.ai.stopped,
                    lead: last.lead_id,
                    serving: o.at_stop() || o.trip_done(),
                    held_by_bus: last.wait_for == Some(me) || last.lead_id == Some(me) || last.body_block == Some(me),
                }
            })
            .collect();
        crate::bus_service::fan_queue(fan, &others)
    }

    /// [ROLLBACK r4d-B2] Timetable bus `i` stands out its layover (or queues for an occupied
    /// berth) across a through lane and the queue behind it asks it to go: a way off
    /// (`bus_service::plan_detour`). None when it need not go; Some(None) when nothing fits.
    fn bus_detour(&self, i: usize, queue_behind_secs: f32, fan: &[usize], fan_secs: f32, fan_queued: usize) -> Option<Option<crate::bus_service::Detour>> {
        let c = &self.cars[i];
        let b = c.bus.as_ref()?;
        let st = &c.ai.state;
        if c.is_rail() || st.change.is_some() || c.ai.passing.is_some() || st.route.get(st.route_index) != Some(&st.lane) {
            return None;
        }
        // queueing behind what holds its stop: a vehicle standing at a stop, or the exit full
        let queued = c.ai.last.why == Why::ExitFull
            || c.ai.last.lead_id.and_then(|id| self.links.by_id.get(&id).map(|&k| &self.cars[k])).is_some_and(|o| o.at_stop() || o.ai.last.why == Why::ExitFull);
        let stop = b.stops.front()?;
        let near = st.route_distance(&self.net, stop.ri, stop.s);
        let left = b.detour_wanted(st.speed, self.day_time, queued && near < 60.0)?;
        let through = self.through.through(&self.net, st.lane, c.id);
        // [round 6 B1] (or the lanes beside its own of the junction object it stands on, and
        // the queue it holds there)
        let covers_own = crate::bus_service::covers_through_lane(&self.net, st.lane, st.lateral, c.ai.half_width, through);
        if !crate::bus_service::layover_asked_to_go(covers_own, queue_behind_secs, self.queued_behind(i), fan, fan_secs, fan_queued) {
            return None;
        }
        // back to where it stands, or to its stop point when that lies on this lane ahead
        let back_to = if stop.ri == st.route_index && stop.s > st.s { stop.s } else { st.s };
        let len = st.front + st.rear;
        let room = |l: usize| self.bus_room(c.id, l, len, c.ai.half_width);
        Some(crate::bus_service::plan_detour(&self.net, st.lane, st.s, back_to, left, &room))
    }

    /// [ROLLBACK r4d-B2] The last resort for timetable bus `i`, which nothing lets out of the
    /// way: when nobody sees it go, and it could come back unseen, the time of day its trip
    /// comes back on the road (its departure from the stop it waits for, less
    /// `bus_service::RESPAWN_LEAD`, at least half a minute on). It goes only by the rule a
    /// bus appears by (`may_appear`, without the buildings: never within 150 m of the
    /// player, nor in the picture within 350 m), and only while a place within
    /// `schedule::RESPAWN_APPROACH` of its stop is free and out of view by that same rule
    /// at this moment (a bus vanished 183 m from the player in plain sight, and then no
    /// place to come back at was ever out of view: a departure lost for nothing).
    fn respawn_at(&mut self, i: usize) -> Option<crate::bus_service::Respawned> {
        let (p, r, id, stop, depart, ty, route) = {
            let c = &self.cars[i];
            let b = c.bus.as_ref()?;
            let stop = *b.stops.front()?;
            let depart = if b.at_stop() { b.leave_at.max(stop.depart) } else { stop.depart };
            let r = (c.ai.state.front + c.ai.state.rear) as f64 * 0.5 + 2.0;
            (c.vehicle.position, r, c.id, stop, depart, c.vehicle.ty.clone(), c.ai.state.route.clone())
        };
        if route.get(stop.ri).is_none() {
            return None;
        }
        // (not again for a while: the tour's bus put back for its departure was taken off
        // again 36 s later at its next layover)
        if self.respawn_block.get(&id).is_some_and(|&t| self.day_time < t) {
            return None;
        }
        if self.viewer.is_some_and(|v| !v.hides_plainly(p, r)) {
            self.respawn_refuse(i, "in the player's sight");
            return None;
        }
        let places = crate::schedule::respawn_places(&self.net, &route, (stop.ri, stop.s));
        let viewer = self.viewer;
        let mut why = None;
        let back = places.iter().any(|&(k, at_s)| {
            let (q, h) = self.net.lanes[route[k]].at(at_s);
            if viewer.is_some_and(|v| !v.hides_plainly(q, 8.0)) {
                why.get_or_insert("every place to come back at is in the player's sight");
                return false;
            }
            if !self.spawn_clear_except(&ty, q, h as f64, Some(i)) {
                why.get_or_insert("no place to come back at is free");
                return false;
            }
            true
        });
        if !back {
            self.respawn_refuse(i, why.unwrap_or("nowhere to come back at"));
            return None;
        }
        self.respawn_refused.remove(&id);
        let at = (depart - crate::bus_service::RESPAWN_LEAD).max(self.day_time + 30.0);
        log::info!("t={:.1}: timetable bus {} taken off the road: no loop or room within range, {} queued behind it; its trip comes back at its stop {} at {:.1} min (departure {:.1} min)", self.day_time, id, self.queued_behind(i), stop.id, at / 60.0, depart / 60.0);
        Some(crate::bus_service::Respawned { car: id, removed: self.day_time, at, depart, stop: stop.id })
    }

    /// [ROLLBACK r4d-B2] Bus `i` stays where it is although nothing lets it out of the way:
    /// `why` it was not taken off the road either, logged once per reason.
    fn respawn_refuse(&mut self, i: usize, why: &'static str) {
        let id = self.cars[i].id;
        if self.respawn_refused.get(&id) != Some(&why) {
            log::info!("t={:.1}: timetable bus {} stays: no loop or room within range, {} queued behind it, and not taken off the road: {why}", self.day_time, id, self.queued_behind(i));
            self.respawn_refused.insert(id, why);
        }
    }

    /// [ROLLBACK r4d-B2] Where on lane `l` timetable bus `me`, `len` metres long and
    /// `half_width` wide, can stand out of the through traffic (`bus_service::room_on_lane`):
    /// another stop's bay off the lane, the lane itself when no vehicle's way uses it, or the
    /// kerb of a parked row (a gap, before it, after it), clear of the lane's junction and
    /// crossings and of every vehicle there. Its origin's place and its offset, if anywhere.
    fn bus_room(&self, me: u64, l: usize, len: f32, half_width: f32) -> Option<(f32, f32)> {
        let lane = self.net.lanes.get(l)?;
        if lane.kind != LaneKind::Street || lane.no_cars {
            return None;
        }
        let bays: Vec<(f32, f32)> = self
            .cars
            .iter()
            .filter(|o| o.id != me)
            .filter_map(|o| Some((o.bus.as_ref()?, &o.ai.state.route)))
            .flat_map(|(b, route)| b.stops.iter().filter(move |x| route.get(x.ri) == Some(&l)).map(|x| (x.s, x.bay)))
            .collect();
        let parked: Vec<Parked> = self.parked.get(&l).cloned().unwrap_or_default();
        let keep_off: Vec<f32> = self.net.crossings.get(l).into_iter().flatten().map(|c| c.at).chain(self.net.walks.get(l).into_iter().flatten().map(|w| w.1)).collect();
        let free = |a: f32, b: f32, lat: f32| {
            !self.cars.iter().any(|o| {
                let st = &o.ai.state;
                st.lane == l && st.s + st.front > a && st.s - st.rear < b && (lat.abs() < 0.1 || (st.lateral - lat).abs() < 2.5)
            })
        };
        crate::bus_service::room_on_lane(lane.length(), lane.width, len, half_width, self.through.used(l), &bays, &parked, &keep_off, &free)
    }

    /// [ROLLBACK r4d-B2] The through lanes as of this frame: the lanes the random traffic is
    /// on (kept), and those on the timetable vehicles' routes now.
    fn note_through_lanes(&mut self) {
        self.through.routes.clear();
        for c in &self.cars {
            if c.bus.is_none() || c.released {
                self.through.driven.insert(c.ai.state.lane);
            } else {
                let st = &c.ai.state;
                for &l in std::iter::once(&st.lane).chain(st.route.iter().skip(st.route_index)) {
                    self.through.routes.entry(l).or_default().push(c.id);
                }
            }
        }
    }

    /// One frame: what the drivers see besides each other (the player and the LAN
    /// players, the people, the parked cars), the bus services and the parking plans,
    /// then the drivers (`omsi_traffic::Simulation::tick`), then the bodies and the
    /// scripts on the workers.
    pub fn tick(&mut self, dt: f32, player: Option<PlayerBox>) {
        self.lamp_dt += dt;
        if self.mirror {
            self.mirror_tick(dt);
            return;
        }
        let t_start = std::time::Instant::now();
        self.time += dt;
        self.last_dt = dt;
        self.player = player;
        let debug = omsi_cfg::env::var_os("OMSI_DEBUG_TRAFFIC").is_some();
        let others = std::mem::take(&mut self.others);
        // the player and the LAN players as the drivers see them
        let out_side = self.sim.out_side() as u8;
        self.player_still = match player {
            Some(p) if p.4.abs() < 0.3 => self.player_still + dt,
            _ => 0.0,
        };
        self.player_signalling = if self.player_blinker == out_side || self.player_blinker == 3 { self.player_signalling + dt } else { 0.0 };
        let mut views: Vec<PlayerView> = Vec::new();
        if let Some((c, h, hl, hw, v)) = player {
            views.push(PlayerView { id: 0, pos: c, heading: h as f32, half_len: hl, half_width: hw, speed: v, still: self.player_still, blinker: self.player_blinker, signalling: self.player_signalling, priority: self.player_priority });
        }
        for &(id, (c, h, hl, hw, v)) in &others {
            let still = self.others_still.entry(id).or_insert(0.0);
            *still = if v.abs() < 0.3 { *still + dt } else { 0.0 };
            views.push(PlayerView { id, pos: c, heading: h as f32, half_len: hl, half_width: hw, speed: v, still: *still, blinker: 0, signalling: 0.0, priority: false });
        }
        self.others_still.retain(|id, _| others.iter().any(|o| o.0 == *id));
        self.sim.players = views;
        self.sim.people = self.people.iter().map(|&(pos, vel, waiting, still, on_crossing, at_kerb, waits_at)| Person { pos, vel, on_crossing, still, waiting_for_bus: waiting, at_kerb, waits_at }).collect();
        self.sim.walkers = self.walkers.clone();
        self.sim.parked = self.parked.clone();
        self.sim.day_time = self.day_time;
        self.sim.time_scale = self.time_scale;
        // the bus services and the parking plans: what they ask of the driver this frame
        self.note_through_lanes();
        // [ROLLBACK r4d-B2] buses taken off the road this frame as the last resort
        let mut respawn: Vec<usize> = Vec::new();
        self.links = CarLinks::of(&self.cars);
        let t_pre = std::time::Instant::now();
        for i in 0..self.cars.len() {
            let way = omsi_traffic::perception::way_lanes(&self.net, &self.cars[i].ai.state, 200.0);
            let (lead_close, follower_close) = if self.cars[i].bus.is_some() { self.bus_queue_flags(i, player) } else { (false, false) };
            // how long somebody has stood right behind this bus (a roadside layover gives
            // way to the traffic it holds up)
            // [round 6 B1] the lanes beside its own it covers at a junction, and the queue it
            // holds there: for its layover's way off only (`bus_detour`), not the roadside
            // layover's early start nor the early-start lead's wait
            let fan = self.fan_cover(i);
            let (fan_secs, fan_queued) = self.fan_queue(i, &fan);
            let queue_behind_secs = if self.cars[i].bus.is_some() {
                let me = &self.cars[i];
                let cars = self.links.following(me.id).iter().map(|&j| &self.cars[j]).filter(|o| o.standing()).map(|o| o.ai.stopped).fold(0.0f32, f32::max);
                // the player's bus standing close behind it, heading the same way, counts too
                let player_behind = player.filter(|p| p.4.abs() < omsi_traffic::driver::STANDING_SPEED).map_or(0.0, |p| {
                    let h = (me.ai.heading as f64).to_radians();
                    let fwd = DVec2::new(h.sin(), h.cos());
                    let rel = (p.0 - me.ai.pos).truncate();
                    let behind = -rel.dot(fwd) as f32 - me.ai.state.rear - p.2;
                    let beside = rel.perp_dot(fwd).abs() as f32;
                    let same_way = ((p.1 - me.ai.heading as f64 + 180.0).rem_euclid(360.0) - 180.0).abs() < 45.0;
                    if same_way && (0.0..15.0).contains(&behind) && beside < 2.5 { self.player_still } else { 0.0 }
                });
                cars.max(player_behind)
            } else {
                0.0
            };
            let mut detour = if self.cars[i].bus.is_some() { self.bus_detour(i, queue_behind_secs, &fan, fan_secs, fan_queued) } else { None };
            // [ROLLBACK r4d-B3] a vehicle close behind it (the player's too), and its body over
            // a through lane
            let (close_behind, covers_through) = if self.cars[i].bus.is_some() {
                let me = &self.cars[i];
                let cars = self.links.following(me.id).iter().any(|&j| self.cars[j].ai.last.lead.is_some_and(|l| l.gap < crate::bus_service::PACE_FOLLOWER));
                let player_close = player.is_some_and(|p| {
                    let h = (me.ai.heading as f64).to_radians();
                    let fwd = DVec2::new(h.sin(), h.cos());
                    let rel = (p.0 - me.ai.pos).truncate();
                    let behind = -rel.dot(fwd) as f32 - me.ai.state.rear - p.2;
                    (0.0..crate::bus_service::PACE_FOLLOWER).contains(&behind) && rel.perp_dot(fwd).abs() < 2.5
                });
                let st = &me.ai.state;
                let through = self.through.through(&self.net, st.lane, me.id);
                (cars || player_close, crate::bus_service::covers_through_lane(&self.net, st.lane, st.lateral, me.ai.half_width, through))
            } else {
                (false, false)
            };
            // [ROLLBACK r4d-B2] no loop and no room within range, a queue behind it: the last
            // resort, out of anybody's sight - off the road, and back on its timetable for
            // its departure (as a bus off a dead end)
            if matches!(detour, Some(None)) {
                if let Some(r) = self.respawn_at(i) {
                    respawn.push(i);
                    self.respawned.push(r);
                    detour = None;
                }
            }
            let day_time = self.day_time;
            let stop_wishes = &self.stop_wishes;
            let net = &self.net;
            let viewer = self.viewer;
            let car = &mut self.cars[i];
            let mut ext = External { at_stop: car.at_stop(), standing_for: car.standing_for(day_time), priority: car.vehicle.var("TrafficPriority").unwrap_or(0.0) > 0.5, ..Default::default() };
            // [round 9 A2 fix pass 2, 3] beyond the horizon with its lamps dark, a random car
            // heading for the loaded tiles' edge ends its trip there moving
            // (`External::edge_trip_end`, the rule `edge_trip_end`)
            if !car.serves_stops() {
                let st = &car.ai.state;
                // (where its front meets the edge, as `plan` rule 10b stops it: the lane's end
                // less half a metre)
                let d_end = way.iter().find(|&&(l, _)| net.lanes.get(l).is_some_and(|x| x.loaded_edge && x.next.is_empty())).map(|&(l, d0)| d0 + net.lanes[l].length() - 0.5 - st.front);
                ext.edge_trip_end = edge_trip_end(EdgeFlag {
                    d_end,
                    beyond: viewer.is_some_and(|v| v.beyond_horizon((car.vehicle.position - v.pos).length(), body_height(&car.vehicle.ty), body_radius(&car.vehicle.ty))),
                    blinker: st.blinker,
                    braking: st.braking,
                    prev: car.ai.external.edge_trip_end,
                    speed: st.speed,
                    comfort_brake: car.ai.params.comfort_brake,
                });
            }
            if let Some(service) = car.bus.as_mut() {
                let wanted = stop_wishes.as_ref().map(|(alighting, waiting)| alighting.contains(&car.id) || service.stops.front().is_some_and(|s| waiting.contains(&s.id)));
                // a stop on a through lane (random traffic runs there) with no bay: the bus
                // at it blocks the lane
                let roadside = service.stops.front().is_some_and(|s| {
                    s.bay.abs() < 0.3 && car.ai.state.route.get(s.ri).and_then(|&l| net.lanes.get(l)).is_some_and(|l| l.density > 0.0)
                });
                let v_nom = net.lanes.get(car.ai.state.lane).map_or(13.9, |l| (l.speed_limit_kmh * car.ai.params.speed_factor).min(car.ai.state.max_speed_kmh).max(10.0) / 3.6);
                let ctx = crate::bus_service::Ctx {
                    close_behind,
                    covers_through,
                    people_hold: car.ai.last.why == Why::People,
                    v_nom,
                    roadside,
                    queue_behind_secs,
                    wanted,
                    net,
                    way: &way,
                    day_time,
                    dt,
                    id: car.id,
                    stopped: car.ai.stopped,
                    passing: car.ai.passing.is_some(),
                    lead_close,
                    follower_close,
                    debug: debug || omsi_cfg::env::var_os("OMSI_DEBUG_PAX").is_some(),
                };
                match detour {
                    Some(Some(d)) => service.start_detour(&mut car.ai.state, net, d, day_time, car.id),
                    Some(None) => service.no_detour(day_time, car.id),
                    None => {}
                }
                ext.stop_at = service.step(&mut car.ai.state, &mut car.vehicle, &ctx);
                // [P8 pass 1, B1] the stops whose waiters it pulls in beside
                ext.serves = [service.stops.front().map(|s| s.id), service.served];
                // [round 6 B2] its trip over at the end of its track: on off the route's end
                ext.trip_over = service.drives_off();
                if std::mem::take(&mut service.rerouted) {
                    // (the route it drives now, for the recording)
                    self.recorder.vehicle_changed(self.sim.time, &car.ai);
                }
                // [ROLLBACK r4d-B3] ahead of its timetable: slower
                ext.pace = Some(service.pace);
                // [round 7 A7] at or into a stop (or a place parked away) its bay; between
                // stops the traffic's rules place it (the kerb swerve, the settle creep: a bay
                // of its lateral target every frame kept both from it)
                ext.bay = service.bay_wish.or(service.at_stop().then_some(car.ai.state.lateral_target));
                ext.blinker_off = matches!(service.phase, Phase::Boarding | Phase::Waiting);
                // [ROLLBACK aiparked-73] a long layover at the terminus: engine off
                car.vehicle.ai_parked = service.parked(day_time);
            }
            // parking: stop beside the space, move over into it, and stand there
            if let Some(mut plan) = car.park {
                let st = &mut car.ai.state;
                match way.iter().find(|w| w.0 == plan.lane) {
                    None if st.lane != plan.lane => car.park = None,
                    found => {
                        let d = found.map(|w| w.1).unwrap_or(-st.s) + plan.s;
                        if d < 80.0 {
                            ext.stop_at = Some(ext.stop_at.map_or(d + st.front + 0.3, |x| x.min(d + st.front + 0.3)));
                            ext.signal = Some(2);
                            let len = (plan.lat.abs() * 6.0).clamp(10.0, 22.0);
                            if d < len + 1.0 {
                                ext.bay = Some(plan.lat);
                                plan.ramped = true;
                            }
                            if d.abs() < 1.5 && st.speed < 0.2 && (st.lateral - plan.lat).abs() < 0.3 {
                                plan.done = true;
                            }
                        }
                        if d < -4.0 && !plan.done {
                            car.park = None;
                        } else {
                            car.park = Some(plan);
                        }
                    }
                }
            }
            car.ai.external = ext;
        }
        self.links = CarLinks::default();
        // the drivers
        let placeholder = || AiVehicle::new(0, VehicleClass::Car, AiState::new(0, 0.0, 0), 0.0, 0);
        // (what the bus services and the parking plans set on the ways themselves, for the
        // recording)
        let steer: Vec<(u64, omsi_traffic::record::Steer)> = self.cars.iter().filter(|c| c.bus.is_some() || c.park.is_some()).map(|c| (c.id, omsi_traffic::record::Steer::of(&c.ai.state))).collect();
        let mut drivers: Vec<AiVehicle> = self.cars.iter_mut().map(|c| std::mem::replace(&mut c.ai, placeholder())).collect();
        // the frame's inputs into the recording (the people's extras beside the very
        // values the frame sees)
        if self.people_records.len() == self.sim.people.len() {
            for (r, p) in self.people_records.iter_mut().zip(&self.sim.people) {
                r.set_person(p);
            }
        }
        let t_rec = std::time::Instant::now();
        let cars = &self.cars;
        self.recorder.frame(&self.sim, dt, &drivers, &self.people_records, &steer, &|i| cars.get(i).map(|c| c.vehicle.ty.def.path.file_name().unwrap_or_default().to_string_lossy().into_owned()).unwrap_or_default());
        let t_sim = std::time::Instant::now();
        self.sim.tick(&self.net, dt, &mut drivers);
        self.tick_inner = [(t_rec - t_pre).as_secs_f64(), (t_sim - t_rec).as_secs_f64(), t_sim.elapsed().as_secs_f64()];
        for (c, v) in self.cars.iter_mut().zip(drivers) {
            c.ai = v;
        }
        self.day_time = self.sim.day_time;
        if debug {
            self.log_bus_lateral();
        }
        let mut remove: Vec<usize> = self.sim.removed.clone();
        remove.extend(respawn.iter().copied());
        remove.sort_unstable_by(|a, b| b.cmp(a));
        remove.dedup();
        let t_plan = std::time::Instant::now();
        // what the scripts are told, and the follow-camera triggers
        let mut frames: Vec<Option<AiFrame>> = Vec::with_capacity(self.cars.len());
        let mut going = vec![false; self.cars.len()];
        for &i in &remove {
            going[i] = true;
        }
        for (i, car) in self.cars.iter().enumerate() {
            if going[i] {
                frames.push(None);
                continue;
            }
            let st = &car.ai.state;
            let last = &car.ai.last;
            if self.first_red.is_none() && last.why == Why::Light && st.speed < 0.5 {
                self.first_red = Some((car.id, self.time));
            }
            if self.first_yield.is_none() && last.why == Why::Junction && st.speed < 0.5 {
                self.first_yield = Some((car.id, self.time));
            }
            if self.first_passer.is_none() && car.ai.passing.is_some() {
                self.first_passer = Some((car.id, self.time));
            }
            if self.first_turner.is_none() && self.net.lanes.get(st.lane).is_some_and(|l| l.turn != 0) {
                self.first_turner = Some((car.id, self.time));
            }
            if st.change.is_some_and(|c| !c.bypass && st.route.is_empty()) && self.last_overtaker.map_or(true, |(id, _)| id != car.id) {
                self.last_overtaker = Some((car.id, self.time));
            }
            let priority_warning = car.ai.external.priority
                && (last.lead.is_some_and(|l| l.gap < PRIORITY_WARN_GAP && l.speed < st.speed + 0.5) || last.stop.is_some_and(|s| s - st.front < PRIORITY_WARN_GAP));
            frames.push(Some(AiFrame {
                speed: st.speed,
                odometer: st.odometer,
                steer_deg: 0.0,
                blinker: car.ai.blinker_shown.0,
                brake: st.braking,
                lights: self.night,
                at_station: car.at_station() as i32,
                at_station_side: car.at_station_side(),
                priority_warning,
            }));
        }
        // Who can be seen: a car out of the view (and farther than the mirrors and the
        // shadows reach) leaves its animations as they are and is not drawn at all.
        if let Some(v) = self.viewer {
            for c in &mut self.cars {
                let p = c.vehicle.position;
                let r = (c.ai.state.front + c.ai.state.rear).abs().max(4.0) as f64 + 2.0;
                c.vehicle.ai_visuals = (p - v.pos).length() < UNSEEN_NEAR || v.frames(p, r);
            }
        }
        let t_par = std::time::Instant::now();
        // The bodies and the scripts of the AI vehicles run in parallel: each car follows
        // its own way and its OMSI script is its own little machine reading only its own
        // state; with thirty cars and a dozen timetable buses they were the largest single
        // cost of a frame.
        {
            use rayon::prelude::*;
            let net = &self.net;
            type Work<'a> = (&'a AiState, &'a mut AiBody, &'a mut VehicleInstance, &'a mut AiFrame, &'a mut std::collections::VecDeque<(f64, DVec3)>, &'a mut f32);
            let mut work: Vec<Work> = self
                .cars
                .iter_mut()
                .zip(frames.iter_mut())
                .filter_map(|(c, f)| {
                    let f = f.as_mut()?;
                    Some((&c.ai.state, &mut c.body, &mut c.vehicle, f, &mut c.rail_trail, &mut c.ai_secs))
                })
                .collect();
            work.sort_by(|a, b| b.5.total_cmp(a.5));
            let mut jobs: Vec<Vec<Work>> = Vec::new();
            for w in work {
                match jobs.last_mut() {
                    Some(job) if *w.5 < AI_JOB_SECS && *job[0].5 < AI_JOB_SECS && job.len() < 4 => job.push(w),
                    _ => jobs.push(vec![w]),
                }
            }
            let profile = omsi_cfg::env::var_os("OMSI_PROFILE").is_some();
            // (a few cars per job: every job handed out wakes a worker, and the waking cost
            // the main thread more than a car's work)
            jobs.into_par_iter().flatten_iter().for_each(|(state, body, vehicle, frame, trail, secs)| {
                let t0 = std::time::Instant::now();
                let ground = vehicle.ground.clone();
                let contact = vehicle.contact.clone();
                let rail = body.kind == MotionKind::Rail;
                if rail {
                    record_rail_trail(trail, state.odometer as f64, state.way_point(net, 0.0));
                }
                let trail = &*trail;
                let behind = |d: f64| rail_behind(trail, state, net, d);
                body.step(
                    dt,
                    state.speed,
                    &|d| if rail && d < 0.0 { behind(-d as f64) } else { state.way_point(net, d) },
                    ground.as_ref().map(|g| g.as_ref() as &dyn Fn(f64, f64) -> Option<f64>),
                    contact.as_deref(),
                );
                body.apply(vehicle);
                if rail && !vehicle.trailers.is_empty() {
                    // the coupled cars (a train's, a tram's sections) on the track it
                    // came along, not dragged round the bends like a road trailer
                    vehicle.retrail(0.0, &|d| Some(behind(d)));
                }
                frame.steer_deg = body.steer;
                let t1 = std::time::Instant::now();
                vehicle.update_ai(dt, frame);
                *secs = t0.elapsed().as_secs_f32();
                if profile && t0.elapsed().as_secs_f64() > 0.01 {
                    log::info!("  slow AI frame: {} body {:.1} ms, scripts {:.1} ms", vehicle.ty.def.path.display(), (t1 - t0).as_secs_f64() * 1000.0, t1.elapsed().as_secs_f64() * 1000.0);
                }
            });
        }
        self.tick_split = [(t_plan - t_start).as_secs_f64(), (t_par - t_plan).as_secs_f64(), t_par.elapsed().as_secs_f64()];
        if omsi_cfg::env::var_os("OMSI_PROFILE").is_some() {
            let (sum, n) = &mut self.tick_inner_sum;
            for (s, v) in sum.iter_mut().zip(self.tick_inner) {
                *s += v;
            }
            *n += 1;
            if *n >= TICK_PROFILE_FRAMES {
                let ms = |x: f64| x * 1000.0 / *n as f64;
                log::info!("traffic profile over {} frames, {} cars: pre-plan {:.2} ms, recorder {:.2} ms, Simulation::tick {:.2} ms per frame", n, self.cars.len(), ms(sum[0]), ms(sum[1]), ms(sum[2]));
                self.tick_inner_sum = ([0.0; 3], 0);
            }
        }
        if debug {
            // a car pulled round harder than a driver would: what way was it given?
            for (c, fr) in self.cars.iter().zip(&frames) {
                if fr.is_none() || c.body.a_lat.abs() < 4.0 || !self.logged_hard.insert(c.id) {
                    continue;
                }
                let st = &c.ai.state;
                let lanes: Vec<String> = std::iter::once(st.lane).chain(st.upcoming()).take(5).map(|l| format!("{l} (len {:.1})", self.net.lanes[l].length())).collect();
                log::info!("t={:.1}: car {} at {:.1} m/s pulled {:.1} m/s² sideways (steering {:.1}°), s {:.1}, way {}", self.time, c.id, st.speed, c.body.a_lat, c.body.steer, st.s, lanes.join(" / "));
            }
        }
        if let Some(f) = self.trace.as_mut() {
            use std::io::Write;
            if let Some((c, h, hl, hw, v)) = player {
                let _ = writeln!(f, "{:.3},0,player,{:.3},{:.3},{:.3},{:.3},0,0,0,{:.3},-1,0,0,0,0,0,0,0,0,0,0,{hl:.2},{hl:.2},{hw:.2},0,,0,None", self.time, c.x, c.y, c.z, h, v);
            }
            let buses_only = omsi_cfg::env::var_os("OMSI_TRACE_AI_BUSES").is_some();
            for (c, fr) in self.cars.iter().zip(&frames) {
                let Some(fr) = fr else { continue };
                if buses_only && !c.serves_stops() {
                    continue;
                }
                let (v, st, last) = (&c.vehicle, &c.ai.state, &c.ai.last);
                let lane_heading = self.net.lanes[st.lane].at(st.s).1.rem_euclid(360.0);
                let _ = writeln!(f, "{:.3},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{},{:.2},{},{},{:.2},{:.2},{},{:.2},{},{},{},{:.2},{:.2},{:.2},{},{},{:.1},{:?},{:.3}", self.time, c.id, v.ty.def.path.file_stem().unwrap_or_default().to_string_lossy(), v.position.x, v.position.y, v.position.z, v.heading, v.pitch, v.bank, fr.steer_deg, st.speed, st.lane, st.s, fr.blinker, self.net.lanes[st.lane].turn, lane_heading, st.lateral, c.at_station() as i32, st.acc, (last.why == Why::Junction) as i32, (last.why == Why::Light) as i32, c.ai.passing.is_some() as i32, st.front, st.rear, c.ai.half_width, c.serves_stops() as i32, last.why.name(), last.stop.map(|s| s - st.front).unwrap_or(999.0).min(999.0), c.bus.as_ref().map(|b| b.phase), self.net.lanes[st.lane].at(st.s).0.z);
            }
        }
        let mut overlaps = 0;
        if omsi_cfg::env::var_os("OMSI_CHECK_OVERLAP").is_some() {
            overlaps = self.check_overlaps(player, &others);
        }
        self.save_on_trouble(overlaps > 0);
        for i in remove {
            let c = self.cars.swap_remove(i);
            {
                let p = c.vehicle.position;
                let r = (c.ai.state.length as f64 * 0.5).max(2.0);
                // ([round 9 A2 fix pass 2] driven off the loaded tiles' edge as its trip's end:
                // the far trip end, not the network's)
                let edge = self.net.lanes.get(c.ai.state.lane).is_some_and(|l| l.loaded_edge && l.next.is_empty());
                let rule = network_end_rule(respawn.contains(&i), edge, c.ai.external.edge_trip_end);
                let seen = self.viewer.is_some_and(|v| v.frames(p, r));
                self.removed_far += (rule == RemoveRule::FarDeadEnd && seen) as usize;
                if rule == RemoveRule::FarDeadEnd && self.debug_population {
                    // ([round 9 A2 fix pass 2] its lamps as it goes: none may glow)
                    log::info!("population t={:.1}: car {} trip end at the loaded tiles' edge, {:.0} m from the eye, in frame {}, speed {:.1}, braking {}, blinker {}, lamps lit {}", self.time, c.id, self.viewer.map(|v| (p - v.pos).length()).unwrap_or(0.0), seen, c.ai.state.speed, c.ai.state.braking, c.ai.state.blinker, self.lamps_lit());
                }
                // ([round 9 A2 fix pass 1] with its distance from the eye, as every removal)
                self.recorder.note_remove_from(c.id, rule, seen, c.ai.waiting, self.viewer.map(|v| (p - v.pos).length() as f32));
            }
            if self.debug_population {
                // [round 7 R7-1] every removal is logged with its rule: the traffic's own
                // (the network's end: no way on) and a timetable bus's last resort (its own
                // line too) as the population's are
                let p = c.vehicle.position;
                let from_eye = self.viewer.map(|v| (p - v.pos).length()).unwrap_or(f64::NAN);
                let rule = if respawn.contains(&i) { "a timetable bus taken off as the last resort" } else { "the network's end, no way on" };
                log::info!("population t={:.1}: car {} removed at ({:.0}, {:.0}), {rule}; {:.0} m from the eye, lane {}, v {:.1}, why {}, stood {:.0} s, waiting {:.1} s", self.time, c.id, p.x, p.y, from_eye, c.ai.state.lane, c.ai.state.speed, c.ai.last.why.name(), c.ai.stopped, c.ai.waiting);
            }
            if c.serves_stops() {
                self.riders_out.push(c.id);
            }
            // [ROLLBACK r4d-B1] a bus still on its trip that drove off a dead end
            if c.bus.as_ref().is_some_and(|b| !b.trip_done()) && !c.released && !self.respawned.iter().any(|r| r.car == c.id) {
                if omsi_cfg::env::var_os("OMSI_DEBUG_TRAFFIC").is_some() {
                    log::info!("scheduled bus {}: off the dead end of lane {} at ({:.0}, {:.0})", c.id, c.ai.state.lane, c.vehicle.position.x, c.vehicle.position.y);
                }
                self.dead_ended.push(c.id);
            }
            self.orphan_sounds.extend(c.sounds);
            // the renders go back to the world at the next sync
            self.released.push(c.render);
            self.released.extend(c.trailer_renders);
        }
    }

    /// What holds each car that has stood for over a minute (the offscreen traffic health
    /// report): why, its indicator, its lane and where it stands.
    pub fn stuck_report(&self) -> Vec<String> {
        self.cars
            .iter()
            .filter(|c| c.ai.stopped > 60.0 && !c.at_stop())
            .map(|c| {
                let st = &c.ai.state;
                let last = &c.ai.last;
                format!(
                    "car {} ({}) stood {:.0} s: {} (stop {:?}, lead gap {:?}, waits for {:?}, impatience {:.2}) blinker {} lane {} s {:.1} at ({:.0}, {:.0})",
                    c.id,
                    c.vehicle.ty.def.path.file_stem().unwrap_or_default().to_string_lossy(),
                    c.ai.stopped,
                    last.why.name(),
                    last.stop.map(|s| (s * 10.0).round() / 10.0),
                    last.lead.map(|l| (l.gap * 10.0).round() / 10.0),
                    last.wait_for,
                    c.ai.impatience.value,
                    c.ai.blinker_shown.0,
                    st.lane,
                    st.s,
                    c.vehicle.position.x,
                    c.vehicle.position.y
                )
            })
            .collect()
    }

    /// Record the traffic into `path` from now on (see `omsi_traffic::record`).
    pub fn start_recording(&mut self, path: &Path) -> std::io::Result<()> {
        self.recorder.start_full(path, &self.net, &self.sim, &self.map_name)
    }

    /// Record into a new file in `recordings_dir`; returns it.
    pub fn start_recording_new(&mut self) -> std::io::Result<std::path::PathBuf> {
        let dir = recordings_dir().ok_or_else(|| std::io::Error::other("no folder for the recordings"))?;
        let path = dir.join(format!("{}-{}.rec", self.map_name, hhmmss()));
        self.start_recording(&path)?;
        Ok(path)
    }

    /// Save the last minute of traffic to a new file in `recordings_dir`; returns it.
    pub fn save_last_minute(&mut self, trigger: &str) -> std::io::Result<std::path::PathBuf> {
        let dir = recordings_dir().ok_or_else(|| std::io::Error::other("no folder for the recordings"))?;
        let path = dir.join(format!("{}-{}-{trigger}.rec", self.map_name, hhmmss()));
        self.recorder.note(self.sim.time, &format!("saved: {trigger}"));
        self.recorder.dump_ring(&self.net, &self.sim, &self.map_name, &path, trigger)
    }

    /// The game changed car `ci`'s way between frames (a timetable bus given its next trip,
    /// let go, its route extended): into the recording.
    pub fn note_vehicle_changed(&mut self, ci: usize) {
        if let Some(c) = self.cars.get(ci) {
            self.recorder.vehicle_changed(self.sim.time, &c.ai);
        }
    }

    /// Save the last minute by itself when something went wrong: a car that has stood for
    /// a minute held by nothing that lets it go by itself (not a light, not its stop), two
    /// cars inside each other (`overlap`: seen by `check_overlaps`; else looked for every
    /// two seconds here).
    fn save_on_trouble(&mut self, overlap: bool) {
        let mut trigger: Option<&str> = None;
        let stuck: Vec<u64> = self.cars.iter().filter(|c| c.ai.stopped > 60.0 && !c.at_stop() && c.ai.last.why != Why::Light && !self.stuck_saved.contains(&c.id)).map(|c| c.id).collect();
        if !stuck.is_empty() {
            let report = self.stuck_report();
            for id in &stuck {
                self.stuck_saved.insert(*id);
                let me = format!("car {id} ");
                for line in report.iter().filter(|l| l.starts_with(&me)) {
                    self.recorder.note(self.sim.time, line);
                }
            }
            trigger = Some("stuck");
        }
        let mut overlap = overlap;
        if !overlap && self.time - self.overlap_checked >= 2.0 {
            self.overlap_checked = self.time;
            if let Some((a, b)) = self.overlapping_pair() {
                self.recorder.note(self.sim.time, &format!("car {a} and car {b} are inside each other"));
                overlap = true;
            }
        }
        if overlap {
            trigger = Some("overlap");
        }
        let Some(trigger) = trigger else { return };
        let (n, last) = self.auto_saves;
        if n >= AUTO_SAVES || (n > 0 && self.time - last < omsi_traffic::record::RING_SECS as f32) {
            return;
        }
        self.auto_saves = (n + 1, self.time);
        match self.save_last_minute(trigger) {
            Ok(p) => log::info!("traffic: {trigger}: the last minute of traffic goes to {}", p.display()),
            Err(e) => log::warn!("traffic: {trigger}: the last minute could not be saved: {e}"),
        }
    }

    /// Two AI cars whose bodies are inside each other by more than 20 cm (their ids): the
    /// first pair `overlaps` finds.
    fn overlapping_pair(&self) -> Option<(u64, u64)> {
        self.overlaps(None, &[]).into_iter().next().map(|(a, b, ..)| (self.cars[a].id, self.cars[b].id))
    }

    /// Every two bodies inside each other by more than `OVERLAP_DEPTH`: (car, other car or
    /// `usize::MAX` for the player box `Some(k)` of `player` then `others`, where). The AI
    /// cars by their `footprints` (coupled parts too), the players by their boxes.
    fn overlaps(&self, player: Option<PlayerBox>, others: &[(u32, PlayerBox)]) -> Vec<(usize, usize, Option<usize>, DVec2)> {
        let feet = self.footprints();
        let boxes: Vec<Footprint> = player
            .iter()
            .chain(others.iter().map(|(_, b)| b))
            .map(|&(c, h, hl, hw, v)| {
                let hr = h.to_radians();
                Footprint { car: usize::MAX, center: c.truncate(), fwd: DVec2::new(hr.sin(), hr.cos()), right: DVec2::new(hr.cos(), -hr.sin()), half_len: hl as f64, half_w: hw as f64, speed: v, z: c.z }
            })
            .collect();
        let inside = |fa: &Footprint, fb: &Footprint| (fa.z - fb.z).abs() <= 3.0 && fa.overlaps(fb, -OVERLAP_DEPTH);
        let mut out = Vec::new();
        for (a, fa) in feet.iter().enumerate() {
            for fb in feet.iter().skip(a + 1) {
                if fb.car != fa.car && inside(fa, fb) {
                    out.push((fa.car, fb.car, None, fa.center));
                }
            }
            for (k, fb) in boxes.iter().enumerate() {
                if inside(fa, fb) {
                    out.push((fa.car, usize::MAX, Some(k), fa.center));
                }
            }
        }
        out
    }

    /// `OMSI_TRAFFIC_NETWORK_DUMP=<folder>`: the network and its junctions as
    /// `<map>-network.json` and `.geojson` there (written again whenever lanes come).
    fn dump_network(&self) {
        let Ok(dir) = omsi_cfg::env::var("OMSI_TRAFFIC_NETWORK_DUMP") else { return };
        let dir = Path::new(&dir);
        let _ = std::fs::create_dir_all(dir);
        let base = dir.join(format!("{}-network", self.map_name));
        let out = [("json", omsi_traffic::export::network_json(&self.net, &self.sim.junctions)), ("geojson", omsi_traffic::export::network_geojson(&self.net, &self.sim.junctions))];
        for (ext, text) in out {
            let path = base.with_extension(ext);
            match std::fs::write(&path, text) {
                Ok(()) => log::info!("traffic: the network ({} lanes) written to {}", self.net.lanes.len(), path.display()),
                Err(e) => log::warn!("traffic: {}: {e}", path.display()),
            }
        }
    }

    /// Log every two bodies inside each other (`overlaps`; each pair once in ten seconds);
    /// returns how many new ones there were.
    fn check_overlaps(&mut self, player: Option<PlayerBox>, others: &[(u32, PlayerBox)]) -> usize {
        static SEEN: std::sync::OnceLock<parking_lot::Mutex<HashMap<(u64, u64), f32>>> = std::sync::OnceLock::new();
        let seen = SEEN.get_or_init(|| parking_lot::Mutex::new(HashMap::new()));
        let why = |c: &AiCar| format!("{} v {:.1} lane {} s {:.1} lat {:.2} why {:?} passing {} change {}", c.vehicle.ty.def.path.file_stem().unwrap_or_default().to_string_lossy(), c.ai.state.speed, c.ai.state.lane, c.ai.state.s, c.ai.state.lateral, c.ai.last.why, c.ai.passing.is_some(), c.ai.state.change.is_some());
        let mut found = 0usize;
        for (a, b, k, at) in self.overlaps(player, others) {
            let ca = &self.cars[a];
            // (the player boxes: u64::MAX for the player, below it the others by their id)
            let (other, desc) = match k {
                None => (self.cars[b].id, format!("car {} ({})", self.cars[b].id, why(&self.cars[b]))),
                Some(k) => {
                    let (id, speed) = if player.is_some() && k == 0 { (u64::MAX, player.map_or(0.0, |p| p.4)) } else {
                        let (pid, b) = others[k - player.is_some() as usize];
                        (u64::MAX - 1 - pid as u64, b.4)
                    };
                    (id, format!("player box {} v {:.1}", u64::MAX - id, speed))
                }
            };
            let key = (ca.id.min(other), ca.id.max(other));
            let mut m = seen.lock();
            if m.get(&key).is_some_and(|t| self.time - *t < 10.0) {
                continue;
            }
            m.insert(key, self.time);
            log::info!("t={:.1}: OVERLAP car {} ({}) with {} at ({:.1}, {:.1})", self.time, ca.id, why(ca), desc, at.x, at.y);
            found += 1;
        }
        found
    }

    /// `OMSI_DEBUG_TRAFFIC`: [round 7 A7] a timetable bus between stops (no bay asked for)
    /// that the traffic's own rules move sideways - the kerb swerve round a parked car or
    /// people, the settle creep of a lane change settled in place - logged as it starts.
    fn log_bus_lateral(&self) {
        static ON: std::sync::OnceLock<parking_lot::Mutex<hashbrown::HashSet<u64>>> = std::sync::OnceLock::new();
        let mut on = ON.get_or_init(Default::default).lock();
        for c in self.cars.iter().filter(|c| c.bus.is_some()) {
            let st = &c.ai.state;
            let free = c.ai.external.bay.is_none() && c.ai.passing.is_none() && st.change.is_none();
            let what = if !free {
                None
            } else if st.lateral_target.abs() > 0.05 {
                Some("swerves")
            } else if st.settled_offset && st.lateral.abs() > 0.05 {
                Some("settles")
            } else {
                None
            };
            match what {
                Some(w) if on.insert(c.id) => log::info!("t={:.1}: timetable bus {} between stops {w}: lateral {:+.2} -> {:+.2} m on lane {} at {:.1} m/s ({:?})", self.time, c.id, st.lateral, st.lateral_target, st.lane, st.speed, c.ai.last.why),
                None => {
                    on.remove(&c.id);
                }
                _ => {}
            }
        }
    }

    /// `OMSI_DEBUG_LIGHTS`: every change of the lights of the chosen programs, with the
    /// game time and the program's cycle position.
    fn log_lights(&mut self) {
        let Some(sel) = self.light_log.as_deref() else {
            return;
        };
        let near = sel.eq_ignore_ascii_case("near");
        let chosen: Vec<usize> = sel
            .split(',')
            .filter_map(|v| v.trim().parse().ok())
            .collect();
        let viewer = self.viewer.map(|v| v.pos);
        for (ci, c) in self.sim.lights.iter().enumerate() {
            let pick = sel.eq_ignore_ascii_case("all")
                || chosen.contains(&ci)
                || (near && viewer.is_some());
            if !pick {
                continue;
            }
            if near {
                // the programs within 150 m of the camera (by the lanes they control)
                let vp = viewer.unwrap();
                let close = self.net.lanes.iter().any(|l| {
                    l.traffic_light.map(|t| t.0) == Some(ci) && (l.start() - vp).length() < 150.0
                });
                if !close {
                    continue;
                }
            }
            for li in 0..c.lights.len() {
                let s = c.state(li);
                let prev = self.light_prev[ci][li];
                if s != prev {
                    let h = self.day_time.rem_euclid(86400.0);
                    log::info!("light {ci}.{li}: {:?} ({s}) at {:02}:{:02}:{:05.2} (cycle {:.2} of {:.0} s{}{})", TrafficLightController::aspect(s), (h / 3600.0) as u32, ((h / 60.0) % 60.0) as u32, h % 60.0, c.time, c.cycle_len(), if c.held { ", held" } else { "" }, if c.request.get(li).copied().unwrap_or(false) { ", requested" } else { "" });
                    self.light_prev[ci][li] = s;
                }
            }
        }
    }

    /// Play the `[sound_ai]` sets of the cars near `listener` (others are silenced).
    /// `street_cond` is the state of the road (see `VehicleHost::street_cond`): the stock
    /// AI sound configuration fades `WetLane_1`/`WetLane_2` in with it, which is what a car
    /// driving past through the wet sounds like. `muffled`: the listener (the player) sits in
    /// a cabin right now, so every AI car's sound is heard through that bodywork and glass -
    /// a passing car's horn does not simply sound like the street outside once the windows
    /// are shut.
    pub fn update_audio(
        &mut self,
        audio: &omsi_audio::AudioEngine,
        listener: DVec3,
        street_cond: f32,
        muffled: bool,
    ) {
        let freed = audio.trim_clips(std::time::Duration::from_secs(60));
        if freed > 0 && omsi_cfg::env::var_os("OMSI_PROFILE").is_some() {
            log::info!(
                "sound clips: {:.1} MB nobody used for a minute let go",
                freed as f64 / 1e6
            );
        }
        for mut s in self.orphan_sounds.drain(..) {
            s.stop_all(audio);
        }
        let near = 250.0;
        for c in &mut self.cars {
            let d = (c.vehicle.position - listener).length();
            if d > near * 1.2 {
                if let Some(mut s) = c.sounds.take() {
                    s.stop_all(audio);
                }
                continue;
            }
            if c.sounds.is_none() && d < near {
                let def = &c.vehicle.ty.def;
                let Some(rel) = def.sound_ai.clone().or_else(|| def.sound.clone()) else {
                    continue;
                };
                let path = omsi_cfg::resolve_path(def.dir(), &rel);
                let cfg = self
                    .sound_cfgs
                    .entry(path.clone())
                    .or_insert_with(|| {
                        omsi_vehicle::SoundCfg::load(&path)
                            .map_err(|e| log::warn!("{e}"))
                            .ok()
                            .map(Arc::new)
                    })
                    .clone();
                if let Some(cfg) = cfg {
                    let dir = path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
                    // an articulated bus's rear section sounds too (its engine, on a pusher)
                    let mut parts = Vec::new();
                    for (i, t) in c.vehicle.trailers.iter().enumerate() {
                        let def = &t.ty.def;
                        let Some(rel) = def.sound_ai.clone().or_else(|| def.sound.clone()) else {
                            continue;
                        };
                        let path = omsi_cfg::resolve_path(def.dir(), &rel);
                        let part = self
                            .sound_cfgs
                            .entry(path.clone())
                            .or_insert_with(|| {
                                omsi_vehicle::SoundCfg::load(&path)
                                    .map_err(|e| log::warn!("{e}"))
                                    .ok()
                                    .map(Arc::new)
                            })
                            .clone();
                        if let Some(part) = part {
                            let dir = path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
                            parts.push((i, part, dir));
                        }
                    }
                    // the clips are read in the background the first time; silent till then
                    let ready = audio.clips_ready(&omsi_audio::SoundSet::clip_paths(&cfg, &dir))
                        && parts.iter().all(|(_, part, dir)| {
                            audio.clips_ready(&omsi_audio::SoundSet::clip_paths(part, dir))
                        });
                    if ready {
                        let number = c.vehicle.number();
                        let mut ss = omsi_audio::SoundSet::new_exterior(audio, &cfg.chosen_for(&number), &dir);
                        for (i, part, dir) in &parts {
                            ss.add_part(*i, omsi_audio::SoundSet::new_exterior(audio, &part.chosen_for(&number), dir));
                        }
                        ss.master = crate::sound_gain(&crate::SOUND_AI);
                        c.vehicle.host.snapshot_triggers = ss.curve_triggers().into_iter().collect();
                        c.sounds = Some(ss);
                    }
                }
            }
            let fired: Vec<String> = std::mem::take(&mut c.vehicle.host.fired_triggers);
            let fired_vars: Vec<(String, Vec<f32>)> = std::mem::take(&mut c.vehicle.host.fired_trigger_vars);
            let fired_files: Vec<(String, String)> =
                std::mem::take(&mut c.vehicle.host.fired_file_triggers);
            c.vehicle.host.street_cond = street_cond;
            c.vehicle.set_engine_var("StreetCond", street_cond);
            if let Some(ss) = c.sounds.as_mut().filter(|_| c.vehicle.ai_parked) {
                // [ROLLBACK aiparked-93] standing out a long layover the engine is off for the
                // ear too: its loops stop (the script's own variables did not always stop them)
                ss.stop_all(audio);
            } else if let Some(ss) = c.sounds.as_mut() {
                ss.set_muffled(muffled);
                let xf = c.vehicle.world_transform();
                let v = &c.vehicle;
                let at_fire = |t: &str, n: &str| -> Option<f32> {
                    let vals = &fired_vars.iter().rev().find(|(k, _)| k.eq_ignore_ascii_case(t))?.1;
                    v.var_slot(n).and_then(|i| vals.get(i).copied())
                };
                ss.update_fired(audio, &|n| v.var(n), &xf, &fired, &at_fire);
                ss.update_parts_fired(
                    audio,
                    &|n| v.var(n),
                    &|i| v.trailers.get(i).map(|t| t.world_transform()),
                    &fired,
                    &at_fire,
                );
                for (t, f) in &fired_files {
                    ss.play_file_trigger(audio, t, f, &|n| v.var(n), &xf);
                }
            }
        }
    }

    /// [round 7 fix pass 1, 12] How far behind its origin (m) the consist of `ty` (its rear
    /// sections, trailers, the cars of a unit) reaches standing straight: each car placed as
    /// `consist_bodies` places it, the farthest corner back. (One straight box: the rear
    /// section of an articulated bus swinging inwards in a bend is not modelled.)
    fn consist_rear(&mut self, ty: &Arc<VehicleType>) -> f32 {
        let (pos, heading) = (DVec3::ZERO, 0.0f64);
        let mut back_most = 0.0f64;
        let (mut origin, mut lead, mut lead_rev) = (pos, ty.clone(), false);
        for (t, rev) in self.trailer_chain(ty) {
            let (back, front) = omsi_sim::vehicle::coupling_points(&lead, lead_rev, &t, rev);
            let (center, car_heading) = omsi_sim::vehicle::coupling_placement(
                origin,
                heading,
                omsi_sim::vehicle::body_reversed(&lead.def, lead_rev),
                back.y,
                omsi_sim::vehicle::body_reversed(&t.def, rev),
                front.y,
            );
            let b = omsi_sim::collision::Obb::from_box(t.def.bounding_box.unwrap_or(DEFAULT_BOX), center, car_heading);
            // (heading 0: forward is +y)
            back_most = b.corners().iter().map(|c| -c.y).fold(back_most, f64::max);
            origin = center;
            lead = t;
            lead_rev = rev;
        }
        back_most as f32
    }

    /// The bodies a vehicle of `ty` would have with its origin at `pos`, heading `heading`,
    /// and its coupled parts straight behind it, each grown by `SPAWN_BODY_CLEAR`.
    fn consist_bodies(&mut self, ty: &Arc<VehicleType>, pos: DVec3, heading: f64) -> Vec<omsi_sim::collision::Obb> {
        let grown = |mut b: omsi_sim::collision::Obb| {
            b.half += glam::DVec2::splat(SPAWN_BODY_CLEAR);
            b
        };
        let mut bodies = vec![grown(omsi_sim::collision::Obb::from_box(
            ty.def.bounding_box.unwrap_or(DEFAULT_BOX),
            pos,
            omsi_sim::vehicle::body_heading(&ty.def, heading, false),
        ))];
        let (mut origin, mut lead, mut lead_rev) = (pos, ty.clone(), false);
        for (t, rev) in self.trailer_chain(ty) {
            let (back, front) = omsi_sim::vehicle::coupling_points(&lead, lead_rev, &t, rev);
            // each car stands along the consist's heading, turned round by its own
            // (absolute) orientation - never by the car in front of it
            let (center, car_heading) = omsi_sim::vehicle::coupling_placement(
                origin,
                heading,
                omsi_sim::vehicle::body_reversed(&lead.def, lead_rev),
                back.y,
                omsi_sim::vehicle::body_reversed(&t.def, rev),
                front.y,
            );
            bodies.push(grown(omsi_sim::collision::Obb::from_box(
                t.def.bounding_box.unwrap_or(DEFAULT_BOX),
                center,
                car_heading,
            )));
            origin = center;
            lead = t;
            lead_rev = rev;
        }
        bodies
    }

    /// The lanes the other way along one-way paths `lanes` (see `Schedule`'s route
    /// building: OMSI's timetable buses drive a path against its direction where the trip's
    /// station links or track say so). Only timetable routes use them: they carry no traffic
    /// and no light. Returns how many were added.
    pub fn add_reverse_twins(&mut self, lanes: &[usize]) -> usize {
        let mut new = Vec::new();
        for &l in lanes {
            if !self.twinned.insert(l) {
                continue;
            }
            let o = &self.net.lanes[l];
            let pts: Vec<DVec3> = o.points.iter().rev().copied().collect();
            let mut t = omsi_sim::traffic::LaneBuilder::polyline(pts, o.kind, o.width);
            t.key = o.key;
            t.reversed = !o.reversed;
            t.speed_limit_kmh = o.speed_limit_kmh;
            t.source = o.source;
            t.offset = o.offset;
            t.name = o.name.clone();
            t.invisible = o.invisible;
            t.priority = o.priority;
            t.no_overtaking = o.no_overtaking;
            t.density = 0.0;
            t.no_cars = true;
            new.push(t);
        }
        let n = new.len();
        if n > 0 {
            if self.recorder.is_recording() {
                self.recorder.lanes_added(self.sim.time, self.net.lanes.len(), &new);
            }
            let added = self.net.extend(new, 1.5);
            self.sim.network_grew(&self.net, added.start);
            log::debug!("traffic: {n} lanes added for timetable routes that drive a one-way path the other way ({:?})", added);
        }
        n
    }

    /// A lane from the end of `a` to the start of `b`, for a timetable route that jumps a
    /// gap in the map (a road piece deleted after the timetable's tracks were recorded - a
    /// dozen such holes of 10 to 150 m on Novi Sad): a smooth curve with the two lanes'
    /// headings at its ends, used by the timetable only. None when the two do not line up.
    pub fn add_connector(&mut self, a: usize, b: usize) -> Option<usize> {
        let (la, lb) = (&self.net.lanes[a], &self.net.lanes[b]);
        let (p0, p1) = (la.end(), lb.start());
        let d = (p1 - p0).truncate();
        let len = d.length();
        if !(2.0..=150.0).contains(&len) || la.kind != lb.kind {
            return None;
        }
        let dir = |h: f32| {
            let r = (h as f64).to_radians();
            glam::DVec2::new(r.sin(), r.cos())
        };
        let (t0, t1) = (dir(la.end_heading()), dir(lb.start_heading()));
        let chord = d / len;
        // the lanes point along the gap or turn across it (a junction whose turning path
        // the map lost, 66 of them on Novi Sad), never a reversal
        if t0.dot(chord) < 0.3 || t1.dot(chord) < 0.3 || t0.dot(t1) < -0.2 {
            return None;
        }
        // (tangents as long as the gap make a straight gap a smooth S; across a turn they
        // would swing out past the corner)
        let len_t = len * if t0.dot(t1) > 0.9 { 1.0 } else { 0.6 };
        let n = ((len / 3.0).ceil() as usize).max(2);
        let pts: Vec<DVec3> = (0..=n)
            .map(|i| {
                let t = i as f64 / n as f64;
                let (h00, h10, h01, h11) = (2.0 * t * t * t - 3.0 * t * t + 1.0, t * t * t - 2.0 * t * t + t, -2.0 * t * t * t + 3.0 * t * t, t * t * t - t * t);
                let xy = p0.truncate() * h00 + t0 * len_t * h10 + p1.truncate() * h01 + t1 * len_t * h11;
                DVec3::new(xy.x, xy.y, p0.z + (p1.z - p0.z) * t)
            })
            .collect();
        let mut l = omsi_sim::traffic::LaneBuilder::polyline(pts, la.kind, la.width);
        l.speed_limit_kmh = la.speed_limit_kmh.min(lb.speed_limit_kmh);
        l.name = "(timetable connector)".into();
        l.density = 0.0;
        l.no_cars = true;
        if self.recorder.is_recording() {
            self.recorder.lanes_added(self.sim.time, self.net.lanes.len(), std::slice::from_ref(&l));
        }
        let added = self.net.extend(vec![l], 1.5);
        self.sim.network_grew(&self.net, added.start);
        log::debug!("traffic: a {len:.0} m connector lane {} from lane {a} to lane {b} for a timetable route", added.start);
        Some(added.start)
    }

    /// Hand timetable bus `ci` the next trip of its tour: its route from the lane it is on
    /// (`route[0]` is that lane, `s` where it is on it) and the trip's stops. It stays where
    /// it stands; a stop right there is served in place (its layover).
    pub fn reroute(&mut self, ci: usize, route: Vec<usize>, s: f32, stops: Vec<(usize, f32, f32, f64, i64, f32)>, layover: bool) {
        let net = &self.net;
        let car = &mut self.cars[ci];
        let lane = car.ai.state.lane;
        car.ai.state.route = route;
        car.ai.state.route_index = 0;
        car.ai.state.lane = lane;
        car.ai.state.s = s;
        car.ai.state.change = None;
        car.ai.state.planned_next = None;
        car.ai.state.ahead.clear();
        car.ai.state.plan_next(net);
        car.released = false;
        self.note_vehicle_changed(ci);
        let car = &mut self.cars[ci];
        let stops = stops.into_iter().map(crate::bus_service::Stop::from_tuple).collect();
        match car.bus.as_mut() {
            Some(b) => b.restart(stops, layover),
            None => {
                let mut b = BusService::new(stops);
                b.layover = layover;
                car.bus = Some(Box::new(b));
            }
        }
    }

    /// Let timetable bus `ci` go at the end of its trip: it drives on as other traffic and
    /// is taken off as soon as nobody sees it.
    pub fn release(&mut self, ci: usize) {
        let net = &self.net;
        let car = &mut self.cars[ci];
        let st = &mut car.ai.state;
        st.route.clear();
        st.route_index = 0;
        st.planned_next = None;
        st.ahead.clear();
        st.plan_next(net);
        if let Some(b) = car.bus.as_mut() {
            b.restart(Vec::new(), false);
        }
        car.released = true;
        self.note_vehicle_changed(ci);
    }

    /// Take all random AI cars off the road now, keeping timetable buses. Returns how many
    /// vehicles were removed. The configured target is unchanged, so random traffic can
    /// populate the roads again normally.
    pub fn clear_random(&mut self, world: &World, renderer: &Renderer, scene: &mut Scene) -> usize {
        let ids: Vec<u64> = self.cars.iter().filter(|c| !c.serves_stops()).map(|c| c.id).collect();
        let removed = ids.len();
        for id in ids {
            self.remove_car(world, renderer, scene, id);
        }
        removed
    }

    /// The AI on the roads: (cars, buses, cars asleep far from everybody, parked cars).
    pub fn counts(&self) -> (usize, usize, usize, usize) {
        let buses = self.cars.iter().filter(|c| c.serves_stops()).count();
        (self.cars.len() - buses, buses, self.dormant.len(), self.parked.values().map(Vec::len).sum())
    }

    /// Take a car off the road now (the player took over its tour).
    pub fn remove_car(
        &mut self,
        world: &World,
        renderer: &Renderer,
        scene: &mut Scene,
        id: u64,
    ) -> bool {
        let Some(i) = self.cars.iter().position(|c| c.id == id) else {
            return false;
        };
        let c = self.cars.swap_remove(i);
        {
            let p = c.vehicle.position;
            let r = (c.ai.state.length as f64 * 0.5).max(2.0);
            self.recorder.note_remove_unnamed(c.id, RemoveRule::Other, self.viewer.is_some_and(|v| v.frames(p, r)), c.ai.waiting, self.viewer.map(|v| (p - v.pos).length() as f32));
        }
        if c.serves_stops() {
            self.riders_out.push(c.id);
        }
        self.orphan_sounds.extend(c.sounds);
        for r in std::iter::once(c.render).chain(c.trailer_renders) {
            world.release_vehicle(renderer, scene, r);
        }
        true
    }

    /// Obstacle boxes of all AI vehicles (for the player's collisions), with the rear
    /// sections of articulated buses and the trailers.
    pub fn boxes(&self, near: DVec3, radius: f64) -> Vec<omsi_sim::collision::Obb> {
        self.cars
            .iter()
            .filter(|c| (c.vehicle.position - near).length() < radius)
            .flat_map(|c| {
                let bb = c
                    .vehicle
                    .ty
                    .def
                    .bounding_box
                    .unwrap_or([2.0, 4.5, 1.6, 0.0, 0.0, 0.8]);
                // moving, and with a mass of its own: a car that runs into the bus is no
                // bulldozer
                let h = c.vehicle.heading.to_radians();
                let v = glam::DVec2::new(h.sin(), h.cos()) * c.ai.state.speed as f64;
                let (mass, id) = (c.vehicle.physics.mass_kg, c.id);
                let rear = c.vehicle.trailers.iter().filter_map(move |t| {
                    t.ty.def.bounding_box.map(|bb| {
                        omsi_sim::collision::Obb::from_box(bb, t.position, t.body_heading())
                            .moving(v, mass, id)
                    })
                });
                std::iter::once(
                    omsi_sim::collision::Obb::from_box(bb, c.vehicle.position, c.vehicle.body_heading())
                        .moving(v, mass, id),
                )
                .chain(rear)
            })
            .collect()
    }

    /// Position and heading of a car by id (None once it is gone).
    pub fn car_pose(&self, id: u64) -> Option<(DVec3, f64)> {
        self.cars
            .iter()
            .find(|c| c.id == id)
            .map(|c| (c.vehicle.position, c.vehicle.heading))
    }

    /// State of light `li` of controller `c` now and the seconds it keeps showing it (the
    /// pedestrians only start across on a green that lasts).
    pub fn light_state(&self, c: usize, li: usize) -> Option<(i32, f32)> {
        let ctl = self.sim.lights.get(c)?;
        Some((ctl.state(li), ctl.remaining(li)))
    }

    /// Seconds until light `li` of controller `c` lets traffic go (0 while it does).
    pub fn light_until_go(&self, c: usize, li: usize) -> Option<f32> {
        self.sim.lights.get(c)?.time_until_go(li)
    }

    /// The light that governs footpath lane `lane`: its own, else the light of the
    /// pedestrian crossing it is a piece of (the unlit pieces over the lanes beyond the lit
    /// stub at the kerb, BeijingCity).
    pub fn crossing_light(&self, lane: usize) -> Option<(usize, usize)> {
        self.sim.crossing_light(&self.net, lane)
    }

    /// Tell a scheduled bus's script who wants in or out (`PAX_Entry<i>_Req`,
    /// `PAX_Exit<i>_Req`) and who stands in its doorways (`_Busy`): the stock AI door
    /// scripts open the rear doors only for a stop request, which comes from the exit
    /// requests.
    pub fn set_pax_requests(&mut self, id: u64, doors: &crate::humans::DoorWants) {
        if let Some(c) = self.cars.iter_mut().find(|c| c.id == id) {
            crate::humans::Humans::write_door_requests(&mut c.vehicle, doors);
        }
    }

    /// Keep a scheduled bus at its stop for at least `secs` more with the doors open:
    /// passengers are still queueing at a door or stepping in.
    /// The passengers' wishes for the timetable buses' next stops (see `stop_wishes`).
    pub fn set_stop_wishes(&mut self, alighting: hashbrown::HashSet<u64>, waiting: hashbrown::HashSet<i64>) {
        self.stop_wishes = Some((alighting, waiting));
    }

    pub fn hold_boarding(&mut self, id: u64, stop: Option<i64>, secs: f32) {
        if let Some(c) = self.cars.iter_mut().find(|c| c.id == id) {
            if let Some(b) = c.bus.as_mut() {
                b.hold(stop, secs);
            }
        }
    }

    /// The street lanes a vehicle standing at `pos` facing `heading` is on and will drive
    /// onto within `reach` metres, each with the distance from the vehicle to its start
    /// (0 for the lane it is on): the way straight on and the gentle turns (within 60° of
    /// the lane before), not every branch of a junction.
    fn sync_drivers(&mut self, world: &World, renderer: &Renderer, scene: &mut Scene) {
        let Some(eye) = self.viewer.map(|v| v.pos) else { return };
        let dt = self.last_dt.max(1.0 / 120.0);
        let mut keep: Vec<u64> = Vec::new();
        for c in &self.cars {
            if !c.serves_stops() || c.released {
                continue;
            }
            let d = (c.vehicle.position - eye).length();
            if d > DRIVER_NEAR * 2.0 {
                continue;
            }
            keep.push(c.id);
            if !self.drivers.contains_key(&c.id) {
                if d > DRIVER_NEAR {
                    continue;
                }
                let figure = match self.driver_pool.pop() {
                    Some(mut f) => {
                        if f.attach(&c.vehicle) {
                            Some(f)
                        } else {
                            self.driver_pool.push(f);
                            None
                        }
                    }
                    None => crate::driver::DriverFigure::new(world, renderer, scene, &c.vehicle, c.id),
                };
                match figure {
                    Some(f) => {
                        self.drivers.insert(c.id, f);
                    }
                    None => continue,
                }
            }
            if let Some(f) = self.drivers.get_mut(&c.id) {
                f.update(renderer, scene, &c.vehicle, &c.render, dt, true, false);
            }
        }
        let gone: Vec<u64> = self.drivers.keys().copied().filter(|id| !keep.contains(id)).collect();
        for id in gone {
            if let Some(mut f) = self.drivers.remove(&id) {
                f.hide(renderer, scene);
                self.driver_pool.push(f);
            }
        }
    }

    /// The `TrafficLightPhase` and `TrafficLightApproach` values of light `li` of
    /// controller `c` (for scenery scripts).
    pub fn light_vars(&self, c: usize, li: usize) -> (f32, f32) {
        self.sim.lights
            .get(c)
            .map(|ctl| {
                (
                    ctl.state(li) as f32,
                    ctl.request.get(li).copied().unwrap_or(false) as i32 as f32,
                )
            })
            .unwrap_or((omsi_sim::traffic::UNLINKED_PHASE as f32, 0.0))
    }

    pub fn sync(&mut self, world: &World, renderer: &Renderer, scene: &mut Scene) {
        // cars that have parked: the parked object stands in their place from now on
        let mut i = 0;
        while i < self.cars.len() {
            match self.cars[i].park {
                Some(p) if p.done => {
                    let c = self.cars.swap_remove(i);
                    let from_eye = self.viewer.map(|v| (c.vehicle.position - v.pos).length() as f32);
                    self.recorder.note_remove_from(c.id, RemoveRule::Parked, false, c.ai.waiting, from_eye);
                    if world.return_parked(renderer, scene, p.key) {
                        let st = &c.ai.state;
                        let spot = Parked { s: p.s, lateral: p.lat, half_width: c.ai.half_width, half_len: (st.front + st.rear) * 0.5 };
                        self.parked.entry(p.lane).or_default().push(spot);
                    }
                    if omsi_cfg::env::var_os("OMSI_DEBUG_TRAFFIC").is_some() {
                        log::info!("car {} has parked (space {})", c.id, p.key);
                    }
                    self.orphan_sounds.extend(c.sounds);
                    self.released.push(c.render);
                    self.released.extend(c.trailer_renders);
                }
                _ => i += 1,
            }
        }
        for r in std::mem::take(&mut self.released) {
            world.release_vehicle(renderer, scene, r);
        }
        self.sync_drivers(world, renderer, scene);
        // traffic light lamps: the lamp's script (or the stock rules) turns the state of its
        // light into the `[visible] red|yellow|green 1` meshes and the coronas, and moves
        // what it animates (a barrier arm); it runs on the time since the last sync (an
        // offscreen run syncs only for its pictures)
        let dt = std::mem::take(&mut self.lamp_dt);
        let near = self.viewer.map(|v| v.pos);
        let debug_lamps = omsi_cfg::env::var_os("OMSI_DEBUG_LAMPS").is_some();
        for lamp in world.light_objects.lock().iter_mut() {
            if debug_lamps && lamp.animated {
                log::info!("moving lamp at ({:.1}, {:.1}), {:.0} m from the viewer", lamp.pos.x, lamp.pos.y, near.map(|p| (lamp.pos - p).length()).unwrap_or(0.0));
            }
            if let Some(p) = near {
                if (lamp.pos - p).length() > 1200.0 {
                    continue;
                }
            }
            let (state, request) = match self
                .controller_of_object
                .get(&lamp.parent)
                .and_then(|&c| self.sim.lights.get(c))
            {
                Some(ctl) if lamp.any_light && ctl.lights.len() > 1 => {
                    // the most open of the crossing's lights (see `LightObject::any_light`)
                    let open = |s: i32| match s {
                        6..=8 => 3,
                        3..=5 => 2,
                        9..=11 => 1,
                        0..=2 => 0,
                        _ => -1,
                    };
                    let li = (0..ctl.lights.len())
                        .max_by_key(|&i| open(ctl.state(i)))
                        .unwrap_or(0);
                    (ctl.state(li), ctl.request.iter().any(|r| *r))
                }
                Some(ctl) => (
                    ctl.state(lamp.index),
                    ctl.request.get(lamp.index).copied().unwrap_or(false),
                ),
                // a lamp that names no crossing, or one whose crossing has no program,
                // reads the engine's dummy (see `UNLINKED_PHASE`): red, as in OMSI
                None => (omsi_sim::traffic::UNLINKED_PHASE, false),
            };
            let (r, y, g) = TrafficLightController::lamps(state);
            let value = |lamp: &crate::scene::LightObject, var: &str| -> f32 {
                // Custom signals can shift phases or blink the standard channels (Numazu
                // pedestrian lamps). Use their script outputs whenever they are available.
                let scripted = lamp.script.as_ref().and_then(|script| {
                    let s = script.lock();
                    // A failed/missing script can still have a varlist of zeroes. Keep
                    // stock fallback behaviour if it has no runnable frame block.
                    if s.program.frame.is_empty() { None } else { s.var(var) }
                });
                crate::scene::traffic_lamp_value(
                    var,
                    scripted,
                    crate::scene::standard_traffic_lamp(var, r, y, g, request),
                )
            };
            if let Some(script) = lamp.script.as_ref() {
                let vars = omsi_sim::scenery::SceneryVars {
                    nightlight: self.night as i32 as f32,
                    in_use: 1.0,
                    traffic_light_phase: state as f32,
                    traffic_light_approach: request as i32 as f32,
                    switch: None,
                };
                let mut s = script.lock();
                s.update(dt, &vars);
                // `OMSI_DEBUG_LAMPS`: where the moving lamps are (barriers) and how far their
                // meshes are turned, each time the lamps are updated
                if lamp.animated && debug_lamps {
                    let turn = s
                        .mesh_transforms
                        .iter()
                        .map(|m| {
                            let (_, r, _) = m.to_scale_rotation_translation();
                            r.to_axis_angle().1.to_degrees()
                        })
                        .fold(0.0f32, f32::max);
                    log::info!("lamp at ({:.1}, {:.1}): light state {:?} (crossing {:?}, light {}{}), meshes turned up to {turn:.0} deg", lamp.pos.x, lamp.pos.y, vars.traffic_light_phase, self.controller_of_object.get(&lamp.parent), lamp.index, if lamp.any_light { ", any" } else { "" });
                }
                if lamp.animated {
                    for (i, (inst, _)) in lamp.instances.iter().enumerate() {
                        if let Some(m) = s.mesh_transforms.get(i) {
                            renderer.set_transform(scene, *inst, lamp.pos, lamp.xf * *m);
                        }
                    }
                    // the lights go with their meshes (a barrier's lamps rise with its arm)
                    for (c, (mi, local, dir)) in lamp.coronas.iter_mut().zip(&lamp.corona_mesh) {
                        if let Some(m) = s.mesh_transforms.get(*mi) {
                            let xf = lamp.xf * *m;
                            c.0.position = lamp.pos + xf.transform_point3(*local).as_dvec3();
                            if *dir != glam::Vec3::ZERO {
                                c.0.direction = xf.transform_vector3(*dir).normalize_or_zero();
                            }
                        }
                    }
                }
            }
            if !lamp.animated {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                (state, request).hash(&mut h);
                if let Some(script) = lamp.script.as_ref() {
                    for v in &script.lock().state.vars {
                        v.to_bits().hash(&mut h);
                    }
                }
                let sig = h.finish();
                if lamp.shown == Some(sig) {
                    continue;
                }
                lamp.shown = Some(sig);
            }
            // Traffic lamps do not enter World's ordinary scripted-object update path.
            // Switch their materials here too, so [matl_item] nightmaps light the LEDs.
            for (inst, slot, base, item, var) in &lamp.variants {
                renderer.set_material(
                    scene,
                    *inst,
                    *slot,
                    if crate::scene::change_picks_item(value(lamp, var)) { *item } else { *base },
                );
            }
            for k in 0..lamp.coronas.len() {
                let v = value(lamp, &lamp.coronas[k].1);
                lamp.lit[k] = v;
            }
            for (k, (inst, cond)) in lamp.instances.iter().enumerate() {
                let visible = match cond {
                    Some((var, want)) => (value(lamp, var) - want).abs() < 0.5,
                    None => true,
                };
                // lenses switched by their material instead (`[alphascale]` and
                // `[matl_lightmap]` on the lamp's variables, #826)
                match lamp.slots.get(k).filter(|s| !s.is_empty()) {
                    Some(slots) => {
                        let known = |v: &str| -> Option<f32> {
                            let scripted = lamp.script.as_ref().and_then(|script| {
                                let s = script.lock();
                                if s.program.frame.is_empty() { None } else { s.var(v) }
                            });
                            scripted
                                .or_else(|| v.trim().parse::<f32>().ok())
                                .or_else(|| crate::scene::standard_traffic_lamp(v, r, y, g, request))
                        };
                        let (alpha, light) = slots.values(&known);
                        renderer.set_params(scene, *inst, &alpha, visible, &[]);
                        renderer.set_slot_light(scene, *inst, &light);
                    }
                    None => renderer.set_params(scene, *inst, &[], visible, &[]),
                }
            }
        }
        // A far car's script textures (its destination sign) stay as they are drawn: OMSI
        // shows them at any distance its model level has them. (They were stood in for by
        // their mean colour beyond 50 m, and every timetable bus coming up the street had
        // a blank sign until it was almost there.) What a far car's scripts redraw goes to
        // the GPU at most every half second, a slice of the cars per frame.
        let tick = (self.time as f64 * 2.0) as u64;
        let mut budget = SCRIPT_UPLOAD_BUDGET;
        // `Envir_Brightness`, which Omsi.exe sets for every road vehicle as for the
        // player's: the stock buses fade their windows by it at night (left at the engine's
        // default of 1, an AI bus under the street lamps kept its daytime brown glass)
        if let Some(d) = self.daylight {
            for c in self.cars.iter_mut().filter(|c| c.vehicle.ai_visuals) {
                let b = d.envir_brightness(world.light_map_light_at(c.vehicle.position));
                c.vehicle.set_var("Envir_Brightness", b);
            }
        }
        for c in &mut self.cars {
            // out of sight (`tick` decided): hidden once, then left alone until it comes
            // into view again - its many per-mesh updates were a third of this stage
            if !c.vehicle.ai_visuals {
                if !c.render.hidden {
                    c.render.hidden = true;
                    for inst in c
                        .render
                        .instances
                        .iter()
                        .chain(c.trailer_renders.iter().flat_map(|r| r.instances.iter()))
                    {
                        renderer.set_params(scene, *inst, &[], false, &[]);
                    }
                }
                continue;
            }
            c.render.hidden = false;
            if let Some(cam) = self.camera {
                let far = (c.vehicle.position - cam).length() > crate::scene::DISPLAYS_FAR;
                let due = c.render.display_tick != tick;
                c.render.displays_far = far && !due;
                if far && due {
                    c.render.display_tick = tick;
                }
            }
            crate::scene::sync_vehicle_textures(renderer, scene, &mut c.vehicle, &c.render, &mut budget);
            crate::scene::sync_vehicle_materials(renderer, scene, &c.vehicle, &mut c.render);
            // a coupled part runs no scripts of its own: its plates, its displays and its
            // switched materials follow the leading vehicle's, as the player's own rear
            // sections do (without this an AI bus's rear section kept the blank textures and
            // the unswitched materials it was built with)
            {
                let mut trailers = std::mem::take(&mut c.vehicle.trailers);
                for (t, r) in trailers.iter_mut().zip(c.trailer_renders.iter_mut()) {
                    crate::scene::sync_vehicle_part(renderer, scene, &c.vehicle, t, r);
                }
                c.vehicle.trailers = trailers;
            }
            // an articulated AI bus (timetable or random traffic) bends its bellows like the
            // player's while it is near enough for the fold to show; farther out its shape
            // just stays as it was, which nobody can tell from still following the road
            if !c.render.skinned.is_empty()
                || c.trailer_renders.iter().any(|r| !r.skinned.is_empty())
            {
                let near = self
                    .camera
                    .map(|cam| (c.vehicle.position - cam).length() < SKIN_DISTANCE)
                    .unwrap_or(true);
                if near {
                    crate::scene::sync_skinned(
                        renderer,
                        scene,
                        &mut c.vehicle,
                        &mut c.render,
                        &mut c.trailer_renders,
                    );
                }
            }
            for (i, inst) in c.render.instances.iter().enumerate() {
                renderer.set_transform(
                    scene,
                    *inst,
                    c.vehicle.position,
                    c.vehicle.mesh_local_transform(i),
                );
                let p = &c.vehicle.mesh_props[i];
                let def = &c.vehicle.ty.model.meshes[c.vehicle.ty.meshes[i].def_index];
                let vp = def.viewpoint;
                let vp_ok = vp == 0 || vp & 4 != 0;
                // AI vehicles do not run every cockpit/material script that the player
                // vehicle runs.  Some models consequently leave an `[alphascale]`
                // variable at zero; applying it to an opaque body makes the traffic bus
                // translucent and reveals the interior through its panels.  Opaque slots
                // are never allowed to be faded by a dynamic alpha value; windows and
                // explicitly alpha-tested/blended slots retain their authored behavior.
                let mut alpha = p.slot_alpha.clone();
                for (slot, mat) in scene.instances[*inst].materials.iter().enumerate() {
                    if scene
                        .materials
                        .get(*mat)
                        .is_some_and(|m| m.alpha == omsi_render::AlphaMode::Opaque)
                    {
                        if let Some(a) = alpha.get_mut(slot) {
                            *a = 1.0;
                        }
                    }
                }
                renderer.set_params(scene, *inst, &alpha, p.visible && vp_ok, &p.slot_uv);
                renderer.set_slot_light(scene, *inst, &p.slot_light);
                renderer.set_slot_night(scene, *inst, &p.slot_night);
                renderer.set_interior(scene, *inst, p.interior);
            }
            for (t, r) in c.vehicle.trailers.iter().zip(&c.trailer_renders) {
                for (i, inst) in r.instances.iter().enumerate() {
                    renderer.set_transform(scene, *inst, t.position, t.mesh_local_transform(i));
                    let p = &t.mesh_props[i];
                    let def = &t.ty.model.meshes[t.ty.meshes[i].def_index];
                    let vp = def.viewpoint;
                    let vp_ok = vp == 0 || vp & 4 != 0;
                    let mut alpha = p.slot_alpha.clone();
                    for (slot, mat) in scene.instances[*inst].materials.iter().enumerate() {
                        if scene
                            .materials
                            .get(*mat)
                            .is_some_and(|m| m.alpha == omsi_render::AlphaMode::Opaque)
                        {
                            if let Some(a) = alpha.get_mut(slot) {
                                *a = 1.0;
                            }
                        }
                    }
                    renderer.set_params(scene, *inst, &alpha, p.visible && vp_ok, &p.slot_uv);
                    renderer.set_slot_light(scene, *inst, &p.slot_light);
                    renderer.set_slot_night(scene, *inst, &p.slot_night);
                    renderer.set_interior(scene, *inst, p.interior);
                }
            }
        }
    }
}

/// The prefetch reach (m): how far round the player the tiles must be loaded so that the loaded
/// tiles' edge still lies beyond the perceptual `horizon` when the next tile arrives - the horizon
/// plus the distance covered during one tile load (`load_secs`) at `road_speed` (m/s). The edge
/// is where the traffic network ends without the map ending; seen, it is a dead end on an open
/// road (OMSI geometry: the edge is not a place; a SUMO network's border is where its roads end).
pub fn prefetch_target(horizon: f64, road_speed: f64, load_secs: f64) -> f64 {
    horizon + road_speed.max(0.0) * load_secs.max(0.0)
}

/// [round 10 S4] Is a loaded-edge lane ending at `end` in sight of the eye at `eye`: within the
/// perceptual `horizon` (m, straight from the eye, as the horizon is reckoned).
fn edge_in_sight(end: DVec3, eye: DVec3, horizon: f64) -> bool {
    (end - eye).length() <= horizon
}

/// [round 9 A4 fix pass 2] The lanes whose every way on ends at a loaded edge in sight within
/// EDGE_LOOKAHEAD (`Network::ends_at_loaded_edge`, gathered back from the edges once rather than
/// walked forward for every spawn place): no car is put down or woken there.
fn edge_bound_of(net: &Network) -> hashbrown::HashSet<usize> {
    let mut bound: hashbrown::HashMap<usize, f64> = net.lanes.iter().enumerate().filter(|(_, l)| l.loaded_edge && l.edge_in_sight && l.next.is_empty()).map(|(i, _)| (i, 0.0)).collect();
    let mut work: Vec<usize> = bound.keys().copied().collect();
    while let Some(m) = work.pop() {
        for &p in net.prev.get(m).map(|v| v.as_slice()).unwrap_or(&[]) {
            if bound.contains_key(&p) {
                continue;
            }
            let lp = &net.lanes[p];
            let far = lp.next.iter().map(|n| bound.get(n).map(|d| d + net.lanes[*n].length() as f64)).collect::<Option<Vec<f64>>>();
            if let Some(d) = far.and_then(|v| v.into_iter().reduce(f64::max)).filter(|d| *d <= omsi_sim::traffic::EDGE_LOOKAHEAD) {
                bound.insert(p, d);
                work.push(p);
            }
        }
    }
    bound.into_keys().collect()
}

/// The horizon `prefetch_target` keeps the edge beyond: a body `height` high with its lamps dark
/// (a lit lamp has no two-pixel distance), capped by the draw distance as every horizon is (so
/// the reach is at most the renderer's object distance plus one tile load; with `max_obj_dist`
/// auto, the view distance), never within `NEAR_HIDE` (as
/// `Viewer::beyond_horizon`: the far trip end acts beyond both).
/// A stated trade, flagged for the user: the reach uses the dark horizon by dusk and night too.
/// With the lamps lit an edge is in sight out to the renderer's reach (1.05 x the object
/// distance: ~3150 m with the user's 3000 m), so beyond the view distance (2500 m) A4 keeps the
/// roads to it empty at night; loading out to the lit reach instead (~3.25 km) would cost ~1.7x
/// the tiles (coordinator decision, round 10: memory over the night picture at 2.5-3 km).
fn edge_horizon(viewer: Viewer, height: f64) -> f64 {
    Viewer { lamps_lit: false, ..viewer }.horizon(height).max(NEAR_HIDE)
}

impl Traffic {
    /// `prefetch_target` for this frame's picture (`viewer`), `None` before the first frame:
    /// the `edge_horizon` of the tallest random-traffic body (lamps dark: with them lit the edge
    /// is in sight out to the renderer's reach and A4 keeps the cars off roads ending there), and
    /// the road's speed at the player (`player`:
    /// position, speed in m/s) - the faster of its street lane's speed limit and its own speed.
    pub fn prefetch_reach(&self, player: Option<(DVec3, f64)>, load_secs: f64) -> Option<f64> {
        let viewer = self.viewer?;
        let height = self.types.iter().filter(|t| t.2 == LaneKind::Street).map(|t| body_height(&t.0)).fold(CAR_HEIGHT, f64::max);
        let road_speed = player.map_or(0.0, |(pos, speed)| {
            let limit = self.net.nearest_lane_near(pos, LaneKind::Street).map_or(0.0, |(l, _, _)| self.net.lanes[l].speed_limit_kmh as f64 / 3.6);
            limit.max(speed.abs())
        });
        Some(prefetch_target(edge_horizon(viewer, height), road_speed, load_secs))
    }

    /// [round 8 H8-7, fix pass 1] Mark the street lanes that end at the edge of the loaded tiles
    /// (`Lane::loaded_edge`): no lane on, and the road's way on - `LANE_LINK_TOL` past the end,
    /// where a lane going on would be linked - lies on a tile of the map whose lanes are not
    /// loaded. The tiles reach at least `prefetch_reach` round the player, so the edge lies
    /// beyond the perceptual horizon; a fixed `--radius` square keeps its edge where it is
    /// (Jiangzhaikou with `--radius 2`: lanes 149-155 at y = -2455, 873 m ahead).
    pub fn mark_loaded_edges(&mut self, world: &World) {
        let ts = omsi_map::tile_size();
        let before: Vec<(bool, bool)> = self.net.lanes.iter().map(|l| (l.loaded_edge, l.edge_in_sight)).collect();
        for l in self.net.lanes.iter_mut() {
            l.loaded_edge = l.kind == LaneKind::Street && l.next.is_empty() && !l.points.is_empty() && {
                let (end, h) = (l.end(), (l.end_heading() as f64).to_radians());
                let probe = end + DVec3::new(h.sin(), h.cos(), 0.0) * LANE_LINK_TOL;
                loaded_edge_at(probe, ts, &|k| world.has_tile(k), &self.lane_tiles)
            };
        }
        self.edge_lanes = self.net.lanes.iter().enumerate().filter(|(_, l)| l.loaded_edge).map(|(i, _)| i).collect();
        self.set_edges_in_sight();
        self.edge_bound = edge_bound_of(&self.net);
        // (recorded, so that a replay has the flags as they were from here on)
        let changed: Vec<(usize, bool, bool)> = self.net.lanes.iter().enumerate().filter(|(i, l)| before[*i] != (l.loaded_edge, l.edge_in_sight)).map(|(i, l)| (i, l.loaded_edge, l.edge_in_sight)).collect();
        self.recorder.lane_flags(self.sim.time, &changed);
    }

    /// [round 10 S4] Mark each loaded-edge lane in sight or not (`Lane::edge_in_sight`): its end
    /// within the perceptual horizon of the eye (`Viewer::horizon` of the tallest random-traffic
    /// body, never within NEAR_HIDE; only the draw distance while the lamps are lit, so then
    /// every edge within it is in sight), and with the flags changed gather `edge_bound` again. Only an edge in sight is
    /// routed away from and kept clear of new cars (A4); beyond the horizon a car drives to it
    /// and its trip ends there out of sight (the far trip end, A2), as a SUMO vehicle leaves the
    /// network at its border. Before the first frame (no eye) every edge counts as in sight.
    /// Changes are recorded (`Record::LaneFlags`), so a replay routes with the flags as they were.
    pub fn mark_edges_in_sight(&mut self) {
        let changed = self.set_edges_in_sight();
        if !changed.is_empty() {
            self.edge_bound = edge_bound_of(&self.net);
            let flags: Vec<(usize, bool, bool)> = changed.iter().map(|&i| (i, self.net.lanes[i].loaded_edge, self.net.lanes[i].edge_in_sight)).collect();
            self.recorder.lane_flags(self.sim.time, &flags);
        }
    }

    /// `Lane::edge_in_sight` of the edge lanes for this frame's eye; returns the lanes changed.
    fn set_edges_in_sight(&mut self) -> Vec<usize> {
        let horizon = self.viewer.map(|v| {
            let height = self.types.iter().filter(|t| t.2 == LaneKind::Street).map(|t| body_height(&t.0)).fold(CAR_HEIGHT, f64::max);
            (v.pos, v.horizon(height).max(NEAR_HIDE))
        });
        let mut changed = Vec::new();
        for &i in &self.edge_lanes {
            let l = &mut self.net.lanes[i];
            let seen = horizon.is_none_or(|(eye, h)| edge_in_sight(l.end(), eye, h));
            if l.edge_in_sight != seen {
                changed.push(i);
            }
            l.edge_in_sight = seen;
        }
        changed
    }

    /// Take in what the tiles loaded since the last call brought: their lanes (linked into
    /// the network, whose existing indices stay valid), their parked cars (sorted onto the
    /// lanes once those are in) and the light programs of their crossings.
    pub fn add_tiles(&mut self, world: &World) -> usize {
        let (new, parked_cars, tiles) = take_from_tiles(world);
        let n = new.len();
        let mut added = self.net.lanes.len()..self.net.lanes.len();
        if n > 0 {
            if self.recorder.is_recording() {
                self.recorder.lanes_added(self.sim.time, self.net.lanes.len(), &new);
            }
            added = self.net.extend(new, LANE_LINK_TOL);
            self.sim.network_grew(&self.net, added.start);
            self.street_weight += self.net.lanes[added.clone()].iter().filter_map(street_lane_weight).sum::<f64>();
            log::debug!(
                "traffic: {} lanes added ({} in all)",
                added.len(),
                self.net.lanes.len()
            );
        }
        if !tiles.is_empty() {
            self.lane_tiles.extend(tiles);
            self.lanes_generation += 1;
        }
        if n > 0 {
            self.mark_loaded_edges(world);
        }
        self.sort_parked(parked_cars, added);
        // new crossings bring their light programs; the running ones keep their clocks
        // (a program is never taken away again: the world's list only grows)
        let lights = world.traffic_lights.lock();
        if lights.len() > self.sim.lights.len() {
            let from = self.sim.lights.len();
            self.recorder.lights_added(self.sim.time, from, &lights[from..]);
            self.sim.lights.extend(lights[from..].iter().cloned());
            self.light_prev
                .extend(lights[from..].iter().map(|c| vec![-100; c.lights.len()]));
            self.controller_of_object = world.controller_of_object.lock().clone();
        }
        drop(lights);
        if n > 0 {
            self.dump_network();
        }
        n
    }
}

/// What the tiles placed since the last call hand to the traffic: their lanes, their parked
/// cars and which tiles they were (taken together, see `World::lane_tiles`).
fn take_from_tiles(
    world: &World,
) -> (
    Vec<omsi_sim::traffic::Lane>,
    Vec<ParkedCar>,
    Vec<(i32, i32)>,
) {
    let mut lanes = world.lanes.lock();
    let parked = std::mem::take(&mut *world.parked_cars.lock());
    let tiles = std::mem::take(&mut *world.lane_tiles.lock());
    let mut new = std::mem::take(&mut *lanes);
    // The tiles are read in parallel and hand in their lanes in the order they finish:
    // sorted by their map identity, the lanes a set of tiles brings are numbered alike in
    // every run, and so is the random traffic drawn from them (a run can be repeated to
    // look at what a car did).
    new.sort_by(|a, b| {
        let first = |l: &omsi_sim::traffic::Lane| {
            l.points
                .first()
                .map(|p| (p.x.to_bits(), p.y.to_bits()))
                .unwrap_or((0, 0))
        };
        (
            a.key.map(|k| (k.tile, k.id, k.path)),
            a.reversed,
            a.source,
            first(a),
        )
            .cmp(&(
                b.key.map(|k| (k.tile, k.id, k.path)),
                b.reversed,
                b.source,
                first(b),
            ))
    });
    (new, parked, tiles)
}

// ---------------------------------------------------------------------------------------
// LAN play (see `lan_world`): a host keeps traffic around every player and tells the
// clients its light programs; a client draws the host's cars instead of its own.

impl Traffic {
    /// Street traffic around the other players of a LAN session too (host): each of them
    /// gets its own share of cars where no other player's share lies already.
    fn populate_lan_centers(
        &mut self,
        world: &World,
        renderer: &Renderer,
        scene: &mut Scene,
        center: DVec3,
        target: usize,
    ) {
        let centers = self.lan_centers.clone();
        let mut done = vec![center];
        for c in centers {
            if done
                .iter()
                .any(|d| (*d - c).truncate().length() < self.spawn_radius)
            {
                continue;
            }
            self.populate_kind(world, renderer, scene, c, LaneKind::Street, target);
            done.push(c);
        }
    }

    pub fn is_mirror(&self) -> bool {
        self.mirror
    }

    /// Draw the host's traffic from now on (`on`), or simulate our own again: either way
    /// every car there is now goes (ours make room for the host's, the host's copies
    /// cannot drive on by themselves).
    pub fn set_mirror(&mut self, world: &World, renderer: &Renderer, scene: &mut Scene, on: bool) {
        if self.mirror == on {
            return;
        }
        self.mirror = on;
        if !on {
            let day_time = self.day_time;
            for ctl in &mut self.sim.lights {
                reset_light_runtime(ctl, day_time);
            }
        }
        let ids: Vec<u64> = self.cars.iter().map(|c| c.id).collect();
        for id in ids {
            self.remove_car(world, renderer, scene, id);
        }
        self.initial = !on;
        log::info!(
            "traffic: {}",
            if on {
                "the LAN host's traffic is drawn instead of our own"
            } else {
                "simulating our own traffic again"
            }
        );
    }

    /// A client's frame: interpolate the host's light clocks, without re-evaluating its
    /// stop and jump points from the client's incomplete traffic requests.
    fn mirror_tick(&mut self, dt: f32) {
        self.time += dt;
        self.day_time += dt as f64 * self.time_scale;
        self.last_dt = dt;
        let day_time = self.day_time;
        for c in self.sim.lights.iter_mut() {
            mirror_light_tick(c, dt, day_time);
        }
        self.log_lights();
    }

    /// A car of the host's traffic, standing at `pos` (client). Its id is the host's.
    #[allow(clippy::too_many_arguments)]
    pub fn add_mirror_car(
        &mut self,
        world: &World,
        renderer: &Renderer,
        scene: &mut Scene,
        id: u64,
        ty: Arc<VehicleType>,
        scheme: Option<usize>,
        scheduled: bool,
        pos: DVec3,
        heading: f64,
    ) -> usize {
        let mut host = omsi_sim::VehicleHost::new(omsi_sim::SimClock::default());
        host.font_lib = Some(world.fonts.clone());
        let scheme = scheme.filter(|i| *i < ty.paint_schemes.len());
        host.paint_scheme = Some(scheme);
        let mut vehicle = VehicleInstance::new(ty.clone(), host);
        // (the host's poses say where it stands; nothing here pulls it onto the ground)
        vehicle.ground = None;
        vehicle.apply_paint_vars(scheme);
        let render = world.add_vehicle_shared(renderer, scene, &ty, scheme, None);
        let trailer_renders =
            self.attach_trailers(world, renderer, scene, &mut vehicle, scheme, &render);
        if !ty.model.text_textures.is_empty() {
            vehicle.init_text_textures(&mut world.fonts.lock(), &|p| {
                omsi_texture::decode_file(p)
                    .ok()
                    .map(|i| (i.width, i.height, i.rgba))
            });
        }
        vehicle.position = pos;
        vehicle.heading = heading;
        let (front, rear, half_width) = extents(&ty, 4.5);
        let rear = self.consist_rear(&ty).max(rear);
        let mut state = AiState::new(0, 0.0, id);
        state.front = front;
        state.rear = rear;
        state.length = front + rear;
        let body = AiBody::new(&ty.def, MotionKind::Road);
        let mut ai = AiVehicle::new(id, if scheduled { VehicleClass::ScheduledBus } else { VehicleClass::Car }, state, half_width, id);
        ai.fresh = 0.0;
        self.cars.push(AiCar {
            id,
            ai,
            vehicle,
            render,
            trailer_renders,
            body,
            bus: scheduled.then(|| Box::new(BusService::new(Vec::new()))),
            sounds: None,
            seed: 0,
            scheme: None,
            rail_trail: Default::default(),
            ai_secs: 0.0,
            consist_reversed: false,
            park: None,
            released: false,
        });
        self.cars.len() - 1
    }

    /// A vehicle type the traffic has loaded already (the random traffic's types and the
    /// coupled parts), by file.
    pub fn loaded_type(&self, path: &Path) -> Option<Arc<VehicleType>> {
        self.types
            .iter()
            .map(|t| &t.0)
            .chain(self.trailer_types.values().flatten())
            .find(|t| t.def.path == path)
            .cloned()
    }

    /// The light programs of the crossings within `radius` of `near` (host): (crossing
    /// object, position in the cycle, clock held at a stop point).
    pub fn light_states(&self, near: DVec3, radius: f64) -> Vec<(i64, f64, bool)> {
        let mut ctls: Vec<usize> = self
            .net
            .lanes
            .iter()
            .filter(|l| {
                l.traffic_light.is_some()
                    && l.points
                        .first()
                        .map(|p| (*p - near).truncate().length() < radius)
                        .unwrap_or(false)
            })
            .filter_map(|l| l.traffic_light.map(|t| t.0))
            .collect();
        ctls.sort_unstable();
        ctls.dedup();
        self.controller_of_object
            .iter()
            .filter(|(_, c)| ctls.binary_search(c).is_ok())
            .filter_map(|(obj, c)| {
                let ctl = self.sim.lights.get(*c)?;
                Some((*obj, ctl.time, ctl.held))
            })
            .collect()
    }

    /// Where the host's light program of crossing `object` stands (client).
    pub fn set_light_state(&mut self, object: i64, time: f64, held: bool) {
        if let Some(ctl) = self
            .controller_of_object
            .get(&object)
            .and_then(|c| self.sim.lights.get_mut(*c))
        {
            set_mirror_light_clock(ctl, time, held);
        }
    }
}

fn mirror_light_tick(ctl: &mut TrafficLightController, dt: f32, day_time: f64) {
    ctl.request.fill(false);
    ctl.start(day_time);
    if !ctl.held {
        ctl.time = (ctl.time + dt.max(0.0) as f64).rem_euclid(ctl.cycle_len());
    }
}

fn set_mirror_light_clock(ctl: &mut TrafficLightController, time: f64, held: bool) {
    // A crossing may receive its first snapshot before its first tick. Mark its clock
    // started now, so the time-of-day seed cannot replace the host's position later.
    ctl.start(time - ctl.offset as f64);
    ctl.time = time.rem_euclid(ctl.cycle_len());
    ctl.held = held;
}

fn reset_light_runtime(ctl: &mut TrafficLightController, day_time: f64) {
    // Stop/jump bookkeeping belongs to the clock's previous owner. Keep the program
    // and current position, but discard its old requests and visited points.
    ctl.start(day_time);
    let mut fresh = TrafficLightController::new(ctl.lights.clone(), ctl.cycle);
    fresh.offset = ctl.offset;
    fresh.approach = ctl.approach.clone();
    fresh.stops = ctl.stops.clone();
    fresh.start(ctl.time - ctl.offset as f64);
    *ctl = fresh;
}

#[cfg(test)]
mod mirror_light_tests {
    use super::{mirror_light_tick, reset_light_runtime, set_mirror_light_clock};
    use omsi_sim::traffic::TrafficLightController;

    fn program() -> TrafficLightController {
        TrafficLightController::from_program(
            vec![(vec![(0, 4.0), (6, 4.0)], Some(25.0))],
            Some(8.0),
            &[[0.0, 4.0, 1.0]],
            &[[0.0, 6.0, 1.0, 1.0]],
        )
    }

    #[test]
    fn host_hold_survives_missing_local_requests() {
        let mut ctl = program();
        ctl.stops[0].if_request = false;
        ctl.start(4.0);
        set_mirror_light_clock(&mut ctl, 4.0, true);
        for _ in 0..60 {
            mirror_light_tick(&mut ctl, 0.1, 20_000.0);
        }
        assert_eq!(ctl.time, 4.0);
        assert!(ctl.held);
        assert_eq!(ctl.state(0), 6);
    }

    #[test]
    fn unheld_host_clock_crosses_local_stop_and_jump_points() {
        let mut ctl = program();
        ctl.start(3.5);
        set_mirror_light_clock(&mut ctl, 3.5, false);
        mirror_light_tick(&mut ctl, 1.0, 20_000.0);
        assert_eq!(ctl.time, 4.5);
        assert_eq!(ctl.state(0), 6);
        mirror_light_tick(&mut ctl, 2.0, 20_000.0);
        assert_eq!(ctl.time, 6.5);
        assert!(!ctl.held);
    }

    #[test]
    fn first_snapshot_is_not_replaced_by_the_day_time_seed() {
        let mut ctl = program();
        ctl.offset = 0.75;
        set_mirror_light_clock(&mut ctl, 6.5, false);
        mirror_light_tick(&mut ctl, 0.0, 20_000.0);
        assert_eq!(ctl.time, 6.5);
        assert_eq!(ctl.state(0), 6);
    }

    #[test]
    fn a_new_crossing_uses_the_day_clock_until_its_first_snapshot() {
        let mut ctl = program();
        ctl.offset = 0.75;
        mirror_light_tick(&mut ctl, 0.0, 10.0);
        assert_eq!(ctl.time, 2.75);
        set_mirror_light_clock(&mut ctl, 6.5, true);
        mirror_light_tick(&mut ctl, 1.0, 10.0);
        assert_eq!(ctl.time, 6.5);
        assert!(ctl.held);
    }

    #[test]
    fn host_seek_release_and_cycle_wrap_replace_interpolation() {
        let mut ctl = program();
        set_mirror_light_clock(&mut ctl, 7.75, false);
        mirror_light_tick(&mut ctl, 0.5, 0.0);
        assert_eq!(ctl.time, 0.25);
        assert_eq!(ctl.state(0), 0);
        set_mirror_light_clock(&mut ctl, 10.5, true);
        mirror_light_tick(&mut ctl, 2.0, 0.0);
        assert_eq!(ctl.time, 2.5);
        set_mirror_light_clock(&mut ctl, 1.5, false);
        mirror_light_tick(&mut ctl, 1.0, 0.0);
        assert_eq!(ctl.time, 2.5);
        assert!(!ctl.held);
        mirror_light_tick(&mut ctl, -1.0, 0.0);
        assert_eq!(ctl.time, 2.5);
    }

    #[test]
    fn returning_to_local_simulation_discards_a_previously_passed_stop() {
        let mut ctl = program();
        ctl.offset = 0.75;
        ctl.start(3.25);
        ctl.request[0] = true;
        ctl.advance(0.0);
        assert!(!ctl.held);
        set_mirror_light_clock(&mut ctl, 4.0, false);
        reset_light_runtime(&mut ctl, 20_000.0);
        assert_eq!(ctl.lights, vec![vec![(0, 4.0), (6, 4.0)]]);
        assert_eq!(ctl.approach, vec![Some(25.0)]);
        assert_eq!(ctl.offset, 0.75);
        assert_eq!(ctl.stops.len(), 2);
        assert_eq!(ctl.request, vec![false]);
        ctl.advance(0.0);
        assert!(ctl.held);
        assert_eq!(ctl.time, 4.0);
    }

    #[test]
    fn returning_to_local_simulation_discards_a_previous_backward_jump() {
        let mut ctl = program();
        ctl.start(5.5);
        ctl.advance(0.5);
        assert_eq!(ctl.time, 1.0);
        set_mirror_light_clock(&mut ctl, 6.0, false);
        reset_light_runtime(&mut ctl, 20_000.0);
        ctl.advance(0.0);
        assert_eq!(ctl.time, 1.0);
        assert!(!ctl.held);
    }

    #[test]
    fn returning_to_local_simulation_rechecks_a_host_hold() {
        let mut ctl = program();
        ctl.stops[0].if_request = false;
        set_mirror_light_clock(&mut ctl, 4.0, true);
        reset_light_runtime(&mut ctl, 20_000.0);
        assert!(!ctl.held);
        ctl.advance(0.5);
        assert!(!ctl.held);
        assert_eq!(ctl.time, 4.5);
    }

    #[test]
    fn a_new_crossing_keeps_its_first_seed_when_returning_to_local_simulation() {
        let mut ctl = program();
        ctl.offset = 0.75;
        reset_light_runtime(&mut ctl, 10.0);
        assert_eq!(ctl.time, 2.75);
        ctl.start(20_000.0);
        ctl.advance(0.5);
        assert_eq!(ctl.time, 3.25);
    }
}

/// How many times the base street target the neighbourhood asks for, from the trafficdensity
/// of each street lane starting near the player (0 for a no_cars lane): the count of lanes
/// per about 250 (1 to 4) times their mean density (0 to 2).
fn road_scale(near_density: &[f32]) -> f32 {
    let road = (near_density.len() as f32 / 250.0).clamp(1.0, 4.0);
    if near_density.is_empty() {
        return road;
    }
    let mean = near_density.iter().sum::<f32>() / near_density.len() as f32;
    road * mean.clamp(0.0, 2.0)
}

/// Whether the point `p` of a car's way lies in the player's box round `centre` (`wide` to
/// either side, `ahead` in front and `behind` behind it) - on the same level only: a bus
/// under a bridge held up the traffic on the bridge above it (#753). 4 m, as for the other
#[cfg(test)]
mod budget_count_tests {
    use super::budget_count;
    use glam::DVec3;

    /// [round 9 A1] r8-jzk: a queue at the loaded tiles' edge 870 m off held the street
    /// target. Counted near the centre, the far cars do not count, a standing car near does
    /// (round 7 B2), a timetable bus and a car on another kind of lane never do; with no
    /// centre (the aircraft) every car of the kind counts.
    #[test]
    fn a1_far_cars_do_not_hold_the_budget() {
        let c = DVec3::new(464.6, -1578.8, 0.0);
        let cars = [
            (c + DVec3::new(0.0, 120.0, 0.0), false, true),  // near, counts
            (c + DVec3::new(0.0, -399.0, 0.0), false, true), // near, counts
            (c + DVec3::new(10.0, -870.0, 0.0), false, true), // the far queue
            (c + DVec3::new(0.0, 50.0, 0.0), true, true),    // a timetable bus
            (c + DVec3::new(0.0, 60.0, 0.0), false, false),  // another kind
        ];
        assert_eq!(budget_count(cars.iter().copied(), Some((c, 400.0))), 2);
        assert_eq!(budget_count(cars.iter().copied(), None), 3);
    }
}

#[cfg(test)]
mod edge_flag_tests {
    use super::{edge_trip_end, network_end_rule, EdgeFlag, RemoveRule};

    fn flag() -> EdgeFlag {
        EdgeFlag { d_end: Some(300.0), beyond: true, blinker: 0, braking: false, prev: false, speed: 12.0, comfort_brake: 3.0 }
    }

    /// [round 9 A2 fix pass 3] The flag: set heading for the edge beyond the horizon, lamps dark;
    /// not without the edge ahead (a tile loaded: the lane has a way on), inside the horizon, with
    /// an indicator, or braking when not yet set; held through a brake on the way once set.
    #[test]
    fn a2_the_edge_flag_is_set_and_held() {
        assert!(edge_trip_end(flag()));
        assert!(!edge_trip_end(EdgeFlag { d_end: None, ..flag() }), "not heading for the edge (or its tile loaded)");
        assert!(!edge_trip_end(EdgeFlag { beyond: false, ..flag() }), "inside the horizon");
        assert!(!edge_trip_end(EdgeFlag { blinker: 1, ..flag() }), "an indicator");
        assert!(!edge_trip_end(EdgeFlag { braking: true, ..flag() }), "braking, not yet set");
        assert!(edge_trip_end(EdgeFlag { braking: true, prev: true, ..flag() }), "held through a brake on the way");
        assert!(!edge_trip_end(EdgeFlag { d_end: None, prev: true, ..flag() }), "the tile loaded: cleared");
    }

    /// [pass 3, 1] Within its comfortable braking distance of the end (12 m/s at 3 m/s²: 24 m) a
    /// braking car loses the flag and stops at the edge; [pass 3, 2] closer than it can stop at
    /// MAX_BRAKE (9 m) a set flag holds whatever the horizon, the indicator or the brake.
    #[test]
    fn a2_the_flag_by_braking_distance() {
        let near = EdgeFlag { d_end: Some(20.0), prev: true, ..flag() };
        assert!(edge_trip_end(near), "within 24 m, not braking: held");
        assert!(!edge_trip_end(EdgeFlag { braking: true, ..near }), "within 24 m, braking: stops at the edge");
        assert!(!edge_trip_end(EdgeFlag { beyond: false, ..near }), "inside the horizon and able to stop: cleared, stops");
        let late = EdgeFlag { d_end: Some(5.0), prev: true, ..flag() };
        assert!(edge_trip_end(EdgeFlag { beyond: false, ..late }), "the horizon jumped, it cannot stop: held");
        assert!(edge_trip_end(EdgeFlag { braking: true, blinker: 1, ..late }), "cannot stop: held");
        assert!(!edge_trip_end(EdgeFlag { prev: false, beyond: false, ..late }), "never set: not set inside the horizon");
    }

    /// [pass 3, 3] The rule a network-end removal is recorded by.
    #[test]
    fn a2_the_far_trip_end_is_recorded_as_such() {
        assert_eq!(network_end_rule(false, true, true), RemoveRule::FarDeadEnd);
        assert_eq!(network_end_rule(false, true, false), RemoveRule::NetworkEnd, "off the edge unflagged");
        assert_eq!(network_end_rule(false, false, true), RemoveRule::NetworkEnd, "the map's own end");
        assert_eq!(network_end_rule(true, true, true), RemoveRule::BusLastResort);
    }
}

#[cfg(test)]
mod road_scale_tests {
    use super::road_scale;

    #[test]
    fn path_density_scales_the_street_target() {
        assert!((road_scale(&[1.0; 100]) - 1.0).abs() < 1e-6);
        assert!((road_scale(&[0.2; 100]) - 0.2).abs() < 1e-6);
        assert_eq!(road_scale(&[0.0; 100]), 0.0);
        assert!((road_scale(&[1.0; 500]) - 2.0).abs() < 1e-6);
    }
}

#[cfg(test)]
mod group_density_tests {
    use omsi_sim::traffic::pool_density as uvg_density;

    /// Berlin-Spandau's `unsched_vehgroups.txt`: NormalCars 1, Trucks 0, Commercials 1,
    /// Ambulance 1, GDRCars 0.
    const SPANDAU: [i32; 5] = [1, 0, 1, 1, 0];
    #[test]
    fn a_group_off_by_default_drives_where_a_path_asks_for_it() {
        // a Falkensee path: no rule for the normal cars, the GDR cars asked for
        let rules = [(4u16, 1.0f32)];
        assert_eq!(uvg_density(&rules, &SPANDAU, 4), 1.0);
        assert_eq!(uvg_density(&rules, &SPANDAU, 0), 1.0);
        // and nowhere else
        assert_eq!(uvg_density(&[], &SPANDAU, 4), 0.0);
        assert_eq!(uvg_density(&[(0, 0.5)], &SPANDAU, 1), 0.0);
    }

    #[test]
    fn a_default_follows_the_first_group_on_the_path() {
        // commercials (default 1) take the normal cars' density of the path
        assert_eq!(uvg_density(&[(0, 0.4)], &SPANDAU, 2), 0.4);
        assert_eq!(uvg_density(&[(0, 0.0)], &SPANDAU, 2), 0.0);
        assert_eq!(uvg_density(&[], &SPANDAU, 2), 1.0);
        // an own rule wins
        assert_eq!(uvg_density(&[(0, 0.4), (2, 2.0)], &SPANDAU, 2), 2.0);
    }

    #[test]
    fn defaults_naming_each_other_end() {
        assert_eq!(uvg_density(&[], &[1, 3, 2], 1), 0.0);
    }
}



#[cfg(test)]
mod respawn_sight_tests {
    use super::{Viewer, NEAR_HIDE, NEVER_VANISH_WITHIN};
    use glam::DVec3;

    /// A viewer at the origin looking along +Y, a 90 degree picture, seeing 900 m.
    fn viewer() -> Viewer {
        Viewer { pos: DVec3::ZERO, forward: DVec3::Y, tan_x: 1.0, tan_y: 1.0, range: 900.0, min_size: 0.0, max_dist: 0.0, fov: 1.57, px_height: 0.0, horizon_fov: 0.0, lamps_lit: false }
    }

    /// [ROLLBACK r4d-B2] A bus is taken off the road by the rule it comes back by: never
    /// near the player, nor in the picture within 350 m, whatever the collision boxes say.
    #[test]
    fn a_bus_goes_only_where_it_could_come_back() {
        let v = viewer();
        // 120 m behind the player, out of the picture: the mirrors see it
        assert!(!v.hides_plainly(DVec3::new(0.0, -120.0, 0.0), 8.0));
        assert!(NEVER_VANISH_WITHIN > 120.0);
        // 183 m ahead, in plain sight (the bus lost at Anwai)
        assert!(!v.hides_plainly(DVec3::new(0.0, 183.0, 0.0), 8.0));
        // 183 m behind: in no mirror's range, but a turn of the head sees it
        assert!(!v.hides_plainly(DVec3::new(0.0, -183.0, 0.0), 8.0));
        assert!(NEAR_HIDE > 183.0);
        // 400 m behind, out of the picture
        assert!(v.hides_plainly(DVec3::new(0.0, -400.0, 0.0), 8.0));
        // 400 m ahead, in the picture
        assert!(!v.hides_plainly(DVec3::new(0.0, 400.0, 0.0), 8.0));
        // beyond what is drawn at all
        assert!(v.hides_plainly(DVec3::new(0.0, 950.0, 0.0), 8.0));
        // the renderer's culling leaves a bus out from 300 m: it goes there
        let v = viewer().with_culling(0.0, 300.0);
        assert!(v.hides_plainly(DVec3::new(0.0, 330.0, 0.0), 8.0));
        assert!(!v.hides_plainly(DVec3::new(0.0, 280.0, 0.0), 8.0));
    }

    /// [round 9 A2] The perceptual horizon: a 1.5 m car covers two rows of a 900-line picture
    /// with a 60 degree field of view at 585 m (r8-jzk's tile-edge queue at 869 m was 1.3 rows),
    /// at 701 m on 1080 lines; a 3 m bus twice as far; with the picture's height unknown it is
    /// VISIBLE_RANGE for a car. [fix pass 1] Never within NEAR_HIDE however small the picture;
    /// none at all with the lamps lit and the draw distance unknown; a triple screen's narrower
    /// centre panel pushes it out.
    #[test]
    fn a2_the_horizon_is_where_a_body_is_two_pixel_rows() {
        use super::{VISIBLE_RANGE, CAR_HEIGHT, DOT_PIXELS};
        let mut v = viewer();
        v.fov = 60f64.to_radians();
        assert_eq!(v.horizon(CAR_HEIGHT), VISIBLE_RANGE);
        let v9 = v.with_screen(900, None);
        assert!((v9.horizon(CAR_HEIGHT) - 584.6).abs() < 0.5, "{}", v9.horizon(CAR_HEIGHT));
        let rows = |v: &Viewer, d: f64| CAR_HEIGHT / d * v.px_height / (2.0 * (v.fov * 0.5).tan());
        assert!((rows(&v9, v9.horizon(CAR_HEIGHT)) - DOT_PIXELS).abs() < 1e-9);
        assert!(rows(&v9, 869.0) < 1.4);
        assert!(v9.beyond_horizon(600.0, CAR_HEIGHT, 2.0) && !v9.beyond_horizon(570.0, CAR_HEIGHT, 2.0));
        assert!((v9.horizon(3.0) - 2.0 * v9.horizon(CAR_HEIGHT)).abs() < 1e-6, "a bus twice as far");
        assert!(!v9.beyond_horizon(1000.0, 3.0, 6.0));
        assert!((v.with_screen(1080, None).horizon(CAR_HEIGHT) - 701.5).abs() < 0.5);
        let tiny = v.with_screen(100, None);
        assert!(tiny.horizon(CAR_HEIGHT) < 100.0 && !tiny.beyond_horizon(NEAR_HIDE - 1.0, CAR_HEIGHT, 2.0) && tiny.beyond_horizon(NEAR_HIDE, CAR_HEIGHT, 2.0));
        let lit = v9.with_lamps(true);
        assert!(lit.horizon(CAR_HEIGHT).is_infinite() && !lit.beyond_horizon(5000.0, CAR_HEIGHT, 2.0));
        let triple = v.with_screen(900, Some(29.0));
        assert!(triple.horizon(CAR_HEIGHT) > 2.0 * v9.horizon(CAR_HEIGHT), "{}", triple.horizon(CAR_HEIGHT));
        assert_eq!(v.with_screen(900, Some(90.0)).horizon(CAR_HEIGHT), v9.horizon(CAR_HEIGHT), "the narrower of the two");
    }

    /// [round 9 A2 fix pass 1] The set of trips ending at the edge: the head at the edge 869 m
    /// off goes; a follower 500 m off (inside the horizon) stays, and so does the car behind
    /// it 700 m off (beyond, but in a queue): only the car AT the edge goes.
    #[test]
    fn a2_only_the_car_at_the_edge_goes() {
        use omsi_traffic::{Decision, Vehicle};
        use omsi_sim::traffic::Lead;
        let queued = |lead: u64| Vehicle { last: Decision { lead_id: Some(lead), lead: Some(Lead { gap: 2.5, speed: 0.0, acc: 0.0 }), ..Default::default() }, ..Default::default() };
        let head = Vehicle::default();
        let (f1, f2) = (queued(1), queued(2));
        let mut v = viewer();
        v.fov = 60f64.to_radians();
        let v = v.with_screen(900, None);
        let d = |m: f64| v.beyond_horizon(m, super::CAR_HEIGHT, 2.0);
        let cars = vec![(1u64, true, d(869.0), &head), (2, false, d(500.0), &f1), (3, false, d(700.0), &f2)];
        let gone: Vec<u64> = super::far_trip_ends(cars.into_iter()).into_iter().collect();
        assert_eq!(gone, vec![1]);
        assert!(super::far_trip_ends(vec![(1u64, true, d(500.0), &head)].into_iter()).is_empty(), "inside the horizon");
        // [fix pass 2] one that stopped there brakes: its brake lamps glow, it stays
        let mut braking = Vehicle::default();
        braking.state.braking = true;
        assert!(super::far_trip_ends(vec![(1u64, true, d(869.0), &braking)].into_iter()).is_empty(), "a braking car stays");
    }
}

#[cfg(test)]
mod prefetch_tests {
    use super::*;

    /// The prefetch reach: the horizon a 1.5 m car has on a 900-line, 60 degree picture (585 m)
    /// plus the distance driven during one tile load at the road's speed; a lit picture keeps
    /// the dark horizon (no area reaches a lamp's), a tiny one NEAR_HIDE, a bus twice the car's.
    #[test]
    fn the_reach_is_the_horizon_plus_one_load_at_the_road_speed() {
        let v = Viewer { pos: DVec3::ZERO, forward: DVec3::Y, tan_x: 1.0, tan_y: 1.0, range: 900.0, min_size: 0.0, max_dist: 0.0, fov: 60f64.to_radians(), px_height: 0.0, horizon_fov: 0.0, lamps_lit: false }.with_screen(900, None);
        let h = edge_horizon(v, CAR_HEIGHT);
        assert!((h - 584.6).abs() < 0.5, "{h}");
        assert_eq!(edge_horizon(v.with_lamps(true), CAR_HEIGHT), h, "lit lamps: the dark horizon");
        assert!((edge_horizon(v, 3.0) - 2.0 * h).abs() < 1e-6);
        assert_eq!(edge_horizon(v.with_screen(100, None), CAR_HEIGHT), NEAR_HIDE);
        // 50 km/h and a 1.2 s load: 16.7 m further
        assert!((prefetch_target(h, 50.0 / 3.6, 1.2) - (h + 16.666)).abs() < 0.01);
        assert_eq!(prefetch_target(h, 0.0, 3.0), h, "standing: the horizon");
        assert_eq!(prefetch_target(h, -5.0, 1.0), h, "no negative speed");
        assert_eq!(prefetch_target(h, 10.0, 0.0), h, "no load measured yet");
        // the horizon is the nearer of the two-pixel distance and the renderer's reach (the object
        // distance held out to MAX_OBJ_DIST_HOLD; the Low preset's 600 m with `max_obj_dist` auto):
        // a 4.4 m double-decker's (1715 m) is 630 m there and the reach 630 m plus one load; at
        // 2500 m it is the two-pixel distance
        let hold = omsi_render::MAX_OBJ_DIST_HOLD as f64;
        let low = v.with_culling(0.0, 600.0);
        assert_eq!(edge_horizon(low, 4.4), 600.0 * hold);
        assert_eq!(low.with_lamps(true).horizon(CAR_HEIGHT), 600.0 * hold, "lit lamps: the reach (lamps are culled with the car)");
        assert!((prefetch_target(edge_horizon(low, 4.4), 50.0 / 3.6, 1.2) - (600.0 * hold + 16.667)).abs() < 0.01);
        let far = v.with_culling(0.0, 2500.0);
        assert!((edge_horizon(far, 4.4) - 4.4 / CAR_HEIGHT * h).abs() < 1e-6, "{}", edge_horizon(far, 4.4));
        assert_eq!(far.with_lamps(true).horizon(CAR_HEIGHT), 2500.0 * hold);
        // a body: beyond its two-pixel distance, or where the renderer does not draw it (`draws`,
        // the same reach plus its padded radius) - by day and, lamps culled with it, by night
        let lit = low.with_lamps(true);
        let r = 6.0;
        let drawn_to = 600.0 * hold + r * 1.5 + 1.0;
        assert!(!lit.beyond_horizon(drawn_to - 1.0, 4.4, r) && lit.beyond_horizon(drawn_to + 1.0, 4.4, r));
        assert!(!low.beyond_horizon(drawn_to - 1.0, 4.4, r) && low.beyond_horizon(drawn_to + 1.0, 4.4, r));
        // a lit car too small for the renderer's minimum size (the user's 0.005) still glows: at
        // 2000 m with objects to 3000 m it is not beyond the horizon; dark, it is (size-culled and
        // past its two-pixel distance)
        let user = v.with_culling(0.005, 3000.0);
        assert!(!user.draws(2000.0, 2.0) && user.with_lamps(true).draws(2000.0, 2.0));
        assert!(!user.with_lamps(true).beyond_horizon(2000.0, CAR_HEIGHT, 2.0));
        assert!(user.beyond_horizon(2000.0, CAR_HEIGHT, 2.0));
    }
}

#[cfg(test)]
mod edge_sight_tests {
    use super::*;

    /// [round 10 S4] One horizon (648 m, the harness's picture): an edge 500 m from the eye is in
    /// sight - avoided by A4, its road kept clear of new cars - one 2500 m off is not: the road to
    /// it is a way like any other and gets cars.
    #[test]
    fn s4_an_edge_at_500_m_is_avoided_and_one_at_2500_m_is_not() {
        let horizon = 648.0;
        let eye = DVec3::ZERO;
        for (d, seen) in [(500.0, true), (2500.0, false)] {
            let mut net = Network::default();
            let mut a = omsi_sim::traffic::LaneBuilder::polyline(vec![DVec3::new(0.0, d - 200.0, 0.0), DVec3::new(0.0, d - 100.0, 0.0)], LaneKind::Street, 3.5);
            a.source = 1;
            let mut b = omsi_sim::traffic::LaneBuilder::polyline(vec![DVec3::new(0.0, d - 100.0, 0.0), DVec3::new(0.0, d, 0.0)], LaneKind::Street, 3.5);
            b.source = 1;
            net.lanes.push(a);
            net.lanes.push(b);
            net.link(LANE_LINK_TOL);
            net.lanes[1].loaded_edge = true;
            net.lanes[1].edge_in_sight = edge_in_sight(net.lanes[1].end(), eye, horizon);
            assert_eq!(net.lanes[1].edge_in_sight, seen, "edge at {d} m");
            assert_eq!(net.ends_at_loaded_edge(0), seen, "routed away from at {d} m");
            assert_eq!(edge_bound_of(&net).contains(&0), seen, "kept clear of new cars at {d} m");
        }
    }
}

#[cfg(test)]
mod loaded_edge_tests {
    use super::*;

    /// [round 8 H8-7, fix pass 1] A lane end whose way on lies on a tile of the map that is not
    /// loaded is the loaded tiles' edge (the car stops there and goes once hidden, a dead-end
    /// removal); one whose way on lies on a loaded tile, or off the map, is the network's end; the
    /// removal is a gated dead-end rule.
    #[test]
    fn h8_7_the_loaded_edge_is_a_tile_of_the_map_not_loaded() {
        let ts = 300.0;
        let map: hashbrown::HashSet<(i32, i32)> = [(0, 0), (0, 1)].into_iter().collect();
        let has = |k: (i32, i32)| map.contains(&k);
        let mut loaded: hashbrown::HashSet<(i32, i32)> = [(0, 0)].into_iter().collect();
        let probe = DVec3::new(10.0, 300.0 + LANE_LINK_TOL, 0.0);
        assert!(loaded_edge_at(probe, ts, &has, &loaded), "the next tile is the map's, not loaded");
        loaded.insert((0, 1));
        assert!(!loaded_edge_at(probe, ts, &has, &loaded), "loaded: a real end");
        assert!(!loaded_edge_at(DVec3::new(-10.0, 100.0, 0.0), ts, &has, &loaded), "off the map");
        assert_eq!(removal_rule("at the loaded tiles' edge, out of sight"), RemoveRule::DeadEnd);
        assert_eq!(removal_rule("at the loaded tiles' edge, in view after its grace"), RemoveRule::InViewAfterGrace);
        assert!(RemoveRule::DeadEnd.gated() && RemoveRule::InViewAfterGrace.gated());
        // [round 9 A2, fix pass 1] the trip's end at the edge beyond the horizon
        assert_eq!(removal_rule("at the loaded tiles' edge, beyond the horizon"), RemoveRule::FarDeadEnd);
    }
}
