//! The offscreen harness's moving player.
//!
//! `--follow-line` (`LineFollower`): the player's bus drives its duty's trip as a timetable
//! bus of the AI drives one. The player's vehicle is a physics vehicle outside the traffic's
//! cars, so the AI controller cannot drive it directly: a timetable bus of the player's own
//! type takes the player's departure on where the player's bus stands
//! (`Schedule::drive_player_trip`), and every frame the player's vehicle is put where that
//! bus's body is (`pose_player`: the AI's own `AiBody` pose, its speed, its rear sections).
//! So the player's bus stops at its stops for a dwell with the doors open, obeys the lights
//! and gives way as an AI timetable bus does (`bus_service`, `omsi_traffic`), and moves as
//! the AI's bodies move (`ai_motion`). The physics and scripts of the player's vehicle do not
//! run meanwhile; its doors stay shut (the people board the AI bus in its place). The other
//! drivers see the AI bus, not a player's outline (`Traffic::tick` gets no player box while
//! it drives): what they do round the player's bus is what they do round a timetable bus.
//! When the bus is lost (taken off the road, its trip over and let go) the player's bus
//! stands where it was left.

use crate::traffic::Traffic;
use glam::{DVec3, Vec3};
use omsi_sim::ai_motion::AiBody;
use omsi_sim::VehicleInstance;

/// Seconds between two tries to put the AI bus on the player's trip when it could not be put
/// there (a tile not loaded yet, the lane taken): the timetable's own retry
/// (`Placed::Busy` waits 3 s).
const RETRY_EVERY: f32 = 3.0;

/// The player's vehicle put where the AI's body is: its pose (`AiBody::apply`, as the AI
/// vehicles get it), its speed and its steering, its rigid body placed there moving at that
/// speed, and the rear sections where the AI vehicle's are (`ai_rear`).
pub(crate) fn pose_player(body: &AiBody, speed: f32, ai_rear: &[(DVec3, f64)], v: &mut VehicleInstance) {
    body.apply(v);
    v.physics.speed = speed;
    v.physics.steer_deg = body.steer;
    let (position, heading) = (v.position, v.heading);
    if let Some(rb) = v.rigid.as_mut() {
        rb.place(position, heading);
        rb.velocity = rb.orientation.mul_vec3(Vec3::Y) * speed;
    }
    for (part, &(p, h)) in v.trailers.iter_mut().zip(ai_rear) {
        part.set_pose(p, h);
    }
}

/// `--follow-line`: the AI bus that drives the player's trip, and the player's vehicle
/// following it.
#[derive(Default)]
pub(crate) struct LineFollower {
    /// The AI bus's car id once it is on the road.
    pub car: Option<u64>,
    /// When to try again to put it there (run seconds).
    next_try: f32,
    /// Why it was not put there the last time.
    pub refused: Option<String>,
    /// The run second it was lost, and why.
    pub lost: Option<(f32, &'static str)>,
}

impl LineFollower {
    /// Whether the player's vehicle is driven by the AI bus now.
    pub fn driving(&self) -> bool {
        self.car.is_some() && self.lost.is_none()
    }

    /// Put the AI bus on the player's trip, unless it is there or was lost; a try every
    /// `RETRY_EVERY` s.
    #[allow(clippy::too_many_arguments)]
    pub fn place(
        &mut self,
        t_s: f32,
        schedule: &mut crate::schedule::Schedule,
        duty: &crate::schedule::PlayerDuty,
        world: &crate::scene::World,
        traffic: &mut Traffic,
        renderer: &omsi_render::Renderer,
        scene: &mut omsi_render::Scene,
        bus: &VehicleInstance,
    ) {
        if self.car.is_some() || self.lost.is_some() || t_s < self.next_try {
            return;
        }
        self.next_try = t_s + RETRY_EVERY;
        match schedule.drive_player_trip(world, traffic, renderer, scene, traffic.day_time, duty, bus) {
            Ok(id) => {
                self.car = Some(id);
                self.refused = None;
            }
            Err(e) => {
                if self.refused.as_deref() != Some(e.as_str()) {
                    log::info!("follow line t={t_s:.1}: {e}");
                }
                self.refused = Some(e);
            }
        }
    }

    /// The player's vehicle put where the AI bus is this frame (`pose_player`); true while
    /// it drives it.
    pub fn pose(&mut self, t_s: f32, traffic: &Traffic, v: &mut VehicleInstance) -> bool {
        let Some(id) = self.car else { return false };
        if self.lost.is_some() {
            return false;
        }
        let Some(c) = traffic.cars.iter().find(|c| c.id == id) else {
            self.lost = Some((t_s, "taken off the road"));
            log::info!("follow line t={t_s:.1}: bus {id} taken off the road; the player's bus stands where it was left");
            return false;
        };
        if c.released || !c.serves_stops() {
            self.lost = Some((t_s, "trip over"));
            log::info!("follow line t={t_s:.1}: bus {id} let go after its trip; the player's bus stands where it was left");
            return false;
        }
        let rear: Vec<(DVec3, f64)> = c.vehicle.trailers.iter().map(|t| (t.position, t.heading)).collect();
        pose_player(&c.body, c.ai.state.speed, &rear, v);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus_service::{BusService, Ctx, Stop};
    use omsi_sim::ai_motion::MotionKind;
    use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network, TrafficLightController};
    use omsi_traffic::{Simulation, Vehicle, VehicleClass, Why};

    const DT: f32 = 1.0 / 30.0;

    fn lane(points: Vec<DVec3>, id: i64, source: u32) -> omsi_sim::traffic::Lane {
        let mut l = LaneBuilder::polyline(points, LaneKind::Street, 3.5);
        l.source = source;
        l.key = Some(LaneKey { tile: (0, 0), id, path: 0 });
        l
    }

    /// North 150 m to a light, 20 m through it, a right bend of 30 m radius, east 2 km.
    fn route() -> Network {
        let mut net = Network::default();
        net.lanes.push(lane(vec![DVec3::ZERO, DVec3::new(0.0, 150.0, 0.0)], 1, 1));
        let mut light = lane(vec![DVec3::new(0.0, 150.0, 0.0), DVec3::new(0.0, 170.0, 0.0)], 9, 2);
        light.priority = 128.0;
        light.traffic_light = Some((0, 0));
        net.lanes.push(light);
        let r = 30.0;
        let bend: Vec<DVec3> = (0..=24)
            .map(|k| {
                let a = k as f64 / 24.0 * std::f64::consts::FRAC_PI_2;
                DVec3::new(r - r * a.cos(), 170.0 + r * a.sin(), 0.0)
            })
            .collect();
        net.lanes.push(lane(bend, 2, 1));
        net.lanes.push(lane(vec![DVec3::new(r, 200.0, 0.0), DVec3::new(2000.0, 200.0, 0.0)], 3, 1));
        net.link(1.5);
        net.compute_conflicts();
        net
    }

    /// A two-axle bus, 12 m long, as the player's `VehicleInstance` and the AI's body see it.
    fn bus() -> VehicleInstance {
        use omsi_vehicle::vehicle::Axle;
        let mut def = omsi_vehicle::Vehicle { mass: 11.0, inv_min_turn_radius: 0.1, cog_height: 1.0, ..Default::default() };
        def.axles = vec![Axle { long: 3.0, max_width: 2.0, spring: 300.0, damper: 20.0, wheel_diameter: 1.0, ..Default::default() }, Axle { long: -2.9, max_width: 2.0, spring: 300.0, damper: 20.0, wheel_diameter: 1.0, ..Default::default() }];
        let program = omsi_script::compile(&omsi_script::CompileInput::default());
        let ty = std::sync::Arc::new(omsi_sim::VehicleType {
            def,
            model: Default::default(),
            model_dir: std::path::PathBuf::new(),
            program: std::sync::Arc::new(program),
            meshes: Vec::new(),
            paint_schemes: Vec::new(),
            texchanges: Vec::new(),
            wheel_meshes: Vec::new(),
            suspension_axles: Vec::new(),
            missing_packs: Vec::new(),
            mesh_bounds: Vec::new(),
            mesh_boxes: Vec::new(),
        });
        VehicleInstance::new(ty, omsi_sim::VehicleHost::new(Default::default()))
    }

    /// The player's bus put where an AI timetable bus driving the route is (`pose_player`),
    /// the bus driven as `Traffic::tick` drives one (its service's stop, pace and bay into
    /// the driver, `Simulation::tick`, its body stepped along its way): it keeps within its
    /// lane on the straight and through the bend, stops at its stop with the doors open and
    /// at the red light, and goes on when the light turns green.
    #[test]
    fn the_player_follows_its_route_in_its_lane_and_stops_at_its_stop_and_at_a_red_light() {
        let net = route();
        // red for 90 s, then green
        let ctl = TrafficLightController::new(vec![vec![(0, 90.0), (6, 90.0)]], 180.0);
        let mut sim = Simulation::new(&net, vec![ctl]);
        let mut st = AiState::new(0, 10.0, 1);
        (st.front, st.rear, st.length, st.max_speed_kmh, st.veh_type) = (6.0, 6.0, 12.0, 50.0, 0);
        st.route = vec![0, 1, 2, 3];
        let mut ai = vec![Vehicle::new(1, VehicleClass::ScheduledBus, st, 1.25, 7919)];
        ai[0].fresh = 0.0;
        // (its stop on the first straight, and the next one far down the last lane: the trip
        // goes on past the bend)
        let stop = Stop { ri: 0, s: 60.0, bay: 0.0, depart: 0.0, id: 7, side: 0.0 };
        let mut service = BusService::new(vec![stop, Stop { ri: 3, s: 1500.0, id: 8, ..stop }]);
        let mut ai_vehicle = bus();
        let mut player = bus();
        let mut body = AiBody::new(&player.ty.def, MotionKind::Road);
        let half_lane = 3.5 / 2.0;
        let (mut worst, mut served_at, mut stood_at_red, mut on_red) = (0.0f64, None, false, false);
        let (mut t, mut stopped) = (0.0f64, 0.0f32);
        while t < 200.0 && !(ai[0].state.lane == 3 && ai[0].state.s > 50.0) {
            let way = omsi_traffic::perception::way_lanes(&net, &ai[0].state, 200.0);
            let ctx = Ctx { net: &net, way: &way, day_time: t, dt: DT, id: 1, stopped, passing: false, lead_close: false, follower_close: false, wanted: None, debug: false, roadside: false, queue_behind_secs: 0.0, v_nom: 13.9, people_hold: false, close_behind: false, covers_through: false };
            let stop_at = service.step(&mut ai[0].state, &mut ai_vehicle, &ctx);
            ai[0].external.stop_at = stop_at;
            ai[0].external.at_stop = service.at_stop();
            ai[0].external.pace = Some(service.pace);
            ai[0].external.bay = service.bay_wish.or(service.at_stop().then_some(ai[0].state.lateral_target));
            sim.tick(&net, DT, &mut ai);
            assert!(sim.removed.is_empty(), "the bus was taken off its route at {t:.1} s");
            let (state, speed) = (&ai[0].state, ai[0].state.speed);
            let way_point = |d: f32| state.way_point(&net, d);
            if t == 0.0 {
                body.place(&way_point, None, None, speed);
            } else {
                body.step(DT, speed, &way_point, None, None);
            }
            pose_player(&body, speed, &[], &mut player);
            stopped = if speed < 0.3 { stopped + DT } else { 0.0 };
            // within its lane: the player's origin off the middle line of the lane under it
            let off = state.route.iter().filter_map(|&l| net.lanes[l].nearest_point(player.position)).map(|(_, d)| d).fold(f64::MAX, f64::min);
            worst = worst.max(off);
            assert!((player.physics.speed - speed).abs() < 1e-6);
            let red = sim.lights[0].state(0) < 6;
            if service.at_station() && player.physics.speed < 0.3 && served_at.is_none() {
                served_at = Some((t, player.position.y));
            }
            if red && state.lane == 0 && speed < 0.3 && ai[0].last.why == Why::Light {
                stood_at_red = true;
            }
            if red && state.lane != 0 {
                on_red = true;
            }
            t += DT as f64;
        }
        assert!(worst <= half_lane, "{worst:.2} m off its lane's middle line");
        let (when, y) = served_at.expect("never stood at its stop with the doors open");
        // (at its stop point, where `BusService::step` brings it to rest)
        assert!((y - 60.0).abs() < 2.0, "served its stop at y {y:.1} ({when:.1} s)");
        assert!(stood_at_red, "never stood at the red light");
        assert!(!on_red, "drove past the red light");
        assert!(ai[0].state.lane == 3, "never got through the bend after the green ({t:.0} s, lane {})", ai[0].state.lane);
    }
}
