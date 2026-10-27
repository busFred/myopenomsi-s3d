//! What the traffic is told about the player and the LAN buses.

use super::*;

/// Position, heading, half extents and speed of the player's vehicle for the AI's
/// obstacle checks.
pub(crate) fn player_outline(p: &Player) -> traffic::PlayerBox {
    vehicle_outline(&p.vehicle, p.vehicle.physics.speed)
}

/// `player_outline` of any vehicle moving at `speed` (m/s).
pub(crate) fn vehicle_outline(v: &omsi_sim::VehicleInstance, speed: f32) -> traffic::PlayerBox {
    let bb =
        v.ty.def
            .bounding_box
            .unwrap_or([2.5, 11.0, 3.0, 0.0, 0.0, 1.5]);
    let h = v.heading.to_radians();
    let centre = v.position
        + DVec3::new(
            (bb[3] as f64) * h.cos() + (bb[4] as f64) * h.sin(),
            -(bb[3] as f64) * h.sin() + (bb[4] as f64) * h.cos(),
            0.0,
        );
    (centre, v.heading, bb[1] * 0.5, bb[0] * 0.5, speed)
}

/// Everything of the player's besides the bus's own box that the traffic has to keep out
/// of: the rear sections of an articulated bus or a coupled trailer (the traffic saw only
/// the front section and drove into the back of a turning GN92), and the vehicles placed
/// by hand from the vehicle list, with their trailers - under ids of their own beside the
/// LAN players'.
pub(crate) fn own_outlines(player: Option<&Player>, placed: &[Player]) -> Vec<(u32, traffic::PlayerBox)> {
    let mut out = Vec::new();
    let mut add = |v: &omsi_sim::VehicleInstance, base: u32, whole: bool| {
        let speed = v.physics.speed;
        if whole {
            out.push((base, vehicle_outline(v, speed)));
        }
        for (k, t) in v.trailers.iter().enumerate() {
            let Some(bb) = t.ty.def.bounding_box else { continue };
            let h = t.heading.to_radians();
            let centre = t.position
                + DVec3::new(
                    (bb[3] as f64) * h.cos() + (bb[4] as f64) * h.sin(),
                    -(bb[3] as f64) * h.sin() + (bb[4] as f64) * h.cos(),
                    0.0,
                );
            out.push((base + 1 + k as u32, (centre, t.heading, bb[1] * 0.5, bb[0] * 0.5, speed)));
        }
    };
    if let Some(p) = player {
        add(&p.vehicle, 0xFFFF_0000, false);
    }
    for (i, q) in placed.iter().enumerate() {
        add(&q.vehicle, 0xFFFE_0000 - (i as u32) * 16, true);
    }
    out
}

/// The LAN players' buses as obstacles for the AI traffic (their speed as last sent).
pub(crate) fn lan_outlines(game: &lan::LanGame) -> Vec<(u32, traffic::PlayerBox)> {
    game.remotes
        .iter()
        .map(|(id, r)| (*id, vehicle_outline(r.vehicle(), r.last.speed_kmh / 3.6)))
        .collect()
}

/// What the traffic needs to know every frame besides the time: where the player looks
/// from and how far the picture reaches (the renderer's culling), the day of the week, who walks
/// the footpaths, what hides what.
pub(crate) fn traffic_inputs(
    t: &mut traffic::Traffic,
    cam: Option<&Camera>,
    aspect: f64,
    screen: (u32, Option<f32>),
    extent: Option<(f64, f64)>,
    fog: f64,
    clock: &omsi_sim::SimClock,
    humans: Option<&humans::Humans>,
    player: Option<&Player>,
    render: &omsi_render::RenderOptions,
    loaded: f64,
) {
    if let Some(c) = cam {
        t.viewer = Some(
            traffic::Viewer::new(c, aspect, fog)
                .with_screen(screen.0, screen.1)
                .with_lamps(t.lamps_lit())
                .with_extent(extent)
                .with_culling(render.min_obj_size, render.max_obj_dist)
                .with_loaded(loaded),
        );
        // which loaded edges A4 routes away from follows the eye and its horizon
        t.mark_edges_in_sight();
    }
    t.weekday = clock.weekday();
    t.walkers = humans.map(|h| h.strollers()).unwrap_or_default();
    t.people = humans.map(|h| h.on_foot()).unwrap_or_default();
    t.people_records = humans.map(|h| h.records()).unwrap_or_default();
    t.people_gone = humans.map(|h| h.gone().to_vec()).unwrap_or_default();
    // the player's obstacle boxes follow the streamed tiles; without a player the world's
    // own are asked
    t.occluders = player.and_then(|p| p.vehicle.collision.clone());
}

/// Posts (`[crashmode_pole]`) the vehicle knocked over this frame: laid on the ground from
/// their foot in the direction they were hit.
pub(crate) fn lay_down_poles(
    world: &World,
    renderer: &Renderer,
    scene: &mut Scene,
    vehicle: &mut omsi_sim::VehicleInstance,
) {
    for (id, push) in std::mem::take(&mut vehicle.knocked_now) {
        if let Some(pos) = world.lay_down_pole(renderer, scene, id, push) {
            log::info!("knocked over post {id} at ({:.1}, {:.1})", pos.x, pos.y);
        }
    }
}

/// Whether the AI vehicles have their lights on (`Traffic::night`): Omsi switches them on below a
/// light value of 0.75, before the street lamps (0.6), and off after them in the morning; and by
/// day in fog, rain, snow or under a closed cloud cover, as drivers do. ([round 9 A2 fix pass 2]
/// one rule for the window and the offscreen harness, which had the light value alone.)
pub(crate) fn ai_lights_on(daylight: &omsi_sim::Daylight, weather: Option<&omsi_content::weather::Weather>) -> bool {
    let gloomy = weather.is_some_and(|w| {
        let (kind, rate) = crate::weather_setup::precip_of(w);
        w.fog.0 < 600.0 || (kind != 0 && rate > 0.05) || w.clouds.0.trim().to_ascii_lowercase().starts_with("overcast")
    });
    daylight.brightness < 0.75 || gloomy
}
