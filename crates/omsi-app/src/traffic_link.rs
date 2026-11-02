//! What the traffic is told about the player and the LAN buses.

use super::*;

/// Position, heading, half extents and speed (`speed`, m/s) of a vehicle for the AI's
/// obstacle checks: the player's (`player_seen`), a LAN player's (`lan_outlines`).
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
pub(crate) fn own_outlines(player: Option<&omsi_sim::VehicleInstance>, placed: &[Player]) -> Vec<(u32, traffic::PlayerBox)> {
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
    if let Some(v) = player {
        add(v, 0xFFFF_0000, false);
    }
    for (i, q) in placed.iter().enumerate() {
        add(&q.vehicle, 0xFFFE_0000 - (i as u32) * 16, true);
    }
    out
}

/// The player as the traffic sees it this frame: the player's box (`vehicle_outline`), the
/// rest of the player's own vehicles (`own_outlines`), its right of way (its script's
/// `TrafficPriority`), its indicator, and the timetable bus that drives it, if one. One rule for
/// the player's own driving and for the offscreen harness's stand-in (`--follow-line`,
/// `follow_line::LineFollower`), so a harness run sees the player's bus as a real drive does:
/// the player's vehicle is put where the stand-in is every frame (`follow_line::pose_player`,
/// its speed and its rear sections too), so its box and its other sections are the player's
/// as always; its indicator is the stand-in's (`AiState::blinker`: the stand-in is the one at
/// the wheel, and the player's scripts that would show the player's own do not run); and the
/// traffic is told which car that is (`Traffic::stand_in`), so that the drivers see it as the
/// player only (`omsi_traffic::StandIn`).
pub(crate) struct PlayerSeen {
    pub outline: Option<traffic::PlayerBox>,
    pub parts: Vec<(u32, traffic::PlayerBox)>,
    pub priority: bool,
    pub blinker: u8,
    pub stand_in: Option<u64>,
}

/// `PlayerSeen` of the player's vehicle `player`, the vehicles `placed` by hand, and the
/// timetable bus `stand_in` driving the player's vehicle.
pub(crate) fn player_seen(player: Option<&omsi_sim::VehicleInstance>, placed: &[Player], stand_in: Option<&traffic::AiCar>) -> PlayerSeen {
    PlayerSeen {
        outline: player.map(|v| vehicle_outline(v, v.physics.speed)),
        parts: own_outlines(player, placed),
        priority: player.and_then(|v| v.var("TrafficPriority")).is_some_and(|x| x > 0.5),
        blinker: match stand_in {
            Some(c) => c.ai.state.blinker.clamp(0, 3) as u8,
            None => player.map(lan::indicator).unwrap_or(0),
        },
        stand_in: stand_in.filter(|_| player.is_some()).map(|c| c.id),
    }
}

/// Tell the traffic about the player for its next `tick` (`PlayerSeen`; the LAN players'
/// boxes in `Traffic::others` are set before): returns the player's box for that tick.
pub(crate) fn show_player(t: &mut traffic::Traffic, seen: PlayerSeen) -> Option<traffic::PlayerBox> {
    t.others.extend(seen.parts);
    t.player_priority = seen.priority;
    t.player_blinker = seen.blinker;
    t.stand_in = seen.stand_in;
    seen.outline
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
    (t.people, t.people_records, t.walkers) = humans.map(|h| h.for_traffic()).unwrap_or_default();
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
