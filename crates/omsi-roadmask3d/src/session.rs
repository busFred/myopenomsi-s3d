//! One loaded window: the map's tiles round its middle in the game's scene, the overlay over
//! them, the clock and the lighting of the moment shown, the camera modes. The window viewer
//! and the headless shot both drive a session.

use crate::hud::Hud;
use crate::overlay::{Layer, Overlay, ProfileEdge};
use crate::timeline::{FrameSnap, Timeline};
use anyhow::{Context, Result};
use glam::{DVec3, Vec3, Vec4};
use omsi_render::{Camera, Lighting, Renderer, Scene};
use omsi_sim::SimClock;
use openomsi_game::tool::{self, ToolSettings, World};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// How the camera is placed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CamMode {
    /// Flown by hand.
    Free,
    /// The recorded player's seat: its position and heading each frame.
    Seat,
    /// Behind and above vehicle `id`.
    Follow(u64),
    /// Straight down from `height` metres over the window's middle (or where it was panned to).
    TopDown(f64),
}

impl CamMode {
    pub fn name(self) -> String {
        match self {
            CamMode::Free => "free (WASD/QE + right drag, shift fast)".into(),
            CamMode::Seat => "player seat".into(),
            CamMode::Follow(id) => format!("follow #{id}"),
            CamMode::TopDown(h) => format!("top-down {h:.0} m (PgUp/PgDn, wheel)"),
        }
    }

    /// `seat`, `top[:H]`, `follow:N`, `free`.
    pub fn parse(s: &str) -> Option<CamMode> {
        let l = s.trim().to_ascii_lowercase();
        if l == "seat" || l == "player" {
            Some(CamMode::Seat)
        } else if l == "free" {
            Some(CamMode::Free)
        } else if let Some(rest) = l.strip_prefix("top") {
            let h = rest.trim_start_matches(':').trim();
            Some(CamMode::TopDown(if h.is_empty() { 30.0 } else { h.parse().ok()? }))
        } else if let Some(n) = l.strip_prefix("follow:").or_else(|| l.strip_prefix("follow")) {
            Some(CamMode::Follow(n.trim().parse().ok()?))
        } else {
            None
        }
    }
}

pub struct Session {
    pub settings: ToolSettings,
    pub renderer: Renderer,
    pub scene: Scene,
    pub world: World,
    pub timeline: Timeline,
    pub overlay: Overlay,
    pub hud: Hud,
    pub envir: Option<omsi_content::Envir>,
    pub weather: omsi_content::weather::Weather,
    pub clock: SimClock,
    pub lighting: Lighting,
    lit_at: f64,
    pub tiles_loaded: usize,
    pub load_secs: f32,
}

/// `YYYYMMDD` as (year, day of the year).
pub fn parse_date(code: i32) -> Option<(i32, i32)> {
    let (y, m, d) = (code / 10000, (code / 100) % 100, code % 100);
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let lens = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let day: i32 = lens[..(m - 1) as usize].iter().sum::<i32>() + d;
    Some((y, day))
}

impl Session {
    /// Load the map's tiles within `radius_m` of the window's middle into `renderer`'s scene
    /// and build the static overlay.
    pub fn new(renderer: Renderer, root: &Path, settings: ToolSettings, timeline: Timeline, profile: Vec<ProfileEdge>, date: Option<i32>, radius_m: f64) -> Result<Session> {
        let t0 = Instant::now();
        let mut clock = SimClock::default();
        if let Some((y, d)) = date.and_then(parse_date) {
            clock.year = y;
            clock.day_of_year = d;
        }
        clock.time = timeline.first_time();
        let world = tool::open_world(root, &timeline.map, &clock)?;
        let mut scene = renderer.new_scene();
        let ts = omsi_map::tile_size();
        let centre = timeline.centre;
        let tile_of = |p: DVec3| ((p.x / ts).floor() as i32, (p.y / ts).floor() as i32);
        let r_tiles = (radius_m / ts).ceil().max(1.0) as i32;
        let (cx, cy) = tile_of(centre);
        // the square round the centre, cut to the circle the game's streamer loads
        let tiles: Vec<(i32, i32, PathBuf)> = world
            .select_tiles(Some((cx, cy)), Some(r_tiles))
            .into_iter()
            .filter(|t| {
                let (tx, ty) = ((t.0 as f64 + 0.5) * ts, (t.1 as f64 + 0.5) * ts);
                let d = ((tx - centre.x).powi(2) + (ty - centre.y).powi(2)).sqrt();
                d <= radius_m + ts * 0.75
            })
            .collect();
        if tiles.is_empty() {
            anyhow::bail!("no tiles of {} within {radius_m:.0} m of ({:.0}, {:.0})", timeline.map, centre.x, centre.y);
        }
        log::info!("map {}: loading {} tiles round ({:.0}, {:.0})", timeline.map, tiles.len(), centre.x, centre.y);
        let stats = world.build_scene(&renderer, &mut scene, &tiles).context("loading the tiles")?;
        log::info!("{} tiles, {} objects, {} splines, {} textures in {:.1}s", stats.tiles, stats.objects, stats.splines, stats.textures, t0.elapsed().as_secs_f32());
        let envir = omsi_content::Envir::load(&root.join("envir.cfg")).ok();
        let weather = tool::default_weather(root);
        tool::setup_sky(root, &renderer, &mut scene, envir.as_ref(), &weather);
        let daylight = tool::daylight(&clock, envir.as_ref());
        world.set_lamps(&renderer, &mut scene, daylight.lamps_on);
        world.update_night_modes(&renderer, &mut scene, &clock, daylight.brightness);
        let lighting = tool::lighting(&daylight, &weather, &settings);
        renderer.prepare(&mut scene);
        let mut overlay = Overlay::new(DVec3::new(centre.x, centre.y, 0.0));
        overlay.build_static(&renderer, &mut scene, &world, &timeline, &profile);
        Ok(Session {
            settings,
            renderer,
            scene,
            world,
            timeline,
            overlay,
            hud: Hud::new(),
            envir,
            weather,
            lit_at: clock.time,
            clock,
            lighting,
            tiles_loaded: tiles.len(),
            load_secs: t0.elapsed().as_secs_f32(),
        })
    }

    /// Show the moment `day_time` (s): the clock, the sun, the lamps (every half minute).
    pub fn set_time(&mut self, day_time: f64) {
        self.clock.time = day_time;
        if (day_time - self.lit_at).abs() >= 30.0 || self.lighting.sun_dir == Vec3::ZERO {
            let daylight = tool::daylight(&self.clock, self.envir.as_ref());
            self.world.set_lamps(&self.renderer, &mut self.scene, daylight.lamps_on);
            self.world.update_night_modes(&self.renderer, &mut self.scene, &self.clock, daylight.brightness);
            self.lighting = tool::lighting(&daylight, &self.weather, &self.settings);
            self.lit_at = day_time;
        }
    }

    /// The frame's dynamic overlay.
    pub fn lay_frame(&mut self, frame: usize, hovered: Option<u64>, pinned: Option<u64>, mode: CamMode) {
        let f = &self.timeline.frames[frame];
        self.overlay.update_frame(&self.renderer, &mut self.scene, &self.world, &self.timeline, f, hovered, pinned, mode != CamMode::Seat);
        self.world.finish_texture_upgrades(&self.renderer, &mut self.scene);
    }

    /// The camera of `mode` at `frame`; `free` is the hand-flown one (and where a top-down
    /// camera was panned to).
    pub fn camera(&self, mode: CamMode, frame: &FrameSnap, free: &Camera) -> Camera {
        let ground = |p: DVec3| -> f64 {
            match self.world.walk_height(p.x, p.y) {
                Some(g) if (g - p.z).abs() < 1.5 => g,
                _ => p.z,
            }
        };
        let base = Camera { near: 0.1, far: 6000.0, fov_deg: 60.0, roll: 0.0, ..*free };
        match mode {
            CamMode::Free => *free,
            CamMode::Seat => match frame.players.first() {
                Some(p) => {
                    let (s, c) = (p.heading as f64).to_radians().sin_cos();
                    let fwd = DVec3::new(s, c, 0.0);
                    let left = DVec3::new(-c, s, 0.0);
                    let z = ground(p.pos);
                    let eye = DVec3::new(p.pos.x, p.pos.y, z) + fwd * (p.half_len as f64 - 1.6) + left * (p.half_width as f64 * 0.55) + DVec3::Z * 2.35;
                    Camera { position: eye, yaw: p.heading, pitch: free.pitch.clamp(-30.0, 15.0), ..base }
                }
                None => *free,
            },
            CamMode::Follow(id) => match frame.vehicles.iter().find(|v| v.id == id) {
                Some(v) => {
                    let (s, c) = (v.heading as f64).to_radians().sin_cos();
                    let fwd = DVec3::new(s, c, 0.0);
                    let z = ground(v.pos);
                    let eye = DVec3::new(v.pos.x, v.pos.y, z) - fwd * (v.rear as f64 + 11.0) + DVec3::Z * 5.5;
                    Camera { position: eye, yaw: v.heading, pitch: -20.0, ..base }
                }
                None => *free,
            },
            CamMode::TopDown(h) => {
                let g = self.world.walk_height(free.position.x, free.position.y).unwrap_or(self.timeline.centre.z);
                Camera { position: DVec3::new(free.position.x, free.position.y, g + h), yaw: free.yaw, pitch: -89.0, ..base }
            }
        }
    }

    /// The hand camera at the window's middle, looking north-east down at it.
    pub fn start_camera(&self) -> Camera {
        let c = self.timeline.centre;
        let g = self.world.walk_height(c.x, c.y).unwrap_or(c.z);
        Camera { position: DVec3::new(c.x - 25.0, c.y - 25.0, g + 14.0), yaw: 45.0, pitch: -24.0, roll: 0.0, fov_deg: 60.0, near: 0.1, far: 6000.0 }
    }

    /// A world point in window pixels (None behind the camera).
    pub fn project(&self, cam: &Camera, p: DVec3, w: f32, h: f32) -> Option<(f32, f32)> {
        let vp = cam.view_proj(w / h.max(1.0), self.overlay.origin);
        let rel = (p - self.overlay.origin).as_vec3();
        let clip = vp * Vec4::new(rel.x, rel.y, rel.z, 1.0);
        if clip.w <= 0.01 {
            return None;
        }
        let x = (clip.x / clip.w * 0.5 + 0.5) * w;
        let y = (1.0 - (clip.y / clip.w * 0.5 + 0.5)) * h;
        Some((x, y))
    }

    /// Where the cursor's ray meets the ground plane at height `z` (world).
    pub fn cursor_on_plane(&self, cam: &Camera, cx: f32, cy: f32, w: f32, h: f32, z: f64) -> Option<DVec3> {
        let ndc = (cx / w * 2.0 - 1.0, 1.0 - cy / h * 2.0);
        let (o, d) = cam.ray(ndc.0, ndc.1, w / h.max(1.0), self.overlay.origin);
        let zr = (z - self.overlay.origin.z) as f32;
        if d.z.abs() < 1e-5 {
            return None;
        }
        let t = (zr - o.z) / d.z;
        if t <= 0.0 {
            return None;
        }
        let p = o + d * t;
        Some(self.overlay.origin + DVec3::new(p.x as f64, p.y as f64, p.z as f64))
    }

    /// The vehicle under the cursor (its ground box), and the person (within 0.6 m of the
    /// capsule's axis at the ground or at hip height).
    pub fn pick(&self, cam: &Camera, frame: &FrameSnap, cx: f32, cy: f32, w: f32, h: f32) -> (Option<u64>, Option<usize>) {
        let mut best: Option<(f64, u64)> = None;
        for v in &frame.vehicles {
            let Some(hit) = self.cursor_on_plane(cam, cx, cy, w, h, v.pos.z + 0.75) else { continue };
            let rel = (hit - v.pos).truncate();
            let (s, c) = (v.heading as f64).to_radians().sin_cos();
            let along = rel.x * s + rel.y * c;
            let across = rel.x * c - rel.y * s;
            if along <= v.front as f64 + 0.2 && along >= -(v.rear as f64) - 0.2 && across.abs() <= v.half_width as f64 + 0.2 {
                let d = (hit - cam.position).length();
                if best.map(|b| d < b.0).unwrap_or(true) {
                    best = Some((d, v.id));
                }
            }
        }
        let mut person: Option<(f64, usize)> = None;
        for (i, p) in frame.people.iter().enumerate() {
            for dz in [0.0, 0.9] {
                let Some(hit) = self.cursor_on_plane(cam, cx, cy, w, h, p.pos.z + dz) else { continue };
                let d = (hit - p.pos).truncate().length();
                if d < 0.6 && person.map(|b| d < b.0).unwrap_or(true) {
                    person = Some((d, i));
                }
            }
        }
        (best.map(|b| b.1), person.map(|p| p.1))
    }

    /// The on-screen text of a frame: the status line, the labels, the pinned vehicle's
    /// fields, the legend, the hover tip.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_hud(&mut self, cam: &Camera, frame_ix: usize, mode: CamMode, speed: f64, playing: bool, pinned: Option<u64>, hovered: (Option<u64>, Option<usize>), cursor: Option<(f32, f32)>, legend: bool, w: f32, h: f32, keys_help: bool) {
        let frame = self.timeline.frames[frame_ix].clone();
        let px = (h / 60.0).clamp(12.0, 22.0);
        let n = self.timeline.frames.len();
        let layers: String = Layer::ALL.iter().enumerate().map(|(i, _)| if self.overlay.layers_on[i] { char::from(b'1' + i as u8) } else { '·' }).collect();
        let mut lines = vec![
            format!(
                "{}  frame {}/{}  {}{:.2}x   camera: {}   layers {}{}",
                hms_frac(frame.day_time),
                frame_ix + 1,
                n,
                if playing { "▶ " } else { "⏸ " },
                speed,
                mode.name(),
                layers,
                if self.overlay.heights { "  heights" } else { "" }
            ),
            format!("{} · {} vehicles, {} people, {} lanes · player {} · tiles {} ({:.1}s) · cam ({:.0}, {:.0}, {:.1}) yaw {:.0} pitch {:.0}", self.timeline.map, frame.vehicles.len(), frame.people.len(), self.overlay.counts.lanes, frame.players.first().map(|p| format!("{:.1} km/h", p.speed * 3.6)).unwrap_or("-".into()), self.tiles_loaded, self.load_secs, cam.position.x, cam.position.y, cam.position.z, cam.yaw, cam.pitch),
        ];
        if keys_help {
            lines.push("space play · , . step · [ ] ±5 s · - = speed · Home start · L legend · H heights · P / click pin · N next vehicle · F1 free F2 seat F3 follow F4 top · PgUp/PgDn height · F5 hide this · Esc quit".into());
            lines.push(Layer::ALL.iter().enumerate().map(|(i, l)| format!("{} {}{}", i + 1, l.name(), if self.overlay.layers_on[i] { "" } else { " (off)" })).collect::<Vec<_>>().join(" · "));
        }
        if self.overlay.heights {
            lines.push(format!("heights: the kerb and road edges tinted by their height over the spline, blue {:.2} m .. red {:.2} m", self.overlay.z_range.0, self.overlay.z_range.1));
        }
        self.hud.panel(&self.renderer, &mut self.scene, &lines, px, 8.0, 8.0, [0.92, 0.94, 0.97, 1.0]);

        // labels over the vehicles
        if self.overlay.layers_on[Layer::Labels as usize] {
            for v in &frame.vehicles {
                let height = match v.class.as_str() {
                    "Bus" | "ScheduledBus" | "Truck" | "Rail" => 3.0,
                    _ => 1.5,
                };
                let top = DVec3::new(v.pos.x, v.pos.y, v.pos.z + height + 0.3);
                if let Some((x, y)) = self.project(cam, top, w, h) {
                    if x < -50.0 || y < -50.0 || x > w + 50.0 || y > h + 50.0 {
                        continue;
                    }
                    let dist = (top - cam.position).length() as f32;
                    if dist > 90.0 && Some(v.id) != pinned {
                        continue;
                    }
                    let text = if v.why.is_empty() || v.state == crate::geometry::VehicleState::Moving { format!("#{}", v.id) } else { format!("#{} {}", v.id, v.why) };
                    let colour = if Some(v.id) == pinned { [1.0, 0.3, 0.85, 1.0] } else { [1.0, 1.0, 1.0, 1.0] };
                    // (smaller with distance, down to two thirds)
                    let size = px * 0.85 * (1.0 - 0.33 * (dist / 90.0).min(1.0));
                    self.hud.label(&self.renderer, &mut self.scene, &text, size, x, y - size, colour);
                }
            }
        }

        // the pinned vehicle's fields
        if let Some(id) = pinned {
            let rows = match frame.vehicles.iter().find(|v| v.id == id) {
                Some(v) => vec![
                    format!("pinned #{} {} {}", v.id, v.class, v.type_name),
                    format!("lane {} s {:.1} m  speed {:.1} km/h  heading {:.0}", v.lane, v.s, v.speed * 3.6, v.heading),
                    format!("box front {:.2} rear {:.2} half width {:.2} m", v.front, v.rear, v.half_width),
                    format!("why: {}  standing {:.1} s", if v.why.is_empty() { "-" } else { v.why }, v.standing_for),
                    format!("lead {}  waits for {}  stop in {}", v.lead_id.map(|x| format!("#{x}")).unwrap_or("-".into()), v.wait_for.map(id_name).unwrap_or("-".into()), v.stop_in.map(|x| format!("{x:.1} m")).unwrap_or("-".into())),
                    format!("way ahead: {}", if v.ahead.is_empty() { "-".to_string() } else { v.ahead.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(" › ") }),
                    format!("pos ({:.1}, {:.1}, {:.2})", v.pos.x, v.pos.y, v.pos.z),
                ],
                None => vec![format!("pinned #{id}: not in this frame")],
            };
            let (pw, _) = self.hud.panel_size(&rows, px);
            self.hud.panel(&self.renderer, &mut self.scene, &rows, px, w - 8.0 - pw, 8.0, [1.0, 0.75, 0.95, 1.0]);
        }

        // the hover tip
        if let Some((cx, cy)) = cursor {
            let mut tip = Vec::new();
            if let Some(id) = hovered.0 {
                if let Some(v) = frame.vehicles.iter().find(|v| v.id == id) {
                    tip.push(format!("#{} {} {:.1} km/h {}", v.id, v.class, v.speed * 3.6, if v.why.is_empty() { "" } else { v.why }));
                    tip.push(format!("lane {} s {:.1}  width {:.2} m  lead {}", v.lane, v.s, 2.0 * v.half_width, v.lead_id.map(|x| format!("#{x}")).unwrap_or("-".into())));
                }
            }
            if let Some(i) = hovered.1 {
                if let Some(p) = frame.people.get(i) {
                    tip.push(format!("person: {}{}{}", p.activity, if p.at_kerb { " · at kerb" } else { "" }, if p.on_crossing { " · on crossing" } else { "" }));
                    if !p.decision.is_empty() {
                        tip.push(format!("decision: {}", p.decision));
                    }
                    tip.push(format!("pos ({:.1}, {:.1}, {:.2})", p.pos.x, p.pos.y, p.pos.z));
                }
            }
            if !tip.is_empty() {
                self.hud.panel(&self.renderer, &mut self.scene, &tip, px * 0.9, cx + 14.0, cy + 14.0, [1.0, 1.0, 1.0, 1.0]);
            }
        }

        if legend {
            let rows = self.overlay.legend();
            self.hud.legend(&self.renderer, &mut self.scene, &rows, px * 0.85, 8.0, h * 0.2);
        }
    }
}

/// A vehicle id as text: a player's (`PLAYER_ID_BASE` + its id) as "player N".
pub fn id_name(id: u64) -> String {
    match id.checked_sub(omsi_traffic::sim::PLAYER_ID_BASE) {
        Some(n) => format!("player {n}"),
        None => format!("#{id}"),
    }
}

/// `HH:MM:SS.t` of a day time.
pub fn hms_frac(t: f64) -> String {
    let s = t.max(0.0);
    let whole = s.floor() as i64;
    format!("{:02}:{:02}:{:02}.{}", whole / 3600, (whole / 60) % 60, whole % 60, ((s - whole as f64) * 10.0) as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_and_times() {
        assert_eq!(parse_date(20240301), Some((2024, 61)));
        assert_eq!(parse_date(20230301), Some((2023, 60)));
        assert_eq!(parse_date(20231231), Some((2023, 365)));
        assert_eq!(parse_date(20231301), None);
        assert_eq!(hms_frac(8.0 * 3600.0 + 14.0 * 60.0 + 3.25), "08:14:03.2");
        assert_eq!(id_name(12), "#12");
        assert_eq!(id_name(omsi_traffic::sim::PLAYER_ID_BASE), "player 0");
    }

    #[test]
    fn camera_modes_parse() {
        assert_eq!(CamMode::parse("seat"), Some(CamMode::Seat));
        assert_eq!(CamMode::parse("top"), Some(CamMode::TopDown(30.0)));
        assert_eq!(CamMode::parse("top:45"), Some(CamMode::TopDown(45.0)));
        assert_eq!(CamMode::parse("follow:12"), Some(CamMode::Follow(12)));
        assert_eq!(CamMode::parse("1,2,3,4,5"), None);
    }
}
