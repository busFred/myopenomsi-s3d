//! The game's scene for standalone tools: what `omsi-roadmask3d` needs to draw a recorded
//! traffic window inside the real map - opening a map and its tiles with the game's scene
//! code ([`World`]), the graphics instance, the settings' render options and window size,
//! the sky, the weather and the lighting of a moment of the day, the `--cam` syntax.
//!
//! Nothing in the game depends on a tool: this module only hands out what the game has.
//! A tool writes nothing under `~/.openomsi` through it (the settings are read only).

pub use crate::scene::{LoadStats, World};
use anyhow::{Context, Result};
use omsi_render::{Camera, Lighting, RenderOptions, Renderer, Scene};
use omsi_sim::{Daylight, SimClock};
use std::path::Path;

/// The native graphics instance, as the game opens it for its window and its pictures.
pub fn graphics_instance() -> wgpu::Instance {
    crate::startup::graphics_instance()
}

/// The parts of the game's `settings.cfg` a tool draws with.
#[derive(Debug, Clone)]
pub struct ToolSettings {
    pub render: RenderOptions,
    /// `resolution=WxH` where the settings ask for one.
    pub resolution: Option<(u32, u32)>,
    pub shadows: bool,
    pub detail_textures: bool,
    pub vsync: bool,
    pub enhanced: bool,
    /// `view_distance=<m>` or the game's default: how far round the camera the window
    /// streams the map's tiles.
    pub view_distance: f64,
}

/// The settings as they are (read, never written), with the graphics statics of the
/// renderer set as the game sets them (enhanced, classic, clouds).
pub fn settings() -> ToolSettings {
    let s = crate::settings::Settings::load();
    crate::launcher_statics();
    ToolSettings {
        render: s.render_options(),
        resolution: crate::settings::Settings::resolution(),
        shadows: s.shadows,
        detail_textures: s.detail_textures,
        vsync: s.vsync,
        enhanced: s.enhanced,
        view_distance: crate::settings::view_distance().unwrap_or(crate::settings::DEFAULT_VIEW_DISTANCE).max(omsi_map::tile_size()),
    }
}

/// Open map `name` (the folder under `maps`) of the installation at `root` at the clock's
/// date, its season as that date has it. The tiles are loaded with
/// [`World::select_tiles`] and [`World::build_scene`].
pub fn open_world(root: &Path, name: &str, clock: &SimClock) -> Result<World> {
    let cfg = omsi_cfg::resolve_path(root, &format!("maps\\{name}\\global.cfg"));
    if !omsi_cfg::vfs::is_file(&cfg) {
        anyhow::bail!("map {name}: no global.cfg at {}", cfg.display());
    }
    let world = World::open(root, &cfg, clock.date_code()).with_context(|| format!("opening {}", cfg.display()))?;
    *world.start_clock.lock() = clock.clone();
    let kind = world.global.season_kind(clock.day_of_year);
    omsi_texture::set_season_folder(omsi_map::global::Season::folder(kind).map(|f| f.to_string()));
    Ok(world)
}

/// The clear weather the game starts in without a choice (`Weather/#CAVOK.owt`, a few fair-
/// weather clouds), or a clear sky when the file is not there.
pub fn default_weather(root: &Path) -> omsi_content::weather::Weather {
    match omsi_content::weather::Weather::load(&omsi_cfg::resolve_path(root, "Weather/#CAVOK.owt")) {
        Ok(mut w) => {
            if w.clouds.0.trim().starts_with("-1") {
                w.clouds = ("Cumulus 1".into(), 100.0);
            }
            w
        }
        Err(e) => {
            log::warn!("weather #CAVOK: {e}");
            omsi_content::weather::Weather { fog: (50000.0, 1.0), ..Default::default() }
        }
    }
}

/// The sky textures and the cloud field of `weather` into the scene (envir.cfg's pictures).
pub fn setup_sky(root: &Path, renderer: &Renderer, scene: &mut Scene, envir: Option<&omsi_content::Envir>, weather: &omsi_content::weather::Weather) {
    crate::weather_setup::setup_sky_at(root, renderer, scene, envir, Some(weather));
}

/// The sun and the three envir lights at the clock's moment.
pub fn daylight(clock: &SimClock, envir: Option<&omsi_content::Envir>) -> Daylight {
    Daylight::compute(clock, envir)
}

/// The lighting of a frame at the clock's moment in `weather`, as the game's pictures are
/// lit (without a player's vehicle: no cab box, no puddle capture).
pub fn lighting(daylight: &Daylight, weather: &omsi_content::weather::Weather, settings: &ToolSettings) -> Lighting {
    let mut l = crate::weather_setup::weather_lighting(daylight, weather, crate::weather_setup::cloud_drift_at(weather, 0.0), 0.0, settings.shadows);
    l.detail = settings.detail_textures;
    l
}

/// `x,y,z,yaw,pitch[,fov]` as the game's `--cam` reads it.
pub fn parse_cam(s: &str) -> Result<Camera> {
    crate::cli::parse_cam(s)
}
