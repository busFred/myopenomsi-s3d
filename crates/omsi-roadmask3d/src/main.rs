//! omsi-roadmask3d: a recorded traffic window replayed inside the real 3D scene, with a
//! segmentation-mask overlay over it - the lanes' footprints by kind, the map's kerb and
//! road edges from the spline profiles, the recorded vehicles as translucent blocks by
//! state, the people as capsules by decision - and a free camera.
//!
//! ```text
//! omsi-roadmask3d [<window.rec>] [--list] [--at HH:MM] [--root <omsi root>] [--radius M]
//!                 [--center x,y] [--date YYYYMMDD] [--start HH:MM:SS]
//!                 [--shot <png> --t HH:MM:SS --cam seat|top[:H]|follow:N|x,y,z,yaw,pitch[,fov] [--size WxH]]
//! ```
//!
//! Without a recording a file dialog (or `--list`) picks one under `~/.openomsi/traffic`;
//! a recording longer than three minutes is cut to 90 s round `--at` first. The map's tiles
//! round the window's middle are loaded with the game's scene code within the game's view
//! distance (`--radius` metres otherwise) and drawn by the game's renderer in a window at
//! the settings' resolution. `--shot` renders pictures headlessly instead (`--cam` per shot,
//! in order; `--t` the moment). Nothing under `~/.openomsi` is written but the output folder
//! `~/.openomsi/traffic/mask3d/` (the shots, a cut). `OMSI_MASK3D_DEBUG=1` lists the lanes
//! near the window's middle against the scene's heights on stderr. Standalone: no game
//! binary depends on this crate.

mod geometry;
mod hud;
mod overlay;
mod session;
mod timeline;
mod viewer;

use anyhow::{Context, Result};
use omsi_roadmask::{hms, mapdata, profile, recording};
use overlay::ProfileEdge;
use session::{CamMode, Session};
use std::path::{Path, PathBuf};
use timeline::Timeline;

const HELP: &str = "omsi-roadmask3d: a recorded traffic window replayed inside the real 3D scene with a mask overlay
(lanes by kind, the map's kerb and road edges, vehicles as blocks by state, people as capsules) and a free camera.

usage: omsi-roadmask3d [<recording.rec>] [options]

  <recording.rec>   a traffic recording (a `traffic_cases cut` window, or a session recording).
                    None given: a file dialog opens in ~/.openomsi/traffic (or see --list).

options:
  --list            list the recordings in ~/.openomsi/traffic (newest first) and read a number
  --at HH:MM        a recording longer than 3 minutes is cut to 90 s round this time first
                    (the cut .rec goes to ~/.openomsi/traffic/mask3d/)
  --root <folder>   the OMSI 2 folder (else OMSI_ROOT, else the launcher's launcher.json)
  --radius M        load the map's tiles within M metres of the window's middle (default: the
                    game's view distance setting, 1200 m)
  --center x,y      the window's middle (world m; default: where the player was at its middle)
  --date YYYYMMDD   the day shown: the map's chrono folders, the season and the sun (default 1989-05-30)
  --start HH:MM:SS  the window opens at this game time (default: the recording's first frame)
  --view MODE       the window's first camera: free, seat, top[:H], follow:N (default seat)
  --shot <png>      render a picture headlessly instead of opening a window (repeatable)
  --cam MODE        the camera of the shot(s), in order (the last one repeats): seat, top[:H],
                    follow:N, or x,y,z,yaw,pitch[,fov] as the game's --cam; default seat
  --t HH:MM:SS      the moment of the shot(s) (default: the window's middle)
  --size WxH        the shot's size (default: the settings' resolution, else 1600x900)
  --adapter NAME    draw on the graphics adapter whose name contains NAME (e.g. Intel): a card
                    whose memory other programs hold runs out of it while the map loads
  -h, --help        this text

keys: space play/pause · , . step a frame · [ ] ±5 s · - = speed 0.25x-4x · Home/End
      1-7 layers (footprints, lane edges, kerb/road edges, stop lines, vehicles, people, labels)
      L legend · H heights · P / left click pin the hovered vehicle · N next vehicle · F5 help line
      F1 free (WASD/QE, shift fast, right drag looks) · F2 player seat · F3 follow · F4 top-down (PgUp/PgDn, wheel)
";

const FLAGS: &[&str] = &["--list", "--help", "-h"];

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).target(env_logger::Target::Stderr).init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print!("{HELP}");
        return;
    }
    if let Err(e) = run(&args) {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn opt<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(|s| s.as_str())
}

fn opts<'a>(args: &'a [String], name: &str) -> Vec<&'a str> {
    args.iter().enumerate().filter(|(_, a)| *a == name).filter_map(|(i, _)| args.get(i + 1)).map(|s| s.as_str()).collect()
}

fn positional(args: &[String]) -> Vec<&String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if FLAGS.contains(&args[i].as_str()) {
            i += 1;
            continue;
        }
        if args[i].starts_with("--") {
            i += 2;
            continue;
        }
        out.push(&args[i]);
        i += 1;
    }
    out
}

fn ask(prompt: &str) -> Result<String> {
    use std::io::Write;
    print!("{prompt}");
    std::io::stdout().flush()?;
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line)? == 0 {
        anyhow::bail!("no input");
    }
    Ok(line)
}

/// The output folder: the only place under ~/.openomsi this tool writes.
fn out_dir() -> PathBuf {
    mapdata::data_dir().join("traffic").join("mask3d")
}

/// The recording to show: the argument, else the dialog or the list; a long one cut to a
/// window round `--at`.
fn choose_recording(args: &[String]) -> Result<PathBuf> {
    let traffic_dir = mapdata::data_dir().join("traffic");
    let rec = match positional(args).first() {
        Some(p) => PathBuf::from(p),
        None if args.iter().any(|a| a == "--list") => {
            let list = recording::list_recordings(&traffic_dir);
            if list.is_empty() {
                anyhow::bail!("no .rec files under {}", traffic_dir.display());
            }
            println!("recordings in {} (newest first):", traffic_dir.display());
            for (i, (p, _, size)) in list.iter().enumerate() {
                let name = p.strip_prefix(&traffic_dir).unwrap_or(p).display();
                match recording::header_info(p) {
                    Ok(h) => println!("{:>3}  {:>8.1} MB  {:<14} {:>6} lanes  started {}{}  {}", i + 1, *size as f64 / 1e6, h.map, h.lanes, hms(h.started), h.trigger.as_deref().map(|t| format!(" ({t})")).unwrap_or_default(), name),
                    Err(e) => println!("{:>3}  {:>8.1} MB  (unreadable: {e})  {}", i + 1, *size as f64 / 1e6, name),
                }
            }
            let n: usize = ask(&format!("number (1-{}): ", list.len()))?.trim().parse()?;
            list.get(n.wrapping_sub(1)).map(|x| x.0.clone()).context("no such number")?
        }
        None => {
            println!("choose a recording (a file dialog; --list for a numbered list instead)");
            recording::pick_recording(&traffic_dir).context("no recording chosen")?
        }
    };
    if !rec.is_file() {
        anyhow::bail!("{} is not a file", rec.display());
    }
    let (first, last, long) = recording::frame_span(&rec, recording::LONG_SECS).map_err(|e| anyhow::anyhow!("{e}"))?;
    if !long {
        return Ok(rec);
    }
    let at = match opt(args, "--at") {
        Some(t) => recording::parse_hhmm(t).map_err(|e| anyhow::anyhow!("{e}"))?,
        None => {
            println!("{} runs from {} (game time) past {}, more than {} minutes: a 90 s window is cut first", rec.display(), hms(first), hms(last), (recording::LONG_SECS / 60.0) as i64);
            recording::parse_hhmm(&ask("the window's middle, HH:MM in game time: ")?).map_err(|e| anyhow::anyhow!("{e}"))?
        }
    };
    std::fs::create_dir_all(out_dir())?;
    let stem = rec.file_stem().and_then(|s| s.to_str()).unwrap_or("window").to_string();
    let cut = out_dir().join(format!("{stem}-at{}.rec", &hms(at).replace(':', "")[..4]));
    println!("cutting {} .. {} of {} into {} ...", hms(at - recording::WINDOW_HALF), hms(at + recording::WINDOW_HALF), rec.display(), cut.display());
    let (a, b) = recording::cut_window(&rec, at, &cut).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("cut: frames {} .. {}", hms(a), hms(b));
    Ok(cut)
}

/// The visual kerb and road edges of the splines behind the window's lanes, from their
/// `.sli` cross-sections (as omsi-roadmask derives them), laid along their placements.
fn profile_edges(root: &Path, tl: &Timeline, date: Option<i32>) -> Vec<ProfileEdge> {
    let mut map = match mapdata::MapData::open(root.to_path_buf(), &tl.map, date) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("map data: {e} (no kerb and road edges)");
            return Vec::new();
        }
    };
    for lane in &tl.lanes {
        let l = &tl.net.lanes[lane.index];
        if let Some(k) = l.key {
            map.resolve(lane.index, k, l.source, &l.name);
        }
    }
    let mut out = Vec::new();
    let mut without = 0usize;
    for si in &map.splines {
        let Some(def) = &si.sli else {
            without += 1;
            continue;
        };
        let side = if si.map.mirror { -1.0 } else { 1.0 };
        let n = profile::station_count(&si.curve);
        let cs = profile::cross_section(def);
        for e in &cs.edges {
            if e.kind == "seam" {
                continue;
            }
            let pts = profile::lay_along(&si.curve, e.x as f64, e.z1 as f64, side, n);
            out.push(ProfileEdge { kind: e.kind, z: e.z1, points: pts });
        }
    }
    println!("spline instances behind the lanes: {} ({} without their .sli); {} edge lines", map.splines.len(), without, out.len());
    for p in map.problems.iter().take(8) {
        println!("  {p}");
    }
    out
}

/// A renderer without a window: the graphics card first, then the processor's graphics,
/// a software renderer last (a card whose memory other programs hold is passed over).
fn headless_renderer(instance: &wgpu::Instance, options: &omsi_render::RenderOptions, name: Option<&str>) -> Result<omsi_render::Renderer> {
    let mut adapters: Vec<wgpu::Adapter> = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    if let Some(n) = name {
        adapters.retain(|a| a.get_info().name.to_ascii_lowercase().contains(&n.to_ascii_lowercase()));
        if adapters.is_empty() {
            anyhow::bail!("no graphics adapter named like {n:?}");
        }
    }
    adapters.sort_by_key(|a| match a.get_info().device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu | wgpu::DeviceType::Other => 2,
        wgpu::DeviceType::Cpu => 3,
    });
    let mut failures = Vec::new();
    for adapter in adapters {
        let info = adapter.get_info();
        let what = format!("{} ({:?})", info.name, info.backend);
        match pollster::block_on(omsi_render::Renderer::new_on(adapter, None, Some(wgpu::TextureFormat::Rgba8UnormSrgb), options.clone())) {
            Ok(r) => {
                if !failures.is_empty() {
                    eprintln!("graphics: drawing on {what}; before it: {}", failures.join("; "));
                }
                return Ok(r);
            }
            Err(e) => failures.push(format!("{what}: {e:#}")),
        }
    }
    anyhow::bail!("no graphics device could be opened: {}", failures.join("; "))
}

fn run(args: &[String]) -> Result<()> {
    let rec = choose_recording(args)?;
    let radius_opt: Option<f64> = opt(args, "--radius").map(|s| s.parse()).transpose().context("--radius M")?;
    let date: Option<i32> = opt(args, "--date").map(|s| s.parse()).transpose().context("--date YYYYMMDD")?;
    let centre = match opt(args, "--center") {
        Some(c) => {
            let (a, b) = c.split_once(',').context("--center x,y")?;
            Some((a.trim().parse()?, b.trim().parse()?))
        }
        None => None,
    };

    // the OMSI root and the content roots, as the game has them; the settings read only
    let root = mapdata::omsi_root(opt(args, "--root")).map_err(|e| anyhow::anyhow!("{e}"))?;
    if !root.is_dir() {
        anyhow::bail!("OMSI root {} is not a folder", root.display());
    }
    if let Some(c) = mapdata::game_content_dir() {
        println!("content folder (mods): {}", c.display());
        omsi_cfg::add_content_root(c);
    }
    omsi_cfg::add_content_root(root.clone());
    println!("OMSI root: {}", root.display());
    let settings = openomsi_game::tool::settings();
    println!("settings.cfg: resolution {}, view distance {:.0} m, enhanced {}, shadows {} (read, not written)", settings.resolution.map(|(w, h)| format!("{w}x{h}")).unwrap_or("automatic".into()), settings.view_distance, settings.enhanced, settings.shadows);
    let radius_m = radius_opt.unwrap_or(settings.view_distance);

    println!("reading {} ...", rec.display());
    let tl = Timeline::load(&rec, 150.0, centre).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("map {} · {} frames {} .. {} · centre ({:.1}, {:.1}) · {} lanes over ({:.0}, {:.0})..({:.0}, {:.0}), {} kerb spots", tl.map, tl.frames.len(), hms(tl.first_time()), hms(tl.last_time()), tl.centre.x, tl.centre.y, tl.lanes.len(), tl.extent.0.x, tl.extent.0.y, tl.extent.1.x, tl.extent.1.y, tl.kerb_spots.len());
    let profile = profile_edges(&root, &tl, date);
    let title = format!("{} · road mask 3D", rec.file_stem().and_then(|s| s.to_str()).unwrap_or("window"));

    let shots = opts(args, "--shot");
    if shots.is_empty() {
        let start_time = opt(args, "--start").map(recording::parse_hhmm).transpose().map_err(|e| anyhow::anyhow!("{e}"))?;
        let start_mode = opt(args, "--view").map(|v| CamMode::parse(v).context("--view free|seat|top[:H]|follow:N")).transpose()?.unwrap_or(CamMode::Seat);
        return viewer::Viewer::run(viewer::Pending { root, settings, timeline: Some(tl), profile, date, radius_m, title, start_time, start_mode, adapter: opt(args, "--adapter").map(str::to_string) });
    }

    // headless pictures
    let (w, h) = match opt(args, "--size") {
        Some(s) => {
            let (a, b) = s.split_once('x').context("--size WxH")?;
            (a.parse()?, b.parse()?)
        }
        None => settings.resolution.unwrap_or((1600, 900)),
    };
    let t = match opt(args, "--t") {
        Some(t) => recording::parse_hhmm(t).map_err(|e| anyhow::anyhow!("{e}"))?,
        None => (tl.first_time() + tl.last_time()) * 0.5,
    };
    let cams = opts(args, "--cam");
    let instance = openomsi_game::tool::graphics_instance();
    let renderer = headless_renderer(&instance, &settings.render, opt(args, "--adapter"))?;
    println!("adapter: {}", renderer.adapter_name);
    println!("loading the map round the window (radius {radius_m:.0} m) ...");
    let mut s = Session::new(renderer, &root, settings, tl, profile, date, radius_m)?;
    println!("{} tiles in {:.1} s; {} lanes, {} stop lines, {} kerb spots, {} profile edges", s.tiles_loaded, s.load_secs, s.overlay.counts.lanes, s.overlay.counts.stop_lines, s.overlay.counts.kerb_spots, s.overlay.counts.profile_edges);
    let t = t.clamp(s.timeline.first_time(), s.timeline.last_time());
    let frame_ix = s.timeline.frame_at(t);
    s.set_time(t);
    for (i, shot) in shots.iter().enumerate() {
        let cam_text = cams.get(i).or(cams.last()).copied().unwrap_or("seat");
        let frame = s.timeline.frames[frame_ix].clone();
        let free = s.start_camera();
        let (mode, cam) = match CamMode::parse(cam_text) {
            Some(m) => {
                let mut free = free;
                if let CamMode::TopDown(_) = m {
                    free.position = s.timeline.centre;
                    free.yaw = 0.0;
                }
                (m, s.camera(m, &frame, &free))
            }
            None => (CamMode::Free, openomsi_game::tool::parse_cam(cam_text)?),
        };
        let pinned = if let CamMode::Follow(id) = mode { Some(id) } else { None };
        s.lay_frame(frame_ix, None, pinned, mode);
        s.hud.begin(&s.renderer, &mut s.scene);
        s.draw_hud(&cam, frame_ix, mode, 1.0, false, pinned, (None, None), None, true, w as f32, h as f32, false);
        // (the first picture warms the textures up; the second is the one kept)
        let lighting = s.lighting.clone();
        let _ = s.renderer.render_to_image(&mut s.scene, w, h, &cam, &lighting)?;
        s.world.finish_texture_upgrades(&s.renderer, &mut s.scene);
        let pixels = s.renderer.render_to_image(&mut s.scene, w, h, &cam, &lighting)?;
        let path = {
            let p = PathBuf::from(shot);
            if p.is_absolute() || p.components().count() > 1 { p } else { out_dir().join(p) }
        };
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        image::save_buffer(&path, &pixels, w, h, image::ColorType::Rgba8)?;
        println!("{}: {w}x{h} at {} from {} - camera ({:.1}, {:.1}, {:.2}) yaw {:.0} pitch {:.0}", path.display(), hms(t), mode.name(), cam.position.x, cam.position.y, cam.position.z, cam.yaw, cam.pitch);
    }
    Ok(())
}
