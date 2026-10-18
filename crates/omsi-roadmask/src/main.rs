//! omsi-roadmask: the road and pavement boundaries of a recorded traffic window, as the
//! simulation has them (the lanes' footprints) and as the map draws them (the kerb and
//! edge lines of the splines' `.sli` cross-sections, the crossing objects' paths and mesh
//! outlines), with the recorded vehicles and people replayed over them.
//!
//! ```text
//! omsi-roadmask [<recording.rec>] [<out.html>] [--list] [--at HH:MM] [--no-open] [--root <omsi root>] [--center x,y] [--radius N] [--fps N] [--date YYYYMMDD] [--title T]
//! ```
//!
//! From a terminal without arguments: a file dialog (or `--list`, a numbered list) picks
//! the recording under `~/.openomsi/traffic`, the page goes to
//! `~/.openomsi/traffic/mask/<stem>-roadmask.html` and opens in the browser; a recording
//! longer than three minutes is cut to a 90 s window round `--at` (asked for) first.
//!
//! The OMSI root is `--root`, else `OMSI_ROOT`, else the launcher's `root`
//! (`~/.openomsi/launcher.json`); the game's `settings.cfg` is read and reported, never
//! written. `--date` applies the map's chrono folders active on that day (a recording does
//! not carry the date; without it, none). Standalone: the game binaries do not depend on
//! this crate.

mod mapdata;
mod profile;
mod recording;

use glam::DVec3;
use omsi_sim::traffic::{arc_point, LaneKind};
use omsi_traffic::record::{PersonRecord, Record};
use omsi_traffic::replay::Replay;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

const PAGE: &str = include_str!("roadmask.html");

/// A person's body radius (m), as the traffic has it.
const PERSON_RADIUS: f64 = 0.28;
/// Lanes and map geometry are taken this far beyond the radius (m).
const MARGIN: f64 = 30.0;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const HELP: &str = "omsi-roadmask: the road and pavement boundaries of a recorded traffic window as one HTML page -
the lanes' footprints as the simulation has them, and the map's visual kerb and road edges (from the
splines' .sli cross-sections and the crossing objects' meshes), with the recording replayed over them.

usage: omsi-roadmask [<recording.rec>] [<out.html>] [options]

  <recording.rec>   a traffic recording (a `traffic_cases cut` window, or a session recording).
                    None given: a file dialog opens in ~/.openomsi/traffic (or see --list).
  <out.html>        the page; default ~/.openomsi/traffic/mask/<recording stem>-roadmask.html

options:
  --list            list the recordings in ~/.openomsi/traffic (newest first) and read a number
  --at HH:MM        a recording longer than 3 minutes is cut to 90 s round this time first
                    (asked for when not given; the cut .rec goes beside the page)
  --no-open         do not open the page in the browser afterwards
  --root <folder>   the OMSI 2 folder (else OMSI_ROOT, else the launcher's launcher.json)
  --center x,y      the window's centre (world m; default: where the player was at its middle)
  --radius N        metres round the centre (100)
  --fps N           frames per second sampled (5)
  --date YYYYMMDD   apply the map's chrono folders active on that day (none without it)
  --title T         the page's title
  -h, --help        this text

Nothing under ~/.openomsi but the page (and the cut) is written. Keys on the page: space play/pause,
arrows step, M layers (lanes / visual edges / both), H height ramp.
";

/// Options that take no value.
const FLAGS: &[&str] = &["--list", "--no-open", "--help", "-h"];

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .target(env_logger::Target::Stderr)
        .init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print!("{HELP}");
        return;
    }
    let positional: Vec<&String> = {
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
    };
    if positional.len() > 2 {
        eprint!("{HELP}");
        std::process::exit(2);
    }
    let code = match prepare(&positional, &args)
        .and_then(|(rec, out)| run(&rec, &out, &args).map(|_| out))
    {
        Ok(out) => {
            if !args.iter().any(|a| a == "--no-open") {
                match recording::open_in_browser(&out) {
                    Ok(()) => println!("opened in the browser (--no-open leaves it)"),
                    Err(e) => println!("could not open the browser ({e}): open {}", out.display()),
                }
            }
            0
        }
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    };
    std::process::exit(code);
}

/// The recording to render and the page to write: the arguments, else the dialog or the
/// list; a long recording cut to a window first.
fn prepare(
    positional: &[&String],
    args: &[String],
) -> Res<(std::path::PathBuf, std::path::PathBuf)> {
    let traffic_dir = mapdata::data_dir().join("traffic");
    let rec = match positional.first() {
        Some(p) => std::path::PathBuf::from(p),
        None if args.iter().any(|a| a == "--list") => {
            let list = recording::list_recordings(&traffic_dir);
            if list.is_empty() {
                return Err(format!("no .rec files under {}", traffic_dir.display()).into());
            }
            println!("recordings in {} (newest first):", traffic_dir.display());
            for (i, (p, _, size)) in list.iter().enumerate() {
                let name = p.strip_prefix(&traffic_dir).unwrap_or(p).display();
                match recording::header_info(p) {
                    Ok(h) => println!(
                        "{:>3}  {:>8.1} MB  {:<14} {:>6} lanes  started {}{}  {}",
                        i + 1,
                        *size as f64 / 1e6,
                        h.map,
                        h.lanes,
                        hms(h.started),
                        h.trigger
                            .as_deref()
                            .map(|t| format!(" ({t})"))
                            .unwrap_or_default(),
                        name
                    ),
                    Err(e) => println!(
                        "{:>3}  {:>8.1} MB  (unreadable: {e})  {}",
                        i + 1,
                        *size as f64 / 1e6,
                        name
                    ),
                }
            }
            let n: usize = ask(&format!("number (1-{}): ", list.len()))?
                .trim()
                .parse()?;
            list.get(n.wrapping_sub(1))
                .map(|x| x.0.clone())
                .ok_or("no such number")?
        }
        None => {
            println!("choose a recording (a file dialog; --list for a numbered list instead)");
            recording::pick_recording(&traffic_dir).ok_or("no recording chosen")?
        }
    };
    if !rec.is_file() {
        return Err(format!("{} is not a file", rec.display()).into());
    }
    let stem = rec
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("window")
        .to_string();
    let out = match positional.get(1) {
        Some(o) => std::path::PathBuf::from(o),
        None => traffic_dir
            .join("mask")
            .join(format!("{stem}-roadmask.html")),
    };
    if let Some(dir) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    // a long recording: a 90 s window round the time asked for
    let (first, last, long) = recording::frame_span(&rec, recording::LONG_SECS)?;
    if !long {
        return Ok((rec, out));
    }
    let at = match opt(args, "--at") {
        Some(t) => recording::parse_hhmm(t)?,
        None => {
            println!(
                "{} runs from {} (game time) past {}, more than {} minutes: a 90 s window is cut first",
                rec.display(),
                hms(first),
                hms(last),
                (recording::LONG_SECS / 60.0) as i64
            );
            recording::parse_hhmm(&ask("the window's middle, HH:MM in game time (as the recording's clock, not the file name's): ")?)?
        }
    };
    let cut_dir = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.to_path_buf())
        .unwrap_or_default();
    let cut = cut_dir.join(format!(
        "{stem}-at{}.rec",
        hms(at).replace(':', "")[..4].to_string()
    ));
    println!(
        "cutting {} .. {} of {} into {} ...",
        hms(at - recording::WINDOW_HALF),
        hms(at + recording::WINDOW_HALF),
        rec.display(),
        cut.display()
    );
    let (a, b) = recording::cut_window(&rec, at, &cut)?;
    println!("cut: frames {} .. {}", hms(a), hms(b));
    let out = match positional.get(1) {
        Some(_) => out,
        None => cut_dir.join(format!(
            "{}-roadmask.html",
            cut.file_stem().and_then(|s| s.to_str()).unwrap_or(&stem)
        )),
    };
    Ok((cut, out))
}

fn ask(prompt: &str) -> Res<String> {
    use std::io::Write;
    print!("{prompt}");
    std::io::stdout().flush()?;
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line)? == 0 {
        return Err("no input".into());
    }
    Ok(line)
}

fn opt<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

pub(crate) fn hms(t: f64) -> String {
    let s = t.max(0.0).round() as i64;
    format!("{:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
}

fn r2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

fn r1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

/// The window's middle (day time, s): the HHMM of a `caseNN-HHMM.rec` name, else the
/// middle of the recorded frames.
fn window_mid(path: &Path, r: &Replay) -> f64 {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    if let Some(hhmm) = stem.split('-').nth(1) {
        if hhmm.len() == 4 {
            if let (Ok(h), Ok(m)) = (hhmm[..2].parse::<f64>(), hhmm[2..].parse::<f64>()) {
                return h * 3600.0 + m * 60.0;
            }
        }
    }
    let days: Vec<f64> = r
        .records
        .iter()
        .filter_map(|r| {
            if let Record::Frame(f) = r {
                Some(f.day_time)
            } else {
                None
            }
        })
        .collect();
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

struct Sample {
    t: f64,
    /// [id, x, y, heading, lane, s]
    vehicles: Vec<(u64, f64, f64, f64, usize, f64)>,
    /// (x, y, state, decision, at_kerb, on_crossing, leg lane)
    people: Vec<(f64, f64, String, Option<String>, bool, bool, Option<usize>)>,
    /// [x, y, heading, speed, half length, half width]
    players: Vec<[f64; 6]>,
}

fn run(fixture: &Path, out: &Path, args: &[String]) -> Res<()> {
    let fps: f64 = opt(args, "--fps")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(5.0);
    let radius: f64 = opt(args, "--radius")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(100.0);
    let date: Option<i32> = opt(args, "--date").map(|s| s.parse()).transpose()?;

    // the OMSI root and the content roots, as the game has them (mods first, then the
    // installation); the settings are read for the record only
    let root = mapdata::omsi_root(opt(args, "--root"))?;
    if !root.is_dir() {
        return Err(format!("OMSI root {} is not a folder", root.display()).into());
    }
    if let Some(c) = mapdata::game_content_dir() {
        println!("content folder (mods): {}", c.display());
        omsi_cfg::add_content_root(c);
    }
    omsi_cfg::add_content_root(root.clone());
    println!("OMSI root: {}", root.display());
    match mapdata::read_settings() {
        Some(s) => println!(
            "settings.cfg: {} keys read (not written){}",
            s.len(),
            s.get("language")
                .map(|l| format!(", language {l}"))
                .unwrap_or_default()
        ),
        None => println!("settings.cfg: none"),
    }

    // the recording
    println!("reading {} ...", fixture.display());
    let mut r = Replay::load(fixture)?;
    let mid = window_mid(fixture, &r);
    let (cx, cy) = match opt(args, "--center") {
        Some(c) => {
            let (a, b) = c.split_once(',').ok_or("--center x,y")?;
            (a.parse()?, b.parse()?)
        }
        None => {
            let p = player_at(&r, mid).ok_or("no player in the recording: give --center x,y")?;
            (p.x, p.y)
        }
    };
    if !r.at(0) {
        return Err("the recording has no keyframe".into());
    }
    println!(
        "map {} · centre ({cx:.1}, {cy:.1}) · radius {radius} m · window middle {}",
        r.header.map,
        hms(mid)
    );
    let near = |p: DVec3, rad: f64| ((p.x - cx).powi(2) + (p.y - cy).powi(2)).sqrt() <= rad;

    // replay, sampling at `fps`
    let mut samples: Vec<Sample> = Vec::new();
    let mut next_t = f64::NEG_INFINITY;
    let mut vinfo: BTreeMap<u64, Value> = BTreeMap::new();
    while r.step() {
        let t = r.sim.day_time;
        if t + 1e-6 < next_t {
            continue;
        }
        next_t = if next_t.is_finite() {
            (next_t + 1.0 / fps).max(t)
        } else {
            t + 1.0 / fps
        };
        r.sim.update_positions(&r.net, &mut r.vehicles);
        let mut vs = Vec::new();
        for v in &r.vehicles {
            if !near(v.pos, radius + 10.0) {
                continue;
            }
            vinfo.entry(v.id).or_insert_with(|| {
                json!([
                    r2(v.state.front as f64),
                    r2(v.state.rear as f64),
                    r2(v.half_width as f64),
                    format!("{:?}", v.class),
                    r.types.get(&v.id).cloned().unwrap_or_default()
                ])
            });
            vs.push((
                v.id,
                v.pos.x,
                v.pos.y,
                v.heading as f64,
                v.state.lane,
                v.state.s as f64,
            ));
        }
        let people = frame_people(&r)
            .iter()
            .filter(|p| !p.state.contains("aboard") && near(p.pos, radius + 5.0))
            .map(|p| {
                (
                    p.pos.x,
                    p.pos.y,
                    p.state.to_string(),
                    p.decision.as_ref().map(|d| d.to_string()),
                    p.at_kerb,
                    p.on_crossing,
                    p.leg.map(|l| l.0),
                )
            })
            .collect();
        let players = r
            .sim
            .players
            .iter()
            .map(|p| {
                [
                    p.pos.x,
                    p.pos.y,
                    p.heading as f64,
                    p.speed as f64,
                    p.half_len as f64,
                    p.half_width as f64,
                ]
            })
            .collect();
        samples.push(Sample {
            t,
            vehicles: vs,
            people,
            players,
        });
    }
    println!(
        "{} frames sampled, {} vehicles seen",
        samples.len(),
        vinfo.len()
    );

    // the lanes of the window
    let net = &r.net;
    let lane_ids: Vec<usize> = (0..net.lanes.len())
        .filter(|&l| {
            matches!(
                net.lanes[l].kind,
                LaneKind::Street | LaneKind::Sidewalk | LaneKind::Rail
            ) && net.lanes[l]
                .points
                .iter()
                .any(|&p| near(p, radius + MARGIN))
        })
        .collect();

    // the map behind them
    let mut map = mapdata::MapData::open(root.clone(), &r.header.map, date)?;
    println!(
        "map folder {} ({}){}",
        map.map_dir.display(),
        map.global.friendly_name.trim(),
        if date.is_some() {
            format!(" · chrono folders applied: {}", map.chrono_dirs.len())
        } else {
            String::new()
        }
    );
    let mut lane_ref: HashMap<usize, (bool, usize)> = HashMap::new();
    let mut no_key = 0usize;
    for &l in &lane_ids {
        let lane = &net.lanes[l];
        match lane.key {
            Some(k) => {
                if let Some(x) = map.resolve(l, k, lane.source, &lane.name) {
                    lane_ref.insert(l, x);
                }
            }
            None => no_key += 1,
        }
    }

    let kind_code = |k: LaneKind| match k {
        LaneKind::Street => "S",
        LaneKind::Sidewalk => "W",
        LaneKind::Rail => "R",
        LaneKind::Air => "A",
    };
    let pts_json = |pts: &[DVec3]| -> Vec<Value> {
        pts.iter()
            .map(|p| json!([r2(p.x - cx), r2(p.y - cy)]))
            .collect()
    };
    let lanes_json: Vec<Value> = lane_ids
        .iter()
        .map(|&l| {
            let lane = &net.lanes[l];
            let key = lane
                .key
                .map(|k| json!([k.tile.0, k.tile.1, k.id, k.path]))
                .unwrap_or(Value::Null);
            let (sidx, oidx) = match lane_ref.get(&l) {
                Some(&(true, i)) => (json!(i), Value::Null),
                Some(&(false, i)) => (Value::Null, json!(i)),
                None => (Value::Null, Value::Null),
            };
            json!([
                l,
                kind_code(lane.kind),
                r2(lane.width as f64),
                pts_json(&lane.points),
                key,
                lane.source,
                lane.reversed as u8,
                r2(lane.offset as f64),
                lane.name,
                sidx,
                oidx,
                lane.one_way as u8
            ])
        })
        .collect();

    // the splines' cross-sections laid along their centrelines
    let mut n_strips = 0usize;
    let mut n_edges = 0usize;
    let mut z_range = (f64::INFINITY, f64::NEG_INFINITY);
    let mut without_sli = Vec::new();
    // (a .sli found but with no drawable profile: an invisible helper path, or a profile
    // of zero width)
    let mut nothing_drawn = Vec::new();
    let splines_json: Vec<Value> = map
        .splines
        .iter()
        .map(|si| {
            let side = if si.map.mirror { -1.0 } else { 1.0 };
            let n = profile::station_count(&si.curve);
            let centre = profile::lay_along(&si.curve, 0.0, 0.0, 1.0, n);
            let mut strips = Vec::new();
            let mut edges = Vec::new();
            match &si.sli {
                Some(def) => {
                    let cs = profile::cross_section(def);
                    for s in &cs.strips {
                        let l = profile::lay_along(&si.curve, s.x0 as f64, s.z0 as f64, side, n);
                        let rr = profile::lay_along(&si.curve, s.x1 as f64, s.z1 as f64, side, n);
                        z_range = (z_range.0.min(s.z0.min(s.z1) as f64), z_range.1.max(s.z0.max(s.z1) as f64));
                        strips.push(json!({"x0": r2(s.x0 as f64), "x1": r2(s.x1 as f64), "z0": r2(s.z0 as f64), "z1": r2(s.z1 as f64), "tex": s.texture, "cls": s.class, "l": pts_json(&l), "r": pts_json(&rr)}));
                    }
                    for e in &cs.edges {
                        let line = profile::lay_along(&si.curve, e.x as f64, e.z1 as f64, side, n);
                        edges.push(json!({"x": r2(e.x as f64), "z0": r2(e.z0 as f64), "z1": r2(e.z1 as f64), "kind": e.kind, "pts": pts_json(&line)}));
                    }
                    if cs.strips.is_empty() {
                        nothing_drawn.push(si.file.clone());
                    }
                    n_strips += cs.strips.len();
                    n_edges += cs.edges.len();
                }
                None => without_sli.push(si.file.clone()),
            }
            json!({
                "tile": [si.key.0, si.key.1], "id": si.key.2, "file": si.file,
                "path": si.sli_path.as_ref().map(|p| p.display().to_string()),
                "length": r2(si.curve.length), "radius": r2(si.curve.radius), "mirror": si.map.mirror,
                "lanes": si.lanes, "centre": pts_json(&centre), "strips": strips, "edges": edges,
                "paths": si.sli.as_ref().map(|d| d.paths.iter().enumerate().map(|(i, p)| json!([i, p.kind, r2(p.start[0] as f64), r2(p.width as f64), p.direction])).collect::<Vec<_>>()).unwrap_or_default(),
            })
        })
        .collect();

    // the crossing objects: their paths as arcs, their outlines
    let mut n_footprints = 0usize;
    let objects_json: Vec<Value> = map
        .objects
        .iter()
        .map(|oi| {
            let mut paths = Vec::new();
            if let Some(sco) = &oi.sco {
                for (pi, p) in sco.paths.iter().enumerate() {
                    let v = &p.params;
                    if v.len() < 11 {
                        continue;
                    }
                    let start2 =
                        mapdata::object_to_world(oi.pos, oi.heading, v[0] as f64, v[1] as f64);
                    let start = DVec3::new(start2.x, start2.y, oi.pos.z + v[2] as f64);
                    let (heading, radius, length) =
                        (v[3] as f64 + oi.heading, v[4] as f64, v[5] as f64);
                    if length <= 0.01 {
                        continue;
                    }
                    let n = ((length / 2.0).ceil() as usize).clamp(1, 400);
                    let pts: Vec<DVec3> = (0..=n)
                        .map(|i| {
                            let (q, _) =
                                arc_point(start, heading, length * i as f64 / n as f64, radius);
                            DVec3::new(q.x, q.y, start.z)
                        })
                        .collect();
                    paths.push(json!([
                        pi,
                        p.kind,
                        r2(p.width as f64),
                        p.direction,
                        pts_json(&pts)
                    ]));
                }
            }
            let (hull, how) = match &oi.footprint {
                Ok((h, how)) => {
                    n_footprints += 1;
                    (
                        Value::Array(
                            h.iter()
                                .map(|q| {
                                    let w = mapdata::object_to_world(oi.pos, oi.heading, q.x, q.y);
                                    json!([r2(w.x - cx), r2(w.y - cy)])
                                })
                                .collect(),
                        ),
                        json!(how),
                    )
                }
                Err(why) => (Value::Null, json!(format!("none: {why}"))),
            };
            json!({
                "tile": [oi.key.0, oi.key.1], "id": oi.key.2, "file": oi.file,
                "path": oi.sco_path.as_ref().map(|p| p.display().to_string()),
                "pos": [r2(oi.pos.x - cx), r2(oi.pos.y - cy)], "heading": r1(oi.heading),
                "lanes": oi.lanes, "paths": paths, "hull": hull, "footprint": how,
            })
        })
        .collect();

    // the frames
    let mut strs: Vec<String> = Vec::new();
    let mut str_ix: HashMap<String, usize> = HashMap::new();
    let mut si = |s: &str| -> usize {
        if let Some(&i) = str_ix.get(s) {
            return i;
        }
        strs.push(s.to_string());
        str_ix.insert(s.to_string(), strs.len() - 1);
        strs.len() - 1
    };
    let mut frames = Vec::new();
    let mut prev_people: Vec<String> = Vec::new();
    for s in &samples {
        let v: Vec<Value> = s
            .vehicles
            .iter()
            .filter(|v| ((v.1 - cx).powi(2) + (v.2 - cy).powi(2)).sqrt() <= radius)
            .map(|v| json!([v.0, r2(v.1 - cx), r2(v.2 - cy), r1(v.3), v.4, r1(v.5)]))
            .collect();
        let rows: Vec<String> = s
            .people
            .iter()
            .filter(|p| ((p.0 - cx).powi(2) + (p.1 - cy).powi(2)).sqrt() <= radius)
            .map(|p| {
                serde_json::to_string(&json!([
                    r2(p.0 - cx),
                    r2(p.1 - cy),
                    si(&p.2),
                    p.3.as_ref().map(|d| json!(si(d))).unwrap_or(Value::Null),
                    p.4 as u8,
                    p.5 as u8,
                    p.6
                ]))
                .unwrap_or_default()
            })
            .collect();
        // (a row the same as one of the previous frame's goes by its index there: people standing)
        let prev_ix: HashMap<&str, usize> = prev_people
            .iter()
            .enumerate()
            .map(|(i, t)| (t.as_str(), i))
            .collect();
        let mut pk = Vec::new();
        let mut p = Vec::new();
        let mut carried = Vec::new();
        let mut fresh = Vec::new();
        for row in rows {
            match prev_ix.get(row.as_str()) {
                Some(&i) => {
                    pk.push(json!(i));
                    carried.push(row);
                }
                None => {
                    p.push(serde_json::from_str::<Value>(&row).unwrap_or(Value::Null));
                    fresh.push(row);
                }
            }
        }
        drop(prev_ix);
        carried.extend(fresh);
        prev_people = carried;
        let pl: Vec<Value> = s
            .players
            .iter()
            .map(|p| {
                json!([
                    r2(p[0] - cx),
                    r2(p[1] - cy),
                    r1(p[2]),
                    r1(p[3]),
                    r2(p[4]),
                    r2(p[5])
                ])
            })
            .collect();
        let mut fr = serde_json::Map::new();
        fr.insert("t".into(), json!(r2(s.t)));
        fr.insert("v".into(), Value::Array(v));
        fr.insert("p".into(), Value::Array(p));
        if !pk.is_empty() {
            fr.insert("pk".into(), Value::Array(pk));
        }
        fr.insert("pl".into(), Value::Array(pl));
        frames.push(Value::Object(fr));
    }

    let title = opt(args, "--title")
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            format!(
                "{} · road mask",
                fixture
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("window")
            )
        });
    let doc = json!({
        "meta": {
            "fixture": fixture.file_name().and_then(|s| s.to_str()),
            "map": r.header.map,
            "root": root.display().to_string(),
            "center": [r2(cx), r2(cy)],
            "radius": radius,
            "fps": fps,
            "mid": mid,
            "mid_hms": hms(mid),
            "start": samples.first().map(|s| s.t),
            "end": samples.last().map(|s| s.t),
            "person_radius": PERSON_RADIUS,
            "z_range": if z_range.0.is_finite() { json!([r2(z_range.0), r2(z_range.1)]) } else { Value::Null },
            "problems": map.problems,
            "fields": {
                "lanes": "[id, kind S/W/R, width, points, key [tile x, tile y, id, path]|null, source (1 spline, 2 object), reversed, offset, file, spline index|null, object index|null, one_way]",
                "frames.v": "[id, x, y, heading_deg, lane, s]",
                "frames.p": "[x, y, state (strs), decision (strs)|null, at_kerb, on_crossing, leg lane|null]; the frame's people = previous frame's at the indices pk, then p",
                "frames.pl": "[x, y, heading_deg, speed, half_len, half_width]",
                "vinfo": "id: [front, rear, half_width, class, type]",
                "splines": "{tile, id, file, path, length, radius, mirror, lanes, centre, strips [{x0,x1,z0,z1,tex,cls,l,r}], edges [{x,z0,z1,kind,pts}], paths [[i, kind, x, width, direction]]}",
                "objects": "{tile, id, file, path, pos, heading, lanes, paths [[i, kind, width, direction, pts]], hull|null, footprint}",
            },
        },
        "strs": strs,
        "vinfo": vinfo,
        "lanes": lanes_json,
        "splines": splines_json,
        "objects": objects_json,
        "frames": frames,
    });
    let data = serde_json::to_string(&doc)?.replace("</", "<\\/");
    let html = PAGE
        .replace("__TITLE__", &esc_html(&title))
        .replace("__DATA__", &data);
    if let Some(dir) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(out, &html)?;

    // what was and was not resolved
    let with_sli = map.splines.iter().filter(|s| s.sli.is_some()).count();
    println!();
    println!("lanes in the window: {} ({} spline lanes, {} object lanes, {} without a key, {} unresolved)", lane_ids.len(), lane_ids.iter().filter(|l| lane_ref.get(l).is_some_and(|x| x.0)).count(), lane_ids.iter().filter(|l| lane_ref.get(l).is_some_and(|x| !x.0)).count(), no_key, lane_ids.iter().filter(|l| net.lanes[**l].key.is_some() && !lane_ref.contains_key(l)).count());
    println!("spline instances: {} ({with_sli} with their .sli: {n_strips} strips, {n_edges} edge lines; {} without)", map.splines.len(), map.splines.len() - with_sli);
    without_sli.sort();
    without_sli.dedup();
    for f in &without_sli {
        println!("  .sli not found: {f}");
    }
    nothing_drawn.sort();
    nothing_drawn.dedup();
    for f in &nothing_drawn {
        println!("  .sli with no drawable profile (nothing to show): {f}");
    }
    println!(
        "crossing objects: {} ({n_footprints} with a footprint, {} without)",
        map.objects.len(),
        map.objects.len() - n_footprints
    );
    for o in map.objects.iter().filter(|o| o.footprint.is_err()) {
        println!(
            "  no footprint: {} (tile {},{} id {}): {}",
            o.file,
            o.key.0,
            o.key.1,
            o.key.2,
            o.footprint.as_ref().err().cloned().unwrap_or_default()
        );
    }
    if !map.problems.is_empty() {
        println!("problems ({}):", map.problems.len());
        for p in &map.problems {
            println!("  {p}");
        }
    }
    println!(
        "{}: {:.0} KB, {} frames",
        out.display(),
        html.len() as f64 / 1024.0,
        frames.len()
    );
    Ok(())
}

fn esc_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}
