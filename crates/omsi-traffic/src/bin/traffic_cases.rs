//! traffic_cases: turn a long traffic recording into incident cases and look at them.
//!
//! ```text
//! traffic_cases info <rec>
//! traffic_cases cut <rec> <out_dir> <HH:MM>[,<HH:MM>...]
//! traffic_cases trace <fixture> <out.json> [--center x,y] [--radius 150] [--fps 5] [--max-kb 400]
//! traffic_cases diagnose <fixture> [--radius 200] [--min-stand 15]
//! traffic_cases viewer <trace.json> <out.html> [--title ...]
//! traffic_cases index <out_dir>
//! ```
//!
//! `info` and `cut` stream the recording once (a session recording does not fit in
//! memory parsed); `trace` and `diagnose` replay a cut fixture headless.

use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use glam::DVec3;
use hashbrown::{HashMap, HashSet};
use omsi_sim::traffic::{LaneKind, Network, TrafficLightController};
use omsi_traffic::perception::{way_lanes, Seen};
use omsi_traffic::record::{Header, PersonRecord, Record};
use omsi_traffic::replay::Replay;
use omsi_traffic::sim::PLAYER_ID_BASE;
use omsi_traffic::{Hold, Vehicle, Why};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

type Res<T> = Result<T, Box<dyn Error>>;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let r = match args.get(1).map(|s| s.as_str()) {
        Some("info") if args.len() >= 3 => info(Path::new(&args[2])),
        Some("cut") if args.len() >= 5 => cut(Path::new(&args[2]), Path::new(&args[3]), &args[4]),
        Some("trace") if args.len() >= 4 => {
            trace(Path::new(&args[2]), Path::new(&args[3]), &args[4..])
        }
        Some("diagnose") if args.len() >= 3 => diagnose(Path::new(&args[2]), &args[3..]),
        Some("viewer") if args.len() >= 4 => {
            viewer(Path::new(&args[2]), Path::new(&args[3]), &args[4..])
        }
        Some("index") if args.len() >= 3 => index(Path::new(&args[2])),
        _ => {
            eprintln!(
                "usage: traffic_cases info|cut|trace|diagnose|viewer|index ... (see the source)"
            );
            std::process::exit(2);
        }
    };
    if let Err(e) = r {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------------------
// helpers

fn hms(t: f64) -> String {
    if !t.is_finite() {
        return "--:--:--".into();
    }
    let s = t.max(0.0).round() as i64;
    format!("{:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
}

fn parse_hhmm(s: &str) -> Res<f64> {
    let (h, m) = s
        .trim()
        .split_once(':')
        .ok_or_else(|| format!("not HH:MM: {s}"))?;
    Ok(h.parse::<f64>()? * 3600.0 + m.parse::<f64>()? * 60.0)
}

fn opt<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn r1(x: f64) -> f64 {
    let v = (x * 10.0).round() / 10.0;
    if v == 0.0 {
        0.0
    } else {
        v
    }
}

fn r2(x: f64) -> f64 {
    let v = (x * 100.0).round() / 100.0;
    if v == 0.0 {
        0.0
    } else {
        v
    }
}

fn why_name(w: Why) -> &'static str {
    match w {
        Why::None => "none",
        other => other.name(),
    }
}

fn hold_name(h: Option<Hold>) -> Option<&'static str> {
    h.map(|h| match h {
        Hold::Foe => "foe",
        Hold::ExitFull => "exit_full",
    })
}

fn fmt_id(id: u64) -> String {
    if id >= PLAYER_ID_BASE {
        format!("PLAYER{}", id - PLAYER_ID_BASE)
    } else {
        format!("#{id}")
    }
}

/// The lines of a zlib-compressed recording, one at a time.
struct Lines {
    r: BufReader<ZlibDecoder<BufReader<File>>>,
    pub buf: Vec<u8>,
    /// None while reading; Some(true) when the stream ended cleanly.
    pub clean: Option<bool>,
    pub error: Option<String>,
    pub bytes: u64,
}

impl Lines {
    fn open(p: &Path) -> Res<Lines> {
        let f = File::open(p)?;
        Ok(Lines {
            r: BufReader::with_capacity(
                1 << 22,
                ZlibDecoder::new(BufReader::with_capacity(1 << 22, f)),
            ),
            buf: Vec::with_capacity(1 << 20),
            clean: None,
            error: None,
            bytes: 0,
        })
    }

    /// The next line (without its end), false at the end.
    fn next(&mut self) -> bool {
        if self.clean.is_some() {
            return false;
        }
        self.buf.clear();
        match self.r.read_until(b'\n', &mut self.buf) {
            Ok(0) => {
                self.clean = Some(true);
                false
            }
            Ok(n) => {
                self.bytes += n as u64;
                if self.buf.last() == Some(&b'\n') {
                    self.buf.pop();
                    true
                } else {
                    self.clean = Some(false);
                    self.error = Some("the last line has no end".into());
                    false
                }
            }
            Err(e) => {
                self.clean = Some(false);
                self.error = Some(e.to_string());
                false
            }
        }
    }
}

/// The record kind, its time and (a frame's) day time, read off the start of a line.
fn peek(line: &[u8]) -> (&str, f64, Option<f64>) {
    let head = &line[..line.len().min(400)];
    let kind = (|| {
        let a = head.iter().position(|&c| c == b'"')? + 1;
        let b = a + head[a..].iter().position(|&c| c == b'"')?;
        std::str::from_utf8(&head[a..b]).ok()
    })()
    .unwrap_or("?");
    let num = |key: &[u8]| -> Option<f64> {
        let i = head.windows(key.len()).position(|w| w == key)? + key.len();
        let j = i + head[i..].iter().position(|&c| c == b',' || c == b'}')?;
        std::str::from_utf8(&head[i..j]).ok()?.parse().ok()
    };
    let time = num(b"\"time\":").unwrap_or(f64::NAN);
    let day = if kind == "Frame" {
        num(b"\"day_time\":")
    } else {
        None
    };
    (kind, time, day)
}

/// The first player's position in a frame line (x, y), cheaply.
fn peek_player(line: &[u8]) -> Option<(f64, f64)> {
    let key = b"\"players\":[{";
    let i = line.windows(key.len()).position(|w| w == key)? + key.len();
    let key2 = b"\"pos\":[";
    let j = i
        + line[i..i + 200.min(line.len() - i)]
            .windows(key2.len())
            .position(|w| w == key2)?
        + key2.len();
    let k = j + line[j..].iter().position(|&c| c == b']')?;
    let s = std::str::from_utf8(&line[j..k]).ok()?;
    let mut it = s.split(',').map(|v| v.parse::<f64>().ok());
    Some((it.next()??, it.next()??))
}

// ---------------------------------------------------------------------------------------
// info

fn info(path: &Path) -> Res<()> {
    let mut lines = Lines::open(path)?;
    if !lines.next() {
        return Err("empty recording".into());
    }
    let header: Header = omsi_traffic::finite::from_str(std::str::from_utf8(&lines.buf)?)?;
    let header_bytes = lines.buf.len();
    println!(
        "file: {} ({:.2} GB compressed)",
        path.display(),
        std::fs::metadata(path)?.len() as f64 / 1e9
    );
    println!("header: version {} map {} left_hand {} lanes {} lights {} started {} trigger {:?} derived {} ({:.1} MB)", header.version, header.map, header.left_hand, header.lanes.len(), header.lights.len(), hms(header.started_day_time), header.trigger, header.derived.as_ref().map(|d| format!("yes, {} junctions", d.junctions.nodes.len())).unwrap_or("no".into()), header_bytes as f64 / 1e6);
    drop(header);
    let mut counts: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    let (mut first_day, mut last_day) = (f64::NAN, f64::NAN);
    let (mut first_t, mut last_t) = (f64::NAN, f64::NAN);
    let mut keyframes = 0u64;
    let mut lanes_added = 0usize;
    let mut lights_added = 0usize;
    let mut minute = -1i64;
    let mut notes: Vec<String> = Vec::new();
    let mut scales: BTreeMap<String, u64> = BTreeMap::new();
    let mut offset = f64::NAN;
    while lines.next() {
        let (kind, time, day) = peek(&lines.buf);
        let e = counts.entry(kind.to_string()).or_default();
        e.0 += 1;
        e.1 += lines.buf.len() as u64 + 1;
        if first_t.is_nan() {
            first_t = time;
        }
        if time.is_finite() {
            last_t = time;
        }
        match kind {
            "Frame" => {
                if let Some(d) = day {
                    if first_day.is_nan() {
                        first_day = d;
                    }
                    last_day = d;
                    offset = d - time;
                    let m = (d / 60.0).floor() as i64;
                    if m != minute {
                        minute = m;
                        let p = peek_player(&lines.buf);
                        let head = std::str::from_utf8(&lines.buf[..lines.buf.len().min(300)])
                            .unwrap_or("");
                        let ts = head
                            .split("\"time_scale\":")
                            .nth(1)
                            .and_then(|s| s.split(',').next())
                            .unwrap_or("?")
                            .to_string();
                        *scales.entry(ts).or_default() += 1;
                        println!(
                            "  minute {} (t={:.0}): player {}",
                            hms(d),
                            time,
                            p.map(|(x, y)| format!("({x:.1}, {y:.1})"))
                                .unwrap_or("-".into())
                        );
                    }
                }
            }
            "Keyframe" => keyframes += 1,
            "LanesAdded" | "LightsAdded" | "Note" => {
                if let Ok(r) =
                    omsi_traffic::finite::from_str::<Record>(std::str::from_utf8(&lines.buf)?)
                {
                    match r {
                        Record::LanesAdded { lanes, .. } => lanes_added += lanes.len(),
                        Record::LightsAdded { lights, .. } => lights_added += lights.len(),
                        Record::Note { time, text } => {
                            notes.push(format!("{} {}", hms(time + offset), text))
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    println!("records by kind (count, MB uncompressed):");
    for (k, (n, b)) in &counts {
        println!("  {k:<15} {n:>9} {:>10.1}", *b as f64 / 1e6);
    }
    println!("decompressed: {:.2} GB", (lines.bytes) as f64 / 1e9);
    println!(
        "day_time: {} .. {} ({:.1} min); sim time {:.1} .. {:.1}",
        hms(first_day),
        hms(last_day),
        (last_day - first_day) / 60.0,
        first_t,
        last_t
    );
    println!("keyframes: {keyframes}; lanes added later: {lanes_added}; lights added later: {lights_added}; time_scale samples: {scales:?}");
    println!("notes: {}", notes.len());
    for n in notes.iter().take(400) {
        println!("  note {n}");
    }
    match lines.clean {
        Some(true) => println!("zlib stream: ended cleanly"),
        _ => println!(
            "zlib stream: CUT SHORT ({})",
            lines.error.unwrap_or_default()
        ),
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// cut

struct OpenCase {
    t1: f64,
    path: PathBuf,
    out: ZlibEncoder<BufWriter<File>>,
    lines: u64,
}

fn cut(path: &Path, out_dir: &Path, times: &str) -> Res<()> {
    std::fs::create_dir_all(out_dir)?;
    let mut cases: Vec<(usize, f64, String)> = Vec::new();
    for (k, t) in times.split(',').enumerate() {
        cases.push((k + 1, parse_hhmm(t)?, t.trim().replace(':', "")));
    }
    let mut pending: Vec<(usize, f64, String)> = cases.clone();
    pending.sort_by(|a, b| a.1.total_cmp(&b.1));
    pending.reverse(); // pop from the back: the earliest first
    let mut lines = Lines::open(path)?;
    if !lines.next() {
        return Err("empty recording".into());
    }
    let header = lines.buf.clone();
    let mut added_before: Vec<Vec<u8>> = Vec::new();
    let mut parked_before: Option<Vec<u8>> = None;
    let mut block: Vec<Vec<u8>> = Vec::new();
    let mut open: Vec<OpenCase> = Vec::new();
    let mut done: Vec<PathBuf> = Vec::new();
    let mut offset = f64::NAN;
    while lines.next() {
        let line = std::mem::take(&mut lines.buf);
        let (kind, time, day) = peek(&line);
        let kind = kind.to_string();
        if let Some(d) = day {
            offset = d - time;
        }
        let est = day.unwrap_or(time + offset);
        if kind == "Keyframe" {
            // cases whose window starts before this keyframe start at the last one
            while let Some(c) = pending.last() {
                let t0 = c.1 - 45.0;
                if !(est > t0) {
                    break;
                }
                let c = pending.pop().unwrap();
                let p = out_dir.join(format!("case{:02}-{}.rec", c.0, c.2));
                let mut out = ZlibEncoder::new(
                    BufWriter::with_capacity(1 << 22, File::create(&p)?),
                    Compression::default(),
                );
                out.write_all(&header)?;
                out.write_all(b"\n")?;
                let mut n = 0;
                for l in &added_before {
                    out.write_all(l)?;
                    out.write_all(b"\n")?;
                    n += 1;
                }
                if let Some(l) = &parked_before {
                    out.write_all(l)?;
                    out.write_all(b"\n")?;
                    n += 1;
                }
                for l in &block {
                    out.write_all(l)?;
                    out.write_all(b"\n")?;
                    n += 1;
                }
                eprintln!(
                    "case {:02} ({}): opened at keyframe {:.1}, {} lines carried",
                    c.0,
                    hms(c.1),
                    est,
                    n
                );
                open.push(OpenCase {
                    t1: c.1 + 45.0,
                    path: p,
                    out,
                    lines: n,
                });
            }
            for l in block.drain(..) {
                match peek(&l).0 {
                    "LanesAdded" | "LightsAdded" => added_before.push(l),
                    "Parked" => parked_before = Some(l),
                    _ => {}
                }
            }
        }
        // close the cases past their end
        if kind == "Frame" {
            let mut k = 0;
            while k < open.len() {
                if day.unwrap_or(f64::NAN) > open[k].t1 {
                    let c = open.swap_remove(k);
                    let lines_n = c.lines;
                    c.out.finish()?.flush()?;
                    eprintln!("  closed {} ({} lines)", c.path.display(), lines_n);
                    done.push(c.path);
                } else {
                    k += 1;
                }
            }
        }
        for c in open.iter_mut() {
            c.out.write_all(&line)?;
            c.out.write_all(b"\n")?;
            c.lines += 1;
        }
        if kind == "Keyframe" || !block.is_empty() {
            block.push(line);
        } else {
            match kind.as_str() {
                "LanesAdded" | "LightsAdded" => added_before.push(line),
                "Parked" => parked_before = Some(line),
                _ => {}
            }
        }
        if pending.is_empty() && open.is_empty() {
            break;
        }
    }
    for c in open.drain(..) {
        c.out.finish()?.flush()?;
        eprintln!("  closed at the end of the recording: {}", c.path.display());
        done.push(c.path);
    }
    for c in &pending {
        eprintln!("case {:02} ({}): not in the recording", c.0, hms(c.1));
    }
    // check each one reads
    done.sort();
    for p in &done {
        let size = std::fs::metadata(p)?.len();
        match Replay::load(p) {
            Ok(r) => {
                let frames = r
                    .records
                    .iter()
                    .filter(|r| matches!(r, Record::Frame(_)))
                    .count();
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
                println!(
                    "{}: {:.1} MB, {} records, {} frames, {} keyframes, {} .. {}, skipped {}, truncated {}",
                    p.display(),
                    size as f64 / 1e6,
                    r.records.len(),
                    frames,
                    r.keyframe_count(),
                    hms(days.first().copied().unwrap_or(f64::NAN)),
                    hms(days.last().copied().unwrap_or(f64::NAN)),
                    r.skipped.len(),
                    r.truncated
                );
            }
            Err(e) => println!("{}: DOES NOT LOAD: {e}", p.display()),
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// replay helpers

/// The middle of a fixture's window: from its name (`caseNN-HHMM`), else the middle of
/// its frames.
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

/// The first player's position at the frame nearest day time `t`.
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

fn dist2(a: DVec3, cx: f64, cy: f64) -> f64 {
    ((a.x - cx).powi(2) + (a.y - cy).powi(2)).sqrt()
}

fn simplify(pts: &[(f64, f64)], tol: f64) -> Vec<(f64, f64)> {
    if pts.len() <= 2 {
        return pts.to_vec();
    }
    let mut keep = vec![false; pts.len()];
    keep[0] = true;
    keep[pts.len() - 1] = true;
    let mut stack = vec![(0usize, pts.len() - 1)];
    while let Some((a, b)) = stack.pop() {
        let (ax, ay) = pts[a];
        let (bx, by) = pts[b];
        let (dx, dy) = (bx - ax, by - ay);
        let len = (dx * dx + dy * dy).sqrt().max(1e-9);
        let mut worst = (0.0, a);
        for (k, &(px, py)) in pts.iter().enumerate().take(b).skip(a + 1) {
            let d = ((px - ax) * dy - (py - ay) * dx).abs() / len;
            if d > worst.0 {
                worst = (d, k);
            }
        }
        if worst.0 > tol {
            keep[worst.1] = true;
            stack.push((a, worst.1));
            stack.push((worst.1, b));
        }
    }
    pts.iter()
        .zip(keep)
        .filter(|(_, k)| *k)
        .map(|(p, _)| *p)
        .collect()
}

// ---------------------------------------------------------------------------------------
// trace

/// The street lanes' segments near the centre in 10 m cells: is a point on (or by) a
/// carriageway.
struct RoadGrid {
    lanes: usize,
    cells: HashMap<(i32, i32), Vec<(f64, f64, f64, f64, f64)>>,
}

impl RoadGrid {
    fn new(net: &Network, cx: f64, cy: f64, rad: f64) -> RoadGrid {
        let mut cells: HashMap<(i32, i32), Vec<(f64, f64, f64, f64, f64)>> = HashMap::new();
        for lane in &net.lanes {
            if lane.kind != LaneKind::Street
                || !lane.points.iter().any(|p| dist2(*p, cx, cy) <= rad + 30.0)
            {
                continue;
            }
            let hw = lane.width.max(2.5) as f64 * 0.5;
            for w in lane.points.windows(2) {
                let (a, b) = (w[0], w[1]);
                let (x0, x1) = (a.x.min(b.x) - 6.0, a.x.max(b.x) + 6.0);
                let (y0, y1) = (a.y.min(b.y) - 6.0, a.y.max(b.y) + 6.0);
                for gx in (x0 / 10.0).floor() as i32..=(x1 / 10.0).floor() as i32 {
                    for gy in (y0 / 10.0).floor() as i32..=(y1 / 10.0).floor() as i32 {
                        cells
                            .entry((gx, gy))
                            .or_default()
                            .push((a.x, a.y, b.x, b.y, hw));
                    }
                }
            }
        }
        RoadGrid {
            lanes: net.lanes.len(),
            cells,
        }
    }

    /// Within `margin` metres of a carriageway's edge (or on it).
    fn near(&self, x: f64, y: f64, margin: f64) -> bool {
        let Some(segs) = self
            .cells
            .get(&((x / 10.0).floor() as i32, (y / 10.0).floor() as i32))
        else {
            return false;
        };
        segs.iter().any(|&(ax, ay, bx, by, hw)| {
            let (dx, dy) = (bx - ax, by - ay);
            let t =
                (((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy).max(1e-9)).clamp(0.0, 1.0);
            let d = ((ax + dx * t - x).powi(2) + (ay + dy * t - y).powi(2)).sqrt();
            d <= hw + margin
        })
    }
}

struct VSample {
    id: u64,
    x: f64,
    y: f64,
    row: Vec<Value>,
}

struct Sample {
    t: f64,
    vehicles: Vec<VSample>,
    /// (x, y, state, decision, still, on_crossing)
    people: Vec<(f64, f64, String, Option<String>, f64, bool)>,
    players: Vec<[f64; 4]>,
    lights: Vec<(usize, usize, i32)>,
    /// Every light's state, per controller.
    all_lights: Vec<Vec<i32>>,
}

fn trace(fixture: &Path, out: &Path, args: &[String]) -> Res<()> {
    let fps: f64 = opt(args, "--fps")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(5.0);
    let radius: f64 = opt(args, "--radius")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(150.0);
    let max_kb: usize = opt(args, "--max-kb")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(400);
    let mut r = Replay::load(fixture)?;
    let mid = window_mid(fixture, &r);
    let (cx, cy) = match opt(args, "--center") {
        Some(c) => {
            let (a, b) = c.split_once(',').ok_or("--center x,y")?;
            (a.parse()?, b.parse()?)
        }
        None => {
            let p = player_at(&r, mid).ok_or("no player in the fixture")?;
            (p.x, p.y)
        }
    };
    if !r.at(0) {
        return Err("no keyframe".into());
    }
    let notes: Vec<String> = r
        .records
        .iter()
        .filter_map(|x| {
            if let Record::Note { time, text } = x {
                Some(format!("t={time:.1} {text}"))
            } else {
                None
            }
        })
        .collect();
    // sample at a fine rate once; thin out when writing
    let base = 20.0f64;
    let mut samples: Vec<Sample> = Vec::new();
    let mut next_t = f64::NEG_INFINITY;
    let mut vinfo: BTreeMap<u64, Value> = BTreeMap::new();
    let sample_r = radius + 10.0;
    // the lanes, the lights near the centre
    let near_lane = |net: &Network, l: usize, rad: f64| {
        net.lanes[l].points.iter().any(|p| dist2(*p, cx, cy) <= rad)
    };
    let all_people = args.iter().any(|a| a == "--all-people");
    let mut road: Option<RoadGrid> = None;
    while r.step() {
        let t = r.sim.day_time;
        if t + 1e-6 < next_t {
            continue;
        }
        next_t = if next_t.is_finite() {
            next_t + 1.0 / base
        } else {
            t + 1.0 / base
        };
        if t > next_t {
            next_t = t + 1.0 / base;
        }
        r.sim.update_positions(&r.net, &mut r.vehicles);
        let mut vs = Vec::new();
        for v in &r.vehicles {
            if dist2(v.pos, cx, cy) > sample_r {
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
            let d = &v.last;
            let wf: Value = match d.wait_for {
                Some(w) if w >= PLAYER_ID_BASE => json!(format!("P{}", w - PLAYER_ID_BASE)),
                Some(w) => json!(w),
                None => Value::Null,
            };
            let row = vec![
                json!(v.id),
                json!(r1(v.pos.x - cx)),
                json!(r1(v.pos.y - cy)),
                json!(v.heading.round() as i64),
                json!(r1(v.state.speed as f64)),
                json!(why_name(d.why)),
                d.stop.map(|s| json!(r1(s as f64))).unwrap_or(Value::Null),
                d.lead_id
                    .or(d.body_block)
                    .map(|x| json!(x))
                    .unwrap_or(Value::Null),
                wf,
                json!(r1(v.impatience.value as f64)),
                json!(v.passing.is_some() as u8),
                json!(v.state.lane),
                json!(r1(v.state.s as f64)),
                json!(d.held as u8),
                hold_name(d.hold).map(|h| json!(h)).unwrap_or(Value::Null),
            ];
            vs.push(VSample {
                id: v.id,
                x: v.pos.x,
                y: v.pos.y,
                row,
            });
        }
        let mut ps = Vec::new();
        // (the network may grow during the window: the grid follows it)
        if road
            .as_ref()
            .map(|g| g.lanes != r.net.lanes.len())
            .unwrap_or(true)
        {
            road = Some(RoadGrid::new(&r.net, cx, cy, sample_r + 20.0));
        }
        let grid = road.as_ref().unwrap();
        for p in frame_people(&r) {
            if p.state.contains("aboard") || dist2(p.pos, cx, cy) > sample_r {
                continue;
            }
            let waiting = p.decision.as_deref().map(|d| d != "go").unwrap_or(false);
            if !all_people
                && !p.on_crossing
                && !grid.near(p.pos.x, p.pos.y, if waiting { 2.0 } else { 0.5 })
            {
                continue;
            }
            ps.push((
                p.pos.x,
                p.pos.y,
                p.state.to_string(),
                p.decision.as_ref().map(|d| d.to_string()),
                p.still as f64,
                p.on_crossing,
            ));
        }
        let players = r
            .sim
            .players
            .iter()
            .map(|p| {
                [
                    r1(p.pos.x - cx),
                    r1(p.pos.y - cy),
                    p.heading.round() as f64,
                    r1(p.speed as f64),
                ]
            })
            .collect();
        let all_lights = r
            .sim
            .lights
            .iter()
            .map(|c| (0..c.lights.len()).map(|li| c.state(li)).collect())
            .collect();
        samples.push(Sample {
            t,
            vehicles: vs,
            people: ps,
            players,
            lights: Vec::new(),
            all_lights,
        });
    }
    // the static part (for the largest radius)
    let net = &r.net;
    let foot_of_crossing: HashSet<usize> = r.sim.junctions.of_foot_lane.keys().copied().collect();
    let mut lane_set: Vec<usize> = Vec::new();
    for l in 0..net.lanes.len() {
        let lane = &net.lanes[l];
        let keep = match lane.kind {
            LaneKind::Street | LaneKind::Rail => true,
            LaneKind::Sidewalk => foot_of_crossing.contains(&l) || lane.traffic_light.is_some(),
            _ => false,
        };
        if keep && near_lane(net, l, radius + 50.0) {
            lane_set.push(l);
        }
    }
    let lane_in: HashSet<usize> = lane_set.iter().copied().collect();
    let mut lights: Vec<(usize, usize)> = lane_set
        .iter()
        .filter_map(|&l| net.lanes[l].traffic_light)
        .collect();
    let mut crossings = Vec::new();
    for (j, node) in r.sim.junctions.nodes.iter().enumerate() {
        for c in &node.crossings {
            if c.foot_lanes.iter().any(|l| lane_in.contains(l)) {
                crossings.push(json!({"j": j, "lanes": c.foot_lanes, "light": c.light.map(|(a, b)| json!([a, b]))}));
                if let Some(li) = c.light {
                    lights.push(li);
                }
            }
        }
    }
    lights.sort();
    lights.dedup();
    let mut stops = Vec::new();
    let mut seen_stop: HashSet<(i64, i64)> = HashSet::new();
    for &l in &lane_set {
        let lane = &net.lanes[l];
        if lane.kind != LaneKind::Street {
            continue;
        }
        let is_mov_start = r
            .sim
            .junctions
            .of_lane
            .get(&l)
            .map(|&(j, m)| {
                r.sim.junctions.nodes[j].movements[m].lane == l
                    && net
                        .prev
                        .get(l)
                        .map(|p| p.iter().all(|&q| !r.sim.junctions.of_lane.contains_key(&q)))
                        .unwrap_or(true)
            })
            .unwrap_or(false);
        if lane.traffic_light.is_some() || is_mov_start {
            let (p, h) = lane.at(0.0);
            let key = ((p.x * 2.0).round() as i64, (p.y * 2.0).round() as i64);
            if seen_stop.insert(key) {
                stops.push(json!([
                    r1(p.x - cx),
                    r1(p.y - cy),
                    h.round() as i64,
                    l,
                    lane.traffic_light.map(|(a, b)| json!([a, b]))
                ]));
            }
        }
    }
    let lanes_json = |rad: f64| -> Vec<Value> {
        lane_set
            .iter()
            .filter(|&&l| near_lane(net, l, rad + 50.0))
            .map(|&l| {
                let lane = &net.lanes[l];
                let pts: Vec<(f64, f64)> =
                    lane.points.iter().map(|p| (p.x - cx, p.y - cy)).collect();
                let pts = simplify(&pts, 0.25);
                let kind = match lane.kind {
                    LaneKind::Street => "S",
                    LaneKind::Sidewalk => "W",
                    LaneKind::Rail => "R",
                    LaneKind::Air => "A",
                };
                json!([
                    l,
                    kind,
                    r1(lane.width as f64),
                    lane.traffic_light.map(|(a, b)| json!([a, b])),
                    lane.no_overtaking as u8,
                    r2(lane.density as f64),
                    lane.turn,
                    lane.rule_bus as u8,
                    pts.iter()
                        .map(|&(x, y)| json!([r1(x), r1(y)]))
                        .collect::<Vec<_>>()
                ])
            })
            .collect()
    };
    for s in samples.iter_mut() {
        s.lights = lights
            .iter()
            .map(|&(c, li)| {
                (
                    c,
                    li,
                    s.all_lights
                        .get(c)
                        .and_then(|x| x.get(li))
                        .copied()
                        .unwrap_or(-1),
                )
            })
            .collect();
    }
    // write with thinning until it fits
    let tries: Vec<(f64, f64)> = {
        let mut v = vec![(fps, radius)];
        for &(f, rr) in &[
            (4.0, radius),
            (5.0, 120.0f64.min(radius)),
            (4.0, 120.0f64.min(radius)),
            (10.0 / 3.0, 120.0f64.min(radius)),
            (2.5, 120.0f64.min(radius)),
            (2.5, 100.0f64.min(radius)),
            (2.0, 100.0f64.min(radius)),
            (2.0, 80.0f64.min(radius)),
            (1.0, 80.0f64.min(radius)),
        ] {
            if f <= fps && !v.contains(&(f, rr)) {
                v.push((f, rr));
            }
        }
        v
    };
    let mut last_text = String::new();
    let mut used = (fps, radius);
    for &(f, rad) in &tries {
        let step = (base / f).round().max(1.0) as usize;
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
        let mut prev_rows: HashMap<u64, Vec<Value>> = HashMap::new();
        let mut prev_people: Vec<String> = Vec::new();
        let mut prev_lights: HashMap<(usize, usize), i32> = HashMap::new();
        let mut vids: BTreeMap<u64, Value> = BTreeMap::new();
        for s in samples.iter().step_by(step) {
            let mut v = Vec::new();
            let mut k = Vec::new();
            let mut now_rows: HashMap<u64, Vec<Value>> = HashMap::new();
            for vs in &s.vehicles {
                if ((vs.x - cx).powi(2) + (vs.y - cy).powi(2)).sqrt() > rad {
                    continue;
                }
                // [id, mask, the fields 1.. whose bit (field - 1) is set]: only what changed
                let mut mask = 0u32;
                let mut vals = Vec::new();
                let prev = prev_rows.get(&vs.id);
                for f in 1..vs.row.len() {
                    if prev.map(|p| p[f] != vs.row[f]).unwrap_or(true) {
                        mask |= 1 << (f - 1);
                        vals.push(vs.row[f].clone());
                    }
                }
                if mask == 0 {
                    k.push(json!(vs.id));
                } else {
                    let mut e = vec![json!(vs.id), json!(mask)];
                    e.extend(vals);
                    v.push(Value::Array(e));
                }
                vids.entry(vs.id)
                    .or_insert_with(|| vinfo.get(&vs.id).cloned().unwrap_or(Value::Null));
                now_rows.insert(vs.id, vs.row.clone());
            }
            prev_rows = now_rows;
            // people: [x, y, state, decision, standing since (day time, 0 walking), on_crossing];
            // a row the same as one of the previous frame's goes by its index there (pk)
            let rows: Vec<Value> = s
                .people
                .iter()
                .filter(|p| ((p.0 - cx).powi(2) + (p.1 - cy).powi(2)).sqrt() <= rad)
                .map(|p| {
                    json!([
                        r1(p.0 - cx),
                        r1(p.1 - cy),
                        si(&p.2),
                        p.3.as_ref().map(|d| json!(si(d))).unwrap_or(Value::Null),
                        if p.4 > 0.5 {
                            (s.t - p.4).round() as i64
                        } else {
                            0
                        },
                        p.5 as u8
                    ])
                })
                .collect();
            let prev_ix: HashMap<&str, usize> = prev_people
                .iter()
                .enumerate()
                .map(|(i, t)| (t.as_str(), i))
                .collect();
            let mut pk = Vec::new();
            let mut carried: Vec<String> = Vec::new();
            let mut p = Vec::new();
            let mut fresh: Vec<String> = Vec::new();
            for row in rows {
                let text = serde_json::to_string(&row)?;
                match prev_ix.get(text.as_str()) {
                    Some(&i) => {
                        pk.push(json!(i));
                        carried.push(text);
                    }
                    None => {
                        p.push(row);
                        fresh.push(text);
                    }
                }
            }
            carried.extend(fresh);
            drop(prev_ix);
            prev_people = carried;
            let mut l = Vec::new();
            for &(c, li, st) in &s.lights {
                if prev_lights.get(&(c, li)) != Some(&st) {
                    l.push(json!([c, li, st]));
                    prev_lights.insert((c, li), st);
                }
            }
            let mut fr = serde_json::Map::new();
            fr.insert("t".into(), json!(r2(s.t)));
            fr.insert("v".into(), Value::Array(v));
            if !k.is_empty() {
                fr.insert("k".into(), Value::Array(k));
            }
            fr.insert("p".into(), Value::Array(p));
            if !pk.is_empty() {
                fr.insert("pk".into(), Value::Array(pk));
            }
            fr.insert("pl".into(), json!(s.players));
            if !l.is_empty() {
                fr.insert("l".into(), Value::Array(l));
            }
            frames.push(Value::Object(fr));
        }
        let lanes = lanes_json(rad);
        let lanes_in: HashSet<u64> = lanes.iter().filter_map(|l| l[0].as_u64()).collect();
        let cr: Vec<Value> = crossings
            .iter()
            .filter(|c| {
                c["lanes"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .any(|x| x.as_u64().map(|x| lanes_in.contains(&x)).unwrap_or(false))
                    })
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        let st: Vec<Value> = stops
            .iter()
            .filter(|s| {
                s[0].as_f64()
                    .unwrap_or(0.0)
                    .hypot(s[1].as_f64().unwrap_or(0.0))
                    <= rad + 50.0
            })
            .cloned()
            .collect();
        let doc = json!({
            "meta": {
                "fixture": fixture.file_name().and_then(|s| s.to_str()),
                "map": r.header.map,
                "center": [r2(cx), r2(cy)],
                "radius": rad,
                "fps": r2(f),
                "mid": mid,
                "mid_hms": hms(mid),
                "start": samples.first().map(|s| s.t),
                "end": samples.last().map(|s| s.t),
                "notes": notes,
                "fields": {
                    "v": "full row [id, x, y, heading_deg, speed, why, stop_dist|null, lead_id|null (lead or body_block), wait_for|null (\"P<n>\" a player), impatience, passing, lane, s, held, hold|null], sent as [id, mask, values of the fields f>=1 whose bit (f-1) is set] (only what changed since the vehicle's previous frame); k: ids unchanged",
                    "p": "[x, y, state(strs index), decision(strs index)|null, standing since (day time; still = t - since; 0 walking), on_crossing]; the frame's people = previous frame's people at the indices pk, then p. Only people on a carriageway (0.5 m), on a crossing, or waiting at a kerb within 2 m of one (unless --all-people)",
                    "pl": "[x, y, heading_deg, speed]",
                    "l": "[ctl, light, state] when it changed (state: 0-2 red, 3-5 red+yellow, 6-8 green, 9-11 yellow)",
                    "lanes": "[index, kind S/W/R, width, light [ctl,li]|null, no_overtaking, density, turn, bus, points]",
                    "vinfo": "id: [front, rear, half_width, class, type]",
                    "stops": "[x, y, heading_deg, lane, light|null]",
                },
            },
            "strs": strs,
            "vinfo": vids,
            "lanes": lanes,
            "crossings": cr,
            "stops": st,
            "frames": frames,
        });
        last_text = serde_json::to_string(&doc)?;
        {
            let part = |k: &str| {
                serde_json::to_string(&doc[k])
                    .map(|t| t.len() / 1024)
                    .unwrap_or(0)
            };
            let fr = doc["frames"].as_array().cloned().unwrap_or_default();
            let sub = |k: &str| {
                fr.iter()
                    .map(|f| serde_json::to_string(&f[k]).map(|t| t.len()).unwrap_or(0))
                    .sum::<usize>()
                    / 1024
            };
            eprintln!("  try fps {f:.2} radius {rad}: {} KB (lanes {} KB, frames {} KB: v {} k {} p {} pl {} l {})", last_text.len() / 1024, part("lanes"), part("frames"), sub("v"), sub("k"), sub("p"), sub("pl"), sub("l"));
        }
        used = (f, rad);
        if last_text.len() <= max_kb * 1024 {
            break;
        }
    }
    std::fs::write(out, &last_text)?;
    println!(
        "{}: {:.0} KB, fps {:.2}, radius {} m, centre ({:.1}, {:.1}), {} samples",
        out.display(),
        last_text.len() as f64 / 1024.0,
        used.0,
        used.1,
        cx,
        cy,
        samples.len()
    );
    Ok(())
}

// ---------------------------------------------------------------------------------------
// diagnose

struct Snap {
    t: f64,
    vehicles: Vec<Vehicle>,
    by_id: HashMap<u64, usize>,
    people: Vec<PersonRecord>,
    lights: Vec<TrafficLightController>,
    players: Vec<omsi_traffic::PlayerView>,
    /// The junction way each vehicle near the bus claims (perceived again after the frame):
    /// (junction, movement, distance to the movement's lane, the chain of pieces).
    jv: HashMap<u64, (usize, usize, f32, Vec<(usize, f32)>)>,
}

#[derive(Default, Clone)]
struct Stand {
    start: f64,
    end: f64,
    /// Seconds stood already when the window began (the vehicle stood at the start).
    before: f64,
    whys: BTreeMap<&'static str, u32>,
    ended: bool,
}

struct Hop {
    id: u64,
    text: String,
    /// The next vehicle and how this one is linked to it.
    next: Option<(u64, &'static str)>,
    /// (vehicle index in the snapshot, the decision used: the current one, or the last
    /// one with a reason when the vehicle stands with none now)
    eff: Option<(usize, omsi_traffic::Decision)>,
    standing: bool,
}

fn snap_idx(snaps: &[Snap], t: f64) -> usize {
    (0..snaps.len())
        .min_by(|&a, &b| (snaps[a].t - t).abs().total_cmp(&(snaps[b].t - t).abs()))
        .unwrap_or(0)
}

fn decision_next(d: &omsi_traffic::Decision) -> Option<(u64, &'static str)> {
    match d.why {
        Why::Junction | Why::ExitFull => d
            .wait_for
            .map(|w| (w, "yields to"))
            .or(d.lead_id.map(|w| (w, "behind")))
            .or(d.body_block.map(|w| (w, "body-blocked by"))),
        Why::Lead | Why::Merge | Why::LetOut => d
            .lead_id
            .map(|w| (w, "behind"))
            .or(d.body_block.map(|w| (w, "body-blocked by")))
            .or(d.wait_for.map(|w| (w, "waits for"))),
        _ => d.body_block.map(|w| (w, "body-blocked by")),
    }
}

/// One vehicle's decision in snapshot `k`, and what it waits on.
fn hop(net: &Network, jn: &omsi_traffic::Junctions, snaps: &[Snap], k: usize, id: u64) -> Hop {
    let s = &snaps[k];
    if id >= PLAYER_ID_BASE {
        let n = (id - PLAYER_ID_BASE) as u32;
        let text = s
            .players
            .iter()
            .find(|p| p.id == n)
            .map(|p| {
                format!(
                    "{} (the player's bus) v {:.2} still {:.0}s blinker {}",
                    fmt_id(id),
                    p.speed,
                    p.still,
                    p.blinker
                )
            })
            .unwrap_or(format!("{} (player)", fmt_id(id)));
        let standing = s
            .players
            .iter()
            .find(|p| p.id == n)
            .map(|p| p.speed.abs() < 0.3)
            .unwrap_or(false);
        return Hop {
            id,
            text,
            next: None,
            eff: None,
            standing,
        };
    }
    let Some(&i) = s.by_id.get(&id) else {
        return Hop {
            id,
            text: format!("{} (not on the road)", fmt_id(id)),
            next: None,
            eff: None,
            standing: false,
        };
    };
    let v = &s.vehicles[i];
    let mut d = v.last;
    let standing = v.state.speed < 0.3;
    let mut recent = String::new();
    // a car let go stands a moment with no reason (the start reaction): what held it last
    if standing && d.why == Why::None && !d.held {
        for j in (k.saturating_sub(10)..k).rev() {
            if let Some(&i2) = snaps[j].by_id.get(&id) {
                let d2 = snaps[j].vehicles[i2].last;
                if d2.why != Why::None {
                    d = d2;
                    recent = format!(" (why=none now; {:.1}s earlier)", s.t - snaps[j].t);
                    break;
                }
            }
        }
    }
    let lane_name = net
        .lanes
        .get(v.state.lane)
        .map(|l| l.name.rsplit(['\\', '/']).next().unwrap_or("").to_string())
        .unwrap_or_default();
    let mut text = format!(
        "{} {:?} lane {} ({}) s {:.1} v {:.2} stood {:.0}s why={}{} stop={} held={} hold={} imp={:.2}{}{}",
        fmt_id(id),
        v.class,
        v.state.lane,
        lane_name,
        v.state.s,
        v.state.speed,
        v.stopped.max(v.progress.1),
        why_name(d.why),
        recent,
        d.stop.map(|x| format!("{x:.1}")).unwrap_or("-".into()),
        d.held,
        hold_name(d.hold).unwrap_or("-"),
        v.impatience.value,
        if v.passing.is_some() { " PASSING" } else { "" },
        if v.external.at_stop { " AT_STOP" } else { "" },
    );
    let next = decision_next(&d);
    if let Some((j, m, entry, _)) = s.jv.get(&id) {
        let ml = jn
            .nodes
            .get(*j)
            .and_then(|n| n.movements.get(*m))
            .map(|mv| mv.lane);
        let turn = ml
            .and_then(|l| net.lanes.get(l))
            .map(|l| match l.turn {
                1 => "left",
                2 => "right",
                _ => "straight",
            })
            .unwrap_or("?");
        let _ = write!(
            text,
            " [junction {j} movement {m} (lane {}, {turn}) in {entry:.1} m]",
            ml.map(|l| l.to_string()).unwrap_or("?".into())
        );
    }
    if let Some((c, li, dist)) = light_on_way(net, v) {
        if dist < 80.0 {
            let st = s.lights.get(c).map(|x| x.state(li)).unwrap_or(-1);
            let _ = write!(
                text,
                " [light {c}/{li} {} in {:.0} m]",
                aspect_name(st),
                dist
            );
        }
    }
    if next.is_none() {
        if let Some(seen) = d.lead_seen {
            let _ = write!(text, " lead_seen={seen:?}");
        }
        if let Some(l) = d.lead {
            let _ = write!(text, " lead_gap={:.1} lead_v={:.1}", l.gap, l.speed);
        }
    }
    Hop {
        id,
        text,
        next,
        eff: Some((i, d)),
        standing,
    }
}

fn light_on_way(net: &Network, v: &Vehicle) -> Option<(usize, usize, f32)> {
    way_lanes(net, &v.state, 150.0)
        .into_iter()
        .find_map(|(l, d)| net.lanes[l].traffic_light.map(|(c, li)| (c, li, d)))
}

struct Chain {
    id: u64,
    k: usize,
    hops: Vec<Hop>,
    /// Index in `hops` where the cycle starts.
    cycle: Option<usize>,
}

fn build_chain(
    net: &Network,
    jn: &omsi_traffic::Junctions,
    snaps: &[Snap],
    k: usize,
    start: u64,
) -> Chain {
    let mut hops: Vec<Hop> = Vec::new();
    let mut cur = start;
    let mut cycle = None;
    loop {
        if let Some(p) = hops.iter().position(|h| h.id == cur) {
            cycle = Some(p);
            break;
        }
        let mut h = hop(net, jn, snaps, k, cur);
        let next = h.next;
        // how the two ways through meet, by the junction table
        if let Some((n, "yields to")) = next {
            if let (Some((j1, m1, _, _)), Some((j2, m2, _, c2))) =
                (snaps[k].jv.get(&cur), snaps[k].jv.get(&n))
            {
                if j1 == j2 {
                    let foes = &jn.nodes[*j1].movements[*m1].foes;
                    let rel: Vec<String> = foes
                        .iter()
                        .filter(|f| f.movement == *m2 || c2.iter().any(|c| c.0 == f.movement))
                        .map(|f| {
                            format!(
                                "{:?} yields={} mine {:.1}..{:.1} theirs {:.1}..{:.1}",
                                f.kind, f.yields, f.mine.0, f.mine.1, f.theirs.0, f.theirs.1
                            )
                        })
                        .collect();
                    let _ = write!(
                        h.text,
                        " {{foe m{m2}: {}}}",
                        if rel.is_empty() {
                            "NOT in my foe list".to_string()
                        } else {
                            rel.join("; ")
                        }
                    );
                } else {
                    let _ = write!(h.text, " {{foe at another junction {j2}}}");
                }
            }
        }
        hops.push(h);
        match next {
            Some((n, _)) if hops.len() < 40 => cur = n,
            _ => break,
        }
    }
    Chain {
        id: start,
        k,
        hops,
        cycle,
    }
}

fn aspect_name(s: i32) -> &'static str {
    match s {
        0..=2 => "red",
        3..=5 => "red+yellow",
        6 | 7 => "green",
        8 => "green+yellow",
        9..=11 => "yellow",
        _ => "dark",
    }
}

/// Why the terminal of a chain stands: (class, detail).
fn terminal_reason(
    net: &Network,
    snaps: &[Snap],
    k: usize,
    h: &Hop,
    t0: f64,
    t1: f64,
) -> (&'static str, String) {
    let s = &snaps[k];
    if h.id >= PLAYER_ID_BASE {
        return ("player", format!("the player's bus: {}", h.text));
    }
    let Some((i, d)) = h.eff else {
        return ("other", format!("{} is not on the road", fmt_id(h.id)));
    };
    let v = &s.vehicles[i];
    let t = h.id;
    // the terminal's own reasons over the window while it stood
    let mut hist: BTreeMap<&'static str, u32> = BTreeMap::new();
    let mut stood = 0.0;
    for sn in snaps.iter().filter(|x| x.t >= t0 && x.t <= t1) {
        if let Some(&j) = sn.by_id.get(&t) {
            let w = &sn.vehicles[j];
            if w.state.speed < 0.3 {
                stood += 0.5;
                *hist.entry(why_name(w.last.why)).or_default() += 1;
            }
        }
    }
    let tot: u32 = hist.values().sum::<u32>().max(1);
    let hist_s = hist
        .iter()
        .map(|(k, n)| format!("{k} {:.0}%", *n as f64 * 100.0 / tot as f64))
        .collect::<Vec<_>>()
        .join(", ");
    let own = format!(
        " [{} stood {:.0} s of the window: {}]",
        fmt_id(t),
        stood,
        hist_s
    );
    if !h.standing {
        return (
            "transient",
            format!(
                "the chain ends at {} which moves (v {:.1}, why={}){own}",
                fmt_id(t),
                v.state.speed,
                why_name(d.why)
            ),
        );
    }
    let r = match d.why {
        Why::People | Why::Zebra => {
            let at = d.stop.map(|x| v.state.way_point(net, x)).unwrap_or(v.pos);
            let near = s
                .people
                .iter()
                .map(|p| (dist2(p.pos, at.x, at.y), p))
                .min_by(|a, b| a.0.total_cmp(&b.0));
            match near {
                Some((dd, p)) => (
                    "pedestrian",
                    format!(
                        "{} stops ({}) for a person {:.1} m from its stop point: state \"{}\", decision {:?}, still {:.0} s, on_crossing {}, speed {:.2}{}",
                        fmt_id(t),
                        why_name(d.why),
                        dd,
                        p.state,
                        p.decision.as_deref().unwrap_or("-"),
                        p.still,
                        p.on_crossing,
                        p.vel.length(),
                        p.leg.map(|(l, s, len)| format!(", footpath {l} at {s:.1}/{len:.1}")).unwrap_or_default()
                    ),
                ),
                None => ("pedestrian", format!("{} stops ({}) but nobody is near its stop point", fmt_id(t), why_name(d.why))),
            }
        }
        Why::Light => match light_on_way(net, v) {
            Some((c, li, dist)) => {
                let st_now = s.lights.get(c).map(|x| x.state(li)).unwrap_or(-1);
                let green = snaps[k..]
                    .iter()
                    .find(|x| {
                        x.t <= t1 + 1.0
                            && TrafficLightController::allows_go(
                                x.lights.get(c).map(|x| x.state(li)).unwrap_or(0),
                            )
                    })
                    .map(|x| x.t);
                let green_any = snaps.iter().any(|x| {
                    x.t >= t0
                        && x.t <= t1
                        && TrafficLightController::allows_go(
                            x.lights.get(c).map(|x| x.state(li)).unwrap_or(0),
                        )
                });
                match green {
                    Some(g) => {
                        let left = snaps.iter().find(|x| x.t >= g && x.by_id.get(&t).map(|&j| x.vehicles[j].state.speed > 0.3).unwrap_or(true)).map(|x| x.t);
                        let after = match left {
                            Some(l) if l - g > 6.0 => format!("it STAYED {:.0} s after the green", l - g),
                            Some(l) => format!("it left {:.0} s after", l - g),
                            None => "it NEVER LEFT after the green".to_string(),
                        };
                        ("light", format!("{} at light {c}/{li} ({:.0} m ahead) state {st_now} ({}); green at {}; {after}", fmt_id(t), dist, aspect_name(st_now), hms(g)))
                    }
                    None => ("light", format!("{} at light {c}/{li} ({:.0} m ahead) state {st_now} ({}); not green again in the window (green at all in it: {green_any})", fmt_id(t), dist, aspect_name(st_now))),
                }
            }
            None => (
                "light",
                format!(
                    "{} why=light but no light on its way within 150 m",
                    fmt_id(t)
                ),
            ),
        },
        _ if v.passing.is_some() => (
            "passing",
            format!(
                "{} is passing ({:?}) why={}",
                fmt_id(t),
                v.passing.map(|p| (p.lane, p.aborted)),
                why_name(d.why)
            ),
        ),
        Why::Junction | Why::ExitFull => (
            "junction-nofoe",
            format!(
                "{} why={} hold={} with no foe named",
                fmt_id(t),
                why_name(d.why),
                hold_name(d.hold).unwrap_or("-")
            ),
        ),
        Why::Service => (
            "service",
            format!(
                "{} is a bus at its stop (at_stop {}, standing_for {:.0} s)",
                fmt_id(t),
                v.external.at_stop,
                v.external.standing_for
            ),
        ),
        Why::Lead => match d.lead_seen {
            Some(Seen::Parked) => (
                "parked",
                format!(
                    "{} is behind a PARKED car (gap {:.1})",
                    fmt_id(t),
                    d.lead.map(|l| l.gap).unwrap_or(-1.0)
                ),
            ),
            Some(Seen::Player(p)) => (
                "player",
                format!(
                    "{} follows the player's bus {p} (gap {:.1})",
                    fmt_id(t),
                    d.lead.map(|l| l.gap).unwrap_or(-1.0)
                ),
            ),
            other => (
                "lead-standing",
                format!(
                    "{} why=lead, lead {:?} not a vehicle on the road",
                    fmt_id(t),
                    other
                ),
            ),
        },
        _ => (
            "lead-standing",
            format!(
                "{} stands with why={} stop={:?} held={} (names no reason)",
                fmt_id(t),
                why_name(d.why),
                d.stop,
                d.held
            ),
        ),
    };
    (r.0, format!("{}{own}", r.1))
}

fn diagnose(fixture: &Path, args: &[String]) -> Res<()> {
    let radius: f64 = opt(args, "--radius")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(200.0);
    let min_stand: f64 = opt(args, "--min-stand")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(15.0);
    let mut r = Replay::load(fixture)?;
    let mid = window_mid(fixture, &r);
    let c = player_at(&r, mid).ok_or("no player")?;
    let (cx, cy) = (c.x, c.y);
    let (t0, t1) = (mid - 45.0, mid + 45.0);
    if !r.at(0) {
        return Err("no keyframe".into());
    }
    println!(
        "== {} : window {} .. {} (mid {}), centre = player at mid ({:.1}, {:.1}), radius {} m",
        fixture.display(),
        hms(t0),
        hms(t1),
        hms(mid),
        cx,
        cy,
        radius
    );
    for x in &r.records {
        if let Record::Note { time, text } = x {
            println!(
                "game note t={time:.1}: {}",
                text.lines().next().unwrap_or("")
            );
        }
    }
    let mut stands: HashMap<u64, Stand> = HashMap::new();
    let mut done: HashMap<u64, Vec<Stand>> = HashMap::new();
    let mut snaps: Vec<Snap> = Vec::new();
    let mut next_snap = f64::NEG_INFINITY;
    let mut first = true;
    let mut last_t = 0.0;
    let mut player_track: Vec<(f64, f64, f64, f32)> = Vec::new();
    while r.step() {
        let t = r.sim.day_time;
        last_t = t;
        r.sim.update_positions(&r.net, &mut r.vehicles);
        let mut present: HashSet<u64> = HashSet::new();
        // (the area: round the bus at the window's middle, and round the bus as it drives)
        let bus = r.sim.players.first().map(|p| p.pos);
        let in_area = |q: DVec3, extra: f64| {
            dist2(q, cx, cy) <= radius + extra
                || bus
                    .map(|b| dist2(q, b.x, b.y) <= 150.0 + extra)
                    .unwrap_or(false)
        };
        for v in &r.vehicles {
            if !in_area(v.pos, 0.0) {
                continue;
            }
            present.insert(v.id);
            let standing = v.state.speed < 0.3;
            if standing {
                let e = stands.entry(v.id).or_insert_with(|| Stand {
                    start: t,
                    end: t,
                    before: if first {
                        v.stopped.max(0.0) as f64
                    } else {
                        0.0
                    },
                    ..Default::default()
                });
                e.end = t;
                *e.whys.entry(why_name(v.last.why)).or_default() += 1;
            } else if let Some(mut s) = stands.remove(&v.id) {
                s.ended = true;
                done.entry(v.id).or_default().push(s);
            }
        }
        // left the area or the road
        let gone: Vec<u64> = stands
            .keys()
            .filter(|k| !present.contains(*k))
            .copied()
            .collect();
        for k in gone {
            if let Some(mut s) = stands.remove(&k) {
                s.ended = true;
                done.entry(k).or_default().push(s);
            }
        }
        first = false;
        if t >= next_snap {
            next_snap = t + 0.5;
            let by_id = r
                .vehicles
                .iter()
                .enumerate()
                .map(|(i, v)| (v.id, i))
                .collect();
            if let Some(p) = r.sim.players.first() {
                player_track.push((t, p.pos.x, p.pos.y, p.speed));
            }
            let mut jv = HashMap::new();
            // (the frame's occupancy has the indices from before vehicles were taken off;
            // the next frame builds it again in any case)
            r.sim.build_occupancy(&r.net, &r.vehicles);
            for (i, v) in r.vehicles.iter().enumerate() {
                if !in_area(v.pos, 60.0) {
                    continue;
                }
                let p = omsi_traffic::perception::perceive(&r.net, &r.sim, &r.vehicles, i);
                if let Some(j) = p.junction {
                    jv.insert(v.id, (j.junction, j.movement, j.entry, j.chain.clone()));
                }
            }
            snaps.push(Snap {
                t,
                vehicles: r.vehicles.clone(),
                by_id,
                people: frame_people(&r)
                    .iter()
                    .filter(|p| in_area(p.pos, 50.0))
                    .cloned()
                    .collect(),
                lights: r.sim.lights.clone(),
                players: r.sim.players.clone(),
                jv,
            });
        }
    }
    for (k, s) in stands.drain() {
        done.entry(k).or_default().push(s);
    }
    if let Some((t, ds, id)) = r.worst() {
        println!("replay divergence at keyframes: worst {ds:.2} m (vehicle {id} at t={t:.1}); keyframes compared {}", r.divergence.len());
    }
    // player
    if let (Some(a), Some(b)) = (player_track.first(), player_track.last()) {
        let moved: f64 = player_track
            .windows(2)
            .map(|w| ((w[1].1 - w[0].1).powi(2) + (w[1].2 - w[0].2).powi(2)).sqrt())
            .sum();
        let still = player_track.iter().filter(|p| p.3.abs() < 0.3).count() as f64 * 0.5;
        println!(
            "player: {} ({:.1},{:.1}) -> {} ({:.1},{:.1}), drove {:.0} m, stood {:.0} s",
            hms(a.0),
            a.1 - cx,
            a.2 - cy,
            hms(b.0),
            b.1 - cx,
            b.2 - cy,
            moved,
            still
        );
    }
    // the player's stands: what was around the bus
    {
        let mut k = 0;
        while k < snaps.len() {
            let still = |sn: &Snap| {
                sn.players
                    .first()
                    .map(|p| p.speed.abs() < 0.3)
                    .unwrap_or(false)
            };
            if !still(&snaps[k]) || snaps[k].t < t0 || snaps[k].t > t1 {
                k += 1;
                continue;
            }
            let a = k;
            while k < snaps.len() && still(&snaps[k]) {
                k += 1;
            }
            let b = k - 1;
            let dur = snaps[b].t - snaps[a].t;
            if dur < 8.0 {
                continue;
            }
            let m = (a + b) / 2;
            let sn = &snaps[m];
            let p = sn.players[0];
            let f = p.fwd();
            let front = p.pos.truncate() + f * p.half_len as f64;
            println!("PLAYER stood {:.0} s ({} .. {}) at ({:.0},{:.0}) heading {:.0}, blinker {}; at {}:", dur, hms(snaps[a].t), hms(snaps[b].t), p.pos.x - cx, p.pos.y - cy, p.heading, p.blinker, hms(sn.t));
            let mut near: Vec<(f64, usize)> = sn
                .vehicles
                .iter()
                .enumerate()
                .map(|(i, v)| (dist2(v.pos, p.pos.x, p.pos.y), i))
                .filter(|x| x.0 < 35.0)
                .collect();
            near.sort_by(|x, y| x.0.total_cmp(&y.0));
            for (d, i) in near.into_iter().take(12) {
                let v = &sn.vehicles[i];
                let rel = (v.pos.truncate() - p.pos.truncate()).dot(f);
                let h = hop(&r.net, &r.sim.junctions, &snaps, m, v.id);
                let link = h
                    .next
                    .map(|(n, how)| format!(" --{how}--> {}", fmt_id(n)))
                    .unwrap_or_default();
                println!(
                    "    {:>4.0} m ({}): {}{}",
                    d,
                    if rel >= 0.0 { "ahead" } else { "behind" },
                    h.text,
                    link
                );
            }
            for q in sn
                .people
                .iter()
                .filter(|q| ((q.pos.truncate() - front).length()) < 12.0)
                .take(8)
            {
                println!("    person {:.1} m from the bus front: state \"{}\", decision {:?}, still {:.0} s, on_crossing {}, speed {:.2}{}", (q.pos.truncate() - front).length(), q.state, q.decision.as_deref().unwrap_or("-"), q.still, q.on_crossing, q.vel.length(), q.leg.map(|(l, s, len)| format!(", footpath {l} at {s:.1}/{len:.1}")).unwrap_or_default());
            }
        }
    }
    // the stuck vehicles: their longest stand inside the window
    let mut stuck: Vec<(u64, Stand)> = Vec::new();
    for (id, list) in &done {
        let best = list
            .iter()
            .map(|s| (s.end.min(t1) - s.start.max(t0), s))
            .filter(|(d, _)| *d > 0.0)
            .max_by(|a, b| a.0.total_cmp(&b.0));
        if let Some((d, s)) = best {
            if d >= min_stand {
                stuck.push((*id, s.clone()));
            }
        }
    }
    stuck.sort_by(|a, b| a.1.start.total_cmp(&b.1.start).then(a.0.cmp(&b.0)));
    // the lights near the bus over the window: how long each showed what
    {
        let mut lights: Vec<(usize, usize)> = Vec::new();
        for (l, lane) in r.net.lanes.iter().enumerate() {
            let _ = l;
            if let Some(li) = lane.traffic_light {
                if lane
                    .points
                    .first()
                    .map(|p| dist2(*p, cx, cy) <= 150.0)
                    .unwrap_or(false)
                    && !lights.contains(&li)
                {
                    lights.push(li);
                }
            }
        }
        lights.sort();
        if !lights.is_empty() {
            println!("lights within 150 m (seconds in the window: green / red / other; changes; clock held at a stop point):");
        }
        for (c, li) in lights {
            let (mut g, mut red, mut o, mut held, mut changes) = (0.0, 0.0, 0.0, 0.0, 0);
            let mut last = None;
            for sn in snaps.iter().filter(|x| x.t >= t0 && x.t <= t1) {
                let Some(ctl) = sn.lights.get(c) else {
                    continue;
                };
                let st = ctl.state(li);
                match aspect_name(st) {
                    "green" | "green+yellow" => g += 0.5,
                    "red" => red += 0.5,
                    _ => o += 0.5,
                }
                if ctl.held {
                    held += 0.5;
                }
                if last.is_some() && last != Some(aspect_name(st)) {
                    changes += 1;
                }
                last = Some(aspect_name(st));
            }
            let prog = r
                .sim
                .lights
                .get(c)
                .map(|x| {
                    format!(
                        "cycle {:.0} s, phases {:?}, stop points {}",
                        x.cycle,
                        x.lights.get(li).cloned().unwrap_or_default(),
                        x.stops.len()
                    )
                })
                .unwrap_or_default();
            println!("  light {c}/{li}: green {g:.0} / red {red:.0} / other {o:.0}; {changes} changes; held {held:.0} s; {prog}");
        }
    }
    println!(
        "vehicles within {radius} m that stood >= {min_stand} s in the window: {}",
        stuck.len()
    );
    let net = &r.net;
    // one moment for all: when most of them stand together (chains read at one instant
    // are consistent); a vehicle not standing then is read at the middle of its stand
    let covers = |s: &Stand, t: f64| s.start <= t && s.end >= t;
    let mut kstar = 0;
    let mut best = 0;
    for (k, sn) in snaps.iter().enumerate() {
        if sn.t < t0 || sn.t > t1 {
            continue;
        }
        let n = stuck.iter().filter(|(_, s)| covers(s, sn.t)).count();
        if n > best {
            kstar = k;
            best = n;
        }
    }
    if !stuck.is_empty() {
        println!(
            "common moment: {} ({} of {} standing)",
            hms(snaps[kstar].t),
            best,
            stuck.len()
        );
    }
    let chains: Vec<Chain> = stuck
        .iter()
        .map(|(id, s)| {
            let k = if covers(s, snaps[kstar].t) {
                kstar
            } else {
                snap_idx(&snaps, (s.start.max(t0) + s.end.min(t1)) * 0.5)
            };
            build_chain(net, &r.sim.junctions, &snaps, k, *id)
        })
        .collect();
    for ((id, s), ch) in stuck.iter().zip(&chains) {
        let total: u32 = s.whys.values().sum();
        let pct: Vec<String> = s
            .whys
            .iter()
            .map(|(k, n)| format!("{k} {:.0}%", *n as f64 * 100.0 / total.max(1) as f64))
            .collect();
        let snap = &snaps[ch.k];
        let (lane, sv, pos) = snap
            .by_id
            .get(id)
            .map(|&i| {
                (
                    snap.vehicles[i].state.lane,
                    snap.vehicles[i].state.s,
                    snap.vehicles[i].pos,
                )
            })
            .unwrap_or((usize::MAX, 0.0, DVec3::ZERO));
        println!();
        println!(
            "{} lane {} s {:.1} at ({:.0},{:.0}) from the bus: stood {:.0} s in the window ({} .. {}{}){}; why {}",
            fmt_id(*id),
            lane,
            sv,
            pos.x - cx,
            pos.y - cy,
            s.end.min(t1) - s.start.max(t0),
            hms(s.start),
            hms(s.end),
            if s.ended { "" } else { ", still standing at the end" },
            if s.before > 0.5 { format!(", already {:.0} s before", s.before) } else { String::new() },
            pct.join(", ")
        );
        println!(
            "  chain at {}{}:",
            hms(snap.t),
            if ch.cycle.is_some() { " (CYCLE)" } else { "" }
        );
        for (k, h) in ch.hops.iter().enumerate() {
            let link = h
                .next
                .map(|(n, how)| format!("  --{how}--> {}", fmt_id(n)))
                .unwrap_or_default();
            println!("    {}{}{}", "  ".repeat(k.min(6)), h.text, link);
        }
    }
    // groups: by the cycle, else by the terminal
    println!();
    println!("== ROOT CAUSES");
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (k, ch) in chains.iter().enumerate() {
        let key = match ch.cycle {
            Some(p) => {
                let mut ids: Vec<u64> = ch.hops[p..].iter().map(|h| h.id).collect();
                ids.sort();
                format!("cycle {ids:?}")
            }
            None => format!("t {}", ch.hops.last().map(|h| h.id).unwrap_or(ch.id)),
        };
        groups.entry(key).or_default().push(k);
    }
    let mut lines_out: Vec<(f64, String)> = Vec::new();
    let mut focus: Vec<(f64, f64, f64, &'static str, f64)> = Vec::new();
    let mut class_count: BTreeMap<&'static str, usize> = BTreeMap::new();
    for members in groups.values() {
        let ch = &chains[members[0]];
        let ids: Vec<String> = members.iter().map(|&k| fmt_id(chains[k].id)).collect();
        let longest = members
            .iter()
            .map(|&k| {
                let s = &stuck[k].1;
                s.end.min(t1) - s.start.max(t0)
            })
            .fold(0.0, f64::max);
        let all_ended = members.iter().all(|&k| stuck[k].1.ended);
        let resolved = if all_ended {
            "resolved in the window"
        } else {
            "STILL STANDING at the window's end"
        };
        let snap = &snaps[ch.k];
        let last = ch.hops.last().unwrap();
        let where_ = snap
            .by_id
            .get(&last.id)
            .map(|&i| {
                format!(
                    " at ({:.0},{:.0})",
                    snap.vehicles[i].pos.x - cx,
                    snap.vehicles[i].pos.y - cy
                )
            })
            .unwrap_or_default();
        let (class, detail) = match ch.cycle {
            Some(p) => {
                let cyc = &ch.hops[p..];
                let moving: Vec<String> = cyc
                    .iter()
                    .filter(|h| !h.standing)
                    .map(|h| fmt_id(h.id))
                    .collect();
                let parts: Vec<String> = cyc
                    .iter()
                    .map(|h| format!("{} {}", fmt_id(h.id), h.next.map(|n| n.1).unwrap_or("")))
                    .collect();
                let kind = if moving.is_empty() {
                    format!("all {} stand", cyc.len())
                } else {
                    format!("transient: {} moving", moving.join(" "))
                };
                (
                    "cycle",
                    format!(
                        "{} -> {} ({kind}) at {}",
                        parts.join(" -> "),
                        fmt_id(cyc[0].id),
                        hms(snap.t)
                    ),
                )
            }
            None => {
                let prev_link = ch
                    .hops
                    .iter()
                    .rev()
                    .nth(1)
                    .and_then(|h| h.next)
                    .map(|n| n.1);
                let (c, d) = terminal_reason(net, &snaps, ch.k, last, t0, t1);
                // the one that yields: does it face a red light itself (then the yield
                // costs nothing, the light holds it anyway)?
                let yielder_red =
                    ch.hops
                        .iter()
                        .rev()
                        .nth(1)
                        .and_then(|h| h.eff)
                        .and_then(|(i, _)| {
                            let v = &snap.vehicles[i];
                            let (lc, li, dist) = light_on_way(net, v)?;
                            let st = snap.lights.get(lc)?.state(li);
                            (!TrafficLightController::allows_go(st) && dist < 15.0).then(|| {
                                format!("light {lc}/{li} {} in {dist:.0} m", aspect_name(st))
                            })
                        });
                match (prev_link, c) {
                    (Some("yields to"), c) if last.standing && c != "player" && yielder_red.is_some() => (
                        "light",
                        format!(
                            "{} faces its own red ({}) and also yields to {} which stands: {c}: {d}",
                            ch.hops.iter().rev().nth(1).map(|h| fmt_id(h.id)).unwrap_or_default(),
                            yielder_red.clone().unwrap_or_default(),
                            fmt_id(last.id)
                        ),
                    ),
                    (Some("yields to"), c) if last.standing && c != "player" => (
                        "junction-foe",
                        format!(
                            "{} yields to {} which stands: {c}: {d}",
                            ch.hops
                                .iter()
                                .rev()
                                .nth(1)
                                .map(|h| fmt_id(h.id))
                                .unwrap_or_default(),
                            fmt_id(last.id)
                        ),
                    ),
                    _ => (c, d),
                }
            }
        };
        *class_count.entry(class).or_default() += 1;
        // how near the bus's way through the window it is
        let (path_d, abs) = match snap.by_id.get(&last.id) {
            Some(&i) => {
                let q = snap.vehicles[i].pos;
                (
                    player_track
                        .iter()
                        .filter(|p| p.0 >= t0 && p.0 <= t1)
                        .map(|p| ((q.x - p.1).powi(2) + (q.y - p.2).powi(2)).sqrt())
                        .fold(f64::INFINITY, f64::min),
                    Some(q),
                )
            }
            None => (f64::INFINITY, None),
        };
        if let Some(q) = abs {
            focus.push((members.len() as f64 * longest, q.x, q.y, class, path_d));
        }
        lines_out.push((-longest, format!("ROOT CAUSE {class} [{} vehicles: {}; longest {:.0} s; {}]{where_}, {:.0} m from the bus's path: {}", members.len(), ids.join(" "), longest, resolved, path_d, detail)));
    }
    lines_out.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (_, l) in &lines_out {
        println!("{l}");
    }
    if stuck.is_empty() {
        println!("ROOT CAUSE none: nothing stood >= {min_stand} s within {radius} m in the window");
    }
    println!(
        "== SUMMARY: {} stuck vehicles, {} groups: {:?}",
        stuck.len(),
        groups.len(),
        class_count
    );
    // the place to look at: the biggest stuck group near the bus's way
    let pick = focus
        .iter()
        .filter(|f| f.4 <= 120.0)
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .or(focus.iter().max_by(|a, b| a.0.total_cmp(&b.0)));
    if let Some(f) = pick {
        println!(
            "FOCUS {:.1},{:.1} ({} group, {:.0} m from the bus's path)",
            f.1, f.2, f.3, f.4
        );
    }
    let _ = last_t;
    Ok(())
}

// ---------------------------------------------------------------------------------------
// viewer

const VIEWER_HTML: &str = include_str!("traffic_cases_viewer.html");
/// A case page (its HTML, the trace and the diagnosis in it) stays under this.
const PAGE_MAX: usize = 445 * 1024;

fn esc_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// One vehicle of a diagnosis that stood long.
struct DiagCar {
    id: String,
    start: String,
    stood: u32,
    why: String,
    head: String,
    at: String,
    cycle: bool,
    hops: Vec<String>,
}

/// What a `caseNN-diagnose.txt` says, for the pages.
struct Diag {
    head: Vec<String>,
    roots: Vec<String>,
    cars: Vec<DiagCar>,
}

fn parse_diag(text: &str) -> Diag {
    let lines: Vec<&str> = text.lines().collect();
    let mut d = Diag {
        head: Vec::new(),
        roots: Vec::new(),
        cars: Vec::new(),
    };
    let mut i = 0;
    while i < lines.len() {
        let l = lines[i];
        if l.starts_with("ROOT CAUSE") {
            d.roots.push(l.to_string());
        } else if l.starts_with("== ") && !l.starts_with("== ROOT")
            || l.starts_with("player:")
            || l.starts_with("common moment")
            || l.starts_with("replay divergence")
        {
            d.head.push(l.to_string());
        } else if l.starts_with('#')
            && l.as_bytes().get(1).is_some_and(|c| c.is_ascii_digit())
            && l.contains(" in the window (")
        {
            let id: String = l[1..].chars().take_while(|c| c.is_ascii_digit()).collect();
            let stood = l
                .split("stood ")
                .nth(1)
                .and_then(|s| s.split(' ').next())
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0)
                .round() as u32;
            let start = l
                .split("in the window (")
                .nth(1)
                .map(|s| s.chars().take(8).collect())
                .unwrap_or_default();
            // the commonest reason while it stood
            let why = l
                .rsplit("; why ")
                .next()
                .map(|s| {
                    s.split(", ")
                        .filter_map(|p| {
                            let (w, pct) = p.rsplit_once(' ')?;
                            Some((
                                pct.trim_end_matches('%').parse::<f64>().ok()?,
                                w.to_string(),
                            ))
                        })
                        .max_by(|a, b| a.0.total_cmp(&b.0))
                        .map(|x| x.1)
                        .unwrap_or_default()
                })
                .unwrap_or_default();
            let mut at = String::new();
            let mut cycle = false;
            let mut raw: Vec<String> = Vec::new();
            let mut j = i + 1;
            if let Some(c) = lines
                .get(j)
                .filter(|c| c.trim_start().starts_with("chain at "))
            {
                at = c
                    .trim()
                    .trim_start_matches("chain at ")
                    .chars()
                    .take(8)
                    .collect();
                cycle = c.contains("CYCLE");
                j += 1;
                while j < lines.len() && lines[j].starts_with("    ") {
                    raw.push(lines[j].trim().to_string());
                    j += 1;
                }
            }
            // a hop short (to its reason), the last one whole; a long chain's middle left out
            let n = raw.len();
            let short = |h: &str| -> String {
                let (main, link) = match h.find("  --") {
                    Some(k) => (&h[..k], &h[k..]),
                    None => (h, ""),
                };
                let main = main.split(" stop=").next().unwrap_or(main);
                // (the lane's file name adds nothing here)
                let main = match (main.find(" ("), main.find(".sli)")) {
                    (Some(a), Some(b)) if b > a => format!("{}{}", &main[..a], &main[b + 5..]),
                    _ => main.replace(" ()", ""),
                };
                format!("{main}{}", link.replace("  ", " "))
            };
            let mut hops: Vec<String> = Vec::new();
            for (k, h) in raw.iter().enumerate() {
                if n > 6 && k == 3 {
                    hops.push(format!("… {} more …", n - 5));
                }
                if n > 6 && (3..n - 2).contains(&k) {
                    continue;
                }
                hops.push(if k + 1 == n {
                    h.replace(" ()", "")
                } else {
                    short(h)
                });
            }
            d.cars.push(DiagCar {
                id,
                start,
                stood,
                why,
                head: l.to_string(),
                at,
                cycle,
                hops,
            });
            i = j;
            continue;
        }
        i += 1;
    }
    d
}

/// `#123` in (escaped) text as links to `page#pin=123&t=…`, at the vehicle's stand start
/// when it was diagnosed, else at `t`.
fn linkify(text: &str, page: &str, starts: &HashMap<String, String>, t: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(k) = rest.find('#') {
        out.push_str(&rest[..k]);
        let digits: String = rest[k + 1..]
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        if digits.is_empty() {
            out.push('#');
            rest = &rest[k + 1..];
            continue;
        }
        let tt = starts.get(&digits).map(|s| s.as_str()).unwrap_or(t);
        let _ = write!(
            out,
            "<a class=\"vid\" href=\"{page}#pin={digits}&amp;t={tt}&amp;zoom=2\">#{digits}</a>"
        );
        rest = &rest[k + 1 + digits.len()..];
    }
    out.push_str(rest);
    out
}

/// The case prefix ("case07") of a file name.
fn case_prefix(p: &Path) -> String {
    p.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .split('-')
        .next()
        .unwrap_or("")
        .trim_end_matches(".html")
        .to_string()
}

/// The screenshots of a case next to its page (`caseNN-shot-K.png`).
fn shots_of(dir: &Path, case: &str) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .map(|it| {
            it.filter_map(|e| e.ok())
                .filter_map(|e| e.file_name().to_str().map(|s| s.to_string()))
                .filter(|n| n.starts_with(&format!("{case}-shot-")) && n.ends_with(".png"))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

fn viewer(trace_path: &Path, out: &Path, args: &[String]) -> Res<()> {
    let data = std::fs::read_to_string(trace_path)?;
    let title = opt(args, "--title")
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            trace_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("trace")
                .to_string()
        });
    let dir = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let case = case_prefix(out);
    // the user's screenshots, as files next to the page
    let shots = shots_of(dir, &case);
    let shots_html = if shots.is_empty() {
        String::new()
    } else {
        let imgs: String = shots.iter().map(|s| format!("<a href=\"{s}\" target=\"_blank\" title=\"open full size\"><img src=\"{s}\" alt=\"{s}\" loading=\"lazy\"></a>")).collect();
        format!("<details class=\"shotbox\" open><summary>What the player saw ({} screenshot{})</summary><div class=\"shots\">{imgs}</div></details>", shots.len(), if shots.len() == 1 { "" } else { "s" })
    };
    // the diagnosis
    let diag_path = opt(args, "--diagnose")
        .map(PathBuf::from)
        .unwrap_or_else(|| dir.join(format!("{case}-diagnose.txt")));
    let diag = std::fs::read_to_string(&diag_path)
        .ok()
        .map(|t| parse_diag(&t));
    // (the JSON goes into a script element: no "</" may appear in it)
    let data = data.replace("</", "<\\/");
    let page = |g: &str| {
        VIEWER_HTML
            .replace("__TITLE__", &esc_html(&title))
            .replace("__SHOTS__", &shots_html)
            .replace("__DIAG__", g)
            .replace("__DATA__", &data)
    };
    let to_json = |d: &Diag, trimmed: usize| -> String {
        let cars: Vec<Value> = d.cars.iter().map(|c| json!({"id": c.id, "start": c.start, "stood": c.stood, "why": c.why, "head": c.head, "at": c.at, "cycle": c.cycle, "hops": c.hops})).collect();
        serde_json::to_string(
            &json!({"head": d.head, "roots": d.roots, "cars": cars, "trimmed": trimmed}),
        )
        .unwrap_or("{}".into())
        .replace("</", "<\\/")
    };
    let html = match diag {
        Some(mut d) => {
            // fit the page: the chains of the vehicles that stood least go first
            let mut order: Vec<usize> = (0..d.cars.len()).collect();
            order.sort_by_key(|&k| d.cars[k].stood);
            let mut trimmed = 0;
            let mut html = page(&to_json(&d, trimmed));
            let mut it = order.into_iter();
            while html.len() > PAGE_MAX {
                let Some(k) = it.next() else { break };
                if !d.cars[k].hops.is_empty() {
                    d.cars[k].hops.clear();
                    trimmed += 1;
                    html = page(&to_json(&d, trimmed));
                }
            }
            if trimmed > 0 {
                println!(
                    "  {trimmed} chains left out to keep the page under {} KB",
                    PAGE_MAX / 1024
                );
            }
            html
        }
        None => page("null"),
    };
    std::fs::write(out, &html)?;
    println!("{}: {:.0} KB", out.display(), html.len() as f64 / 1024.0);
    Ok(())
}

fn index(dir: &Path) -> Res<()> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().map(|x| x == "html").unwrap_or(false)
                && p.file_name()
                    .and_then(|s| s.to_str())
                    .map(|s| s.starts_with("case"))
                    .unwrap_or(false)
        })
        .collect();
    entries.sort();
    let mut rows = String::new();
    for p in &entries {
        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let case = case_prefix(p);
        let diag_name = format!("{case}-diagnose.txt");
        let diag = std::fs::read_to_string(dir.join(&diag_name))
            .ok()
            .map(|t| parse_diag(&t));
        let title = std::fs::read_to_string(p)
            .ok()
            .and_then(|t| {
                t.split("<title>")
                    .nth(1)
                    .and_then(|x| x.split("</title>").next())
                    .map(|s| s.to_string())
            })
            .unwrap_or(case.clone());
        let causes = match &diag {
            Some(d) => {
                let starts: HashMap<String, String> = d
                    .cars
                    .iter()
                    .map(|c| (c.id.clone(), c.start.clone()))
                    .collect();
                d.roots
                    .iter()
                    .take(4)
                    .map(|r| {
                        // (a line's ids open at the stand start of the first diagnosed one in it)
                        let first = r
                            .split('#')
                            .skip(1)
                            .map(|s| {
                                s.chars()
                                    .take_while(|c| c.is_ascii_digit())
                                    .collect::<String>()
                            })
                            .find(|id| starts.contains_key(id));
                        let t = first
                            .and_then(|id| starts.get(&id).cloned())
                            .unwrap_or_default();
                        let body = r.trim_start_matches("ROOT CAUSE ");
                        let (class, rest) = body.split_once(' ').unwrap_or((body, ""));
                        format!(
                            "<b>{}</b> {}",
                            esc_html(class),
                            linkify(&esc_html(rest), name, &starts, &t)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("<br>")
            }
            None => String::new(),
        };
        let thumbs: String = shots_of(dir, &case).iter().map(|s| format!("<a href=\"{s}\" target=\"_blank\"><img src=\"{s}\" width=\"150\" loading=\"lazy\"></a>")).collect();
        let _ = writeln!(
            rows,
            "<tr><td><a href=\"{name}\">{}</a><div class=\"th\">{thumbs}</div></td><td>{:.0} KB</td><td class=\"rc\">{causes}</td><td><a href=\"{diag_name}\">diagnose</a></td></tr>",
            esc_html(&title),
            std::fs::metadata(p).map(|m| m.len()).unwrap_or(0) as f64 / 1024.0
        );
    }
    let html = format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Traffic cases</title><style>body{{font:13px/1.4 system-ui,sans-serif;color:var(--foreground,#222);margin:16px}}table{{border-collapse:collapse}}td,th{{border:1px solid var(--border,#ccc);padding:4px 8px;vertical-align:top;text-align:left}}a{{color:var(--accent,#2563eb)}}a.vid{{font-weight:600;text-decoration:none;border-bottom:1px dotted}}.rc{{font-size:12px;color:var(--muted-foreground,#555);max-width:700px;word-break:break-word}}.th img{{margin:4px 4px 0 0;border:1px solid var(--border,#ccc);border-radius:3px}}</style></head><body><h2>Traffic cases</h2><p>Vehicle ids open the case's viewer with that vehicle pinned at the start of its stand.</p><table><tr><th>case</th><th>size</th><th>root causes (diagnose)</th><th></th></tr>\n{rows}</table></body></html>"
    );
    let out = dir.join("index.html");
    std::fs::write(&out, html)?;
    println!("{}: {} cases", out.display(), entries.len());
    Ok(())
}
