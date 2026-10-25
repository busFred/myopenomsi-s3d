//! Picking and preparing a recording from a terminal: the `.rec` files under
//! `~/.openomsi/traffic` (a native file dialog, or a numbered list), what a recording's
//! header says without reading it whole, whether it runs longer than a window, and the
//! 90 s cut of a long one (the streaming cut `traffic_cases cut` makes, for one time).

use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use omsi_traffic::record::Header;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// A window cut from a long recording runs this long (s), centred on the time asked for
/// (as `traffic_cases cut`: 45 s either side).
pub const WINDOW_HALF: f64 = 45.0;
/// A recording whose frames span more than this (s) is cut to a window first.
pub const LONG_SECS: f64 = 180.0;

/// The lines of a zlib-compressed recording, one at a time (a session recording does not
/// fit in memory parsed).
pub struct Lines {
    r: BufReader<ZlibDecoder<BufReader<File>>>,
    pub buf: Vec<u8>,
    /// None while reading; Some(true) when the stream ended cleanly.
    pub clean: Option<bool>,
    pub error: Option<String>,
    pub bytes: u64,
}

impl Lines {
    pub fn open(p: &Path) -> std::io::Result<Lines> {
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
    pub fn next(&mut self) -> bool {
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
pub fn peek(line: &[u8]) -> (&str, f64, Option<f64>) {
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

/// What a recording's header says, read from its first line only.
pub struct HeaderInfo {
    pub map: String,
    pub started: f64,
    pub trigger: Option<String>,
    pub lanes: usize,
}

pub fn header_info(path: &Path) -> Res<HeaderInfo> {
    let mut lines = Lines::open(path)?;
    if !lines.next() {
        return Err("an empty recording".into());
    }
    let h: Header = omsi_traffic::finite::from_str(std::str::from_utf8(&lines.buf)?)?;
    Ok(HeaderInfo {
        map: h.map,
        started: h.started_day_time,
        trigger: h.trigger,
        lanes: h.lanes.len(),
    })
}

/// The day-time span of a recording's frames: (first, last, longer than `limit`). The scan
/// stops as soon as the span exceeds `limit`, so a session recording of gigabytes is not
/// read whole; a short one is read to its end.
pub fn frame_span(path: &Path, limit: f64) -> Res<(f64, f64, bool)> {
    let mut lines = Lines::open(path)?;
    let (mut first, mut last) = (f64::NAN, f64::NAN);
    lines.next(); // the header
    while lines.next() {
        let (kind, _, day) = peek(&lines.buf);
        if kind != "Frame" {
            continue;
        }
        if let Some(d) = day {
            if first.is_nan() {
                first = d;
            }
            last = d;
            if last - first > limit {
                return Ok((first, last, true));
            }
        }
    }
    Ok((first, last, false))
}

/// Cut the window `at ± WINDOW_HALF` (day time, s) out of `path` into `out`, as
/// `traffic_cases cut` does: the header, the lanes, lights and parked cars that came
/// before, the last keyframe before the window's start and every record from it to the
/// first frame past its end. Returns (first, last) day time written.
pub fn cut_window(path: &Path, at: f64, out: &Path) -> Res<(f64, f64)> {
    let t0 = at - WINDOW_HALF;
    let t1 = at + WINDOW_HALF;
    let mut lines = Lines::open(path)?;
    if !lines.next() {
        return Err("an empty recording".into());
    }
    let header = lines.buf.clone();
    let mut added_before: Vec<Vec<u8>> = Vec::new();
    let mut parked_before: Option<Vec<u8>> = None;
    // the records since the last keyframe (that keyframe first)
    let mut block: Vec<Vec<u8>> = Vec::new();
    let mut open: Option<ZlibEncoder<BufWriter<File>>> = None;
    let mut written = 0u64;
    let (mut first, mut last) = (f64::NAN, f64::NAN);
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
            // the window starts before this keyframe: open at the last one (a keyframe
            // already past the window's end is too late: the window lies before the
            // recording, or in a gap)
            if open.is_none() && est > t0 && !(est > t1) {
                let mut w = ZlibEncoder::new(
                    BufWriter::with_capacity(1 << 22, File::create(out)?),
                    Compression::default(),
                );
                for l in std::iter::once(&header)
                    .chain(added_before.iter())
                    .chain(parked_before.iter())
                    .chain(block.iter())
                {
                    w.write_all(l)?;
                    w.write_all(b"\n")?;
                    written += 1;
                }
                open = Some(w);
            }
            for l in block.drain(..) {
                match peek(&l).0 {
                    "LanesAdded" | "LightsAdded" | "LaneFlags" => added_before.push(l),
                    "Parked" => parked_before = Some(l),
                    _ => {}
                }
            }
        }
        if let Some(w) = open.as_mut() {
            if kind == "Frame" {
                if let Some(d) = day {
                    if d > t1 {
                        break;
                    }
                    if first.is_nan() {
                        first = d;
                    }
                    last = d;
                }
            }
            w.write_all(&line)?;
            w.write_all(b"\n")?;
            written += 1;
        }
        if kind == "Keyframe" || !block.is_empty() {
            block.push(line);
        } else {
            match kind.as_str() {
                "LanesAdded" | "LightsAdded" | "LaneFlags" => added_before.push(line),
                "Parked" => parked_before = Some(line),
                _ => {}
            }
        }
    }
    match open {
        Some(w) => {
            w.finish()?.flush()?;
            log::info!("cut {}: {written} lines", out.display());
            if first.is_nan() {
                let _ = std::fs::remove_file(out);
                return Err(format!("no frames between {} and {} (game time, not the clock in the file name)", crate::hms(t0), crate::hms(t1)).into());
            }
            Ok((first, last))
        }
        None => Err(format!("{} is not in the recording (game time, not the clock in the file name; no keyframe before it)", crate::hms(at)).into()),
    }
}

/// The recordings under `dir` and its folders one level down (not the pages' folder),
/// newest first.
pub fn list_recordings(dir: &Path) -> Vec<(PathBuf, std::time::SystemTime, u64)> {
    let mut out = Vec::new();
    let mut visit = |d: &Path| {
        if let Ok(rd) = std::fs::read_dir(d) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("rec")) {
                    if let Ok(m) = e.metadata() {
                        out.push((p, m.modified().unwrap_or(std::time::UNIX_EPOCH), m.len()));
                    }
                }
            }
        }
    };
    visit(dir);
    if let Ok(rd) = std::fs::read_dir(dir) {
        let mut subs: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        subs.sort();
        for s in subs {
            visit(&s);
        }
    }
    out.sort_by(|a, b| b.1.cmp(&a.1));
    out
}

/// A native file dialog for a recording, starting in `dir`.
pub fn pick_recording(dir: &Path) -> Option<PathBuf> {
    let mut d = rfd::FileDialog::new()
        .set_title("Choose a traffic recording")
        .add_filter("Traffic recording", &["rec"]);
    if dir.is_dir() {
        d = d.set_directory(dir);
    }
    d.pick_file()
}

/// Open `page` in the default browser (detached; a failure is reported, not fatal).
pub fn open_in_browser(page: &Path) -> std::io::Result<()> {
    let mut cmd = if cfg!(target_os = "macos") {
        let mut c = std::process::Command::new("open");
        c.arg(page);
        c
    } else if cfg!(windows) {
        let mut c = std::process::Command::new("cmd");
        c.args(["/C", "start", ""]).arg(page);
        c
    } else {
        let mut c = std::process::Command::new("xdg-open");
        c.arg(page);
        c
    };
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map(|_| ())
}

/// `HH:MM` or `HH:MM:SS` as seconds of the day.
pub fn parse_hhmm(s: &str) -> Res<f64> {
    let mut it = s.trim().split(':');
    let h: f64 = it.next().ok_or("not HH:MM")?.parse()?;
    let m: f64 = it.next().ok_or("not HH:MM")?.parse()?;
    let sec: f64 = it.next().map(|x| x.parse()).transpose()?.unwrap_or(0.0);
    Ok(h * 3600.0 + m * 60.0 + sec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peeks_kind_time_and_day() {
        let (k, t, d) = peek(br#"{"Frame":{"time":12.5,"dt":0.05,"day_time":30000.25,"x":1}}"#);
        assert_eq!((k, t, d), ("Frame", 12.5, Some(30000.25)));
        let (k, t, d) = peek(br#"{"Keyframe":{"time":10.0,"vehicles":[]}}"#);
        assert_eq!((k, t, d), ("Keyframe", 10.0, None));
    }

    #[test]
    fn parses_times() {
        assert_eq!(parse_hhmm("08:42").unwrap(), 8.0 * 3600.0 + 42.0 * 60.0);
        assert_eq!(
            parse_hhmm(" 8:05:30 ").unwrap(),
            8.0 * 3600.0 + 5.0 * 60.0 + 30.0
        );
        assert!(parse_hhmm("842").is_err());
    }
}
