//! The pieces of `omsi-roadmask` other standalone tools share (`omsi-roadmask3d`): picking
//! and cutting a recording, the OMSI root and the map data behind a window's lanes, the
//! splines' visual cross-sections. The binary is `main.rs`; no game binary depends on this.

pub mod mapdata;
pub mod profile;
pub mod recording;

/// `HH:MM:SS` of a day time in seconds.
pub fn hms(t: f64) -> String {
    let s = t.max(0.0).round() as i64;
    format!("{:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
}
