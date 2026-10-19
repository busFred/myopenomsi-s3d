//! The visual cross-section of a spline type and how an x offset of it is laid along a
//! placed spline's centreline ([`lay_along`]). The cross-section itself (the drawn strips,
//! the kerb faces, the edges, a path's drivable extent) is `omsi_scenery::profile`, shared
//! with the game's lane builder; it is re-exported here.

use glam::DVec3;
use omsi_geometry::SplineCurve;
pub use omsi_scenery::profile::*;

/// How many stations a placed spline's cross-section lines are laid with: one every 2 m,
/// and in a curve one every 1.5 degrees of turn at least.
pub fn station_count(curve: &SplineCurve) -> usize {
    let len = curve.length.max(0.0);
    let mut n = (len / 2.0).ceil() as usize;
    if curve.radius != 0.0 && curve.radius.is_finite() {
        let turn_deg = (len / curve.radius.abs()).to_degrees();
        n = n.max((turn_deg / 1.5).ceil() as usize);
    }
    n.clamp(2, 800)
}

/// The line at lateral offset `x` (right positive in the spline's own frame) and height `z`
/// of a spline placed as `curve`, `side` -1 for a mirrored placement (its cross-section
/// turned over), at `n` stations from the start to the end.
pub fn lay_along(curve: &SplineCurve, x: f64, z: f64, side: f64, n: usize) -> Vec<DVec3> {
    let n = n.max(1);
    (0..=n)
        .map(|i| curve.offset_point(curve.length * i as f64 / n as f64, side * x, z))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec2;
    use omsi_scenery::Spline;
    use std::path::{Path, PathBuf};

    /// The OMSI root as the tool finds it (`--root` has no say in a test): `OMSI_ROOT`, else
    /// the launcher's `root`. None: the test is skipped.
    fn root() -> Option<PathBuf> {
        crate::mapdata::omsi_root(None).ok()
    }

    /// The first `.sli` under `dir` (two levels deep) whose profile has a kerb face.
    fn find_kerbed_sli(dir: &Path, depth: usize, budget: &mut usize) -> Option<PathBuf> {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
            .ok()?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .collect();
        entries.sort();
        for p in &entries {
            if *budget == 0 {
                return None;
            }
            if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("sli")) {
                *budget -= 1;
                if let Ok(def) = Spline::load(p) {
                    let cs = cross_section(&def);
                    if cs.edges.iter().any(|e| e.kind == "kerb") && !cs.strips.is_empty() {
                        return Some(p.clone());
                    }
                }
            }
        }
        if depth > 0 {
            for p in &entries {
                if p.is_dir() {
                    if let Some(f) = find_kerbed_sli(p, depth - 1, budget) {
                        return Some(f);
                    }
                }
            }
        }
        None
    }

    #[test]
    fn parses_a_real_sli_from_the_root() {
        let Some(root) = root() else {
            eprintln!("skipped: no OMSI root (OMSI_ROOT or ~/.openomsi/launcher.json)");
            return;
        };
        // a known StreetCreator file when the root has it: pavements ±11.6 .. ±7.6 at 0.25,
        // kerbs at ±7.6 down to the road at 0.10
        let known = omsi_cfg::resolve_path(
            &root,
            "Splines\\ADDON_SimpleStreets\\LHDRQ_20_4_spur_15m_narrow_Sidewalk.sli",
        );
        if known.is_file() {
            let def = Spline::load(&known).expect("the .sli reads");
            assert_eq!(def.textures.len(), 3, "{}", known.display());
            let cs = cross_section(&def);
            assert!(
                (cs.x_min + 11.6).abs() < 1e-3 && (cs.x_max - 11.6).abs() < 1e-3,
                "extent {} .. {}",
                cs.x_min,
                cs.x_max
            );
            let kerbs: Vec<&EdgeLine> = cs.edges.iter().filter(|e| e.kind == "kerb").collect();
            assert_eq!(kerbs.len(), 2, "{:?}", cs.edges);
            assert!((kerbs[0].x + 7.6).abs() < 1e-3 && (kerbs[1].x - 7.6).abs() < 1e-3);
            assert!((kerbs[0].z0 - 0.10).abs() < 1e-3 && (kerbs[0].z1 - 0.25).abs() < 1e-3);
            assert!(cs.strips.iter().any(|s| s.class == "walk" && s.x1 <= -7.5));
            assert!(cs
                .strips
                .iter()
                .any(|s| s.class == "road" && s.x0 >= -7.7 && s.x1 <= 7.7));
            // every street path's drivable extent ends at those kerbs (or at the next lane):
            // from the outermost lanes the kerb side reaches 7.6 - |x|
            for (i, p) in def.paths.iter().enumerate().filter(|(_, p)| p.kind == 0) {
                let (l, r) = lane_drivable(&def, i, false, false);
                let x = p.start[0];
                assert!(l <= 7.6 + x + 1e-3 && r <= 7.6 - x + 1e-3, "path {i} at {x}: {l} {r}");
                assert!(l >= p.width * 0.5 && r >= p.width * 0.5, "path {i} at {x}: {l} {r}");
            }
        }
        // and any kerbed spline type of the installation
        let mut budget = 400;
        let Some(file) = find_kerbed_sli(&omsi_cfg::resolve_path(&root, "Splines"), 2, &mut budget)
        else {
            eprintln!("skipped: no .sli with a kerb face found under Splines/");
            return;
        };
        let def = Spline::load(&file).expect("the .sli reads");
        let cs = cross_section(&def);
        assert!(cs.x_max > cs.x_min, "{}", file.display());
        assert!(!cs.strips.is_empty());
        // (the drawn extent's edges: "outer", or "kerb" when the profile ends in a vertical face)
        for x in [cs.x_min, cs.x_max] {
            assert!(
                cs.edges
                    .iter()
                    .any(|e| (e.x - x).abs() < SAME_X && (e.kind == "outer" || e.kind == "kerb")),
                "{}: no edge at {x}: {:?}",
                file.display(),
                cs.edges
            );
        }
        for s in &cs.strips {
            assert!(s.x0 < s.x1 && s.x0 >= cs.x_min - 1e-4 && s.x1 <= cs.x_max + 1e-4);
        }
    }

    /// BeijingCity's JiananDong_1.sli (round 5 case01, the Wuluju stop): one road strip
    /// -5.4 .. 6.4 with no kerb, two 1.9 m street paths at ±1.8 and a pavement path at
    /// 3.845 laid on the asphalt. The eastbound lane at +1.8 may use the road out to the
    /// far strip edge on its right (4.6 m) and to the oncoming lane's painted edge on its
    /// left (2.65 m); the pavement path is no edge.
    #[test]
    fn jianandong_lane_reaches_the_far_road_strip_edge() {
        let Some(root) = root() else {
            eprintln!("skipped: no OMSI root");
            return;
        };
        let file = omsi_cfg::resolve_path(&root, "Splines\\Beijing_splines\\JiananDong_1.sli");
        if !file.is_file() {
            eprintln!("skipped: {} not installed", file.display());
            return;
        }
        let def = Spline::load(&file).expect("the .sli reads");
        let cs = cross_section(&def);
        assert!(!cs.edges.iter().any(|e| e.kind == "kerb"), "{:?}", cs.edges);
        let fwd = def.paths.iter().position(|p| p.kind == 0 && p.start[0] > 0.0).expect("the +1.8 path");
        let (l, r) = lane_drivable(&def, fwd, false, false);
        assert!((l - 2.3).abs() < 1e-3 && (r - 4.6).abs() < 1e-3, "left {l} right {r}");
        let back = def.paths.iter().position(|p| p.kind == 0 && p.start[0] < 0.0).expect("the -1.8 path");
        let (l, r) = lane_drivable(&def, back, false, true);
        assert!((l - 2.3).abs() < 1e-3 && (r - 3.6).abs() < 1e-3, "left {l} right {r}");
    }

    fn curve(radius: f64) -> SplineCurve {
        SplineCurve::from_map(
            &omsi_map::MapSpline {
                pos: [100.0, 200.0, 5.0],
                heading: 30.0,
                length: 40.0,
                radius,
                ..Default::default()
            },
            DVec2::new(1000.0, 2000.0),
        )
    }

    #[test]
    fn offsets_along_a_curved_centreline_keep_their_distance_from_the_arc_centre() {
        for (radius, side) in [(50.0, 1.0), (-50.0, 1.0), (50.0, -1.0)] {
            let c = curve(radius);
            let d = SplineCurve::dir(30.0);
            let right = DVec2::new(d.y, -d.x);
            let centre = c.start.truncate() + right * radius.abs() * radius.signum();
            let n = station_count(&c);
            assert!(n >= 30, "stations {n}");
            let x = 3.0;
            let line = lay_along(&c, x, 0.0, side, n);
            assert_eq!(line.len(), n + 1);
            // a right turn (radius > 0) has its right offset inside: radius - x; mirrored,
            // the same offset lies on the left
            let want = radius.abs() - radius.signum() * side * x;
            for p in &line {
                let r = (p.truncate() - centre).length();
                assert!(
                    (r - want).abs() < 1e-6,
                    "radius {radius} side {side}: {r} vs {want}"
                );
            }
            // it runs the whole spline: its ends sit beside the centreline's ends
            let (a, b) = (c.point_at(0.0), c.end_point());
            assert!((line[0].truncate() - a.truncate()).length() - x < 1e-6);
            assert!((line[n].truncate() - b.truncate()).length() - x < 1e-6);
            // the first offset point is start + side * x to the right of the start heading
            let first = c.start.truncate() + right * side * x;
            assert!((line[0].truncate() - first).length() < 1e-9);
        }
        // straight: every point exactly x off the line
        let c = curve(0.0);
        let d = SplineCurve::dir(30.0);
        let line = lay_along(&c, 2.5, 0.0, 1.0, station_count(&c));
        for p in &line {
            let rel = p.truncate() - c.start.truncate();
            let across = rel.x * d.y - rel.y * d.x;
            assert!((across - 2.5).abs() < 1e-9);
        }
    }
}
