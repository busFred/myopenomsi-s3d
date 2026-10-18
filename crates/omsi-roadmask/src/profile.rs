//! The visual cross-section of a spline type, from its `.sli`: the drawn strips
//! (`[profile]` / `[profilepnt]`: x across, z up, texture) sorted into road, pavement
//! and the rest, the kerb faces (two consecutive points at the same x and different
//! heights) and every edge between strips; and how such an x offset is laid along a
//! placed spline's centreline ([`lay_along`]).

use glam::DVec3;
use omsi_geometry::SplineCurve;
use omsi_scenery::Spline;

/// A vertical face in a profile counts as a kerb from this height difference (m).
pub const KERB_MIN_STEP: f32 = 0.02;
/// Two profile points closer than this across (m) are one x.
pub const SAME_X: f32 = 0.05;
/// A strip of unknown texture this much above the lowest road strip is "raised".
pub const RAISED_ABOVE_ROAD: f32 = 0.05;

/// One drawn band of the cross-section between two profile points.
#[derive(Debug, Clone, PartialEq)]
pub struct Strip {
    pub x0: f32,
    pub x1: f32,
    /// Heights above the spline at `x0` and `x1`.
    pub z0: f32,
    pub z1: f32,
    pub texture: String,
    /// "road", "walk", "ground", "raised" or "other" (see [`classify_texture`]).
    pub class: &'static str,
}

/// A line along the spline at one x of the cross-section.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeLine {
    pub x: f32,
    pub z0: f32,
    pub z1: f32,
    /// "kerb": a vertical face; "outer": the drawn extent's edge; "edge": a change of
    /// surface class or height; "seam": a texture seam within one surface (a lane line).
    pub kind: &'static str,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct CrossSection {
    pub strips: Vec<Strip>,
    pub edges: Vec<EdgeLine>,
    pub x_min: f32,
    pub x_max: f32,
}

/// What a spline texture's name says it is.
pub fn classify_texture(name: &str) -> &'static str {
    let n = name.to_ascii_lowercase();
    const WALK: &[&str] = &[
        "gehweg",
        "sidewalk",
        "side_walk",
        "walk",
        "fuss",
        "fuß",
        "pave",
        "pflaster",
        "platte",
        "bord",
        "kerb",
        "curb",
        "buerger",
        "bürger",
        "footway",
        "ped",
        "plaster",
        "tiles",
        "fliese",
        "trottoir",
        "chodnik",
    ];
    const GROUND: &[&str] = &[
        "grass", "gras", "rasen", "green", "dirt", "earth", "erde", "sand", "kies", "gravel",
        "schotter", "wiese", "boden", "ground",
    ];
    const ROAD: &[&str] = &[
        "asph",
        "street",
        "str_",
        "road",
        "lane",
        "teer",
        "fahrbahn",
        "strasse",
        "straße",
        "spur",
        "beton",
        "concrete",
        "kopfstein",
        "cobble",
        "tarmac",
        "rq_",
        "mark",
        "line",
        "arrow",
        "pfeil",
    ];
    if WALK.iter().any(|k| n.contains(k)) {
        "walk"
    } else if GROUND.iter().any(|k| n.contains(k)) {
        "ground"
    } else if ROAD.iter().any(|k| n.contains(k)) {
        "road"
    } else {
        "other"
    }
}

/// The cross-section of a spline type.
pub fn cross_section(def: &Spline) -> CrossSection {
    let mut cs = CrossSection::default();
    let tex_name = |i: usize| {
        def.textures
            .get(i)
            .map(|t| t.file.clone())
            .unwrap_or_default()
    };
    let mut any = false;
    for prof in &def.profiles {
        let tex = tex_name(prof.texture);
        for w in prof.points.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            if !(a.x.is_finite() && b.x.is_finite() && a.z.is_finite() && b.z.is_finite()) {
                continue;
            }
            for p in [a, b] {
                if !any {
                    cs.x_min = p.x;
                    cs.x_max = p.x;
                    any = true;
                }
                cs.x_min = cs.x_min.min(p.x);
                cs.x_max = cs.x_max.max(p.x);
            }
            if (b.x - a.x).abs() < SAME_X {
                if (b.z - a.z).abs() >= KERB_MIN_STEP {
                    cs.edges.push(EdgeLine {
                        x: (a.x + b.x) * 0.5,
                        z0: a.z.min(b.z),
                        z1: a.z.max(b.z),
                        kind: "kerb",
                    });
                }
                continue;
            }
            let (lo, hi) = if a.x < b.x { (a, b) } else { (b, a) };
            cs.strips.push(Strip {
                x0: lo.x,
                x1: hi.x,
                z0: lo.z,
                z1: hi.z,
                texture: tex.clone(),
                class: classify_texture(&tex),
            });
        }
    }
    // a strip of unknown texture that stands above the road is a raised surface (a pavement
    // whose texture says nothing)
    let road_z = cs
        .strips
        .iter()
        .filter(|s| s.class == "road")
        .map(|s| s.z0.min(s.z1))
        .fold(f32::INFINITY, f32::min);
    let road_z = if road_z.is_finite() {
        road_z
    } else {
        cs.strips
            .iter()
            .map(|s| s.z0.min(s.z1))
            .fold(f32::INFINITY, f32::min)
    };
    if road_z.is_finite() {
        for s in &mut cs.strips {
            if s.class == "other" && s.z0.min(s.z1) >= road_z + RAISED_ABOVE_ROAD {
                s.class = "raised";
            }
        }
    }
    // the edges between strips: every strip end that is not a kerb already
    let mut xs: Vec<(f32, f32)> = cs
        .strips
        .iter()
        .flat_map(|s| [(s.x0, s.z0), (s.x1, s.z1)])
        .collect();
    xs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut i = 0;
    while i < xs.len() {
        let (x, z) = xs[i];
        let mut j = i + 1;
        while j < xs.len() && (xs[j].0 - x).abs() < SAME_X {
            j += 1;
        }
        i = j;
        if cs
            .edges
            .iter()
            .any(|e| e.kind == "kerb" && (e.x - x).abs() < SAME_X)
        {
            continue;
        }
        let kind = if (x - cs.x_min).abs() < SAME_X || (x - cs.x_max).abs() < SAME_X {
            "outer"
        } else {
            let left: Vec<&Strip> = cs
                .strips
                .iter()
                .filter(|s| (s.x1 - x).abs() < SAME_X)
                .collect();
            let right: Vec<&Strip> = cs
                .strips
                .iter()
                .filter(|s| (s.x0 - x).abs() < SAME_X)
                .collect();
            let changes = left.iter().any(|l| {
                right
                    .iter()
                    .any(|r| l.class != r.class || (l.z1 - r.z0).abs() >= KERB_MIN_STEP)
            });
            if left.is_empty() || right.is_empty() || changes {
                "edge"
            } else {
                "seam"
            }
        };
        cs.edges.push(EdgeLine {
            x,
            z0: z,
            z1: z,
            kind,
        });
    }
    cs.edges.sort_by(|a, b| a.x.total_cmp(&b.x));
    cs
}

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
    use omsi_scenery::{SplineProfile, SplineProfilePoint};
    use std::path::{Path, PathBuf};

    fn pt(x: f32, z: f32) -> SplineProfilePoint {
        SplineProfilePoint {
            x,
            z,
            u: 0.0,
            v_scale: 0.1,
        }
    }

    #[test]
    fn kerb_and_edges_of_a_hand_made_profile() {
        // pavement 2.5 m wide at +0.12, a 0.12 m kerb face, a 3.5 m road lane, a seam, another lane
        let mut def = Spline::default();
        def.textures = vec![
            omsi_scenery::sli::SplineTexture {
                file: "str_gehweg06.bmp".into(),
                ..Default::default()
            },
            omsi_scenery::sli::SplineTexture {
                file: "str_asphdrk_1line.bmp".into(),
                ..Default::default()
            },
        ];
        def.profiles = vec![
            SplineProfile {
                texture: 0,
                points: vec![pt(-6.0, 0.12), pt(-3.5, 0.12)],
            },
            SplineProfile {
                texture: 0,
                points: vec![pt(-3.5, 0.12), pt(-3.5, 0.0)],
            },
            SplineProfile {
                texture: 1,
                points: vec![pt(-3.5, 0.0), pt(0.0, 0.0)],
            },
            SplineProfile {
                texture: 1,
                points: vec![pt(0.0, 0.0), pt(3.5, 0.0)],
            },
        ];
        let cs = cross_section(&def);
        assert_eq!(cs.x_min, -6.0);
        assert_eq!(cs.x_max, 3.5);
        assert_eq!(cs.strips.len(), 3);
        assert_eq!(cs.strips[0].class, "walk");
        assert_eq!(cs.strips[1].class, "road");
        let kinds: Vec<(f32, &str)> = cs.edges.iter().map(|e| (e.x, e.kind)).collect();
        assert_eq!(
            kinds,
            vec![
                (-6.0, "outer"),
                (-3.5, "kerb"),
                (0.0, "seam"),
                (3.5, "outer")
            ]
        );
        let kerb = &cs.edges[1];
        assert!((kerb.z0 - 0.0).abs() < 1e-6 && (kerb.z1 - 0.12).abs() < 1e-6);
    }

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
