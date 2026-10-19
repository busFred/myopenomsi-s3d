//! The visual cross-section of a spline type, from its `.sli`: the drawn strips
//! (`[profile]` / `[profilepnt]`: x across, z up, texture) sorted into road, pavement
//! and the rest, the kerb faces (two consecutive points at the same x and different
//! heights) and every edge between strips; and from them the drivable extent of a
//! traffic path ([`lane_drivable`]): how far either side of its axis the drawn road goes
//! on before a kerb, a pavement or the asphalt's end.
//!
//! (Shared by the game's lane builder and the `omsi-roadmask` tool; laying an x offset
//! along a placed spline is the tool's own, it needs the map's curve.)

use crate::Spline;

/// A vertical face in a profile counts as a kerb from this height difference (m).
pub const KERB_MIN_STEP: f32 = 0.02;
/// Two profile points closer than this across (m) are one x.
pub const SAME_X: f32 = 0.05;
/// A strip of unknown texture this much above the lowest road strip is "raised".
pub const RAISED_ABOVE_ROAD: f32 = 0.05;
/// A path lies on a strip when the drawn surface is within this (m) of the path's height
/// (a path floats a centimetre or three over its asphalt; a deck 1 m up is another road).
pub const PATH_ON_STRIP_DZ: f32 = 0.5;

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

impl Strip {
    /// A surface a car may stand on: asphalt by its texture's name, or one the name says
    /// nothing about that is not raised over the road.
    pub fn drivable(&self) -> bool {
        self.class == "road" || self.class == "other"
    }

    /// The drawn height at lateral `x` (interpolated between the ends).
    pub fn z_at(&self, x: f32) -> f32 {
        let w = self.x1 - self.x0;
        if w.abs() < 1e-6 {
            return self.z0;
        }
        let t = ((x - self.x0) / w).clamp(0.0, 1.0);
        self.z0 + (self.z1 - self.z0) * t
    }
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

impl CrossSection {
    /// The drivable strip under lateral `x` at height `z`, if the drawn surface there is one
    /// a car may stand on (the nearest in height when strips overlap, a deck over a road).
    fn strip_under(&self, x: f32, z: f32) -> Option<usize> {
        let mut best: Option<(f32, usize)> = None;
        for (i, s) in self.strips.iter().enumerate() {
            if !s.drivable() || x < s.x0 - SAME_X || x > s.x1 + SAME_X {
                continue;
            }
            let dz = (s.z_at(x) - z).abs();
            if dz <= PATH_ON_STRIP_DZ && best.is_none_or(|b| dz < b.0) {
                best = Some((dz, i));
            }
        }
        best.map(|b| b.1)
    }

    fn kerb_at(&self, x: f32) -> bool {
        self.edges.iter().any(|e| e.kind == "kerb" && (e.x - x).abs() < SAME_X)
    }

    /// How far the drivable surface under (`x`, `z`) goes on towards -x and towards +x
    /// (metres from `x`): across the strips a car may stand on that join without a step,
    /// up to a kerb face, a pavement or ground strip, a step of `KERB_MIN_STEP` or the
    /// drawn extent's end. None when nothing drivable is drawn under the point.
    pub fn drivable_extent(&self, x: f32, z: f32) -> Option<(f32, f32)> {
        let start = self.strip_under(x, z)?;
        let walk = |to_pos: bool| -> f32 {
            let mut cur = &self.strips[start];
            loop {
                let (edge, z_edge) = if to_pos { (cur.x1, cur.z1) } else { (cur.x0, cur.z0) };
                if self.kerb_at(edge) {
                    return edge;
                }
                let next = self.strips.iter().find(|s| {
                    let (join, z_join) = if to_pos { (s.x0, s.z0) } else { (s.x1, s.z1) };
                    s.drivable() && (join - edge).abs() < SAME_X && (z_join - z_edge).abs() < KERB_MIN_STEP && s.x1 - s.x0 > SAME_X
                });
                match next {
                    Some(n) if !std::ptr::eq(n, cur) => cur = n,
                    _ => return edge,
                }
            }
        };
        let neg = walk(false);
        let pos = walk(true);
        Some(((x - neg).max(0.0), (pos - x).max(0.0)))
    }
}

/// The drivable extent of street path `path_index` of spline type `def` to the left and
/// right of its axis in its direction of travel (metres): the drawn road's reach either
/// side ([`CrossSection::drivable_extent`]), cut at the near edge of the next street path
/// of the same spline on that side (the neighbour's or the oncoming lane's painted edge),
/// never less than the path's own half width. `mirror`: the spline is placed mirrored
/// (its cross-section turned over); `reversed`: the lane runs against the spline's
/// direction. A path over no drawn road (an invisible helper spline, a Human_path) gets
/// its half width either side.
pub fn lane_drivable(def: &Spline, path_index: usize, mirror: bool, reversed: bool) -> (f32, f32) {
    let Some(p) = def.paths.get(path_index) else { return (0.0, 0.0) };
    let half = p.width * 0.5;
    let (x, z) = (p.start[0], p.start[2]);
    let cs = cross_section(def);
    let (mut neg, mut pos) = match cs.drivable_extent(x, z) {
        Some(e) => e,
        None => (half, half),
    };
    for (i, o) in def.paths.iter().enumerate() {
        if i == path_index || o.kind != 0 || (o.start[2] - z).abs() > PATH_ON_STRIP_DZ {
            continue;
        }
        let ox = o.start[0];
        if ox > x {
            pos = pos.min(ox - o.width * 0.5 - x);
        } else if ox < x {
            neg = neg.min(x - (ox + o.width * 0.5));
        }
    }
    let (neg, pos) = (neg.max(half), pos.max(half));
    // in the placed spline's frame, +x is the right of its direction; mirrored, the
    // cross-section is turned over; a reversed lane's right is the spline's left
    let (left, right) = if mirror { (pos, neg) } else { (neg, pos) };
    if reversed {
        (right, left)
    } else {
        (left, right)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sli::SplineTexture;
    use crate::{PathDef, SplineProfile, SplineProfilePoint};

    fn pt(x: f32, z: f32) -> SplineProfilePoint {
        SplineProfilePoint {
            x,
            z,
            u: 0.0,
            v_scale: 0.1,
        }
    }

    fn tex(file: &str) -> SplineTexture {
        SplineTexture {
            file: file.into(),
            ..Default::default()
        }
    }

    fn path(x: f32, z: f32, width: f32, direction: i32) -> PathDef {
        PathDef {
            kind: 0,
            start: [x, 0.0, z],
            end: [0.0; 3],
            width,
            direction,
            params: Vec::new(),
        }
    }

    /// Pavement 2.5 m wide at +0.12, a 0.12 m kerb face, a 3.5 m road lane, a seam, another
    /// lane (the hand-made profile of the roadmask tool's first test).
    fn kerbed() -> Spline {
        let mut def = Spline::default();
        def.textures = vec![tex("str_gehweg06.bmp"), tex("str_asphdrk_1line.bmp")];
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
        def
    }

    #[test]
    fn kerb_and_edges_of_a_hand_made_profile() {
        let cs = cross_section(&kerbed());
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

    #[test]
    fn the_drivable_extent_runs_across_seams_to_the_kerb_and_the_asphalts_end() {
        let cs = cross_section(&kerbed());
        // a lane axis at -1.75 on the first road strip: 1.75 m to the kerb at -3.5, across
        // the seam at 0 to the drawn end at 3.5
        let (neg, pos) = cs.drivable_extent(-1.75, 0.02).unwrap();
        assert!((neg - 1.75).abs() < 1e-5 && (pos - 5.25).abs() < 1e-5, "{neg} {pos}");
        // on the pavement: nothing drivable
        assert_eq!(cs.drivable_extent(-5.0, 0.12), None);
        // a path 1 m up (a deck) lies on no strip
        assert_eq!(cs.drivable_extent(-1.75, 1.2), None);
    }

    #[test]
    fn a_lanes_drivable_extent_stops_at_the_next_street_path_and_follows_mirror_and_direction() {
        let mut def = kerbed();
        // the oncoming lane at +1.75 (3.5 m wide to the drawn end), mine at -1.75
        def.paths = vec![path(-1.75, 0.0, 3.0, 0), path(1.75, 0.0, 3.0, 1)];
        // plain: left to the kerb (1.75), right to the oncoming lane's painted edge at
        // 1.75 - 1.5 = 0.25 -> 2.0 m
        assert_eq!(lane_drivable(&def, 0, false, false), (1.75, 2.0));
        // the oncoming path runs backwards: its right is the spline's left, so the kerb
        // side is its left... from +1.75 the drawn end at 3.5 lies 1.75 m to +x and my
        // edge at -0.25 lies 2.0 m to -x; reversed: (left, right) = (to +x, to -x)
        assert_eq!(lane_drivable(&def, 1, false, true), (1.75, 2.0));
        // mirrored (and the forward path then runs backwards, as the lane builder does):
        // the cross-section turned over, the kerb on the other side, the same numbers
        assert_eq!(lane_drivable(&def, 0, true, true), (1.75, 2.0));
        assert_eq!(lane_drivable(&def, 0, true, false), (2.0, 1.75));
        // never less than the path's own half width, even on a strip narrower than it
        def.paths[0].width = 5.0;
        assert_eq!(lane_drivable(&def, 0, false, false), (2.5, 2.5));
    }

    #[test]
    fn a_path_over_no_road_gets_its_half_width() {
        let mut def = Spline::default();
        def.paths = vec![path(0.0, 0.0, 2.0, 0)];
        assert_eq!(lane_drivable(&def, 0, false, false), (1.0, 1.0));
        // JiananDong_1.sli: one road strip -5.4 .. 6.4 at 0.1, no kerb; two 1.9 m street
        // paths at -1.8 (backwards) and +1.8, a pavement path at 3.845 that is not a street
        // path: the right-hand lane's extent runs to the strip's far edges, cut at the
        // oncoming lane's painted edge on the left only
        def.textures = vec![tex("str_asphdrk.bmp")];
        def.profiles = vec![SplineProfile {
            texture: 0,
            points: vec![pt(-5.4, 0.1), pt(6.4, 0.1)],
        }];
        def.paths = vec![path(-1.8, 0.1, 1.9, 1), path(1.8, 0.1, 1.9, 0), PathDef { kind: 1, ..path(3.845, 0.25, 1.567, 2) }];
        let (l, r) = lane_drivable(&def, 1, false, false);
        assert!((l - 2.65).abs() < 1e-5 && (r - 4.6).abs() < 1e-5, "{l} {r}");
        let (l, r) = lane_drivable(&def, 0, false, true);
        assert!((l - 2.65).abs() < 1e-5 && (r - 3.6).abs() < 1e-5, "{l} {r}");
    }
}
