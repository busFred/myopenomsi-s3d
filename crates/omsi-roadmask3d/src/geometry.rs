//! The overlay's geometry, as triangles for the game's renderer: a lane's footprint from
//! its centreline and width (mitred at the joints), thin lines as flat ribbons, the
//! transverse bands of a striped or hatched piece, a vehicle's box, a person's capsule,
//! a rectangle's outline; and the colours of the mask (a lane's kind, a vehicle's state, a
//! person's decision). All of it in metres relative to one origin, flat on the ground
//! unless the points carry heights.

use glam::{DVec3, Vec2, Vec3};
use omsi_geometry::MeshData;

/// A person's body radius (m), as the traffic has it.
pub const PERSON_RADIUS: f32 = 0.28;

/// Triangles being gathered into one mesh, one material slot per colour.
#[derive(Default)]
pub struct Builder {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    /// Indices per slot (merged into ranges at the end).
    slots: Vec<Vec<u32>>,
}

impl Builder {
    pub fn new(slots: usize) -> Builder {
        Builder { slots: vec![Vec::new(); slots], ..Default::default() }
    }

    fn vertex(&mut self, p: Vec3, n: Vec3) -> u32 {
        self.positions.push(p);
        self.normals.push(n);
        self.uvs.push(Vec2::new(p.x * 0.1, p.y * 0.1));
        (self.positions.len() - 1) as u32
    }

    /// A triangle (both windings are drawn: the renderer sees generated meshes from both sides).
    pub fn tri(&mut self, slot: usize, a: Vec3, b: Vec3, c: Vec3) {
        let n = (b - a).cross(c - a).normalize_or(Vec3::Z);
        let n = if n.z < 0.0 { -n } else { n };
        let i = self.vertex(a, n);
        let j = self.vertex(b, n);
        let k = self.vertex(c, n);
        self.slots[slot].extend([i, j, k]);
    }

    /// A quad from four corners in order round it.
    pub fn quad(&mut self, slot: usize, a: Vec3, b: Vec3, c: Vec3, d: Vec3) {
        self.tri(slot, a, b, c);
        self.tri(slot, a, c, d);
    }

    /// A ribbon between a left and a right line of the same length (the footprint of a lane).
    pub fn ribbon(&mut self, slot: usize, left: &[Vec3], right: &[Vec3]) {
        for i in 1..left.len().min(right.len()) {
            self.quad(slot, left[i - 1], right[i - 1], right[i], left[i]);
        }
    }

    /// A ring of triangles round `centre` at `radius`, flat at height `z`.
    pub fn disc(&mut self, slot: usize, centre: Vec3, radius: f32, segments: usize) {
        for k in 0..segments {
            let a0 = k as f32 / segments as f32 * std::f32::consts::TAU;
            let a1 = (k + 1) as f32 / segments as f32 * std::f32::consts::TAU;
            self.tri(slot, centre, centre + Vec3::new(a0.cos(), a0.sin(), 0.0) * radius, centre + Vec3::new(a1.cos(), a1.sin(), 0.0) * radius);
        }
    }

    pub fn finish(self) -> MeshData {
        let mut indices = Vec::new();
        let mut ranges = Vec::new();
        for (slot, idx) in self.slots.into_iter().enumerate() {
            if idx.is_empty() {
                continue;
            }
            ranges.push((indices.len() as u32, idx.len() as u32, slot as u32));
            indices.extend(idx);
        }
        MeshData { positions: self.positions, normals: self.normals, uvs: self.uvs, ranges, indices, one_sided: false }
    }
}

/// The unit direction of a flat polyline at each of its points: the bisector of the two
/// segments at a joint, the segment's own at the ends. Repeated points are skipped.
pub fn directions(points: &[Vec3]) -> Vec<Vec3> {
    let n = points.len();
    let seg = |i: usize| -> Vec3 {
        let d = points[i + 1] - points[i];
        Vec3::new(d.x, d.y, 0.0).normalize_or_zero()
    };
    (0..n)
        .map(|i| {
            if n < 2 {
                return Vec3::Y;
            }
            let before = if i > 0 { Some(seg(i - 1)) } else { None };
            let after = if i + 1 < n { Some(seg(i)) } else { None };
            let d = match (before, after) {
                (Some(a), Some(b)) => {
                    let s = a + b;
                    // (a hairpin: the bisector vanishes; the segment ahead stands in)
                    if s.length() < 1e-4 {
                        b
                    } else {
                        s.normalize()
                    }
                }
                (Some(a), None) => a,
                (None, Some(b)) => b,
                (None, None) => Vec3::Y,
            };
            if d.length() < 1e-6 {
                Vec3::Y
            } else {
                d
            }
        })
        .collect()
}

/// The left and right edges of a lane `width` wide along `centre` (right positive in the
/// lane's own direction), mitred at the joints: each point is pushed out along the
/// bisector's normal by half the width over the cosine of half the turn, so that the two
/// segments' edges meet at one point and the band keeps its width in the straight parts.
/// The mitre is held at twice the half width (a sharp joint does not spike).
pub fn footprint_edges(centre: &[Vec3], width: f32) -> (Vec<Vec3>, Vec<Vec3>) {
    let dirs = directions(centre);
    let n = centre.len();
    let mut left = Vec::with_capacity(n);
    let mut right = Vec::with_capacity(n);
    let half = width * 0.5;
    for i in 0..n {
        let d = dirs[i];
        // the normal to the right of the direction (x east, y north: right of north is east)
        let normal = Vec3::new(d.y, -d.x, 0.0);
        let mut scale = 1.0;
        if i > 0 && i + 1 < n {
            let a = centre[i] - centre[i - 1];
            let a = Vec3::new(a.x, a.y, 0.0).normalize_or_zero();
            let cos_half = a.dot(d).clamp(0.05, 1.0);
            scale = (1.0 / cos_half).min(2.0);
        }
        let off = normal * half * scale;
        left.push(centre[i] - off);
        right.push(centre[i] + off);
    }
    (left, right)
}

/// A footprint as one polygon (the left edge forward, the right edge back).
#[cfg(test)]
pub fn footprint_polygon(centre: &[Vec3], width: f32) -> Vec<Vec3> {
    let (l, mut r) = footprint_edges(centre, width);
    r.reverse();
    let mut poly = l;
    poly.extend(r);
    poly
}

/// Double the signed area of a flat polygon (positive counter-clockwise).
#[cfg(test)]
pub fn area2(poly: &[Vec3]) -> f32 {
    let n = poly.len();
    (0..n).map(|i| {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        a.x * b.y - b.x * a.y
    }).sum()
}

/// Length of a polyline (flat).
pub fn length(points: &[Vec3]) -> f32 {
    points.windows(2).map(|w| (w[1] - w[0]).truncate().length()).sum()
}

/// The point and direction `s` metres along a polyline (clamped to its ends).
pub fn along(points: &[Vec3], s: f32) -> (Vec3, Vec3) {
    if points.len() < 2 {
        return (points.first().copied().unwrap_or(Vec3::ZERO), Vec3::Y);
    }
    let mut rest = s.max(0.0);
    for w in points.windows(2) {
        let d = w[1] - w[0];
        let len = d.truncate().length();
        if rest <= len || len == 0.0 {
            let t = if len > 0.0 { rest / len } else { 0.0 };
            return (w[0] + d * t, Vec3::new(d.x, d.y, 0.0).normalize_or(Vec3::Y));
        }
        rest -= len;
    }
    let w = &points[points.len() - 2..];
    let d = w[1] - w[0];
    (w[1], Vec3::new(d.x, d.y, 0.0).normalize_or(Vec3::Y))
}

/// A thin flat ribbon `width` wide along a line (lane edges, kerb lines, outlines).
pub fn line_ribbon(b: &mut Builder, slot: usize, points: &[Vec3], width: f32) {
    if points.len() < 2 {
        return;
    }
    let (l, r) = footprint_edges(points, width);
    b.ribbon(slot, &l, &r);
}

/// Transverse bands across a lane: every `period` metres a band `band` metres long over
/// the lane's whole width (a crossing's stripes, a junction piece's hatching).
pub fn bands(b: &mut Builder, slot: usize, centre: &[Vec3], width: f32, period: f32, band: f32) {
    let total = length(centre);
    let mut s = 0.0;
    while s < total {
        let s1 = (s + band).min(total);
        let (p0, d0) = along(centre, s);
        let (p1, d1) = along(centre, s1);
        let n0 = Vec3::new(d0.y, -d0.x, 0.0) * width * 0.5;
        let n1 = Vec3::new(d1.y, -d1.x, 0.0) * width * 0.5;
        b.quad(slot, p0 - n0, p0 + n0, p1 + n1, p1 - n1);
        s += period;
    }
}

/// The four ground corners of a vehicle's box at `pos` heading `heading_deg` (clockwise
/// from north), `front` metres ahead of the position, `rear` behind, `half_width` either
/// side: front-left, front-right, rear-right, rear-left.
pub fn box_corners(pos: Vec3, heading_deg: f32, front: f32, rear: f32, half_width: f32) -> [Vec3; 4] {
    let (s, c) = heading_deg.to_radians().sin_cos();
    let fwd = Vec3::new(s, c, 0.0);
    let right = Vec3::new(c, -s, 0.0);
    [pos + fwd * front - right * half_width, pos + fwd * front + right * half_width, pos - fwd * rear + right * half_width, pos - fwd * rear - right * half_width]
}

/// A closed block over the ground corners, `height` tall (the recorded vehicle as a mask).
pub fn block(b: &mut Builder, slot: usize, base: [Vec3; 4], height: f32) {
    let top: Vec<Vec3> = base.iter().map(|p| *p + Vec3::Z * height).collect();
    b.quad(slot, top[0], top[1], top[2], top[3]);
    for i in 0..4 {
        let j = (i + 1) % 4;
        b.quad(slot, base[i], base[j], top[j], top[i]);
    }
}

/// The outline of a flat rectangle as four ribbons `width` wide at the corners' height.
pub fn outline(b: &mut Builder, slot: usize, corners: &[Vec3], width: f32) {
    let n = corners.len();
    for i in 0..n {
        let seg = [corners[i], corners[(i + 1) % n]];
        line_ribbon(b, slot, &seg, width);
    }
}

/// A standing capsule over `foot`: a cylinder of `radius` from the ground to `height -
/// radius` with a dome over it (a person).
pub fn capsule(b: &mut Builder, slot: usize, foot: Vec3, radius: f32, height: f32, segments: usize) {
    let shaft = (height - radius).max(radius);
    let ring = |z: f32, r: f32| -> Vec<Vec3> { (0..segments).map(|k| { let a = k as f32 / segments as f32 * std::f32::consts::TAU; foot + Vec3::new(a.cos() * r, a.sin() * r, z) }).collect() };
    let bottom = ring(0.0, radius);
    let top = ring(shaft, radius);
    for i in 0..segments {
        let j = (i + 1) % segments;
        b.quad(slot, bottom[i], bottom[j], top[j], top[i]);
    }
    // the dome in three steps
    let steps = 3;
    let mut prev = top;
    for k in 1..=steps {
        let t = k as f32 / steps as f32 * std::f32::consts::FRAC_PI_2;
        let r = radius * t.cos();
        let z = shaft + radius * t.sin();
        let next = ring(z, r.max(0.001));
        for i in 0..segments {
            let j = (i + 1) % segments;
            b.quad(slot, prev[i], prev[j], next[j], next[i]);
        }
        prev = next;
    }
}

/// Where a capsule stands for a recorded person: its foot on the ground `ground_z` where
/// that is known close to the recorded height, else at the recorded height (a person is
/// recorded at the pavement's height, which may differ from the mesh a little).
pub fn capsule_foot(recorded: DVec3, ground_z: Option<f64>, origin: DVec3) -> Vec3 {
    let z = match ground_z {
        Some(g) if (g - recorded.z).abs() < 1.5 => g,
        _ => recorded.z,
    };
    (DVec3::new(recorded.x, recorded.y, z) - origin).as_vec3()
}

// -------------------------------------------------------------------------------------------
// colours

/// What a lane is to the mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LaneClass {
    /// A street lane in its path's direction.
    Street,
    /// The reversed twin of a two-way path.
    Reversed,
    /// A bus lane: `no_cars`, or a density of 0.
    Bus,
    Rail,
    Pavement,
    /// A pedestrian piece of a crossing.
    Crossing,
    /// A short lit stub (a traffic light's own piece of pavement).
    LitStub,
}

impl LaneClass {
    /// The lane's class from what the simulation knows of it.
    pub fn of(kind: omsi_sim::traffic::LaneKind, reversed: bool, no_cars: bool, density: f32, has_light: bool, length_m: f32, crossing_piece: bool) -> LaneClass {
        use omsi_sim::traffic::LaneKind as K;
        match kind {
            K::Rail | K::Air => LaneClass::Rail,
            K::Sidewalk => {
                if has_light && length_m <= omsi_sim::traffic::STUB_MAX + 0.1 {
                    LaneClass::LitStub
                } else if crossing_piece {
                    LaneClass::Crossing
                } else {
                    LaneClass::Pavement
                }
            }
            K::Street => {
                if no_cars || density <= 0.0 {
                    LaneClass::Bus
                } else if reversed {
                    LaneClass::Reversed
                } else {
                    LaneClass::Street
                }
            }
        }
    }

    /// Fill colour (straight alpha).
    pub fn colour(self) -> [f32; 4] {
        match self {
            LaneClass::Street => [0.16, 0.72, 0.30, 0.45],
            LaneClass::Reversed => [0.45, 0.85, 0.45, 0.45],
            LaneClass::Bus => [0.10, 0.70, 0.70, 0.50],
            LaneClass::Rail => [0.62, 0.45, 0.27, 0.45],
            LaneClass::Pavement => [0.76, 0.69, 0.42, 0.45],
            LaneClass::Crossing => [0.95, 0.95, 0.98, 0.30],
            LaneClass::LitStub => [0.98, 0.90, 0.20, 0.70],
        }
    }

    pub const ALL: [LaneClass; 7] = [LaneClass::Street, LaneClass::Reversed, LaneClass::Bus, LaneClass::Rail, LaneClass::Pavement, LaneClass::Crossing, LaneClass::LitStub];

    pub fn name(self) -> &'static str {
        match self {
            LaneClass::Street => "street lane",
            LaneClass::Reversed => "reversed twin",
            LaneClass::Bus => "bus lane / no_cars",
            LaneClass::Rail => "rail",
            LaneClass::Pavement => "pavement",
            LaneClass::Crossing => "crossing piece (striped)",
            LaneClass::LitStub => "lit stub",
        }
    }
}

/// A recorded vehicle's state, by what it was doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VehicleState {
    Moving,
    Lead,
    People,
    Zebra,
    Yield,
    Light,
    Exit,
    /// Standing for anything else (service, merge, end ...).
    Other,
}

impl VehicleState {
    /// From the planner's reason (`Why::name`) and whether the vehicle stands.
    pub fn of(why: &str, standing: bool) -> VehicleState {
        if !standing {
            return VehicleState::Moving;
        }
        match why {
            "lead" => VehicleState::Lead,
            "people" => VehicleState::People,
            "zebra" => VehicleState::Zebra,
            "yield" => VehicleState::Yield,
            "light" => VehicleState::Light,
            "exit" => VehicleState::Exit,
            _ => VehicleState::Other,
        }
    }

    pub fn colour(self) -> [f32; 4] {
        match self {
            VehicleState::Moving => [0.55, 0.60, 0.66, 0.40],
            VehicleState::Lead => [0.31, 0.56, 0.97, 0.55],
            VehicleState::People => [0.71, 0.49, 0.86, 0.55],
            VehicleState::Zebra => [0.85, 0.40, 0.90, 0.55],
            VehicleState::Yield => [0.94, 0.53, 0.24, 0.55],
            VehicleState::Light => [0.90, 0.33, 0.29, 0.55],
            VehicleState::Exit => [0.98, 0.75, 0.20, 0.55],
            VehicleState::Other => [0.25, 0.70, 0.31, 0.55],
        }
    }

    pub const ALL: [VehicleState; 8] = [VehicleState::Moving, VehicleState::Lead, VehicleState::People, VehicleState::Zebra, VehicleState::Yield, VehicleState::Light, VehicleState::Exit, VehicleState::Other];

    pub fn name(self) -> &'static str {
        match self {
            VehicleState::Moving => "moving",
            VehicleState::Lead => "standing: lead",
            VehicleState::People => "standing: people",
            VehicleState::Zebra => "standing: zebra",
            VehicleState::Yield => "standing: yield",
            VehicleState::Light => "standing: light",
            VehicleState::Exit => "standing: exit",
            VehicleState::Other => "standing: other",
        }
    }
}

/// What a recorded person decided, as the recording says it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PersonState {
    Go,
    RedLight,
    Waits,
    Caught,
    Yields,
    /// No decision recorded (walking along, standing about).
    None,
}

impl PersonState {
    pub fn of(decision: Option<&str>) -> PersonState {
        let Some(d) = decision.map(str::trim).filter(|d| !d.is_empty()) else { return PersonState::None };
        let l = d.to_ascii_lowercase();
        if l == "go" {
            PersonState::Go
        } else if l.starts_with("red") {
            PersonState::RedLight
        } else if l.starts_with("caught") {
            PersonState::Caught
        } else if l.contains("yield") {
            PersonState::Yields
        } else {
            PersonState::Waits
        }
    }

    pub fn colour(self) -> [f32; 4] {
        match self {
            PersonState::Go => [0.20, 0.85, 0.35, 0.80],
            PersonState::RedLight => [0.92, 0.25, 0.22, 0.80],
            PersonState::Waits => [0.98, 0.78, 0.20, 0.80],
            PersonState::Caught => [0.98, 0.50, 0.10, 0.80],
            PersonState::Yields => [0.65, 0.45, 0.90, 0.80],
            PersonState::None => [0.85, 0.85, 0.85, 0.75],
        }
    }

    pub const ALL: [PersonState; 6] = [PersonState::Go, PersonState::RedLight, PersonState::Waits, PersonState::Caught, PersonState::Yields, PersonState::None];

    pub fn name(self) -> &'static str {
        match self {
            PersonState::Go => "person: go",
            PersonState::RedLight => "person: red light",
            PersonState::Waits => "person: waits",
            PersonState::Caught => "person: caught by the change",
            PersonState::Yields => "person: yields",
            PersonState::None => "person: no decision",
        }
    }
}

/// The height tint of an edge line: blue low, red high over the window's heights.
pub fn height_colour(z: f32, lo: f32, hi: f32) -> [f32; 4] {
    let t = if hi > lo { ((z - lo) / (hi - lo)).clamp(0.0, 1.0) } else { 0.5 };
    [t, 0.3 + 0.4 * (1.0 - (2.0 * t - 1.0).abs()), 1.0 - t, 0.95]
}

#[cfg(test)]
mod tests {
    use super::*;
    use omsi_sim::traffic::LaneKind;

    fn v(x: f32, y: f32) -> Vec3 {
        Vec3::new(x, y, 0.0)
    }

    #[test]
    fn a_straight_lane_has_parallel_edges_at_half_the_width() {
        let centre = [v(0.0, 0.0), v(0.0, 10.0), v(0.0, 20.0)];
        let (l, r) = footprint_edges(&centre, 3.0);
        // heading north: left is west (-x), right is east (+x)
        for (a, b) in l.iter().zip(&r) {
            assert!((a.x + 1.5).abs() < 1e-5, "left {a:?}");
            assert!((b.x - 1.5).abs() < 1e-5, "right {b:?}");
        }
        let poly = footprint_polygon(&centre, 3.0);
        assert_eq!(poly.len(), 6);
        assert!((area2(&poly).abs() - 2.0 * 60.0).abs() < 1e-3);
    }

    #[test]
    fn a_right_angle_joint_is_mitred_to_one_corner() {
        // north, then east: the outer (left) corner lies at (-1, 11), the inner at (1, 9)
        let centre = [v(0.0, 0.0), v(0.0, 10.0), v(10.0, 10.0)];
        let (l, r) = footprint_edges(&centre, 2.0);
        assert!((l[1] - v(-1.0, 11.0)).length() < 1e-4, "outer {:?}", l[1]);
        assert!((r[1] - v(1.0, 9.0)).length() < 1e-4, "inner {:?}", r[1]);
        // the ends keep the plain half width
        assert!((l[0] - v(-1.0, 0.0)).length() < 1e-5);
        assert!((r[2] - v(10.0, 9.0)).length() < 1e-5);
        // the polygon's area: 20 m of lane 2 m wide, the outer corner adding what the inner takes
        let poly = footprint_polygon(&centre, 2.0);
        assert!((area2(&poly).abs() - 2.0 * 40.0).abs() < 1e-3, "{}", area2(&poly));
    }

    #[test]
    fn a_sharp_joint_does_not_spike() {
        let centre = [v(0.0, 0.0), v(0.0, 10.0), v(0.5, 0.0)];
        let (l, r) = footprint_edges(&centre, 2.0);
        for p in l.iter().chain(&r) {
            assert!((p - centre[1]).length() < 2.5 || (p - centre[0]).length() < 2.5 || (p - centre[2]).length() < 2.5, "{p:?}");
        }
    }

    #[test]
    fn a_ribbon_carries_the_height_of_its_points() {
        let centre = [Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 5.0, 2.0)];
        let mut b = Builder::new(1);
        line_ribbon(&mut b, 0, &centre, 0.2);
        let m = b.finish();
        assert_eq!(m.ranges, vec![(0, 6, 0)]);
        assert!(m.positions.iter().all(|p| p.z >= 1.0 - 1e-6 && p.z <= 2.0 + 1e-6));
    }

    #[test]
    fn along_walks_the_polyline() {
        let line = [v(0.0, 0.0), v(0.0, 10.0), v(10.0, 10.0)];
        assert!((length(&line) - 20.0).abs() < 1e-6);
        let (p, d) = along(&line, 5.0);
        assert!((p - v(0.0, 5.0)).length() < 1e-6 && (d - Vec3::Y).length() < 1e-6);
        let (p, d) = along(&line, 15.0);
        assert!((p - v(5.0, 10.0)).length() < 1e-6 && (d - Vec3::X).length() < 1e-6);
        let (p, _) = along(&line, 99.0);
        assert!((p - v(10.0, 10.0)).length() < 1e-6);
    }

    #[test]
    fn bands_cover_the_lane_every_period() {
        let centre = [v(0.0, 0.0), v(0.0, 10.0)];
        let mut b = Builder::new(1);
        bands(&mut b, 0, &centre, 2.0, 1.0, 0.5);
        let m = b.finish();
        // ten bands of two triangles
        assert_eq!(m.indices.len(), 10 * 6);
        assert!(m.positions.iter().all(|p| p.x.abs() <= 1.0 + 1e-5));
    }

    #[test]
    fn a_box_is_laid_by_heading_front_and_rear() {
        // heading east (90): front +x
        let c = box_corners(v(0.0, 0.0), 90.0, 3.0, 1.0, 0.9);
        assert!((c[0] - v(3.0, 0.9)).length() < 1e-5, "{:?}", c[0]);
        assert!((c[1] - v(3.0, -0.9)).length() < 1e-5, "{:?}", c[1]);
        assert!((c[2] - v(-1.0, -0.9)).length() < 1e-5, "{:?}", c[2]);
        assert!((c[3] - v(-1.0, 0.9)).length() < 1e-5, "{:?}", c[3]);
        let mut b = Builder::new(2);
        block(&mut b, 1, c, 1.5);
        let m = b.finish();
        // a top and four sides, in slot 1 only
        assert_eq!(m.ranges, vec![(0, 30, 1)]);
        assert!((m.positions.iter().map(|p| p.z).fold(0.0, f32::max) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn a_capsule_stands_on_the_ground_where_it_is_known() {
        let origin = DVec3::new(1000.0, 2000.0, 0.0);
        let rec = DVec3::new(1010.0, 2005.0, 3.2);
        assert_eq!(capsule_foot(rec, Some(3.0), origin), Vec3::new(10.0, 5.0, 3.0));
        // a ground far off (under a bridge) is not it
        assert_eq!(capsule_foot(rec, Some(-4.0), origin), Vec3::new(10.0, 5.0, 3.2));
        assert_eq!(capsule_foot(rec, None, origin), Vec3::new(10.0, 5.0, 3.2));
        let mut b = Builder::new(1);
        capsule(&mut b, 0, Vec3::ZERO, PERSON_RADIUS, 1.7, 8);
        let m = b.finish();
        let top = m.positions.iter().map(|p| p.z).fold(0.0, f32::max);
        assert!((top - 1.7).abs() < 1e-4, "{top}");
        assert!(m.positions.iter().all(|p| p.truncate().length() <= PERSON_RADIUS + 1e-4));
    }

    #[test]
    fn lane_classes() {
        assert_eq!(LaneClass::of(LaneKind::Street, false, false, 1.0, false, 50.0, false), LaneClass::Street);
        assert_eq!(LaneClass::of(LaneKind::Street, true, false, 1.0, false, 50.0, false), LaneClass::Reversed);
        assert_eq!(LaneClass::of(LaneKind::Street, true, true, 1.0, false, 50.0, false), LaneClass::Bus);
        assert_eq!(LaneClass::of(LaneKind::Street, false, false, 0.0, false, 50.0, false), LaneClass::Bus);
        assert_eq!(LaneClass::of(LaneKind::Sidewalk, false, false, 1.0, false, 50.0, false), LaneClass::Pavement);
        assert_eq!(LaneClass::of(LaneKind::Sidewalk, false, false, 1.0, false, 8.0, true), LaneClass::Crossing);
        assert_eq!(LaneClass::of(LaneKind::Sidewalk, false, false, 1.0, true, 1.4, true), LaneClass::LitStub);
        assert_eq!(LaneClass::of(LaneKind::Sidewalk, false, false, 1.0, true, 8.0, false), LaneClass::Pavement);
        assert_eq!(LaneClass::of(LaneKind::Rail, false, false, 1.0, false, 50.0, false), LaneClass::Rail);
    }

    #[test]
    fn vehicle_states_follow_the_planners_reason_only_while_standing() {
        assert_eq!(VehicleState::of("lead", false), VehicleState::Moving);
        assert_eq!(VehicleState::of("lead", true), VehicleState::Lead);
        assert_eq!(VehicleState::of("people", true), VehicleState::People);
        assert_eq!(VehicleState::of("zebra", true), VehicleState::Zebra);
        assert_eq!(VehicleState::of("yield", true), VehicleState::Yield);
        assert_eq!(VehicleState::of("light", true), VehicleState::Light);
        assert_eq!(VehicleState::of("exit", true), VehicleState::Exit);
        assert_eq!(VehicleState::of("service", true), VehicleState::Other);
        assert_eq!(VehicleState::of("", true), VehicleState::Other);
        // every state has its own colour
        let mut seen = std::collections::HashSet::new();
        for s in VehicleState::ALL {
            assert!(seen.insert(s.colour().map(|c| (c * 1000.0) as i32)), "{s:?} shares a colour");
        }
    }

    #[test]
    fn person_states_from_the_recorded_decision() {
        assert_eq!(PersonState::of(None), PersonState::None);
        assert_eq!(PersonState::of(Some("")), PersonState::None);
        assert_eq!(PersonState::of(Some("go")), PersonState::Go);
        assert_eq!(PersonState::of(Some("red light")), PersonState::RedLight);
        assert_eq!(PersonState::of(Some("caught by the change: back to the kerb")), PersonState::Caught);
        assert_eq!(PersonState::of(Some("yields to car 12")), PersonState::Yields);
        assert_eq!(PersonState::of(Some("waits for 3 cars")), PersonState::Waits);
    }

    #[test]
    fn heights_tint_blue_to_red() {
        assert_eq!(height_colour(0.0, 0.0, 1.0)[2], 1.0);
        assert_eq!(height_colour(1.0, 0.0, 1.0)[0], 1.0);
        assert_eq!(height_colour(5.0, 2.0, 2.0)[0], 0.5);
    }
}
