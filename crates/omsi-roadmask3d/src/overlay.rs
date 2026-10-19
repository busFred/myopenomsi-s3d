//! The mask laid into the scene: translucent, unlit meshes a few centimetres over the
//! ground, drawn with depth (hidden by a building, over the road), as instances of the
//! game's renderer in the last render phase without depth writes. The lanes, their edges,
//! the map's visual kerb and road edges, the stop lines and kerb spots are built once; the
//! vehicles, people, the player's bus and the pinned vehicle's lines are rebuilt every frame
//! (their meshes freed and the scene's arrays cut back to the static part).

use crate::geometry::{self, Builder, LaneClass, PersonState, VehicleState, PERSON_RADIUS};
use crate::timeline::{FrameSnap, Timeline, VehicleSnap};
use glam::{DVec3, Mat4, Vec3};
use omsi_geometry::MeshData;
use omsi_render::{AlphaMode, MaterialExtra, MaterialId, MeshId, RenderPhase, Renderer, Scene};
use openomsi_game::tool::World;
use std::collections::HashMap;

/// The overlay's layers (keys 1-7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layer {
    Footprints = 0,
    LaneEdges = 1,
    ProfileEdges = 2,
    Stops = 3,
    Vehicles = 4,
    People = 5,
    Labels = 6,
}

impl Layer {
    pub const ALL: [Layer; 7] = [Layer::Footprints, Layer::LaneEdges, Layer::ProfileEdges, Layer::Stops, Layer::Vehicles, Layer::People, Layer::Labels];
    pub fn name(self) -> &'static str {
        match self {
            Layer::Footprints => "lane footprints",
            Layer::LaneEdges => "lane edges",
            Layer::ProfileEdges => "kerb / road edges (.sli)",
            Layer::Stops => "stop lines, kerb spots",
            Layer::Vehicles => "vehicles",
            Layer::People => "people",
            Layer::Labels => "labels",
        }
    }
}

/// A visual edge line of a spline's cross-section laid along its placement.
pub struct ProfileEdge {
    /// "kerb", "outer", "edge" (seams are not drawn).
    pub kind: &'static str,
    /// Height over the spline (m).
    pub z: f32,
    pub points: Vec<DVec3>,
}

/// Heights of the footprints and lines over the recorded points (m): the lanes lie on
/// the road mesh, which is pulled a little towards the camera by the renderer.
const LIFT_FILL: f32 = 0.06;
const LIFT_BANDS: f32 = 0.08;
const LIFT_EDGES: f32 = 0.09;
const LIFT_STOPS: f32 = 0.10;
const LIFT_PROFILE: f32 = 0.05;
/// Line widths (m).
const W_LANE_EDGE: f32 = 0.08;
const W_KERB: f32 = 0.11;
const W_EDGE: f32 = 0.07;
const W_OUTLINE: f32 = 0.14;
const W_STRIP: f32 = 0.10;

/// Fixed colours of the mask's lines.
const C_LANE_EDGE: [f32; 4] = [0.04, 0.04, 0.05, 0.85];
const C_KERB: [f32; 4] = [1.0, 1.0, 1.0, 0.95];
const C_EDGE: [f32; 4] = [0.70, 0.80, 0.95, 0.90];
const C_STRIPE: [f32; 4] = [1.0, 1.0, 1.0, 0.55];
const C_HATCH: [f32; 4] = [1.0, 1.0, 1.0, 0.33];
const C_STOP: [f32; 4] = [1.0, 1.0, 1.0, 0.95];
const C_STOP_GAP: [f32; 4] = [1.0, 1.0, 1.0, 0.45];
const C_KERB_SPOT: [f32; 4] = [1.0, 0.83, 0.24, 0.95];
const C_PLAYER: [f32; 4] = [1.0, 1.0, 1.0, 0.95];
const C_PIN: [f32; 4] = [1.0, 0.0, 0.78, 0.95];
const C_STRIP: [f32; 4] = [1.0, 0.55, 0.82, 0.95];
const C_WAY: [f32; 4] = [0.0, 0.90, 1.0, 0.70];
const C_ATTENTION: [f32; 4] = [1.0, 0.25, 0.25, 0.95];
const C_AT_KERB: [f32; 4] = [1.0, 0.83, 0.24, 0.90];
/// The height tint has this many steps.
const HEIGHT_STEPS: usize = 8;

pub struct Overlay {
    pub origin: DVec3,
    materials: HashMap<[i32; 4], MaterialId>,
    /// (layer, instance) of the static part; the profile edges twice: by kind and by height.
    static_instances: Vec<(Layer, usize)>,
    profile_by_kind: Vec<usize>,
    profile_by_height: Vec<usize>,
    /// The scene's array lengths after the static part: the dynamic part is cut back to them.
    static_meshes: usize,
    static_instance_count: usize,
    dynamic_instances: Vec<(Layer, usize)>,
    dynamic_meshes: Vec<MeshId>,
    pub layers_on: [bool; 7],
    pub heights: bool,
    /// The profile edges' height range over their splines (m): what the height tint spans.
    pub z_range: (f32, f32),
    pub counts: Counts,
}

#[derive(Default, Debug, Clone)]
pub struct Counts {
    pub lanes: usize,
    pub stop_lines: usize,
    pub kerb_spots: usize,
    pub profile_edges: usize,
}

fn key(c: [f32; 4]) -> [i32; 4] {
    c.map(|v| (v * 1000.0).round() as i32)
}

impl Overlay {
    pub fn new(origin: DVec3) -> Overlay {
        Overlay {
            origin,
            materials: HashMap::new(),
            static_instances: Vec::new(),
            profile_by_kind: Vec::new(),
            profile_by_height: Vec::new(),
            static_meshes: 0,
            static_instance_count: 0,
            dynamic_instances: Vec::new(),
            dynamic_meshes: Vec::new(),
            layers_on: [true; 7],
            heights: false,
            z_range: (0.0, 1.0),
            counts: Counts::default(),
        }
    }

    fn material(&mut self, renderer: &Renderer, scene: &mut Scene, colour: [f32; 4]) -> MaterialId {
        let k = key(colour);
        if let Some(&m) = self.materials.get(&k) {
            return m;
        }
        let m = renderer.add_material_extra(scene, None, AlphaMode::Blend, colour, true, None, None, None, None, [0.0; 3], MaterialExtra { no_z_write: true, ..Default::default() });
        self.materials.insert(k, m);
        m
    }

    fn rel(&self, p: DVec3, lift: f32) -> Vec3 {
        (p - self.origin).as_vec3() + Vec3::Z * lift
    }

    /// Put a built mesh into the scene with its colours; the instance's index.
    fn place(&mut self, renderer: &Renderer, scene: &mut Scene, mesh: MeshData, colours: &[[f32; 4]]) -> Option<usize> {
        if mesh.indices.is_empty() {
            return None;
        }
        let mats: Vec<MaterialId> = colours.iter().map(|c| self.material(renderer, scene, *c)).collect();
        let id = renderer.add_mesh(scene, &mesh);
        let inst = renderer.add_instance(scene, id, self.origin, Mat4::IDENTITY, mats);
        let i = &mut scene.instances[inst];
        i.render_phase = RenderPhase::AfterVehicles;
        i.casts_shadow = false;
        i.any_distance = true;
        Some(inst)
    }

    /// Build the static part. Every colour the dynamic part uses is made here too, so that
    /// nothing is added to the materials after the static marker.
    ///
    /// The lanes are laid on the ground the scene has (`World::walk_height`, the road and
    /// pavement meshes) where that is within 1.5 m of the recorded height: a lane's own
    /// points may lie a few centimetres under the road mesh, and the mask would be under
    /// the road with them (a bridge's lane, far over the ground, keeps its own height).
    pub fn build_static(&mut self, renderer: &Renderer, scene: &mut Scene, world: &World, tl: &Timeline, profile: &[ProfileEdge]) {
        let ground = |p: DVec3| -> DVec3 {
            match world.walk_height(p.x, p.y) {
                Some(g) if (g - p.z).abs() < 1.5 => DVec3::new(p.x, p.y, g.max(p.z)),
                _ => p,
            }
        };
        for s in VehicleState::ALL {
            self.material(renderer, scene, s.colour());
        }
        for s in PersonState::ALL {
            self.material(renderer, scene, s.colour());
        }
        for c in [C_PLAYER, C_PIN, C_STRIP, C_WAY, C_ATTENTION, C_AT_KERB] {
            self.material(renderer, scene, c);
        }
        // OMSI_MASK3D_DEBUG=1: the lanes near the middle against the scene's heights
        if std::env::var_os("OMSI_MASK3D_DEBUG").is_some() {
            for lane in &tl.lanes {
                let Some(p) = lane.points.iter().find(|p| (p.x - tl.centre.x).abs() < 40.0 && (p.y - tl.centre.y).abs() < 40.0) else { continue };
                eprintln!("lane {} {:?} w {:.2} at ({:.1}, {:.1}) z {:.2}: walk {:?} ground {:?}", lane.index, lane.class, lane.width, p.x, p.y, p.z, world.walk_height(p.x, p.y).map(|z| (z * 100.0).round() / 100.0), world.ground_height(p.x, p.y).map(|z| (z * 100.0).round() / 100.0));
            }
        }
        // footprints by class, the bands over crossings and junction pieces
        let classes = LaneClass::ALL;
        let mut fill = Builder::new(classes.len());
        let mut bands = Builder::new(2);
        let mut edges = Builder::new(1);
        for lane in &tl.lanes {
            let centre: Vec<Vec3> = lane.points.iter().map(|p| self.rel(ground(*p), LIFT_FILL)).collect();
            let slot = classes.iter().position(|c| *c == lane.class).unwrap_or(0);
            let (l, r) = geometry::footprint_edges(&centre, lane.width);
            fill.ribbon(slot, &l, &r);
            let raised: Vec<Vec3> = centre.iter().map(|p| *p + Vec3::Z * (LIFT_BANDS - LIFT_FILL)).collect();
            if lane.class == LaneClass::Crossing {
                geometry::bands(&mut bands, 0, &raised, lane.width, 1.0, 0.5);
            } else if lane.junction_piece {
                geometry::bands(&mut bands, 1, &raised, lane.width, 1.2, 0.3);
            }
            let lifted: Vec<Vec3> = centre.iter().map(|p| *p + Vec3::Z * (LIFT_EDGES - LIFT_FILL)).collect();
            let (l, r) = geometry::footprint_edges(&lifted, lane.width);
            geometry::line_ribbon(&mut edges, 0, &l, W_LANE_EDGE);
            geometry::line_ribbon(&mut edges, 0, &r, W_LANE_EDGE);
        }
        self.counts.lanes = tl.lanes.len();
        let colours: Vec<[f32; 4]> = classes.iter().map(|c| c.colour()).collect();
        if let Some(i) = self.place(renderer, scene, fill.finish(), &colours) {
            self.static_instances.push((Layer::Footprints, i));
        }
        if let Some(i) = self.place(renderer, scene, bands.finish(), &[C_STRIPE, C_HATCH]) {
            self.static_instances.push((Layer::Footprints, i));
        }
        if let Some(i) = self.place(renderer, scene, edges.finish(), &[C_LANE_EDGE]) {
            self.static_instances.push((Layer::LaneEdges, i));
        }

        // the map's visual edges: by kind, and by height
        let mut by_kind = Builder::new(2);
        let mut by_height = Builder::new(HEIGHT_STEPS);
        // (the tint is by the edge's height over its spline - the profile's own z - which
        // is what tells a kerb top from the road beside it; the world height would only
        // say where the hill is)
        let (plo, phi) = (profile.iter().map(|e| e.z).fold(f32::INFINITY, f32::min), profile.iter().map(|e| e.z).fold(f32::NEG_INFINITY, f32::max));
        for e in profile {
            let pts: Vec<Vec3> = e.points.iter().map(|p| self.rel(*p, LIFT_PROFILE)).collect();
            let (slot, w) = if e.kind == "kerb" { (0, W_KERB) } else { (1, W_EDGE) };
            geometry::line_ribbon(&mut by_kind, slot, &pts, w);
            let t = if phi > plo { ((e.z - plo) / (phi - plo)).clamp(0.0, 0.999) } else { 0.5 };
            geometry::line_ribbon(&mut by_height, (t * HEIGHT_STEPS as f32) as usize, &pts, w);
        }
        self.counts.profile_edges = profile.len();
        if let Some(i) = self.place(renderer, scene, by_kind.finish(), &[C_KERB, C_EDGE]) {
            self.static_instances.push((Layer::ProfileEdges, i));
            self.profile_by_kind.push(i);
        }
        let ramp: Vec<[f32; 4]> = (0..HEIGHT_STEPS).map(|k| geometry::height_colour((k as f32 + 0.5) / HEIGHT_STEPS as f32, 0.0, 1.0)).collect();
        if let Some(i) = self.place(renderer, scene, by_height.finish(), &ramp) {
            self.static_instances.push((Layer::ProfileEdges, i));
            self.profile_by_height.push(i);
        }
        if phi.is_finite() {
            self.z_range = (plo, phi.max(plo + 0.01));
        }

        // stop lines and kerb spots
        let mut stops = Builder::new(3);
        for lane in &tl.lanes {
            if let Some((p, h)) = lane.stop_line {
                let (s, c) = h.to_radians().sin_cos();
                let fwd = Vec3::new(s, c, 0.0);
                let right = Vec3::new(c, -s, 0.0) * lane.width * 0.5;
                let at = self.rel(ground(p), LIFT_STOPS);
                stops.quad(0, at - right, at + right, at + right + fwd * 0.2, at - right + fwd * 0.2);
                // the 0.6 m a car's front keeps before the line
                let gap = at + fwd * 0.6;
                stops.quad(1, gap - right, gap + right, gap + right + fwd * 0.06, gap - right + fwd * 0.06);
                self.counts.stop_lines += 1;
            }
        }
        for k in &tl.kerb_spots {
            let at = self.rel(ground(*k), LIFT_STOPS + 0.02);
            stops.disc(2, at, 0.28, 4);
        }
        self.counts.kerb_spots = tl.kerb_spots.len();
        if let Some(i) = self.place(renderer, scene, stops.finish(), &[C_STOP, C_STOP_GAP, C_KERB_SPOT]) {
            self.static_instances.push((Layer::Stops, i));
        }

        self.static_meshes = scene.meshes.len();
        self.static_instance_count = scene.instances.len();
        self.apply_visibility(renderer, scene);
    }

    /// Show the layers as switched (and the profile edges by kind or by height).
    pub fn apply_visibility(&mut self, renderer: &Renderer, scene: &mut Scene) {
        let list: Vec<(Layer, usize)> = self.static_instances.iter().chain(self.dynamic_instances.iter()).copied().collect();
        for (layer, i) in list {
            let mut on = self.layers_on[layer as usize];
            if self.profile_by_kind.contains(&i) {
                on &= !self.heights;
            }
            if self.profile_by_height.contains(&i) {
                on &= self.heights;
            }
            renderer.set_params(scene, i, &[], on, &[]);
        }
    }

    pub fn toggle_layer(&mut self, renderer: &Renderer, scene: &mut Scene, layer: Layer) {
        self.layers_on[layer as usize] = !self.layers_on[layer as usize];
        self.apply_visibility(renderer, scene);
    }

    pub fn toggle_heights(&mut self, renderer: &Renderer, scene: &mut Scene) {
        self.heights = !self.heights;
        self.apply_visibility(renderer, scene);
    }

    /// Take the dynamic part out of the scene.
    fn clear_dynamic(&mut self, renderer: &Renderer, scene: &mut Scene) {
        for (_, i) in self.dynamic_instances.drain(..) {
            renderer.remove_instance(scene, i);
        }
        for m in self.dynamic_meshes.drain(..) {
            renderer.free_mesh(scene, m);
        }
        renderer.truncate(scene, self.static_meshes, scene.textures.len(), scene.materials.len(), self.static_instance_count);
    }

    fn place_dynamic(&mut self, renderer: &Renderer, scene: &mut Scene, layer: Layer, mesh: MeshData, colours: &[[f32; 4]]) {
        let n = scene.meshes.len();
        if let Some(i) = self.place(renderer, scene, mesh, colours) {
            self.dynamic_instances.push((layer, i));
            self.dynamic_meshes.push(n);
        }
    }

    /// Lay out one frame: the vehicles as blocks by state, the player's bus as a wire box,
    /// the people as capsules, and for the hovered and pinned vehicle its people strip, its
    /// attention line to its lead and its intended way.
    ///
    /// `player_box` false leaves the player's wire box out (the camera sits in it).
    #[allow(clippy::too_many_arguments)]
    pub fn update_frame(&mut self, renderer: &Renderer, scene: &mut Scene, world: &World, tl: &Timeline, frame: &FrameSnap, hovered: Option<u64>, pinned: Option<u64>, player_box: bool) {
        self.clear_dynamic(renderer, scene);
        let ground = |p: DVec3| -> f64 {
            match world.walk_height(p.x, p.y) {
                Some(g) if (g - p.z).abs() < 1.5 => g,
                _ => p.z,
            }
        };

        // vehicles
        let mut veh = Builder::new(VehicleState::ALL.len());
        for v in &frame.vehicles {
            let foot = self.rel(DVec3::new(v.pos.x, v.pos.y, ground(v.pos)), 0.02);
            let corners = geometry::box_corners(foot, v.heading, v.front, v.rear, v.half_width);
            let height = match v.class.as_str() {
                "Bus" | "ScheduledBus" | "Truck" | "Rail" => 3.0,
                _ => 1.5,
            };
            let slot = VehicleState::ALL.iter().position(|s| *s == v.state).unwrap_or(0);
            geometry::block(&mut veh, slot, corners, height);
        }
        let colours: Vec<[f32; 4]> = VehicleState::ALL.iter().map(|s| s.colour()).collect();
        self.place_dynamic(renderer, scene, Layer::Vehicles, veh.finish(), &colours);

        // the player's bus: a wire box
        let mut pl = Builder::new(1);
        for p in frame.players.iter().filter(|_| player_box) {
            let foot = self.rel(DVec3::new(p.pos.x, p.pos.y, ground(p.pos)), 0.06);
            let base = geometry::box_corners(foot, p.heading, p.half_len, p.half_len, p.half_width);
            let top: Vec<Vec3> = base.iter().map(|c| *c + Vec3::Z * 3.0).collect();
            geometry::outline(&mut pl, 0, &base, W_OUTLINE);
            geometry::outline(&mut pl, 0, &top, W_OUTLINE);
            for k in 0..4 {
                geometry::line_ribbon(&mut pl, 0, &[base[k], top[k]], W_OUTLINE);
            }
        }
        self.place_dynamic(renderer, scene, Layer::Vehicles, pl.finish(), &[C_PLAYER]);

        // the hovered and pinned vehicles' lines
        let mut lines = Builder::new(4);
        let mut marked: Vec<&VehicleSnap> = Vec::new();
        for id in [pinned, hovered].into_iter().flatten() {
            if let Some(v) = frame.vehicles.iter().find(|v| v.id == id) {
                if !marked.iter().any(|m| m.id == id) {
                    marked.push(v);
                }
            }
        }
        for v in marked {
            let foot = self.rel(DVec3::new(v.pos.x, v.pos.y, ground(v.pos)), 0.12);
            let corners = geometry::box_corners(foot, v.heading, v.front, v.rear, v.half_width);
            geometry::outline(&mut lines, 0, &corners, W_OUTLINE);
            // the strip of road it watches for people: its way ahead, half width + 0.3 m either side
            let way = tl.way_ahead(v.lane, v.s, &v.ahead, 30.0);
            if way.len() >= 2 {
                let pts: Vec<Vec3> = way.iter().map(|p| self.rel(*p, 0.11)).collect();
                let (l, r) = geometry::footprint_edges(&pts, 2.0 * (v.half_width + 0.3));
                geometry::line_ribbon(&mut lines, 1, &l, W_STRIP);
                geometry::line_ribbon(&mut lines, 1, &r, W_STRIP);
                // its intended way, as a band down the middle
                geometry::line_ribbon(&mut lines, 2, &pts, 0.25);
            }
            if Some(v.id) == pinned {
                // the attention line to what it follows or waits for
                // (a player waited for: PLAYER_ID_BASE + its index among the players)
                let target = v.lead_id.or(v.wait_for).and_then(|id| match id.checked_sub(omsi_traffic::sim::PLAYER_ID_BASE) {
                    Some(n) => frame.players.get(n as usize).map(|p| p.pos),
                    None => frame.vehicles.iter().find(|o| o.id == id).map(|o| o.pos),
                });
                if let Some(o) = target {
                    let a = self.rel(DVec3::new(v.pos.x, v.pos.y, ground(v.pos)), 1.2);
                    let b = self.rel(DVec3::new(o.x, o.y, ground(o)), 1.2);
                    geometry::line_ribbon(&mut lines, 3, &[a, b], 0.12);
                    // a vertical tick at the target
                    geometry::line_ribbon(&mut lines, 3, &[b, b + Vec3::Z * 1.5], 0.12);
                }
            }
        }
        self.place_dynamic(renderer, scene, Layer::Vehicles, lines.finish(), &[C_PIN, C_STRIP, C_WAY, C_ATTENTION]);

        // people
        let n_states = PersonState::ALL.len();
        let mut people = Builder::new(n_states + 1);
        for p in &frame.people {
            let foot = geometry::capsule_foot(p.pos, world.walk_height(p.pos.x, p.pos.y), self.origin) + Vec3::Z * 0.02;
            let slot = PersonState::ALL.iter().position(|s| *s == p.state).unwrap_or(n_states - 1);
            geometry::capsule(&mut people, slot, foot, PERSON_RADIUS, 1.7, 10);
            if p.at_kerb {
                // a flat ring on the ground round somebody standing at a kerb
                let ring: Vec<Vec3> = (0..=16).map(|k| { let a = k as f32 / 16.0 * std::f32::consts::TAU; foot + Vec3::new(a.cos(), a.sin(), 0.0) * (PERSON_RADIUS + 0.2) + Vec3::Z * 0.03 }).collect();
                geometry::line_ribbon(&mut people, n_states, &ring, 0.06);
            }
        }
        let mut colours: Vec<[f32; 4]> = PersonState::ALL.iter().map(|s| s.colour()).collect();
        colours.push(C_AT_KERB);
        self.place_dynamic(renderer, scene, Layer::People, people.finish(), &colours);

        self.apply_visibility(renderer, scene);
    }

    /// The legend's rows: (colour, name).
    pub fn legend(&self) -> Vec<([f32; 4], String)> {
        let mut rows = Vec::new();
        for c in LaneClass::ALL {
            rows.push((c.colour(), c.name().to_string()));
        }
        rows.push((C_HATCH, "junction piece (hatched)".into()));
        rows.push((C_LANE_EDGE, "lane edge (±width/2)".into()));
        rows.push((C_KERB, "kerb face (.sli)".into()));
        rows.push((C_EDGE, "road / pavement edge (.sli)".into()));
        rows.push((C_STOP, "stop line (thin: the 0.6 m gap)".into()));
        rows.push((C_KERB_SPOT, "kerb spot (a crossing's end)".into()));
        for s in VehicleState::ALL {
            rows.push((s.colour(), s.name().to_string()));
        }
        rows.push((C_PLAYER, "player's bus (wire box)".into()));
        rows.push((C_PIN, "pinned / hovered vehicle".into()));
        rows.push((C_STRIP, "its people strip (half width + 0.3 m)".into()));
        rows.push((C_WAY, "its intended way".into()));
        rows.push((C_ATTENTION, "its attention line (lead / waits for)".into()));
        for s in PersonState::ALL {
            rows.push((s.colour(), s.name().to_string()));
        }
        rows.push((C_AT_KERB, "person at a kerb (ring)".into()));
        rows
    }
}
