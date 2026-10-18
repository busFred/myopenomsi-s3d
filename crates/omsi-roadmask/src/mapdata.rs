//! The map side: the OMSI root and the game's settings (read, never written), the tiles a
//! window's lanes name, the spline instances and crossing objects behind the lanes' keys,
//! the spline types' `.sli` files and the objects' footprints.

use glam::{DVec2, DVec3, Vec3};
use omsi_geometry::SplineCurve;
use omsi_map::{GlobalCfg, MapObject, MapSpline, Tile};
use omsi_scenery::{SceneryObject, Spline};
use omsi_sim::traffic::LaneKey;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::rc::Rc;

/// The user's data folder, as the game and the launcher have it (`~/.openomsi`). Nothing is
/// created there: this tool only reads.
pub fn data_dir() -> PathBuf {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_default();
    home.join(".openomsi")
}

/// The launcher's configuration (`launcher.json`), read only.
fn launcher_json() -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(data_dir().join("launcher.json")).ok()?;
    serde_json::from_str(&text).ok()
}

/// The OMSI 2 folder: `--root`, else `OMSI_ROOT`, else the launcher's `root`.
pub fn omsi_root(flag: Option<&str>) -> Result<PathBuf, String> {
    if let Some(r) = flag {
        return Ok(PathBuf::from(r));
    }
    if let Some(r) = std::env::var_os("OMSI_ROOT") {
        return Ok(PathBuf::from(r));
    }
    let root = launcher_json()
        .and_then(|v| v.get("root").and_then(|r| r.as_str()).map(PathBuf::from))
        .filter(|p| !p.as_os_str().is_empty());
    root.ok_or_else(|| {
        format!(
            "no OMSI root: pass --root, or set `root` in {}",
            data_dir().join("launcher.json").display()
        )
    })
}

/// The game's content folder (mods), as the game adds it before the installation: the
/// `openOMSI` folder beside the configured game binary (see `omsi_cfg::content_folder_of`).
pub fn game_content_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("OMSI_CONTENT") {
        return Some(PathBuf::from(d)).filter(|p| p.is_dir());
    }
    let game = launcher_json()?.get("game")?.as_str()?.to_string();
    let dir = Path::new(&game).parent()?.to_path_buf();
    let c = omsi_cfg::content_folder_of(&dir);
    (c.is_dir()
        && (c.join("maps").is_dir()
            || c.join("Splines").is_dir()
            || c.join("Sceneryobjects").is_dir()))
    .then_some(c)
}

/// The game's `settings.cfg` as `key=value` lines, read only (what it says is reported; the
/// map data does not depend on it).
pub fn read_settings() -> Option<BTreeMap<String, String>> {
    let text = std::fs::read_to_string(data_dir().join("settings.cfg")).ok()?;
    Some(
        text.lines()
            .filter(|l| !l.trim_start().starts_with('#'))
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
            .collect(),
    )
}

/// The `.sli`'s cross-section laid along one placed spline.
pub struct SplineInstance {
    pub key: (i32, i32, i64),
    pub map: MapSpline,
    pub curve: SplineCurve,
    /// The `.sli` as the tile names it, and where it was found.
    pub file: String,
    pub sli: Option<Rc<Spline>>,
    pub sli_path: Option<PathBuf>,
    pub lanes: Vec<usize>,
}

/// A placed crossing object whose `[path]`s are lanes.
pub struct ObjectInstance {
    pub key: (i32, i32, i64),
    pub file: String,
    pub pos: DVec3,
    pub heading: f64,
    pub sco: Option<Rc<SceneryObject>>,
    pub sco_path: Option<PathBuf>,
    /// The plan outline of the object's meshes (world x east, y north), how it was got
    /// ("mesh hull of N vertices", "[boundingbox]"), or why there is none.
    pub footprint: Result<(Vec<DVec2>, String), String>,
    pub lanes: Vec<usize>,
}

pub struct MapData {
    pub root: PathBuf,
    pub map_dir: PathBuf,
    pub global: GlobalCfg,
    pub chrono_dirs: Vec<PathBuf>,
    tiles: HashMap<(i32, i32), Option<Tile>>,
    sli_cache: HashMap<String, Option<(Rc<Spline>, PathBuf)>>,
    sco_cache: HashMap<String, Option<(Rc<SceneryObject>, PathBuf)>>,
    pub splines: Vec<SplineInstance>,
    pub objects: Vec<ObjectInstance>,
    by_key: HashMap<(i32, i32, i64), (bool, usize)>,
    /// What could not be resolved, one line each (deduplicated).
    pub problems: Vec<String>,
    /// Tiles that were asked for and not found.
    pub missing_tiles: Vec<(i32, i32)>,
}

impl MapData {
    /// Open map `name` (the recording header's folder name) under `root`. `date`
    /// (YYYYMMDD) picks the chrono folders to apply; none applies no chrono.
    pub fn open(root: PathBuf, name: &str, date: Option<i32>) -> Result<MapData, String> {
        let map_dir = omsi_cfg::resolve_path(&root, &format!("maps\\{name}"));
        let global_path = omsi_cfg::resolve_path(&map_dir, "global.cfg");
        if !global_path.is_file() {
            return Err(format!(
                "map {name}: no global.cfg at {}",
                global_path.display()
            ));
        }
        let global =
            GlobalCfg::load(&global_path).map_err(|e| format!("{}: {e}", global_path.display()))?;
        omsi_map::configure_grid(&global);
        let chrono_dirs = date
            .map(|d| omsi_map::active_chrono_dirs(&map_dir, d))
            .unwrap_or_default();
        Ok(MapData {
            root,
            map_dir,
            global,
            chrono_dirs,
            tiles: HashMap::new(),
            sli_cache: HashMap::new(),
            sco_cache: HashMap::new(),
            splines: Vec::new(),
            objects: Vec::new(),
            by_key: HashMap::new(),
            problems: Vec::new(),
            missing_tiles: Vec::new(),
        })
    }

    fn problem(&mut self, text: String) {
        if !self.problems.contains(&text) {
            self.problems.push(text);
        }
    }

    /// The tile (tx, ty) with the chrono patches applied and fitted to the world grid, read
    /// once; None when its file is missing.
    fn tile(&mut self, tx: i32, ty: i32) -> Option<&Tile> {
        if !self.tiles.contains_key(&(tx, ty)) {
            let name = format!("tile_{tx}_{ty}.map");
            let path = omsi_cfg::resolve_path(&self.map_dir, &name);
            let mut loaded = None;
            if path.is_file() {
                match Tile::load(&path) {
                    Ok(mut tile) => {
                        for c in &self.chrono_dirs {
                            let p = omsi_cfg::resolve_path(c, &name);
                            if p.is_file() {
                                match Tile::load(&p) {
                                    Ok(patch) => {
                                        tile.apply_chrono(&patch);
                                    }
                                    Err(e) => log::warn!("{e}"),
                                }
                            }
                        }
                        tile.fit_to_world_grid(ty);
                        loaded = Some(tile);
                    }
                    Err(e) => self.problem(format!("tile {name}: {e}")),
                }
            } else {
                self.missing_tiles.push((tx, ty));
                self.problem(format!(
                    "tile {name} not found in {}",
                    self.map_dir.display()
                ));
            }
            self.tiles.insert((tx, ty), loaded);
        }
        self.tiles.get(&(tx, ty)).and_then(|t| t.as_ref())
    }

    fn sli(&mut self, file: &str) -> Option<(Rc<Spline>, PathBuf)> {
        let k = file.trim().to_ascii_lowercase().replace('/', "\\");
        if let Some(c) = self.sli_cache.get(&k) {
            return c.clone();
        }
        let path = omsi_cfg::resolve_path(&self.root, file.trim());
        let got = if path.is_file() {
            match Spline::load(&path) {
                Ok(s) => Some((Rc::new(s), path)),
                Err(e) => {
                    self.problem(format!("spline {file}: {e}"));
                    None
                }
            }
        } else {
            self.problem(format!("spline file not found: {file}"));
            None
        };
        self.sli_cache.insert(k, got.clone());
        got
    }

    fn sco(&mut self, file: &str) -> Option<(Rc<SceneryObject>, PathBuf)> {
        let k = file.trim().to_ascii_lowercase().replace('/', "\\");
        if let Some(c) = self.sco_cache.get(&k) {
            return c.clone();
        }
        let path = omsi_cfg::resolve_path(&self.root, file.trim());
        let got = if path.is_file() {
            match SceneryObject::load(&path) {
                Ok(s) => Some((Rc::new(s), path)),
                Err(e) => {
                    self.problem(format!("object {file}: {e}"));
                    None
                }
            }
        } else {
            self.problem(format!("object file not found: {file}"));
            None
        };
        self.sco_cache.insert(k, got.clone());
        got
    }

    /// Resolve lane `lane` (index) with key `key` and the lane's `source` (1 spline, 2
    /// object, 0 unknown): the spline or object instance it belongs to, created on first
    /// sight. Returns (is_spline, index) or None with the problem recorded.
    pub fn resolve(
        &mut self,
        lane: usize,
        key: LaneKey,
        source: u32,
        lane_name: &str,
    ) -> Option<(bool, usize)> {
        let k = (key.tile.0, key.tile.1, key.id);
        if let Some(&(is_spline, i)) = self.by_key.get(&k) {
            if is_spline {
                self.splines[i].lanes.push(lane);
            } else {
                self.objects[i].lanes.push(lane);
            }
            return Some((is_spline, i));
        }
        let origin = {
            let (x, y) = omsi_map::tile_local_to_world(key.tile.0, key.tile.1, 0.0, 0.0);
            DVec2::new(x, y)
        };
        let Some(tile) = self.tile(key.tile.0, key.tile.1) else {
            return None;
        };
        let spline = tile
            .splines
            .iter()
            .find(|s| s.id == key.id && !s.deleted)
            .cloned();
        let object: Option<MapObject> = tile
            .objects
            .iter()
            .chain(tile.attach_objects.iter())
            .find(|o| o.id == key.id)
            .cloned();
        match (spline, object) {
            (Some(s), _) if source != 2 => {
                let curve = SplineCurve::from_map(&s, origin);
                let got = self.sli(&s.file);
                let curve = match &got {
                    Some((def, _)) => curve.with_sli(def),
                    None => curve,
                };
                let inst = SplineInstance {
                    key: k,
                    file: s.file.clone(),
                    map: s,
                    curve,
                    sli: got.as_ref().map(|g| g.0.clone()),
                    sli_path: got.map(|g| g.1),
                    lanes: vec![lane],
                };
                self.splines.push(inst);
                let i = self.splines.len() - 1;
                self.by_key.insert(k, (true, i));
                Some((true, i))
            }
            (_, Some(o)) => {
                let (x, y) =
                    omsi_map::tile_local_to_world(key.tile.0, key.tile.1, o.pos[0], o.pos[1]);
                let pos = DVec3::new(x, y, o.pos[2]);
                let got = self.sco(&o.file);
                let footprint = match &got {
                    Some((sco, path)) => footprint(sco, path),
                    None => Err("the .sco was not found".to_string()),
                };
                if let Err(why) = &footprint {
                    self.problem(format!(
                        "object {} (tile {},{} id {}): no footprint: {why}",
                        o.file, key.tile.0, key.tile.1, key.id
                    ));
                }
                let inst = ObjectInstance {
                    key: k,
                    file: o.file.clone(),
                    pos,
                    heading: o.rot[0],
                    sco: got.as_ref().map(|g| g.0.clone()),
                    sco_path: got.map(|g| g.1),
                    footprint,
                    lanes: vec![lane],
                };
                self.objects.push(inst);
                let i = self.objects.len() - 1;
                self.by_key.insert(k, (false, i));
                Some((false, i))
            }
            (Some(_), None) => {
                self.problem(format!("lane {lane} ({lane_name}): key names spline id {} of tile {},{} but the lane came from an object", key.id, key.tile.0, key.tile.1));
                None
            }
            (None, None) => {
                self.problem(format!("lane {lane} ({lane_name}): id {} is neither a spline nor an object of tile {},{}", key.id, key.tile.0, key.tile.1));
                None
            }
        }
    }
}

/// How many mesh vertices a footprint takes in at most.
const HULL_MAX_VERTICES: usize = 400_000;

/// The plan outline of a scenery object in the world: the convex hull of its first LOD's
/// meshes (x right, y forward in the object frame, turned by its heading), else its
/// `[boundingbox]`.
fn footprint(sco: &SceneryObject, sco_path: &Path) -> Result<(Vec<DVec2>, String), String> {
    let sco_dir = sco_path.parent().ok_or("no folder")?.to_path_buf();
    let (model, model_dir) = match &sco.model_file {
        Some(m) => {
            let mp = omsi_cfg::resolve_path(&sco_dir, m);
            match omsi_model::Model::load(&mp) {
                Ok(model) => (
                    model,
                    mp.parent()
                        .map(|p| p.to_path_buf())
                        .unwrap_or(sco_dir.clone()),
                ),
                Err(e) => return bbox_footprint(sco).ok_or(format!("model {m}: {e}")),
            }
        }
        None => (sco.model.clone(), sco_dir.clone()),
    };
    let meshes = if model.lods.is_empty() {
        &model.meshes[..]
    } else {
        model.lod_meshes(0)
    };
    let mut pts: Vec<DVec2> = Vec::new();
    let mut files = 0usize;
    let mut missing = Vec::new();
    for md in meshes.iter().take(80) {
        if md.file.trim().is_empty() {
            continue;
        }
        let mesh_path =
            omsi_cfg::resolve_path(&omsi_cfg::resolve_path(&model_dir, "model"), &md.file);
        let mesh_path = if mesh_path.is_file() {
            mesh_path
        } else {
            omsi_cfg::resolve_path(&model_dir, &md.file)
        };
        let mesh = match omsi_o3d::load_mesh(&mesh_path) {
            Ok(m) => m,
            Err(_) => {
                missing.push(md.file.clone());
                continue;
            }
        };
        files += 1;
        for v in mesh.vertices.iter() {
            if pts.len() >= HULL_MAX_VERTICES {
                break;
            }
            let p: Vec3 = if mesh.has_transform {
                mesh.transform.transform_point3(v.position)
            } else {
                v.position
            };
            if p.x.is_finite() && p.y.is_finite() {
                pts.push(DVec2::new(p.x as f64, p.y as f64));
            }
        }
    }
    if pts.len() < 3 {
        let why = if files == 0 {
            format!(
                "no mesh could be read ({} named{})",
                meshes.len(),
                if missing.is_empty() {
                    String::new()
                } else {
                    format!(": {}", missing.join(", "))
                }
            )
        } else {
            "the meshes have too few vertices".to_string()
        };
        return bbox_footprint(sco).ok_or(why);
    }
    let n = pts.len();
    let hull = convex_hull(pts);
    let mut how = format!(
        "mesh hull of {n} vertices in {files} mesh file{}",
        if files == 1 { "" } else { "s" }
    );
    if !missing.is_empty() {
        how.push_str(&format!(
            " ({} mesh file{} not read)",
            missing.len(),
            if missing.len() == 1 { "" } else { "s" }
        ));
    }
    Ok((hull, how))
}

fn bbox_footprint(sco: &SceneryObject) -> Option<(Vec<DVec2>, String)> {
    let bb = sco.bounding_box?;
    let (sx, sy) = (bb[0] as f64, bb[1] as f64);
    let (cx, cy) = (bb[3] as f64, bb[4] as f64);
    if sx <= 0.0 || sy <= 0.0 {
        return None;
    }
    Some((
        vec![
            DVec2::new(cx - sx / 2.0, cy - sy / 2.0),
            DVec2::new(cx + sx / 2.0, cy - sy / 2.0),
            DVec2::new(cx + sx / 2.0, cy + sy / 2.0),
            DVec2::new(cx - sx / 2.0, cy + sy / 2.0),
        ],
        "[boundingbox] of the .sco".to_string(),
    ))
}

/// Andrew's monotone chain: the convex hull of `pts` counter-clockwise.
pub fn convex_hull(mut pts: Vec<DVec2>) -> Vec<DVec2> {
    pts.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    pts.dedup_by(|a, b| (a.x - b.x).abs() < 1e-9 && (a.y - b.y).abs() < 1e-9);
    if pts.len() < 3 {
        return pts;
    }
    let cross =
        |o: DVec2, a: DVec2, b: DVec2| (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    let mut lower: Vec<DVec2> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<DVec2> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// A point of an object's frame (x right, y forward) in the world, for an object at `pos`
/// turned clockwise by `heading` degrees (as the game places its `[path]`s).
pub fn object_to_world(pos: DVec3, heading: f64, x: f64, y: f64) -> DVec2 {
    let h = heading.to_radians();
    let (sh, ch) = (h.sin(), h.cos());
    DVec2::new(pos.x + x * ch + y * sh, pos.y - x * sh + y * ch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hull_of_a_square_with_inner_points() {
        let pts = vec![
            DVec2::new(0.0, 0.0),
            DVec2::new(2.0, 0.0),
            DVec2::new(2.0, 2.0),
            DVec2::new(0.0, 2.0),
            DVec2::new(1.0, 1.0),
            DVec2::new(0.5, 1.5),
        ];
        let h = convex_hull(pts);
        assert_eq!(h.len(), 4);
        assert!(h
            .iter()
            .all(|p| (p.x == 0.0 || p.x == 2.0) && (p.y == 0.0 || p.y == 2.0)));
    }

    #[test]
    fn object_frame_turns_clockwise() {
        // heading 90: the object's forward (+y) points east
        let p = object_to_world(DVec3::ZERO, 90.0, 0.0, 1.0);
        assert!((p.x - 1.0).abs() < 1e-9 && p.y.abs() < 1e-9);
        let q = object_to_world(DVec3::ZERO, 90.0, 1.0, 0.0);
        assert!(q.x.abs() < 1e-9 && (q.y + 1.0).abs() < 1e-9);
    }
}
