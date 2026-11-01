//! The body classes of the meeting places (`omsi_sim::traffic::SWEPT_CLASSES`) against the AI types
//! of the installed stock maps BeijingCity and X10 Berlin: every type is held by its class
//! (`swept_class`), and every dimension of a class is its largest member's.
use omsi_sim::ai_motion::{rear_off_tracking, rotation_point};
use omsi_sim::traffic::{swept_class, SWEPT_CLASSES};
use omsi_sim::vehicle::VehicleType;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The vehicle files a map's `ailists.cfg` names (the first field of each line).
fn ai_files(root: &Path, map: &str) -> Vec<String> {
    let Ok(bytes) = std::fs::read(root.join("maps").join(map).join("ailists.cfg")) else { return Vec::new() };
    let text: String = bytes.iter().map(|&b| b as char).collect();
    let mut out: Vec<String> = Vec::new();
    for line in text.lines() {
        let f = line.split('\t').next().unwrap_or("").trim();
        let low = f.to_ascii_lowercase();
        if low.starts_with("vehicles\\") && (low.ends_with(".bus") || low.ends_with(".ovh")) && !out.iter().any(|o| o == f) {
            out.push(f.to_string());
        }
    }
    out
}

/// A type's body as the traffic and its careful driver's line take it: (length bumper to bumper of
/// the whole consist, half width, reach from the rotation point to the front bumper, rear sections'
/// off-tracking), the coupled parts straight behind it.
fn body(root: &Path, ty: VehicleType) -> (f32, f32, f32, f32) {
    let bb = ty.def.bounding_box.unwrap_or([2.5, 4.5, 1.5, 0.0, 0.0, 0.0]);
    let (front, rear, hw) = (bb[1] * 0.5 + bb[4], bb[1] * 0.5 - bb[4], (bb[0] * 0.5).max(0.5));
    let mut parts = vec![Arc::new(ty)];
    let (mut back_most, mut origin) = (rear, 0.0f32);
    while let Some(next) = parts.last().unwrap().def.couple_back_path() {
        let Ok(t) = VehicleType::load_ai(root, &next) else { break };
        if parts.len() > 4 {
            break;
        }
        let (back, fr) = omsi_sim::vehicle::coupling_points(parts.last().unwrap(), false, &t, false);
        let tb = t.def.bounding_box.unwrap_or(bb);
        origin += back.y - fr.y;
        back_most = back_most.max(-(origin + tb[4] - tb[1] * 0.5));
        parts.push(Arc::new(t));
    }
    let refs: Vec<&VehicleType> = parts.iter().map(|p| &**p).collect();
    (front + back_most, hw, front - rotation_point(&refs[0].def).0, rear_off_tracking(&refs))
}

#[test]
fn swept_classes_hold_the_fleet() {
    let root = std::env::var_os("OMSI_ROOT").map(PathBuf::from).or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Game/omsi2")));
    let Some(root) = root.filter(|r| r.join("maps/BeijingCity/ailists.cfg").exists()) else {
        eprintln!("skipped: no BeijingCity");
        return;
    };
    let n = SWEPT_CLASSES.len();
    let mut most = vec![[0.0f32; 4]; n];
    let mut members = vec![0usize; n];
    for map in ["BeijingCity", "X10 Berlin"] {
        for f in ai_files(&root, map) {
            let path = omsi_cfg::resolve_path(&root, &f.replace('\\', "/"));
            let Ok(ty) = VehicleType::load_ai(&root, &path) else { continue };
            // (rail vehicles drive no street meeting places)
            if ty.def.is_rail() {
                continue;
            }
            let (len, hw, reach, pull) = body(&root, ty);
            let c = swept_class(hw, reach, pull);
            let k = SWEPT_CLASSES[c];
            println!("{map:12} {f:60} class {c}: length {len:.4} half width {hw:.4} reach {reach:.4} rear {pull:.4}");
            assert!(hw <= k.half_width + 1e-3 && reach <= k.reach + 1e-3 && pull <= k.rear_pull + 1e-3 && len <= k.length + 0.01, "{f} is not held by class {c}: {k:?}");
            members[c] += 1;
            for (m, v) in most[c].iter_mut().zip([len, hw, reach, pull]) {
                *m = m.max(v);
            }
        }
    }
    for c in 0..n {
        let k = SWEPT_CLASSES[c];
        assert!(members[c] > 0, "class {c} has no member");
        let have = [k.length, k.half_width, k.reach, k.rear_pull];
        for (d, (m, h)) in most[c].iter().zip(have).enumerate() {
            assert!(h >= m - 1e-3 && h - m < 0.01, "class {c} dimension {d}: largest member {m:.4}, class {h:.3} (the member's rounded up to the centimetre)");
        }
    }
}
