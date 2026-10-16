//! The network as the traffic sees it, written out for looking at with other tools: every
//! lane with its rules and links, every junction with its ways through and who is in
//! whose way (`network_json`), and the same on a map (`network_geojson`, in the map's own
//! metres: x east, y north).

use crate::junction::{Junction, Junctions};
use omsi_sim::traffic::{Lane, LaneKind, Network};
use serde_json::{json, Value};

fn kind_name(k: LaneKind) -> &'static str {
    match k {
        LaneKind::Street => "street",
        LaneKind::Sidewalk => "sidewalk",
        LaneKind::Rail => "rail",
        LaneKind::Air => "air",
    }
}

/// A float that may be infinite (no conflict on a way through): null then.
fn num(v: f32) -> Value {
    if v.is_finite() {
        json!(v)
    } else {
        Value::Null
    }
}

fn lane_props(net: &Network, i: usize, l: &Lane) -> Value {
    json!({
        "index": i,
        "key": l.key.map(|k| json!({"tile": [k.tile.0, k.tile.1], "id": k.id, "path": k.path})),
        "kind": kind_name(l.kind),
        "width": num(l.width),
        "offset": num(l.offset),
        "reversed": l.reversed,
        "one_way": l.one_way,
        "source": l.source,
        "name": l.name,
        "density": num(l.density),
        "group_density": l.group_density,
        "no_cars": l.no_cars,
        "rule_bus": l.rule_bus,
        "rule_trucks": l.rule_trucks,
        "priority": num(l.priority),
        "no_overtaking": l.no_overtaking,
        "speed_limit_kmh": num(l.speed_limit_kmh),
        "traffic_light": l.traffic_light,
        "turn": l.turn,
        "left": l.left,
        "right": l.right,
        "next": l.next,
        "prev": net.prev.get(i).cloned().unwrap_or_default(),
        "length": num(l.length()),
    })
}

fn junction_props(j: &Junction) -> Value {
    let movements: Vec<Value> = j
        .movements
        .iter()
        .map(|m| {
            json!({
                "lane": m.lane,
                "entries": m.entries,
                "exits": m.exits,
                "continues": m.continues,
                "light": m.light,
                "foes": m.foes.iter().map(|f| json!({
                    "movement": f.movement,
                    "kind": format!("{:?}", f.kind),
                    "mine": [num(f.mine.0), num(f.mine.1)],
                    "theirs": [num(f.theirs.0), num(f.theirs.1)],
                    "yields": f.yields,
                })).collect::<Vec<_>>(),
                "conflict_start": num(m.conflict_start),
                "conflict_end": num(m.conflict_end),
            })
        })
        .collect();
    let crossings: Vec<Value> = j
        .crossings
        .iter()
        .map(|c| {
            json!({
                "foot_lanes": c.foot_lanes,
                "light": c.light,
                "cuts": c.cuts.iter().map(|k| json!({"movement": k.movement, "at": num(k.at), "foot_lane": k.foot_lane, "foot_at": num(k.foot_at)})).collect::<Vec<_>>(),
            })
        })
        .collect();
    json!({
        "key": {"tile": [j.key.0 .0, j.key.0 .1], "id": j.key.1},
        "movements": movements,
        "crossings": crossings,
    })
}

/// The lanes and the junctions as one JSON document: `{"lanes": [...], "junctions": [...]}`.
pub fn network_json(net: &Network, js: &Junctions) -> String {
    let lanes: Vec<Value> = net.lanes.iter().enumerate().map(|(i, l)| lane_props(net, i, l)).collect();
    let junctions: Vec<Value> = js.nodes.iter().map(junction_props).collect();
    json!({"left_hand": net.left_hand, "lanes": lanes, "junctions": junctions}).to_string()
}

/// The lanes as GeoJSON LineStrings and the junctions as MultiPoints (the ends of their
/// ways through), with the properties of [`network_json`].
pub fn network_geojson(net: &Network, js: &Junctions) -> String {
    let mut features: Vec<Value> = Vec::with_capacity(net.lanes.len() + js.nodes.len());
    for (i, l) in net.lanes.iter().enumerate() {
        let coords: Vec<[f64; 3]> = l.points.iter().map(|p| [p.x, p.y, p.z]).collect();
        features.push(json!({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": lane_props(net, i, l),
        }));
    }
    for j in &js.nodes {
        let mut pts: Vec<[f64; 3]> = Vec::new();
        for m in &j.movements {
            if let Some(l) = net.lanes.get(m.lane) {
                for p in [l.start(), l.end()] {
                    let q = [p.x, p.y, p.z];
                    if !pts.contains(&q) {
                        pts.push(q);
                    }
                }
            }
        }
        let mut props = junction_props(j);
        props["feature"] = json!("junction");
        features.push(json!({
            "type": "Feature",
            "geometry": {"type": "MultiPoint", "coordinates": pts},
            "properties": props,
        }));
    }
    json!({"type": "FeatureCollection", "features": features}).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    use omsi_sim::traffic::{LaneBuilder, LaneKey};

    fn lane(from: (f64, f64), to: (f64, f64), id: i64, source: u32, path: u16) -> Lane {
        let mut l = LaneBuilder::polyline(vec![DVec3::new(from.0, from.1, 0.0), DVec3::new(to.0, to.1, 0.0)], LaneKind::Street, 3.5);
        l.source = source;
        l.key = Some(LaneKey { tile: (0, 0), id, path });
        l
    }

    #[test]
    fn the_export_parses_and_has_every_lane_and_junction() {
        let mut net = Network::default();
        net.lanes.push(lane((-200.0, 0.0), (-15.0, 0.0), 1, 1, 0));
        net.lanes.push(lane((-15.0, 0.0), (15.0, 0.0), 9, 2, 0));
        net.lanes.push(lane((15.0, 0.0), (300.0, 0.0), 2, 1, 0));
        net.lanes.push(lane((0.0, -200.0), (0.0, -15.0), 3, 1, 0));
        net.lanes.push(lane((0.0, -15.0), (0.0, 15.0), 9, 2, 1));
        net.lanes.push(lane((0.0, 15.0), (0.0, 300.0), 4, 1, 0));
        net.link(1.5);
        let js = Junctions::compile(&net);
        let v: Value = serde_json::from_str(&network_json(&net, &js)).expect("the JSON parses");
        assert_eq!(v["lanes"].as_array().map(|a| a.len()), Some(6));
        assert_eq!(v["junctions"].as_array().map(|a| a.len()), Some(1));
        assert_eq!(v["lanes"][0]["next"], json!([1]));
        assert_eq!(v["junctions"][0]["movements"].as_array().map(|a| a.len()), Some(2));
        let g: Value = serde_json::from_str(&network_geojson(&net, &js)).expect("the GeoJSON parses");
        let features = g["features"].as_array().expect("features");
        assert_eq!(features.len(), 7);
        assert_eq!(features[0]["geometry"]["type"], "LineString");
        assert_eq!(features[6]["geometry"]["type"], "MultiPoint");
    }
}
