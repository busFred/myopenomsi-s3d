//! Junctions compiled once from the network (SUMO's `NBNode` and `NBRequest`, in spirit).
//!
//! In an OMSI map a junction is a crossing scenery object: all the street and rail
//! `[path]`s of one object are the ways through it (the *movements*), and the network
//! already knows where any two of them meet (`Network::crossings`). This module turns
//! that into a table: for every movement, the list of foe movements with the stretch of
//! each lane where the two are in each other's way, and who gives way to whom under the
//! map's rules. At run time a driver looks its movement up and reads the table; no
//! geometry is computed per frame.

use hashbrown::HashMap;
use omsi_sim::traffic::{LaneKind, Network};

/// How two movements of a junction meet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoeKind {
    /// The lanes cross each other.
    Cross,
    /// Both end in the same place: one runs into the other.
    Merge,
    /// The object says they block each other (`[blockpath]`) without a geometric meeting.
    Block,
}

/// A movement that is in the way of another one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Foe {
    /// Index into the junction's `movements`.
    pub movement: usize,
    pub kind: FoeKind,
    /// The stretch along *my* lane where the two touch: (from, to) in metres from the
    /// lane's start.
    pub mine: (f32, f32),
    /// The same along the foe's lane.
    pub theirs: (f32, f32),
    /// I give way to this foe (by `[rule] priority`, right before left, a train first).
    pub yields: bool,
}

/// One way through a junction: one of its lanes.
#[derive(Debug, Clone, PartialEq)]
pub struct Movement {
    /// The lane in the network.
    pub lane: usize,
    pub kind: LaneKind,
    /// Lanes that lead into this movement from outside the junction (where the queue
    /// stands; the stop line is their end).
    pub entries: Vec<usize>,
    /// The first lanes outside the junction that this movement leads to, following
    /// movement after movement inside the object. A car may enter only when there is room
    /// on one of them ("keep the junction clear").
    pub exits: Vec<usize>,
    /// Movements inside the object this one continues into (a long way through built
    /// from several `[path]`s).
    pub continues: Vec<usize>,
    /// The light that controls entry (object instance, light index), if any.
    pub light: Option<(usize, usize)>,
    pub foes: Vec<Foe>,
    /// The far end of the last stretch where any foe touches this lane (m from its start):
    /// past it the car has left every conflict of the junction.
    pub conflict_end: f32,
    /// The near end of the first conflict (m): before it the car is inside the junction
    /// but in nobody's way yet.
    pub conflict_start: f32,
}

/// A crossing object as a node of the traffic network.
#[derive(Debug, Clone, PartialEq)]
pub struct Junction {
    /// Tile and object id of the crossing object.
    pub key: ((i32, i32), i64),
    pub movements: Vec<Movement>,
}

impl Junction {
    /// The movement of lane `lane`, if it is one of this junction's.
    pub fn movement_of(&self, lane: usize) -> Option<usize> {
        self.movements.iter().position(|m| m.lane == lane)
    }
}

/// All junctions of a network, with lookups by lane.
#[derive(Debug, Clone, Default)]
pub struct Junctions {
    pub nodes: Vec<Junction>,
    /// Lane inside a junction → (junction, movement).
    pub of_lane: HashMap<usize, (usize, usize)>,
    /// Lane that leads into a junction → the junctions and movements it enters (a lane
    /// may fork into several movements: straight on and a turn).
    pub entries: HashMap<usize, Vec<(usize, usize)>>,
    /// How many lanes of the network were compiled; `extend` carries on from there.
    compiled: usize,
}

impl Junctions {
    /// Compile every crossing object of the network.
    pub fn compile(net: &Network) -> Junctions {
        let mut j = Junctions::default();
        j.extend(net, 0);
        j
    }

    /// Compile the crossing objects whose lanes start at index `first` (lanes come in
    /// whole objects when a tile loads). Lanes before `first` are already compiled;
    /// their entries and exits into the new lanes are patched up.
    pub fn extend(&mut self, net: &Network, first: usize) {
        let first = first.min(self.compiled);
        // group the new junction lanes by object
        let mut by_object: HashMap<((i32, i32), i64), Vec<usize>> = HashMap::new();
        for (i, l) in net.lanes.iter().enumerate().skip(first) {
            if l.source != 2 || !matches!(l.kind, LaneKind::Street | LaneKind::Rail) {
                continue;
            }
            let Some(k) = l.key else { continue };
            by_object.entry((k.tile, k.id)).or_default().push(i);
        }
        // (an object whose lanes were extended with reverse twins: compiled again whole)
        let mut keys: Vec<_> = by_object.keys().copied().collect();
        keys.sort_unstable();
        for &key in &keys {
            let mut lanes = by_object.remove(&key).unwrap_or_default();
            if let Some(old) = self.nodes.iter().position(|n| n.key == key) {
                for m in &self.nodes[old].movements {
                    if !lanes.contains(&m.lane) {
                        lanes.push(m.lane);
                    }
                }
                lanes.sort_unstable();
                let node = Self::build(net, key, &lanes);
                self.nodes[old] = node;
            } else {
                lanes.sort_unstable();
                self.nodes.push(Self::build(net, key, &lanes));
            }
        }
        // the old junctions' ways in and out may lie on the new lanes
        let old_keys: Vec<((i32, i32), i64)> = self.nodes.iter().filter(|n| !keys.contains(&n.key)).map(|n| n.key).collect();
        for key in old_keys {
            let Some(idx) = self.nodes.iter().position(|n| n.key == key) else { continue };
            let lanes: Vec<usize> = self.nodes[idx].movements.iter().map(|m| m.lane).collect();
            let fresh = Self::build(net, key, &lanes);
            for (m, f) in self.nodes[idx].movements.iter_mut().zip(fresh.movements) {
                m.entries = f.entries;
                m.exits = f.exits;
                m.continues = f.continues;
            }
        }
        self.compiled = net.lanes.len();
        self.rebuild_lookups();
    }

    fn rebuild_lookups(&mut self) {
        self.of_lane.clear();
        self.entries.clear();
        for (ji, j) in self.nodes.iter().enumerate() {
            for (mi, m) in j.movements.iter().enumerate() {
                self.of_lane.insert(m.lane, (ji, mi));
                for &e in &m.entries {
                    self.entries.entry(e).or_default().push((ji, mi));
                }
            }
        }
    }

    fn build(net: &Network, key: ((i32, i32), i64), lanes: &[usize]) -> Junction {
        let inside = |l: usize| lanes.contains(&l);
        let mut movements: Vec<Movement> = lanes
            .iter()
            .map(|&lane| {
                let l = &net.lanes[lane];
                let entries: Vec<usize> = net.prev.get(lane).map(|p| p.iter().copied().filter(|&p| !inside(p)).collect()).unwrap_or_default();
                let continues: Vec<usize> = l.next.iter().copied().filter(|&n| inside(n)).collect();
                // exits: the first lanes outside, through at most a few inner pieces
                let mut exits: Vec<usize> = Vec::new();
                let mut frontier: Vec<usize> = vec![lane];
                let mut seen: Vec<usize> = vec![lane];
                for _ in 0..6 {
                    let mut nxt = Vec::new();
                    for f in frontier {
                        for &n in &net.lanes[f].next {
                            if inside(n) {
                                if !seen.contains(&n) {
                                    seen.push(n);
                                    nxt.push(n);
                                }
                            } else if !exits.contains(&n) {
                                exits.push(n);
                            }
                        }
                    }
                    if nxt.is_empty() {
                        break;
                    }
                    frontier = nxt;
                }
                Movement { lane, kind: l.kind, entries, exits, continues, light: l.traffic_light, foes: Vec::new(), conflict_end: 0.0, conflict_start: f32::INFINITY }
            })
            .collect();
        // foes from the network's crossings, both ways
        for mi in 0..movements.len() {
            let lane = movements[mi].lane;
            let mut foes = Vec::new();
            for c in net.crossings.get(lane).map(|v| v.as_slice()).unwrap_or(&[]) {
                let Some(fm) = lanes.iter().position(|&l| l == c.other) else { continue };
                let kind = if c.merge {
                    FoeKind::Merge
                } else if c.before >= net.lanes[lane].length() * 0.5 - 0.01 && c.at == net.lanes[lane].length() * 0.5 {
                    // a `[blockpath]` pair is recorded as the whole of both lanes
                    FoeKind::Block
                } else {
                    FoeKind::Cross
                };
                let mine = ((c.at - c.before).max(0.0), (c.at + c.after).min(net.lanes[lane].length()));
                let theirs = ((c.other_at - c.other_before).max(0.0), (c.other_at + c.other_after).min(net.lanes[c.other].length()));
                foes.push(Foe { movement: fm, kind, mine, theirs, yields: net.must_yield(lane, c.other) });
            }
            // (a movement that continues into another inner piece is in the way of
            // everything that piece is in the way of, as far as the queue behind the stop
            // line is concerned: the continuation's foes are reached later along the way)
            foes.sort_by(|a, b| a.mine.0.total_cmp(&b.mine.0));
            let (start, end) = foes.iter().fold((f32::INFINITY, 0.0f32), |(s, e), f| (s.min(f.mine.0), e.max(f.mine.1)));
            movements[mi].foes = foes;
            movements[mi].conflict_start = start;
            movements[mi].conflict_end = end;
        }
        Junction { key, movements }
    }

    /// The junction and movement that lane `lane` is part of.
    pub fn movement(&self, lane: usize) -> Option<(&Junction, &Movement)> {
        let &(j, m) = self.of_lane.get(&lane)?;
        Some((&self.nodes[j], &self.nodes[j].movements[m]))
    }

    /// Is lane `lane` inside a junction?
    pub fn is_inside(&self, lane: usize) -> bool {
        self.of_lane.contains_key(&lane)
    }

    /// The whole way through the junction starting at movement `(j, m)`: its lane and the
    /// inner pieces it continues into (first continuation each time), with the distance
    /// from the start of the first to the end of the last.
    pub fn way_through(&self, net: &Network, j: usize, m: usize) -> (Vec<usize>, f32) {
        let node = &self.nodes[j];
        let mut out = vec![node.movements[m].lane];
        let mut len = net.lanes[node.movements[m].lane].length();
        let mut cur = m;
        for _ in 0..6 {
            let Some(&n) = node.movements[cur].continues.first() else { break };
            out.push(node.movements[n].lane);
            len += net.lanes[node.movements[n].lane].length();
            cur = n;
        }
        (out, len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    use omsi_sim::traffic::{LaneBuilder, LaneKey};

    /// A lane of a crossing object (`source` 2) with a map key.
    fn piece(from: (f64, f64), to: (f64, f64), path: u16, priority: f32) -> omsi_sim::traffic::Lane {
        let mut l = LaneBuilder::polyline(vec![DVec3::new(from.0, from.1, 0.0), DVec3::new(to.0, to.1, 0.0)], LaneKind::Street, 3.0);
        l.source = 2;
        l.key = Some(LaneKey { tile: (0, 0), id: 7, path });
        l.priority = priority;
        l
    }

    /// An approach lane of a spline (`source` 1).
    fn road(from: (f64, f64), to: (f64, f64), id: i64) -> omsi_sim::traffic::Lane {
        let mut l = LaneBuilder::polyline(vec![DVec3::new(from.0, from.1, 0.0), DVec3::new(to.0, to.1, 0.0)], LaneKind::Street, 3.0);
        l.source = 1;
        l.key = Some(LaneKey { tile: (0, 0), id, path: 0 });
        l
    }

    /// Four-way crossing: main road along x (priority 192), side road along y (64).
    fn four_way() -> Network {
        let mut net = Network::default();
        net.lanes.push(road((-60.0, 0.0), (-10.0, 0.0), 1)); // 0: west approach
        net.lanes.push(piece((-10.0, 0.0), (10.0, 0.0), 0, 192.0)); // 1: main through
        net.lanes.push(road((10.0, 0.0), (60.0, 0.0), 2)); // 2: east exit
        net.lanes.push(road((0.0, -60.0), (0.0, -10.0), 3)); // 3: south approach
        net.lanes.push(piece((0.0, -10.0), (0.0, 10.0), 1, 64.0)); // 4: side through
        net.lanes.push(road((0.0, 10.0), (0.0, 60.0), 4)); // 5: north exit
        net.link(1.5);
        net.compute_conflicts();
        net
    }

    #[test]
    fn a_crossing_object_becomes_one_junction_with_a_foe_table() {
        let net = four_way();
        let js = Junctions::compile(&net);
        assert_eq!(js.nodes.len(), 1);
        let j = &js.nodes[0];
        assert_eq!(j.movements.len(), 2);
        let main = &j.movements[j.movement_of(1).unwrap()];
        let side = &j.movements[j.movement_of(4).unwrap()];
        assert_eq!(main.entries, vec![0]);
        assert_eq!(main.exits, vec![2]);
        assert_eq!(side.entries, vec![3]);
        assert_eq!(side.exits, vec![5]);
        assert_eq!(main.foes.len(), 1);
        assert_eq!(side.foes.len(), 1);
        assert!(!main.foes[0].yields, "the main road goes first");
        assert!(side.foes[0].yields, "the side road gives way");
        assert_eq!(main.foes[0].kind, FoeKind::Cross);
        // the meeting place is around the middle of both pieces
        assert!(main.foes[0].mine.0 < 10.0 && main.foes[0].mine.1 > 10.0);
        assert!(side.foes[0].theirs.0 < 10.0 && side.foes[0].theirs.1 > 10.0);
        assert_eq!(js.entries.get(&0).unwrap(), &vec![(0, j.movement_of(1).unwrap())]);
        assert!(js.is_inside(4) && !js.is_inside(3));
    }

    #[test]
    fn extending_the_network_adds_junctions_and_keeps_the_old_ones() {
        let mut net = four_way();
        let mut js = Junctions::compile(&net);
        let first = net.lanes.len();
        let mut far = Vec::new();
        let mut p = piece((200.0, 0.0), (220.0, 0.0), 0, 128.0);
        p.key = Some(LaneKey { tile: (1, 0), id: 9, path: 0 });
        far.push(p);
        let mut q = piece((210.0, -10.0), (210.0, 10.0), 1, 128.0);
        q.key = Some(LaneKey { tile: (1, 0), id: 9, path: 1 });
        far.push(q);
        net.extend(far, 1.5);
        js.extend(&net, first);
        assert_eq!(js.nodes.len(), 2);
        assert!(js.is_inside(1) && js.is_inside(first));
        // right before left between equal priorities: the one from the right goes first
        let j = &js.nodes[1];
        let east = &j.movements[j.movement_of(first).unwrap()];
        let north = &j.movements[j.movement_of(first + 1).unwrap()];
        assert_ne!(east.foes[0].yields, north.foes[0].yields);
    }
}
