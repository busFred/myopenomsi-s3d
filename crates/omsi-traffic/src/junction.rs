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
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FoeKind {
    /// The lanes cross each other.
    Cross,
    /// Both end in the same place: one runs into the other.
    Merge,
    /// The object says they block each other (`[blockpath]`) without a geometric meeting.
    Block,
}

/// A movement that is in the way of another one.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
    /// Stretches of this lane (from, to; m from its start) where its centre line comes within
    /// `MEET_DIST` of another way of the object that is no foe of it (no meeting place is
    /// compiled between the two: lanes converging whose starts lie under 3 m apart are no
    /// merge to `compute_conflicts`). A car held at the junction is not stopped in or past one
    /// short of the stretch it waits for (`plan`). (Pieces of one way, and forks from one
    /// point, are left out.)
    #[serde(default)]
    pub converge: Vec<(f32, f32)>,
}

/// Where a pedestrian crossing cuts a way through the junction.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Cut {
    /// Index into the junction's `movements`.
    pub movement: usize,
    /// Distance along the movement's lane.
    pub at: f32,
    /// The footpath lane that cuts it, and the distance along it.
    pub foot_lane: usize,
    pub foot_at: f32,
}

/// A pedestrian crossing of the junction: the footpath pieces across the carriageway
/// (the lit stub at the kerb, the pieces over each lane, a median) and where they cut the
/// ways through. The light is the one any of its pieces carries; the pieces without one
/// are governed by it too (BeijingCity lights a 1 m stub at the kerb and nothing else).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PedCrossing {
    pub foot_lanes: Vec<usize>,
    pub cuts: Vec<Cut>,
    /// (controller, light index), None for a zebra or an unlit crossing.
    pub light: Option<(usize, usize)>,
}

/// A crossing object as a node of the traffic network.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Junction {
    /// Tile and object id of the crossing object.
    pub key: ((i32, i32), i64),
    pub movements: Vec<Movement>,
    pub crossings: Vec<PedCrossing>,
}

impl Junction {
    /// The movement of lane `lane`, if it is one of this junction's.
    pub fn movement_of(&self, lane: usize) -> Option<usize> {
        self.movements.iter().position(|m| m.lane == lane)
    }
}

/// All junctions of a network, with lookups by lane.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Junctions {
    pub nodes: Vec<Junction>,
    /// Lane inside a junction → (junction, movement).
    pub of_lane: HashMap<usize, (usize, usize)>,
    /// Lane that leads into a junction → the junctions and movements it enters (a lane
    /// may fork into several movements: straight on and a turn).
    pub entries: HashMap<usize, Vec<(usize, usize)>>,
    /// Footpath lane of a pedestrian crossing → (junction, crossing).
    pub of_foot_lane: HashMap<usize, (usize, usize)>,
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
            self.nodes[idx].crossings = fresh.crossings;
        }
        self.compiled = net.lanes.len();
        self.rebuild_lookups();
    }

    fn rebuild_lookups(&mut self) {
        self.of_lane.clear();
        self.entries.clear();
        self.of_foot_lane.clear();
        for (ji, j) in self.nodes.iter().enumerate() {
            for (mi, m) in j.movements.iter().enumerate() {
                self.of_lane.insert(m.lane, (ji, mi));
                for &e in &m.entries {
                    self.entries.entry(e).or_default().push((ji, mi));
                }
            }
            for (ci, c) in j.crossings.iter().enumerate() {
                for &f in &c.foot_lanes {
                    self.of_foot_lane.insert(f, (ji, ci));
                }
            }
        }
    }

    /// The pedestrian crossing footpath lane `lane` is a piece of.
    pub fn crossing_of(&self, lane: usize) -> Option<&PedCrossing> {
        let &(j, c) = self.of_foot_lane.get(&lane)?;
        self.nodes.get(j)?.crossings.get(c)
    }

    /// The light that governs footpath lane `lane`: its own, else that of the crossing it
    /// is a piece of (the unlit pieces over the lanes beyond the lit stub at the kerb).
    pub fn crossing_light(&self, net: &Network, lane: usize) -> Option<(usize, usize)> {
        net.lanes.get(lane)?.traffic_light.or_else(|| self.crossing_of(lane)?.light)
    }

    fn build(net: &Network, key: ((i32, i32), i64), lanes: &[usize]) -> Junction {
        let inside = |l: usize| lanes.contains(&l);
        let mut movements: Vec<Movement> = lanes
            .iter()
            .map(|&lane| {
                let l = &net.lanes[lane];
                let entries: Vec<usize> = net.prev.get(lane).map(|p| p.iter().copied().filter(|&p| !inside(p)).collect()).unwrap_or_default();
                // (as movement indices: the lanes are the movements, in this order)
                let continues: Vec<usize> = l.next.iter().filter_map(|&n| lanes.iter().position(|&x| x == n)).collect();
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
                Movement { lane, kind: l.kind, entries, exits, continues, light: l.traffic_light, foes: Vec::new(), conflict_end: 0.0, conflict_start: f32::INFINITY, converge: Vec::new() }
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
        let converge = Self::converging(net, lanes);
        for (m, c) in movements.iter_mut().zip(converge) {
            m.converge = c;
        }
        let crossings = Self::crossings(net, &movements);
        Junction { key, movements, crossings }
    }

    /// The pieces of junction `j` a vehicle's way runs through, in order, each with how
    /// far along the whole way through its start lies: `way` is the vehicle's lanes ahead
    /// with the distance from its origin to each one's start. A way through built from
    /// several pieces (a lead-in, the turn, the way out: BeijingCity's objects chain them,
    /// and a lead-in forks into the straight way and the turn) meets its foes on the later
    /// pieces; the driver's claim stands for all the pieces it will take.
    pub fn chain_along(&self, j: usize, way: &[(usize, f32)]) -> Vec<(usize, f32)> {
        let mut out: Vec<(usize, f32)> = Vec::new();
        let mut first_d = 0.0;
        for &(l, d) in way {
            match self.of_lane.get(&l) {
                Some(&(jj, m)) if jj == j => {
                    if out.is_empty() {
                        first_d = d;
                    }
                    out.push((m, d - first_d));
                }
                _ => {
                    if !out.is_empty() {
                        break;
                    }
                }
            }
        }
        out
    }

    /// The foes of a way through `j` over the pieces `chain` (as from `chain_along`), with
    /// the meeting places at their distance along the whole way.
    pub fn foes_along(&self, j: usize, chain: &[(usize, f32)]) -> Vec<Foe> {
        self.foes_along_pieces(j, chain).into_iter().map(|(_, f)| f).collect()
    }

    /// [`Junctions::foes_along`], each foe with the piece of the chain it meets.
    pub fn foes_along_pieces(&self, j: usize, chain: &[(usize, f32)]) -> Vec<(usize, Foe)> {
        let node = &self.nodes[j];
        let mut out = Vec::new();
        for &(m, off) in chain {
            for f in &node.movements[m].foes {
                out.push((m, Foe { movement: f.movement, kind: f.kind, mine: (f.mine.0 + off, f.mine.1 + off), theirs: f.theirs, yields: f.yields }));
            }
        }
        out.sort_by(|a, b| a.1.mine.0.total_cmp(&b.1.mine.0));
        out
    }

    /// Two movements merging into one another that no rule of the table separates (neither
    /// gives way to the other): they take turns (`reservation`'s zipper).
    pub fn is_zipper(&self, j: usize, mine: usize, foe: &Foe) -> bool {
        foe.kind == FoeKind::Merge && !foe.yields && !self.nodes[j].movements[foe.movement].foes.iter().any(|f| f.movement == mine && f.yields)
    }

    /// Per lane of the object (in `lanes`' order): the stretches where its centre line comes
    /// within `MEET_DIST` of another street lane of the object with no meeting place compiled
    /// between them (`Movement::converge`; `traffic_cases converge` lists them).
    fn converging(net: &Network, lanes: &[usize]) -> Vec<Vec<(f32, f32)>> {
        let near = omsi_sim::traffic::MEET_DIST;
        let street: Vec<usize> = lanes.iter().copied().filter(|&l| net.lanes[l].kind == LaneKind::Street && net.lanes[l].points.len() >= 2).collect();
        let bbox = |l: usize| {
            let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
            for p in &net.lanes[l].points {
                x0 = x0.min(p.x);
                y0 = y0.min(p.y);
                x1 = x1.max(p.x);
                y1 = y1.max(p.y);
            }
            (x0 - near, y0 - near, x1 + near, y1 + near)
        };
        let boxes: Vec<(f64, f64, f64, f64)> = street.iter().map(|&l| bbox(l)).collect();
        let leads = |from: usize, to: usize| -> bool {
            let mut at = vec![from];
            for _ in 0..3 {
                at = at.iter().flat_map(|&l| net.lanes[l].next.iter().copied()).collect();
                if at.contains(&to) {
                    return true;
                }
            }
            false
        };
        let mut out: Vec<Vec<(f32, f32)>> = vec![Vec::new(); lanes.len()];
        for (x, &a) in street.iter().enumerate() {
            let la = &net.lanes[a];
            for (y, &b) in street.iter().enumerate() {
                if x == y {
                    continue;
                }
                let (ba, bb) = (boxes[x], boxes[y]);
                if ba.2 < bb.0 || bb.2 < ba.0 || ba.3 < bb.1 || bb.3 < ba.1 {
                    continue;
                }
                let lb = &net.lanes[b];
                if la.key.map(|k| k.path) == lb.key.map(|k| k.path) || net.crossings.get(a).is_some_and(|c| c.iter().any(|c| c.other == b)) {
                    continue;
                }
                if (la.start() - lb.start()).truncate().length() <= 0.5 || leads(a, b) || leads(b, a) {
                    continue;
                }
                let (mut s0, mut s1) = (f32::INFINITY, f32::NEG_INFINITY);
                let mut s = 0.0f32;
                while s <= la.length() {
                    if lb.nearest_point(la.at(s).0).is_some_and(|(_, d)| d < near) {
                        s0 = s0.min(s);
                        s1 = s1.max(s);
                    }
                    s += 0.5;
                }
                if s0.is_finite() {
                    let k = lanes.iter().position(|&l| l == a).unwrap();
                    out[k].push((s0, s1));
                }
            }
        }
        out
    }

    /// The pedestrian crossings of the object: every footpath lane that cuts a way through
    /// it, chained with the short pieces linked before and after it (the lit stub at the
    /// kerb, the piece over the next lane, a median piece), one crossing per chain.
    fn crossings(net: &Network, movements: &[Movement]) -> Vec<PedCrossing> {
        let mut cuts: Vec<Cut> = Vec::new();
        for (mi, m) in movements.iter().enumerate() {
            if m.kind != LaneKind::Street {
                continue;
            }
            for &(w, sa, sw) in net.walks.get(m.lane).map(|v| v.as_slice()).unwrap_or(&[]) {
                cuts.push(Cut { movement: mi, at: sa, foot_lane: w, foot_at: sw });
            }
        }
        // The crossings are the connected pieces among the footpath lanes that cut a
        // street, the short pieces (a median, the pieces between two lanes) and the lit
        // stubs, joined where one leads into the next. A long piece that cuts no street is
        // a pavement along the kerb and ends the chain, so two crossings of one object stay
        // apart; a lane belongs to one crossing only. (Built from each cutting lane with a
        // cap of 16 pieces, a 33 m crossing of 2 m pieces split in two, the halves
        // overlapped, and a lit stub beside a longer piece was left out.)
        let cutting = |l: usize| cuts.iter().any(|c| c.foot_lane == l);
        let member = |l: usize| -> bool {
            let lane = &net.lanes[l];
            lane.kind == LaneKind::Sidewalk && (cutting(l) || lane.length() <= 6.0 || lane.traffic_light.is_some())
        };
        let mut out: Vec<PedCrossing> = Vec::new();
        let mut taken: Vec<usize> = Vec::new();
        for c in &cuts {
            if taken.contains(&c.foot_lane) {
                continue;
            }
            let mut chain = vec![c.foot_lane];
            taken.push(c.foot_lane);
            let mut k = 0;
            while k < chain.len() && chain.len() < 64 {
                let l = chain[k];
                let lane = &net.lanes[l];
                let around: Vec<usize> = lane.next.iter().copied().chain(net.prev.get(l).map(|v| v.iter().copied()).into_iter().flatten()).collect();
                for o in around {
                    if !taken.contains(&o) && member(o) {
                        chain.push(o);
                        taken.push(o);
                    }
                }
                k += 1;
            }
            let light = chain.iter().find_map(|&l| net.lanes[l].traffic_light);
            let mine: Vec<Cut> = cuts.iter().copied().filter(|x| chain.contains(&x.foot_lane)).collect();
            out.push(PedCrossing { foot_lanes: chain, cuts: mine, light });
        }
        out
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
