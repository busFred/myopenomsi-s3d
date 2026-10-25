//! Junctions compiled once from the network (SUMO's `NBNode` and `NBRequest`, in spirit).
//!
//! In an OMSI map a junction is a crossing scenery object: all the street and rail
//! `[path]`s of one object are the ways through it (the *movements*), and the network
//! already knows where any two of them meet (`Network::crossings`). Lanes that meet outside
//! any one object - a join of spline lanes, or two lanes drawn across each other with no object
//! (`Network::compile_lane_crossings`) - are movements of the object one of them belongs to, or
//! of a junction of their own (`JOIN_TILE`), as SUMO has a node wherever two connections meet.
//! This module turns
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
    /// A merge of two lanes running side by side, not a joint (`Crossing::side`).
    #[serde(default)]
    pub side: bool,
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
    /// Per stretch of `converge`, the movement it converges with.
    #[serde(default)]
    pub converge_with: Vec<usize>,
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
/// ways through. The light is the first any of its pieces carries; each piece without one
/// is governed by the lit stub before it (BeijingCity lights a 1 m stub at the kerb and
/// nothing else, and joins the four sides of a junction into one chain: Taijichang j28's
/// 56 pieces carry lights 18.0 to 18.3).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PedCrossing {
    pub foot_lanes: Vec<usize>,
    pub cuts: Vec<Cut>,
    /// (controller, light index), None for a zebra or an unlit crossing.
    pub light: Option<(usize, usize)>,
    /// Per piece of `foot_lanes`: the light governing it (`piece_lights`). (Empty in a
    /// recording from before them: then `light` governs every piece.)
    #[serde(default)]
    pub lights: Vec<Option<(usize, usize)>>,
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

/// The tile of a junction made of a lane join of spline lanes alone (`Junctions::extend`):
/// its key is (this tile, the join's first lane).
pub const JOIN_TILE: i32 = i32::MIN;

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
    /// Compile every crossing object of the network, and the places where lanes meet outside
    /// any one object (`extend`).
    pub fn compile(net: &Network) -> Junctions {
        let mut j = Junctions::default();
        j.extend(net, 0);
        j
    }

    /// Compile the junctions again with the lanes added from index `first` (lanes come in
    /// whole objects when a tile loads): every crossing object with all its lanes (an object
    /// extended with reverse twins is compiled whole), and every lane join
    /// (`Network::compile_joins`) or lane crossing outside the objects
    /// (`Network::compile_lane_crossings`, `Crossing::outside`) whose two lanes are not of one
    /// object: a spline lane joining or crossing an object's way goes into that object as one
    /// of its movements, and spline lanes alone make a junction of their own (`JOIN_TILE`), so
    /// that every such meeting place is in one junction table (`foe_between`). The swept path
    /// of an object's piece over a lane leading into it (`Network::compile_swept`) groups
    /// nothing: that lane stays an approach, where the queue stands. Junctions keep their index (a new
    /// one comes at the end), so what a vehicle remembers of one (`ease_for`)
    /// stays right.
    pub fn extend(&mut self, net: &Network, first: usize) {
        let _ = first;
        let object = |l: usize| -> Option<((i32, i32), i64)> {
            let lane = net.lanes.get(l)?;
            if lane.source != 2 || !matches!(lane.kind, LaneKind::Street | LaneKind::Rail) {
                return None;
            }
            lane.key.map(|k| (k.tile, k.id))
        };
        let mut lanes_of: HashMap<((i32, i32), i64), Vec<usize>> = HashMap::new();
        let mut keys: Vec<((i32, i32), i64)> = Vec::new();
        for i in 0..net.lanes.len() {
            if let Some(k) = object(i) {
                let v = lanes_of.entry(k).or_insert_with(|| {
                    keys.push(k);
                    Vec::new()
                });
                v.push(i);
            }
        }
        // the lane joins not inside one object, grouped (a lane joining two others is one
        // place); a group with an object's lane goes into that object
        let mut parent: HashMap<usize, usize> = HashMap::new();
        // (the lanes each join runs into)
        let mut into: Vec<(usize, usize)> = Vec::new();
        fn find(parent: &mut HashMap<usize, usize>, x: usize) -> usize {
            let mut r = x;
            while let Some(&p) = parent.get(&r) {
                if p == r {
                    break;
                }
                r = p;
            }
            parent.insert(x, r);
            r
        }
        for a in 0..net.lanes.len().min(net.crossings.len()) {
            for c in &net.crossings[a] {
                let b = c.other;
                // (a meeting place between lanes of no one object: a lane join, or lanes
                // converging side by side before one, `Network::compile_joins`, or anywhere,
                // `Network::compile_outside_converging`; or two lanes crossing outside any one
                // object, `Network::compile_lane_crossings`. The swept
                // path of an object's piece over a lane leading into it, `compile_swept`, leaves
                // that lane an approach: its queue stands there.)
                if b <= a || b >= net.lanes.len() {
                    continue;
                }
                if object(a).is_some() && object(a) == object(b) {
                    continue;
                }
                if !c.merge && !c.outside {
                    continue;
                }
                let shared: Vec<usize> = net.lanes[a].next.iter().copied().filter(|n| net.lanes[b].next.contains(n)).collect();
                let ways_on: Vec<usize> = if shared.is_empty() { net.lanes[a].next.iter().chain(&net.lanes[b].next).copied().collect() } else { shared };
                for n in ways_on {
                    into.push((a, n));
                }
                parent.entry(a).or_insert(a);
                parent.entry(b).or_insert(b);
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                if ra != rb {
                    parent.insert(ra.max(rb), ra.min(rb));
                }
            }
        }
        let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
        let mut join_keys: Vec<((i32, i32), i64)> = Vec::new();
        let members: Vec<usize> = parent.keys().copied().collect();
        for l in members {
            let r = find(&mut parent, l);
            groups.entry(r).or_default().push(l);
        }
        let mut into_of: HashMap<usize, Vec<usize>> = HashMap::new();
        for (a, n) in into {
            let r = find(&mut parent, a);
            into_of.entry(r).or_default().push(n);
        }
        for (root, mut lanes) in groups {
            lanes.sort_unstable();
            // (an object among the lanes, else among the lanes they run into: a spline lane
            // joining an object's way in, as part of that object, so the way through it is one
            // junction - a junction of its own just before the object hid the object's foes
            // until the car was on its first piece, viz2 case11 #40894)
            let objects: Vec<((i32, i32), i64)> = {
                let mut o: Vec<_> = lanes.iter().filter_map(|&l| object(l)).collect();
                if o.is_empty() {
                    o = into_of.get(&root).map(|v| v.iter().filter_map(|&n| object(n)).collect()).unwrap_or_default();
                }
                o.sort_unstable();
                o.dedup();
                o
            };
            let outside: Vec<usize> = lanes.iter().copied().filter(|&l| object(l).is_none()).collect();
            match objects.first() {
                Some(&k) => lanes_of.entry(k).or_default().extend(outside),
                None => {
                    let k = ((JOIN_TILE, JOIN_TILE), root as i64);
                    join_keys.push(k);
                    lanes_of.insert(k, outside);
                }
            }
        }
        // in place, by key; new ones at the end
        let mut placed: Vec<((i32, i32), i64)> = Vec::new();
        for idx in 0..self.nodes.len() {
            let key = self.nodes[idx].key;
            if let Some(lanes) = lanes_of.get_mut(&key) {
                lanes.sort_unstable();
                lanes.dedup();
                self.nodes[idx] = Self::build(net, key, lanes);
                placed.push(key);
            }
        }
        keys.sort_unstable();
        // (the objects first, as compiled before, then the joins of spline lanes)
        keys.extend(join_keys);
        for key in keys {
            if placed.contains(&key) {
                continue;
            }
            if let Some(lanes) = lanes_of.get_mut(&key) {
                lanes.sort_unstable();
                lanes.dedup();
                self.nodes.push(Self::build(net, key, lanes));
                placed.push(key);
            }
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
        if let Some(own) = net.lanes.get(lane)?.traffic_light {
            return Some(own);
        }
        let &(j, c) = self.of_foot_lane.get(&lane)?;
        let c = self.nodes.get(j)?.crossings.get(c)?;
        match c.foot_lanes.iter().position(|&f| f == lane).and_then(|k| c.lights.get(k)) {
            Some(&l) => l,
            None => c.light,
        }
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
                Movement { lane, kind: l.kind, entries, exits, continues, light: l.traffic_light, foes: Vec::new(), conflict_end: 0.0, conflict_start: f32::INFINITY, converge: Vec::new(), converge_with: Vec::new() }
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
                foes.push(Foe { movement: fm, kind, mine, theirs, yields: net.must_yield(lane, c.other), side: c.side });
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
            m.converge = c.iter().map(|&(a, b, _)| (a, b)).collect();
            m.converge_with = c.iter().map(|&(_, _, with)| with).collect();
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
                out.push((m, Foe { movement: f.movement, kind: f.kind, mine: (f.mine.0 + off, f.mine.1 + off), theirs: f.theirs, yields: f.yields, side: f.side }));
            }
        }
        out.sort_by(|a, b| a.1.mine.0.total_cmp(&b.1.mine.0));
        out
    }

    /// Two movements merging into one another that no rule of the table separates (neither
    /// gives way to the other): SUMO's zipper node. (`reservation`'s zipper speed takes every
    /// merge that crosses nothing, the table ordering the pair where it says who yields.)
    pub fn is_zipper(&self, j: usize, mine: usize, foe: &Foe) -> bool {
        foe.kind == FoeKind::Merge && !foe.yields && !self.nodes[j].movements[foe.movement].foes.iter().any(|f| f.movement == mine && f.yields)
    }

    /// Per lane of the object (in `lanes`' order): the stretches where its centre line comes
    /// within `MEET_DIST` of another street lane of the object with no meeting place compiled
    /// between them, and that lane's movement (`Movement::converge`, `converge_with`;
    /// `traffic_cases converge` lists them).
    fn converging(net: &Network, lanes: &[usize]) -> Vec<Vec<(f32, f32, usize)>> {
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
        let mut out: Vec<Vec<(f32, f32, usize)>> = vec![Vec::new(); lanes.len()];
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
                    let with = lanes.iter().position(|&l| l == b).unwrap();
                    out[k].push((s0, s1, with));
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
            let lights = Self::piece_lights(net, &chain);
            let mine: Vec<Cut> = cuts.iter().copied().filter(|x| chain.contains(&x.foot_lane)).collect();
            out.push(PedCrossing { foot_lanes: chain, cuts: mine, light, lights });
        }
        out
    }

    /// The light governing each piece of a crossing chain: its own, else that of the
    /// nearest lit piece before it along the chain (walking back towards the kerb its walk
    /// starts at). A piece with no lit piece before it has the chain's light when the
    /// chain has one light group, and none when it has several: it lies before the stubs
    /// (a corner piece at the kerb), whose own lights govern what follows them. (The
    /// chain's first light for every piece put 18.1 on Taijichang's 18.2 side, case09;
    /// 7 of BeijingCity's 59 lit crossings carry several groups.)
    fn piece_lights(net: &Network, chain: &[usize]) -> Vec<Option<(usize, usize)>> {
        let mut groups: Vec<(usize, usize)> = chain.iter().filter_map(|&l| net.lanes[l].traffic_light).collect();
        groups.sort_unstable();
        groups.dedup();
        let single = (groups.len() == 1).then(|| groups[0]);
        chain
            .iter()
            .map(|&l| {
                if let Some(own) = net.lanes[l].traffic_light {
                    return Some(own);
                }
                // (shortest walk back over the chain's own pieces, metres)
                let mut best: Option<(f32, (usize, usize))> = None;
                let mut frontier: Vec<(usize, f32)> = vec![(l, 0.0)];
                let mut reached: Vec<(usize, f32)> = vec![(l, 0.0)];
                while let Some((at, d)) = frontier.pop() {
                    for &p in net.prev.get(at).map(|v| v.as_slice()).unwrap_or(&[]) {
                        let dp = d + net.lanes[p].length();
                        if !chain.contains(&p) || reached.iter().any(|&(r, rd)| r == p && rd <= dp) {
                            continue;
                        }
                        reached.push((p, dp));
                        match net.lanes[p].traffic_light {
                            Some(g) => {
                                if best.is_none_or(|(bd, _)| dp < bd) {
                                    best = Some((dp, g));
                                }
                            }
                            None => frontier.push((p, dp)),
                        }
                    }
                }
                best.map(|b| b.1).or(single)
            })
            .collect()
    }

    /// The junction and movement that lane `lane` is part of.
    pub fn movement(&self, lane: usize) -> Option<(&Junction, &Movement)> {
        let &(j, m) = self.of_lane.get(&lane)?;
        Some((&self.nodes[j], &self.nodes[j].movements[m]))
    }

    /// The meeting place of lanes `a` and `b`, two pieces of one junction: the
    /// foe of `a`'s movement that is `b`'s (None outside a junction or with no meeting place).
    pub fn foe_between(&self, a: usize, b: usize) -> Option<&Foe> {
        let (&(ja, ma), &(jb, mb)) = (self.of_lane.get(&a)?, self.of_lane.get(&b)?);
        if ja != jb {
            return None;
        }
        self.nodes[ja].movements[ma].foes.iter().find(|f| f.movement == mb)
    }

    /// Is lane `lane` inside a junction?
    pub fn is_inside(&self, lane: usize) -> bool {
        self.of_lane.contains_key(&lane)
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

    /// Two spline roads drawn across each other with no crossing object (viz case01: AndingR_noside
    /// 191/3-4 across 723743/3): the crossing is the junction, as in SUMO every crossing of two
    /// connections is a foe of its node. The pair gets a crossing meeting place in one junction
    /// table, with the map's priority where the paths carry one and right before left between
    /// equals; a road passing over the other on a bridge gets none.
    #[test]
    fn two_roads_crossing_with_no_object_meet_in_one_junction() {
        let build = |main: Option<f32>, dz: f64| {
            let mut net = Network::default();
            let mut ew = road((-60.0, 0.0), (60.0, 0.0), 1); // 0: west to east
            let mut sn = road((0.0, -60.0), (0.0, 60.0), 2); // 1: south to north
            if let Some(p) = main {
                ew.priority = p;
                sn.priority = 64.0;
            }
            for q in &mut sn.points {
                q.z = dz;
            }
            net.lanes.push(ew);
            net.lanes.push(sn);
            net.link(1.5);
            net.compute_conflicts();
            net
        };
        // equal priorities: right before left - the eastbound car has the northbound one coming
        // from its right and gives way
        let net = build(None, 0.0);
        let c = net.crossings[0].iter().find(|c| c.other == 1).expect("a meeting place on 0");
        assert!(c.outside && net.crossings[0].len() == 1 && net.crossings[1].len() == 1);
        assert!(!c.merge && (c.at - 60.0).abs() < 0.1 && (c.other_at - 60.0).abs() < 0.1);
        assert!(c.before > 0.5 && c.after > 0.5, "the stretch where the two bodies touch");
        let js = Junctions::compile(&net);
        assert_eq!(js.nodes.len(), 1);
        let j = &js.nodes[0];
        assert_eq!(j.key.0, (JOIN_TILE, JOIN_TILE));
        let (ew, sn) = (&j.movements[j.movement_of(0).unwrap()], &j.movements[j.movement_of(1).unwrap()]);
        assert_eq!((ew.foes.len(), sn.foes.len()), (1, 1));
        assert_eq!((ew.foes[0].kind, sn.foes[0].kind), (FoeKind::Cross, FoeKind::Cross));
        assert!(ew.foes[0].yields && !sn.foes[0].yields, "right before left");
        assert!(ew.foes[0].mine.0 < 60.0 && ew.foes[0].mine.1 > 60.0);
        // the map's priority on the paths decides first
        let js = Junctions::compile(&build(Some(192.0), 0.0));
        let j = &js.nodes[0];
        assert!(!j.movements[j.movement_of(0).unwrap()].foes[0].yields, "the main road goes first");
        assert!(j.movements[j.movement_of(1).unwrap()].foes[0].yields);
        // within the same-level band still a crossing; a deck above it is a bridge
        assert_eq!(build(None, omsi_sim::traffic::LEVEL_DZ - 0.5).crossings[0].len(), 1);
        let bridge = build(None, 6.0);
        assert!(bridge.crossings[0].is_empty());
        assert!(Junctions::compile(&bridge).nodes.is_empty());
    }

    /// Two three-lane spline roads crossing with no object, one lane of the east-west road past a
    /// bus bay's gap and so linked to no neighbour (viz case01 AndingR_noside 723743/9), another
    /// of three times its neighbours' density (191/6): every lane of one road meets every lane of
    /// the other with the same right of way - the road's lanes count for the road class
    /// (`Network::road_class`), as SUMO ranks edges, not lanes - here right before left.
    #[test]
    fn every_lane_of_two_crossing_roads_has_one_right_of_way() {
        let mut net = Network::default();
        let lane = |from: (f64, f64), to: (f64, f64), id: i64, path: u16| {
            let mut l = road(from, to, id);
            l.key = Some(LaneKey { tile: (0, 0), id, path });
            l
        };
        for (path, y) in [(0u16, 0.0), (1, 3.0), (2, 12.0)] {
            net.lanes.push(lane((-60.0, y), (60.0, y), 1, path));
        }
        for (path, x) in [(0u16, 0.0), (1, -3.0), (2, -6.0)] {
            net.lanes.push(lane((x, 60.0), (x, -60.0), 2, path));
        }
        net.lanes[1].density = 3.0;
        net.link(1.5);
        net.compute_conflicts();
        assert_eq!((net.road_class(0).lanes, net.road_class(2).lanes, net.road_class(3).lanes), (3, 3, 3));
        assert_eq!(net.road_class(0).density, net.road_class(1).density);
        let js = Junctions::compile(&net);
        let j = &js.nodes[0];
        let yields: Vec<bool> = (0..3).flat_map(|ew| (3..6).map(move |sn| (ew, sn))).map(|(ew, sn)| j.movements[j.movement_of(ew).unwrap()].foes.iter().find(|f| j.movements[f.movement].lane == sn).expect("a foe").yields).collect();
        assert_eq!(yields.len(), 9);
        // the southbound road comes from the eastbound one's left: the eastbound goes first
        assert!(yields.iter().all(|&y| !y), "{yields:?}");
    }

    /// Two spline roads drawn over each other with no object and no link between them (X10's
    /// parallel ramp splines under a metre apart): a side merge in one junction table, a zipper
    /// between equals. Two splines' lanes side by side as a road's lanes (2.4 m apart, 3 m wide)
    /// and two lanes forking from one lane get none.
    #[test]
    fn two_roads_over_each_other_with_no_object_merge_but_a_road_s_lanes_and_a_fork_do_not() {
        let build = |gap: f64, fork: bool| {
            let mut net = Network::default();
            net.lanes.push(road((0.0, 0.0), (100.0, 0.0), 1)); // 0
            // 1: from 5 m on, or from 0's start (2 leads into both: a fork)
            net.lanes.push(road((if fork { 0.0 } else { 5.0 }, gap), (100.0, gap), 2));
            net.lanes.push(road((-50.0, 0.0), (0.0, 0.0), 3)); // 2: leads into 0
            net.link(1.5);
            net.compute_conflicts();
            net
        };
        let net = build(0.8, false);
        let c = net.crossings[0].iter().find(|c| c.other == 1).expect("a side merge");
        assert!(c.merge && c.side && !c.outside);
        let js = Junctions::compile(&net);
        assert_eq!(js.nodes.len(), 1);
        let j = &js.nodes[0];
        let m0 = j.movement_of(0).unwrap();
        let f = j.movements[m0].foes.iter().find(|f| j.movements[f.movement].lane == 1).expect("a foe");
        assert_eq!(f.kind, FoeKind::Merge);
        assert!(js.is_zipper(0, m0, f));
        assert!(build(2.4, false).crossings[0].is_empty(), "a road's lanes");
        assert!(build(0.8, true).crossings[0].iter().all(|c| c.other != 1), "a fork");
    }

    /// A spline road drawn across a piece of a crossing object it does not lead into: the pair
    /// meets in the object's table (the spline lane one of its movements, as a spline lane joining
    /// an object's way is), and a tile loaded later finds its crossings with the lanes already
    /// there.
    #[test]
    fn a_road_across_an_object_piece_meets_it_in_the_object_table_also_when_loaded_later() {
        let mut net = four_way();
        let first = net.lanes.len();
        // a road along x = 5, crossing the main piece 1 (y = 0) between -10 and 10, outside the
        // side piece 4 (x = 0)
        net.extend(vec![road((5.0, -40.0), (5.0, 40.0), 9)], 1.5);
        assert!(net.crossings[first].len() == 1 && net.crossings[first][0].other == 1 && net.crossings[first][0].outside);
        let js = Junctions::compile(&net);
        assert_eq!(js.nodes.len(), 1);
        let j = &js.nodes[0];
        let (main, cut) = (&j.movements[j.movement_of(1).unwrap()], &j.movements[j.movement_of(first).unwrap()]);
        assert!(main.foes.iter().any(|f| f.movement == j.movement_of(first).unwrap() && f.kind == FoeKind::Cross && !f.yields), "the priority-192 main road goes first");
        assert!(cut.foes.iter().any(|f| f.movement == j.movement_of(1).unwrap() && f.kind == FoeKind::Cross && f.yields));
    }
}
