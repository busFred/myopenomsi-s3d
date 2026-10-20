//! Second adversarial review: scenarios against the nine BeijingCity fixes.
#![allow(dead_code, unused_imports)]

use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network, TrafficLightController};
use omsi_traffic::{Junctions, PlayerView, Simulation, Vehicle, VehicleClass, Why};

const DT: f32 = 1.0 / 30.0;

// ---- data: AnwaiDajieCross parsed by hand (paths + use_traffic_light) -------------------

fn anwai_net() -> Option<Network> {
    let root = std::env::var_os("OMSI_ROOT")?;
    let p = std::path::PathBuf::from(root).join("Sceneryobjects/BEIJINGCITY/AnwaiDajieCross.sco");
    let text = std::fs::read_to_string(&p).ok()?;
    let lines: Vec<&str> = text.lines().map(|l| l.trim()).collect();
    let mut paths: Vec<(Vec<f64>, i32)> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if lines[i] == "[path]" {
            let v: Vec<f64> = lines[i + 1..i + 13].iter().map(|s| s.parse().unwrap()).collect();
            paths.push((v, -1));
            i += 13;
            continue;
        }
        if lines[i] == "[use_traffic_light]" {
            let t: i32 = lines[i + 1].parse().unwrap();
            paths.last_mut().unwrap().1 = t;
            i += 2;
            continue;
        }
        i += 1;
    }
    let mut net = Network::default();
    for (pi, (v, tl)) in paths.iter().enumerate() {
        let start = DVec3::new(v[0], v[1], v[2]);
        let (h, r, len) = (v[3], v[4], v[5]);
        if len <= 0.01 {
            continue;
        }
        let kind = LaneKind::from_code(v[8] as i32);
        let turn = match v[11] as i32 {
            2 => 1,
            3 => 2,
            _ => 0,
        };
        let dir = v[10] as i32;
        let mut push = |reverse: bool| {
            let mut l = LaneBuilder::arc(start, h, len, r, v[7], kind, v[9] as f32);
            if reverse {
                let pts: Vec<DVec3> = l.points.iter().rev().copied().collect();
                l = LaneBuilder::polyline(pts, kind, v[9] as f32);
            }
            l.turn = turn;
            l.source = 2;
            l.key = Some(LaneKey { tile: (0, 0), id: 1, path: pi as u16 });
            l.reversed = reverse;
            l.traffic_light = if *tl >= 0 { Some((0, *tl as usize)) } else { None };
            net.lanes.push(l);
        };
        match dir {
            1 => push(true),
            2 => {
                push(false);
                push(true)
            }
            _ => push(false),
        }
    }
    net.link(1.5);
    net.compute_conflicts();
    Some(net)
}

#[test]
fn data_anwai_crossings_and_chains() {
    let Some(net) = anwai_net() else {
        eprintln!("skipped: OMSI_ROOT");
        return;
    };
    let js = Junctions::compile(&net);
    let j = &js.nodes[0];
    let side: Vec<usize> = (0..net.lanes.len()).filter(|&i| net.lanes[i].kind == LaneKind::Sidewalk).collect();
    let lit_side: Vec<usize> = side.iter().copied().filter(|&i| net.lanes[i].traffic_light.is_some()).collect();
    let cutting: Vec<usize> = side.iter().copied().filter(|&w| (0..net.lanes.len()).any(|l| net.walks[l].iter().any(|x| x.0 == w))).collect();
    eprintln!("sidewalk lanes {} lit {} cutting a street lane {}", side.len(), lit_side.len(), cutting.len());
    for &w in &lit_side {
        let l = &net.lanes[w];
        let k = l.key.unwrap().path;
        eprintln!(
            "  lit sidewalk lane {w} path {k} light {:?} len {:.1} start {:?} end {:?} next {:?} prev {:?} in-crossing {:?}",
            l.traffic_light,
            l.length(),
            l.start().truncate(),
            l.end().truncate(),
            l.next,
            net.prev.get(w),
            js.of_foot_lane.get(&w)
        );
    }
    for (ci, c) in j.crossings.iter().enumerate() {
        let lens: Vec<String> = c.foot_lanes.iter().map(|&f| format!("{}:{:.1}{}", f, net.lanes[f].length(), if net.lanes[f].traffic_light.is_some() { "*" } else { "" })).collect();
        let lights: Vec<_> = c.foot_lanes.iter().filter_map(|&f| net.lanes[f].traffic_light).collect();
        eprintln!("crossing {ci}: light {:?} all lights {:?} cuts {} pieces {}", c.light, lights, c.cuts.len(), lens.join(" "));
        let pts: Vec<glam::DVec2> = c.foot_lanes.iter().flat_map(|&f| [net.lanes[f].start().truncate(), net.lanes[f].end().truncate()]).collect();
        let (lo, hi) = pts.iter().fold((glam::DVec2::splat(f64::MAX), glam::DVec2::splat(f64::MIN)), |(a, b), p| (a.min(*p), b.max(*p)));
        let headings: std::collections::BTreeSet<i32> = c.foot_lanes.iter().map(|&f| ((net.lanes[f].start_heading() as i32 + 360) % 180) / 10 * 10).collect();
        eprintln!("   extent {:?} .. {:?}, piece headings (mod 180, 10° bins) {:?}", lo, hi, headings);
    }
    // ways: forks and chained lights
    let mut forks = 0;
    let mut second_light = 0;
    for (mi, m) in j.movements.iter().enumerate() {
        if m.continues.len() > 1 {
            forks += 1;
            if forks <= 5 {
                eprintln!("fork: movement {mi} lane {} continues {:?} turn {} light {:?}", m.lane, m.continues, net.lanes[m.lane].turn, m.light);
            }
        }
        let ch = { let mut out = vec![(mi, 0.0f32)]; let mut off = net.lanes[m.lane].length(); let mut cur = mi; for _ in 0..5 { let Some(&n) = j.movements[cur].continues.first() else { break }; out.push((n, off)); off += net.lanes[j.movements[n].lane].length(); cur = n; } out };
        let lights: Vec<_> = ch.iter().filter_map(|&(n, off)| j.movements[n].light.map(|l| (l, off, net.lanes[j.movements[n].lane].start_heading()))).collect();
        if lights.len() > 1 && !j.movements[mi].entries.is_empty() {
            second_light += 1;
            eprintln!("way from movement {mi} carries lights {:?}", lights);
        }
    }
    eprintln!("movements {} with forks {} with two lights on the chain {}", j.movements.len(), forks, second_light);
    // forks where a later branch meets foes the first branch's chain does not carry
    let chain_foes = |start: usize| -> std::collections::BTreeSet<usize> {
        let mut out = std::collections::BTreeSet::new();
        let mut cur = start;
        for _ in 0..6 {
            for f in &j.movements[cur].foes {
                out.insert(f.movement);
            }
            match j.movements[cur].continues.first() {
                Some(&n) => cur = n,
                None => break,
            }
        }
        out
    };
    let (mut uncovered, mut branches) = (0, 0);
    for m in j.movements.iter().filter(|m| m.continues.len() > 1) {
        let first = chain_foes(m.continues[0]);
        for &b in &m.continues[1..] {
            branches += 1;
            let other = chain_foes(b);
            if other.iter().any(|f| !first.contains(f)) {
                uncovered += 1;
            }
        }
    }
    eprintln!("fork branches after the first: {branches}; of them meeting foes the claim (first branch) does not carry: {uncovered}");
    let turns_behind_fork = j.movements.iter().enumerate().filter(|(mi, m)| net.lanes[m.lane].turn != 0 && j.movements.iter().any(|p| p.continues.len() > 1 && p.continues[1..].contains(mi))).count();
    eprintln!("turn pieces that are a non-first branch of a fork: {turns_behind_fork}");
    let entries = j.movements.iter().filter(|m| !m.entries.is_empty()).count();
    eprintln!("movements with entries from outside: {}", entries);
}

// ---- scenario helpers (from tests/rules.rs) ---------------------------------------------

fn road(from: (f64, f64), to: (f64, f64), id: i64) -> omsi_sim::traffic::Lane {
    let mut l = LaneBuilder::polyline(vec![DVec3::new(from.0, from.1, 0.0), DVec3::new(to.0, to.1, 0.0)], LaneKind::Street, 3.5);
    l.source = 1;
    l.key = Some(LaneKey { tile: (0, 0), id, path: 0 });
    l
}

fn bent(pts: &[(f64, f64)], obj: i64, path: u16, priority: f32) -> omsi_sim::traffic::Lane {
    let mut l = LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), LaneKind::Street, 3.5);
    l.source = 2;
    l.key = Some(LaneKey { tile: (0, 0), id: obj, path });
    l.priority = priority;
    l
}

/// The west approach's left turn (to the north). `POLY=1`: a polyline that runs 15 m
/// along the straight piece first (it then crosses it); else a quarter circle from the
/// fork (it only shares its start with the straight piece).
fn left_w(path: u16) -> omsi_sim::traffic::Lane {
    if std::env::var_os("POLY").is_some() {
        return bent(&[(-15.0, -1.75), (0.0, -1.75), (1.75, 0.0), (1.75, 15.0)], 9, path, 128.0);
    }
    let r = 16.75;
    let mut l = LaneBuilder::arc(DVec3::new(-15.0, -1.75, 0.0), 90.0, std::f64::consts::FRAC_PI_2 * r, -r, 0.0, LaneKind::Street, 3.5);
    l.source = 2;
    l.key = Some(LaneKey { tile: (0, 0), id: 9, path });
    l
}

/// The east approach's left turn (to the south).
fn left_e(path: u16) -> omsi_sim::traffic::Lane {
    if std::env::var_os("POLY").is_some() {
        return bent(&[(15.0, 1.75), (0.0, 1.75), (-1.75, 0.0), (-1.75, -15.0)], 9, path, 128.0);
    }
    let r = 16.75;
    let mut l = LaneBuilder::arc(DVec3::new(15.0, 1.75, 0.0), 270.0, std::f64::consts::FRAC_PI_2 * r, -r, 0.0, LaneKind::Street, 3.5);
    l.source = 2;
    l.key = Some(LaneKey { tile: (0, 0), id: 9, path });
    l
}

/// Send `v` into the first way flagged `turn` (1 left, 2 right) along its road (the only way
/// on until a fork offers it), and plan on from there.
fn wish(v: &mut Vehicle, net: &Network, turn: i32) {
    let mut way: Vec<usize> = Vec::new();
    let mut at = v.state.lane;
    loop {
        let next = &net.lanes[at].next;
        let n = next.iter().copied().find(|&n| net.lanes[n].turn == turn);
        match (n, next.as_slice()) {
            (Some(n), _) => {
                way.push(n);
                break;
            }
            (None, [only]) => {
                way.push(*only);
                at = *only;
            }
            _ => panic!("no way with that turn"),
        }
    }
    v.state.planned_next = Some(way[0]);
    v.state.ahead = way[1..].to_vec();
    v.state.plan_next(net);
}

fn car(id: u64, lane: usize, s: f32, speed: f32) -> Vehicle {
    let mut st = AiState::new(lane, s, id);
    st.speed = speed;
    st.front = 2.0;
    st.rear = 2.0;
    st.length = 4.0;
    st.max_speed_kmh = 100.0;
    st.veh_type = 0;
    let mut v = Vehicle::new(id, VehicleClass::Car, st, 0.9, id * 7919);
    v.fresh = 0.0;
    v
}

struct Run {
    min_gap: f32,
    min_pair: (u64, u64),
    hardest_brake: f32,
    hardest_by: u64,
}

fn run(net: &Network, sim: &mut Simulation, vehicles: &mut Vec<Vehicle>, secs: f32, mut each: impl FnMut(f32, &Simulation, &mut [Vehicle])) -> Run {
    let mut r = Run { min_gap: f32::INFINITY, min_pair: (0, 0), hardest_brake: 0.0, hardest_by: 0 };
    let mut t = 0.0;
    while t < secs {
        sim.tick(net, DT, vehicles);
        for &i in &sim.removed {
            vehicles.remove(i);
        }
        for v in vehicles.iter() {
            if -v.state.acc > r.hardest_brake {
                r.hardest_brake = -v.state.acc;
                r.hardest_by = v.id;
            }
        }
        for (a, va) in vehicles.iter().enumerate() {
            for vb in vehicles.iter().skip(a + 1) {
                let d = va.obb().separation(&vb.obb()) as f32;
                if d < r.min_gap {
                    r.min_gap = d;
                    r.min_pair = (va.id, vb.id);
                }
            }
        }
        each(t, sim, vehicles.as_mut_slice());
        t += DT;
    }
    r
}

fn probe(t: f32, frames: &mut u32, cars: &[Vehicle]) {
    *frames += 1;
    if std::env::var_os("PROBE").is_some() && *frames % 15 == 0 {
        let line: Vec<String> = cars
            .iter()
            .map(|c| format!("[{} L{} s={:.1} v={:.1} a={:.1} why={:?} hold={:?}]", c.id, c.state.lane, c.state.s, c.state.speed, c.state.acc, c.last.why, c.last.hold))
            .collect();
        eprintln!("t={:.1} {}", t, line.join(" "));
    }
}

fn finish(mut net: Network) -> Network {
    net.link(1.5);
    net.compute_conflicts();
    net
}

// ---- 3 / regression: the width rule ------------------------------------------------------

/// Spandau's Altstaedter Ring (str_10spur_altstaedterring5.sli): five lanes a direction,
/// every street path 1.0 m wide, 3.0 m apart. 1138 of 11295 street paths in the install's
/// .sli files are 0.5..1.6 m wide.
#[test]
fn w1_lanes_drawn_one_metre_wide_are_closed_to_every_car() {
    let build = |w: f32| {
        let mut net = Network::default();
        for (k, off) in [0.0f32, 3.0].into_iter().enumerate() {
            let mut l = road((0.0, -(off as f64)), (2000.0, -(off as f64)), 1);
            l.offset = off;
            l.width = w;
            l.key = Some(LaneKey { tile: (0, 0), id: 1, path: k as u16 });
            net.lanes.push(l);
        }
        finish(net)
    };
    for w in [1.0f32, 3.0] {
        let net = build(w);
        let neighbours = net.lanes[1].left == Some(0);
        let c = car(1, 1, 0.0, 0.0);
        let open = omsi_traffic::open_to(&net.lanes[0], &c);
        let mut sim = Simulation::new(&net, vec![]);
        let mut slow = car(1, 1, 120.0, 5.5);
        slow.state.max_speed_kmh = 20.0;
        slow.state.change_cooldown = 100.0;
        let mut fast = car(2, 1, 80.0, 13.0);
        fast.state.change_cooldown = 0.0;
        let mut cars = vec![slow, fast];
        let mut changed = false;
        run(&net, &mut sim, &mut cars, 25.0, |_, _, cars| changed |= cars[1].state.lane != 1 || cars[1].state.change.is_some());
        eprintln!("W1 width {w}: neighbours {neighbours}, open_to {:?}, fast car changed lane {changed}", open);
    }
}

#[test]
fn w2_neighbour_boundaries() {
    let pair = |gap: f32, w: f32| {
        let mut a = road((0.0, 0.0), (10.0, 0.0), 1);
        let mut b = road((0.0, 0.0), (10.0, 0.0), 1);
        a.offset = 0.0;
        b.offset = gap;
        a.width = w;
        b.width = w;
        omsi_sim::traffic::Lane::beside_each_other(&a, &b)
    };
    for (gap, w) in [(4.0, 2.0), (4.01, 2.0), (4.5, 0.0), (4.5, 1.0), (3.75, 0.0), (5.6, 3.75), (6.0, 3.75)] {
        eprintln!("W2 offset gap {gap} width {w}: beside {}", pair(gap, w));
    }
}

// ---- 4: the player's bus straddling two lanes ------------------------------------------

#[test]
fn p2_a_car_changes_beside_the_players_bus_that_straddles_two_lanes() {
    // three lanes at y = 0, -3.5, -7; the bus drives along y = -1.6 (over lanes 0 and 1:
    // its right side at -2.85, a car's left side in lane 1 at -2.6)
    let mut net = Network::default();
    for (k, off) in [0.0f32, 3.5, 7.0].into_iter().enumerate() {
        let mut l = road((0.0, -(off as f64)), (3000.0, -(off as f64)), 1);
        l.offset = off;
        l.width = 3.0;
        l.key = Some(LaneKey { tile: (0, 0), id: 1, path: k as u16 });
        net.lanes.push(l);
    }
    let net = finish(net);
    for y in [-1.6f64, -2.0] {
        let mut sim = Simulation::new(&net, vec![]);
        sim.players.push(PlayerView { id: 7, pos: DVec3::new(98.0, y, 0.0), heading: 90.0, half_len: 6.0, half_width: 1.25, speed: 11.0, still: 0.0, blinker: 0, signalling: 0.0, priority: false });
        let mut slow = car(1, 2, 140.0, 5.5);
        slow.state.max_speed_kmh = 20.0;
        slow.state.change_cooldown = 100.0;
        let mut fast = car(2, 2, 100.0, 11.0);
        fast.state.change_cooldown = 0.0;
        let mut cars = vec![slow, fast];
        let (mut worst, mut hard, mut changed_at) = (f64::INFINITY, 0.0f32, None);
        for k in 0..(20.0 / DT) as u32 {
            sim.players[0].pos.x += 11.0 * DT as f64;
            run(&net, &mut sim, &mut cars, DT, |_, _, _| {});
            let bus = sim.players[0].obb(0.0, 0.0);
            worst = worst.min(cars[1].obb().separation(&bus));
            hard = hard.max(-cars[1].state.acc);
            if changed_at.is_none() && (cars[1].state.change.is_some() || cars[1].state.lane != 2) {
                changed_at = Some((k as f32 * DT, cars[1].pos.x - sim.players[0].pos.x));
            }
        }
        eprintln!("P2 bus along y {y}: occupancy lanes {:?}; the car started a change at (t, dx to bus) {changed_at:?}, closest to the bus {worst:.2} m, hardest brake {hard:.1}", sim.player_occupancy.keys().collect::<Vec<_>>());
    }
}

// ---- 2: the other ways onto a bus lane ------------------------------------------------------

#[test]
fn b1_the_way_on_into_a_bus_lane() {
    let bus_lane = |l: &mut omsi_sim::traffic::Lane| {
        // BeijingCity: group 0 density 0, the other groups 1, the lane's own density the max
        l.density = 1.0;
        l.group_density = vec![(0, 0.0), (1, 1.0)];
    };
    for with_normal in [false, true] {
        let mut entered = 0;
        for seed in 0..20u64 {
            let mut net = Network::default();
            net.lanes.push(road((0.0, 0.0), (300.0, 0.0), 1)); // 0
            let mut b = road((300.0, 0.0), (2000.0, 0.0), 2); // 1 bus lane
            bus_lane(&mut b);
            net.lanes.push(b);
            if with_normal {
                net.lanes.push(road((300.0, 0.0), (2000.0, 10.0), 3)); // 2 normal
            }
            let net = finish(net);
            let mut sim = Simulation::new(&net, vec![]);
            let mut c = car(1 + seed * 13, 0, 100.0, 10.0);
            c.state.traffic_pool = Some((0, std::sync::Arc::new(vec![1, 1])));
            c.state.plan_next(&net);
            let mut cars = vec![c];
            let mut on_bus = false;
            run(&net, &mut sim, &mut cars, 30.0, |_, _, cars| on_bus |= cars.first().map(|c| c.state.lane == 1).unwrap_or(false));
            if on_bus {
                entered += 1;
            }
        }
        eprintln!("B1 bus lane {} the only way on: a group-0 car drove onto it in {entered} of 20 runs", if with_normal { "is not" } else { "is" });
    }
    // the oncoming lane a bus lane: does a car pass a parked car on it?
    let mut net = Network::default();
    let mut a = road((0.0, 0.0), (2000.0, 0.0), 1);
    a.width = 3.0;
    let mut b = road((2000.0, 3.5), (0.0, 3.5), 2);
    b.width = 3.0;
    bus_lane(&mut b);
    net.lanes.push(a);
    net.lanes.push(b);
    let net = finish(net);
    let mut sim = Simulation::new(&net, vec![]);
    sim.parked.insert(0, vec![omsi_traffic::Parked { s: 200.0, lateral: 0.0, half_width: 0.9, half_len: 2.2 }]);
    let mut c = car(1, 0, 100.0, 12.0);
    c.state.traffic_pool = Some((0, std::sync::Arc::new(vec![1, 1])));
    let mut cars = vec![c];
    let mut went_out = false;
    run(&net, &mut sim, &mut cars, 30.0, |_, _, cars| went_out |= cars[0].passing.map(|p| p.lane.is_some()).unwrap_or(false));
    eprintln!("B1 the oncoming lane is a bus lane (closed to group 0): the car passed on it: {went_out}");
}

// ---- 9: a standing obstacle on a two-lane road with a solid line --------------------------

#[test]
fn s1_a_car_behind_a_standing_obstacle_with_a_solid_lane_line() {
    for solid in [false, true] {
        let mut net = Network::default();
        for (k, off) in [0.0f32, 3.5].into_iter().enumerate() {
            let mut l = road((0.0, -(off as f64)), (2000.0, -(off as f64)), 1);
            l.offset = off;
            l.width = 3.0;
            l.no_overtaking = solid;
            l.key = Some(LaneKey { tile: (0, 0), id: 1, path: k as u16 });
            net.lanes.push(l);
        }
        let net = finish(net);
        let mut sim = Simulation::new(&net, vec![]);
        sim.parked.insert(1, vec![omsi_traffic::Parked { s: 200.0, lateral: 0.0, half_width: 0.9, half_len: 2.2 }]);
        let mut cars = vec![car(1, 1, 100.0, 12.0)];
        cars[0].state.change_cooldown = 0.0;
        run(&net, &mut sim, &mut cars, 60.0, |_, _, _| {});
        eprintln!("S1 solid {solid}: after 60 s the car is on lane {} at s {:.1} (the parked car is at 200 on lane 1)", cars[0].state.lane, cars[0].state.s);
    }
}

// ---- 7: a fork after the lead-in ---------------------------------------------------------

/// West approach (0) → lead-in piece (1, 25 m) → forks into straight (S) and a left turn (T)
/// north across the oncoming straight piece (7). `turn_first`: T comes first in the lead-in's
/// `next` (so in `continues`).
fn fork_junction(turn_first: bool, lead_in: f64) -> (Network, usize, usize) {
    let mut net = Network::default();
    let x0 = -15.0 - lead_in;
    net.lanes.push(road((-600.0, -1.75), (x0, -1.75), 1)); // 0
    net.lanes.push(bent(&[(x0, -1.75), (-15.0, -1.75)], 9, 0, 128.0)); // 1 lead-in
    let s = bent(&[(-15.0, -1.75), (15.0, -1.75)], 9, 1, 128.0);
    let mut t = left_w(2);
    t.turn = 1;
    let (si, ti) = if turn_first {
        net.lanes.push(t);
        net.lanes.push(s);
        (3, 2)
    } else {
        net.lanes.push(s);
        net.lanes.push(t);
        (2, 3)
    };
    net.lanes.push(road((15.0, -1.75), (1500.0, -1.75), 2)); // 4
    net.lanes.push(road((1.75, 15.0), (1.75, 1500.0), 3)); // 5
    net.lanes.push(road((1400.0, 1.75), (15.0, 1.75), 4)); // 6 oncoming approach
    net.lanes.push(bent(&[(15.0, 1.75), (-15.0, 1.75)], 9, 3, 128.0)); // 7 oncoming straight
    net.lanes.push(road((-15.0, 1.75), (-1500.0, 1.75), 5)); // 8
    (finish(net), si, ti)
}

fn fork_run(turn_first: bool) -> (Option<f32>, Run, f32, f32) {
    let (net, _si, ti) = fork_junction(turn_first, 25.0);
    let js = Junctions::compile(&net);
    let (_, lead) = js.movement(1).unwrap();
    let (_, turn) = js.movement(ti).unwrap();
    eprintln!(
        "fork turn_first={turn_first}: lead-in continues {:?}, conflict_start {:.1}, turn yields {}; turn foes (lane, kind, mine, yields) {:?}; lead-in foes {:?}",
        lead.continues,
        lead.conflict_start,
        turn.foes.iter().any(|f| f.yields),
        turn.foes.iter().map(|f| (net_lane(&js, f.movement), f.kind, f.mine, f.yields)).collect::<Vec<_>>(),
        lead.foes.iter().map(|f| (net_lane(&js, f.movement), f.kind, f.mine, f.yields)).collect::<Vec<_>>()
    );
    let mut sim = Simulation::new(&net, vec![]);
    // stream: 31 cars 25 m apart at 13 m/s (60 s)
    let mut cars: Vec<Vehicle> = (0..31u64).map(|k| car(10 + k, 6, 1350.0 - 25.0 * k as f32, 13.0)).collect();
    let mut me = car(1, 0, 400.0, 11.0);
    wish(&mut me, &net, 1);
    cars.push(me);
    let (mut frames, mut entered_turn, mut my_brake, mut stream_brake) = (0, None, 0.0f32, 0.0f32);
    let r = run(&net, &mut sim, &mut cars, 75.0, |t, _, cars| {
        let me = cars.iter().find(|c| c.id == 1).unwrap();
        probe(t, &mut frames, std::slice::from_ref(me));
        my_brake = my_brake.max(-me.state.acc);
        for c in cars.iter().filter(|c| c.id >= 10) {
            stream_brake = stream_brake.max(-c.state.acc);
        }
        if me.state.lane == ti && entered_turn.is_none() {
            entered_turn = Some(t);
        }
    });
    let me = cars.iter().find(|c| c.id == 1).unwrap();
    eprintln!(
        "fork turn_first={turn_first}: entered the turn at {entered_turn:?}, own hardest brake {my_brake:.1}, stream hardest {stream_brake:.1}, min gap {:.2} ({:?}), end lane {} s {:.1}",
        r.min_gap, r.min_pair, me.state.lane, me.state.s
    );
    (entered_turn, r, my_brake, stream_brake)
}

#[test]
fn f1_a_left_turner_behind_a_lead_in_that_forks() {
    let good = fork_run(true);
    let bad = fork_run(false);
    assert!(good.1.min_gap > 0.3 && bad.1.min_gap > 0.3, "touched: {:.2} / {:.2}", good.1.min_gap, bad.1.min_gap);
    assert!(bad.2 < 4.5, "the left turner braked {:.1} m/s² inside the junction when its claim followed the straight branch", bad.2);
    assert!(bad.3 < 3.0, "a stream car braked {:.1} for the turner", bad.3);
}

// ---- 7: 90 s against a stream --------------------------------------------------------------

#[test]
fn f2_a_left_turner_facing_a_stream_for_90_s_never_goes_into_it() {
    let (net, _, ti) = fork_junction(true, 25.0);
    let mut sim = Simulation::new(&net, vec![]);
    // 13 m/s, 25 m apart: a car every 1.9 s for 90 s; the approach is 1385 m
    let mut cars: Vec<Vehicle> = (0..53u64).map(|k| car(10 + k, 6, 1350.0 - 25.0 * k as f32, 13.0)).collect();
    // the remaining stream on the oncoming exit and beyond will not matter
    let mut me = car(1, 0, 520.0, 8.0);
    wish(&mut me, &net, 1);
    cars.push(me);
    let (mut frames, mut entered, mut stream_brake, mut longest_wait) = (0, None, 0.0f32, 0.0f32);
    let mut wait = 0.0f32;
    let mut last_stream_past = 0.0f32;
    let r = run(&net, &mut sim, &mut cars, 105.0, |t, _, cars| {
        let me = cars.iter().find(|c| c.id == 1).unwrap().clone();
        probe(t, &mut frames, std::slice::from_ref(&me));
        if std::env::var_os("PROBE2").is_some() && t > 19.5 && t < 23.5 && frames % 3 == 0 {
            let near: Vec<String> = cars.iter().filter(|c| c.id >= 10 && ((c.state.lane == 6 && c.state.s > 1300.0) || c.state.lane == 7)).map(|c| format!("{}:L{} s{:.1} v{:.1} a{:.1}", c.id, c.state.lane, c.state.s, c.state.speed, c.state.acc)).collect();
            eprintln!("t={t:.2} me L{} s{:.2} v{:.1} imp {:.2} why {:?} wait_for {:?} | {}", me.state.lane, me.state.s, me.state.speed, me.impatience.value, me.last.why, me.last.wait_for, near.join(" "));
        }
        for c in cars.iter().filter(|c| c.id >= 10) {
            stream_brake = stream_brake.max(-c.state.acc);
            if c.state.lane == 7 {
                last_stream_past = t;
            }
        }
        wait = if me.state.speed < 0.3 { wait + DT } else { 0.0 };
        longest_wait = longest_wait.max(wait);
        if me.state.lane == ti && entered.is_none() {
            entered = Some(t);
        }
    });
    eprintln!("F2: turner entered its turn at {entered:?}; the last stream car was on the oncoming piece at {last_stream_past:.1} s; stream hardest brake {stream_brake:.1}; min gap {:.2} {:?}; turner waited {longest_wait:.0} s", r.min_gap, r.min_pair);
    assert!(r.min_gap > 0.3);
    assert!(stream_brake < 3.0, "a stream car braked {stream_brake:.1} for the turner");
}

// ---- 7: two left turners facing each other ------------------------------------------------

#[test]
fn f3_two_left_turners_facing_each_other() {
    let mut net = Network::default();
    net.lanes.push(road((-400.0, -1.75), (-15.0, -1.75), 1)); // 0 W approach
    let mut tw = left_w(0);
    tw.turn = 1;
    net.lanes.push(tw); // 1 W left (to north)
    net.lanes.push(road((1.75, 15.0), (1.75, 1500.0), 2)); // 2
    net.lanes.push(road((400.0, 1.75), (15.0, 1.75), 3)); // 3 E approach
    let mut te = left_e(1);
    te.turn = 1;
    net.lanes.push(te); // 4 E left (to south)
    net.lanes.push(road((-1.75, -15.0), (-1.75, -1500.0), 4)); // 5
    // the straights, each crossing the other's turn
    net.lanes.push(bent(&[(-15.0, -1.75), (15.0, -1.75)], 9, 2, 128.0)); // 6 W straight
    net.lanes.push(road((15.0, -1.75), (1500.0, -1.75), 5)); // 7
    net.lanes.push(bent(&[(15.0, 1.75), (-15.0, 1.75)], 9, 3, 128.0)); // 8 E straight
    net.lanes.push(road((-15.0, 1.75), (-1500.0, 1.75), 6)); // 9
    let net = finish(net);
    let js = Junctions::compile(&net);
    let (_, m) = js.movement(1).unwrap();
    eprintln!("F3 W-left foes: {:?}", m.foes.iter().map(|f| (net_lane(&js, f.movement), f.yields)).collect::<Vec<_>>());
    for offset in [0.0f32, 1.0, 3.0] {
        let mut sim = Simulation::new(&net, vec![]);
        let mut a = car(1, 0, 300.0, 10.0);
        wish(&mut a, &net, 1);
        let mut b = car(2, 3, 300.0 - offset, 10.0);
        wish(&mut b, &net, 1);
        // a straight car behind each turner
        let c = car(3, 0, 280.0, 10.0);
        let d = car(4, 3, 280.0 - offset, 10.0);
        let mut cars = vec![a, b, c, d];
        let (mut frames, mut done) = (0, [None; 2]);
        let r = run(&net, &mut sim, &mut cars, 60.0, |t, _, cars| {
            probe(t, &mut frames, cars);
            if (t - 30.0).abs() < DT * 0.5 {
                for c in cars.iter() {
                    eprintln!("F3 offset {offset} t=30: car {} lane {} next {:?} speed {:.1} why {:?} wait_for(index) {:?} held {} impatience {:.2}", c.id, c.state.lane, c.state.planned_next, c.state.speed, c.last.why, c.last.wait_for, c.last.held, c.impatience.value);
                }
            }
            for (k, id) in [1u64, 2].into_iter().enumerate() {
                if let Some(c) = cars.iter().find(|c| c.id == id) {
                    if (c.state.lane == 2 || c.state.lane == 5) && done[k].is_none() {
                        done[k] = Some(t);
                    }
                }
            }
        });
        eprintln!("F3 offset {offset}: turners out at {done:?}, min gap {:.2} {:?}, hardest brake {:.1} by {}", r.min_gap, r.min_pair, r.hardest_brake, r.hardest_by);
        assert!(r.min_gap > 0.3, "touched at offset {offset}");
        assert!(done.iter().all(|d| d.map(|t| t < 45.0).unwrap_or(false)), "a turner was not through in 45 s: {done:?}");
    }
}

/// The plainest case: one approach that forks into a left turn and straight on, nothing
/// else in the junction. A left turner with a straight-on car right behind it.
#[test]
fn f3b_a_left_turner_with_a_straight_car_behind_it_and_nothing_else() {
    let mut net = Network::default();
    net.lanes.push(road((-400.0, -1.75), (-15.0, -1.75), 1)); // 0 W approach
    let mut tw = left_w(0);
    tw.turn = 1;
    net.lanes.push(tw); // 1 W left (to north)
    net.lanes.push(road((1.75, 15.0), (1.75, 1500.0), 2)); // 2
    net.lanes.push(bent(&[(-15.0, -1.75), (15.0, -1.75)], 9, 2, 128.0)); // 3 W straight
    net.lanes.push(road((15.0, -1.75), (1500.0, -1.75), 5)); // 4
    let net = finish(net);
    let js = Junctions::compile(&net);
    let (_, m) = js.movement(1).unwrap();
    eprintln!("F3b left turn foes (lane, kind, mine, theirs, yields): {:?}", m.foes.iter().map(|f| (net_lane(&js, f.movement), f.kind, f.mine, f.theirs, f.yields)).collect::<Vec<_>>());
    let mut sim = Simulation::new(&net, vec![]);
    let mut a = car(1, 0, 300.0, 10.0);
    wish(&mut a, &net, 1);
    let mut b = car(3, 0, 285.0, 10.0);
    b.state.planned_next = Some(3);
    b.state.ahead = vec![4];
    let mut cars = vec![a, b];
    let (mut frames, mut out) = (0, None);
    let r = run(&net, &mut sim, &mut cars, 60.0, |t, _, cars| {
        probe(t, &mut frames, cars);
        if cars.iter().any(|c| c.id == 1 && c.state.lane == 2) && out.is_none() {
            out = Some(t);
        }
    });
    let c1 = cars.iter().find(|c| c.id == 1).unwrap();
    eprintln!("F3b: the turner got out at {out:?}; at the end it is on lane {} s {:.1} waiting for {:?}; min gap {:.2}", c1.state.lane, c1.state.s, c1.last.wait_for, r.min_gap);
    assert!(out.is_some(), "the left turner waited 60 s for the car behind it");
}

fn net_lane(js: &Junctions, m: usize) -> usize {
    js.nodes[0].movements[m].lane
}

// ---- 7: the conflict on the third piece, 65 m in ----------------------------------------

/// The car's way: approach (0) → P1 (1, 40 m) → P2 (2, 25 m) → P3 (3, 30 m, crosses the
/// main road piece Q at its middle, priority 64). Main road: approach (5) → Q (6, 192).
fn deep_conflict() -> Network {
    let mut net = Network::default();
    net.lanes.push(road((-600.0, 0.0), (-80.0, 0.0), 1)); // 0
    net.lanes.push(bent(&[(-80.0, 0.0), (-40.0, 0.0)], 9, 0, 64.0)); // 1
    net.lanes.push(bent(&[(-40.0, 0.0), (-15.0, 0.0)], 9, 1, 64.0)); // 2
    net.lanes.push(bent(&[(-15.0, 0.0), (15.0, 0.0)], 9, 2, 64.0)); // 3
    net.lanes.push(road((15.0, 0.0), (1500.0, 0.0), 2)); // 4
    net.lanes.push(road((0.0, -1400.0), (0.0, -15.0), 3)); // 5
    net.lanes.push(bent(&[(0.0, -15.0), (0.0, 15.0)], 9, 3, 192.0)); // 6
    net.lanes.push(road((0.0, 15.0), (0.0, 1500.0), 4)); // 7
    finish(net)
}

#[test]
fn f4_a_conflict_on_the_third_piece_is_waited_for_before_it() {
    let net = deep_conflict();
    let js = Junctions::compile(&net);
    let (_, p1) = js.movement(1).unwrap();
    eprintln!("F4 P1 continues {:?} conflict_start {:.1} end {:.1} foes {}", p1.continues, p1.conflict_start, p1.conflict_end, p1.foes.len());
    let mut sim = Simulation::new(&net, vec![]);
    let mut cars: Vec<Vehicle> = (0..20u64).map(|k| car(10 + k, 5, 1300.0 - 25.0 * k as f32, 13.0)).collect();
    cars.push(car(1, 0, 400.0, 12.0));
    let (mut frames, mut stood_at, mut in_conflict_while_stream) = (0, None, false);
    let mut stream_brake = 0.0f32;
    let r = run(&net, &mut sim, &mut cars, 60.0, |t, _, cars| {
        let me = cars.iter().find(|c| c.id == 1).unwrap().clone();
        probe(t, &mut frames, std::slice::from_ref(&me));
        if me.state.speed < 0.3 && stood_at.is_none() {
            stood_at = Some((me.state.lane, me.state.s, t));
        }
        for c in cars.iter().filter(|c| c.id >= 10) {
            stream_brake = stream_brake.max(-c.state.acc);
            if c.state.lane == 6 && me.state.lane == 3 && me.state.s + 2.0 > 13.0 && me.state.s - 2.0 < 17.0 {
                in_conflict_while_stream = true;
            }
        }
    });
    let me = cars.iter().find(|c| c.id == 1).unwrap();
    eprintln!("F4: first stood at (lane, s, t) {stood_at:?}; in the conflict with a stream car on Q: {in_conflict_while_stream}; stream brake {stream_brake:.1}; min gap {:.2}; hardest {:.1} by {}; end lane {} s {:.0}", r.min_gap, r.hardest_brake, r.hardest_by, me.state.lane, me.state.s);
    assert!(r.min_gap > 0.3 && !in_conflict_while_stream);
    assert!(stream_brake < 3.0);
}

#[test]
fn f5_a_car_granted_go_gets_a_wait_on_the_lead_in() {
    let net = deep_conflict();
    // the car goes on an empty main road; a main road car appears while it is on P1 / P2
    for (appear_at, dist) in [(5.5f32, 120.0f32), (6.5, 90.0), (7.5, 70.0), (8.0, 60.0), (8.0, 50.0), (8.5, 50.0), (8.5, 45.0), (9.0, 45.0), (9.0, 40.0)] {
        let mut sim = Simulation::new(&net, vec![]);
        let mut cars = vec![car(1, 0, 450.0, 12.0)];
        let mut frames = 0;
        let r1 = run(&net, &mut sim, &mut cars, appear_at, |t, _, cars| probe(t, &mut frames, cars));
        let me = cars[0].clone();
        cars.push(car(20, 5, 1385.0 - dist, 13.0));
        let mut my_brake = 0.0f32;
        let mut main_brake = 0.0f32;
        let r2 = run(&net, &mut sim, &mut cars, 20.0, |t, _, cars| {
            probe(t + appear_at, &mut frames, cars);
            if let Some(c) = cars.iter().find(|c| c.id == 1) {
                my_brake = my_brake.max(-c.state.acc);
            }
            if let Some(c) = cars.iter().find(|c| c.id == 20) {
                main_brake = main_brake.max(-c.state.acc);
            }
        });
        eprintln!(
            "F5 main car appears at {appear_at} s {dist} m out, side car then on lane {} s {:.1} v {:.1}: side car hardest {:.1}, main car hardest {:.1}, min gap {:.2}",
            me.state.lane, me.state.s, me.state.speed, my_brake, main_brake, r2.min_gap.min(r1.min_gap)
        );
        assert!(r2.min_gap > 0.3, "touched");
        assert!(my_brake < 4.5, "the side car braked {my_brake:.1} inside the junction");
    }
}

#[test]
fn f5b_both_present_from_the_start_the_main_road_car_is_not_braked_hard() {
    let net = deep_conflict();
    let mut worst = (0.0f32, 0.0f32, f32::INFINITY);
    for k in 0..30 {
        let main_s = 1100.0 + 6.0 * k as f32;
        let mut sim = Simulation::new(&net, vec![]);
        let mut cars = vec![car(1, 0, 400.0, 12.0), car(20, 5, main_s, 13.0)];
        let (mut main_brake, mut side_brake) = (0.0f32, 0.0f32);
        let r = run(&net, &mut sim, &mut cars, 25.0, |_, _, cars| {
            for c in cars.iter() {
                if c.id == 20 {
                    main_brake = main_brake.max(-c.state.acc);
                } else {
                    side_brake = side_brake.max(-c.state.acc);
                }
            }
        });
        if main_brake > worst.1 {
            worst = (main_s, main_brake, r.min_gap);
        }
        if main_brake > 3.0 || r.min_gap < 1.0 {
            eprintln!("F5b main car from s {main_s}: main hardest {main_brake:.1}, side hardest {side_brake:.1}, min gap {:.2}", r.min_gap);
        }
    }
    eprintln!("F5b worst (main start, main brake, gap): {worst:?}");
    assert!(worst.1 < 3.0, "the main road car braked {:.1} for a side road car", worst.1);
}

// ---- 7: a straight car of priority on a chained way, a side road car waiting ------------

#[test]
fn f6_a_main_road_car_on_a_chained_way_does_not_wait_for_a_side_road_car() {
    // main road: approach (0) → M1 lead-in (1, 30 m) → M2 (2, crosses the side piece, 192)
    // side road: approach (4) → S (5, 64)
    let mut net = Network::default();
    net.lanes.push(road((-600.0, 0.0), (-45.0, 0.0), 1)); // 0
    net.lanes.push(bent(&[(-45.0, 0.0), (-15.0, 0.0)], 9, 0, 192.0)); // 1
    net.lanes.push(bent(&[(-15.0, 0.0), (15.0, 0.0)], 9, 1, 192.0)); // 2
    net.lanes.push(road((15.0, 0.0), (1500.0, 0.0), 2)); // 3
    net.lanes.push(road((0.0, -300.0), (0.0, -15.0), 3)); // 4
    net.lanes.push(bent(&[(0.0, -15.0), (0.0, 15.0)], 9, 2, 64.0)); // 5
    net.lanes.push(road((0.0, 15.0), (0.0, 1500.0), 4)); // 6
    let net = finish(net);
    let js = Junctions::compile(&net);
    let (_, s) = js.movement(5).unwrap();
    let (_, m1) = js.movement(1).unwrap();
    eprintln!("F6 side yields to: {:?}; M1 chained foes {:?}", s.foes.iter().map(|f| (net_lane(&js, f.movement), f.yields)).collect::<Vec<_>>(), m1.foes.iter().map(|f| (net_lane(&js, f.movement), f.yields, f.mine)).collect::<Vec<_>>());
    for side_s in [270.0f32, 280.0, 283.0] {
        let mut sim = Simulation::new(&net, vec![]);
        // the side car stands at its line; the main road car comes at 13 m/s from 120 m
        let side = car(2, 4, side_s, 0.0);
        let main = car(1, 0, 555.0 - 120.0, 13.0);
        let mut cars = vec![main, side];
        let (mut frames, mut held, mut main_brake, mut main_min_v) = (0, 0.0f32, 0.0f32, f32::INFINITY);
        let r = run(&net, &mut sim, &mut cars, 25.0, |t, _, cars| {
            probe(t, &mut frames, cars);
            let m = cars.iter().find(|c| c.id == 1).unwrap();
            if m.last.why == Why::Junction {
                held += DT;
            }
            main_brake = main_brake.max(-m.state.acc);
            main_min_v = main_min_v.min(m.state.speed);
        });
        eprintln!("F6 side car at {side_s}: main car held {held:.1} s by the junction, hardest brake {main_brake:.1}, slowest {main_min_v:.1}; min gap {:.2}", r.min_gap);
        assert!(r.min_gap > 0.3);
        assert!(held < 0.1, "the main road car waited {held:.1} s for a side road car");
    }
}

// ---- 8: a rolling queue at a yellow -------------------------------------------------------

#[test]
fn y1_a_rolling_queue_at_8_m_s_and_a_3_s_yellow() {
    let mut net = Network::default();
    net.lanes.push(road((0.0, 0.0), (400.0, 0.0), 1));
    let mut p = bent(&[(400.0, 0.0), (430.0, 0.0)], 9, 0, 128.0);
    p.traffic_light = Some((0, 0));
    net.lanes.push(p);
    net.lanes.push(road((430.0, 0.0), (2000.0, 0.0), 2));
    let net = finish(net);
    let mut worst_red_entry: Option<(u64, f32, f32)> = None;
    let mut worst_brake = (0.0f32, 0u64, 0.0f32);
    for k0 in 0..20 {
        // the first car's front is d0 from the line when the yellow comes (t = 20 s)
        let d0 = k0 as f32 * 3.0;
        let yellow: f32 = std::env::var("YELLOW").ok().and_then(|v| v.parse().ok()).unwrap_or(3.0);
        let ctl = TrafficLightController::new(vec![vec![(6, 20.0), (9, yellow), (0, 30.0)]], 50.0 + yellow);
        let mut sim = Simulation::new(&net, vec![ctl]);
        let start = 400.0 - 2.0 - d0 - 8.0 * 20.0;
        let mut cars: Vec<Vehicle> = (0..6u64)
            .map(|k| {
                let mut c = car(k + 1, 0, start - 15.0 * k as f32, 8.0);
                c.state.max_speed_kmh = 28.8;
                c.state.desire = 1.0;
                c
            })
            .collect();
        let mut frames = 0;
        let mut entered: [Option<(f32, i32)>; 6] = [None; 6];
        let r = run(&net, &mut sim, &mut cars, 50.0, |t, sim, cars| {
            if std::env::var_os("PROBE").is_some() && d0 == 15.0 {
                probe(t, &mut frames, cars);
            }
            for c in cars.iter() {
                let k = c.id as usize - 1;
                if c.state.lane != 0 && entered[k].is_none() {
                    entered[k] = Some((t, sim.lights[0].state(0)));
                }
            }
        });
        for (k, e) in entered.iter().enumerate() {
            if let Some((t, 0)) = e {
                let late = t - 20.0 - std::env::var("YELLOW").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(3.0);
                if worst_red_entry.map(|w| late > w.2).unwrap_or(true) {
                    worst_red_entry = Some((k as u64 + 1, d0, late));
                }
            }
        }
        if r.hardest_brake > worst_brake.0 {
            worst_brake = (r.hardest_brake, r.hardest_by, d0);
        }
        let states: Vec<String> = entered.iter().map(|e| e.map(|(t, s)| format!("{t:.1}/{s}")).unwrap_or("-".into())).collect();
        eprintln!("Y1 d0 {d0:>4.0}: entries (t/state) {} min gap {:.2}", states.join(" "), r.min_gap);
        assert!(r.min_gap > 0.3);
    }
    eprintln!("Y1 worst red entry (car, d0, s after red began): {worst_red_entry:?}; hardest brake {:?}", worst_brake);
    assert!(worst_brake.0 < 4.5, "braked {:.1}", worst_brake.0);
}

// ---- 1: a car put down past its stop line ------------------------------------------------

#[test]
fn r1_a_car_on_the_lit_piece_past_its_line_at_red() {
    // our way: approach (0) → lit piece (1, 30 m, crosses the cross street at its middle)
    // cross street: approach (3) → piece (4), green for it
    let mut net = Network::default();
    net.lanes.push(road((-600.0, -1.75), (-15.0, -1.75), 1)); // 0
    let mut p = bent(&[(-15.0, -1.75), (15.0, -1.75)], 9, 0, 128.0);
    p.traffic_light = Some((0, 0));
    net.lanes.push(p); // 1
    net.lanes.push(road((15.0, -1.75), (1500.0, -1.75), 2)); // 2
    net.lanes.push(road((1.75, -1400.0), (1.75, -15.0), 3)); // 3
    let mut q = bent(&[(1.75, -15.0), (1.75, 15.0)], 9, 1, 128.0);
    q.traffic_light = Some((0, 1));
    net.lanes.push(q); // 4
    net.lanes.push(road((1.75, 15.0), (1.75, 1500.0), 4)); // 5
    let net = finish(net);
    for s0 in [0.2f32, 0.4, 0.6, 1.0, 2.3, 2.8, 5.0] {
        // light 0 (ours) red for 40 s, light 1 (cross) green for 40 s
        let ctl = TrafficLightController::new(vec![vec![(0, 40.0), (6, 40.0)], vec![(6, 40.0), (0, 40.0)]], 80.0);
        let mut sim = Simulation::new(&net, vec![ctl]);
        // a sparse cross stream: every 60 m at 13 m/s
        let mut cars: Vec<Vehicle> = (0..15u64).map(|k| car(10 + k, 3, 1300.0 - 60.0 * k as f32, 13.0)).collect();
        cars.push(if s0 < 2.0 { car(1, 0, 585.0 - 2.0 + s0, 0.0) } else { car(1, 1, s0 - 2.0, 0.0) });
        let (mut frames, mut crossed_at) = (0, None);
        let r = run(&net, &mut sim, &mut cars, 35.0, |t, _, cars| {
            let me = cars.iter().find(|c| c.id == 1).unwrap().clone();
            probe(t, &mut frames, std::slice::from_ref(&me));
            if ((me.state.lane == 1 && me.state.s > 17.0) || me.state.lane == 2) && crossed_at.is_none() {
                crossed_at = Some(t);
            }
        });
        eprintln!("R1 put down standing with its front {s0} m past the line (red for 40 s): crossed the cross street at {crossed_at:?}, min gap {:.2} {:?}", r.min_gap, r.min_pair);
        assert!(s0 < 0.5 || crossed_at.is_none(), "a car standing with its front {s0} m past its line crossed on red at {crossed_at:?}");
    }
}

// ---- 4: the player's bus in a bay after the junction --------------------------------------

#[test]
fn p1_the_players_bus_in_a_bay_just_past_the_junction() {
    let mut net = Network::default();
    net.lanes.push(road((-600.0, 0.0), (-15.0, 0.0), 1)); // 0
    net.lanes.push(bent(&[(-15.0, 0.0), (15.0, 0.0)], 9, 0, 128.0)); // 1
    net.lanes.push(road((15.0, 0.0), (1500.0, 0.0), 2)); // 2
    net.lanes.push(road((11.0, -300.0), (11.0, -15.0), 3)); // 3
    net.lanes.push(bent(&[(11.0, -15.0), (11.0, 15.0)], 9, 1, 128.0)); // 4
    net.lanes.push(road((11.0, 15.0), (11.0, 1500.0), 4)); // 5
    let net = finish(net);
    for lateral in [0.0f64, 3.0] {
        let mut sim = Simulation::new(&net, vec![]);
        // a 12 m bus standing in a bay to the right (south) of the exit, its rear 3 m past
        // the piece
        sim.players.push(PlayerView { id: 7, pos: DVec3::new(15.0 + 3.0 + 6.0, -lateral, 0.0), heading: 90.0, half_len: 6.0, half_width: 1.25, speed: 0.0, still: 30.0, blinker: 0, signalling: 0.0, priority: false });
        let mut cars = vec![car(1, 0, 500.0, 10.0)];
        let (mut frames, mut held_exit) = (0, 0.0f32);
        run(&net, &mut sim, &mut cars, 30.0, |t, _, cars| {
            probe(t, &mut frames, cars);
            if cars[0].last.hold == Some(omsi_traffic::Hold::ExitFull) {
                held_exit += DT;
            }
        });
        eprintln!("P1 bus {lateral} m to the side: car held {held_exit:.1} s for no room past the junction; car at lane {} s {:.1}", cars[0].state.lane, cars[0].state.s);
    }
}

/// The tick with many claims at one big object: 150 cars on AnwaiDajieCross's street
/// pieces (every one of them claims; `decide` walks the expanded claims per foe).
#[test]
#[ignore]
fn data_cost_of_many_claims_at_anwai() {
    let Some(net) = anwai_net() else { return };
    let streets: Vec<usize> = (0..net.lanes.len()).filter(|&i| net.lanes[i].kind == LaneKind::Street && net.lanes[i].length() > 6.0).collect();
    for n in [50usize, 150] {
        let mut sim = Simulation::new(&net, vec![TrafficLightController::new(vec![vec![(6, 30.0), (0, 30.0)]; 8], 60.0)]);
        let mut cars: Vec<Vehicle> = (0..n).map(|k| {
            let l = streets[(k * 7919) % streets.len()];
            let mut c = car(k as u64 + 1, l, 1.0 + ((k * 31) % 5) as f32, 3.0);
            c.state.plan_next(&net);
            c
        }).collect();
        for _ in 0..10 {
            sim.tick(&net, DT, &mut cars);
            for &i in &sim.removed {
                cars.remove(i);
            }
        }
        let t0 = std::time::Instant::now();
        let frames = 60;
        for _ in 0..frames {
            sim.tick(&net, DT, &mut cars);
            for &i in &sim.removed {
                cars.remove(i);
            }
        }
        eprintln!("COST {} cars on AnwaiDajieCross: {:.2} ms per tick", n, t0.elapsed().as_secs_f64() * 1000.0 / frames as f64);
    }
}

fn seg_hit(a: glam::DVec2, b: glam::DVec2, c: glam::DVec2, d: glam::DVec2) -> bool {
    let cr = |o: glam::DVec2, p: glam::DVec2, q: glam::DVec2| (p - o).perp_dot(q - o);
    let (d1, d2, d3, d4) = (cr(c, d, a), cr(c, d, b), cr(a, b, c), cr(a, b, d));
    (d1 * d2 < 0.0) && (d3 * d4 < 0.0)
}

#[test]
fn data_anwai_lit_stubs_reach() {
    let Some(net) = anwai_net() else { return };
    let js = Junctions::compile(&net);
    let streets: Vec<usize> = (0..net.lanes.len()).filter(|&i| net.lanes[i].kind == LaneKind::Street).collect();
    let cuts_geo = |w: usize| -> usize {
        let wl = &net.lanes[w];
        streets
            .iter()
            .filter(|&&s| {
                let sl = &net.lanes[s];
                wl.points.windows(2).any(|a| sl.points.windows(2).any(|b| seg_hit(a[0].truncate(), a[1].truncate(), b[0].truncate(), b[1].truncate())))
            })
            .count()
    };
    for stub in (0..net.lanes.len()).filter(|&i| net.lanes[i].kind == LaneKind::Sidewalk && net.lanes[i].traffic_light.is_some()) {
        let mut chain = vec![stub];
        let mut k = 0;
        while k < chain.len() && chain.len() < 60 {
            let l = chain[k];
            let around: Vec<usize> = net.lanes[l].next.iter().copied().chain(net.prev.get(l).cloned().unwrap_or_default()).collect();
            for o in around {
                if !chain.contains(&o) && net.lanes[o].kind == LaneKind::Sidewalk && net.lanes[o].length() <= 4.0 {
                    chain.push(o);
                }
            }
            k += 1;
        }
        let walks_cut = chain.iter().filter(|&&w| streets.iter().any(|&s| net.walks[s].iter().any(|x| x.0 == w))).count();
        let geo_cut: usize = chain.iter().map(|&w| cuts_geo(w)).sum();
        let in_crossings: Vec<_> = chain.iter().filter_map(|w| js.of_foot_lane.get(w)).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        eprintln!("stub {stub} light {:?}: chain of short pieces {} (cap 60), pieces in walks {}, geometric street hits {}, crossings touched {:?}", net.lanes[stub].traffic_light, chain.len(), walks_cut, geo_cut, in_crossings);
    }
    // any short sidewalk piece geometrically over a street lane but not in any crossing
    let missed: Vec<usize> = (0..net.lanes.len()).filter(|&w| net.lanes[w].kind == LaneKind::Sidewalk && cuts_geo(w) > 0 && !js.of_foot_lane.contains_key(&w)).collect();
    eprintln!("sidewalk lanes over a street lane but in no crossing: {} {:?}", missed.len(), missed);
    let overlap: Vec<usize> = (0..net.lanes.len())
        .filter(|&w| js.nodes[0].crossings.iter().filter(|c| c.foot_lanes.contains(&w)).count() > 1)
        .collect();
    eprintln!("footpath lanes in more than one crossing: {} {:?}", overlap.len(), overlap);
    for w in &overlap {
        let owners: Vec<_> = js.nodes[0].crossings.iter().enumerate().filter(|(_, c)| c.foot_lanes.contains(w)).map(|(i, c)| (i, c.light)).collect();
        eprintln!("  lane {w}: crossings {:?}, lookup says {:?}", owners, js.of_foot_lane.get(w));
    }
}
