//! Round 11 area R (roundabout entry capacity): a saturated entry into a single-lane ring with a
//! circulating stream, driven in the simulation. The entry's flow is held against the HCM 6
//! single-lane roundabout entry capacity, no circulating car brakes beyond its comfortable brake
//! for an entering one, and no entrant brakes beyond the brake inside the object. `cargo test --release -p omsi-traffic --test round11r -- --nocapture`
use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network};
use omsi_traffic::{Simulation, Vehicle, VehicleClass};

const DT: f32 = 1.0 / 30.0;
/// X10's ring at lane 1637: radius 12.5 m.
const R: f64 = 12.5;
/// The least headway (s) of the fed circulating stream: about the car-following model's own at the
/// ring's speed (tau 1.3 s plus a car and its standing gap at ~5 m/s).
const MIN_HEADWAY: f32 = 2.5;

fn lane(pts: Vec<(f64, f64)>, source: u32, id: i64, path: u16, priority: f32) -> omsi_sim::traffic::Lane {
    let mut l = LaneBuilder::polyline(pts.into_iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), LaneKind::Street, 3.0);
    l.source = source;
    l.key = Some(LaneKey { tile: (0, 0), id, path });
    l.priority = priority;
    l
}

/// The ring's points from angle `a` to `b` (degrees, counter-clockwise: right-hand traffic).
fn arc(a: f64, b: f64) -> Vec<(f64, f64)> {
    let n = ((b - a) / 2.0).ceil() as usize;
    (0..=n).map(|i| (a + (b - a) * i as f64 / n as f64).to_radians()).map(|t| (R * t.cos(), R * t.sin())).collect()
}

/// A cubic Bezier from `p0` to `p1` leaving `p0` along `d0` and arriving at `p1` along `d1` (unit
/// vectors), its inner control points `k` m along them: `n` + 1 points.
fn bezier(p0: (f64, f64), d0: (f64, f64), p1: (f64, f64), d1: (f64, f64), k: f64, n: usize) -> Vec<(f64, f64)> {
    let (c0, c1) = ((p0.0 + k * d0.0, p0.1 + k * d0.1), (p1.0 - k * d1.0, p1.1 - k * d1.1));
    (0..=n)
        .map(|i| {
            let t = i as f64 / n as f64;
            let (a, b, c, d) = ((1.0 - t).powi(3), 3.0 * (1.0 - t).powi(2) * t, 3.0 * (1.0 - t) * t * t, t.powi(3));
            (a * p0.0 + b * c0.0 + c * c1.0 + d * p1.0, a * p0.1 + b * c0.1 + c * c1.1 + d * p1.1)
        })
        .collect()
}

/// A single-lane ring (centre at the origin, radius `R`, counter-clockwise) with one entry, as
/// X10's. Its legs are radial and cross nothing, as a roundabout's legs: the circulating way comes
/// in along the radius at -150 degrees (feeder 0, 100 m, curving 12 m from the ring into it at
/// -120 degrees), round the ring (1), through the junction object's ring piece (2, the right of
/// way) to the merge point M at angle 0; the entry (approach 3, 1000 m, the object's entry piece
/// 4, giving way) runs into M at 40 degrees to the ring; after M the ring goes on (5) and leaves at
/// 60 degrees, curving out to the radius at 90 degrees (6, 300 m out).
fn ring_entry(zebra: bool) -> Network {
    let tangent = |t: f64| (-t.to_radians().sin(), t.to_radians().cos());
    let radial = |t: f64| (t.to_radians().cos(), t.to_radians().sin());
    let at = |t: f64| (R * t.to_radians().cos(), R * t.to_radians().sin());
    let m = at(0.0);
    let dir = (-(40f64.to_radians().sin()), 40f64.to_radians().cos());
    let piece = if zebra { 16.0 } else { 8.0 };
    let e0 = (m.0 - piece * dir.0, m.1 - piece * dir.1);
    // (the feeder: inwards along the radius at -150 degrees to 12 m off the ring, then into it at -120)
    let u_in = radial(-150.0);
    let p1 = ((R + 12.0) * u_in.0, (R + 12.0) * u_in.1);
    let mut feeder = vec![(p1.0 + 100.0 * u_in.0, p1.1 + 100.0 * u_in.1)];
    feeder.extend(bezier(p1, (-u_in.0, -u_in.1), at(-120.0), tangent(-120.0), 6.0, 16));
    // (the way out: out of the ring at 60 degrees to 12 m off it on the radius at 90, then along it)
    let u_out = radial(90.0);
    let pa = ((R + 12.0) * u_out.0, (R + 12.0) * u_out.1);
    let mut away = bezier(at(60.0), tangent(60.0), pa, u_out, 6.0, 16);
    away.push((pa.0 + 300.0 * u_out.0, pa.1 + 300.0 * u_out.1));
    let mut net = Network::default();
    net.lanes.push(lane(feeder, 1, 1, 0, 0.0)); // 0 feeder
    net.lanes.push(lane(arc(-120.0, -40.0), 1, 2, 0, 0.0)); // 1 ring
    net.lanes.push(lane(arc(-40.0, 0.0), 2, 9, 0, 192.0)); // 2 ring piece
    net.lanes.push(lane(vec![(e0.0 - 1000.0 * dir.0, e0.1 - 1000.0 * dir.1), e0], 1, 3, 0, 0.0)); // 3 approach
    net.lanes.push(lane(vec![e0, m], 2, 9, 1, 64.0)); // 4 entry piece
    net.lanes.push(lane(arc(0.0, 60.0), 1, 4, 0, 0.0)); // 5 ring on
    net.lanes.push(lane(away, 1, 5, 0, 0.0)); // 6 away
    if zebra {
        // 7: a footpath across the entry 9 m before M, as X10's at lane 7269
        let c = (m.0 - 9.0 * dir.0, m.1 - 9.0 * dir.1);
        let mut w = LaneBuilder::polyline(vec![DVec3::new(c.0 - 4.0 * dir.1, c.1 + 4.0 * dir.0, 0.0), DVec3::new(c.0 + 4.0 * dir.1, c.1 - 4.0 * dir.0, 0.0)], LaneKind::Sidewalk, 1.5);
        w.source = 2;
        w.key = Some(LaneKey { tile: (0, 0), id: 9, path: 2 });
        net.lanes.push(w);
    }
    net.link(1.5);
    net.compute_conflicts();
    net
}

fn car(id: u64, lane: usize, s: f32, speed: f32) -> Vehicle {
    let mut st = AiState::new(lane, s, id);
    st.speed = speed;
    st.front = 2.2;
    st.rear = 2.2;
    st.length = 4.4;
    st.max_speed_kmh = 100.0;
    st.veh_type = 0;
    let mut v = Vehicle::new(id, VehicleClass::Car, st, 0.9, id * 7919);
    v.fresh = 0.0;
    v
}

struct EntryRun {
    /// Cars into the ring after M from the entry, from the ring, over the counted window (s).
    entered: usize,
    circulated: usize,
    window: f32,
    /// The hardest braking of a circulating car on the ring piece or after M beyond its own
    /// comfortable brake (m/s², positive: harder), and the closest two boxes came (m).
    over_comfort: f32,
    /// The hardest braking of an entrant on its approach or entry piece beyond the brake inside
    /// the object, 1.5 times its comfortable one (`Claim::brake`: the most the resolver reckons it
    /// stops with there), m/s²; nothing is excused, a stop a car cannot make gently is made later.
    entrant_over: f32,
    over_worst: String,
    entrant_worst: String,
    min_sep: f32,
    /// Entrants still on the approach at the end (the entry was saturated throughout).
    queue_left: usize,
}

impl EntryRun {
    fn hcm_ratio(&self) -> (f32, f32, f32, f32) {
        let per_h = |n: usize| n as f32 * 3600.0 / self.window;
        let (e, v_c) = (per_h(self.entered), per_h(self.circulated));
        // HCM 6 single-lane entry against one circulating lane (ALDOT manual Eq. 2.3): c = 1380 exp(-1.02e-3 v_c)
        let cap = 1380.0 * (-1.02e-3 * v_c).exp();
        (e, v_c, cap, e / cap)
    }
}

/// The ring with a circulating car fed in on average every `headway` s (shifted exponential
/// headways from `seed`, when the feeder's start is clear)
/// and the entry's approach full of standing cars (one every 7 m from its end): `secs` s driven,
/// counted from `warm` s on.
fn run_entry(zebra: bool, headway: f32, secs: f32, warm: f32, seed: u64) -> EntryRun {
    let net = ring_entry(zebra);
    let mut sim = Simulation::new(&net, vec![]);
    // (the legs cross nothing: the junction object is the ring's only junction)
    assert_eq!(sim.junctions.nodes.len(), 1, "the ring's legs make junctions of their own: {:?}", sim.junctions.nodes.iter().map(|n| n.movements.len()).collect::<Vec<_>>());
    let mut cars: Vec<Vehicle> = Vec::new();
    let approach = net.lanes[3].length();
    let mut next_id = 1u64;
    let mut s = approach - 2.5;
    while s > 5.0 {
        cars.push(car(next_id, 3, s, 0.0));
        next_id += 1;
        s -= 7.0;
    }
    let entrants: std::collections::HashSet<u64> = cars.iter().map(|c| c.id).collect();
    let mut last_lane: std::collections::HashMap<u64, usize> = cars.iter().map(|c| (c.id, c.state.lane)).collect();
    let mut last_v: std::collections::HashMap<u64, f32> = std::collections::HashMap::new();
    let (mut t, mut next_spawn) = (0.0f32, 0.0f32);
    let mut rng = seed;
    let mut run = EntryRun { entered: 0, circulated: 0, window: secs - warm, over_comfort: f32::NEG_INFINITY, entrant_over: f32::NEG_INFINITY, over_worst: String::new(), entrant_worst: String::new(), min_sep: f32::INFINITY, queue_left: 0 };
    while t < secs {
        if t >= next_spawn && !cars.iter().any(|c| c.state.lane == 0 && c.state.s < 12.0) {
            cars.push(car(next_id, 0, 3.0, 13.9));
            next_id += 1;
            // (a shifted exponential headway: `MIN_HEADWAY` plus an exponential of mean
            // `headway - MIN_HEADWAY`, the circulating stream HCM's gap-acceptance model assumes)
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((rng >> 11) as f64 / (1u64 << 53) as f64).max(1e-9) as f32;
            next_spawn = t + MIN_HEADWAY - (headway - MIN_HEADWAY) * u.ln();
        }
        sim.tick(&net, DT, &mut cars);
        for &i in sim.removed.iter().rev() {
            cars.remove(i);
        }
        if std::env::var_os("R11R_WATCH").is_some() && ((t * 30.0).round() as i32) % std::env::var("R11R_EVERY").ok().and_then(|e| e.parse().ok()).unwrap_or(6) == 0 {
            for (c, g) in &sim.claims {
                let Some(v) = cars.get(c.vehicle) else { continue };
                if entrants.contains(&v.id) && (v.state.lane == 4 || (v.state.lane == 3 && v.state.s > approach - 25.0)) {
                    println!("W {t:.2} #{} l{} s {:.1} v {:.2} acc {:.2} lead {:?} stop {:?} why {:?} go {} zip {:?}", v.id, v.state.lane, c.s, c.speed, v.state.accel, v.last.lead.map(|l| (l.gap, l.speed)), v.last.stop, v.last.why, g.go, g.zipper.map(|z| (cars.get(z.foe).map(|f| f.id), (z.gap * 10.0).round() / 10.0, z.hold.map(|h| (h * 10.0).round() / 10.0), z.stop, (z.arrive * 100.0).round() / 100.0)));
                }
            }
        }
        // (the bodies near the merge: within 30 m of M)
        let near: Vec<&Vehicle> = cars.iter().filter(|c| (c.pos.truncate() - glam::DVec2::new(R, 0.0)).length() < 30.0).collect();
        for (a, va) in near.iter().enumerate() {
            for vb in near.iter().skip(a + 1) {
                run.min_sep = run.min_sep.min(va.obb().separation(&vb.obb()) as f32);
            }
        }
        for va in cars.iter() {
            let lane = va.state.lane;
            if lane == 5 && last_lane.get(&va.id) != Some(&5) && std::env::var_os("R11R_EVENTS").is_some() {
                println!("EV {t:.2} {} {} v {:.2}", if entrants.contains(&va.id) { "E" } else { "C" }, va.id, va.state.speed);
            }
            if t >= warm && lane == 5 && last_lane.get(&va.id) != Some(&5) {
                if entrants.contains(&va.id) {
                    run.entered += 1;
                } else {
                    run.circulated += 1;
                }
            }
            last_lane.insert(va.id, lane);
            if !entrants.contains(&va.id) && (1..=2).contains(&lane) || !entrants.contains(&va.id) && lane == 5 {
                if let Some(&v0) = last_v.get(&va.id) {
                    let over = (v0 - va.state.speed) / DT - va.params.comfort_brake;
                    if over > run.over_comfort {
                        run.over_worst = format!("t {t:.2} #{} lane {} s {:.1} v {v0:.2} -> {:.2} why {:?} lead {:?}", va.id, lane, va.state.s, va.state.speed, va.last.why, va.last.lead_id);
                    }
                    run.over_comfort = run.over_comfort.max(over);
                }
            }
            if entrants.contains(&va.id) {
                if let Some(&v0) = last_v.get(&va.id) {
                    let over = (v0 - va.state.speed) / DT - 1.5 * va.state.decel;
                    if (3..=4).contains(&va.state.lane) && over > run.entrant_over {
                        run.entrant_over = over;
                        run.entrant_worst = format!("t {t:.2} #{} lane {} s {:.1} v {v0:.2} -> {:.2} why {:?}", va.id, va.state.lane, va.state.s, va.state.speed, va.last.why);
                    }
                }
            }
            last_v.insert(va.id, va.state.speed);
        }
        t += DT;
    }
    run.queue_left = cars.iter().filter(|c| c.state.lane == 3).count();
    run
}

/// The extra critical headway (s) the car-following model (IDM) asks of this ring over HCM 6's
/// drivers, and so the lower bound of the entry's flow over the HCM capacity: the HCM 6 single-lane
/// entry model c = 1380 exp(-1.02e-3 v_c) is c = (3600/tf) exp(-v_c (tc - tf/2)/3600) with
/// follow-up tf = 2.6 s and critical headway tc = 5.0 s (A = 3600/tf, B = (tc - tf/2)/3600; ALDOT
/// Roundabout Planning, Design, and Operations Manual (Dec 2015), Table 2.3 and Equation 2.3,
/// https://www.dot.state.al.us/publications/Design/pdf/TrafficSafetyOp/RoundaboutPlanningDesignOpManual.pdf).
/// A longer critical headway tc + d scales it by exp(-v_c d / 3600). Here
/// d = 4.1 s: this ring circulates at ~4.8 m/s (sqrt of the drivers' bend acceleration times its
/// 12.5 m radius), where the IDM lets a car in ahead of a circulating one only if that is
/// `reservation::follow_room` behind it, s* sqrt(a/b) with s* = s0 + v T + v Δv / (2 sqrt(a b))
/// = 13.6 m for a = 1.8, b = 2.0 m/s², s0 = 2 m, T = 1.3 s and an entrant from rest: 2.8 s at
/// 4.8 m/s; and the entrant itself keeps its time gap T = 1.3 s behind the car it lets by. The
/// flow is held to between exp(-v_c d / 3600) of HCM's capacity and HCM's capacity itself.
const IDM_LAG: f32 = 2.8 + 1.3;

fn check(zebra: bool) {
    let seed = std::env::var("R11R_SEED").ok().and_then(|e| e.parse().ok()).unwrap_or(11);
    let run = run_entry(zebra, 6.0, 660.0, 60.0, seed);
    let (e, v_c, cap, ratio) = run.hcm_ratio();
    println!("zebra {zebra}: entering {e:.0}/h circulating {v_c:.0}/h HCM 6 capacity {cap:.0}/h ratio {ratio:.2}; circulating braking beyond comfort {:.2} m/s²; closest boxes {:.2} m; queue left {}", run.over_comfort, run.min_sep, run.queue_left);
    println!("worst circulating {}", run.over_worst);
    println!("worst entrant braking beyond 1.5 x comfort {:.2} m/s²: {}", run.entrant_over, run.entrant_worst);
    assert!((450.0..=750.0).contains(&v_c), "circulating {v_c:.0}/h, not about 600");
    assert!(run.queue_left > 0, "the entry's queue ran out: not saturated");
    assert!(run.over_comfort <= 0.0, "a circulating car braked {:.2} m/s² beyond its comfortable brake", run.over_comfort);
    assert!(run.entrant_over <= 0.01, "an entrant braked {:.2} m/s² beyond the brake inside the object: {}", run.entrant_over, run.entrant_worst);
    assert!(run.min_sep > 0.0, "two bodies came {:.2} m apart", run.min_sep);
    let lowest = (-v_c * IDM_LAG / 3600.0).exp();
    assert!((lowest..=1.0).contains(&ratio), "entry flow {e:.0}/h is {ratio:.2} of the HCM 6 capacity {cap:.0}/h at {v_c:.0}/h circulating, outside {lowest:.2}..1");
}

/// [R-1] A circulating stream of about 600 veh/h (on average a car every 6 s) and a saturated entry
/// whose waiting place is at the start of the merge: the entry's flow lies within `IDM_LAG`'s band of
/// the HCM 6 single-lane entry capacity at the measured circulating flow, no circulating car brakes
/// harder than its comfortable brake, no entrant harder than the brake inside the object, and no two bodies touch.
#[test]
fn r11r_a_saturated_ring_entry_admits_the_hcm_capacity_and_no_circulating_car_brakes_hard() {
    check(false);
}

/// [R-1] The same with a footpath across the entry 9 m before the merge point, as X10's ring
/// 1637: the queue's head waits short of the crossing, ~11 m back from the merge stretch, and
/// moves off while the circulating car it lets by is still passing, as its own arrival from rest
/// comes no sooner than its turn (`ZipCap::hold`); the band and the safety as above.
#[test]
fn r11r_an_entry_waiting_back_from_the_merge_moves_off_while_the_circulating_car_passes() {
    check(true);
}
