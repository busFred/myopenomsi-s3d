//! Round 11 area R (roundabout entry capacity): a saturated entry into a single-lane ring with a
//! circulating stream, driven in the simulation. The entry's flow is held against the HCM 6
//! single-lane roundabout entry capacity, and no circulating car brakes beyond its comfortable
//! brake for an entering one. `cargo test --release -p omsi-traffic --test round11r -- --nocapture`
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

/// A single-lane ring (centre at the origin, radius `R`, counter-clockwise) with one entry, as
/// X10's: the circulating way comes in by a straight feeder (0) round the ring (1), through the
/// junction object's ring piece (2, the right of way) to the merge point M at angle 0; the entry
/// (approach 3, 1000 m, the object's entry piece 4, giving way) runs into M at 40 degrees to the
/// ring; after M the ring goes on (5) and leaves by a straight (6).
fn ring_entry(zebra: bool) -> Network {
    let tangent = |t: f64| (-t.to_radians().sin(), t.to_radians().cos());
    let at = |t: f64| (R * t.to_radians().cos(), R * t.to_radians().sin());
    let (p0, d0) = (at(-150.0), tangent(-150.0));
    let m = at(0.0);
    let dir = (-(40f64.to_radians().sin()), 40f64.to_radians().cos());
    let piece = if zebra { 16.0 } else { 8.0 };
    let e0 = (m.0 - piece * dir.0, m.1 - piece * dir.1);
    let (p6, d6) = (at(60.0), tangent(60.0));
    let mut net = Network::default();
    net.lanes.push(lane(vec![(p0.0 - 100.0 * d0.0, p0.1 - 100.0 * d0.1), p0], 1, 1, 0, 0.0)); // 0 feeder
    net.lanes.push(lane(arc(-150.0, -40.0), 1, 2, 0, 0.0)); // 1 ring
    net.lanes.push(lane(arc(-40.0, 0.0), 2, 9, 0, 192.0)); // 2 ring piece
    net.lanes.push(lane(vec![(e0.0 - 1000.0 * dir.0, e0.1 - 1000.0 * dir.1), e0], 1, 3, 0, 0.0)); // 3 approach
    net.lanes.push(lane(vec![e0, m], 2, 9, 1, 64.0)); // 4 entry piece
    net.lanes.push(lane(arc(0.0, 60.0), 1, 4, 0, 0.0)); // 5 ring on
    net.lanes.push(lane(vec![p6, (p6.0 + 300.0 * d6.0, p6.1 + 300.0 * d6.1)], 1, 5, 0, 0.0)); // 6 away
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
    min_sep: f32,
    /// Entrants still on the approach at the end (the entry was saturated throughout).
    queue_left: usize,
}

impl EntryRun {
    fn hcm_ratio(&self) -> (f32, f32, f32, f32) {
        let per_h = |n: usize| n as f32 * 3600.0 / self.window;
        let (e, v_c) = (per_h(self.entered), per_h(self.circulated));
        // HCM 6 ch. 22 single-lane entry against one circulating lane: c = 1380 exp(-1.02e-3 v_c)
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
    let mut run = EntryRun { entered: 0, circulated: 0, window: secs - warm, over_comfort: f32::NEG_INFINITY, min_sep: f32::INFINITY, queue_left: 0 };
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
        if std::env::var_os("R11R_WATCH").is_some() && ((t * 30.0).round() as i32) % 6 == 0 {
            for (c, g) in &sim.claims {
                let Some(v) = cars.get(c.vehicle) else { continue };
                if entrants.contains(&v.id) && (v.state.lane == 4 || (v.state.lane == 3 && v.state.s > approach - 25.0)) {
                    println!("W {t:.2} #{} l{} s {:.1} v {:.2} why {:?} go {} zip {:?}", v.id, v.state.lane, c.s, c.speed, v.last.why, g.go, g.zipper.map(|z| (cars.get(z.foe).map(|f| f.id), (z.gap * 10.0).round() / 10.0, z.hold.map(|h| (h * 10.0).round() / 10.0), z.stop, (z.arrive * 100.0).round() / 100.0)));
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
            if !entrants.contains(&va.id) && (lane == 2 || lane == 5) {
                if let Some(&v0) = last_v.get(&va.id) {
                    run.over_comfort = run.over_comfort.max((v0 - va.state.speed) / DT - va.params.comfort_brake);
                }
            }
            last_v.insert(va.id, va.state.speed);
        }
        t += DT;
    }
    run.queue_left = cars.iter().filter(|c| c.state.lane == 3).count();
    run
}

/// The band of the entry's flow over the HCM 6 capacity (`EntryRun::hcm_ratio`) the ring's
/// entry is held to: no more than the HCM capacity itself (1.0), and no less than 0.55 - this
/// ring circulates at ~4.8 m/s (its 12.5 m radius at the drivers' bend acceleration), where the
/// car-following model (IDM) lets a car in ahead of a circulating one only ~13 m ahead of it (the
/// room it needs to follow a car starting from rest braking no harder than comfortably,
/// `reservation::follow_room`), a lag of ~3.8 s at the merge, and an entrant from rest passes the
/// merge ~3 s behind the circulating car it let by: ~7 s of gap for the first car against HCM's
/// critical headway of ~5 s (its follow-up, ~2.8 s, is HCM's 2.6 s). At the tip, the entry with a
/// footpath crossing between its waiting place and the ring ran at 0.37.
const HCM_BAND: (f32, f32) = (0.55, 1.0);

fn check(zebra: bool) {
    let run = run_entry(zebra, 6.0, 660.0, 60.0, 11);
    let (e, v_c, cap, ratio) = run.hcm_ratio();
    println!("zebra {zebra}: entering {e:.0}/h circulating {v_c:.0}/h HCM 6 capacity {cap:.0}/h ratio {ratio:.2}; circulating braking beyond comfort {:.2} m/s²; closest boxes {:.2} m; queue left {}", run.over_comfort, run.min_sep, run.queue_left);
    assert!((450.0..=750.0).contains(&v_c), "circulating {v_c:.0}/h, not about 600");
    assert!(run.queue_left > 0, "the entry's queue ran out: not saturated");
    assert!(run.over_comfort <= 0.0, "a circulating car braked {:.2} m/s² beyond its comfortable brake", run.over_comfort);
    assert!(run.min_sep > 0.0, "two bodies came {:.2} m apart", run.min_sep);
    assert!((HCM_BAND.0..=HCM_BAND.1).contains(&ratio), "entry flow {e:.0}/h is {ratio:.2} of the HCM 6 capacity {cap:.0}/h at {v_c:.0}/h circulating");
}

/// [R-1] A circulating stream of about 600 veh/h (on average a car every 6 s) and a saturated entry
/// whose waiting place is at the start of the merge: the entry's flow lies within `HCM_BAND` of
/// the HCM 6 single-lane entry capacity at the measured circulating flow, no circulating car brakes
/// harder than its comfortable brake, and no two bodies touch.
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
