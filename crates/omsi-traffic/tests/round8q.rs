//! Round 8 area Q8 (queues and lane use): the packed-queue escape (Q8-1), the bypass inside a
//! junction piece (Q8-2), LC2013's keep-right layer (Q8-3), J7's stop placement after attribution
//! (Q8-4).
use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKey, LaneKind, Network};
use omsi_traffic::{Simulation, Vehicle, VehicleClass};

const DT: f32 = 1.0 / 30.0;

fn lane(pts: &[(f64, f64)], kind: LaneKind, width: f32) -> omsi_sim::traffic::Lane {
    LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), kind, width)
}
fn road(pts: &[(f64, f64)], id: i64, path: u16) -> omsi_sim::traffic::Lane {
    let mut l = lane(pts, LaneKind::Street, 3.0);
    l.source = 1;
    l.key = Some(LaneKey { tile: (0, 0), id, path });
    l
}
fn finish(mut net: Network) -> Network {
    net.link(1.5);
    net.compute_conflicts();
    net
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
fn run(net: &Network, sim: &mut Simulation, cars: &mut Vec<Vehicle>, secs: f32, mut each: impl FnMut(f32, &Simulation, &[Vehicle])) {
    let mut t = 0.0f32;
    while t < secs {
        sim.tick(net, DT, cars);
        for &i in &sim.removed {
            cars.remove(i);
        }
        each(t, sim, cars);
        t += DT;
    }
}

/// Two same-direction lanes, 0 left at y=0 and 1 right at y=-3.5, 0..210 m, running on into 2
/// and 3 to 600 m.
fn two_lanes() -> Network {
    let mut net = Network::default();
    let mut a = road(&[(0.0, 0.0), (210.0, 0.0)], 1, 0);
    a.offset = 0.0;
    let mut b = road(&[(0.0, -3.5), (210.0, -3.5)], 1, 1);
    b.offset = 3.5;
    b.drivable = (1.5, 1.5);
    let mut a2 = road(&[(210.0, 0.0), (600.0, 0.0)], 2, 0);
    a2.offset = 0.0;
    let mut b2 = road(&[(210.0, -3.5), (600.0, -3.5)], 2, 1);
    b2.offset = 3.5;
    b2.drivable = (1.5, 1.5);
    net.lanes.extend([a, b, a2, b2]);
    finish(net)
}


/// Car 1 on the left lane at 12 m/s; `slow`: a car 45 m ahead of it on the right lane at that
/// speed. (When it began to move onto the right lane, the most its keep-right wish reached, and
/// whether it was past the slower car then.)
fn keep_right_run(slow: Option<f32>, secs: f32) -> (Option<f32>, f32, bool) {
    let net = two_lanes();
    let mut sim = Simulation::new(&net, vec![]);
    let mut me = car(1, 0, 20.0, 12.0);
    me.state.change_cooldown = 0.0;
    me.state.max_speed_kmh = 43.2;
    let mut cars = vec![me];
    if let Some(v) = slow {
        let mut o = car(2, 1, 69.0, v);
        o.state.max_speed_kmh = v * 3.6;
        o.state.change_cooldown = 100.0;
        cars.push(o);
    }
    let (mut right, mut most, mut past) = (None, 0.0f32, true);
    run(&net, &mut sim, &mut cars, secs, |t, _, cars| {
        let Some(me) = cars.iter().find(|c| c.id == 1) else { return };
        most = most.max(me.keep_right);
        if right.is_none() && (me.state.lane == 1 || me.state.lane == 3 || me.state.change.is_some_and(|c| c.to == 1 || c.to == 3)) {
            right = Some(t);
            past = cars.iter().find(|c| c.id == 2).is_none_or(|o| o.pos.x + 2.0 < me.pos.x - 2.0);
        }
    });
    (right, most, past)
}

#[test]
fn q83_keep_right_is_lc2013s_accumulated_wish_not_a_distance_gate() {
    // SUMO LC2013's keepRight layer: the wish accumulates by the share of the acceptance time the
    // driver could keep its speed on the lane beside, over KEEP_RIGHT_TIME; the change comes once
    // it reaches the threshold - not at once, and not while a slower car there would soon have to
    // be overtaken (round 5 asked for 70 m free there and 60 m to the own lead)
    let (free, _, _) = keep_right_run(None, 30.0);
    let (slow, most, past) = keep_right_run(Some(6.0), 30.0);
    eprintln!("keep right: free lane at {free:?}; slower car 45 m ahead there: {slow:?} (wish {most:.2}, past it {past})");
    let t = free.expect("never kept right onto a free lane");
    assert!(t >= omsi_traffic::plan::KEEP_RIGHT_TIME, "kept right at {t:.1} s, before the wish could accumulate");
    // (it overtakes the slower car first, then keeps right in front of it)
    assert!(slow.is_some() && past, "kept right before passing the slower car ({slow:?})");
}
