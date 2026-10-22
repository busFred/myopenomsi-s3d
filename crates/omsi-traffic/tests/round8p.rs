//! Round 8 P8 (people and buses): pull back into the lane (P8-4), buses and waiters (P8-1),
//! the pass's sweep (P8-2), the people strip on a bend (P8-5).
use glam::DVec3;
use omsi_sim::traffic::{AiState, LaneBuilder, LaneKind, Network};
use omsi_traffic::{Simulation, Vehicle, VehicleClass, Why};

const DT: f32 = 1.0 / 30.0;

fn lane(pts: &[(f64, f64)]) -> omsi_sim::traffic::Lane {
    LaneBuilder::polyline(pts.iter().map(|p| DVec3::new(p.0, p.1, 0.0)).collect(), LaneKind::Street, 3.5)
}

fn finish(lanes: Vec<omsi_sim::traffic::Lane>) -> Network {
    let mut net = Network { lanes, ..Default::default() };
    net.link(1.5);
    net.build_grid();
    net.compute_conflicts();
    net
}

fn vehicle(id: u64, class: VehicleClass, route: Vec<usize>, s: f32, speed: f32, half_len: f32, half_width: f32) -> Vehicle {
    let mut st = AiState::new(route[0], s, id);
    st.speed = speed;
    st.front = half_len;
    st.rear = half_len;
    st.length = 2.0 * half_len;
    st.max_speed_kmh = 60.0;
    st.route = route;
    let mut v = Vehicle::new(id, class, st, half_width, id * 7919);
    v.fresh = 0.0;
    v
}

fn bus(id: u64, route: Vec<usize>, s: f32, speed: f32) -> Vehicle {
    vehicle(id, VehicleClass::ScheduledBus, route, s, speed, 6.0, 1.25)
}

fn car(id: u64, route: Vec<usize>, s: f32, speed: f32) -> Vehicle {
    vehicle(id, VehicleClass::Car, route, s, speed, 2.25, 0.9)
}

/// A bus standing 2.8 m out in a bay on a straight road east, its service just sent it back
/// (bay 0, a ramp of `ramp` m laid), a car coming in the lane at `v` from `gap` behind the
/// bus's rear: (when the bus moves off, its largest heading off the lane while standing, the
/// least distance between the two, the car's hardest braking).
fn pull_out(gap: f32, v: f32, ramp: f32) -> (f32, f32, f64, f32) {
    let net = finish(vec![lane(&[(0.0, 0.0), (400.0, 0.0)])]);
    let mut sim = Simulation::new(&net, vec![]);
    let mut b = bus(1, vec![0], 200.0, 0.0);
    b.state.lateral = 2.8;
    b.state.lateral_target = 0.0;
    b.external.bay = Some(0.0);
    let odo = b.state.odometer;
    b.state.lateral_ramp = (2.8, 0.0, odo, ramp);
    let c = car(2, vec![0], 200.0 - 6.0 - gap - 2.25, v);
    let mut vs = vec![b, c];
    let (mut min, mut maxdec, mut go_t, mut hd) = (f64::INFINITY, 0.0f32, -1.0f32, 0.0f32);
    let mut prev = v;
    let mut t = 0.0;
    sim.update_positions(&net, &mut vs);
    while t < 20.0 {
        sim.tick(&net, DT, &mut vs);
        sim.update_positions(&net, &mut vs);
        if go_t < 0.0 && vs[0].state.speed > 0.3 {
            go_t = t;
        }
        if go_t < 0.0 {
            hd = hd.max((vs[0].heading - 90.0).abs());
        }
        maxdec = maxdec.max((prev - vs[1].state.speed) / DT);
        prev = vs[1].state.speed;
        min = min.min(vs[0].obb().separation(&vs[1].obb()));
        t += DT;
    }
    (go_t, hd, min, maxdec)
}

/// [P8-4, pass 1 S1] A car at 13.9 m/s 40 m behind is not cut up: the bus waits for it (the
/// gap a lane change accepts in front of it), standing straight in its bay meanwhile, and the
/// car does not brake for it at all; a car standing 5 m behind does not keep the bus in.
#[test]
fn p84_a_bus_leaving_its_bay_waits_for_the_car_coming_and_stands_straight() {
    let (go, hd, min, dec) = pull_out(40.0, 13.9, 12.0);
    eprintln!("40 m at 13.9: goes {go:.1} s, heading off {hd:.1} deg, min sep {min:.2}, car's braking {dec:.2}");
    assert!(go > 2.0, "the bus pulled out in front of the car at once ({go:.1} s)");
    assert!(hd < 1.0, "the bus stood turned {hd:.1} degrees into the lane");
    assert!(min > 0.5 && dec < 1.0, "the car had to brake for the bus: {dec:.2} m/s², {min:.2} m");
    let (go, _, min, _) = pull_out(5.0, 0.0, 12.0);
    assert!(go >= 0.0 && go < 3.0 && min > 0.3, "a car standing behind kept the bus in: goes {go:.1}, {min:.2} m");
}

/// [P8-4] One clock for everybody: a car held standing out of its lane accrues the give-up
/// waiting time like any held car.
#[test]
fn p84_the_hold_runs_the_waiting_clock() {
    let net = finish(vec![lane(&[(0.0, 0.0), (400.0, 0.0)])]);
    let mut sim = Simulation::new(&net, vec![]);
    let mut p = car(1, vec![0], 200.0, 0.0);
    p.state.lateral = 2.5;
    p.state.lateral_target = 0.0;
    let odo = p.state.odometer;
    p.state.lateral_ramp = (2.5, 0.0, odo, 12.0);
    // a van stopped in the lane beside its nose for good (held by its own stop)
    let mut van = car(2, vec![0], 204.0, 0.0);
    van.external.stop_at = Some(0.1);
    let mut vs = vec![p, van];
    let mut t = 0.0;
    while t < 6.0 {
        vs[1].external.stop_at = Some(vs[1].state.front + 0.1);
        sim.tick(&net, DT, &mut vs);
        sim.update_positions(&net, &mut vs);
        t += DT;
    }
    assert_eq!(vs[0].last.why, Why::PullOut);
    assert!(vs[0].waiting > 5.0, "waiting clock {:.1}", vs[0].waiting);
}
