//! May this vehicle use that lane: one rule for the population, the way on at a fork, a
//! lane change, a pass on the oncoming lane and the lane beside. The maps say it in
//! several ways at once (BeijingCity keeps the normal cars off its bus lanes with a
//! `trafficdensity 0` for their group alone and no `no_cars`; Grundorf uses `no_cars` and
//! `bus`; cycle lanes are 1 m wide), and every stage used to ask a different subset.

use crate::sim::Vehicle;
use omsi_sim::traffic::{Lane, LaneKind};

/// Why a lane is closed to a vehicle (for the debug overlay).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Closed {
    /// Not a street (a footpath, a rail, a flight path).
    Kind,
    /// `[rule] no_cars`, `bus`, `trucks` by the vehicle's `[ai_veh_type]`.
    Rules,
    /// No random traffic of this vehicle's group drives here (`trafficdensity 0`).
    Density,
}

/// Is `lane` open to `v` for driving along? A timetable vehicle on its route is open
/// anywhere its route goes; this answers for the driver's own choices.
pub fn open_to(lane: &Lane, v: &Vehicle) -> Result<(), Closed> {
    let st = &v.state;
    let kind_ok = match lane.kind {
        LaneKind::Street => !v.is_rail() && v.class != crate::driver::VehicleClass::Air,
        LaneKind::Rail => v.is_rail(),
        LaneKind::Air => v.class == crate::driver::VehicleClass::Air,
        LaneKind::Sidewalk => false,
    };
    if !kind_ok {
        return Err(Closed::Kind);
    }
    if !lane.allows(st.veh_type) {
        return Err(Closed::Rules);
    }
    // a scheduled bus goes where its route goes; everybody else needs the lane to carry
    // traffic of their group
    if st.route.is_empty() {
        // (a vehicle of no group is a normal car: where the path names its groups, the
        // first group's share counts, not the most any group gets - `density` is the
        // greatest over all groups, so that a lane stays open to whoever is asked for)
        let density = match st.traffic_pool.as_ref() {
            Some((pool, defaults)) => lane.pool_density(defaults, *pool),
            None if lane.group_density.is_empty() => lane.density,
            None => lane.pool_density(&[1], 0),
        };
        if density <= 0.0 {
            return Err(Closed::Density);
        }
    }
    // (no test on the path's width: a tenth of the stock maps' street paths are drawn a
    // metre wide where the lane is a full one - Spandau's ten-lane Altstädter Ring is
    // all 1 m paths - and BeijingCity's bus lane is as wide as its cycle lane. What keeps
    // a car off a cycle lane beyond a strip is the strip: `Lane::beside_each_other`.)
    Ok(())
}

/// Open, as a plain yes or no.
pub fn is_open(lane: &Lane, v: &Vehicle) -> bool {
    open_to(lane, v).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::VehicleClass;
    use glam::DVec3;
    use omsi_sim::traffic::{AiState, LaneBuilder};

    fn car() -> Vehicle {
        let mut st = AiState::new(0, 0.0, 1);
        st.veh_type = 0;
        Vehicle::new(1, VehicleClass::Car, st, 0.9, 1)
    }

    fn lane(width: f32) -> Lane {
        LaneBuilder::polyline(vec![DVec3::ZERO, DVec3::new(50.0, 0.0, 0.0)], LaneKind::Street, width)
    }

    #[test]
    fn a_bus_lane_by_group_density_is_closed_to_a_normal_car() {
        let mut l = lane(3.0);
        let c = car();
        assert_eq!(open_to(&l, &c), Ok(()));
        // BeijingCity: no rule but the normal cars' density set to 0
        l.density = 0.0;
        assert_eq!(open_to(&l, &c), Err(Closed::Density));
        // a pooled car reads its own group
        let mut pooled = car();
        pooled.state.traffic_pool = Some((0, std::sync::Arc::new(vec![1, 2])));
        l.density = 1.0;
        l.group_density = vec![(0, 0.0), (1, 1.0)];
        assert_eq!(open_to(&l, &pooled), Err(Closed::Density));
        pooled.state.traffic_pool = Some((1, std::sync::Arc::new(vec![1, 2])));
        assert_eq!(open_to(&l, &pooled), Ok(()));
    }

    #[test]
    fn a_narrow_path_is_still_a_lane_and_a_route_opens_a_closed_one() {
        // (Spandau draws full lanes as 1 m paths)
        let l = lane(1.0);
        let mut c = car();
        assert_eq!(open_to(&l, &c), Ok(()));
        c.state.route = vec![0];
        let mut closed = lane(3.0);
        closed.density = 0.0;
        assert_eq!(open_to(&closed, &c), Ok(()));
        closed.no_cars = true;
        assert_eq!(open_to(&closed, &c), Err(Closed::Rules));
    }
}
