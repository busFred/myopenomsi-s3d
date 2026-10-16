//! The driver: a named parameter set per vehicle class (SUMO's `vType`), seeded jitter
//! for personality, and *impatience*, the one escape hatch. A driver who waits grows
//! impatient; impatience shrinks the gaps they accept and the yielding they do, so queues
//! and deadlocks resolve themselves without special cases.

use omsi_sim::traffic::AiState;

/// What kind of vehicle the driver is in: sets the preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum VehicleClass {
    Car,
    Taxi,
    Bus,
    Truck,
    /// A timetable bus with a route and stops.
    ScheduledBus,
    Rail,
    Air,
}

impl VehicleClass {
    /// From the vehicle's `[ai_veh_type]` (0 car, 1 taxi, 2 bus, 3 truck; -1 a timetable
    /// bus) and the lane kind it drives on.
    pub fn from_code(veh_type: i32, kind: omsi_sim::traffic::LaneKind) -> VehicleClass {
        use omsi_sim::traffic::LaneKind;
        match kind {
            LaneKind::Rail => VehicleClass::Rail,
            LaneKind::Air => VehicleClass::Air,
            _ => match veh_type {
                -1 => VehicleClass::ScheduledBus,
                1 => VehicleClass::Taxi,
                2 => VehicleClass::Bus,
                3 => VehicleClass::Truck,
                _ => VehicleClass::Car,
            },
        }
    }

    pub fn is_bus(self) -> bool {
        matches!(self, VehicleClass::Bus | VehicleClass::ScheduledBus)
    }

    pub fn heavy(self) -> bool {
        matches!(self, VehicleClass::Bus | VehicleClass::ScheduledBus | VehicleClass::Truck)
    }
}

/// Everything about how a driver drives, in one place. Units: metres, seconds, m/s².
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DriverParams {
    /// Time gap kept to the car ahead (IDM `tau`).
    pub tau: f32,
    /// Standing gap to the car ahead (IDM `s0`).
    pub min_gap: f32,
    /// Pulling away (IDM `a`).
    pub accel: f32,
    /// Comfortable braking (IDM `b`).
    pub comfort_brake: f32,
    /// Wants the limit times this.
    pub speed_factor: f32,
    /// Sideways acceleration accepted in a bend.
    pub lat_accel: f32,
    /// Seconds from the way clearing to moving off.
    pub reaction: f32,
    /// Gap in the crossing traffic accepted at a junction (s), before impatience.
    pub gap_accept: f32,
    /// Gap accepted in the lane changed into (s, to the car behind there).
    pub lc_gap: f32,
    /// Lane-change motivation weights (LMRS / LC2013): the route needing it, a faster
    /// lane, letting a neighbour in, keeping right.
    pub lc_strategic: f32,
    pub lc_speed_gain: f32,
    pub lc_cooperate: f32,
    pub lc_keep_right: f32,
    /// How fast impatience grows while waiting (per second) and where it caps.
    pub impatience_rate: f32,
    pub impatience_max: f32,
    /// Patience at a yellow light: go through when the stop would need more than this
    /// braking (m/s²).
    pub yellow_brake: f32,
}

impl DriverParams {
    /// The preset of a class, before jitter.
    pub fn preset(class: VehicleClass) -> DriverParams {
        let car = DriverParams {
            tau: 1.3,
            min_gap: 2.0,
            accel: 1.8,
            comfort_brake: 2.0,
            speed_factor: 1.0,
            lat_accel: 2.2,
            reaction: 0.8,
            gap_accept: 4.0,
            lc_gap: 1.5,
            lc_strategic: 1.0,
            lc_speed_gain: 0.6,
            lc_cooperate: 0.5,
            lc_keep_right: 0.45,
            impatience_rate: 1.0 / 30.0,
            impatience_max: 1.0,
            yellow_brake: 2.5,
        };
        match class {
            VehicleClass::Car => car,
            VehicleClass::Taxi => DriverParams { tau: 1.0, accel: 2.2, speed_factor: 1.08, gap_accept: 3.2, impatience_rate: 1.0 / 20.0, ..car },
            VehicleClass::Bus | VehicleClass::ScheduledBus => DriverParams { tau: 1.6, min_gap: 2.5, accel: 1.0, comfort_brake: 1.5, lat_accel: 1.4, gap_accept: 5.0, lc_gap: 2.0, lc_speed_gain: 0.2, yellow_brake: 1.8, ..car },
            VehicleClass::Truck => DriverParams { tau: 1.8, min_gap: 3.0, accel: 0.8, comfort_brake: 1.5, speed_factor: 0.95, lat_accel: 1.3, gap_accept: 5.5, lc_gap: 2.5, lc_speed_gain: 0.2, yellow_brake: 1.8, ..car },
            VehicleClass::Rail => DriverParams { tau: 3.0, min_gap: 10.0, accel: 0.8, comfort_brake: 0.8, lat_accel: 1.0, gap_accept: 0.0, impatience_rate: 0.0, yellow_brake: 1.0, ..car },
            VehicleClass::Air => DriverParams { tau: 5.0, min_gap: 50.0, accel: 1.0, comfort_brake: 0.5, lat_accel: 1.0, gap_accept: 0.0, impatience_rate: 0.0, ..car },
        }
    }

    /// A driver of `class` with the personality drawn from `seed`: one in five is quick,
    /// one in five slow, the rest in between, in every trait at once (a hurried driver
    /// also keeps less distance and accepts smaller gaps).
    pub fn seeded(class: VehicleClass, seed: u64) -> DriverParams {
        let p = Self::preset(class);
        if matches!(class, VehicleClass::Rail | VehicleClass::Air) {
            return p;
        }
        let mut x = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut unit = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 11) as f32 / (1u64 << 53) as f32
        };
        // hurry in -1 (slow) .. +1 (quick), bell-shaped from three draws
        let hurry = (unit() + unit() + unit()) / 1.5 - 1.0;
        let j = |v: f32, span: f32| v * (1.0 + hurry * span);
        DriverParams {
            tau: j(p.tau, -0.3),
            min_gap: j(p.min_gap, -0.25),
            accel: j(p.accel, 0.3),
            comfort_brake: j(p.comfort_brake, 0.2),
            speed_factor: j(p.speed_factor, 0.1),
            lat_accel: j(p.lat_accel, 0.2),
            reaction: j(p.reaction, -0.5),
            gap_accept: j(p.gap_accept, -0.3),
            lc_gap: j(p.lc_gap, -0.3),
            // (jittered a little only: the desire has to clear the planner's threshold
            // for every driver, or a slow driver sits behind a crawler for ever)
            lc_speed_gain: j(p.lc_speed_gain, 0.2),
            impatience_rate: j(p.impatience_rate, 0.5),
            ..p
        }
    }

    /// Copy the parameters the operational model reads into the car's state.
    pub fn apply(&self, st: &mut AiState) {
        st.headway = self.tau;
        st.min_gap = self.min_gap;
        st.accel = self.accel;
        st.decel = self.comfort_brake;
        st.desire = self.speed_factor;
        st.lat_accel = self.lat_accel;
        st.reaction = self.reaction;
        st.accept_gap = self.gap_accept;
    }
}

/// How long a driver has been held, turned into willingness to push: 0 fresh, 1 fed up.
/// SUMO's `impatience`: it scales the accepted gaps and lets a driver who has waited for
/// ever take a turn they would otherwise give.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct Impatience {
    pub value: f32,
    /// Seconds standing in the current hold.
    pub waited: f32,
}

impl Impatience {
    /// Tick: grows while `held`, cools down twice as fast while moving.
    pub fn update(&mut self, p: &DriverParams, dt: f32, held: bool) {
        if held {
            self.waited += dt;
            self.value = (self.value + p.impatience_rate * dt).min(p.impatience_max);
        } else {
            self.waited = 0.0;
            self.value = (self.value - 2.0 * p.impatience_rate * dt).max(0.0);
        }
    }

    /// The gap (s) accepted now: the driver's own, down to a third of it when fed up.
    pub fn gap(&self, base: f32) -> f32 {
        base * (1.0 - 0.67 * self.value)
    }

    /// A driver this impatient takes a way they would yield on, if the foe is still
    /// farther than this many seconds away.
    pub fn claims_after(&self) -> Option<f32> {
        if self.value >= 0.999 {
            Some(2.5)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn personalities_vary_in_step() {
        let a = DriverParams::seeded(VehicleClass::Car, 1);
        let b = DriverParams::seeded(VehicleClass::Car, 2);
        let c = DriverParams::seeded(VehicleClass::Car, 3);
        assert!(a != b || b != c);
        for p in [a, b, c] {
            // a quicker driver keeps a shorter gap, and the other way round
            let quick = p.accel > DriverParams::preset(VehicleClass::Car).accel;
            assert_eq!(quick, p.tau < DriverParams::preset(VehicleClass::Car).tau);
            assert!(p.tau > 0.5 && p.tau < 2.5);
        }
    }

    #[test]
    fn impatience_grows_waiting_and_shrinks_the_gap() {
        let p = DriverParams::preset(VehicleClass::Car);
        let mut imp = Impatience::default();
        assert_eq!(imp.gap(4.0), 4.0);
        for _ in 0..600 {
            imp.update(&p, 0.1, true);
        }
        assert!(imp.value > 0.99);
        assert!(imp.gap(4.0) < 1.4);
        assert!(imp.claims_after().is_some());
        for _ in 0..100 {
            imp.update(&p, 0.1, false);
        }
        assert!(imp.value < 0.7);
        assert!(imp.claims_after().is_none());
    }
}
