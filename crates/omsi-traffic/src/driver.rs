//! The driver: a named parameter set per vehicle class (SUMO's `vType`), seeded jitter
//! for personality, and *impatience*, the one escape hatch. A driver who waits grows
//! impatient; impatience shrinks the gaps they accept and the yielding they do, so queues
//! and deadlocks resolve themselves without special cases.

use omsi_sim::traffic::AiState;

/// [round 7 D1] Under this speed (m/s) a vehicle stands: one threshold for every rule that
/// asks whether a vehicle (a lead, a foe, a claimant, the player) stands (`Claim::standing`
/// adds its 1 m/s hysteresis for a claimant that stood).
pub const STANDING_SPEED: f32 = 0.3;

/// A fed-up driver (`Impatience::claims_after`) takes a way they would yield on when the foe
/// is still this many seconds away (s; the figure since b9568aa, not jittered per driver).
pub const FED_UP_AFTER: f32 = 2.5;

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

    /// [round 7 A4] A timetable bus: it serves stops (pulls in beside its people, keeps to
    /// its bay, is never given up). A random bus (`Bus`) is heavy traffic like a truck.
    pub fn serves_stops(self) -> bool {
        self == VehicleClass::ScheduledBus
    }

    /// A heavy vehicle (a bus, timetable or random, or a truck): its body and driver.
    pub fn heavy(self) -> bool {
        matches!(self, VehicleClass::Bus | VehicleClass::ScheduledBus | VehicleClass::Truck)
    }
}

/// A vehicle heavier than this (t, its `[mass]`) takes bends no harder than
/// `HEAVY_LAT_ACCEL`, whatever its class (`DriverParams::by_mass`): a van of `[ai_veh_type]`
/// 0 over 6 t bends like a lorry (the app's rule, by mass, until round 7).
pub const HEAVY_MASS: f32 = 6.0;
/// The bend acceleration of buses, lorries and heavy vans (m/s²; the app's value until
/// round 7).
pub const HEAVY_LAT_ACCEL: f32 = 1.6;

/// Everything about how a driver drives, in one place. Units: metres, seconds, m/s².
/// (Read from recordings: a field a recording lacks - one removed or added on another branch
/// - reads as unknown (NaN) and is then the preset's of the vehicle's class,
/// `DriverParams::fill_missing`; `record::VERSION`'s compatibility rule.)
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
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
    /// Gap accepted in the lane changed into (s, to the car behind there).
    pub lc_gap: f32,
    /// Lane-change motivation weights (LMRS / LC2013): the route needing it, a faster
    /// lane, keeping right. (LC2013's cooperative weight has no rule here yet.)
    pub lc_strategic: f32,
    pub lc_speed_gain: f32,
    pub lc_keep_right: f32,
    /// How fast impatience grows while waiting (per second) and where it caps.
    pub impatience_rate: f32,
    pub impatience_max: f32,
}

impl Default for DriverParams {
    /// Every field unknown (NaN): what a field a recording lacks is read as, until
    /// `fill_missing` gives it the vehicle's class's preset.
    fn default() -> DriverParams {
        let n = f32::NAN;
        DriverParams { tau: n, min_gap: n, accel: n, comfort_brake: n, speed_factor: n, lat_accel: n, reaction: n, lc_gap: n, lc_strategic: n, lc_speed_gain: n, lc_keep_right: n, impatience_rate: n, impatience_max: n }
    }
}

impl DriverParams {
    /// [fix pass 2, 5] The fields a recording lacked (NaN, `Default`) take the preset of
    /// `class` (a bus's `comfort_brake` is a bus's, not a car's).
    pub fn fill_missing(&mut self, class: VehicleClass) {
        let p = DriverParams::preset(class);
        macro_rules! fill {
            ($($f:ident),*) => { $( if self.$f.is_nan() { self.$f = p.$f; } )* };
        }
        fill!(tau, min_gap, accel, comfort_brake, speed_factor, lat_accel, reaction, lc_gap, lc_strategic, lc_speed_gain, lc_keep_right, impatience_rate, impatience_max);
    }

    /// The preset of a class, before jitter.
    pub fn preset(class: VehicleClass) -> DriverParams {
        let car = DriverParams {
            tau: 1.3,
            min_gap: 2.0,
            accel: 1.8,
            comfort_brake: 2.0,
            speed_factor: 1.0,
            // (the app overrode it with 2.4-3.0 by seed until round 7; at 2.7 the junction
            // test j6 rolls into the merge stretch: the crate's value stays the one value)
            lat_accel: 2.2,
            reaction: 0.8,
            lc_gap: 1.5,
            lc_strategic: 1.0,
            lc_speed_gain: 0.6,
            lc_keep_right: 0.45,
            impatience_rate: 1.0 / 30.0,
            impatience_max: 1.0,
        };
        // (buses and lorries take bends more gently than cars: the app's 1.6 for every
        // heavy vehicle, until round 7)
        let bus = DriverParams { tau: 1.6, min_gap: 2.5, accel: 1.0, comfort_brake: 1.5, lat_accel: HEAVY_LAT_ACCEL, lc_gap: 2.0, lc_speed_gain: 0.2, ..car };
        match class {
            VehicleClass::Car => car,
            VehicleClass::Taxi => DriverParams { tau: 1.0, accel: 2.2, speed_factor: 1.08, impatience_rate: 1.0 / 20.0, ..car },
            VehicleClass::Bus => bus,
            // a timetable bus brakes for its stops the way the town's drivers brake for a
            // light: with 1.5 m/s² of "comfortable" braking the planner braked at half that
            // and crept up to every stop for a hundred metres (the app's override until
            // round 7, now the one value the IDM and the plan both read)
            VehicleClass::ScheduledBus => DriverParams { comfort_brake: 2.1, ..bus },
            VehicleClass::Truck => DriverParams { tau: 1.8, min_gap: 3.0, accel: 0.8, comfort_brake: 1.5, speed_factor: 0.95, lat_accel: HEAVY_LAT_ACCEL, lc_gap: 2.5, lc_speed_gain: 0.2, ..car },
            VehicleClass::Rail => DriverParams { tau: 3.0, min_gap: 10.0, accel: 0.8, comfort_brake: 0.8, lat_accel: 1.0, impatience_rate: 0.0, ..car },
            // (an aircraft follows its flight path at its speed: bends never slow it, the
            // app's 50 m/s² until round 7; it speeds up and slows down at 0.5 m/s²)
            VehicleClass::Air => DriverParams { tau: 5.0, min_gap: 50.0, accel: 0.5, comfort_brake: 0.5, lat_accel: 50.0, impatience_rate: 0.0, ..car },
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
            lc_gap: j(p.lc_gap, -0.3),
            // (jittered a little only: the desire has to clear the planner's threshold
            // for every driver, or a slow driver sits behind a crawler for ever)
            lc_speed_gain: j(p.lc_speed_gain, 0.2),
            impatience_rate: j(p.impatience_rate, 0.5),
            ..p
        }
    }

    /// The parameters of a vehicle of `mass_t` tonnes: one over `HEAVY_MASS` takes bends no
    /// harder than `HEAVY_LAT_ACCEL`.
    pub fn by_mass(self, mass_t: f32) -> DriverParams {
        if mass_t > HEAVY_MASS {
            DriverParams { lat_accel: self.lat_accel.min(HEAVY_LAT_ACCEL), ..self }
        } else {
            self
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
    /// farther than this many seconds away (`FED_UP_AFTER`, the same for every driver).
    pub fn claims_after(&self) -> Option<f32> {
        if self.value >= 0.999 {
            Some(FED_UP_AFTER)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod mass_tests {
    use super::{DriverParams, HEAVY_LAT_ACCEL};
    use crate::VehicleClass;

    /// A car-class vehicle over 6 t bends like a lorry; a light one keeps its own.
    #[test]
    fn a5_a_heavy_van_bends_like_a_lorry() {
        let car = DriverParams::seeded(VehicleClass::Car, 7);
        assert_eq!(car.by_mass(7.5).lat_accel, car.lat_accel.min(HEAVY_LAT_ACCEL));
        assert_eq!(car.by_mass(1.3).lat_accel, car.lat_accel);
        assert!(car.by_mass(7.5).lat_accel <= HEAVY_LAT_ACCEL);
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
