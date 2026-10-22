//! AI road traffic, second implementation.
//!
//! The map's `[path]` lanes, their proximity links, the meeting places of a crossing
//! object's lanes and the light programs come from `omsi_sim::traffic` unchanged. On top
//! of them this crate compiles each crossing object into a [`junction::Junction`] with a
//! foe matrix once, gives every vehicle a named [`driver::DriverParams`] set, and runs the
//! frame as three pure stages per vehicle: [`perception`] (what the driver sees, plain
//! values), [`plan`] (one decision) and the operational step (`omsi_sim::traffic::AiState`,
//! the car-following model and the lane geometry). Cross-vehicle state is written in one
//! place only, the junction [`reservation`]s. Nothing here renders, loads a vehicle or
//! touches the GPU, so the whole thing runs headless in the scenario tests.
//!
//! [`record`] writes what a frame reads from outside, frame by frame, and [`replay`] runs
//! such a recording again headless: a jam seen in the game becomes a test. [`export`]
//! writes the network and its junctions out for other tools.

pub mod access;
pub mod driver;
pub mod export;
pub mod finite;
pub mod junction;
pub mod perception;
pub mod plan;
pub mod record;
pub mod replay;
pub mod removal;
pub mod reservation;
pub mod sim;

pub use access::{is_open, open_to, Closed};
pub use driver::{DriverParams, Impatience, VehicleClass};
pub use junction::{Cut, Foe, FoeKind, Junction, Junctions, Movement, PedCrossing};
pub use perception::{Headway, JunctionView, NeighbourView, Perception, Seen, SignalView};
pub use plan::{Plan, Start, Why};
pub use reservation::{Claim, Grant, Hold};
pub use sim::{Decision, External, Parked, Passing, Person, PlayerView, Simulation, Vehicle};
