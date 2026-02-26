// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! SimulatorPyExt - Python wrapper for the Rust deterministic simulator.

use std::rc::Rc;
use std::time::Duration;
use std::time::Instant;

use agentbus_api::environment::Clock;
use pyo3::prelude::*;

use crate::simulator::Simulator;

/// Python wrapper for the Rust Simulator.
///
/// This is a thin bridge that allows Python's RustBackedEventLoop to schedule
/// callbacks on the Rust simulator. The callbacks are stored in Python (by handle_id),
/// and Rust calls `_fire_handle(handle_id)` when the scheduled time arrives.
#[pyclass(unsendable)]
pub struct SimulatorPyExt {
    simulator: Rc<Simulator>,
    start_instant: Instant,
}

impl SimulatorPyExt {
    /// Get the underlying simulator (for use from Rust code).
    pub fn simulator(&self) -> &Rc<Simulator> {
        &self.simulator
    }

    /// Create from an existing simulator (for Rust-side construction).
    pub fn from_simulator(simulator: Rc<Simulator>) -> Self {
        let start_instant = simulator.clock.current_time();
        Self {
            simulator,
            start_instant,
        }
    }
}

#[pymethods]
impl SimulatorPyExt {
    /// Create a new simulator with the given seed.
    #[new]
    fn new(seed: u64) -> Self {
        let simulator = Rc::new(Simulator::new(seed));
        let start_instant = simulator.clock.current_time();
        Self {
            simulator,
            start_instant,
        }
    }

    /// Schedule a callback to fire after `delay_ns` nanoseconds.
    ///
    /// When the delay expires, Rust will call `event_loop._fire_handle(handle_id)`.
    /// The actual callback is stored in Python's event loop by handle_id.
    fn schedule_callback(
        &self,
        py: Python<'_>,
        delay_ns: u64,
        handle_id: u64,
        event_loop: Py<PyAny>,
    ) -> PyResult<()> {
        let duration = Duration::from_nanos(delay_ns);
        let event_loop_clone = event_loop.clone_ref(py);

        self.simulator.executor.spawn_after(
            async move {
                Python::attach(|py| {
                    if let Err(e) = event_loop_clone.call_method1(py, "_fire_handle", (handle_id,))
                    {
                        eprintln!("Error firing handle {}: {:?}", handle_id, e);
                    }
                });
            },
            duration,
            &self.simulator.clock,
        );

        Ok(())
    }

    /// Get the current virtual time in nanoseconds since simulator start.
    fn current_time_ns(&self) -> u64 {
        let current = self.simulator.clock.current_time();
        current.duration_since(self.start_instant).as_nanos() as u64
    }

    /// Run the simulator
    fn run(&self) -> PyResult<()> {
        self.simulator.run();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;

    #[test]
    fn test_simulator_creation() {
        let seed: u64 = rand::thread_rng().r#gen();
        println!("Test seed: {}", seed);
        let sim = SimulatorPyExt::new(seed);
        assert_eq!(sim.current_time_ns(), 0);
    }
}
