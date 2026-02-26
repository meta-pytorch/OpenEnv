// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Python module entry point for simulator_py_ext.

use agentbus_tests::simulator_py_ext::SimulatorPyExt;
use pyo3::prelude::*;

/// Python module for the deterministic simulator.
#[pymodule]
fn simulator_py_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SimulatorPyExt>()?;
    Ok(())
}
