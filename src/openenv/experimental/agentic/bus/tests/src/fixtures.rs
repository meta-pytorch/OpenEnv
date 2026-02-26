// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Common test fixtures for AgentBus implementations

use std::rc::Rc;

use anyhow::Result;
use fbinit::FacebookInit;

use crate::simulator::Simulator;

/// Trait that defines how to create test fixtures for an AgentBus implementation.
/// Implement this trait for each backend you want to test.
///
/// The fixture instance owns the shared state/infrastructure (e.g., shared memory, DB connection, etc.)
/// and can create multiple implementation instances that share that state.
pub trait AgentBusTestFixture: Sized {
    /// The environment type used for this fixture
    type Env: agentbus_api::environment::Environment + 'static;

    /// The AgentBus implementation type
    type AgentBusImpl: agentbus_api::AgentBus + 'static;

    /// Get the environment for this fixture
    fn get_env(&self) -> Rc<Self::Env>;

    /// Create an implementation instance that shares this fixture's state
    fn create_impl(&self) -> Self::AgentBusImpl;
}

/// Construction trait for fixtures that can be created with a Simulator
pub trait SimulatorFixture: Sized {
    fn new(simulator: Simulator) -> Self;
}

/// Construction trait for fixtures that require async initialization
pub trait IntegrationFixture: Sized {
    fn new_async(fb: FacebookInit) -> impl std::future::Future<Output = Result<Self>>;
}

pub mod integration;
pub mod simtest;
pub mod write_once_agentbus_fixture;

pub use write_once_agentbus_fixture::WriteOnceAgentBusGenericFixture;
