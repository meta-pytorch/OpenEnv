// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Common test fixtures for WriteOnceSpace implementations

use std::rc::Rc;

use agentbus_api::WriteOnceSpace;
use agentbus_api::environment::Environment;

/// Trait that defines how to create test fixtures for a WriteOnceSpace implementation.
/// Implement this trait for each backend you want to test.
///
/// The fixture instance owns the shared state/infrastructure (e.g., shared memory, DB connection)
/// and can create multiple implementation instances that share that state.
pub trait WriteOnceSpaceTestFixture {
    /// The environment type used for this fixture
    type Env: Environment + 'static;

    /// The WriteOnceSpace implementation type
    type WriteOnceSpaceImpl: WriteOnceSpace + 'static;

    /// Get the environment for this fixture
    fn get_env(&self) -> Rc<Self::Env>;

    /// Create an implementation instance that shares this fixture's state.
    fn create_impl(&self) -> Self::WriteOnceSpaceImpl;
}

pub mod simtest;
