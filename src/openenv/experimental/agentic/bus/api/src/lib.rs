// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! AgentBus API - Core traits and interfaces
//!
//! This crate defines the fundamental traits and types for AgentBus:
//! - `AgentBus` trait: The main interface for AgentBus implementations
//! - `Environment` trait: Clock and randomness abstraction for testing

pub mod environment;
pub mod helpers;
pub mod traits;
pub mod validation;
pub mod write_once_space;

// Re-export commonly used items
// Re-export proto types for convenience
pub use agent_bus_proto_rust::agent_bus::*;
pub use environment::Clock;
pub use environment::Environment;
pub use environment::RealEnvironment;
pub use helpers::get_payload_type;
pub use helpers::payload_matches_filter;
pub use traits::AgentBus;
pub use validation::validate_bus_id;
pub use write_once_space::WriteOnceError;
pub use write_once_space::WriteOnceResult;
pub use write_once_space::WriteOnceSpace;
