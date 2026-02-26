// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Simple in-memory implementation of the AgentBus.
//!
//! This is meant for testing and demo purposes.
//! Notably, it's a reference implementation for the AgentBus spec.
//! It uses a shared in-memory object to store backing state.
//! It is not meant for distributed / production use.

mod in_memory_agentbus;
mod in_memory_agentbus_state;

pub use in_memory_agentbus::InMemoryAgentBus;
pub use in_memory_agentbus_state::InMemoryAgentBusState;
