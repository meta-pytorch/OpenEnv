// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! WriteOnce-based AgentBus implementation.
//!
//! This crate provides an AgentBus implementation backed by write-once address spaces.
//! It includes:
//! - `WriteOnceAgentBus`: An AgentBus that uses WriteOnceSpace for storage
//! - `ChanneledWriteOnceSpace`: A channel-based wrapper for async WriteOnceSpace access
//! - `InMemoryWriteOnceSpace`: A simple in-memory WriteOnceSpace implementation

mod channeled_write_once_space;
mod in_memory_write_once_space;
mod in_memory_write_once_space_state;
mod write_once_agentbus;

pub use channeled_write_once_space::ChanneledWriteOnceSpace;
pub use channeled_write_once_space::ChanneledWriteOnceSpaceBackend;
pub use in_memory_write_once_space::InMemoryWriteOnceSpace;
pub use write_once_agentbus::WriteOnceAgentBus;
