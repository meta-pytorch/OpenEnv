// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Import the generated proto types
pub use agent_bus_proto_rust::agent_bus::*;
use anyhow::Result;

/// Core trait that all AgentBus implementations must implement
/// This allows pluggable implementations with different state management approaches
pub trait AgentBus {
    /// Handle a propose request - add a command to the log
    fn propose(
        &self,
        request: ProposeRequest,
    ) -> impl std::future::Future<Output = Result<ProposeResponse>>;

    /// Handle a poll request - retrieve commands from the log
    fn poll(&self, request: PollRequest)
    -> impl std::future::Future<Output = Result<PollResponse>>;
}
