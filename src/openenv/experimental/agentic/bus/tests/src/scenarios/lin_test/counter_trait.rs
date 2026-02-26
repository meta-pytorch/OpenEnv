// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Counter API trait for linearizability testing

use super::linearizability_tracker::ExecutedCommand;

/// Result of a counter operation, containing the intention ID and the new value.
#[derive(Clone, Debug)]
pub struct CommandResult {
    pub intention_id: i64,
    pub value: i64,
}

/// A simple counter API for testing linearizability.
/// Both operations are atomic and return the intention ID and new value.
pub trait Counter {
    /// Increment the counter by 1 and return the intention ID and new value.
    async fn increment(&self) -> Result<CommandResult, String>;

    /// Decrement the counter by 1 and return the intention ID and new value.
    async fn decrement(&self) -> Result<CommandResult, String>;

    /// Get the history of executed commands in linearization order.
    async fn get_command_history(&self) -> Vec<ExecutedCommand>;
}
