// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Common fault injection configuration shared across test implementations

use rand::rngs::StdRng;

/// Possible fates for a request
#[derive(Debug)]
pub enum Fate {
    /// Request succeeds: committed to bus and client receives success
    Success,
    /// Commit happens, then client receives error (e.g. because response is delayed)
    CommitThenError,
    /// Lost request: will never be committed and client receives error
    Lost,
    /// Client receives error, then commit happens in background (e.g. because request is delayed)
    ErrorThenCommit,
}

/// Configuration for fault injection behavior
#[derive(Debug, Clone)]
pub struct FaultConfig {
    /// Probability of Lost request
    pub prob_lost: f64,
    /// Probability of CommitThenError
    pub prob_commit_then_error: f64,
    /// Probability of ErrorThenCommit
    pub prob_error_then_commit: f64,
}

impl FaultConfig {
    /// Validate that probabilities are valid
    pub fn validate(&self) -> Result<(), String> {
        if self.prob_lost < 0.0
            || self.prob_commit_then_error < 0.0
            || self.prob_error_then_commit < 0.0
        {
            return Err("Probabilities must be >= 0.0".to_string());
        }
        if self.prob_lost + self.prob_commit_then_error + self.prob_error_then_commit > 1.0 {
            return Err("Sum of probabilities must be <= 1.0".to_string());
        }
        Ok(())
    }

    /// Determine the fate of a request using the configured probabilities
    pub fn determine_fate(&self, rng: &mut StdRng) -> Fate {
        use rand::Rng as _;
        let rand_val = rng.gen_range(0.0..1.0);
        if rand_val < self.prob_lost {
            Fate::Lost
        } else if rand_val < self.prob_lost + self.prob_commit_then_error {
            Fate::CommitThenError
        } else if rand_val
            < self.prob_lost + self.prob_commit_then_error + self.prob_error_then_commit
        {
            Fate::ErrorThenCommit
        } else {
            Fate::Success
        }
    }
}

impl Default for FaultConfig {
    fn default() -> Self {
        Self {
            prob_lost: 0.1,
            prob_commit_then_error: 0.1,
            prob_error_then_commit: 0.1,
        }
    }
}
