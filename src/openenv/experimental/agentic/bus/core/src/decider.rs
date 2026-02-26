// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Decider component for agentbus
//!
//! The Decider polls an AgentBus for votes and policies, maintains internal
//! VoteRecord state, and proposes Commits or Aborts when decisions are made.

use std::collections::HashMap;
use std::rc::Rc;

use agent_bus_proto_rust::agent_bus::*;
use agentbus_api::AgentBus;
use agentbus_api::environment::Environment;
use thiserror::Error;

/// Error type for Decider operations
#[derive(Error, Debug)]
pub enum DeciderError {
    #[error("AgentBus call failed: {0}")]
    FailedAgentBusCall(#[from] anyhow::Error),

    #[error("Unknown payload type encountered at log position {0}")]
    UnknownPayloadType(i64),
}

/// VoteRecord trait for tracking votes and making decisions for a single intention
/// Each VoteRecord instance tracks exactly one intention's voting state
pub trait VoteRecord: std::fmt::Debug {
    /// Register this intention
    /// Returns Some((decision, reason)) for immediate decision, None to wait for votes
    /// decision: true for commit, false for abort
    /// reason: explanation for the decision
    fn register(&mut self) -> Option<(bool, String)>;

    /// Apply a vote to this intention
    /// Returns Some((decision, reason)) for commit/abort, None for undecided
    /// decision: true for commit, false for abort
    /// reason: explanation for the decision
    fn apply_vote(&mut self, vote: &Vote) -> Option<(bool, String)>;
}

/// Factory function type for creating VoteRecords based on decider policy
pub type VoteRecordFactory = fn(decider_policy: i32) -> Box<dyn VoteRecord>;

/// State tracked for each intention being voted on
struct IntentionState {
    vote_tracker: Box<dyn VoteRecord>,
    decision_made: bool,
}

/// Decider component that polls an AgentBus and proposes commits/aborts
pub struct Decider<T: AgentBus, E: Environment> {
    agent_bus: T,
    agent_bus_id: String,
    next_log_position: i64,
    current_decider_policy: i32,
    intention_states: HashMap<i64, IntentionState>,
    vote_tracker_factory: VoteRecordFactory,
    environment: Rc<E>,
}

impl<T: AgentBus, E: Environment> Decider<T, E> {
    /// Create a new Decider for the given AgentBus
    pub fn new(
        agent_bus: T,
        agent_bus_id: String,
        initial_decider_policy: i32,
        vote_tracker_factory: VoteRecordFactory,
        environment: Rc<E>,
    ) -> Self {
        Self {
            agent_bus,
            agent_bus_id,
            next_log_position: 0,
            current_decider_policy: initial_decider_policy,
            intention_states: HashMap::new(),
            vote_tracker_factory,
            environment,
        }
    }

    /// Poll the AgentBus and process new entries
    /// Returns the number of entries processed
    /// TODO: right now if this returns a DeciderError, it is not
    /// safe to retry on the same Decider object.
    /// We need to add tests and then provide finer-grained retry on
    /// specific error types.
    pub async fn poll_and_decide(&mut self) -> Result<usize, DeciderError> {
        let payload_types = vec![
            SelectivePollType::Intention as i32,
            SelectivePollType::Vote as i32,
            SelectivePollType::DeciderPolicy as i32,
            SelectivePollType::VoterPolicy as i32,
            SelectivePollType::Commit as i32,
            SelectivePollType::Abort as i32,
        ];

        let poll_request = PollRequest {
            agent_bus_id: self.agent_bus_id.clone(),
            start_log_position: self.next_log_position,
            max_entries: 64,
            filter: Some(PayloadTypeFilter { payload_types }),
        };

        let poll_response = self.agent_bus.poll(poll_request).await.map_err(|e| {
            DeciderError::FailedAgentBusCall(anyhow::anyhow!("Poll failed: {:?}", e))
        })?;

        let entries_count = poll_response.entries.len();

        for entry in poll_response.entries {
            self.process_entry(entry).await?;
        }

        Ok(entries_count)
    }

    /// Run the decider in a loop, continuously polling and deciding
    /// Sleeps for the specified duration when there are no new entries
    /// TODO: right now if this returns a DeciderError, it is not
    /// safe to retry on the same Decider object.
    /// We need to add tests and then provide finer-grained retry on
    /// specific error types.
    pub async fn run(&mut self, poll_interval: std::time::Duration) -> Result<(), DeciderError> {
        loop {
            let entries_processed = match self.poll_and_decide().await {
                Ok(count) => count,
                Err(DeciderError::FailedAgentBusCall(e)) => {
                    //TODO: for now we propagate the error up without retrying;
                    //later, we need to add tests and retry carefully at
                    //different layers to ensure idempotence
                    tracing::error!("AgentBus call failed: {}", e);
                    return Err(DeciderError::FailedAgentBusCall(e));
                }
                Err(DeciderError::UnknownPayloadType(position)) => {
                    //TODO: for now we propagate the error up without retrying;
                    //later, we need to add tests and decide whether to skip or
                    //get stuck retrying safely at a particular entry or panic
                    tracing::error!("Unknown payload type at log position {}", position);
                    return Err(DeciderError::UnknownPayloadType(position));
                }
            };

            if entries_processed == 0 {
                // No new entries, sleep before next poll
                let sleep_future = self.environment.sleep(poll_interval);
                sleep_future.await;
            }
            // If we processed entries, continue immediately to next poll
        }
    }

    /// Process a single BusEntry
    async fn process_entry(&mut self, entry: BusEntry) -> Result<(), DeciderError> {
        let log_position = entry.header.as_ref().map(|h| h.log_position).unwrap_or(0);

        if let Some(payload) = entry.payload {
            match payload.payload {
                Some(payload::Payload::Intention(_intention)) => {
                    self.process_intention(log_position).await?;
                }
                Some(payload::Payload::Vote(vote)) => {
                    self.process_vote(log_position, vote).await?;
                }
                Some(payload::Payload::DeciderPolicy(decider_policy)) => {
                    self.process_decider_policy(decider_policy).await?;
                }
                Some(payload::Payload::VoterPolicy(_)) => {
                    // Voter policy doesn't affect decider logic
                }
                Some(payload::Payload::Commit(_)) | Some(payload::Payload::Abort(_)) => {
                    // Commits and aborts are materialized decisions, no action needed
                }
                Some(payload::Payload::Control(_)) => {
                    // Control entries should have been filtered out by poll request
                    return Err(DeciderError::UnknownPayloadType(log_position));
                }
                Some(payload::Payload::InferenceInput(_))
                | Some(payload::Payload::InferenceOutput(_))
                | Some(payload::Payload::ActionOutput(_))
                | Some(payload::Payload::AgentInput(_))
                | Some(payload::Payload::AgentOutput(_)) => {
                    // Inference logging, action output, and agent I/O entries don't affect decider logic
                }
                None => {
                    // Missing or unknown payload
                    return Err(DeciderError::UnknownPayloadType(log_position));
                }
            }
        }

        self.next_log_position = log_position + 1;

        Ok(())
    }

    /// Process an Intention entry
    async fn process_intention(&mut self, log_position: i64) -> Result<(), DeciderError> {
        let mut vote_tracker = (self.vote_tracker_factory)(self.current_decider_policy);

        let decision = vote_tracker.register();

        let decision_made = decision.is_some();

        self.intention_states.insert(
            log_position,
            IntentionState {
                vote_tracker,
                decision_made,
            },
        );

        if let Some((commit, reason)) = decision {
            self.propose_decision(log_position, commit, reason).await?;
        }

        Ok(())
    }

    /// Process a Vote entry
    async fn process_vote(&mut self, _log_position: i64, vote: Vote) -> Result<(), DeciderError> {
        let intention_id = vote.intention_id;

        if let Some(state) = self.intention_states.get_mut(&intention_id) {
            if state.decision_made {
                return Ok(());
            }

            let decision = state.vote_tracker.apply_vote(&vote);

            if let Some((commit, reason)) = decision {
                state.decision_made = true;
                self.propose_decision(intention_id, commit, reason).await?;
            }
        }
        // TODO: intention not found: should we panic or log and/or throw?
        // For now, we log and continue, since decider may have started playing mid-way through the agentbus
        else {
            tracing::warn!("Vote for unknown intention: {:?}", vote);
        }
        Ok(())
    }

    /// Process a DeciderPolicy entry
    async fn process_decider_policy(&mut self, decider_policy: i32) -> Result<(), DeciderError> {
        // Update the current decider policy
        self.current_decider_policy = decider_policy;
        // Note: We don't update existing VoteRecords, only new intentions will use the new policy
        Ok(())
    }

    /// Propose a Commit or Abort decision to the AgentBus
    async fn propose_decision(
        &mut self,
        intention_id: i64,
        commit: bool,
        reason: String,
    ) -> Result<(), DeciderError> {
        let payload = if commit {
            Payload {
                payload: Some(payload::Payload::Commit(Commit {
                    intention_id,
                    reason,
                })),
            }
        } else {
            Payload {
                payload: Some(payload::Payload::Abort(Abort {
                    intention_id,
                    reason,
                })),
            }
        };

        let propose_request = ProposeRequest {
            agent_bus_id: self.agent_bus_id.clone(),
            payload: Some(payload),
        };

        self.agent_bus.propose(propose_request).await.map_err(|e| {
            DeciderError::FailedAgentBusCall(anyhow::anyhow!("Propose failed: {:?}", e))
        })?;

        tracing::debug!(
            intention_id = intention_id,
            decision = if commit { "COMMIT" } else { "ABORT" },
            "Successfully proposed decision"
        );

        Ok(())
    }
}
