// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Helper functions for AgentBus implementations

use agent_bus_proto_rust::agent_bus::Payload;
use agent_bus_proto_rust::agent_bus::SelectivePollType;
use agent_bus_proto_rust::agent_bus::payload;

/// Returns the SelectivePollType for a payload, or None if the payload is empty.
pub fn get_payload_type(payload: &Payload) -> Option<i32> {
    match &payload.payload {
        Some(payload::Payload::Intention(_)) => Some(SelectivePollType::Intention as i32),
        Some(payload::Payload::Vote(_)) => Some(SelectivePollType::Vote as i32),
        Some(payload::Payload::DeciderPolicy(_)) => Some(SelectivePollType::DeciderPolicy as i32),
        Some(payload::Payload::VoterPolicy(_)) => Some(SelectivePollType::VoterPolicy as i32),
        Some(payload::Payload::Commit(_)) => Some(SelectivePollType::Commit as i32),
        Some(payload::Payload::Abort(_)) => Some(SelectivePollType::Abort as i32),
        Some(payload::Payload::Control(_)) => Some(SelectivePollType::Control as i32),
        Some(payload::Payload::InferenceInput(_)) => Some(SelectivePollType::InferenceInput as i32),
        Some(payload::Payload::InferenceOutput(_)) => {
            Some(SelectivePollType::InferenceOutput as i32)
        }
        Some(payload::Payload::ActionOutput(_)) => Some(SelectivePollType::ActionOutput as i32),
        Some(payload::Payload::AgentInput(_)) => Some(SelectivePollType::AgentInput as i32),
        Some(payload::Payload::AgentOutput(_)) => Some(SelectivePollType::AgentOutput as i32),
        None => None,
    }
}

/// Checks if a payload matches the given filter.
/// Returns true if filter is None (no filtering) or if the payload type is in the filter.
/// Returns false if the payload has no type or doesn't match the filter.
pub fn payload_matches_filter(payload: &Payload, filter: &Option<Vec<i32>>) -> bool {
    match filter {
        None => true,
        Some(types) => match get_payload_type(payload) {
            Some(payload_type) => types.contains(&payload_type),
            None => false,
        },
    }
}
