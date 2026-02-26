// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

use agent_bus_proto_rust::agent_bus::*;
use agentbus_api::AgentBus;

pub async fn propose_string_intention<T: AgentBus>(
    impl_instance: &T,
    agent_bus_id: String,
    content: String,
) -> i64 {
    let payload = Payload {
        payload: Some(payload::Payload::Intention(Intention {
            intention: Some(intention::Intention::StringIntention(content)),
        })),
    };
    let request = ProposeRequest {
        agent_bus_id,
        payload: Some(payload),
        ..Default::default()
    };
    impl_instance
        .propose(request)
        .await
        .expect("Propose should succeed")
        .log_position
}

pub async fn propose_decider_policy<T: AgentBus>(
    impl_instance: &T,
    agent_bus_id: String,
    decider_policy: i32,
) -> i64 {
    let payload = Payload {
        payload: Some(payload::Payload::DeciderPolicy(decider_policy)),
    };
    let request = ProposeRequest {
        agent_bus_id,
        payload: Some(payload),
        ..Default::default()
    };
    impl_instance
        .propose(request)
        .await
        .expect("Propose should succeed")
        .log_position
}

pub async fn propose_vote<T: AgentBus>(
    impl_instance: &T,
    agent_bus_id: String,
    intention_id: i64,
    vote: bool,
) -> i64 {
    let payload = Payload {
        payload: Some(payload::Payload::Vote(Vote {
            intention_id,
            abstract_vote: Some(VoteType {
                vote_type: Some(vote_type::VoteType::BooleanVote(vote)),
            }),
            ..Default::default()
        })),
    };
    let request = ProposeRequest {
        agent_bus_id,
        payload: Some(payload),
        ..Default::default()
    };
    impl_instance
        .propose(request)
        .await
        .expect("Vote should succeed")
        .log_position
}

pub async fn poll<T: AgentBus>(
    impl_instance: &T,
    agent_bus_id: String,
    start_log_position: i64,
    max_entries: i16,
) -> PollResponse {
    let poll_request = PollRequest {
        agent_bus_id,
        start_log_position,
        max_entries: max_entries as i32,
        filter: None, // No filter means return all entries
        ..Default::default()
    };
    impl_instance
        .poll(poll_request)
        .await
        .expect("Poll should succeed")
}

pub async fn poll_selective<T: AgentBus>(
    impl_instance: &T,
    agent_bus_id: String,
    start_log_position: i64,
    max_entries: i16,
    payload_types: Vec<i32>,
) -> PollResponse {
    let poll_request = PollRequest {
        agent_bus_id,
        start_log_position,
        max_entries: max_entries as i32,
        filter: Some(PayloadTypeFilter { payload_types }),
        ..Default::default()
    };
    impl_instance
        .poll(poll_request)
        .await
        .expect("Poll should succeed")
}

/// Maps a Payload to its corresponding SelectivePollType with exhaustive matching.
/// This ensures compile-time checking - if a new Payload variant is added, this match will fail to compile.
pub fn payload_to_selective_poll_type(payload: &Payload) -> i32 {
    if let Some(ref p) = payload.payload {
        match p {
            payload::Payload::Intention(_) => SelectivePollType::Intention as i32,
            payload::Payload::Vote(_) => SelectivePollType::Vote as i32,
            payload::Payload::DeciderPolicy(_) => SelectivePollType::DeciderPolicy as i32,
            payload::Payload::Commit(_) => SelectivePollType::Commit as i32,
            payload::Payload::Abort(_) => SelectivePollType::Abort as i32,
            payload::Payload::VoterPolicy(_) => SelectivePollType::VoterPolicy as i32,
            payload::Payload::Control(_) => SelectivePollType::Control as i32,
            payload::Payload::InferenceInput(_) => SelectivePollType::InferenceInput as i32,
            payload::Payload::InferenceOutput(_) => SelectivePollType::InferenceOutput as i32,
            payload::Payload::ActionOutput(_) => SelectivePollType::ActionOutput as i32,
            payload::Payload::AgentInput(_) => SelectivePollType::AgentInput as i32,
            payload::Payload::AgentOutput(_) => SelectivePollType::AgentOutput as i32,
        }
    } else {
        panic!("Cannot map missing or unknown payload to SelectivePollType")
    }
}

/// Creates a payload from a variant name string (non-exhaustive match).
pub fn variant_name_to_payload(variant_name: &str) -> Payload {
    let payload_inner = match variant_name {
        "intention" => payload::Payload::Intention(Intention {
            intention: Some(intention::Intention::StringIntention("dummy".to_string())),
        }),
        "vote" => payload::Payload::Vote(Vote::default()),
        "deciderPolicy" => payload::Payload::DeciderPolicy(DeciderPolicy::OffByDefault as i32),
        "commit" => payload::Payload::Commit(Commit::default()),
        "abort" => payload::Payload::Abort(Abort::default()),
        "voterPolicy" => payload::Payload::VoterPolicy(VoterPolicy::default()),
        "control" => payload::Payload::Control(Control {
            control: Some(control::Control::AgentInput("dummy".to_string())),
        }),
        _ => unreachable!(
            "Unknown Payload variant '{}' - please add a mapping",
            variant_name
        ),
    };
    Payload {
        payload: Some(payload_inner),
    }
}

/// Creates a sample payload for each SelectivePollType (non-exhaustive match).
/// Returns None for unknown SelectivePollType variants.
pub fn selective_poll_type_to_payload(poll_type: i32) -> Option<Payload> {
    let poll_type_enum = SelectivePollType::try_from(poll_type).ok()?;
    let payload_inner = match poll_type_enum {
        SelectivePollType::Intention => payload::Payload::Intention(Intention {
            intention: Some(intention::Intention::StringIntention("test".to_string())),
        }),
        SelectivePollType::Vote => payload::Payload::Vote(Vote {
            intention_id: 0,
            abstract_vote: Some(VoteType {
                vote_type: Some(vote_type::VoteType::BooleanVote(true)),
            }),
            ..Default::default()
        }),
        SelectivePollType::DeciderPolicy => {
            payload::Payload::DeciderPolicy(DeciderPolicy::OnByDefault as i32)
        }
        SelectivePollType::Commit => payload::Payload::Commit(Commit {
            intention_id: 0,
            reason: "test".to_string(),
            ..Default::default()
        }),
        SelectivePollType::Abort => payload::Payload::Abort(Abort {
            intention_id: 0,
            reason: "test".to_string(),
            ..Default::default()
        }),
        SelectivePollType::VoterPolicy => payload::Payload::VoterPolicy(VoterPolicy {
            prompt_override: "test".to_string(),
            ..Default::default()
        }),
        SelectivePollType::Control => payload::Payload::Control(Control {
            control: Some(control::Control::AgentInput("test".to_string())),
        }),
        SelectivePollType::InferenceInput => payload::Payload::InferenceInput(InferenceInput {
            inference_input: Some(inference_input::InferenceInput::StringInferenceInput(
                "test".to_string(),
            )),
        }),
        SelectivePollType::InferenceOutput => payload::Payload::InferenceOutput(InferenceOutput {
            inference_output: Some(inference_output::InferenceOutput::StringInferenceOutput(
                "test".to_string(),
            )),
        }),
        SelectivePollType::ActionOutput => payload::Payload::ActionOutput(ActionOutput {
            intention_id: 0,
            ..Default::default()
        }),
        SelectivePollType::AgentInput => payload::Payload::AgentInput(AgentInput {
            agent_input: Some(agent_input::AgentInput::StringAgentInput(
                "test".to_string(),
            )),
        }),
        SelectivePollType::AgentOutput => payload::Payload::AgentOutput(AgentOutput {
            agent_output: Some(agent_output::AgentOutput::StringAgentOutput(
                "test".to_string(),
            )),
        }),
        SelectivePollType::Unspecified => {
            return None; // Unspecified doesn't have a meaningful payload
        }
    };
    Some(Payload {
        payload: Some(payload_inner),
    })
}
