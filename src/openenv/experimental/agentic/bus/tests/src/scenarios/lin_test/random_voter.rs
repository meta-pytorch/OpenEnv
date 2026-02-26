// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Random boolean voter for testing

use std::cell::RefCell;
use std::rc::Rc;

use agent_bus_proto_rust::agent_bus::Payload;
use agent_bus_proto_rust::agent_bus::PollRequest;
use agent_bus_proto_rust::agent_bus::ProposeRequest;
use agent_bus_proto_rust::agent_bus::Vote;
use agentbus_api::AgentBus;
use agentbus_api::environment::Environment;
use rand::Rng;

pub struct RandomVoter<T: AgentBus, E: Environment> {
    agent_bus_impl: T,
    agent_bus_id: String,
    pub env: Rc<E>,
    next_log_position: RefCell<i64>,
}

impl<T: AgentBus, E: Environment> RandomVoter<T, E> {
    pub fn new(agent_bus_impl: T, agent_bus_id: String, env: Rc<E>) -> Self {
        Self {
            agent_bus_impl,
            agent_bus_id,
            env,
            next_log_position: RefCell::new(0),
        }
    }

    pub async fn poll_and_vote_once(&self) {
        let start_log_position = *self.next_log_position.borrow();
        let payload_types =
            vec![agent_bus_proto_rust::agent_bus::SelectivePollType::Intention as i32];
        let poll_result = self
            .agent_bus_impl
            .poll(PollRequest {
                agent_bus_id: self.agent_bus_id.clone(),
                start_log_position,
                max_entries: 1000,
                filter: Some(agent_bus_proto_rust::agent_bus::PayloadTypeFilter { payload_types }),
                ..Default::default()
            })
            .await
            .expect("Poll should succeed");

        for entry in poll_result.entries {
            *self.next_log_position.borrow_mut() = entry.header.as_ref().unwrap().log_position + 1;

            if let Some(ref payload) = entry.payload {
                if let Some(agent_bus_proto_rust::agent_bus::payload::Payload::Intention(
                    ref intention,
                )) = payload.payload
                {
                    if let Some(
                        agent_bus_proto_rust::agent_bus::intention::Intention::StringIntention(_),
                    ) = intention.intention
                    {
                        let intention_id = entry.header.as_ref().unwrap().log_position;
                        let random_vote = self.env.with_rng(|rng| rng.gen_bool(0.5));

                        let vote_payload = Payload {
                            payload: Some(agent_bus_proto_rust::agent_bus::payload::Payload::Vote(Vote {
                                abstract_vote: Some(agent_bus_proto_rust::agent_bus::VoteType {
                                    vote_type: Some(agent_bus_proto_rust::agent_bus::vote_type::VoteType::BooleanVote(random_vote)),
                                }),
                                intention_id,
                                ..Default::default()
                            })),
                        };

                        self.agent_bus_impl
                            .propose(ProposeRequest {
                                agent_bus_id: self.agent_bus_id.clone(),
                                payload: Some(vote_payload),
                                ..Default::default()
                            })
                            .await
                            .expect("Vote proposal should succeed");
                    }
                }
            }
        }
    }
}
