// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

use std::cell::RefCell;
use std::rc::Rc;

use agent_bus_proto_rust::agent_bus::*;
use agentbus_api::environment::Clock;
use agentbus_api::environment::Environment;
use anyhow::Result;
use tracing::debug;
use tracing::error;

use crate::in_memory_agentbus_state::InMemoryAgentBusState;

pub struct InMemoryAgentBus<E: Environment> {
    state: Rc<RefCell<InMemoryAgentBusState>>,
    environment: Rc<E>,
}

impl<E: Environment> InMemoryAgentBus<E> {
    pub fn new(environment: Rc<E>) -> Self {
        Self {
            state: Rc::new(RefCell::new(InMemoryAgentBusState::new())),
            environment,
        }
    }

    pub fn environment(&self) -> Rc<E> {
        self.environment.clone()
    }
}

impl<E: Environment> Clone for InMemoryAgentBus<E> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            environment: self.environment.clone(),
        }
    }
}

impl<E: Environment> InMemoryAgentBus<E> {
    pub async fn propose(&self, request: ProposeRequest) -> Result<ProposeResponse> {
        let result = self.state.borrow_mut().propose(request.clone());

        match &result {
            Ok(response) => {
                let current_time = self.environment.with_clock(|clock| clock.current_time());
                debug!(
                    agent_bus_id = request.agent_bus_id,
                    position = response.log_position,
                    current_time = ?current_time,
                    "Intention added to AgentBus"
                );
            }
            Err(e) => {
                error!(
                    agent_bus_id = request.agent_bus_id,
                    error = %e,
                    "Error adding intention"
                );
            }
        }

        result
    }

    pub async fn poll(&self, request: PollRequest) -> Result<PollResponse> {
        let result = self.state.borrow().poll(request.clone());

        match &result {
            Ok(response) => {
                let current_position = self
                    .state
                    .borrow()
                    .get_current_position(&request.agent_bus_id);
                debug!(
                    agent_bus_id = request.agent_bus_id,
                    start_position = request.start_log_position,
                    max_entries = request.max_entries,
                    entries_found = response.entries.len(),
                    current_position = current_position,
                    complete = response.complete,
                    "Poll for AgentBus"
                );
            }
            Err(e) => {
                error!(
                    agent_bus_id = request.agent_bus_id,
                    start_position = request.start_log_position,
                    error = %e,
                    "Error polling entries"
                );
            }
        }

        result
    }
}

// Implement the AgentBus trait for InMemoryAgentBus
impl<E: Environment + 'static> agentbus_api::AgentBus for InMemoryAgentBus<E> {
    async fn propose(&self, request: ProposeRequest) -> Result<ProposeResponse> {
        self.propose(request).await
    }

    async fn poll(&self, request: PollRequest) -> Result<PollResponse> {
        self.poll(request).await
    }
}
