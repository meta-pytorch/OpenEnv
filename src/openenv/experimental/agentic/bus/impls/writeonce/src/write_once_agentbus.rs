// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! WriteOnceAgentBus: An AgentBus backed by a write-once address space.
//!
//! The WriteOnceSpace handles routing via space_id internally. Multiple WriteOnceAgentBus
//! instances can share the same WriteOnceSpace, enabling testing of contention and retry-on-conflict.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use agent_bus_proto_rust::agent_bus::*;
use agentbus_api::WriteOnceError;
use agentbus_api::WriteOnceSpace;
use agentbus_api::environment::Clock;
use agentbus_api::environment::Environment;
use agentbus_api::payload_matches_filter;
use agentbus_api::validate_bus_id;
use anyhow::Result;
use bytes::Bytes;
use prost::Message as ProstMessage;
use tracing::debug;
use tracing::error;

const MAX_POLL_ENTRIES: usize = 64;

/// Shared state across all clones of a WriteOnceAgentBus
struct WriteOnceAgentBusState<W: WriteOnceSpace, E: Environment> {
    space: W,
    environment: Rc<E>,
    next_positions: HashMap<String, u64>,
}

impl<W: WriteOnceSpace, E: Environment> WriteOnceAgentBusState<W, E> {
    fn get_next_position(&mut self, bus_id: &str) -> u64 {
        *self.next_positions.entry(bus_id.to_string()).or_insert(0)
    }

    fn set_next_position(&mut self, bus_id: &str, position: u64) {
        self.next_positions.insert(bus_id.to_string(), position);
    }
}

/// WriteOnceAgentBus - AgentBus backed by a write-once address space.
///
/// This is a lightweight handle that can be cloned. All state is shared
/// via Rc<RefCell<...>>.
///
/// Generic over:
/// - `W: WriteOnceSpace + Clone` - the write-once storage backend (cloneable for async ops)
/// - `E: Environment` - the environment for clock access
pub struct WriteOnceAgentBus<W: WriteOnceSpace + Clone, E: Environment> {
    state: Rc<RefCell<WriteOnceAgentBusState<W, E>>>,
}

impl<W: WriteOnceSpace + Clone, E: Environment> Clone for WriteOnceAgentBus<W, E> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}

impl<W: WriteOnceSpace + Clone + 'static, E: Environment> WriteOnceAgentBus<W, E> {
    /// Create a new WriteOnceAgentBus with the given WriteOnceSpace and Environment.
    ///
    /// Multiple WriteOnceAgentBus clones share the same underlying space.
    pub fn new(space: W, environment: Rc<E>) -> Self {
        Self {
            state: Rc::new(RefCell::new(WriteOnceAgentBusState {
                space,
                environment,
                next_positions: HashMap::new(),
            })),
        }
    }
}

impl<W: WriteOnceSpace + Clone, E: Environment + 'static> agentbus_api::AgentBus
    for WriteOnceAgentBus<W, E>
{
    async fn propose(&self, request: ProposeRequest) -> Result<ProposeResponse> {
        validate_bus_id(&request.agent_bus_id)?;

        let payload = request
            .payload
            .ok_or_else(|| anyhow::anyhow!("Missing or unknown payload in request"))?;

        let serialized = serialize_payload(&payload);

        // Get the space, environment, and current position, releasing the borrow before await
        let (mut space, environment, mut position) = {
            let mut state = self.state.borrow_mut();
            let position = state.get_next_position(&request.agent_bus_id);
            (state.space.clone(), state.environment.clone(), position)
        };

        // Try to write at position; if slot is taken, advance and retry.
        // This handles:
        // 1. Crash recovery: on restart, we discover the actual tail
        // 2. Contention: when multiple instances share the same WriteOnceSpace,
        //    they race to claim slots and retry on conflict
        loop {
            match space
                .write(&request.agent_bus_id, position, serialized.clone())
                .await
            {
                Ok(()) => {
                    // Update next_position in state
                    self.state
                        .borrow_mut()
                        .set_next_position(&request.agent_bus_id, position + 1);

                    let current_time = environment.with_clock(|clock| clock.current_time());
                    debug!(
                        agent_bus_id = request.agent_bus_id,
                        position = position,
                        current_time = ?current_time,
                        "Intention added to AgentBus"
                    );

                    return Ok(ProposeResponse {
                        log_position: position as i64,
                    });
                }
                Err(WriteOnceError::AddressAlreadyExists(_)) => {
                    // Slot already taken, advance and retry
                    debug!(
                        "WriteOnceAgentBus: position {} taken, retrying at {}",
                        position,
                        position + 1
                    );
                    position += 1;
                }
                Err(WriteOnceError::BackendUnavailable(msg)) => {
                    error!(
                        agent_bus_id = request.agent_bus_id,
                        error = %msg,
                        "WriteOnceSpace backend unavailable"
                    );
                    return Err(anyhow::anyhow!(
                        "WriteOnceSpace backend unavailable: {}",
                        msg
                    ));
                }
                Err(e @ WriteOnceError::NotImplemented) => {
                    return Err(anyhow::anyhow!("{}", e));
                }
            }
        }
    }

    async fn poll(&self, request: PollRequest) -> Result<PollResponse> {
        let max_entries = (request.max_entries as usize).min(MAX_POLL_ENTRIES);
        let start_position = request.start_log_position.max(0) as u64;

        let payload_types = request.filter.as_ref().map(|f| f.payload_types.clone());

        // If filter is set to Some(vec![]), return no entries
        if let Some(ref filter) = payload_types {
            if filter.is_empty() {
                return Ok(PollResponse {
                    entries: vec![],
                    complete: true,
                });
            }
        }

        // Get the space and scan for end position
        let (space, mut end_position) = {
            let mut state = self.state.borrow_mut();
            let position = state.get_next_position(&request.agent_bus_id);
            (state.space.clone(), position)
        };

        // Scan to find the actual end position (in case other instances wrote)
        while space
            .read(&request.agent_bus_id, end_position)
            .await
            .is_some()
        {
            end_position += 1;
        }

        // Update next_position in state
        self.state
            .borrow_mut()
            .set_next_position(&request.agent_bus_id, end_position);

        if end_position == 0 {
            return Ok(PollResponse {
                entries: vec![],
                complete: true,
            });
        }

        let mut entries = Vec::new();
        let mut complete = true;
        let mut pos = start_position;

        while pos < end_position {
            let serialized = space.read(&request.agent_bus_id, pos).await;

            if let Some(serialized) = serialized {
                if let Some(payload) = deserialize_payload(&serialized) {
                    // Apply filter if present
                    if !payload_matches_filter(&payload, &payload_types) {
                        pos += 1;
                        continue;
                    }

                    if entries.len() >= max_entries {
                        complete = false;
                        break;
                    }

                    entries.push(BusEntry {
                        header: Some(Header {
                            log_position: pos as i64,
                        }),
                        payload: Some(payload),
                    });
                }
            }
            pos += 1;
        }

        Ok(PollResponse { entries, complete })
    }
}

fn serialize_payload(payload: &Payload) -> Bytes {
    Bytes::from(ProstMessage::encode_to_vec(payload))
}

fn deserialize_payload(bytes: &Bytes) -> Option<Payload> {
    <Payload as ProstMessage>::decode(&bytes[..]).ok()
}
