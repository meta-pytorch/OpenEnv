// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

use std::collections::HashMap;

use agent_bus_proto_rust::agent_bus::*;
use agentbus_api::payload_matches_filter;
use agentbus_api::validate_bus_id;
use anyhow::Result;

/// Maximum number of entries that can be returned in a single poll request
const MAX_POLL_ENTRIES: usize = 64;

/// In-memory state implementation for AgentBus service
#[derive(Debug, Default)]
pub struct InMemoryAgentBusState {
    // Server side state
    // TODO: Consider Vec<Option<Payload>> if we need to model
    // holes in the logs.
    buses: HashMap<String, Vec<BusEntry>>,
}

impl InMemoryAgentBusState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn propose(&mut self, request: ProposeRequest) -> Result<ProposeResponse> {
        // Validate bus ID
        validate_bus_id(&request.agent_bus_id)?;

        let payload = request
            .payload
            .ok_or_else(|| anyhow::anyhow!("Missing or unknown payload in request"))?;

        let bus = self
            .buses
            .entry(request.agent_bus_id)
            .or_insert_with(Vec::new);
        let position = bus.len() as i64;

        let entry = BusEntry {
            header: Some(Header {
                log_position: position,
            }),
            payload: Some(payload),
        };

        bus.push(entry);

        Ok(ProposeResponse {
            log_position: position,
        })
    }

    pub fn poll(&self, request: PollRequest) -> Result<PollResponse> {
        let max_entries = (request.max_entries as usize).min(MAX_POLL_ENTRIES);

        let payload_types = request.filter.as_ref().map(|f| f.payload_types.clone());

        let (entries, complete) = self.fetch_entries(
            request.agent_bus_id,
            request.start_log_position,
            max_entries,
            &payload_types,
        )?;

        Ok(PollResponse { entries, complete })
    }

    // TODO: this is terribly inefficient right now; implement indexes for the entry filters
    // Returns:
    //   Vec<BusEntry>: the entries matching the request
    //   bool `complete`: whether all matching entries were returned, false if more available
    fn fetch_entries(
        &self,
        agent_bus_id: String,
        start_position: i64,
        max_entries: usize,
        payload_types: &Option<Vec<i32>>,
    ) -> Result<(Vec<BusEntry>, bool)> {
        let entries = self
            .buses
            .get(&agent_bus_id)
            .map(|bus| {
                let start_idx = start_position.max(0) as usize;

                let slice = if start_idx < bus.len() {
                    &bus[start_idx..]
                } else {
                    &[]
                };

                // If filter is set to Some(vec![]), return no entries
                if let Some(filter) = payload_types {
                    if filter.is_empty() {
                        return (vec![], true);
                    }
                }

                let mut result = Vec::new();
                let mut complete = true;

                for (i, entry) in slice.iter().enumerate() {
                    // Apply filter if present
                    if let Some(ref payload) = entry.payload {
                        if !payload_matches_filter(payload, payload_types) {
                            continue;
                        }
                    } else {
                        continue; // Skip entries with no payload
                    }

                    if result.len() >= max_entries {
                        // complete=false because there are more entries matching the request to add
                        complete = false;
                        break;
                    }

                    // Need to reconstruct entry with correct log position
                    let log_entry = BusEntry {
                        header: Some(Header {
                            log_position: (start_idx + i) as i64,
                        }),
                        payload: entry.payload.clone(),
                    };
                    result.push(log_entry);
                }

                (result, complete)
            })
            .unwrap_or((Vec::new(), true));

        Ok(entries)
    }

    pub fn get_current_position(&self, agent_bus_id: &str) -> i64 {
        self.buses
            .get(agent_bus_id)
            .map_or(0, |bus| bus.len() as i64)
    }
}
