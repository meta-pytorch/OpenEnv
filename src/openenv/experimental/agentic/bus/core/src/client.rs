// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! AgentBus client wrapper that implements the AgentBus trait
//!
//! This wrapper allows Decider to work with both:
//! - Direct AgentBus implementations (for simtests)
//! - gRPC client connections (for production DeciderService)

use agent_bus_proto_rust::agent_bus::PollRequest;
use agent_bus_proto_rust::agent_bus::PollResponse;
use agent_bus_proto_rust::agent_bus::ProposeRequest;
use agent_bus_proto_rust::agent_bus::ProposeResponse;
use agent_bus_proto_rust::agent_bus::agent_bus_service_client::AgentBusServiceClient;
use agentbus_api::AgentBus;
use anyhow::Result;
use tonic::transport::Channel;
use tracing::warn;

/// Wrapper around a gRPC AgentBusService client that implements the AgentBus trait
///
/// This allows Decider to work with remote AgentBus instances via gRPC while
/// maintaining the same interface as direct AgentBus implementations.
///
/// # Example
/// ```ignore
/// use agent_bus_client::AgentBusClient;
/// use agent_bus_decider::Decider;
/// use std::sync::Arc;
///
/// // Create gRPC client
/// let channel = tonic::transport::Channel::from_static("http://[::1]:9999")
///     .connect()
///     .await?;
/// let client = AgentBusClient::new(channel);
///
/// // Wrap in Arc
/// let client = Arc::new(client);
///
/// // Use with Decider
/// let decider = Decider::new(client, agent_bus_id, policy, factory);
/// ```
#[derive(Clone)]
pub struct AgentBusClient {
    /// The gRPC client that talks to the remote AgentBus service
    client: AgentBusServiceClient<Channel>,
}

impl AgentBusClient {
    /// Create a new AgentBusClient wrapping a gRPC channel
    pub fn new(channel: Channel) -> Self {
        Self {
            client: AgentBusServiceClient::new(channel),
        }
    }

    /// Create a new AgentBusClient by connecting to a host:port
    pub async fn connect(addr: impl Into<String>) -> Result<Self> {
        let channel = Channel::from_shared(addr.into())?.connect().await?;
        Ok(Self::new(channel))
    }
}

impl AgentBus for AgentBusClient {
    async fn propose(&self, request: ProposeRequest) -> Result<ProposeResponse> {
        // Clone the client to get a mutable reference
        let mut client = self.client.clone();

        // Forward to gRPC client
        let response = client.propose(request.clone()).await.map_err(|e| {
            // Log the error for observability
            warn!(
                error = ?e,
                agent_bus_id = request.agent_bus_id,
                "gRPC client propose failed"
            );
            anyhow::anyhow!("gRPC client propose error: {}", e)
        })?;

        Ok(response.into_inner())
    }

    async fn poll(&self, request: PollRequest) -> Result<PollResponse> {
        // Clone the client to get a mutable reference
        let mut client = self.client.clone();

        // Forward to gRPC client
        let response = client.poll(request.clone()).await.map_err(|e| {
            // Log the error for observability
            warn!(
                error = ?e,
                agent_bus_id = request.agent_bus_id,
                start_position = request.start_log_position,
                max_entries = request.max_entries,
                "gRPC client poll failed"
            );
            anyhow::anyhow!("gRPC client poll error: {}", e)
        })?;

        Ok(response.into_inner())
    }
}
