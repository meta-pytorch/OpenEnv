// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Module declarations
pub mod client;
pub mod decider;
pub mod server_lib;
pub mod vote_trackers;
pub mod voter;

// Re-export from agentbus_api
// Re-exports from proto
pub use agent_bus_proto_rust::agent_bus::agent_bus_service_server::AgentBusService as AgentBusServiceTrait;
pub use agent_bus_proto_rust::agent_bus::agent_bus_service_server::AgentBusServiceServer;
pub use agent_bus_proto_rust::agent_bus::*;
pub use agentbus_api::AgentBus;
pub use agentbus_api::Environment;
pub use agentbus_api::RealEnvironment;
use anyhow::Result as AnyhowResult;
use async_trait::async_trait;
use fbinit::FacebookInit;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::mpsc::unbounded_channel;
use tokio::sync::oneshot;
use tonic::Request;
use tonic::Response;
use tonic::Status;

//This is a thread-safe gRPC handler;
//Internally, it uses a channel to communicate with a single-threaded implementation of AgentBus.

#[derive(Clone)]
pub struct AgentBusHandler {
    _fb: FacebookInit,
    propose_tx: UnboundedSender<(
        ProposeRequest,
        oneshot::Sender<AnyhowResult<ProposeResponse>>,
    )>,
    poll_tx: UnboundedSender<(PollRequest, oneshot::Sender<AnyhowResult<PollResponse>>)>,
}

impl AgentBusHandler {
    /// Create a handler with a custom factory function
    /// The factory function will be called on a dedicated thread to create the AgentBus implementation
    pub fn new<F, T>(fb: FacebookInit, factory: F) -> Self
    where
        F: FnOnce() -> T + Send + 'static,
        T: AgentBus,
    {
        let (propose_tx, propose_rx) = unbounded_channel();
        let (poll_tx, poll_rx) = unbounded_channel();

        // Spawn a dedicated thread with a single-threaded runtime to run the AgentBus implementation
        // MB: We don't use tokio spawn, since that would require making the AgentBus implement Send;
        // thread overhead is okay since we don't expect large numbers of handlers to be created.
        // Also, we don't handle errors gracefully or join the thread; if it dies, client RPCs will fail.
        // This is reasonable for now since this is a long-lived service that runs while the process exists.
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create tokio runtime for worker");

            // Use LocalSet to enable spawn_local for running additional local tasks (e.g., decider)
            let local = tokio::task::LocalSet::new();
            local.block_on(&rt, async move {
                // Call the factory to create the implementation
                let agent_bus_impl = factory();
                Self::run_worker(agent_bus_impl, propose_rx, poll_rx).await;
            });
        });

        Self {
            _fb: fb,
            propose_tx,
            poll_tx,
        }
    }

    /// Worker task that processes messages from all channels
    /// Generic over any implementation of the AgentBus trait
    async fn run_worker<T: AgentBus>(
        agent_bus_impl: T,
        mut propose_rx: UnboundedReceiver<(
            ProposeRequest,
            oneshot::Sender<AnyhowResult<ProposeResponse>>,
        )>,
        mut poll_rx: UnboundedReceiver<(PollRequest, oneshot::Sender<AnyhowResult<PollResponse>>)>,
    ) {
        loop {
            tokio::select! {
                Some((request, response_tx)) = propose_rx.recv() => {
                    let result = agent_bus_impl.propose(request).await;
                    // Ignore send errors - client may have disconnected/timed out
                    let _ = response_tx.send(result);
                }
                Some((request, response_tx)) = poll_rx.recv() => {
                    let result = agent_bus_impl.poll(request).await;
                    // Ignore send errors - client may have disconnected/timed out
                    let _ = response_tx.send(result);
                }
                else => break,
            }
        }
    }

    /// Helper to handle channel-based RPC call
    async fn channel_call<Req, Resp>(
        tx: &UnboundedSender<(Req, oneshot::Sender<AnyhowResult<Resp>>)>,
        request: Req,
        method_name: &'static str,
    ) -> Result<Resp, Status> {
        let (response_tx, response_rx) = oneshot::channel();

        tx.send((request, response_tx))
            .map_err(|_| Status::internal(format!("{}: worker thread terminated", method_name)))?;

        response_rx
            .await
            .map_err(|_| Status::internal(format!("{}: worker thread terminated", method_name)))?
            .map_err(|e| Status::internal(format!("{}: {}", method_name, e)))
    }
}

#[async_trait]
impl AgentBusServiceTrait for AgentBusHandler {
    async fn propose(
        &self,
        request: Request<ProposeRequest>,
    ) -> Result<Response<ProposeResponse>, Status> {
        let response =
            Self::channel_call(&self.propose_tx, request.into_inner(), "propose").await?;
        Ok(Response::new(response))
    }

    async fn poll(&self, request: Request<PollRequest>) -> Result<Response<PollResponse>, Status> {
        let response = Self::channel_call(&self.poll_tx, request.into_inner(), "poll").await?;
        Ok(Response::new(response))
    }
}
