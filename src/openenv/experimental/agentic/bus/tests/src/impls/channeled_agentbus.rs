// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! ChanneledAgentBus: A channel-based wrapper around InMemoryAgentBusState
//!
//! This implementation wraps InMemoryAgentBusState and communicates via channels.
//! It supports latency injection via distributions.

use std::rc::Rc;

use agentbus_api::environment::Environment;
use agentbus_api::traits::*;
use agentbus_simple::InMemoryAgentBusState;
use futures::channel::mpsc::UnboundedReceiver;
use futures::channel::mpsc::UnboundedSender;
use futures::channel::mpsc::unbounded;
use futures::channel::oneshot;
use futures::stream::StreamExt;
use rand::distributions::Distribution;

use crate::simulator::Simulator;

/// Type aliases for cleaner signatures
type ProposeResult = anyhow::Result<ProposeResponse>;
type ProposeResponseTx = oneshot::Sender<ProposeResult>;

type PollResult = anyhow::Result<PollResponse>;
type PollResponseTx = oneshot::Sender<PollResult>;

/// Messages sent on the internal channel
#[derive(Debug)]
enum Message {
    Proposal {
        request: ProposeRequest,
        response_tx: ProposeResponseTx,
    },
    Poll {
        request: PollRequest,
        response_tx: PollResponseTx,
    },
}

/// ChanneledAgentBus - frontend handle for proposing to the bus
#[derive(Clone)]
pub struct ChanneledAgentBus {
    tx: UnboundedSender<Message>,
}

/// ChanneledAgentBusBackend - background worker that processes requests
pub struct ChanneledAgentBusBackend<D> {
    rx: UnboundedReceiver<Message>,
    state: InMemoryAgentBusState,
    latency_distribution: D,
    env: Rc<Simulator>,
}

impl ChanneledAgentBus {
    pub fn new<D>(
        latency_distribution: D,
        env: Rc<Simulator>,
    ) -> (Self, ChanneledAgentBusBackend<D>) {
        let (tx, rx) = unbounded();
        let frontend = Self { tx };
        let backend = ChanneledAgentBusBackend {
            rx,
            state: InMemoryAgentBusState::new(),
            latency_distribution,
            env,
        };
        (frontend, backend)
    }

    pub async fn propose(&self, request: ProposeRequest) -> ProposeResult {
        let (response_tx, response_rx) = oneshot::channel();

        self.tx
            .unbounded_send(Message::Proposal {
                request,
                response_tx,
            })
            .map_err(|_| anyhow::anyhow!("Failed to send proposal"))?;

        response_rx
            .await
            .map_err(|_| anyhow::anyhow!("Failed to receive response"))?
    }

    pub async fn poll(&self, request: PollRequest) -> PollResult {
        let (response_tx, response_rx) = oneshot::channel();

        self.tx
            .unbounded_send(Message::Poll {
                request,
                response_tx,
            })
            .map_err(|_| anyhow::anyhow!("Failed to send poll"))?;

        response_rx
            .await
            .map_err(|_| anyhow::anyhow!("Failed to receive response"))?
    }
}

impl AgentBus for ChanneledAgentBus {
    async fn propose(&self, request: ProposeRequest) -> anyhow::Result<ProposeResponse> {
        self.propose(request).await
    }

    async fn poll(&self, request: PollRequest) -> anyhow::Result<PollResponse> {
        self.poll(request).await
    }
}

impl<D: Distribution<u64>> ChanneledAgentBusBackend<D> {
    /// Run the background worker loop
    /// Processes incoming requests and delegates to InMemoryAgentBusServiceState
    pub async fn run(mut self) {
        while let Some(message) = self.rx.next().await {
            match message {
                Message::Proposal {
                    request,
                    response_tx,
                } => {
                    self.handle_proposal(request, response_tx).await;
                }
                Message::Poll {
                    request,
                    response_tx,
                } => {
                    self.handle_poll(request, response_tx).await;
                }
            }
        }
        tracing::debug!("ChanneledAgentBusBackend: channel closed, shutting down");
    }

    async fn handle_proposal(&mut self, request: ProposeRequest, response_tx: ProposeResponseTx) {
        let latency_ms = self
            .env
            .with_rng(|rng| self.latency_distribution.sample(rng));
        let latency = std::time::Duration::from_millis(latency_ms);

        self.env.sleep(latency).await;
        let result = self.state.propose(request);
        let _ = response_tx.send(result);
    }

    async fn handle_poll(&mut self, request: PollRequest, response_tx: PollResponseTx) {
        let latency_ms = self
            .env
            .with_rng(|rng| self.latency_distribution.sample(rng));
        let latency = std::time::Duration::from_millis(latency_ms);

        self.env.sleep(latency).await;

        let result = self.state.poll(request);
        let _ = response_tx.send(result);
    }
}
