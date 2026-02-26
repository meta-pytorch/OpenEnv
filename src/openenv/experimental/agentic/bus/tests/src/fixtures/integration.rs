// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

use std::net::SocketAddr;
use std::rc::Rc;

use agentbus_api::environment::RealEnvironment;
use agentbus_core::AgentBusHandler;
use agentbus_core::AgentBusServiceServer;
use agentbus_core::client::AgentBusClient;
use anyhow::Result;
use fbinit::FacebookInit;
use tokio::time::Duration;
use tonic::transport::Server;

use super::AgentBusTestFixture;
use super::IntegrationFixture;

pub struct IntegrationTestFixture {
    server_handle: tokio::task::JoinHandle<()>,
    #[allow(dead_code)]
    fb: FacebookInit,
    #[allow(dead_code)]
    addr: SocketAddr,
    env: Rc<RealEnvironment>,
    client: AgentBusClient,
}

impl AgentBusTestFixture for IntegrationTestFixture {
    type Env = RealEnvironment;
    type AgentBusImpl = AgentBusClient;

    fn get_env(&self) -> Rc<Self::Env> {
        self.env.clone()
    }

    fn create_impl(&self) -> Self::AgentBusImpl {
        // Per tonic docs: cloning is cheap and the underlying communication channel is shared.
        // https://docs.rs/tonic/latest/tonic/client/index.html
        self.client.clone()
    }
}

impl IntegrationTestFixture {
    pub async fn new_async(fb: FacebookInit) -> Result<Self> {
        let (server_handle, port) = create_test_grpc_server(fb).await?;
        let addr: SocketAddr = format!("127.0.0.1:{}", port).parse().unwrap();
        tokio::time::sleep(Duration::from_millis(500)).await;

        let env = Rc::new(RealEnvironment::new());

        // Create the client once
        let client = crate::common::integration_utils::make_client(fb, addr).await?;

        Ok(Self {
            server_handle,
            fb,
            addr,
            env,
            client,
        })
    }
}

impl Drop for IntegrationTestFixture {
    fn drop(&mut self) {
        self.server_handle.abort();
    }
}

impl IntegrationFixture for IntegrationTestFixture {
    async fn new_async(fb: FacebookInit) -> Result<Self> {
        Self::new_async(fb).await
    }
}

async fn create_test_grpc_server(fb: FacebookInit) -> Result<(tokio::task::JoinHandle<()>, u16)> {
    let port = crate::common::integration_utils::find_free_port().await?;
    let env = RealEnvironment::new();
    let handler = AgentBusHandler::new(fb, move || {
        use std::rc::Rc;

        use agentbus_simple::InMemoryAgentBus;

        InMemoryAgentBus::new(Rc::new(env))
    });

    let service = AgentBusServiceServer::new(handler);
    let addr: SocketAddr = format!("127.0.0.1:{}", port).parse().unwrap();

    let server_handle = tokio::spawn(async move {
        let _ = Server::builder().add_service(service).serve(addr).await;
    });

    Ok((server_handle, port))
}
