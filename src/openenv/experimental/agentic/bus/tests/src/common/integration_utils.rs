// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Integration test utilities for gRPC client/server testing

use std::net::SocketAddr;

use agentbus_core::client::AgentBusClient;
use anyhow::Result;
use fbinit::FacebookInit;

pub async fn find_free_port() -> Result<u16> {
    use tokio::net::TcpListener;
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

pub async fn make_client(_fb: FacebookInit, addr: SocketAddr) -> Result<AgentBusClient> {
    let endpoint = format!("http://{}", addr);
    AgentBusClient::connect(&endpoint).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[fbinit::test]
    async fn test_error_handling(fb: FacebookInit) -> Result<()> {
        let port = find_free_port().await?;
        let addr: SocketAddr = format!("127.0.0.1:{}", port).parse().unwrap();

        // Try to connect to a server that doesn't exist - should fail
        let client_result = make_client(fb, addr).await;

        match client_result {
            Ok(_) => Err(anyhow::anyhow!(
                "Expected connection to fail when server unavailable"
            )),
            Err(_) => Ok(()),
        }
    }
}
