// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! FaultInjectingAgentBus: A wrapper that injects faults into any AgentBus implementation
//!
//! This wrapper takes any AgentBus and adds fault injection capabilities based on FaultConfig.

use std::rc::Rc;

use agentbus_api::environment::Environment;
use agentbus_api::traits::*;

use crate::common::fault_config::Fate;
use crate::common::fault_config::FaultConfig;
use crate::simulator::Simulator;

/// FaultInjectingAgentBus - wraps any AgentBus with fault injection
pub struct FaultInjectingAgentBus<A: AgentBus> {
    inner: Rc<A>,
    fault_config: FaultConfig,
    env: Rc<Simulator>,
}

impl<A: AgentBus> Clone for FaultInjectingAgentBus<A> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            fault_config: self.fault_config.clone(),
            env: self.env.clone(),
        }
    }
}

impl<A: AgentBus> FaultInjectingAgentBus<A> {
    pub fn new(inner: A, fault_config: FaultConfig, env: Rc<Simulator>) -> Result<Self, String> {
        fault_config.validate()?;
        Ok(Self {
            inner: Rc::new(inner),
            fault_config,
            env,
        })
    }

    fn determine_fate(&self) -> Fate {
        self.env
            .with_rng(|rng| self.fault_config.determine_fate(rng))
    }
}

impl<A: AgentBus + 'static> AgentBus for FaultInjectingAgentBus<A> {
    async fn propose(&self, request: ProposeRequest) -> anyhow::Result<ProposeResponse> {
        let fate = self.determine_fate();

        match fate {
            Fate::Success => self.inner.propose(request).await,
            Fate::Lost => Err(anyhow::anyhow!("Lost")),
            Fate::CommitThenError => {
                let _ = self.inner.propose(request).await;
                Err(anyhow::anyhow!("CommitThenError"))
            }
            Fate::ErrorThenCommit => {
                let inner_clone = self.inner.clone();
                self.env.spawn(async move {
                    let _ = inner_clone.propose(request).await;
                });
                Err(anyhow::anyhow!("ErrorThenCommit"))
            }
        }
    }

    async fn poll(&self, request: PollRequest) -> anyhow::Result<PollResponse> {
        self.inner.poll(request).await
    }
}
