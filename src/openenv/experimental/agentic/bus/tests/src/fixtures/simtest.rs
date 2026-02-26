// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Test fixtures for different AgentBus implementations

use std::rc::Rc;

use agentbus_simple::InMemoryAgentBus;
use rand::distributions::Uniform;

use super::AgentBusTestFixture;
use super::SimulatorFixture;
use crate::common::fault_config::FaultConfig;
use crate::impls::chained_agentbus::ChainedAgentBus;
use crate::impls::channeled_agentbus::ChanneledAgentBus;
use crate::impls::fault_injecting_agentbus::FaultInjectingAgentBus;
use crate::simulator::Simulator;

/// Fixture for testing InMemoryAgentBus
/// Owns an InMemoryAgentBus instance that can be cloned to share state
pub struct SimpleMemoryFixture {
    agentbus: InMemoryAgentBus<Simulator>,
}

impl AgentBusTestFixture for SimpleMemoryFixture {
    type Env = Simulator;
    type AgentBusImpl = InMemoryAgentBus<Simulator>;

    fn get_env(&self) -> Rc<Self::Env> {
        self.agentbus.environment()
    }

    fn create_impl(&self) -> Self::AgentBusImpl {
        // self.agentbus owns an Rc to the backend state; the Rc is cloned here.
        self.agentbus.clone()
    }
}

impl SimulatorFixture for SimpleMemoryFixture {
    fn new(simulator: Simulator) -> Self {
        Self {
            agentbus: InMemoryAgentBus::new(Rc::new(simulator)),
        }
    }
}

/// Fixture for testing ChanneledAgentBus (wrapper around InMemoryAgentBus)
pub struct ChanneledAgentBusFixture {
    agentbus: ChanneledAgentBus,
    env: Rc<Simulator>,
}

impl AgentBusTestFixture for ChanneledAgentBusFixture {
    type Env = Simulator;
    type AgentBusImpl = ChanneledAgentBus;

    fn get_env(&self) -> Rc<Self::Env> {
        self.env.clone()
    }

    fn create_impl(&self) -> Self::AgentBusImpl {
        // self.agentbus is a wrapper around a channel sender; the sender is cloned here. The
        // receiver and backend state are owned by the `run()` task in the simulator.
        self.agentbus.clone()
    }
}

impl SimulatorFixture for ChanneledAgentBusFixture {
    fn new(simulator: Simulator) -> Self {
        let env = Rc::new(simulator);
        let (frontend, backend) = ChanneledAgentBus::new(Uniform::new(0, 10), env.clone());
        let _backend_handle = env.spawn(backend.run());
        Self {
            agentbus: frontend,
            env,
        }
    }
}

/// Generic fixture for testing any AgentBus implementation with fault injection
pub struct FaultInjectingFixture<F: SimulatorFixture + AgentBusTestFixture<Env = Simulator>> {
    inner: F,
    config: FaultConfig,
}

impl<F: SimulatorFixture + AgentBusTestFixture<Env = Simulator>> FaultInjectingFixture<F> {
    pub fn new_with_config(simulator: Simulator, config: FaultConfig) -> Self {
        Self {
            inner: F::new(simulator),
            config,
        }
    }
}

impl<F> AgentBusTestFixture for FaultInjectingFixture<F>
where
    F: SimulatorFixture + AgentBusTestFixture<Env = Simulator>,
    F::AgentBusImpl: agentbus_api::AgentBus + 'static,
{
    type Env = Simulator;
    type AgentBusImpl = FaultInjectingAgentBus<F::AgentBusImpl>;

    fn get_env(&self) -> Rc<Self::Env> {
        self.inner.get_env()
    }

    fn create_impl(&self) -> Self::AgentBusImpl {
        let backing_bus = self.inner.create_impl();
        FaultInjectingAgentBus::new(backing_bus, self.config.clone(), self.inner.get_env())
            .expect("Failed to create FaultInjectingAgentBus")
    }
}

impl<F> SimulatorFixture for FaultInjectingFixture<F>
where
    F: SimulatorFixture + AgentBusTestFixture<Env = Simulator>,
    F::AgentBusImpl: agentbus_api::AgentBus + 'static,
{
    fn new(simulator: Simulator) -> Self {
        Self::new_with_config(
            simulator,
            FaultConfig {
                prob_lost: 0.0,
                prob_commit_then_error: 0.0,
                prob_error_then_commit: 0.0,
            },
        )
    }
}

/// Generic fixture for testing ChainedAgentBus with configurable mode
pub struct ChainedAgentBusGenericFixture<const MODE: u8> {
    agentbus: ChainedAgentBus,
}

impl<const MODE: u8> AgentBusTestFixture for ChainedAgentBusGenericFixture<MODE> {
    type Env = Simulator;
    type AgentBusImpl = ChainedAgentBus;

    fn get_env(&self) -> Rc<Self::Env> {
        self.agentbus.environment()
    }

    fn create_impl(&self) -> Self::AgentBusImpl {
        self.agentbus.clone()
    }
}

impl<const MODE: u8> SimulatorFixture for ChainedAgentBusGenericFixture<MODE> {
    fn new(simulator: Simulator) -> Self {
        let env = Rc::new(simulator);
        let agentbus = match MODE {
            0 => ChainedAgentBus::new_correct(env),
            1 => ChainedAgentBus::new_buggy_poll(env),
            _ => panic!("Invalid mode"),
        };
        Self { agentbus }
    }
}

/// Correct chain replication mode
pub type ChainedAgentBusFixture = ChainedAgentBusGenericFixture<0>;

/// Buggy poll mode for linearizability checker testing
pub type ChainedAgentBusBuggyPollFixture = ChainedAgentBusGenericFixture<1>;
