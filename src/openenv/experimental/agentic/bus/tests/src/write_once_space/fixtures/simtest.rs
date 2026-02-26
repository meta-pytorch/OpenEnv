// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Test fixtures for WriteOnceSpace implementations using the simulator

use std::rc::Rc;

use agentbus_writeonce::ChanneledWriteOnceSpace;
use agentbus_writeonce::InMemoryWriteOnceSpace;

use super::WriteOnceSpaceTestFixture;
use crate::fixtures::SimulatorFixture;
use crate::simulator::Simulator;

/// Fixture for testing InMemoryWriteOnceSpace directly.
pub struct InMemoryWriteOnceSpaceFixture {
    space: InMemoryWriteOnceSpace,
    env: Rc<Simulator>,
}

impl WriteOnceSpaceTestFixture for InMemoryWriteOnceSpaceFixture {
    type Env = Simulator;
    type WriteOnceSpaceImpl = InMemoryWriteOnceSpace;

    fn get_env(&self) -> Rc<Self::Env> {
        self.env.clone()
    }

    fn create_impl(&self) -> Self::WriteOnceSpaceImpl {
        self.space.clone()
    }
}

impl SimulatorFixture for InMemoryWriteOnceSpaceFixture {
    fn new(simulator: Simulator) -> Self {
        Self {
            space: InMemoryWriteOnceSpace::new(),
            env: Rc::new(simulator),
        }
    }
}

/// Fixture for testing ChanneledWriteOnceSpace explicitly.
pub struct ChanneledWriteOnceSpaceFixture {
    space: ChanneledWriteOnceSpace,
    env: Rc<Simulator>,
}

impl WriteOnceSpaceTestFixture for ChanneledWriteOnceSpaceFixture {
    type Env = Simulator;
    type WriteOnceSpaceImpl = ChanneledWriteOnceSpace;

    fn get_env(&self) -> Rc<Self::Env> {
        self.env.clone()
    }

    fn create_impl(&self) -> Self::WriteOnceSpaceImpl {
        self.space.clone()
    }
}

impl SimulatorFixture for ChanneledWriteOnceSpaceFixture {
    fn new(simulator: Simulator) -> Self {
        let env = Rc::new(simulator);
        let (channeled_space, space_backend) =
            ChanneledWriteOnceSpace::new(InMemoryWriteOnceSpace::new());
        let _space_handle = env.spawn(space_backend.run());
        Self {
            space: channeled_space,
            env,
        }
    }
}
