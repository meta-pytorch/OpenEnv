// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Generic fixture for WriteOnceAgentBus composed from WriteOnceSpace fixtures.
//!
//! This enables recursive fixture composition - AgentBus fixtures can be built
//! from WriteOnceSpace fixtures, inheriting their construction traits.

use std::rc::Rc;

use agentbus_writeonce::WriteOnceAgentBus;
use anyhow::Result;
use fbinit::FacebookInit;

use super::AgentBusTestFixture;
use super::IntegrationFixture;
use super::SimulatorFixture;
use crate::simulator::Simulator;
use crate::write_once_space::fixtures::WriteOnceSpaceTestFixture;

/// Generic fixture for WriteOnceAgentBus composed from WriteOnceSpace fixtures.
///
/// This allows AgentBus fixtures to be composed from WriteOnceSpace fixtures,
/// enabling recursive fixture composition. The fixture inherits the construction
/// trait from the underlying space fixture.
pub struct WriteOnceAgentBusGenericFixture<WOF: WriteOnceSpaceTestFixture>
where
    WOF::WriteOnceSpaceImpl: Clone,
{
    space_fixture: Rc<WOF>,
    agentbus: WriteOnceAgentBus<WOF::WriteOnceSpaceImpl, WOF::Env>,
}

impl<WOF: WriteOnceSpaceTestFixture + 'static> WriteOnceAgentBusGenericFixture<WOF>
where
    WOF::WriteOnceSpaceImpl: Clone + 'static,
{
    pub fn new(space_fixture: Rc<WOF>) -> Self {
        let space = space_fixture.create_impl();
        let env = space_fixture.get_env();
        Self {
            agentbus: WriteOnceAgentBus::new(space, env),
            space_fixture,
        }
    }
}

impl<WOF: WriteOnceSpaceTestFixture + 'static> AgentBusTestFixture
    for WriteOnceAgentBusGenericFixture<WOF>
where
    WOF::WriteOnceSpaceImpl: Clone + 'static,
{
    type Env = WOF::Env;
    type AgentBusImpl = WriteOnceAgentBus<WOF::WriteOnceSpaceImpl, WOF::Env>;

    fn get_env(&self) -> Rc<Self::Env> {
        self.space_fixture.get_env()
    }

    fn create_impl(&self) -> Self::AgentBusImpl {
        self.agentbus.clone()
    }
}

impl<WOF> SimulatorFixture for WriteOnceAgentBusGenericFixture<WOF>
where
    WOF: WriteOnceSpaceTestFixture + SimulatorFixture + 'static,
    WOF::WriteOnceSpaceImpl: Clone + 'static,
{
    fn new(simulator: Simulator) -> Self {
        let space_fixture = Rc::new(WOF::new(simulator));
        Self::new(space_fixture)
    }
}

impl<WOF> IntegrationFixture for WriteOnceAgentBusGenericFixture<WOF>
where
    WOF: WriteOnceSpaceTestFixture + IntegrationFixture + 'static,
    WOF::WriteOnceSpaceImpl: Clone + 'static,
{
    async fn new_async(fb: FacebookInit) -> Result<Self> {
        let space_fixture = Rc::new(WOF::new_async(fb).await?);
        Ok(Self::new(space_fixture))
    }
}
