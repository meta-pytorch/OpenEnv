// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Consolidated macro-based test invocations

use agentbus_tests::agent_bus_tests;
use agentbus_tests::buggy_tests;
use agentbus_tests::failure_injection_tests;
use agentbus_tests::fixtures::AgentBusTestFixture;
use agentbus_tests::fixtures::WriteOnceAgentBusGenericFixture;
use agentbus_tests::fixtures::integration::IntegrationTestFixture;
use agentbus_tests::fixtures::simtest::ChainedAgentBusBuggyPollFixture;
use agentbus_tests::fixtures::simtest::ChainedAgentBusFixture;
use agentbus_tests::fixtures::simtest::ChanneledAgentBusFixture;
use agentbus_tests::fixtures::simtest::FaultInjectingFixture;
use agentbus_tests::fixtures::simtest::SimpleMemoryFixture;
use agentbus_tests::integration_tests;
use agentbus_tests::write_once_space::fixtures::simtest::ChanneledWriteOnceSpaceFixture;
use agentbus_tests::write_once_space::fixtures::simtest::InMemoryWriteOnceSpaceFixture;
use agentbus_tests::write_once_space_tests;

#[rustfmt::skip]
mod tests {
use super::*;

// =============================================================================
// AgentBus tests
// =============================================================================

// Deterministic simulator-based tests for AgentBus implementations
agent_bus_tests!(SimpleMemoryFixture, simple_memory);
agent_bus_tests!(ChanneledAgentBusFixture, channeled_agentbus);
agent_bus_tests!(ChainedAgentBusFixture, chained_agentbus);
agent_bus_tests!(WriteOnceAgentBusGenericFixture<InMemoryWriteOnceSpaceFixture>, write_once_in_memory);
agent_bus_tests!(WriteOnceAgentBusGenericFixture<ChanneledWriteOnceSpaceFixture>, write_once_channeled);

// Verify that failure-injecting implementations correctly inject failures
failure_injection_tests!(FaultInjectingFixture<SimpleMemoryFixture>, simple_memory);
failure_injection_tests!(FaultInjectingFixture<ChanneledAgentBusFixture>, channeled_agentbus);
failure_injection_tests!(FaultInjectingFixture<ChainedAgentBusFixture>, chained_agentbus);
failure_injection_tests!(FaultInjectingFixture<WriteOnceAgentBusGenericFixture<InMemoryWriteOnceSpaceFixture>>, write_once_in_memory);
failure_injection_tests!(FaultInjectingFixture<WriteOnceAgentBusGenericFixture<ChanneledWriteOnceSpaceFixture>>, write_once_channeled);

// End-to-end tests with real Thrift server and client
integration_tests!(IntegrationTestFixture, integration);

// Test that buggy implementations are caught by the linearizability checker
// Seed is chosen to reliably trigger linearizability violation
buggy_tests!(ChainedAgentBusBuggyPollFixture, chained_agentbus_buggy_poll, 1);

// =============================================================================
// WriteOnceSpace tests
// =============================================================================

write_once_space_tests!(InMemoryWriteOnceSpaceFixture, in_memory);
write_once_space_tests!(ChanneledWriteOnceSpaceFixture, channeled);
}
