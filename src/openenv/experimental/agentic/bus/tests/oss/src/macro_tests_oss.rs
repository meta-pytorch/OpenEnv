// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! OSS integration tests (eg. DynamoDB integration tests)

mod write_once_space;

use agentbus_tests::fixtures::WriteOnceAgentBusGenericFixture;
use agentbus_tests::integration_tests;
use agentbus_tests::write_once_space_integration_tests;
use write_once_space::fixtures::DynamoWriteOnceSpaceFixture;

#[rustfmt::skip]
mod tests {
use super::*;

// =============================================================================
// AgentBus tests
// =============================================================================

integration_tests!(WriteOnceAgentBusGenericFixture<DynamoWriteOnceSpaceFixture>, dynamo);

// =============================================================================
// WriteOnceSpace tests
// =============================================================================

write_once_space_integration_tests!(DynamoWriteOnceSpaceFixture, dynamo);
}
