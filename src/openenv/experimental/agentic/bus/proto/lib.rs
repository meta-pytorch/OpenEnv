// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// This file is only used for Cargo builds, not Buck builds
// For Cargo: this is the crate root that includes generated proto code

pub mod agent_bus {
    tonic::include_proto!("agent_bus");
}
