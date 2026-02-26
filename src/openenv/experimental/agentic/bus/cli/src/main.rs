// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! AgentBus CLI - Command-line tool for interacting with AgentBus via gRPC

use agentbus_cli_lib::CommonArgs;
use anyhow::Result;
use clap::Parser;
use fbinit::FacebookInit;

/// AgentBus CLI - Command-line interface for AgentBus operations
#[derive(Parser, Debug)]
#[clap(name = "agentbus", about = "AgentBus command-line interface")]
struct Args {
    #[clap(flatten)]
    common: CommonArgs,
}

#[fbinit::main]
async fn main(fb: FacebookInit) -> Result<()> {
    agentbus_cli_lib::init_logging();

    let args = Args::parse();
    let client = args.common.connect_grpc().await?;

    agentbus_cli_lib::run(&client, args.common.agent_bus_id, args.common.command).await
}
