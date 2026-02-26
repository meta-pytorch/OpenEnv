// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

use std::rc::Rc;

use agentbus_core::RealEnvironment;
use agentbus_core::server_lib::CommonServerArgs;
use agentbus_core::server_lib::ParsedBackend;
use agentbus_writeonce::InMemoryWriteOnceSpace;
use agentbus_writeonce::WriteOnceAgentBus;
use anyhow::Result;
use clap::Parser;
use fbinit::FacebookInit;

#[derive(Parser, Debug)]
#[clap(name = "agent_bus_service", about = "AgentBus gRPC service")]
struct Args {
    #[clap(flatten)]
    common: CommonServerArgs,

    /// Backend URL (memory://, http://host:port, or https://host:port)
    #[clap(long)]
    agentbus: Option<String>,
}

#[fbinit::main]
fn main(fb: FacebookInit) -> Result<()> {
    agentbus_core::server_lib::init_logging();

    let args = Args::parse();

    if let Some(url) = &args.agentbus {
        match agentbus_core::server_lib::parse_backend_url(url)? {
            ParsedBackend::Memory => {}
            ParsedBackend::Http(grpc_url) => {
                return agentbus_core::server_lib::run_relay_server(fb, &args.common, grpc_url);
            }
            ParsedBackend::Other => {
                anyhow::bail!(
                    "Unsupported URL '{}'. Supported schemes: memory://, http://, https://",
                    url
                );
            }
        }
    }

    agentbus_core::server_lib::run_server(fb, &args.common, || {
        let env = Rc::new(RealEnvironment::new());
        WriteOnceAgentBus::new(InMemoryWriteOnceSpace::new(), env)
    })
}
