// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! DeciderService - Standalone service that runs a Decider against a remote AgentBus
//!
//! This service connects to a remote AgentBus via thrift and continuously polls for
//! new entries, making commit/abort decisions based on the configured voting policy.

use std::io::IsTerminal;
use std::io::stderr;
use std::rc::Rc;
use std::time::Duration;

use agent_bus_proto_rust::agent_bus::*;
use agentbus_core::AgentBus;
use agentbus_core::RealEnvironment;
// AgentBus imports
use agentbus_core::client::AgentBusClient;
use agentbus_core::decider::Decider;
use agentbus_core::vote_trackers::create_vote_tracker_for_decider_policy;
use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use fbinit::FacebookInit;
use futures::StreamExt;
use signal_hook::consts::signal::SIGINT;
use signal_hook::consts::signal::SIGTERM;
use signal_hook_tokio::Signals;
use tokio_retry::Retry;
use tokio_retry::strategy::ExponentialBackoff;
use tracing::Level;
use tracing::info;
use tracing_glog::Glog;
use tracing_glog::GlogFields;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Registry;
use tracing_subscriber::layer::SubscriberExt;

/// Command-line arguments for DeciderService
#[derive(Parser, Debug)]
#[clap(name = "decider_service", about = "AgentBus Decider Service")]
struct Args {
    /// The agent bus ID to operate on
    #[clap(long, env = "AGENT_BUS_ID")]
    agent_bus_id: String,

    /// Direct host:port to connect to (e.g., "localhost:9999").
    /// If specified, bypasses service discovery.
    #[clap(long, env = "AGENT_BUS_HOST", conflicts_with = "tiername")]
    host: Option<String>,

    /// Tier name for service discovery (e.g., "agentbus.prod")
    #[clap(long, env = "TIERNAME", default_value = "agentbus.prod")]
    tiername: String,

    /// Polling interval in milliseconds
    #[clap(long, env = "POLL_INTERVAL_MS", default_value = "100")]
    poll_interval_ms: u64,

    /// Log level (trace, debug, info, warn, error)
    #[clap(long, env = "LOG_LEVEL", default_value_t = Level::INFO)]
    log_level: Level,
}

#[fbinit::main]
async fn main(_fb: FacebookInit) -> Result<()> {
    let args = Args::parse();

    init_logging(args.log_level)?;

    // Use ON_BY_DEFAULT decider policy
    let initial_decider_policy = DeciderPolicy::OnByDefault as i32;

    info!(
        agent_bus_id = args.agent_bus_id,
        connection = if args.host.is_some() {
            "direct"
        } else {
            "service_discovery"
        },
        poll_interval_ms = args.poll_interval_ms,
        "Starting DeciderService with ON_BY_DEFAULT decider policy"
    );

    // Create gRPC client to connect to remote AgentBus
    let endpoint = if let Some(host) = &args.host {
        // Direct connection to host:port
        info!(
            host = %host,
            "Connecting to AgentBus at {}", host
        );

        format!("http://{}", host)
    } else {
        // Service discovery via tiername would go here
        // For now, use a default endpoint
        info!(
            "Connecting to AgentBus via service discovery: {}",
            args.tiername
        );

        // TODO: Implement service discovery for gRPC
        return Err(anyhow::anyhow!(
            "Service discovery not yet implemented for gRPC. Please use --host to specify a direct connection."
        ));
    };

    // Create gRPC client
    let agent_bus_client = AgentBusClient::connect(&endpoint)
        .await
        .context("Failed to create AgentBus gRPC client")?;

    wait_for_agentbus_connection(&agent_bus_client, &args.agent_bus_id).await?;

    // Create RealEnvironment for production use
    let environment = Rc::new(RealEnvironment::new());

    // Create Decider using AgentBusClient and RealEnvironment
    let mut decider = Decider::new(
        agent_bus_client,
        args.agent_bus_id,
        initial_decider_policy,
        create_vote_tracker_for_decider_policy,
        environment,
    );

    // Set up signal handler for graceful shutdown
    let mut signals =
        Signals::new([SIGTERM, SIGINT]).context("Failed to set up signal handlers")?;
    let signal_handle = signals.handle();

    // Spawn signal handler task
    let shutdown_signal = tokio::spawn(async move {
        signals.next().await;
        info!("Received shutdown signal");
    });

    // Run the decider using its built-in run method with shutdown handling
    let poll_interval = Duration::from_millis(args.poll_interval_ms);
    let result = tokio::select! {
        _ = shutdown_signal => {
            info!("Shutting down gracefully");
            Ok(())
        }
        result = decider.run(poll_interval) => result.map_err(anyhow::Error::from),
    };

    // Clean up
    signal_handle.close();

    result
}

/// Waits for AgentBus to become available with exponential backoff retry.
async fn wait_for_agentbus_connection(
    agent_bus_client: &AgentBusClient,
    agent_bus_id: &str,
) -> Result<()> {
    let retry_strategy = ExponentialBackoff::from_millis(1000)
        .max_delay(Duration::from_secs(16))
        .take(5);

    let result = Retry::spawn(retry_strategy, || async {
        let poll_request = PollRequest {
            agent_bus_id: agent_bus_id.to_string(),
            start_log_position: 0,
            max_entries: 1,
            filter: None,
            ..Default::default()
        };

        agent_bus_client
            .poll(poll_request)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to poll AgentBus: {:?}", e))
    })
    .await;

    match result {
        Ok(_) => {
            info!("Successfully connected to AgentBus");
            Ok(())
        }
        Err(e) => {
            tracing::error!(
                agent_bus_id = agent_bus_id,
                error = ?e,
                "Failed to connect to AgentBus after all retries"
            );
            Err(e)
        }
    }
}

/// Initialize logging with glog format
fn init_logging(log_level: Level) -> Result<()> {
    let fmt = tracing_subscriber::fmt::Layer::default()
        .with_ansi(stderr().is_terminal())
        .with_writer(std::io::stderr)
        .event_format(Glog::default().with_timer(tracing_glog::LocalTime::default()))
        .fmt_fields(GlogFields::default());

    // Build the filter from the log level argument
    // Set the log level for the agent_bus_decider crate and the decider_service binary
    let level_str = log_level.as_str();
    let filter_str = format!(
        "agent_bus_decider={},decider_service={}",
        level_str, level_str
    );
    let filter = EnvFilter::try_new(&filter_str)
        .with_context(|| format!("Invalid log level: {}", level_str))?;

    let subscriber = Registry::default().with(filter).with(fmt);
    tracing::subscriber::set_global_default(subscriber)
        .context("Failed to set global subscriber")?;

    Ok(())
}
