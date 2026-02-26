// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! AgentBus server shared library
//!
//! Provides common server functionality used by both the OSS binary
//! and the Meta-internal binary.

use std::io::IsTerminal;
use std::io::stderr;
use std::net::SocketAddr;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use fbinit::FacebookInit;
use futures::StreamExt;
use signal_hook::consts::signal::SIGINT;
use signal_hook::consts::signal::SIGTERM;
use signal_hook_tokio::Signals;
use tracing::error;
use tracing::info;
use tracing_glog::Glog;
use tracing_glog::GlogFields;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Registry;
use tracing_subscriber::layer::SubscriberExt;

use crate::AgentBus;
use crate::AgentBusHandler;
use crate::AgentBusServiceServer;
use crate::DeciderPolicy;
use crate::RealEnvironment;
use crate::client::AgentBusClient;
use crate::decider::Decider;
use crate::vote_trackers::create_vote_tracker_for_decider_policy;
use crate::voter::API_KEY_ENV_VAR;
use crate::voter::OpenAIClient;
use crate::voter::Voter;
use crate::voter::VoterConfig;

/// Parsed backend URL scheme.
pub enum ParsedBackend {
    /// In-memory backend (`memory://`)
    Memory,
    /// HTTP/HTTPS relay to an upstream gRPC server (`http://` or `https://`)
    Http(String),
    /// Unrecognized scheme; callers can handle additional schemes (e.g. `zippydb://`)
    Other,
}

/// Parse a backend URL into a known scheme.
pub fn parse_backend_url(url: &str) -> Result<ParsedBackend> {
    let (scheme, _) = url
        .split_once("://")
        .ok_or_else(|| anyhow::anyhow!("Invalid URL '{}': expected <scheme>://...", url))?;
    match scheme {
        "memory" => Ok(ParsedBackend::Memory),
        "http" | "https" => Ok(ParsedBackend::Http(url.to_string())),
        _ => Ok(ParsedBackend::Other),
    }
}

/// Run the server with an HTTP relay backend that forwards requests to an
/// upstream gRPC AgentBus server at the given URL.
pub fn run_relay_server(fb: FacebookInit, args: &CommonServerArgs, grpc_url: String) -> Result<()> {
    let endpoint = tonic::transport::Channel::from_shared(grpc_url)?;

    run_server(fb, args, move || {
        let channel = endpoint.connect_lazy();
        AgentBusClient::new(channel)
    })
}

/// Common server arguments shared by all AgentBus server variants.
/// Use `#[clap(flatten)]` to embed these in your binary's `Args` struct.
#[derive(clap::Args, Debug)]
pub struct CommonServerArgs {
    /// Port to listen on for gRPC connections
    #[clap(short, long, env = "AGENT_BUS_PORT", default_value = "9999")]
    pub port: u16,

    /// Enable the in-process decider for the specified agent_bus_id
    #[clap(long)]
    pub run_decider: Option<String>,

    /// Decider polling interval in milliseconds
    #[clap(long, env = "DECIDER_POLL_INTERVAL_MS", default_value = "100")]
    pub decider_poll_interval_ms: u64,

    /// Enable the in-process voter for the specified agent_bus_id
    #[clap(long)]
    pub run_voter: Option<String>,

    /// Voter polling interval in milliseconds
    #[clap(long, env = "VOTER_POLL_INTERVAL_MS", default_value = "100")]
    pub voter_poll_interval_ms: u64,

    /// Override OpenAI-compatible API endpoint for the voter
    /// Default: https://api.openai.com/v1 (OpenAI-compatible endpoint)
    #[clap(long, env = "VOTER_API_ENDPOINT")]
    pub voter_api_endpoint: Option<String>,

    /// Override API key for the voter's LLM endpoint
    /// Default: reads from LLM_API_KEY environment variable
    #[clap(long, env = "VOTER_API_KEY")]
    pub voter_api_key: Option<String>,

    /// Override LLM model name for the voter
    /// Default: claude-sonnet-4-5 (or LLM_MODEL env var)
    #[clap(long, env = "VOTER_MODEL")]
    pub voter_model: Option<String>,
}

pub fn init_logging() {
    let fmt = tracing_subscriber::fmt::Layer::default()
        .with_ansi(stderr().is_terminal())
        .with_writer(std::io::stderr)
        .event_format(Glog::default().with_timer(tracing_glog::LocalTime::default()))
        .fmt_fields(GlogFields::default());

    let filter = EnvFilter::from_default_env();

    let subscriber = Registry::default().with(filter).with(fmt);
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set global subscriber");
}

pub async fn run_grpc_service(handler: AgentBusHandler, port: u16) -> Result<()> {
    let addr: SocketAddr = format!("0.0.0.0:{}", port).parse()?;

    let service = AgentBusServiceServer::new(handler);

    let mut signals = Signals::new([SIGTERM, SIGINT])?;

    let server = tonic::transport::Server::builder()
        .add_service(service)
        .serve_with_shutdown(addr, async move {
            signals.next().await;
            info!("Shutting down...");
            signals.handle().close();
        });

    server.await?;
    Ok(())
}

pub async fn run_decider_loop<T: AgentBus>(
    agent_bus: T,
    agent_bus_id: String,
    poll_interval_ms: u64,
) {
    let poll_interval = Duration::from_millis(poll_interval_ms);
    let environment = Rc::new(RealEnvironment::new());
    let initial_policy = DeciderPolicy::OnByDefault as i32;

    let mut decider = Decider::new(
        agent_bus,
        agent_bus_id.clone(),
        initial_policy,
        create_vote_tracker_for_decider_policy,
        environment,
    );

    info!(
        agent_bus_id = %agent_bus_id,
        poll_interval_ms = poll_interval_ms,
        "Starting in-process decider with ON_BY_DEFAULT policy"
    );

    if let Err(e) = decider.run(poll_interval).await {
        error!(error = ?e, "Decider error");
    }
}

fn create_voter_config(args: &CommonServerArgs) -> Option<(String, VoterConfig)> {
    let agent_bus_id = args.run_voter.clone()?;

    let mut config = VoterConfig::default();
    if let Some(ref endpoint) = args.voter_api_endpoint {
        config.api_endpoint = endpoint.clone();
    }
    if let Some(ref key) = args.voter_api_key {
        config.api_key = Some(key.clone());
    }
    if let Some(ref model) = args.voter_model {
        config.model = model.clone();
    }
    assert!(
        config.api_key.is_some(),
        "No API key found for voter LLM client. Set {} env var or use --voter-api-key.",
        API_KEY_ENV_VAR
    );

    Some((agent_bus_id, config))
}

fn create_voter<T: AgentBus + Clone>(
    agent_bus: &T,
    agent_bus_id: String,
    config: VoterConfig,
) -> Voter<T, RealEnvironment> {
    let llm_client = Arc::new(OpenAIClient::new(
        config.api_endpoint.clone(),
        config.api_key.clone(),
    ));
    let environment = Rc::new(RealEnvironment::new());

    info!(
        agent_bus_id = %agent_bus_id,
        model = %config.model,
        endpoint = %config.api_endpoint,
        "Creating voter with OpenAI-compatible LLM client"
    );

    Voter::with_llm_client(
        agent_bus.clone(),
        agent_bus_id,
        environment,
        llm_client,
        config,
    )
}

pub async fn run_voter_loop<T: AgentBus, E: crate::Environment>(
    mut voter: Voter<T, E>,
    poll_interval_ms: u64,
) {
    let poll_interval = Duration::from_millis(poll_interval_ms);

    info!(poll_interval_ms = poll_interval_ms, "Starting voter");

    if let Err(e) = voter.run(poll_interval).await {
        error!(error = ?e, "Voter error");
    }
}

/// Run the AgentBus server with the given bus factory.
///
/// This is the main entry point for both OSS and Meta server binaries.
/// The `make_bus` closure creates the storage backend on the worker thread.
pub fn run_server<T, F>(fb: FacebookInit, args: &CommonServerArgs, make_bus: F) -> Result<()>
where
    T: AgentBus + Clone + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    let run_decider = args.run_decider.clone();
    let decider_poll_interval_ms = args.decider_poll_interval_ms;
    let voter_poll_interval_ms = args.voter_poll_interval_ms;
    let voter_config = create_voter_config(args);
    let port = args.port;

    // Clone for logging after the closure moves the originals
    let decider_id_for_log = args.run_decider.clone();
    let voter_id_for_log = args.run_voter.clone();

    let handler = AgentBusHandler::new(fb, move || {
        let agent_bus = make_bus();

        if let Some(agent_bus_id) = run_decider {
            let decider_bus = agent_bus.clone();
            tokio::task::spawn_local(run_decider_loop(
                decider_bus,
                agent_bus_id,
                decider_poll_interval_ms,
            ));
        }

        if let Some((agent_bus_id, config)) = voter_config {
            let voter = create_voter(&agent_bus, agent_bus_id, config);
            tokio::task::spawn_local(run_voter_loop(voter, voter_poll_interval_ms));
        }

        agent_bus
    });

    match (&decider_id_for_log, &voter_id_for_log) {
        (Some(decider_id), Some(voter_id)) => {
            info!(
                port = port,
                decider_agent_bus_id = %decider_id,
                voter_agent_bus_id = %voter_id,
                decider_poll_interval_ms = decider_poll_interval_ms,
                voter_poll_interval_ms = voter_poll_interval_ms,
                "Starting AgentBusService gRPC service with in-process decider and voter"
            );
        }
        (Some(agent_bus_id), None) => {
            info!(
                port = port,
                agent_bus_id = %agent_bus_id,
                poll_interval_ms = decider_poll_interval_ms,
                "Starting AgentBusService gRPC service with in-process decider"
            );
        }
        (None, Some(agent_bus_id)) => {
            info!(
                port = port,
                agent_bus_id = %agent_bus_id,
                poll_interval_ms = voter_poll_interval_ms,
                "Starting AgentBusService gRPC service with in-process voter"
            );
        }
        (None, None) => {
            info!(port = port, "Starting AgentBusService gRPC service");
        }
    }

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(run_grpc_service(handler, port))
}
