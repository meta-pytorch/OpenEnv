// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! PyO3 extension that runs the complete agentbus infrastructure
//! (gRPC server + decider + voter) within the Python process.

use std::io::IsTerminal as _;
use std::io::stderr;
use std::rc::Rc;

use agentbus_core::RealEnvironment;
use agentbus_writeonce::InMemoryWriteOnceSpace;
use agentbus_writeonce::WriteOnceAgentBus;
use pyo3::prelude::*;
use server_py_ext_lib::StandaloneHandler;
use tokio::sync::mpsc::unbounded_channel;
use tokio::sync::oneshot;
use tracing_glog::Glog;
use tracing_glog::GlogFields;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Registry;
use tracing_subscriber::layer::SubscriberExt;

/// In-process AgentBus server accessible from Python.
///
/// Starts a gRPC server, decider, and voter all within the current process.
/// The gRPC server listens on [::] (all interfaces, IPv4+IPv6) with an OS-assigned port.
#[pyclass]
struct AgentBusInProcess {
    #[pyo3(get)]
    port: u16,
    shutdown_tx: Option<oneshot::Sender<()>>,
}

#[pymethods]
impl AgentBusInProcess {
    #[new]
    #[pyo3(signature = (
        agent_bus_id = "0".to_string(),
        port = 0,
        run_decider = true,
        run_voter = false,
        decider_poll_interval_ms = 100,
        voter_poll_interval_ms = 100,
    ))]
    fn new(
        agent_bus_id: String,
        port: u16,
        run_decider: bool,
        run_voter: bool,
        decider_poll_interval_ms: u64,
        voter_poll_interval_ms: u64,
    ) -> PyResult<Self> {
        let (propose_tx, propose_rx) = unbounded_channel();
        let (poll_tx, poll_rx) = unbounded_channel();

        let agent_bus_id_worker = agent_bus_id.clone();

        // Thread A: single-threaded worker with AgentBus + Decider + Voter
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create tokio runtime for agentbus worker");

            let local = tokio::task::LocalSet::new();
            local.block_on(&rt, async move {
                let env = Rc::new(RealEnvironment::new());
                let agent_bus = WriteOnceAgentBus::new(InMemoryWriteOnceSpace::new(), env);

                server_py_ext_lib::spawn_loops_and_run(
                    agent_bus,
                    &agent_bus_id_worker,
                    run_decider,
                    run_voter,
                    decider_poll_interval_ms,
                    voter_poll_interval_ms,
                    propose_rx,
                    poll_rx,
                )
                .await;
            });
        });

        let handler = StandaloneHandler::new(propose_tx, poll_tx);
        let (assigned_port, shutdown_tx) = server_py_ext_lib::start_grpc_server(handler, port)?;

        Ok(Self {
            port: assigned_port,
            shutdown_tx: Some(shutdown_tx),
        })
    }

    fn stop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}

impl Drop for AgentBusInProcess {
    fn drop(&mut self) {
        self.stop();
    }
}

#[pymodule]
fn server_py_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    init_logging();

    m.add_class::<AgentBusInProcess>()?;
    Ok(())
}

fn init_logging() {
    let fmt = tracing_subscriber::fmt::Layer::default()
        .with_ansi(stderr().is_terminal())
        .with_writer(std::io::stderr)
        .event_format(Glog::default().with_timer(tracing_glog::LocalTime::default()))
        .fmt_fields(GlogFields::default());

    let filter = EnvFilter::from_default_env();

    let subscriber = Registry::default().with(filter).with(fmt);
    // Use try variant: the host Python process may already have a tracing subscriber.
    let _ = tracing::subscriber::set_global_default(subscriber);
}
