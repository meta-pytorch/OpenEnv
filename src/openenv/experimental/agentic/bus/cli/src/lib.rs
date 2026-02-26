// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! AgentBus CLI shared library
//!
//! Provides commands, REPL, and helpers for interacting with AgentBus.
//! Used by both the OSS CLI and the Meta-internal CLI.

use std::io::IsTerminal;
use std::io::stderr;

use agent_bus_proto_rust::agent_bus::*;
use agentbus_api::AgentBus;
use agentbus_core::client::AgentBusClient;
use anyhow::Context;
use anyhow::Result;
use clap::Subcommand;
use colored::Colorize;
use tracing::info;
use tracing_glog::Glog;
use tracing_glog::GlogFields;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Registry;
use tracing_subscriber::layer::SubscriberExt;

/// Common CLI arguments shared by all AgentBus CLI variants.
/// Use `#[clap(flatten)]` to embed these in your binary's `Args` struct.
#[derive(clap::Args, Debug)]
pub struct CommonArgs {
    /// Agent bus ID to operate on
    #[clap(long, env = "AGENT_BUS_ID")]
    pub agent_bus_id: String,

    /// Direct host:port to connect to (e.g., "localhost:9999").
    /// If specified, bypasses service discovery.
    #[clap(long, env = "AGENT_BUS_HOST", conflicts_with = "tiername")]
    pub host: Option<String>,

    /// Tier name for service discovery (e.g., "agentbus.prod")
    #[clap(long, env = "TIERNAME", default_value = "agentbus.prod")]
    pub tiername: String,

    #[clap(subcommand)]
    pub command: Commands,
}

impl CommonArgs {
    /// Connect to AgentBus via gRPC, using either direct host or service discovery.
    pub async fn connect_grpc(&self) -> Result<AgentBusClient> {
        let endpoint = if let Some(host) = &self.host {
            info!(
                agent_bus_id = self.agent_bus_id,
                host = %host,
                "Connecting to AgentBus at {}", host
            );
            format!("http://{}", host)
        } else {
            info!(
                agent_bus_id = self.agent_bus_id,
                tiername = %self.tiername,
                "Connecting to AgentBus via service discovery"
            );
            return Err(anyhow::anyhow!(
                "Service discovery not yet implemented for gRPC. Please use --host to specify a direct connection."
            ));
        };

        AgentBusClient::connect(&endpoint)
            .await
            .context("Failed to create AgentBus gRPC client")
    }
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Propose a new intention to the AgentBus
    Intention {
        /// The intention string to propose
        #[clap(value_name = "INTENTION")]
        intention: String,
    },
    /// Propose a decider policy change to the AgentBus
    DeciderPolicy {
        /// The decider policy to propose (OFF_BY_DEFAULT, ON_BY_DEFAULT, or FIRST_BOOLEAN_WINS)
        #[clap(value_name = "POLICY")]
        policy: String,
    },
    /// Propose a voter policy change to the AgentBus
    VoterPolicy {
        /// The prompt override text to override the voter's prompt
        #[clap(value_name = "PROMPT_OVERRIDE")]
        prompt_update: String,
    },
    /// Propose a control signal (agentInput) to the AgentBus
    Control {
        /// The agent input text
        #[clap(value_name = "INPUT")]
        input: String,
    },
    /// Poll entries from the AgentBus
    Poll {
        /// Starting log position
        #[clap(long)]
        start: i64,

        /// Maximum number of entries to retrieve
        #[clap(long)]
        limit: i16,
    },
    /// Start interactive REPL mode
    Repl,
}

/// Run the CLI with the given bus, agent_bus_id, and command.
pub async fn run(bus: &impl AgentBus, agent_bus_id: String, command: Commands) -> Result<()> {
    match command {
        Commands::Intention { intention } => {
            propose_intention(bus, &agent_bus_id, &intention).await?;
        }
        Commands::DeciderPolicy { policy } => {
            propose_decider_policy(bus, &agent_bus_id, &policy).await?;
        }
        Commands::VoterPolicy { prompt_update } => {
            propose_voter_policy(bus, &agent_bus_id, &prompt_update).await?;
        }
        Commands::Control { input } => {
            propose_control(bus, &agent_bus_id, &input).await?;
        }
        Commands::Poll { start, limit } => {
            poll_entries(bus, &agent_bus_id, start, limit).await?;
        }
        Commands::Repl => {
            run_repl(bus, agent_bus_id).await?;
        }
    }
    Ok(())
}

/// Propose a new intention to the AgentBus
async fn propose_intention(
    bus: &impl AgentBus,
    agent_bus_id: &str,
    intention_str: &str,
) -> Result<()> {
    info!(intention = %intention_str, "Proposing intention");

    let payload = Payload {
        payload: Some(payload::Payload::Intention(Intention {
            intention: Some(intention::Intention::StringIntention(
                intention_str.to_owned(),
            )),
        })),
    };
    let request = ProposeRequest {
        agent_bus_id: agent_bus_id.to_owned(),
        payload: Some(payload),
        ..Default::default()
    };

    let response = bus
        .propose(request)
        .await
        .context("Failed to propose intention")?;

    println!(
        "✓ Intention proposed successfully at log position: {}",
        response.log_position
    );
    println!("  Intention: \"{}\"", intention_str);
    println!("  Agent Bus ID: {}", agent_bus_id);

    Ok(())
}

/// Propose a decider policy change to the AgentBus
async fn propose_decider_policy(
    bus: &impl AgentBus,
    agent_bus_id: &str,
    policy_str: &str,
) -> Result<()> {
    info!(decider_policy = %policy_str, "Proposing decider policy");

    // Parse the decider policy string
    let decider_policy = match policy_str.to_uppercase().as_str() {
        "OFF_BY_DEFAULT" | "OFF" => DeciderPolicy::OffByDefault as i32,
        "ON_BY_DEFAULT" | "ON" => DeciderPolicy::OnByDefault as i32,
        "FIRST_BOOLEAN_WINS" | "FIRST" => DeciderPolicy::FirstBooleanWins as i32,
        _ => {
            return Err(anyhow::anyhow!(
                "Invalid decider policy: {}. Valid options: OFF_BY_DEFAULT, ON_BY_DEFAULT, FIRST_BOOLEAN_WINS",
                policy_str
            ));
        }
    };

    let payload = Payload {
        payload: Some(payload::Payload::DeciderPolicy(decider_policy)),
    };
    let request = ProposeRequest {
        agent_bus_id: agent_bus_id.to_owned(),
        payload: Some(payload),
        ..Default::default()
    };

    let response = bus
        .propose(request)
        .await
        .context("Failed to propose policy")?;

    println!(
        "✓ Decider policy proposed successfully at log position: {}",
        response.log_position
    );
    println!("  Decider Policy: {}", policy_str);
    println!("  Agent Bus ID: {}", agent_bus_id);

    Ok(())
}

/// Propose a voter policy change to the AgentBus
async fn propose_voter_policy(
    bus: &impl AgentBus,
    agent_bus_id: &str,
    prompt_update: &str,
) -> Result<()> {
    info!(prompt_update = %prompt_update, "Proposing voter policy");

    let voter_policy = VoterPolicy {
        prompt_override: prompt_update.to_owned(),
        ..Default::default()
    };
    let payload = Payload {
        payload: Some(payload::Payload::VoterPolicy(voter_policy)),
    };
    let request = ProposeRequest {
        agent_bus_id: agent_bus_id.to_owned(),
        payload: Some(payload),
        ..Default::default()
    };

    let response = bus.propose(request).await.map_err(|e| {
        eprintln!("Error: Failed to propose voter policy");
        eprintln!("Exception details: {:#?}", e);
        anyhow::anyhow!("Failed to propose voter policy: {}", e)
    })?;

    println!(
        "✓ Voter policy proposed successfully at log position: {}",
        response.log_position
    );
    println!("  Prompt Override: \"{}\"", prompt_update);
    println!("  Agent Bus ID: {}", agent_bus_id);

    Ok(())
}

/// Propose a control signal (agentInput) to the AgentBus
async fn propose_control(bus: &impl AgentBus, agent_bus_id: &str, input: &str) -> Result<()> {
    info!(input = %input, "Proposing control signal");

    let control = Control {
        control: Some(control::Control::AgentInput(input.to_owned())),
    };
    let payload = Payload {
        payload: Some(payload::Payload::Control(control)),
    };
    let request = ProposeRequest {
        agent_bus_id: agent_bus_id.to_owned(),
        payload: Some(payload),
        ..Default::default()
    };

    let response = bus.propose(request).await.map_err(|e| {
        anyhow::anyhow!(
            "Failed to propose control signal with input: '{}' to agent_bus_id: {}. Error: {:?}",
            input,
            agent_bus_id,
            e
        )
    })?;

    println!(
        "✓ Control signal proposed successfully at log position: {}",
        response.log_position
    );
    println!("  Agent Input: \"{}\"", input);
    println!("  Agent Bus ID: {}", agent_bus_id);

    Ok(())
}

/// Print a single bus entry
fn print_entry(entry: &BusEntry) {
    let position = entry.header.as_ref().map(|h| h.log_position).unwrap_or(0);
    print!("[Position {}] ", position);

    if let Some(ref payload) = entry.payload {
        match &payload.payload {
            Some(payload::Payload::Intention(intention)) => {
                println!("{}", "INTENTION:".bright_cyan());
                if let Some(ref int) = intention.intention {
                    match int {
                        intention::Intention::StringIntention(s) => {
                            for line in s.lines() {
                                println!("\t{}", line.white().on_black());
                            }
                        }
                    }
                } else {
                    println!("<unknown>");
                }
            }
            Some(payload::Payload::Vote(vote)) => {
                if let Some(ref vote_type) = vote.abstract_vote {
                    if let Some(vote_type::VoteType::BooleanVote(b)) = vote_type.vote_type {
                        let colored_vote_tag = if b {
                            "VOTE:".bright_green()
                        } else {
                            "VOTE:".red()
                        };
                        print!(
                            "{} intention_id={}, vote=",
                            colored_vote_tag, vote.intention_id
                        );
                        println!("{}", b);
                        // Print the vote reason if available
                        if let Some(ref vote_info) = vote.info {
                            if let Some(vote_info::VoteInfo::ExternalLlmVoteInfo(ref llm_info)) =
                                vote_info.vote_info
                            {
                                let colored_reason = if b {
                                    llm_info.reason.bright_green()
                                } else {
                                    llm_info.reason.red()
                                };
                                println!("\tReason: {}", colored_reason);
                            }
                        }
                    } else {
                        print!(
                            "{} intention_id={}, vote=",
                            "VOTE:".blue(),
                            vote.intention_id
                        );
                        println!("<unknown>");
                    }
                } else {
                    print!(
                        "{} intention_id={}, vote=",
                        "VOTE:".blue(),
                        vote.intention_id
                    );
                    println!("<unknown>");
                }
            }
            Some(payload::Payload::Commit(commit)) => {
                println!(
                    "{} intention_id={}",
                    "COMMIT:".bright_green(),
                    commit.intention_id
                );
                println!("\tReason: {}", commit.reason.bright_green());
            }
            Some(payload::Payload::Abort(abort)) => {
                println!("{} intention_id={}", "ABORT:".red(), abort.intention_id);
                println!("\tReason: {}", abort.reason.red());
            }
            Some(payload::Payload::DeciderPolicy(decider_policy)) => {
                let policy_name = match DeciderPolicy::try_from(*decider_policy) {
                    Ok(DeciderPolicy::OffByDefault) => "OffByDefault",
                    Ok(DeciderPolicy::OnByDefault) => "OnByDefault",
                    Ok(DeciderPolicy::FirstBooleanWins) => "FirstBooleanWins",
                    _ => "Unknown",
                };
                println!("{} {}", "DECIDER_POLICY:".yellow(), policy_name);
            }
            Some(payload::Payload::VoterPolicy(voter_policy)) => {
                println!(
                    "{} prompt_override={:?}",
                    "VOTER_POLICY:".magenta(),
                    voter_policy.prompt_override
                );
            }
            Some(payload::Payload::Control(control)) => {
                println!("{}", "CONTROL:".bright_magenta());
                if let Some(ref ctrl) = control.control {
                    match ctrl {
                        control::Control::AgentInput(input) => {
                            println!("\tAgent Input: \"{}\"", input.bright_magenta());
                        }
                    }
                } else {
                    println!("\t<unknown>");
                }
            }
            Some(payload::Payload::InferenceInput(inference_input)) => {
                println!("{}", "INFERENCE_INPUT:".bright_blue());
                if let Some(ref input) = inference_input.inference_input {
                    match input {
                        inference_input::InferenceInput::StringInferenceInput(s) => {
                            for line in s.lines() {
                                println!("\t{}", line.white().on_black());
                            }
                        }
                    }
                } else {
                    println!("\t<unknown>");
                }
            }
            Some(payload::Payload::InferenceOutput(inference_output)) => {
                println!("{}", "INFERENCE_OUTPUT:".bright_blue());
                if let Some(ref output) = inference_output.inference_output {
                    match output {
                        inference_output::InferenceOutput::StringInferenceOutput(s) => {
                            for line in s.lines() {
                                println!("\t{}", line.white().on_black());
                            }
                        }
                    }
                } else {
                    println!("\t<unknown>");
                }
            }
            Some(payload::Payload::ActionOutput(action_output)) => {
                println!(
                    "{} intention_id={}",
                    "ACTION_OUTPUT:".bright_yellow(),
                    action_output.intention_id
                );
                if let Some(ref output) = action_output.action_output {
                    match output {
                        action_output::ActionOutput::StringActionOutput(s) => {
                            for line in s.lines() {
                                println!("\t{}", line.white().on_black());
                            }
                        }
                    }
                } else {
                    println!("\t<unknown>");
                }
            }
            Some(payload::Payload::AgentInput(agent_input)) => {
                println!("{}", "AGENT_INPUT:".bright_green());
                if let Some(ref input) = agent_input.agent_input {
                    match input {
                        agent_input::AgentInput::StringAgentInput(s) => {
                            for line in s.lines() {
                                println!("\t{}", line.white().on_black());
                            }
                        }
                    }
                } else {
                    println!("\t<unknown>");
                }
            }
            Some(payload::Payload::AgentOutput(agent_output)) => {
                println!("{}", "AGENT_OUTPUT:".bright_green());
                if let Some(ref output) = agent_output.agent_output {
                    match output {
                        agent_output::AgentOutput::StringAgentOutput(s) => {
                            for line in s.lines() {
                                println!("\t{}", line.white().on_black());
                            }
                        }
                    }
                } else {
                    println!("\t<unknown>");
                }
            }
            None => {
                println!("{}", "UNKNOWN".bright_black());
            }
        }
    } else {
        println!("{}", "UNKNOWN".bright_black());
    }
}

/// Poll entries from the AgentBus
async fn poll_entries(
    bus: &impl AgentBus,
    agent_bus_id: &str,
    start_position: i64,
    max_entries: i16,
) -> Result<()> {
    info!(
        start_position = start_position,
        max_entries = max_entries,
        "Polling entries"
    );

    let request = PollRequest {
        agent_bus_id: agent_bus_id.to_owned(),
        start_log_position: start_position,
        max_entries: max_entries as i32,
        filter: None,
        ..Default::default()
    };

    let response = bus.poll(request).await.context("Failed to poll entries")?;

    if response.entries.is_empty() {
        println!("No entries found");
        println!("  Agent Bus ID: {}", agent_bus_id);
        println!("  Start Position: {}", start_position);
        return Ok(());
    }

    println!(
        "✓ Retrieved {} entries (complete: {})",
        response.entries.len(),
        response.complete
    );
    println!("  Agent Bus ID: {}", agent_bus_id);
    println!();

    for entry in &response.entries {
        print_entry(entry);
    }

    Ok(())
}

/// Autocomplete helper for the REPL
#[derive(Clone)]
struct AgentBusCompleter;

const COMMANDS: &[&str] = &[
    "intention",
    "decider-policy",
    "voter-policy",
    "control",
    "poll",
    "tail",
    "set-id",
    "help",
    "quit",
    "exit",
];
const DECIDER_POLICIES: &[&str] = &["OFF_BY_DEFAULT", "ON_BY_DEFAULT", "FIRST_BOOLEAN_WINS"];

impl rustyline::completion::Completer for AgentBusCompleter {
    type Candidate = String;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Self::Candidate>)> {
        let line = &line[..pos];
        let parts: Vec<&str> = line.split_whitespace().collect();

        // Completing the first word (command)
        if parts.is_empty() || (parts.len() == 1 && !line.ends_with(' ')) {
            let prefix = parts.first().unwrap_or(&"");
            let matches: Vec<String> = COMMANDS
                .iter()
                .filter(|cmd| cmd.starts_with(prefix))
                .map(|s| s.to_string())
                .collect();
            return Ok((line.len() - prefix.len(), matches));
        }

        // Completing decider-policy argument
        if parts[0] == "decider-policy" && parts.len() <= 2 {
            let prefix = parts.get(1).unwrap_or(&"").to_uppercase();
            let matches: Vec<String> = DECIDER_POLICIES
                .iter()
                .filter(|p| p.starts_with(&prefix))
                .map(|s| s.to_string())
                .collect();
            let last_word_start = line.rfind(' ').map(|i| i + 1).unwrap_or(0);
            return Ok((last_word_start, matches));
        }

        Ok((pos, vec![]))
    }
}

impl rustyline::hint::Hinter for AgentBusCompleter {
    type Hint = String;
}

impl rustyline::highlight::Highlighter for AgentBusCompleter {}

impl rustyline::validate::Validator for AgentBusCompleter {}

impl rustyline::Helper for AgentBusCompleter {}

/// Find the end position of the log
async fn find_end_position(bus: &impl AgentBus, agent_bus_id: &str) -> Result<i64> {
    let mut pos = 0i64;
    let step = 100i64;

    loop {
        let request = PollRequest {
            agent_bus_id: agent_bus_id.to_owned(),
            start_log_position: pos,
            max_entries: step as i32,
            filter: None,
            ..Default::default()
        };

        let response = bus.poll(request).await?;

        if response.entries.is_empty() {
            break;
        }

        if response.complete {
            pos = response
                .entries
                .last()
                .unwrap()
                .header
                .as_ref()
                .unwrap()
                .log_position
                + 1;
            break;
        }

        pos += response.entries.len() as i64;
    }

    Ok(pos)
}

/// Find the tail of the log and display the last N entries, returning the end position
async fn find_tail(bus: &impl AgentBus, agent_bus_id: &str, tail_n: i16) -> Result<i64> {
    let pos = find_end_position(bus, agent_bus_id).await?;
    let start = (pos - tail_n as i64).max(0);
    poll_entries(bus, agent_bus_id, start, tail_n).await?;
    Ok(pos)
}

/// Tail with follow mode - continuously show new entries
async fn tail_follow(bus: &impl AgentBus, agent_bus_id: &str, tail_n: i16) -> Result<()> {
    use tokio::io::AsyncReadExt;
    use tokio::io::stdin;
    use tokio::time::Duration;
    use tokio::time::sleep;

    // First, show the last N entries and get the end position
    let mut pos = find_tail(bus, agent_bus_id, tail_n).await?;

    println!("\n[Following... Press any key to exit]");

    let mut stdin = stdin();
    let mut buf = [0u8; 1];

    loop {
        tokio::select! {
            _ = stdin.read(&mut buf) => {
                println!("\n[Exiting follow mode]");
                break;
            }
            _ = sleep(Duration::from_millis(500)) => {
                let request = PollRequest {
                    agent_bus_id: agent_bus_id.to_owned(),
                    start_log_position: pos,
                    max_entries: 100,
                    filter: None,
                    ..Default::default()
                };

                match bus.poll(request).await {
                    Ok(response) => {
                        for entry in &response.entries {
                            print_entry(entry);
                            pos = entry.header.as_ref().unwrap().log_position + 1;
                        }
                    }
                    Err(e) => {
                        println!("Error polling: {}", e);
                        break;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Run interactive REPL mode
async fn run_repl(bus: &impl AgentBus, initial_agent_bus_id: String) -> Result<()> {
    use rustyline::Editor;
    use rustyline::error::ReadlineError;

    let mut agent_bus_id = initial_agent_bus_id;

    println!("AgentBus REPL - Interactive Mode");
    println!("Agent Bus ID: {}", agent_bus_id);
    println!();
    println!("Commands:");
    println!("  intention <text>           - Propose a new intention");
    println!(
        "  decider-policy <policy>    - Propose a decider policy (OFF_BY_DEFAULT, ON_BY_DEFAULT, FIRST_BOOLEAN_WINS)"
    );
    println!("  voter-policy <text>        - Propose a voter policy prompt override");
    println!("  control <text>             - Propose a control signal (agentInput)");
    println!("  poll <start> <limit>       - Poll entries (both parameters required)");
    println!("  tail [-f] [n]              - Show last n entries (default: 10), -f to follow");
    println!("  set-id <id>                - Change the current agent bus ID");
    println!("  help                       - Show this help");
    println!("  quit or exit               - Exit REPL");
    println!();
    println!("Press Tab for autocomplete");
    println!();

    let config = rustyline::Config::builder()
        .completion_type(rustyline::CompletionType::Circular)
        .build();
    let mut rl = Editor::with_config(config)?;
    rl.set_helper(Some(AgentBusCompleter));

    loop {
        let readline = rl.readline("agentbus> ");

        let line = match readline {
            Ok(line) => {
                rl.add_history_entry(line.as_str())?;
                line
            }
            Err(ReadlineError::Interrupted) => {
                // Ctrl+C
                println!("^C");
                continue;
            }
            Err(ReadlineError::Eof) => {
                // Ctrl+D
                println!("Goodbye!");
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        };

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.splitn(2, ' ').collect();
        let command = parts[0];

        match command {
            "quit" | "exit" => {
                println!("Goodbye!");
                break;
            }
            "help" => {
                println!("Commands:");
                println!("  intention <text>           - Propose a new intention");
                println!(
                    "  decider-policy <policy>    - Propose a decider policy (OFF_BY_DEFAULT, ON_BY_DEFAULT, FIRST_BOOLEAN_WINS)"
                );
                println!("  voter-policy <text>        - Propose a voter policy prompt override");
                println!("  control <text>             - Propose a control signal (agentInput)");
                println!("  poll <start> <limit>       - Poll entries (both parameters required)");
                println!(
                    "  tail [-f] [n]              - Show last n entries (default: 10), -f to follow"
                );
                println!("  set-id <id>                - Change the current agent bus ID");
                println!("  help                       - Show this help");
                println!("  quit or exit               - Exit REPL");
            }
            "set-id" => {
                if parts.len() < 2 {
                    println!("Error: set-id requires an ID");
                    println!("Usage: set-id <id>");
                    continue;
                }
                let new_id = parts[1].to_string();
                agent_bus_id = new_id;
                println!("✓ Current agent bus ID set to: {}", agent_bus_id);
            }
            "intention" => {
                if parts.len() < 2 {
                    println!("Error: intention requires a text string");
                    println!("Usage: intention <text>");
                    continue;
                }
                let intention = parts[1].to_string();

                match propose_intention(bus, &agent_bus_id, &intention).await {
                    Ok(()) => {}
                    Err(e) => {
                        println!("Error: {}", e);
                    }
                }
            }
            "decider-policy" => {
                if parts.len() < 2 {
                    println!("Error: decider-policy requires a policy type");
                    println!("Usage: decider-policy <policy>");
                    println!("Valid policies: OFF_BY_DEFAULT, ON_BY_DEFAULT, FIRST_BOOLEAN_WINS");
                    continue;
                }
                let policy = parts[1].to_string();

                match propose_decider_policy(bus, &agent_bus_id, &policy).await {
                    Ok(()) => {}
                    Err(e) => {
                        println!("Error: {}", e);
                    }
                }
            }
            "voter-policy" => {
                if parts.len() < 2 {
                    println!("Error: voter-policy requires a prompt override text");
                    println!("Usage: voter-policy <text>");
                    continue;
                }
                let prompt_update = parts[1].to_string();

                match propose_voter_policy(bus, &agent_bus_id, &prompt_update).await {
                    Ok(()) => {}
                    Err(e) => {
                        println!("Error: {}", e);
                    }
                }
            }
            "control" => {
                if parts.len() < 2 {
                    println!("Error: control requires input text");
                    println!("Usage: control <text>");
                    continue;
                }
                let input = parts[1].to_string();

                match propose_control(bus, &agent_bus_id, &input).await {
                    Ok(()) => {}
                    Err(e) => {
                        println!("Error: {}", e);
                    }
                }
            }
            "poll" => {
                if parts.len() < 2 {
                    println!("Error: poll requires both start and limit parameters");
                    println!("Usage: poll <start> <limit>");
                    continue;
                }

                let poll_args: Vec<&str> = parts[1].split_whitespace().collect();

                if poll_args.len() < 2 {
                    println!("Error: poll requires both start and limit parameters");
                    println!("Usage: poll <start> <limit>");
                    continue;
                }

                let start = match poll_args[0].parse::<i64>() {
                    Ok(s) => s,
                    Err(_) => {
                        println!("Error: invalid start position '{}'", poll_args[0]);
                        continue;
                    }
                };

                let limit = match poll_args[1].parse::<i16>() {
                    Ok(l) => l,
                    Err(_) => {
                        println!("Error: invalid limit '{}'", poll_args[1]);
                        continue;
                    }
                };

                match poll_entries(bus, &agent_bus_id, start, limit).await {
                    Ok(()) => {}
                    Err(e) => {
                        println!("Error: {}", e);
                    }
                }
            }
            "tail" => {
                let args = if parts.len() >= 2 {
                    parts[1].split_whitespace().collect::<Vec<_>>()
                } else {
                    vec![]
                };

                let mut tail_n = 10i16;
                let mut follow = false;

                for arg in args {
                    if arg == "-f" {
                        follow = true;
                    } else if let Ok(n) = arg.parse::<i16>() {
                        tail_n = n;
                    } else {
                        println!("Error: invalid argument '{}'", arg);
                        println!("Usage: tail [-f] [n]");
                        continue;
                    }
                }

                if follow {
                    match tail_follow(bus, &agent_bus_id, tail_n).await {
                        Ok(()) => {}
                        Err(e) => {
                            println!("Error: {}", e);
                        }
                    }
                } else {
                    match find_tail(bus, &agent_bus_id, tail_n).await {
                        Ok(_) => {}
                        Err(e) => {
                            println!("Error: {}", e);
                        }
                    }
                }
            }
            _ => {
                println!("Unknown command: {}", command);
                println!("Type 'help' for available commands");
            }
        }
    }

    Ok(())
}

/// Initialize logging with glog format
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
