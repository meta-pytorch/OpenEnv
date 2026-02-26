// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Counter implementation backed by AgentBus

use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;
use std::time::Duration;

use agent_bus_proto_rust::agent_bus::BusEntry;
use agent_bus_proto_rust::agent_bus::Intention;
use agent_bus_proto_rust::agent_bus::Payload;
use agent_bus_proto_rust::agent_bus::PollRequest;
use agent_bus_proto_rust::agent_bus::ProposeRequest;
use agentbus_api::AgentBus;
use agentbus_api::environment::Environment;
use rand::Rng;

use super::counter_trait::CommandResult;
use super::counter_trait::Counter;
use super::linearizability_tracker::ExecutedCommand;

enum CounterOp {
    Increment,
    Decrement,
    Noop,
}

impl fmt::Display for CounterOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CounterOp::Increment => write!(f, "inc"),
            CounterOp::Decrement => write!(f, "dec"),
            CounterOp::Noop => write!(f, "noop"),
        }
    }
}

impl CounterOp {
    fn from_string(s: &str) -> Option<Self> {
        match s {
            "inc" => Some(CounterOp::Increment),
            "dec" => Some(CounterOp::Decrement),
            "noop" => Some(CounterOp::Noop),
            _ => None,
        }
    }

    fn needs_commit(&self) -> bool {
        !matches!(self, CounterOp::Noop)
    }
}

pub struct AgentBusCounter<T: AgentBus, E: Environment> {
    agent_bus_impl: T,
    agent_bus_id: String,
    env: Rc<E>,
    state: RefCell<CounterState>,
    skip_commits: bool,
    max_poll_entries: i32,
}

/// A simple sequential counter for use as a linearizability specification.
pub struct SequentialCounter {
    value: RefCell<i64>,
    next_intention_id: RefCell<i64>,
    command_history: RefCell<Vec<ExecutedCommand>>,
}

impl SequentialCounter {
    pub fn new() -> Self {
        Self {
            value: RefCell::new(0),
            next_intention_id: RefCell::new(0),
            command_history: RefCell::new(Vec::new()),
        }
    }

    pub fn get_value(&self) -> i64 {
        *self.value.borrow()
    }

    pub fn apply_operation(&self, intention_id: i64, operation: &str) -> i64 {
        let mut value = self.value.borrow_mut();
        match operation {
            "inc" => *value += 1,
            "dec" => *value -= 1,
            "noop" => {}
            _ => {}
        }
        self.command_history.borrow_mut().push(ExecutedCommand {
            intention_id,
            operation: operation.to_string(),
        });
        *value
    }

    fn execute(&self, operation: &str) -> CommandResult {
        let mut intention_id = self.next_intention_id.borrow_mut();
        let id = *intention_id;
        *intention_id += 1;
        drop(intention_id);
        let value = self.apply_operation(id, operation);
        CommandResult {
            intention_id: id,
            value,
        }
    }
}

impl Counter for SequentialCounter {
    async fn increment(&self) -> Result<CommandResult, String> {
        Ok(self.execute("inc"))
    }

    async fn decrement(&self) -> Result<CommandResult, String> {
        Ok(self.execute("dec"))
    }

    async fn get_command_history(&self) -> Vec<ExecutedCommand> {
        self.command_history.borrow().clone()
    }
}

struct CounterState {
    counter: SequentialCounter,
    next_log_position: i64,
    buffered_intentions: std::collections::HashMap<i64, CounterOp>,
}

/// Result of processing a single log entry
enum EntryResult {
    Continue,
    FoundMyOp(i64),
    Aborted,
}

impl<T: AgentBus, E: Environment> AgentBusCounter<T, E> {
    pub fn new(agent_bus_impl: T, agent_bus_id: String, env: Rc<E>, _worker_id: usize) -> Self {
        Self {
            agent_bus_impl,
            agent_bus_id,
            env,
            state: RefCell::new(CounterState {
                counter: SequentialCounter::new(),
                next_log_position: 0,
                buffered_intentions: std::collections::HashMap::new(),
            }),
            skip_commits: false,
            max_poll_entries: 1000,
        }
    }

    pub fn with_skip_commits(mut self) -> Self {
        self.skip_commits = true;
        self
    }

    pub fn with_max_poll_entries(mut self, max_entries: i32) -> Self {
        self.max_poll_entries = max_entries;
        self
    }

    async fn execute_op(&self, op: CounterOp) -> Result<CommandResult, String> {
        let needs_commit = op.needs_commit();

        let my_intention_id = self.propose_intention(&op).await;
        let proposed_log_position = my_intention_id;

        loop {
            // For noop, return once we've seen any new entries
            if !needs_commit && self.state.borrow().next_log_position > proposed_log_position {
                return Ok(CommandResult {
                    intention_id: my_intention_id,
                    value: self.state.borrow().counter.get_value(),
                });
            }

            match self.poll_and_process(my_intention_id).await {
                EntryResult::FoundMyOp(value) => {
                    return Ok(CommandResult {
                        intention_id: my_intention_id,
                        value,
                    });
                }
                EntryResult::Aborted => {
                    return Err(format!("Intention {} was aborted", my_intention_id));
                }
                EntryResult::Continue => {
                    let sleep_millis = self.env.with_rng(|rng| rng.gen_range(1..5));
                    self.env.sleep(Duration::from_millis(sleep_millis)).await;
                }
            }
        }
    }

    async fn propose_intention(&self, op: &CounterOp) -> i64 {
        let intention_payload = Payload {
            payload: Some(
                agent_bus_proto_rust::agent_bus::payload::Payload::Intention(Intention {
                    intention: Some(
                        agent_bus_proto_rust::agent_bus::intention::Intention::StringIntention(
                            op.to_string(),
                        ),
                    ),
                }),
            ),
        };
        let response = self
            .agent_bus_impl
            .propose(ProposeRequest {
                agent_bus_id: self.agent_bus_id.clone(),
                payload: Some(intention_payload),
                ..Default::default()
            })
            .await
            .expect("Propose should succeed");
        response.log_position
    }

    async fn poll_and_process(&self, my_intention_id: i64) -> EntryResult {
        let start_log_position = self.state.borrow().next_log_position;
        let poll_result = self
            .agent_bus_impl
            .poll(PollRequest {
                agent_bus_id: self.agent_bus_id.clone(),
                start_log_position,
                max_entries: self.max_poll_entries,
                filter: None,
                ..Default::default()
            })
            .await
            .expect("Poll should succeed");

        let mut result = EntryResult::Continue;
        for entry in poll_result.entries {
            let log_position = entry.header.as_ref().unwrap().log_position;
            self.state.borrow_mut().next_log_position = log_position + 1;

            let entry_result = if self.skip_commits {
                self.process_entry_skip_commits(&entry, log_position, my_intention_id)
            } else {
                self.process_entry_with_commits(&entry, log_position, my_intention_id)
            };

            match entry_result {
                EntryResult::Continue => {}
                other => result = other,
            }
        }
        result
    }

    fn process_entry_skip_commits(
        &self,
        entry: &BusEntry,
        log_position: i64,
        my_intention_id: i64,
    ) -> EntryResult {
        if let Some(ref payload) = entry.payload {
            if let Some(agent_bus_proto_rust::agent_bus::payload::Payload::Intention(intention)) =
                &payload.payload
            {
                if let Some(
                    agent_bus_proto_rust::agent_bus::intention::Intention::StringIntention(s),
                ) = &intention.intention
                {
                    if let Some(parsed_op) = CounterOp::from_string(s) {
                        if parsed_op.needs_commit() {
                            let value =
                                self.state.borrow().counter.apply_operation(log_position, s);
                            if log_position == my_intention_id {
                                return EntryResult::FoundMyOp(value);
                            }
                        }
                    }
                }
            }
        }
        EntryResult::Continue
    }

    fn process_entry_with_commits(
        &self,
        entry: &BusEntry,
        log_position: i64,
        my_intention_id: i64,
    ) -> EntryResult {
        let Some(ref payload) = entry.payload else {
            return EntryResult::Continue;
        };

        match &payload.payload {
            Some(agent_bus_proto_rust::agent_bus::payload::Payload::Intention(intention)) => {
                if let Some(
                    agent_bus_proto_rust::agent_bus::intention::Intention::StringIntention(s),
                ) = &intention.intention
                {
                    if let Some(parsed_op) = CounterOp::from_string(s) {
                        if parsed_op.needs_commit() {
                            self.state
                                .borrow_mut()
                                .buffered_intentions
                                .insert(log_position, parsed_op);
                        }
                    }
                }
            }
            Some(agent_bus_proto_rust::agent_bus::payload::Payload::Commit(commit)) => {
                let mut state = self.state.borrow_mut();
                if let Some(op) = state.buffered_intentions.remove(&commit.intention_id) {
                    let value = state
                        .counter
                        .apply_operation(commit.intention_id, &op.to_string());
                    if commit.intention_id == my_intention_id {
                        return EntryResult::FoundMyOp(value);
                    }
                }
            }
            Some(agent_bus_proto_rust::agent_bus::payload::Payload::Abort(abort)) => {
                self.state
                    .borrow_mut()
                    .buffered_intentions
                    .remove(&abort.intention_id);
                if abort.intention_id == my_intention_id {
                    return EntryResult::Aborted;
                }
            }
            _ => {}
        }
        EntryResult::Continue
    }
}

impl<T: AgentBus, E: Environment> Counter for AgentBusCounter<T, E> {
    async fn increment(&self) -> Result<CommandResult, String> {
        self.execute_op(CounterOp::Increment).await
    }

    async fn decrement(&self) -> Result<CommandResult, String> {
        self.execute_op(CounterOp::Decrement).await
    }

    async fn get_command_history(&self) -> Vec<ExecutedCommand> {
        let _ = self.execute_op(CounterOp::Noop).await;
        self.state.borrow().counter.command_history.borrow().clone()
    }
}
