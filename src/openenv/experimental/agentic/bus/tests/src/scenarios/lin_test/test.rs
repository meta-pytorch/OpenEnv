// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Multi-worker counter linearizability test

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use agent_bus_proto_rust::agent_bus::DeciderPolicy;
use agent_bus_proto_rust::agent_bus::ProposeRequest;
use agentbus_api::AgentBus;
use agentbus_api::environment::Environment;
use agentbus_core::decider::Decider;
use agentbus_core::vote_trackers::create_vote_tracker_for_decider_policy;
use anyhow::Result;
use rand::Rng;
use tokio::sync::Barrier;

use super::counter_impl::AgentBusCounter;
use super::counter_impl::SequentialCounter;
use super::counter_trait::Counter;
use super::counter_worker::CounterWorker;
use super::linearizability_tracker::ExecutedCommand;
use super::linearizability_tracker::LinearizabilityTracker;
use super::random_voter::RandomVoter;
use super::tracking_counter::TrackingCounter;
use crate::fixtures::AgentBusTestFixture;
use crate::simulator::Simulator;

/// Test with skip_commits mode - no decider/voter needed
pub async fn run_test_multi_worker_counter_skip_commits<F>(fixture: &F) -> Result<()>
where
    F: AgentBusTestFixture<Env = Simulator>,
{
    run_multi_worker_counter_test(fixture, true, None).await
}

/// Test with skip_commits mode and a fixed seed (for deterministic buggy tests)
pub async fn run_test_multi_worker_counter_skip_commits_with_seed<F>(fixture: &F) -> Result<()>
where
    F: AgentBusTestFixture<Env = Simulator>,
{
    run_multi_worker_counter_test(fixture, true, None).await
}

/// Default test uses random poll size (1-64)
pub async fn run_test_multi_worker_counter<F>(fixture: &F) -> Result<()>
where
    F: AgentBusTestFixture<Env = Simulator>,
{
    run_multi_worker_counter_test(fixture, false, None).await
}

/// Test with large poll batch size (1000)
pub async fn run_test_multi_worker_counter_large_poll<F>(fixture: &F) -> Result<()>
where
    F: AgentBusTestFixture<Env = Simulator>,
{
    run_multi_worker_counter_test(fixture, false, Some(1000)).await
}

/// Test with small poll batch size (1) - catches noop position bugs
pub async fn run_test_multi_worker_counter_small_poll<F>(fixture: &F) -> Result<()>
where
    F: AgentBusTestFixture<Env = Simulator>,
{
    run_multi_worker_counter_test(fixture, false, Some(1)).await
}

async fn run_multi_worker_counter_test<F>(
    fixture: &F,
    skip_commits: bool,
    max_poll_entries: Option<i32>,
) -> Result<()>
where
    F: AgentBusTestFixture<Env = Simulator>,
{
    let env_rc = fixture.get_env();

    let agent_bus_id = format!("bus-{}", env_rc.with_rng(|rng| rng.r#gen::<u64>()));
    let num_workers: usize = env_rc.with_rng(|rng| rng.gen_range(2..5));
    let num_ops_per_worker: usize = env_rc.with_rng(|rng| rng.gen_range(2..6));

    let barrier = Arc::new(Barrier::new(num_workers));

    let tracker: Rc<LinearizabilityTracker<i64>> = LinearizabilityTracker::new();

    let workers: Vec<_> = (0..num_workers)
        .map(|idx| {
            let impl_instance = fixture.create_impl();
            let mut counter =
                AgentBusCounter::new(impl_instance, agent_bus_id.clone(), env_rc.clone(), idx);
            if skip_commits {
                counter = counter.with_skip_commits();
            }
            let actual_max = match max_poll_entries {
                Some(max) => max,
                None => env_rc.with_rng(|rng| rng.gen_range(1..=64)),
            };
            counter = counter.with_max_poll_entries(actual_max);
            let tracking_counter = TrackingCounter::new(counter, env_rc.clone(), tracker.clone());
            CounterWorker::new(tracking_counter, env_rc.clone(), num_ops_per_worker)
        })
        .collect();

    let stop_components = Rc::new(Cell::new(false));

    if !skip_commits {
        spawn_decider_and_voters(fixture, &env_rc, &agent_bus_id, stop_components.clone());
    }

    let handles = spawn_workers(
        workers,
        &env_rc,
        barrier,
        num_workers,
        if skip_commits {
            None
        } else {
            Some(stop_components)
        },
    );

    env_rc.run();

    let mut worker_histories: Vec<Vec<ExecutedCommand>> = Vec::new();
    for handle in handles {
        let command_history: Vec<ExecutedCommand> = handle.await.expect("Worker should complete");
        worker_histories.push(command_history);
    }

    let sequential_counter = SequentialCounter::new();
    tracker.verify(&worker_histories, |cmd| {
        sequential_counter.apply_operation(cmd.intention_id, &cmd.operation)
    })
}

fn spawn_decider_and_voters<F: AgentBusTestFixture<Env = Simulator>>(
    fixture: &F,
    env_rc: &Rc<Simulator>,
    agent_bus_id: &str,
    stop_components: Rc<Cell<bool>>,
) {
    let policy = DeciderPolicy::FirstBooleanWins as i32;

    let policy_impl = fixture.create_impl();
    let agent_bus_id_for_policy = agent_bus_id.to_string();
    env_rc.spawn_named(
        async move {
            let policy_payload = agent_bus_proto_rust::agent_bus::Payload {
                payload: Some(
                    agent_bus_proto_rust::agent_bus::payload::Payload::DeciderPolicy(policy),
                ),
            };
            policy_impl
                .propose(ProposeRequest {
                    agent_bus_id: agent_bus_id_for_policy,
                    payload: Some(policy_payload),
                })
                .await
                .expect("Policy change should succeed");
        },
        Some("policy_setter".to_string()),
    );

    let decider_impl = fixture.create_impl();
    let decider_env = env_rc.clone();
    let agent_bus_id_for_decider = agent_bus_id.to_string();
    let stop_for_decider = stop_components.clone();
    env_rc.spawn_named(
        async move {
            let mut decider = Decider::new(
                decider_impl,
                agent_bus_id_for_decider,
                policy,
                create_vote_tracker_for_decider_policy,
                decider_env.clone(),
            );
            loop {
                if stop_for_decider.get() {
                    break;
                }
                decider
                    .poll_and_decide()
                    .await
                    .expect("Decider should succeed");
                let sleep_millis = decider_env.with_rng(|rng| rng.gen_range(1..5));
                decider_env
                    .sleep(std::time::Duration::from_millis(sleep_millis))
                    .await;
            }
        },
        Some("decider".to_string()),
    );

    let voter_impl = fixture.create_impl();
    let voter = RandomVoter::new(voter_impl, agent_bus_id.to_string(), env_rc.clone());
    env_rc.spawn_named(
        async move {
            loop {
                if stop_components.get() {
                    break;
                }
                voter.poll_and_vote_once().await;
                let sleep_millis = voter.env.with_rng(|rng| rng.gen_range(1..5));
                voter
                    .env
                    .sleep(std::time::Duration::from_millis(sleep_millis))
                    .await;
            }
        },
        Some("voter_0".to_string()),
    );
}

fn spawn_workers<C: Counter + 'static>(
    workers: Vec<CounterWorker<C>>,
    env_rc: &Rc<Simulator>,
    barrier: Arc<Barrier>,
    num_workers: usize,
    stop_components: Option<Rc<Cell<bool>>>,
) -> Vec<crate::simulator::SimulatorHandle<Vec<ExecutedCommand>>> {
    let mut handles = Vec::new();
    for (idx, worker) in workers.into_iter().enumerate() {
        let barrier_clone = barrier.clone();
        let stop_clone = stop_components.clone();
        let is_last_worker = idx == num_workers - 1;
        let handle = env_rc.spawn_named(
            async move {
                worker.run_workload().await;
                barrier_clone.wait().await;
                let command_history = worker.get_command_history().await;
                if is_last_worker {
                    if let Some(stop) = stop_clone {
                        stop.set(true);
                    }
                }
                command_history
            },
            Some(format!("worker_{}", idx)),
        );
        handles.push(handle);
    }
    handles
}
