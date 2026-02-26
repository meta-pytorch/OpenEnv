// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Generic test scenarios that can run against any AgentBus implementation

use std::sync::Arc;

use agent_bus_proto_rust::agent_bus::PollRequest;
use agent_bus_proto_rust::agent_bus::ProposeRequest;
use agentbus_api::AgentBus;
use agentbus_api::environment::Environment;
use anyhow::Result;
use rand::Rng;
use tokio::sync::Barrier;

use crate::fixtures::AgentBusTestFixture;
use crate::fixtures::SimulatorFixture;
use crate::simulator::Simulator;

// Test an AgentBus service from multiple nodes (/clients)
// in a deterministic simulation test.
// The test is generic and can be run against any AgentBus implementation.

struct WorkloadClient<T: AgentBus> {
    agent_bus_impl: T,
    agent_bus_id: String,
}

impl<T: AgentBus> WorkloadClient<T> {
    fn new(agent_bus_impl: T, agent_bus_id: String) -> Self {
        Self {
            agent_bus_impl,
            agent_bus_id,
        }
    }

    async fn propose(&self, payload_str: String) {
        let payload = agent_bus_proto_rust::agent_bus::Payload {
            payload: Some(
                agent_bus_proto_rust::agent_bus::payload::Payload::Intention(
                    agent_bus_proto_rust::agent_bus::Intention {
                        intention: Some(
                            agent_bus_proto_rust::agent_bus::intention::Intention::StringIntention(
                                payload_str,
                            ),
                        ),
                    },
                ),
            ),
        };
        let request = ProposeRequest {
            agent_bus_id: self.agent_bus_id.clone(),
            payload: Some(payload),
            ..Default::default()
        };
        self.agent_bus_impl
            .propose(request)
            .await
            .expect("Propose should succeed");
    }

    async fn run_workload(&self, prefix: &str, count: usize) {
        for i in 1..=count {
            self.propose(format!("{}{}", prefix, i)).await;
        }
    }

    async fn get_concatenated_commands(&self) -> String {
        let poll_request = PollRequest {
            agent_bus_id: self.agent_bus_id.clone(),
            start_log_position: 0,
            max_entries: 1000, // Request up to 1000 entries
            ..Default::default()
        };

        let poll_result = self
            .agent_bus_impl
            .poll(poll_request)
            .await
            .expect("Poll should succeed");

        poll_result
            .entries
            .iter()
            .filter_map(|entry| {
                // Extract the stringIntention from Intention as the command string
                // Skip commit/abort entries
                if let Some(ref payload) = entry.payload {
                    if let Some(agent_bus_proto_rust::agent_bus::payload::Payload::Intention(
                        ref intention,
                    )) = payload.payload
                    {
                        if let Some(
                            agent_bus_proto_rust::agent_bus::intention::Intention::StringIntention(
                                ref string_data,
                            ),
                        ) = intention.intention
                        {
                            return Some(string_data.clone());
                        }
                    }
                }
                None
            })
            .collect::<Vec<_>>()
            .join("")
    }
}

/// Helper function to run the multi-node scenario with a given fixture
fn run_scenario<F>(fixture: F) -> String
where
    F: AgentBusTestFixture<Env = Simulator>,
{
    use futures::executor::block_on;
    let env_rc = fixture.get_env();

    let agent_bus_id = format!("bus-{}", env_rc.with_rng(|rng| rng.r#gen::<u64>()));
    let num_threads: usize = env_rc.with_rng(|rng| rng.gen_range(1..5));
    let num_proposes_per_thread: usize = env_rc.with_rng(|rng| rng.gen_range(1..5));

    // Create barrier for synchronization between threads
    let barrier = Arc::new(Barrier::new(num_threads));

    // Create multiple clients upfront - each thread gets its own client and implementation
    let clients: Vec<_> = (0..num_threads)
        .map(|_| {
            let impl_instance = fixture.create_impl();
            WorkloadClient::new(impl_instance, agent_bus_id.clone())
        })
        .collect();

    let prefixes = ["A", "B", "C", "D", "E", "F"];

    // Spawn K threads; each one has its own client and runs workload
    let mut handles = Vec::new();
    for (i, (client, prefix)) in clients.into_iter().zip(prefixes.iter()).enumerate() {
        let barrier_clone = barrier.clone();
        let prefix = *prefix;
        let is_first = i == 0;
        let agent_bus_id_clone = agent_bus_id.clone();
        let handle = env_rc.spawn(async move {
            // First task sets the policy
            if is_first {
                let policy_payload = agent_bus_proto_rust::agent_bus::Payload {
                    payload: Some(
                        agent_bus_proto_rust::agent_bus::payload::Payload::DeciderPolicy(
                            agent_bus_proto_rust::agent_bus::DeciderPolicy::FirstBooleanWins as i32,
                        ),
                    ),
                };
                let policy_request = ProposeRequest {
                    agent_bus_id: agent_bus_id_clone.clone(),
                    payload: Some(policy_payload),
                };
                client
                    .agent_bus_impl
                    .propose(policy_request)
                    .await
                    .expect("Policy change should succeed");
            }

            // Run workload: issue N proposes in a loop
            client.run_workload(prefix, num_proposes_per_thread).await;

            // Wait on barrier until all threads reach this point
            barrier_clone.wait().await;

            // Verify results independently
            let result = client.get_concatenated_commands().await;
            assert_eq!(result.len(), 2 * num_proposes_per_thread * num_threads);

            drop(client);

            result
        });
        handles.push(handle);
    }

    // Run all tasks concurrently
    drop(fixture);
    env_rc.run();

    // Await all handles and collect results
    let mut results = Vec::new();
    for handle in handles {
        let result = block_on(handle).expect("Task should complete successfully");
        results.push(result);
    }

    // All results should be identical (shared state)
    let first_result = &results[0];
    for result in &results[1..] {
        assert_eq!(
            result, first_result,
            "All clients should see the same concatenated commands"
        );
    }

    first_result.clone()
}

/// Test multiple nodes sharing the same state and environment
///
/// This test runs the scenario twice with the same seed to verify determinism.
/// Since it needs to create two fixtures with the same seed, it creates
/// fixtures internally.
pub fn run_test_multiple_nodes<F>() -> Result<()>
where
    F: SimulatorFixture + AgentBusTestFixture<Env = Simulator>,
{
    let seed: u64 = rand::thread_rng().r#gen();

    // Run 1: Create fixture with seed
    let simulator1 = Simulator::new(seed);
    let fixture1 = F::new(simulator1);
    let run1 = run_scenario(fixture1);

    // Run 2: Create another fixture with the same seed
    let simulator2 = Simulator::new(seed);
    let fixture2 = F::new(simulator2);
    let run2 = run_scenario(fixture2);

    assert_eq!(
        run1, run2,
        "Both runs should produce identical results (determinism)"
    );

    Ok(())
}
