// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Failure injection test scenarios for AgentBus implementations
//!
//! These tests verify that AgentBus implementations handle failure scenarios
//! correctly when configured with fault injection.

use std::rc::Rc;
use std::time::Duration;

use agent_bus_proto_rust::agent_bus::*;
use agentbus_api::AgentBus;
use agentbus_api::environment::Environment;

use crate::simulator::Simulator;

/// Helper function to run a fault injection test with a given configuration
pub fn run_fault_test<Impl>(
    agentbus: Impl,
    env: Rc<Simulator>,
    should_succeed: bool,
    should_commit: bool,
    expected_error_msg: Option<&str>,
) where
    Impl: AgentBus + Clone + 'static,
{
    let payload = Payload {
        payload: Some(payload::Payload::Intention(Intention {
            intention: Some(intention::Intention::StringIntention("test".to_string())),
        })),
    };
    let request = ProposeRequest {
        agent_bus_id: "bus-1".to_string(),
        payload: Some(payload.clone()),
    };

    let poll_request = PollRequest {
        agent_bus_id: "bus-1".to_string(),
        start_log_position: 0,
        max_entries: 10,
        filter: None,
    };

    let agentbus_clone1 = agentbus.clone();
    let agentbus_clone2 = agentbus.clone();
    drop(agentbus);

    let sleep_duration = Duration::from_secs(u32::MAX as u64); // ~136 years
    let sleep_fut = env.sleep(sleep_duration);
    let handle = env.spawn(async move {
        let result = agentbus_clone1.propose(request).await;
        // Sleep for "infinite" time to ensure that delayed commits (if any) complete.
        // (Alternative: call run() twice; the first will return once everything quiesces.)
        sleep_fut.await;
        let poll_response = agentbus_clone2.poll(poll_request).await;
        (result, poll_response)
    });

    env.run();

    let (result, poll_result) = futures::executor::block_on(handle).expect("Task should complete");

    if should_succeed {
        let response = result.expect("Proposal should succeed");
        assert_eq!(response.log_position, 0);
    } else {
        assert!(result.is_err());
        if let Some(msg) = expected_error_msg {
            let err = result.expect_err("Proposal should fail");
            assert!(
                err.to_string().contains(msg),
                "Error should contain: {}",
                msg
            );
        }
    }

    let poll_response = poll_result.expect("Poll should succeed");
    if should_commit {
        assert_eq!(poll_response.entries.len(), 1);
        assert_eq!(
            poll_response.entries[0]
                .header
                .as_ref()
                .unwrap()
                .log_position,
            0
        );
        assert_eq!(poll_response.entries[0].payload, Some(payload));
    } else {
        assert_eq!(poll_response.entries.len(), 0);
    }
    assert!(poll_response.complete);
}

fn make_request(bus_id: String, value: &str) -> ProposeRequest {
    ProposeRequest {
        agent_bus_id: bus_id,
        payload: Some(Payload {
            payload: Some(payload::Payload::Intention(Intention {
                intention: Some(intention::Intention::StringIntention(value.to_string())),
            })),
        }),
    }
}

pub fn run_permutations_with_faults_test<Impl>(agentbus: Impl, env: Rc<Simulator>)
where
    Impl: AgentBus + Clone + 'static,
{
    use std::collections::HashMap;

    const MAX_RUNS: i64 = 100000;
    const EXPECTED_OUTCOMES: [&str; 16] = [
        "", // all lost
        "a", "b", "c", //
        "ab", "ac", "ba", "bc", "ca", "cb", //
        "abc", "acb", "bac", "bca", "cab", "cba", //
    ];

    let mut observed_counts = HashMap::new();
    let mut bus_id = 0;

    // Run until we see all variants or hit the max
    while observed_counts.len() < EXPECTED_OUTCOMES.len() && bus_id < MAX_RUNS {
        let bus_id_str = format!("bus-{}", bus_id);

        // Client 1 for bus_id: propose "a", await, then propose "b"
        let agentbus_clone1 = agentbus.clone();
        let bus_id_str1 = bus_id_str.clone();
        env.spawn(async move {
            let _ = agentbus_clone1
                .propose(make_request(bus_id_str1.clone(), "a"))
                .await;
            let _ = agentbus_clone1
                .propose(make_request(bus_id_str1, "b"))
                .await;
        });

        // Client 2 for bus_id: propose "c"
        let agentbus_clone2 = agentbus.clone();
        let bus_id_str2 = bus_id_str.clone();
        env.spawn(async move {
            let _ = agentbus_clone2
                .propose(make_request(bus_id_str2, "c"))
                .await;
        });

        // Let things play out for this bus_id.
        env.run();

        // Poll this bus to get the outcome
        let agentbus_clone = agentbus.clone();
        let poll_request = PollRequest {
            agent_bus_id: bus_id_str,
            start_log_position: 0,
            max_entries: 100,
            filter: None,
        };
        let handle = env.spawn(async move { agentbus_clone.poll(poll_request).await });
        env.run();
        let poll_result = futures::executor::block_on(handle).expect("Task should complete");

        let outcome = if let Ok(response) = poll_result {
            response
                .entries
                .iter()
                .filter_map(|entry| {
                    entry.payload.as_ref().and_then(|payload| {
                        if let Some(payload::Payload::Intention(intention)) = &payload.payload {
                            if let Some(intention::Intention::StringIntention(s)) =
                                &intention.intention
                            {
                                Some(s.as_str())
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                })
                .collect::<String>()
        } else {
            String::new()
        };

        *observed_counts.entry(outcome).or_insert(0) += 1;

        bus_id += 1;
    }

    drop(agentbus);
    env.run();

    // Print all counts
    eprintln!("\nRan {} iterations", bus_id);
    eprintln!("All outcomes:");
    for expected in &EXPECTED_OUTCOMES {
        let count = observed_counts.get(*expected).unwrap_or(&0);
        let display = if expected.is_empty() {
            "(empty)"
        } else {
            expected
        };
        eprintln!("  {}: {} times", display, count);
    }

    // Verify we observed all expected outcomes
    assert_eq!(
        observed_counts.len(),
        EXPECTED_OUTCOMES.len(),
        "Should observe exactly {} distinct outcomes, but only saw {}. Missing: {:?}",
        EXPECTED_OUTCOMES.len(),
        observed_counts.len(),
        EXPECTED_OUTCOMES
            .iter()
            .filter(|&&e| !observed_counts.contains_key(e))
            .collect::<Vec<_>>()
    );
    for &expected in &EXPECTED_OUTCOMES {
        assert!(
            observed_counts.contains_key(expected),
            "Expected outcome '{}' was not observed",
            expected
        );
    }
}
