// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

use agent_bus_proto_rust::agent_bus::*;
use agentbus_api::AgentBus;
use agentbus_api::environment::Environment;
use rand::Rng;

use crate::common::helpers::payload_to_selective_poll_type;
use crate::common::helpers::poll;
use crate::common::helpers::poll_selective;
use crate::common::helpers::propose_decider_policy;
use crate::common::helpers::propose_string_intention;
use crate::common::helpers::propose_vote;
use crate::common::helpers::variant_name_to_payload;
use crate::fixtures::AgentBusTestFixture;

pub async fn run_test_bounds<F: AgentBusTestFixture>(fixture: &F) -> anyhow::Result<()> {
    let environment = fixture.get_env();
    let impl_instance = fixture.create_impl();
    let agent_bus_id: String = format!("bus-{}", environment.with_rng(|rng| rng.r#gen::<u64>()));
    let num_entries: usize = environment.with_rng(|rng| rng.gen_range(5..15));
    for i in 1..=num_entries {
        propose_string_intention(&impl_instance, agent_bus_id.clone(), format!("entry{}", i)).await;
    }

    let max_entries_request: i16 =
        environment.with_rng(|rng| rng.gen_range((num_entries as i16)..100));
    let result = poll(&impl_instance, agent_bus_id.clone(), 0, max_entries_request).await;

    assert_eq!(
        result.entries.len(),
        num_entries,
        "Should retrieve all {} entries",
        num_entries
    );

    let result = poll(&impl_instance, agent_bus_id.clone(), 0, 0).await;

    assert_eq!(
        result.entries.len(),
        0,
        "maxEntries=0 should return no entries"
    );

    let limited_max_entries: i16 =
        environment.with_rng(|rng| rng.gen_range(1..(num_entries as i16)));
    let result = poll(&impl_instance, agent_bus_id.clone(), 0, limited_max_entries).await;

    assert_eq!(
        result.entries.len(),
        limited_max_entries as usize,
        "Should respect maxEntries limit"
    );

    let beyond_offset: i64 = environment.with_rng(|rng| rng.gen_range(1..20));
    let max_entries_beyond: i16 = environment.with_rng(|rng| rng.gen_range(5..50));
    let result = poll(
        &impl_instance,
        agent_bus_id.clone(),
        num_entries as i64 + beyond_offset,
        max_entries_beyond,
    )
    .await;

    assert_eq!(
        result.entries.len(),
        0,
        "Polling beyond available entries should return empty"
    );

    let middle_position: i64 = environment.with_rng(|rng| rng.gen_range(1..(num_entries as i64)));
    let max_entries_middle: i16 = environment.with_rng(|rng| rng.gen_range(5..50));
    let result = poll(
        &impl_instance,
        agent_bus_id.clone(),
        middle_position,
        max_entries_middle,
    )
    .await;

    assert!(
        result.entries.len() <= (num_entries - middle_position as usize),
        "Should return entries from middle of log"
    );

    let different_agent_bus_id = format!("{}-different", agent_bus_id);
    let max_entries_diff: i16 = environment.with_rng(|rng| rng.gen_range(5..50));
    let result = poll(&impl_instance, different_agent_bus_id, 0, max_entries_diff).await;

    assert_eq!(
        result.entries.len(),
        0,
        "Different agentBusId should have no entries"
    );

    let server_max_entries = 64;
    let extra_entries: usize = environment.with_rng(|rng| rng.gen_range(10..50));
    let total_entries = server_max_entries + extra_entries;

    let test_agent_bus_id = format!("bus-{}", environment.with_rng(|rng| rng.r#gen::<u64>()));

    for i in 1..=total_entries {
        propose_string_intention(
            &impl_instance,
            test_agent_bus_id.clone(),
            format!("entry{}", i),
        )
        .await;
    }

    let request_max_entries: i16 = environment.with_rng(|rng| rng.gen_range(500..2000));
    let result = poll(
        &impl_instance,
        test_agent_bus_id.clone(),
        0,
        request_max_entries,
    )
    .await;

    assert_eq!(
        result.entries.len(),
        server_max_entries,
        "Server should cap at MAX_POLL_ENTRIES ({}), even when requesting more",
        server_max_entries
    );

    let request_max_entries_second: i16 = environment.with_rng(|rng| rng.gen_range(500..2000));
    let result = poll(
        &impl_instance,
        test_agent_bus_id.clone(),
        server_max_entries as i64,
        request_max_entries_second,
    )
    .await;

    assert_eq!(
        result.entries.len(),
        extra_entries,
        "Should get remaining entries on second poll"
    );
    Ok(())
}

pub async fn run_test_complete_flag<F: AgentBusTestFixture>(fixture: &F) -> anyhow::Result<()> {
    let environment = fixture.get_env();
    let impl_instance = fixture.create_impl();
    let agent_bus_id: String = format!("bus-{}", environment.with_rng(|rng| rng.r#gen::<u64>()));
    let num_intentions = 5;
    let num_votes = 3;

    for i in 0..num_intentions {
        propose_string_intention(
            &impl_instance,
            agent_bus_id.clone(),
            format!("intention{}", i),
        )
        .await;
    }
    for i in 0..num_votes {
        propose_vote(&impl_instance, agent_bus_id.clone(), i, i % 2 == 0).await;
    }

    let mut intention_filter = vec![];
    intention_filter.push(SelectivePollType::Intention as i32);

    let result = poll_selective(
        &impl_instance,
        agent_bus_id.clone(),
        0,
        num_intentions,
        intention_filter.clone(),
    )
    .await;
    assert_eq!(
        result.entries.len(),
        num_intentions as usize,
        "Should return all intentions"
    );
    assert!(
        result.complete,
        "Should be complete when all matching entries returned despite votes after"
    );

    let result = poll_selective(&impl_instance, agent_bus_id.clone(), 0, 3, intention_filter).await;
    assert_eq!(
        result.entries.len(),
        3,
        "Should respect maxEntries with filter"
    );
    assert!(
        !result.complete,
        "Should not be complete when limited by maxEntries with filter"
    );
    Ok(())
}

pub async fn run_test_selective_poll<F: AgentBusTestFixture>(fixture: &F) -> anyhow::Result<()> {
    let environment = fixture.get_env();
    let impl_instance = fixture.create_impl();
    let agent_bus_id: String = format!("bus-{}", environment.with_rng(|rng| rng.r#gen::<u64>()));
    let num_intentions: usize = environment.with_rng(|rng| rng.gen_range(3..8));
    let num_votes: usize = environment.with_rng(|rng| rng.gen_range(3..8));
    let num_policies: usize = environment.with_rng(|rng| rng.gen_range(2..5));

    let mut intention_positions = Vec::new();
    let mut vote_positions = Vec::new();
    let mut policy_positions = Vec::new();

    for i in 0..num_intentions {
        let pos = propose_string_intention(
            &impl_instance,
            agent_bus_id.clone(),
            format!("intention{}", i),
        )
        .await;
        intention_positions.push(pos);
    }

    for i in 0..num_votes {
        let intention_id = if !intention_positions.is_empty() {
            intention_positions[i % intention_positions.len()]
        } else {
            0
        };
        let vote = i % 2 == 0;
        let pos = propose_vote(&impl_instance, agent_bus_id.clone(), intention_id, vote).await;
        vote_positions.push(pos);
    }

    for i in 0..num_policies {
        let policy = match i % 3 {
            0 => DeciderPolicy::OffByDefault as i32,
            1 => DeciderPolicy::OnByDefault as i32,
            _ => DeciderPolicy::FirstBooleanWins as i32,
        };
        let pos = propose_decider_policy(&impl_instance, agent_bus_id.clone(), policy).await;
        policy_positions.push(pos);
    }

    let total_entries = num_intentions + num_votes + num_policies;

    let result = poll(&impl_instance, agent_bus_id.clone(), 0, 100).await;
    assert_eq!(
        result.entries.len(),
        total_entries,
        "Poll without filter should return all entries"
    );

    let mut intention_filter = vec![];
    intention_filter.push(SelectivePollType::Intention as i32);
    let result = poll_selective(
        &impl_instance,
        agent_bus_id.clone(),
        0,
        100,
        intention_filter,
    )
    .await;
    assert_eq!(
        result.entries.len(),
        num_intentions,
        "Selective poll should return only intentions"
    );
    for entry in &result.entries {
        assert!(
            matches!(&entry.payload, Some(payload) if matches!(&payload.payload, Some(agent_bus_proto_rust::agent_bus::payload::Payload::Intention(_)))),
            "All entries should be intentions"
        );
    }

    let mut vote_filter = vec![];
    vote_filter.push(SelectivePollType::Vote as i32);
    let result = poll_selective(&impl_instance, agent_bus_id.clone(), 0, 100, vote_filter).await;
    assert_eq!(
        result.entries.len(),
        num_votes,
        "Selective poll should return only votes"
    );
    for entry in &result.entries {
        assert!(
            matches!(&entry.payload, Some(payload) if matches!(&payload.payload, Some(agent_bus_proto_rust::agent_bus::payload::Payload::Vote(_)))),
            "All entries should be votes"
        );
    }

    let mut policy_filter = vec![];
    policy_filter.push(SelectivePollType::DeciderPolicy as i32);
    let result = poll_selective(&impl_instance, agent_bus_id.clone(), 0, 100, policy_filter).await;
    assert_eq!(
        result.entries.len(),
        num_policies,
        "Selective poll should return only policies"
    );
    for entry in &result.entries {
        assert!(
            matches!(&entry.payload, Some(payload) if matches!(&payload.payload, Some(agent_bus_proto_rust::agent_bus::payload::Payload::DeciderPolicy(_)))),
            "All entries should be policies"
        );
    }

    let mut intention_vote_filter = vec![];
    intention_vote_filter.push(SelectivePollType::Intention as i32);
    intention_vote_filter.push(SelectivePollType::Vote as i32);
    let result = poll_selective(
        &impl_instance,
        agent_bus_id.clone(),
        0,
        100,
        intention_vote_filter,
    )
    .await;
    assert_eq!(
        result.entries.len(),
        num_intentions + num_votes,
        "Selective poll should return intentions and votes"
    );
    for entry in &result.entries {
        assert!(
            matches!(&entry.payload, Some(payload) if matches!(&payload.payload, Some(agent_bus_proto_rust::agent_bus::payload::Payload::Intention(_))))
                || matches!(&entry.payload, Some(payload) if matches!(&payload.payload, Some(agent_bus_proto_rust::agent_bus::payload::Payload::Vote(_)))),
            "All entries should be intentions or votes"
        );
    }

    let empty_filter = vec![];
    let result = poll_selective(&impl_instance, agent_bus_id.clone(), 0, 100, empty_filter).await;
    assert_eq!(
        result.entries.len(),
        0,
        "Empty filter should return no entries"
    );

    let limited_max: i16 = environment.with_rng(|rng| rng.gen_range(1..num_intentions as i16));
    let mut intention_filter = vec![];
    intention_filter.push(SelectivePollType::Intention as i32);
    let result = poll_selective(
        &impl_instance,
        agent_bus_id.clone(),
        0,
        limited_max,
        intention_filter,
    )
    .await;
    assert_eq!(
        result.entries.len(),
        limited_max as usize,
        "Selective poll should respect maxEntries"
    );
    Ok(())
}

pub async fn run_test_with_decider<F: AgentBusTestFixture>(fixture: &F) -> anyhow::Result<()> {
    let environment = fixture.get_env();
    let decider_impl = fixture.create_impl();
    let agent_bus = fixture.create_impl();
    use agentbus_core::decider::Decider;
    use agentbus_core::vote_trackers::create_vote_tracker_for_decider_policy;

    let agent_bus_id: String = format!("bus-{}", environment.with_rng(|rng| rng.r#gen::<u64>()));

    propose_decider_policy(
        &agent_bus,
        agent_bus_id.clone(),
        DeciderPolicy::OnByDefault as i32,
    )
    .await;

    let mut decider = Decider::new(
        decider_impl,
        agent_bus_id.clone(),
        DeciderPolicy::OnByDefault as i32,
        create_vote_tracker_for_decider_policy,
        environment.clone(),
    );

    for i in 1..=3 {
        propose_string_intention(&agent_bus, agent_bus_id.clone(), format!("command{}", i)).await;
    }

    let entries_processed = decider
        .poll_and_decide()
        .await
        .expect("Decider should process entries");

    assert_eq!(entries_processed, 4, "Should process 4 entries");

    let poll_result = poll(&agent_bus, agent_bus_id.clone(), 0, 100).await;

    assert_eq!(
        poll_result.entries.len(),
        7,
        "Should have 1 policy + 3 intentions + 3 commits"
    );
    assert_eq!(
        poll_result
            .entries
            .iter()
            .filter(|e| matches!(&e.payload, Some(payload) if matches!(&payload.payload, Some(agent_bus_proto_rust::agent_bus::payload::Payload::Commit(_)))))
            .count(),
        3,
        "Should have 3 commits"
    );

    propose_decider_policy(
        &agent_bus,
        agent_bus_id.clone(),
        DeciderPolicy::FirstBooleanWins as i32,
    )
    .await;

    let intention_pos1 =
        propose_string_intention(&agent_bus, agent_bus_id.clone(), "vote1".to_string()).await;
    let intention_pos2 =
        propose_string_intention(&agent_bus, agent_bus_id.clone(), "vote2".to_string()).await;

    decider
        .poll_and_decide()
        .await
        .expect("Decider should process policy change and intentions");

    propose_vote(&agent_bus, agent_bus_id.clone(), intention_pos1, true).await;
    propose_vote(&agent_bus, agent_bus_id.clone(), intention_pos2, false).await;

    decider
        .poll_and_decide()
        .await
        .expect("Decider should process votes");

    let poll_result2 = poll(&agent_bus, agent_bus_id.clone(), 0, 100).await;
    assert_eq!(
        poll_result2
            .entries
            .iter()
            .filter(|e| matches!(&e.payload, Some(payload) if matches!(&payload.payload, Some(agent_bus_proto_rust::agent_bus::payload::Payload::Commit(_)))))
            .count(),
        4,
        "Should have 4 total commits"
    );
    assert_eq!(
        poll_result2
            .entries
            .iter()
            .filter(|e| matches!(&e.payload, Some(payload) if matches!(&payload.payload, Some(agent_bus_proto_rust::agent_bus::payload::Payload::Abort(_)))))
            .count(),
        1,
        "Should have 1 abort"
    );

    // Test vote for non-existent intention
    let nonexistent_intention_id = 999999;
    let entries_before = poll(&agent_bus, agent_bus_id.clone(), 0, 100)
        .await
        .entries
        .len();

    propose_vote(
        &agent_bus,
        agent_bus_id.clone(),
        nonexistent_intention_id,
        true,
    )
    .await;

    // Decider should process the vote without crashing
    decider
        .poll_and_decide()
        .await
        .expect("Decider should handle vote for non-existent intention gracefully");

    // Verify no commit or abort was created for the non-existent intention
    let entries_after = poll(&agent_bus, agent_bus_id.clone(), 0, 100).await.entries;
    assert_eq!(
        entries_after.len(),
        entries_before + 1,
        "Should have only the vote entry added, no commit/abort"
    );

    // Verify the last entry is the vote (not a commit or abort)
    assert!(
        matches!(
            &entries_after.last().unwrap().payload,
            Some(payload) if matches!(&payload.payload, Some(agent_bus_proto_rust::agent_bus::payload::Payload::Vote(_)))
        ),
        "Last entry should be the vote for non-existent intention"
    );

    // Verify still only 4 commits and 1 abort (no new decisions)
    assert_eq!(
        entries_after
            .iter()
            .filter(|e| matches!(&e.payload, Some(payload) if matches!(&payload.payload, Some(agent_bus_proto_rust::agent_bus::payload::Payload::Commit(_)))))
            .count(),
        4,
        "Should still have only 4 commits"
    );
    assert_eq!(
        entries_after
            .iter()
            .filter(|e| matches!(&e.payload, Some(payload) if matches!(&payload.payload, Some(agent_bus_proto_rust::agent_bus::payload::Payload::Abort(_)))))
            .count(),
        1,
        "Should still have only 1 abort"
    );
    Ok(())
}

/// Test that verifies every SelectivePollType enum variant has a corresponding Payload variant.
/// This uses the thrift-generated enumerate() method to iterate through all SelectivePollType values.
/// The test will fail at compile time if new Payload variants are added (due to exhaustive matching in payload_to_selective_poll_type),
/// and will fail at runtime if new SelectivePollType variants are added without corresponding Payloads.
#[test]
fn test_selective_poll_type_coverage() {
    use crate::common::helpers::selective_poll_type_to_payload;

    // Iterate through all SelectivePollType variants using the thrift-generated enumerate() method
    // Manually list all SelectivePollType variants
    let poll_types = vec![
        SelectivePollType::Intention as i32,
        SelectivePollType::Vote as i32,
        SelectivePollType::DeciderPolicy as i32,
        SelectivePollType::Commit as i32,
        SelectivePollType::Abort as i32,
        SelectivePollType::VoterPolicy as i32,
        SelectivePollType::Control as i32,
    ];
    for poll_type in poll_types {
        let payload = selective_poll_type_to_payload(poll_type).unwrap_or_else(|| {
            panic!(
                "No corresponding Payload variant found for SelectivePollType::{:?}. \
             Please add a mapping in selective_poll_type_to_payload.",
                poll_type
            )
        });

        // Verify the reverse mapping works correctly
        let mapped_poll_type = payload_to_selective_poll_type(&payload);
        assert_eq!(
            mapped_poll_type, poll_type,
            "Payload created for {:?} maps back to {:?} instead",
            poll_type, mapped_poll_type
        );
    }
}

/// Test that enumerates through all Payload variant names and proposes each to the agentbus.
/// Uses the thrift-generated variant_names() method to list all Payload variants.
pub async fn run_test_all_payload_types<F: AgentBusTestFixture>(fixture: &F) -> anyhow::Result<()> {
    let environment = fixture.get_env();
    let impl_instance = fixture.create_impl();
    let agent_bus_id: String = format!("bus-{}", environment.with_rng(|rng| rng.r#gen::<u64>()));

    // List all Payload variant names manually (proto doesn't have reflection like Thrift)
    let variant_names = vec![
        "intention",
        "vote",
        "deciderPolicy",
        "commit",
        "abort",
        "voterPolicy",
        "control",
    ];
    let payload_count = variant_names.len();

    for variant_name in variant_names {
        let payload = variant_name_to_payload(variant_name);

        let request = agent_bus_proto_rust::agent_bus::ProposeRequest {
            agent_bus_id: agent_bus_id.clone(),
            payload: Some(payload),
            ..Default::default()
        };
        impl_instance
            .propose(request)
            .await
            .expect("Propose should succeed");
    }

    let result = poll(&impl_instance, agent_bus_id.clone(), 0, 100).await;
    assert_eq!(
        result.entries.len(),
        payload_count,
        "Should have {} entries (one of each payload type)",
        payload_count
    );

    for entry in &result.entries {
        let poll_type = payload_to_selective_poll_type(entry.payload.as_ref().unwrap());
        let mut filter = vec![];
        filter.push(poll_type);

        let selective_result =
            poll_selective(&impl_instance, agent_bus_id.clone(), 0, 100, filter).await;

        assert_eq!(
            selective_result.entries.len(),
            1,
            "Selective poll should return exactly one entry for {:?}",
            poll_type
        );

        assert_eq!(
            payload_to_selective_poll_type(selective_result.entries[0].payload.as_ref().unwrap()),
            poll_type,
            "Returned entry should match the requested poll type"
        );
    }
    Ok(())
}

/// Test that validates bus ID string validation is enforced across all AgentBus implementations.
pub async fn run_test_bus_id_validation<F: AgentBusTestFixture>(fixture: &F) -> anyhow::Result<()> {
    use agentbus_api::validation::MAX_BUS_ID_LEN;
    use agentbus_api::validation::is_valid_bus_id_char;

    let impl_instance = fixture.create_impl();

    let make_propose_request = |bus_id: String| agent_bus_proto_rust::agent_bus::ProposeRequest {
        agent_bus_id: bus_id,
        payload: Some(agent_bus_proto_rust::agent_bus::Payload {
            payload: Some(
                agent_bus_proto_rust::agent_bus::payload::Payload::Intention(
                    agent_bus_proto_rust::agent_bus::Intention {
                        intention: Some(
                            agent_bus_proto_rust::agent_bus::intention::Intention::StringIntention(
                                "test".to_string(),
                            ),
                        ),
                    },
                ),
            ),
        }),
        ..Default::default()
    };

    // Test valid bus IDs
    let max_length_id = "a".repeat(MAX_BUS_ID_LEN);
    let valid_ids = vec![
        "a",
        "valid-bus-id",
        "valid_bus_id",
        "valid.bus.id",
        "valid/bus/id",
        "a1b2c3",
        "MixedCase123",
        &max_length_id, // exactly at max length
    ];
    for valid_id in valid_ids {
        let result =
            propose_string_intention(&impl_instance, valid_id.to_string(), "test".to_string())
                .await;
        assert!(
            result >= 0,
            "Valid bus ID '{}' should be accepted",
            valid_id
        );
    }

    // Test empty bus ID (should fail)
    let empty_result = impl_instance
        .propose(make_propose_request("".to_string()))
        .await;
    assert!(empty_result.is_err(), "Empty bus ID should be rejected");

    // Test bus ID exceeding max length (should fail)
    let long_id = "a".repeat(MAX_BUS_ID_LEN + 1);
    let long_result = impl_instance.propose(make_propose_request(long_id)).await;
    assert!(
        long_result.is_err(),
        "Bus ID exceeding max length should be rejected"
    );

    // Test invalid characters (should fail)
    // Check all possible byte values to find invalid characters
    let invalid_chars: Vec<char> = (0u8..=255u8)
        .filter_map(|b| {
            let c = b as char;
            if !is_valid_bus_id_char(c) {
                Some(c)
            } else {
                None
            }
        })
        .collect();
    assert!(
        invalid_chars.contains(&'#'),
        "Test setup error: should have '#' as invalid character"
    );

    for invalid_char in invalid_chars {
        let invalid_id = format!("bus{}id", invalid_char);
        let invalid_result = impl_instance
            .propose(make_propose_request(invalid_id))
            .await;
        assert!(
            invalid_result.is_err(),
            "Bus ID with invalid character '{}' (byte {}) should be rejected",
            invalid_char.escape_debug(),
            invalid_char as u8
        );
    }
    Ok(())
}
