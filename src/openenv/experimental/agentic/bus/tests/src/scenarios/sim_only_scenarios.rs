// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

use agentbus_api::environment::Clock;
use agentbus_api::environment::Environment;
use agentbus_core::decider::Decider;
use agentbus_core::vote_trackers::create_vote_tracker_for_decider_policy;
use rand::Rng;

use crate::common::helpers::poll;
use crate::common::helpers::propose_decider_policy;
use crate::common::helpers::propose_string_intention;
use crate::fixtures::AgentBusTestFixture;
use crate::simulator::Simulator;

pub async fn run_test_with_decider_run_loop<F: AgentBusTestFixture<Env = Simulator>>(
    fixture: &F,
) -> anyhow::Result<()> {
    let environment = fixture.get_env();
    let decider_impl = fixture.create_impl();
    let impl_instance = fixture.create_impl();
    let agent_bus_id = format!("bus-{}", environment.with_rng(|rng| rng.r#gen::<u64>()));
    let poll_interval = std::time::Duration::from_millis(10);

    let start_time = environment.with_clock(|clock| clock.current_time());

    let sleep1 = environment.sleep(poll_interval);
    let sleep2 = environment.sleep(poll_interval * 2);

    propose_decider_policy(
        &impl_instance,
        agent_bus_id.clone(),
        agent_bus_proto_rust::agent_bus::DeciderPolicy::OnByDefault as i32,
    )
    .await;

    let mut decider = Decider::new(
        decider_impl,
        agent_bus_id.clone(),
        agent_bus_proto_rust::agent_bus::DeciderPolicy::OnByDefault as i32,
        create_vote_tracker_for_decider_policy,
        environment.clone(),
    );

    decider.poll_and_decide().await.expect("poll 1");

    sleep1.await;

    decider.poll_and_decide().await.expect("poll 2");

    sleep2.await;

    propose_string_intention(&impl_instance, agent_bus_id.clone(), "test".to_string()).await;

    let count = decider.poll_and_decide().await.expect("poll 3");
    assert_eq!(count, 1, "Should process 1 intention");

    let result = poll(&impl_instance, agent_bus_id, 0, 100).await;
    assert!(
        result
            .entries
            .iter()
            .any(|e| matches!(&e.payload, Some(payload) if matches!(&payload.payload, Some(agent_bus_proto_rust::agent_bus::payload::Payload::Commit(_)))))
    );

    let end_time = environment.with_clock(|clock| clock.current_time());
    let elapsed = end_time.duration_since(start_time);
    let expected = poll_interval * 2;
    assert!(
        elapsed >= expected,
        "Clock should advance by at least 2 poll intervals (expected: {:?}, got: {:?})",
        expected,
        elapsed
    );
    Ok(())
}
