// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Macros for generating tests that verify linearizability violations are detected

#[macro_export]
macro_rules! buggy_test {
    ($test_name:ident, $suffix:ident, $fixture:ty, $seed:expr) => {
        paste::paste! {
            #[test]
            fn [<$test_name _ $suffix>]() {
                use $crate::fixtures::AgentBusTestFixture;
                use $crate::fixtures::SimulatorFixture;
                let simulator = $crate::simulator::Simulator::new($seed);
                let fixture = <$fixture as SimulatorFixture>::new(simulator);
                let env_rc = fixture.get_env();

                let handle = env_rc.spawn(async move {
                    $crate::scenarios::[<run_ $test_name>](&fixture).await
                });

                env_rc.run();
                let result = $crate::futures::executor::block_on(handle).expect("Task should complete");

                assert!(
                    result.is_err(),
                    "Expected linearizability violation to be detected, but test passed. Seed: {}",
                    $seed
                );
            }
        }
    };
}

/// Macro to generate a buggy test that verifies a buggy AgentBus implementation
/// is caught by the linearizability checker.
///
/// # Usage
/// ```ignore
/// buggy_tests!(ChainedAgentBusBuggyPollFixture, chained_agentbus_buggy_poll, 1);
/// ```
///
/// This generates a test function like:
/// - `test_multi_worker_counter_skip_commits_with_seed_chained_agentbus_buggy_poll`
#[macro_export]
macro_rules! buggy_tests {
    ($fixture:ty, $suffix:ident, $seed:expr) => {
        $crate::buggy_test!(
            test_multi_worker_counter_skip_commits_with_seed,
            $suffix,
            $fixture,
            $seed
        );
    };
}
