// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Macros for generating simulator-based test suites

#[macro_export]
macro_rules! gen_test {
    // Tests that accept a fixture.
    // We assume tests:
    //  - use run_test_* naming,
    //  - return Result<T> for some T, and
    //  - are exported from the scenarios module.
    ($test_name:ident, $suffix:ident, $fixture:ty) => {
        paste::paste! {
            #[test]
            fn [<$test_name _ $suffix>]() {
                use $crate::fixtures::AgentBusTestFixture;
                use $crate::fixtures::SimulatorFixture;
                use rand::Rng;
                let seed: u64 = rand::thread_rng().r#gen();
                let simulator = $crate::simulator::Simulator::new(seed);
                let fixture = <$fixture as SimulatorFixture>::new(simulator);
                let env_rc = fixture.get_env();
                let handle = env_rc.spawn(async move {
                    $crate::scenarios::[<run_ $test_name>](&fixture).await
                });
                env_rc.run();
                $crate::futures::executor::block_on(handle).expect("Test scenario should complete successfully").unwrap();
            }
        }
    };
    // Tests that create fixtures internally.
    // Same assumptions as above.
    ($test_name:ident, $suffix:ident, $fixture:ty, no_fixture) => {
        paste::paste! {
            #[test]
            fn [<$test_name _ $suffix>]() {
                $crate::scenarios::[<run_ $test_name>]::<$fixture>().unwrap();
            }
        }
    };
}

/// Macro to generate a complete simulator-based test suite for an AgentBus implementation.
///
/// # Usage
/// ```ignore
/// agent_bus_tests!(SimpleMemoryFixture, simple_memory);
/// ```
///
/// This generates test functions like:
/// - `test_multiple_nodes_simple_memory`
/// - `test_bounds_simple_memory`
/// - `test_with_decider_simple_memory`
/// - `test_multi_worker_counter_simple_memory`
#[macro_export]
macro_rules! agent_bus_tests {
    ($fixture:ty, $suffix:ident) => {
        $crate::gen_test!(test_multiple_nodes, $suffix, $fixture, no_fixture);

        $crate::gen_test!(test_multi_worker_counter, $suffix, $fixture);
        $crate::gen_test!(test_multi_worker_counter_large_poll, $suffix, $fixture);
        $crate::gen_test!(test_multi_worker_counter_small_poll, $suffix, $fixture);
        $crate::gen_test!(test_bounds, $suffix, $fixture);
        $crate::gen_test!(test_selective_poll, $suffix, $fixture);
        $crate::gen_test!(test_complete_flag, $suffix, $fixture);
        $crate::gen_test!(test_all_payload_types, $suffix, $fixture);
        $crate::gen_test!(test_bus_id_validation, $suffix, $fixture);
        $crate::gen_test!(test_with_decider, $suffix, $fixture);
        $crate::gen_test!(test_with_decider_run_loop, $suffix, $fixture);
    };
}
