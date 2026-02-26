// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Macros for generating simulator-based test suites for WriteOnceSpace

#[macro_export]
macro_rules! write_once_space_gen_test {
    ($test_name:ident, $suffix:ident, $fixture:ty) => {
        paste::paste! {
            #[test]
            fn [<$test_name _ $suffix>]() {
                use $crate::write_once_space::fixtures::WriteOnceSpaceTestFixture;
                use $crate::fixtures::SimulatorFixture;
                use rand::Rng;
                let seed: u64 = rand::thread_rng().r#gen();
                let simulator = $crate::simulator::Simulator::new(seed);
                let fixture = <$fixture as SimulatorFixture>::new(simulator);
                let env_rc = fixture.get_env();
                let handle = env_rc.spawn(async move {
                    $crate::write_once_space::scenarios::[<run_ $test_name>](&fixture).await
                });
                env_rc.run();
                $crate::futures::executor::block_on(handle).expect("Test scenario should complete successfully").unwrap();
            }
        }
    };
}

/// Macro to generate a complete simulator-based test suite for a WriteOnceSpace implementation.
///
/// # Usage
/// ```ignore
/// write_once_space_tests!(InMemoryWriteOnceSpaceFixture, in_memory);
/// ```
///
/// This generates test functions like:
/// - `test_write_then_read_in_memory`
/// - `test_read_nonexistent_in_memory`
/// - `test_write_once_semantics_in_memory`
/// - `test_tail_in_memory`
/// - `test_different_clients_in_memory`
#[macro_export]
macro_rules! write_once_space_tests {
    ($fixture:ty, $suffix:ident) => {
        $crate::write_once_space_gen_test!(test_write_then_read, $suffix, $fixture);
        $crate::write_once_space_gen_test!(test_read_nonexistent, $suffix, $fixture);
        $crate::write_once_space_gen_test!(test_write_once_semantics, $suffix, $fixture);
        $crate::write_once_space_gen_test!(test_tail, $suffix, $fixture);
        $crate::write_once_space_gen_test!(test_different_clients, $suffix, $fixture);
    };
}
