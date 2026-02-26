// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Failure injection test macros

/// Macro to generate a single failure injection test
#[macro_export]
macro_rules! fi_test {
    (
        $fixture:ty,
        $prefix:ident,
        $test_name:ident,
        $prob_lost:expr,
        $prob_commit_then_error:expr,
        $prob_error_then_commit:expr,
        $expect_success:expr,
        $expect_response:expr,
        $expected_error:expr
    ) => {
        paste::paste! {
            #[test]
            fn [<test_ $prefix _fi_ $test_name>]() {
                use agentbus_tests::fixtures::AgentBusTestFixture;
                use agentbus_tests::common::fault_config::FaultConfig;
                use agentbus_tests::simulator::Simulator;

                let seed: u64 = rand::random();
                let config = FaultConfig {
                    prob_lost: $prob_lost,
                    prob_commit_then_error: $prob_commit_then_error,
                    prob_error_then_commit: $prob_error_then_commit,
                };
                let fixture = <$fixture>::new_with_config(Simulator::new(seed), config);
                let agentbus = fixture.create_impl();
                let env = fixture.get_env();
                drop(fixture);
                $crate::scenarios::failure_injection_scenarios::run_fault_test(
                    agentbus,
                    env,
                    /* should_succeed: */ $expect_success,
                    /* should_commit: */ $expect_response,
                    /* expected_error_msg: */ $expected_error,
                );
            }
        }
    };
}

/// Macro to generate all failure injection tests for a given fixture
#[macro_export]
macro_rules! failure_injection_tests {
    ($fixture:ty, $prefix:ident) => {
        $crate::fi_test!(
            $fixture, $prefix, success, /* prob_lost: */ 0.0,
            /* prob_commit_then_error: */ 0.0, /* prob_error_then_commit: */ 0.0,
            /* expect_success: */ true, /* expect_response: */ true,
            /* expected_error: */ None
        );
        $crate::fi_test!(
            $fixture,
            $prefix,
            lost_request,
            /* prob_lost: */ 1.0,
            /* prob_commit_then_error: */ 0.0,
            /* prob_error_then_commit: */ 0.0,
            /* expect_success: */ false,
            /* expect_response: */ false,
            /* expected_error: */ Some("Lost")
        );
        $crate::fi_test!(
            $fixture,
            $prefix,
            commit_then_error,
            /* prob_lost: */ 0.0,
            /* prob_commit_then_error: */ 1.0,
            /* prob_error_then_commit: */ 0.0,
            /* expect_success: */ false,
            /* expect_response: */ true,
            /* expected_error: */ Some("CommitThenError")
        );
        $crate::fi_test!(
            $fixture,
            $prefix,
            error_then_commit,
            /* prob_lost: */ 0.0,
            /* prob_commit_then_error: */ 0.0,
            /* prob_error_then_commit: */ 1.0,
            /* expect_success: */ false,
            /* expect_response: */ true,
            /* expected_error: */ Some("ErrorThenCommit")
        );

        paste::paste! {
            #[test]
            fn [<test_ $prefix _fi_permutations>]() {
                use agentbus_tests::common::fault_config::FaultConfig;
                use agentbus_tests::simulator::Simulator;

                let seed: u64 = rand::random();
                let config = FaultConfig {
                    prob_lost: 0.25,
                    prob_commit_then_error: 0.25,
                    prob_error_then_commit: 0.25,
                };
                let fixture = <$fixture>::new_with_config(Simulator::new(seed), config);
                let agentbus = fixture.create_impl();
                let env = fixture.get_env();
                drop(fixture);
                $crate::scenarios::failure_injection_scenarios::run_permutations_with_faults_test(
                    agentbus,
                    env,
                );
            }
        }
    };
}
