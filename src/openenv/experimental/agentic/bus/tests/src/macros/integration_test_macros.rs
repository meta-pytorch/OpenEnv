// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary

//! Macros for generating integration test suites

#[macro_export]
macro_rules! integration_test {
    ($test_name:ident, $suffix:ident, $fixture:ty) => {
        paste::paste! {
            #[fbinit::test]
            async fn [<$test_name _ $suffix>](fb: fbinit::FacebookInit) -> anyhow::Result<()> {
                use $crate::fixtures::IntegrationFixture;
                let fixture = <$fixture as IntegrationFixture>::new_async(fb).await?;
                $crate::scenarios::[<run_ $test_name>](&fixture).await?;
                Ok(())
            }
        }
    };
}

/// Macro to generate a complete integration test suite.
///
/// # Usage
/// ```ignore
/// integration_tests!(IntegrationTestFixture, integration);
/// ```
///
/// This generates test functions like:
/// - `test_bounds_integration`
/// - `test_selective_poll_integration`
/// - `test_with_decider_integration`
#[macro_export]
macro_rules! integration_tests {
    ($fixture:ty, $suffix:ident) => {
        $crate::integration_test!(test_bounds, $suffix, $fixture);
        $crate::integration_test!(test_selective_poll, $suffix, $fixture);
        $crate::integration_test!(test_complete_flag, $suffix, $fixture);
        $crate::integration_test!(test_all_payload_types, $suffix, $fixture);
        $crate::integration_test!(test_bus_id_validation, $suffix, $fixture);
        $crate::integration_test!(test_with_decider, $suffix, $fixture);
    };
}
