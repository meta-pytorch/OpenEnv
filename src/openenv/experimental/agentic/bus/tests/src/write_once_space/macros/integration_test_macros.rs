// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Macros for generating integration test suites for WriteOnceSpace

#[macro_export]
macro_rules! write_once_space_integration_test {
    ($test_name:ident, $suffix:ident, $fixture:ty) => {
        paste::paste! {
            #[fbinit::test]
            async fn [<$test_name _ $suffix>](fb: fbinit::FacebookInit) -> anyhow::Result<()> {
                let fixture = <$fixture>::new_async(fb).await?;
                $crate::write_once_space::scenarios::[<run_ $test_name>](&fixture).await?;
                Ok(())
            }
        }
    };
}

/// Macro to generate a complete integration test suite for a WriteOnceSpace implementation.
///
/// # Usage
/// ```ignore
/// write_once_space_integration_tests!(ZippyWriteOnceSpaceFixture, zippy);
/// ```
///
/// This generates test functions like:
/// - `test_write_then_read_zippy`
/// - `test_read_nonexistent_zippy`
/// - `test_write_once_semantics_zippy`
/// - `test_tail_zippy`
/// - `test_different_clients_zippy`
#[macro_export]
macro_rules! write_once_space_integration_tests {
    ($fixture:ty, $suffix:ident) => {
        $crate::write_once_space_integration_test!(test_write_then_read, $suffix, $fixture);
        $crate::write_once_space_integration_test!(test_read_nonexistent, $suffix, $fixture);
        $crate::write_once_space_integration_test!(test_write_once_semantics, $suffix, $fixture);
        $crate::write_once_space_integration_test!(test_tail, $suffix, $fixture);
        $crate::write_once_space_integration_test!(test_different_clients, $suffix, $fixture);
    };
}
