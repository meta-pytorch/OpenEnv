// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Test scenarios for WriteOnceSpace implementations

use agentbus_api::WriteOnceError;
use agentbus_api::WriteOnceSpace;
use anyhow::Result;
use bytes::Bytes;

use crate::write_once_space::fixtures::WriteOnceSpaceTestFixture;

const DEFAULT_SPACE: &str = "default";

/// Test basic write then read cycle.
pub async fn run_test_write_then_read<F: WriteOnceSpaceTestFixture>(fixture: &F) -> Result<()> {
    let mut client = fixture.create_impl();

    client.write(DEFAULT_SPACE, 0, Bytes::from("hello")).await?;
    let value = client.read(DEFAULT_SPACE, 0).await;
    assert_eq!(value, Some(Bytes::from("hello")));

    Ok(())
}

/// Test reading from a non-existent address returns None.
pub async fn run_test_read_nonexistent<F: WriteOnceSpaceTestFixture>(fixture: &F) -> Result<()> {
    let client = fixture.create_impl();

    let value = client.read(DEFAULT_SPACE, 999).await;
    assert_eq!(value, None);

    Ok(())
}

/// Test write-once semantics: first write succeeds, second fails with AddressAlreadyExists.
pub async fn run_test_write_once_semantics<F: WriteOnceSpaceTestFixture>(
    fixture: &F,
) -> Result<()> {
    let mut client = fixture.create_impl();

    client.write(DEFAULT_SPACE, 0, Bytes::from("first")).await?;

    let result = client.write(DEFAULT_SPACE, 0, Bytes::from("second")).await;
    assert!(matches!(
        result,
        Err(WriteOnceError::AddressAlreadyExists(0))
    ));

    let value = client.read(DEFAULT_SPACE, 0).await;
    assert_eq!(value, Some(Bytes::from("first")));

    Ok(())
}

/// Test that tail returns the first unwritten slot.
/// Skips gracefully if the backend does not implement tail yet.
pub async fn run_test_tail<F: WriteOnceSpaceTestFixture>(fixture: &F) -> Result<()> {
    let mut client = fixture.create_impl();

    let tail = match client.tail(DEFAULT_SPACE).await {
        Ok(t) => t,
        Err(WriteOnceError::NotImplemented) => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    assert_eq!(tail, 0);

    client.write(DEFAULT_SPACE, 0, Bytes::from("a")).await?;
    let tail = client.tail(DEFAULT_SPACE).await?;
    assert!(tail >= 1);

    client.write(DEFAULT_SPACE, 1, Bytes::from("b")).await?;
    let tail = client.tail(DEFAULT_SPACE).await?;
    assert!(tail >= 2);

    Ok(())
}

pub async fn run_test_different_clients<F: WriteOnceSpaceTestFixture>(fixture: &F) -> Result<()> {
    let mut client1 = fixture.create_impl();
    let mut client2 = fixture.create_impl();

    client1
        .write(DEFAULT_SPACE, 0, Bytes::from("val-0"))
        .await?;
    client2
        .write(DEFAULT_SPACE, 1, Bytes::from("val-1"))
        .await?;

    for client in [&mut client1, &mut client2] {
        assert_eq!(
            client.read(DEFAULT_SPACE, 0).await,
            Some(Bytes::from("val-0"))
        );
        assert_eq!(
            client.read(DEFAULT_SPACE, 1).await,
            Some(Bytes::from("val-1"))
        );
        for i in 0..2 {
            let result = client.write(DEFAULT_SPACE, i, Bytes::from("fail")).await;
            assert!(matches!(
                result,
                Err(WriteOnceError::AddressAlreadyExists(addr)) if addr == i
            ));
        }
    }

    Ok(())
}
