// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Write-Once Address Space API
//!
//! Defines a trait for an address space where each address can only be written once.
//! Subsequent writes to the same address return an error.

use bytes::Bytes;
use thiserror::Error;

/// Error type for write-once address space operations
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum WriteOnceError {
    #[error("address already exists: {0}")]
    AddressAlreadyExists(u64),
    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),
    #[error("not implemented")]
    NotImplemented,
}

/// Result type alias for write-once operations
pub type WriteOnceResult<T> = std::result::Result<T, WriteOnceError>;

/// A write-once address space where each address can only be written to once.
///
/// Implementations provide storage where:
/// - `write` succeeds only if the address is empty
/// - `write` fails with `AddressAlreadyExists` if the address has a value
/// - `read` returns the value at an address, or None if not present
///
/// The `space_id` parameter allows a single implementation to handle multiple
/// logical spaces. Implementations route operations based on space_id.
pub trait WriteOnceSpace {
    /// Write a value to the given address within a space.
    ///
    /// Returns `Ok(())` if the write succeeds (address was empty).
    /// Returns `Err(WriteOnceError::AddressAlreadyExists)` if the address already has a value.
    fn write(
        &mut self,
        space_id: &str,
        address: u64,
        value: Bytes,
    ) -> impl std::future::Future<Output = WriteOnceResult<()>>;

    /// Read the value at the given address within a space.
    ///
    /// Returns `Some(value)` if the address has a value.
    /// Returns `None` if the address is empty.
    fn read(
        &self,
        space_id: &str,
        address: u64,
    ) -> impl std::future::Future<Output = Option<Bytes>>;

    /// Returns the first unwritten slot in the space.
    ///
    /// This is not strongly consistent / linearizable; it only has to correspond
    /// to an unordered, non-atomic scan of the space that occurs within the
    /// linearization span.
    fn tail(&self, space_id: &str) -> impl std::future::Future<Output = WriteOnceResult<u64>>;
}
