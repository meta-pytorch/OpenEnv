// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Validation utilities for AgentBus

use anyhow::Result;

/// Minimum length for a bus ID
pub const MIN_BUS_ID_LEN: usize = 1;

/// Maximum length for a bus ID
pub const MAX_BUS_ID_LEN: usize = 256;

/// Checks if a character is valid for a bus ID.
/// Valid characters are ASCII alphanumeric, hyphens, underscores, dots, and slashes.
pub fn is_valid_bus_id_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' || c == '/'
}

/// Validates a bus ID string
pub fn validate_bus_id(bus_id: &str) -> Result<()> {
    let len = bus_id.len();
    if len < MIN_BUS_ID_LEN || len > MAX_BUS_ID_LEN {
        return Err(anyhow::anyhow!(
            "Bus ID length must be between {} and {} characters, got {}",
            MIN_BUS_ID_LEN,
            MAX_BUS_ID_LEN,
            len
        ));
    }

    if !bus_id.chars().all(is_valid_bus_id_char) {
        return Err(anyhow::anyhow!(
            "Bus ID can only contain ASCII alphanumeric characters, hyphens, underscores, dots, and slashes"
        ));
    }

    Ok(())
}
