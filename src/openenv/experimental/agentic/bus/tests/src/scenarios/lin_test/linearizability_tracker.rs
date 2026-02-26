// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Generic linearizability checker for testing concurrent data structures.
//!
//! This module provides a reusable linearizability tracker that can verify
//! that concurrent operations on any data structure (counter, queue, etc.)
//! respect linearizability: there exists a total order of operations that
//! is consistent with the partial order imposed by real-time constraints.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Debug;
use std::rc::Rc;
use std::time::Instant;

use anyhow::Result;
use anyhow::bail;

/// Timestamp recording for an operation's start and end times.
#[derive(Clone, Debug)]
pub struct OperationTimestamp {
    pub intention_id: i64,
    pub start_time: Instant,
    pub end_time: Instant,
}

/// A record of an executed command in the linearization order.
/// This is object-agnostic - it just stores the intention ID and operation string.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutedCommand {
    pub intention_id: i64,
    pub operation: String,
}

/// A generic linearizability tracker that records operation results and timestamps,
/// then verifies that a proposed total order is consistent with both the return
/// values and the partial order imposed by real-time constraints.
///
/// Type parameter V is the return value type of operations (e.g., i64 for a counter).
pub struct LinearizabilityTracker<V> {
    results: RefCell<HashMap<i64, V>>,
    timestamps: RefCell<Vec<OperationTimestamp>>,
}

impl<V: Clone + PartialEq + Debug> LinearizabilityTracker<V> {
    pub fn new() -> Rc<Self> {
        Rc::new(Self {
            results: RefCell::new(HashMap::new()),
            timestamps: RefCell::new(Vec::new()),
        })
    }

    /// Record the result of an operation.
    pub fn record_result(&self, intention_id: i64, value: V) {
        self.results.borrow_mut().insert(intention_id, value);
    }

    /// Record the timestamp of an operation.
    pub fn record_timestamp(&self, timestamp: OperationTimestamp) {
        self.timestamps.borrow_mut().push(timestamp);
    }

    /// Verify linearizability given the command histories from all workers.
    ///
    /// This method checks:
    /// 1. All workers observed the same total order of commands
    /// 2. The number of committed operations matches recorded results
    /// 3. Replaying operations in the total order produces the same return values
    /// 4. The total order respects the partial order (real-time constraints)
    ///
    /// The replay function should apply the given command to a sequential
    /// specification and return the expected result value.
    pub fn verify<F>(&self, worker_histories: &[Vec<ExecutedCommand>], mut replay: F) -> Result<()>
    where
        F: FnMut(&ExecutedCommand) -> V,
    {
        let command_history = worker_histories
            .first()
            .expect("Should have at least one worker");

        self.verify_consistent_histories(worker_histories, command_history)?;

        let recorded_results = self.results.borrow();
        self.verify_result_count(command_history, &recorded_results)?;
        self.verify_replay_values(command_history, &recorded_results, &mut replay)?;
        drop(recorded_results);

        self.verify_partial_order(command_history)?;

        Ok(())
    }

    fn verify_consistent_histories(
        &self,
        worker_histories: &[Vec<ExecutedCommand>],
        command_history: &[ExecutedCommand],
    ) -> Result<()> {
        for (idx, worker_history) in worker_histories.iter().enumerate() {
            if worker_history.len() != command_history.len() {
                bail!(
                    "Worker {} history length {} differs from expected length {}",
                    idx,
                    worker_history.len(),
                    command_history.len()
                );
            }
            for (cmd_idx, worker_cmd) in worker_history.iter().enumerate() {
                if worker_cmd != &command_history[cmd_idx] {
                    bail!(
                        "Worker {} command at index {} differs: {:?} vs {:?}",
                        idx,
                        cmd_idx,
                        worker_cmd,
                        command_history[cmd_idx]
                    );
                }
            }
        }
        Ok(())
    }

    fn verify_result_count(
        &self,
        command_history: &[ExecutedCommand],
        recorded_results: &HashMap<i64, V>,
    ) -> Result<()> {
        if command_history.len() != recorded_results.len() {
            bail!(
                "Command history length {} should equal recorded results length {}",
                command_history.len(),
                recorded_results.len()
            );
        }
        Ok(())
    }

    fn verify_replay_values<F>(
        &self,
        command_history: &[ExecutedCommand],
        recorded_results: &HashMap<i64, V>,
        replay: &mut F,
    ) -> Result<()>
    where
        F: FnMut(&ExecutedCommand) -> V,
    {
        for cmd in command_history {
            let replay_value = replay(cmd);
            if let Some(worker_value) = recorded_results.get(&cmd.intention_id) {
                if worker_value != &replay_value {
                    bail!(
                        "Worker returned {:?} for intention {}, but replay produced {:?}",
                        worker_value,
                        cmd.intention_id,
                        replay_value
                    );
                }
            }
        }
        Ok(())
    }

    fn verify_partial_order(&self, command_history: &[ExecutedCommand]) -> Result<()> {
        let timestamps = self.timestamps.borrow();
        let position_map: HashMap<i64, usize> = command_history
            .iter()
            .enumerate()
            .map(|(pos, cmd)| (cmd.intention_id, pos))
            .collect();

        for (i, ts_a) in timestamps.iter().enumerate() {
            for ts_b in timestamps.iter().skip(i + 1) {
                if ts_a.end_time < ts_b.start_time {
                    let pos_a = position_map
                        .get(&ts_a.intention_id)
                        .expect("A should be in history");
                    let pos_b = position_map
                        .get(&ts_b.intention_id)
                        .expect("B should be in history");
                    if pos_a >= pos_b {
                        bail!(
                            "Partial order violation: op {} (pos {}) ended before op {} (pos {}) started, but appears later",
                            ts_a.intention_id,
                            pos_a,
                            ts_b.intention_id,
                            pos_b
                        );
                    }
                }
                if ts_b.end_time < ts_a.start_time {
                    let pos_a = position_map
                        .get(&ts_a.intention_id)
                        .expect("A should be in history");
                    let pos_b = position_map
                        .get(&ts_b.intention_id)
                        .expect("B should be in history");
                    if pos_b >= pos_a {
                        bail!(
                            "Partial order violation: op {} (pos {}) ended before op {} (pos {}) started, but appears later",
                            ts_b.intention_id,
                            pos_b,
                            ts_a.intention_id,
                            pos_a
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmd(id: i64, op: &str) -> ExecutedCommand {
        ExecutedCommand {
            intention_id: id,
            operation: op.to_string(),
        }
    }

    fn ts(id: i64, start_ms: u64, end_ms: u64, base: Instant) -> OperationTimestamp {
        OperationTimestamp {
            intention_id: id,
            start_time: base + std::time::Duration::from_millis(start_ms),
            end_time: base + std::time::Duration::from_millis(end_ms),
        }
    }

    fn counter_replay() -> impl FnMut(&ExecutedCommand) -> i64 {
        let mut val = 0i64;
        move |cmd| {
            match cmd.operation.as_str() {
                "inc" => val += 1,
                "dec" => val -= 1,
                _ => {}
            }
            val
        }
    }

    #[test]
    fn test_verify_succeeds_for_valid_history() {
        let tracker: Rc<LinearizabilityTracker<i64>> = LinearizabilityTracker::new();
        tracker.record_result(0, 1);
        tracker.record_result(1, 2);

        let history = vec![cmd(0, "inc"), cmd(1, "inc")];
        let result = tracker.verify(&[history.clone(), history], counter_replay());
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_fails_on_inconsistent_history_length() {
        let tracker: Rc<LinearizabilityTracker<i64>> = LinearizabilityTracker::new();
        tracker.record_result(0, 1);

        let h1 = vec![cmd(0, "inc")];
        let h2 = vec![cmd(0, "inc"), cmd(1, "inc")];
        let result = tracker.verify(&[h1, h2], counter_replay());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("length"));
    }

    #[test]
    fn test_verify_fails_on_inconsistent_history_order() {
        let tracker: Rc<LinearizabilityTracker<i64>> = LinearizabilityTracker::new();
        tracker.record_result(0, 1);
        tracker.record_result(1, 2);

        let h1 = vec![cmd(0, "inc"), cmd(1, "inc")];
        let h2 = vec![cmd(1, "inc"), cmd(0, "inc")];
        let result = tracker.verify(&[h1, h2], counter_replay());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("differs"));
    }

    #[test]
    fn test_verify_fails_on_result_count_mismatch() {
        let tracker: Rc<LinearizabilityTracker<i64>> = LinearizabilityTracker::new();
        tracker.record_result(0, 1);
        // Missing result for intention 1

        let history = vec![cmd(0, "inc"), cmd(1, "inc")];
        let result = tracker.verify(&[history], counter_replay());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("length"));
    }

    #[test]
    fn test_verify_fails_on_replay_value_mismatch() {
        let tracker: Rc<LinearizabilityTracker<i64>> = LinearizabilityTracker::new();
        tracker.record_result(0, 1);
        tracker.record_result(1, 999); // Wrong value - should be 2

        let history = vec![cmd(0, "inc"), cmd(1, "inc")];
        let result = tracker.verify(&[history], counter_replay());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("replay"));
    }

    #[test]
    fn test_verify_fails_on_partial_order_violation() {
        let tracker: Rc<LinearizabilityTracker<i64>> = LinearizabilityTracker::new();
        // Record values that match replay order [cmd(1), cmd(0)]
        tracker.record_result(1, 1);
        tracker.record_result(0, 2);

        let base = Instant::now();
        // Op 0 ends at 100ms, op 1 starts at 200ms => op 0 must come before op 1
        tracker.record_timestamp(ts(0, 0, 100, base));
        tracker.record_timestamp(ts(1, 200, 300, base));

        // History has op 1 before op 0 - violates partial order (op 0 finished first)
        let history = vec![cmd(1, "inc"), cmd(0, "inc")];
        let result = tracker.verify(&[history], counter_replay());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Partial order"));
    }

    #[test]
    fn test_verify_allows_concurrent_ops_in_any_order() {
        let base = Instant::now();
        // Overlapping operations - no ordering constraint
        let timestamps = [ts(0, 0, 150, base), ts(1, 100, 200, base)];

        // Order [0, 1]: replay produces 0→1, 1→2
        let tracker1: Rc<LinearizabilityTracker<i64>> = LinearizabilityTracker::new();
        tracker1.record_result(0, 1);
        tracker1.record_result(1, 2);
        for t in &timestamps {
            tracker1.record_timestamp(t.clone());
        }
        assert!(
            tracker1
                .verify(&[vec![cmd(0, "inc"), cmd(1, "inc")]], counter_replay())
                .is_ok()
        );

        // Order [1, 0]: replay produces 1→1, 0→2
        let tracker2: Rc<LinearizabilityTracker<i64>> = LinearizabilityTracker::new();
        tracker2.record_result(1, 1);
        tracker2.record_result(0, 2);
        for t in &timestamps {
            tracker2.record_timestamp(t.clone());
        }
        assert!(
            tracker2
                .verify(&[vec![cmd(1, "inc"), cmd(0, "inc")]], counter_replay())
                .is_ok()
        );
    }
}
