// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! A counter wrapper that automatically records timestamps and results for linearizability checking.

use std::rc::Rc;

use agentbus_api::Clock;
use agentbus_api::environment::Environment;

use super::counter_trait::CommandResult;
use super::counter_trait::Counter;
use super::linearizability_tracker::ExecutedCommand;
use super::linearizability_tracker::LinearizabilityTracker;
use super::linearizability_tracker::OperationTimestamp;

pub struct TrackingCounter<C: Counter, E: Environment> {
    inner: C,
    env: Rc<E>,
    tracker: Rc<LinearizabilityTracker<i64>>,
}

impl<C: Counter, E: Environment> TrackingCounter<C, E> {
    pub fn new(inner: C, env: Rc<E>, tracker: Rc<LinearizabilityTracker<i64>>) -> Self {
        Self {
            inner,
            env,
            tracker,
        }
    }

    fn record(&self, start_time: std::time::Instant, result: &Result<CommandResult, String>) {
        let end_time = self.env.with_clock(|clock| clock.current_time());
        if let Ok(cmd_result) = result {
            self.tracker
                .record_result(cmd_result.intention_id, cmd_result.value);
            self.tracker.record_timestamp(OperationTimestamp {
                intention_id: cmd_result.intention_id,
                start_time,
                end_time,
            });
        }
        //TODO: record and track errors as well
    }
}

impl<C: Counter, E: Environment> Counter for TrackingCounter<C, E> {
    async fn increment(&self) -> Result<CommandResult, String> {
        let start_time = self.env.with_clock(|clock| clock.current_time());
        let result = self.inner.increment().await;
        self.record(start_time, &result);
        result
    }

    async fn decrement(&self) -> Result<CommandResult, String> {
        let start_time = self.env.with_clock(|clock| clock.current_time());
        let result = self.inner.decrement().await;
        self.record(start_time, &result);
        result
    }

    async fn get_command_history(&self) -> Vec<ExecutedCommand> {
        self.inner.get_command_history().await
    }
}
