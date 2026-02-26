// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

//! Worker that performs counter operations

use std::rc::Rc;

use agentbus_api::environment::Environment;
use rand::Rng;

use super::counter_trait::Counter;
use super::linearizability_tracker::ExecutedCommand;

pub struct CounterWorker<C: Counter> {
    counter: C,
    operations: Vec<bool>,
}

impl<C: Counter> CounterWorker<C> {
    pub fn new<E: Environment>(counter: C, env: Rc<E>, num_ops: usize) -> Self {
        let operations: Vec<bool> = (0..num_ops)
            .map(|_| env.with_rng(|rng| rng.gen_bool(0.5)))
            .collect();
        Self {
            counter,
            operations,
        }
    }

    pub async fn run_workload(&self) {
        for &do_increment in &self.operations {
            let _ = if do_increment {
                self.counter.increment().await
            } else {
                self.counter.decrement().await
            };
        }
    }

    pub async fn get_command_history(&self) -> Vec<ExecutedCommand> {
        self.counter.get_command_history().await
    }
}
