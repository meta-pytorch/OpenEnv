// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;
use std::task::Context;
use std::task::Wake;
use std::task::Waker;

// Import the Environment trait from environment module
use agentbus_api::environment::{Clock, Environment};
use futures::channel::oneshot;
use rand::Rng;
use rand::SeedableRng;
use rand::distributions::Distribution;
use rand::distributions::Uniform;
use rand::rngs::StdRng;

/// Metrics collected during simulator execution
#[derive(Debug, Clone)]
pub struct RunStats {
    pub iterations: usize,
    pub final_blocked_tasks: usize,
}

/// Handle returned by Simulator::spawn - represents a spawned deterministic task
pub struct SimulatorHandle<T> {
    receiver: oneshot::Receiver<T>,
}

impl<T> SimulatorHandle<T> {
    fn new(receiver: oneshot::Receiver<T>) -> Self {
        Self { receiver }
    }
}

impl<T> Future for SimulatorHandle<T> {
    type Output = Result<T, oneshot::Canceled>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> std::task::Poll<Self::Output> {
        Pin::new(&mut self.receiver).poll(cx)
    }
}

/// MB: Simulator supports spawning of tasks on a deterministic, single-threaded executor.
/// Tasks are spawned with a logical timestamp, and are executed in deterministic order.
/// To ensure determinism, it uses a logical clock to advance time, and a RNG with a specific seed.
/// Note: We have a level of indirection via oneshot channel to the executor since it's difficult
/// storing futures that return different types in a single container.
pub struct Simulator {
    pub executor: DeterministicExecutor,
    pub rng: RefCell<StdRng>,
    pub clock: LogicalClock,
}

impl Simulator {
    pub fn new(seed: u64) -> Self {
        eprintln!("Simulator seed: {}", seed);
        let rng = StdRng::seed_from_u64(seed);
        let clock = LogicalClock::new();
        Self {
            executor: DeterministicExecutor::new(),
            rng: RefCell::new(rng),
            clock,
        }
    }

    pub fn with_jitter(seed: u64, jitter_distribution: Uniform<u64>) -> Self {
        eprintln!("Simulator seed: {}", seed);
        let rng = StdRng::seed_from_u64(seed);
        let clock = LogicalClock::new();
        Self {
            executor: DeterministicExecutor::with_jitter(jitter_distribution),
            rng: RefCell::new(rng),
            clock,
        }
    }

    /// Spawn a task in the simulator
    pub fn spawn<F, T>(&self, fut: F) -> SimulatorHandle<T>
    where
        F: Future<Output = T> + 'static,
        T: 'static,
    {
        self.spawn_named(fut, None)
    }

    /// Spawn a task in the simulator with an optional name for debugging
    pub fn spawn_named<F, T>(&self, fut: F, name: Option<String>) -> SimulatorHandle<T>
    where
        F: Future<Output = T> + 'static,
        T: 'static,
    {
        let (sender, receiver) = oneshot::channel();
        let wrapper_fut = async move {
            let result = fut.await;
            let _ = sender.send(result);
        };
        self.executor.spawn_named(wrapper_fut, name);
        SimulatorHandle::new(receiver)
    }

    /// Run all spawned tasks to completion in deterministic order
    pub fn run(&self) -> RunStats {
        let seed = self.rng.borrow_mut().r#gen();
        let mut child_rng = StdRng::seed_from_u64(seed);
        self.executor.run(&mut child_rng, &self.clock)
    }
}

impl Environment for Simulator {
    type Clock = LogicalClock;

    fn with_rng<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut StdRng) -> R,
    {
        f(&mut self.rng.borrow_mut())
    }

    fn with_clock<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Self::Clock) -> R,
    {
        f(&self.clock)
    }

    fn sleep(&self, duration: std::time::Duration) -> impl Future<Output = ()> + 'static {
        let (sender, receiver) = oneshot::channel();

        self.executor.spawn_after(
            async move {
                let _ = sender.send(());
            },
            duration,
            &self.clock,
        );

        async move {
            let _ = receiver.await;
        }
    }
}

// Task with logical timestamp for deterministic ordering
// Note: No Send bound needed since Simulator is single-threaded
struct Task {
    target_time: Option<std::time::Instant>,
    sequence: u64,
    future: Pin<Box<dyn Future<Output = ()> + 'static>>,
    #[allow(dead_code)]
    name: Option<String>,
}

impl Ord for Task {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.target_time
            .cmp(&other.target_time)
            .then_with(|| self.sequence.cmp(&other.sequence))
    }
}

impl PartialOrd for Task {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Task {
    fn eq(&self, other: &Self) -> bool {
        self.target_time == other.target_time && self.sequence == other.sequence
    }
}

impl Eq for Task {}

// Waker implementation for the simulator
struct SimulatorWaker {
    sequence: u64,
    wakes: Rc<RefCell<Vec<u64>>>,
    thread_id: std::thread::ThreadId,
}

// Safety: The simulator is single-threaded. Although Waker requires Send+Sync,
// this waker will only ever be accessed from the thread that created it. We
// verify this with a runtime check.
// See: https://github.com/rust-lang/libs-team/issues/191
unsafe impl Send for SimulatorWaker {}
unsafe impl Sync for SimulatorWaker {}

impl Wake for SimulatorWaker {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref()
    }

    fn wake_by_ref(self: &Arc<Self>) {
        // Verify we're on the same thread where the waker was created
        let current_thread = std::thread::current().id();
        if current_thread != self.thread_id {
            panic!(
                "SimulatorWaker used across threads! Created on {:?}, used on {:?}",
                self.thread_id, current_thread
            );
        }

        // Add task sequence to wake queue
        self.wakes.borrow_mut().push(self.sequence);
    }
}

// We maintain an unordered list of events; a sorted list of sleep events; and blocked tasks.
pub struct DeterministicExecutor {
    // Tasks ready to poll
    events: RefCell<Vec<Task>>,
    // Tasks waiting on a time point
    sleeps: RefCell<BinaryHeap<Reverse<Task>>>,
    // Blocked tasks
    blocked: RefCell<HashMap<u64, Task>>,
    // Id's of previously blocked tasks
    wakes: Rc<RefCell<Vec<u64>>>,
    // Jitter added to event timestamps (in microseconds). WARNING: If this is set to
    // Uniform::new(0, 1), events will always have timestamp equal to current time, which
    // means sleeps will never execute if any events exist, causing potential hangs.
    jitter_distribution: Uniform<u64>,
    // Sequence number for tasks, incremented on each spawn. Used to break ties when
    // multiple sleeps have the same target timestamp, ensuring deterministic ordering.
    next_sequence: RefCell<u64>,
}

impl DeterministicExecutor {
    pub fn new() -> Self {
        Self::with_jitter(Uniform::new(100, 501))
    }

    pub fn with_jitter(jitter_distribution: Uniform<u64>) -> Self {
        Self {
            events: RefCell::new(Vec::new()),
            sleeps: RefCell::new(BinaryHeap::new()),
            blocked: RefCell::new(HashMap::new()),
            wakes: Rc::new(RefCell::new(Vec::new())),
            jitter_distribution,
            next_sequence: RefCell::new(0),
        }
    }

    pub fn spawn<F>(&self, fut: F)
    where
        F: Future<Output = ()> + 'static,
    {
        self.spawn_named(fut, None);
    }

    pub fn spawn_named<F>(&self, fut: F, name: Option<String>)
    where
        F: Future<Output = ()> + 'static,
    {
        let sequence = *self.next_sequence.borrow();
        *self.next_sequence.borrow_mut() += 1;
        let task = Task {
            target_time: None,
            sequence,
            future: Box::pin(fut),
            name,
        };
        self.events.borrow_mut().push(task);
    }

    pub fn spawn_after<F>(&self, fut: F, duration: std::time::Duration, clock: &LogicalClock)
    where
        F: Future<Output = ()> + 'static,
    {
        let sequence = *self.next_sequence.borrow();
        *self.next_sequence.borrow_mut() += 1;
        let target_time = clock.current_time() + duration;
        let task = Task {
            target_time: Some(target_time),
            sequence,
            future: Box::pin(fut),
            name: None,
        };
        self.sleeps.borrow_mut().push(Reverse(task));
    }

    fn next_event_time(
        &self,
        rng: &mut StdRng,
        clock: &LogicalClock,
    ) -> Option<std::time::Instant> {
        if self.events.borrow().is_empty() {
            return None;
        }
        let jitter_us = self.jitter_distribution.sample(rng);
        Some(clock.current_time() + std::time::Duration::from_micros(jitter_us))
    }

    fn next_sleep_time(&self) -> Option<std::time::Instant> {
        self.sleeps
            .borrow()
            .peek()
            .and_then(|Reverse(sleep)| sleep.target_time)
    }

    pub fn run(&self, rng: &mut StdRng, clock: &LogicalClock) -> RunStats {
        const MAX_ITERATIONS: usize = 100_000;
        let mut iteration_count = 0;

        loop {
            if iteration_count >= MAX_ITERATIONS {
                panic!(
                    "Simulator exceeded {} iterations, likely infinite loop!",
                    MAX_ITERATIONS
                );
            }

            // Process wake queue: move woken tasks from blocked to events.
            let to_wake: Vec<u64> = self.wakes.borrow_mut().drain(..).collect();
            for seq in to_wake {
                if let Some(task) = self.blocked.borrow_mut().remove(&seq) {
                    self.events.borrow_mut().push(task);
                }
                // "As long as the executor keeps running and the task is not finished,
                // it is guaranteed that each invocation of wake() (or wake_by_ref())
                // will be followed by at least one poll()"
                // https://doc.rust-lang.org/std/task/struct.Waker.html#method.wake
                // Here: if the task is not finished and not in `blocked`, it must
                // be in `events`, so it will be polled again.
            }

            // Task selection protocol:
            // 1. Compute next event time (if events exist)
            // 2. Peek at next sleep time (if sleeps exist)
            // 3. Run whichever comes first
            // Note: If event_time == sleep_time, events win (handled by the _ pattern below).
            let next_event_time = self.next_event_time(rng, clock);
            let next_sleep_time = self.next_sleep_time();

            let (task_to_run, target_time) = match (next_event_time, next_sleep_time) {
                (None, None) => break,
                (Some(event_time), Some(sleep_time)) if sleep_time < event_time => {
                    let task = self.sleeps.borrow_mut().pop().map(|Reverse(t)| t);
                    (task, sleep_time)
                }
                (Some(event_time), _) => {
                    let event_index = rng.gen_range(0..self.events.borrow().len());
                    let task = self.events.borrow_mut().swap_remove(event_index);
                    (Some(task), event_time)
                }
                (None, Some(sleep_time)) => {
                    let task = self.sleeps.borrow_mut().pop().map(|Reverse(t)| t);
                    (task, sleep_time)
                }
            };

            if let Some(mut task) = task_to_run {
                // Advance clock to the target time
                clock.advance_to(target_time);

                let waker = Arc::new(SimulatorWaker {
                    sequence: task.sequence,
                    wakes: Rc::clone(&self.wakes),
                    thread_id: std::thread::current().id(),
                });
                let waker = Waker::from(waker);
                let mut cx = Context::from_waker(&waker);

                if task.future.as_mut().poll(&mut cx).is_pending() {
                    // Move task to blocked
                    self.blocked.borrow_mut().insert(task.sequence, task);
                }
            }

            iteration_count += 1;
        }

        RunStats {
            iterations: iteration_count,
            final_blocked_tasks: self.blocked.borrow().len(),
        }
    }
}

/// Logical clock for deterministic testing - time advances only when explicitly set
pub struct LogicalClock {
    current_time: RefCell<std::time::Instant>,
}

impl LogicalClock {
    pub fn new() -> Self {
        Self {
            current_time: RefCell::new(std::time::Instant::now()),
        }
    }

    /// Advance clock to the given time, but never go backwards
    pub fn advance_to(&self, target_time: std::time::Instant) {
        let mut current = self.current_time.borrow_mut();
        if target_time > *current {
            *current = target_time;
        }
    }
}

impl Clock for LogicalClock {
    fn current_time(&self) -> std::time::Instant {
        *self.current_time.borrow()
    }
}

#[cfg(test)]
mod simulator_tests;
