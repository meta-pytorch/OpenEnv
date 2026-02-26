// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

use rand::SeedableRng;
use rand::rngs::StdRng;

/// Minimal clock abstraction for time sampling
pub trait Clock {
    /// Get the current time as an Instant
    fn current_time(&self) -> std::time::Instant;
}

/// Real-time clock that uses system time
pub struct RealClock;

impl RealClock {
    pub fn new() -> Self {
        Self
    }
}

impl Clock for RealClock {
    fn current_time(&self) -> std::time::Instant {
        std::time::Instant::now()
    }
}

/// Environment trait that allows the same code to run in production or simulation tests
/// Provides access to clock and RNG for deterministic testing
pub trait Environment {
    type Clock: Clock;

    fn with_rng<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut StdRng) -> R;

    fn with_clock<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Self::Clock) -> R;

    fn sleep(
        &self,
        duration: std::time::Duration,
    ) -> impl std::future::Future<Output = ()> + 'static;
}

/// Production environment using real clock and entropy-seeded RNG
pub struct RealEnvironment {
    rng: std::cell::RefCell<StdRng>,
    clock: RealClock,
}

impl RealEnvironment {
    pub fn new() -> Self {
        Self {
            rng: std::cell::RefCell::new(StdRng::from_entropy()),
            clock: RealClock::new(),
        }
    }
}

impl Environment for RealEnvironment {
    type Clock = RealClock;

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

    fn sleep(
        &self,
        duration: std::time::Duration,
    ) -> impl std::future::Future<Output = ()> + 'static {
        tokio::time::sleep(duration)
    }
}
