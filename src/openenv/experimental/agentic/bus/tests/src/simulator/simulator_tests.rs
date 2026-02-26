// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

use agentbus_api::environment::Clock;
use agentbus_api::environment::Environment;
use futures::StreamExt;
use rand::Rng;
use rand::SeedableRng;
use rand::distributions::Uniform;
use rand::rngs::StdRng;

// Import from the same crate since this is test_srcs
use crate::simulator::Simulator;

/// Test that sleep works through the Environment trait
#[test]
fn test_environment_trait_sleep() {
    fn test_with_env<E: Environment>(
        env: &mut E,
        duration: Duration,
    ) -> impl std::future::Future<Output = std::time::Instant> + 'static + use<'_, E> {
        let start_time = env.with_clock(|clock| clock.current_time());
        let sleep_fut = env.sleep(duration);
        async move {
            sleep_fut.await;
            start_time
        }
    }

    let seed: u64 = rand::thread_rng().r#gen();
    let mut env = Simulator::with_jitter(seed, Uniform::new(1, 101));
    let sleep_duration = Duration::from_millis(50);

    let fut = test_with_env(&mut env, sleep_duration);
    let handle = env.spawn(fut);

    env.run();

    let start_time = futures::executor::block_on(handle).expect("Task should complete");
    let end_time = env.with_clock(|clock| clock.current_time());
    let elapsed = end_time - start_time;

    assert!(
        elapsed >= sleep_duration,
        "Sleep should advance time by at least the sleep duration"
    );
}

/// Test multiple simultaneous sleeps for correctness and determinism
#[test]
fn test_multiple_simultaneous_sleeps() {
    fn run_scenario(seed: u64) -> (Duration, Vec<usize>) {
        let env = Simulator::with_jitter(seed, Uniform::new(100, 200));
        let execution_order = Rc::new(RefCell::new(Vec::new()));

        let ms = env.rng.borrow_mut().gen_range(1..=100);
        let sleep_duration = Duration::from_millis(ms);
        let start_time = env.clock.current_time();

        let mut handles = Vec::new();

        for task_id in 0..3 {
            let order_clone = execution_order.clone();
            handles.push(env.spawn(async move {
                order_clone.borrow_mut().push(task_id);
            }));
        }

        for task_id in 3..8 {
            let order_clone = execution_order.clone();
            let sleep_fut = env.sleep(sleep_duration);
            handles.push(env.spawn(async move {
                sleep_fut.await;
                order_clone.borrow_mut().push(task_id);
            }));
        }

        for task_id in 8..10 {
            let order_clone = execution_order.clone();
            handles.push(env.spawn(async move {
                order_clone.borrow_mut().push(task_id);
            }));
        }

        env.run();

        for handle in handles {
            futures::executor::block_on(handle).expect("Task should complete");
        }

        let total_elapsed = env.clock.current_time() - start_time;
        let order = execution_order.borrow().clone();
        (total_elapsed, order)
    }

    let seed: u64 = rand::thread_rng().r#gen();

    let (elapsed1, order1) = run_scenario(seed);
    let (elapsed2, order2) = run_scenario(seed);

    assert_eq!(
        order1, order2,
        "Execution order should be deterministic with same seed"
    );
    assert_eq!(
        elapsed1, elapsed2,
        "Clock advancement should be deterministic with same seed"
    );
}

/// Test multiple sleeps with different durations for correctness and determinism
#[test]
fn test_staggered_sleeps() {
    fn run_scenario(seed: u64) -> (Duration, Duration, Vec<usize>) {
        let env = Simulator::with_jitter(seed, Uniform::new(1, 101));
        let start_time = env.clock.current_time();
        let execution_order = Rc::new(RefCell::new(Vec::new()));

        let mut sleep_durations = Vec::new();
        for _ in 0..5 {
            let ms = env.rng.borrow_mut().gen_range(1..=100);
            sleep_durations.push(Duration::from_millis(ms));
        }
        let max_duration = *sleep_durations.iter().max().unwrap();

        let mut handles = Vec::new();

        for task_id in 0..2 {
            let order_clone = execution_order.clone();
            handles.push(env.spawn(async move {
                order_clone.borrow_mut().push(task_id);
            }));
        }

        for (idx, duration) in sleep_durations.iter().enumerate() {
            let task_id = idx + 2;
            let order_clone = execution_order.clone();
            let sleep_fut = env.sleep(*duration);
            handles.push(env.spawn(async move {
                sleep_fut.await;
                order_clone.borrow_mut().push(task_id);
            }));
        }

        let order_clone = execution_order.clone();
        handles.push(env.spawn(async move {
            order_clone.borrow_mut().push(7);
        }));

        env.run();

        for handle in handles {
            futures::executor::block_on(handle).expect("Task should complete");
        }

        let total_elapsed = env.clock.current_time() - start_time;
        let order = execution_order.borrow().clone();
        (total_elapsed, max_duration, order)
    }

    let seed: u64 = rand::thread_rng().r#gen();

    let (elapsed1, max_duration1, order1) = run_scenario(seed);
    let (elapsed2, max_duration2, order2) = run_scenario(seed);

    assert!(
        elapsed1 >= max_duration1,
        "Clock should advance to at least the maximum sleep duration"
    );
    assert_eq!(
        order1, order2,
        "Execution order should be deterministic with same seed"
    );
    assert_eq!(
        elapsed1, elapsed2,
        "Clock advancement should be deterministic with same seed"
    );
    assert_eq!(
        max_duration1, max_duration2,
        "RNG-generated sleep durations should be deterministic"
    );
}

/// Test that the simulator executes deterministically with the same seed
#[test]
fn test_simulator_determinism() {
    fn run_scenario(seed: u64) -> (Vec<i32>, Vec<i32>) {
        let env = Simulator::with_jitter(seed, Uniform::new(0, 501));
        let shared_results = Rc::new(RefCell::new(Vec::new()));
        let mut handles = Vec::new();

        let num_tasks = env.with_rng(|rng| rng.r#gen_range(1..=10));

        // Spawn tasks that use RNG
        for _ in 0..num_tasks {
            let seed: u64 = env.with_rng(|rng| rng.r#gen());
            let results_clone = shared_results.clone();
            let handle = env.spawn(async move {
                let mut task_rng = StdRng::seed_from_u64(seed);
                let value = task_rng.gen_range(1..=100);
                results_clone.borrow_mut().push(value);
                value // Return the value
            });
            handles.push(handle);
        }

        env.run();

        // Collect results from handles
        let mut handle_results = Vec::new();
        for handle in handles {
            handle_results.push(futures::executor::block_on(handle).expect("Task should complete"));
        }

        // Get results from shared state
        let shared_vec = shared_results.borrow().clone();

        // Within this run, verify that handle results and shared results contain the same values
        let mut handle_sorted = handle_results.clone();
        let mut shared_sorted = shared_vec.clone();
        handle_sorted.sort();
        shared_sorted.sort();
        assert_eq!(
            handle_sorted, shared_sorted,
            "Handle return values and shared state should contain the same elements"
        );

        (shared_vec, handle_results)
    }

    let seed: u64 = rand::thread_rng().r#gen();

    // Run twice with the same seed
    let (run1_shared, run1_handles) = run_scenario(seed);
    let (run2_shared, run2_handles) = run_scenario(seed);

    // Should produce identical results (determinism)
    assert_eq!(
        run1_shared, run2_shared,
        "Shared results should be deterministic"
    );
    assert_eq!(
        run1_handles, run2_handles,
        "Handle results should be deterministic"
    );
}

/// Test that many senders to one receiver completes efficiently within iteration budget.
#[test]
fn test_many_senders_one_receiver() {
    use futures::channel::oneshot;

    let seed: u64 = rand::thread_rng().r#gen();
    let env = Rc::new(Simulator::new(seed));
    let (tx, mut rx) = futures::channel::mpsc::unbounded();

    let n = 10000;

    // Spawn N senders, each waiting for an echo back
    for i in 0..n {
        let tx = tx.clone();
        env.spawn(async move {
            let (echo_tx, echo_rx) = oneshot::channel();
            let _ = tx.unbounded_send((i, echo_tx));
            // Wait for echo from receiver
            let echoed = echo_rx.await.expect("Should receive echo");
            assert_eq!(echoed, i, "Echo should match sent value");
        });
    }

    // Spawn one receiver that echoes messages back
    let received = Rc::new(RefCell::new(0));
    let received_clone = received.clone();
    let env_clone = env.clone();
    env.spawn(async move {
        while let Some((msg, echo_tx)) = rx.next().await {
            *received_clone.borrow_mut() += 1;
            let _ = echo_tx.send(msg);
            env_clone.sleep(Duration::from_millis(0)).await;
        }
    });

    let stats = env.run();

    assert_eq!(*received.borrow(), n, "Should receive all messages");
    assert_eq!(
        stats.final_blocked_tasks, 1,
        "receiver task should be blocked"
    );

    // Allow the receiver to complete
    drop(tx);
    // Wait for receiver to complete
    let stats = env.run();
    assert_eq!(stats.iterations, 1, "expected single iteration");
    assert_eq!(stats.final_blocked_tasks, 0, "all tasks should be done");
    assert_eq!(*received.borrow(), n, "no more messages should be received");
}

/// Test that small jitter with large sleep completes within iteration budget.
#[test]
fn test_small_jitter_large_sleep() {
    let seed: u64 = rand::thread_rng().r#gen();
    let env = Rc::new(Simulator::with_jitter(seed, Uniform::new(1, 2)));
    let start_time = env.clock.current_time();
    let sleep_duration = Duration::from_secs(10);

    let handle = env.spawn({
        let sleep_fut = env.sleep(sleep_duration);
        async move {
            sleep_fut.await;
        }
    });

    env.run();
    futures::executor::block_on(handle).expect("Sleep should complete");

    let elapsed = env.clock.current_time() - start_time;
    assert!(
        elapsed >= sleep_duration,
        "Sleep should complete and advance time by at least {:?}, but got {:?}",
        sleep_duration,
        elapsed
    );
}

/// Test that zero jitter with any sleep completes within iteration budget.
#[test]
fn test_zero_jitter_with_sleep() {
    let seed: u64 = rand::thread_rng().r#gen();
    let env = Rc::new(Simulator::with_jitter(seed, Uniform::new(0, 1)));
    let start_time = env.clock.current_time();
    let sleep_duration = Duration::from_millis(100);

    let handle = env.spawn({
        let sleep_fut = env.sleep(sleep_duration);
        async move {
            sleep_fut.await;
        }
    });

    env.run();
    futures::executor::block_on(handle).expect("Sleep should complete");

    let elapsed = env.clock.current_time() - start_time;
    assert_eq!(
        elapsed, sleep_duration,
        "Sleep should complete even with zero jitter and advance time by {:?}, but got {:?}",
        sleep_duration, elapsed
    );
}

/// Test that waker-based execution achieves minimal iteration count.
#[test]
fn test_minimal_iterations() {
    use futures::channel::oneshot;

    for _ in 0..100 {
        let seed: u64 = rand::thread_rng().r#gen();
        let env = Rc::new(Simulator::with_jitter(seed, Uniform::new(0, 1)));

        let (tx0, rx0) = oneshot::channel();
        let (tx1, rx1) = oneshot::channel();
        let (tx2, rx2) = oneshot::channel();

        // Node 0
        env.spawn(async move {
            if let Ok(msg) = rx0.await {
                let _ = tx1.send(msg + 1);
            }
        });

        // Node 1
        env.spawn(async move {
            if let Ok(msg) = rx1.await {
                let _ = tx2.send(msg + 1);
            }
        });

        // Node 2
        let final_val = Rc::new(RefCell::new(0));
        let final_val_clone = final_val.clone();
        env.spawn(async move {
            if let Ok(msg) = rx2.await {
                *final_val_clone.borrow_mut() = msg;
            }
        });

        // Sender: sleep for 1 second, then send
        let env_clone = env.clone();
        env.spawn(async move {
            env_clone.sleep(Duration::from_secs(1)).await;
            let _ = tx0.send(0);
        });

        let stats = env.run();

        assert_eq!(*final_val.borrow(), 2);
        assert_eq!(stats.final_blocked_tasks, 0);
        assert_eq!(stats.iterations, 9); // 2 polls per rx and 3 for the sender and its sleep.
    }
}
