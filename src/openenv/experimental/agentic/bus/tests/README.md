# AgentBus Test Suite

Tests for AgentBus implementations using both deterministic simulation and live gRPC integration.

## Structure

```
tests/
├── common/
│   ├── fixtures.rs        # Core AgentBusTestFixture trait
│   ├── helpers.rs         # Test helper functions
│   ├── mod.rs             # Module declarations
│   └── test_scenarios.rs  # Shared test logic (runs on both Rc and Arc)
├── generic_simtests/      # Simulator-based tests (Rc)
│   ├── fixtures.rs        # Implementation fixtures and simulator setup
│   ├── macros.rs          # Test generation macro
│   ├── mod.rs             # Module declarations
│   ├── multi_node.rs      # Multi-node test scenarios
│   └── sim_only_scenarios.rs # Simulator-only test scenarios
├── integration_tests/     # gRPC server tests (Arc)
│   ├── fixtures.rs        # Integration test fixture
│   └── single_node.rs     # Single-node tests using shared scenarios
├── simtests.rs            # Simulator test entry
└── integration_tests.rs   # Integration test entry
```

## Test Sharing

Single-node test scenarios in `common/test_scenarios.rs` run against both:
- **Simulator**: `Rc<impl AgentBus>` via deterministic `Simulator` environment
- **Integration**: `Arc<AgentBusClient>` via live gRPC server over localhost

The same test logic validates both code paths using generic `Deref` bounds.

## Adding a New Implementation

### 1. Create Fixture

In `generic_simtests/fixtures.rs`, implement `AgentBusTestFixture` (defined in `common/fixtures.rs`) and `SimulatorFixture` (defined in `generic_simtests/fixtures.rs`):

```rust
pub struct MyFixture {
    state: Rc<RefCell<MyAgentBusState>>,
    env: Rc<Simulator>,
}

impl AgentBusTestFixture for MyFixture {
    type Env = Simulator;
    type AgentBusImpl = MyAgentBusImpl<Simulator>;
    type Impl = Rc<MyAgentBusImpl<Simulator>>;

    fn get_env(&self) -> Rc<Self::Env> {
        self.env.clone()
    }

    fn create_impl(&self) -> Self::Impl {
        Rc::new(MyAgentBusImpl::new(self.state.clone(), self.env.clone()))
    }
}

impl SimulatorFixture for MyFixture {
    fn new(simulator: Simulator) -> Self {
        Self {
            state: Rc::new(RefCell::new(MyAgentBusState::new())),
            env: Rc::new(simulator),
        }
    }
}
```

**Key contract**: Multiple `create_impl()` calls create instances sharing a single logical service.

### 2. Add Macro Invocation

In `simtests.rs`:

```rust
agent_bus_tests!(MyFixture, my_fixture);
```

### 3. Build and Run

```bash
buck test //agentbus:simtests
```

## Adding New Single-Node Tests

### For tests using standard setup

1. Add test function to `common/test_scenarios.rs`:
```rust
pub async fn run_test_new_feature<
    T: AgentBus,
    R: std::ops::Deref<Target = T> + Clone,
    E: agent_bus_environment::Environment,
>(
    impl_instance: R,
    environment: std::rc::Rc<E>,
) {
    // Test logic using impl_instance.deref() and environment.with_rng(...)
}
```

2. Add a `test_scenario!` macro call in `generic_simtests/macros.rs` (in the `agent_bus_tests!` macro):
```rust
// In agent_bus_tests! macro, add:
$crate::test_scenario!($fixture, $prefix, new_feature);
```

This automatically generates `test_{prefix}_single_node_new_feature` that calls `run_test_new_feature`.

3. Call from `integration_tests/single_node.rs` using the macro:
```rust
integration_test!(test_new_feature, test_scenarios::run_test_new_feature);
```

### For tests requiring custom setup

If your test needs custom setup (e.g., `setup_simulator_test_without_jitter`), add it manually to the `agent_bus_tests!` macro:

```rust
paste::paste! {
    #[tokio::test]
    async fn [<test_ $prefix _single_node_new_custom_feature>]() {
        let (impl_instance, env_rc) = $crate::generic_simtests::fixtures::setup_custom::<$fixture>();
        $crate::common::test_scenarios::run_test_new_custom_feature(impl_instance, env_rc).await;
    }
}
```
