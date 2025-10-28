# Cognitive Manufacturing Environment

An OpenEnv-compliant environment for AI agents to control and optimize manufacturing operations with 30 specialized tools across quality management, inventory control, energy optimization, and predictive analytics.

**Author**: Mohammad Mowas | is.mohammad@hotmail.com

## Overview

The Cognitive Manufacturing Environment provides a comprehensive simulation of a production facility where AI agents must balance multiple objectives:
- **Safety** (highest priority): Prevent failures, overheating, excessive vibration
- **Throughput**: Maximize production output
- **Quality**: Minimize defects
- **Cost**: Optimize operational expenses
- **Sustainability**: Improve energy efficiency

## Features

- **30 specialized tools** for complete manufacturing control
- **Multi-machine production line** simulation (4-stage pipeline)
- **Database persistence** (PostgreSQL/SQLite support)
- **ML-powered analytics** (predictive maintenance, anomaly detection, quality prediction)
- **Reinforcement learning** optimization (Q-learning)
- **Quality management** with defect tracking
- **Inventory control** and material ordering
- **Energy monitoring** and power optimization
- **Scenario simulation** and schedule optimization

## Quick Start

### Installation

```bash
# Install OpenEnv
pip install -e .

# Install cognitive manufacturing dependencies
pip install sqlalchemy sentence-transformers pandas scikit-learn numpy scipy
```

### Basic Usage

```python
from envs.cognitive_manufacturing import CognitiveManufacturingEnvironment, ManufacturingAction

# Create environment with all features enabled
env = CognitiveManufacturingEnvironment(
    multi_machine=True,       # Enable 4-machine production line
    enable_database=True,     # Enable data persistence
    enable_ml=True,          # Enable ML analytics
)

# Reset environment
observation = env.reset()

# Take an action
action = ManufacturingAction(
    tool_name="ReadSensors",
    parameters={"machine_id": "M1"}
)
observation, reward, done, info = env.step(action)
```

### Server Mode

```bash
# Start HTTP server
python -m uvicorn envs.cognitive_manufacturing.server.app:app --host 127.0.0.1 --port 8001

# Server will be available at http://127.0.0.1:8001
```

### Docker Deployment

```bash
docker build -t cognitive-manufacturing:latest -f src/envs/cognitive_manufacturing/server/Dockerfile .
docker run -p 8001:8001 cognitive-manufacturing:latest
```

## Available Tools (30 total)

### Basic Control (5 tools)
1. **ReadSensors** - Get real-time sensor data
2. **CheckHealth** - Health diagnostics
3. **AdjustSpeed** - Control machine speed (0-100%)
4. **ScheduleMaintenance** - Schedule/perform maintenance
5. **SendAlert** - Send notifications

### Production Line (3 tools)
6. **GetLineStatus** - Production line status
7. **TransferMaterial** - Material transfer between machines
8. **OptimizeLineSpeed** - Multi-machine speed optimization

### Data Management (7 tools)
9. **SaveProductionData** - Save runs to database
10. **QueryProductionHistory** - Query historical data
11. **ExecuteSQL** - Custom SQL queries
12. **ExportToCSV** - Export data to CSV
13. **ImportFromCSV** - Import data from CSV
14. **SearchKnowledge** - Semantic search in knowledge base
15. **AddKnowledge** - Add documents to knowledge base

### ML Analytics (5 tools)
16. **PredictMaintenance** - Failure prediction with Random Forest
17. **DetectAnomaly** - Anomaly detection with Isolation Forest
18. **PredictQuality** - Quality prediction with Linear Regression
19. **OptimizeWithRL** - Reinforcement learning optimization (Q-learning)
20. **ForecastDemand** - Time series demand forecasting

### Advanced Management (10 tools)
21. **InspectProduct** - Detailed quality inspection (visual/dimensional/functional)
22. **RecordDefect** - Log and track defects with severity levels
23. **UpdateQCThresholds** - Adjust quality control parameters
24. **CheckInventory** - Inventory levels with reorder recommendations
25. **OrderMaterials** - Material ordering with priority
26. **UpdateStockLevels** - Update inventory (add/remove/set)
27. **MonitorEnergyUsage** - Track power consumption and costs
28. **SetPowerMode** - Adjust power modes (eco/normal/high_performance)
29. **SimulateScenario** - What-if scenario simulations
30. **OptimizeProductionSchedule** - Production schedule optimization

## Environment Modes

### Single Machine Mode
```python
env = CognitiveManufacturingEnvironment()
```
- 1 machine (M1)
- Basic production simulation

### Production Line Mode
```python
env = CognitiveManufacturingEnvironment(multi_machine=True)
```
- 4 machines (M1 → M2 → M3 → M4)
- Material flow through buffers
- Bottleneck detection and optimization

### Full Feature Mode
```python
env = CognitiveManufacturingEnvironment(
    multi_machine=True,
    enable_database=True,
    enable_ml=True
)
```
- All 30 tools available
- Database persistence
- ML-powered analytics

## Machine Simulation

The physics-based simulator models realistic machine behavior:
- **Temperature dynamics**: Heat generation from operation, cooling over time
- **Vibration**: Increases with speed and wear
- **Wear accumulation**: Gradual degradation during operation
- **Failure probability**: Based on temperature, vibration, and wear levels
- **Production output**: Speed-dependent with quality penalties
- **Energy consumption**: Power usage based on speed and operating mode

## Reward System

Multi-objective reward calculation:
- **Safety**: Penalty for dangerous conditions (overheating, high vibration)
- **Throughput**: Reward for production output
- **Quality**: Penalty for defects
- **Cost**: Penalty for operational expenses
- **Sustainability**: Reward for energy efficiency

## Database Schema

When `enable_database=True`, the environment persists:
- Production runs with metadata
- Sensor readings (temperature, vibration, speed, health)
- Machine events (speed changes, maintenance, alerts)
- Production units with quality scores
- Knowledge base with vector embeddings

Supports PostgreSQL and SQLite.

## ML Models

When `enable_ml=True`, the environment provides:
- **Random Forest** for predictive maintenance
- **Isolation Forest** for anomaly detection
- **Linear Regression** for quality prediction
- **Q-learning** for reinforcement learning optimization
- **Exponential Smoothing** for demand forecasting

Models train automatically from historical data when sufficient samples are available.

## Architecture

```
CognitiveManufacturingEnvironment
├── Simulator (physics-based machine dynamics)
├── Production Line (4-machine pipeline with buffers)
├── Database Manager (PostgreSQL/SQLite persistence)
├── ML Service (5 ML models)
├── Reward Calculator (multi-objective rewards)
└── Tools (30 specialized tools)
```

## Dependencies

- **Core**: OpenEnv framework
- **Database**: SQLAlchemy
- **ML**: scikit-learn, numpy, scipy
- **NLP**: sentence-transformers (for semantic search)
- **Data**: pandas (for CSV operations)

## Examples

See OpenEnv `examples/` directory for usage examples.

## License

MIT License - See LICENSE file for details.

## Author

**Mohammad Mowas**
Email: is.mohammad@hotmail.com

Developed as part of the OpenEnv framework for AI agent environments.
