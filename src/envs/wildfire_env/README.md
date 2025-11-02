# 🌲 Wildfire Environment

Autonomous wildfire-control simulation for reinforcement-learning agents, built on the [OpenEnv](https://github.com/openenv) framework.  
Agents must contain spreading fires using **water**, **firebreaks**, and **timing strategies** under changing **wind** and **humidity** conditions.

[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://hub.docker.com/)
[![Python](https://img.shields.io/badge/python-3.10+-green)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/backend-fastapi-teal)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---
## 🔥 Why Wildfire Simulation?

Wildland fires are intensifying globally due to climate change — increasing the urgency for **AI-assisted decision-making**.  
This environment explores how intelligent systems can **control** fire spread in real time, under limited resources.

### Research Motivation
✅ Based on real wildfire science inspired by:
- **Rothermel Surface Fire Spread Model** (USDA Forest Service)
- **MITRE Fireline’s SimFire** — physics-informed RL fire simulator
- **SimHarness** — RL evaluation for disaster response

### Application Goals
| Research Theme | Role in This Environment |
|---|---|
| Resource-Constrained Planning | Finite water + firebreak budgets |
| Fire Spread + Containment Strategy | Directional wind & moisture effects |
| Disaster Response RL | Safety-focused reward design |
| LLM Agents for Control Tasks | Text-based action decision making |

This makes WildfireEnv a **fast, controllable**, and **open benchmark** for applied RL and LLM reasoning.

---

## 🔥 Environment Overview

This environment models **forest-fire dynamics** influenced by:
- **Wind direction** (8 directions + calm)
- **Humidity** (suppresses ignition)
- **Fuel type and spread rate**
- **Limited resources** (water units, break materials)
- **Time pressure** (each step costs reward)

The goal is to **minimize fire spread** and **total burned area** while using resources efficiently.

---

## 🧱 Grid Encoding

| Code | Meaning        | Color (Visualization) |
|------|----------------|-----------------------|
| 0    | Ash (burned)   | Black ⚫              |
| 1    | Fuel           | Green 🟩              |
| 2    | Burning        | Red 🔥                |
| 3    | Firebreak      | Brown 🟫              |
| 4    | Water/Damp     | Blue 🔵               |

---

## ⚙️ Architecture

```
┌────────────────────────────────────────────┐
│ RL Agent / LLM Trainer (Client)            │
│   wildfire_env.step(WildfireAction(...))   │
└──────────────────┬─────────────────────────┘
                   │ HTTP
┌──────────────────▼─────────────────────────┐
│ FastAPI Server (Docker)                    │
│   WildfireEnvironment                      │
│     ├─ Handles wind, humidity, spread      │
│     ├─ Applies agent actions               │
│     ├─ Updates grid + reward shaping       │
│     └─ Returns WildfireObservation         │
└────────────────────────────────────────────┘
```

---

## 🚀 Installation & Usage

### Option 1: Local Development (no Docker)

**Requirements:**
- Python 3.10 +
- FastAPI + Uvicorn
- NumPy + Matplotlib (for visualization)

```bash
pip install fastapi uvicorn numpy matplotlib requests
```

Run server locally:
```bash
python -m envs.wildfire_env.server.app
```

Client usage:
```python
from envs.wildfire_env import WildfireEnv, WildfireAction

env = WildfireEnv(base_url="http://localhost:8000")

result = env.reset()
print(f"🔥 Fires: {result.observation.burning_count}, 💧 Water left: {result.observation.remaining_water}")

for _ in range(5):
    result = env.step(WildfireAction(action="water", x=10, y=10))
    print(f"Reward: {result.reward}, Burning left: {result.observation.burning_count}")

env.close()
```

---

### Option 2: Docker (Recommended)

Build the image:
```bash
cd OpenEnv
docker build   -f src/envs/wildfire_env/server/Dockerfile   -t wildfire-env:latest .
```

Run the container:
```bash
docker run -p 8000:8000 wildfire-env:latest
```

Connect via client:
```python
from envs.wildfire_env import WildfireEnv, WildfireAction
env = WildfireEnv.from_docker_image("wildfire-env:latest")
result = env.reset()
print(f"Active fires: {result.observation.burning_count}")
result = env.step(WildfireAction(action="break", x=8, y=12))
print(f"Reward: {result.reward}")
env.close()
```

---

## 🌦️ Configuration

| Variable | Description | Default |
|-----------|--------------|----------|
| `WILDFIRE_WIDTH` | Grid width | 32 |
| `WILDFIRE_HEIGHT` | Grid height | 32 |
| `WILDFIRE_HUMIDITY` | Initial humidity [0–1] | 0.25 |
| `WILDFIRE_WIND` | Wind direction (`N`, `NE`, `E`, `SE`, `S`, `SW`, `W`, `NW`, `CALM`) | Random |
| `WILDFIRE_SEED` | RNG seed | 3407 |
| `WILDFIRE_MAX_STEPS` | Max steps per episode | 128 |
| `WILDFIRE_WATER_CAPACITY` | Water units available | 8 |
| `WILDFIRE_BREAK_CAPACITY` | Firebreak materials | 50 |

---

## 🧠 API Reference

### `WildfireAction`
```python
@dataclass
class WildfireAction(Action):
    action: str              # "water" | "break" | "wait"
    x: Optional[int] = None  # Target X
    y: Optional[int] = None  # Target Y
```

### `WildfireObservation`
```python
@dataclass
class WildfireObservation(Observation):
    grid: List[int]
    width: int
    height: int
    step: int
    wind_dir: str
    humidity: float
    burning_count: int
    burned_count: int
    remaining_water: int
    remaining_breaks: int
    reward_hint: float
```

### `WildfireState`
```python
@dataclass
class WildfireState(State):
    episode_id: str
    step_count: int
    total_burned: int
    total_extinguished: int
    remaining_water: int
    remaining_breaks: int
    wind_dir: str
    humidity: float
```

---
## Sample rendering to see wildfire simulation

**Note:** This example requires Jupyter notebook or IPython environment for the `clear_output` and `display` functions. For standalone Python scripts, see `examples/wildfire.py`.

```python
import matplotlib.pyplot as plt
import numpy as np
import time

from IPython.display import clear_output, display
import matplotlib.colors as mcolors
from envs.wildfire_env import WildfireEnv, WildfireAction

client = WildfireEnv("http://localhost:8000")


cmap = mcolors.ListedColormap([
    "black",         # 0 = ash
    "green",         # 1 = fuel
    "red",           # 2 = burning
    "saddlebrown",   # 3 = firebreak
    "blue"           # 4 = water
])

norm = mcolors.BoundaryNorm([0, 1, 2, 3, 4, 5], cmap.N)


plt.ion() 
fig, ax = plt.subplots(figsize=(5, 5))
plt.axis("off")


res = client.reset()
obs = res.observation
grid = np.array(obs.grid).reshape(obs.height, obs.width)


im = ax.imshow(grid, cmap=cmap, norm=norm)


title_text = ax.set_title(
    f"Step {obs.step} | Burning={obs.burning_count} | Burned={obs.burned_count}\n"
    f"Wind={obs.wind_dir} | Humidity={obs.humidity:.2f}",
    color="black", 
    fontsize=10
)



print("Starting smooth animation...")
for _ in range(100): 
    clear_output(wait=True) 

    new_grid = np.array(obs.grid).reshape(obs.height, obs.width)

    im.set_data(new_grid)

    title_text.set_text(
        f"Step {obs.step} | Burning={obs.burning_count} | Burned={obs.burned_count}\n"
        f"Wind={obs.wind_dir} | Humidity={obs.humidity:.2f}"
    )

    
    display(fig) 
    
  
    time.sleep(0.3) 

   
    res = client.step(WildfireAction(action="wait"))
    obs = res.observation

    if obs.burning_count == 0:
        print(f"🔥 Fire has fully burned out after {obs.step} steps.")
        break

plt.ioff() # Turn off interactive mode
plt.close(fig) # Close the figure at the end
print("Animation complete.")

```

===


## 🧪 Example Training Loop (GRPO/LLM)

```python
from envs.wildfire_env import WildfireEnv, WildfireAction
import random

env = WildfireEnv.from_docker_image("wildfire-env:latest")

for episode in range(3):
    result = env.reset()
    total_reward = 0

    while not result.done:
        a = random.choice(["water", "break", "wait"])
        x, y = random.randint(0, 15), random.randint(0, 15)
        result = env.step(WildfireAction(action=a, x=x, y=y))
        total_reward += result.reward or 0

    print(f"Episode {episode}: total_reward={total_reward:.2f}")

env.close()
```

---

## 🧰 DockerHub & GitHub Build

Build and push:

```bash
docker build -t openenv-base:latest -f src/core/containers/images/Dockerfile .
docker build -t ghcr.io/<your_username>/openenv-wildfire:latest -f src/envs/wildfire_env/server/Dockerfile .
docker push ghcr.io/<your_username>/openenv-wildfire:latest
```

GitHub Action matrix entry:
```yaml
strategy:
  matrix:
    image:
      - name: wildfire-env
        dockerfile: src/envs/wildfire_env/server/Dockerfile
```

---

## 🧭 References

- [OpenEnv Framework](https://github.com/openenv)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Reinforcement Learning Introduction](https://spinningup.openai.com/en/latest/)
- [Fire Spread Simulation Models (USFS Research)](https://www.fs.fed.us/rm/pubs/rmrs_gtr371.html)

---
## 🔖 Citations

```bibtex
@techreport{rothermel2022surface,
  title     = {The Rothermel Surface Fire Spread Model and Associated Developments},
  author    = {Andrews, Patricia L. and Rothermel, Richard C.},
  year      = {2022},
  institution = {USDA Forest Service},
  number    = {RMRS-GTR-371},
  url       = {https://www.fs.usda.gov/rm/pubs_series/rmrs/gtr/rmrs_gtr371.pdf}
}

@article{tapley2023reinforcement,
  title   = {Reinforcement Learning for Wildfire Mitigation in Simulated Disaster Environments},
  author  = {Tapley, A. and Dotter, M. and Doyle, M. and others},
  journal = {arXiv preprint arXiv:2311.15925},
  year    = {2023},
  url     = {https://arxiv.org/abs/2311.15925}
}

@misc{mitrefireline2023simfire,
  author = {{MITRE Fireline Project}},
  title  = {SimFire: Wildfire Simulator for Decision-Support and AI Research},
  year   = {2023},
  howpublished = {\url{https://github.com/mitrefireline/simfire}}
}

@misc{wildfire-openenv-2025,
  title  = {Wildfire Environment for OpenEnv: Containment-Focused RL Simulation},
  author = {Harikrishnan, Ram Sankar},
  year   = {2025},
  url    = {https://github.com/<your_username>/openenv-wildfire}
}
```
