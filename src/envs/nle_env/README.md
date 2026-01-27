# NetHack Learning Environment (NLE) for OpenEnv

A reinforcement learning environment based on NetHack 3.6.6, wrapped for the OpenEnv framework.

## Overview

NetHack is one of the oldest and most challenging roguelike games, featuring:
- **Procedurally generated dungeons** - Every episode is unique
- **Complex action space** - 113+ discrete actions (movement, combat, magic, inventory management)
- **Rich observation space** - 14+ observation types including dungeon map, stats, inventory, messages
- **Challenging gameplay** - One of the hardest RL benchmarks available
- **Deterministic (with seeding)** - Reproducible episodes for evaluation

This environment wraps the [NetHack Learning Environment (NLE)](https://github.com/facebookresearch/nle) project, which provides a Gym interface to NetHack.

## Quick Start

### Using Docker (Recommended)

```python
from envs.nle_env import NLEEnv, NLEAction

# Automatically start container and connect
env = NLEEnv.from_docker_image("nle-env:latest")

# Reset to start a new game
result = env.reset()
print(f"Episode started: {result.observation.message}")

# Take actions in the game
for step in range(100):
    # Action IDs: 0-112 (movement, commands, etc.)
    action = NLEAction(action_id=0)  # Move north
    result = env.step(action)

    print(f"Step {step}: Reward={result.reward}, Done={result.done}")

    if result.done:
        print("Episode ended!")
        break

env.close()
```

### Building the Docker Image

```bash
# Build from repository root (not from server directory)
cd /Users/sanyambhutani/GH/OpenEnv
docker build -f src/envs/nle_env/server/Dockerfile -t nle-env:latest .
```

**Note:** Building NLE from source can take 5-10 minutes as it compiles NetHack C code.

### Running the Server Locally

```bash
# Install NLE (requires cmake, build-essential)
pip install nle gym

# Run the server
python -m envs.nle_env.server.app

# Server will be available at http://localhost:8000
```

## Action Space

NLE uses a discrete action space with 113 actions:

| Action ID Range | Category | Examples |
|----------------|----------|----------|
| 0-7 | Cardinal movement | North, South, East, West |
| 8-15 | Diagonal movement | NE, SE, SW, NW |
| 16-20 | Stair navigation | Up, Down |
| 21-112 | Commands | Eat, Search, Apply, Quaff, Read, etc. |

Common actions:
```python
# Movement
NLEAction(action_id=0)   # Move north (k)
NLEAction(action_id=1)   # Move east (l)
NLEAction(action_id=2)   # Move south (j)
NLEAction(action_id=3)   # Move west (h)

# Interactions
NLEAction(action_id=37)  # Eat (e)
NLEAction(action_id=50)  # Search (s)
NLEAction(action_id=104) # Inventory (i)
NLEAction(action_id=86)  # Wait (.)
```

For a complete action mapping, see [NLE Actions Documentation](https://github.com/facebookresearch/nle/blob/main/nle/nethack/actions.py).

## Observation Space

NLE provides rich observations about the game state. With OpenEnv's beefy compute assumption, all observations are included by default:

### Core Observations
- **glyphs** `(21, 79)`: Symbolic dungeon map representation
- **blstats** `(26,)`: Bottom-line stats (HP, MaxHP, XP, Gold, etc.)
- **message** `(256,)`: Latest game message as byte array

### Visual Observations
- **chars** `(21, 79)`: ASCII character display
- **colors** `(21, 79)`: Color codes for display
- **specials** `(21, 79)`: Special attributes (bold, inverse, etc.)

### Inventory Observations
- **inv_glyphs** `(55,)`: Inventory item glyphs
- **inv_strs** `(55, 80)`: Inventory item descriptions
- **inv_letters** `(55,)`: Inventory item letters (a-z, A-Z)
- **inv_oclasses** `(55,)`: Inventory object classes

### Terminal Observations (for rendering)
- **tty_chars** `(24, 80)`: Full terminal character display
- **tty_colors** `(24, 80)`: Full terminal colors
- **tty_cursor** `(2,)`: Terminal cursor position [row, col]

### Extended Observations
- **screen_descriptions** `(21, 79, 80)`: Text descriptions of dungeon cells
- **program_state** `(6,)`: Internal program state
- **internal** `(9,)`: Internal game state
- **misc** `(4,)`: Miscellaneous info

All observations are serialized as nested lists (converted from numpy arrays) for JSON compatibility.

## Reward Structure

By default, NLE uses **score delta** as the reward:
```
reward = current_score - previous_score
```

Score increases by:
- Defeating monsters
- Collecting gold
- Advancing to deeper dungeon levels
- Finding items
- Gaining experience points

## Episode Termination

Episodes end when:
1. **Death** - Character dies (most common)
2. **Ascension** - Player completes the game (very rare!)
3. **Aborted** - Max episode steps reached (default: 5000)
4. **Task Successful** - For task-specific environments

Check the end status:
```python
result = env.step(action)
if result.done:
    state = env.state()
    print(f"End status: {state.end_status}")
    # Possible values: RUNNING, DEATH, TASK_SUCCESSFUL, ABORTED
```

## Configuration

Configure the environment via environment variables or Docker args:

```bash
# Task variant (default: score)
export NLE_TASK=score

# Character (role-race-gender-alignment)
export NLE_CHARACTER=mon-hum-neu-mal

# Max episode steps (default: 5000)
export NLE_MAX_STEPS=10000
```

### Character Options

Format: `role-race-gender-alignment`

**Roles:** Archaeologist (arc), Barbarian (bar), Caveman (cav), Healer (hea), Knight (kni), Monk (mon), Priest (pri), Ranger (ran), Rogue (rog), Samurai (sam), Tourist (tou), Valkyrie (val), Wizard (wiz)

**Races:** Human (hum), Dwarf (dwa), Elf (elf), Gnome (gno), Orc (orc)

**Genders:** Male (mal), Female (fem)

**Alignments:** Lawful (law), Neutral (neu), Chaotic (cha)

Example: `wiz-elf-fem-cha` = Female Elven Chaotic Wizard

## Example: Random Agent

```python
import random
from envs.nle_env import NLEEnv, NLEAction

env = NLEEnv.from_docker_image("nle-env:latest")

episodes = 10
for episode in range(episodes):
    result = env.reset()
    total_reward = 0

    while True:
        # Random action
        action = NLEAction(action_id=random.randint(0, 112))
        result = env.step(action)

        total_reward += result.reward or 0

        if result.done:
            state = env.state()
            print(f"Episode {episode}: Reward={total_reward:.1f}, "
                  f"Steps={state.step_count}, Status={state.end_status}")
            break

env.close()
```

## Example: Rendering Game State

```python
import numpy as np
from envs.nle_env import NLEEnv, NLEAction

env = NLEEnv.from_docker_image("nle-env:latest")
result = env.reset()

# Get terminal display
tty_chars = np.array(result.observation.tty_chars)
tty_colors = np.array(result.observation.tty_colors)

# Print ASCII display
for row in tty_chars:
    print(''.join(chr(c) for c in row))

# Get game message
message = bytes(result.observation.message)
print(f"Message: {message[:message.index(b'\\0')].decode('ascii')}")

# Get stats
blstats = result.observation.blstats
print(f"HP: {blstats[10]}/{blstats[11]}, Gold: {blstats[13]}, "
      f"XP Level: {blstats[18]}")

env.close()
```

## Performance Considerations

With **beefy compute** (64+ cores, 256GB+ RAM, 10Gbps network):
- Observation size: ~140KB per step (all observation types)
- Network overhead: Negligible (<1ms on fast network)
- Memory: ~200-500MB per container
- Throughput: 100+ parallel environments easily

**Optimizations are NOT needed** - just run it simple with JSON serialization!

## Task Variants (Future)

Current implementation: **NetHackScore** (maximize game score)

Planned task variants:
- **NetHackStaircase** - Reach the stairs down
- **NetHackOracle** - Find the Oracle
- **NetHackGold** - Collect gold
- **NetHackEat** - Maximize hunger satisfaction
- **NetHackScout** - Maximize exploration

## Troubleshooting

### Build Issues

If Docker build fails with cmake errors:
```bash
# Ensure cmake is recent enough (3.15+)
cmake --version
```

### Container Won't Start

Check logs:
```bash
docker logs <container-id>
```

Common issues:
- NLE compilation failed → Check cmake, build-essential installed
- Import errors → Check PYTHONPATH set correctly
- Port already in use → Use different port mapping

### Slow Performance

If you experience slowness even with beefy compute:
1. Check network latency: `ping <server-ip>`
2. Monitor CPU: NLE is CPU-intensive for dungeon generation
3. Check Docker resources: Ensure containers have sufficient CPU allocation

## References

- [NLE GitHub](https://github.com/facebookresearch/nle)
- [NLE Paper (NeurIPS 2020)](https://arxiv.org/abs/2006.13760)
- [NetHack Wiki](https://nethackwiki.com)
- [NetHack Official Site](https://nethack.org)

## Citation

If you use NLE in your research, please cite:

```bibtex
@inproceedings{kuettler2020nethack,
  title={The NetHack Learning Environment},
  author={K{\"u}ttler, Heinrich and Nardelli, Nantas and Miller, Alexander H and
          Raileanu, Roberta and Selvatici, Marco and Grefenstette, Edward and
          Rockt{\"a}schel, Tim},
  booktitle={Proceedings of NeurIPS},
  year={2020}
}
```
