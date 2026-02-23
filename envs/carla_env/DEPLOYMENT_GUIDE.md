# CARLA Environment - Deployment Guide

Quick reference for deploying the CARLA environment.

## Deployment

The primary deployment is a **standalone CARLA 0.10.0 image** with full physics simulation.

### HuggingFace Spaces (Recommended)

```bash
openenv push envs/carla_env --repo-id username/carla-env
# Then configure GPU T4/A10G in Space settings
```

### Local Docker

```bash
docker build -t carla-env:latest -f server/Dockerfile .
docker run --gpus all -p 8000:8000 carla-env:latest
```

### Specifications

| | Value |
|---|---|
| **Dockerfile** | `server/Dockerfile` |
| **GPU** | NVIDIA T4 (minimum) or A10G (recommended) |
| **CARLA** | 0.10.0 + Unreal Engine 5.5, bundled |
| **Image size** | ~15GB |
| **Build time** | 30-60 minutes |
| **Startup time** | 60-90 seconds |
| **Memory** | ~8-12GB RAM |
| **VRAM** | 16GB+ |

### Configuration

```bash
CARLA_SCENARIO=trolley_saves  # Scenario name
CARLA_HOST=localhost           # CARLA server host
CARLA_PORT=2000                # CARLA server port
CARLA_MODE=real                # real (default in Docker) or mock (tests only)
```

## GPU Selection

### NVIDIA T4 (16GB VRAM) — Minimum
- $0.60/hour on HF Spaces
- Works for all scenarios
- May experience occasional OOM on complex scenes

### NVIDIA A10G (24GB VRAM) — Recommended
- $1.10/hour on HF Spaces
- Stable and performant
- Recommended for production deployments

## Rendering Modes

### RenderOffScreen (Default)

```bash
./CarlaUnreal.sh -RenderOffScreen -opengl -quality-level=Low -carla-rpc-port=2000 -fps=20
```

- GPU renders frames offscreen (no display needed)
- Supports `capture_image` action for camera observations
- Moderate GPU usage (~30-40% on A10G)

### nullrhi (Alternative)

```bash
./CarlaUnreal.sh -nullrhi -carla-rpc-port=2000 -fps=20
```

- No rendering at all — text-only observations
- Lighter GPU usage (~15-20% on A10G)
- Faster startup (50-70s)
- `capture_image` will not work

To switch: edit `server/Dockerfile`, remove OpenGL dependencies and change the CARLA launch command.

## Advanced: Client-Server Architecture

For multi-user scenarios, `Dockerfile.real` provides a lightweight CPU client that connects to an external CARLA server:

```bash
docker build -t carla-env-client:latest -f server/Dockerfile.real .
docker run -p 8000:8000 \
  -e CARLA_HOST=your-carla-server.com \
  -e CARLA_PORT=2000 \
  carla-env-client:latest
```

This is useful when multiple researchers share one GPU CARLA server.

## Testing & Validation

### Health Check

```bash
curl https://your-deployment.hf.space/health
```

### Functional Test

```bash
# Reset environment
curl -X POST https://your-deployment.hf.space/reset

# Step with action
curl -X POST https://your-deployment.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "observe"}}'

# Get state
curl https://your-deployment.hf.space/state
```

## Troubleshooting

**"CARLA process died during startup"**
- Check GPU is available (`nvidia-smi`)
- Ensure running as non-root user (CARLA 0.10.0 requirement)
- Increase GPU memory (upgrade to A10G)

**"libGL error: failed to load driver"**
- Verify OpenGL libraries installed (for RenderOffScreen)
- Or switch to nullrhi mode

**"Refusing to run with root privileges"**
- CARLA 0.10.0 requires non-root user — see `server/Dockerfile` for proper user setup

**"Module not found: carla_env"**
- Set `PYTHONPATH=/app` in environment

## Mock Mode (Testing Only)

Mock mode (`CARLA_MODE=mock`) provides simulated physics for automated tests and CI. No CARLA or GPU needed. Not intended for production use.

```bash
# Run tests locally
PYTHONPATH=src:envs uv run pytest tests/envs/test_carla_environment.py -v
```
