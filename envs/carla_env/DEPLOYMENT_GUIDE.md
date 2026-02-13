# CARLA Environment - Deployment Guide

Quick reference for choosing and configuring deployment modes.

## 🎯 Quick Decision Tree

```
Need CARLA physics?
├─ NO  → Mock Mode (CPU, free, instant)
└─ YES → Need GPU?
    ├─ Have GPU → Standalone Mode (all-in-one)
    └─ No GPU  → Client Mode (connect to external CARLA server)
```

## 📊 Mode Comparison

### Hardware Requirements

| Mode | CPU | RAM | GPU | VRAM | Disk |
|------|-----|-----|-----|------|------|
| **Mock** | 1 core | 512MB | None | - | 2GB |
| **Client** | 2 cores | 1GB | None | - | 2GB |
| **Standalone** | 4 cores | 8GB | NVIDIA | 16GB+ | 15GB |

### Performance Characteristics

| Mode | Build Time | Startup Time | Step Latency | FPS |
|------|-----------|--------------|--------------|-----|
| **Mock** | 5 min | <10s | ~50ms | 20 |
| **Client** | 5 min | <10s* | ~100ms | 10-20 |
| **Standalone** | 30-60 min | 60-90s | ~100ms | 10-20 |

*If CARLA server already running

### Cost Estimates (HuggingFace Spaces)

| Mode | Hardware | Cost/Hour | Cost/Month (24/7) |
|------|----------|-----------|-------------------|
| **Mock** | CPU (2 cores) | Free | Free |
| **Client** | CPU (2 cores) | Free | Free* |
| **Standalone** | T4 GPU | $0.60 | $432 |
| **Standalone** | A10G GPU | $1.10 | $792 |

*Client is free, but you pay for CARLA server separately

## 🔧 Configuration Files

### Mock Mode

**Dockerfile**: `server/Dockerfile`

**Key settings**:
```python
# In app.py or environment variable
CARLA_MODE=mock
CARLA_SCENARIO=trolley_saves
```

**No CARLA installation needed** - pure Python simulation.

### Client Mode

**Dockerfile**: `server/Dockerfile.real`

**Key settings**:
```bash
CARLA_MODE=real
CARLA_HOST=your-carla-server.com  # External CARLA server
CARLA_PORT=2000
CARLA_SCENARIO=trolley_saves
```

**Dependencies**:
- External CARLA 0.10.0 server must be running
- Network connectivity to CARLA server
- Python package: `carla-ue5-api==0.10.0`

### Standalone Mode

**Dockerfile**: `server/Dockerfile.real-standalone`

**Key settings**:
```bash
CARLA_MODE=real
CARLA_HOST=localhost  # CARLA runs in same container
CARLA_PORT=2000
CARLA_SCENARIO=trolley_saves

# CARLA startup command (in Dockerfile):
./CarlaUnreal.sh -RenderOffScreen -opengl -quality-level=Low -carla-rpc-port=2000 -fps=20
```

**Dependencies**:
- NVIDIA GPU with 16GB+ VRAM
- CUDA 11.8+
- OpenGL libraries (for RenderOffScreen)
- XDG user directories

## 🎨 Rendering Modes (Standalone Only)

### RenderOffScreen (Current Default)

**Command**:
```bash
./CarlaUnreal.sh -RenderOffScreen -opengl -quality-level=Low -carla-rpc-port=2000 -fps=20
```

**Characteristics**:
- GPU renders frames offscreen (no display)
- Text-only observations by default
- Camera sensors can be added later
- Moderate GPU usage (~30-40% on A10G)
- Startup: 60-90 seconds

**When to Use**:
- Future multimodal support desired
- Camera sensors might be added later
- GPU resources available
- Research flexibility important

### nullrhi (Alternative)

**Command**:
```bash
./CarlaUnreal.sh -nullrhi -carla-rpc-port=2000 -fps=20
```

**Characteristics**:
- No rendering at all (null render hardware interface)
- Text-only observations only
- Lighter GPU usage (~15-20% on A10G)
- Faster startup: 50-70 seconds
- Camera sensors require rebuild

**When to Use**:
- Text-only scenarios forever
- Maximum efficiency needed
- GPU costs are critical
- Following PrimeIntellect approach

**To Switch**: See "How to Switch to nullrhi" in README.md

## 🚀 Deployment Commands

### Mock Mode

```bash
# Local Docker
docker build -t carla-env:latest -f server/Dockerfile .
docker run -p 8000:8000 carla-env:latest

# HuggingFace Spaces (free tier)
openenv push --repo-id your-username/carla-env
```

### Client Mode

```bash
# Local Docker (connects to external CARLA at CARLA_HOST)
docker build -t carla-env-client:latest -f server/Dockerfile.real .
docker run -p 8000:8000 \
  -e CARLA_HOST=your-carla-server.com \
  -e CARLA_PORT=2000 \
  carla-env-client:latest

# HuggingFace Spaces (free CPU tier)
# Note: Set CARLA_HOST in Space settings to point to your CARLA server
openenv push --repo-id your-username/carla-env-client
```

### Standalone Mode

```bash
# Local Docker (requires GPU)
docker build -t carla-env-standalone:latest -f server/Dockerfile.real-standalone .
docker run --gpus all -p 8000:8000 carla-env-standalone:latest

# HuggingFace Spaces (GPU tier)
# Note: Enable GPU hardware in Space settings (T4 or A10G)
openenv push --repo-id your-username/carla-env-real
```

## 📈 GPU Selection (Standalone Mode)

### NVIDIA T4 (16GB VRAM)

**Pros**:
- Cheaper ($0.60/hour)
- Available on HuggingFace Spaces
- Works for text-only scenarios

**Cons**:
- Tight on memory (may OOM on complex scenes)
- Slower than A10G
- Less headroom for camera sensors

**Recommended For**:
- Budget-conscious deployments
- Text-only scenarios
- Testing before scaling up

### NVIDIA A10G (24GB VRAM)

**Pros**:
- Stable and performant
- Headroom for camera sensors
- Faster rendering
- Future-proof

**Cons**:
- More expensive ($1.10/hour)
- May be overkill for text-only

**Recommended For**:
- Production deployments
- Multimodal scenarios (with cameras)
- Research requiring stability
- Long-running experiments

## 🧪 Testing & Validation

### Health Check

```bash
# Quick test (all modes)
curl https://your-deployment.hf.space/health

# Python script
python examples/check_carla_health.py
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

### Performance Benchmarks

Expected response times (50th percentile):

| Operation | Mock | Client | Standalone |
|-----------|------|--------|------------|
| `/reset` | 50ms | 150ms | 800ms |
| `/step` | 50ms | 150ms | 1.4s |
| `/state` | 10ms | 50ms | 400ms |

## 🐛 Troubleshooting

### Mock Mode

**Issue**: "Module not found: carla_env"
- **Fix**: Set `PYTHONPATH=/app` in environment

### Client Mode

**Issue**: "Cannot connect to CARLA server"
- **Fix**: Verify `CARLA_HOST` and `CARLA_PORT` are correct
- **Fix**: Ensure CARLA server is running and accessible
- **Fix**: Check firewall rules (port 2000)

### Standalone Mode

**Issue**: "CARLA process died during startup"
- **Fix**: Check GPU is available (`nvidia-smi`)
- **Fix**: Ensure running as non-root user
- **Fix**: Increase GPU memory (upgrade to A10G)

**Issue**: "libGL error: failed to load driver"
- **Fix**: Verify OpenGL libraries installed (for RenderOffScreen)
- **Fix**: Or switch to nullrhi mode

**Issue**: "Refusing to run with root privileges"
- **Fix**: CARLA 0.10.0 requires non-root user
- **Fix**: See Dockerfile.real-standalone for proper user setup

## 📚 Additional Resources

- **README.md**: Complete feature documentation
- **OFFSCREEN_RENDERING.md**: Technical details on rendering modes
- **examples/**: Example Python clients
- **CARLA 0.10.0 Docs**: https://carla.readthedocs.io/

## 🔄 Migration Guide

### From Mock to Standalone

1. Ensure GPU available
2. Use `Dockerfile.real-standalone`
3. Set `CARLA_MODE=real`
4. Wait 60-90s for startup
5. Same API, higher fidelity

### From Standalone to Client

1. Deploy CARLA server separately
2. Use `Dockerfile.real`
3. Set `CARLA_HOST` to server address
4. Deploy client (CPU-only, cheaper)
5. Same API, distributed architecture

### From RenderOffScreen to nullrhi

1. Edit Dockerfile (remove OpenGL libs)
2. Change CARLA command to `-nullrhi`
3. Rebuild image
4. Expect 10-20% faster startup
5. Same text observations, no camera support
