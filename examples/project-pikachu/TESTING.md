# Pokemon Environment Testing Guide

This document provides comprehensive testing instructions for the Pokemon battle environment integration with OpenEnv.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Testing (Without Docker)](#local-testing-without-docker)
3. [Docker Testing](#docker-testing)
4. [Test Scenarios](#test-scenarios)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

1. **Python 3.9+**
2. **Node.js 18+** (for Pokemon Showdown server)
3. **Docker** (optional, for containerized testing)

### Python Dependencies

```bash
cd /Users/sanyambhutani/GH/OpenEnv
pip install -r examples/project-pikachu/poke_env/server/requirements.txt
```

Key dependencies:
- `poke-env>=0.9.0`
- `fastapi>=0.104.0`
- `uvicorn>=0.24.0`

---

## Local Testing (Without Docker)

### Step 1: Start Pokemon Showdown Server

```bash
# Clone Pokemon Showdown (if not already done)
cd /tmp
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown

# Install dependencies
npm install

# Configure (use example config)
cp config/config-example.js config/config.js

# Start server without security (for local testing)
node pokemon-showdown start --no-security
```

The server should now be running on `http://localhost:8000`.

**Verification**: Open http://localhost:8000 in a browser - you should see the Pokemon Showdown interface.

### Step 2: Test Environment Directly (Python)

This tests the environment class directly without HTTP:

```bash
cd /Users/sanyambhutani/GH/OpenEnv
python examples/project-pikachu/test_local_pokemon.py
```

**Expected Output**:
```
=============================================================================
TEST 1: Environment Creation
=============================================================================
✅ Environment created successfully
   Player: player_abc12345
   Format: gen9randombattle
   Reward mode: sparse

=============================================================================
TEST 2: Environment Reset
=============================================================================
Calling reset()...
✅ Reset successful!
   Episode ID: 123e4567-e89b-12d3-a456-426614174000
   Battle ID: gen9randombattle-12345
   Turn: 0
   Active Pokemon: pikachu (HP: 100.0%)
   Opponent: charizard (HP: 100.0%)
   ...

[6 tests total - all should pass]

🎉 All tests passed! Pokemon environment is working correctly!
```

### Step 3: Start HTTP Server

In a separate terminal:

```bash
cd /Users/sanyambhutani/GH/OpenEnv
export PYTHONPATH=/Users/sanyambhutani/GH/OpenEnv/src
python -m poke_env.server.app
```

Or use uvicorn directly:

```bash
cd /Users/sanyambhutani/GH/OpenEnv
export PYTHONPATH=/Users/sanyambhutani/GH/OpenEnv/src
uvicorn poke_env.server.app:app --host 0.0.0.0 --port 9980 --reload
```

**Expected Output**:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9980 (Press CTRL+C to quit)
```

**Verification**:
```bash
curl http://localhost:9980/health
# Should return: {"status":"healthy"}
```

### Step 4: Test HTTP Client

In another terminal:

```bash
cd /Users/sanyambhutani/GH/OpenEnv
python examples/project-pikachu/test_http_pokemon.py
```

**Expected Output**:
```
=============================================================================
Pokemon Environment HTTP Test Suite
=============================================================================

Testing server at: http://localhost:9980
Make sure the server is running!

Press Enter to start tests...

=============================================================================
TEST 1: Health Check
=============================================================================
✅ Server is healthy!
   Status: {'status': 'healthy'}

[6 tests total - all should pass]

🎉 All tests passed! HTTP client is working correctly!
```

---

## Docker Testing

### Step 1: Build Docker Image

```bash
cd /Users/sanyambhutani/GH/OpenEnv/examples/project-pikachu/poke_env/server

# Build the image
bash build_docker.sh

# Or manually:
docker build -t pokemon-env:latest -f Dockerfile ../../../..
```

### Step 2: Run Docker Container

```bash
docker run -d \
  --name pokemon-env-test \
  -p 8000:8000 \
  -p 9980:9980 \
  pokemon-env:latest
```

**Verification**:
```bash
# Check logs
docker logs -f pokemon-env-test

# Should see both processes starting:
# - Pokemon Showdown server
# - OpenEnv HTTP server

# Check health
curl http://localhost:9980/health
curl http://localhost:8000  # Should return HTML
```

### Step 3: Test Against Docker Container

```bash
cd /Users/sanyambhutani/GH/OpenEnv
python examples/project-pikachu/test_http_pokemon.py --url http://localhost:9980
```

### Step 4: Cleanup

```bash
docker stop pokemon-env-test
docker rm pokemon-env-test
```

---

## Test Scenarios

### Test 1: Basic Functionality

**What it tests**: Environment creation, reset, single step

**How to run**:
```bash
python examples/project-pikachu/test_local_pokemon.py
```

**Success criteria**:
- Environment creates without errors
- Reset returns valid observation
- Step executes and updates state

### Test 2: Full Battle

**What it tests**: Complete battle from start to finish

**Expected behavior**:
- Battle runs for multiple turns
- Actions execute correctly
- Battle ends with win/loss/tie
- Rewards computed properly

### Test 3: Illegal Move Handling

**What it tests**: Error recovery

**Test case**: Send action with out-of-bounds index (e.g., move index 99)

**Expected behavior**:
- Server doesn't crash
- Error is caught and logged
- Random fallback action is taken
- Battle continues normally

**How to verify**:
```python
# In test script, check metadata:
if "last_error" in obs.metadata:
    print(f"Error caught: {obs.metadata['last_error']}")
    print(f"Illegal count: {obs.metadata['illegal_action_count']}")
```

### Test 4: Dense Rewards

**What it tests**: Reward shaping

**Expected behavior**:
- Non-zero rewards on intermediate steps
- Rewards correlate with battle progress
- Positive for fainting opponent Pokemon
- Negative for losing own Pokemon

**How to run**:
```python
env = PokemonEnvironment(reward_mode="dense")
```

### Test 5: Concurrent Battles

**What it tests**: Multiple clients

**How to run**:
```bash
# Terminal 1: Start server
python -m poke_env.server.app

# Terminal 2: Client 1
python examples/project-pikachu/test_http_pokemon.py

# Terminal 3: Client 2 (simultaneously)
python examples/project-pikachu/test_http_pokemon.py
```

**Expected behavior**:
- Each client gets independent battles
- No interference between battles
- Both complete successfully

### Test 6: Long-Running Battle

**What it tests**: Stability over extended operation

**How to test**:
```python
# Modify test script to run multiple battles
for i in range(10):
    env.reset()
    # ... battle ...
```

**Expected behavior**:
- No memory leaks
- Consistent performance
- Clean battle cleanup

---

## Troubleshooting

### Problem: "Connection refused" or "Failed to connect"

**Cause**: Pokemon Showdown server not running

**Solution**:
```bash
# Check if Showdown is running
curl http://localhost:8000

# If not, start it:
cd pokemon-showdown
node pokemon-showdown start --no-security
```

### Problem: "Battle did not start within 10 seconds"

**Cause**: Pokemon Showdown server is slow or overloaded

**Solution**:
1. Check Showdown logs for errors
2. Restart Showdown server
3. Increase timeout in code (edit `pokemon_environment.py`, line ~500)

### Problem: "No module named 'poke_env'"

**Cause**: Import path issue or not installed

**Solution**:
```bash
# Install poke-env
pip install poke-env>=0.9.0

# Set PYTHONPATH
export PYTHONPATH=/Users/sanyambhutani/GH/OpenEnv/src
```

### Problem: "Event loop is closed" errors

**Cause**: Async/event loop management issue

**Solution**:
- This should be handled by the new implementation
- If you see this, it's a bug - file an issue
- Check that you're using the rewritten `pokemon_environment.py`

### Problem: Tests hang indefinitely

**Cause**: Deadlock in event loop synchronization

**Debug steps**:
1. Enable debug logging:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
2. Check for timeout messages
3. Verify both players are responding

### Problem: "Illegal action" errors

**Expected behavior**: These should be caught and logged, not crash

**If crashing**:
- Check logs for error details
- Verify action validation logic
- Check that fallback random action works

### Problem: Docker build fails

**Common causes**:
1. **Base image not found**: Build openenv-base first
   ```bash
   cd /Users/sanyambhutani/GH/OpenEnv
   docker build -t openenv-base:latest -f docker/Dockerfile .
   ```

2. **Network issues**: Check internet connection for npm/pip downloads

3. **Disk space**: Check available space
   ```bash
   docker system df
   docker system prune  # Clean up if needed
   ```

### Problem: Battles are very slow

**Possible causes**:
1. **Network latency**: Use local Showdown server, not remote
2. **Logging overhead**: Reduce log level to WARNING
3. **Server overload**: Check CPU usage

**Solution**:
```python
# Reduce logging
import logging
logging.getLogger("poke_env").setLevel(logging.WARNING)
```

---

## Performance Benchmarks

Expected performance on modern hardware:

- **Battle initialization**: < 2 seconds
- **Single step**: < 0.5 seconds
- **Full battle (50 turns)**: < 30 seconds
- **Concurrent battles (4x)**: Should not exceed 2x single battle time

If performance is significantly worse, check:
1. CPU usage
2. Network latency
3. Python event loop responsiveness

---

## Next Steps

Once all tests pass:

1. **Integrate with OpenEnv examples**: Create example scripts in `examples/`
2. **Benchmark performance**: Run extended stress tests
3. **Add more battle formats**: Test gen8ou, gen9vgc2024, etc.
4. **Custom teams**: Test with specific team compositions
5. **RL training**: Integrate with RL frameworks (Ray RLlib, Stable-Baselines3)

---

## Getting Help

If tests are failing:

1. **Check logs**: Enable DEBUG logging
2. **Verify setup**: Ensure Showdown is running
3. **Test individually**: Run one test at a time
4. **File issue**: If bug found, create GitHub issue with:
   - Full error message
   - Steps to reproduce
   - System information
   - Logs

Happy testing! 🎮⚡
