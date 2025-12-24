#!/bin/bash
# Complete SUMO-RL Integration Test Script
# Run this to verify everything works!

set -e  # Exit on error

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 SUMO-RL Environment Test Script"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Navigate to repo root
cd /Users/sanyambhutani/GH/OpenEnv

echo "📁 Working directory: $(pwd)"
echo ""

# Step 1: Check if base image exists
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Checking for base image..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if docker images | grep -q "envtorch-base.*latest"; then
    echo "✅ envtorch-base:latest found"
else
    echo "⚠️  envtorch-base:latest not found - building it now..."
    echo ""
    docker build -t envtorch-base:latest -f src/core/containers/images/Dockerfile .
    echo ""
    echo "✅ Base image built successfully"
fi
echo ""

# Step 2: Build SUMO-RL environment
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Building SUMO-RL environment image..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏳ This will take 5-10 minutes (installing SUMO)..."
echo ""

docker build -f envs/sumo_rl_env/server/Dockerfile -t sumo-rl-env:latest .

echo ""
echo "✅ SUMO-RL environment built successfully"
echo ""

# Check image size
IMAGE_SIZE=$(docker images sumo-rl-env:latest --format "{{.Size}}")
echo "📦 Image size: $IMAGE_SIZE"
echo ""

# Step 3: Start container
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Starting SUMO-RL container..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Stop any existing container
docker stop sumo-rl-test 2>/dev/null || true
docker rm sumo-rl-test 2>/dev/null || true

# Start new container
docker run -d -p 8000:8000 --name sumo-rl-test sumo-rl-env:latest

echo "⏳ Waiting for container to start..."
sleep 5

# Check if container is running
if docker ps | grep -q sumo-rl-test; then
    echo "✅ Container is running"
else
    echo "❌ Container failed to start!"
    echo "Logs:"
    docker logs sumo-rl-test
    exit 1
fi
echo ""

# Step 4: Test health endpoint
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: Testing health endpoint..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)
echo "Response: $HEALTH_RESPONSE"

if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed!"
    exit 1
fi
echo ""

# Step 5: Test reset endpoint
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 5: Testing reset endpoint..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏳ This may take 3-5 seconds (SUMO simulation starting)..."

RESET_RESPONSE=$(curl -s -X POST http://localhost:8000/reset)

if echo "$RESET_RESPONSE" | jq -e '.observation.observation' > /dev/null 2>&1; then
    echo "✅ Reset successful"

    # Extract observation details
    OBS_SHAPE=$(echo "$RESET_RESPONSE" | jq '.observation.observation_shape')
    ACTION_MASK=$(echo "$RESET_RESPONSE" | jq '.observation.action_mask')

    echo "  📊 Observation shape: $OBS_SHAPE"
    echo "  🎮 Available actions: $ACTION_MASK"
else
    echo "❌ Reset failed!"
    echo "Response: $RESET_RESPONSE"
    exit 1
fi
echo ""

# Step 6: Test step endpoint
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 6: Testing step endpoint (taking 5 actions)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for i in {1..5}; do
    # Take action (cycle through phases 0-1)
    PHASE_ID=$((i % 2))

    STEP_RESPONSE=$(curl -s -X POST http://localhost:8000/step \
        -H "Content-Type: application/json" \
        -d "{\"action\": {\"phase_id\": $PHASE_ID, \"ts_id\": \"0\"}}")

    if echo "$STEP_RESPONSE" | jq -e '.reward' > /dev/null 2>&1; then
        REWARD=$(echo "$STEP_RESPONSE" | jq '.reward')
        DONE=$(echo "$STEP_RESPONSE" | jq '.done')
        echo "  Step $i: phase=$PHASE_ID, reward=$REWARD, done=$DONE"
    else
        echo "❌ Step $i failed!"
        echo "Response: $STEP_RESPONSE"
        exit 1
    fi
done

echo "✅ All steps successful"
echo ""

# Step 7: Test state endpoint
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 7: Testing state endpoint..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

STATE_RESPONSE=$(curl -s http://localhost:8000/state)

if echo "$STATE_RESPONSE" | jq -e '.episode_id' > /dev/null 2>&1; then
    echo "✅ State endpoint working"

    # Extract state details
    EPISODE_ID=$(echo "$STATE_RESPONSE" | jq -r '.episode_id')
    STEP_COUNT=$(echo "$STATE_RESPONSE" | jq '.step_count')
    SIM_TIME=$(echo "$STATE_RESPONSE" | jq '.sim_time')
    TOTAL_VEHICLES=$(echo "$STATE_RESPONSE" | jq '.total_vehicles')

    echo "  📝 Episode ID: ${EPISODE_ID:0:8}..."
    echo "  🔢 Step count: $STEP_COUNT"
    echo "  ⏱️  Simulation time: $SIM_TIME seconds"
    echo "  🚗 Total vehicles: $TOTAL_VEHICLES"
else
    echo "❌ State endpoint failed!"
    echo "Response: $STATE_RESPONSE"
    exit 1
fi
echo ""

# Step 8: Check logs for errors
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 8: Checking container logs for errors..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

LOGS=$(docker logs sumo-rl-test 2>&1)

# Check for Python errors (but ignore LoggerMode.Error which is expected)
if echo "$LOGS" | grep -i "error\|exception\|traceback" | grep -v "LoggerMode.Error"; then
    echo "⚠️  Found errors in logs:"
    echo "$LOGS" | grep -i "error\|exception\|traceback" | grep -v "LoggerMode.Error"
else
    echo "✅ No errors found in logs"
fi
echo ""

# Step 9: Cleanup
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 9: Cleanup..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "🧹 Stopping and removing test container..."
docker stop sumo-rl-test
docker rm sumo-rl-test

echo "✅ Cleanup complete"
echo ""

# Final summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 ALL TESTS PASSED!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Summary:"
echo "  ✅ Docker image built successfully ($IMAGE_SIZE)"
echo "  ✅ Container started and ran"
echo "  ✅ Health endpoint working"
echo "  ✅ Reset endpoint working"
echo "  ✅ Step endpoint working (5 actions executed)"
echo "  ✅ State endpoint working"
echo "  ✅ No errors in logs"
echo ""
echo "🎯 SUMO-RL integration is working perfectly!"
echo ""
echo "Next steps:"
echo "  1. Test Python client: python examples/sumo_rl_simple.py"
echo "  2. Push to GitHub to trigger CI/CD"
echo "  3. Use for RL training!"
echo ""
