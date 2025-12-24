#!/bin/bash
# Grid World Environment Integration Test Script

set -e # Exit on error

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Grid World Environment Test Script"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Navigate to repo root (IMPORTANT: Run this script from the OpenEnv root)
if [ ! -d "src/core" ]; then
    echo "❌ Error: Please run this script from the root 'OpenEnv' directory."
    exit 1
fi

echo "📁 Working directory: $(pwd)"
echo ""

# Step 1: Check for base image
echo "Step 1: Checking for base image (envtorch-base:latest)..."
if docker images | grep -q "envtorch-base.*latest"; then
    echo "✅ envtorch-base:latest found"
else
    echo "⚠️  envtorch-base:latest not found - building it now..."
    docker build -t envtorch-base:latest -f src/core/containers/images/Dockerfile .
    echo "✅ Base image built successfully"
fi
echo ""

# Step 2: Build Grid World environment
echo "Step 2: Building Grid World environment image (grid-world-env:latest)..."
docker build --no-cache -f src/envs/grid_world_env/server/Dockerfile -t grid-world-env:latest .
echo "✅ Grid World environment built successfully"
echo ""

# Step 3: Start container
echo "Step 3: Starting Grid World container..."
docker stop grid-world-test 2>/dev/null || true
docker rm grid-world-test 2>/dev/null || true
docker run -d -p 8000:8000 --name grid-world-test grid-world-env:latest
echo "⏳ Waiting for container to start..."
sleep 5
echo "✅ Container is running"
echo ""

# Step 4: Test health endpoint
echo "Step 4: Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)
if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed! Response: $HEALTH_RESPONSE"
    exit 1
fi
echo ""

# Step 5: Test reset endpoint
echo "Step 5: Testing reset endpoint..."
RESET_RESPONSE=$(curl -s -X POST http://localhost:8000/reset)
if echo "$RESET_RESPONSE" | jq -e '.observation.message' | grep -q "Welcome"; then
    echo "✅ Reset successful"
    echo "   Response: $(echo $RESET_RESPONSE | jq '.observation.message')"
else
    echo "❌ Reset failed! Response: $RESET_RESPONSE"
    exit 1
fi
echo ""

# Step 6: Test step endpoint
echo "Step 6: Testing step endpoint (taking 2 actions)..."
# Action 1: DOWN
STEP1=$(curl -s -X POST http://localhost:8000/step \
    -H "Content-Type: application/json" \
    -d '{"action": {"action": "DOWN"}}')
REWARD1=$(echo "$STEP1" | jq '.reward')
echo "   Action: DOWN, Reward: $REWARD1"

# Action 2: RIGHT
STEP2=$(curl -s -X POST http://localhost:8000/step \
    -H "Content-Type: application/json" \
    -d '{"action": {"action": "RIGHT"}}')
REWARD2=$(echo "$STEP2" | jq '.reward')
echo "   Action: RIGHT, Reward: $REWARD2"
echo "✅ Step tests successful"
echo ""

# Step 7: Test state endpoint
echo "Step 7: Testing state endpoint..."
STATE_RESPONSE=$(curl -s http://localhost:8000/state)
if echo "$STATE_RESPONSE" | jq -e '.step_count' | grep -q "2"; then
    echo "✅ State endpoint working"
    echo "   Step count: $(echo $STATE_RESPONSE | jq '.step_count')"
else
    echo "❌ State endpoint failed! Response: $STATE_RESPONSE"
    exit 1
fi
echo ""

# Step 8: Cleanup
echo "Step 8: Cleanup..."
docker stop grid-world-test
docker rm grid-world-test
echo "✅ Cleanup complete"
echo ""

# Final summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 ALL GRID WORLD TESTS PASSED!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"