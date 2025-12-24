#!/bin/bash
# Comprehensive Docker test for Atari environment
# Tests: Build, Start, Health, Reset, Step, State, Cleanup

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="atari-env"
IMAGE_TAG="test"
CONTAINER_NAME="atari-env-test"
PORT="8765"  # Use non-standard port to avoid conflicts
HEALTH_RETRIES=30
HEALTH_DELAY=2

# Cleanup function
cleanup() {
    echo -e "\n${BLUE}Cleaning up...${NC}"
    docker stop ${CONTAINER_NAME} 2>/dev/null || true
    docker rm ${CONTAINER_NAME} 2>/dev/null || true
    echo -e "${GREEN}✓${NC} Cleanup complete"
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Header
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ATARI ENVIRONMENT DOCKER TEST"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗${NC} Docker is not installed"
    exit 1
fi
echo -e "${GREEN}✓${NC} Docker is installed"

if ! command -v curl &> /dev/null; then
    echo -e "${RED}✗${NC} curl is not installed"
    exit 1
fi
echo -e "${GREEN}✓${NC} curl is installed"

# Check if we're in the right directory
if [ ! -f "envs/atari_env/server/Dockerfile" ]; then
    echo -e "${RED}✗${NC} Must run from OpenEnv root directory"
    exit 1
fi
echo -e "${GREEN}✓${NC} In correct directory"

# Step 1: Build Docker image
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}STEP 1: Building Docker Image${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo "Building ${IMAGE_NAME}:${IMAGE_TAG}..."
if docker build -f envs/atari_env/server/Dockerfile -t ${IMAGE_NAME}:${IMAGE_TAG} . 2>&1 | tee /tmp/atari_build.log | tail -n 20; then
    echo -e "${GREEN}✓${NC} Docker image built successfully"
else
    echo -e "${RED}✗${NC} Docker build failed"
    echo "See /tmp/atari_build.log for full output"
    exit 1
fi

# Check image exists
if docker image inspect ${IMAGE_NAME}:${IMAGE_TAG} &> /dev/null; then
    IMAGE_SIZE=$(docker image inspect ${IMAGE_NAME}:${IMAGE_TAG} --format='{{.Size}}' | awk '{print $1/1024/1024}')
    echo -e "${GREEN}✓${NC} Image size: ${IMAGE_SIZE} MB"
else
    echo -e "${RED}✗${NC} Image not found after build"
    exit 1
fi

# Step 2: Start container
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}STEP 2: Starting Container${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Clean up any existing container
docker rm -f ${CONTAINER_NAME} 2>/dev/null || true

echo "Starting container on port ${PORT}..."
docker run -d \
    --name ${CONTAINER_NAME} \
    -p ${PORT}:8000 \
    -e ATARI_GAME=pong \
    -e ATARI_OBS_TYPE=ram \
    -e ATARI_FRAMESKIP=4 \
    ${IMAGE_NAME}:${IMAGE_TAG}

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Container started: ${CONTAINER_NAME}"
else
    echo -e "${RED}✗${NC} Failed to start container"
    exit 1
fi

# Wait for container to be running
sleep 2
if docker ps | grep -q ${CONTAINER_NAME}; then
    echo -e "${GREEN}✓${NC} Container is running"
else
    echo -e "${RED}✗${NC} Container is not running"
    docker logs ${CONTAINER_NAME}
    exit 1
fi

# Step 3: Wait for health check
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}STEP 3: Waiting for Server${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo "Waiting for server to be ready (timeout: ${HEALTH_RETRIES}s)..."
for i in $(seq 1 ${HEALTH_RETRIES}); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Server is ready (${i}s)"
        break
    fi

    if [ $i -eq ${HEALTH_RETRIES} ]; then
        echo -e "${RED}✗${NC} Server did not become ready in time"
        echo "Container logs:"
        docker logs ${CONTAINER_NAME}
        exit 1
    fi

    echo -n "."
    sleep ${HEALTH_DELAY}
done

# Step 4: Test health endpoint
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}STEP 4: Testing Health Endpoint${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

HEALTH_RESPONSE=$(curl -s http://localhost:${PORT}/health)
echo "Response: ${HEALTH_RESPONSE}"

if echo "${HEALTH_RESPONSE}" | grep -q "healthy"; then
    echo -e "${GREEN}✓${NC} Health endpoint working"
else
    echo -e "${RED}✗${NC} Health endpoint failed"
    exit 1
fi

# Step 5: Test reset endpoint
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}STEP 5: Testing Reset Endpoint${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

RESET_RESPONSE=$(curl -s -X POST http://localhost:${PORT}/reset -H "Content-Type: application/json" -d '{}')

if [ -z "${RESET_RESPONSE}" ]; then
    echo -e "${RED}✗${NC} Reset endpoint returned empty response"
    docker logs ${CONTAINER_NAME} | tail -20
    exit 1
fi

echo "Response (first 200 chars): ${RESET_RESPONSE:0:200}..."

# Check if response contains expected fields
if echo "${RESET_RESPONSE}" | grep -q "observation" && \
   echo "${RESET_RESPONSE}" | grep -q "screen" && \
   echo "${RESET_RESPONSE}" | grep -q "legal_actions"; then
    echo -e "${GREEN}✓${NC} Reset endpoint working"

    # Extract some info
    SCREEN_LEN=$(echo "${RESET_RESPONSE}" | grep -o '"screen":\[[^]]*\]' | wc -c)
    echo "  Screen data length: ${SCREEN_LEN} chars"
else
    echo -e "${RED}✗${NC} Reset response missing required fields"
    echo "Full response: ${RESET_RESPONSE}"
    exit 1
fi

# Step 6: Test step endpoint
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}STEP 6: Testing Step Endpoint${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

STEP_PAYLOAD='{"action": {"action_id": 0, "game_name": "pong"}}'
STEP_RESPONSE=$(curl -s -X POST http://localhost:${PORT}/step -H "Content-Type: application/json" -d "${STEP_PAYLOAD}")

if [ -z "${STEP_RESPONSE}" ]; then
    echo -e "${RED}✗${NC} Step endpoint returned empty response"
    docker logs ${CONTAINER_NAME} | tail -20
    exit 1
fi

echo "Response (first 200 chars): ${STEP_RESPONSE:0:200}..."

# Check if response contains expected fields
if echo "${STEP_RESPONSE}" | grep -q "observation" && \
   echo "${STEP_RESPONSE}" | grep -q "reward" && \
   echo "${STEP_RESPONSE}" | grep -q "done"; then
    echo -e "${GREEN}✓${NC} Step endpoint working"

    # Extract reward and done
    REWARD=$(echo "${STEP_RESPONSE}" | grep -o '"reward":[^,}]*' | cut -d: -f2)
    DONE=$(echo "${STEP_RESPONSE}" | grep -o '"done":[^,}]*' | cut -d: -f2)
    echo "  Reward: ${REWARD}"
    echo "  Done: ${DONE}"
else
    echo -e "${RED}✗${NC} Step response missing required fields"
    echo "Full response: ${STEP_RESPONSE}"
    exit 1
fi

# Step 7: Test state endpoint
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}STEP 7: Testing State Endpoint${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

STATE_RESPONSE=$(curl -s http://localhost:${PORT}/state)

if [ -z "${STATE_RESPONSE}" ]; then
    echo -e "${RED}✗${NC} State endpoint returned empty response"
    docker logs ${CONTAINER_NAME} | tail -20
    exit 1
fi

echo "Response: ${STATE_RESPONSE}"

# Check if response contains expected fields
if echo "${STATE_RESPONSE}" | grep -q "episode_id" && \
   echo "${STATE_RESPONSE}" | grep -q "step_count" && \
   echo "${STATE_RESPONSE}" | grep -q "game_name"; then
    echo -e "${GREEN}✓${NC} State endpoint working"

    # Extract info
    GAME_NAME=$(echo "${STATE_RESPONSE}" | grep -o '"game_name":"[^"]*"' | cut -d'"' -f4)
    STEP_COUNT=$(echo "${STATE_RESPONSE}" | grep -o '"step_count":[^,}]*' | cut -d: -f2)
    echo "  Game: ${GAME_NAME}"
    echo "  Steps: ${STEP_COUNT}"
else
    echo -e "${RED}✗${NC} State response missing required fields"
    echo "Full response: ${STATE_RESPONSE}"
    exit 1
fi

# Step 8: Test multiple steps
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}STEP 8: Testing Multiple Steps${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo "Taking 10 steps..."
TOTAL_REWARD=0
for i in {1..10}; do
    ACTION_ID=$((RANDOM % 3))  # Random action 0-2
    STEP_PAYLOAD="{\"action\": {\"action_id\": ${ACTION_ID}, \"game_name\": \"pong\"}}"
    STEP_RESPONSE=$(curl -s -X POST http://localhost:${PORT}/step -H "Content-Type: application/json" -d "${STEP_PAYLOAD}")

    if ! echo "${STEP_RESPONSE}" | grep -q "observation"; then
        echo -e "${RED}✗${NC} Step ${i} failed"
        exit 1
    fi

    REWARD=$(echo "${STEP_RESPONSE}" | grep -o '"reward":[^,}]*' | cut -d: -f2 | sed 's/null/0/')
    DONE=$(echo "${STEP_RESPONSE}" | grep -o '"done":[^,}]*' | cut -d: -f2)

    echo "  Step ${i}: action=${ACTION_ID}, reward=${REWARD}, done=${DONE}"

    if [ "${DONE}" = "true" ]; then
        echo "  Episode completed early at step ${i}"
        break
    fi
done

echo -e "${GREEN}✓${NC} Multiple steps completed successfully"

# Step 9: Check container logs for errors
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}STEP 9: Checking Container Logs${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

LOGS=$(docker logs ${CONTAINER_NAME} 2>&1)

if echo "${LOGS}" | grep -i "error" | grep -v "LoggerMode.Error"; then
    echo -e "${YELLOW}⚠${NC}  Found errors in logs:"
    echo "${LOGS}" | grep -i "error" | head -5
else
    echo -e "${GREEN}✓${NC} No errors in container logs"
fi

if echo "${LOGS}" | grep -i "exception"; then
    echo -e "${RED}✗${NC} Found exceptions in logs:"
    echo "${LOGS}" | grep -i "exception" | head -5
    exit 1
else
    echo -e "${GREEN}✓${NC} No exceptions in container logs"
fi

# Final Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ ALL DOCKER TESTS PASSED${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Summary:"
echo "  ✓ Docker image built successfully"
echo "  ✓ Container started and ran"
echo "  ✓ Health endpoint working"
echo "  ✓ Reset endpoint working"
echo "  ✓ Step endpoint working"
echo "  ✓ State endpoint working"
echo "  ✓ Multiple steps working"
echo "  ✓ No errors or exceptions"
echo ""
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Container: ${CONTAINER_NAME}"
echo "Port: ${PORT}"
echo ""
echo "To keep container running: docker start ${CONTAINER_NAME}"
echo "To view logs: docker logs ${CONTAINER_NAME}"
echo ""
