#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Automated test script for all OpenSpiel games in Docker
# Usage: ./test_docker_all_games.sh

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="openspiel-env:latest"
CONTAINER_NAME="openspiel-test"
PORT=8000
HEALTH_CHECK_URL="http://localhost:${PORT}/health"
MAX_WAIT=30

# Games to test
GAMES=("catch" "tic_tac_toe" "kuhn_poker" "cliff_walking" "2048" "blackjack")

# Results tracking
declare -a RESULTS
PASSED=0
FAILED=0

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}OpenSpiel Docker Integration Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to cleanup containers
cleanup() {
    echo -e "${YELLOW}Cleaning up containers...${NC}"
    docker stop ${CONTAINER_NAME} 2>/dev/null || true
    docker rm ${CONTAINER_NAME} 2>/dev/null || true
}

# Function to wait for server health
wait_for_health() {
    local game=$1
    echo -e "  ⏳ Waiting for server to be ready..."

    for i in $(seq 1 $MAX_WAIT); do
        if curl -s -f ${HEALTH_CHECK_URL} > /dev/null 2>&1; then
            echo -e "  ${GREEN}✓${NC} Server ready (${i}s)"
            return 0
        fi
        sleep 1
    done

    echo -e "  ${RED}✗${NC} Server health check failed after ${MAX_WAIT}s"
    return 1
}

# Function to test a game
test_game() {
    local game=$1
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Testing: ${game}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Stop any existing container
    cleanup

    # Start container with game
    echo -e "  🐳 Starting Docker container..."
    docker run -d \
        --name ${CONTAINER_NAME} \
        -p ${PORT}:8000 \
        -e OPENSPIEL_GAME=${game} \
        ${IMAGE_NAME} > /dev/null

    # Wait for server to be ready
    if ! wait_for_health ${game}; then
        echo -e "  ${RED}✗ FAILED${NC} - Server did not start"
        RESULTS+=("${game}:FAILED:Server did not start")
        FAILED=$((FAILED + 1))
        cleanup
        return 1
    fi

    # Run Python client test
    echo -e "  🎮 Running Python client test..."
    if NO_PROXY=localhost,127.0.0.1 HTTP_PROXY= HTTPS_PROXY= \
       PYTHONPATH=$PWD/src:$PYTHONPATH \
       python3 examples/openspiel_simple.py > /tmp/test_${game}.log 2>&1; then

        # Check if episode completed successfully
        if grep -q "Episode finished!" /tmp/test_${game}.log; then
            echo -e "  ${GREEN}✓ PASSED${NC} - Episode completed successfully"
            RESULTS+=("${game}:PASSED")
            PASSED=$((PASSED + 1))
        else
            echo -e "  ${RED}✗ FAILED${NC} - Episode did not complete"
            RESULTS+=("${game}:FAILED:Episode incomplete")
            FAILED=$((FAILED + 1))
        fi
    else
        echo -e "  ${RED}✗ FAILED${NC} - Python client error"
        RESULTS+=("${game}:FAILED:Client error")
        FAILED=$((FAILED + 1))
    fi

    # Cleanup
    cleanup
}

# Run tests for all games
for game in "${GAMES[@]}"; do
    test_game ${game}
done

# Print summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

for result in "${RESULTS[@]}"; do
    IFS=':' read -r game status message <<< "$result"
    if [ "$status" == "PASSED" ]; then
        echo -e "  ${GREEN}✓${NC} ${game}"
    else
        echo -e "  ${RED}✗${NC} ${game} - ${message}"
    fi
done

echo ""
echo -e "Total: ${PASSED} passed, ${FAILED} failed out of ${#GAMES[@]} games"
echo ""

# Exit with appropriate code
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}All tests PASSED! 🎉${NC}"
    echo -e "${GREEN}========================================${NC}"
    exit 0
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Some tests FAILED${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
