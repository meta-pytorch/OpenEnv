#!/bin/bash
# Test Day 2 actions via HTTP API

echo "=========================================="
echo "Day 2 HTTP API Integration Test"
echo "=========================================="
echo ""

# Start the server in background (mock mode)
echo "Starting CARLA environment server (mock mode)..."
cd /Users/sergiopaniegoblanco/Documents/Projects/OpenEnv
PYTHONPATH=src:envs uv run python -m openenv.cli.http_server \
    --env carla_env \
    --port 8765 \
    --mode mock &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server to start..."
sleep 3

# Test 1: Reset
echo ""
echo "Test 1: Reset environment"
curl -s -X POST http://localhost:8765/reset \
    -H "Content-Type: application/json" \
    -d '{"scenario_name": "trolley_saves"}' | python3 -m json.tool | head -20

# Test 2: brake_vehicle action
echo ""
echo "Test 2: brake_vehicle action (Day 2)"
curl -s -X POST http://localhost:8765/step \
    -H "Content-Type: application/json" \
    -d '{
        "action_type": "brake_vehicle",
        "brake_intensity": 0.7
    }' | python3 -m json.tool | head -20

# Test 3: maintain_speed action
echo ""
echo "Test 3: maintain_speed action (Day 2)"
curl -s -X POST http://localhost:8765/step \
    -H "Content-Type: application/json" \
    -d '{
        "action_type": "maintain_speed",
        "target_speed_kmh": 25.0
    }' | python3 -m json.tool | head -20

# Test 4: improved lane_change with target_lane_id
echo ""
echo "Test 4: lane_change with target_lane_id (Day 2 improvement)"
curl -s -X POST http://localhost:8765/step \
    -H "Content-Type: application/json" \
    -d '{
        "action_type": "lane_change",
        "target_lane_id": "lane_1"
    }' | python3 -m json.tool | head -20

# Test 5: observe action (backward compatibility)
echo ""
echo "Test 5: observe action (backward compatibility)"
curl -s -X POST http://localhost:8765/step \
    -H "Content-Type: application/json" \
    -d '{
        "action_type": "observe"
    }' | python3 -m json.tool | head -20

# Cleanup
echo ""
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo ""
echo "=========================================="
echo "✅ Day 2 HTTP API tests complete!"
echo "=========================================="
