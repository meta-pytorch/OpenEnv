# Test Pokemon environment Docker image
# Similar to test_atari_docker.sh

set -e

IMAGE_NAME="${1:-pokemon-env:latest}"
CONTAINER_NAME="pokemon-env-test"

echo "=========================================================================="
echo "Testing Pokemon Environment Docker Image"
echo "=========================================================================="
echo ""
echo "Image: $IMAGE_NAME"
echo ""

# Clean up any existing container
echo "Cleaning up any existing test containers..."
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

echo ""
echo "Starting container..."
docker run -d \
    -p 9000:9000 \
    -p 8000:8000 \
    --name "$CONTAINER_NAME" \
    "$IMAGE_NAME"

echo "Waiting for services to start..."
sleep 15

echo ""
echo "Checking Pokemon Showdown (port 8000)..."
if curl -s http://localhost:8000 > /dev/null; then
    echo "✅ Pokemon Showdown is running"
else
    echo "❌ Pokemon Showdown is NOT running"
    docker logs "$CONTAINER_NAME"
    docker stop "$CONTAINER_NAME"
    docker rm "$CONTAINER_NAME"
    exit 1
fi

echo ""
echo "Checking OpenEnv API (port 9000)..."
if curl -s http://localhost:9000/health > /dev/null; then
    echo "✅ OpenEnv API is running"
else
    echo "❌ OpenEnv API is NOT running"
    docker logs "$CONTAINER_NAME"
    docker stop "$CONTAINER_NAME"
    docker rm "$CONTAINER_NAME"
    exit 1
fi

echo ""
echo "Testing environment with Python client..."

python3 << 'EOF'
import sys
try:
    # Add src to path
    sys.path.insert(0, 'src')
    
    from envs.pokemon_env import PokemonEnv, PokemonAction
    
    print("Connecting to Pokemon environment...")
    env = PokemonEnv(base_url="http://localhost:9000")
    
    print("Resetting environment...")
    result = env.reset()
    
    print(f"✅ Active Pokemon: {result.observation.active_pokemon.species}")
    print(f"✅ HP: {result.observation.active_pokemon.hp_percent}%")
    print(f"✅ Available moves: {len(result.observation.available_moves)}")
    
    print("\nTaking action...")
    action = PokemonAction(action_type="move", action_index=0)
    result = env.step(action)
    
    print(f"✅ Turn: {result.observation.turn}")
    print(f"✅ Reward: {result.reward}")
    
    env.close()
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

TEST_RESULT=$?

echo ""
echo "Cleaning up..."
docker stop "$CONTAINER_NAME"
docker rm "$CONTAINER_NAME"

if [ $TEST_RESULT -eq 0 ]; then
    echo ""
    echo "=========================================================================="
    echo "✅ All tests passed!"
    echo "=========================================================================="
    echo ""
    exit 0
else
    echo ""
    echo "=========================================================================="
    echo "❌ Tests failed!"
    echo "=========================================================================="
    echo ""
    exit 1
fi
