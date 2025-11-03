set -e

echo "========================================"
echo "Pokemon Environment - Manual Start"
echo "========================================"
echo ""

echo "Starting Pokemon Showdown server on port 8000..."
cd /pokemon-showdown
node pokemon-showdown start --no-security &
SHOWDOWN_PID=$!

echo "Waiting for Pokemon Showdown to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000 > /dev/null 2>&1; then
        echo "✅ Pokemon Showdown is ready!"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 1
done

if ! curl -s http://localhost:8000 > /dev/null 2>&1; then
    echo "❌ Pokemon Showdown failed to start"
    exit 1
fi

echo ""
echo "Starting Pokemon OpenEnv server on port 9000..."
cd /app
export PYTHONPATH=/app/src
exec uvicorn envs.pokemon_env.server.app:app --host 0.0.0.0 --port 9000
