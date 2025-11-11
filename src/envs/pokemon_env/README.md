# Pokemon Battle Environment

OpenEnv integration for Pokemon battles using poke-env and Pokemon Showdown.

## Features

- ✅ Full Pokemon battle simulation via poke-env
- ✅ HTTP-based OpenEnv interface
- ✅ Configurable reward modes (sparse/dense)
- ✅ Memory leak prevention with automatic cleanup
- ✅ Thread-safe concurrent request handling
- ✅ Comprehensive battle state tracking
- ✅ Gen 9 support with modern mechanics

## Quick Start

### Local Development

```bash
# Start Pokemon Showdown
cd /tmp && git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown && npm install
node pokemon-showdown start --no-security

# Start Pokemon Environment Server
export PYTHONPATH=/Users/sanyambhutani/GH/OpenEnv/src
python -m envs.pokemon_env.server.app
```

### Using the HTTP Client

```python
from envs.pokemon_env import PokemonEnv, PokemonAction

# Connect to server
client = PokemonEnv(base_url="http://localhost:9980")

# Reset and play
result = client.reset()
print(f"Active: {result.observation.active_pokemon.species}")

# Take action
action = PokemonAction(action_type="move", action_index=0)
result = client.step(action)
print(f"Reward: {result.reward}, Done: {result.done}")
```

### Docker

```bash
# Build both images (run from project root directory)
docker build -t pokemon-showdown:latest -f src/envs/pokemon_env/server/Dockerfile.showdown .
docker build -t pokemon-env:latest -f src/envs/pokemon_env/server/Dockerfile.pokemonenv .

# Create Docker network for container communication
docker network create pokemon-network

# Run Pokemon Showdown server
docker run -d --name pokemon-showdown --network pokemon-network -p 8000:8000 pokemon-showdown:latest

# Run OpenEnv server (pointing to the Showdown container)
docker run -d --name pokemon-env --network pokemon-network -p 9980:9980 pokemon-env:latest

# Test
curl http://localhost:9980/health  # Test OpenEnv server
```

## Configuration

Environment variables:
- `POKEMON_BATTLE_FORMAT` - Battle format (default: `gen8randombattle`)
- `POKEMON_REWARD_MODE` - Reward mode: `sparse` or `dense` (default: `sparse`)
- `POKEMON_MAX_TURNS` - Maximum turns per battle (default: `1000`)
- `POKEMON_PLAYER_USERNAME` - Player username (default: auto-generated)

## Architecture

### Battle Flow

```
HTTP Client → FastAPI Server → PokemonEnvironment (Container 2)
                                      ↓
                              OpenEnvPokemonPlayer
                                      ↓
                              poke-env (POKE_LOOP)
                                      ↓
                     Pokemon Showdown Server (Container 1)
                              (WebSocket)
```

### Key Design Decisions

1. **Single Lock**: Both `reset()` and `step()` use the same lock to prevent concurrent access
2. **Memory Cleanup**: Old battles are cleaned up every 10 episodes
3. **Battle Cancellation**: Previous battle tasks are cancelled on reset
4. **Event Loop Bridge**: Proper async synchronization between FastAPI and poke-env loops
5. **State Validation**: Checks if battle is finished before allowing step()

## Testing

See `/examples/project-pikachu/` for comprehensive test scripts:
- `test_local_pokemon.py` - Direct environment testing
- `test_http_pokemon.py` - HTTP client testing
- `TESTING.md` - Full testing guide

## Known Limitations

- Single battle at a time (no concurrent battles per environment instance)
- Random battles only tested (custom teams supported but untested)
- Singles format only (doubles would require model changes)

## Performance

- Battle initialization: < 2s
- Step execution: < 0.5s
- Full battle (50 turns): < 30s
- Memory: Stable over 100+ episodes (with automatic cleanup)

## Troubleshooting

See `/examples/project-pikachu/TESTING.md` for detailed troubleshooting guide.

Common issues:
- **Connection refused**: Pokemon Showdown not running
- **Battle timeout**: Server overloaded, restart Showdown
- **Memory growth**: Cleanup should handle this automatically

## Credits

- [poke-env](https://github.com/hsahovic/poke-env) - Pokemon battle simulation
- [Pokemon Showdown](https://github.com/smogon/pokemon-showdown) - Battle engine
- [OpenEnv](https://github.com/meta-pytorch/openenv) - HTTP environment framework
