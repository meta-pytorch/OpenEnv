# CodingEnv HTTP Server

This directory contains the HTTP server implementation for the CodingEnvironment.

## Running Locally

### Prerequisites
```bash
pip install fastapi uvicorn
```

### Start the server
```bash
# From the project root (/Users/pankit/git/envtorch)
cd src
uvicorn envs.coding_env.server.app:app --reload --host 0.0.0.0 --port 8000
```

The server will be available at `http://localhost:8000`

### API Endpoints

- `POST /reset` - Reset the environment
- `POST /step` - Execute a code action
- `GET /state` - Get current environment state
- `GET /health` - Health check

### Test with curl

```bash
# Health check
curl http://localhost:8000/health

# Reset
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{}'

# Execute code
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "code": "print(\"Hello from HTTP!\")"
    },
    "timeout_s": 15
  }'

# Get state
curl http://localhost:8000/state
```
