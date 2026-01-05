#!/bin/bash
# Setup script for shared Gitea instance
# This script starts Gitea, waits for it to be ready, and creates the admin user
# Requires: .env file with GITEA_USERNAME and GITEA_PASSWORD

set -e

# Load credentials from .env file
if [ -f .env ]; then
    export $(cat .env | grep -E '^(GITEA_USERNAME|GITEA_PASSWORD)=' | xargs)
else
    echo "❌ Error: .env file not found"
    echo "   Please copy .env.example to .env and configure credentials"
    exit 1
fi

echo "====================================="
echo "Setting up shared Gitea instance"
echo "====================================="
echo

# Start Gitea with docker-compose
echo "1. Starting Gitea container..."
docker-compose -f envs/git_env/docker-compose.gitea.yml up -d

# Wait for Gitea to be healthy
echo "2. Waiting for Gitea to be ready..."
timeout=60
elapsed=0
while [ $elapsed -lt $timeout ]; do
    if docker exec openenv-gitea curl -sf http://localhost:3000/ > /dev/null 2>&1; then
        echo "   ✓ Gitea is ready!"
        break
    fi
    echo "   Waiting... (${elapsed}s/${timeout}s)"
    sleep 2
    elapsed=$((elapsed + 2))
done

if [ $elapsed -ge $timeout ]; then
    echo "   ✗ Timeout waiting for Gitea"
    exit 1
fi

# Initialize Gitea (POST to root URL)
echo "3. Initializing Gitea configuration..."
docker exec openenv-gitea curl -s -X POST \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "db_type=sqlite3" \
    -d "db_path=%2Fdata%2Fgitea%2Fgitea.db" \
    -d "app_name=Gitea" \
    -d "repo_root_path=%2Fdata%2Fgit%2Frepositories" \
    -d "run_user=git" \
    -d "domain=gitea" \
    -d "http_port=3000" \
    -d "app_url=http%3A%2F%2Fgitea%3A3000%2F" \
    -d "log_root_path=%2Fdata%2Fgitea%2Flog" \
    -d "offline_mode=on" \
    http://localhost:3000/ > /dev/null || echo "   (Config may already exist)"

# Create admin user
echo "4. Creating admin user ($GITEA_USERNAME)..."
docker exec openenv-gitea su git -c \
    "gitea admin user create --username $GITEA_USERNAME --password $GITEA_PASSWORD --email ${GITEA_USERNAME}@local.env --admin" \
    2>&1 | grep -q "already exists" && echo "   ✓ User already exists" || echo "   ✓ User created"

echo
echo "====================================="
echo "✓ Gitea setup complete!"
echo "====================================="
echo
echo "Gitea is now available at:"
echo "  - Web UI: http://localhost:3000"
echo "  - From containers: http://gitea:3000"
echo
echo "Admin credentials are configured from .env file"
echo
echo "To stop Gitea:"
echo "  docker-compose -f envs/git_env/docker-compose.gitea.yml down"
echo
echo "To remove all data:"
echo "  docker-compose -f envs/git_env/docker-compose.gitea.yml down -v"
echo
