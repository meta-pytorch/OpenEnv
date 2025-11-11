set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../../../../.." && pwd )"

IMAGE_NAME="${1:-pokemon-env}"
IMAGE_TAG="${2:-latest}"
BASE_IMAGE="${3:-openenv-base:latest}"

cd "$REPO_ROOT"

# Build the image
docker build \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    -f src/envs/pokemon_env/server/Dockerfile \
    -t "$IMAGE_NAME:$IMAGE_TAG" \
    .
