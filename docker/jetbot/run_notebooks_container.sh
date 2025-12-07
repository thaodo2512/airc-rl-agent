#!/usr/bin/env bash
set -euo pipefail

# Build the jetbot image and launch Jupyter for notebooks/* inside a container.

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required but not found in PATH." >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE_TAG="${JETBOT_IMAGE_TAG:-airc-jetbot-notebooks}"
CONTAINER_NAME="${JETBOT_CONTAINER_NAME:-airc-jetbot-nb}"
HOST_PORT="${JETBOT_NOTEBOOK_PORT:-8888}"
PYTHONPATH_IN_CONTAINER="${PYTHONPATH_IN_CONTAINER:-/opt/ai-rc-car}"

# Enable BuildKit for faster builds (optional, set to 0 if you have issues)
export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"

# --- LOGIC UPDATE ---
# Default to using Nvidia Runtime if not specified, as this is a Jetson project.
USE_NVIDIA_RUNTIME="${USE_NVIDIA_RUNTIME:-1}"

RUNTIME_FLAG=()
if [[ "${USE_NVIDIA_RUNTIME}" == "1" ]]; then
echo "-> NVIDIA Runtime Enabled (GPU Support)"
  RUNTIME_FLAG+=(--runtime nvidia)
else
  echo "-> WARNING: NVIDIA Runtime Disabled (CPU Only)"
fi

# Control rebuild behavior (default: reuse existing image if present)
REBUILD="${JETBOT_REBUILD:-0}"
if [[ "${REBUILD}" == "1" ]]; then
  echo "** Rebuilding image ${IMAGE_TAG} from docker/jetbot/Dockerfile (JETBOT_REBUILD=1)"
  docker build -t "${IMAGE_TAG}" -f "${REPO_ROOT}/docker/jetbot/Dockerfile" "${REPO_ROOT}"
else
  if docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
    echo "** Using existing image ${IMAGE_TAG} (set JETBOT_REBUILD=1 to force rebuild)"
  else
    echo "** Image ${IMAGE_TAG} not found, building now"
    docker build -t "${IMAGE_TAG}" -f "${REPO_ROOT}/docker/jetbot/Dockerfile" "${REPO_ROOT}"
  fi
fi

echo "** Starting notebook container ${CONTAINER_NAME} on port ${HOST_PORT}"

# --- COMMAND UPDATE ---
# Added: --privileged (for GPIO/I2C/Camera)
# Added: --shm-size (for PyTorch/TF stability)
# Added: Argus socket mount so CSI camera works in container
exec docker run --rm -it \
  --name "${CONTAINER_NAME}" \
  --privileged \
  --shm-size=1g \
  --network host \
  -p "${HOST_PORT}:8888" \
  -v "${REPO_ROOT}:/opt/ai-rc-car" \
  -v /tmp/argus_socket:/tmp/argus_socket \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -w /opt/ai-rc-car \
  -e AI_RC_CAR_HOME=/opt/ai-rc-car \
  -e PYTHONPATH="${PYTHONPATH_IN_CONTAINER}" \
  -e DISPLAY="${DISPLAY:-:0}" \
  "${RUNTIME_FLAG[@]}" \
  "${IMAGE_TAG}" \
  jupyter notebook notebooks \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token='' \
    --NotebookApp.password=''
