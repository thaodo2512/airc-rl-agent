#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

DEFAULT_IMAGE="learning-racer-sim"
DEFAULT_CONFIG="$REPO_ROOT/config.yml"
DEFAULT_VAE="$REPO_ROOT/vae.torch"
DEFAULT_LOG_DIR="$REPO_ROOT/model_log"
DEFAULT_MODEL="$REPO_ROOT/model"
DEFAULT_DEVICE="cuda"
DEFAULT_STEPS="5000"
BASE_IMAGE="pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime"  # Hardcoded GPU base
PLATFORM="linux/amd64"  # Hardcoded for amd64 arch

usage() {
  cat <<'EOF'
Usage: sim.sh [options] <command>

Commands:
  build     Build the simulator image.
  train     Run training against DonkeySim.
  demo      Run a demo rollout.
  shell     Start an interactive shell.

Options:
  --image TAG      Image name/tag (default: learning-racer-sim)
  --config FILE    Config file (default: <repo>/config.yml)
  --vae FILE       VAE model (default: <repo>/vae.torch)
  --log-dir DIR    Model logs dir (default: <repo>/model_log)
  --model FILE     Policy model for demo (default: <repo>/model)
  --device DEV     Torch device (cpu|cuda, default: cuda)
  --steps N        Demo steps (default: 5000)
  -h, --help       Show this help.

Examples:
  sim.sh build
  sim.sh train --vae ~/vae.torch --log-dir ~/model_log
  sim.sh demo --model ~/model --vae ~/vae.torch
  sim.sh shell
EOF
}

IMAGE="$DEFAULT_IMAGE"
CONFIG_PATH="$DEFAULT_CONFIG"
VAE_PATH="$DEFAULT_VAE"
LOG_DIR="$DEFAULT_LOG_DIR"
MODEL_PATH="$DEFAULT_MODEL"
DEVICE="$DEFAULT_DEVICE"
STEPS="$DEFAULT_STEPS"
COMMAND=""

require_file() {
  local path=$1
  local label=$2
  if [[ ! -f "$path" ]]; then
    echo "Missing $label: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path=$1
  local label=$2
  if [[ ! -d "$path" ]]; then
    mkdir -p "$path" || {
      echo "Failed to create $label: $path" >&2
      exit 1
    }
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    build|train|demo|shell)
      COMMAND="$1"
      shift
      ;;
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --vae)
      VAE_PATH="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$COMMAND" ]]; then
  usage >&2
  exit 1
fi

DOCKER_RUN=(docker run --rm -it --network host --gpus all --platform "$PLATFORM")

case "$COMMAND" in
  build)
    docker build -t "$IMAGE" -f "$SCRIPT_DIR/Dockerfile" --build-arg BASE_IMAGE="$BASE_IMAGE" --platform "$PLATFORM" "$REPO_ROOT"
    ;;
  train)
    require_file "$CONFIG_PATH" "config file"
    require_file "$VAE_PATH" "VAE model"
    require_dir "$LOG_DIR" "log directory"
    "${DOCKER_RUN[@]}" \
      -v "$CONFIG_PATH":/workspace/airc-rl-agent/config.yml:ro \
      -v "$VAE_PATH":/workspace/airc-rl-agent/vae.torch:ro \
      -v "$LOG_DIR":/workspace/airc-rl-agent/model_log \
      "$IMAGE" train -robot sim -vae vae.torch -config config.yml -device "$DEVICE"
    ;;
  demo)
    require_file "$CONFIG_PATH" "config file"
    require_file "$VAE_PATH" "VAE model"
    require_file "$MODEL_PATH" "policy model"
    "${DOCKER_RUN[@]}" \
      -v "$CONFIG_PATH":/workspace/airc-rl-agent/config.yml:ro \
      -v "$VAE_PATH":/workspace/airc-rl-agent/vae.torch:ro \
      -v "$MODEL_PATH":/workspace/airc-rl-agent/model:ro \
      "$IMAGE" demo -robot sim -model model -vae vae.torch -config config.yml -device "$DEVICE" -steps "$STEPS"
    ;;
  shell)
    require_dir "$LOG_DIR" "log directory"
    RUN_CMD=("${DOCKER_RUN[@]}" --entrypoint /bin/bash)
    [[ -f "$CONFIG_PATH" ]] && RUN_CMD+=(-v "$CONFIG_PATH":/workspace/airc-rl-agent/config.yml:ro)
    [[ -f "$VAE_PATH" ]] && RUN_CMD+=(-v "$VAE_PATH":/workspace/airc-rl-agent/vae.torch:ro)
    RUN_CMD+=(-v "$LOG_DIR":/workspace/airc-rl-agent/model_log)
    "${RUN_CMD[@]}" "$IMAGE"
    ;;
esac
