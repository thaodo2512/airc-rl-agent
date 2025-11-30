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
DEFAULT_ROBOT="sim"
DEFAULT_SIM_VERSION="v21.07.24"
DEFAULT_SIM_IMAGE="learning-racer-donkey-sim"
DEFAULT_SIM_PORT="9091"
DEFAULT_SIM_NAME="learning-racer-sim-env"
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
  sim       Build and launch DonkeySim inside Docker (headless by default).
  run-all   Start DonkeySim (Docker) and run train/demo against it.

Options:
  --image TAG      Image name/tag (default: learning-racer-sim)
  --config FILE    Config file (default: <repo>/config.yml)
  --vae FILE       VAE model (default: <repo>/vae.torch)
  --log-dir DIR    Model logs dir (default: <repo>/model_log)
  --model FILE     Policy model for demo (default: <repo>/model)
  --device DEV     Torch device (cpu|cuda, default: cuda)
  --steps N        Demo steps (default: 5000)
  --robot NAME     Robot driver for racer (default: sim)
  --sim-version V  DonkeySim release tag (default: v21.07.24)
  --sim-image IMG  DonkeySim Docker image (default: learning-racer-donkey-sim:<version>)
  --sim-port PORT  Port DonkeySim listens on (default: 9091)
  --headless       Headless DonkeySim launch (default)
  --no-headless    Launch DonkeySim with a GUI (if available)
  --sim-name NAME  Container name for the sim (default: learning-racer-sim-env)
  --mode MODE      Mode for run-all: train|demo (default: train)
  -h, --help       Show this help.

Examples:
  sim.sh build
  sim.sh train --vae ~/vae.torch --log-dir ~/model_log
  sim.sh demo --model ~/model --vae ~/vae.torch
  sim.sh shell
  sim.sh sim
  sim.sh sim --sim-image my/sim:latest
  sim.sh run-all --vae ~/vae.torch --log-dir ./model_log
  sim.sh run-all --mode demo --model ~/model --vae ~/vae.torch --steps 2000
EOF
}

IMAGE="$DEFAULT_IMAGE"
CONFIG_PATH="$DEFAULT_CONFIG"
VAE_PATH="$DEFAULT_VAE"
LOG_DIR="$DEFAULT_LOG_DIR"
MODEL_PATH="$DEFAULT_MODEL"
DEVICE="$DEFAULT_DEVICE"
STEPS="$DEFAULT_STEPS"
ROBOT="$DEFAULT_ROBOT"
SIM_VERSION="$DEFAULT_SIM_VERSION"
SIM_IMAGE=""
SIM_HEADLESS=1
SIM_PORT="$DEFAULT_SIM_PORT"
SIM_NAME="$DEFAULT_SIM_NAME"
SIM_MODE="train"
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

ensure_sim_image() {
  local image_ref=$1
  if docker image inspect "$image_ref" >/dev/null 2>&1; then
    return
  fi
  echo "Building DonkeySim Docker image ($image_ref) with version $SIM_VERSION..."
  docker build -t "$image_ref" \
    -f "$SCRIPT_DIR/DonkeySim.Dockerfile" \
    --build-arg DONKEYSIM_VERSION="$SIM_VERSION" \
    --platform "$PLATFORM" \
    "$SCRIPT_DIR"
}

start_sim_container() {
  local image_ref="$1"
  local cid
  ensure_sim_image "$image_ref"
  # Remove any stale container with the same name
  docker rm -f "$SIM_NAME" >/dev/null 2>&1 || true
  cid=$(docker run -d --rm \
    --name "$SIM_NAME" \
    --network host \
    --platform "$PLATFORM" \
    -e "DONKEYSIM_PORT=$SIM_PORT" \
    -e "DONKEYSIM_HEADLESS=$SIM_HEADLESS" \
    "$image_ref")
  echo "$cid"
}

stop_sim_container() {
  docker rm -f "$SIM_NAME" >/dev/null 2>&1 || true
}

wait_for_sim() {
  local retries=30
  local delay=1
  echo "Waiting for DonkeySim to accept connections on port $SIM_PORT ..."
  for ((i=1; i<=retries; i++)); do
    if (echo > "/dev/tcp/127.0.0.1/$SIM_PORT") >/dev/null 2>&1; then
      echo "DonkeySim is up."
      return 0
    fi
    sleep "$delay"
  done
  echo "Timed out waiting for DonkeySim on port $SIM_PORT" >&2
  return 1
}

run_sim() {
  local image_ref="$SIM_IMAGE"
  if [[ -z "$image_ref" ]]; then
    image_ref="$DEFAULT_SIM_IMAGE:$SIM_VERSION"
  fi
  ensure_sim_image "$image_ref"
  echo "Starting DonkeySim in Docker ($image_ref) on port $SIM_PORT (headless=$SIM_HEADLESS)"
  local run_cmd=(
    docker run --rm -it
    --network host
    --platform "$PLATFORM"
    --gpus all
    -e "DONKEYSIM_PORT=$SIM_PORT"
    -e "DONKEYSIM_HEADLESS=$SIM_HEADLESS"
  )

  if [[ "$SIM_HEADLESS" -eq 0 ]]; then
    # Mount host display for GUI mode
    run_cmd+=(
      -e "DISPLAY=${DISPLAY:-:0}"
      -v /tmp/.X11-unix:/tmp/.X11-unix
    )
    if [[ -n "${XAUTHORITY:-}" && -f "$XAUTHORITY" ]]; then
      run_cmd+=(-v "$XAUTHORITY":/root/.Xauthority:ro)
    fi
    if [[ -e /dev/dri ]]; then
      run_cmd+=(--device /dev/dri)
    fi
  fi

  run_cmd+=("$image_ref")
  exec "${run_cmd[@]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    build|train|demo|shell|sim|run-all)
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
    --robot)
      ROBOT="$2"
      shift 2
      ;;
    --sim-version)
      SIM_VERSION="$2"
      shift 2
      ;;
    --sim-image)
      SIM_IMAGE="$2"
      shift 2
      ;;
    --sim-port)
      SIM_PORT="$2"
      shift 2
      ;;
    --sim-name)
      SIM_NAME="$2"
      shift 2
      ;;
    --mode)
      SIM_MODE="$2"
      shift 2
      ;;
    --headless)
      SIM_HEADLESS=1
      shift
      ;;
    --no-headless)
      SIM_HEADLESS=0
      shift
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

run_train() {
  require_file "$CONFIG_PATH" "config file"
  require_file "$VAE_PATH" "VAE model"
  require_dir "$LOG_DIR" "log directory"
  "${DOCKER_RUN[@]}" \
    -v "$CONFIG_PATH":/workspace/airc-rl-agent/config.yml:ro \
    -v "$VAE_PATH":/workspace/airc-rl-agent/vae.torch:ro \
    -v "$LOG_DIR":/workspace/airc-rl-agent/model_log \
    "$IMAGE" train -robot "$ROBOT" -vae vae.torch -config config.yml -device "$DEVICE"
}

run_demo() {
  require_file "$CONFIG_PATH" "config file"
  require_file "$VAE_PATH" "VAE model"
  require_file "$MODEL_PATH" "policy model"
  "${DOCKER_RUN[@]}" \
    -v "$CONFIG_PATH":/workspace/airc-rl-agent/config.yml:ro \
    -v "$VAE_PATH":/workspace/airc-rl-agent/vae.torch:ro \
    -v "$MODEL_PATH":/workspace/airc-rl-agent/model:ro \
    "$IMAGE" demo -robot "$ROBOT" -model model -vae vae.torch -config config.yml -device "$DEVICE" -steps "$STEPS"
}

run_all() {
  local image_ref="$SIM_IMAGE"
  if [[ -z "$image_ref" ]]; then
    image_ref="$DEFAULT_SIM_IMAGE:$SIM_VERSION"
  fi
  local cid
  cid=$(start_sim_container "$image_ref")
  trap stop_sim_container EXIT
  wait_for_sim || { stop_sim_container; exit 1; }
  if [[ "$SIM_MODE" == "demo" ]]; then
    run_demo
  else
    run_train
  fi
}

case "$COMMAND" in
  build)
    docker build -t "$IMAGE" -f "$SCRIPT_DIR/Dockerfile" --build-arg BASE_IMAGE="$BASE_IMAGE" --platform "$PLATFORM" "$REPO_ROOT"
    ;;
  train)
    run_train
    ;;
  demo)
    run_demo
    ;;
  shell)
    require_dir "$LOG_DIR" "log directory"
    RUN_CMD=("${DOCKER_RUN[@]}" --entrypoint /bin/bash)
    [[ -f "$CONFIG_PATH" ]] && RUN_CMD+=(-v "$CONFIG_PATH":/workspace/airc-rl-agent/config.yml:ro)
    [[ -f "$VAE_PATH" ]] && RUN_CMD+=(-v "$VAE_PATH":/workspace/airc-rl-agent/vae.torch:ro)
    RUN_CMD+=(-v "$LOG_DIR":/workspace/airc-rl-agent/model_log)
    "${RUN_CMD[@]}" "$IMAGE"
    ;;
  sim)
    run_sim
    ;;
  run-all)
    run_all
    ;;
esac
