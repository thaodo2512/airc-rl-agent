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
DEFAULT_SIM_VERSION="v21.07.24"
DEFAULT_SIM_CACHE="$REPO_ROOT/.sim"
DEFAULT_SIM_URL_BASE="https://github.com/tawnkramer/gym-donkeycar/releases/download"
DEFAULT_SIM_PORT="9091"
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
  sim       Download and launch DonkeySim locally (Linux default).

Options:
  --image TAG      Image name/tag (default: learning-racer-sim)
  --config FILE    Config file (default: <repo>/config.yml)
  --vae FILE       VAE model (default: <repo>/vae.torch)
  --log-dir DIR    Model logs dir (default: <repo>/model_log)
  --model FILE     Policy model for demo (default: <repo>/model)
  --device DEV     Torch device (cpu|cuda, default: cuda)
  --steps N        Demo steps (default: 5000)
  --sim-version V  DonkeySim release tag (default: v21.07.24)
  --sim-cache DIR  Where to cache DonkeySim downloads (default: <repo>/.sim)
  --sim-url URL    Override download URL for DonkeySim zip
  --sim-binary BIN Use an existing DonkeySim binary (skip download)
  --sim-port PORT  Port DonkeySim listens on (default: 9091)
  --headless       Force headless DonkeySim launch (default)
  --no-headless    Launch DonkeySim with a GUI (if available)
  -h, --help       Show this help.

Examples:
  sim.sh build
  sim.sh train --vae ~/vae.torch --log-dir ~/model_log
  sim.sh demo --model ~/model --vae ~/vae.torch
  sim.sh shell
  sim.sh sim
  sim.sh sim --sim-binary /opt/DonkeySim/DonkeySim.x86_64
EOF
}

IMAGE="$DEFAULT_IMAGE"
CONFIG_PATH="$DEFAULT_CONFIG"
VAE_PATH="$DEFAULT_VAE"
LOG_DIR="$DEFAULT_LOG_DIR"
MODEL_PATH="$DEFAULT_MODEL"
DEVICE="$DEFAULT_DEVICE"
STEPS="$DEFAULT_STEPS"
SIM_VERSION="$DEFAULT_SIM_VERSION"
SIM_CACHE="$DEFAULT_SIM_CACHE"
SIM_URL=""
SIM_BINARY_PATH=""
SIM_HEADLESS=1
SIM_PORT="$DEFAULT_SIM_PORT"
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

find_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
  elif command -v python >/dev/null 2>&1; then
    echo "python"
  else
    echo ""
  fi
}

download_file() {
  local url=$1
  local dest=$2
  if command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$dest"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$dest" "$url"
  else
    echo "Need curl or wget to download $url" >&2
    exit 1
  fi
}

extract_zip() {
  local archive=$1
  local target=$2
  local py_bin
  py_bin=$(find_python)
  if [[ -z "$py_bin" ]]; then
    echo "Python is required to extract $archive" >&2
    exit 1
  fi
  "$py_bin" - "$archive" "$target" <<'PY'
import sys
import zipfile
archive, target = sys.argv[1:]
with zipfile.ZipFile(archive) as zf:
    zf.extractall(target)
PY
}

ensure_sim_binary() {
  local os archive sim_dir sim_bin=""
  os=$(uname -s)
  case "$os" in
    Linux*) archive="DonkeySimLinux.zip"; sim_dir="DonkeySimLinux" ;;
    Darwin*) archive="DonkeySimMac.zip"; sim_dir="DonkeySimMac.app" ;;
    *)
      echo "Unsupported platform for DonkeySim auto-launch: $os" >&2
      exit 1
      ;;
  esac

  local version="$SIM_VERSION"
  local base_dir="$SIM_CACHE/$version"
  require_dir "$base_dir" "sim cache directory"

  if [[ -n "$SIM_BINARY_PATH" && -f "$SIM_BINARY_PATH" ]]; then
    SIM_BIN="$SIM_BINARY_PATH"
    return
  fi

  local download_url="${SIM_URL:-$DEFAULT_SIM_URL_BASE/$version/$archive}"
  local download_path="$SIM_CACHE/${version}-${archive}"

  if [[ ! -f "$download_path" ]]; then
    echo "Downloading DonkeySim ($download_url)..."
    download_file "$download_url" "$download_path"
  fi

  if [[ ! -d "$base_dir/$sim_dir" ]]; then
    echo "Extracting DonkeySim to $base_dir"
    extract_zip "$download_path" "$base_dir"
  fi

  local candidates=(
    "$base_dir/$sim_dir/DonkeySim.x86_64"
    "$base_dir/$sim_dir/donkey_sim.x86_64"
    "$base_dir/$sim_dir/DonkeySim"
  )
  for cand in "${candidates[@]}"; do
    if [[ -f "$cand" ]]; then
      sim_bin="$cand"
      break
    fi
  done

  if [[ -z "$sim_bin" ]]; then
    echo "Could not locate DonkeySim binary after extraction. Checked: ${candidates[*]}" >&2
    exit 1
  fi

  chmod +x "$sim_bin"
  SIM_BIN="$sim_bin"
}

run_sim() {
  ensure_sim_binary
  echo "Starting DonkeySim from $SIM_BIN on port $SIM_PORT (headless=$SIM_HEADLESS)"
  local args=("$SIM_BIN" "--port" "$SIM_PORT")
  if (( SIM_HEADLESS )); then
    args+=("-batchmode" "-nographics")
  fi
  exec "${args[@]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    build|train|demo|shell|sim)
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
    --sim-version)
      SIM_VERSION="$2"
      shift 2
      ;;
    --sim-cache)
      SIM_CACHE="$2"
      shift 2
      ;;
    --sim-url)
      SIM_URL="$2"
      shift 2
      ;;
    --sim-binary)
      SIM_BINARY_PATH="$2"
      shift 2
      ;;
    --sim-port)
      SIM_PORT="$2"
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
  sim)
    run_sim
    ;;
esac
