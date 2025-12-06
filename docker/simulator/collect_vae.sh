#!/usr/bin/env bash

# Wrapper to run the DonkeySim frame collector inside the trainer image.
# Example:
#   ./docker/simulator/collect_vae.sh --host localhost --port 9091
# By default this collects 10k frames from the DonkeySim map into ./dataset,
# which plugs directly into notebooks/colabo/VAE_CNN.ipynb for VAE training.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

IMAGE="${IMAGE:-learning-racer-sim}"
WORKDIR="/workspace/airc-rl-agent"

GPU_FLAGS=()
if [[ "${USE_GPU:-1}" == "1" ]]; then
  GPU_FLAGS=(--gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)
fi

DEFAULT_FRAMES="${DEFAULT_FRAMES:-10000}"
DEFAULT_OUT="${DEFAULT_OUT:-dataset}"

COLLECT_ARGS=("$@")
frames_set=0
out_set=0
for arg in "${COLLECT_ARGS[@]}"; do
  case "$arg" in
    --frames|--frames=*) frames_set=1 ;;
    --out|--out=*) out_set=1 ;;
  esac
done

if (( frames_set == 0 )); then
  COLLECT_ARGS+=(--frames "$DEFAULT_FRAMES")
fi

if (( out_set == 0 )); then
  COLLECT_ARGS+=(--out "$DEFAULT_OUT")
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Trainer image '$IMAGE' not found. Build it first: ./docker/simulator/sim.sh build" >&2
  exit 1
fi

docker run --rm -it \
  --network host \
  "${GPU_FLAGS[@]}" \
  -v "$REPO_ROOT":"$WORKDIR" \
  -w "$WORKDIR" \
  --entrypoint python \
  "$IMAGE" \
  scripts/donkey_vae_tools.py collect "${COLLECT_ARGS[@]}"
