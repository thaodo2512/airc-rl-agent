#!/usr/bin/env bash

set -euo pipefail

PORT="${DONKEYSIM_PORT:-9091}"
HEADLESS="${DONKEYSIM_HEADLESS:-1}"
BIN="${DONKEYSIM_BIN:-/opt/donkeysim/DonkeySimLinux/donkey_sim.x86_64}"

if [[ -f "$BIN" && ! -x "$BIN" ]]; then
  chmod +x "$BIN" || true
fi

if [[ ! -x "$BIN" ]]; then
  alt_bin=$(find /opt/donkeysim -maxdepth 2 -type f \( -name "DonkeySim.x86_64" -o -name "donkey_sim.x86_64" \) | head -n 1 || true)
  if [[ -n "${alt_bin:-}" ]]; then
    BIN="$alt_bin"
    if [[ -f "$BIN" && ! -x "$BIN" ]]; then
      chmod +x "$BIN" || true
    fi
  fi
fi

if [[ ! -x "$BIN" ]]; then
  echo "DonkeySim binary not found: $BIN" >&2
  exit 1
fi

args=("$BIN" "--port" "$PORT")
if [[ "$HEADLESS" != "0" ]]; then
  args+=("-batchmode" "-nographics")
fi

echo "Launching DonkeySim: ${args[*]}"
exec "${args[@]}"
