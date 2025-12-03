#!/usr/bin/env bash
set -euo pipefail

# Helper for setting up a venv and launching the repo's notebooks via Jupyter.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
PORT="${NOTEBOOK_PORT:-8888}"

usage() {
  cat <<'EOF'
Usage:
  notebooks.sh setup                  # create .venv and install deps (installs torch)
  notebooks.sh setup-no-torch         # same as setup but skips torch (use if torch preinstalled)
  notebooks.sh list                   # show notebook targets
  notebooks.sh run <target>           # start Jupyter for a notebook

Targets:
  ui                  notebooks/user_interface.ipynb
  ui-nopad            notebooks/user_interface_without_gamepad.ipynb
  recorder            notebooks/recorder/recorder.ipynb
  vae-cnn             notebooks/colabo/VAE_CNN.ipynb
  jetbot-data         notebooks/utility/jetbot/data_collection.ipynb
  jetbot-data-nopad   notebooks/utility/jetbot/data_collection_withoutgamepad.ipynb
  jetbot-vae          notebooks/utility/jetbot/vae_viewer.ipynb
  jetracer-data       notebooks/utility/jetraecr/data_collection.ipynb
  jetracer-data-nopad notebooks/utility/jetraecr/data_collection_wituoutgamepad.ipynb
  jetracer-vae        notebooks/utility/jetraecr/vae_viewer.ipynb

Env vars:
  VENV_DIR        override venv path (default: .venv)
  NOTEBOOK_PORT   override Jupyter port (default: 8888)
  TORCH_SPEC      torch version spec (default: torch==2.4.1); set to "skip" to avoid installing torch
EOF
}

list_targets() {
  usage | sed -n '/^Targets:/,$p'
}

setup_env() {
  local skip_torch="${1:-0}"
  python3 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install --upgrade pip
  if [[ "${TORCH_SPEC:-}" != "skip" && "$skip_torch" != "1" ]]; then
    "$VENV_DIR/bin/pip" install "${TORCH_SPEC:-torch==2.4.1}"
  fi
  "$VENV_DIR/bin/pip" install -e "$REPO_ROOT"
  "$VENV_DIR/bin/pip" install notebook ipywidgets
  echo "Virtualenv ready at $VENV_DIR"
}

ensure_env() {
  if [[ ! -x "$VENV_DIR/bin/jupyter" ]]; then
    echo "Jupyter not found in $VENV_DIR. Run '$0 setup' first." >&2
    exit 1
  fi
}

start_notebook() {
  local nb_path="$1"
  ensure_env
  if [[ ! -f "$REPO_ROOT/$nb_path" ]]; then
    echo "Notebook not found: $nb_path" >&2
    exit 1
  fi
  export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
  cd "$REPO_ROOT"
  echo "Starting Jupyter at http://localhost:$PORT with $nb_path"
  exec "$VENV_DIR/bin/jupyter" notebook "$nb_path" --no-browser --port "$PORT"
}

case "${1:-}" in
  setup)
    setup_env 0
    ;;
  setup-no-torch)
    setup_env 1
    ;;
  list)
    list_targets
    ;;
  run)
    target="${2:-}"
    case "$target" in
      ui)                  start_notebook "notebooks/user_interface.ipynb" ;;
      ui-nopad)            start_notebook "notebooks/user_interface_without_gamepad.ipynb" ;;
      recorder)            start_notebook "notebooks/recorder/recorder.ipynb" ;;
      vae-cnn)             start_notebook "notebooks/colabo/VAE_CNN.ipynb" ;;
      jetbot-data)         start_notebook "notebooks/utility/jetbot/data_collection.ipynb" ;;
      jetbot-data-nopad)   start_notebook "notebooks/utility/jetbot/data_collection_withoutgamepad.ipynb" ;;
      jetbot-vae)          start_notebook "notebooks/utility/jetbot/vae_viewer.ipynb" ;;
      jetracer-data)       start_notebook "notebooks/utility/jetraecr/data_collection.ipynb" ;;
      jetracer-data-nopad) start_notebook "notebooks/utility/jetraecr/data_collection_wituoutgamepad.ipynb" ;;
      jetracer-vae)        start_notebook "notebooks/utility/jetraecr/vae_viewer.ipynb" ;;
      *)
        usage
        exit 1
        ;;
    esac
    ;;
  *)
    usage
    exit 1
    ;;
esac
