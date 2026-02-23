#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$ROOT_DIR/scripts/analyze_efficiency.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "$SCRIPT" ]]; then
  echo "Error: missing $SCRIPT" >&2
  exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: python not found in PATH (set PYTHON_BIN to override)." >&2
  exit 1
fi

usage() {
  cat <<'USAGE'
Usage:
  ./run_efficiency_reports.sh [RUN_DIR ...]

Behavior:
  - If RUN_DIR arguments are provided, each must contain stats.json.
  - If no args are provided, it auto-discovers output/* directories with stats.json.

Output:
  For each run dir, writes plots to:
    <RUN_DIR>/visualizations_efficiency/
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

collect_runs() {
  local runs=()

  if [[ $# -gt 0 ]]; then
    for dir in "$@"; do
      if [[ -d "$dir" && -f "$dir/stats.json" ]]; then
        runs+=("$dir")
      else
        echo "Skipping '$dir' (expected directory with stats.json)" >&2
      fi
    done
  else
    shopt -s nullglob
    for dir in output/*; do
      if [[ -d "$dir" && -f "$dir/stats.json" ]]; then
        runs+=("$dir")
      fi
    done
  fi

  if [[ ${#runs[@]} -eq 0 ]]; then
    echo "No run directories found with stats.json" >&2
    return 1
  fi

  printf '%s\n' "${runs[@]}"
}

mapfile -t RUN_DIRS < <(collect_runs "$@")

for run_dir in "${RUN_DIRS[@]}"; do
  stats_path="$run_dir/stats.json"
  out_dir="$run_dir/visualizations_efficiency"

  rm -rf "$out_dir"
  mkdir -p "$out_dir"

  echo "==> Processing $run_dir"
  "$PYTHON_BIN" "$SCRIPT" --stats "$stats_path" --output-dir "$out_dir"
  echo "Saved visualizations to $out_dir"
  echo
done
