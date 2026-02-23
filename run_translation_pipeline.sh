#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER_PY="$ROOT_DIR/scripts/run_jobs_multi_lang.py"

HOST="${HOST:-127.0.0.1}"
PORTS="${PORTS:-11440,11441,11442}"
GPUS="${GPUS:-0,1,2}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/.ollama_local/logs}"
PID_DIR="${PID_DIR:-$ROOT_DIR/.ollama_local/pids}"
MODELS_DIR="${MODELS_DIR:-/mnt/nabu_main/ollama_models}"

KEEP_SERVERS=0
SKIP_START=0
START_ONLY=0
STOP_ONLY=0
REUSE_EXISTING=0
PURGE_EXISTING=0
STARTED_ANY=0
PULL_MODELS=0

ARGS=()

usage() {
  cat <<'USAGE'
Usage:
  ./run_translation_pipeline.sh [pipeline flags] -- <scripts/run_jobs_multi_lang.py args>

Pipeline flags:
  --host HOST         Host for ollama servers (default: 127.0.0.1)
  --ports CSV         Ports list (default: 11440,11441,11442)
  --gpus CSV          GPU ids list (default: 0,1,2)
  --log-dir DIR       Log directory for ollama servers
  --pid-dir DIR       PID directory for ollama servers
  --models-dir DIR    Ollama models directory for started servers
  --keep-servers      Leave ollama servers running after the job
  --reuse-existing    Reuse servers already listening on requested ports
                      (ensure those servers are GPU-pinned)
  --purge-existing    Stop any running ollama servers on requested ports
                      before starting new ones
  --pull-models       Pull missing models before running
  --skip-start        Do not start servers (assume they are already running)
  --start-only        Start servers and exit
  --stop              Stop servers started by this script and exit
  -h, --help          Show this help

Examples:
  ./run_translation_pipeline.sh -- \
    --source-file data/1143_en.txt \
    --prompt-template main_prompt_template.txt \
    --pair de \
    --pair ru \
    --models "aya-expanse:32b,qwen2.5:32b,deepseek-r1:32b" \
    --out-root output/multi_lang_run

Environment overrides (optional):
  PORTS, GPUS, HOST, LOG_DIR, PID_DIR, MODELS_DIR, OLLAMA_MODELS,
  OLLAMA_MAX_LOADED_MODELS, OLLAMA_KEEP_ALIVE, PYTHON_BIN
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-servers)
      KEEP_SERVERS=1
      shift
      ;;
    --reuse-existing)
      REUSE_EXISTING=1
      shift
      ;;
    --purge-existing)
      PURGE_EXISTING=1
      shift
      ;;
    --pull-models)
      PULL_MODELS=1
      shift
      ;;
    --skip-start)
      SKIP_START=1
      shift
      ;;
    --start-only)
      START_ONLY=1
      KEEP_SERVERS=1
      shift
      ;;
    --stop)
      STOP_ONLY=1
      shift
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --ports)
      PORTS="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --pid-dir)
      PID_DIR="$2"
      shift 2
      ;;
    --models-dir)
      MODELS_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      ARGS+=("$@")
      break
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "$RUNNER_PY" ]]; then
  echo "Error: missing runner script at $RUNNER_PY"
  exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "Error: ollama binary not found in PATH."
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: python not found in PATH (set PYTHON_BIN to override)."
  exit 1
fi

if [[ $REUSE_EXISTING -eq 1 && $PURGE_EXISTING -eq 1 ]]; then
  echo "Error: --reuse-existing and --purge-existing are mutually exclusive."
  exit 2
fi

detect_models_dir() {
  if [[ -n "$MODELS_DIR" ]]; then
    return 0
  fi
  if ! command -v systemctl >/dev/null 2>&1; then
    return 0
  fi
  local line=""
  if command -v rg >/dev/null 2>&1; then
    line="$(systemctl cat ollama 2>/dev/null | rg -o 'OLLAMA_MODELS=[^" ]+' | head -n 1 || true)"
  else
    line="$(systemctl cat ollama 2>/dev/null | grep -o 'OLLAMA_MODELS=[^\" ]*' | head -n 1 || true)"
  fi
  if [[ -n "$line" ]]; then
    MODELS_DIR="${line#OLLAMA_MODELS=}"
  fi
}

IFS=',' read -r -a PORT_ARR <<< "$PORTS"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"

if [[ ${#PORT_ARR[@]} -ne ${#GPU_ARR[@]} ]]; then
  echo "Error: --ports and --gpus must have the same number of entries."
  exit 2
fi

detect_models_dir

SERVERS=()
for idx in "${!PORT_ARR[@]}"; do
  SERVERS+=("${HOST}:${PORT_ARR[$idx]}")
done
SERVERS_CSV="$(IFS=','; echo "${SERVERS[*]}")"

mkdir -p "$LOG_DIR" "$PID_DIR"

ensure_writable_path() {
  local path="$1"
  if [[ -e "$path" && ! -w "$path" ]]; then
    rm -f "$path" 2>/dev/null || true
  fi
  if [[ -e "$path" && ! -w "$path" ]]; then
    echo "Error: $path is not writable. Remove or chown it (e.g., sudo rm -f \"$path\" or sudo chown \"$USER\" \"$path\")."
    exit 1
  fi
}

ensure_writable_dir() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    mkdir -p "$dir" 2>/dev/null || true
  fi
  if [[ ! -w "$dir" ]]; then
    echo "Error: directory is not writable: $dir"
    exit 1
  fi
}

stop_servers() {
  if [[ ! -d "$PID_DIR" ]]; then
    echo "No PID directory found at $PID_DIR"
    return 0
  fi
  shopt -s nullglob
  local pid_files=("$PID_DIR"/*.pid)
  if [[ ${#pid_files[@]} -eq 0 ]]; then
    echo "No PID files found in $PID_DIR"
    return 0
  fi
  for pid_file in "${pid_files[@]}"; do
    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "Stopping ollama PID $pid (from $pid_file)"
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
  done
}

if [[ $STOP_ONLY -eq 1 ]]; then
  stop_servers
  exit 0
fi

is_port_open() {
  local port="$1"
  "$PYTHON_BIN" - "$HOST" "$port" <<'PY'
import socket
import sys
host = sys.argv[1]
port = int(sys.argv[2])
s = socket.socket()
s.settimeout(0.2)
try:
    s.connect((host, port))
    sys.exit(0)
except Exception:
    sys.exit(1)
finally:
    s.close()
PY
}

find_pids_by_port() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true
    return 0
  fi
  if command -v ss >/dev/null 2>&1; then
    ss -lptn "sport = :$port" 2>/dev/null | awk -F 'pid=' 'NR>1 {split($2,a,","); print a[1]}' || true
    return 0
  fi
  "$PYTHON_BIN" - "$port" <<'PY'
import os
import sys

port = int(sys.argv[1])

def inodes_for_port(path):
    try:
        with open(path) as f:
            next(f, None)
            for line in f:
                parts = line.split()
                if len(parts) < 10:
                    continue
                local_addr = parts[1]
                state = parts[3]
                if state != "0A":  # LISTEN
                    continue
                _, port_hex = local_addr.split(":")
                if int(port_hex, 16) == port:
                    yield parts[9]
    except FileNotFoundError:
        return

inodes = set(inodes_for_port("/proc/net/tcp")) | set(inodes_for_port("/proc/net/tcp6"))
if not inodes:
    sys.exit(0)

for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    fd_dir = f"/proc/{pid}/fd"
    try:
        fds = os.listdir(fd_dir)
    except (FileNotFoundError, PermissionError):
        continue
    found = False
    for fd in fds:
        try:
            target = os.readlink(os.path.join(fd_dir, fd))
        except OSError:
            continue
        if target.startswith("socket:["):
            inode = target[8:-1]
            if inode in inodes:
                print(pid)
                found = True
                break
    if found:
        continue
PY
}

purge_port_if_ollama() {
  local port="$1"
  local pids
  pids="$(find_pids_by_port "$port")"
  local killed_any=0
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    local cmdline=""
    if [[ -r "/proc/$pid/cmdline" ]]; then
      cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    fi
    if [[ "$cmdline" == *ollama* ]]; then
      echo "Purging ollama PID $pid on $HOST:$port"
      kill "$pid" 2>/dev/null || true
      if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
      fi
      killed_any=1
    else
      echo "Error: port $port is in use by non-ollama process (PID $pid: $cmdline)"
      exit 1
    fi
  done <<< "$pids"

  for _ in {1..25}; do
    if ! is_port_open "$port"; then
      return 0
    fi
    sleep 0.2
  done

  if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet ollama; then
    echo "Port $port is still in use; stopping systemd ollama service."
    systemctl stop ollama >/dev/null 2>&1 || true
    for _ in {1..25}; do
      if ! is_port_open "$port"; then
        return 0
      fi
      sleep 0.2
    done
  fi

  if [[ $killed_any -eq 0 ]]; then
    echo "Error: port $port is in use, but no ollama PID was found to purge."
  else
    echo "Error: port $port is still in use after purge."
  fi
  exit 1
}

wait_for_server() {
  local port="$1"
  "$PYTHON_BIN" - "$HOST" "$port" <<'PY'
import sys
import time
import urllib.request

host = sys.argv[1]
port = sys.argv[2]
url = f"http://{host}:{port}/api/tags"
deadline = time.time() + 30
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            if resp.status == 200:
                sys.exit(0)
    except Exception:
        time.sleep(0.5)
sys.exit(1)
PY
}

PIDS=()

start_servers() {
  local max_loaded="${OLLAMA_MAX_LOADED_MODELS:-1}"
  ensure_writable_dir "$LOG_DIR"
  ensure_writable_dir "$PID_DIR"
  for idx in "${!PORT_ARR[@]}"; do
    local port="${PORT_ARR[$idx]}"
    local gpu="${GPU_ARR[$idx]}"
    local pid_file="$PID_DIR/ollama_gpu${gpu}_port${port}.pid"
    local log_file="$LOG_DIR/ollama_gpu${gpu}_port${port}.log"

    ensure_writable_path "$pid_file"
    ensure_writable_path "$log_file"

    if is_port_open "$port"; then
      if [[ $REUSE_EXISTING -eq 1 ]]; then
        echo "Port $port is already in use on $HOST; reusing existing server."
        continue
      fi
      if [[ $PURGE_EXISTING -eq 1 ]]; then
        purge_port_if_ollama "$port"
        if is_port_open "$port"; then
          echo "Error: port $port is still in use on $HOST."
          exit 1
        fi
      else
        echo "Error: port $port is already in use on $HOST."
        exit 1
      fi
    fi

    echo "Starting ollama on $HOST:$port (GPU $gpu)"
    local env_vars=(CUDA_VISIBLE_DEVICES="$gpu" OLLAMA_HOST="$HOST:$port" OLLAMA_MAX_LOADED_MODELS="$max_loaded")
    if [[ -n "${OLLAMA_KEEP_ALIVE:-}" ]]; then
      env_vars+=(OLLAMA_KEEP_ALIVE="$OLLAMA_KEEP_ALIVE")
    fi
    if [[ -n "$MODELS_DIR" ]]; then
      env_vars+=(OLLAMA_MODELS="$MODELS_DIR")
    fi
    env "${env_vars[@]}" ollama serve >"$log_file" 2>&1 &
    local pid=$!
    echo "$pid" > "$pid_file"
    PIDS+=("$pid")
    STARTED_ANY=1
  done

  for port in "${PORT_ARR[@]}"; do
    if ! wait_for_server "$port"; then
      echo "Error: ollama did not start on $HOST:$port"
      exit 1
    fi
  done
}

cleanup() {
  if [[ $STARTED_ANY -eq 0 ]]; then
    return 0
  fi
  if [[ $KEEP_SERVERS -eq 0 ]]; then
    stop_servers
    return 0
  fi
  echo "Leaving ollama servers running."
}

if [[ $SKIP_START -eq 0 ]]; then
  trap cleanup EXIT INT TERM
  start_servers
else
  for port in "${PORT_ARR[@]}"; do
    if ! is_port_open "$port"; then
      echo "Error: expected ollama server on $HOST:$port but it is not reachable."
      exit 1
    fi
  done
fi

if [[ $START_ONLY -eq 1 ]]; then
  echo "Servers started. Exiting due to --start-only."
  exit 0
fi

args_have_flag() {
  local flag="$1"
  for arg in "${ARGS[@]}"; do
    if [[ "$arg" == "$flag" || "$arg" == "$flag="* ]]; then
      return 0
    fi
  done
  return 1
}

extract_flag_value() {
  local flag="$1"
  local idx=0
  while [[ $idx -lt ${#ARGS[@]} ]]; do
    local arg="${ARGS[$idx]}"
    if [[ "$arg" == "$flag" ]]; then
      local next_idx=$((idx + 1))
      if [[ $next_idx -lt ${#ARGS[@]} ]]; then
        echo "${ARGS[$next_idx]}"
        return 0
      fi
      break
    fi
    if [[ "$arg" == "$flag="* ]]; then
      echo "${arg#${flag}=}"
      return 0
    fi
    idx=$((idx + 1))
  done
  return 1
}

ensure_models() {
  local models_csv="$1"
  local host_port="${HOST}:${PORT_ARR[0]}"
  IFS=',' read -r -a model_arr <<< "$models_csv"
  local missing=()
  for model in "${model_arr[@]}"; do
    local trimmed
    trimmed="$(echo "$model" | xargs)"
    if [[ -z "$trimmed" ]]; then
      continue
    fi
    if ! OLLAMA_HOST="http://${host_port}" ollama show "$trimmed" >/dev/null 2>&1; then
      missing+=("$trimmed")
    fi
  done

  if [[ ${#missing[@]} -eq 0 ]]; then
    return 0
  fi

  if [[ $PULL_MODELS -eq 1 ]]; then
    echo "Pulling missing models: ${missing[*]}"
    for model in "${missing[@]}"; do
      OLLAMA_HOST="http://${host_port}" ollama pull "$model"
    done
    return 0
  fi

  echo "Error: missing models in Ollama store: ${missing[*]}"
  echo "Use --pull-models or run: OLLAMA_HOST=http://${host_port} ollama pull <model>"
  exit 1
}

CMD=("$PYTHON_BIN" "$RUNNER_PY")
if ! args_have_flag "--servers"; then
  CMD+=("--servers" "$SERVERS_CSV")
fi
if ! args_have_flag "--server-gpus"; then
  CMD+=("--server-gpus" "$GPUS")
fi
CMD+=("${ARGS[@]}")

MODELS_CSV="$(extract_flag_value --models || true)"
if [[ -n "$MODELS_CSV" ]]; then
  ensure_models "$MODELS_CSV"
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
