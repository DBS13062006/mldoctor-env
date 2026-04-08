#!/usr/bin/env bash
#
# run_baselines_v2.sh — Windows + Linux + macOS safe baseline runner

set -uo pipefail

N_RUNS="${1:-3}"
PORT="${2:-7860}"
HOST="127.0.0.1"
BASE_URL="http://${HOST}:${PORT}"
RESULTS_FILE="baseline_results.md"
LOG_DIR="baseline_logs"

TASKS=("obvious_failure_diagnosis" "subtle_divergence_diagnosis" "adversarial_compound_failure")
TASK_LABELS=("Easy" "Medium" "Hard")

PY="${PY:-python}"
command -v "$PY" >/dev/null 2>&1 || PY=python3

if [ ! -f "inference.py" ]; then
  echo "ERROR: inference.py not found. Run from the repo root." >&2
  exit 1
fi

if [ -f ".env" ]; then
  set -a; source .env; set +a
fi
MODEL_DISPLAY="${MODEL_NAME:-unknown-model}"
mkdir -p "$LOG_DIR"

is_port_free() {
  "$PY" - "$PORT" <<'PY'
import socket, sys
port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("127.0.0.1", port))
    s.close()
    sys.exit(0)
except OSError:
    sys.exit(1)
PY
}

kill_pid_tree() {
  local pid="$1"
  "$PY" - "$pid" <<'PY' 2>/dev/null || kill -9 "$1" 2>/dev/null
import sys, os, signal
pid = int(sys.argv[1])
try:
    import psutil
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        try: child.kill()
        except: pass
    parent.kill()
except ImportError:
    try: os.kill(pid, signal.SIGKILL)
    except: pass
except Exception:
    pass
PY
  sleep 1
}

wait_for_port_free() {
  for i in 1 2 3 4 5 6 7 8 9 10; do
    if is_port_free; then return 0; fi
    sleep 1
  done
  return 1
}

wait_for_server_ready() {
  for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    if curl -s --max-time 2 -X POST "${BASE_URL}/reset" \
         -H "Content-Type: application/json" -d '{}' >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

verify_patch_via_http() {
  curl -s --max-time 5 -X POST "${BASE_URL}/reset" \
       -H "Content-Type: application/json" -d '{}' >/dev/null
  local step_response
  step_response=$(curl -s --max-time 5 -X POST "${BASE_URL}/step" \
    -H "Content-Type: application/json" \
    -d '{"action": {"name": "prescribe", "args": {"failure_mode": "mode_collapse", "config_diff": {"foo": 1}}}}')
  echo "$step_response" | "$PY" -c "
import json, sys
try:
    data = json.load(sys.stdin)
    done = data.get('done', False)
    if done is True:
        print('  [sentinel] PATCHED env confirmed (wrong prescribe ends episode)')
        sys.exit(0)
    else:
        print('  [sentinel] FAIL: wrong prescribe did NOT end episode (done='+str(done)+')', file=sys.stderr)
        sys.exit(1)
except Exception as e:
    print('  [sentinel] FAIL: '+str(e), file=sys.stderr)
    sys.exit(2)
"
  return $?
}

start_server_for_task() {
  local task="$1"
  echo "  [server] ensuring port ${PORT} is free ..."
  if ! wait_for_port_free; then
    echo "  [server] ERROR: port ${PORT} still in use after 10s." >&2
    return 1
  fi

  echo "  [server] starting MLDoctorEnv for task=${task} ..."
  MLDOCTOR_TASK="$task" PYTHONPATH=. nohup "$PY" -m uvicorn \
    mldoctor_env.server.app:app \
    --host "$HOST" --port "$PORT" \
    > "${LOG_DIR}/server_${task}.log" 2>&1 &
  local pid=$!
  echo "$pid" > "${LOG_DIR}/server_${task}.pid"
  echo "  [server] PID=$pid"

  if ! wait_for_server_ready; then
    echo "  [server] ERROR: server did not become ready. Check ${LOG_DIR}/server_${task}.log" >&2
    return 1
  fi
  echo "  [server] ready"

  if ! verify_patch_via_http; then
    return 1
  fi
}

sweep_uvicorn_processes() {
  "$PY" - <<'PY' 2>/dev/null
try:
    import psutil
    for p in psutil.process_iter(['cmdline']):
        try:
            cl = ' '.join(p.info.get('cmdline') or [])
            if 'uvicorn' in cl and 'mldoctor_env' in cl:
                p.kill()
        except Exception:
            pass
except ImportError:
    pass
PY
  sleep 1
}

stop_server_for_task() {
  local task="$1"
  if [ -f "${LOG_DIR}/server_${task}.pid" ]; then
    local pid
    pid=$(cat "${LOG_DIR}/server_${task}.pid")
    echo "  [server] killing PID=$pid"
    kill_pid_tree "$pid"
    rm -f "${LOG_DIR}/server_${task}.pid"
  fi
  sweep_uvicorn_processes
  if ! wait_for_port_free; then
    echo "  [server] WARNING: port ${PORT} still in use after kill" >&2
  fi
}

run_inference_once() {
  local task="$1"; local run_idx="$2"; local outfile="$3"
  echo "  [inference] task=${task} run=${run_idx} ..."
  MLDOCTOR_TASK="$task" BASE_URL="$BASE_URL" \
    "$PY" inference.py 2>&1 | tee "$outfile"
}

parse_end_line() {
  local log="$1"; local line
  line=$(grep -E "^\[END\]" "$log" | tail -1 | tr -d '\r')
  if [ -z "$line" ]; then echo "false 0 0.000"; return; fi
  local success steps score
  success=$(echo "$line" | sed -n 's/.*success=\([a-z]*\).*/\1/p')
  steps=$(echo "$line" | sed -n 's/.*steps=\([0-9]*\).*/\1/p')
  score=$(echo "$line" | sed -n 's/.*score=\([0-9.]*\).*/\1/p')
  echo "${success:-false} ${steps:-0} ${score:-0.000}"
}

run_random_baseline() {
  PYTHONPATH=. "$PY" - <<'PY'
import random, sys, json
from mldoctor_env.server.environment import MLDoctorEnvironment
from mldoctor_env.models import MLDoctorAction
ACTIONS = ["inspect_loss_curve","inspect_grad_norms","inspect_hyperparams","inspect_dataset_stats","inspect_error_log","submit"]
TASKS = ["obvious_failure_diagnosis","subtle_divergence_diagnosis","adversarial_compound_failure"]
results = {}
random.seed(0)
for task in TASKS:
    scores = []
    for trial in range(20):
        env = MLDoctorEnvironment(task_id=task, seed=trial)
        env.reset()
        total = 0.0
        for _ in range(env.max_steps):
            r = env.step(MLDoctorAction(name=random.choice(ACTIONS)))
            total += r.reward
            if r.done: break
        scores.append(max(0.0, min(1.0, total)))
    results[task] = {"mean": sum(scores)/len(scores), "min": min(scores), "max": max(scores)}
print(json.dumps(results))
PY
}

echo "================================================================"
echo "  MLDoctorEnv baseline runner v2 (Windows-safe + sentinel)"
echo "================================================================"
echo "  N runs per task : $N_RUNS"
echo "  Server port     : $PORT"
echo "  Model           : $MODEL_DISPLAY"
echo ""

declare -A LLM_MEANS LLM_MINS LLM_MAXS LLM_SUCCESSES

for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  label="${TASK_LABELS[$i]}"
  echo "----------------------------------------------------------------"
  echo " Task ${i}: ${task} (${label})"
  echo "----------------------------------------------------------------"

  if ! start_server_for_task "$task"; then
    echo "  [error] skipping ${task}"
    LLM_MEANS[$task]="ERR"; LLM_MINS[$task]="ERR"; LLM_MAXS[$task]="ERR"; LLM_SUCCESSES[$task]="0/0"
    continue
  fi

  scores=(); successes=0
  for run in $(seq 1 "$N_RUNS"); do
    log="${LOG_DIR}/inference_${task}_run${run}.log"
    run_inference_once "$task" "$run" "$log"
    read -r success steps score < <(parse_end_line "$log")
    scores+=("$score")
    if [ "$success" = "true" ]; then successes=$((successes+1)); fi
    echo "  [parsed] run=${run} success=${success} steps=${steps} score=${score}"
  done

  stop_server_for_task "$task"

  mean_score=$(printf '%s\n' "${scores[@]}" | awk '{s+=$1;n++} END{if(n>0) printf "%.3f", s/n; else print "0.000"}')
  min_score=$(printf '%s\n' "${scores[@]}" | awk 'NR==1||$1<m{m=$1} END{printf "%.3f", m}')
  max_score=$(printf '%s\n' "${scores[@]}" | awk 'NR==1||$1>m{m=$1} END{printf "%.3f", m}')

  LLM_MEANS[$task]="$mean_score"; LLM_MINS[$task]="$min_score"; LLM_MAXS[$task]="$max_score"
  LLM_SUCCESSES[$task]="${successes}/${N_RUNS}"

  echo "  ${label} summary: mean=${mean_score} min=${min_score} max=${max_score} successes=${successes}/${N_RUNS}"
  echo ""
done

echo "================================================================"
echo "  Random baseline (20 trials per task)"
echo "================================================================"
RAND_JSON=$(run_random_baseline)
echo "$RAND_JSON" | "$PY" -m json.tool

OUT="$RESULTS_FILE"
{
  echo "## Baseline scores"
  echo ""
  echo "Measured by running \`inference.py\` ${N_RUNS} times per task against \`${MODEL_DISPLAY}\` via the Hugging Face router. Random baseline is a uniform-random policy across 20 trials."
  echo ""
  echo "| Task | Difficulty | Random (mean) | ${MODEL_DISPLAY} (mean) | min | max | success rate |"
  echo "|---|---|---|---|---|---|---|"
  for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"
    label="${TASK_LABELS[$i]}"
    rand_mean=$(echo "$RAND_JSON" | "$PY" -c "import json,sys; d=json.load(sys.stdin); print(f\"{d['$task']['mean']:.3f}\")")
    echo "| \`${task}\` | ${label} | ${rand_mean} | ${LLM_MEANS[$task]:-n/a} | ${LLM_MINS[$task]:-n/a} | ${LLM_MAXS[$task]:-n/a} | ${LLM_SUCCESSES[$task]:-n/a} |"
  done
  echo ""
  echo "*Generated by \`run_baselines_v2.sh\` on $(date -u +%Y-%m-%dT%H:%M:%SZ).*"
} | tee "$OUT"

echo ""
echo "Done. Markdown table written to: $OUT"
