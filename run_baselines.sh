#!/usr/bin/env bash
#
# run_baselines.sh — measure MLDoctorEnv baseline scores across all 3 tasks

set -uo pipefail

N_RUNS="${1:-3}"
PORT="${2:-7860}"
HOST="127.0.0.1"
BASE_URL="http://${HOST}:${PORT}"
RESULTS_FILE="baseline_results.md"
LOG_DIR="baseline_logs"

TASKS=(
  "obvious_failure_diagnosis"
  "subtle_divergence_diagnosis"
  "adversarial_compound_failure"
)
TASK_LABELS=(
  "Easy"
  "Medium"
  "Hard"
)

if [ ! -f "inference.py" ]; then
  echo "ERROR: inference.py not found. Run this from the repo root." >&2
  exit 1
fi

if [ ! -f ".env" ]; then
  echo "WARNING: .env not found. Make sure HF_TOKEN, API_BASE_URL, MODEL_NAME"
  echo "         are exported in your shell, or this will fail."
fi

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

MODEL_DISPLAY="${MODEL_NAME:-unknown-model}"

mkdir -p "$LOG_DIR"

kill_uvicorn_on_port() {
  if command -v fuser &>/dev/null; then
    fuser -k "${PORT}/tcp" 2>/dev/null || true
  else
    pkill -f "uvicorn.*--port ${PORT}" 2>/dev/null || true
  fi
  sleep 1
}

start_server_for_task() {
  local task="$1"
  kill_uvicorn_on_port
  echo "  [server] starting MLDoctorEnv for task=${task} on ${BASE_URL} ..."
  MLDOCTOR_TASK="$task" PYTHONPATH=. nohup python -m uvicorn \
    mldoctor_env.server.app:app \
    --host "$HOST" --port "$PORT" \
    > "${LOG_DIR}/server_${task}.log" 2>&1 &
  local pid=$!
  echo "$pid" > "${LOG_DIR}/server_${task}.pid"
  for i in 1 2 3 4 5 6 7 8 9 10; do
    if curl -s --max-time 2 -X POST "${BASE_URL}/reset" \
         -H "Content-Type: application/json" -d '{}' >/dev/null 2>&1; then
      echo "  [server] ready after ${i}s"
      return 0
    fi
    sleep 1
  done
  echo "  [server] FAILED to come up. Check ${LOG_DIR}/server_${task}.log" >&2
  return 1
}

stop_server_for_task() {
  local task="$1"
  if [ -f "${LOG_DIR}/server_${task}.pid" ]; then
    kill "$(cat "${LOG_DIR}/server_${task}.pid")" 2>/dev/null || true
    rm -f "${LOG_DIR}/server_${task}.pid"
  fi
  kill_uvicorn_on_port
}

run_inference_once() {
  local task="$1"
  local run_idx="$2"
  local outfile="$3"
  echo "  [inference] task=${task} run=${run_idx} ..."
  MLDOCTOR_TASK="$task" BASE_URL="$BASE_URL" \
    python inference.py 2>&1 | tee "$outfile"
}

parse_end_line() {
  local log="$1"
  local line
  line=$(grep -E "^\[END\]" "$log" | tail -1)
  if [ -z "$line" ]; then
    echo "false 0 0.000"
    return
  fi
  local success steps score
  success=$(echo "$line" | sed -n 's/.*success=\([a-z]*\).*/\1/p')
  steps=$(echo "$line" | sed -n 's/.*steps=\([0-9]*\).*/\1/p')
  score=$(echo "$line" | sed -n 's/.*score=\([0-9.]*\).*/\1/p')
  echo "${success:-false} ${steps:-0} ${score:-0.000}"
}

run_random_baseline() {
  python3 - <<'PYEOF'
import random, sys, json
sys.path.insert(0, '.')
from mldoctor_env.server.environment import MLDoctorEnvironment
from mldoctor_env.models import MLDoctorAction

ACTIONS = [
    "inspect_loss_curve", "inspect_grad_norms", "inspect_hyperparams",
    "inspect_dataset_stats", "inspect_error_log", "submit",
]
TASKS = [
    "obvious_failure_diagnosis",
    "subtle_divergence_diagnosis",
    "adversarial_compound_failure",
]
TRIALS = 20
results = {}
random.seed(0)
for task in TASKS:
    scores = []
    for trial in range(TRIALS):
        env = MLDoctorEnvironment(task_id=task, seed=trial)
        env.reset()
        total = 0.0
        for _ in range(env.max_steps):
            a = MLDoctorAction(name=random.choice(ACTIONS))
            r = env.step(a)
            total += r.reward
            if r.done:
                break
        scores.append(max(0.0, min(1.0, total)))
    mean = sum(scores) / len(scores)
    lo, hi = min(scores), max(scores)
    results[task] = {"mean": mean, "min": lo, "max": hi}

print(json.dumps(results))
PYEOF
}

echo "================================================================"
echo "  MLDoctorEnv baseline runner"
echo "================================================================"
echo "  N runs per task : $N_RUNS"
echo "  Server port     : $PORT"
echo "  Model           : $MODEL_DISPLAY"
echo "  Logs directory  : $LOG_DIR/"
echo ""

declare -A LLM_MEANS
declare -A LLM_MINS
declare -A LLM_MAXS
declare -A LLM_SUCCESSES

for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  label="${TASK_LABELS[$i]}"
  echo "----------------------------------------------------------------"
  echo " Task ${i}/${#TASKS[@]}: ${task}  (${label})"
  echo "----------------------------------------------------------------"

  start_server_for_task "$task" || { echo "skipping $task"; continue; }

  scores=()
  successes=0
  for run in $(seq 1 "$N_RUNS"); do
    log="${LOG_DIR}/inference_${task}_run${run}.log"
    run_inference_once "$task" "$run" "$log"
    read -r success steps score < <(parse_end_line "$log")
    scores+=("$score")
    if [ "$success" = "true" ]; then
      successes=$((successes + 1))
    fi
    echo "  [parsed] run=${run} success=${success} steps=${steps} score=${score}"
  done

  stop_server_for_task "$task"

  mean_score=$(printf '%s\n' "${scores[@]}" | awk '{s+=$1; n++} END{if(n>0) printf "%.3f", s/n; else print "0.000"}')
  min_score=$(printf '%s\n' "${scores[@]}" | awk 'NR==1 || $1<m {m=$1} END{printf "%.3f", m}')
  max_score=$(printf '%s\n' "${scores[@]}" | awk 'NR==1 || $1>m {m=$1} END{printf "%.3f", m}')

  LLM_MEANS[$task]="$mean_score"
  LLM_MINS[$task]="$min_score"
  LLM_MAXS[$task]="$max_score"
  LLM_SUCCESSES[$task]="${successes}/${N_RUNS}"

  echo ""
  echo "  ${label} summary: mean=${mean_score}  min=${min_score}  max=${max_score}  successes=${successes}/${N_RUNS}"
  echo ""
done

echo "================================================================"
echo "  Random baseline (20 trials per task, no LLM, no HTTP)"
echo "================================================================"
RAND_JSON=$(run_random_baseline)
echo "$RAND_JSON" | python3 -m json.tool

OUT="$RESULTS_FILE"
{
  echo "## Baseline scores"
  echo ""
  echo "Measured by running \`inference.py\` ${N_RUNS} times per task against \`${MODEL_DISPLAY}\` via the Hugging Face router. Random baseline is a uniform-random policy over the action set across 20 trials."
  echo ""
  echo "| Task | Difficulty | Random | ${MODEL_DISPLAY} (mean) | min | max | success rate |"
  echo "|---|---|---|---|---|---|---|"

  for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"
    label="${TASK_LABELS[$i]}"
    rand_mean=$(echo "$RAND_JSON" | python3 -c "import json,sys; print(f\"{json.load(sys.stdin)['$task']['mean']:.3f}\")")
    echo "| \`${task}\` | ${label} | ${rand_mean} | ${LLM_MEANS[$task]:-n/a} | ${LLM_MINS[$task]:-n/a} | ${LLM_MAXS[$task]:-n/a} | ${LLM_SUCCESSES[$task]:-n/a} |"
  done
  echo ""
  echo "*Generated by \`run_baselines.sh\` on $(date -u +%Y-%m-%dT%H:%M:%SZ).*"
} | tee "$OUT"

echo ""
echo "================================================================"
echo "  Done. Markdown table written to: $OUT"
echo "  Per-run logs in: $LOG_DIR/"
echo "  Paste the table into README.md under the 'Baseline scores' heading."
echo "================================================================"
