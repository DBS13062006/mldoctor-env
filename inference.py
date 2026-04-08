"""
Baseline inference script for MLDoctorEnv.

Reads from environment variables or a local `.env` file:
  HF_TOKEN        — Hugging Face access token (required; starts with hf_...)
  API_BASE_URL    — OpenAI-compatible endpoint (default: HF router)
  MODEL_NAME      — model id (default: Qwen/Qwen2.5-72B-Instruct)
  BASE_URL        — MLDoctorEnv server URL (default: http://localhost:7860)
  MLDOCTOR_TASK   — task id (default: obvious_failure_diagnosis)
  MLDOCTOR_MAX_STEPS — step cap (default: 15)

If HF_TOKEN is not set, the script will prompt for it interactively and
persist it to `.env` for future runs.

Emits the strict [START] / [STEP] / [END] format required by the hackathon.
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import time
from getpass import getpass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

try:
    from openai import OpenAI
except ImportError as exc:
    print(f"[FATAL] openai package not installed: {exc}", file=sys.stderr)
    sys.exit(2)


# --- .env loader -------------------------------------------------------------

def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        # Do not override variables already present in the environment.
        os.environ.setdefault(k, v)


_load_dotenv()


def _persist_to_dotenv(key: str, value: str) -> None:
    env_path = Path(".env")
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    if f"{key}=" in existing:
        return
    with env_path.open("a", encoding="utf-8") as f:
        if existing and not existing.endswith("\n"):
            f.write("\n")
        f.write(f"{key}={value}\n")


def _require_hf_token() -> str:
    tok = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    if tok:
        return tok
    print("HF_TOKEN not found in environment or .env file.", flush=True)
    try:
        tok = getpass("Paste your Hugging Face token (hf_...): ").strip()
    except (EOFError, KeyboardInterrupt):
        tok = ""
    if not tok:
        print("[FATAL] HF_TOKEN is required.", file=sys.stderr)
        sys.exit(2)
    _persist_to_dotenv("HF_TOKEN", tok)
    os.environ["HF_TOKEN"] = tok
    return tok


# --- Configuration -----------------------------------------------------------

API_KEY      = _require_hf_token()
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"
BASE_URL     = (os.getenv("BASE_URL")    or "http://localhost:7860").rstrip("/")
TASK_NAME    = os.getenv("MLDOCTOR_TASK", "obvious_failure_diagnosis")
BENCHMARK    = "mldoctor_env"
MAX_STEPS    = int(os.getenv("MLDOCTOR_MAX_STEPS", "15"))
TEMPERATURE  = float(os.getenv("MLDOCTOR_TEMPERATURE", "0.2"))
MAX_TOKENS   = int(os.getenv("MLDOCTOR_MAX_TOKENS", "400"))


VALID_ACTIONS = {
    "inspect_loss_curve", "inspect_grad_norms", "inspect_hyperparams",
    "inspect_dataset_stats", "inspect_error_log", "hypothesize",
    "request_ablation", "prescribe", "submit",
}
VALID_MODES = {
    "nan_explosion", "vanishing_gradients", "exploding_gradients",
    "mode_collapse", "dead_relus", "label_leakage", "distribution_shift",
    "lr_too_high", "lr_too_low", "batchnorm_eval_bug", "oom_crash",
    "bad_normalization",
}


# --- Stdout helpers (exact format required by hackathon) ---------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# --- Prompt construction -----------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert ML engineer debugging a failing training run. Your job is
to diagnose the root cause and prescribe a minimal config fix.

You can take actions ONE AT A TIME. On each turn, output a single JSON object
with this exact schema:

  {"name": "<action_name>", "args": {<key>: <value>, ...}}

Available actions:
  - inspect_loss_curve         args: {window_start?: int, window_end?: int}
  - inspect_grad_norms         args: {layer?: "all"|<layer_name>}
  - inspect_hyperparams        args: {key?: <hparam_name>}
  - inspect_dataset_stats      args: {}
  - inspect_error_log          args: {last_n?: int}
  - hypothesize                args: {failure_mode: <category>}
  - request_ablation           args: {config_change: {<key>: <value>}}
  - prescribe                  args: {failure_mode: <category>,
                                       config_diff: {<key>: <value>}}
  - submit                     args: {}

The ONLY valid failure_mode categories are:
  nan_explosion, vanishing_gradients, exploding_gradients, mode_collapse,
  dead_relus, label_leakage, distribution_shift, lr_too_high, lr_too_low,
  batchnorm_eval_bug, oom_crash, bad_normalization

Strategy:
  1. Read the incident header carefully.
  2. Inspect 2-4 different evidence types to form a hypothesis.
  3. Call hypothesize once you're confident.
  4. Optionally request_ablation to confirm.
  5. Call prescribe with the correct category AND a config_diff that matches
     the canonical fix (e.g. {"learning_rate": 1e-4} for nan_explosion).
  6. Call submit.

Output ONLY a single JSON object. No prose. No code fences. No commentary.
""").strip()


def build_user_prompt(observation: Dict[str, Any], history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "(none yet)"
    return textwrap.dedent(f"""
        Task: {observation.get('task_id')}  (difficulty: {observation.get('task_difficulty')})
        Incident: {observation.get('incident_header')}

        Evidence already gathered: {observation.get('inspections_made')}
        Current hypothesis: {observation.get('current_hypothesis')}
        Step {observation.get('step_count')} of {observation.get('max_steps')}.

        Last action result:
        {observation.get('last_action_result')}

        Recent action history:
        {history_block}

        Your next single-action JSON:
    """).strip()


# --- Robust JSON extraction --------------------------------------------------

def _extract_json_object(text: str) -> Optional[str]:
    """Find the first balanced {...} JSON object in text."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def parse_model_action(text: str) -> Dict[str, Any]:
    fallback = {"name": "inspect_loss_curve", "args": {}}
    blob = _extract_json_object(text or "")
    if not blob:
        return fallback
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return fallback
    if not isinstance(obj, dict):
        return fallback
    name = obj.get("name") or obj.get("action") or "inspect_loss_curve"
    if name not in VALID_ACTIONS:
        return fallback
    args = obj.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    # Validate failure_mode where relevant.
    if name in ("hypothesize", "prescribe"):
        fm = args.get("failure_mode")
        if fm not in VALID_MODES:
            return fallback
    return {"name": name, "args": args}


# --- Model client ------------------------------------------------------------

def get_model_action(client: OpenAI, observation: Dict[str, Any],
                     history: List[str]) -> Dict[str, Any]:
    user_prompt = build_user_prompt(observation, history)
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            text = (completion.choices[0].message.content or "").strip()
            return parse_model_action(text)
        except Exception as exc:
            last_exc = exc
            time.sleep(1.0 * (attempt + 1))
    print(f"[DEBUG] Model request failed after retries: {last_exc}", flush=True)
    return {"name": "inspect_loss_curve", "args": {}}


def action_to_str(a: Dict[str, Any]) -> str:
    args = a.get("args") or {}
    if not args:
        return f"{a['name']}()"
    parts = []
    for k, v in args.items():
        if isinstance(v, (dict, list)):
            parts.append(f"{k}={json.dumps(v, separators=(',', ':'))}")
        else:
            parts.append(f"{k}={v}")
    return f"{a['name']}({','.join(parts)})"


# --- HTTP env client ---------------------------------------------------------

class HttpEnv:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.Client(base_url=base_url, timeout=60.0)

    def wait_ready(self, timeout: float = 30.0) -> None:
        deadline = time.time() + timeout
        last: Optional[Exception] = None
        while time.time() < deadline:
            try:
                r = self.client.post("/reset", json={})
                if r.status_code == 200:
                    return
                last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            except Exception as exc:
                last = exc
            time.sleep(1.0)
        raise RuntimeError(f"Env server at {self.base_url} not ready: {last}")

    def reset(self) -> Dict[str, Any]:
        r = self.client.post("/reset", json={})
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        r = self.client.post("/step", json={"action": action})
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self.client.close()


# --- Main loop ---------------------------------------------------------------

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False
    final_error: Optional[str] = None

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    env = HttpEnv(BASE_URL)
    try:
        env.wait_ready(timeout=30.0)
        result = env.reset()
        observation = result.get("observation", {}) or {}

        for step in range(1, MAX_STEPS + 1):
            if result.get("done"):
                break

            action = get_model_action(client, observation, history)
            step_error: Optional[str] = None
            try:
                result = env.step(action)
            except Exception as exc:
                step_error = f"step_failed:{exc}"
                log_step(step=step, action=action_to_str(action),
                         reward=0.0, done=True, error=step_error)
                final_error = step_error
                break

            observation = result.get("observation", {}) or {}
            reward = float(result.get("reward") or 0.0)
            done = bool(result.get("done"))

            rewards.append(reward)
            steps_taken = step
            history.append(
                f"step {step}: {action_to_str(action)} -> r={reward:+.2f}"
            )

            log_step(step=step, action=action_to_str(action),
                     reward=reward, done=done, error=None)

            if done:
                break

        score = max(0.0, min(1.0, sum(rewards)))
        success = score >= 0.5

    except Exception as exc:
        final_error = str(exc)
        print(f"[DEBUG] Episode error: {exc}", flush=True)
    finally:
        env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        if final_error:
            print(f"[DEBUG] final_error={final_error}", flush=True)


if __name__ == "__main__":
    main()
