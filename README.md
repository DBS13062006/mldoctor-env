---
title: MLDoctorEnv
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - ml-debugging
  - agentic-rl
license: bsd-3-clause
---

# MLDoctorEnv 🩺 — diagnose failing ML training runs

> An OpenEnv environment that turns the universal "why is my training broken?" debugging loop into an agentic task. Built for the OpenEnv Hackathon Round 1.

## What this environment does

Every ML practitioner has stared at a TensorBoard loss curve gone haywire and asked "why?" The diagnosis process is structured: look at the loss curve, look at the gradient norms, check the hyperparameters, scan the error log, form a hypothesis, optionally try a small ablation, then prescribe a fix.

MLDoctorEnv turns this loop into an OpenEnv environment where an LLM agent has to do the diagnosis on synthesized run reports. Because we synthesized the runs ourselves, we know the ground-truth root cause, so the grader is fully deterministic.

## Action space

A single discriminated `MLDoctorAction` with a `name` and free-form `args`:

| Action name | Args | Effect |
|---|---|---|
| `inspect_loss_curve` | `{window_start?, window_end?}` | Returns a slice of the loss curve. |
| `inspect_grad_norms` | `{layer?}` | Returns gradient norms by layer. |
| `inspect_hyperparams` | `{key?}` | Returns hyperparameters. |
| `inspect_dataset_stats` | `{}` | Returns dataset statistics. |
| `inspect_error_log` | `{last_n?}` | Returns the last N lines of the training stdout. |
| `hypothesize` | `{failure_mode}` | Commit to a failure category. |
| `request_ablation` | `{config_change}` | Simulate a config change and observe its effect. |
| `prescribe` | `{failure_mode, config_diff}` | Commit to a category + minimal fix. |
| `submit` | `{}` | End the episode with current state. |

Valid `failure_mode` values: `nan_explosion`, `vanishing_gradients`, `exploding_gradients`, `mode_collapse`, `dead_relus`, `label_leakage`, `distribution_shift`, `lr_too_high`, `lr_too_low`, `batchnorm_eval_bug`, `oom_crash`, `bad_normalization`.

## Reward function

> **Design philosophy.** Dense, per-step, deterministic, asymmetric, and capped. The reward is a *teaching signal* — it tells an agent at every step whether its most recent action moved it closer to a correct diagnosis, not just whether the final answer was right.

### Why dense, not sparse

A sparse end-of-episode reward (e.g. +1 if the prescription is correct, 0 otherwise) would technically work, but has three problems we explicitly designed against:

1. **Credit assignment is impossible.** With 15 steps in the hard task and a single binary reward at the end, an RL agent has no signal to learn *which* of its 15 actions was the good one. Dense per-step rewards solve this directly.
2. **Random and broken agents look identical.** Both score 0. We want a noise floor that distinguishes "agent that explores rationally and fails at the final answer" from "agent that flails randomly." The +0.05-per-inspection signal gives exploring agents a non-zero floor (~0.10 in our measured baseline) while leaving plenty of headroom for prescription rewards.
3. **It produces flat learning curves.** RL training on sparse signals is notoriously slow. Since this env is intended for *post-training research* (the "RL" in OpenEnv), we want it to be trainable, not just evaluable. Dense rewards make it usable as a training environment, not just a benchmark.

### The reward table (with rationale for each line)

| Event | Reward | Why this number |
|---|---|---|
| **First-time inspection** (5 distinct types, each fires once max) | **+0.05** | Rewards information-gathering. Caps at +0.25 total so the agent can't farm inspections forever. 0.05 is small enough that an inspection-only agent hits a ceiling well below the 0.5 success threshold. |
| **Redundant inspection** (same type called twice) | **−0.02** | Discourages the most common LLM failure mode on long-context evidence tasks: re-reading the same artifact instead of moving forward. Penalty magnitude is smaller than the original reward (0.02 < 0.05) so an *accidental* repeat doesn't destroy the run — it just costs ground. |
| **Correct hypothesis** | **+0.15** | Hypothesizing is a *commitment*. We reward it more than an inspection because it requires synthesis across the gathered evidence. |
| **Wrong hypothesis** | **−0.05** | Asymmetric on purpose. We *want* the agent to hypothesize even when uncertain, because hypothesizing forces structured thinking. A symmetric ±0.15 would discourage hypothesis attempts entirely. The 3:1 reward-to-penalty ratio is calibrated to encourage hypothesizing when confidence > ~25%. |
| **Useful ablation** (config_change touches the true root cause's `fix_key`) | **+0.10** | Rewards the scientific method. An agent that says "let me try lowering the learning rate and see what happens" before committing is doing exactly what a senior ML engineer does. |
| **Useless ablation** (config_change touches an unrelated key) | **0** (not negative) | Exploratory ablations are not penalised. Trying the wrong fix is a normal part of debugging — the *implicit cost* is the step it consumed from the budget, not an explicit penalty. |
| **Correct prescription category** | **+0.30** | The biggest single reward. Picking the right category requires correctly synthesizing all gathered evidence into one of 12 named failure modes. |
| **Correct prescription config diff** (right key + right direction/value) | **+0.20** | A two-stage prescription reward: category first (+0.30), correct fix second (+0.20). A model that knows *what's* wrong but not *how to fix it* still gets partial credit, which is realistic — diagnosis and treatment are different skills in real ML engineering. |
| **Wrong prescription** | **−0.10** | A wrong prescription is the most committed-yet-incorrect action. The penalty is larger than wrong-hypothesis (−0.05) because prescription is a higher-stakes commitment. |
| **Submit without prescribing** | **0** | Voluntary early-exit. Agent locks in whatever cumulative reward it has and ends the episode. |

### Why a perfect episode sums to exactly 1.00

Deliberate, so grader normalization is trivial: `score = clamp(sum(rewards), 0.0, 1.0)`. No `MAX_TOTAL_REWARD` constants to maintain in two places.

The maximum-reward path through the env is:

```
inspect_loss_curve     +0.05
inspect_grad_norms     +0.05
inspect_hyperparams    +0.05
inspect_dataset_stats  +0.05
inspect_error_log      +0.05    (subtotal: +0.25, hits the inspection cap)
hypothesize(correct)   +0.15
request_ablation(useful) +0.10
prescribe(correct)     +0.50    (+0.30 category + +0.20 fix in one call)
                       ─────
                       =1.00
```

The full perfect trajectory uses **8 environment steps** and leaves 2/4/7 step buffer on easy/medium/hard. A high-skill agent doesn't have to race the clock — there is room to think, to try a wrong hypothesis and recover, and to ablate before committing.

### Anti-gaming analysis

We considered the following reward-hacking strategies and confirmed each is blocked. (Numbers below assume the easy task with `max_steps=10` unless noted.)

**1. "Inspect forever and never commit."** Capped at +0.25. Hard ceiling well below the 0.5 success threshold. An inspection-only agent cannot pass.

**2. "Spam inspections of the same type to farm reward."** Each inspection type only pays once. Repeats cost −0.02. Capped.

**3. "Guess every failure mode at random."** Empirically verified: hypothesizing all 12 modes in sequence on the easy task yields cumulative reward ≈ −0.30 and the episode ends from step exhaustion. Strictly worse than not hypothesizing.

**4. "Brute-force prescriptions until one hits."** On the patched env, the episode ends on the **first** `prescribe` action regardless of correctness. The agent gets exactly one shot at the diagnosis. Expected value of a 0-evidence random prescription on the hard task: `(1/12) × 0.50 + (11/12) × (−0.10) ≈ −0.05`. Negative in expectation. The optimal strategy is therefore *gather evidence first*, then prescribe.

**5. "Skip evidence and prescribe immediately if you happen to guess right."** A 0-evidence correct prescription scores `+0.50` and ends the episode. Hit rates: easy (4 eligible modes) → 25%, medium (8) → 12.5%, hard (12) → 8.3%. Expected value across tasks ranges from `+0.025` (easy) to `−0.05` (hard) — at best break-even, at worst losing. The math discourages laziness.

**6. "Submit immediately to lock in zero."** `submit` gives 0 reward and ends the episode. Score = 0. Strictly worse than any reasonable strategy.

**7. "Run useless ablations to fill steps."** Useless ablations give 0 reward but consume the step budget, leaving fewer steps for productive actions. Pure waste — no upside, real cost in opportunity.

### Penalty rationale: why we use them at all

Some RL environments avoid negative rewards on the theory that they cause training instability. We use them for three specific reasons:

- **They make the noise floor non-trivial.** Without penalties, a do-nothing agent and a do-the-wrong-thing agent are indistinguishable. We want signal *between* failures, not just between success and failure.
- **They model real-world cost.** Wrong prescriptions in real ML engineering waste hours of debugging time. Encoding that as −0.10 makes the env honest.
- **They keep the score range bounded but informative.** Total reward is bounded in roughly [−0.5, 1.0]. The grader clips to [0, 1], so penalties shape behavior without producing scores below the floor.

### Episode boundaries

The episode ends when **any** of these fires:

| Trigger | Reason |
|---|---|
| `submit` action called | Agent voluntarily ends the episode. |
| `prescribe` action called | One-shot diagnosis: the agent commits and the episode ends. |
| `step_count >= max_steps` | Step budget exhausted (10 / 12 / 15 for easy / medium / hard). |

The cumulative reward is summed across the entire trajectory and clipped to `[0.0, 1.0]` by the grader. Whatever sub-1.0 score the agent has accumulated at the end is its final score.

### Difficulty interaction

The reward function is **identical** for all three tasks. The difficulty laddering comes from the **environment**, not the reward shaping:

| Knob | Easy | Medium | Hard |
|---|---|---|---|
| Number of failure modes possible | 4 | 8 | 12 |
| Loss curve length | 100 | 500 | 1000 |
| Has secondary masking failure | no | no | yes |
| Has red-herring hyperparams | no | no | yes |
| Max steps | 10 | 12 | 15 |

Same reward function, harder world. This is intentional: it lets a researcher train an agent on the easy task and evaluate it on the hard task without worrying that scores aren't comparable across tasks.

### Alternatives we considered and rejected

| Alternative | Why we rejected it |
|---|---|
| **Sparse end-of-episode reward** (+1 if prescription correct, 0 else) | No credit assignment, no signal during exploration, no distinction between random and broken agents. Hard to train on, hard to debug. |
| **LLM-as-judge reward** | Non-deterministic, expensive at evaluation time, vulnerable to prompt injection from the agent's own outputs, and the hackathon explicitly wants reproducible scores. |
| **Reward shaped from human demonstrations** | Would need a corpus of expert traces. Too expensive for a 48-hour hackathon and reduces reproducibility. |
| **Continuous reward based on distance to ground truth** (cosine similarity between prescribed config diff and true config diff) | Adds floating-point noise, harder to interpret, and the discrete category-correct + fix-correct decomposition is closer to how a senior engineer would actually grade a junior. |
| **Dense reward only, no penalties** | Loses anti-gaming guarantees. Random and committed-but-wrong agents become indistinguishable. |
| **Multi-shot prescription with retry penalty** | We tested this — without the one-shot patch, `prescribe` doesn't end the episode and each wrong attempt only costs −0.10. The expected value of brute-force prescription got too close to break-even. One-shot prescription is cleaner, matches real-world cost, and produces a sharper success signal. |

### Anti-gaming guarantees enforced by tests

Every claim in this section is asserted by [tests/test_reward_invariants.py](tests/test_reward_invariants.py) (33 tests across 5 test classes, mirroring the structure of this section). Run `pytest tests/test_reward_invariants.py -v` to verify. The full suite (`pytest tests/ -v`) runs **38 tests** in under 5 seconds.

### Reward function in code

The full reward logic lives in `mldoctor_env/server/environment.py` and is implemented as small, named methods (`_do_inspection`, `_do_hypothesize`, `_do_ablation`, `_do_prescribe`) so each event's reward is auditable in one place. The reward returned by `step()` is rounded to 4 decimal places before being sent to the agent, which keeps `[STEP]` log lines clean and makes test assertions stable.

## Tasks

| Task | Difficulty | Max steps |
|---|---|---|
| `obvious_failure_diagnosis` | easy | 10 |
| `subtle_divergence_diagnosis` | medium | 12 |
| `adversarial_compound_failure` | hard | 15 |

## Baseline scores

Measured by running `inference.py` 3 times per task against `Qwen/Qwen2.5-72B-Instruct` via the Hugging Face router. Random baseline is a uniform-random policy over the action set across 20 trials.

| Task | Difficulty | Random | Qwen2.5-72B (mean) | min | max | success rate |
|---|---|---|---|---|---|---|
| `obvious_failure_diagnosis` | Easy | 0.092 | 0.517 | 0.000 | 0.800 | 2/3 |
| `subtle_divergence_diagnosis` | Medium | 0.090 | 0.523 | 0.140 | 0.950 | 1/3 |
| `adversarial_compound_failure` | Hard | 0.102 | 0.253 | 0.020 | 0.720 | 1/3 |

Generated via `run_baselines.sh`. Per-run logs in `baseline_logs/`.

## Setup

```bash
pip install -e .
uvicorn mldoctor_env.server.app:app --host 0.0.0.0 --port 7860
```

Or Docker:

```bash
docker build -t mldoctor-env .
docker run --rm -p 7860:7860 mldoctor-env
```

## Running baseline inference

```bash
export HF_TOKEN=hf_xxx
export BASE_URL=http://localhost:7860
export MLDOCTOR_TASK=obvious_failure_diagnosis
python inference.py
```

If `HF_TOKEN` is not set, `inference.py` will prompt for it interactively and persist it to a local `.env` file.
