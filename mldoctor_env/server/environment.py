"""The MLDoctorEnv environment — reset, step, state."""

import math
import uuid
from typing import Any, Dict, Optional

try:
    from openenv.core.environment import Environment
except ImportError:
    try:
        from openenv.environment import Environment  # type: ignore
    except ImportError:
        class Environment:  # type: ignore
            pass

from ..failure_taxonomy import FAILURE_MODES, is_correct_prescription
from ..models import MLDoctorAction, MLDoctorObservation, MLDoctorState, StepResult
from ..synthesizer import RunReportSynthesizer, RunReport

INSPECTION_ACTIONS = {
    "inspect_loss_curve",
    "inspect_grad_norms",
    "inspect_hyperparams",
    "inspect_dataset_stats",
    "inspect_error_log",
}


class MLDoctorEnvironment(Environment):
    DEFAULT_MAX_STEPS = {
        "obvious_failure_diagnosis": 10,
        "subtle_divergence_diagnosis": 12,
        "adversarial_compound_failure": 15,
    }
    DIFFICULTY = {
        "obvious_failure_diagnosis": "easy",
        "subtle_divergence_diagnosis": "medium",
        "adversarial_compound_failure": "hard",
    }

    def __init__(self, task_id: str = "obvious_failure_diagnosis", seed: int = 0):
        self.task_id = task_id
        self.synth = RunReportSynthesizer(seed=seed)
        self.max_steps = self.DEFAULT_MAX_STEPS[task_id]
        self.report: Optional[RunReport] = None
        self.state: Optional[MLDoctorState] = None
        self.inspections_made: set = set()
        self.last_action_result: str = ""
        self.current_hypothesis: Optional[str] = None
        self.prescribed = False
        self.submitted = False

    def reset(self) -> StepResult:
        self.report = self.synth.generate_for_task(self.task_id)
        self.state = MLDoctorState(
            task_id=self.task_id,
            episode_id=str(uuid.uuid4())[:8],
        )
        self.inspections_made = set()
        self.last_action_result = (
            "Run report loaded. Use inspect_* actions to gather evidence."
        )
        self.current_hypothesis = None
        self.prescribed = False
        self.submitted = False
        return StepResult(observation=self._observe(), reward=0.0, done=False)

    def step(self, action: MLDoctorAction) -> StepResult:
        assert self.state is not None and self.report is not None, "call reset()"
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}

        name = action.name
        args = action.args or {}

        if name in INSPECTION_ACTIONS:
            reward += self._do_inspection(name, args)
        elif name == "hypothesize":
            reward += self._do_hypothesize(args)
        elif name == "request_ablation":
            reward += self._do_ablation(args)
        elif name == "prescribe":
            r, _correct = self._do_prescribe(args)
            reward += r
            done = True   # one-shot diagnosis: prescribe always ends the episode
        elif name == "submit":
            done = True
            self.submitted = True
            self.last_action_result = (
                "Episode submitted with current hypothesis and prescription."
            )
        else:
            self.last_action_result = f"Unknown action: {name}"
            reward -= 0.05

        self.state.step_count += 1
        self.state.cumulative_reward += reward
        if self.state.step_count >= self.max_steps:
            done = True

        return StepResult(observation=self._observe(),
                          reward=round(reward, 4), done=done, info=info)

    def state_(self) -> MLDoctorState:
        return self.state  # type: ignore

    def _observe(self) -> MLDoctorObservation:
        return MLDoctorObservation(
            task_id=self.task_id,
            task_difficulty=self.DIFFICULTY[self.task_id],
            incident_header=self.report.incident_header if self.report else "",
            last_action_result=self.last_action_result,
            inspections_made=sorted(self.inspections_made),
            current_hypothesis=self.current_hypothesis,
            prescribed=self.prescribed,
            submitted=self.submitted,
            step_count=self.state.step_count if self.state else 0,
            max_steps=self.max_steps,
        )

    def _do_inspection(self, name: str, args: Dict[str, Any]) -> float:
        assert self.report is not None
        first_time = name not in self.inspections_made
        self.inspections_made.add(name)
        reward = 0.05 if first_time else -0.02

        if name == "inspect_loss_curve":
            ws = int(args.get("window_start", 0))
            we = int(args.get("window_end", -1))
            curve = self.report.loss_curve
            slice_ = curve[ws:] if we == -1 else curve[ws:we]
            def fmt(x):
                return f"{x:.4f}" if isinstance(x, float) and not math.isnan(x) else "nan"
            head = ", ".join(fmt(x) for x in slice_[:8])
            tail = ", ".join(fmt(x) for x in slice_[-8:])
            finite = [x for x in slice_ if isinstance(x, float) and not math.isnan(x)]
            mn = min(finite) if finite else float('nan')
            mx = max(finite) if finite else float('nan')
            self.last_action_result = (
                f"Loss curve ({len(slice_)} pts). First 8: [{head}]  Last 8: [{tail}]. "
                f"Min={mn:.4f}, Max={mx:.4f}"
            )

        elif name == "inspect_grad_norms":
            layer = args.get("layer", "all")
            gn = self.report.grad_norms
            if layer == "all":
                summary = ", ".join(
                    f"{k}: mean={sum(v)/len(v):.4f}" for k, v in gn.items()
                )
                self.last_action_result = f"Grad norms by layer: {summary}"
            else:
                seq = gn.get(layer, [])
                self.last_action_result = (
                    f"Grad norms for {layer}: first 5 = {seq[:5]}, "
                    f"max = {max(seq) if seq else 'n/a'}"
                )

        elif name == "inspect_hyperparams":
            key = args.get("key")
            hp = self.report.hyperparams
            if key:
                self.last_action_result = f"Hyperparam {key} = {hp.get(key)}"
            else:
                self.last_action_result = f"Hyperparams: {hp}"

        elif name == "inspect_dataset_stats":
            self.last_action_result = f"Dataset stats: {self.report.dataset_stats}"

        elif name == "inspect_error_log":
            n = int(args.get("last_n", 20))
            lines = self.report.error_log[-n:]
            self.last_action_result = "Error log: " + " | ".join(lines)

        return reward

    def _do_hypothesize(self, args: Dict[str, Any]) -> float:
        assert self.report is not None
        guess = args.get("failure_mode", "")
        self.current_hypothesis = guess
        if guess == self.report.true_failure_mode:
            self.last_action_result = f"Hypothesis '{guess}' recorded. (correct)"
            return 0.15
        else:
            self.last_action_result = f"Hypothesis '{guess}' recorded. (incorrect)"
            return -0.05

    def _do_ablation(self, args: Dict[str, Any]) -> float:
        assert self.report is not None
        change = args.get("config_change", {}) or {}
        true_mode = self.report.true_failure_mode
        spec = FAILURE_MODES[true_mode]
        target_key = spec["fix_key"]
        if target_key in change:
            self.last_action_result = (
                f"Ablation simulating {change}: loss curve responds positively. "
                f"This change appears to address the root cause."
            )
            return 0.10
        else:
            self.last_action_result = (
                f"Ablation simulating {change}: no meaningful change observed."
            )
            return 0.0

    def _do_prescribe(self, args: Dict[str, Any]) -> tuple:
        assert self.report is not None
        mode = args.get("failure_mode", "")
        diff = args.get("config_diff", {}) or {}
        cat_ok, cfg_ok = is_correct_prescription(
            self.report.true_failure_mode, mode, diff
        )
        self.prescribed = True
        reward = 0.0
        if cat_ok:
            reward += 0.30
        else:
            reward -= 0.10
        if cfg_ok:
            reward += 0.20
        finished = cat_ok and cfg_ok
        self.last_action_result = (
            f"Prescription: mode={mode}, diff={diff}. "
            f"Category {'correct' if cat_ok else 'wrong'}, "
            f"config {'correct' if cfg_ok else 'wrong'}."
        )
        return reward, finished
