"""Generate synthetic training-run reports for each failure mode."""

import math
import random
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .failure_taxonomy import FAILURE_MODES, modes_for_difficulty


@dataclass
class RunReport:
    run_id: str
    incident_header: str
    loss_curve: List[float]
    grad_norms: Dict[str, List[float]]
    hyperparams: Dict[str, object]
    dataset_stats: Dict[str, object]
    error_log: List[str]
    true_failure_mode: str
    secondary_failure_mode: Optional[str] = None


class RunReportSynthesizer:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def _base_hparams(self) -> Dict[str, object]:
        return {
            "learning_rate": 1e-3, "batch_size": 64, "optimizer": "adam",
            "weight_decay": 1e-4, "init": "xavier", "activation": "relu",
            "grad_clip": None, "lr_schedule": "constant", "dropout": 0.1,
            "epochs": 50, "input_normalization": "standardize",
            "bn_track_running_stats": True, "discriminator_lr": 1e-4,
            "data_split": "random",
        }

    def _base_dataset_stats(self) -> Dict[str, object]:
        return {
            "num_train": 50000, "num_val": 10000, "num_classes": 100,
            "input_mean": [0.0, 0.0, 0.0], "input_std": [1.0, 1.0, 1.0],
            "label_balance": "uniform", "feature_label_correlation_max": 0.04,
        }

    def _make_curve_lr_too_high(self, length):
        base = 4.5
        return [round(base + 2.0 * math.sin(i * 0.6) + self.np_rng.normal(0, 0.7), 4)
                for i in range(length)]

    def _make_curve_nan_explosion(self, length):
        out = []
        for i in range(length):
            if i < 30:
                out.append(round(4.5 - 0.02 * i + self.np_rng.normal(0, 0.1), 4))
            elif i < 50:
                out.append(round(4.0 + (i - 30) * 2.5, 4))
            else:
                out.append(float("nan"))
                break
        while len(out) < length:
            out.append(float("nan"))
        return out

    def _make_curve_lr_too_low(self, length):
        return [round(4.6 - 0.0005 * i + self.np_rng.normal(0, 0.02), 4)
                for i in range(length)]

    def _make_curve_label_leakage(self, length):
        out = []
        for i in range(length):
            if i < 20:
                out.append(round(4.5 - 0.22 * i + self.np_rng.normal(0, 0.05), 4))
            else:
                out.append(round(0.05 + self.np_rng.normal(0, 0.01), 4))
        return out

    def _make_curve_distribution_shift(self, length):
        return [round(2.0 - 0.001 * i + self.np_rng.normal(0, 0.05), 4)
                for i in range(length)]

    def _make_curve_vanishing(self, length):
        return [round(4.6 + self.np_rng.normal(0, 0.02), 4) for _ in range(length)]

    def _make_curve_exploding(self, length):
        out = []
        for i in range(length):
            if self.rng.random() < 0.04:
                out.append(round(50 + self.np_rng.normal(0, 5), 4))
            else:
                out.append(round(2.0 - 0.001 * i + self.np_rng.normal(0, 0.1), 4))
        return out

    def _make_curve_default_diverging(self, length):
        return [round(3.0 + 0.001 * i + self.np_rng.normal(0, 0.1), 4)
                for i in range(length)]

    def _make_curve(self, mode, length):
        return {
            "lr_too_high": self._make_curve_lr_too_high,
            "nan_explosion": self._make_curve_nan_explosion,
            "lr_too_low": self._make_curve_lr_too_low,
            "label_leakage": self._make_curve_label_leakage,
            "distribution_shift": self._make_curve_distribution_shift,
            "vanishing_gradients": self._make_curve_vanishing,
            "exploding_gradients": self._make_curve_exploding,
        }.get(mode, self._make_curve_default_diverging)(length)

    def _make_grad_norms(self, mode, length):
        layers = ["layer_0", "layer_4", "layer_8", "layer_12"]
        out = {}
        if mode == "vanishing_gradients":
            for i, lname in enumerate(layers):
                base = 10 ** (-i - 2)
                out[lname] = [round(base + self.np_rng.normal(0, base * 0.1), 6)
                              for _ in range(length)]
        elif mode == "exploding_gradients":
            for lname in layers:
                seq = []
                for j in range(length):
                    if self.rng.random() < 0.03:
                        seq.append(round(self.np_rng.uniform(500, 2000), 2))
                    else:
                        seq.append(round(self.np_rng.uniform(0.1, 1.0), 4))
                out[lname] = seq
        elif mode == "dead_relus":
            for lname in layers:
                out[lname] = [round(self.np_rng.uniform(0.0, 0.05), 5)
                              for _ in range(length)]
        else:
            for lname in layers:
                out[lname] = [round(self.np_rng.uniform(0.1, 1.5), 4)
                              for _ in range(length)]
        return out

    def _hparams_for_mode(self, mode):
        h = self._base_hparams()
        if mode == "lr_too_high": h["learning_rate"] = 0.5
        elif mode == "nan_explosion":
            h["learning_rate"] = 1.0; h["grad_clip"] = None
        elif mode == "lr_too_low": h["learning_rate"] = 1e-7
        elif mode == "exploding_gradients": h["grad_clip"] = None
        elif mode == "vanishing_gradients":
            h["init"] = "xavier"; h["activation"] = "sigmoid"
        elif mode == "dead_relus": h["activation"] = "relu"
        elif mode == "batchnorm_eval_bug": h["bn_track_running_stats"] = False
        elif mode == "oom_crash": h["batch_size"] = 1024
        elif mode == "bad_normalization": h["input_normalization"] = "none"
        return h

    def _dataset_stats_for_mode(self, mode):
        d = self._base_dataset_stats()
        if mode == "label_leakage":
            d["feature_label_correlation_max"] = 0.97
        elif mode == "distribution_shift":
            d["train_val_distance_kl"] = 1.4
        elif mode == "bad_normalization":
            d["input_mean"] = [127.5, 127.5, 127.5]
            d["input_std"] = [80.0, 80.0, 80.0]
        return d

    def _error_log_for_mode(self, mode):
        base = [
            "INFO  Starting training run",
            "INFO  Loading dataset",
            "INFO  Model has 12.4M parameters",
            "INFO  Beginning epoch 1",
        ]
        if mode == "nan_explosion":
            base += ["WARN  Step 47: loss is nan",
                     "ERROR  Aborting training: non-finite loss"]
        elif mode == "oom_crash":
            base += ["ERROR  RuntimeError: CUDA out of memory. Tried to allocate 8.4 GiB",
                     "ERROR  Training aborted"]
        elif mode == "exploding_gradients":
            base += ["WARN  Gradient norm spike at step 213: 1422.7"]
        return base

    def _incident_header(self, mode, length):
        templates = {
            "nan_explosion": "Loss reached NaN at step ~50 of training run on CIFAR-100",
            "lr_too_high": "Loss is oscillating wildly and not converging on a ResNet-18 ImageNet experiment",
            "lr_too_low": f"Training has run for {length} steps but loss has barely moved on a vision transformer",
            "label_leakage": "Train accuracy is 99% but validation accuracy is at chance — something is wrong",
            "distribution_shift": "Train loss is decreasing but validation loss is climbing and the gap is widening",
            "vanishing_gradients": "Loss is flat after epoch 1 on a 24-layer MLP",
            "exploding_gradients": "Training is mostly fine but I see occasional huge loss spikes that ruin runs",
            "dead_relus": "Model seems to have stopped learning halfway through; many neurons look inactive",
            "batchnorm_eval_bug": "Train accuracy is great but eval accuracy is much worse than I expect",
            "oom_crash": "Training crashed early — runtime error in the log",
            "bad_normalization": "Loss starts unusually high and decreases very slowly from there",
            "mode_collapse": "GAN generator outputs look identical across the batch — diversity has collapsed",
        }
        return templates.get(mode, "Training is failing for an unknown reason")

    def generate(self, mode: str, length: Optional[int] = None) -> RunReport:
        if length is None:
            length = 100
        return RunReport(
            run_id=str(uuid.uuid4())[:8],
            incident_header=self._incident_header(mode, length),
            loss_curve=self._make_curve(mode, length),
            grad_norms=self._make_grad_norms(mode, length),
            hyperparams=self._hparams_for_mode(mode),
            dataset_stats=self._dataset_stats_for_mode(mode),
            error_log=self._error_log_for_mode(mode),
            true_failure_mode=mode,
        )

    def generate_for_task(self, task_id: str) -> RunReport:
        if task_id == "obvious_failure_diagnosis":
            mode = self.rng.choice(modes_for_difficulty("easy"))
            return self.generate(mode, length=100)
        elif task_id == "subtle_divergence_diagnosis":
            mode = self.rng.choice(modes_for_difficulty("medium"))
            return self.generate(mode, length=500)
        elif task_id == "adversarial_compound_failure":
            mode = self.rng.choice(modes_for_difficulty("hard"))
            report = self.generate(mode, length=1000)
            secondaries = [m for m in modes_for_difficulty("medium") if m != mode]
            if secondaries:
                report.secondary_failure_mode = self.rng.choice(secondaries)
            return report
        else:
            raise ValueError(f"Unknown task_id: {task_id}")
