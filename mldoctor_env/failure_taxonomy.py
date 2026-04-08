"""The 12 failure modes — the spine of the project."""

FAILURE_MODES = {
    "nan_explosion": {
        "cause": "Learning rate way too high; weights blow up to NaN.",
        "signature": ["loss_curve", "error_log"],
        "fix_key": "learning_rate",
        "fix_direction": "down",
        "fix_value": 1e-4,
        "difficulties": ["easy", "medium"],
    },
    "vanishing_gradients": {
        "cause": "Deep network with bad init or sigmoid stack — early-layer grads ~0.",
        "signature": ["grad_norms", "hyperparams"],
        "fix_key": "init",
        "fix_direction": "replace",
        "fix_value": "kaiming",
        "difficulties": ["medium", "hard"],
    },
    "exploding_gradients": {
        "cause": "No gradient clipping; grads occasionally spike.",
        "signature": ["grad_norms", "hyperparams"],
        "fix_key": "grad_clip",
        "fix_direction": "replace",
        "fix_value": 1.0,
        "difficulties": ["easy", "medium"],
    },
    "mode_collapse": {
        "cause": "Generative model collapses to a single output mode.",
        "signature": ["loss_curve", "dataset_stats"],
        "fix_key": "discriminator_lr",
        "fix_direction": "down",
        "fix_value": 1e-5,
        "difficulties": ["hard"],
    },
    "dead_relus": {
        "cause": "Many neurons stuck at 0 because lr was briefly too high.",
        "signature": ["grad_norms", "hyperparams"],
        "fix_key": "activation",
        "fix_direction": "replace",
        "fix_value": "leaky_relu",
        "difficulties": ["medium"],
    },
    "label_leakage": {
        "cause": "A feature is a near-copy of the label.",
        "signature": ["loss_curve", "dataset_stats"],
        "fix_key": "feature_audit",
        "fix_direction": "replace",
        "fix_value": "drop_leaking_feature",
        "difficulties": ["medium", "hard"],
    },
    "distribution_shift": {
        "cause": "Train and val distributions differ.",
        "signature": ["loss_curve", "dataset_stats"],
        "fix_key": "data_split",
        "fix_direction": "replace",
        "fix_value": "stratified_reshuffle",
        "difficulties": ["medium", "hard"],
    },
    "lr_too_high": {
        "cause": "LR above stable region but not infinity.",
        "signature": ["loss_curve", "hyperparams"],
        "fix_key": "learning_rate",
        "fix_direction": "down",
        "fix_value": 1e-3,
        "difficulties": ["easy"],
    },
    "lr_too_low": {
        "cause": "LR too small, almost no learning.",
        "signature": ["loss_curve", "hyperparams"],
        "fix_key": "learning_rate",
        "fix_direction": "up",
        "fix_value": 1e-3,
        "difficulties": ["easy"],
    },
    "batchnorm_eval_bug": {
        "cause": "BN running stats unreliable in eval mode.",
        "signature": ["loss_curve", "hyperparams"],
        "fix_key": "bn_track_running_stats",
        "fix_direction": "replace",
        "fix_value": True,
        "difficulties": ["hard"],
    },
    "oom_crash": {
        "cause": "Batch size or model too big for memory.",
        "signature": ["error_log", "hyperparams"],
        "fix_key": "batch_size",
        "fix_direction": "down",
        "fix_value": 16,
        "difficulties": ["easy"],
    },
    "bad_normalization": {
        "cause": "Input features not normalized to unit scale.",
        "signature": ["loss_curve", "dataset_stats"],
        "fix_key": "input_normalization",
        "fix_direction": "replace",
        "fix_value": "standardize",
        "difficulties": ["medium"],
    },
}


def modes_for_difficulty(difficulty: str) -> list:
    return [k for k, v in FAILURE_MODES.items() if difficulty in v["difficulties"]]


def is_correct_prescription(true_mode: str, prescribed_mode: str,
                             prescribed_diff: dict) -> tuple:
    if true_mode != prescribed_mode:
        return False, False
    spec = FAILURE_MODES[true_mode]
    fix_key = spec["fix_key"]
    if fix_key not in prescribed_diff:
        return True, False
    proposed = prescribed_diff[fix_key]
    direction = spec["fix_direction"]
    correct_val = spec["fix_value"]
    if direction == "replace":
        return True, (proposed == correct_val)
    elif direction == "down":
        return True, (isinstance(proposed, (int, float)) and proposed < correct_val * 10)
    elif direction == "up":
        return True, (isinstance(proposed, (int, float)) and proposed > correct_val / 10)
    return True, False
