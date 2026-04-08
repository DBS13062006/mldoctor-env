"""
test_reward_invariants.py — anti-gaming guarantees for the MLDoctorEnv reward.
"""

import random

import pytest

from mldoctor_env.failure_taxonomy import FAILURE_MODES
from mldoctor_env.models import MLDoctorAction
from mldoctor_env.server.environment import MLDoctorEnvironment


INSPECTION_TYPES = [
    "inspect_loss_curve",
    "inspect_grad_norms",
    "inspect_hyperparams",
    "inspect_dataset_stats",
    "inspect_error_log",
]

REWARD_INSPECTION_FIRST    = 0.05
REWARD_INSPECTION_REPEAT   = -0.02
REWARD_HYPOTHESIS_CORRECT  = 0.15
REWARD_HYPOTHESIS_WRONG    = -0.05
REWARD_ABLATION_USEFUL     = 0.10
REWARD_ABLATION_USELESS    = 0.0
REWARD_PRESCRIBE_CAT       = 0.30
REWARD_PRESCRIBE_FIX       = 0.20
REWARD_PRESCRIBE_WRONG     = -0.10
REWARD_INSPECTION_CAP      = REWARD_INSPECTION_FIRST * len(INSPECTION_TYPES)
REWARD_PERFECT             = 1.00


def fresh_env(task: str = "obvious_failure_diagnosis", seed: int = 0) -> MLDoctorEnvironment:
    env = MLDoctorEnvironment(task_id=task, seed=seed)
    env.reset()
    return env


def correct_prescribe_args(env: MLDoctorEnvironment) -> dict:
    true_mode = env.report.true_failure_mode
    spec = FAILURE_MODES[true_mode]
    return {
        "failure_mode": true_mode,
        "config_diff": {spec["fix_key"]: spec["fix_value"]},
    }


def wrong_prescribe_args(env: MLDoctorEnvironment) -> dict:
    true_mode = env.report.true_failure_mode
    wrong = next(m for m in FAILURE_MODES if m != true_mode)
    return {"failure_mode": wrong, "config_diff": {"foo": 1}}


def all_inspections(env: MLDoctorEnvironment) -> float:
    total = 0.0
    for name in INSPECTION_TYPES:
        r = env.step(MLDoctorAction(name=name))
        total += r.reward
    return total


class TestRewardShapeMatchesDocs:
    def test_first_time_inspection_pays_005(self):
        env = fresh_env()
        r = env.step(MLDoctorAction(name="inspect_loss_curve"))
        assert r.reward == pytest.approx(REWARD_INSPECTION_FIRST)

    def test_repeat_inspection_costs_002(self):
        env = fresh_env()
        env.step(MLDoctorAction(name="inspect_loss_curve"))
        r = env.step(MLDoctorAction(name="inspect_loss_curve"))
        assert r.reward == pytest.approx(REWARD_INSPECTION_REPEAT)

    def test_correct_hypothesis_pays_015(self):
        env = fresh_env()
        true_mode = env.report.true_failure_mode
        r = env.step(MLDoctorAction(name="hypothesize", args={"failure_mode": true_mode}))
        assert r.reward == pytest.approx(REWARD_HYPOTHESIS_CORRECT)

    def test_wrong_hypothesis_costs_005(self):
        env = fresh_env()
        wrong = next(m for m in FAILURE_MODES if m != env.report.true_failure_mode)
        r = env.step(MLDoctorAction(name="hypothesize", args={"failure_mode": wrong}))
        assert r.reward == pytest.approx(REWARD_HYPOTHESIS_WRONG)

    def test_useful_ablation_pays_010(self):
        env = fresh_env()
        fix_key = FAILURE_MODES[env.report.true_failure_mode]["fix_key"]
        r = env.step(MLDoctorAction(name="request_ablation",
                                    args={"config_change": {fix_key: "test_value"}}))
        assert r.reward == pytest.approx(REWARD_ABLATION_USEFUL)

    def test_useless_ablation_pays_zero(self):
        env = fresh_env()
        true_fix_key = FAILURE_MODES[env.report.true_failure_mode]["fix_key"]
        bogus_key = "definitely_not_a_real_hparam"
        assert bogus_key != true_fix_key
        r = env.step(MLDoctorAction(name="request_ablation",
                                    args={"config_change": {bogus_key: 1}}))
        assert r.reward == pytest.approx(REWARD_ABLATION_USELESS)

    def test_correct_prescription_pays_050(self):
        env = fresh_env()
        r = env.step(MLDoctorAction(name="prescribe", args=correct_prescribe_args(env)))
        assert r.reward == pytest.approx(REWARD_PRESCRIBE_CAT + REWARD_PRESCRIBE_FIX)

    def test_wrong_prescription_costs_010(self):
        env = fresh_env()
        r = env.step(MLDoctorAction(name="prescribe", args=wrong_prescribe_args(env)))
        assert r.reward == pytest.approx(REWARD_PRESCRIBE_WRONG)


class TestPerfectEpisodeSumsToOne:
    def test_perfect_trajectory(self):
        env = fresh_env()
        true_mode = env.report.true_failure_mode
        spec = FAILURE_MODES[true_mode]

        rewards = []
        for name in INSPECTION_TYPES:
            r = env.step(MLDoctorAction(name=name))
            rewards.append(r.reward)
            assert not r.done

        r = env.step(MLDoctorAction(name="hypothesize", args={"failure_mode": true_mode}))
        rewards.append(r.reward)
        assert not r.done

        r = env.step(MLDoctorAction(name="request_ablation",
                                    args={"config_change": {spec["fix_key"]: spec["fix_value"]}}))
        rewards.append(r.reward)
        assert not r.done

        r = env.step(MLDoctorAction(name="prescribe", args=correct_prescribe_args(env)))
        rewards.append(r.reward)
        assert r.done

        assert sum(rewards) == pytest.approx(REWARD_PERFECT)

    def test_perfect_episode_uses_exactly_8_steps(self):
        env = fresh_env()
        true_mode = env.report.true_failure_mode
        spec = FAILURE_MODES[true_mode]
        for name in INSPECTION_TYPES:
            env.step(MLDoctorAction(name=name))
        env.step(MLDoctorAction(name="hypothesize", args={"failure_mode": true_mode}))
        env.step(MLDoctorAction(name="request_ablation",
                                args={"config_change": {spec["fix_key"]: spec["fix_value"]}}))
        env.step(MLDoctorAction(name="prescribe", args=correct_prescribe_args(env)))
        assert env.state.step_count == 8


class TestAntiGamingInvariants:
    def test_inspection_only_agent_capped_at_025(self):
        env = fresh_env()
        total = 0.0
        done = False
        for _ in range(50):
            for name in INSPECTION_TYPES:
                r = env.step(MLDoctorAction(name=name))
                total += r.reward
                if r.done:
                    done = True
                    break
            if done:
                break
        assert total <= REWARD_INSPECTION_CAP + 1e-6

    def test_inspection_only_agent_cannot_pass(self):
        env = fresh_env()
        total = all_inspections(env)
        assert total == pytest.approx(REWARD_INSPECTION_CAP)
        assert total < 0.5

    def test_repeated_inspection_cannot_beat_first_time_reward(self):
        env = fresh_env()
        first = env.step(MLDoctorAction(name="inspect_loss_curve"))
        second = env.step(MLDoctorAction(name="inspect_loss_curve"))
        third = env.step(MLDoctorAction(name="inspect_loss_curve"))
        assert first.reward == pytest.approx(REWARD_INSPECTION_FIRST)
        assert second.reward == pytest.approx(REWARD_INSPECTION_REPEAT)
        assert third.reward == pytest.approx(REWARD_INSPECTION_REPEAT)

    def test_spamming_one_inspection_yields_negative_eventually(self):
        env = fresh_env()
        total = 0.0
        for _ in range(8):
            r = env.step(MLDoctorAction(name="inspect_loss_curve"))
            total += r.reward
            if r.done:
                break
        assert total < 0

    def test_hypothesizing_all_modes_is_strictly_negative(self):
        env = fresh_env()
        total = 0.0
        for mode in list(FAILURE_MODES.keys()):
            r = env.step(MLDoctorAction(name="hypothesize", args={"failure_mode": mode}))
            total += r.reward
            if r.done:
                break
        assert total < 0

    def test_wrong_prescription_ends_episode(self):
        env = fresh_env()
        r = env.step(MLDoctorAction(name="prescribe", args=wrong_prescribe_args(env)))
        assert r.done is True

    def test_correct_prescription_ends_episode(self):
        env = fresh_env()
        r = env.step(MLDoctorAction(name="prescribe", args=correct_prescribe_args(env)))
        assert r.done is True

    def test_brute_force_prescribe_expected_value_negative_on_hard(self):
        prob_correct = 1 / 12
        ev = (prob_correct * (REWARD_PRESCRIBE_CAT + REWARD_PRESCRIBE_FIX)
              + (1 - prob_correct) * REWARD_PRESCRIBE_WRONG)
        assert ev < 0

    def test_zero_evidence_prescribe_pays_full_when_correct(self):
        env = fresh_env()
        r = env.step(MLDoctorAction(name="prescribe", args=correct_prescribe_args(env)))
        assert r.reward == pytest.approx(REWARD_PRESCRIBE_CAT + REWARD_PRESCRIBE_FIX)
        assert r.done

    def test_zero_evidence_prescribe_strategy_loses_in_expectation_on_hard(self):
        random.seed(42)
        scores = []
        for trial in range(100):
            env = MLDoctorEnvironment(task_id="adversarial_compound_failure", seed=trial)
            env.reset()
            mode = random.choice(list(FAILURE_MODES.keys()))
            spec = FAILURE_MODES[mode]
            r = env.step(MLDoctorAction(name="prescribe",
                                        args={"failure_mode": mode,
                                              "config_diff": {spec["fix_key"]: spec["fix_value"]}}))
            scores.append(max(0.0, min(1.0, r.reward)))
        mean = sum(scores) / len(scores)
        assert mean < 0.10

    def test_submit_immediately_scores_zero(self):
        env = fresh_env()
        r = env.step(MLDoctorAction(name="submit"))
        assert r.done is True
        assert r.reward == pytest.approx(0.0)

    def test_useless_ablations_consume_steps_with_no_reward(self):
        env = fresh_env()
        bogus_key = "definitely_not_a_real_hparam"
        for _ in range(3):
            r = env.step(MLDoctorAction(name="request_ablation",
                                        args={"config_change": {bogus_key: 1}}))
            assert r.reward == pytest.approx(REWARD_ABLATION_USELESS)
        assert env.state.step_count == 3


class TestRewardDeterminism:
    def test_same_seed_same_trajectory(self):
        env_a = fresh_env(seed=123)
        env_b = fresh_env(seed=123)
        actions = [
            MLDoctorAction(name="inspect_loss_curve"),
            MLDoctorAction(name="inspect_hyperparams"),
            MLDoctorAction(name="hypothesize", args={"failure_mode": "lr_too_high"}),
        ]
        rewards_a = [env_a.step(a).reward for a in actions]
        rewards_b = [env_b.step(a).reward for a in actions]
        assert rewards_a == rewards_b

    def test_score_clipping_to_unit_interval(self):
        env = fresh_env()
        actions = [
            MLDoctorAction(name="inspect_loss_curve"),
            MLDoctorAction(name="inspect_loss_curve"),
            MLDoctorAction(name="hypothesize", args={"failure_mode": "mode_collapse"}),
            MLDoctorAction(name="prescribe", args=wrong_prescribe_args(env)),
        ]
        rewards = []
        for a in actions:
            r = env.step(a)
            rewards.append(r.reward)
            if r.done:
                break
        total = sum(rewards)
        assert -0.5 <= total <= 1.0


class TestRewardConsistencyAcrossTasks:
    @pytest.mark.parametrize("task", [
        "obvious_failure_diagnosis",
        "subtle_divergence_diagnosis",
        "adversarial_compound_failure",
    ])
    def test_first_time_inspection_pays_005_on_all_tasks(self, task):
        env = fresh_env(task=task)
        r = env.step(MLDoctorAction(name="inspect_loss_curve"))
        assert r.reward == pytest.approx(REWARD_INSPECTION_FIRST)

    @pytest.mark.parametrize("task", [
        "obvious_failure_diagnosis",
        "subtle_divergence_diagnosis",
        "adversarial_compound_failure",
    ])
    def test_correct_prescribe_pays_050_on_all_tasks(self, task):
        env = fresh_env(task=task)
        r = env.step(MLDoctorAction(name="prescribe", args=correct_prescribe_args(env)))
        assert r.reward == pytest.approx(REWARD_PRESCRIBE_CAT + REWARD_PRESCRIBE_FIX)
        assert r.done

    @pytest.mark.parametrize("task,expected_max_steps", [
        ("obvious_failure_diagnosis", 10),
        ("subtle_divergence_diagnosis", 12),
        ("adversarial_compound_failure", 15),
    ])
    def test_max_steps_per_task(self, task, expected_max_steps):
        env = fresh_env(task=task)
        assert env.max_steps == expected_max_steps
