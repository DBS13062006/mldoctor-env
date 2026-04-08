"""Basic smoke tests for MLDoctorEnv."""

import math

from mldoctor_env.failure_taxonomy import FAILURE_MODES
from mldoctor_env.models import MLDoctorAction
from mldoctor_env.server.environment import MLDoctorEnvironment
from mldoctor_env.synthesizer import RunReportSynthesizer


def test_taxonomy_has_12_modes():
    assert len(FAILURE_MODES) == 12


def test_synthesizer_generates_distinct_curves():
    s = RunReportSynthesizer(seed=1)
    a = s.generate("lr_too_high")
    b = s.generate("lr_too_low")
    assert a.loss_curve != b.loss_curve


def test_nan_explosion_curve_has_nan():
    s = RunReportSynthesizer(seed=1)
    r = s.generate("nan_explosion")
    assert any(isinstance(x, float) and math.isnan(x) for x in r.loss_curve)


def test_environment_reset_step_basic():
    env = MLDoctorEnvironment(task_id="obvious_failure_diagnosis", seed=0)
    result = env.reset()
    assert result.observation.task_id == "obvious_failure_diagnosis"
    assert result.observation.step_count == 0

    result = env.step(MLDoctorAction(name="inspect_loss_curve"))
    assert result.observation.step_count == 1
    assert "inspect_loss_curve" in result.observation.inspections_made


def test_correct_prescription_scoring():
    env = MLDoctorEnvironment(task_id="obvious_failure_diagnosis", seed=0)
    env.reset()
    true_mode = env.report.true_failure_mode
    spec = FAILURE_MODES[true_mode]
    result = env.step(MLDoctorAction(
        name="prescribe",
        args={"failure_mode": true_mode,
              "config_diff": {spec["fix_key"]: spec["fix_value"]}},
    ))
    assert result.reward >= 0.4
