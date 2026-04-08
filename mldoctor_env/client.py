"""Client wrapper for talking to the MLDoctorEnv server."""

try:
    from openenv.core.env_client import EnvClient
except ImportError:
    try:
        from openenv.client import EnvClient  # type: ignore
    except ImportError:
        EnvClient = object  # type: ignore

from .models import MLDoctorAction, MLDoctorObservation, StepResult


class MLDoctorEnv(EnvClient):  # type: ignore[misc]
    """Async client for the MLDoctorEnv FastAPI server."""
    action_cls = MLDoctorAction
    observation_cls = MLDoctorObservation
    step_result_cls = StepResult
