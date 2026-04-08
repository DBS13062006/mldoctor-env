"""FastAPI application factory for MLDoctorEnv."""

import os

from fastapi import FastAPI

from ..models import MLDoctorAction, MLDoctorObservation, StepResult
from .environment import MLDoctorEnvironment


TASK_ID = os.getenv("MLDOCTOR_TASK", "obvious_failure_diagnosis")
SEED = int(os.getenv("MLDOCTOR_SEED", "0"))

env = MLDoctorEnvironment(task_id=TASK_ID, seed=SEED)


def _build_app() -> FastAPI:
    """Hand-rolled FastAPI app exposing the OpenEnv-compatible endpoints.

    (openenv-core's create_fastapi_app expects its own Action/Observation
    base classes; we use plain pydantic and expose /reset and /step directly
    so the env works standalone and remains portable.)
    """
    app = FastAPI(title="MLDoctorEnv")

    @app.get("/")
    def root():
        return {"name": "mldoctor_env", "task": TASK_ID}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/reset")
    def reset(payload: dict | None = None) -> dict:
        result: StepResult = env.reset()
        return result.model_dump()

    @app.post("/step")
    def step(payload: dict) -> dict:
        action_data = payload.get("action", payload)
        action = MLDoctorAction(**action_data)
        result: StepResult = env.step(action)
        return result.model_dump()

    return app


app = _build_app()
