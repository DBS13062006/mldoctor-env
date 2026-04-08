"""Pydantic models for MLDoctorEnv: Action, Observation, State, StepResult."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


ActionName = Literal[
    "inspect_loss_curve",
    "inspect_grad_norms",
    "inspect_hyperparams",
    "inspect_dataset_stats",
    "inspect_error_log",
    "hypothesize",
    "request_ablation",
    "prescribe",
    "submit",
]


class MLDoctorAction(BaseModel):
    name: ActionName
    args: Dict[str, Any] = Field(default_factory=dict)


class MLDoctorObservation(BaseModel):
    task_id: str
    task_difficulty: str
    incident_header: str
    last_action_result: str = ""
    inspections_made: List[str] = Field(default_factory=list)
    current_hypothesis: Optional[str] = None
    prescribed: bool = False
    submitted: bool = False
    step_count: int = 0
    max_steps: int = 15


class MLDoctorState(BaseModel):
    task_id: str
    episode_id: str
    step_count: int = 0
    cumulative_reward: float = 0.0


class StepResult(BaseModel):
    observation: MLDoctorObservation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)
