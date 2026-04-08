"""Task graders."""

from typing import List


def grade_trajectory(rewards: List[float]) -> float:
    if not rewards:
        return 0.0
    total = sum(rewards)
    return max(0.0, min(1.0, total))


def grade_obvious_failure(rewards: List[float]) -> float:
    return grade_trajectory(rewards)


def grade_subtle_divergence(rewards: List[float]) -> float:
    return grade_trajectory(rewards)


def grade_adversarial(rewards: List[float]) -> float:
    return grade_trajectory(rewards)


GRADERS = {
    "obvious_failure_diagnosis": grade_obvious_failure,
    "subtle_divergence_diagnosis": grade_subtle_divergence,
    "adversarial_compound_failure": grade_adversarial,
}
