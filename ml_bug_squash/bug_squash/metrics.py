from __future__ import annotations

import math


def accuracy_score(y_true: list[int], y_pred: list[int]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("cannot score an empty label set")

    correct = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred)
    return correct / len(y_true)


def binary_log_loss(y_true: list[int], y_prob: list[float]) -> float:
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    if not y_true:
        raise ValueError("cannot score an empty label set")

    total = 0.0
    for truth, prob in zip(y_true, y_prob):
        clipped = min(max(prob, 1e-9), 1.0 - 1e-9)
        total += -(
            truth * math.log(clipped) + (1 - truth) * math.log(1.0 - clipped)
        )

    return total / len(y_true)
