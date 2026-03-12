from __future__ import annotations

import math


def sigmoid(value: float) -> float:
    if value >= 0.0:
        scaled = math.exp(-value)
        return 1.0 / (1.0 + scaled)

    scaled = math.exp(value)
    return scaled / (1.0 + scaled)


def dot_product(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class LogisticRegressionGD:
    def __init__(self, learning_rate: float = 0.1, epochs: int = 300) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: list[float] | None = None
        self.bias = 0.0

    def fit(
        self,
        rows: list[list[float]],
        labels: list[int],
    ) -> "LogisticRegressionGD":
        if len(rows) != len(labels):
            raise ValueError("rows and labels must have the same length")
        if not rows:
            raise ValueError("cannot train on an empty dataset")

        self.weights = [0.0 for _ in rows[0]]
        self.bias = 0.0

        for _ in range(self.epochs):
            grad_w = [0.0 for _ in self.weights]
            grad_b = 0.0

            for row, label in zip(rows, labels):
                prediction = sigmoid(dot_product(row, self.weights) + self.bias)
                error = prediction - label
                grad_b += error

                for index, value in enumerate(row):
                    grad_w[index] += error * value

            scale = 1.0 / len(rows)
            for index in range(len(self.weights)):
                self.weights[index] += self.learning_rate * grad_w[index] * scale
            self.bias += self.learning_rate * grad_b * scale

        return self

    def predict_proba(self, rows: list[list[float]]) -> list[float]:
        if self.weights is None:
            raise ValueError("model must be fit before prediction")

        return [sigmoid(dot_product(row, self.weights) + self.bias) for row in rows]

    def predict(self, rows: list[list[float]], threshold: float = 0.5) -> list[int]:
        return [1 if prob >= threshold else 0 for prob in self.predict_proba(rows)]
