from __future__ import annotations

import math


class StandardScaler:
    def __init__(self) -> None:
        self.means: list[float] = []
        self.scales: list[float] = []

    def fit(self, rows: list[list[float]]) -> "StandardScaler":
        if not rows:
            raise ValueError("cannot fit on an empty dataset")

        num_features = len(rows[0])
        self.means = []
        self.scales = []

        for feature_index in range(num_features):
            values = [row[feature_index] for row in rows]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            scale = math.sqrt(variance) or 1.0
            self.means.append(mean)
            self.scales.append(scale)

        return self

    def transform(self, rows: list[list[float]]) -> list[list[float]]:
        if not self.means or not self.scales:
            raise ValueError("fit must be called before transform")

        transformed: list[list[float]] = []
        for row in rows:
            transformed.append(
                [
                    (value - mean) * scale
                    for value, mean, scale in zip(row, self.means, self.scales)
                ]
            )
        return transformed

    def fit_transform(self, rows: list[list[float]]) -> list[list[float]]:
        return self.fit(rows).transform(rows)
