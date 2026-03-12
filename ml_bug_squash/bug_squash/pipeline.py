from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .datasets import load_customer_churn
from .metrics import accuracy_score, binary_log_loss
from .model import LogisticRegressionGD
from .preprocessing import StandardScaler
from .splitting import train_test_split


@dataclass(frozen=True)
class TrainingSummary:
    test_accuracy: float
    test_log_loss: float
    train_size: int
    test_size: int


class ChurnTrainingPipeline:
    def __init__(
        self,
        learning_rate: float = 0.2,
        epochs: int = 500,
        test_ratio: float = 0.3,
        seed: int = 13,
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.test_ratio = test_ratio
        self.seed = seed

    def run(self, dataset_path: str | Path | None = None) -> TrainingSummary:
        dataset = load_customer_churn(dataset_path)
        split = train_test_split(
            dataset.rows,
            dataset.labels,
            test_ratio=self.test_ratio,
            seed=self.seed,
        )

        scaler = StandardScaler()
        train_rows = scaler.fit_transform(split.train_rows)
        test_rows = scaler.transform(split.test_rows)

        model = LogisticRegressionGD(
            learning_rate=self.learning_rate,
            epochs=self.epochs,
        )
        model.fit(train_rows, split.train_labels)

        probabilities = model.predict_proba(test_rows)
        predictions = model.predict(test_rows)

        return TrainingSummary(
            test_accuracy=accuracy_score(split.test_labels, predictions),
            test_log_loss=binary_log_loss(split.test_labels, probabilities),
            train_size=len(split.train_rows),
            test_size=len(split.test_rows),
        )
