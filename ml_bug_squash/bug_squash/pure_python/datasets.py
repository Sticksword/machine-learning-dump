from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dataset:
    feature_names: list[str]
    rows: list[list[float]]
    labels: list[int]

    @property
    def num_examples(self) -> int:
        return len(self.rows)

    @property
    def num_features(self) -> int:
        return len(self.feature_names)


def _default_dataset_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "customer_churn.csv"


def load_customer_churn(dataset_path: str | Path | None = None) -> Dataset:
    path = Path(dataset_path) if dataset_path is not None else _default_dataset_path()

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "churned" not in reader.fieldnames:
            raise ValueError("dataset must include a 'churned' column")

        feature_names = [name for name in reader.fieldnames if name != "churned"]
        rows: list[list[float]] = []
        labels: list[int] = []

        for row in reader:
            rows.append([float(row[name]) for name in feature_names])
            labels.append(int(row["churned"]))

    return Dataset(feature_names=feature_names, rows=rows, labels=labels)
