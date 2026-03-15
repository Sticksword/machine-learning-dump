from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class ChurnTensorDataset:
    feature_names: list[str]
    features: Tensor
    labels: Tensor


@dataclass(frozen=True)
class TabularSplit:
    train_features: Tensor
    train_labels: Tensor
    test_features: Tensor
    test_labels: Tensor


def _shared_dataset_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "customer_churn.csv"


def load_customer_churn_tensors(
    dataset_path: str | Path | None = None,
) -> ChurnTensorDataset:
    path = Path(dataset_path) if dataset_path is not None else _shared_dataset_path()

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "churned" not in reader.fieldnames:
            raise ValueError("dataset must include a 'churned' column")

        feature_names = [name for name in reader.fieldnames if name != "churned"]
        rows: list[list[float]] = []
        labels: list[list[float]] = []

        for row in reader:
            rows.append([float(row[name]) for name in feature_names])
            labels.append([float(row["churned"])])

    return ChurnTensorDataset(
        feature_names=feature_names,
        features=torch.tensor(rows, dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.float32),
    )


def make_train_test_split(
    features: Tensor,
    labels: Tensor,
    test_ratio: float = 0.3,
    seed: int = 13,
) -> TabularSplit:
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have the same number of rows")
    if features.shape[0] < 2:
        raise ValueError("need at least two examples to split")
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(features.shape[0], generator=generator)
    split_index = int(features.shape[0] * (1.0 - test_ratio))
    split_index = max(1, min(features.shape[0] - 1, split_index))

    train_indices = permutation[:split_index]
    test_indices = permutation[split_index:]

    return TabularSplit(
        train_features=features[train_indices],
        train_labels=labels[train_indices],
        test_features=features[test_indices],
        test_labels=labels[test_indices],
    )


def make_dataloaders(
    split: TabularSplit,
    batch_size: int = 4,
    seed: int = 13,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    train_dataset = TensorDataset(split.train_features, split.train_labels)
    test_dataset = TensorDataset(split.test_features, split.test_labels)
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
