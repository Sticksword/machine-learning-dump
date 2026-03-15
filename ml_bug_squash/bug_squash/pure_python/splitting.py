from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class DatasetSplit:
    train_rows: list[list[float]]
    train_labels: list[int]
    test_rows: list[list[float]]
    test_labels: list[int]


def train_test_split(
    rows: list[list[float]],
    labels: list[int],
    test_ratio: float = 0.25,
    seed: int = 7,
) -> DatasetSplit:
    if len(rows) != len(labels):
        raise ValueError("rows and labels must have the same length")
    if len(rows) < 2:
        raise ValueError("need at least two examples to split")
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1")

    row_indices = list(range(len(rows)))
    label_indices = list(range(len(labels)))
    rng = random.Random(seed)
    rng.shuffle(row_indices)
    rng.shuffle(label_indices)

    shuffled_rows = [rows[index] for index in row_indices]
    shuffled_labels = [labels[index] for index in label_indices]

    split_index = int(len(rows) * (1.0 - test_ratio))
    split_index = max(1, min(len(rows) - 1, split_index))

    return DatasetSplit(
        train_rows=shuffled_rows[:split_index],
        train_labels=shuffled_labels[:split_index],
        test_rows=shuffled_rows[split_index:],
        test_labels=shuffled_labels[split_index:],
    )
