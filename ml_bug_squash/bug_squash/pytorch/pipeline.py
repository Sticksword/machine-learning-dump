from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD

from .data import TabularSplit, load_customer_churn_tensors, make_dataloaders, make_train_test_split
from .model import ChurnNet
from .preprocessing import TensorStandardizer
from .training import evaluate, train_epoch


@dataclass(frozen=True)
class TorchTrainingSummary:
    test_accuracy: float
    test_loss: float
    train_size: int
    test_size: int


class TorchChurnPipeline:
    def __init__(
        self,
        hidden_dim: int = 8,
        dropout: float = 0.25,
        learning_rate: float = 0.1,
        epochs: int = 40,
        batch_size: int = 4,
        test_ratio: float = 0.3,
        seed: int = 13,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.seed = seed

    def run(self, dataset_path: str | Path | None = None) -> TorchTrainingSummary:
        torch.manual_seed(self.seed)

        dataset = load_customer_churn_tensors(dataset_path)
        split = make_train_test_split(
            dataset.features,
            dataset.labels,
            test_ratio=self.test_ratio,
            seed=self.seed,
        )

        standardizer = TensorStandardizer.fit(split.train_features)
        normalized_split = TabularSplit(
            train_features=standardizer.transform_split(split.train_features),
            train_labels=split.train_labels,
            test_features=standardizer.transform_split(split.test_features),
            test_labels=split.test_labels,
        )

        train_loader, _ = make_dataloaders(
            normalized_split,
            batch_size=self.batch_size,
            seed=self.seed,
        )
        _, test_loader = make_dataloaders(
            TabularSplit(
                train_features=normalized_split.train_features,
                train_labels=normalized_split.train_labels,
                test_features=normalized_split.test_features,
                test_labels=split.test_labels,
            ),
            batch_size=self.batch_size,
            seed=self.seed,
        )

        model = ChurnNet(
            num_features=dataset.features.shape[1],
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )
        optimizer = SGD(model.parameters(), lr=self.learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(self.epochs):
            train_epoch(model, train_loader, optimizer, loss_fn)

        evaluation = evaluate(model, test_loader, loss_fn)
        return TorchTrainingSummary(
            test_accuracy=evaluation.accuracy,
            test_loss=evaluation.average_loss,
            train_size=normalized_split.train_features.shape[0],
            test_size=normalized_split.test_features.shape[0],
        )
