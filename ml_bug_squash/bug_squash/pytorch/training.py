from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EvalResult:
    average_loss: float
    accuracy: float
    logits: Tensor
    labels: Tensor


def predict_labels_from_logits(logits: Tensor) -> Tensor:
    return (logits >= 0.5).to(torch.int64)


def train_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: Optimizer,
    loss_fn: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for features, labels in loader:
        logits = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()

        batch_size = features.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples


def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    loss_fn: nn.Module,
) -> EvalResult:
    total_loss = 0.0
    total_examples = 0
    logits_chunks: list[Tensor] = []
    label_chunks: list[Tensor] = []

    with torch.no_grad():
        for features, labels in loader:
            logits = model(features)
            loss = loss_fn(logits, labels)

            logits_chunks.append(logits)
            label_chunks.append(labels)

            batch_size = features.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size

    all_logits = torch.cat(logits_chunks, dim=0)
    all_labels = torch.cat(label_chunks, dim=0)
    predictions = predict_labels_from_logits(all_logits)
    accuracy = (predictions == all_labels.to(torch.int64)).float().mean().item()

    return EvalResult(
        average_loss=total_loss / total_examples,
        accuracy=accuracy,
        logits=all_logits,
        labels=all_labels,
    )
