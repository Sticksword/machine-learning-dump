from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from bug_squash.pytorch.model import ChurnNet
from bug_squash.pytorch.training import evaluate, predict_labels_from_logits, train_epoch


class TrainingLoopTest(unittest.TestCase):
    def test_predict_labels_uses_zero_logit_boundary(self) -> None:
        logits = torch.tensor([[-0.2], [0.1], [0.9], [-3.0]], dtype=torch.float32)

        predictions = predict_labels_from_logits(logits)

        self.assertEqual(predictions.view(-1).tolist(), [0, 1, 1, 0])

    def test_train_epoch_updates_model_weights(self) -> None:
        torch.manual_seed(0)
        features = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]], dtype=torch.float32)
        labels = torch.tensor([[0.0], [0.0], [1.0], [1.0]], dtype=torch.float32)
        loader = DataLoader(TensorDataset(features, labels), batch_size=2, shuffle=False)

        model = nn.Linear(1, 1)
        optimizer = SGD(model.parameters(), lr=0.1)
        loss_fn = nn.BCEWithLogitsLoss()

        before = [parameter.detach().clone() for parameter in model.parameters()]
        train_epoch(model, loader, optimizer, loss_fn)
        after = [parameter.detach().clone() for parameter in model.parameters()]

        self.assertTrue(
            any(not torch.allclose(start, end) for start, end in zip(before, after))
        )

    def test_evaluate_is_deterministic_for_dropout_model(self) -> None:
        torch.manual_seed(0)
        features = torch.randn(8, 4)
        labels = torch.tensor([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0]])
        loader = DataLoader(TensorDataset(features, labels), batch_size=4, shuffle=False)

        model = ChurnNet(num_features=4, hidden_dim=8, dropout=0.5)
        loss_fn = nn.BCEWithLogitsLoss()

        first = evaluate(model, loader, loss_fn)
        second = evaluate(model, loader, loss_fn)

        self.assertTrue(torch.allclose(first.logits, second.logits))


if __name__ == "__main__":
    unittest.main()
