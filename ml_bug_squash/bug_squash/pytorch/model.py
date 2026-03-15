from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class ChurnNet(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 8,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)
