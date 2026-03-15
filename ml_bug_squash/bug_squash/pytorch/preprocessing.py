from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class TensorStandardizer:
    mean: Tensor
    std: Tensor

    @classmethod
    def fit(cls, features: Tensor) -> "TensorStandardizer":
        if features.numel() == 0:
            raise ValueError("cannot fit on an empty tensor")

        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        return cls(mean=mean, std=std)

    def transform(self, features: Tensor) -> Tensor:
        return (features - self.mean) / self.std

    def transform_split(self, split_features: Tensor) -> Tensor:
        return self.transform(split_features)
