from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseTrajectoryModel(nn.Module, ABC):
    def __init__(self, obs_len: int, pred_len: int):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict future coordinates given observed ones."""

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
