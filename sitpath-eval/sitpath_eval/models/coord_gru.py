from __future__ import annotations

import torch
from torch import nn

from sitpath_eval.models.base_model import BaseTrajectoryModel


class CoordGRU(BaseTrajectoryModel):
    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        hidden_size: int = 64,
        num_layers: int = 1,
        **kwargs,
    ):
        super().__init__(obs_len, pred_len)
        self.gru = nn.GRU(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features, hidden = self.gru(obs)
        last_hidden = hidden[-1]
        preds = []
        step_hidden = last_hidden.unsqueeze(0)
        for _ in range(self.pred_len):
            out = self.head(step_hidden.squeeze(0))
            preds.append(out)
        return torch.stack(preds, dim=1)
