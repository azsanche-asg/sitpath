from __future__ import annotations

import torch
from torch import nn

from sitpath_eval.models.base_model import BaseTrajectoryModel


class RasterGRU(BaseTrajectoryModel):
    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        in_channels: int = 3,
        cnn_dim: int = 32,
        hidden_size: int = 64,
        num_layers: int = 1,
        **kwargs,
    ):
        super().__init__(obs_len, pred_len)
        self.obs_len = obs_len
        self.pred_len = pred_len
        channels = in_channels
        convs = []
        for _ in range(4):
            convs.append(
                nn.Conv2d(channels, cnn_dim, kernel_size=3, stride=2, padding=1)
            )
            convs.append(nn.ReLU(inplace=True))
            channels = cnn_dim
        self.cnn = nn.Sequential(*convs)
        self.fc = nn.Linear(cnn_dim * 4 * 4, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, _, _ = x.shape
        device = x.device
        cnn_feats = []
        for t in range(seq_len):
            frame = x[:, t]
            feat = self.cnn(frame)
            feat = feat.view(batch_size, -1)
            cnn_feats.append(torch.relu(self.fc(feat)))
        encoded = torch.stack(cnn_feats, dim=1)
        _, hidden = self.gru(encoded)
        preds = []
        step_input = torch.zeros(batch_size, 1, self.gru.hidden_size, device=device)
        hidden_state = hidden
        for _ in range(self.pred_len):
            out, hidden_state = self.gru(step_input, hidden_state)
            preds.append(self.head(out[:, -1, :]))
        return torch.stack(preds, dim=1)
