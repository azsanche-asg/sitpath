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

        layers = []
        channels = in_channels
        dim = cnn_dim
        for _ in range(4):
            layers.append(
                nn.Conv2d(channels, dim, kernel_size=3, stride=2, padding=1)
            )
            layers.append(nn.ReLU(inplace=True))
            channels = dim
        self.cnn = nn.Sequential(*layers)
        self.fc = nn.Linear(dim * 4 * 4, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 2)

    def encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(frame)
        feat = feat.view(feat.size(0), -1)
        return torch.relu(self.fc(feat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, _, _ = x.shape
        device = x.device

        encoded = []
        for t in range(seq_len):
            encoded.append(self.encode_frame(x[:, t]))
        encoded = torch.stack(encoded, dim=1)

        _, hidden = self.gru(encoded)
        outputs = []
        step_input = torch.zeros(batch_size, 1, self.gru.hidden_size, device=device)
        hidden_state = hidden
        for _ in range(self.pred_len):
            out, hidden_state = self.gru(step_input, hidden_state)
            pred = self.head(out[:, -1, :])
            outputs.append(pred)
        return torch.stack(outputs, dim=1)
