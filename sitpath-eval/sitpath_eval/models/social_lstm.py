from __future__ import annotations

import torch
from torch import nn

from sitpath_eval.models.base_model import BaseTrajectoryModel


class SocialLSTM(BaseTrajectoryModel):
    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        hidden_size: int = 64,
        social_dim: int = 32,
        pooling_radius: float = 2.0,
        **kwargs,
    ):
        super().__init__(obs_len, pred_len)
        self.pooling_radius = pooling_radius
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True)
        self.social_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, social_dim),
            nn.ReLU(),
            nn.Linear(social_dim, hidden_size),
        )
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, 2)

    def forward(self, obs_batch: torch.Tensor) -> torch.Tensor:
        batch_size = obs_batch.size(0)
        device = obs_batch.device
        # Simple Social-LSTM baseline with mean-pooling
        enc_out, (hidden, cell) = self.encoder(obs_batch)
        final_hidden = enc_out[:, -1, :]

        diff = final_hidden.unsqueeze(1) - final_hidden.unsqueeze(0)
        dist = torch.norm(diff, dim=-1)
        mask = (dist <= self.pooling_radius).float()
        pooled = []
        neighbors = final_hidden
        for i in range(batch_size):
            weights = mask[i].unsqueeze(-1)
            denom = weights.sum() + 1e-6
            mean_neighbor = (neighbors * weights).sum(dim=0) / denom
            pooled_vec = torch.cat([final_hidden[i], mean_neighbor], dim=-1)
            pooled.append(self.social_mlp(pooled_vec))
        pooled_tensor = torch.stack(pooled, dim=0)

        decoder_input = torch.zeros(batch_size, self.pred_len, self.hidden_size, device=device)
        dec_out, _ = self.decoder(decoder_input, (pooled_tensor.unsqueeze(0), cell))
        preds = self.out(dec_out)
        return preds
