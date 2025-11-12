from __future__ import annotations

import math

import torch
from torch import nn

from sitpath_eval.models.base_model import BaseTrajectoryModel


def positional_encoding(seq_len: int, dim: int) -> torch.Tensor:
    positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe


class CoordTransformer(BaseTrajectoryModel):
    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        **kwargs,
    ):
        super().__init__(obs_len, pred_len)
        self.input_proj = nn.Linear(2, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(d_model, 2)
        self.register_buffer("enc_pe", positional_encoding(obs_len + pred_len, d_model), persistent=False)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = obs.shape
        device = obs.device
        enc_pe = self.enc_pe[:seq_len].to(device)
        dec_pe = self.enc_pe[: self.pred_len].to(device)

        enc_in = self.input_proj(obs) + enc_pe
        memory = self.transformer.encoder(enc_in)

        tgt = torch.zeros(batch_size, self.pred_len, 2, device=device)
        outputs = []
        dec_input = self.input_proj(tgt) + dec_pe

        for step in range(self.pred_len):
            tgt_step = dec_input[:, : step + 1]
            out = self.transformer.decoder(tgt_step, memory)
            pred = self.head(out[:, -1:])
            outputs.append(pred)
        return torch.cat(outputs, dim=1)
