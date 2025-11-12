from __future__ import annotations

import math

import torch
from torch import nn

from sitpath_eval.models.base_model import BaseTrajectoryModel


def positional_encoding(max_len: int, dim: int) -> torch.Tensor:
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim))
    pe = torch.zeros(max_len, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class SitPathTransformer(BaseTrajectoryModel):
    def __init__(
        self,
        vocab_size: int,
        token_dim: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        pred_len: int = 12,
        **kwargs,
    ):
        obs_len = kwargs.pop("obs_len", 0)
        super().__init__(obs_len, pred_len)
        self.vocab_size = vocab_size
        self.token_dim = token_dim
        self.embedding = nn.Embedding(vocab_size, token_dim)
        self.transformer = nn.Transformer(
            d_model=token_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(token_dim, vocab_size)
        self.register_buffer("pos_enc", positional_encoding(512, token_dim), persistent=False)

    def _positional_slice(self, length: int, device) -> torch.Tensor:
        if length > self.pos_enc.size(0):
            raise ValueError("Sequence length exceeds positional encoding capacity.")
        return self.pos_enc[:length].to(device)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.numel() == 0:
            raise ValueError("Tokens input must be non-empty.")
        device = tokens.device
        src_len = tokens.size(1)
        src = self.embedding(tokens) + self._positional_slice(src_len, device)
        memory = self.transformer.encoder(src)

        tgt_template = torch.zeros(tokens.size(0), self.pred_len, self.token_dim, device=device)
        logits = []
        for step in range(self.pred_len):
            tgt_step = tgt_template[:, : step + 1] + self._positional_slice(step + 1, device)
            out = self.transformer.decoder(tgt_step, memory)
            step_logits = self.head(out[:, -1, :])
            logits.append(step_logits)
        return torch.stack(logits, dim=1)

    def sample(self, tokens: torch.Tensor, K: int = 20, temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(tokens) / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            draws = [dist.sample() for _ in range(K)]
            samples = torch.stack(draws, dim=-1)
        return samples
