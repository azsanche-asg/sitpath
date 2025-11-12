from __future__ import annotations

import torch
from torch import nn

from sitpath_eval.models.base_model import BaseTrajectoryModel


class SitPathGRU(BaseTrajectoryModel):
    def __init__(
        self,
        vocab_size: int,
        token_dim: int = 64,
        hidden_size: int = 64,
        pred_len: int = 12,
        **kwargs,
    ):
        obs_len = kwargs.pop("obs_len", 0)
        super().__init__(obs_len, pred_len)
        self.vocab_size = vocab_size
        self.token_dim = token_dim
        self.embedding = nn.Embedding(vocab_size, token_dim)
        self.gru = nn.GRU(token_dim, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.numel() == 0:
            raise ValueError("Tokens input must be non-empty.")
        # clamp synthetic token ids to valid range (for ablation safety)
        vocab_size = self.embedding.num_embeddings
        tokens = tokens % vocab_size
        emb = self.embedding(tokens)
        _, hidden = self.gru(emb)
        hidden_state = hidden
        batch_size = tokens.size(0)
        step_input = torch.zeros(batch_size, 1, self.token_dim, device=tokens.device)
        logits = []
        for _ in range(self.pred_len):
            out, hidden_state = self.gru(step_input, hidden_state)
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
