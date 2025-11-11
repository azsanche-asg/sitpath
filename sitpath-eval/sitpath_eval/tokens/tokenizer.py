from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from sitpath_eval.tokens.vocab import Vocabulary


class SitPathTokenizer:
    """Quantizes trajectory deltas into discrete token ids."""

    def __init__(
        self,
        vocab: Vocabulary,
        M: int = 16,
        R: float = 5.0,
        B: int = 4,
        include_tempo: bool = True,
        include_stall: bool = True,
    ):
        self.vocab = vocab
        self.M = M
        self.R = R
        self.B = B
        self.include_tempo = include_tempo
        self.include_stall = include_stall

    def __call__(self, points: Sequence[Sequence[float]]) -> List[int]:
        arr = np.asarray(points, dtype=np.float32)
        return self.encode_trajectory(arr)

    def encode_trajectory(self, xy_seq: np.ndarray) -> List[int]:
        if len(xy_seq) == 0:
            return []
        deltas = np.diff(xy_seq, axis=0, prepend=xy_seq[:1])
        tokens = []
        for dx, dy in deltas:
            sector = self.sector_index(dx, dy)
            radial = self.radial_bin(np.hypot(dx, dy))
            tempo = self.tempo_bin(np.hypot(dx, dy)) if self.include_tempo else 0
            stall = int(np.isclose(dx, 0.0) and np.isclose(dy, 0.0)) if self.include_stall else 0
            token_id = self.vocab.encode_tuple((sector, radial, tempo, stall))
            tokens.append(token_id)
        return tokens

    def sector_index(self, dx: float, dy: float) -> int:
        angle = np.arctan2(dy, dx)
        normalized = (angle + np.pi) / (2 * np.pi)
        return int(normalized * self.M) % self.M

    def radial_bin(self, radius: float) -> int:
        capped = min(radius, self.R)
        return min(int((capped / self.R) * self.B), self.B - 1)

    def tempo_bin(self, speed: float) -> int:
        if speed < 0.5:
            return 0
        if speed < 1.5:
            return 1
        return 2
