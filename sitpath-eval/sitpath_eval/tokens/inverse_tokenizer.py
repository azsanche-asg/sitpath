from __future__ import annotations

from typing import List

import numpy as np

from sitpath_eval.tokens.vocab import Vocabulary


def decode_tokens(tokens: List[int], vocab: Vocabulary) -> np.ndarray:
    """Toy inverse tokenizer returning cumulative positions."""

    positions = [(0.0, 0.0)]
    for token in tokens:
        sector, radial, tempo, stall = vocab.decode_id(token)
        angle = (sector / vocab.sector_count) * 2 * np.pi - np.pi
        radius = (radial + 1) / vocab.radial_bins
        dx = radius * np.cos(angle)
        dy = radius * np.sin(angle)
        if stall:
            dx = dy = 0.0
        last_x, last_y = positions[-1]
        positions.append((last_x + dx, last_y + dy))
    return np.array(positions, dtype=np.float32)
