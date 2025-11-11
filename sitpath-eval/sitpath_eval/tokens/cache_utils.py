from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def precompute_tokens(dataset, tokenizer, out_path: str) -> None:
    """Encode dataset trajectories and persist token ids as .npz."""

    token_cache = []
    for item in dataset:
        coords = item["pos"].numpy()
        tokens = tokenizer.encode_trajectory(coords)
        token_cache.append(tokens)

    arr = np.array([np.array(tok, dtype=np.int64) for tok in token_cache], dtype=object)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, tokens=arr)
