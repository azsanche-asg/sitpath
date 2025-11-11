from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple


def split_dataset(
    trajectories: Sequence[Dict],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split trajectories into train/val/test with deterministic ordering."""

    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be >=0 and sum to < 1")

    shuffled = list(trajectories)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)

    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    return train, val, test
