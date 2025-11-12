from __future__ import annotations

from typing import Dict

import numpy as np


def ade(preds: np.ndarray, gts: np.ndarray) -> float:
    return float(np.linalg.norm(preds - gts, axis=-1).mean())


def fde(preds: np.ndarray, gts: np.ndarray) -> float:
    return float(np.linalg.norm(preds[:, -1] - gts[:, -1], axis=-1).mean())


def minade_k(preds_k: np.ndarray, gts: np.ndarray) -> float:
    diffs = preds_k - gts[None, ...]
    ade_per_k = np.linalg.norm(diffs, axis=-1).mean(axis=-1)
    min_per_sample = ade_per_k.min(axis=0)
    return float(min_per_sample.mean())


def miss_rate(preds: np.ndarray, gts: np.ndarray, thresh: float = 2.0) -> float:
    final_d = np.linalg.norm(preds[:, -1] - gts[:, -1], axis=-1)
    misses = final_d > thresh
    return float(misses.mean())


def compute_metrics(preds: np.ndarray, gts: np.ndarray) -> Dict[str, float]:
    result = {
        "ade": ade(preds, gts),
        "fde": fde(preds, gts),
        "miss_rate": miss_rate(preds, gts),
    }
    if preds.ndim == 4:
        result["minade_k"] = minade_k(preds, gts)
    return result
