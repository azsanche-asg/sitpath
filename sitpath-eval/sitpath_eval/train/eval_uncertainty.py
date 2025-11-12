from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from sitpath_eval.train.eval_metrics import aggregate_metrics, save_metrics_table


def negative_log_likelihood(preds: np.ndarray, gts: np.ndarray, sigma: float = 1.0) -> float:
    diff = preds - gts
    squared = np.sum(diff**2, axis=-1)
    nll = squared / (2 * sigma**2) + np.log(sigma * np.sqrt(2 * np.pi))
    return float(np.mean(nll))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if not np.any(mask):
            continue
        bin_conf = np.mean(probs[mask])
        bin_acc = np.mean(labels[mask])
        ece += np.abs(bin_acc - bin_conf) * (np.sum(mask) / len(probs))
    return float(ece)


def diversity_at_k(samples: np.ndarray, mr_mask: Optional[np.ndarray] = None) -> float:
    samples = np.asarray(samples)
    # Accepts (B,K,2) or (B,K,T,2); uses final step if 4D
    if samples.ndim == 4:
        samples = samples[:, :, -1, :]
    batch, k, _ = samples.shape
    dists = []
    for i in range(batch):
        if mr_mask is not None and mr_mask[i]:
            continue
        for a in range(k):
            for b in range(a + 1, k):
                d = np.linalg.norm(samples[i, a] - samples[i, b])
                dists.append(d)
    if not dists:
        return 0.0
    return float(np.mean(dists))


def compute_uncertainty_metrics(preds_k: np.ndarray, gts: np.ndarray, probs: Optional[np.ndarray] = None):
    means = preds_k.mean(axis=1)
    metrics = {
        "nll": negative_log_likelihood(means, gts),
        "div_k": diversity_at_k(preds_k),
    }
    if probs is not None:
        labels = (np.linalg.norm(means - gts, axis=-1) < 2.0).astype(float)
        metrics["ece"] = expected_calibration_error(probs, labels)
    else:
        metrics["ece"] = float("nan")
    return metrics


def aggregate_uncertainty(results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    return aggregate_metrics(results)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    preds_k = rng.normal(size=(32, 5, 2))
    gts = rng.normal(size=(32, 2))
    probs = rng.uniform(size=(32,))
    metrics = compute_uncertainty_metrics(preds_k, gts, probs)
    agg = aggregate_uncertainty([metrics])
    out_dir = Path("artifacts/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_metrics_table(agg, out_dir / "demo_uncertainty.csv", out_dir / "demo_uncertainty.tex")
