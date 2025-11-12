from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

from sitpath_eval.train.eval_metrics import aggregate_metrics, save_metrics_table
from sitpath_eval.train.metrics import ade


def apply_edit_rule(trajs: np.ndarray, rule: str) -> np.ndarray:
    edited = np.array(trajs, copy=True)
    if rule == "avoid_front":
        edited[..., 1] = -np.abs(edited[..., 1])
    elif rule == "keep_right":
        edited[..., 0] = np.abs(edited[..., 0])
    elif rule == "slow_down":
        displacements = np.diff(edited, axis=1, prepend=edited[:, :1])
        scaled = displacements * 0.5
        edited = edited[:, :1] + np.cumsum(scaled[:, 1:], axis=1)
    else:
        raise ValueError(f"Unknown rule {rule}")
    return edited


def compute_constraint_satisfaction(orig: np.ndarray, edited: np.ndarray, rule: str) -> float:
    if rule == "avoid_front":
        return float(np.mean(edited[..., 1] < orig[..., 1]))
    if rule == "keep_right":
        return float(np.mean(edited[..., 0] >= 0))
    if rule == "slow_down":
        orig_speed = np.linalg.norm(np.diff(orig, axis=1), axis=-1).mean(axis=1)
        edited_speed = np.linalg.norm(np.diff(edited, axis=1), axis=-1).mean(axis=1)
        return float(np.mean(edited_speed < orig_speed))
    raise ValueError(f"Unknown rule {rule}")


def controllability_metrics(orig: np.ndarray, edited: np.ndarray, gts: np.ndarray, rule: str):
    csr = compute_constraint_satisfaction(orig, edited, rule)
    orig_end = orig[:, -1]
    edited_end = edited[:, -1]
    gr = 1 - np.linalg.norm(edited_end - orig_end, axis=-1).mean() / (
        np.linalg.norm(orig_end - gts[:, -1], axis=-1).mean() + 1e-6
    )
    # Ensure equal pred_len before ADE computation
    min_len = min(orig.shape[1], edited.shape[1], gts.shape[1])
    orig_aligned = orig[:, :min_len, :]
    edited_aligned = edited[:, :min_len, :]
    gts_aligned = gts[:, :min_len, :]
    delta_ade = ade(edited_aligned, gts_aligned) - ade(orig_aligned, gts_aligned)
    return {"rule": rule, "csr": csr, "gr": gr, "delta_ade": delta_ade}


def aggregate_controllability(results: List[Dict[str, float]]):
    metrics = [{k: v for k, v in res.items() if k != "rule"} for res in results]
    return aggregate_metrics(metrics)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    orig = rng.normal(size=(32, 12, 2)).cumsum(axis=1)
    gts = orig + rng.normal(scale=0.1, size=orig.shape)
    rule = "avoid_front"
    edited = apply_edit_rule(orig, rule)
    metrics = controllability_metrics(orig, edited, gts, rule)
    agg = aggregate_controllability([metrics])
    out_dir = Path("artifacts/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_metrics_table(agg, out_dir / "demo_controllability.csv", out_dir / "demo_controllability.tex")
