from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from sitpath_eval.models.coord_gru import CoordGRU
from sitpath_eval.train.metrics import compute_metrics
from sitpath_eval.train.eval_metrics import aggregate_metrics
from sitpath_eval.utils.device import get_device


def subsample_dataset(dataset, fraction: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(dataset)
    size = max(1, int(n * fraction))
    indices = rng.choice(n, size=size, replace=False)
    return Subset(dataset, indices)


def train_and_evaluate(model_cls, dataset, fractions=None, **train_kwargs):
    if fractions is None:
        fractions = [0.1, 0.25, 1.0]
    results = []
    device = get_device("train")
    epochs = train_kwargs.get("epochs", 5)
    batch_size = train_kwargs.get("batch_size", 16)

    for frac in fractions:
        subset = subsample_dataset(dataset, frac)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        model = model_cls().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=train_kwargs.get("lr", 1e-3))
        loss_fn = torch.nn.MSELoss()

        for _ in range(epochs):
            model.train()
            for obs, targets in loader:
                obs = obs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                preds = model(obs)
                loss = loss_fn(preds, targets)
                loss.backward()
                optimizer.step()

        model.eval()
        preds_list = []
        targets_list = []
        with torch.no_grad():
            for obs, targets in loader:
                obs = obs.to(device)
                targets = targets.to(device)
                preds = model(obs)
                preds_list.append(preds.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
        preds = np.concatenate(preds_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        metrics = compute_metrics(preds, targets)
        results.append({"fraction": frac, "metrics": metrics})
    return results


def aggregate_by_fraction(results: List[Dict]) -> Dict[float, Dict[str, Dict[str, float]]]:
    grouped: Dict[float, List[Dict[str, float]]] = {}
    for entry in results:
        grouped.setdefault(entry["fraction"], []).append(entry["metrics"])
    aggregated = {}
    for frac, metric_runs in grouped.items():
        aggregated[frac] = aggregate_metrics(metric_runs)
    return aggregated


def save_efficiency_table(data: Dict[float, Dict[str, Dict[str, float]]], path_csv: str, path_tex: str) -> None:
    csv_path = Path(path_csv)
    tex_path = Path(path_tex)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = sorted(next(iter(data.values())).keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["Fraction"]
        for metric in metrics:
            header.extend([f"{metric}_mean", f"{metric}_ci_lower", f"{metric}_ci_upper"])
        writer.writerow(header)
        for frac, metric_vals in sorted(data.items()):
            row = [frac]
            for metric in metrics:
                stats = metric_vals[metric]
                row.extend([stats["mean"], stats["ci_lower"], stats["ci_upper"]])
            writer.writerow(row)

    lines = ["\\begin{tabular}{l" + "ccc" * len(metrics) + "}", "Fraction & " + " & ".join(
        f"{metric} Mean & {metric} CI Low & {metric} CI Up" for metric in metrics
    ) + " \\\\", "\\hline"]
    for frac, metric_vals in sorted(data.items()):
        cells = [f"{frac:.2f}"]
        for metric in metrics:
            stats = metric_vals[metric]
            cells.extend([f"{stats['mean']:.3f}", f"{stats['ci_lower']:.3f}", f"{stats['ci_upper']:.3f}"])
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\end{tabular}")
    tex_path.write_text("\n".join(lines))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    data = rng.normal(size=(64, 20, 2)).astype(np.float32)
    trajectories = data.cumsum(axis=1)
    obs = torch.from_numpy(trajectories[:, :8])
    targets = torch.from_numpy(trajectories[:, 8:])
    dataset = torch.utils.data.TensorDataset(obs, targets)
    results = train_and_evaluate(CoordGRU, dataset, fractions=[0.1, 0.5, 1.0], epochs=1)
    aggregated = aggregate_by_fraction(results)
    out_dir = Path("artifacts/tables")
    save_efficiency_table(aggregated, out_dir / "demo_efficiency.csv", out_dir / "demo_efficiency.tex")
