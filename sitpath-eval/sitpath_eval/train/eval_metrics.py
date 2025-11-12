from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float, float]:
    values = np.asarray(values)
    rng = np.random.default_rng(0)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = samples.mean(axis=1)
    lower = np.percentile(means, 100 * (alpha / 2))
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return float(values.mean()), float(lower), float(upper)


def aggregate_metrics(results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    agg: Dict[str, Dict[str, float]] = {}
    metric_names = results[0].keys()
    for metric in metric_names:
        values = np.array([run[metric] for run in results])
        mean, lower, upper = bootstrap_ci(values)
        agg[metric] = {
            "mean": mean,
            "ci_lower": lower,
            "ci_upper": upper,
        }
    return agg


def save_metrics_table(metrics_dict: Dict[str, Dict[str, float]], path_csv: str, path_tex: str) -> None:
    csv_path = Path(path_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path = Path(path_tex)
    tex_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Mean", "95% CI Lower", "95% CI Upper"])
        for metric, stats in metrics_dict.items():
            writer.writerow([metric, stats["mean"], stats["ci_lower"], stats["ci_upper"]])

    lines = ["\\begin{tabular}{lccc}", "Metric & Mean & 95\\% CI Lower & 95\\% CI Upper \\\\", "\\hline"]
    for metric, stats in metrics_dict.items():
        lines.append(f"{metric} & {stats['mean']:.3f} & {stats['ci_lower']:.3f} & {stats['ci_upper']:.3f} \\\\")
    lines.append("\\end{tabular}")
    tex_path.write_text("\n".join(lines))


if __name__ == "__main__":
    demo_results = [
        {"ade": 0.8, "fde": 1.5, "miss_rate": 0.2},
        {"ade": 0.85, "fde": 1.6, "miss_rate": 0.18},
        {"ade": 0.78, "fde": 1.4, "miss_rate": 0.22},
    ]
    agg = aggregate_metrics(demo_results)
    out_dir = Path("artifacts/tables")
    save_metrics_table(agg, out_dir / "demo_metrics.csv", out_dir / "demo_metrics.tex")
