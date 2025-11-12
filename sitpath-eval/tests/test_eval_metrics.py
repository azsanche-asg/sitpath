import json
from pathlib import Path

import numpy as np
import pytest

from sitpath_eval.cli import eval_cli
from sitpath_eval.train.eval_metrics import (
    aggregate_metrics,
    bootstrap_ci,
    save_metrics_table,
)


def test_bootstrap_ci_monotonic():
    values = np.array([1, 2, 3, 4, 5])
    mean, low, up = bootstrap_ci(values, n_boot=100, alpha=0.1)
    assert low < mean < up


def test_aggregate_metrics_structure():
    results = [
        {"ade": 1.0, "fde": 2.0},
        {"ade": 1.1, "fde": 2.1},
        {"ade": 0.9, "fde": 1.9},
    ]
    agg = aggregate_metrics(results)
    assert "ade" in agg and "fde" in agg
    assert "ci_lower" in agg["ade"]
    assert "ci_upper" in agg["ade"]


def test_save_metrics_table_outputs(tmp_path):
    metrics = {
        "ade": {"mean": 1.0, "ci_lower": 0.9, "ci_upper": 1.1},
        "fde": {"mean": 2.0, "ci_lower": 1.8, "ci_upper": 2.2},
    }
    csv_path = tmp_path / "metrics.csv"
    tex_path = tmp_path / "metrics.tex"
    save_metrics_table(metrics, csv_path, tex_path)

    assert csv_path.exists()
    assert tex_path.exists()
    lines = csv_path.read_text().splitlines()
    assert "Metric,Mean,95% CI Lower,95% CI Upper" in lines[0]


def test_eval_cli_metrics(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    data = {"ade": 1.0, "fde": 2.0}
    for i in range(3):
        (logs_dir / f"run_{i}.json").write_text(json.dumps(data))

    out_dir = tmp_path / "tables"

    eval_cli.main(
        [
            "metrics",
            "--runs",
            str(logs_dir / "*.json"),
            "--outdir",
            str(out_dir),
        ]
    )

    assert (out_dir / "aggregated_metrics.csv").exists()
    assert (out_dir / "aggregated_metrics.tex").exists()
