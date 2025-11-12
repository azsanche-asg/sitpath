import json
from pathlib import Path

import torch

from sitpath_eval.cli import eval_cli
from sitpath_eval.train.eval_data_efficiency import (
    aggregate_by_fraction,
    save_efficiency_table,
    subsample_dataset,
    train_and_evaluate,
)
from sitpath_eval.models.coord_gru import CoordGRU
from sitpath_eval.utils.device import get_device

DEVICE = get_device("test")  # dynamic device selection for safe testing


def make_dataset(n=20):
    torch.manual_seed(0)
    obs = torch.randn(n, 8, 2, device=DEVICE)
    targets = torch.randn(n, 12, 2, device=DEVICE)
    return torch.utils.data.TensorDataset(obs, targets)


def test_subsample_dataset_length():
    dataset = make_dataset()
    subset = subsample_dataset(dataset, fraction=0.5, seed=0)
    assert len(subset) == max(1, int(len(dataset) * 0.5))


def test_train_and_evaluate_returns_metrics():
    dataset = make_dataset()
    results = train_and_evaluate(CoordGRU, dataset, fractions=[0.5], epochs=1)
    assert len(results) == 1
    entry = results[0]
    assert "fraction" in entry and entry["fraction"] == 0.5
    assert all(key in entry["metrics"] for key in ("ade", "fde", "miss_rate"))


def test_aggregate_by_fraction_structure():
    dataset = make_dataset()
    results = train_and_evaluate(CoordGRU, dataset, fractions=[0.5], epochs=1)
    agg = aggregate_by_fraction(results)
    assert 0.5 in agg
    assert "ade" in agg[0.5]
    assert "ci_lower" in agg[0.5]["ade"]


def test_save_efficiency_table_outputs(tmp_path):
    data = {
        0.5: {
            "ade": {"mean": 1.0, "ci_lower": 0.9, "ci_upper": 1.1},
        }
    }
    csv_path = tmp_path / "eff.csv"
    tex_path = tmp_path / "eff.tex"
    save_efficiency_table(data, csv_path, tex_path)
    assert csv_path.exists()
    assert tex_path.exists()


def test_eval_cli_data_efficiency(tmp_path, monkeypatch):
    out_dir = tmp_path / "tables"
    eval_cli.main(
        [
            "data_efficiency",
            "--model",
            "coord_gru",
            "--fractions",
            "0.5",
            "--outdir",
            str(out_dir),
            "--epochs",
            "1",
        ]
    )
    assert any(out_dir.glob("data_efficiency_coord_gru.csv"))
