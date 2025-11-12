import os
os.environ.setdefault("SITPATH_MODE", "auto")

import torch

from sitpath_eval.cli import eval_cli
from sitpath_eval.train.eval_ablation import (
    ablation_grid,
    aggregate_ablation,
    save_ablation_table,
    train_and_eval_ablation,
)
from sitpath_eval.utils.device import get_device

device = get_device("auto")  # dynamic device selection for safe testing


def make_dataset():
    torch.manual_seed(0)
    obs = torch.randint(0, 16, (10, 8), device=device)
    targets = torch.randint(0, 16, (10, 12), device=device)
    return torch.utils.data.TensorDataset(obs, targets)


def test_ablation_grid_structure():
    grid = ablation_grid()
    assert len(grid) == 36
    assert all(set(cfg.keys()) == {"M", "R", "collapse", "tempo"} for cfg in grid)


def test_train_and_eval_ablation_metrics():
    dataset = make_dataset()
    results = train_and_eval_ablation(dataset=dataset, grid=ablation_grid()[:2], epochs=1)
    assert len(results) == 2
    for entry in results:
        assert "ade" in entry and "fde" in entry and "minade_k" in entry


def test_aggregate_ablation_keys():
    dataset = make_dataset()
    results = train_and_eval_ablation(dataset=dataset, grid=ablation_grid()[:1], epochs=1)
    agg = aggregate_ablation(results)
    key = list(agg.keys())[0]
    assert "ade" in agg[key]
    assert "ci_lower" in agg[key]["ade"]


def test_save_ablation_table_outputs(tmp_path):
    data = {
        (8, 3, True, "on"): {
            "ade": {"mean": 1.0},
            "fde": {"mean": 2.0},
            "minade_k": {"mean": 1.5},
            "miss_rate": {"mean": 0.1},
        }
    }
    csv_path = tmp_path / "ablation.csv"
    tex_path = tmp_path / "ablation.tex"
    save_ablation_table(data, csv_path, tex_path)
    assert csv_path.exists()
    assert tex_path.exists()


def test_eval_cli_ablation(tmp_path):
    out_dir = tmp_path / "tables"
    eval_cli.main(
        [
            "ablation",
            "--model",
            "sitpath_gru",
            "--dataset",
            "eth_ucy",
            "--outdir",
            str(out_dir),
            "--epochs",
            "1",
        ]
    )
    assert any(out_dir.glob("ablation_sitpath_gru_eth_ucy.csv"))
