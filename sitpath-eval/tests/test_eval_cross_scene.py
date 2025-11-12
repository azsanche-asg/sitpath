import os
os.environ.setdefault("SITPATH_MODE", "auto")

import pytest

from sitpath_eval.cli import eval_cli
from sitpath_eval.models.coord_gru import CoordGRU
from sitpath_eval.train.eval_cross_scene import (
    aggregate_cross_scene,
    get_scene_splits,
    save_cross_scene_table,
    train_and_eval_cross_scene,
)


def test_get_scene_splits_eth_ucy():
    splits = get_scene_splits("eth_ucy")
    assert splits
    for train, test in splits:
        assert isinstance(train, list)
        assert isinstance(test, list)
        assert len(test) == 1


def test_train_and_eval_cross_scene_returns_metrics():
    splits = get_scene_splits("eth_ucy")[:1]
    results = train_and_eval_cross_scene(CoordGRU, "eth_ucy", splits, epochs=1)
    assert len(results) == 1
    metrics = results[0]["metrics"]
    assert "ade" in metrics and "fde" in metrics and "miss_rate" in metrics


def test_aggregate_cross_scene_structure():
    splits = get_scene_splits("eth_ucy")[:1]
    results = train_and_eval_cross_scene(CoordGRU, "eth_ucy", splits, epochs=1)
    agg = aggregate_cross_scene(results)
    assert "ade" in agg
    assert "ci_lower" in agg["ade"]
    assert "ci_upper" in agg["ade"]


def test_save_cross_scene_table_outputs(tmp_path):
    data = {
        "ade": {"mean": 1.0, "ci_lower": 0.9, "ci_upper": 1.1},
    }
    csv_path = tmp_path / "cross.csv"
    tex_path = tmp_path / "cross.tex"
    save_cross_scene_table(data, csv_path, tex_path)
    assert csv_path.exists() and tex_path.exists()


def test_eval_cli_cross_scene(tmp_path):
    out_dir = tmp_path / "tables"
    eval_cli.main(
        [
            "cross_scene",
            "--model",
            "coord_gru",
            "--dataset",
            "eth_ucy",
            "--outdir",
            str(out_dir),
            "--epochs",
            "1",
        ]
    )
    assert any(out_dir.glob("cross_scene_coord_gru_eth_ucy.csv"))
