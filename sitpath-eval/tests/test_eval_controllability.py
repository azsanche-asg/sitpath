import os
os.environ.setdefault("SITPATH_MODE", "auto")

import numpy as np

from sitpath_eval.cli import eval_cli
from sitpath_eval.train.eval_controllability import (
    aggregate_controllability,
    apply_edit_rule,
    controllability_metrics,
    compute_constraint_satisfaction,
)


def make_trajs(batch=4, pred_len=12):
    rng = np.random.default_rng(0)
    orig = rng.normal(size=(batch, pred_len, 2)).cumsum(axis=1)
    gts = orig + rng.normal(scale=0.1, size=orig.shape)
    return orig, gts


def test_apply_edit_rule_shape_and_effect():
    orig, _ = make_trajs()
    edited = apply_edit_rule(orig, "keep_right")
    assert edited.shape == orig.shape
    assert np.all(edited[..., 0] >= 0)


def test_compute_constraint_satisfaction_bounds():
    orig, _ = make_trajs()
    edited = apply_edit_rule(orig, "avoid_front")
    csr = compute_constraint_satisfaction(orig, edited, "avoid_front")
    assert 0.0 <= csr <= 1.0


def test_controllability_metrics_keys():
    orig, gts = make_trajs()
    edited = apply_edit_rule(orig, "slow_down")
    metrics = controllability_metrics(orig, edited, gts, "slow_down")
    for key in ("csr", "gr", "delta_ade"):
        assert key in metrics


def test_aggregate_controllability_structure():
    orig, gts = make_trajs()
    edited = apply_edit_rule(orig, "keep_right")
    metrics = controllability_metrics(orig, edited, gts, "keep_right")
    agg = aggregate_controllability([metrics])
    assert "csr" in agg and "ci_lower" in agg["csr"]


def test_eval_cli_controllability(tmp_path):
    out_dir = tmp_path / "tables"
    eval_cli.main(
        [
            "controllability",
            "--model",
            "sitpath_gru",
            "--rule",
            "avoid_front",
            "--dataset",
            "eth_ucy",
            "--outdir",
            str(out_dir),
        ]
    )
    assert any(out_dir.glob("controllability_sitpath_gru_avoid_front_eth_ucy.csv"))
