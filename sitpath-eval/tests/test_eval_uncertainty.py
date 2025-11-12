import os
os.environ.setdefault("SITPATH_MODE", "auto")

import numpy as np

from sitpath_eval.cli import eval_cli
from sitpath_eval.train.eval_uncertainty import (
    compute_uncertainty_metrics,
    diversity_at_k,
    expected_calibration_error,
    negative_log_likelihood,
)


def test_negative_log_likelihood_positive():
    preds = np.zeros((10, 2))
    gts = np.zeros((10, 2))
    nll = negative_log_likelihood(preds, gts, sigma=1.0)
    assert nll > 0


def test_expected_calibration_error_bounds():
    probs = np.linspace(0.1, 0.9, 10)
    labels = (probs > 0.5).astype(float)
    ece = expected_calibration_error(probs, labels)
    assert 0 <= ece <= 1


def test_diversity_at_k_increases_with_spread():
    tight = np.zeros((1, 5, 2))
    spread = np.random.normal(size=(1, 5, 2))
    d_tight = diversity_at_k(tight)
    d_spread = diversity_at_k(spread)
    assert d_spread >= d_tight


def test_compute_uncertainty_metrics_keys():
    preds_k = np.random.normal(size=(8, 3, 2))
    gts = np.random.normal(size=(8, 2))
    probs = np.random.uniform(size=(8,))
    metrics = compute_uncertainty_metrics(preds_k, gts, probs)
    for key in ("nll", "ece", "div_k"):
        assert key in metrics


def test_eval_cli_uncertainty(tmp_path):
    out_dir = tmp_path / "tables"
    eval_cli.main(
        [
            "uncertainty",
            "--model",
            "coord_gru",
            "--dataset",
            "eth_ucy",
            "--outdir",
            str(out_dir),
            "--samples",
            "5",
        ]
    )
    assert any(out_dir.glob("uncertainty_coord_gru_eth_ucy.csv"))
