import csv
import numpy as np
import pytest

from sitpath_eval.train.callbacks import CSVLogger, EarlyStopping
from sitpath_eval.train.metrics import ade, compute_metrics, fde, minade_k, miss_rate


def random_preds_targets(n=4, t=6):
    rng = np.random.default_rng(0)
    preds = rng.normal(size=(n, t, 2))
    targets = rng.normal(size=(n, t, 2))
    return preds, targets


def test_metrics_functions_return_reasonable_values():
    preds, targets = random_preds_targets()
    preds_k = np.stack([preds, preds + 0.1], axis=0)

    metrics = [
        ade(preds, targets),
        fde(preds, targets),
        minade_k(preds_k, targets),
        miss_rate(preds, targets),
    ]

    for value in metrics:
        assert isinstance(value, float)
        assert 0 <= value < 100


def test_compute_metrics_contains_core_keys():
    preds, targets = random_preds_targets()
    result = compute_metrics(preds, targets)

    assert "ade" in result
    assert "fde" in result


def test_early_stopping_triggers():
    stopper = EarlyStopping(patience=2, min_delta=0.0)
    values = [1.0, 1.1, 1.2]
    flags = [stopper.step(v) for v in values]

    assert flags[-1] is True


def test_csv_logger_writes_headers(tmp_path):
    log_path = tmp_path / "logs.csv"
    logger = CSVLogger(str(log_path))
    logger.log({"epoch": 1, "train_loss": 0.5, "val_loss": 0.6})
    logger.close()

    with log_path.open() as fh:
        reader = csv.reader(fh)
        header = next(reader)

    assert header == ["epoch", "train_loss", "val_loss"]
