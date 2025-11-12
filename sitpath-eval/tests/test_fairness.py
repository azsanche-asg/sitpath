import os
os.environ.setdefault("SITPATH_MODE", "auto")

import pytest

from sitpath_eval.models.coord_gru import CoordGRU
from sitpath_eval.models.coord_transformer import CoordTransformer
from sitpath_eval.train.fairness import assert_capacity_parity, count_trainable_params


def test_fairness_params_relaxed():
    # Make two small GRUs with slightly different hidden sizes and check a 10% tolerance passes.
    a = CoordGRU(obs_len=8, pred_len=12, hidden_size=40)
    b = CoordGRU(obs_len=8, pred_len=12, hidden_size=42)
    assert count_trainable_params(a) > 0
    assert count_trainable_params(b) > 0
    assert_capacity_parity(a, b, rel_tol_params=0.10)


def test_fairness_params_strict_fails():
    # Transformer vs GRU should fail a strict 2% tolerance.
    a = CoordGRU(obs_len=8, pred_len=12, hidden_size=32)
    b = CoordTransformer(obs_len=8, pred_len=12, d_model=64)
    with pytest.raises(AssertionError):
        assert_capacity_parity(a, b, rel_tol_params=0.02)
