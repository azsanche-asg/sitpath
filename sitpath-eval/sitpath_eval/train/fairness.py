from __future__ import annotations

import math
from typing import Optional

import torch.nn as nn


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def try_count_flops(model: nn.Module) -> Optional[float]:
    """
    Optional: return approximate FLOPs if ptflops or fvcore is available.
    If neither is installed, return None (the caller can fall back to params-only).
    """
    try:
        from ptflops import get_model_complexity_info

        # Use a tiny dummy input shape typical for your seq models, e.g. (8, 2)
        # Note: wrap model if needed to accept a single input tensor.
        macs, _ = get_model_complexity_info(
            model,
            (8, 2),
            as_strings=False,
            print_per_layer_stat=False,
        )
        return float(macs) * 2.0  # MACs->FLOPs approximate
    except Exception:
        return None


def assert_capacity_parity(
    model_a: nn.Module,
    model_b: nn.Module,
    rel_tol_params: float = 0.02,
    rel_tol_flops: Optional[float] = None,
):
    """
    Raise AssertionError if models exceed capacity tolerance.

    - Params must match within rel_tol_params (default 2%).
    - If FLOPs available for both, they must match within rel_tol_flops (default 5% if None).
    """

    pa = count_trainable_params(model_a)
    pb = count_trainable_params(model_b)
    maxp = max(pa, pb)
    if maxp == 0:
        raise AssertionError("Zero-parameter model encountered.")
    if abs(pa - pb) / maxp > rel_tol_params:
        raise AssertionError(f"Param parity failed: {pa} vs {pb} (tol={rel_tol_params * 100:.1f}%).")

    fa = try_count_flops(model_a)
    fb = try_count_flops(model_b)
    if fa is not None and fb is not None:
        tol = 0.05 if rel_tol_flops is None else rel_tol_flops
        maxf = max(fa, fb)
        if maxf == 0 or not math.isfinite(maxf):
            return
        if abs(fa - fb) / maxf > tol:
            raise AssertionError(
                f"FLOPs parity failed: {fa:.3e} vs {fb:.3e} (tol={tol * 100:.1f}%)."
            )
