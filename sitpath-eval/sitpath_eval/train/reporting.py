from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CapacityReport:
    params: int
    flops: str
    model_name: str


def summarize_capacity(model, model_name: str) -> CapacityReport:
    from .fairness import count_trainable_params, try_count_flops

    params = count_trainable_params(model)
    flops = try_count_flops(model)
    flops_str = f"{flops:.3e}" if flops is not None else "n/a"
    return CapacityReport(params=params, flops=flops_str, model_name=model_name)
