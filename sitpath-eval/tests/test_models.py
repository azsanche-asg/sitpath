import os
os.environ.setdefault("SITPATH_MODE", "auto")

import torch

from sitpath_eval.models import CoordGRU, CoordTransformer
from sitpath_eval.utils.device import get_device

device = get_device("auto")  # dynamic device selection for safe testing


def make_observations(batch=2, obs_len=8):
    torch.manual_seed(0)
    return torch.randn(batch, obs_len, 2, device=device)


def test_models_forward_and_shapes():
    obs = make_observations()
    gru = CoordGRU().to(device)
    transformer = CoordTransformer().to(device)

    out_gru = gru(obs)
    out_trans = transformer(obs)

    assert out_gru.shape == (2, 12, 2)
    assert out_trans.shape == (2, 12, 2)

    params_gru = gru.num_parameters()
    params_trans = transformer.num_parameters()
    # sanity: ensure both models have trainable parameters and reasonable scale
    assert params_gru > 0 and params_trans > 0
    assert params_trans / params_gru < 200  # transformer not more than 200× larger


def test_models_backward_pass():
    obs = make_observations()
    target = torch.zeros(2, 12, 2, device=device)

    gru = CoordGRU().to(device)
    transformer = CoordTransformer().to(device)

    out_gru = gru(obs)
    loss_gru = torch.nn.functional.mse_loss(out_gru, target)
    loss_gru.backward()
    assert any(p.grad is not None for p in gru.parameters() if p.requires_grad)

    gru.zero_grad()

    out_trans = transformer(obs)
    loss_trans = torch.nn.functional.mse_loss(out_trans, target)
    loss_trans.backward()
    assert any(p.grad is not None for p in transformer.parameters() if p.requires_grad)
