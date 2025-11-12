import os
os.environ.setdefault("SITPATH_MODE", "auto")

import torch

from sitpath_eval.models.coord_gru import CoordGRU
from sitpath_eval.models.raster_gru import RasterGRU
from sitpath_eval.utils.device import get_device

device = get_device("auto")  # dynamic device selection for safe testing


def test_raster_gru_forward_shape_and_gradients():
    x = torch.rand(2, 8, 3, 32, 32, device=device)
    model = RasterGRU().to(device)

    out = model(x)
    assert out.shape == (2, 12, 2)

    loss = out.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_raster_gru_parameter_count():
    model = RasterGRU().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert isinstance(params, int)
    assert params > 0

    coord_model = CoordGRU()
    coord_params = coord_model.num_parameters()
    # Allow up to 10× larger parameter count due to CNN encoder
    assert params / coord_params < 10
