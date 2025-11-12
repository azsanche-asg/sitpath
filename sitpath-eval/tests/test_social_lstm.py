import torch

from sitpath_eval.models.coord_gru import CoordGRU
from sitpath_eval.models.social_lstm import SocialLSTM


def test_social_lstm_forward_and_shape():
    obs = torch.rand(4, 8, 2)
    model = SocialLSTM()
    out = model(obs)
    assert out.shape == (4, 12, 2)


def test_social_lstm_gradients_and_capacity():
    obs = torch.rand(4, 8, 2)
    model = SocialLSTM()
    out = model(obs)
    loss = out.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)

    params = model.num_parameters()
    coord_params = CoordGRU().num_parameters()
    assert params > 0
    assert params / coord_params < 3


def test_social_lstm_pooling_radius_zero_matches_independent():
    obs = torch.rand(2, 8, 2)
    social_model = SocialLSTM(pooling_radius=0.0)
    coord_model = CoordGRU()
    social_out = social_model(obs)
    coord_out = coord_model(obs)
    assert torch.allclose(social_out, coord_out, atol=1e-4)
