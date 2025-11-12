import torch

from sitpath_eval.models import SitPathGRU, SitPathTransformer


def random_tokens(batch=2, seq_len=8, vocab_size=32):
    torch.manual_seed(0)
    return torch.randint(0, vocab_size, (batch, seq_len))


def test_sitpath_models_forward_and_shapes():
    tokens = random_tokens()

    gru = SitPathGRU(vocab_size=32)
    trans = SitPathTransformer(vocab_size=32)

    logits_gru = gru(tokens)
    logits_trans = trans(tokens)

    assert logits_gru.shape == (2, 12, 32)
    assert logits_trans.shape == (2, 12, 32)


def test_sitpath_models_sampling_and_gradients():
    tokens = random_tokens()
    target = torch.zeros(2, 12, 32)

    gru = SitPathGRU(vocab_size=32)
    trans = SitPathTransformer(vocab_size=32)

    samples_gru = gru.sample(tokens, K=3)
    samples_trans = trans.sample(tokens, K=3)

    assert samples_gru.shape == (2, 12, 3)
    assert samples_trans.shape == (2, 12, 3)
    assert samples_gru.dtype == torch.long
    assert samples_trans.dtype == torch.long

    assert gru.num_parameters() > 0
    assert trans.num_parameters() > 0

    logits_gru = gru(tokens)
    loss_gru = torch.nn.functional.mse_loss(logits_gru, target)
    loss_gru.backward()
    assert any(p.grad is not None for p in gru.parameters() if p.requires_grad)

    gru.zero_grad()

    logits_trans = trans(tokens)
    loss_trans = torch.nn.functional.mse_loss(logits_trans, target)
    loss_trans.backward()
    assert any(p.grad is not None for p in trans.parameters() if p.requires_grad)
