import numpy as np
import torch

from sitpath_eval.tokens import (
    SitPathTokenizer,
    Vocabulary,
    decode_tokens,
    precompute_tokens,
)
from sitpath_eval.utils.device import get_device

DEVICE = get_device("test")  # dynamic device selection for safe testing


def make_tokenizer(M=8, R=5.0, B=4):
    vocab = Vocabulary(sector_count=M, radial_bins=B)
    tokenizer = SitPathTokenizer(vocab=vocab, M=M, R=R, B=B)
    return tokenizer, vocab


def test_tokenizer_encodes_arange_sequence():
    tokenizer, _ = make_tokenizer()
    traj = np.stack([np.arange(10), np.arange(10)], axis=1).astype(np.float32)
    tokens = tokenizer.encode_trajectory(traj)

    assert len(tokens) == 10
    assert isinstance(tokens[0], int)


def test_vocabulary_roundtrip():
    vocab = Vocabulary()
    token_tuple = (3, 1, 2, 0)
    token_id = vocab.encode_tuple(token_tuple)

    assert vocab.decode_id(token_id) == token_tuple


def test_precompute_tokens_creates_npz(tmp_path):
    tokenizer, _ = make_tokenizer()
    traj = np.stack([np.arange(6), np.arange(6)], axis=1).astype(np.float32)
    dataset = [{"pos": torch.tensor(traj, device=DEVICE)}]
    out_path = tmp_path / "cache" / "tokens.npz"

    precompute_tokens(dataset, tokenizer, str(out_path))

    assert out_path.exists()
    npz = np.load(out_path, allow_pickle=True)
    tokens_arr = npz["tokens"]
    assert len(tokens_arr) == 1
    assert len(tokens_arr[0]) == len(traj)


def test_inverse_tokenizer_returns_coordinates():
    tokenizer, vocab = make_tokenizer()
    traj = np.stack([np.arange(5), np.arange(5)], axis=1).astype(np.float32)
    tokens = tokenizer.encode_trajectory(traj)

    decoded = decode_tokens(tokens, vocab)

    assert decoded.shape == (len(tokens) + 1, 2)
