from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sitpath_eval.models import (
    CoordGRU,
    CoordTransformer,
    SitPathGRU,
    SitPathTransformer,
)
from sitpath_eval.models.raster_gru import RasterGRU
from sitpath_eval.tokens import Vocabulary
from sitpath_eval.tokens.tokenizer import SitPathTokenizer
from sitpath_eval.train.fairness import (
    assert_capacity_parity,
    count_trainable_params,
    try_count_flops,
)
from sitpath_eval.train.metrics import compute_metrics

OBS_LEN = 8
PRED_LEN = 12
ARTIFACTS_DIR = Path("artifacts")
MODEL_DIR = ARTIFACTS_DIR / "models"
VOCAB_PATH = ARTIFACTS_DIR / "vocab.json"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SPECS = {
    "coord_gru": {"cls": CoordGRU, "kind": "coord"},
    "coord_transformer": {"cls": CoordTransformer, "kind": "coord"},
    "sitpath_gru": {"cls": SitPathGRU, "kind": "token"},
    "sitpath_transformer": {"cls": SitPathTransformer, "kind": "token"},
    "raster_gru": {"cls": RasterGRU, "kind": "raster"},
}

DEFAULT_VOCAB_SIZE = 32


def load_or_build_vocab(path: Path = VOCAB_PATH, default_size: int = DEFAULT_VOCAB_SIZE):
    if path.exists():
        vocab = Vocabulary.load(path)
    else:
        vocab = Vocabulary()
        for s in range(vocab.sector_count):
            for r in range(vocab.radial_bins):
                if len(vocab) >= default_size:
                    break
                vocab.add((s, r, 0, 0))
            if len(vocab) >= default_size:
                break
    tokenizer = SitPathTokenizer(vocab=vocab)
    return vocab, tokenizer


def make_synthetic_dataset(
    num_samples: int = 64,
    obs_len: int = OBS_LEN,
    pred_len: int = PRED_LEN,
    mode: str = "coord",
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> Tuple[TensorDataset, TensorDataset]:
    rng = np.random.default_rng(0)
    if mode == "coord":
        total_len = obs_len + pred_len
        data = rng.normal(size=(num_samples, total_len, 2)).astype(np.float32)
        trajectories = data.cumsum(axis=1)
        obs = trajectories[:, :obs_len]
        targets = trajectories[:, obs_len:]
        tensor_x = torch.from_numpy(obs)
        tensor_y = torch.from_numpy(targets)
    elif mode == "token":
        obs = rng.integers(0, vocab_size, size=(num_samples, obs_len))
        targets = rng.integers(0, vocab_size, size=(num_samples, pred_len))
        tensor_x = torch.from_numpy(obs.astype(np.int64))
        tensor_y = torch.from_numpy(targets.astype(np.int64))
    elif mode == "raster":
        obs = torch.rand(num_samples, obs_len, 3, 32, 32)
        targets = torch.rand(num_samples, pred_len, 2)
        tensor_x = obs
        tensor_y = targets
    else:
        raise ValueError(f"Unknown mode {mode}")
    midpoint = num_samples // 2
    train_ds = TensorDataset(tensor_x[:midpoint], tensor_y[:midpoint])
    val_ds = TensorDataset(tensor_x[midpoint:], tensor_y[midpoint:])
    return train_ds, val_ds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SitPath training CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train", help="Run a dummy training loop.")
    train_p.add_argument("--epochs", type=int, default=3)
    train_p.add_argument("--batch-size", type=int, default=16)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--model", choices=list(MODEL_SPECS.keys()), default="coord_gru")
    train_p.add_argument("--enforce_parity", action="store_true")

    eval_p = subparsers.add_parser("eval", help="Evaluate saved model.")
    eval_p.add_argument("--model", choices=list(MODEL_SPECS.keys()), default="coord_gru")
    return parser


def train_command(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = MODEL_SPECS[args.model]
    kind = spec["kind"]
    is_token_model = kind == "token"
    is_raster_model = kind == "raster"

    if is_token_model:
        vocab, tokenizer = load_or_build_vocab()
        vocab_size = len(vocab)
        print(f"[sitpath-eval] Training SitPath model with vocab size {vocab_size}")
        print(f"[sitpath-eval] Loaded tokenizer={tokenizer.__class__.__name__}")
        ModelCls = spec["cls"]
        model = ModelCls(vocab_size=vocab_size, pred_len=PRED_LEN, obs_len=OBS_LEN).to(device)
        train_ds, val_ds = make_synthetic_dataset(mode="token", vocab_size=vocab_size)
        loss_fn = torch.nn.CrossEntropyLoss()
    elif is_raster_model:
        ModelCls = spec["cls"]
        model = ModelCls(obs_len=OBS_LEN, pred_len=PRED_LEN).to(device)
        params = model.num_parameters()
        print(f"[sitpath-eval] Training RasterGRU baseline (model params: {params})")
        train_ds, val_ds = make_synthetic_dataset(mode="raster")
        loss_fn = torch.nn.MSELoss()
    else:
        train_ds, val_ds = make_synthetic_dataset()
        ModelCls = spec["cls"]
        model = ModelCls(obs_len=OBS_LEN, pred_len=PRED_LEN).to(device)
        loss_fn = torch.nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    if args.enforce_parity and not (is_token_model or is_raster_model):
        baseline_name = "coord_gru" if args.model != "coord_gru" else "coord_transformer"
        baseline_cls = MODEL_SPECS[baseline_name]["cls"]
        baseline_model = baseline_cls(obs_len=OBS_LEN, pred_len=PRED_LEN)
        params_ok = False
        flops_state = "unknown"
        base_params = count_trainable_params(baseline_model)
        comp_params = count_trainable_params(model)
        base_flops = try_count_flops(baseline_model)
        comp_flops = try_count_flops(model)
        try:
            assert_capacity_parity(baseline_model, model)
            params_ok = True
            if base_flops is not None and comp_flops is not None:
                flops_state = "true"
            elif base_flops is None and comp_flops is None:
                flops_state = "unknown"
            else:
                flops_state = "partial"
        except AssertionError as exc:
            print(f"[PARITY] failure: {exc}")
            raise
        print(
            f"[PARITY] params_ok={'true' if params_ok else 'false'} "
            f"flops_ok={flops_state} tol_params=2% tol_flops=5%"
        )
    else:
        params = model.num_parameters()
        print(
            f"[PARITY] model={args.model} params={params} "
            "single-model run; use --enforce_parity for paired paper experiments."
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for obs, targets in train_loader:
            obs = obs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            preds = model(obs)
            if is_token_model:
                vocab_size = preds.size(-1)
                loss = loss_fn(preds.view(-1, vocab_size), targets.view(-1))
            elif is_raster_model:
                loss = loss_fn(preds, targets)
            else:
                loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
        print(f"[sitpath-eval] Epoch {epoch} loss {loss.item():.4f}")

    model_path = MODEL_DIR / f"{args.model}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[sitpath-eval] saved model to {model_path}")


def eval_command(args: argparse.Namespace) -> None:
    model_path = MODEL_DIR / f"{args.model}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found for {args.model}. Run train first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = MODEL_SPECS[args.model]
    kind = spec["kind"]
    is_token_model = kind == "token"
    is_raster_model = kind == "raster"

    if is_token_model:
        vocab, _ = load_or_build_vocab()
        vocab_size = len(vocab)
        _, val_ds = make_synthetic_dataset(mode="token", vocab_size=vocab_size)
        ModelCls = spec["cls"]
        model = ModelCls(vocab_size=vocab_size, pred_len=PRED_LEN, obs_len=OBS_LEN).to(device)
    elif is_raster_model:
        _, val_ds = make_synthetic_dataset(mode="raster")
        ModelCls = spec["cls"]
        model = ModelCls(obs_len=OBS_LEN, pred_len=PRED_LEN).to(device)
    else:
        _, val_ds = make_synthetic_dataset()
        ModelCls = spec["cls"]
        model = ModelCls(obs_len=OBS_LEN, pred_len=PRED_LEN).to(device)

    val_loader = DataLoader(val_ds, batch_size=16)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds_list = []
    targets_list = []
    with torch.no_grad():
        for obs, targets in val_loader:
            obs = obs.to(device)
            targets = targets.to(device)
            preds = model(obs)
            if is_token_model:
                pred_ids = preds.argmax(dim=-1)
                preds_list.append(pred_ids.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
            elif is_raster_model:
                preds_list.append(preds.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
            else:
                preds_list.append(preds.cpu().numpy())
                targets_list.append(targets.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    if is_token_model:
        accuracy = (preds == targets).mean()
        print(f"[sitpath-eval] token eval accuracy={accuracy:.3f}")
    elif is_raster_model:
        mse = np.mean((preds - targets) ** 2)
        print(f"[sitpath-eval] raster eval mse={mse:.4f}")
    else:
        metrics = compute_metrics(preds, targets)
        print(f"[sitpath-eval] eval metrics: {metrics}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        train_command(args)
    elif args.command == "eval":
        eval_command(args)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
