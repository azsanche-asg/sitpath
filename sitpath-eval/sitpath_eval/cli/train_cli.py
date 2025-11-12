from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sitpath_eval.models import CoordGRU, CoordTransformer
from sitpath_eval.train.metrics import compute_metrics

MODEL_DIR = Path("artifacts/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_MAP = {
    "coord_gru": CoordGRU,
    "coord_transformer": CoordTransformer,
}


def make_synthetic_dataset(
    num_samples: int = 64,
    obs_len: int = 8,
    pred_len: int = 12,
) -> Tuple[TensorDataset, TensorDataset]:
    rng = np.random.default_rng(0)
    total_len = obs_len + pred_len
    data = rng.normal(size=(num_samples, total_len, 2)).astype(np.float32)
    trajectories = data.cumsum(axis=1)
    obs = trajectories[:, :obs_len]
    targets = trajectories[:, obs_len:]
    tensor_x = torch.from_numpy(obs)
    tensor_y = torch.from_numpy(targets)
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
    train_p.add_argument("--model", choices=list(MODEL_MAP.keys()), default="coord_gru")

    eval_p = subparsers.add_parser("eval", help="Evaluate saved model.")
    eval_p.add_argument("--model", choices=list(MODEL_MAP.keys()), default="coord_gru")
    return parser


def train_command(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds = make_synthetic_dataset()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    ModelCls = MODEL_MAP[args.model]
    model = ModelCls().to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for obs, targets in train_loader:
            obs = obs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            preds = model(obs)
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
    _, val_ds = make_synthetic_dataset()
    val_loader = DataLoader(val_ds, batch_size=16)
    ModelCls = MODEL_MAP[args.model]
    model = ModelCls().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds_list = []
    targets_list = []
    with torch.no_grad():
        for obs, targets in val_loader:
            obs = obs.to(device)
            targets = targets.to(device)
            preds = model(obs)
            preds_list.append(preds.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
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
