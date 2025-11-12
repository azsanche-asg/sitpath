from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sitpath_eval.train.metrics import compute_metrics

MODEL_PATH = Path("artifacts/model.pt")


def make_synthetic_dataset(num_samples: int = 64, seq_len: int = 20) -> Tuple[TensorDataset, TensorDataset]:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(num_samples, seq_len, 2)).astype(np.float32)
    targets = data.cumsum(axis=1)
    tensor_x = torch.from_numpy(data)
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

    subparsers.add_parser("eval", help="Evaluate saved model.")
    return parser


def train_command(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds = make_synthetic_dataset()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = nn.Linear(2, 2).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
        print(f"[sitpath-eval] epoch {epoch} loss {loss.item():.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[sitpath-eval] saved model to {MODEL_PATH}")


def eval_command(args: argparse.Namespace) -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model checkpoint not found. Run train first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_ds = make_synthetic_dataset()
    val_loader = DataLoader(val_ds, batch_size=16)
    model = nn.Linear(2, 2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    preds_list = []
    targets_list = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            preds_list.append(preds.cpu().numpy())
            targets_list.append(y.cpu().numpy())

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
