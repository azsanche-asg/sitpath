from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from sitpath_eval.models import CoordGRU, CoordTransformer, SocialLSTM
from sitpath_eval.train.eval_metrics import aggregate_metrics, save_metrics_table
from sitpath_eval.train.eval_data_efficiency import (
    aggregate_by_fraction,
    save_efficiency_table,
    train_and_evaluate,
)
from sitpath_eval.train.eval_cross_scene import (
    aggregate_cross_scene,
    get_scene_splits,
    save_cross_scene_table,
    train_and_eval_cross_scene,
)


def load_metrics_files(pattern: str) -> List[Dict[str, float]]:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No metric files matched pattern: {pattern}")
    results = []
    for path in files:
        p = Path(path)
        if p.suffix == ".json":
            data = json.loads(p.read_text())
        elif p.suffix == ".csv":
            import csv

            with p.open() as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
                if not rows:
                    continue
                data = {k: float(v) for k, v in rows[-1].items()}
        else:
            raise ValueError(f"Unsupported file type: {p}")
        results.append({k: float(v) for k, v in data.items()})
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SitPath evaluation utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    metrics_parser = subparsers.add_parser("metrics", help="Aggregate metric files into tables.")
    metrics_parser.add_argument("--runs", required=True, help="Glob pattern for metric files.")
    metrics_parser.add_argument("--outdir", default="artifacts/tables")

    eff_parser = subparsers.add_parser("data_efficiency", help="Run data-efficiency sweeps.")
    eff_parser.add_argument(
        "--model",
        choices=["coord_gru", "coord_transformer", "social_lstm"],
        default="coord_gru",
    )
    eff_parser.add_argument(
        "--fractions",
        nargs="+",
        default=["0.1", "0.25", "1.0"],
        help="Data fractions to evaluate.",
    )
    eff_parser.add_argument("--outdir", default="artifacts/tables")
    eff_parser.add_argument("--epochs", type=int, default=2)

    cross_parser = subparsers.add_parser("cross_scene", help="Leave-one-scene-out evaluation.")
    cross_parser.add_argument(
        "--model",
        choices=["coord_gru", "coord_transformer", "social_lstm"],
        default="coord_gru",
    )
    cross_parser.add_argument("--dataset", choices=["eth_ucy", "sdd_mini"], default="eth_ucy")
    cross_parser.add_argument("--outdir", default="artifacts/tables")
    cross_parser.add_argument("--epochs", type=int, default=5)
    return parser


def metrics_command(args: argparse.Namespace) -> None:
    results = load_metrics_files(args.runs)
    metrics = aggregate_metrics(results)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "aggregated_metrics.csv"
    tex_path = out_dir / "aggregated_metrics.tex"
    save_metrics_table(metrics, csv_path, tex_path)
    print(f"[sitpath-eval] Saved aggregated metrics to {csv_path} and {tex_path}")


def make_synthetic_coord_dataset(obs_len: int = 8, pred_len: int = 12, samples: int = 64):
    rng = np.random.default_rng(0)
    seq = obs_len + pred_len
    data = rng.normal(size=(samples, seq, 2)).astype(np.float32)
    trajectories = data.cumsum(axis=1)
    obs = torch.from_numpy(trajectories[:, :obs_len])
    targets = torch.from_numpy(trajectories[:, obs_len:])
    return torch.utils.data.TensorDataset(obs, targets)


MODEL_REGISTRY = {
    "coord_gru": CoordGRU,
    "coord_transformer": CoordTransformer,
    "social_lstm": SocialLSTM,
}


def data_efficiency_command(args: argparse.Namespace) -> None:
    fractions = [float(f) for f in args.fractions]
    dataset = make_synthetic_coord_dataset()
    model_cls = MODEL_REGISTRY[args.model]
    results = train_and_evaluate(
        model_cls,
        dataset,
        fractions=fractions,
        epochs=args.epochs,
    )
    aggregated = aggregate_by_fraction(results)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"data_efficiency_{args.model}.csv"
    tex_path = out_dir / f"data_efficiency_{args.model}.tex"
    save_efficiency_table(aggregated, csv_path, tex_path)
    print(f"[sitpath-eval] Saved data-efficiency results to {out_dir}")


def cross_scene_command(args: argparse.Namespace) -> None:
    splits = get_scene_splits(args.dataset)
    model_cls = MODEL_REGISTRY[args.model]
    results = train_and_eval_cross_scene(
        model_cls,
        args.dataset,
        splits,
        epochs=args.epochs,
    )
    aggregated = aggregate_cross_scene(results)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"cross_scene_{args.model}_{args.dataset}.csv"
    tex_path = out_dir / f"cross_scene_{args.model}_{args.dataset}.tex"
    save_cross_scene_table(aggregated, csv_path, tex_path)
    print(f"[sitpath-eval] Saved cross-scene generalization results to {out_dir}")


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "metrics":
        metrics_command(args)
    elif args.command == "data_efficiency":
        data_efficiency_command(args)
    elif args.command == "cross_scene":
        cross_scene_command(args)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
