from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from sitpath_eval.train.eval_metrics import aggregate_metrics, save_metrics_table


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


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "metrics":
        metrics_command(args)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
