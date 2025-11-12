from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from sitpath_eval.viz import plots
from sitpath_eval.viz.latex_tables import save_latex_table


def detect_table_type(df: pd.DataFrame) -> str:
    if {"M", "R", "tempo"}.issubset(df.columns):
        return "ablation"
    if "fraction" in df.columns:
        return "data_efficiency"
    return "generic"


def plot_from_table(df: pd.DataFrame, table_path: Path, figs_dir: Path) -> int:
    table_type = detect_table_type(df)
    figures = 0
    if table_type == "ablation":
        for metric in ["ADE", "FDE", "minADE_k", "MR"]:
            if metric in df.columns:
                out_path = figs_dir / f"{table_path.stem}_{metric.lower()}_ablation.png"
                plots.plot_ablation_bars(df, metric, str(out_path))
                figures += 1
    elif table_type == "data_efficiency":
        for metric in df.columns:
            if metric.lower().startswith(("ade", "fde", "mr")):
                out_path = figs_dir / f"{table_path.stem}_{metric.lower()}_eff.png"
                plots.plot_data_efficiency(df.assign(model=table_path.stem), metric, str(out_path))
                figures += 1
    else:
        metric_candidates = [c for c in df.columns if c.upper() in {"ADE", "FDE", "MR"}]
        for metric in metric_candidates:
            out_path = figs_dir / f"{table_path.stem}_{metric.lower()}_bars.png"
            plots.plot_metric_bars(df, metric, df.columns[0], str(out_path))
            figures += 1
    return figures


def save_latex(df: pd.DataFrame, table_path: Path, latex_dir: Path) -> None:
    out_path = latex_dir / f"{table_path.stem}.tex"
    save_latex_table(df, str(out_path), caption=f"{table_path.stem} results")


def pack_command(args: argparse.Namespace) -> None:
    tables_dir = Path(args.tables_dir)
    figs_dir = Path(args.figs_dir)
    latex_dir = tables_dir / "latex"
    csv_files = sorted(glob.glob(str(tables_dir / "*.csv")))
    total_figs = 0
    total_tables = 0
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        total_figs += plot_from_table(df, Path(csv_path), figs_dir)
        save_latex(df, Path(csv_path), latex_dir)
        total_tables += 1
    print(f"[viz_cli] Saved {total_figs} figures and {total_tables} tables to {figs_dir} / {latex_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualization utilities for SitPath.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack = subparsers.add_parser("pack", help="Generate plots and LaTeX tables from CSV artifacts.")
    pack.add_argument("--tables_dir", default="artifacts/tables")
    pack.add_argument("--figs_dir", default="artifacts/figs")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "pack":
        pack_command(args)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
