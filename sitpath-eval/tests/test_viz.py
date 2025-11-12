import os
import pandas as pd
from pathlib import Path

from sitpath_eval.cli import viz_cli
from sitpath_eval.viz.plots import plot_metric_bars
from sitpath_eval.viz.latex_tables import save_latex_table


def test_plot_metric_bars(tmp_path):
    df = pd.DataFrame({"model": ["a", "b"], "ADE": [1.0, 2.0]})
    out_path = tmp_path / "ade_bars.pdf"
    plot_metric_bars(df, "ADE", "model", str(out_path))
    assert out_path.exists()


def test_save_latex_table(tmp_path):
    df = pd.DataFrame({"col": [1, 2]})
    out_path = tmp_path / "table.tex"
    save_latex_table(df, str(out_path), caption="Demo", label="tab:demo")
    text = out_path.read_text()
    assert "Demo" in text


def test_viz_cli_pack(tmp_path):
    tables_dir = tmp_path / "tables"
    figs_dir = tmp_path / "figs"
    tables_dir.mkdir()
    df = pd.DataFrame({"model": ["x", "y"], "ADE": [1.0, 0.5]})
    csv_path = tables_dir / "metrics.csv"
    df.to_csv(csv_path, index=False)
    viz_cli.main(["pack", "--tables_dir", str(tables_dir), "--figs_dir", str(figs_dir)])
    assert any(figs_dir.glob("*.png"))
