import os

import pandas as pd


def save_latex_table(df: pd.DataFrame, out_path: str, caption: str = "", label: str = "tab:results"):
    """Save DataFrame as LaTeX tabular with consistent formatting."""
    latex = df.to_latex(index=False, float_format="%.3f", caption=caption, label=label, escape=False)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(latex)
    print(f"Saved LaTeX table → {out_path}")
