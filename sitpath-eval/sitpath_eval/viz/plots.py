import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)


def plot_metric_bars(df: pd.DataFrame, metric: str, group_col: str, out_path: str):
    """Bar plot (e.g., ADE/FDE per model)."""
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x=group_col, y=metric, color="skyblue")
    plt.title(f"{metric.upper()} by {group_col}")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_data_efficiency(df: pd.DataFrame, metric: str, out_path: str):
    """Line plot: performance vs data fraction."""
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df, x="fraction", y=metric, hue="model", marker="o")
    plt.title(f"Data Efficiency – {metric.upper()}")
    plt.xscale("log")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_ablation_bars(df: pd.DataFrame, metric: str, out_path: str):
    """Grouped bars for ablation (e.g., M/R/tempo)."""
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x="M", y=metric, hue="tempo", palette="Set2")
    plt.title(f"Ablation on M – {metric.upper()}")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
