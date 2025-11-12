from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sitpath_eval.models.sitpath_gru import SitPathGRU
from sitpath_eval.tokens.vocab import Vocabulary
from sitpath_eval.train.eval_metrics import aggregate_metrics
from sitpath_eval.train.metrics import compute_metrics, minade_k


def ablation_grid():
    configs = []
    for M, R, collapse, tempo in product([8, 16, 32], [3, 5, 8], [True, False], ["on", "off"]):
        configs.append({"M": M, "R": R, "collapse": collapse, "tempo": tempo})
    return configs


DEFAULT_VOCAB_SIZE = 32


def build_vocab_from_config(cfg, target_size: int = DEFAULT_VOCAB_SIZE) -> Vocabulary:
    vocab = Vocabulary(sector_count=cfg["M"], radial_bins=cfg["R"])
    for sector in range(cfg["M"]):
        for radial in range(cfg["R"]):
            vocab.add((sector, radial, 0, 0))
            if len(vocab) >= target_size:
                return vocab
    return vocab


def build_synthetic_token_dataset(samples: int = 64, obs_len: int = 8, pred_len: int = 12, vocab_size: int = DEFAULT_VOCAB_SIZE):
    rng = np.random.default_rng(0)
    obs = rng.integers(0, vocab_size, size=(samples, obs_len))
    targets = rng.integers(0, vocab_size, size=(samples, pred_len))
    return TensorDataset(torch.from_numpy(obs), torch.from_numpy(targets))


def tokens_to_coords(tokens: np.ndarray) -> np.ndarray:
    arr = tokens.astype(np.float32)
    return np.stack([arr, arr], axis=-1)


def train_and_eval_ablation(model_cls=SitPathGRU, dataset=None, grid=None, **train_kwargs):
    if grid is None:
        grid = ablation_grid()
    if dataset is None:
        dataset = build_synthetic_token_dataset()
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=train_kwargs.get("batch_size", 16), shuffle=True)

    for cfg in grid:
        vocab = build_vocab_from_config(cfg)
        model = model_cls(vocab_size=len(vocab)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=train_kwargs.get("lr", 1e-3))
        loss_fn = torch.nn.CrossEntropyLoss()

        for _ in range(train_kwargs.get("epochs", 3)):
            model.train()
            for obs, targets in loader:
                obs = obs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                logits = model(obs)
                loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
                loss.backward()
                optimizer.step()

        model.eval()
        preds_list, targets_list = [], []
        with torch.no_grad():
            for obs, targets in loader:
                logits = model(obs.to(device))
                preds = logits.argmax(dim=-1).cpu().numpy()
                preds_list.append(preds)
                targets_list.append(targets.numpy())
        preds_tokens = np.concatenate(preds_list, axis=0)
        targets_tokens = np.concatenate(targets_list, axis=0)
        preds_coords = tokens_to_coords(preds_tokens)
        targets_coords = tokens_to_coords(targets_tokens)

        metrics = compute_metrics(preds_coords, targets_coords)
        samples = np.stack(
            [
                preds_coords + np.random.normal(scale=0.05, size=preds_coords.shape)
                for _ in range(3)
            ],
            axis=0,
        )
        metrics["minade_k"] = minade_k(samples, targets_coords)
        results.append({**cfg, **metrics})
    return results


def aggregate_ablation(results: List[Dict[str, float]]):
    grouped = {}
    for entry in results:
        key = (entry["M"], entry["R"], entry["collapse"], entry["tempo"])
        grouped.setdefault(key, []).append({k: v for k, v in entry.items() if k not in {"M", "R", "collapse", "tempo"}})
    agg = {}
    for key, metrics in grouped.items():
        agg[key] = aggregate_metrics(metrics)
    return agg


def save_ablation_table(agg, path_csv, path_tex):
    path_csv = Path(path_csv)
    path_tex = Path(path_tex)
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    path_tex.parent.mkdir(parents=True, exist_ok=True)

    import csv

    metrics = ["ade", "fde", "minade_k", "miss_rate"]

    with path_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["M", "R", "collapse", "tempo"] + [m.upper() for m in metrics])
        for (M, R, collapse, tempo), stats in agg.items():
            row = [M, R, collapse, tempo]
            for m in metrics:
                row.append(stats[m]["mean"])
            writer.writerow(row)

    header = "M & R & collapse & tempo & ADE & FDE & minADE_k & MR \\\\"
    lines = ["\\begin{tabular}{l l l l l l l l}", header, "\\hline"]
    for (M, R, collapse, tempo), stats in agg.items():
        lines.append(
            f"{M} & {R} & {collapse} & {tempo} & "
            f"{stats['ade']['mean']:.3f} & {stats['fde']['mean']:.3f} & "
            f"{stats['minade_k']['mean']:.3f} & {stats['miss_rate']['mean']:.3f} \\\\"
        )
    lines.append("\\end{tabular}")
    path_tex.write_text("\n".join(lines))


if __name__ == "__main__":
    dataset = build_synthetic_token_dataset()
    results = train_and_eval_ablation(dataset=dataset, grid=ablation_grid())
    agg = aggregate_ablation(results)
    out_dir = Path("artifacts/tables")
    save_ablation_table(agg, out_dir / "demo_ablation.csv", out_dir / "demo_ablation.tex")
