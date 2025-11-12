from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from sitpath_eval.models.coord_gru import CoordGRU
from sitpath_eval.train.eval_metrics import aggregate_metrics, save_metrics_table
from sitpath_eval.train.metrics import compute_metrics


def get_scene_splits(dataset_name: str) -> List[Tuple[List[str], List[str]]]:
    """Return predetermined leave-one-scene-out splits."""

    if dataset_name.lower() == "eth_ucy":
        scenes = ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]
    elif dataset_name.lower() == "sdd_mini":
        scenes = ["scene1", "scene2", "scene3"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    splits = []
    for idx, scene in enumerate(scenes):
        train = [s for s in scenes if s != scene]
        test = [scene]
        splits.append((train, test))
    # Leave-one-scene-out splits
    return splits


def build_synthetic_scene_dataset(scene_ids: Sequence[str], obs_len: int = 8, pred_len: int = 12):
    rng = np.random.default_rng(abs(hash(tuple(scene_ids))) % (2**32))
    n = 16 * len(scene_ids)
    seq = obs_len + pred_len
    data = rng.normal(size=(n, seq, 2)).astype(np.float32)
    traj = data.cumsum(axis=1)
    obs = torch.from_numpy(traj[:, :obs_len])
    targets = torch.from_numpy(traj[:, obs_len:])
    return torch.utils.data.TensorDataset(obs, targets)


def train_and_eval_cross_scene(model_cls, dataset_name: str, splits, **train_kwargs):
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = train_kwargs.get("epochs", 3)
    batch_size = train_kwargs.get("batch_size", 16)

    for train_scenes, test_scenes in splits:
        train_ds = build_synthetic_scene_dataset(train_scenes)
        test_ds = build_synthetic_scene_dataset(test_scenes)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)

        model = model_cls().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=train_kwargs.get("lr", 1e-3))
        loss_fn = torch.nn.MSELoss()

        for _ in range(epochs):
            model.train()
            for obs, targets in train_loader:
                obs = obs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                preds = model(obs)
                loss = loss_fn(preds, targets)
                loss.backward()
                optimizer.step()

        preds_list = []
        targets_list = []
        model.eval()
        with torch.no_grad():
            for obs, targets in test_loader:
                obs = obs.to(device)
                targets = targets.to(device)
                preds = model(obs)
                preds_list.append(preds.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
        preds = np.concatenate(preds_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        metrics = compute_metrics(preds, targets)
        results.append({"test_scene": test_scenes, "metrics": metrics})
    return results


def aggregate_cross_scene(results):
    metrics_list = [entry["metrics"] for entry in results]
    return aggregate_metrics(metrics_list)


def save_cross_scene_table(agg, path_csv, path_tex):
    csv_path = Path(path_csv)
    tex_path = Path(path_tex)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Mean", "CI Lower", "CI Upper"])
        for metric, stats in agg.items():
            writer.writerow([metric, stats["mean"], stats["ci_lower"], stats["ci_upper"]])

    lines = ["\\begin{tabular}{lccc}", "Metric & Mean & CI Lower & CI Upper \\\\", "\\hline"]
    for metric, stats in agg.items():
        lines.append(f"{metric} & {stats['mean']:.3f} & {stats['ci_lower']:.3f} & {stats['ci_upper']:.3f} \\\\")
    lines.append("\\end{tabular}")
    tex_path.write_text("\n".join(lines))


if __name__ == "__main__":
    splits = get_scene_splits("eth_ucy")
    results = train_and_eval_cross_scene(CoordGRU, "eth_ucy", splits, epochs=1)
    agg = aggregate_cross_scene(results)
    out_dir = Path("artifacts/tables")
    save_cross_scene_table(agg, out_dir / "demo_cross_scene.csv", out_dir / "demo_cross_scene.tex")
