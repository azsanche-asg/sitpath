from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_FPS = 2.5
OBS_LEN = 8
PRED_LEN = 12


class ETHUCYDataset(Dataset):
    """Minimal loader for ETH/UCY pedestrian trajectories."""

    def __init__(self, trajectories: Iterable[Dict]):
        records = list(trajectories)
        self._sequence_len = OBS_LEN + PRED_LEN
        self._sequences = self._build_sequences(records)

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> Dict:
        seq = self._sequences[idx]
        coords = torch.tensor([[row["x"], row["y"]] for row in seq], dtype=torch.float32)
        frames = [row["frame_id"] for row in seq]
        return {
            "scene_id": seq[0]["scene_id"],
            "agent_id": seq[0]["agent_id"],
            "frames": frames,
            "pos": coords,
        }

    @staticmethod
    def download(root_dir: str) -> None:
        path = Path(root_dir).expanduser()
        print(
            f"[sitpath-eval] Download ETH/UCY data manually into {path} "
            "(CSV or NPZ with columns: scene_id,agent_id,frame_id,x,y)."
        )

    @staticmethod
    def load_split(root_dir: str, split: str = "train") -> List[Dict]:
        path = Path(root_dir).expanduser() / f"{split}.csv"
        if path.suffix == ".npz":
            return ETHUCYDataset._load_npz(path)
        if path.exists():
            return ETHUCYDataset._load_csv(path)
        alt = path.with_suffix(".npz")
        if alt.exists():
            return ETHUCYDataset._load_npz(alt)
        raise FileNotFoundError(f"Missing split file: {path} or {alt}")

    @staticmethod
    def _load_csv(path: Path) -> List[Dict]:
        with path.open() as fh:
            reader = csv.DictReader(fh, fieldnames=["scene_id", "agent_id", "frame_id", "x", "y"])
            return [ETHUCYDataset._coerce_row(row) for row in reader]

    @staticmethod
    def _load_npz(path: Path) -> List[Dict]:
        data = np.load(path, allow_pickle=True)
        records = data.get("trajectories")
        if records is None:
            raise ValueError(f"NPZ file {path} missing 'trajectories'")
        return [ETHUCYDataset._coerce_row(json.loads(json.dumps(rec))) for rec in records]

    @staticmethod
    def _coerce_row(row: Dict) -> Dict:
        return {
            "scene_id": row["scene_id"],
            "agent_id": int(row["agent_id"]),
            "frame_id": int(row["frame_id"]),
            "x": float(row["x"]),
            "y": float(row["y"]),
        }

    def _build_sequences(self, records: List[Dict]) -> List[List[Dict]]:
        total_len = self._sequence_len
        grouped: Dict[str, List[Dict]] = {}
        for row in records:
            key = f"{row['scene_id']}::{row['agent_id']}"
            grouped.setdefault(key, []).append(row)
        sequences: List[List[Dict]] = []
        for group_rows in grouped.values():
            group_rows.sort(key=lambda r: r["frame_id"])
            if len(group_rows) < total_len:
                continue
            for start in range(0, len(group_rows) - total_len + 1, total_len):
                window = group_rows[start : start + total_len]
                if len(window) == total_len:
                    sequences.append(window)
        return sequences
