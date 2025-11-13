import numpy as np
import pandas as pd
from pathlib import Path

RAW_ROOT = Path(__file__).resolve().parents[1] / "datasets"
OUT_ROOT = Path(__file__).resolve().parents[1] / "sitpath-data" / "eth_ucy"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SCENES = ["ETH", "HOTEL", "ZARA1", "ZARA2"]


def load_obsmat(scene: str) -> pd.DataFrame:
    path = RAW_ROOT / scene / "obsmat.txt"
    if not path.exists():
        print(f"[prepare_eth_ucy] Skipping {scene}: not found")
        return pd.DataFrame()
    print(f"[prepare_eth_ucy] Loading {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = [
        "frame_id",
        "agent_id",
        "pos_x",
        "pos_z",
        "pos_y",
        "vel_x",
        "vel_z",
        "vel_y",
    ]
    df = df[["frame_id", "agent_id", "pos_x", "pos_y"]].copy()
    df.columns = ["frame_id", "agent_id", "x", "y"]
    df["frame_id"] = df["frame_id"].round().astype(int)
    df["agent_id"] = df["agent_id"].astype(int)
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    df["scene_id"] = scene.lower()
    return df


def main():
    all_dfs = []
    for scene in SCENES:
        df = load_obsmat(scene)
        if len(df):
            all_dfs.append(df)
    if not all_dfs:
        print("[prepare_eth_ucy] No scenes loaded, exiting.")
        return
    full = pd.concat(all_dfs, ignore_index=True)
    full = full.sort_values(["scene_id", "agent_id", "frame_id"])
    print(f"[prepare_eth_ucy] Loaded {len(full)} total rows.")

    agents = full["agent_id"].unique()
    rs = np.random.RandomState(42)
    rs.shuffle(agents)
    n = len(agents)
    n_train, n_val = int(0.7 * n), int(0.1 * n)
    splits = {
        "train": agents[:n_train],
        "val": agents[n_train : n_train + n_val],
        "test": agents[n_train + n_val :],
    }

    for split, agent_ids in splits.items():
        df_split = full[full["agent_id"].isin(agent_ids)]
        out_path = OUT_ROOT / f"{split}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_split[["scene_id", "agent_id", "frame_id", "x", "y"]].to_csv(
            out_path, index=False, header=False
        )
        print(f"[prepare_eth_ucy] Wrote {len(df_split)} rows → {out_path}")

    print("[prepare_eth_ucy] Done.")


if __name__ == "__main__":
    main()
