from __future__ import annotations

from typing import Dict, List


def make_synthetic_ethucy(n_agents: int = 4, n_frames: int = 20) -> List[Dict]:
    """Generate a synthetic ETH/UCY-like dataset for tests and docs."""

    records = []
    for scene_id in range(1, 3):
        for agent_id in range(n_agents):
            for frame_id in range(n_frames):
                records.append(
                    {
                        "scene_id": f"scene_{scene_id}",
                        "agent_id": agent_id,
                        "frame_id": frame_id,
                        "x": agent_id + frame_id * 0.1,
                        "y": scene_id + frame_id * 0.1,
                    }
                )
    return records
