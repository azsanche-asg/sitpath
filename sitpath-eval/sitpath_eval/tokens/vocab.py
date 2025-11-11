from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


class Vocabulary:
    """Bidirectional mapping between token tuples and ids."""

    def __init__(self, sector_count: int = 16, radial_bins: int = 4):
        self.sector_count = sector_count
        self.radial_bins = radial_bins
        self._token_to_id: Dict[Tuple[int, int, int, int], int] = {}
        self._id_to_token: Dict[int, Tuple[int, int, int, int]] = {}

    def add(self, token_tuple: Tuple[int, int, int, int]) -> int:
        if token_tuple not in self._token_to_id:
            idx = len(self._token_to_id)
            self._token_to_id[token_tuple] = idx
            self._id_to_token[idx] = token_tuple
        return self._token_to_id[token_tuple]

    def encode_tuple(self, token_tuple: Tuple[int, int, int, int]) -> int:
        return self.add(token_tuple)

    def decode_id(self, token_id: int) -> Tuple[int, int, int, int]:
        if token_id not in self._id_to_token:
            raise KeyError(f"Unknown token id {token_id}")
        return self._id_to_token[token_id]

    def __len__(self) -> int:
        return len(self._id_to_token)

    def save(self, path: str) -> None:
        p = Path(path)
        payload = {str(k): list(v) for k, v in self._id_to_token.items()}
        data = {
            "sector_count": self.sector_count,
            "radial_bins": self.radial_bins,
            "tokens": payload,
        }
        p.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        p = Path(path)
        data = json.loads(p.read_text())
        vocab = cls(sector_count=data["sector_count"], radial_bins=data["radial_bins"])
        for token_id, token_tuple in data["tokens"].items():
            idx = int(token_id)
            tup = tuple(token_tuple)
            vocab._token_to_id[tup] = idx
            vocab._id_to_token[idx] = tup
        return vocab
