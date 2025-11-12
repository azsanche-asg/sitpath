from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple


class Vocabulary:
    """Bidirectional mapping between token tuples and ids."""

    def __init__(self, sector_count: int = 16, radial_bins: int = 4):
        self.sector_count = sector_count
        self.radial_bins = radial_bins
        self.tuple_to_id: Dict[Tuple[int, int, int, int], int] = {}
        self.id_to_tuple: Dict[int, Tuple[int, int, int, int]] = {}
        self._next_id = 0

    def add(self, token_tuple: Tuple[int, int, int, int]) -> int:
        if token_tuple not in self.tuple_to_id:
            idx = self._next_id
            self.tuple_to_id[token_tuple] = idx
            self.id_to_tuple[idx] = token_tuple
            self._next_id += 1
        return self.tuple_to_id[token_tuple]

    def encode_tuple(self, token_tuple: Tuple[int, int, int, int]) -> int:
        return self.add(token_tuple)

    def decode_id(self, token_id: int) -> Tuple[int, int, int, int]:
        if token_id not in self.id_to_tuple:
            raise KeyError(f"Unknown token id {token_id}")
        return self.id_to_tuple[token_id]

    def save(self, path: str) -> None:
        p = Path(path)
        payload = {str(k): list(v) for k, v in self.id_to_tuple.items()}
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
            vocab.tuple_to_id[tup] = idx
            vocab.id_to_tuple[idx] = tup
        if vocab.id_to_tuple:
            vocab._next_id = max(vocab.id_to_tuple) + 1
        else:
            vocab._next_id = 0
        return vocab

    def to_dict(self) -> Dict[int, Tuple[int, int, int, int]]:
        return dict(self.id_to_tuple)

    def __len__(self) -> int:
        return len(self.id_to_tuple)

    def __repr__(self) -> str:
        return f"Vocabulary(size={len(self)}, next_id={self._next_id})"
