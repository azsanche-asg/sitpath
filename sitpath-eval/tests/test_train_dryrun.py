import os
os.environ.setdefault("SITPATH_MODE", "auto")

import torch
from torch.utils.data import TensorDataset

from sitpath_eval.cli import train_cli
from sitpath_eval.utils.device import get_device

device = get_device("auto")  # dynamic device selection for safe testing


def test_train_cli_dry_run(monkeypatch, tmp_path, capsys):
    def tiny_dataset(*args, **kwargs):
        obs = torch.zeros(2, 8, 2, device=device)
        targets = torch.zeros(2, 12, 2, device=device)
        ds = TensorDataset(obs, targets)
        return ds, ds

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(train_cli, "MODEL_DIR", model_dir)
    monkeypatch.setattr(train_cli, "make_synthetic_dataset", tiny_dataset)

    train_cli.main(["train", "--epochs", "1", "--model", "coord_gru", "--batch-size", "2"])

    captured = capsys.readouterr()
    assert "Epoch 1" in captured.out
    assert (model_dir / "coord_gru.pt").exists()
