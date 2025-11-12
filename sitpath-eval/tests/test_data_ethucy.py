import torch

from sitpath_eval.data import ETHUCYDataset, make_synthetic_ethucy, split_dataset
from sitpath_eval.data.eth_ucy import OBS_LEN, PRED_LEN
from sitpath_eval.utils.device import get_device

TEST_DEVICE = get_device("test")  # dynamic device selection for safe testing


def test_split_dataset_preserves_counts():
    trajectories = make_synthetic_ethucy()
    train, val, test = split_dataset(trajectories, val_ratio=0.2, test_ratio=0.2, seed=0)

    assert len(train) + len(val) + len(test) == len(trajectories)
    assert len(val) == int(len(trajectories) * 0.2)
    assert len(test) == int(len(trajectories) * 0.2)


def test_ethucy_dataset_returns_sequence_tensors():
    trajectories = make_synthetic_ethucy()
    dataset = ETHUCYDataset(trajectories)

    sample = dataset[0]
    expected_len = OBS_LEN + PRED_LEN

    assert sample["pos"].shape == (expected_len, 2)
    assert sample["pos"].dtype == torch.float32
    assert sample["pos"].device == TEST_DEVICE
