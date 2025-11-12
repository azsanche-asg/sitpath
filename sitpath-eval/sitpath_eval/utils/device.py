import os

import torch


def get_device(mode: str = "auto") -> torch.device:
    """
    Smart device selector.

    mode:
      - "test"  -> always use CPU (for unit tests)
      - "train" -> prefer GPU if available, else CPU
      - "auto"  -> read from env var SITPATH_MODE ("test" or "train"); default CPU if unknown

    Example:
        device = get_device("train")
    """
    env_mode = os.getenv("SITPATH_MODE", "").lower()
    if mode == "auto" and env_mode in {"test", "train"}:
        mode = env_mode

    if mode == "test":
        return torch.device("cpu")
    if mode == "train":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def print_device_info(device: torch.device) -> None:
    """Print a friendly summary of the selected device."""
    print(f"🧠 Using device: {device}")
    if device.type == "cuda":
        print("    ", torch.cuda.get_device_name(0))
        print(f"    Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.1f} MiB")
