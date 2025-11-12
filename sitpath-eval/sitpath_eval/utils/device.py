import os
import sys

import torch


def get_device(mode: str = "auto") -> torch.device:
    """
    Smart device selector with auto test detection.
    - "test"  -> always CPU
    - "train" -> prefer GPU if available
    - "auto"  -> detect mode from env or context
    """
    running_pytest = "pytest" in sys.modules or any("pytest" in arg for arg in sys.argv)
    env_mode = os.getenv("SITPATH_MODE", "").lower()

    if mode == "auto":
        if env_mode in {"test", "train"}:
            mode = env_mode
        elif running_pytest:
            mode = "test"
        else:
            mode = "train"

    if mode == "test":
        return torch.device("cpu")
    if mode == "train":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def print_device_info(device: torch.device) -> None:
    """Print a friendly summary of the selected device."""
    print(f"🧠 Using device: {device}")
    if device.type == "cuda":
        print("   ", torch.cuda.get_device_name(0))
        print(f"   Memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.1f} MiB")
