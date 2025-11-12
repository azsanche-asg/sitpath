import os
#os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # disable CUDA for tests unless re-enabled

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
        device = torch.device("cpu")
    elif mode == "train":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    if "pytest" in sys.modules and torch.cuda.is_available():
        # force CPU even if CUDA_VISIBLE_DEVICES was ignored
        return torch.device("cpu")

    return device


def print_device_info(device: torch.device) -> None:
    """Print a friendly summary of the selected device."""
    print(f"🧠 Using device: {device}")
    if device.type == "cuda":
        print("   ", torch.cuda.get_device_name(0))
        print(f"   Memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.1f} MiB")
