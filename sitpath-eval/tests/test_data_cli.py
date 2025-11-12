import os
os.environ.setdefault("SITPATH_MODE", "auto")

import subprocess
import sys


def test_data_cli_download_runs_successfully():
    cmd = [
        sys.executable,
        "-m",
        "sitpath_eval.cli.data_cli",
        "download",
        "--dataset",
        "eth_ucy",
        "--root",
        "/tmp/data",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
