import os
from pathlib import Path

import nbformat
import pytest

NOTEBOOKS_DIR = Path("notebooks")
EXPECTED = [
    "00_setup.ipynb",
    "01_precompute_tokens.ipynb",
    "02_train_baselines.ipynb",
    "03_eval_metrics.ipynb",
    "04_data_efficiency.ipynb",
    "05_cross_scene_uncertainty.ipynb",
    "06_controllability_ablation.ipynb",
    "run_all.ipynb",
]


@pytest.mark.skipif("COLAB_GPU" in os.environ, reason="Skip notebook tests on Colab runtime")
def test_notebooks_exist_and_import_device():
    for nb_name in EXPECTED:
        path = NOTEBOOKS_DIR / nb_name
        assert path.exists(), f"Missing notebook {nb_name}"
        nb = nbformat.read(path, as_version=4)
        first_code = next((cell for cell in nb.cells if cell.cell_type == "code"), None)
        assert first_code is not None
        assert "get_device" in "".join(first_code.source), f"{nb_name} missing get_device import"


def test_run_all_references_every_notebook():
    path = NOTEBOOKS_DIR / "run_all.ipynb"
    nb = nbformat.read(path, as_version=4)
    source = "".join("".join(cell.source) for cell in nb.cells if cell.cell_type == "code")
    for nb_name in EXPECTED[:-1]:
        assert nb_name in source, f"run_all.ipynb missing reference to {nb_name}"
