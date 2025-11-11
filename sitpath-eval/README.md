# sitpath-eval

Minimal evaluation suite for the CVPR paper **"SitPath"**, bundling the components needed to reproduce the paper’s reported metrics.

## Features
- Datasets: lightweight loaders for SitPath benchmark splits.
- Tokenizer: shared text preprocessing utilities for prompts and captions.
- Baselines: reference SitPath models plus configuration hooks for custom runs.
- Metrics: accuracy and calibration metrics used in the paper.

## Quick start (Colab-friendly)
```bash
git clone https://github.com/your-org/sitpath-eval.git
cd sitpath-eval
pip install -e .
pytest
```

## Directory layout
- `pyproject.toml` – project metadata and dependencies.
- `README.md` – usage notes and overview.
- `sitpath_eval/` – library source (datasets, tokenizer, baselines, metrics).
- `tests/` – smoke tests ensuring the package imports correctly.
