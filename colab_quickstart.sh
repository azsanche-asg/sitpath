#!/bin/bash
echo "🚀 SitPath Evaluation Quickstart – Colab Mode"
if [ ! -d "sitpath-eval" ]; then
  git clone https://github.com/<your-org>/sitpath-eval.git
fi
cd sitpath-eval
pip install -e .
pytest -q --maxfail=1 --disable-warnings || exit 1
jupyter notebook notebooks/00_setup.ipynb
echo "✅ Ready – open notebooks 01–06 or run_all.ipynb to reproduce results."
