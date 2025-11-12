# SitPath Evaluation Suite — Colab Replication Guide

## 🔧 Setup (Colab)
```bash
!git clone https://github.com/<your-org>/sitpath-eval.git
%cd sitpath-eval
!pip install -e .
!pytest -q --disable-warnings
# Expected: ✔ 59 passed, 1 skipped (approx.)
```

If using the pre-built Colab environment:
```bash
%cd /content
!bash colab_quickstart.sh
```

## ⚙️ Notebook Pipeline (00–06)

| Notebook | Purpose | Outputs → `artifacts/` |
| --- | --- | --- |
| 00_setup | install deps & print GPU/CPU info | stdout summary |
| 01_precompute_tokens | build SitPath vocab + token cache | `tokens/*.npz`, `vocab.json` |
| 02_train_baselines | train Coord/SitPath/Raster/Social baselines | `logs/*.json`, `models/` |
| 03_eval_metrics | compute ADE/FDE/MR/minADEₖ | `tables/core_metrics.{csv,tex}` |
| 04_data_efficiency | evaluate @ 10/25/100 % data | `tables/data_efficiency.*` |
| 05_cross_scene_uncertainty | LOO/LOSO + NLL/ECE/Diversity | `tables/{cross_scene,uncertainty}.*` |
| 06_controllability_ablation | editability + tokenizer ablations | `tables/controllability.*`, `tables/ablation.*`, `figs/*.pdf` |

## 🚀 End-to-End Run
```bash
!python notebooks/run_all.ipynb
# or open run_all.ipynb in Colab UI and “Run All Cells”
```
Produces: `results_pack.zip` with all CSV, LaTeX tables, and figures.

## 📊 Artifacts Structure
```
artifacts/
 ├─ logs/
 ├─ models/
 ├─ tables/     → *.csv, *.tex
 ├─ figs/       → *.pdf
 └─ tokens/     → *.npz + vocab.json
```

## 🧪 Repro Checklist
- ✅ GPU type reported by 00_setup
- ✅ All tests pass
- ✅ No missing artifacts after run_all
- ✅ Figures render without warnings

## 📞 Support
Questions? Open an issue or email <contact@your-org.com>.
