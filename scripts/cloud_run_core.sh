#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install pillow torchvision transformers matplotlib

export PYTHONPATH=src

python scripts/build_cpp.py
python scripts/train_calibration.py --output configs/calibration.json --engine cpp
python scripts/run_benchmarks.py --output-dir results_core --engine cpp --calibration configs/calibration.json
python scripts/plot_paper_figures.py --input-dir results_core --output-dir figures_core
python scripts/run_sosp_suite.py --output-dir results_sosp --engine cpp --calibration configs/calibration.json
python scripts/plot_sosp_figures.py --input-dir results_sosp --output-dir figures_sosp
python scripts/run_energy_benchmarks.py --output-dir results_energy --device auto
