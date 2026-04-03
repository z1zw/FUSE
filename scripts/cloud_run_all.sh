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
python scripts/run_hardware_benchmarks.py --output-dir runs/local_gpu/hardware --device auto
python scripts/run_model_zoo_eval.py --output-root runs/model_zoo --cifar-root /root/autodl-pub/cifar-10 --coco-root /root/autodl-pub/COCO2017 --device auto --sample-count 10
python scripts/setup_open_source_eval.py --tool all
python scripts/run_dramsim3_flow.py --fetch --run --output-dir reports/dramsim3_run
python scripts/run_cacti_flow.py --fetch --run --output-dir reports/cacti_run
python scripts/run_openroad_flow.py --fetch --run --mode docker --output-dir openroad_results
