# Reproduction Notes

## Environment

- Python 3.10 or newer
- Packages from `requirements.txt`
- `g++` for the C++ backend

## Commands

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/build_cpp.py
python scripts/train_calibration.py --output configs/calibration.json --engine cpp
python scripts/run_benchmarks.py --output-dir results --engine cpp --calibration configs/calibration.json
python scripts/plot_paper_figures.py --input-dir results --output-dir figures
python scripts/run_sosp_suite.py --output-dir results_sosp --engine cpp --calibration configs/calibration.json
python scripts/plot_sosp_figures.py --input-dir results_sosp --output-dir figures_sosp
pytest -q
```

## Directory Layout

```text
configs/        calibration files
cpp/            C++ backend
rtl/            RTL sources
scripts/        command-line scripts
src/morphaccel/ Python package
tests/          tests
```

## Outputs

```text
results/        benchmark CSV files
results_sosp/   SOSP evaluation CSV files
figures/        benchmark plots
figures_sosp/   SOSP plots
```
