# FUSE

FUSE is a research codebase for studying template reuse, runtime scheduling, and secure execution across mixed accelerator workloads. The repository includes a Python front end, a C++ timing core, RTL sketches, and scripts for reproducing the main evaluation flows used in the project.

The public project name is `FUSE`. Some internal Python package paths still use `src/morphaccel` for compatibility with the existing code and scripts.

## Repository Layout

```text
cpp/          C++ timing core
rtl/          RTL reference modules
scripts/      experiment and plotting entry points
src/          Python package and evaluation logic
tests/        regression tests
configs/      calibration and tool configuration
```

## Setup

Use Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

If you run commands from the repository root, set `PYTHONPATH=src` when needed.

## Build The C++ Backend

```bash
python scripts/build_cpp.py
```

## Run The Main Benchmark Suite

```bash
python scripts/run_benchmarks.py --output-dir results --engine cpp
python scripts/plot_paper_figures.py --input-dir results --output-dir figures
```

## Run The SOSP-Style Evaluation

```bash
python scripts/run_sosp_suite.py --output-dir results_sosp --engine cpp --calibration configs/calibration.json
python scripts/plot_sosp_figures.py --input-dir results_sosp --output-dir figures_sosp
```

## Run Tests

```bash
pytest -q
```

## Notes

- The repository contains both Python and C++ simulation paths.
- The hardware benchmarking path can use CPU or CUDA depending on the local environment.
- Several scripts expect local datasets or tool installations for the full end-to-end workflow.
