# Reproduction Notes

This repository is organized around a small set of repeatable entry points rather than a single monolithic script. In practice, the workflow is:

1. install the Python dependencies
2. build the C++ backend
3. generate or load calibration data
4. run the benchmark suite or the SOSP-style suite
5. render figures from the generated CSV files

## Minimal Setup

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/build_cpp.py
```

## Calibration

```bash
python scripts/train_calibration.py --output configs/calibration.json --engine cpp
```

## Core Evaluation

```bash
python scripts/run_benchmarks.py --output-dir results --engine cpp --calibration configs/calibration.json
python scripts/plot_paper_figures.py --input-dir results --output-dir figures
```

## SOSP-Style Evaluation

```bash
python scripts/run_sosp_suite.py --output-dir results_sosp --engine cpp --calibration configs/calibration.json
python scripts/plot_sosp_figures.py --input-dir results_sosp --output-dir figures_sosp
```

## Extended Evaluation Paths

- `scripts/run_security_eval.py`
- `scripts/run_case_studies.py`
- `scripts/run_load_sweep.py`
- `scripts/run_extended_compare.py`
- `scripts/run_secure_components.py`

## Testing

```bash
pytest -q
```

## Practical Notes

- Some flows depend on optional local tools or datasets.
- The C++ and Python simulation paths are both maintained and covered by tests.
- Figure generation scripts expect the corresponding CSV outputs to exist first.
