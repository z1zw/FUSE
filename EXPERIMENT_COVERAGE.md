# Experiment Coverage

The repository supports three main evaluation layers.

## 1. Core Benchmarking

The baseline benchmark flow covers latency, energy, mixed-workload behavior, memory effects, reuse behavior, scheduler behavior, and reconfiguration cost. The main entry points are:

- `scripts/run_benchmarks.py`
- `scripts/plot_paper_figures.py`

## 2. Security And Stability Evaluation

The extended evaluation flow covers security-oriented metrics, prompt stability, trade-off studies, and secure execution modes. The main entry points are:

- `scripts/run_security_eval.py`
- `scripts/run_sosp_suite.py`
- `scripts/run_sosp_plus_security.py`

## 3. End-To-End Studies

The repository also includes case studies, load sweeps, ablations, secure component breakdowns, and comparison runs. The main entry points are:

- `scripts/run_case_studies.py`
- `scripts/run_load_sweep.py`
- `scripts/run_extended_compare.py`
- `scripts/run_secure_components.py`

## Outputs

Most experiment flows produce CSV summaries first and render figures afterward. The exact output directory depends on the script arguments, but the standard pattern is:

- metrics and tables in an output directory such as `results/`
- optional raw tables in a nested `raw/` directory
- rendered figures in a figure directory such as `figures/`
