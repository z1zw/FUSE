# Experiment Coverage

## Commands

```bash
python scripts/run_benchmarks.py --output-dir results --engine cpp
python scripts/plot_paper_figures.py --input-dir results --output-dir figures
python scripts/run_all.py
python scripts/run_security_eval.py
python scripts/run_case_studies.py
python scripts/run_load_sweep.py
python scripts/run_extended_compare.py
python scripts/run_secure_components.py
```

## Tasks

- Benchmark generation
- Figure generation
- Security evaluation
- Case study runs
- Load sweep runs
- Comparison runs
- Secure component runs

## Directory Layout

```text
results/        benchmark outputs
results_plus/   extended outputs
figures/        benchmark figures
figures_plus/   extended figures
scripts/        execution entry points
```
