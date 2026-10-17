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
python scripts/run_all.py
pytest -q
python scripts/run_llm_zoo_eval.py --output-root runs/llm_zoo --offline
python scripts/run_llm_frontend_eval.py --output-root runs/llm_frontend
python scripts/run_concurrent_eval.py --output-root runs/concurrent --offline
python scripts/run_reuse_breakdown.py --offline
python scripts/run_granularity_ablation.py --offline
python scripts/run_sensitivity.py && python scripts/make_sensitivity_table.py
python scripts/make_llm_tables.py --zoo runs/llm_zoo --frontend runs/llm_frontend --out-dir sosp
python scripts/plot_bold_figures.py
```

## Re-simulating Without Models or a GPU

The language-model studies separate measurement from simulation, so after a
change to the engine the simulation half can be redone from what is on disk:

```bash
python scripts/run_llm_zoo_eval.py --output-root runs/llm_zoo --reuse-native --no-native --offline
python scripts/resimulate_llm_frontend.py          # replays runs/llm_frontend/frontend_replies.csv
python scripts/run_concurrent_eval.py --offline --answer-lengths runs/concurrent/answer_lengths.json
```

The zoo loads its GPU measurements from `llm_native.csv`, the request-compiler
study replays the model replies it already collected, and the concurrent study
reads the answer lengths it measured. All three then run only the simulator.

## The Engine the Paper Reports

The published results in `results/raw` were produced by the C++ engine, and
`scripts/verify_cpp_repro.py` reproduces all 103 stored scenarios from it
exactly, by re-running the binary on the input tables stored beside each
result. The Python engine in `src/fuse/sim/` implements the same model but
does not agree with it numerically, so a reproduction run must use
`--engine cpp` and should set `FUSE_REQUIRE_CPP=1`, which turns the silent
fallback to Python into an error. Each metrics file records its producer in
an `engine` row.

`cpp/build/` is not committed. `scripts/build_cpp.py` builds the engine with
`g++` or `cl`; a cross-compiled Windows binary from `x86_64-w64-mingw32-g++`
with `-static` also works and needs no runtime DLLs.

## Directory Layout

```text
configs/        calibration files
cpp/            C++ backend
rtl/            RTL sources
scripts/        command-line scripts
src/            FUSE Python package
tests/          tests
```

## Outputs

```text
results/        benchmark CSV files
results_plus/   extended evaluation CSV files
figures/        benchmark plots
figures_plus/   extended evaluation plots
runs/llm_zoo/       language-model results, per model and combined
runs/llm_frontend/  request-compiler results
```

## Language-Model Evaluation

The two language-model scripts are the only ones that read models from the
Hugging Face cache. `--offline` keeps them off the network entirely; without it
they will download what is missing. The zoo needs roughly 16 GB of GPU memory
for the 7B models, and `--models` selects a subset if that is not available.

Measuring and simulating are separable. `--reuse-native` loads the GPU
measurements a previous run left in the output root and re-runs only the
simulation, so a change to the hardware mapping can be evaluated against exactly
the measurements already taken.

Hosted models in the request-compiler study are reached through an
OpenAI-compatible gateway named by `FUSE_LLM_GATEWAY` and `FUSE_LLM_TOKEN`;
neither is stored in the repository. `--no-gateway` runs the local models alone.

`make_llm_tables.py` writes the LaTeX tables, the figures and `llm_numbers.tex`
into `sosp/`. Every number the evaluation prose quotes is a macro from that
file, so the text cannot disagree with the run that produced it.
