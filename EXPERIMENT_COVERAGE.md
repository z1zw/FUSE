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
python scripts/run_llm_zoo_eval.py --output-root runs/llm_zoo --sample-count 8 --gen-tokens 64
python scripts/run_llm_frontend_eval.py --output-root runs/llm_frontend
python scripts/run_concurrent_eval.py --output-root runs/concurrent --offline
python scripts/make_llm_tables.py --zoo runs/llm_zoo --frontend runs/llm_frontend \
    --concurrent runs/concurrent --out-dir sosp
```

## Tasks

- Benchmark generation
- Figure generation
- Security evaluation
- Case study runs
- Load sweep runs
- Comparison runs
- Secure component runs
- Language-model runs (real local models, then FUSE vs baselines)
- Language-model request-compiler runs (local and hosted models)
- Concurrent three-domain secure serving (one fabric vs. dedicated engines)

## Directory Layout

```text
results/        benchmark outputs
results_plus/   extended outputs
figures/        benchmark figures
figures_plus/   extended figures
runs/llm_zoo/   language-model measurements, simulations and figures
runs/llm_frontend/  request-compiler study
runs/concurrent/    concurrent three-domain study
scripts/        execution entry points
```

## Language-Model Experiments

`run_llm_zoo_eval.py` runs each model in the zoo for real on the local GPU over
GSM8K prompts, then rebuilds the same requests as FUSE task graphs and replays
them against a Transformer ASIC, a tensor-core baseline and a multi-core system.
It also replays each request as a private-serving trace (AES, the model, and
Groth16) and sweeps how much of the offered load is security work.

`run_llm_frontend_eval.py` asks real models to compile a service request into a
plan, then executes that plan with and without FUSE's canonicalizer. Hosted
models are reached through an OpenAI-compatible gateway; credentials come from
the environment and are never stored in the repository:

```bash
export FUSE_LLM_GATEWAY="https://<your-gateway>"
export FUSE_LLM_TOKEN="<your-token>"
python scripts/run_llm_frontend_eval.py --output-root runs/llm_frontend
```

Add `--offline` to either script to resolve models and datasets from the local
Hugging Face cache only, and `--no-gateway` to run local models alone.

## Concurrent Secure Serving

`run_concurrent_eval.py` runs the language model together with the two security
tasks a private service is obliged to perform, all three in flight at once. The
request shapes come from three public datasets (GSM8K, MMLU, HumanEval) and the
volume of the security work is derived from the same bytes: AES covers the
prompt, the reply and retrieved SciQ passages, and the Groth16 circuit is a
Merkle inclusion proof whose depth follows from the corpus size.

The baseline is three engines that cannot help each other, each on its own slice
of the same RFU budget, with every split searched and the best kept. Two
variants are reported: `partitioned` gives each engine the same datapath as
FUSE, isolating the effect of carving the array up, while `dedicated` also gives
each engine its domain-specific datapath advantage.

`--array-sizes` sweeps the total RFU count, which is the axis that matters: the
advantage of sharing is an area-efficiency argument and it decays once the array
is wider than a single domain can use.
