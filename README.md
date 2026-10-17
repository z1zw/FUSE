# FUSE

## Project Structure

- `configs/`: calibration and tool configuration
- `cpp/`: C++ backend
- `rtl/`: RTL modules
- `scripts/`: command-line entry points
- `src/`: Python package
- `tests/`: regression tests

## Dependencies

- Python packages in `requirements.txt`
- `g++` for the C++ backend
- PyTorch for hardware-oriented evaluation

## Run Commands

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/build_cpp.py
python scripts/run_benchmarks.py --engine cpp
pytest -q
```

## Which Engine

Two simulators implement the same model: the C++ engine in `cpp/` and the
Python engine in `src/fuse/sim/`. **Every number in the paper and in
`results/raw` comes from the C++ engine.** The two do not agree numerically,
so a run that falls back to Python will not match the published figures.
`run_benchmarks.py` warns on stderr when that happens; set
`FUSE_REQUIRE_CPP=1` to make it an error instead. Every metrics file records
which engine produced it in its `engine` row.

To confirm the C++ engine still reproduces the published results, from the
exact input tables stored beside them:

```bash
python scripts/verify_cpp_repro.py            # all 103 stored scenarios
python scripts/verify_cpp_repro.py --pattern fig4
```

## Language-Model Evaluation

```bash
# real local models on GSM8K, then FUSE vs. the baselines
python scripts/run_llm_zoo_eval.py --output-root runs/llm_zoo --offline

# language models as the request compiler (hosted models need a gateway)
export FUSE_LLM_GATEWAY="https://<your-gateway>"
export FUSE_LLM_TOKEN="<your-token>"
python scripts/run_llm_frontend_eval.py --output-root runs/llm_frontend

# language model plus its obligatory AES and Groth16 work, all concurrent
python scripts/run_concurrent_eval.py --output-root runs/concurrent --offline

# regenerate the paper tables, macros and figures from the results
python scripts/make_llm_tables.py --zoo runs/llm_zoo --frontend runs/llm_frontend \
    --concurrent runs/concurrent --out-dir sosp
```

The zoo needs a CUDA GPU with roughly 16 GB to run the 7B models; pass
`--models` to select a subset, `--no-native` to skip the GPU measurement, and
`--reuse-native` to re-simulate against measurements already on disk.

## Environment Requirements

- Python 3.10 or newer
- Windows, Linux, or macOS
- CUDA-capable GPU only if GPU evaluation is needed
- For the language-model evaluation: a CUDA GPU with ~16 GB, plus
  `transformers`, `datasets` and `nvidia-ml-py` from `requirements.txt`

## Review and Reproduction Notes

To facilitate paper review and code reproduction, the repository carries the
evaluation outputs alongside the code. `results/` holds every stored scenario
with the input tables it was simulated from (`scripts/verify_cpp_repro.py`
replays all of them), `runs/` holds the language-model evaluations
(`llm_zoo`, `llm_frontend`, `concurrent`, `concurrent_levels`), and
`figures/`, `reports/`, `openroad_results/` and `openroad_work/` hold the
rendered figures and the memory, area and timing reports. `deliverable/` is
a packaged copy of an earlier snapshot of these for reviewers. The CPU and
GPU baselines were measured on an Intel Xeon Platinum 8470Q and an NVIDIA
RTX 5080; everything else is simulation and runs on any machine with a C++
compiler. The released artifacts rely on open-source infrastructure vendored
under `external/`: [DRAMsim3](https://github.com/umd-memsys/DRAMsim3) for
memory-system evaluation, [CACTI](https://github.com/HewlettPackard/cacti)
for cache and buffer modelling, and
[OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts)
for physical-design reporting.
