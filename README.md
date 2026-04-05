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
python scripts/run_benchmarks.py --engine py
pytest -q
```

## Environment Requirements

- Python 3.10 or newer
- Windows, Linux, or macOS
- CUDA-capable GPU only if GPU evaluation is needed

## Review and Reproduction Notes

To facilitate paper review and code reproduction, we additionally provide evaluation results collected on an NVIDIA RTX 4090 and a 16-vCPU Intel(R) Xeon(R) Gold 6430 platform. The extra evaluation artifacts are organized under `results/`, `figures/`, `reports/`, `openroad_results/`, `openroad_work/`, and `runs/`, with a reviewer-friendly packaged copy under `deliverable/`. The released artifacts also rely on open-source infrastructure, including [DRAMSim3](https://github.com/umd-memsys/DRAMsim3) for memory-system evaluation and [OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts) for physical-design-oriented reporting and implementation analysis.
