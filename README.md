# FUSE

## Tasks

- Build the C++ backend
- Run the benchmark suite
- Run the SOSP evaluation flow
- Run the hardware benchmark flow
- Run tests

## Commands

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/build_cpp.py
python scripts/run_benchmarks.py --output-dir results --engine cpp
python scripts/plot_paper_figures.py --input-dir results --output-dir figures
python scripts/run_sosp_suite.py --output-dir results_sosp --engine cpp --calibration configs/calibration.json
python scripts/plot_sosp_figures.py --input-dir results_sosp --output-dir figures_sosp
python scripts/run_hardware_benchmarks.py --output-dir runs/local_gpu/hardware --device auto
pytest -q
```

## Dependencies

- Python 3.10 or newer
- Packages listed in `requirements.txt`
- `g++` for the C++ backend
- CUDA-capable PyTorch only if GPU execution is needed

## Project Structure

```text
configs/        calibration and tool configuration
cpp/            C++ timing backend
rtl/            RTL reference modules
scripts/        command-line entry points
src/morphaccel/ Python package
tests/          regression tests
```
