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
