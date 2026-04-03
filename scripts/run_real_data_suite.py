from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

def _run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=str(cwd), check=True)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output-root', type=Path, default=Path('runs/local_gpu'))
    parser.add_argument('--engine', choices=['gpu', 'cpp', 'py'], default='gpu')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda:0'], default='auto')
    parser.add_argument('--skip-prepare', action='store_true')
    parser.add_argument('--llm-limit', type=int, default=512)
    parser.add_argument('--image-limit', type=int, default=256)
    parser.add_argument('--medical-limit', type=int, default=512)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    python = sys.executable
    _run([python, 'scripts/init_fuse_data_layout.py', '--data-root', str(args.data_root)], root)
    if not args.skip_prepare:
        _run([python, 'scripts/prepare_real_datasets.py', '--data-root', str(args.data_root), '--medical-limit', str(args.medical_limit), '--image-limit', str(args.image_limit), '--llm-limit', str(args.llm_limit)], root)
    _run([python, 'scripts/run_scenarios.py', '--data-root', str(args.data_root / 'prepared'), '--output-root', str(args.output_root / 'scenarios'), '--mode', 'fuse', '--engine', args.engine], root)
    _run([python, 'scripts/run_hardware_benchmarks.py', '--output-dir', str(args.output_root / 'hardware'), '--device', args.device], root)
    print(args.output_root)
if __name__ == '__main__':
    main()
