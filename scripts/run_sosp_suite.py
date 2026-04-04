from __future__ import annotations
import argparse
from pathlib import Path
from fuse.sosp import run_sosp_suite

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    parser.add_argument('--engine', type=str, default='cpp')
    parser.add_argument('--calibration', type=Path, default=Path('configs/calibration.json'))
    args = parser.parse_args()
    run_sosp_suite(args.output_dir, engine=args.engine, calibration_path=args.calibration)
if __name__ == '__main__':
    main()
