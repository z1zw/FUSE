from __future__ import annotations
import argparse
from pathlib import Path
from fuse.benchmarks import load_calibration
from fuse.sosp import SOSPSuite

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    parser.add_argument('--engine', type=str, default='cpp')
    parser.add_argument('--calibration', type=Path, default=Path('configs/calibration.json'))
    args = parser.parse_args()
    suite = SOSPSuite(engine=args.engine, calibration=load_calibration(args.calibration))
    suite.run_ablations(args.output_dir)
if __name__ == '__main__':
    main()
