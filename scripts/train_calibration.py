from __future__ import annotations
import argparse
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from fuse.benchmarks import BenchmarkSuite, load_calibration_targets

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('configs/calibration.json'))
    parser.add_argument('--engine', choices=['cpp', 'py'], default='cpp')
    parser.add_argument('--targets', type=Path, default=None)
    args = parser.parse_args()
    suite = BenchmarkSuite(engine=args.engine, calibration_targets=load_calibration_targets(args.targets))
    artifacts = suite.run(None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifacts.calibration, indent=2), encoding='utf-8')
    print(args.output)
if __name__ == '__main__':
    main()
