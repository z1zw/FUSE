from __future__ import annotations
import argparse
from pathlib import Path
import json
from morphaccel.benchmarks import BenchmarkSuite

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('configs/calibration.json'))
    parser.add_argument('--engine', choices=['cpp', 'py'], default='cpp')
    args = parser.parse_args()
    suite = BenchmarkSuite(engine=args.engine)
    artifacts = suite.run(None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifacts.calibration, indent=2), encoding='utf-8')
    print(args.output)
if __name__ == '__main__':
    main()
