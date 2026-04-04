from __future__ import annotations
import argparse
from pathlib import Path
from fuse.scenarios import ScenarioSuite

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('data/scenarios'))
    parser.add_argument('--output-root', type=Path, default=Path('results_scenarios'))
    parser.add_argument('--mode', choices=['fuse', 'fuse', 'secure_ct', 'secure_elastic'], default='fuse')
    parser.add_argument('--engine', choices=['py', 'cpp', 'gpu'], default='py')
    args = parser.parse_args()
    suite = ScenarioSuite(mode=args.mode, engine=args.engine)
    artifacts = suite.run(args.data_root, args.output_root)
    print(f'{args.output_root} ({len(artifacts)} scenes)')
if __name__ == '__main__':
    main()
