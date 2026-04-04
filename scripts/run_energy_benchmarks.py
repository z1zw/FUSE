from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from fuse.energy_benchmarks import EnergyBenchmarkSuite

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('results_energy'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda:0'], default='auto')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    suite = EnergyBenchmarkSuite(device=args.device, warmup=args.warmup, repeat=args.repeat, seed=args.seed)
    suite.run(args.output_dir)
    print(args.output_dir)
if __name__ == '__main__':
    main()
