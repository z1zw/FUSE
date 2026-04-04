from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from fuse.benchmarks import load_calibration
from fuse.sosp_plus import SOSPPlusSuite

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('results_plus'))
    parser.add_argument('--engine', type=str, default='cpp')
    parser.add_argument('--calibration', type=Path, default=Path('configs/calibration.json'))
    args = parser.parse_args()
    suite = SOSPPlusSuite(engine=args.engine, calibration=load_calibration(args.calibration))
    security = {'fig13_security': pd.read_csv(args.output_dir / 'fig13_security.csv'), 'fig15_secure_tradeoff': pd.read_csv(args.output_dir / 'fig15_secure_tradeoff.csv')}
    cases = {'fig16_case_studies': pd.read_csv(args.output_dir / 'fig16_case_studies.csv')}
    loads = {'fig19_load_sweep': pd.read_csv(args.output_dir / 'fig19_load_sweep.csv')}
    frames = suite.run_extended_compare(args.output_dir, security, cases, loads)
    for name, df in frames.items():
        df.to_csv(args.output_dir / f'{name}.csv', index=False)
if __name__ == '__main__':
    main()
