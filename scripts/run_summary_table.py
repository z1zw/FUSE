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
    base = suite._load_or_run_base(args.output_dir, load_raw=False)
    security = {'fig13_security': pd.read_csv(args.output_dir / 'fig13_security.csv'), 'fig15_secure_tradeoff': pd.read_csv(args.output_dir / 'fig15_secure_tradeoff.csv')}
    cases = {'fig16_case_studies': pd.read_csv(args.output_dir / 'fig16_case_studies.csv')}
    loads = {'fig19_load_sweep': pd.read_csv(args.output_dir / 'fig19_load_sweep.csv')}
    ablations = {'fig17_ablation_plus': pd.read_csv(args.output_dir / 'fig17_ablation_plus.csv')}
    components = {'fig21_secure_components': pd.read_csv(args.output_dir / 'fig21_secure_components.csv')}
    compare = {'fig20_extended_compare': pd.read_csv(args.output_dir / 'fig20_extended_compare.csv')}
    table = suite.build_summary_table(base, security, cases, loads, ablations, components, compare)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / 'table1_sosp_summary.csv', index=False)
if __name__ == '__main__':
    main()
