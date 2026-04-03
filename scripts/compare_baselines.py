from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, default=Path('results'))
    parser.add_argument('--output', type=Path, default=Path('results/comparison_summary.csv'))
    args = parser.parse_args()
    fig4 = pd.read_csv(args.input_dir / 'fig4_latency.csv')
    fig6 = pd.read_csv(args.input_dir / 'fig6_mixed.csv')
    rows = []
    for domain in ['Groth16', 'VGG16', 'AES-128']:
        part = fig4[fig4['domain'] == domain]
        rows.append({'case': domain, 'metric': 'mean_latency_ratio', 'value': part['norm'].mean()})
    rows.append({'case': 'Mixed', 'metric': 'mean_latency_reduction', 'value': 1.0 - (fig6['morph_makespan_ms'] / fig6['baseline_makespan_ms']).mean()})
    rows.append({'case': 'Mixed', 'metric': 'mean_utilization_gain', 'value': (fig6['morph_util'] / fig6['baseline_util']).mean()})
    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(args.output)
if __name__ == '__main__':
    main()
