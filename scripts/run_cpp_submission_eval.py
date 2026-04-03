from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
from morphaccel.benchmarks import run_default_suite

def _now() -> str:
    return datetime.now().isoformat(timespec='seconds')

def _record(log_rows, step: str, status: str, started_at: float, detail: dict) -> None:
    log_rows.append({'step': step, 'status': status, 'started_at': started_at, 'finished_at': time.perf_counter(), 'timestamp': _now(), 'detail_json': json.dumps(detail, ensure_ascii=False)})

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=Path('results_cpp_submission'))
    parser.add_argument('--figures-dir', type=Path, default=Path('figures_cpp_submission'))
    parser.add_argument('--engine', choices=['cpp'], default='cpp')
    parser.add_argument('--calibration', type=Path, default=None)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    log_rows = []
    started = time.perf_counter()
    artifacts = run_default_suite(args.results_dir, engine=args.engine, calibration_path=args.calibration)
    _record(log_rows, 'benchmark_suite', 'completed', started, {'engine': args.engine, 'results_dir': str(args.results_dir), 'dataframes': sorted(list(artifacts.dataframes.keys())), 'rawframes': sorted(list(artifacts.rawframes.keys())), 'traces': sorted(list(artifacts.traces.keys())), 'scalars': artifacts.scalars})
    started = time.perf_counter()
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / 'plot_paper_figures.py'), '--input-dir', str(args.results_dir), '--output-dir', str(args.figures_dir)], check=True)
    audit_path = args.figures_dir / 'figure_audit.csv'
    audit = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
    _record(log_rows, 'paper_figures', 'completed', started, {'figures_dir': str(args.figures_dir), 'audit_rows': int(len(audit)), 'figures': sorted([path.name for path in args.figures_dir.glob('*.png')])})
    process_log = pd.DataFrame(log_rows)
    process_log.to_csv(args.results_dir / 'process_log.csv', index=False)
    manifest = {'engine': args.engine, 'timestamp': _now(), 'results_dir': str(args.results_dir), 'figures_dir': str(args.figures_dir), 'result_csvs': sorted([path.name for path in args.results_dir.glob('*.csv')]), 'raw_csv_count': len(list((args.results_dir / 'raw').glob('*.csv'))) if (args.results_dir / 'raw').exists() else 0, 'figure_pngs': sorted([path.name for path in args.figures_dir.glob('*.png')]), 'figure_audit': str(audit_path), 'process_log': str(args.results_dir / 'process_log.csv')}
    (args.results_dir / 'submission_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(args.results_dir)
    print(args.figures_dir)
if __name__ == '__main__':
    main()
