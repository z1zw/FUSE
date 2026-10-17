"""Do the published results reproduce from the C++ engine?

The numbers in ``results/raw`` were produced by the C++ engine, not the Python
one. Two things show it: the stored metrics carry a ``buffer_ops`` column the
Python engine never emits, and the nested ``<scenario>/scenario/`` directories
are written only on the C++ path. ``cpp/build`` ships empty, so a run on a
machine without ``g++`` or ``cl`` falls back to the Python engine, and the two
engines do not agree. On the overall-performance points the Python engine puts
AES-128 overhead at 9.9% where the paper reports 4.9%, and Groth16 at 1.34x
where the paper reports 1.28x.

This script settles the question without re-deriving anything. Each scenario
directory under ``results/raw`` kept the exact input tables that were handed to
the engine when the paper's numbers were produced::

    results/raw/<scenario>/scenario/tasks.csv
                                   /microfragments.csv
                                   /representative_paths.csv
                                   /<mode>_metrics.csv     <- what it returned

So the binary is rebuilt from ``cpp/src`` and re-run on those same tables, and
its output is compared column by column against what is stored beside them.
Nothing is regenerated, so a difference can only come from the engine.

    python scripts/verify_cpp_repro.py                 # every scenario
    python scripts/verify_cpp_repro.py --pattern fig4  # just one figure
    python scripts/verify_cpp_repro.py --tolerance 0    # exact equality

Exit status is 0 when every scenario reproduces and 1 otherwise, so it can
stand in a reproduction pipeline.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

INPUTS = ('tasks.csv', 'microfragments.csv', 'representative_paths.csv')
# Metrics that describe the run rather than the machine, compared in this order.
INTERESTING = (
    'makespan_cycles', 'makespan_ms', 'dispatches', 'result_reuse_hits',
    'structural_reuse_hits', 'result_reuse_rate', 'structural_reuse_rate',
    'switch_cycles_total', 'rfu_utilization', 'energy_nj',
    'fvm_peak_bytes', 'fvm_avg_residency_bytes', 'fvm_reads', 'fvm_writes',
    'fvm_redundant_reads', 'lrc_template_hits', 'lrc_template_misses',
    'lrc_switch_events', 'lrc_prefetch_events', 'scheduler_decisions',
    'kappa_key_match_ratio', 'template_hit_ratio', 'valid_fragment_ratio',
    'buffer_ops', 'representative_path_count', 'representative_path_reuse_rate',
)


def read_metrics(path: Path) -> dict[str, str]:
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if rows and 'key' in rows[0]:
        return {r['key']: r['value'] for r in rows}
    # The engine also writes a two-column file without a header row.
    with path.open(newline='', encoding='utf-8') as handle:
        return {r[0]: r[1] for r in csv.reader(handle) if len(r) >= 2}


def scenarios(pattern: str | None):
    for metrics in sorted((ROOT / 'results' / 'raw').glob('*/scenario/*_metrics.csv')):
        scenario_dir = metrics.parent
        if any(not (scenario_dir / name).exists() for name in INPUTS):
            continue
        name = scenario_dir.parent.name
        if pattern and pattern not in name:
            continue
        yield name, scenario_dir, metrics, metrics.name[:-len('_metrics.csv')]


def compare(stored, fresh, tolerance):
    diffs = []
    for key in INTERESTING:
        if key not in stored:
            continue
        if key not in fresh:
            diffs.append(f'{key}: stored {stored[key]}, absent now')
            continue
        try:
            a, b = float(stored[key]), float(fresh[key])
        except ValueError:
            if str(stored[key]) != str(fresh[key]):
                diffs.append(f'{key}: stored {stored[key]!r} now {fresh[key]!r}')
            continue
        if abs(a - b) > max(tolerance, abs(a) * tolerance):
            diffs.append(f'{key}: stored {a!r} now {b!r}')
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pattern', help='only scenarios whose name contains this')
    ap.add_argument('--tolerance', type=float, default=1e-9,
                    help='relative tolerance, 0 for exact equality')
    ap.add_argument('--verbose', action='store_true',
                    help='list every scenario, not only the ones that differ')
    args = ap.parse_args()

    from fuse.cpp_bridge import CppSimulatorBridge

    bridge = CppSimulatorBridge(ROOT)
    try:
        binary = bridge.build()
    except (OSError, subprocess.SubprocessError) as exc:
        print(f'cannot build the C++ engine: {exc}', file=sys.stderr)
        print('This check needs g++ or cl. The published results came from '
              'the C++ engine, so without it there is nothing to compare '
              'against.', file=sys.stderr)
        return 2
    if not Path(binary).exists():
        print(f'the build reported success but {binary} is missing',
              file=sys.stderr)
        return 2
    print(f'engine: {binary}')

    out_root = ROOT / 'results' / 'repro_check'
    out_root.mkdir(parents=True, exist_ok=True)

    total = reproduced = 0
    failures = []
    for name, scenario_dir, metrics_path, mode in scenarios(args.pattern):
        total += 1
        fresh_dir = out_root / name
        fresh_dir.mkdir(parents=True, exist_ok=True)
        fresh_metrics = fresh_dir / f'{mode}_metrics.csv'
        cmd = [str(binary),
               '--tasks', str(scenario_dir / 'tasks.csv'),
               '--microfragments', str(scenario_dir / 'microfragments.csv'),
               '--paths', str(scenario_dir / 'representative_paths.csv'),
               '--mode', mode,
               '--metrics-out', str(fresh_metrics),
               '--events-out', str(fresh_dir / f'{mode}_events.csv')]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            failures.append((name, mode, [f'the engine failed: '
                                          f'{exc.stderr.decode(errors="replace")[:200]}']))
            continue
        diffs = compare(read_metrics(metrics_path), read_metrics(fresh_metrics),
                        args.tolerance)
        if diffs:
            failures.append((name, mode, diffs))
        else:
            reproduced += 1
            if args.verbose:
                print(f'  ok    {name} [{mode}]')

    print(f'\nscenarios compared: {total}')
    print(f'reproduced exactly: {reproduced}')
    if failures:
        print(f'differing:          {len(failures)}\n')
        for name, mode, diffs in failures:
            print(f'  {name} [{mode}]')
            for diff in diffs[:8]:
                print(f'      {diff}')
            if len(diffs) > 8:
                print(f'      ... and {len(diffs) - 8} more')
        return 1
    print('\nthe published results reproduce from the C++ engine')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
