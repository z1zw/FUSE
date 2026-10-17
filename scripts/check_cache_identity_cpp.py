"""Does the C++ engine refuse a cached result to a different request?

The Python engine's lookup now compares the consumer's data identity with the
producer's (``FeatureVectorManager._valid``), which is contract C4. The C++
engine carries the same check in ``cache_valid``. This script exercises the
C++ one directly, through the same serializer and binary the benchmarks use.

Two scenarios, identical in every parameter:

    REPLAY     one AES graph submitted 16 times. One request served 16 times,
               so its operands are the same each time and the hits are the
               legitimate ones 4.3 reports.
    DISTINCT   16 AES graphs built separately with different task_ids. Sixteen
               different requests, so no hit between them is admissible.

If C4 holds, REPLAY keeps its hits and DISTINCT has none.

    python scripts/check_cache_identity_cpp.py
"""
from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from fuse.cpp_bridge import CppSimulatorBridge
from fuse.ir import RuntimeTask
from fuse.serialization import ScenarioSerializer
from fuse.sim.simulator import CYCLES_PER_MS
from fuse.workloads import build_aes128_task

N = 16


def tasks_for(graphs):
    # The arrival pattern of benchmarks._raw_fig10, so a producer has retired
    # before the next consumer dispatches.
    out, arrival_ms = [], 0.0
    for g in graphs:
        for i in range(N // len(graphs)):
            out.append(RuntimeTask(runtime_id=f'{g.task_id}_rt{i}', task_graph=g,
                                   arrival_cycle=int(arrival_ms * CYCLES_PER_MS)))
            arrival_ms += 0.045 if i % 2 == 0 else 0.12
    return out


def run(binary, graphs, label):
    with tempfile.TemporaryDirectory(prefix='fuse_c4_') as tmp:
        d = Path(tmp)
        files = ScenarioSerializer().write_tables(
            ScenarioSerializer().build_tables(graphs, tasks_for(graphs)), d)
        metrics = d / 'fuse_metrics.csv'
        subprocess.run([str(binary), '--tasks', str(files.task_path),
                        '--microfragments', str(files.micro_path),
                        '--paths', str(files.path_path), '--mode', 'fuse',
                        '--metrics-out', str(metrics),
                        '--events-out', str(d / 'fuse_events.csv')],
                       check=True, capture_output=True)
        with metrics.open(newline='', encoding='utf-8') as f:
            rows = list(csv.reader(f))
        m = {r[0]: r[1] for r in rows if len(r) >= 2 and r[0] != 'key'}
    return (label, int(float(m['dispatches'])),
            int(float(m['result_reuse_hits'])), float(m['result_reuse_rate']),
            float(m['makespan_ms']))


def main() -> int:
    binary = CppSimulatorBridge(ROOT).build()
    print(f'engine: {binary}\n')
    one = build_aes128_task(4096, 'pair_aes')
    rows = [run(binary, [one], 'REPLAY   one request x16'),
            run(binary, [build_aes128_task(4096, f'aes_{i}') for i in range(N)],
                'DISTINCT 16 requests')]
    print(f'{"scenario":28s} {"dispatch":>9s} {"hits":>7s} {"rate":>8s} '
          f'{"makespan_ms":>12s}')
    for label, disp, hits, rate, ms in rows:
        print(f'{label:28s} {disp:9d} {hits:7d} {rate:8.4f} {ms:12.4f}')
    replay, distinct = rows
    ok = replay[2] > 0 and distinct[2] == 0
    print()
    print('C4 holds in the C++ engine' if ok else
          'C4 VIOLATED: a result crossed between distinct requests')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
