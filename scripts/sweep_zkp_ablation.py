"""Which Groth16 configuration reproduces the ablation number the paper reports?

The paper states a 91.3% latency reduction against the degraded build, which is
a 11.5x speedup. results/fig9_ablation.csv gives 85.3% and 6.80x. The ablation
is not touched by results/calibration.json, so the difference has to come from
the circuit the two runs were given. This sweeps the circuit size and the round
count and prints what each produces.

    python scripts/sweep_zkp_ablation.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

#: What the paper claims, as a latency ratio between the two builds.
PAPER_SPEEDUP = 11.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='*',
                        default=[4096, 8192, 16384, 32768, 65536, 131072])
    parser.add_argument('--rounds', type=int, nargs='*', default=[1, 2, 4, 8])
    parser.add_argument('--rfus', type=int, default=16)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'runs' / 'census' / 'zkp_ablation_sweep.csv')
    args = parser.parse_args()

    from fuse.ir import RuntimeTask
    from fuse.llm_zoo import LLMZooExperimentSuite
    from fuse.workloads.zkp import build_groth16_like_task

    rows = []
    for size in args.sizes:
        for rounds in args.rounds:
            record = {'circuit_size': size, 'rounds': rounds}
            task = build_groth16_like_task(circuit_size=size, rounds=rounds)
            decomposition = LLMZooExperimentSuite._decompose([task])
            runtime = [RuntimeTask(runtime_id='zkp', task_graph=task, arrival_cycle=0)]
            for mode in ('fuse', 'degraded'):
                metrics, _ = LLMZooExperimentSuite._simulate(
                    [task], runtime, mode, decomposition=decomposition)
                record[f'{mode}_ms'] = float(metrics['makespan_ms'])
            record['speedup'] = record['degraded_ms'] / record['fuse_ms']
            record['reduction_pct'] = 100.0 * (1.0 - record['fuse_ms'] / record['degraded_ms'])
            rows.append(record)
            print(f"  size {size:7d}  rounds {rounds}  "
                  f"degraded {record['degraded_ms']:9.4f} ms  "
                  f"FUSE {record['fuse_ms']:9.4f} ms  "
                  f"speedup {record['speedup']:6.2f}x  "
                  f"({record['reduction_pct']:5.1f}%)", flush=True)

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    best = frame.iloc[(frame['speedup'] - PAPER_SPEEDUP).abs().argmin()]
    print(f'\nclosest to the reported {PAPER_SPEEDUP}x: '
          f"circuit_size={int(best['circuit_size'])}, rounds={int(best['rounds'])}, "
          f"speedup={best['speedup']:.2f}x ({best['reduction_pct']:.1f}%)")
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
