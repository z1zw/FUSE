"""What FUSE's own mechanisms buy on the language-model domain.

Figure 9 ablates FUSE against a degraded build of itself -- same datapath, same
array, with template pinning and representative-path reuse switched off -- on
Groth16, an MLP and VGG16. The language-model domain was added to the paper
after that figure was made and is missing from it, so this runs the same
ablation over the model zoo and writes it in the same shape.

    python scripts/run_llm_ablation.py --output runs/llm_zoo/llm_ablation.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from fuse.llm_zoo import DEFAULT_TILING_WIDTH, LLM_ZOO
from fuse.workloads.llm import build_llm_prefill_task, spec_from_hf_config

#: FUSE against a build of itself with the reuse machinery disabled. The
#: baseline modes of the zoo are other people's hardware; this one is ours.
MODES = ('fuse', 'degraded')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'runs' / 'llm_zoo' / 'llm_ablation.csv')
    parser.add_argument('--rfus', type=int, default=16)
    parser.add_argument('--width', type=int, default=DEFAULT_TILING_WIDTH)
    parser.add_argument('--prompt-tokens', type=int, default=55)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()

    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

    from transformers import AutoConfig

    from fuse.ir import RuntimeTask
    from fuse.llm_zoo import LLMZooExperimentSuite

    rows = []
    for meta in LLM_ZOO:
        spec = spec_from_hf_config(AutoConfig.from_pretrained(meta['hf_id']),
                                   name=meta['name'], label=meta['label'],
                                   hf_id=meta['hf_id'])
        task = build_llm_prefill_task(
            spec, seq_len=args.prompt_tokens, width=args.width,
            content_tag='ablation', task_id=f'{spec.name}_ablation',
        )
        record = {'model': spec.name, 'model_label': spec.label,
                  'parameter_count': int(spec.parameter_count())}
        decomposition = LLMZooExperimentSuite._decompose([task])
        runtime = [RuntimeTask(runtime_id='prefill', task_graph=task, arrival_cycle=0)]
        for mode in MODES:
            metrics, _ = LLMZooExperimentSuite._simulate(
                [task], runtime, mode, decomposition=decomposition)
            record[f'{mode}_latency_ms'] = float(metrics['makespan_ms'])
            record[f'{mode}_energy_j'] = float(metrics['energy_nj']) / 1e9
            record[f'{mode}_structural_reuse'] = float(metrics['structural_reuse_rate'])
            record[f'{mode}_result_reuse'] = float(metrics['result_reuse_rate'])
            record[f'{mode}_switch_cycles'] = float(metrics['switch_cycles_total'])
        record['speedup'] = record['degraded_latency_ms'] / record['fuse_latency_ms']
        record['energy_gain'] = record['degraded_energy_j'] / record['fuse_energy_j']
        rows.append(record)
        print(f"  {spec.label:16s} degraded {record['degraded_latency_ms']:9.3f} ms  "
              f"FUSE {record['fuse_latency_ms']:9.3f} ms  "
              f"speedup {record['speedup']:.3f}x", flush=True)

    frame = pd.DataFrame(rows).sort_values('parameter_count')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f'\nwrote {args.output}')


if __name__ == '__main__':
    main()
