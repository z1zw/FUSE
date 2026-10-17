"""What decides whether FUSE's reuse machinery shortens a request?

The ablation gives 8.6x on a small Groth16 circuit, 2.2x on a large one, 1.03x
on an MLP and 1.000x on VGG16 and on a decoder. Read as four domains these look
unrelated. Read as one axis they may not be: reconfiguration is a fixed cost per
fragment, so it is worth removing exactly when a fragment is small enough for
that cost to matter.

This measures both quantities for every workload, so the claim can be plotted
rather than asserted: work per fragment on one axis, what the machinery is
worth on the other.

    python scripts/run_granularity_ablation.py --offline
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

#: Groth16 is swept across circuit sizes because it is the one domain whose
#: fragment granularity we can vary without changing the workload.
ZKP_SIZES = [4096, 8192, 16384, 32768, 65536, 131072]
CNN_BATCHES = [8, 16, 32, 64]
AES_BLOCKS = [1024, 2048, 4096, 8192]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'runs' / 'census' / 'granularity.csv')
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()

    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

    from transformers import AutoConfig

    from fuse.ir import RuntimeTask
    from fuse.llm_zoo import DEFAULT_TILING_WIDTH, LLM_ZOO, LLMZooExperimentSuite
    from fuse.workloads.aes import build_aes128_task
    from fuse.workloads.cnn import build_mlp_path_task, build_vgg16_like_task
    from fuse.workloads.llm import build_llm_prefill_task, spec_from_hf_config
    from fuse.workloads.zkp import build_groth16_like_task

    cases = []
    for size in ZKP_SIZES:
        cases.append(('Groth16', f'{size // 1024}k',
                      build_groth16_like_task(circuit_size=size, rounds=args.rounds)))
    for batch in CNN_BATCHES:
        cases.append(('VGG16', f'b{batch}', build_vgg16_like_task(batch_size=batch)))
    for blocks in AES_BLOCKS:
        cases.append(('AES-128', f'{blocks // 1024}k', build_aes128_task(blocks=blocks)))
    cases.append(('MLP', 'b32', build_mlp_path_task(batch_size=32)))
    for meta in LLM_ZOO:
        spec = spec_from_hf_config(AutoConfig.from_pretrained(meta['hf_id']),
                                   name=meta['name'], label=meta['label'],
                                   hf_id=meta['hf_id'])
        cases.append(('LLM', meta['label'],
                      build_llm_prefill_task(spec, seq_len=55,
                                             width=DEFAULT_TILING_WIDTH,
                                             content_tag='gran',
                                             task_id=f'{spec.name}_gran')))

    rows = []
    for domain, point, task in cases:
        decomposition = LLMZooExperimentSuite._decompose([task])
        fragments = decomposition.microfragments
        runtime = [RuntimeTask(runtime_id='t', task_graph=task, arrival_cycle=0)]
        record = {'domain': domain, 'point': point,
                  'fragments': len(fragments),
                  'total_ops': sum(f.estimated_ops for f in fragments)}
        record['ops_per_fragment'] = record['total_ops'] / max(1, record['fragments'])
        for mode in ('fuse', 'degraded'):
            metrics, _ = LLMZooExperimentSuite._simulate(
                [task], runtime, mode, decomposition=decomposition)
            record[f'{mode}_ms'] = float(metrics['makespan_ms'])
            record[f'{mode}_switch'] = float(metrics['switch_cycles_total'])
        record['speedup'] = record['degraded_ms'] / record['fuse_ms']
        record['switch_gain'] = record['degraded_switch'] / max(1.0, record['fuse_switch'])
        rows.append(record)
        print(f"  {domain:9s} {point:16s} {record['ops_per_fragment']:10.0f} ops/frag  "
              f"speedup {record['speedup']:6.2f}x", flush=True)

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f'\nwrote {args.output}')

    corr = frame['ops_per_fragment'].apply('log').corr(frame['speedup'].apply('log'))
    print(f'log-log correlation between work per fragment and speedup: {corr:.3f}')


if __name__ == '__main__':
    main()
