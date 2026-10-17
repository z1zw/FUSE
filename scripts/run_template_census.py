"""Which instruction templates each domain's fragments fall into.

The paper's claim is that heterogeneous workloads share execution structure.
Figure 8 asserts it for the ZKP path alone, from a single-bar fan-out panel and
a template pool with no other domain to compare against. This decomposes all
four domains the same way and counts what comes out, so the overlap is
something the figure can show rather than something the text claims.

    python scripts/run_template_census.py --output runs/census/template_census.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'runs' / 'census' / 'template_census.csv')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()

    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

    from transformers import AutoConfig

    from fuse.reuse import MicrofragmentDecomposer
    from fuse.llm_zoo import DEFAULT_TILING_WIDTH, LLM_ZOO
    from fuse.workloads.aes import build_aes128_task
    from fuse.workloads.cnn import build_vgg16_like_task
    from fuse.workloads.llm import build_llm_prefill_task, spec_from_hf_config
    from fuse.workloads.zkp import build_groth16_like_task

    graphs = [
        ('Groth16', build_groth16_like_task(circuit_size=16384, rounds=4)),
        ('VGG16', build_vgg16_like_task(batch_size=16)),
        ('AES-128', build_aes128_task(blocks=8192)),
    ]
    # One representative model, so the LLM column is a workload and not a mean.
    meta = next(m for m in LLM_ZOO if m['name'] == 'qwen2_5_0_5b')
    spec = spec_from_hf_config(AutoConfig.from_pretrained(meta['hf_id']),
                               name=meta['name'], label=meta['label'],
                               hf_id=meta['hf_id'])
    graphs.append(('LLM', build_llm_prefill_task(
        spec, seq_len=55, width=DEFAULT_TILING_WIDTH,
        content_tag='census', task_id='llm_census')))

    rows = []
    for domain, graph in graphs:
        decomposition = MicrofragmentDecomposer().decompose_tasks([graph])
        fragments = decomposition.microfragments
        families = Counter(f.template_family for f in fragments)
        structural = {f.structural_hash for f in fragments}
        semantic = {f.semantic_hash for f in fragments}
        total = len(fragments)
        for family, count in sorted(families.items()):
            rows.append({'domain': domain, 'template_family': family,
                         'count': count, 'share': 100.0 * count / total,
                         'fragments': total,
                         'structural_classes': len(structural),
                         'semantic_classes': len(semantic)})
        print(f'  {domain:9s} {total:6d} fragments, '
              f'{len(structural):5d} structural classes '
              f'({total / max(1, len(structural)):6.1f}x), '
              f'{len(semantic):5d} semantic', flush=True)

    # How many domains does one structural class turn up in? This is the
    # cross-domain sharing the paper claims, measured rather than asserted.
    from collections import defaultdict
    where = defaultdict(set)
    for domain, graph in graphs:
        for f in MicrofragmentDecomposer().decompose_tasks([graph]).microfragments:
            where[f.structural_hash].add(domain)
    fanout = Counter(len(v) for v in where.values())
    total_classes = sum(fanout.values())
    print()
    print('structural classes by number of domains they appear in:')
    for k in sorted(fanout):
        print(f'   {k} domain(s): {fanout[k]:4d}  ({100.0 * fanout[k] / total_classes:5.1f}%)')
        rows.append({'domain': '_fanout', 'template_family': f'{k}',
                     'count': fanout[k], 'share': 100.0 * fanout[k] / total_classes,
                     'fragments': total_classes, 'structural_classes': total_classes,
                     'semantic_classes': 0})

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f'\nwrote {args.output}')

    wide = frame[frame['domain'] != '_fanout'].pivot(index='template_family', columns='domain',
                       values='share').fillna(0.0)
    print('\ntemplate family share (%):')
    print(wide.round(1).to_string())


if __name__ == '__main__':
    main()
