"""Take the reuse chain apart: configuration reuse, then value reuse.

The ablation elsewhere compares FUSE against a degraded build with both
mechanisms off, so it can only report what the two are worth together. This
runs a third arm in between:

    degraded        no template pinning, no value cache
    template_only   templates pin and representative paths shorten execution,
                    but a fragment never receives a previously computed result
    fuse            both

Two subtractions attribute the gap:

    degraded -> template_only    what configuration reuse is worth
    template_only -> fuse        what value reuse is worth

Arrival order matters and is varied deliberately. Within one request nothing
recurs, so a value cache has nothing to hit and the whole gain is
configuration. Across requests that arrive in sequence the cache does hit, and
the balance moves. Both regimes are reported.

    python scripts/run_reuse_breakdown.py --offline
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

ARMS = ('degraded', 'template_only', 'fuse')


def measure(graphs, arrivals, suite):
    """Run the three arms over one set of graphs and attribute the gap."""
    from fuse.ir import RuntimeTask

    dec = suite._decompose(graphs)
    runtime = [RuntimeTask(runtime_id=f't{i}', task_graph=g, arrival_cycle=a)
               for i, (g, a) in enumerate(zip(graphs, arrivals))]
    rec = {}
    for arm in ARMS:
        m, _ = suite._simulate(graphs, runtime, arm, decomposition=dec)
        rec[f'{arm}_ms'] = float(m['makespan_ms'])
        rec[f'{arm}_switch'] = float(m['switch_cycles_total'])
        rec[f'{arm}_struct'] = 100.0 * float(m['structural_reuse_rate'])
        rec[f'{arm}_value'] = 100.0 * float(m['result_reuse_rate'])
    span = rec['degraded_ms'] - rec['fuse_ms']
    rec['gain_total_ms'] = span
    rec['gain_config_ms'] = rec['degraded_ms'] - rec['template_only_ms']
    rec['gain_value_ms'] = rec['template_only_ms'] - rec['fuse_ms']
    rec['config_share'] = 100.0 * rec['gain_config_ms'] / span if span else 0.0
    rec['value_share'] = 100.0 * rec['gain_value_ms'] / span if span else 0.0
    rec['switch_saved_pct'] = (100.0 * (1 - rec['template_only_switch']
                                        / rec['degraded_switch'])
                               if rec['degraded_switch'] else 0.0)
    rec['gain_total_pct'] = (100.0 * span / rec['degraded_ms']
                             if rec['degraded_ms'] else 0.0)
    # A split is only meaningful when the gap is above scheduling jitter and,
    # for the value half, when the cache actually reported hits. Otherwise the
    # template -> fuse delta is reordering, not reuse, and is left unattributed.
    rec['attributable'] = bool(rec['gain_total_pct'] >= 0.5
                               and (rec['fuse_value'] > 0
                                    or rec['gain_value_ms'] <= 0))
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'runs' / 'census' / 'reuse_breakdown.csv')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()

    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

    from transformers import AutoConfig

    from fuse.llm_zoo import DEFAULT_TILING_WIDTH, LLM_ZOO, LLMZooExperimentSuite
    from fuse.workloads.aes import build_aes128_task
    from fuse.workloads.cnn import build_vgg16_like_task
    from fuse.workloads.llm import build_llm_prefill_task, spec_from_hf_config
    from fuse.workloads.zkp import build_groth16_like_task

    suite = LLMZooExperimentSuite
    meta = next(m for m in LLM_ZOO if m['name'] == 'qwen2_5_0_5b')
    spec = spec_from_hf_config(AutoConfig.from_pretrained(meta['hf_id']),
                               name=meta['name'], label=meta['label'], hf_id=meta['hf_id'])

    def llm(tag):
        return build_llm_prefill_task(spec, seq_len=55, width=DEFAULT_TILING_WIDTH,
                                      content_tag=tag, task_id=f'llm_{tag}')

    single = [('Groth16', [build_groth16_like_task(circuit_size=16384, rounds=4)]),
              ('VGG16', [build_vgg16_like_task(batch_size=16)]),
              ('AES-128', [build_aes128_task(blocks=8192)]),
              ('LLM', [llm('chain')])]

    # The same request served a second time, arriving once the first has
    # finished. "Same" means the same operands, which the cache identifies by
    # the originating request (contract C4), so the graph is replayed under
    # one task_id. A request built separately under another task_id is a
    # different request to the cache, which is the LLM (new prompt) arm.
    repeated = [('Groth16', [build_groth16_like_task(circuit_size=16384, rounds=4,
                                                     task_id='g') for _ in (1, 2)]),
                ('AES-128', [build_aes128_task(blocks=8192, task_id='a')
                             for _ in (1, 2)]),
                ('LLM (same prompt)', [llm('req'), llm('req')]),
                ('LLM (new prompt)', [llm('req1'), llm('req2')])]

    rows = []
    for name, graphs in single:
        rec = {'workload': name, 'regime': 'single request'}
        rec.update(measure(graphs, [0] * len(graphs), suite))
        rows.append(rec)

    for name, graphs in repeated:
        # Stagger so the second request starts after the first has retired,
        # which is what a served queue looks like.
        first = measure(graphs[:1], [0], suite)
        gap = int(first['degraded_ms'] * 1e6 * 1.2) + 1000
        rec = {'workload': name, 'regime': 'repeated request'}
        rec.update(measure(graphs, [i * gap for i in range(len(graphs))], suite))
        rows.append(rec)

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    print(f'{"workload":10s} {"regime":18s} {"degraded":>10s} {"template":>10s} '
          f'{"fuse":>10s}  {"config%":>8s} {"value%":>7s} {"cache hit%":>10s}')
    for _, r in frame.iterrows():
        print(f'{r["workload"]:10s} {r["regime"]:18s} {r["degraded_ms"]:10.4f} '
              f'{r["template_only_ms"]:10.4f} {r["fuse_ms"]:10.4f}  '
              + (f'{r["config_share"]:8.1f} {r["value_share"]:7.1f} '
                 f'{r["fuse_value"]:10.1f}' if r['attributable']
                 else f'{"--":>8s} {"--":>7s} {r["fuse_value"]:10.1f}'
                      f'   (gain {r["gain_total_pct"]:.2f}% = jitter)'))
    print(f'\nwrote {args.output}')


if __name__ == '__main__':
    main()
