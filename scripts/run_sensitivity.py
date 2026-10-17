"""Sensitivity of the reported ratios to the simulator's own parameters.

A reviewer asked, fairly, whether the latency ratio ISOMER reports against the
transformer ASIC is a property of the hardware or of the model we gave the
baseline. The ASIC's datapath advantage is a constant we set, so the question
has to be answered by moving it, together with every other parameter the
timing model exposes.

Five axes, each swept while the other four stay at the configuration the paper
evaluates:

    compute        every entry of DEFAULT_OP_LATENCY scaled
    bandwidth      memory_bandwidth_bytes_per_cycle
    cache          fragment_cache_bytes
    reconfig       base_switch_cycles
    asic_adv       the transformer ASIC's datapath advantage on its own domain

The first four also sweep the Groth16 ablation, so the headline gain over the
degraded build can be checked for the same robustness.

    python scripts/run_sensitivity.py --output results/sensitivity.csv
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from fuse import reuse as reuse_mod
from fuse.ir import RuntimeTask
from fuse.reuse import MicrofragmentDecomposer
from fuse.sim.simulator import FuseSimulator, SimulationConfig
from fuse.workloads.zkp import build_groth16_like_task

#: The configuration Table 1 reports, which is the centre of every sweep.
BASE = dict(rfu_count=16,
            fragment_cache_bytes=2 * 1024 * 1024,
            base_switch_cycles=2300,
            memory_bandwidth_bytes_per_cycle=64)
LLM_FRESHNESS = 5_000_000

#: Each axis: the knob, the five settings, and which study it applies to.
AXES = {
    'compute':   [0.5, 0.75, 1.0, 1.5, 2.0],
    'bandwidth': [16, 32, 64, 128, 256],
    'cache':     [512 * 1024, 1024 * 1024, 2 * 1024 * 1024,
                  4 * 1024 * 1024, 8 * 1024 * 1024],
    'reconfig':  [575, 1150, 2300, 4600, 9200],
    'asic_adv':  [1.00, 1.04, 1.08, 1.16, 1.32],
}
#: The default of each axis, so a row can be marked as the evaluated point.
CENTRE = {'compute': 1.0, 'bandwidth': 64, 'cache': 2 * 1024 * 1024,
          'reconfig': 2300, 'asic_adv': 1.08}

PRISTINE_LATENCY = dict(reuse_mod.DEFAULT_OP_LATENCY)


def set_compute_scale(scale: float) -> None:
    """Scale the op latency table in place; estimate_latency_cycles reads it
    at call time, so mutating the dict is enough."""
    reuse_mod.DEFAULT_OP_LATENCY.clear()
    reuse_mod.DEFAULT_OP_LATENCY.update(
        {k: max(1, int(round(v * scale))) for k, v in PRISTINE_LATENCY.items()})


def config_for(axis: str, value, mode: str) -> SimulationConfig:
    kw = dict(BASE)
    if axis == 'bandwidth':
        kw['memory_bandwidth_bytes_per_cycle'] = int(value)
    elif axis == 'cache':
        kw['fragment_cache_bytes'] = int(value)
    elif axis == 'reconfig':
        kw['base_switch_cycles'] = int(value)
    return SimulationConfig(mode=mode, freshness_window_cycles=LLM_FRESHNESS,
                            **kw)


def run(tasks, decomposition, mode, axis, value, asic_adv=None):
    """One simulation. The ASIC advantage is patched per call, since it lives
    in a method rather than in the configuration."""
    cfg = config_for(axis, value, mode)
    sim = FuseSimulator(config=cfg)
    if asic_adv is not None and mode == 'transformer_asic':
        original = FuseSimulator._baseline_speed_factor

        def patched(self, task_type, _adv=asic_adv, _orig=original):
            factor = _orig(self, task_type)
            return _adv if task_type == 'LLM' else factor
        sim._baseline_speed_factor = patched.__get__(sim, FuseSimulator)
    runtime = [RuntimeTask(runtime_id=f'r{i}', task_graph=t, arrival_cycle=0)
               for i, t in enumerate(tasks)]
    result = sim.run(runtime, decomposition.microfragments)
    return result.metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'results' / 'sensitivity.csv')
    ap.add_argument('--prompt-tokens', type=int, default=55)
    ap.add_argument('--circuit', type=int, default=16384)
    ap.add_argument('--skip-llm', action='store_true')
    args = ap.parse_args()

    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
    rows = []

    # ---------------------------------------------------- the LLM ratios ----
    if not args.skip_llm:
        from transformers import AutoConfig
        from fuse.llm_zoo import DEFAULT_TILING_WIDTH, LLM_ZOO
        from fuse.workloads.llm import build_llm_prefill_task, spec_from_hf_config

        for meta in LLM_ZOO:
            spec = spec_from_hf_config(
                AutoConfig.from_pretrained(meta['hf_id']), name=meta['name'],
                label=meta['label'], hf_id=meta['hf_id'])
            task = build_llm_prefill_task(
                spec, seq_len=args.prompt_tokens, width=DEFAULT_TILING_WIDTH,
                content_tag='sens', task_id=f'{spec.name}_sens')
            decomposition = MicrofragmentDecomposer().decompose_tasks([task])
            for axis, values in AXES.items():
                for value in values:
                    set_compute_scale(value if axis == 'compute' else 1.0)
                    adv = value if axis == 'asic_adv' else CENTRE['asic_adv']
                    ours = run([task], decomposition, 'fuse', axis, value)
                    asic = run([task], decomposition, 'transformer_asic', axis,
                               value, asic_adv=adv)
                    rows.append({
                        'study': 'llm_vs_asic', 'workload': spec.label,
                        'axis': axis, 'value': value,
                        'is_centre': value == CENTRE[axis],
                        'isomer_ms': float(ours['makespan_ms']),
                        'baseline_ms': float(asic['makespan_ms']),
                        'ratio': float(ours['makespan_ms']) /
                                 float(asic['makespan_ms']),
                        'isomer_switch_cycles': float(ours['switch_cycles_total']),
                        'structural_reuse': float(ours['structural_reuse_rate']),
                    })
                    print(f"  llm {spec.label:16s} {axis:10s} {value!s:>9s} "
                          f"ratio {rows[-1]['ratio']:.4f}", flush=True)
            set_compute_scale(1.0)

    # ------------------------------------------- the Groth16 ablation -------
    task = build_groth16_like_task(circuit_size=args.circuit, rounds=4,
                                   task_id='groth16_sens')
    decomposition = MicrofragmentDecomposer().decompose_tasks([task])
    for axis in ('compute', 'bandwidth', 'cache', 'reconfig'):
        for value in AXES[axis]:
            set_compute_scale(value if axis == 'compute' else 1.0)
            ours = run([task], decomposition, 'fuse', axis, value)
            degraded = run([task], decomposition, 'degraded', axis, value)
            rows.append({
                'study': 'groth16_ablation', 'workload': 'Groth16 16k',
                'axis': axis, 'value': value,
                'is_centre': value == CENTRE[axis],
                'isomer_ms': float(ours['makespan_ms']),
                'baseline_ms': float(degraded['makespan_ms']),
                'ratio': float(degraded['makespan_ms']) /
                         float(ours['makespan_ms']),
                'isomer_switch_cycles': float(ours['switch_cycles_total']),
                'structural_reuse': float(ours['structural_reuse_rate']),
            })
            print(f"  zkp {axis:10s} {value!s:>9s} "
                  f"speedup {rows[-1]['ratio']:.3f}", flush=True)
    set_compute_scale(1.0)

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='') as fh:
        fh.write('# Sensitivity of the reported ratios to simulator parameters.\n')
        fh.write('# ratio: llm_vs_asic is ISOMER/ASIC latency (lower is better);\n')
        fh.write('#        groth16_ablation is degraded/ISOMER (higher is better).\n')
        fh.write('# is_centre marks the configuration the paper evaluates.\n')
        out.to_csv(fh, index=False)
    print(f'\n{len(out)} rows -> {args.output}')


if __name__ == '__main__':
    main()
