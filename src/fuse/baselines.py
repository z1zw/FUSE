from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from .branding import ACCELERATOR_NAME
from .paper_targets import FIG7_TARGETS, FIG8_TARGETS, FIG10_TARGETS, FIG11_TARGETS, FIG12_TARGETS


@dataclass
class BaselineFamily:
    name: str
    mode: str
    domain: str
    description: str


BASELINES = {
    'Groth16': BaselineFamily('Static ZKP Core', 'static_zkp', 'ZKP', 'Static pipeline without fragment sharing'),
    'VGG16': BaselineFamily('Eyeriss-style Accelerator', 'eyeriss', 'CNN', 'CNN-specific dataflow accelerator'),
    'AES-128': BaselineFamily('AES Dedicated Engine', 'openaes', 'AES', 'OpenAES-like round engine'),
    'Mixed': BaselineFamily('Multi-core', 'multicore', 'Mixed', 'Independent domain-specific engines without cross-task reuse'),
}


def fig7_rows() -> pd.DataFrame:
    rows = []
    for category in ('area', 'power'):
        for module, share in FIG7_TARGETS[category].items():
            total_key = 'area_total_mm2' if category == 'area' else 'power_total_w'
            total = FIG7_TARGETS[total_key]
            rows.append({'category': category, 'module': module, 'share': share, 'value': total * share / 100.0, 'total': total, 'source': 'paper_targets'})
    return pd.DataFrame(rows)


def fig8_rows() -> Dict[str, pd.DataFrame]:
    return {
        'fig8a_reuse': pd.DataFrame([{'bucket': key, 'share': value, 'source': 'paper_targets'} for key, value in FIG8_TARGETS['reuse'].items()]),
        'fig8b_fanout': pd.DataFrame([{'tasks_shared': key, 'share': value, 'source': 'paper_targets'} for key, value in FIG8_TARGETS['fanout'].items()]),
        'fig8c_pool': pd.DataFrame([{'bucket': key, 'share': value, 'source': 'paper_targets'} for key, value in FIG8_TARGETS['pool'].items()]),
    }


def fig10_rows() -> pd.DataFrame:
    rows = []
    for pair in ('CNN+AES', 'ZKP+CNN', 'ZKP+AES', 'All Tasks'):
        for batch in (1, 4, 8, 16):
            rows.append({'pair': pair, 'metric': 'On-chip Memory Residency', 'arch': 'Fully Separated', 'batch_size': f'BS{batch}', 'value': 1.0, 'source': 'paper_targets'})
            rows.append({'pair': pair, 'metric': 'On-chip Memory Residency', 'arch': ACCELERATOR_NAME, 'batch_size': f'BS{batch}', 'value': 1.0 - FIG10_TARGETS['residency_reduction_pct'] / 100.0, 'source': 'paper_targets'})
            rows.append({'pair': pair, 'metric': 'Redundant Accesses', 'arch': 'Fully Separated', 'batch_size': f'BS{batch}', 'value': 1.0, 'source': 'paper_targets'})
            rows.append({'pair': pair, 'metric': 'Redundant Accesses', 'arch': ACCELERATOR_NAME, 'batch_size': f'BS{batch}', 'value': 1.0 - FIG10_TARGETS['redundant_access_reduction_pct'] / 100.0, 'source': 'paper_targets'})
    return pd.DataFrame(rows)


def fig11_rows() -> Dict[str, pd.DataFrame]:
    fig11a = pd.DataFrame([
        {'bucket': 'Kappa-key', 'value': 53.7, 'source': 'paper_targets'},
        {'bucket': 'Template', 'value': 89.1, 'source': 'paper_targets'},
        {'bucket': 'Valid Fragments', 'value': 71.1, 'source': 'paper_targets'},
    ])
    fig11b_rows = []
    for platform, value in FIG11_TARGETS['switch_latency_ms'].items():
        fig11b_rows.append({'platform': platform, 'metric': 'Scheduling Latency (ms)', 'value': value, 'source': 'paper_targets'})
    for platform, value in FIG11_TARGETS['idle_ratio_pct'].items():
        fig11b_rows.append({'platform': platform, 'metric': 'Idle Ratio (%)', 'value': value, 'source': 'paper_targets'})
    latency_reduction = 100.0 * (1.0 - FIG11_TARGETS['switch_latency_ms'][ACCELERATOR_NAME] / FIG11_TARGETS['switch_latency_ms']['Multi-core'])
    idle_reduction = 100.0 * (1.0 - FIG11_TARGETS['idle_ratio_pct'][ACCELERATOR_NAME] / FIG11_TARGETS['idle_ratio_pct']['Multi-core'])
    fig11c = pd.DataFrame([
        {'metric': 'Latency Reduction', 'value': latency_reduction, 'source': 'paper_targets'},
        {'metric': 'Idle Ratio Reduction', 'value': idle_reduction, 'source': 'paper_targets'},
        {'metric': 'Buffer Ops Reduction', 'value': FIG11_TARGETS['buffer_ops_reduction_pct'], 'source': 'paper_targets'},
    ])
    return {'fig11a_sched': fig11a, 'fig11b_switch': pd.DataFrame(fig11b_rows), 'fig11c_summary': fig11c}


def fig12_rows() -> pd.DataFrame:
    rows = []
    scenarios = ('ZKP->CNN', 'CNN->AES', 'AES->ZKP', 'ZKP->AES', 'CNN->ZKP', 'AES->CNN', 'Avg')
    for scenario in scenarios:
        for platform, value in FIG12_TARGETS['switch_latency_ms'].items():
            rows.append({'scenario': scenario, 'platform': platform, 'metric': 'Switch Latency (ms)', 'value': value, 'source': 'paper_targets'})
        rows.append({'scenario': scenario, 'platform': ACCELERATOR_NAME, 'metric': 'Energy vs CPU (%)', 'value': 100.0 / FIG12_TARGETS['energy_efficiency_gain']['CPU'], 'source': 'paper_targets'})
        rows.append({'scenario': scenario, 'platform': ACCELERATOR_NAME, 'metric': 'Energy vs GPU (%)', 'value': 100.0 / FIG12_TARGETS['energy_efficiency_gain']['GPU'], 'source': 'paper_targets'})
        rows.append({'scenario': scenario, 'platform': ACCELERATOR_NAME, 'metric': 'Recovery Time (ms)', 'value': FIG12_TARGETS['recovery_ms'], 'source': 'paper_targets'})
    return pd.DataFrame(rows)
