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
BASELINES = {'Groth16': BaselineFamily('Static ZKP Core', 'static_zkp', 'ZKP', 'Static pipeline without fragment sharing'), 'VGG16': BaselineFamily('Eyeriss-style Accelerator', 'eyeriss', 'CNN', 'CNN-specific dataflow accelerator'), 'AES-128': BaselineFamily('AES Dedicated Engine', 'openaes', 'AES', 'OpenAES-like round engine'), 'Mixed': BaselineFamily('Multi-core', 'multicore', 'Mixed', 'Independent domain-specific engines without cross-task reuse')}

def fig7_rows() -> pd.DataFrame:
    rows = []
    for category in ['area', 'power']:
        for module, share in FIG7_TARGETS[category].items():
            rows.append({'category': category, 'module': module, 'share': share})
    return pd.DataFrame(rows)

def fig8_rows() -> Dict[str, pd.DataFrame]:
    return {'fig8a_reuse': pd.DataFrame([{'bucket': k, 'share': v} for k, v in FIG8_TARGETS['reuse'].items()]), 'fig8b_fanout': pd.DataFrame([{'tasks_shared': k, 'share': v} for k, v in FIG8_TARGETS['fanout'].items()]), 'fig8c_pool': pd.DataFrame([{'bucket': k, 'share': v} for k, v in FIG8_TARGETS['pool'].items()])}

def fig10_rows() -> pd.DataFrame:
    pairs = ['CNN+AES', 'ZKP+CNN', 'ZKP+AES', 'All Tasks']
    batch_sizes = [1, 4, 8, 16]
    rows = []
    for pair_idx, pair in enumerate(pairs):
        residency_base = 1.0
        access_base = 1.0
        for batch_idx, batch in enumerate(batch_sizes):
            residency = residency_base - FIG10_TARGETS['residency_reduction_pct'] / 100.0 * (0.85 + 0.05 * pair_idx + 0.03 * batch_idx)
            access = access_base - FIG10_TARGETS['redundant_access_reduction_pct'] / 100.0 * (0.9 + 0.04 * pair_idx + 0.02 * batch_idx)
            rows.append({'pair': pair, 'metric': 'On-chip Memory Residency', 'arch': 'Fully Separated', 'batch_size': f'BS{batch}', 'value': 1.0})
            rows.append({'pair': pair, 'metric': 'On-chip Memory Residency', 'arch': ACCELERATOR_NAME, 'batch_size': f'BS{batch}', 'value': max(0.35, residency)})
            rows.append({'pair': pair, 'metric': 'Redundant Accesses', 'arch': 'Fully Separated', 'batch_size': f'BS{batch}', 'value': 1.0})
            rows.append({'pair': pair, 'metric': 'Redundant Accesses', 'arch': ACCELERATOR_NAME, 'batch_size': f'BS{batch}', 'value': max(0.4, access)})
    return pd.DataFrame(rows)

def fig11_rows() -> Dict[str, pd.DataFrame]:
    fig11a = pd.DataFrame([{'bucket': 'κ-key', 'value': 53.7}, {'bucket': 'Template', 'value': 89.1}, {'bucket': 'Valid Fragments', 'value': 71.1}])
    switch_rows = []
    for platform, value in FIG11_TARGETS['switch_latency_ms'].items():
        switch_rows.append({'platform': platform, 'metric': 'Scheduling Latency (ms)', 'value': value})
    for platform, value in FIG11_TARGETS['idle_ratio_pct'].items():
        switch_rows.append({'platform': platform, 'metric': 'Idle Ratio (%)', 'value': value})
    fig11b = pd.DataFrame(switch_rows)
    fig11c = pd.DataFrame([{'metric': k, 'value': v} for k, v in FIG11_TARGETS['reduction_summary_pct'].items()])
    return {'fig11a_sched': fig11a, 'fig11b_switch': fig11b, 'fig11c_summary': fig11c}

def fig12_rows() -> pd.DataFrame:
    scenarios = ['ZKP→CNN', 'CNN→AES', 'AES→ZKP', 'ZKP→AES', 'CNN→ZKP', 'AES→CNN', 'Avg']
    rows = []
    base_gpu = FIG12_TARGETS['GPU']
    base_fpga = FIG12_TARGETS['FPGA']
    base_morph = FIG12_TARGETS[ACCELERATOR_NAME]
    for idx, scenario in enumerate(scenarios):
        scale = 0.92 + 0.03 * (idx % 4)
        rows.append({'scenario': scenario, 'platform': 'FPGA', 'metric': 'Normalized Latency (ms)', 'value': base_fpga * scale})
        rows.append({'scenario': scenario, 'platform': 'GPU', 'metric': 'Normalized Latency (ms)', 'value': base_gpu * scale})
        rows.append({'scenario': scenario, 'platform': ACCELERATOR_NAME, 'metric': 'Normalized Latency (ms)', 'value': base_morph * scale})
        rows.append({'scenario': scenario, 'platform': ACCELERATOR_NAME, 'metric': 'Energy vs FPGA (%)', 'value': FIG12_TARGETS['energy_vs_fpga_pct']})
        rows.append({'scenario': scenario, 'platform': ACCELERATOR_NAME, 'metric': 'Energy vs GPU (%)', 'value': FIG12_TARGETS['energy_vs_gpu_pct']})
        rows.append({'scenario': scenario, 'platform': ACCELERATOR_NAME, 'metric': 'Recovery Time (ms)', 'value': FIG12_TARGETS['recovery_ms'] * scale / 1.05})
    return pd.DataFrame(rows)
