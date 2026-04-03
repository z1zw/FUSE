from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .branding import ACCELERATOR_NAME
from .viz import use_style as use_shared_style
PALETTE = {ACCELERATOR_NAME: '#D94841', 'Insecure LLM': '#6C757D', 'Canonical IR': '#4F7CAC', 'Secure-CT': '#2A9D8F', 'Secure-Rand': '#7C6EE6', 'Secure-Elastic': '#0F766E', 'Multi-core': '#495057', 'GPU-Tensor': '#C79500', 'Transformer-ASIC': '#1F7A8C'}

def use_style() -> None:
    use_shared_style()

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=320)
    plt.close(fig)

def _mode_color(mode: str) -> str:
    return PALETTE.get(mode, '#4F7CAC')

def plot_fig13(df: pd.DataFrame, out_dir: Path) -> None:
    use_style()
    attacks = [a for a in ['latency', 'reuse', 'schedule', 'fusion'] if a in set(df['attack'])]
    modes = list(df['mode'].unique())
    fig, axes = plt.subplots(1, 2, figsize=(16.4, 5.4), gridspec_kw={'width_ratios': [1.45, 1.0]})
    x = np.arange(len(attacks))
    width = 0.1 if len(modes) > 6 else 0.12
    for idx, mode in enumerate(modes):
        part = df[df['mode'] == mode].set_index('attack').reindex(attacks).reset_index()
        axes[0].bar(x + (idx - (len(modes) - 1) / 2.0) * width, part['accuracy'], width=width, color=_mode_color(mode), label=mode)
    axes[0].axhline(0.5, color='#333333', linestyle=':', linewidth=1.2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([a.capitalize() for a in attacks])
    axes[0].set_ylim(0.45, max(1.0, float(df['accuracy'].max()) + 0.06))
    axes[0].set_ylabel('Attack accuracy')
    axes[0].set_title('LLM-facing attack surface')
    axes[0].legend(ncol=4, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 1.22))
    score = df.groupby('mode', as_index=False)['security_score'].mean().sort_values('security_score', ascending=False)
    bars = axes[1].barh(score['mode'], score['security_score'], color=[_mode_color(mode) for mode in score['mode']])
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel('Mean security score')
    axes[1].set_title('Aggregate resilience')
    for bar, value in zip(bars, score['security_score']):
        axes[1].text(value + 0.01, bar.get_y() + bar.get_height() / 2.0, f'{value:.2f}', va='center', fontsize=9)
    fig.suptitle('Fig. 13. Secure Execution Closes LLM-Induced Side Channels', y=1.02, fontsize=15)
    _save(fig, out_dir / 'fig13_security.png')

def plot_fig14(df: pd.DataFrame, out_dir: Path) -> None:
    use_style()
    summary = df.groupby('mode', as_index=False).agg(semantic=('unique_semantic_ratio', 'mean'), schedule=('unique_schedule_ratio', 'mean'), latency_cv=('latency_cv', 'mean'), verifier=('verified_pass_rate', 'mean'), canonical=('canonical_consistency', 'mean'))
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.6))
    for row in summary.to_dict(orient='records'):
        axes[0].scatter(row['semantic'], row['schedule'], s=720 * row['verifier'] + 80, color=_mode_color(row['mode']), alpha=0.88, edgecolors='white', linewidths=1.0)
        axes[0].text(row['semantic'] + 0.015, row['schedule'] + 0.015, row['mode'], fontsize=9)
    axes[0].set_xlabel('Semantic variance')
    axes[0].set_ylabel('Schedule variance')
    axes[0].set_xlim(0.0, max(1.05, float(summary['semantic'].max()) + 0.05))
    axes[0].set_ylim(0.0, max(1.05, float(summary['schedule'].max()) + 0.05))
    axes[0].set_title('Prompt perturbation propagation')
    order = summary.sort_values(['canonical', 'verifier'], ascending=[False, False])
    width = 0.35
    x = np.arange(len(order))
    axes[1].bar(x - width / 2.0, order['canonical'], width=width, color='#4F7CAC', label='Canonical consistency')
    axes[1].bar(x + width / 2.0, order['verifier'], width=width, color='#2A9D8F', label='Verifier pass rate')
    axes[1].plot(x, order['latency_cv'], color='#D94841', marker='o', linewidth=2.0, label='Latency CV')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(order['mode'], rotation=20)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title('Determinism and verifier robustness')
    axes[1].legend(fontsize=8, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.19))
    fig.suptitle('Fig. 14. Canonical IR Stabilizes LLM Variability', y=1.02, fontsize=15)
    _save(fig, out_dir / 'fig14_llm_stability.png')

def plot_fig15(df: pd.DataFrame, out_dir: Path) -> None:
    use_style()
    fig, ax = plt.subplots(figsize=(9.8, 6.4))
    for row in df.to_dict(orient='records'):
        ax.scatter(row['latency_norm'], row['security_score'], s=760 * row['determinism_score'] + 80, color=_mode_color(row['mode']), alpha=0.9, edgecolors='white', linewidths=1.2)
        ax.text(row['latency_norm'] + 0.02, row['security_score'] + 0.012, row['mode'], fontsize=9)
    ax.set_xlabel('Latency ratio')
    ax.set_ylabel('Security score')
    ax.set_xlim(0.9, max(2.25, float(df['latency_norm'].max()) + 0.16))
    ax.set_ylim(0.0, 1.05)
    ax.set_title('Performance-security-determinism frontier')
    legend_sizes = [0.35, 0.65, 0.9]
    handles = [ax.scatter([], [], s=760 * s + 80, color='#B7BDC8', alpha=0.65, edgecolors='white') for s in legend_sizes]
    ax.legend(handles, [f'Determinism {s:.2f}' for s in legend_sizes], title='Bubble size', loc='lower left')
    fig.suptitle('Fig. 15. Secure-Elastic Narrows the Security-Performance Gap', y=1.02, fontsize=15)
    _save(fig, out_dir / 'fig15_secure_tradeoff.png')

def plot_fig16(df: pd.DataFrame, out_dir: Path) -> None:
    use_style()
    selected_modes = [mode for mode in [ACCELERATOR_NAME, 'Secure-Elastic', 'Secure-CT', 'Multi-core', 'GPU-Tensor'] if mode in set(df['mode'])]
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.6))
    p99 = df[df['mode'].isin(selected_modes)].pivot(index='case', columns='mode', values='p99_ms').reindex(columns=selected_modes)
    x = np.arange(len(p99.index))
    width = 0.15
    for idx, mode in enumerate(selected_modes):
        axes[0].bar(x + (idx - (len(selected_modes) - 1) / 2.0) * width, p99[mode], width=width, color=_mode_color(mode), label=mode)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(p99.index, rotation=18)
    axes[0].set_ylabel('End-to-end p99 (ms)')
    axes[0].set_title('Tail latency')
    axes[0].legend(fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.2))
    throughput = df[df['mode'].isin(selected_modes)].pivot(index='case', columns='mode', values='throughput_tps').reindex(columns=selected_modes)
    for mode in selected_modes:
        axes[1].plot(throughput.index, throughput[mode], marker='o', linewidth=2.1, color=_mode_color(mode), label=mode)
    axes[1].tick_params(axis='x', rotation=18)
    axes[1].set_ylabel('Requests per second')
    axes[1].set_title('Service throughput')
    energy = df[df['mode'].isin(selected_modes)].pivot(index='case', columns='mode', values='energy_per_request_nj').reindex(columns=selected_modes)
    image = axes[2].imshow(energy.values, aspect='auto', cmap='magma_r')
    axes[2].set_xticks(np.arange(len(selected_modes)))
    axes[2].set_xticklabels(selected_modes, rotation=20)
    axes[2].set_yticks(np.arange(len(energy.index)))
    axes[2].set_yticklabels(energy.index)
    axes[2].set_title('Energy per request')
    for i in range(energy.shape[0]):
        for j in range(energy.shape[1]):
            axes[2].text(j, i, f'{energy.values[i, j]:.1f}', ha='center', va='center', color='white', fontsize=7)
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle('Fig. 16. End-to-End Service Narratives Across Six Case Studies', y=1.03, fontsize=15)
    _save(fig, out_dir / 'fig16_case_studies.png')

def plot_fig17(df: pd.DataFrame, out_dir: Path) -> None:
    use_style()
    workloads = list(df['workload'].unique())
    configs = list(df['configuration'].unique())
    metrics = ['latency_norm', 'energy_norm', 'leakage_proxy']
    titles = ['Latency', 'Energy', 'Leakage proxy']
    cmaps = ['magma_r', 'magma_r', 'viridis_r']
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.4))
    for ax, metric, title, cmap in zip(axes, metrics, titles, cmaps):
        table = df.pivot(index='configuration', columns='workload', values=metric).reindex(index=configs, columns=workloads)
        image = ax.imshow(table.values, aspect='auto', cmap=cmap)
        ax.set_xticks(np.arange(len(workloads)))
        ax.set_xticklabels(workloads)
        ax.set_yticks(np.arange(len(configs)))
        ax.set_yticklabels(configs)
        ax.set_title(title)
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                ax.text(j, i, f'{table.values[i, j]:.2f}', ha='center', va='center', color='white', fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle('Fig. 17. System Ablations on Canonicalization, Reuse, and Secure Scheduling', y=1.03, fontsize=15)
    _save(fig, out_dir / 'fig17_ablation_plus.png')

def plot_fig18(df: pd.DataFrame, out_dir: Path) -> None:
    use_style()
    selected_modes = [mode for mode in [ACCELERATOR_NAME, 'Secure-Elastic', 'Multi-core'] if mode in set(df['mode'])]
    cases = list(df['case'].unique())
    stages = list(df['stage'].unique())
    fig, axes = plt.subplots(1, len(selected_modes), figsize=(18.5, 5.6), sharey=True)
    if len(selected_modes) == 1:
        axes = [axes]
    for ax, mode in zip(axes, selected_modes):
        part = df[df['mode'] == mode]
        y = np.arange(len(cases))
        left = np.zeros(len(cases))
        for stage_idx, stage in enumerate(stages):
            values = []
            for case in cases:
                row = part[(part['case'] == case) & (part['stage'] == stage)]
                values.append(float(row.iloc[0]['share']) if not row.empty else 0.0)
            color = plt.cm.tab20(stage_idx / max(1, len(stages) - 1))
            ax.barh(y, values, left=left, color=color, label=stage)
            left += np.asarray(values)
        ax.set_yticks(y)
        ax.set_yticklabels(cases)
        ax.set_xlim(0.0, 1.0)
        ax.set_title(mode)
        ax.set_xlabel('Latency share')
    axes[0].legend(fontsize=8, ncol=min(4, len(stages)), loc='upper center', bbox_to_anchor=(1.6, 1.23))
    fig.suptitle('Fig. 18. Service-Stage Decomposition Reveals Critical Paths', y=1.03, fontsize=15)
    _save(fig, out_dir / 'fig18_service_breakdown.png')

def plot_fig19(df: pd.DataFrame, out_dir: Path) -> None:
    use_style()
    cases = list(df['case'].unique())
    modes = [mode for mode in [ACCELERATOR_NAME, 'Secure-Elastic', 'Secure-CT', 'Multi-core', 'GPU-Tensor'] if mode in set(df['mode'])]
    cols = 3
    rows = int(np.ceil(len(cases) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18.5, 4.6 * rows), squeeze=False)
    for ax, case in zip(axes.ravel(), cases):
        part = df[df['case'] == case]
        for mode in modes:
            curve = part[part['mode'] == mode].sort_values('load_factor')
            ax.plot(curve['load_factor'], curve['p99_ms'], marker='o', linewidth=2.0, color=_mode_color(mode), label=mode)
        ax.set_title(case)
        ax.set_xlabel('Load factor')
        ax.set_ylabel('End-to-end p99 (ms)')
    for ax in axes.ravel()[len(cases):]:
        ax.axis('off')
    axes[0, 0].legend(fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(1.7, 1.25))
    fig.suptitle('Fig. 19. Load Sweeps Stress Secure Mode Under Service Bursts', y=1.02, fontsize=15)
    _save(fig, out_dir / 'fig19_load_sweep.png')

def plot_fig20(df: pd.DataFrame, out_dir: Path) -> None:
    use_style()
    cases = list(df['case'].unique())
    modes = list(df['mode'].unique())
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 5.8))
    latency = df.pivot(index='case', columns='mode', values='latency_gain_vs_multicore').reindex(index=cases, columns=modes)
    security = df.pivot(index='case', columns='mode', values='security_score').reindex(index=cases, columns=modes)
    for ax, table, title, cmap in zip(axes, [latency, security], ['Latency gain vs Multi-core', 'Security score'], ['magma', 'viridis']):
        image = ax.imshow(table.values, aspect='auto', cmap=cmap)
        ax.set_xticks(np.arange(len(modes)))
        ax.set_xticklabels(modes, rotation=20)
        ax.set_yticks(np.arange(len(cases)))
        ax.set_yticklabels(cases)
        ax.set_title(title)
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                value = table.values[i, j]
                text = 'nan' if pd.isna(value) else f'{value:.2f}'
                ax.text(j, i, text, ha='center', va='center', color='white', fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle('Fig. 20. Extended Comparison Across Services, Security, and Scale', y=1.03, fontsize=15)
    _save(fig, out_dir / 'fig20_extended_compare.png')

def plot_fig21(df: pd.DataFrame, out_dir: Path) -> None:
    use_style()
    workloads = list(df['workload'].unique())
    configs = list(df['configuration'].unique())
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.4))
    for ax, metric, title, cmap in zip(axes, ['latency_norm', 'leakage_proxy', 'balance_score'], ['Latency norm', 'Leakage proxy', 'Balance score'], ['magma_r', 'viridis_r', 'plasma']):
        table = df.pivot(index='configuration', columns='workload', values=metric).reindex(index=configs, columns=workloads)
        image = ax.imshow(table.values, aspect='auto', cmap=cmap)
        ax.set_xticks(np.arange(len(workloads)))
        ax.set_xticklabels(workloads)
        ax.set_yticks(np.arange(len(configs)))
        ax.set_yticklabels(configs)
        ax.set_title(title)
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                ax.text(j, i, f'{table.values[i, j]:.2f}', ha='center', va='center', color='white', fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle('Fig. 21. Secure-Elastic Component Ablations', y=1.03, fontsize=15)
    _save(fig, out_dir / 'fig21_secure_components.png')
