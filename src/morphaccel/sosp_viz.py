from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .branding import ACCELERATOR_NAME
from .viz import use_style as use_shared_style
PALETTE = {ACCELERATOR_NAME: '#D94841', 'Insecure LLM': '#6C757D', 'Canonical IR': '#4F7CAC', 'Secure-CT': '#2A9D8F', 'Secure-Rand': '#7C6EE6', 'Multi-core': '#495057', 'GPU-Tensor': '#C79500', 'Transformer-ASIC': '#1F7A8C'}

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
    attacks = ['latency', 'reuse', 'schedule']
    modes = list(df['mode'].unique())
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.0), gridspec_kw={'width_ratios': [1.25, 1.0]})
    x = np.arange(len(attacks))
    width = 0.13
    for idx, mode in enumerate(modes):
        part = df[df['mode'] == mode].set_index('attack').reindex(attacks).reset_index()
        axes[0].bar(x + (idx - (len(modes) - 1) / 2.0) * width, part['accuracy'], width=width, color=_mode_color(mode), label=mode)
    axes[0].axhline(0.5, color='#444444', linestyle='--', linewidth=1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([a.capitalize() for a in attacks])
    axes[0].set_ylim(0.45, max(1.0, float(df['accuracy'].max()) + 0.05))
    axes[0].set_ylabel('Attack accuracy')
    axes[0].set_title('Security attacks across deployment modes')
    axes[0].legend(ncol=3, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 1.18))
    score = df.groupby('mode', as_index=False)['security_score'].mean().sort_values('security_score', ascending=False)
    bars = axes[1].barh(score['mode'], score['security_score'], color=[_mode_color(mode) for mode in score['mode']])
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel('Mean security score')
    axes[1].set_title('Aggregate resilience')
    for bar, value in zip(bars, score['security_score']):
        axes[1].text(value + 0.01, bar.get_y() + bar.get_height() / 2.0, f'{value:.2f}', va='center', fontsize=9)
    fig.suptitle('Fig. 13. LLM-Facing Side-Channel Robustness', y=1.02, fontsize=15)
    _save(fig, out_dir / 'fig13_security.png')

def plot_fig14(df: pd.DataFrame, out_dir: Path) -> None:
    use_style()
    modes = list(df['mode'].unique())
    metrics = [('unique_semantic_ratio', 'Semantic variance', True), ('unique_schedule_ratio', 'Schedule variance', True), ('latency_cv', 'Latency CV', True), ('verified_pass_rate', 'Verifier pass rate', False)]
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 8.2))
    for ax, (metric, title, lower_better) in zip(axes.ravel(), metrics):
        summary = df.groupby('mode', as_index=False)[metric].mean()
        summary = summary.sort_values(metric, ascending=lower_better)
        colors = [_mode_color(mode) for mode in summary['mode']]
        bars = ax.barh(summary['mode'], summary[metric], color=colors)
        ax.set_title(title)
        if lower_better:
            ax.set_xlim(0.0, max(1.0, float(summary[metric].max()) * 1.15))
        else:
            ax.set_xlim(0.0, 1.05)
        for bar, value in zip(bars, summary[metric]):
            ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2.0, f'{value:.2f}', va='center', fontsize=9)
    fig.suptitle('Fig. 14. Prompt Perturbation Stability and IR Determinism', y=1.02, fontsize=15)
    _save(fig, out_dir / 'fig14_llm_stability.png')

def plot_fig15(df: pd.DataFrame, out_dir: Path) -> None:
    use_style()
    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    for row in df.to_dict(orient='records'):
        ax.scatter(row['latency_norm'], row['security_score'], s=700 * row['determinism_score'] + 80, color=_mode_color(row['mode']), alpha=0.88, edgecolors='white', linewidths=1.0)
        ax.text(row['latency_norm'] + 0.02, row['security_score'] + 0.01, row['mode'], fontsize=9)
    ax.set_xlabel('Latency ratio')
    ax.set_ylabel('Security score')
    ax.set_xlim(0.9, max(2.1, float(df['latency_norm'].max()) + 0.15))
    ax.set_ylim(0.0, 1.05)
    ax.set_title('Performance-security-determinism trade-off')
    legend_sizes = [0.35, 0.65, 0.9]
    handles = [ax.scatter([], [], s=700 * s + 80, color='#B0B8C5', alpha=0.6, edgecolors='white') for s in legend_sizes]
    ax.legend(handles, [f'Determinism {s:.2f}' for s in legend_sizes], title='Bubble size', loc='lower left')
    fig.suptitle('Fig. 15. Secure Deployment Trade Space', y=1.02, fontsize=15)
    _save(fig, out_dir / 'fig15_secure_tradeoff.png')

def plot_fig16(df: pd.DataFrame, out_dir: Path) -> None:
    use_style()
    selected_modes = [mode for mode in [ACCELERATOR_NAME, 'Secure-Rand', 'Secure-CT', 'Multi-core', 'GPU-Tensor'] if mode in set(df['mode'])]
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.2))
    p99 = df[df['mode'].isin(selected_modes)].pivot(index='case', columns='mode', values='p99_ms').reindex(columns=selected_modes)
    x = np.arange(len(p99.index))
    width = 0.14
    for idx, mode in enumerate(selected_modes):
        axes[0].bar(x + (idx - (len(selected_modes) - 1) / 2.0) * width, p99[mode], width=width, color=_mode_color(mode), label=mode)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(p99.index, rotation=18)
    axes[0].set_ylabel('p99 latency (ms)')
    axes[0].set_title('Tail latency across case studies')
    axes[0].legend(fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.22))
    energy_column = 'energy_per_request_nj' if 'energy_per_request_nj' in df.columns else 'energy_per_task_nj'
    energy = df[df['mode'].isin(selected_modes)].pivot(index='case', columns='mode', values=energy_column).reindex(columns=selected_modes)
    for mode in selected_modes:
        axes[1].plot(energy.index, energy[mode], marker='o', linewidth=2.0, color=_mode_color(mode), label=mode)
    axes[1].set_ylabel('Energy per task (nJ)')
    axes[1].set_title('Energy efficiency across case studies')
    axes[1].tick_params(axis='x', rotation=18)
    fig.suptitle('Fig. 16. Mixed-Domain Service Case Studies', y=1.03, fontsize=15)
    _save(fig, out_dir / 'fig16_case_studies.png')

def plot_fig17(df: pd.DataFrame, out_dir: Path) -> None:
    use_style()
    workloads = list(df['workload'].unique())
    configs = list(df['configuration'].unique())
    metrics = ['latency_norm', 'energy_norm', 'leakage_proxy']
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.2))
    for ax, metric, title in zip(axes, metrics, ['Latency', 'Energy', 'Leakage proxy']):
        table = df.pivot(index='configuration', columns='workload', values=metric).reindex(index=configs, columns=workloads)
        image = ax.imshow(table.values, aspect='auto', cmap='magma_r' if metric != 'leakage_proxy' else 'viridis_r')
        ax.set_xticks(np.arange(len(workloads)))
        ax.set_xticklabels(workloads)
        ax.set_yticks(np.arange(len(configs)))
        ax.set_yticklabels(configs)
        ax.set_title(title)
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                value = table.values[i, j]
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', color='white', fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle('Fig. 17. Ablations on Canonicalization, Reuse, and Secure Scheduling', y=1.03, fontsize=15)
    _save(fig, out_dir / 'fig17_ablation_plus.png')
