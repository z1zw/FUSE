from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .branding import ACCELERATOR_NAME
PALETTE = {'baseline': '#34495E', 'morph': '#E76F51', 'accent': '#2878B5', 'green': '#2A9D8F', 'gold': '#F4A261', 'teal': '#1B998B', 'purple': '#7A5CFA', 'sand': '#E9C46A', 'cyan': '#48BFE3', 'gray': '#7D8597', 'rose': '#D1495B', 'indigo': '#5A189A', 'mint': '#80ED99', 'soft_bg': '#F5F7FB', 'panel_bg': '#EEF3FB', 'panel_alt': '#FFF6EC'}
FIGURE_AUDIT = {'fig4_latency': {'title': 'Fig. 4. Latency Against Domain-Specific Baselines', 'x_label': 'Input size or batch size', 'y_label': 'Latency (ms) / latency ratio', 'legend': ['Baseline', ACCELERATOR_NAME], 'colors': [PALETTE['baseline'], PALETTE['morph'], PALETTE['green'], PALETTE['gold']], 'layout': '1x4 panels'}, 'fig5_energy': {'title': 'Fig. 5. Energy Efficiency Across Baselines', 'x_label': 'Platform', 'y_label': 'Normalized energy efficiency (x)', 'legend': [ACCELERATOR_NAME, 'Software and hardware baselines'], 'colors': [PALETTE['baseline'], PALETTE['accent'], PALETTE['green'], PALETTE['morph'], PALETTE['purple'], PALETTE['teal'], PALETTE['gold']], 'layout': '1x3 panels'}, 'fig6_mixed': {'title': 'Fig. 6. Mixed-Trace Makespan and Redundant Reads', 'x_label': 'Interleave window (ms)', 'y_label': 'Makespan (ms), redundant reads', 'legend': ['Multi-core', ACCELERATOR_NAME, 'Multi-core redundant reads', f'{ACCELERATOR_NAME} redundant reads'], 'colors': [PALETTE['baseline'], PALETTE['morph'], PALETTE['accent'], PALETTE['green']], 'layout': 'dual-axis single panel'}, 'fig7_area_power': {'title': 'Fig. 7. Physical Cost Breakdown from OpenROAD and CACTI', 'x_label': 'Breakdown panel', 'y_label': 'Area / power / energy / timing', 'legend': ['Logic area', 'Logic power', 'Memory macros', 'Timing and utilization'], 'colors': [PALETTE['accent'], PALETTE['morph'], PALETTE['green'], PALETTE['gold'], PALETTE['teal'], PALETTE['sand']], 'layout': '2x2 panels'}, 'fig8_reuse': {'title': 'Fig. 8. Reuse Decomposition and RFU Fan-Out', 'x_label': 'Share (%) or reuse class', 'y_label': 'Share (%)', 'legend': ['Reuse composition', 'Cross-task fan-out', 'Template pool composition'], 'colors': [PALETTE['accent'], PALETTE['gold'], PALETTE['green'], PALETTE['morph'], PALETTE['sand']], 'layout': '1x3 panels'}, 'fig9_ablation': {'title': 'Fig. 9. Ablation Against a Degraded Design', 'x_label': 'Latency (ms) or throughput (ops/ms)', 'y_label': 'Configuration', 'legend': ['Degraded', ACCELERATOR_NAME], 'colors': [PALETTE['baseline'], PALETTE['morph']], 'layout': '1x3 panels'}, 'fig10_memory': {'title': 'Fig. 10. On-Chip Residency and Redundant Accesses', 'x_label': 'Batch size', 'y_label': 'Normalized value (ratio)', 'legend': ['Fully Separated', ACCELERATOR_NAME], 'colors': [PALETTE['baseline'], PALETTE['morph']], 'layout': '2x4 panels'}, 'fig11': {'title': 'Fig. 11. Cross-Task Scheduling and Reuse Quality', 'x_label': 'Scheduling quality and reduction metrics', 'y_label': 'Ratio, latency, or absolute count', 'legend': ['Fragment matching', 'Platform comparison', 'Reduction summary', 'Transition annotation'], 'colors': [PALETTE['accent'], PALETTE['morph'], PALETTE['green'], PALETTE['gold']], 'layout': '2x2 panels'}, 'fig12_reconfig': {'title': 'Fig. 12. Transition Overhead, Energy, and Recovery', 'x_label': 'Scenario', 'y_label': 'Transition overhead (ms), energy ratio (%), recovery time (ms)', 'legend': ['CPU', 'GPU', ACCELERATOR_NAME], 'colors': [PALETTE['baseline'], PALETTE['gold'], PALETTE['morph'], PALETTE['green'], PALETTE['accent'], PALETTE['purple']], 'layout': '2x2 panels'}}

def use_style() -> None:
    mpl.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans', 'font.size': 12.0, 'axes.titlesize': 14.0, 'axes.labelsize': 11.8, 'axes.titleweight': 'semibold', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.18, 'grid.linestyle': '--', 'legend.frameon': False, 'figure.facecolor': 'white', 'axes.facecolor': PALETTE['soft_bg'], 'savefig.facecolor': 'white', 'savefig.bbox': 'tight', 'axes.axisbelow': True, 'xtick.labelsize': 10.8, 'ytick.labelsize': 10.8})

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=320)
    plt.close(fig)

def _label_bars(ax, bars, fmt: str, *, dy: float=0.0) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + dy, fmt.format(value), ha='center', va='bottom', fontsize=9)

def _metric_alias(frame: pd.DataFrame, canonical: str, *aliases: str) -> pd.DataFrame:
    return frame[frame['metric'].isin({canonical, *aliases})].copy()

def _transition_metric(frame: pd.DataFrame) -> pd.DataFrame:
    part = _metric_alias(frame, 'Transition Overhead (ms)', 'Switch Latency (ms)')
    if not part.empty:
        part.loc[:, 'metric'] = 'Transition Overhead (ms)'
    return part

def plot_fig4(df: pd.DataFrame, out_dir: Path) -> None:
    use_style()
    fig = plt.figure(figsize=(15.6, 4.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.9], wspace=0.3)
    for idx, domain in enumerate(['Groth16', 'VGG16', 'AES-128']):
        ax = fig.add_subplot(gs[0, idx])
        part = df[df['domain'] == domain]
        ax.plot(part['x_label'], part['baseline_ms'], marker='o', linewidth=2.2, color=PALETTE['baseline'], label=part.iloc[0]['baseline_name'])
        ax.plot(part['x_label'], part['morph_ms'], marker='s', linewidth=2.4, color=PALETTE['morph'], label=ACCELERATOR_NAME)
        ax.set_title(domain, pad=8)
        ax.set_xlabel('Input size' if domain != 'VGG16' else 'Batch size')
        ax.set_ylabel('Latency (ms)')
        ax.legend(loc='upper left', fontsize=8)
    ax = fig.add_subplot(gs[0, 3])
    summary = df.groupby('domain', as_index=False)['norm'].mean()
    bars = ax.bar(summary['domain'], summary['norm'], color=[PALETTE['green'] if v <= 1.0 else PALETTE['gold'] for v in summary['norm']], width=0.62)
    ax.axhline(1.0, color=PALETTE['baseline'], linewidth=1.2, linestyle='--')
    ax.set_title('Normalized latency', pad=8)
    ax.set_ylabel('Latency ratio')
    ax.set_ylim(0.75, max(1.15, float(summary['norm'].max()) + 0.05))
    for bar, value in zip(bars, summary['norm']):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.01, f'{value:.2f}x', ha='center', va='bottom', fontsize=9)
    fig.suptitle(FIGURE_AUDIT['fig4_latency']['title'], y=1.02, fontsize=15)
    _save(fig, out_dir / 'fig4_latency.png')

def plot_fig5(df: pd.DataFrame, out_dir: Path) -> None:
    use_style()
    tasks = list(df['task'].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.9), sharey=True)
    palette = [PALETTE['baseline'], PALETTE['accent'], PALETTE['green'], PALETTE['morph'], PALETTE['purple'], PALETTE['teal'], PALETTE['gold']]
    for ax, task in zip(axes, tasks):
        part = df[df['task'] == task].sort_values('efficiency_norm')
        colors = [PALETTE['morph'] if p == ACCELERATOR_NAME else palette[i % len(palette)] for i, p in enumerate(part['platform'])]
        bars = ax.bar(part['platform'], part['efficiency_norm'], color=colors, width=0.72)
        ax.set_yscale('log')
        ax.set_title(task)
        ax.tick_params(axis='x', rotation=30)
        ax.set_ylabel('Normalized energy efficiency (x)')
        for bar, value, platform in zip(bars, part['efficiency_norm'], part['platform']):
            if value >= 4.0 or platform == ACCELERATOR_NAME:
                ax.text(bar.get_x() + bar.get_width() / 2.0, value * 1.08, f'{value:.1f}x', ha='center', va='bottom', fontsize=8)
    fig.suptitle(FIGURE_AUDIT['fig5_energy']['title'], y=1.02, fontsize=15)
    _save(fig, out_dir / 'fig5_energy.png')

def plot_fig6(df: pd.DataFrame, out_dir: Path) -> None:
    use_style()
    fig, ax1 = plt.subplots(figsize=(9.6, 5.1))
    x = np.arange(len(df))
    width = 0.34
    ax1.bar(x - width / 2.0, df['baseline_makespan_ms'], width=width, color=PALETTE['baseline'], label='Multi-core')
    ax1.bar(x + width / 2.0, df['morph_makespan_ms'], width=width, color=PALETTE['morph'], label=ACCELERATOR_NAME)
    ax1.set_ylabel('Makespan (ms)')
    ax1.set_xlabel('Interleave window (ms)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['window_ms'])
    ax2 = ax1.twinx()
    ax2.plot(x, df['baseline_redundant_reads'], color=PALETTE['accent'], marker='o', linewidth=2.0, label='Multi-core redundant reads')
    ax2.plot(x, df['morph_redundant_reads'], color=PALETTE['green'], marker='s', linewidth=2.0, label=f'{ACCELERATOR_NAME} redundant reads')
    ax2.set_ylabel('Redundant reads')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', ncol=2, fontsize=8)
    fig.suptitle(FIGURE_AUDIT['fig6_mixed']['title'], y=1.02, fontsize=15)
    _save(fig, out_dir / 'fig6_mixed.png')

def plot_fig7(df: pd.DataFrame, out_dir: Path) -> None:
    use_style()
    fig, axes = plt.subplots(2, 2, figsize=(15.6, 9.2))
    logic_area = df[df['category'] == 'logic_area'].copy()
    logic_power = df[df['category'] == 'logic_power'].copy()
    timing = df[df['category'] == 'timing'].copy()
    probe = df[df['category'] == 'probe'].copy()
    memory_area = df[df['category'] == 'memory_area'].copy()
    memory_energy = df[df['category'] == 'memory_energy'].copy()
    memory_leakage = df[df['category'] == 'memory_leakage'].copy()

    ax = axes[0, 0]
    ax.set_facecolor(PALETTE['panel_bg'])
    if not logic_area.empty:
        max_area = float(logic_area['value'].max())
        bars = ax.bar(logic_area['module'], logic_area['value'], color=[PALETTE['accent']] * len(logic_area), width=0.62)
        for bar, value in zip(bars, logic_area['value']):
            ax.text(bar.get_x() + bar.get_width() / 2.0, value + max(max_area * 0.02, 0.0002), f'{value:.4f} mm2', ha='center', va='bottom', fontsize=9)
        ax.set_ylim(0.0, max_area * 1.16)
    ax.set_title('(a) Logic footprint from OpenROAD')
    ax.set_ylabel('Area (mm2)')
    ax.tick_params(axis='x', rotation=10)

    ax = axes[0, 1]
    ax.set_facecolor(PALETTE['panel_alt'])
    power_labels = logic_power['module'].tolist()
    power_values = logic_power['value'].to_numpy()
    bars = ax.bar(power_labels, power_values, color=[PALETTE['morph'], PALETTE['gold']][:len(power_labels)], width=0.58)
    if len(power_values):
        ax.set_ylim(0.0, float(power_values.max()) * 1.18 + 0.00001)
    ax.set_ylabel('Power (W)')
    ax.set_title('(b) Logic power split and timing headroom')
    for bar, value in zip(bars, power_values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + max(float(power_values.max()) * 0.03, 0.00002), f'{value:.4f} W', ha='center', va='bottom', fontsize=9)
    if not timing.empty:
        fmax = float(timing['value'].iloc[0])
        util = float(probe['value'].iloc[0]) if not probe.empty else 0.0
        ax.text(0.98, 0.95, f'Fmax: {fmax:.1f} MHz\nRFU util: {util:.3f}', transform=ax.transAxes, ha='right', va='top', fontsize=9.2, bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='#B5C0D0'))

    ax = axes[1, 0]
    ax.set_facecolor(PALETTE['soft_bg'])
    area_order = ['Feature buffer area', 'Fragment cache area', 'Micro-op buffer area']
    area_part = memory_area.set_index('module').reindex(area_order).reset_index()
    bars = ax.bar(area_part['module'], area_part['value'], color=[PALETTE['teal'], PALETTE['green'], PALETTE['sand']][:len(area_part)], width=0.58)
    ax.set_title('(c) Memory macro area')
    ax.set_ylabel('Area (mm2)')
    ax.tick_params(axis='x', rotation=12)
    for bar, value in zip(bars, area_part['value']):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + max(float(area_part['value'].max()) * 0.02, 0.2), f'{value:.2f}', ha='center', va='bottom', fontsize=9)

    ax = axes[1, 1]
    ax.set_facecolor('#F7F0FF')
    leak_part = memory_leakage.copy()
    leak_part['macro'] = leak_part['module'].str.replace(' leakage', '', regex=False)
    energy_part = memory_energy.copy()
    energy_part['macro'] = energy_part['module'].str.replace(' read energy', '', regex=False)
    x = np.arange(len(leak_part))
    width = 0.34
    leak_bars = ax.bar(x - width / 2.0, leak_part['value'], width=width, color=PALETTE['purple'], label='Leakage (mW)')
    ax.set_xticks(x)
    ax.set_xticklabels(leak_part['macro'], rotation=12)
    ax.set_ylabel('Leakage (mW)')
    ax.set_title('(d) Memory energy and leakage')
    ax2 = ax.twinx()
    energy_bars = ax2.bar(x + width / 2.0, energy_part['value'], width=width, color=PALETTE['accent'], label='Read energy (nJ)')
    ax2.set_ylabel('Read energy (nJ/access)')
    for bar, value in zip(leak_bars, leak_part['value']):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + max(float(leak_part['value'].max()) * 0.02, 40.0), f'{value:.0f}', ha='center', va='bottom', fontsize=8.8)
    for bar, value in zip(energy_bars, energy_part['value']):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, value + max(float(energy_part['value'].max()) * 0.03, 0.04), f'{value:.2f}', ha='center', va='bottom', fontsize=8.8)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc='upper right', fontsize=8.8)

    fig.suptitle(FIGURE_AUDIT['fig7_area_power']['title'], y=1.02, fontsize=15)
    _save(fig, out_dir / 'fig7_area_power.png')

def plot_fig8(fig8a: pd.DataFrame, fig8b: pd.DataFrame, fig8c: pd.DataFrame, out_dir: Path) -> None:
    use_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8))
    bars = axes[0].barh(fig8a['bucket'], fig8a['share'], color=[PALETTE['accent'], PALETTE['gold'], PALETTE['green']])
    axes[0].set_title('Reuse composition')
    axes[0].set_xlabel('Share (%)')
    for bar, value in zip(bars, fig8a['share']):
        axes[0].text(value + 0.7, bar.get_y() + bar.get_height() / 2.0, f'{value:.1f}%', va='center', fontsize=9)
    bars = axes[1].bar(fig8b['tasks_shared'], fig8b['share'], color=[PALETTE['baseline'], PALETTE['accent'], PALETTE['morph']])
    axes[1].set_title('Cross-task fan-out')
    axes[1].set_ylabel('Share (%)')
    for bar, value in zip(bars, fig8b['share']):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, value + 1.0, f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
    bars = axes[2].bar(fig8c['bucket'], fig8c['share'], color=[PALETTE['accent'], PALETTE['morph'], PALETTE['green'], PALETTE['gold'], PALETTE['sand']][:len(fig8c)])
    axes[2].set_title('Template pool composition')
    axes[2].set_ylabel('Share (%)')
    axes[2].tick_params(axis='x', rotation=25)
    for bar, value in zip(bars, fig8c['share']):
        axes[2].text(bar.get_x() + bar.get_width() / 2.0, value + 0.8, f'{value:.0f}%', ha='center', va='bottom', fontsize=9)
    fig.suptitle(FIGURE_AUDIT['fig8_reuse']['title'], y=1.03, fontsize=15)
    _save(fig, out_dir / 'fig8_reuse.png')

def plot_fig9(df: pd.DataFrame, out_dir: Path) -> None:
    use_style()
    fig = plt.figure(figsize=(16.4, 5.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 0.95], wspace=0.34)
    workload_order = ['Groth16', 'MLP', 'VGG16']
    x = np.arange(len(workload_order))
    width = 0.34
    degraded_latency = [float(df[(df['task'] == task) & (df['metric'] == 'latency_degraded')]['value'].iloc[0]) for task in workload_order]
    fuse_latency = [float(df[(df['task'] == task) & (df['metric'] == 'latency_morph')]['value'].iloc[0]) for task in workload_order]
    degraded_tp = [float(df[(df['task'] == task) & (df['metric'] == 'throughput_degraded')]['value'].iloc[0]) for task in workload_order]
    fuse_tp = [float(df[(df['task'] == task) & (df['metric'] == 'throughput_morph')]['value'].iloc[0]) for task in workload_order]
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor(PALETTE['panel_bg'])
    bars0 = ax0.bar(x - width / 2.0, degraded_latency, width=width, color=PALETTE['baseline'], label='Degraded latency', edgecolor='#22313F', linewidth=0.6)
    bars1 = ax0.bar(x + width / 2.0, fuse_latency, width=width, color=PALETTE['morph'], label=f'{ACCELERATOR_NAME} latency', edgecolor='#22313F', linewidth=0.6)
    ax0.set_yscale('log')
    ax0.set_xticks(x)
    ax0.set_xticklabels(workload_order)
    ax0.set_title('(a) Latency across workloads', pad=10)
    ax0.set_xlabel('Workload')
    ax0.set_ylabel('Latency (ms, log scale)')
    for bars in [bars0, bars1]:
        for bar in bars:
            value = bar.get_height()
            ax0.text(bar.get_x() + bar.get_width() / 2.0, value * 1.08, f'{value:.3g}', ha='center', va='bottom', fontsize=8.8)
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_facecolor(PALETTE['panel_alt'])
    bars2 = ax1.bar(x - width / 2.0, degraded_tp, width=width, color=PALETTE['accent'], label='Degraded throughput', edgecolor='#22313F', linewidth=0.6)
    bars3 = ax1.bar(x + width / 2.0, fuse_tp, width=width, color=PALETTE['green'], label=f'{ACCELERATOR_NAME} throughput', edgecolor='#22313F', linewidth=0.6)
    ax1.set_yscale('log')
    ax1.set_xticks(x)
    ax1.set_xticklabels(workload_order)
    ax1.set_title('(b) Throughput across workloads', pad=10)
    ax1.set_xlabel('Workload')
    ax1.set_ylabel('Throughput (ops/ms, log scale)')
    for bars in [bars2, bars3]:
        for bar in bars:
            value = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, value * 1.08, f'{value:.3g}', ha='center', va='bottom', fontsize=8.8)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor('#F8F1FF')
    latency_speedup = np.array(degraded_latency) / np.array(fuse_latency)
    throughput_gain = np.array(fuse_tp) / np.array(degraded_tp)
    width2 = 0.36
    bars4 = ax2.bar(x - width2 / 2.0, latency_speedup, width=width2, color=PALETTE['purple'], label='Latency speedup', edgecolor='#22313F', linewidth=0.6)
    bars5 = ax2.bar(x + width2 / 2.0, throughput_gain, width=width2, color=PALETTE['gold'], label='Throughput gain', edgecolor='#22313F', linewidth=0.6)
    ax2.axhline(1.0, color='#4A4E69', linewidth=1.2, linestyle='--')
    ax2.set_xticks(x)
    ax2.set_xticklabels(workload_order)
    ax2.set_title('(c) Relative gain over degraded', pad=10)
    ax2.set_xlabel('Workload')
    ax2.set_ylabel('Improvement ratio (x)')
    ax2.set_ylim(0.0, max(float(max(latency_speedup.max(), throughput_gain.max())) * 1.18, 1.4))
    _label_bars(ax2, bars4, '{:.2f}x', dy=0.04)
    _label_bars(ax2, bars5, '{:.2f}x', dy=0.04)
    fig.legend(handles=[mpl.patches.Patch(color=PALETTE['baseline'], label='Degraded latency'), mpl.patches.Patch(color=PALETTE['morph'], label=f'{ACCELERATOR_NAME} latency'), mpl.patches.Patch(color=PALETTE['accent'], label='Degraded throughput'), mpl.patches.Patch(color=PALETTE['green'], label=f'{ACCELERATOR_NAME} throughput'), mpl.patches.Patch(color=PALETTE['purple'], label='Latency speedup'), mpl.patches.Patch(color=PALETTE['gold'], label='Throughput gain')], loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.01), fontsize=9.4)
    fig.suptitle(FIGURE_AUDIT['fig9_ablation']['title'], y=1.02, fontsize=15)
    _save(fig, out_dir / 'fig9_ablation.png')

def plot_fig10(df: pd.DataFrame, out_dir: Path) -> None:
    use_style()
    pairs = list(df['pair'].unique())
    metrics = list(df['metric'].unique())
    fig, axes = plt.subplots(2, 4, figsize=(16.6, 8.6), sharey='row')
    for col, pair in enumerate(pairs):
        for row, metric in enumerate(metrics):
            ax = axes[row, col]
            part = df[(df['pair'] == pair) & (df['metric'] == metric)]
            pivot = part.pivot(index='batch_size', columns='arch', values='value').reset_index()
            x = np.arange(len(pivot))
            width = 0.35
            ax.bar(x - width / 2.0, pivot['Fully Separated'], width=width, color=PALETTE['baseline'], label='Fully Separated')
            ax.bar(x + width / 2.0, pivot[ACCELERATOR_NAME], width=width, color=PALETTE['morph'], label=ACCELERATOR_NAME)
            ax.set_xticks(x)
            ax.set_xticklabels(pivot['batch_size'])
            ax.set_ylim(0.0, 1.2)
            ax.set_title(f'{pair}\n{metric}')
            if col == 0:
                ax.set_ylabel('Normalized value (ratio)')
            if row == 1:
                ax.set_xlabel('Batch size')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(FIGURE_AUDIT['fig10_memory']['title'], y=1.08, fontsize=15)
    _save(fig, out_dir / 'fig10_memory.png')

def plot_fig11(fig11a: pd.DataFrame, fig11b: pd.DataFrame, fig11c: pd.DataFrame, fig12: pd.DataFrame, out_dir: Path) -> None:
    use_style()
    fig, axes = plt.subplots(2, 2, figsize=(15.4, 10.0))

    ax = axes[0, 0]
    ax.set_facecolor(PALETTE['panel_bg'])
    order = ['Kappa-key', 'Template', 'Valid Fragments']
    match_part = fig11a.set_index('bucket').reindex(order).reset_index()
    bars = ax.barh(match_part['bucket'], match_part['value'], color=[PALETTE['accent'], PALETTE['gold'], PALETTE['morph']], edgecolor='#22313F', linewidth=0.6)
    ax.set_title('(a) Fragment and template match quality')
    ax.set_xlabel('Ratio (%)')
    ax.set_xlim(0.0, 105.0)
    for bar, value in zip(bars, match_part['value']):
        ax.text(value + 1.0, bar.get_y() + bar.get_height() / 2.0, f'{value:.1f}%', va='center', fontsize=9)

    ax = axes[0, 1]
    ax.set_facecolor(PALETTE['panel_alt'])
    sched = fig11b[fig11b['metric'] == 'Scheduling Latency (ms)'].copy()
    idle = fig11b[fig11b['metric'] == 'Idle Ratio (%)'].copy()
    x = np.arange(len(sched))
    width = 0.34
    bars_sched = ax.bar(x - width / 2.0, sched['value'], width=width, color=PALETTE['accent'], label='Scheduling latency (ms)')
    ax.set_ylabel('Scheduling latency (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(sched['platform'])
    ax.set_title('(b) Scheduler latency and idle ratio')
    ax2 = ax.twinx()
    bars_idle = ax2.bar(x + width / 2.0, idle['value'], width=width, color=PALETTE['gold'], label='Idle ratio (%)')
    ax2.set_ylabel('Idle ratio (%)')
    for bar in bars_sched:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + max(0.004, value * 0.05), f'{value:.3f}', ha='center', va='bottom', fontsize=8.8)
    for bar in bars_idle:
        value = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, value + max(0.4, value * 0.01), f'{value:.1f}%', ha='center', va='bottom', fontsize=8.8)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc='upper left', fontsize=8.6)

    ax = axes[1, 0]
    ax.set_facecolor(PALETTE['soft_bg'])
    metric_order = ['Scheduling latency', 'Idle ratio', 'Redundant reads']
    baseline_vals = []
    fuse_vals = []
    for metric in metric_order:
        baseline_vals.append(float(fig11c[(fig11c['metric'] == metric) & (fig11c['platform'] == 'Multi-core')]['absolute'].iloc[0]))
        fuse_vals.append(float(fig11c[(fig11c['metric'] == metric) & (fig11c['platform'] == ACCELERATOR_NAME)]['absolute'].iloc[0]))
    x = np.arange(len(metric_order))
    width = 0.32
    bars_base = ax.bar(x - width / 2.0, baseline_vals, width=width, color=PALETTE['baseline'], label='Multi-core')
    bars_fuse = ax.bar(x + width / 2.0, fuse_vals, width=width, color=PALETTE['morph'], label=ACCELERATOR_NAME)
    ax.set_xticks(x)
    ax.set_xticklabels(['Sched. ms', 'Idle %', 'Redundant\nreads'])
    ax.set_title('(c) Absolute scheduling and reuse cost')
    ax.set_ylabel('Absolute value')
    peak = max(baseline_vals + fuse_vals)
    for bars in [bars_base, bars_fuse]:
        for bar in bars:
            value = bar.get_height()
            label = f'{value:.3f}' if value < 1.0 else f'{value:.1f}'
            ax.text(bar.get_x() + bar.get_width() / 2.0, value + max(peak * 0.02, 0.02), label, ha='center', va='bottom', fontsize=8.6)
    ax.legend(loc='upper right', fontsize=8.8)

    ax = axes[1, 1]
    ax.set_facecolor('#F7F0FF')
    reduction_labels = ['Sched. reduction', 'Idle delta', 'Read reduction']
    reduction_values = [
        100.0 - float(fig11c[(fig11c['metric'] == 'Scheduling latency') & (fig11c['platform'] == ACCELERATOR_NAME)]['value'].iloc[0]),
        100.0 - float(fig11c[(fig11c['metric'] == 'Idle ratio') & (fig11c['platform'] == ACCELERATOR_NAME)]['value'].iloc[0]),
        100.0 - float(fig11c[(fig11c['metric'] == 'Redundant reads') & (fig11c['platform'] == ACCELERATOR_NAME)]['value'].iloc[0]),
    ]
    bars = ax.barh(reduction_labels, reduction_values, color=[PALETTE['accent'], PALETTE['green'], PALETTE['morph']], edgecolor='#22313F', linewidth=0.6)
    ax.axvline(0.0, color='#4A4E69', linewidth=1.0)
    ax.set_title('(d) Reduction summary and handoff context')
    ax.set_xlabel('Relative change vs Multi-core (%)')
    span = max(abs(min(reduction_values)), abs(max(reduction_values)))
    ax.set_xlim(-max(5.0, span * 1.25), max(40.0, span * 1.25))
    for bar, value in zip(bars, reduction_values):
        offset = 1.0 if value >= 0 else -1.0
        ha = 'left' if value >= 0 else 'right'
        ax.text(value + offset, bar.get_y() + bar.get_height() / 2.0, f'{value:.1f}%', va='center', ha=ha, fontsize=9)
    transition_rows = _transition_metric(fig12)
    transition_rows = transition_rows[(transition_rows['platform'] == ACCELERATOR_NAME) & (transition_rows['scenario'] != 'Avg')]
    recovery_avg = float(fig12[(fig12['platform'] == ACCELERATOR_NAME) & (fig12['scenario'] == 'Avg') & (fig12['metric'] == 'Recovery Time (ms)')]['value'].iloc[0])
    if not transition_rows.empty:
        fastest = transition_rows.nsmallest(1, 'value').iloc[0]
        slowest = transition_rows.nlargest(1, 'value').iloc[0]
        note = f'Fastest handoff: {fastest["scenario"]} = {float(fastest["value"]):.3f} ms\nSlowest handoff: {slowest["scenario"]} = {float(slowest["value"]):.3f} ms\nAvg recovery: {recovery_avg:.2f} ms'
        ax.text(0.98, 0.06, note, transform=ax.transAxes, ha='right', va='bottom', fontsize=8.8, bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#B5C0D0'))
    fig.suptitle(FIGURE_AUDIT['fig11']['title'], y=0.99, fontsize=15)
    _save(fig, out_dir / 'fig11.png')

def plot_fig12(df: pd.DataFrame, out_dir: Path) -> None:
    use_style()
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 9.6))
    transition = _transition_metric(df)
    scenario_order = [name for name in transition['scenario'].unique().tolist() if name != 'Avg']
    platforms = ['CPU', 'GPU', ACCELERATOR_NAME]

    ax = axes[0, 0]
    ax.set_facecolor(PALETTE['panel_bg'])
    width = 0.24
    x = np.arange(len(scenario_order))
    for idx, platform in enumerate(platforms):
        part = transition[(transition['platform'] == platform) & (transition['scenario'].isin(scenario_order))]
        part = part.set_index('scenario').reindex(scenario_order).reset_index()
        ax.bar(x + (idx - 1) * width, part['value'], width=width, color=[PALETTE['baseline'], PALETTE['gold'], PALETTE['morph']][idx], label=platform)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_order, rotation=25)
    ax.set_ylabel('Transition overhead (ms)')
    ax.set_title('(a) Per-scenario transition overhead')
    ax.legend(fontsize=8.6, ncol=3, loc='upper right')

    ax = axes[0, 1]
    ax.set_facecolor(PALETTE['panel_alt'])
    energy_cpu = df[(df['platform'] == ACCELERATOR_NAME) & (df['metric'] == 'Energy vs CPU (%)') & (df['scenario'].isin(scenario_order))]
    energy_cpu = energy_cpu.set_index('scenario').reindex(scenario_order).reset_index()
    energy_gpu = df[(df['platform'] == ACCELERATOR_NAME) & (df['metric'] == 'Energy vs GPU (%)') & (df['scenario'].isin(scenario_order))]
    energy_gpu = energy_gpu.set_index('scenario').reindex(scenario_order).reset_index()
    ax.plot(scenario_order, energy_cpu['value'], marker='o', linewidth=2.2, color=PALETTE['green'], label='Energy / CPU (%)')
    if not energy_gpu.empty and energy_gpu['value'].notna().any():
        ax.plot(scenario_order, energy_gpu['value'], marker='s', linewidth=2.2, color=PALETTE['accent'], label='Energy / GPU (%)')
    ax.set_ylabel('Energy ratio (%)')
    ax.set_title('(b) Energy ratio across handoffs')
    ax.tick_params(axis='x', rotation=25)
    ax.legend(fontsize=8.8, loc='upper right')

    ax = axes[1, 0]
    ax.set_facecolor(PALETTE['soft_bg'])
    recovery = df[(df['platform'] == ACCELERATOR_NAME) & (df['metric'] == 'Recovery Time (ms)') & (df['scenario'].isin(scenario_order))]
    recovery = recovery.set_index('scenario').reindex(scenario_order).reset_index()
    bars = ax.bar(recovery['scenario'], recovery['value'], color=PALETTE['purple'], width=0.62)
    ax.set_ylabel('Recovery time (ms)')
    ax.set_title('(c) Recovery time after transition')
    ax.tick_params(axis='x', rotation=25)
    for bar, value in zip(bars, recovery['value']):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + max(float(recovery['value'].max()) * 0.03, 0.01), f'{value:.2f}', ha='center', va='bottom', fontsize=8.8)

    ax = axes[1, 1]
    ax.set_facecolor('#F7F0FF')
    avg_transition = transition[transition['scenario'] == 'Avg'].set_index('platform').reindex(platforms)
    avg_energy_cpu = float(df[(df['platform'] == ACCELERATOR_NAME) & (df['scenario'] == 'Avg') & (df['metric'] == 'Energy vs CPU (%)')]['value'].iloc[0])
    avg_energy_gpu_rows = df[(df['platform'] == ACCELERATOR_NAME) & (df['scenario'] == 'Avg') & (df['metric'] == 'Energy vs GPU (%)')]
    avg_energy_gpu = float(avg_energy_gpu_rows['value'].iloc[0]) if not avg_energy_gpu_rows.empty else np.nan
    avg_recovery = float(df[(df['platform'] == ACCELERATOR_NAME) & (df['scenario'] == 'Avg') & (df['metric'] == 'Recovery Time (ms)')]['value'].iloc[0])
    labels = ['CPU', 'GPU', ACCELERATOR_NAME, 'Energy/CPU', 'Energy/GPU', 'Recovery']
    values = [float(avg_transition.loc['CPU', 'value']), float(avg_transition.loc['GPU', 'value']), float(avg_transition.loc[ACCELERATOR_NAME, 'value']), avg_energy_cpu, avg_energy_gpu if not np.isnan(avg_energy_gpu) else 0.0, avg_recovery]
    bars = ax.bar(labels, values, color=[PALETTE['baseline'], PALETTE['gold'], PALETTE['morph'], PALETTE['green'], PALETTE['accent'], PALETTE['purple']], width=0.62)
    ax.set_title('(d) Average cross-platform summary')
    ax.set_ylabel('Mixed-unit summary')
    ax.tick_params(axis='x', rotation=20)
    top = max(values)
    for bar, label, value in zip(bars, labels, values):
        suffix = ' ms' if label in {'CPU', 'GPU', ACCELERATOR_NAME, 'Recovery'} else '%'
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + max(top * 0.02, 0.02), f'{value:.2f}{suffix}', ha='center', va='bottom', fontsize=8.6)
    fig.suptitle(FIGURE_AUDIT['fig12_reconfig']['title'], y=1.03, fontsize=15)
    _save(fig, out_dir / 'fig12_reconfig.png')

def figure_audit_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for name, spec in FIGURE_AUDIT.items():
        rows.append({'figure_id': name, 'title': spec['title'], 'x_label': spec['x_label'], 'y_label': spec['y_label'], 'legend': '; '.join(spec['legend']), 'colors': '; '.join(spec['colors']), 'layout': spec['layout']})
    return rows
