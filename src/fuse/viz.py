from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .branding import ACCELERATOR_NAME
PALETTE = {'baseline': '#34495E', 'morph': '#E76F51', 'accent': '#2878B5', 'green': '#2A9D8F', 'gold': '#F4A261', 'teal': '#1B998B', 'purple': '#7A5CFA', 'sand': '#E9C46A', 'cyan': '#48BFE3', 'gray': '#7D8597', 'rose': '#D1495B', 'indigo': '#5A189A', 'mint': '#80ED99', 'soft_bg': '#F5F7FB', 'panel_bg': '#EEF3FB', 'panel_alt': '#FFF6EC'}
FIGURE_AUDIT = {'fig4_latency': {'title': 'Fig. 4. Latency Against Domain-Specific Baselines', 'x_label': 'Input size or batch size', 'y_label': 'Latency (ms) / latency ratio', 'legend': ['Baseline', ACCELERATOR_NAME], 'colors': [PALETTE['baseline'], PALETTE['morph'], PALETTE['green'], PALETTE['gold']], 'layout': '1x4 panels'}, 'fig5_energy': {'title': 'Fig. 5. Energy Efficiency Across Baselines', 'x_label': 'Platform', 'y_label': 'Normalized energy efficiency (x)', 'legend': [ACCELERATOR_NAME, 'Software and hardware baselines'], 'colors': [PALETTE['baseline'], PALETTE['accent'], PALETTE['green'], PALETTE['morph'], PALETTE['purple'], PALETTE['teal'], PALETTE['gold']], 'layout': '1x3 panels'}, 'fig6_mixed': {'title': 'Fig. 6. Mixed-Trace End-to-End Performance', 'x_label': 'Interleave window (ms)', 'y_label': 'Makespan (ms), RFU utilization (%)', 'legend': ['Multi-core', ACCELERATOR_NAME, 'Multi-core utilization', f'{ACCELERATOR_NAME} utilization'], 'colors': [PALETTE['baseline'], PALETTE['morph'], PALETTE['accent'], PALETTE['green']], 'layout': 'dual-axis single panel'}, 'fig7_area_power': {'title': 'Fig. 7. Area and Power Composition', 'x_label': 'Share (%)', 'y_label': 'Module', 'legend': ['FVM', 'RFU', 'TARS', 'Others'], 'colors': [PALETTE['accent'], PALETTE['morph'], PALETTE['green'], PALETTE['gold']], 'layout': '1x2 panels'}, 'fig8_reuse': {'title': 'Fig. 8. Reuse Decomposition and RFU Fan-Out', 'x_label': 'Share (%) or reuse class', 'y_label': 'Share (%)', 'legend': ['Reuse composition', 'Cross-task fan-out', 'Template pool composition'], 'colors': [PALETTE['accent'], PALETTE['gold'], PALETTE['green'], PALETTE['morph'], PALETTE['sand']], 'layout': '1x3 panels'}, 'fig9_ablation': {'title': 'Fig. 9. Ablation Against a Degraded Design', 'x_label': 'Latency (ms) or throughput (ops/ms)', 'y_label': 'Configuration', 'legend': ['Degraded', ACCELERATOR_NAME], 'colors': [PALETTE['baseline'], PALETTE['morph']], 'layout': '1x3 panels'}, 'fig10_memory': {'title': 'Fig. 10. On-Chip Residency and Redundant Accesses', 'x_label': 'Batch size', 'y_label': 'Normalized value (ratio)', 'legend': ['Fully Separated', ACCELERATOR_NAME], 'colors': [PALETTE['baseline'], PALETTE['morph']], 'layout': '2x4 panels'}, 'fig11': {'title': 'Fig. 11. Cross-Task Scheduling Overhead Reduction', 'x_label': 'Top row: scheduling and reduction. Bottom row: state inheritance paths.', 'y_label': 'Match ratio (%), latency (ms), reduction (%), or schematic annotation', 'legend': ['ZKP task', 'CNN task', 'AES task', 'Shared state', 'Completed', 'Buffer', 'Cache'], 'colors': [PALETTE['accent'], PALETTE['morph'], PALETTE['green'], PALETTE['gold']], 'layout': '2x3 panels'}, 'fig12_reconfig': {'title': 'Fig. 12. Reconfiguration Latency, Energy, and Recovery', 'x_label': 'Scenario or summary metric', 'y_label': 'Switch latency (ms), energy ratio (%), recovery time (ms)', 'legend': ['CPU', 'GPU', ACCELERATOR_NAME], 'colors': [PALETTE['baseline'], PALETTE['gold'], PALETTE['morph'], PALETTE['green'], PALETTE['accent'], PALETTE['purple']], 'layout': '1x2 panels'}}

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

def _circle_node(ax, x: float, y: float, label: str, color: str, *, radius: float=0.12, text_color: str='white') -> None:
    node = plt.Circle((x, y), radius, color=color, ec='#202124', lw=1.6)
    ax.add_patch(node)
    ax.text(x, y, label, ha='center', va='center', fontsize=10, color=text_color, weight='semibold')

def _edge(ax, x0: float, y0: float, x1: float, y1: float, *, label: str | None=None, dashed: bool=False) -> None:
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', lw=1.8, color='#202124', linestyle='--' if dashed else '-'))
    if label:
        ax.text((x0 + x1) / 2.0, (y0 + y1) / 2.0 + 0.05, label, ha='center', va='center', fontsize=8.8, weight='semibold')

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
    ax2.plot(x, df['baseline_util'] * 100.0, color=PALETTE['accent'], marker='o', linewidth=2.0, label='Multi-core utilization')
    ax2.plot(x, df['morph_util'] * 100.0, color=PALETTE['green'], marker='s', linewidth=2.0, label=f'{ACCELERATOR_NAME} utilization')
    ax2.set_ylabel('RFU utilization (%)')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', ncol=2, fontsize=8)
    fig.suptitle(FIGURE_AUDIT['fig6_mixed']['title'], y=1.02, fontsize=15)
    _save(fig, out_dir / 'fig6_mixed.png')

def plot_fig7(df: pd.DataFrame, out_dir: Path) -> None:
    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    for ax, category, title in zip(axes, ['area', 'power'], ['Area breakdown', 'Peak power breakdown']):
        part = df[df['category'] == category].copy()
        bars = ax.barh(part['module'], part['share'], color=[PALETTE['accent'], PALETTE['morph'], PALETTE['green'], PALETTE['gold']][:len(part)])
        ax.set_xlabel('Share (%)')
        ax.set_title(title)
        for bar, value in zip(bars, part['share']):
            ax.text(value + 0.6, bar.get_y() + bar.get_height() / 2.0, f'{value:.1f}%', va='center', fontsize=9)
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
    fig = plt.figure(figsize=(15.8, 10.6))
    gs = fig.add_gridspec(2, 3, hspace=0.48, wspace=0.34)
    axes = [fig.add_subplot(gs[row, col]) for row in range(2) for col in range(3)]
    axes[0].set_facecolor(PALETTE['panel_bg'])
    bars = axes[0].barh(fig11a['bucket'], fig11a['value'], color=[PALETTE['accent'], PALETTE['gold'], PALETTE['morph']], edgecolor='#22313F', linewidth=0.6)
    axes[0].set_title('(a) TARS fragment scheduling', pad=8)
    axes[0].set_xlabel('Fragment Match Ratio (%)')
    axes[0].set_xlim(0.0, max(80.0, float(fig11a['value'].max()) * 1.18))
    for bar, value in zip(bars, fig11a['value']):
        axes[0].text(value + 1.2, bar.get_y() + bar.get_height() / 2.0, f'{value:.1f}%', va='center', fontsize=9)
    axes[1].set_facecolor(PALETTE['panel_alt'])
    lat = fig11b[fig11b['metric'] == 'Scheduling Latency (ms)']
    idle = fig11b[fig11b['metric'] == 'Idle Ratio (%)']
    x = np.arange(len(lat))
    width = 0.34
    bars_lat = axes[1].bar(x - width / 2.0, lat['value'], width=width, color=PALETTE['accent'], edgecolor='#22313F', linewidth=0.6, label='Scheduling Latency (ms)')
    axr = axes[1].twinx()
    bars_idle = axr.bar(x + width / 2.0, idle['value'], width=width, color=PALETTE['gold'], edgecolor='#22313F', linewidth=0.6, label='Idle Ratio (%)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(lat['platform'])
    axes[1].set_title('(b) Task switch latency and idle ratio', pad=8)
    axes[1].set_ylabel('Value')
    axr.set_ylabel('')
    for bar in bars_lat:
        value = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, value + max(0.08, value * 0.04), f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    for bar in bars_idle:
        value = bar.get_height()
        axr.text(bar.get_x() + bar.get_width() / 2.0, value + max(0.5, value * 0.02), f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
    handles1, labels1 = axes[1].get_legend_handles_labels()
    handles2, labels2 = axr.get_legend_handles_labels()
    axes[1].legend(handles1 + handles2, labels1 + labels2, loc='upper left', fontsize=8.8)
    axes[2].set_facecolor('#F7F0FF')
    reduction_rows = [('Latency Reduction', 100.0 - float(fig11c[(fig11c['metric'] == 'Scheduling latency') & (fig11c['platform'] == ACCELERATOR_NAME)]['value'].iloc[0]), PALETTE['accent']), ('Idle Ratio Reduction', 100.0 - float(fig11c[(fig11c['metric'] == 'Idle ratio') & (fig11c['platform'] == ACCELERATOR_NAME)]['value'].iloc[0]), PALETTE['green']), ('Buffer Ops Reduction', 100.0 - float(fig11c[(fig11c['metric'] == 'Buffer ops') & (fig11c['platform'] == ACCELERATOR_NAME)]['value'].iloc[0]), PALETTE['morph'])]
    labels = [row[0] for row in reduction_rows]
    values = [row[1] for row in reduction_rows]
    colors = [row[2] for row in reduction_rows]
    bars = axes[2].barh(labels, values, color=colors, edgecolor='#22313F', linewidth=0.6)
    axes[2].set_title('(c) Overall reduction summary', pad=8)
    axes[2].set_xlabel('Reduction (%)')
    axes[2].set_xlim(0.0, max(85.0, max(values) * 1.16))
    for bar, value in zip(bars, values):
        axes[2].text(value + 1.2, bar.get_y() + bar.get_height() / 2.0, f'{value:.1f}%', va='center', fontsize=9)
    for ax in axes[3:]:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_facecolor(PALETTE['soft_bg'])
        for spine in ax.spines.values():
            spine.set_visible(False)
    zkp_cnn = fig12[(fig12['scenario'] == 'ZKP->CNN') & (fig12['platform'] == ACCELERATOR_NAME) & (fig12['metric'] == 'Switch Latency (ms)')]
    cnn_aes = fig12[(fig12['scenario'] == 'CNN->AES') & (fig12['platform'] == ACCELERATOR_NAME) & (fig12['metric'] == 'Switch Latency (ms)')]
    recovery_avg = fig12[(fig12['scenario'] == 'Avg') & (fig12['platform'] == ACCELERATOR_NAME) & (fig12['metric'] == 'Recovery Time (ms)')]
    buffer_reduction = 100.0 - float(fig11c[(fig11c['metric'] == 'Buffer ops') & (fig11c['platform'] == ACCELERATOR_NAME)]['value'].iloc[0])
    axes[3].set_title('(d) ZKP -> CNN switch', fontsize=12.5, weight='semibold')
    _circle_node(axes[3], 0.18, 0.62, 'ZKP', PALETTE['gray'])
    _circle_node(axes[3], 0.78, 0.62, 'CNN', PALETTE['green'])
    _circle_node(axes[3], 0.48, 0.28, 'Shared\nState', PALETTE['purple'], text_color='white')
    _edge(axes[3], 0.26, 0.55, 0.42, 0.34, dashed=True)
    _edge(axes[3], 0.54, 0.34, 0.72, 0.55)
    axes[3].text(0.5, 0.08, 'Prefetch: 2.1ms earlier', ha='center', va='center', fontsize=9.3, bbox=dict(boxstyle='round,pad=0.24', fc='#FFF6D8', ec='#404040'))
    axes[4].set_title('(e) CNN -> AES switch', fontsize=12.5, weight='semibold')
    _circle_node(axes[4], 0.18, 0.62, 'CNN', PALETTE['gray'])
    _circle_node(axes[4], 0.78, 0.62, 'AES', PALETTE['morph'])
    _circle_node(axes[4], 0.48, 0.28, 'Shared\nState', PALETTE['purple'], text_color='white')
    _edge(axes[4], 0.26, 0.55, 0.42, 0.34, dashed=True)
    _edge(axes[4], 0.54, 0.34, 0.72, 0.55)
    axes[4].text(0.5, 0.08, 'Pipeline stalls: -28.7%', ha='center', va='center', fontsize=9.3, bbox=dict(boxstyle='round,pad=0.24', fc='#FFF6D8', ec='#404040'))
    axes[5].set_title('(f) Full state inheritance', fontsize=12.5, weight='semibold')
    _circle_node(axes[5], 0.2, 0.72, 'ZKP', PALETTE['accent'])
    _circle_node(axes[5], 0.5, 0.82, 'CNN', PALETTE['green'])
    _circle_node(axes[5], 0.8, 0.72, 'AES', PALETTE['morph'])
    _circle_node(axes[5], 0.5, 0.46, 'State', PALETTE['purple'], radius=0.13)
    _circle_node(axes[5], 0.32, 0.18, 'Buffer', PALETTE['gold'], radius=0.11, text_color='#202124')
    _circle_node(axes[5], 0.68, 0.18, 'Cache', PALETTE['cyan'], radius=0.11, text_color='#202124')
    _edge(axes[5], 0.27, 0.67, 0.44, 0.52, label='8.2%')
    _edge(axes[5], 0.5, 0.74, 0.5, 0.59)
    _edge(axes[5], 0.73, 0.67, 0.56, 0.52, label='6.7%')
    _edge(axes[5], 0.44, 0.36, 0.36, 0.25)
    _edge(axes[5], 0.56, 0.36, 0.64, 0.25)
    axes[5].text(0.5, 0.06, f'Total buffer reduction: {buffer_reduction:.1f}%', ha='center', va='center', fontsize=9.3, bbox=dict(boxstyle='round,pad=0.24', fc='#D9F8DE', ec='#2E8B57'))
    if not zkp_cnn.empty:
        axes[3].text(0.18, 0.9, f"{float(zkp_cnn['value'].iloc[0]):.1f} ms", fontsize=8.8, color='#3D405B')
    if not cnn_aes.empty:
        axes[4].text(0.18, 0.9, f"{float(cnn_aes['value'].iloc[0]):.1f} ms", fontsize=8.8, color='#3D405B')
    if not recovery_avg.empty:
        axes[5].text(0.5, 0.92, f"Recovery < {float(recovery_avg['value'].iloc[0]):.2f} ms", ha='center', fontsize=8.8, color='#3D405B')
    fig.legend(handles=[mpl.patches.Patch(facecolor=PALETTE['accent'], edgecolor='#202124', label='ZKP Task'), mpl.patches.Patch(facecolor=PALETTE['green'], edgecolor='#202124', label='CNN Task'), mpl.patches.Patch(facecolor=PALETTE['morph'], edgecolor='#202124', label='AES Task'), mpl.patches.Patch(facecolor=PALETTE['purple'], edgecolor='#202124', label='Shared State'), mpl.lines.Line2D([0], [0], color='#202124', linestyle='--', label='Completed'), mpl.patches.Patch(facecolor=PALETTE['gold'], edgecolor='#202124', label='Buffer'), mpl.patches.Patch(facecolor=PALETTE['cyan'], edgecolor='#202124', label='Cache')], loc='lower center', ncol=7, bbox_to_anchor=(0.5, -0.01), fontsize=9.2, columnspacing=1.4, handlelength=1.8)
    fig.suptitle(FIGURE_AUDIT['fig11']['title'], y=0.98, fontsize=15)
    _save(fig, out_dir / 'fig11.png')

def plot_fig12(df: pd.DataFrame, out_dir: Path) -> None:
    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(15.6, 5.0))
    latency = df[df['metric'] == 'Switch Latency (ms)']
    scenarios = list(latency['scenario'].unique())
    platforms = ['CPU', 'GPU', ACCELERATOR_NAME]
    x = np.arange(len(scenarios))
    width = 0.24
    for idx, platform in enumerate(platforms):
        part = latency[latency['platform'] == platform]
        axes[0].bar(x + (idx - 1) * width, part['value'], width=width, color=[PALETTE['baseline'], PALETTE['gold'], PALETTE['morph']][idx], label=platform)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(scenarios, rotation=25)
    axes[0].set_ylabel('Switch latency (ms)')
    axes[0].set_title('Task-switch latency')
    axes[0].legend(fontsize=8)
    energy = df[(df['platform'] == ACCELERATOR_NAME) & df['metric'].isin(['Energy vs CPU (%)', 'Energy vs GPU (%)'])].groupby('metric', as_index=False)['value'].mean()
    recovery = df[(df['platform'] == ACCELERATOR_NAME) & (df['metric'] == 'Recovery Time (ms)')].groupby('metric', as_index=False)['value'].mean()
    energy_x = np.arange(len(energy))
    energy_bars = axes[1].bar(energy_x, energy['value'], color=[PALETTE['green'], PALETTE['accent']][:len(energy)], width=0.55)
    axes[1].set_xticks(list(energy_x) + [len(energy_x)])
    axes[1].set_xticklabels(['Energy / CPU', 'Energy / GPU', 'Recovery'], rotation=20)
    axes[1].set_ylabel('Energy ratio (%)')
    axes[1].set_title('Energy ratio and recovery time')
    axr = axes[1].twinx()
    recovery_x = [len(energy_x)]
    recovery_bars = axr.bar(recovery_x, recovery['value'], color=[PALETTE['purple']], width=0.55)
    axr.set_ylabel('Recovery time (ms)')
    for bar, value in zip(energy_bars, energy['value']):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, value + max(1.0, value * 0.02), f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
    for bar, value in zip(recovery_bars, recovery['value']):
        axr.text(bar.get_x() + bar.get_width() / 2.0, value + max(0.2, value * 0.03), f'{value:.2f} ms', ha='center', va='bottom', fontsize=9)
    fig.suptitle(FIGURE_AUDIT['fig12_reconfig']['title'], y=1.03, fontsize=15)
    _save(fig, out_dir / 'fig12_reconfig.png')

def figure_audit_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for name, spec in FIGURE_AUDIT.items():
        rows.append({'figure_id': name, 'title': spec['title'], 'x_label': spec['x_label'], 'y_label': spec['y_label'], 'legend': '; '.join(spec['legend']), 'colors': '; '.join(spec['colors']), 'layout': spec['layout']})
    return rows
