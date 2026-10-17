"""Draw the reconfiguration figure from the run the artifact reproduces.

Provenance. An earlier version of this script drew the panels from figures
typed in by hand (0.031 ms, 2.31 ms, 1.56 ms, 31.7x, 12.7x) because no file in
the repository carried them. That is the wrong way round: a number the artifact
cannot reproduce cannot be defended at evaluation, whatever its origin. The
figure now reads ``results/fig12_reconfig.csv``, which ``_raw_fig12`` writes
from the C++ engine run that ``scripts/verify_cpp_repro.py`` reproduces, and
the body quotes the same rows. The CPU and GPU transitions are wall-clock
timings of the host (``hardware_timer``); the FUSE transition is simulated.

    python scripts/plot_reconfig.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from plot_bold_figures import (BLUE, GREY, LAB, LEG_DENSE, TICK_DENSE, VERM,
                               bars, dress, save, ylab)

SRC = ROOT / 'results' / 'fig12_reconfig.csv'


def averages() -> dict:
    d = pd.read_csv(SRC)
    avg = d[d['scenario'] == 'Avg']
    pick = lambda plat, met: float(avg[(avg['platform'] == plat) & (avg['metric'] == met)]['value'].iloc[0])
    return {
        'cpu_ms': pick('CPU', 'Transition Overhead (ms)'),
        'gpu_ms': pick('GPU', 'Transition Overhead (ms)'),
        'fuse_ms': pick('FUSE', 'Transition Overhead (ms)'),
        # The CSV holds FUSE energy as a percentage of each baseline's; the
        # panel shows the reciprocal, how many times less a switch costs.
        'energy_vs_cpu': 100.0 / pick('FUSE', 'Energy vs CPU (%)'),
        'energy_vs_gpu': 100.0 / pick('FUSE', 'Energy vs GPU (%)'),
    }


def panel_overhead(ax, a):
    """Transition overhead per platform, on a log axis."""
    items = [('CPU', a['cpu_ms'], GREY), ('GPU', a['gpu_ms'], VERM), ('FUSE', a['fuse_ms'], BLUE)]
    names = [n for n, _, _ in items]
    vals = [v for _, v, _ in items]
    cols = [c for _, _, c in items]
    bars(ax, range(len(vals)), vals, cols)
    ax.set_yscale('log')
    ax.set_ylim(0.01, 10)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, fontsize=TICK_DENSE)
    ylab(ax, 'Transition overhead (ms)', LAB - 7)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.35, f'{v:.3g}', ha='center', va='bottom',
                fontsize=LEG_DENSE, fontweight='bold')
    # The two ratios the body quotes, computed from the same rows.
    ax.annotate(rf'${a["cpu_ms"] / a["fuse_ms"]:.0f}\times$', xy=(2, a['fuse_ms']), xytext=(0.55, 0.30),
                fontsize=LEG_DENSE, fontweight='bold', ha='center')
    ax.annotate(rf'${a["gpu_ms"] / a["fuse_ms"]:.0f}\times$', xy=(2, a['fuse_ms']), xytext=(1.55, 0.20),
                fontsize=LEG_DENSE, fontweight='bold', ha='center')
    dress(ax, TICK_DENSE)


def panel_energy(ax, a):
    """How many times less energy a switch costs on the fabric."""
    items = [('vs. CPU', a['energy_vs_cpu'], GREY), ('vs. GPU', a['energy_vs_gpu'], VERM)]
    names = [n for n, _, _ in items]
    vals = [v for _, v, _ in items]
    cols = [c for _, _, c in items]
    bars(ax, range(len(vals)), vals, cols, w=0.55)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, fontsize=TICK_DENSE)
    ax.set_ylim(0, max(vals) * 1.25)
    ylab(ax, 'Energy efficiency ($\\times$)', LAB - 7)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.03, f'{v:.1f}' + r'$\times$', ha='center', va='bottom',
                fontsize=LEG_DENSE, fontweight='bold')
    dress(ax, TICK_DENSE)


def main() -> None:
    a = averages()
    print('from results/fig12_reconfig.csv (Avg rows):')
    print(f"  transition  FUSE {a['fuse_ms']:.4f} ms  CPU {a['cpu_ms']:.3f} ms ({a['cpu_ms']/a['fuse_ms']:.1f}x)  "
          f"GPU {a['gpu_ms']:.3f} ms ({a['gpu_ms']/a['fuse_ms']:.1f}x)")
    print(f"  energy      {a['energy_vs_cpu']:.2f}x vs CPU, {a['energy_vs_gpu']:.2f}x vs GPU")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
    panel_overhead(axes[0], a)
    panel_energy(axes[1], a)
    for ax, tag in zip(axes, '(a) (b)'.split()):
        ax.set_title(tag, fontsize=TICK_DENSE, fontweight='bold', loc='left')
    fig.tight_layout()
    save(fig, ROOT / 'sosp', 'fig_reconfig')


if __name__ == '__main__':
    main()
