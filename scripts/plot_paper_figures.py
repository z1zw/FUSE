from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from fuse.viz import figure_audit_rows, plot_fig10, plot_fig11, plot_fig12, plot_fig4, plot_fig5, plot_fig6, plot_fig7, plot_fig8, plot_fig9

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, default=Path('results'))
    parser.add_argument('--output-dir', type=Path, default=Path('figures'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_fig4(pd.read_csv(args.input_dir / 'fig4_latency.csv'), args.output_dir)
    plot_fig5(pd.read_csv(args.input_dir / 'fig5_energy.csv'), args.output_dir)
    plot_fig6(pd.read_csv(args.input_dir / 'fig6_mixed.csv'), args.output_dir)
    plot_fig7(pd.read_csv(args.input_dir / 'fig7_area_power.csv'), args.output_dir)
    plot_fig8(pd.read_csv(args.input_dir / 'fig8a_reuse.csv'), pd.read_csv(args.input_dir / 'fig8b_fanout.csv'), pd.read_csv(args.input_dir / 'fig8c_pool.csv'), args.output_dir)
    plot_fig9(pd.read_csv(args.input_dir / 'fig9_ablation.csv'), args.output_dir)
    plot_fig10(pd.read_csv(args.input_dir / 'fig10_memory.csv'), args.output_dir)
    fig11a = pd.read_csv(args.input_dir / 'fig11a_sched.csv')
    fig11b = pd.read_csv(args.input_dir / 'fig11b_switch.csv')
    fig11c = pd.read_csv(args.input_dir / 'fig11c_summary.csv')
    fig12 = pd.read_csv(args.input_dir / 'fig12_reconfig.csv')
    plot_fig11(fig11a, fig11b, fig11c, fig12, args.output_dir)
    plot_fig12(fig12, args.output_dir)
    audit = pd.DataFrame(figure_audit_rows())
    audit['input_dir'] = str(args.input_dir)
    audit['output_dir'] = str(args.output_dir)
    audit['png_path'] = audit['figure_id'].map(lambda x: str(args.output_dir / f'{x}.png'))
    audit['png_bytes'] = audit['png_path'].map(lambda x: Path(x).stat().st_size if Path(x).exists() else 0)
    audit.to_csv(args.output_dir / 'figure_audit.csv', index=False)
    (args.output_dir / 'figure_audit.json').write_text(json.dumps(audit.to_dict(orient='records'), indent=2), encoding='utf-8')
if __name__ == '__main__':
    main()
