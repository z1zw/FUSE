from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from morphaccel.sosp_plus_viz import plot_fig18, plot_fig19, plot_fig20, plot_fig21
from morphaccel.sosp_viz import plot_fig13, plot_fig14, plot_fig15, plot_fig16, plot_fig17
from morphaccel.viz import figure_audit_rows, plot_fig10, plot_fig11, plot_fig12, plot_fig4, plot_fig5, plot_fig6, plot_fig7, plot_fig8, plot_fig9

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, default=Path('results'))
    parser.add_argument('--output-dir', type=Path, default=Path('figures_plus'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_fig4(pd.read_csv(args.input_dir / 'fig4_latency.csv'), args.output_dir)
    plot_fig5(pd.read_csv(args.input_dir / 'fig5_energy.csv'), args.output_dir)
    plot_fig6(pd.read_csv(args.input_dir / 'fig6_mixed.csv'), args.output_dir)
    plot_fig7(pd.read_csv(args.input_dir / 'fig7_area_power.csv'), args.output_dir)
    plot_fig8(pd.read_csv(args.input_dir / 'fig8a_reuse.csv'), pd.read_csv(args.input_dir / 'fig8b_fanout.csv'), pd.read_csv(args.input_dir / 'fig8c_pool.csv'), args.output_dir)
    plot_fig9(pd.read_csv(args.input_dir / 'fig9_ablation.csv'), args.output_dir)
    plot_fig10(pd.read_csv(args.input_dir / 'fig10_memory.csv'), args.output_dir)
    plot_fig11(pd.read_csv(args.input_dir / 'fig11a_sched.csv'), pd.read_csv(args.input_dir / 'fig11b_switch.csv'), pd.read_csv(args.input_dir / 'fig11c_summary.csv'), pd.read_csv(args.input_dir / 'fig12_reconfig.csv'), args.output_dir)
    plot_fig12(pd.read_csv(args.input_dir / 'fig12_reconfig.csv'), args.output_dir)
    plot_fig13(pd.read_csv(args.input_dir / 'fig13_security.csv'), args.output_dir)
    plot_fig14(pd.read_csv(args.input_dir / 'fig14_llm_stability.csv'), args.output_dir)
    plot_fig15(pd.read_csv(args.input_dir / 'fig15_secure_tradeoff.csv'), args.output_dir)
    plot_fig16(pd.read_csv(args.input_dir / 'fig16_case_studies.csv'), args.output_dir)
    plot_fig17(pd.read_csv(args.input_dir / 'fig17_ablation_plus.csv'), args.output_dir)
    plot_fig18(pd.read_csv(args.input_dir / 'fig18_service_breakdown.csv'), args.output_dir)
    plot_fig19(pd.read_csv(args.input_dir / 'fig19_load_sweep.csv'), args.output_dir)
    plot_fig20(pd.read_csv(args.input_dir / 'fig20_extended_compare.csv'), args.output_dir)
    plot_fig21(pd.read_csv(args.input_dir / 'fig21_secure_components.csv'), args.output_dir)
    audit = pd.DataFrame(figure_audit_rows())
    extra = [{'figure_id': 'fig13_security', 'title': 'Fig. 13. Security attacks and aggregate resilience', 'x_label': 'Attack class', 'y_label': 'Attack accuracy / security score', 'legend': 'Deployment modes', 'colors': 'mode palette', 'layout': '1x2 panels'}, {'figure_id': 'fig14_llm_stability', 'title': 'Fig. 14. Stability and determinism', 'x_label': 'Mode', 'y_label': 'Variance / verifier score', 'legend': 'Mode palette', 'colors': 'mode palette', 'layout': '2x2 panels'}, {'figure_id': 'fig15_secure_tradeoff', 'title': 'Fig. 15. Security-performance trade space', 'x_label': 'Latency ratio', 'y_label': 'Security score', 'legend': 'Bubble size = determinism', 'colors': 'mode palette', 'layout': 'single scatter'}, {'figure_id': 'fig16_case_studies', 'title': 'Fig. 16. Mixed-domain case studies', 'x_label': 'Case or mode', 'y_label': 'Latency / throughput / energy', 'legend': 'Mode palette', 'colors': 'mode palette', 'layout': 'multi-panel'}, {'figure_id': 'fig17_ablation_plus', 'title': 'Fig. 17. Canonicalization and scheduling ablations', 'x_label': 'Workload', 'y_label': 'Configuration', 'legend': 'Heatmap colorbar', 'colors': 'heatmaps', 'layout': '1x3 panels'}, {'figure_id': 'fig18_service_breakdown', 'title': 'Fig. 18. Service-stage decomposition', 'x_label': 'Latency share', 'y_label': 'Case', 'legend': 'Stage palette', 'colors': 'tab20', 'layout': '1xN panels'}, {'figure_id': 'fig19_load_sweep', 'title': 'Fig. 19. Load-sweep stress test', 'x_label': 'Load factor', 'y_label': 'p99 latency (ms)', 'legend': 'Mode palette', 'colors': 'mode palette', 'layout': 'grid panels'}, {'figure_id': 'fig20_extended_compare', 'title': 'Fig. 20. Extended comparison', 'x_label': 'Mode', 'y_label': 'Case', 'legend': 'Heatmap colorbar', 'colors': 'heatmaps', 'layout': '1x2 panels'}, {'figure_id': 'fig21_secure_components', 'title': 'Fig. 21. Secure-Elastic component ablations', 'x_label': 'Workload', 'y_label': 'Configuration', 'legend': 'Heatmap colorbar', 'colors': 'heatmaps', 'layout': '1x3 panels'}]
    extra_df = pd.DataFrame(extra)
    extra_df['input_dir'] = str(args.input_dir)
    extra_df['output_dir'] = str(args.output_dir)
    audit['input_dir'] = str(args.input_dir)
    audit['output_dir'] = str(args.output_dir)
    combined = pd.concat([audit, extra_df], ignore_index=True)
    combined['png_path'] = combined['figure_id'].map(lambda x: str(args.output_dir / f'{x}.png'))
    combined['png_bytes'] = combined['png_path'].map(lambda x: Path(x).stat().st_size if Path(x).exists() else 0)
    combined.to_csv(args.output_dir / 'figure_audit.csv', index=False)
    (args.output_dir / 'figure_audit.json').write_text(json.dumps(combined.to_dict(orient='records'), indent=2), encoding='utf-8')
if __name__ == '__main__':
    main()
