from __future__ import annotations
from pathlib import Path
import pandas as pd
from .branding import ACCELERATOR_NAME

def build_cn_report(results_dir: Path) -> str:
    results_dir = Path(results_dir)
    fig13 = pd.read_csv(results_dir / 'fig13_security.csv')
    fig15 = pd.read_csv(results_dir / 'fig15_secure_tradeoff.csv')
    fig16 = pd.read_csv(results_dir / 'fig16_case_studies.csv')
    fig18 = pd.read_csv(results_dir / 'fig18_service_breakdown.csv')
    fig19 = pd.read_csv(results_dir / 'fig19_load_sweep.csv')
    fig20 = pd.read_csv(results_dir / 'fig20_extended_compare.csv')
    fig21 = pd.read_csv(results_dir / 'fig21_secure_components.csv')
    manifest = pd.read_csv(results_dir / 'case_manifest.csv')
    security_rank = fig13.groupby('mode', as_index=False)['security_score'].mean().sort_values('security_score', ascending=False)
    trade_rank = fig15.sort_values('balanced_score', ascending=False)
    case_summary = fig20.groupby('mode', as_index=False).agg(latency_gain_vs_multicore=('latency_gain_vs_multicore', 'mean'), energy_gain_vs_multicore=('energy_gain_vs_multicore', 'mean'), throughput_gain_vs_multicore=('throughput_gain_vs_multicore', 'mean'), security_score=('security_score', 'mean'), balanced_score=('balanced_score', 'mean')).sort_values('balanced_score', ascending=False)
    peak = fig19[fig19['load_factor'] == fig19['load_factor'].max()]
    peak_multicore = peak[peak['mode'] == 'Multi-core'].set_index('case')
    narrative_rows = []
    for case in manifest.to_dict(orient='records'):
        case_name = case['case']
        secure = fig16[(fig16['case'] == case_name) & (fig16['mode'] == 'Secure-Elastic')]
        multicore = fig16[(fig16['case'] == case_name) & (fig16['mode'] == 'Multi-core')]
        fuse = fig16[(fig16['case'] == case_name) & (fig16['mode'] == ACCELERATOR_NAME)]
        if secure.empty or multicore.empty or fuse.empty:
            continue
        secure_row = secure.iloc[0]
        multicore_row = multicore.iloc[0]
        fuse_row = fuse.iloc[0]
        stage_part = fig18[(fig18['case'] == case_name) & (fig18['mode'] == 'Secure-Elastic')].sort_values('share', ascending=False)
        hot_stage = stage_part.iloc[0]['stage'] if not stage_part.empty else '未知'
        peak_secure = peak[(peak['case'] == case_name) & (peak['mode'] == 'Secure-Elastic')]
        peak_secure_row = peak_secure.iloc[0] if not peak_secure.empty else None
        peak_gain = float(peak_multicore.loc[case_name]['p99_ms'] / max(float(peak_secure_row['p99_ms']) if peak_secure_row is not None else float(secure_row['p99_ms']), 1e-09)) if case_name in peak_multicore.index else 0.0
        narrative_rows.append({'case': case_name, 'storyline': case['storyline'], 'focus': case['focus'], 'slo_ms': float(case['slo_ms']), 'secure_p99': float(secure_row['p99_ms']), 'multicore_p99': float(multicore_row['p99_ms']), 'fuse_p99': float(fuse_row['p99_ms']), 'secure_vs_multicore': float(multicore_row['p99_ms'] / max(secure_row['p99_ms'], 1e-09)), 'secure_vs_fuse': float(fuse_row['p99_ms'] / max(secure_row['p99_ms'], 1e-09)), 'secure_energy_gain': float(multicore_row['energy_per_request_nj'] / max(secure_row['energy_per_request_nj'], 1e-09)), 'secure_throughput_gain': float(secure_row['throughput_tps'] / max(multicore_row['throughput_tps'], 1e-09)), 'secure_slo_miss': float(secure_row['slo_miss_ratio']), 'peak_gain': peak_gain, 'hot_stage': hot_stage})
    narrative = pd.DataFrame(narrative_rows).sort_values('secure_vs_multicore', ascending=False)
    component_rank = fig21.groupby('configuration', as_index=False)['balance_score'].mean().sort_values('balance_score', ascending=False)
    lines = []
    lines.append(f'# {ACCELERATOR_NAME} 中文实验报告')
    lines.append('')
    lines.append('## 1. 总结')
    lines.append('')
    lines.append(f"- 当前平均安全分数最高的模式是 **{security_rank.iloc[0]['mode']}**，平均安全分数为 **{security_rank.iloc[0]['security_score']:.3f}**。")
    lines.append(f"- 当前综合平衡分数最高的模式是 **{trade_rank.iloc[0]['mode']}**，平衡分数为 **{trade_rank.iloc[0]['balanced_score']:.3f}**。")
    secure_case = case_summary[case_summary['mode'] == 'Secure-Elastic'].iloc[0]
    lines.append(f"- Secure-Elastic 相对 Multi-core 的跨案例平均时延增益为 **{secure_case['latency_gain_vs_multicore']:.2f}x**，能耗增益为 **{secure_case['energy_gain_vs_multicore']:.2f}x**，吞吐增益为 **{secure_case['throughput_gain_vs_multicore']:.2f}x**。")
    lines.append('')
    lines.append('## 2. Secure-Elastic 组件观察')
    lines.append('')
    top_components = '、'.join(component_rank.head(3)['configuration'].tolist())
    lines.append(f'从组件消融结果看，综合表现最好的配置主要集中在：{top_components}。这说明 bucketization、family pinning、RFU masking、epoch rotation 和 prefetch 的组合效果，决定了 Secure-Elastic 与其他基线之间的差距。')
    lines.append('')
    lines.append('## 3. 端到端案例')
    lines.append('')
    for row in narrative.to_dict(orient='records'):
        lines.append(f"### {row['case']}")
        lines.append('')
        lines.append(f"- 场景叙事：{row['storyline']}")
        lines.append(f"- 关注点：{row['focus']}")
        lines.append(f"- SLO：{row['slo_ms']:.1f} ms")
        lines.append(f"- Secure-Elastic p99：{row['secure_p99']:.2f} ms，Multi-core p99：{row['multicore_p99']:.2f} ms，{ACCELERATOR_NAME} p99：{row['fuse_p99']:.2f} ms")
        lines.append(f"- Secure-Elastic 相对 Multi-core 的 p99 增益：**{row['secure_vs_multicore']:.2f}x**")
        lines.append(f"- Secure-Elastic 相对 Multi-core 的吞吐增益：**{row['secure_throughput_gain']:.2f}x**")
        lines.append(f"- Secure-Elastic 相对 Multi-core 的能耗增益：**{row['secure_energy_gain']:.2f}x**")
        lines.append(f"- 峰值负载下的 p99 增益：**{row['peak_gain']:.2f}x**")
        lines.append(f"- 当前主导阶段：**{row['hot_stage']}**")
        lines.append(f"- 当前 SLO miss ratio：**{row['secure_slo_miss']:.3f}**")
        lines.append('')
    lines.append('## 4. 跨模式汇总')
    lines.append('')
    lines.append('| 模式 | 平均时延增益 | 平均能耗增益 | 平均吞吐增益 | 平均安全分数 | 平均平衡分数 |')
    lines.append('| --- | ---: | ---: | ---: | ---: | ---: |')
    for row in case_summary.to_dict(orient='records'):
        lines.append(f"| {row['mode']} | {row['latency_gain_vs_multicore']:.2f} | {row['energy_gain_vs_multicore']:.2f} | {row['throughput_gain_vs_multicore']:.2f} | {row['security_score']:.2f} | {row['balanced_score']:.2f} |")
    lines.append('')
    lines.append('## 5. 正文建议')
    lines.append('')
    lines.append('1. 强调 FUSE 的收益来自端到端服务链，而不是单个 kernel 的局部最优。')
    lines.append('2. Secure-Elastic 应被表述为面向安全服务尾延迟的执行模式，而不是 Secure-CT 的简单轻量变体。')
    lines.append('3. 跨案例结论应写成 workload-sensitive，而不是写成单一模式在所有任务上全面领先。')
    lines.append('')
    return '\n'.join(lines) + '\n'

def write_cn_report(results_dir: Path, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_cn_report(results_dir), encoding='utf-8')
    return output_path
