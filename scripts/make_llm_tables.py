"""Turn the language-model results into the LaTeX tables and figures the paper includes.

Every number in the generated tables is read from the result CSVs, so the paper
cannot drift from the run that produced it.

    python scripts/make_llm_tables.py --zoo runs/llm_zoo --frontend runs/llm_frontend \
        --out-dir sosp
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _fmt(value: Any, digits: int = 2, dash: str = '--') -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return dash
    return f'{float(value):.{digits}f}'


def _pct(value: Any, digits: int = 1, dash: str = '--') -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return dash
    scaled = 100.0 * float(value)
    # a value that rounds to zero should not be printed as '-0.00'
    if abs(scaled) < 0.5 * 10 ** (-digits):
        scaled = 0.0
    return f'{scaled:.{digits}f}'


def _escape(text: str) -> str:
    return str(text).replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')


#: Each open-weight model carries its paper. The five hosted compilers of
#: Table~3 are commercial endpoints with no paper, so they carry none, and
#: the caption says what they are instead.
MODEL_CITES = {
    'GPT-2': 'radford2019gpt2',
    'Qwen2.5-0.5B': 'qwen2024qwen25',
    'Qwen2.5-7B': 'qwen2024qwen25',
    'TinyLlama-1.1B': 'zhang2024tinyllama',
    'OLMoE-1B-7B': 'muennighoff2024olmoe',
    'Mistral-7B': 'jiang2023mistral',
}


def _cited(label: str) -> str:
    """The label as the table prints it, with its citation where one exists."""
    key = MODEL_CITES.get(str(label).strip())
    text = _escape(label)
    return f'{text}~\\cite{{{key}}}' if key else text


def model_table(zoo: Path) -> str:
    """The zoo itself: what each model is, and what it does on the local GPU."""
    summary = pd.read_csv(zoo / 'llm_summary.csv')
    rows: List[str] = []
    for _, row in summary.sort_values('parameter_count').iterrows():
        arch_path = zoo / str(row['model']) / 'arch.json'
        arch: Dict[str, Any] = {}
        if arch_path.exists():
            import json

            arch = json.loads(arch_path.read_text(encoding='utf-8'))
        layers = arch.get('layers', '--')
        hidden = arch.get('hidden', '--')
        heads = arch.get('heads', '--')
        kv_heads = arch.get('kv_heads', '--')
        attention = f'{heads}/{kv_heads}'
        if arch.get('moe_experts'):
            ff = f"MoE {arch['moe_top_k']}/{arch['moe_experts']}"
        else:
            ff = f"dense {arch.get('ff_dim', '--')}"
        params_b = float(row['parameter_count']) / 1e9
        rows.append(
            ' & '.join(
                [
                    _cited(row['model_label']),
                    f'{params_b:.2f}',
                    str(layers),
                    str(hidden),
                    attention,
                    ff,
                    _fmt(row.get('native_prefill_ms'), 1),
                    _fmt(row.get('native_decode_ms_per_token'), 2),
                    _fmt(row.get('native_power_w'), 0),
                    _fmt(row.get('native_energy_j'), 1),
                ]
            )
            + r' \\'
        )
    body = '\n'.join(rows)
    return rf"""\begin{{table}}[!t]
\centering
\caption{{Language models evaluated on the local machine. Attention is reported as
query/key-value heads, so a smaller second number is grouped-query attention.
The last four columns are measured on an RTX 5080 through Hugging Face
\texttt{{transformers}}, averaged over the sampled GSM8K prompts. They describe
that software stack rather than the peak throughput of the device.}}
\label{{tab:llm-zoo}}
\small
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{|l|r|r|r|c|l|r|r|r|r|}}
\hline
\textbf{{Model}} & \textbf{{Params}} & \textbf{{Layers}} & \textbf{{$d$}} &
\textbf{{Heads}} & \textbf{{FFN}} &
\textbf{{Prefill}} & \textbf{{Decode}} & \textbf{{Power}} & \textbf{{Energy}} \\
 & (B) & & & (Q/KV) & & (ms) & (ms/tok) & (W) & (J/req) \\
\hline
{body}
\hline
\end{{tabular}}
\end{{table}}
"""


def results_table(zoo: Path) -> str:
    """FUSE against the baselines, per model."""
    summary = pd.read_csv(zoo / 'llm_summary.csv')
    rows: List[str] = []
    for _, row in summary.sort_values('parameter_count').iterrows():
        rows.append(
            ' & '.join(
                [
                    _cited(row['model_label']),
                    _fmt(row.get('fuse_total_ms'), 1),
                    _fmt(row.get('transformer_asic_latency_ratio'), 3),
                    _fmt(row.get('gpu_tensor_latency_ratio'), 3),
                    _fmt(row.get('multicore_latency_ratio'), 3),
                    _fmt(row.get('gpu_tensor_energy_gain'), 2),
                    _fmt(row.get('multicore_energy_gain'), 2),
                    _pct(row.get('fuse_rfu_utilization')),
                    _pct(row.get('fuse_structural_reuse_rate')),
                ]
            )
            + r' \\'
        )
    body = '\n'.join(rows)
    return rf"""\begin{{table}}[!t]
\centering
\caption{{\textbf{{FUSE on language-model inference.}} Latency is FUSE over
the baseline, so below $1.0$ is faster. Energy is the baseline over FUSE, so
above $1.0$ is cheaper. One request is a GSM8K prompt plus 64 generated
tokens.}}
\label{{tab:llm-results}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{|l|r|r|r|r|r|r|r|r|}}
\hline
\textbf{{Model}} & \textbf{{FUSE}} & \multicolumn{{3}}{{c|}}{{\textbf{{Latency ratio vs.}}}} &
\multicolumn{{2}}{{c|}}{{\textbf{{Energy gain vs.}}}} & \textbf{{RFU}} & \textbf{{Reuse}} \\
 & (ms) & T-ASIC & GPU-T & M-core & GPU-T & M-core & util.\ (\%) & (\%) \\
\hline
{body}
\hline
\end{{tabular}}
\end{{table}}
"""


def serving_table(zoo: Path) -> str:
    """The private-serving trace: what security costs, and what FUSE saves."""
    summary = pd.read_csv(zoo / 'llm_summary.csv')
    rows: List[str] = []
    for _, row in summary.sort_values('parameter_count').iterrows():
        rows.append(
            ' & '.join(
                [
                    _cited(row['model_label']),
                    _fmt(row.get('security_overhead_fuse'), 3),
                    _fmt(row.get('security_overhead_multicore'), 3),
                    _pct(row.get('serving_latency_reduction')),
                    _pct(row.get('serving_switch_reduction')),
                    _pct(row.get('serving_aes_latency_reduction')),
                    _pct(row.get('serving_zkp_latency_reduction')),
                    _fmt(row.get('serving_energy_gain'), 2),
                ]
            )
            + r' \\'
        )
    body = '\n'.join(rows)
    return rf"""\begin{{table}}[!t]
\centering
\caption{{\textbf{{Private-serving trace}} (AES over the payload, the language
model, and Groth16 integrity proofs). The overhead columns are the makespan of
the secured trace divided by the same trace with the security tasks removed, so
$1.00$ means security was absorbed for free. The remaining columns are
reductions against Multi-core, the only baseline able to run all three domains.
The per-task AES and ZKP means are taken over a handful of tasks per trace and
are correspondingly noisy.}}
\label{{tab:llm-serving}}
\small
\setlength{{\tabcolsep}}{{5pt}}
\begin{{tabular}}{{|l|r|r|r|r|r|r|r|}}
\hline
\textbf{{Model}} & \multicolumn{{2}}{{c|}}{{\textbf{{Security overhead}}}} &
\multicolumn{{4}}{{c|}}{{\textbf{{FUSE reduction vs.\ Multi-core (\%)}}}} & \textbf{{Energy}} \\
 & FUSE & M-core & span & switch & AES & ZKP & gain \\
\hline
{body}
\hline
\end{{tabular}}
\end{{table}}
"""


def frontend_table(frontend: Path) -> Optional[str]:
    path = frontend / 'frontend_summary.csv'
    if not path.exists():
        return None
    summary = pd.read_csv(path)
    # Where a compiler runs is a property of the group, so it heads the block
    # rather than repeating down a column.
    # Only the local group is called out; the caption says the rest are hosted,
    # so a header for them would state it twice.
    labels = {'hosted': None, 'local': 'Run locally on one GPU'}
    rows: List[str] = []
    for block, kind in enumerate(('hosted', 'local')):
        group = summary[summary['kind'] == kind]
        if group.empty:
            continue
        if labels[kind]:
            rows.append(r'\hline')
            rows.append(rf'\multicolumn{{7}}{{|l|}}{{\textit{{{labels[kind]}}}}} \\')
            rows.append(r'\hline')
        for _, row in group.sort_values('plan_fidelity', ascending=False).iterrows():
            rows.append(
                ' & '.join(
                    [
                        _cited(row['model']),
                        _pct(row.get('parse_rate'), 0),
                        _pct(row.get('plan_fidelity'), 0),
                        _fmt(row.get('mean_compile_s'), 1),
                        _pct(row.get('canonical_agreement'), 0),
                        _pct(row.get('fuse_result_reuse')),
                        _pct(row.get('latency_reduction')),
                    ]
                )
                + r' \\'
            )
    body = '\n'.join(r for r in rows if r)
    return rf"""\begin{{table}}[!t]
\centering
\caption{{\textbf{{Language models as the request compiler.}} The first five
compilers are reached over a hosted API, the rest run locally on one GPU. Each
service is stated four ways. \emph{{Parses}}: a reply yields a schedulable plan.
\emph{{Sizes}}: that plan recovers the stated values. \emph{{Agree}}: all four
wordings compile to one graph. \emph{{Reuse}}: results shared between them.
Without canonicalization the last two are zero everywhere.}}
\label{{tab:llm-frontend}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{|l|r|r|r|r|r|r|}}
\hline
\textbf{{Compiler}} & \textbf{{Parses}} & \textbf{{Sizes}} &
\textbf{{Compile}} & \textbf{{Agree}} & \textbf{{Reuse}} & \textbf{{Span}} \\
 & (\%) & (\%) & (s) & (\%) & (\%) & $-$\% \\
\hline
{body}
\hline
\end{{tabular}}
\end{{table}}
"""


def concurrent_table(concurrent: Path, reference_rfus: int = 16, level: Optional[int] = None) -> Optional[str]:
    """One fabric against its best fixed partition, at the evaluated array size."""
    path = concurrent / 'concurrent_results.csv'
    if not path.exists():
        return None
    data = pd.read_csv(path)
    if level is None:
        levels = sorted(data['security_per_request'].unique())
        level = levels[len(levels) // 2]
    part = data[(data['rfus'] == reference_rfus) & (data['security_per_request'] == level)]
    if part.empty:
        return None
    # Dataset-major, so one request shape can be read down the page, with the
    # blocks ruled off and the best margin in each block set in bold.
    rows: List[str] = []
    for block, (dataset, group) in enumerate(part.groupby('dataset_label', sort=True)):
        rows.append(r'\hline' if block else '')
        prompt = int(group['prompt_tokens'].iloc[0])
        rows.append(rf'\multicolumn{{6}}{{|l|}}{{\textit{{{_escape(dataset)}}}, '
                    rf'{prompt}-token prompts}} \\')
        rows.append(r'\hline')
        best = group['speedup_vs_partitioned'].max()
        for _, row in group.sort_values('fuse_ms').iterrows():
            speedup = _fmt(row.get('speedup_vs_partitioned'), 2)
            if abs(float(row['speedup_vs_partitioned']) - best) < 5e-3:
                speedup = rf'\textbf{{{speedup}}}'
            rows.append(
                ' & '.join([
                    _cited(row['model_label']),
                    _fmt(row.get('fuse_ms'), 1),
                    _fmt(row.get('partitioned_ms'), 1),
                    speedup,
                    _fmt(row.get('speedup_vs_dedicated'), 2),
                    _fmt(row.get('energy_gain_vs_partitioned'), 2),
                ])
                + r' \\'
            )
    body = '\n'.join(r for r in rows if r)
    return rf"""\begin{{table}}[!t]
\centering
\caption{{\textbf{{Concurrent secure serving}} on a {reference_rfus}-RFU
array, {level} AES and {level} Groth16 tasks per request. \emph{{Partition}}
slices the array between engines with FUSE's datapath, while \emph{{Dedicated}} adds
each domain's own. Both get their best split, always {{\ttfamily 14/1/1}}. Best
speedup per group in bold.}}
\label{{tab:concurrent}}
\small
\setlength{{\tabcolsep}}{{5pt}}
\begin{{tabular}}{{|l|r|r|r|r|r|}}
\hline
\textbf{{Model}} & \textbf{{FUSE}} & \textbf{{Partition}} &
\multicolumn{{2}}{{c|}}{{\textbf{{Speedup vs.}}}} & \textbf{{Energy}} \\
 & (ms) & (ms) & part. & ded. & gain \\
\hline
{body}
\hline
\end{{tabular}}
\end{{table}}
"""


def concurrent_macros(concurrent: Path, reference_rfus: int = 16,
                      levels_sweep: Optional[Path] = None) -> List[Tuple[str, str]]:
    """Numbers the concurrent-serving prose quotes."""
    path = concurrent / 'concurrent_results.csv'
    out: List[Tuple[str, str]] = []
    if not path.exists():
        return out
    data = pd.read_csv(path)
    levels = sorted(data['security_per_request'].unique())
    level = levels[len(levels) // 2]
    out.append(('concModels', str(data['model_label'].nunique())))
    out.append(('concDatasets', str(data['dataset_label'].nunique())))
    out.append(('concLevel', str(int(level))))
    out.append(('concLevelMax', str(int(max(levels)))))
    out.append(('concArrayMin', str(int(data['rfus'].min()))))
    out.append(('concArrayMax', str(int(data['rfus'].max()))))
    out.append(('concRefArray', str(int(reference_rfus))))

    ref = data[(data['rfus'] == reference_rfus) & (data['security_per_request'] == level)]
    if not ref.empty:
        out.append(('concRefSpeedupMin', _fmt(ref['speedup_vs_partitioned'].min(), 2)))
        out.append(('concRefSpeedupMax', _fmt(ref['speedup_vs_partitioned'].max(), 2)))
        out.append(('concRefSpeedupMean', _fmt(ref['speedup_vs_partitioned'].mean(), 2)))
        out.append(('concRefEnergyMean', _fmt(ref['energy_gain_vs_partitioned'].mean(), 2)))
        out.append(('concRefDedSpeedupMin', _fmt(ref['speedup_vs_dedicated'].min(), 2)))
        out.append(('concRefDedSpeedupMax', _fmt(ref['speedup_vs_dedicated'].max(), 2)))
        out.append(('concRefDedSpeedupMean', _fmt(ref['speedup_vs_dedicated'].mean(), 2)))
        losers = ref[ref['speedup_vs_dedicated'] < 1.0]['model_label'].unique()
        if len(losers):
            out.append(('concRefDedLoser', _escape(', '.join(sorted(losers)))))
        out.append(('concRefTaxMax', _pct(ref['security_tax'].abs().max(), 2)))
        out.append(('concRefReserved', _fmt(ref['partitioned_reserved'].mean(), 1)))

    at_level = data[data['security_per_request'] == level]
    small = at_level[at_level['rfus'] == at_level['rfus'].min()]
    large = at_level[at_level['rfus'] == at_level['rfus'].max()]
    if not small.empty:
        out.append(('concSmallSpeedupMean', _fmt(small['speedup_vs_partitioned'].mean(), 2)))
        out.append(('concSmallSpeedupMax', _fmt(small['speedup_vs_partitioned'].max(), 2)))
    if not large.empty:
        out.append(('concLargeSpeedupMean', _fmt(large['speedup_vs_partitioned'].mean(), 2)))

    # where sharing stops being worth anything
    crossing = (
        at_level.groupby('rfus')['speedup_vs_partitioned'].mean().reset_index()
        .sort_values('rfus')
    )
    threshold = crossing[crossing['speedup_vs_partitioned'] < 1.02]
    if not threshold.empty:
        out.append(('concCrossover', str(int(threshold['rfus'].iloc[0]))))

    heavy = data[(data['rfus'] == reference_rfus) & (data['security_per_request'] == max(levels))]
    if not heavy.empty:
        out.append(('concHeavySpeedupMean', _fmt(heavy['speedup_vs_partitioned'].mean(), 2)))
    # The obligation level is swept separately, at the evaluated array size, so
    # both files are consulted for anything that depends on it.
    pooled = [data]
    levels_path = (levels_sweep / 'concurrent_results.csv') if levels_sweep else None
    if levels_path is not None and levels_path.exists():
        extra = pd.read_csv(levels_path)
        pooled.append(extra)
        spread = (extra.groupby(['model_label', 'dataset_label', 'rfus'])['speedup_vs_partitioned']
                  .agg(lambda v: v.max() - v.min()).max())
        out.append(('concSpeedupLevelSpread', _fmt(spread, 4)))
        out.append(('concLevelSweepMax', str(int(extra['security_per_request'].max()))))
        out.append(('concLevelSweepMin', str(int(extra['security_per_request'].min()))))
        out.append(('concLevelSweepModels', str(int(extra['model_label'].nunique()))))
    combined = pd.concat(pooled, ignore_index=True)
    # The tax needs more digits than a percentage point.  Only a positive value
    # is a cost; the negative ones mean the secured trace finished sooner than
    # the bare one, which is scheduling perturbation rather than a saving, so
    # the two directions are reported separately instead of as one magnitude.
    out.append(('concTaxWorst', _pct(combined['security_tax'].max(), 3)))
    worst = combined.loc[combined['security_tax'].idxmax()]
    out.append(('concTaxWorstWhere',
                _escape(f"{worst['model_label']} at {int(worst['rfus'])} RFUs")))
    out.append(('concTaxNoise', _pct(abs(combined['security_tax'].min()), 3)))

    scaling_path = concurrent / 'rfu_scaling.csv'
    if scaling_path.exists():
        scaling = pd.read_csv(scaling_path)
        one = scaling[scaling['dataset'] == scaling['dataset'].iloc[0]]
        points, peaks = [], []
        for _, part in one.groupby('model_label', sort=False):
            part = part.sort_values('rfus')
            base = float(part[part['rfus'] == 1]['makespan_ms'].iloc[0])
            speedup = (base / part['makespan_ms']).tolist()
            rfus = part['rfus'].tolist()
            peak = max(speedup)
            peaks.append(peak)
            points.append(next((rfus[i] for i, v in enumerate(speedup) if v >= 0.98 * peak),
                               rfus[-1]))
        if points:
            # Every model above the smallest saturates at the same width, which is
            # the mapping's doing rather than the model's.
            common = max(set(points), key=points.count)
            out.append(('concSaturationRfus', str(int(common))))
            out.append(('concSaturationMin', str(int(min(points)))))
            out.append(('concPeakSpeedupMax', _fmt(max(peaks), 0)))

    env_path = concurrent / 'concurrent_env.json'
    if env_path.exists():
        import json as _json

        env = _json.loads(env_path.read_text(encoding='utf-8'))
        params = [m.get('params_b', 0.0) for m in env.get('models', [])]
        if params:
            out.append(('concParamMin', _fmt(min(params), 2)))
            out.append(('concParamMax', _fmt(max(params), 0)))

    answers_path = concurrent / 'answer_lengths.json'
    if answers_path.exists():
        import json as _json

        answers = _json.loads(answers_path.read_text(encoding='utf-8'))
        per_dataset = answers.get('datasets', {})
        if per_dataset:
            out.append(('concAnswerModels', str(len(answers.get('models', [])))))
            out.append(('concAnswerSamples', str(int(answers.get('samples', 0)))))
            medians = {k: v['median_of_models'] for k, v in per_dataset.items()}
            for key, tag in (('gsm8k', 'Gsm'), ('mmlu', 'Mmlu'), ('humaneval', 'HumanEval')):
                if key in per_dataset:
                    rec = per_dataset[key]
                    out.append((f'concAnswer{tag}', _fmt(rec['median_of_models'], 0)))
                    out.append((f'concAnswer{tag}Min', _fmt(rec['min_of_models'], 0)))
                    out.append((f'concAnswer{tag}Max', _fmt(rec['max_of_models'], 0)))
            if medians:
                out.append(('concAnswerAcrossDatasets',
                            _fmt(max(medians.values()) / max(min(medians.values()), 1e-9), 1)))
            # how much more the choice of model matters than the choice of dataset
            spreads = [rec['max_of_models'] / max(rec['min_of_models'], 1e-9)
                       for rec in per_dataset.values()]
            out.append(('concAnswerAcrossModels', _fmt(max(spreads), 1)))

    shapes_path = concurrent / 'request_shapes.csv'
    if shapes_path.exists():
        shapes = pd.read_csv(shapes_path)
        out.append(('concCorpusSize', str(int(shapes['corpus_size'].iloc[0]))))
        out.append(('concZkpConstraints', str(int(shapes['zkp_constraints'].iloc[0]))))
        out.append(('concAesBlocksMin', str(int(shapes['aes_blocks'].min()))))
        out.append(('concAesBlocksMax', str(int(shapes['aes_blocks'].max()))))
        out.append(('concRetrievedBytes', str(int(shapes['retrieved_bytes'].iloc[0]))))
    return out


def macros_used_by(section: Path) -> List[str]:
    """Every \\llm... macro the prose references."""
    if not section.exists():
        return []
    import re

    return sorted(set(re.findall(r'\\(llm[A-Za-z]+)', section.read_text(encoding='utf-8'))))


def macros(zoo: Path, frontend: Path, section: Optional[Path] = None,
           concurrent: Optional[Path] = None, reference_rfus: int = 16,
           levels_sweep: Optional[Path] = None) -> str:
    """Define the numbers the prose quotes, so the text cannot drift from the run."""
    lines: List[str] = [
        '% Generated by scripts/make_llm_tables.py -- do not edit by hand.',
    ]
    defined: List[str] = []

    def define(name: str, value: str) -> None:
        lines.append(rf'\newcommand{{\{name}}}{{{value}}}')
        defined.append(name)

    summary = pd.read_csv(zoo / 'llm_summary.csv')
    define('llmModelCount', str(len(summary)))
    smallest = summary['parameter_count'].min() / 1e9
    largest = summary['parameter_count'].max() / 1e9
    define('llmParamMin', f'{smallest:.2f}')
    define('llmParamMax', f'{largest:.1f}')
    define('llmPromptTokens', _fmt(summary['prompt_tokens_mean'].mean(), 0))
    define('llmGenTokens', str(int(summary['gen_tokens'].iloc[0])))

    for column, prefix in (
        ('transformer_asic_latency_ratio', 'llmTasicRatio'),
        ('gpu_tensor_latency_ratio', 'llmGpuRatio'),
        ('multicore_latency_ratio', 'llmMulticoreRatio'),
        ('gpu_tensor_energy_gain', 'llmGpuEnergy'),
        ('multicore_energy_gain', 'llmMulticoreEnergy'),
        ('transformer_asic_energy_gain', 'llmTasicEnergy'),
    ):
        if column in summary.columns:
            define(f'{prefix}Min', _fmt(summary[column].min(), 3 if 'Ratio' in prefix else 2))
            define(f'{prefix}Max', _fmt(summary[column].max(), 3 if 'Ratio' in prefix else 2))
            define(f'{prefix}Mean', _fmt(summary[column].mean(), 3 if 'Ratio' in prefix else 2))

    for column, name in (
        ('fuse_rfu_utilization', 'llmRfuUtil'),
        ('fuse_structural_reuse_rate', 'llmReuseRate'),
        ('serving_switch_reduction', 'llmSwitchReduction'),
        ('serving_latency_reduction', 'llmServingSpan'),
        ('serving_energy_gain', 'llmServingEnergy'),
    ):
        if column in summary.columns:
            digits = 2 if name == 'llmServingEnergy' else 1
            if name == 'llmServingEnergy':
                define(f'{name}Mean', _fmt(summary[column].mean(), digits))
            else:
                define(f'{name}Mean', _pct(summary[column].mean(), digits))
                define(f'{name}Min', _pct(summary[column].min(), digits))
                define(f'{name}Max', _pct(summary[column].max(), digits))

    for column, name in (
        ('security_overhead_fuse', 'llmSecOverheadFuse'),
        ('security_overhead_multicore', 'llmSecOverheadMulticore'),
    ):
        if column in summary.columns:
            define(f'{name}Mean', _fmt(summary[column].mean(), 3))
            define(f'{name}Max', _fmt(summary[column].max(), 3))

    if 'transformer_asic_energy_gain' in summary.columns:
        # The gain is baseline/FUSE, so its reciprocal is what FUSE costs.
        gain = summary['transformer_asic_energy_gain']
        define('llmTasicEnergyCost', _fmt((1.0 / gain.clip(lower=1e-9)).mean(), 2))

    if 'serving_aes_latency_reduction' in summary.columns:
        reductions = summary['serving_aes_latency_reduction']
        define('llmAesBest', _pct(reductions.max(), 1))
        define('llmAesWorst', _pct(reductions.min(), 1))
        define('llmAesWorstMag', _pct(abs(reductions.min()), 1))
    if 'serving_aes_worst_reduction' in summary.columns:
        # reduction = 1 - fuse/multicore, so fuse/multicore = 1 - reduction.
        worst = summary['serving_aes_worst_reduction'].min()
        define('llmAesSweepWorstMag', _fmt(1.0 - worst, 1))
        if 'serving_aes_worst_scale' in summary.columns:
            row = summary.loc[summary['serving_aes_worst_reduction'].idxmin()]
            define('llmAesWorstScale', _fmt(row['serving_aes_worst_scale'], 0))

    moe = summary[summary['family'] == 'MoE'] if 'family' in summary.columns else pd.DataFrame()
    if not moe.empty and 'transformer_asic_latency_ratio' in moe.columns:
        define('llmMoeRatio', _fmt(moe['transformer_asic_latency_ratio'].mean(), 3))
        define('llmMoeLabel', _escape(moe['model_label'].iloc[0]))

    frontend_path = frontend / 'frontend_summary.csv'
    if frontend_path.exists():
        fsum = pd.read_csv(frontend_path)
        define('llmFrontendCount', str(len(fsum)))
        define('llmFrontendHosted', str(int((fsum['kind'] == 'hosted').sum())))
        define('llmFrontendLocal', str(int((fsum['kind'] == 'local').sum())))
        define('llmFrontendParseMean', _pct(fsum['parse_rate'].mean(), 0))
        define('llmFrontendFidelityMean', _pct(fsum['plan_fidelity'].mean(), 0))
        for kind, tag in (('hosted', 'llmFrontendHosted'), ('local', 'llmFrontendLocal')):
            side = fsum[fsum['kind'] == kind]
            if side.empty:
                continue
            define(f'{tag}Parse', _pct(side['parse_rate'].mean(), 0))
            define(f'{tag}ParseMin', _pct(side['parse_rate'].min(), 0))
            define(f'{tag}Fidelity', _pct(side['plan_fidelity'].mean(), 0))
            define(f'{tag}FidelityMin', _pct(side['plan_fidelity'].min(), 0))
            if 'canonical_agreement' in side.columns:
                define(f'{tag}CanonAgree', _pct(side['canonical_agreement'].mean(), 0))
            if 'fuse_result_reuse' in side.columns:
                define(f'{tag}Reuse', _pct(side['fuse_result_reuse'].mean(), 1))
        # Canonicalization can only unify plans that are genuinely equivalent, so
        # the agreement figure is reported over the compilers that produced the
        # same plan for every wording in the first place.
        if {'canonical_agreement', 'plan_fidelity'}.issubset(fsum.columns):
            faithful = fsum[fsum['plan_fidelity'] >= 0.9]
            if not faithful.empty:
                define('llmFrontendFaithfulCount', str(len(faithful)))
                define('llmFrontendFaithfulCanonAgree', _pct(faithful['canonical_agreement'].mean(), 0))
                define('llmFrontendFaithfulPlainAgree', _pct(faithful['unprotected_agreement'].mean(), 0))
        for column, name in (
            ('canonical_agreement', 'llmFrontendCanonAgree'),
            ('unprotected_agreement', 'llmFrontendPlainAgree'),
            ('latency_reduction', 'llmFrontendSpan'),
            ('fuse_result_reuse', 'llmFrontendReuse'),
            ('unprotected_result_reuse', 'llmFrontendPlainReuse'),
        ):
            if column in fsum.columns:
                define(f'{name}Mean', _pct(fsum[column].mean(), 1))
        if {'unprotected_switch_cycles', 'fuse_switch_cycles'} <= set(fsum.columns):
            ratio = 1.0 - (fsum['fuse_switch_cycles'] / fsum['unprotected_switch_cycles'].clip(lower=1e-9))
            define('llmFrontendSwitchMean', _pct(ratio.mean(), 1))

    for name, value in concurrent_macros(concurrent or Path(), reference_rfus,
                                         levels_sweep=levels_sweep):
        define(f'llm{name[0].upper()}{name[1:]}' if not name.startswith('llm') else name, value)

    # A partial run must still build the paper: anything the prose references but
    # this run could not measure falls back to a dash, which is visible in the
    # output rather than a compile failure.
    missing = [name for name in macros_used_by(section or Path()) if name not in defined]
    if missing:
        lines.append('')
        lines.append('% Not produced by this run:')
        for name in missing:
            lines.append(rf'\providecommand{{\{name}}}{{--}}')
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zoo', type=Path, default=ROOT / 'runs' / 'llm_zoo')
    parser.add_argument('--frontend', type=Path, default=ROOT / 'runs' / 'llm_frontend')
    parser.add_argument('--concurrent', type=Path, default=ROOT / 'runs' / 'concurrent')
    parser.add_argument('--reference-rfus', type=int, default=16)
    parser.add_argument('--concurrent-levels', type=Path,
                        default=ROOT / 'runs' / 'concurrent_levels',
                        help='supplementary sweep over the obligation level')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'sosp')
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    if (args.zoo / 'llm_summary.csv').exists():
        for name, text in (
            ('tab_llm_zoo.tex', model_table(args.zoo)),
            ('tab_llm_results.tex', results_table(args.zoo)),
            ('tab_llm_serving.tex', serving_table(args.zoo)),
        ):
            target = args.out_dir / name
            target.write_text(text, encoding='utf-8')
            written.append(target)
        for figure in ('fig_llm_latency.pdf', 'fig_llm_energy.pdf', 'fig_llm_secure_serving.pdf'):
            source = args.zoo / figure
            if source.exists():
                shutil.copy2(source, args.out_dir / figure)
                written.append(args.out_dir / figure)
    else:
        print(f'! no zoo summary at {args.zoo}')

    conc = concurrent_table(args.concurrent, reference_rfus=args.reference_rfus)
    if conc is not None:
        target = args.out_dir / 'tab_concurrent.tex'
        target.write_text(conc, encoding='utf-8')
        written.append(target)
        source = args.concurrent / 'fig_concurrent_secure.pdf'
        if source.exists():
            shutil.copy2(source, args.out_dir / 'fig_concurrent_secure.pdf')
            written.append(args.out_dir / 'fig_concurrent_secure.pdf')
    else:
        print(f'! no concurrent results at {args.concurrent}')

    table = frontend_table(args.frontend)
    if table is not None:
        target = args.out_dir / 'tab_llm_frontend.tex'
        target.write_text(table, encoding='utf-8')
        written.append(target)
        source = args.frontend / 'fig_llm_frontend.pdf'
        if source.exists():
            shutil.copy2(source, args.out_dir / 'fig_llm_frontend.pdf')
            written.append(args.out_dir / 'fig_llm_frontend.pdf')
    else:
        print(f'! no front-end summary at {args.frontend}')

    if (args.zoo / 'llm_summary.csv').exists():
        target = args.out_dir / 'llm_numbers.tex'
        target.write_text(
            macros(args.zoo, args.frontend, args.out_dir / 'sec_llm.tex',
                   concurrent=args.concurrent, reference_rfus=args.reference_rfus,
                   levels_sweep=args.concurrent_levels),
            encoding='utf-8')
        written.append(target)

    for path in written:
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
