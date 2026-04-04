from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import hashlib
import json
import math
import numpy as np
import pandas as pd
from .benchmarks import BenchmarkSuite, BenchmarkArtifacts, load_calibration
from .branding import ACCELERATOR_NAME
from .ir import RuntimeTask, TaskGraph
from .llm import FUSEVerifier, HeuristicLLMFrontend
from .security import binary_security_score, mutual_information_binary, pearson_corr, run_binary_attacks, trace_features
from .sim.simulator import CYCLES_PER_MS
from .workloads import build_aes128_task, build_groth16_like_task, build_mlp_path_task, build_transformer_block_task, build_vgg16_like_task

@dataclass
class SOSPArtifacts:
    base: BenchmarkArtifacts
    dataframes: Dict[str, pd.DataFrame]
    rawframes: Dict[str, pd.DataFrame]

class SOSPSuite:

    def __init__(self, engine: str='cpp', calibration: Dict[str, Dict[str, float]] | None=None, seed: int=7):
        self.engine = engine
        self.seed = seed
        self.base_suite = BenchmarkSuite(engine=engine, calibration=calibration)
        self.frontend = HeuristicLLMFrontend()
        self.verifier = FUSEVerifier()

    def run(self, output_dir: Path) -> SOSPArtifacts:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = output_dir / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        base = self._load_or_run_base(output_dir)
        security_frames = self.run_security(output_dir)
        case_frames = self.run_case_studies(output_dir)
        ablation_frames = self.run_ablations(output_dir)
        table = self.build_summary_table(base, security_frames, case_frames, ablation_frames)
        table.to_csv(output_dir / 'table1_sosp_summary.csv', index=False)
        dataframes = {**base.dataframes, 'fig13_security': security_frames['fig13_security'], 'fig14_llm_stability': security_frames['fig14_llm_stability'], 'fig15_secure_tradeoff': security_frames['fig15_secure_tradeoff'], 'fig16_case_studies': case_frames['fig16_case_studies'], 'fig17_ablation_plus': ablation_frames['fig17_ablation_plus'], 'table1_sosp_summary': table}
        rawframes = {**base.rawframes, **security_frames, **case_frames, **ablation_frames, 'table1_sosp_summary': table}
        for name, df in dataframes.items():
            if isinstance(df, pd.DataFrame):
                df.to_csv(output_dir / f'{name}.csv', index=False)
        for name, df in rawframes.items():
            if isinstance(df, pd.DataFrame) and name not in dataframes:
                df.to_csv(raw_dir / f'{name}.csv', index=False)
        return SOSPArtifacts(base=base, dataframes=dataframes, rawframes=rawframes)

    def _persist_frames(self, output_dir: Path, frames: Dict[str, pd.DataFrame]) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, df in frames.items():
            if isinstance(df, pd.DataFrame):
                df.to_csv(output_dir / f'{name}.csv', index=False)

    def _persist_raw_frames(self, raw_dir: Path, frames: Dict[str, pd.DataFrame], skip: set[str] | None=None) -> None:
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        skip = skip or set()
        for name, df in frames.items():
            if isinstance(df, pd.DataFrame) and name not in skip:
                df.to_csv(raw_dir / f'{name}.csv', index=False)

    def _load_or_run_base(self, output_dir: Path, load_raw: bool=True) -> BenchmarkArtifacts:
        required = ['fig4_latency.csv', 'fig5_energy.csv', 'fig6_mixed.csv', 'fig7_area_power.csv', 'fig8a_reuse.csv', 'fig8b_fanout.csv', 'fig8c_pool.csv', 'fig9_ablation.csv', 'fig10_memory.csv', 'fig11a_sched.csv', 'fig11b_switch.csv', 'fig11c_summary.csv', 'fig12_reconfig.csv']
        if all(((output_dir / name).exists() for name in required)):
            dataframes = {'fig4_latency': pd.read_csv(output_dir / 'fig4_latency.csv'), 'fig5_energy': pd.read_csv(output_dir / 'fig5_energy.csv'), 'fig6_mixed': pd.read_csv(output_dir / 'fig6_mixed.csv'), 'fig7_area_power': pd.read_csv(output_dir / 'fig7_area_power.csv'), 'fig8a_reuse': pd.read_csv(output_dir / 'fig8a_reuse.csv'), 'fig8b_fanout': pd.read_csv(output_dir / 'fig8b_fanout.csv'), 'fig8c_pool': pd.read_csv(output_dir / 'fig8c_pool.csv'), 'fig9_ablation': pd.read_csv(output_dir / 'fig9_ablation.csv'), 'fig10_memory': pd.read_csv(output_dir / 'fig10_memory.csv'), 'fig11a_sched': pd.read_csv(output_dir / 'fig11a_sched.csv'), 'fig11b_switch': pd.read_csv(output_dir / 'fig11b_switch.csv'), 'fig11c_summary': pd.read_csv(output_dir / 'fig11c_summary.csv'), 'fig12_reconfig': pd.read_csv(output_dir / 'fig12_reconfig.csv')}
            rawframes = {}
            raw_dir = output_dir / 'raw'
            if load_raw and raw_dir.exists():
                for csv_path in raw_dir.glob('*.csv'):
                    rawframes[csv_path.stem] = pd.read_csv(csv_path)
            calibration = {}
            cal_path = output_dir / 'calibration.json'
            if cal_path.exists():
                calibration = json.loads(cal_path.read_text(encoding='utf-8'))
            return BenchmarkArtifacts(dataframes=dataframes, rawframes=rawframes, traces={}, scalars={}, calibration=calibration)
        return self.base_suite.run(output_dir)

    def run_security(self, output_dir: Path) -> Dict[str, pd.DataFrame]:
        raw_dir = Path(output_dir) / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        mode_specs = [('Insecure LLM', False, 'fuse'), ('Canonical IR', True, 'fuse'), ('Secure-CT', True, 'secure_ct'), ('Secure-Rand', True, 'secure_randomized'), ('Multi-core', True, 'multicore')]
        trial_rows: List[Dict[str, object]] = []
        stability_rows: List[Dict[str, object]] = []
        variant_catalog = self.frontend.variant_prompts()
        selected_intents = ['zkp', 'crypto', 'llm', 'hybrid']
        for mode_name, secure, sim_mode in mode_specs:
            for intent in selected_intents:
                semantic_sigs: List[str] = []
                schedule_sigs: List[str] = []
                canonical_hashes: List[str] = []
                latencies: List[float] = []
                verified: List[int] = []
                for idx, variant in enumerate(variant_catalog[intent]):
                    compiled = self.frontend.compile_prompt(variant.prompt, secure=secure, task_prefix=f"{intent}_{mode_name.lower().replace(' ', '_').replace('-', '_')}_{idx}")
                    verify = self.verifier.verify_graphs(compiled.graphs)
                    verified.append(1 if verify.ok else 0)
                    runtime_tasks = self._replicated_runtime_tasks(compiled.graphs, repeats=2 if len(compiled.graphs) == 1 else 1, prefix=f'{intent}_{idx}')
                    scenario_name = f"fig13_{mode_name.lower().replace(' ', '_').replace('-', '_')}_{intent}_{idx}"
                    result, events = self.base_suite._simulate(compiled.graphs, runtime_tasks, mode=sim_mode, scenario_name=scenario_name, raw_dir=raw_dir)
                    features = trace_features(events, result.metrics)
                    semantic_sig = self._semantic_signature(compiled.graphs)
                    schedule_sig = self._schedule_signature(events)
                    semantic_sigs.append(semantic_sig)
                    schedule_sigs.append(schedule_sig)
                    canonical_hashes.append(compiled.canonical_hash)
                    latencies.append(float(features['latency_ms']))
                    trial_rows.append({'mode': mode_name, 'sim_mode': sim_mode, 'intent': intent, 'label': 1 if variant.label == 'beta' else 0, 'prompt_id': idx, 'prompt_hash': compiled.prompt_hash, 'canonical_hash': compiled.canonical_hash, 'semantic_signature': semantic_sig, 'schedule_signature': schedule_sig, 'verified': 1 if verify.ok else 0, **features})
                latency_mean = float(np.mean(latencies)) if latencies else 0.0
                stability_rows.append({'mode': mode_name, 'intent': intent, 'unique_semantic_ratio': len(set(semantic_sigs)) / max(1, len(semantic_sigs)), 'unique_schedule_ratio': len(set(schedule_sigs)) / max(1, len(schedule_sigs)), 'canonical_consistency': 1.0 / max(1, len(set(canonical_hashes))), 'latency_cv': float(np.std(latencies) / max(latency_mean, 1e-09)) if latencies else 0.0, 'verified_pass_rate': float(np.mean(verified)) if verified else 0.0})
        trials = pd.DataFrame(trial_rows)
        stability = pd.DataFrame(stability_rows)
        attack_rows: List[Dict[str, object]] = []
        feature_map = {'latency': 'latency_ms', 'reuse': 'reuse_rate', 'schedule': 'template_switches'}
        for mode_name, part in trials.groupby('mode'):
            for attack in run_binary_attacks(part, label_col='label'):
                feature = feature_map[attack.attack]
                mi = mutual_information_binary(part['label'], part[feature])
                corr = pearson_corr(part['label'], part[feature])
                attack_rows.append({'mode': mode_name, 'attack': attack.attack, 'accuracy': attack.accuracy, 'mutual_information': mi, 'correlation': corr, 'security_score': binary_security_score(attack.accuracy, mi, corr)})
        fig13 = pd.DataFrame(attack_rows)
        fig15 = self._build_tradeoff_table(fig13, stability, trials)
        return {'fig13_security': fig13, 'fig14_llm_stability': stability, 'fig15_secure_tradeoff': fig15, 'fig13_security_trials': trials}

    def run_case_studies(self, output_dir: Path) -> Dict[str, pd.DataFrame]:
        raw_dir = Path(output_dir) / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        modes = [(ACCELERATOR_NAME, 'fuse'), ('Secure-CT', 'secure_ct'), ('Secure-Rand', 'secure_randomized'), ('Multi-core', 'multicore'), ('GPU-Tensor', 'gpu_tensor')]
        case_defs = self._case_definitions()
        summary_rows: List[Dict[str, object]] = []
        raw_rows: List[Dict[str, object]] = []
        for case in case_defs:
            graphs, runtime_tasks = (case['graphs'], case['runtime_tasks'])
            for mode_name, sim_mode in modes:
                scenario_name = f"fig16_{case['name'].lower().replace(' ', '_').replace('-', '_')}_{sim_mode}"
                result, events = self.base_suite._simulate(graphs, runtime_tasks, mode=sim_mode, scenario_name=scenario_name, raw_dir=raw_dir)
                latencies = self._task_latencies(events, runtime_tasks)
                if not latencies:
                    latencies = [float(result.metrics.get('makespan_ms', 0.0))]
                p50 = float(np.percentile(latencies, 50))
                p99 = float(np.percentile(latencies, 99))
                throughput = len(runtime_tasks) / max(float(result.metrics.get('makespan_ms', 0.0)) / 1000.0, 1e-09)
                slo_miss = float(np.mean(np.asarray(latencies) > case['slo_ms']))
                energy_per_task = float(result.metrics.get('energy_nj', 0.0)) / max(1, len(runtime_tasks))
                features = trace_features(events, result.metrics)
                summary_rows.append({'case': case['name'], 'mode': mode_name, 'p50_ms': p50, 'p99_ms': p99, 'throughput_tps': throughput, 'energy_per_task_nj': energy_per_task, 'slo_miss_ratio': slo_miss, 'rfu_utilization': float(result.metrics.get('rfu_utilization', 0.0)), 'reuse_rate': features['reuse_rate']})
                for runtime_id, latency in zip([task.runtime_id for task in runtime_tasks], latencies):
                    raw_rows.append({'case': case['name'], 'mode': mode_name, 'runtime_id': runtime_id, 'latency_ms': latency})
        return {'fig16_case_studies': pd.DataFrame(summary_rows), 'fig16_case_latencies': pd.DataFrame(raw_rows)}

    def run_ablations(self, output_dir: Path) -> Dict[str, pd.DataFrame]:
        raw_dir = Path(output_dir) / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        workloads = {'Groth16': self.frontend.compile_prompt('compile a groth16 prover for a 4k-gate rollup circuit', secure=True, task_prefix='ablate_groth').graphs, 'Transformer': self.frontend.compile_prompt('compile a transformer block for llm prefill seq 1024 hidden 3072 heads 24', secure=True, task_prefix='ablate_llm').graphs, 'Hybrid': self.frontend.compile_prompt('build a hybrid agent pipeline with llm prefill seq 512 aes 512 and groth16 2048', secure=True, task_prefix='ablate_hybrid').graphs}
        configs = [(f'Full {ACCELERATOR_NAME}', True, 'fuse'), ('- Canonical IR', False, 'fuse'), ('- FVM reuse', True, 'no_fvm'), ('- LRC prefetch', True, 'no_prefetch'), ('- TARS scorer', True, 'fixed_schedule'), ('- Cross-task reuse', True, 'multicore'), ('Secure-Rand', True, 'secure_randomized'), ('Secure-CT', True, 'secure_ct')]
        rows: List[Dict[str, object]] = []
        for workload, base_graphs in workloads.items():
            prompt_map = {'Groth16': 'compile a groth16 prover for a 4k-gate rollup circuit', 'Transformer': 'compile a transformer block for llm prefill seq 1024 hidden 3072 heads 24', 'Hybrid': 'build a hybrid agent pipeline with llm prefill seq 512 aes 512 and groth16 2048'}
            latency_ref = None
            energy_ref = None
            for config_name, secure, sim_mode in configs:
                graphs = self.frontend.compile_prompt(prompt_map[workload], secure=secure, task_prefix=f"{workload.lower()}_{config_name.lower().replace(' ', '_').replace('-', '_')}").graphs
                runtime_tasks = self._replicated_runtime_tasks(graphs, repeats=2 if len(graphs) == 1 else 1, prefix=f'{workload}_{sim_mode}')
                scenario_name = f"fig17_{workload.lower()}_{config_name.lower().replace(' ', '_').replace('-', '_')}"
                result, events = self.base_suite._simulate(graphs, runtime_tasks, mode=sim_mode, scenario_name=scenario_name, raw_dir=raw_dir)
                features = trace_features(events, result.metrics)
                leakage_proxy = min(1.0, 0.45 * features['reuse_rate'] + 0.25 * features['structural_reuse_rate'] + 0.2 * features['template_switches'] + 0.1 * features['rfu_entropy'])
                latency = float(result.metrics.get('makespan_ms', 0.0))
                energy = float(result.metrics.get('energy_nj', 0.0))
                if config_name == f'Full {ACCELERATOR_NAME}':
                    latency_ref = max(latency, 1e-09)
                    energy_ref = max(energy, 1e-09)
                rows.append({'workload': workload, 'configuration': config_name, 'latency_norm': latency / max(latency_ref or latency, 1e-09), 'energy_norm': energy / max(energy_ref or energy, 1e-09), 'leakage_proxy': leakage_proxy, 'rfu_utilization': float(result.metrics.get('rfu_utilization', 0.0))})
        return {'fig17_ablation_plus': pd.DataFrame(rows)}

    def build_summary_table(self, base: BenchmarkArtifacts, security_frames: Dict[str, pd.DataFrame], case_frames: Dict[str, pd.DataFrame], ablation_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        fig4 = base.dataframes['fig4_latency']
        for domain, part in fig4.groupby('domain'):
            rows.append({'category': 'domain', 'name': domain, 'metric': 'mean_speedup_vs_baseline', 'value': float((part['baseline_ms'] / part['morph_ms']).mean())})
        fig13 = security_frames['fig13_security']
        for mode, part in fig13.groupby('mode'):
            rows.append({'category': 'security', 'name': mode, 'metric': 'mean_security_score', 'value': float(part['security_score'].mean())})
        fig16 = case_frames['fig16_case_studies']
        baseline = fig16[fig16['mode'] == 'Multi-core'].set_index('case')
        for case, part in fig16.groupby('case'):
            morph = part[part['mode'] == ACCELERATOR_NAME].iloc[0]
            base_row = baseline.loc[case]
            rows.append({'category': 'case', 'name': case, 'metric': 'p99_gain_vs_multicore', 'value': float(base_row['p99_ms'] / max(morph['p99_ms'], 1e-09))})
            rows.append({'category': 'case', 'name': case, 'metric': 'energy_gain_vs_multicore', 'value': float(base_row['energy_per_task_nj'] / max(morph['energy_per_task_nj'], 1e-09))})
        fig17 = ablation_frames['fig17_ablation_plus']
        for workload, part in fig17.groupby('workload'):
            secure = part[part['configuration'] == 'Secure-CT'].iloc[0]
            rows.append({'category': 'ablation', 'name': workload, 'metric': 'secure_ct_overhead', 'value': float(secure['latency_norm'])})
        return pd.DataFrame(rows)

    def _build_tradeoff_table(self, fig13: pd.DataFrame, stability: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
        perf = trials.groupby('mode', as_index=False)[['latency_ms', 'energy_nj']].mean()
        sec = fig13.groupby('mode', as_index=False)['security_score'].mean()
        stab = stability.groupby('mode', as_index=False).agg(canonical_consistency=('canonical_consistency', 'mean'), latency_cv=('latency_cv', 'mean'), verified_pass_rate=('verified_pass_rate', 'mean'), unique_semantic_ratio=('unique_semantic_ratio', 'mean'))
        merged = perf.merge(sec, on='mode').merge(stab, on='mode')
        lat_ref = max(1e-09, float(merged['latency_ms'].min()))
        energy_ref = max(1e-09, float(merged['energy_nj'].min()))
        rows = []
        for row in merged.to_dict(orient='records'):
            determinism = max(0.0, min(1.0, 0.45 * row['canonical_consistency'] + 0.35 * row['verified_pass_rate'] + 0.2 * (1.0 - min(1.0, row['latency_cv'] / 0.25))))
            balanced = 0.35 * (lat_ref / max(row['latency_ms'], 1e-09)) + 0.2 * (energy_ref / max(row['energy_nj'], 1e-09)) + 0.25 * row['security_score'] + 0.2 * determinism
            rows.append({'mode': row['mode'], 'latency_norm': row['latency_ms'] / lat_ref, 'energy_norm': row['energy_nj'] / energy_ref, 'security_score': row['security_score'], 'determinism_score': determinism, 'balanced_score': balanced})
        return pd.DataFrame(rows)

    def _semantic_signature(self, graphs: Sequence[TaskGraph]) -> str:
        values = []
        for graph in graphs:
            for frag in graph.topological_fragments():
                values.append(frag.semantic_group or frag.fragment_id)
        payload = json.dumps(values, sort_keys=False)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _schedule_signature(self, events: pd.DataFrame) -> str:
        if events.empty:
            return '0'
        dispatch = events[events['event'] == 'dispatch'].copy()
        payload = [f'{row.template_family}:{row.rfu_id}:{int(row.reused_result)}' for row in dispatch.itertuples(index=False)]
        return hashlib.sha256('|'.join(payload).encode()).hexdigest()[:16]

    def _replicated_runtime_tasks(self, graphs: Sequence[TaskGraph], repeats: int, prefix: str) -> List[RuntimeTask]:
        runtime_tasks: List[RuntimeTask] = []
        arrival_ms = 0.0
        gap = 0.25 if len(graphs) > 1 else 0.35
        for rep in range(repeats):
            for index, graph in enumerate(graphs):
                runtime_tasks.append(RuntimeTask(runtime_id=f'{prefix}_r{rep}_g{index}', task_graph=graph, arrival_cycle=int(arrival_ms * CYCLES_PER_MS)))
                arrival_ms += gap + 0.05 * (index % 3)
        return runtime_tasks

    def _task_latencies(self, events: pd.DataFrame, runtime_tasks: Sequence[RuntimeTask]) -> List[float]:
        arrival = {task.runtime_id: task.arrival_cycle for task in runtime_tasks}
        if events.empty:
            return []
        complete = events[events['event'] == 'complete'].copy()
        if complete.empty:
            return []
        finish = complete.groupby('task_runtime_id', as_index=False)['cycle'].max()
        latencies = []
        for row in finish.itertuples(index=False):
            start = arrival.get(row.task_runtime_id, 0)
            latencies.append((float(row.cycle) - float(start)) / CYCLES_PER_MS)
        return latencies

    def _case_definitions(self) -> List[Dict[str, object]]:
        rag_graphs = [build_transformer_block_task(seq_len=1024, hidden_dim=3072, heads=24, task_id='rag_llm'), build_aes128_task(blocks=1024, task_id='rag_aes'), build_groth16_like_task(circuit_size=2048, rounds=4, task_id='rag_zkp')]
        rollup_graphs = [build_groth16_like_task(circuit_size=4096, rounds=4, task_id='rollup_zkp'), build_transformer_block_task(seq_len=768, hidden_dim=2048, heads=16, task_id='rollup_llm'), build_aes128_task(blocks=2048, task_id='rollup_aes')]
        vla_graphs = [build_vgg16_like_task(batch_size=4, task_id='vla_cnn'), build_transformer_block_task(seq_len=768, hidden_dim=2048, heads=16, task_id='vla_llm'), build_aes128_task(blocks=512, task_id='vla_aes')]
        edge_graphs = [build_transformer_block_task(seq_len=384, hidden_dim=1536, heads=12, task_id='edge_llm'), build_mlp_path_task(batch_size=16, hidden_dim=1536, task_id='edge_mlp'), build_aes128_task(blocks=512, task_id='edge_aes'), build_groth16_like_task(circuit_size=1024, rounds=4, task_id='edge_zkp')]
        return [{'name': 'RAG Gateway', 'graphs': rag_graphs, 'runtime_tasks': self._make_case_runtime_tasks(rag_graphs, [2, 3, 2], [0.45, 0.18, 0.35]), 'slo_ms': 95.0}, {'name': 'Rollup Coprocessor', 'graphs': rollup_graphs, 'runtime_tasks': self._make_case_runtime_tasks(rollup_graphs, [2, 2, 4], [0.4, 0.5, 0.14]), 'slo_ms': 120.0}, {'name': 'Vision-Language Agent', 'graphs': vla_graphs, 'runtime_tasks': self._make_case_runtime_tasks(vla_graphs, [2, 2, 3], [0.3, 0.4, 0.12]), 'slo_ms': 110.0}, {'name': 'Secure Edge Assistant', 'graphs': edge_graphs, 'runtime_tasks': self._make_case_runtime_tasks(edge_graphs, [3, 3, 4, 2], [0.2, 0.09, 0.08, 0.25]), 'slo_ms': 80.0}]

    def _make_case_runtime_tasks(self, graphs: Sequence[TaskGraph], counts: Sequence[int], gaps: Sequence[float]) -> List[RuntimeTask]:
        runtime_tasks: List[RuntimeTask] = []
        arrival = 0.0
        for graph, count, gap in zip(graphs, counts, gaps):
            for idx in range(count):
                runtime_tasks.append(RuntimeTask(runtime_id=f'{graph.task_id}_{idx}', task_graph=graph, arrival_cycle=int(arrival * CYCLES_PER_MS)))
                arrival += gap * (1.0 + 0.06 * (idx % 3))
        runtime_tasks.sort(key=lambda task: task.arrival_cycle)
        return runtime_tasks

def run_sosp_suite(output_dir: Path, engine: str='cpp', calibration_path: Path | None=None) -> SOSPArtifacts:
    suite = SOSPSuite(engine=engine, calibration=load_calibration(calibration_path))
    return suite.run(Path(output_dir))
