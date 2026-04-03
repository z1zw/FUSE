from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence
import json
import numpy as np
import pandas as pd
from .benchmarks import BenchmarkArtifacts, load_calibration
from .branding import ACCELERATOR_NAME
from .case_studies import CaseStudyCatalog, CaseStudyDefinition
from .ir import RuntimeTask, TaskGraph
from .security import FEATURE_COLUMNS, binary_security_score, mutual_information_binary, pearson_corr, run_binary_attacks, trace_features
from .sim.simulator import CYCLES_PER_MS
from .sosp import SOSPArtifacts, SOSPSuite

@dataclass
class SOSPPlusArtifacts:
    base: BenchmarkArtifacts
    dataframes: Dict[str, pd.DataFrame]
    rawframes: Dict[str, pd.DataFrame]

class SOSPPlusSuite(SOSPSuite):

    def __init__(self, engine: str='cpp', calibration: Dict[str, Dict[str, float]] | None=None, seed: int=7):
        super().__init__(engine=engine, calibration=calibration, seed=seed)
        self.catalog = CaseStudyCatalog(self.frontend)

    def run(self, output_dir: Path) -> SOSPPlusArtifacts:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = output_dir / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        base = self._load_or_run_base(output_dir, load_raw=False)
        self._persist_frames(output_dir, base.dataframes)
        manifest = self.catalog.manifest()
        self._persist_frames(output_dir, {'case_manifest': manifest})
        security_frames = self.run_security(output_dir)
        self._persist_frames(output_dir, {'fig13_security': security_frames['fig13_security'], 'fig14_llm_stability': security_frames['fig14_llm_stability'], 'fig15_secure_tradeoff': security_frames['fig15_secure_tradeoff']})
        self._persist_raw_frames(raw_dir, security_frames, skip={'fig13_security', 'fig14_llm_stability', 'fig15_secure_tradeoff'})
        case_frames = self.run_case_studies(output_dir)
        self._persist_frames(output_dir, {'fig16_case_studies': case_frames['fig16_case_studies'], 'fig18_service_breakdown': case_frames['fig18_service_breakdown']})
        self._persist_raw_frames(raw_dir, case_frames, skip={'fig16_case_studies', 'fig18_service_breakdown'})
        load_frames = self.run_load_sweep(output_dir)
        self._persist_frames(output_dir, {'fig19_load_sweep': load_frames['fig19_load_sweep']})
        self._persist_raw_frames(raw_dir, load_frames, skip={'fig19_load_sweep'})
        ablation_frames = self.run_ablations(output_dir)
        self._persist_frames(output_dir, {'fig17_ablation_plus': ablation_frames['fig17_ablation_plus']})
        self._persist_raw_frames(raw_dir, ablation_frames, skip={'fig17_ablation_plus'})
        secure_component_frames = self.run_secure_components(output_dir)
        self._persist_frames(output_dir, {'fig21_secure_components': secure_component_frames['fig21_secure_components']})
        self._persist_raw_frames(raw_dir, secure_component_frames, skip={'fig21_secure_components'})
        compare_frames = self.run_extended_compare(output_dir, security_frames, case_frames, load_frames)
        self._persist_frames(output_dir, {'fig20_extended_compare': compare_frames['fig20_extended_compare']})
        self._persist_raw_frames(raw_dir, compare_frames, skip={'fig20_extended_compare'})
        table = self.build_summary_table(base, security_frames, case_frames, load_frames, ablation_frames, secure_component_frames, compare_frames)
        self._persist_frames(output_dir, {'table1_sosp_summary': table})
        dataframes = {**base.dataframes, 'fig13_security': security_frames['fig13_security'], 'fig14_llm_stability': security_frames['fig14_llm_stability'], 'fig15_secure_tradeoff': security_frames['fig15_secure_tradeoff'], 'fig16_case_studies': case_frames['fig16_case_studies'], 'fig17_ablation_plus': ablation_frames['fig17_ablation_plus'], 'fig18_service_breakdown': case_frames['fig18_service_breakdown'], 'fig19_load_sweep': load_frames['fig19_load_sweep'], 'fig20_extended_compare': compare_frames['fig20_extended_compare'], 'fig21_secure_components': secure_component_frames['fig21_secure_components'], 'case_manifest': manifest, 'table1_sosp_summary': table}
        rawframes = {**base.rawframes, **security_frames, **case_frames, **load_frames, **ablation_frames, **secure_component_frames, **compare_frames, 'case_manifest': manifest, 'table1_sosp_summary': table}
        return SOSPPlusArtifacts(base=base, dataframes=dataframes, rawframes=rawframes)

    def run_security(self, output_dir: Path) -> Dict[str, pd.DataFrame]:
        raw_dir = Path(output_dir) / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        mode_specs = [('Insecure LLM', False, 'morphaccel'), ('Canonical IR', True, 'morphaccel'), ('Secure-CT', True, 'secure_ct'), ('Secure-Rand', True, 'secure_randomized'), ('Secure-Elastic', True, 'secure_elastic'), ('Multi-core', True, 'multicore'), ('GPU-Tensor', True, 'gpu_tensor')]
        feature_map = {'latency': 'latency_ms', 'reuse': 'reuse_rate', 'schedule': 'template_switches', 'fusion': None}
        trial_rows: List[Dict[str, object]] = []
        stability_rows: List[Dict[str, object]] = []
        variant_catalog = self.frontend.variant_prompts()
        selected_intents = ['zkp', 'vision', 'crypto', 'llm', 'hybrid', 'service']
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
                    fusion_signal = float(pd.DataFrame([{k: features.get(k, 0.0) for k in FEATURE_COLUMNS}])[FEATURE_COLUMNS].mean(axis=1).iloc[0])
                    semantic_sig = self._semantic_signature(compiled.graphs)
                    schedule_sig = self._schedule_signature(events)
                    semantic_sigs.append(semantic_sig)
                    schedule_sigs.append(schedule_sig)
                    canonical_hashes.append(compiled.canonical_hash)
                    latencies.append(float(features['latency_ms']))
                    trial_rows.append({'mode': mode_name, 'sim_mode': sim_mode, 'intent': intent, 'label': 1 if variant.label == 'beta' else 0, 'prompt_id': idx, 'prompt_hash': compiled.prompt_hash, 'canonical_hash': compiled.canonical_hash, 'semantic_signature': semantic_sig, 'schedule_signature': schedule_sig, 'verified': 1 if verify.ok else 0, **features, 'fusion_signal': fusion_signal})
                latency_mean = float(np.mean(latencies)) if latencies else 0.0
                stability_rows.append({'mode': mode_name, 'intent': intent, 'unique_semantic_ratio': len(set(semantic_sigs)) / max(1, len(semantic_sigs)), 'unique_schedule_ratio': len(set(schedule_sigs)) / max(1, len(schedule_sigs)), 'canonical_consistency': 1.0 / max(1, len(set(canonical_hashes))), 'latency_cv': float(np.std(latencies) / max(latency_mean, 1e-09)) if latencies else 0.0, 'verified_pass_rate': float(np.mean(verified)) if verified else 0.0})
        trials = pd.DataFrame(trial_rows)
        stability = pd.DataFrame(stability_rows)
        attack_rows: List[Dict[str, object]] = []
        for mode_name, part in trials.groupby('mode'):
            for attack in run_binary_attacks(part, label_col='label'):
                source = part[feature_map[attack.attack]] if feature_map[attack.attack] else part['fusion_signal']
                mi = mutual_information_binary(part['label'], source)
                corr = pearson_corr(part['label'], source)
                attack_rows.append({'mode': mode_name, 'attack': attack.attack, 'accuracy': attack.accuracy, 'mutual_information': mi, 'correlation': corr, 'security_score': binary_security_score(attack.accuracy, mi, corr)})
        fig13 = pd.DataFrame(attack_rows)
        fig15 = self._build_tradeoff_table(fig13, stability, trials)
        return {'fig13_security': fig13, 'fig14_llm_stability': stability, 'fig15_secure_tradeoff': fig15, 'fig13_security_trials': trials}

    def _build_tradeoff_table(self, fig13: pd.DataFrame, stability: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
        perf = trials.groupby('mode', as_index=False)[['latency_ms', 'energy_nj']].mean()
        sec = fig13.groupby('mode', as_index=False)['security_score'].mean()
        stab = stability.groupby('mode', as_index=False).agg(canonical_consistency=('canonical_consistency', 'mean'), latency_cv=('latency_cv', 'mean'), verified_pass_rate=('verified_pass_rate', 'mean'), unique_semantic_ratio=('unique_semantic_ratio', 'mean'))
        merged = perf.merge(sec, on='mode').merge(stab, on='mode')
        ref_latency = float(merged[merged['mode'] == 'Canonical IR']['latency_ms'].iloc[0]) if (merged['mode'] == 'Canonical IR').any() else float(merged['latency_ms'].median())
        ref_energy = float(merged[merged['mode'] == 'Canonical IR']['energy_nj'].iloc[0]) if (merged['mode'] == 'Canonical IR').any() else float(merged['energy_nj'].median())
        rows = []
        for row in merged.to_dict(orient='records'):
            determinism = max(0.0, min(1.0, 0.45 * row['canonical_consistency'] + 0.35 * row['verified_pass_rate'] + 0.2 * (1.0 - min(1.0, row['latency_cv'] / 0.25))))
            latency_term = ref_latency / max(row['latency_ms'], 1e-09)
            energy_term = ref_energy / max(row['energy_nj'], 1e-09)
            balanced = 0.45 * latency_term + 0.15 * energy_term + 0.25 * row['security_score'] + 0.15 * determinism
            rows.append({'mode': row['mode'], 'latency_norm': row['latency_ms'] / max(ref_latency, 1e-09), 'energy_norm': row['energy_nj'] / max(ref_energy, 1e-09), 'security_score': row['security_score'], 'determinism_score': determinism, 'balanced_score': balanced})
        return pd.DataFrame(rows)

    def run_case_studies(self, output_dir: Path) -> Dict[str, pd.DataFrame]:
        raw_dir = Path(output_dir) / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        modes = [(ACCELERATOR_NAME, 'morphaccel'), ('Secure-Elastic', 'secure_elastic'), ('Secure-CT', 'secure_ct'), ('Secure-Rand', 'secure_randomized'), ('Multi-core', 'multicore'), ('GPU-Tensor', 'gpu_tensor'), ('Transformer-ASIC', 'transformer_asic')]
        summary_rows: List[Dict[str, object]] = []
        breakdown_rows: List[Dict[str, object]] = []
        raw_rows: List[Dict[str, object]] = []
        timeline_rows: List[Dict[str, object]] = []
        for case in self.catalog.build():
            runtime_tasks = case.build_runtime_tasks(load_factor=1.0)
            for mode_name, sim_mode in modes:
                scenario_name = f'fig16_{case.slug}_{sim_mode}'
                result, events = self.base_suite._simulate(case.graphs, runtime_tasks, mode=sim_mode, scenario_name=scenario_name, raw_dir=raw_dir)
                latency_map = self._task_latency_map(events, runtime_tasks)
                e2e_samples = self._compose_service_samples(case, runtime_tasks, latency_map)
                stage_rows = self._stage_latency_rows(case, runtime_tasks, latency_map, mode_name)
                for row in stage_rows:
                    breakdown_rows.append({'case': case.name, **row})
                for runtime_id, value in latency_map.items():
                    task = next((task for task in runtime_tasks if task.runtime_id == runtime_id))
                    raw_rows.append({'case': case.name, 'mode': mode_name, 'runtime_id': runtime_id, 'task_type': self._task_stage_name(task), 'latency_ms': value})
                for row in self._timeline(case.name, mode_name, events):
                    timeline_rows.append(row)
                features = trace_features(events, result.metrics)
                p50 = float(np.percentile(e2e_samples, 50)) if e2e_samples else 0.0
                p99 = float(np.percentile(e2e_samples, 99)) if e2e_samples else 0.0
                throughput = len(e2e_samples) / max(float(result.metrics.get('makespan_ms', 0.0)) / 1000.0, 1e-09)
                slo_miss = float(np.mean(np.asarray(e2e_samples) > case.slo_ms)) if e2e_samples else 0.0
                energy_per_request = float(result.metrics.get('energy_nj', 0.0)) / max(1, len(e2e_samples))
                stage_df = pd.DataFrame(stage_rows)
                critical = float(stage_df['p95_ms'].sum()) if not stage_df.empty else p99
                summary_rows.append({'case': case.name, 'mode': mode_name, 'p50_ms': p50, 'p99_ms': p99, 'critical_path_ms': critical, 'throughput_tps': throughput, 'energy_per_request_nj': energy_per_request, 'slo_miss_ratio': slo_miss, 'rfu_utilization': float(result.metrics.get('rfu_utilization', 0.0)), 'reuse_rate': features['reuse_rate'], 'storyline': case.storyline, 'focus': case.focus, 'slo_ms': case.slo_ms})
        return {'fig16_case_studies': pd.DataFrame(summary_rows), 'fig18_service_breakdown': pd.DataFrame(breakdown_rows), 'fig16_case_latencies': pd.DataFrame(raw_rows), 'fig16_case_timeline': pd.DataFrame(timeline_rows)}

    def run_load_sweep(self, output_dir: Path) -> Dict[str, pd.DataFrame]:
        raw_dir = Path(output_dir) / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        modes = [(ACCELERATOR_NAME, 'morphaccel'), ('Secure-Elastic', 'secure_elastic'), ('Secure-CT', 'secure_ct'), ('Multi-core', 'multicore'), ('GPU-Tensor', 'gpu_tensor')]
        rows: List[Dict[str, object]] = []
        for case in self.catalog.build():
            for load_factor in case.load_factors:
                runtime_tasks = case.build_runtime_tasks(load_factor=load_factor)
                for mode_name, sim_mode in modes:
                    scenario_name = f"fig19_{case.slug}_{str(load_factor).replace('.', 'p')}_{sim_mode}"
                    result, events = self.base_suite._simulate(case.graphs, runtime_tasks, mode=sim_mode, scenario_name=scenario_name, raw_dir=raw_dir)
                    latency_map = self._task_latency_map(events, runtime_tasks)
                    e2e_samples = self._compose_service_samples(case, runtime_tasks, latency_map)
                    rows.append({'case': case.name, 'mode': mode_name, 'load_factor': load_factor, 'p99_ms': float(np.percentile(e2e_samples, 99)) if e2e_samples else 0.0, 'p50_ms': float(np.percentile(e2e_samples, 50)) if e2e_samples else 0.0, 'throughput_tps': len(e2e_samples) / max(float(result.metrics.get('makespan_ms', 0.0)) / 1000.0, 1e-09), 'slo_miss_ratio': float(np.mean(np.asarray(e2e_samples) > case.slo_ms)) if e2e_samples else 0.0, 'stall_cycles': float(result.metrics.get('scheduler_pipeline_stall_cycles', 0.0)), 'energy_per_request_nj': float(result.metrics.get('energy_nj', 0.0)) / max(1, len(e2e_samples))})
        return {'fig19_load_sweep': pd.DataFrame(rows)}

    def run_ablations(self, output_dir: Path) -> Dict[str, pd.DataFrame]:
        raw_dir = Path(output_dir) / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        prompts = {'Groth16': 'compile a groth16 prover for a 4k-gate rollup circuit', 'Transformer': 'compile a transformer block for llm prefill seq 1024 hidden 3072 heads 24', 'Hybrid': 'build a hybrid agent pipeline with llm prefill seq 512 aes 512 and groth16 2048', 'RAG': 'build a private rag gateway with llm prefill seq 768 hidden 3072 heads 24 rerank mlp hidden 2048 aes 1024 groth16 2048'}
        configs = [(f'Full {ACCELERATOR_NAME}', True, 'morphaccel'), ('- Canonical IR', False, 'morphaccel'), ('- FVM reuse', True, 'no_fvm'), ('- LRC prefetch', True, 'no_prefetch'), ('- TARS scorer', True, 'fixed_schedule'), ('Secure-Rand', True, 'secure_randomized'), ('Secure-CT', True, 'secure_ct'), ('Secure-Elastic', True, 'secure_elastic')]
        rows: List[Dict[str, object]] = []
        for workload, prompt in prompts.items():
            latency_ref = None
            energy_ref = None
            for config_name, secure, sim_mode in configs:
                graphs = self.frontend.compile_prompt(prompt, secure=secure, task_prefix=f"{workload.lower()}_{config_name.lower().replace(' ', '_').replace('-', '_')}").graphs
                runtime_tasks = self._replicated_runtime_tasks(graphs, repeats=2 if len(graphs) == 1 else 1, prefix=f'{workload}_{sim_mode}')
                scenario_name = f"fig17_{workload.lower()}_{config_name.lower().replace(' ', '_').replace('-', '_')}"
                result, events = self.base_suite._simulate(graphs, runtime_tasks, mode=sim_mode, scenario_name=scenario_name, raw_dir=raw_dir)
                features = trace_features(events, result.metrics)
                leakage_proxy = min(1.0, 0.35 * features['reuse_rate'] + 0.2 * features['structural_reuse_rate'] + 0.2 * features['template_switches'] + 0.15 * features['rfu_entropy'] + 0.1 * features['template_entropy'])
                latency = float(result.metrics.get('makespan_ms', 0.0))
                energy = float(result.metrics.get('energy_nj', 0.0))
                if config_name == f'Full {ACCELERATOR_NAME}':
                    latency_ref = max(latency, 1e-09)
                    energy_ref = max(energy, 1e-09)
                rows.append({'workload': workload, 'configuration': config_name, 'latency_norm': latency / max(latency_ref or latency, 1e-09), 'energy_norm': energy / max(energy_ref or energy, 1e-09), 'leakage_proxy': leakage_proxy, 'rfu_utilization': float(result.metrics.get('rfu_utilization', 0.0))})
        return {'fig17_ablation_plus': pd.DataFrame(rows)}

    def run_secure_components(self, output_dir: Path) -> Dict[str, pd.DataFrame]:
        raw_dir = Path(output_dir) / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        prompts = {'Rollup': 'compile a rollup sequencer service with groth16 4096 aes 2048 transformer seq 512 hidden 2048 heads 16', 'RAG': 'build a private rag gateway with llm prefill seq 768 hidden 3072 heads 24 rerank mlp hidden 2048 aes 1024 groth16 2048', 'Edge': 'compile a confidential edge copilot with llm prefill seq 384 hidden 1536 heads 12 rerank mlp hidden 1024 aes 512 groth16 1024'}
        configs = [('Secure-Elastic', 'secure_elastic'), ('- Bucketization', 'secure_elastic_no_bucket'), ('- Family pinning', 'secure_elastic_no_pinning'), ('- RFU masking', 'secure_elastic_no_mask'), ('- Epoch rotation', 'secure_elastic_no_epoch'), ('- Prefetch', 'secure_elastic_no_prefetch'), ('Secure-CT', 'secure_ct'), ('Secure-Rand', 'secure_randomized'), (ACCELERATOR_NAME, 'morphaccel')]
        rows: List[Dict[str, object]] = []
        for workload, prompt in prompts.items():
            latency_ref = None
            energy_ref = None
            for config_name, sim_mode in configs:
                graphs = self.frontend.compile_prompt(prompt, secure=True, task_prefix=f"secure_{workload.lower()}_{config_name.lower().replace(' ', '_').replace('-', '_')}").graphs
                runtime_tasks = self._replicated_runtime_tasks(graphs, repeats=1, prefix=f'{workload}_{sim_mode}')
                scenario_name = f"fig21_{workload.lower()}_{config_name.lower().replace(' ', '_').replace('-', '_')}"
                result, events = self.base_suite._simulate(graphs, runtime_tasks, mode=sim_mode, scenario_name=scenario_name, raw_dir=raw_dir)
                features = trace_features(events, result.metrics)
                leakage_proxy = min(1.0, 0.3 * features['reuse_rate'] + 0.25 * features['structural_reuse_rate'] + 0.2 * features['template_switches'] + 0.15 * features['rfu_entropy'] + 0.1 * features['template_entropy'])
                latency = float(result.metrics.get('makespan_ms', 0.0))
                energy = float(result.metrics.get('energy_nj', 0.0))
                if config_name == 'Secure-Elastic':
                    latency_ref = max(latency, 1e-09)
                    energy_ref = max(energy, 1e-09)
                balance = 0.6 * (1.0 / max(latency / max(latency_ref or latency, 1e-09), 1e-09)) + 0.4 * (1.0 - leakage_proxy)
                rows.append({'workload': workload, 'configuration': config_name, 'latency_norm': latency / max(latency_ref or latency, 1e-09), 'energy_norm': energy / max(energy_ref or energy, 1e-09), 'leakage_proxy': leakage_proxy, 'balance_score': balance})
        return {'fig21_secure_components': pd.DataFrame(rows)}

    def run_extended_compare(self, output_dir: Path, security_frames: Dict[str, pd.DataFrame], case_frames: Dict[str, pd.DataFrame], load_frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        fig16 = case_frames['fig16_case_studies']
        fig13 = security_frames['fig13_security']
        fig15 = security_frames['fig15_secure_tradeoff']
        fig19 = load_frames['fig19_load_sweep']
        security_mean = fig13.groupby('mode', as_index=False)['security_score'].mean()
        det_mean = fig15[['mode', 'determinism_score', 'balanced_score']]
        baseline = fig16[fig16['mode'] == 'Multi-core'].set_index('case')
        peak_load = fig19[fig19['load_factor'] == fig19['load_factor'].max()].copy()
        peak_baseline = peak_load[peak_load['mode'] == 'Multi-core'].set_index('case')
        rows: List[Dict[str, object]] = []
        for case, part in fig16.groupby('case'):
            base = baseline.loc[case]
            peak = peak_baseline.loc[case]
            for row in part.to_dict(orient='records'):
                peak_row = peak_load[(peak_load['case'] == case) & (peak_load['mode'] == row['mode'])]
                peak_entry = peak_row.iloc[0] if not peak_row.empty else None
                sec = security_mean[security_mean['mode'] == row['mode']]
                det = det_mean[det_mean['mode'] == row['mode']]
                rows.append({'case': case, 'mode': row['mode'], 'latency_gain_vs_multicore': float(base['p99_ms'] / max(row['p99_ms'], 1e-09)), 'energy_gain_vs_multicore': float(base['energy_per_request_nj'] / max(row['energy_per_request_nj'], 1e-09)), 'throughput_gain_vs_multicore': float(row['throughput_tps'] / max(base['throughput_tps'], 1e-09)), 'slo_gain_vs_multicore': float(base['slo_miss_ratio'] / max(row['slo_miss_ratio'], 1e-09)) if row['slo_miss_ratio'] > 0 else float(base['slo_miss_ratio'] == 0.0), 'peak_load_p99_gain': float(peak['p99_ms'] / max(float(peak_entry['p99_ms']) if peak_entry is not None else row['p99_ms'], 1e-09)), 'security_score': float(sec.iloc[0]['security_score']) if not sec.empty else np.nan, 'determinism_score': float(det.iloc[0]['determinism_score']) if not det.empty else np.nan, 'balanced_score': float(det.iloc[0]['balanced_score']) if not det.empty else np.nan})
        return {'fig20_extended_compare': pd.DataFrame(rows)}

    def build_summary_table(self, base: BenchmarkArtifacts, security_frames: Dict[str, pd.DataFrame], case_frames: Dict[str, pd.DataFrame], load_frames: Dict[str, pd.DataFrame], ablation_frames: Dict[str, pd.DataFrame], secure_component_frames: Dict[str, pd.DataFrame], compare_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        fig4 = base.dataframes['fig4_latency']
        for domain, part in fig4.groupby('domain'):
            rows.append({'category': 'domain', 'name': domain, 'metric': 'mean_speedup_vs_baseline', 'value': float((part['baseline_ms'] / part['morph_ms']).mean())})
        fig13 = security_frames['fig13_security']
        fig15 = security_frames['fig15_secure_tradeoff']
        for mode, part in fig13.groupby('mode'):
            rows.append({'category': 'security', 'name': mode, 'metric': 'mean_security_score', 'value': float(part['security_score'].mean())})
        for row in fig15.to_dict(orient='records'):
            rows.append({'category': 'security', 'name': row['mode'], 'metric': 'balanced_score', 'value': float(row['balanced_score'])})
        fig16 = case_frames['fig16_case_studies']
        baseline = fig16[fig16['mode'] == 'Multi-core'].set_index('case')
        for case, part in fig16.groupby('case'):
            if case not in baseline.index:
                continue
            base_row = baseline.loc[case]
            elastic = part[part['mode'] == 'Secure-Elastic']
            if not elastic.empty:
                elastic = elastic.iloc[0]
                rows.append({'category': 'case', 'name': case, 'metric': 'secure_elastic_p99_gain_vs_multicore', 'value': float(base_row['p99_ms'] / max(elastic['p99_ms'], 1e-09))})
                rows.append({'category': 'case', 'name': case, 'metric': 'secure_elastic_energy_gain_vs_multicore', 'value': float(base_row['energy_per_request_nj'] / max(elastic['energy_per_request_nj'], 1e-09))})
        fig19 = load_frames['fig19_load_sweep']
        peak = fig19[fig19['load_factor'] == fig19['load_factor'].max()]
        peak_base = peak[peak['mode'] == 'Multi-core'].set_index('case')
        for case, part in peak.groupby('case'):
            elastic = part[part['mode'] == 'Secure-Elastic']
            if not elastic.empty and case in peak_base.index:
                base_row = peak_base.loc[case]
                elastic = elastic.iloc[0]
                rows.append({'category': 'load', 'name': case, 'metric': 'peakload_secure_elastic_p99_gain_vs_multicore', 'value': float(base_row['p99_ms'] / max(elastic['p99_ms'], 1e-09))})
                rows.append({'category': 'load', 'name': case, 'metric': 'peakload_secure_elastic_slo_miss_ratio', 'value': float(elastic['slo_miss_ratio'])})
        fig17 = ablation_frames['fig17_ablation_plus']
        for workload, part in fig17.groupby('workload'):
            secure = part[part['configuration'] == 'Secure-Elastic']
            if not secure.empty:
                secure = secure.iloc[0]
                rows.append({'category': 'ablation', 'name': workload, 'metric': 'secure_elastic_latency_norm_vs_morphaccel', 'value': float(secure['latency_norm'])})
        fig21 = secure_component_frames['fig21_secure_components']
        for workload, part in fig21.groupby('workload'):
            best = part.sort_values('balance_score', ascending=False).iloc[0]
            rows.append({'category': 'secure_component', 'name': workload, 'metric': 'best_balance_configuration', 'value': best['configuration']})
        fig20 = compare_frames['fig20_extended_compare']
        for mode, part in fig20.groupby('mode'):
            rows.append({'category': 'compare', 'name': mode, 'metric': 'mean_latency_gain_vs_multicore', 'value': float(part['latency_gain_vs_multicore'].mean())})
        return pd.DataFrame(rows)

    def _task_latency_map(self, events: pd.DataFrame, runtime_tasks: Sequence[RuntimeTask]) -> Dict[str, float]:
        arrival = {task.runtime_id: task.arrival_cycle for task in runtime_tasks}
        if events.empty:
            return {}
        complete = events[events['event'] == 'complete'].copy()
        if complete.empty:
            return {}
        finish = complete.groupby('task_runtime_id', as_index=False)['cycle'].max()
        out: Dict[str, float] = {}
        for row in finish.itertuples(index=False):
            out[str(row.task_runtime_id)] = (float(row.cycle) - float(arrival.get(row.task_runtime_id, 0))) / CYCLES_PER_MS
        return out

    def _task_stage_name(self, task: RuntimeTask) -> str:
        model = str(task.task_graph.metadata.get('model', task.task_graph.task_type))
        if task.task_graph.task_type == 'LLM':
            return 'LLM'
        if task.task_graph.task_type == 'AES':
            return 'AES'
        if task.task_graph.task_type == 'ZKP':
            return 'ZKP'
        if model == 'MLP-path':
            return 'MLP'
        return str(task.task_graph.task_type)

    def _compose_service_samples(self, case: CaseStudyDefinition, runtime_tasks: Sequence[RuntimeTask], latency_map: Dict[str, float]) -> List[float]:
        by_graph: List[List[float]] = []
        for graph in case.graphs:
            values = [latency_map[task.runtime_id] for task in runtime_tasks if task.task_graph.task_id == graph.task_id and task.runtime_id in latency_map]
            if values:
                by_graph.append(values)
        if not by_graph:
            return []
        n = max((len(values) for values in by_graph))
        samples = []
        for idx in range(n):
            total = 0.0
            for values in by_graph:
                total += values[min(idx, len(values) - 1)]
            samples.append(total)
        return samples

    def _stage_latency_rows(self, case: CaseStudyDefinition, runtime_tasks: Sequence[RuntimeTask], latency_map: Dict[str, float], mode_name: str) -> List[Dict[str, object]]:
        total = 0.0
        stage_rows: List[Dict[str, object]] = []
        for order, graph in enumerate(case.graphs):
            values = [latency_map[task.runtime_id] for task in runtime_tasks if task.task_graph.task_id == graph.task_id and task.runtime_id in latency_map]
            if not values:
                continue
            mean_ms = float(np.mean(values))
            total += mean_ms
            stage_rows.append({'mode': mode_name, 'stage': self._task_stage_name(next((task for task in runtime_tasks if task.task_graph.task_id == graph.task_id))), 'order': order, 'mean_ms': mean_ms, 'p95_ms': float(np.percentile(values, 95)), 'count': len(values)})
        if total > 0.0:
            for row in stage_rows:
                row['share'] = row['mean_ms'] / total
        else:
            for row in stage_rows:
                row['share'] = 0.0
        return stage_rows

    def _timeline(self, case_name: str, mode_name: str, events: pd.DataFrame, bucket_ms: float=0.5) -> List[Dict[str, object]]:
        if events.empty:
            return []
        dispatch = events[events['event'] == 'dispatch'].copy()
        if dispatch.empty or 'finish_cycle' not in dispatch.columns:
            return []
        dispatch['start_ms'] = dispatch['cycle'].astype(float) / CYCLES_PER_MS
        dispatch['finish_ms'] = dispatch['finish_cycle'].astype(float) / CYCLES_PER_MS
        max_time = float(dispatch['finish_ms'].max()) if not dispatch.empty else 0.0
        bucket_edges = np.arange(0.0, max_time + bucket_ms, bucket_ms)
        rows: List[Dict[str, object]] = []
        for start in bucket_edges:
            end = start + bucket_ms
            active = int(((dispatch['start_ms'] < end) & (dispatch['finish_ms'] >= start)).sum())
            if active == 0 and start > max_time:
                continue
            rows.append({'case': case_name, 'mode': mode_name, 'bucket_start_ms': round(float(start), 4), 'active_ops': active})
        return rows

def run_sosp_plus_suite(output_dir: Path, engine: str='cpp', calibration_path: Path | None=None) -> SOSPPlusArtifacts:
    suite = SOSPPlusSuite(engine=engine, calibration=load_calibration(calibration_path))
    return suite.run(Path(output_dir))
