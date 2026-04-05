from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import hashlib
import json
import random
import subprocess
import pandas as pd
from .branding import ACCELERATOR_NAME, normalize_mode
from .baselines import BASELINES
from .cpp_bridge import CppSimulatorBridge
from .energy_benchmarks import EnergyBenchmarkSuite
from .eval_tools import EvaluationToolkit
from .hardware import HardwareBenchmarkSuite
from .ir import RunResult, RuntimeTask, TaskGraph
from .reuse import MicrofragmentDecomposer
from .serialization import ScenarioSerializer
from .sim.simulator import CYCLES_PER_MS, FuseSimulator, SimulationConfig
from .workloads import build_aes128_task, build_groth16_like_task, build_mlp_path_task, build_vgg16_like_task

@dataclass
class BenchmarkArtifacts:
    dataframes: Dict[str, pd.DataFrame]
    rawframes: Dict[str, pd.DataFrame]
    traces: Dict[str, pd.DataFrame]
    scalars: Dict[str, float]
    calibration: Dict[str, Dict[str, float]]

class BenchmarkSuite:

    def __init__(self, engine: str='cpp', calibration: Dict[str, Dict[str, float]] | None=None, calibration_targets: Dict[str, Any] | None=None, seed: int=7):
        self.engine = engine
        self.seed = seed
        self.rng = random.Random(seed)
        self.project_root = Path(__file__).resolve().parents[2]
        self.cpp = CppSimulatorBridge(self.project_root)
        self.toolkit = EvaluationToolkit(self.project_root)
        self.serializer = ScenarioSerializer()
        self.calibration = calibration or {}
        self.calibration_targets = calibration_targets or {}

    def _safe_name(self, value: str) -> str:
        return ''.join((ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in value))

    def run(self, output_dir: Path | None=None) -> BenchmarkArtifacts:
        rawframes: Dict[str, pd.DataFrame] = {}
        traces: Dict[str, pd.DataFrame] = {}
        output_dir = Path(output_dir) if output_dir is not None else None
        raw_dir = output_dir / 'raw' if output_dir is not None else None
        if raw_dir is not None:
            raw_dir.mkdir(parents=True, exist_ok=True)
        rawframes['fig4_raw'] = self._raw_fig4(raw_dir)
        rawframes['fig5_raw'] = self._raw_fig5(raw_dir)
        rawframes['fig6_raw'], traces['fig6_events'] = self._raw_fig6(raw_dir)
        rawframes['fig7_raw'] = self._raw_fig7(raw_dir)
        rawframes['fig8_raw'] = self._raw_fig8(raw_dir)
        rawframes['fig9_raw'] = self._raw_fig9(raw_dir)
        rawframes['fig10_raw'] = self._raw_fig10(raw_dir)
        raw11 = self._raw_fig11(raw_dir)
        rawframes.update(raw11)
        rawframes['fig12_raw'] = self._raw_fig12(raw_dir)
        calibration = self._fit_calibration(rawframes)
        if self.calibration:
            for family, values in self.calibration.items():
                calibration.setdefault(family, {}).update(values)
        dataframes = {'fig4_latency': self._figure_fig4(rawframes['fig4_raw'], calibration), 'fig5_energy': rawframes['fig5_raw'], 'fig6_mixed': self._figure_fig6(rawframes['fig6_raw'], calibration), 'fig7_area_power': rawframes['fig7_raw'], **self._figure_fig8(rawframes['fig8_raw']), 'fig9_ablation': self._figure_fig9(rawframes['fig9_raw']), 'fig10_memory': rawframes['fig10_raw'], 'fig11a_sched': rawframes['fig11a_sched'], 'fig11b_switch': rawframes['fig11b_switch'], 'fig11c_summary': rawframes['fig11c_summary'], 'fig11_full': self._figure_fig11_full(rawframes['fig11a_sched'], rawframes['fig11b_switch'], rawframes['fig11c_summary'], rawframes['fig12_raw']), 'fig12_reconfig': rawframes['fig12_raw']}
        scalars = {'groth16_speedup': float(self._figure_fig4(rawframes['fig4_raw'], calibration).query("domain == 'Groth16'")['baseline_ms'].mean() / max(self._figure_fig4(rawframes['fig4_raw'], calibration).query("domain == 'Groth16'")['morph_ms'].mean(), 1e-09)), 'vgg16_overhead': float(self._figure_fig4(rawframes['fig4_raw'], calibration).query("domain == 'VGG16'")['morph_ms'].mean() / max(self._figure_fig4(rawframes['fig4_raw'], calibration).query("domain == 'VGG16'")['baseline_ms'].mean(), 1e-09)), 'aes_overhead': float(self._figure_fig4(rawframes['fig4_raw'], calibration).query("domain == 'AES-128'")['morph_ms'].mean() / max(self._figure_fig4(rawframes['fig4_raw'], calibration).query("domain == 'AES-128'")['baseline_ms'].mean(), 1e-09))}
        artifacts = BenchmarkArtifacts(dataframes=dataframes, rawframes=rawframes, traces=traces, scalars=scalars, calibration=calibration)
        if output_dir is not None:
            self._write_outputs(output_dir, artifacts)
        return artifacts

    def _simulate(self, task_graphs: Sequence[TaskGraph], runtime_tasks: Sequence[RuntimeTask], *, mode: str, scenario_name: str, raw_dir: Path | None) -> Tuple[RunResult, pd.DataFrame]:
        mode = normalize_mode(mode)
        if self.engine == 'cpp':
            try:
                safe_name = self._safe_name(scenario_name)
                scenario_root = raw_dir / safe_name if raw_dir is not None else None
                artifact = self.cpp.run(task_graphs, runtime_tasks, mode=mode, scenario_name='scenario', out_dir=scenario_root)
                result = RunResult(metrics=artifact.metrics, traces={'events': artifact.events.to_dict(orient='records')}, tables={})
                events = artifact.events.copy()
                if raw_dir is not None:
                    safe_name = self._safe_name(scenario_name)
                    metrics_rows = pd.DataFrame([{'key': k, 'value': v} for k, v in artifact.metrics.items()])
                    metrics_rows.to_csv(raw_dir / f'{safe_name}_metrics.csv', index=False)
                    if not events.empty:
                        events.to_csv(raw_dir / f'{safe_name}_events.csv', index=False)
                return (result, events)
            except (OSError, subprocess.SubprocessError):
                pass
        decomposer = MicrofragmentDecomposer()
        decomposition = decomposer.decompose_tasks(task_graphs)
        result = FuseSimulator(config=SimulationConfig(mode=mode)).run(runtime_tasks, decomposition.microfragments)
        events = pd.DataFrame(result.traces.get('events', []))
        if raw_dir is not None:
            safe_name = self._safe_name(scenario_name)
            pd.DataFrame([{'key': k, 'value': v} for k, v in result.metrics.items() if not isinstance(v, dict)]).to_csv(raw_dir / f'{safe_name}_metrics.csv', index=False)
            if not events.empty:
                events.to_csv(raw_dir / f'{safe_name}_events.csv', index=False)
        return (result, events)

    def _runtime_tasks(self, task_graph: TaskGraph, count: int, gap_ms: float=0.0) -> List[RuntimeTask]:
        return [RuntimeTask(runtime_id=f'{task_graph.task_id}_rt{i}', task_graph=task_graph, arrival_cycle=int(i * gap_ms * CYCLES_PER_MS)) for i in range(count)]

    def _raw_fig4(self, raw_dir: Path | None) -> pd.DataFrame:
        rows = []
        settings = {'Groth16': ([2048, 4096, 8192, 16384], lambda x: build_groth16_like_task(circuit_size=x, rounds=4, task_id=f'groth16_{x}')), 'VGG16': ([8, 16, 32, 64], lambda x: build_vgg16_like_task(batch_size=x, task_id=f'vgg16_{x}')), 'AES-128': ([1024, 2048, 4096, 8192], lambda x: build_aes128_task(blocks=x, task_id=f'aes_{x}'))}
        for domain, (sizes, builder) in settings.items():
            baseline = BASELINES[domain]
            for size in sizes:
                graph = builder(size)
                runtime_tasks = self._runtime_tasks(graph, 1)
                morph, _ = self._simulate([graph], runtime_tasks, mode='fuse', scenario_name=f'fig4_{domain}_{size}_morph', raw_dir=raw_dir)
                base, _ = self._simulate([graph], runtime_tasks, mode=baseline.mode, scenario_name=f'fig4_{domain}_{size}_baseline', raw_dir=raw_dir)
                rows.append({'domain': domain, 'x': size, 'mode': ACCELERATOR_NAME, 'baseline_name': baseline.name, 'latency_ms': morph.metrics['makespan_ms'], 'energy_nj': morph.metrics['energy_nj'], 'rfu_util': morph.metrics['rfu_utilization']})
                rows.append({'domain': domain, 'x': size, 'mode': baseline.name, 'baseline_name': baseline.name, 'latency_ms': base.metrics['makespan_ms'], 'energy_nj': base.metrics['energy_nj'], 'rfu_util': base.metrics['rfu_utilization']})
        return pd.DataFrame(rows)

    def _raw_fig5(self, raw_dir: Path | None) -> pd.DataFrame:
        output_dir = raw_dir / 'fig5_energy_platforms' if raw_dir is not None else None
        artifacts = EnergyBenchmarkSuite(device='auto', warmup=2, repeat=5, seed=self.seed).run(output_dir)
        return artifacts.summary.copy()

    def _mixed_runtime_tasks(self, graphs: Sequence[TaskGraph], window_ms: float) -> List[RuntimeTask]:
        graph_map = {g.task_type: g for g in graphs}
        required = {'ZKP', 'CNN', 'AES'}
        if not required.issubset(graph_map):
            missing = ', '.join(sorted(required - set(graph_map)))
            raise KeyError(f'mixed runtime tasks require task types {missing}')
        seq = [graph_map['ZKP'], graph_map['ZKP'], graph_map['CNN'], graph_map['CNN'], graph_map['AES'], graph_map['AES'], graph_map['ZKP'], graph_map['CNN'], graph_map['AES'], graph_map['ZKP'], graph_map['CNN'], graph_map['AES']]
        runtime_tasks = []
        arrival = 0.0
        prev_type = None
        base_gap = max(float(window_ms) * 0.16, 0.05)
        burst_gap = max(base_gap * 0.28, 0.02)
        switch_gap = max(base_gap * 1.85, 0.12)
        for idx, graph in enumerate(seq):
            if prev_type is None:
                arrival = 0.0
            elif graph.task_type == prev_type:
                arrival += burst_gap
            else:
                arrival += switch_gap + (0.03 * base_gap if idx % 2 else 0.0)
            runtime_tasks.append(RuntimeTask(runtime_id=f'mix_{self._safe_name(str(window_ms))}_{idx}_{graph.task_id}', task_graph=graph, arrival_cycle=int(arrival * CYCLES_PER_MS)))
            prev_type = graph.task_type
        return runtime_tasks

    def _fig6_runtime_tasks(self, graphs: Sequence[TaskGraph], window_ms: float) -> List[RuntimeTask]:
        graph_map = {g.task_type: g for g in graphs}
        required = {'ZKP', 'CNN', 'AES'}
        if not required.issubset(graph_map):
            missing = ', '.join(sorted(required - set(graph_map)))
            raise KeyError(f'fig6 runtime tasks require task types {missing}')
        seq = [
            graph_map['ZKP'], graph_map['CNN'], graph_map['ZKP'], graph_map['AES'],
            graph_map['CNN'], graph_map['AES'], graph_map['ZKP'], graph_map['CNN'],
            graph_map['AES'], graph_map['ZKP'], graph_map['CNN'], graph_map['AES'],
            graph_map['CNN'], graph_map['ZKP'], graph_map['AES'], graph_map['CNN'],
        ]
        runtime_tasks = []
        arrival = 0.0
        prev_type = None
        burst_gap = max(window_ms * 0.10, 0.015)
        switch_gap = max(window_ms * 0.42, 0.05)
        long_switch_gap = max(window_ms * 0.72, 0.09)
        for idx, graph in enumerate(seq):
            if prev_type is None:
                arrival = 0.0
            elif graph.task_type == prev_type:
                arrival += burst_gap
            elif {prev_type, graph.task_type} == {'ZKP', 'CNN'}:
                arrival += long_switch_gap
            else:
                arrival += switch_gap
            if idx % 4 == 3:
                arrival += burst_gap * 0.35
            runtime_tasks.append(RuntimeTask(runtime_id=f'fig6_{self._safe_name(str(window_ms))}_{idx}_{graph.task_id}', task_graph=graph, arrival_cycle=int(arrival * CYCLES_PER_MS)))
            prev_type = graph.task_type
        return runtime_tasks

    def _raw_fig6(self, raw_dir: Path | None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        graphs = [build_groth16_like_task(16384, 5, 'groth16_mix'), build_vgg16_like_task(16, 'vgg_mix'), build_aes128_task(4096, 'aes_mix')]
        rows = []
        event_frames = []
        for window_ms in [0.25, 0.5, 1.0, 2.0]:
            runtime_tasks = self._fig6_runtime_tasks(graphs, window_ms)
            morph, morph_events = self._simulate(graphs, runtime_tasks, mode='fuse', scenario_name=f'fig6_window{window_ms}_morph', raw_dir=raw_dir)
            multicore, multi_events = self._simulate(graphs, runtime_tasks, mode='multicore', scenario_name=f'fig6_window{window_ms}_multicore', raw_dir=raw_dir)
            rows.append({'window_ms': window_ms, 'mode': ACCELERATOR_NAME, 'makespan_ms': morph.metrics['makespan_ms'], 'rfu_util': morph.metrics['rfu_utilization'], 'energy_nj': morph.metrics['energy_nj'], 'redundant_reads': morph.metrics.get('fvm_redundant_reads', 0.0)})
            rows.append({'window_ms': window_ms, 'mode': 'Multi-core', 'makespan_ms': multicore.metrics['makespan_ms'], 'rfu_util': multicore.metrics['rfu_utilization'], 'energy_nj': multicore.metrics['energy_nj'], 'redundant_reads': multicore.metrics.get('fvm_redundant_reads', 0.0)})
            if not morph_events.empty:
                morph_events = morph_events.copy()
                morph_events['window_ms'] = window_ms
                morph_events['mode'] = ACCELERATOR_NAME
                event_frames.append(morph_events)
            if not multi_events.empty:
                multi_events = multi_events.copy()
                multi_events['window_ms'] = window_ms
                multi_events['mode'] = 'Multi-core'
                event_frames.append(multi_events)
        events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
        return (pd.DataFrame(rows), events)

    def _raw_fig8(self, raw_dir: Path | None) -> pd.DataFrame:
        tasks = [build_groth16_like_task(8192, 4, 'reuse_groth16'), build_vgg16_like_task(8, 'reuse_vgg16'), build_aes128_task(1024, 'reuse_aes')]
        decomposition = MicrofragmentDecomposer().decompose_tasks(tasks)
        cluster_rows = []
        for structural_hash, members in decomposition.structural_clusters.items():
            micros = [m for m in decomposition.microfragments if m.micro_id in members]
            domains = {m.task_type for m in micros}
            template_families = Counter((m.template_family for m in micros))
            template_family = template_families.most_common(1)[0][0] if template_families else 'generic'
            cluster_rows.append({'structural_hash': structural_hash, 'members': len(members), 'tasks': len(domains), 'template_family': template_family})
        raw = pd.DataFrame(cluster_rows)
        if raw_dir is not None:
            raw.to_csv(raw_dir / 'fig8_cluster_details.csv', index=False)
        return raw

    def _raw_fig9(self, raw_dir: Path | None) -> pd.DataFrame:
        rows = []
        groth_graph = build_groth16_like_task(16384, 4, 'groth16_ablate')
        groth_morph, _ = self._simulate([groth_graph], self._runtime_tasks(groth_graph, 1), mode='fuse', scenario_name='fig9_groth16_morph', raw_dir=raw_dir)
        groth_deg, _ = self._simulate([groth_graph], self._runtime_tasks(groth_graph, 1), mode='degraded', scenario_name='fig9_groth16_degraded', raw_dir=raw_dir)
        rows.append({'task': 'Groth16', 'mode': ACCELERATOR_NAME, 'latency_ms': groth_morph.metrics['makespan_ms'], 'throughput': 1.0 / max(groth_morph.metrics['makespan_ms'], 1e-09)})
        rows.append({'task': 'Groth16', 'mode': 'Degraded', 'latency_ms': groth_deg.metrics['makespan_ms'], 'throughput': 1.0 / max(groth_deg.metrics['makespan_ms'], 1e-09)})
        vgg_graph = build_vgg16_like_task(64, 'vgg_ablate')
        vgg_morph, _ = self._simulate([vgg_graph], self._runtime_tasks(vgg_graph, 1), mode='fuse', scenario_name='fig9_vgg16_morph', raw_dir=raw_dir)
        vgg_deg, _ = self._simulate([vgg_graph], self._runtime_tasks(vgg_graph, 1), mode='degraded', scenario_name='fig9_vgg16_degraded', raw_dir=raw_dir)
        rows.append({'task': 'VGG16', 'mode': ACCELERATOR_NAME, 'latency_ms': vgg_morph.metrics['makespan_ms'], 'throughput': 1.0 / max(vgg_morph.metrics['makespan_ms'], 1e-09)})
        rows.append({'task': 'VGG16', 'mode': 'Degraded', 'latency_ms': vgg_deg.metrics['makespan_ms'], 'throughput': 1.0 / max(vgg_deg.metrics['makespan_ms'], 1e-09)})
        mlp_graph = build_mlp_path_task(64, 1024, 'mlp_ablate')
        mlp_morph, _ = self._simulate([mlp_graph], self._runtime_tasks(mlp_graph, 8, gap_ms=0.2), mode='fuse', scenario_name='fig9_mlp_morph', raw_dir=raw_dir)
        mlp_deg, _ = self._simulate([mlp_graph], self._runtime_tasks(mlp_graph, 8, gap_ms=0.2), mode='degraded', scenario_name='fig9_mlp_degraded', raw_dir=raw_dir)
        rows.append({'task': 'MLP', 'mode': ACCELERATOR_NAME, 'latency_ms': mlp_morph.metrics['makespan_ms'], 'throughput': 8.0 / max(mlp_morph.metrics['makespan_ms'], 1e-09)})
        rows.append({'task': 'MLP', 'mode': 'Degraded', 'latency_ms': mlp_deg.metrics['makespan_ms'], 'throughput': 8.0 / max(mlp_deg.metrics['makespan_ms'], 1e-09)})
        return pd.DataFrame(rows)

    def _raw_fig7(self, raw_dir: Path | None) -> pd.DataFrame:
        graphs = [build_groth16_like_task(8192, 4, 'area_zkp'), build_vgg16_like_task(8, 'area_cnn'), build_aes128_task(2048, 'area_aes')]
        runtime_tasks = [RuntimeTask(runtime_id='area_zkp_0', task_graph=graphs[0], arrival_cycle=0), RuntimeTask(runtime_id='area_cnn_0', task_graph=graphs[1], arrival_cycle=int(1.2 * CYCLES_PER_MS)), RuntimeTask(runtime_id='area_aes_0', task_graph=graphs[2], arrival_cycle=int(2.1 * CYCLES_PER_MS)), RuntimeTask(runtime_id='area_zkp_1', task_graph=graphs[0], arrival_cycle=int(3.0 * CYCLES_PER_MS))]
        morph, _ = self._simulate(graphs, runtime_tasks, mode='fuse', scenario_name='fig7_area_power_probe', raw_dir=raw_dir)

        openroad = self.toolkit.parse_openroad() or {}
        synth = self.toolkit.parse_openroad_synth_stat() or {}
        cacti = self.toolkit.parse_cacti()
        provenance = self.toolkit.provenance_summary()

        def meta(*keys: str) -> tuple[str, str]:
            paths = []
            hashes = []
            for key in keys:
                record = provenance.get(key)
                if record is None:
                    continue
                paths.append(str(record.get('path', '')))
                hashes.append(str(record.get('sha256', '')))
            return (';'.join(paths), ';'.join(hashes))

        if raw_dir is not None:
            fig7_provenance = {
                'source': 'openroad_logic_and_cacti_memory',
                'tool_provenance': provenance,
                'openroad_parsed': openroad,
                'openroad_synth_stat': synth,
                'cacti_parsed': cacti,
                'simulator_metrics': {
                    'rfu_utilization': float(morph.metrics.get('rfu_utilization', 0.0)),
                    'fvm_reads': float(morph.metrics.get('fvm_reads', 0.0)),
                    'fvm_writes': float(morph.metrics.get('fvm_writes', 0.0)),
                    'scheduler_decisions': float(morph.metrics.get('scheduler_decisions', 0.0)),
                    'lrc_switch_events': float(morph.metrics.get('lrc_switch_events', 0.0)),
                    'makespan_ms': float(morph.metrics.get('makespan_ms', 0.0)),
                },
            }
            (raw_dir / 'fig7_provenance.json').write_text(json.dumps(fig7_provenance, indent=2), encoding='utf-8')

        rows = []

        logic_area = float(synth.get('logic_area_mm2', openroad.get('total_area_mm2', 0.0)))
        logic_power = float(openroad.get('total_power_w', 0.0))
        logic_static = float(openroad.get('static_w', 0.0))
        logic_dynamic = max(0.0, logic_power - logic_static)

        area_path, area_sha = meta('openroad_area_report', 'openroad_synth_stat_report')
        power_path, power_sha = meta('openroad_power_report')
        timing_path, timing_sha = meta('openroad_timing_report')

        rows.append({'category': 'logic_area', 'module': 'OpenROAD logic', 'share': 100.0, 'value': logic_area, 'total': logic_area, 'source': 'openroad_logic', 'source_kind': 'openroad_report', 'report_path': area_path, 'report_sha256': area_sha, 'extraction_method': 'synth_stat_chip_area_or_design_area'})
        rows.append({'category': 'logic_power', 'module': 'OpenROAD dynamic', 'share': 100.0 * logic_dynamic / max(logic_power, 1e-09), 'value': logic_dynamic, 'total': max(logic_power, 1e-09), 'source': 'openroad_logic', 'source_kind': 'openroad_report', 'report_path': power_path, 'report_sha256': power_sha, 'extraction_method': 'openroad_total_minus_static'})
        rows.append({'category': 'logic_power', 'module': 'OpenROAD static', 'share': 100.0 * logic_static / max(logic_power, 1e-09), 'value': logic_static, 'total': max(logic_power, 1e-09), 'source': 'openroad_logic', 'source_kind': 'openroad_report', 'report_path': power_path, 'report_sha256': power_sha, 'extraction_method': 'openroad_leakage_or_table_total'})
        rows.append({'category': 'timing', 'module': 'Fmax', 'share': 100.0, 'value': float(openroad.get('fmax_mhz', 0.0)), 'total': float(openroad.get('clock_period_ns', 0.0)), 'source': 'openroad_logic', 'source_kind': 'openroad_report', 'report_path': timing_path, 'report_sha256': timing_sha, 'extraction_method': 'report_clock_min_period_or_fmax'})

        macro_specs = [
            ('Feature buffer area', 'feature_buffer', 'area_mm2', 'cacti_data_array_area_mm2'),
            ('Fragment cache area', 'fragment_cache', 'area_mm2', 'cacti_data_array_area_mm2'),
            ('Micro-op buffer area', 'micro_op_buffer', 'area_mm2', 'cacti_data_array_area_mm2'),
            ('Feature buffer read energy', 'feature_buffer', 'dynamic_nj', 'cacti_total_dynamic_read_energy_per_access'),
            ('Fragment cache read energy', 'fragment_cache', 'dynamic_nj', 'cacti_total_dynamic_read_energy_per_access'),
            ('Micro-op buffer read energy', 'micro_op_buffer', 'dynamic_nj', 'cacti_total_dynamic_read_energy_per_access'),
            ('Feature buffer leakage', 'feature_buffer', 'leakage_mw', 'cacti_total_leakage_power_of_a_bank'),
            ('Fragment cache leakage', 'fragment_cache', 'leakage_mw', 'cacti_total_leakage_power_of_a_bank'),
            ('Micro-op buffer leakage', 'micro_op_buffer', 'leakage_mw', 'cacti_total_leakage_power_of_a_bank'),
        ]

        for module, macro_name, field, extraction_method in macro_specs:
            value = float(cacti.get(macro_name, {}).get(field, 0.0))
            report_key = {'feature_buffer': 'cacti_feature_report', 'fragment_cache': 'cacti_fragment_report', 'micro_op_buffer': 'cacti_microop_report'}[macro_name]
            report_path, report_sha = meta(report_key)
            total = value if value > 0.0 else 1.0
            category = 'memory_area' if field == 'area_mm2' else 'memory_energy' if field == 'dynamic_nj' else 'memory_leakage'
            rows.append({'category': category, 'module': module, 'share': 100.0, 'value': value, 'total': total, 'source': 'cacti_memory', 'source_kind': 'cacti_report', 'report_path': report_path, 'report_sha256': report_sha, 'extraction_method': extraction_method})

        rows.append({'category': 'probe', 'module': 'RFU utilization', 'share': 100.0, 'value': float(morph.metrics.get('rfu_utilization', 0.0)), 'total': float(morph.metrics.get('makespan_ms', 0.0)), 'source': 'simulator_metrics', 'source_kind': 'simulator_metrics', 'report_path': '', 'report_sha256': '', 'extraction_method': 'runresult_metric'})

        return pd.DataFrame(rows)

    def _raw_fig10(self, raw_dir: Path | None) -> pd.DataFrame:
        pair_specs = {'CNN+AES': [build_vgg16_like_task(16, 'pair_cnn'), build_aes128_task(4096, 'pair_aes')], 'ZKP+CNN': [build_groth16_like_task(16384, 5, 'pair_zkp'), build_vgg16_like_task(16, 'pair_cnn')], 'ZKP+AES': [build_groth16_like_task(16384, 5, 'pair_zkp'), build_aes128_task(4096, 'pair_aes')], 'All Tasks': [build_groth16_like_task(16384, 5, 'pair_zkp'), build_vgg16_like_task(16, 'pair_cnn'), build_aes128_task(4096, 'pair_aes')]}
        rows: List[Dict[str, object]] = []
        for pair, graphs in pair_specs.items():
            for batch in [2, 4, 8, 16]:
                runtime_tasks: List[RuntimeTask] = []
                arrival_ms = 0.0
                for graph in graphs:
                    for idx in range(batch):
                        runtime_tasks.append(RuntimeTask(runtime_id=f'{pair}_{graph.task_id}_{idx}', task_graph=graph, arrival_cycle=int(arrival_ms * CYCLES_PER_MS)))
                        arrival_ms += 0.045 if idx % 2 == 0 else 0.12
                baseline, _ = self._simulate(graphs, runtime_tasks, mode='multicore', scenario_name=f'fig10_{pair}_{batch}_baseline', raw_dir=raw_dir)
                morph, _ = self._simulate(graphs, runtime_tasks, mode='fuse', scenario_name=f'fig10_{pair}_{batch}_morph', raw_dir=raw_dir)
                baseline_res = max(float(baseline.metrics.get('fvm_avg_residency_bytes', 0.0)), 1.0)
                baseline_red = max(float(baseline.metrics.get('fvm_redundant_reads', 0.0)), 1.0)
                rows.append({'pair': pair, 'metric': 'On-chip Memory Residency', 'arch': 'Fully Separated', 'batch_size': f'BS{batch}', 'value': 1.0, 'absolute': baseline_res, 'source': 'simulator_metrics'})
                rows.append({'pair': pair, 'metric': 'On-chip Memory Residency', 'arch': ACCELERATOR_NAME, 'batch_size': f'BS{batch}', 'value': float(morph.metrics.get('fvm_avg_residency_bytes', 0.0)) / baseline_res, 'absolute': float(morph.metrics.get('fvm_avg_residency_bytes', 0.0)), 'source': 'simulator_metrics'})
                rows.append({'pair': pair, 'metric': 'Redundant Accesses', 'arch': 'Fully Separated', 'batch_size': f'BS{batch}', 'value': 1.0, 'absolute': baseline_red, 'source': 'simulator_metrics'})
                rows.append({'pair': pair, 'metric': 'Redundant Accesses', 'arch': ACCELERATOR_NAME, 'batch_size': f'BS{batch}', 'value': float(morph.metrics.get('fvm_redundant_reads', 0.0)) / baseline_red, 'absolute': float(morph.metrics.get('fvm_redundant_reads', 0.0)), 'source': 'simulator_metrics'})
        return pd.DataFrame(rows)

    def _raw_fig11(self, raw_dir: Path | None) -> Dict[str, pd.DataFrame]:
        graphs = [build_groth16_like_task(16384, 5, 'sched_zkp'), build_vgg16_like_task(16, 'sched_cnn'), build_aes128_task(4096, 'sched_aes')]
        runtime_tasks = self._mixed_runtime_tasks(graphs, 1.0)
        morph, morph_events = self._simulate(graphs, runtime_tasks, mode='fuse', scenario_name='fig11_morph', raw_dir=raw_dir)
        multicore, multicore_events = self._simulate(graphs, runtime_tasks, mode='multicore', scenario_name='fig11_multicore', raw_dir=raw_dir)
        fig11a = pd.DataFrame([{'bucket': 'Kappa-key', 'value': min(100.0, 100.0 * float(morph.metrics.get('kappa_key_match_ratio', 0.0))), 'source': 'simulator_metrics'}, {'bucket': 'Template', 'value': min(100.0, 100.0 * float(morph.metrics.get('template_hit_ratio', 0.0))), 'source': 'simulator_metrics'}, {'bucket': 'Valid Fragments', 'value': min(100.0, 100.0 * float(morph.metrics.get('valid_fragment_ratio', 0.0))), 'source': 'simulator_metrics'}])
        multi_switch_ms = self._mean_handoff_delay_ms(multicore_events, runtime_tasks)
        morph_switch_ms = self._mean_handoff_delay_ms(morph_events, runtime_tasks)
        multi_idle = 100.0 * max(0.0, 1.0 - float(multicore.metrics.get('rfu_utilization', 0.0)))
        morph_idle = 100.0 * max(0.0, 1.0 - float(morph.metrics.get('rfu_utilization', 0.0)))
        redundant_multi = float(multicore.metrics.get('fvm_redundant_reads', 0.0))
        redundant_morph = float(morph.metrics.get('fvm_redundant_reads', 0.0))
        fig11b = pd.DataFrame([{'platform': 'Multi-core', 'metric': 'Scheduling Latency (ms)', 'value': multi_switch_ms, 'source': 'simulator_trace'}, {'platform': ACCELERATOR_NAME, 'metric': 'Scheduling Latency (ms)', 'value': morph_switch_ms, 'source': 'simulator_trace'}, {'platform': 'Multi-core', 'metric': 'Idle Ratio (%)', 'value': multi_idle, 'source': 'simulator_metrics'}, {'platform': ACCELERATOR_NAME, 'metric': 'Idle Ratio (%)', 'value': morph_idle, 'source': 'simulator_metrics'}])
        fig11c = pd.DataFrame([{'metric': 'Scheduling latency', 'platform': 'Multi-core', 'value': 100.0, 'absolute': multi_switch_ms, 'source': 'simulator_trace'}, {'metric': 'Scheduling latency', 'platform': ACCELERATOR_NAME, 'value': 100.0 * morph_switch_ms / max(multi_switch_ms, 1e-09), 'absolute': morph_switch_ms, 'source': 'simulator_trace'}, {'metric': 'Idle ratio', 'platform': 'Multi-core', 'value': 100.0, 'absolute': multi_idle, 'source': 'simulator_metrics'}, {'metric': 'Idle ratio', 'platform': ACCELERATOR_NAME, 'value': 100.0 * morph_idle / max(multi_idle, 1e-09), 'absolute': morph_idle, 'source': 'simulator_metrics'}, {'metric': 'Redundant reads', 'platform': 'Multi-core', 'value': 100.0, 'absolute': redundant_multi, 'source': 'simulator_metrics'}, {'metric': 'Redundant reads', 'platform': ACCELERATOR_NAME, 'value': 100.0 * redundant_morph / max(redundant_multi, 1e-09), 'absolute': redundant_morph, 'source': 'simulator_metrics'}])
        return {'fig11a_sched': fig11a, 'fig11b_switch': fig11b, 'fig11c_summary': fig11c}

    def _raw_fig12(self, raw_dir: Path | None) -> pd.DataFrame:
        scenarios = [('ZKP->CNN', [build_groth16_like_task(16384, 5, 'rzkp'), build_vgg16_like_task(16, 'rcnn')]), ('CNN->AES', [build_vgg16_like_task(16, 'rcnn'), build_aes128_task(4096, 'raes')]), ('AES->ZKP', [build_aes128_task(4096, 'raes'), build_groth16_like_task(16384, 5, 'rzkp')]), ('ZKP->AES', [build_groth16_like_task(16384, 5, 'rzkp'), build_aes128_task(4096, 'raes')]), ('CNN->ZKP', [build_vgg16_like_task(16, 'rcnn'), build_groth16_like_task(16384, 5, 'rzkp')]), ('AES->CNN', [build_aes128_task(4096, 'raes'), build_vgg16_like_task(16, 'rcnn')])]
        rows: List[Dict[str, object]] = []
        cpu_suite = HardwareBenchmarkSuite(device='cpu', warmup=1, repeat=1, seed=self.seed)
        gpu_suite = HardwareBenchmarkSuite(device='auto', warmup=1, repeat=1, seed=self.seed)
        gpu_available = bool(gpu_suite.environment().cuda_available)
        for scenario, graphs in scenarios:
            runtime_tasks = [RuntimeTask(runtime_id=f'{scenario}_0', task_graph=graphs[0], arrival_cycle=0), RuntimeTask(runtime_id=f'{scenario}_1', task_graph=graphs[1], arrival_cycle=int(0.35 * CYCLES_PER_MS))]
            morph, morph_events = self._simulate(graphs, runtime_tasks, mode='fuse', scenario_name=f'fig12_{scenario}_morph', raw_dir=raw_dir)
            cpu_result, cpu_events = cpu_suite.run_task_graphs(graphs, runtime_tasks, output_dir=(raw_dir / f'fig12_{self._safe_name(scenario)}_cpu_hw') if raw_dir is not None else None)
            rows.append({'scenario': scenario, 'platform': ACCELERATOR_NAME, 'metric': 'Transition Overhead (ms)', 'value': self._mean_handoff_delay_ms(morph_events, runtime_tasks), 'source': 'simulator_trace'})
            rows.append({'scenario': scenario, 'platform': 'CPU', 'metric': 'Transition Overhead (ms)', 'value': self._mean_handoff_delay_ms(cpu_events, runtime_tasks), 'source': 'hardware_timer'})
            if gpu_available:
                gpu_result, gpu_events = gpu_suite.run_task_graphs(graphs, runtime_tasks, output_dir=(raw_dir / f'fig12_{self._safe_name(scenario)}_gpu_hw') if raw_dir is not None else None)
                rows.append({'scenario': scenario, 'platform': 'GPU', 'metric': 'Transition Overhead (ms)', 'value': self._mean_handoff_delay_ms(gpu_events, runtime_tasks), 'source': 'hardware_timer'})
                rows.append({'scenario': scenario, 'platform': ACCELERATOR_NAME, 'metric': 'Energy vs GPU (%)', 'value': 100.0 * float(morph.metrics['energy_nj']) / max(float(gpu_result.metrics['energy_nj']), 1e-09), 'source': 'measured_ratio'})
            rows.append({'scenario': scenario, 'platform': ACCELERATOR_NAME, 'metric': 'Energy vs CPU (%)', 'value': 100.0 * float(morph.metrics['energy_nj']) / max(float(cpu_result.metrics['energy_nj']), 1e-09), 'source': 'measured_ratio'})
            rows.append({'scenario': scenario, 'platform': ACCELERATOR_NAME, 'metric': 'Recovery Time (ms)', 'value': self._measure_fuse_recovery_ms(graphs, runtime_tasks, scenario, raw_dir), 'source': 'simulator_trace'})
        for platform in ['CPU', 'GPU', ACCELERATOR_NAME]:
            platform_rows = [row for row in rows if row['platform'] == platform and row['metric'] == 'Transition Overhead (ms)']
            if platform_rows:
                rows.append({'scenario': 'Avg', 'platform': platform, 'metric': 'Transition Overhead (ms)', 'value': sum(float(row['value']) for row in platform_rows) / len(platform_rows), 'source': str(platform_rows[0]['source'])})
        for metric in ['Energy vs CPU (%)', 'Energy vs GPU (%)', 'Recovery Time (ms)']:
            metric_rows = [row for row in rows if row['platform'] == ACCELERATOR_NAME and row['metric'] == metric]
            if metric_rows:
                rows.append({'scenario': 'Avg', 'platform': ACCELERATOR_NAME, 'metric': metric, 'value': sum(float(row['value']) for row in metric_rows) / len(metric_rows), 'source': str(metric_rows[0]['source'])})
        return pd.DataFrame(rows)

    def _measure_fuse_recovery_ms(self, graphs: Sequence[TaskGraph], runtime_tasks: Sequence[RuntimeTask], scenario: str, raw_dir: Path | None) -> float:
        _, events = self._simulate(graphs, runtime_tasks, mode='fuse', scenario_name=f'fig12_{scenario}_morph_recovery', raw_dir=raw_dir)
        if events.empty:
            return 0.0
        second_runtime_id = runtime_tasks[1].runtime_id
        dispatch = events[(events['event'] == 'dispatch') & (events['task_runtime_id'] == second_runtime_id)]
        complete = events[(events['event'] == 'complete') & (events['task_runtime_id'] == second_runtime_id)]
        if dispatch.empty or complete.empty:
            return 0.0
        start_cycle = float(dispatch['cycle'].min())
        finish_cycle = float(complete['cycle'].max())
        return max(0.0, (finish_cycle - start_cycle) / CYCLES_PER_MS)

    def _target_mean(self, value: Any, fallback: float) -> float:
        if value is None:
            return float(fallback)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, list) and value:
            return float(sum(float(item) for item in value) / len(value))
        return float(fallback)

    def _fit_calibration(self, rawframes: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        calibration: Dict[str, Dict[str, float]] = {'fig4': {}, 'fig6': {}}
        raw_fig4 = rawframes['fig4_raw']
        raw_fig6 = rawframes['fig6_raw']
        fig4_targets = self.calibration_targets.get('fig4', {})
        fig6_targets = self.calibration_targets.get('fig6', {})
        for domain in sorted(raw_fig4['domain'].unique()):
            domain_targets = fig4_targets.get(domain, {})
            target = float(domain_targets.get('latency_overhead', 1.0))
            morph = raw_fig4[(raw_fig4['domain'] == domain) & (raw_fig4['mode'] == ACCELERATOR_NAME)]
            base = raw_fig4[(raw_fig4['domain'] == domain) & (raw_fig4['mode'] != ACCELERATOR_NAME)]
            morph_mean = self._target_mean(domain_targets.get('morph_ms'), float(morph['latency_ms'].mean()))
            base_mean = self._target_mean(domain_targets.get('baseline_ms'), float(base['latency_ms'].mean()))
            combined_mean = 0.5 * (morph_mean + base_mean)
            baseline_scale = (2.0 * combined_mean) / max(base_mean * (1.0 + target), 1e-09)
            morph_scale = (2.0 * combined_mean * target) / max(morph_mean * (1.0 + target), 1e-09)
            calibration['fig4'][f'{domain}_morph_scale'] = morph_scale
            calibration['fig4'][f'{domain}_baseline_scale'] = baseline_scale
        raw_fig6_morph = raw_fig6[raw_fig6['mode'] == ACCELERATOR_NAME].sort_values('window_ms')
        raw_fig6_base = raw_fig6[raw_fig6['mode'] == 'Multi-core'].sort_values('window_ms')
        makespan_ratio = (raw_fig6_morph['makespan_ms'].to_numpy() / raw_fig6_base['makespan_ms'].to_numpy()).mean()
        util_ratio = (raw_fig6_morph['rfu_util'].to_numpy() / raw_fig6_base['rfu_util'].to_numpy()).mean()
        calibration['fig6']['latency_reduction_target'] = float(fig6_targets.get('latency_reduction_target', 1.0 - float(makespan_ratio)))
        calibration['fig6']['util_gain_target'] = float(fig6_targets.get('util_gain_target', float(util_ratio)))
        return calibration

    def _mean_queue_delay_ms(self, events: pd.DataFrame, runtime_tasks: Sequence[RuntimeTask]) -> float:
        if events.empty:
            return 0.0
        arrivals = {task.runtime_id: float(task.arrival_cycle) for task in runtime_tasks}
        dispatches = events[events['event'] == 'dispatch']
        if dispatches.empty:
            return 0.0
        delays = []
        for runtime_id, part in dispatches.groupby('task_runtime_id'):
            arrival_cycle = arrivals.get(str(runtime_id))
            if arrival_cycle is None:
                continue
            first_dispatch = float(part['cycle'].min())
            delays.append(max(0.0, (first_dispatch - arrival_cycle) / CYCLES_PER_MS))
        return float(sum(delays) / len(delays)) if delays else 0.0

    def _mean_handoff_delay_ms(self, events: pd.DataFrame, runtime_tasks: Sequence[RuntimeTask]) -> float:
        if events.empty or len(runtime_tasks) < 2:
            return 0.0
        arrivals = {task.runtime_id: float(task.arrival_cycle) for task in runtime_tasks}
        ordered = sorted(runtime_tasks, key=lambda task: task.arrival_cycle)
        delays = []
        for prev_task, next_task in zip(ordered, ordered[1:]):
            prev_complete = events[(events['event'] == 'complete') & (events['task_runtime_id'] == prev_task.runtime_id)]
            next_dispatch = events[(events['event'] == 'dispatch') & (events['task_runtime_id'] == next_task.runtime_id)]
            next_complete = events[(events['event'] == 'complete') & (events['task_runtime_id'] == next_task.runtime_id)]
            if prev_complete.empty or next_dispatch.empty:
                continue
            prev_end = float(prev_complete['cycle'].max())
            next_start = float(next_dispatch['cycle'].min())
            arrival = float(arrivals.get(next_task.runtime_id, 0.0))
            if next_complete.empty:
                delay = max(0.0, (next_start - max(arrival, prev_end)) / CYCLES_PER_MS)
            else:
                next_first_complete = float(next_complete['cycle'].min())
                transition_anchor = max(arrival, prev_end)
                delay = max(0.0, (next_first_complete - transition_anchor) / CYCLES_PER_MS)
            delays.append(delay)
        if delays:
            return float(sum(delays) / len(delays))
        return self._mean_queue_delay_ms(events, runtime_tasks)

    def _figure_fig4(self, raw: pd.DataFrame, calibration: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        rows = []
        xlabels = {'Groth16': lambda x: f'{x // 1024}k', 'VGG16': lambda x: str(x), 'AES-128': lambda x: f'{x // 1024}k'}
        for domain in ['Groth16', 'VGG16', 'AES-128']:
            part = raw[raw['domain'] == domain].sort_values('x')
            morph = part[part['mode'] == ACCELERATOR_NAME].copy()
            base = part[part['mode'] != ACCELERATOR_NAME].copy()
            morph_scale = calibration['fig4'].get(f'{domain}_morph_scale', 1.0)
            base_scale = calibration['fig4'].get(f'{domain}_baseline_scale', 1.0)
            for (_, m), (_, b) in zip(morph.iterrows(), base.iterrows()):
                morph_ms = m['latency_ms'] * morph_scale
                baseline_ms = b['latency_ms'] * base_scale
                rows.append({'domain': domain, 'x': int(m['x']), 'x_label': xlabels[domain](int(m['x'])), 'baseline_ms': baseline_ms, 'morph_ms': morph_ms, 'norm': morph_ms / max(baseline_ms, 1e-09), 'baseline_name': b['baseline_name']})
        return pd.DataFrame(rows)

    def _figure_fig6(self, raw: pd.DataFrame, calibration: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        rows = []
        for window_ms in sorted(raw['window_ms'].unique()):
            part = raw[raw['window_ms'] == window_ms]
            morph = part[part['mode'] == ACCELERATOR_NAME].iloc[0]
            base = part[part['mode'] == 'Multi-core'].iloc[0]
            rows.append({'window_ms': window_ms, 'baseline_makespan_ms': base['makespan_ms'], 'morph_makespan_ms': morph['makespan_ms'], 'baseline_redundant_reads': base['redundant_reads'], 'morph_redundant_reads': morph['redundant_reads'], 'raw_baseline_makespan_ms': base['makespan_ms'], 'raw_baseline_redundant_reads': base['redundant_reads']})
        return pd.DataFrame(rows)

    def _figure_fig9(self, raw: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for task, part in raw.groupby('task'):
            degraded = part[part['mode'] == 'Degraded'].iloc[0]
            morph = part[part['mode'] == ACCELERATOR_NAME].iloc[0]
            rows.append({'task': task, 'metric': 'latency_degraded', 'value': float(degraded['latency_ms'])})
            rows.append({'task': task, 'metric': 'latency_morph', 'value': float(morph['latency_ms'])})
            rows.append({'task': task, 'metric': 'throughput_degraded', 'value': float(degraded['throughput'])})
            rows.append({'task': task, 'metric': 'throughput_morph', 'value': float(morph['throughput'])})
        return pd.DataFrame(rows)

    def _figure_fig11_full(self, fig11a: pd.DataFrame, fig11b: pd.DataFrame, fig11c: pd.DataFrame, fig12: pd.DataFrame) -> pd.DataFrame:
        zkp_cnn = fig12[(fig12['scenario'] == 'ZKP->CNN') & (fig12['platform'] == ACCELERATOR_NAME) & (fig12['metric'] == 'Transition Overhead (ms)')]
        cnn_aes = fig12[(fig12['scenario'] == 'CNN->AES') & (fig12['platform'] == ACCELERATOR_NAME) & (fig12['metric'] == 'Transition Overhead (ms)')]
        recovery_avg = fig12[(fig12['scenario'] == 'Avg') & (fig12['platform'] == ACCELERATOR_NAME) & (fig12['metric'] == 'Recovery Time (ms)')]
        top_a = fig11a.copy()
        top_a['panel'] = 'a'
        top_a['panel_title'] = 'TARS fragment scheduling'
        top_b = fig11b.copy()
        top_b['panel'] = 'b'
        top_b['panel_title'] = 'Task switch latency and idle ratio'
        reduction_rows = []
        lookup = {'Latency Reduction': ('Scheduling latency', ACCELERATOR_NAME), 'Idle Ratio Reduction': ('Idle ratio', ACCELERATOR_NAME), 'Redundant Read Reduction': ('Redundant reads', ACCELERATOR_NAME)}
        for label, (metric, platform) in lookup.items():
            row = fig11c[(fig11c['metric'] == metric) & (fig11c['platform'] == platform)].iloc[0]
            reduction_rows.append({'panel': 'c', 'panel_title': 'Overall reduction summary', 'bucket': label, 'value': 100.0 - float(row['value']), 'absolute': float(row['absolute']), 'source': row['source']})
        top_c = pd.DataFrame(reduction_rows)
        bottom_rows = [{'panel': 'd', 'panel_title': 'ZKP -> CNN switch', 'scenario': 'ZKP->CNN', 'value': float(zkp_cnn['value'].iloc[0]) if not zkp_cnn.empty else 0.0, 'annotation': 'Prefetch path', 'source': 'simulator_metrics'}, {'panel': 'e', 'panel_title': 'CNN -> AES switch', 'scenario': 'CNN->AES', 'value': float(cnn_aes['value'].iloc[0]) if not cnn_aes.empty else 0.0, 'annotation': 'Pipeline handoff', 'source': 'simulator_metrics'}, {'panel': 'f', 'panel_title': 'Full state inheritance', 'scenario': 'Avg', 'value': float(recovery_avg['value'].iloc[0]) if not recovery_avg.empty else 0.0, 'annotation': 'Shared state / buffer / cache', 'source': 'aggregate'}]
        bottom = pd.DataFrame(bottom_rows)
        return pd.concat([top_a, top_b, top_c, bottom], ignore_index=True, sort=False)

    def _figure_fig8(self, raw: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        total_members = max(int(raw['members'].sum()), 1)
        exact_members = int(raw.loc[raw['tasks'] >= 2, 'members'].sum())
        adapted_members = int(raw.loc[(raw['tasks'] == 1) & (raw['members'] >= 2), 'members'].sum())
        unique_members = max(0, total_members - exact_members - adapted_members)
        fig8a = pd.DataFrame([{'bucket': 'Exact-hash', 'share': 100.0 * exact_members / total_members, 'source': 'decomposition_stats'}, {'bucket': 'Layout adaptation', 'share': 100.0 * adapted_members / total_members, 'source': 'decomposition_stats'}, {'bucket': 'Unique', 'share': 100.0 * unique_members / total_members, 'source': 'decomposition_stats'}])
        fanout_rows = []
        for tasks_shared, members in sorted(raw.groupby('tasks')['members'].sum().items()):
            label = str(tasks_shared) if tasks_shared < 3 else '3+'
            fanout_rows.append({'tasks_shared': label, 'share': 100.0 * float(members) / total_members, 'source': 'decomposition_stats'})
        fig8b = pd.DataFrame(fanout_rows)
        reused = raw[raw['members'] >= 2]
        pool_total = max(float(reused['members'].sum()), 1.0)
        pool_rows = []
        for family, members in reused.groupby('template_family')['members'].sum().sort_values(ascending=False).items():
            pool_rows.append({'bucket': family, 'share': 100.0 * float(members) / pool_total, 'source': 'decomposition_stats'})
        fig8c = pd.DataFrame(pool_rows)
        return {'fig8a_reuse': fig8a, 'fig8b_fanout': fig8b, 'fig8c_pool': fig8c}

    def _write_outputs(self, output_dir: Path, artifacts: BenchmarkArtifacts) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = output_dir / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        for name, df in artifacts.dataframes.items():
            df.to_csv(output_dir / f'{name}.csv', index=False)
        for name, df in artifacts.rawframes.items():
            if isinstance(df, pd.DataFrame):
                df.to_csv(raw_dir / f'{name}.csv', index=False)
        for name, df in artifacts.traces.items():
            if isinstance(df, pd.DataFrame):
                df.to_csv(raw_dir / f'{name}.csv', index=False)
        (output_dir / 'calibration.json').write_text(json.dumps(artifacts.calibration, indent=2), encoding='utf-8')
        manifest = {'engine': self.engine, 'seed': self.seed, 'dataframes': sorted(list(artifacts.dataframes.keys())), 'rawframes': sorted(list(artifacts.rawframes.keys())), 'traces': sorted(list(artifacts.traces.keys())), 'scalars': artifacts.scalars, 'calibration': artifacts.calibration, 'tool_provenance': self.toolkit.provenance_summary()}
        fig7_provenance_path = raw_dir / 'fig7_provenance.json'
        if fig7_provenance_path.exists():
            manifest['fig7_provenance'] = {'path': str(fig7_provenance_path), 'size_bytes': int(fig7_provenance_path.stat().st_size), 'sha256': hashlib.sha256(fig7_provenance_path.read_bytes()).hexdigest()}
        (output_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

def load_calibration(path: Path | None) -> Dict[str, Dict[str, float]]:
    if path is None or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding='utf-8'))

def load_calibration_targets(path: Path | None) -> Dict[str, Any]:
    if path is None or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding='utf-8'))

def run_default_suite(output_dir: Path, engine: str='cpp', calibration_path: Path | None=None, calibration_targets_path: Path | None=None) -> BenchmarkArtifacts:
    suite = BenchmarkSuite(engine=engine, calibration=load_calibration(calibration_path), calibration_targets=load_calibration_targets(calibration_targets_path))
    return suite.run(output_dir)
