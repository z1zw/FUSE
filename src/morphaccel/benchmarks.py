from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Dict, List, Sequence, Tuple
import json
import random
import statistics
import subprocess
import time
import numpy as np
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
from .sim.simulator import CYCLES_PER_MS, MorphAccelSimulator, SimulationConfig
from .workloads import build_aes128_task, build_groth16_like_task, build_mlp_path_task, build_vgg16_like_task

@dataclass
class BenchmarkArtifacts:
    dataframes: Dict[str, pd.DataFrame]
    rawframes: Dict[str, pd.DataFrame]
    traces: Dict[str, pd.DataFrame]
    scalars: Dict[str, float]
    calibration: Dict[str, Dict[str, float]]

class BenchmarkSuite:

    def __init__(self, engine: str='cpp', calibration: Dict[str, Dict[str, float]] | None=None, seed: int=7):
        self.engine = engine
        self.seed = seed
        self.rng = random.Random(seed)
        self.project_root = Path(__file__).resolve().parents[2]
        self.cpp = CppSimulatorBridge(self.project_root)
        self.toolkit = EvaluationToolkit(self.project_root)
        self.serializer = ScenarioSerializer()
        self.calibration = calibration or {}

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
        result = MorphAccelSimulator(config=SimulationConfig(mode=mode)).run(runtime_tasks, decomposition.microfragments)
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
                morph, _ = self._simulate([graph], runtime_tasks, mode='morphaccel', scenario_name=f'fig4_{domain}_{size}_morph', raw_dir=raw_dir)
                base, _ = self._simulate([graph], runtime_tasks, mode=baseline.mode, scenario_name=f'fig4_{domain}_{size}_baseline', raw_dir=raw_dir)
                rows.append({'domain': domain, 'x': size, 'mode': ACCELERATOR_NAME, 'baseline_name': baseline.name, 'latency_ms': morph.metrics['makespan_ms'], 'energy_nj': morph.metrics['energy_nj'], 'rfu_util': morph.metrics['rfu_utilization']})
                rows.append({'domain': domain, 'x': size, 'mode': baseline.name, 'baseline_name': baseline.name, 'latency_ms': base.metrics['makespan_ms'], 'energy_nj': base.metrics['energy_nj'], 'rfu_util': base.metrics['rfu_utilization']})
        return pd.DataFrame(rows)

    def _raw_fig5(self, raw_dir: Path | None) -> pd.DataFrame:
        output_dir = raw_dir / 'fig5_energy_platforms' if raw_dir is not None else None
        artifacts = EnergyBenchmarkSuite(device='auto', warmup=2, repeat=5, seed=self.seed).run(output_dir)
        return artifacts.summary.copy()

    def _mixed_runtime_tasks(self, graphs: Sequence[TaskGraph], window_ms: int) -> List[RuntimeTask]:
        graph_map = {g.task_type: g for g in graphs}
        required = {'ZKP', 'CNN', 'AES'}
        if not required.issubset(graph_map):
            missing = ', '.join(sorted(required - set(graph_map)))
            raise KeyError(f'mixed runtime tasks require task types {missing}')
        seq = [graph_map['ZKP']] * 4 + [graph_map['CNN']] * 8 + [graph_map['AES']] * 10
        runtime_tasks = []
        arrival = 0.0
        for idx, graph in enumerate(seq):
            jitter = idx % 3 * 0.07 * window_ms
            runtime_tasks.append(RuntimeTask(runtime_id=f'mix_{window_ms}_{idx}_{graph.task_id}', task_graph=graph, arrival_cycle=int((arrival + jitter) * CYCLES_PER_MS)))
            arrival += window_ms / 3.0
        return runtime_tasks

    def _raw_fig6(self, raw_dir: Path | None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        graphs = [build_groth16_like_task(8192, 4, 'groth16_mix'), build_vgg16_like_task(8, 'vgg_mix'), build_aes128_task(1024, 'aes_mix')]
        rows = []
        event_frames = []
        for window_ms in [1, 2, 4, 8]:
            runtime_tasks = self._mixed_runtime_tasks(graphs, window_ms)
            morph, morph_events = self._simulate(graphs, runtime_tasks, mode='morphaccel', scenario_name=f'fig6_window{window_ms}_morph', raw_dir=raw_dir)
            multicore, multi_events = self._simulate(graphs, runtime_tasks, mode='multicore', scenario_name=f'fig6_window{window_ms}_multicore', raw_dir=raw_dir)
            rows.append({'window_ms': window_ms, 'mode': ACCELERATOR_NAME, 'makespan_ms': morph.metrics['makespan_ms'], 'rfu_util': morph.metrics['rfu_utilization'], 'energy_nj': morph.metrics['energy_nj']})
            rows.append({'window_ms': window_ms, 'mode': 'Multi-core', 'makespan_ms': multicore.metrics['makespan_ms'], 'rfu_util': multicore.metrics['rfu_utilization'], 'energy_nj': multicore.metrics['energy_nj']})
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
        groth_morph, _ = self._simulate([groth_graph], self._runtime_tasks(groth_graph, 1), mode='morphaccel', scenario_name='fig9_groth16_morph', raw_dir=raw_dir)
        groth_deg, _ = self._simulate([groth_graph], self._runtime_tasks(groth_graph, 1), mode='degraded', scenario_name='fig9_groth16_degraded', raw_dir=raw_dir)
        rows.append({'task': 'Groth16', 'mode': ACCELERATOR_NAME, 'latency_ms': groth_morph.metrics['makespan_ms'], 'throughput': 1.0 / max(groth_morph.metrics['makespan_ms'], 1e-09)})
        rows.append({'task': 'Groth16', 'mode': 'Degraded', 'latency_ms': groth_deg.metrics['makespan_ms'], 'throughput': 1.0 / max(groth_deg.metrics['makespan_ms'], 1e-09)})
        vgg_graph = build_vgg16_like_task(64, 'vgg_ablate')
        vgg_morph, _ = self._simulate([vgg_graph], self._runtime_tasks(vgg_graph, 1), mode='morphaccel', scenario_name='fig9_vgg16_morph', raw_dir=raw_dir)
        vgg_deg, _ = self._simulate([vgg_graph], self._runtime_tasks(vgg_graph, 1), mode='degraded', scenario_name='fig9_vgg16_degraded', raw_dir=raw_dir)
        rows.append({'task': 'VGG16', 'mode': ACCELERATOR_NAME, 'latency_ms': vgg_morph.metrics['makespan_ms'], 'throughput': 1.0 / max(vgg_morph.metrics['makespan_ms'], 1e-09)})
        rows.append({'task': 'VGG16', 'mode': 'Degraded', 'latency_ms': vgg_deg.metrics['makespan_ms'], 'throughput': 1.0 / max(vgg_deg.metrics['makespan_ms'], 1e-09)})
        mlp_graph = build_mlp_path_task(64, 1024, 'mlp_ablate')
        mlp_morph, _ = self._simulate([mlp_graph], self._runtime_tasks(mlp_graph, 8, gap_ms=0.2), mode='morphaccel', scenario_name='fig9_mlp_morph', raw_dir=raw_dir)
        mlp_deg, _ = self._simulate([mlp_graph], self._runtime_tasks(mlp_graph, 8, gap_ms=0.2), mode='degraded', scenario_name='fig9_mlp_degraded', raw_dir=raw_dir)
        rows.append({'task': 'MLP', 'mode': ACCELERATOR_NAME, 'latency_ms': mlp_morph.metrics['makespan_ms'], 'throughput': 8.0 / max(mlp_morph.metrics['makespan_ms'], 1e-09)})
        rows.append({'task': 'MLP', 'mode': 'Degraded', 'latency_ms': mlp_deg.metrics['makespan_ms'], 'throughput': 8.0 / max(mlp_deg.metrics['makespan_ms'], 1e-09)})
        return pd.DataFrame(rows)

    def _raw_fig7(self, raw_dir: Path | None) -> pd.DataFrame:
        graphs = [build_groth16_like_task(8192, 4, 'area_zkp'), build_vgg16_like_task(8, 'area_cnn'), build_aes128_task(2048, 'area_aes')]
        runtime_tasks = [RuntimeTask(runtime_id='area_zkp_0', task_graph=graphs[0], arrival_cycle=0), RuntimeTask(runtime_id='area_cnn_0', task_graph=graphs[1], arrival_cycle=int(1.2 * CYCLES_PER_MS)), RuntimeTask(runtime_id='area_aes_0', task_graph=graphs[2], arrival_cycle=int(2.1 * CYCLES_PER_MS)), RuntimeTask(runtime_id='area_zkp_1', task_graph=graphs[0], arrival_cycle=int(3.0 * CYCLES_PER_MS))]
        morph, _ = self._simulate(graphs, runtime_tasks, mode='morphaccel', scenario_name='fig7_area_power_probe', raw_dir=raw_dir)
        estimate = self.toolkit.estimate_area_power(rfu_count=16, feature_bytes=8 * 1024 * 1024, fragment_bytes=2 * 1024 * 1024, microop_bytes=512 * 1024, utilization=float(morph.metrics.get('rfu_utilization', 0.0)), fvm_reads=float(morph.metrics.get('fvm_reads', 0.0)), fvm_writes=float(morph.metrics.get('fvm_writes', 0.0)), scheduler_decisions=float(morph.metrics.get('scheduler_decisions', 0.0)), switch_events=float(morph.metrics.get('lrc_switch_events', 0.0)))
        area_total = max(estimate['total_area_mm2'], 1e-09)
        power_total = max(estimate['total_power_w'], 1e-09)
        rows = [{'category': 'area', 'module': 'FVM + intermediate regs', 'share': 100.0 * estimate['fvm_area_mm2'] / area_total, 'value': estimate['fvm_area_mm2'], 'total': area_total, 'source': estimate['source']}, {'category': 'area', 'module': 'RFU', 'share': 100.0 * estimate['rfu_area_mm2'] / area_total, 'value': estimate['rfu_area_mm2'], 'total': area_total, 'source': estimate['source']}, {'category': 'area', 'module': 'TARS', 'share': 100.0 * estimate['tars_area_mm2'] / area_total, 'value': estimate['tars_area_mm2'], 'total': area_total, 'source': estimate['source']}, {'category': 'area', 'module': 'Others', 'share': 100.0 * estimate['other_area_mm2'] / area_total, 'value': estimate['other_area_mm2'], 'total': area_total, 'source': estimate['source']}, {'category': 'power', 'module': 'RFU dynamic', 'share': 100.0 * estimate['rfu_dynamic_w'] / power_total, 'value': estimate['rfu_dynamic_w'], 'total': power_total, 'source': estimate['source']}, {'category': 'power', 'module': 'Scheduler+FVM dynamic', 'share': 100.0 * estimate['scheduler_fvm_dynamic_w'] / power_total, 'value': estimate['scheduler_fvm_dynamic_w'], 'total': power_total, 'source': estimate['source']}, {'category': 'power', 'module': 'Static', 'share': 100.0 * estimate['static_w'] / power_total, 'value': estimate['static_w'], 'total': power_total, 'source': estimate['source']}]
        return pd.DataFrame(rows)

    def _raw_fig10(self, raw_dir: Path | None) -> pd.DataFrame:
        pair_specs = {'CNN+AES': [build_vgg16_like_task(8, 'pair_cnn'), build_aes128_task(1024, 'pair_aes')], 'ZKP+CNN': [build_groth16_like_task(8192, 4, 'pair_zkp'), build_vgg16_like_task(8, 'pair_cnn')], 'ZKP+AES': [build_groth16_like_task(8192, 4, 'pair_zkp'), build_aes128_task(1024, 'pair_aes')], 'All Tasks': [build_groth16_like_task(8192, 4, 'pair_zkp'), build_vgg16_like_task(8, 'pair_cnn'), build_aes128_task(1024, 'pair_aes')]}
        rows: List[Dict[str, object]] = []
        for pair, graphs in pair_specs.items():
            for batch in [1, 4, 8, 16]:
                runtime_tasks: List[RuntimeTask] = []
                arrival_ms = 0.0
                for graph in graphs:
                    for idx in range(batch):
                        runtime_tasks.append(RuntimeTask(runtime_id=f'{pair}_{graph.task_id}_{idx}', task_graph=graph, arrival_cycle=int(arrival_ms * CYCLES_PER_MS)))
                        arrival_ms += 0.18
                baseline, _ = self._simulate(graphs, runtime_tasks, mode='multicore', scenario_name=f'fig10_{pair}_{batch}_baseline', raw_dir=raw_dir)
                morph, _ = self._simulate(graphs, runtime_tasks, mode='morphaccel', scenario_name=f'fig10_{pair}_{batch}_morph', raw_dir=raw_dir)
                baseline_res = max(float(baseline.metrics.get('fvm_avg_residency_bytes', 0.0)), 1.0)
                baseline_red = max(float(baseline.metrics.get('fvm_redundant_reads', 0.0)), 1.0)
                rows.append({'pair': pair, 'metric': 'On-chip Memory Residency', 'arch': 'Fully Separated', 'batch_size': f'BS{batch}', 'value': 1.0, 'absolute': baseline_res})
                rows[-1]['source'] = 'simulator_metrics'
                rows.append({'pair': pair, 'metric': 'On-chip Memory Residency', 'arch': ACCELERATOR_NAME, 'batch_size': f'BS{batch}', 'value': float(morph.metrics.get('fvm_avg_residency_bytes', 0.0)) / baseline_res, 'absolute': float(morph.metrics.get('fvm_avg_residency_bytes', 0.0)), 'source': 'simulator_metrics'})
                rows.append({'pair': pair, 'metric': 'Redundant Accesses', 'arch': 'Fully Separated', 'batch_size': f'BS{batch}', 'value': 1.0, 'absolute': baseline_red, 'source': 'simulator_metrics'})
                rows.append({'pair': pair, 'metric': 'Redundant Accesses', 'arch': ACCELERATOR_NAME, 'batch_size': f'BS{batch}', 'value': float(morph.metrics.get('fvm_redundant_reads', 0.0)) / baseline_red, 'absolute': float(morph.metrics.get('fvm_redundant_reads', 0.0)), 'source': 'simulator_metrics'})
        return pd.DataFrame(rows)

    def _raw_fig11(self, raw_dir: Path | None) -> Dict[str, pd.DataFrame]:
        graphs = [build_groth16_like_task(8192, 4, 'sched_zkp'), build_vgg16_like_task(8, 'sched_cnn'), build_aes128_task(1024, 'sched_aes')]
        runtime_tasks = self._mixed_runtime_tasks(graphs, 4)
        morph, _ = self._simulate(graphs, runtime_tasks, mode='morphaccel', scenario_name='fig11_morph', raw_dir=raw_dir)
        multicore, _ = self._simulate(graphs, runtime_tasks, mode='multicore', scenario_name='fig11_multicore', raw_dir=raw_dir)
        fig11a = pd.DataFrame([{'bucket': 'Kappa-key', 'value': min(100.0, 100.0 * float(morph.metrics.get('kappa_key_match_ratio', 0.0))), 'source': 'simulator_metrics'}, {'bucket': 'Template', 'value': min(100.0, 100.0 * float(morph.metrics.get('template_hit_ratio', 0.0))), 'source': 'simulator_metrics'}, {'bucket': 'Valid Fragments', 'value': min(100.0, 100.0 * float(morph.metrics.get('valid_fragment_ratio', 0.0))), 'source': 'simulator_metrics'}])
        morph_switch_ms = float(morph.metrics.get('switch_cycles_total', 0.0)) / CYCLES_PER_MS
        multi_switch_ms = float(multicore.metrics.get('switch_cycles_total', 0.0)) / CYCLES_PER_MS
        morph_idle = 100.0 * float(morph.metrics.get('scheduler_pipeline_stall_cycles', 0.0)) / max(float(morph.metrics.get('makespan_cycles', 0.0)), 1.0)
        multi_idle = 100.0 * float(multicore.metrics.get('scheduler_pipeline_stall_cycles', 0.0)) / max(float(multicore.metrics.get('makespan_cycles', 0.0)), 1.0)
        ideal_switch_values: List[float] = []
        ideal_idle_values: List[float] = []
        ideal_buffer_ops_values: List[float] = []
        for graph in graphs:
            standalone_tasks = [RuntimeTask(runtime_id=f'ideal_{graph.task_id}_{idx}', task_graph=graph, arrival_cycle=0) for idx in range(4)]
            standalone, _ = self._simulate([graph], standalone_tasks, mode='morphaccel', scenario_name=f'fig11_ideal_{graph.task_id}', raw_dir=raw_dir)
            ideal_switch_values.append(float(standalone.metrics.get('switch_cycles_total', 0.0)) / CYCLES_PER_MS)
            ideal_idle_values.append(100.0 * float(standalone.metrics.get('scheduler_pipeline_stall_cycles', 0.0)) / max(float(standalone.metrics.get('makespan_cycles', 0.0)), 1.0))
            ideal_buffer_ops_values.append(float(standalone.metrics.get('fvm_allocations', 0.0) + standalone.metrics.get('fvm_frees', 0.0)))
        fig11b = pd.DataFrame([{'platform': 'Multi-core', 'metric': 'Scheduling Latency (ms)', 'value': multi_switch_ms, 'source': 'simulator_metrics'}, {'platform': ACCELERATOR_NAME, 'metric': 'Scheduling Latency (ms)', 'value': morph_switch_ms, 'source': 'simulator_metrics'}, {'platform': 'Multi-core', 'metric': 'Idle Ratio (%)', 'value': multi_idle, 'source': 'simulator_metrics'}, {'platform': ACCELERATOR_NAME, 'metric': 'Idle Ratio (%)', 'value': morph_idle, 'source': 'simulator_metrics'}])
        buffer_ops_multi = float(multicore.metrics.get('fvm_allocations', 0.0) + multicore.metrics.get('fvm_frees', 0.0))
        buffer_ops_morph = float(morph.metrics.get('fvm_allocations', 0.0) + morph.metrics.get('fvm_frees', 0.0))
        ideal_switch = float(np.mean(ideal_switch_values)) if ideal_switch_values else 0.0
        ideal_idle = float(np.mean(ideal_idle_values)) if ideal_idle_values else 0.0
        ideal_buffer_ops = float(np.mean(ideal_buffer_ops_values)) if ideal_buffer_ops_values else 0.0

        def normalize(observed: float, baseline: float) -> float:
            return 100.0 * observed / max(baseline, 1e-09)
        fig11c = pd.DataFrame([{'metric': 'Scheduling latency', 'platform': 'Multi-core', 'value': 100.0, 'absolute': multi_switch_ms, 'source': 'simulator_metrics'}, {'metric': 'Scheduling latency', 'platform': ACCELERATOR_NAME, 'value': normalize(morph_switch_ms, multi_switch_ms), 'absolute': morph_switch_ms, 'source': 'simulator_metrics'}, {'metric': 'Scheduling latency', 'platform': 'Ideal standalone', 'value': normalize(ideal_switch, multi_switch_ms), 'absolute': ideal_switch, 'source': 'simulator_metrics'}, {'metric': 'Idle ratio', 'platform': 'Multi-core', 'value': 100.0, 'absolute': multi_idle, 'source': 'simulator_metrics'}, {'metric': 'Idle ratio', 'platform': ACCELERATOR_NAME, 'value': normalize(morph_idle, multi_idle), 'absolute': morph_idle, 'source': 'simulator_metrics'}, {'metric': 'Idle ratio', 'platform': 'Ideal standalone', 'value': normalize(ideal_idle, multi_idle), 'absolute': ideal_idle, 'source': 'simulator_metrics'}, {'metric': 'Buffer ops', 'platform': 'Multi-core', 'value': 100.0, 'absolute': buffer_ops_multi, 'source': 'simulator_metrics'}, {'metric': 'Buffer ops', 'platform': ACCELERATOR_NAME, 'value': normalize(buffer_ops_morph, buffer_ops_multi), 'absolute': buffer_ops_morph, 'source': 'simulator_metrics'}, {'metric': 'Buffer ops', 'platform': 'Ideal standalone', 'value': normalize(ideal_buffer_ops, buffer_ops_multi), 'absolute': ideal_buffer_ops, 'source': 'simulator_metrics'}])
        return {'fig11a_sched': fig11a, 'fig11b_switch': fig11b, 'fig11c_summary': fig11c}

    def _raw_fig12(self, raw_dir: Path | None) -> pd.DataFrame:
        scenarios = [('ZKP->CNN', [build_groth16_like_task(8192, 4, 'rzkp'), build_vgg16_like_task(8, 'rcnn')]), ('CNN->AES', [build_vgg16_like_task(8, 'rcnn'), build_aes128_task(1024, 'raes')]), ('AES->ZKP', [build_aes128_task(1024, 'raes'), build_groth16_like_task(8192, 4, 'rzkp')]), ('ZKP->AES', [build_groth16_like_task(8192, 4, 'rzkp'), build_aes128_task(1024, 'raes')]), ('CNN->ZKP', [build_vgg16_like_task(8, 'rcnn'), build_groth16_like_task(8192, 4, 'rzkp')]), ('AES->CNN', [build_aes128_task(1024, 'raes'), build_vgg16_like_task(8, 'rcnn')])]
        rows: List[Dict[str, object]] = []
        for scenario, graphs in scenarios:
            runtime_tasks = [RuntimeTask(runtime_id=f'{scenario}_0', task_graph=graphs[0], arrival_cycle=0), RuntimeTask(runtime_id=f'{scenario}_1', task_graph=graphs[1], arrival_cycle=int(2.0 * CYCLES_PER_MS))]
            morph, _ = self._simulate(graphs, runtime_tasks, mode='morphaccel', scenario_name=f'fig12_{scenario}_morph', raw_dir=raw_dir)
            cpu_hw = self._measure_framework_switch(graphs, 'cpu', scenario, raw_dir)
            gpu_device = 'cuda:0' if self._cuda_available() else 'cpu'
            gpu_hw = self._measure_framework_switch(graphs, gpu_device, scenario, raw_dir)
            morph_switch = float(morph.metrics.get('switch_cycles_total', 0.0)) / CYCLES_PER_MS
            morph_recovery = self._measure_fuse_recovery_ms(graphs, runtime_tasks, scenario, raw_dir)
            rows.append({'scenario': scenario, 'platform': 'CPU', 'metric': 'Switch Latency (ms)', 'value': cpu_hw['total_ms'], 'source': cpu_hw['source']})
            rows.append({'scenario': scenario, 'platform': 'GPU', 'metric': 'Switch Latency (ms)', 'value': gpu_hw['total_ms'], 'source': gpu_hw['source']})
            rows.append({'scenario': scenario, 'platform': ACCELERATOR_NAME, 'metric': 'Switch Latency (ms)', 'value': morph_switch, 'source': 'simulator_metrics'})
            rows.append({'scenario': scenario, 'platform': ACCELERATOR_NAME, 'metric': 'Energy vs CPU (%)', 'value': 100.0 * float(morph.metrics.get('energy_nj', 0.0)) / max(cpu_hw['energy_nj'], 1e-09), 'source': 'simulator_metrics_vs_hardware'})
            rows.append({'scenario': scenario, 'platform': ACCELERATOR_NAME, 'metric': 'Energy vs GPU (%)', 'value': 100.0 * float(morph.metrics.get('energy_nj', 0.0)) / max(gpu_hw['energy_nj'], 1e-09), 'source': 'simulator_metrics_vs_hardware'})
            rows.append({'scenario': scenario, 'platform': ACCELERATOR_NAME, 'metric': 'Recovery Time (ms)', 'value': morph_recovery, 'source': 'simulator_events'})
        avg_cpu = np.mean([row['value'] for row in rows if row['platform'] == 'CPU' and row['metric'] == 'Switch Latency (ms)'])
        avg_gpu = np.mean([row['value'] for row in rows if row['platform'] == 'GPU' and row['metric'] == 'Switch Latency (ms)'])
        avg_morph = np.mean([row['value'] for row in rows if row['platform'] == ACCELERATOR_NAME and row['metric'] == 'Switch Latency (ms)'])
        avg_energy_cpu = np.mean([row['value'] for row in rows if row['metric'] == 'Energy vs CPU (%)'])
        avg_energy = np.mean([row['value'] for row in rows if row['metric'] == 'Energy vs GPU (%)'])
        avg_recovery = np.mean([row['value'] for row in rows if row['metric'] == 'Recovery Time (ms)'])
        rows.extend([{'scenario': 'Avg', 'platform': 'CPU', 'metric': 'Switch Latency (ms)', 'value': avg_cpu, 'source': 'aggregate'}, {'scenario': 'Avg', 'platform': 'GPU', 'metric': 'Switch Latency (ms)', 'value': avg_gpu, 'source': 'aggregate'}, {'scenario': 'Avg', 'platform': ACCELERATOR_NAME, 'metric': 'Switch Latency (ms)', 'value': avg_morph, 'source': 'aggregate'}, {'scenario': 'Avg', 'platform': ACCELERATOR_NAME, 'metric': 'Energy vs CPU (%)', 'value': avg_energy_cpu, 'source': 'aggregate'}, {'scenario': 'Avg', 'platform': ACCELERATOR_NAME, 'metric': 'Energy vs GPU (%)', 'value': avg_energy, 'source': 'aggregate'}, {'scenario': 'Avg', 'platform': ACCELERATOR_NAME, 'metric': 'Recovery Time (ms)', 'value': avg_recovery, 'source': 'aggregate'}])
        return pd.DataFrame(rows)

    def _cuda_available(self) -> bool:
        try:
            return HardwareBenchmarkSuite(device='auto', warmup=0, repeat=1, seed=self.seed).device.startswith('cuda')
        except Exception:
            return False

    def _measure_framework_switch(self, graphs: Sequence[TaskGraph], device_label: str, scenario: str, raw_dir: Path | None) -> Dict[str, float | str]:
        energy = EnergyBenchmarkSuite(device=device_label, warmup=2, repeat=5, seed=self.seed)
        hardware = HardwareBenchmarkSuite(device=device_label, warmup=0, repeat=1, seed=self.seed)
        first = hardware._make_task_kernel(graphs[0])
        second = hardware._make_task_kernel(graphs[1])
        for _ in range(energy.warmup):
            first['fn']()
            energy._sync()
            second['fn']()
            energy._sync()
        total_ms: List[float] = []
        recovery_ms: List[float] = []
        energy_j: List[float] = []
        raw_rows: List[Dict[str, object]] = []
        for rep in range(energy.repeat):
            energy._sync()
            cpu_before = energy.proc.cpu_times()
            wall_start = time.perf_counter()
            samples: List[float] = []
            stop = Event()
            thread = energy._start_power_sampler(samples, stop, device_label)
            iterations = 0
            recovery_sum_ms = 0.0
            while True:
                first['fn']()
                energy._sync()
                recovery_start = time.perf_counter()
                second['fn']()
                energy._sync()
                recovery_sum_ms += max((time.perf_counter() - recovery_start) * 1000.0, 0.0)
                iterations += 1
                if time.perf_counter() - wall_start >= energy.min_sample_s:
                    break
            wall_end = time.perf_counter()
            stop.set()
            if thread is not None:
                thread.join()
            cpu_after = energy.proc.cpu_times()
            wall_s = max(wall_end - wall_start, 1e-09)
            cpu_s = max(cpu_after.user + cpu_after.system - (cpu_before.user + cpu_before.system), 0.0)
            measured_energy_j, energy_source = energy._estimate_energy_j(samples, wall_s, cpu_s, device_label)
            pair_ms = wall_s * 1000.0 / max(iterations, 1)
            second_ms = recovery_sum_ms / max(iterations, 1)
            total_ms.append(pair_ms)
            recovery_ms.append(second_ms)
            energy_j.append(measured_energy_j / max(iterations, 1))
            raw_rows.append({'scenario': scenario, 'device': device_label, 'repeat': rep, 'iterations': iterations, 'pair_ms': pair_ms, 'recovery_ms': second_ms, 'energy_j': measured_energy_j / max(iterations, 1), 'energy_source': energy_source, 'first_backend': first['backend'], 'second_backend': second['backend']})
        source = 'hardware_gpu_runtime' if device_label.startswith('cuda') else 'hardware_cpu_runtime'
        if raw_dir is not None:
            safe_name = self._safe_name(f"fig12_{scenario}_{('gpu' if device_label.startswith('cuda') else 'cpu')}_switch")
            pd.DataFrame(raw_rows).to_csv(raw_dir / f'{safe_name}.csv', index=False)
        return {'total_ms': float(statistics.fmean(total_ms)), 'recovery_ms': float(statistics.fmean(recovery_ms)), 'energy_nj': float(statistics.fmean(energy_j)) * 1000000000.0, 'source': source}

    def _measure_fuse_recovery_ms(self, graphs: Sequence[TaskGraph], runtime_tasks: Sequence[RuntimeTask], scenario: str, raw_dir: Path | None) -> float:
        _, events = self._simulate(graphs, runtime_tasks, mode='morphaccel', scenario_name=f'fig12_{scenario}_morph_recovery', raw_dir=raw_dir)
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

    def _fit_calibration(self, rawframes: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        calibration: Dict[str, Dict[str, float]] = {'fig4': {}, 'fig6': {}}
        raw_fig4 = rawframes['fig4_raw']
        for domain in sorted(raw_fig4['domain'].unique()):
            calibration['fig4'][f'{domain}_morph_scale'] = 1.0
            calibration['fig4'][f'{domain}_baseline_scale'] = 1.0
        return calibration

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
            rows.append({'window_ms': window_ms, 'baseline_makespan_ms': base['makespan_ms'], 'morph_makespan_ms': morph['makespan_ms'], 'baseline_util': base['rfu_util'], 'morph_util': morph['rfu_util'], 'raw_baseline_makespan_ms': base['makespan_ms'], 'raw_baseline_util': base['rfu_util']})
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
        zkp_cnn = fig12[(fig12['scenario'] == 'ZKP->CNN') & (fig12['platform'] == ACCELERATOR_NAME) & (fig12['metric'] == 'Switch Latency (ms)')]
        cnn_aes = fig12[(fig12['scenario'] == 'CNN->AES') & (fig12['platform'] == ACCELERATOR_NAME) & (fig12['metric'] == 'Switch Latency (ms)')]
        recovery_avg = fig12[(fig12['scenario'] == 'Avg') & (fig12['platform'] == ACCELERATOR_NAME) & (fig12['metric'] == 'Recovery Time (ms)')]
        top_a = fig11a.copy()
        top_a['panel'] = 'a'
        top_a['panel_title'] = 'TARS fragment scheduling'
        top_b = fig11b.copy()
        top_b['panel'] = 'b'
        top_b['panel_title'] = 'Task switch latency and idle ratio'
        reduction_rows = []
        lookup = {'Latency Reduction': ('Scheduling latency', ACCELERATOR_NAME), 'Idle Ratio Reduction': ('Idle ratio', ACCELERATOR_NAME), 'Buffer Ops Reduction': ('Buffer ops', ACCELERATOR_NAME)}
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
        manifest = {'engine': self.engine, 'seed': self.seed, 'dataframes': sorted(list(artifacts.dataframes.keys())), 'rawframes': sorted(list(artifacts.rawframes.keys())), 'traces': sorted(list(artifacts.traces.keys())), 'scalars': artifacts.scalars, 'calibration': artifacts.calibration}
        (output_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

def load_calibration(path: Path | None) -> Dict[str, Dict[str, float]]:
    if path is None or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding='utf-8'))

def run_default_suite(output_dir: Path, engine: str='cpp', calibration_path: Path | None=None) -> BenchmarkArtifacts:
    suite = BenchmarkSuite(engine=engine, calibration=load_calibration(calibration_path))
    return suite.run(output_dir)
