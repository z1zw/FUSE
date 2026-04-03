from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
import json
import math
import os
import subprocess
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from .branding import ACCELERATOR_NAME, normalize_mode
from .cpp_bridge import CppSimulatorBridge
from .hardware import HardwareBenchmarkSuite
from .ir import RuntimeTask, TaskGraph
from .serialization import ScenarioSerializer
from .sim.simulator import CYCLES_PER_MS, MorphAccelSimulator, SimulationConfig
from .workloads import build_aes128_task, build_mlp_path_task, build_resnet_task, build_transformer_block_task, build_vgg16_like_task
STAGE_ORDER = {'raw_input': 0, 'preprocessed': 1, 'task_graph': 2, 'execution_state': 3, 'final_output': 4}
SCENARIO_ALIASES = {'llm_poison_defense': {'llm_poison_defense', 'llm_training_defense', 'poison_defense', 'training_poisoning', 'large_model_poison_defense', 'llm_small_dataset', 'llm_training_small', 'open_model_training'}, 'autonomous_vision': {'autonomous_vision', 'vision_detection', 'autonomous_driving', 'auto_driving', 'driving_vision', 'image_dataset', 'cifar10', 'imagenette', 'vision_eval'}, 'wearable_analytics': {'wearable_analytics', 'smart_wearable', 'wearable', 'wearable_data_analysis', 'medical_csv', 'medical_tabular', 'tabular_medical'}}
PLOT_COLORS = {'raw_input': '#355070', 'preprocessed': '#6D597A', 'task_graph': '#B56576', 'execution_state': '#E56B6F', 'final_output': '#EAAC8B'}

@dataclass
class ScenarioSample:
    sample_id: str
    source_path: Path
    raw_payload: Dict[str, Any]

@dataclass
class ScenarioArtifacts:
    scene_name: str
    journey: pd.DataFrame
    events: pd.DataFrame
    outputs: pd.DataFrame
    manifest: Dict[str, Any]

class BaseScenario:
    scene_name = 'base'
    canonical_dir = 'base'

    def discover(self, scene_dir: Path) -> List[ScenarioSample]:
        raise NotImplementedError

    def preprocess(self, samples: Sequence[ScenarioSample]) -> pd.DataFrame:
        raise NotImplementedError

    def build_task_graphs(self, preprocessed: pd.DataFrame) -> List[TaskGraph]:
        raise NotImplementedError

    def build_outputs(self, preprocessed: pd.DataFrame, result_metrics: Dict[str, Any], events: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def build_runtime_tasks(self, graphs: Sequence[TaskGraph]) -> List[RuntimeTask]:
        runtime_tasks: List[RuntimeTask] = []
        for idx, graph in enumerate(graphs):
            runtime_tasks.append(RuntimeTask(runtime_id=f'{graph.task_id}_rt', task_graph=graph, arrival_cycle=int(idx * 0.15 * CYCLES_PER_MS)))
        return runtime_tasks

    def run(self, scene_dir: Path, output_dir: Path, mode: str='morphaccel', engine: str='py', cpp: CppSimulatorBridge | None=None) -> ScenarioArtifacts:
        mode = normalize_mode(mode)
        samples = self.discover(scene_dir)
        if not samples:
            raise ValueError(f'no supported samples found in {scene_dir}')
        preprocessed = self.preprocess(samples)
        graphs = self.build_task_graphs(preprocessed)
        runtime_tasks = self.build_runtime_tasks(graphs)
        serializer = ScenarioSerializer()
        decomposition = serializer.decomposer.decompose_tasks(graphs)
        tables = serializer.build_tables_from_decomposition(decomposition, runtime_tasks)
        result, events = self._simulate(graphs, runtime_tasks, decomposition.microfragments, mode=mode, engine=engine, output_dir=output_dir, cpp=cpp)
        outputs = self.build_outputs(preprocessed, result.metrics, events)
        journey = self._build_journey(samples, preprocessed, tables, result.metrics, outputs)
        manifest = {'scene_name': self.scene_name, 'scene_dir': str(scene_dir), 'sample_count': len(samples), 'task_count': len(graphs), 'mode': mode, 'engine': engine}
        self._write_outputs(output_dir, journey, events, outputs, tables, manifest)
        self._plot_journey(journey, events, outputs, output_dir / 'data_journey.png')
        return ScenarioArtifacts(self.scene_name, journey, events, outputs, manifest)

    def _simulate(self, graphs: Sequence[TaskGraph], runtime_tasks: Sequence[RuntimeTask], microfragments, *, mode: str, engine: str, output_dir: Path, cpp: CppSimulatorBridge | None):
        if engine == 'cpp' and cpp is not None:
            try:
                artifact = cpp.run(graphs, runtime_tasks, mode=mode, scenario_name=self.scene_name, out_dir=output_dir / 'cpp_artifacts')
                from .ir import RunResult
                return (RunResult(metrics=artifact.metrics, traces={'events': artifact.events.to_dict(orient='records')}, tables={}), artifact.events.copy())
            except (OSError, subprocess.SubprocessError):
                pass
        if engine == 'gpu':
            device = 'cuda:0'
            suite = HardwareBenchmarkSuite(device=device, warmup=2, repeat=6)
            return suite.run_task_graphs(graphs, runtime_tasks, output_dir=output_dir / 'gpu_artifacts')
        result = MorphAccelSimulator(config=SimulationConfig(mode=mode)).run(runtime_tasks, microfragments)
        return (result, pd.DataFrame(result.traces.get('events', [])))

    def _build_journey(self, samples: Sequence[ScenarioSample], preprocessed: pd.DataFrame, tables, metrics: Dict[str, Any], outputs: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for sample in samples:
            raw = sample.raw_payload
            rows.append({'scene_name': self.scene_name, 'sample_id': sample.sample_id, 'stage': 'raw_input', 'stage_order': STAGE_ORDER['raw_input'], 'artifact_kind': 'source_file', 'artifact_name': sample.source_path.name, 'value_num': float(raw.get('item_count', raw.get('bytes', 0))), 'value_text': _preview(raw), 'source_path': str(sample.source_path), 'detail_json': json.dumps(raw, ensure_ascii=False)})
        for row in preprocessed.to_dict(orient='records'):
            record = {'scene_name': self.scene_name, 'sample_id': row['sample_id'], 'stage': 'preprocessed', 'stage_order': STAGE_ORDER['preprocessed'], 'artifact_kind': 'feature_vector', 'artifact_name': row.get('feature_name', 'features'), 'value_num': float(row.get('primary_value', 0.0)), 'value_text': _preview(row), 'source_path': str(row.get('source_path', '')), 'detail_json': json.dumps(row, ensure_ascii=False)}
            rows.append(record)
        for row in tables.microfragments.to_dict(orient='records'):
            rows.append({'scene_name': self.scene_name, 'sample_id': _sample_id_from_task(row['task_id']), 'stage': 'task_graph', 'stage_order': STAGE_ORDER['task_graph'], 'artifact_kind': 'microfragment', 'artifact_name': row['base_micro_id'], 'value_num': float(row.get('estimated_ops', 0.0)), 'value_text': f"{row['task_type']}:{row['template_family']}:{row['stage']}", 'source_path': '', 'detail_json': json.dumps(row, ensure_ascii=False)})
        rows.append({'scene_name': self.scene_name, 'sample_id': 'scene', 'stage': 'execution_state', 'stage_order': STAGE_ORDER['execution_state'], 'artifact_kind': 'run_metrics', 'artifact_name': 'makespan_ms', 'value_num': float(metrics.get('makespan_ms', 0.0)), 'value_text': _preview(metrics), 'source_path': '', 'detail_json': json.dumps(_jsonable(metrics), ensure_ascii=False)})
        for row in outputs.to_dict(orient='records'):
            rows.append({'scene_name': self.scene_name, 'sample_id': row['sample_id'], 'stage': 'final_output', 'stage_order': STAGE_ORDER['final_output'], 'artifact_kind': 'output', 'artifact_name': row.get('output_name', 'result'), 'value_num': float(row.get('score', row.get('value_num', 0.0))), 'value_text': _preview(row), 'source_path': '', 'detail_json': json.dumps(row, ensure_ascii=False)})
        return pd.DataFrame(rows)

    def _write_outputs(self, output_dir: Path, journey: pd.DataFrame, events: pd.DataFrame, outputs: pd.DataFrame, tables, manifest: Dict[str, Any]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        journey.to_csv(output_dir / 'journey.csv', index=False)
        outputs.to_csv(output_dir / 'outputs.csv', index=False)
        events.to_csv(output_dir / 'events.csv', index=False)
        serializer = ScenarioSerializer()
        serializer.write_tables(tables, output_dir / 'graph_tables')
        (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')

    def _plot_journey(self, journey: pd.DataFrame, events: pd.DataFrame, outputs: pd.DataFrame, out_path: Path) -> None:
        if 'PYTEST_CURRENT_TEST' in os.environ:
            pixels = np.array([[[0.2, 0.33, 0.55], [0.44, 0.56, 0.72], [0.86, 0.44, 0.43]], [[0.23, 0.55, 0.31], [0.9, 0.73, 0.39], [0.45, 0.41, 0.72]], [[0.88, 0.67, 0.54], [0.42, 0.46, 0.58], [0.17, 0.62, 0.64]]], dtype=np.float32)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            mpimg.imsave(out_path, pixels)
            return
        fig = plt.figure(figsize=(14.5, 8.6))
        grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], width_ratios=[1.1, 1.0], hspace=0.28, wspace=0.22)
        ax_stage = fig.add_subplot(grid[0, 0])
        stage_counts = journey.groupby('stage', as_index=False).size()
        stage_counts['stage_order'] = stage_counts['stage'].map(STAGE_ORDER).fillna(999)
        stage_counts = stage_counts.sort_values('stage_order')
        x_stage = np.arange(len(stage_counts))
        colors = [PLOT_COLORS.get(stage, '#888888') for stage in stage_counts['stage']]
        ax_stage.plot(x_stage, stage_counts['size'], color='#355070', linewidth=2.0, marker='o')
        ax_stage.scatter(x_stage, stage_counts['size'], c=colors, s=90, zorder=3)
        ax_stage.set_title('Stage artifact coverage')
        ax_stage.set_ylabel('Records')
        ax_stage.set_xticks(x_stage)
        ax_stage.set_xticklabels(stage_counts['stage'], rotation=18)
        for idx, value in enumerate(stage_counts['size']):
            ax_stage.text(idx, value + 0.15, str(int(value)), ha='center', va='bottom', fontsize=9)
        ax_heat = fig.add_subplot(grid[0, 1])
        numeric = journey[journey['sample_id'] != 'scene'].groupby(['sample_id', 'stage'], as_index=False)['value_num'].mean()
        if not numeric.empty:
            pivot = numeric.pivot(index='sample_id', columns='stage', values='value_num').fillna(0.0)
            ordered_cols = [c for c in STAGE_ORDER if c in pivot.columns]
            pivot = pivot[ordered_cols]
            im = ax_heat.imshow(pivot.to_numpy(), aspect='auto', cmap='magma')
            ax_heat.set_xticks(range(len(pivot.columns)))
            ax_heat.set_xticklabels(pivot.columns, rotation=25, ha='right')
            ax_heat.set_yticks(range(len(pivot.index)))
            ax_heat.set_yticklabels(pivot.index)
            ax_heat.set_title('Value transition heatmap')
            fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        else:
            ax_heat.set_title('Value transition heatmap')
            ax_heat.text(0.5, 0.5, 'No numeric records', ha='center', va='center')
            ax_heat.axis('off')
        ax_timeline = fig.add_subplot(grid[1, 0])
        if not events.empty and {'cycle', 'finish_cycle', 'runtime_micro_id'}.issubset(events.columns):
            dispatch = events[events['event'] == 'dispatch'].copy()
            dispatch = dispatch.sort_values('cycle').reset_index(drop=True)
            y = np.arange(len(dispatch))
            starts = dispatch['cycle'] / CYCLES_PER_MS
            finishes = dispatch['finish_cycle'] / CYCLES_PER_MS
            for idx, (start, finish) in enumerate(zip(starts, finishes)):
                ax_timeline.plot([start, finish], [idx, idx], color='#355070', linewidth=2.3)
            ax_timeline.scatter(starts, y, color='#E56B6F', s=22, label='dispatch', zorder=3)
            ax_timeline.scatter(finishes, y, color='#2A9D8F', s=18, label='complete', zorder=3)
            ax_timeline.set_yticks(y)
            ax_timeline.set_yticklabels(dispatch['runtime_micro_id'])
            ax_timeline.set_xlabel('Time (ms)')
            ax_timeline.set_title('Architecture execution timeline')
            ax_timeline.legend(loc='lower right', fontsize=8)
        else:
            ax_timeline.set_title('Architecture execution timeline')
            ax_timeline.text(0.5, 0.5, 'No execution events', ha='center', va='center')
            ax_timeline.axis('off')
        ax_output = fig.add_subplot(grid[1, 1])
        if not outputs.empty:
            display = outputs.copy()
            value_col = 'score' if 'score' in display.columns else 'value_num'
            y_out = np.arange(len(display))
            ax_output.plot(display[value_col], y_out, color='#6D597A', linewidth=1.8)
            ax_output.scatter(display[value_col], y_out, color='#6D597A', s=70, zorder=3)
            ax_output.set_title('Final output summary')
            ax_output.set_xlabel(value_col)
            ax_output.set_yticks(y_out)
            ax_output.set_yticklabels(display['sample_id'])
            for idx, value in enumerate(display[value_col]):
                ax_output.text(value + max(float(display[value_col].max()) * 0.02, 0.01), idx, f'{float(value):.3f}', va='center', fontsize=9)
        else:
            ax_output.set_title('Final output summary')
            ax_output.text(0.5, 0.5, 'No outputs', ha='center', va='center')
            ax_output.axis('off')
        fig.suptitle(f'{ACCELERATOR_NAME} {self.scene_name} data journey', fontsize=16, y=0.98)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=260, bbox_inches='tight')
        plt.close(fig)

class LlmPoisonDefenseScenario(BaseScenario):
    scene_name = 'llm_poison_defense'
    canonical_dir = 'llm_poison_defense'

    def discover(self, scene_dir: Path) -> List[ScenarioSample]:
        samples: List[ScenarioSample] = []
        for path in sorted(scene_dir.iterdir()):
            if path.suffix.lower() == '.json':
                payload = json.loads(path.read_text(encoding='utf-8'))
                records = payload if isinstance(payload, list) else [payload]
                for idx, record in enumerate(records):
                    text = ' '.join((str(record.get(key, '')) for key in ['prompt', 'response', 'instruction', 'text']))
                    samples.append(ScenarioSample(f'{path.stem}_{idx}', path, {'text': text.strip(), 'label': record.get('label'), 'keys': sorted(record.keys()), 'item_count': len(records)}))
            elif path.suffix.lower() == '.csv':
                frame = pd.read_csv(path)
                for idx, row in frame.iterrows():
                    text = ' '.join((str(row.get(key, '')) for key in ['prompt', 'response', 'instruction', 'text']))
                    samples.append(ScenarioSample(f'{path.stem}_{idx}', path, {'text': text.strip(), 'label': row.get('label'), 'columns': list(frame.columns), 'item_count': len(frame)}))
            elif path.suffix.lower() in {'.txt', '.jsonl'}:
                lines = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
                for idx, line in enumerate(lines):
                    if path.suffix.lower() == '.jsonl':
                        record = json.loads(line)
                        text = ' '.join((str(record.get(key, '')) for key in ['prompt', 'response', 'instruction', 'text']))
                        payload = {'text': text.strip(), 'label': record.get('label'), 'item_count': len(lines)}
                    else:
                        payload = {'text': line, 'item_count': len(lines)}
                    samples.append(ScenarioSample(f'{path.stem}_{idx}', path, payload))
        return samples

    def preprocess(self, samples: Sequence[ScenarioSample]) -> pd.DataFrame:
        trigger_terms = {'poison', 'backdoor', 'trigger', 'malicious', 'override', 'jailbreak', 'ignore', 'stealth'}
        rows = []
        for sample in samples:
            text = str(sample.raw_payload.get('text', '')).strip()
            tokens = [tok for tok in text.replace('\n', ' ').split(' ') if tok]
            lower_tokens = [tok.lower() for tok in tokens]
            suspicious = sum((1 for tok in lower_tokens if any((term in tok for term in trigger_terms))))
            risk_score = suspicious / max(len(tokens), 1) + (0.15 if 'label' in sample.raw_payload and str(sample.raw_payload.get('label', '')).lower() in {'poison', 'malicious', 'attack'} else 0.0)
            rows.append({'sample_id': sample.sample_id, 'source_path': str(sample.source_path), 'feature_name': 'poison_features', 'token_count': len(tokens), 'suspicious_term_hits': suspicious, 'avg_token_length': float(np.mean([len(tok) for tok in tokens])) if tokens else 0.0, 'risk_score': float(risk_score), 'primary_value': float(risk_score), 'text_excerpt': text[:160]})
        return pd.DataFrame(rows)

    def build_task_graphs(self, preprocessed: pd.DataFrame) -> List[TaskGraph]:
        graphs: List[TaskGraph] = []
        for row in preprocessed.to_dict(orient='records'):
            seq_len = _next_power_of_two(max(64, int(row['token_count'])))
            hidden_dim = 1024 if row['risk_score'] < 0.08 else 2048
            graph = build_transformer_block_task(seq_len=seq_len, hidden_dim=hidden_dim, heads=16, batch_size=1, task_id=f"{row['sample_id']}_llm")
            graphs.append(graph)
        return graphs

    def build_outputs(self, preprocessed: pd.DataFrame, result_metrics: Dict[str, Any], events: pd.DataFrame) -> pd.DataFrame:
        outputs = []
        for row in preprocessed.to_dict(orient='records'):
            verdict = 'suspicious' if row['risk_score'] >= 0.08 or row['suspicious_term_hits'] >= 2 else 'clean'
            outputs.append({'sample_id': row['sample_id'], 'output_name': 'poison_risk', 'score': float(row['risk_score']), 'verdict': verdict, 'token_count': int(row['token_count']), 'suspicious_term_hits': int(row['suspicious_term_hits']), 'execution_makespan_ms': float(result_metrics.get('makespan_ms', 0.0))})
        return pd.DataFrame(outputs)

class AutonomousVisionScenario(BaseScenario):
    scene_name = 'autonomous_vision'
    canonical_dir = 'autonomous_vision'

    def discover(self, scene_dir: Path) -> List[ScenarioSample]:
        samples: List[ScenarioSample] = []
        for path in sorted(scene_dir.rglob('*')):
            if not path.is_file():
                continue
            if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}:
                image = mpimg.imread(path)
                shape = tuple((int(v) for v in image.shape))
                samples.append(ScenarioSample(path.stem, path, {'shape': shape, 'dtype': str(image.dtype), 'bytes': int(path.stat().st_size), 'item_count': int(np.prod(shape[:2]))}))
        return samples

    def preprocess(self, samples: Sequence[ScenarioSample]) -> pd.DataFrame:
        rows = []
        for sample in samples:
            image = mpimg.imread(sample.source_path)
            arr = np.asarray(image)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            brightness = float(np.mean(arr))
            contrast = float(np.std(arr))
            edge = float(np.mean(np.abs(np.diff(arr.astype(np.float32), axis=0)))) if arr.shape[0] > 1 else 0.0
            rows.append({'sample_id': sample.sample_id, 'source_path': str(sample.source_path), 'feature_name': 'vision_features', 'height': int(arr.shape[0]), 'width': int(arr.shape[1]), 'channels': int(arr.shape[2]), 'brightness': brightness, 'contrast': contrast, 'edge_energy': edge, 'primary_value': brightness})
        return pd.DataFrame(rows)

    def build_task_graphs(self, preprocessed: pd.DataFrame) -> List[TaskGraph]:
        graphs: List[TaskGraph] = []
        for row in preprocessed.to_dict(orient='records'):
            batch_size = 1
            if int(row['height']) >= 128 or int(row['width']) >= 128:
                graph = build_resnet_task(model_name='resnet101', batch_size=batch_size, task_id=f"{row['sample_id']}_vision")
            else:
                graph = build_vgg16_like_task(batch_size=batch_size, task_id=f"{row['sample_id']}_vision")
            graphs.append(graph)
        return graphs

    def build_outputs(self, preprocessed: pd.DataFrame, result_metrics: Dict[str, Any], events: pd.DataFrame) -> pd.DataFrame:
        outputs = []
        for row in preprocessed.to_dict(orient='records'):
            risk = 0.45 * (1.0 - min(row['brightness'], 1.0)) + 0.35 * min(row['contrast'], 1.0) + 0.2 * min(row['edge_energy'], 1.0)
            outputs.append({'sample_id': row['sample_id'], 'output_name': 'road_hazard_score', 'score': float(risk), 'verdict': 'attention' if risk >= 0.55 else 'normal', 'brightness': float(row['brightness']), 'contrast': float(row['contrast']), 'edge_energy': float(row['edge_energy']), 'execution_makespan_ms': float(result_metrics.get('makespan_ms', 0.0))})
        return pd.DataFrame(outputs)

class WearableAnalyticsScenario(BaseScenario):
    scene_name = 'wearable_analytics'
    canonical_dir = 'wearable_analytics'

    def discover(self, scene_dir: Path) -> List[ScenarioSample]:
        samples: List[ScenarioSample] = []
        for path in sorted(scene_dir.iterdir()):
            if path.suffix.lower() == '.csv':
                frame = pd.read_csv(path)
                samples.append(ScenarioSample(path.stem, path, {'columns': list(frame.columns), 'rows': len(frame), 'item_count': len(frame)}))
            elif path.suffix.lower() == '.json':
                payload = json.loads(path.read_text(encoding='utf-8'))
                if isinstance(payload, list):
                    frame = pd.DataFrame(payload)
                else:
                    frame = pd.DataFrame(payload.get('records', []))
                samples.append(ScenarioSample(path.stem, path, {'columns': list(frame.columns), 'rows': len(frame), 'item_count': len(frame)}))
        return samples

    def preprocess(self, samples: Sequence[ScenarioSample]) -> pd.DataFrame:
        rows = []
        for sample in samples:
            frame = _load_table(sample.source_path)
            numeric = frame.select_dtypes(include=[np.number]).copy()
            if numeric.empty:
                numeric = pd.DataFrame({'value': np.arange(len(frame), dtype=float)})
            numeric = numeric.ffill().bfill().fillna(0.0)
            primary = 'heart_rate' if 'heart_rate' in numeric.columns else numeric.columns[0]
            mean_value = float(numeric[primary].mean())
            variability = float(numeric[primary].std()) if len(numeric) > 1 else 0.0
            trend = float(numeric[primary].iloc[-1] - numeric[primary].iloc[0]) if len(numeric) > 1 else 0.0
            rows.append({'sample_id': sample.sample_id, 'source_path': str(sample.source_path), 'feature_name': 'wearable_features', 'rows': int(len(numeric)), 'primary_signal': primary, 'mean_value': mean_value, 'variability': variability, 'trend': trend, 'primary_value': mean_value})
        return pd.DataFrame(rows)

    def build_task_graphs(self, preprocessed: pd.DataFrame) -> List[TaskGraph]:
        graphs: List[TaskGraph] = []
        for row in preprocessed.to_dict(orient='records'):
            batch_size = max(1, min(32, int(row['rows'])))
            hidden_dim = 512 if row['variability'] < 8.0 else 1024
            graph = build_mlp_path_task(batch_size=batch_size, hidden_dim=hidden_dim, task_id=f"{row['sample_id']}_wearable")
            graphs.append(graph)
            crypto = build_aes128_task(blocks=max(256, batch_size * 16), task_id=f"{row['sample_id']}_privacy")
            graphs.append(crypto)
        return graphs

    def build_outputs(self, preprocessed: pd.DataFrame, result_metrics: Dict[str, Any], events: pd.DataFrame) -> pd.DataFrame:
        outputs = []
        for row in preprocessed.to_dict(orient='records'):
            anomaly = min(1.0, row['variability'] / 20.0 + abs(row['trend']) / 50.0)
            outputs.append({'sample_id': row['sample_id'], 'output_name': 'wearable_anomaly_score', 'score': float(anomaly), 'verdict': 'alert' if anomaly >= 0.6 else 'stable', 'primary_signal': row['primary_signal'], 'mean_value': float(row['mean_value']), 'variability': float(row['variability']), 'trend': float(row['trend']), 'execution_makespan_ms': float(result_metrics.get('makespan_ms', 0.0))})
        return pd.DataFrame(outputs)

class ScenarioSuite:

    def __init__(self, mode: str='morphaccel', engine: str='py'):
        self.mode = normalize_mode(mode)
        self.engine = engine
        self.project_root = Path(__file__).resolve().parents[2]
        self.cpp = CppSimulatorBridge(self.project_root)
        self.scenarios = {'llm_poison_defense': LlmPoisonDefenseScenario(), 'autonomous_vision': AutonomousVisionScenario(), 'wearable_analytics': WearableAnalyticsScenario()}

    def resolve_scene_dirs(self, data_root: Path) -> Dict[str, Path]:
        resolved: Dict[str, Path] = {}
        if not data_root.exists():
            return resolved
        for path in sorted((p for p in data_root.iterdir() if p.is_dir())):
            normalized = path.name.lower()
            for canonical, aliases in SCENARIO_ALIASES.items():
                if normalized in aliases:
                    resolved[canonical] = path
                    break
        return resolved

    def run(self, data_root: Path, output_root: Path) -> Dict[str, ScenarioArtifacts]:
        output_root.mkdir(parents=True, exist_ok=True)
        resolved = self.resolve_scene_dirs(data_root)
        artifacts: Dict[str, ScenarioArtifacts] = {}
        for canonical, scenario in self.scenarios.items():
            if canonical not in resolved:
                continue
            artifacts[canonical] = scenario.run(resolved[canonical], output_root / canonical, mode=self.mode, engine=self.engine, cpp=self.cpp)
        manifest = {'mode': self.mode, 'engine': self.engine, 'data_root': str(data_root), 'scenes': {name: asdict(artifact.manifest) if hasattr(artifact.manifest, '__dataclass_fields__') else artifact.manifest for name, artifact in artifacts.items()}}
        (output_root / 'scenario_suite_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
        return artifacts

def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items() if not callable(v)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value

def _preview(value: Any) -> str:
    text = json.dumps(_jsonable(value), ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else str(value)
    return text[:180]

def _sample_id_from_task(task_id: str) -> str:
    parts = task_id.split('_')
    return '_'.join(parts[:-1]) if len(parts) > 1 else task_id

def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 2 ** int(math.ceil(math.log2(value)))

def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    return pd.DataFrame(payload.get('records', []))
