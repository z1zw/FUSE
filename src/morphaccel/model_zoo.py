from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Sequence
import json
import math
import numpy as np
import pandas as pd
from .branding import ACCELERATOR_NAME
from .cloud_suite import CloudExperimentSuite, CloudSample
from .ir import RuntimeTask, TaskGraph
from .sim.simulator import CYCLES_PER_MS
from .workloads import build_clip_vit_b32_task, build_resnet_task, build_vgg16_like_task
DEFAULT_COCO_ROOT = Path('/root/autodl-pub/COCO2017')
DEFAULT_CIFAR_ROOT = Path('/root/autodl-pub/cifar-10')
MODEL_SPECS = [{'name': 'vgg16', 'kind': 'cnn', 'label': 'VGG16'}, {'name': 'resnet50', 'kind': 'cnn', 'label': 'ResNet50'}, {'name': 'resnet101', 'kind': 'cnn', 'label': 'ResNet101'}, {'name': 'clip_vit_b32', 'kind': 'clip', 'label': 'CLIP ViT-B/32'}]
PARAMETER_COUNT_M = {'vgg16': 138.36, 'resnet50': 25.56, 'resnet101': 44.55, 'clip_vit_b32': 151.28}

def _plt():
    import matplotlib.pyplot as plt
    return plt
DATASET_BATCH_SIZE = {'cifar10': 1, 'coco2017': 1}

class ModelZooExperimentSuite(CloudExperimentSuite):

    def __init__(self, device: str='auto', warmup: int=2, repeat: int=5, seed: int=7, sample_count: int=10, pretrained: bool=True):
        super().__init__(device=device, warmup=warmup, repeat=repeat, seed=seed, sample_count=sample_count, pretrained=pretrained)

    def run_model_zoo(self, output_root: Path, *, cifar_root: Path=DEFAULT_CIFAR_ROOT, coco_root: Path=DEFAULT_COCO_ROOT) -> Dict[str, pd.DataFrame]:
        output_root.mkdir(parents=True, exist_ok=True)
        summary_rows: List[Dict[str, Any]] = []
        sample_rows: List[Dict[str, Any]] = []
        output_rows: List[Dict[str, Any]] = []
        manifests: List[Dict[str, Any]] = []
        datasets = [('cifar10', Path(cifar_root), self._load_cifar10), ('coco2017', Path(coco_root), self._load_coco2017)]
        for dataset_name, dataset_root, loader in datasets:
            if not dataset_root.exists():
                manifests.append({'dataset': dataset_name, 'root': str(dataset_root), 'present': False})
                continue
            all_samples = loader(dataset_root)
            image_samples = [sample for sample in all_samples if sample.image_path is not None]
            if not image_samples:
                manifests.append({'dataset': dataset_name, 'root': str(dataset_root), 'present': True, 'sample_count': 0})
                continue
            chosen = self._select_samples(image_samples, self.sample_count)
            labels = self._labels_for_dataset(dataset_name, chosen)
            dataset_out = output_root / dataset_name
            dataset_out.mkdir(parents=True, exist_ok=True)
            for spec in MODEL_SPECS:
                model_out = dataset_out / spec['name']
                model_out.mkdir(parents=True, exist_ok=True)
                native_runner = self._image_runner(spec['name'], spec['kind'], labels)
                parameter_count_m = float(native_runner.get('parameter_count_m', PARAMETER_COUNT_M[spec['name']]))
                native_df = self._run_native_per_sample(dataset_name, chosen, spec, native_runner, parameter_count_m, model_out)
                fuse_df = self._run_fuse_per_sample(dataset_name, chosen, spec, parameter_count_m, model_out)
                joined = native_df.merge(fuse_df[['sample_id', 'fuse_latency_ms', 'fuse_task_runtime_id']], on='sample_id', how='left')
                joined['latency_speedup'] = joined['native_latency_ms'] / joined['fuse_latency_ms'].clip(lower=1e-09)
                joined['dataset'] = dataset_name
                joined['model'] = spec['name']
                joined['model_label'] = spec['label']
                joined['parameter_count_m'] = parameter_count_m
                joined.to_csv(model_out / 'sample_comparison.csv', index=False)
                output_rows.extend(joined.to_dict(orient='records'))
                summary_rows.extend(self._summary_rows(dataset_name, spec, parameter_count_m, joined))
                sample_rows.extend(self._render_model_grid(dataset_name, spec, parameter_count_m, chosen, joined, model_out / f"{spec['name']}_sample_grid.png"))
            manifests.append({'dataset': dataset_name, 'root': str(dataset_root), 'present': True, 'sample_ids': [sample.sample_id for sample in chosen], 'sample_count': len(chosen), 'models': [spec['name'] for spec in MODEL_SPECS], 'device': self.device})
        summary_df = pd.DataFrame(summary_rows)
        outputs_df = pd.DataFrame(output_rows)
        samples_df = pd.DataFrame(sample_rows)
        if not summary_df.empty:
            summary_df.to_csv(output_root / 'model_zoo_summary.csv', index=False)
        if not outputs_df.empty:
            outputs_df.to_csv(output_root / 'model_zoo_outputs.csv', index=False)
        if not samples_df.empty:
            samples_df.to_csv(output_root / 'model_zoo_sample_grids.csv', index=False)
        (output_root / 'model_zoo_manifest.json').write_text(json.dumps(manifests, indent=2, ensure_ascii=False), encoding='utf-8')
        if not summary_df.empty:
            self._plot_latency(summary_df, output_root / 'model_zoo_latency.png')
            self._plot_throughput(summary_df, output_root / 'model_zoo_throughput.png')
            self._plot_params_vs_latency(summary_df, output_root / 'model_zoo_params_latency.png')
        return {'summary': summary_df, 'outputs': outputs_df, 'samples': samples_df}

    def _select_samples(self, samples: Sequence[CloudSample], count: int) -> List[CloudSample]:
        count = min(max(1, count), len(samples))
        if count == len(samples):
            return list(samples)
        return self.rng.sample(list(samples), k=count)

    def _run_native_per_sample(self, dataset_name: str, samples: Sequence[CloudSample], spec: Dict[str, str], runner: Dict[str, Any], parameter_count_m: float, out_dir: Path) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for sample in samples:
            payload = runner['prepare']([sample])
            stats = self._measure(lambda: runner['infer'](payload), 1.0)
            prediction_rows = runner['summarize']([sample], runner['infer'](payload))
            pred = prediction_rows[0] if prediction_rows else {}
            rows.append({'dataset': dataset_name, 'sample_id': sample.sample_id, 'image_path': str(sample.image_path) if sample.image_path is not None else '', 'label': sample.label, 'model': spec['name'], 'model_label': spec['label'], 'mode': 'native', 'backend': runner['backend'], 'parameter_count_m': parameter_count_m, 'native_latency_ms': float(stats['mean_ms']), 'native_p50_ms': float(stats['p50_ms']), 'native_p99_ms': float(stats['p99_ms']), 'native_throughput': float(stats['throughput']), 'prediction': str(pred.get('prediction', '')), 'score': float(pred.get('score', 0.0))})
        frame = pd.DataFrame(rows)
        frame.to_csv(out_dir / 'native_per_sample.csv', index=False)
        return frame

    def _run_fuse_per_sample(self, dataset_name: str, samples: Sequence[CloudSample], spec: Dict[str, str], parameter_count_m: float, out_dir: Path) -> pd.DataFrame:
        graph = self._task_graph(dataset_name, spec['name'], spec['kind'], DATASET_BATCH_SIZE[dataset_name])
        graph.metadata['parameter_count_m'] = parameter_count_m
        runtime_tasks = [RuntimeTask(runtime_id=f'{graph.task_id}_{idx}', task_graph=graph, arrival_cycle=int(idx * 0.12 * CYCLES_PER_MS)) for idx, _ in enumerate(samples)]
        result, events = self.hardware.run_task_graphs([graph], runtime_tasks, output_dir=out_dir / 'fuse_runtime')
        rows = []
        task_latency = result.metrics.get('task_latency_ms', {})
        for sample, runtime_task in zip(samples, runtime_tasks):
            rows.append({'dataset': dataset_name, 'sample_id': sample.sample_id, 'model': spec['name'], 'mode': ACCELERATOR_NAME, 'backend': str(result.metrics.get('backend', '')), 'parameter_count_m': parameter_count_m, 'fuse_latency_ms': float(task_latency.get(runtime_task.runtime_id, 0.0)), 'fuse_task_runtime_id': runtime_task.runtime_id})
        frame = pd.DataFrame(rows)
        frame.to_csv(out_dir / 'fuse_per_sample.csv', index=False)
        if not events.empty:
            events.to_csv(out_dir / 'fuse_events.csv', index=False)
        return frame

    def _summary_rows(self, dataset_name: str, spec: Dict[str, str], parameter_count_m: float, joined: pd.DataFrame) -> List[Dict[str, Any]]:
        rows = []
        native_mean = float(joined['native_latency_ms'].mean())
        native_p50 = float(joined['native_latency_ms'].median())
        native_p99 = float(np.percentile(joined['native_latency_ms'], 99))
        native_tp = float(1000.0 / max(native_mean, 1e-09))
        fuse_mean = float(joined['fuse_latency_ms'].mean())
        fuse_p50 = float(joined['fuse_latency_ms'].median())
        fuse_p99 = float(np.percentile(joined['fuse_latency_ms'], 99))
        fuse_tp = float(1000.0 / max(fuse_mean, 1e-09))
        rows.append({'dataset': dataset_name, 'model': spec['name'], 'model_label': spec['label'], 'mode': 'native', 'parameter_count_m': parameter_count_m, 'mean_ms': native_mean, 'p50_ms': native_p50, 'p99_ms': native_p99, 'throughput': native_tp, 'sample_count': len(joined), 'backend': str(joined['backend'].iloc[0])})
        rows.append({'dataset': dataset_name, 'model': spec['name'], 'model_label': spec['label'], 'mode': ACCELERATOR_NAME, 'parameter_count_m': parameter_count_m, 'mean_ms': fuse_mean, 'p50_ms': fuse_p50, 'p99_ms': fuse_p99, 'throughput': fuse_tp, 'sample_count': len(joined), 'backend': 'fuse_runtime'})
        return rows

    def _render_model_grid(self, dataset_name: str, spec: Dict[str, str], parameter_count_m: float, samples: Sequence[CloudSample], joined: pd.DataFrame, out_path: Path) -> List[Dict[str, Any]]:
        cols = 5
        rows = max(1, math.ceil(len(samples) / cols))
        plt = _plt()
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows))
        axes_arr = np.array(axes).reshape(rows, cols)
        records: List[Dict[str, Any]] = []
        for ax, sample in zip(axes_arr.flatten(), samples):
            image = plt.imread(sample.image_path)
            row = joined[joined['sample_id'] == sample.sample_id].head(1)
            pred = str(row.iloc[0]['prediction']) if not row.empty else 'n/a'
            native_ms = float(row.iloc[0]['native_latency_ms']) if not row.empty else 0.0
            fuse_ms = float(row.iloc[0]['fuse_latency_ms']) if not row.empty else 0.0
            score = float(row.iloc[0]['score']) if not row.empty else 0.0
            ax.imshow(image)
            ax.set_title(f'{sample.sample_id}\nlabel={sample.label}\npred={pred} ({score:.2f})\nbase={native_ms:.2f} ms | fuse={fuse_ms:.2f} ms', fontsize=8.6)
            ax.axis('off')
            records.append({'dataset': dataset_name, 'model': spec['name'], 'model_label': spec['label'], 'parameter_count_m': parameter_count_m, 'sample_id': sample.sample_id, 'image_path': str(sample.image_path), 'label': sample.label, 'grid_path': str(out_path)})
        for ax in axes_arr.flatten()[len(samples):]:
            ax.axis('off')
        fig.suptitle(f"{dataset_name}: {spec['label']} vs {ACCELERATOR_NAME} ({parameter_count_m:.2f}M params)", fontsize=16)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=280, bbox_inches='tight')
        plt.close(fig)
        return records

    def _task_graph(self, dataset_name: str, model_name: str, model_kind: str, batch_size: int) -> TaskGraph:
        if model_name == 'vgg16':
            graph = build_vgg16_like_task(batch_size=batch_size, task_id=f'{dataset_name}_vgg16')
        elif model_name == 'resnet50':
            graph = build_resnet_task('resnet50', batch_size=batch_size, task_id=f'{dataset_name}_resnet50')
        elif model_name == 'resnet101':
            graph = build_resnet_task('resnet101', batch_size=batch_size, task_id=f'{dataset_name}_resnet101')
        elif model_name == 'clip_vit_b32':
            graph = build_clip_vit_b32_task(batch_size=batch_size, task_id=f'{dataset_name}_clip_vit_b32')
        else:
            raise ValueError(f'unsupported model for model zoo: {model_name}')
        graph.metadata['dataset'] = dataset_name
        graph.metadata['model'] = model_name
        graph.metadata['batch_size'] = batch_size
        return graph

    def _image_runner(self, model_name: str, model_kind: str, labels: Sequence[str]):
        runner = super()._image_runner(model_name, model_kind, labels)
        runner['parameter_count_m'] = PARAMETER_COUNT_M[model_name]
        return runner

    def _plot_latency(self, summary_df: pd.DataFrame, out_path: Path) -> None:
        plt = _plt()
        fig, axes = plt.subplots(1, 2, figsize=(15.6, 5.0), sharey=False)
        for ax, dataset_name in zip(axes, ['cifar10', 'coco2017']):
            part = summary_df[summary_df['dataset'] == dataset_name].copy()
            models = [spec['name'] for spec in MODEL_SPECS if spec['name'] in set(part['model'])]
            x = np.arange(len(models))
            width = 0.36
            native_vals = [float(part[(part['model'] == model) & (part['mode'] == 'native')]['mean_ms'].iloc[0]) for model in models]
            fuse_vals = [float(part[(part['model'] == model) & (part['mode'] == ACCELERATOR_NAME)]['mean_ms'].iloc[0]) for model in models]
            native_bars = ax.bar(x - width / 2.0, native_vals, width=width, color='#34495E', label='Native')
            fuse_bars = ax.bar(x + width / 2.0, fuse_vals, width=width, color='#E76F51', label=ACCELERATOR_NAME)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{self._display_name(model)}\n{PARAMETER_COUNT_M[model]:.1f}M' for model in models], rotation=0)
            ax.set_ylabel('Mean latency per image (ms)')
            ax.set_title(dataset_name.upper())
            for bar in list(native_bars) + list(fuse_bars):
                value = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, value + max(0.02, value * 0.03), f'{value:.2f}', ha='center', va='bottom', fontsize=8.5)
        fig.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.03))
        fig.suptitle('Model-zoo latency on real datasets', fontsize=15)
        fig.savefig(out_path, dpi=320, bbox_inches='tight')
        plt.close(fig)

    def _plot_throughput(self, summary_df: pd.DataFrame, out_path: Path) -> None:
        plt = _plt()
        fig, axes = plt.subplots(1, 2, figsize=(15.6, 5.0), sharey=False)
        for ax, dataset_name in zip(axes, ['cifar10', 'coco2017']):
            part = summary_df[summary_df['dataset'] == dataset_name].copy()
            models = [spec['name'] for spec in MODEL_SPECS if spec['name'] in set(part['model'])]
            x = np.arange(len(models))
            width = 0.36
            native_vals = [float(part[(part['model'] == model) & (part['mode'] == 'native')]['throughput'].iloc[0]) for model in models]
            fuse_vals = [float(part[(part['model'] == model) & (part['mode'] == ACCELERATOR_NAME)]['throughput'].iloc[0]) for model in models]
            native_bars = ax.bar(x - width / 2.0, native_vals, width=width, color='#2878B5', label='Native')
            fuse_bars = ax.bar(x + width / 2.0, fuse_vals, width=width, color='#2A9D8F', label=ACCELERATOR_NAME)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{self._display_name(model)}\n{PARAMETER_COUNT_M[model]:.1f}M' for model in models], rotation=0)
            ax.set_ylabel('Throughput (images/s)')
            ax.set_title(dataset_name.upper())
            for bar in list(native_bars) + list(fuse_bars):
                value = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, value + max(0.02, value * 0.03), f'{value:.1f}', ha='center', va='bottom', fontsize=8.5)
        fig.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.03))
        fig.suptitle('Model-zoo throughput on real datasets', fontsize=15)
        fig.savefig(out_path, dpi=320, bbox_inches='tight')
        plt.close(fig)

    def _plot_params_vs_latency(self, summary_df: pd.DataFrame, out_path: Path) -> None:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(9.8, 6.2))
        markers = {'native': 'o', ACCELERATOR_NAME: 's'}
        colors = {'cifar10': '#5A189A', 'coco2017': '#1B998B'}
        for _, row in summary_df.iterrows():
            ax.scatter(float(row['parameter_count_m']), float(row['mean_ms']), s=120, marker=markers[str(row['mode'])], color=colors[str(row['dataset'])], alpha=0.85, edgecolors='#202124', linewidths=0.6)
            ax.annotate(f"{row['model_label']} ({row['dataset']}, {row['mode']})", (float(row['parameter_count_m']), float(row['mean_ms'])), textcoords='offset points', xytext=(6, 6), fontsize=8.3)
        ax.set_xlabel('Parameter count (M)')
        ax.set_ylabel('Mean latency per image (ms)')
        ax.set_title('Parameter scale vs latency')
        handles = [plt.Line2D([0], [0], marker='o', color='w', label='Native', markerfacecolor='#666666', markeredgecolor='#202124', markersize=8), plt.Line2D([0], [0], marker='s', color='w', label=ACCELERATOR_NAME, markerfacecolor='#666666', markeredgecolor='#202124', markersize=8), plt.Line2D([0], [0], marker='o', color='w', label='CIFAR10', markerfacecolor=colors['cifar10'], markeredgecolor='#202124', markersize=8), plt.Line2D([0], [0], marker='o', color='w', label='COCO2017', markerfacecolor=colors['coco2017'], markeredgecolor='#202124', markersize=8)]
        ax.legend(handles=handles, loc='upper left')
        fig.savefig(out_path, dpi=320, bbox_inches='tight')
        plt.close(fig)

    def _display_name(self, model_name: str) -> str:
        for spec in MODEL_SPECS:
            if spec['name'] == model_name:
                return str(spec['label'])
        return model_name
