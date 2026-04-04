from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence
import json
import random
import statistics
import time
import numpy as np
import pandas as pd
from .branding import ACCELERATOR_NAME
from .hardware import HardwareBenchmarkSuite, _optional_torch
from .ir import RuntimeTask, TaskGraph
from .serialization import ScenarioSerializer
from .sim.simulator import CYCLES_PER_MS
from .workloads import build_resnet_task, build_transformer_block_task, build_vgg16_like_task

def _optional_import(name: str):
    try:
        return __import__(name, fromlist=['*'])
    except Exception:
        return None
torch = _optional_torch()
torchvision = _optional_import('torchvision')
tv_models = _optional_import('torchvision.models')
tv_transforms = _optional_import('torchvision.transforms')
transformers = _optional_import('transformers')
PILImage = None
try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None

def _plt():
    import matplotlib.pyplot as plt
    return plt

@dataclass
class CloudSample:
    sample_id: str
    payload_path: Path
    label: str
    image_path: Path | None = None
    text: str = ''

@dataclass
class BenchmarkRow:
    dataset: str
    model: str
    mode: str
    batch_size: int
    sample_count: int
    mean_ms: float
    p50_ms: float
    p99_ms: float
    throughput: float
    device: str
    backend: str

class CloudExperimentSuite:

    def __init__(self, device: str='auto', warmup: int=2, repeat: int=5, seed: int=7, sample_count: int=6, pretrained: bool=True):
        self.device = self._resolve_device(device)
        self.warmup = warmup
        self.repeat = repeat
        self.seed = seed
        self.sample_count = sample_count
        self.pretrained = pretrained
        self.rng = random.Random(seed)
        self.hardware = HardwareBenchmarkSuite(device=self.device, warmup=warmup, repeat=repeat, seed=seed)
        self.serializer = ScenarioSerializer()

    def _resolve_device(self, requested: str) -> str:
        if requested != 'auto':
            return requested
        if torch is not None and torch.cuda.is_available():
            return 'cuda:0'
        return 'cpu'

    def run(self, data_root: Path, output_root: Path, dataset_roots: Dict[str, Path] | None=None) -> Dict[str, pd.DataFrame]:
        output_root.mkdir(parents=True, exist_ok=True)
        dataset_roots = dataset_roots or {}
        manifests: List[Dict[str, Any]] = []
        summaries: List[pd.DataFrame] = []
        outputs: List[pd.DataFrame] = []
        sample_tables: List[pd.DataFrame] = []
        configs = [
            ('cifar10', self._load_cifar10, [{'name': 'vgg16', 'kind': 'cnn'}], 'vgg16'),
            ('coco2017', self._load_coco2017, [{'name': 'clip_vit_b32', 'kind': 'clip'}], 'clip_vit_b32'),
            ('alpaca', self._load_alpaca, [{'name': 'alpaca_transformer', 'kind': 'llm'}], None),
        ]
        for dataset_name, loader, models, representative in configs:
            dataset_dir = dataset_roots.get(dataset_name, data_root / dataset_name)
            if not dataset_dir.exists():
                continue
            samples = loader(dataset_dir)
            if not samples:
                continue
            dataset_out = output_root / dataset_name
            result = self._run_dataset(dataset_name, samples, models, representative, dataset_out)
            manifests.append(result['manifest'])
            summaries.append(result['summary'])
            outputs.append(result['outputs'])
            if not result['samples'].empty:
                sample_tables.append(result['samples'])
        summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
        outputs_df = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
        sample_df = pd.concat(sample_tables, ignore_index=True) if sample_tables else pd.DataFrame()
        if not summary_df.empty:
            summary_df.to_csv(output_root / 'cloud_benchmark_summary.csv', index=False)
        if not outputs_df.empty:
            outputs_df.to_csv(output_root / 'cloud_model_outputs.csv', index=False)
        if not sample_df.empty:
            sample_df.to_csv(output_root / 'cloud_sample_comparisons.csv', index=False)
        (output_root / 'cloud_manifest.json').write_text(json.dumps(manifests, indent=2, ensure_ascii=False), encoding='utf-8')
        return {'summary': summary_df, 'outputs': outputs_df, 'sample_comparisons': sample_df}

    def _run_dataset(self, dataset_name: str, samples: Sequence[CloudSample], models: Sequence[Dict[str, str]], representative: str, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_rows: List[Dict[str, Any]] = []
        output_frames: List[pd.DataFrame] = []
        sample_compare = pd.DataFrame()
        labels = self._labels_for_dataset(dataset_name, samples)
        for model_spec in models:
            model_out = out_dir / model_spec['name']
            model_out.mkdir(parents=True, exist_ok=True)
            native_row, outputs_df = self._run_native(dataset_name, samples, model_spec['name'], model_spec['kind'], labels, model_out)
            fuse_row, fuse_metrics = self._run_fuse(dataset_name, model_spec['name'], model_spec['kind'], outputs_df['sample_id'].tolist(), model_out)
            comparison = pd.DataFrame([{'dataset': dataset_name, 'model': model_spec['name'], 'native_mean_ms': native_row.mean_ms, 'fuse_mean_ms': fuse_row.mean_ms, 'latency_ratio': fuse_row.mean_ms / max(native_row.mean_ms, 1e-09), 'native_throughput': native_row.throughput, 'fuse_throughput': fuse_row.throughput}])
            comparison.to_csv(model_out / 'comparison.csv', index=False)
            summary_rows.append(asdict(native_row))
            summary_rows.append(asdict(fuse_row))
            output_frames.append(outputs_df.assign(dataset=dataset_name, model=model_spec['name']))
            if representative and model_spec['name'] == representative and (dataset_name in {'cifar10', 'coco2017'}):
                sample_compare = self._render_samples(dataset_name, model_spec['name'], samples, outputs_df, fuse_metrics, out_dir / f'{representative}_sample_grid.png')
        summary_df = pd.DataFrame(summary_rows)
        outputs_df = pd.concat(output_frames, ignore_index=True) if output_frames else pd.DataFrame()
        summary_df.to_csv(out_dir / 'benchmark_summary.csv', index=False)
        outputs_df.to_csv(out_dir / 'model_outputs.csv', index=False)
        if not sample_compare.empty:
            sample_compare.to_csv(out_dir / 'sample_comparison.csv', index=False)
        manifest = {'dataset': dataset_name, 'sample_count': len(samples), 'device': self.device, 'models': [model['name'] for model in models]}
        (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
        return {'manifest': manifest, 'summary': summary_df, 'outputs': outputs_df, 'samples': sample_compare}

    def _load_cifar10(self, dataset_dir: Path) -> List[CloudSample]:
        labels = self._load_label_csv(dataset_dir / 'labels.csv')
        image_files = sorted([path for path in dataset_dir.rglob('*') if path.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
        return [CloudSample(sample_id=path.stem, payload_path=path, image_path=path, label=labels.get(path.stem, labels.get(path.name, path.parent.name))) for path in image_files]

    def _load_coco2017(self, dataset_dir: Path) -> List[CloudSample]:
        image_dir = dataset_dir / 'val2017'
        annotation_path = dataset_dir / 'annotations' / 'instances_val2017.json'
        if not image_dir.exists() or not annotation_path.exists():
            return []
        payload = json.loads(annotation_path.read_text(encoding='utf-8'))
        category_names = {int(row['id']): row['name'] for row in payload.get('categories', [])}
        images = {int(row['id']): row['file_name'] for row in payload.get('images', [])}
        labels: Dict[int, List[str]] = {}
        for ann in payload.get('annotations', []):
            labels.setdefault(int(ann['image_id']), []).append(category_names.get(int(ann['category_id']), str(ann['category_id'])))
        samples: List[CloudSample] = []
        for image_id, file_name in images.items():
            image_path = image_dir / file_name
            if image_path.exists():
                label = '|'.join(sorted(set(labels.get(image_id, [])))) or 'unknown'
                samples.append(CloudSample(sample_id=str(image_id), payload_path=image_path, image_path=image_path, label=label))
        return samples

    def _load_alpaca(self, dataset_dir: Path) -> List[CloudSample]:
        samples: List[CloudSample] = []
        for path in sorted(dataset_dir.iterdir()):
            if path.suffix.lower() == '.json':
                payload = json.loads(path.read_text(encoding='utf-8'))
                for idx, record in enumerate(payload if isinstance(payload, list) else [payload]):
                    samples.append(CloudSample(sample_id=f'{path.stem}_{idx}', payload_path=path, label=str(record.get('output', ''))[:96], text=self._alpaca_text(record)))
            elif path.suffix.lower() == '.jsonl':
                for idx, line in enumerate(path.read_text(encoding='utf-8').splitlines()):
                    if line.strip():
                        record = json.loads(line)
                        samples.append(CloudSample(sample_id=f'{path.stem}_{idx}', payload_path=path, label=str(record.get('output', ''))[:96], text=self._alpaca_text(record)))
            elif path.suffix.lower() == '.csv':
                frame = pd.read_csv(path)
                for idx, row in frame.iterrows():
                    record = row.to_dict()
                    samples.append(CloudSample(sample_id=f'{path.stem}_{idx}', payload_path=path, label=str(record.get('output', ''))[:96], text=self._alpaca_text(record)))
        return samples

    def _alpaca_text(self, record: Dict[str, Any]) -> str:
        parts = [str(record.get('instruction', '')).strip(), str(record.get('input', '')).strip()]
        return '\n'.join([part for part in parts if part])

    def _load_label_csv(self, path: Path) -> Dict[str, str]:
        if not path.exists():
            return {}
        frame = pd.read_csv(path)
        if {'sample_id', 'label'}.issubset(frame.columns):
            return {str(row['sample_id']): str(row['label']) for row in frame.to_dict(orient='records')}
        if {'file_name', 'label'}.issubset(frame.columns):
            return {str(row['file_name']): str(row['label']) for row in frame.to_dict(orient='records')}
        return {}

    def _labels_for_dataset(self, dataset_name: str, samples: Sequence[CloudSample]) -> List[str]:
        if dataset_name == 'cifar10':
            return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        if dataset_name == 'coco2017':
            return sorted({label for sample in samples for label in sample.label.split('|') if label and label != 'unknown'})[:80]
        return []

    def _run_native(self, dataset_name: str, samples: Sequence[CloudSample], model_name: str, model_kind: str, labels: Sequence[str], out_dir: Path):
        batches = [list(samples[idx:idx + self._batch_size(dataset_name)]) for idx in range(0, min(len(samples), 24), self._batch_size(dataset_name))]
        runner = self._image_runner(model_name, model_kind, labels) if dataset_name in {'cifar10', 'coco2017'} else self._text_runner(model_name, model_kind)
        prepared = [runner['prepare'](batch) for batch in batches]
        stats = self._measure(lambda: runner['infer'](prepared[0]), float(len(batches[0])))
        rows: List[Dict[str, Any]] = []
        for batch, payload in zip(batches, prepared):
            rows.extend(runner['summarize'](batch, runner['infer'](payload)))
        outputs = pd.DataFrame(rows)
        outputs.to_csv(out_dir / 'native_outputs.csv', index=False)
        return (BenchmarkRow(dataset_name, model_name, 'native', self._batch_size(dataset_name), len(samples), stats['mean_ms'], stats['p50_ms'], stats['p99_ms'], stats['throughput'], self.device, runner['backend']), outputs)

    def _run_fuse(self, dataset_name: str, model_name: str, model_kind: str, sample_ids: Sequence[str], out_dir: Path):
        graph = self._task_graph(dataset_name, model_name, model_kind, self._batch_size(dataset_name))
        runtime_tasks = [RuntimeTask(runtime_id=f'{graph.task_id}_{idx}', task_graph=graph, arrival_cycle=int(idx * 0.1 * CYCLES_PER_MS)) for idx, _ in enumerate(sample_ids)]
        decomposition = self.serializer.decomposer.decompose_tasks([graph])
        tables = self.serializer.build_tables_from_decomposition(decomposition, runtime_tasks)
        result, events = self.hardware.run_task_graphs([graph], runtime_tasks, output_dir=out_dir / 'fuse_runtime')
        tables.tasks.to_csv(out_dir / 'tasks.csv', index=False)
        tables.microfragments.to_csv(out_dir / 'microfragments.csv', index=False)
        tables.representative_paths.to_csv(out_dir / 'representative_paths.csv', index=False)
        events.to_csv(out_dir / 'fuse_events.csv', index=False)
        metrics = pd.DataFrame([{'sample_id': sample_id, 'fuse_latency_ms': float(result.metrics.get('task_latency_ms', {}).get(runtime_task.runtime_id, 0.0)), 'task_runtime_id': runtime_task.runtime_id} for sample_id, runtime_task in zip(sample_ids, runtime_tasks)])
        metrics.to_csv(out_dir / 'fuse_sample_metrics.csv', index=False)
        row = BenchmarkRow(dataset_name, model_name, ACCELERATOR_NAME, self._batch_size(dataset_name), len(sample_ids), float(result.metrics['makespan_ms']) / max(len(sample_ids), 1), float(result.metrics['makespan_ms']) / max(len(sample_ids), 1), float(result.metrics['makespan_ms']) / max(len(sample_ids), 1), float(len(sample_ids)) / max(float(result.metrics['makespan_ms']) / 1000.0, 1e-09), self.device, str(result.metrics.get('backend', '')))
        return (row, metrics)

    def _batch_size(self, dataset_name: str) -> int:
        return 4 if dataset_name == 'alpaca' else 8

    def _measure(self, fn, work_items: float):
        latencies: List[float] = []
        for _ in range(self.warmup):
            fn()
        for _ in range(self.repeat):
            self._sync()
            start = time.perf_counter_ns()
            fn()
            self._sync()
            latencies.append((time.perf_counter_ns() - start) / 1000000.0)
        return {'mean_ms': statistics.fmean(latencies), 'p50_ms': float(np.percentile(latencies, 50)), 'p99_ms': float(np.percentile(latencies, 99)), 'throughput': work_items / max(statistics.fmean(latencies) / 1000.0, 1e-09)}

    def _sync(self):
        if torch is not None and self.device.startswith('cuda'):
            torch.cuda.synchronize()

    def _task_graph(self, dataset_name: str, model_name: str, model_kind: str, batch_size: int) -> TaskGraph:
        if model_name == 'vgg16':
            graph = build_vgg16_like_task(batch_size=batch_size, task_id=f'{dataset_name}_vgg16')
        elif model_name == 'resnet101':
            graph = build_resnet_task('resnet101', batch_size=batch_size, task_id=f'{dataset_name}_resnet101')
        else:
            seq_len = 197 if model_kind in {'vit', 'clip'} else 256
            graph = build_transformer_block_task(seq_len=seq_len, hidden_dim=768, heads=12, batch_size=batch_size, task_id=f"{dataset_name}_{model_name.replace('/', '_')}")
        graph.metadata['dataset'] = dataset_name
        graph.metadata['model'] = model_name
        graph.metadata['batch_size'] = batch_size
        return graph

    def _image_runner(self, model_name: str, model_kind: str, labels: Sequence[str]):
        if model_kind == 'clip':
            return self._clip_runner(labels)
        return self._torchvision_runner(model_name)

    def _text_runner(self, model_name: str, model_kind: str):
        if torch is None or transformers is None:
            return self._numpy_text_runner(model_name, model_kind)
        tokenizer_cls = getattr(transformers, 'AutoTokenizer', None)
        model_cls = getattr(transformers, 'AutoModelForCausalLM', None) if model_kind == 'decoder' else getattr(transformers, 'AutoModel', None)
        if tokenizer_cls is None or model_cls is None:
            return self._numpy_text_runner(model_name, model_kind)
        tokenizer = tokenizer_cls.from_pretrained(model_name)
        model = model_cls.from_pretrained(model_name).to(self.device)
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

        def prepare(batch: Sequence[CloudSample]):
            encoded = tokenizer([sample.text for sample in batch], return_tensors='pt', padding=True, truncation=True, max_length=256)
            return {key: value.to(self.device) for key, value in encoded.items()}

        def infer(payload: Dict[str, Any]):
            with torch.inference_mode():
                return model(**payload)

        def summarize(batch: Sequence[CloudSample], result):
            rows = []
            if model_kind == 'decoder':
                logits = result.logits[:, -1, :].float()
                scores, indices = torch.softmax(logits, dim=-1).max(dim=-1)
                for sample, score, index in zip(batch, scores.tolist(), indices.tolist()):
                    rows.append({'sample_id': sample.sample_id, 'label': sample.label, 'prediction': tokenizer.decode([int(index)]).strip(), 'score': float(score), 'text_excerpt': sample.text[:180]})
            else:
                norms = torch.linalg.vector_norm(result.last_hidden_state[:, 0, :].float(), dim=-1)
                for sample, score in zip(batch, norms.tolist()):
                    rows.append({'sample_id': sample.sample_id, 'label': sample.label, 'prediction': 'cls_embedding', 'score': float(score), 'text_excerpt': sample.text[:180]})
            return rows
        return {'backend': 'transformers_decoder' if model_kind == 'decoder' else 'transformers_bert', 'prepare': prepare, 'infer': infer, 'summarize': summarize}

    def _numpy_text_runner(self, model_name: str, model_kind: str):

        def prepare(batch: Sequence[CloudSample]):
            return np.array([len(sample.text.split()) for sample in batch], dtype=np.float32)

        def infer(payload):
            return payload * (0.2 if model_kind == 'bert' else 0.1) + 1.0

        def summarize(batch: Sequence[CloudSample], result):
            return [{'sample_id': sample.sample_id, 'label': sample.label, 'prediction': 'cls_embedding' if model_kind == 'bert' else 'token_forecast', 'score': float(score), 'text_excerpt': sample.text[:180]} for sample, score in zip(batch, np.ravel(result).tolist())]
        return {'backend': 'numpy_cpu', 'prepare': prepare, 'infer': infer, 'summarize': summarize}

    def _torchvision_runner(self, model_name: str):
        if torch is None or tv_models is None or tv_transforms is None or (PILImage is None):
            return self._numpy_image_runner()
        weights = None
        transform = None
        categories = [f'class_{idx}' for idx in range(1000)]
        if model_name == 'vgg16':
            enum = getattr(tv_models, 'VGG16_Weights', None)
            model_fn = tv_models.vgg16
        elif model_name == 'resnet50':
            enum = getattr(tv_models, 'ResNet50_Weights', None)
            model_fn = tv_models.resnet50
        elif model_name == 'resnet101':
            enum = getattr(tv_models, 'ResNet101_Weights', None)
            model_fn = tv_models.resnet101
        else:
            enum = getattr(tv_models, 'ViT_B_16_Weights', None)
            model_fn = tv_models.vit_b_16
        if self.pretrained and enum is not None:
            try:
                weights = enum.DEFAULT
            except Exception:
                weights = None
        model = model_fn(weights=weights).to(self.device)
        model.eval()
        if weights is not None:
            transform = weights.transforms()
            categories = list(weights.meta.get('categories', categories))
        else:
            transform = tv_transforms.Compose([tv_transforms.Resize((224, 224)), tv_transforms.ToTensor(), tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        def prepare(batch: Sequence[CloudSample]):
            return torch.stack([transform(PILImage.open(sample.image_path).convert('RGB')) for sample in batch], dim=0).to(self.device)

        def infer(payload):
            with torch.inference_mode():
                return model(payload)

        def summarize(batch: Sequence[CloudSample], result):
            probs = torch.softmax(result.float(), dim=-1)
            scores, indices = probs.max(dim=-1)
            return [{'sample_id': sample.sample_id, 'payload_path': str(sample.payload_path), 'label': sample.label, 'prediction': categories[int(index)] if int(index) < len(categories) else f'class_{int(index)}', 'score': float(score)} for sample, score, index in zip(batch, scores.tolist(), indices.tolist())]
        return {'backend': 'torchvision_cuda' if self.device.startswith('cuda') else 'torchvision_cpu', 'prepare': prepare, 'infer': infer, 'summarize': summarize}

    def _numpy_image_runner(self):
        labels = self._labels_for_dataset('cifar10', [])

        def prepare(batch: Sequence[CloudSample]):
            data = []
            plt = _plt()
            for sample in batch:
                image = np.asarray(plt.imread(sample.image_path), dtype=np.float32)
                if image.ndim == 2:
                    image = np.repeat(image[..., None], 3, axis=-1)
                data.append(np.mean(image, axis=(0, 1)))
            return np.stack(data, axis=0)

        def infer(payload):
            logits = payload @ np.linspace(0.5, 1.5, payload.shape[-1], dtype=np.float32).reshape(-1, 1)
            return np.repeat(logits, len(labels), axis=1) + np.linspace(0.0, 1.0, len(labels), dtype=np.float32)

        def summarize(batch: Sequence[CloudSample], result):
            indices = np.argmax(result, axis=-1)
            scores = np.max(result, axis=-1)
            return [{'sample_id': sample.sample_id, 'payload_path': str(sample.payload_path), 'label': sample.label, 'prediction': labels[int(index)] if labels else f'class_{int(index)}', 'score': float(score)} for sample, index, score in zip(batch, indices, scores)]
        return {'backend': 'numpy_cpu', 'prepare': prepare, 'infer': infer, 'summarize': summarize}

    def _clip_runner(self, labels: Sequence[str]):
        if torch is None or transformers is None or PILImage is None:
            return self._numpy_image_runner()
        model_cls = getattr(transformers, 'CLIPModel', None)
        proc_cls = getattr(transformers, 'CLIPProcessor', None)
        if model_cls is None or proc_cls is None:
            return self._numpy_image_runner()
        model_id = 'openai/clip-vit-base-patch32'
        processor = proc_cls.from_pretrained(model_id)
        model = model_cls.from_pretrained(model_id).to(self.device)
        model.eval()
        prompts = [f'a photo of a {label}' for label in labels] if labels else ['a photo of an object']

        def prepare(batch: Sequence[CloudSample]):
            payload = processor(text=prompts, images=[PILImage.open(sample.image_path).convert('RGB') for sample in batch], return_tensors='pt', padding=True)
            return {key: value.to(self.device) if hasattr(value, 'to') else value for key, value in payload.items()}

        def infer(payload):
            with torch.inference_mode():
                return model(**payload)

        def summarize(batch: Sequence[CloudSample], result):
            probs = torch.softmax(result.logits_per_image.float(), dim=-1)
            scores, indices = probs.max(dim=-1)
            return [{'sample_id': sample.sample_id, 'payload_path': str(sample.payload_path), 'label': sample.label, 'prediction': prompts[int(index)].replace('a photo of a ', ''), 'score': float(score)} for sample, score, index in zip(batch, scores.tolist(), indices.tolist())]
        return {'backend': 'transformers_clip', 'prepare': prepare, 'infer': infer, 'summarize': summarize}

    def _render_samples(self, dataset_name: str, model_name: str, samples: Sequence[CloudSample], outputs_df: pd.DataFrame, fuse_metrics: pd.DataFrame, out_path: Path) -> pd.DataFrame:
        image_samples = [sample for sample in samples if sample.image_path is not None]
        chosen = self.rng.sample(image_samples, k=min(self.sample_count, len(image_samples))) if image_samples else []
        if not chosen:
            return pd.DataFrame()
        plt = _plt()
        joined = outputs_df.merge(fuse_metrics, on='sample_id', how='left')
        fig, axes = plt.subplots(2, 3, figsize=(15.6, 9.2))
        rows: List[Dict[str, Any]] = []
        for ax, sample in zip(axes.flatten(), chosen):
            image = plt.imread(sample.image_path)
            row = joined[joined['sample_id'] == sample.sample_id].head(1)
            prediction = str(row.iloc[0]['prediction']) if not row.empty else 'n/a'
            score = float(row.iloc[0]['score']) if not row.empty else 0.0
            fuse_latency = float(row.iloc[0]['fuse_latency_ms']) if not row.empty and 'fuse_latency_ms' in row else 0.0
            ax.imshow(image)
            ax.set_title(f'{sample.sample_id}\nlabel={sample.label}\npred={prediction} ({score:.3f})\nfuse={fuse_latency:.3f} ms', fontsize=9)
            ax.axis('off')
            rows.append({'dataset': dataset_name, 'model': model_name, 'sample_id': sample.sample_id, 'image_path': str(sample.image_path), 'label': sample.label, 'prediction': prediction, 'score': score, 'fuse_latency_ms': fuse_latency})
        for ax in axes.flatten()[len(chosen):]:
            ax.axis('off')
        fig.suptitle(f'{ACCELERATOR_NAME} sample comparison: {dataset_name} / {model_name}', fontsize=16)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=260, bbox_inches='tight')
        plt.close(fig)
        return pd.DataFrame(rows)
