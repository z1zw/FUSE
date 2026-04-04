from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence
import json
import math
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
tv_datasets = _optional_import('torchvision.datasets')
tv_models = _optional_import('torchvision.models')
tv_transforms = _optional_import('torchvision.transforms')
transformers = _optional_import('transformers')
_TorchDatasetBase = torch.utils.data.Dataset if torch is not None else object

@dataclass
class TrainResult:
    model: str
    output_dir: Path
    summary: pd.DataFrame
    process: pd.DataFrame
    validation: pd.DataFrame
    fuse_step_metrics: pd.DataFrame

class TrainingExport:

    def __init__(self, device: str='auto', warmup: int=2, repeat: int=5, seed: int=7):
        self.device = self._resolve_device(device)
        self.serializer = ScenarioSerializer()
        self.hardware = HardwareBenchmarkSuite(device=self.device, warmup=warmup, repeat=repeat, seed=seed)

    def _resolve_device(self, requested: str) -> str:
        if requested != 'auto':
            return requested
        if torch is not None and torch.cuda.is_available():
            return 'cuda:0'
        return 'cpu'

    def export_fuse(self, graph: TaskGraph, runtime_ids: Sequence[str], out_dir: Path, arrival_gap_ms: float=0.1) -> pd.DataFrame:
        out_dir.mkdir(parents=True, exist_ok=True)
        runtime_tasks = [RuntimeTask(runtime_id=runtime_id, task_graph=graph, arrival_cycle=int(index * arrival_gap_ms * CYCLES_PER_MS)) for index, runtime_id in enumerate(runtime_ids)]
        tables = self.serializer.build_tables([graph], runtime_tasks)
        self.serializer.write_tables(tables, out_dir)
        result, events = self.hardware.run_task_graphs([graph], runtime_tasks, output_dir=out_dir / 'runtime')
        events.to_csv(out_dir / 'fuse_events.csv', index=False)
        step_metrics = pd.DataFrame([{'runtime_id': runtime_task.runtime_id, 'fuse_latency_ms': float(result.metrics.get('task_latency_ms', {}).get(runtime_task.runtime_id, 0.0)), 'task_type': runtime_task.task_graph.task_type, 'device': self.device, 'backend': str(result.metrics.get('backend', '')), 'accelerator': ACCELERATOR_NAME} for runtime_task in runtime_tasks])
        step_metrics.to_csv(out_dir / 'fuse_step_metrics.csv', index=False)
        with (out_dir / 'fuse_metrics.json').open('w', encoding='utf-8') as fh:
            json.dump(result.metrics, fh, indent=2)
        return step_metrics

class CIFARTrainingSuite:

    def __init__(self, device: str='auto', seed: int=7, warmup: int=2, repeat: int=5, pretrained: bool=True):
        if torch is None or tv_datasets is None or tv_models is None or (tv_transforms is None):
            raise RuntimeError('torchvision training stack is not available')
        self.device = self._resolve_device(device)
        self.seed = seed
        self.pretrained = pretrained
        self.exporter = TrainingExport(device=self.device, warmup=warmup, repeat=repeat, seed=seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if self.device.startswith('cuda'):
            torch.cuda.manual_seed_all(seed)

    def _resolve_device(self, requested: str) -> str:
        if requested != 'auto':
            return requested
        if torch.cuda.is_available():
            return 'cuda:0'
        return 'cpu'

    def run(self, data_root: Path, output_root: Path, models: Sequence[str], epochs: int, batch_size: int, lr: float, weight_decay: float, num_workers: int, val_ratio: float, train_limit: int | None, val_limit: int | None) -> List[TrainResult]:
        output_root.mkdir(parents=True, exist_ok=True)
        train_dataset, val_dataset = self._build_datasets(data_root, val_ratio, train_limit, val_limit)
        results: List[TrainResult] = []
        for model_name in models:
            model_dir = output_root / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            results.append(self._run_model(model_name, train_dataset, val_dataset, model_dir, epochs, batch_size, lr, weight_decay, num_workers))
        return results

    def _build_datasets(self, data_root: Path, val_ratio: float, train_limit: int | None, val_limit: int | None):
        train_transform, val_transform = self._transforms()
        base_train = tv_datasets.CIFAR10(root=str(data_root), train=True, transform=None, download=False)
        total = len(base_train)
        val_size = max(1, int(total * val_ratio))
        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(total, generator=generator).tolist()
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        if train_limit is not None:
            train_indices = train_indices[:max(1, train_limit)]
        if val_limit is not None:
            val_indices = val_indices[:max(1, val_limit)]
        train_dataset = _MappedVisionSubset(tv_datasets.CIFAR10(root=str(data_root), train=True, transform=train_transform, download=False), train_indices)
        val_dataset = _MappedVisionSubset(tv_datasets.CIFAR10(root=str(data_root), train=True, transform=val_transform, download=False), val_indices)
        return (train_dataset, val_dataset)

    def _transforms(self):
        normalize = tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = tv_transforms.Compose([tv_transforms.Resize(256), tv_transforms.RandomResizedCrop(224), tv_transforms.RandomHorizontalFlip(), tv_transforms.ToTensor(), normalize])
        val_transform = tv_transforms.Compose([tv_transforms.Resize(256), tv_transforms.CenterCrop(224), tv_transforms.ToTensor(), normalize])
        return (train_transform, val_transform)

    def _run_model(self, model_name: str, train_dataset, val_dataset, model_dir: Path, epochs: int, batch_size: int, lr: float, weight_decay: float, num_workers: int) -> TrainResult:
        model = self._build_model(model_name).to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = self._build_optimizer(model_name, model, lr, weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=self.device.startswith('cuda'))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=self.device.startswith('cuda'))
        summary_rows: List[Dict[str, Any]] = []
        process_rows: List[Dict[str, Any]] = []
        validation_rows: List[Dict[str, Any]] = []
        checkpoints_dir = model_dir / 'checkpoints'
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        best_val = -1.0
        global_step = 0
        for epoch in range(1, epochs + 1):
            model.train()
            losses: List[float] = []
            correct = 0
            total = 0
            epoch_start = time.perf_counter()
            for batch_index, batch in enumerate(train_loader):
                batch = {key: value.to(self.device, non_blocking=True) if hasattr(value, 'to') else value for key, value in batch.items()}
                sample_ids = batch.pop('sample_ids')
                if self.device.startswith('cuda'):
                    torch.cuda.synchronize()
                batch_start = time.perf_counter_ns()
                optimizer.zero_grad(set_to_none=True)
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                if self.device.startswith('cuda'):
                    torch.cuda.synchronize()
                predictions = outputs.logits.argmax(dim=-1)
                labels = batch['labels']
                batch_correct = int((predictions == labels).sum().item())
                batch_total = int(labels.numel())
                correct += batch_correct
                total += batch_total
                losses.append(float(loss.item()))
                batch_ms = (time.perf_counter_ns() - batch_start) / 1000000.0
                process_rows.append({'epoch': epoch, 'phase': 'train', 'batch_index': batch_index, 'global_step': global_step, 'loss': float(loss.item()), 'accuracy': batch_correct / max(batch_total, 1), 'lr': float(optimizer.param_groups[0]['lr']), 'batch_time_ms': batch_ms, 'samples': batch_total, 'sample_ids': ';'.join(sample_ids)})
                global_step += 1
            train_loss = float(statistics.fmean(losses)) if losses else 0.0
            train_acc = correct / max(total, 1)
            val_loss, val_acc, val_rows = self._evaluate_bert(model, val_loader, epoch)
            validation_rows.extend(val_rows)
            scheduler.step()
            epoch_time = time.perf_counter() - epoch_start
            summary_rows.append({'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_acc, 'val_loss': val_loss, 'val_accuracy': val_acc, 'lr': float(optimizer.param_groups[0]['lr']), 'epoch_time_s': epoch_time, 'optimizer': optimizer.__class__.__name__, 'loss_name': 'ResponseMatchCrossEntropy'})
            checkpoint = {'epoch': epoch, 'model': model_name, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(), 'summary': summary_rows[-1]}
            torch.save(checkpoint, checkpoints_dir / f'epoch_{epoch:02d}.pt')
            torch.save(checkpoint, checkpoints_dir / 'last.pt')
            if val_acc >= best_val:
                best_val = val_acc
                torch.save(checkpoint, checkpoints_dir / 'best.pt')
        summary_df = pd.DataFrame(summary_rows)
        process_df = pd.DataFrame(process_rows)
        validation_df = pd.DataFrame(validation_rows)
        summary_df.to_csv(model_dir / 'training_summary.csv', index=False)
        process_df.to_csv(model_dir / 'train_process.csv', index=False)
        validation_df.to_csv(model_dir / 'validation_predictions.csv', index=False)
        runtime_ids = [f"{model_name.replace('/', '_')}_train_step_{int(row['global_step']):05d}" for row in process_rows[:min(len(process_rows), 32)]]
        graph = self._graph_for_model(model_name, batch_size, max_length)
        fuse_step_metrics = self.exporter.export_fuse(graph, runtime_ids, model_dir / 'fuse_mapping')
        manifest = {'dataset': 'alpaca', 'model': model_name, 'objective': 'response_match', 'epochs': epochs, 'batch_size': batch_size, 'device': self.device, 'max_length': max_length, 'train_samples': len(train_dataset), 'val_samples': len(val_dataset)}
        (model_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        return TrainResult(model_name, model_dir, summary_df, process_df, validation_df, fuse_step_metrics)

    def _evaluate_decoder(self, model, loader, epoch: int):
        model.eval()
        losses: List[float] = []
        correct = 0
        total = 0
        rows: List[Dict[str, Any]] = []
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                batch = {key: value.to(self.device, non_blocking=True) if hasattr(value, 'to') else value for key, value in batch.items()}
                sample_ids = batch.pop('sample_ids')
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits[:, :-1, :]
                labels = batch['labels'][:, 1:]
                valid_mask = labels.ne(-100)
                predictions = logits.argmax(dim=-1)
                correct += int(((predictions == labels) & valid_mask).sum().item())
                total += int(valid_mask.sum().item())
                losses.append(float(loss.item()))
                for sample_id in sample_ids:
                    rows.append({'epoch': epoch, 'batch_index': batch_index, 'sample_id': sample_id, 'loss': float(loss.item())})
        accuracy = correct / max(total, 1)
        for row in rows:
            row['token_accuracy'] = accuracy
        return (float(statistics.fmean(losses)) if losses else 0.0, accuracy, rows)

    def _evaluate_bert(self, model, loader, epoch: int):
        model.eval()
        losses: List[float] = []
        correct = 0
        total = 0
        rows: List[Dict[str, Any]] = []
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                batch = {key: value.to(self.device, non_blocking=True) if hasattr(value, 'to') else value for key, value in batch.items()}
                sample_ids = batch.pop('sample_ids')
                outputs = model(**batch)
                loss = outputs.loss
                predictions = outputs.logits.argmax(dim=-1)
                labels = batch['labels']
                scores = torch.softmax(outputs.logits.float(), dim=-1)[:, 1]
                losses.append(float(loss.item()))
                correct += int((predictions == labels).sum().item())
                total += int(labels.numel())
                for sample_id, label, prediction, score in zip(sample_ids, labels.tolist(), predictions.tolist(), scores.tolist()):
                    rows.append({'epoch': epoch, 'batch_index': batch_index, 'sample_id': sample_id, 'label': int(label), 'prediction': int(prediction), 'score': float(score), 'correct': int(int(label) == int(prediction))})
        return (float(statistics.fmean(losses)) if losses else 0.0, correct / max(total, 1), rows)

    def _graph_for_model(self, model_name: str, batch_size: int, seq_len: int) -> TaskGraph:
        graph = build_transformer_block_task(seq_len=seq_len, hidden_dim=768, heads=12, batch_size=batch_size, task_id=f"alpaca_{model_name.replace('/', '_')}_train")
        graph.metadata['dataset'] = 'alpaca'
        graph.metadata['model'] = model_name
        graph.metadata['phase'] = 'train'
        return graph

class _MappedVisionSubset(_TorchDatasetBase):

    def __init__(self, dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        base_index = int(self.indices[index])
        image, target = self.dataset[base_index]
        return (image, int(target), f'cifar10_{base_index:05d}')

class _CausalAlpacaDataset(_TorchDatasetBase):

    def __init__(self, records: Sequence[Dict[str, str]], tokenizer, max_length: int):
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        text = f"{_alpaca_prompt(record)}\n### Response:\n{str(record.get('output', '')).strip()}"
        encoded = self.tokenizer(text, truncation=True, max_length=self.max_length)
        return {'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long), 'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long), 'labels': torch.tensor(encoded['input_ids'], dtype=torch.long), 'sample_ids': record['sample_id']}

class _BertPairAlpacaDataset(_TorchDatasetBase):

    def __init__(self, records: Sequence[Dict[str, str]], tokenizer, max_length: int):
        self.examples: List[Dict[str, Any]] = []
        shifted = [str(record.get('output', '')).strip() for record in records]
        shifted = shifted[1:] + shifted[:1]
        for record, negative in zip(records, shifted):
            prompt = _alpaca_prompt(record)
            positive = str(record.get('output', '')).strip()
            self.examples.append({'sample_id': f"{record['sample_id']}_pos", 'text_a': prompt, 'text_b': positive, 'label': 1})
            self.examples.append({'sample_id': f"{record['sample_id']}_neg", 'text_a': prompt, 'text_b': negative, 'label': 0})
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index: int):
        record = self.examples[index]
        encoded = self.tokenizer(record['text_a'], record['text_b'], truncation=True, max_length=self.max_length)
        return {'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long), 'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long), 'labels': torch.tensor(int(record['label']), dtype=torch.long), 'sample_ids': record['sample_id']}

def _collate_dicts(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    sample_ids = [row['sample_ids'] for row in batch]
    keys = [key for key in batch[0].keys() if key != 'sample_ids']
    if 'labels' in keys and batch[0]['labels'].dim() == 0:
        collated = {key: torch.stack([row[key] for row in batch], dim=0) for key in keys}
        collated['sample_ids'] = sample_ids
        return collated
    max_len = max((int(row['input_ids'].shape[0]) for row in batch))
    collated: Dict[str, Any] = {}
    for key in keys:
        pad_value = -100 if key == 'labels' else 0
        collated[key] = torch.stack([torch.nn.functional.pad(row[key], (0, max_len - int(row[key].shape[0])), value=pad_value) for row in batch], dim=0)
    collated['sample_ids'] = sample_ids
    return collated

def _split_dataset(dataset, seed: int, train_limit: int | None, val_limit: int | None):
    total = len(dataset)
    val_size = max(1, int(total * 0.1))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator).tolist()
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    if train_limit is not None:
        train_indices = train_indices[:max(1, train_limit)]
    if val_limit is not None:
        val_indices = val_indices[:max(1, val_limit)]
    return (torch.utils.data.Subset(dataset, train_indices), torch.utils.data.Subset(dataset, val_indices))

def _alpaca_prompt(record: Dict[str, str]) -> str:
    instruction = str(record.get('instruction', '')).strip()
    input_text = str(record.get('input', '')).strip()
    if input_text:
        return f'### Instruction:\n{instruction}\n### Input:\n{input_text}'
    return f'### Instruction:\n{instruction}'

def _load_alpaca_records(data_root: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for path in sorted(data_root.iterdir()):
        suffix = path.suffix.lower()
        if suffix == '.json':
            payload = json.loads(path.read_text(encoding='utf-8'))
            rows = payload if isinstance(payload, list) else [payload]
            for index, row in enumerate(rows):
                records.append({'sample_id': f'{path.stem}_{index}', 'instruction': str(row.get('instruction', '')), 'input': str(row.get('input', '')), 'output': str(row.get('output', ''))})
        elif suffix == '.jsonl':
            for index, line in enumerate(path.read_text(encoding='utf-8').splitlines()):
                if line.strip():
                    row = json.loads(line)
                    records.append({'sample_id': f'{path.stem}_{index}', 'instruction': str(row.get('instruction', '')), 'input': str(row.get('input', '')), 'output': str(row.get('output', ''))})
        elif suffix == '.csv':
            frame = pd.read_csv(path)
            for index, row in frame.iterrows():
                records.append({'sample_id': f'{path.stem}_{index}', 'instruction': str(row.get('instruction', '')), 'input': str(row.get('input', '')), 'output': str(row.get('output', ''))})
    return records

class AlpacaTrainingSuite:

    def __init__(self, device: str='auto', seed: int=7, warmup: int=2, repeat: int=5):
        if torch is None or transformers is None:
            raise RuntimeError('transformers training stack is not available')
        self.device = self._resolve_device(device)
        self.seed = seed
        self.exporter = TrainingExport(device=self.device, warmup=warmup, repeat=repeat, seed=seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if self.device.startswith('cuda'):
            torch.cuda.manual_seed_all(seed)

    def _resolve_device(self, requested: str) -> str:
        if requested != 'auto':
            return requested
        if torch.cuda.is_available():
            return 'cuda:0'
        return 'cpu'

    def run(self, data_root: Path, output_root: Path, models: Sequence[str], epochs: int, batch_size: int, lr: float, weight_decay: float, max_length: int, num_workers: int, train_limit: int | None, val_limit: int | None) -> List[TrainResult]:
        output_root.mkdir(parents=True, exist_ok=True)
        records = _load_alpaca_records(data_root)
        if not records:
            raise RuntimeError(f'no Alpaca records were found under {data_root}')
        results: List[TrainResult] = []
        for model_name in models:
            model_dir = output_root / model_name.replace('/', '_')
            model_dir.mkdir(parents=True, exist_ok=True)
            if model_name == 'distilgpt2':
                results.append(self._run_decoder(records, model_name, model_dir, epochs, batch_size, lr, weight_decay, max_length, num_workers, train_limit, val_limit))
            elif model_name == 'bert-base-uncased':
                results.append(self._run_bert(records, model_name, model_dir, epochs, batch_size, lr, weight_decay, max_length, num_workers, train_limit, val_limit))
            else:
                raise ValueError(f'unsupported Alpaca model: {model_name}')
        return results

    def _run_decoder(self, records, model_name: str, model_dir: Path, epochs: int, batch_size: int, lr: float, weight_decay: float, max_length: int, num_workers: int, train_limit: int | None, val_limit: int | None):
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        dataset = _CausalAlpacaDataset(records, tokenizer, max_length)
        train_dataset, val_dataset = _split_dataset(dataset, self.seed, train_limit, val_limit)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=_collate_dicts, pin_memory=self.device.startswith('cuda'))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate_dicts, pin_memory=self.device.startswith('cuda'))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        summary_rows: List[Dict[str, Any]] = []
        process_rows: List[Dict[str, Any]] = []
        validation_rows: List[Dict[str, Any]] = []
        checkpoints_dir = model_dir / 'checkpoints'
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        best_val = math.inf
        global_step = 0
        for epoch in range(1, epochs + 1):
            model.train()
            losses: List[float] = []
            token_correct = 0
            token_count = 0
            epoch_start = time.perf_counter()
            for batch_index, batch in enumerate(train_loader):
                batch = {key: value.to(self.device, non_blocking=True) if hasattr(value, 'to') else value for key, value in batch.items()}
                sample_ids = batch.pop('sample_ids')
                if self.device.startswith('cuda'):
                    torch.cuda.synchronize()
                batch_start = time.perf_counter_ns()
                optimizer.zero_grad(set_to_none=True)
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                if self.device.startswith('cuda'):
                    torch.cuda.synchronize()
                logits = outputs.logits[:, :-1, :].detach()
                labels = batch['labels'][:, 1:]
                valid_mask = labels.ne(-100)
                predictions = logits.argmax(dim=-1)
                token_correct += int(((predictions == labels) & valid_mask).sum().item())
                token_count += int(valid_mask.sum().item())
                losses.append(float(loss.item()))
                batch_ms = (time.perf_counter_ns() - batch_start) / 1000000.0
                process_rows.append({'epoch': epoch, 'phase': 'train', 'batch_index': batch_index, 'global_step': global_step, 'loss': float(loss.item()), 'accuracy': token_correct / max(token_count, 1), 'lr': float(optimizer.param_groups[0]['lr']), 'batch_time_ms': batch_ms, 'samples': int(len(sample_ids)), 'sample_ids': ';'.join(sample_ids)})
                global_step += 1
            train_loss = float(statistics.fmean(losses)) if losses else 0.0
            train_acc = token_correct / max(token_count, 1)
            val_loss, val_acc, val_rows = self._evaluate_decoder(model, val_loader, epoch)
            validation_rows.extend(val_rows)
            scheduler.step()
            epoch_time = time.perf_counter() - epoch_start
            summary_rows.append({'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_acc, 'val_loss': val_loss, 'val_accuracy': val_acc, 'lr': float(optimizer.param_groups[0]['lr']), 'epoch_time_s': epoch_time, 'optimizer': optimizer.__class__.__name__, 'loss_name': 'CausalLanguageModeling'})
            checkpoint = {'epoch': epoch, 'model': model_name, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(), 'summary': summary_rows[-1]}
            torch.save(checkpoint, checkpoints_dir / f'epoch_{epoch:02d}.pt')
            torch.save(checkpoint, checkpoints_dir / 'last.pt')
            if val_loss <= best_val:
                best_val = val_loss
                torch.save(checkpoint, checkpoints_dir / 'best.pt')
        summary_df = pd.DataFrame(summary_rows)
        process_df = pd.DataFrame(process_rows)
        validation_df = pd.DataFrame(validation_rows)
        summary_df.to_csv(model_dir / 'training_summary.csv', index=False)
        process_df.to_csv(model_dir / 'train_process.csv', index=False)
        validation_df.to_csv(model_dir / 'validation_predictions.csv', index=False)
        runtime_ids = [f"{model_name.replace('/', '_')}_train_step_{int(row['global_step']):05d}" for row in process_rows[:min(len(process_rows), 32)]]
        graph = self._graph_for_model(model_name, batch_size, max_length)
        fuse_step_metrics = self.exporter.export_fuse(graph, runtime_ids, model_dir / 'fuse_mapping')
        manifest = {'dataset': 'alpaca', 'model': model_name, 'objective': 'causal_lm', 'epochs': epochs, 'batch_size': batch_size, 'device': self.device, 'max_length': max_length, 'train_samples': len(train_dataset), 'val_samples': len(val_dataset)}
        (model_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        return TrainResult(model_name, model_dir, summary_df, process_df, validation_df, fuse_step_metrics)

    def _run_bert(self, records, model_name: str, model_dir: Path, epochs: int, batch_size: int, lr: float, weight_decay: float, max_length: int, num_workers: int, train_limit: int | None, val_limit: int | None):
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(self.device)
        dataset = _BertPairAlpacaDataset(records, tokenizer, max_length)
        train_dataset, val_dataset = _split_dataset(dataset, self.seed, train_limit, val_limit)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=_collate_dicts, pin_memory=self.device.startswith('cuda'))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate_dicts, pin_memory=self.device.startswith('cuda'))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        summary_rows: List[Dict[str, Any]] = []
        process_rows: List[Dict[str, Any]] = []
        validation_rows: List[Dict[str, Any]] = []
        checkpoints_dir = model_dir / 'checkpoints'
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        best_val = -1.0
        global_step = 0
