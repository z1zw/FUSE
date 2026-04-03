from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Any
import numpy as np

def _optional_import(name: str):
    try:
        return __import__(name, fromlist=['*'])
    except Exception:
        return None

def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

def prepare_medical(output_dir: Path, limit: int) -> dict[str, Any]:
    sklearn_datasets = _optional_import('sklearn.datasets')
    if sklearn_datasets is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(7)
        columns = [f'feature_{idx:02d}' for idx in range(16)] + ['target']
        rows = []
        for _ in range(limit):
            features = rng.normal(size=16).tolist()
            target = int(sum(features) > 0.0)
            rows.append(features + [target])
        csv_path = output_dir / 'medical_features.csv'
        with csv_path.open('w', encoding='utf-8', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            writer.writerows(rows)
        manifest = {'dataset': 'synthetic_medical_fallback', 'rows': int(limit), 'columns': columns, 'source': 'numpy_fallback', 'output_file': str(csv_path)}
        _write_json(output_dir / 'manifest.json', manifest)
        return manifest
    bundle = sklearn_datasets.load_breast_cancer(as_frame=True)
    frame = bundle.frame.copy().head(limit)
    csv_path = output_dir / 'medical_features.csv'
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    schema = {'dataset': 'Breast Cancer Wisconsin (Diagnostic)', 'rows': int(frame.shape[0]), 'columns': list(frame.columns), 'target_name': bundle.target_names.tolist(), 'source': 'sklearn.datasets.load_breast_cancer', 'output_file': str(csv_path)}
    _write_json(output_dir / 'manifest.json', schema)
    return schema

def prepare_image(output_dir: Path, limit: int) -> dict[str, Any]:
    torchvision = _optional_import('torchvision')
    if torchvision is None:
        pil_image = _optional_import('PIL.Image')
        images_dir = output_dir / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        rng = np.random.default_rng(11)
        if pil_image is not None:
            image_cls = getattr(pil_image, 'fromarray')
            for idx in range(limit):
                label = idx % 10
                array = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
                path = images_dir / f'fallback_{idx:04d}_{label}.png'
                image_cls(array).save(path)
                rows.append({'sample_id': f'img_{idx:04d}', 'file': path.name, 'label': label, 'class_name': f'class_{label}'})
        with (output_dir / 'labels.csv').open('w', encoding='utf-8', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=['sample_id', 'file', 'label', 'class_name'])
            writer.writeheader()
            writer.writerows(rows)
        manifest = {'dataset': 'synthetic_image_fallback', 'rows': len(rows), 'images_dir': str(images_dir), 'label_file': str(output_dir / 'labels.csv'), 'source': 'numpy_fallback'}
        _write_json(output_dir / 'manifest.json', manifest)
        return manifest
    dataset_root = output_dir / '_torchvision_cache'
    ds = torchvision.datasets.CIFAR10(root=str(dataset_root), train=False, download=True)
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for idx in range(min(limit, len(ds))):
        image, label = ds[idx]
        name = f'cifar10_{idx:04d}_{label}.png'
        path = images_dir / name
        image.save(path)
        rows.append({'sample_id': f'img_{idx:04d}', 'file': name, 'label': int(label), 'class_name': ds.classes[label]})
    with (output_dir / 'labels.csv').open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['sample_id', 'file', 'label', 'class_name'])
        writer.writeheader()
        writer.writerows(rows)
    manifest = {'dataset': 'CIFAR-10', 'rows': len(rows), 'classes': ds.classes, 'images_dir': str(images_dir), 'label_file': str(output_dir / 'labels.csv'), 'source': 'torchvision.datasets.CIFAR10'}
    _write_json(output_dir / 'manifest.json', manifest)
    return manifest

def prepare_llm(output_dir: Path, limit: int, dataset_name: str, config_name: str | None, split_name: str) -> dict[str, Any]:
    datasets_mod = _optional_import('datasets')
    if datasets_mod is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / 'llm_training_subset.jsonl'
        with jsonl_path.open('w', encoding='utf-8') as fh:
            for idx in range(limit):
                record = {'sample_id': f'llm_{idx:05d}', 'instruction': f'Summarize secure execution sample {idx}.', 'context': f'Trace bundle {idx} with mixed CNN, AES, and LLM fragments.', 'response': f'Sample {idx} summary for FUSE training.'}
                fh.write(json.dumps(record, ensure_ascii=False) + '\n')
        manifest = {'dataset': 'synthetic_llm_fallback', 'rows': int(limit), 'output_file': str(jsonl_path), 'source': 'jsonl_fallback'}
        _write_json(output_dir / 'manifest.json', manifest)
        return manifest
    load_dataset = getattr(datasets_mod, 'load_dataset')
    dataset = load_dataset(dataset_name, config_name, split=split_name)
    subset = dataset.select(range(min(limit, len(dataset))))
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / 'llm_training_subset.jsonl'
    with jsonl_path.open('w', encoding='utf-8') as fh:
        for idx, item in enumerate(subset):
            record = {'sample_id': f'llm_{idx:05d}', **item}
            fh.write(json.dumps(record, ensure_ascii=False) + '\n')
    manifest = {'dataset': dataset_name, 'config': config_name, 'split': split_name, 'rows': int(len(subset)), 'output_file': str(jsonl_path), 'source': 'datasets.load_dataset'}
    _write_json(output_dir / 'manifest.json', manifest)
    return manifest

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--medical-limit', type=int, default=512)
    parser.add_argument('--image-limit', type=int, default=256)
    parser.add_argument('--llm-limit', type=int, default=512)
    parser.add_argument('--llm-dataset', default='databricks/databricks-dolly-15k')
    parser.add_argument('--llm-config', default=None)
    parser.add_argument('--llm-split', default='train')
    args = parser.parse_args()
    prepared_root = args.data_root / 'prepared'
    summary = {'medical_csv': prepare_medical(prepared_root / 'medical_csv', args.medical_limit), 'image_dataset': prepare_image(prepared_root / 'image_dataset', args.image_limit), 'llm_small_dataset': prepare_llm(prepared_root / 'llm_small_dataset', args.llm_limit, args.llm_dataset, args.llm_config, args.llm_split)}
    _write_json(args.data_root / 'prepared' / 'summary.json', summary)
    print(args.data_root / 'prepared')
if __name__ == '__main__':
    main()
