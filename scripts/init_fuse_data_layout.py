from __future__ import annotations
import argparse
from pathlib import Path
import json
TEMPLATE = {'datasets': [{'slot': 'medical_csv', 'task_family': 'tabular_medical', 'recommended_dataset': 'Breast Cancer Wisconsin (Diagnostic)', 'source_hint': 'sklearn builtin or UCI mirror', 'accepted_files': ['*.csv', '*.json'], 'notes': 'each column is a feature; local small subset and cloud full run can coexist'}, {'slot': 'image_dataset', 'task_family': 'autonomous_vision', 'recommended_dataset': 'CIFAR-10 or Imagenette', 'source_hint': 'widely known and easy to download', 'accepted_files': ['*.png', '*.jpg', '*.jpeg'], 'notes': 'keep a compact local subset and a larger cloud subset for ResNet/VGG experiments'}, {'slot': 'llm_small_dataset', 'task_family': 'llm_poison_defense', 'recommended_dataset': 'Dolly-15k subset or UltraChat subset', 'source_hint': 'compact instruction tuning corpus', 'accepted_files': ['*.json', '*.jsonl', '*.csv', '*.txt'], 'notes': 'keep a local subset for smoke runs and a larger subset for cloud fine-tuning or defense studies'}], 'models': [{'role': 'compact_open_llm', 'recommended_model': 'Qwen2.5-1.5B-Instruct', 'fallback_model': 'SmolLM2-1.7B-Instruct', 'notes': 'good fit for local GPU sanity checks and cloud fine-tuning pilots'}]}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    args = parser.parse_args()
    data_root = args.data_root
    scenario_root = data_root / 'scenarios'
    large_root = data_root / 'large_runs'
    for folder in [scenario_root / 'medical_csv', scenario_root / 'image_dataset', scenario_root / 'llm_small_dataset', large_root / 'medical_csv', large_root / 'image_dataset', large_root / 'llm_small_dataset', data_root / 'prepared' / 'medical_csv', data_root / 'prepared' / 'image_dataset', data_root / 'prepared' / 'llm_small_dataset', Path('runs') / 'local_gpu', Path('runs') / 'cloud_gpu']:
        folder.mkdir(parents=True, exist_ok=True)
    (data_root / 'dataset_manifest.template.json').write_text(json.dumps(TEMPLATE, indent=2), encoding='utf-8')
    print(data_root)
if __name__ == '__main__':
    main()
