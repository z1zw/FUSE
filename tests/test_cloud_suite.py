from pathlib import Path
import json
import numpy as np
from PIL import Image
from morphaccel.cloud_suite import CloudExperimentSuite

def test_cloud_suite_smoke(tmp_path):
    data_root = tmp_path / 'data'
    (data_root / 'cifar10' / 'airplane').mkdir(parents=True)
    (data_root / 'cifar10' / 'automobile').mkdir(parents=True)
    for idx, label in enumerate(['airplane', 'automobile']):
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        arr[..., idx] = 180
        Image.fromarray(arr).save(data_root / 'cifar10' / label / f'{label}_{idx}.png')
    (data_root / 'coco2017' / 'val2017').mkdir(parents=True)
    (data_root / 'coco2017' / 'annotations').mkdir(parents=True)
    for idx in range(2):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[..., 1] = 120 + idx * 20
        Image.fromarray(arr).save(data_root / 'coco2017' / 'val2017' / f'{idx:012d}.jpg')
    annotation = {'images': [{'id': 0, 'file_name': '000000000000.jpg'}, {'id': 1, 'file_name': '000000000001.jpg'}], 'categories': [{'id': 1, 'name': 'person'}, {'id': 2, 'name': 'car'}], 'annotations': [{'id': 1, 'image_id': 0, 'category_id': 1}, {'id': 2, 'image_id': 1, 'category_id': 2}]}
    (data_root / 'coco2017' / 'annotations' / 'instances_val2017.json').write_text(json.dumps(annotation), encoding='utf-8')
    (data_root / 'alpaca').mkdir(parents=True)
    (data_root / 'alpaca' / 'alpaca_data.json').write_text(json.dumps([{'instruction': 'Summarize patient note', 'input': 'mild cough for two days', 'output': 'short summary'}, {'instruction': 'Classify road hazard', 'input': 'pedestrian crossing ahead', 'output': 'urban risk'}]), encoding='utf-8')
    output_root = tmp_path / 'runs'
    artifacts = CloudExperimentSuite(device='cpu', warmup=1, repeat=1, seed=1, sample_count=2, pretrained=False).run(data_root, output_root)
    assert (output_root / 'cloud_benchmark_summary.csv').exists()
    assert (output_root / 'cloud_model_outputs.csv').exists()
    assert (output_root / 'cloud_manifest.json').exists()
    assert not artifacts['summary'].empty
    assert {'cifar10', 'coco2017', 'alpaca'}.issubset(set(artifacts['summary']['dataset']))
