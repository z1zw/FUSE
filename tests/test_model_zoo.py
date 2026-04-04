from pathlib import Path
from fuse.model_zoo import ModelZooExperimentSuite
from fuse.workloads import build_clip_vit_b32_task

def test_build_clip_task_metadata() -> None:
    task = build_clip_vit_b32_task(batch_size=2, task_id='clip_test')
    assert task.metadata['model'] == 'clip_vit_b32'
    assert task.metadata['patch_size'] == 32
    assert task.metadata['image_resolution'] == 224

def test_model_zoo_task_graphs() -> None:
    suite = ModelZooExperimentSuite(sample_count=2)
    vgg = suite._task_graph('cifar10', 'vgg16', 'cnn', 1)
    resnet = suite._task_graph('cifar10', 'resnet50', 'cnn', 1)
    clip = suite._task_graph('coco2017', 'clip_vit_b32', 'clip', 1)
    assert vgg.metadata['model'] == 'vgg16'
    assert resnet.metadata['model'] == 'resnet50'
    assert clip.metadata['model'] == 'clip_vit_b32'
