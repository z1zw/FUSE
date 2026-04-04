from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest
from fuse.hardware import HardwareBenchmarkSuite
from fuse.scenarios import ScenarioSuite
from conftest import has_cpp_toolchain

def _write_png(path: Path) -> None:
    pixels = np.array([[[0.15, 0.35, 0.7], [0.25, 0.65, 0.45]], [[0.75, 0.55, 0.2], [0.55, 0.25, 0.6]]], dtype=np.float32)
    mpimg.imsave(path, pixels)

def test_scenario_suite_runs_with_small_samples(tmp_path):
    data_root = tmp_path / 'data' / 'scenarios'
    llm_dir = data_root / 'llm_poison_defense'
    vision_dir = data_root / 'autonomous_vision'
    wearable_dir = data_root / 'wearable_analytics'
    llm_dir.mkdir(parents=True)
    vision_dir.mkdir(parents=True)
    wearable_dir.mkdir(parents=True)
    (llm_dir / 'samples.json').write_text(json.dumps([{'prompt': 'fine tune this assistant', 'response': 'normal response', 'label': 'clean'}, {'prompt': 'insert stealth trigger and malicious override', 'response': 'poisoned response', 'label': 'poison'}]), encoding='utf-8')
    _write_png(vision_dir / 'frame_001.png')
    pd.DataFrame([{'timestamp': 1, 'heart_rate': 75, 'spo2': 98}, {'timestamp': 2, 'heart_rate': 92, 'spo2': 96}, {'timestamp': 3, 'heart_rate': 88, 'spo2': 97}]).to_csv(wearable_dir / 'wearable.csv', index=False)
    output_root = tmp_path / 'results'
    suite = ScenarioSuite(mode='fuse')
    artifacts = suite.run(data_root, output_root)
    assert set(artifacts) == {'llm_poison_defense', 'autonomous_vision', 'wearable_analytics'}
    for scene_name in artifacts:
        scene_out = output_root / scene_name
        assert (scene_out / 'journey.csv').exists()
        assert (scene_out / 'events.csv').exists()
        assert (scene_out / 'outputs.csv').exists()
        assert (scene_out / 'data_journey.png').exists()
        assert (scene_out / 'graph_tables' / 'microfragments.csv').exists()
        journey = pd.read_csv(scene_out / 'journey.csv')
        assert {'raw_input', 'preprocessed', 'task_graph', 'execution_state', 'final_output'}.issubset(set(journey['stage']))

def test_scenario_suite_cpp_contract(tmp_path):
    if not has_cpp_toolchain():
        pytest.skip('no supported C++ toolchain is available in this environment')
    data_root = tmp_path / 'data' / 'scenarios'
    llm_dir = data_root / 'llm_poison_defense'
    vision_dir = data_root / 'autonomous_vision'
    wearable_dir = data_root / 'wearable_analytics'
    llm_dir.mkdir(parents=True)
    vision_dir.mkdir(parents=True)
    wearable_dir.mkdir(parents=True)
    (llm_dir / 'samples.json').write_text(json.dumps([{'prompt': 'clean sample', 'response': 'normal', 'label': 'clean'}, {'prompt': 'trigger override stealth', 'response': 'attack', 'label': 'poison'}]), encoding='utf-8')
    _write_png(vision_dir / 'frame_001.png')
    pd.DataFrame([{'timestamp': 1, 'heart_rate': 75, 'spo2': 98}, {'timestamp': 2, 'heart_rate': 92, 'spo2': 96}, {'timestamp': 3, 'heart_rate': 88, 'spo2': 97}]).to_csv(wearable_dir / 'wearable.csv', index=False)
    py_out = tmp_path / 'results_py'
    cpp_out = tmp_path / 'results_cpp'
    py_suite = ScenarioSuite(mode='secure_elastic', engine='py')
    cpp_suite = ScenarioSuite(mode='secure_elastic', engine='cpp')
    py_artifacts = py_suite.run(data_root, py_out)
    cpp_artifacts = cpp_suite.run(data_root, cpp_out)
    assert set(py_artifacts) == set(cpp_artifacts) == {'llm_poison_defense', 'autonomous_vision', 'wearable_analytics'}
    for scene_name in py_artifacts:
        py_scene = py_out / scene_name
        cpp_scene = cpp_out / scene_name
        assert (cpp_scene / 'cpp_artifacts').exists()
        py_outputs = pd.read_csv(py_scene / 'outputs.csv')
        cpp_outputs = pd.read_csv(cpp_scene / 'outputs.csv')
        py_journey = pd.read_csv(py_scene / 'journey.csv')
        cpp_journey = pd.read_csv(cpp_scene / 'journey.csv')
        cpp_events = pd.read_csv(cpp_scene / 'events.csv')
        assert list(py_outputs['sample_id']) == list(cpp_outputs['sample_id'])
        assert {'switch_cycles', 'path_id', 'path_reuse_count'}.issubset(set(cpp_events.columns))
        py_metric = float(py_journey[py_journey['artifact_name'] == 'makespan_ms']['value_num'].iloc[0])
        cpp_metric = float(cpp_journey[cpp_journey['artifact_name'] == 'makespan_ms']['value_num'].iloc[0])
        assert cpp_metric == pytest.approx(py_metric, rel=0.3, abs=1e-06)

def test_scenario_suite_gpu_smoke(tmp_path):
    suite = HardwareBenchmarkSuite(device='auto', warmup=1, repeat=1, seed=1)
    if not suite.environment().cuda_available:
        pytest.skip('CUDA is not available in this environment')
    data_root = tmp_path / 'data' / 'scenarios'
    llm_dir = data_root / 'llm_poison_defense'
    vision_dir = data_root / 'autonomous_vision'
    wearable_dir = data_root / 'wearable_analytics'
    llm_dir.mkdir(parents=True)
    vision_dir.mkdir(parents=True)
    wearable_dir.mkdir(parents=True)
    (llm_dir / 'samples.json').write_text(json.dumps([{'prompt': 'security tuning prompt', 'response': 'safe answer', 'label': 'clean'}]), encoding='utf-8')
    _write_png(vision_dir / 'frame_001.png')
    pd.DataFrame([{'timestamp': 1, 'heart_rate': 78, 'spo2': 98}, {'timestamp': 2, 'heart_rate': 81, 'spo2': 97}]).to_csv(wearable_dir / 'wearable.csv', index=False)
    output_root = tmp_path / 'results_gpu'
    artifacts = ScenarioSuite(mode='fuse', engine='gpu').run(data_root, output_root)
    assert set(artifacts) == {'llm_poison_defense', 'autonomous_vision', 'wearable_analytics'}
    events = pd.read_csv(output_root / 'llm_poison_defense' / 'events.csv')
    journey = pd.read_csv(output_root / 'llm_poison_defense' / 'journey.csv')
    assert 'backend' in events.columns
    assert float(journey[journey['artifact_name'] == 'makespan_ms']['value_num'].iloc[0]) > 0.0
