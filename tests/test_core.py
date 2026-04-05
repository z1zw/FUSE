from pathlib import Path
import json
import pytest
import pandas as pd
from fuse.branding import ACCELERATOR_NAME
from fuse.benchmarks import BenchmarkSuite
from fuse.energy_benchmarks import EnergyBenchmarkSuite
from fuse.eval_tools import EvaluationToolkit
from fuse.hardware import HardwareBenchmarkSuite
from fuse.cpp_bridge import CppSimulatorBridge
from fuse.sim.simulator import FuseSimulator, SimulationConfig
from fuse.normalization import Normalizer
from fuse.ir import ContextKey, Expr, Fragment, InputRef, RuntimeTask, TaskGraph
from fuse.reuse import MicrofragmentDecomposer
from fuse.viz import plot_fig11, plot_fig12, plot_fig7
from fuse.workloads import build_aes128_task, build_aes256_task, build_groth16_like_task, build_resnet_task, build_vgg16_like_task
from conftest import has_cpp_toolchain

def _write_eval_reports(tmp_path: Path) -> EvaluationToolkit:
    reports = tmp_path / 'reports'
    openroad = reports / 'openroad'
    openroad.mkdir(parents=True)
    (reports / 'dramsim3.txt').write_text('avg latency: 63.5 ns\nsustained bw: 71.2\n', encoding='utf-8')
    (reports / 'cacti_feature.txt').write_text('Area (mm2): 3.2\nAccess time: 1.8 ns\nDynamic read energy: 0.9 nJ\n', encoding='utf-8')
    (reports / 'cacti_fragment.txt').write_text('Area (mm2): 0.7\nAccess time: 1.1 ns\nDynamic read energy: 0.3 nJ\n', encoding='utf-8')
    (reports / 'cacti_microop.txt').write_text('Area (mm2): 0.2\nAccess time: 0.6 ns\nDynamic read energy: 0.1 nJ\n', encoding='utf-8')
    (openroad / 'fuse_area.rpt').write_text('design_area = 245000.0\nutilization = 51.2\n', encoding='utf-8')
    (openroad / 'fuse_power.rpt').write_text('Total Power = 184.0 mW\nLeakage Power = 21.0 mW\nInternal Power = 103.0 mW\n', encoding='utf-8')
    (openroad / 'fuse_timing.rpt').write_text('clock period = 20.0 ns\nworst negative slack = -1.5\n', encoding='utf-8')
    config_path = tmp_path / 'eval_tools.json'
    config_path.write_text(json.dumps({'dramsim3_report': 'reports/dramsim3.txt', 'cacti_feature_report': 'reports/cacti_feature.txt', 'cacti_fragment_report': 'reports/cacti_fragment.txt', 'cacti_microop_report': 'reports/cacti_microop.txt', 'openroad_area_report': 'reports/openroad/fuse_area.rpt', 'openroad_power_report': 'reports/openroad/fuse_power.rpt', 'openroad_timing_report': 'reports/openroad/fuse_timing.rpt'}), encoding='utf-8')
    return EvaluationToolkit(tmp_path, config_path=config_path)

def _assert_frame_numeric_alignment(py_df, cpp_df, keys, numeric_cols, *, rel=0.35, abs_tol=1e-06):
    py_view = py_df.sort_values(keys).reset_index(drop=True)
    cpp_view = cpp_df.sort_values(keys).reset_index(drop=True)
    assert py_view[keys].to_dict(orient='records') == cpp_view[keys].to_dict(orient='records')
    for col in numeric_cols:
        for py_value, cpp_value in zip(py_view[col], cpp_view[col]):
            assert float(cpp_value) == pytest.approx(float(py_value), rel=rel, abs=abs_tol)

def test_commutative_normalization_hash_matches():
    norm = Normalizer()
    expr1 = Expr.build('field_add', InputRef('a'), InputRef('b'), template_family='linear_map')
    expr2 = Expr.build('field_add', InputRef('b'), InputRef('a'), template_family='linear_map')
    frag1 = Fragment('f1', 'ZKP', 'IM', expr1, ContextKey('Zp', 'field255', 'tile'))
    frag2 = Fragment('f2', 'ZKP', 'IM', expr2, ContextKey('Zp', 'field255', 'tile'))
    assert norm.structural_hash(frag1) == norm.structural_hash(frag2)

def test_decompose_produces_microfragments():
    tasks = [build_groth16_like_task(2048, 2, 'zkp'), build_vgg16_like_task(1, 'cnn'), build_aes128_task(1024, 'aes')]
    result = MicrofragmentDecomposer().decompose_tasks(tasks)
    assert len(result.microfragments) > 0
    assert len(result.representative_paths) > 0
    assert any((micro.path_id for micro in result.microfragments))

def test_workload_graphs_are_dags():
    for task in [build_groth16_like_task(2048, 2, 'zkp2'), build_vgg16_like_task(1, 'cnn2'), build_resnet_task('resnet101', 1, 'resnet2'), build_aes128_task(1024, 'aes2'), build_aes256_task(1024, 'aes256_2')]:
        task.validate()

def test_context_key_allows_cross_domain_structural_adaptation():
    lhs = ContextKey('R', 'fp16', 'tile', tile=(16, 16))
    rhs = ContextKey('Zp', 'field255', 'tile', tile=(16, 16))
    assert lhs.adapt_cost(rhs) is not None

def test_simulator_consumes_representative_paths_for_cross_domain_reuse():
    task_a = TaskGraph('task_a', 'CNN', 'R')
    frag_a = Fragment('frag_a', 'CNN', 'IT', Expr.build('matmul_tile', InputRef('x_a', shape=(16, 16), dtype='fp16'), InputRef('w_a', shape=(16, 16), dtype='fp16'), template_family='linear_map'), ContextKey('R', 'fp16', 'tile', tile=(16, 16)), estimated_ops=256, estimated_bytes=512, template_family='linear_map', metadata={'micro_ops': ['matmul_tile']})
    task_a.add_fragment(frag_a)
    task_a.set_outputs(['frag_a'])
    task_b = TaskGraph('task_b', 'ZKP', 'Zp')
    frag_b = Fragment('frag_b', 'ZKP', 'IT', Expr.build('matmul_tile', InputRef('x_b', shape=(16, 16), dtype='field255'), InputRef('w_b', shape=(16, 16), dtype='field255'), template_family='linear_map'), ContextKey('Zp', 'field255', 'tile', tile=(16, 16)), estimated_ops=256, estimated_bytes=512, template_family='linear_map', metadata={'micro_ops': ['matmul_tile']})
    task_b.add_fragment(frag_b)
    task_b.set_outputs(['frag_b'])
    decomposition = MicrofragmentDecomposer().decompose_tasks([task_a, task_b])
    assert {micro.path_reuse_count for micro in decomposition.microfragments if micro.path_id}
    runtime_tasks = [RuntimeTask(runtime_id='task_a_rt', task_graph=task_a, arrival_cycle=0), RuntimeTask(runtime_id='task_b_rt', task_graph=task_b, arrival_cycle=100)]
    result = FuseSimulator(config=SimulationConfig(mode='fuse')).run(runtime_tasks, decomposition.microfragments)
    dispatches = [event for event in result.traces['events'] if event['event'] == 'dispatch']
    assert any((event.get('path_id') for event in dispatches))
    assert any((event.get('structural_reuse') and not event.get('reused_result') for event in dispatches))
    assert float(result.metrics['representative_path_reuse_rate']) > 0.0

def test_cpp_bridge_smoke(tmp_path):
    if not has_cpp_toolchain():
        pytest.skip('no supported C++ toolchain is available in this environment')
    root = Path(__file__).resolve().parents[1]
    bridge = CppSimulatorBridge(root)
    graph = build_aes128_task(1024, 'aes_cpp')
    runtime_tasks = BenchmarkSuite(engine='cpp')._runtime_tasks(graph, 1)
    result = bridge.run([graph], runtime_tasks, mode='fuse', scenario_name='smoke', out_dir=tmp_path)
    assert float(result.metrics['makespan_ms']) > 0.0
    assert result.events.shape[0] > 0

@pytest.mark.parametrize('mode', ['fuse', 'secure_ct', 'secure_elastic', 'secure_elastic_no_bucket', 'secure_elastic_no_pinning', 'secure_elastic_no_mask', 'secure_elastic_no_epoch', 'secure_elastic_no_prefetch', 'multicore'])
def test_cpp_bridge_aligns_with_python_contract(tmp_path, mode):
    if not has_cpp_toolchain():
        pytest.skip('no supported C++ toolchain is available in this environment')
    root = Path(__file__).resolve().parents[1]
    graph = build_aes128_task(1024, 'aes_cpp_contract')
    runtime_tasks = BenchmarkSuite(engine='py')._runtime_tasks(graph, 1)
    py_suite = BenchmarkSuite(engine='py')
    py_result, py_events = py_suite._simulate([graph], runtime_tasks, mode=mode, scenario_name=f'contract_py_{mode}', raw_dir=None)
    cpp_result = CppSimulatorBridge(root).run([graph], runtime_tasks, mode=mode, scenario_name=f'contract_cpp_{mode}', out_dir=tmp_path)
    required_metrics = {'makespan_ms', 'energy_nj', 'rfu_utilization', 'result_reuse_rate', 'structural_reuse_rate', 'switch_cycles_total', 'fvm_avg_residency_bytes', 'fvm_redundant_reads', 'lrc_switch_events', 'scheduler_pipeline_stall_cycles', 'kappa_key_match_ratio', 'template_hit_ratio', 'valid_fragment_ratio'}
    assert required_metrics.issubset(set(cpp_result.metrics))
    if not py_events.empty:
        assert {'switch_cycles', 'template_family', 'reused_result', 'structural_reuse', 'path_id', 'path_reuse_count'}.issubset(set(cpp_result.events.columns))
    assert cpp_result.metrics['makespan_ms'] == pytest.approx(py_result.metrics['makespan_ms'], rel=0.25, abs=1e-06)
    assert cpp_result.metrics['rfu_utilization'] == pytest.approx(py_result.metrics['rfu_utilization'], rel=0.25, abs=1e-06)
    assert cpp_result.metrics['switch_cycles_total'] == pytest.approx(py_result.metrics['switch_cycles_total'], rel=0.3, abs=1.0)

def test_benchmark_suite_smoke(tmp_path):
    suite = BenchmarkSuite(engine='py')
    suite.toolkit = _write_eval_reports(tmp_path / 'eval')
    artifacts = suite.run(tmp_path)
    assert (tmp_path / 'fig4_latency.csv').exists()
    assert (tmp_path / 'raw' / 'fig4_raw.csv').exists()
    assert not artifacts.dataframes['fig4_latency'].empty

def test_benchmark_suite_cpp_run_smoke(tmp_path):
    if not has_cpp_toolchain():
        pytest.skip('no supported C++ toolchain is available in this environment')
    suite = BenchmarkSuite(engine='cpp')
    suite.toolkit = _write_eval_reports(tmp_path / 'eval')
    artifacts = suite.run(tmp_path)
    assert (tmp_path / 'fig6_mixed.csv').exists()
    assert (tmp_path / 'fig11a_sched.csv').exists()
    assert (tmp_path / 'fig12_reconfig.csv').exists()
    assert (tmp_path / 'raw' / 'fig6_raw.csv').exists()
    assert not artifacts.dataframes['fig6_mixed'].empty
    assert not artifacts.dataframes['fig11a_sched'].empty
    assert not artifacts.dataframes['fig12_reconfig'].empty
    assert not artifacts.dataframes['fig12_reconfig']['source'].astype(str).str.contains('analytic_fallback').any()
    assert 'FPGA' not in set(artifacts.dataframes['fig12_reconfig']['platform'].astype(str))

def test_benchmark_suite_cpp_mixed_trace_aligns_with_python(tmp_path):
    if not has_cpp_toolchain():
        pytest.skip('no supported C++ toolchain is available in this environment')
    py_suite = BenchmarkSuite(engine='py')
    cpp_suite = BenchmarkSuite(engine='cpp')
    py_suite.toolkit = _write_eval_reports(tmp_path / 'py_eval')
    cpp_suite.toolkit = _write_eval_reports(tmp_path / 'cpp_eval')
    py_raw_dir = tmp_path / 'py_raw'
    cpp_raw_dir = tmp_path / 'cpp_raw'
    py_raw_dir.mkdir()
    cpp_raw_dir.mkdir()
    py_raw6, _ = py_suite._raw_fig6(py_raw_dir)
    cpp_raw6, _ = cpp_suite._raw_fig6(cpp_raw_dir)
    _assert_frame_numeric_alignment(py_raw6, cpp_raw6, ['window_ms', 'mode'], ['makespan_ms', 'rfu_util', 'energy_nj'], rel=0.35, abs_tol=1e-06)
    py_fig6 = py_suite._figure_fig6(py_raw6, py_suite._fit_calibration({'fig4_raw': py_suite._raw_fig4(None), 'fig6_raw': py_raw6}))
    cpp_fig6 = cpp_suite._figure_fig6(cpp_raw6, cpp_suite._fit_calibration({'fig4_raw': cpp_suite._raw_fig4(None), 'fig6_raw': cpp_raw6}))
    _assert_frame_numeric_alignment(py_fig6, cpp_fig6, ['window_ms'], ['baseline_makespan_ms', 'morph_makespan_ms', 'baseline_util', 'morph_util'], rel=0.35, abs_tol=1e-06)
    py_fig11_dir = tmp_path / 'py_fig11'
    cpp_fig11_dir = tmp_path / 'cpp_fig11'
    py_fig11_dir.mkdir()
    cpp_fig11_dir.mkdir()
    py_fig11 = py_suite._raw_fig11(py_fig11_dir)
    cpp_fig11 = cpp_suite._raw_fig11(cpp_fig11_dir)
    _assert_frame_numeric_alignment(py_fig11['fig11a_sched'], cpp_fig11['fig11a_sched'], ['bucket'], ['value'], rel=0.35, abs_tol=1e-06)
    _assert_frame_numeric_alignment(py_fig11['fig11b_switch'], cpp_fig11['fig11b_switch'], ['platform', 'metric'], ['value'], rel=0.35, abs_tol=1e-06)
    _assert_frame_numeric_alignment(py_fig11['fig11c_summary'], cpp_fig11['fig11c_summary'], ['metric'], ['value'], rel=0.35, abs_tol=1.0)
    py_fig12_dir = tmp_path / 'py_fig12'
    cpp_fig12_dir = tmp_path / 'cpp_fig12'
    py_fig12_dir.mkdir()
    cpp_fig12_dir.mkdir()
    py_fig12 = py_suite._raw_fig12(py_fig12_dir)
    cpp_fig12 = cpp_suite._raw_fig12(cpp_fig12_dir)
    _assert_frame_numeric_alignment(py_fig12, cpp_fig12, ['scenario', 'platform', 'metric'], ['value'], rel=0.35, abs_tol=1e-06)

def test_eval_toolkit_requires_real_reports():
    toolkit = EvaluationToolkit(Path(__file__).resolve().parents[1])
    with pytest.raises(FileNotFoundError):
        toolkit.estimate_area_power(rfu_count=16, feature_bytes=8 * 1024 * 1024, fragment_bytes=2 * 1024 * 1024, microop_bytes=512 * 1024, utilization=0.6, fvm_reads=100, fvm_writes=50, scheduler_decisions=200, switch_events=40)
    with pytest.raises(FileNotFoundError):
        toolkit.estimate_dram(1024.0)

def test_eval_toolkit_parses_external_reports(tmp_path):
    reports = tmp_path / 'reports'
    reports.mkdir()
    (reports / 'dramsim3.txt').write_text('avg latency: 63.5 ns\nsustained bw: 71.2\n', encoding='utf-8')
    (reports / 'cacti_feature.txt').write_text('Area (mm2): 3.2\nAccess time: 1.8 ns\nDynamic read energy: 0.9 nJ\n', encoding='utf-8')
    (reports / 'cacti_fragment.txt').write_text('Area (mm2): 0.7\nAccess time: 1.1 ns\nDynamic read energy: 0.3 nJ\n', encoding='utf-8')
    (reports / 'cacti_microop.txt').write_text('Area (mm2): 0.2\nAccess time: 0.6 ns\nDynamic read energy: 0.1 nJ\n', encoding='utf-8')
    (reports / 'dc_area.rpt').write_text('Total cell area: 11.03\nRFU 3.27\nTARS 1.95\nFVM 4.54\n', encoding='utf-8')
    (reports / 'ptpx_power.rpt').write_text('Total Power 8.27 W\nRFU 3.81 W\nScheduler+FVM 3.19 W\nStatic 1.27 W\n', encoding='utf-8')
    config_path = tmp_path / 'eval_tools.json'
    config_path.write_text(json.dumps({'dramsim3_report': 'reports/dramsim3.txt', 'cacti_feature_report': 'reports/cacti_feature.txt', 'cacti_fragment_report': 'reports/cacti_fragment.txt', 'cacti_microop_report': 'reports/cacti_microop.txt', 'dc_report': 'reports/dc_area.rpt', 'ptpx_report': 'reports/ptpx_power.rpt'}), encoding='utf-8')
    toolkit = EvaluationToolkit(tmp_path, config_path=config_path)
    dram = toolkit.estimate_dram(1024.0)
    estimate = toolkit.estimate_area_power(rfu_count=16, feature_bytes=8 * 1024 * 1024, fragment_bytes=2 * 1024 * 1024, microop_bytes=512 * 1024, utilization=0.5, fvm_reads=100, fvm_writes=50, scheduler_decisions=200, switch_events=40)
    assert dram['source'] == 'dramsim3_report'
    assert estimate['source'] == 'dc_ptpx_reports'
    assert estimate['total_area_mm2'] == pytest.approx(11.03)
    assert estimate['total_power_w'] == pytest.approx(8.27)

def test_eval_toolkit_parses_openroad_reports(tmp_path):
    toolkit = _write_eval_reports(tmp_path)
    parsed = toolkit.parse_openroad()
    estimate = toolkit.estimate_area_power(rfu_count=16, feature_bytes=8 * 1024 * 1024, fragment_bytes=2 * 1024 * 1024, microop_bytes=512 * 1024, utilization=0.5, fvm_reads=100, fvm_writes=50, scheduler_decisions=200, switch_events=40)
    assert parsed is not None
    assert parsed['total_area_mm2'] == pytest.approx(0.245)
    assert parsed['total_power_w'] == pytest.approx(0.184)
    assert parsed['fmax_mhz'] == pytest.approx(1000.0 / 21.5)
    assert estimate['source'] == 'openroad_sky130'

def test_fig9_is_derived_from_raw_results():
    suite = BenchmarkSuite(engine='py')
    raw = suite._raw_fig9(None)
    fig9 = suite._figure_fig9(raw)
    assert {'latency_degraded', 'latency_morph', 'throughput_degraded', 'throughput_morph'}.issubset(set(fig9['metric']))
    assert float(fig9[fig9['metric'] == 'latency_morph']['value'].iloc[0]) > 0.0

def test_fig8_is_derived_from_decomposition():
    suite = BenchmarkSuite(engine='py')
    raw = suite._raw_fig8(None)
    figs = suite._figure_fig8(raw)
    assert {'fig8a_reuse', 'fig8b_fanout', 'fig8c_pool'} == set(figs)
    assert not figs['fig8a_reuse'].empty
    assert set(figs['fig8a_reuse']['bucket']) == {'Exact-hash', 'Layout adaptation', 'Unique'}

def test_hardware_benchmark_suite_smoke(tmp_path):
    suite = HardwareBenchmarkSuite(device='cpu', warmup=1, repeat=2, seed=1)
    artifacts = suite.run(tmp_path, cnn_models=['vgg16', 'resnet101'], cnn_batches=[1], cnn_image_size=16, cnn_width_mult=0.125, aes_key_bits=[128, 256], aes_blocks=[16], zkp_sizes=[64], mixed_rounds=1)
    assert (tmp_path / 'hardware_summary.csv').exists()
    assert (tmp_path / 'hardware_mixed_trace.csv').exists()
    assert (tmp_path / 'hardware_env.json').exists()
    assert not artifacts.summary.empty
    assert {'CNN', 'AES', 'ZKP'}.issubset(set(artifacts.summary['task']))
    assert {'vgg16_inference', 'resnet101_inference', 'aes128_ecb_encrypt', 'aes256_ecb_encrypt'}.issubset(set(artifacts.summary['kernel']))

def test_energy_benchmark_suite_smoke(tmp_path):
    suite = EnergyBenchmarkSuite(device='cpu', warmup=1, repeat=1, seed=1)
    artifacts = suite.run(tmp_path)
    assert (tmp_path / 'fig5_energy.csv').exists()
    assert (tmp_path / 'fig5_energy_raw.csv').exists()
    assert (tmp_path / 'fig5_energy_env.json').exists()
    assert not artifacts.summary.empty
    assert {'ZKP (Groth16)', 'CNN (VGG16)', 'Crypto (AES-128)'}.issubset(set(artifacts.summary['task']))
    assert {ACCELERATOR_NAME, f'{ACCELERATOR_NAME}-SW'}.issubset(set(artifacts.summary['platform']))
    assert set(artifacts.summary['energy_source']).issubset({'cpu_process_proxy', 'gpu_nvml'})

def test_repo_results_plot_contract(tmp_path):
    root = Path(__file__).resolve().parents[1]
    results_dir = root / 'results'
    plot_fig7(pd.read_csv(results_dir / 'fig7_area_power.csv'), tmp_path)
    plot_fig11(
        pd.read_csv(results_dir / 'fig11a_sched.csv'),
        pd.read_csv(results_dir / 'fig11b_switch.csv'),
        pd.read_csv(results_dir / 'fig11c_summary.csv'),
        pd.read_csv(results_dir / 'fig12_reconfig.csv'),
        tmp_path,
    )
    plot_fig12(pd.read_csv(results_dir / 'fig12_reconfig.csv'), tmp_path)
    assert (tmp_path / 'fig7_area_power.png').exists()
    assert (tmp_path / 'fig11.png').exists()
    assert (tmp_path / 'fig12_reconfig.png').exists()
