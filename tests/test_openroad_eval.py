from __future__ import annotations
import json
from pathlib import Path
from fuse.eval_tools import EvaluationToolkit

def test_parse_openroad_metrics_json(tmp_path: Path) -> None:
    metrics_path = tmp_path / 'fuse_metrics.json'
    config_path = tmp_path / 'eval_tools.json'
    metrics_path.write_text(json.dumps({'constraints__clocks__details': ['clk: 20.0000'], 'finish__design__instance__area': 1103000.0, 'finish__design__instance__utilization': 0.551, 'finish__power__internal__total': 5.0, 'finish__power__switching__total': 2.0, 'finish__power__leakage__total': 1.5, 'finish__power__total': 8.5, 'finish__timing__setup__ws': -1.25}), encoding='utf-8')
    config_path.write_text(json.dumps({'openroad_metrics_json': str(metrics_path)}), encoding='utf-8')
    toolkit = EvaluationToolkit(tmp_path, config_path=config_path)
    parsed = toolkit.parse_openroad()
    assert parsed is not None
    assert parsed['total_area_mm2'] == 1.103
    assert parsed['openroad_utilization_pct'] == 55.1
    assert parsed['total_power_w'] == 8.5
    assert parsed['static_w'] == 1.5
    assert parsed['rfu_dynamic_w'] == 7.0
    assert parsed['wns_ns'] == -1.25
    assert parsed['clock_period_ns'] == 21.25
    assert round(parsed['fmax_mhz'], 6) == round(1000.0 / 21.25, 6)

def test_require_area_power_splits_macros_by_role(tmp_path: Path) -> None:
    openroad_report = tmp_path / '6_finish.rpt'
    synth_stat = tmp_path / 'synth_stat.txt'
    feature_report = tmp_path / 'feature.txt'
    fragment_report = tmp_path / 'fragment.txt'
    microop_report = tmp_path / 'microop.txt'
    config_path = tmp_path / 'eval_tools.json'
    openroad_report.write_text('\n'.join([
        'clk period_min = 3.90 fmax = 256.39',
        'worst slack max 16.10',
        'Total                  2.28e-04   1.82e-04   1.13e-09   4.10e-04 100.0%',
    ]), encoding='utf-8')
    synth_stat.write_text('\n'.join([
        "=== fuse_top ===",
        "   Chip area for module '\\fuse_top': 2329.734400",
        '     of which used for sequential elements: 880.844800 (37.81%)',
    ]), encoding='utf-8')
    feature_report.write_text('\n'.join([
        'Access time (ns): 6.86362',
        'Total dynamic read energy per access (nJ): 2.22871',
        'Total dynamic write energy per access (nJ): 2.4168',
        'Total leakage power of a bank (mW): 10553.1',
        'Data array: Area (mm2): 63.5173',
    ]), encoding='utf-8')
    fragment_report.write_text('\n'.join([
        'Access time (ns): 3.80674',
        'Total dynamic read energy per access (nJ): 1.10891',
        'Total dynamic write energy per access (nJ): 1.18247',
        'Total leakage power of a bank (mW): 2658.1',
        'Data array: Area (mm2): 18.5393',
    ]), encoding='utf-8')
    microop_report.write_text('\n'.join([
        'Access time (ns): 2.00067',
        'Total dynamic read energy per access (nJ): 0.459116',
        'Total dynamic write energy per access (nJ): 0.546945',
        'Total leakage power of a bank (mW): 670.722',
        'Data array: Area (mm2): 4.34727',
    ]), encoding='utf-8')
    config_path.write_text(json.dumps({
        'openroad_area_report': str(openroad_report),
        'openroad_power_report': str(openroad_report),
        'openroad_timing_report': str(openroad_report),
        'openroad_synth_stat_report': str(synth_stat),
        'cacti_feature_report': str(feature_report),
        'cacti_fragment_report': str(fragment_report),
        'cacti_microop_report': str(microop_report),
    }), encoding='utf-8')
    toolkit = EvaluationToolkit(tmp_path, config_path=config_path)
    parsed = toolkit.estimate_area_power(rfu_count=16, feature_bytes=8 * 1024 * 1024, fragment_bytes=2 * 1024 * 1024, microop_bytes=512 * 1024, utilization=0.6, fvm_reads=100.0, fvm_writes=50.0, scheduler_decisions=200.0, switch_events=40.0, makespan_ms=10.0)
    component_sum = parsed['fvm_area_mm2'] + parsed['rfu_area_mm2'] + parsed['tars_area_mm2'] + parsed['other_area_mm2']
    assert round(parsed['fvm_area_mm2'], 4) == 63.5173
    assert parsed['rfu_area_mm2'] > 4.34727
    assert parsed['tars_area_mm2'] > 18.5393
    assert abs(parsed['total_area_mm2'] - component_sum) < 1e-9
    assert parsed['total_power_w'] > 0.00041
    assert parsed['static_w'] > 0.0
    assert parsed['rfu_dynamic_w'] > 0.0
    assert parsed['scheduler_fvm_dynamic_w'] > 0.0
