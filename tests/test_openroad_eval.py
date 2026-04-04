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
