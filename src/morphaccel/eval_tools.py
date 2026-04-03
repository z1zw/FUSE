from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable
import json
import re
import pandas as pd

@dataclass
class ToolReportConfig:
    dramsim3_report: Path | None = None
    cacti_feature_report: Path | None = None
    cacti_fragment_report: Path | None = None
    cacti_microop_report: Path | None = None
    dc_report: Path | None = None
    ptpx_report: Path | None = None
    openroad_area_report: Path | None = None
    openroad_power_report: Path | None = None
    openroad_timing_report: Path | None = None
    openroad_metrics_json: Path | None = None
    openroad_metrics_csv: Path | None = None

def _maybe_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path

class EvaluationToolkit:

    def __init__(self, project_root: Path, config_path: Path | None=None):
        self.project_root = Path(project_root)
        self.config_path = config_path or self.project_root / 'configs' / 'eval_tools.json'
        self.config = self._load_config()

    def _load_config(self) -> ToolReportConfig:
        if not self.config_path.exists():
            return ToolReportConfig()
        payload = json.loads(self.config_path.read_text(encoding='utf-8'))
        return ToolReportConfig(dramsim3_report=_maybe_path(self.project_root, payload.get('dramsim3_report')), cacti_feature_report=_maybe_path(self.project_root, payload.get('cacti_feature_report')), cacti_fragment_report=_maybe_path(self.project_root, payload.get('cacti_fragment_report')), cacti_microop_report=_maybe_path(self.project_root, payload.get('cacti_microop_report')), dc_report=_maybe_path(self.project_root, payload.get('dc_report')), ptpx_report=_maybe_path(self.project_root, payload.get('ptpx_report')), openroad_area_report=_maybe_path(self.project_root, payload.get('openroad_area_report')), openroad_power_report=_maybe_path(self.project_root, payload.get('openroad_power_report')), openroad_timing_report=_maybe_path(self.project_root, payload.get('openroad_timing_report')), openroad_metrics_json=_maybe_path(self.project_root, payload.get('openroad_metrics_json')), openroad_metrics_csv=_maybe_path(self.project_root, payload.get('openroad_metrics_csv')))

    def parse_dramsim3(self) -> Dict[str, float] | None:
        path = self.config.dramsim3_report
        if path is None or not path.exists():
            return None
        text = path.read_text(encoding='utf-8', errors='ignore')
        latency = self._match_float(text, ['avg[_ ]latency[^0-9]*([0-9.]+)', 'average[_ ]read[_ ]latency[^0-9]*([0-9.]+)'])
        bw = self._match_float(text, ['sustained[_ ]bw[^0-9]*([0-9.]+)', 'bandwidth[^0-9]*([0-9.]+)'])
        out: Dict[str, float] = {}
        if latency is not None:
            out['dram_avg_latency_ns'] = latency
        if bw is not None:
            out['dram_sustained_bw_gbps'] = bw
        return out or None

    def parse_cacti(self) -> Dict[str, Dict[str, float]]:
        reports = {'feature_buffer': self.config.cacti_feature_report, 'fragment_cache': self.config.cacti_fragment_report, 'micro_op_buffer': self.config.cacti_microop_report}
        out: Dict[str, Dict[str, float]] = {}
        for name, path in reports.items():
            if path is None or not path.exists():
                continue
            text = path.read_text(encoding='utf-8', errors='ignore')
            area_mm2 = self._match_float(text, ['area[^0-9]*([0-9.]+)\\s*mm\\^?2', 'area \\(mm2\\)[^0-9]*([0-9.]+)'])
            read_ns = self._match_float(text, ['access time[^0-9]*([0-9.]+)\\s*ns', 'read latency[^0-9]*([0-9.]+)'])
            dyn_nj = self._match_float(text, ['dynamic read energy[^0-9]*([0-9.]+)\\s*nJ', 'dynamic energy[^0-9]*([0-9.]+)\\s*nJ'])
            values = {}
            if area_mm2 is not None:
                values['area_mm2'] = area_mm2
            if read_ns is not None:
                values['read_ns'] = read_ns
            if dyn_nj is not None:
                values['dynamic_nj'] = dyn_nj
            if values:
                out[name] = values
        return out

    def parse_dc_area(self) -> Dict[str, float] | None:
        path = self.config.dc_report
        if path is None or not path.exists():
            return None
        text = path.read_text(encoding='utf-8', errors='ignore')
        total = self._match_float(text, ['Total cell area:\\s*([0-9.]+)', 'Total area:\\s*([0-9.]+)'])
        if total is None:
            return None
        out: Dict[str, float] = {'total_area_mm2': total}
        for key, patterns in {'rfu_area_mm2': ['RFU[^0-9]*([0-9.]+)'], 'tars_area_mm2': ['TARS[^0-9]*([0-9.]+)', 'Scheduler[^0-9]*([0-9.]+)'], 'fvm_area_mm2': ['FVM[^0-9]*([0-9.]+)', 'feature buffer[^0-9]*([0-9.]+)']}.items():
            value = self._match_float(text, patterns)
            if value is not None:
                out[key] = value
        return out

    def parse_ptpx_power(self) -> Dict[str, float] | None:
        path = self.config.ptpx_report
        if path is None or not path.exists():
            return None
        text = path.read_text(encoding='utf-8', errors='ignore')
        total = self._match_float(text, ['Total Power[^0-9]*([0-9.]+)\\s*W', 'Total\\s+([0-9.]+)\\s*W'])
        if total is None:
            return None
        out: Dict[str, float] = {'total_power_w': total}
        for key, patterns in {'rfu_dynamic_w': ['RFU[^0-9]*([0-9.]+)\\s*W'], 'scheduler_fvm_dynamic_w': ['Scheduler\\+FVM[^0-9]*([0-9.]+)\\s*W', 'FVM[^0-9]*([0-9.]+)\\s*W'], 'static_w': ['Static[^0-9]*([0-9.]+)\\s*W', 'Leakage[^0-9]*([0-9.]+)\\s*W']}.items():
            value = self._match_float(text, patterns)
            if value is not None:
                out[key] = value
        return out

    def parse_openroad(self) -> Dict[str, float] | None:
        metrics_json_path = self.config.openroad_metrics_json
        metrics_csv_path = self.config.openroad_metrics_csv
        area_path = self.config.openroad_area_report
        power_path = self.config.openroad_power_report
        timing_path = self.config.openroad_timing_report
        if not any((path is not None and path.exists() for path in [metrics_json_path, metrics_csv_path, area_path, power_path, timing_path])):
            return None
        out: Dict[str, float] = {}
        if metrics_json_path is not None and metrics_json_path.exists():
            payload = json.loads(metrics_json_path.read_text(encoding='utf-8'))
            out.update(self._parse_openroad_metrics_mapping(payload))
        elif metrics_csv_path is not None and metrics_csv_path.exists():
            frame = pd.read_csv(metrics_csv_path)
            if not frame.empty:
                row = frame.iloc[0].to_dict()
                out.update(self._parse_openroad_metrics_mapping(row))
        if area_path is not None and area_path.exists():
            text = area_path.read_text(encoding='utf-8', errors='ignore')
            area_um2 = self._match_float(text, ['design area\\s*([0-9.]+)\\s*u\\^?2', 'design_area[^0-9]*([0-9.]+)', 'area[^0-9]*([0-9.]+)\\s*um\\^?2'])
            util_pct = self._match_float(text, ['utilization[^0-9]*([0-9.]+)', 'core utilization[^0-9]*([0-9.]+)'])
            if area_um2 is not None and 'total_area_mm2' not in out:
                out['total_area_mm2'] = area_um2 / 1000000.0
            if util_pct is not None and 'openroad_utilization_pct' not in out:
                out['openroad_utilization_pct'] = util_pct
        if power_path is not None and power_path.exists():
            text = power_path.read_text(encoding='utf-8', errors='ignore')
            total_mw = self._match_float(text, ['total\\s+[a-z ]*power[^0-9]*([0-9.]+)\\s*mw', 'total power[^0-9]*([0-9.]+)\\s*mw', 'total[^0-9]*([0-9.]+)\\s*mw'])
            leakage_mw = self._match_float(text, ['leakage[^0-9]*([0-9.]+)\\s*mw', 'static[^0-9]*([0-9.]+)\\s*mw'])
            dynamic_mw = self._match_float(text, ['internal[^0-9]*([0-9.]+)\\s*mw', 'switching[^0-9]*([0-9.]+)\\s*mw', 'dynamic[^0-9]*([0-9.]+)\\s*mw'])
            if total_mw is not None and 'total_power_w' not in out:
                out['total_power_w'] = total_mw / 1000.0
            if leakage_mw is not None and 'static_w' not in out:
                out['static_w'] = leakage_mw / 1000.0
            if dynamic_mw is not None and 'rfu_dynamic_w' not in out:
                out['rfu_dynamic_w'] = dynamic_mw / 1000.0
        if timing_path is not None and timing_path.exists():
            text = timing_path.read_text(encoding='utf-8', errors='ignore')
            period_ns = self._match_float(text, ['clock period[^0-9]*([0-9.]+)\\s*ns', 'period[^0-9]*([0-9.]+)\\s*ns'])
            wns_ns = self._match_float(text, ['worst negative slack[^-0-9]*([-0-9.]+)', '\\bwns\\b[^-0-9]*([-0-9.]+)'])
            if period_ns is not None:
                achieved_period_ns = period_ns
                if wns_ns is not None and wns_ns < 0.0:
                    achieved_period_ns = period_ns - wns_ns
                if 'clock_period_ns' not in out:
                    out['clock_period_ns'] = achieved_period_ns
                if 'fmax_mhz' not in out:
                    out['fmax_mhz'] = 1000.0 / max(achieved_period_ns, 1e-09)
            if wns_ns is not None and 'wns_ns' not in out:
                out['wns_ns'] = wns_ns
        return out or None

    def _parse_openroad_metrics_mapping(self, payload: Dict[str, object]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        area_um2 = self._coerce_float(payload.get('finish__design__instance__area'), payload.get('finish__design__instance__area__stdcell'), payload.get('design__instance__area'))
        util = self._coerce_float(payload.get('finish__design__instance__utilization'), payload.get('globalroute__design__instance__utilization'), payload.get('cts__design__instance__utilization'))
        total_power_w = self._coerce_float(payload.get('finish__power__total'), payload.get('globalroute__power__total'), payload.get('cts__power__total'))
        leakage_w = self._coerce_float(payload.get('finish__power__leakage__total'), payload.get('globalroute__power__leakage__total'), payload.get('cts__power__leakage__total'))
        internal_w = self._coerce_float(payload.get('finish__power__internal__total'), payload.get('globalroute__power__internal__total'), payload.get('cts__power__internal__total'))
        switching_w = self._coerce_float(payload.get('finish__power__switching__total'), payload.get('globalroute__power__switching__total'), payload.get('cts__power__switching__total'))
        setup_ws_ns = self._coerce_float(payload.get('finish__timing__setup__ws'), payload.get('globalroute__timing__setup__ws'), payload.get('cts__timing__setup__ws'))
        clock_period_ns = self._infer_clock_period_ns(payload)
        if area_um2 is not None:
            out['total_area_mm2'] = area_um2 / 1000000.0
        if util is not None:
            out['openroad_utilization_pct'] = util * 100.0 if util <= 1.0 else util
        if total_power_w is not None:
            out['total_power_w'] = total_power_w
        if leakage_w is not None:
            out['static_w'] = leakage_w
        if internal_w is not None or switching_w is not None:
            out['rfu_dynamic_w'] = max((internal_w or 0.0) + (switching_w or 0.0), 0.0)
        if setup_ws_ns is not None:
            out['wns_ns'] = setup_ws_ns
        if clock_period_ns is not None:
            achieved_period_ns = clock_period_ns
            if setup_ws_ns is not None and setup_ws_ns < 0.0:
                achieved_period_ns = clock_period_ns - setup_ws_ns
            out['clock_period_ns'] = achieved_period_ns
            out['fmax_mhz'] = 1000.0 / max(achieved_period_ns, 1e-09)
        return out

    def _infer_clock_period_ns(self, payload: Dict[str, object]) -> float | None:
        direct = self._coerce_float(payload.get('constraints__clocks__period'), payload.get('timing__clock__period'))
        if direct is not None:
            return direct
        details = payload.get('constraints__clocks__details')
        if isinstance(details, list):
            for item in details:
                if isinstance(item, str):
                    match = re.search(':\\s*([0-9.]+)', item)
                    if match:
                        return float(match.group(1))
        return None

    def _coerce_float(self, *values: object) -> float | None:
        for value in values:
            if value is None:
                continue
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    continue
        return None

    def estimate_dram(self, total_bytes: float) -> Dict[str, float]:
        parsed = self.parse_dramsim3()
        if parsed is not None:
            return {**parsed, 'source': 'dramsim3_report'}
        sustained_bw_gbps = 76.8
        avg_latency_ns = 42.0 + min(90.0, total_bytes / max(1.0, 64.0 * 1024.0 * 1024.0))
        return {'dram_avg_latency_ns': avg_latency_ns, 'dram_sustained_bw_gbps': sustained_bw_gbps, 'source': 'analytic_fallback'}

    def estimate_memory_macros(self, feature_bytes: int, fragment_bytes: int, microop_bytes: int) -> Dict[str, Dict[str, float]]:
        parsed = self.parse_cacti()
        if parsed:
            return parsed

        def _fallback(bytes_used: int, area_scale: float, read_scale: float) -> Dict[str, float]:
            kb = bytes_used / 1024.0
            return {'area_mm2': max(0.02, kb * area_scale), 'read_ns': 0.35 + kb * read_scale, 'dynamic_nj': max(0.01, kb * 0.0015)}
        return {'feature_buffer': _fallback(feature_bytes, area_scale=0.00042, read_scale=2e-05), 'fragment_cache': _fallback(fragment_bytes, area_scale=0.0005, read_scale=3e-05), 'micro_op_buffer': _fallback(microop_bytes, area_scale=0.00036, read_scale=1.5e-05)}

    def estimate_area_power(self, *, rfu_count: int, feature_bytes: int, fragment_bytes: int, microop_bytes: int, utilization: float, fvm_reads: float, fvm_writes: float, scheduler_decisions: float, switch_events: float) -> Dict[str, float]:
        dc = self.parse_dc_area()
        ptpx = self.parse_ptpx_power()
        openroad = self.parse_openroad()
        macros = self.estimate_memory_macros(feature_bytes, fragment_bytes, microop_bytes)
        if openroad is not None:
            total_area = openroad.get('total_area_mm2', 0.0)
            total_power = openroad.get('total_power_w', 0.0)
            fvm_area = macros['feature_buffer']['area_mm2'] + macros['fragment_cache']['area_mm2'] + 0.14
            tars_area = 0.1 + 0.012 * rfu_count
            other_area = 0.26 + macros['micro_op_buffer']['area_mm2']
            residual_area = max(total_area - fvm_area - tars_area - other_area, 0.0)
            rfu_area = residual_area if residual_area > 0.0 else max(total_area * 0.28, 0.01)
            static_w = openroad.get('static_w', max(total_power * 0.18, 0.0))
            rfu_dynamic = openroad.get('rfu_dynamic_w', max(total_power - static_w, 0.0))
            scheduler_fvm_dynamic = max(total_power - static_w - rfu_dynamic, 0.0)
            return {'total_area_mm2': total_area, 'fvm_area_mm2': fvm_area, 'rfu_area_mm2': rfu_area, 'tars_area_mm2': tars_area, 'other_area_mm2': other_area, 'total_power_w': total_power, 'rfu_dynamic_w': rfu_dynamic, 'scheduler_fvm_dynamic_w': scheduler_fvm_dynamic, 'static_w': static_w, 'fmax_mhz': openroad.get('fmax_mhz', 0.0), 'clock_period_ns': openroad.get('clock_period_ns', 0.0), 'wns_ns': openroad.get('wns_ns', 0.0), 'source': 'openroad_sky130'}
        if dc is not None and ptpx is not None:
            total_area = dc['total_area_mm2']
            total_power = ptpx['total_power_w']
            return {'total_area_mm2': total_area, 'fvm_area_mm2': dc.get('fvm_area_mm2', macros['feature_buffer']['area_mm2'] + macros['fragment_cache']['area_mm2'] + 0.18), 'rfu_area_mm2': dc.get('rfu_area_mm2', total_area * 0.28), 'tars_area_mm2': dc.get('tars_area_mm2', total_area * 0.14), 'other_area_mm2': max(0.0, total_area - dc.get('fvm_area_mm2', 0.0) - dc.get('rfu_area_mm2', 0.0) - dc.get('tars_area_mm2', 0.0)), 'total_power_w': total_power, 'rfu_dynamic_w': ptpx.get('rfu_dynamic_w', total_power * 0.45), 'scheduler_fvm_dynamic_w': ptpx.get('scheduler_fvm_dynamic_w', total_power * 0.38), 'static_w': ptpx.get('static_w', total_power * 0.17), 'source': 'dc_ptpx_reports'}
        fvm_area = macros['feature_buffer']['area_mm2'] + macros['fragment_cache']['area_mm2'] + 0.14
        rfu_area = rfu_count * 0.19
        tars_area = 0.1 + 0.012 * rfu_count
        other_area = 0.26 + macros['micro_op_buffer']['area_mm2']
        total_area = fvm_area + rfu_area + tars_area + other_area
        rfu_dynamic = max(0.3, utilization * rfu_count * 0.26)
        scheduler_fvm_dynamic = max(0.25, (fvm_reads + fvm_writes) * 0.00045 + scheduler_decisions * 0.00012 + switch_events * 0.00018)
        static_w = max(0.18, total_area * 0.22)
        total_power = rfu_dynamic + scheduler_fvm_dynamic + static_w
        return {'total_area_mm2': total_area, 'fvm_area_mm2': fvm_area, 'rfu_area_mm2': rfu_area, 'tars_area_mm2': tars_area, 'other_area_mm2': other_area, 'total_power_w': total_power, 'rfu_dynamic_w': rfu_dynamic, 'scheduler_fvm_dynamic_w': scheduler_fvm_dynamic, 'static_w': static_w, 'source': 'analytic_fallback'}

    def _match_float(self, text: str, patterns: Iterable[str]) -> float | None:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None

def write_eval_tool_template(path: Path) -> None:
    payload = {'dramsim3_report': 'reports/dramsim3.txt', 'cacti_feature_report': 'reports/cacti_feature.txt', 'cacti_fragment_report': 'reports/cacti_fragment.txt', 'cacti_microop_report': 'reports/cacti_microop.txt', 'dc_report': 'reports/dc_area.rpt', 'ptpx_report': 'reports/ptpx_power.rpt', 'openroad_area_report': 'reports/openroad/fuse_area.rpt', 'openroad_power_report': 'reports/openroad/fuse_power.rpt', 'openroad_timing_report': 'reports/openroad/fuse_timing.rpt', 'openroad_metrics_json': 'reports/openroad/fuse_metrics.json', 'openroad_metrics_csv': 'reports/openroad/fuse_metrics.csv'}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
