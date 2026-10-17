from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable
import hashlib
import json
import re
import pandas as pd

FLOAT_RE = r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)'


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
    openroad_synth_stat_report: Path | None = None


def _maybe_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


class EvaluationToolkit:

    def __init__(self, project_root: Path, config_path: Path | None = None):
        self.project_root = Path(project_root)
        self.config_path = config_path or self.project_root / 'configs' / 'eval_tools.json'
        self.config = self._load_config()

    def _load_config(self) -> ToolReportConfig:
        if not self.config_path.exists():
            return ToolReportConfig()
        payload = json.loads(self.config_path.read_text(encoding='utf-8'))
        return ToolReportConfig(
            dramsim3_report=_maybe_path(self.project_root, payload.get('dramsim3_report')),
            cacti_feature_report=_maybe_path(self.project_root, payload.get('cacti_feature_report')),
            cacti_fragment_report=_maybe_path(self.project_root, payload.get('cacti_fragment_report')),
            cacti_microop_report=_maybe_path(self.project_root, payload.get('cacti_microop_report')),
            dc_report=_maybe_path(self.project_root, payload.get('dc_report')),
            ptpx_report=_maybe_path(self.project_root, payload.get('ptpx_report')),
            openroad_area_report=_maybe_path(self.project_root, payload.get('openroad_area_report')),
            openroad_power_report=_maybe_path(self.project_root, payload.get('openroad_power_report')),
            openroad_timing_report=_maybe_path(self.project_root, payload.get('openroad_timing_report')),
            openroad_metrics_json=_maybe_path(self.project_root, payload.get('openroad_metrics_json')),
            openroad_metrics_csv=_maybe_path(self.project_root, payload.get('openroad_metrics_csv')),
            openroad_synth_stat_report=_maybe_path(self.project_root, payload.get('openroad_synth_stat_report')),
        )

    def parse_dramsim3(self) -> Dict[str, float] | None:
        path = self.config.dramsim3_report
        if path is None or not path.exists():
            return None
        text = path.read_text(encoding='utf-8', errors='ignore')
        latency = self._match_float(
            text,
            [rf'avg[_ ]latency[^0-9-]*{FLOAT_RE}', rf'average[_ ]read[_ ]latency[^0-9-]*{FLOAT_RE}'],
        )
        bw = self._match_float(
            text,
            [rf'sustained[_ ]bw[^0-9-]*{FLOAT_RE}', rf'bandwidth[^0-9-]*{FLOAT_RE}'],
        )
        out: Dict[str, float] = {}
        if latency is not None:
            out['dram_avg_latency_ns'] = latency
        if bw is not None:
            out['dram_sustained_bw_gbps'] = bw
        return out or None

    def parse_cacti(self) -> Dict[str, Dict[str, float]]:
        reports = {
            'feature_buffer': self.config.cacti_feature_report,
            'fragment_cache': self.config.cacti_fragment_report,
            'micro_op_buffer': self.config.cacti_microop_report,
        }
        out: Dict[str, Dict[str, float]] = {}
        for name, path in reports.items():
            if path is None or not path.exists():
                continue
            text = path.read_text(encoding='utf-8', errors='ignore')
            area_mm2 = self._match_float(
                text,
                [
                    rf'data array:\s*area \(mm2\):\s*{FLOAT_RE}',
                    rf'area \(mm2\)[^0-9-]*{FLOAT_RE}',
                    rf'area[^0-9-]*{FLOAT_RE}\s*mm\^?2',
                ],
            )
            read_ns = self._match_float(
                text,
                [rf'access time[^0-9-]*{FLOAT_RE}\s*ns', rf'read latency[^0-9-]*{FLOAT_RE}'],
            )
            read_nj = self._match_float(
                text,
                [
                    rf'total dynamic read energy per access \(nj\):\s*{FLOAT_RE}',
                    rf'data array:\s*total dynamic read energy/access\s+\(nj\):\s*{FLOAT_RE}',
                    rf'dynamic read energy[^0-9-]*{FLOAT_RE}\s*nj',
                    rf'dynamic energy[^0-9-]*{FLOAT_RE}\s*nj',
                ],
            )
            write_nj = self._match_float(
                text,
                [
                    rf'total dynamic write energy per access \(nj\):\s*{FLOAT_RE}',
                    rf'dynamic write energy[^0-9-]*{FLOAT_RE}\s*nj',
                ],
            )
            leakage_mw = self._match_float(
                text,
                [
                    rf'total leakage power of a bank \(mw\):\s*{FLOAT_RE}',
                    rf'total leakage read/write power of a bank \(mw\):\s*{FLOAT_RE}',
                    rf'leakage power[^0-9-]*{FLOAT_RE}\s*mw',
                ],
            )
            values = {}
            if area_mm2 is not None:
                values['area_mm2'] = area_mm2
            if read_ns is not None:
                values['read_ns'] = read_ns
            if read_nj is not None:
                values['dynamic_nj'] = read_nj
            if write_nj is not None:
                values['dynamic_write_nj'] = write_nj
            if leakage_mw is not None:
                values['leakage_mw'] = leakage_mw
            if values:
                out[name] = values
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
                out.update(self._parse_openroad_metrics_mapping(frame.iloc[0].to_dict()))

        if area_path is not None and area_path.exists():
            text = area_path.read_text(encoding='utf-8', errors='ignore')
            area_um2 = self._match_float(
                text,
                [
                    rf'design area\s*{FLOAT_RE}\s*um\^?2',
                    rf'design_area[^0-9-]*{FLOAT_RE}',
                    rf'area[^0-9-]*{FLOAT_RE}\s*um\^?2',
                ],
            )
            util_pct = self._match_float(
                text,
                [rf'([0-9.]+)\s*%\s*utilization', rf'utilization[^0-9-]*{FLOAT_RE}', rf'core utilization[^0-9-]*{FLOAT_RE}'],
            )
            if area_um2 is not None and 'total_area_mm2' not in out:
                out['total_area_mm2'] = area_um2 / 1000000.0
            if util_pct is not None and 'openroad_utilization_pct' not in out:
                out['openroad_utilization_pct'] = util_pct

        if power_path is not None and power_path.exists():
            text = power_path.read_text(encoding='utf-8', errors='ignore')

            total_w = self._match_float(text, [rf'total\s+[a-z ]*power[^0-9-]*{FLOAT_RE}\s*W', rf'total power[^0-9-]*{FLOAT_RE}\s*W'])
            total_mw = self._match_float(text, [rf'total\s+[a-z ]*power[^0-9-]*{FLOAT_RE}\s*mW', rf'total power[^0-9-]*{FLOAT_RE}\s*mW'])
            leakage_w = self._match_float(text, [rf'leakage[^0-9-]*{FLOAT_RE}\s*W', rf'static[^0-9-]*{FLOAT_RE}\s*W'])
            leakage_mw = self._match_float(text, [rf'leakage[^0-9-]*{FLOAT_RE}\s*mW', rf'static[^0-9-]*{FLOAT_RE}\s*mW'])
            dynamic_w = self._match_float(text, [rf'internal[^0-9-]*{FLOAT_RE}\s*W', rf'switching[^0-9-]*{FLOAT_RE}\s*W', rf'dynamic[^0-9-]*{FLOAT_RE}\s*W'])
            dynamic_mw = self._match_float(text, [rf'internal[^0-9-]*{FLOAT_RE}\s*mW', rf'switching[^0-9-]*{FLOAT_RE}\s*mW', rf'dynamic[^0-9-]*{FLOAT_RE}\s*mW'])

            table_match = re.search(
                rf'Total\s+{FLOAT_RE}\s+{FLOAT_RE}\s+{FLOAT_RE}\s+{FLOAT_RE}\s+100\.0%',
                text,
                flags=re.IGNORECASE,
            )
            if table_match:
                internal_tab = float(table_match.group(1))
                switching_tab = float(table_match.group(2))
                leakage_tab = float(table_match.group(3))
                total_tab = float(table_match.group(4))
            else:
                internal_tab = None
                switching_tab = None
                leakage_tab = None
                total_tab = None

            if total_w is not None and 'total_power_w' not in out:
                out['total_power_w'] = total_w
            elif total_mw is not None and 'total_power_w' not in out:
                out['total_power_w'] = total_mw / 1000.0
            elif total_tab is not None and 'total_power_w' not in out:
                out['total_power_w'] = total_tab

            if leakage_w is not None and 'static_w' not in out:
                out['static_w'] = leakage_w
            elif leakage_mw is not None and 'static_w' not in out:
                out['static_w'] = leakage_mw / 1000.0
            elif leakage_tab is not None and 'static_w' not in out:
                out['static_w'] = leakage_tab

            if dynamic_w is not None and 'rfu_dynamic_w' not in out:
                out['rfu_dynamic_w'] = dynamic_w
            elif dynamic_mw is not None and 'rfu_dynamic_w' not in out:
                out['rfu_dynamic_w'] = dynamic_mw / 1000.0
            elif internal_tab is not None and switching_tab is not None and 'rfu_dynamic_w' not in out:
                out['rfu_dynamic_w'] = internal_tab + switching_tab

        if timing_path is not None and timing_path.exists():
            text = timing_path.read_text(encoding='utf-8', errors='ignore')
            period_ns = self._match_float(
                text,
                [
                    rf'clock period[^0-9-]*{FLOAT_RE}\s*ns',
                    rf'period[^0-9-]*{FLOAT_RE}\s*ns',
                    rf'period_min\s*=\s*{FLOAT_RE}',
                ],
            )
            fmax_mhz = self._match_float(text, [rf'fmax\s*=\s*{FLOAT_RE}'])
            wns_ns = self._match_float(
                text,
                [
                    rf'worst negative slack[^0-9-]*{FLOAT_RE}',
                    rf'worst slack[^0-9-]*{FLOAT_RE}',
                    rf'\bwns\b[^0-9-]*{FLOAT_RE}',
                ],
            )
            if period_ns is not None:
                achieved_period_ns = period_ns
                if wns_ns is not None and wns_ns < 0.0:
                    achieved_period_ns = period_ns - wns_ns
                if 'clock_period_ns' not in out:
                    out['clock_period_ns'] = achieved_period_ns
                if 'fmax_mhz' not in out:
                    out['fmax_mhz'] = 1000.0 / max(achieved_period_ns, 1e-09)
            elif fmax_mhz is not None and 'fmax_mhz' not in out:
                out['fmax_mhz'] = fmax_mhz
            if wns_ns is not None and 'wns_ns' not in out:
                out['wns_ns'] = wns_ns

        return out or None

    def parse_openroad_synth_stat(self) -> Dict[str, float] | None:
        path = self._resolve_openroad_synth_stat_path()
        if path is None or not path.exists():
            return None
        text = path.read_text(encoding='utf-8', errors='ignore')
        chip_area_um2 = self._match_float(text, [rf"Chip area for module '.*?':\s*{FLOAT_RE}"])
        seq_area_um2 = self._match_float(text, [rf'of which used for sequential elements:\s*{FLOAT_RE}'])
        seq_share_pct = self._match_float(text, [rf'of which used for sequential elements:\s*{FLOAT_RE}\s*\(({FLOAT_RE})%\)'], group=2)
        out: Dict[str, float] = {}
        if chip_area_um2 is not None:
            out['logic_area_mm2'] = chip_area_um2 / 1000000.0
        if seq_area_um2 is not None:
            out['sequential_area_mm2'] = seq_area_um2 / 1000000.0
        if seq_share_pct is not None:
            out['sequential_share_pct'] = seq_share_pct
        return out or None

    def provenance_summary(self) -> Dict[str, Dict[str, object]]:
        out: Dict[str, Dict[str, object]] = {}
        for name in ['dramsim3_report', 'cacti_feature_report', 'cacti_fragment_report', 'cacti_microop_report', 'openroad_area_report', 'openroad_power_report', 'openroad_timing_report', 'openroad_metrics_json', 'openroad_metrics_csv']:
            path = getattr(self.config, name)
            record = self._provenance_record(path)
            if record is not None:
                out[name] = record
        synth_path = self._resolve_openroad_synth_stat_path()
        synth_record = self._provenance_record(synth_path)
        if synth_record is not None:
            out['openroad_synth_stat_report'] = synth_record
        return out

    def parse_dc_ptpx(self) -> Dict[str, float] | None:
        """Synopsys Design Compiler area and PrimeTime PX power, the flow 4.1
        describes. Both reports are needed; either missing yields None."""
        dc_path = self.config.dc_report
        ptpx_path = self.config.ptpx_report
        if dc_path is None or ptpx_path is None or not dc_path.exists() or not ptpx_path.exists():
            return None
        dc = dc_path.read_text(encoding='utf-8', errors='ignore')
        ptpx = ptpx_path.read_text(encoding='utf-8', errors='ignore')
        total_area = self._match_float(dc, [rf'total[_ ]cell[_ ]area[^0-9-]*{FLOAT_RE}', rf'total[_ ]area[^0-9-]*{FLOAT_RE}'])
        total_power = self._match_float(ptpx, [rf'total[_ ]power[^0-9-]*{FLOAT_RE}'])
        if total_area is None or total_power is None:
            return None

        def component(text: str, name: str) -> float:
            value = self._match_float(text, [rf'(?im)^\s*{re.escape(name)}\b[^0-9-]*{FLOAT_RE}'])
            return float(value) if value is not None else 0.0

        rfu_area = component(dc, 'RFU')
        tars_area = component(dc, 'TARS')
        fvm_area = component(dc, 'FVM')
        rfu_power = component(ptpx, 'RFU')
        scheduler_fvm_power = component(ptpx, 'Scheduler+FVM')
        static_power = component(ptpx, 'Static')
        return {
            'total_area_mm2': float(total_area),
            'fvm_area_mm2': fvm_area,
            'rfu_area_mm2': rfu_area,
            'tars_area_mm2': tars_area,
            'other_area_mm2': max(float(total_area) - rfu_area - tars_area - fvm_area, 0.0),
            'total_power_w': float(total_power),
            'rfu_dynamic_w': rfu_power,
            'scheduler_fvm_dynamic_w': scheduler_fvm_power,
            'static_w': static_power,
        }

    def require_dram(self, total_bytes: float) -> Dict[str, float]:
        parsed = self.parse_dramsim3()
        if parsed is None:
            raise FileNotFoundError(f'DRAMSim3 report not found via {self.config_path}')
        return {**parsed, 'source': 'dramsim3_report'}

    def require_memory_macros(self) -> Dict[str, Dict[str, float]]:
        parsed = self.parse_cacti()
        missing = {'feature_buffer', 'fragment_cache', 'micro_op_buffer'} - set(parsed)
        if missing:
            raise FileNotFoundError(f"CACTI reports missing for: {', '.join(sorted(missing))}")
        return parsed

    def require_area_power(
        self,
        *,
        rfu_count: int,
        feature_bytes: int,
        fragment_bytes: int,
        microop_bytes: int,
        utilization: float = 0.0,
        fvm_reads: float = 0.0,
        fvm_writes: float = 0.0,
        scheduler_decisions: float = 0.0,
        switch_events: float = 0.0,
        makespan_ms: float = 1.0,
    ) -> Dict[str, float]:
        # Signed-off tool reports take precedence over the open-source estimate.
        dc_ptpx = self.parse_dc_ptpx()
        if dc_ptpx is not None:
            return {**dc_ptpx, 'source': 'dc_ptpx_reports'}

        openroad = self.parse_openroad()
        if openroad is not None:
            synth = self.parse_openroad_synth_stat() or {}
            macros = self.require_memory_macros()

            logic_area = max(float(synth.get('logic_area_mm2', openroad.get('total_area_mm2', 0.0))), 0.0)
            seq_share = min(max(float(synth.get('sequential_share_pct', 37.5)) / 100.0, 0.20), 0.75)
            util = min(max(float(utilization), 0.0), 1.0)

            rfu_logic_share = min(max(0.40 + 0.40 * seq_share + 0.10 * util, 0.45), 0.72)
            tars_logic_share = min(max(0.14 + 0.04 * min(max(float(switch_events) / max(float(scheduler_decisions), 1.0), 0.0), 1.0), 0.12), 0.24)
            other_logic_share = max(0.0, 1.0 - rfu_logic_share - tars_logic_share)

            feature = macros['feature_buffer']
            fragment = macros['fragment_cache']
            microop = macros['micro_op_buffer']

            feature_area = float(feature.get('area_mm2', 0.0))
            fragment_area = float(fragment.get('area_mm2', 0.0))
            microop_area = float(microop.get('area_mm2', 0.0))

            fvm_area = feature_area
            rfu_area = microop_area + logic_area * rfu_logic_share
            tars_area = fragment_area + logic_area * tars_logic_share
            other_area = logic_area * other_logic_share
            total_area = fvm_area + rfu_area + tars_area + other_area

            duration_s = max(float(makespan_ms), 1.0) / 1000.0
            fvm_access_total = max(float(fvm_reads) + float(fvm_writes), 0.0)
            feature_ratio = feature_area / max(feature_area + fragment_area, 1e-09)
            feature_reads = max(float(fvm_reads), 0.0) * feature_ratio
            feature_writes = max(float(fvm_writes), 0.0) * feature_ratio
            fragment_reads = max(float(fvm_reads), 0.0) - feature_reads
            fragment_writes = max(float(fvm_writes), 0.0) - feature_writes
            microop_reads = max(float(scheduler_decisions), 0.0)
            microop_writes = max(float(switch_events), 0.0)

            feature_dynamic = self._macro_dynamic_power_w(feature, feature_reads, feature_writes, duration_s)
            fragment_dynamic = self._macro_dynamic_power_w(fragment, fragment_reads, fragment_writes, duration_s)
            microop_dynamic = self._macro_dynamic_power_w(microop, microop_reads, microop_writes, duration_s)
            macro_leakage = max(float(feature.get('leakage_mw', 0.0)), 0.0) / 1000.0 + max(float(fragment.get('leakage_mw', 0.0)), 0.0) / 1000.0 + max(float(microop.get('leakage_mw', 0.0)), 0.0) / 1000.0

            logic_total_power = max(float(openroad.get('total_power_w', 0.0)), 0.0)
            logic_static = max(float(openroad.get('static_w', logic_total_power * 0.12)), 0.0)
            logic_dynamic = max(logic_total_power - logic_static, 0.0)

            rfu_logic_dynamic = logic_dynamic * rfu_logic_share
            scheduler_logic_dynamic = max(logic_dynamic - rfu_logic_dynamic, 0.0)

            rfu_dynamic = rfu_logic_dynamic + microop_dynamic
            scheduler_fvm_dynamic = scheduler_logic_dynamic + feature_dynamic + fragment_dynamic
            static_w = logic_static + macro_leakage
            total_power = rfu_dynamic + scheduler_fvm_dynamic + static_w

            return {
                'total_area_mm2': total_area,
                'fvm_area_mm2': fvm_area,
                'rfu_area_mm2': rfu_area,
                'tars_area_mm2': tars_area,
                'other_area_mm2': other_area,
                'total_power_w': total_power,
                'rfu_dynamic_w': rfu_dynamic,
                'scheduler_fvm_dynamic_w': scheduler_fvm_dynamic,
                'static_w': static_w,
                'logic_area_mm2': logic_area,
                'fmax_mhz': openroad.get('fmax_mhz', 0.0),
                'clock_period_ns': openroad.get('clock_period_ns', 0.0),
                'wns_ns': openroad.get('wns_ns', 0.0),
                'source': 'openroad_cacti_split',
            }

        raise FileNotFoundError(f'area/power reports not found via {self.config_path}')

    def _parse_openroad_metrics_mapping(self, payload: Dict[str, object]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        area_um2 = self._coerce_float(
            payload.get('finish__design__instance__area'),
            payload.get('finish__design__instance__area__stdcell'),
            payload.get('design__instance__area'),
        )
        util = self._coerce_float(
            payload.get('finish__design__instance__utilization'),
            payload.get('globalroute__design__instance__utilization'),
            payload.get('cts__design__instance__utilization'),
        )
        total_power_w = self._coerce_float(
            payload.get('finish__power__total'),
            payload.get('globalroute__power__total'),
            payload.get('cts__power__total'),
        )
        leakage_w = self._coerce_float(
            payload.get('finish__power__leakage__total'),
            payload.get('globalroute__power__leakage__total'),
            payload.get('cts__power__leakage__total'),
        )
        internal_w = self._coerce_float(
            payload.get('finish__power__internal__total'),
            payload.get('globalroute__power__internal__total'),
            payload.get('cts__power__internal__total'),
        )
        switching_w = self._coerce_float(
            payload.get('finish__power__switching__total'),
            payload.get('globalroute__power__switching__total'),
            payload.get('cts__power__switching__total'),
        )
        setup_ws_ns = self._coerce_float(
            payload.get('finish__timing__setup__ws'),
            payload.get('globalroute__timing__setup__ws'),
            payload.get('cts__timing__setup__ws'),
        )
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

    def _resolve_openroad_synth_stat_path(self) -> Path | None:
        candidates: list[Path] = []
        if self.config.openroad_synth_stat_report is not None:
            candidates.append(self.config.openroad_synth_stat_report)
        if self.config.openroad_area_report is not None:
            candidates.append(self.config.openroad_area_report.with_name('fuse_synth_stat.txt'))
            candidates.append(self.config.openroad_area_report.with_name('synth_stat.txt'))
        candidates.append(self.project_root / 'external' / 'OpenROAD-flow-scripts' / 'flow' / 'reports' / 'sky130hd' / 'fuse_top' / 'base' / 'synth_stat.txt')
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _macro_dynamic_power_w(self, stats: Dict[str, float], reads: float, writes: float, duration_s: float) -> float:
        read_nj = max(float(stats.get('dynamic_nj', 0.0)), 0.0)
        write_nj = max(float(stats.get('dynamic_write_nj', read_nj)), 0.0)
        total_nj = max(float(reads), 0.0) * read_nj + max(float(writes), 0.0) * write_nj
        return total_nj * 1e-09 / max(duration_s, 1e-09)

    def _infer_clock_period_ns(self, payload: Dict[str, object]) -> float | None:
        direct = self._coerce_float(payload.get('constraints__clocks__period'), payload.get('timing__clock__period'))
        if direct is not None:
            return direct
        details = payload.get('constraints__clocks__details')
        if isinstance(details, list):
            for item in details:
                if isinstance(item, str):
                    match = re.search(r':\s*([0-9.]+)', item)
                    if match:
                        return float(match.group(1))
        return None

    def _provenance_record(self, path: Path | None) -> Dict[str, object] | None:
        if path is None:
            return None
        path = Path(path)
        if not path.exists():
            return None
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        return {'path': str(path.resolve()), 'size_bytes': int(path.stat().st_size), 'sha256': digest.hexdigest()}

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
        return self.require_dram(total_bytes)

    def estimate_memory_macros(self, feature_bytes: int, fragment_bytes: int, microop_bytes: int) -> Dict[str, Dict[str, float]]:
        return self.require_memory_macros()

    def estimate_area_power(
        self,
        *,
        rfu_count: int,
        feature_bytes: int,
        fragment_bytes: int,
        microop_bytes: int,
        utilization: float,
        fvm_reads: float,
        fvm_writes: float,
        scheduler_decisions: float,
        switch_events: float,
        makespan_ms: float = 1.0,
    ) -> Dict[str, float]:
        return self.require_area_power(
            rfu_count=rfu_count,
            feature_bytes=feature_bytes,
            fragment_bytes=fragment_bytes,
            microop_bytes=microop_bytes,
            utilization=utilization,
            fvm_reads=fvm_reads,
            fvm_writes=fvm_writes,
            scheduler_decisions=scheduler_decisions,
            switch_events=switch_events,
            makespan_ms=makespan_ms,
        )

    def _match_float(self, text: str, patterns: Iterable[str], group: int = 1) -> float | None:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                try:
                    return float(match.group(group))
                except ValueError:
                    continue
        return None
