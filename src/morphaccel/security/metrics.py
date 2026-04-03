from __future__ import annotations
import math
from typing import Iterable, Sequence
import numpy as np
import pandas as pd
FEATURE_COLUMNS = ['latency_ms', 'energy_nj', 'reuse_rate', 'structural_reuse_rate', 'template_switches', 'rfu_entropy', 'template_entropy', 'dispatch_gap_mean', 'dispatch_gap_p90', 'dispatches']

def safe_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)

def pearson_corr(x: Sequence[float], y: Sequence[float]) -> float:
    xs = safe_array(x)
    ys = safe_array(y)
    if xs.size == 0 or ys.size == 0:
        return 0.0
    if np.allclose(xs.std(), 0.0) or np.allclose(ys.std(), 0.0):
        return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])

def entropy(values: Sequence[float | int | str]) -> float:
    series = pd.Series(list(values))
    probs = series.value_counts(normalize=True).values.astype(float)
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log2(np.clip(probs, 1e-12, None))).sum())

def normalized_entropy(values: Sequence[float | int | str]) -> float:
    series = pd.Series(list(values))
    k = max(1, int(series.nunique()))
    if k <= 1:
        return 0.0
    return entropy(values) / math.log2(k)

def mutual_information_binary(labels: Sequence[int], scores: Sequence[float], bins: int=8) -> float:
    ys = safe_array(labels).astype(int)
    xs = safe_array(scores)
    if ys.size == 0:
        return 0.0
    edges = np.linspace(xs.min() - 1e-09, xs.max() + 1e-09, max(2, bins + 1))
    digitized = np.digitize(xs, edges[1:-1], right=False)
    mi = 0.0
    for y in [0, 1]:
        py = max(1e-12, float((ys == y).mean()))
        for bucket in np.unique(digitized):
            px = max(1e-12, float((digitized == bucket).mean()))
            pxy = float(((ys == y) & (digitized == bucket)).mean())
            if pxy > 0.0:
                mi += pxy * math.log2(pxy / (px * py))
    return float(max(0.0, mi))

def binary_security_score(accuracy: float, mi: float, corr: float) -> float:
    leak = (max(0.5, accuracy) - 0.5) * 2.0
    mi_term = min(1.0, mi / 0.5)
    corr_term = min(1.0, abs(corr))
    return float(max(0.0, 1.0 - (0.5 * leak + 0.25 * mi_term + 0.25 * corr_term)))

def trace_features(events: pd.DataFrame, metrics: dict) -> dict:
    dispatch = events[events['event'] == 'dispatch'].copy() if not events.empty else pd.DataFrame(columns=['event', 'cycle', 'template_family', 'rfu_id', 'reused_result', 'structural_reuse'])
    if dispatch.empty:
        gaps = np.array([0.0])
        template_switches = 0.0
        rfu_h = 0.0
        template_h = 0.0
        reuse_rate = 0.0
        structural_rate = 0.0
    else:
        cycles = dispatch['cycle'].astype(float).values
        gaps = np.diff(cycles) if len(cycles) > 1 else np.array([0.0])
        template_switches = float((dispatch['template_family'].astype(str).shift(1) != dispatch['template_family'].astype(str)).fillna(False).sum()) / max(1.0, float(len(dispatch)))
        rfu_h = normalized_entropy(dispatch['rfu_id'].astype(str).tolist())
        template_h = normalized_entropy(dispatch['template_family'].astype(str).tolist())
        reuse_rate = float(dispatch['reused_result'].astype(float).mean())
        structural_rate = float(dispatch['structural_reuse'].astype(float).mean())
    return {'latency_ms': float(metrics.get('makespan_ms', 0.0)), 'energy_nj': float(metrics.get('energy_nj', 0.0)), 'reuse_rate': reuse_rate, 'structural_reuse_rate': structural_rate, 'template_switches': template_switches, 'rfu_entropy': rfu_h, 'template_entropy': template_h, 'dispatch_gap_mean': float(np.mean(gaps)), 'dispatch_gap_p90': float(np.quantile(gaps, 0.9)) if gaps.size else 0.0, 'dispatches': float(len(dispatch))}
