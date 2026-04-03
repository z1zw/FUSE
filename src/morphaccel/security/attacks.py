from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np
import pandas as pd
from .metrics import FEATURE_COLUMNS

@dataclass
class AttackResult:
    attack: str
    accuracy: float

class ThresholdAttack:

    def fit_predict(self, values: Sequence[float], labels: Sequence[int]) -> float:
        xs = np.asarray(list(values), dtype=float)
        ys = np.asarray(list(labels), dtype=int)
        if xs.size <= 1:
            return 0.5
        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]
        split = max(1, xs.size // 2)
        train_x = xs[:split]
        train_y = ys[:split]
        test_x = xs[split:]
        test_y = ys[split:]
        if test_x.size == 0:
            test_x = train_x
            test_y = train_y
        candidates = np.unique(train_x)
        best_acc = 0.5
        best_thr = float(candidates[0])
        best_sign = 1.0
        for thr in candidates:
            for sign in [1.0, -1.0]:
                pred = (sign * train_x >= sign * thr).astype(int)
                acc = float((pred == train_y).mean())
                if acc >= best_acc:
                    best_acc = acc
                    best_thr = float(thr)
                    best_sign = sign
        pred = (best_sign * test_x >= best_sign * best_thr).astype(int)
        return float((pred == test_y).mean())

class NearestCentroidAttack:

    def fit_predict(self, frame: pd.DataFrame, label_col: str, feature_cols: Sequence[str]) -> float:
        if frame.empty:
            return 0.5
        data = frame[list(feature_cols)].to_numpy(dtype=float)
        labels = frame[label_col].astype(int).to_numpy()
        split = max(1, len(frame) // 2)
        train_x = data[:split]
        train_y = labels[:split]
        test_x = data[split:] if split < len(frame) else data
        test_y = labels[split:] if split < len(frame) else labels
        mean = train_x.mean(axis=0, keepdims=True)
        std = train_x.std(axis=0, keepdims=True)
        std[std < 1e-09] = 1.0
        train_z = (train_x - mean) / std
        test_z = (test_x - mean) / std
        centroids = {}
        for label in sorted(set(train_y.tolist())):
            centroids[label] = train_z[train_y == label].mean(axis=0)
        preds = []
        for row in test_z:
            label = min(centroids.keys(), key=lambda key: float(np.linalg.norm(row - centroids[key])))
            preds.append(label)
        return float((np.asarray(preds, dtype=int) == test_y).mean())

def run_binary_attacks(frame: pd.DataFrame, label_col: str='label') -> list[AttackResult]:
    if frame.empty:
        return [AttackResult('latency', 0.5), AttackResult('reuse', 0.5), AttackResult('schedule', 0.5)]
    threshold = ThresholdAttack().fit_predict(frame['latency_ms'], frame[label_col])
    reuse_cols = ['reuse_rate', 'structural_reuse_rate', 'energy_nj', 'dispatches']
    schedule_cols = ['template_switches', 'rfu_entropy', 'template_entropy', 'dispatch_gap_mean', 'dispatch_gap_p90']
    centroid = NearestCentroidAttack()
    reuse_acc = centroid.fit_predict(frame, label_col, reuse_cols)
    schedule_acc = centroid.fit_predict(frame, label_col, schedule_cols)
    return [AttackResult('latency', threshold), AttackResult('reuse', reuse_acc), AttackResult('schedule', schedule_acc)]
