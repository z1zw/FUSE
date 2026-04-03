from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence
import json
import shutil
import subprocess
import sys
import tempfile
import pandas as pd
from .branding import CPP_BINARY_STEM, normalize_mode
from .ir import RunResult, RuntimeTask, TaskGraph
from .serialization import ScenarioSerializer

@dataclass
class CppRunArtifacts:
    metrics: Dict[str, float]
    events: pd.DataFrame
    files: Dict[str, Path]

class CppSimulatorBridge:

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.cpp_dir = self.project_root / 'cpp'
        self.build_dir = self.cpp_dir / 'build'
        binary_name = f'{CPP_BINARY_STEM}.exe' if sys.platform.startswith('win') else CPP_BINARY_STEM
        self.binary = self.build_dir / binary_name
        self.serializer = ScenarioSerializer()

    def build(self, force: bool=False) -> Path:
        sources = [self.cpp_dir / 'src' / 'csv.cpp', self.cpp_dir / 'src' / 'engine.cpp', self.cpp_dir / 'src' / 'models.cpp', self.cpp_dir / 'src' / 'main.cpp']
        self.build_dir.mkdir(parents=True, exist_ok=True)
        if self.binary.exists() and (not force):
            binary_mtime = self.binary.stat().st_mtime
            newest_source = max((source.stat().st_mtime for source in sources))
            if binary_mtime >= newest_source:
                return self.binary
        cmd = [shutil.which('g++') or 'g++', '-std=c++17', '-O3', '-I', str(self.cpp_dir / 'include'), *[str(s) for s in sources], '-o', str(self.binary)]
        subprocess.run(cmd, check=True)
        return self.binary

    def run(self, task_graphs: Sequence[TaskGraph], runtime_tasks: Sequence[RuntimeTask], *, mode: str='morphaccel', scenario_name: str='scenario', out_dir: Path | None=None) -> CppRunArtifacts:
        mode = normalize_mode(mode)
        binary = self.build()
        owned_tmp = None
        scenario_dir = out_dir
        if scenario_dir is None:
            owned_tmp = tempfile.TemporaryDirectory(prefix='fuse_cpp_')
            scenario_dir = Path(owned_tmp.name)
        else:
            scenario_dir.mkdir(parents=True, exist_ok=True)
        scenario_subdir = scenario_dir / scenario_name
        tables = self.serializer.build_tables(task_graphs, runtime_tasks)
        files = self.serializer.write_tables(tables, scenario_subdir)
        metrics_path = scenario_subdir / f'{mode}_metrics.csv'
        events_path = scenario_subdir / f'{mode}_events.csv'
        cmd = [str(binary), '--tasks', str(files.task_path), '--microfragments', str(files.micro_path), '--paths', str(files.path_path), '--mode', mode, '--metrics-out', str(metrics_path), '--events-out', str(events_path)]
        subprocess.run(cmd, check=True)
        metrics_df = pd.read_csv(metrics_path)
        metrics = {str(k): self._coerce(v) for k, v in metrics_df.itertuples(index=False, name=None)}
        events = pd.read_csv(events_path) if events_path.exists() and events_path.stat().st_size else pd.DataFrame()
        artifact = CppRunArtifacts(metrics=metrics, events=events, files={'tasks': files.task_path, 'microfragments': files.micro_path, 'paths': files.path_path, 'metrics': metrics_path, 'events': events_path})
        if owned_tmp is not None:
            owned_tmp.cleanup()
        return artifact

    def run_result(self, task_graphs: Sequence[TaskGraph], runtime_tasks: Sequence[RuntimeTask], *, mode: str='morphaccel', scenario_name: str='scenario', out_dir: Path | None=None) -> RunResult:
        artifact = self.run(task_graphs, runtime_tasks, mode=mode, scenario_name=scenario_name, out_dir=out_dir)
        traces = {'events': artifact.events.to_dict(orient='records')}
        return RunResult(metrics=artifact.metrics, traces=traces, tables={})

    def _coerce(self, value):
        if isinstance(value, str):
            try:
                if '.' in value or 'e' in value.lower():
                    return float(value)
                return int(value)
            except Exception:
                return value
        return value
