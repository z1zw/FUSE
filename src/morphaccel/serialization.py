from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
import csv
import pandas as pd
from .ir import MicroFragment, RuntimeTask, TaskGraph
from .reuse import DecompositionResult, MicrofragmentDecomposer

@dataclass
class ScenarioTables:
    tasks: pd.DataFrame
    microfragments: pd.DataFrame
    representative_paths: pd.DataFrame

@dataclass
class ScenarioFiles:
    task_path: Path
    micro_path: Path
    path_path: Path

class ScenarioSerializer:

    def __init__(self, decomposer: MicrofragmentDecomposer | None=None):
        self.decomposer = decomposer or MicrofragmentDecomposer()

    def build_tables(self, task_graphs: Sequence[TaskGraph], runtime_tasks: Sequence[RuntimeTask]) -> ScenarioTables:
        decomposition = self.decomposer.decompose_tasks(task_graphs)
        return self.build_tables_from_decomposition(decomposition, runtime_tasks)

    def build_tables_from_decomposition(self, decomposition: DecompositionResult, runtime_tasks: Sequence[RuntimeTask]) -> ScenarioTables:
        tasks_rows: List[Dict] = []
        micro_rows: List[Dict] = []
        path_rows: List[Dict] = []
        by_task: Dict[str, List[MicroFragment]] = {}
        for micro in decomposition.microfragments:
            by_task.setdefault(micro.task_id, []).append(micro)
        dep_map = {micro.micro_id: list(micro.deps) for micro in decomposition.microfragments}
        for runtime_task in runtime_tasks:
            graph = runtime_task.task_graph
            tasks_rows.append({'runtime_id': runtime_task.runtime_id, 'task_id': graph.task_id, 'task_type': graph.task_type, 'domain': graph.domain, 'arrival_cycle': runtime_task.arrival_cycle})
            task_micros = by_task[graph.task_id]
            for micro in task_micros:
                runtime_micro_id = f'{runtime_task.runtime_id}:{micro.micro_id}'
                runtime_deps = [f'{runtime_task.runtime_id}:{dep}' for dep in dep_map[micro.micro_id]]
                micro_rows.append({'runtime_micro_id': runtime_micro_id, 'runtime_id': runtime_task.runtime_id, 'task_id': micro.task_id, 'task_type': micro.task_type, 'stage': micro.stage, 'parent_fragment_id': micro.parent_fragment_id, 'base_micro_id': micro.micro_id, 'op': micro.op, 'template_family': micro.template_family, 'structural_hash': micro.structural_hash, 'semantic_hash': micro.semantic_hash, 'semantic_group': micro.semantic_group or '', 'domain': micro.context.domain, 'dtype': micro.context.dtype, 'layout': micro.context.layout, 'tile': 'x'.join((str(v) for v in micro.context.tile)), 'quant': micro.context.quant, 'delta_t': micro.context.delta_t, 'estimated_ops': micro.estimated_ops, 'estimated_bytes': micro.estimated_bytes, 'deps': ';'.join(runtime_deps)})
        for path in decomposition.representative_paths.values():
            path_rows.append({'path_id': path.path_id, 'members': ';'.join(path.members), 'signature': ';'.join(path.structural_signature), 'template_families': ';'.join(path.template_families), 'reuse_count': path.reuse_count})
        return ScenarioTables(tasks=pd.DataFrame(tasks_rows), microfragments=pd.DataFrame(micro_rows), representative_paths=pd.DataFrame(path_rows))

    def write_tables(self, tables: ScenarioTables, out_dir: Path) -> ScenarioFiles:
        out_dir.mkdir(parents=True, exist_ok=True)
        task_path = out_dir / 'tasks.csv'
        micro_path = out_dir / 'microfragments.csv'
        path_path = out_dir / 'representative_paths.csv'
        tables.tasks.to_csv(task_path, index=False)
        tables.microfragments.to_csv(micro_path, index=False)
        tables.representative_paths.to_csv(path_path, index=False)
        return ScenarioFiles(task_path=task_path, micro_path=micro_path, path_path=path_path)
