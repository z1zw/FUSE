from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple
from ..ir import MicroFragment, RuntimeTask

@dataclass
class RuntimeMicro:
    runtime_micro_id: str
    task_runtime_id: str
    base_micro: MicroFragment
    deps: Set[str] = field(default_factory=set)
    succs: Set[str] = field(default_factory=set)
    arrival_cycle: int = 0
    ready_cycle: Optional[int] = None
    start_cycle: Optional[int] = None
    finish_cycle: Optional[int] = None
    reused_result: bool = False
    structural_reuse: bool = False
    template_prefetched: bool = False

class TaskAwareRuntimeScheduler:

    def __init__(self, starvation_window: int=128):
        self.starvation_window = starvation_window
        self.ready_sets: Dict[str, Deque[str]] = defaultdict(deque)
        self.runtime_micros: Dict[str, RuntimeMicro] = {}
        self.running: List[Tuple[int, str, int]] = []
        self.arrival_order = 0
        self.prefetch_earliness_cycles: List[int] = []
        self.pipeline_stall_cycles = 0
        self.schedule_decisions = 0
        self.round_robin_task_types: Deque[str] = deque()
        self.task_type_last_cycle: Dict[str, int] = defaultdict(int)
        self.ready_probe_count = 0
        self.ready_match_stats = {'kappa_key': 0, 'template_hit': 0, 'valid_fragment': 0}

    def install_runtime_task(self, runtime_task: RuntimeTask, micros: Iterable[MicroFragment], global_deps: Dict[str, List[str]]) -> None:
        task_types_seen = set(self.round_robin_task_types)
        if runtime_task.task_graph.task_type not in task_types_seen:
            self.round_robin_task_types.append(runtime_task.task_graph.task_type)
        for micro in micros:
            runtime_id = f'{runtime_task.runtime_id}:{micro.micro_id}'
            deps = {f'{runtime_task.runtime_id}:{dep}' for dep in global_deps[micro.micro_id]}
            node = RuntimeMicro(runtime_micro_id=runtime_id, task_runtime_id=runtime_task.runtime_id, base_micro=micro, deps=deps, arrival_cycle=runtime_task.arrival_cycle)
            self.runtime_micros[runtime_id] = node
        for runtime_id, node in self.runtime_micros.items():
            if node.task_runtime_id != runtime_task.runtime_id:
                continue
            for dep in list(node.deps):
                if dep in self.runtime_micros:
                    self.runtime_micros[dep].succs.add(runtime_id)
        for node in self.runtime_micros.values():
            if node.task_runtime_id == runtime_task.runtime_id and (not node.deps):
                self.mark_ready(node.runtime_micro_id, runtime_task.arrival_cycle)

    def mark_ready(self, runtime_micro_id: str, cycle: int) -> None:
        node = self.runtime_micros[runtime_micro_id]
        if node.ready_cycle is not None:
            return
        node.ready_cycle = cycle
        self.ready_sets[node.base_micro.task_type].append(runtime_micro_id)

    def pop_ready(self, cycle: int, score_fn) -> Optional[str]:
        if not any(self.ready_sets.values()):
            return None
        for _ in range(max(1, len(self.round_robin_task_types))):
            task_type = self.round_robin_task_types[0]
            self.round_robin_task_types.rotate(-1)
            queue = self.ready_sets[task_type]
            if not queue:
                continue
            candidates = list(queue)
            best = max(candidates, key=lambda rid: score_fn(self.runtime_micros[rid], cycle))
            queue.remove(best)
            self.schedule_decisions += 1
            self.task_type_last_cycle[task_type] = cycle
            return best
        oldest = None
        oldest_cycle = None
        oldest_queue = None
        for task_type, queue in self.ready_sets.items():
            for rid in queue:
                rc = self.runtime_micros[rid].ready_cycle or cycle
                if oldest_cycle is None or rc < oldest_cycle:
                    oldest = rid
                    oldest_cycle = rc
                    oldest_queue = queue
        if oldest is not None and oldest_queue is not None:
            oldest_queue.remove(oldest)
            self.schedule_decisions += 1
        return oldest

    def finish(self, runtime_micro_id: str, cycle: int) -> None:
        node = self.runtime_micros[runtime_micro_id]
        node.finish_cycle = cycle
        for succ_id in node.succs:
            succ = self.runtime_micros[succ_id]
            succ.deps.discard(runtime_micro_id)
            if not succ.deps:
                self.mark_ready(succ_id, cycle)

    def outstanding(self) -> bool:
        return any((node.finish_cycle is None for node in self.runtime_micros.values()))

    def next_finish(self) -> Optional[int]:
        active = [node.finish_cycle for node in self.runtime_micros.values() if node.finish_cycle is not None]
        return min(active) if active else None

    def completed_for_task(self, runtime_id: str) -> bool:
        nodes = [n for n in self.runtime_micros.values() if n.task_runtime_id == runtime_id]
        return bool(nodes) and all((n.finish_cycle is not None for n in nodes))

    def task_finish_cycle(self, runtime_id: str) -> int:
        nodes = [n.finish_cycle for n in self.runtime_micros.values() if n.task_runtime_id == runtime_id]
        return max((c for c in nodes if c is not None))

    def stats(self) -> Dict[str, float]:
        probe_den = float(self.ready_probe_count) if self.ready_probe_count else 1.0
        return {'scheduler_decisions': self.schedule_decisions, 'scheduler_prefetch_mean_earliness_cycles': sum(self.prefetch_earliness_cycles) / len(self.prefetch_earliness_cycles) if self.prefetch_earliness_cycles else 0.0, 'scheduler_pipeline_stall_cycles': self.pipeline_stall_cycles, 'kappa_key_match_ratio': self.ready_match_stats['kappa_key'] / probe_den, 'template_hit_ratio': self.ready_match_stats['template_hit'] / probe_den, 'valid_fragment_ratio': self.ready_match_stats['valid_fragment'] / probe_den}
