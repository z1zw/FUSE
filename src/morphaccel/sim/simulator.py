from __future__ import annotations
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from ..branding import normalize_mode
from ..ir import MicroFragment, RunResult, RuntimeTask
from ..reuse import estimate_latency_cycles
from .controller import LightweightReconfigurationController
from .fvm import FeatureVectorManager
from .rfu import RFUCluster
from .scheduler import RuntimeMicro, TaskAwareRuntimeScheduler
CYCLES_PER_MS = 1000000

@dataclass
class SimulationConfig:
    rfu_count: int = 16
    fvm_capacity_bytes: int = 8 * 1024 * 1024
    fragment_cache_bytes: int = 2 * 1024 * 1024
    micro_op_buffer_bytes: int = 512 * 1024
    prefetch_depth: int = 8
    base_switch_cycles: int = 2300
    freshness_window_cycles: int = 5000000
    memory_bandwidth_bytes_per_cycle: int = 64
    mode: str = 'morphaccel'
    task_specific_speedup: Dict[str, float] = field(default_factory=lambda: {'ZKP': 1.0, 'CNN': 1.0, 'AES': 1.0})

class MorphAccelSimulator:

    def __init__(self, config: SimulationConfig | None=None):
        self.config = config or SimulationConfig()
        self.config.mode = normalize_mode(self.config.mode)
        self.rfus = RFUCluster(count=self.config.rfu_count)
        self.fvm = FeatureVectorManager(capacity_bytes=self.config.fvm_capacity_bytes)
        self.lrc = LightweightReconfigurationController(prefetch_depth=self.config.prefetch_depth, base_switch_cycles=self.config.base_switch_cycles)
        self.scheduler = TaskAwareRuntimeScheduler()
        self.trace_events: List[Dict[str, Any]] = []

    def _reuse_enabled(self) -> bool:
        return self.config.mode in {'morphaccel', 'secure_ct'} or self.config.mode.startswith('secure_elastic')

    def _baseline_speed_factor(self, task_type: str) -> float:
        factors = {'static_zkp': {'ZKP': 1.05, 'CNN': 0.92, 'AES': 0.92}, 'eyeriss': {'ZKP': 0.9, 'CNN': 1.08, 'AES': 0.9}, 'openaes': {'ZKP': 0.94, 'CNN': 0.94, 'AES': 1.03}, 'gpu_tensor': {'ZKP': 0.96, 'CNN': 1.12, 'AES': 0.97}, 'multicore': {'ZKP': 0.98, 'CNN': 0.98, 'AES': 0.98}}
        return factors.get(self.config.mode, {}).get(task_type, 1.0)

    def run(self, runtime_tasks: Sequence[RuntimeTask], microfragments: Sequence[MicroFragment]) -> RunResult:
        micro_by_task = defaultdict(list)
        dep_map = {micro.micro_id: list(micro.deps) for micro in microfragments}
        for micro in microfragments:
            micro_by_task[micro.task_id].append(micro)
        arrivals = sorted(runtime_tasks, key=lambda t: t.arrival_cycle)
        arrival_index = 0
        clock = 0
        running_heap: List[Tuple[int, str, int]] = []
        completed_tasks: Dict[str, int] = {}
        while arrival_index < len(arrivals) or running_heap or self._has_ready_items():
            while arrival_index < len(arrivals) and arrivals[arrival_index].arrival_cycle <= clock:
                runtime_task = arrivals[arrival_index]
                micros = micro_by_task[runtime_task.task_graph.task_id]
                self.scheduler.install_runtime_task(runtime_task, micros, dep_map)
                arrival_index += 1
            free_rfus = self.rfus.free_rfus(clock)
            while free_rfus and self._has_ready_items():
                runtime_micro_id = self.scheduler.pop_ready(clock, lambda node, cyc: self._score(node, cyc))
                if runtime_micro_id is None:
                    break
                node = self.scheduler.runtime_micros[runtime_micro_id]
                rfu = self._choose_rfu(free_rfus, node.base_micro)
                free_rfus.remove(rfu)
                duration, meta = self._execution_duration(node, rfu, clock)
                node.start_cycle = clock
                node.reused_result = meta['reused_result']
                node.structural_reuse = meta['structural_reuse']
                node.template_prefetched = meta['template_prefetched']
                finish = rfu.assign(node.base_micro, clock, duration)
                heappush(running_heap, (finish, runtime_micro_id, rfu.rfu_id))
                self.trace_events.append({'event': 'dispatch', 'cycle': clock, 'finish_cycle': finish, 'runtime_micro_id': runtime_micro_id, 'task_runtime_id': node.task_runtime_id, 'task_type': node.base_micro.task_type, 'template_family': node.base_micro.template_family, 'reused_result': meta['reused_result'], 'structural_reuse': meta['structural_reuse'], 'switch_cycles': meta['switch_cycles'], 'rfu_id': rfu.rfu_id})
                if node.template_prefetched:
                    self.scheduler.prefetch_earliness_cycles.append(self.config.base_switch_cycles)
            if running_heap:
                next_finish, runtime_micro_id, rfu_id = heappop(running_heap)
                clock = max(clock, next_finish)
                simultaneous = [(next_finish, runtime_micro_id, rfu_id)]
                while running_heap and running_heap[0][0] == next_finish:
                    simultaneous.append(heappop(running_heap))
                for finish_cycle, runtime_micro_id, rfu_id in simultaneous:
                    node = self.scheduler.runtime_micros[runtime_micro_id]
                    self.scheduler.finish(runtime_micro_id, finish_cycle)
                    self.fvm.write(node.base_micro, finish_cycle)
                    for succ_id in node.succs:
                        succ = self.scheduler.runtime_micros[succ_id]
                        self.lrc.prefetch(succ.base_micro.template_family)
                    if self.scheduler.completed_for_task(node.task_runtime_id):
                        completed_tasks[node.task_runtime_id] = self.scheduler.task_finish_cycle(node.task_runtime_id)
                    self.trace_events.append({'event': 'complete', 'cycle': finish_cycle, 'runtime_micro_id': runtime_micro_id, 'task_runtime_id': node.task_runtime_id, 'task_type': node.base_micro.task_type, 'rfu_id': rfu_id})
            else:
                next_arrival = arrivals[arrival_index].arrival_cycle if arrival_index < len(arrivals) else None
                if next_arrival is None:
                    break
                if next_arrival > clock and (not self._has_ready_items()):
                    self.scheduler.pipeline_stall_cycles += next_arrival - clock
                clock = max(clock, next_arrival)
        makespan = max(completed_tasks.values()) if completed_tasks else 0
        metrics = self._collect_metrics(runtime_tasks, completed_tasks, makespan)
        traces = {'events': self.trace_events, 'completed_tasks': completed_tasks}
        return RunResult(metrics=metrics, traces=traces)

    def _has_ready_items(self) -> bool:
        return any(self.scheduler.ready_sets.values())

    def _choose_rfu(self, free_rfus, micro: MicroFragment):
        exact = [rfu for rfu in free_rfus if rfu.loaded_template == micro.template_family]
        if exact:
            return exact[0]
        supported = [rfu for rfu in free_rfus if micro.template_family in rfu.supported_templates]
        if supported:
            return supported[0]
        return free_rfus[0]

    def _score(self, node: RuntimeMicro, cycle: int) -> float:
        micro = node.base_micro
        self.scheduler.ready_probe_count += 1
        score = 0.0
        if self.fvm.peek(micro.semantic_hash, cycle, self.config.freshness_window_cycles) is not None:
            score += 12.0
            self.scheduler.ready_match_stats['valid_fragment'] += 1
        if micro.template_family in self.lrc.prefetch_queue:
            score += 5.0
            self.scheduler.ready_match_stats['template_hit'] += 1
        score += 1.5 * max(0, cycle - (node.ready_cycle or cycle)) / max(1, self.scheduler.starvation_window)
        if micro.context.layout in {'tile', 'block', 'state4x4', 'nchw'}:
            score += 1.0
            self.scheduler.ready_match_stats['kappa_key'] += 1
        return score

    def _execution_duration(self, node: RuntimeMicro, rfu, cycle: int) -> Tuple[int, Dict[str, bool]]:
        micro = node.base_micro
        secure_elastic_modes = {'secure_elastic', 'secure_elastic_no_bucket', 'secure_elastic_no_pinning', 'secure_elastic_no_mask', 'secure_elastic_no_epoch', 'secure_elastic_no_prefetch'}
        base_cycles = estimate_latency_cycles(micro)
        mem_cycles = max(1, micro.estimated_bytes // max(1, self.config.memory_bandwidth_bytes_per_cycle)) if micro.estimated_bytes else 0
        switch_cycles = 0
        reused_result = False
        structural_reuse = False
        template_prefetched = micro.template_family in self.lrc.prefetch_queue and self.config.mode != 'secure_elastic_no_prefetch'
        cache_entry = None
        if self.config.mode != 'degraded' and (not self.lrc.blocked(micro.semantic_hash, cycle)):
            cache_entry = self.fvm.lookup(micro.semantic_hash, cycle, self.config.freshness_window_cycles)
        allow_result_reuse = self._reuse_enabled()
        if self.config.mode == 'secure_ct' and rfu.loaded_template != micro.template_family:
            allow_result_reuse = False
        if cache_entry is not None and allow_result_reuse:
            reused_result = self.config.mode != 'secure_ct'
            if self.config.mode == 'morphaccel':
                base_cycles = 1
                mem_cycles = 1
            elif self.config.mode == 'secure_ct':
                base_cycles = max(2, base_cycles // 2)
                mem_cycles = max(2, mem_cycles // 2)
            elif self.config.mode == 'secure_elastic_no_mask':
                base_cycles = max(1, base_cycles // 2)
                mem_cycles = max(1, mem_cycles // 2)
            else:
                base_cycles = max(2, base_cycles // 3 + 1)
                mem_cycles = max(2, mem_cycles // 3 + 1)
        else:
            self.lrc.negative_probe(micro.semantic_hash, cycle + max(1, self.config.base_switch_cycles // 2))
        if self.config.mode == 'degraded':
            switch_cycles = self.config.base_switch_cycles
        elif self.config.mode == 'task_specific':
            switch_cycles = 0
            factor = self.config.task_specific_speedup.get(micro.task_type, 1.0)
            base_cycles = max(1, int(base_cycles / max(factor, 1e-06)))
        elif self.config.mode == 'multicore':
            factor = self._baseline_speed_factor(micro.task_type)
            base_cycles = max(1, int(base_cycles / max(factor, 1e-06)))
            mem_cycles += 3
            switch_cycles = 0 if rfu.loaded_template == micro.template_family else max(1, self.config.base_switch_cycles // 3)
        elif self.config.mode == 'gpu_tensor':
            factor = self._baseline_speed_factor(micro.task_type)
            base_cycles = max(1, int(base_cycles / max(factor, 1e-06)))
            mem_cycles += 6
            switch_cycles = 0 if rfu.loaded_template == micro.template_family else max(1, self.config.base_switch_cycles // 2)
        elif self.config.mode in {'static_zkp', 'eyeriss', 'openaes'}:
            factor = self._baseline_speed_factor(micro.task_type)
            base_cycles = max(1, int(base_cycles / max(factor, 1e-06)))
            if micro.task_type == 'CNN' and self.config.mode == 'eyeriss':
                mem_cycles = max(1, mem_cycles - 1)
            switch_cycles = 0 if rfu.loaded_template == micro.template_family else max(1, self.config.base_switch_cycles // 8)
        else:
            switch_cycles = self.lrc.switch_cost(rfu.loaded_template, micro.template_family, prefetched=template_prefetched)
            if switch_cycles == 0 and rfu.loaded_template == micro.template_family:
                structural_reuse = True
        if self.config.mode in secure_elastic_modes:
            switch_cycles = max(1, switch_cycles // 2) if switch_cycles else 0
            if self.config.mode == 'secure_elastic_no_pinning':
                switch_cycles += max(1, self.config.base_switch_cycles // 5)
            if self.config.mode == 'secure_elastic_no_bucket':
                mem_cycles += 2
            if self.config.mode == 'secure_elastic_no_epoch':
                mem_cycles += 1
            if reused_result and self.config.mode == 'secure_elastic_no_mask':
                structural_reuse = True
        if self.config.mode == 'secure_ct':
            base_cycles = max(base_cycles, estimate_latency_cycles(micro) + 2)
            mem_cycles = max(mem_cycles, 2)
        elif self.config.mode in secure_elastic_modes:
            base_cycles = max(base_cycles, max(2, estimate_latency_cycles(micro) // 2))
            mem_cycles = max(mem_cycles, 2)
        total = max(1, base_cycles + mem_cycles + switch_cycles)
        return (total, {'reused_result': reused_result, 'structural_reuse': structural_reuse, 'template_prefetched': template_prefetched, 'switch_cycles': switch_cycles})

    def _collect_metrics(self, runtime_tasks, completed_tasks, makespan: int) -> Dict[str, Any]:
        per_task_latencies_ms = {}
        task_type_latencies: Dict[str, List[float]] = defaultdict(list)
        for runtime_task in runtime_tasks:
            finish = completed_tasks.get(runtime_task.runtime_id, runtime_task.arrival_cycle)
            latency_ms = (finish - runtime_task.arrival_cycle) / CYCLES_PER_MS
            per_task_latencies_ms[runtime_task.runtime_id] = latency_ms
            task_type_latencies[runtime_task.task_graph.task_type].append(latency_ms)
        task_type_mean_latency_ms = {task_type: sum(values) / len(values) if values else 0.0 for task_type, values in task_type_latencies.items()}
        events = self.trace_events
        result_reuse = sum((1 for e in events if e['event'] == 'dispatch' and e['reused_result']))
        structural_reuse = sum((1 for e in events if e['event'] == 'dispatch' and e['structural_reuse']))
        dispatches = sum((1 for e in events if e['event'] == 'dispatch'))
        switch_cycles_total = sum((int(e.get('switch_cycles', 0)) for e in events if e['event'] == 'dispatch'))
        utilization = self.rfus.utilization(makespan)
        energy = self._estimate_energy(makespan)
        metrics = {'mode': self.config.mode, 'makespan_cycles': makespan, 'makespan_ms': makespan / CYCLES_PER_MS, 'dispatches': dispatches, 'result_reuse_hits': result_reuse, 'structural_reuse_hits': structural_reuse, 'result_reuse_rate': result_reuse / dispatches if dispatches else 0.0, 'structural_reuse_rate': structural_reuse / dispatches if dispatches else 0.0, 'switch_cycles_total': switch_cycles_total, 'task_latency_ms': per_task_latencies_ms, 'task_type_mean_latency_ms': task_type_mean_latency_ms, 'rfu_utilization': utilization, 'energy_nj': energy}
        metrics.update(self.fvm.stats())
        metrics.update(self.lrc.stats())
        metrics.update(self.scheduler.stats())
        return metrics

    def _estimate_energy(self, makespan: int) -> float:
        compute_coeff = 0.65
        fvm_read_coeff = 0.12
        fvm_write_coeff = 0.15
        scheduling_coeff = 0.08
        switch_coeff = 0.25
        static_coeff = 0.02
        if self.config.mode == 'gpu_tensor':
            compute_coeff = 0.92
            fvm_read_coeff = 0.18
            fvm_write_coeff = 0.2
            scheduling_coeff = 0.1
            switch_coeff = 0.34
            static_coeff = 0.032
        elif self.config.mode == 'multicore':
            compute_coeff = 0.72
            fvm_read_coeff = 0.14
            fvm_write_coeff = 0.17
            switch_coeff = 0.28
            static_coeff = 0.028
        elif self.config.mode in {'static_zkp', 'eyeriss', 'openaes'}:
            compute_coeff = 0.6
            fvm_read_coeff = 0.1
            fvm_write_coeff = 0.12
            static_coeff = 0.018
        compute_nj = self.rfus.stats(makespan)['rfu_executed_cycles'] * compute_coeff
        fvm_nj = self.fvm.total_reads * fvm_read_coeff + self.fvm.total_writes * fvm_write_coeff
        scheduling_nj = self.scheduler.schedule_decisions * scheduling_coeff + self.lrc.switch_events * switch_coeff
        static_nj = makespan * static_coeff * self.config.rfu_count
        return compute_nj + fvm_nj + scheduling_nj + static_nj
FUSESimulator = MorphAccelSimulator
