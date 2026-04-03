from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from ..ir import MicroFragment

@dataclass
class RFU:
    rfu_id: int
    supported_templates: List[str]
    busy_until: int = 0
    loaded_template: Optional[str] = None
    active_micro_id: Optional[str] = None
    executed_cycles: int = 0
    idle_cycles: int = 0
    completed: List[str] = field(default_factory=list)

    def is_free(self, cycle: int) -> bool:
        return cycle >= self.busy_until

    def assign(self, micro: MicroFragment, start_cycle: int, duration: int) -> int:
        self.active_micro_id = micro.micro_id
        self.loaded_template = micro.template_family
        self.busy_until = start_cycle + duration
        self.executed_cycles += duration
        self.completed.append(micro.micro_id)
        return self.busy_until

class RFUCluster:

    def __init__(self, count: int=16):
        families = [['linear_map', 'reduction', 'nonlinear_lookup', 'permutation', 'generic'] for _ in range(count)]
        self.rfus = [RFU(rfu_id=i, supported_templates=families[i]) for i in range(count)]

    def free_rfus(self, cycle: int) -> List[RFU]:
        free = []
        for rfu in self.rfus:
            if rfu.is_free(cycle):
                free.append(rfu)
            else:
                rfu.idle_cycles += 0
        return free

    def utilization(self, makespan: int) -> float:
        total_exec = sum((rfu.executed_cycles for rfu in self.rfus))
        denom = max(1, makespan * len(self.rfus))
        return total_exec / denom

    def stats(self, makespan: int) -> Dict[str, float]:
        return {'rfu_count': len(self.rfus), 'rfu_utilization': self.utilization(makespan), 'rfu_executed_cycles': sum((r.executed_cycles for r in self.rfus))}
