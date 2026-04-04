from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple
from ..ir import MicroFragment

@dataclass
class TemplateState:
    loaded_template: Optional[str] = None
    ready_cycle: int = 0
    history: List[str] = field(default_factory=list)

class LightweightReconfigurationController:

    def __init__(self, prefetch_depth: int=4, base_switch_cycles: int=1800):
        self.prefetch_depth = prefetch_depth
        self.base_switch_cycles = base_switch_cycles
        self.prefetch_queue: Deque[str] = deque(maxlen=prefetch_depth)
        self.template_hits = 0
        self.template_misses = 0
        self.prefetch_events = 0
        self.switch_events = 0
        self.negative_cache: Dict[Tuple[str, str], int] = {}

    def switch_cost(self, current_template: Optional[str], next_template: str, prefetched: bool=False) -> int:
        if current_template == next_template:
            self.template_hits += 1
            return 0
        if prefetched or next_template in self.prefetch_queue:
            self.template_hits += 1
            self.switch_events += 1
            return max(1, self.base_switch_cycles // 10)
        self.template_misses += 1
        self.switch_events += 1
        return self.base_switch_cycles

    def prefetch(self, template_family: str) -> None:
        if template_family in self.prefetch_queue:
            return
        self.prefetch_queue.append(template_family)
        self.prefetch_events += 1

    def negative_probe(self, semantic_hash: str, until_cycle: int) -> None:
        self.negative_cache[semantic_hash, 'reuse'] = until_cycle

    def blocked(self, semantic_hash: str, cycle: int) -> bool:
        return self.negative_cache.get((semantic_hash, 'reuse'), -1) >= cycle

    def stats(self) -> Dict[str, int]:
        return {'lrc_template_hits': self.template_hits, 'lrc_template_misses': self.template_misses, 'lrc_switch_events': self.switch_events, 'lrc_prefetch_events': self.prefetch_events}
