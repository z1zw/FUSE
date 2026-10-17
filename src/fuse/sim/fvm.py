from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
from ..ir import MicroFragment

@dataclass
class CacheEntry:
    key: str
    created_cycle: int
    last_access_cycle: int
    bytes_used: int
    producer: str
    data_id: str
    task_type: str
    context_repr: str
    hits: int = 0

class FeatureVectorManager:

    def __init__(self, capacity_bytes: int=8 * 1024 * 1024):
        self.capacity_bytes = capacity_bytes
        self.entries: Dict[str, CacheEntry] = {}
        self.eviction_queue: Deque[str] = deque()
        self.current_bytes = 0
        self.peak_bytes = 0
        self.total_reads = 0
        self.total_writes = 0
        self.total_redundant_reads = 0
        self.allocation_events = 0
        self.free_events = 0
        self.residency_samples: List[int] = []

    def peek(self, semantic_hash: str, cycle: int, freshness_window: int, data_id: str='') -> Optional[CacheEntry]:
        entry = self.entries.get(semantic_hash)
        if entry is None:
            return None
        if not self._valid(entry, data_id):
            return None
        if freshness_window >= 0 and cycle - entry.created_cycle > freshness_window:
            return None
        return entry

    def _valid(self, entry: CacheEntry, data_id: str) -> bool:
        """C4: a cached result is returned only to the operands that
        produced it. The identity is the originating request, which the
        request compiler qualifies per prompt; an empty identity is
        unconstrained, for callers that supply none."""
        if not data_id or not entry.data_id:
            return True
        return entry.data_id == data_id

    def lookup(self, semantic_hash: str, cycle: int, freshness_window: int, data_id: str='') -> Optional[CacheEntry]:
        entry = self.entries.get(semantic_hash)
        self.total_reads += 1
        self.residency_samples.append(self.current_bytes)
        if entry is None:
            self.total_redundant_reads += 1
            return None
        if not self._valid(entry, data_id):
            self.total_redundant_reads += 1
            return None
        if freshness_window >= 0 and cycle - entry.created_cycle > freshness_window:
            self.total_redundant_reads += 1
            return None
        entry.last_access_cycle = cycle
        entry.hits += 1
        return entry

    def write(self, micro: MicroFragment, cycle: int) -> None:
        bytes_used = max(32, micro.estimated_bytes or micro.estimated_ops * 4)
        self._ensure_capacity(bytes_used)
        entry = CacheEntry(key=micro.semantic_hash, created_cycle=cycle, last_access_cycle=cycle, bytes_used=bytes_used, producer=micro.micro_id, data_id=micro.task_id, task_type=micro.task_type, context_repr=str(micro.context))
        if micro.semantic_hash not in self.entries:
            self.eviction_queue.append(micro.semantic_hash)
            self.current_bytes += bytes_used
            self.allocation_events += 1
        else:
            prev = self.entries[micro.semantic_hash]
            self.current_bytes += bytes_used - prev.bytes_used
        self.entries[micro.semantic_hash] = entry
        self.total_writes += 1
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)
        self.residency_samples.append(self.current_bytes)

    def _ensure_capacity(self, incoming_bytes: int) -> None:
        while self.current_bytes + incoming_bytes > self.capacity_bytes and self.eviction_queue:
            key = self.eviction_queue.popleft()
            entry = self.entries.pop(key, None)
            if entry is None:
                continue
            self.current_bytes -= entry.bytes_used
            self.free_events += 1

    def stats(self) -> Dict[str, float]:
        residency_avg = float(sum(self.residency_samples) / len(self.residency_samples)) if self.residency_samples else 0.0
        return {'fvm_peak_bytes': self.peak_bytes, 'fvm_avg_residency_bytes': residency_avg, 'fvm_reads': self.total_reads, 'fvm_writes': self.total_writes, 'fvm_redundant_reads': self.total_redundant_reads, 'fvm_allocations': self.allocation_events, 'fvm_frees': self.free_events}
