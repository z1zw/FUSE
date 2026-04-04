from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Sequence
from ..ir import TaskGraph
from .canonicalizer import CanonicalTaskCanonicalizer

@dataclass
class VerificationResult:
    ok: bool
    checks: List[str]
    canonical_hash: str

class FuseVerifier:

    def __init__(self):
        self.canonicalizer = CanonicalTaskCanonicalizer()

    def verify_graphs(self, graphs: Sequence[TaskGraph]) -> VerificationResult:
        checks: List[str] = []
        ok = True
        for graph in graphs:
            try:
                graph.validate()
                checks.append(f'{graph.task_id}:dag')
            except Exception as exc:
                ok = False
                checks.append(f'{graph.task_id}:invalid:{exc}')
            for frag in graph.fragments.values():
                if frag.semantic_group and '@prompt_' in frag.semantic_group:
                    ok = False
                    checks.append(f'{graph.task_id}:{frag.fragment_id}:prompt_tag')
                if any((key in frag.metadata for key in ['prompt_hash', 'prompt_text', 'prompt_variant', 'runtime_seed', 'secret_tag', 'nonce'])):
                    ok = False
                    checks.append(f'{graph.task_id}:{frag.fragment_id}:forbidden_metadata')
        canonical_hash = self.canonicalizer.canonical_hash(graphs)
        checks.append(f'canonical:{canonical_hash}')
        return VerificationResult(ok=ok, checks=checks, canonical_hash=canonical_hash)

    def equivalent(self, lhs: Sequence[TaskGraph], rhs: Sequence[TaskGraph]) -> bool:
        return self.canonicalizer.canonical_hash(lhs) == self.canonicalizer.canonical_hash(rhs)
FUSEVerifier = FuseVerifier
