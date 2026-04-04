from __future__ import annotations
from dataclasses import replace
import hashlib
import json
import re
from typing import Dict, Iterable, List, Sequence
from ..ir import Fragment, TaskGraph
from ..normalization import Normalizer

class CanonicalTaskCanonicalizer:

    def __init__(self):
        self.normalizer = Normalizer()

    def canonicalize_graph(self, graph: TaskGraph, task_id: str | None=None) -> TaskGraph:
        graph.validate()
        ordering = sorted(graph.topological_fragments(), key=lambda frag: (frag.stage, frag.template_family, self.normalizer.structural_hash(self._scrub_fragment(frag)), frag.fragment_id))
        mapping = {frag.fragment_id: f'f{idx:03d}' for idx, frag in enumerate(ordering)}
        clone = TaskGraph(task_id or graph.task_id, graph.task_type, graph.domain, self._scrub_metadata(graph.metadata))
        for frag in ordering:
            new_frag = replace(frag, fragment_id=mapping[frag.fragment_id], deps=sorted((mapping[dep] for dep in frag.deps)), semantic_group=self._canonical_group(frag.semantic_group), metadata=self._scrub_metadata(frag.metadata), expr=self.normalizer.normalize_expr(frag.expr, structural=False))
            clone.add_fragment(new_frag)
        clone.set_outputs([mapping.get(out, out) for out in graph.outputs])
        return clone

    def canonicalize_graphs(self, graphs: Sequence[TaskGraph]) -> List[TaskGraph]:
        return [self.canonicalize_graph(graph, task_id=graph.task_id) for graph in graphs]

    def canonical_hash(self, graphs: Sequence[TaskGraph]) -> str:
        payload = []
        for graph in self.canonicalize_graphs(graphs):
            signature = self.graph_signature(graph)
            signature.pop('task_id', None)
            payload.append(signature)
        encoded = json.dumps(sorted(payload, key=lambda item: (item['task_type'], item['domain'], len(item['fragments']))), sort_keys=True)
        return hashlib.sha256(encoded.encode()).hexdigest()[:20]

    def graph_signature(self, graph: TaskGraph) -> Dict[str, object]:
        fragments = []
        for frag in graph.topological_fragments():
            scrubbed = self._scrub_fragment(frag)
            fragments.append({'fragment_id': scrubbed.fragment_id, 'stage': scrubbed.stage, 'task_type': scrubbed.task_type, 'template_family': scrubbed.template_family, 'semantic_group': scrubbed.semantic_group, 'deps': sorted(scrubbed.deps), 'structural_hash': self.normalizer.structural_hash(scrubbed), 'semantic_hash': self.normalizer.semantic_hash(scrubbed), 'estimated_ops': scrubbed.estimated_ops, 'estimated_bytes': scrubbed.estimated_bytes})
        return {'task_id': graph.task_id, 'task_type': graph.task_type, 'domain': graph.domain, 'outputs': sorted(graph.outputs), 'fragments': fragments}

    def _canonical_group(self, value: str | None) -> str | None:
        if value is None:
            return None
        value = re.sub('@prompt_[0-9a-f]+', '', value)
        value = re.sub('@variant_[0-9a-f]+', '', value)
        value = re.sub('\\s+', '_', value).strip('_')
        return value or None

    def _scrub_metadata(self, metadata: Dict[str, object]) -> Dict[str, object]:
        cleaned = {}
        for key, value in metadata.items():
            if key in {'prompt_hash', 'prompt_text', 'prompt_variant', 'runtime_seed', 'secret_tag', 'nonce'}:
                continue
            cleaned[key] = value
        return cleaned

    def _scrub_fragment(self, frag: Fragment) -> Fragment:
        return replace(frag, semantic_group=self._canonical_group(frag.semantic_group), metadata=self._scrub_metadata(frag.metadata))
