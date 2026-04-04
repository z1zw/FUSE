from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import replace
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple
from .ir import Fragment, MicroFragment, RepresentativePath, TaskGraph
from .normalization import Normalizer

@dataclass
class DecompositionResult:
    microfragments: List[MicroFragment]
    representative_paths: Dict[str, RepresentativePath]
    task_to_micro_sequences: Dict[str, List[str]]
    structural_clusters: Dict[str, List[str]]
DEFAULT_OP_LATENCY = {'field_mul': 2, 'field_add': 1, 'field_reduce': 1, 'fft_butterfly': 2, 'poseidon_sbox': 2, 'poseidon_mix': 1, 'msm_bucket': 2, 'reduce_tree': 1, 'im2col': 1, 'matmul_tile': 2, 'relu': 1, 'maxpool': 1, 'gelu': 2, 'layernorm': 2, 'subbytes': 1, 'shift_rows': 1, 'mix_columns': 2, 'add_round_key': 1, 'key_expand': 1, 'linear_map': 2, 'permutation': 1, 'nonlinear_lookup': 2, 'generic': 1}

class MicrofragmentDecomposer:

    def __init__(self, normalizer: Normalizer | None=None):
        self.normalizer = normalizer or Normalizer()

    def decompose_tasks(self, tasks: Sequence[TaskGraph]) -> DecompositionResult:
        all_microfragments: List[MicroFragment] = []
        task_to_sequence: Dict[str, List[str]] = {}
        structural_clusters: Dict[str, List[str]] = defaultdict(list)
        for task in tasks:
            task.validate()
            sequence: List[str] = []
            for fragment in task.topological_fragments():
                micro_ops = self._micro_ops_for_fragment(fragment)
                prev_micro_id: str | None = None
                for index, op_name in enumerate(micro_ops):
                    micro_id = f'{task.task_id}:{fragment.fragment_id}:{index}'
                    structural_hash = self._structural_hash(fragment, op_name, index)
                    semantic_hash = self._semantic_hash(fragment, op_name, index)
                    micro = MicroFragment(micro_id=micro_id, parent_fragment_id=fragment.fragment_id, task_id=task.task_id, task_type=task.task_type, stage=fragment.stage, op=op_name, template_family=self._template_family(op_name, fragment), context=fragment.context, estimated_ops=max(1, fragment.estimated_ops // max(1, len(micro_ops))), estimated_bytes=max(0, fragment.estimated_bytes // max(1, len(micro_ops))), deps=[] if prev_micro_id is None else [prev_micro_id], semantic_group=f'{fragment.semantic_group}:{op_name}:{index % 2}', structural_hash=structural_hash, semantic_hash=semantic_hash, metadata={'fragment_id': fragment.fragment_id, 'index_in_fragment': index})
                    all_microfragments.append(micro)
                    sequence.append(micro_id)
                    structural_clusters[structural_hash].append(micro_id)
                    prev_micro_id = micro_id
            self._bind_fragment_dependencies(task, all_microfragments)
            task_to_sequence[task.task_id] = sequence
        representative_paths = self._build_representative_paths(all_microfragments, task_to_sequence)
        all_microfragments = self._annotate_path_bindings(all_microfragments, representative_paths)
        return DecompositionResult(microfragments=all_microfragments, representative_paths=representative_paths, task_to_micro_sequences=task_to_sequence, structural_clusters=dict(structural_clusters))

    def _micro_ops_for_fragment(self, fragment: Fragment) -> List[str]:
        if 'micro_ops' in fragment.metadata:
            return list(fragment.metadata['micro_ops'])
        expr = self.normalizer.normalize_expr(fragment.expr, structural=True)
        ops = self._collect_ops(expr)
        return ops or [fragment.expr.op]

    def _collect_ops(self, expr) -> List[str]:
        ops: List[str] = []
        if hasattr(expr, 'args'):
            for arg in expr.args:
                if hasattr(arg, 'op'):
                    ops.extend(self._collect_ops(arg))
        if hasattr(expr, 'op'):
            if expr.op not in {'literal', 'const'}:
                ops.append(expr.op)
        return ops

    def _structural_hash(self, fragment: Fragment, op_name: str, index: int) -> str:
        base = self.normalizer.structural_hash(fragment)
        return f'{base}:{op_name}:{fragment.stage}:{fragment.template_family or op_name}'

    def _semantic_hash(self, fragment: Fragment, op_name: str, index: int) -> str:
        base = self.normalizer.semantic_hash(fragment)
        return f'{base}:{op_name}:{index % 2}'

    def _template_family(self, op_name: str, fragment: Fragment) -> str:
        alias = {'field_mul': 'linear_map', 'field_add': 'linear_map', 'fft_butterfly': 'linear_map', 'matmul_tile': 'linear_map', 'im2col': 'permutation', 'relu': 'nonlinear_lookup', 'gelu': 'nonlinear_lookup', 'subbytes': 'nonlinear_lookup', 'poseidon_sbox': 'nonlinear_lookup', 'shift_rows': 'permutation', 'mix_columns': 'linear_map', 'maxpool': 'reduction', 'reduce_tree': 'reduction', 'layernorm': 'reduction', 'add_round_key': 'linear_map', 'key_expand': 'linear_map', 'msm_bucket': 'reduction'}
        return alias.get(op_name, fragment.template_family or 'generic')

    def _bind_fragment_dependencies(self, task: TaskGraph, all_microfragments: List[MicroFragment]) -> None:
        by_fragment: Dict[str, List[MicroFragment]] = defaultdict(list)
        for micro in all_microfragments:
            if micro.task_id == task.task_id:
                by_fragment[micro.parent_fragment_id].append(micro)
        for fragment in task.topological_fragments():
            current = by_fragment[fragment.fragment_id]
            if not current:
                continue
            first_micro = current[0]
            for dep_fragment_id in fragment.deps:
                dep_list = by_fragment.get(dep_fragment_id, [])
                if dep_list:
                    last_dep = dep_list[-1].micro_id
                    if last_dep not in first_micro.deps:
                        first_micro.deps.append(last_dep)

    def _build_representative_paths(self, all_microfragments: Sequence[MicroFragment], task_to_sequence: Dict[str, List[str]]) -> Dict[str, RepresentativePath]:
        by_structural = defaultdict(list)
        lookup = {m.micro_id: m for m in all_microfragments}
        for micro in all_microfragments:
            by_structural[micro.structural_hash].append(micro)
        sequences = {task_id: [lookup[m].structural_hash for m in micro_ids] for task_id, micro_ids in task_to_sequence.items()}
        micro_sequences = {task_id: list(micro_ids) for task_id, micro_ids in task_to_sequence.items()}
        ngrams: Counter[Tuple[str, ...]] = Counter()
        for seq in sequences.values():
            max_n = min(4, len(seq))
            for n in range(2, max_n + 1):
                for i in range(0, len(seq) - n + 1):
                    ngrams[tuple(seq[i:i + n])] += 1
        chosen = [ngram for ngram, count in ngrams.items() if count >= 2]
        chosen.sort(key=lambda item: (-len(item), -ngrams[item], item))
        representative_paths: Dict[str, RepresentativePath] = {}
        assigned_prefixes: set[Tuple[str, ...]] = set()
        for idx, signature in enumerate(chosen):
            if signature in assigned_prefixes:
                continue
            members: List[str] = []
            template_families: Tuple[str, ...] | None = None
            seen_members = set()
            for task_id, seq in sequences.items():
                micro_ids = micro_sequences[task_id]
                for i in range(0, len(seq) - len(signature) + 1):
                    if tuple(seq[i:i + len(signature)]) != signature:
                        continue
                    window = micro_ids[i:i + len(signature)]
                    if template_families is None:
                        template_families = tuple((lookup[m].template_family for m in window))
                    for member in window:
                        if member not in seen_members:
                            seen_members.add(member)
                            members.append(member)
            if not members:
                continue
            representative_paths[f'RP{idx:03d}'] = RepresentativePath(path_id=f'RP{idx:03d}', members=members, structural_signature=signature, template_families=template_families or (lookup[members[0]].template_family,), reuse_count=ngrams[signature])
            assigned_prefixes.add(signature)
        for idx, (structural_hash, items) in enumerate(sorted(by_structural.items())):
            if any((structural_hash in rp.structural_signature for rp in representative_paths.values())):
                continue
            micro = items[0]
            pid = f'SINGLE{idx:03d}'
            representative_paths[pid] = RepresentativePath(path_id=pid, members=[m.micro_id for m in items], structural_signature=(structural_hash,), template_families=(micro.template_family,), reuse_count=len(items))
        return representative_paths

    def _annotate_path_bindings(self, all_microfragments: Sequence[MicroFragment], representative_paths: Dict[str, RepresentativePath]) -> List[MicroFragment]:
        best_binding: Dict[str, Tuple[int, int, str, int]] = {}
        for path in representative_paths.values():
            member_to_position = {member: idx for idx, member in enumerate(path.members)}
            score = (len(path.structural_signature), path.reuse_count)
            for member, position in member_to_position.items():
                incumbent = best_binding.get(member)
                if incumbent is not None and incumbent[:2] >= score:
                    continue
                best_binding[member] = (score[0], score[1], path.path_id, position)
        annotated: List[MicroFragment] = []
        for micro in all_microfragments:
            binding = best_binding.get(micro.micro_id)
            if binding is None:
                annotated.append(micro)
                continue
            _, reuse_count, path_id, position = binding
            annotated.append(replace(micro, path_id=path_id, path_position=position, path_reuse_count=reuse_count))
        return annotated

def estimate_latency_cycles(micro: MicroFragment) -> int:
    base = DEFAULT_OP_LATENCY.get(micro.op, DEFAULT_OP_LATENCY.get(micro.template_family, 1))
    scale = max(1, micro.estimated_ops // 64)
    memory_penalty = 1 if micro.estimated_bytes > 0 else 0
    return max(1, base * scale + memory_penalty)
