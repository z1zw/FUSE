from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import hashlib
import json
import math
import networkx as nx
FrozenValue = Union[str, int, float, bool, Tuple['FrozenValue', ...]]

def _freeze(value: Any) -> FrozenValue:
    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return '<none>'
    if isinstance(value, (list, tuple)):
        return tuple((_freeze(v) for v in value))
    if isinstance(value, dict):
        return tuple(((str(k), _freeze(v)) for k, v in sorted(value.items())))
    return repr(value)

@dataclass(frozen=True)
class InputRef:
    name: str
    shape: Tuple[int, ...] = ()
    dtype: str = 'generic'

    def to_structural(self) -> Tuple[str, Tuple[int, ...], str]:
        return ('input', self.shape, self.dtype)

    def to_semantic(self) -> Tuple[str, str, Tuple[int, ...], str]:
        return ('input', self.name, self.shape, self.dtype)

@dataclass(frozen=True)
class Const:
    value: FrozenValue

    def to_structural(self) -> Tuple[str, str]:
        value_type = type(self.value).__name__
        return ('const', value_type)

    def to_semantic(self) -> Tuple[str, FrozenValue]:
        return ('const', self.value)
ExprArg = Union['Expr', InputRef, Const]

@dataclass(frozen=True)
class Expr:
    op: str
    args: Tuple[ExprArg, ...] = ()
    attrs: Tuple[Tuple[str, FrozenValue], ...] = ()
    template_family: str = 'generic'

    @staticmethod
    def build(op: str, *args: ExprArg, template_family: str='generic', **attrs: Any) -> 'Expr':
        return Expr(op=op, args=tuple(args), attrs=tuple(((str(k), _freeze(v)) for k, v in sorted(attrs.items()))), template_family=template_family)

    def to_dict(self) -> Dict[str, Any]:

        def encode(arg: ExprArg) -> Any:
            if isinstance(arg, Expr):
                return arg.to_dict()
            if isinstance(arg, InputRef):
                return {'input': arg.to_semantic()}
            if isinstance(arg, Const):
                return {'const': arg.to_semantic()}
            raise TypeError(f'Unsupported arg: {type(arg)!r}')
        return {'op': self.op, 'args': [encode(a) for a in self.args], 'attrs': list(self.attrs), 'template_family': self.template_family}

    def structural_digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

@dataclass(frozen=True)
class ContextKey:
    domain: str
    dtype: str
    layout: str
    tile: Tuple[int, ...] = ()
    quant: str = 'none'
    delta_t: int = 0

    def adapt_cost(self, other: 'ContextKey') -> Optional[int]:
        if self.domain != other.domain:
            return None
        if self.dtype != other.dtype:
            numeric = {'int8', 'int16', 'int32', 'uint8', 'uint32', 'field255', 'fp16', 'fp32'}
            if self.dtype not in numeric or other.dtype not in numeric:
                return None
            return 2
        cost = 0
        if self.layout != other.layout:
            cost += 1
        if self.tile != other.tile:
            if math.prod(self.tile or (1,)) != math.prod(other.tile or (1,)):
                return None
            cost += 1
        if self.quant != other.quant:
            return None
        return cost

@dataclass
class Fragment:
    fragment_id: str
    task_type: str
    stage: str
    expr: Expr
    context: ContextKey
    deps: List[str] = field(default_factory=list)
    estimated_ops: int = 1
    estimated_bytes: int = 0
    template_family: str = 'generic'
    semantic_group: Optional[str] = None
    executor: Optional[Callable[[Dict[str, Any]], Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.template_family:
            self.template_family = self.expr.template_family or self.expr.op
        if self.semantic_group is None:
            self.semantic_group = self.fragment_id

@dataclass
class MicroFragment:
    micro_id: str
    parent_fragment_id: str
    task_id: str
    task_type: str
    stage: str
    op: str
    template_family: str
    context: ContextKey
    estimated_ops: int
    estimated_bytes: int
    deps: List[str] = field(default_factory=list)
    semantic_group: Optional[str] = None
    structural_hash: str = ''
    semantic_hash: str = ''
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RepresentativePath:
    path_id: str
    members: List[str]
    structural_signature: Tuple[str, ...]
    template_families: Tuple[str, ...]
    reuse_count: int = 0

class TaskGraph:

    def __init__(self, task_id: str, task_type: str, domain: str, metadata: Optional[Dict[str, Any]]=None):
        self.task_id = task_id
        self.task_type = task_type
        self.domain = domain
        self.metadata = metadata or {}
        self.fragments: Dict[str, Fragment] = {}
        self.graph = nx.DiGraph()
        self.outputs: List[str] = []

    def add_fragment(self, fragment: Fragment) -> None:
        self.fragments[fragment.fragment_id] = fragment
        self.graph.add_node(fragment.fragment_id)
        for dep in fragment.deps:
            self.graph.add_edge(dep, fragment.fragment_id)

    def set_outputs(self, fragment_ids: Sequence[str]) -> None:
        self.outputs = list(fragment_ids)

    def validate(self) -> None:
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError(f'TaskGraph {self.task_id} contains a cycle')
        missing = set(self.graph.nodes()) - set(self.fragments)
        if missing:
            raise KeyError(f'TaskGraph {self.task_id} missing fragment definitions: {sorted(missing)}')

    def topological_fragments(self) -> List[Fragment]:
        self.validate()
        return [self.fragments[node] for node in nx.topological_sort(self.graph)]

    def roots(self) -> List[str]:
        return [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]

    def leaves(self) -> List[str]:
        return [n for n in self.graph.nodes if self.graph.out_degree(n) == 0]

    def copy(self, task_id: Optional[str]=None) -> 'TaskGraph':
        clone = TaskGraph(task_id or self.task_id, self.task_type, self.domain, dict(self.metadata))
        for fragment in self.topological_fragments():
            clone.add_fragment(Fragment(**{**fragment.__dict__}))
        clone.set_outputs(self.outputs)
        return clone

    def summary(self) -> Dict[str, Any]:
        return {'task_id': self.task_id, 'task_type': self.task_type, 'domain': self.domain, 'fragments': len(self.fragments), 'edges': self.graph.number_of_edges(), 'stages': sorted({f.stage for f in self.fragments.values()})}

@dataclass
class RuntimeTask:
    runtime_id: str
    task_graph: TaskGraph
    arrival_cycle: int = 0
    runtime_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RunResult:
    metrics: Dict[str, Any]
    traces: Dict[str, Any]
    tables: Dict[str, Any] = field(default_factory=dict)
