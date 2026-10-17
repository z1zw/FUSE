"""Task graphs for real large language models.

The builders in this module turn a concrete LLM architecture (layer count,
hidden width, attention head layout, feed-forward shape, MoE routing) into a
FUSE ``TaskGraph``.  Unlike :mod:`fuse.workloads.transformer`, which describes a
single generic block, these graphs describe the whole model so that the runtime
observes the repeated per-layer structure that fragment reuse exploits.

Cost model.  ``estimated_ops`` counts multiply-accumulate (or elementwise)
operations.  ``estimate_latency_cycles`` turns a ``matmul_tile`` micro-op into
``2 * ops // 64`` cycles, i.e. 32 MACs per cycle per RFU, which matches the
32-lane SIMD datapath in the evaluated FUSE configuration.  ``estimated_bytes``
counts the weight and activation traffic that crosses the on-chip boundary; the
simulator converts it with the configured memory bandwidth, so decode steps come
out memory-bound and prefill steps compute-bound, as on real hardware.

Shape of the graph.  Work a real implementation issues concurrently becomes
sibling fragments rather than a chain: the three projections, ``width`` output
tiles of each large matmul, ``head_groups`` attention head groups, the gate and
up branches of a gated feed-forward, and the routed experts of an MoE layer.
Without that the graph is one long dependency chain and the scheduler sees no
width to fill the array with.

Reuse keys.  Two fragments may share a cached *result* only when they compute
the same values, so each fragment's semantic group is qualified by the weights
it reads (its layer and output tile) and by the activations it consumes (the
``content_tag``, which identifies the token span).  Layers therefore never
alias, while the *structural* signature -- which ignores operand identity --
still matches across them and keeps driving template pinning and
representative-path reuse.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from ..ir import ContextKey, Expr, Fragment, InputRef, TaskGraph

BYTES_PER_ELEMENT = 2
DEFAULT_DTYPE = 'fp16'


@dataclass(frozen=True)
class LLMArchSpec:
    """Architecture of a concrete language model."""

    name: str
    label: str
    hf_id: str
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    ff_dim: int
    vocab: int
    norm: str = 'rmsnorm'
    activation: str = 'silu'
    gated_ff: bool = True
    moe_experts: int = 0
    moe_top_k: int = 0
    tie_embeddings: bool = False

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def attn_dim(self) -> int:
        return self.heads * self.head_dim

    @property
    def is_moe(self) -> bool:
        return self.moe_experts > 0 and self.moe_top_k > 0

    @property
    def active_experts(self) -> int:
        return min(self.moe_top_k, self.moe_experts) if self.is_moe else 1

    def parameter_count(self) -> int:
        """Approximate parameter count implied by the recorded shapes."""
        per_layer_attn = self.hidden * (self.attn_dim + 2 * self.kv_dim) + self.attn_dim * self.hidden
        ff_matrices = 3 if self.gated_ff else 2
        per_expert_ff = ff_matrices * self.hidden * self.ff_dim
        experts = self.moe_experts if self.is_moe else 1
        per_layer_ff = experts * per_expert_ff + (self.hidden * self.moe_experts if self.is_moe else 0)
        embeddings = self.hidden * self.vocab * (1 if self.tie_embeddings else 2)
        return self.layers * (per_layer_attn + per_layer_ff) + embeddings


def _first(config: Any, *names: str, default: Any = None) -> Any:
    for candidate in names:
        value = getattr(config, candidate, None)
        if value is not None:
            return value
    return default


def spec_from_hf_config(config: Any, *, name: str, label: str, hf_id: str) -> LLMArchSpec:
    """Read a :mod:`transformers` config into an :class:`LLMArchSpec`."""
    layers = int(_first(config, 'num_hidden_layers', 'n_layer', default=12))
    hidden = int(_first(config, 'hidden_size', 'n_embd', default=768))
    heads = int(_first(config, 'num_attention_heads', 'n_head', default=12))
    kv_heads = int(_first(config, 'num_key_value_heads', default=heads) or heads)
    head_dim = int(_first(config, 'head_dim', default=0) or (hidden // max(1, heads)))
    ff_dim = int(_first(config, 'intermediate_size', 'n_inner', default=0) or 4 * hidden)
    vocab = int(_first(config, 'vocab_size', default=32000))
    model_type = str(getattr(config, 'model_type', '')).lower()
    activation = str(_first(config, 'hidden_act', 'activation_function', default='silu'))
    dense_gpt_family = {'gpt2', 'gptj', 'gpt_neo', 'opt', 'bloom'}
    gated = model_type not in dense_gpt_family
    norm = 'layernorm' if model_type in dense_gpt_family else 'rmsnorm'
    moe_experts = int(_first(config, 'num_experts', 'num_local_experts', 'n_routed_experts', default=0) or 0)
    moe_top_k = int(_first(config, 'num_experts_per_tok', 'num_experts_per_token', default=0) or 0)
    return LLMArchSpec(
        name=name,
        label=label,
        hf_id=hf_id,
        layers=layers,
        hidden=hidden,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        ff_dim=ff_dim,
        vocab=vocab,
        norm=norm,
        activation=activation,
        gated_ff=gated,
        moe_experts=moe_experts,
        moe_top_k=moe_top_k,
        tie_embeddings=bool(_first(config, 'tie_word_embeddings', default=False)),
    )


class _GraphBuilder:
    """Accumulates fragments for one model pass.

    ``semantic_group`` names the *role* a fragment plays (``ff_up``,
    ``attn_score``, ...).  Two fragments may only share a cached *result* when
    they compute the same values, so the key the FVM sees is the role qualified
    by the weights it reads (the fragment id, which names the layer and output
    tile) and by the activations it consumes (``content_tag``, which identifies
    the token span being processed).
    Fragments in different layers therefore never alias, while the *structural*
    hash -- which ignores operand identity -- still matches across layers and
    keeps driving template pinning and representative-path reuse.
    """

    def __init__(self, task: TaskGraph, spec: LLMArchSpec, batch_size: int, content_tag: str):
        self.task = task
        self.spec = spec
        self.batch = batch_size
        self.content_tag = content_tag
        self.previous: Optional[str] = None

    def add(
        self,
        fragment_id: str,
        *,
        stage: str,
        op: str,
        template_family: str,
        micro_ops: Sequence[str],
        ops: int,
        byte_count: int,
        tile: Sequence[int],
        semantic_group: str,
        deps: Optional[Sequence[str]] = None,
        layout: str = 'tile',
    ) -> str:
        dependencies = list(deps) if deps is not None else ([self.previous] if self.previous else [])
        # ``fragment_id`` already names the layer and the output tile, so it is
        # the weight identity; ``content_tag`` is the activation identity.
        qualified_group = f'{semantic_group}@{fragment_id}@{self.content_tag}'
        fragment = Fragment(
            fragment_id=fragment_id,
            task_type='LLM',
            stage=stage,
            expr=Expr.build(
                op,
                InputRef(qualified_group, shape=tuple(int(v) for v in tile), dtype=DEFAULT_DTYPE),
                template_family=template_family,
            ),
            context=ContextKey(
                domain='R',
                dtype=DEFAULT_DTYPE,
                layout=layout,
                tile=tuple(int(v) for v in tile),
                delta_t=2_000_000,
            ),
            deps=dependencies,
            estimated_ops=max(1, int(ops)),
            estimated_bytes=max(0, int(byte_count)),
            template_family=template_family,
            semantic_group=qualified_group,
            metadata={'micro_ops': list(micro_ops), 'role': semantic_group},
        )
        self.task.add_fragment(fragment)
        self.previous = fragment_id
        return fragment_id


def _split_evenly(total: int, parts: int) -> list:
    """Split ``total`` into ``parts`` positive chunks that sum to ``total``."""
    parts = max(1, min(parts, total))
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def _tiles_for(spec: LLMArchSpec, width: int) -> int:
    """How many output tiles one matmul is split into.

    The target is ``width`` concurrently runnable fragments per matmul stage.
    An MoE layer already offers that width through its routed experts, so its
    per-expert matrices are left untiled rather than multiplying the two.
    """
    return max(1, width // max(1, spec.active_experts))


def _add_layer(
    builder: _GraphBuilder,
    spec: LLMArchSpec,
    layer: int,
    batch: int,
    query_tokens: int,
    context_tokens: int,
    head_groups: int,
    width: int,
) -> None:
    """Emit one decoder layer.

    Work that a real implementation runs concurrently is emitted as sibling
    fragments so the scheduler sees the width that is actually available: the
    three projections, output tiles of each large matmul, the attention head
    groups, the gate/up branches of a gated feed-forward, and the routed
    experts of an MoE layer.  A matmul is tiled along its output dimension, so
    the tiles are independent; ``ff_down`` reduces over the full feed-forward
    dimension and therefore waits on every activation tile.
    """
    prefix = f'L{layer:02d}'
    hidden = spec.hidden
    tokens = batch * query_tokens
    act_bytes = tokens * hidden * BYTES_PER_ELEMENT
    tiles = _tiles_for(spec, width)

    norm_in = builder.add(
        f'{prefix}_norm_in',
        stage='SC',
        op=spec.norm,
        template_family='reduction',
        micro_ops=['layernorm'],
        ops=tokens * hidden * 2,
        byte_count=act_bytes,
        tile=(batch, query_tokens, hidden),
        semantic_group='attn_norm_in',
    )

    # Query projection is tiled across heads; the (much narrower) key and value
    # projections of a grouped-query model stay whole.
    q_tiles = []
    for index, slice_dim in enumerate(_split_evenly(spec.attn_dim, tiles)):
        q_tiles.append(
            builder.add(
                f'{prefix}_q_proj_{index:02d}',
                stage='IM',
                op='matmul_tile',
                template_family='linear_map',
                micro_ops=['matmul_tile'],
                ops=tokens * hidden * slice_dim,
                byte_count=hidden * slice_dim * BYTES_PER_ELEMENT + tokens * slice_dim * BYTES_PER_ELEMENT,
                tile=(batch, query_tokens, hidden),
                semantic_group='attn_q_proj',
                deps=[norm_in],
            )
        )
    kv_tiles = {}
    for tag in ('k', 'v'):
        kv_tiles[tag] = builder.add(
            f'{prefix}_{tag}_proj',
            stage='IM',
            op='matmul_tile',
            template_family='linear_map',
            micro_ops=['matmul_tile'],
            ops=tokens * hidden * spec.kv_dim,
            byte_count=hidden * spec.kv_dim * BYTES_PER_ELEMENT + tokens * spec.kv_dim * BYTES_PER_ELEMENT,
            tile=(batch, query_tokens, hidden),
            semantic_group=f'attn_{tag}_proj',
            deps=[norm_in],
        )

    rope_q = builder.add(
        f'{prefix}_q_rope',
        stage='IM',
        op='rope',
        template_family='permutation',
        micro_ops=['permutation'],
        ops=tokens * spec.attn_dim,
        byte_count=tokens * spec.attn_dim * BYTES_PER_ELEMENT,
        tile=(batch, query_tokens, spec.attn_dim),
        semantic_group='attn_q_rope',
        deps=q_tiles,
    )
    rope_k = builder.add(
        f'{prefix}_k_rope',
        stage='IM',
        op='rope',
        template_family='permutation',
        micro_ops=['permutation'],
        ops=tokens * spec.kv_dim,
        byte_count=tokens * spec.kv_dim * BYTES_PER_ELEMENT,
        tile=(batch, query_tokens, spec.kv_dim),
        semantic_group='attn_k_rope',
        deps=[kv_tiles['k']],
    )

    context_ids = []
    for group_index, group_heads in enumerate(_split_evenly(spec.heads, head_groups)):
        group_kv_heads = max(1, round(group_heads * spec.kv_heads / max(1, spec.heads)))
        score_ops = batch * group_heads * query_tokens * context_tokens * spec.head_dim
        kv_bytes = batch * group_kv_heads * context_tokens * spec.head_dim * BYTES_PER_ELEMENT
        score = builder.add(
            f'{prefix}_h{group_index:02d}_score',
            stage='IT',
            op='matmul_tile',
            template_family='linear_map',
            micro_ops=['matmul_tile'],
            ops=score_ops,
            byte_count=kv_bytes,
            tile=(batch, group_heads, query_tokens, spec.head_dim),
            semantic_group='attn_score',
            deps=[rope_q, rope_k],
        )
        softmax = builder.add(
            f'{prefix}_h{group_index:02d}_softmax',
            stage='SC',
            op='softmax',
            template_family='reduction',
            micro_ops=['layernorm', 'field_reduce'],
            ops=batch * group_heads * query_tokens * context_tokens,
            byte_count=batch * group_heads * query_tokens * context_tokens * BYTES_PER_ELEMENT,
            tile=(batch, group_heads, query_tokens, min(context_tokens, 256)),
            semantic_group='attn_softmax',
            deps=[score],
        )
        context_ids.append(
            builder.add(
                f'{prefix}_h{group_index:02d}_context',
                stage='IT',
                op='matmul_tile',
                template_family='linear_map',
                micro_ops=['matmul_tile'],
                ops=score_ops,
                byte_count=kv_bytes,
                tile=(batch, group_heads, query_tokens, spec.head_dim),
                semantic_group='attn_context',
                deps=[softmax, kv_tiles['v']],
            )
        )

    attn_out_tiles = []
    for index, slice_dim in enumerate(_split_evenly(hidden, tiles)):
        attn_out_tiles.append(
            builder.add(
                f'{prefix}_attn_out_{index:02d}',
                stage='IT',
                op='matmul_tile',
                template_family='linear_map',
                micro_ops=['matmul_tile'],
                ops=tokens * spec.attn_dim * slice_dim,
                byte_count=spec.attn_dim * slice_dim * BYTES_PER_ELEMENT + tokens * slice_dim * BYTES_PER_ELEMENT,
                tile=(batch, query_tokens, hidden),
                semantic_group='attn_out',
                deps=context_ids,
            )
        )
    resid_attn = builder.add(
        f'{prefix}_resid_attn',
        stage='SC',
        op='add',
        template_family='linear_map',
        micro_ops=['field_add'],
        ops=tokens * hidden,
        byte_count=act_bytes,
        tile=(batch, query_tokens, hidden),
        semantic_group='residual_attn',
        deps=attn_out_tiles + [norm_in],
    )
    norm_ff = builder.add(
        f'{prefix}_norm_ff',
        stage='SC',
        op=spec.norm,
        template_family='reduction',
        micro_ops=['layernorm'],
        ops=tokens * hidden * 2,
        byte_count=act_bytes,
        tile=(batch, query_tokens, hidden),
        semantic_group='ff_norm',
        deps=[resid_attn],
    )

    ff_root = norm_ff
    if spec.is_moe:
        ff_root = builder.add(
            f'{prefix}_router',
            stage='SC',
            op='matmul_tile',
            template_family='linear_map',
            micro_ops=['matmul_tile'],
            ops=tokens * hidden * spec.moe_experts,
            byte_count=hidden * spec.moe_experts * BYTES_PER_ELEMENT,
            tile=(batch, query_tokens, hidden),
            semantic_group='moe_router',
            deps=[norm_ff],
        )

    ff_down_ids = []
    branches = ('gate', 'up') if spec.gated_ff else ('up',)
    for expert in range(spec.active_experts):
        tag = f'{prefix}_e{expert:02d}' if spec.is_moe else prefix
        act_ids = []
        for index, slice_dim in enumerate(_split_evenly(spec.ff_dim, tiles)):
            branch_ids = []
            for branch in branches:
                branch_ids.append(
                    builder.add(
                        f'{tag}_ff_{branch}_{index:02d}',
                        stage='IT',
                        op='matmul_tile',
                        template_family='linear_map',
                        micro_ops=['matmul_tile'],
                        ops=tokens * hidden * slice_dim,
                        byte_count=hidden * slice_dim * BYTES_PER_ELEMENT,
                        tile=(batch, query_tokens, hidden),
                        semantic_group=f'ff_{branch}',
                        deps=[ff_root],
                    )
                )
            act_ids.append(
                builder.add(
                    f'{tag}_ff_act_{index:02d}',
                    stage='SC',
                    op=spec.activation,
                    template_family='nonlinear_lookup',
                    micro_ops=['gelu'],
                    ops=tokens * slice_dim,
                    byte_count=tokens * slice_dim * BYTES_PER_ELEMENT,
                    tile=(batch, query_tokens, spec.ff_dim),
                    semantic_group='ff_act',
                    deps=branch_ids,
                )
            )
        for index, slice_dim in enumerate(_split_evenly(hidden, tiles)):
            ff_down_ids.append(
                builder.add(
                    f'{tag}_ff_down_{index:02d}',
                    stage='IT',
                    op='matmul_tile',
                    template_family='linear_map',
                    micro_ops=['matmul_tile'],
                    ops=tokens * spec.ff_dim * slice_dim,
                    byte_count=spec.ff_dim * slice_dim * BYTES_PER_ELEMENT + tokens * slice_dim * BYTES_PER_ELEMENT,
                    tile=(batch, query_tokens, hidden),
                    semantic_group='ff_down',
                    deps=act_ids,
                )
            )

    builder.add(
        f'{prefix}_resid_ff',
        stage='SC',
        op='add',
        template_family='linear_map',
        micro_ops=['field_add'],
        ops=tokens * hidden,
        byte_count=act_bytes,
        tile=(batch, query_tokens, hidden),
        semantic_group='residual_ff',
        deps=ff_down_ids + [resid_attn],
    )


def build_llm_task(
    spec: LLMArchSpec,
    *,
    seq_len: int,
    gen_len: int = 0,
    batch_size: int = 1,
    phase: str = 'prefill',
    layers: Optional[int] = None,
    head_groups: int = 8,
    width: int = 8,
    vocab_splits: int = 8,
    content_tag: Optional[str] = None,
    task_id: Optional[str] = None,
) -> TaskGraph:
    """Build the task graph for one forward pass of ``spec``.

    ``phase='prefill'`` processes ``seq_len`` tokens at once.  ``phase='decode'``
    emits a single-token step against a ``seq_len``-long KV cache, which is the
    memory-bound regime that dominates generation.  ``head_groups`` and
    ``vocab_splits`` control how far attention and the output projection are
    spread across the reconfigurable units.
    """
    layer_count = int(layers) if layers else spec.layers
    layer_count = max(1, min(layer_count, spec.layers))
    query_tokens = seq_len if phase == 'prefill' else 1
    context_tokens = seq_len
    resolved_id = task_id or f'{spec.name}_{phase}_s{seq_len}'
    resolved_tag = content_tag if content_tag is not None else resolved_id
    task = TaskGraph(
        task_id=resolved_id,
        task_type='LLM',
        domain='R',
        metadata={
            'model': spec.name,
            'model_label': spec.label,
            'hf_id': spec.hf_id,
            'phase': phase,
            'seq_len': seq_len,
            'gen_len': gen_len,
            'hidden_dim': spec.hidden,
            'heads': spec.heads,
            'kv_heads': spec.kv_heads,
            'head_dim': spec.head_dim,
            'ff_dim': spec.ff_dim,
            'layers': layer_count,
            'total_layers': spec.layers,
            'batch_size': batch_size,
            'moe_experts': spec.moe_experts,
            'moe_top_k': spec.moe_top_k,
            'head_groups': head_groups,
            'width': width,
            'vocab_splits': vocab_splits,
            'content_tag': resolved_tag,
            'parameter_count': spec.parameter_count(),
        },
    )
    builder = _GraphBuilder(task, spec, batch_size, resolved_tag)
    batch = max(1, batch_size)
    tokens = batch * query_tokens
    hidden = spec.hidden
    act_bytes = tokens * hidden * BYTES_PER_ELEMENT

    builder.add(
        'embed',
        stage='IM',
        op='gather',
        template_family='permutation',
        micro_ops=['im2col'],
        ops=tokens * hidden,
        byte_count=act_bytes + tokens * 4,
        tile=(batch, query_tokens, hidden),
        semantic_group='token_embed',
        deps=[],
    )

    for layer in range(layer_count):
        _add_layer(builder, spec, layer, batch, query_tokens, context_tokens, head_groups, width)

    final_norm = builder.add(
        'final_norm',
        stage='SC',
        op=spec.norm,
        template_family='reduction',
        micro_ops=['layernorm'],
        ops=tokens * hidden * 2,
        byte_count=act_bytes,
        tile=(batch, query_tokens, hidden),
        semantic_group='final_norm',
    )
    # Only the last position needs logits, both when prefilling and when
    # decoding, so the output projection is sized for a single token and split
    # across the vocabulary the way tensor-parallel decoders do.
    head_ids = []
    for split_index, split_vocab in enumerate(_split_evenly(spec.vocab, vocab_splits)):
        head_ids.append(
            builder.add(
                f'lm_head_{split_index:02d}',
                stage='IT',
                op='matmul_tile',
                template_family='linear_map',
                micro_ops=['matmul_tile'],
                ops=batch * hidden * split_vocab,
                byte_count=hidden * split_vocab * BYTES_PER_ELEMENT + batch * split_vocab * BYTES_PER_ELEMENT,
                tile=(batch, 1, hidden),
                semantic_group='lm_head',
                deps=[final_norm],
            )
        )
    task.set_outputs(head_ids)
    task.validate()
    return task


def build_llm_prefill_task(
    spec: LLMArchSpec,
    *,
    seq_len: int,
    batch_size: int = 1,
    layers: Optional[int] = None,
    head_groups: int = 8,
    width: int = 8,
    content_tag: Optional[str] = None,
    task_id: Optional[str] = None,
) -> TaskGraph:
    return build_llm_task(
        spec, seq_len=seq_len, batch_size=batch_size, phase='prefill', layers=layers,
        head_groups=head_groups, width=width, content_tag=content_tag, task_id=task_id,
    )


def build_llm_decode_task(
    spec: LLMArchSpec,
    *,
    context_len: int,
    batch_size: int = 1,
    layers: Optional[int] = None,
    head_groups: int = 8,
    width: int = 8,
    content_tag: Optional[str] = None,
    task_id: Optional[str] = None,
) -> TaskGraph:
    return build_llm_task(
        spec, seq_len=context_len, batch_size=batch_size, phase='decode', layers=layers,
        head_groups=head_groups, width=width, content_tag=content_tag, task_id=task_id,
    )
