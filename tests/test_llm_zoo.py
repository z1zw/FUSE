"""Regression tests for the language-model workloads and evaluation."""
from __future__ import annotations

import json

import pytest

from fuse.ir import RuntimeTask
from fuse.llm_frontend_eval import (
    SERVICE_INTENTS,
    extract_plan,
    graphs_from_plan,
    inject_prompt_variance,
    plan_fidelity,
    raw_graph_hash,
)
from fuse.llm.canonicalizer import CanonicalTaskCanonicalizer
from fuse.llm.verifier import FuseVerifier
from fuse.reuse import MicrofragmentDecomposer
from fuse.sim.simulator import FuseSimulator, SimulationConfig
from fuse.workloads.llm import (
    LLMArchSpec,
    build_llm_decode_task,
    build_llm_prefill_task,
    spec_from_hf_config,
)

DENSE = LLMArchSpec(
    name='dense', label='Dense', hf_id='test/dense', layers=4, hidden=512, heads=8,
    kv_heads=2, head_dim=64, ff_dim=1376, vocab=32000,
)
MOE = LLMArchSpec(
    name='moe', label='MoE', hf_id='test/moe', layers=3, hidden=512, heads=8,
    kv_heads=8, head_dim=64, ff_dim=256, vocab=32000, moe_experts=16, moe_top_k=4,
)


def test_prefill_graph_is_a_dag_with_every_layer():
    task = build_llm_prefill_task(DENSE, seq_len=64)
    task.validate()
    assert task.metadata['layers'] == DENSE.layers
    for layer in range(DENSE.layers):
        assert any(f.startswith(f'L{layer:02d}_') for f in task.fragments)
    assert task.outputs and all(o.startswith('lm_head') for o in task.outputs)


def test_layer_exposes_parallel_work():
    """Q/K/V, head groups and matmul tiles must be siblings, not a chain."""
    task = build_llm_prefill_task(DENSE, seq_len=64, head_groups=4, width=4)
    roots_of_layer = [
        f for f in task.fragments
        if f.startswith('L01_') and task.fragments[f].deps == ['L01_norm_in']
    ]
    # three projections, tiled query included
    assert len(roots_of_layer) >= 3
    score_fragments = [f for f in task.fragments if f.startswith('L01_h') and f.endswith('_score')]
    assert len(score_fragments) == 4


def test_moe_layer_emits_one_branch_per_routed_expert():
    task = build_llm_prefill_task(MOE, seq_len=32)
    downs = [f for f in task.fragments if f.startswith('L00_e') and '_ff_down' in f]
    experts = {f.split('_ff_down')[0] for f in downs}
    assert len(experts) == MOE.active_experts
    assert any(f == 'L00_router' for f in task.fragments)


def test_layers_do_not_share_a_result_cache_key():
    """Different layers hold different weights, so they must not alias."""
    task = build_llm_prefill_task(DENSE, seq_len=32)
    groups = [task.fragments[f].semantic_group for f in task.fragments if f.endswith('_norm_in')]
    assert len(set(groups)) == len(groups)
    # ...but the structural signature, which ignores operands, still matches.
    decomposition = MicrofragmentDecomposer().decompose_tasks([task])
    by_role = {}
    for micro in decomposition.microfragments:
        if micro.parent_fragment_id.endswith('_norm_in'):
            by_role.setdefault(micro.structural_hash, set()).add(micro.parent_fragment_id)
    assert any(len(fragments) == DENSE.layers for fragments in by_role.values())


def test_content_tag_separates_requests():
    a = build_llm_prefill_task(DENSE, seq_len=32, content_tag='req-a', task_id='a')
    b = build_llm_prefill_task(DENSE, seq_len=32, content_tag='req-b', task_id='b')
    groups_a = {f.semantic_group for f in a.fragments.values()}
    groups_b = {f.semantic_group for f in b.fragments.values()}
    assert not (groups_a & groups_b)
    shared = build_llm_prefill_task(DENSE, seq_len=32, content_tag='req-a', task_id='c')
    assert {f.semantic_group for f in shared.fragments.values()} == groups_a


def test_decode_step_is_cheaper_than_prefill():
    prefill = build_llm_prefill_task(DENSE, seq_len=128, task_id='p')
    decode = build_llm_decode_task(DENSE, context_len=128, task_id='d')
    prefill_ops = sum(f.estimated_ops for f in prefill.fragments.values())
    decode_ops = sum(f.estimated_ops for f in decode.fragments.values())
    assert decode_ops < prefill_ops


def test_simulation_is_reproducible_across_instances():
    task = build_llm_prefill_task(DENSE, seq_len=48, task_id='rep')
    decomposition = MicrofragmentDecomposer().decompose_tasks([task])
    runtime = [RuntimeTask(runtime_id='rt', task_graph=task, arrival_cycle=0)]
    runs = [
        FuseSimulator(config=SimulationConfig(mode='fuse')).run(runtime, decomposition.microfragments)
        .metrics['makespan_cycles']
        for _ in range(3)
    ]
    assert len(set(runs)) == 1


def test_transformer_asic_baseline_is_available():
    task = build_llm_prefill_task(DENSE, seq_len=48, task_id='base')
    decomposition = MicrofragmentDecomposer().decompose_tasks([task])
    runtime = [RuntimeTask(runtime_id='rt', task_graph=task, arrival_cycle=0)]
    metrics = FuseSimulator(config=SimulationConfig(mode='transformer_asic')).run(
        runtime, decomposition.microfragments
    ).metrics
    assert metrics['mode'] == 'transformer_asic'
    assert metrics['makespan_ms'] > 0


def test_spec_from_hf_config_reads_grouped_query_and_moe():
    class Config:
        model_type = 'olmoe'
        num_hidden_layers = 16
        hidden_size = 2048
        num_attention_heads = 16
        num_key_value_heads = 4
        intermediate_size = 1024
        vocab_size = 50304
        hidden_act = 'silu'
        num_experts = 64
        num_experts_per_tok = 8

    spec = spec_from_hf_config(Config(), name='x', label='X', hf_id='x/y')
    assert spec.kv_heads == 4 and spec.head_dim == 128
    assert spec.is_moe and spec.active_experts == 8
    assert spec.parameter_count() > 0


# --------------------------------------------------------------- front-end


@pytest.mark.parametrize(
    'raw',
    [
        '{"tasks":[{"type":"llm","seq_len":768,"hidden":3072,"heads":24}]}',
        '```json\n{"tasks":[{"type":"llm","seq_len":768,"hidden":3072,"heads":24}]}\n```',
        'Sure! Here is the plan:\n{"tasks":[{"type":"llm","seq_len":768,"hidden":3072,"heads":24}]}\nHope that helps.',
    ],
)
def test_plan_extraction_survives_wrapping(raw):
    plan = extract_plan(raw)
    assert plan.parse_ok
    assert plan.tasks[0]['seq_len'] == 768


@pytest.mark.parametrize('raw', ['', 'no json here', '{"oops": 1}', '{"tasks": []}', '{"tasks":[{"type":"bogus"}]}'])
def test_plan_extraction_rejects_unusable_replies(raw):
    assert not extract_plan(raw).parse_ok


def test_plan_values_are_clamped_to_supported_ranges():
    plan = extract_plan('{"tasks":[{"type":"llm","seq_len":999999,"hidden":-4,"heads":24}]}')
    assert plan.parse_ok
    assert plan.tasks[0]['seq_len'] == 8192
    assert plan.tasks[0]['hidden'] == 64


def test_plan_fidelity_scores_against_the_request():
    service = SERVICE_INTENTS[0]
    perfect = extract_plan(json.dumps({'tasks': [
        {'type': 'llm', **service['reference']['llm']},
        {'type': 'aes', **service['reference']['aes']},
        {'type': 'zkp', **service['reference']['zkp']},
    ]}))
    assert plan_fidelity(perfect, service['reference']) == 1.0
    partial = extract_plan(json.dumps({'tasks': [{'type': 'llm', **service['reference']['llm']}]}))
    assert 0.0 < plan_fidelity(partial, service['reference']) < 1.0


def test_canonicalization_collapses_wordings_that_variance_keeps_apart():
    plan = extract_plan('{"tasks":[{"type":"llm","seq_len":512,"hidden":2048,"heads":16},'
                        '{"type":"aes","blocks":1024}]}')
    canonicalizer = CanonicalTaskCanonicalizer()
    graphs_a = graphs_from_plan(plan, task_prefix='m_intent_0')
    graphs_b = graphs_from_plan(plan, task_prefix='m_intent_1')

    tagged_a = inject_prompt_variance(graphs_a, 'build it this way')
    tagged_b = inject_prompt_variance(graphs_b, 'build it that way instead')
    assert raw_graph_hash(tagged_a) != raw_graph_hash(tagged_b)

    canonical_a = canonicalizer.canonicalize_graphs(graphs_a)
    canonical_b = canonicalizer.canonicalize_graphs(graphs_b)
    assert canonicalizer.canonical_hash(canonical_a) == canonicalizer.canonical_hash(canonical_b)
    assert FuseVerifier().verify_graphs(canonical_a).ok
