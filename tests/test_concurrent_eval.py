"""Regression tests for the concurrent three-domain study."""
from __future__ import annotations

import math

import pytest

from fuse.concurrent_eval import (
    BYTES_PER_AES_BLOCK,
    POSEIDON_CONSTRAINTS_PER_LEVEL,
    ConcurrentSecureSuite,
    RequestShape,
)
from fuse.workloads.llm import LLMArchSpec

SPEC = LLMArchSpec(name='t', label='Tiny', hf_id='t/t', layers=4, hidden=512, heads=8,
                   kv_heads=2, head_dim=64, ff_dim=1376, vocab=32000)

SHAPE = RequestShape(dataset='d', label='D', shape='s', samples=8, prompt_tokens=48,
                     gen_tokens=64, payload_bytes=240, retrieved_bytes=1880, corpus_size=974)


def suite(total_rfus=8, specialized=False):
    return ConcurrentSecureSuite(spec=SPEC, total_rfus=total_rfus, span_ms=10.0,
                                 width=4, specialized_engines=specialized)


def test_security_volume_follows_from_the_request():
    expected_bytes = SHAPE.payload_bytes + SHAPE.retrieved_bytes + SHAPE.gen_tokens * 4
    assert SHAPE.aes_blocks == math.ceil(expected_bytes / BYTES_PER_AES_BLOCK)
    depth = math.ceil(math.log2(SHAPE.corpus_size))
    assert SHAPE.zkp_constraints == depth * POSEIDON_CONSTRAINTS_PER_LEVEL


def test_bigger_corpus_deepens_the_proof():
    small = RequestShape('d', 'D', 's', 1, 48, 64, 240, 0, 1_000)
    large = RequestShape('d', 'D', 's', 1, 48, 64, 240, 0, 1_000_000)
    assert large.zkp_constraints > small.zkp_constraints


def test_all_three_domains_are_in_flight_together():
    pairs = suite().domain_pairs(SHAPE, llm_requests=2, security_per_request=4)
    assert set(pairs) == {'LLM', 'AES', 'ZKP'}
    assert len(pairs['LLM']) == 2
    assert len(pairs['AES']) == len(pairs['ZKP']) == 8
    # arrivals of each domain are spread over the window rather than batched at 0
    for domain in ('AES', 'ZKP'):
        arrivals = [a for a, _ in pairs[domain]]
        assert max(arrivals) > min(arrivals)
    assert {g.task_type for _, g in pairs['LLM']} == {'LLM'}
    assert {g.task_type for _, g in pairs['AES']} == {'AES'}
    assert {g.task_type for _, g in pairs['ZKP']} == {'ZKP'}


def test_no_security_means_no_security_tasks():
    pairs = suite().domain_pairs(SHAPE, llm_requests=2, security_per_request=0)
    assert pairs['AES'] == [] and pairs['ZKP'] == []


def _curve(values):
    """Makespan curve with a nominal energy attached to each engine size."""
    return {n: (ms, 0.001 * n) for n, ms in values.items()}


def test_best_split_uses_the_whole_budget_and_minimises_the_slowest_engine():
    curves = {
        'LLM': _curve({1: 100.0, 2: 50.0, 3: 34.0, 4: 26.0, 5: 22.0, 6: 20.0}),
        'AES': _curve({1: 5.0, 2: 3.0, 3: 2.0, 4: 1.5, 5: 1.2, 6: 1.0}),
        'ZKP': _curve({1: 6.0, 2: 3.5, 3: 2.5, 4: 2.0, 5: 1.5, 6: 1.2}),
    }
    span, energy, split = ConcurrentSecureSuite.best_split(curves, 8)
    assert sum(split) == 8
    assert span == max(curves['LLM'][split[0]][0], curves['AES'][split[1]][0],
                       curves['ZKP'][split[2]][0])
    # the language model is the bottleneck, so it should take what it can
    assert split[0] == 6
    assert span == pytest.approx(20.0)
    # the two fast engines idle until the model finishes, and are charged for it
    assert energy > sum(curves[d][n][1] for d, n in zip(('LLM', 'AES', 'ZKP'), split))


def test_best_split_never_starves_an_engine():
    curves = {d: _curve({n: 1.0 for n in range(1, 8)}) for d in ('LLM', 'AES', 'ZKP')}
    _, _, split = ConcurrentSecureSuite.best_split(curves, 6)
    assert min(split) >= 1 and sum(split) == 6


def test_curves_cover_every_engine_size():
    s = suite(total_rfus=6)
    pairs = s.domain_pairs(SHAPE, llm_requests=1, security_per_request=1)
    curves = s.domain_curves(pairs, max_rfus=6, specialized=False)
    assert set(curves) == {'LLM', 'AES', 'ZKP'}
    for domain in curves:
        assert set(curves[domain]) == set(range(1, 6))
        assert all(ms > 0 and energy >= 0 for ms, energy in curves[domain].values())


def test_more_rfus_never_slow_the_model_down():
    s = suite(total_rfus=8)
    frame = s.rfu_scaling(SHAPE, llm_requests=1)
    spans = frame.sort_values('rfus')['makespan_ms'].tolist()
    assert all(later <= earlier + 1e-9 for earlier, later in zip(spans, spans[1:]))


def test_specialisation_only_applies_to_the_dedicated_baseline():
    plain, special = suite(specialized=False), suite(specialized=True)
    assert plain._engine_mode('LLM') == 'fuse'
    assert special._engine_mode('LLM') == 'transformer_asic'
    assert special._engine_mode('AES') == 'openaes'
    assert special._engine_mode('ZKP') == 'static_zkp'


def test_evaluate_reports_a_split_that_adds_up():
    outcome = suite(total_rfus=8).evaluate(SHAPE, llm_requests=1, security_per_request=1)
    total = (outcome['dedicated_split_llm'] + outcome['dedicated_split_aes']
             + outcome['dedicated_split_zkp'])
    assert total == 8
    assert outcome['fuse_ms'] > 0 and outcome['dedicated_ms'] > 0
    assert outcome['speedup_vs_dedicated'] == pytest.approx(
        outcome['dedicated_ms'] / outcome['fuse_ms'])


def test_security_is_free_only_while_the_model_dominates():
    """A light obligation disappears into the model; a heavy one does not.

    The tax is what the security tasks add to the service, and it is only
    negligible while the model is large enough to hide them.  Asserting that it
    grows with the obligation keeps the claim honest: it is a property of the
    ratio between the two, not a free lunch.
    """
    s = suite(total_rfus=8)
    taxes = [s.evaluate(SHAPE, llm_requests=2, security_per_request=level)['security_tax']
             for level in (1, 4, 8)]
    assert abs(taxes[0]) < 0.01, 'one security task per request should vanish'
    assert taxes[0] < taxes[1] < taxes[2], 'the tax must grow with the obligation'


def test_specialisation_is_chosen_per_call_not_per_instance():
    """A single sweep asks for both baselines about the same workload.

    The bug this guards against: ``domain_curves(specialized=True)`` reading the
    flag off the instance instead of its argument, which silently made the
    dedicated baseline a copy of the partitioned one.
    """
    s = suite(total_rfus=6, specialized=False)
    assert s.engine_mode('LLM', specialized=True) == 'transformer_asic'
    assert s.engine_mode('LLM', specialized=False) == 'fuse'

    pairs = s.domain_pairs(SHAPE, llm_requests=1, security_per_request=1)
    plain = s.domain_curves(pairs, max_rfus=5, specialized=False)
    special = s.domain_curves(pairs, max_rfus=5, specialized=True)
    # a transformer engine is faster on its own domain, so the curves must differ
    assert any(plain['LLM'][n][0] != special['LLM'][n][0] for n in plain['LLM'])
