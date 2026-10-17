"""Contract C4: a cached result is returned only to the request that produced it.

The value hash binds operand references, not bytes, so two graphs built from
the same bare builder arguments hash alike. The lookup must therefore compare
the consumer's originating request with the producer's. Replaying one graph is
one request served many times and its hits are legitimate; distinct graphs are
distinct requests and may never share a result.
"""
from __future__ import annotations

import pytest

from fuse.ir import RuntimeTask
from fuse.reuse import MicrofragmentDecomposer
from fuse.sim.fvm import FeatureVectorManager
from fuse.sim.simulator import CYCLES_PER_MS, FuseSimulator, SimulationConfig
from fuse.workloads import build_aes128_task


def _simulate(graphs, repeats):
    # The arrival pattern of benchmarks._raw_fig10, so a producer has retired
    # before the next consumer dispatches.
    tasks, arrival_ms = [], 0.0
    for graph in graphs:
        for i in range(repeats):
            tasks.append(RuntimeTask(runtime_id=f'{graph.task_id}_rt{i}',
                                     task_graph=graph,
                                     arrival_cycle=int(arrival_ms * CYCLES_PER_MS)))
            arrival_ms += 0.045 if i % 2 == 0 else 0.12
    decomposition = MicrofragmentDecomposer().decompose_tasks(graphs)
    return FuseSimulator(config=SimulationConfig(mode='fuse')).run(
        tasks, decomposition.microfragments).metrics


def test_same_request_served_repeatedly_reuses_results():
    metrics = _simulate([build_aes128_task(1024, 'one_request')], repeats=8)
    assert metrics['result_reuse_hits'] > 0


def test_distinct_requests_never_share_a_result():
    graphs = [build_aes128_task(1024, f'request_{i}') for i in range(8)]
    metrics = _simulate(graphs, repeats=1)
    assert metrics['result_reuse_hits'] == 0


def test_lookup_refuses_a_foreign_identity():
    fvm = FeatureVectorManager()
    graph = build_aes128_task(256, 'producer')
    micro = MicrofragmentDecomposer().decompose_tasks([graph]).microfragments[0]
    fvm.write(micro, cycle=0)
    assert fvm.lookup(micro.semantic_hash, 10, 1000, micro.task_id) is not None
    assert fvm.lookup(micro.semantic_hash, 10, 1000, 'someone_else') is None
    # An empty identity is unconstrained, for callers that supply none.
    assert fvm.lookup(micro.semantic_hash, 10, 1000) is not None


@pytest.mark.parametrize('mode', ['fuse', 'template_only', 'degraded'])
def test_identity_check_does_not_disturb_other_modes(mode):
    graph = build_aes128_task(512, 'any')
    decomposition = MicrofragmentDecomposer().decompose_tasks([graph])
    tasks = [RuntimeTask(runtime_id='x', task_graph=graph, arrival_cycle=0)]
    metrics = FuseSimulator(config=SimulationConfig(mode=mode)).run(
        tasks, decomposition.microfragments).metrics
    assert metrics['dispatches'] > 0
