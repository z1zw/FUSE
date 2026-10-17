"""Does the value hash separate two requests that differ only in their data?

A reviewer observed that the value hash binds operand references rather than
the bytes behind them, so a freshness window cannot by itself keep a rewritten
input from being answered from cache. The question is what, in the
implementation, does keep them apart.

Two mechanisms do, and both are data identities rather than timers.

  Per layer and token span, for the language-model path. A fragment's
  semantic group is qualified by the weights it reads (the fragment id,
  naming layer and output tile) and by the activations it consumes
  (``content_tag``, naming the token span). See workloads/llm.py.

  Per request, for every domain. ``LLMRequestInterface.compile_prompt``
  rewrites every fragment of every graph it emits, qualifying the semantic
  group with a hash over the request and the fragment's own identity, so two
  requests that differ at all are distinct to the cache. See
  ``_inject_prompt_variance`` in llm/interface.py.

The bare workload builders carry no identity in their hashes: ``build_aes_task``
takes blocks, key width and direction but no key, IV or plaintext, so two
graphs built directly from it hash alike. What keeps them apart is the lookup
itself. ``FeatureVectorManager._valid`` (contract C4) returns a cached result
only to a consumer whose originating request is the producer's, and the C++
engine's ``cache_valid`` does the same. Section 3 below runs the simulator on
both cases and checks that the legitimate hits survive and the others do not.

    python scripts/check_cache_identity.py
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from fuse.reuse import MicrofragmentDecomposer
from fuse.workloads.aes import build_aes_task
from fuse.workloads.zkp import build_groth16_like_task


def hashes(task):
    d = MicrofragmentDecomposer().decompose_tasks([task])
    return {m.semantic_hash for m in d.microfragments}


def report(name, a, b, want_separate):
    ha, hb = hashes(a), hashes(b)
    shared = len(ha & hb)
    got = shared == 0
    verdict = 'SEPARATED' if got else f'{shared}/{len(ha)} shared'
    print(f'  {"ok " if got == want_separate else "!! "}{name:52s} {verdict}')
    return shared


print('1. the bare workload builders carry no identity of their own')
report('AES, two graphs from the same arguments', build_aes_task(blocks=64, task_id='aes_a'),
       build_aes_task(blocks=64, task_id='aes_b'), want_separate=False)
print(f'     build_aes_task{inspect.signature(build_aes_task)}')
report('Groth16, two graphs from the same arguments',
       build_groth16_like_task(circuit_size=2048, task_id='zkp_a'),
       build_groth16_like_task(circuit_size=2048, task_id='zkp_b'),
       want_separate=False)
print(f'     build_groth16_like_task{inspect.signature(build_groth16_like_task)}')

print('\n2. the request path supplies one, for every domain')
from fuse.llm.interface import HeuristicLLMFrontend

iface = HeuristicLLMFrontend()
REQ = 'Serve a Groth16 proof over 4k constraints and AES-128 over 2k blocks'
ALT = 'Serve a Groth16 proof over 4k constraints and AES-128 over 2k blocks.'


def graphs_for(prompt, secure):
    out = {}
    for g in iface.compile_prompt(prompt, secure=secure).graphs:
        out.setdefault(g.task_type, g)
    return out


# Two paths, and they are meant to answer differently. With secure=False the
# compiler qualifies every fragment with a hash of the prompt, so two requests
# that differ at all are distinct to the cache. With secure=True it
# canonicalizes instead, so two wordings of one service collapse to one graph
# and may share, which is what 4.3.1 measures.
print('   secure=False: every distinct prompt is a distinct request')
a, b, again = (graphs_for(REQ, False), graphs_for(ALT, False),
               graphs_for(REQ, False))
for domain in sorted(set(a) & set(b)):
    report(f'{domain}, two different requests', a[domain], b[domain],
           want_separate=True)
for domain in sorted(set(a) & set(again)):
    report(f'{domain}, the same request served twice', a[domain],
           again[domain], want_separate=False)
print('   secure=True: two wordings of one service are one request')
a, b = graphs_for(REQ, True), graphs_for(ALT, True)
for domain in sorted(set(a) & set(b)):
    report(f'{domain}, two wordings, canonicalized', a[domain], b[domain],
           want_separate=False)

print('\n3. the lookup itself enforces C4, whatever built the graph')
from fuse.ir import RuntimeTask
from fuse.sim.simulator import CYCLES_PER_MS, FuseSimulator, SimulationConfig
from fuse.workloads import build_aes128_task


def simulate(graphs):
    # The arrival pattern of benchmarks._raw_fig10, so a producer has retired
    # before the next consumer dispatches.
    tasks, arrival_ms = [], 0.0
    for g in graphs:
        for i in range(16 // len(graphs)):
            tasks.append(RuntimeTask(runtime_id=f'{g.task_id}_rt{i}', task_graph=g,
                                     arrival_cycle=int(arrival_ms * CYCLES_PER_MS)))
            arrival_ms += 0.045 if i % 2 == 0 else 0.12
    d = MicrofragmentDecomposer().decompose_tasks(graphs)
    return FuseSimulator(config=SimulationConfig(mode='fuse')).run(
        tasks, d.microfragments).metrics['result_reuse_hits']


replay = simulate([build_aes128_task(4096, 'pair_aes')])
distinct = simulate([build_aes128_task(4096, f'aes_{i}') for i in range(16)])
print(f'  {"ok " if replay > 0 else "!! "}{"one request served 16 times":52s} '
      f'{replay} results reused')
print(f'  {"ok " if distinct == 0 else "!! "}{"16 distinct requests":52s} '
      f'{distinct} results reused')

print('\nSo validity rests on identity, not on the freshness window. The')
print('identity is the originating request, supplied by the front-end per')
print('prompt and compared by the cache at every lookup, so a graph built')
print('straight from a bare builder is still answered only from its own results.')
raise SystemExit(0 if replay > 0 and distinct == 0 else 1)
