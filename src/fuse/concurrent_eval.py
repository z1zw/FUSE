"""Concurrent three-domain secure inference.

A privacy-preserving language-model service does not get to choose whether it
runs AES and Groth16: encrypting the payload and attesting to the result are
obligations that arrive with every request.  The question this module asks is
therefore not which of the three is fastest, but what the two obligatory ones
cost the service -- and whether one reconfigurable fabric absorbs them better
than dedicated engines do.

Two things make the comparison mean something.

*The baseline is genuinely separate hardware.*  ``multicore`` in the simulator
shares the same RFU pool as FUSE and only pays a penalty factor, so it cannot
express "three engines that cannot help each other".  Here each domain is
simulated on its own slice of the RFU budget and the makespan is the slowest
engine, because they run concurrently on separate silicon.  The slice sizes are
chosen by searching every split and keeping the best one, so the baseline is
handed its optimal partition and FUSE's margin is a lower bound.

*The workload comes from public data.*  Three datasets supply three different
request shapes, and the volume of the security work is derived from the same
bytes rather than assumed: AES covers the prompt, the generated reply and the
retrieved context, and the Groth16 circuit is a Merkle inclusion proof whose
depth follows from the size of the retrieval corpus.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .branding import ACCELERATOR_NAME
from .ir import RuntimeTask, TaskGraph
from .reuse import MicrofragmentDecomposer
from .sim.simulator import CYCLES_PER_MS, FuseSimulator, SimulationConfig
from .workloads import build_aes128_task, build_groth16_like_task
from .workloads.llm import LLMArchSpec, build_llm_prefill_task

#: Public datasets, one per request shape.  They differ in what actually
#: stresses the array -- prompt length and generated length -- rather than in
#: subject matter.
DATASETS: List[Dict[str, Any]] = [
    {
        'name': 'gsm8k',
        'label': 'GSM8K',
        'shape': 'reasoning: medium prompt, long answer',
        'args': ('gsm8k', 'main'),
        'split': 'test',
        'field': 'question',
        'gen_tokens': 256,
    },
    {
        'name': 'mmlu',
        'label': 'MMLU',
        'shape': 'knowledge: short prompt, short answer',
        'args': ('cais/mmlu', 'all'),
        'split': 'test',
        'field': 'question',
        'gen_tokens': 32,
    },
    {
        'name': 'humaneval',
        'label': 'HumanEval',
        'shape': 'code: long prompt, long answer',
        'args': ('openai_humaneval',),
        'split': 'test',
        'field': 'prompt',
        'gen_tokens': 192,
    },
]

#: Architectures included in the sweep for their shape alone.  This study is
#: pure simulation, so it needs a model's configuration and not its weights,
#: which lets the range extend past what the local GPU could ever load.  They
#: are deliberately kept out of :data:`fuse.llm_zoo.LLM_ZOO`, which is a list of
#: models that really are executed.
SIMULATION_ONLY_MODELS: List[Dict[str, str]] = [
    {'name': 'qwen2_5_14b', 'label': 'Qwen2.5-14B', 'hf_id': 'Qwen/Qwen2.5-14B-Instruct'},
    {'name': 'qwen2_5_32b', 'label': 'Qwen2.5-32B', 'hf_id': 'Qwen/Qwen2.5-32B-Instruct'},
]


#: Retrieval corpus.  Its passages are the documents a private gateway decrypts
#: on the way in, and its size fixes the depth of the inclusion proof.
CORPUS = {'name': 'sciq', 'args': ('sciq',), 'split': 'test', 'field': 'support'}

#: Constraints for one Poseidon compression inside a Merkle path.  Poseidon is
#: the standard choice for in-circuit hashing precisely because it is cheap;
#: this is the widely quoted figure for a 2-to-1 compression over BN254.
POSEIDON_CONSTRAINTS_PER_LEVEL = 240

#: Retrieved passages per request.
DEFAULT_RETRIEVAL_DEPTH = 4

#: How far the fragment cache retains an entry, matched to the language-model
#: fragment lifetimes and applied identically to every configuration.
FRESHNESS_WINDOW_CYCLES = 600 * CYCLES_PER_MS

TOTAL_RFUS = 16

#: Static power of one idle RFU.  This mirrors ``static_coeff`` in
#: :meth:`FuseSimulator._estimate_energy`, which charges 0.02 nJ per cycle per
#: RFU; at the 1 GHz clock that is 0.02 W.  It has to come from there rather
#: than from the reported power breakdown, because the energy it is added to is
#: the simulator's, and mixing the two accountings would overstate the idle
#: cost of a partitioned design.
STATIC_POWER_W_PER_RFU = 0.02

BYTES_PER_AES_BLOCK = 16
BYTES_PER_TOKEN = 4


@dataclass
class RequestShape:
    """One dataset's request profile, measured from the data itself."""

    dataset: str
    label: str
    shape: str
    samples: int
    prompt_tokens: int
    gen_tokens: int
    payload_bytes: int
    retrieved_bytes: int
    corpus_size: int

    @property
    def aes_blocks(self) -> int:
        total = self.payload_bytes + self.retrieved_bytes + self.gen_tokens * BYTES_PER_TOKEN
        return max(1, math.ceil(total / BYTES_PER_AES_BLOCK))

    @property
    def zkp_constraints(self) -> int:
        depth = max(1, math.ceil(math.log2(max(self.corpus_size, 2))))
        return max(256, depth * POSEIDON_CONSTRAINTS_PER_LEVEL)


@dataclass
class ConcurrentArtifacts:
    shapes: pd.DataFrame
    results: pd.DataFrame
    scaling: pd.DataFrame
    environment: Dict[str, Any]


#: Hosted models asked how long an answer to each dataset actually is.  Their
#: architectures are not public, so they cannot be simulated on the fabric; what
#: they can supply is the generated length, which is otherwise a guess and which
#: sets both the decode cost and the volume of the payload to encrypt.
DEFAULT_ANSWER_MODELS: Tuple[str, ...] = (
    'gpt-5.4',
    'deepseek-v4-pro',
    'kimi-k2.6',
    'minimax-m3',
    'gemini-3.1-pro-preview',
)


def measure_answer_lengths(
    models: Sequence[str] = DEFAULT_ANSWER_MODELS,
    samples: int = 6,
    seed: int = 7,
    max_tokens: int = 2048,
    base_url: Optional[str] = None,
    token: Optional[str] = None,
    timeout_s: float = 180.0,
) -> Dict[str, Any]:
    """Ask hosted models the datasets' own questions and record answer lengths.

    Returns a record per dataset holding every model's mean, so the spread
    between them stays visible rather than being averaged away.  The credential
    comes from the environment and is never written into the result.
    """
    import os

    import requests
    from datasets import load_dataset

    base = (base_url or os.environ.get('FUSE_LLM_GATEWAY', '')).rstrip('/')
    key = token or os.environ.get('FUSE_LLM_TOKEN', '')
    if not (base and key):
        raise RuntimeError('set FUSE_LLM_GATEWAY and FUSE_LLM_TOKEN to measure answer lengths')
    headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}

    rng = np.random.default_rng(seed)
    out: Dict[str, Any] = {'models': list(models), 'samples': samples, 'max_tokens': max_tokens,
                           'datasets': {}}
    for spec in DATASETS:
        data = load_dataset(*spec['args'], split=spec['split'])
        take = min(samples, len(data))
        idx = rng.choice(len(data), size=take, replace=False)
        prompts = [str(data[int(i)][spec['field']]) for i in idx]
        per_model: Dict[str, Any] = {}
        for model in models:
            lengths: List[int] = []
            for prompt in prompts:
                try:
                    response = requests.post(
                        f'{base}/v1/chat/completions', headers=headers, timeout=timeout_s,
                        json={'model': model,
                              'messages': [{'role': 'user', 'content': prompt}],
                              'max_tokens': max_tokens, 'temperature': 0},
                    )
                    response.raise_for_status()
                    completion = response.json().get('usage', {}).get('completion_tokens')
                    if completion:
                        lengths.append(int(completion))
                except Exception as exc:  # one flaky model must not sink the measurement
                    print(f'    ! {model} on {spec["label"]}: {type(exc).__name__}', flush=True)
                    break
            if lengths:
                per_model[model] = {
                    'mean': float(np.mean(lengths)),
                    'median': float(np.median(lengths)),
                    'min': int(min(lengths)),
                    'max': int(max(lengths)),
                    'n': len(lengths),
                    'truncated': int(sum(1 for v in lengths if v >= max_tokens)),
                }
                print(f'    {model:24s} {spec["label"]:10s} median={per_model[model]["median"]:6.0f} '
                      f'tok (n={len(lengths)})', flush=True)
        if per_model:
            medians = [v['median'] for v in per_model.values()]
            out['datasets'][spec['name']] = {
                'label': spec['label'],
                'per_model': per_model,
                'median_of_models': float(np.median(medians)),
                'min_of_models': float(min(medians)),
                'max_of_models': float(max(medians)),
            }
    return out


def gen_tokens_from_measurement(measurement: Dict[str, Any], dataset: str, fallback: int) -> int:
    """Generated length for one dataset, as the median across the models asked.

    The median rather than the mean, because one model that runs to the token
    limit should not drag the figure with it.
    """
    record = measurement.get('datasets', {}).get(dataset)
    if not record:
        return fallback
    return max(8, int(round(float(record['median_of_models']))))


def _tokens_from_bytes(byte_len: int) -> int:
    """Rough token count when a tokenizer is not wanted; ~4 bytes per token."""
    return max(8, int(round(byte_len / 4.0)))


def load_request_shapes(
    sample_count: int = 64,
    retrieval_depth: int = DEFAULT_RETRIEVAL_DEPTH,
    seed: int = 7,
    tokenizer_id: Optional[str] = None,
    answer_lengths: Optional[Dict[str, Any]] = None,
) -> List[RequestShape]:
    """Measure each dataset's request profile, and the corpus the proof covers."""
    from datasets import load_dataset

    tokenizer = None
    if tokenizer_id:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    corpus = load_dataset(*CORPUS['args'], split=CORPUS['split'])
    passages = [str(corpus[i][CORPUS['field']]) for i in range(len(corpus))]
    passages = [p for p in passages if p.strip()]
    corpus_size = len(passages)
    mean_passage_bytes = float(np.mean([len(p.encode('utf-8')) for p in passages]))

    rng = np.random.default_rng(seed)
    shapes: List[RequestShape] = []
    for spec in DATASETS:
        data = load_dataset(*spec['args'], split=spec['split'])
        take = min(sample_count, len(data))
        idx = rng.choice(len(data), size=take, replace=False)
        texts = [str(data[int(i)][spec['field']]) for i in idx]
        byte_lens = [len(t.encode('utf-8')) for t in texts]
        if tokenizer is not None:
            tok_lens = [len(tokenizer(t)['input_ids']) for t in texts]
        else:
            tok_lens = [_tokens_from_bytes(b) for b in byte_lens]
        shapes.append(
            RequestShape(
                dataset=spec['name'],
                label=spec['label'],
                shape=spec['shape'],
                samples=take,
                prompt_tokens=int(round(float(np.mean(tok_lens)))),
                gen_tokens=(gen_tokens_from_measurement(answer_lengths, spec['name'], int(spec['gen_tokens']))
                            if answer_lengths else int(spec['gen_tokens'])),
                payload_bytes=int(round(float(np.mean(byte_lens)))),
                retrieved_bytes=int(round(retrieval_depth * mean_passage_bytes)),
                corpus_size=corpus_size,
            )
        )
    return shapes


class ConcurrentSecureSuite:
    """Runs the concurrent three-domain study."""

    def __init__(
        self,
        spec: LLMArchSpec,
        total_rfus: int = TOTAL_RFUS,
        span_ms: float = 20.0,
        width: int = 8,
        specialized_engines: bool = True,
        seed: int = 7,
    ):
        self.spec = spec
        self.total_rfus = total_rfus
        self.span_ms = span_ms
        self.width = width
        self.specialized_engines = specialized_engines
        self.seed = seed
        self._cache: Dict[Tuple[Any, ...], float] = {}

    # ------------------------------------------------------------- workload

    def domain_pairs(
        self, shape: RequestShape, llm_requests: int, security_per_request: int
    ) -> Dict[str, List[Tuple[float, TaskGraph]]]:
        """Arrivals per domain, spread over the window so all three are in flight."""
        out: Dict[str, List[Tuple[float, TaskGraph]]] = {'LLM': [], 'AES': [], 'ZKP': []}
        for i in range(llm_requests):
            out['LLM'].append(
                (
                    i * self.span_ms / max(llm_requests, 1),
                    build_llm_prefill_task(
                        self.spec,
                        seq_len=shape.prompt_tokens,
                        width=self.width,
                        content_tag=f'{shape.dataset}:{i}',
                        task_id=f'{shape.dataset}_llm{i}',
                    ),
                )
            )
        n_sec = llm_requests * max(0, security_per_request)
        for i in range(n_sec):
            arrival = i * self.span_ms / max(n_sec, 1)
            out['AES'].append(
                (arrival, build_aes128_task(blocks=shape.aes_blocks, task_id=f'{shape.dataset}_aes{i}'))
            )
            out['ZKP'].append(
                (
                    arrival,
                    build_groth16_like_task(
                        circuit_size=shape.zkp_constraints, rounds=4,
                        task_id=f'{shape.dataset}_zkp{i}',
                    ),
                )
            )
        return out

    # ------------------------------------------------------------ execution

    def _simulate(self, pairs: Sequence[Tuple[float, TaskGraph]], rfus: int, mode: str) -> Dict[str, Any]:
        if not pairs:
            return {'makespan_ms': 0.0, 'energy_nj': 0.0, 'rfu_utilization': 0.0,
                    'task_type_mean_latency_ms': {}, 'switch_cycles_total': 0.0, 'dispatches': 0}
        graphs = [g for _, g in pairs]
        tasks = [
            RuntimeTask(runtime_id=f'{g.task_id}_rt', task_graph=g, arrival_cycle=int(a * CYCLES_PER_MS))
            for a, g in pairs
        ]
        decomposition = MicrofragmentDecomposer().decompose_tasks(graphs)
        config = SimulationConfig(
            mode=mode, rfu_count=rfus, freshness_window_cycles=FRESHNESS_WINDOW_CYCLES
        )
        return FuseSimulator(config=config).run(tasks, decomposition.microfragments).metrics

    #: The fixed pipeline a dedicated engine would be built around.
    SPECIALIST_MODES = {'LLM': 'transformer_asic', 'AES': 'openaes', 'ZKP': 'static_zkp'}

    @classmethod
    def engine_mode(cls, domain: str, specialized: bool) -> str:
        """Datapath of one engine.

        Takes the choice as an argument rather than reading it off the instance,
        because a single sweep asks for both answers about the same workload.
        """
        return cls.SPECIALIST_MODES[domain] if specialized else 'fuse'

    def _engine_mode(self, domain: str) -> str:
        return self.engine_mode(domain, self.specialized_engines)

    # ------------------------------------------------------------------ run

    def evaluate(
        self, shape: RequestShape, llm_requests: int, security_per_request: int
    ) -> Dict[str, Any]:
        per_domain = self.domain_pairs(shape, llm_requests, security_per_request)
        shared = sorted(sum(per_domain.values(), []), key=lambda p: p[0])

        fuse = self._simulate(shared, self.total_rfus, 'fuse')
        llm_only = self._simulate(per_domain['LLM'], self.total_rfus, 'fuse')
        curves = self.domain_curves(per_domain, self.total_rfus, self.specialized_engines)
        ded_ms, ded_j, split = self.best_split(curves, self.total_rfus)

        per_type = fuse.get('task_type_mean_latency_ms', {})
        fuse_ms = float(fuse['makespan_ms'])
        return {
            'dataset': shape.dataset,
            'dataset_label': shape.label,
            'llm_requests': llm_requests,
            'security_per_request': security_per_request,
            'prompt_tokens': shape.prompt_tokens,
            'aes_blocks': shape.aes_blocks,
            'zkp_constraints': shape.zkp_constraints,
            'fuse_ms': fuse_ms,
            'fuse_energy_j': float(fuse['energy_nj']) / 1e9,
            'fuse_rfu_utilization': float(fuse['rfu_utilization']),
            'fuse_llm_latency_ms': float(per_type.get('LLM', 0.0)),
            'fuse_aes_latency_ms': float(per_type.get('AES', 0.0)),
            'fuse_zkp_latency_ms': float(per_type.get('ZKP', 0.0)),
            'llm_only_ms': float(llm_only['makespan_ms']),
            'security_tax': fuse_ms / max(float(llm_only['makespan_ms']), 1e-9) - 1.0,
            'dedicated_ms': ded_ms,
            'dedicated_energy_j': ded_j,
            'dedicated_split_llm': split[0],
            'dedicated_split_aes': split[1],
            'dedicated_split_zkp': split[2],
            'speedup_vs_dedicated': ded_ms / max(fuse_ms, 1e-9),
            'energy_gain_vs_dedicated': ded_j / max(float(fuse['energy_nj']) / 1e9, 1e-9),
        }

    def domain_curves(
        self,
        per_domain: Dict[str, List[Tuple[float, TaskGraph]]],
        max_rfus: int,
        specialized: bool,
    ) -> Dict[str, Dict[int, Tuple[float, float]]]:
        """Makespan of each domain alone, for every engine size up to ``max_rfus``.

        Computed once and reused for every array size, since a split of 16 and a
        split of 32 ask the same question of the same engine.
        """
        curves: Dict[str, Dict[int, Tuple[float, float]]] = {}
        for domain, pairs in per_domain.items():
            mode = self.engine_mode(domain, specialized)
            entries: Dict[int, Tuple[float, float]] = {}
            for rfus in range(1, max_rfus):
                metrics = self._simulate(pairs, rfus, mode)
                entries[rfus] = (float(metrics['makespan_ms']),
                                 float(metrics['energy_nj']) / 1e9)
            curves[domain] = entries
        return curves

    @staticmethod
    def best_split(
        curves: Dict[str, Dict[int, Tuple[float, float]]], total: int
    ) -> Tuple[float, float, Tuple[int, int, int]]:
        """The partition of ``total`` RFUs that the baseline would actually choose.

        Returns the makespan of the slowest engine, and the energy of all three.
        Engines that finish early keep burning static power until the request is
        done, which is the cost of owning silicon that only one domain can use,
        so the energy is charged over the whole span rather than each engine's
        own runtime.
        """
        best: Optional[Tuple[float, Tuple[int, int, int]]] = None
        for a in range(1, total - 1):
            for b in range(1, total - a):
                c = total - a - b
                if c < 1:
                    continue
                if a not in curves['LLM'] or b not in curves['AES'] or c not in curves['ZKP']:
                    continue
                span = max(curves['LLM'][a][0], curves['AES'][b][0], curves['ZKP'][c][0])
                if best is None or span < best[0]:
                    best = (span, (a, b, c))
        assert best is not None, f'no valid split for {total} RFUs'
        span, split = best
        energy = 0.0
        for domain, size in zip(('LLM', 'AES', 'ZKP'), split):
            engine_ms, engine_j = curves[domain][size]
            idle_ms = max(0.0, span - engine_ms)
            idle_j = STATIC_POWER_W_PER_RFU * size * idle_ms / 1000.0
            energy += engine_j + idle_j
        return span, energy, split

    def rfu_scaling(self, shape: RequestShape, llm_requests: int) -> pd.DataFrame:
        """How the language model alone scales with the array it is given.

        This is why partitioning hurts: every RFU reserved for a security engine
        is one the model cannot use when that engine is idle.
        """
        pairs = self.domain_pairs(shape, llm_requests, 0)['LLM']
        rows = []
        for rfus in range(1, self.total_rfus + 1):
            m = self._simulate(pairs, rfus, 'fuse')
            rows.append({'dataset': shape.dataset, 'dataset_label': shape.label,
                         'rfus': rfus, 'makespan_ms': float(m['makespan_ms']),
                         'rfu_utilization': float(m['rfu_utilization'])})
        return pd.DataFrame(rows)


def run_study(
    models: Sequence[LLMArchSpec],
    shapes: Sequence[RequestShape],
    output_root: Path,
    array_sizes: Sequence[int] = (4, 8, 12, 16, 24, 32),
    security_levels: Sequence[int] = (1, 4, 16),
    llm_requests: int = 2,
    span_ms: float = 20.0,
    width: int = 8,
    reference_rfus: int = TOTAL_RFUS,
    seed: int = 7,
) -> ConcurrentArtifacts:
    """Sweep request shape, model and array size.

    Two baselines are reported at every point.  ``partitioned`` gives each
    engine the same datapath as FUSE, so the only difference is that the array
    is carved up; that isolates the effect this study is about.  ``dedicated``
    additionally gives each engine the datapath advantage its domain would buy,
    which is more realistic but imports the speed-factor constants used
    elsewhere in the evaluation.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    array_sizes = sorted(set(int(n) for n in array_sizes))
    max_rfus = max(array_sizes)

    rows: List[Dict[str, Any]] = []
    scaling_frames: List[pd.DataFrame] = []

    for spec in models:
        for shape in shapes:
            ref = ConcurrentSecureSuite(spec=spec, total_rfus=max_rfus, span_ms=span_ms,
                                        width=width, specialized_engines=False, seed=seed)
            curve = ref.rfu_scaling(shape, llm_requests)
            curve['model'] = spec.name
            curve['model_label'] = spec.label
            scaling_frames.append(curve)
            print(f'[concurrent] {spec.label} x {shape.label}: prompt={shape.prompt_tokens} tok, '
                  f'AES={shape.aes_blocks} blk, ZKP={shape.zkp_constraints} constraints', flush=True)

            for level in security_levels:
                per_domain = ref.domain_pairs(shape, llm_requests, level)
                shared = sorted(sum(per_domain.values(), []), key=lambda p: p[0])
                curves = {
                    'partitioned': ref.domain_curves(per_domain, max_rfus, specialized=False),
                    'dedicated': ref.domain_curves(per_domain, max_rfus, specialized=True),
                }
                llm_only_pairs = ref.domain_pairs(shape, llm_requests, 0)['LLM']

                for rfus in array_sizes:
                    fuse = ref._simulate(shared, rfus, 'fuse')
                    fuse_ms = float(fuse['makespan_ms'])
                    llm_only_ms = float(ref._simulate(llm_only_pairs, rfus, 'fuse')['makespan_ms'])
                    per_type = fuse.get('task_type_mean_latency_ms', {})
                    record: Dict[str, Any] = {
                        'model': spec.name, 'model_label': spec.label,
                        'dataset': shape.dataset, 'dataset_label': shape.label,
                        'rfus': rfus, 'security_per_request': level,
                        'llm_requests': llm_requests,
                        'prompt_tokens': shape.prompt_tokens,
                        'aes_blocks': shape.aes_blocks,
                        'zkp_constraints': shape.zkp_constraints,
                        'fuse_ms': fuse_ms,
                        'fuse_energy_j': float(fuse['energy_nj']) / 1e9,
                        'fuse_rfu_utilization': float(fuse['rfu_utilization']),
                        'fuse_llm_latency_ms': float(per_type.get('LLM', 0.0)),
                        'fuse_aes_latency_ms': float(per_type.get('AES', 0.0)),
                        'fuse_zkp_latency_ms': float(per_type.get('ZKP', 0.0)),
                        'llm_only_ms': llm_only_ms,
                        'security_tax': fuse_ms / max(llm_only_ms, 1e-9) - 1.0,
                    }
                    for tag in ('partitioned', 'dedicated'):
                        span, energy, split = ConcurrentSecureSuite.best_split(curves[tag], rfus)
                        record[f'{tag}_ms'] = span
                        record[f'{tag}_energy_j'] = energy
                        record[f'energy_gain_vs_{tag}'] = energy / max(record['fuse_energy_j'], 1e-9)
                        record[f'{tag}_split'] = f'{split[0]}/{split[1]}/{split[2]}'
                        record[f'{tag}_reserved'] = rfus - split[0]
                        record[f'speedup_vs_{tag}'] = span / max(fuse_ms, 1e-9)
                    rows.append(record)
                    print(f'    sec x{level:<3d} RFUs={rfus:3d}  FUSE={fuse_ms:8.2f} ms  '
                          f'partitioned={record["partitioned_ms"]:8.2f} ({record["partitioned_split"]})  '
                          f'speedup={record["speedup_vs_partitioned"]:.2f}x', flush=True)

    results = pd.DataFrame(rows)
    scaling = pd.concat(scaling_frames, ignore_index=True) if scaling_frames else pd.DataFrame()
    shapes_df = pd.DataFrame([
        {
            'dataset': s.dataset, 'label': s.label, 'shape': s.shape, 'samples': s.samples,
            'prompt_tokens': s.prompt_tokens, 'gen_tokens': s.gen_tokens,
            'payload_bytes': s.payload_bytes, 'retrieved_bytes': s.retrieved_bytes,
            'corpus_size': s.corpus_size, 'aes_blocks': s.aes_blocks,
            'zkp_constraints': s.zkp_constraints,
        }
        for s in shapes
    ])
    environment = {
        'datasets': [d['name'] for d in DATASETS],
        'corpus': CORPUS['name'],
        'array_sizes': list(array_sizes),
        'security_levels': list(security_levels),
        'llm_requests': llm_requests,
        'span_ms': span_ms,
        'tiling_width': width,
        'reference_rfus': reference_rfus,
        'poseidon_constraints_per_level': POSEIDON_CONSTRAINTS_PER_LEVEL,
        'freshness_window_cycles': FRESHNESS_WINDOW_CYCLES,
        'models': [{'name': m.name, 'label': m.label, 'layers': m.layers, 'hidden': m.hidden,
                    'heads': m.heads, 'kv_heads': m.kv_heads, 'ff_dim': m.ff_dim,
                    'params_b': m.parameter_count() / 1e9} for m in models],
    }
    shapes_df.to_csv(output_root / 'request_shapes.csv', index=False)
    results.to_csv(output_root / 'concurrent_results.csv', index=False)
    scaling.to_csv(output_root / 'rfu_scaling.csv', index=False)
    (output_root / 'concurrent_env.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')

    artifacts = ConcurrentArtifacts(shapes_df, results, scaling, environment)
    if not results.empty:
        render_figure(artifacts, output_root / 'fig_concurrent_secure.pdf',
                      reference_rfus=reference_rfus)
    return artifacts


def render_figure(artifacts: ConcurrentArtifacts, out_path: Path,
                  reference_rfus: int = TOTAL_RFUS) -> None:
    """Why a reserved RFU is a loss, and when it stops being one."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    results, scaling = artifacts.results, artifacts.scaling
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.6, 4.2))
    palette = ['#E76F51', '#2A9D8F', '#264653', '#E9C46A', '#8E7DBE', '#B5651D']

    if not scaling.empty:
        first = scaling['dataset'].iloc[0]
        subset = scaling[scaling['dataset'] == first]
        top = 1.0
        for i, (label, part) in enumerate(subset.groupby('model_label', sort=False)):
            part = part.sort_values('rfus')
            base = float(part[part['rfus'] == 1]['makespan_ms'].iloc[0])
            speedup = base / part['makespan_ms']
            top = max(top, float(speedup.max()))
            ax_a.plot(part['rfus'], speedup, marker='o', markersize=3.6, linewidth=1.6,
                      color=palette[i % len(palette)], label=label)
        limit = int(subset['rfus'].max())
        ax_a.plot([1, limit], [1, limit], linestyle=':', linewidth=1.2, color='#777777',
                  label='perfect scaling')
        ax_a.set_ylim(0, max(top, 2.0) * 1.15)
        ax_a.set_xlabel('RFUs given to the language model')
        ax_a.set_ylabel('Speedup over a single RFU')
        ax_a.set_title('(a) The model stops being able to use more array')
        ax_a.legend(fontsize=7.6, frameon=False)
        ax_a.grid(alpha=0.25, linewidth=0.5)

    levels = sorted(results['security_per_request'].unique())
    level = levels[len(levels) // 2]
    part_all = results[results['security_per_request'] == level]
    for i, (label, part) in enumerate(part_all.groupby('model_label', sort=False)):
        grouped = part.groupby('rfus')['speedup_vs_partitioned'].mean().reset_index()
        ax_b.plot(grouped['rfus'], grouped['speedup_vs_partitioned'], marker='s', markersize=4.2,
                  linewidth=1.7, color=palette[i % len(palette)], label=label)
    ax_b.axhline(1.0, color='#333333', linewidth=0.9, linestyle='--')
    ax_b.axvline(reference_rfus, color='#999999', linewidth=0.9, linestyle=':')
    ax_b.text(reference_rfus, ax_b.get_ylim()[1], f' evaluated config ({reference_rfus})',
              fontsize=7, color='#666666', va='top', ha='left')
    ax_b.set_xscale('log', base=2)
    ax_b.set_xlabel('Total RFUs on the chip')
    ax_b.set_ylabel(f'{ACCELERATOR_NAME} vs its best fixed partition')
    ax_b.set_title(f'(b) Sharing pays while the array is scarce (security x{level})')
    ax_b.legend(fontsize=7.6, frameon=False)
    ax_b.grid(alpha=0.25, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
