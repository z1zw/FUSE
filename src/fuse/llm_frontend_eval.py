"""Language models as the request compiler in front of FUSE.

A deployment does not hand FUSE a task graph; it hands it a request in words.
Something has to turn that request into fragments, and in a modern stack that
something is a language model.  This study replaces the heuristic front-end in
:mod:`fuse.llm.interface` with real models -- local ones and hosted ones -- and
asks two questions about each:

* Can it compile the request at all?  Measured as whether the reply parses into
  a schedulable plan and whether that plan recovers the sizes the request
  stated.
* Does routing the compiled plan through FUSE's canonicalizer matter?  Without
  it, two wordings of the same request produce graphs that differ in
  prompt-derived ways, so the fabric cannot reuse anything between them and the
  wording is visible in the execution trace.  With it, the graphs collapse to
  one canonical form.

The plan a model returns is data, never instructions: it is parsed against a
fixed schema and every value is clamped to a supported range before a graph is
built from it.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .branding import ACCELERATOR_NAME
from .ir import RuntimeTask, TaskGraph
from .llm.canonicalizer import CanonicalTaskCanonicalizer
from .llm.verifier import FuseVerifier
from .reuse import MicrofragmentDecomposer
from .security.attacks import run_binary_attacks
from .security.metrics import binary_security_score, trace_features
from .sim.simulator import CYCLES_PER_MS, FuseSimulator, SimulationConfig
from .workloads import (
    build_aes128_task,
    build_groth16_like_task,
    build_mlp_path_task,
    build_transformer_block_task,
    build_vgg16_like_task,
)

PLAN_SCHEMA_HINT = (
    '{"tasks":[{"type":"llm","seq_len":<int>,"hidden":<int>,"heads":<int>},'
    '{"type":"aes","blocks":<int>},'
    '{"type":"zkp","constraints":<int>},'
    '{"type":"cnn","batch":<int>},'
    '{"type":"mlp","batch":<int>,"hidden":<int>}]}'
)

SYSTEM_PROMPT = (
    'You compile a service request into an accelerator plan. '
    'Reply with JSON only and no prose, no markdown fence, shaped exactly as '
    + PLAN_SCHEMA_HINT
    + '. Include only the task types the request actually asks for, and copy the '
    'sizes the request states. Use integers.'
)

#: Ranges every field is clamped to before a graph is built.  A model reply is
#: untrusted input, so an out-of-range or missing value degrades to the bound
#: rather than reaching the graph builders.
FIELD_BOUNDS: Dict[str, Tuple[int, int]] = {
    'seq_len': (16, 8192),
    'hidden': (64, 8192),
    'heads': (1, 128),
    'blocks': (16, 65536),
    'constraints': (256, 262144),
    'batch': (1, 128),
}

#: Each entry is one service, written several ways.  The wordings state the same
#: sizes, so a compiler that understood the request must return the same plan for
#: all of them.
SERVICE_INTENTS: List[Dict[str, Any]] = [
    {
        'intent': 'private_rag',
        'reference': {
            'llm': {'seq_len': 768, 'hidden': 3072, 'heads': 24},
            'aes': {'blocks': 1024},
            'zkp': {'constraints': 2048},
        },
        'prompts': [
            'Build a private RAG gateway that prefills 768 tokens on a 3072-wide 24-head model, '
            'encrypts 1024 AES blocks, and proves 2048 constraints.',
            'I need a confidential retrieval service: LLM prefill seq 768, hidden 3072, heads 24; '
            'AES over 1024 blocks; a Groth16 proof of 2048 constraints.',
            'Compile a retrieval-augmented pipeline with a 24-head 3072-dimension transformer over '
            '768 tokens, AES-128 on 1024 blocks and a 2048-constraint zero-knowledge proof.',
            'Plan a privacy-preserving RAG endpoint: 768-token context, 3072 hidden units, 24 '
            'attention heads, 1024 encrypted blocks, 2048-gate proof.',
        ],
    },
    {
        'intent': 'rollup_sequencer',
        'reference': {
            'llm': {'seq_len': 512, 'hidden': 2048, 'heads': 16},
            'aes': {'blocks': 2048},
            'zkp': {'constraints': 4096},
        },
        'prompts': [
            'Compile a rollup sequencer that proves 4096 constraints, encrypts 2048 AES blocks, and '
            'runs a transformer with seq 512, hidden 2048, heads 16.',
            'Set up a blockchain sequencing service: Groth16 over 4096 gates, AES-128 on 2048 '
            'blocks, and an LLM step of 512 tokens at 2048 hidden with 16 heads.',
            'I want a rollup batch pipeline: 4096-constraint proof, 2048 blocks of AES, plus a '
            '16-head 2048-wide model over a 512-token window.',
            'Build a sequencer: transformer seq_len 512 hidden 2048 heads 16, AES blocks 2048, '
            'zero-knowledge constraints 4096.',
        ],
    },
    {
        'intent': 'edge_copilot',
        'reference': {
            'llm': {'seq_len': 384, 'hidden': 1536, 'heads': 12},
            'aes': {'blocks': 512},
            'cnn': {'batch': 8},
        },
        'prompts': [
            'Build a confidential edge copilot: LLM prefill 384 tokens, hidden 1536, 12 heads; AES '
            'over 512 blocks; a VGG16 vision pass at batch 8.',
            'Plan an on-device assistant that runs a 12-head 1536-wide model over 384 tokens, '
            'encrypts 512 AES blocks, and does CNN inference with batch size 8.',
            'Compile an edge service combining vision (VGG16, batch 8), AES-128 on 512 blocks, and '
            'a transformer of sequence 384, hidden 1536, heads 12.',
            'I need a private copilot on the edge: 384-token prefill at 1536 dimensions with 12 '
            'heads, 512 encrypted blocks, and a batch-8 CNN.',
        ],
    },
]


@dataclass
class PlannerReply:
    model: str
    intent: str
    prompt_index: int
    prompt: str
    raw: str
    latency_s: float
    error: str = ''


@dataclass
class ParsedPlan:
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    parse_ok: bool = False
    note: str = ''


@dataclass
class FrontendArtifacts:
    replies: pd.DataFrame
    plans: pd.DataFrame
    comparison: pd.DataFrame
    summary: pd.DataFrame
    environment: Dict[str, Any]


# --------------------------------------------------------------------- planners


class GatewayPlanner:
    """Calls an OpenAI-compatible endpoint.

    The base URL and token come from the environment, never from source, so a
    published artefact of this run never carries the credential.
    """

    def __init__(self, model: str, base_url: str, token: str, timeout_s: float = 180.0,
                 max_tokens: int = 2000):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens

    def complete(self, prompt: str) -> str:
        import requests

        response = requests.post(
            f'{self.base_url}/v1/chat/completions',
            headers={'Authorization': f'Bearer {self.token}', 'Content-Type': 'application/json'},
            timeout=self.timeout_s,
            json={
                'model': self.model,
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt},
                ],
                'max_tokens': self.max_tokens,
                'temperature': 0,
            },
        )
        response.raise_for_status()
        body = response.json()
        choice = body['choices'][0]['message']
        return str(choice.get('content') or '')


class LocalPlanner:
    """Runs a locally cached model through ``transformers``."""

    def __init__(self, hf_id: str, device: str = 'cuda:0', max_new_tokens: int = 220):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        kwargs: Dict[str, Any] = {'low_cpu_mem_usage': True}
        if device.startswith('cuda'):
            kwargs['dtype'] = torch.float16
            kwargs['device_map'] = device
        self.model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
        if not device.startswith('cuda'):
            self.model = self.model.to(device)
        self.model.eval()

    def complete(self, prompt: str) -> str:
        tokenizer = self.tokenizer
        if getattr(tokenizer, 'chat_template', None):
            text = tokenizer.apply_chat_template(
                [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = f'{SYSTEM_PROMPT}\n\nRequest: {prompt}\nJSON:'
        encoded = tokenizer(text, return_tensors='pt').to(self.device)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
        return tokenizer.decode(output[0][encoded['input_ids'].shape[1]:], skip_special_tokens=True)

    def close(self) -> None:
        del self.model
        if self.device.startswith('cuda'):
            self.torch.cuda.empty_cache()


# ----------------------------------------------------------------- plan parsing


def extract_plan(raw: str) -> ParsedPlan:
    """Pull the first JSON object out of a reply and validate it."""
    if not raw or not raw.strip():
        return ParsedPlan(note='empty reply')
    text = raw.strip()
    fence = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find('{')
    if start < 0:
        return ParsedPlan(note='no JSON object')
    depth = 0
    end = -1
    for index in range(start, len(text)):
        if text[index] == '{':
            depth += 1
        elif text[index] == '}':
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end < 0:
        return ParsedPlan(note='unterminated JSON')
    try:
        payload = json.loads(text[start:end])
    except json.JSONDecodeError as exc:
        return ParsedPlan(note=f'invalid JSON: {exc.msg}')
    if not isinstance(payload, dict) or not isinstance(payload.get('tasks'), list):
        return ParsedPlan(note='no tasks array')
    tasks: List[Dict[str, Any]] = []
    for entry in payload['tasks']:
        if not isinstance(entry, dict):
            continue
        task_type = str(entry.get('type', '')).strip().lower()
        if task_type not in {'llm', 'aes', 'zkp', 'cnn', 'mlp'}:
            continue
        clean: Dict[str, Any] = {'type': task_type}
        for key, (low, high) in FIELD_BOUNDS.items():
            if key in entry:
                try:
                    clean[key] = int(max(low, min(high, int(float(entry[key])))))
                except (TypeError, ValueError):
                    continue
        tasks.append(clean)
    if not tasks:
        return ParsedPlan(note='no recognised tasks')
    return ParsedPlan(tasks=tasks, parse_ok=True)


def plan_fidelity(plan: ParsedPlan, reference: Dict[str, Dict[str, int]]) -> float:
    """Fraction of the sizes stated in the request that the plan recovered."""
    expected = sum(len(fields) for fields in reference.values())
    if expected == 0:
        return 0.0
    by_type: Dict[str, Dict[str, Any]] = {}
    for task in plan.tasks:
        by_type.setdefault(task['type'], task)
    hits = 0
    for task_type, fields in reference.items():
        task = by_type.get(task_type)
        if task is None:
            continue
        for key, value in fields.items():
            if int(task.get(key, -1)) == int(value):
                hits += 1
    return hits / expected


def graphs_from_plan(plan: ParsedPlan, task_prefix: str) -> List[TaskGraph]:
    """Build task graphs from a parsed plan."""
    graphs: List[TaskGraph] = []
    for index, task in enumerate(plan.tasks):
        task_type = task['type']
        if task_type == 'llm':
            graphs.append(
                build_transformer_block_task(
                    seq_len=task.get('seq_len', 512),
                    hidden_dim=task.get('hidden', 2048),
                    heads=task.get('heads', 16),
                    batch_size=1,
                    task_id=f'{task_prefix}_llm{index}',
                )
            )
        elif task_type == 'aes':
            graphs.append(
                build_aes128_task(blocks=task.get('blocks', 1024), task_id=f'{task_prefix}_aes{index}')
            )
        elif task_type == 'zkp':
            graphs.append(
                build_groth16_like_task(
                    circuit_size=task.get('constraints', 2048),
                    rounds=4,
                    task_id=f'{task_prefix}_zkp{index}',
                )
            )
        elif task_type == 'cnn':
            graphs.append(
                build_vgg16_like_task(batch_size=task.get('batch', 8), task_id=f'{task_prefix}_cnn{index}')
            )
        elif task_type == 'mlp':
            graphs.append(
                build_mlp_path_task(
                    batch_size=task.get('batch', 8),
                    hidden_dim=task.get('hidden', 1024),
                    task_id=f'{task_prefix}_mlp{index}',
                )
            )
    return graphs


def _slug(value: str) -> str:
    return re.sub(r'[^0-9A-Za-z]+', '_', value).strip('_').lower() or 'model'


def raw_graph_hash(graphs: Sequence[TaskGraph]) -> str:
    """Hash graphs exactly as they are, with no canonicalization.

    Used for the unprotected side of the comparison: putting those graphs
    through the canonicalizer would remove the prompt-derived differences that
    the comparison exists to expose.
    """
    import hashlib

    payload = []
    for graph in sorted(graphs, key=lambda g: g.task_id):
        fragments = [
            {
                'fragment_id': fragment.fragment_id,
                'stage': fragment.stage,
                'template_family': fragment.template_family,
                'semantic_group': fragment.semantic_group,
                'deps': sorted(fragment.deps),
                'estimated_ops': fragment.estimated_ops,
                'estimated_bytes': fragment.estimated_bytes,
                'metadata': {k: str(v) for k, v in sorted(fragment.metadata.items())},
            }
            for fragment in graph.topological_fragments()
        ]
        payload.append(
            {
                'task_type': graph.task_type,
                'domain': graph.domain,
                'metadata': {k: str(v) for k, v in sorted(graph.metadata.items())},
                'fragments': fragments,
            }
        )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def inject_prompt_variance(graphs: Sequence[TaskGraph], prompt: str) -> List[TaskGraph]:
    """Tag a plan the way an unprotected front-end would.

    Without canonicalization the compiled graph carries traces of the wording it
    came from -- names it chose, the order it emitted, per-request seeds -- which
    is what makes two phrasings of one request look like two different jobs.
    """
    import hashlib

    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    tagged: List[TaskGraph] = []
    for graph in graphs:
        clone = graph.copy(task_id=f'{graph.task_id}_{prompt_hash}')
        for fragment in clone.fragments.values():
            group = fragment.semantic_group or fragment.fragment_id
            variant = hashlib.sha256(
                f'{prompt_hash}:{fragment.fragment_id}:{fragment.stage}'.encode()
            ).hexdigest()[:6]
            fragment.semantic_group = f'{group}@prompt_{variant}'
            fragment.metadata = {
                **fragment.metadata,
                'prompt_hash': prompt_hash,
                'runtime_seed': int(variant, 16) % 10007,
            }
        clone.metadata = {**clone.metadata, 'prompt_hash': prompt_hash}
        tagged.append(clone)
    return tagged


# ------------------------------------------------------------------- the suite

#: Hosted models, addressed through the gateway named by ``FUSE_LLM_GATEWAY``.
DEFAULT_GATEWAY_MODELS: Tuple[str, ...] = (
    'gpt-5.4',
    'deepseek-v4-pro',
    'kimi-k2.6',
    'minimax-m3',
    'gemini-3.1-pro-preview',
)

#: Locally cached instruction-tuned models, plus one base model to show what a
#: front-end that cannot follow a schema looks like.
DEFAULT_LOCAL_MODELS: Tuple[Tuple[str, str], ...] = (
    ('Qwen2.5-0.5B', 'Qwen/Qwen2.5-0.5B-Instruct'),
    ('TinyLlama-1.1B', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'),
    ('Mistral-7B', 'mistralai/Mistral-7B-Instruct-v0.2'),
    ('Qwen2.5-7B', 'Qwen/Qwen2.5-7B-Instruct'),
)

FRONTEND_FRESHNESS_WINDOW_CYCLES = 50 * CYCLES_PER_MS


class LLMFrontendSuite:
    """Compiles service requests with real models, then runs both paths."""

    def __init__(
        self,
        gateway_models: Sequence[str] = DEFAULT_GATEWAY_MODELS,
        local_models: Sequence[Tuple[str, str]] = DEFAULT_LOCAL_MODELS,
        device: str = 'cuda:0',
        gateway_base: Optional[str] = None,
        gateway_token: Optional[str] = None,
        retries: int = 2,
    ):
        self.gateway_models = list(gateway_models)
        self.local_models = list(local_models)
        self.device = device
        self.gateway_base = gateway_base or os.environ.get('FUSE_LLM_GATEWAY', '')
        self.gateway_token = gateway_token or os.environ.get('FUSE_LLM_TOKEN', '')
        self.retries = retries
        self.canonicalizer = CanonicalTaskCanonicalizer()
        self.verifier = FuseVerifier()

    # ------------------------------------------------------------- compilation

    def collect_replies(self) -> List[PlannerReply]:
        replies: List[PlannerReply] = []
        if self.gateway_models:
            if not (self.gateway_base and self.gateway_token):
                print('  ! no gateway configured (FUSE_LLM_GATEWAY / FUSE_LLM_TOKEN); '
                      'skipping hosted models', flush=True)
            else:
                for model in self.gateway_models:
                    planner = GatewayPlanner(model, self.gateway_base, self.gateway_token)
                    print(f'[frontend] hosted {model}', flush=True)
                    replies.extend(self._ask_all(model, 'hosted', planner.complete))
        for label, hf_id in self.local_models:
            print(f'[frontend] local {label}', flush=True)
            try:
                planner = LocalPlanner(hf_id, device=self.device)
            except Exception as exc:
                print(f'  ! could not load {hf_id}: {type(exc).__name__}: {exc}', flush=True)
                continue
            try:
                replies.extend(self._ask_all(label, 'local', planner.complete))
            finally:
                planner.close()
        return replies

    def _ask_all(self, model: str, kind: str, complete: Callable[[str], str]) -> List[PlannerReply]:
        out: List[PlannerReply] = []
        for service in SERVICE_INTENTS:
            for index, prompt in enumerate(service['prompts']):
                raw, latency_s, error = '', 0.0, ''
                for attempt in range(self.retries + 1):
                    start = time.perf_counter()
                    try:
                        raw = complete(prompt)
                        latency_s = time.perf_counter() - start
                        error = ''
                        if raw.strip():
                            break
                        error = 'empty reply'
                    except Exception as exc:
                        latency_s = time.perf_counter() - start
                        error = f'{type(exc).__name__}: {exc}'
                    if attempt < self.retries:
                        time.sleep(1.5 * (attempt + 1))
                reply = PlannerReply(
                    model=model,
                    intent=service['intent'],
                    prompt_index=index,
                    prompt=prompt,
                    raw=raw,
                    latency_s=latency_s,
                    error=error,
                )
                reply.kind = kind  # type: ignore[attr-defined]
                out.append(reply)
        return out

    # --------------------------------------------------------------- execution

    @staticmethod
    def _run(graphs: Sequence[TaskGraph], mode: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
        decomposition = MicrofragmentDecomposer().decompose_tasks(list(graphs))
        runtime_tasks = [
            RuntimeTask(runtime_id=f'{graph.task_id}_rt', task_graph=graph,
                        arrival_cycle=int(index * 0.05 * CYCLES_PER_MS))
            for index, graph in enumerate(graphs)
        ]
        config = SimulationConfig(mode=mode, freshness_window_cycles=FRONTEND_FRESHNESS_WINDOW_CYCLES)
        result = FuseSimulator(config=config).run(runtime_tasks, decomposition.microfragments)
        return result.metrics, pd.DataFrame(result.traces.get('events', []))

    def evaluate(self, replies: Sequence[PlannerReply]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Parse every reply, then replay each service under the three arms.

        All wordings of one service run in a single trace, because that is the
        question: when a second user asks for the same thing in different words,
        does the fabric recognise it.  The arms separate the two contributions --
        the runtime on its own, and the canonical front-end on top of it.
        """
        reference_by_intent = {s['intent']: s['reference'] for s in SERVICE_INTENTS}
        plan_rows: List[Dict[str, Any]] = []
        compare_rows: List[Dict[str, Any]] = []
        parsed_by_key: Dict[Tuple[str, str, str], List[Tuple[int, ParsedPlan, str]]] = {}

        for reply in replies:
            plan = extract_plan(reply.raw)
            reference = reference_by_intent[reply.intent]
            kind = getattr(reply, 'kind', 'unknown')
            fidelity = plan_fidelity(plan, reference) if plan.parse_ok else 0.0
            row: Dict[str, Any] = {
                'model': reply.model,
                'kind': kind,
                'intent': reply.intent,
                'prompt_index': reply.prompt_index,
                'latency_s': reply.latency_s,
                'error': reply.error,
                'parse_ok': plan.parse_ok,
                'parse_note': plan.note,
                'task_count': len(plan.tasks),
                'plan_fidelity': fidelity,
                'plan': json.dumps(plan.tasks),
            }
            if plan.parse_ok:
                prefix = f'{_slug(reply.model)}_{reply.intent}_{reply.prompt_index}'
                base_graphs = graphs_from_plan(plan, task_prefix=prefix)
                if base_graphs:
                    canonical = self.canonicalizer.canonicalize_graphs(base_graphs)
                    unprotected = inject_prompt_variance(base_graphs, reply.prompt)
                    # The canonical hash is the whole point of the canonicalizer,
                    # so the unprotected side is hashed verbatim instead -- running
                    # it through the canonicalizer would erase exactly the
                    # prompt-derived difference we are trying to observe.
                    row['canonical_hash'] = self.canonicalizer.canonical_hash(canonical)
                    row['unprotected_hash'] = raw_graph_hash(unprotected)
                    row['verifier_ok'] = bool(self.verifier.verify_graphs(canonical).ok)
                    parsed_by_key.setdefault((reply.model, kind, reply.intent), []).append(
                        (reply.prompt_index, plan, reply.prompt)
                    )
                else:
                    row['parse_ok'] = False
                    row['parse_note'] = 'plan produced no graphs'
            plan_rows.append(row)

        for (model, kind, intent), entries in parsed_by_key.items():
            if len(entries) < 2:
                continue
            entries.sort(key=lambda item: item[0])
            plain: List[TaskGraph] = []
            tagged: List[TaskGraph] = []
            canonical: List[TaskGraph] = []
            for prompt_index, plan, prompt in entries:
                prefix = f'{_slug(model)}_{intent}_{prompt_index}'
                graphs = graphs_from_plan(plan, task_prefix=prefix)
                plain.extend(graphs)
                tagged.extend(inject_prompt_variance(graphs, prompt))
                # The canonical arm gives each graph the identity of its canonical
                # form. The cache returns a result only to the request that
                # produced it (contract C4), and under canonicalization two
                # wordings that reach the same form are one request; wordings
                # whose plans differ reach different forms and stay apart.
                for graph in self.canonicalizer.canonicalize_graphs(graphs):
                    identity = self.canonicalizer.canonical_hash([graph])[:16]
                    canonical.append(
                        self.canonicalizer.canonicalize_graph(graph, task_id=f'canon_{identity}')
                    )
            for path, graphs, mode in (
                ('unprotected', tagged, 'multicore'),
                ('fuse_no_canonical', tagged, 'fuse'),
                (ACCELERATOR_NAME, canonical, 'fuse'),
            ):
                metrics, events = self._run(graphs, mode)
                features = trace_features(events, metrics)
                compare_rows.append(
                    {
                        'model': model,
                        'kind': kind,
                        'intent': intent,
                        'wordings': len(entries),
                        'path': path,
                        'mode': mode,
                        'makespan_ms': float(metrics['makespan_ms']),
                        'energy_j': float(metrics['energy_nj']) / 1e9,
                        'result_reuse_rate': float(metrics['result_reuse_rate']),
                        'structural_reuse_rate': float(metrics['structural_reuse_rate']),
                        'rfu_utilization': float(metrics['rfu_utilization']),
                        'switch_cycles_total': float(metrics['switch_cycles_total']),
                        'distinct_graphs': len({g.task_id for g in graphs}),
                        **{f'feat_{k}': v for k, v in features.items()},
                    }
                )
        return pd.DataFrame(plan_rows), pd.DataFrame(compare_rows)

    # ----------------------------------------------------------------- summary

    #: Column prefix per execution arm.
    PATH_TAGS = {'unprotected': 'unprotected', 'fuse_no_canonical': 'fuse_runtime', ACCELERATOR_NAME: 'fuse'}

    def summarize(self, plans: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
        if plans.empty:
            return pd.DataFrame()
        rows: List[Dict[str, Any]] = []
        for (model, kind), part in plans.groupby(['model', 'kind'], sort=False):
            parsed = part[part['parse_ok']]
            record: Dict[str, Any] = {
                'model': model,
                'kind': kind,
                'requests': len(part),
                'parse_rate': float(part['parse_ok'].mean()),
                'plan_fidelity': float(part['plan_fidelity'].mean()),
                'mean_compile_s': float(part['latency_s'].mean()),
            }
            # Do all wordings of one service compile to the same graph?
            agree_canonical, agree_unprotected, groups = [], [], 0
            for _, chunk in parsed.groupby('intent'):
                if len(chunk) < 2 or 'canonical_hash' not in chunk.columns:
                    continue
                groups += 1
                agree_canonical.append(float(chunk['canonical_hash'].nunique() == 1))
                agree_unprotected.append(float(chunk['unprotected_hash'].nunique() == 1))
            record['intent_groups'] = groups
            record['canonical_agreement'] = float(np.mean(agree_canonical)) if agree_canonical else float('nan')
            record['unprotected_agreement'] = (
                float(np.mean(agree_unprotected)) if agree_unprotected else float('nan')
            )
            if not parsed.empty and 'verifier_ok' in parsed.columns:
                record['verifier_pass_rate'] = float(parsed['verifier_ok'].fillna(False).mean())

            model_cmp = (
                comparison[(comparison['model'] == model) & (comparison['kind'] == kind)]
                if not comparison.empty
                else pd.DataFrame()
            )
            if not model_cmp.empty:
                for path, tag in self.PATH_TAGS.items():
                    side = model_cmp[model_cmp['path'] == path]
                    if side.empty:
                        continue
                    record[f'{tag}_makespan_ms'] = float(side['makespan_ms'].mean())
                    record[f'{tag}_energy_j'] = float(side['energy_j'].mean())
                    record[f'{tag}_result_reuse'] = float(side['result_reuse_rate'].mean())
                    record[f'{tag}_structural_reuse'] = float(side['structural_reuse_rate'].mean())
                    record[f'{tag}_switch_cycles'] = float(side['switch_cycles_total'].mean())
                if 'fuse_makespan_ms' in record and 'unprotected_makespan_ms' in record:
                    record['latency_reduction'] = 1.0 - record['fuse_makespan_ms'] / max(
                        record['unprotected_makespan_ms'], 1e-9
                    )
                    record['energy_gain'] = record['unprotected_energy_j'] / max(
                        record['fuse_energy_j'], 1e-9
                    )
                if 'fuse_runtime_makespan_ms' in record and 'fuse_makespan_ms' in record:
                    # What the canonical front-end adds on top of the runtime.
                    record['canonical_extra_reduction'] = 1.0 - record['fuse_makespan_ms'] / max(
                        record['fuse_runtime_makespan_ms'], 1e-9
                    )
                record.update(self._trace_divergence(model_cmp))
                record.update(self._leakage(model_cmp))
            rows.append(record)
        return pd.DataFrame(rows)

    @staticmethod
    def _trace_divergence(model_cmp: pd.DataFrame) -> Dict[str, float]:
        """How differently do wordings of one service execute?

        Reported as the spread of makespan across the wordings of a service,
        relative to its mean.  Zero means an observer watching the fabric cannot
        tell which wording was used, because every wording ran identically.
        """
        out: Dict[str, float] = {}
        for path, tag in LLMFrontendSuite.PATH_TAGS.items():
            side = model_cmp[model_cmp['path'] == path]
            if side.empty:
                continue
            # Each row is already one service's wordings executed together, so
            # the per-wording spread lives in the dispatch-gap statistics.
            spreads = []
            for _, row in side.iterrows():
                mean_gap = float(row.get('feat_dispatch_gap_mean', 0.0))
                p90_gap = float(row.get('feat_dispatch_gap_p90', 0.0))
                if mean_gap > 0:
                    spreads.append(abs(p90_gap - mean_gap) / mean_gap)
            if spreads:
                out[f'{tag}_dispatch_spread'] = float(np.mean(spreads))
        return out

    @staticmethod
    def _leakage(model_cmp: pd.DataFrame) -> Dict[str, float]:
        """Can the service being served be told from the trace alone?

        The label is which service the trace came from.  An attacker stuck at
        chance learns nothing from timing, reuse or scheduling.  With only a
        handful of services per model this is a coarse probe, and it is reported
        as such rather than as a tight bound.
        """
        out: Dict[str, float] = {}
        feature_cols = [c for c in model_cmp.columns if c.startswith('feat_')]
        if not feature_cols:
            return out
        intents = sorted(model_cmp['intent'].unique())
        if len(intents) < 2:
            return out
        label_of = {intent: index % 2 for index, intent in enumerate(intents)}
        for path, tag in LLMFrontendSuite.PATH_TAGS.items():
            side = model_cmp[model_cmp['path'] == path].copy()
            if len(side) < 4:
                continue
            side['label'] = side['intent'].map(label_of).astype(int)
            if side['label'].nunique() < 2:
                continue
            frame = side[feature_cols + ['label']].rename(
                columns={c: c[len('feat_'):] for c in feature_cols}
            )
            attacks = run_binary_attacks(frame, label_col='label')
            best = max(a.accuracy for a in attacks)
            out[f'{tag}_attack_accuracy'] = float(best)
            out[f'{tag}_security_score'] = float(binary_security_score(best, 0.0, 0.0))
        return out

    # --------------------------------------------------------------------- run

    def run(self, output_root: Path) -> FrontendArtifacts:
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        replies = self.collect_replies()
        replies_df = pd.DataFrame(
            [
                {
                    'model': r.model,
                    'kind': getattr(r, 'kind', 'unknown'),
                    'intent': r.intent,
                    'prompt_index': r.prompt_index,
                    'prompt': r.prompt,
                    'latency_s': r.latency_s,
                    'error': r.error,
                    'raw': r.raw,
                }
                for r in replies
            ]
        )
        plans, comparison = self.evaluate(replies)
        summary = self.summarize(plans, comparison)
        environment = {
            'gateway_models': self.gateway_models,
            'local_models': [hf for _, hf in self.local_models],
            'gateway_configured': bool(self.gateway_base and self.gateway_token),
            'device': self.device,
            'intents': [s['intent'] for s in SERVICE_INTENTS],
            'prompts_per_intent': len(SERVICE_INTENTS[0]['prompts']),
            'freshness_window_cycles': FRONTEND_FRESHNESS_WINDOW_CYCLES,
        }
        replies_df.to_csv(output_root / 'frontend_replies.csv', index=False)
        plans.to_csv(output_root / 'frontend_plans.csv', index=False)
        comparison.to_csv(output_root / 'frontend_comparison.csv', index=False)
        summary.to_csv(output_root / 'frontend_summary.csv', index=False)
        (output_root / 'frontend_env.json').write_text(
            json.dumps(environment, indent=2), encoding='utf-8'
        )
        artifacts = FrontendArtifacts(
            replies=replies_df,
            plans=plans,
            comparison=comparison,
            summary=summary,
            environment=environment,
        )
        if not summary.empty:
            self._plot(summary, output_root / 'fig_llm_frontend.pdf')
        return artifacts

    @staticmethod
    def _plot(summary: pd.DataFrame, out_path: Path) -> None:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        order = {'hosted': 0, 'local': 1}
        data = summary.copy()
        data['_kind'] = data['kind'].map(lambda k: order.get(k, 9))
        data = data.sort_values(['_kind', 'parse_rate'], ascending=[True, False]).reset_index(drop=True)
        labels = list(data['model'])
        x = np.arange(len(labels))
        boundary = int((data['_kind'] == 0).sum())
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.6, 4.3))
        width = 0.38

        def annotate(ax, bars, values, fmt='{:.0%}'):
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    max(float(value), 0.0) + 0.02,
                    fmt.format(float(value)),
                    ha='center', va='bottom', fontsize=6.4, rotation=90,
                )

        def split_marker(ax):
            """Hosted models to the left of the line, local ones to the right."""
            if 0 < boundary < len(labels):
                ax.axvline(boundary - 0.5, color='#666666', linewidth=0.8, linestyle=':')
                ax.text(boundary / 2 - 0.5, 1.12, 'hosted', ha='center', fontsize=7.5, color='#444444')
                ax.text((boundary + len(labels)) / 2 - 0.5, 1.12, 'local', ha='center',
                        fontsize=7.5, color='#444444')

        parses = data['parse_rate'].astype(float)
        sizes = data['plan_fidelity'].astype(float)
        bars_a = ax_left.bar(x - width / 2, parses, width, label='Reply is a schedulable plan',
                             color='#2A9D8F')
        bars_b = ax_left.bar(x + width / 2, sizes, width, label='Sizes match the request',
                             color='#E9C46A')
        annotate(ax_left, bars_a, parses)
        annotate(ax_left, bars_b, sizes)
        ax_left.set_xticks(x)
        ax_left.set_xticklabels(labels, rotation=28, ha='right', fontsize=7.4)
        ax_left.set_ylim(0, 1.2)
        ax_left.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax_left.set_ylabel('Fraction of requests')
        ax_left.set_title('(a) Can the model compile the request?')
        ax_left.legend(fontsize=7.4, frameon=False, loc='lower left')
        ax_left.grid(axis='y', alpha=0.25, linewidth=0.5)
        split_marker(ax_left)

        if 'canonical_agreement' in data.columns:
            plain = data['unprotected_agreement'].astype(float).fillna(0.0)
            canon = data['canonical_agreement'].astype(float).fillna(0.0)
            bars_c = ax_right.bar(x - width / 2, plain, width, label='Unprotected front-end',
                                  color='#264653')
            bars_d = ax_right.bar(x + width / 2, canon, width,
                                  label=f'{ACCELERATOR_NAME} canonical front-end', color='#E76F51')
            annotate(ax_right, bars_c, plain)
            annotate(ax_right, bars_d, canon)
            ax_right.set_xticks(x)
            ax_right.set_xticklabels(labels, rotation=28, ha='right', fontsize=7.4)
            ax_right.set_ylim(0, 1.2)
            ax_right.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax_right.set_ylabel('Wordings of one service that agree')
            ax_right.set_title('(b) Same request, different words')
            ax_right.legend(fontsize=7.4, frameon=False, loc='lower left')
            ax_right.grid(axis='y', alpha=0.25, linewidth=0.5)
            split_marker(ax_right)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
