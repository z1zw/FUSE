from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from .ir import RuntimeTask, TaskGraph
from .llm import HeuristicLLMFrontend
from .sim.simulator import CYCLES_PER_MS


@dataclass
class CaseStudyDefinition:
    name: str
    slug: str
    prompt: str
    storyline: str
    focus: str
    slo_ms: float
    counts: Tuple[int, ...]
    gaps_ms: Tuple[float, ...]
    load_factors: Tuple[float, ...]
    graphs: List[TaskGraph]

    def stage_names(self) -> List[str]:
        out = []
        for graph in self.graphs:
            model = str(graph.metadata.get('model', graph.task_type))
            if graph.task_type == 'LLM':
                out.append('LLM')
            elif graph.task_type == 'AES':
                out.append('AES')
            elif graph.task_type == 'ZKP':
                out.append('ZKP')
            elif model == 'MLP-path':
                out.append('MLP')
            else:
                out.append(graph.task_type)
        return out

    def build_runtime_tasks(self, load_factor: float = 1.0) -> List[RuntimeTask]:
        factor = max(0.5, float(load_factor))
        runtime_tasks: List[RuntimeTask] = []
        arrival = 0.0
        for graph, base_count, gap in zip(self.graphs, self.counts, self.gaps_ms):
            scaled_count = max(1, int(round(base_count * factor)))
            scaled_gap = gap / max(0.8, factor)
            for idx in range(scaled_count):
                runtime_tasks.append(
                    RuntimeTask(
                        runtime_id=f'{self.slug}_{graph.task_id}_{idx}',
                        task_graph=graph,
                        arrival_cycle=int(arrival * CYCLES_PER_MS),
                        runtime_metadata={'case': self.slug, 'load_factor': factor},
                    )
                )
                arrival += scaled_gap * (1.0 + 0.05 * (idx % 3))
        runtime_tasks.sort(key=lambda task: task.arrival_cycle)
        return runtime_tasks

    def manifest_row(self) -> Dict[str, object]:
        return {
            'case': self.name,
            'slug': self.slug,
            'slo_ms': self.slo_ms,
            'storyline': self.storyline,
            'focus': self.focus,
            'stages': ' + '.join(self.stage_names()),
            'base_requests': int(sum(self.counts)),
            'load_factors': ','.join((f'{x:.1f}' for x in self.load_factors)),
            'prompt': self.prompt,
        }


class CaseStudyCatalog:
    def __init__(self, frontend: HeuristicLLMFrontend | None = None):
        self.frontend = frontend or HeuristicLLMFrontend()

    def build(self) -> List[CaseStudyDefinition]:
        specs = [
            {
                'name': 'Private RAG Gateway',
                'slug': 'private_rag_gateway',
                'prompt': 'build a private rag gateway with llm prefill seq 768 hidden 3072 heads 24 rerank mlp hidden 2048 aes 1024 groth16 2048',
                'storyline': 'A retrieval request enters a private gateway, passes through a compact reranker, triggers context prefill, encrypts document shards, and emits an asynchronous audit proof.',
                'focus': 'Emphasize multi-stage reuse and tail-latency compression in secure mode.',
                'slo_ms': 95.0,
                'counts': (2, 4, 2, 1),
                'gaps_ms': (0.18, 0.08, 0.14, 0.32),
                'load_factors': (0.5, 1.0, 1.5, 2.0),
            },
            {
                'name': 'Rollup Sequencer',
                'slug': 'rollup_sequencer',
                'prompt': 'compile a rollup sequencer service with groth16 4096 aes 2048 transformer seq 512 hidden 2048 heads 16',
                'storyline': 'The sequencer proves state before batch submission, encrypts transaction bundles, and uses a small LLM to pre-screen anomalous blocks.',
                'focus': 'Highlight the advantage of secure elastic execution over secure-ct and generic GPU baselines in a ZKP-dominant pipeline.',
                'slo_ms': 118.0,
                'counts': (2, 4, 2),
                'gaps_ms': (0.34, 0.12, 0.28),
                'load_factors': (0.5, 1.0, 1.5, 2.0),
            },
            {
                'name': 'Vision-Language Guard',
                'slug': 'vision_language_guard',
                'prompt': 'generate a vision-language guard service with vgg16 batch 4 llm prefill seq 640 hidden 2048 heads 16 aes 512',
                'storyline': 'Camera frames go through a vision encoder, then a text-side policy classifier, while sensitive clips are encrypted for archival.',
                'focus': 'Highlight shared template pools across CNN and LLM stages together with service-level tail latency.',
                'slo_ms': 102.0,
                'counts': (3, 2, 3),
                'gaps_ms': (0.1, 0.16, 0.08),
                'load_factors': (0.5, 1.0, 1.5, 2.0),
            },
            {
                'name': 'Tenant AI Gateway',
                'slug': 'tenant_ai_gateway',
                'prompt': 'emit a tenant ai gateway with transformer seq 896 hidden 3072 heads 24 aes 2048 rerank mlp hidden 1024 groth16 3072',
                'storyline': 'A multi-tenant gateway routes enterprise requests, reranks them, serves generation, and attaches encryption with billing proofs.',
                'focus': 'Highlight stability and isolation gains for mixed-tenant consolidation under secure mode.',
                'slo_ms': 112.0,
                'counts': (2, 4, 3, 2),
                'gaps_ms': (0.22, 0.07, 0.14, 0.24),
                'load_factors': (0.5, 1.0, 1.5, 2.0),
            },
            {
                'name': 'Fraud Stream',
                'slug': 'fraud_stream',
                'prompt': 'build a fraud stream pipeline with mlp hidden 1536 batch 32 aes 1024 groth16 2048 transformer seq 384 hidden 1536 heads 12',
                'storyline': 'A payment stream is scored, encrypted, proven with lightweight evidence, and then classified by a compact language model for semantic clustering.',
                'focus': 'Highlight queue resilience when short and long jobs share the same execution fabric under high concurrency.',
                'slo_ms': 88.0,
                'counts': (5, 4, 2, 2),
                'gaps_ms': (0.05, 0.06, 0.18, 0.16),
                'load_factors': (0.5, 1.0, 1.5, 2.0),
            },
            {
                'name': 'Confidential Edge Copilot',
                'slug': 'confidential_edge_copilot',
                'prompt': 'compile a confidential edge copilot with llm prefill seq 384 hidden 1536 heads 12 rerank mlp hidden 1024 aes 512 groth16 1024',
                'storyline': 'An edge assistant performs compact inference, reranking, session encryption, and audit proof generation on a constrained device.',
                'focus': 'Highlight end-to-end secure elastic gains under resource-constrained deployment conditions.',
                'slo_ms': 74.0,
                'counts': (3, 4, 2, 1),
                'gaps_ms': (0.09, 0.05, 0.08, 0.22),
                'load_factors': (0.5, 1.0, 1.5, 2.0),
            },
        ]
        out: List[CaseStudyDefinition] = []
        for spec in specs:
            compiled = self.frontend.compile_prompt(spec['prompt'], secure=True, task_prefix=spec['slug'])
            out.append(
                CaseStudyDefinition(
                    name=spec['name'],
                    slug=spec['slug'],
                    prompt=spec['prompt'],
                    storyline=spec['storyline'],
                    focus=spec['focus'],
                    slo_ms=spec['slo_ms'],
                    counts=tuple(spec['counts']),
                    gaps_ms=tuple(spec['gaps_ms']),
                    load_factors=tuple(spec['load_factors']),
                    graphs=compiled.graphs,
                )
            )
        return out

    def manifest(self) -> pd.DataFrame:
        return pd.DataFrame([case.manifest_row() for case in self.build()])
