from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple
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

    def build_runtime_tasks(self, load_factor: float=1.0) -> List[RuntimeTask]:
        factor = max(0.5, float(load_factor))
        runtime_tasks: List[RuntimeTask] = []
        arrival = 0.0
        for graph, base_count, gap in zip(self.graphs, self.counts, self.gaps_ms):
            scaled_count = max(1, int(round(base_count * factor)))
            scaled_gap = gap / max(0.8, factor)
            for idx in range(scaled_count):
                runtime_tasks.append(RuntimeTask(runtime_id=f'{self.slug}_{graph.task_id}_{idx}', task_graph=graph, arrival_cycle=int(arrival * CYCLES_PER_MS), runtime_metadata={'case': self.slug, 'load_factor': factor}))
                arrival += scaled_gap * (1.0 + 0.05 * (idx % 3))
        runtime_tasks.sort(key=lambda task: task.arrival_cycle)
        return runtime_tasks

    def manifest_row(self) -> Dict[str, object]:
        return {'case': self.name, 'slug': self.slug, 'slo_ms': self.slo_ms, 'storyline': self.storyline, 'focus': self.focus, 'stages': ' + '.join(self.stage_names()), 'base_requests': int(sum(self.counts)), 'load_factors': ','.join((f'{x:.1f}' for x in self.load_factors)), 'prompt': self.prompt}

class CaseStudyCatalog:

    def __init__(self, frontend: HeuristicLLMFrontend | None=None):
        self.frontend = frontend or HeuristicLLMFrontend()

    def build(self) -> List[CaseStudyDefinition]:
        specs = [{'name': 'Private RAG Gateway', 'slug': 'private_rag_gateway', 'prompt': 'build a private rag gateway with llm prefill seq 768 hidden 3072 heads 24 rerank mlp hidden 2048 aes 1024 groth16 2048', 'storyline': '检索增强问答请求进入网关，先走轻量重排，再触发上下文预填充，同时对文档块做加密并异步生成审计证明。', 'focus': '强调 secure mode 下多阶段复用与尾延迟压缩。', 'slo_ms': 95.0, 'counts': (2, 4, 2, 1), 'gaps_ms': (0.18, 0.08, 0.14, 0.32), 'load_factors': (0.5, 1.0, 1.5, 2.0)}, {'name': 'Rollup Sequencer', 'slug': 'rollup_sequencer', 'prompt': 'compile a rollup sequencer service with groth16 4096 aes 2048 transformer seq 512 hidden 2048 heads 16', 'storyline': '排序器在批次提交前先做状态证明，再对交易批加密，同时利用小型 LLM 对异常区块做预筛。', 'focus': '强调 ZKP 主导场景下 secure elastic 对 secure-ct 与通用 GPU 的优势。', 'slo_ms': 118.0, 'counts': (2, 4, 2), 'gaps_ms': (0.34, 0.12, 0.28), 'load_factors': (0.5, 1.0, 1.5, 2.0)}, {'name': 'Vision-Language Guard', 'slug': 'vision_language_guard', 'prompt': 'generate a vision-language guard service with vgg16 batch 4 llm prefill seq 640 hidden 2048 heads 16 aes 512', 'storyline': '摄像头帧先经视觉编码器，再通过文本判定模块生成策略标签，同时对敏感片段加密存档。', 'focus': '强调跨 CNN 与 LLM 的共享模板池和服务级 tail latency。', 'slo_ms': 102.0, 'counts': (3, 2, 3), 'gaps_ms': (0.1, 0.16, 0.08), 'load_factors': (0.5, 1.0, 1.5, 2.0)}, {'name': 'Tenant AI Gateway', 'slug': 'tenant_ai_gateway', 'prompt': 'emit a tenant ai gateway with transformer seq 896 hidden 3072 heads 24 aes 2048 rerank mlp hidden 1024 groth16 3072', 'storyline': '多租户网关对不同企业请求做统一接入，先做路由重排，再做生成推理，同时附带加密与账务证明。', 'focus': '强调 secure mode 对多租户混部的稳定性和隔离收益。', 'slo_ms': 112.0, 'counts': (2, 4, 3, 2), 'gaps_ms': (0.22, 0.07, 0.14, 0.24), 'load_factors': (0.5, 1.0, 1.5, 2.0)}, {'name': 'Fraud Stream', 'slug': 'fraud_stream', 'prompt': 'build a fraud stream pipeline with mlp hidden 1536 batch 32 aes 1024 groth16 2048 transformer seq 384 hidden 1536 heads 12', 'storyline': '支付流实时打分、加密、出具轻量证明，并用小模型对可疑模式进行语义归类。', 'focus': '强调高并发短作业与长作业混部下的队列韧性。', 'slo_ms': 88.0, 'counts': (5, 4, 2, 2), 'gaps_ms': (0.05, 0.06, 0.18, 0.16), 'load_factors': (0.5, 1.0, 1.5, 2.0)}, {'name': 'Confidential Edge Copilot', 'slug': 'confidential_edge_copilot', 'prompt': 'compile a confidential edge copilot with llm prefill seq 384 hidden 1536 heads 12 rerank mlp hidden 1024 aes 512 groth16 1024', 'storyline': '边缘端助理在资源紧张的设备上执行小模型推理、轻量重排、会话加密与审计证明。', 'focus': '强调资源受限情况下 secure elastic 的端到端收益。', 'slo_ms': 74.0, 'counts': (3, 4, 2, 1), 'gaps_ms': (0.09, 0.05, 0.08, 0.22), 'load_factors': (0.5, 1.0, 1.5, 2.0)}]
        out: List[CaseStudyDefinition] = []
        for spec in specs:
            compiled = self.frontend.compile_prompt(spec['prompt'], secure=True, task_prefix=spec['slug'])
            out.append(CaseStudyDefinition(name=spec['name'], slug=spec['slug'], prompt=spec['prompt'], storyline=spec['storyline'], focus=spec['focus'], slo_ms=spec['slo_ms'], counts=tuple(spec['counts']), gaps_ms=tuple(spec['gaps_ms']), load_factors=tuple(spec['load_factors']), graphs=compiled.graphs))
        return out

    def manifest(self) -> pd.DataFrame:
        return pd.DataFrame([case.manifest_row() for case in self.build()])
