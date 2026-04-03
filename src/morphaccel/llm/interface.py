from __future__ import annotations
from dataclasses import dataclass
import hashlib
import re
from typing import Dict, List, Sequence
from ..ir import Fragment, TaskGraph
from ..workloads import build_aes128_decrypt_task, build_aes128_task, build_groth16_like_task, build_groth16_verify_like_task, build_mlp_path_task, build_transformer_block_task, build_vgg16_like_task
from .canonicalizer import CanonicalTaskCanonicalizer

@dataclass
class CompiledPrompt:
    prompt: str
    graphs: List[TaskGraph]
    prompt_hash: str
    canonical_hash: str
    secure: bool

@dataclass
class PromptVariant:
    intent: str
    label: str
    prompt: str

class HeuristicLLMFrontend:

    def __init__(self):
        self.canonicalizer = CanonicalTaskCanonicalizer()

    def compile_prompt(self, prompt: str, secure: bool=True, task_prefix: str='prompt') -> CompiledPrompt:
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        graphs = self._infer_graphs(prompt, task_prefix=task_prefix)
        if secure:
            graphs = self.canonicalizer.canonicalize_graphs(graphs)
        else:
            graphs = [self._inject_prompt_variance(graph, prompt_hash, prompt) for graph in graphs]
        canonical_hash = self.canonicalizer.canonical_hash(graphs)
        return CompiledPrompt(prompt=prompt, graphs=graphs, prompt_hash=prompt_hash, canonical_hash=canonical_hash, secure=secure)

    def variant_prompts(self) -> Dict[str, List[PromptVariant]]:
        catalog = {'zkp': ['compile a groth16 prover for an 4k-gate rollup circuit', 'build a zk rollup proof pipeline with groth16 and 4096 constraints', 'generate a groth16 proving graph for 4k constraints', 'emit a rollup prover schedule using groth16 for 4096 gates'], 'vision': ['compile a vgg16 vision batch pipeline with batch 8', 'generate a cnn inference graph for vgg16 bs8', 'emit a vgg16 batch-8 serving pipeline', 'build a vision inference schedule around vgg16 batch size 8'], 'crypto': ['compile aes-128 encryption for 1024 blocks', 'build a crypto pipeline with aes128 and 1k blocks', 'emit an aes round graph for 1024 blocks', 'generate an aes-128 batch encryption schedule for 1k blocks'], 'llm': ['compile a transformer block for llm prefill seq 1024 hidden 3072 heads 24', 'generate an llm attention block with sequence 1024 hidden 3072 24 heads', 'build a transformer prefill graph seq=1024 d=3072 h=24', 'emit a decoder-block schedule for a 1024-token llm context'], 'hybrid': ['build a hybrid agent pipeline with llm prefill seq 512 aes 512 and groth16 2048', 'compile a mixed llm plus aes plus groth16 service seq 512 aes 512 groth16 2048', 'generate an agentic secure inference pipeline with transformer, aes, and zkp', 'emit a hybrid service graph for transformer 512, aes 512, groth16 2048'], 'service': ['build a private rag gateway with llm prefill seq 768 hidden 3072 heads 24 rerank mlp hidden 2048 aes 1024 groth16 2048', 'compile a rollup sequencer service with groth16 4096 aes 2048 transformer seq 512 hidden 2048 heads 16', 'generate a tenant ai gateway with transformer seq 896 hidden 3072 heads 24 aes 2048 rerank mlp hidden 1024 groth16 3072', 'compile a confidential edge copilot with llm prefill seq 384 hidden 1536 heads 12 rerank mlp hidden 1024 aes 512 groth16 1024']}
        out: Dict[str, List[PromptVariant]] = {}
        for intent, prompts in catalog.items():
            variants = []
            for idx, prompt in enumerate(prompts):
                for label in ['alpha', 'beta']:
                    variants.append(PromptVariant(intent=intent, label=label, prompt=f'{prompt} nonce {label}'))
            out[intent] = variants
        return out

    def _infer_graphs(self, prompt: str, task_prefix: str) -> List[TaskGraph]:
        text = prompt.lower()
        seq = self._extract_number(text, ['seq(?:uence)?\\s*(\\d+)', 'context\\s*(\\d+)', 'token\\w*\\s*(\\d+)'], 1024)
        hidden = self._extract_number(text, ['hidden\\s*(\\d+)', 'd\\s*=\\s*(\\d+)'], 4096)
        heads = self._extract_number(text, ['heads?\\s*(\\d+)', 'h\\s*=\\s*(\\d+)'], 32)
        batch = self._extract_number(text, ['batch\\s*(\\d+)'], 8)
        aes_blocks = self._extract_scaled_number(text, ['aes[- ]?128.*?(\\d+)k', 'aes.*?(\\d+)k', 'aes.*?(\\d+)'], 2048)
        groth = self._extract_scaled_number(text, ['groth16.*?(\\d+)k', 'groth16.*?(\\d+)', 'constraint\\w*\\s*(\\d+)k', 'constraint\\w*\\s*(\\d+)'], 4096)
        graphs: List[TaskGraph] = []
        if any((token in text for token in ['llm', 'transformer', 'attention', 'decoder', 'prefill'])):
            graphs.append(build_transformer_block_task(seq_len=seq, hidden_dim=hidden, heads=heads, batch_size=1, task_id=f'{task_prefix}_llm'))
        if any((token in text for token in ['groth16', 'zk', 'proof', 'rollup'])):
            if any((token in text for token in ['verify', 'verification', 'verifier'])):
                graphs.append(build_groth16_verify_like_task(circuit_size=groth, task_id=f'{task_prefix}_zkp_verify'))
            else:
                graphs.append(build_groth16_like_task(circuit_size=groth, rounds=4, task_id=f'{task_prefix}_zkp'))
        if any((token in text for token in ['vgg', 'vision', 'cnn'])):
            graphs.append(build_vgg16_like_task(batch_size=batch, task_id=f'{task_prefix}_cnn'))
        if any((token in text for token in ['aes', 'encrypt', 'crypto'])):
            if any((token in text for token in ['decrypt', 'decryption'])):
                graphs.append(build_aes128_decrypt_task(blocks=aes_blocks, task_id=f'{task_prefix}_aes_decrypt'))
            else:
                graphs.append(build_aes128_task(blocks=aes_blocks, task_id=f'{task_prefix}_aes'))
        if any((token in text for token in ['mlp', 'rerank', 'reranker'])):
            graphs.append(build_mlp_path_task(batch_size=batch, hidden_dim=hidden, task_id=f'{task_prefix}_mlp'))
        if not graphs:
            graphs.append(build_transformer_block_task(seq_len=seq, hidden_dim=hidden, heads=heads, batch_size=1, task_id=f'{task_prefix}_llm'))
        return graphs

    def _inject_prompt_variance(self, graph: TaskGraph, prompt_hash: str, prompt: str) -> TaskGraph:
        clone = graph.copy(task_id=f'{graph.task_id}_{prompt_hash}')
        for frag in clone.fragments.values():
            group = frag.semantic_group or frag.fragment_id
            variant = hashlib.sha256(f'{prompt_hash}:{frag.fragment_id}:{frag.stage}'.encode()).hexdigest()[:6]
            frag.semantic_group = f'{group}@prompt_{variant}'
            frag.metadata = {**frag.metadata, 'prompt_hash': prompt_hash, 'prompt_variant': prompt[:64], 'runtime_seed': int(variant, 16) % 10007}
        clone.metadata = {**clone.metadata, 'prompt_hash': prompt_hash, 'prompt_text': prompt[:128]}
        return clone

    def _extract_number(self, text: str, patterns: Sequence[str], default: int) -> int:
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))
        return default

    def _extract_scaled_number(self, text: str, patterns: Sequence[str], default: int) -> int:
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = int(match.group(1))
                if 'k' in match.group(0):
                    return value * 1024
                return value
        return default
