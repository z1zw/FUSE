"""Language-model evaluation for FUSE.

Three things happen here, for each model in the zoo:

1. The model is run for real on the local GPU over prompts from a public
   dataset.  That gives measured prefill latency, per-token decode latency and
   NVML-measured energy, plus the true tokenized prompt lengths.
2. Those measured lengths and the model's own architecture are turned into FUSE
   task graphs and replayed through the runtime simulator under FUSE and under
   each baseline, so the comparison uses the same workload the GPU just ran.
3. The same requests are replayed as a private-serving trace -- AES over the
   payload, the language model itself, and a Groth16 integrity proof -- which is
   the cross-task setting FUSE targets.

Steps 2 and 3 are simulated, as everywhere else in the evaluation; step 1 is
measured on the machine running the script.
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .branding import ACCELERATOR_NAME, display_mode_name
from .ir import RuntimeTask, TaskGraph
from .reuse import MicrofragmentDecomposer
from .sim.simulator import CYCLES_PER_MS, FuseSimulator, SimulationConfig
from .workloads import build_aes128_task, build_groth16_like_task
from .workloads.llm import (
    LLMArchSpec,
    build_llm_decode_task,
    build_llm_prefill_task,
    spec_from_hf_config,
)

#: Models that fit on a single 16 GB consumer GPU in fp16.  The set spans three
#: orders of magnitude of parameter count and three architecture families
#: (absolute-position dense, grouped-query dense, and sparse mixture-of-experts)
#: so the reuse results are not an artefact of one layer shape.
LLM_ZOO: List[Dict[str, str]] = [
    {'name': 'gpt2', 'label': 'GPT-2', 'hf_id': 'gpt2', 'family': 'dense-MHA'},
    {'name': 'qwen2_5_0_5b', 'label': 'Qwen2.5-0.5B', 'hf_id': 'Qwen/Qwen2.5-0.5B-Instruct', 'family': 'dense-GQA'},
    {'name': 'tinyllama_1_1b', 'label': 'TinyLlama-1.1B', 'hf_id': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'family': 'dense-GQA'},
    {'name': 'olmoe_1b_7b', 'label': 'OLMoE-1B-7B', 'hf_id': 'allenai/OLMoE-1B-7B-0924-Instruct', 'family': 'MoE'},
    {'name': 'mistral_7b', 'label': 'Mistral-7B', 'hf_id': 'mistralai/Mistral-7B-Instruct-v0.2', 'family': 'dense-GQA'},
    {'name': 'qwen2_5_7b', 'label': 'Qwen2.5-7B', 'hf_id': 'Qwen/Qwen2.5-7B-Instruct', 'family': 'dense-GQA'},
]

#: Baselines the language-model graphs are replayed against.  ``transformer_asic``
#: is the domain-specific competitor, matching how Groth16 is compared against a
#: static ZKP core and VGG16 against an Eyeriss-style array.
LLM_BASELINE_MODES: Tuple[str, ...] = ('transformer_asic', 'gpu_tensor', 'multicore')

#: Simulator mode that turns FUSE's own mechanisms on.
ACCELERATOR_NAME_MODE = 'fuse'

#: Baseline for the mixed private-serving trace.  The single-domain engines are
#: deliberately absent: neither a transformer accelerator nor a tensor core can
#: execute a Groth16 proof or an AES round, so a multi-core system holding one
#: dedicated engine per domain is the only baseline that can run the trace.
SERVING_BASELINE_MODES: Tuple[str, ...] = ('multicore',)

#: Multipliers on the security volume per request.  Scale 0 is inference only,
#: so the ratio against scale 1 states what privacy and integrity cost.  A
#: cryptographic fragment is three orders of magnitude cheaper than a decoder
#: layer, so the sweep stops where the trace stays tractable rather than trying
#: to equalise the two kinds of work.
CRYPTO_SCALES: Tuple[float, ...] = (0.0, 1.0, 2.0, 4.0)

#: Fragment-cache retention.  A language-model fragment occupies the array for
#: far longer than an AES round, so the 5 ms default expires every entry before
#: it can be consulted.  600 ms covers one request at the sizes evaluated here
#: and is applied identically to every mode.
LLM_FRESHNESS_WINDOW_CYCLES = 600 * CYCLES_PER_MS

#: Output tiles each large matmul is split into.  Raising it towards the RFU
#: count fills the array better and lowers latency, but it also multiplies the
#: fragment count, and the representative-path search in the decomposer is
#: superlinear in that, which makes the security sweep intractable.  The
#: FUSE-to-baseline ratios are unchanged either way.
DEFAULT_TILING_WIDTH = 8

DEFAULT_GEN_TOKENS = 64
DECODE_PROBE_FRACTIONS: Tuple[float, ...] = (0.25, 0.5, 1.0)


def _plt():
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    return plt


@dataclass
class PromptSample:
    sample_id: str
    text: str
    prompt_tokens: int = 0
    payload_bytes: int = 0


@dataclass
class LLMZooArtifacts:
    native: pd.DataFrame
    simulated: pd.DataFrame
    secure_serving: pd.DataFrame
    summary: pd.DataFrame
    environment: Dict[str, Any]


class _PowerSampler:
    """Samples board power through NVML while a region of code runs."""

    def __init__(self, device: str, poll_interval_s: float = 0.01):
        self.enabled = device.startswith('cuda')
        self.poll_interval_s = poll_interval_s
        self.samples: List[float] = []
        self._stop = Event()
        self._thread: Optional[Thread] = None
        self._handle = None
        self._pynvml = None

    def __enter__(self) -> '_PowerSampler':
        if not self.enabled:
            return self
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.enabled = False
            return self

        def _poll() -> None:
            while not self._stop.is_set():
                try:
                    self.samples.append(float(self._pynvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0)
                except Exception:
                    break
                self._stop.wait(self.poll_interval_s)

        self._thread = Thread(target=_poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass

    @property
    def mean_power_w(self) -> float:
        return float(statistics.fmean(self.samples)) if self.samples else 0.0


class LLMZooExperimentSuite:
    """Runs the language-model study end to end."""

    def __init__(
        self,
        device: str = 'auto',
        dataset: str = 'gsm8k',
        sample_count: int = 8,
        gen_tokens: int = DEFAULT_GEN_TOKENS,
        warmup: int = 1,
        repeat: int = 3,
        seed: int = 7,
        models: Optional[Sequence[str]] = None,
        measure_native: bool = True,
        zkp_circuit_size: int = 65536,
        aes_kib_per_request: float = 256.0,
        proofs_per_request: float = 2.0,
        mix_sweep: bool = True,
        reuse_native: bool = False,
        width: int = DEFAULT_TILING_WIDTH,
    ):
        self.aes_kib_per_request = aes_kib_per_request
        self.proofs_per_request = proofs_per_request
        self.mix_sweep = mix_sweep
        self.reuse_native = reuse_native
        self.width = width
        self.serving_requests = 4
        self.dataset = dataset
        self.sample_count = max(1, sample_count)
        self.gen_tokens = max(1, gen_tokens)
        self.warmup = max(0, warmup)
        self.repeat = max(1, repeat)
        self.seed = seed
        self.run_native = measure_native
        self.zkp_circuit_size = zkp_circuit_size
        self.torch = self._optional_torch()
        self.device = self._resolve_device(device)
        selected = set(models) if models else None
        self.models = [m for m in LLM_ZOO if selected is None or m['name'] in selected]
        if not self.models:
            raise ValueError(f'no models selected from the zoo: {sorted(m["name"] for m in LLM_ZOO)}')

    # ---------------------------------------------------------------- set-up

    @staticmethod
    def _optional_torch():
        try:
            import torch

            return torch
        except Exception:
            return None

    def _resolve_device(self, requested: str) -> str:
        if requested != 'auto':
            return requested
        if self.torch is not None and self.torch.cuda.is_available():
            return 'cuda:0'
        return 'cpu'

    def environment(self) -> Dict[str, Any]:
        import platform
        import sys

        device_name = 'CPU'
        if self.torch is not None and self.device.startswith('cuda'):
            try:
                device_name = self.torch.cuda.get_device_name(0)
            except Exception:
                device_name = 'CUDA device'
        return {
            'platform': platform.platform(),
            'python': sys.version.split()[0],
            'torch': getattr(self.torch, '__version__', None),
            'device': self.device,
            'device_name': device_name,
            'dataset': self.dataset,
            'sample_count': self.sample_count,
            'gen_tokens': self.gen_tokens,
            'repeat': self.repeat,
            'warmup': self.warmup,
            'seed': self.seed,
            'freshness_window_cycles': LLM_FRESHNESS_WINDOW_CYCLES,
            'tiling_width': self.width,
            'baseline_modes': list(LLM_BASELINE_MODES),
            'serving_baseline_modes': list(SERVING_BASELINE_MODES),
            'zkp_circuit_size': self.zkp_circuit_size,
            'aes_kib_per_request': self.aes_kib_per_request,
            'proofs_per_request': self.proofs_per_request,
            'crypto_scales': list(CRYPTO_SCALES) if self.mix_sweep else [1.0],
            'models': [m['hf_id'] for m in self.models],
        }

    # --------------------------------------------------------------- dataset

    def load_prompts(self) -> List[PromptSample]:
        """Load ``sample_count`` prompts from the configured public dataset."""
        from datasets import load_dataset

        if self.dataset == 'gsm8k':
            data = load_dataset('gsm8k', 'main', split='test')
            field_name = 'question'
        elif self.dataset == 'mmlu':
            data = load_dataset('cais/mmlu', 'all', split='test')
            field_name = 'question'
        else:
            raise ValueError(f'unsupported dataset: {self.dataset}')
        rng = np.random.default_rng(self.seed)
        indices = rng.choice(len(data), size=min(self.sample_count, len(data)), replace=False)
        samples: List[PromptSample] = []
        for rank, index in enumerate(sorted(int(i) for i in indices)):
            text = str(data[index][field_name]).strip()
            samples.append(
                PromptSample(
                    sample_id=f'{self.dataset}-{index:05d}',
                    text=text,
                    payload_bytes=len(text.encode('utf-8')),
                )
            )
        return samples

    # ------------------------------------------------------ native execution

    def _load_model(self, hf_id: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        dtype = self.torch.float16 if self.device.startswith('cuda') else self.torch.float32
        kwargs: Dict[str, Any] = {'dtype': dtype, 'low_cpu_mem_usage': True}
        if self.device.startswith('cuda'):
            kwargs['device_map'] = self.device
        model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
        if not self.device.startswith('cuda'):
            model = model.to(self.device)
        model.eval()
        return tokenizer, model

    def _sync(self) -> None:
        if self.torch is not None and self.device.startswith('cuda'):
            self.torch.cuda.synchronize()

    def _time_ms(self, fn) -> Tuple[float, float]:
        """Run ``fn`` ``repeat`` times; return mean wall time and mean power."""
        for _ in range(self.warmup):
            fn()
        self._sync()
        samples_ms: List[float] = []
        with _PowerSampler(self.device) as sampler:
            for _ in range(self.repeat):
                self._sync()
                start = time.perf_counter()
                fn()
                self._sync()
                samples_ms.append((time.perf_counter() - start) * 1000.0)
        return statistics.fmean(samples_ms), sampler.mean_power_w

    def measure_native(self, model_meta: Dict[str, str], samples: Sequence[PromptSample]) -> pd.DataFrame:
        """Run the real model over the prompts and record what the GPU does."""
        tokenizer, model = self._load_model(model_meta['hf_id'])
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        rows: List[Dict[str, Any]] = []
        try:
            for sample in samples:
                encoded = tokenizer(sample.text, return_tensors='pt').to(self.device)
                input_ids = encoded['input_ids']
                sample.prompt_tokens = int(input_ids.shape[1])

                with self.torch.inference_mode():
                    prefill_ms, prefill_w = self._time_ms(lambda: model(**encoded))
                    # ``min_new_tokens`` holds the token count fixed across
                    # models.  Without it a model that emits end-of-sequence on
                    # the first step stops after one token, and the per-token
                    # decode cost -- which is what this measures -- comes out
                    # near zero for reasons that have nothing to do with speed.
                    total_ms, total_w = self._time_ms(
                        lambda: model.generate(
                            **encoded,
                            max_new_tokens=self.gen_tokens,
                            min_new_tokens=self.gen_tokens,
                            do_sample=False,
                            pad_token_id=pad_id,
                        )
                    )
                decode_ms = max(total_ms - prefill_ms, 0.0)
                per_token_ms = decode_ms / self.gen_tokens
                rows.append(
                    {
                        'model': model_meta['name'],
                        'model_label': model_meta['label'],
                        'hf_id': model_meta['hf_id'],
                        'sample_id': sample.sample_id,
                        'prompt_tokens': sample.prompt_tokens,
                        'gen_tokens': self.gen_tokens,
                        'payload_bytes': sample.payload_bytes,
                        'device': self.device,
                        'native_prefill_ms': prefill_ms,
                        'native_decode_ms': decode_ms,
                        'native_decode_ms_per_token': per_token_ms,
                        'native_total_ms': total_ms,
                        'native_power_w': total_w,
                        'native_energy_j': total_w * total_ms / 1000.0,
                        'native_tokens_per_s': self.gen_tokens / max(decode_ms / 1000.0, 1e-9),
                    }
                )
        finally:
            model = None
            if self.torch is not None and self.device.startswith('cuda'):
                self.torch.cuda.empty_cache()
        return pd.DataFrame(rows)

    def tokenize_only(self, model_meta: Dict[str, str], samples: Sequence[PromptSample]) -> pd.DataFrame:
        """Fill in prompt lengths without running the model."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_meta['hf_id'])
        rows = []
        for sample in samples:
            sample.prompt_tokens = int(len(tokenizer(sample.text)['input_ids']))
            rows.append(
                {
                    'model': model_meta['name'],
                    'model_label': model_meta['label'],
                    'hf_id': model_meta['hf_id'],
                    'sample_id': sample.sample_id,
                    'prompt_tokens': sample.prompt_tokens,
                    'gen_tokens': self.gen_tokens,
                    'payload_bytes': sample.payload_bytes,
                    'device': 'none',
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------ simulation

    def arch_spec(self, model_meta: Dict[str, str]) -> LLMArchSpec:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_meta['hf_id'])
        return spec_from_hf_config(
            config, name=model_meta['name'], label=model_meta['label'], hf_id=model_meta['hf_id']
        )

    @staticmethod
    def _decompose(graphs: Sequence[TaskGraph]) -> Any:
        """Microfragment decomposition is mode-independent, so do it once."""
        return MicrofragmentDecomposer().decompose_tasks(list(graphs))

    @staticmethod
    def _simulate(
        graphs: Sequence[TaskGraph],
        runtime_tasks: Sequence[RuntimeTask],
        mode: str,
        decomposition: Any = None,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        if decomposition is None:
            decomposition = LLMZooExperimentSuite._decompose(graphs)
        config = SimulationConfig(mode=mode, freshness_window_cycles=LLM_FRESHNESS_WINDOW_CYCLES)
        result = FuseSimulator(config=config).run(runtime_tasks, decomposition.microfragments)
        events = pd.DataFrame(result.traces.get('events', []))
        return result.metrics, events

    def simulate_model(
        self,
        model_meta: Dict[str, str],
        spec: LLMArchSpec,
        samples: Sequence[PromptSample],
    ) -> pd.DataFrame:
        """Replay every prompt as prefill plus probed decode steps."""
        rows: List[Dict[str, Any]] = []
        modes = (ACCELERATOR_NAME_MODE,) + LLM_BASELINE_MODES
        for sample in samples:
            prompt_tokens = max(1, sample.prompt_tokens)
            prefill = build_llm_prefill_task(
                spec,
                seq_len=prompt_tokens,
                width=self.width,
                content_tag=f'{sample.sample_id}:prefill',
                task_id=f'{spec.name}_{sample.sample_id}_prefill',
            )
            decode_graphs = []
            for fraction in DECODE_PROBE_FRACTIONS:
                context = prompt_tokens + max(1, int(self.gen_tokens * fraction))
                decode_graphs.append(
                    build_llm_decode_task(
                        spec,
                        context_len=context,
                        width=self.width,
                        content_tag=f'{sample.sample_id}:decode:{fraction}',
                        task_id=f'{spec.name}_{sample.sample_id}_decode_{fraction}',
                    )
                )
            prefill_decomposition = self._decompose([prefill])
            decode_decompositions = [self._decompose([graph]) for graph in decode_graphs]
            for mode in modes:
                prefill_metrics, _ = self._simulate(
                    [prefill],
                    [RuntimeTask(runtime_id='prefill', task_graph=prefill, arrival_cycle=0)],
                    mode,
                    decomposition=prefill_decomposition,
                )
                decode_latencies = []
                decode_energy = []
                for graph, decomposition in zip(decode_graphs, decode_decompositions):
                    metrics, _ = self._simulate(
                        [graph],
                        [RuntimeTask(runtime_id='decode', task_graph=graph, arrival_cycle=0)],
                        mode,
                        decomposition=decomposition,
                    )
                    decode_latencies.append(float(metrics['makespan_ms']))
                    decode_energy.append(float(metrics['energy_nj']))
                per_token_ms = statistics.fmean(decode_latencies)
                per_token_nj = statistics.fmean(decode_energy)
                prefill_ms = float(prefill_metrics['makespan_ms'])
                total_ms = prefill_ms + per_token_ms * self.gen_tokens
                total_nj = float(prefill_metrics['energy_nj']) + per_token_nj * self.gen_tokens
                rows.append(
                    {
                        'model': model_meta['name'],
                        'model_label': model_meta['label'],
                        'family': model_meta['family'],
                        'sample_id': sample.sample_id,
                        'prompt_tokens': prompt_tokens,
                        'gen_tokens': self.gen_tokens,
                        'mode': mode,
                        'mode_label': display_mode_name(mode),
                        'parameter_count': spec.parameter_count(),
                        'sim_prefill_ms': prefill_ms,
                        'sim_decode_ms_per_token': per_token_ms,
                        'sim_total_ms': total_ms,
                        'sim_energy_j': total_nj / 1e9,
                        'sim_tokens_per_s': self.gen_tokens / max(per_token_ms * self.gen_tokens / 1000.0, 1e-9),
                        'rfu_utilization': float(prefill_metrics['rfu_utilization']),
                        'result_reuse_rate': float(prefill_metrics['result_reuse_rate']),
                        'structural_reuse_rate': float(prefill_metrics['structural_reuse_rate']),
                        'switch_cycles_total': float(prefill_metrics['switch_cycles_total']),
                        'representative_paths': float(prefill_metrics.get('representative_path_count', 0.0)),
                        'fragments': len(prefill.fragments),
                    }
                )
        return pd.DataFrame(rows)

    # -------------------------------------------------- private-serving trace

    def _serving_workload(
        self,
        spec: LLMArchSpec,
        samples: Sequence[PromptSample],
        crypto_scale: float,
        window_ms: float,
    ) -> Tuple[List[TaskGraph], List[RuntimeTask]]:
        """Build one private-inference trace.

        Each request carries the three things such a gateway actually does: it
        decrypts the retrieved context and encrypts the reply (AES), answers the
        prompt (the language model), and proves what it was given and what it
        returned (Groth16).  ``crypto_scale`` multiplies the volume of the two
        security tasks, which is what moves the offered load between the
        inference-dominated and security-dominated ends of the mix.
        """
        graphs: List[TaskGraph] = []
        runtime_tasks: List[RuntimeTask] = []
        arrival_ms = 0.0
        aes_blocks = max(1, int(self.aes_kib_per_request * 64 * crypto_scale))
        proofs = max(0, int(round(self.proofs_per_request * crypto_scale)))
        for sample in samples:
            prompt_tokens = max(1, sample.prompt_tokens)
            stages: List[TaskGraph] = []
            if crypto_scale > 0:
                stages.append(
                    build_aes128_task(blocks=aes_blocks, task_id=f'{spec.name}_{sample.sample_id}_aes')
                )
            stages.append(
                build_llm_prefill_task(
                    spec,
                    seq_len=prompt_tokens,
                    width=self.width,
                    content_tag=f'{sample.sample_id}:serve',
                    task_id=f'{spec.name}_{sample.sample_id}_serve',
                )
            )
            for proof in range(proofs):
                stages.append(
                    build_groth16_like_task(
                        circuit_size=self.zkp_circuit_size,
                        rounds=4,
                        task_id=f'{spec.name}_{sample.sample_id}_zkp{proof}',
                    )
                )
            for stage_graph in stages:
                graphs.append(stage_graph)
                runtime_tasks.append(
                    RuntimeTask(
                        runtime_id=f'{stage_graph.task_id}_rt',
                        task_graph=stage_graph,
                        arrival_cycle=int(arrival_ms * CYCLES_PER_MS),
                    )
                )
                arrival_ms += window_ms
        return graphs, runtime_tasks

    def secure_serving_trace(
        self,
        model_meta: Dict[str, str],
        spec: LLMArchSpec,
        samples: Sequence[PromptSample],
        window_ms: float = 1.0,
        crypto_scale: float = 1.0,
    ) -> pd.DataFrame:
        """Replay the private-inference trace under FUSE and under multi-core.

        Only :data:`SERVING_BASELINE_MODES` appear here.  A Transformer ASIC or
        a tensor core cannot execute Groth16 or an AES round at all, so the
        meaningful comparison for a mixed trace is against a multi-core system
        that pays for one dedicated engine per domain -- the same baseline the
        mixed-task evaluation uses elsewhere.
        """
        serving_samples = list(samples)[: self.serving_requests]
        graphs, runtime_tasks = self._serving_workload(spec, serving_samples, crypto_scale, window_ms)
        decomposition = self._decompose(graphs)
        rows: List[Dict[str, Any]] = []
        for mode in (ACCELERATOR_NAME_MODE,) + SERVING_BASELINE_MODES:
            metrics, _ = self._simulate(graphs, runtime_tasks, mode, decomposition=decomposition)
            per_type = metrics.get('task_type_mean_latency_ms', {})
            rows.append(
                {
                    'model': model_meta['name'],
                    'model_label': model_meta['label'],
                    'family': model_meta['family'],
                    'mode': mode,
                    'mode_label': display_mode_name(mode),
                    'requests': len(serving_samples),
                    'crypto_scale': crypto_scale,
                    'aes_kib_per_request': self.aes_kib_per_request * crypto_scale,
                    'proofs_per_request': self.proofs_per_request * crypto_scale,
                    'makespan_ms': float(metrics['makespan_ms']),
                    'energy_j': float(metrics['energy_nj']) / 1e9,
                    'rfu_utilization': float(metrics['rfu_utilization']),
                    'result_reuse_rate': float(metrics['result_reuse_rate']),
                    'structural_reuse_rate': float(metrics['structural_reuse_rate']),
                    'switch_cycles_total': float(metrics['switch_cycles_total']),
                    'lrc_switch_events': float(metrics.get('lrc_switch_events', 0.0)),
                    'llm_latency_ms': float(per_type.get('LLM', 0.0)),
                    'aes_latency_ms': float(per_type.get('AES', 0.0)),
                    'zkp_latency_ms': float(per_type.get('ZKP', 0.0)),
                }
            )
        return pd.DataFrame(rows)

    def serving_mix_sweep(
        self,
        model_meta: Dict[str, str],
        spec: LLMArchSpec,
        samples: Sequence[PromptSample],
        scales: Sequence[float] = CRYPTO_SCALES,
    ) -> pd.DataFrame:
        """Sweep how much of the offered load is security work."""
        frames = [
            self.secure_serving_trace(model_meta, spec, samples, crypto_scale=scale) for scale in scales
        ]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ------------------------------------------------------------------- run

    def run(self, output_root: Path) -> LLMZooArtifacts:
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        native_frames: List[pd.DataFrame] = []
        simulated_frames: List[pd.DataFrame] = []
        serving_frames: List[pd.DataFrame] = []

        for model_meta in self.models:
            print(f'[llm-zoo] {model_meta["label"]} ({model_meta["hf_id"]})', flush=True)
            samples = self.load_prompts()
            native = None
            cached = output_root / model_meta['name'] / 'native.csv'
            if self.reuse_native and cached.exists():
                # Re-simulating must not mean re-measuring: the GPU numbers are a
                # property of the machine, not of the mapping being explored.
                native = pd.read_csv(cached)
                lengths = dict(zip(native['sample_id'], native['prompt_tokens']))
                missing = [s.sample_id for s in samples if s.sample_id not in lengths]
                if missing:
                    print(f'  ! cached measurements do not cover {len(missing)} prompts; re-measuring',
                          flush=True)
                    native = None
                else:
                    for sample in samples:
                        sample.prompt_tokens = int(lengths[sample.sample_id])
                    print(f'  reusing measurements from {cached}', flush=True)
            if native is None and self.run_native and self.torch is not None:
                try:
                    native = self.measure_native(model_meta, samples)
                except Exception as exc:  # a model that will not fit must not sink the run
                    print(f'  ! native run failed ({type(exc).__name__}: {exc}); '
                          f'falling back to tokenization only', flush=True)
                    if self.torch is not None and self.device.startswith('cuda'):
                        self.torch.cuda.empty_cache()
            if native is None:
                native = self.tokenize_only(model_meta, samples)
            native_frames.append(native)
            spec = self.arch_spec(model_meta)
            model_dir = output_root / model_meta['name']
            model_dir.mkdir(parents=True, exist_ok=True)
            native.to_csv(model_dir / 'native.csv', index=False)

            simulated = self.simulate_model(model_meta, spec, samples)
            simulated.to_csv(model_dir / 'simulated.csv', index=False)
            simulated_frames.append(simulated)

            if self.mix_sweep:
                serving = self.serving_mix_sweep(model_meta, spec, samples)
            else:
                serving = self.secure_serving_trace(model_meta, spec, samples)
            serving.to_csv(model_dir / 'secure_serving.csv', index=False)
            serving_frames.append(serving)

            (model_dir / 'arch.json').write_text(
                json.dumps(
                    {
                        **{k: v for k, v in model_meta.items()},
                        'layers': spec.layers,
                        'hidden': spec.hidden,
                        'heads': spec.heads,
                        'kv_heads': spec.kv_heads,
                        'head_dim': spec.head_dim,
                        'ff_dim': spec.ff_dim,
                        'vocab': spec.vocab,
                        'moe_experts': spec.moe_experts,
                        'moe_top_k': spec.moe_top_k,
                        'parameter_count': spec.parameter_count(),
                    },
                    indent=2,
                ),
                encoding='utf-8',
            )

        native_df = pd.concat(native_frames, ignore_index=True) if native_frames else pd.DataFrame()
        simulated_df = pd.concat(simulated_frames, ignore_index=True) if simulated_frames else pd.DataFrame()
        serving_df = pd.concat(serving_frames, ignore_index=True) if serving_frames else pd.DataFrame()
        summary_df = self._summarize(native_df, simulated_df, serving_df)

        native_df.to_csv(output_root / 'llm_native.csv', index=False)
        simulated_df.to_csv(output_root / 'llm_simulated.csv', index=False)
        serving_df.to_csv(output_root / 'llm_secure_serving.csv', index=False)
        summary_df.to_csv(output_root / 'llm_summary.csv', index=False)
        environment = self.environment()
        (output_root / 'llm_env.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')

        artifacts = LLMZooArtifacts(
            native=native_df,
            simulated=simulated_df,
            secure_serving=serving_df,
            summary=summary_df,
            environment=environment,
        )
        self.render_figures(artifacts, output_root)
        return artifacts

    def _summarize(
        self, native: pd.DataFrame, simulated: pd.DataFrame, serving: pd.DataFrame
    ) -> pd.DataFrame:
        if simulated.empty:
            return pd.DataFrame()
        rows: List[Dict[str, Any]] = []
        for (model, label, family), part in simulated.groupby(['model', 'model_label', 'family'], sort=False):
            fuse = part[part['mode'] == ACCELERATOR_NAME_MODE]
            if fuse.empty:
                continue
            record: Dict[str, Any] = {
                'model': model,
                'model_label': label,
                'family': family,
                'parameter_count': float(part['parameter_count'].iloc[0]),
                'prompt_tokens_mean': float(part['prompt_tokens'].mean()),
                'gen_tokens': int(part['gen_tokens'].iloc[0]),
                'fragments': float(fuse['fragments'].mean()),
                'fuse_prefill_ms': float(fuse['sim_prefill_ms'].mean()),
                'fuse_decode_ms_per_token': float(fuse['sim_decode_ms_per_token'].mean()),
                'fuse_total_ms': float(fuse['sim_total_ms'].mean()),
                'fuse_energy_j': float(fuse['sim_energy_j'].mean()),
                'fuse_rfu_utilization': float(fuse['rfu_utilization'].mean()),
                'fuse_structural_reuse_rate': float(fuse['structural_reuse_rate'].mean()),
                'fuse_switch_cycles': float(fuse['switch_cycles_total'].mean()),
            }
            for mode in LLM_BASELINE_MODES:
                other = part[part['mode'] == mode]
                if other.empty:
                    continue
                record[f'{mode}_total_ms'] = float(other['sim_total_ms'].mean())
                record[f'{mode}_energy_j'] = float(other['sim_energy_j'].mean())
                record[f'{mode}_latency_ratio'] = record['fuse_total_ms'] / max(
                    float(other['sim_total_ms'].mean()), 1e-9
                )
                record[f'{mode}_energy_gain'] = float(other['sim_energy_j'].mean()) / max(
                    record['fuse_energy_j'], 1e-9
                )
                record[f'{mode}_switch_cycles'] = float(other['switch_cycles_total'].mean())
            if not native.empty:
                native_part = native[native['model'] == model]
                if not native_part.empty and 'native_total_ms' in native_part.columns:
                    record['native_prefill_ms'] = float(native_part['native_prefill_ms'].mean())
                    record['native_decode_ms_per_token'] = float(
                        native_part['native_decode_ms_per_token'].mean()
                    )
                    record['native_total_ms'] = float(native_part['native_total_ms'].mean())
                    record['native_energy_j'] = float(native_part['native_energy_j'].mean())
                    record['native_power_w'] = float(native_part['native_power_w'].mean())
                    record['native_tokens_per_s'] = float(native_part['native_tokens_per_s'].mean())
                    record['energy_gain_vs_native'] = record['native_energy_j'] / max(
                        record['fuse_energy_j'], 1e-9
                    )
            if not serving.empty:
                serve_part = serving[serving['model'] == model]
                record.update(self._serving_record(serve_part))
            rows.append(record)
        return pd.DataFrame(rows)

    @staticmethod
    def _serving_record(serve_part: pd.DataFrame) -> Dict[str, Any]:
        """Headline numbers from one model's private-serving trace.

        The nominal mix (``crypto_scale == 1``) gives the headline; the sweep
        additionally reports where FUSE's advantage peaks, which is the point
        at which the security tasks stop being a rounding error next to the
        language model.
        """
        record: Dict[str, Any] = {}
        if serve_part.empty:
            return record
        rows: List[Dict[str, float]] = []
        for scale, part in serve_part.groupby('crypto_scale'):
            fuse_row = part[part['mode'] == ACCELERATOR_NAME_MODE]
            multi_row = part[part['mode'] == 'multicore']
            if fuse_row.empty or multi_row.empty:
                continue
            fuse_span = float(fuse_row['makespan_ms'].iloc[0])
            multi_span = float(multi_row['makespan_ms'].iloc[0])
            rows.append(
                {
                    'scale': float(scale),
                    'fuse_ms': fuse_span,
                    'multicore_ms': multi_span,
                    'latency_reduction': 1.0 - fuse_span / max(multi_span, 1e-9),
                    'energy_gain': float(multi_row['energy_j'].iloc[0])
                    / max(float(fuse_row['energy_j'].iloc[0]), 1e-9),
                    'reuse_rate': float(fuse_row['structural_reuse_rate'].iloc[0]),
                    'fuse_switch_cycles': float(fuse_row['switch_cycles_total'].iloc[0]),
                    'multicore_switch_cycles': float(multi_row['switch_cycles_total'].iloc[0]),
                    'fuse_aes_ms': float(fuse_row['aes_latency_ms'].iloc[0]),
                    'multicore_aes_ms': float(multi_row['aes_latency_ms'].iloc[0]),
                    'fuse_zkp_ms': float(fuse_row['zkp_latency_ms'].iloc[0]),
                    'multicore_zkp_ms': float(multi_row['zkp_latency_ms'].iloc[0]),
                }
            )
        if not rows:
            return record
        nominal = next((r for r in rows if abs(r['scale'] - 1.0) < 1e-9), rows[0])
        best = max(rows, key=lambda r: r['latency_reduction'])
        record['serving_fuse_ms'] = nominal['fuse_ms']
        record['serving_multicore_ms'] = nominal['multicore_ms']
        record['serving_latency_reduction'] = nominal['latency_reduction']
        record['serving_energy_gain'] = nominal['energy_gain']
        record['serving_fuse_reuse_rate'] = nominal['reuse_rate']
        record['serving_switch_reduction'] = 1.0 - nominal['fuse_switch_cycles'] / max(
            nominal['multicore_switch_cycles'], 1e-9
        )
        record['serving_best_crypto_scale'] = best['scale']
        record['serving_best_latency_reduction'] = best['latency_reduction']
        record['serving_best_energy_gain'] = best['energy_gain']
        # What privacy and integrity cost on top of inference alone, measured as
        # the makespan ratio between the secured trace and the scale-0 trace.
        baseline = next((r for r in rows if abs(r['scale']) < 1e-9), None)
        if baseline is not None:
            record['security_overhead_fuse'] = nominal['fuse_ms'] / max(baseline['fuse_ms'], 1e-9)
            record['security_overhead_multicore'] = nominal['multicore_ms'] / max(
                baseline['multicore_ms'], 1e-9
            )
            heaviest = max(rows, key=lambda r: r['scale'])
            record['security_overhead_fuse_max'] = heaviest['fuse_ms'] / max(baseline['fuse_ms'], 1e-9)
            record['security_overhead_multicore_max'] = heaviest['multicore_ms'] / max(
                baseline['multicore_ms'], 1e-9
            )
        if nominal['multicore_aes_ms'] > 0:
            record['serving_aes_latency_reduction'] = 1.0 - nominal['fuse_aes_ms'] / nominal['multicore_aes_ms']
        if nominal['multicore_zkp_ms'] > 0:
            record['serving_zkp_latency_reduction'] = 1.0 - nominal['fuse_zkp_ms'] / nominal['multicore_zkp_ms']
        # The nominal mix is the operating point, but a shared fabric can delay
        # the security path badly once security dominates the offered load, and
        # that worst case belongs in the record rather than only in the sweep.
        aes_reductions = [
            1.0 - r['fuse_aes_ms'] / r['multicore_aes_ms']
            for r in rows
            if r['multicore_aes_ms'] > 0
        ]
        if aes_reductions:
            record['serving_aes_worst_reduction'] = min(aes_reductions)
            worst_row = min(
                (r for r in rows if r['multicore_aes_ms'] > 0),
                key=lambda r: 1.0 - r['fuse_aes_ms'] / r['multicore_aes_ms'],
            )
            record['serving_aes_worst_scale'] = worst_row['scale']
        return record

    # --------------------------------------------------------------- figures

    def render_figures(self, artifacts: LLMZooArtifacts, output_root: Path) -> None:
        if artifacts.summary.empty:
            return
        output_root = Path(output_root)
        self._plot_latency(artifacts.summary, output_root / 'fig_llm_latency.pdf')
        self._plot_energy(artifacts.summary, output_root / 'fig_llm_energy.pdf')
        if not artifacts.secure_serving.empty:
            self._plot_serving(artifacts.secure_serving, output_root / 'fig_llm_secure_serving.pdf')

    @staticmethod
    def _ordered(summary: pd.DataFrame) -> pd.DataFrame:
        return summary.sort_values('parameter_count').reset_index(drop=True)

    def _plot_latency(self, summary: pd.DataFrame, out_path: Path) -> None:
        """Absolute latency spans three orders of magnitude while the overhead is
        a few per cent, so each gets its own panel."""
        plt = _plt()
        data = self._ordered(summary)
        labels = list(data['model_label'])
        x = np.arange(len(labels))
        modes = [ACCELERATOR_NAME_MODE] + list(LLM_BASELINE_MODES)
        colors = {'fuse': '#E76F51', 'transformer_asic': '#2A9D8F',
                  'gpu_tensor': '#264653', 'multicore': '#E9C46A'}
        fig, (ax_abs, ax_norm) = plt.subplots(1, 2, figsize=(11.6, 4.1))

        width = 0.8 / len(modes)
        for index, mode in enumerate(modes):
            column = 'fuse_total_ms' if mode == ACCELERATOR_NAME_MODE else f'{mode}_total_ms'
            if column not in data.columns:
                continue
            ax_abs.bar(
                x + (index - (len(modes) - 1) / 2) * width,
                data[column].astype(float),
                width=width,
                label=display_mode_name(mode),
                color=colors.get(mode, '#888888'),
                edgecolor='white',
                linewidth=0.4,
            )
        ax_abs.set_xticks(x)
        ax_abs.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
        ax_abs.set_ylabel('Request latency (ms, log)')
        ax_abs.set_yscale('log')
        ax_abs.set_title('(a) Absolute latency')
        ax_abs.legend(fontsize=7.6, frameon=False, ncol=2)
        ax_abs.grid(axis='y', alpha=0.25, linewidth=0.5)

        # Normalized to the domain-specific engine, which is where the few per
        # cent of reconfiguration overhead actually lives.
        reference = 'transformer_asic_total_ms'
        if reference in data.columns:
            ratio_modes = [ACCELERATOR_NAME_MODE, 'gpu_tensor', 'multicore']
            width = 0.8 / len(ratio_modes)
            for index, mode in enumerate(ratio_modes):
                column = 'fuse_total_ms' if mode == ACCELERATOR_NAME_MODE else f'{mode}_total_ms'
                if column not in data.columns:
                    continue
                values = data[column].astype(float) / data[reference].astype(float)
                bars = ax_norm.bar(
                    x + (index - (len(ratio_modes) - 1) / 2) * width,
                    values,
                    width=width,
                    label=display_mode_name(mode),
                    color=colors.get(mode, '#888888'),
                    edgecolor='white',
                    linewidth=0.4,
                )
                if mode == ACCELERATOR_NAME_MODE:
                    for bar, value in zip(bars, values):
                        ax_norm.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            value + 0.004,
                            f'{value:.3f}',
                            ha='center', va='bottom', fontsize=6.8, rotation=90,
                        )
            ax_norm.axhline(1.0, color='#333333', linewidth=0.9, linestyle='--')
            ax_norm.set_ylim(0.95, 1.10)
            ax_norm.set_xticks(x)
            ax_norm.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
            ax_norm.set_ylabel('Latency / Transformer-ASIC')
            ax_norm.set_title('(b) Normalized to the domain-specific engine')
            ax_norm.legend(fontsize=7.6, frameon=False, ncol=3)
            ax_norm.grid(axis='y', alpha=0.25, linewidth=0.5)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)

    def _plot_energy(self, summary: pd.DataFrame, out_path: Path) -> None:
        plt = _plt()
        data = self._ordered(summary)
        labels = list(data['model_label'])
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(9.6, 4.2))
        colors = {'transformer_asic': '#2A9D8F', 'gpu_tensor': '#264653', 'multicore': '#E9C46A'}
        width = 0.8 / max(1, len(LLM_BASELINE_MODES))
        for index, mode in enumerate(LLM_BASELINE_MODES):
            column = f'{mode}_energy_gain'
            if column not in data.columns:
                continue
            ax.bar(
                x + (index - (len(LLM_BASELINE_MODES) - 1) / 2) * width,
                data[column].astype(float),
                width=width,
                label=f'vs {display_mode_name(mode)}',
                color=colors.get(mode, '#888888'),
                edgecolor='white',
                linewidth=0.4,
            )
        ax.axhline(1.0, color='#333333', linewidth=0.8, linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12, ha='right')
        ax.set_ylabel(f'Energy of baseline / {ACCELERATOR_NAME}')
        ax.set_title('Energy efficiency gain per request')
        ax.legend(fontsize=8.5, frameon=False)
        ax.grid(axis='y', alpha=0.25, linewidth=0.5)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)

    def _plot_serving(self, serving: pd.DataFrame, out_path: Path) -> None:
        """What the shared fabric saves on the private-serving trace, and what
        adding the security tasks costs at all."""
        plt = _plt()
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.6, 4.1))

        labels: List[str] = []
        switch_reduction: List[float] = []
        energy_gain: List[float] = []
        overhead_labels: List[str] = []
        fuse_overhead: List[float] = []
        multi_overhead: List[float] = []

        for label, part in serving.groupby('model_label', sort=False):
            nominal = part[part['crypto_scale'] == 1.0]
            fuse_row = nominal[nominal['mode'] == ACCELERATOR_NAME_MODE]
            multi_row = nominal[nominal['mode'] == 'multicore']
            if not fuse_row.empty and not multi_row.empty:
                labels.append(label)
                switch_reduction.append(
                    100.0 * (1.0 - float(fuse_row['switch_cycles_total'].iloc[0])
                             / max(float(multi_row['switch_cycles_total'].iloc[0]), 1e-9))
                )
                energy_gain.append(
                    float(multi_row['energy_j'].iloc[0])
                    / max(float(fuse_row['energy_j'].iloc[0]), 1e-9)
                )
            spans = {}
            for scale in (0.0, 1.0):
                chunk = part[part['crypto_scale'] == scale]
                f_row = chunk[chunk['mode'] == ACCELERATOR_NAME_MODE]
                m_row = chunk[chunk['mode'] == 'multicore']
                if f_row.empty or m_row.empty:
                    break
                spans[scale] = (float(f_row['makespan_ms'].iloc[0]),
                                float(m_row['makespan_ms'].iloc[0]))
            if len(spans) == 2:
                overhead_labels.append(label)
                fuse_overhead.append(spans[1.0][0] / max(spans[0.0][0], 1e-9))
                multi_overhead.append(spans[1.0][1] / max(spans[0.0][1], 1e-9))

        x = np.arange(len(labels))
        width = 0.38
        bars = ax_left.bar(x - width / 2, switch_reduction, width, color='#2A9D8F')
        for bar, value in zip(bars, switch_reduction):
            ax_left.text(bar.get_x() + bar.get_width() / 2.0, value + 1.0, f'{value:.0f}',
                         ha='center', va='bottom', fontsize=7)
        ax_twin = ax_left.twinx()
        ax_twin.bar(x + width / 2, energy_gain, width, color='#E9C46A')
        ax_twin.set_ylim(0, max(energy_gain + [1.2]) * 1.35)
        ax_twin.set_ylabel('Energy of Multi-core / FUSE')
        ax_left.set_xticks(x)
        ax_left.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
        ax_left.set_ylabel('Task-switch cycles saved (%)')
        ax_left.set_ylim(0, max(switch_reduction + [10.0]) * 1.3)
        ax_left.set_title('(a) What the shared fabric saves')
        handles = [plt.Rectangle((0, 0), 1, 1, color='#2A9D8F'),
                   plt.Rectangle((0, 0), 1, 1, color='#E9C46A')]
        ax_left.legend(handles, ['Switch cycles saved', 'Energy gain'], fontsize=7.6,
                       frameon=False, loc='upper left')
        ax_left.grid(axis='y', alpha=0.25, linewidth=0.5)

        # The per-task security mean over a handful of tasks is too noisy to
        # plot; the overhead ratio is a whole-trace quantity and holds up.
        xo = np.arange(len(overhead_labels))
        ax_right.bar(xo - width / 2, fuse_overhead, width, label=ACCELERATOR_NAME, color='#E76F51')
        ax_right.bar(xo + width / 2, multi_overhead, width, label='Multi-core', color='#E9C46A')
        ax_right.axhline(1.0, color='#333333', linewidth=0.9, linestyle='--')
        ax_right.set_xticks(xo)
        ax_right.set_xticklabels(overhead_labels, rotation=20, ha='right', fontsize=8)
        ax_right.set_ylabel('Makespan with security / without')
        ax_right.set_ylim(0.95, max(fuse_overhead + multi_overhead + [1.05]) * 1.06)
        ax_right.set_title('(b) What adding privacy and integrity costs')
        ax_right.legend(fontsize=7.6, frameon=False)
        ax_right.grid(axis='y', alpha=0.25, linewidth=0.5)

        fig.tight_layout()
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
