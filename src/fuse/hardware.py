from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence
import json
import platform
import socket
import statistics
import sys
import time
import numpy as np
try:
    from cryptography import __version__ as cryptography_version
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
except Exception:
    cryptography_version = None
    Cipher = None
    algorithms = None
    modes = None
from .ir import RunResult, RuntimeTask, TaskGraph
from .sim.simulator import CYCLES_PER_MS

def _optional_torch():
    try:
        import torch
        return torch
    except Exception:
        return None

def torchvision_residual_block(torch, in_channels: int, hidden_channels: int, out_channels: int, *, stride: int, downsample):

    class ResidualBlock(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, bias=False)
            self.bn1 = torch.nn.BatchNorm2d(hidden_channels)
            self.conv2 = torch.nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = torch.nn.BatchNorm2d(hidden_channels)
            self.conv3 = torch.nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.bn3 = torch.nn.BatchNorm2d(out_channels)
            self.relu = torch.nn.ReLU(inplace=False)
            self.downsample = downsample

        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            if self.downsample is not None:
                identity = self.downsample(x)
            out = self.relu(out + identity)
            return out
    return ResidualBlock()

@dataclass
class HardwareEnvironment:
    hostname: str
    platform: str
    python: str
    numpy: str
    cryptography: str
    torch: str | None
    cuda_available: bool
    device: str
    device_name: str

@dataclass
class HardwareArtifacts:
    summary: Any
    mixed_trace: Any
    environment: Dict[str, Any]

class HardwareBenchmarkSuite:

    def __init__(self, device: str='auto', warmup: int=3, repeat: int=10, seed: int=7):
        self.requested_device = device
        self.warmup = warmup
        self.repeat = repeat
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.torch = _optional_torch()
        self.device = self._resolve_device(device)

    def run(self, output_dir: Path | None=None, *, cnn_models: Sequence[str]=('vgg16', 'resnet50', 'resnet101'), cnn_batches: Sequence[int]=(1, 4, 8), cnn_image_size: int | None=None, cnn_width_mult: float | None=None, aes_key_bits: Sequence[int]=(128, 256), aes_blocks: Sequence[int]=(1024, 4096, 16384), zkp_sizes: Sequence[int]=(4096, 16384, 65536), mixed_rounds: int=3) -> HardwareArtifacts:
        rows: List[Dict[str, Any]] = []
        rows.extend(self._run_cnn(cnn_models, cnn_batches, image_size=cnn_image_size, width_mult=cnn_width_mult))
        rows.extend(self._run_aes(aes_key_bits, aes_blocks))
        rows.extend(self._run_zkp(zkp_sizes))
        summary = self._to_frame(rows)
        mixed_trace = self._to_frame(self._run_mixed_trace(mixed_rounds, cnn_model=cnn_models[0] if cnn_models else 'vgg16', aes_key_bits=aes_key_bits[0] if aes_key_bits else 128, cnn_image_size=cnn_image_size, cnn_width_mult=cnn_width_mult))
        environment = asdict(self.environment())
        artifacts = HardwareArtifacts(summary=summary, mixed_trace=mixed_trace, environment=environment)
        if output_dir is not None:
            self._write_outputs(Path(output_dir), artifacts)
        return artifacts

    def environment(self) -> HardwareEnvironment:
        torch_version = getattr(self.torch, '__version__', None)
        cuda_available = bool(self.torch is not None and self.torch.cuda.is_available())
        device_name = 'CPU'
        if self.device.startswith('cuda') and self.torch is not None and cuda_available:
            device_name = self.torch.cuda.get_device_name(0)
        elif self.device == 'cpu' and self.torch is not None:
            device_name = 'CPU'
        return HardwareEnvironment(hostname=socket.gethostname(), platform=platform.platform(), python=sys.version.split()[0], numpy=np.__version__, cryptography=cryptography_version, torch=torch_version, cuda_available=cuda_available, device=self.device, device_name=device_name)

    def _resolve_device(self, requested: str) -> str:
        if requested != 'auto':
            if requested.startswith('cuda'):
                if self.torch is None or not self.torch.cuda.is_available():
                    raise RuntimeError('CUDA was requested but torch with CUDA support is not available')
            return requested
        if self.torch is not None and self.torch.cuda.is_available():
            return 'cuda:0'
        return 'cpu'

    def run_task_graphs(self, task_graphs: Sequence[TaskGraph], runtime_tasks: Sequence[RuntimeTask], *, output_dir: Path | None=None) -> tuple[RunResult, Any]:
        import pandas as pd
        kernels = {graph.task_id: self._make_task_kernel(graph) for graph in task_graphs}
        rows: List[Dict[str, Any]] = []
        clock_ms = 0.0
        dispatches = 0
        for runtime_task in sorted(runtime_tasks, key=lambda item: item.arrival_cycle):
            arrival_ms = runtime_task.arrival_cycle / CYCLES_PER_MS
            if arrival_ms > clock_ms:
                clock_ms = arrival_ms
            kernel = kernels[runtime_task.task_graph.task_id]
            start_ms = clock_ms
            stats = self._measure(kernel['fn'], work_items=float(kernel['work_items']), work_label=kernel['work_label'])
            finish_ms = start_ms + float(stats['mean_ms'])
            rows.append({'event': 'dispatch', 'cycle': int(start_ms * CYCLES_PER_MS), 'finish_cycle': int(finish_ms * CYCLES_PER_MS), 'runtime_micro_id': runtime_task.runtime_id, 'task_runtime_id': runtime_task.runtime_id, 'task_type': runtime_task.task_graph.task_type, 'template_family': kernel['template_family'], 'kernel': kernel.get('kernel_name', kernel['template_family']), 'reused_result': False, 'structural_reuse': False, 'switch_cycles': 0, 'rfu_id': 0, 'backend': kernel['backend']})
            rows.append({'event': 'complete', 'cycle': int(finish_ms * CYCLES_PER_MS), 'finish_cycle': int(finish_ms * CYCLES_PER_MS), 'runtime_micro_id': runtime_task.runtime_id, 'task_runtime_id': runtime_task.runtime_id, 'task_type': runtime_task.task_graph.task_type, 'template_family': kernel['template_family'], 'kernel': kernel.get('kernel_name', kernel['template_family']), 'reused_result': False, 'structural_reuse': False, 'switch_cycles': 0, 'rfu_id': 0, 'backend': kernel['backend']})
            clock_ms = finish_ms
            dispatches += 1
        events = pd.DataFrame(rows)
        power_w = 45.0 if self.device.startswith('cuda') else 12.0
        makespan_ms = clock_ms
        energy_nj = makespan_ms * power_w * 1000000.0
        metrics = {'mode': 'hardware_gpu' if self.device.startswith('cuda') else 'hardware_cpu', 'backend': 'torch_cuda' if self.device.startswith('cuda') else 'torch_cpu' if self.torch is not None else 'numpy_cpu', 'device': self.device, 'makespan_cycles': int(makespan_ms * CYCLES_PER_MS), 'makespan_ms': makespan_ms, 'dispatches': dispatches, 'result_reuse_hits': 0, 'structural_reuse_hits': 0, 'result_reuse_rate': 0.0, 'structural_reuse_rate': 0.0, 'switch_cycles_total': 0.0, 'rfu_utilization': 1.0 if dispatches else 0.0, 'energy_nj': energy_nj, 'task_latency_ms': {runtime_task.runtime_id: float(events[events['task_runtime_id'] == runtime_task.runtime_id]['finish_cycle'].max() - runtime_task.arrival_cycle) / CYCLES_PER_MS for runtime_task in runtime_tasks}, 'task_type_mean_latency_ms': self._task_type_latency(events, runtime_tasks), 'fvm_reads': 0.0, 'fvm_writes': 0.0, 'fvm_allocations': 0.0, 'fvm_frees': 0.0, 'fvm_avg_residency_bytes': 0.0, 'fvm_redundant_reads': 0.0, 'lrc_switch_events': 0.0, 'scheduler_decisions': float(dispatches), 'scheduler_pipeline_stall_cycles': 0.0, 'kappa_key_match_ratio': 0.0, 'template_hit_ratio': 0.0, 'valid_fragment_ratio': 0.0}
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            events.to_csv(output_dir / 'hardware_events.csv', index=False)
            with (output_dir / 'hardware_metrics.json').open('w', encoding='utf-8') as fh:
                json.dump(metrics, fh, indent=2)
        return (RunResult(metrics=metrics, traces={'events': events.to_dict(orient='records')}, tables={}), events)

    def _sync(self) -> None:
        if self.torch is not None and self.device.startswith('cuda'):
            self.torch.cuda.synchronize()

    def _measure(self, fn: Callable[[], Any], *, work_items: float, work_label: str) -> Dict[str, float]:
        for _ in range(self.warmup):
            fn()
        samples_ms: List[float] = []
        for _ in range(self.repeat):
            self._sync()
            start = time.perf_counter_ns()
            fn()
            self._sync()
            end = time.perf_counter_ns()
            samples_ms.append((end - start) / 1000000.0)
        mean_ms = statistics.fmean(samples_ms)
        p50_ms = float(np.percentile(samples_ms, 50))
        p99_ms = float(np.percentile(samples_ms, 99))
        min_ms = min(samples_ms)
        max_ms = max(samples_ms)
        throughput = work_items / max(mean_ms / 1000.0, 1e-09)
        return {'mean_ms': mean_ms, 'p50_ms': p50_ms, 'p99_ms': p99_ms, 'min_ms': min_ms, 'max_ms': max_ms, 'throughput': throughput, 'work_items': work_items, 'repeat': self.repeat, 'warmup': self.warmup, 'work_label': work_label}

    def _run_cnn(self, models: Sequence[str], batches: Sequence[int], *, image_size: int | None, width_mult: float | None) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        resolved_image = image_size or (224 if self.device.startswith('cuda') else 96)
        for model_name in models:
            model_key = model_name.lower()
            if self.torch is not None and resolved_image >= 32:
                resolved_width = width_mult or (1.0 if self.device.startswith('cuda') else 0.25)
                model, dtype, kernel_name = self._build_torch_cnn_model(model_key, resolved_width)
                model.eval()
                model = model.to(self.device, dtype=dtype)
                for batch in batches:
                    kernel = self._make_cnn_kernel(batch=batch, image_size=resolved_image, width_mult=resolved_width, model=model, dtype=dtype)
                    with self.torch.inference_mode():
                        stats = self._measure(kernel, work_items=float(batch), work_label='images/s')
                    rows.append({'task': 'CNN', 'kernel': kernel_name, 'backend': 'torch_cuda' if self.device.startswith('cuda') else 'torch_cpu', 'device': self.device, 'config': f'model={model_key},batch={batch},image={resolved_image},width_mult={resolved_width}', 'source': 'hardware_timer', **stats})
                continue
            resolved_width = width_mult or 0.125
            for batch in batches:
                kernel = self._make_cnn_kernel(batch=batch, image_size=resolved_image, width_mult=resolved_width)
                stats = self._measure(kernel, work_items=float(batch), work_label='images/s')
                rows.append({'task': 'CNN', 'kernel': f'{model_key}_inference', 'backend': 'numpy_cpu', 'device': 'cpu', 'config': f'model={model_key},batch={batch},image={resolved_image},width_mult={resolved_width}', 'source': 'hardware_timer', **stats})
        return rows

    def _run_aes(self, key_bits_list: Sequence[int], blocks_list: Sequence[int]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for key_bits in key_bits_list:
            for blocks in blocks_list:
                _kernel = self._make_aes_kernel(blocks, key_bits=key_bits)
                stats = self._measure(_kernel, work_items=float(blocks * 16), work_label='bytes/s')
                rows.append({'task': 'AES', 'kernel': f'aes{key_bits}_ecb_encrypt', 'backend': 'cryptography_cpu' if Cipher is not None else 'numpy_cpu', 'device': 'cpu', 'config': f'key_bits={key_bits},blocks={blocks}', 'source': 'hardware_timer', **stats})
        return rows

    def _run_zkp(self, sizes: Sequence[int]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        prime = 2013265921
        for size in sizes:
            rows.append(self._measure_field_mac(size, prime))
            rows.append(self._measure_fft(size))
        return rows

    def _measure_field_mac(self, size: int, prime: int) -> Dict[str, Any]:
        _kernel, backend = self._make_field_mac_kernel(size, prime)
        stats = self._measure(_kernel, work_items=float(size), work_label='field_ops/s')
        return {'task': 'ZKP', 'kernel': 'field_mac', 'backend': backend, 'device': self.device if backend.startswith('torch') else 'cpu', 'config': f'size={size},prime={prime}', 'source': 'hardware_timer', **stats}

    def _measure_fft(self, size: int) -> Dict[str, Any]:
        if self.torch is not None:
            dtype = self.torch.complex64
            x = self.torch.randn(size, device=self.device, dtype=self.torch.float32) + 1j * self.torch.randn(size, device=self.device, dtype=self.torch.float32)
            x = x.to(dtype)

            def _kernel() -> Any:
                return self.torch.fft.fft(x)
            stats = self._measure(_kernel, work_items=float(size), work_label='fft_points/s')
            return {'task': 'ZKP', 'kernel': 'fft_poly', 'backend': 'torch_cuda' if self.device.startswith('cuda') else 'torch_cpu', 'device': self.device, 'config': f'size={size}', 'source': 'hardware_timer', **stats}
        x = self.rng.standard_normal(size).astype(np.float32) + 1j * self.rng.standard_normal(size).astype(np.float32)

        def _kernel() -> Any:
            return np.fft.fft(x)
        stats = self._measure(_kernel, work_items=float(size), work_label='fft_points/s')
        return {'task': 'ZKP', 'kernel': 'fft_poly', 'backend': 'numpy_cpu', 'device': 'cpu', 'config': f'size={size}', 'source': 'hardware_timer', **stats}

    def _task_type_latency(self, events, runtime_tasks: Sequence[RuntimeTask]) -> Dict[str, float]:
        values: Dict[str, List[float]] = {}
        for runtime_task in runtime_tasks:
            completed = events[(events['event'] == 'complete') & (events['task_runtime_id'] == runtime_task.runtime_id)]
            finish_cycle = int(completed['finish_cycle'].max()) if not completed.empty else runtime_task.arrival_cycle
            latency_ms = float(finish_cycle - runtime_task.arrival_cycle) / CYCLES_PER_MS
            values.setdefault(runtime_task.task_graph.task_type, []).append(latency_ms)
        return {task_type: float(statistics.fmean(samples)) for task_type, samples in values.items()}

    def _make_task_kernel(self, graph: TaskGraph) -> Dict[str, Any]:
        model = str(graph.metadata.get('model', '')).lower()
        if graph.task_type == 'LLM':
            seq_len = int(graph.metadata.get('seq_len', 512))
            hidden_dim = int(graph.metadata.get('hidden_dim', 1024))
            heads = int(graph.metadata.get('heads', 8))
            batch_size = int(graph.metadata.get('batch_size', 1))
            fn = self._make_llm_kernel(seq_len=seq_len, hidden_dim=hidden_dim, heads=heads, batch_size=batch_size)
            return {'fn': fn, 'backend': 'torch_cuda' if self.device.startswith('cuda') else 'torch_cpu', 'work_items': batch_size, 'work_label': 'requests/s', 'template_family': 'linear_map'}
        if graph.task_type == 'CNN' and model == 'mlp-path':
            batch_size = int(graph.metadata.get('batch_size', 1))
            hidden_dim = int(graph.metadata.get('hidden_dim', 512))
            fn = self._make_mlp_kernel(batch_size=batch_size, hidden_dim=hidden_dim)
            return {'fn': fn, 'backend': 'torch_cuda' if self.device.startswith('cuda') else 'torch_cpu' if self.torch is not None else 'numpy_cpu', 'work_items': batch_size, 'work_label': 'samples/s', 'template_family': 'linear_map'}
        if graph.task_type == 'CNN':
            batch_size = int(graph.metadata.get('batch_size', 1))
            image_size = 224 if self.device.startswith('cuda') else 96
            width_mult = 1.0 if self.device.startswith('cuda') else 0.25
            model_name = str(graph.metadata.get('model', 'vgg16')).replace('-like', '').lower()
            if self.torch is not None:
                model_impl, dtype, kernel_name = self._build_torch_cnn_model(model_name, width_mult)
                model_impl.eval()
                model_impl = model_impl.to(self.device, dtype=dtype)
                fn = self._make_cnn_kernel(batch=batch_size, image_size=image_size, width_mult=width_mult, model=model_impl, dtype=dtype)
                backend = 'torch_cuda' if self.device.startswith('cuda') else 'torch_cpu'
            else:
                fn = self._make_cnn_kernel(batch=batch_size, image_size=image_size, width_mult=width_mult)
                backend = 'numpy_cpu'
                kernel_name = f'{model_name}_inference'
            return {'fn': fn, 'backend': backend, 'work_items': batch_size, 'work_label': 'images/s', 'template_family': 'conv2d', 'kernel_name': kernel_name}
        if graph.task_type == 'AES':
            blocks = int(graph.metadata.get('blocks', 1024))
            key_bits = int(graph.metadata.get('key_bits', 128))
            return {'fn': self._make_aes_kernel(blocks, key_bits=key_bits), 'backend': 'cryptography_cpu' if Cipher is not None else 'numpy_cpu', 'work_items': blocks * 16, 'work_label': 'bytes/s', 'template_family': 'sbox_lookup', 'kernel_name': f'aes{key_bits}_ecb_encrypt'}
        if graph.task_type == 'ZKP':
            size = int(graph.metadata.get('circuit_size', 4096))
            fn, backend = self._make_field_mac_kernel(size, 2013265921)
            return {'fn': fn, 'backend': backend, 'work_items': size, 'work_label': 'field_ops/s', 'template_family': 'field_mul'}
        fn = lambda: None
        return {'fn': fn, 'backend': 'noop', 'work_items': 1.0, 'work_label': 'tasks/s', 'template_family': 'generic'}

    def _run_mixed_trace(self, rounds: int, *, cnn_model: str, aes_key_bits: int, cnn_image_size: int | None, cnn_width_mult: float | None) -> List[Dict[str, Any]]:
        rounds = max(1, rounds)
        resolved_image = cnn_image_size or (224 if self.device.startswith('cuda') else 32)
        resolved_width = cnn_width_mult or (1.0 if self.device.startswith('cuda') else 0.125)
        if self.torch is not None and resolved_image >= 32:
            model, dtype, _ = self._build_torch_cnn_model(cnn_model.lower(), resolved_width)
            model.eval()
            model = model.to(self.device, dtype=dtype)
            cnn_kernel = self._make_cnn_kernel(batch=1, image_size=resolved_image, width_mult=resolved_width, model=model, dtype=dtype)
            zkp_kernel, zkp_backend = self._make_field_mac_kernel(4096, 2013265921)
        else:
            cnn_kernel = self._make_cnn_kernel(batch=1, image_size=resolved_image, width_mult=resolved_width)
            zkp_kernel, zkp_backend = self._make_field_mac_kernel(4096, 2013265921)
        aes_kernel = self._make_aes_kernel(1024, key_bits=aes_key_bits)

        def _kernel() -> Any:
            for _ in range(rounds):
                zkp_kernel()
                cnn_kernel()
                aes_kernel()
            return None
        stats = self._measure(_kernel, work_items=float(rounds * 3), work_label='tasks/s')
        return [{'task': 'Mixed', 'kernel': 'zkp_cnn_aes_round_robin', 'backend': self._mixed_backend_label({'backend': 'torch_cuda' if self.torch is not None and self.device.startswith('cuda') else 'torch_cpu' if self.torch is not None else 'numpy_cpu'}, {'backend': 'cryptography_cpu'}, {'backend': zkp_backend}), 'device': self.device, 'config': f'rounds={rounds}', 'source': 'hardware_timer', **stats}]

    def _mixed_backend_label(self, cnn_row: Dict[str, Any], aes_row: Dict[str, Any], zkp_row: Dict[str, Any]) -> str:
        labels = {cnn_row['backend'], aes_row['backend'], zkp_row['backend']}
        return '+'.join(sorted(labels))

    def _build_torch_vggish(self, width_mult: float):
        torch = self.torch
        assert torch is not None
        channels = [max(8, int(c * width_mult)) for c in [64, 128, 256, 512, 512]]
        blocks = [2, 2, 3, 3, 3] if width_mult >= 0.75 else [1, 1, 2, 2, 2]
        layers: List[Any] = []
        in_channels = 3
        for block_depth, out_channels in zip(blocks, channels):
            for _ in range(block_depth):
                layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                layers.append(torch.nn.ReLU(inplace=False))
                in_channels = out_channels
            layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(torch.nn.AdaptiveAvgPool2d((1, 1)))
        model = torch.nn.Sequential(*layers)
        dtype = torch.float16 if self.device.startswith('cuda') else torch.float32
        return (model, dtype)

    def _build_torch_resnetish(self, depth: int, width_mult: float):
        torch = self.torch
        assert torch is not None
        block_map = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3]}
        widths = [max(32, int(c * width_mult)) for c in [64, 128, 256, 512]]
        blocks = block_map[depth]
        layers: List[Any] = [torch.nn.Conv2d(3, widths[0], kernel_size=7, stride=2, padding=3, bias=False), torch.nn.BatchNorm2d(widths[0]), torch.nn.ReLU(inplace=False), torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        in_channels = widths[0]
        for stage_idx, (num_blocks, out_channels) in enumerate(zip(blocks, widths)):
            stride = 1 if stage_idx == 0 else 2
            for block_idx in range(num_blocks):
                current_stride = stride if block_idx == 0 else 1
                hidden = max(32, out_channels // 4)
                downsample = None
                if current_stride != 1 or in_channels != out_channels:
                    downsample = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=current_stride, bias=False), torch.nn.BatchNorm2d(out_channels))
                layers.append(torchvision_residual_block(torch, in_channels, hidden, out_channels, stride=current_stride, downsample=downsample))
                in_channels = out_channels
        layers.append(torch.nn.AdaptiveAvgPool2d((1, 1)))
        model = torch.nn.Sequential(*layers)
        dtype = torch.float16 if self.device.startswith('cuda') else torch.float32
        return (model, dtype)

    def _build_torch_cnn_model(self, model_name: str, width_mult: float):
        if model_name == 'vgg16':
            model, dtype = self._build_torch_vggish(width_mult)
            return (model, dtype, 'vgg16_inference')
        if model_name == 'resnet50':
            model, dtype = self._build_torch_resnetish(50, width_mult)
            return (model, dtype, 'resnet50_inference')
        if model_name == 'resnet101':
            model, dtype = self._build_torch_resnetish(101, width_mult)
            return (model, dtype, 'resnet101_inference')
        raise ValueError(f'unsupported cnn model: {model_name}')

    def _build_numpy_vggish_weights(self, width_mult: float) -> List[np.ndarray]:
        channels = [max(8, int(c * width_mult)) for c in [32, 64]]
        dims = [3, *channels]
        weights: List[np.ndarray] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            weights.append(self.rng.standard_normal((in_dim, out_dim), dtype=np.float32) / max(in_dim, 1))
        return weights

    def _make_cnn_kernel(self, *, batch: int, image_size: int, width_mult: float, model=None, dtype=None) -> Callable[[], Any]:
        if self.torch is not None and model is not None:
            x = self.torch.randn(batch, 3, image_size, image_size, device=self.device, dtype=dtype)
            return lambda: model(x)
        x = self.rng.standard_normal((batch, 3), dtype=np.float32)
        weights = self._build_numpy_vggish_weights(width_mult)

        def _kernel() -> Any:
            out = x
            for weight in weights:
                out = np.maximum(out @ weight, 0.0)
            return out
        return _kernel

    def _make_mlp_kernel(self, *, batch_size: int, hidden_dim: int) -> Callable[[], Any]:
        if self.torch is not None:
            dtype = self.torch.float16 if self.device.startswith('cuda') else self.torch.float32
            model = self.torch.nn.Sequential(self.torch.nn.Linear(hidden_dim, hidden_dim * 2), self.torch.nn.GELU(), self.torch.nn.Linear(hidden_dim * 2, hidden_dim)).to(self.device, dtype=dtype)
            x = self.torch.randn(batch_size, hidden_dim, device=self.device, dtype=dtype)
            return lambda: model(x)
        x = self.rng.standard_normal((batch_size, hidden_dim), dtype=np.float32)
        w1 = self.rng.standard_normal((hidden_dim, hidden_dim * 2), dtype=np.float32)
        w2 = self.rng.standard_normal((hidden_dim * 2, hidden_dim), dtype=np.float32)
        return lambda: np.tanh(x @ w1) @ w2

    def _make_llm_kernel(self, *, seq_len: int, hidden_dim: int, heads: int, batch_size: int) -> Callable[[], Any]:
        if self.torch is None:
            q = self.rng.standard_normal((batch_size, seq_len, hidden_dim), dtype=np.float32)
            k = self.rng.standard_normal((batch_size, seq_len, hidden_dim), dtype=np.float32)
            v = self.rng.standard_normal((batch_size, seq_len, hidden_dim), dtype=np.float32)
            ff1 = self.rng.standard_normal((hidden_dim, hidden_dim * 2), dtype=np.float32)
            ff2 = self.rng.standard_normal((hidden_dim * 2, hidden_dim), dtype=np.float32)
            scale = max(hidden_dim // max(1, heads), 1) ** (-0.5)

            def _kernel() -> Any:
                score = q @ np.swapaxes(k, -1, -2) * scale
                score = np.exp(score - np.max(score, axis=-1, keepdims=True))
                attn = score / np.maximum(np.sum(score, axis=-1, keepdims=True), 1e-06)
                ctx = attn @ v
                hidden = np.tanh(ctx @ ff1)
                return hidden @ ff2
            return _kernel
        torch = self.torch
        dtype = torch.float16 if self.device.startswith('cuda') else torch.float32
        model_dim = hidden_dim
        head_dim = max(64, hidden_dim // max(1, heads))
        q = torch.randn(batch_size, seq_len, model_dim, device=self.device, dtype=dtype)
        k = torch.randn(batch_size, seq_len, model_dim, device=self.device, dtype=dtype)
        v = torch.randn(batch_size, seq_len, model_dim, device=self.device, dtype=dtype)
        w1 = torch.randn(model_dim, model_dim * 2, device=self.device, dtype=dtype)
        w2 = torch.randn(model_dim * 2, model_dim, device=self.device, dtype=dtype)
        scale = float(head_dim) ** (-0.5)

        def _kernel() -> Any:
            score = torch.matmul(q, k.transpose(-1, -2)) * scale
            attn = torch.softmax(score, dim=-1)
            ctx = torch.matmul(attn, v)
            hidden = torch.nn.functional.gelu(torch.matmul(ctx, w1))
            return torch.matmul(hidden, w2)
        return _kernel

    def _make_aes_kernel(self, blocks: int, *, key_bits: int=128) -> Callable[[], Any]:
        if Cipher is None or algorithms is None or modes is None:
            data = self.rng.integers(0, 256, size=blocks * 16, dtype=np.uint8).astype(np.uint8)
            return lambda: np.bitwise_xor(data, np.uint8(165))
        key = bytes(self.rng.integers(0, 256, size=key_bits // 8, dtype=np.uint8).tolist())
        data = bytes(self.rng.integers(0, 256, size=blocks * 16, dtype=np.uint8).tolist())

        def _kernel() -> Any:
            encryptor = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
            return encryptor.update(data) + encryptor.finalize()
        return _kernel

    def _make_field_mac_kernel(self, size: int, prime: int) -> tuple[Callable[[], Any], str]:
        if self.torch is not None:
            a = self.torch.randint(0, prime, (size,), device=self.device, dtype=self.torch.int64)
            b = self.torch.randint(0, prime, (size,), device=self.device, dtype=self.torch.int64)

            def _kernel() -> Any:
                prod = a * b % prime
                return int(prod.sum().item() % prime)
            backend = 'torch_cuda' if self.device.startswith('cuda') else 'torch_cpu'
            return (_kernel, backend)
        a = self.rng.integers(0, prime, size=size, dtype=np.int64)
        b = self.rng.integers(0, prime, size=size, dtype=np.int64)

        def _kernel() -> Any:
            prod = a * b % prime
            return int(np.sum(prod, dtype=np.int64) % prime)
        return (_kernel, 'numpy_cpu')

    def _to_frame(self, rows: Sequence[Dict[str, Any]]):
        import pandas as pd
        return pd.DataFrame(rows)

    def _write_outputs(self, output_dir: Path, artifacts: HardwareArtifacts) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts.summary.to_csv(output_dir / 'hardware_summary.csv', index=False)
        artifacts.mixed_trace.to_csv(output_dir / 'hardware_mixed_trace.csv', index=False)
        (output_dir / 'hardware_env.json').write_text(json.dumps(artifacts.environment, indent=2), encoding='utf-8')
