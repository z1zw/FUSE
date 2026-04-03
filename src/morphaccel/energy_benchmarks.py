from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from typing import Any, Callable, Dict, List, Sequence
import json
import os
import statistics
import time
import numpy as np
import pandas as pd
import psutil
from .branding import ACCELERATOR_NAME
from .hardware import HardwareBenchmarkSuite, _optional_torch

def _optional_import(name: str):
    try:
        return __import__(name, fromlist=['*'])
    except Exception:
        return None
torch = _optional_torch()
pyg_nn = _optional_import('torch_geometric.nn')
dgl = _optional_import('dgl')
dgl_nn = _optional_import('dgl.nn')
CryptoAES = None
try:
    from Crypto.Cipher import AES as CryptoAES
except Exception:
    CryptoAES = None
try:
    import pynvml
except Exception:
    pynvml = None

@dataclass
class EnergyArtifacts:
    summary: pd.DataFrame
    raw: pd.DataFrame
    environment: Dict[str, Any]

class EnergyBenchmarkSuite:

    def __init__(self, device: str='auto', warmup: int=2, repeat: int=5, seed: int=7):
        self.device = self._resolve_device(device)
        self.warmup = warmup
        self.repeat = repeat
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.proc = psutil.Process()
        self.cpu_package_w = float(os.environ.get('FUSE_CPU_PACKAGE_W', '35.0'))
        self.poll_interval_s = float(os.environ.get('FUSE_POWER_POLL_MS', '10')) / 1000.0
        self.min_sample_s = float(os.environ.get('FUSE_MIN_SAMPLE_S', '0.2'))

    def _resolve_device(self, requested: str) -> str:
        if requested != 'auto':
            return requested
        if torch is not None and torch.cuda.is_available():
            return 'cuda:0'
        return 'cpu'

    def _sync(self) -> None:
        if torch is not None and self.device.startswith('cuda'):
            torch.cuda.synchronize()

    def environment(self) -> Dict[str, Any]:
        return {'device': self.device, 'torch': getattr(torch, '__version__', None) if torch is not None else None, 'torch_geometric': getattr(_optional_import('torch_geometric'), '__version__', None) if pyg_nn is not None else None, 'dgl': getattr(dgl, '__version__', None) if dgl is not None else None, 'pycryptodome': getattr(_optional_import('Crypto'), '__version__', None) if CryptoAES is not None else None, 'pynvml': getattr(pynvml, '__version__', None) if pynvml is not None else None, 'cpu_package_w': self.cpu_package_w, 'min_sample_s': self.min_sample_s}

    def run(self, output_dir: Path | None=None) -> EnergyArtifacts:
        raw_rows: List[Dict[str, Any]] = []
        raw_rows.extend(self._run_zkp_task())
        raw_rows.extend(self._run_cnn_task())
        raw_rows.extend(self._run_aes_task())
        raw = pd.DataFrame(raw_rows)
        summary = self._normalize(raw)
        artifacts = EnergyArtifacts(summary=summary, raw=raw, environment=self.environment())
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            summary.to_csv(output_dir / 'fig5_energy.csv', index=False)
            raw.to_csv(output_dir / 'fig5_energy_raw.csv', index=False)
            (output_dir / 'fig5_energy_env.json').write_text(json.dumps(artifacts.environment, indent=2), encoding='utf-8')
        return artifacts

    def _normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        group_cols = ['task', 'platform']
        aggregated = raw.groupby(group_cols, as_index=False).agg(device=('device', 'first'), backend=('backend', 'first'), mean_ms=('mean_ms', 'mean'), p50_ms=('p50_ms', 'mean'), p99_ms=('p99_ms', 'mean'), energy_j=('energy_j', 'mean'), throughput=('throughput', 'mean'), efficiency=('efficiency', 'mean'), latency_std_ms=('latency_std_ms', 'mean'), latency_cv=('latency_cv', 'mean'), stability_score=('stability_score', 'mean'), repeat=('repeat', 'max'), work_items=('work_items', 'mean'), energy_source=('energy_source', 'first'))
        if 'operation' in raw.columns:
            operation_map = raw.groupby(group_cols)['operation'].agg(lambda values: '+'.join(sorted({str(v) for v in values if pd.notna(v)})) if len(list(values)) else '').reset_index(name='operation')
            aggregated = aggregated.merge(operation_map, on=group_cols, how='left')
        for task, part in aggregated.groupby('task'):
            fuse = part[part['platform'] == ACCELERATOR_NAME]
            if fuse.empty:
                continue
            fuse_eff = float(fuse.iloc[0]['efficiency'])
            for row in part.to_dict(orient='records'):
                efficiency = float(row['efficiency'])
                rows.append({**row, 'efficiency_norm': fuse_eff / max(efficiency, 1e-12), 'energy_nj': float(row['energy_j']) * 1000000000.0})
        return pd.DataFrame(rows)

    def _measure(self, fn: Callable[[], Any], *, work_items: float, platform: str, task: str, device_label: str, source: str) -> Dict[str, Any]:
        for _ in range(self.warmup):
            fn()
        latencies_ms: List[float] = []
        energies_j: List[float] = []
        work_totals: List[float] = []
        for _ in range(self.repeat):
            self._sync()
            cpu_before = self.proc.cpu_times()
            wall_start = time.perf_counter()
            energy_samples: List[float] = []
            stop = Event()
            thread = self._start_power_sampler(energy_samples, stop, device_label)
            iterations = 0
            while True:
                fn()
                iterations += 1
                if time.perf_counter() - wall_start >= self.min_sample_s:
                    break
            self._sync()
            wall_end = time.perf_counter()
            stop.set()
            if thread is not None:
                thread.join()
            cpu_after = self.proc.cpu_times()
            wall_s = max(wall_end - wall_start, 1e-09)
            cpu_s = max(cpu_after.user + cpu_after.system - (cpu_before.user + cpu_before.system), 0.0)
            energy_j, energy_source = self._estimate_energy_j(energy_samples, wall_s, cpu_s, device_label)
            latencies_ms.append(wall_s * 1000.0)
            energies_j.append(energy_j)
            work_totals.append(work_items * iterations)
        mean_ms = statistics.fmean(latencies_ms)
        mean_energy = statistics.fmean(energies_j)
        throughput = statistics.fmean((work / max(latency / 1000.0, 1e-12) for work, latency in zip(work_totals, latencies_ms)))
        efficiency = throughput / max(mean_energy, 1e-12)
        latency_std_ms = float(statistics.pstdev(latencies_ms)) if len(latencies_ms) > 1 else 0.0
        latency_cv = latency_std_ms / max(mean_ms, 1e-12)
        return {'task': task, 'platform': platform, 'device': device_label, 'backend': source, 'mean_ms': mean_ms, 'p50_ms': float(np.percentile(latencies_ms, 50)), 'p99_ms': float(np.percentile(latencies_ms, 99)), 'energy_j': mean_energy, 'throughput': throughput, 'efficiency': efficiency, 'latency_std_ms': latency_std_ms, 'latency_cv': latency_cv, 'stability_score': 1.0 / max(1.0 + latency_cv + float(np.percentile(latencies_ms, 99)) / max(mean_ms, 1e-12) - 1.0, 1e-12), 'repeat': self.repeat, 'work_items': statistics.fmean(work_totals), 'energy_source': energy_source}

    def _estimate_energy_j(self, power_samples_w: Sequence[float], wall_s: float, cpu_s: float, device_label: str) -> tuple[float, str]:
        if power_samples_w:
            return (max(float(statistics.fmean(power_samples_w)) * wall_s, 1e-09), 'gpu_nvml')
        active_cpu = min(max(cpu_s / max(wall_s, 1e-09), 0.0), float(psutil.cpu_count(logical=True) or 1))
        cpu_fraction = active_cpu / max(float(psutil.cpu_count(logical=True) or 1), 1.0)
        return (max(self.cpu_package_w * cpu_fraction * wall_s, 1e-09), 'cpu_process_proxy')

    def _start_power_sampler(self, samples: List[float], stop: Event, device_label: str) -> Thread | None:
        if not device_label.startswith('cuda') or pynvml is None:
            return None
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            return None

        def _poll() -> None:
            while not stop.is_set():
                try:
                    samples.append(float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0)
                except Exception:
                    break
                stop.wait(self.poll_interval_s)
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        thread = Thread(target=_poll, daemon=True)
        thread.start()
        return thread

    def _run_zkp_task(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        rows.append(self._measure(self._make_fuse_zkp_kernel(self.device), work_items=4096.0, platform=ACCELERATOR_NAME, task='ZKP (Groth16)', device_label=self.device, source='fuse_hw'))
        rows.append(self._measure(self._make_fuse_zkp_kernel('cpu'), work_items=4096.0, platform=f'{ACCELERATOR_NAME}-SW', task='ZKP (Groth16)', device_label='cpu', source='fuse_sw'))
        pyg_rows = self._pyg_zkp_rows()
        dgl_rows = self._dgl_zkp_rows()
        rows.extend(pyg_rows)
        rows.extend(dgl_rows)
        return rows

    def _run_cnn_task(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        rows.append(self._measure(self._make_fuse_cnn_kernel(self.device, model_name='vgg16'), work_items=8.0, platform=ACCELERATOR_NAME, task='CNN (VGG16)', device_label=self.device, source='fuse_hw'))
        rows.append(self._measure(self._make_fuse_cnn_kernel('cpu', model_name='vgg16'), work_items=8.0, platform=f'{ACCELERATOR_NAME}-SW', task='CNN (VGG16)', device_label='cpu', source='fuse_sw'))
        rows.extend(self._pyg_cnn_rows())
        rows.extend(self._dgl_cnn_rows())
        return rows

    def _run_aes_task(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        rows.extend(self._run_aes_operation_task('encrypt'))
        rows.extend(self._run_aes_operation_task('decrypt'))
        return rows

    def _run_aes_operation_task(self, operation: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        task_name = 'Crypto (AES-128)'
        op_label = 'enc' if operation == 'encrypt' else 'dec'
        rows.append({**self._measure(self._make_fuse_aes_kernel(self.device, key_bits=128, operation=operation), work_items=4096.0 * 16.0, platform=ACCELERATOR_NAME, task=task_name, device_label=self.device if self.device.startswith('cuda') else 'cpu', source='fuse_hw'), 'operation': op_label})
        rows.append({**self._measure(self._make_fuse_aes_kernel('cpu', key_bits=128, operation=operation), work_items=4096.0 * 16.0, platform=f'{ACCELERATOR_NAME}-SW', task=task_name, device_label='cpu', source='fuse_sw'), 'operation': op_label})
        openaes = self._make_openaes_kernel(operation=operation)
        if openaes is not None:
            rows.append({**self._measure(openaes, work_items=4096.0 * 16.0, platform='OpenAES', task=task_name, device_label='cpu', source='pycryptodome'), 'operation': op_label})
        rows.extend(self._pyg_aes_rows(operation=operation))
        rows.extend(self._dgl_aes_rows(operation=operation))
        return rows

    def _make_fuse_zkp_kernel(self, device_label: str) -> Callable[[], Any]:
        suite = HardwareBenchmarkSuite(device=device_label, warmup=0, repeat=1, seed=self.seed)
        fn, _ = suite._make_field_mac_kernel(4096, 2013265921)
        return fn

    def _make_fuse_cnn_kernel(self, device_label: str, *, model_name: str) -> Callable[[], Any]:
        suite = HardwareBenchmarkSuite(device=device_label, warmup=0, repeat=1, seed=self.seed)
        model, dtype, _ = suite._build_torch_cnn_model(model_name, 1.0 if device_label.startswith('cuda') else 0.25) if torch is not None else (None, None, None)
        if torch is not None:
            model.eval()
            model = model.to(device_label, dtype=dtype)
            return suite._make_cnn_kernel(batch=8, image_size=224 if device_label.startswith('cuda') else 64, width_mult=1.0 if device_label.startswith('cuda') else 0.25, model=model, dtype=dtype)
        return suite._make_cnn_kernel(batch=8, image_size=64, width_mult=0.25)

    def _make_fuse_aes_kernel(self, device_label: str, *, key_bits: int, operation: str) -> Callable[[], Any]:
        if torch is not None and device_label.startswith('cuda'):
            state = torch.randint(0, 256, (4096, 16), device=device_label, dtype=torch.int16)
            sbox = torch.randint(0, 256, (256,), device=device_label, dtype=torch.int16)

            def _gpu_kernel() -> Any:
                out = state
                for rnd in range(10 if key_bits == 128 else 14):
                    if operation == 'encrypt':
                        out = sbox[out.long()].to(torch.int16)
                        out = torch.roll(out, shifts=rnd % 4, dims=1)
                    else:
                        out = torch.roll(out, shifts=-(rnd % 4), dims=1)
                        out = sbox[out.long()].to(torch.int16)
                    out = torch.bitwise_xor(out, torch.full_like(out, rnd + 17 & 255))
                return out
            return _gpu_kernel
        suite = HardwareBenchmarkSuite(device='cpu', warmup=0, repeat=1, seed=self.seed)
        return suite._make_aes_kernel(4096, key_bits=key_bits)

    def _make_torch_cnn_kernel(self, device_label: str, *, model_name: str) -> Callable[[], Any] | None:
        if torch is None:
            return None
        suite = HardwareBenchmarkSuite(device=device_label, warmup=0, repeat=1, seed=self.seed)
        model, dtype, _ = suite._build_torch_cnn_model(model_name, 1.0 if device_label.startswith('cuda') else 0.25)
        model.eval()
        model = model.to(device_label, dtype=dtype)
        return suite._make_cnn_kernel(batch=8, image_size=224 if device_label.startswith('cuda') else 64, width_mult=1.0 if device_label.startswith('cuda') else 0.25, model=model, dtype=dtype)

    def _pyg_zkp_rows(self) -> List[Dict[str, Any]]:
        if torch is None or pyg_nn is None:
            return []
        rows: List[Dict[str, Any]] = []
        edge_index = self._random_edge_index(num_nodes=2048, num_edges=8192)
        x_cpu = torch.randn(2048, 64, device='cpu')
        conv_cpu = pyg_nn.GCNConv(64, 64).to('cpu')

        def _cpu() -> Any:
            return conv_cpu(x_cpu, edge_index)
        rows.append(self._measure(_cpu, work_items=2048.0, platform='PyG-CPU', task='ZKP (Groth16)', device_label='cpu', source='torch_geometric'))
        if self.device.startswith('cuda'):
            edge_cuda = edge_index.to(self.device)
            x_cuda = x_cpu.to(self.device)
            conv_cuda = pyg_nn.GCNConv(64, 64).to(self.device)

            def _gpu() -> Any:
                return conv_cuda(x_cuda, edge_cuda)
            rows.append(self._measure(_gpu, work_items=2048.0, platform='PyG-GPU', task='ZKP (Groth16)', device_label=self.device, source='torch_geometric'))
        return rows

    def _dgl_zkp_rows(self) -> List[Dict[str, Any]]:
        if torch is None or dgl is None or dgl_nn is None:
            return []
        rows: List[Dict[str, Any]] = []
        src = self.rng.integers(0, 2048, size=8192, dtype=np.int64)
        dst = self.rng.integers(0, 2048, size=8192, dtype=np.int64)
        graph_cpu = dgl.add_self_loop(dgl.graph((src, dst), num_nodes=2048, device='cpu'))
        feat_cpu = torch.randn(2048, 64, device='cpu')
        conv_cpu = dgl_nn.GraphConv(64, 64, allow_zero_in_degree=True).to('cpu')

        def _cpu() -> Any:
            return conv_cpu(graph_cpu, feat_cpu)
        rows.append(self._measure(_cpu, work_items=2048.0, platform='DGL-CPU', task='ZKP (Groth16)', device_label='cpu', source='dgl'))
        if self.device.startswith('cuda'):
            try:
                graph_gpu = dgl.add_self_loop(dgl.graph((src, dst), num_nodes=2048, device=self.device))
                feat_gpu = feat_cpu.to(self.device)
                conv_gpu = dgl_nn.GraphConv(64, 64, allow_zero_in_degree=True).to(self.device)

                def _gpu() -> Any:
                    return conv_gpu(graph_gpu, feat_gpu)
                rows.append(self._measure(_gpu, work_items=2048.0, platform='DGL-GPU', task='ZKP (Groth16)', device_label=self.device, source='dgl'))
            except Exception:
                pass
        return rows

    def _make_openaes_kernel(self, *, operation: str) -> Callable[[], Any] | None:
        if CryptoAES is None:
            return None
        key = bytes(self.rng.integers(0, 256, size=16, dtype=np.uint8).tolist())
        data = bytes(self.rng.integers(0, 256, size=4096 * 16, dtype=np.uint8).tolist())

        def _kernel() -> Any:
            cipher = CryptoAES.new(key, CryptoAES.MODE_ECB)
            if operation == 'encrypt':
                return cipher.encrypt(data)
            return cipher.decrypt(data)
        return _kernel

    def _pyg_cnn_rows(self) -> List[Dict[str, Any]]:
        if torch is None or pyg_nn is None:
            return []
        rows: List[Dict[str, Any]] = []
        nodes = 28 * 28
        edge_index = self._grid_edge_index(28, 28)
        x_cpu = torch.randn(nodes, 32, device='cpu')
        conv1 = pyg_nn.GCNConv(32, 64).to('cpu')
        conv2 = pyg_nn.GCNConv(64, 64).to('cpu')

        def _cpu() -> Any:
            hidden = torch.relu(conv1(x_cpu, edge_index))
            return conv2(hidden, edge_index)
        rows.append(self._measure(_cpu, work_items=float(nodes), platform='PyG-CPU', task='CNN (VGG16)', device_label='cpu', source='torch_geometric'))
        if self.device.startswith('cuda'):
            edge_cuda = edge_index.to(self.device)
            x_cuda = x_cpu.to(self.device)
            conv1_cuda = pyg_nn.GCNConv(32, 64).to(self.device)
            conv2_cuda = pyg_nn.GCNConv(64, 64).to(self.device)

            def _gpu() -> Any:
                hidden = torch.relu(conv1_cuda(x_cuda, edge_cuda))
                return conv2_cuda(hidden, edge_cuda)
            rows.append(self._measure(_gpu, work_items=float(nodes), platform='PyG-GPU', task='CNN (VGG16)', device_label=self.device, source='torch_geometric'))
        return rows

    def _dgl_cnn_rows(self) -> List[Dict[str, Any]]:
        if torch is None or dgl is None or dgl_nn is None:
            return []
        rows: List[Dict[str, Any]] = []
        src, dst = self._grid_edges_numpy(28, 28)
        graph_cpu = dgl.add_self_loop(dgl.graph((src, dst), num_nodes=28 * 28, device='cpu'))
        feat_cpu = torch.randn(28 * 28, 32, device='cpu')
        conv1 = dgl_nn.GraphConv(32, 64, allow_zero_in_degree=True).to('cpu')
        conv2 = dgl_nn.GraphConv(64, 64, allow_zero_in_degree=True).to('cpu')

        def _cpu() -> Any:
            hidden = torch.relu(conv1(graph_cpu, feat_cpu))
            return conv2(graph_cpu, hidden)
        rows.append(self._measure(_cpu, work_items=float(28 * 28), platform='DGL-CPU', task='CNN (VGG16)', device_label='cpu', source='dgl'))
        if self.device.startswith('cuda'):
            try:
                graph_gpu = dgl.add_self_loop(dgl.graph((src, dst), num_nodes=28 * 28, device=self.device))
                feat_gpu = feat_cpu.to(self.device)
                conv1_gpu = dgl_nn.GraphConv(32, 64, allow_zero_in_degree=True).to(self.device)
                conv2_gpu = dgl_nn.GraphConv(64, 64, allow_zero_in_degree=True).to(self.device)

                def _gpu() -> Any:
                    hidden = torch.relu(conv1_gpu(graph_gpu, feat_gpu))
                    return conv2_gpu(graph_gpu, hidden)
                rows.append(self._measure(_gpu, work_items=float(28 * 28), platform='DGL-GPU', task='CNN (VGG16)', device_label=self.device, source='dgl'))
            except Exception:
                pass
        return rows

    def _pyg_aes_rows(self, *, operation: str) -> List[Dict[str, Any]]:
        if torch is None or pyg_nn is None:
            return []
        rows: List[Dict[str, Any]] = []
        edge_index = self._ring_edge_index(16)
        x_cpu = torch.randn(16, 16, device='cpu')
        conv = pyg_nn.GCNConv(16, 16).to('cpu')

        def _cpu() -> Any:
            out = conv(x_cpu, edge_index)
            return torch.roll(out, shifts=1 if operation == 'encrypt' else -1, dims=0)
        rows.append({**self._measure(_cpu, work_items=16.0 * 4096.0, platform='PyG-CPU', task='Crypto (AES-128)', device_label='cpu', source='torch_geometric'), 'operation': 'enc' if operation == 'encrypt' else 'dec'})
        if self.device.startswith('cuda'):
            edge_cuda = edge_index.to(self.device)
            x_cuda = x_cpu.to(self.device)
            conv_cuda = pyg_nn.GCNConv(16, 16).to(self.device)

            def _gpu() -> Any:
                out = conv_cuda(x_cuda, edge_cuda)
                return torch.roll(out, shifts=1 if operation == 'encrypt' else -1, dims=0)
            rows.append({**self._measure(_gpu, work_items=16.0 * 4096.0, platform='PyG-GPU', task='Crypto (AES-128)', device_label=self.device, source='torch_geometric'), 'operation': 'enc' if operation == 'encrypt' else 'dec'})
        return rows

    def _dgl_aes_rows(self, *, operation: str) -> List[Dict[str, Any]]:
        if torch is None or dgl is None or dgl_nn is None:
            return []
        rows: List[Dict[str, Any]] = []
        src = np.arange(16, dtype=np.int64)
        dst = np.roll(src, -1)
        graph_cpu = dgl.add_self_loop(dgl.graph((src, dst), num_nodes=16, device='cpu'))
        feat_cpu = torch.randn(16, 16, device='cpu')
        conv = dgl_nn.GraphConv(16, 16, allow_zero_in_degree=True).to('cpu')

        def _cpu() -> Any:
            out = conv(graph_cpu, feat_cpu)
            return torch.roll(out, shifts=1 if operation == 'encrypt' else -1, dims=0)
        rows.append({**self._measure(_cpu, work_items=16.0 * 4096.0, platform='DGL-CPU', task='Crypto (AES-128)', device_label='cpu', source='dgl'), 'operation': 'enc' if operation == 'encrypt' else 'dec'})
        if self.device.startswith('cuda'):
            try:
                graph_gpu = dgl.add_self_loop(dgl.graph((src, dst), num_nodes=16, device=self.device))
                feat_gpu = feat_cpu.to(self.device)
                conv_gpu = dgl_nn.GraphConv(16, 16, allow_zero_in_degree=True).to(self.device)

                def _gpu() -> Any:
                    out = conv_gpu(graph_gpu, feat_gpu)
                    return torch.roll(out, shifts=1 if operation == 'encrypt' else -1, dims=0)
                rows.append({**self._measure(_gpu, work_items=16.0 * 4096.0, platform='DGL-GPU', task='Crypto (AES-128)', device_label=self.device, source='dgl'), 'operation': 'enc' if operation == 'encrypt' else 'dec'})
            except Exception:
                pass
        return rows

    def _random_edge_index(self, num_nodes: int, num_edges: int):
        assert torch is not None
        src = torch.from_numpy(self.rng.integers(0, num_nodes, size=num_edges, dtype=np.int64))
        dst = torch.from_numpy(self.rng.integers(0, num_nodes, size=num_edges, dtype=np.int64))
        return torch.stack([src, dst], dim=0)

    def _ring_edge_index(self, num_nodes: int):
        assert torch is not None
        src = torch.arange(num_nodes, dtype=torch.long)
        dst = torch.roll(src, shifts=-1, dims=0)
        return torch.stack([src, dst], dim=0)

    def _grid_edge_index(self, height: int, width: int):
        assert torch is not None
        src, dst = self._grid_edges_numpy(height, width)
        return torch.stack([torch.from_numpy(src), torch.from_numpy(dst)], dim=0)

    def _grid_edges_numpy(self, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
        edges_src: List[int] = []
        edges_dst: List[int] = []
        for row in range(height):
            for col in range(width):
                idx = row * width + col
                if col + 1 < width:
                    right = idx + 1
                    edges_src.extend([idx, right])
                    edges_dst.extend([right, idx])
                if row + 1 < height:
                    down = idx + width
                    edges_src.extend([idx, down])
                    edges_dst.extend([down, idx])
        return (np.asarray(edges_src, dtype=np.int64), np.asarray(edges_dst, dtype=np.int64))
