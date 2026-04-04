import pandas as pd
from fuse.branding import ACCELERATOR_NAME
from fuse.energy_benchmarks import EnergyBenchmarkSuite

def test_normalize_aggregates_encrypt_and_decrypt_rows() -> None:
    suite = EnergyBenchmarkSuite(device='cpu', warmup=0, repeat=1, seed=7)
    raw = pd.DataFrame([{'task': 'Crypto (AES-128)', 'platform': ACCELERATOR_NAME, 'device': 'cpu', 'backend': 'fuse_hw', 'mean_ms': 1.0, 'p50_ms': 1.0, 'p99_ms': 1.2, 'energy_j': 1.0, 'throughput': 10.0, 'efficiency': 10.0, 'latency_std_ms': 0.1, 'latency_cv': 0.1, 'stability_score': 0.9, 'repeat': 1, 'work_items': 1.0, 'energy_source': 'cpu_process_proxy', 'operation': 'enc'}, {'task': 'Crypto (AES-128)', 'platform': ACCELERATOR_NAME, 'device': 'cpu', 'backend': 'fuse_hw', 'mean_ms': 1.2, 'p50_ms': 1.2, 'p99_ms': 1.3, 'energy_j': 1.4, 'throughput': 10.0, 'efficiency': 8.0, 'latency_std_ms': 0.1, 'latency_cv': 0.08, 'stability_score': 0.88, 'repeat': 1, 'work_items': 1.0, 'energy_source': 'cpu_process_proxy', 'operation': 'dec'}, {'task': 'Crypto (AES-128)', 'platform': 'OpenAES', 'device': 'cpu', 'backend': 'pycryptodome', 'mean_ms': 2.0, 'p50_ms': 2.0, 'p99_ms': 2.2, 'energy_j': 3.0, 'throughput': 10.0, 'efficiency': 4.0, 'latency_std_ms': 0.2, 'latency_cv': 0.1, 'stability_score': 0.75, 'repeat': 1, 'work_items': 1.0, 'energy_source': 'cpu_process_proxy', 'operation': 'enc'}, {'task': 'Crypto (AES-128)', 'platform': 'OpenAES', 'device': 'cpu', 'backend': 'pycryptodome', 'mean_ms': 2.2, 'p50_ms': 2.2, 'p99_ms': 2.4, 'energy_j': 3.4, 'throughput': 10.0, 'efficiency': 3.0, 'latency_std_ms': 0.2, 'latency_cv': 0.09, 'stability_score': 0.73, 'repeat': 1, 'work_items': 1.0, 'energy_source': 'cpu_process_proxy', 'operation': 'dec'}])
    summary = suite._normalize(raw)
    assert len(summary) == 2
    fuse_row = summary[summary['platform'] == ACCELERATOR_NAME].iloc[0]
    openaes_row = summary[summary['platform'] == 'OpenAES'].iloc[0]
    assert fuse_row['operation'] == 'dec+enc'
    assert openaes_row['operation'] == 'dec+enc'
    assert fuse_row['efficiency_norm'] == 1.0
    assert openaes_row['efficiency_norm'] > 1.0
