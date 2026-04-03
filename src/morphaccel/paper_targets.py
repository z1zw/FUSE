from .branding import ACCELERATOR_NAME
FIG4_TARGETS = {'Groth16': {'morph_last_ms': 245.0, 'baseline_ratio': 1.03}, 'VGG16': {'morph_last_ms': 485.0, 'baseline_ratio': 1.0 / 1.08}, 'AES-128': {'morph_last_ms': 35.6, 'baseline_ratio': 1.0 / 1.02}}
FIG7_TARGETS = {'area': {'FVM + intermediate regs': 41.2, 'RFU': 29.6, 'TARS': 18.4, 'Others': 10.8}, 'power': {'RFU dynamic': 46.1, 'Scheduler+FVM dynamic': 38.5, 'Static': 15.4}, 'power_total_w': 8.27, 'area_total_mm2': 11.03}
FIG8_TARGETS = {'reuse': {'Exact-hash reuse': 52.0, 'Layout-adapted reuse': 15.4, 'Unique to ZKP': 32.6}, 'fanout': {'1 task': 47.9, '2 tasks': 33.6, '3 tasks': 18.5}, 'pool': {'MatMul+GELU': 34.0, 'FieldMul+Poseidon': 31.0, 'MixColumns': 17.0, 'Other reused paths': 18.0}}
FIG9_TARGETS = {'groth16_reduction_pct': 14.2, 'vgg16_throughput_gain': 1.26, 'vgg16_latency_overhead_pct': 8.9}
FIG10_TARGETS = {'residency_reduction_pct': 36.8, 'redundant_access_reduction_pct': 22.3}
FIG11_TARGETS = {'prefetch_earlier_ms': 2.1, 'pipeline_stall_reduction_pct': 28.7, 'switch_latency_ms': {'Multi-core': 19.6, ACCELERATOR_NAME: 4.8}, 'idle_ratio_pct': {'Multi-core': 61.4, ACCELERATOR_NAME: 16.7}, 'reduction_summary_pct': {'Latency Reduction': 75.5, 'Idle Ratio Reduction': 72.4, 'Buffer Ops Reduction': 18.5}}
FIG12_TARGETS = {ACCELERATOR_NAME: 2.3, 'GPU': 4.6, 'FPGA': 31.7, 'energy_vs_fpga_pct': 87.2, 'energy_vs_gpu_pct': 36.5, 'recovery_ms': 0.3}
