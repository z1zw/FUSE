from .branding import ACCELERATOR_NAME

FIG4_TARGETS = {
    'Groth16': {'latency_overhead': 1.28},
    'VGG16': {'latency_overhead': 1.05},
    'AES-128': {'latency_overhead': 1.03},
}

FIG7_TARGETS = {
    'area': {
        'FVM + intermediate regs': 54.9,
        'RFU': 36.3,
        'TARS': 3.5,
        'Others': 5.3,
    },
    'power': {
        'RFU dynamic': 10.1,
        'Scheduler+FVM dynamic': 28.0,
        'Static': 61.9,
    },
    'power_total_w': 2.98,
    'area_total_mm2': 8.38,
}

FIG8_TARGETS = {
    'reuse': {'Exact-hash': 67.4, 'Layout adaptation': 27.5, 'Unique': 5.1},
    'fanout': {'1': 47.9, '2': 33.6, '3+': 18.5},
    'pool': {'MatMul+GELU': 34.0, 'FieldMul+Poseidon': 31.0, 'MixColumns': 17.0, 'Other reused paths': 18.0},
}

FIG9_TARGETS = {
    'groth16_latency_reduction_pct': 91.3,
    'groth16_throughput_gain': 11.5,
    'vgg16_throughput_gain': 1.0004,
    'vgg16_latency_overhead_pct': 0.04,
}

FIG10_TARGETS = {
    'residency_reduction_pct': 36.8,
    'redundant_access_reduction_pct': 22.3,
}

FIG11_TARGETS = {
    'switch_latency_ms': {'Multi-core': 1.039, ACCELERATOR_NAME: 0.334},
    'idle_ratio_pct': {'Multi-core': 12.9, ACCELERATOR_NAME: 12.7},
    'buffer_ops_reduction_pct': 18.5,
}

FIG12_TARGETS = {
    'switch_latency_ms': {'CPU': 2.31, 'GPU': 1.56, ACCELERATOR_NAME: 0.031},
    'energy_efficiency_gain': {'CPU': 31.7, 'GPU': 12.7},
    'recovery_ms': 0.3,
}
