from __future__ import annotations
ACCELERATOR_NAME = 'FUSE'
PACKAGE_LEGACY_NAME = 'morphaccel'
CPP_BINARY_STEM = 'fuse_cpp'
MODE_ALIASES = {'fuse': 'fuse', 'morphaccel': 'fuse', 'secure_ct': 'secure_ct', 'secure_elastic': 'secure_elastic', 'secure_elastic_no_bucket': 'secure_elastic_no_bucket', 'secure_elastic_no_pinning': 'secure_elastic_no_pinning', 'secure_elastic_no_mask': 'secure_elastic_no_mask', 'secure_elastic_no_epoch': 'secure_elastic_no_epoch', 'secure_elastic_no_prefetch': 'secure_elastic_no_prefetch', 'secure_rand': 'secure_rand', 'multicore': 'multicore', 'gpu_tensor': 'gpu_tensor', 'static_zkp': 'static_zkp', 'eyeriss': 'eyeriss', 'openaes': 'openaes', 'degraded': 'degraded', 'task_specific': 'task_specific'}
DISPLAY_MODE_NAMES = {'fuse': ACCELERATOR_NAME, 'secure_ct': 'Secure-CT', 'secure_elastic': 'Secure-Elastic', 'secure_elastic_no_bucket': '- Bucketization', 'secure_elastic_no_pinning': '- Family pinning', 'secure_elastic_no_mask': '- RFU masking', 'secure_elastic_no_epoch': '- Epoch rotation', 'secure_elastic_no_prefetch': '- Prefetch', 'secure_rand': 'Secure-Rand', 'multicore': 'Multi-core', 'gpu_tensor': 'GPU-Tensor', 'static_zkp': 'Static-ZKP', 'eyeriss': 'Eyeriss', 'openaes': 'OpenAES', 'degraded': 'Degraded', 'task_specific': ACCELERATOR_NAME}

def normalize_mode(mode: str) -> str:
    return MODE_ALIASES.get(mode, mode)

def display_mode_name(mode: str) -> str:
    return DISPLAY_MODE_NAMES.get(normalize_mode(mode), mode)
