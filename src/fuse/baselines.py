from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaselineFamily:
    name: str
    mode: str
    domain: str
    description: str


BASELINES = {
    'Groth16': BaselineFamily('Static ZKP Core', 'static_zkp', 'ZKP', 'Static pipeline without fragment sharing'),
    'VGG16': BaselineFamily('Eyeriss-style Accelerator', 'eyeriss', 'CNN', 'CNN-specific dataflow accelerator'),
    'AES-128': BaselineFamily('AES Dedicated Engine', 'openaes', 'AES', 'OpenAES-like round engine'),
    'Mixed': BaselineFamily('Multi-core', 'multicore', 'Mixed', 'Independent domain-specific engines without cross-task reuse'),
}
