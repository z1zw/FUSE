from __future__ import annotations

import importlib
import pkgutil
import sys

import fuse as _fuse

for name in getattr(_fuse, '__all__', []):
    globals()[name] = getattr(_fuse, name)

for module_info in pkgutil.walk_packages(_fuse.__path__, prefix='fuse.'):
    fuse_name = module_info.name
    public_name = fuse_name.replace('fuse', 'FUSE', 1)
    sys.modules.setdefault(public_name, importlib.import_module(fuse_name))

sys.modules.setdefault('FUSE', _fuse)

__all__ = getattr(_fuse, '__all__', [])
