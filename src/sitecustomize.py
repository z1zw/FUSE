from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import pkgutil
import sys


class _MorphaccelAliasLoader(importlib.abc.Loader):
    def __init__(self, fullname: str, target: str):
        self.fullname = fullname
        self.target = target

    def create_module(self, spec):
        module = importlib.import_module(self.target)
        sys.modules[self.fullname] = module
        return module

    def exec_module(self, module):
        return None


class _MorphaccelAliasFinder(importlib.abc.MetaPathFinder):
    PREFIX = 'morphaccel'
    TARGET_PREFIX = 'fuse'

    def find_spec(self, fullname: str, path, target=None):
        if fullname == self.PREFIX:
            target_name = self.TARGET_PREFIX
        elif fullname.startswith(self.PREFIX + '.'):
            target_name = self.TARGET_PREFIX + fullname[len(self.PREFIX):]
        else:
            return None
        target_spec = importlib.util.find_spec(target_name)
        if target_spec is None:
            return None
        is_package = bool(target_spec.submodule_search_locations)
        spec = importlib.util.spec_from_loader(fullname, _MorphaccelAliasLoader(fullname, target_name), is_package=is_package)
        if spec is not None and is_package:
            spec.submodule_search_locations = list(target_spec.submodule_search_locations or [])
        return spec


for module_info in pkgutil.walk_packages(importlib.import_module('fuse').__path__, prefix='fuse.'):
    target_name = module_info.name
    legacy_name = target_name.replace('fuse', 'morphaccel', 1)
    sys.modules.setdefault(legacy_name, importlib.import_module(target_name))

sys.modules.setdefault('morphaccel', importlib.import_module('fuse'))

if not any(isinstance(finder, _MorphaccelAliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _MorphaccelAliasFinder())
