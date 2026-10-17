import sys
import shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def has_cpp_toolchain() -> bool:
    """True when the C++ engine can be run: a compiler is present, or a
    prebuilt engine that is at least as new as its sources already sits in
    cpp/build, which cpp_bridge.build() uses without compiling."""
    if shutil.which('g++') is not None or shutil.which('cl') is not None:
        return True
    try:
        from fuse.cpp_bridge import CppSimulatorBridge
        return CppSimulatorBridge(ROOT).build().exists()
    except Exception:
        return False
