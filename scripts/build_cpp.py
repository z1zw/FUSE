from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from fuse.cpp_bridge import CppSimulatorBridge

def main() -> None:
    bridge = CppSimulatorBridge(ROOT)
    try:
        print(bridge.build())
    except Exception as exc:
        print(f'cpp-build-unavailable: {exc}')
        print('falling back to Python simulator at runtime')
if __name__ == '__main__':
    main()
