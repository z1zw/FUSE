from __future__ import annotations
from pathlib import Path
from morphaccel.cpp_bridge import CppSimulatorBridge

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    bridge = CppSimulatorBridge(root)
    try:
        print(bridge.build())
    except Exception as exc:
        print(f'cpp-build-unavailable: {exc}')
        print('falling back to Python simulator at runtime')
if __name__ == '__main__':
    main()
