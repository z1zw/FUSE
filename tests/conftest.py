import sys
import shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def has_cpp_toolchain() -> bool:
    return shutil.which('g++') is not None or shutil.which('cl') is not None
