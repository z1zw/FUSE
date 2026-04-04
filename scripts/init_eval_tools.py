from __future__ import annotations
import argparse
from pathlib import Path
from fuse.eval_tools import write_eval_tool_template

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('configs/eval_tools.json'))
    args = parser.parse_args()
    write_eval_tool_template(args.output)
    print(args.output)
if __name__ == '__main__':
    main()
