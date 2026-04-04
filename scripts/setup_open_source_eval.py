from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from fuse.open_source_eval import OPEN_SOURCE_TOOLS, clone_or_update_repo, host_build_tools, write_json

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tool', choices=['all', *OPEN_SOURCE_TOOLS.keys()], default='all')
    parser.add_argument('--update', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports' / 'open_source_tools_manifest.json')
    args = parser.parse_args()
    selected = list(OPEN_SOURCE_TOOLS.keys()) if args.tool == 'all' else [args.tool]
    repos: dict[str, str] = {}
    for tool in selected:
        repo_dir = clone_or_update_repo(tool, root=ROOT, update=args.update)
        repos[tool] = str(repo_dir)
    payload = {'tools': repos, 'host_build_tools': host_build_tools()}
    write_json(args.output, payload)
    print(args.output)
if __name__ == '__main__':
    main()
