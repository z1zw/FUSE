from __future__ import annotations
import argparse
from pathlib import Path
from fuse.dashboard import export_dashboard

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=Path('results_plus'))
    parser.add_argument('--output', type=Path, default=Path('web/dashboard.html'))
    args = parser.parse_args()
    export_dashboard(args.results_dir, args.output)
if __name__ == '__main__':
    main()
