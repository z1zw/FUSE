from __future__ import annotations
import argparse
from pathlib import Path
from morphaccel.report import write_cn_report

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=Path('results_plus'))
    parser.add_argument('--output', type=Path, default=Path('docs/实验报告.md'))
    args = parser.parse_args()
    write_cn_report(args.results_dir, args.output)
if __name__ == '__main__':
    main()
