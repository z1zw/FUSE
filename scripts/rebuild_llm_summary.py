"""Re-derive the combined language-model summary and figures from per-model results.

Useful when a summary metric is added or a figure is restyled: the per-model
CSVs under ``<root>/<model>/`` are the record of the run, and everything above
them is derived, so nothing has to be measured or simulated again.

    python scripts/rebuild_llm_summary.py --root runs/llm_zoo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fuse.llm_zoo import LLMZooArtifacts, LLMZooExperimentSuite

PARTS = ('native.csv', 'simulated.csv', 'secure_serving.csv')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'runs' / 'llm_zoo')
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()

    root = args.root
    frames = {name: [] for name in PARTS}
    models = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        paths = [model_dir / name for name in PARTS]
        if not all(p.exists() for p in paths):
            print(f'  skipping {model_dir.name}: incomplete')
            continue
        for name, path in zip(PARTS, paths):
            frames[name].append(pd.read_csv(path))
        models.append(model_dir.name)
    if not models:
        raise SystemExit(f'no complete model directories under {root}')

    native = pd.concat(frames['native.csv'], ignore_index=True)
    simulated = pd.concat(frames['simulated.csv'], ignore_index=True)
    serving = pd.concat(frames['secure_serving.csv'], ignore_index=True)

    suite = LLMZooExperimentSuite(measure_native=False)
    summary = suite._summarize(native, simulated, serving)

    native.to_csv(root / 'llm_native.csv', index=False)
    simulated.to_csv(root / 'llm_simulated.csv', index=False)
    serving.to_csv(root / 'llm_secure_serving.csv', index=False)
    summary.to_csv(root / 'llm_summary.csv', index=False)
    if not args.no_figures:
        suite.render_figures(
            LLMZooArtifacts(native, simulated, serving, summary, {}), root
        )
    print(f'rebuilt {len(models)} models: {", ".join(models)}')
    print(f'wrote {root / "llm_summary.csv"}')


if __name__ == '__main__':
    main()
