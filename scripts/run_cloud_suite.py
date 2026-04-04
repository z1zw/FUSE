from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from fuse.cloud_suite import CloudExperimentSuite

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('data/cloud_runs'))
    parser.add_argument('--output-root', type=Path, default=Path('runs/cloud_suite'))
    parser.add_argument('--coco-root', type=Path, default=None)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda:0'], default='auto')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--sample-count', type=int, default=6)
    parser.add_argument('--no-pretrained', action='store_true')
    args = parser.parse_args()
    suite = CloudExperimentSuite(device=args.device, warmup=args.warmup, repeat=args.repeat, seed=args.seed, sample_count=args.sample_count, pretrained=not args.no_pretrained)
    dataset_roots = {}
    if args.coco_root is not None:
        dataset_roots['coco2017'] = args.coco_root
    suite.run(args.data_root, args.output_root, dataset_roots=dataset_roots)
    print(args.output_root)
if __name__ == '__main__':
    main()
