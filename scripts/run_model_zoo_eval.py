from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from fuse.model_zoo import DEFAULT_CIFAR_ROOT, DEFAULT_COCO_ROOT, ModelZooExperimentSuite

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, default=Path('runs/model_zoo'))
    parser.add_argument('--cifar-root', type=Path, default=DEFAULT_CIFAR_ROOT)
    parser.add_argument('--coco-root', type=Path, default=DEFAULT_COCO_ROOT)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda:0'], default='auto')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--sample-count', type=int, default=10)
    parser.add_argument('--no-pretrained', action='store_true')
    args = parser.parse_args()
    suite = ModelZooExperimentSuite(device=args.device, warmup=args.warmup, repeat=args.repeat, seed=args.seed, sample_count=args.sample_count, pretrained=not args.no_pretrained)
    suite.run_model_zoo(args.output_root, cifar_root=args.cifar_root, coco_root=args.coco_root)
    print(args.output_root)
if __name__ == '__main__':
    main()
