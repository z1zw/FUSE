from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from morphaccel.training import CIFARTrainingSuite

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('runs/train_cifar'))
    parser.add_argument('--models', type=str, default='vgg16,resnet101,vit_b_16')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda:0'], default='auto')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--train-limit', type=int, default=None)
    parser.add_argument('--val-limit', type=int, default=None)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--no-pretrained', action='store_true')
    args = parser.parse_args()
    suite = CIFARTrainingSuite(device=args.device, seed=args.seed, warmup=args.warmup, repeat=args.repeat, pretrained=not args.no_pretrained)
    models = [model.strip() for model in args.models.split(',') if model.strip()]
    suite.run(data_root=args.data_root, output_root=args.output_root, models=models, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay, num_workers=args.num_workers, val_ratio=args.val_ratio, train_limit=args.train_limit, val_limit=args.val_limit)
    print(args.output_root)
if __name__ == '__main__':
    main()
