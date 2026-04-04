from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from fuse.training import AlpacaTrainingSuite

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('runs/train_alpaca'))
    parser.add_argument('--models', type=str, default='distilgpt2,bert-base-uncased')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda:0'], default='auto')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-05)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--train-limit', type=int, default=None)
    parser.add_argument('--val-limit', type=int, default=None)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeat', type=int, default=5)
    args = parser.parse_args()
    suite = AlpacaTrainingSuite(device=args.device, seed=args.seed, warmup=args.warmup, repeat=args.repeat)
    models = [model.strip() for model in args.models.split(',') if model.strip()]
    suite.run(data_root=args.data_root, output_root=args.output_root, models=models, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, num_workers=args.num_workers, train_limit=args.train_limit, val_limit=args.val_limit)
    print(args.output_root)
if __name__ == '__main__':
    main()
