from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from fuse.benchmarks import run_default_suite
from fuse.hardware import HardwareBenchmarkSuite

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    parser.add_argument('--engine', choices=['cpp', 'py', 'gpu'], default='cpp')
    parser.add_argument('--calibration', type=Path, default=None)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda:0'], default='auto')
    parser.add_argument('--cnn-models', choices=['vgg16', 'resnet50', 'resnet101'], nargs='+', default=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--cnn-batches', type=int, nargs='+', default=[1, 4, 8])
    parser.add_argument('--cnn-image-size', type=int, default=None)
    parser.add_argument('--cnn-width-mult', type=float, default=None)
    parser.add_argument('--aes-key-bits', type=int, nargs='+', choices=[128, 256], default=[128, 256])
    parser.add_argument('--aes-blocks', type=int, nargs='+', default=[1024, 4096, 16384])
    parser.add_argument('--zkp-sizes', type=int, nargs='+', default=[4096, 16384, 65536])
    parser.add_argument('--mixed-rounds', type=int, default=3)
    args = parser.parse_args()
    if args.engine == 'gpu':
        HardwareBenchmarkSuite(device=args.device).run(args.output_dir, cnn_models=args.cnn_models, cnn_batches=args.cnn_batches, cnn_image_size=args.cnn_image_size, cnn_width_mult=args.cnn_width_mult, aes_key_bits=args.aes_key_bits, aes_blocks=args.aes_blocks, zkp_sizes=args.zkp_sizes, mixed_rounds=args.mixed_rounds)
    else:
        run_default_suite(args.output_dir, engine=args.engine, calibration_path=args.calibration)
    print(args.output_dir)
if __name__ == '__main__':
    main()
