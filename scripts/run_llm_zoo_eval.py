"""Run the language-model evaluation: real local models, then FUSE vs baselines.

Example:
    python scripts/run_llm_zoo_eval.py --output-root runs/llm_zoo --sample-count 8
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fuse.llm_zoo import DEFAULT_GEN_TOKENS, DEFAULT_TILING_WIDTH, LLM_ZOO, LLMZooExperimentSuite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path('runs/llm_zoo'))
    parser.add_argument('--dataset', choices=['gsm8k', 'mmlu'], default='gsm8k')
    parser.add_argument('--sample-count', type=int, default=8)
    parser.add_argument('--gen-tokens', type=int, default=DEFAULT_GEN_TOKENS)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument(
        '--models',
        nargs='*',
        default=None,
        help=f'subset of {[m["name"] for m in LLM_ZOO]}; defaults to all',
    )
    parser.add_argument(
        '--no-native',
        action='store_true',
        help='skip running the models on the GPU and only tokenize the prompts',
    )
    parser.add_argument('--zkp-circuit-size', type=int, default=65536)
    parser.add_argument('--width', type=int, default=DEFAULT_TILING_WIDTH,
                        help='matmul output tiles per stage; higher fills the RFU array '
                             'but makes the security sweep much slower')
    parser.add_argument('--reuse-native', action='store_true',
                        help='load GPU measurements from a previous run in the same output root')
    parser.add_argument(
        '--offline',
        action='store_true',
        help='resolve models and datasets from the local Hugging Face cache only',
    )
    args = parser.parse_args()

    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

    suite = LLMZooExperimentSuite(
        device=args.device,
        dataset=args.dataset,
        sample_count=args.sample_count,
        gen_tokens=args.gen_tokens,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
        models=args.models,
        measure_native=not args.no_native,
        zkp_circuit_size=args.zkp_circuit_size,
        width=args.width,
        reuse_native=args.reuse_native,
    )
    artifacts = suite.run(args.output_root)
    print(f'\nwrote {args.output_root}')
    if not artifacts.summary.empty:
        columns = [c for c in ['model_label', 'fuse_total_ms', 'transformer_asic_latency_ratio',
                               'multicore_energy_gain', 'serving_latency_reduction'] if c in artifacts.summary.columns]
        print(artifacts.summary[columns].to_string(index=False))


if __name__ == '__main__':
    main()
