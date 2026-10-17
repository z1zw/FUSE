"""Concurrent three-domain secure inference: one fabric against dedicated engines.

A privacy-preserving service runs AES and Groth16 whether it wants to or not.
This asks what those obligations cost, and whether one reconfigurable fabric
absorbs them better than three engines that cannot help each other.

Request shapes come from three public datasets and the volume of the security
work is derived from the same bytes, so nothing about the load is assumed. The
baseline is given its best partition of the RFU budget, so the margin reported
for FUSE is a lower bound.

    python scripts/run_concurrent_eval.py --output-root runs/concurrent --offline
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fuse.concurrent_eval import (
    DEFAULT_ANSWER_MODELS,
    SIMULATION_ONLY_MODELS,
    load_request_shapes,
    measure_answer_lengths,
    run_study,
)
from fuse.llm_zoo import LLM_ZOO
from fuse.workloads.llm import spec_from_hf_config

DEFAULT_MODELS = ['gpt2', 'qwen2_5_0_5b', 'tinyllama_1_1b', 'mistral_7b', 'qwen2_5_14b', 'qwen2_5_32b']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path('runs/concurrent'))
    parser.add_argument('--models', nargs='*', default=DEFAULT_MODELS,
                        help='models from the zoo, plus the simulation-only entries '
                             f'{[m["name"] for m in SIMULATION_ONLY_MODELS]}')
    parser.add_argument('--array-sizes', type=int, nargs='*', default=[4, 8, 12, 16, 24, 32])
    parser.add_argument('--security-levels', type=int, nargs='*', default=[1, 4, 16])
    parser.add_argument('--llm-requests', type=int, default=2)
    parser.add_argument('--sample-count', type=int, default=64)
    parser.add_argument('--retrieval-depth', type=int, default=4)
    parser.add_argument('--span-ms', type=float, default=20.0)
    parser.add_argument('--width', type=int, default=8)
    parser.add_argument('--reference-rfus', type=int, default=16,
                        help='the configuration the paper evaluates, marked on the figure')
    parser.add_argument('--tokenize', action='store_true',
                        help="use a real tokenizer for prompt lengths instead of a byte estimate")
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--answer-lengths', type=Path, default=None,
                        help='JSON of measured answer lengths; reused if present')
    parser.add_argument('--measure-answer-lengths', action='store_true',
                        help='ask hosted models how long an answer really is, '
                             'instead of assuming a generated length')
    parser.add_argument('--answer-models', nargs='*', default=list(DEFAULT_ANSWER_MODELS))
    parser.add_argument('--answer-samples', type=int, default=6)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()

    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

    from transformers import AutoConfig

    specs = []
    for name in args.models:
        known = list(LLM_ZOO) + SIMULATION_ONLY_MODELS
        meta = next((m for m in known if m['name'] == name), None)
        if meta is None:
            raise SystemExit(f'unknown model {name!r}; choose from {[m["name"] for m in known]}')
        specs.append(spec_from_hf_config(
            AutoConfig.from_pretrained(meta['hf_id']),
            name=meta['name'], label=meta['label'], hf_id=meta['hf_id'],
        ))

    answer_lengths = None
    cache = args.answer_lengths or (args.output_root / 'answer_lengths.json')
    if cache.exists():
        answer_lengths = json.loads(cache.read_text(encoding='utf-8'))
        print(f'reusing measured answer lengths from {cache}')
    elif args.measure_answer_lengths:
        print('measuring answer lengths on the hosted models:')
        answer_lengths = measure_answer_lengths(models=args.answer_models,
                                                samples=args.answer_samples,
                                                seed=args.seed)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(answer_lengths, indent=2), encoding='utf-8')
        print(f'wrote {cache}')

    shapes = load_request_shapes(
        sample_count=args.sample_count,
        retrieval_depth=args.retrieval_depth,
        seed=args.seed,
        tokenizer_id=specs[0].hf_id if args.tokenize else None,
        answer_lengths=answer_lengths,
    )

    artifacts = run_study(
        models=specs,
        shapes=shapes,
        output_root=args.output_root,
        array_sizes=args.array_sizes,
        security_levels=args.security_levels,
        llm_requests=args.llm_requests,
        span_ms=args.span_ms,
        width=args.width,
        reference_rfus=args.reference_rfus,
        seed=args.seed,
    )
    print(f'\nwrote {args.output_root}')
    ref = artifacts.results[artifacts.results['rfus'] == args.reference_rfus]
    if not ref.empty:
        cols = ['model_label', 'dataset_label', 'security_per_request', 'fuse_ms',
                'partitioned_ms', 'speedup_vs_partitioned', 'speedup_vs_dedicated', 'security_tax']
        print(f'\nat the evaluated {args.reference_rfus}-RFU configuration:')
        print(ref[cols].to_string(index=False))


if __name__ == '__main__':
    main()
