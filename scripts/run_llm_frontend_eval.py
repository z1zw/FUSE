"""Compile service requests with real language models, then run both paths.

Hosted models are reached through an OpenAI-compatible gateway.  Credentials are
read from the environment and are never written into the repository:

    $env:FUSE_LLM_GATEWAY = "https://<your-gateway>"
    $env:FUSE_LLM_TOKEN   = "<your-token>"
    python scripts/run_llm_frontend_eval.py --output-root runs/llm_frontend

Pass --no-gateway to run the locally cached models only.
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

from fuse.llm_frontend_eval import (
    DEFAULT_GATEWAY_MODELS,
    DEFAULT_LOCAL_MODELS,
    LLMFrontendSuite,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path('runs/llm_frontend'))
    parser.add_argument('--gateway-models', nargs='*', default=list(DEFAULT_GATEWAY_MODELS))
    parser.add_argument(
        '--local-models',
        nargs='*',
        default=None,
        help='Hugging Face ids; defaults to the cached instruction-tuned set',
    )
    parser.add_argument('--no-gateway', action='store_true', help='skip hosted models')
    parser.add_argument('--no-local', action='store_true', help='skip locally cached models')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--retries', type=int, default=2)
    parser.add_argument('--offline', action='store_true', help='local models from cache only')
    args = parser.parse_args()

    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

    if args.local_models is None:
        local = list(DEFAULT_LOCAL_MODELS)
    else:
        local = [(hf_id.split('/')[-1], hf_id) for hf_id in args.local_models]
    if args.no_local:
        local = []

    suite = LLMFrontendSuite(
        gateway_models=[] if args.no_gateway else args.gateway_models,
        local_models=local,
        device=args.device,
        retries=args.retries,
    )
    artifacts = suite.run(args.output_root)
    print(f'\nwrote {args.output_root}')
    if not artifacts.summary.empty:
        columns = [
            c
            for c in [
                'model', 'kind', 'parse_rate', 'plan_fidelity',
                'unprotected_agreement', 'canonical_agreement',
                'latency_reduction', 'unprotected_attack_accuracy', 'fuse_attack_accuracy',
            ]
            if c in artifacts.summary.columns
        ]
        print(artifacts.summary[columns].to_string(index=False))


if __name__ == '__main__':
    main()
