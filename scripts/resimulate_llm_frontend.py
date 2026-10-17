"""Re-run the request-compiler study's simulation from the replies already on disk.

The study has two halves. Language models are asked to compile each service
request into a plan, and those replies are kept in ``frontend_replies.csv``.
The plans are then parsed and every service is replayed on the simulator under
the three arms. Only the first half needs the models, a gateway and a GPU; the
second is a function of the replies and the engine.

When the engine changes, this re-runs the second half alone, so the published
replies are kept and the simulation is brought up to date, instead of asking
the models again and getting different replies.

    python scripts/resimulate_llm_frontend.py                       # in place
    python scripts/resimulate_llm_frontend.py --output-root <dir>   # elsewhere
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pandas as pd

from fuse.llm_frontend_eval import (FRONTEND_FRESHNESS_WINDOW_CYCLES, SERVICE_INTENTS,
                                    LLMFrontendSuite, PlannerReply)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replies', type=Path, default=ROOT / 'runs' / 'llm_frontend' / 'frontend_replies.csv')
    ap.add_argument('--output-root', type=Path, default=ROOT / 'runs' / 'llm_frontend')
    args = ap.parse_args()

    frame = pd.read_csv(args.replies, keep_default_na=False)
    replies = []
    for row in frame.itertuples(index=False):
        reply = PlannerReply(model=str(row.model), intent=str(row.intent),
                             prompt_index=int(row.prompt_index), prompt=str(row.prompt),
                             raw=str(row.raw), latency_s=float(row.latency_s),
                             error=str(row.error))
        reply.kind = str(row.kind)  # type: ignore[attr-defined]
        replies.append(reply)
    print(f'{len(replies)} stored replies from {frame["model"].nunique()} models')

    # No models are consulted: the suite is used only for evaluate and summarize.
    suite = LLMFrontendSuite(gateway_models=[], local_models=[])
    plans, comparison = suite.evaluate(replies)
    summary = suite.summarize(plans, comparison)

    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / 'frontend_replies.csv', index=False)
    plans.to_csv(out / 'frontend_plans.csv', index=False)
    comparison.to_csv(out / 'frontend_comparison.csv', index=False)
    summary.to_csv(out / 'frontend_summary.csv', index=False)
    env_path = out / 'frontend_env.json'
    env = json.loads(env_path.read_text(encoding='utf-8')) if env_path.exists() else {}
    env.update({
        'resimulated_from_stored_replies': True,
        'freshness_window_cycles': FRONTEND_FRESHNESS_WINDOW_CYCLES,
        'intents': [s['intent'] for s in SERVICE_INTENTS],
    })
    env_path.write_text(json.dumps(env, indent=2), encoding='utf-8')

    cols = [c for c in ['model', 'kind', 'parse_rate', 'plan_fidelity', 'canonical_agreement',
                        'fuse_result_reuse', 'latency_reduction'] if c in summary.columns]
    print(summary[cols].to_string(index=False))
    print(f'\nwrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
