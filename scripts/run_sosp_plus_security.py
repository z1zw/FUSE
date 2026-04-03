from __future__ import annotations
import argparse
from pathlib import Path
from morphaccel.benchmarks import load_calibration
from morphaccel.sosp_plus import SOSPPlusSuite

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('results_plus'))
    parser.add_argument('--engine', type=str, default='cpp')
    parser.add_argument('--calibration', type=Path, default=Path('configs/calibration.json'))
    args = parser.parse_args()
    suite = SOSPPlusSuite(engine=args.engine, calibration=load_calibration(args.calibration))
    frames = suite.run_security(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    suite._persist_frames(args.output_dir, {'fig13_security': frames['fig13_security'], 'fig14_llm_stability': frames['fig14_llm_stability'], 'fig15_secure_tradeoff': frames['fig15_secure_tradeoff']})
    suite._persist_raw_frames(raw_dir, frames, skip={'fig13_security', 'fig14_llm_stability', 'fig15_secure_tradeoff'})
if __name__ == '__main__':
    main()
