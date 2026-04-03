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
    frames = suite.run_ablations(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    suite._persist_frames(args.output_dir, {'fig17_ablation_plus': frames['fig17_ablation_plus']})
    suite._persist_raw_frames(raw_dir, frames, skip={'fig17_ablation_plus'})
if __name__ == '__main__':
    main()
