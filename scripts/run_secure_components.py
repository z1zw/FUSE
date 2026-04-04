from __future__ import annotations
import argparse
from pathlib import Path
from fuse.benchmarks import load_calibration
from fuse.sosp_plus import SOSPPlusSuite

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('results_plus'))
    parser.add_argument('--engine', type=str, default='cpp')
    parser.add_argument('--calibration', type=Path, default=Path('configs/calibration.json'))
    args = parser.parse_args()
    suite = SOSPPlusSuite(engine=args.engine, calibration=load_calibration(args.calibration))
    frames = suite.run_secure_components(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    suite._persist_frames(args.output_dir, {'fig21_secure_components': frames['fig21_secure_components']})
    suite._persist_raw_frames(raw_dir, frames, skip={'fig21_secure_components'})
if __name__ == '__main__':
    main()
