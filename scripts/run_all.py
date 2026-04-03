from __future__ import annotations
from pathlib import Path
import os
import subprocess
import sys

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(root / 'src')
    subprocess.run([sys.executable, str(root / 'scripts' / 'build_cpp.py')], check=True, env=env)
    subprocess.run([sys.executable, str(root / 'scripts' / 'train_calibration.py'), '--output', str(root / 'configs' / 'calibration.json'), '--engine', 'cpp'], check=True, env=env)
    subprocess.run([sys.executable, str(root / 'scripts' / 'run_sosp_suite.py'), '--output-dir', str(root / 'results'), '--engine', 'cpp', '--calibration', str(root / 'configs' / 'calibration.json')], check=True, env=env)
    subprocess.run([sys.executable, str(root / 'scripts' / 'plot_sosp_figures.py'), '--input-dir', str(root / 'results'), '--output-dir', str(root / 'figures')], check=True, env=env)
    subprocess.run([sys.executable, str(root / 'scripts' / 'compare_baselines.py'), '--input-dir', str(root / 'results'), '--output', str(root / 'results' / 'comparison_summary.csv')], check=True, env=env)
if __name__ == '__main__':
    main()
