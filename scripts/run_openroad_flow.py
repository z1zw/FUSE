from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from fuse.eval_tools import EvaluationToolkit
from fuse.open_source_eval import clone_or_update_repo, host_build_tools
PLATFORM_DEFAULT = 'sky130hd'
DESIGN_NAME = 'fuse_top'
DESIGN_ALIAS = 'fuse'
RTL_FILES = [ROOT / 'rtl' / 'fuse_top.v', ROOT / 'rtl' / 'rfu.v', ROOT / 'rtl' / 'tars.v', ROOT / 'rtl' / 'fvm.v', ROOT / 'rtl' / 'lrc.v']

def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')

def stage_design(*, work_dir: Path, platform: str, clock_period_ns: float, place_density: float, die_area: tuple[int, int, int, int], core_area: tuple[int, int, int, int]) -> dict[str, Path]:
    design_dir = work_dir / 'flow' / 'designs' / platform / DESIGN_ALIAS
    src_dir = design_dir / 'src'
    src_dir.mkdir(parents=True, exist_ok=True)
    for rtl in RTL_FILES:
        shutil.copy2(rtl, src_dir / rtl.name)
    die_area_s = ' '.join((str(v) for v in die_area))
    core_area_s = ' '.join((str(v) for v in core_area))
    config = '\n'.join([f'export DESIGN_NAME = {DESIGN_NAME}', f'export PLATFORM = {platform}', 'export VERILOG_FILES = $(sort $(wildcard $(DESIGN_HOME)/src/*.v))', 'export SDC_FILE = $(DESIGN_HOME)/constraint.sdc', f'export DIE_AREA = {die_area_s}', f'export CORE_AREA = {core_area_s}', 'export CLOCK_PORT = clk', f'export CLOCK_PERIOD = {clock_period_ns:.3f}', f'export PLACE_DENSITY = {place_density:.3f}', 'export MAX_FANOUT_CONSTRAINT = 12', 'export MAX_TRANSITION_CONSTRAINT = 0.75', 'export MAX_CAPACITANCE_CONSTRAINT = 0.20', ''])
    io_delay = max(clock_period_ns * 0.1, 1.0)
    sdc = '\n'.join([f'create_clock -period {clock_period_ns:.3f} [get_ports clk]', f'set_input_delay {io_delay:.3f} -clock clk [remove_from_collection [all_inputs] [get_ports clk]]', f'set_output_delay {io_delay:.3f} -clock clk [all_outputs]', 'set_driving_cell -lib_cell sky130_fd_sc_hd__buf_4 [remove_from_collection [all_inputs] [get_ports clk]]', 'set_load 0.05 [all_outputs]', ''])
    write_file(design_dir / 'config.mk', config)
    write_file(design_dir / 'constraint.sdc', sdc)
    return {'design_dir': design_dir, 'config_mk': design_dir / 'config.mk', 'constraint_sdc': design_dir / 'constraint.sdc'}

def locate_repo(flow_dir: Path) -> Path:
    if (flow_dir / 'flow').exists():
        return flow_dir
    if flow_dir.name == 'flow' and (flow_dir / 'Makefile').exists():
        return flow_dir.parent
    raise FileNotFoundError(f'OpenROAD-flow-scripts root not found under {flow_dir}')

def copy_design_into_repo(staged_design_dir: Path, flow_root: Path, platform: str) -> Path:
    target_dir = flow_root / 'flow' / 'designs' / platform / DESIGN_ALIAS
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(staged_design_dir, target_dir)
    return target_dir

def _candidate_files(base_dir: Path, names: Iterable[str]) -> list[Path]:
    return [base_dir / name for name in names]

def copy_reports(flow_root: Path, output_dir: Path, platform: str) -> dict[str, Path]:
    reports_dir = output_dir / 'reports' / 'openroad'
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_root = flow_root / 'flow' / 'reports' / platform / DESIGN_ALIAS / DESIGN_NAME
    logs_root = flow_root / 'flow' / 'logs' / platform / DESIGN_ALIAS / DESIGN_NAME
    results_root = flow_root / 'flow' / 'results' / platform / DESIGN_ALIAS / DESIGN_NAME
    candidates = {'fuse_area.rpt': _candidate_files(report_root, ['6_finish.rpt', 'final.rpt', 'metrics.rpt']), 'fuse_power.rpt': _candidate_files(report_root, ['6_finish.rpt', '5_1_grt_sta.power.rpt', 'power.rpt']), 'fuse_timing.rpt': _candidate_files(report_root, ['6_finish.rpt', '5_1_grt_sta.summary.rpt', 'timing.rpt']), 'fuse_metrics.json': _candidate_files(logs_root, ['6_finish.json', 'metrics.json', 'metadata.json']), 'fuse_metrics.csv': _candidate_files(logs_root, ['metrics.csv']), 'fuse_final.def': _candidate_files(results_root, ['6_final.def', 'final.def']), 'fuse_final.v': _candidate_files(results_root, ['6_final.v', 'final.v']), 'fuse_final.sdc': _candidate_files(results_root, ['6_final.sdc', 'final.sdc']), 'fuse_final.gds': _candidate_files(results_root, ['6_final.gds', 'final.gds'])}
    copied: dict[str, Path] = {}
    for name, options in candidates.items():
        for option in options:
            if option.exists():
                dest = reports_dir / name
                shutil.copy2(option, dest)
                copied[name] = dest
                break
    return copied

def write_tool_config(output_dir: Path, copied: dict[str, Path]) -> Path:
    config = {'openroad_area_report': str(copied.get('fuse_area.rpt', '')), 'openroad_power_report': str(copied.get('fuse_power.rpt', '')), 'openroad_timing_report': str(copied.get('fuse_timing.rpt', '')), 'openroad_metrics_json': str(copied.get('fuse_metrics.json', '')), 'openroad_metrics_csv': str(copied.get('fuse_metrics.csv', ''))}
    config_path = output_dir / 'openroad_eval_tools.json'
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    return config_path

def git_head(repo_dir: Path) -> str | None:
    try:
        result = subprocess.run(['git', '-C', str(repo_dir), 'rev-parse', 'HEAD'], check=True, text=True, capture_output=True)
        return result.stdout.strip()
    except Exception:
        return None

def docker_run_command(*, flow_root: Path, image: str, platform: str, threads: int | None, target: str) -> list[str]:
    flow_mount = flow_root / 'flow'
    cmd = ['docker', 'run', '--rm', '-v', f'{flow_mount.as_posix()}:/OpenROAD-flow-scripts/flow', '-w', '/OpenROAD-flow-scripts/flow', image, 'bash', '-lc', _make_make_command(platform=platform, threads=threads, target=target)]
    return cmd

def local_run_command(*, flow_root: Path, platform: str, threads: int | None, target: str) -> list[str]:
    shell = 'bash' if shutil.which('bash') else None
    if shell is None:
        raise RuntimeError('Local OpenROAD mode requires bash on PATH')
    return [shell, '-lc', f"cd '{(flow_root / 'flow').as_posix()}' && {_make_make_command(platform=platform, threads=threads, target=target)}"]

def _make_make_command(*, platform: str, threads: int | None, target: str) -> str:
    thread_part = f' -j {threads}' if threads and threads > 0 else ''
    return f'make{thread_part} DESIGN_CONFIG=designs/{platform}/{DESIGN_ALIAS}/config.mk {target}'.strip()

def run_flow(*, flow_root: Path, mode: str, docker_image: str, platform: str, threads: int | None, target: str, output_dir: Path) -> subprocess.CompletedProcess:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = docker_run_command(flow_root=flow_root, image=docker_image, platform=platform, threads=threads, target=target) if mode == 'docker' else local_run_command(flow_root=flow_root, platform=platform, threads=threads, target=target)
    if mode == 'docker':
        _verify_docker_available()
    return subprocess.run(cmd, check=True, text=True, capture_output=True, env=dict(os.environ))

def _verify_docker_available() -> None:
    if shutil.which('docker') is None:
        raise RuntimeError('Docker mode requires docker on PATH')
    probe = subprocess.run(['docker', 'info'], text=True, capture_output=True, env=dict(os.environ))
    if probe.returncode != 0:
        message = (probe.stderr or probe.stdout or '').strip()
        raise RuntimeError(f'Docker daemon is unavailable: {message}')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', type=Path, default=ROOT / 'openroad_work')
    parser.add_argument('--flow-dir', type=Path, default=None)
    parser.add_argument('--fetch', action='store_true')
    parser.add_argument('--update', action='store_true')
    parser.add_argument('--mode', choices=['docker', 'local'], default='docker')
    parser.add_argument('--docker-image', default='openroad/orfs:latest')
    parser.add_argument('--platform', default=PLATFORM_DEFAULT)
    parser.add_argument('--clock-period-ns', type=float, default=20.0)
    parser.add_argument('--place-density', type=float, default=0.55)
    parser.add_argument('--die-area', nargs=4, type=int, metavar=('LX', 'LY', 'UX', 'UY'), default=(0, 0, 1600, 1600))
    parser.add_argument('--core-area', nargs=4, type=int, metavar=('LX', 'LY', 'UX', 'UY'), default=(20, 20, 1580, 1580))
    parser.add_argument('--threads', type=int, default=None)
    parser.add_argument('--target', default='')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'openroad_results')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    flow_dir = args.flow_dir
    if args.fetch:
        flow_dir = clone_or_update_repo('openroad', root=ROOT, update=args.update)
    if flow_dir is None:
        flow_dir = ROOT / 'external' / 'OpenROAD-flow-scripts'
    flow_root = locate_repo(flow_dir)
    staged = stage_design(work_dir=args.work_dir, platform=args.platform, clock_period_ns=args.clock_period_ns, place_density=args.place_density, die_area=tuple(args.die_area), core_area=tuple(args.core_area))
    repo_design_dir = copy_design_into_repo(staged['design_dir'], flow_root, args.platform)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'flow_root': str(flow_root), 'flow_git_head': git_head(flow_root), 'staged_design_dir': str(staged['design_dir']), 'repo_design_dir': str(repo_design_dir), 'platform': args.platform, 'mode': args.mode, 'docker_image': args.docker_image if args.mode == 'docker' else None, 'clock_period_ns': args.clock_period_ns, 'place_density': args.place_density, 'die_area': list(args.die_area), 'core_area': list(args.core_area), 'target': args.target, 'threads': args.threads, 'run_requested': bool(args.run), 'host_tools': host_build_tools()}
    if args.run:
        manifest['run_command'] = docker_run_command(flow_root=flow_root, image=args.docker_image, platform=args.platform, threads=args.threads, target=args.target) if args.mode == 'docker' else local_run_command(flow_root=flow_root, platform=args.platform, threads=args.threads, target=args.target)
        try:
            result = run_flow(flow_root=flow_root, mode=args.mode, docker_image=args.docker_image, platform=args.platform, threads=args.threads, target=args.target, output_dir=output_dir)
            copied = copy_reports(flow_root, output_dir, args.platform)
            config_path = write_tool_config(output_dir, copied)
            toolkit = EvaluationToolkit(ROOT, config_path=config_path)
            parsed = toolkit.parse_openroad()
            manifest['reports'] = {k: str(v) for k, v in copied.items()}
            manifest['tool_config'] = str(config_path)
            manifest['parsed'] = parsed
            manifest['stdout_path'] = str(output_dir / 'openroad_stdout.log')
            manifest['stderr_path'] = str(output_dir / 'openroad_stderr.log')
            (output_dir / 'openroad_stdout.log').write_text(result.stdout or '', encoding='utf-8')
            (output_dir / 'openroad_stderr.log').write_text(result.stderr or '', encoding='utf-8')
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            manifest['run_failed'] = True
            manifest['error'] = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                manifest['stdout_path'] = str(output_dir / 'openroad_stdout.log')
                manifest['stderr_path'] = str(output_dir / 'openroad_stderr.log')
                (output_dir / 'openroad_stdout.log').write_text(exc.stdout or '', encoding='utf-8')
                (output_dir / 'openroad_stderr.log').write_text(exc.stderr or '', encoding='utf-8')
            (output_dir / 'openroad_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            raise
    (output_dir / 'openroad_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(output_dir)
if __name__ == '__main__':
    main()
