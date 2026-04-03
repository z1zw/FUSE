from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from morphaccel.open_source_eval import clone_or_update_repo, dramsim3_binary, git_head, host_build_tools, platform_summary, resolve_tool_repo, run_dramsim3, write_dramsim3_trace, write_json

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=None)
    parser.add_argument('--config', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports' / 'dramsim3')
    parser.add_argument('--total-bytes', type=int, default=256 * 1024 * 1024)
    parser.add_argument('--burst-bytes', type=int, default=64)
    parser.add_argument('--stride-bytes', type=int, default=64)
    parser.add_argument('--read-ratio', type=float, default=0.75)
    parser.add_argument('--issue-gap-cycles', type=int, default=32)
    parser.add_argument('--cycles', type=int, default=250000)
    parser.add_argument('--fetch', action='store_true')
    parser.add_argument('--update', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    repo_dir = args.repo or resolve_tool_repo('dramsim3', ROOT)
    if args.fetch:
        repo_dir = clone_or_update_repo('dramsim3', root=ROOT, update=args.update)
    config_path = args.config or repo_dir / 'configs' / 'DDR4_8Gb_x8_3200.ini'
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / 'fuse_offchip.trace'
    request_count = write_dramsim3_trace(trace_path, total_bytes=args.total_bytes, burst_bytes=args.burst_bytes, read_ratio=args.read_ratio, stride_bytes=args.stride_bytes, issue_gap_cycles=args.issue_gap_cycles)
    run_command = [str(dramsim3_binary(repo_dir) or repo_dir / 'build' / 'dramsim3main'), str(config_path), '-c', str(int(args.cycles)), '-t', str(trace_path), '-o', str(output_dir)]
    manifest = {'repo_dir': str(repo_dir), 'repo_git_head': git_head(repo_dir), 'config_path': str(config_path), 'trace_path': str(trace_path), 'request_count': request_count, 'run_requested': bool(args.run), 'binary_present': dramsim3_binary(repo_dir) is not None, 'run_command': run_command, 'host_tools': host_build_tools(), 'platform': platform_summary()}
    if args.run:
        try:
            if not config_path.exists():
                raise RuntimeError(f'DRAMSim3 config not found: {config_path}')
            result = run_dramsim3(repo_dir, config_path=config_path, trace_path=trace_path, output_dir=output_dir, cycles=args.cycles)
            manifest['stdout_path'] = str(output_dir / 'dramsim3_stdout.log')
            manifest['stderr_path'] = str(output_dir / 'dramsim3_stderr.log')
            (output_dir / 'dramsim3_stdout.log').write_text(result.stdout or '', encoding='utf-8')
            (output_dir / 'dramsim3_stderr.log').write_text(result.stderr or '', encoding='utf-8')
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            manifest['run_failed'] = True
            manifest['error'] = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                manifest['stdout_path'] = str(output_dir / 'dramsim3_stdout.log')
                manifest['stderr_path'] = str(output_dir / 'dramsim3_stderr.log')
                (output_dir / 'dramsim3_stdout.log').write_text(exc.stdout or '', encoding='utf-8')
                (output_dir / 'dramsim3_stderr.log').write_text(exc.stderr or '', encoding='utf-8')
            write_json(output_dir / 'dramsim3_manifest.json', manifest)
            raise
    write_json(output_dir / 'dramsim3_manifest.json', manifest)
    print(output_dir)
if __name__ == '__main__':
    main()
