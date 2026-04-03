from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from morphaccel.open_source_eval import cacti_binary, clone_or_update_repo, git_head, host_build_tools, platform_summary, resolve_tool_repo, run_cacti, write_cacti_config, write_json

def stage_one(output_dir: Path, name: str, size_bytes: int, technology_um: float) -> dict[str, str]:
    cfg_path = output_dir / f'{name}.cfg'
    report_path = output_dir / f'{name}.txt'
    write_cacti_config(cfg_path, size_bytes=size_bytes, technology_um=technology_um)
    return {'config': str(cfg_path), 'report': str(report_path)}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports' / 'cacti')
    parser.add_argument('--technology-um', type=float, default=0.032)
    parser.add_argument('--feature-bytes', type=int, default=8 * 1024 * 1024)
    parser.add_argument('--fragment-bytes', type=int, default=2 * 1024 * 1024)
    parser.add_argument('--microop-bytes', type=int, default=512 * 1024)
    parser.add_argument('--fetch', action='store_true')
    parser.add_argument('--update', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    repo_dir = args.repo or resolve_tool_repo('cacti', ROOT)
    if args.fetch:
        repo_dir = clone_or_update_repo('cacti', root=ROOT, update=args.update)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    staged = {'feature_buffer': stage_one(output_dir, 'feature_buffer', args.feature_bytes, args.technology_um), 'fragment_cache': stage_one(output_dir, 'fragment_cache', args.fragment_bytes, args.technology_um), 'micro_op_buffer': stage_one(output_dir, 'micro_op_buffer', args.microop_bytes, args.technology_um)}
    binary = cacti_binary(repo_dir)
    manifest = {'repo_dir': str(repo_dir), 'repo_git_head': git_head(repo_dir), 'binary_present': binary is not None, 'run_requested': bool(args.run), 'host_tools': host_build_tools(), 'platform': platform_summary(), 'artifacts': staged, 'run_commands': {name: [str(binary or repo_dir / 'cacti'), '-infile', value['config']] for name, value in staged.items()}}
    if args.run:
        try:
            for name, value in staged.items():
                result = run_cacti(repo_dir, config_path=Path(value['config']), output_path=Path(value['report']))
                stdout_path = output_dir / f'{name}.stdout.log'
                stderr_path = output_dir / f'{name}.stderr.log'
                stdout_path.write_text(result.stdout or '', encoding='utf-8')
                stderr_path.write_text(result.stderr or '', encoding='utf-8')
                value['stdout'] = str(stdout_path)
                value['stderr'] = str(stderr_path)
            manifest['binary_present'] = cacti_binary(repo_dir) is not None
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            manifest['run_failed'] = True
            manifest['error'] = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                (output_dir / 'cacti_stdout.log').write_text(exc.stdout or '', encoding='utf-8')
                (output_dir / 'cacti_stderr.log').write_text(exc.stderr or '', encoding='utf-8')
                manifest['stdout_path'] = str(output_dir / 'cacti_stdout.log')
                manifest['stderr_path'] = str(output_dir / 'cacti_stderr.log')
            write_json(output_dir / 'cacti_manifest.json', manifest)
            raise
    write_json(output_dir / 'cacti_manifest.json', manifest)
    print(output_dir)
if __name__ == '__main__':
    main()
