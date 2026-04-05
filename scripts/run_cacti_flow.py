from __future__ import annotations
import argparse
from pathlib import Path
import re
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from fuse.open_source_eval import cacti_binary, clone_or_update_repo, file_metadata, git_head, host_build_tools, platform_summary, resolve_tool_repo, run_cacti, write_json

def stage_one(repo_dir: Path, output_dir: Path, name: str, size_bytes: int, technology_um: float) -> dict[str, str]:
    cfg_path = output_dir / f'{name}.cfg'
    report_path = output_dir / f'{name}.txt'
    template_path = repo_dir / 'cache.cfg'
    text = template_path.read_text(encoding='utf-8', errors='ignore')

    replacements = {
        r'(?m)^-size \(bytes\) .*': f'-size (bytes) {int(size_bytes)}',
        r'(?m)^-block size \(bytes\) .*': '-block size (bytes) 64',
        r'(?m)^-associativity .*': '-associativity 1',
        r'(?m)^-read-write port .*': '-read-write port 1',
        r'(?m)^-exclusive read port .*': '-exclusive read port 0',
        r'(?m)^-exclusive write port .*': '-exclusive write port 0',
        r'(?m)^-single ended read ports .*': '-single ended read ports 0',
        r'(?m)^-UCA bank count .*': '-UCA bank count 1',
        r'(?m)^-technology \(u\) .*': f'-technology (u) {technology_um:.3f}',
        r'(?m)^-output/input bus width .*': '-output/input bus width 512',
        r'(?m)^-operating temperature \(K\) .*': '-operating temperature (K) 360',
        r'(?m)^-cache type ".*"': '-cache type "cache"',
        r'(?m)^-tag size \(b\) .*': '-tag size (b) "default"',
        r'(?m)^-access mode \(normal, sequential, fast\) - ".*"': '-access mode (normal, sequential, fast) - "normal"',
        r'(?m)^-Data array cell type - ".*"': '-Data array cell type - "itrs-hp"',
        r'(?m)^-Data array peripheral type - ".*"': '-Data array peripheral type - "itrs-hp"',
        r'(?m)^-Tag array cell type - ".*"': '-Tag array cell type - "itrs-hp"',
        r'(?m)^-Tag array peripheral type - ".*"': '-Tag array peripheral type - "itrs-hp"',
    }

    for pattern, replacement in replacements.items():
        text, count = re.subn(pattern, replacement, text, count=1)
        if count != 1:
            raise RuntimeError(f'failed to update CACTI template pattern: {pattern}')

    cfg_path.write_text(text, encoding='utf-8')
    return {'config': str(cfg_path), 'report': str(report_path)}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports' / 'cacti')
    parser.add_argument('--technology-um', type=float, default=0.065)
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
    staged = {'feature_buffer': stage_one(repo_dir, output_dir, 'feature_buffer', args.feature_bytes, args.technology_um), 'fragment_cache': stage_one(repo_dir, output_dir, 'fragment_cache', args.fragment_bytes, args.technology_um), 'micro_op_buffer': stage_one(repo_dir, output_dir, 'micro_op_buffer', args.microop_bytes, args.technology_um)}
    binary = cacti_binary(repo_dir)
    for value in staged.values():
        value['config_meta'] = file_metadata(Path(value['config']))
    manifest = {
        'repo_dir': str(repo_dir),
        'repo_git_head': git_head(repo_dir),
        'binary_present': binary is not None,
        'binary_meta': file_metadata(binary or repo_dir / 'cacti'),
        'run_requested': bool(args.run),
        'cwd': str(repo_dir.resolve()),
        'host_tools': host_build_tools(),
        'platform': platform_summary(),
        'artifacts': staged,
        'run_commands': {name: [str(binary or repo_dir / 'cacti'), '-infile', value['config']] for name, value in staged.items()},
    }
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
                value['report_meta'] = file_metadata(Path(value['report']))
                value['stdout_meta'] = file_metadata(stdout_path)
                value['stderr_meta'] = file_metadata(stderr_path)
            manifest['binary_present'] = cacti_binary(repo_dir) is not None
            manifest['binary_meta'] = file_metadata(cacti_binary(repo_dir) or repo_dir / 'cacti')
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
