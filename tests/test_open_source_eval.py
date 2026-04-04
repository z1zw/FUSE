from pathlib import Path
from fuse.open_source_eval import write_cacti_config, write_dramsim3_trace

def test_write_dramsim3_trace(tmp_path: Path) -> None:
    trace_path = tmp_path / 'probe.trace'
    request_count = write_dramsim3_trace(trace_path, total_bytes=256, burst_bytes=64, read_ratio=0.5, stride_bytes=64, issue_gap_cycles=10)
    text = trace_path.read_text(encoding='utf-8').strip().splitlines()
    assert request_count == 4
    assert text[0] == '0x20000000 READ 0'
    assert text[-1] == '0x200000C0 WRITE 30'

def test_write_cacti_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / 'feature.cfg'
    write_cacti_config(cfg_path, size_bytes=4096, technology_um=0.032)
    text = cfg_path.read_text(encoding='utf-8')
    assert '-size (bytes) 4096' in text
    assert '-technology (u) 0.032' in text
    assert '-cache type "ram"' in text
