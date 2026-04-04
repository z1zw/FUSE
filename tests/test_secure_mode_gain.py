from pathlib import Path
from fuse.benchmarks import load_calibration
from fuse.sosp_plus import SOSPPlusSuite

def test_secure_elastic_improves_over_secure_ct(tmp_path):
    suite = SOSPPlusSuite(engine='cpp', calibration=load_calibration(Path(__file__).resolve().parents[1] / 'configs' / 'calibration.json'))
    case = next((case for case in suite.catalog.build() if case.name == 'Private RAG Gateway'))
    runtime_tasks = case.build_runtime_tasks(load_factor=1.0)
    elastic, _ = suite.base_suite._simulate(case.graphs, runtime_tasks, mode='secure_elastic', scenario_name='elastic', raw_dir=tmp_path)
    secure_ct, _ = suite.base_suite._simulate(case.graphs, runtime_tasks, mode='secure_ct', scenario_name='secure_ct', raw_dir=tmp_path)
    assert float(elastic.metrics['makespan_ms']) < float(secure_ct.metrics['makespan_ms'])
    assert float(elastic.metrics['result_reuse_rate']) > float(secure_ct.metrics['result_reuse_rate'])
