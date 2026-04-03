from pathlib import Path
from morphaccel.benchmarks import load_calibration
from morphaccel.llm import HeuristicLLMFrontend, MorphAccelVerifier
from morphaccel.sosp import SOSPSuite
from morphaccel.sosp_plus import SOSPPlusSuite
from morphaccel.workloads import build_transformer_block_task

def test_transformer_workload_is_dag():
    task = build_transformer_block_task(seq_len=512, hidden_dim=1024, heads=16, task_id='tfm_test')
    task.validate()
    assert len(task.fragments) >= 8

def test_llm_canonicalization_stabilizes_variants():
    frontend = HeuristicLLMFrontend()
    verifier = MorphAccelVerifier()
    prompt_a = 'compile a transformer block for llm prefill seq 2048 hidden 4096 heads 32'
    prompt_b = 'generate an llm attention block with sequence 2048 hidden 4096 32 heads'
    insecure_a = frontend.compile_prompt(prompt_a, secure=False, task_prefix='a')
    insecure_b = frontend.compile_prompt(prompt_b, secure=False, task_prefix='b')
    secure_a = frontend.compile_prompt(prompt_a, secure=True, task_prefix='a')
    secure_b = frontend.compile_prompt(prompt_b, secure=True, task_prefix='b')
    assert insecure_a.prompt_hash != insecure_b.prompt_hash
    assert verifier.verify_graphs(secure_a.graphs).ok
    assert verifier.equivalent(secure_a.graphs, secure_b.graphs)

def test_sosp_security_tables_exist(tmp_path):
    suite = SOSPSuite(engine='cpp', calibration=load_calibration(Path(__file__).resolve().parents[1] / 'configs' / 'calibration.json'))
    frames = suite.run_security(tmp_path)
    assert not frames['fig13_security'].empty
    assert not frames['fig14_llm_stability'].empty
    assert not frames['fig15_secure_tradeoff'].empty

def test_sosp_plus_service_variants_and_helpers_work(tmp_path):
    frontend = HeuristicLLMFrontend()
    variants = frontend.variant_prompts()
    assert 'service' in variants
    assert variants['service']
    suite = SOSPPlusSuite(engine='py')
    frames = suite.run_security(tmp_path)
    assert not frames['fig13_security'].empty
