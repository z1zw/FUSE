from fuse.workloads import build_aes128_decrypt_task, build_aes128_encrypt_task, build_groth16_prove_like_task, build_groth16_verify_like_task

def test_groth16_verify_workload_has_verify_metadata() -> None:
    graph = build_groth16_verify_like_task(circuit_size=2048, pairings=2, task_id='verify_case')
    assert graph.metadata['phase'] == 'verify'
    assert graph.metadata['algorithm'] == 'Groth16-like'
    assert graph.outputs == ['verify_decision']
    pairing_ops = graph.fragments['pairing_0'].metadata['micro_ops']
    assert 'miller_loop' in pairing_ops
    assert 'final_exponent' in pairing_ops

def test_groth16_prove_workload_has_prove_metadata() -> None:
    graph = build_groth16_prove_like_task(circuit_size=4096, rounds=2, task_id='prove_case')
    assert graph.metadata['phase'] == 'prove'
    assert graph.outputs == ['msm_final']

def test_aes_encrypt_and_decrypt_workloads_are_distinct() -> None:
    enc = build_aes128_encrypt_task(blocks=512, task_id='enc')
    dec = build_aes128_decrypt_task(blocks=512, task_id='dec')
    assert enc.metadata['operation'] == 'encrypt'
    assert dec.metadata['operation'] == 'decrypt'
    assert enc.outputs == ['ciphertext_merge']
    assert dec.outputs == ['plaintext_merge']
    assert 'subbytes' in enc.fragments['round_0_tile_0'].metadata['micro_ops']
    assert 'inv_subbytes' in dec.fragments['round_0_tile_0'].metadata['micro_ops']
