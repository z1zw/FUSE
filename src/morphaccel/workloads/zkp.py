from __future__ import annotations
from typing import List
from ..ir import ContextKey, Expr, Fragment, InputRef, TaskGraph

def build_groth16_like_task(circuit_size: int=16384, rounds: int=4, task_id: str='groth16') -> TaskGraph:
    task = TaskGraph(task_id=task_id, task_type='ZKP', domain='Zp', metadata={'circuit_size': circuit_size, 'rounds': rounds, 'algorithm': 'Groth16-like', 'phase': 'prove'})
    tile = 1024
    tiles = max(2, circuit_size // tile)
    poseidon_nodes: List[str] = []
    prev_poseidon = None
    for rnd in range(rounds):
        ntt_nodes = []
        for t in range(tiles):
            gate_id = f'gate_r{rnd}_t{t}'
            gate_expr = Expr.build('field_mac', InputRef(f'a_{rnd}_{t}', shape=(tile,), dtype='field255'), InputRef(f'b_{rnd}_{t}', shape=(tile,), dtype='field255'), template_family='linear_map', modulo='p')
            gate = Fragment(fragment_id=gate_id, task_type='ZKP', stage='IM', expr=gate_expr, context=ContextKey(domain='Zp', dtype='field255', layout='tile', tile=(tile,), delta_t=4000000), deps=[] if prev_poseidon is None else [prev_poseidon], estimated_ops=tile * 3, estimated_bytes=tile * 16, template_family='linear_map', semantic_group=f'constraint_cluster_{t % 4}', metadata={'micro_ops': ['field_mul', 'field_add', 'field_reduce']})
            task.add_fragment(gate)
            ntt_id = f'ntt_r{rnd}_t{t}'
            ntt_expr = Expr.build('fft_butterfly', InputRef(f'gate_out_{rnd}_{t}', shape=(tile,), dtype='field255'), template_family='linear_map', stage=rnd)
            ntt = Fragment(fragment_id=ntt_id, task_type='ZKP', stage='IT', expr=ntt_expr, context=ContextKey(domain='Zp', dtype='field255', layout='tile', tile=(tile,), delta_t=4000000), deps=[gate_id], estimated_ops=tile * 2, estimated_bytes=tile * 12, template_family='linear_map', semantic_group=f'ntt_cluster_{t % 2}', metadata={'micro_ops': ['fft_butterfly', 'fft_butterfly', 'field_add']})
            task.add_fragment(ntt)
            ntt_nodes.append(ntt_id)
        poseidon_id = f'poseidon_r{rnd}'
        poseidon_expr = Expr.build('poseidon_mix', InputRef(f'ntt_merge_{rnd}', shape=(3,), dtype='field255'), template_family='nonlinear_lookup')
        poseidon = Fragment(fragment_id=poseidon_id, task_type='ZKP', stage='SC', expr=poseidon_expr, context=ContextKey(domain='Zp', dtype='field255', layout='vector', tile=(3,), delta_t=4000000), deps=ntt_nodes, estimated_ops=512, estimated_bytes=4096, template_family='nonlinear_lookup', semantic_group=f'poseidon_round_{rnd % 2}', metadata={'micro_ops': ['poseidon_sbox', 'poseidon_mix', 'field_add']})
        task.add_fragment(poseidon)
        poseidon_nodes.append(poseidon_id)
        prev_poseidon = poseidon_id
    msm = Fragment(fragment_id='msm_final', task_type='ZKP', stage='SC', expr=Expr.build('bucket_accum', InputRef('poseidon_out', shape=(tiles,), dtype='field255'), template_family='reduction'), context=ContextKey(domain='Zp', dtype='field255', layout='bucket', tile=(tiles,), delta_t=4000000), deps=poseidon_nodes, estimated_ops=max(1024, circuit_size // 2), estimated_bytes=max(4096, circuit_size * 4), template_family='reduction', semantic_group='msm_final_group', metadata={'micro_ops': ['msm_bucket', 'reduce_tree']})
    task.add_fragment(msm)
    task.set_outputs(['msm_final'])
    return task

def build_groth16_prove_like_task(circuit_size: int=16384, rounds: int=4, task_id: str='groth16_prove') -> TaskGraph:
    graph = build_groth16_like_task(circuit_size=circuit_size, rounds=rounds, task_id=task_id)
    graph.metadata['phase'] = 'prove'
    return graph

def build_groth16_verify_like_task(circuit_size: int=4096, pairings: int=3, task_id: str='groth16_verify') -> TaskGraph:
    task = TaskGraph(task_id=task_id, task_type='ZKP', domain='Zp', metadata={'circuit_size': circuit_size, 'pairings': pairings, 'algorithm': 'Groth16-like', 'phase': 'verify'})
    tile = 512
    tiles = max(2, circuit_size // tile)
    prep_nodes: List[str] = []
    for idx in range(tiles):
        prep_id = f'public_input_prep_{idx}'
        prep = Fragment(fragment_id=prep_id, task_type='ZKP', stage='IM', expr=Expr.build('public_input_accum', InputRef(f'pub_{idx}', shape=(tile,), dtype='field255'), template_family='linear_map'), context=ContextKey(domain='Zp', dtype='field255', layout='tile', tile=(tile,), delta_t=2000000), estimated_ops=tile * 2, estimated_bytes=tile * 16, template_family='linear_map', semantic_group=f'verify_input_cluster_{idx % 2}', metadata={'micro_ops': ['field_mul', 'field_add']})
        task.add_fragment(prep)
        prep_nodes.append(prep_id)
    msm_id = 'verify_msm'
    msm = Fragment(fragment_id=msm_id, task_type='ZKP', stage='IT', expr=Expr.build('msm_verify', InputRef('prepared_inputs', shape=(tiles,), dtype='field255'), template_family='linear_map'), context=ContextKey(domain='Zp', dtype='field255', layout='bucket', tile=(tiles,), delta_t=2000000), deps=prep_nodes, estimated_ops=max(1024, circuit_size // 3), estimated_bytes=max(4096, circuit_size * 3), template_family='linear_map', semantic_group='verify_msm_group', metadata={'micro_ops': ['msm_bucket', 'reduce_tree']})
    task.add_fragment(msm)
    pairing_nodes: List[str] = []
    for idx in range(pairings):
        pair_id = f'pairing_{idx}'
        pairing = Fragment(fragment_id=pair_id, task_type='ZKP', stage='SC', expr=Expr.build('pairing_check', InputRef(f'pairing_input_{idx}', shape=(12,), dtype='field255'), template_family='nonlinear_lookup'), context=ContextKey(domain='Zp', dtype='field255', layout='vector', tile=(12,), delta_t=2000000), deps=[msm_id], estimated_ops=1536, estimated_bytes=6144, template_family='nonlinear_lookup', semantic_group=f'pairing_round_{idx % 2}', metadata={'micro_ops': ['miller_loop', 'final_exponent', 'field_add']})
        task.add_fragment(pairing)
        pairing_nodes.append(pair_id)
    final_id = 'verify_decision'
    final = Fragment(fragment_id=final_id, task_type='ZKP', stage='SC', expr=Expr.build('verify_reduce', InputRef('pairing_results', shape=(pairings,), dtype='field255'), template_family='reduction'), context=ContextKey(domain='Zp', dtype='field255', layout='vector', tile=(pairings,), delta_t=2000000), deps=pairing_nodes, estimated_ops=256, estimated_bytes=1024, template_family='reduction', semantic_group='verify_reduce_group', metadata={'micro_ops': ['pairing_reduce', 'decision']})
    task.add_fragment(final)
    task.set_outputs([final_id])
    return task
