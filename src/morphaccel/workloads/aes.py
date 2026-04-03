from __future__ import annotations
from typing import List
from ..ir import ContextKey, Expr, Fragment, InputRef, TaskGraph

def build_aes_task(blocks: int=8192, key_bits: int=128, task_id: str='aes', operation: str='encrypt') -> TaskGraph:
    if key_bits not in {128, 256}:
        raise ValueError(f'unsupported AES key size: {key_bits}')
    if operation not in {'encrypt', 'decrypt'}:
        raise ValueError(f'unsupported AES operation: {operation}')
    rounds = 14 if key_bits == 256 else 10
    key_bytes = key_bits // 8
    task = TaskGraph(task_id=task_id, task_type='AES', domain='GF(2^8)', metadata={'blocks': blocks, 'rounds': rounds, 'key_bits': key_bits, 'algorithm': f'AES-{key_bits}', 'operation': operation})
    block_tile = 256
    tiles = max(1, blocks // block_tile)
    key = Fragment(fragment_id='key_expand', task_type='AES', stage='IM', expr=Expr.build('key_expand', InputRef('key', shape=(key_bytes,), dtype='uint8'), template_family='linear_map'), context=ContextKey(domain='GF(2^8)', dtype='uint8', layout='state4x4', tile=(key_bytes,), delta_t=1000000), estimated_ops=240 if key_bits == 256 else 176, estimated_bytes=240 if key_bits == 256 else 176, template_family='linear_map', semantic_group='key_schedule', metadata={'micro_ops': ['key_expand' if operation == 'encrypt' else 'inverse_key_expand']})
    task.add_fragment(key)
    final_round_nodes: List[str] = []
    for t in range(tiles):
        prev = 'key_expand'
        for rnd in range(rounds):
            frag_id = f'round_{rnd}_tile_{t}'
            expr = Expr.build('aes_round', InputRef(f'state_{rnd}_{t}', shape=(16,), dtype='uint8'), template_family='linear_map', round=rnd, key_bits=key_bits)
            if operation == 'encrypt':
                micro_ops = ['subbytes', 'shift_rows']
                if rnd < rounds - 1:
                    micro_ops.append('mix_columns')
            else:
                micro_ops = ['inv_shift_rows', 'inv_subbytes']
                if rnd < rounds - 1:
                    micro_ops.append('inv_mix_columns')
            micro_ops.append('add_round_key')
            frag = Fragment(fragment_id=frag_id, task_type='AES', stage='IT' if rnd < rounds - 1 else 'SC', expr=expr, context=ContextKey(domain='GF(2^8)', dtype='uint8', layout='state4x4', tile=(16,), delta_t=1000000), deps=[prev], estimated_ops=block_tile * (88 if key_bits == 256 and rnd < rounds - 1 else 80 if rnd < rounds - 1 else 72 if key_bits == 256 else 64), estimated_bytes=block_tile * 16, template_family='linear_map', semantic_group=f'aes_round_{rnd % 2}_tile_{t % 4}', metadata={'micro_ops': micro_ops})
            task.add_fragment(frag)
            prev = frag_id
        final_round_nodes.append(prev)
    merge = Fragment(fragment_id='ciphertext_merge' if operation == 'encrypt' else 'plaintext_merge', task_type='AES', stage='SC', expr=Expr.build('concat', *(InputRef(f'out_{i}', shape=(16,), dtype='uint8') for i in range(min(4, tiles))), template_family='permutation'), context=ContextKey(domain='GF(2^8)', dtype='uint8', layout='block', tile=(block_tile, 16), delta_t=1000000), deps=final_round_nodes, estimated_ops=blocks, estimated_bytes=blocks * 16, template_family='permutation', semantic_group='ciphertext_merge' if operation == 'encrypt' else 'plaintext_merge', metadata={'micro_ops': ['shift_rows'] if operation == 'encrypt' else ['inv_shift_rows']})
    task.add_fragment(merge)
    task.set_outputs([merge.fragment_id])
    return task

def build_aes128_task(blocks: int=8192, task_id: str='aes128') -> TaskGraph:
    return build_aes_task(blocks=blocks, key_bits=128, task_id=task_id, operation='encrypt')

def build_aes256_task(blocks: int=8192, task_id: str='aes256') -> TaskGraph:
    return build_aes_task(blocks=blocks, key_bits=256, task_id=task_id, operation='encrypt')

def build_aes128_encrypt_task(blocks: int=8192, task_id: str='aes128_encrypt') -> TaskGraph:
    return build_aes_task(blocks=blocks, key_bits=128, task_id=task_id, operation='encrypt')

def build_aes128_decrypt_task(blocks: int=8192, task_id: str='aes128_decrypt') -> TaskGraph:
    return build_aes_task(blocks=blocks, key_bits=128, task_id=task_id, operation='decrypt')

def build_aes256_encrypt_task(blocks: int=8192, task_id: str='aes256_encrypt') -> TaskGraph:
    return build_aes_task(blocks=blocks, key_bits=256, task_id=task_id, operation='encrypt')

def build_aes256_decrypt_task(blocks: int=8192, task_id: str='aes256_decrypt') -> TaskGraph:
    return build_aes_task(blocks=blocks, key_bits=256, task_id=task_id, operation='decrypt')
