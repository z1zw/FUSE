from __future__ import annotations
from typing import List, Sequence, Tuple
from ..ir import ContextKey, Expr, Fragment, InputRef, TaskGraph
VGG16_BLOCKS = [(2, 64, 224, 3), (2, 128, 112, 64), (3, 256, 56, 128), (3, 512, 28, 256), (3, 512, 14, 512)]
RESNET_SPECS = {'resnet50': {'stages': [(3, 256, 56, 64), (4, 512, 28, 256), (6, 1024, 14, 512), (3, 2048, 7, 1024)], 'stem_channels': 64}, 'resnet101': {'stages': [(3, 256, 56, 64), (4, 512, 28, 256), (23, 1024, 14, 512), (3, 2048, 7, 1024)], 'stem_channels': 64}}

def build_vgg16_like_task(batch_size: int=16, task_id: str='vgg16') -> TaskGraph:
    task = TaskGraph(task_id=task_id, task_type='CNN', domain='R', metadata={'batch_size': batch_size, 'model': 'VGG16-like'})
    prev = None
    block_outputs: List[str] = []
    layer_id = 0
    for block_idx, (num_convs, channels, spatial, in_channels) in enumerate(VGG16_BLOCKS):
        for conv_idx in range(num_convs):
            im2col_id = f'im2col_{layer_id}'
            im2col_expr = Expr.build('sliding_window', InputRef(f'x_{layer_id}', shape=(batch_size, in_channels, spatial, spatial), dtype='fp16'), template_family='permutation', kernel=(3, 3), stride=1, padding=1)
            im2col = Fragment(fragment_id=im2col_id, task_type='CNN', stage='IM', expr=im2col_expr, context=ContextKey(domain='R', dtype='fp16', layout='nchw', tile=(batch_size, channels, 16, 16), delta_t=2000000), deps=[] if prev is None else [prev], estimated_ops=batch_size * spatial * spatial * 9, estimated_bytes=batch_size * in_channels * spatial * spatial * 2, template_family='permutation', semantic_group=f'window_cluster_{block_idx}_{conv_idx % 2}', metadata={'micro_ops': ['im2col']})
            task.add_fragment(im2col)
            conv_id = f'conv_{layer_id}'
            conv_expr = Expr.build('conv_gemm', InputRef(f'cols_{layer_id}', shape=(batch_size * spatial * spatial, in_channels * 9), dtype='fp16'), InputRef(f'w_{layer_id}', shape=(channels, in_channels * 9), dtype='fp16'), template_family='linear_map')
            conv = Fragment(fragment_id=conv_id, task_type='CNN', stage='IT', expr=conv_expr, context=ContextKey(domain='R', dtype='fp16', layout='tile', tile=(batch_size, channels, 16, 16), delta_t=2000000), deps=[im2col_id], estimated_ops=batch_size * spatial * spatial * channels * in_channels * 9 // 256, estimated_bytes=batch_size * channels * spatial * spatial * 2, template_family='linear_map', semantic_group=f'kernel_bank_{block_idx % 3}', metadata={'micro_ops': ['matmul_tile']})
            task.add_fragment(conv)
            relu_id = f'relu_{layer_id}'
            relu_expr = Expr.build('relu', InputRef(f'conv_{layer_id}', shape=(batch_size, channels, spatial, spatial), dtype='fp16'), template_family='nonlinear_lookup')
            relu_frag = Fragment(fragment_id=relu_id, task_type='CNN', stage='SC', expr=relu_expr, context=ContextKey(domain='R', dtype='fp16', layout='nchw', tile=(batch_size, channels, 16, 16), delta_t=2000000), deps=[conv_id], estimated_ops=batch_size * channels * spatial * spatial, estimated_bytes=batch_size * channels * spatial * spatial * 2, template_family='nonlinear_lookup', semantic_group=f'activation_bank_{block_idx % 2}', metadata={'micro_ops': ['relu']})
            task.add_fragment(relu_frag)
            prev = relu_id
            in_channels = channels
            layer_id += 1
        pool_id = f'pool_b{block_idx}'
        pool_expr = Expr.build('max', InputRef(f'relu_last_b{block_idx}', shape=(batch_size, channels, spatial, spatial), dtype='fp16'), template_family='reduction', kernel=2, stride=2)
        pool = Fragment(fragment_id=pool_id, task_type='CNN', stage='SC', expr=pool_expr, context=ContextKey(domain='R', dtype='fp16', layout='nchw', tile=(batch_size, channels, 8, 8), delta_t=2000000), deps=[prev] if prev else [], estimated_ops=batch_size * channels * (spatial // 2) * (spatial // 2) * 4, estimated_bytes=batch_size * channels * spatial * spatial * 2, template_family='reduction', semantic_group=f'pool_group_{block_idx}', metadata={'micro_ops': ['maxpool']})
        task.add_fragment(pool)
        prev = pool_id
        block_outputs.append(pool_id)
    task.set_outputs([prev] if prev else block_outputs)
    return task

def build_mlp_path_task(batch_size: int=32, hidden_dim: int=1024, task_id: str='mlp') -> TaskGraph:
    task = TaskGraph(task_id=task_id, task_type='CNN', domain='R', metadata={'batch_size': batch_size, 'hidden_dim': hidden_dim, 'model': 'MLP-path'})
    gemm = Fragment(fragment_id='gemm', task_type='CNN', stage='IT', expr=Expr.build('matmul_tile', InputRef('x', shape=(batch_size, hidden_dim), dtype='fp16'), InputRef('w', shape=(hidden_dim, hidden_dim), dtype='fp16'), template_family='linear_map'), context=ContextKey(domain='R', dtype='fp16', layout='tile', tile=(batch_size, hidden_dim), delta_t=2000000), estimated_ops=batch_size * hidden_dim * hidden_dim // 64, estimated_bytes=batch_size * hidden_dim * 2, template_family='linear_map', semantic_group='mlp_gemm', metadata={'micro_ops': ['matmul_tile']})
    gelu = Fragment(fragment_id='gelu', task_type='CNN', stage='SC', expr=Expr.build('gelu', InputRef('gemm_out', shape=(batch_size, hidden_dim), dtype='fp16'), template_family='nonlinear_lookup'), context=ContextKey(domain='R', dtype='fp16', layout='tile', tile=(batch_size, hidden_dim), delta_t=2000000), deps=['gemm'], estimated_ops=batch_size * hidden_dim, estimated_bytes=batch_size * hidden_dim * 2, template_family='nonlinear_lookup', semantic_group='mlp_act', metadata={'micro_ops': ['gelu']})
    norm = Fragment(fragment_id='norm', task_type='CNN', stage='SC', expr=Expr.build('layernorm', InputRef('gelu_out', shape=(batch_size, hidden_dim), dtype='fp16'), template_family='reduction'), context=ContextKey(domain='R', dtype='fp16', layout='tile', tile=(batch_size, hidden_dim), delta_t=2000000), deps=['gelu'], estimated_ops=batch_size * hidden_dim * 2, estimated_bytes=batch_size * hidden_dim * 2, template_family='reduction', semantic_group='mlp_norm', metadata={'micro_ops': ['layernorm']})
    task.add_fragment(gemm)
    task.add_fragment(gelu)
    task.add_fragment(norm)
    task.set_outputs(['norm'])
    return task

def build_resnet_task(model_name: str='resnet101', batch_size: int=8, task_id: str='resnet') -> TaskGraph:
    model_key = model_name.lower()
    if model_key not in RESNET_SPECS:
        raise ValueError(f'unsupported resnet model: {model_name}')
    spec = RESNET_SPECS[model_key]
    task = TaskGraph(task_id=task_id, task_type='CNN', domain='R', metadata={'batch_size': batch_size, 'model': model_key})
    stem = Fragment(fragment_id='stem_conv', task_type='CNN', stage='IM', expr=Expr.build('conv_gemm', InputRef('image', shape=(batch_size, 3, 224, 224), dtype='fp16'), InputRef('stem_w', shape=(spec['stem_channels'], 3 * 7 * 7), dtype='fp16'), template_family='linear_map'), context=ContextKey(domain='R', dtype='fp16', layout='nchw', tile=(batch_size, spec['stem_channels'], 56, 56), delta_t=2000000), estimated_ops=batch_size * 112 * 112 * spec['stem_channels'] * 3 * 49 // 512, estimated_bytes=batch_size * 3 * 224 * 224 * 2, template_family='linear_map', semantic_group='resnet_stem', metadata={'micro_ops': ['conv7x7', 'maxpool']})
    task.add_fragment(stem)
    prev = 'stem_conv'
    for stage_idx, (blocks, out_channels, spatial, in_channels) in enumerate(spec['stages']):
        for block_idx in range(blocks):
            proj_id = f'stage{stage_idx}_block{block_idx}_proj'
            proj = Fragment(fragment_id=proj_id, task_type='CNN', stage='IT', expr=Expr.build('matmul_tile', InputRef(f'res_{stage_idx}_{block_idx}', shape=(batch_size, spatial * spatial, in_channels), dtype='fp16'), InputRef(f'w_proj_{stage_idx}_{block_idx}', shape=(in_channels, out_channels), dtype='fp16'), template_family='linear_map'), context=ContextKey(domain='R', dtype='fp16', layout='tile', tile=(batch_size, out_channels, min(spatial, 28), min(spatial, 28)), delta_t=2000000), deps=[prev], estimated_ops=batch_size * spatial * spatial * in_channels * out_channels // 512, estimated_bytes=batch_size * out_channels * spatial * spatial * 2, template_family='linear_map', semantic_group=f'resnet_stage{stage_idx}_proj', metadata={'micro_ops': ['conv1x1']})
            task.add_fragment(proj)
            bottleneck_id = f'stage{stage_idx}_block{block_idx}_bottleneck'
            bottleneck = Fragment(fragment_id=bottleneck_id, task_type='CNN', stage='IT', expr=Expr.build('conv_gemm', InputRef(f'proj_{stage_idx}_{block_idx}', shape=(batch_size, spatial, spatial, out_channels), dtype='fp16'), InputRef(f'w_bneck_{stage_idx}_{block_idx}', shape=(out_channels, out_channels * 9), dtype='fp16'), template_family='linear_map'), context=ContextKey(domain='R', dtype='fp16', layout='tile', tile=(batch_size, out_channels, min(spatial, 28), min(spatial, 28)), delta_t=2000000), deps=[proj_id], estimated_ops=batch_size * spatial * spatial * out_channels * 9 // 32, estimated_bytes=batch_size * out_channels * spatial * spatial * 2, template_family='linear_map', semantic_group=f'resnet_stage{stage_idx}_bottleneck', metadata={'micro_ops': ['conv3x3']})
            task.add_fragment(bottleneck)
            merge_id = f'stage{stage_idx}_block{block_idx}_merge'
            merge = Fragment(fragment_id=merge_id, task_type='CNN', stage='SC', expr=Expr.build('add', InputRef(f'skip_{stage_idx}_{block_idx}', shape=(batch_size, out_channels, spatial, spatial), dtype='fp16'), InputRef(f'main_{stage_idx}_{block_idx}', shape=(batch_size, out_channels, spatial, spatial), dtype='fp16'), template_family='linear_map'), context=ContextKey(domain='R', dtype='fp16', layout='nchw', tile=(batch_size, out_channels, min(spatial, 28), min(spatial, 28)), delta_t=2000000), deps=[bottleneck_id], estimated_ops=batch_size * out_channels * spatial * spatial, estimated_bytes=batch_size * out_channels * spatial * spatial * 2, template_family='linear_map', semantic_group=f'resnet_stage{stage_idx}_merge', metadata={'micro_ops': ['residual_add', 'relu']})
            task.add_fragment(merge)
            prev = merge_id
    head = Fragment(fragment_id='global_pool_fc', task_type='CNN', stage='SC', expr=Expr.build('global_avg_pool', InputRef('resnet_tail', shape=(batch_size, 2048, 7, 7), dtype='fp16'), template_family='reduction'), context=ContextKey(domain='R', dtype='fp16', layout='nchw', tile=(batch_size, 2048, 7, 7), delta_t=2000000), deps=[prev], estimated_ops=batch_size * 2048 * 49, estimated_bytes=batch_size * 2048 * 2, template_family='reduction', semantic_group='resnet_head', metadata={'micro_ops': ['avgpool', 'fc']})
    task.add_fragment(head)
    task.set_outputs(['global_pool_fc'])
    return task
