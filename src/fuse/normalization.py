from __future__ import annotations
from dataclasses import replace
import hashlib
import json
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from .ir import Const, ContextKey, Expr, ExprArg, Fragment, InputRef
COMMUTATIVE_OPS = {'add', 'mul', 'field_add', 'field_mul', 'xor', 'max', 'bucket_accum'}
ASSOCIATIVE_OPS = COMMUTATIVE_OPS | {'concat'}
LINEAR_MACROS = {'mix_columns', 'fft_butterfly', 'matmul_tile', 'conv_mac'}
BLACK_BOX_MACROS = {'sbox_lookup', 'poseidon_sbox', 'relu', 'gelu'}
AFFINE_INDEX_OPS = {'sliding_window', 'gather', 'tile_extract', 'permute'}
ALIAS_TABLE = {'conv_gemm': 'matmul_tile', 'field_mac': 'field_mul_add', 'mix_shift': 'mix_columns', 'aes_lookup': 'sbox_lookup'}

class Normalizer:

    def normalize_expr(self, expr: Expr, structural: bool=True) -> Expr:
        op = ALIAS_TABLE.get(expr.op, expr.op)
        normalized_args = [self.normalize_arg(arg, structural=structural) for arg in expr.args]
        normalized_attrs = self._normalize_attrs(op, dict(expr.attrs), structural=structural)
        if op in ASSOCIATIVE_OPS:
            flattened: List[ExprArg] = []
            for arg in normalized_args:
                if isinstance(arg, Expr) and arg.op == op:
                    flattened.extend(arg.args)
                else:
                    flattened.append(arg)
            normalized_args = flattened
        if op in COMMUTATIVE_OPS:
            normalized_args = sorted(normalized_args, key=self._sort_key)
        folded = self._constant_fold(op, normalized_args, normalized_attrs)
        if folded is not None:
            return folded
        template_family = expr.template_family or op
        if op in LINEAR_MACROS:
            template_family = 'linear_map'
        elif op in BLACK_BOX_MACROS:
            template_family = 'nonlinear_lookup'
        elif template_family in {'conv_mac', 'matmul_tile'}:
            template_family = 'linear_map'
        elif template_family in {'shift_rows', 'permute', 'byte_shuffle'}:
            template_family = 'permutation'
        return Expr(op=op, args=tuple(normalized_args), attrs=tuple(sorted(((k, self._freeze(v)) for k, v in normalized_attrs.items()))), template_family=template_family)

    def normalize_arg(self, arg: ExprArg, structural: bool=True) -> ExprArg:
        if isinstance(arg, Expr):
            return self.normalize_expr(arg, structural=structural)
        if isinstance(arg, InputRef):
            if structural:
                return InputRef(name='_', shape=arg.shape, dtype=self._structural_dtype(arg.dtype))
            return arg
        if isinstance(arg, Const):
            if structural:
                return Const(value=type(arg.value).__name__)
            return arg
        raise TypeError(f'Unsupported arg type: {type(arg)!r}')

    def structural_hash(self, fragment: Fragment) -> str:
        expr = self.normalize_expr(fragment.expr, structural=True)
        payload = {'op': expr.op, 'template_family': expr.template_family, 'expr': expr.to_dict(), 'stage': fragment.stage}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def semantic_hash(self, fragment: Fragment) -> str:
        expr = self.normalize_expr(fragment.expr, structural=False)
        payload = {'expr': expr.to_dict(), 'context': fragment.context.__dict__, 'semantic_group': fragment.semantic_group, 'stage': fragment.stage, 'task_type': fragment.task_type}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]

    def _normalize_attrs(self, op: str, attrs: Dict[str, Any], structural: bool) -> Dict[str, Any]:
        if op in AFFINE_INDEX_OPS:
            start = attrs.pop('start', None)
            offset = attrs.pop('offset', None)
            if start is not None or offset is not None:
                attrs['affine_origin'] = 'dynamic' if structural else (start, offset)
        if structural and 'name' in attrs:
            attrs.pop('name')
        if structural and 'runtime_seed' in attrs:
            attrs.pop('runtime_seed')
        return attrs

    def _constant_fold(self, op: str, args: Sequence[ExprArg], attrs: Dict[str, Any]) -> Expr | None:
        if not args or not all((isinstance(a, Const) for a in args)):
            return None
        values = [a.value for a in args if isinstance(a, Const)]
        try:
            if op in {'add', 'field_add'}:
                return Expr.build('const', Const(sum(values)))
            if op in {'mul', 'field_mul'}:
                out = 1
                for v in values:
                    out *= v
                return Expr.build('const', Const(out))
            if op == 'xor':
                out = 0
                for v in values:
                    out ^= int(v)
                return Expr.build('const', Const(out))
            if op == 'max':
                return Expr.build('const', Const(max(values)))
        except Exception:
            return None
        return None

    def context_compatible(self, lhs: ContextKey, rhs: ContextKey) -> bool:
        return lhs.adapt_cost(rhs) is not None

    def context_adaptation_cost(self, lhs: ContextKey, rhs: ContextKey) -> int | None:
        return lhs.adapt_cost(rhs)

    def _sort_key(self, arg: ExprArg) -> Tuple[str, str]:
        if isinstance(arg, Expr):
            return ('expr', json.dumps(arg.to_dict(), sort_keys=True))
        if isinstance(arg, InputRef):
            return ('input', json.dumps(arg.to_structural(), sort_keys=True))
        if isinstance(arg, Const):
            return ('const', json.dumps(arg.to_structural(), sort_keys=True))
        return ('other', repr(arg))

    def _freeze(self, value: Any) -> Any:
        if isinstance(value, dict):
            return tuple(((k, self._freeze(v)) for k, v in sorted(value.items())))
        if isinstance(value, (list, tuple)):
            return tuple((self._freeze(v) for v in value))
        return value

    def _structural_dtype(self, dtype: str) -> str:
        numeric = {'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'field255', 'fp16', 'fp32'}
        if dtype in numeric:
            return 'numeric'
        return dtype
