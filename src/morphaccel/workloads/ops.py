from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Iterable, List, Sequence, Tuple
import numpy as np
FIELD_PRIME = 2147483647

def field_add(x: np.ndarray, y: np.ndarray, p: int=FIELD_PRIME) -> np.ndarray:
    return (x + y) % p

def field_mul(x: np.ndarray, y: np.ndarray, p: int=FIELD_PRIME) -> np.ndarray:
    return x * y % p

def field_reduce(x: np.ndarray, p: int=FIELD_PRIME) -> np.ndarray:
    return x % p

def toy_poseidon_round(state: np.ndarray, round_const: int=7, p: int=FIELD_PRIME) -> np.ndarray:
    x = (state + round_const) % p
    x = np.power(x, 5, dtype=np.int64) % p
    mix = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]], dtype=np.int64)
    return (mix[:len(x), :len(x)] @ x.reshape(-1, 1)).reshape(-1) % p

def next_power_of_two(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()

def primitive_root_of_unity(n: int, p: int=FIELD_PRIME) -> int:
    g = 3
    return pow(g, (p - 1) // n, p)

def ntt(values: np.ndarray, p: int=FIELD_PRIME) -> np.ndarray:
    n = len(values)
    root = primitive_root_of_unity(n, p)
    out = np.zeros_like(values)
    for k in range(n):
        acc = 0
        for j, xj in enumerate(values):
            acc = (acc + int(xj) * pow(root, j * k, p)) % p
        out[k] = acc
    return out

def msm_like(points: np.ndarray, scalars: np.ndarray, p: int=FIELD_PRIME) -> int:
    return int(np.sum(points * scalars % p) % p)

def im2col(x: np.ndarray, kernel_size: Tuple[int, int], stride: int=1, padding: int=0) -> np.ndarray:
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    n, c, h, w = x.shape
    kh, kw = kernel_size
    oh = (h - kh) // stride + 1
    ow = (w - kw) // stride + 1
    cols = []
    for i in range(0, h - kh + 1, stride):
        for j in range(0, w - kw + 1, stride):
            window = x[:, :, i:i + kh, j:j + kw].reshape(n, -1)
            cols.append(window)
    return np.stack(cols, axis=1).reshape(n * oh * ow, -1)

def conv2d_im2col(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None=None, stride: int=1, padding: int=0) -> np.ndarray:
    n, c, h, w = x.shape
    oc, _, kh, kw = weight.shape
    cols = im2col(x, (kh, kw), stride=stride, padding=padding)
    out = cols @ weight.reshape(oc, -1).T
    if bias is not None:
        out += bias
    oh = (h + 2 * padding - kh) // stride + 1
    ow = (w + 2 * padding - kw) // stride + 1
    return out.reshape(n, oh, ow, oc).transpose(0, 3, 1, 2)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)

def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def maxpool2d(x: np.ndarray, kernel_size: int=2, stride: int | None=None) -> np.ndarray:
    stride = stride or kernel_size
    n, c, h, w = x.shape
    oh = (h - kernel_size) // stride + 1
    ow = (w - kernel_size) // stride + 1
    out = np.zeros((n, c, oh, ow), dtype=x.dtype)
    for i in range(oh):
        for j in range(ow):
            window = x[:, :, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size]
            out[:, :, i, j] = np.max(window, axis=(2, 3))
    return out

def layernorm(x: np.ndarray, eps: float=1e-05) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)
SBOX = np.array([99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215, 171, 118, 202, 130, 201, 125, 250, 89, 71, 240, 173, 212, 162, 175, 156, 164, 114, 192, 183, 253, 147, 38, 54, 63, 247, 204, 52, 165, 229, 241, 113, 216, 49, 21, 4, 199, 35, 195, 24, 150, 5, 154, 7, 18, 128, 226, 235, 39, 178, 117, 9, 131, 44, 26, 27, 110, 90, 160, 82, 59, 214, 179, 41, 227, 47, 132, 83, 209, 0, 237, 32, 252, 177, 91, 106, 203, 190, 57, 74, 76, 88, 207, 208, 239, 170, 251, 67, 77, 51, 133, 69, 249, 2, 127, 80, 60, 159, 168, 81, 163, 64, 143, 146, 157, 56, 245, 188, 182, 218, 33, 16, 255, 243, 210, 205, 12, 19, 236, 95, 151, 68, 23, 196, 167, 126, 61, 100, 93, 25, 115, 96, 129, 79, 220, 34, 42, 144, 136, 70, 238, 184, 20, 222, 94, 11, 219, 224, 50, 58, 10, 73, 6, 36, 92, 194, 211, 172, 98, 145, 149, 228, 121, 231, 200, 55, 109, 141, 213, 78, 169, 108, 86, 244, 234, 101, 122, 174, 8, 186, 120, 37, 46, 28, 166, 180, 198, 232, 221, 116, 31, 75, 189, 139, 138, 112, 62, 181, 102, 72, 3, 246, 14, 97, 53, 87, 185, 134, 193, 29, 158, 225, 248, 152, 17, 105, 217, 142, 148, 155, 30, 135, 233, 206, 85, 40, 223, 140, 161, 137, 13, 191, 230, 66, 104, 65, 153, 45, 15, 176, 84, 187, 22], dtype=np.uint8)
RCON = np.array([1, 2, 4, 8, 16, 32, 64, 128, 27, 54], dtype=np.uint8)

def xtime(a: int) -> int:
    return (a << 1 ^ 27) & 255 if a & 128 else a << 1 & 255

def mix_single_column(a: np.ndarray) -> np.ndarray:
    t = a[0] ^ a[1] ^ a[2] ^ a[3]
    u = a[0]
    a0 = a[0] ^ t ^ xtime(int(a[0] ^ a[1]))
    a1 = a[1] ^ t ^ xtime(int(a[1] ^ a[2]))
    a2 = a[2] ^ t ^ xtime(int(a[2] ^ a[3]))
    a3 = a[3] ^ t ^ xtime(int(a[3] ^ u))
    return np.array([a0, a1, a2, a3], dtype=np.uint8)

def sub_bytes(state: np.ndarray) -> np.ndarray:
    return SBOX[state]

def shift_rows(state: np.ndarray) -> np.ndarray:
    s = state.reshape(4, 4).copy()
    for r in range(4):
        s[r] = np.roll(s[r], -r)
    return s.reshape(-1)

def mix_columns(state: np.ndarray) -> np.ndarray:
    s = state.reshape(4, 4).T.copy()
    for i in range(4):
        s[i] = mix_single_column(s[i])
    return s.T.reshape(-1)

def add_round_key(state: np.ndarray, round_key: np.ndarray) -> np.ndarray:
    return np.bitwise_xor(state, round_key)

def rot_word(word: np.ndarray) -> np.ndarray:
    return np.roll(word, -1)

def sub_word(word: np.ndarray) -> np.ndarray:
    return SBOX[word]

def key_expansion(key: np.ndarray) -> List[np.ndarray]:
    words = [key[i:i + 4].copy() for i in range(0, 16, 4)]
    for i in range(4, 44):
        temp = words[i - 1].copy()
        if i % 4 == 0:
            temp = sub_word(rot_word(temp))
            temp[0] ^= RCON[i // 4 - 1]
        words.append(np.bitwise_xor(words[i - 4], temp))
    round_keys = []
    for r in range(11):
        round_keys.append(np.concatenate(words[4 * r:4 * (r + 1)]))
    return round_keys

def aes128_encrypt_block(block: np.ndarray, key: np.ndarray) -> np.ndarray:
    round_keys = key_expansion(key)
    state = add_round_key(block, round_keys[0])
    for rnd in range(1, 10):
        state = sub_bytes(state)
        state = shift_rows(state)
        state = mix_columns(state)
        state = add_round_key(state, round_keys[rnd])
    state = sub_bytes(state)
    state = shift_rows(state)
    state = add_round_key(state, round_keys[10])
    return state
