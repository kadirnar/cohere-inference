"""Triton projection kernel for the decoder steps of kernels/stream_decoder.py with up to 32 rows.

  linear            x W^T + bias (and ReLU) for up to 32 rows: weight tiles streamed once through tensor cores,
                    the bias added before the single rounding, as cuBLAS's epilogue does

The decoder's other step kernels are CUDA (kernels/csrc/decoder_ops.cu, dec_w8.cu).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---- linear --------------------------------------------------------------------------------------------------------


@triton.jit
def _linear(x_ptr, w_ptr, b_ptr, y_ptr, M, N, K, stride_xm, stride_ym,
            RELU: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    m = tl.arange(0, BLOCK_M)
    rows = m < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + m[:, None] * stride_xm + k[None, :], mask=rows[:, None], other=0.0)
        w = tl.load(w_ptr + n[:, None] * K + k[None, :])
        acc = tl.dot(x, tl.trans(w), acc)
    acc += tl.load(b_ptr + n).to(tl.float32)[None, :]
    if RELU:
        acc = tl.maximum(acc, 0.0)
    tl.store(y_ptr + m[:, None] * stride_ym + n[None, :], acc.to(y_ptr.dtype.element_ty), mask=rows[:, None])


# (rows up to, N, K) -> (BLOCK_N, BLOCK_K, warps, stages), fastest in CUDA graphs on an A10 for the decoder's shapes:
# qkv (3072, 1024), o / cross q / cross o (1024, 1024), fc1 (4096, 1024), fc2 (1024, 4096), head (16384, 1024)
LINEAR_CONFIGS = {
    (1, 3072, 1024): (64, 256, 4, 3), (1, 1024, 1024): (16, 128, 2, 4), (1, 4096, 1024): (32, 128, 2, 2),
    (1, 1024, 4096): (16, 128, 2, 4), (1, 16384, 1024): (16, 256, 4, 3),
    (2, 3072, 1024): (32, 128, 4, 3), (2, 1024, 1024): (16, 128, 2, 4), (2, 4096, 1024): (16, 128, 2, 2),
    (2, 1024, 4096): (16, 128, 2, 4), (2, 16384, 1024): (16, 256, 4, 3),
    (4, 3072, 1024): (32, 128, 4, 3), (4, 1024, 1024): (32, 128, 4, 3), (4, 4096, 1024): (16, 128, 2, 2),
    (4, 1024, 4096): (32, 256, 4, 4), (4, 16384, 1024): (16, 256, 4, 3),
    (8, 3072, 1024): (64, 128, 4, 4), (8, 1024, 1024): (32, 256, 4, 3), (8, 4096, 1024): (32, 128, 2, 2),
    (8, 1024, 4096): (32, 256, 4, 3), (8, 16384, 1024): (16, 256, 4, 3),
    (16, 3072, 1024): (64, 256, 8, 3), (16, 1024, 1024): (32, 64, 2, 4), (16, 4096, 1024): (32, 128, 2, 2),
    (16, 1024, 4096): (32, 256, 4, 3), (16, 16384, 1024): (32, 256, 4, 3),
    (32, 3072, 1024): (64, 128, 8, 3), (32, 1024, 1024): (32, 128, 4, 3), (32, 4096, 1024): (32, 128, 4, 2),
    (32, 1024, 4096): (32, 256, 4, 4), (32, 16384, 1024): (32, 256, 4, 3),
}


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, relu: bool = False) -> torch.Tensor:
    """x (M, K) @ weight (N, K)^T + bias, optionally ReLU, for M <= 32."""
    m, k = x.shape
    n = weight.shape[0]
    rows = next(r for r in (1, 2, 4, 8, 16, 32) if r >= m)
    block_n, block_k, num_warps, num_stages = LINEAR_CONFIGS.get((rows, n, k), (32, 128, 4, 3))
    assert n % block_n == 0 and k % block_k == 0 and x.stride(1) == 1 and weight.is_contiguous()
    y = torch.empty(m, n, device=x.device, dtype=x.dtype)
    _linear[(n // block_n,)](x, weight, bias, y, m, n, k, x.stride(0), y.stride(0), RELU=relu,
                             BLOCK_M=max(16, triton.next_power_of_2(m)), BLOCK_N=block_n, BLOCK_K=block_k,
                             num_warps=num_warps, num_stages=num_stages)
    return y
