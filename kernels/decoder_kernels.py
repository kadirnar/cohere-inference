"""Triton kernels for one decoding step of the Cohere Transcribe decoder (kernels/graph_decoder.py).

A step feeds one token per row through 8 layers; with few rows each operation is small, so the step is mostly kernel
launches and weight reads. These kernels cut the launches and fold the small operations into their neighbours, with
the model's arithmetic (bf16 storage, fp32 accumulation, a rounding wherever transformers rounds):

  linear            x W^T + bias (and ReLU) for up to 32 rows: weight tiles streamed once through tensor cores,
                    the bias added before the single rounding, as cuBLAS's epilogue does
  add_layernorm     the residual sum and the next LayerNorm in one pass (x += y, h = LayerNorm(x))
  embed_layernorm   token + position embedding, the embedding LayerNorm and the first layer's LayerNorm
  self_attention    q, k, v from the fused projection: the new key and value written to the cache and attended
                    together with the cached ones, over the valid positions only (split over programs for few rows)
  decode_attention  the same over the audio frames for cross-attention, each row with its own number of frames
  pick              greedy choice (first index among equal logits, as torch.argmax), prompt forcing, end of rows
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


# ---- LayerNorm -----------------------------------------------------------------------------------------------------


@triton.jit
def _layernorm(x, w_ptr, b_ptr, eps, N: tl.constexpr):
    c = tl.arange(0, N)
    xf = x.to(tl.float32)
    mean = tl.sum(xf, axis=0) / N
    xc = xf - mean
    var = tl.sum(xc * xc, axis=0) / N
    return xc * tl.rsqrt(var + eps) * tl.load(w_ptr + c).to(tl.float32) + tl.load(b_ptr + c).to(tl.float32)


@triton.jit
def _add_layernorm(x_ptr, y_ptr, w_ptr, b_ptr, h_ptr, eps, stride_x, stride_y, stride_h, N: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, N)
    x = tl.load(x_ptr + row * stride_x + c).to(tl.float32) + tl.load(y_ptr + row * stride_y + c).to(tl.float32)
    x = x.to(x_ptr.dtype.element_ty)  # the residual stream is bf16, as in transformers
    tl.store(x_ptr + row * stride_x + c, x)
    tl.store(h_ptr + row * stride_h + c, _layernorm(x, w_ptr, b_ptr, eps, N).to(h_ptr.dtype.element_ty))


def add_layernorm(x: torch.Tensor, y: torch.Tensor, weight, bias, eps: float, h: torch.Tensor) -> None:
    """x += y (in place, rounded to x's dtype), then h = LayerNorm(x); (rows, N) each."""
    _add_layernorm[(x.shape[0],)](x, y, weight, bias, h, eps, x.stride(0), y.stride(0), h.stride(0), N=x.shape[1],
                                  num_warps=4)


@triton.jit
def _embed_layernorm(tok_ptr, pos_ptr, emb_ptr, pemb_ptr, w0_ptr, b0_ptr, eps0, w1_ptr, b1_ptr, eps1, x_ptr, h_ptr,
                     stride_x, stride_h, N: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, N)
    t = tl.load(tok_ptr + row)
    p = tl.load(pos_ptr)
    e = tl.load(emb_ptr + t * N + c).to(tl.float32) + tl.load(pemb_ptr + p * N + c).to(tl.float32)
    x = _layernorm(e.to(x_ptr.dtype.element_ty), w0_ptr, b0_ptr, eps0, N).to(x_ptr.dtype.element_ty)
    tl.store(x_ptr + row * stride_x + c, x)
    tl.store(h_ptr + row * stride_h + c, _layernorm(x, w1_ptr, b1_ptr, eps1, N).to(h_ptr.dtype.element_ty))


def embed_layernorm(tokens, pos, embed, pos_table, ln0, ln1, x, h) -> None:
    """x = LayerNorm0(embed[tokens] + pos_table[pos]) and h = LayerNorm1(x), for every row of tokens."""
    _embed_layernorm[(tokens.shape[0],)](tokens, pos, embed, pos_table, *ln0, *ln1, x, h, x.stride(0), h.stride(0),
                                         N=x.shape[1], num_warps=4)


# ---- attention -----------------------------------------------------------------------------------------------------


@triton.jit
def _attend_range(q, k_base, v_base, stride_kt, start, end, m_i, l_i, acc, D: tl.constexpr, BLOCK_T: tl.constexpr):
    """Online softmax of q over keys [start, end), continuing from (m_i, l_i, acc)."""
    d = tl.arange(0, D)
    for t0 in range(start, end, BLOCK_T):
        t = t0 + tl.arange(0, BLOCK_T)
        valid = t < end
        k = tl.load(k_base + t[:, None] * stride_kt + d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        scores = tl.where(valid, tl.sum(k * q[None, :], axis=1), float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        v = tl.load(v_base + t[:, None] * stride_kt + d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new
    return m_i, l_i, acc


@triton.jit
def _finish(m_i, l_i, acc, part_ptr, out_ptr, b, h, s, stride_ob, stride_oh, D: tl.constexpr, SPLITS: tl.constexpr):
    d = tl.arange(0, D)
    if SPLITS == 1:
        tl.store(out_ptr + b * stride_ob + h * stride_oh + d, (acc / l_i).to(out_ptr.dtype.element_ty))
    else:  # partial softmax of this split: running max, sum and weighted values (an empty split leaves -inf, 0, 0)
        part = part_ptr + ((b * tl.num_programs(1) + h) * SPLITS + s) * (D + 2)
        tl.store(part + d, acc)
        tl.store(part + D, m_i)
        tl.store(part + D + 1, l_i)


@triton.jit
def _split_range(n, s, SPLITS: tl.constexpr, BLOCK_T: tl.constexpr):
    chunk = tl.cdiv(tl.cdiv(n, SPLITS), BLOCK_T) * BLOCK_T
    start = s * chunk
    return start, tl.minimum(n, start + chunk)


@triton.jit
def _cross_attention(q_ptr, k_ptr, v_ptr, len_ptr, part_ptr, out_ptr, scale,
                     stride_qb, stride_qh, stride_kb, stride_kh, stride_kt, stride_ob, stride_oh,
                     D: tl.constexpr, BLOCK_T: tl.constexpr, SPLITS: tl.constexpr):
    b, h, s = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start, end = _split_range(tl.load(len_ptr + b), s, SPLITS, BLOCK_T)
    q = tl.load(q_ptr + b * stride_qb + h * stride_qh + tl.arange(0, D)).to(tl.float32) * scale
    m_i = tl.zeros([], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([], dtype=tl.float32)
    acc = tl.zeros([D], dtype=tl.float32)
    m_i, l_i, acc = _attend_range(q, k_ptr + b * stride_kb + h * stride_kh, v_ptr + b * stride_kb + h * stride_kh,
                                  stride_kt, start, end, m_i, l_i, acc, D, BLOCK_T)
    _finish(m_i, l_i, acc, part_ptr, out_ptr, b, h, s, stride_ob, stride_oh, D, SPLITS)


@triton.jit
def _self_attention(qkv_ptr, k_ptr, v_ptr, pos_ptr, part_ptr, out_ptr, scale, stride_qkv,
                    stride_kb, stride_kh, stride_kt, stride_ob, stride_oh,
                    H: tl.constexpr, D: tl.constexpr, BLOCK_T: tl.constexpr, SPLITS: tl.constexpr):
    b, h, s = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p = tl.load(pos_ptr)  # the new token's position: keys 0..p-1 are in the cache
    d = tl.arange(0, D)
    row = qkv_ptr + b * stride_qkv + h * D
    q = tl.load(row + d).to(tl.float32) * scale
    k_base = k_ptr + b * stride_kb + h * stride_kh
    v_base = v_ptr + b * stride_kb + h * stride_kh
    start, end = _split_range(p, s, SPLITS, BLOCK_T)
    if s == 0:  # the new key and value: into the cache (no program reads position p in this step) and attended here
        kn = tl.load(row + H * D + d)
        vn = tl.load(row + 2 * H * D + d)
        tl.store(k_base + p * stride_kt + d, kn)
        tl.store(v_base + p * stride_kt + d, vn)
        m_i = tl.sum(kn.to(tl.float32) * q, axis=0)
        l_i = tl.zeros([], dtype=tl.float32) + 1.0
        acc = vn.to(tl.float32)
    else:
        m_i = tl.zeros([], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([], dtype=tl.float32)
        acc = tl.zeros([D], dtype=tl.float32)
    m_i, l_i, acc = _attend_range(q, k_base, v_base, stride_kt, start, end, m_i, l_i, acc, D, BLOCK_T)
    _finish(m_i, l_i, acc, part_ptr, out_ptr, b, h, s, stride_ob, stride_oh, D, SPLITS)


@triton.jit
def _combine_splits(part_ptr, out_ptr, stride_ob, stride_oh, D: tl.constexpr, SPLITS: tl.constexpr):
    b, h = tl.program_id(0), tl.program_id(1)
    base = part_ptr + (b * tl.num_programs(1) + h) * SPLITS * (D + 2)
    sp = tl.arange(0, SPLITS)
    m = tl.load(base + sp * (D + 2) + D)
    l = tl.load(base + sp * (D + 2) + D + 1)
    w = tl.where(l > 0, tl.exp(m - tl.max(m, axis=0)), 0.0)
    acc = tl.load(base + sp[:, None] * (D + 2) + tl.arange(0, D)[None, :])
    out = tl.sum(acc * w[:, None], axis=0) / tl.sum(l * w, axis=0)
    tl.store(out_ptr + b * stride_ob + h * stride_oh + tl.arange(0, D), out.to(out_ptr.dtype.element_ty))


def _tiles(splits: int) -> tuple[int, int]:
    """(keys per block, warps): small tiles when a row's keys are spread over many programs (A10 measurements)."""
    return (32, 1) if splits >= 8 else (64, 2)


def _combine(scratch, out, splits: int) -> None:
    b, h, d = out.shape
    _combine_splits[(b, h)](scratch, out, out.stride(0), out.stride(1), D=d, SPLITS=splits, num_warps=1)


def decode_attention(q, k, v, lengths, scale: float, splits: int = 1, scratch=None) -> torch.Tensor:
    """softmax(q k^T * scale) v over the first lengths[b] keys of each row: q (B, H, D), k and v (B, H, T, D) with the
    same strides, lengths int32 (B,). splits > 1 spreads each row's keys over that many programs (scratch: at least
    B*H*splits*(D+2) fp32) and combines them in a second kernel."""
    b, h, d = q.shape
    assert k.stride() == v.stride() and k.stride(3) == 1 and q.stride(2) == 1
    out = torch.empty(b, h, d, device=q.device, dtype=q.dtype)
    if splits > 1 and scratch is None:
        scratch = torch.empty(b * h * splits * (d + 2), device=q.device, dtype=torch.float32)
    block_t, num_warps = _tiles(splits)
    _cross_attention[(b, h, splits)](q, k, v, lengths, scratch if splits > 1 else out, out, scale, q.stride(0),
                                     q.stride(1), k.stride(0), k.stride(1), k.stride(2), out.stride(0), out.stride(1),
                                     D=d, BLOCK_T=block_t, SPLITS=splits, num_warps=num_warps)
    if splits > 1:
        _combine(scratch, out, splits)
    return out


def self_attention(qkv, k_cache, v_cache, pos, heads: int, scale: float, splits: int = 1, scratch=None) -> torch.Tensor:
    """Attention of the new token (q, k, v side by side in qkv (B, 3*H*D)) over the cached positions 0..pos-1 and
    itself; its key and value are written to the caches (B, H, S, D) at pos. Returns (B, H*D)."""
    b = qkv.shape[0]
    d = k_cache.shape[-1]
    out = torch.empty(b, heads, d, device=qkv.device, dtype=qkv.dtype)
    if splits > 1 and scratch is None:
        scratch = torch.empty(b * heads * splits * (d + 2), device=qkv.device, dtype=torch.float32)
    block_t, num_warps = _tiles(splits)
    _self_attention[(b, heads, splits)](qkv, k_cache, v_cache, pos, scratch if splits > 1 else out, out, scale,
                                        qkv.stride(0), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
                                        out.stride(0), out.stride(1), H=heads, D=d, BLOCK_T=block_t, SPLITS=splits,
                                        num_warps=num_warps)
    if splits > 1:
        _combine(scratch, out, splits)
    return out.view(b, heads * d)


# ---- greedy choice -------------------------------------------------------------------------------------------------


@triton.jit
def _pick(logits_ptr, V, stride_l, forced_ptr, stride_f, out_ptr, stride_o, finished_ptr, tokens_ptr, pos_ptr,
          prompt_len_ptr, PAD: tl.constexpr, EOS: tl.constexpr, BLOCK_V: tl.constexpr):
    row = tl.program_id(0)
    best = tl.zeros([], dtype=tl.float32) - float("inf")
    arg = tl.zeros([], dtype=tl.int64)
    for v0 in range(0, V, BLOCK_V):
        x = tl.load(logits_ptr + row * stride_l + v0 + tl.arange(0, BLOCK_V)).to(tl.float32)
        mx = tl.max(x, axis=0)
        ix = tl.argmax(x, axis=0).to(tl.int64) + v0  # first index of the block's maximum
        arg = tl.where(mx > best, ix, arg)  # strictly greater: an earlier block keeps a tie
        best = tl.maximum(best, mx)
    at = tl.load(pos_ptr) + 1
    in_prompt = at < tl.load(prompt_len_ptr)
    done = tl.load(finished_ptr + row) != 0
    new = tl.where(in_prompt, tl.load(forced_ptr + row * stride_f + at), tl.where(done, PAD, arg))
    tl.store(out_ptr + row * stride_o + at, new)
    tl.store(finished_ptr + row, (done | ((in_prompt == 0) & (new == EOS))).to(tl.uint8))
    tl.store(tokens_ptr + row, new)


def pick(logits, forced, out, finished, tokens, pos, prompt_len, pad: int, eos: int) -> None:
    """Next token of each row: the prompt's while inside it, else the argmax of the logits (pad once finished);
    written to out[:, pos + 1] and tokens, finished (uint8) set at the end-of-text token."""
    v = logits.shape[1]
    block = 2048
    assert v % block == 0
    _pick[(logits.shape[0],)](logits, v, logits.stride(0), forced, forced.stride(0), out, out.stride(0), finished,
                              tokens, pos, prompt_len, PAD=pad, EOS=eos, BLOCK_V=block, num_warps=4)
