"""A Triton kernel for the convolution module of the Parakeet (FastConformer) encoder of Cohere Transcribe.

transformers' ParakeetEncoderConvolutionModule works in (batch, channels, time): it transposes the input, runs the
1x1 convolutions through cuDNN, and makes separate passes for GLU, the padding mask (from a (batch, 1, T, T) mask
reduced on every layer), the depthwise convolution, BatchNorm and SiLU, then transposes back. Here everything stays
in (batch, time, channels): the 1x1 convolutions are plain matrix products, and one Triton kernel computes
GLU -> padding mask -> depthwise convolution (kernel 9) -> BatchNorm (inference statistics, folded with the depthwise
bias into one scale and shift per channel) -> SiLU, accumulating in fp32.

    from kernels.conv_module import patch_conv_modules
    patch_conv_modules(model.model.encoder)
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _glu_dwconv_silu(y_ptr, valid_ptr, w_ptr, shift_ptr, out_ptr, T, C,
                     stride_yb, stride_yt, stride_ob, stride_ot,
                     K: tl.constexpr, PAD: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_t, pid_c, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = c < C
    acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    for k in tl.static_range(K):
        ts = t + (k - PAD)
        inside = (ts >= 0) & (ts < T)
        valid = tl.load(valid_ptr + b * T + ts, mask=inside, other=0) != 0  # padded frames count as zeros
        m = (inside & valid)[:, None] & cmask[None, :]
        row = y_ptr + b * stride_yb + ts[:, None] * stride_yt
        a = tl.load(row + c[None, :], mask=m, other=0.0).to(tl.float32)
        g = tl.load(row + (c[None, :] + C), mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + c * K + k, mask=cmask, other=0.0)
        acc += (a * tl.sigmoid(g)) * w[None, :]
    acc += tl.load(shift_ptr + c, mask=cmask, other=0.0)[None, :]
    out = acc * tl.sigmoid(acc)
    dst = out_ptr + b * stride_ob + t[:, None] * stride_ot + c[None, :]
    tl.store(dst, out.to(out_ptr.dtype.element_ty), mask=(t < T)[:, None] & cmask[None, :])


def glu_dwconv_silu(y: torch.Tensor, valid: torch.Tensor, weight: torch.Tensor, shift: torch.Tensor,
                    block_t: int = 8, block_c: int = 128, num_warps: int = 2) -> torch.Tensor:
    """y (batch, T, 2C) -> SiLU(depthwise_conv(mask(GLU(y))) * scale + shift), (batch, T, C); weight (C, K) already
    multiplied by the BatchNorm scale, shift (C) = depthwise bias * scale + BatchNorm shift, valid (batch, T) uint8.
    Small tiles (8 frames x 128 channels, 2 warps) were fastest on an A10 for every shape tried: larger ones spill
    registers across the 9 taps."""
    b, t, c2 = y.shape
    c, k = weight.shape
    assert c2 == 2 * c and y.stride(2) == 1
    out = torch.empty(b, t, c, device=y.device, dtype=y.dtype)
    grid = (triton.cdiv(t, block_t), triton.cdiv(c, block_c), b)
    _glu_dwconv_silu[grid](y, valid, weight, shift, out, t, c, y.stride(0), y.stride(1), out.stride(0), out.stride(1),
                           K=k, PAD=(k - 1) // 2, BLOCK_T=block_t, BLOCK_C=block_c, num_warps=num_warps)
    return out


def fast_forward(self, hidden_states, attention_mask=None):
    """Same result as ParakeetEncoderConvolutionModule.forward, for (batch, time, channels) input and output."""
    b, t, _ = hidden_states.shape
    if attention_mask is not None:  # (batch, 1, T, T): row 0 is valid_0 & valid_j, i.e. the frames' validity
        valid = attention_mask[:, 0, 0, :].to(torch.uint8).contiguous()
    else:
        valid = torch.ones(b, t, dtype=torch.uint8, device=hidden_states.device)
    y = F.linear(hidden_states, self._w1, self._b1)
    z = glu_dwconv_silu(y, valid, self._wd, self._shift)
    return F.linear(z, self._w2, self._b2)


def patch_conv_modules(encoder) -> int:
    """Fold each ParakeetEncoderConvolutionModule's weights for the kernel and replace its forward (inference only:
    BatchNorm uses its running statistics). Returns how many were patched."""
    n = 0
    for module in encoder.modules():
        if type(module).__name__ != "ParakeetEncoderConvolutionModule":
            continue
        assert not module.training and type(module.activation).__name__ in ("SiLUActivation", "SiLU")
        bn, dw = module.norm, module.depthwise_conv
        scale = bn.weight.float() / torch.sqrt(bn.running_var.float() + bn.eps)
        shift = bn.bias.float() - bn.running_mean.float() * scale
        module._wd = (dw.weight[:, 0, :].float() * scale[:, None]).contiguous()
        module._shift = ((dw.bias.float() if dw.bias is not None else 0.0) * scale + shift).contiguous()
        module._w1 = module.pointwise_conv1.weight[:, :, 0].contiguous()
        module._b1 = module.pointwise_conv1.bias
        module._w2 = module.pointwise_conv2.weight[:, :, 0].contiguous()
        module._b2 = module.pointwise_conv2.bias
        module.forward = types.MethodType(fast_forward, module)
        n += 1
    return n
