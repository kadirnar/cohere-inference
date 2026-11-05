"""Faster relative-position attention for the Parakeet (FastConformer) encoder of Cohere Transcribe, same maths.

transformers' ParakeetEncoderAttention projects the relative positions once per batch item (they are the same for all
items), then builds the (batch, heads, T, 2T-1) position scores and makes three more full passes over them (padding for
the shift, scaling, masking) before attention. Here the positions are projected once, the scaling is folded into the
query, and the Transformer-XL shift is a zero-copy strided view, so only the masking pass is left.

    from kernels.relpos_attention import patch_encoder
    patch_encoder(model.model.encoder)
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F


def fast_forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
    b, t, _ = hidden_states.shape
    h, d = self.config.num_attention_heads, self.head_dim
    q = self.q_proj(hidden_states).view(b, t, h, d).transpose(1, 2)
    k = self.k_proj(hidden_states).view(b, t, h, d).transpose(1, 2)
    v = self.v_proj(hidden_states).view(b, t, h, d).transpose(1, 2)
    qu = q + self.bias_u.view(1, h, 1, d)
    qv = (q + self.bias_v.view(1, h, 1, d)) * self.scaling
    p = 2 * t - 1
    # positions T-1 ... -(T-1): the same for every item, so projected once
    r = self.relative_k_proj(position_embeddings[:1]).view(1, p, h, d).permute(0, 2, 3, 1)
    bd = torch.matmul(qv, r)  # (b, h, t, p)
    # Transformer-XL shift: score[i, j] = bd[i, t - 1 - i + j], a strided view (rows do not overlap)
    bd = bd.as_strided((b, h, t, t), (h * t * p, t * p, p - 1, 1), bd.storage_offset() + t - 1)
    if attention_mask is not None:
        bd = bd.masked_fill(attention_mask.logical_not(), torch.finfo(bd.dtype).min)
    out = F.scaled_dot_product_attention(qu, k, v, attn_mask=bd, dropout_p=0.0, scale=self.scaling)
    return self.o_proj(out.transpose(1, 2).reshape(b, t, h * d)), None


def patch_encoder(encoder) -> int:
    """Replace the forward of every ParakeetEncoderAttention in the encoder; returns how many were patched."""
    n = 0
    for module in encoder.modules():
        if type(module).__name__ == "ParakeetEncoderAttention":
            module.forward = types.MethodType(fast_forward, module)
            n += 1
    return n
