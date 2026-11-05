"""The Parakeet (FastConformer) encoder of Cohere Transcribe on packed variable-length batches, through the CUDA
kernels in kernels/csrc.

transformers pads every piece of a batch to the longest and masks: the padded frames cost the same GEMM work as real
ones, the attention builds a (batch, heads, T, 2T-1) position-score tensor and makes several passes over it, and every
LayerNorm, activation, residual sum and transpose is a separate pass over memory. Here each piece goes through the
subsampling convolutions on its own, channels-last (no padding to mask), and the valid frames of all pieces are
concatenated for the 48 layers, each 14 kernels launched from C++ (encoder_loop.cpp):

    h = LN(x);  t = silu(h W1 + b1);  x += 0.5 (t W2 + b2)                       feed-forward 1: 2 GEMMs, fused epilogues
    h = LN(x);  qkv = h Wqkv + b;  a = relpos_attention(qkv);  x += a Wo + bo   attention: the position term from a
                                                                                per-layer table of projected distances
    h = LN(x);  g = glu(h W1c + b);  z = silu(bn(dwconv(g)));  x += z W2c + b   convolution module, per sequence
    h = LN(x);  t = silu(h W3 + b3);  x += 0.5 (t W4 + b4);  x = LN_out(x)      feed-forward 2, the output LayerNorm
                                                                                fused with the next layer's first one

In bf16 the arithmetic follows the module's roundings (each Linear's output, the GLU, the depthwise convolution, the
BatchNorm, the position scores); attention and GEMMs accumulate in fp32. With int8=True every GEMM is W8A8 (int8
weights per output channel; activations per row, quantized by the LayerNorm, attention-output and convolution kernels
that produce them, the feed-forwards' 5120-wide ones per row and 256 columns by the first GEMM's epilogue).

    enc = FastEncoder(model.model.encoder, int8=True)
    hidden, lengths = enc(features, frames)   # features (B, T, 128) padded, frames per piece -> (sum, 1280), lengths
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from kernels.cuda_ext import EPI_BIAS, EPI_CONV_RELU, EPI_GLU, ext

GEMM_CFG = 8  # 128 x 128 tiles, 4 warps, 3 stages: fastest on an A10 for the encoder's shapes
# int8 configs (kernels/csrc/gemm_int8.cu), fastest on an A10: per-row A scales with 64 x 64 warp tiles, the grouped
# K=5120 input with 64 x 32; the feed-forwards' first GEMM with 256-column tiles, so their int8 output (the second's
# input) has one scale per row and 256 columns
I8_CFG, I8_CFG_GROUPED, I8_CFG_FF1, FF_GROUP = 0, 3, 4, 256


def quantize_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int8 per output channel (row): (int8 weights, fp32 scales)."""
    s = w.float().abs().amax(-1).clamp_min(1e-12) / 127
    return torch.round(w.float() / s[:, None]).clamp(-127, 127).to(torch.int8).contiguous(), s.contiguous()


def gemm(a, w, b, epi=EPI_BIAS, out=None, resid=None, alpha=1.0, cfg=GEMM_CFG):
    n = w.shape[0] // 2 if epi == EPI_GLU else w.shape[0]
    if out is None:
        out = torch.empty(a.shape[0], n, device=a.device, dtype=a.dtype)
    ext.gemm(a, w, b, out, epi, resid, alpha, cfg)
    return out


class _Layer:
    def __init__(self, blk, rk, int8: bool = False):
        ln = lambda m: (m.weight, m.bias, m.eps)  # noqa: E731
        self.ln_ff1, self.ln_att, self.ln_conv = ln(blk.norm_feed_forward1), ln(blk.norm_self_att), ln(blk.norm_conv)
        self.ln_ff2, self.ln_out = ln(blk.norm_feed_forward2), ln(blk.norm_out)
        f1, f2 = blk.feed_forward1, blk.feed_forward2
        self.ff1 = (f1.linear1.weight, f1.linear1.bias, f1.linear2.weight, f1.linear2.bias)
        self.ff2 = (f2.linear1.weight, f2.linear1.bias, f2.linear2.weight, f2.linear2.bias)
        at = blk.self_attn
        self.wqkv = torch.cat([at.q_proj.weight, at.k_proj.weight, at.v_proj.weight]).contiguous()
        self.bqkv = torch.cat([at.q_proj.bias, at.k_proj.bias, at.v_proj.bias]).contiguous()
        self.wo, self.bo = at.o_proj.weight, at.o_proj.bias
        self.bu, self.bv = at.bias_u.contiguous(), at.bias_v.contiguous()
        self.rk = rk
        cv = blk.conv
        c = cv.depthwise_conv.in_channels
        w1, b1 = cv.pointwise_conv1.weight[:, :, 0], cv.pointwise_conv1.bias
        # GLU halves interleaved (a_0, g_0, a_1, g_1, ...) so one epilogue sees both of a channel's values
        self.wpw1 = torch.stack([w1[:c], w1[c:]], 1).reshape(2 * c, -1).contiguous()
        self.bpw1 = torch.stack([b1[:c], b1[c:]], 1).reshape(2 * c).contiguous()
        self.wdw = cv.depthwise_conv.weight[:, 0, :].float().contiguous()
        self.bdw = cv.depthwise_conv.bias.float().contiguous()
        bn = cv.norm
        scale = bn.weight.float() / torch.sqrt(bn.running_var.float() + bn.eps)
        self.bn_scale = scale.contiguous()
        self.bn_shift = (bn.bias.float() - bn.running_mean.float() * scale).contiguous()
        self.wpw2, self.bpw2 = cv.pointwise_conv2.weight[:, :, 0].contiguous(), cv.pointwise_conv2.bias
        if int8:  # every GEMM of the layer; the bf16 weights are released (1.8 GB of the 48 layers)
            self.q_ff1 = (quantize_rows(self.ff1[0]), quantize_rows(self.ff1[2]))
            self.q_ff2 = (quantize_rows(self.ff2[0]), quantize_rows(self.ff2[2]))
            self.q_qkv, self.q_pw1 = quantize_rows(self.wqkv), quantize_rows(self.wpw1)
            self.q_o, self.q_pw2 = quantize_rows(self.wo), quantize_rows(self.wpw2)
            gone = torch.empty(0, device=self.wo.device, dtype=self.wo.dtype)
            self.ff1 = (gone, self.ff1[1], gone, self.ff1[3])
            self.ff2 = (gone, self.ff2[1], gone, self.ff2[3])
            self.wqkv = self.wpw1 = self.wo = self.wpw2 = gone
            for lin in (f1.linear1, f1.linear2, f2.linear1, f2.linear2, at.q_proj, at.k_proj, at.v_proj, at.o_proj,
                        cv.pointwise_conv1, cv.pointwise_conv2):
                lin.weight.data = gone


    def fields(self, int8: bool) -> list[torch.Tensor]:
        """This layer's tensors in encoder_loop.cpp's order (enum F)."""
        out = [*self.ln_ff1[:2], *self.ln_att[:2], *self.ln_conv[:2], *self.ln_ff2[:2], *self.ln_out[:2],
               *self.ff1, *self.ff2, self.wqkv, self.bqkv, self.wo, self.bo, self.bu, self.bv, self.rk, self.wpw1,
               self.bpw1, self.wdw, self.bdw, self.bn_scale, self.bn_shift, self.wpw2, self.bpw2]
        if int8:
            for q in (*self.q_ff1, *self.q_ff2, self.q_qkv, self.q_pw1, self.q_o, self.q_pw2):
                out += list(q)
        return out


class FastEncoder:
    MAX_LEN = 512  # encoder frames (41 s of audio): the distance table covers -(MAX_LEN-1) .. MAX_LEN-1+64

    def __init__(self, encoder, int8: bool = False):
        """int8: W8A8 for every GEMM of the layers; weights per output channel, activations per row (the
        feed-forwards' second GEMM: per row and 256 columns)."""
        cfg = encoder.config
        self.int8 = int8
        assert cfg.hidden_size == 1280 and cfg.num_attention_heads == 8 and not cfg.scale_input
        self.encoder = encoder
        self.heads, self.head_dim = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
        self.scale = self.head_dim ** -0.5
        p = next(encoder.parameters())
        # sinusoids of relative distances d = i - j, interleaved (sin, cos) as ParakeetEncoderRelPositionalEncoding
        self.rel_off = self.MAX_LEN - 1
        d = torch.arange(-(self.MAX_LEN - 1), self.MAX_LEN + 64, device=p.device, dtype=torch.float)
        freqs = d[:, None] * encoder.encode_positions.inv_freq.to(p.device, torch.float)
        pos = torch.stack([freqs.sin(), freqs.cos()], -1).reshape(len(d), -1).to(p.dtype)
        with torch.no_grad():
            self.layers = [_Layer(blk, F.linear(pos, blk.self_attn.relative_k_proj.weight).contiguous(), int8)
                           for blk in encoder.layers]
            # subsampling: [conv0, ReLU, dw, pw, ReLU, dw, pw, ReLU] + linear; the linear's input columns reordered
            # from (channel, frequency) to the channels-last (frequency, channel)
            sub = encoder.subsampling
            conv0, dw2, pw3, dw5, pw6 = (sub.layers[i] for i in (0, 2, 3, 5, 6))
            f32 = lambda t: t.float().reshape(t.shape[0], -1).contiguous()  # noqa: E731
            self.sub_c0 = (f32(conv0.weight), conv0.bias.float().contiguous(), f32(dw2.weight), dw2.bias.float().contiguous())
            self.sub_pw3 = (pw3.weight[:, :, 0, 0].contiguous(), pw3.bias)
            self.sub_dw5 = (f32(dw5.weight), dw5.bias.float().contiguous())
            self.sub_pw6 = (pw6.weight[:, :, 0, 0].contiguous(), pw6.bias)
            ch, out_f = pw6.out_channels, sub.linear.in_features // pw6.out_channels
            self.sub_lin = (sub.linear.weight.view(-1, ch, out_f).transpose(1, 2).reshape(-1, ch * out_f).contiguous(),
                            sub.linear.bias)
        # per layer, the tensors in encoder_loop.cpp's field order, and the LayerNorms' epsilons
        self.layer_tensors = [ly.fields(int8) for ly in self.layers]
        self.layer_eps = [e for ly in self.layers for e in (ly.ln_ff1[2], ly.ln_att[2], ly.ln_conv[2], ly.ln_ff2[2],
                                                            ly.ln_out[2])]

    @torch.no_grad()
    def subsample(self, features: torch.Tensor, frames: list[int]) -> tuple[torch.Tensor, list[int]]:
        """Each piece's first frames[i] log-mel frames of the padded batch features (B, T, 128) through the
        subsampling convolutions, on its own (no padding), channels-last: packed (N, 1280) and the lengths."""
        dev = features.device
        half = lambda n: [(t - 1) // 2 + 1 for t in n]  # noqa: E731  3x3, stride 2, padding 1
        t2 = half(half(frames))
        t3 = half(t2)
        rows = lambda ts: torch.tensor([(i, t) for i, n in enumerate(ts) for t in range(n)],  # noqa: E731
                                       dtype=torch.int32).to(dev, non_blocking=True)
        i32 = lambda v: torch.tensor(v, dtype=torch.int32).to(dev, non_blocking=True)  # noqa: E731
        n2, n3 = sum(t2), sum(t3)
        a2 = torch.empty(n2, 32, 256, device=dev, dtype=features.dtype)
        ext.sub_conv0_dw(features.contiguous(), i32(frames), rows(t2), *self.sub_c0, a2)
        a3 = gemm(a2.view(n2 * 32, 256), *self.sub_pw3, EPI_CONV_RELU)
        a5 = torch.empty(n3, 16, 256, device=dev, dtype=features.dtype)
        ext.sub_dw(a3.view(n2, 32, 256), i32([sum(t2[:i]) for i in range(len(t2))]), i32(t2), rows(t3), *self.sub_dw5, a5)
        a6 = gemm(a5.view(n3 * 16, 256), *self.sub_pw6, EPI_CONV_RELU)
        return gemm(a6.view(n3, 4096), *self.sub_lin, EPI_BIAS), t3

    def metadata(self, lengths: list[int], device) -> dict:
        starts, s = [], 0
        for n in lengths:
            starts.append(s)
            s += n
        tiles = [(st, n, q0, 0) for st, n in zip(starts, lengths) for q0 in range(0, n, 64)]
        lo = torch.tensor([st for st, n in zip(starts, lengths) for _ in range(n)], dtype=torch.int32)
        hi = torch.tensor([st + n for st, n in zip(starts, lengths) for _ in range(n)], dtype=torch.int32)
        return {"tiles": torch.tensor(tiles, dtype=torch.int32).to(device, non_blocking=True),
                "lo": lo.to(device, non_blocking=True), "hi": hi.to(device, non_blocking=True),
                "max_len": max(lengths)}

    @torch.no_grad()
    def layers_forward(self, x: torch.Tensor, meta: dict) -> torch.Tensor:
        """The 48 layers on packed frames x (N, 1280), in place (kernels/csrc/encoder_loop.cpp); returns x."""
        assert meta["max_len"] <= self.MAX_LEN
        ext.encoder_layers(x, self.layer_tensors, self.layer_eps, meta["tiles"], meta["lo"], meta["hi"], self.rel_off,
                           meta["max_len"], self.scale, self.int8,
                           [GEMM_CFG, I8_CFG, I8_CFG_GROUPED, I8_CFG_FF1, FF_GROUP])
        return x

    @torch.no_grad()
    def __call__(self, features: torch.Tensor, frames: list[int]) -> tuple[torch.Tensor, list[int]]:
        """Packed hidden states (sum of lengths, 1280) and the lengths of the pieces whose first frames[i] frames are
        in the padded batch features (B, T, 128)."""
        x, lengths = self.subsample(features, frames)
        return self.layers_forward(x, self.metadata(lengths, x.device)), lengths
