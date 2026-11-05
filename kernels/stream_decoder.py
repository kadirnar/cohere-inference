"""Greedy decoding of Cohere Transcribe with continuous batching: rows join and leave between steps.

The decoder state lives in SLOTS slots: per-slot KV caches (self-attention positions and cross-attention audio frames),
the token being fed, its position, the forced prompt, the token limit and the end flag. A step feeds one token for
every row of the active list (row r -> slot active[r]); one CUDA graph per bucket of row counts, so a step is one graph
launch whatever positions and audio lengths its rows have. Rows past the real ones point to a dummy slot that never
advances. The host admits a piece by filling a free slot (its cross-attention keys and values are written from the
encoder's packed output by dec_scatter_kv) and putting it in the active list; the prompt is fed by the steps
themselves (the forced tokens), so new rows ride along with running ones; a finished slot is copied out and dropped
from the list. Kernels: kernels/csrc/decoder_ops.cu (embedding, residual + LayerNorm, flash-decoding attention, greedy
choice) and the projections (Triton for up to 32 rows, cuBLAS above).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from kernels.cuda_ext import ext
from kernels.decoder_kernels import linear as tri_linear


def linear(x, w, b, relu=False):
    if x.shape[0] <= 32:
        return tri_linear(x, w, b, relu)
    y = F.linear(x, w, b)
    return F.relu(y) if relu else y


W8_CONFIGS = {  # (N, K) -> {rows up to: (columns per CTA, K splits)}, fastest on an A10 in CUDA graphs, L2 cold
    (1024, 4096): {1: (64, 4), 8: (16, 4), 16: (16, 4), 24: (16, 2), 32: (32, 4), 48: (64, 4), 64: (64, 4)},
    (3072, 1024): {1: (32, 1), 8: (32, 1), 16: (32, 1), 24: (32, 1), 32: (32, 1), 48: (64, 1), 64: (64, 1)},
    (4096, 1024): {1: (32, 1), 8: (32, 1), 16: (32, 1), 24: (64, 1), 32: (64, 1), 48: (64, 1), 64: (64, 1)},
    (16384, 1024): {1: (16, 1), 8: (16, 1), 16: (32, 1), 24: (64, 1), 32: (64, 1), 48: (64, 1), 64: (64, 1)},
}


W8_PERM = (0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15)


def prepare_w8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """int8 weights per output channel in dec_w8_linear's layout: (uint8 = int8 + 128, K reordered within every 16
    so an ldmatrix row is an mma B fragment), fp32 scales."""
    n, k = w.shape
    s = w.float().abs().amax(1).clamp_min(1e-12) / 127
    q = torch.round(w.float() / s[:, None]).clamp(-127, 127) + 128
    perm = torch.tensor(W8_PERM, device=w.device)
    return q.to(torch.uint8).view(n, k // 16, 16)[:, :, perm].reshape(n, k).contiguous(), s.contiguous()


class _Layer:
    def __init__(self, layer):
        sa, ca = layer.self_attn, layer.encoder_attn
        ln = lambda m: (m.weight, m.bias, m.eps)  # noqa: E731
        self.ln1, self.ln2, self.ln3 = ln(layer.input_layernorm), ln(layer.post_attention_layernorm), ln(layer.final_layernorm)
        self.qkv = (torch.cat([sa.q_proj.weight, sa.k_proj.weight, sa.v_proj.weight]).contiguous(),
                    torch.cat([sa.q_proj.bias, sa.k_proj.bias, sa.v_proj.bias]).contiguous())
        self.o = (sa.o_proj.weight, sa.o_proj.bias)
        self.cq = (ca.q_proj.weight, ca.q_proj.bias)
        self.co = (ca.o_proj.weight, ca.o_proj.bias)
        self.fc1 = (layer.mlp.fc1.weight, layer.mlp.fc1.bias)
        self.fc2 = (layer.mlp.fc2.weight, layer.mlp.fc2.bias)


class StreamDecoder:
    BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128)

    def __init__(self, model, slots: int = 128, max_new_tokens: int = 448, max_prompt: int = 16,
                 max_audio_s: float = 36.0, buckets=None, int8: bool = False):
        """int8: the self-attention q/k/v, feed-forward and output-head projections with int8 weights (W8A16,
        dec_w8_linear): half the bytes a step reads for them; the three 1024 x 1024 ones (o and cross-attention q, o)
        stay bf16, where the int8 kernel is no faster. The choice is per projection, not per batch size."""
        dec, cfg = model.model.decoder, model.config
        self.buckets = tuple(b for b in (buckets or self.BUCKETS) if b <= slots)
        self.model, self.max_new_tokens = model, max_new_tokens
        self.h, self.d, self.width = cfg.num_attention_heads, cfg.head_dim, cfg.hidden_size
        assert self.d == 128 and self.width == 1024
        self.eos, self.pad = cfg.eos_token_id, cfg.pad_token_id
        self.scale = self.d ** -0.5
        self.embed, self.pos_table = dec.embed_tokens.weight, dec.pos_emb.weight
        ln = lambda m: (m.weight, m.bias, m.eps)  # noqa: E731
        self.emb_ln, self.final_ln = ln(dec.embedding_layernorm), ln(dec.norm)
        self.head = (model.proj_out.weight, model.proj_out.bias)
        self.proj = (dec.proj.weight, dec.proj.bias)
        self.layers = [_Layer(layer) for layer in dec.layers]
        self.int8 = int8
        if int8:
            for ly in self.layers:
                ly.qkv_w8, ly.fc1_w8, ly.fc2_w8 = prepare_w8(ly.qkv[0]), prepare_w8(ly.fc1[0]), prepare_w8(ly.fc2[0])
            self.head_w8 = prepare_w8(self.head[0])
        self.cross_w = torch.cat([w for layer in dec.layers for w in (layer.encoder_attn.k_proj.weight,
                                                                       layer.encoder_attn.v_proj.weight)]).contiguous()
        self.cross_b = torch.cat([b for layer in dec.layers for b in (layer.encoder_attn.k_proj.bias,
                                                                       layer.encoder_attn.v_proj.bias)]).contiguous()
        L, S = len(self.layers), slots
        self.slots, self.dummy = S, S
        self.max_prompt = max_prompt
        self.seq = max_prompt + max_new_tokens + 1
        self.frames = model.model.encoder._get_subsampling_output_length(torch.tensor([int(max_audio_s * 100) + 1])).item()
        dev, dt = self.embed.device, self.embed.dtype
        self.device = dev
        self.self_k = torch.zeros(L, S + 1, self.h, self.seq, self.d, device=dev, dtype=torch.int8 if int8 else dt)
        self.self_v = torch.zeros_like(self.self_k)
        if int8:
            self.self_ks = torch.ones(L, S + 1, self.h, self.seq, device=dev, dtype=torch.float32)
            self.self_vs = torch.ones_like(self.self_ks)
        # int8: the self- and cross-attention keys and values in int8 with one fp32 scale per 128-vector (half the
        # bytes the attention reads per step, the largest part of a step's traffic beside the weights)
        self.cross_k = torch.zeros(L, S + 1, self.h, self.frames, self.d, device=dev,
                                   dtype=torch.int8 if int8 else dt)
        self.cross_v = torch.zeros_like(self.cross_k)
        if int8:
            self.cross_ks = torch.ones(L, S + 1, self.h, self.frames, device=dev, dtype=torch.float32)
            self.cross_vs = torch.ones_like(self.cross_ks)
        i32 = dict(device=dev, dtype=torch.int32)
        self.cross_len = torch.ones(S + 1, **i32)
        self.tok = torch.full((S + 1,), self.pad, device=dev, dtype=torch.long)
        self.pos = torch.zeros(S + 1, **i32)
        self.plen = torch.ones(S + 1, **i32)
        self.limit = torch.zeros(S + 1, **i32)
        self.finished = torch.ones(S + 1, device=dev, dtype=torch.uint8)
        self.forced = torch.full((S + 1, max_prompt), self.pad, device=dev, dtype=torch.long)
        self.out = torch.full((S + 1, self.seq + 1), self.pad, device=dev, dtype=torch.long)
        self.active = torch.full((max(self.buckets),), S, **i32)
        bm = max(self.buckets)
        self.x = torch.zeros(bm, self.width, device=dev, dtype=dt)
        self.hn = torch.zeros(bm, self.width, device=dev, dtype=dt)
        self.att = torch.zeros(bm, self.width, device=dev, dtype=dt)
        self.part = torch.empty(max(b * self.h * self._splits(b) * (self.d + 2) for b in self.buckets), device=dev,
                                dtype=torch.float32)
        self.w8_ws = torch.empty(8 << 20, device=dev, dtype=torch.float32)  # split-K partials of dec_w8_linear
        self.w8_counters = torch.zeros(1024, device=dev, dtype=torch.int32)  # per column tile, reset by the kernel
        # host side: the end flags after each step (pinned, copied by the step itself)
        self.flags = torch.ones(S + 1, dtype=torch.uint8).pin_memory()
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        # high priority: while the encoder of the next pieces runs beside it, the steps' blocks are scheduled first
        self.stream = torch.cuda.Stream(priority=-1)
        self._capture()

    @staticmethod
    def _splits(b: int) -> int:
        """Attention programs per row and head: one, except the cross-attention of one or two rows (A10, CUDA
        graphs: more splits only add the combine kernel)."""
        return 4 if b <= 2 else 1

    # ---- the step ----------------------------------------------------------------------------------------------

    def _w8(self, x, wq, bias, relu=False):
        """x @ dequant(wq)^T + bias through dec_w8_linear, 64 rows at a time."""
        (w, scale), (n, k) = wq, wq[0].shape
        y = torch.empty(x.shape[0], n, device=x.device, dtype=x.dtype)
        configs = W8_CONFIGS[(n, k)]
        for r in range(0, x.shape[0], 64):
            m = min(64, x.shape[0] - r)
            bn, splits = configs[next(c for c in sorted(configs) if c >= m)]
            ext.dec_w8_linear(x[r:r + m], w, scale, bias, y[r:r + m], relu, bn, splits, self.w8_ws,
                              self.w8_counters)
        return y

    @staticmethod
    def _proj(x, ly, name):
        """The o, cross q or cross o projection: bf16 weights (the int8 kernel is no faster at 1024 x 1024)."""
        return linear(x, *getattr(ly, name))

    def _qkv(self, hn, ly):
        return self._w8(hn, ly.qkv_w8, ly.qkv[1]) if self.int8 else linear(hn, *ly.qkv)

    def _ffn(self, hn, ly):
        if self.int8:
            return self._w8(self._w8(hn, ly.fc1_w8, ly.fc1[1], relu=True), ly.fc2_w8, ly.fc2[1])
        return linear(linear(hn, *ly.fc1, relu=True), *ly.fc2)

    def _head(self, hn):
        return self._w8(hn, self.head_w8, self.head[1]) if self.int8 else linear(hn, *self.head)

    def _step(self, b: int) -> None:
        act, x, hn, a, splits = self.active[:b], self.x[:b], self.hn[:b], self.att[:b], self._splits(b)
        ext.dec_embed(act, self.tok, self.pos, self.embed, self.pos_table, *self.emb_ln, *self.layers[0].ln1, x, hn)
        for i, ly in enumerate(self.layers):
            nxt = self.layers[i + 1] if i + 1 < len(self.layers) else None
            qkv = self._qkv(hn, ly)
            if self.int8:
                ext.dec_attention_q8(True, act, qkv, self.self_k[i], self.self_v[i], self.self_ks[i], self.self_vs[i],
                                     self.pos, self.scale, 1, self.part, a)
            else:
                ext.dec_attention(True, act, qkv, self.self_k[i], self.self_v[i], self.pos, self.scale, 1, self.part, a)
            ext.dec_add_ln(x, self._proj(a, ly, "o"), *ly.ln2, hn)
            if self.int8:
                ext.dec_attention_q8(False, act, self._proj(hn, ly, "cq"), self.cross_k[i], self.cross_v[i],
                                     self.cross_ks[i], self.cross_vs[i], self.cross_len, self.scale, splits, self.part, a)
            else:
                ext.dec_attention(False, act, linear(hn, *ly.cq), self.cross_k[i], self.cross_v[i], self.cross_len,
                                  self.scale, splits, self.part, a)
            ext.dec_add_ln(x, self._proj(a, ly, "co"), *ly.ln3, hn)
            following = nxt.ln1 if nxt else self.final_ln
            ext.dec_add_ln(x, self._ffn(hn, ly), *following, hn)
        ext.dec_pick(self._head(hn), act, self.dummy, self.tok, self.pos, self.finished, self.forced, self.plen,
                     self.limit, self.out, self.pad, self.eos)
        self.flags.copy_(self.finished, non_blocking=True)

    @torch.no_grad()
    def _capture(self) -> None:
        pool = torch.cuda.graph_pool_handle()
        with torch.cuda.stream(self.stream):
            for b in reversed(self.buckets):  # largest first: the smaller graphs fit in the pool it leaves
                for _ in range(2):  # compile the Triton kernels, set up cuBLAS
                    self._step(b)
                torch.cuda.current_stream().synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool, stream=self.stream):
                    self._step(b)
                self.graphs[b] = g
            self.reset()
        self.stream.synchronize()

    def reset(self) -> None:
        """All slots free (on the decoder stream)."""
        with torch.cuda.stream(self.stream):
            self.finished.fill_(1)
            self.pos.zero_()
            self.cross_len.fill_(1)
            self.active.fill_(self.dummy)

    # ---- host side -----------------------------------------------------------------------------------------------

    def bucket(self, rows: int) -> int:
        return next(b for b in self.buckets if b >= rows)

    def fill_cross(self, hidden: torch.Tensor, lengths: list[int], slots: list[int]) -> None:
        """Cross-attention keys and values of packed encoder frames (sum(lengths), 1280) into the given slots, on the
        current stream (the caller orders the decoder stream after it)."""
        kv = F.linear(F.linear(hidden, *self.proj), self.cross_w, self.cross_b)
        slot_of = torch.repeat_interleave(torch.tensor(slots, dtype=torch.int32), torch.tensor(lengths))
        frame_of = torch.cat([torch.arange(n, dtype=torch.int32) for n in lengths])
        slot_of, frame_of = slot_of.to(self.device, non_blocking=True), frame_of.to(self.device, non_blocking=True)
        if self.int8:
            ext.dec_scatter_kv_q8(kv, slot_of, frame_of, self.cross_k, self.cross_v, self.cross_ks, self.cross_vs)
        else:
            ext.dec_scatter_kv(kv, slot_of, frame_of, self.cross_k, self.cross_v)

    def admit(self, slots: list[int], prompts: list[torch.Tensor], frames: list[int],
              max_new: list[int] | None = None) -> None:
        """Start decoding in the given slots (on the decoder stream): the prompt forced, cross_len frames of audio."""
        n = len(slots)
        idx = torch.tensor(slots, dtype=torch.long)
        forced = torch.full((n, self.max_prompt), self.pad, dtype=torch.long)
        plen = torch.tensor([len(p) for p in prompts], dtype=torch.int32)
        for i, p in enumerate(prompts):
            forced[i, :len(p)] = p
        limit = plen - 1 + torch.tensor(max_new or [self.max_new_tokens] * n, dtype=torch.int32)
        with torch.cuda.stream(self.stream):
            d = lambda t: t.to(self.device, non_blocking=True)  # noqa: E731
            di = d(idx)
            self.forced[di] = d(forced)
            self.out[di] = self.pad
            self.out[di, :self.max_prompt] = d(forced)
            self.tok[di] = d(forced[:, 0])
            self.pos[di] = 0
            self.plen[di] = d(plen)
            self.limit[di] = d(limit)
            self.cross_len[di] = d(torch.tensor(frames, dtype=torch.int32))
            self.finished[di] = 0
        self.flags[idx] = 0

    def set_active(self, slots: list[int]) -> int:
        """The active list for the next steps (on the decoder stream); returns the bucket."""
        b = self.bucket(max(1, len(slots)))
        host = torch.full((b,), self.dummy, dtype=torch.int32)
        host[:len(slots)] = torch.tensor(slots, dtype=torch.int32)
        with torch.cuda.stream(self.stream):
            self.active[:b].copy_(host.pin_memory(), non_blocking=True)
        return b

    def step(self, bucket: int) -> None:
        with torch.cuda.stream(self.stream):
            self.graphs[bucket].replay()
