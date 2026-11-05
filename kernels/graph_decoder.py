"""Greedy decoding of Cohere Transcribe with the decoder step replayed as a CUDA graph.

transformers' generate runs the decoder step by step from Python: each of the 8 layers launches its kernels one at a
time, the encoder states are projected again on every step, the masks are rebuilt and the KV cache grows by
concatenation. Here the step (one token for every row of the batch) works on buffers allocated once - the self-attention
KV cache, the cross-attention keys and values of the batch's audio, the residual stream, the tokens and the position -
and is captured once per batch size as a CUDA graph, so a step is one graph launch. Its kernels are mostly Triton
(kernels/decoder_kernels.py): projections with the bias fused, the residual sum fused into the next LayerNorm, the new
key and value written to the cache by the attention kernel, attention over the valid part of the caches only (the
current position for self-attention, each row's audio frames for cross-attention, read from device memory, so one
graph serves every position and audio length), and the greedy choice with the prompt forcing and the end-of-text
bookkeeping in one kernel. The prompt (4 tokens with language=auto, 10 with a language) is fed in one pass, as
generate does, captured as its own graph per batch size and prompt length. The host runs LAG steps ahead of the
end-of-text check, which each step copies to pinned memory, so the GPU never waits for it and at most LAG + 1 steps run
past the end. The arithmetic is the model's (bf16 weights and activations, fp32 accumulation and softmax), tokens the
argmax of the logits as in generate's greedy search; outputs match transformers' up to rounding order, like a different
batch size does.

    decoder = GraphDecoder(model)                     # captures the graphs (well under a second)
    tokens = decoder.generate(input_features, attention_mask, prompt_ids)   # (rows, prompt + generated)
"""

from __future__ import annotations

import threading

import torch
import torch.nn.functional as F

from kernels.decoder_kernels import add_layernorm, decode_attention, embed_layernorm, linear, pick, self_attention


class _Layer:
    """One decoder layer's weights, with q, k and v of self-attention stacked into one projection."""

    def __init__(self, layer):
        sa, ca = layer.self_attn, layer.encoder_attn
        ln = lambda m: (m.weight, m.bias, m.eps)  # noqa: E731
        self.ln1, self.ln2 = ln(layer.input_layernorm), ln(layer.post_attention_layernorm)
        self.ln3 = ln(layer.final_layernorm)
        self.qkv = (torch.cat([sa.q_proj.weight, sa.k_proj.weight, sa.v_proj.weight]).contiguous(),
                    torch.cat([sa.q_proj.bias, sa.k_proj.bias, sa.v_proj.bias]).contiguous())
        self.o = (sa.o_proj.weight, sa.o_proj.bias)
        self.cq = (ca.q_proj.weight, ca.q_proj.bias)
        self.co = (ca.o_proj.weight, ca.o_proj.bias)
        self.fc1 = (layer.mlp.fc1.weight, layer.mlp.fc1.bias)
        self.fc2 = (layer.mlp.fc2.weight, layer.mlp.fc2.bias)


class GraphDecoder:
    """Greedy decoding for CohereAsrForConditionalGeneration (bf16, CUDA) with one captured step per batch size in
    BUCKETS; a batch runs on the smallest bucket that holds it, the extra rows idle (finished from the start)."""

    BUCKETS = (1, 2, 4, 8, 16, 32)
    ENCODE_ROWS = 4  # rows per encoder call: the encoder's CUDA graphs, and activations bounded for big batches
    SPLITS = {1: 16, 2: 8, 4: 4, 8: 2, 16: 1, 32: 1}  # attention programs per row and head (A10 measurements)

    LAG = 1  # steps the host launches beyond the last one whose end-of-text check it has read

    def __init__(self, model, encode=None, max_new_tokens: int = 448, prompt_lengths: tuple[int, ...] = (4, 10),
                 max_prompt: int = 16, max_audio_s: float = 36.0):
        """encode(features, mask) -> (hidden states, valid frames): the encoder, e.g. a GraphEncoder; by default
        the model's own. Prompts of other lengths than prompt_lengths go through the step one token at a time."""
        dec, cfg = model.model.decoder, model.config
        self.model, self.max_new_tokens, self.prompt_lengths = model, max_new_tokens, prompt_lengths
        self.encode = encode or self._encode
        self.h, self.d, self.width = cfg.num_attention_heads, cfg.head_dim, cfg.hidden_size
        self.eos, self.pad = cfg.eos_token_id, cfg.pad_token_id
        self.scale = self.d ** -0.5
        self.embed, self.pos_table = dec.embed_tokens.weight, dec.pos_emb.weight
        ln = lambda m: (m.weight, m.bias, m.eps)  # noqa: E731
        self.emb_ln, self.final_ln = ln(dec.embedding_layernorm), ln(dec.norm)
        self.head = (model.proj_out.weight, model.proj_out.bias)
        self.proj = (dec.proj.weight, dec.proj.bias)
        self.layers = [_Layer(layer) for layer in dec.layers]
        # cross-attention keys and values of every layer in one projection of the encoder states
        self.cross_w = torch.cat([w for layer in dec.layers for w in (layer.encoder_attn.k_proj.weight,
                                                                       layer.encoder_attn.v_proj.weight)]).contiguous()
        self.cross_b = torch.cat([b for layer in dec.layers for b in (layer.encoder_attn.k_proj.bias,
                                                                       layer.encoder_attn.v_proj.bias)]).contiguous()
        n_layers, bm = len(self.layers), max(self.BUCKETS)
        self.max_batch = bm
        self.seq = max_prompt + max_new_tokens + 1
        self.max_features = int(max_audio_s * 100) + 1  # log-mel frames of the longest audio
        self.frames = model.model.encoder._get_subsampling_output_length(torch.tensor([self.max_features])).item()
        dev, dt = self.embed.device, self.embed.dtype
        self.self_k = torch.zeros(n_layers, bm, self.h, self.seq, self.d, device=dev, dtype=dt)
        self.self_v = torch.zeros_like(self.self_k)
        self.cross_k = torch.zeros(n_layers, bm, self.h, self.frames, self.d, device=dev, dtype=dt)
        self.cross_v = torch.zeros_like(self.cross_k)
        self.cross_len = torch.ones(bm, device=dev, dtype=torch.int32)
        self.x = torch.zeros(bm, self.width, device=dev, dtype=dt)  # residual stream
        self.hn = torch.zeros(bm, self.width, device=dev, dtype=dt)  # its LayerNorm, input of the next projection
        self.tokens = torch.zeros(bm, device=dev, dtype=torch.long)
        self.forced = torch.zeros(bm, self.seq + 1, device=dev, dtype=torch.long)
        self.out = torch.full((bm, self.seq + 1), self.pad, device=dev, dtype=torch.long)
        self.finished = torch.zeros(bm, device=dev, dtype=torch.uint8)
        self.pos = torch.zeros(1, device=dev, dtype=torch.long)  # position of the token being fed
        self.prompt_len = torch.zeros(1, device=dev, dtype=torch.long)
        self.all_done = torch.zeros(1, device=dev, dtype=torch.uint8)
        self.flag = torch.zeros(1, dtype=torch.uint8).pin_memory()  # all_done after the latest finished step
        self.scratch = torch.empty(max(b * self.h * s * (self.d + 2) for b, s in self.SPLITS.items()), device=dev,
                                   dtype=torch.float32)
        self.lock = threading.Lock()
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.prefills: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self._capture()

    def _encode(self, features, mask):
        out = self.model.model.encoder(features, attention_mask=mask)
        return out.last_hidden_state, out.attention_mask

    # ---- the step ----------------------------------------------------------------------------------------------

    def _step(self, b: int) -> None:
        """Feed tokens[:b] at position pos; write the next tokens (the prompt's while inside it)."""
        h, d, splits = self.h, self.d, self.SPLITS[b]
        x, hn = self.x[:b], self.hn[:b]
        embed_layernorm(self.tokens[:b], self.pos, self.embed, self.pos_table, self.emb_ln, self.layers[0].ln1, x, hn)
        for i, ly in enumerate(self.layers):
            a = self_attention(linear(hn, *ly.qkv), self.self_k[i, :b], self.self_v[i, :b], self.pos, h, self.scale,
                               splits, self.scratch)
            add_layernorm(x, linear(a, *ly.o), *ly.ln2, hn)
            a = decode_attention(linear(hn, *ly.cq).view(b, h, d), self.cross_k[i, :b], self.cross_v[i, :b],
                                 self.cross_len[:b], self.scale, splits, self.scratch)
            add_layernorm(x, linear(a.view(b, h * d), *ly.co), *ly.ln3, hn)
            following = self.layers[i + 1].ln1 if i + 1 < len(self.layers) else self.final_ln
            add_layernorm(x, linear(linear(hn, *ly.fc1, relu=True), *ly.fc2), *following, hn)
        pick(linear(hn, *self.head), self.forced[:b], self.out[:b], self.finished[:b], self.tokens[:b], self.pos,
             self.prompt_len, self.pad, self.eos)
        self.pos.add_(1)
        self._report(b)

    def _report(self, b: int) -> None:
        """Whether every row has ended, to the host's pinned flag."""
        torch.amin(self.finished[:b], 0, keepdim=True, out=self.all_done)
        self.flag.copy_(self.all_done, non_blocking=True)

    def _prefill(self, b: int, p: int) -> None:
        """Feed the whole prompt (forced[:, :p], the same in every row) in one pass, with transformers' operations:
        the KV cache of positions 0..p-1 and the first generated token, as p steps would give."""
        h, d, w = self.h, self.d, self.width
        x = F.layer_norm(F.embedding(self.forced[:b, :p], self.embed) + self.pos_table[:p], (w,), *self.emb_ln)
        audio = (torch.arange(self.frames, device=x.device) < self.cross_len[:b, None])[:, None, None, :]
        for i, ly in enumerate(self.layers):
            q, k, v = F.linear(F.layer_norm(x, (w,), *ly.ln1), *ly.qkv).view(b, p, 3, h, d).permute(2, 0, 3, 1, 4)
            self.self_k[i, :b, :, :p] = k
            self.self_v[i, :b, :, :p] = v
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
            x = x + F.linear(a.transpose(1, 2).reshape(b, p, w), *ly.o)
            q = F.linear(F.layer_norm(x, (w,), *ly.ln2), *ly.cq).view(b, p, h, d).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, self.cross_k[i, :b], self.cross_v[i, :b], attn_mask=audio,
                                               scale=self.scale)
            x = x + F.linear(a.transpose(1, 2).reshape(b, p, w), *ly.co)
            x = x + F.linear(F.relu(F.linear(F.layer_norm(x, (w,), *ly.ln3), *ly.fc1)), *ly.fc2)
        self.pos.fill_(p - 1)
        pick(linear(F.layer_norm(x[:, -1], (w,), *self.final_ln), *self.head), self.forced[:b], self.out[:b],
             self.finished[:b], self.tokens[:b], self.pos, self.prompt_len, self.pad, self.eos)
        self.pos.fill_(p)
        self._report(b)

    @torch.no_grad()
    def _capture(self) -> None:
        pool = torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream()
        idle = torch.tensor([self.pad], device=self.tokens.device)
        for b in reversed(self.BUCKETS):  # largest first: the smaller graphs then fit in the pool it leaves
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):  # compile the Triton kernels and set up cuBLAS before capturing
                for _ in range(2):
                    self._reset(b, b, idle)
                    self._step(b)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            self._reset(b, b, idle)
            with torch.cuda.graph(g, pool=pool):
                self._step(b)
            self.graphs[b] = g
            for p in self.prompt_lengths:
                prompt = idle.repeat(p)
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(2):
                        self._reset(b, b, prompt)
                        self._prefill(b, p)
                torch.cuda.current_stream().wait_stream(side)
                g = torch.cuda.CUDAGraph()
                self._reset(b, b, prompt)
                with torch.cuda.graph(g, pool=pool):
                    self._prefill(b, p)
                self.prefills[(b, p)] = g
        torch.cuda.synchronize()

    # ---- a batch -----------------------------------------------------------------------------------------------

    def _reset(self, rows: int, bucket: int, prompt: torch.Tensor) -> None:
        p = len(prompt)
        self.tokens[:bucket] = prompt[0]
        self.forced[:bucket, :p] = prompt
        self.out[:bucket].fill_(self.pad)
        self.out[:bucket, :p] = prompt
        self.prompt_len.fill_(p)
        self.pos.zero_()
        self.finished[:bucket] = 0
        self.finished[rows:bucket] = 1
        self.cross_len[rows:bucket] = 1
        self.flag[0] = 0  # no step of an earlier batch is still running: generate waits for its last one

    def _cross(self, r: int, features: torch.Tensor, mask: torch.Tensor, length: int) -> None:
        """Encode rows r.. (trimmed to their longest) and store their cross-attention keys and values."""
        hidden, valid = self.encode(features[:, :length], mask[:, :length])
        n, frames = hidden.shape[:2]
        kv = F.linear(F.linear(hidden, *self.proj), self.cross_w, self.cross_b)
        kv = kv.view(n, frames, len(self.layers), 2, self.h, self.d).permute(2, 3, 0, 4, 1, 5)
        self.cross_k[:, r:r + n, :, :frames].copy_(kv[:, 0])
        self.cross_v[:, r:r + n, :, :frames].copy_(kv[:, 1])
        self.cross_len[r:r + n].copy_(valid.sum(-1))

    def fits(self, rows: int, features: int) -> bool:
        """Whether a batch of rows with up to this many log-mel frames fits the buffers."""
        return rows <= self.max_batch and features <= self.max_features

    @torch.no_grad()
    def generate(self, input_features: torch.Tensor, attention_mask: torch.Tensor,
                 prompt: torch.Tensor) -> torch.Tensor:
        """Token ids (rows, prompt + generated; pad after each row's end) of greedy decoding with max_new_tokens,
        for input_features (rows, frames, 128) bf16 and attention_mask (rows, frames) on the GPU, prompt (P,)."""
        rows = len(input_features)
        bucket = next(b for b in self.BUCKETS if b >= rows)
        prompt = prompt.to(self.tokens.device)
        lengths = attention_mask.sum(-1).tolist()
        assert rows <= self.max_batch and max(lengths) <= self.max_features, (rows, max(lengths))
        with self.lock:  # the encoder's graph buffers too are only valid until the next call
            for r in range(0, rows, self.ENCODE_ROWS):
                self._cross(r, input_features[r:r + self.ENCODE_ROWS], attention_mask[r:r + self.ENCODE_ROWS],
                            max(lengths[r:r + self.ENCODE_ROWS]))
            self._reset(rows, bucket, prompt)
            g, p = self.graphs[bucket], len(prompt)
            ring = [torch.cuda.Event() for _ in range(self.LAG + 1)]
            if (bucket, p) in self.prefills:
                self.prefills[(bucket, p)].replay()
            else:
                for _ in range(p):
                    g.replay()
            ring[0].record()
            done, launched = p, 1  # steps done (tokens written up to out[:, done]); launches recorded in ring
            while done < p - 1 + self.max_new_tokens:  # the last step writes token max_new_tokens
                if launched > self.LAG:  # read the check of the launch LAG before the newest
                    ring[(launched - 1 - self.LAG) % len(ring)].synchronize()
                    if self.flag[0]:
                        break
                g.replay()
                ring[launched % len(ring)].record()
                done, launched = done + 1, launched + 1
            torch.cuda.current_stream().synchronize()  # nothing of this batch may write the flag after it returns
            return self.out[:rows, : done + 1].clone()
