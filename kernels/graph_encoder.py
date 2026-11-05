"""The Parakeet encoder of Cohere Transcribe replayed as CUDA graphs for small batches.

At batch 1 the encoder is held back by the CPU, not the GPU: its 48 layers launch about 4,200 kernels, some 40-50 ms
of launches for 16-30 ms of GPU work. Captured as a CUDA graph, a forward is a single launch. A graph has fixed shapes,
so the input is padded with zero frames up to the next multiple of BUCKET frames (2 s) and masked, the way a batch
pads its shorter pieces (masked frames do not reach the valid ones). One graph per batch size in BATCHES and length,
captured at start-up, largest first so that all of them share one pool of activation memory (about 0.8 GB);
kernels/graph_decoder.py encodes bigger batches 4 pieces at a time, which also keeps their activations small. Other
shapes run as usual.

    encode = GraphEncoder(model.model.encoder, max_frames=3001)
    hidden, valid = encode(features, mask)   # (B, T', 1280) and (B, T') at the bucket's length; use before next call
"""

from __future__ import annotations

import torch


class GraphEncoder:
    BATCHES = (1, 2, 4)
    BUCKET = 200  # log-mel frames (10 ms each)

    def __init__(self, encoder, max_frames: int):
        self.encoder = encoder
        p = next(encoder.parameters())
        mels = encoder.config.num_mel_bins
        self.lengths = list(range(self.BUCKET, max_frames + self.BUCKET, self.BUCKET))
        self.graphs: dict[tuple[int, int], tuple] = {}
        pool = torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream()
        with torch.no_grad():  # largest first: each smaller graph then fits in the pool memory the larger left
            for b in reversed(self.BATCHES):
                for t in reversed(self.lengths):
                    x = torch.zeros(b, t, mels, device=p.device, dtype=p.dtype)
                    m = torch.ones(b, t, device=p.device, dtype=torch.bool)
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        encoder(x, attention_mask=m)
                    torch.cuda.current_stream().wait_stream(side)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=pool):
                        out = encoder(x, attention_mask=m)
                    self.graphs[(b, t)] = (g, x, m, out.last_hidden_state, out.attention_mask)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # the warm-up runs' blocks, cached for the side stream only

    @torch.no_grad()
    def __call__(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(hidden states, valid frames as int) of features (B, T, mels) with mask (B, T) bool, both on the GPU. With a
        graph the results live in its buffers until the next call with the same shape bucket."""
        b, t = features.shape[:2]
        bb = next((n for n in self.BATCHES if n >= b), None)
        tt = next((n for n in self.lengths if n >= t), None)
        if bb is None or tt is None:
            out = self.encoder(features, attention_mask=mask)
            return out.last_hidden_state, out.attention_mask
        g, x, m, hidden, valid = self.graphs[(bb, tt)]
        x.zero_()
        x[:b, :t].copy_(features)
        m.fill_(True)  # rows past the batch: zeros, all valid (never read)
        m[:b].fill_(False)
        m[:b, :t].copy_(mask)
        g.replay()
        return hidden[:b], valid[:b]
