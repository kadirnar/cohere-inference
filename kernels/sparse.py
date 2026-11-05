"""2:4 sparse feed-forwards for the encoder (FFN_SPARSE, opt-in): the second linear layer of both feed-forward modules
of every layer pruned to two nonzero weights in every four along K (SparseGPT, one shot: each pruned weight's error
spread over the row's remaining ones through the inverse Hessian of the layer's inputs), and run by
kernels/csrc/gemm_sparse.cu as (kept values (N, K / 2) int8, selectors (N, K / 32) int32).

Each group of four keeps two positions, in increasing order (the nonzero ones, made up with zero positions when a
group has fewer); its 4-bit selector holds the first position in its low 2 bits and the second in the high 2, eight
selectors a 32-bit word, the group with the lowest K in the lowest nibble.

Measured on an A10 with Cohere's base model: the 50-clip encoder 3.5 % faster (the sparse GEMM 12 % faster than the
dense one at the 150 W cap: both are held back more by moving activations through L2 than by the tensor cores) for
FLEURS WER ar_eg 6.90 -> 7.03 and en_us 6.30 -> 6.66 (calibrated on 275 of the clips; 6.81 -> 7.16 and 6.34 -> 6.80
on the others).

    python -m kernels.sparse MODEL_DIR OUT.pt CALIBRATION_WAV ...   # the pruned weights (bf16), per layer
    wc, e = compress_24(w_int8)
"""

from __future__ import annotations

import sys

import torch


def compress_24(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = w.shape
    assert w.dtype == torch.int8 and k % 32 == 0
    g = w.view(n, k // 4, 4)
    nz = g != 0
    assert int(nz.sum(-1).max()) <= 2, "more than two nonzero values in a group of four"
    # positions to keep: nonzero first (stable), then zero ones, the first two of them in increasing order
    order = torch.argsort((~nz).to(torch.int8) * 4 + torch.arange(4, device=w.device), dim=-1)[..., :2]
    order = torch.sort(order, dim=-1).values
    kept = torch.gather(g, -1, order).reshape(n, k // 2).contiguous()
    nib = (order[..., 0] | (order[..., 1] << 2)).to(torch.int64).view(n, k // 32, 8)
    words = (nib << (4 * torch.arange(8, device=w.device))).sum(-1)
    words = torch.where(words >= 1 << 31, words - (1 << 32), words).to(torch.int32).contiguous()
    return kept, words


def prune_24_magnitude(w: torch.Tensor) -> torch.Tensor:
    """w with the two smallest-magnitude values of every four along K set to zero."""
    g = w.view(w.shape[0], -1, 4).clone()
    idx = g.float().abs().topk(2, dim=-1, largest=False).indices
    return g.scatter_(-1, idx, 0).view_as(w)


def sparsegpt_24(w: torch.Tensor, h: torch.Tensor, blocksize: int = 128, percdamp: float = 0.01) -> torch.Tensor:
    """w (N, K) pruned 2:4 along K, the remaining weights updated to compensate (SparseGPT); h (K, K) the Hessian
    2 X^T X / n of the layer's inputs X (n, K)."""
    w = w.float().clone()
    cols = w.shape[1]
    h = h.clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    h += percdamp * torch.mean(torch.diag(h)) * torch.eye(cols, device=h.device)
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        w1 = w[:, i1:i2].clone()
        q1 = torch.zeros_like(w1)
        err1 = torch.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]
        mask1 = torch.zeros_like(w1, dtype=torch.bool)
        for i in range(i2 - i1):
            col, d = w1[:, i], hinv1[i, i]
            if i % 4 == 0:
                tmp = w1[:, i:i + 4] ** 2 / (torch.diag(hinv1)[i:i + 4].reshape(1, -1)) ** 2
                mask1.scatter_(1, i + torch.topk(tmp, 2, dim=1, largest=False)[1], True)
            q = col.clone()
            q[mask1[:, i]] = 0
            q1[:, i] = q
            e = (col - q) / d
            w1[:, i:] -= e.unsqueeze(1).matmul(hinv1[i, i:].unsqueeze(0))
            err1[:, i] = e
        w[:, i1:i2] = q1
        w[:, i2:] -= err1.matmul(hinv[i1:i2, i2:])
    return w


@torch.no_grad()
def prune_model(model_dir: str, wavs: list[str], out: str, layers_per_pass: int = 16) -> None:
    """SparseGPT 2:4 of every encoder feed-forward's second linear layer, from the dense model's activations on the
    calibration recordings (transformers' encoder in bf16); saves {"<layer>.feed_forward<1|2>": bf16 weight}."""
    import soundfile as sf
    from transformers import CohereAsrForConditionalGeneration, CohereAsrProcessor

    proc = CohereAsrProcessor.from_pretrained(model_dir)
    model = CohereAsrForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
    feats = [proc.feature_extractor([sf.read(f, dtype="float32", always_2d=True)[0].mean(1)], sampling_rate=16000,
                                    return_tensors="pt")["input_features"] for f in wavs]
    names = [(i, m) for i in range(len(model.model.encoder.layers)) for m in ("feed_forward1", "feed_forward2")]
    pruned = {}
    for g0 in range(0, len(names), layers_per_pass):  # the Hessians of a few layers at a time (100 MB each)
        group = names[g0:g0 + layers_per_pass]
        hs, counts, hooks = {}, {}, []
        for i, m in group:
            lin = getattr(model.model.encoder.layers[i], m).linear2
            key = f"{i}.{m}"
            hs[key], counts[key] = torch.zeros(lin.in_features, lin.in_features, device="cuda"), 0

            def hook(mod, inp, key=key):
                x = inp[0].float().reshape(-1, inp[0].shape[-1])
                hs[key].addmm_(x.t(), x)
                counts[key] += x.shape[0]

            hooks.append(lin.register_forward_pre_hook(hook))
        for f in feats:
            model.model.encoder(input_features=f.cuda().bfloat16())
        for h in hooks:
            h.remove()
        for i, m in group:
            key = f"{i}.{m}"
            lin = getattr(model.model.encoder.layers[i], m).linear2
            pruned[key] = sparsegpt_24(lin.weight.data, hs.pop(key) * (2 / counts[key])).to(torch.bfloat16).cpu()
        torch.cuda.empty_cache()
        print(f"layers {g0}-{g0 + len(group) - 1} pruned", flush=True)
    torch.save(pruned, out)


if __name__ == "__main__":
    prune_model(sys.argv[1], sys.argv[3:], sys.argv[2])
