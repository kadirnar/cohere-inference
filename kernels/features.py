"""Cohere Transcribe's log-mel features on the GPU, for a whole batch of pieces at once.

The processor computes them on the CPU, one request at a time (about 7 ms for 8 s of audio, single-threaded); under
load that is hundreds of milliseconds of CPU per batch while the GPU waits. The same steps run here on the GPU for the
batch: the deterministic dither (CohereAsrFeatureExtractor seeds torch's CPU generator with the piece's length, so it
is drawn on the CPU, where the request's thread adds it: dither()), pre-emphasis, the STFT (cuFFT, fp32, Hann window of
400 in 512), power, mel filter bank (fp32), log, per-feature normalisation over the valid frames, padded frames zeroed.
As the service always fed the model, a piece of n samples gives n // 160 + 1 frames, the last one zero.
kernels/csrc/features.cu works on the valid frames only (3 kernels and cuFFT); reference() is the same computation in
torch ops on the padded batch (3x the time), which it matches but for one bf16 step in 0.02 % of the values (each 0.07
% from the CPU extractor's).

    x = dither(samples)                                   # in the request's thread
    feats, frames = Features(processor.feature_extractor)([x1, x2, ...])   # (B, T, 128) bf16 on the GPU, frame counts
"""

from __future__ import annotations

import numpy as np
import torch

from kernels.cuda_ext import ext

LOG_GUARD = 2 ** -24
EPS = 1e-5


def dither(x: np.ndarray, amount: float = 1e-5) -> np.ndarray:
    """x + amount * the extractor's noise for a piece of this length (torch CPU generator seeded with the length)."""
    n = len(x)
    if amount <= 0 or n == 0:
        return np.asarray(x, dtype=np.float32)
    g = torch.Generator()
    g.manual_seed(n)
    noise = torch.randn(n, dtype=torch.float32, generator=g)
    return (torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)) + amount * noise).numpy()


class Features:
    def __init__(self, extractor, device="cuda"):
        fe = extractor
        assert fe.n_fft == 512 and fe.hop_length == 160 and fe.win_length == 400
        self.n_fft, self.hop, self.win = fe.n_fft, fe.hop_length, fe.win_length
        self.preemphasis = fe.preemphasis
        self.device = torch.device(device)
        self.window = torch.hann_window(self.win, periodic=False, device=self.device)
        self.mel = fe.mel_filters.to(self.device, torch.float32)  # (128, 257)
        self.mels = self.mel.shape[0]
        left = (self.n_fft - self.win) // 2  # torch.stft centres the window in n_fft
        self.window512 = torch.zeros(self.n_fft, device=self.device)
        self.window512[left:left + self.win] = self.window
        nz = (self.mel != 0).cpu()
        first = nz.int().argmax(1)
        end = self.mel.shape[1] - nz.flip(1).int().argmax(1)
        self.band = torch.stack([first, torch.where(nz.any(1), end, first)], 1).int().contiguous().to(self.device)
        self.pinned = torch.empty(0, dtype=torch.float32).pin_memory()  # the samples' staging buffer, grown as needed
        self.copied = None  # event: the last copy out of it

    @torch.no_grad()
    def __call__(self, pieces: list) -> tuple[torch.Tensor, list[int]]:
        """Normalised log-mel frames (B, T, 128) bf16 of dithered pieces (numpy arrays, or fp32 tensors already on
        the GPU), zero past each piece's frames, and the frame counts (n // 160 + 1)."""
        lengths = [len(p) for p in pieces]
        total = sum(lengths)
        if all(isinstance(p, torch.Tensor) for p in pieces):
            flat = torch.cat(pieces)
        else:
            if self.copied is not None:
                self.copied.synchronize()  # the previous call's samples are out of the staging buffer
            if self.pinned.numel() < total:
                self.pinned = torch.empty(max(total, 2 * self.pinned.numel()), dtype=torch.float32).pin_memory()
            np.concatenate([np.asarray(p, dtype=np.float32) for p in pieces], out=self.pinned[:total].numpy())
            flat = self.pinned[:total].to(self.device, non_blocking=True)
            self.copied = torch.cuda.Event()
            self.copied.record()
        valid = [n // self.hop for n in lengths]
        meta = torch.tensor(np.concatenate([np.cumsum([0] + lengths[:-1]), lengths, np.cumsum([0] + valid)]),
                            dtype=torch.int64).to(self.device, non_blocking=True)
        out = torch.zeros(len(pieces), max(lengths) // self.hop + 1, self.mels, device=self.device,
                          dtype=torch.bfloat16)
        ext.logmel_features(flat, meta, sum(valid), self.window512, self.mel, self.band, self.preemphasis, out)
        return out, [n + 1 for n in valid]

    @torch.no_grad()
    def reference(self, pieces: list[np.ndarray]) -> tuple[torch.Tensor, list[int]]:
        """__call__ in torch ops on the padded batch."""
        lengths = [len(p) for p in pieces]
        n_max = max(lengths)
        flat = torch.from_numpy(np.concatenate([np.asarray(p, dtype=np.float32) for p in pieces])).pin_memory()
        flat = flat.to(self.device, non_blocking=True)
        lens = torch.tensor(lengths, dtype=torch.long).to(self.device, non_blocking=True)
        starts = torch.tensor(np.cumsum([0] + lengths[:-1]), dtype=torch.long).to(self.device, non_blocking=True)
        t = torch.arange(n_max, device=self.device)
        valid = t[None, :] < lens[:, None]
        x = torch.where(valid, flat[(starts[:, None] + t[None, :]).clamp_max(flat.numel() - 1)], 0.0)
        if self.preemphasis:
            x = torch.cat([x[:, :1], x[:, 1:] - self.preemphasis * x[:, :-1]], 1).masked_fill_(~valid, 0.0)
        spec = torch.stft(x, self.n_fft, hop_length=self.hop, win_length=self.win, window=self.window,
                          return_complex=True, pad_mode="constant")
        power = torch.view_as_real(spec).pow(2).sum(-1).sqrt().pow(2)  # as the extractor: |X| then squared
        feats = torch.log(torch.matmul(self.mel, power) + LOG_GUARD).transpose(1, 2)  # (B, F, 128)
        valid_frames = torch.div(lens + self.n_fft // 2 * 2 - self.n_fft, self.hop, rounding_mode="floor")
        mask = (torch.arange(feats.shape[1], device=self.device)[None, :] < valid_frames[:, None])[..., None]
        masked = feats * mask
        mean = masked.sum(1, keepdim=True) / valid_frames[:, None, None]
        var = ((masked - mean) ** 2 * mask).sum(1, keepdim=True) / (valid_frames[:, None, None] - 1)
        feats = ((feats - mean) / (var.sqrt() + EPS)) * mask
        return feats.to(torch.bfloat16), [n // self.hop + 1 for n in lengths]
