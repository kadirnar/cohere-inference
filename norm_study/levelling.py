"""Audio helpers of the level study (client side): decoding with the channels kept, and the per-channel levelled mix
that the endpoint's level normalisation grew out of (the same rule as the asr-train harness's common.balanced_mix).
"""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf
import soxr

SR = 16000
_VAD = None


def decode(data: bytes) -> np.ndarray:
    """An audio file's bytes as (samples, channels) float32 at 16 kHz."""
    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return soxr.resample(x, sr, SR).astype(np.float32) if sr != SR else x


def balanced_mix(x: np.ndarray, target_dbfs: float = -20.0, max_gain_db: float = 30.0) -> np.ndarray:
    """A two-channel call (samples, channels) summed to mono after bringing each channel's speech to the same level:
    the RMS of its loudest 30 % of 50 ms frames inside Silero's speech regions, moved to target_dbfs (gain at most
    max_gain_db). A channel with under 1 s of speech keeps its level; copies of one channel are just averaged."""
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    global _VAD
    if x.ndim == 1 or x.shape[1] == 1:
        return x.reshape(-1)
    if np.corrcoef(x[:, 0], x[:, 1])[0, 1] > 0.9:  # dual mono
        return x.mean(axis=1)
    if _VAD is None:
        _VAD = load_silero_vad(onnx=True)
    out, n = [], int(0.05 * SR)
    for c in range(x.shape[1]):
        ch = np.ascontiguousarray(x[:, c])
        _VAD.reset_states()
        regions = get_speech_timestamps(torch.from_numpy(ch), _VAD, sampling_rate=SR, threshold=0.5)
        speech = np.concatenate([ch[r["start"]:r["end"]] for r in regions]) if regions else np.zeros(0, np.float32)
        if len(speech) < SR:
            out.append(ch)
            continue
        frames = speech[: len(speech) // n * n].reshape(-1, n)
        rms = np.sqrt((frames ** 2).mean(axis=1))
        level = 20 * np.log10(np.sort(rms)[-max(1, int(len(rms) * 0.3)):].mean() + 1e-9)
        out.append(ch * 10 ** (min(max_gain_db, target_dbfs - level) / 20))
    mix = np.sum(out, axis=0)  # summed: each talker stays at target_dbfs in the mix
    return (mix / max(1.0, float(np.abs(mix).max()) / 0.99)).astype(np.float32)
