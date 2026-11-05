"""Where the service's inference time goes, stage by stage (FAST): the log-mel features (GPU), the packed encoder
(subsampling and 48 layers), and decoding (the cross-attention keys and values, then the steps), for one clip, 16 and
32 clips in one call, and for audio over 30 s Silero's cut at pauses.

    MODEL=/path/to/model python bench/stages.py   # from the repository root, after test/make_clips.py
"""

import os
import statistics
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import server  # noqa: E402

SR = 16000


def sync_time(fn, repeats=5):
    out, times = None, []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return out, statistics.median(times)


def stages(engine, audio, label):
    pipe = engine.pipe
    pieces = [pipe.prepare(a) for a in audio]
    prompt = engine.prompt("auto", True)
    (feats, frames), t_feat = sync_time(lambda: pipe.features(pieces))
    (hidden, lengths), t_enc = sync_time(lambda: pipe.encoder(feats, frames))
    tokens, t_all = sync_time(lambda: pipe.submit(pieces, prompt).result())
    steps = max(len(t) for t in tokens) - len(prompt)
    t_dec = t_all - t_feat - t_enc
    seconds = sum(len(a) for a in audio) / SR
    print(f"{label:<28} audio {seconds:6.1f} s | features {1000 * t_feat:5.1f} ms | encoder {1000 * t_enc:6.1f} ms "
          f"({sum(lengths)} frames) | decoder {1000 * t_dec:6.1f} ms for {steps} steps "
          f"({1000 * t_dec / max(steps, 1):4.2f} ms/step) | total {1000 * t_all:6.1f} ms", flush=True)


def main():
    assert server.FAST, "run with FAST on"
    engine = server.Engine(os.environ.get("MODEL", server.MODEL_ID))
    engine.warm_up()
    clips = [sf.read(ROOT / "test" / f"{n}.wav", dtype="float32")[0] for n in
             [f"fleurs_ar_{k}" for k in range(8)] + [f"librispeech_clean_{k}" for k in range(8)]]
    long = sf.read(ROOT / "test" / "long_mixed.wav", dtype="float32")[0]
    stages(engine, [clips[1]], "1 clip (17 s)")
    stages(engine, [clips[8]], "1 clip (6 s, English)")
    stages(engine, clips, "16 clips in one call")
    stages(engine, clips * 2, "32 clips in one call")
    spans, _ = engine.prepare(long[:, None], True)
    _, t_cut = sync_time(lambda: engine.pieces(long), repeats=3)
    print(f"Silero cut of a {len(long) / SR:.0f} s recording into {len(spans)} pieces: {1000 * t_cut:.0f} ms", flush=True)
    stages(engine, [long[s:e] for s, e in spans], f"its {len(spans)} pieces in one call")


if __name__ == "__main__":
    main()
