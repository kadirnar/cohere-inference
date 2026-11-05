"""Where the service's inference time goes, stage by stage (with GRAPHS on): the log-mel features (CPU, per piece), the
encoder (CUDA graphs, 4 pieces at a time), the decoder (the rest of GraphDecoder.generate: the cross-attention keys and
values, the prompt pass and the steps) and, for audio over 30 s, Silero's cut at pauses.

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
    feats, t_feat = sync_time(lambda: [engine.features(a) for a in audio])
    n = max(len(f) for f in feats)
    x = torch.zeros(len(feats), n, feats[0].shape[1])
    mask = torch.zeros(len(feats), n, dtype=torch.bool)
    for i, f in enumerate(feats):
        x[i, :len(f)], mask[i, :len(f)] = f, True
    x, mask = x.cuda().to(torch.bfloat16), mask.cuda()
    g = engine.graph
    rows = g.ENCODE_ROWS
    lengths = [len(f) for f in feats]

    def encode():
        for r in range(0, len(feats), rows):
            m = max(lengths[r:r + rows])
            g.encode(x[r:r + rows, :m], mask[r:r + rows, :m])

    with torch.no_grad():
        _, t_enc = sync_time(encode)
        out, t_gen = sync_time(lambda: g.generate(x, mask, engine.prompt("auto", True)))
    steps = out.shape[1] - len(engine.prompt("auto", True))
    seconds = sum(len(a) for a in audio) / SR
    print(f"{label:<34} audio {seconds:6.1f} s | features {1000 * t_feat:6.1f} ms | encoder {1000 * t_enc:6.1f} ms | "
          f"decoder {1000 * (t_gen - t_enc):6.1f} ms for {steps} steps ({1000 * (t_gen - t_enc) / steps:4.2f} ms/step) "
          f"| total {1000 * (t_feat + t_gen):6.1f} ms", flush=True)


def main():
    engine = server.Engine(os.environ.get("MODEL", server.MODEL_ID))
    assert engine.graph, "run with GRAPHS on"
    clips = [sf.read(ROOT / "test" / f"{n}.wav", dtype="float32")[0] for n in
             [f"fleurs_ar_{k}" for k in range(8)] + [f"librispeech_clean_{k}" for k in range(8)]]
    long = sf.read(ROOT / "test" / "long_mixed.wav", dtype="float32")[0]
    stages(engine, [clips[1]], "warm-up")
    stages(engine, [clips[1]], "1 clip (17 s)")
    stages(engine, [clips[8]], "1 clip (6 s, English)")
    stages(engine, clips, "16 clips in one batch")
    stages(engine, clips * 2, "32 clips in one batch")
    spans, _ = engine.prepare(long[:, None], True)
    _, t_cut = sync_time(lambda: engine.pieces(long), repeats=3)
    print(f"Silero cut of a {len(long) / SR:.0f} s recording into {len(spans)} pieces: {1000 * t_cut:.0f} ms", flush=True)
    stages(engine, [long[s:e] for s, e in spans], f"its {len(spans)} pieces in one batch")


if __name__ == "__main__":
    main()
