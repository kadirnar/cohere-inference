"""Latency of one request by audio length: the model card's plain transformers code against this service's pipeline.

Pieces of 3, 7, 15 and 25 s and the whole of test/long_mixed.wav (114 s, Arabic and English speech), each timed alone
on the GPU (median of --repeats runs after a warm-up), end to end from 16 kHz samples to text:

  card      the card's code on its own copy of the model, transformers as installed (none of this service's
            kernels): processor (which cuts audio over 35 s itself), generate, greedy, 448 new tokens
  service   server.Engine with FAST and INT8 as set in the environment: level, pieces (Silero over 30 s),
            log-mel per piece, decoding

    MODEL=/path/to/model python bench/latency.py   # from the repository root
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
import server  # noqa: E402
from speed import Card  # noqa: E402

SR = 16000


def timed(fn, repeats):
    fn()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    from transformers import CohereAsrForConditionalGeneration

    model_dir = os.environ.get("MODEL", server.MODEL_ID)
    engine = server.Engine(model_dir)
    if hasattr(engine, "warm_up"):
        engine.warm_up()
    kw = {} if os.path.isdir(model_dir) else {"revision": server.REVISION}
    plain = CohereAsrForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16, **kw).cuda().eval()
    card = Card(engine.processor, plain)
    x, sr = sf.read(ROOT / "test" / "long_mixed.wav", dtype="float32", always_2d=True)
    assert sr == SR
    x = np.ascontiguousarray(x)
    print(f"{'audio':>8}{'card':>10}{'service':>10}{'speed-up':>10}")
    for seconds in (3, 7, 15, 25, None):
        piece = x if seconds is None else x[20 * SR:(20 + seconds) * SR]
        tc = timed(lambda: card(piece), args.repeats)
        ts = timed(lambda: engine(piece, "auto", True, True), args.repeats)
        print(f"{len(piece) / SR:>7.0f}s{1000 * tc:>8.0f}ms{1000 * ts:>8.0f}ms{tc / ts:>9.1f}x", flush=True)


if __name__ == "__main__":
    main()
