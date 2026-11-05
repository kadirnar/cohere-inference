"""Inference time of this service's pipeline against plain transformers use of the same model (the model card's code),
on one GPU, one recording at a time, each timed after a warm-up as the median of --repeats runs.

  card      the card's code on its own copy of the model, transformers as installed (none of this service's
            kernels): the whole recording to the processor (which cuts audio over 35 s itself at low-energy points), the
            channels averaged, greedy, 448 new tokens, the model's own language header
  service   server.Engine as the service runs it (with FAST and INT8 as set in the environment): level
            normalisation, audio up to 30 s whole and longer audio cut at pauses (Silero) into pieces of at most 25 s
            decoded in batches, the same decoding

Short clips come from the Arabic Arena's built sets (run from its checkout); long recordings are test/long_mixed.wav
and, with AUDIOLUNA_PARQUET, the 20 Audioluna calls (private: only timings and pooled WER are printed; transcripts go
to RESULTS_DIR/bench-{card,service}/audioluna.json for norm_study/gs_score.py). With --url, the service is also timed
end to end over HTTP.

    cd $ARENA_REPO && MODEL=/path/to/model python /path/to/cohere-inference/bench/speed.py [--url http://127.0.0.1:8086]
"""

import argparse
import io
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import soxr
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import server  # noqa: E402

SR = 16000


def load(data: bytes) -> np.ndarray:
    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return np.ascontiguousarray(soxr.resample(x, sr, SR) if sr != SR else x, dtype=np.float32)


class Card:
    """The model card's usage, one recording per call."""

    def __init__(self, processor, model):
        self.processor, self.model = processor, model

    @torch.no_grad()
    def __call__(self, x: np.ndarray) -> str:
        inputs = self.processor(audio=[x.mean(axis=1)], sampling_rate=SR, return_tensors="pt", language="ar",
                                punctuation=True)
        inputs["decoder_input_ids"] = inputs["decoder_input_ids"][:, :4]  # the model's own language header
        chunks = inputs.pop("audio_chunk_index", None)
        inputs = {k: v.cuda().to(torch.bfloat16) if v.is_floating_point() else v.cuda() for k, v in inputs.items()}
        out = self.model.generate(**inputs, do_sample=False, max_new_tokens=server.MAX_NEW_TOKENS)
        texts = self.processor.tokenizer.batch_decode(out, skip_special_tokens=True)
        if chunks is not None:  # the processor cut the recording: its chunks come back in order
            return " ".join(" ".join(t.split()) for t in texts)
        return " ".join(texts[0].split())


def timed(fn, x, repeats):
    out, times = None, []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn(x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return out, statistics.median(times)


def http(url, x, repeats):
    import requests

    b = io.BytesIO()
    sf.write(b, x, SR, format="WAV", subtype="FLOAT")
    data, times = b.getvalue(), []
    for _ in range(repeats):
        t = time.perf_counter()
        r = requests.post(url + "/transcribe", files={"file": ("a.wav", data)}, data={"language": "auto"}, timeout=900)
        r.raise_for_status()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--short", type=int, default=50, help="clips per short set")
    ap.add_argument("--url", help="also time the running service over HTTP")
    args = ap.parse_args()

    from transformers import CohereAsrForConditionalGeneration

    model_dir = os.environ.get("MODEL", server.MODEL_ID)
    engine = server.Engine(model_dir)
    kw = {} if os.path.isdir(model_dir) else {"revision": server.REVISION}
    plain = CohereAsrForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16, **kw).cuda().eval()
    card = Card(engine.processor, plain)
    service = lambda x: " ".join(s["text"] for s in engine(x, "auto", True, True) if s["text"])  # noqa: E731
    groups = {"short clips (FLEURS ar + LibriSpeech)": [], "long recordings": []}
    for lang, name in (("ar", "fleurs_ar"), ("en", "librispeech_clean")):
        for r in pq.read_table(f"data/{lang}/{name}/clean.parquet", columns=["audio"]).slice(0, args.short).to_pylist():
            groups["short clips (FLEURS ar + LibriSpeech)"].append((None, load(r["audio"]["bytes"])))
    groups["long recordings"].append((None, load((ROOT / "test" / "long_mixed.wav").read_bytes())))
    audioluna = os.environ.get("AUDIOLUNA_PARQUET")
    if audioluna:
        t = pq.read_table(audioluna, columns=["audio", "file_name"]).to_pydict()
        groups["Audioluna calls (stereo, 20 + 1 copy)"] = [(n, load(a["bytes"])) for a, n in zip(t["audio"], t["file_name"])]

    for x in (groups["short clips (FLEURS ar + LibriSpeech)"][0][1], groups["long recordings"][0][1]):  # warm-up
        card(x), service(x)
    print(f"{'inputs':<40}{'audio s':>9}{'card s':>9}{'RTFx':>7}{'service s':>11}{'RTFx':>7}{'ratio':>7}"
          + (f"{'HTTP s':>9}" if args.url else ""), flush=True)
    outs = {"card": {}, "service": {}}
    for group, items in groups.items():
        audio = sum(len(x) for _, x in items) / SR
        tc = ts = th = 0.0
        same = 0
        for name, x in items:
            a, t1 = timed(card, x, args.repeats)
            b, t2 = timed(service, x, args.repeats)
            tc, ts = tc + t1, ts + t2
            same += a == b
            if args.url:
                th += http(args.url.rstrip("/"), x, args.repeats)
            if name:
                outs["card"][name], outs["service"][name] = a, b
        line = (f"{group:<40}{audio:>9.1f}{tc:>9.2f}{audio / tc:>7.0f}{ts:>11.2f}{audio / ts:>7.0f}{ts / tc:>7.2f}"
                + (f"{th:>9.2f}" if args.url else ""))
        print(line + f"   same text {same}/{len(items)}", flush=True)
    if audioluna and os.environ.get("RESULTS_DIR"):
        for k, v in outs.items():
            d = Path(os.environ["RESULTS_DIR"]) / f"bench-{k}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "audioluna.json").write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
