"""Score the endpoint with the Arabic Arena's own pipeline and compare it with the published s86k run.

The built rows of a few sets go to the endpoint as float32 WAV at their stored sampling rate (the endpoint resamples
to 16 kHz with soxr, as the Arena runner does), once with the suite's language forced (the Arena protocol) and once in
auto mode. Both are scored by sttarena.runner.score_rows: the suite's normaliser and references, headline WER v2
(L2sp). The forced outputs are also compared clip by clip with the published predictions of the same rows.

    cd $ARENA_REPO   # a checkout of github.com/kadirnar/arabic-arena with its built data and published predictions
    python /path/to/cohere-inference/test/arena_check.py http://127.0.0.1:8086
"""

import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import soundfile as sf

from sttarena.data import Suite, iter_rows
from sttarena.runner import _clean_hyp, result_entry, score_rows

BASE = ""  # the endpoint, from the command line
MODEL = "cohere-transcribe-ar-sft-v9-full-s86k"
SETS = [("arabic", "fleurs_ar", "clean"), ("arabic", "sada", "telephony_v2"), ("arabic", "casablanca", "callcenter_v2"),
        ("arabic", "mixat_cs", "clean"), ("arabic", "cv18_ar", "noisy"),
        ("english", "fleurs_en", "clean"), ("english", "edacc", "clean"), ("english", "earnings22", "telephony"),
        ("english", "librispeech_clean", "callcenter")]


def wav(audio, sr: int) -> bytes:
    b = io.BytesIO()
    sf.write(b, audio, sr, format="WAV", subtype="FLOAT")
    return b.getvalue()


def ask(data: bytes, language: str) -> dict:
    for attempt in range(5):
        r = requests.post(BASE + "/transcribe", files={"file": ("clip.wav", data)},
                          data={"language": language, "punctuation": "true"}, timeout=600)
        if r.status_code != 503:
            r.raise_for_status()
            return r.json()
        time.sleep(2 + attempt)
    r.raise_for_status()


def main() -> None:
    global BASE
    BASE = sys.argv[1].rstrip("/")
    t0 = time.time()
    print(f"{'set':<32}{'clips':>6}{'published':>11}{'forced':>9}{'auto':>8}{'same text':>11}{'auto picks':>16}")
    for suite_name, name, cond in SETS:
        suite = Suite.load(suite_name)
        lang = suite.language
        rows = list(iter_rows(lang, name, cond))
        pub = {p["id"]: p for p in map(json.loads, open(f"results/{lang}/{MODEL}/predictions/{name}__{cond}.jsonl"))}
        published = json.load(open(f"results/{lang}/{MODEL}/summary.json"))["results"][f"{name}/{cond}"]["wer_v2"]

        def both(r):
            data = wav(r["audio"], r["sampling_rate"])
            return ask(data, lang), ask(data, "auto")

        with ThreadPoolExecutor(3) as ex:
            out = list(ex.map(both, rows))
        forced = [_clean_hyp(f["text"]) for f, _ in out]
        auto = [_clean_hyp(a["text"]) for _, a in out]
        picks = {}
        for _, a in out:
            picks[a["language"] or "-"] = picks.get(a["language"] or "-", 0) + 1
        failed = [False] * len(rows)
        wf = result_entry(score_rows(suite, name, rows, forced, failed, cond), rows, name, cond, "check", None)["wer_v2"]
        wa = result_entry(score_rows(suite, name, rows, auto, failed, cond), rows, name, cond, "check", None)["wer_v2"]
        same = sum(h == pub[r["id"]]["hyp"] for r, h in zip(rows, forced))
        print(f"{name + '/' + cond:<32}{len(rows):>6}{100 * published:>11.2f}{100 * wf:>9.2f}{100 * wa:>8.2f}"
              f"{100 * same / len(rows):>10.1f}%  {json.dumps(picks)}", flush=True)
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
