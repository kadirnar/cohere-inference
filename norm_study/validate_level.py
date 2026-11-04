"""Validation of the endpoint's input level normalisation (level=true) against the plain channel mean (level=false):
Arena sets with the language forced (WER v2, the Arena's scorer), the same clips made 40 and 60 dB quieter, and the
Arena's noise-only clips (share of clips that get any text / 3+ words: hallucination).

    cd $ARENA_REPO
    python /path/to/cohere-inference/norm_study/validate_level.py http://127.0.0.1:8086 150 [quiet]
"""

import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import soundfile as sf

from sttarena.data import Suite, iter_rows
from sttarena.runner import _clean_hyp, result_entry, score_rows

COLS = None  # every column; "quiet" as a third argument: only level=true at 0, -40 and -60 dB
SETS = [("arabic", "fleurs_ar", "clean"), ("arabic", "sada", "telephony_v2"), ("arabic", "casablanca", "callcenter_v2"),
        ("english", "librispeech_clean", "clean"), ("english", "edacc", "clean"), ("english", "earnings22", "telephony")]


def ask(base, audio, sr, language, level):
    b = io.BytesIO()
    sf.write(b, audio, sr, format="WAV", subtype="FLOAT")
    for attempt in range(5):
        r = requests.post(base + "/transcribe", files={"file": ("clip.wav", b.getvalue())},
                          data={"language": language, "level": "true" if level else "false"}, timeout=600)
        if r.status_code != 503:
            r.raise_for_status()
            return _clean_hyp(r.json()["text"])
        time.sleep(2 + attempt)
    r.raise_for_status()


def main():
    global COLS
    base, n = sys.argv[1].rstrip("/"), int(sys.argv[2])
    if sys.argv[3:] == ["quiet"]:
        COLS = [("on", 0, True), ("-40 on", -40, True), ("-60 on", -60, True)]
    cols = COLS or [("off", 0, False), ("on", 0, True), ("-40 off", -40, False), ("-40 on", -40, True), ("-60 off", -60, False),
            ("-60 on", -60, True)]
    print(f"{'set':<30}{'clips':>6}" + "".join(f"{c[0]:>10}" for c in cols), flush=True)
    for suite_name, name, cond in SETS:
        suite = Suite.load(suite_name)
        lang = suite.language
        rows = list(iter_rows(lang, name, cond, n))
        failed = [False] * len(rows)
        cells = []
        for _, gain, level in cols:
            g = 10 ** (gain / 20)
            with ThreadPoolExecutor(3) as ex:
                hyps = list(ex.map(lambda r: ask(base, (r["audio"] * g).astype(np.float32), r["sampling_rate"], lang, level),
                                   rows))
            cells.append(100 * result_entry(score_rows(suite, name, rows, hyps, failed, cond), rows, name, cond, "x",
                                            None)["wer_v2"])
        print(f"{name + '/' + cond:<30}{len(rows):>6}" + "".join(f"{c:>10.2f}" for c in cells), flush=True)
    for suite_name in ("arabic", "english"):
        lang = Suite.load(suite_name).language
        rows = list(iter_rows(lang, "noise_only", "clean"))
        line = f"noise_only ({lang}, {len(rows)} clips):"
        for level in (False, True):
            with ThreadPoolExecutor(3) as ex:
                hyps = list(ex.map(lambda r: ask(base, r["audio"], r["sampling_rate"], lang, level), rows))
            anyw = sum(bool(h.split()) for h in hyps) / len(rows)
            three = sum(len(h.split()) >= 3 for h in hyps) / len(rows)
            line += f"  level={'on' if level else 'off'}: any text {100 * anyw:.1f} %, 3+ words {100 * three:.1f} %"
        print(line, flush=True)


if __name__ == "__main__":
    main()
