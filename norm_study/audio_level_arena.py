"""Input level study on Arabic Arena sets through the endpoint (language forced, as the Arena runs it), scored with
the Arena's own scorer (WER v2). Each clip is sent at its stored sampling rate after one level change:

  orig       as built
  -20dB      quieter by 20 dB          -40dB, -60dB   much quieter (dither and the log floor start to matter)
  +12clip    louder by 12 dB, clipped to [-1, 1]
  rms-20     whole clip scaled to an RMS of -20 dBFS
  peak-1     whole clip scaled to a peak of -1 dBFS
  -60>rms    the -60 dB copy, then rms-20 (does normalising undo the loss?)

    cd $ARENA_REPO
    python /path/to/cohere-inference/norm_study/audio_level_arena.py http://127.0.0.1:8086 200
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test"))
import arena_check  # noqa: E402
from sttarena.data import Suite, iter_rows  # noqa: E402
from sttarena.runner import _clean_hyp, result_entry, score_rows  # noqa: E402

SETS = [("arabic", "fleurs_ar", "clean"), ("arabic", "sada", "telephony_v2"), ("english", "librispeech_clean", "clean"),
        ("english", "edacc", "clean")]


def db(x, d):
    return (x * 10 ** (d / 20)).astype(np.float32)


def rms_to(x, target=-20.0):
    r = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) or 1e-12
    return db(x, target - 20 * np.log10(r))


def peak_to(x, target=-1.0):
    p = float(np.max(np.abs(x))) or 1e-12
    return db(x, target - 20 * np.log10(p))


VARIANTS = {"orig": lambda x: x, "-20dB": lambda x: db(x, -20), "-40dB": lambda x: db(x, -40),
            "-60dB": lambda x: db(x, -60), "+12clip": lambda x: np.clip(db(x, 12), -1, 1),
            "rms-20": rms_to, "peak-1": peak_to, "-60>rms": lambda x: rms_to(db(x, -60))}


def main():
    arena_check.BASE = sys.argv[1].rstrip("/")
    n = int(sys.argv[2])
    print(f"{'set':<30}{'clips':>6}" + "".join(f"{v:>10}" for v in VARIANTS), flush=True)
    for suite_name, name, cond in SETS:
        suite = Suite.load(suite_name)
        lang = suite.language
        rows = list(iter_rows(lang, name, cond, n))
        failed = [False] * len(rows)
        cells = []
        for v, f in VARIANTS.items():
            def one(r):
                return _clean_hyp(arena_check.ask(arena_check.wav(f(r["audio"]), r["sampling_rate"]), lang)["text"])
            with ThreadPoolExecutor(3) as ex:
                hyps = list(ex.map(one, rows))
            w = result_entry(score_rows(suite, name, rows, hyps, failed, cond), rows, name, cond, "check", None)["wer_v2"]
            cells.append(100 * w)
        print(f"{name + '/' + cond:<30}{len(rows):>6}" + "".join(f"{c:>10.2f}" for c in cells), flush=True)


if __name__ == "__main__":
    main()
