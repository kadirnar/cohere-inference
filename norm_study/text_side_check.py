"""Two text-side checks on the published s86k Arena predictions (CPU only, public data):

1. normalized_text is safe: scoring the L2-normalised hypothesis instead of the raw one leaves the Arena's WER v2
   unchanged (the normaliser is idempotent on these texts), and how it moves the normalize_ext WER.
2. what collapsing runaway repetitions (a word or a 2-4 word phrase repeated 5+ times in a row, kept twice) would do.

    cd $ARENA_REPO && python /path/to/cohere-inference/norm_study/text_side_check.py
"""

import json
import re
import sys
from pathlib import Path

from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import textnorm  # noqa: E402
from sttarena.data import Suite, iter_rows  # noqa: E402
from sttarena.runner import result_entry, score_rows  # noqa: E402

M = "cohere-transcribe-ar-sft-v9-full-s86k"
SETS = [("arabic", "fleurs_ar", "clean"), ("arabic", "sada", "telephony_v2"), ("arabic", "casablanca", "callcenter_v2"),
        ("arabic", "mixat_cs", "clean"), ("arabic", "mgb3_egy", "clean"), ("arabic", "shams_lev", "clean"),
        ("english", "edacc", "clean"), ("english", "voxpopuli_en", "clean"), ("english", "ami_sdm", "clean")]
REPEAT = re.compile(r"\b((?:\S+\s+){0,3}?\S+)(?:\s+\1){4,}")


def collapse(t):
    return REPEAT.sub(lambda m: m.group(1) + " " + m.group(1), t)


def main():
    print(f"{'set':<28}{'raw':>8}{'L2-norm':>9}{'collapse':>10}{'ext raw':>9}{'ext L2':>8}{'changed':>9}")
    for suite_name, name, cond in SETS:
        suite = Suite.load(suite_name)
        lang = suite.language
        rows = [{k: v for k, v in r.items() if k != "audio"} for r in iter_rows(lang, name, cond)]
        pub = {p["id"]: p for p in map(json.loads, open(f"results/{lang}/{M}/predictions/{name}__{cond}.jsonl"))}
        raw = [pub[r["id"]]["hyp"] for r in rows]
        normed = [textnorm.normalize(h, lang) for h in raw]
        coll = [collapse(h) for h in raw]
        failed = [False] * len(rows)
        w = {k: 100 * result_entry(score_rows(suite, name, rows, h, failed, cond), rows, name, cond, "x", None)["wer_v2"]
             for k, h in (("raw", raw), ("norm", normed), ("coll", coll))}
        ext = {}
        for k, hs in (("raw", raw), ("norm", normed)):
            e = n = 0
            for r, h in zip(rows, hs):
                rr = textnorm.words(r["text"], lang, "ext")
                e += Levenshtein.distance(rr, textnorm.words(h, lang, "ext"))
                n += len(rr)
            ext[k] = 100 * e / n
        changed = sum(a != b for a, b in zip(raw, coll))
        print(f"{name + '/' + cond:<28}{w['raw']:>8.2f}{w['norm']:>9.2f}{w['coll']:>10.2f}{ext['raw']:>9.2f}"
              f"{ext['norm']:>8.2f}{changed:>9}", flush=True)


if __name__ == "__main__":
    main()
