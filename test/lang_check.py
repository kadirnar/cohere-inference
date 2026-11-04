"""Language check of a running endpoint on Arena sets (first N rows of each), sent to 127.0.0.1 only: the language the
model picks in auto mode, and WER of auto mode against the forced language, with one light normaliser for both.

    cd $ARENA_REPO && python /path/to/cohere-inference/test/lang_check.py http://127.0.0.1:8086
"""

import json
import re
import sys
import time

import pyarrow.parquet as pq
import requests
from rapidfuzz.distance import Levenshtein

BASE = sys.argv[1].rstrip("/")
DATA = "data"  # the built sets of the Arabic Arena checkout this runs from
SETS = [("en", "fleurs_en", "clean", 150), ("en", "librispeech_other", "clean", 150), ("en", "common_voice_en", "clean", 150),
        ("en", "edacc", "clean", 150), ("en", "earnings22", "clean", 100), ("en", "librispeech_clean", "telephony", 100),
        ("en", "fleurs_en", "callcenter", 100),
        ("ar", "fleurs_ar", "clean", 150), ("ar", "cv18_ar", "clean", 150), ("ar", "sada", "clean", 150),
        ("ar", "mgb2", "clean", 100), ("ar", "sada", "callcenter_v2", 100), ("ar", "casablanca", "clean", 100),
        ("ar", "mixat_cs", "clean", 150), ("ar", "tedxtn_cs", "clean", 150), ("ar", "algerian_cs", "clean", 100)]


def norm(t: str) -> list[str]:
    t = re.sub(r"[ً-ْـٰ]", "", t.lower())  # harakat, tatweel, dagger alef
    t = re.sub(r"[إأآٱ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"[^\w\s']", " ", t).split()


def ask(audio: bytes, language: str) -> dict:
    r = requests.post(BASE + "/transcribe", files={"file": ("clip.wav", audio)}, data={"language": language}, timeout=300)
    r.raise_for_status()
    return r.json()


def main() -> None:
    total = {"en": [0, 0, 0, 0], "ar": [0, 0, 0, 0]}  # errors auto, errors forced, words, clips in the other language
    t0 = time.time()
    print(f"{'set':<34}{'clips':>6}{'auto picks (ar/en)':>20}{'WER auto':>10}{'WER forced':>12}{'same text':>11}")
    for lang, name, cond, n in SETS:
        rows = pq.read_table(f"{DATA}/{lang}/{name}/{cond}.parquet", columns=["audio", "text"]).slice(0, n).to_pylist()
        picks, ea, ef, words, same = {"ar": 0, "en": 0, "": 0}, 0, 0, 0, 0
        for r in rows:
            audio = r["audio"]["bytes"] if isinstance(r["audio"], dict) else r["audio"]
            a, f = ask(audio, "auto"), ask(audio, lang)
            picks[a["language"] if a["language"] in picks else ""] += 1
            ref = norm(r["text"])
            ea += Levenshtein.distance(ref, norm(a["text"]))
            ef += Levenshtein.distance(ref, norm(f["text"]))
            words += len(ref)
            same += norm(a["text"]) == norm(f["text"])
        other = picks["en" if lang == "ar" else "ar"]
        for i, v in enumerate((ea, ef, words, other)):
            total[lang][i] += v
        print(f"{name + '/' + cond:<34}{len(rows):>6}{picks['ar']:>13}/{picks['en']:<6}{100 * ea / words:>10.2f}"
              f"{100 * ef / words:>12.2f}{100 * same / len(rows):>10.1f}%", flush=True)
    for lang, (ea, ef, w, other) in total.items():
        print(f"ALL {lang}: WER auto {100 * ea / w:.2f}, forced {100 * ef / w:.2f}; auto picked the other language on {other} clips")
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
