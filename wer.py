"""WER of the endpoint on a test set: every recording is sent to the endpoint and the transcripts are scored against
the references with the Arabic Arena's normaliser (L2, dialect-lenient; Whisper's for English) and with normalize_ext
of the Test Arabic ASR Benchmark (textnorm). Errors are pooled over recordings (edits / reference words).

    .venv/bin/python wer.py test.jsonl [--url http://127.0.0.1:8086] [--language auto] [--no-level] [--out hyps.jsonl]

test.jsonl has one {"audio": "path/relative/to/the/file.wav", "text": "reference"} per line (a .csv with audio,text
columns works too).
"""

import argparse
import csv
import json
from pathlib import Path

import requests
from rapidfuzz.distance import Levenshtein

import textnorm


def rows(path: Path):
    if path.suffix == ".csv":
        with open(path, encoding="utf-8") as f:
            yield from csv.DictReader(f)
    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("test")
    ap.add_argument("--url", default="http://127.0.0.1:8086")
    ap.add_argument("--language", default="auto", choices=["auto", "ar", "en"])
    ap.add_argument("--no-level", action="store_true", help="the plain channel mean instead of level normalisation")
    ap.add_argument("--out", help="write the transcripts here (jsonl)")
    args = ap.parse_args()
    test = Path(args.test)
    tot = {"l2": [0, 0], "ext": [0, 0]}
    out = open(args.out, "w", encoding="utf-8") if args.out else None
    for k, r in enumerate(rows(test), 1):
        with open(test.parent / r["audio"], "rb") as f:
            resp = requests.post(args.url.rstrip("/") + "/transcribe", files={"file": f},
                                 data={"language": args.language, "level": "false" if args.no_level else "true"},
                                 timeout=1800)
        resp.raise_for_status()
        hyp = resp.json()["text"]
        language = textnorm.language_of(r["text"], hyp)
        for norm, t in tot.items():
            ref = textnorm.words(r["text"], language, norm)
            t[0] += Levenshtein.distance(ref, textnorm.words(hyp, language, norm))
            t[1] += len(ref)
        if out:
            out.write(json.dumps({"audio": r["audio"], "hyp": hyp}, ensure_ascii=False) + "\n")
        print(f"\r{k} recordings", end="", flush=True)
    print()
    for norm, (e, n) in tot.items():
        name = "Arabic Arena L2" if norm == "l2" else "normalize_ext"
        print(f"WER ({name}): {100 * e / max(n, 1):.2f} %  ({e} edits / {n} reference words)")


if __name__ == "__main__":
    main()
