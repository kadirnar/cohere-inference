"""Do transcripts change when requests share GPU batches? N clips (FLEURS Arabic and LibriSpeech from an Arabic Arena
checkout) are sent one at a time and then all at once; the two sets of transcripts are compared and both scored
against the references with the Arena's L2 normaliser (textnorm).

    cd $ARENA_REPO && python /path/to/cohere-inference/bench/batch_consistency.py http://127.0.0.1:8086 100
"""

import io
import sys
import threading
from pathlib import Path

import pyarrow.parquet as pq
import requests
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import textnorm  # noqa: E402


def main():
    url, n = sys.argv[1].rstrip("/"), int(sys.argv[2])
    rows = []
    for lang, name in (("ar", "fleurs_ar"), ("en", "librispeech_clean")):
        t = pq.read_table(f"data/{lang}/{name}/clean.parquet", columns=["audio", "text"]).slice(0, n // 2)
        rows += [(r["audio"]["bytes"], r["text"]) for r in t.to_pylist()]
    ask = lambda a: requests.post(url + "/transcribe", files={"file": ("c.wav", io.BytesIO(a))},  # noqa: E731
                                  data={"language": "auto"}, timeout=900).json()["text"]
    single = [ask(a) for a, _ in rows]
    together = [None] * len(rows)
    gate = threading.Barrier(len(rows))

    def one(i):
        gate.wait()
        together[i] = ask(rows[i][0])

    threads = [threading.Thread(target=one, args=(i,)) for i in range(len(rows))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    same = sum(a == b for a, b in zip(single, together))
    for name, hyps in (("one at a time", single), ("all at once", together)):
        e = w = 0
        for (_, ref), h in zip(rows, hyps):
            lang = textnorm.language_of(ref, h)
            r = textnorm.words(ref, lang)
            e += Levenshtein.distance(r, textnorm.words(h, lang))
            w += len(r)
        print(f"{name:<14} WER (Arena L2) {100 * e / w:.2f} %")
    print(f"identical transcripts: {same}/{len(rows)}")


if __name__ == "__main__":
    main()
