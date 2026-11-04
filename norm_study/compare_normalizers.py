"""How normalize_ext (the Test Arabic benchmark's normaliser, also in the inference repo's wer.py) and normalize_elm
compare with the Arabic Arena's normaliser (v2 levels L1 / L2, Whisper's English normaliser) on real transcripts: the
published s86k predictions of every Arabic and English Arena set (public data).

Per set: pooled word error rate under each normaliser. Over all utterances: the token pairs that normalize_ext counts
as substitutions but the Arena's L2 folds to the same word, the insertions/deletions L2 drops (fillers, ...), and
texts on which normalize_ext is not idempotent.

    cd $ARENA_REPO   # for the published predictions under results/
    python /path/to/cohere-inference/norm_study/compare_normalizers.py > compare.txt
"""

import collections
import glob
import json
import os
import sys
from pathlib import Path

from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from textnorm.arabic_norm import normalize_elm, normalize_ext  # noqa: E402
from textnorm.arena_normalizers import get_text_normalizer  # noqa: E402

MODEL = "cohere-transcribe-ar-sft-v9-full-s86k"


def errors(r, h):
    return Levenshtein.distance(r, h)


def main():
    pairs, l2_drops, ext_drops, not_idem = collections.Counter(), collections.Counter(), collections.Counter(), []
    for lang in ("ar", "en"):
        l1, l2 = get_text_normalizer(lang, "L1"), get_text_normalizer(lang, "L2")
        print(f"\n== {lang}: pooled WER (%) of the published s86k transcripts, uncapped")
        print(f"{'set':<34}{'utts':>6}{'elm':>8}{'ext':>8}{'L1':>8}{'L2':>8}{'ext-L2':>8}")
        tot = collections.Counter()
        for path in sorted(glob.glob(f"results/{lang}/{MODEL}/predictions/*.jsonl")):
            key = os.path.basename(path)[:-6]
            if key.startswith("noise_only"):
                continue
            c = collections.Counter()
            for line in open(path, encoding="utf-8"):
                p = json.loads(line)
                ref, hyp = p["ref"], p["hyp"]
                toks = {"elm": (normalize_elm(ref).split(), normalize_elm(hyp).split()),
                        "ext": (normalize_ext(ref).split(), normalize_ext(hyp).split()),
                        "L1": (l1.words(ref), l1.words(hyp)), "L2": (l2.words(ref), l2.words(hyp))}
                if not toks["L2"][0]:
                    continue
                for k, (r, h) in toks.items():
                    c[k + "_e"] += errors(r, h)
                    c[k + "_n"] += len(r)
                c["utts"] += 1
                for text in (ref, hyp):
                    once = normalize_ext(text)
                    if normalize_ext(once) != once and len(not_idem) < 40:
                        not_idem.append((key, text, once, normalize_ext(once)))
                # what ext counts that L2 forgives, token by token
                if lang == "ar":
                    r, h = toks["ext"]
                    for op, i, j in Levenshtein.editops(r, h):
                        if op == "replace":
                            a, b = r[i], h[j]
                            if l2.canon(a) == l2.canon(b) or "".join(l2.words(a)) == "".join(l2.words(b)):
                                pairs[(a, b)] += 1
                        elif op == "delete" and not l2.words(r[i]):
                            l2_drops[r[i]] += 1
                        elif op == "insert" and not l2.words(h[j]):
                            l2_drops[h[j]] += 1
                    r2, h2 = toks["L2"]
                    for op, i, j in Levenshtein.editops(r2, h2):
                        w = r2[i] if op == "delete" else h2[j] if op == "insert" else None
                        if w is not None and not normalize_ext(w).split():
                            ext_drops[w] += 1
            tot.update(c)
            w = {k: 100 * c[k + "_e"] / c[k + "_n"] for k in ("elm", "ext", "L1", "L2")}
            print(f"{key:<34}{c['utts']:>6}{w['elm']:>8.2f}{w['ext']:>8.2f}{w['L1']:>8.2f}{w['L2']:>8.2f}"
                  f"{w['ext'] - w['L2']:>+8.2f}")
        w = {k: 100 * tot[k + "_e"] / tot[k + "_n"] for k in ("elm", "ext", "L1", "L2")}
        print(f"{'ALL ' + lang:<34}{tot['utts']:>6}{w['elm']:>8.2f}{w['ext']:>8.2f}{w['L1']:>8.2f}{w['L2']:>8.2f}"
              f"{w['ext'] - w['L2']:>+8.2f}")
    print("\n== Arabic: substitutions under ext that L2 folds to one word (top 80)")
    for (a, b), n in pairs.most_common(80):
        print(f"  {n:>5}  {a}  ->  {b}")
    print(f"  total {sum(pairs.values())} in {len(pairs)} distinct pairs")
    print("\n== Arabic: words ext keeps (and counts as ins/del) that L2 drops (top 40)")
    for w_, n in l2_drops.most_common(40):
        print(f"  {n:>5}  {w_}")
    print("\n== Arabic: words L2 keeps (ins/del) that ext drops (top 20)")
    for w_, n in ext_drops.most_common(20):
        print(f"  {n:>5}  {w_}")
    print(f"\n== normalize_ext not idempotent on {len(not_idem)} of the texts checked (first 40 shown)")
    for key, text, once, twice in not_idem:
        print(f"  [{key}] {text[:80]!r}\n      once : {once[:80]!r}\n      twice: {twice[:80]!r}")


if __name__ == "__main__":
    main()
