"""Score Audioluna transcripts against the GS (golden standard) references of the Audioluna review site.

GS: human-reviewed transcripts of the 20 unique calls (recording 10 duplicates 09 and has zero weight), from gs.json
of the review site; the site's per-recording shards also carry 22 providers' transcripts (shards 14 and 17 are not
published, so providers are compared on the other 18 calls). Our models' transcripts come from
<tag>/audioluna.json files under RESULTS_DIR.

Each pair is scored with two reproducible normalisers applied to both sides: normalize_ext (our usual Audioluna one)
and the Arabic Arena's L2 normaliser (dialect-lenient; it also folds Latin loanwords and numbers). Edits are pooled
over calls (sum of edits / sum of GS words). The site's own context-adjusted WER (policy gs-meaning-v5-numeric-local,
not published as code) is shown for the providers on the same calls, to check that the two rankings agree.

    AUDIOLUNA_REVIEW=.../site RESULTS_DIR=... python norm_study/gs_score.py TAG [TAG ...]

AUDIOLUNA_REVIEW is a local copy of the review site's gs.json and data/recording-XX.json; RESULTS_DIR holds
<tag>/audioluna.json files (file name -> transcript). Both are private: keep them out of this repository.
"""

import json
import os
import sys
from pathlib import Path

from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from textnorm.arabic_norm import normalize_ext  # noqa: E402
from textnorm.arena_normalizers import get_text_normalizer  # noqa: E402

SITE = Path(os.environ.get("AUDIOLUNA_REVIEW", "site"))
RESULTS = Path(os.environ.get("RESULTS_DIR", "results"))
L2 = get_text_normalizer("ar", "L2")
NORMS = {"ext": lambda t: normalize_ext(t).split(), "L2": lambda t: L2.words(t)}


def spearmanr(a, b):
    """Spearman rank correlation (average ranks for ties), as a 1-tuple like scipy's result."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r, i = [0.0] * len(v), 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return (cov / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5,)


def load():
    g = json.load(open(SITE / "gs.json", encoding="utf-8"))
    recs = [r for r in g["recordings"] if str(r["unique_scoring_weight"]) == "1"]
    gs = {r["filename"]: r["transcript_raw"] for r in recs}
    rid = {r["filename"]: r["recording_id"] for r in recs}
    site = {r["filename"]: {k: v for k, v in r["scores"].items()} for r in recs}
    providers = {}
    for r in recs:
        p = SITE / r["shard"]
        if p.stat().st_size == 0:
            continue
        for k, t in json.load(open(p, encoding="utf-8"))["transcripts"].items():
            if k != "gs" and t.get("status") == "complete":
                providers.setdefault(k, {})[r["filename"]] = t["text"]
    cohort = {f: ("AR" if rid[f] in ("01", "02", "03", "04", "05", "06", "07", "08", "20", "21") else "EN") for f in gs}
    return gs, site, providers, cohort


def pooled(hyps, gs, files, norm):
    e = n = 0
    for f in files:
        r, h = NORMS[norm](gs[f]), NORMS[norm](hyps.get(f, ""))
        e += Levenshtein.distance(r, h)
        n += len(r)
    return 100 * e / n


def main():
    gs, site, providers, cohort = load()
    full = sorted(gs)
    with_shards = sorted(set.intersection(*(set(v) for v in providers.values())))
    ours = {}
    for tag in sys.argv[1:]:
        p = RESULTS / tag / "audioluna.json"
        if p.exists():
            ours[tag] = json.load(open(p, encoding="utf-8"))
    print(f"GS: {len(full)} unique calls; providers' transcripts on {len(with_shards)} of them\n")
    print(f"== {len(with_shards)} calls (the ones with providers' transcripts): WER vs GS (%)")
    print(f"{'source':<28}{'site adj.':>10}{'L2':>8}{'ext':>8}{'L2 AR':>8}{'L2 EN':>8}")
    rows = []
    for k, hyps in list(providers.items()) + [("OURS " + t, h) for t, h in ours.items()]:
        adj = None
        if k in providers:
            e = sum(site[f][k]["edits"] for f in with_shards if k in site[f])
            n = sum(site[f][k]["reference_tokens"] for f in with_shards if k in site[f])
            adj = 100 * e / n
        ar = [f for f in with_shards if cohort[f] == "AR"]
        en = [f for f in with_shards if cohort[f] == "EN"]
        rows.append((pooled(hyps, gs, with_shards, "L2"), k, adj, pooled(hyps, gs, with_shards, "ext"),
                     pooled(hyps, gs, ar, "L2"), pooled(hyps, gs, en, "L2")))
    for l2, k, adj, ext, l2ar, l2en in sorted(rows):
        a = f"{adj:10.2f}" if adj is not None else f"{'':>10}"
        print(f"{k:<28}{a}{l2:8.2f}{ext:8.2f}{l2ar:8.2f}{l2en:8.2f}")
    prov = [(adj, l2, ext) for l2, k, adj, ext, _, _ in rows if adj is not None]
    print(f"\nSpearman (site adj. vs L2) over {len(prov)} providers: {spearmanr([p[0] for p in prov], [p[1] for p in prov])[0]:.3f};"
          f" (site adj. vs ext): {spearmanr([p[0] for p in prov], [p[2] for p in prov])[0]:.3f}")
    if ours:
        print(f"\n== all {len(full)} unique calls: our models vs GS (%)")
        print(f"{'model':<28}{'L2':>8}{'ext':>8}{'L2 AR':>8}{'L2 EN':>8}")
        ar = [f for f in full if cohort[f] == "AR"]
        en = [f for f in full if cohort[f] == "EN"]
        for t, h in sorted(ours.items(), key=lambda kv: pooled(kv[1], gs, full, "L2")):
            print(f"{t:<28}{pooled(h, gs, full, 'L2'):8.2f}{pooled(h, gs, full, 'ext'):8.2f}"
                  f"{pooled(h, gs, ar, 'L2'):8.2f}{pooled(h, gs, en, 'L2'):8.2f}")


if __name__ == "__main__":
    main()
