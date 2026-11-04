"""Check of the scoring path used by arena_check.py: re-score the published s86k predictions of the same sets with
sttarena.runner.score_rows and compare with the published WER v2 (and v1, CER v2) of those sets.

    cd $ARENA_REPO && python /path/to/cohere-inference/test/scorer_check.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arena_check import MODEL, SETS  # noqa: E402
from sttarena.data import Suite, iter_rows  # noqa: E402
from sttarena.runner import result_entry, score_rows  # noqa: E402

ok = True
for suite_name, name, cond in SETS:
    suite = Suite.load(suite_name)
    lang = suite.language
    rows = [{k: v for k, v in r.items() if k != "audio"} for r in iter_rows(lang, name, cond)]
    pub = {p["id"]: p for p in map(json.loads, open(f"results/{lang}/{MODEL}/predictions/{name}__{cond}.jsonl"))}
    hyps = [pub[r["id"]]["hyp"] for r in rows]
    failed = [bool(pub[r["id"]]["diag"].get("failed")) for r in rows]
    mine = result_entry(score_rows(suite, name, rows, hyps, failed, cond), rows, name, cond, "check", None)
    p = json.load(open(f"results/{lang}/{MODEL}/summary.json"))["results"][f"{name}/{cond}"]
    same = all(abs(mine[k] - p[k]) < 1e-12 for k in ("wer_v2", "wer", "cer_v2"))
    ok &= same
    print(f"{name + '/' + cond:<32} rows {len(rows):>5}  WER v2 {100 * mine['wer_v2']:6.2f} vs published "
          f"{100 * p['wer_v2']:6.2f}  v1 {100 * mine['wer']:6.2f} vs {100 * p['wer']:6.2f}  CER v2 {100 * mine['cer_v2']:5.2f}"
          f" vs {100 * p['cer_v2']:5.2f}  {'identical' if same else 'DIFFERENT'}")
print("scoring path reproduces the published numbers exactly" if ok else "MISMATCH")
