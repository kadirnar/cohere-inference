"""Clip-by-clip check of where the endpoint's outputs differ from the published Arena run, and why.

For each set: the model through the Arena's own backend and runner (transcribe_rows) with the Arena's batch size (32)
and with batch size 1, the endpoint with the suite's language forced, and the published predictions. All four are
scored by sttarena.runner.score_rows (WER v2) and compared pairwise (share of identical transcripts). Per-clip outputs
go to logs/compare_<set>__<condition>.jsonl of this repository (public datasets only).

    cd $ARENA_REPO
    python /path/to/cohere-inference/test/compare_check.py http://127.0.0.1:8086
"""

import json
import sys
from pathlib import Path

import soxr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arena_check  # noqa: E402
from sttarena.data import Suite, iter_rows  # noqa: E402
from sttarena.models.base import SR  # noqa: E402
from sttarena.models.registry import load_model  # noqa: E402
from sttarena.runner import _clean_hyp, result_entry, score_rows, transcribe_rows  # noqa: E402

SETS = [("arabic", "sada", "telephony_v2"), ("arabic", "casablanca", "callcenter_v2"), ("english", "edacc", "clean"),
        ("arabic", "fleurs_ar", "clean")]


def main() -> None:
    arena_check.BASE = sys.argv[1].rstrip("/")
    model = load_model(arena_check.MODEL)
    for suite_name, name, cond in SETS:
        suite = Suite.load(suite_name)
        lang = suite.language
        rows = list(iter_rows(lang, name, cond))
        sent = [arena_check.wav(r["audio"], r["sampling_rate"]) for r in rows]  # what the endpoint gets
        for r in rows:  # what the runner gives the model
            if r["sampling_rate"] != SR:
                r["audio"] = soxr.resample(r["audio"], r["sampling_rate"], SR).astype("float32")
        pub = {p["id"]: p["hyp"] for p in map(json.loads, open(
            f"results/{lang}/{arena_check.MODEL}/predictions/{name}__{cond}.jsonl"))}
        runs = {"published": [pub[r["id"]] for r in rows],
                "batch32": transcribe_rows(model, rows, lang, 32, f"{name}/{cond} batch 32")[0],
                "batch1": transcribe_rows(model, rows, lang, 1, f"{name}/{cond} batch 1")[0],
                "endpoint": [_clean_hyp(arena_check.ask(b, lang)["text"]) for b in sent]}
        failed = [False] * len(rows)
        wer = {k: result_entry(score_rows(suite, name, rows, h, failed, cond), rows, name, cond, "check", None)["wer_v2"]
               for k, h in runs.items()}
        same = lambda a, b: sum(x == y for x, y in zip(runs[a], runs[b])) / len(rows)  # noqa: E731
        print(f"== {name}/{cond} ({len(rows)} clips)  WER v2: " + "  ".join(f"{k} {100 * v:.2f}" for k, v in wer.items()))
        print(f"   identical text: endpoint=batch1 {100 * same('endpoint', 'batch1'):.1f}%  "
              f"batch32=published {100 * same('batch32', 'published'):.1f}%  "
              f"batch1=batch32 {100 * same('batch1', 'batch32'):.1f}%  endpoint=published "
              f"{100 * same('endpoint', 'published'):.1f}%", flush=True)
        (HERE.parent / "logs").mkdir(exist_ok=True)
        with open(HERE.parent / "logs" / f"compare_{name}__{cond}.jsonl", "w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                f.write(json.dumps({"id": r["id"], "duration": r.get("duration"), "ref": r["text"],
                                    **{k: v[i] for k, v in runs.items()}}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
