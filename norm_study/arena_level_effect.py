"""Exact effect of the service's input level normalisation on a model's Arabic Arena scores.

Only the clips that level=true changes (logs/arena_changed_clips.json, from arena_input_check.py) can get a different
transcript, so only they are run again, through a running service with the suite's language forced (the Arena
protocol), one clip at a time, with level=false and level=true. Every other clip keeps its published transcript. Each
affected set is then scored with the Arena's own scorer, and the suite's Arena WER v2 recomputed with its weights:

  published   the published run (batch 32, no level normalisation)
  batch1      changed clips re-run one at a time without level normalisation (the batch-size effect alone)
  level       changed clips re-run one at a time with level normalisation

    cd $ARENA_REPO && python /path/to/cohere-inference/norm_study/arena_level_effect.py MODEL_KEY http://127.0.0.1:8086
"""

import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import soundfile as sf

from sttarena import leaderboard as lb
from sttarena.data import Suite, iter_rows
from sttarena.runner import _clean_hyp, result_entry, score_rows

ROOT = Path(__file__).resolve().parents[1]


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
    model, base = sys.argv[1], sys.argv[2].rstrip("/")
    changed = json.load(open(ROOT / "logs" / "arena_changed_clips.json"))
    report = {}
    for suite_name in ("arabic", "english"):
        suite = Suite.load(suite_name)
        lang = suite.language
        weights = lb.set_weights(suite)
        summary = json.load(open(f"results/{lang}/{model}/summary.json"))["results"]
        per_set = {k: {"published": r["wer_v2"]} for k, r in summary.items() if r.get("wer_v2") is not None}
        for key, ids in changed.items():
            if not key.startswith(lang + "/"):
                continue
            _, name, cond = key.split("/")
            ids = set(ids)
            rows = list(iter_rows(lang, name, cond))
            pub = {p["id"]: p["hyp"] for p in map(json.loads, open(f"results/{lang}/{model}/predictions/{name}__{cond}.jsonl"))}
            todo = [r for r in rows if r["id"] in ids]
            for variant, level in (("batch1", False), ("level", True)):
                with ThreadPoolExecutor(3) as ex:
                    new = dict(zip([r["id"] for r in todo],
                                   ex.map(lambda r: ask(base, r["audio"], r["sampling_rate"], lang, level), todo)))
                hyps = [new.get(r["id"], pub[r["id"]]) for r in rows]
                if name == "noise_only":
                    per_set.setdefault(f"{name}/{cond}", {})[variant + "_3plus"] = sum(len(h.split()) >= 3 for h in hyps) / len(rows)
                    continue
                res = result_entry(score_rows(suite, name, rows, hyps, [False] * len(rows), cond), rows, name, cond, "x", None)
                per_set[f"{name}/{cond}"][variant] = res["wer_v2"]
            print(f"  {key:<34} {len(todo):>4} clips: " + "  ".join(
                f"{v} {100 * per_set[f'{name}/{cond}'][v]:.2f}" for v in ("published", "batch1", "level")
                if v in per_set.get(f"{name}/{cond}", {})), flush=True)
        arena = {}
        for variant in ("published", "batch1", "level"):
            scores = {k: v.get(variant, v["published"]) for k, v in per_set.items() if "published" in v}
            arena[variant] = lb.combine(scores, weights)
        report[suite_name] = arena
        print(f"{model} {suite_name}: Arena v2 " + "  ".join(f"{k} {100 * v:.3f}" for k, v in arena.items()), flush=True)
    (ROOT / "logs" / f"arena_level_effect_{model}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
