"""How many Arabic Arena clips the service's audio pipeline would feed to the model differently from the Arena's own
runner (plain clip at 16 kHz): clips that level=true changes, and clips over 30 s (cut into pieces by the service).
Every set and condition of both suites the published s86k run covers. CPU only, public data.

    cd $ARENA_REPO && python /path/to/cohere-inference/norm_study/arena_input_check.py [processes]
"""

import importlib.util
import io
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import soxr

ROOT = Path(__file__).resolve().parents[1]
MODEL = "cohere-transcribe-ar-sft-v9-full-s86k"
_engine = None


def engine():
    global _engine
    if _engine is None:
        sys.path.insert(0, str(ROOT))
        spec = importlib.util.spec_from_file_location("server", ROOT / "server.py")
        server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server)
        _engine = server.Engine  # level() needs no model
    return _engine


def check(task):
    lang, key = task
    name, cond = key.split("/")
    e = engine()
    t = pq.read_table(f"data/{lang}/{name}/{cond}.parquet", columns=["id", "audio"]).to_pylist()
    changed, long_ = [], 0
    for r in t:
        a = r["audio"]
        x, sr = sf.read(io.BytesIO(a["bytes"] if isinstance(a, dict) else a), dtype="float32", always_2d=True)
        x = np.ascontiguousarray(soxr.resample(x, sr, 16000) if sr != 16000 else x, dtype=np.float32)
        long_ += len(x) > 30 * 16000
        if not np.array_equal(e.level(x), x.mean(axis=1).astype(np.float32)):
            changed.append(r["id"])
    return lang, key, len(t), changed, long_


def main():
    tasks = []
    for lang in ("ar", "en"):
        for key in json.load(open(f"results/{lang}/{MODEL}/summary.json"))["results"]:
            tasks.append((lang, key))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool:
        out = pool.map(check, tasks)
    tot = {"ar": [0, 0, 0], "en": [0, 0, 0]}
    changed = {}
    for lang, key, n, ch, lg in out:
        tot[lang][0] += n
        tot[lang][1] += len(ch)
        tot[lang][2] += lg
        if ch:
            changed[f"{lang}/{key}"] = ch
    for lang, (n, c, lg) in tot.items():
        print(f"{lang}: {n} clips, changed by level=true: {c}, longer than 30 s: {lg}")
    for k, ids in sorted(changed.items()):
        print(f"  {k}: {len(ids)} changed")
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "arena_changed_clips.json").write_text(json.dumps(changed, indent=1))


if __name__ == "__main__":
    main()
