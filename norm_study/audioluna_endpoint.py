"""The 20 Audioluna calls through the deployed endpoint (127.0.0.1 only), the model's own language, with the server's
input level normalisation on and off; transcripts to $RESULTS_DIR/<tag>-{on,off}/audioluna.json (private: keep them
out of this repository) for gs_score.py.

    AUDIOLUNA_PARQUET=.../train-00000-of-00001.parquet RESULTS_DIR=... \
        python norm_study/audioluna_endpoint.py http://127.0.0.1:8086 ep2
"""

import io
import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq
import requests
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from levelling import SR, decode  # noqa: E402

DATA = os.environ.get("AUDIOLUNA_PARQUET", "")  # the dataset's data/train-00000-of-00001.parquet
RESULTS = Path(os.environ.get("RESULTS_DIR", "results"))


def main():
    base, tag = sys.argv[1].rstrip("/"), sys.argv[2]
    t = pq.read_table(DATA, columns=["audio", "file_name"]).to_pydict()
    out = {"on": {}, "off": {}}
    for audio, name in zip(t["audio"], t["file_name"]):
        x = decode(audio["bytes"] if isinstance(audio, dict) else audio)
        b = io.BytesIO()
        sf.write(b, x, SR, format="WAV", subtype="FLOAT")
        for k in out:
            r = requests.post(base + "/transcribe", files={"file": ("call.wav", b.getvalue())},
                              data={"language": "auto", "level": "true" if k == "on" else "false"}, timeout=900)
            r.raise_for_status()
            out[k][name] = r.json()["text"]
    for k, v in out.items():
        d = RESULTS / f"{tag}-{k}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "audioluna.json").write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", d, flush=True)


if __name__ == "__main__":
    main()
