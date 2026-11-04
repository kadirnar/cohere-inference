"""Input level study on the 20 Audioluna calls through the endpoint (127.0.0.1 only: the calls are private), the model's
own language. Each call is decoded at 16 kHz with its channels kept, then sent in one of four forms:

  mean        the channels as they are (the endpoint averages them)       [the endpoint today]
  level       each channel brought to the same speech level before the sum (common.balanced_mix, no muting)
  mean+rms    the channel mean scaled to an RMS of -20 dBFS
  level+rms   the levelled mix scaled to an RMS of -20 dBFS

Transcripts go to $RESULTS_DIR/ep-<form>/audioluna.json (private: keep them out of this repository), to be scored
against GS by gs_score.py.

    AUDIOLUNA_PARQUET=.../train-00000-of-00001.parquet RESULTS_DIR=... \
        python norm_study/audio_level_audioluna.py http://127.0.0.1:8086
"""

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from levelling import SR, balanced_mix, decode  # noqa: E402

DATA = os.environ.get("AUDIOLUNA_PARQUET", "")  # the dataset's data/train-00000-of-00001.parquet
RESULTS = Path(os.environ.get("RESULTS_DIR", "results"))


def rms_to(x, target=-20.0):
    r = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) or 1e-12
    return (x * 10 ** ((target - 20 * np.log10(r)) / 20)).astype(np.float32)


def send(base, x):
    b = io.BytesIO()
    sf.write(b, x, SR, format="WAV", subtype="FLOAT")
    r = requests.post(base + "/transcribe", files={"file": ("call.wav", b.getvalue())}, data={"language": "auto"},
                      timeout=900)
    r.raise_for_status()
    return r.json()["text"]


def main():
    base = sys.argv[1].rstrip("/")
    t = pq.read_table(DATA, columns=["audio", "file_name"]).to_pydict()
    out = {k: {} for k in ("mean", "level", "mean+rms", "level+rms")}
    for audio, name in zip(t["audio"], t["file_name"]):
        x = decode(audio["bytes"] if isinstance(audio, dict) else audio)
        lev = balanced_mix(x) if x.shape[1] == 2 else x[:, 0]
        forms = {"mean": x, "level": lev, "mean+rms": rms_to(x.mean(axis=1)), "level+rms": rms_to(lev)}
        for k, y in forms.items():
            out[k][name] = send(base, y)
        print(f"{len(out['mean'])}/{len(t['file_name'])} calls ({x.shape[1]} ch, {len(x) / SR:.0f} s)", flush=True)
    for k, v in out.items():
        d = RESULTS / f"ep-{k.replace('+', '-')}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "audioluna.json").write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", d)


if __name__ == "__main__":
    main()
