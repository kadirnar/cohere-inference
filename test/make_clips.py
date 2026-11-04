"""Build the test clips of run_tests.py from an Arabic Arena checkout: the first 8 clips of FLEURS Arabic and of
LibriSpeech test-clean (both CC-BY 4.0) with the published s86k transcripts, a 114 s mixed Arabic/English file, a
44.1 kHz stereo copy, m4a / webm / ogg encodings (ffmpeg), a playlist that points at a local file (must be refused)
and an empty file. Nothing here is committed to git.

    cd $ARENA_REPO && python /path/to/cohere-inference/test/make_clips.py
"""

import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

OUT = Path(__file__).resolve().parent
MODEL = "cohere-transcribe-ar-sft-v9-full-s86k"


def main():
    clips = {}
    for name, lang in (("fleurs_ar", "ar"), ("librispeech_clean", "en")):
        t = pq.read_table(f"data/{lang}/{name}/clean.parquet")
        hyps = {}
        for line in open(f"results/{lang}/{MODEL}/predictions/{name}__clean.jsonl", encoding="utf-8"):
            d = json.loads(line)
            hyps[d["id"]] = (d["ref"], d["hyp"])
        for k, r in enumerate(t.slice(0, 8).to_pylist()):
            a = r["audio"]
            x, sr = sf.read(io.BytesIO(a["bytes"] if isinstance(a, dict) else a), dtype="float32")
            path = OUT / f"{name}_{k}.wav"
            sf.write(path, x, sr)
            ref, hyp = hyps.get(r["id"], ("", ""))
            clips[f"test/{path.name}"] = {"id": r["id"], "seconds": round(len(x) / sr, 2), "sr": sr, "ref": ref,
                                          "arena_hyp": hyp}
    (OUT / "clips.json").write_text(json.dumps(clips, ensure_ascii=False, indent=1), encoding="utf-8")
    parts = [OUT / f"fleurs_ar_{k}.wav" for k in range(5)] + [OUT / "librispeech_clean_5.wav", OUT / "librispeech_clean_7.wav"]
    gap = np.zeros(16000, dtype=np.float32)
    x = np.concatenate([np.concatenate([sf.read(p, dtype="float32")[0], gap]) for p in parts])
    sf.write(OUT / "long_mixed.wav", x, 16000)
    ff = ["ffmpeg", "-v", "error", "-y"]
    subprocess.run(ff + ["-i", str(OUT / "long_mixed.wav"), "-ar", "44100", "-ac", "2", str(OUT / "long_stereo_44k.wav")], check=True)
    subprocess.run(ff + ["-i", str(OUT / "fleurs_ar_1.wav"), "-c:a", "aac", "-b:a", "64k", str(OUT / "fleurs_ar_1.m4a")], check=True)
    subprocess.run(ff + ["-i", str(OUT / "fleurs_ar_1.wav"), "-c:a", "libopus", "-b:a", "32k", str(OUT / "fleurs_ar_1.webm")], check=True)
    subprocess.run(ff + ["-i", str(OUT / "librispeech_clean_0.wav"), "-c:a", "libopus", str(OUT / "librispeech_clean_0.ogg")], check=True)
    (OUT / "attack.m3u8").write_text("#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXTINF:10,\nfile:///etc/hostname\n#EXT-X-ENDLIST\n")
    (OUT / "empty.wav").write_bytes(b"")
    print(f"{len(clips)} clips and the derived files in {OUT}")


if __name__ == "__main__":
    main()
