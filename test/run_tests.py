"""Check a running endpoint against the Arena hypotheses of the same clips, and its handling of formats and errors.

    python test/run_tests.py http://127.0.0.1:8086   # from the repository root, after test/make_clips.py
"""

import json
import re
import subprocess
import sys

base = sys.argv[1].rstrip("/")
clips = json.load(open("test/clips.json", encoding="utf-8"))


def post(route, path, **fields):
    args = ["curl", "-s", "-m", "300", "-w", "\n%{http_code}", "-F", f"file=@{path}"]
    for k, v in fields.items():
        args += ["-F", f"{k}={v}"]
    out = subprocess.run(args + [base + route], capture_output=True, text=True).stdout
    body, code = out.rsplit("\n", 1)
    return int(code), body


def norm(t):
    t = re.sub(r"[ً-ْـ]", "", t)  # harakat, tatweel
    t = re.sub(r"[^\w\s']", " ", t.lower())
    return " ".join(t.split())


same = 0
for path, c in clips.items():
    lang = "ar" if "fleurs_ar" in path else "en"
    code, body = post("/transcribe", path, language=lang)
    r = json.loads(body)
    ok = norm(r["text"]) == norm(c["arena_hyp"])
    same += ok
    if not ok:
        print(f"  differs from the Arena run: {path}\n    api  : {r['text']}\n    arena: {c['arena_hyp']}")
    code2, body2 = post("/transcribe", path)
    r2 = json.loads(body2)
    print(f"{path:<32} {code} forced={lang} same_as_arena={ok} | auto: {code2} language={r2['language']} "
          f"same_as_forced={norm(r2['text']) == norm(r['text'])} {r2['processing_time']} s")
print(f"{same}/{len(clips)} forced-language outputs equal the Arena run's after light normalisation")

code, body = post("/transcribe", "test/long_mixed.wav")
long = json.loads(body)
print("long_mixed", code, "duration", long["duration"], "language", long["language"], "pieces", len(long["segments"]),
      "time", long["processing_time"])
for s in long["segments"]:
    print(f"   {s['start']:7.2f}-{s['end']:7.2f} {s['language']} {s['text'][:70]}")
code, body = post("/transcribe", "test/long_stereo_44k.wav")
st = json.loads(body)
print("stereo 44.1 kHz copy", code, "same text as mono:", norm(st["text"]) == norm(long["text"]), "duration", st["duration"])
code, body = post("/transcribe", "test/fleurs_ar_1.m4a", language="ar")
m4a = json.loads(body)
print("m4a (ffmpeg path)", code, "same as wav:", norm(m4a["text"]) == norm(clips["test/fleurs_ar_1.wav"]["arena_hyp"]),
      m4a["text"][:60])
print("playlist pointing at a local file:", post("/transcribe", "test/attack.m3u8"))
print("bad language:", post("/transcribe", "test/fleurs_ar_0.wav", language="fr"))
open("test/empty.wav", "wb").close()
print("empty file:", post("/transcribe", "test/empty.wav"))
print("openai json:", post("/v1/audio/transcriptions", "test/librispeech_clean_0.wav", model="whisper-1"))
print("openai text:", post("/v1/audio/transcriptions", "test/librispeech_clean_0.wav", response_format="text"))
code, body = post("/v1/audio/transcriptions", "test/long_mixed.wav", response_format="verbose_json", language="ar")
v = json.loads(body)
print("openai verbose_json:", code, v["language"], v["duration"], len(v["segments"]), "segments")
