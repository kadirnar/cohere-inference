"""Public speech-to-text endpoint for Cohere-AR/cohere-transcribe-ar-sft-v9-full-s86k on this A10, without a key.

    GET  /                          test page: upload a file or record from the microphone, see the transcript
    GET  /info                      model, limits and usage as JSON
    GET  /health                    liveness and queue length
    POST /transcribe                multipart form: file (any audio ffmpeg reads), language = auto | ar | en
                                    (default auto), punctuation = true | false (default true, used with a language),
                                    level = true | false (input level normalisation, default true), reference
                                    (optional text; the response then carries the WER against it)
    POST /v1/audio/transcriptions   OpenAI-style form: file, model (ignored), language, response_format = json | text |
                                    verbose_json
    GET  /docs                      interactive API docs

The model card's protocol, as in the evaluations: 16 kHz mono, audio up to 30 s in one piece and longer audio cut at
pauses (Silero speech regions) into pieces of at most 25 s, greedy decoding, and the model writes its own language
header unless a language is given. Before that, the input level is normalised (see the LEVEL constants; level=false
takes the plain mean of the channels instead, as the Arabic Arena feeds the model). The text is the model's own;
normalized_text is the same text under the Arabic Arena's scoring normaliser (textnorm), for matching and WER.
Requests are prepared in parallel threads and their pieces decoded together in shared GPU batches (Batcher); at
most MAX_WAITING are in flight, the rest get 503. Audio and text are kept in memory only (temporary files go to TMPDIR,
a RAM disk in start.sh, and are deleted with the request); the log has timings, sizes and uvicorn's access lines
(client address, route, status), no audio or text.

    MODEL=Cohere-AR/cohere-transcribe-ar-sft-v9-full-s86k PORT=8086 python server.py
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

import textnorm

MODEL_ID = "Cohere-AR/cohere-transcribe-ar-sft-v9-full-s86k"
REVISION = "3ec483a7134410c26537dd975dc62b6a2c8636a6"
# MODEL: the Hub id (gated weights: HF_TOKEN of an account with access; loaded at REVISION) or a local copy of
# that commit
MODEL = os.environ.get("MODEL", MODEL_ID)
PORT = int(os.environ.get("PORT", "8086"))
MAX_MB = float(os.environ.get("MAX_MB", "100"))
MAX_MINUTES = float(os.environ.get("MAX_MINUTES", "30"))
MAX_WAITING = int(os.environ.get("MAX_WAITING", "128"))  # requests in flight before new ones get 503
SR = 16000
WHOLE_S = 30  # audio up to this long goes to the model in one piece, as the card says and the Arena runs it
PIECE_S = 25  # longer audio is cut at pauses into pieces of at most this long
# Pieces of all waiting requests are decoded together: at most MAX_BATCH pieces and BATCH_SECONDS of (padded) audio
# per GPU batch, after waiting BATCH_WAIT_S for requests that arrive at the same moment.
BATCH_SECONDS = 600
MAX_BATCH = 32
BATCH_WAIT_S = 0.005
MAX_NEW_TOKENS = 448  # the model card's value (the Arabic Arena runs it with 440)
LANGUAGES = ("auto", "ar", "en")
# Input level (on by default, level=false turns it off), measured from the signal alone, so the only network a request
# runs is the ASR model (and Silero for audio over 30 s, as without it). A channel's speech level is the RMS of its
# loudest 20 % of 50 ms frames; a channel whose loudest and quietest 20 % of frames are less than 12 dB apart (silence,
# steady noise or tones) has no speech to level and is never amplified. A two-channel call: each channel's speech is
# brought to -20 dBFS (at most +60 dB) and the channels are summed, so both sides are equally loud. Mono (and copies of
# one channel): left exactly as it is unless its speech is under -45 dBFS, where WER starts to suffer (the feature
# extractor's dither floor), and then brought to -20 dBFS. The result is kept under 0.99 full scale.
FRAME = SR // 20
TOP_SHARE, SPEECH_RANGE_DB = 0.2, 12.0
LEVEL_DBFS, LEVEL_MAX_GAIN_DB, QUIET_DBFS = -20.0, 60.0, -45.0
# Demuxers ffmpeg may use; playlist and concat formats (hls, dash, concat, ...) stay out, so an upload cannot make
# ffmpeg open other files or URLs.
DEMUXERS = "wav,w64,mp3,flac,ogg,mov,matroska,aac,amr,amrnb,amrwb,aiff,caf,au,asf,wv,ape,gsm,mpeg,mpegts,avi,flv"
HEADER = re.compile(r"<\|emo:[^|>]*\|>\s*<\|([a-z]{2})\|>")
ARABIC = re.compile(r"[؀-ۿ]")

log = logging.getLogger("s86k")


class BadAudio(ValueError):
    pass


# ---- audio ---------------------------------------------------------------------------------------------------------


def ffmpeg_decode(data: bytes) -> tuple[np.ndarray, int]:
    """(samples, channels) float32 and the sample rate of the first audio stream, decoded by ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".audio") as f:
        f.write(data)
        f.flush()
        opts = ["-v", "error", "-protocol_whitelist", "file", "-format_whitelist", DEMUXERS]
        probe = subprocess.run(["ffprobe", *opts, "-select_streams", "a:0", "-show_entries",
                                "stream=channels,sample_rate", "-of", "json", f.name], capture_output=True, timeout=60)
        try:
            stream = json.loads(probe.stdout)["streams"][0]
            channels, rate = int(stream["channels"]), int(stream["sample_rate"])
        except (ValueError, KeyError, IndexError):
            raise BadAudio("could not read an audio stream from the file") from None
        out = subprocess.run(["ffmpeg", "-nostdin", *opts, "-i", f.name, "-map", "0:a:0", "-t", str(MAX_MINUTES * 60 + 1),
                              "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"], capture_output=True, timeout=180)
    if out.returncode:
        raise BadAudio("could not decode the audio")
    x = np.frombuffer(out.stdout, dtype=np.float32)
    return x[: len(x) // channels * channels].reshape(-1, channels), rate


def load_audio(data: bytes) -> np.ndarray:
    """(samples, channels) float32 at 16 kHz, resampled with soxr."""
    try:
        with sf.SoundFile(io.BytesIO(data)) as f:
            if f.frames > (MAX_MINUTES * 60 + 1) * f.samplerate:
                raise BadAudio(f"audio too long (max {MAX_MINUTES:g} minutes)")
            x, rate = f.read(dtype="float32", always_2d=True), f.samplerate
    except BadAudio:
        raise
    except Exception:  # not a format libsndfile reads: m4a, aac, webm, ...
        x, rate = ffmpeg_decode(data)
    x = soxr.resample(x, rate, SR) if rate != SR else x
    return np.ascontiguousarray(x, dtype=np.float32)


# ---- model ---------------------------------------------------------------------------------------------------------


class Engine:
    def __init__(self, model_dir: str):
        from transformers import CohereAsrForConditionalGeneration, CohereAsrProcessor

        kw = {} if os.path.isdir(model_dir) else {"revision": REVISION}
        self.processor = CohereAsrProcessor.from_pretrained(model_dir, **kw)
        self.model = CohereAsrForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16, **kw).cuda().eval()
        self._local = threading.local()

    @property
    def vad(self):
        """Silero for this thread (it keeps state, so concurrent requests each need their own)."""
        from silero_vad import load_silero_vad

        if not hasattr(self._local, "vad"):
            self._local.vad = load_silero_vad(onnx=True)
        return self._local.vad

    def pieces(self, x: np.ndarray) -> list[tuple[int, int]]:
        """(start, end) samples: the whole audio up to 30 s, otherwise Silero speech regions packed into pieces of at
        most 25 s, 0.2 s around each."""
        from silero_vad import get_speech_timestamps

        if len(x) <= WHOLE_S * SR:
            return [(0, len(x))]
        self.vad.reset_states()
        regions = get_speech_timestamps(torch.from_numpy(np.ascontiguousarray(x)), self.vad, sampling_rate=SR,
                                        threshold=0.5, min_speech_duration_ms=200, min_silence_duration_ms=300,
                                        max_speech_duration_s=PIECE_S - 1)
        out, start, end = [], None, None
        for r in regions:
            if start is not None and r["end"] - start > (PIECE_S - 0.4) * SR:
                out.append((max(0, start - SR // 5), min(len(x), end + SR // 5)))
                start = None
            start, end = (r["start"] if start is None else start), r["end"]
        if start is not None:
            out.append((max(0, start - SR // 5), min(len(x), end + SR // 5)))
        return out

    @staticmethod
    def speech_level(ch: np.ndarray) -> float | None:
        """dBFS of the loudest TOP_SHARE of 50 ms frames; None when there is no speech to level (under 0.25 s, digital
        silence, or loudest and quietest frames under SPEECH_RANGE_DB apart)."""
        n = len(ch) // FRAME
        if n < 5:
            return None
        e = np.sort((ch[: n * FRAME].astype(np.float64).reshape(n, FRAME) ** 2).mean(axis=1))
        k = max(1, int(n * TOP_SHARE))
        top, low = float(e[-k:].mean()), float(e[:k].mean())
        if top <= 1e-12 or 10 * np.log10(top / max(low, 1e-20)) < SPEECH_RANGE_DB:
            return None
        return 10 * float(np.log10(top))

    @classmethod
    def level(cls, x: np.ndarray) -> np.ndarray:
        """Mono for the model: see the LEVEL constants. Returns the plain mean for audio that needs no change."""
        def gained(ch, lv):
            return ch * float(10 ** (min(LEVEL_MAX_GAIN_DB, LEVEL_DBFS - lv) / 20))

        chans = [x[:, c] for c in range(x.shape[1])]
        if len(chans) > 1 and min(chans[0].std(), chans[1].std()) > 0 and np.corrcoef(chans[0], chans[1])[0, 1] > 0.9:
            chans = [x.mean(axis=1)]  # copies of one channel
        if len(chans) == 1:
            ch = chans[0].astype(np.float32)
            lv = cls.speech_level(ch)
            if lv is None or lv >= QUIET_DBFS:
                return ch
            mix = gained(ch, lv)
        else:
            mix = np.sum([ch if (lv := cls.speech_level(ch)) is None else gained(ch, lv) for ch in chans], axis=0)
        return (mix / max(1.0, float(np.abs(mix).max()) / 0.99)).astype(np.float32)

    def prepare(self, x: np.ndarray, level: bool = True) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """(mono audio, pieces) of (samples, channels) audio: levelled, or the mean of the channels with level=False."""
        x = self.level(x) if level else x.mean(axis=1).astype(np.float32)
        return x, self.pieces(x)

    @staticmethod
    def batches(lengths: list[int]) -> list[list[int]]:
        """Indices grouped into GPU batches, shortest first, within MAX_BATCH pieces and BATCH_SECONDS padded."""
        out, cur = [], []
        for i in sorted(range(len(lengths)), key=lambda i: lengths[i]):
            if cur and (lengths[i] * (len(cur) + 1) > BATCH_SECONDS * SR or len(cur) >= MAX_BATCH):
                out.append(cur)
                cur = []
            cur.append(i)
        return out + [cur] if cur else out

    @torch.no_grad()
    def decode(self, audio: list[np.ndarray], language: str, punctuation: bool) -> list[tuple[str, str]]:
        """(text, language) of each piece, in one generate call."""
        auto = language == "auto"
        inputs = self.processor(audio=audio, sampling_rate=SR, language="ar" if auto else language,
                                punctuation=punctuation, return_tensors="pt")
        if auto:
            inputs["decoder_input_ids"] = inputs["decoder_input_ids"][:, :4]  # up to <|emo:undefined|>
        inputs.pop("audio_chunk_index", None)  # pieces are within the feature extractor's 30 s, so never cut
        inputs = {k: v.cuda().to(torch.bfloat16) if v.is_floating_point() else v.cuda() for k, v in inputs.items()}
        gen = self.model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
        assert len(gen) == len(audio)
        tok, out = self.processor.tokenizer, []
        for raw, text in zip(tok.batch_decode(gen), tok.batch_decode(gen, skip_special_tokens=True)):
            text = " ".join(text.split())
            if auto:
                m = HEADER.search(raw)
                out.append((text, m.group(1) if m else ("ar" if ARABIC.search(text) else "en")))
            else:
                out.append((text, language))
        return out

    def __call__(self, x: np.ndarray, language: str = "auto", punctuation: bool = True, level: bool = True) -> list[dict]:
        """Segments of one recording ((samples, channels)), decoded on its own (the scripts' entry point)."""
        x, spans = self.prepare(x, level)
        audio = [x[s:e] for s, e in spans]
        res: list[tuple[str, str]] = [("", language)] * len(spans)
        for idx in self.batches([len(a) for a in audio]):
            for i, r in zip(idx, self.decode([audio[i] for i in idx], language, punctuation)):
                res[i] = r
        return segments(spans, res)


def segments(spans: list[tuple[int, int]], results: list[tuple[str, str]]) -> list[dict]:
    return [{"start": round(s / SR, 2), "end": round(e / SR, 2), "language": lang, "text": text}
            for (s, e), (text, lang) in zip(spans, results)]


class Batcher:
    """Decodes the pieces of every waiting request together: one loop owns the GPU and takes all requests queued while
    it was busy, groups their pieces by prompt (language, punctuation) and runs them in shared batches."""

    def __init__(self, engine: Engine):
        self.engine, self.queue = engine, asyncio.Queue()

    async def submit(self, audio: list[np.ndarray], language: str, punctuation: bool) -> list[tuple[str, str]]:
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((audio, language, punctuation, future))
        return await future

    async def run(self) -> None:
        while True:
            jobs = [await self.queue.get()]
            await asyncio.sleep(BATCH_WAIT_S)
            while not self.queue.empty():
                jobs.append(self.queue.get_nowait())
            results = [[None] * len(audio) for audio, _, _, _ in jobs]
            groups: dict[tuple[str, bool], list[tuple[int, int]]] = {}
            for j, (audio, language, punctuation, _) in enumerate(jobs):
                for k in range(len(audio)):
                    groups.setdefault((language, punctuation), []).append((j, k))
            left = [len(audio) for audio, _, _, _ in jobs]
            for j, n in enumerate(left):
                if n == 0:  # no speech found: nothing to decode
                    jobs[j][3].set_result([])
            try:
                for (language, punctuation), items in groups.items():
                    audio = [jobs[j][0][k] for j, k in items]
                    for idx in self.engine.batches([len(a) for a in audio]):
                        out = await asyncio.to_thread(self.engine.decode, [audio[i] for i in idx], language, punctuation)
                        for i, r in zip(idx, out):
                            j, k = items[i]
                            results[j][k] = r
                            left[j] -= 1
                            if left[j] == 0 and not jobs[j][3].done():  # answer each request as soon as it is complete
                                jobs[j][3].set_result(results[j])
                log.info("batched requests=%d pieces=%d", len(jobs), sum(len(r) for r in results))
            except Exception as e:  # noqa: BLE001 (passed on to every waiting request)
                for _, _, _, future in jobs:
                    if not future.done():
                        future.set_exception(e)


# ---- app -----------------------------------------------------------------------------------------------------------

engine: Engine | None = None
batcher: Batcher | None = None
waiting = 0


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine, batcher
    t = time.time()
    engine = await asyncio.to_thread(Engine, MODEL)
    await asyncio.to_thread(engine, np.zeros((SR, 1), dtype=np.float32))  # warm-up
    batcher = Batcher(engine)
    task = asyncio.create_task(batcher.run())
    log.info("ready: %s (%s) from %s in %.0f s", MODEL_ID, REVISION[:10], MODEL, time.time() - t)
    yield
    task.cancel()


app = FastAPI(title="Cohere-AR sft-v9-full-s86k speech-to-text", version=REVISION[:10], lifespan=lifespan,
              description="Arabic, English and code-switched speech to text with "
                          f"[{MODEL_ID}](https://huggingface.co/{MODEL_ID}) (commit {REVISION[:10]}). No key needed.")


@app.middleware("http")
async def limit_size(request: Request, call_next):
    n = request.headers.get("content-length", "")
    if n.isdigit() and int(n) > MAX_MB * 1e6 + 1e5:
        return JSONResponse({"detail": f"file too large (max {MAX_MB:g} MB)"}, status_code=413)
    return await call_next(request)


async def run(file: UploadFile, language: str, punctuation: bool, level: bool = True, reference: str = "") -> dict:
    global waiting
    if language not in LANGUAGES:
        raise HTTPException(400, f"language must be one of: {', '.join(LANGUAGES)}")
    t0 = time.time()
    data = await file.read()
    await file.close()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > MAX_MB * 1e6:
        raise HTTPException(413, f"file too large (max {MAX_MB:g} MB)")
    try:
        x = await asyncio.to_thread(load_audio, data)
    except BadAudio as e:
        raise HTTPException(400, str(e)) from None
    except subprocess.TimeoutExpired:
        raise HTTPException(400, "decoding the audio took too long") from None
    duration = len(x) / SR
    if duration > MAX_MINUTES * 60:
        raise HTTPException(413, f"audio too long (max {MAX_MINUTES:g} minutes)")
    if duration < 0.1:
        raise HTTPException(400, "audio too short")
    if batcher is None:
        raise HTTPException(503, "model is loading, try again shortly")
    if waiting >= MAX_WAITING:
        raise HTTPException(503, "busy, try again in a minute")
    waiting += 1
    try:
        mono, spans = await asyncio.to_thread(engine.prepare, x, level)
        t1 = time.time()
        segs = segments(spans, await batcher.submit([mono[s:e] for s, e in spans], language, punctuation))
    finally:
        waiting -= 1
    t2 = time.time()
    seconds: dict[str, float] = {}
    for s in segs:
        s["normalized_text"] = textnorm.normalize(s["text"], s["language"] if s["language"] in ("ar", "en") else None)
        if s["text"]:
            seconds[s["language"]] = seconds.get(s["language"], 0) + s["end"] - s["start"]
    log.info("transcribed audio_s=%.1f bytes=%d channels=%d language=%s level=%s pieces=%d prep_s=%.2f queue_gpu_s=%.2f "
             "total_s=%.2f", duration, len(data), x.shape[1], language, level, len(segs), t1 - t0, t2 - t1, t2 - t0)
    text = " ".join(s["text"] for s in segs if s["text"])
    out = {"model": MODEL_ID, "revision": REVISION,
           "language": max(seconds, key=seconds.get) if seconds else (language if language != "auto" else ""),
           "duration": round(duration, 2), "processing_time": round(t2 - t0, 2), "level": level, "text": text,
           "normalized_text": " ".join(s["normalized_text"] for s in segs if s["normalized_text"]),
           "segments": segs}
    if reference.strip():
        out["wer"] = textnorm.wer(reference, text)
    return out


PAGE = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def page():
    return PAGE


@app.get("/info")
def info():
    return {"model": MODEL_ID, "revision": REVISION, "auth": "none",
            "routes": {"POST /transcribe": "multipart: file, language = auto | ar | en, punctuation = true | false, "
                                           "level = true | false (input level normalisation), reference (optional "
                                           "text: the response then has the WER)",
                       "POST /v1/audio/transcriptions": "OpenAI-style: file, language, response_format = json | text | "
                                                        "verbose_json",
                       "GET /": "test page", "GET /docs": "interactive API docs", "GET /health": "liveness"},
            "limits": {"max_file_mb": MAX_MB, "max_audio_minutes": MAX_MINUTES, "queue": MAX_WAITING},
            "example": "curl -F file=@call.wav -F language=auto <this url>/transcribe"}


@app.get("/health")
def health():
    return {"status": "ok" if engine else "loading", "waiting": waiting}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form("auto"), punctuation: bool = Form(True),
                     level: bool = Form(True), reference: str = Form("")):
    return await run(file, language, punctuation, level, reference)


@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(file: UploadFile = File(...), model: str = Form(MODEL_ID), language: str = Form(""),
                                response_format: str = Form("json")):
    if response_format not in ("json", "text", "verbose_json"):
        raise HTTPException(400, "response_format must be json, text or verbose_json")
    r = await run(file, language if language in ("ar", "en") else "auto", True)
    if response_format == "text":
        return PlainTextResponse(r["text"])
    if response_format == "verbose_json":
        return {"task": "transcribe", "language": r["language"], "duration": r["duration"], "text": r["text"],
                "segments": [{"id": i, "start": s["start"], "end": s["end"], "text": s["text"]}
                             for i, s in enumerate(r["segments"])]}
    return {"text": r["text"]}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.Formatter.converter = time.gmtime
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
