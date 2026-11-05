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
HTTP worker processes (WORKERS) parse the requests, decode, level and dither the audio and write the text; one GPU
process holds the model (see main()). On the GPU (kernels/pipeline.py) the pieces of all waiting requests are encoded
together in chunks and decoded with continuous batching: a piece joins the decoder batch as soon as it is encoded and
leaves it as soon as it ends, and each request is answered when its last piece ends. At most MAX_WAITING requests are in
flight per worker, the rest get 503. The model runs through the CUDA kernels in kernels/csrc, its GEMMs, decoder
weights and attention caches in int8 unless INT8=0 (FAST=0: transformers' own code, in shared batches, one process);
the text is the same as plain transformers gives up to rounding (INT8: up to quantization, see README). Audio and text are kept in memory
only (temporary files go to TMPDIR, a RAM disk in start.sh, and are deleted with the request); the log has timings,
sizes and uvicorn's access lines (client address, route, status), no audio or text.

    MODEL=Cohere-AR/cohere-transcribe-ar-sft-v9-full-s86k PORT=8086 python server.py
"""

from __future__ import annotations

import os

# One CPU thread per library call: the parallelism is across requests (each in its own thread), and per-call thread
# pools in every request thread of every HTTP worker ran into the process's thread limit ("libgomp: Thread creation
# failed") under 100 simultaneous requests. Set before numpy and torch are imported; the environment can override.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import asyncio
import gc
import io
import json
import logging
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
from kernels.features import dither

MODEL_ID = "Cohere-AR/cohere-transcribe-ar-sft-v9-full-s86k"
REVISION = "3ec483a7134410c26537dd975dc62b6a2c8636a6"
# MODEL: the Hub id (gated weights: HF_TOKEN of an account with access; loaded at REVISION) or a local copy of
# that commit
MODEL = os.environ.get("MODEL", MODEL_ID)
PORT = int(os.environ.get("PORT", "8086"))
MAX_MB = float(os.environ.get("MAX_MB", "100"))
MAX_MINUTES = float(os.environ.get("MAX_MINUTES", "30"))
MAX_WAITING = int(os.environ.get("MAX_WAITING", "128"))  # requests in flight (per HTTP worker) before new ones get 503
SR = 16000
WHOLE_S = 30  # audio up to this long goes to the model in one piece, as the card says and the Arena runs it
PIECE_S = 25  # longer audio is cut at pauses into pieces of at most this long
# FAST (default): features, encoder and decoder through the CUDA kernels of kernels/csrc, scheduled by
# kernels/pipeline.py (GPU log-mel, packed encoder in chunks shortest first, continuous-batching decoder with SLOTS
# slots). FAST=0: transformers' own code (generate), the pieces of all waiting requests in shared batches of at most
# MAX_BATCH pieces and BATCH_SECONDS of padded audio, after waiting BATCH_WAIT_S for requests arriving together.
FAST = os.environ.get("FAST", "1") != "0"
SLOTS = int(os.environ.get("SLOTS", "128"))  # pieces decoded at once (more wait for a free slot)
# INT8 (FAST only): the encoder's large GEMMs in int8 (W8A8, kernels/csrc/gemm_int8.cu), about 1.3x faster encoding.
# On FLEURS ar_eg / en_us with the base model the WER was unchanged (6.91 -> 6.88 %, 6.39 -> 6.33 %) and 93-94 % of
# transcripts identical to bf16's; INT8=0 keeps the model's own bf16 arithmetic.
INT8 = os.environ.get("INT8", "1") != "0"
# FFN_SPARSE (INT8, opt-in): a file of 2:4-pruned weights of the encoder feed-forwards' second linear layers, made for
# this model by `python -m kernels.sparse` (kernels/sparse.py): the encoder ~3.5 % faster for ~0.1-0.5 points of WER
FFN_SPARSE = os.environ.get("FFN_SPARSE") or None
# With FAST, WORKERS HTTP worker processes (default 8) parse the requests, decode and level the audio and write the text,
# and one GPU process (this file with --gpu) runs the model for all of them over a local socket: in one process the
# HTTP loop, the request threads and the threads launching the GPU work slowed each other down through the GIL.
# WORKERS=0: everything in one process.
WORKERS = int(os.environ.get("WORKERS", "8")) if FAST else 0
BATCH_SECONDS = 600
MAX_BATCH = 32
BATCH_WAIT_S = 0.005
HOP = 160  # samples per log-mel frame
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


class AudioTooLong(BadAudio):
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
                raise AudioTooLong(f"audio too long (max {MAX_MINUTES:g} minutes)")
            x, rate = f.read(dtype="float32", always_2d=True), f.samplerate
    except BadAudio:
        raise
    except Exception:  # not a format libsndfile reads: m4a, aac, webm, ...
        x, rate = ffmpeg_decode(data)
    x = soxr.resample(x, rate, SR) if rate != SR else x
    return np.ascontiguousarray(x, dtype=np.float32)


# ---- model ---------------------------------------------------------------------------------------------------------


class Text:
    """What a request needs besides the model: the processor (tokenizer: prompts and text), the input level and the
    packing of speech regions into pieces. The GPU process's Engine is one; each HTTP worker process has its own."""

    def __init__(self, model_dir: str):
        from transformers import CohereAsrProcessor

        self.kw = {} if os.path.isdir(model_dir) else {"revision": REVISION}
        self.processor = CohereAsrProcessor.from_pretrained(model_dir, **self.kw)
        self._prompts: dict[tuple[str, bool], torch.Tensor] = {}

    @staticmethod
    def pack(regions: list[dict], n: int) -> list[tuple[int, int]]:
        """Speech regions (Silero's, in samples) of audio of n samples packed into pieces of at most PIECE_S, 0.2 s
        around each."""
        out, start, end = [], None, None
        for r in regions:
            if start is not None and r["end"] - start > (PIECE_S - 0.4) * SR:
                out.append((max(0, start - SR // 5), min(n, end + SR // 5)))
                start = None
            start, end = (r["start"] if start is None else start), r["end"]
        if start is not None:
            out.append((max(0, start - SR // 5), min(n, end + SR // 5)))
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

    def mono(self, x: np.ndarray, level: bool = True) -> np.ndarray:
        """(samples, channels) to the model's mono: levelled, or the plain mean of the channels with level=False."""
        return self.level(x) if level else x.mean(axis=1).astype(np.float32)

    def prompt(self, language: str, punctuation: bool) -> torch.Tensor:
        """The decoder prompt: the full one for a forced language, up to <|emo:undefined|> for language=auto."""
        key = (language, punctuation)
        if key not in self._prompts:
            ids = self.processor(audio=[np.zeros(SR, np.float32)], sampling_rate=SR, punctuation=punctuation,
                                 language="ar" if language == "auto" else language, return_tensors="pt")
            self._prompts[key] = ids["decoder_input_ids"][0, :4] if language == "auto" else ids["decoder_input_ids"][0]
        return self._prompts[key]

    def texts(self, gen: list, language: str) -> list[tuple[str, str]]:
        """(text, language) of each piece's token ids (prompt included); with language=auto the language is the
        model's own header (or, without one, Arabic if the text has Arabic letters)."""
        auto = language == "auto"
        tok, out = self.processor.tokenizer, []
        for raw, text in zip(tok.batch_decode(gen), tok.batch_decode(gen, skip_special_tokens=True)):
            text = " ".join(text.split())
            if auto:
                m = HEADER.search(raw)
                out.append((text, m.group(1) if m else ("ar" if ARABIC.search(text) else "en")))
            else:
                out.append((text, language))
        return out


class Engine(Text):
    """The model on the GPU: the CUDA pipeline (FAST) or transformers' generate, and Silero for long audio."""

    def __init__(self, model_dir: str):
        from transformers import CohereAsrForConditionalGeneration

        super().__init__(model_dir)
        self.model = CohereAsrForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16,
                                                                       **self.kw).cuda().eval()
        self.pipe = None
        if FAST:
            from kernels.pipeline import Pipeline

            self.pipe = Pipeline(self.model, self.processor.feature_extractor, slots=SLOTS,
                                 max_new_tokens=MAX_NEW_TOKENS, int8=INT8, sparse=FFN_SPARSE)
        from kernels.silero_parallel import ParallelSilero

        self.vad = ParallelSilero("cuda")  # Silero's own probabilities and regions, the network run in parallel

    def pieces(self, x: np.ndarray) -> list[tuple[int, int]]:
        """(start, end) samples: the whole audio up to 30 s, otherwise Silero speech regions packed into pieces of at
        most 25 s, 0.2 s around each."""
        if len(x) <= WHOLE_S * SR:
            return [(0, len(x))]
        regions = self.vad.speech_timestamps(x, threshold=0.5, min_speech_duration_ms=200, min_silence_duration_ms=300,
                                             max_speech_duration_s=PIECE_S - 1)
        return self.pack(regions, len(x))

    def features(self, audio: np.ndarray) -> torch.Tensor:
        """Normalised log-mel frames (frames, 128) of one piece, computed on the CPU in the request's own thread."""
        return self.processor.feature_extractor([audio], sampling_rate=SR, return_tensors="pt")["input_features"][0]

    def prepare(self, x: np.ndarray, level: bool = True) -> tuple[list[tuple[int, int]], list]:
        """(pieces, their model input) of (samples, channels) audio, levelled (or the mean of the channels with
        level=False): the dithered samples for the GPU pipeline (FAST), the log-mel frames otherwise."""
        x = self.mono(x, level)
        spans = self.pieces(x)
        if self.pipe:
            return spans, [self.pipe.prepare(x[s:e]) for s, e in spans]
        return spans, [self.features(x[s:e]) for s, e in spans]

    @staticmethod
    def batches(frames: list[int]) -> list[list[int]]:
        """Indices grouped into GPU batches, shortest first, within MAX_BATCH pieces and BATCH_SECONDS padded."""
        out, cur = [], []
        for i in sorted(range(len(frames)), key=lambda i: frames[i]):
            if cur and (frames[i] * HOP * (len(cur) + 1) > BATCH_SECONDS * SR or len(cur) >= MAX_BATCH):
                out.append(cur)
                cur = []
            cur.append(i)
        return out + [cur] if cur else out

    @torch.no_grad()
    def decode(self, feats: list[torch.Tensor], language: str, punctuation: bool) -> list[tuple[str, str]]:
        """(text, language) of each piece with transformers' generate (FAST=0), decoded together: the pieces' frames
        padded with zeros and masked, as the processor pads a batch; greedy."""
        n = max(len(f) for f in feats)
        x = torch.zeros(len(feats), n, feats[0].shape[1])
        mask = torch.zeros(len(feats), n, dtype=torch.bool)
        for i, f in enumerate(feats):
            x[i, :len(f)], mask[i, :len(f)] = f, True
        x, mask, ids = x.cuda().to(torch.bfloat16), mask.cuda(), self.prompt(language, punctuation)
        gen = self.model.generate(input_features=x, attention_mask=mask, do_sample=False,
                                  decoder_input_ids=ids.repeat(len(feats), 1).cuda(), max_new_tokens=MAX_NEW_TOKENS)
        assert len(gen) == len(feats)
        return self.texts(list(gen), language)

    def __call__(self, x: np.ndarray, language: str = "auto", punctuation: bool = True, level: bool = True) -> list[dict]:
        """Segments of one recording ((samples, channels)), decoded on its own (the scripts' entry point)."""
        spans, feats = self.prepare(x, level)
        if self.pipe:
            return segments(spans, self.texts(self.pipe.submit(feats, self.prompt(language, punctuation)).result(),
                                              language))
        res: list[tuple[str, str]] = [("", language)] * len(spans)
        for idx in self.batches([len(f) for f in feats]):
            for i, r in zip(idx, self.decode([feats[i] for i in idx], language, punctuation)):
                res[i] = r
        return segments(spans, res)

    def warm_up(self) -> None:
        """Decode noise with each prompt in batches of 1 and 2 and cut 40 s of it, so the first requests do not set up
        cuBLAS, cuDNN or the tokenizer."""
        rng = np.random.default_rng(0)
        for language in ("auto", "ar"):
            for batch in (1, 2):
                audio = (rng.standard_normal((5 * SR * batch, 1)) * 0.05).astype(np.float32)
                if self.pipe:
                    pieces = [self.pipe.prepare(a) for a in np.split(audio[:, 0], batch)]
                    self.texts(self.pipe.submit(pieces, self.prompt(language, True)).result(), language)
                else:
                    self.decode([self.features(a) for a in np.split(audio[:, 0], batch)], language, True)
        self.pieces((rng.standard_normal(40 * SR) * 0.05).astype(np.float32))
        if self.pipe:  # a burst of pieces of every length, so the first real one finds the GPU memory pools grown
            burst = [self.pipe.prepare((rng.standard_normal(int(s * SR)) * 0.05).astype(np.float32))
                     for s in np.linspace(2, WHOLE_S, 64)]
            self.pipe.submit(burst, self.prompt("auto", True), [8] * len(burst)).result()


def segments(spans: list[tuple[int, int]], results: list[tuple[str, str]]) -> list[dict]:
    return [{"start": round(s / SR, 2), "end": round(e / SR, 2), "language": lang, "text": text}
            for (s, e), (text, lang) in zip(spans, results)]


class Batcher:
    """FAST=0: decodes the pieces of every waiting request together with transformers' generate: one loop owns the GPU
    and takes all requests queued while it was busy, groups their pieces by prompt (language, punctuation) and runs
    them in shared batches. (With FAST the GPU pipeline does its own batching.)"""

    def __init__(self, engine: Engine):
        self.engine, self.queue = engine, asyncio.Queue()
        self.gpu = ThreadPoolExecutor(1, thread_name_prefix="gpu")  # not shared with the requests' preparation

    async def submit(self, feats: list, language: str, punctuation: bool) -> list[tuple[str, str]]:
        if self.engine.pipe:
            tokens = await asyncio.wrap_future(self.engine.pipe.submit(feats, self.engine.prompt(language, punctuation)))
            return await asyncio.to_thread(self.engine.texts, tokens, language)
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((feats, language, punctuation, future))
        return await future

    async def run(self) -> None:
        while True:
            jobs = [await self.queue.get()]
            await asyncio.sleep(BATCH_WAIT_S)
            while not self.queue.empty():
                jobs.append(self.queue.get_nowait())
            results = [[None] * len(feats) for feats, _, _, _ in jobs]
            groups: dict[tuple[str, bool], list[tuple[int, int]]] = {}
            for j, (feats, language, punctuation, _) in enumerate(jobs):
                for k in range(len(feats)):
                    groups.setdefault((language, punctuation), []).append((j, k))
            left = [len(feats) for feats, _, _, _ in jobs]
            for j, n in enumerate(left):
                if n == 0:  # no speech found: nothing to decode
                    jobs[j][3].set_result([])
            try:
                for (language, punctuation), items in groups.items():
                    feats = [jobs[j][0][k] for j, k in items]
                    for idx in self.engine.batches([len(f) for f in feats]):
                        out = await asyncio.get_running_loop().run_in_executor(
                            self.gpu, self.engine.decode, [feats[i] for i in idx], language, punctuation)
                        for i, r in zip(idx, out):
                            j, k = items[i]
                            results[j][k] = r
                            left[j] -= 1
                            if left[j] == 0 and not jobs[j][3].done():  # answer each request as soon as it is complete
                                jobs[j][3].set_result(results[j])
                log.info("batched requests=%d pieces=%d", len(jobs), sum(len(r) for r in results))
            except Exception:  # noqa: BLE001 (logged once; each waiting request fails with its own error)
                log.exception("decoding failed: requests=%d", len(jobs))
                for _, _, _, future in jobs:
                    if not future.done():
                        future.set_exception(RuntimeError("decoding failed"))


# ---- GPU process and its clients ------------------------------------------------------------------------------------
# Messages over multiprocessing.connection (a Unix socket, an authkey from the parent): (kind, id, *args) in,
# ("ok" | "err", id, result) out. Kinds: "pieces" (levelled mono samples over WHOLE_S -> spans), "decode" (dithered
# pieces, language, punctuation -> token ids per piece).


def serve_gpu(address: str) -> None:
    """The GPU process: the Engine, answering the HTTP workers' requests until the parent process goes away."""
    from multiprocessing.connection import Listener

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.Formatter.converter = time.gmtime
    sys.setswitchinterval(0.0005)
    parent = os.getppid()
    t = time.time()
    gpu = Engine(MODEL)
    gpu.warm_up()
    gc.collect()
    gc.freeze()  # the model's and libraries' objects out of the collector's sight: its full passes paused the GPU threads
    vad_pool = ThreadPoolExecutor(4, thread_name_prefix="vad")
    listener = Listener(address, family="AF_UNIX", authkey=bytes.fromhex(os.environ["GPU_AUTHKEY"]))
    log.info("GPU process ready: %s from %s in %.0f s", MODEL_ID, MODEL, time.time() - t)

    def watch_parent():
        while os.getppid() == parent:
            time.sleep(1)
        os._exit(0)

    def handle(conn):
        lock = threading.Lock()

        def reply(kind, rid, result):
            try:
                with lock:
                    conn.send((kind, rid, result))
            except OSError:  # the worker went away (it is restarted); its requests are gone with it
                pass

        def answer(rid, fn):
            try:
                reply("ok", rid, fn())
            except Exception as e:  # noqa: BLE001 (the worker's request fails with it)
                log.exception("GPU request failed")
                reply("err", rid, repr(e))

        def decoded(rid, fut):
            if fut.exception() is not None:
                reply("err", rid, repr(fut.exception()))
            else:
                reply("ok", rid, fut.result())

        while True:
            try:
                kind, rid, *args = conn.recv()
            except (EOFError, OSError):
                return
            if kind == "pieces":
                x, = args
                vad_pool.submit(answer, rid, lambda x=x: gpu.pieces(x))
            elif kind == "decode":
                pieces, language, punctuation = args
                fut = gpu.pipe.submit(pieces, gpu.prompt(language, punctuation))
                fut.add_done_callback(lambda f, rid=rid: decoded(rid, f))

    threading.Thread(target=watch_parent, daemon=True).start()
    while True:
        threading.Thread(target=handle, args=(listener.accept(),), daemon=True).start()


class Remote:
    """An HTTP worker's connection to the GPU process: awaitable calls, answered by a reader thread."""

    def __init__(self, address: str):
        from multiprocessing.connection import Client

        self.conn = Client(address, family="AF_UNIX", authkey=bytes.fromhex(os.environ["GPU_AUTHKEY"]))
        self.lock = threading.Lock()
        self.pending: dict[int, tuple] = {}
        self.ids = 0
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while True:
            kind, rid, result = self.conn.recv()
            loop, fut = self.pending.pop(rid)
            if kind == "ok":
                loop.call_soon_threadsafe(lambda f=fut, r=result: f.done() or f.set_result(r))
            else:
                loop.call_soon_threadsafe(lambda f=fut, r=result: f.done() or f.set_exception(RuntimeError(r)))

    async def call(self, kind: str, *args):
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        with self.lock:
            self.ids += 1
            self.pending[self.ids] = (loop, fut)
            self.conn.send((kind, self.ids, *args))
        return await fut


# ---- app -----------------------------------------------------------------------------------------------------------

engine: Engine | None = None  # the model in this process (WORKERS=0)
batcher: Batcher | None = None
text: Text | None = None  # an HTTP worker's tokenizer and audio preparation (WORKERS > 0) ...
remote: Remote | None = None  # ... and its connection to the GPU process
waiting = 0


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine, batcher
    # request threads (decoding, levelling) beside the GPU threads: more of them, and a GIL handed over within 0.5 ms
    # so the GPU threads' launches are not held up by 5 ms slices of Python work
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(64, thread_name_prefix="request"))
    sys.setswitchinterval(0.0005)
    t = time.time()
    if os.environ.get("GPU_SOCKET"):  # an HTTP worker of the GPU process started by __main__
        global text, remote
        if not logging.getLogger().handlers:  # a spawned worker: the parent's logging setup is not inherited
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
            logging.Formatter.converter = time.gmtime
        text = await asyncio.to_thread(Text, MODEL)
        await asyncio.to_thread(text.texts, [text.prompt("auto", True)], "auto")  # tokenizer warmed up
        await asyncio.to_thread(warm_worker)
        remote = Remote(os.environ["GPU_SOCKET"])
        gc.collect()
        gc.freeze()
        log.info("HTTP worker %d ready", os.getpid())
        yield
        return
    engine = await asyncio.to_thread(Engine, MODEL)
    await asyncio.to_thread(engine.warm_up)
    gc.collect()
    gc.freeze()
    batcher = Batcher(engine)
    task = asyncio.create_task(batcher.run()) if not engine.pipe else None
    log.info("ready: %s (%s) from %s in %.0f s", MODEL_ID, REVISION[:10], MODEL, time.time() - t)
    yield
    if task:
        task.cancel()


app = FastAPI(title="Cohere-AR sft-v9-full-s86k speech-to-text", version=REVISION[:10], lifespan=lifespan,
              description="Arabic, English and code-switched speech to text with "
                          f"[{MODEL_ID}](https://huggingface.co/{MODEL_ID}) (commit {REVISION[:10]}). No key needed.")


class LimitSize:
    """413 for a declared body over MAX_MB before it is read, and the request's arrival time for the log. Plain ASGI:
    FastAPI's @app.middleware("http") (BaseHTTPMiddleware) doubled the time 100 simultaneous uploads took. In an HTTP
    worker, POST /transcribe is answered here when its form is a plain one (fast_form): FastAPI's form parsing,
    validation and response encoding took ~5 ms of the worker's event loop per request, which a burst's requests
    queued for."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            n = headers.get(b"content-length", b"")
            if n.isdigit() and int(n) > MAX_MB * 1e6 + 1e5:
                await JSONResponse({"detail": f"file too large (max {MAX_MB:g} MB)"}, status_code=413)(scope, receive, send)
                return
            scope.setdefault("state", {})["arrived"] = time.time()  # before the body is read and parsed
            ctype = headers.get(b"content-type", b"")
            if remote and scope["method"] == "POST" and scope["path"] == "/transcribe" and ctype.startswith(
                    b"multipart/form-data"):
                await self.transcribe(scope, receive, send, ctype)
                return
        await self.app(scope, receive, send)

    async def transcribe(self, scope, receive, send, ctype: bytes):
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                return  # the client went away
            chunks.append(message.get("body", b""))
            size += len(chunks[-1])
            if size > MAX_MB * 1e6 + 1e5:
                await JSONResponse({"detail": f"file too large (max {MAX_MB:g} MB)"}, status_code=413)(scope, receive, send)
                return
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        form = fast_form(body, ctype)
        if form is None:  # anything unusual: FastAPI parses and validates it as before
            replayed = False

            async def again():
                nonlocal replayed
                if replayed:
                    return await receive()
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}

            await self.app(scope, again, send)
            return
        try:
            response = JSONResponse(await transcribe_data(form["file"], form["language"], form["punctuation"],
                                                          form["level"], form["reference"], scope["state"]["arrived"]))
        except HTTPException as e:
            response = JSONResponse({"detail": e.detail}, status_code=e.status_code, headers=e.headers)
        await response(scope, receive, send)


FORM_BOOL = {"1": True, "on": True, "t": True, "true": True, "y": True, "yes": True,
             "0": False, "off": False, "f": False, "false": False, "n": False, "no": False}  # pydantic's
FORM_NAME = re.compile(rb'(?i)^content-disposition:\s*form-data;\s*name="([^"]*)"(;\s*filename="([^"]*)")?\s*$')


def fast_form(body: bytes, ctype: bytes) -> dict | None:
    """The fields of a /transcribe multipart body (file bytes; language, punctuation, level, reference with
    /transcribe's defaults), or None for any form FastAPI should parse and validate itself (another layout, a field
    twice, an empty or unparsed value, a file field without a filename)."""
    m = re.search(rb'boundary="?([^";\s]+)"?', ctype)
    if not m:
        return None
    parts = body.split(b"--" + m.group(1))
    if len(parts) < 3 or parts[0] not in (b"", b"\r\n") or not parts[-1].startswith(b"--"):
        return None
    form: dict = {}
    for part in parts[1:-1]:
        head, sep, value = part.partition(b"\r\n\r\n")
        if not sep or not head.startswith(b"\r\n") or not value.endswith(b"\r\n"):
            return None
        name = filename = None
        for line in head[2:].split(b"\r\n"):
            d = FORM_NAME.match(line)
            if d:
                name, filename = d.group(1).decode(), d.group(3)
            elif not line.lower().startswith(b"content-type:"):
                return None
        if name is None or name in form:
            return None
        value = value[:-2]
        if name == "file":
            if not filename:
                return None
            form["file"] = value
        elif name in ("language", "punctuation", "level", "reference"):
            try:
                text_value = value.decode()
            except UnicodeDecodeError:
                return None
            if not text_value:
                return None
            if name in ("punctuation", "level"):
                if text_value.lower() not in FORM_BOOL:
                    return None
                form[name] = FORM_BOOL[text_value.lower()]
            else:
                form[name] = text_value
    if "file" not in form:
        return None
    return {"language": "auto", "punctuation": True, "level": True, "reference": "", **form}


app.add_middleware(LimitSize)


async def run(request: Request, file: UploadFile, language: str, punctuation: bool, level: bool = True,
              reference: str = "") -> dict:
    if language not in LANGUAGES:
        raise HTTPException(400, f"language must be one of: {', '.join(LANGUAGES)}")
    arrived = getattr(request.state, "arrived", time.time())
    data = await file.read()
    await file.close()
    return await transcribe_data(data, language, punctuation, level, reference, arrived)


async def transcribe_data(data: bytes, language: str, punctuation: bool, level: bool, reference: str,
                          arrived: float) -> dict:
    global waiting
    if language not in LANGUAGES:
        raise HTTPException(400, f"language must be one of: {', '.join(LANGUAGES)}")
    t0 = time.time()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > MAX_MB * 1e6:
        raise HTTPException(413, f"file too large (max {MAX_MB:g} MB)")
    if batcher is None and remote is None:
        raise HTTPException(503, "model is loading, try again shortly")
    if waiting >= MAX_WAITING:
        raise HTTPException(503, "busy, try again in a minute")
    waiting += 1
    try:
        try:
            if remote:
                if inline(data):  # a few ms of work: in a thread it waited its turn for the GIL with the others'
                    x, mono = load_mono(data, level)
                else:
                    x, mono = await asyncio.to_thread(load_mono, data, level)
                spans = [(0, len(mono))] if len(mono) <= WHOLE_S * SR else await remote.call("pieces", mono)
                if len(spans) == 1 and inline(data):
                    feats = [dither(mono)]
                else:
                    feats = await asyncio.to_thread(lambda: [dither(mono[s:e]) for s, e in spans])
            else:
                x, spans, feats = await asyncio.to_thread(load_and_prepare, data, level)
        except AudioTooLong as e:
            raise HTTPException(413, str(e)) from None
        except BadAudio as e:
            raise HTTPException(400, str(e)) from None
        except subprocess.TimeoutExpired:
            raise HTTPException(400, "decoding the audio took too long") from None
        t1 = time.time()
        if remote:
            tokens = await remote.call("decode", feats, language, punctuation) if feats else []
            if not tokens:
                segs = segments(spans, [])
            elif inline(data) and len(spans) == 1:
                segs = segments(spans, text.texts(tokens, language))
            else:
                segs = segments(spans, await asyncio.to_thread(text.texts, tokens, language))
        else:
            segs = segments(spans, await batcher.submit(feats, language, punctuation))
    finally:
        waiting -= 1
    t2 = time.time()
    if remote and inline(data) and len(segs) == 1 and not reference.strip():
        return respond(segs, x, data, language, level, reference, arrived, t0, t1, t2)
    return await asyncio.to_thread(respond, segs, x, data, language, level, reference, arrived, t0, t1, t2)


def inline(data: bytes) -> bool:
    """An upload an HTTP worker prepares on its event loop: under 1.5 MB (45 s of 16-bit 16 kHz mono) in a format
    libsndfile reads (no ffmpeg run). Six requests of a worker in threads took ~40 ms to get through their 3 ms each."""
    return len(data) <= 1_500_000 and data[:4] in (b"RIFF", b"fLaC", b"OggS")


def load_checked(data: bytes) -> np.ndarray:
    """Decoded samples of an upload; AudioTooLong / BadAudio for audio over MAX_MINUTES or under 0.1 s."""
    x = load_audio(data)
    duration = len(x) / SR
    if duration > MAX_MINUTES * 60:
        raise AudioTooLong(f"audio too long (max {MAX_MINUTES:g} minutes)")
    if duration < 0.1:
        raise BadAudio("audio too short")
    return x


def load_mono(data: bytes, level: bool) -> tuple[np.ndarray, np.ndarray]:
    """An HTTP worker's part: decoded (samples, channels) and the levelled mono for the model."""
    x = load_checked(data)
    return x, text.mono(x, level)


def warm_worker() -> None:
    """An HTTP worker's request path once (decode, level, dither) on a synthetic upload, before the first request."""
    buf = io.BytesIO()
    sf.write(buf, (np.random.default_rng(0).standard_normal((5 * SR, 2)) * 0.05).astype(np.float32), SR, format="WAV")
    x, mono = load_mono(buf.getvalue(), True)
    dither(mono)


def load_and_prepare(data: bytes, level: bool) -> tuple[np.ndarray, list[tuple[int, int]], list]:
    """Decoded samples, pieces and model input of an upload (in a worker thread, WORKERS=0)."""
    x = load_checked(data)
    spans, feats = engine.prepare(x, level)
    return x, spans, feats


def respond(segs: list[dict], x: np.ndarray, data: bytes, language: str, level: bool, reference: str, arrived: float,
            t0: float, t1: float, t2: float) -> dict:
    """The response of a transcribed request: normalized text, language, WER (in a worker thread)."""
    duration = len(x) / SR
    seconds: dict[str, float] = {}
    for s in segs:
        s["normalized_text"] = textnorm.normalize(s["text"], s["language"] if s["language"] in ("ar", "en") else None)
        if s["text"]:
            seconds[s["language"]] = seconds.get(s["language"], 0) + s["end"] - s["start"]
    log.info("transcribed audio_s=%.1f bytes=%d channels=%d language=%s level=%s pieces=%d upload_s=%.3f prep_s=%.3f "
             "queue_gpu_s=%.3f total_s=%.3f", duration, len(data), x.shape[1], language, level, len(segs), t0 - arrived,
             t1 - t0, t2 - t1, t2 - arrived)
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
    return {"status": "ok" if engine or remote else "loading", "waiting": waiting}


@app.post("/transcribe")
async def transcribe(request: Request, file: UploadFile = File(...), language: str = Form("auto"),
                     punctuation: bool = Form(True), level: bool = Form(True), reference: str = Form("")):
    return await run(request, file, language, punctuation, level, reference)


@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(request: Request, file: UploadFile = File(...), model: str = Form(MODEL_ID),
                                language: str = Form(""), response_format: str = Form("json")):
    if response_format not in ("json", "text", "verbose_json"):
        raise HTTPException(400, "response_format must be json, text or verbose_json")
    r = await run(request, file, language if language in ("ar", "en") else "auto", True)
    if response_format == "text":
        return PlainTextResponse(r["text"])
    if response_format == "verbose_json":
        return {"task": "transcribe", "language": r["language"], "duration": r["duration"], "text": r["text"],
                "segments": [{"id": i, "start": s["start"], "end": s["end"], "text": s["text"]}
                             for i, s in enumerate(r["segments"])]}
    return {"text": r["text"]}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.Formatter.converter = time.gmtime
    if not WORKERS:
        uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
        return
    import secrets

    private = tempfile.mkdtemp(prefix="s86k-gpu-")  # mode 0700
    address = os.path.join(private, "gpu.sock")
    os.environ["GPU_SOCKET"], os.environ["GPU_AUTHKEY"] = address, secrets.token_hex(16)
    gpu = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--gpu", address])
    try:
        while not os.path.exists(address):  # the GPU process listens once its model is loaded and warmed up
            if gpu.poll() is not None:
                raise SystemExit(f"GPU process exited with {gpu.returncode}")
            time.sleep(0.2)
        uvicorn.run("server:app", host="127.0.0.1", port=PORT, log_level="info", workers=WORKERS,
                    app_dir=os.path.dirname(os.path.abspath(__file__)))
    finally:
        gpu.terminate()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--gpu":
        serve_gpu(sys.argv[2])
    else:
        main()
