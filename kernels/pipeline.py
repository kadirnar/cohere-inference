"""The GPU side of the service: log-mel features, encoder and decoder of every waiting piece, scheduled for latency.

Two threads own the GPU, each with its own CUDA stream:

  encoder thread   takes the waiting pieces, shortest first, up to CHUNK_FRAMES encoder frames and as many as there are
                   free decoder slots (their samples were copied to the GPU as they arrived); computes their
                   features (kernels/features.py), runs the packed encoder
                   (kernels/fast_encoder.py) and writes their cross-attention keys and values into the slots; then
                   hands the chunk to the decoder thread and starts the next one
  decoder thread   runs decoding steps (kernels/stream_decoder.py) over every admitted piece: a chunk joins the batch as
                   soon as it is encoded, a piece leaves it as soon as it ends (its tokens are copied out and its
                   request answered), so short pieces are not held back by long ones and the encoder of the next chunk
                   runs while the current ones decode

The host reads the end flags one step behind (LAG), so the GPU never waits for it. A slot's flag is trusted only from
a step launched after the slot was admitted (an earlier step may still copy the previous occupant's flag).

    pipe = Pipeline(model, processor.feature_extractor)
    tokens = pipe.submit([samples1, samples2], prompt).result()   # per piece: prompt + generated ids (np.int64)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field

import numpy as np
import torch

from kernels.fast_encoder import FastEncoder
from kernels.features import Features, dither
from kernels.stream_decoder import StreamDecoder


log = logging.getLogger("s86k")


@dataclass
class _Job:
    n: int
    future: Future
    tokens: list = field(default_factory=list)
    left: int = 0


@dataclass
class _Piece:
    job: _Job
    k: int
    samples: np.ndarray  # dithered
    gpu: torch.Tensor  # the same on the GPU, copied on arrival (upload stream) ...
    uploaded: torch.cuda.Event  # ... done when this event is
    prompt: torch.Tensor
    seq: int  # arrival order
    max_new: int | None = None
    arrived: float = field(default_factory=time.perf_counter)


class Pipeline:
    CHUNK_FRAMES = 3000  # encoder frames per encoder call (about 4 minutes of audio)
    ORDER = "short_then_long"  # the shortest third first (early answers), then the longest (a short decoding tail)
    MAX_WAIT_S = 0.75  # pieces waiting longer go first, whatever their length
    LAG = 1

    def __init__(self, model, extractor, slots: int = 128, max_new_tokens: int = 448, int8: bool = False,
                 sparse: str | None = None):
        self.features = Features(extractor)
        self.encoder = FastEncoder(model.model.encoder, int8=int8, sparse=sparse)
        self.decoder = StreamDecoder(model, slots=slots, max_new_tokens=max_new_tokens, int8=int8)
        self.enc_stream = torch.cuda.Stream()
        self.up_stream = torch.cuda.Stream()  # the pieces' samples to the GPU as they arrive, off the encoder's path
        self.lock = threading.Condition()
        self.waiting: list[_Piece] = []
        self.free = list(range(slots))[::-1]
        self.ready: deque = deque()  # encoded chunks for the decoder thread: (slots, pieces, frames)
        self.seq = 0
        self.out_host = torch.zeros(slots + 1, self.decoder.out.shape[1], dtype=torch.long).pin_memory()
        self.stats = {"steps": 0, "chunks": 0}
        self.trace: list | None = None  # set to a list to record (time, what, detail) of chunks and deliveries
        self._threads = [threading.Thread(target=self._encoder_loop, daemon=True, name="encoder"),
                         threading.Thread(target=self._decoder_loop, daemon=True, name="decoder")]
        for t in self._threads:
            t.start()

    # ---- requests --------------------------------------------------------------------------------------------------

    @staticmethod
    def prepare(samples: np.ndarray) -> np.ndarray:
        """The extractor's dither, added in the caller's thread (CPU, about 1 ms per 10 s)."""
        return dither(samples)

    def submit(self, pieces: list[np.ndarray], prompt: torch.Tensor, max_new: list[int] | None = None) -> Future:
        """Decode dithered pieces (prepare()) with this prompt; the future's result is one int64 array per piece.
        max_new: tokens to generate at most per piece (default the decoder's max_new_tokens)."""
        job = _Job(len(pieces), Future(), [None] * len(pieces), len(pieces))
        if not pieces:
            job.future.set_result([])
            return job.future
        with torch.cuda.stream(self.up_stream):
            dev = self.decoder.device
            gpu = [torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(dev, non_blocking=True)
                   for x in pieces]
            uploaded = torch.cuda.Event()
            uploaded.record(self.up_stream)
        with self.lock:
            for k, x in enumerate(pieces):
                limit = max_new[k] if max_new else None
                self.waiting.append(_Piece(job, k, x, gpu[k], uploaded, prompt, self.seq, limit))
                self.seq += 1
            self.lock.notify_all()
        return job.future

    # ---- encoder thread ------------------------------------------------------------------------------------------

    def _take(self) -> tuple[list[_Piece], list[int]]:
        """The next chunk (lock held), within CHUNK_FRAMES and the free slots: first every piece that has waited over
        MAX_WAIT_S, oldest first (under steady load the ordering below would otherwise starve some), then the others
        in ORDER: the shortest third first (their requests answered early), then the longest (a short decoding tail)."""
        if not self.waiting or not self.free:
            return [], []
        now = time.perf_counter()
        late = sorted((p for p in self.waiting if now - p.arrived > self.MAX_WAIT_S), key=lambda p: p.seq)
        rest = sorted((p for p in self.waiting if now - p.arrived <= self.MAX_WAIT_S), key=lambda p: len(p.samples))
        if self.ORDER == "short_then_long" and len(rest) > 8:
            k = len(rest) // 3
            rest = rest[:k] + rest[k:][::-1]
        queue = late + rest
        chunk, frames = [], 0
        for p in queue:
            f = len(p.samples) // 1280 + 1
            if chunk and (frames + f > self.CHUNK_FRAMES or len(chunk) >= len(self.free)):
                break
            chunk.append(p)
            frames += f
        self.waiting = queue[len(chunk):]
        slots = [self.free.pop() for _ in chunk]
        return chunk, slots

    @torch.no_grad()
    def _encoder_loop(self) -> None:
        torch.cuda.set_device(self.decoder.device)
        while True:
            with self.lock:
                chunk, slots = self._take()
                while not chunk:
                    self.lock.wait()
                    chunk, slots = self._take()
            started = time.perf_counter()
            if self.trace is not None:
                self.trace.append((started, "encode", len(chunk), len(self.free)))
            try:
                with torch.cuda.stream(self.enc_stream):
                    for ev in {id(p.uploaded): p.uploaded for p in chunk}.values():
                        self.enc_stream.wait_event(ev)
                    feats, frames = self.features([p.gpu for p in chunk])
                    hidden, lengths = self.encoder(feats, frames)
                    self.decoder.fill_cross(hidden, lengths, slots)
                    done = torch.cuda.Event()
                    done.record(self.enc_stream)
                done.synchronize()
            except Exception as e:  # noqa: BLE001 (each request of the chunk fails with it)
                with self.lock:
                    self.free.extend(slots)
                for p in chunk:
                    if not p.job.future.done():
                        p.job.future.set_exception(e)
                continue
            if self.trace is not None:
                self.trace.append((time.perf_counter(), "encoded", len(chunk), sum(lengths)))
            log.info("encoded pieces=%d frames=%d in %.0f ms (oldest waited %.0f ms)", len(chunk), sum(lengths),
                     1000 * (time.perf_counter() - started), 1000 * (started - min(p.arrived for p in chunk)))
            with self.lock:
                self.ready.append((slots, chunk, lengths))
                self.stats["chunks"] += 1
                self.lock.notify_all()

    # ---- decoder thread ------------------------------------------------------------------------------------------

    @torch.no_grad()
    def _decoder_loop(self) -> None:
        torch.cuda.set_device(self.decoder.device)
        dec = self.decoder
        active: list[int] = []
        piece_of: dict[int, _Piece] = {}
        admitted_at: dict[int, int] = {}
        steps: deque = deque()  # (step index, event)
        copies: deque = deque()  # (event, [(slot, piece)]) finished rows on their way to the host
        step_no, bucket, changed = 0, 1, False
        while True:
            with self.lock:
                while not self.ready and not active and not copies:
                    self.lock.wait()
                new = list(self.ready)
                self.ready.clear()
            for slots, chunk, lengths in new:
                dec.admit(slots, [p.prompt for p in chunk], lengths,
                          [p.max_new or dec.max_new_tokens for p in chunk])
                for s, p in zip(slots, chunk):
                    piece_of[s] = p
                    admitted_at[s] = step_no
                active += slots
                changed = True
            if active:
                if changed:
                    bucket = dec.set_active(active)
                    changed = False
                dec.step(bucket)
                ev = torch.cuda.Event()
                ev.record(dec.stream)
                steps.append((step_no, ev))
                step_no += 1
                self.stats["steps"] += 1
            # end flags of the step LAG behind the newest (or of every step when nothing runs)
            while steps and (len(steps) > self.LAG or not active):
                k, ev = steps.popleft()
                ev.synchronize()
                flags = dec.flags.numpy()
                ended = [s for s in active if flags[s] and admitted_at[s] <= k]
                if ended:
                    active = [s for s in active if s not in set(ended)]
                    changed = True
                    with torch.cuda.stream(dec.stream):
                        for s in ended:
                            self.out_host[s].copy_(dec.out[s], non_blocking=True)
                        cev = torch.cuda.Event()
                        cev.record(dec.stream)
                    copies.append((cev, [(s, piece_of.pop(s)) for s in ended]))
            while copies and (copies[0][0].query() or not active):
                cev, rows = copies.popleft()
                cev.synchronize()
                for s, p in rows:
                    self._deliver(p, self.out_host[s].numpy().copy())
                with self.lock:
                    self.free.extend(s for s, _ in rows)
                    self.lock.notify_all()

    def _deliver(self, p: _Piece, row: np.ndarray) -> None:
        plen = len(p.prompt)
        gen = row[plen:]
        end = np.nonzero(gen == self.decoder.eos)[0]
        n = plen + (end[0] + 1 if len(end) else min(len(gen), self.decoder.max_new_tokens))
        job = p.job
        job.tokens[p.k] = row[:n]
        job.left -= 1
        if job.left == 0 and not job.future.done():
            job.future.set_result(job.tokens)
