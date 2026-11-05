# cohere-inference

A speech-to-text service for
[Cohere-AR/cohere-transcribe-ar-sft-v9-full-s86k](https://huggingface.co/Cohere-AR/cohere-transcribe-ar-sft-v9-full-s86k)
(commit `3ec483a713`, a community fine-tune of
[CohereLabs/cohere-transcribe-arabic-07-2026](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026), not
made or endorsed by Cohere): Arabic in many dialects, English and code-switched Arabic-English, phone calls included.
One GPU process with its own CUDA kernels and a few FastAPI worker processes on one GPU, a test page, an OpenAI-style
route, input level normalisation, scoring-normalised text and WER against a reference; optionally published through a
Cloudflare quick tunnel, without a key. The evaluation and
normalisation-study scripts behind its defaults are in `test/` and `norm_study/`.

## Quick start

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install torch                      # the build for your CUDA version
pip install -r requirements.txt
export HF_TOKEN=...                    # the weights are gated on the Hub (access granted on request)
PORT=8086 python server.py             # http://127.0.0.1:8086 (test page), /docs (API docs)
```

`MODEL` may point to a local copy of the commit instead (then `HF_HUB_OFFLINE=1`). `./start.sh` runs the server in a
tmux session, restarts it if it exits and, unless `TUNNEL=0`, opens a public
[Cloudflare quick tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/)
to it (needs `cloudflared`; the `https://*.trycloudflare.com` address is printed in `logs/tunnel.log` and changes when
the tunnel restarts). Its settings come from the environment or from a `local.env` file next to it (not in git).
About 6.7 GB of GPU memory (10 GB with `INT8=0`). The first start compiles the CUDA kernels of `kernels/csrc` (needs
`nvcc` and `ninja`; about a minute, cached in `~/.cache/torch_extensions`); after that start-up (loading the weights,
capturing the decoder's CUDA graphs, a warm-up burst) takes about 15 s.

## API

| Route | |
|---|---|
| `GET /` | test page: upload a file or record from the microphone, pick the language, optionally paste a reference |
| `POST /transcribe` | multipart form, see below |
| `POST /v1/audio/transcriptions` | OpenAI-style: `file`, `model` (ignored), `language`, `response_format` = `json` / `text` / `verbose_json` |
| `GET /info`, `GET /health`, `GET /docs` | model and limits, liveness and queue, interactive docs |

`/transcribe` fields: `file` (any audio libsndfile or ffmpeg reads), `language` = `auto` (default: the model writes its
own language header) / `ar` / `en`, `punctuation` (with a forced language), `level` = `true` (default) / `false`,
`reference` (optional text: the response then carries the WER against it).

```bash
curl -F file=@call.wav https://<host>/transcribe
curl -F file=@call.wav -F language=ar -F "reference=..." https://<host>/transcribe
```

```python
from openai import OpenAI
client = OpenAI(base_url="https://<host>/v1", api_key="none")
print(client.audio.transcriptions.create(model="s86k", file=open("call.wav", "rb")).text)
```

The response has `text` (the model's own words), `normalized_text` (the same under the Arabic Arena's scoring
normaliser), `language`, `duration`, `processing_time`, `level`, `segments` (start, end, language, text,
normalized_text) and, with a reference, `wer` (`l2`: Arabic Arena normaliser, `ext`: the Test Arabic ASR Benchmark's
`normalize_ext`; edits, reference words, WER). Limits: 100 MB, 30 minutes of audio, 128 requests in flight per HTTP worker
(503 beyond); the pieces of all waiting requests are encoded together and decoded with continuous batching (see Speed). Audio and transcripts are kept in memory only (uploads over 1 MB spool to `TMPDIR`); the
log holds timings, sizes and access lines (client address, route, status), no audio or text.

## What happens to the audio

1. Decoding: libsndfile (wav, flac, ogg, mp3), otherwise ffmpeg (m4a, aac, webm, ...) with `-protocol_whitelist file`
   and a demuxer whitelist, so an uploaded playlist or concat file cannot open other files or URLs.
2. 16 kHz with soxr, channels kept.
3. Input level (`level=true`), measured from the signal alone, so a request runs no network besides the ASR model
   (and Silero for audio over 30 s, as without it): a channel's speech level is the RMS of its loudest 20 % of 50 ms
   frames, and a channel whose loudest and quietest 20 % of frames are less than 12 dB apart (silence, steady noise,
   tones) is never amplified. A two-channel call has each channel's speech brought to -20 dBFS (at most +60 dB) before
   the sum, so both sides are equally loud. Mono audio is left bit for bit as it is unless its speech is under -45 dBFS;
   then it is brought to -20 dBFS. `level=false` takes the plain mean of the channels, as the Arabic Arena feeds the
   model.
4. Audio up to 30 s goes to the model whole (as the model card says and the Arena runs it); longer audio is cut at
   pauses (Silero's own probabilities and regions, its network run over all windows at once on the GPU) into pieces of
   at most 25 s, decoded in batches.
5. Greedy decoding, at most 448 new tokens (the card's value), through the CUDA kernels of `kernels/csrc` (see Speed).
   With `language=auto` the decoder prompt stops after
   `<|emo:undefined|>` and the model writes its own language header, per piece; with `ar` / `en` the full prompt with
   `<|pnc|>` is used. No post-processing of the text.

## Speed

Everything after the audio decoding runs through CUDA kernels written for this model (`kernels/csrc`, compiled on the
first start; `FAST=0` runs transformers' own code instead), scheduled for many simultaneous requests:

- **Processes.** One GPU process (`server.py --gpu`, started by `server.py`) holds the model; `WORKERS` HTTP worker
  processes (default 8) parse the uploads, decode and level the audio, add the feature extractor's deterministic
  dither and turn token ids into text, and talk to the GPU process over a local socket. In one process the event
  loop, the request threads and the threads launching GPU work held each other up through the GIL.
- **Features** (`kernels/features.py`): the log-mel frames of a whole batch of pieces on the GPU (cuFFT and a few
  kernels); 0.06 % of the bf16 values differ from the CPU extractor's, by one unit in the last place.
- **Encoder** (FastConformer, 48 layers, 1.9 B parameters, `kernels/fast_encoder.py`). The pieces' frames are packed
  (no padding at all) and each layer is 14 kernels launched from C++ (`encoder_loop.cpp`): GEMMs with their epilogues
  fused (bias, SiLU, GLU, residual sum; `gemm.cu`, a cp.async / ldmatrix / mma.sync kernel that matches cuBLAS's speed
  and bits), LayerNorm (`encoder_ops.cu`), a relative-position flash attention whose position term is computed from a
  per-layer table of projected distances inside the kernel (no (T, 2T) score tensor), the convolution module's
  depthwise convolution with BatchNorm and SiLU, and the subsampling convolutions per piece, channels-last (the first
  convolution and the depthwise one after it in one kernel). With `INT8` (default) every GEMM is int8 (W8A8:
  `gemm_int8.cu`, weights per output channel, activations per row, quantized by the kernel that produces them; on the
  A10's 150 W cap int8 tensor cores do about 1.4x the bf16 work).
- **Decoder** (8 layers, `kernels/stream_decoder.py`, `decoder_ops.cu`): continuous batching over 128 slots. A piece
  joins the batch as soon as its encoder chunk is done (its cross-attention keys and values written into its slot)
  and leaves it at its end-of-text token, when its request is answered; one CUDA graph per bucket of row counts runs a
  step for any mix of positions and audio lengths (rows read their slot through an active list). Kernels: embedding
  and LayerNorms, residual + LayerNorm, flash-decoding attention over the slot's cache, greedy choice with prompt
  forcing. With `INT8` the q/k/v, feed-forward and output-head weights are int8 (W8A16, `dec_w8.cu`: widened to bf16
  in registers, split-K) and the self- and cross-attention caches are int8 with a scale per 128-vector.
- **Scheduling** (`kernels/pipeline.py`): an encoder thread takes the waiting pieces in chunks of up to 3,000 encoder
  frames (any that waited over 0.75 s first, then the shortest third, then the longest, so early requests are answered
  early and the last chunk decodes quickly) while a decoder thread steps on its own CUDA stream at high priority, so decoding runs beside the encoding
  of the next chunk. The host reads end-of-text flags one step behind; the GPU never waits for it.
- **Silero** (audio over 30 s). Only its LSTM depends on earlier windows, so the STFT and convolutions run on all
  windows at once and the LSTM over the whole recording in a few cuDNN calls (`silero_parallel.py`): the same
  probabilities as silero-vad's onnxruntime loop (largest difference 1.7e-5, no window on the other side of a
  threshold) and the same regions on all 22 test recordings, 67 times faster.

**100 simultaneous requests** (`bench/concurrency.py --dir`: the first 100 FLEURS ar_eg test clips, 1,127 s of audio,
posted at the same moment over HTTP from the same machine; A10; 6 bursts each; before = commit `dc1d204`, CUDA graphs
and Triton kernels, one process). These figures use Cohere's base model (`cohere-transcribe-03-2026`, same
architecture) because the s86k weights were not on the machine:

| | p50 latency | slowest | audio seconds per second |
|---|---|---|---|
| before | 1.66-1.94 s | 2.97-3.01 s | 377 |
| now, `INT8=0` (bf16 arithmetic) | 0.88-1.01 s | 1.41-1.50 s | 770 |
| now, `INT8=1` (default) | 0.63-0.74 s | 0.99-1.05 s | 1,110 |

So 100 requests are all answered within about a second: 10 ms of GPU time per request. Under steady load
(`--sustain 30`: 100 clients, each sending its next clip as soon as the last is answered) the service answers 116
requests per second (1,307 audio seconds per second, 8.6 ms of GPU per request) with a p50 latency of 0.65 s, p99 1.54
s and none over 1.70 s; pieces waiting over 0.75 s go first, whatever their length (with 0.5 s: p50 0.97 s, p99 1.32
s, but bursts end later). Per request the latency cannot
come near that: a decoding step reads ~130 MB of weights and caches however few rows it has (0.65 ms on an A10), and a
clip needs one step per token (36-96 for these clips).

Where the time goes (`bench/stages.py`, `INT8=1`): one 14 s Arabic clip takes 1 ms of features, 18 ms of encoder
(178 frames) and 66 ms of decoder (96 steps, 0.68 ms each); 32 clips in one call 14 ms, 190 ms (4,371 frames) and 141
ms (136 steps, 1.04 ms each). The encoder of the 100 clips (14,160 frames, 4 chunks): transformers' code 1,825 ms,
packed bf16 kernels 969 ms, int8 636 ms. A decoding step: 0.65 / 0.70 / 0.95 / 1.42 / 2.53 ms for 1 / 8 / 32 / 64 /
128 rows (bf16 graphs before: 0.83 / 0.92 / 1.24 / 1.91 / 2.91). It is still about twice the time its bytes take at
the A10's bandwidth (a step of ~90 kernels; a 2 MB read alone takes 5.3 us in a kernel of its own against 4.0 us
streamed). A persistent kernel running the whole step (one CTA per SM, grid barriers of ~1 us between its ~66 phases,
the weights streamed ahead through cp.async) was tried and came out slower, 0.73 ms for one row and 1.47 ms for 32:
every phase keeps a chain of dependent latencies (barrier, the activations from L2, the cross-warp sum), and shared
memory holds too few weight units ahead to keep the memory busy through the short phases. One request at a time (HTTP, FLEURS clips, median):
65 ms under 8 s of audio, 77 ms for 8-12 s, 94 ms for 12-26 s (before: 84, 107, 145 ms).

## Validation

All Arena figures are WER v2 (the Arena's own scorer and normaliser, per-utterance errors capped at the reference
length), with the endpoint called over HTTP like any client.

**Same transcripts as the model.** With transformers' own code, on 2,028 clips (SADA telephony, Casablanca call-centre,
EdAcc, FLEURS Arabic) the endpoint's text equaled the model run through the Arena's runner one clip at a time on 100 %
of clips; the Arena's runner at its batch size of 32 reproduces the published predictions on 100 %. Batch 1 and batch
32 differ on 1.6-12.7 % of clips (bf16 padding) and by at most 0.1 WER. The table below was measured on s86k with the
previous kernels (commit `dc1d204`): they computed the same thing in another order, about 97 % of clips identical to
transformers' output, the rest differing the way a different batch size does, and the WER where it was.

The current kernels were checked with Cohere's base model (`cohere-transcribe-03-2026`, the same architecture; the
s86k weights were not on the machine) on FLEURS ar_eg (428 clips) and en_us (647), `auto`, against transformers'
`generate` one clip at a time (WER with the Arena L2 normaliser):

| | ar_eg WER | identical | en_us WER | identical |
|---|---|---|---|---|
| transformers' `generate`, batch 1 | 6.90 | | 6.37 | |
| `INT8=0` (bf16 arithmetic) | 6.91-6.93 | 414 (96.7 %) | 6.34-6.36 | 629 (97.2 %) |
| `INT8=1` (default) | 6.80-6.95 | 384 (89.7 %) | 6.33-6.42 | 598 (92.4 %) |

(Ranges over runs with all clips submitted at once: which pieces share a batch depends on timing, and with it the
decoder's summation order; identical counts from one run.) int8 changes about one transcript in ten, the way a
different batch size does, and not the WER; still, run
`test/arena_check.py` against s86k before relying on the Arena figures below with `INT8=1`. Re-scoring the published predictions with
`test/scorer_check.py` reproduces the published WER v2, v1 and CER exactly.

| Set (clips) | published | forced, transformers' code (Nov 3) | forced, now | `auto`, now |
|---|---|---|---|---|
| fleurs_ar/clean (428) | 3.72 | 3.67 | 3.72 | 3.74 |
| sada/telephony_v2 (300) | 22.84 | 22.91 | 22.78 | 22.76 |
| casablanca/callcenter_v2 (300) | 39.54 | 39.64 | 39.54 | 39.51 |
| mixat_cs/clean (1,000) | 10.94 | 10.96 | 10.93 | 10.95 |
| cv18_ar/noisy (300) | 8.75 | 8.82 | 8.82 | 8.88 |
| fleurs_en/clean (647) | 5.52 | 5.52 | 5.34 | 5.35 |
| edacc/clean (1,000) | 12.61 | 12.65 | 12.65 | 12.70 |
| earnings22/telephony (300) | 7.45 | 7.41 | 7.29 | 7.37 |
| librispeech_clean/callcenter (300) | 4.53 | 4.53 | 4.60 | 4.58 |
| mean of the 5 Arabic / 4 English sets | 17.16 / 7.53 | 17.20 / 7.53 | 17.16 / 7.47 | 17.17 / 7.50 |

The Nov 3 column predates the input levelling (FLEURS-en clean 5.52 → 5.35 is its doing, see below).

In `auto` mode the model picked the right language on 897 of 900 English and 1,146 of 1,150 Arabic clips
(`test/lang_check.py`); code-switched clips get an Arabic header with the English words kept.

**Input level** (`norm_study/validate_level.py`, 150 clips per set, language forced): at the clips' own level the six
sets score exactly as with `level=false`; 40 and 60 dB quieter:

| Set | own level | -40 dB: off → on | -60 dB: off → on |
|---|---|---|---|
| fleurs_ar/clean | 3.45 | 3.55 → 3.41 | 9.46 → 3.45 |
| sada/telephony_v2 | 22.15 | 25.26 → 21.80 | 44.17 → 22.28 |
| casablanca/callcenter_v2 | 52.19 | 56.58 → 52.40 | 74.46 → 53.93 |
| librispeech_clean/clean | 1.03 | 1.07 → 1.14 | 1.64 → 1.14 |
| edacc/clean | 12.55 | 15.51 → 12.59 | 24.79 → 12.59 |
| earnings22/telephony | 7.05 | 8.22 → 7.26 | 19.88 → 7.13 |

On the Arena's noise-only clips the share with 3+ words is 35.7 % → 35.7 % (Arabic) and 96.0 % → 96.3 % (English, one
clip). Over every Arena clip (`norm_study/arena_input_check.py`, `arena_level_effect.py`) `level=true` changes 37 of
30,080 Arabic and 697 of 17,305 English clips (quiet FLEURS-en and AMI recordings, speech around -57 dBFS) and moves
s86k's Arena WER v2 from 12.279 to 12.280 (Arabic) and from 6.042 to 6.032 (English; FLEURS-en clean 5.52 → 5.35). On
20 private two-channel and mono call recordings (Audioluna, scored against human-reviewed references with the Arena
normaliser) the WER goes from 12.57 (`level=false`) to 12.26. The measurement costs no time (`level=false` and `true`
take the same 0.27 s for a 6.8 s clip and 1.9 s for a 114 s two-channel call). Two earlier versions measured speech
with Silero instead: one also levelled normal-level mono audio and cost 0.1-0.5 WER on three sets at their own level;
the other matched these figures (Audioluna 12.38) but added 0.05-1.7 s per request, a second network pass.

**Concurrent requests** (`bench/concurrency.py`: 50 different FLEURS / LibriSpeech clips, 454 s of audio, posted at the
same moment from the same machine). Requests are prepared in parallel threads and the pieces of all waiting requests
are decoded together, at most 32 per GPU batch; each request is answered as soon as its own pieces are done.

| | answered | p50 latency | slowest | audio seconds per second |
|---|---|---|---|---|
| one request at a time, 8 queued (before) | 8 of 50 (42 × 503) | 1.98 s | 2.80 s | 24 |
| one request at a time, queue of 100 | 50 | 8.54 s | 17.72 s | 26 |
| shared batches, transformers' code, local | 50 | 2.11 s | 3.78 s | 120 |
| shared batches, transformers' code, through the public tunnel | 50 | 5.80 s | 7.31 s | 62 |
| shared batches, CUDA graphs and Triton kernels (`dc1d204`), local | 50 | 0.84 s | 1.41 s | 320 |
| the same through the public tunnel | 50 | 1.08 s | 1.51 s | 300 |

(s86k; for 100 requests with the current kernels see Speed.)

A lone request takes 0.08 s for a 9 s clip (0.3 s before). Sharing a batch no longer changes a transcript: 100 of 100
clips come out identical one at a time and all at once (`bench/batch_consistency.py`, Arena L2 WER 2.88 % both ways;
98 of 100 with transformers' code, 2.82 % and 3.04 %). The Audioluna calls score 12.19 (12.22 with transformers' code,
12.74 for the model card's plain code, which cuts long audio itself).

## Normalisers (`textnorm/`, `norm_study/`)

`norm_study/compare_normalizers.py` scored the published s86k transcripts of every Arena set under four normalisers
(pooled WER, uncapped):

| | `normalize_elm` | `normalize_ext` | Arena L1 | Arena L2 |
|---|---|---|---|---|
| Arabic (29,749 utterances) | 26.48 | 24.20 | 24.76 | 23.06 |
| English (16,982 utterances) | 35.27 | 8.15 | 7.58 | 7.58 |

`normalize_ext` counts dialect spellings of one word as errors (الي/اللي, انو/انه, كتير/كثير, ...: up to +4 WER on
Levantine, Egyptian and Moroccan sets), keeps laughter, fillers and the `غير_واضح` markup, lacks English number and
contraction rules, and is not idempotent when a standalone `ال` is left (for example from a spelled Latin letter). It is
kept unchanged in `textnorm/arabic_norm.py` for figures comparable with its benchmark. The Arena's L2 normaliser ranks
the 22 providers of the Audioluna review site in the order of the site's human-reviewed, context-adjusted WER
(Spearman 0.997, `normalize_ext` 0.991; `norm_study/gs_score.py`), so the service reports both and uses L2 for
`normalized_text`.

`norm_study/text_side_check.py`: scoring `normalized_text` instead of `text` leaves the Arena WER unchanged (6 of 9 sets
identical, the rest within 0.09), but `normalize_ext` scores it worse (L2 writes numbers as words, `okay` as `اوكي`), so
clients should score `text` with their own normaliser. Collapsing runaway repetitions changed 0-2 utterances per set and
no WER, so the service does not do it. `norm_study/output_conventions.py`: the model writes no punctuation or capitals,
mostly no hamza (21 % of Arabic outputs against 64 % of references) and numbers as digits more often than the references.

## Layout

| Path | |
|---|---|
| `server.py` | the service |
| `static/index.html` | the test page |
| `textnorm/` | normalisers: the Arabic Arena's (vendored) and the Test Arabic ASR Benchmark's `normalize_ext` |
| `wer.py` | WER of the service on a test set (`.jsonl` / `.csv` of audio paths and references) |
| `start.sh` | tmux + restart loop + optional quick tunnel |
| `test/` | `make_clips.py` (public test clips from an Arena checkout), `run_tests.py` (formats, errors, routes), `arena_check.py`, `compare_check.py`, `scorer_check.py`, `lang_check.py` |
| `norm_study/` | the normaliser and input-level studies above |
| `kernels/` | the CUDA kernels (`csrc/`) and their Python side: GPU features, packed encoder, continuous-batching decoder, the scheduling pipeline; Silero run in parallel (see Speed) |
| `bench/` | `latency.py` and `speed.py` (against the model card's code), `stages.py`, `concurrency.py`, `batch_consistency.py` |

The Arena scripts run from a checkout of the Arabic Arena (`github.com/kadirnar/arabic-arena`, branch `scoring-v2`) with
its built data and published predictions (`cd $ARENA_REPO; python /path/to/test/arena_check.py URL`). The Audioluna
scripts read the private call recordings and references from paths given in environment variables and write
transcripts outside the repository.

## Provenance and licences

- `textnorm/arena_normalizers/`: unchanged copy of `sttarena/normalizers` of github.com/kadirnar/arabic-arena
  (branch `scoring-v2`, commit `ccddc3f`).
- `textnorm/arabic_norm.py`: unchanged copy of the Test Arabic ASR Benchmark's normaliser
  (huggingface.co/datasets/kadirnar/smart-turn-arabic-asr-benchmark, CC-BY-4.0).
- The base model is Apache-2.0; per its card, some of the fine-tune's training sets are licensed for non-commercial use
  only or share-alike: check them before commercial use.
