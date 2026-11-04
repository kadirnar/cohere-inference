# cohere-inference

A speech-to-text service for
[Cohere-AR/cohere-transcribe-ar-sft-v9-full-s86k](https://huggingface.co/Cohere-AR/cohere-transcribe-ar-sft-v9-full-s86k)
(commit `3ec483a713`, a community fine-tune of
[CohereLabs/cohere-transcribe-arabic-07-2026](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026), not
made or endorsed by Cohere): Arabic in many dialects, English and code-switched Arabic-English, phone calls included.
One FastAPI process on one GPU, a test page, an OpenAI-style route, input level normalisation, scoring-normalised text
and WER against a reference; optionally published through a Cloudflare quick tunnel, without a key. The evaluation and
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
About 5 GB of GPU memory.

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
`normalize_ext`; edits, reference words, WER). Limits: 100 MB, 30 minutes of audio, one request at a time on the GPU
with 8 waiting (503 beyond). Audio and transcripts are kept in memory only (uploads over 1 MB spool to `TMPDIR`); the
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
   pauses (Silero) into pieces of at most 25 s, decoded in batches.
5. Greedy decoding, at most 448 new tokens (the card's value). With `language=auto` the decoder prompt stops after
   `<|emo:undefined|>` and the model writes its own language header, per piece; with `ar` / `en` the full prompt with
   `<|pnc|>` is used. No post-processing of the text.

## Validation

All Arena figures are WER v2 (the Arena's own scorer and normaliser, per-utterance errors capped at the reference
length), with the endpoint called over HTTP like any client.

**Same transcripts as the model.** On 2,028 clips (SADA telephony, Casablanca call-centre, EdAcc, FLEURS Arabic) the
endpoint's text equals the model run through the Arena's runner one clip at a time on 100 % of clips; the Arena's runner
at its batch size of 32 reproduces the published predictions on 100 %. Batch 1 and batch 32 differ on 1.6-12.7 % of
clips (bf16 padding) and by at most 0.1 WER. Re-scoring the published predictions with `test/scorer_check.py`
reproduces the published WER v2, v1 and CER exactly.

| Set (clips) | published | endpoint, language forced | endpoint, `auto` |
|---|---|---|---|
| fleurs_ar/clean (428) | 3.72 | 3.67 | 3.66 |
| sada/telephony_v2 (300) | 22.84 | 22.91 | 22.84 |
| casablanca/callcenter_v2 (300) | 39.54 | 39.64 | 39.67 |
| mixat_cs/clean (1,000) | 10.94 | 10.96 | 10.96 |
| cv18_ar/noisy (300) | 8.75 | 8.82 | 8.88 |
| fleurs_en/clean (647) | 5.52 | 5.52 | 5.48 |
| edacc/clean (1,000) | 12.61 | 12.65 | 12.70 |
| earnings22/telephony (300) | 7.45 | 7.41 | 7.43 |
| librispeech_clean/callcenter (300) | 4.53 | 4.53 | 4.54 |

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
