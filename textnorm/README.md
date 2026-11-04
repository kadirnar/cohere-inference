# textnorm

- `arena_normalizers/`: unchanged copy of `sttarena/normalizers` from github.com/kadirnar/arabic-arena (branch
  `scoring-v2`, commit ccddc3f), the Arabic Arena's scoring normaliser (`NORMALIZER_VERSION` in `levels.py`). Needs
  `regex` and `whisper-normalizer==1.0.0`.
- `arabic_norm.py`: unchanged copy of `code/arabic_norm.py` of the Test Arabic ASR Benchmark
  (huggingface.co/datasets/kadirnar/smart-turn-arabic-asr-benchmark, CC-BY-4.0); `normalize_ext` is its normaliser.
  Known quirk: a standalone `ال` (for example from a spelled Latin letter) can disappear on a second pass, so the
  function is not idempotent on such text. It is kept unchanged so that scores stay comparable with that benchmark.
