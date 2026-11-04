"""Text normalisers of the endpoint: scoring-normalised output text and WER against a reference.

arena_normalizers  the Arabic Arena's normaliser (github.com/kadirnar/arabic-arena, branch scoring-v2 at ccddc3f,
                   sttarena/normalizers, unchanged): level L2 for Arabic and mixed text (orthography, dialect spelling
                   variants of one word, fillers and laughter, numbers as words, common Latin loanwords) and Whisper's
                   English normaliser for English. Its WER ranks the 22 providers of the Audioluna review site in the
                   same order as the site's context-adjusted WER (Spearman 0.997).
arabic_norm        normalize_ext of the Test Arabic ASR Benchmark (unchanged copy, CC-BY-4.0), for figures comparable
                   with that benchmark and with the earlier Audioluna numbers.
"""

from __future__ import annotations

import functools
import re

from rapidfuzz.distance import Levenshtein

from .arabic_norm import normalize_ext
from .arena_normalizers import get_text_normalizer

ARABIC = re.compile(r"[؀-ۿ]")


@functools.cache
def _arena(language: str):
    return get_text_normalizer(language, "L2")


def language_of(*texts: str) -> str:
    """The normaliser to use: Arabic's (which also handles Latin words) when any text has Arabic letters."""
    return "ar" if any(ARABIC.search(t or "") for t in texts) else "en"


def words(text: str, language: str, norm: str = "l2") -> list[str]:
    if norm == "ext":
        return normalize_ext(text or "").split()
    return _arena(language).words(text or "")


def normalize(text: str, language: str | None = None) -> str:
    """Scoring-normalised text (Arena L2): for matching and WER, not for display."""
    return " ".join(words(text, language or language_of(text)))


def wer(reference: str, hypothesis: str) -> dict:
    """Word error rate of a hypothesis against a reference under both normalisers (edits / reference words)."""
    language = language_of(reference, hypothesis)
    out = {"language_normaliser": language}
    for norm in ("l2", "ext"):
        r, h = words(reference, language, norm), words(hypothesis, language, norm)
        e = Levenshtein.distance(r, h)
        out[norm] = {"wer": round(e / len(r), 4) if r else None, "edits": e, "reference_words": len(r)}
    return out
