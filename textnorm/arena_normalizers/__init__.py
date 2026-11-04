from __future__ import annotations

import unicodedata
import warnings
from typing import Callable

from .arabic import ArabicTextNormalizer
from .levels import LEVELS, NORMALIZER_VERSION, ArabicNormalizer


def get_normalizer(language: str) -> Callable[[str], str]:
    """Return the v1 scoring normalizer for a language code ("ar", "en")."""
    if language == "ar":
        return ArabicTextNormalizer()
    if language == "en":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            from whisper_normalizer.english import EnglishTextNormalizer
        return EnglishTextNormalizer()
    raise ValueError(f"No normalizer for language {language!r}")


class EnglishNormalizer:
    """English has no dialect-spelling levels: L1 and above are the Whisper English normalizer."""

    def __init__(self, level: str = "L1", markup=None, **_):
        if level not in LEVELS:
            raise ValueError(f"unknown level {level!r}; known: {', '.join(LEVELS)}")
        self.level = level
        self._norm = get_normalizer("en")
        self.config = {"level": level, "normalizer": "whisper-english"}

    def tokens(self, text: str) -> list[str]:
        if self.level == "L0":
            return unicodedata.normalize("NFC", text or "").split()
        return self._norm(text or "").split()

    def canon(self, w: str) -> str:
        return w

    words = tokens

    def __call__(self, text: str) -> str:
        return " ".join(self.tokens(text))


def get_text_normalizer(language: str, level: str = "L2", markup=None, **options):
    """A v2 normalizer (tokens / canon / words) for a language at one level."""
    if language == "ar":
        return ArabicNormalizer(level, markup, **options)
    if language == "en":
        return EnglishNormalizer(level, markup, **options)
    raise ValueError(f"No normalizer for language {language!r}")


__all__ = ["ArabicTextNormalizer", "ArabicNormalizer", "EnglishNormalizer", "LEVELS", "NORMALIZER_VERSION",
           "get_normalizer", "get_text_normalizer"]
