"""Multi-level Arabic normalization for scoring (normalizer v2).

Every level includes the previous one and is applied identically to references and hypotheses.

=====  ==================  =============================================================  ==========
level  name                adds                                                           MGB-3 floor
=====  ==================  =============================================================  ==========
L0     raw                 NFC + whitespace split                                         20.8 %
L1     orthographic        the v1 arena normalizer + safe fixes: spoken times, decimals,   13.2 %
                           percent and fractions; ڨ ڭ گ→ق, ڢ→ف, ک→ك, ی→ي ...; a space at
                           Arabic|Latin boundaries; Latin accents; final يي→ي
L1m    corpus markup       @@@ event tags, غير_واضح, '#' partial words, laughter tokens   10.7 %
L2     dialect-lenient     fillers and interjections, elongation, a same-lexeme variant    8.6 %
                           table, seven audited dialect rules, dialect number words       (7.0 % with
                                                                                          space-insensitive
                                                                                          alignment)
L3     consonant skeleton  ث→ت, ذ→د, ظ→ض, ق→ا, vowel letters dropped. Diagnostic only:     4.6 %
                           it merges ده/دي and هو/هي
=====  ==================  =============================================================  ==========

The floor column is the pairwise WER between four human transcribers of the same MGB-3 audio:
the share of "errors" that is only spelling. `tokens()` gives surface tokens and `canon()` the
folded form of one token (or of a concatenation, for space-insensitive alignment); `words()` is
the canonical word sequence used for plain WER at that level.
"""

from __future__ import annotations

import functools
import unicodedata

import regex as re

from .arabic import ArabicTextNormalizer
from .markup import LAUGHTER, MarkupSpec, post_l1
from .numbers import prepass
from .variants import ALL_RULES, VARIANTS_VERSION, VariantFolder

LEVELS = ("L0", "L1", "L1m", "L2", "L3")
NORMALIZER_VERSION = f"ar-2.0+variants-{VARIANTS_VERSION}"

# --- L1 safe fixes ---------------------------------------------------------------------------
_EXTRA_LETTERS = str.maketrans({
    "ڨ": "ق", "ڧ": "ق", "ڭ": "ق", "گ": "ق",  # /g/ from ق (Maghrebi ڨ ڭ, Iraqi/Gulf گ), as CODA* spells it
    "ڢ": "ف", "ک": "ك", "ی": "ي", "ې": "ي", "ے": "ي", "ە": "ه", "ھ": "ه", "ۀ": "ه", "ۃ": "ه",
    "ٹ": "ت", "ڈ": "د", "ژ": "ز", "چ": "ج",
})
_SCRIPT_BOUNDARY = re.compile(r"(?<=\p{Arabic})(?=[A-Za-z])|(?<=[A-Za-z])(?=\p{Arabic})")
_FINAL_YY = re.compile(r"(?<=\S)يي(?!\S)")
# zero-width and bidi controls (RLM/LRM/ZWJ ...); a zero-width space separates words
_ZWSP = re.compile(r"[​⁠﻿]")
_FORMAT = re.compile(r"\p{Cf}")
# digits glued to Latin letters are part of a Latin-script word (Arabizi "3aref", "mp3", "4G"):
# shield them from the number verbalizer with private-use code points
_LATIN_DIGIT_TOKEN = re.compile(r"[A-Za-z]+\d[\dA-Za-z]*|\d+[A-Za-z][\dA-Za-z]*")
_SHIELD = str.maketrans("0123456789", "".join(chr(0xE030 + d) for d in range(10)))
_UNSHIELD = str.maketrans("".join(chr(0xE030 + d) for d in range(10)), "0123456789")


def _strip_latin_accents(s: str) -> str:
    out: list[str] = []
    for ch in unicodedata.normalize("NFD", s):
        if unicodedata.category(ch) == "Mn" and out and "a" <= out[-1].lower() <= "z":
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


# --- L2 token filters ------------------------------------------------------------------------
_FILLER = re.compile(r"^(?:ا+|ا*م{2,}|م{2,}|ا+ه{2,}|ه{3,}|ا{2,}ه|ه+م{2,}|ه{2,}م+|ء+|اوو+|ايي+)$")
# Hesitations and interjections. ها, ام, اي and اف are left alone: they are also words
# (اي "yes" in Gulf/Levantine, ام "mother"/"or").
INTERJECTIONS = {
    "safe": frozenset({"اه", "اها", "اوه", "هه", "يوه", "اوف", "اهه", "هاه"}),
    "all": frozenset({"اه", "اها", "ها", "اوه", "هه", "يوه", "اف", "اوف", "ام", "اهه", "هاه", "اي"}),
    "none": frozenset(),
}
_ELONGATION = re.compile(r"(\p{L})\1{2,}")

# --- L3 skeleton ------------------------------------------------------------------------------
_SKELETON_MAP = str.maketrans({"ث": "ت", "ذ": "د", "ظ": "ض", "ق": "ا"})
_FINAL_VOWEL = re.compile(r"(?<=\S)[هاوي]$")
_MEDIAL_VOWEL = re.compile(r"(?<=\S)[اوي]")
_DOUBLE = re.compile(r"(\p{L})\1+")
_HAS_ARABIC = re.compile(r"\p{Arabic}")


def skeleton(w: str) -> str:
    if not _HAS_ARABIC.search(w):
        return w
    w = w.translate(_SKELETON_MAP)
    w = _FINAL_VOWEL.sub("", w)
    w = _MEDIAL_VOWEL.sub("", w)
    return _DOUBLE.sub(r"\1", w)


class ArabicNormalizer:
    """Normalizer v2 at one level. Call it on text to get the canonical string."""

    def __init__(self, level: str = "L2", markup=None, *, rules: tuple[str, ...] = ALL_RULES,
                 variants: bool = True, numbers: bool = True, fillers: bool = True,
                 interjections: str = "safe", elongation: bool = True, fih: bool = False):
        if level not in LEVELS:
            raise ValueError(f"unknown level {level!r}; known: {', '.join(LEVELS)}")
        if interjections not in INTERJECTIONS:
            raise ValueError(f"interjections must be one of {', '.join(INTERJECTIONS)}")
        self.level = level
        self._rank = LEVELS.index(level)
        self.markup = MarkupSpec.parse(markup)
        self._v1 = ArabicTextNormalizer()
        self.fillers, self.elongation = fillers, elongation
        self.interjections = INTERJECTIONS[interjections]
        self.folder = VariantFolder(rules, table=variants, numbers=numbers, fih=fih) if self._rank >= 3 else None
        self.config = {"level": level, "markup": list(self.markup.names), "rules": list(rules) if self.folder else [],
                       "variants": variants, "numbers": numbers, "fillers": fillers, "interjections": interjections,
                       "elongation": elongation, "fih": fih, "version": NORMALIZER_VERSION}
        self.canon = functools.lru_cache(maxsize=1 << 18)(self._canon)

    def __repr__(self) -> str:
        return f"ArabicNormalizer({self.level!r})"

    # -- surface tokens ------------------------------------------------------------------------
    def l1(self, text: str) -> str:
        text = _FORMAT.sub("", _ZWSP.sub(" ", text or ""))
        text = _LATIN_DIGIT_TOKEN.sub(lambda m: m.group().translate(_SHIELD), text)
        text = prepass(text)
        text = text.translate(_EXTRA_LETTERS)
        text = _SCRIPT_BOUNDARY.sub(" ", text)
        text = _strip_latin_accents(text)
        return _FINAL_YY.sub("ي", self._v1(text)).translate(_UNSHIELD)

    def tokens(self, text: str) -> list[str]:
        if not text:
            return []
        if self._rank == 0:
            return unicodedata.normalize("NFC", text).split()
        if self._rank >= 2:
            text = self.markup.strip(text)
        t = self.l1(text)
        if self._rank == 1:
            return t.split()
        t = post_l1(t)
        if self._rank == 2:
            return t.split()
        t = self.folder.apply_multiword(t)
        out = []
        for w in t.split():
            if self.fillers and _FILLER.match(w):
                continue
            if self.elongation:
                w = _ELONGATION.sub(r"\1", w)
                if (self.fillers and _FILLER.match(w)) or LAUGHTER.match(w):  # هاهااا is laughter once collapsed
                    continue
            if w in self.interjections:
                continue
            out.append(w)
        return out

    # -- canonical form ------------------------------------------------------------------------
    def _canon(self, w: str) -> str:
        if self.folder is None:
            return w
        w = self.folder(w)
        if self._rank == 4:
            w = " ".join(skeleton(x) for x in w.split())
        return w

    def words(self, text: str) -> list[str]:
        if self.folder is None:
            return self.tokens(text)
        out: list[str] = []
        for w in self.tokens(text):
            out.extend(self.canon(w).split())
        return out

    def __call__(self, text: str) -> str:
        return " ".join(self.words(text))
