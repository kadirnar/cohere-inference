"""Arabic text normalization for WER/CER scoring.

Arabic transcripts in public corpora are inconsistent: some carry full diacritics (tashkeel),
some use different hamza/alef spellings, Eastern-Arabic digits, tatweel, Quranic marks and
Arabic punctuation. Without normalization, WER mostly measures orthographic conventions
instead of recognition quality. The rules below are applied identically to references and
hypotheses and follow common practice in Arabic ASR evaluation (MGB, Open Universal Arabic
ASR Leaderboard):

1. Unicode NFKC (folds presentation forms such as U+FEFB to base letters)
2. Remove diacritics (fatha, damma, kasra, tanwin, shadda, sukun, superscript alef, ...)
3. Remove tatweel (kashida) and Quranic annotation marks
4. Unify alef variants  أ إ آ ٱ  -> ا
5. Hamza carriers  ؤ -> و , ئ -> ي ; bare hamza ء dropped ; پ -> ب , ڤ -> ف
   (identical to the Open Universal Arabic ASR Leaderboard normalizer)
6. Alef maqsura  ى -> ي ; ta marbuta  ة -> ه  (stricter than that leaderboard: these are
   spelled inconsistently in dialectal transcripts and carry almost no recognition signal)
7. Eastern-Arabic / Persian digits -> ASCII digits
8. Numbers are verbalized: "25" -> "خمسه وعشرين" (see below)
9. Drop punctuation and symbols (Arabic and Latin), lowercase Latin (code-switching)
10. Attach a standalone conjunction و to the following word ("و ذهب" -> "وذهب")
11. Number words are folded to one spelling ("عشرون" -> "عشرين", "مائة" -> "ميه")
12. Collapse whitespace

Numbers (8, 11): references almost always spell numbers out, while many models write digits,
so "25" against "خمسة وعشرون" would count as two errors for a correct transcript. Digits are
written out in the genitive/accusative form that dialectal speech uses ("عشرين", "الفين"),
and the nominative MSA forms of number words ("عشرون", "ألفان", "مائتان") are folded to it on
both sides. Digit strings with a leading zero or more than 12 digits (phone and account
numbers) are read digit by digit. Feminine counting forms ("ثلاث عشرة") and dialect-only
spellings ("تلاتين") are left as they are.
"""

from __future__ import annotations

import unicodedata

import regex as re

_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u08D3-\u08FF]")
_TATWEEL = "ـ"
_ALEF = re.compile(r"[آأإٱٲٳ]")
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
# Every Unicode punctuation / symbol char becomes a space (covers ، ؛ ؟ « » etc.)
_PUNCT = re.compile(r"[\p{P}\p{S}]")
# Bracketed annotations frequently found in corpora: [laughter], <noise>, (inaudible)
_TAGS = re.compile(r"\[[^\]]*\]|<[^>]*>|\{[^}]*\}")
_SPACES = re.compile(r"\s+")
_LETTERS = str.maketrans({"ؤ": "و", "ئ": "ي", "ء": None, "پ": "ب", "ڤ": "ف"})
_WAW = re.compile(r"(?:^|(?<=\s))و\s+(?=\S)")

# --- numbers (already-normalized spellings: ة->ه, أ->ا, ئ->ي) ---
_ONES = ["صفر", "واحد", "اثنين", "ثلاثه", "اربعه", "خمسه", "سته", "سبعه", "ثمانيه", "تسعه"]
_TEENS = ["عشره", "احد عشر", "اثني عشر", *(f"{w} عشر" for w in _ONES[3:])]
_STEMS = ["", "", "", "ثلاث", "اربع", "خمس", "ست", "سبع", "ثمان", "تسع"]
_TENS = ["", "", "عشرين", *(f"{s}ين" for s in _STEMS[3:])]
_HUNDREDS = ["", "ميه", "ميتين", *(f"{s}ميه" for s in _STEMS[3:])]
_SCALES = [(10**9, "مليار", "مليارين", "مليارات"), (10**6, "مليون", "مليونين", "ملايين"), (1000, "الف", "الفين", "الاف")]
_THOUSANDS_SEP = re.compile(r"(?<=\d)[,٬](?=\d{3}(?!\d))")
_NUMBER = re.compile(r"\d+")
_SPLIT_HUNDREDS = re.compile(r"(?<!\S)(و?)(ثلاث|اربع|خمس|ست|سبع|ثمان|تسع)\s(?:مايه|ميه)(?!\S)")
_NUMBER_WORDS = {
    **{f"{s}ون": f"{s}ين" for s in ["عشر", *_STEMS[3:]]},
    **{f"{s}مايه": f"{s}ميه" for s in _STEMS[3:]},
    "اثنان": "اثنين", "اثنا": "اثني", "مايه": "ميه", "مايتين": "ميتين", "مايتان": "ميتين", "ميتان": "ميتين",
    "الفان": "الفين", "مليونان": "مليونين", "ملياران": "مليارين",
}


def _below_1000(n: int) -> list[str]:
    h, r = divmod(n, 100)
    parts = [_HUNDREDS[h]] if h else []
    if 0 < r < 10:
        parts.append(_ONES[r])
    elif 10 <= r < 20:
        parts.append(_TEENS[r - 10])
    elif r:
        t, o = divmod(r, 10)
        parts += [_ONES[o], _TENS[t]] if o else [_TENS[t]]
    return parts


def number_to_words(digits: str) -> str:
    """Arabic words for an ASCII digit string, in the normalized spelling used for scoring."""
    if len(digits) > 12 or (len(digits) > 1 and digits[0] == "0"):
        return " ".join(_ONES[int(d)] for d in digits)
    n = int(digits)
    if n == 0:
        return _ONES[0]
    parts = []
    for value, one, two, plural in _SCALES:
        q, n = divmod(n, value)
        if q == 1:
            parts.append(one)
        elif q == 2:
            parts.append(two)
        elif q:
            parts.append(" و ".join(_below_1000(q)) + " " + (plural if q <= 10 else one))
    parts += _below_1000(n)
    return " و ".join(parts)


def _fold_number_words(text: str) -> str:
    text = _SPLIT_HUNDREDS.sub(r"\1\2ميه", text)
    out = []
    for tok in text.split(" "):
        if tok in _NUMBER_WORDS:
            tok = _NUMBER_WORDS[tok]
        elif tok[:1] == "و" and tok[1:] in _NUMBER_WORDS:
            tok = "و" + _NUMBER_WORDS[tok[1:]]
        out.append(tok)
    return " ".join(out)


class ArabicTextNormalizer:
    def __init__(self, normalize_ta_marbuta: bool = True, normalize_alef_maqsura: bool = True,
                 verbalize_numbers: bool = True):
        self.normalize_ta_marbuta = normalize_ta_marbuta
        self.normalize_alef_maqsura = normalize_alef_maqsura
        self.verbalize_numbers = verbalize_numbers

    def __call__(self, text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFKC", text)
        text = _TAGS.sub(" ", text)
        text = _DIACRITICS.sub("", text)
        text = text.replace(_TATWEEL, "")
        text = _ALEF.sub("ا", text)
        text = text.translate(_LETTERS)
        if self.normalize_alef_maqsura:
            text = text.replace("ى", "ي")
        if self.normalize_ta_marbuta:
            text = text.replace("ة", "ه")
        text = text.translate(_DIGITS)
        if self.verbalize_numbers:
            text = _THOUSANDS_SEP.sub("", text)
            text = _NUMBER.sub(lambda m: f" {number_to_words(m.group())} ", text)
        text = _PUNCT.sub(" ", text)
        text = _WAW.sub("و", _SPACES.sub(" ", text.lower()).strip())
        return _fold_number_words(text) if self.verbalize_numbers else text
