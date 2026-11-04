# Unchanged copy of code/arabic_norm.py from https://huggingface.co/datasets/kadirnar/smart-turn-arabic-asr-benchmark
# (CC-BY-4.0): the normaliser of the Test Arabic ASR Benchmark, used by scripts/leaderboard.py so its scores are
# comparable with the published ones. Do not edit; asr_api/textnorm.py is the project's own normaliser.
"""Arabic ASR text normalizers.

normalize_elm(text)  -- verbatim copy of the Open Universal Arabic ASR Leaderboard normalizer
                        (github.com/Natural-Language-Processing-Elm/open_universal_arabic_asr_leaderboard, eval.py)
normalize_ext(text)  -- extended normalizer used for the main ranking. It is a superset of the
                        leaderboard rules plus conventions from other public evaluation code:
    * Unicode NFKD + removal of every combining mark (all tashkeel, superscript alef, Quranic marks,
      hamza/madda above/below)                                   [Voxtral-Arabic normalization.py]
    * tatweel removal, alef variants -> ا, ى -> ي, ة -> ه         [Voxtral-Arabic, MGB/SADA/Casablanca practice]
    * ؤ -> و, ئ -> ي, ء removed, پ -> ب, ڤ -> ف                   [leaderboard]
    * Persian/Urdu letter forms ک ی ۀ ھ -> ك ي ه ه
    * Eastern Arabic / Persian digits -> 0-9                      [leaderboard, Voxtral-Arabic]
    * all Unicode punctuation and symbols -> space (never merges words)   [Whisper BasicTextNormalizer]
    * script/digit boundaries split (e.g. و3 -> و 3)              [Voxtral-Arabic]
    * standalone conjunction و attached to the next word          [leaderboard]
    * hesitation fillers (اه، آه، ااه، امم، ممم ...) removed       [Whisper EnglishTextNormalizer drops uh/um]
    * spoken Arabic numbers -> digits, thousands separators removed, decimal point and percent sign
      spelled out, so "1453", "١٤٥٣" and "ألف وأربعمئة وثلاثة وخمسين" all become "1453"
                                                                  [Whisper EnglishNumberNormalizer, Voxtral-Arabic]
Both functions must be applied identically to reference and hypothesis.
"""
import re
import unicodedata


# --------------------------------------------------------------------------------------------
# 1. Open Universal Arabic ASR Leaderboard normalizer (verbatim)
# --------------------------------------------------------------------------------------------
def normalize_elm(text):
    # Remove punctuation
    punctuation = r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~،؛؟]'
    text = re.sub(punctuation, '', text)

    # Remove diacritics
    diacritics = r'[ً-ْ]'  # Arabic diacritical marks (Fatha, Damma, etc.)
    text = re.sub(diacritics, '', text)

    # Normalize Hamzas and Maddas
    text = re.sub('پ', 'ب', text)
    text = re.sub('ڤ', 'ف', text)
    text = re.sub(r'[آ]', 'ا', text)
    text = re.sub(r'[أإ]', 'ا', text)
    text = re.sub(r'[ؤ]', 'و', text)
    text = re.sub(r'[ئ]', 'ي', text)
    text = re.sub(r'[ء]', '', text)

    # Transliterate Eastern Arabic numerals to Western Arabic numerals
    eastern_to_western_numerals = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
    }
    for eastern, western in eastern_to_western_numerals.items():
        text = text.replace(eastern, western)

    # Attach the conjunction و to the following word
    text = re.sub(r'(^|\s)و\s+', r'\1و', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# --------------------------------------------------------------------------------------------
# 2. Extended normalizer
# --------------------------------------------------------------------------------------------
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_LETTERS = str.maketrans({
    "ٱ": "ا", "ٲ": "ا", "ٳ": "ا", "ٵ": "ا",
    "ى": "ي", "ی": "ي", "ې": "ي", "ێ": "ي",
    "ة": "ه", "ۀ": "ه", "ە": "ه", "ھ": "ه", "ہ": "ه",
    "ک": "ك", "ڪ": "ك",
    "پ": "ب", "ڤ": "ف", "ڥ": "ف",
    "ء": "", "ـ": "",  # standalone hamza, tatweel
    "‌": " ", "‍": "", "‎": "", "‏": "", "﻿": "",
})
_ARABIC_LETTERS = r"ء-يٮ-ۓۺ-ۿ"

_FILLER_RE = re.compile(r"^(?:ا+ه*|ا*م{2,}|ه+م{2,}|uh+|um+|hm+|ah+|eh+|mm+)$")

# ---- number words (spelled in normalized orthography: no hamza, ة->ه, ى->ي) ----
_ONE = {"واحد", "واحده", "احد", "احدي"}
_UNITS = {
    **{w: 1 for w in _ONE},
    "صفر": 0,
    "اثنان": 2, "اثنين": 2, "اثنتان": 2, "اثنتين": 2, "اثنا": 2, "اثني": 2, "اثنتا": 2, "اثنتي": 2,
    "اتنين": 2, "تنين": 2, "اتنان": 2,
    "ثلاث": 3, "ثلاثه": 3, "تلات": 3, "تلاته": 3,
    "اربع": 4, "اربعه": 4,
    "خمس": 5, "خمسه": 5,
    "ست": 6, "سته": 6,
    "سبع": 7, "سبعه": 7,
    "ثمان": 8, "ثماني": 8, "ثمانيه": 8, "تمن": 8, "تماني": 8, "تمانيه": 8,
    "تسع": 9, "تسعه": 9,
}
_TEN_WORDS = {"عشر", "عشره"}           # 10, or the teen suffix after a unit
_TEENS = {"حداش": 11, "حداشر": 11, "احداش": 11, "اطناش": 12, "اطناشر": 12, "اتناش": 12, "اتناشر": 12,
          "طناش": 12, "تلتاش": 13, "تلتاشر": 13, "اربعتاش": 14, "اربعتاشر": 14, "ربعتاش": 14,
          "خمستاش": 15, "خمستاشر": 15, "ستاش": 16, "ستاشر": 16, "سبعتاش": 17, "سبعتاشر": 17,
          "تمنتاش": 18, "تمنتاشر": 18, "تمنطاش": 18, "تسعتاش": 19, "تسعتاشر": 19}
_TENS = {}
for v, forms in {20: ["عشرون", "عشرين"], 30: ["ثلاثون", "ثلاثين", "تلاتين"], 40: ["اربعون", "اربعين"],
                 50: ["خمسون", "خمسين"], 60: ["ستون", "ستين"], 70: ["سبعون", "سبعين"],
                 80: ["ثمانون", "ثمانين", "تمانين"], 90: ["تسعون", "تسعين"]}.items():
    for f in forms:
        _TENS[f] = v
_HUNDRED = {"ميه", "مايه", "مئه", "مائه", "ميت", "مئت"}  # 100 (also ×100 after a unit: ثلاث مئة)
_HUNDREDS = {}
for v, forms in {200: ["ميتان", "ميتين", "مايتان", "مايتين", "مئتان", "مئتين", "مائتان", "مائتين"]}.items():
    for f in forms:
        _HUNDREDS[f] = v
for u, stems in {3: ["ثلاث", "تلات", "تلت"], 4: ["اربع", "ربع"], 5: ["خمس"], 6: ["ست"], 7: ["سبع"],
                 8: ["ثمان", "ثماني", "تمن", "تمان"], 9: ["تسع"]}.items():
    for s in stems:
        for h in ["ميه", "مايه", "مئه", "مائه"]:
            _HUNDREDS[s + h] = u * 100
_SCALES = {"الف": 1000, "الاف": 1000, "الوف": 1000, "مليون": 10**6, "ملايين": 10**6, "مليار": 10**9,
           "مليارات": 10**9}
_DUAL_SCALES = {"الفا": 2000, "الفان": 2000, "الفين": 2000, "مليونا": 2 * 10**6, "مليونان": 2 * 10**6,
                "مليونين": 2 * 10**6, "مليارا": 2 * 10**9, "ملياران": 2 * 10**9, "مليارين": 2 * 10**9}
_NUMBER_WORDS = set(_UNITS) | _TEN_WORDS | set(_TEENS) | set(_TENS) | _HUNDRED | set(_HUNDREDS) | set(_SCALES) \
    | set(_DUAL_SCALES)
_PERCENT = {"بالميه", "بالمايه", "بالمئه", "بالمائه"}
_ORDINALS = {"الثاني": 2, "الثانيه": 2, "الثالث": 3, "الثالثه": 3, "الرابع": 4, "الرابعه": 4, "الخامس": 5,
             "الخامسه": 5, "السادس": 6, "السادسه": 6, "السابع": 7, "السابعه": 7, "الثامن": 8, "الثامنه": 8,
             "التاسع": 9, "التاسعه": 9, "العاشر": 10, "العاشره": 10, "الحادي": 1, "الحاديه": 1}
_DEF_TENS = {"ال" + w: v for w, v in _TENS.items()}
_INF = float("inf")

# spelling variants and abbreviations the audio cannot disambiguate
_WORD_MAP = {"ابن": "بن", "فا": "ف", "ايه": "اي", "السوفياتي": "السوفيتي", "السوفياتيه": "السوفيتيه", "ميلاديه": "ميلادي",
             "هجريه": "هجري", "كيلومترا": "كيلومتر", "مترا": "متر", "مربعا": "مربع", "مكعبا": "مكعب",
             "جرام": "غرام", "غراما": "غرام", "كيلوجرام": "كيلوغرام", "كيلوغراما": "كيلوغرام",
             "سنتيمترا": "سنتيمتر", "مليمترا": "مليمتر", "كيلومترات": "كيلومتر", "امتار": "متر",
             "كيلوغرامات": "كيلوغرام", "غرامات": "غرام", "سنتيمترات": "سنتيمتر", "مليمترات": "مليمتر"}
_UNIT_ABBR = {"كم": "كيلومتر", "ملم": "مليمتر", "مم": "مليمتر", "سم": "سنتيمتر", "كغ": "كيلوغرام",
              "كجم": "كيلوغرام", "كلغ": "كيلوغرام", "غ": "غرام", "جم": "غرام", "غم": "غرام", "ه": "ها"}
_LATIN_LETTERS = dict(zip("abcdefghijklmnopqrstuvwxyz",
                          ["ايه", "بي", "سي", "دي", "اي", "اف", "جي", "اتش", "اي", "جاي", "كاي", "ال", "ام", "ان",
                           "او", "بي", "كيو", "ار", "اس", "تي", "يو", "في", "دبليو", "اكس", "واي", "زد"]))


def _component(tok):
    """Return (kind, value) for a number token, else None."""
    if tok.isdigit():
        return "digit", int(tok)
    if tok in _UNITS: return "unit", _UNITS[tok]
    if tok in _TEN_WORDS: return "ten_word", 10
    if tok in _TEENS: return "teen", _TEENS[tok]
    if tok in _TENS: return "ten", _TENS[tok]
    if tok in _HUNDRED: return "hundred", 100
    if tok in _HUNDREDS: return "hundreds", _HUNDREDS[tok]
    if tok in _SCALES: return "scale", _SCALES[tok]
    if tok in _DUAL_SCALES: return "dual", _DUAL_SCALES[tok]
    return None


def _split_conj(tok):
    """'وخمسين' -> (True, 'خمسين') when the remainder is a number word."""
    if tok not in _NUMBER_WORDS and tok.startswith("و") and tok[1:] in _NUMBER_WORDS:
        return True, tok[1:]
    return False, tok


def _place(v):
    """Largest power of ten dividing v (the next additive part must be smaller): 200 -> 100, 1990 -> 10."""
    if v == 0:
        return 1
    p = 1
    while v % (p * 10) == 0:
        p *= 10
    return p


def _parse_numbers(tokens):
    """Collapse spoken / mixed number expressions into digits.
    Arabic composes numbers in descending order (ألف ثمانمئة ستة وتسعون) except units before tens
    (ستة وتسعون), with و between parts; scale words multiply (3 ملايين، ستمئة ألف)."""
    out, i, n = [], 0, len(tokens)
    while i < n:
        conj0, w0 = _split_conj(tokens[i])
        if _component(w0) is None:
            out.append(tokens[i]); i += 1; continue
        total = current = 0
        limit, last, j, used = _INF, None, i, []
        while j < n:
            step = 1
            if j == i:
                conj, w = False, w0
            elif tokens[j] == "و" and j + 1 < n and _component(tokens[j + 1]) is not None:
                conj, w, step = True, tokens[j + 1], 2
            else:
                conj, w = _split_conj(tokens[j])
            c = _component(w)
            if c is None:
                break
            kind, v = c
            additive = kind in ("unit", "teen", "ten", "hundreds", "digit", "ten_word")
            small_unit = last == "unit" or (last == "digit" and 0 < current < 10)
            if j > i:
                if kind == "scale":
                    ok = (not conj and current > 0 and last not in ("scale", "dual")) or (conj and v < limit)
                elif kind == "dual":
                    ok = conj and v / 2 < limit
                elif kind == "hundred":
                    ok = (not conj and small_unit and 0 < current < 10) or (conj and 100 < limit)
                elif kind == "ten_word" and not conj:
                    ok = last == "unit"                                     # teen: ثلاثة عشر
                elif (kind == "ten" or (kind == "digit" and v % 10 == 0 and 10 <= v < 100)) and conj and small_unit:
                    ok = True                                               # units before tens: ستة وتسعون
                else:
                    ok = additive and v < limit and (conj or last in ("scale", "dual", "hundreds", "hundred"))
                if not ok:
                    break
            if kind == "scale":
                if j > i and not conj:
                    total += current * v
                else:
                    total += current + v
                current, limit = 0, v
            elif kind == "dual":
                total += current + v; current, limit = 0, v / 2
            elif kind == "hundred":
                if j > i and not conj and small_unit and 0 < current < 10:
                    current *= 100
                else:
                    current += 100
                limit = 100
            elif kind == "ten_word" and not conj and last == "unit":
                current += 10; limit = 1
            else:
                current += v
                limit = {"hundreds": 100, "unit": 1, "teen": 1, "ten": 1, "ten_word": 1}.get(kind, None) or _place(v)
                if kind == "digit" and v < 10:
                    limit = 1
            used.append(w); last = kind; j += step
        if len(used) == 1:
            w = used[0]
            if w.isdigit() or w in _ONE - {"واحد", "واحده"} or w == "عشر":
                out.append(tokens[i]); i += 1; continue      # keep lone digits verbatim, keep ambiguous words
        if conj0:
            out.append("و")
        out.append(str(total + current))
        i = j
    return out


def _ordinals(toks):
    """الثاني عشر -> 12, السادس والعشرين -> 26, العشرين -> 20, and drop ال before digits (القرن ال12)."""
    out, k = [], 0
    while k < len(toks):
        t = toks[k]
        nxt = toks[k + 1] if k + 1 < len(toks) else ""
        if t in _ORDINALS:
            v = _ORDINALS[t]
            if nxt in _TEN_WORDS:
                out.append(str(v + 10)); k += 2; continue
            if nxt.startswith("و") and nxt[1:] in _DEF_TENS:
                out.append(str(v + _DEF_TENS[nxt[1:]])); k += 2; continue
            if v == 1:
                out.append(t); k += 1; continue
            out.append(str(v)); k += 1; continue
        if t in _DEF_TENS:
            out.append(str(_DEF_TENS[t])); k += 1; continue
        if t == "ال" and nxt.isdigit():
            k += 1; continue
        out.append(t); k += 1
    return out


def _rewrite(toks):
    out, k = [], 0
    while k < len(toks):
        t = toks[k]
        nxt = toks[k + 1] if k + 1 < len(toks) else ""
        if len(t) == 1 and t in _LATIN_LETTERS:
            t = _LATIN_LETTERS[t]
        t = _WORD_MAP.get(t, t).replace("اميرك", "امريك")
        if t == "في" and nxt in {"الميه", "المايه", "المئه", "المائه"}:
            out.append("بالميه"); k += 2; continue
        if t in _PERCENT:
            t = "بالميه"
        if t == "كيلو" and _WORD_MAP.get(nxt, nxt) in {"متر", "غرام", "واط"}:
            out.append("كيلو" + _WORD_MAP.get(nxt, nxt)); k += 2; continue
        if t == "عبد" and nxt.startswith("ال") and len(nxt) > 2:
            out.append("عبد" + nxt); k += 2; continue
        out.append(t); k += 1
    return out


def _after_numbers(toks):
    """Unit / era abbreviations after a number, and the decimal word."""
    out, k = [], 0
    while k < len(toks):
        t = toks[k]
        nxt = toks[k + 1] if k + 1 < len(toks) else ""
        if out and out[-1].isdigit():
            if t == "ق" and nxt == "م":
                out += ["قبل", "الميلاد"]; k += 2; continue
            if t == "م":
                out.append("ميلادي" if 500 <= int(out[-1]) <= 2100 else "متر"); k += 1; continue
            if t in _UNIT_ABBR:
                out.append(_UNIT_ABBR[t]); k += 1; continue
            if t == "فاصل" and nxt.isdigit():
                out.append("فاصله"); k += 1; continue
        out.append(t); k += 1
    # spoken decimals "ثمانية وواحد من عشرة" (8 و 1 من 10) -> 8 فاصله 1; zero groups "000" -> 0 0 0
    res, k = [], 0
    while k < len(out):
        if (k + 4 < len(out) + 0 and k + 4 <= len(out) - 1 and out[k].isdigit() and out[k + 1] == "و"
                and out[k + 2].isdigit() and out[k + 3] == "من" and out[k + 4] in ("10", "100", "1000")):
            res += [out[k], "فاصله", out[k + 2]]; k += 5; continue
        if out[k].isdigit() and len(out[k]) in (2, 3) and set(out[k]) == {"0"}:
            res += ["0"] * len(out[k]); k += 1; continue
        res.append(out[k]); k += 1
    return res


def normalize_ext(text):
    if not text:
        return ""
    text = str(text).replace("²", " مربع ").replace("³", " مكعب ")
    text = unicodedata.normalize("NFKC", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = unicodedata.normalize("NFC", text).lower()
    text = text.translate(_DIGITS).translate(_LETTERS)
    # numeric punctuation before generic punctuation removal
    text = re.sub(r"(?<=\d)[,٬'](?=\d{3}\b)", "", text)
    text = re.sub(r"(?<=\d)[.٫](?=\d)", " فاصله ", text)
    text = re.sub(r"[%٪]", " بالميه ", text)
    # punctuation and symbols -> space
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    # split letter/digit and Arabic/Latin boundaries
    text = re.sub(rf"(?<=[{_ARABIC_LETTERS}a-z])(?=\d)|(?<=\d)(?=[{_ARABIC_LETTERS}a-z])", " ", text)
    text = re.sub(rf"(?<=[{_ARABIC_LETTERS}])(?=[a-z])|(?<=[a-z])(?=[{_ARABIC_LETTERS}])", " ", text)
    toks = [t for t in text.split() if not _FILLER_RE.match(t)]
    toks = _rewrite(toks)
    toks = _ordinals(toks)
    toks = _parse_numbers(toks)
    toks = _after_numbers(toks)
    # attach a standalone و to the following (non-numeric) word, as in the leaderboard normalizer
    out, k = [], 0
    while k < len(toks):
        if toks[k] == "و" and k + 1 < len(toks) and not toks[k + 1].isdigit():
            out.append("و" + toks[k + 1]); k += 2; continue
        out.append(toks[k]); k += 1
    return " ".join(out)


if __name__ == "__main__":
    tests = [
        "بعد سقوط القسطنطينية في يد العثمانيين عام 1453، كانت المورة",
        "بعد سقوط القسطنطينية في يد العثمانيين عام الف واربعمئة وثلاثة وخمسين كانت المورة",
        "عام ١٤٥٣ كانت",
        "حوالي سبعون في المئة.", "حوالي 70%.", "حوالي 70 في المئة", "حوالي سبعين بالمائة",
        "ترجم الكتاب إلى اللاتينية على يد رجل دين غير معروف آآآه في نهاية القرن الثالث عشر",
        "تبلغ كثافة الحديد 7.87 غرام", "تبلغ كثافة الحديد سبعة فاصلة ثمانية وسبعين",
        "ثلاث مئة وخمسة وعشرون", "325", "ثلاثمائة و خمسة و عشرون", "ثلاثة آلاف", "3,000", "الفين واربعه وعشرين",
        "واحد", "واحد وعشرين", "اثنا عشر", "أَلْفَيْن", "و 3 كتب", "وثلاثة كتب", "هم يقولون امم نعم", "يعني اه",
        "ستمئة ألف", "600 ألف", "ألف ثمانمئة ستة وتسعون", "1896", "الف واحد وثلاثون", "1031",
        "3 ملايين ومئتان و87 ألف وخمسمية وتسعون", "3287590", "مليار و225 مليون", "مليار ومئتان وخمسة وعشرون مليون",
        "1990 و 2000", "ستة وسبعة", "عشرين ثلاثين", "القرن ال12", "القرن الثاني عشر", "السادس والعشرين", "26",
        "10 كم", "عشرة كيلومترات", "عام 1453م", "500 ق.م", "كم²", "كيلو مترا مربعا", "فيتامين A", "عبد العزيز",
        "محمد ابن سلمان", "الولايات المتحدة الأميركية", "رقم 0555", "واحد", "1", "الحادي عشر", "العشرين",
        "هَذِهِ الْمَدْرَسَةُ", "مستشفى", "مسؤول", "سئل", "Value Metal", "١٢٫٥",
    ]
    for t in tests:
        print(f"{t!r:60} -> ext={normalize_ext(t)!r}  elm={normalize_elm(t)!r}")
