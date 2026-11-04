"""Number handling that the v1 verbalizer gets wrong, and number-word tables for the v2 levels.

The v1 normalizer (arabic.py) verbalizes every digit run on its own, so "06:30" became
"صفر سته ثلاثين", "6.5" became "سته خمسه", "50%" lost its spoken "percent", and NFKC turned
"29¾" into "293⁄4". `prepass` rewrites these forms into the words a speaker says *before* the
v1 normalizer runs; the remaining digits are then verbalized as before.

`DIALECT_NUMBERS` maps dialect spellings of number words (Egyptian/Levantine تلاتين, Gulf ثلطعش,
Maghrebi طناش ...) onto the canonical spelling the verbalizer emits. It is applied at level L2.
"""

from __future__ import annotations

import regex as re

_VULGAR = {"½": " ونص", "¼": " وربع", "¾": " وثلاث ارباع", "⅓": " وثلث", "⅔": " وثلثين"}
_EAST = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_TIME = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])")
_DECIMAL = re.compile(r"(?<![\d.٫])(\d+)[.٫](\d+)(?![\d.٫])")
_PERCENT = re.compile(r"(\d)\s*[%٪]")


def _time(m: re.Match) -> str:
    h, mins = int(m.group(1)), int(m.group(2))
    return f" {h} " + (f"و{mins} " if mins else "")


def _decimal(m: re.Match) -> str:
    whole, frac = m.group(1), m.group(2)
    if frac.startswith("0"):  # "3.05" is read digit by digit after the point
        return f" {whole} فاصل " + " ".join(frac) + " "
    return f" {whole} فاصل {frac} "


def prepass(text: str) -> str:
    """Rewrite times, decimals, percentages and vulgar fractions into spoken form (digits stay digits)."""
    for k, v in _VULGAR.items():
        text = text.replace(k, v)
    text = text.translate(_EAST)
    text = _TIME.sub(_time, text)
    text = _DECIMAL.sub(_decimal, text)
    return _PERCENT.sub(r"\1 في الميه ", text)


def _with_waw(table: dict[str, str]) -> dict[str, str]:
    """Add the conjunction-prefixed form of every single-word entry ("وتلاتين" -> "وثلاثين")."""
    out = dict(table)
    for k, v in table.items():
        if " " not in k and not k.startswith("و"):
            out.setdefault("و" + k, "و" + v)
    return out


# Spellings after L1 (ة->ه, hamza dropped). Words that collide with other words are deliberately
# absent: ست ("lady"/six), سبع ("lion"), عشر ("tenth"), تمن ("price"), ميت ("dead").
DIALECT_NUMBERS: dict[str, str] = _with_waw({
    # units and tens (EGY/LEV ت for ث)
    "تلاته": "ثلاثه", "تلات": "ثلاث", "تلاتين": "ثلاثين",
    "تمانيه": "ثمانيه", "تمنيه": "ثمانيه", "تمانين": "ثمانين", "تمنين": "ثمانين",
    "اتنين": "اثنين", "تنين": "اثنين", "ثنين": "اثنين", "ثنتين": "اثنين", "اتنتين": "اثنين",
    # hundreds and thousands
    "تلتميه": "ثلاثميه", "تلاتميه": "ثلاثميه", "تمنميه": "ثمانميه", "تمانميه": "ثمانميه",
    "ربعميه": "اربعميه", "متين": "ميتين", "تلاف": "الاف",
    # teens -> the two-word form the verbalizer emits (EGY/LEV -شر, GLF -عش, MAG -اش)
    "حداشر": "احد عشر", "احداشر": "احد عشر", "حدعشر": "احد عشر", "احدعش": "احد عشر", "حضاش": "احد عشر",
    "اتناشر": "اثني عشر", "اطناشر": "اثني عشر", "اثنعش": "اثني عشر", "ثنعش": "اثني عشر", "طناش": "اثني عشر",
    "تلتاشر": "ثلاثه عشر", "تلطاشر": "ثلاثه عشر", "ثلطعش": "ثلاثه عشر", "تلطاش": "ثلاثه عشر",
    "اربعتاشر": "اربعه عشر", "اربعطاشر": "اربعه عشر", "اربعطعش": "اربعه عشر", "ربعطاش": "اربعه عشر",
    "خمستاشر": "خمسه عشر", "خمسطاشر": "خمسه عشر", "خمسطعش": "خمسه عشر", "خمسطاش": "خمسه عشر",
    "ستاشر": "سته عشر", "سطاشر": "سته عشر", "سطعش": "سته عشر", "سطاش": "سته عشر",
    "سبعتاشر": "سبعه عشر", "سبعطاشر": "سبعه عشر", "سبعطعش": "سبعه عشر", "سبعطاش": "سبعه عشر",
    "تمنتاشر": "ثمانيه عشر", "تمنطاشر": "ثمانيه عشر", "ثمنطعش": "ثمانيه عشر", "تمنطاش": "ثمانيه عشر",
    "تسعتاشر": "تسعه عشر", "تسعطاشر": "تسعه عشر", "تسعطعش": "تسعه عشر", "تسعطاش": "تسعه عشر",
})
