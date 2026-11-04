"""Dialect spelling folds for level L2: a same-lexeme variant table (data) and seven audited rules.

Dialect Arabic has no standard orthography, so two careful transcribers of the same MGB-3
Egyptian audio disagree on 13 % of words under the v1 normalizer. Most of that is spelling:
حيبقى/هيبقى (future marker), كانوا/كانو (silent alef), باحب/بحب (1sg b-), قولت/قلت (hollow
verbs), برضو/برضه, ف/في. Folding them on both sides removes noise without hiding recognition
errors: on the five MGB-3 2017 systems the ranking is identical at every level, and system
WER drops 2.1–3.9 points against 6.2 for humans.

Rules carry exception lists; every rule's merges were audited on the union vocabulary of all
reference sets (`sttarena audit variants` re-runs that audit). The table lives in
data/ar_variants.tsv so that it can be reviewed and versioned like the scorer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import regex as re

from .numbers import DIALECT_NUMBERS

VARIANTS_VERSION = "1.0"
_TABLE_PATH = Path(__file__).with_name("data") / "ar_variants.tsv"


@dataclass(frozen=True)
class Variant:
    variant: str
    canonical: str
    dialects: str
    kind: str
    source: str


def load_table(path: Path = _TABLE_PATH) -> list[Variant]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) != 5:
            raise ValueError(f"{path}: expected 5 tab-separated columns, got {len(cols)}: {line!r}")
        rows.append(Variant(*cols))
    keys = [r.variant for r in rows]
    dup = {k for k in keys if keys.count(k) > 1}
    if dup:
        raise ValueError(f"{path}: duplicate variants {sorted(dup)}")
    return rows


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern
    repl: str
    exception: re.Pattern | None = None  # skip the rule when this matches the token
    exception_words: frozenset[str] = frozenset()  # skip when the token minus a و/ف prefix is listed
    note: str = ""

    def apply(self, w: str) -> str:
        if self.exception is not None and self.exception.search(w):
            return w
        if self.exception_words and (w in self.exception_words or w.lstrip("وف") in self.exception_words):
            return w
        return self.pattern.sub(self.repl, w)


_BA_ALEF_EXCEPTIONS = frozenset({
    "باقي", "باين", "بايظ", "بارد", "باسم", "باشا", "بابا", "باريس", "باكستان", "بارك", "باري", "باكر", "باطل",
    "باهي", "بارح", "بالظبط", "باقيه", "باقيين", "باشر", "باحث", "باحثين", "باقه", "بانيو", "باكو", "باتشان",
    "باريه", "باكيت", "باتريك", "بانكو", "بايه", "باني", "بادي", "باول", "بانيه", "باهر", "باسل", "باسمه",
    "باسمها", "بانه", "بانها", "بانهم", "بانك", "بانكم", "بانني", "باننا", "باكرا", "بامكان", "بامكانك",
    "بامكانه", "بامكانها", "بامكانهم", "بامكاننا", "بامكاني", "باعتبار", "باعتباره", "باعتبارها",
    # found by the collision audit (sttarena audit variants): بارده "cold" would become برده -> برضه "also",
    # بايدي "with my hands" would become بيدي "with my hand"
    "بارده", "باردين", "بايدي", "بايديهم", "بايديكم", "بايدينا", "بايديها", "بايديه",
})

RULES: dict[str, Rule] = {r.name: r for r in [
    # future marker ح/ه, optionally with the 1sg alef: حيبقي/هيبقي, حنعمل/هنعمل (EGY/LEV)
    Rule("future_h", re.compile(r"^([وف]?)[حه]ا?(?=[يتن]\S{2,}$)"), r"\1ه",
         exception=re.compile(r"^[وفب]?(?:حيا[هتة]|حين|حتم|حنا?ن|حنين|حيوان|حتي|حتا|هيا|هيئ|هيك|هين|هنا|هناك|هتاف)"),
         note="EGY/LEV future prefix"),
    # 1sg future with alef: حاعمل/هاعمل/هعمل; not before د (Moroccan هاد...)
    Rule("future_h_1sg", re.compile(r"^([وف]?)[حه]ا(?=[^\sيتنلد]\S{2,}$)"), r"\1ه", note="EGY 1sg future"),
    # negation ما...ش with or without alef: مافيش/مفيش, ماعرفش/معرفش
    Rule("neg_ma_sh", re.compile(r"^([وف]?)ما(?=\S{2,}ش$)"), r"\1م", note="EGY/LEV circumfix negation"),
    # silent alef after plural و (alef al-fariqa): كانوا/كانو
    Rule("alef_fariqa", re.compile(r"(?<=\S{2})وا$"), "و", exception=re.compile(r"^(?:[وفبل]?ال|لل)"),
         note="silent alef"),
    # 2pl -كم / -كو: معاكم/معاكو (EGY/GLF)
    Rule("kum_ku", re.compile(r"(?<=\S{2})كم$"), "كو",
         exception=re.compile(r"^(?:[وفبل]?ال|لل)|(?:حكم|حاكم|تحكم|تراكم|محكم|مركم)$"), note="2pl suffix"),
    # 1sg present b- written with alef: باحب/بحب (EGY/LEV), with a list of real ba- words
    Rule("ba_alef", re.compile(r"^([وف]?)با(?=[^\sال]\S{1,}$)"), r"\1ب", exception_words=_BA_ALEF_EXCEPTIONS,
         note="EGY/LEV 1sg b-imperfect"),
    # short u written as و in hollow-verb perfects: قولت/قلت, شوفت/شفت, روحت/رحت
    Rule("hollow_u", re.compile(r"^([وفب]?)([قشرخن])و(?=[لفحم](?:ت|تي|تو|تم|نا)\S*$)"), r"\1\2",
         note="hollow verbs"),
]}
ALL_RULES = tuple(RULES)

# Egyptian فيه ("there is" / "in it") vs في ("in"): -0.29 WER on MGB-3 humans but it merges two
# words, so it is off by default.
FIH = {"فيه": "في"}


class VariantFolder:
    """Maps one L2 token to its canonical spelling: table first, then the rules, then the table again."""

    def __init__(self, rules: tuple[str, ...] = ALL_RULES, table: bool = True, numbers: bool = True,
                 fih: bool = False):
        unknown = [r for r in rules if r not in RULES]
        if unknown:
            raise ValueError(f"unknown rules {unknown}; known: {', '.join(RULES)}")
        self.rules = [RULES[r] for r in rules]
        self.table: dict[str, str] = {}
        if table:
            self.table.update({v.variant: v.canonical for v in load_table()})
        if numbers:
            self.table.update(DIALECT_NUMBERS)
        if fih:
            self.table.update(FIH)
        self.multiword = sorted(((k, v) for k, v in self.table.items() if " " in k), key=lambda kv: -len(kv[0]))
        self._multi_re = [(re.compile(rf"(?<!\S){re.escape(k)}(?!\S)"), v) for k, v in self.multiword]

    def apply_multiword(self, text: str) -> str:
        for pat, v in self._multi_re:
            text = pat.sub(v, text)
        return text

    def __call__(self, w: str) -> str:
        table = self.table
        if w in table:
            return table[w]
        for rule in self.rules:
            w = rule.apply(w)
        if w in table:
            return table[w]
        # a conjunction/preposition prefix on a table word: "وبرضو" -> "وبرضه" (one-word forms only: a prefix
        # glued to a two-word number, "بحضاش" -> "باحد عشر", would be re-folded on a second pass)
        if len(w) > 4 and w[0] in "وفب" and w[1:] in table and " " not in table[w[1:]]:
            return w[0] + table[w[1:]]
        return w
