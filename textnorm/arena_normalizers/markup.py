"""Corpus markup that leaks into references as words.

Arabic corpora mark events, unclear speech and partial words inline, and none of the HF
dataset cards document it. After the v1 normalizer these tags become ordinary reference
words: 4.8 % of MGB-3 reference words (`ال#` partial words, `آ#` filled pauses), 7.2 % of
MGB-5 (`@@@موسيقى`, `غير_واضح`, `###`), 0.6 % of SADA (`cleaned_text` turns `#غير_واضح` into
the word `غيرواضح`). A model that correctly outputs nothing for `@@@فراغ` scored 100 % WER.

The patterns run on raw text, on references and hypotheses alike (some models emit `(music)`).
Parentheses are *not* stripped in general: they hold real speech in FLEURS (clauses), MGB-3
(code-switched English) and Casablanca (French, Qur'an).

A suite can pick patterns per set (`markup: [event_tag, unclear]`, or `none`) and add its own
(`markup_extra: [{pattern: ..., repl: " "}]`); by default every pattern is applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import regex as re

# name -> (pattern, replacement); order matters (unclear tags before the generic '#' rules)
PATTERNS: dict[str, tuple[re.Pattern, str]] = {
    # MGB-5 event tags: @@@موسيقى @@@فراغ @@@ضوضاء @@@ضحك
    "event_tag": (re.compile(r"@@@\S*"), " "),
    # unclear speech: غير_واضح (MGB-3/5), غيرواضح (SADA cleaned_text), with tag prefixes. "غير واضح"
    # with a space is ordinary Arabic ("unclear") and is left alone.
    "unclear": (re.compile(r"[#@]*\(?[وب]?[غع]ير[_\-]?و?\S*واضح\S*\)?|[#@]*\(?[غع]ير_\S*\)?"), " "),
    # English event tags in parentheses (Casablanca); other parentheses are speech
    "paren_event_en": (re.compile(
        r"\((?:music|singing|singinig|laughter|laughs?|noise|applause|silence|crosstalk|inaudible)\)", re.I), " "),
    # partial words / filled pauses marked with '#' inside or at the end of a token (MGB-3/5 'ال#', 'آ#')
    "partial_hash": (re.compile(r"(?<!\S)(?!#)\S*[^\s#]#\S*"), " "),
    # '#'-prefixed tokens: MGB-5 '###word' marks a real word, SADA raw '#آه' a tag word -> keep the word
    "hash_prefix": (re.compile(r"(?<!\S)#{1,3}(?=\S)"), ""),
    # MGB-5 '**'
    "stars": (re.compile(r"\*{2,}"), " "),
}
DEFAULT = tuple(PATTERNS)

# token-level rules applied after the L1 normalizer
LAUGHTER = re.compile(r"^(?:(?:ها){2,}|ه{3,}|(?:هه)+|ههاه\S*|هههه\S*|خخخ+)$")
_STANDALONE_AL = re.compile(r"(?:^|(?<=\s))ال\s+(?=\S)")  # MGB-3 "الـ (word)" -> article + next word


@dataclass(frozen=True)
class MarkupSpec:
    """Which markup patterns to strip, plus optional set-specific (pattern, replacement) pairs."""

    names: tuple[str, ...] = DEFAULT
    extra: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @classmethod
    def parse(cls, spec) -> "MarkupSpec":
        """From suite YAML: None/"default" -> all patterns, "none"/[] -> nothing, or a list of names."""
        if spec is None or spec == "default":
            return cls()
        if isinstance(spec, MarkupSpec):
            return spec
        if spec == "none" or spec == []:
            return cls(names=())
        names = tuple(spec)
        unknown = [n for n in names if n not in PATTERNS]
        if unknown:
            raise ValueError(f"unknown markup patterns {unknown}; known: {', '.join(PATTERNS)}")
        return cls(names=names)

    def with_extra(self, extra: list[dict] | None) -> "MarkupSpec":
        if not extra:
            return self
        return MarkupSpec(self.names, self.extra + tuple((e["pattern"], e.get("repl", " ")) for e in extra))

    def strip(self, text: str) -> str:
        for name in self.names:
            pat, repl = PATTERNS[name]
            text = pat.sub(repl, text)
        for pattern, repl in self.extra:
            text = re.sub(pattern, repl, text)
        return text


def post_l1(text: str) -> str:
    """Token-level markup cleanup on L1-normalized text: attach a standalone article, drop laughter."""
    text = _STANDALONE_AL.sub("ال", text)
    return " ".join(w for w in text.split() if not LAUGHTER.match(w))
