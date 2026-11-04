"""What the s86k endpoint's raw text looks like, next to the references: character classes and token habits over the
published Arena predictions (public data), Arabic and English suites.

    cd $ARENA_REPO && python /path/to/cohere-inference/norm_study/output_conventions.py
"""

import collections
import glob
import json
import re

M = "cohere-transcribe-ar-sft-v9-full-s86k"
R = "results"  # the Arabic Arena checkout this runs from
CLASSES = {
    "hamza forms أ إ آ ؤ ئ ء": "[أإآؤئء]", "alef wasla ٱ": "ٱ", "ta marbuta ة": "ة", "alef maqsura ى": "ى",
    "tatweel ـ": "ـ", "harakat": "[ً-ْ]", "dagger alef": "ٰ", "Arabic-Indic digits": "[٠-٩۰-۹]",
    "Western digits": "[0-9]", "Arabic punctuation ، ؟ ؛": "[،؟؛]", "Latin punctuation": r"[.,!?;:\"()\-]",
    "apostrophe": "['’]", "percent %": "[%٪]", "Latin letters": "[A-Za-z]", "Latin uppercase": "[A-Z]",
    "Persian letters ک ی گ چ پ ژ ڤ": "[کیگچپژڤ]", "special tokens <|..|>": r"<\|", "zero-width / bidi": "[​-‏‪-‮⁦-⁩﻿]",
}


def main():
    for lang in ("ar", "en"):
        texts = {"hyp": [], "ref": []}
        for p in glob.glob(f"{R}/{lang}/{M}/predictions/*.jsonl"):
            if "noise_only" in p:
                continue
            for line in open(p, encoding="utf-8"):
                d = json.loads(line)
                texts["hyp"].append(d["hyp"])
                texts["ref"].append(d["ref"])
        print(f"\n== {lang}: {len(texts['hyp'])} utterances; share of utterances containing each class (hyp / ref)")
        for name, pat in CLASSES.items():
            rx = re.compile(pat)
            h = sum(bool(rx.search(t)) for t in texts["hyp"]) / len(texts["hyp"])
            r = sum(bool(rx.search(t)) for t in texts["ref"]) / len(texts["ref"])
            print(f"  {name:<34}{100 * h:7.2f}%  {100 * r:7.2f}%")
        words = collections.Counter(w for t in texts["hyp"] for w in t.split())
        num = sum(n for w, n in words.items() if re.fullmatch(r"[0-9٠-٩]+([.,][0-9]+)*", w))
        print(f"  hyp tokens {sum(words.values())}, digit tokens {num}")
        reps = sum(1 for t in texts["hyp"] if re.search(r"(\b\S+\b)(\s+\1){4,}", t))
        print(f"  hyps with a word repeated 5+ times in a row: {reps}")
        if lang == "ar":
            latin = collections.Counter(w for t in texts["hyp"] for w in t.split() if re.search("[A-Za-z]", w))
            print("  most common Latin tokens in Arabic hyps:", latin.most_common(15))


if __name__ == "__main__":
    main()
