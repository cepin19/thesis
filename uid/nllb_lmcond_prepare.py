#!/usr/bin/env python3
"""Inputs for source-conditioned LM scoring, p_LM(y | x), of every pair in data/nllb/pairs.tsv (NLLB outputs + human references).

Writes data/nllb/lmcond/pairs.<prompt>.tsv with columns id, context, text. The text follows the context after one space.
Newlines are escaped as a literal \\n (surprisal.py --unescape-nl).
  en      : English: {src} / Czech:
  cs      : Anglicky: {src} / Česky:
  fewshot : two fixed example pairs, then the en prompt
  en_shuf : the en prompt with another segment's source (fixed derangement over unique sources): control for the prompt itself
"""
import numpy as np
import pandas as pd

import nllb_prepare as P

OUT = P.ND / "lmcond"
SHOTS = ("English: The weather was cold yesterday.\\nCzech: Včera bylo chladno.\\n\\n"
         "English: She did not want to answer the question.\\nCzech: Nechtěla na tu otázku odpovědět.\\n\\n")
PROMPTS = {"en": "English: {}\\nCzech:", "cs": "Anglicky: {}\\nČesky:", "fewshot": SHOTS + "English: {}\\nCzech:", "en_shuf": "English: {}\\nCzech:"}


def main():
    OUT.mkdir(exist_ok=True)
    p = pd.read_csv(P.ND / "pairs.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
    print("texts or sources with a literal backslash-n:", p["text"].str.contains(r"\\n", regex=True).sum(), p["src"].str.contains(r"\\n", regex=True).sum())
    u = sorted(set(p["src"]))
    perm = np.random.default_rng(1).permutation(len(u))
    shuf = {u[perm[k]]: u[perm[(k + 1) % len(u)]] for k in range(len(u))}  # a single cycle: no source maps to itself
    for name, tmpl in PROMPTS.items():
        src = p["src"].map(shuf) if name == "en_shuf" else p["src"]
        pd.DataFrame({"id": p["id"], "context": [tmpl.format(s) for s in src], "text": p["text"]}).to_csv(
            OUT / f"pairs.{name}.tsv", sep="\t", index=False, quoting=3)
    print(len(p), "pairs,", len(u), "sources")


if __name__ == "__main__":
    main()
