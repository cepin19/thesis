#!/usr/bin/env python3
r"""Mendeley poems (10.17632/gpmg8hdshh.1), EN->FR: write sources.tsv (src_id, text) and human.tsv (src_id, ref, text).

Lines inside a poem are joined by a literal \n. A source with several translations keeps them all (ref = h0, h1, ...).
"""
import glob
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parent.parent / "data/literary/mendeley"


def clean(t):
    return "\\n".join(" ".join(l.split()) for l in t.replace("\t", " ").strip().split("\n"))


df = pd.read_csv(glob.glob(str(D / "english_french_poems-*.csv"))[0], keep_default_na=False)
df["s"], df["t"] = df["original_poem"].map(clean), df["translated_poem"].map(clean)
ids = {s: f"p{i:03d}" for i, s in enumerate(df["s"].drop_duplicates())}
pd.DataFrame({"src_id": list(ids.values()), "text": list(ids)}).to_csv(D / "sources.tsv", sep="\t", index=False, quoting=3)
df["src_id"] = df["s"].map(ids)
df["ref"] = "h" + df.groupby("src_id").cumcount().astype(str)
df[["src_id", "ref", "t"]].rename(columns={"t": "text"}).to_csv(D / "human.tsv", sep="\t", index=False, quoting=3)
print(len(ids), "sources,", len(df), "human translations")
