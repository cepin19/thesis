#!/usr/bin/env python3
r"""Collect one literary set into the common format.

Reads data/literary/<set>/{sources.tsv (src_id, text), human.tsv (src_id, ref, text), trans.<cond>.tsv (src_id, text)}.
Writes conds.tsv (src_id, cond, ref, text_id), texts.tsv (id, text: all unique texts incl. sources) and
texts_tgt.tsv (target-language texts only, for LMs that only know the target language).
"""
import sys
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parent.parent / "data/literary" / sys.argv[1]
rd = lambda p: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3)
src = rd(D / "sources.tsv")
h = rd(D / "human.tsv").assign(cond="human")
parts = [h]
for p in sorted(D.glob("trans.*.tsv")):
    c = p.name.split(".")[1]
    parts.append(rd(p).assign(cond=c, ref=c))
t = pd.concat(parts)
t = t[t["text"].str.strip() != ""]
ids = {x: f"t{i:06d}" for i, x in enumerate(pd.concat([src["text"], t["text"]]).drop_duplicates())}
t["text_id"] = t["text"].map(ids)
src["text_id"] = src["text"].map(ids)
t[["src_id", "cond", "ref", "text_id"]].to_csv(D / "conds.tsv", sep="\t", index=False)
src[["src_id", "text_id"]].to_csv(D / "src_ids.tsv", sep="\t", index=False)
pd.DataFrame({"id": list(ids.values()), "text": list(ids)}).to_csv(D / "texts.tsv", sep="\t", index=False, quoting=3)
tg = t.drop_duplicates("text_id")
pd.DataFrame({"id": tg["text_id"], "text": tg["text"]}).to_csv(D / "texts_tgt.tsv", sep="\t", index=False, quoting=3)
print(sys.argv[1], len(src), "sources;", t.groupby("cond").size().to_dict(), ";", len(ids), "unique texts")
