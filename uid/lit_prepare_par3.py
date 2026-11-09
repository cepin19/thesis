#!/usr/bin/env python3
r"""PAR3 (Thai et al. 2022): literary prose, source + Google Translate + 2+ human EN translations -> common format.

Sample: per book at most 50 paragraphs, seed 1. A paragraph qualifies if Google Translate has 20-300 words and every human
translation has 0.5-2x as many words (guards against bad alignments; PAR3 was aligned via Google Translate).
Writes sources.tsv, human.tsv, trans.gt.tsv; lit_collect.py follows. books.tsv maps src_id to book and source language.
The pickle (data/literary/par3_full/Par3_v1/par3.pkl) holds plain containers only; loading approved by the user 2026-11-02.
"""
import pickle
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "data/literary"
D = ROOT / "par3"
D.mkdir(exist_ok=True)
data = pickle.load(open(ROOT / "par3_full/Par3_v1/par3.pkl", "rb"))
rng = random.Random(1)
clean = lambda t: " ".join(str(t).replace("\t", " ").split())
src, hum, gt, books = [], [], [], []
for book in sorted(data):
    b = data[book]
    tr = {k: v["translator_paras"] for k, v in sorted(b["translator_data"].items())}
    ok = []
    for i, g in enumerate(b["gt_paras"]):
        n = len(g.split())
        if not 20 <= n <= 300 or len(tr) < 2:
            continue
        if all(0.5 * n <= len(p[i].split()) <= 2 * n for p in tr.values()):
            ok.append(i)
    rng.shuffle(ok)
    for i in ok[:50]:
        sid = f"{book}#{i}"
        src.append((sid, clean(b["source_paras"][i])))
        gt.append((sid, clean(b["gt_paras"][i])))
        books.append((sid, book, book.rsplit("_", 1)[-1]))
        hum += [(sid, k, clean(p[i])) for k, p in tr.items()]
pd.DataFrame(src, columns=["src_id", "text"]).to_csv(D / "sources.tsv", sep="\t", index=False, quoting=3)
pd.DataFrame(hum, columns=["src_id", "ref", "text"]).to_csv(D / "human.tsv", sep="\t", index=False, quoting=3)
pd.DataFrame(gt, columns=["src_id", "text"]).to_csv(D / "trans.gt.tsv", sep="\t", index=False, quoting=3)
pd.DataFrame(books, columns=["src_id", "book", "lang"]).to_csv(D / "books.tsv", sep="\t", index=False)
print(len(src), "paragraphs from", len({b[1] for b in books}), "books;", len(hum), "human translations")
