#!/usr/bin/env python3
"""Inputs for scoring the NLLB sweep (needs data/nllb/trans.*.tsv).

data/nllb/pairs.tsv: id, src, text = every (source, translation) pair to force-decode through NLLB: NLLB outputs and human
references. data/nllb/texts.cs.tsv: NLLB outputs the LMs have not scored yet (the rest are in data/wmt_human/surp.cs.*).
"""
from pathlib import Path

import pandas as pd

import measures as M
import udpipe_io as U

ND = U.DATA.parent / "nllb"
CONDS = ["beam1", "beam2", "beam5", "beam10", "beam50", "sample"]


def read_trans(c):
    return pd.read_csv(ND / f"trans.{c}.tsv", sep="\t", dtype={"id": str}, keep_default_na=False, quoting=3)


def main():
    en = U.read_texts("en")
    id2src = dict(zip(en["id"], en["text"]))
    src2id = {v: k for k, v in id2src.items()}
    pairs = set()
    for c in CONDS:
        t = read_trans(c)
        pairs |= {(i, x) for i, x in zip(t["id"], t["text"]) if x}
    nllb_texts = {x for _, x in pairs}
    rows = pd.concat([pd.read_csv(U.DATA / f"wmt{y}.en-cs.tsv", sep="\t", keep_default_na=False) for y in (20, 21, 22)])
    h = rows[rows["is_human"].astype(str) == "True"]
    pairs |= {(src2id[M.norm_text(s)], M.norm_text(x)) for s, x in zip(h["src"], h["hyp"])}
    pairs = sorted(pairs)
    pd.DataFrame([(f"p{k}", id2src[s], x) for k, (s, x) in enumerate(pairs)], columns=["id", "src", "text"]).to_csv(
        ND / "pairs.tsv", sep="\t", index=False, quoting=3)
    known = set(U.read_texts("cs")["text"])
    new = sorted(nllb_texts - known)
    pd.DataFrame({"id": [f"n{k}" for k in range(len(new))], "text": new}).to_csv(ND / "texts.cs.tsv", sep="\t", index=False,
                                                                                 quoting=3)
    print(len(pairs), "pairs,", len(new), "new LM texts")


if __name__ == "__main__":
    main()
