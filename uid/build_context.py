#!/usr/bin/env python3
"""Inputs for context-aware surprisal: every row gets the preceding segments of the same document, from the same
translation (the system's or translator's own output), or from the source for English.
Output: data/wmt_human/texts_ctx.{cs,en}.tsv with columns id, context, text.
cs id = year|system|doc|seg, en id = year|doc|seg."""
import pandas as pd

import measures as M
import udpipe_io as U


def cs_id(r):
    return f"{r.year}|{r.system}|{r.doc_id}|{r.seg_idx}"


def en_id(r):
    return f"{r.year}|{r.doc_id}|{r.seg_idx}"


def main():
    df = pd.concat([pd.read_csv(U.DATA / f"wmt{y}.en-cs.tsv", sep="\t", keep_default_na=False) for y in (20, 21, 22)])
    df = df.sort_values(["year", "system", "doc_id", "seg_idx"])
    assert not df.duplicated(["year", "system", "doc_id", "seg_idx"]).any()
    cs, en = [], []
    for _, g in df.groupby(["year", "system", "doc_id"], sort=False):
        ctx = []
        for r in g.itertuples():
            t = M.norm_text(r.hyp)
            cs.append((cs_id(r), " ".join(ctx), t))
            ctx.append(t.strip())
    for _, g in df.drop_duplicates(["year", "doc_id", "seg_idx"]).groupby(["year", "doc_id"], sort=False):
        ctx = []
        for r in g.itertuples():
            t = M.norm_text(r.src)
            en.append((en_id(r), " ".join(ctx), t))
            ctx.append(t.strip())
    for name, rows in (("cs", cs), ("en", en)):
        pd.DataFrame(rows, columns=["id", "context", "text"]).to_csv(U.DATA / f"texts_ctx.{name}.tsv", sep="\t",
                                                                     index=False, quoting=3, escapechar=None)
        print(name, len(rows))


if __name__ == "__main__":
    main()
