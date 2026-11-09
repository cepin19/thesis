#!/usr/bin/env python3
r"""Original Czech verse (CCV): is surprisal less uniform than in news, and where do the peaks sit?

usage: analyze_ccv.py <lm> [<lm> ...]
  baseline.csv : first 20 scored words of each poem (no preceding context, like segment-level WMT) vs WMT20-22 human references
                 with 15-25 words. Mean, SD, CV, Gini, lv_norm, bootstrap CI of the difference over texts. Poem openings are
                 line-broken; WMT is not.
  rhyme.csv    : line-final word surprisal by rhyme role (first member, response, unrhymed), and within-poem OLS with log
                 characters as covariate, cluster bootstrap over poems. Only poems containing the compared roles count.
  position.csv : word surprisal / poem mean by position in the line (start, middle, final; first word of the poem dropped).
Older Czech (19th c.) inflates surprisal under modern LMs, so mean differences from news are not only about verse.
Writes results/literary/ccv/.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_literary as L
import measures as M
import udpipe_io as U

ROOT = Path(__file__).resolve().parent.parent
D, OUT = ROOT / "data/literary/ccv", ROOT / "results/literary/ccv"
RNG = np.random.default_rng(1)
REPORT = ["mean", "sd", "cv", "gini", "lv_norm"]


def wmt_human(lm):
    rows = pd.concat([pd.read_csv(U.DATA / f"wmt{y}.en-cs.tsv", sep="\t", keep_default_na=False) for y in (20, 21, 22)])
    hum = {M.norm_text(x) for x in rows[rows["is_human"].astype(str) == "True"]["hyp"]}
    t = U.read_texts("cs")
    id2 = dict(zip(t["id"], t["text"]))
    out = []
    for line in open(U.DATA / f"surp.cs.{lm}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        x = id2[r["id"]]
        if x not in hum:
            continue
        s = M.units(x, r["offsets"], r["surp"], "untok")
        if 16 <= len(s) <= 26:
            m = M.measures(s[1:])
            out.append({k: m[k] for k in REPORT})
    return pd.DataFrame(out)


def fe(d, role, base):
    d = d[d.role.isin([role, base])].copy()
    d = d[d.groupby("poem")["role"].transform("nunique") == 2]
    d["x"] = (d.role == role).astype(float)
    d["lc"] = np.log(d.chars)
    dm = d[["s", "x", "lc"]] - d.groupby("poem")[["s", "x", "lc"]].transform("mean")
    pid = d["poem"].values
    u, inv = np.unique(pid, return_inverse=True)
    X, Y = dm[["x", "lc"]].values, dm["s"].values
    A_ = np.zeros((len(u), 2, 2))
    B_ = np.zeros((len(u), 2))
    np.add.at(A_, inv, X[:, :, None] * X[:, None, :])
    np.add.at(B_, inv, X * Y[:, None])
    est = np.linalg.solve(A_.sum(0), B_.sum(0))[0]
    bs = []
    for _ in range(1000):
        i = RNG.integers(0, len(u), len(u))
        bs.append(np.linalg.solve(A_[i].sum(0), B_[i].sum(0))[0])
    return est, *np.percentile(bs, [2.5, 97.5]), len(u)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    text = {i: t.replace("\\n", "\n") for i, t in pd.read_csv(D / "texts_tgt.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)[["id", "text"]].values}
    meta = {r["id"]: r for r in map(json.loads, open(D / "meta.jsonl", encoding="utf-8"))}
    base, rhy, pos = [], [], []
    for lm in sys.argv[1:]:
        open_, ends, posi = [], [], []
        for line in open(D / f"surp.{lm}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            t = text[r["id"]]
            s, ln = L.words(t, r)
            if len(s) >= 21:
                m = M.measures(s[1:21])
                open_.append({k: m[k] for k in REPORT})
            if len(s) < 6 or s.mean() <= 0:
                continue
            wd = [w.group() for w in re.finditer(r"\S+", t)][:len(s)]
            for i in range(1, len(s)):
                p = "final" if i == len(s) - 1 or ln[i + 1] != ln[i] else ("start" if ln[i - 1] != ln[i] else "middle")
                posi.append((r["id"], p, s[i] / s.mean()))
            roles = meta[r["id"]]["rhyme"]
            for i in range(len(s)):
                if (i == len(s) - 1 or ln[i + 1] != ln[i]) and ln[i] < len(roles) and (i < len(s) - 1 or ln[i] == len(roles) - 1):
                    ends.append(dict(poem=r["id"], role=roles[ln[i]], s=s[i], chars=max(len(re.sub(r"\W", "", wd[i])), 1)))
        v, w = pd.DataFrame(open_), wmt_human(lm)
        for m in REPORT:
            i = RNG.integers(0, len(v), (2000, len(v)))
            j = RNG.integers(0, len(w), (2000, len(w)))
            bs = v[m].values[i].mean(1) - w[m].values[j].mean(1)
            base.append(dict(lm=lm, measure=m, verse=v[m].mean(), news_human=w[m].mean(), diff=v[m].mean() - w[m].mean(),
                             lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), n_verse=len(v), n_news=len(w)))
        e = pd.DataFrame(ends)
        for ro, x in e.groupby("role"):
            rhy.append(dict(lm=lm, kind="level", contrast=ro, bits=x.s.mean(), lo=np.nan, hi=np.nan, n=len(x), chars=x.chars.mean()))
        for a, b in [("response", "first"), ("none", "first"), ("none", "response")]:
            est, lo, hi, n = fe(e, a, b)
            rhy.append(dict(lm=lm, kind="within_poem", contrast=f"{a}-{b}", bits=est, lo=lo, hi=hi, n=n, chars=np.nan))
        q = pd.DataFrame(posi, columns=["poem", "pos", "rel"]).groupby(["poem", "pos"])["rel"].mean().unstack()
        for c in q:
            pos.append(dict(lm=lm, pos=c, rel_surprisal=q[c].mean(), n_poems=q[c].notna().sum()))
    pd.DataFrame(base).to_csv(OUT / "baseline.csv", index=False)
    pd.DataFrame(rhy).to_csv(OUT / "rhyme.csv", index=False)
    pd.DataFrame(pos).to_csv(OUT / "position.csv", index=False)
    print("written", OUT)


if __name__ == "__main__":
    main()
