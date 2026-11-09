#!/usr/bin/env python3
r"""Surprisal uniformity on literary sets: human vs MT, per source, under each LM.

usage: analyze_literary.py <set> <lm> [<lm> ...] [--prose] [--nopunct]

Words are whitespace units; whitespace-only tokens (newlines) are excluded, so a line's first word does not inherit the
surprisal of the line break. Measures per text: mean, sd, cv, gini, lv, lv_norm (whole text; first word dropped).
Verse only (texts with line breaks):
  line_cv / line_lv_norm : per line, averaged weighted by line length (LV never crosses a line break)
  between_line : SD of line means / text mean
  lineend_excess : (mean line-final word - mean other words) / text mean
  peak_lift : share of the text's top-10% words that are line-final / share of all words that are line-final
              (> 1: the peaks cluster at line ends)
  lastline_excess : (mean last line - mean other lines) / text mean
  ac1 : lag-1 autocorrelation of word surprisal
Output in results/literary/<set>/:
  levels.csv    : mean of each measure by LM and condition
  contrasts.csv : MT - human by LM; raw (paired bootstrap over sources) and adjusted for log length + mean surprisal
                  (within-source regression, not for `mean`); all in human SDs
  tracking.csv  : r(source, target) of each measure, and MT - human with a bootstrap CI (LMs that scored the sources)
  spread.csv    : sources with 2+ human translations: mean |human - human| vs |human - MT| in the same units
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_wmt_human as A
import measures as M

ROOT = Path(__file__).resolve().parent.parent
SPREAD = ["mean", "sd", "cv", "gini", "lv", "lv_norm"]
VERSE = ["line_cv", "line_lv_norm", "between_line", "lineend_excess", "peak_lift", "lastline_excess", "ac1"]
MIN_WORDS = 5


def words(text, rec, unit="untok"):
    """Word surprisals and each word's line index, without whitespace-only tokens or words past a truncation.

    unit="nopunct" drops punctuation (a full stop MT adds to a line otherwise inflates the line-final word)."""
    keep = [(o, s) for o, s in zip(rec["offsets"], rec["surp"]) if text[o[0]:o[1]].strip()]
    if not keep:
        return np.array([]), np.array([], dtype=int)
    off, sur = zip(*keep)
    vals = M.units(text, off, sur, unit)
    starts = M.unit_starts(text, unit)
    n = int((starts < max(b for _, b in off)).sum())
    breaks = np.array([m.start() for m in re.finditer(r"\n", text)], dtype=int)
    return vals[:n], np.searchsorted(breaks, starts[:n])


def text_measures(s, line, verse):
    if len(s) < MIN_WORDS + 1:
        return None
    m = M.measures(s[1:])
    r = {k: m[k] for k in ["n"] + SPREAD}
    if not verse or line.max() < 1:
        return r
    parts = [s[line == i] for i in np.unique(line)]
    good = [p for p in parts if len(p) >= 2]
    if not good or s.mean() <= 0:
        return r
    sm = M.sentence_measures(good)
    r["line_cv"], r["line_lv_norm"] = sm["cv"], sm["lv_norm"]
    r["between_line"] = np.std([p.mean() for p in parts]) / s.mean()
    is_end = np.r_[line[1:] != line[:-1], True]
    r["lineend_excess"] = (s[is_end].mean() - s[~is_end].mean()) / s.mean() if (~is_end).any() else np.nan
    top = s >= np.quantile(s, 0.9)
    r["peak_lift"] = is_end[top].mean() / is_end.mean()
    last = line == line.max()
    r["lastline_excess"] = (s[last].mean() - s[~last].mean()) / s.mean()
    r["ac1"] = np.corrcoef(s[:-1], s[1:])[0, 1] if s.std() > 0 else np.nan
    return r


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    name, lms = args[0], args[1:]
    verse = "--prose" not in sys.argv
    unit = "nopunct" if "--nopunct" in sys.argv else "untok"
    D, OUT = ROOT / "data/literary" / name, ROOT / "results/literary" / (name + ("_nopunct" if unit == "nopunct" else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    rd = lambda p: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    text = {i: t.replace("\\n", "\n") for i, t in zip(*rd(D / "texts.tsv")[["id", "text"]].values.T)}
    conds, src_ids = rd(D / "conds.tsv"), rd(D / "src_ids.tsv")
    measures_ = SPREAD + (VERSE if verse else [])
    lev, con, trk, spr = [], [], [], []
    for lm in lms:
        vals = {}
        for line in open(D / f"surp.{lm}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            s, ln = words(text[r["id"]], r, unit)
            vals[r["id"]] = text_measures(s, ln, verse)
        got = [dict(src_id=a, cond=b, ref=c, **vals[t]) for a, b, c, t in conds.values if vals.get(t)]
        df = pd.DataFrame(got)
        for m in measures_:
            if m not in df:
                df[m] = np.nan
        for c, x in df.groupby("cond"):
            lev.append(dict(lm=lm, cond=c, n_texts=len(x), n_src=x.src_id.nunique(), words=x.n.mean(), **x[measures_].mean()))
        seg = df.groupby(["src_id", "cond"])[["n"] + measures_].mean().reset_index()
        hsd = df[df.cond == "human"][measures_].std()
        s_df = None
        if (D / "texts_tgt.tsv").exists() and lm in ("eurollm9b", "bloom1b7"):
            s_df = pd.DataFrame([dict(src_id=a, **vals[t]) for a, t in src_ids.values if vals.get(t)]).set_index("src_id")
        for c in sorted(set(df.cond) - {"human"}):
            p = seg[seg.cond.isin([c, "human"])].pivot(index="src_id", columns="cond")
            x = df[df.cond.isin([c, "human"])].copy()
            x["seg"], x["is_c"], x["log_n"] = x["src_id"], (x["cond"] == c).astype(float), np.log(x["n"])
            for m in measures_:
                dd = (p[m][c] - p[m]["human"]).dropna().values
                if len(dd) < 10:
                    continue
                est, lo, hi = A.boot_ci(dd)
                y = x.dropna(subset=[m, "mean"])
                xs = ["is_c", "log_n"] + ([] if m == "mean" else ["mean"])
                ae, al, ah, ns = A.fe_regression(y, m, xs, B=500)
                con.append(dict(lm=lm, cond=c, measure=m, n_src=len(dd), raw=est, lo=lo, hi=hi, raw_sd=est / hsd[m],
                                adj_sd=ae / hsd[m], adj_lo=al / hsd[m], adj_hi=ah / hsd[m]))
            if s_df is not None:
                for m in [k for k in measures_ if k in s_df and k != "mean"]:
                    w = pd.DataFrame({"s": s_df[m], "c": p[m][c], "h": p[m]["human"]}).dropna()
                    if len(w) < 30:
                        continue
                    rng = np.random.default_rng(1)
                    rc = lambda q: (np.corrcoef(q.s, q.c)[0, 1], np.corrcoef(q.s, q.h)[0, 1])
                    bs = []
                    for _ in range(1000):
                        q = w.iloc[rng.integers(0, len(w), len(w))]
                        a, b = rc(q)
                        bs.append(a - b)
                    a, b = rc(w)
                    trk.append(dict(lm=lm, cond=c, measure=m, n=len(w), r_mt=a, r_human=b, diff=a - b,
                                    lo=np.quantile(bs, .025), hi=np.quantile(bs, .975)))
        # human-human spread vs human-MT spread on the same sources
        multi = df[df.cond == "human"].groupby("src_id").filter(lambda x: len(x) >= 2)
        if len(multi):
            for m in ["mean", "sd", "cv", "gini", "lv_norm"]:
                hh = multi.groupby("src_id")[m].apply(lambda v: np.abs(np.subtract.outer(v.values, v.values))[np.triu_indices(len(v), 1)].mean())
                hm = multi.groupby("src_id")[m].mean()
                for c in sorted(set(df.cond) - {"human"}):
                    mt = seg[seg.cond == c].set_index("src_id")[m]
                    w = pd.DataFrame({"hh": hh, "hbar": hm, "mt": mt}).dropna()
                    spr.append(dict(lm=lm, cond=c, measure=m, n_src=len(w), human_human=w.hh.mean(),
                                    human_mt=(w.hbar - w.mt).abs().mean(), mt_inside=np.nan))
    pd.DataFrame(lev).to_csv(OUT / "levels.csv", index=False)
    pd.DataFrame(con).to_csv(OUT / "contrasts.csv", index=False)
    pd.DataFrame(trk).to_csv(OUT / "tracking.csv", index=False)
    pd.DataFrame(spr).drop(columns="mt_inside", errors="ignore").to_csv(OUT / "spread.csv", index=False)
    print("written", OUT)


if __name__ == "__main__":
    main()
