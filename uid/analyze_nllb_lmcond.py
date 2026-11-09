#!/usr/bin/env python3
"""Uniformity of the NLLB sweep and human references under source-conditioned LMs, p_LM(y|x) (data/nllb/lmcond/).

Same segments, conditions and statistics as analyze_nllb.py, for every LM x prompt (en, cs, fewshot, en_shuf). Word level (untok) and
LM tokens, first unit dropped. Writes results/nllb_lmcond/:
  levels    : mean of every measure per LM, prompt, unit, condition
  contrasts : condition - human and condition - beam1, raw and adjusted for log length + mean surprisal, in human SDs
  gain      : per condition, mean bits per word without source p(y) minus with source, and with the shuffled source; the difference
              (shuf - real) is the effect of the source itself. r_nllb = mean per-segment r(word surprisal, NLLB's word surprisal)
  pmi       : per word, s_LM = s_LM|x + PMI_LM; per-segment variance decomposition under each LM's own conditional
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_wmt_human as A
import measures as M
import nllb_prepare as P
import udpipe_io as U

OUT = Path(__file__).resolve().parent.parent / "results/nllb_lmcond"
LD = P.ND / "lmcond"
LMS = ["czgpt2xl", "eurollm9b", "customgpt2"]
PROMPTS = ["en", "cs", "fewshot", "en_shuf"]
ORDER = P.CONDS + ["human"]
REPORT = ["mean", "sd", "cv", "gini", "lv", "lv_norm"]


def load(path):
    return {r["id"]: r for r in map(json.loads, open(path, encoding="utf-8"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lms", nargs="+", default=LMS)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = pd.concat([pd.read_csv(U.DATA / f"wmt{y}.en-cs.tsv", sep="\t", keep_default_na=False) for y in (20, 21, 22)])
    en = U.read_texts("en")
    src2id, id2src = dict(zip(en["text"], en["id"])), dict(zip(en["id"], en["text"]))
    rows["src_id"] = rows["src"].map(lambda s: src2id[M.norm_text(s)])
    src_year = rows.drop_duplicates("src_id").set_index("src_id")["year"]
    h = rows[rows["is_human"].astype(str) == "True"]
    tab = pd.DataFrame([dict(src_id=i, cond="human", ref=s, text=M.norm_text(x)) for i, s, x in zip(h["src_id"], h["system"], h["hyp"])]).drop_duplicates(["src_id", "ref"])
    for c in P.CONDS:
        t = P.read_trans(c)
        tab = pd.concat([tab, pd.DataFrame({"src_id": t["id"], "cond": c, "ref": c, "text": t["text"]})])
    tab = tab[tab["text"] != ""].copy()
    tab["year"] = tab["src_id"].map(src_year)
    pairs = pd.read_csv(P.ND / "pairs.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
    pid = dict(zip(zip(pairs["src"], pairs["text"]), pairs["id"]))
    cs = U.read_texts("cs")
    new = pd.read_csv(P.ND / "texts.cs.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
    t2known, t2new = dict(zip(cs["text"], cs["id"])), dict(zip(new["text"], new["id"]))
    nllb = load(P.ND / "cond.jsonl")
    tab["pid"] = [pid[(id2src[s], x)] for s, x in zip(tab["src_id"], tab["text"])]

    def words(text, x):
        return M.units(text, x["offsets"], x["surp"], "untok")[1:]

    nllb_w = {p: words(x, nllb[p]) for p, x in zip(tab["pid"], tab["text"])}
    lev, con, gain, pmi = [], [], [], []
    for lm in args.lms:
        # plain p(y): models added later score all texts in lmcond/texts.tsv; the original three reuse the earlier files
        base = {}
        if (LD / f"surp.{lm}.plain.jsonl").exists():
            plain = load(LD / f"surp.{lm}.plain.jsonl")
            t2id = pd.read_csv(LD / "texts.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3).set_index("text")["id"].to_dict()
            for x in tab["text"].unique():
                base[x] = words(x, plain[t2id[x]])
        else:
            base_known, base_new = load(U.DATA / f"surp.cs.{lm}.jsonl"), load(P.ND / f"surp.{lm}.jsonl")
            for x in tab["text"].unique():
                base[x] = words(x, base_known[t2known[x]] if x in t2known else base_new[t2new[x]])
        per = {}
        for prompt in PROMPTS:
            d = load(LD / f"surp.{lm}.{prompt}.jsonl")
            recs, w = [], {}
            for r in tab.itertuples():
                x = d[r.pid]
                wv = words(r.text, x)
                w[(r.src_id, r.ref)] = wv
                key = dict(src_id=r.src_id, cond=r.cond, ref=r.ref, year=r.year)
                for unit, s in {"token": np.array(x["surp"], dtype=float)[1:], "untok": wv}.items():
                    m = M.measures(s)
                    recs.append(key | {"unit": unit, "n": m["n"], **{k: m[k] for k in REPORT}})
            per[prompt] = w
            df = pd.DataFrame(recs)
            df = df[df["n"] >= 3].copy()
            seg = df.groupby(["year", "src_id", "cond", "unit"])[["n"] + REPORT].mean().reset_index()
            lv = seg.groupby(["unit", "cond"])[["n"] + REPORT].mean().reset_index()
            lev.append(lv.assign(lm=lm, prompt=prompt))
            for unit, du in df.groupby("unit"):
                su = seg[seg.unit == unit]
                hsd = du[du.cond == "human"][REPORT].std()
                for scope, yrs in [("all", (2020, 2021, 2022)), (2020, (2020,)), (2021, (2021,)), (2022, (2022,))]:
                    dy, sy = du[du.year.isin(yrs)], su[su.year.isin(yrs)]
                    for b in ("human", "beam1"):
                        for c in ORDER:
                            if c == b:
                                continue
                            pv = sy[sy.cond.isin([c, b])].pivot(index="src_id", columns="cond")
                            xx = dy[dy.cond.isin([c, b])].copy()
                            xx["seg"], xx["is_c"], xx["log_n"] = xx["src_id"], (xx["cond"] == c).astype(float), np.log(xx["n"])
                            for m in REPORT:
                                est, lo, hi = A.boot_ci((pv[m][c] - pv[m][b]).values)
                                ae, al, ah, ns = A.fe_regression(xx, m, ["is_c", "log_n"] + ([] if m == "mean" else ["mean"]), B=200)
                                con.append(dict(lm=lm, prompt=prompt, unit=unit, scope=scope, cond=c, base=b, measure=m, raw=est, lo=lo, hi=hi,
                                                raw_sd=est / hsd[m], adj_sd=ae / hsd[m], adj_lo=al / hsd[m], adj_hi=ah / hsd[m], n_seg=ns))
            print(lm, prompt, flush=True)
        for r in tab.itertuples():
            k = (r.src_id, r.ref)
            b, n = base[r.text], nllb_w[r.pid]
            if len(b) < 3:
                continue
            rec = dict(lm=lm, cond=r.cond, src_id=r.src_id, base=b.mean())
            for prompt in PROMPTS:
                a = per[prompt][k]
                rec["mean_" + prompt] = a.mean()
                rec["r_nllb_" + prompt] = np.corrcoef(a, n)[0, 1] if a.std() > 0 and n.std() > 0 else np.nan
                rec["r_base_" + prompt] = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
                if prompt != "en_shuf":
                    p = b - a
                    cm = np.cov(a, p)
                    pmi.append(dict(lm=lm, prompt=prompt, cond=r.cond, var_lm=b.var(ddof=1), var_cond=cm[0, 0], var_pmi=cm[1, 1], two_cov=2 * cm[0, 1],
                                    mean_lm=b.mean(), mean_cond=a.mean(), mean_pmi=p.mean()))
            gain.append(rec)
    pd.concat(lev).to_csv(out / "levels.csv", index=False)
    pd.DataFrame(con).to_csv(out / "contrasts.csv", index=False)
    pd.DataFrame(gain).groupby(["lm", "cond"]).mean(numeric_only=True).reset_index().to_csv(out / "gain.csv", index=False)
    pd.DataFrame(pmi).groupby(["lm", "prompt", "cond"]).mean(numeric_only=True).reset_index().to_csv(out / "pmi.csv", index=False)


if __name__ == "__main__":
    main()
