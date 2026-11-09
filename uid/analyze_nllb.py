#!/usr/bin/env python3
"""Does beam search make NLLB output more uniform, under NLLB's own p(y|x) and under external LMs p(y)?

Conditions: greedy (beam1), beam 2/5/10/50, ancestral sampling, and the human references, on the same source segments (WMT20-22
EN->CS). Segment level, no document context. Distributions: nllb (force-decoded p(y|x)), czgpt2xl, eurollm9b, customgpt2.
Units: token (model's own tokens; first dropped; for nllb also token_eos with the closing </s>) and untok (words; first dropped).
  levels    : mean of every measure per condition
  contrasts : condition - human and condition - beam1, raw (bootstrap over segments) and adjusted for log length + mean
              surprisal (within-segment regression); in human SDs
  pmi       : per word, s_LM = s_cond + PMI; per-segment variance decomposition and r(s_LM, s_cond) (untok)
  quality   : corpus BLEU and chrF against the human references, mean length
Writes results/nllb/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu

import analyze_wmt_human as A
import measures as M
import nllb_prepare as P
import udpipe_io as U

OUT = M.__file__ and Path(__file__).resolve().parent.parent / "results/nllb"
ND = P.ND
LMS = ["czgpt2xl", "eurollm9b", "customgpt2"]
ORDER = P.CONDS + ["human"]
REPORT = ["mean", "sd", "cv", "gini", "lv", "lv_norm"]


def load_jsonl(path, id2text):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out[id2text[r["id"]]] = r
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pd.concat([pd.read_csv(U.DATA / f"wmt{y}.en-cs.tsv", sep="\t", keep_default_na=False) for y in (20, 21, 22)])
    en = U.read_texts("en")
    src2id = dict(zip(en["text"], en["id"]))
    id2src = dict(zip(en["id"], en["text"]))
    rows["src_id"] = rows["src"].map(lambda s: src2id[M.norm_text(s)])
    h = rows[rows["is_human"].astype(str) == "True"]
    # one row per source id: years and human references (a source seen twice keeps its first year)
    src_year = rows.drop_duplicates("src_id").set_index("src_id")["year"]
    tab = [dict(src_id=i, cond="human", ref=s, text=M.norm_text(x)) for i, s, x in zip(h["src_id"], h["system"], h["hyp"])]
    tab = pd.DataFrame(tab).drop_duplicates(["src_id", "ref"])
    for c in P.CONDS:
        t = P.read_trans(c)
        tab = pd.concat([tab, pd.DataFrame({"src_id": t["id"], "cond": c, "ref": c, "text": t["text"]})])
    tab = tab[tab["text"] != ""].copy()
    tab["year"] = tab["src_id"].map(src_year)
    pairs = pd.read_csv(ND / "pairs.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
    pair_id = dict(zip(zip(pairs["src"], pairs["text"]), pairs["id"]))
    id2pair = dict(zip(pairs["id"], pairs["text"]))
    cond = {r["id"]: r for r in map(json.loads, open(ND / "cond.jsonl", encoding="utf-8"))}
    # NLLB outputs: the generated token ids (nllb_score.py --ids-dir); human references have only the tokenizer's segmentation
    gen = {(r["cond"], r["src_id"]): r for r in map(json.loads, open(ND / "cond_ids.jsonl", encoding="utf-8"))}
    lm = {}
    for name in LMS:
        d = load_jsonl(U.DATA / f"surp.cs.{name}.jsonl", dict(zip(*U.read_texts("cs")[["id", "text"]].values.T)))
        new = pd.read_csv(ND / "texts.cs.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3)
        d.update(load_jsonl(ND / f"surp.{name}.jsonl", dict(zip(new["id"], new["text"]))))
        lm[name] = d

    recs, pmi = [], []
    for r in tab.itertuples():
        key = {"src_id": r.src_id, "cond": r.cond, "ref": r.ref, "year": r.year}
        c = gen.get((r.cond, r.src_id)) or cond[pair_id[(id2src[r.src_id], r.text)]]
        sources = {"nllb": c} | {name: lm[name][r.text] for name in LMS}
        words = {}
        for dist, x in sources.items():
            tok = np.array(x["surp"], dtype=float)
            w = M.units(r.text, x["offsets"], x["surp"], "untok")
            words[dist] = w[1:]
            variants = {"token": tok[1:], "untok": w[1:]}
            if dist == "nllb":
                variants["token_eos"] = np.append(tok[1:], x["eos"])
            for unit, s in variants.items():
                m = M.measures(s)
                recs.append(key | {"dist": dist, "unit": unit, **{k: m[k] for k in ["n"] + REPORT}})
        for name in LMS:
            a, b = words[name], words["nllb"]
            if len(a) >= 3 and len(a) == len(b):
                p = a - b
                cm = np.cov(b, p)
                pmi.append(key | {"lm": name, "var_lm": a.var(ddof=1), "var_cond": cm[0, 0], "var_pmi": cm[1, 1],
                                  "two_cov": 2 * cm[0, 1], "mean_lm": a.mean(), "mean_cond": b.mean(), "mean_pmi": p.mean()})
    df = pd.DataFrame(recs)
    df = df[df["n"] >= 3].copy()
    # average the two human references per segment for the levels and raw contrasts
    keys = ["year", "src_id", "cond", "dist", "unit"]
    seg = df.groupby(keys)[["n"] + REPORT].mean().reset_index()
    seg.groupby(["dist", "unit", "cond"])[["n"] + REPORT].mean().reset_index().to_csv(OUT / "levels.csv", index=False)
    seg.groupby(["year", "dist", "unit", "cond"])[["n"] + REPORT].mean().reset_index().to_csv(OUT / "levels_by_year.csv", index=False)

    con = []
    for (dist, unit), d in df.groupby(["dist", "unit"]):
        s = seg[(seg.dist == dist) & (seg.unit == unit)]
        hsd = d[d.cond == "human"][REPORT].std()
        for scope, yrs in [("all", (2020, 2021, 2022)), (2020, (2020,)), (2021, (2021,)), (2022, (2022,))]:
            dy, sy = d[d.year.isin(yrs)], s[s.year.isin(yrs)]
            for base in ("human", "beam1"):
                for c in ORDER:
                    if c == base:
                        continue
                    p = sy[sy.cond.isin([c, base])].pivot(index="src_id", columns="cond")
                    x = dy[dy.cond.isin([c, base])].copy()
                    x["seg"] = x["src_id"]
                    x["is_c"] = (x["cond"] == c).astype(float)
                    x["log_n"] = np.log(x["n"])
                    for m in REPORT:
                        est, lo, hi = A.boot_ci((p[m][c] - p[m][base]).values)
                        xs = ["is_c", "log_n"] + ([] if m == "mean" else ["mean"])
                        ae, al, ah, ns = A.fe_regression(x, m, xs, B=500)
                        con.append(dict(dist=dist, unit=unit, scope=scope, cond=c, base=base, measure=m, raw=est, lo=lo, hi=hi,
                                        raw_sd=est / hsd[m], adj_sd=ae / hsd[m], adj_lo=al / hsd[m], adj_hi=ah / hsd[m], n_seg=ns))
        print(dist, unit, flush=True)
    pd.DataFrame(con).to_csv(OUT / "contrasts.csv", index=False)

    pm = pd.DataFrame(pmi)
    pm.groupby(["lm", "cond"])[["mean_lm", "mean_cond", "mean_pmi", "var_lm", "var_cond", "var_pmi", "two_cov"]].mean().reset_index().to_csv(
        OUT / "pmi.csv", index=False)

    q = []
    for y in (2020, 2021, 2022):
        ty = tab[tab.year == y]
        hy = ty[ty.cond == "human"]
        ids = sorted(ty[ty.cond == "beam1"]["src_id"])
        refs = [hy[hy.ref == r].set_index("src_id")["text"].reindex(ids).fillna("").tolist() for r in hy["ref"].unique()]
        for c in P.CONDS:
            hyp = ty[ty.cond == c].set_index("src_id")["text"].reindex(ids).fillna("").tolist()
            q.append(dict(year=y, cond=c, bleu=sacrebleu.corpus_bleu(hyp, refs).score, chrf=sacrebleu.corpus_chrf(hyp, refs).score,
                          words=np.mean([len(t.split()) for t in hyp]), human_words=np.mean([len(t.split()) for t in refs[0]])))
    pd.DataFrame(q).to_csv(OUT / "quality.csv", index=False)


if __name__ == "__main__":
    main()
