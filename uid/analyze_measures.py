#!/usr/bin/env python3
r"""All (non-)uniformity measures of MEASURES.md on one dataset. usage: analyze_measures.py <dataset> [--quick N] [--quality] [--only m1,m2] [--out name] [--pm]

Datasets: wmt (WMT20-22 EN->CS, segment level; human vs all MT pooled; also by year), wmt_hh (translator vs translator, WMT21/22: sanity check S5), wmt_doc (same, concatenated per document, scored with
document context; no null samples, no conditional run), nllb (decoding sweep; bases human and beam1), mendeley, mendeley_flat, chakrabarty,
par3, ccv (original verse vs human news: first 20 scored words, unpaired; full poems in levels only).
--pm: word surprisal with the Pimentel & Meister correction (bow.py, measures_ext.pm_correction), from the --bow rescoring (submit_pm.sh);
results/measures_pm/. Calibrated z-scores are absent there (no null samples), and typicality still uses uncorrected token entropy.
Distributions: <lm> = p(y); <lm>_cond = external LM with the source prompt (lit_cond_inputs.py; texts whose prompt was cut are excluded);
nllb = force-decoded p(y|x).
Output in results/measures/<dataset>/:
  levels.csv             mean of every measure by distribution, unit, condition
  contrasts.csv          condition - base: raw (paired bootstrap over sources) and adjusted for log length + mean surprisal within source
                         (not for `mean`); in human SDs. Raw only, unpaired, for ccv
  reliability_profile.csv  per text, correlation of word profiles between two distributions
  reliability_measure.csv  across texts, correlation of each measure's values between two distributions
--quality: results/measures_quality/<dataset>/<filter>/ on translations passing a filter applied to human and MT alike; sources without a surviving
human translation are dropped. Filters: human DA (WMT only: da90, da95; docda85, docda90 on document means), reference-free CometKiwi and XCOMET-XL at the
dataset's pooled median and 75th percentile, xcomet_noerr (above median, no major or critical error span). Effects stay in unfiltered human SDs.
Also pass_rates.csv, and for wmt validation.csv (scores vs DA, and within-source correlation of scores and DA with mean, CV, lv_norm).
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import analyze_wmt_human as A
import lit_cond_inputs as C
import measures as M
import measures_ext as E
import nllb_prepare as P
import udpipe_io as U

ROOT = Path(__file__).resolve().parent.parent
LIT = ROOT / "data/literary"
RNG = np.random.default_rng(1)
ONLY = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None  # contrast only these measures
OUT_ROOT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None  # results/<this>/ instead of the defaults
PM = "--pm" in sys.argv  # Pimentel & Meister correction: read the --bow rescoring (surpb*, cond_bow, cond_ids_bow); null samples are absent
if PM:
    E.PM = True
    OUT_ROOT = OUT_ROOT or "measures_pm"


def src(path):
    """Scored file for the current mode: the --bow rescoring under --pm."""
    if not PM:
        return path
    p = str(path).replace("/surpn_cond.", "/surpb_cond.").replace("/surpn.", "/surpb.").replace("/surpctx.", "/surpb_ctx.")
    return Path(p.replace("cond_ids_null.jsonl", "cond_ids_bow.jsonl").replace("cond_null.jsonl", "cond_bow.jsonl"))
rd = lambda p: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3)
unesc = lambda t: t.replace("\\n", "\n")
LMS = {"mendeley": ["eurollm9b", "bloom1b7", "gptfr"], "mendeley_flat": ["eurollm9b", "bloom1b7", "gptfr"], "chakrabarty": ["eurollm9b", "gpt2xl"],
       "par3": ["eurollm9b", "gpt2xl"], "wmt": ["czgpt2xl", "eurollm9b", "customgpt2"], "nllb": ["czgpt2xl", "eurollm9b", "customgpt2"],
       "wmt_doc": ["czgpt2xl", "eurollm9b", "customgpt2"], "ccv": ["czgpt2xl", "eurollm9b"]}


def stream(files, id2text, units, limit=None, keyfn=None):
    """{key: {unit: (measures, s)}} for every JSONL row; keyfn maps a row to its key (default: the id)."""
    out = {}
    for f in files:
        for i, line in enumerate(open(src(f), encoding="utf-8")):
            if limit and i >= limit:
                break
            r = json.loads(line)
            if r.get("ctx_cut") or r["id"] not in id2text:
                continue
            t = id2text[r["id"]]
            e = {}
            for u in units:
                w = E.to_words(t, r, u)
                m = E.extended(w)
                if m:
                    e[u] = (m, w["s"].astype(np.float32))
            out[keyfn(r) if keyfn else r["id"]] = e
    return out


def wmt_rows():
    rows = pd.concat([pd.read_csv(U.DATA / f"wmt{y}.en-cs.tsv", sep="\t", keep_default_na=False) for y in (20, 21, 22)])
    en = U.read_texts("en")
    s2i = dict(zip(en["text"], en["id"]))
    rows["src_id"] = rows["src"].map(lambda s: s2i[M.norm_text(s)])
    rows["text"] = rows["hyp"].map(M.norm_text)
    rows["human"] = rows["is_human"].astype(str) == "True"
    return rows


def build(ds, units, limit):
    """rows: DataFrame src_id, cond, ref, scope; plus one key column per distribution. dists: {dist: {key: {unit: (measures, s)}}}."""
    dists = {}
    if ds == "wmt":
        rows = wmt_rows()
        df = pd.DataFrame({"src_id": rows.src_id, "cond": np.where(rows.human, "human", "mt"), "ref": rows.system, "scope": rows.year.astype(str), "text": rows.text})
        df["da"] = pd.to_numeric(rows.da_raw, errors="coerce").values
        i2t = dict(zip(*U.read_texts("cs")[["id", "text"]].values.T))
        ci = rd(U.DATA / "cond_in.tsv")
        ci2t = dict(zip(ci["id"], ci["text"].map(unesc)))
        for lm in LMS[ds]:
            dists[lm] = stream([U.DATA / f"surpn.cs.{lm}.jsonl"], i2t, units, limit, keyfn=lambda r: i2t[r["id"]])
            dists[f"{lm}_cond"] = stream([U.DATA / f"surpn_cond.{lm}.jsonl"], ci2t, units, limit)
            df[f"key_{lm}"] = df["text"]
            df[f"key_{lm}_cond"] = [C.cond_id(a, b) for a, b in zip(df.src_id, df.text)]
        return df, dists, ["human"]
    if ds == "wmt_hh":  # sanity check S5: translator vs translator, WMT21/22, same contrasts as MT vs human
        rows = wmt_rows()
        rows = rows[rows.human & rows.year.isin([2021, 2022])]
        first = rows.groupby("year").system.transform("min")
        df = pd.DataFrame({"src_id": rows.src_id, "cond": np.where(rows.system == first, "human", "human2"), "ref": rows.system, "scope": rows.year.astype(str), "text": rows.text})
        i2t = dict(zip(*U.read_texts("cs")[["id", "text"]].values.T))
        ci = rd(U.DATA / "cond_in.tsv")
        ci2t = dict(zip(ci["id"], ci["text"].map(unesc)))
        for lm in LMS["wmt"]:
            dists[lm] = stream([U.DATA / f"surpn.cs.{lm}.jsonl"], i2t, units, limit, keyfn=lambda r: i2t[r["id"]])
            dists[f"{lm}_cond"] = stream([U.DATA / f"surpn_cond.{lm}.jsonl"], ci2t, units, limit)
            df[f"key_{lm}"] = df["text"]
            df[f"key_{lm}_cond"] = [C.cond_id(a, b) for a, b in zip(df.src_id, df.text)]
        return df, dists, ["human"]
    if ds == "wmt_doc":
        rows = wmt_rows()
        t = rd(U.DATA / "texts_ctx.cs.tsv")
        i2t = dict(zip(t["id"], t["text"]))
        for lm in LMS[ds]:
            segs = {}
            for i, line in enumerate(open(src(U.DATA / f"surpctx.cs.{lm}.jsonl"), encoding="utf-8")):
                if limit and i >= limit:
                    break
                r = json.loads(line)
                y, sysn, doc, seg = r["id"].split("|")
                for u in units:
                    segs.setdefault((y, sysn, doc, u), []).append((int(seg), E.to_words(i2t[r["id"]], r, u)))
            res = {}
            for (y, sysn, doc, u), parts in segs.items():
                w = E.concat_words([p for _, p in sorted(parts, key=lambda x: x[0])])
                m = E.extended(w)
                if m:
                    res.setdefault(f"{y}|{sysn}|{doc}", {})[u] = (m, w["s"].astype(np.float32))
            dists[lm] = res
        keys = rows[["year", "system", "doc_id", "human"]].drop_duplicates()
        df = pd.DataFrame({"src_id": keys.year.astype(str) + "|" + keys.doc_id, "cond": np.where(keys.human, "human", "mt"), "ref": keys.system, "scope": keys.year.astype(str)})
        rows["da"] = pd.to_numeric(rows.da_raw, errors="coerce")
        rows["qid"] = [C.cond_id(a, b) for a, b in zip(rows.src_id, rows.text)]
        doc_da = rows.groupby(["year", "system", "doc_id"])["da"].mean()
        df["da"] = [doc_da.get((y, x, d)) for y, x, d in zip(keys.year, keys.system, keys.doc_id)]
        df.attrs["doc_rows"] = rows[["year", "system", "doc_id", "qid"]]
        for lm in LMS[ds]:
            df[f"key_{lm}"] = keys.year.astype(str).values + "|" + keys.system.values + "|" + keys.doc_id.values
        return df, dists, ["human"]
    if ds == "nllb":
        rows = wmt_rows()
        en = U.read_texts("en")
        id2src = dict(zip(en["id"], en["text"]))
        h = rows[rows.human]
        year = rows.drop_duplicates("src_id").set_index("src_id")["year"].astype(str)
        tab = [pd.DataFrame({"src_id": h.src_id, "cond": "human", "ref": h.system, "text": h.text}).drop_duplicates(["src_id", "ref"])]
        for c in P.CONDS:
            t = P.read_trans(c)
            tab.append(pd.DataFrame({"src_id": t["id"], "cond": c, "ref": c, "text": t["text"]}))
        df = pd.concat(tab)
        df = df[df.text != ""].copy()
        df["scope"] = df.src_id.map(year)
        i2t = dict(zip(*U.read_texts("cs")[["id", "text"]].values.T))
        new = rd(P.ND / "texts.cs.tsv")
        i2n = dict(zip(new["id"], new["text"]))
        ci = pd.concat([rd(U.DATA / "cond_in.tsv"), rd(P.ND / "cond_in.tsv")])
        ci2t = dict(zip(ci["id"], ci["text"].map(unesc)))
        for lm in LMS[ds]:
            dists[lm] = stream([U.DATA / f"surpn.cs.{lm}.jsonl"], i2t, units, limit, keyfn=lambda r: i2t[r["id"]]) | stream([P.ND / f"surpn.{lm}.jsonl"], i2n, units, limit, keyfn=lambda r: i2n[r["id"]])
            dists[f"{lm}_cond"] = stream([U.DATA / f"surpn_cond.{lm}.jsonl", P.ND / f"surpn_cond.{lm}.jsonl"], ci2t, units, limit)
            df[f"key_{lm}"] = df["text"]
            df[f"key_{lm}_cond"] = [C.cond_id(a, b) for a, b in zip(df.src_id, df.text)]
        pairs = rd(P.ND / "pairs.tsv")
        p2t = dict(zip(pairs["id"], pairs["text"]))
        pid = dict(zip(zip(pairs["src"], pairs["text"]), pairs["id"]))
        gen = rd(P.ND / "pairs_ids.tsv")
        g2t = dict(zip(gen["id"], gen["text"]))
        dists["nllb"] = stream([P.ND / "cond_null.jsonl"], p2t, units, limit) | stream([P.ND / "cond_ids_null.jsonl"], g2t, units, limit)
        df["key_nllb"] = [f"{c}:{s}" if f"{c}:{s}" in g2t else pid.get((id2src[s], t)) for c, s, t in zip(df.cond, df.src_id, df.text)]
        return df, dists, ["human", "beam1"]
    if ds == "ccv":
        t = rd(LIT / "ccv/texts_tgt.tsv")
        i2t = dict(zip(t["id"], t["text"].map(unesc)))
        wr = wmt_rows()
        hum = wr[wr.human]
        c2t = dict(zip(*U.read_texts("cs")[["id", "text"]].values.T))
        hset = set(hum.text)
        for lm in LMS[ds]:
            res = {}
            for f, m_, tag in [(LIT / f"ccv/surpn.{lm}.jsonl", i2t, "verse"), (U.DATA / f"surpn.cs.{lm}.jsonl", c2t, "news")]:
                for i, line in enumerate(open(src(f), encoding="utf-8")):
                    if limit and i >= limit:
                        break
                    r = json.loads(line)
                    x = m_[r["id"]]
                    if tag == "news" and x not in hset:
                        continue
                    for u in units:
                        w = E.to_words(x, r, u)
                        if w is None:
                            continue
                        if tag == "verse":
                            m = E.extended(w)
                            if m:
                                res.setdefault(f"poem|{r['id']}", {})[u] = (m, w["s"].astype(np.float32))
                            if len(w["s"]) >= 21:
                                m = E.extended(E.cut_words(w, 21))
                                res.setdefault(f"open|{r['id']}", {})[u] = (m, w["s"][:21].astype(np.float32))
                        elif 16 <= len(w["s"]) <= 26:
                            m = E.extended(w)
                            if m:
                                res.setdefault(f"open|{r['id']}", {})[u] = (m, w["s"].astype(np.float32))
            dists[lm] = res
        keys = sorted({k for d in dists.values() for k in d})
        cond = ["poem" if k.startswith("poem") else ("verse" if k.split("|")[1] in i2t else "news") for k in keys]
        df = pd.DataFrame({"src_id": keys, "cond": cond, "ref": keys, "scope": "all"})
        for lm in LMS[ds]:
            df[f"key_{lm}"] = keys
        return df, dists, []
    # literary
    D = LIT / ds
    conds, texts = rd(D / "conds.tsv"), rd(D / "texts.tsv")
    id2 = dict(zip(texts["id"], texts["text"]))
    df = pd.DataFrame({"src_id": conds.src_id, "cond": conds.cond, "ref": conds.ref, "scope": "all", "tid": conds.text_id})
    i2t = {k: unesc(v) for k, v in id2.items()}
    ci = rd(D / "cond_in.tsv")
    ci2t = dict(zip(ci["id"], ci["text"].map(unesc)))
    for lm in LMS[ds]:
        dists[lm] = stream([D / f"surpn.{lm}.jsonl"], i2t, units, limit)
        dists[f"{lm}_cond"] = stream([D / f"surpn_cond.{lm}.jsonl"], ci2t, units, limit)
        df[f"key_{lm}"] = df["tid"]
        df[f"key_{lm}_cond"] = [C.cond_id(s, id2[t]) for s, t in zip(df.src_id, df.tid)]
    return df, dists, ["human"]


def tables(df, dists, units, ds):
    """One long table: a row per (row in df, dist, unit), with state measures added."""
    long_ = {}
    for dist, d in dists.items():
        for u in units:
            vals = [e[u][1][1:] for e in d.values() if u in e]
            if not vals:
                continue
            centres = E.fit_states(np.concatenate(vals))
            mu_c = np.concatenate(vals).mean()  # gv: variance around the corpus mean of this distribution and unit
            cache = {k: e[u][0] | E.state_measures(e[u][1].astype(float), centres) | {"gv": float(np.mean((e[u][1][1:] - mu_c) ** 2))}
                     for k, e in d.items() if u in e}
            base = df[["src_id", "cond", "ref", "scope"]].copy()
            base["key"] = df[f"key_{dist}"].values
            rec = pd.DataFrame([cache.get(k, {}) for k in base["key"]], index=base.index)
            long_[(dist, u)] = pd.concat([base, rec], axis=1).dropna(subset=["n"])
    return long_


def contrasts(long_, bases, ds, B=300, hsd_from=None, skip_conds=(), by_scope=True):
    """hsd_from: long tables whose human SD is the unit (the unfiltered data, so filtered effects stay comparable)."""
    out = []
    for (dist, u), d in long_.items():
        ms = [c for c in d.columns if c not in ("src_id", "cond", "ref", "scope", "key", "n")]
        ref = (hsd_from or long_)[(dist, u)]
        hsd = ref[ref.cond == (bases[0] if bases else "news")][ms].std()
        scopes = [("all", None)] + ([(s, s) for s in sorted(set(d.scope))] if by_scope and d.scope.nunique() > 1 else [])
        for scope, sv in scopes:
            ds_ = d if sv is None else d[d.scope == sv]
            seg = ds_.groupby(["src_id", "cond"])[["n"] + ms].mean().reset_index()
            for base in bases:
                for c in sorted(set(ds_.cond) - {base} - set(skip_conds)):
                    p = seg[seg.cond.isin([c, base])].pivot(index="src_id", columns="cond")
                    x = ds_[ds_.cond.isin([c, base])].copy()
                    x["seg"], x["is_c"], x["log_n"] = x["src_id"], (x["cond"] == c).astype(float), np.log(x["n"])
                    for m in [k for k in ms if not ONLY or k in ONLY]:
                        if (m, c) not in p.columns or (m, base) not in p.columns:
                            continue
                        dd = (p[m][c] - p[m][base]).dropna().values
                        if len(dd) < 10:
                            continue
                        est, lo, hi = A.boot_ci(dd)
                        xs = ["is_c", "log_n"] + ([] if m == "mean" else ["mean"])
                        ae, al, ah, ns = A.fe_regression(x, m, xs, B=B)
                        sd = hsd[m] if hsd[m] > 0 else np.nan
                        out.append(dict(dist=dist, unit=u, scope=scope, cond=c, base=base, measure=m, n_src=len(dd), raw=est, lo=lo, hi=hi, raw_sd=est / sd,
                                        adj_sd=ae / sd, adj_lo=al / sd, adj_hi=ah / sd))
        if not bases:  # ccv: unpaired verse - news
            a, b = d[d.cond == "verse"], d[d.cond == "news"]
            for m in ms:
                x, y = a[m].dropna().values, b[m].dropna().values
                if len(x) < 10 or len(y) < 10:
                    continue
                bs = x[RNG.integers(0, len(x), (2000, len(x)))].mean(1) - y[RNG.integers(0, len(y), (2000, len(y)))].mean(1)
                sd = y.std() if y.std() > 0 else np.nan
                out.append(dict(dist=dist, unit=u, scope="all", cond="verse", base="news", measure=m, n_src=len(x), raw=x.mean() - y.mean(),
                                lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), raw_sd=(x.mean() - y.mean()) / sd))
        print(ds, dist, u, flush=True)
    return pd.DataFrame(out)


def reliability(long_, dists, df, units):
    prof, meas = [], []
    names = list(dists)
    for u in units:
        for a, b in itertools.combinations(names, 2):
            if (a, u) not in long_ or (b, u) not in long_:
                continue
            ka, kb = df[f"key_{a}"].values, df[f"key_{b}"].values
            idx = RNG.permutation(len(df))[:4000]
            pe, sp, conds = [], [], []
            for i in idx:
                ea, eb = dists[a].get(ka[i], {}).get(u), dists[b].get(kb[i], {}).get(u)
                if ea is None or eb is None or len(ea[1]) != len(eb[1]) or ea[1][1:].std() == 0 or eb[1][1:].std() == 0:
                    continue
                pe.append(np.corrcoef(ea[1][1:], eb[1][1:])[0, 1])
                sp.append(spearmanr(ea[1][1:], eb[1][1:]).statistic)
                conds.append(df["cond"].values[i])
            if pe:
                g = pd.DataFrame({"p": pe, "s": sp, "c": conds})
                prof.append(dict(a=a, b=b, unit=u, cond="all", n=len(g), pearson=g.p.mean(), spearman=g.s.mean()))
                for c, x in g.groupby("c"):
                    prof.append(dict(a=a, b=b, unit=u, cond=c, n=len(x), pearson=x.p.mean(), spearman=x.s.mean()))
            ks = ["src_id", "cond", "ref"]
            la, lb = long_[(a, u)], long_[(b, u)]
            mg = la.merge(lb, on=ks, suffixes=("_a", "_b"))
            for m in [c for c in la.columns if c not in ks + ["scope", "key", "n"]]:
                if m + "_a" not in mg or m + "_b" not in mg:
                    continue
                w = mg[[m + "_a", m + "_b"]].dropna()
                if len(w) >= 30 and w.iloc[:, 0].std() > 0 and w.iloc[:, 1].std() > 0:
                    meas.append(dict(a=a, b=b, unit=u, measure=m, n=len(w), pearson=np.corrcoef(w.iloc[:, 0], w.iloc[:, 1])[0, 1],
                                     spearman=spearmanr(w.iloc[:, 0], w.iloc[:, 1]).statistic))
    return pd.DataFrame(prof), pd.DataFrame(meas)


QUAL_FILES = {"wmt": ["wmt"], "wmt_doc": ["wmt"], "nllb": ["wmt", "nllb"], "mendeley": ["mendeley"], "mendeley_flat": ["mendeley_flat"],
              "chakrabarty": ["chakrabarty"], "par3": ["par3"]}


def load_quality(ds, metric):
    q = pd.concat([pd.read_csv(ROOT / f"data/quality/{n}.{metric}.tsv", sep="\t", dtype={"id": str}) for n in QUAL_FILES[ds]])
    return q.drop_duplicates("id").set_index("id")


def quality_masks(ds, df):
    """{filter name: boolean array over df}, plus df with the score columns. Filters apply to human and MT alike."""
    masks = {}
    if ds == "wmt":
        masks |= {"da90": (df.da >= 90).values, "da95": (df.da >= 95).values}
    if ds == "wmt_doc":
        masks |= {"docda85": (df.da >= 85).values, "docda90": (df.da >= 90).values}
    cond_cols = [c for c in df.columns if c.startswith("key_") and c.endswith("_cond")]
    for metric in ("kiwi", "xcomet"):
        q = load_quality(ds, metric)
        if ds == "wmt_doc":
            r = df.attrs["doc_rows"].copy()
            r["score"] = r["qid"].map(q["score"])
            doc = r.groupby(["year", "system", "doc_id"])["score"].mean()
            df[metric] = [doc.get((int(s.split("|")[0]), x, s.split("|", 1)[1])) for s, x in zip(df.src_id, df.ref)]
        else:
            df[metric] = df[cond_cols[0]].map(q["score"]).values
            if metric == "xcomet":
                df["n_major"] = df[cond_cols[0]].map(q["n_major"]).values
        for pct in (50, 75):
            masks[f"{metric}_q{pct}"] = (df[metric] >= df[metric].quantile(pct / 100)).values
    if "n_major" in df:
        masks["xcomet_noerr"] = ((df["n_major"] == 0) & (df["xcomet"] >= df["xcomet"].quantile(0.5))).values
    return masks


def run_quality(ds, df, dists, long_, bases, units):
    OUT = ROOT / "results" / (OUT_ROOT + "_quality" if OUT_ROOT else "measures_quality") / ds
    OUT.mkdir(parents=True, exist_ok=True)
    masks = quality_masks(ds, df)
    skip = ["sample", "beam2", "beam10"] if ds == "nllb" else ["multiml50"] if ds == "chakrabarty" else []
    rates, val = [], []
    if ds == "wmt":
        for metric in ("kiwi", "xcomet"):
            for c, x in [("all", df), ("mt", df[df.cond == "mt"]), ("human", df[df.cond == "human"])]:
                w = x[["da", metric, "src_id"]].dropna()
                dm = w[["da", metric]] - w.groupby("src_id")[["da", metric]].transform("mean")
                val.append(dict(metric=metric, group=c, n=len(w), r_da=w.da.corr(w[metric]), r_da_within_source=dm.da.corr(dm[metric])))
            for dist in LMS[ds]:
                d = long_[(dist, "nopunct")]
                w = pd.concat([d[["src_id", "mean", "cv", "lv_norm"]], df.loc[d.index, [metric, "da"]]], axis=1).dropna()
                cols = ["mean", "cv", "lv_norm", metric, "da"]
                dm = w[cols] - w.groupby("src_id")[cols].transform("mean")
                val.append(dict(metric=metric, group="bias_" + dist, n=len(w), r_mean=dm["mean"].corr(dm[metric]), r_cv=dm["cv"].corr(dm[metric]), r_lv_norm=dm["lv_norm"].corr(dm[metric]),
                                da_r_mean=dm["mean"].corr(dm["da"]), da_r_cv=dm["cv"].corr(dm["da"]), da_r_lv_norm=dm["lv_norm"].corr(dm["da"])))
        pd.DataFrame(val).to_csv(OUT / "validation.csv", index=False)
    for name, mask in masks.items():
        keep = df[mask]
        ok = set(keep[keep.cond == "human"].src_id)
        keep = keep[keep.src_id.isin(ok)]
        idx = set(keep.index)
        for c, x in df.groupby("cond"):
            rates.append(dict(filter=name, cond=c, n_all=len(x), pass_rate=mask[x.index].mean(), n_kept_paired=int((keep.cond == c).sum()), n_src=keep[keep.cond == c].src_id.nunique()))
        sub = {k: d[d.index.isin(idx)] for k, d in long_.items()}
        F = OUT / name
        F.mkdir(exist_ok=True)
        lev = []
        for (dist, u), d in sub.items():
            ms = [c for c in d.columns if c not in ("src_id", "cond", "ref", "scope", "key")]
            for c, x in d.groupby("cond"):
                lev.append(dict(dist=dist, unit=u, cond=c, n_texts=len(x), n_src=x.src_id.nunique(), **x[ms].mean()))
        pd.DataFrame(lev).to_csv(F / "levels.csv", index=False)
        contrasts(sub, ["human"], f"{ds} {name}", B=150, hsd_from=long_, skip_conds=skip, by_scope=ds in ("wmt", "wmt_doc")).to_csv(F / "contrasts.csv", index=False)
    pd.DataFrame(rates).to_csv(OUT / "pass_rates.csv", index=False)
    print("written", OUT)


def main():
    ds = sys.argv[1]
    limit = int(sys.argv[sys.argv.index("--quick") + 1]) if "--quick" in sys.argv else None
    units = ["untok"] if ds == "nllb" else ["untok", "nopunct"]
    OUT = ROOT / "results" / (OUT_ROOT or "measures") / (ds + ("_quick" if limit else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    df, dists, bases = build(ds, units, limit)
    df = df.reset_index(drop=True)
    long_ = tables(df, dists, units, ds)
    if "--quality" in sys.argv:
        run_quality(ds, df, dists, long_, bases, units)
        return
    lev = []
    for (dist, u), d in long_.items():
        ms = [c for c in d.columns if c not in ("src_id", "cond", "ref", "scope", "key")]
        for c, x in d.groupby("cond"):
            lev.append(dict(dist=dist, unit=u, cond=c, n_texts=len(x), n_src=x.src_id.nunique(), **x[ms].mean()))
    pd.DataFrame(lev).to_csv(OUT / "levels.csv", index=False)
    contrasts(long_, bases, ds).to_csv(OUT / "contrasts.csv", index=False)
    if ONLY:
        print("written", OUT)
        return
    pr, me = reliability(long_, dists, df, units)
    pr.to_csv(OUT / "reliability_profile.csv", index=False)
    me.to_csv(OUT / "reliability_measure.csv", index=False)
    print("written", OUT)


if __name__ == "__main__":
    main()
