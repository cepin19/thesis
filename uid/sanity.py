#!/usr/bin/env python3
r"""Sanity checks: does the pipeline measure something real? Predictions are in TODO.md §5d (written before running).

usage: sanity.py {prep_that,prep_cola,prep_ns,prep_manip,an_that,an_cola,an_ns,an_manip,an_order}
prep_* write inputs to data/sanity/ for run_surp_bow.sh; an_* write results/sanity/*.csv.
All word surprisals use the Pimentel & Meister correction (measures_ext.PM = True).

S1 that   : optional "that" in English complement clauses (Jaeger 2010). Onset surprisal is scored without "that" (context = sentence up to the
            complement, without "that"; text = the clause from its first word). Logistic regression of that-presence on standardised onset
            surprisal + verb that-rate (leave-one-out) + log verb frequency + onset length + verb-clause distance; bootstrap over sentences.
S2 cola   : CoLA acceptability ~ length + mean surprisal (+ a uniformity measure); 5-fold cross-validated log loss and AUC gain, and the coefficient.
S3 ns     : Natural Stories, sentence mean log RT ~ words + mean surprisal (+ measure); sentence scored with the preceding story as context.
S6 manip  : Czech WMT human references with words randomly replaced by corpus words, or adjacent words swapped, at 5/10/20%.
S4 order  : vn_ratio and ac1 levels from results/measures (no new scoring).
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import measures_ext as E
import udpipe_io as U

E.PM = True
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/sanity"
OUT = ROOT / "results/sanity"
RNG = np.random.default_rng(1)
EN_LMS = ["gpt2xl", "eurollm9b"]
CS_LMS = ["czgpt2xl", "eurollm9b"]
MEAS = ["cv", "gini", "sd", "lv_norm", "vn_ratio", "ac1", "max", "sl1.25", "sl1.5", "skew", "d2_abs_rel", "spec_centroid"]


def write(path, rows, cols):
    D.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=cols).to_csv(path, sep="\t", index=False, quoting=3)
    print(path, len(rows))


def clean(t):
    return " ".join(str(t).replace("\t", " ").split())


# ---------------------------------------------------------------- S1: "that"
def conllu_sents(path):
    """Yield (text, tokens) with char offsets from # text and SpaceAfter."""
    text, toks = None, []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("# text = "):
            text = line[9:]
        elif not line:
            if text is not None and toks:
                pos = 0
                ok = True
                for t in toks:
                    i = text.find(t["form"], pos)
                    if i < 0:
                        ok = False
                        break
                    t["start"], t["end"] = i, i + len(t["form"])
                    pos = t["end"]
                if ok:
                    yield text, toks
            text, toks = None, []
        elif not line.startswith("#"):
            f = line.split("\t")
            if "-" in f[0] or "." in f[0]:
                continue
            toks.append(dict(id=int(f[0]), form=f[1], lemma=f[2].lower(), upos=f[3], head=int(f[6]), deprel=f[7]))


def that_items(text, toks):
    by = {t["id"]: t for t in toks}
    kids = {}
    for t in toks:
        kids.setdefault(t["head"], []).append(t)

    def subtree(i):
        out, st = [], [i]
        while st:
            x = st.pop()
            out.append(x)
            st += [k["id"] for k in kids.get(x, [])]
        return sorted(out)

    for t in toks:
        if t["deprel"] != "ccomp" or by.get(t["head"], {}).get("upos") != "VERB":
            continue
        verb = by[t["head"]]
        sub = subtree(t["id"])
        if sub[0] < verb["id"]:
            continue  # clause before the verb
        marks = [k for k in kids.get(t["id"], []) if k["deprel"] == "mark"]
        first = by[sub[0]]
        if marks and not (marks[0]["lemma"] == "that" and marks[0]["id"] == sub[0]):
            continue  # if, whether, ...
        that = int(bool(marks))
        onset = by[sub[1]] if that and len(sub) > 1 else first
        prev = by.get((marks[0]["id"] if that else onset["id"]) - 1)
        if prev is None or prev["upos"] == "PUNCT" or onset["upos"] == "PUNCT":
            continue  # quotes and reported speech after punctuation
        cut = marks[0]["start"] if that else onset["start"]
        yield dict(verb=verb["lemma"], that=that, context=text[:cut].rstrip(), text=text[onset["start"]:], onset=onset["form"],
                   dist=sub[0] - verb["id"] - 1, onset_len=len(onset["form"]))


def prep_that():
    rows = []
    for p in sorted((U.DATA / "udpipe").glob("en.*.conllu")):
        for text, toks in conllu_sents(p):
            rows += [dict(corpus="wmt", **x) for x in that_items(text, toks)]
    bp = D / "books.en.conllu"
    if bp.exists():
        for text, toks in conllu_sents(bp):
            rows += [dict(corpus="books", **x) for x in that_items(text, toks)]
    df = pd.DataFrame(rows)
    df = df[df.context.str.len() > 0].reset_index(drop=True)
    df["id"] = [f"t{i}" for i in range(len(df))]
    df.drop(columns=["context", "text"]).to_csv(D / "that_meta.tsv", sep="\t", index=False)
    write(D / "that_in.tsv", df[["id", "context", "text"]].values.tolist(), ["id", "context", "text"])
    print(df.groupby("corpus").that.agg(["size", "mean"]))


def first_word_surprisal(rec, text):
    """Surprisal of the first whitespace word, corrected; the context's trailing space belongs to its first token (P_before = its mass)."""
    end = re.match(r"\S+", text).end()
    idx = [i for i, (a, b) in enumerate(rec["offsets"]) if a < end and text[a:b].strip()]
    if not idx:
        return np.nan
    s = sum(rec["surp"][i] for i in idx)
    lb = rec["bow_ws"]
    return s + lb[idx[0]] - lb[idx[-1] + 1]


def logit_fit(X, y, B=500):
    from sklearn.linear_model import LogisticRegression
    def fit(Xb, yb):
        return LogisticRegression(C=1e6, max_iter=2000).fit(Xb, yb).coef_[0]
    est = fit(X, y)
    bs = [fit(X[i], y[i]) for i in (RNG.integers(0, len(y), len(y)) for _ in range(B))]
    return est, np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)


def an_that():
    meta = pd.read_csv(D / "that_meta.tsv", sep="\t", keep_default_na=False)
    inp = pd.read_csv(D / "that_in.tsv", sep="\t", keep_default_na=False, quoting=3)
    txt = dict(zip(inp.id, inp.text))
    res = []
    for lm in EN_LMS:
        s = {r["id"]: first_word_surprisal(r, txt[r["id"]]) for r in map(json.loads, open(D / f"that.{lm}.jsonl")) if not r.get("ctx_cut")}
        d = meta.assign(surp=meta.id.map(s)).dropna(subset=["surp"])
        for corpus, x in [("all", d)] + list(d.groupby("corpus")):
            x = x.copy()
            g = x.groupby("verb").that
            x["verb_rate"] = (g.transform("sum") - x.that) / (g.transform("size") - 1).replace(0, np.nan)
            x["log_vfreq"] = np.log(g.transform("size"))
            x = x[g.transform("size") >= 5].dropna(subset=["verb_rate"])
            cols = ["surp", "verb_rate", "log_vfreq", "onset_len", "dist"]
            X = ((x[cols] - x[cols].mean()) / x[cols].std()).values
            est, lo, hi = logit_fit(X, x.that.values)
            for k, c in enumerate(cols):
                res.append(dict(lm=lm, corpus=corpus, n=len(x), that_rate=x.that.mean(), term=c, coef=est[k], lo=lo[k], hi=hi[k]))
            res.append(dict(lm=lm, corpus=corpus, n=len(x), term="raw: onset surprisal with that - without", coef=x[x.that == 1].surp.mean() - x[x.that == 0].surp.mean()))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res).to_csv(OUT / "that.csv", index=False)
    print(pd.DataFrame(res).query("term == 'surp'").to_string(index=False))


# ---------------------------------------------------------------- S2: CoLA
def prep_cola():
    rows = []
    for f in ("in_domain_train", "in_domain_dev", "out_of_domain_dev"):
        d = pd.read_csv(ROOT / f"cola_public/raw/{f}.tsv", sep="\t", header=None, quoting=3, names=["src", "label", "orig", "text"])
        rows += [(f"{f}#{i}", clean(t), int(l)) for i, (t, l) in enumerate(zip(d.text, d.label))]
    pd.DataFrame(rows, columns=["id", "text", "label"]).to_csv(D / "cola_meta.tsv", sep="\t", index=False, quoting=3)
    write(D / "cola_in.tsv", [r[:2] for r in rows], ["id", "text"])


def measures_for(path, texts, unit="untok", drop_cut=True):
    """drop_cut=False keeps texts whose preceding context was shortened (Natural Stories: long story contexts, not a source prompt)."""
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if (drop_cut and r.get("ctx_cut")) or r["id"] not in texts:
            continue
        m = E.extended(E.to_words(texts[r["id"]], r, unit))
        if m:
            out[r["id"]] = m
    return pd.DataFrame(out).T


def cv_gain(X0, X1, y, kind):
    """5-fold CV: (base score, full score) as log loss for binary y, R^2 for continuous y."""
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import log_loss, r2_score, roc_auc_score
    folds = RNG.permutation(len(y)) % 5
    p0, p1 = np.zeros(len(y)), np.zeros(len(y))
    for k in range(5):
        tr, te = folds != k, folds == k
        if kind == "bin":
            p0[te] = LogisticRegression(max_iter=2000).fit(X0[tr], y[tr]).predict_proba(X0[te])[:, 1]
            p1[te] = LogisticRegression(max_iter=2000).fit(X1[tr], y[tr]).predict_proba(X1[te])[:, 1]
        else:
            p0[te] = LinearRegression().fit(X0[tr], y[tr]).predict(X0[te])
            p1[te] = LinearRegression().fit(X1[tr], y[tr]).predict(X1[te])
    if kind == "bin":
        return dict(logloss_base=log_loss(y, p0), logloss_full=log_loss(y, p1), auc_base=roc_auc_score(y, p0), auc_full=roc_auc_score(y, p1))
    return dict(r2_base=r2_score(y, p0), r2_full=r2_score(y, p1))


def coef(X, y, kind, B=300):
    from sklearn.linear_model import LinearRegression, LogisticRegression
    f = (lambda a, b: LogisticRegression(C=1e6, max_iter=2000).fit(a, b).coef_[0][-1]) if kind == "bin" else (lambda a, b: LinearRegression().fit(a, b).coef_[-1])
    bs = [f(X[i], y[i]) for i in (RNG.integers(0, len(y), len(y)) for _ in range(B))]
    return f(X, y), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def predictive(df, y, kind, base, tag):
    res = []
    z = lambda c: ((c - c.mean()) / c.std()).values[:, None]
    X0 = np.hstack([z(df[b]) for b in base])
    for m in MEAS:
        if m not in df or df[m].isna().mean() > 0.2:
            continue
        ok = df[m].notna().values & np.isfinite(df[m].astype(float).values)
        X1 = np.hstack([X0, z(df[m].astype(float))])[ok]
        c, lo, hi = coef(X1, y[ok], kind)
        res.append(dict(set=tag, measure=m, n=int(ok.sum()), coef=c, lo=lo, hi=hi, **cv_gain(X0[ok], X1, y[ok], kind)))
    return res


def an_cola():
    meta = pd.read_csv(D / "cola_meta.tsv", sep="\t", quoting=3, keep_default_na=False)
    texts = dict(zip(meta.id, meta.text))
    lab = dict(zip(meta.id, meta.label))
    res = []
    for lm in EN_LMS:
        for unit in ("untok", "nopunct"):
            m = measures_for(D / f"cola.{lm}.jsonl", texts, unit).astype(float)
            m = m[m.n >= 3]
            m["log_n"] = np.log(m.n)
            y = np.array([lab[i] for i in m.index])
            res += [dict(lm=lm, unit=unit, **r) for r in predictive(m, y, "bin", ["log_n", "mean"], "cola")]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res).to_csv(OUT / "cola.csv", index=False)
    print(pd.DataFrame(res)[["lm", "unit", "measure", "coef", "lo", "hi", "auc_base", "auc_full"]].to_string(index=False))


# ---------------------------------------------------------------- S3: Natural Stories
def ns_sentences():
    w = pd.read_csv(ROOT / "naturalstories/naturalstories_RTS/processed_wordinfo.tsv", sep="\t", quoting=3).sort_values(["item", "zone"])
    sents = []
    for item, d in w.groupby("item"):
        cur = []
        for zone, word, rt in d[["zone", "word", "meanItemRT"]].values:
            cur.append((word, rt))
            if re.search(r"[.!?][\"')]*$", str(word)):
                sents.append((item, cur))
                cur = []
        if cur:
            sents.append((item, cur))
    return sents


def prep_ns():
    rows, meta = [], []
    prev = {}
    for k, (item, words) in enumerate(ns_sentences()):
        t = " ".join(str(w) for w, _ in words)
        ctx = " ".join(prev.get(item, []))
        rows.append((f"ns{k}", ctx, t))
        rts = [rt for _, rt in words][1:]  # first word dropped, as in the measures
        meta.append((f"ns{k}", item, len(words), float(np.mean(np.log(rts))) if rts else np.nan))
        prev.setdefault(item, []).append(t)
    pd.DataFrame(meta, columns=["id", "item", "n_words", "log_rt"]).to_csv(D / "ns_meta.tsv", sep="\t", index=False)
    write(D / "ns_in.tsv", rows, ["id", "context", "text"])


def an_ns():
    meta = pd.read_csv(D / "ns_meta.tsv", sep="\t").set_index("id")
    inp = pd.read_csv(D / "ns_in.tsv", sep="\t", quoting=3, keep_default_na=False)
    texts = dict(zip(inp.id, inp.text))
    res = []
    for lm in EN_LMS:
        m = measures_for(D / f"ns.{lm}.jsonl", texts, drop_cut=False).astype(float)
        m = m.join(meta, how="inner").dropna(subset=["log_rt"])
        m = m[m.n >= 4]
        m["log_n"] = np.log(m.n)
        res += [dict(lm=lm, **r) for r in predictive(m, m.log_rt.values, "cont", ["log_n", "mean"], "naturalstories")]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res).to_csv(OUT / "naturalstories.csv", index=False)
    print(pd.DataFrame(res)[["lm", "measure", "n", "coef", "lo", "hi", "r2_base", "r2_full"]].to_string(index=False))


# ---------------------------------------------------------------- S6: manipulation
def prep_manip(n=1500):
    rows = pd.concat([pd.read_csv(U.DATA / f"wmt{y}.en-cs.tsv", sep="\t", keep_default_na=False) for y in (20, 21, 22)])
    hum = sorted({clean(x) for x in rows[rows.is_human.astype(str) == "True"].hyp if len(x.split()) >= 12})
    random.seed(1)
    hum = random.sample(hum, n)
    vocab = [w for t in hum for w in t.split()]
    out = []
    for i, t in enumerate(hum):
        w = t.split()
        out.append((f"m{i}|orig|0", t))
        for rate in (0.05, 0.1, 0.2):
            k = max(1, round(rate * len(w)))
            x = list(w)
            for j in random.sample(range(1, len(w)), k):
                x[j] = random.choice(vocab)
            out.append((f"m{i}|replace|{rate}", " ".join(x)))
            y = list(w)
            for j in random.sample(range(1, len(w) - 1), min(k, len(w) - 2)):
                y[j], y[j + 1] = y[j + 1], y[j]
            out.append((f"m{i}|swap|{rate}", " ".join(y)))
    write(D / "manip_in.tsv", out, ["id", "text"])


def an_manip():
    inp = pd.read_csv(D / "manip_in.tsv", sep="\t", quoting=3, keep_default_na=False)
    texts = dict(zip(inp.id, inp.text))
    res = []
    for lm in CS_LMS:
        m = measures_for(D / f"manip.{lm}.jsonl", texts).astype(float)
        m["src"] = [i.split("|")[0] for i in m.index]
        m["kind"] = [i.split("|")[1] for i in m.index]
        m["rate"] = [float(i.split("|")[2]) for i in m.index]
        orig = m[m.kind == "orig"].set_index("src")
        sd = orig[MEAS + ["mean"]].std()
        for (kind, rate), d in m[m.kind != "orig"].groupby(["kind", "rate"]):
            d = d.set_index("src")
            for c in MEAS + ["mean"]:
                diff = (d[c] - orig[c].reindex(d.index)).dropna()
                b = diff.values[RNG.integers(0, len(diff), (1000, len(diff)))].mean(1)
                res.append(dict(lm=lm, kind=kind, rate=rate, measure=c, diff_sd=diff.mean() / sd[c], lo=np.percentile(b, 2.5) / sd[c], hi=np.percentile(b, 97.5) / sd[c]))
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(res)
    df.to_csv(OUT / "manip.csv", index=False)
    print(df.pivot_table(index=["lm", "measure"], columns=["kind", "rate"], values="diff_sd").round(2).to_string())


def an_order():
    rows = []
    for ds in ("wmt", "wmt_doc", "par3", "mendeley", "chakrabarty", "ccv"):
        for root in ("measures", "measures_pm"):
            p = ROOT / f"results/{root}/{ds}/levels.csv"
            if p.exists():
                l = pd.read_csv(p)
                rows += [dict(source=root, set=ds, **r) for r in l[["dist", "unit", "cond", "vn_ratio", "ac1"]].to_dict("records")]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "order.csv", index=False)
    print(pd.DataFrame(rows).query("cond in ['human', 'verse', 'news']").groupby(["source", "set", "unit"])[["vn_ratio", "ac1"]].agg(["min", "max"]).round(3).to_string())


if __name__ == "__main__":
    globals()[sys.argv[1]]()
