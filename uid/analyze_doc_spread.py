#!/usr/bin/env python3
r"""Do human translators vary more from document to document than MT? The part of the temporal flattening paper that holds up, applied to translation.

Same sources for both sides, so differences in the source are held constant. Spread of a feature = SD / |mean| of its value over a translator's documents.
  WMT20-22    per year, documents = WMT docs (3+ segments), MT system vs human, paired on documents.
  PAR3        per book, documents = paragraphs; Google Translate vs each human translator (2+ translators); 113 books.
  Mendeley    documents = poems; each MT system vs each human reference, EN->FR, whole corpus (h0 covers most poems; later references only multi-translation poems).
  Chakrabarty per source language, documents = 8-line stanzas.
Features (all sets): mean word length, words per document, sentence length, type-token ratio, mean and CV of word surprisal under the LMs (nopunct). WMT adds UD verb,
function and content word ratios. PAR3 and Chakrabarty (English targets) add VADER and TextBlob sentiment. sem_spread is direct: mean cosine distance of a translator's document embeddings (mBERT).
Controls: `no_length` leaves out word count, sentence length and TTR. WMT DA >= 90: only segments where both sides pass. PAR3, Mendeley, Chakrabarty: XCOMET above the set's median.
Comparison: ratio human spread / MT spread (geometric mean over MT-human cells, bootstrap over cells); share of comparisons with human higher. Human-human cells are a yardstick (*_hh sets).
Output: results/doc_spread/{spread.csv, summary.csv}. Direction fixed in TODO 5c: humans higher.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import measures as M
import udpipe_io as U

ROOT = Path(__file__).resolve().parent.parent
LIT = ROOT / "data/literary"
OUT = ROOT / "results/doc_spread"
Q = ROOT / "data/quality"
RNG = np.random.default_rng(1)
FUNC = {"ADP", "CCONJ", "SCONJ", "AUX", "DET", "PRON", "PART"}
CONTENT = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
LENGTH = {"words", "sent_len", "ttr"}
rd = lambda p: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3)
EMB = None
EIDX = None


def emb(t):
    global EMB, EIDX
    if EMB is None:
        EMB = np.load(ROOT / "data/less_diverse/emb.npy", mmap_mode="r")
        EIDX = {x: i for i, x in enumerate(json.load(open(ROOT / "data/less_diverse/emb_texts.json", encoding="utf-8")))}
    return EMB[EIDX[t]] if t in EIDX else None


def text_feats(texts):
    t = " ".join(texts)
    w = re.findall(r"[^\W\d_]+", t)
    if len(w) < 5:
        return None
    s = [x for x in re.split(r"(?<=[.!?])\s+", t) if x.strip()]
    return {"word_len": np.mean([len(x) for x in w]), "words": len(w), "sent_len": len(w) / max(len(s), 1), "ttr": len({x.lower() for x in w}) / len(w)}


def surprisal(path, id2text):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["id"] in id2text:
            s = M.units(id2text[r["id"]], r["offsets"], r["surp"], "nopunct")
            if len(s) >= 3:
                out[id2text[r["id"]]] = s[1:].astype(np.float32)
    return out


def surp_feats(texts, S, name):
    a = [S[t] for t in texts if t in S]
    if not a:
        return {}
    v = np.concatenate(a)
    return {f"surp_mean_{name}": v.mean(), f"surp_cv_{name}": v.std() / v.mean() if v.mean() > 0 else np.nan}


def spread(table):
    """{translator: {doc: features}} -> {translator: {feature: spread}} on documents shared by the pair (caller restricts)."""
    d = pd.DataFrame(table).T.astype(float)
    sd, mu = d.std(), d.mean().abs()
    return (sd / mu.where(mu > 0)).to_dict()


def sem_spread(vecs):
    v = [x for x in vecs if x is not None]  # Chakrabarty has no embeddings
    if len(v) < 3:
        return np.nan
    v = np.stack(v)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    n = len(v)
    return 1 - (v @ v.T)[np.triu_indices(n, 1)].mean()


def compare(group, a_docs, b_docs, a, b, res, tag):
    """a, b: {doc: (features, [texts])}. Spread over the documents both have; a is MT, b is human."""
    com = sorted(set(a_docs) & set(b_docs))
    if len(com) < 5:
        return
    sa, sb = spread({k: a_docs[k][0] for k in com}), spread({k: b_docs[k][0] for k in com})
    for f in sa:
        res.append(dict(set=tag[0], filter=tag[1], group=group, mt=a, human=b, n_docs=len(com), feature=f, mt_spread=sa[f], human_spread=sb.get(f, np.nan)))
    va = [emb_doc(a_docs[k][1]) for k in com]
    vb = [emb_doc(b_docs[k][1]) for k in com]
    res.append(dict(set=tag[0], filter=tag[1], group=group, mt=a, human=b, n_docs=len(com), feature="sem_spread", mt_spread=sem_spread(va), human_spread=sem_spread(vb)))


def emb_doc(texts):
    v = [emb(t) for t in texts]
    v = [x for x in v if x is not None]
    return np.mean(v, axis=0) if v else None


# ---------------------------------------------------------------- WMT
def wmt(res):
    rows = pd.concat([pd.read_csv(U.DATA / f"wmt{y}.en-cs.tsv", sep="\t", keep_default_na=False) for y in (20, 21, 22)])
    rows["text"], rows["da"] = rows["hyp"].map(M.norm_text), pd.to_numeric(rows["da_raw"], errors="coerce")
    rows["human"] = rows["is_human"].astype(str) == "True"
    cs = dict(zip(*U.read_texts("cs")[["text", "id"]].values.T))
    rows["hyp_id"] = rows["text"].map(cs)
    i2t = {v: k for k, v in cs.items()}
    ud = U.load("cs")
    sent = U.load("cs", "udpipe_sent")
    S = {n: surprisal(U.DATA / f"surp.cs.{n}.jsonl", i2t) for n in ("czgpt2xl", "eurollm9b", "customgpt2")}

    def doc_feats(x):
        texts, ids = x["text"].tolist(), x["hyp_id"].tolist()
        f = text_feats(texts)
        if f is None:
            return None
        t = [q for i in ids for q in ud.get(i, [])]
        n = max(sum(q["upos"] != "PUNCT" for q in t), 1)
        ns = sum(len({q["sent"] for q in sent.get(i, [])}) for i in ids)
        f.update({"sent_len": n / max(ns, 1), "verb_ratio": sum(q["upos"] == "VERB" for q in t) / n, "func_ratio": sum(q["upos"] in FUNC for q in t) / n, "content_ratio": sum(q["upos"] in CONTENT for q in t) / n})
        for k, s in S.items():
            f.update(surp_feats(texts, s, k))
        return f, texts

    for y, d in rows.groupby("year"):
        hs = sorted(d[d.human].system.unique())
        ms = sorted(set(d.system) - set(hs))
        for da in (False, True):
            for m in ms:
                for h in hs:
                    x, z = d[d.system == m], d[d.system == h]
                    key = ["doc_id", "seg_idx"]
                    j = x[key + ["da"]].merge(z[key + ["da"]], on=key, suffixes=("_m", "_h"))
                    if da:
                        j = j[(j.da_m >= 90) & (j.da_h >= 90)]
                    keep = j[key]
                    xa, za = x.merge(keep, on=key), z.merge(keep, on=key)
                    A, B = {}, {}
                    for doc in keep.doc_id.unique():
                        a_, b_ = xa[xa.doc_id == doc], za[za.doc_id == doc]
                        if len(a_) >= 3 and len(a_) == len(b_):
                            fa, fb = doc_feats(a_), doc_feats(b_)
                            if fa and fb:
                                A[doc], B[doc] = fa, fb
                    compare(f"{y}", A, B, m, h, res, ("wmt", "da90" if da else "all"))
            if len(hs) == 2 and not da:
                x, z = d[d.system == hs[0]], d[d.system == hs[1]]
                A = {doc: doc_feats(g) for doc, g in x.groupby("doc_id") if len(g) >= 3}
                B = {doc: doc_feats(g) for doc, g in z.groupby("doc_id") if len(g) >= 3}
                compare(f"{y}", {k: v for k, v in A.items() if v}, {k: v for k, v in B.items() if v}, hs[0], hs[1], res, ("wmt_hh", "all"))
        print("wmt", y, flush=True)


# ---------------------------------------------------------------- literary
def literary(ds, lms, res, english, cond_of):
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
    D = LIT / ds
    c, t = rd(D / "conds.tsv"), rd(D / "texts.tsv")
    id2 = dict(zip(t["id"], t["text"]))
    raw = {k: v.replace("\\n", "\n") for k, v in id2.items()}
    flat = {k: v.replace("\\n", " ") for k, v in id2.items()}
    S = {n: surprisal(D / f"surp.{n}.jsonl", raw) for n in lms}
    import lit_cond_inputs as C
    q = pd.read_csv(Q / f"{ds}.xcomet.tsv", sep="\t", dtype={"id": str}).set_index("id")["score"]
    c["qual"] = [q.get(C.cond_id(s, id2[x]), np.nan) for s, x in zip(c.src_id, c.text_id)]
    med = c["qual"].median()
    c["group"] = c.src_id.map(cond_of)
    c = c[~c.cond.isin(["multiml50", "nllb_beam1"])]
    cache = {}

    def feats(tid):
        if tid not in cache:
            f = text_feats([flat[tid]])
            if f is not None:
                for k, s in S.items():
                    f.update(surp_feats([raw[tid]], s, k))
                if english:
                    b = TextBlob(flat[tid]).sentiment
                    v = vader.polarity_scores(flat[tid])
                    f.update({"polarity": b.polarity, "subjectivity": b.subjectivity, "vader_compound": v["compound"], "vader_pos": v["pos"], "vader_neg": v["neg"], "vader_neu": v["neu"]})
            cache[tid] = (f, [flat[tid]]) if f is not None else None
        return cache[tid]

    for good in (False, True):
        x = c[c.qual >= med] if good else c
        for g, d in x.groupby("group"):
            hum = d[d.cond == "human"]
            for r in sorted(hum.ref.unique()):
                H = {s: feats(i) for s, i in hum[hum.ref == r][["src_id", "text_id"]].values}
                H = {k: v for k, v in H.items() if v}
                for m in sorted(set(d.cond) - {"human"}):
                    A = {s: feats(i) for s, i in d[d.cond == m][["src_id", "text_id"]].values}
                    compare(g, {k: v for k, v in A.items() if v}, H, m, r, res, (ds, "xcomet_q50" if good else "all"))
            refs = sorted(hum.ref.unique())
            if len(refs) >= 2:
                a = {s: feats(i) for s, i in hum[hum.ref == refs[0]][["src_id", "text_id"]].values}
                b = {s: feats(i) for s, i in hum[hum.ref == refs[1]][["src_id", "text_id"]].values}
                compare(g, {k: v for k, v in a.items() if v}, {k: v for k, v in b.items() if v}, refs[0], refs[1], res, (ds + "_hh", "xcomet_q50" if good else "all"))
    print(ds, flush=True)


def summarise(df):
    out = []
    df = df.dropna(subset=["mt_spread", "human_spread"])
    df = df[np.isfinite(df.mt_spread) & np.isfinite(df.human_spread)]
    for (st, flt), d in df.groupby(["set", "filter"]):
        for name, sub in [("all features", d), ("no length", d[~d.feature.isin(LENGTH)])] + [(f, d[d.feature == f]) for f in sorted(d.feature.unique())]:
            # one value per MT-human cell: share of features or the single feature
            g = sub.groupby(["group", "mt", "human"]).apply(lambda w: pd.Series({"win": (w.human_spread > w.mt_spread).mean(), "log_ratio": np.log((w.human_spread + 1e-9) / (w.mt_spread + 1e-9)).mean()}))
            hw = sub.assign(w=sub.human_spread > sub.mt_spread)
            k, n = int(hw.w.sum()), len(hw)
            b = g.log_ratio.values
            bs = b[RNG.integers(0, len(b), (2000, len(b)))].mean(1) if len(b) > 1 else b
            out.append(dict(set=st, filter=flt, measure=name, n_cells=len(g), n_comparisons=n, human_higher=k / n if n else np.nan, ratio=float(np.exp(b.mean())), lo=float(np.exp(np.percentile(bs, 2.5))), hi=float(np.exp(np.percentile(bs, 97.5))),
                            p=binomtest(k, n, 0.5, alternative="greater").pvalue if n else np.nan))
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = []
    wmt(res)
    literary("par3", ["eurollm9b", "gpt2xl"], res, True, lambda s: s.split("#")[0])
    literary("chakrabarty", ["eurollm9b", "gpt2xl"], res, True, lambda s: s.split("_")[0])
    literary("mendeley", ["eurollm9b", "bloom1b7", "gptfr"], res, False, lambda s: "all")
    df = pd.DataFrame(res)
    df.to_csv(OUT / "spread.csv", index=False)
    summarise(df).to_csv(OUT / "summary.csv", index=False)
    print("written", OUT)


if __name__ == "__main__":
    main()
