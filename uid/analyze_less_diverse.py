#!/usr/bin/env python3
r"""Is MT less diverse than human translation? Measures 1, 2, 4, 5 of TODO 5c (3 is analyze_measures.py --only). Directions were fixed in TODO.md first.

usage: analyze_less_diverse.py {embed,pairs,htra,shifts,ident}   Output: results/less_diverse/<step>*.csv

embed  : mean-pooled mBERT embedding of every WMT, PAR3 and Mendeley translation (data/less_diverse/emb.*; GPU)
pairs  : (1) per source, pairwise similarity of its translations: chrF, embedding cosine, r of surprisal profiles resampled to 20 points.
         WMT, per year: MT-MT vs human-human vs human-MT. PAR3 and Mendeley: human-human vs human-MT.
htra   : (2) entropy of target lemmas per source content lemma and system, on exactly k=5 aligned occurrences (repeated rarefaction, so system size and
         alignment coverage do not change the count). Each MT-human comparison uses the lemmas available to both.
shifts : (5) per aligned word pair: word class change, deprel change, head correspondence lost, voice change (verbs); per segment: clause-count change.
ident  : (4) cross-validated accuracy of telling two translators/systems apart on 10-segment chunks of the same documents (folds grouped by document).
         delex: POS trigrams + function-word lemmas + sentence length; char: character 2-4-grams. Needs scikit-learn.
WMT comparisons run on all data and at DA >= 90 (both translations of a comparison pass). CIs: bootstrap over documents (pairs, shifts) or lemmas (htra).
Caveats: alignment works better on literal text; 1-2 human translations per year; MT systems share data and architectures.
"""
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import measures as M
import udpipe_io as U

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/less_diverse"
D = ROOT / "data/less_diverse"
LIT = ROOT / "data/literary"
RNG = np.random.default_rng(1)
CONTENT = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
CLASS = {"NOUN": "N", "PROPN": "N", "VERB": "V", "AUX": "V", "ADJ": "A", "ADV": "R"}
CLAUSE = {"ccomp", "xcomp", "advcl", "acl", "csubj", "parataxis"}
FUNC = {"ADP", "CCONJ", "SCONJ", "AUX", "DET", "PRON", "PART"}
rd = lambda p: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def load_rows():
    df = pd.concat([pd.read_csv(U.DATA / f"wmt{y}.en-cs.tsv", sep="\t", keep_default_na=False) for y in (20, 21, 22)])
    df["da"] = pd.to_numeric(df["da_raw"], errors="coerce")
    en = dict(zip(*U.read_texts("en")[["text", "id"]].values.T))
    cs = dict(zip(*U.read_texts("cs")[["text", "id"]].values.T))
    df["src_id"] = df["src"].map(lambda s: en[M.norm_text(s)])
    df["hyp_id"] = df["hyp"].map(lambda s: cs[M.norm_text(s)])
    df["text"] = df["hyp"].map(M.norm_text)
    df["human"] = df["is_human"].astype(str) == "True"
    return df.sort_values(["year", "doc_id", "seg_idx", "system"]).reset_index(drop=True)


def boot_docs(by_doc, B=1000):
    """by_doc: {doc: (sum of differences, count)}; returns the ratio and its bootstrap CI over documents."""
    docs = list(by_doc)
    v = np.array([by_doc[d] for d in docs], dtype=float)  # docs x 2: sum of differences, count
    if len(docs) < 5 or v[:, 1].sum() == 0:
        return np.nan, np.nan, np.nan
    idx = RNG.integers(0, len(docs), (B, len(docs)))
    bs = v[idx, 0].sum(1) / np.maximum(v[idx, 1].sum(1), 1)
    return v[:, 0].sum() / v[:, 1].sum(), *np.percentile(bs, [2.5, 97.5])


# ---------------------------------------------------------------- embeddings
def literary_items():
    out = []
    for ds in ("par3", "mendeley"):
        c, t = rd(LIT / ds / "conds.tsv"), rd(LIT / ds / "texts.tsv")
        txt = dict(zip(t["id"], t["text"]))
        keep = c if ds == "par3" else c[c["cond"].isin(["human", "nllb_beam5", "opus_beam5"])]
        for s, cond, ref, tid in keep[["src_id", "cond", "ref", "text_id"]].values:
            out.append((ds, s, ref, cond == "human", tid, txt[tid].replace("\\n", " ")))
    return out


def embed():
    import torch
    from transformers import AutoModel, AutoTokenizer
    D.mkdir(parents=True, exist_ok=True)
    name = "google-bert/bert-base-multilingual-cased"
    tok, model = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).cuda().half().eval()
    texts = sorted(set(load_rows()["text"]) | {x[5] for x in literary_items()})
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    X = np.zeros((len(texts), 768), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, len(order), 128):
            i = order[b:b + 128]
            e = tok([texts[j] for j in i], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            h = model(**e).last_hidden_state.float()
            m = e["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1)
            X[i] = torch.nn.functional.normalize(v, dim=-1).cpu().numpy()
    np.save(D / "emb.npy", X)
    json.dump(texts, open(D / "emb_texts.json", "w", encoding="utf-8"), ensure_ascii=False)


# ---------------------------------------------------------------- 1: pairs
def resample(s, n=20):
    return np.interp(np.linspace(0, len(s) - 1, n), np.arange(len(s)), s)


def lit_profiles(ds):
    t = rd(LIT / ds / "texts.tsv")
    out = {}
    raw = dict(zip(t["id"], t["text"].map(lambda x: x.replace("\\n", "\n"))))
    for line in open(LIT / ds / "surp.eurollm9b.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["id"] in raw:
            s = M.units(raw[r["id"]], r["offsets"], r["surp"], "nopunct")
            if len(s) >= 6:
                out[raw[r["id"]].replace("\n", " ")] = resample(s[1:])
    return out


def _chrf(a):
    import sacrebleu
    x, y = a
    if x == y:
        return 100.0
    return (sacrebleu.sentence_chrf(x, [y]).score + sacrebleu.sentence_chrf(y, [x]).score) / 2


def pair_table(groups):
    """groups: {(scope, doc, src): [(system, human, text, da)]} -> long table of pairs."""
    embs = np.load(D / "emb.npy", mmap_mode="r")
    idx = {t: i for i, t in enumerate(json.load(open(D / "emb_texts.json", encoding="utf-8")))}
    prof = {}
    cs = dict(zip(*U.read_texts("cs")[["id", "text"]].values.T))
    for line in open(U.DATA / "surp.cs.eurollm9b.jsonl", encoding="utf-8"):
        r = json.loads(line)
        s = M.units(cs[r["id"]], r["offsets"], r["surp"], "nopunct")
        if len(s) >= 6:
            prof[cs[r["id"]]] = resample(s[1:])
    for ds in ("par3", "mendeley"):
        prof.update(lit_profiles(ds))
    rows, jobs = [], []
    for (scope, doc, src), items in groups.items():
        for a, b in itertools.combinations(items, 2):
            kind = "hh" if a[1] and b[1] else ("mm" if not a[1] and not b[1] else "hm")
            rows.append([scope, doc, src, a[0], b[0], kind, a[3], b[3]])
            jobs.append((a[2], b[2]))
    with Pool(16) as p:
        chrf = p.map(_chrf, jobs, chunksize=500)
    cos, prf = [], []
    for x, y in jobs:
        cos.append(float(embs[idx[x]] @ embs[idx[y]]))
        px, py = prof.get(x), prof.get(y)
        prf.append(np.corrcoef(px, py)[0, 1] if px is not None and py is not None and px.std() > 0 and py.std() > 0 else np.nan)
    df = pd.DataFrame(rows, columns=["scope", "doc", "src", "a", "b", "kind", "da_a", "da_b"])
    df["chrf"], df["cos"], df["prof_r"] = chrf, cos, prf
    return df


def pairs():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    groups = defaultdict(list)
    for y, doc, seg, s, h, t, da in rows[["year", "doc_id", "seg_idx", "system", "human", "text", "da"]].values:
        groups[(f"wmt{y}", doc, f"{doc}#{seg}")].append((s, h, t, da))
    for ds, s, ref, h, tid, t in literary_items():
        groups[(ds, s.split("#")[0] if ds == "par3" else s, s)].append((ref, h, t, np.nan))
    df = pair_table(groups)
    df.to_csv(OUT / "pairs_raw.csv.gz", index=False)
    res = []
    for da in (False, True):
        for scope, d in df.groupby("scope"):
            if da and not scope.startswith("wmt"):
                continue
            x = d[(d.da_a >= 90) & (d.da_b >= 90)] if da else d
            for sim in ("chrf", "cos", "prof_r"):
                per = x.dropna(subset=[sim]).groupby(["doc", "src", "kind"])[sim].mean().unstack("kind")
                for c, base in [("mm", "hh"), ("hm", "hh"), ("mm", "hm")]:
                    if c not in per or base not in per:
                        continue
                    w = per[[c, base]].dropna()
                    w = w.assign(diff=w[c] - w[base]).groupby("doc")["diff"].agg(["sum", "count"])
                    e, lo, hi = boot_docs({k: tuple(v) for k, v in zip(w.index, w.values)})
                    res.append(dict(scope=scope, filter="da90" if da else "all", sim=sim, contrast=f"{c}-{base}", n_src=int(w["count"].sum()), diff=e, lo=lo, hi=hi,
                                    mean_c=per[c].mean(), mean_base=per[base].mean()))
                # single-system-pair level: how many MT pairs are closer than the human pair?
                if (x.kind == "hh").any() and (x.kind == "mm").any():
                    hh_src = set(x[x.kind == "hh"].src)
                    y = x[x.src.isin(hh_src)].dropna(subset=[sim])
                    hh = y[y.kind == "hh"][sim].mean()
                    mm = y[y.kind == "mm"].groupby(["a", "b"])[sim].mean()
                    res.append(dict(scope=scope, filter="da90" if da else "all", sim=sim, contrast="mt_pairs_closer_than_hh", n_src=len(mm), diff=(mm > hh).mean(),
                                    mean_c=mm.mean(), mean_base=hh, lo=mm.min(), hi=mm.max()))
    pd.DataFrame(res).to_csv(OUT / "pairs.csv", index=False)


# ---------------------------------------------------------------- 2: lexical translation entropy
def links():
    """(src id, hyp id) -> [(source lemma + upos, target lemma)]"""
    en, cs = U.load("en"), U.load("cs")
    out = {}
    for line in open(U.DATA / "align.jsonl"):
        r = json.loads(line)
        se, sc = en[r["src"]], cs[r["hyp"]]
        out[(r["src"], r["hyp"])] = [(se[i]["lemma"].lower() + "/" + se[i]["upos"], sc[j]["lemma"].lower()) for i, j in r["a"] if se[i]["upos"] in CONTENT and sc[j]["upos"] != "PUNCT"]
    return out


def ent(xs):
    c = np.array(list(Counter(xs).values()), dtype=float)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def rarefied(lst, k=5, R=20, rng=None):
    return np.mean([ent(rng.choice(lst, k, replace=False)) for _ in range(R)])


def htra():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, L = load_rows(), links()
    res = []
    for da in (False, True):
        for y, d in rows.groupby("year"):
            systems = sorted(d.system.unique())
            hs = sorted(d[d.human].system.unique())

            def table(sys_, keep=None):
                t = defaultdict(list)
                x = d[d.system == sys_]
                if keep is not None:
                    x = x[[(a, b) in keep for a, b in zip(x.doc_id, x.seg_idx)]]
                for s, h in x[["src_id", "hyp_id"]].values:
                    for a, b in L.get((s, h), []):
                        t[a].append(b)
                return {a: v for a, v in t.items() if len(v) >= 5}

            def compare(a, b):
                keep = None
                if da:
                    pa = d[(d.system == a) & (d.da >= 90)][["doc_id", "seg_idx"]].apply(tuple, axis=1)
                    pb = d[(d.system == b) & (d.da >= 90)][["doc_id", "seg_idx"]].apply(tuple, axis=1)
                    keep = set(pa) & set(pb)
                ta, tb = table(a, keep), table(b, keep)
                com = sorted(set(ta) & set(tb))
                rng = np.random.default_rng(1)
                return np.array([rarefied(ta[l], rng=rng) - rarefied(tb[l], rng=rng) for l in com]), [rarefied(ta[l], rng=rng) for l in com], [rarefied(tb[l], rng=rng) for l in com]

            cells = []
            for m in [s for s in systems if s not in hs]:
                for h in hs:
                    diff, ea, eb = compare(m, h)
                    if len(diff) >= 20:
                        cells.append((m, h, diff))
                        res.append(dict(year=y, filter="da90" if da else "all", a=m, b=h, kind="mt-human", n_lemmas=len(diff), diff=diff.mean(), ent_a=np.mean(ea), ent_b=np.mean(eb)))
            if cells:
                sysm = pd.DataFrame([(m, h, c.mean()) for m, h, c in cells], columns=["m", "h", "d"])
                B = [np.mean([c[RNG.integers(0, len(c), len(c))].mean() for _, _, c in cells]) for _ in range(500)]
                res.append(dict(year=y, filter="da90" if da else "all", a="ALL_MT", b="humans", kind="mean", n_lemmas=int(np.mean([len(c) for _, _, c in cells])), diff=sysm.d.mean(), lo=np.percentile(B, 2.5), hi=np.percentile(B, 97.5),
                                n_below=int((sysm.d < 0).sum()), n_cells=len(sysm)))
            if len(hs) == 2:
                diff, ea, eb = compare(hs[0], hs[1])
                if len(diff) >= 20:
                    res.append(dict(year=y, filter="da90" if da else "all", a=hs[0], b=hs[1], kind="human-human", n_lemmas=len(diff), diff=diff.mean(), ent_a=np.mean(ea), ent_b=np.mean(eb)))
    pd.DataFrame(res).to_csv(OUT / "htra.csv", index=False)


# ---------------------------------------------------------------- 5: shifts
def passive(tok, sent):
    if "Voice=Pass" in tok["feats"]:
        return True
    return any(t["head"] == tok["id"] and t["sent"] == tok["sent"] and t["deprel"] in ("aux:pass", "expl:pass", "nsubj:pass") for t in sent)


def seg_shifts():
    en, cs = U.load("en"), U.load("cs")
    out = {}
    for line in open(U.DATA / "align.jsonl"):
        r = json.loads(line)
        se, sc = en[r["src"]], cs[r["hyp"]]
        a = {i: j for i, j in r["a"]}
        v = np.zeros(10)
        for i, j in a.items():
            x, y = se[i], sc[j]
            if x["upos"] not in CONTENT or y["upos"] == "PUNCT":
                continue
            v[1] += 1
            v[0] += CLASS.get(x["upos"]) != CLASS.get(y["upos"])
            v[3] += 1
            v[2] += x["deprel"].split(":")[0] != y["deprel"].split(":")[0]
            hx, hy = x["head"], y["head"]
            if hx > 0:
                v[5] += 1
                v[4] += a.get(hx - 1) != hy - 1
            if x["upos"] == "VERB" and CLASS.get(y["upos"]) == "V":
                v[7] += 1
                v[6] += passive(x, se) != passive(y, sc)
        cx = sum(t["deprel"].split(":")[0] in CLAUSE for t in se)
        cy = sum(t["deprel"].split(":")[0] in CLAUSE for t in sc)
        v[8], v[9] = abs(cx - cy), cx + 1
        out[(r["src"], r["hyp"])] = v
    return out


SHIFT = ["wordclass", "deprel", "head_lost", "voice", "clause_change"]


def shifts():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, S = load_rows(), seg_shifts()
    res = []
    empty = np.zeros(10)
    rows["vec"] = [S.get((a, b), empty) for a, b in zip(rows.src_id, rows.hyp_id)]
    for da in (False, True):
        for y, d in rows.groupby("year"):
            hs = sorted(d[d.human].system.unique())
            ms = sorted(set(d.system) - set(hs))

            def pair(a, b):
                x = d[d.system == a].set_index(["doc_id", "seg_idx"])
                z = d[d.system == b].set_index(["doc_id", "seg_idx"])
                com = x.index.intersection(z.index)
                if da:
                    com = com[(x.loc[com, "da"].values >= 90) & (z.loc[com, "da"].values >= 90)]
                docs = com.get_level_values(0)
                va, vb = np.stack(x.loc[com, "vec"].values), np.stack(z.loc[com, "vec"].values)
                return docs, va, vb

            def rates(v):
                return v[..., ::2] / np.maximum(v[..., 1::2], 1)

            pd_all = {}
            for m in ms:
                for h in hs:
                    docs, va, vb = pair(m, h)
                    if len(docs) >= 20:
                        pd_all[(m, h)] = (pd.DataFrame(va).groupby(docs).sum().values, pd.DataFrame(vb).groupby(docs).sum().values)
            if not pd_all:
                continue
            est = np.array([rates(a.sum(0)) - rates(b.sum(0)) for a, b in pd_all.values()])
            B = []
            for _ in range(500):
                one = []
                for a, b in pd_all.values():
                    i = RNG.integers(0, len(a), len(a))
                    one.append(rates(a[i].sum(0)) - rates(b[i].sum(0)))
                B.append(np.mean(one, axis=0))
            lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
            hrate = np.mean([rates(b.sum(0)) for _, b in pd_all.values()], axis=0)
            for k, n in enumerate(SHIFT):
                row = dict(year=y, filter="da90" if da else "all", measure=n, n_cells=len(est), human=hrate[k], diff=est[:, k].mean(), lo=lo[k], hi=hi[k], n_mt_below=int((est[:, k] < 0).sum()))
                if len(hs) == 2:
                    docs, va, vb = pair(hs[0], hs[1])
                    if len(docs) >= 20:
                        row["hh_diff"] = rates(va.sum(0))[k] - rates(vb.sum(0))[k]
                res.append(row)
    pd.DataFrame(res).to_csv(OUT / "shifts.csv", index=False)


# ---------------------------------------------------------------- 4: identifiability
def chunk_docs(d, cs):
    """{(doc, chunk): {system: (delex string, text)}} on segments in doc order, 10 segments per chunk."""
    d = d.sort_values(["doc_id", "seg_idx"])
    per = {}
    for (doc, seg), x in d.groupby(["doc_id", "seg_idx"], sort=False):
        per.setdefault(doc, []).append(x)
    out = defaultdict(dict)
    for doc, segs in per.items():
        for c in range(0, len(segs) - 9, 10):
            for x in segs[c:c + 10]:
                for s, h, t, da in x[["system", "hyp_id", "text", "da"]].values:
                    out[(doc, c)].setdefault(s, []).append((h, t, da))
    return out


def delex(tokens):
    upos = [t["upos"] for t in tokens]
    grams = ["_".join(upos[i:i + 3]) for i in range(len(upos) - 2)]
    func = ["F=" + t["lemma"].lower() for t in tokens if t["upos"] in FUNC]
    return " ".join(grams + func)


def ident():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    cs = U.load("cs")
    res = []
    for da in (False, True):
        for y, d in rows.groupby("year"):
            ch = chunk_docs(d, cs)
            systems = sorted(d.system.unique())
            hs = set(d[d.human].system.unique())
            for a, b in itertools.combinations(systems, 2):
                X_d, X_c, Y, G = [], [], [], []
                for (doc, c), per in ch.items():
                    if a not in per or b not in per or len(per[a]) < 8 or len(per[b]) < 8:
                        continue
                    pa, pb = per[a], per[b]
                    if da:
                        n = min(len(pa), len(pb))
                        keep = [i for i in range(n) if pa[i][2] >= 90 and pb[i][2] >= 90]
                        if len(keep) < 5:
                            continue
                        pa, pb = [pa[i] for i in keep], [pb[i] for i in keep]
                    for lab, p in ((0, pa), (1, pb)):
                        toks = [t for h, _, _ in p for t in cs.get(h, [])]
                        X_d.append(delex(toks))
                        X_c.append(" ".join(t for _, t, _ in p))
                        Y.append(lab)
                        G.append(doc)
                Y, G = np.array(Y), np.array(G)
                if len(Y) < 40 or len(set(G)) < 5:
                    continue
                accs = {}
                for name, X, vec in (("delex", X_d, TfidfVectorizer(token_pattern=r"\S+", sublinear_tf=True, min_df=2)),
                                     ("char", X_c, TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True, min_df=2, max_features=100000))):
                    pred = np.zeros(len(Y))
                    for tr, te in GroupKFold(5).split(X, Y, G):
                        v = vec.fit([X[i] for i in tr])
                        lr = LogisticRegression(C=10, max_iter=1000).fit(v.transform([X[i] for i in tr]), Y[tr])
                        pred[te] = lr.predict(v.transform([X[i] for i in te]))
                    accs[name] = (pred == Y).mean()
                kind = "hh" if a in hs and b in hs else ("mm" if a not in hs and b not in hs else "hm")
                res.append(dict(year=y, filter="da90" if da else "all", a=a, b=b, kind=kind, n=len(Y), **accs))
                print(y, da, a, b, kind, accs, flush=True)
    pd.DataFrame(res).to_csv(OUT / "ident.csv", index=False)


if __name__ == "__main__":
    {"embed": embed, "pairs": pairs, "htra": htra, "shifts": shifts, "ident": ident}[sys.argv[1]]()
