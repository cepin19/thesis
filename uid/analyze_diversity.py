#!/usr/bin/env python3
"""Is MT less diverse than human translation? Corpus-level diversity measures on WMT20-22 EN->CS.

Every system (MT or human) translated the same test set, so each measure is computed per system on the
same source segments and compared as  mean over MT systems - mean over human translations, per year.
CIs: bootstrap over documents (the same resample for all systems). In 2021/22 the two human
translations are also compared with each other (ht_pair_diff) as a yardstick for natural variation.
With --min-da-raw T each comparison (MT system vs human, and human vs human) uses only the segments
where both translations have raw DA >= T.

Measure groups (TODO §3c):
  surp  (2) word-level surprisal distribution (untok words, first word dropped, as in uid/measures.py):
            mean, word SD/CV, quantiles, share above the human p90/p99, share below 1 bit, entropy of the
            0.25-bit histogram, between-segment SD/CV of segment mean surprisal
  tok   (3) token-level LM rank and entropy (needs surpx.*.jsonl from `surprisal.py --extra`):
            share of top-1 / top-5 tokens, mean log2 rank, mean entropy, mean surprisal - entropy
  lex   (5) MATTR-500 (forms, lemmas), MTLD, Yule's K (lemmas), lexical density
  syn   (7) windowed entropy of POS trigrams and of dependency frames, mean tree depth, dependency length,
            clauses per sentence, sentence length CV, word-order crossings and aligned share (uid/align.py)
  doc   (8) within-document content-lemma repetition rate; translation consistency of repeated source
            content lemmas (share of agreeing pairs of aligned target lemmas), nouns vs other
Words for lex/syn/doc are UDPipe 1.4 (UD 2.5) syntactic words without punctuation.

--sent-split: use the sentence-split parse, so tree measures, frames, POS trigrams and sentence length are per real
sentence, and add between-sentence surprisal spread (sent_sd, sent_cv) next to the segment-level seg_sd, seg_cv.
--context: surprisal, rank and entropy given the preceding segments of the document.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import measures as M
import udpipe_io as U

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/wmt_human"
LMS = ["czgpt2xl", "eurollm9b", "customgpt2"]
CONTENT = {"NOUN", "VERB", "ADJ", "ADV"}
CLAUSE = {"ccomp", "xcomp", "advcl", "acl", "acl:relcl", "csubj", "csubj:pass", "parataxis"}
BINS = np.arange(0, 60.25, 0.25)
NB = len(BINS) - 1
RNG = np.random.default_rng(1)

# +1: higher value = more diverse / less predictable; -1: higher = less diverse; 0: no direction
DIRECTION = {
    "mean": 0, "word_sd": 1, "word_cv": 1, "p50": 0, "p90": 0, "p99": 0, "gt_ht_p90": 1, "gt_ht_p99": 1,
    "lt_1bit": -1, "hist_entropy": 1, "seg_sd": 1, "seg_cv": 1, "sent_sd": 1, "sent_cv": 1,
    "top1": -1, "top5": -1, "log2rank": 1, "entropy": 0, "surp_minus_ent": 1,
    "mattr_form": 1, "mattr_lemma": 1, "mtld": 1, "yule_k": -1, "lex_density": 0,
    "pos3_entropy": 1, "frame_entropy": 1, "depth": 0, "dep_len": 0, "clauses": 0, "len_cv": 1,
    "crossing": 1, "crossing_in_sent": 1, "aligned": -1, "sents_per_seg": 0,
    "rep_rate": -1, "consist_noun": -1, "consist_other": -1,
}


# ------------------------------------------------------------------ per-segment features
class Feats:
    """Additive per-segment feature vectors, addressed by name."""

    def __init__(self):
        self.names, self.size = {}, 0

    def add(self, name, k=1):
        self.names[name] = slice(self.size, self.size + k)
        self.size += k

    def __getitem__(self, name):
        return self.names[name]


def build_layout(lms, have_tok):
    F = Feats()
    for lm in lms:
        for n in ("nw", "s", "s2", "nseg", "m", "m2", "nsn", "sm", "sm2"):
            F.add(f"{lm}:{n}")
        F.add(f"{lm}:hist", NB)
        if have_tok[lm]:
            for n in ("nt", "r1", "r5", "lr", "ent", "st"):
                F.add(f"{lm}:{n}")
    for n in ("nsegs", "nsent", "nw", "nw2", "depth", "dl", "ndep", "clauses", "content",
              "al_pairs", "al_disc", "al_pairs_s", "al_disc_s", "al_tgt", "al_ntgt"):
        F.add(f"syn:{n}")
    return F


def word_surp(texts, path, unit="untok"):
    """id -> (unit surprisals without the first unit, char start of each unit)."""
    out, extra = {}, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            t = texts[r["id"]]
            out[r["id"]] = (M.units(t, r["offsets"], r["surp"], unit)[1:], M.unit_starts(t, unit)[1:])
            if "rank" in r:
                extra[r["id"]] = (np.array(r["surp"][1:]), np.array(r["ent"][1:]), np.array(r["rank"][1:]))
    return out, extra


def tree_depth(toks):
    head = {t["id"]: t["head"] for t in toks}
    best = 0
    for t in toks:
        d, h = 0, t["id"]
        while h and d < 200:
            h, d = head.get(h, 0), d + 1
        best = max(best, d)
    return best


def frames(toks):
    kids = defaultdict(list)
    for t in toks:
        if t["deprel"] != "punct" and t["head"]:
            kids[t["head"]].append(t)
    out = []
    for t in toks:
        if t["upos"] == "PUNCT":
            continue
        seq = sorted(kids[t["id"]] + [{"id": t["id"], "deprel": "H"}], key=lambda x: x["id"])
        out.append(t["upos"] + ">" + "_".join(x["deprel"].split(":")[0] for x in seq))
    return out


def by_sentence(toks):
    out = defaultdict(list)
    for t in toks:
        out[t["sent"]].append(t)
    return [out[k] for k in sorted(out)]


def seg_features(keys, F, lms, surp, tok, ud, id2text, starts):
    """keys: (key, cs-id) pairs, where key addresses surprisals. Returns key -> (vector, sequences)."""
    out = {}
    for i, h in keys:
        v = np.zeros(F.size)
        for lm in lms:
            s, us = surp[lm][i]
            if len(s):
                v[F[f"{lm}:nw"]] = len(s)
                v[F[f"{lm}:s"]] = s.sum()
                v[F[f"{lm}:s2"]] = (s ** 2).sum()
                v[F[f"{lm}:nseg"]] = 1
                v[F[f"{lm}:m"]] = s.mean()
                v[F[f"{lm}:m2"]] = s.mean() ** 2
                v[F[f"{lm}:hist"]] = np.histogram(np.clip(s, 0, 59.99), BINS)[0]
                parts = M.split_sentences(s, us, np.array(starts[id2text[h]])) if starts else [s]
                ms = np.array([p.mean() for p in parts])
                v[F[f"{lm}:nsn"]], v[F[f"{lm}:sm"]], v[F[f"{lm}:sm2"]] = len(ms), ms.sum(), (ms ** 2).sum()
            if lm in tok and i in tok[lm]:
                st, ent, rank = tok[lm][i]
                v[F[f"{lm}:nt"]] = len(st)
                v[F[f"{lm}:r1"]] = (rank == 1).sum()
                v[F[f"{lm}:r5"]] = (rank <= 5).sum()
                v[F[f"{lm}:lr"]] = np.log2(rank).sum()
                v[F[f"{lm}:ent"]] = ent.sum()
                v[F[f"{lm}:st"]] = st.sum()
        toks = ud[h]
        words = [t for t in toks if t["upos"] != "PUNCT"]
        sents = by_sentence(toks)
        v[F["syn:nsegs"]] = 1
        v[F["syn:content"]] = sum(t["upos"] in CONTENT for t in words)
        pos3, frame = [], []
        for st_ in sents:
            w = [t for t in st_ if t["upos"] != "PUNCT"]
            n = len(w)
            v[F["syn:nsent"]] += 1
            v[F["syn:nw"]] += n
            v[F["syn:nw2"]] += n * n
            v[F["syn:depth"]] += tree_depth(st_)
            deps = [abs(t["head"] - t["id"]) for t in w if t["head"]]
            v[F["syn:dl"]] += sum(deps)
            v[F["syn:ndep"]] += len(deps)
            v[F["syn:clauses"]] += 1 + sum(t["deprel"] in CLAUSE for t in st_)
            pos = ["<s>"] + [t["upos"] for t in w] + ["</s>"]
            pos3 += ["|".join(pos[k:k + 3]) for k in range(len(pos) - 2)]
            frame += frames(st_)
        seqs = {"form": [t["form"].lower() for t in words], "lemma": [t["lemma"].lower() for t in words],
                "pos3": pos3, "frame": frame,
                "content": [t["lemma"].lower() for t in words if t["upos"] in CONTENT and t["lemma"] != "být"]}
        out[i] = (v, seqs)
    return out


def align_features(F, ud_en, ud_cs, path):
    """Return dict (en-id, cs-id) -> (vector, content links [(src lemma, tgt lemma, src upos)])."""
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            a = np.array(r["a"]).reshape(-1, 2)
            se, sc = ud_en[r["src"]], ud_cs[r["hyp"]]
            keep = [(i, j) for i, j in a if se[i]["upos"] != "PUNCT" and sc[j]["upos"] != "PUNCT"]
            v = np.zeros(F.size)
            k = len(keep)
            if k >= 2:
                jj = np.array([j for _, j in sorted(keep)])
                v[F["syn:al_pairs"]] = k * (k - 1) / 2
                iu = np.triu_indices(k, 1)
                disc = (jj[:, None] > jj[None, :])[iu]
                v[F["syn:al_disc"]] = disc.sum()
                ss = np.array([sc[j]["sent"] for j in jj])
                same = (ss[:, None] == ss[None, :])[iu]
                v[F["syn:al_pairs_s"]] = same.sum()
                v[F["syn:al_disc_s"]] = (disc & same).sum()
            v[F["syn:al_tgt"]] = len({j for _, j in keep})
            v[F["syn:al_ntgt"]] = sum(t["upos"] != "PUNCT" for t in sc)
            links = [(se[i]["lemma"].lower(), sc[j]["lemma"].lower(), se[i]["upos"]) for i, j in keep
                     if se[i]["upos"] in CONTENT and sc[j]["upos"] in CONTENT and se[i]["lemma"] != "be"]
            out[(r["src"], r["hyp"])] = (v, links)
    return out


# ------------------------------------------------------------------ doc-level and sequence measures
def doc_features(segs):
    """segs: list of (seqs, links) of one document -> [rep_tok, rep_n, agree_noun, pairs_noun, agree_o, pairs_o]."""
    seen, rep, n = set(), 0, 0
    by_src = defaultdict(list)
    for seqs, links in segs:
        for l in seqs["content"]:
            rep += l in seen
            n += 1
            seen.add(l)
        for s, t, p in links:
            by_src[(s, p == "NOUN")].append(t)
    out = np.zeros(6)
    out[0], out[1] = rep, n
    for (s, noun), ts in by_src.items():
        k = len(ts)
        if k < 2:
            continue
        c = Counter(ts)
        o = 2 if noun else 4
        out[o] += sum(x * (x - 1) / 2 for x in c.values())
        out[o + 1] += k * (k - 1) / 2
    return out


def mattr(tokens, w=500):
    n = len(tokens)
    if n < w:
        return np.nan
    last, prev = {}, np.empty(n, dtype=np.int64)
    for i, t in enumerate(tokens):
        prev[i] = last.get(t, -1)
        last[t] = i
    i = np.arange(n)
    hi = np.minimum(i, n - w)
    lo = np.maximum(np.maximum(i - w + 1, prev + 1), 0)
    return np.clip(hi - lo + 1, 0, None).sum() / (n - w + 1) / w


def mtld_dir(tokens, thr=0.72):
    factors, types, cnt = 0.0, set(), 0
    for t in tokens:
        types.add(t)
        cnt += 1
        if len(types) / cnt <= thr:
            factors += 1
            types, cnt = set(), 0
    if cnt:
        factors += (1 - len(types) / cnt) / (1 - thr)
    return len(tokens) / factors if factors else np.nan


def mtld(tokens):
    return (mtld_dir(tokens) + mtld_dir(tokens[::-1])) / 2


def yule_k(tokens):
    c = np.array(list(Counter(tokens).values()), dtype=float)
    n = c.sum()
    return 1e4 * ((c ** 2).sum() - n) / n ** 2


def win_entropy(items, w=2000):
    k = len(items) // w
    if k == 0:
        return np.nan
    hs = []
    for b in range(k):
        c = np.array(list(Counter(items[b * w:(b + 1) * w]).values()), dtype=float) / w
        hs.append(-(c * np.log2(c)).sum())
    return float(np.mean(hs))


SEQ_MEASURES = ["mattr_form", "mattr_lemma", "mtld", "yule_k", "pos3_entropy", "frame_entropy"]


def seq_measures(docs, order):
    """docs: list (per doc) of list of seqs dicts; order: doc indices (bootstrap resample)."""
    cat = defaultdict(list)
    for d in order:
        for seqs in docs[d]:
            for k in ("form", "lemma", "pos3", "frame"):
                cat[k].extend(seqs[k])
    return np.array([mattr(cat["form"]), mattr(cat["lemma"]), mtld(cat["form"]), yule_k(cat["lemma"]),
                     win_entropy(cat["pos3"]), win_entropy(cat["frame"])])


# ------------------------------------------------------------------ additive measures from summed features
def hist_quantile(h, q):
    c = np.cumsum(h, axis=-1)
    tot = c[..., -1:]
    k = (c < q * tot).sum(-1)
    return BINS[np.minimum(k, NB - 1)] + 0.125


def add_measures(X, D, F, lms, have_tok, thresholds):
    """X: [..., F.size] summed seg features; D: [..., 6] summed doc features -> dict name -> array."""
    out = {}
    g = lambda n: X[..., F[n]].squeeze(-1) if F[n].stop - F[n].start == 1 else X[..., F[n]]
    with np.errstate(invalid="ignore", divide="ignore"):
        for lm in lms:
            nw, s, s2 = g(f"{lm}:nw"), g(f"{lm}:s"), g(f"{lm}:s2")
            mu = s / nw
            sd = np.sqrt(s2 / nw - mu ** 2)
            ns, m, m2 = g(f"{lm}:nseg"), g(f"{lm}:m"), g(f"{lm}:m2")
            segmu = m / ns
            segsd = np.sqrt(m2 / ns - segmu ** 2)
            h = g(f"{lm}:hist")
            p = h / h.sum(-1, keepdims=True)
            t90, t99 = thresholds[lm]
            out |= {f"{lm}:mean": mu, f"{lm}:word_sd": sd, f"{lm}:word_cv": sd / mu,
                    f"{lm}:p50": hist_quantile(h, .5), f"{lm}:p90": hist_quantile(h, .9),
                    f"{lm}:p99": hist_quantile(h, .99),
                    f"{lm}:gt_ht_p90": p[..., BINS[:-1] >= t90].sum(-1), f"{lm}:gt_ht_p99": p[..., BINS[:-1] >= t99].sum(-1),
                    f"{lm}:lt_1bit": p[..., :4].sum(-1),
                    f"{lm}:hist_entropy": -(np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0)).sum(-1),
                    f"{lm}:seg_sd": segsd, f"{lm}:seg_cv": segsd / segmu}
            nn, sm, sm2 = g(f"{lm}:nsn"), g(f"{lm}:sm"), g(f"{lm}:sm2")
            sentmu = sm / nn
            sentsd = np.sqrt(sm2 / nn - sentmu ** 2)
            out |= {f"{lm}:sent_sd": sentsd, f"{lm}:sent_cv": sentsd / sentmu}
            if have_tok[lm]:
                nt = g(f"{lm}:nt")
                out |= {f"{lm}:top1": g(f"{lm}:r1") / nt, f"{lm}:top5": g(f"{lm}:r5") / nt,
                        f"{lm}:log2rank": g(f"{lm}:lr") / nt, f"{lm}:entropy": g(f"{lm}:ent") / nt,
                        f"{lm}:surp_minus_ent": (g(f"{lm}:st") - g(f"{lm}:ent")) / nt}
        ns, nw, nw2 = g("syn:nsent"), g("syn:nw"), g("syn:nw2")
        lm_ = nw / ns
        out |= {"lex_density": g("syn:content") / nw, "depth": g("syn:depth") / ns,
                "dep_len": g("syn:dl") / g("syn:ndep"), "clauses": g("syn:clauses") / ns,
                "len_cv": np.sqrt(nw2 / ns - lm_ ** 2) / lm_,
                "crossing": g("syn:al_disc") / g("syn:al_pairs"),
                "crossing_in_sent": g("syn:al_disc_s") / g("syn:al_pairs_s"),
                "sents_per_seg": ns / g("syn:nsegs"), "aligned": g("syn:al_tgt") / g("syn:al_ntgt"),
                "rep_rate": D[..., 0] / D[..., 1], "consist_noun": D[..., 2] / D[..., 3],
                "consist_other": D[..., 4] / D[..., 5]}
    return out


# ------------------------------------------------------------------ main
def load_rows():
    df = pd.concat([pd.read_csv(DATA / f"wmt{y}.en-cs.tsv", sep="\t", keep_default_na=False,
                                na_values={"da_raw": [""], "da_z": [""]}) for y in (20, 21, 22)])
    df["da_raw"] = pd.to_numeric(df["da_raw"], errors="coerce")
    en = dict(zip(*U.read_texts("en")[["text", "id"]].values.T))
    cs = dict(zip(*U.read_texts("cs")[["text", "id"]].values.T))
    df["src_id"] = df["src"].map(lambda s: en[M.norm_text(s)])
    df["hyp_id"] = df["hyp"].map(lambda s: cs[M.norm_text(s)])
    df["is_human"] = df["is_human"].astype(str) == "True"
    return df.sort_values(["year", "doc_id", "seg_idx", "system"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-da-raw", type=float, default=None)
    ap.add_argument("--boot", type=int, default=1000, help="resamples for additive measures")
    ap.add_argument("--boot-seq", type=int, default=200, help="resamples for MATTR/MTLD/Yule/entropy")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sent-split", action="store_true")
    ap.add_argument("--context", action="store_true")
    args = ap.parse_args()
    out = args.out or ROOT / ("results/diversity" + ("_sent" if args.sent_split else "") + ("_ctx" if args.context else "")
                              + (f"_da{args.min_da_raw:g}" if args.min_da_raw else ""))
    out.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    rows["year"] = rows["year"].astype(int)
    id2text = dict(zip(*U.read_texts("cs")[["id", "text"]].values.T))
    if args.context:
        ctx = pd.read_csv(DATA / "texts_ctx.cs.tsv", sep="\t", keep_default_na=False, quoting=3)
        texts = dict(zip(ctx["id"], ctx["text"]))
        rows["seg_key"] = [f"{y}|{s}|{d}|{k}" for y, s, d, k in
                           zip(rows["year"], rows["system"], rows["doc_id"], rows["seg_idx"])]
    else:
        texts = id2text
        rows["seg_key"] = rows["hyp_id"]
    surp, tok = {}, {}
    for lm in LMS:
        px = DATA / f"surpctx.cs.{lm}.jsonl" if args.context else DATA / f"surpx.cs.{lm}.jsonl"
        surp[lm], ext = word_surp(texts, px)
        if ext:
            tok[lm] = ext
    have_tok = {lm: lm in tok for lm in LMS}
    print("token rank/entropy available for:", [lm for lm in LMS if have_tok[lm]], flush=True)
    F = build_layout(LMS, have_tok)
    ud_dir = "udpipe_sent" if args.sent_split else "udpipe"
    ud_cs, ud_en = U.load("cs", ud_dir), U.load("en", ud_dir)
    starts = U.sentence_starts("cs") if args.sent_split else None
    keys = rows[["seg_key", "hyp_id"]].drop_duplicates().itertuples(index=False, name=None)
    seg = seg_features(keys, F, LMS, surp, tok, ud_cs, id2text, starts)
    alf = align_features(F, ud_en, ud_cs, DATA / ("align_sent.jsonl" if args.sent_split else "align.jsonl"))
    print("features done", flush=True)

    recs, sysvals = [], []
    for year, yr in rows.groupby("year"):
        docs = sorted(yr["doc_id"].unique())
        dix = {d: k for k, d in enumerate(docs)}
        systems = sorted(yr["system"].unique())
        humans = sorted(yr.loc[yr["is_human"], "system"].unique())
        mts = [s for s in systems if s not in humans]
        piv = yr.pivot_table(index=["doc_id", "seg_idx"], columns="system", values="da_raw")
        bysys = {s: g.set_index(["doc_id", "seg_idx"]) for s, g in yr.groupby("system")}

        # comparisons: (a, b) = (MT, human) for every pair, plus (human1, human2)
        comps = [(m, h) for m in mts for h in humans] + ([tuple(humans)] if len(humans) == 2 else [])

        def mask_for(a, b):
            if args.min_da_raw is None:
                return None
            ok = (piv[a] >= args.min_da_raw) & (piv[b] >= args.min_da_raw)
            return set(ok[ok].index)

        masks = {c: mask_for(*c) for c in comps}

        def system_data(s, mask):
            g = bysys[s]
            X = np.zeros((len(docs), F.size))
            D = np.zeros((len(docs), 6))
            seqdocs = [[] for _ in docs]
            per_doc = defaultdict(list)
            for (d, k), r in g.iterrows():
                if mask is not None and (d, k) not in mask:
                    continue
                v, seqs = seg[r["seg_key"]]
                av, links = alf[(r["src_id"], r["hyp_id"])]
                X[dix[d]] += v + av
                seqdocs[dix[d]].append(seqs)
                per_doc[dix[d]].append((seqs, links))
            for d, segs in per_doc.items():
                D[d] = doc_features(segs)
            return X, D, seqdocs

        cache = {}

        def get(s, c):
            key = (s, None if args.min_da_raw is None else c)
            if key not in cache:
                cache[key] = system_data(s, masks[c])
            return cache[key]

        # human thresholds for gt_ht_p90/p99: pooled over the human translations (full test set)
        Xh = sum(system_data(h, None)[0].sum(0) for h in humans)
        thresholds = {lm: (hist_quantile(Xh[F[f"{lm}:hist"]], .9), hist_quantile(Xh[F[f"{lm}:hist"]], .99))
                      for lm in LMS}

        nd = len(docs)
        W = np.stack([np.bincount(RNG.integers(0, nd, nd), minlength=nd) for _ in range(args.boot)])
        Ws = W[:args.boot_seq]
        point, boot = {}, {}
        for c in comps:
            for s in c:
                X, D, seqdocs = get(s, c)
                key = (s, c if args.min_da_raw is not None else None)
                if key in point:
                    continue
                pa = add_measures(X.sum(0), D.sum(0), F, LMS, have_tok, thresholds)
                ba = add_measures(W @ X, W @ D, F, LMS, have_tok, thresholds)
                ps = seq_measures(seqdocs, range(nd))
                bs = np.stack([seq_measures(seqdocs, np.repeat(np.arange(nd), w)) for w in Ws])
                point[key] = pa | dict(zip(SEQ_MEASURES, ps))
                boot[key] = ba | {m: bs[:, k] for k, m in enumerate(SEQ_MEASURES)}
        names = list(next(iter(point.values())).keys())

        def k_(s, c):
            return (s, c if args.min_da_raw is not None else None)

        for s in systems:
            cs_ = [c for c in comps if s in c and (s in humans or c[1] in humans)]
            if not cs_:
                continue
            for mname in names:
                vals = [point[k_(s, c)][mname] for c in cs_]
                sysvals.append({"year": year, "system": s, "is_human": s in humans, "measure": mname,
                                "value": float(np.nanmean(vals))})
        for mname in names:
            mt_pairs = [(m, h) for m in mts for h in humans]
            diff_p = np.mean([point[k_(m, (m, h))][mname] - point[k_(h, (m, h))][mname] for m, h in mt_pairs])
            nb = len(boot[k_(mts[0], mt_pairs[0])][mname])
            diff_b = np.mean([boot[k_(m, (m, h))][mname][:nb] - boot[k_(h, (m, h))][mname][:nb]
                              for m, h in mt_pairs], axis=0)
            ht = np.mean([point[k_(h, (m, h))][mname] for m, h in mt_pairs])
            mt = np.mean([point[k_(m, (m, h))][mname] for m, h in mt_pairs])
            mt_each = {m: np.mean([point[k_(m, (m, h))][mname] for h in humans]) for m in mts}
            ht_each = [point[k_(h, (m, h))][mname] for m, h in mt_pairs]
            rec = {"year": year, "measure": mname, "ht": ht, "mt": mt, "diff": diff_p,
                   "rel_diff": diff_p / abs(ht) if ht else np.nan,
                   "lo": np.nanpercentile(diff_b, 2.5), "hi": np.nanpercentile(diff_b, 97.5),
                   "mt_below_ht_min": sum(v < np.nanmin(ht_each) for v in mt_each.values()),
                   "mt_above_ht_max": sum(v > np.nanmax(ht_each) for v in mt_each.values()),
                   "n_mt": len(mts)}
            if len(humans) == 2:
                c = tuple(humans)
                rec["ht_pair_diff"] = point[k_(humans[0], c)][mname] - point[k_(humans[1], c)][mname]
                hb = boot[k_(humans[0], c)][mname] - boot[k_(humans[1], c)][mname]
                rec["ht_pair_lo"], rec["ht_pair_hi"] = np.nanpercentile(hb, [2.5, 97.5])
            recs.append(rec)
        print(f"{year} done", flush=True)

    res = pd.DataFrame(recs)
    base = res["measure"].str.split(":").str[-1]
    res["lm"] = np.where(res["measure"].str.contains(":"), res["measure"].str.split(":").str[0], "")
    res["direction"] = base.map(DIRECTION)
    sig = (res["lo"] > 0) | (res["hi"] < 0)
    res["mt_less_diverse"] = np.where(~sig | (res["direction"] == 0), "",
                                      np.where(np.sign(res["diff"]) == -res["direction"], "yes", "NO (more)"))
    res.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(sysvals).to_csv(out / "system_values.csv", index=False)
    with open(out / "summary.md", "w") as f:
        f.write(f"# MT vs HT diversity, WMT20-22 EN->CS{'' if args.min_da_raw is None else f' (DA >= {args.min_da_raw:g})'}"
                f"{', sentence-split' if args.sent_split else ''}{', document context' if args.context else ''}\n\n")
        f.write("diff = mean over MT systems - mean over human translations; 95% CI from a document bootstrap.\n"
                "mt_less_diverse: significant and in the 'less diverse' direction (see DIRECTION in the script).\n\n")
        cols = ["year", "measure", "ht", "mt", "diff", "rel_diff", "lo", "hi", "mt_less_diverse",
                "mt_below_ht_min", "mt_above_ht_max", "n_mt", "ht_pair_diff", "ht_pair_lo", "ht_pair_hi"]
        f.write(res[[c for c in cols if c in res]].to_markdown(index=False, floatfmt=".4g"))
        f.write("\n")
    print(res[["year", "measure", "ht", "mt", "diff", "lo", "hi", "mt_less_diverse"]].to_string())


if __name__ == "__main__":
    main()
