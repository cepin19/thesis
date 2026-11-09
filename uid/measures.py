"""Word-level aggregation of token surprisals and sentence-level uniformity measures.

Unit definitions (`unit=`), all derived from the same token surprisals via char offsets:
  untok   : whitespace-separated words, punctuation attached to the adjacent word
            (the default setting of Jon & Bojar 2024)
  tok     : words and punctuation as separate units (regex \\w+ | [^\\w\\s]+)
  nopunct : as `tok`, but punctuation units are dropped
A token is assigned to the unit containing its first non-space character; unit surprisal is the
sum of its tokens' surprisals.

Measures (s = unit surprisals, optionally without the first unit):
  n, mean, total, sd, cv = sd/mean, gini, lv = mean (s_i - s_{i-1})^2 (local variance, "LV^2" in
  the paper), lv_norm = lv / mean^2 (scale-free LV), gv = mean (s_i - mu_corpus)^2 (needs the
  corpus mean, filled in by add_global_variance), sl = mean s^1.25, max.

Segment level is the default. With sentence starts (`sents=`), the first unit of every sentence is dropped,
sd/cv/gini/lv/lv_norm/sl are computed per sentence and averaged weighted by sentence length (LV never crosses
a sentence boundary), n is the mean sentence length, and mean/total/max/gv are pooled over the segment.
"""
import json
import re

import numpy as np
import pandas as pd

UNIT_RE = re.compile(r"\w+|[^\w\s]+")
MEASURES = ["n", "mean", "total", "sd", "cv", "gini", "lv", "lv_norm", "gv", "sl", "max"]


def norm_text(s):
    return s.replace("\t", " ").replace("\n", " ")


def units(text, offsets, surp, unit):
    if unit == "untok":
        spans = [(m.start(), m.end(), True) for m in re.finditer(r"\S+", text)]
    else:
        spans = [(m.start(), m.end(), m.group()[0].isalnum() or m.group()[0] == "_")
                 for m in UNIT_RE.finditer(text)]
    if not spans:
        return np.array([])
    starts = np.array([s for s, _, _ in spans])
    vals = np.zeros(len(spans))
    for (a, b), v in zip(offsets, surp):
        p = a
        while p < b and text[p].isspace():
            p += 1
        if p >= b:  # whitespace-only token: give it to the next unit
            p = b
        k = max(np.searchsorted(starts, p, side="right") - 1, 0)
        vals[k] += v
    if unit == "nopunct":
        vals = vals[[w for _, _, w in spans]]
    return vals


def unit_starts(text, unit):
    """Char start of every unit, aligned with the output of units()."""
    if unit == "untok":
        return np.array([m.start() for m in re.finditer(r"\S+", text)])
    spans = [(m.start(), m.group()[0].isalnum() or m.group()[0] == "_") for m in UNIT_RE.finditer(text)]
    return np.array([a for a, w in spans if unit != "nopunct" or w])


def split_sentences(s, starts_units, sent_starts):
    """Split unit surprisals into per-sentence arrays given sentence char starts."""
    k = np.searchsorted(sent_starts, starts_units, side="right") - 1
    return [s[k == i] for i in np.unique(k)]


def gini(x):
    if len(x) == 0 or x.mean() == 0:
        return np.nan
    xs = np.sort(x)
    n = len(xs)
    return (2 * np.arange(1, n + 1) - n - 1).dot(xs) / (n * xs.sum())


def measures(s):
    n = len(s)
    if n < 2:
        return {m: np.nan for m in MEASURES} | {"n": n}
    mu, sd = s.mean(), s.std()
    lv = np.mean(np.diff(s) ** 2)
    return {"n": n, "mean": mu, "total": s.sum(), "sd": sd, "cv": sd / mu if mu > 0 else np.nan,
            "gini": gini(s), "lv": lv, "lv_norm": lv / mu ** 2 if mu > 0 else np.nan,
            "gv": np.nan, "sl": np.mean(s ** 1.25), "max": s.max(), "_s": s}


def sentence_measures(parts):
    """parts: per-sentence unit surprisals with the first unit of each sentence already dropped."""
    good = [p for p in parts if len(p) >= 2]
    pooled = np.concatenate(parts) if parts else np.array([])
    if not good:
        return {m: np.nan for m in MEASURES} | {"n": 0}
    per = [measures(p) for p in good]
    w = np.array([len(p) for p in good], dtype=float)
    avg = lambda k: np.average([r[k] for r in per], weights=w)
    return {"n": np.mean([len(p) for p in parts]), "mean": pooled.mean(), "total": pooled.sum(), "sd": avg("sd"),
            "cv": avg("cv"), "gini": avg("gini"), "lv": avg("lv"), "lv_norm": avg("lv_norm"), "gv": np.nan,
            "sl": avg("sl"), "max": pooled.max(), "_s": pooled}


def compute(texts_tsv, surp_jsonl, unit="untok", drop_first=True, index="text", sents=None):
    """Return DataFrame with one column per measure, indexed by normalised text or by id (`index`).

    sents: optional dict text -> sentence char starts; switches to the sentence-level measures."""
    texts = pd.read_csv(texts_tsv, sep="\t", dtype={"id": str}, keep_default_na=False, quoting=3)
    id2text = dict(zip(texts["id"], texts["text"]))
    rows = []
    with open(surp_jsonl, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            t = id2text[r["id"]]
            s = units(t, r["offsets"], r["surp"], unit)
            if sents is not None and len(s):
                parts = split_sentences(s, unit_starts(t, unit), np.array(sents[t]))
                m = sentence_measures([p[1:] for p in parts])
            else:
                m = measures(s[1:] if drop_first else s)
            m["key"] = t if index == "text" else r["id"]
            rows.append(m)
    df = pd.DataFrame(rows).set_index("key")
    return add_global_variance(df)


def add_global_variance(df):
    allv = np.concatenate([s for s in df["_s"] if isinstance(s, np.ndarray)])
    mu = allv.mean()
    df["gv"] = [np.mean((s - mu) ** 2) if isinstance(s, np.ndarray) else np.nan for s in df["_s"]]
    return df.drop(columns="_s")
