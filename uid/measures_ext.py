"""Extended (non-)uniformity measures per text. Definitions and sources: MEASURES.md.

to_words() aggregates token values to word-like units; extended() computes all per-text measures (the first unit is dropped, as in
measures.py). Measures that need a long text return NaN below LONG_MIN units; typicality and calibrated measures need `ent` / `null`.
Surprisal-state measures need states fitted over a corpus: fit_states() and state_measures().
"""
import re

import numpy as np

import measures as M

MIN_UNITS = 5
LONG_MIN = 32
VOWELS = re.compile(r"[aeiouyáéíóúůýěäëïöüàâêîôûæœãõåø]+", re.I)
SENT_END = re.compile(r"[.!?…]+(?=[\s\"'”’»)\]]*(?:\s|$))")
WINDOWS = (1, 2, 4, 8, 16)
SL_K = (0.25, 0.5, 1.25, 1.5, 2.0)
K_STATES = 4
Z_MIN_SD = 1e-3


PM = False  # set by analyze_measures.py --pm: apply the Pimentel & Meister word-probability correction (needs bow_* from --bow scoring)


def pm_correction(text, rec, unit, keep, idx, n_units, unit_starts, unit_ends):
    """Per unit: log2 P_bow(before) - log2 P_bow(after), added to the summed token surprisal (bits). See bow.py.

    Where the word boundary sits is read off the tokens' character spans:
    before: if the unit's first token itself contains the whitespace before the unit (" word"), the model chose that boundary at this
      position: subtract log2 P_bow there. Otherwise the boundary was emitted by an earlier token (a newline token, a token such as
      ".\n\n" in BLOOM, or punctuation when punctuation splits words), or the unit starts the text: P_bow(before) = 1.
    after: if the unit's last token extends past the unit (it ends with whitespace, e.g. ".\n\n"), the boundary is inside that token:
      P_bow(after) = 1. Otherwise add -log2 P_bow at the next position (the end of the text is position n, where EOS counts).
    This keeps corrected word probabilities <= 1. `untok` uses whitespace word starts, `nopunct` also counts punctuation."""
    lb = np.asarray(rec["bow_ws" if unit == "untok" else "bow_wp"], dtype=float)
    off = rec["offsets"]
    keep = np.asarray(keep)
    first = np.full(n_units, len(lb), dtype=int)
    last = np.full(n_units, -1, dtype=int)
    np.minimum.at(first, idx, keep)
    np.maximum.at(last, idx, keep)
    out = np.zeros(n_units)
    for u in np.where(last >= 0)[0]:
        f, l = first[u], last[u]
        before = lb[f] if off[f][0] < unit_starts[u] else 0.0
        after = 0.0 if off[l][1] > unit_ends[u] else lb[min(l + 1, len(lb) - 1)]
        out[u] = before - after
    return out


def to_words(text, rec, unit="untok"):
    """Dict with s, ent, null (units x K), starts, words, line, sent; None if nothing is scored. Whitespace-only tokens are excluded.
    With PM, word surprisal gets the Pimentel & Meister correction and null samples are dropped (the correction cannot be applied to them)."""
    keep = [i for i, (a, b) in enumerate(rec["offsets"]) if text[a:b].strip()]
    if not keep:
        return None
    spans = [(m.start(), m.group()) for m in M.UNIT_RE.finditer(text)] if unit != "untok" else [(m.start(), m.group()) for m in re.finditer(r"\S+", text)]
    if not spans:
        return None
    all_starts = np.array([a for a, _ in spans])
    first = []
    for i in keep:
        a, b = rec["offsets"][i]
        while a < b and text[a].isspace():
            a += 1
        first.append(a)
    idx = np.maximum(np.searchsorted(all_starts, first, side="right") - 1, 0)
    end = max(rec["offsets"][i][1] for i in keep)
    n_all = len(spans)

    def add(values):
        v = np.asarray(values, dtype=np.float64)[keep]
        out = np.zeros((n_all,) + v.shape[1:])
        np.add.at(out, idx, v)
        return out

    mask = all_starts < end
    if unit == "nopunct":
        mask &= np.array([w[0].isalnum() or w[0] == "_" for _, w in spans])
    if not mask.any():
        return None
    corr = pm_correction(text, rec, unit, keep, idx, n_all, all_starts, all_starts + np.array([len(w) for _, w in spans])) if PM and "bow_ws" in rec else None
    starts = all_starts[mask]
    breaks = np.array([m.start() for m in re.finditer(r"\n", text)], dtype=int)
    ends = np.array([m.start() for m in SENT_END.finditer(text)], dtype=int)
    w = {"s": (add(rec["surp"]) + (corr if corr is not None else 0))[mask], "words": [g for g, k in zip((x for _, x in spans), mask) if k],
         "line": np.searchsorted(breaks, starts), "sent": np.searchsorted(ends, starts, side="left")}
    w["ent"] = add(rec["ent"])[mask] if "ent" in rec else None
    if "rank" in rec:  # token ranks of the kept units, without the first unit
        unit_pos = np.cumsum(mask) - 1
        w["rank"] = np.array([r for r, u in zip(np.asarray(rec["rank"], float)[keep], idx) if mask[u] and unit_pos[u] > 0])
    w["null"] = add(rec["null"])[mask] if "null" in rec and corr is None else None
    return w


def concat_words(parts):
    """Concatenate to_words() outputs of consecutive segments into one document; segments stay separate sentences."""
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    shift, sents, lines = 0, [], []
    for p in parts:
        sents.append(p["sent"] + shift)
        shift += int(p["sent"].max()) + 1
    return {"s": np.concatenate([p["s"] for p in parts]), "words": [w for p in parts for w in p["words"]],
            "line": np.zeros(sum(len(p["s"]) for p in parts), dtype=int), "sent": np.concatenate(sents),
            "ent": np.concatenate([p["ent"] for p in parts]) if all(p["ent"] is not None for p in parts) else None, "null": None}


def cut_words(w, n):
    keys = ["s", "line", "sent"] + [k for k in ("ent", "null") if w[k] is not None]
    return {k: v for k, v in w.items() if k != "rank"} | {k: w[k][:n] for k in keys} | {"words": w["words"][:n]}


def n_chars(words):
    return np.array([max(sum(c.isalnum() for c in x), 1) for x in words], dtype=float)


def n_syll(words):
    return np.array([max(len(VOWELS.findall(x)), 1) for x in words], dtype=float)


def acf(x, k):
    x = x - x.mean()
    d = (x * x).sum()
    return (x[:-k] * x[k:]).sum() / d if d > 0 and len(x) > k else np.nan


def _shape(s):
    z = s - s.mean()
    sd = s.std()
    if sd == 0:
        return np.nan, np.nan
    return (z ** 3).mean() / sd ** 3, (z ** 4).mean() / sd ** 4 - 3


def _spectral(s):
    if len(s) < LONG_MIN or s.std() == 0:
        return {}
    z = (s - s.mean()) / s.std()
    p = np.abs(np.fft.rfft(z))[1:] ** 2
    f = np.fft.rfftfreq(len(z))[1:]
    if p.sum() == 0:
        return {}
    q = p / p.sum()
    q0 = q[q > 0]
    good = p > 0
    return {"spec_centroid": (f * q).sum(), "spec_entropy": -(q0 * np.log(q0)).sum() / np.log(len(q)),
            "spec_slope": np.polyfit(np.log(f[good]), np.log(p[good]), 1)[0], "spec_high": q[f > 0.25].sum(), "fisher_g": q.max()}


def _harmonic(s, unit, chars):
    """R2 gain of a first-order sinusoid whose period is the enclosing unit (time scaling), over intercept + log characters."""
    if len(s) < LONG_MIN or s.std() == 0:
        return {}
    ids, inv, cnt = np.unique(unit, return_inverse=True, return_counts=True)
    if len(ids) < 3:
        return {}
    pos = np.zeros(len(s))
    seen = np.zeros(len(ids), dtype=int)
    for i, u in enumerate(inv):
        pos[i] = seen[u] / cnt[u]
        seen[u] += 1
    base = np.c_[np.ones(len(s)), np.log(chars)]
    full = np.c_[base, np.sin(2 * np.pi * pos), np.cos(2 * np.pi * pos)]

    def r2(X):
        b = np.linalg.lstsq(X, s, rcond=None)[0]
        return 1 - ((s - X @ b) ** 2).sum() / ((s - s.mean()) ** 2).sum()

    return {"harm_gain": r2(full) - r2(base)}


def _between(s, sent, line):
    out = {}
    ids = np.unique(sent)
    if len(ids) >= 3 and s.mean() > 0:
        means = np.array([s[sent == i].mean() for i in ids])
        out |= {"sent_var": means.var(), "sent_cv": means.std() / s.mean(), "burstiness": (2.0 ** np.minimum(means, 60)).var()}
    ids = np.unique(line)
    if len(ids) >= 3 and s.mean() > 0:
        means = np.array([s[line == i].mean() for i in ids])
        out["line_cv_between"] = means.std() / s.mean()
    if len(s) >= LONG_MIN:
        v = []
        for k in WINDOWS:
            m = len(s) // k
            v.append(s[:m * k].reshape(m, k).sum(1).var() if m >= 4 else np.nan)
        v = np.array(v)
        ok = np.isfinite(v) & (v > 0)
        if ok.sum() >= 3:
            out["vt_slope"] = np.polyfit(np.log(np.array(WINDOWS)[ok]), np.log(v[ok]), 1)[0]
            if np.isfinite(v[2]) and v[0] > 0:
                out["fano_w4"] = v[2] / (4 * v[0])
    return out


def _z(obs, null):
    null = null[np.isfinite(null)]
    # the samples are rounded to 0.01 bits: below this floor, near-deterministic texts give meaningless z-scores
    return (obs - null.mean()) / null.std() if len(null) >= 5 and null.std() >= Z_MIN_SD else np.nan


def extended(w):
    """All per-text measures for one to_words() output; None if too short."""
    if w is None or len(w["s"]) < MIN_UNITS + 1:
        return None
    s = w["s"][1:]
    words = w["words"][1:]
    line, sent = w["line"][1:], w["sent"][1:]
    mu, sd = s.mean(), s.std()
    if mu <= 0:
        return None
    lv = np.mean(np.diff(s) ** 2)
    sk, ku = _shape(s)
    r = {"n": len(s), "mean": mu, "sd": sd, "cv": sd / mu, "gini": M.gini(s), "lv": lv, "lv_norm": lv / mu ** 2, "lv_sq": lv ** 2, "lv_sq_norm": lv ** 2 / mu ** 4, "max": s.max(), "skew": sk, "kurt": ku}
    r |= {f"sl{k}": (s ** k).mean() for k in SL_K}
    # order vs distribution
    r |= {"vn_ratio": lv / sd ** 2 if sd > 0 else np.nan, "ac1": acf(s, 1), "ac2": acf(s, 2), "ac3": acf(s, 3)}
    # typicality
    if w["ent"] is not None:
        h = w["ent"][1:]
        r |= {"typ": np.abs(s - h).mean(), "typ_rel": np.abs(s - h).mean() / mu, "excess": (s - h).mean(), "ent_mean": h.mean()}
    # mode-seeking: does the model rank the chosen token first?
    if len(w.get("rank", [])):
        r |= {"top1": (w["rank"] == 1).mean(), "top5": (w["rank"] <= 5).mean(), "log2rank": np.log2(w["rank"]).mean()}
    # calibrated against samples from the model given the real prefix
    if w["null"] is not None:
        nl = w["null"][1:]
        nm, ns = nl.mean(0), nl.std(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r |= {"z_mean": _z(mu, nm), "z_cv": _z(sd / mu, ns / nm), "z_lv_norm": _z(lv / mu ** 2, (np.diff(nl, axis=0) ** 2).mean(0) / nm ** 2),
                  "z_gini": _z(r["gini"], np.array([M.gini(nl[:, j]) for j in range(nl.shape[1])])), "null_cv": (ns / nm).mean()}
    # per-unit cost
    c, y = n_chars(words), n_syll(words)
    bpc, bps = s / c, s / y
    r |= {"bpc_mean": bpc.mean(), "bpc_cv": bpc.std() / bpc.mean() if bpc.mean() > 0 else np.nan,
          "bps_mean": bps.mean(), "bps_cv": bps.std() / bps.mean() if bps.mean() > 0 else np.nan}
    # local dynamics
    d1 = np.diff(s)
    d2 = np.diff(d1)
    r |= {"d1_abs": np.abs(d1).mean(), "d1_var": d1.var()}
    if len(d2) >= 4:
        hist = np.histogram(d2, bins=10)[0] if d2.std() > 0 else np.array([len(d2)])
        q = hist[hist > 0] / hist.sum()
        r |= {"d2_var": d2.var(), "d2_abs": np.abs(d2).mean(), "d2_abs_rel": np.abs(d2).mean() / mu,
              "d2_entropy": -(q * np.log2(q)).sum(), "d2_ac1": acf(d2, 1)}
    r |= _spectral(s)
    r |= _harmonic(s, line if line.max() >= 2 else sent, c)
    r |= _between(s, sent, line)
    return r


def fit_states(values, k=K_STATES, iters=50, max_n=300000, seed=1):
    """1-D k-means on pooled word surprisals; returns sorted centres."""
    v = np.asarray(values, dtype=float)
    if len(v) > max_n:
        v = np.random.default_rng(seed).choice(v, max_n, replace=False)
    c = np.quantile(v, (np.arange(k) + 0.5) / k)
    for _ in range(iters):
        lab = np.abs(v[:, None] - c).argmin(1)
        new = np.array([v[lab == i].mean() if (lab == i).any() else c[i] for i in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return np.sort(c)


def state_measures(s, centres):
    """Transition features over states of s[1:]: top occupancy, spike P(top|bottom), recovery P(bottom|top), stay P(top|top), entropy rate."""
    if len(s) < MIN_UNITS + 1:
        return {}
    a = np.abs(s[1:, None] - centres).argmin(1)
    k = len(centres)
    t = np.zeros((k, k))
    np.add.at(t, (a[:-1], a[1:]), 1)
    row = t.sum(1)
    out = {"mk_top": (a == k - 1).mean()}
    if row[0] > 0:
        out["mk_spike"] = t[0, k - 1] / row[0]
    if row[k - 1] > 0:
        out["mk_recovery"] = t[k - 1, 0] / row[k - 1]
        out["mk_stay_top"] = t[k - 1, k - 1] / row[k - 1]
    h = 0.0
    for i in range(k):
        if row[i] > 0:
            p = t[i][t[i] > 0] / row[i]
            h -= row[i] / row.sum() * (p * np.log2(p)).sum()
    out["mk_entropy_rate"] = h
    return out
