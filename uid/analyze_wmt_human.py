#!/usr/bin/env python3
"""MT vs. human translation surprisal uniformity on WMT20-22 EN->CS with human DA scores.

For every (LM, unit) specification:
  A  gap      : paired MT - HT difference per measure (segment = source sentence; HT = mean over the
                human references of that segment, MT = mean over MT systems), per year, with
                95% CI from a bootstrap over segments. Also per MT system.
  B  adjusted : within-segment (segment fixed effects) regression
                    y ~ is_mt + log(n) + mean_surprisal            (B1, all rows)
                    y ~ is_mt + log(n) + mean_surprisal + da_z     (B2, rows with human DA)
                coefficient of is_mt with cluster-bootstrap CI.
  C  tracking : Pearson r(source measure, target measure), MT (avg over systems) vs HT (avg over refs),
                raw and with log-length partialled out of both sides; CI on r_MT - r_HT.
  D  quality  : within-segment correlation between a measure and human DA (do raters prefer more
                uniform translations of the same source?), for MT rows, HT rows and all rows.
  E  system   : Spearman between system mean measure and system mean DA z (humans included).

Writes CSVs + summary.md to results/wmt_human/.
--context: surprisal given the preceding segments of the document (surpctx.*.jsonl).
--sent-split: per-sentence measures averaged over the segment (see measures.py); default is segment level.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import measures as M
import udpipe_io as U

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/wmt_human"
KEY = ["year", "doc_id", "seg_idx"]
REPORT = ["mean", "cv", "gini", "lv", "lv_norm", "gv", "sl", "n"]
RNG = np.random.default_rng(1)


def boot_ci(values_by_seg, stat=np.mean, B=2000):
    v = np.asarray(values_by_seg)
    v = v[~np.isnan(v)]
    if len(v) < 5:
        return np.nan, np.nan, np.nan
    idx = RNG.integers(0, len(v), (B, len(v)))
    bs = stat(v[idx], axis=1)
    return stat(v), *np.percentile(bs, [2.5, 97.5])


def load(lm, unit, min_da_raw=None, context=False, sent_split=False):
    tables = pd.concat([pd.read_csv(DATA / f"wmt{y}.en-cs.tsv", sep="\t", keep_default_na=False,
                                    na_values={"da_raw": [""], "da_z": [""]}) for y in (20, 21, 22)])
    for c in ("da_raw", "da_z"):
        tables[c] = pd.to_numeric(tables[c], errors="coerce")
    ss_cs = U.sentence_starts("cs") if sent_split else None
    ss_en = U.sentence_starts("en") if sent_split else None
    if context:
        tgt = M.compute(DATA / "texts_ctx.cs.tsv", DATA / f"surpctx.cs.{lm}.jsonl", unit=unit, index="id", sents=ss_cs)
        src = M.compute(DATA / "texts_ctx.en.tsv", DATA / "surpctx.en.eurollm9b.jsonl", unit=unit, index="id", sents=ss_en)
        ids = [str(y) + "|" + str(d) + "|" + str(k) for y, d, k in zip(tables["year"], tables["doc_id"], tables["seg_idx"])]
        tables["_src_key"] = ids
        tables["_hyp_key"] = [str(y) + "|" + str(s) + "|" + str(d) + "|" + str(k) for y, s, d, k in
                              zip(tables["year"], tables["system"], tables["doc_id"], tables["seg_idx"])]
    else:
        tgt = M.compute(DATA / "texts.cs.tsv", DATA / f"surp.cs.{lm}.jsonl", unit=unit, sents=ss_cs)
        src = M.compute(DATA / "texts.en.tsv", DATA / "surp.en.eurollm9b.jsonl", unit=unit, sents=ss_en)
        tables["_hyp_key"] = tables["hyp"].map(M.norm_text)
        tables["_src_key"] = tables["src"].map(M.norm_text)
    df = tables.join(tgt, on="_hyp_key").join(src.add_prefix("src_"), on="_src_key")
    df = df[df["n"] >= 3].copy()  # drops empty/one-word outputs
    if min_da_raw is not None:
        # keep only translations rated >= threshold, and only segments where both a human and an
        # MT translation survive, so every analysis runs on the same set of source segments
        df = df[df["da_raw"] >= min_da_raw]
        g = df.groupby(KEY)["is_human"]
        df = df[g.transform("any") & ~g.transform("all")].copy()
    df["is_mt"] = (~df["is_human"].astype(bool)).astype(float)
    df["seg"] = df.groupby(KEY, sort=False).ngroup()
    return df


# ---------------------------------------------------------------- A
def gap_table(df):
    out = []
    for year, d in df.groupby("year"):
        for m in REPORT:
            per = d.groupby(["seg", "is_mt"])[m].mean().unstack()
            if 0.0 not in per or 1.0 not in per:
                continue
            per = per.dropna()
            ht_sd = d.loc[d.is_mt == 0, m].std()
            est, lo, hi = boot_ci((per[1.0] - per[0.0]).values)
            out.append(dict(year=year, measure=m, system="ALL_MT", HT=per[0.0].mean(), MT=per[1.0].mean(),
                            gap=est, lo=lo, hi=hi, gap_in_HT_sd=est / ht_sd, n_seg=len(per)))
            ht = d[d.is_mt == 0].groupby("seg")[m].mean()
            for s, ds in d[d.is_mt == 1].groupby("system"):
                x = ds.groupby("seg")[m].mean()
                j = pd.concat([x, ht], axis=1, keys=["mt", "ht"]).dropna()
                est, lo, hi = boot_ci((j.mt - j.ht).values)
                out.append(dict(year=year, measure=m, system=s, HT=j.ht.mean(), MT=j.mt.mean(),
                                gap=est, lo=lo, hi=hi, gap_in_HT_sd=est / ht_sd, n_seg=len(j)))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- B
def fe_regression(d, y, xs, B=1000):
    """Within-segment OLS; returns coef of xs[0] and cluster-bootstrap CI."""
    d = d.dropna(subset=[y] + xs)
    d = d[d.groupby("seg")["seg"].transform("size") > 1]
    if d["seg"].nunique() < 20 or d[xs[0]].nunique() < 2:
        return np.nan, np.nan, np.nan, 0
    cols = [y] + xs
    dm = d[cols] - d.groupby("seg")[cols].transform("mean")
    X, Y = dm[xs].to_numpy(), dm[y].to_numpy()
    seg = d["seg"].to_numpy()
    useg, inv = np.unique(seg, return_inverse=True)
    k = len(xs)
    XtX = np.zeros((len(useg), k, k))
    XtY = np.zeros((len(useg), k))
    np.add.at(XtX, inv, X[:, :, None] * X[:, None, :])
    np.add.at(XtY, inv, X * Y[:, None])

    def solve(w):
        A = np.tensordot(w, XtX, 1)
        b = w @ XtY
        return np.linalg.lstsq(A, b, rcond=None)[0][0]

    est = solve(np.ones(len(useg)))
    bs = [solve(np.bincount(RNG.integers(0, len(useg), len(useg)), minlength=len(useg)).astype(float))
          for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi, len(useg)


def adjusted_table(df):
    df = df.copy()
    df["log_n"] = np.log(df["n"])
    out = []
    for year, d in df.groupby("year"):
        for m in REPORT:
            if m == "n":
                continue
            controls = [c for c in ["log_n", "mean"] if c != m]
            sd = d.loc[d.is_mt == 0, m].std()
            for name, xs, dd in [("B1_len_mean", ["is_mt"] + controls, d),
                                 ("B2_len_mean_da", ["is_mt"] + controls + ["da_z"], d[d.da_z.notna()])]:
                est, lo, hi, ns = fe_regression(dd, m, xs)
                out.append(dict(year=year, measure=m, model=name, mt_minus_ht=est, lo=lo, hi=hi,
                                in_HT_sd=est / sd, n_seg=ns))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- C
def resid(y, x):
    ok = ~(np.isnan(y) | np.isnan(x))
    r = np.full_like(y, np.nan)
    A = np.c_[np.ones(ok.sum()), x[ok]]
    r[ok] = y[ok] - A @ np.linalg.lstsq(A, y[ok], rcond=None)[0]
    return r


def tracking_table(df, B=2000):
    out = []
    for year, d in df.groupby("year"):
        for m in [x for x in REPORT if x != "n"]:
            d = d.copy()
            d["t_res"] = resid(d[m].to_numpy(float), np.log(d["n"].to_numpy(float)))
            d["s_res"] = resid(d["src_" + m].to_numpy(float), np.log(d["src_n"].to_numpy(float)))
            for variant, tcol, scol in [("raw", m, "src_" + m), ("len_partial", "t_res", "s_res")]:
                # r per system, then averaged within group; bootstrap over segments
                piv_t = d.pivot_table(index="seg", columns="system", values=tcol)
                srcv = d.groupby("seg")[scol].first().reindex(piv_t.index).to_numpy()
                T = piv_t.to_numpy()
                is_mt = np.array([not d.loc[d.system == s, "is_human"].iloc[0] for s in piv_t.columns])

                def r_cols(rows):
                    rs = []
                    for j in range(T.shape[1]):
                        t, s = T[rows, j], srcv[rows]
                        ok = ~(np.isnan(t) | np.isnan(s))
                        rs.append(np.corrcoef(t[ok], s[ok])[0, 1] if ok.sum() > 10 else np.nan)
                    rs = np.array(rs)
                    return np.nanmean(rs[is_mt]), np.nanmean(rs[~is_mt])

                rows = np.arange(len(srcv))
                r_mt, r_ht = r_cols(rows)
                bs = np.array([r_cols(RNG.integers(0, len(rows), len(rows))) for _ in range(B // 4)])
                diff = bs[:, 0] - bs[:, 1]
                out.append(dict(year=year, measure=m, variant=variant, r_MT=r_mt, r_HT=r_ht,
                                diff=r_mt - r_ht, lo=np.percentile(diff, 2.5), hi=np.percentile(diff, 97.5)))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- D
def quality_table(df):
    """Within-segment r(measure, DA z); `partial` additionally removes (within-segment) log length
    and mean surprisal from the measure, since CV = sd/mean rises mechanically as mean falls and
    mean surprisal itself is strongly related to DA."""
    out = []
    d0 = df[df.da_z.notna()].copy()
    d0["log_n"] = np.log(d0["n"])
    for year, d in d0.groupby("year"):
        for m in REPORT:
            controls = [c for c in ["log_n", "mean"] if c != m and not (m == "n" and c == "log_n")]
            for grp, dd in [("MT", d[d.is_mt == 1]), ("HT", d[d.is_mt == 0]), ("all", d)]:
                dd = dd[dd.groupby("seg")["seg"].transform("size") > 1].dropna(subset=[m, "da_z"] + controls)
                if len(dd) <= 10:
                    continue
                cols = [m, "da_z"] + controls
                dm = dd[cols] - dd.groupby("seg")[cols].transform("mean")
                xv, yv = dm[m].to_numpy(), dm["da_z"].to_numpy()
                C = dm[controls].to_numpy()
                xp = xv - C @ np.linalg.lstsq(C, xv, rcond=None)[0]
                us, inv = np.unique(dd["seg"].to_numpy(), return_inverse=True)
                for variant, xx in [("plain", xv), ("partial", xp)]:
                    r = np.corrcoef(xx, yv)[0, 1]
                    bs = []
                    for _ in range(500):
                        w = np.bincount(RNG.integers(0, len(us), len(us)), minlength=len(us))[inv]
                        sel = np.repeat(np.arange(len(xx)), w)
                        bs.append(np.corrcoef(xx[sel], yv[sel])[0, 1])
                    lo, hi = np.percentile(bs, [2.5, 97.5])
                    out.append(dict(year=year, measure=m, group=grp, variant=variant, r_within_seg=r,
                                    lo=lo, hi=hi, n=len(xx)))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- E
def system_table(df):
    out, means = [], []
    d0 = df[df.da_z.notna()]
    for year, d in d0.groupby("year"):
        g = d.groupby("system").agg(**{m: (m, "mean") for m in REPORT}, da_z=("da_z", "mean"),
                                    is_human=("is_human", "first"))
        g["year"] = year
        means.append(g.reset_index())
        for m in REPORT:
            rho, p = spearmanr(g[m], g["da_z"])
            rho_mt, p_mt = spearmanr(g.loc[~g.is_human.astype(bool), m], g.loc[~g.is_human.astype(bool), "da_z"])
            out.append(dict(year=year, measure=m, rho_all=rho, p_all=p, rho_mt_only=rho_mt, p_mt_only=p_mt,
                            n_sys=len(g)))
    return pd.DataFrame(out), pd.concat(means)


def fmt(df):
    return df.to_markdown(index=False, floatfmt=".3f")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lms", nargs="+", default=["czgpt2xl", "eurollm9b", "customgpt2"])
    ap.add_argument("--units", nargs="+", default=["untok", "tok", "nopunct"])
    ap.add_argument("--primary", default="czgpt2xl:untok")
    ap.add_argument("--out", type=Path, default=ROOT / "results/wmt_human")
    ap.add_argument("--min-da-raw", type=float, default=None,
                    help="keep only translations with mean raw DA >= this (and segments with both HT and MT left)")
    ap.add_argument("--context", action="store_true", help="surprisal given the preceding document segments")
    ap.add_argument("--sent-split", action="store_true", help="per-sentence measures instead of segment level")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    allres = {k: [] for k in "ABCDE"}
    sysmeans = []
    for lm in args.lms:
        for unit in args.units:
            spec = f"{lm}:{unit}"
            print("==", spec, flush=True)
            df = load(lm, unit, args.min_da_raw, args.context, args.sent_split)
            res = {"A": gap_table(df), "B": adjusted_table(df), "C": tracking_table(df), "D": quality_table(df)}
            res["E"], sm = system_table(df)
            sm["spec"] = spec
            sysmeans.append(sm)
            for k, v in res.items():
                v.insert(0, "spec", spec)
                allres[k].append(v)
    names = {"A": "gap", "B": "adjusted", "C": "tracking", "D": "quality", "E": "system"}
    tabs = {k: pd.concat(v) for k, v in allres.items()}
    for k, t in tabs.items():
        t.to_csv(args.out / f"{names[k]}.csv", index=False)
    pd.concat(sysmeans).to_csv(args.out / "system_means.csv", index=False)

    # ------------------------------------------------------------ summary
    P = args.primary
    key = ["mean", "cv", "gini", "lv", "lv_norm", "gv"]
    A, Bt, C, D, E = (tabs[k] for k in "ABCDE")
    filt = f", only translations with raw DA >= {args.min_da_raw}" if args.min_da_raw is not None else ""
    lines = [f"# WMT20-22 EN->CS: MT vs HT surprisal uniformity (primary spec `{P}`{filt})\n",
             "Positive gap = MT higher (for cv/gini/lv/gv: MT *less* uniform). 95% CIs: bootstrap over source segments.\n",
             "## A. Raw paired gap, all MT systems pooled vs human references\n",
             fmt(A[(A.spec == P) & (A.system == "ALL_MT") & A.measure.isin(key + ["n"])]
                 .drop(columns=["spec", "system"])),
             "\n## B. Within-segment regression, coefficient of is_MT (controls: log length, mean surprisal [, DA z])\n",
             fmt(Bt[(Bt.spec == P) & Bt.measure.isin(key)].drop(columns="spec")),
             "\n## C. Source tracking: r(src, tgt), MT vs HT\n",
             fmt(C[(C.spec == P) & C.measure.isin(key)].drop(columns="spec")),
             "\n## D. Within-segment correlation of measure with human DA z\n",
             "Partial = measure residualised on (within-segment) log length and mean surprisal.\n",
             fmt(D[(D.spec == P) & D.measure.isin(key)].assign(
                 cell=lambda x: x.r_within_seg.round(3).astype(str) + " [" + x.lo.round(2).astype(str)
                 + ", " + x.hi.round(2).astype(str) + "]").pivot_table(
                 index=["year", "measure"], columns=["variant", "group"], values="cell", aggfunc="first").reset_index()),
             "\n## E. System level: Spearman(system mean measure, system DA z)\n",
             fmt(E[(E.spec == P) & E.measure.isin(key)].drop(columns="spec")),
             "\n## Robustness across LM x unit specifications (adjusted gap B1, sign and significance)\n"]
    b1 = Bt[Bt.model == "B1_len_mean"].copy()
    b1["sig"] = np.where(b1.lo > 0, "+", np.where(b1.hi < 0, "-", "0"))
    rob = b1[b1.measure.isin(key)].pivot_table(index=["measure", "year"], columns="spec", values="sig",
                                                 aggfunc="first")
    lines.append(rob.to_markdown())
    lines.append("\n`+` MT significantly higher than HT, `-` significantly lower, `0` CI includes 0.\n")
    (args.out / "summary.md").write_text("\n".join(lines))
    print((args.out / "summary.md").read_text())


if __name__ == "__main__":
    main()
