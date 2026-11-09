#!/usr/bin/env python3
"""Build segment-level EN->CS tables with human DA scores for WMT20, WMT21 and WMT22.

Output (one TSV per year, data/wmt_human/wmt{YY}.en-cs.tsv), one row per (system, segment):
    year, doc_id, seg_idx (1-based within doc), line (0-based in the plain-text test set or -1),
    system, is_human, src, hyp, da_raw, da_z, n_judgments

Human translations are scored as ordinary systems in all three years
(WMT20: HUMAN = reference; WMT21: translator-A/B; WMT22: translator-B/C), so the human
translations get the same kind of quality score as MT. Segments with no human judgment are
kept with NaN scores (useful for unfiltered comparisons).

WMT21/22 Appraise judgments are standardised per annotator (z-score over that annotator's
TGT items, as in WMT findings), then averaged per (system, doc, seg). Quality-control items
(itemType BAD/REF) and document-level scores are dropped.
"""
import argparse
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

COLS = ["year", "doc_id", "seg_idx", "line", "system", "is_human", "src", "hyp",
        "da_raw", "da_z", "n_judgments"]


# ---------------------------------------------------------------- WMT20
def parse_sgm(path):
    """Return list of (doc_id, seg_id, text) in file order."""
    out, doc = [], None
    for line in open(path, encoding="utf-8"):
        m = re.search(r'<doc [^>]*docid="([^"]+)"', line)
        if m:
            doc = m.group(1)
            continue
        m = re.search(r'<seg id="?(\d+)"?>(.*)</seg>', line)
        if m:
            out.append((doc, int(m.group(1)), m.group(2)))
    return out


def unescape(s):
    return (s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
            .replace("&apos;", "'").replace("&amp;", "&"))


def build_wmt20():
    base = DATA / "wmt20-systems/wmt20-news-task-primary-submissions"
    src = parse_sgm(base / "sgm/sources/newstest2020-encs-src.en.sgm")
    txt = base / "txt"
    src_lines = (txt / "sources/newstest2020-encs-src.en.txt").read_text(encoding="utf-8").splitlines()
    assert len(src) == len(src_lines)
    key2line = {(d, s): i for i, (d, s, _) in enumerate(src)}

    hyps = {"HUMAN": (txt / "references/newstest2020-encs-ref.cs.txt").read_text(encoding="utf-8").splitlines()}
    for f in sorted((txt / "system-outputs/en-cs").glob("newstest2020.en-cs.*.txt")):
        name = f.name[len("newstest2020.en-cs."):-len(".txt")]
        hyps[name] = f.read_text(encoding="utf-8").splitlines()
    for k, v in hyps.items():
        assert len(v) == len(src_lines), k

    da = pd.read_csv(DATA / "wmt20-metrics/manual-evaluation/DA/ad-seg-scores-en-cs.csv", sep=r"\s+")
    da[["doc_id", "seg"]] = da["SID"].str.rsplit("_", n=1, expand=True)
    da["seg"] = da["seg"].astype(int)
    da = da[da["seg"] > 0]  # seg 0 is the document-level score
    scores = {(r.SYS, r.doc_id, r.seg): (r["RAW.SCR"], r["Z.SCR"], r.N) for _, r in da.iterrows()}
    missing = [k for k in scores if (k[1], k[2]) not in key2line or k[0] not in hyps]
    assert not missing, missing[:5]

    rows = []
    for sysname, lines in hyps.items():
        for i, (d, s, _) in enumerate(src):
            raw, z, n = scores.get((sysname, d, s), (np.nan, np.nan, 0))
            rows.append((2020, d, s, i, sysname, sysname == "HUMAN", unescape(src_lines[i]), lines[i], raw, z, n))
    return pd.DataFrame(rows, columns=COLS)


# ---------------------------------------------------------------- WMT21 / WMT22
def parse_wmt_xml(path):
    """Return dict system -> {(doc_id, seg_idx): text}, plus 'src'. Refs named translator-X."""
    root = ET.parse(path).getroot()
    out = defaultdict(dict)
    order = []
    for doc in root.iter("doc"):
        if doc.get("testsuite"):
            continue
        did = doc.get("id")
        for child in doc:
            if child.tag == "src":
                name = "src"
            elif child.tag == "ref":
                name = "translator-" + child.get("translator")
            elif child.tag == "hyp":
                name = child.get("system")
            else:
                continue
            for seg in child.iter("seg"):
                key = (did, int(seg.get("id")))
                out[name][key] = seg.text or ""
                if name == "src":
                    order.append(key)
    return out, order


def appraise_scores(df, seg_offset):
    """Per-annotator z-scores, averaged per (system, doc, seg)."""
    df = df[(df["itemType"] == "TGT") & (df["isDocScore"].astype(str) != "True")].copy()
    g = df.groupby("username")["score"]
    df["z"] = (df["score"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    df["seg_idx"] = df["itemId"].astype(int) + seg_offset
    agg = df.groupby(["system", "docId", "seg_idx"]).agg(
        da_raw=("score", "mean"), da_z=("z", "mean"), n_judgments=("score", "size"))
    return agg


def build_from_xml(year, xml_path, scores, sys_rename=lambda s: s):
    texts, order = parse_wmt_xml(xml_path)
    line_of = {k: i for i, k in enumerate(order)}
    scores = scores.rename(index=sys_rename, level="system")
    unknown = set(scores.index.get_level_values("system")) - set(texts)
    assert not unknown, unknown
    rows = []
    for name, segs in texts.items():
        if name == "src":
            continue
        for key, hyp in segs.items():
            if key not in line_of:
                continue
            sk = (name,) + key
            raw, z, n = scores.loc[sk].tolist() if sk in scores.index else (np.nan, np.nan, 0)
            rows.append((year, key[0], key[1], line_of[key], name, name.startswith("translator-"),
                         texts["src"][key], hyp, raw, z, n))
    # every scored item must map to a text
    have = {(r[4], r[1], r[2]) for r in rows}
    lost = [k for k in scores.index if k not in have]
    assert not lost, lost[:5]
    return pd.DataFrame(rows, columns=COLS)


def build_wmt21():
    f = DATA / "wmt21-news-systems/humaneval/newstest2021-EX_and_XY/wmt21-regular.20210930.csv"
    cols = ["username", "system", "itemId", "itemType", "srcLang", "trgLang", "score", "docId",
            "isDocScore", "timeStart", "timeEnd"]
    df = pd.read_csv(f, names=cols, header=None)
    df = df[(df.srcLang == "eng") & (df.trgLang == "ces")]
    # itemId is 0-based in WMT21 (doc-level rows carry itemId == n_segs, removed via isDocScore)
    return build_from_xml(2021, DATA / "wmt21-news-systems/xml/newstest2021.en-cs.all.xml",
                          appraise_scores(df, seg_offset=1))


def build_wmt22():
    d = DATA / "wmt22-news-systems/humaneval/DA+SQM/scores"
    cols = open(d / "header.DocLvl.csv").read().strip().split(",")
    df = pd.read_csv(d / "WMT22.Appraise.DocLvl.20221107.filtered.csv", names=cols, header=None)
    df = df[(df.srcLang == "eng") & (df.trgLang == "ces")]
    # itemId is 1-based in WMT22
    return build_from_xml(2022, DATA / "wmt22-news-systems/xml/wmttest2022.en-cs.all.xml",
                          appraise_scores(df, seg_offset=0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DATA / "wmt_human")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for year, fn in [(2020, build_wmt20), (2021, build_wmt21), (2022, build_wmt22)]:
        df = fn()
        out = args.out_dir / f"wmt{year % 100}.en-cs.tsv"
        df.to_csv(out, sep="\t", index=False)
        scored = df[df.n_judgments > 0]
        print(f"{year}: {len(df)} rows, {df.system.nunique()} systems "
              f"({sorted(df[df.is_human].system.unique())} human), "
              f"{len(scored)} scored rows, {scored.groupby(['doc_id', 'seg_idx']).ngroups} scored segments -> {out}")
        print(scored.groupby("system").da_z.agg(["mean", "size"]).sort_values("mean", ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
