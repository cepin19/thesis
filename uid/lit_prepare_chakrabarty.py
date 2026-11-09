#!/usr/bin/env python3
r"""Chakrabarty et al. 2021 poetry test sets (testdatawithtranslations/<lang>/): gold + published system outputs -> common format.

Line-aligned files with no poem boundaries, so consecutive lines are grouped into pseudo-stanzas of 8 lines (remainder dropped).
A stanza can therefore span two poems; the line-level measures are the safer ones here. Keeps only systems present in all languages.
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "data/literary/chakrabarty"
T = ROOT / "testdatawithtranslations"
K = 8


def rd(p):
    return [" ".join(l.replace("\t", " ").split()) for l in open(p, encoding="utf-8", errors="replace").read().split("\n")]


langs = sorted(p.name for p in T.iterdir() if p.is_dir() and p.name != "ukranian")  # no poetry-tuned output for Ukrainian
sys_of = {}
for l in langs:
    sys_of[l] = {f.name[len(l):-4] for f in (T / l).glob(f"{l}*.txt")}
common = sorted(set.intersection(*sys_of.values()))
print("systems per language:", {l: sorted(s) for l, s in sys_of.items()}, "\ncommon:", common)
src, hum, tr = [], [], {c: [] for c in common}
for l in langs:
    s, g = rd(T / l / f"test{l}.txt"), rd(T / l / f"gold{l}.txt")
    while s and not s[-1]:
        s.pop()
    n = len(s) - len(s) % K
    sysl = {c: rd(T / l / f"{l}{c}.txt") for c in common}
    if len(g) < n or any(len(v) < n for v in sysl.values()):
        print("SKIP misaligned", l, len(s), len(g), {c: len(v) for c, v in sysl.items()})
        continue
    for i in range(0, n, K):
        sid = f"{l}_{i // K:04d}"
        src.append((sid, "\\n".join(s[i:i + K])))
        hum.append((sid, "h0", "\\n".join(g[i:i + K])))
        for c in common:
            tr[c].append((sid, "\\n".join(sysl[c][i:i + K])))
pd.DataFrame(src, columns=["src_id", "text"]).to_csv(ROOT / "sources.tsv", sep="\t", index=False, quoting=3)
pd.DataFrame(hum, columns=["src_id", "ref", "text"]).to_csv(ROOT / "human.tsv", sep="\t", index=False, quoting=3)
for c, v in tr.items():
    c2 = re.sub(r"\W", "", c)
    pd.DataFrame(v, columns=["src_id", "text"]).to_csv(ROOT / f"trans.{c2}.tsv", sep="\t", index=False, quoting=3)
print(len(src), "stanzas")
