#!/usr/bin/env python3
"""Split texts.{cs,en}.tsv into UDPipe input chunks (`split`) and read the parsed chunks back (`load`).

load(lang) -> dict id -> list of token dicts (form, lemma, upos, feats, head, deprel); multiword-token
lines are skipped, so tokens are syntactic words.

Two parses: udpipe/ (one segment = one tree, --tokenizer=presegmented) and udpipe_sent/ (UDPipe sentence
splitter; segments are separated by blank lines, so every segment starts a new paragraph). Tokens carry `sent`.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/wmt_human"
UD = DATA / "udpipe"
NCHUNK = {"cs": 20, "en": 4}


def read_texts(lang):
    return pd.read_csv(DATA / f"texts.{lang}.tsv", sep="\t", dtype={"id": str}, keep_default_na=False, quoting=3)


def split(sdir="udpipe", sep="\n"):
    d = DATA / sdir
    d.mkdir(exist_ok=True)
    for lang, k in NCHUNK.items():
        df = read_texts(lang)
        for c, part in enumerate(np.array_split(np.arange(len(df)), k)):
            (d / f"{lang}.{c}.ids").write_text("\n".join(df["id"].iloc[part]) + "\n")
            (d / f"{lang}.{c}.txt").write_text(sep.join(df["text"].iloc[part].str.strip()) + "\n")


def parse_conllu(path, by_par=False):
    """Return a list of segments, each a flat token list; `sent` = sentence index within the segment.
    by_par=False: every sentence is a segment. Also returns sentence texts per segment."""
    segs, texts, cur, sent = [], [], [], -1

    def close():
        nonlocal cur, sent
        segs.append(cur)
        cur, sent = [], -1

    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("# newpar") and by_par:
            if sent >= 0:
                close()
            texts.append([])
        elif line.startswith("# text = "):
            if not by_par:
                if sent >= 0:
                    close()
                texts.append([])
            sent += 1
            texts[-1].append(line[9:])
        elif line and not line.startswith("#"):
            f = line.split("\t")
            if "-" in f[0] or "." in f[0]:
                continue
            cur.append({"id": int(f[0]), "form": f[1], "lemma": f[2], "upos": f[3], "feats": f[5],
                        "head": int(f[6]), "deprel": f[7], "sent": sent})
    if sent >= 0:
        close()
    return segs, texts


def load(lang, sdir="udpipe"):
    out = {}
    for c in range(NCHUNK[lang]):
        ids = (DATA / sdir / f"{lang}.{c}.ids").read_text().split("\n")[:-1]
        segs, _ = parse_conllu(DATA / sdir / f"{lang}.{c}.conllu", by_par=sdir != "udpipe")
        assert len(ids) == len(segs), (lang, c, len(ids), len(segs))
        out.update(zip(ids, segs))
    return out


def sentence_starts(lang, sdir="udpipe_sent"):
    """text -> char offsets of its sentences. Offsets are recovered by counting non-space characters, because
    UDPipe's `# text` may normalise whitespace."""
    id2text = dict(zip(*read_texts(lang)[["id", "text"]].values.T))
    out = {}
    for c in range(NCHUNK[lang]):
        ids = (DATA / sdir / f"{lang}.{c}.ids").read_text().split("\n")[:-1]
        _, texts = parse_conllu(DATA / sdir / f"{lang}.{c}.conllu", by_par=True)
        assert len(ids) == len(texts), (lang, c, len(ids), len(texts))
        for i, sents in zip(ids, texts):
            t = id2text[i]
            pos = [p for p, ch in enumerate(t) if not ch.isspace()]
            starts, k = [], 0
            for x in sents:
                starts.append(pos[min(k, len(pos) - 1)])
                k += sum(not ch.isspace() for ch in x)
            starts[0] = 0
            out[t] = starts
    return out


if __name__ == "__main__":
    if sys.argv[1] == "split":
        split()
    elif sys.argv[1] == "split-sent":
        split("udpipe_sent", "\n\n")
