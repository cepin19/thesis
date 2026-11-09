#!/usr/bin/env python3
r"""Inputs for conditional surprisal under external LMs: the source as a prompt, target tokens scored.

Template (user's specification): "{source_language}: {source_text} \n {target_language}: {target_text}"
context = "{source_language}: {source_text} \n {target_language}:"; surprisal.py joins context and text with one space.
Line breaks are written as a literal \n and restored by surprisal.py --unescape-nl.
id = "<src_id>|<md5 of target text>" (cond_id). Output: <dir>/cond_in.tsv with id, context, text.
Sets: WMT20-22 EN->CS (human + all systems), NLLB outputs not already in WMT, Mendeley EN->FR (line and flat), Chakrabarty, PAR3.
"""
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PAR3_LANG = dict(fr="French", ru="Russian", de="German", ja="Japanese", zh="Chinese", no="Norwegian", cs="Czech", pt="Portuguese", sv="Swedish",
                 it="Italian", es="Spanish", hu="Hungarian", ta="Tamil", fa="Persian", nl="Dutch", pl="Polish")
CHAK_LANG = dict(dutch="Dutch", german="German", italian="Italian", portugese="Portuguese", romanian="Romanian", russian="Russian", spanish="Spanish",
                 swedish="Swedish")


def cond_id(src_id, text):
    return f"{src_id}|{hashlib.md5(text.encode()).hexdigest()}"


def write(path, rows):
    """rows: (src_id, source language, source text, target language, target text); texts single-line with literal \\n."""
    seen, out = set(), []
    for sid, sl, st, tl, tt in rows:
        i = cond_id(sid, tt)
        if i not in seen and tt.strip():
            seen.add(i)
            out.append((i, f"{sl}: {st} \\n {tl}:", tt))
    pd.DataFrame(out, columns=["id", "context", "text"]).to_csv(path, sep="\t", index=False, quoting=3)
    print(path, len(out))
    return seen


def main():
    import measures as M
    import nllb_prepare as P
    import udpipe_io as U

    wmt = pd.concat([pd.read_csv(U.DATA / f"wmt{y}.en-cs.tsv", sep="\t", keep_default_na=False) for y in (20, 21, 22)])
    en = U.read_texts("en")
    s2i = dict(zip(en["text"], en["id"]))
    i2s = dict(zip(en["id"], en["text"]))
    rows = [(s2i[M.norm_text(s)], "English", M.norm_text(s), "Czech", M.norm_text(h)) for s, h in zip(wmt["src"], wmt["hyp"])]
    seen = write(U.DATA / "cond_in.tsv", rows)
    new = []
    for c in P.CONDS:
        t = P.read_trans(c)
        new += [(i, "English", i2s[i], "Czech", x) for i, x in zip(t["id"], t["text"]) if x and cond_id(i, x) not in seen]
    write(P.ND / "cond_in.tsv", new)

    rd = lambda p: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    for name, lang, tgt in [("mendeley", lambda s: "English", "French"), ("mendeley_flat", lambda s: "English", "French"),
                            ("chakrabarty", lambda s: CHAK_LANG[s.split("_")[0]], "English"), ("par3", None, "English")]:
        D = ROOT / "data/literary" / name
        src = dict(zip(*rd(D / "sources.tsv")[["src_id", "text"]].values.T))
        if name == "par3":
            books = rd(D / "books.tsv")
            m = dict(zip(books["src_id"], books["lang"]))
            lang = lambda s: PAR3_LANG[m[s]]
        parts = [rd(D / "human.tsv")] + [rd(p) for p in sorted(D.glob("trans.*.tsv"))]
        write(D / "cond_in.tsv", [(s, lang(s), src[s], tgt, x) for t in parts for s, x in zip(t["src_id"], t["text"])])


if __name__ == "__main__":
    main()
