#!/usr/bin/env python3
r"""Corpus of Czech Verse (original Czech poems, not translations): sample poems for surprisal scoring.

Sample: at most 5 poems per book, 8-40 lines, duplicates (re-editions) dropped, seed 1, 3000 poems in total.
Writes data/literary/ccv/texts_tgt.tsv (id, text; lines joined by a literal \n) and meta.jsonl
(id, year, author, rhyme: per line first / response / none).
"""
import json
import random
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "data/literary/ccv"
rng = random.Random(1)
books = sorted((D / "ccv").glob("*.json"))
pool, seen = [], set()
for p in books:
    poems = json.load(open(p, encoding="utf-8"))
    rng.shuffle(poems)
    k = 0
    for x in poems:
        lines = [l for st in x["body"] for l in st]
        if not 8 <= len(lines) <= 40:
            continue
        texts = [" ".join(l["text"].replace("\t", " ").split()) for l in lines]
        if not all(texts):
            continue
        t = "\\n".join(texts)
        if t in seen:
            continue
        seen.add(t)
        got, rh = set(), []
        for l in lines:
            r = l.get("rhyme")
            if not r:
                rh.append("none")
            elif r in got:
                rh.append("response")
            else:
                got.add(r)
                rh.append("first")
        # a rhyme index with one line only is not a rhyme
        counts = {}
        for l in lines:
            if l.get("rhyme"):
                counts[l["rhyme"]] = counts.get(l["rhyme"], 0) + 1
        rh = ["none" if l.get("rhyme") and counts[l["rhyme"]] < 2 else c for l, c in zip(lines, rh)]
        pool.append(dict(text=t, year=x["biblio"].get("year"), author=(x.get("p_author") or {}).get("identity"), rhyme=rh))
        k += 1
        if k == 5:
            break
rng.shuffle(pool)
pool = pool[:3000]
with open(D / "texts_tgt.tsv", "w", encoding="utf-8") as f, open(D / "meta.jsonl", "w", encoding="utf-8") as m:
    f.write("id\ttext\n")
    for i, r in enumerate(pool):
        f.write(f"c{i:05d}\t{r['text']}\n")
        m.write(json.dumps(dict(id=f"c{i:05d}", year=r["year"], author=r["author"], rhyme=r["rhyme"]), ensure_ascii=False) + "\n")
print(len(pool), "poems from", len(books), "books")
