#!/usr/bin/env python3
r"""Reference-free quality score of every translation in a cond_in.tsv (lit_cond_inputs.py): CometKiwi or XCOMET-XL.

usage: score_quality.py <cond_in.tsv> <output.tsv> {kiwi,xcomet} [--limit N]
Output: id, score, and for xcomet n_major (error spans of major or critical severity). Literal \n becomes a space.
Run in wmt25/env. Input texts longer than the model's limit are truncated by COMET.
"""
import argparse
import csv
import os
import re
import sys

from comet import download_model, load_from_checkpoint

MODELS = {"kiwi": "Unbabel/wmt22-cometkiwi-da", "xcomet": "Unbabel/XCOMET-XL"}
PROMPT = re.compile(r"^[^:]+: (.*) \\n [^:]+:$", re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("metric", choices=MODELS)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()
    csv.field_size_limit(10 ** 9)
    with open(a.input, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    rows = rows[:a.limit] if a.limit else rows
    data = []
    for r in rows:
        src = PROMPT.match(r["context"]).group(1)
        data.append({"src": src.replace("\\n", " "), "mt": r["text"].replace("\\n", " ")})
    model = load_from_checkpoint(download_model(MODELS[a.metric]))
    pred = model.predict(data, batch_size=a.batch_size, gpus=1)
    spans = None
    md = getattr(pred, "metadata", None)
    if md is not None:
        spans = getattr(md, "error_spans", None)
        if spans is None and isinstance(md, dict):
            spans = md.get("error_spans")
    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    with open(a.output, "w", encoding="utf-8") as f:
        f.write("id\tscore\tn_major\n")
        for i, (r, s) in enumerate(zip(rows, pred.scores)):
            n = "" if spans is None else sum(x.get("severity") in ("major", "critical") for x in spans[i])
            f.write(f"{r['id']}\t{s:.5f}\t{n}\n")
    print(len(rows), "scored; spans:", spans is not None, file=sys.stderr)


if __name__ == "__main__":
    main()
