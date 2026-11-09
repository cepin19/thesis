#!/usr/bin/env python3
"""Translate the unique WMT20-22 English sources (data/wmt_human/texts.en.tsv) with NLLB-200 distilled 600M.

--beams k: beam search (k = 1 is greedy); --sample: ancestral sampling from the model (seed 1).
HF defaults kept and written explicitly: length_penalty 1.0 (score divided by length), early_stopping False, no repeat
constraints. Output: data/nllb/trans.<cond>.tsv with columns id, text, and data/nllb/ids.<cond>.jsonl with the generated token
ids ({"id", "text", "ids"}, without decoder start, language token, </s> and padding). --ids-only writes only the latter.
"""
import argparse
import json
import sys

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import measures as M
import udpipe_io as U

MODEL = "facebook/nllb-200-distilled-600M"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ids-only", action="store_true")
    args = ap.parse_args()
    cond = "sample" if args.sample else f"beam{args.beams}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device).eval()
    bos = tok.convert_tokens_to_ids("ces_Latn")
    torch.manual_seed(1)
    df = U.read_texts("en")
    if args.limit:
        df = df.head(args.limit)
    src = df["text"].tolist()
    order = sorted(range(len(src)), key=lambda i: len(src[i]))
    out = [None] * len(src)
    out_ids = [None] * len(src)
    for b in range(0, len(order), args.batch_size):
        idx = order[b:b + args.batch_size]
        enc = tok([src[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        kw = dict(do_sample=True, top_k=0, top_p=1.0, num_beams=1) if args.sample else \
            dict(do_sample=False, num_beams=args.beams, length_penalty=1.0, early_stopping=False)
        g = model.generate(**enc, forced_bos_token_id=bos, max_length=512, **kw)
        texts = tok.batch_decode(g, skip_special_tokens=True)
        for j, i in enumerate(idx):
            out[i] = M.norm_text(texts[j]).strip()
            seq = g[j, 2:].tolist()  # drop decoder start and language token
            out_ids[i] = seq[:seq.index(tok.eos_token_id)] if tok.eos_token_id in seq else seq
        print(f"{min(b + args.batch_size, len(order))}/{len(order)}", file=sys.stderr, flush=True)
    nd = U.DATA.parent / "nllb"
    with open(nd / f"ids.{cond}.jsonl", "w", encoding="utf-8") as f:
        for i, t, x in zip(df["id"], out, out_ids):
            f.write(json.dumps({"id": i, "text": t, "ids": x}, ensure_ascii=False) + "\n")
    if not args.ids_only:
        path = args.out or nd / f"trans.{cond}.tsv"
        pd.DataFrame({"id": df["id"], "text": out}).to_csv(path, sep="\t", index=False, quoting=3)


if __name__ == "__main__":
    main()
