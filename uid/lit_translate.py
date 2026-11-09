#!/usr/bin/env python3
r"""Translate verse line by line, keeping the line structure, with NLLB or Marian (OPUS-MT).

Input: TSV with src_id, text (lines separated by a literal \n). Output: TSV with src_id, text, same line count.
--lower lowercases the output (for corpora whose human side is lowercased).
"""
import argparse

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--src-lang", help="NLLB source code, e.g. eng_Latn")
    ap.add_argument("--tgt-lang", help="NLLB target code, e.g. fra_Latn")
    ap.add_argument("--lower", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, **({"src_lang": args.src_lang} if args.src_lang else {}))
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()
    gen = {"num_beams": args.beams, "max_length": 256, "do_sample": False}
    if args.tgt_lang:
        gen["forced_bos_token_id"] = tok.convert_tokens_to_ids(args.tgt_lang)
    df = pd.read_csv(args.input, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    uniq = sorted({l.strip() for t in df["text"] for l in t.split("\\n") if l.strip()}, key=len)
    out = {"": ""}
    for b in range(0, len(uniq), args.batch_size):
        chunk = uniq[b:b + args.batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            ids = model.generate(**enc, **gen)
        out.update(zip(chunk, tok.batch_decode(ids, skip_special_tokens=True)))
        print(f"{b + len(chunk)}/{len(uniq)}", flush=True)
    res = []
    for t in df["text"]:
        s = "\\n".join(" ".join(out[l.strip()].split()) for l in t.split("\\n"))
        res.append(s.lower() if args.lower else s)
    pd.DataFrame({"src_id": df["src_id"], "text": res}).to_csv(args.output, sep="\t", index=False, quoting=3)


if __name__ == "__main__":
    main()
