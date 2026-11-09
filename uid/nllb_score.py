#!/usr/bin/env python3
"""Force-decode translations through NLLB: per-token surprisal (bits) under p(y|x), with char offsets.

Input TSV: id, src, text. Output JSONL: {"id", "offsets", "surp", "eos"}. The forced language token is not scored; the
closing </s> is stored separately as `eos`, so `surp` covers the same text as the LM surprisals.

--ids-dir D: score the token ids NLLB actually generated (D/ids.<cond>.jsonl, written by nllb_translate.py) instead of re-tokenising
the decoded text, which changes the tokens of about 9% of beam outputs and inflates their surprisal. --input then holds the
matching rows: columns id, src, cond, src_id, text. Rows carry the extra keys cond and src_id in the output.
"""
import argparse
import json
import math
import sys

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL = "facebook/nllb-200-distilled-600M"


def id_offsets(ids, text, tok):
    """Char span of every token in text, matched on non-space characters; None if the characters differ."""
    special = set(tok.all_special_ids)
    pieces = [("" if i in special else p.replace("\u2581", "").replace(" ", "")) for i, p in zip(ids, tok.convert_ids_to_tokens(ids))]
    pos = [k for k, c in enumerate(text) if not c.isspace()]
    if "".join(pieces) != "".join(text[k] for k in pos):
        return None
    out, n = [], 0
    for p in pieces:
        if p:
            out.append([pos[n], pos[n + len(p) - 1] + 1])
            n += len(p)
        else:
            a = pos[n] if n < len(pos) else len(text)
            out.append([a, a])
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--ids-dir", default=None)
    ap.add_argument("--bow", action="store_true", help="also store bow_ws / bow_wp, the log2 word-start probability at token positions 0..n (see bow.py)")
    ap.add_argument("--null-samples", type=int, default=0, help="K: also store `ent` (entropy per token) and `null` (tokens x K), the "
                    "surprisal of K tokens sampled given the real prefix; same token span as `surp`")
    args = ap.parse_args()
    torch.manual_seed(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    src_tok = AutoTokenizer.from_pretrained(MODEL, src_lang="eng_Latn")
    tgt_tok = AutoTokenizer.from_pretrained(MODEL, src_lang="ces_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device).eval()
    if args.bow:
        import bow as BOW
        ws_mask, wp_mask = BOW.masks(tgt_tok, model.config.vocab_size, extra_ids=(tgt_tok.eos_token_id,))
    df = pd.read_csv(args.input, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    if args.ids_dir:
        lang, eos = tgt_tok.convert_tokens_to_ids("ces_Latn"), tgt_tok.eos_token_id
        ids = {}
        for c in sorted(set(df["cond"])):
            for r in map(json.loads, open(f"{args.ids_dir}/ids.{c}.jsonl", encoding="utf-8")):
                ids[(c, r["id"])] = r["ids"]
        toks = [ids[k] for k in zip(df["cond"], df["src_id"])]
        offs = [id_offsets(x, t, tgt_tok) for x, t in zip(toks, df["text"])]
        keep = [i for i, o in enumerate(offs) if o is not None]
        print(f"{len(df) - len(keep)} of {len(df)} rows dropped: characters differ from text", file=sys.stderr)
        df = df.iloc[keep].reset_index(drop=True)
        enc_t = {"input_ids": [[lang] + toks[i] + [eos] for i in keep], "offset_mapping": [[[0, 0]] + offs[i] + [[0, 0]] for i in keep]}
    else:
        enc_t = tgt_tok(df["text"].tolist(), return_offsets_mapping=True, truncation=True, max_length=1000)
    order = sorted(range(len(df)), key=lambda i: len(enc_t["input_ids"][i]) + len(df["src"].iloc[i]) // 4)
    results = [None] * len(df)
    for b in range(0, len(order), args.batch_size):
        idx = order[b:b + args.batch_size]
        enc = src_tok(df["src"].iloc[idx].tolist(), return_tensors="pt", padding=True, truncation=True,
                      max_length=512).to(device)
        labels = [enc_t["input_ids"][i] for i in idx]  # lang, tokens..., </s>
        L = max(map(len, labels))
        dec = torch.full((len(idx), L), tgt_tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        for j, x in enumerate(labels):
            dec[j, 0] = model.config.decoder_start_token_id
            dec[j, 1:len(x)] = torch.tensor(x[:-1])
            mask[j, :len(x)] = 1
        logits = model(**enc, decoder_input_ids=dec.to(device), decoder_attention_mask=mask.to(device)).logits
        for j, i in enumerate(idx):
            x = labels[j]
            logp = torch.log_softmax(logits[j, :len(x)].float(), dim=-1)
            s = (-logp[torch.arange(len(x)), torch.tensor(x, device=device)] / math.log(2)).tolist()
            results[i] = {"id": df["id"].iloc[i], "offsets": [list(o) for o in enc_t["offset_mapping"][i][1:-1]],
                          "surp": [round(v, 4) for v in s[1:-1]], "eos": round(s[-1], 4)}
            if args.bow:
                # rows 1 .. len-1 predict text tokens 0 .. n-1 and then </s> (position n)
                results[i]["bow_ws"], results[i]["bow_wp"] = BOW.log2_masses(logp[1:], ws_mask, wp_mask)
            if args.null_samples:
                inner = logp[1:-1]
                results[i]["ent"] = [round(v, 4) for v in (-(inner.exp() * inner).sum(-1) / math.log(2)).tolist()]
                if len(inner):
                    drawn = torch.multinomial(inner.exp(), args.null_samples, replacement=True)
                    results[i]["null"] = (-inner.gather(-1, drawn) / math.log(2)).round(decimals=2).tolist()
            if args.ids_dir:
                results[i] |= {"cond": df["cond"].iloc[i], "src_id": df["src_id"].iloc[i]}
        print(f"{min(b + args.batch_size, len(order))}/{len(order)}", file=sys.stderr, flush=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
