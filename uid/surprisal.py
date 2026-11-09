#!/usr/bin/env python3
"""Token-level surprisal (bits) under a causal LM, with character offsets.

Replaces score_gpt2.py for new experiments:
  * model is a CLI argument (no hard-coded checkpoint),
  * log_softmax in fp32 instead of log2(softmax(.)),
  * BOS (or EOS if the model has no BOS) is prepended so the first token is scored,
  * stores token char offsets, so word-level aggregation / punctuation handling is done
    downstream (uid/measures.py) without re-running the LM.

Input: TSV with columns `id` and `text` (header). Output: JSONL, one line per input row:
  {"id": ..., "offsets": [[start, end], ...], "surp": [bits, ...]}
With a `context` column in the input, the context (" "-joined preceding segments) precedes the text and only
the text's tokens are scored; offsets are relative to the text. The last --context-tokens context tokens are kept,
fewer if text + context would exceed --max-tokens.
With --extra also "ent" (entropy of the predictive distribution, bits) and "rank" (1 = the LM's argmax).
"""
import argparse
import json
import math
import sys

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True, help="TSV with columns id, text")
    p.add_argument("--output", required=True, help="JSONL output")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-batch-tokens", type=int, default=None, help="also limit batch size x longest row (large-vocabulary models)")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--context-tokens", type=int, default=256)
    p.add_argument("--extra", action="store_true", help="also store per-token entropy and rank")
    p.add_argument("--unescape-nl", action="store_true", help=r"turn literal \n in text into newlines (verse); offsets refer to the unescaped text")
    p.add_argument("--max-batch-positions", type=int, default=0, help="limit batches to this many padded token positions "
                   "(batch size x longest text), so long texts get smaller batches; 0 = only --batch-size applies")
    p.add_argument("--bow", action="store_true", help="also store `bow_ws` and `bow_wp`: log2 probability that the token at each position starts a "
                   "word (positions 0..n, the last one after the final token); for the Pimentel & Meister correction, see bow.py")
    p.add_argument("--context-chars", type=int, default=4000, help="keep at most this many trailing context characters")
    p.add_argument("--null-samples", type=int, default=0, help="K: store `null` (tokens x K), the surprisal of K tokens sampled "
                   "from the model's distribution given the real prefix; implies --extra")
    args = p.parse_args()
    args.extra = args.extra or args.null_samples > 0
    return args


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(1)
    try:
        tok = AutoTokenizer.from_pretrained(args.model)
    except ValueError:  # e.g. BLOOM: tokenizer.json exists but its tokenizer class is no longer resolved
        tok = PreTrainedTokenizerFast.from_pretrained(args.model)
    assert tok.is_fast, "fast tokenizer needed for offsets"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to(device)
    model.eval()
    limit = getattr(model.config, "max_position_embeddings", None)
    if limit:
        args.max_tokens = min(args.max_tokens, limit)
    if args.bow:
        import bow as BOW
        ws_mask, wp_mask = BOW.masks(tok, model.config.vocab_size, extra_ids=(tok.eos_token_id,))
    start_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else start_id

    df = pd.read_csv(args.input, sep="\t", dtype={"id": str}, keep_default_na=False, quoting=3)
    texts = df["text"].tolist()
    if args.unescape_nl:
        texts = [t.replace("\\n", "\n") for t in texts]
    ctxs = df["context"].tolist() if "context" in df else [""] * len(texts)
    if args.unescape_nl:
        ctxs = [c.replace("\\n", "\n") for c in ctxs]
    # (ctx ids, text ids, text offsets) per row
    enc_rows = []
    n_trunc = n_cut = 0
    for c, t in zip(ctxs, texts):
        tail = c[-args.context_chars:]  # enough characters for --context-tokens; cut at a word boundary
        cut = len(c) > args.context_chars
        if cut and " " in tail:
            tail = tail[tail.index(" ") + 1:]
        joined = tail + " " + t if tail else t
        e = tok(joined, add_special_tokens=False, return_offsets_mapping=True)
        shift = len(tail) + 1 if tail else 0
        k = next((j for j, o in enumerate(e["offset_mapping"]) if o[0] >= len(tail) and (not tail or o[1] > len(tail))),
                 len(e["input_ids"]))
        seg_ids = e["input_ids"][k:][:args.max_tokens - 1]
        offs = [[max(a - shift, 0), b - shift] for a, b in e["offset_mapping"][k:]][:args.max_tokens - 1]
        if len(seg_ids) < len(e["input_ids"]) - k:
            n_trunc += 1
        keep = min(args.context_tokens, args.max_tokens - 1 - len(seg_ids))
        ctx_ids = e["input_ids"][:k][-keep:] if keep > 0 else []
        cut = cut or len(ctx_ids) < k
        n_cut += bool(c) and cut
        enc_rows.append((ctx_ids, seg_ids, offs, cut))
    # sort by length for efficient batching, write in original order
    order = sorted(range(len(texts)), key=lambda i: len(enc_rows[i][0]) + len(enc_rows[i][1]))
    results = [None] * len(texts)
    size = lambda i: len(enc_rows[i][0]) + len(enc_rows[i][1]) + 1
    batches = []
    for i in order:  # ascending length, so the newest text sets the batch's padded length
        if batches and len(batches[-1]) < args.batch_size and (
                not args.max_batch_positions or (len(batches[-1]) + 1) * size(i) <= args.max_batch_positions):
            batches[-1].append(i)
        else:
            batches.append([i])
    done = 0
    for idx in batches:
        ids = [[start_id] + enc_rows[i][0] + enc_rows[i][1] for i in idx]
        L = max(len(x) for x in ids)
        input_ids = torch.full((len(ids), L), pad_id, dtype=torch.long)
        attn = torch.zeros((len(ids), L), dtype=torch.long)
        for j, x in enumerate(ids):
            input_ids[j, :len(x)] = torch.tensor(x)
            attn[j, :len(x)] = 1
        logits = model(input_ids=input_ids.to(device), attention_mask=attn.to(device)).logits
        for j, i in enumerate(idx):
            ctx_ids, seg_ids, offs, cut = enc_rows[i]
            k, n = len(ctx_ids) + 1, len(seg_ids)
            logp = torch.log_softmax(logits[j, k - 1:k + n - 1].float(), dim=-1)
            tgt = input_ids[j, k:k + n].to(device).unsqueeze(-1)
            lp_tgt = logp.gather(-1, tgt)
            r = {"id": df["id"].iloc[i], "offsets": offs,
                 "surp": [round(v, 4) for v in (-lp_tgt.squeeze(-1) / math.log(2)).tolist()]}
            if args.bow:
                # rows k-1 .. k+n-1 predict the tokens at positions 0 .. n of the scored text (n = after the last token)
                full = torch.log_softmax(logits[j, k - 1:k + n].float(), dim=-1)
                r["bow_ws"], r["bow_wp"] = BOW.log2_masses(full, ws_mask, wp_mask)
            if ctxs[i] and cut:
                r["ctx_cut"] = True
            if args.extra:
                r["ent"] = [round(v, 4) for v in (-(logp.exp() * logp).sum(-1) / math.log(2)).tolist()]
                r["rank"] = ((logp > lp_tgt).sum(-1) + 1).tolist()
            if args.null_samples and n:
                drawn = torch.multinomial(logp.exp(), args.null_samples, replacement=True)
                r["null"] = (-logp.gather(-1, drawn) / math.log(2)).round(decimals=2).tolist()
            results[i] = r
        done += len(idx)
        print(f"{done}/{len(order)}", file=sys.stderr, flush=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"done; {n_trunc} texts truncated at {args.max_tokens} tokens, {n_cut} contexts cut", file=sys.stderr)


if __name__ == "__main__":
    main()
