#!/usr/bin/env python3
"""Word alignment source -> translation for every (src, hyp) pair in data/wmt_human/wmt{20,21,22}.en-cs.tsv.

SimAlign "ArgMax" (Jalili Sabet et al. 2020): mBERT (cased) layer-8 embeddings, word vector = mean of its
subwords, cosine similarity matrix, keep (i, j) iff j is the best target for i and i the best source for j.
Words are UDPipe syntactic words (uid/udpipe_io.py), so alignments index the parsed tokens.

Output: data/wmt_human/align.jsonl, one line per unique pair {"src": en-id, "hyp": cs-id, "a": [[i, j], ...]}
(0-based word indices). With --ud-dir udpipe_sent: align_sent.jsonl over the sentence-split parse.
"""
import argparse
import json
import sys

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

import measures as M
import udpipe_io as U

MODEL = "google-bert/bert-base-multilingual-cased"
LAYER = 8


def text2id(lang):
    df = U.read_texts(lang)
    return dict(zip(df["text"], df["id"]))


@torch.no_grad()
def embed(words_list, tok, model, device):
    enc = tok(words_list, is_split_into_words=True, truncation=True, max_length=512, padding=True,
              return_tensors="pt")
    h = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True).hidden_states[LAYER]
    out = []
    for b, words in enumerate(words_list):
        wid = enc.word_ids(b)
        idx = torch.tensor([w if w is not None else -1 for w in wid], device=device)
        n = len(words)
        sums = torch.zeros(n, h.shape[-1], device=device)
        cnt = torch.zeros(n, device=device)
        m = idx >= 0
        sums.index_add_(0, idx[m], h[b][m])
        cnt.index_add_(0, idx[m], torch.ones(int(m.sum()), device=device))
        v = sums / cnt.clamp(min=1).unsqueeze(1)
        out.append(torch.nn.functional.normalize(v, dim=-1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ud-dir", default="udpipe")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()
    rows = pd.concat([pd.read_csv(U.DATA / f"wmt{y}.en-cs.tsv", sep="\t", keep_default_na=False)
                      for y in (20, 21, 22)])
    en_id, cs_id = text2id("en"), text2id("cs")
    pairs = sorted({(en_id[M.norm_text(s)], cs_id[M.norm_text(h)]) for s, h in zip(rows["src"], rows["hyp"])})
    en, cs = U.load("en", args.ud_dir), U.load("cs", args.ud_dir)
    print(f"{len(pairs)} pairs", file=sys.stderr)
    pairs.sort(key=lambda p: len(en[p[0]]) + len(cs[p[1]]))
    bs = 64
    with open(U.DATA / ("align.jsonl" if args.ud_dir == "udpipe" else "align_sent.jsonl"), "w") as f:
        for b in range(0, len(pairs), bs):
            chunk = pairs[b:b + bs]
            ve = embed([[t["form"] for t in en[s]] for s, _ in chunk], tok, model, device)
            vc = embed([[t["form"] for t in cs[h]] for _, h in chunk], tok, model, device)
            for (s, h), a, c in zip(chunk, ve, vc):
                sim = a @ c.T
                fwd, bwd = sim.argmax(1), sim.argmax(0)
                al = [[i, int(j)] for i, j in enumerate(fwd.tolist()) if bwd[j] == i]
                f.write(json.dumps({"src": s, "hyp": h, "a": al}) + "\n")
            if b % (bs * 100) == 0:
                print(f"{b}/{len(pairs)}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
