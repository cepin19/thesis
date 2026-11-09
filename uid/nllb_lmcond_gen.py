#!/usr/bin/env python3
"""Sanity check: can each LM translate under each prompt? Greedy continuation for 8 fixed sources; prints to stdout."""
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import nllb_lmcond_prepare as Q
import udpipe_io as U

import sys
MODELS = sys.argv[1:] or ["utter-project/EuroLLM-9B-2512", "BUT-FIT/Czech-GPT-2-XL-133k", "/lnet/work/people/jon/newLM/gpt2_merged_out/checkpoint-1483000"]
en = U.read_texts("en")
en = en[en["text"].str.len().between(40, 120)].sample(8, random_state=1)["text"].tolist()
for name in MODELS:
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if "EuroLLM" in name else torch.float32).cuda().eval()
    start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    for p in ("en", "cs", "fewshot"):
        for s in en:
            ctx = Q.PROMPTS[p].format(s).replace("\\n", "\n")
            ids = torch.tensor([[start] + tok(ctx, add_special_tokens=False).input_ids]).cuda()
            with torch.no_grad():
                g = m.generate(ids, do_sample=False, max_new_tokens=50, pad_token_id=tok.eos_token_id)
            print(name.split("/")[-1], p, "|", s, "=>", tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).split("\n")[0], flush=True)
    del m
