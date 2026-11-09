"""Word-start probabilities for the Pimentel & Meister (2024) correction of word surprisal.

With beginning-of-word (bow) marking tokenisers (GPT-2, Llama/EuroLLM, BLOOM, Gemma, NLLB sentencepiece), the probability of a word w
after context c is not just the product of its subword probabilities: it also requires that the next token starts a new word,
  p(w | c) = p(subwords of w | c) * P_bow(after w) / P_bow(before w),
where P_bow(position) is the model's total probability, at that position, of tokens that start a word, plus end-of-sequence
("How to Compute the Probability of a Word", EMNLP 2024, https://arxiv.org/abs/2406.14561).

Two word definitions, matching the two unit types in measures.py:
  ws : words are whitespace-delimited (unit `untok`): word-start tokens = tokens starting with whitespace, plus EOS.
  wp : punctuation also delimits words (unit `nopunct`, which then drops punctuation units): word-start tokens additionally include
       tokens starting with punctuation (the paper's "Bug Fix 2" case).
log2_masses(logp) returns, per row, log2 P(ws) and log2 P(wp).
"""
import math

import torch

# raw vocabulary strings that encode a leading whitespace in the tokenisers used here
WS_MARKS = ("Ġ", "▁", "Ċ", "ĉ", "č", " ", "\n", "\t", "\r", "<0x20>", "<0x0A>", "<0x09>", "<0x0D>")


def masks(tok, vocab_size, extra_ids=()):
    """Boolean tensors (vocab_size,) for the ws and wp word-start sets; extra_ids (EOS) are in both."""
    ws = torch.zeros(vocab_size, dtype=torch.bool)
    wp = torch.zeros(vocab_size, dtype=torch.bool)
    for i, raw in enumerate(tok.convert_ids_to_tokens(list(range(min(vocab_size, len(tok)))))):
        if raw is None:
            continue
        lead = raw.startswith(WS_MARKS)
        core = raw
        while core.startswith(WS_MARKS):
            core = core[next(len(m) for m in WS_MARKS if core.startswith(m)):]
        punct = bool(core) and not core.startswith("<0x") and not core[0].isalnum() and core[0] != "_"
        ws[i] = lead
        wp[i] = lead or punct
    for i in extra_ids:
        if i is not None and i < vocab_size:
            ws[i] = wp[i] = True
    return ws, wp


def log2_masses(logp, ws, wp):
    """logp: (positions, vocab) natural-log probabilities -> two lists of log2 word-start probabilities."""
    ws, wp = ws.to(logp.device), wp.to(logp.device)
    a = torch.logsumexp(logp.masked_fill(~ws, -float("inf")), dim=-1) / math.log(2)
    b = torch.logsumexp(logp.masked_fill(~wp, -float("inf")), dim=-1) / math.log(2)
    return [round(v, 4) for v in a.tolist()], [round(v, 4) for v in b.tolist()]
