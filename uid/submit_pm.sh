#!/bin/bash
# Submit rescoring with word-start probabilities for analyze_measures.py --pm, unconditional and with the source prompt (<lm>_cond).
# Skips outputs that already exist. Prints the job ids.
cd /lnet/work/people/jon/perplex
R=uid/run_surp_bow.sh
sub() { [ -s "$3" ] || sbatch --parsable $R "$@"; }
W=data/wmt_human; N=data/nllb; L=data/literary
for m in czgpt2xl eurollm9b customgpt2; do
  sub $m $W/texts.cs.tsv $W/surpb.cs.$m.jsonl 32
  sub $m $W/cond_in.tsv $W/surpb_cond.$m.jsonl 32 cond
  sub $m $N/texts.cs.tsv $N/surpb.$m.jsonl 32
  sub $m $N/cond_in.tsv $N/surpb_cond.$m.jsonl 32 cond
done
sub nllb $N/pairs.tsv $N/cond_bow.jsonl 32
sub nllb $N/pairs_ids.tsv $N/cond_ids_bow.jsonl 32 ids
for s in mendeley mendeley_flat; do
  for m in eurollm9b bloom1b7 gptfr; do
    sub $m $L/$s/texts_tgt.tsv $L/$s/surpb.$m.jsonl 8 nl
    sub $m $L/$s/cond_in.tsv $L/$s/surpb_cond.$m.jsonl 8 cond
  done
done
for s in chakrabarty par3; do
  for m in eurollm9b gpt2xl; do
    sub $m $L/$s/texts_tgt.tsv $L/$s/surpb.$m.jsonl 16 nl
    sub $m $L/$s/cond_in.tsv $L/$s/surpb_cond.$m.jsonl 16 cond
  done
done
for m in eurollm9b czgpt2xl; do
  sub $m $L/ccv/texts_tgt.tsv $L/ccv/surpb.$m.jsonl 16 nl
done
for m in czgpt2xl eurollm9b customgpt2; do
  sub $m $W/texts_ctx.cs.tsv $W/surpb_ctx.cs.$m.jsonl 32 ctx
done
