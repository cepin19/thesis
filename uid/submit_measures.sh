#!/bin/bash
# Submit all rescoring for analyze_measures.py: entropy + null samples, unconditional and with the source prompt (<lm>_cond).
# Skips outputs that already exist. Prints the job ids.
cd /lnet/work/people/jon/perplex
R=uid/run_surp_null.sh
sub() { [ -s "$3" ] || sbatch --parsable $R "$@"; }
W=data/wmt_human; N=data/nllb; L=data/literary
for m in czgpt2xl eurollm9b customgpt2; do
  sub $m $W/texts.cs.tsv $W/surpn.cs.$m.jsonl 32
  sub $m $W/cond_in.tsv $W/surpn_cond.$m.jsonl 32 cond
  sub $m $N/texts.cs.tsv $N/surpn.$m.jsonl 32
  sub $m $N/cond_in.tsv $N/surpn_cond.$m.jsonl 32 cond
done
sub nllb $N/pairs.tsv $N/cond_null.jsonl 32
sub nllb $N/pairs_ids.tsv $N/cond_ids_null.jsonl 32 ids
for s in mendeley mendeley_flat; do
  for m in eurollm9b bloom1b7 gptfr; do
    sub $m $L/$s/texts_tgt.tsv $L/$s/surpn.$m.jsonl 8 nl
    sub $m $L/$s/cond_in.tsv $L/$s/surpn_cond.$m.jsonl 8 cond
  done
done
for s in chakrabarty par3; do
  for m in eurollm9b gpt2xl; do
    sub $m $L/$s/texts_tgt.tsv $L/$s/surpn.$m.jsonl 16 nl
    sub $m $L/$s/cond_in.tsv $L/$s/surpn_cond.$m.jsonl 16 cond
  done
done
for m in eurollm9b czgpt2xl; do
  sub $m $L/ccv/texts_tgt.tsv $L/ccv/surpn.$m.jsonl 16 nl
done
