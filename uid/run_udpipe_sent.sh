#!/bin/bash
#SBATCH -p cpu-troja,cpu-ms
#SBATCH --mem=4G
#SBATCH -J uid-udsent
#SBATCH -o /lnet/work/people/jon/perplex/uid/logs/udsent-%A_%a.out
#SBATCH --array=0-23
# UDPipe 1.4 (UD 2.5 models: cs PDT, en EWT) on data/wmt_human/texts.{cs,en}.tsv, UDPipe sentence splitter, segments separated by blank lines.
# Chunks 0-19: Czech, 20-23: English. Chunk files are made by: python uid/udpipe_io.py split-sent
set -euo pipefail
cd /lnet/work/people/jon/perplex
U=/net/projects/udpipe/bin/udpipe-latest-bin/bin-linux64/udpipe
M=/net/projects/udpipe/models/udpipe-ud-2.5-191206
i=$SLURM_ARRAY_TASK_ID
if [ $i -lt 20 ]; then L=cs; MODEL=$M/czech-pdt-ud-2.5-191206.udpipe; c=$i; else L=en; MODEL=$M/english-ewt-ud-2.5-191206.udpipe; c=$((i-20)); fi
D=data/wmt_human/udpipe_sent
$U --tokenize --tag --parse $MODEL $D/$L.$c.txt > $D/$L.$c.conllu.tmp
mv $D/$L.$c.conllu.tmp $D/$L.$c.conllu
