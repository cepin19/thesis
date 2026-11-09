#!/bin/bash
#SBATCH -p cpu-troja,cpu-ms
#SBATCH --mem=16G
#SBATCH -c 2
#SBATCH -J uid-udb
#SBATCH -o /lnet/work/people/jon/perplex/uid/logs/udb-%j.out
# UDPipe 1.4 (UD 2.5 English EWT) on the English Books corpus, one line = one sentence, for sanity check S1.
set -euo pipefail
cd /lnet/work/people/jon/perplex
U=/net/projects/udpipe/bin/udpipe-latest-bin/bin-linux64/udpipe
M=/net/projects/udpipe/models/udpipe-ud-2.5-191206/english-ewt-ud-2.5-191206.udpipe
awk 'NF >= 6' testsets/Books.en-fr.en > data/sanity/books.en.txt
$U --tokenize --tokenizer=presegmented --tag --parse $M data/sanity/books.en.txt > data/sanity/books.en.conllu.tmp
mv data/sanity/books.en.conllu.tmp data/sanity/books.en.conllu
