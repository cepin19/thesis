#!/bin/bash
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuram40G|gpuram48G"
#SBATCH --mem=48G
#SBATCH -J uid-g4
#SBATCH -o /lnet/work/people/jon/perplex/uid/logs/g4-%A_%a.out
#SBATCH --array=0-4%3
# Gemma 4 E2B (base variant): plain p(y), the 4 prompts; task 5 (optional, --array=5) is the translation check. Own environment: needs transformers >= 5.5.
set -euo pipefail
export HF_HOME=/lnet/work/people/jon/llm_agents/.cache/huggingface
source /lnet/work/people/jon/perplex/uid/.venv-gemma4/bin/activate
cd /lnet/work/people/jon/perplex
D=data/nllb/lmcond; M=google/gemma-4-E2B; T=gemma4e2b
C="--batch-size 32 --max-batch-positions 4000 --dtype float32 --context-tokens 400 --max-tokens 1024"
case $SLURM_ARRAY_TASK_ID in
0) python uid/surprisal.py --model $M --input $D/texts.tsv --output $D/surp.$T.plain.jsonl $C ;;
1) python uid/surprisal.py --model $M --input $D/pairs.en.tsv --output $D/surp.$T.en.jsonl $C --unescape-nl ;;
2) python uid/surprisal.py --model $M --input $D/pairs.cs.tsv --output $D/surp.$T.cs.jsonl $C --unescape-nl ;;
3) python uid/surprisal.py --model $M --input $D/pairs.fewshot.tsv --output $D/surp.$T.fewshot.jsonl $C --unescape-nl ;;
4) python uid/surprisal.py --model $M --input $D/pairs.en_shuf.tsv --output $D/surp.$T.en_shuf.jsonl $C --unescape-nl ;;
5) cd uid && python nllb_lmcond_gen.py $M ;;
esac
