#!/bin/bash
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuram40G|gpuram48G"
#SBATCH --mem=48G
#SBATCH -J uid-lits
#SBATCH -o /lnet/work/people/jon/perplex/uid/logs/lits-%j.out
# usage: sbatch run_lit_score.sh <set> <lm>. Multilingual LMs also score the sources (texts.tsv); others only targets.
set -euo pipefail
export HF_HOME=/lnet/work/people/jon/llm_agents/.cache/huggingface
source /lnet/work/people/jon/student_models/.venv/bin/activate
cd /lnet/work/people/jon/perplex
D=data/literary/$1
case $2 in
eurollm9b) M=utter-project/EuroLLM-9B-2512; T=bfloat16; I=texts.tsv; X=2048 ;;
bloom1b7) M=bigscience/bloom-1b7; T=float32; I=texts.tsv; X=2048 ;;
gptfr) M=asi/gpt-fr-cased-small; T=float32; I=texts_tgt.tsv; X=1024 ;;
gpt2xl) M=gpt2-xl; T=float32; I=texts_tgt.tsv; X=1024 ;;
czgpt2xl) M=BUT-FIT/Czech-GPT-2-XL-133k; T=float32; I=texts_tgt.tsv; X=1024 ;;
*) echo "unknown LM $2"; exit 1 ;;
esac
python uid/surprisal.py --unescape-nl --max-tokens $X --context-tokens 0 --model $M --input $D/$I --output $D/surp.$2.jsonl --batch-size ${3:-8} --dtype $T
