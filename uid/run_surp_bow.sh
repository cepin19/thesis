#!/bin/bash
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuram40G|gpuram48G"
#SBATCH --mem=64G
#SBATCH -J uid-bow
#SBATCH -o /lnet/work/people/jon/perplex/uid/logs/bow-%j.out
# Rescore with word-start probabilities (Pimentel & Meister correction, see bow.py) plus entropy and rank. usage: sbatch run_surp_bow.sh <lm> <input tsv> <output jsonl> [batch] [nl]
# 5th argument: nl = text holds literal \n line breaks; cond = input has a context column with the source prompt (lit_cond_inputs.py);
# for lm=nllb (force-decoding; input needs src), ids = score generated ids from data/nllb.
set -euo pipefail
export HF_HOME=/lnet/work/people/jon/llm_agents/.cache/huggingface
source /lnet/work/people/jon/student_models/.venv/bin/activate
cd /lnet/work/people/jon/perplex
BS=${4:-16}
X=""; MAXT=1024
[ "${5:-}" = nl ] && X="--unescape-nl --context-tokens 0"
[ "${5:-}" = cond ] && X="--unescape-nl --context-tokens 2000 --context-chars 30000"
[ "${5:-}" = ctx ] && X="--context-tokens 256"
case $1 in
nllb) python uid/nllb_score.py --input "$2" --output "$3" --batch-size "$BS" --bow ${5:+--ids-dir data/nllb}; exit 0 ;;
eurollm9b) M=utter-project/EuroLLM-9B-2512; T=bfloat16; MAXT=2048 ;;
bloom1b7) M=bigscience/bloom-1b7; T=float32; MAXT=2048 ;;
gptfr) M=asi/gpt-fr-cased-small; T=float32 ;;
gpt2xl) M=gpt2-xl; T=float32 ;;
czgpt2xl) M=BUT-FIT/Czech-GPT-2-XL-133k; T=float32 ;;
customgpt2) M=/lnet/work/people/jon/newLM/gpt2_merged_out/checkpoint-1483000; T=float32 ;;
*) echo "unknown LM $1"; exit 1 ;;
esac
[ -z "${5:-}" ] && MAXT=512
[ "${5:-}" = ctx ] && MAXT=512
python uid/surprisal.py $X --max-tokens $MAXT --model $M --input "$2" --output "$3" --batch-size "$BS" --dtype $T --bow --extra
