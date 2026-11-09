#!/bin/bash
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuram40G|gpuram48G"
#SBATCH --mem=48G
#SBATCH -J uid-lmc
#SBATCH -o /lnet/work/people/jon/perplex/uid/logs/lmc-%A_%a.out
#SBATCH --array=0-11%4
# Source-conditioned LM surprisal of the NLLB pairs: 3 LMs x 4 prompts.
set -euo pipefail
export HF_HOME=/lnet/work/people/jon/llm_agents/.cache/huggingface
source /lnet/work/people/jon/student_models/.venv/bin/activate
cd /lnet/work/people/jon/perplex
MODELS=("utter-project/EuroLLM-9B-2512 eurollm9b bfloat16 16" "BUT-FIT/Czech-GPT-2-XL-133k czgpt2xl float32 32" "/lnet/work/people/jon/newLM/gpt2_merged_out/checkpoint-1483000 customgpt2 float32 32")
PROMPTS=(en cs fewshot en_shuf)
read -r MODEL TAG DTYPE BS <<< "${MODELS[$((SLURM_ARRAY_TASK_ID / 4))]}"
P=${PROMPTS[$((SLURM_ARRAY_TASK_ID % 4))]}
D=data/nllb/lmcond
python uid/surprisal.py --model "$MODEL" --input $D/pairs.$P.tsv --output $D/surp.$TAG.$P.jsonl --batch-size $BS --dtype $DTYPE --unescape-nl --context-tokens 400 --max-tokens 1024
