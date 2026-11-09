#!/bin/bash
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuram40G|gpuram48G"
#SBATCH --mem=48G
#SBATCH -J uid-surp
#SBATCH -o /lnet/work/people/jon/perplex/uid/logs/surp-%A_%a.out
#SBATCH --array=0-3
# Score WMT20-22 EN->CS texts (data/wmt_human/texts.{cs,en}.tsv) with several LMs.
set -euo pipefail
export HF_HOME=/lnet/work/people/jon/llm_agents/.cache/huggingface
source /lnet/work/people/jon/student_models/.venv/bin/activate
cd /lnet/work/people/jon/perplex

JOBS=(
  "cs utter-project/EuroLLM-9B-2512 eurollm9b bfloat16"
  "cs BUT-FIT/Czech-GPT-2-XL-133k czgpt2xl float32"
  "cs /lnet/work/people/jon/newLM/gpt2_merged_out/checkpoint-1483000 customgpt2 float32"
  "en utter-project/EuroLLM-9B-2512 eurollm9b bfloat16"
)
read -r LANG MODEL TAG DTYPE <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"
OUT=data/wmt_human/surp.${LANG}.${TAG}.jsonl
echo "$(date) $LANG $MODEL -> $OUT"
python uid/surprisal.py --model "$MODEL" --input data/wmt_human/texts.${LANG}.tsv --output "$OUT" --batch-size 32 --dtype "$DTYPE"
echo "$(date) done"
