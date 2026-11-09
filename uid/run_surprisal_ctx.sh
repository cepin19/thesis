#!/bin/bash
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuram40G|gpuram48G"
#SBATCH --mem=64G
#SBATCH -J uid-surpc
#SBATCH -o /lnet/work/people/jon/perplex/uid/logs/surpc-%A_%a.out
#SBATCH --array=0-3
# Surprisal given the preceding segments of the document (data/wmt_human/texts_ctx.{cs,en}.tsv).
set -euo pipefail
export HF_HOME=/lnet/work/people/jon/llm_agents/.cache/huggingface
source /lnet/work/people/jon/student_models/.venv/bin/activate
cd /lnet/work/people/jon/perplex

JOBS=(
  "cs utter-project/EuroLLM-9B-2512 eurollm9b bfloat16 8"
  "cs BUT-FIT/Czech-GPT-2-XL-133k czgpt2xl float32 16"
  "cs /lnet/work/people/jon/newLM/gpt2_merged_out/checkpoint-1483000 customgpt2 float32 16"
  "en utter-project/EuroLLM-9B-2512 eurollm9b bfloat16 8"
)
read -r LANG MODEL TAG DTYPE BS <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"
OUT=data/wmt_human/surpctx.${LANG}.${TAG}.jsonl
echo "$(date) $LANG $MODEL -> $OUT"
python uid/surprisal.py --model "$MODEL" --input data/wmt_human/texts_ctx.${LANG}.tsv --output "$OUT" --batch-size $BS --dtype "$DTYPE" --extra
echo "$(date) done"
