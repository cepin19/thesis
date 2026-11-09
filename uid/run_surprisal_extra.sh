#!/bin/bash
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuram40G|gpuram48G"
#SBATCH --mem=48G
#SBATCH -J uid-surpx
#SBATCH -o /lnet/work/people/jon/perplex/uid/logs/surpx-%A_%a.out
#SBATCH --array=0-2
# As run_surprisal.sh, but also stores per-token LM entropy and rank (--extra) for the Czech texts.
set -euo pipefail
export HF_HOME=/lnet/work/people/jon/llm_agents/.cache/huggingface
source /lnet/work/people/jon/student_models/.venv/bin/activate
cd /lnet/work/people/jon/perplex

JOBS=(
  "cs utter-project/EuroLLM-9B-2512 eurollm9b bfloat16 16"
  "cs BUT-FIT/Czech-GPT-2-XL-133k czgpt2xl float32 32"
  "cs /lnet/work/people/jon/newLM/gpt2_merged_out/checkpoint-1483000 customgpt2 float32 32"
)
read -r LANG MODEL TAG DTYPE BS <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"
OUT=data/wmt_human/surpx.${LANG}.${TAG}.jsonl
echo "$(date) $LANG $MODEL -> $OUT"
python uid/surprisal.py --model "$MODEL" --input data/wmt_human/texts.${LANG}.tsv --output "$OUT" --batch-size $BS --dtype "$DTYPE" --extra
echo "$(date) done"
