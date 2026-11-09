#!/bin/bash
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuram40G|gpuram48G"
#SBATCH --mem=48G
#SBATCH -J uid-nllbs
#SBATCH -o /lnet/work/people/jon/perplex/uid/logs/nllbs-%A_%a.out
#SBATCH --array=0-3
# Force-decode through NLLB (0) and score new NLLB outputs under the external LMs (1-3).
set -euo pipefail
export HF_HOME=/lnet/work/people/jon/llm_agents/.cache/huggingface
source /lnet/work/people/jon/student_models/.venv/bin/activate
cd /lnet/work/people/jon/perplex
D=data/nllb
case $SLURM_ARRAY_TASK_ID in
0) python uid/nllb_score.py --input $D/pairs.tsv --output $D/cond.jsonl ;;
1) python uid/surprisal.py --model utter-project/EuroLLM-9B-2512 --input $D/texts.cs.tsv --output $D/surp.eurollm9b.jsonl --batch-size 32 --dtype bfloat16 ;;
2) python uid/surprisal.py --model BUT-FIT/Czech-GPT-2-XL-133k --input $D/texts.cs.tsv --output $D/surp.czgpt2xl.jsonl --batch-size 32 --dtype float32 ;;
3) python uid/surprisal.py --model /lnet/work/people/jon/newLM/gpt2_merged_out/checkpoint-1483000 --input $D/texts.cs.tsv --output $D/surp.customgpt2.jsonl --batch-size 32 --dtype float32 ;;
esac
