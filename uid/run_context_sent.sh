#!/bin/bash
#SBATCH -p cpu-troja,cpu-ms
#SBATCH --mem=32G
#SBATCH -J uid-cs
#SBATCH -o /lnet/work/people/jon/perplex/uid/logs/cs-%A_%a.out
#SBATCH --array=0-11
# Segment/sentence x context/no context grid for both analyses (no context + segment level already exists).
set -euo pipefail
source /lnet/work/people/jon/student_models/.venv/bin/activate
cd /lnet/work/people/jon/perplex/uid
R=../results
RUNS=(
  "analyze_wmt_human.py --sent-split --out $R/wmt_human_sent"
  "analyze_wmt_human.py --context --out $R/wmt_human_ctx"
  "analyze_wmt_human.py --context --sent-split --out $R/wmt_human_ctx_sent"
  "analyze_wmt_human.py --sent-split --min-da-raw 90 --out $R/wmt_human_sent_da90"
  "analyze_wmt_human.py --context --min-da-raw 90 --out $R/wmt_human_ctx_da90"
  "analyze_wmt_human.py --context --sent-split --min-da-raw 90 --out $R/wmt_human_ctx_sent_da90"
  "analyze_diversity.py --sent-split"
  "analyze_diversity.py --context"
  "analyze_diversity.py --sent-split --context"
  "analyze_diversity.py --sent-split --min-da-raw 90"
  "analyze_diversity.py --context --min-da-raw 90"
  "analyze_diversity.py --sent-split --context --min-da-raw 90"
)
python ${RUNS[$SLURM_ARRAY_TASK_ID]}
