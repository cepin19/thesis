#!/bin/bash
#SBATCH -p cpu-troja,cpu-ms
#SBATCH --mem=24G
#SBATCH -J uid-div
#SBATCH -o /lnet/work/people/jon/perplex/uid/logs/div-%A_%a.out
#SBATCH --array=0-1
# uid/analyze_diversity.py on the full test sets (0) and on segments with raw DA >= 90 (1).
set -euo pipefail
source /lnet/work/people/jon/student_models/.venv/bin/activate
cd /lnet/work/people/jon/perplex/uid
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then python analyze_diversity.py; else python analyze_diversity.py --min-da-raw 90; fi
