#!/bin/bash
#SBATCH -p cpu-troja,cpu-ms
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH -J uid-lmca
#SBATCH -o /lnet/work/people/jon/perplex/uid/logs/lmca-%j.out
set -euo pipefail
source /lnet/work/people/jon/student_models/.venv/bin/activate
cd /lnet/work/people/jon/perplex/uid
python analyze_nllb_lmcond.py
