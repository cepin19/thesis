#!/bin/bash
#SBATCH -p cpu-troja,cpu-ms
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -J uid-g4a
#SBATCH -o /lnet/work/people/jon/perplex/uid/logs/g4a-%j.out
set -euo pipefail
source /lnet/work/people/jon/student_models/.venv/bin/activate
cd /lnet/work/people/jon/perplex/uid
python analyze_nllb_lmcond.py --lms gemma4e2b --out ../results/nllb_lmcond_gemma
