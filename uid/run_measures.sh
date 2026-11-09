#!/bin/bash
#SBATCH -p cpu-troja,cpu-ms
#SBATCH --mem=96G
#SBATCH -c 2
#SBATCH -J uid-meas
#SBATCH -o /lnet/work/people/jon/perplex/uid/logs/meas-%x-%j.out
# usage: sbatch run_measures.sh <dataset> [--quality]
set -euo pipefail
source /lnet/work/people/jon/student_models/.venv/bin/activate
cd /lnet/work/people/jon/perplex/uid
python analyze_measures.py "$@"
