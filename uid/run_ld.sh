#!/bin/bash
#SBATCH -p cpu-troja,cpu-ms
#SBATCH --mem=96G
#SBATCH -c 16
#SBATCH -J uid-ld
#SBATCH -o /lnet/work/people/jon/perplex/uid/logs/ld-%j.out
# usage: sbatch run_ld.sh {pairs,htra,shifts} | sbatch --gres=gpu:1 -p gpu-troja,gpu-ms --constraint="gpuram40G|gpuram48G" run_ld.sh embed
set -euo pipefail
export HF_HOME=/lnet/work/people/jon/llm_agents/.cache/huggingface
source /lnet/work/people/jon/student_models/.venv/bin/activate
cd /lnet/work/people/jon/perplex/uid
python analyze_less_diverse.py "$1"
