#!/bin/bash
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -J uid-align
#SBATCH -o /lnet/work/people/jon/perplex/uid/logs/align-%j.out
set -euo pipefail
export HF_HOME=/lnet/work/people/jon/llm_agents/.cache/huggingface
source /lnet/work/people/jon/student_models/.venv/bin/activate
cd /lnet/work/people/jon/perplex
python uid/align.py
