#!/bin/bash
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuram40G|gpuram48G"
#SBATCH --mem=32G
#SBATCH -J uid-nllb
#SBATCH -o /lnet/work/people/jon/perplex/uid/logs/nllb-%A_%a.out
#SBATCH --array=0-5
# NLLB decoding sweep on the WMT20-22 English sources.
set -euo pipefail
export HF_HOME=/lnet/work/people/jon/llm_agents/.cache/huggingface
source /lnet/work/people/jon/student_models/.venv/bin/activate
cd /lnet/work/people/jon/perplex/uid
JOBS=("--beams 1 --batch-size 128" "--beams 2 --batch-size 64" "--beams 5 --batch-size 32" "--beams 10 --batch-size 16" "--beams 50 --batch-size 4" "--sample --batch-size 128")
python nllb_translate.py ${JOBS[$SLURM_ARRAY_TASK_ID]} ${NLLB_ARGS:-}
