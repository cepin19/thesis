#!/bin/bash
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuram40G|gpuram48G"
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH -J uid-qual
#SBATCH -o /lnet/work/people/jon/perplex/uid/logs/qual-%j.out
# usage: sbatch run_quality.sh <cond_in.tsv> <output.tsv> {kiwi,xcomet} [extra score_quality.py args]
set -euo pipefail
export HF_HOME=/lnet/work/people/jon/llm_agents/.cache/huggingface
export PATH=/lnet/work/people/jon/wmt25/env/bin:$PATH
cd /lnet/work/people/jon/perplex
python uid/score_quality.py "$@"
