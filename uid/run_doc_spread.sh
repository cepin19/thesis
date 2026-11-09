#!/bin/bash
#SBATCH -p cpu-troja,cpu-ms
#SBATCH --mem=64G
#SBATCH -c 2
#SBATCH -J uid-spread
#SBATCH -o /lnet/work/people/jon/perplex/uid/logs/spread-%j.out
set -euo pipefail
cd /lnet/work/people/jon/perplex/uid
../replication/temporal_flattening/.venv/bin/python analyze_doc_spread.py
