#!/bin/bash
#SBATCH -p cpu-troja,cpu-ms
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -J uid-ldi
#SBATCH -o /lnet/work/people/jon/perplex/uid/logs/ldi-%j.out
set -euo pipefail
cd /lnet/work/people/jon/perplex/uid
../replication/temporal_flattening/.venv/bin/python analyze_less_diverse.py ident
