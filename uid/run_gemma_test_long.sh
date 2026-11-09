#!/bin/bash
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuram40G|gpuram48G"
#SBATCH --mem=48G
#SBATCH -J uid-g4t
#SBATCH -o /lnet/work/people/jon/perplex/uid/logs/g4t-%j.out
# Check on the longest texts: position-limited batches must match unbatched scores.
set -euo pipefail
export HF_HOME=/lnet/work/people/jon/llm_agents/.cache/huggingface
source /lnet/work/people/jon/perplex/uid/.venv-gemma4/bin/activate
cd /lnet/work/people/jon/perplex
D=data/nllb/lmcond; A="--model google/gemma-4-E2B --input $D/test.long.tsv --dtype float32 --context-tokens 400 --max-tokens 1024 --unescape-nl"
python uid/surprisal.py $A --output $D/test.long.lim.jsonl --batch-size 32 --max-batch-positions 4000
python uid/surprisal.py $A --output $D/test.long.bs1.jsonl --batch-size 1
python - <<'PY'
import json
import numpy as np
L=lambda n:[json.loads(l) for l in open(f"data/nllb/lmcond/test.long.{n}.jsonl")]
a,b=L("lim"),L("bs1")
print("rows",len(a),len(b),"max diff",max(abs(np.array(p["surp"])-np.array(q["surp"])).max() for p,q in zip(a,b)),"max len",max(len(p["surp"]) for p in a))
PY
torch_peak=1
