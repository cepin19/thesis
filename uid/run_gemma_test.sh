#!/bin/bash
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuram40G|gpuram48G"
#SBATCH --mem=32G
#SBATCH -J uid-g4t
#SBATCH -o /lnet/work/people/jon/perplex/uid/logs/g4t-%j.out
set -euo pipefail
export HF_HOME=/lnet/work/people/jon/llm_agents/.cache/huggingface
source /lnet/work/people/jon/perplex/uid/.venv-gemma4/bin/activate
cd /lnet/work/people/jon/perplex
D=data/nllb/lmcond
for dt in bfloat16 float32; do python uid/surprisal.py --model google/gemma-4-E2B --input $D/test.fewshot.tsv --output $D/test.$dt.jsonl --batch-size 4 --dtype $dt --unescape-nl --context-tokens 400 --max-tokens 1024; done
for bs in 1 20; do python uid/surprisal.py --model google/gemma-4-E2B --input $D/test.fewshot.tsv --output $D/test.bs$bs.jsonl --batch-size $bs --dtype float32 --unescape-nl --context-tokens 400 --max-tokens 1024; done
python - <<'PY'
import json
import numpy as np
L=lambda n:[json.loads(l) for l in open(f"data/nllb/lmcond/test.{n}.jsonl")]
a,b,c,d=L("bfloat16"),L("float32"),L("bs1"),L("bs20")
f=lambda x,y:max(abs(np.array(p["surp"])-np.array(q["surp"])).max() for p,q in zip(x,y))
print("max diff bf16-fp32",f(a,b),"bs1-bs20",f(c,d)); print("mean surp",np.mean([np.mean(r["surp"]) for r in b])); print(b[0]["surp"][:15],b[0]["offsets"][:6])
PY
