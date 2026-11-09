#!/bin/bash
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuram40G|gpuram48G"
#SBATCH --mem=48G
#SBATCH -J uid-pmt
#SBATCH -o /lnet/work/people/jon/perplex/uid/logs/pmt-%j.out
set -euo pipefail
export HF_HOME=/lnet/work/people/jon/llm_agents/.cache/huggingface
source /lnet/work/people/jon/student_models/.venv/bin/activate
cd /lnet/work/people/jon/perplex
D=data/pm
python uid/surprisal.py --bow --extra --model BUT-FIT/Czech-GPT-2-XL-133k --input $D/test.cs.tsv --output $D/test.czgpt2xl.jsonl --dtype float32 --batch-size 16
python uid/surprisal.py --bow --extra --model utter-project/EuroLLM-9B-2512 --input $D/test.cs.tsv --output $D/test.eurollm9b.jsonl --dtype bfloat16 --batch-size 16
python uid/nllb_score.py --bow --input $D/test.pairs.tsv --output $D/test.nllb.jsonl --ids-dir data/nllb
python - <<'PY'
import json, sys
import numpy as np
sys.path.insert(0, "uid")
import bow as B
from transformers import AutoTokenizer
for name in ["BUT-FIT/Czech-GPT-2-XL-133k", "utter-project/EuroLLM-9B-2512", "facebook/nllb-200-distilled-600M"]:
    t = AutoTokenizer.from_pretrained(name)
    ws, wp = B.masks(t, len(t), extra_ids=(t.eos_token_id,))
    ids = t("Muž řekl, že přijde.", add_special_tokens=False)["input_ids"]
    print(name, "vocab", len(t), "ws", int(ws.sum()), "wp", int(wp.sum()), [(x, bool(ws[i]), bool(wp[i])) for x, i in zip(t.convert_ids_to_tokens(ids), ids)])
for f in ["czgpt2xl", "eurollm9b", "nllb"]:
    R = [json.loads(l) for l in open(f"data/pm/test.{f}.jsonl")]
    ws = np.concatenate([r["bow_ws"] for r in R]); wp = np.concatenate([r["bow_wp"] for r in R])
    assert all(len(r["bow_ws"]) == len(r["surp"]) + 1 for r in R)
    print(f, "rows", len(R), "P_ws median %.3f, share < 0.5: %.3f; P_wp median %.3f; wp>=ws %s; end-of-text P_ws median %.3f" % (np.median(2 ** ws), np.mean(2 ** ws < .5), np.median(2 ** wp), bool((wp >= ws - 1e-3).all()), np.median([2 ** r["bow_ws"][-1] for r in R])))
PY
