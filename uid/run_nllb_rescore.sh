#!/bin/bash
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuram40G|gpuram48G"
#SBATCH --mem=32G
#SBATCH -J uid-nllbr
#SBATCH -o /lnet/work/people/jon/perplex/uid/logs/nllbr-%j.out
# Force-decode the generated token ids of every NLLB condition whose text matches trans.<cond>.tsv.
set -euo pipefail
export HF_HOME=/lnet/work/people/jon/llm_agents/.cache/huggingface
source /lnet/work/people/jon/student_models/.venv/bin/activate
cd /lnet/work/people/jon/perplex/uid
python - <<'PY'
import json
import pandas as pd
import nllb_prepare as P
import udpipe_io as U
en = U.read_texts("en"); id2src = dict(zip(en["id"], en["text"]))
rows = []
for c in P.CONDS:
    t = dict(zip(*P.read_trans(c)[["id", "text"]].values.T))
    n = bad = 0
    for r in map(json.loads, open(P.ND / f"ids.{c}.jsonl", encoding="utf-8")):
        n += 1
        if r["text"] != t[r["id"]]:
            bad += 1
            continue
        if r["text"]:
            rows.append((f"{c}:{r['id']}", id2src[r["id"]], c, r["id"], r["text"]))
    print(c, n, "rows,", bad, "differ from trans")
pd.DataFrame(rows, columns=["id", "src", "cond", "src_id", "text"]).to_csv(P.ND / "pairs_ids.tsv", sep="\t", index=False, quoting=3)
PY
python nllb_score.py --input ../data/nllb/pairs_ids.tsv --output ../data/nllb/cond_ids.jsonl --ids-dir ../data/nllb
