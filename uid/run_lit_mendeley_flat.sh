#!/bin/bash
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuram40G|gpuram48G"
#SBATCH --mem=48G
#SBATCH -J uid-litf
#SBATCH -o /lnet/work/people/jon/perplex/uid/logs/litf-%j.out
# Control for line-by-line MT: EN->FR poems translated sentence by sentence, all texts flattened (no line breaks).
set -euo pipefail
export HF_HOME=/lnet/work/people/jon/llm_agents/.cache/huggingface
source /lnet/work/people/jon/student_models/.venv/bin/activate
cd /lnet/work/people/jon/perplex
S=data/literary/mendeley; D=data/literary/mendeley_flat; mkdir -p $D
python - <<'E'
import re
import pandas as pd
S, D = "data/literary/mendeley/", "data/literary/mendeley_flat/"
rd = lambda p: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3)
flat = lambda t: " ".join(l for l in t.split("\\n") if l.strip())
src, hum = rd(S + "sources.tsv"), rd(S + "human.tsv")
out = src.copy(); out["text"] = out["text"].map(flat)
out.to_csv(D + "sources.tsv", sep="\t", index=False, quoting=3)
hum["text"] = hum["text"].map(flat); hum.to_csv(D + "human.tsv", sep="\t", index=False, quoting=3)
spl = out.copy(); spl["text"] = spl["text"].map(lambda t: "\\n".join(re.split(r"(?<=[.!?;:])\s+", t)))
spl.to_csv(D + "split.tsv", sep="\t", index=False, quoting=3)
E
N=facebook/nllb-200-distilled-600M
python uid/lit_translate.py --model $N --src-lang eng_Latn --tgt-lang fra_Latn --beams 5 --lower --input $D/split.tsv --output $D/trans.nllb_sent5.tsv
python uid/lit_translate.py --model Helsinki-NLP/opus-mt-en-fr --beams 5 --lower --input $D/split.tsv --output $D/trans.opus_sent5.tsv
sed -i 's/\\n/ /g' $D/trans.nllb_sent5.tsv $D/trans.opus_sent5.tsv
python uid/lit_collect.py mendeley_flat
