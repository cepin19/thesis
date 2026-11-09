#!/bin/bash
#SBATCH -p gpu-troja,gpu-ms
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpuram40G|gpuram48G"
#SBATCH --mem=48G
#SBATCH -J uid-litm
#SBATCH -o /lnet/work/people/jon/perplex/uid/logs/litm-%A_%a.out
#SBATCH --array=0-5
# Mendeley EN->FR. 0-2: MT (line by line). 3-5: surprisal of all sources and target texts (needs 0-2 and lit_collect.py).
set -euo pipefail
export HF_HOME=/lnet/work/people/jon/llm_agents/.cache/huggingface
source /lnet/work/people/jon/student_models/.venv/bin/activate
cd /lnet/work/people/jon/perplex
D=data/literary/mendeley
NLLB=facebook/nllb-200-distilled-600M
case $SLURM_ARRAY_TASK_ID in
0) python uid/lit_translate.py --model $NLLB --src-lang eng_Latn --tgt-lang fra_Latn --beams 1 --lower --input $D/sources.tsv --output $D/trans.nllb_beam1.tsv ;;
1) python uid/lit_translate.py --model $NLLB --src-lang eng_Latn --tgt-lang fra_Latn --beams 5 --lower --input $D/sources.tsv --output $D/trans.nllb_beam5.tsv ;;
2) python uid/lit_translate.py --model Helsinki-NLP/opus-mt-en-fr --beams 5 --lower --input $D/sources.tsv --output $D/trans.opus_beam5.tsv ;;
3) python uid/surprisal.py --unescape-nl --max-tokens 2048 --context-tokens 0 --model utter-project/EuroLLM-9B-2512 --input $D/texts.tsv --output $D/surp.eurollm9b.jsonl --batch-size 8 --dtype bfloat16 ;;
4) python uid/surprisal.py --unescape-nl --max-tokens 2048 --context-tokens 0 --model bigscience/bloom-1b7 --input $D/texts.tsv --output $D/surp.bloom1b7.jsonl --batch-size 8 --dtype float32 ;;
5) python uid/surprisal.py --unescape-nl --max-tokens 1024 --context-tokens 0 --model asi/gpt-fr-cased-small --input $D/texts_fr.tsv --output $D/surp.gptfr.jsonl --batch-size 8 --dtype float32 ;;
esac
