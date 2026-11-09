#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Score MT output with xCOMET-XL and optionally include error spans. "
            "Writes one JSONL row per sentence pair."
        )
    )
    p.add_argument("--src", required=True, type=Path, help="Source file (one sentence per line).")
    p.add_argument("--mt", required=True, type=Path, help="MT hypothesis file (one sentence per line).")
    p.add_argument("--ref", type=Path, default=None, help="Optional reference file (one sentence per line).")
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSONL path. Parent directory is created automatically.",
    )
    p.add_argument(
        "--model",
        default="Unbabel/XCOMET-XL",
        help="Model name for COMET download_model (default: Unbabel/XCOMET-XL).",
    )
    p.add_argument("--batch-size", type=int, default=8, help="Batch size for model.predict.")
    p.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs passed to model.predict (set 0 to force CPU).",
    )
    p.add_argument(
        "--keep-empty-lines",
        action="store_true",
        help="Keep empty lines instead of dropping them.",
    )
    return p


def read_lines(path: Path, keep_empty_lines: bool) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.rstrip("\n")
            if s or keep_empty_lines:
                lines.append(s)
    return lines


def extract_error_spans_from_metadata(metadata: Any, idx: int) -> list[dict[str, Any]]:
    if metadata is None:
        return []

    if hasattr(metadata, "error_spans"):
        maybe = getattr(metadata, "error_spans")
        if isinstance(maybe, list) and idx < len(maybe):
            spans = maybe[idx]
            return spans if isinstance(spans, list) else []

    if isinstance(metadata, dict):
        maybe = metadata.get("error_spans")
        if isinstance(maybe, list) and idx < len(maybe):
            spans = maybe[idx]
            return spans if isinstance(spans, list) else []

    return []


def main() -> None:
    args = build_arg_parser().parse_args()

    src = read_lines(args.src, args.keep_empty_lines)
    mt = read_lines(args.mt, args.keep_empty_lines)
    ref = read_lines(args.ref, args.keep_empty_lines) if args.ref else None

    n = len(src)
    if len(mt) != n:
        raise ValueError(f"Mismatched line counts: src={len(src)} mt={len(mt)}")
    if ref is not None and len(ref) != n:
        raise ValueError(f"Mismatched line counts: src={len(src)} ref={len(ref)}")

    data: list[dict[str, str]] = []
    for i in range(n):
        row = {"src": src[i], "mt": mt[i]}
        if ref is not None:
            row["ref"] = ref[i]
        data.append(row)

    from comet import download_model, load_from_checkpoint

    model_path = download_model(args.model)
    model = load_from_checkpoint(model_path)
    model_output = model.predict(
        data,
        batch_size=args.batch_size,
        gpus=args.gpus,
        progress_bar=True,
    )

    seg_scores = list(model_output.scores)
    metadata = getattr(model_output, "metadata", None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for i, (src_sent, mt_sent, score) in enumerate(zip(src, mt, seg_scores)):
            error_spans = extract_error_spans_from_metadata(metadata, i)
            rec = {
                "pair_id": i,
                "src": src_sent,
                "mt": mt_sent,
                "xcomet_xl_score": float(score),
                "has_error_spans": bool(error_spans),
                "error_spans": error_spans,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(seg_scores)} rows to {args.output}")
    system_score = getattr(model_output, "system_score", None)
    if system_score is not None:
        print(f"System score: {float(system_score):.6f}")


if __name__ == "__main__":
    main()
